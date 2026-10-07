"""Stage 2 -- template set selection (DESIGN.md S2e).

Scores reads against candidate alleles under a platform error model, then
picks the template set of size CN (from cn.py) maximising a k-component
mixture likelihood (EM-fitted weights). Candidates are every allele with
>=1 real read alignment, narrowed by mismatch rate (no FuFiHLA top-N cap).

Two read scorers, chosen by the caller:
- `context_log_likelihoods()`: ONT R10 default; Badread per-7-mer error
  model applied along each (read, allele) CIGAR.
- `read_log_likelihoods()`: flat per-base binomial on NM; for the HiFi
  milestone-1 check, with a caller-supplied `ErrorModel`.

The LRT statistic 2 * (best mixture logL - best single-allele logL) replaces
FuFiHLA's 4x-coverage homozygosity rule. It is reported raw, not as a
p-value: the allele space is discrete, so chi-square asymptotics don't hold.
"""
from __future__ import annotations

import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pysam
import yaml

from assign import DEFAULT_POOL_FASTA, GeneAssignmentMatrix, realign_to_candidates

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_ONT_R10_MODEL = REPO_ROOT / "resources" / "models" / "read_error_ont_r10.yaml"

# Below this many combos, process-pool overhead exceeds the gain.
_MIN_COMBOS_FOR_PARALLEL = 200

# Drop candidates whose aggregate mismatch rate exceeds this multiple of the
# best candidate's (ONT path only). Provisional (invariant 11).
_CANDIDATE_MISMATCH_RATE_MULTIPLIER = 10.0

# Floor on the best rate before applying the multiplier, so a 0.0 best rate
# cannot filter out a real second allele. Provisional.
_CANDIDATE_MISMATCH_RATE_FLOOR = 0.01

# Cap on candidates surviving the prefilter. Provisional.
_MAX_CANDIDATES_AFTER_PREFILTER = 50

# Survivors are ranked by read depth, then rate: rate alone let 1-read
# flukes crowd true alleles out of the cap. Per read-indistinguishable
# signature keep at most 2 (a real het pair can share a signature).
_MAX_CANDIDATES_PER_SIGNATURE = 2

# Pre-search floor: a combo whose EM-implied hard read split gives any
# component fewer than this many reads cannot win the argmax (summed logL
# otherwise lets 2 extreme reads beat a broad true signal; H2009 DQB1). The
# ratio floor is disabled (0.0): the EM split underestimates real skew, so
# any ratio traded one set of cases for another. Skewed real splits are left
# to pipeline.py's post-consensus ratio check. Provisional.
_ONT_MIN_READS_PER_HAPLOTYPE = 3
_ONT_MIN_HAPLOTYPE_READ_RATIO = 0.0

sys.path.insert(0, str(REPO_ROOT / "input"))
from fastq import CandidateRead  # noqa: E402

from error_model import KmerErrorModel, load_kmer_error_model, read_allele_log_likelihood  # noqa: E402

_EM_MAX_ITER = 100
_EM_TOL = 1e-8


@dataclass(frozen=True)
class ErrorModel:
    """Flat per-base binomial model (HiFi validation path)."""

    platform: str
    edit_error_rate: float
    unaligned_edit_rate: float


def load_error_model(path: Path) -> ErrorModel:
    """Load a versioned flat-model file (invariant 10)."""
    raw = yaml.safe_load(path.read_text())
    return ErrorModel(
        platform=raw["platform"],
        edit_error_rate=float(raw["edit_error_rate"]),
        unaligned_edit_rate=float(raw["unaligned_edit_rate"]),
    )


def read_log_likelihoods(
    matrix: GeneAssignmentMatrix,
    error_model: ErrorModel,
    columns: tuple[str, ...] | None = None,
) -> np.ndarray:
    """log P(read | allele) under the flat model, shape (n_reads, n_alleles),
    or (n_reads, len(columns)) if `columns` is given.

    Aligned cells: binomial over NM. Unaligned cells: the read's best span
    in this gene scored at `unaligned_edit_rate` (a floor, never log(0))."""
    eps = error_model.edit_error_rate
    aligned_ll = np.where(
        matrix.aln_block_len > 0,
        matrix.nm * np.log(eps) + (matrix.aln_block_len - matrix.nm) * np.log(1 - eps),
        0.0,
    )

    # Unaligned floor span: the read's longest alignment in this gene (> 0).
    row_span = matrix.aln_block_len.max(axis=1, keepdims=True)
    floor_eps = error_model.unaligned_edit_rate
    floor_ll = row_span * np.log(floor_eps)

    full = np.where(matrix.aln_block_len > 0, aligned_ll, floor_ll)
    if columns is None:
        return full
    idx = [matrix.allele_ids.index(a) for a in columns]
    return full[:, idx]


@dataclass(frozen=True)
class OntR10Config:
    context_model: KmerErrorModel
    unaligned_edit_rate: float


def load_ont_r10_config(path: Path = DEFAULT_ONT_R10_MODEL) -> OntR10Config:
    """Load the shipped ONT R10 config (invariant 10): context model plus the
    floor rate for cells with no alignment."""
    raw = yaml.safe_load(path.read_text())
    context_model_path = (path.parent / raw["context_model"]).resolve()
    return OntR10Config(
        context_model=load_kmer_error_model(context_model_path),
        unaligned_edit_rate=float(raw["unaligned_edit_rate"]),
    )


def load_allele_sequences(pool_fasta: Path, allele_ids: Sequence[str]) -> dict[str, str]:
    """Stream pool_fasta once, keeping only the requested sequences."""
    wanted = set(allele_ids)
    found: dict[str, str] = {}
    with pysam.FastxFile(str(pool_fasta)) as fh:
        for rec in fh:
            if rec.name in wanted:
                if rec.sequence is None:
                    raise ValueError(f"{pool_fasta}: record {rec.name!r} has no sequence")
                found[rec.name] = rec.sequence
                if len(found) == len(wanted):
                    break
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{pool_fasta}: missing sequence(s) for {sorted(missing)}")
    return found


def context_log_likelihoods(
    matrix: GeneAssignmentMatrix,
    read_sequences: dict[str, str],
    allele_sequences: dict[str, str],
    candidates: tuple[str, ...],
    config: OntR10Config,
) -> np.ndarray:
    """Context-aware log P(read | allele) for `candidates`, shape
    (n_reads, len(candidates)) -- the ONT R10 scorer (invariant 1).

    Cells without an alignment get the flat floor over the read's best span.
    Aligned cells are also charged the floor rate for read bases they don't
    cover relative to `row_span` (the best span any candidate achieved):
    minimap2 soft-clips freely, so otherwise a shorter candidate explaining
    fewer bases always scores higher. `row_span`, not the read length, is
    the reference: ultralong reads extend far beyond the gene.

    Capping this penalty at a candidate's own record length, or forgiving it
    for candidates of equal core length, were both tried and reverted: each
    let shorter wrong records win across many genes.
    """
    col_index = {a: i for i, a in enumerate(matrix.allele_ids)}
    row_span = matrix.aln_block_len.max(axis=1)
    floor_per_base = math.log(config.unaligned_edit_rate)

    n_reads = len(matrix.read_ids)
    ll = np.empty((n_reads, len(candidates)), dtype=np.float64)
    for i, read_id in enumerate(matrix.read_ids):
        floor_ll = float(row_span[i]) * floor_per_base
        read_seq = read_sequences[read_id]
        for k, allele in enumerate(candidates):
            span = matrix.spans.get((i, col_index[allele]))
            if span is None:
                ll[i, k] = floor_ll
                continue
            aligned_ll = read_allele_log_likelihood(
                config.context_model, allele_sequences[allele], read_seq,
                ref_start=span.ref_start, ref_end=span.ref_end,
                read_start=span.read_start, read_end=span.read_end,
                strand=span.strand, cigar=span.cigar,
            )
            uncovered = float(row_span[i]) - (span.read_end - span.read_start)
            ll[i, k] = aligned_ll + uncovered * floor_per_base
    return ll


def candidate_alleles(matrix: GeneAssignmentMatrix) -> tuple[str, ...]:
    """Alleles with >=1 real read alignment, sorted (invariant 9)."""
    has_support = (matrix.aln_block_len > 0).any(axis=0)
    return tuple(sorted(a for a, keep in zip(matrix.allele_ids, has_support) if keep))


def narrow_candidates_by_mismatch_rate(
    matrix: GeneAssignmentMatrix,
    candidates: tuple[str, ...],
    *,
    multiplier: float = _CANDIDATE_MISMATCH_RATE_MULTIPLIER,
    floor: float = _CANDIDATE_MISMATCH_RATE_FLOOR,
    max_candidates: int = _MAX_CANDIDATES_AFTER_PREFILTER,
) -> tuple[str, ...]:
    """Drop candidates whose aggregate mismatch rate exceeds
    `multiplier` x max(best rate, `floor`), using existing alignment stats.

    Survivors sharing a signature (same set of aligned reads, same rate
    bucket) are capped at `_MAX_CANDIDATES_PER_SIGNATURE`: IPD often has
    dozens of names differing only outside every read's span, which would
    otherwise fill the cap. Stage 4 re-types against the full pool, so no
    naming precision is lost. Remaining survivors are ranked by (depth desc,
    rate asc, name) and cut to `max_candidates`. Output is sorted."""
    col_index = {a: i for i, a in enumerate(matrix.allele_ids)}
    cols = [col_index[a] for a in candidates]
    aln = matrix.aln_block_len[:, cols]
    nm = matrix.nm[:, cols]
    aligned = aln > 0
    total_aln = np.where(aligned, aln, 0).sum(axis=0)
    total_nm = np.where(aligned, nm, 0).sum(axis=0)
    mismatch_rate = total_nm / total_aln
    n_reads_hit = aligned.sum(axis=0)

    threshold = multiplier * max(float(mismatch_rate.min()), floor)

    groups: dict[tuple[bytes, int], list[tuple[float, str, int]]] = {}
    for j, a in enumerate(candidates):
        rate = float(mismatch_rate[j])
        if rate > threshold:
            continue
        key = (aligned[:, j].tobytes(), round(rate / floor))
        groups.setdefault(key, []).append((rate, a, int(n_reads_hit[j])))

    representatives: list[tuple[float, str, int]] = []
    for members in groups.values():
        members.sort(key=lambda t: (t[0], t[1]))
        representatives.extend(members[:_MAX_CANDIDATES_PER_SIGNATURE])

    ranked = sorted(representatives, key=lambda t: (-t[2], t[0], t[1]))
    return tuple(sorted(a for _, a, _ in ranked[:max_candidates]))


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def _em_mixture_log_likelihood(ll_sub: np.ndarray) -> tuple[float, np.ndarray]:
    """EM for a k-component mixture over one template set. `ll_sub` is
    (n_reads, k). Returns (marginal logL at convergence, mixing weights)."""
    n_reads, k = ll_sub.shape
    weights = np.full(k, 1.0 / k)
    prev_ll = -np.inf
    for _ in range(_EM_MAX_ITER):
        log_weighted = ll_sub + np.log(weights)[None, :]
        log_row_total = _logsumexp(log_weighted, axis=1)
        total_ll = float(log_row_total.sum())

        # E-step: per-read responsibilities.
        resp = np.exp(log_weighted - log_row_total[:, None])
        # M-step: mixing weights are the mean responsibility per component.
        weights = resp.mean(axis=0)
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()

        if total_ll - prev_ll < _EM_TOL:
            prev_ll = total_ll
            break
        prev_ll = total_ll

    # Final logL at the converged weights.
    log_weighted = ll_sub + np.log(weights)[None, :]
    final_ll = float(_logsumexp(log_weighted, axis=1).sum())
    return final_ll, weights


def _hard_read_counts(ll_sub: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-component read counts from a hard argmax at the combo's own EM
    weights (cheap proxy for consensus.py's read partition)."""
    log_weighted = ll_sub + np.log(weights)[None, :]
    hard = np.argmax(log_weighted, axis=1)
    return np.bincount(hard, minlength=ll_sub.shape[1])


def _passes_read_count_floor(
    counts: np.ndarray, min_reads_per_haplotype: int, min_haplotype_read_ratio: float,
) -> bool:
    """True iff every component clears both the absolute and ratio floors.
    Zero thresholds make this a no-op."""
    if int(counts.min()) < min_reads_per_haplotype:
        return False
    max_count = int(counts.max())
    if max_count > 0 and counts.min() / max_count < min_haplotype_read_ratio:
        return False
    return True


# Set once per worker process so log_likelihoods is pickled once per worker,
# not per combo.
_worker_log_likelihoods: np.ndarray | None = None
_worker_col_index: dict[str, int] | None = None
_worker_min_reads_per_haplotype: int = 0
_worker_min_haplotype_read_ratio: float = 0.0


def _init_combo_worker(
    log_likelihoods: np.ndarray, col_index: dict[str, int],
    min_reads_per_haplotype: int = 0, min_haplotype_read_ratio: float = 0.0,
) -> None:
    global _worker_log_likelihoods, _worker_col_index
    global _worker_min_reads_per_haplotype, _worker_min_haplotype_read_ratio
    _worker_log_likelihoods = log_likelihoods
    _worker_col_index = col_index
    _worker_min_reads_per_haplotype = min_reads_per_haplotype
    _worker_min_haplotype_read_ratio = min_haplotype_read_ratio


def _score_combo(combo: tuple[str, ...]) -> tuple[float, tuple[str, ...], np.ndarray, bool]:
    assert _worker_log_likelihoods is not None and _worker_col_index is not None
    cols = [_worker_col_index[a] for a in combo]
    ll_sub = _worker_log_likelihoods[:, cols]
    set_ll, set_weights = _em_mixture_log_likelihood(ll_sub)
    eligible = _passes_read_count_floor(
        _hard_read_counts(ll_sub, set_weights),
        _worker_min_reads_per_haplotype, _worker_min_haplotype_read_ratio,
    )
    return set_ll, combo, set_weights, eligible


@dataclass(frozen=True)
class TemplateCall:
    gene: str
    cn: int
    alleles: tuple[str, ...]
    weights: tuple[float, ...]
    log_likelihood: float
    n_candidates: int
    lrt_statistic: float
    single_allele_log_likelihood: float
    single_allele: str
    # Full narrowed candidate set, so Stage 4 can realign against every
    # allele with read support (minimap2's length bias can hide a shorter
    # correct allele from a fresh whole-pool search).
    candidates: tuple[str, ...]


def _single_allele_call(
    gene: str, cn: int, candidates: tuple[str, ...],
    best_single_ll: float, best_single_allele: str,
) -> TemplateCall:
    """A single-allele call with `cn` preserved."""
    return TemplateCall(
        gene=gene, cn=cn, alleles=(best_single_allele,),
        weights=(1.0,), log_likelihood=best_single_ll,
        n_candidates=len(candidates), lrt_statistic=0.0,
        single_allele_log_likelihood=best_single_ll,
        single_allele=best_single_allele,
        candidates=candidates,
    )


def select_template_set(
    gene: str,
    candidates: tuple[str, ...],
    log_likelihoods: np.ndarray,
    cn: int,
    *,
    n_threads: int = 1,
    min_reads_per_haplotype: int = 0,
    min_haplotype_read_ratio: float = 0.0,
) -> TemplateCall:
    """Best template set of size `cn` from `candidates`, given a
    (n_reads, len(candidates)) log-likelihood matrix from either scorer.
    Also returns the LRT against the single-allele baseline.

    `min_reads_per_haplotype`/`min_haplotype_read_ratio` disqualify combos
    whose EM-implied read split fails either floor (default off). If every
    combo fails, or `0 < len(candidates) < cn` (homozygous with one
    supported allele), returns a single-allele call with `cn` preserved and
    LRT 0. No candidates at all raises.

    Deterministic (invariant 9): candidates must be sorted; ties keep the
    first combo. `n_threads > 1` parallelises the combo search; map() keeps
    call order, so results match the sequential path.
    """
    if cn < 1:
        raise ValueError(f"cn must be >= 1, got {cn} for gene {gene!r}")
    if len(candidates) == 0:
        raise ValueError(
            f"{gene}: no candidate allele(s) with read support at all "
            f"(need >= cn={cn} to form a template set)"
        )
    if log_likelihoods.shape[1] != len(candidates):
        raise ValueError(
            f"{gene}: log_likelihoods has {log_likelihoods.shape[1]} columns, "
            f"expected {len(candidates)} (one per candidate)"
        )

    # Single-allele baseline.
    single_scores = [
        (float(log_likelihoods[:, j].sum()), a) for j, a in enumerate(candidates)
    ]
    best_single_ll, best_single_allele = max(
        single_scores, key=lambda t: (t[0], t[1])
    )

    if cn == 1 or len(candidates) < cn:
        return _single_allele_call(gene, cn, candidates, best_single_ll, best_single_allele)

    col_index = {a: j for j, a in enumerate(candidates)}
    combos = list(itertools.combinations(candidates, cn))
    best_ll = -np.inf
    best_set: tuple[str, ...] | None = None
    best_weights: np.ndarray | None = None

    if n_threads <= 1 or len(combos) < _MIN_COMBOS_FOR_PARALLEL:
        for combo in combos:
            cols = [col_index[a] for a in combo]
            ll_sub = log_likelihoods[:, cols]
            set_ll, set_weights = _em_mixture_log_likelihood(ll_sub)
            if not _passes_read_count_floor(
                _hard_read_counts(ll_sub, set_weights),
                min_reads_per_haplotype, min_haplotype_read_ratio,
            ):
                continue
            if set_ll > best_ll:
                best_ll, best_set, best_weights = set_ll, combo, set_weights
    else:
        chunksize = max(1, len(combos) // (n_threads * 4))
        with ProcessPoolExecutor(
            max_workers=n_threads, initializer=_init_combo_worker,
            initargs=(log_likelihoods, col_index, min_reads_per_haplotype, min_haplotype_read_ratio),
        ) as pool:
            for set_ll, combo, set_weights, eligible in pool.map(_score_combo, combos, chunksize=chunksize):
                if eligible and set_ll > best_ll:
                    best_ll, best_set, best_weights = set_ll, combo, set_weights

    if best_set is None:
        # Every combo failed the read-count floor: collapse.
        return _single_allele_call(gene, cn, candidates, best_single_ll, best_single_allele)

    assert best_weights is not None
    lrt = 2.0 * (best_ll - best_single_ll)
    return TemplateCall(
        gene=gene, cn=cn, alleles=best_set,
        weights=tuple(float(w) for w in best_weights),
        log_likelihood=best_ll, n_candidates=len(candidates),
        lrt_statistic=lrt, single_allele_log_likelihood=best_single_ll,
        single_allele=best_single_allele,
        candidates=candidates,
    )


def select_template_set_flat(
    matrix: GeneAssignmentMatrix,
    cn: int,
    error_model: ErrorModel,
    *,
    n_threads: int = 1,
) -> TemplateCall:
    """Flat-model wrapper (HiFi validation path)."""
    candidates = candidate_alleles(matrix)
    ll = read_log_likelihoods(matrix, error_model, columns=candidates)
    return select_template_set(matrix.gene, candidates, ll, cn, n_threads=n_threads)


def select_template_set_ont_r10(
    matrix: GeneAssignmentMatrix,
    reads: Sequence[CandidateRead],
    cn: int,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    config: OntR10Config | None = None,
    n_threads: int = 1,
    minimap2_bin: str = "minimap2",
    min_reads_per_haplotype: int = _ONT_MIN_READS_PER_HAPLOTYPE,
    min_haplotype_read_ratio: float = _ONT_MIN_HAPLOTYPE_READ_RATIO,
) -> TemplateCall:
    """Stage 2 entry point for ONT R10 (invariant 1).

    Narrows candidates (narrow_candidates_by_mismatch_rate), densely realigns
    every read against every narrowed candidate (Stage 1's n_best=1 matrix
    is too sparse to pick among near-identical names), scores with the
    context model and runs select_template_set() with the pre-search
    read-count floor. `n_threads` parallelises realignment and the combo
    search."""
    if config is None:
        config = load_ont_r10_config()
    candidates = narrow_candidates_by_mismatch_rate(matrix, candidate_alleles(matrix))
    read_id_set = set(matrix.read_ids)
    candidate_reads = [r for r in reads if r.read_id in read_id_set]
    allele_sequences = load_allele_sequences(pool_fasta, candidates)

    dense_matrix = realign_to_candidates(
        matrix.gene, candidate_reads, allele_sequences,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    dense_read_sequences = {
        r.read_id: r.sequence for r in candidate_reads if r.read_id in set(dense_matrix.read_ids)
    }
    ll = context_log_likelihoods(
        dense_matrix, dense_read_sequences, allele_sequences, candidates, config,
    )
    return select_template_set(
        matrix.gene, candidates, ll, cn, n_threads=n_threads,
        min_reads_per_haplotype=min_reads_per_haplotype,
        min_haplotype_read_ratio=min_haplotype_read_ratio,
    )
