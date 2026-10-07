"""Stage 4 -- typing with per-field posteriors (DESIGN.md S4).

(Not named `typing.py`: that shadows the stdlib module on sys.path.)

Scores the Stage 3 consensus, not the raw reads, against the gene's full
pool: the consensus is wrapped as one synthetic read and run through
assign.build_gene_matrices() and genotype.context_log_likelihoods().

Field GQs marginalise the posterior over every candidate sharing the called
allele's first k fields, k = 1..n_fields. Uniform prior over candidates (a
frequency/LD prior would bias against rare/novel haplotypes, DESIGN.md S6).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "input"))
from fastq import AnchorHit, CandidateRead  # noqa: E402

from assign import (  # noqa: E402
    DEFAULT_POOL_FASTA,
    DEFAULT_POOL_META,
    AlignmentSpan,
    GeneAssignmentMatrix,
    build_gene_matrices,
    realign_to_candidates,
    subset_matrix_by_read_ids,
)
from consensus import HaplotypeConsensus  # noqa: E402
from error_model import build_aligned_strings  # noqa: E402
from genotype import (  # noqa: E402
    OntR10Config,
    candidate_alleles,
    context_log_likelihoods,
    load_allele_sequences,
    load_ont_r10_config,
)

DEFAULT_TYPING_PARAMS = REPO_ROOT / "resources" / "models" / "stage4_typing.yaml"
DEFAULT_POOL_MODELS = REPO_ROOT / "resources" / "templates" / "pool" / "ref.gene.models.jsonl.gz"

_MIN_P_WRONG = 1e-6
_MAX_GQ = 60.0

# Mismatch rates within this tolerance tie, and coverage length decides.
# Widening it (0.001) and replacing it with a two-proportion z-test were
# both tried and reverted: mismatch counts against a consensus are fixed
# sequence differences, not sampling noise, so either broke many genes.
# Provisional (invariant 11).
_MISMATCH_RATE_TIE_TOLERANCE = 0.0001

# Min secondary/primary score ratio for the dense-realignment fallback in
# _type_haplotype_from_single_row_matrix(). assign.py's -p0.95 drops a
# correct but shorter-UTR record entirely against a full-length consensus;
# this only decides whether a record exists, the rate rule still judges it.
# Provisional.
_STAGE4_FALLBACK_MIN_SECONDARY_RATIO = 0.5

# Synthetic read id for the consensus; never a real allele id.
_CONSENSUS_READ_ID = "consensus"


# Weight of a CDS mismatch in the called-allele mismatch rate. Field-3/4
# differences are non-coding by IPD convention, so a wrong lineage shows CDS
# differences while true novel variation is mostly non-coding (HG03009
# DRB1*14:04:01:01: 21 intronic, 0 CDS; rival *14:07:01: 13 with 5 CDS).
# Provisional (invariant 11).
#
# _GeneModelLite / _locate / _diff_against_reference_lite duplicate
# novelty.py (which imports this module, so importing back is circular).
# Keep them in sync.
_CDS_MISMATCH_WEIGHT = 5.0


@dataclass(frozen=True)
class _GeneModelLite:
    cds_exons: tuple[tuple[int, int], ...]
    introns: tuple[tuple[int | None, int, int], ...]
    utrs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _RawVariant:
    pos: int
    ref: str
    alt: str


_GENE_MODEL_CACHE: dict[Path, dict[str, _GeneModelLite]] = {}


def _all_gene_models_lite(models_path: Path) -> dict[str, _GeneModelLite]:
    """All gene models from `models_path`, parsed once per process (called
    per candidate, so re-reading the file each time was prohibitively slow)."""
    cached = _GENE_MODEL_CACHE.get(models_path)
    if cached is not None:
        return cached
    import gzip
    import json

    out: dict[str, _GeneModelLite] = {}
    if models_path.exists():
        with gzip.open(models_path, "rt") as fh:
            for line in fh:
                obj = json.loads(line)
                out[obj["allele"]] = _GeneModelLite(
                    cds_exons=tuple((s, e) for s, e in obj["cds_exons"]),
                    introns=tuple((n, s, e) for n, s, e in obj["introns"]),
                    utrs=tuple((s, e) for s, e in obj["utrs"]),
                )
    _GENE_MODEL_CACHE[models_path] = out
    return out


def _load_gene_model_lite(allele: str, models_path: Path = DEFAULT_POOL_MODELS) -> _GeneModelLite | None:
    return _all_gene_models_lite(models_path).get(allele)


def _locate_lite(pos1: int, gm: _GeneModelLite) -> str:
    if any(s <= pos1 <= e for s, e in gm.cds_exons):
        return "cds"
    if any(s <= pos1 <= e for _, s, e in gm.introns):
        return "intron"
    if any(s <= pos1 <= e for s, e in gm.utrs):
        return "utr"
    return "unannotated"


def _diff_against_reference_lite(
    ref_seq: str, consensus_seq: str, span: AlignmentSpan
) -> list[_RawVariant]:
    aligned_ref, aligned_consensus = build_aligned_strings(
        ref_seq, consensus_seq,
        ref_start=span.ref_start, ref_end=span.ref_end,
        read_start=span.read_start, read_end=span.read_end,
        strand=span.strand, cigar=span.cigar,
    )
    variants: list[_RawVariant] = []
    ref_pos = span.ref_start
    i, n = 0, len(aligned_ref)
    while i < n:
        r, q = aligned_ref[i], aligned_consensus[i]
        if r != "-":
            if q != r:
                variants.append(_RawVariant(pos=ref_pos + 1, ref=r, alt=(q if q != "-" else "")))
            ref_pos += 1
            i += 1
        else:
            j = i
            while j < n and aligned_ref[j] == "-":
                j += 1
            variants.append(_RawVariant(pos=ref_pos + 1, ref="", alt=aligned_consensus[i:j]))
            i = j
    return variants


def _cds_weighted_mismatches(
    allele: str, consensus_seq: str, span: AlignmentSpan, allele_seq: str,
) -> float:
    """Mismatch count against `allele`, weighted by base length (like `nm`)
    with CDS bases weighted `_CDS_MISMATCH_WEIGHT`. With no CDS variant
    this equals nm, so per-block counting (which let one long STR insertion
    count as one mismatch) is avoided. Weight 1 everywhere if the allele has
    no gene model."""
    gm = _load_gene_model_lite(allele)
    variants = _diff_against_reference_lite(allele_seq, consensus_seq, span)
    total = 0.0
    for v in variants:
        base_len = max(len(v.ref), len(v.alt))
        weight = _CDS_MISMATCH_WEIGHT if (gm is not None and _locate_lite(v.pos, gm) == "cds") else 1.0
        total += base_len * weight
    return total


@dataclass(frozen=True)
class TypingParams:
    ambiguity_log_margin: float

    @classmethod
    def load(cls, path: Path = DEFAULT_TYPING_PARAMS) -> "TypingParams":
        raw = yaml.safe_load(path.read_text())
        return cls(ambiguity_log_margin=float(raw["ambiguity_log_margin"]))


@dataclass(frozen=True)
class TypingCall:
    gene: str
    consensus_template_allele: str  # Stage 3's own selected template, for provenance
    called_allele: str              # Stage 4's nearest-posterior allele for this consensus
    n_fields: int
    field_gq: tuple[float, ...]     # GQ1..GQ(n_fields), phred-scaled
    posterior: dict[str, float]     # candidate allele -> posterior mass
    ambiguity_set: tuple[str, ...]  # candidates within TypingParams.ambiguity_log_margin of the best
    called_allele_span: AlignmentSpan  # consensus-vs-called_allele CIGAR (stages/novelty.py's diff input)


def _fields_of(allele: str) -> tuple[str, ...]:
    if "*" not in allele:
        raise ValueError(f"{allele!r}: not a star-allele name, cannot split into fields")
    return tuple(allele.split("*", 1)[1].split(":"))


def _phred(p_wrong: float) -> float:
    return min(-10.0 * math.log10(max(p_wrong, _MIN_P_WRONG)), _MAX_GQ)


def type_haplotype(
    consensus: HaplotypeConsensus,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    config: OntR10Config | None = None,
    params: TypingParams | None = None,
    extra_candidates: tuple[str, ...] = (),
    reads: Sequence[CandidateRead] | None = None,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
) -> TypingCall:
    """Type one Stage 3 consensus against its gene's full pool. Raises if the
    consensus aligns to no pool allele.

    `called_allele` is chosen by CDS-weighted mismatch rate, then coverage
    length on ties -- not argmax(posterior). The posterior's
    uncovered-length penalty makes a shorter but clean record (DRB1
    *15:01:01:03) lose to a longer one with real mismatches. `posterior`,
    `field_gq` and `ambiguity_set` still come from the context-model score.
    A Stage 1 read-depth gate on candidates was tried and reverted: per-read
    depth carries the same record-length bias."""
    if config is None:
        config = load_ont_r10_config()
    if params is None:
        params = TypingParams.load()

    synthetic_read = CandidateRead(
        read_id=_CONSENSUS_READ_ID,
        sequence=consensus.sequence,
        quality=None,
        hits=(AnchorHit(gene=consensus.gene, allele="__consensus__", coverage=1.0, mismatch_rate=0.0),),
    )
    matrices = build_gene_matrices(
        [synthetic_read], pool_fasta=pool_fasta, pool_meta=pool_meta,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    matrix = matrices.get(consensus.gene)
    if matrix is None or not matrix.read_ids:
        raise ValueError(
            f"{consensus.gene}: consensus for template {consensus.template_allele!r} "
            f"did not align to any allele in {pool_fasta} -- cannot type"
        )
    return _type_haplotype_from_single_row_matrix(
        consensus, matrix, pool_fasta=pool_fasta, config=config, params=params,
        extra_candidates=extra_candidates, reads=reads,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )


def _break_tie_by_real_reads(
    gene: str,
    tied_candidates: tuple[str, ...],
    reads: Sequence[CandidateRead],
    assigned_read_ids: tuple[str, ...],
    *,
    pool_fasta: Path,
    config: OntR10Config,
    n_threads: int,
    minimap2_bin: str,
) -> str | None:
    """Break a rate/coverage tie by summing per-read context log-likelihoods
    over this haplotype's real assigned reads (the consensus can't show a
    split vote). Returns None if no assigned reads are found or none
    realign, so the caller keeps its default pick."""
    read_id_set = set(assigned_read_ids)
    own_reads = [r for r in reads if r.read_id in read_id_set]
    if not own_reads:
        return None
    allele_sequences = load_allele_sequences(pool_fasta, tied_candidates)
    matrix = realign_to_candidates(
        gene, own_reads, allele_sequences, n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    if not matrix.read_ids:
        return None
    read_sequences = {
        r.read_id: r.sequence for r in own_reads if r.read_id in set(matrix.read_ids)
    }
    ll = context_log_likelihoods(matrix, read_sequences, allele_sequences, tied_candidates, config)
    totals = ll.sum(axis=0)
    return tied_candidates[int(np.argmax(totals))]


def _type_haplotype_from_single_row_matrix(
    consensus: HaplotypeConsensus,
    matrix: GeneAssignmentMatrix,
    *,
    pool_fasta: Path,
    config: OntR10Config,
    params: TypingParams,
    extra_candidates: tuple[str, ...] = (),
    reads: Sequence[CandidateRead] | None = None,
    n_threads: int = 1,
    minimap2_bin: str = "minimap2",
) -> TypingCall:
    """Scoring body of `type_haplotype()` for a one-row matrix (row 0 is
    this consensus, so candidate_alleles() reflects only its own support).

    `extra_candidates` (Stage 2's narrowed set): if the naive best candidate
    has nm > 0 and some extra candidate is missing from the whole-pool hits,
    densely realign the consensus against found + extra candidates.
    minimap2's length bias can hide a shorter correct record (TAP2
    *01:01:07:06 vs *05). Exact matches skip this."""
    read_id = matrix.read_ids[0]
    candidates = candidate_alleles(matrix)
    if not candidates:
        raise ValueError(f"{consensus.gene}: no candidate alleles with real alignment support")

    naive_col_index = {a: i for i, a in enumerate(matrix.allele_ids)}
    naive_best_rate = min(
        int(matrix.nm[0, naive_col_index[a]]) / int(matrix.aln_block_len[0, naive_col_index[a]])
        for a in candidates
    )

    missing = tuple(a for a in extra_candidates if a not in candidates)
    if missing and naive_best_rate > 0.0:
        all_candidates = tuple(sorted(set(candidates) | set(extra_candidates)))
        allele_sequences_dense = load_allele_sequences(pool_fasta, all_candidates)
        synthetic_read = CandidateRead(
            read_id=read_id, sequence=consensus.sequence, quality=None,
            hits=(AnchorHit(gene=consensus.gene, allele="__consensus__", coverage=1.0, mismatch_rate=0.0),),
        )
        matrix = realign_to_candidates(
            consensus.gene, [synthetic_read], allele_sequences_dense,
            n_threads=n_threads, minimap2_bin=minimap2_bin,
            min_secondary_ratio=_STAGE4_FALLBACK_MIN_SECONDARY_RATIO,
        )
        read_id = matrix.read_ids[0]
        candidates = candidate_alleles(matrix)

    allele_sequences = load_allele_sequences(pool_fasta, candidates)
    read_sequences = {read_id: consensus.sequence}
    ll = context_log_likelihoods(
        matrix, read_sequences, allele_sequences, candidates, config,
    )
    log_probs = ll[0]

    m = float(log_probs.max())
    unnorm = np.exp(log_probs - m)
    posterior = unnorm / unnorm.sum()

    col_index = {a: i for i, a in enumerate(matrix.allele_ids)}

    def _rate_and_coverage(allele: str) -> tuple[float, int]:
        j = col_index[allele]
        aln = int(matrix.aln_block_len[0, j])
        span = matrix.spans[(0, j)]
        weighted_mismatches = _cds_weighted_mismatches(
            allele, consensus.sequence, span, allele_sequences[allele],
        )
        return weighted_mismatches / aln, span.read_end - span.read_start

    def _tiebreak_key(i: int) -> tuple[int, int]:
        rate, coverage = _rate_and_coverage(candidates[i])
        return round(rate / _MISMATCH_RATE_TIE_TOLERANCE), -coverage

    best_idx = min(range(len(candidates)), key=_tiebreak_key)

    # Genuine rate/coverage tie (the consensus can't express a split vote,
    # e.g. TAP2 *03 vs *06): let the real assigned reads decide. Otherwise
    # keep the deterministic first candidate.
    tied_idx = [i for i in range(len(candidates)) if _tiebreak_key(i) == _tiebreak_key(best_idx)]
    if len(tied_idx) > 1 and reads is not None and consensus.assigned_read_ids:
        tied_candidates = tuple(candidates[i] for i in tied_idx)
        refined = _break_tie_by_real_reads(
            consensus.gene, tied_candidates, reads, consensus.assigned_read_ids,
            pool_fasta=pool_fasta, config=config, n_threads=n_threads, minimap2_bin=minimap2_bin,
        )
        if refined is not None:
            best_idx = candidates.index(refined)

    called_allele = candidates[best_idx]
    best_ll = float(log_probs[best_idx])

    allele_col = col_index[called_allele]
    called_allele_span = matrix.spans.get((0, allele_col))
    if called_allele_span is None:
        raise ValueError(
            f"{consensus.gene}: {called_allele!r} was a scored candidate but "
            f"assign.py captured no alignment span for it -- matrix construction bug"
        )

    candidate_fields = {c: _fields_of(c) for c in candidates}
    called_fields = candidate_fields[called_allele]
    n_fields = len(called_fields)
    field_gq = []
    for k in range(1, n_fields + 1):
        prefix = called_fields[:k]
        mass = sum(
            float(p) for c, p in zip(candidates, posterior)
            if candidate_fields[c][:k] == prefix
        )
        field_gq.append(_phred(1.0 - mass))

    ambiguity = tuple(sorted(
        c for c, lp in zip(candidates, log_probs)
        if best_ll - float(lp) <= params.ambiguity_log_margin
    ))

    return TypingCall(
        gene=consensus.gene,
        consensus_template_allele=consensus.template_allele,
        called_allele=called_allele,
        n_fields=n_fields,
        field_gq=tuple(field_gq),
        posterior={c: float(p) for c, p in zip(candidates, posterior)},
        ambiguity_set=ambiguity,
        called_allele_span=called_allele_span,
    )


def type_haplotypes_batch(
    consensuses: Sequence[tuple[str, HaplotypeConsensus]],
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    config: OntR10Config | None = None,
    params: TypingParams | None = None,
    extra_candidates_by_key: dict[str, tuple[str, ...]] | None = None,
    reads: Sequence[CandidateRead] | None = None,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
) -> tuple[dict[str, TypingCall], dict[str, str]]:
    """Type many consensuses with one minimap2 call. Each consensus is then
    scored on its own single-row subset of the batched matrix, so it never
    sees another haplotype's candidates; results match calling
    `type_haplotype()` per consensus.

    Returns `(typing_calls, failures)` keyed by the `consensuses` keys; each
    key lands in exactly one. Per-consensus failures don't affect others.

    `extra_candidates_by_key`: per-key fallback candidates (see
    `_type_haplotype_from_single_row_matrix`). `reads`: Stage 0 reads for
    the read-level tie-break; None disables it."""
    if config is None:
        config = load_ont_r10_config()
    if params is None:
        params = TypingParams.load()
    if extra_candidates_by_key is None:
        extra_candidates_by_key = {}
    if not consensuses:
        return {}, {}

    synthetic_reads = [
        CandidateRead(
            read_id=key, sequence=hap.sequence, quality=None,
            hits=(AnchorHit(gene=hap.gene, allele="__consensus__", coverage=1.0, mismatch_rate=0.0),),
        )
        for key, hap in consensuses
    ]
    matrices = build_gene_matrices(
        synthetic_reads, pool_fasta=pool_fasta, pool_meta=pool_meta,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )

    typing_calls: dict[str, TypingCall] = {}
    failures: dict[str, str] = {}
    for key, hap in consensuses:
        matrix = matrices.get(hap.gene)
        if matrix is None or key not in matrix.read_ids:
            failures[key] = (
                f"{hap.gene}: consensus for template {hap.template_allele!r} "
                f"did not align to any allele in {pool_fasta} -- cannot type"
            )
            continue
        row_matrix = subset_matrix_by_read_ids(matrix, {key})
        assert row_matrix is not None  # key in matrix.read_ids guarantees >=1 surviving row
        try:
            typing_calls[key] = _type_haplotype_from_single_row_matrix(
                hap, row_matrix, pool_fasta=pool_fasta, config=config, params=params,
                extra_candidates=extra_candidates_by_key.get(key, ()), reads=reads,
                n_threads=n_threads, minimap2_bin=minimap2_bin,
            )
        except ValueError as e:
            failures[key] = str(e)
    return typing_calls, failures
