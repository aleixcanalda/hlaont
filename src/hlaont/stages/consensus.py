"""Stage 3 -- consensus with per-base QV (DESIGN.md S3/S4 intro, "Read
partition, then consensus. Every base gets a QV.").

Template-and-polish, not de novo assembly (DESIGN.md's non-goals, S1): each
haplotype copy in Stage 2's winning `TemplateCall` is polished against its
own selected template using the reads assigned to it, reusing the CIGAR
alignments `stages/assign.py` already captured (`GeneAssignmentMatrix.spans`)
rather than a fresh multiple-sequence-alignment/POA pass. This keeps the
positional numbering identical to the template's own IPD coordinates, which
is exactly what `stages/novelty.py` (S5) needs to look a variant up against
`resources/templates/pool/ref.gene.models.jsonl.gz`'s CDS/exon/intron/UTR
ranges with no coordinate translation.

**Read partition is not this module's decision.** Stage 2's `TemplateCall`
for `cn=2` always contains two *distinct* candidate alleles (`itertools.
combinations` cannot repeat one), even for a truly homozygous sample --
DESIGN.md S2e's LRT (`TemplateCall.lrt_statistic` vs `single_allele`) is how
a caller detects that case, not something this module re-derives. There is
no re-derived-for-15X threshold here for "collapse to homozygous" (invariant
11 -- nothing to fit it against yet without real ONT R10 fixtures), so that
call belongs to the orchestration layer once it exists: pass `call.alleles`/
`call.weights` as-is for a heterozygous call, or a manufactured
single-allele `TemplateCall` (one allele, weight 1.0) for a collapsed one.
This module only ever partitions/polishes whatever alleles/weights it is
handed.

Per-base error rate is each read's own *aggregate* mismatch rate against its
assigned template (`GeneAssignmentMatrix.mismatch_rate()`, already computed
by assign.py) -- a flat per-read rate, not the finer k-mer context model
`stages/error_model.py` uses for Stage 2's allele-selection scoring. This is
a real simplification, flagged rather than silently assumed: the eventual
homopolymer-context-aware per-position model is DESIGN.md S3's own stated
target ("fitted from our own cohort... participants who are homozygous-
reference... give a direct empirical estimate"), which needs real cohort
data `calib/` doesn't have yet. Revisit once it does.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "input"))
from fastq import CandidateRead  # noqa: E402

from assign import DEFAULT_POOL_FASTA, GeneAssignmentMatrix, realign_to_candidates  # noqa: E402
from error_model import build_aligned_strings  # noqa: E402
from genotype import (  # noqa: E402
    OntR10Config,
    TemplateCall,
    context_log_likelihoods,
    load_allele_sequences,
    load_ont_r10_config,
)

# Floor/ceiling on a read's own aggregate mismatch rate before it is used as
# a per-base error probability below -- a pragmatic numerical guard (a
# perfect-match read would otherwise contribute log(0)), not a fitted
# parameter, in the same spirit as error_model.py's own _FLOOR_PROB.
_MIN_READ_ERR = 1e-4
_MAX_READ_ERR = 0.5

# Phred-scale floor/ceiling on P(consensus call wrong) -- bounds the reported
# QV to a sane range (0-60) rather than an unbounded float; same rationale
# as the read-error floor above.
_MIN_P_WRONG = 1e-6
_MAX_QV = 60.0

_BASES = ("A", "C", "G", "T")
_DELETION = "-"


@dataclass(frozen=True)
class ConsensusVariant:
    """One position where the consensus differs from the selected template,
    in the template's own 1-based coordinate numbering (matches
    resources/templates/pool/ref.gene.models.jsonl.gz directly, S5's payoff).

    A substitution/deletion has `pos` at the affected template base
    (`ref` = template base(s), `alt` = consensus base or '' for a deletion).
    An insertion has `ref=''`, `pos` = the template position it sits before,
    `alt` = the inserted sequence. Not VCF-anchored (no leading shared base)
    -- that conversion is an output-writing concern, not this module's."""

    pos: int
    ref: str
    alt: str
    qv: float
    depth: int
    # Reads whose own observation at this position disagreed with the
    # winning call (2026-09-13, added for stages/novelty.py's read-support
    # variant filter -- see that module's `_MIN_VARIANT_SUPPORT`). Distinct
    # from `qv`: qv is the model-based P(wrong) that the project's own
    # 2026-09-11/12 sessions found saturates at 60 regardless of real
    # disagreement; `dissent` is the raw vote count `_qv_and_best` already
    # has on hand while computing that saturating score, kept here instead
    # of discarded.
    dissent: int


@dataclass(frozen=True)
class HaplotypeConsensus:
    gene: str
    template_allele: str
    n_reads_assigned: int
    sequence: str
    qv: tuple[int, ...]
    variants: tuple[ConsensusVariant, ...]
    # The real read IDs assign_reads_to_haplotypes() partitioned to this
    # haplotype copy (2026-09-02, added for stages/hla_typing.py's
    # read-level tie-break -- see that module's docstring). Previously
    # computed here and discarded once n_reads_assigned's count was taken;
    # kept now so a downstream caller can re-score against this
    # haplotype's own real reads instead of just the one collapsed
    # consensus sequence.
    assigned_read_ids: tuple[str, ...] = ()


def read_haplotype_responsibilities(
    matrix: GeneAssignmentMatrix,
    reads: Sequence[CandidateRead],
    call: TemplateCall,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    config: OntR10Config | None = None,
) -> np.ndarray:
    """(n_reads, cn) responsibility matrix for exactly `call.alleles`, read
    order matching `matrix.read_ids` -- a single E-step at `call.weights`
    (already EM-converged by stages/genotype.py's select_template_set), not
    a fresh EM run."""
    if config is None:
        config = load_ont_r10_config()
    read_id_set = set(matrix.read_ids)
    read_sequences = {r.read_id: r.sequence for r in reads if r.read_id in read_id_set}
    allele_sequences = load_allele_sequences(pool_fasta, call.alleles)
    ll = context_log_likelihoods(
        matrix, read_sequences, allele_sequences, call.alleles, config
    )
    log_weighted = ll + np.log(np.asarray(call.weights))[None, :]
    m = np.max(log_weighted, axis=1, keepdims=True)
    unnorm = np.exp(log_weighted - m)
    return unnorm / unnorm.sum(axis=1, keepdims=True)


def assign_reads_to_haplotypes(responsibilities: np.ndarray) -> np.ndarray:
    """Hard read->haplotype assignment: argmax responsibility per read.
    np.argmax keeps the first (lowest-index) maximum on a tie, matching
    `call.alleles`'s fixed order -- deterministic (invariant 9), not an
    arbitrary tie-break."""
    return np.argmax(responsibilities, axis=1)


def _read_err_rate(matrix: GeneAssignmentMatrix, read_idx: int, allele_idx: int) -> float:
    rate = float(matrix.mismatch_rate()[read_idx, allele_idx])
    return min(max(rate, _MIN_READ_ERR), _MAX_READ_ERR)


def _qv_and_best(scores: dict[str, float], preferred_on_tie: str) -> tuple[str, float]:
    """argmax over `scores` (log-likelihoods), ties broken toward
    `preferred_on_tie` (the template's own current state -- do not call a
    variant on a tie), plus the phred-scaled P(wrong) for the winner."""
    order = [preferred_on_tie] + [s for s in scores if s != preferred_on_tie]
    best = max(order, key=lambda s: scores[s])
    values = np.array(list(scores.values()))
    m = values.max()
    total = m + math.log(np.sum(np.exp(values - m)))
    p_best = math.exp(scores[best] - total)
    p_wrong = max(1.0 - p_best, _MIN_P_WRONG)
    qv = min(-10.0 * math.log10(p_wrong), _MAX_QV)
    return best, qv


def _call_substitution_or_deletion(
    observations: list[tuple[str, float]], template_base: str
) -> tuple[str, float, int, int]:
    """observations: list of (observed_base_or_'' -for-deletion, read_err).
    Returns (consensus_state, qv, depth, dissent); consensus_state is a
    base or '-'. 5-state model (4 bases + deletion), 4 alternatives per
    state -- a standard base-caller approximation, not a fitted
    substitution/deletion balance (see module docstring)."""
    if not observations:
        return template_base, 0.0, 0, 0

    states = _BASES + (_DELETION,)
    scores = {s: 0.0 for s in states}
    for obs, err in observations:
        obs_state = obs if obs else _DELETION
        log_match = math.log(1.0 - err)
        log_mismatch = math.log(err / 4.0)
        for s in states:
            scores[s] += log_match if s == obs_state else log_mismatch

    best, qv = _qv_and_best(scores, preferred_on_tie=template_base)
    dissent = sum(1 for obs, _ in observations if (obs if obs else _DELETION) != best)
    return best, qv, len(observations), dissent


def _call_insertion(
    observations: list[tuple[str, float]]
) -> tuple[str, float, int, int]:
    """observations: (inserted_sequence_or_''-for-no-insertion, read_err)
    for every read with an opinion at this anchor position (both reads that
    inserted something here and reads that, covering this position, did
    not). Returns (consensus_insert, qv, depth, dissent); '' means no
    insertion called. Ties (including "no evidence either way beats a lone
    read's homopolymer slip") are broken toward '' -- conservative by
    construction, consistent with DESIGN.md's caution about over-calling
    indels from single-sample ONT homopolymer noise (S1/S6)."""
    if not observations:
        return "", 0.0, 0, 0

    distinct = sorted({obs for obs, _ in observations})
    if distinct == [""]:
        return "", 0.0, len(observations), 0

    states = tuple(distinct) if "" in distinct else ("",) + tuple(distinct)
    n_alt = max(len(states) - 1, 1)
    scores = {s: 0.0 for s in states}
    for obs, err in observations:
        log_match = math.log(1.0 - err)
        log_mismatch = math.log(err / n_alt)
        for s in states:
            scores[s] += log_match if s == obs else log_mismatch

    best, qv = _qv_and_best(scores, preferred_on_tie="")
    dissent = sum(1 for obs, _ in observations if obs != best)
    return best, qv, len(observations), dissent


def _haplotype_pileup(
    matrix: GeneAssignmentMatrix,
    reads: Sequence[CandidateRead],
    template_allele: str,
    template_seq: str,
    assigned_read_ids: Sequence[str],
) -> tuple[dict[int, list[tuple[str, float]]], dict[int, list[tuple[str, float]]]]:
    """Builds (substitution/deletion observations, insertion observations)
    per 0-based template position, from the CIGAR alignments assign.py
    already captured for each assigned read against `template_allele`
    (falls back to skipping a read with no captured span there -- no
    alignment means no pileup evidence, the same "no information, not a
    zero" stance the rest of this codebase takes)."""
    read_seq_by_id = {r.read_id: r.sequence for r in reads}
    read_index = {rid: i for i, rid in enumerate(matrix.read_ids)}
    allele_index = {a: i for i, a in enumerate(matrix.allele_ids)}
    allele_col = allele_index[template_allele]

    observations: dict[int, list[tuple[str, float]]] = {}
    insertions: dict[int, list[tuple[str, float]]] = {}

    for read_id in assigned_read_ids:
        row = read_index[read_id]
        span = matrix.spans.get((row, allele_col))
        if span is None:
            continue
        read_seq = read_seq_by_id[read_id]
        aligned_ref, aligned_read = build_aligned_strings(
            template_seq, read_seq,
            ref_start=span.ref_start, ref_end=span.ref_end,
            read_start=span.read_start, read_end=span.read_end,
            strand=span.strand, cigar=span.cigar,
        )
        err = _read_err_rate(matrix, row, allele_col)

        # One pass over this read's gapped alignment: record its
        # substitution/deletion observation at every ref position it
        # spans, and its insertion event (if any) at each insertion anchor.
        # `positions_seen` collects every ref position this read covered,
        # so a second, separate loop can add its implicit "no insertion"
        # vote everywhere it *didn't* insert -- without that, a lone read's
        # homopolymer-slip insertion would have no opposing evidence at all.
        ref_pos = span.ref_start
        positions_seen: list[int] = []
        inserted_at: set[int] = set()
        i = 0
        n = len(aligned_ref)
        while i < n:
            r, q = aligned_ref[i], aligned_read[i]
            if r != "-":
                obs = q if q != "-" else ""
                observations.setdefault(ref_pos, []).append((obs, err))
                positions_seen.append(ref_pos)
                ref_pos += 1
                i += 1
            else:
                j = i
                buf = []
                while j < n and aligned_ref[j] == "-":
                    buf.append(aligned_read[j])
                    j += 1
                insertions.setdefault(ref_pos, []).append(("".join(buf), err))
                inserted_at.add(ref_pos)
                i = j
        for pos in positions_seen:
            if pos not in inserted_at:
                insertions.setdefault(pos, []).append(("", err))

    return observations, insertions


def build_haplotype_consensus(
    gene: str,
    matrix: GeneAssignmentMatrix,
    reads: Sequence[CandidateRead],
    template_allele: str,
    template_seq: str,
    assigned_read_ids: Sequence[str],
) -> HaplotypeConsensus:
    """Polish `template_allele` against the reads assigned to it (from
    assign_reads_to_haplotypes()), producing a consensus sequence, one QV
    per emitted base, and a variant list relative to the template."""
    observations, insertions = _haplotype_pileup(
        matrix, reads, template_allele, template_seq, assigned_read_ids
    )

    seq_out: list[str] = []
    qv_out: list[int] = []
    variants: list[ConsensusVariant] = []

    for ref_pos0 in range(len(template_seq)):
        pos1 = ref_pos0 + 1
        ins_obs = insertions.get(ref_pos0, [])
        ins_state, ins_qv, ins_depth, ins_dissent = _call_insertion(ins_obs)
        if ins_state:
            seq_out.extend(ins_state)
            qv_out.extend([round(ins_qv)] * len(ins_state))
            variants.append(ConsensusVariant(
                pos=pos1, ref="", alt=ins_state, qv=ins_qv, depth=ins_depth, dissent=ins_dissent
            ))

        template_base = template_seq[ref_pos0].upper()
        call_state, call_qv, depth, dissent = _call_substitution_or_deletion(
            observations.get(ref_pos0, []), template_base
        )
        if call_state == _DELETION:
            variants.append(ConsensusVariant(
                pos=pos1, ref=template_base, alt="", qv=call_qv, depth=depth, dissent=dissent
            ))
        else:
            if call_state != template_base:
                variants.append(ConsensusVariant(
                    pos=pos1, ref=template_base, alt=call_state, qv=call_qv, depth=depth, dissent=dissent
                ))
            seq_out.append(call_state)
            qv_out.append(round(call_qv))

    trailing_pos = len(template_seq)
    ins_state, ins_qv, ins_depth, ins_dissent = _call_insertion(insertions.get(trailing_pos, []))
    if ins_state:
        seq_out.extend(ins_state)
        qv_out.extend([round(ins_qv)] * len(ins_state))
        variants.append(ConsensusVariant(
            pos=trailing_pos + 1, ref="", alt=ins_state, qv=ins_qv, depth=ins_depth, dissent=ins_dissent
        ))

    return HaplotypeConsensus(
        gene=gene,
        template_allele=template_allele,
        n_reads_assigned=len(assigned_read_ids),
        sequence="".join(seq_out),
        qv=tuple(qv_out),
        variants=tuple(variants),
        assigned_read_ids=tuple(assigned_read_ids),
    )


def build_gene_consensus(
    matrix: GeneAssignmentMatrix,
    reads: Sequence[CandidateRead],
    call: TemplateCall,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    config: OntR10Config | None = None,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
) -> tuple[HaplotypeConsensus, ...]:
    """Stage 3 entry point: partition reads across call.alleles
    (read_haplotype_responsibilities + assign_reads_to_haplotypes), then
    build one HaplotypeConsensus per haplotype copy, in call.alleles order
    (invariant 9 -- fixed order, not a set).

    **Re-aligns against call.alleles first, added after a real HG00097
    finding.** `matrix` (stages/assign.py's build_gene_matrices() output)
    was built at Stage 1's low `n_best` (that function's own docstring) --
    each read has only *one* reported hit anywhere in the gene's full
    pool, which for a template-heavy gene often isn't against either of
    `call.alleles` specifically (confirmed on real data alongside the same
    gap in stages/genotype.py's select_template_set_ont_r10(), which this
    mirrors: HLA-A's true allele had a captured span for only 3 of 26
    genuinely-matching reads). `_haplotype_pileup()` below silently skips
    any read with no captured span for its assigned template ("no
    alignment means no pileup evidence") -- on the original sparse
    `matrix` that would starve a haplotype's consensus of most of its real
    reads even where the read *partition* got the right answer. Using
    assign.realign_to_candidates() once here, up front, means both the
    partition (read_haplotype_responsibilities) and the pileup
    (build_haplotype_consensus) work from the same dense, accurate
    per-candidate alignment data."""
    read_id_set = set(matrix.read_ids)
    candidate_reads = [r for r in reads if r.read_id in read_id_set]
    allele_sequences = load_allele_sequences(pool_fasta, call.alleles)
    dense_matrix = realign_to_candidates(
        matrix.gene, candidate_reads, allele_sequences,
        n_threads=n_threads, minimap2_bin=minimap2_bin,
    )

    resp = read_haplotype_responsibilities(dense_matrix, reads, call, pool_fasta=pool_fasta, config=config)
    hard = assign_reads_to_haplotypes(resp)

    results = []
    for h, allele in enumerate(call.alleles):
        assigned_read_ids = tuple(
            rid for rid, hap in zip(dense_matrix.read_ids, hard) if hap == h
        )
        results.append(build_haplotype_consensus(
            dense_matrix.gene, dense_matrix, reads, allele, allele_sequences[allele], assigned_read_ids
        ))
    return tuple(results)


def build_phased_pair_consensus(
    gene: str,
    reads_by_hp: dict[int, Sequence[CandidateRead]],
    matrix_by_hp: dict[int, GeneAssignmentMatrix],
    allele_by_hp: dict[int, str],
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    n_threads: int = 4,
    minimap2_bin: str = "minimap2",
) -> dict[int, HaplotypeConsensus]:
    """DESIGN.md S10.4 counterpart of `build_gene_consensus()` for the
    already-known-phased case. Added 2026-08-28 (lever 4, the development notes'
    phased-pipeline performance investigation): `stages/pipeline.py`'s
    phased loop used to call `build_gene_consensus()` once per haplotype
    when both of a gene's haplotypes produced a Stage 2 call -- two
    separate small `realign_to_candidates()` minimap2 calls per gene where
    one combined call (mirroring how the *unphased* path already realigns
    once per gene for a whole `cn=2` call via `call.alleles`) would do.
    `reads_by_hp`/`allele_by_hp` (keyed by the real `HP` value, 1 or 2) give
    each haplotype's own read pool and its Stage 2-selected single allele,
    already decided independently before this runs -- template *selection*
    is inherently a per-haplotype-pool decision (each haplotype's own
    matrix only ever sees its own reads), so there is nothing to reselect
    here, only the realign+consensus step to batch.

    **`matrix_by_hp` (each haplotype's own already Stage-1-gated
    `GeneAssignmentMatrix`, e.g. via `subset_matrix_by_read_ids()`) is what
    each haplotype's own reads are restricted to before realigning --
    caught by a real-data diff, not by inspection.** A first version of
    this function used `reads_by_hp[hp]` directly (every real HP-tagged
    read for that haplotype), which skips
    `assign.build_gene_matrices()`'s own `max_read_mismatch_rate` gate that
    `build_gene_consensus()` always applies via its own `matrix.read_ids`
    restriction (`read_id_set = set(matrix.read_ids)`,
    `candidate_reads = [r for r in reads if r.read_id in read_id_set]`) --
    `realign_to_candidates()` itself does no such gating (its own
    docstring: "reads is already the caller's known-relevant set"), so
    skipping it let poor-quality reads that Stage 1 had already excluded
    back into the pileup. Confirmed as a real, not theoretical, gap: a real
    HG00097 rerun showed `n_reads_assigned` inflated across nearly every
    phased gene and two real call flips (HLA-V hap2 novel:utr -> exact,
    HLA-DRB1 hap1's consensus length/variant count both shifting) before
    this fix. Restricting to `matrix_by_hp[hp].read_ids` here reproduces
    exactly what two separate `build_gene_consensus()` calls would have
    fed into their own `realign_to_candidates()` calls.

    **Read partition here is the real, known HP split, not EM
    responsibility -- deliberately not a call into `build_gene_consensus()`
    itself.** That function partitions reads across `call.alleles` by
    fitted E-step responsibility (`read_haplotype_responsibilities`), the
    right tool when two alleles' reads are genuinely mixed and have to be
    inferred (the *unphased* case this project falls back to when no real
    HP/PS tags are usable) -- reusing it here would silently replace this
    function's whole reason to exist (a real, already-known phase split)
    with a re-derived one, and could reassign a read to the *other*
    haplotype's allele purely on likelihood even though its real
    chromosome of origin is already known and must not be second-guessed.
    Each haplotype's own `assigned_read_ids` is instead exactly its own
    gated read ids that (a) survived the combined realign at all and (b)
    have a captured span specifically against *its own* selected allele --
    identical to what two separate `build_gene_consensus(..., cn=1)` calls
    would have produced (that function's own single-candidate case
    trivially assigns every present read to the sole candidate, since a
    one-column softmax responsibility is always 1.0), just computed from
    one shared alignment pass instead of two."""
    allele_sequences = load_allele_sequences(pool_fasta, sorted(set(allele_by_hp.values())))
    reads_by_id = {r.read_id: r for hp_reads in reads_by_hp.values() for r in hp_reads}
    gated_ids_by_hp = {hp: set(matrix.read_ids) for hp, matrix in matrix_by_hp.items()}
    all_gated_reads = [
        reads_by_id[rid] for hp in sorted(matrix_by_hp) for rid in matrix_by_hp[hp].read_ids
    ]
    dense_matrix = realign_to_candidates(
        gene, all_gated_reads, allele_sequences, n_threads=n_threads, minimap2_bin=minimap2_bin,
    )
    read_index = {rid: i for i, rid in enumerate(dense_matrix.read_ids)}
    allele_index = {a: i for i, a in enumerate(dense_matrix.allele_ids)}

    results: dict[int, HaplotypeConsensus] = {}
    for hp, allele in allele_by_hp.items():
        allele_col = allele_index[allele]
        own_ids = gated_ids_by_hp[hp]
        assigned_read_ids = tuple(
            rid for rid in dense_matrix.read_ids
            if rid in own_ids and (read_index[rid], allele_col) in dense_matrix.spans
        )
        results[hp] = build_haplotype_consensus(
            gene, dense_matrix, all_gated_reads, allele, allele_sequences[allele], assigned_read_ids
        )
    return results
