"""Stage 3 -- consensus with per-base QV (DESIGN.md S3).

Template-and-polish: each allele of Stage 2's TemplateCall is polished with
the reads assigned to it, from their CIGAR alignments, keeping the
template's IPD coordinates (what novelty.py needs). Homozygous collapse is
the caller's decision; this module polishes whatever alleles/weights it
gets.

Per-base error is each read's aggregate mismatch rate against its template
(flat per read), a stand-in for a cohort-fitted homopolymer-aware model.
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

# Clamp on a read's mismatch rate used as a per-base error (avoids log(0)).
_MIN_READ_ERR = 1e-4
_MAX_READ_ERR = 0.5

# Clamp on P(wrong), bounding QV to 0-60.
_MIN_P_WRONG = 1e-6
_MAX_QV = 60.0

_BASES = ("A", "C", "G", "T")
_DELETION = "-"


@dataclass(frozen=True)
class ConsensusVariant:
    """A consensus-vs-template difference in the template's 1-based
    coordinates. Substitution/deletion: `pos` at the template base, `alt`
    '' for a deletion. Insertion: `ref` '', `pos` = the base it precedes.
    Not VCF-anchored."""

    pos: int
    ref: str
    alt: str
    qv: float
    depth: int
    # Reads disagreeing with the winning call (QV saturates; novelty.py
    # filters on this instead).
    dissent: int


@dataclass(frozen=True)
class HaplotypeConsensus:
    gene: str
    template_allele: str
    n_reads_assigned: int
    sequence: str
    qv: tuple[int, ...]
    variants: tuple[ConsensusVariant, ...]
    # Reads partitioned to this haplotype (for hla_typing's read tie-break).
    assigned_read_ids: tuple[str, ...] = ()


def read_haplotype_responsibilities(
    matrix: GeneAssignmentMatrix,
    reads: Sequence[CandidateRead],
    call: TemplateCall,
    *,
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    config: OntR10Config | None = None,
) -> np.ndarray:
    """(n_reads, cn) responsibilities for `call.alleles`: one E-step at
    Stage 2's converged weights."""
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
    """Hard assignment by argmax responsibility (ties -> first allele)."""
    return np.argmax(responsibilities, axis=1)


def _read_err_rate(matrix: GeneAssignmentMatrix, read_idx: int, allele_idx: int) -> float:
    rate = float(matrix.mismatch_rate()[read_idx, allele_idx])
    return min(max(rate, _MIN_READ_ERR), _MAX_READ_ERR)


def _qv_and_best(scores: dict[str, float], preferred_on_tie: str) -> tuple[str, float]:
    """argmax of log-likelihood `scores` (ties -> `preferred_on_tie`, the
    template's state) and its phred-scaled P(wrong)."""
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
    """observations: (base or '' for deletion, read_err). Returns
    (state, qv, depth, dissent); state is a base or '-'. 5-state model."""
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
    """observations: (inserted seq or '' for none, read_err) for every read
    covering this anchor. Returns (insert, qv, depth, dissent); ties go to
    '' (conservative against ONT homopolymer slips)."""
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
    """Per 0-based template position: (substitution/deletion observations,
    insertion observations) from each assigned read's captured CIGAR. Reads
    without a span are skipped."""
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

        # Record substitution/deletion and insertion observations; every
        # covered position without an insertion gets a "no insertion" vote
        # below.
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
    """Polish `template_allele` with its assigned reads: consensus, per-base
    QV and variants relative to the template."""
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
    """Stage 3 entry point: realign reads densely against call.alleles (the
    n_best=1 Stage 1 matrix lacks spans for most reads), partition them, and
    build one HaplotypeConsensus per allele in call.alleles order."""
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
    """Phased counterpart of `build_gene_consensus()`: one realignment for
    both haplotypes' selected alleles (`allele_by_hp`).

    Each haplotype's reads are restricted to `matrix_by_hp[hp].read_ids`
    (keeps Stage 1's mismatch-rate gate) and partitioned by their real HP,
    not by EM responsibility. A read is assigned to its haplotype if it has
    a span against that haplotype's allele. Equivalent to two cn=1
    build_gene_consensus() calls."""
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
