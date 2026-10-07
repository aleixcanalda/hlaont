"""Unit tests for stages/c4_form.py (A/B by exon 26, long/short by intron 9)
on synthetic matrices."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from assign import AlignmentSpan, GeneAssignmentMatrix  # noqa: E402
from c4_form import (  # noqa: E402
    C4AbCall,
    C4AlleleAnnotation,
    C4ExonSignaturePosition,
    C4FormSignature,
    classify_a_vs_b,
    classify_long_vs_short,
    estimate_c4_per_gene_cn,
)
from cn import GeneCN  # noqa: E402


def _c4_gene_cn(cn_estimate: float | None, ci_low: float | None = None, ci_high: float | None = None) -> GeneCN:
    return GeneCN(
        gene="C4", gene_class="III", depth=0, calibrator_median=44.0,
        cn_estimate=cn_estimate, ci_low=ci_low, ci_high=ci_high,
        is_zero_call=False, corroborated=False,
    )

# exon 26 occupies 0-based [20, 30) (1-based local coords (21, 30)) of a
# 40bp reference; all-'A' in the reference at the 3 discriminating offsets.
_REF_SEQ = "N" * 20 + "AAAAAAAAAA" + "N" * 10
_SIGNATURE = C4FormSignature(
    exon26_number=26, exon26_length=10,
    discriminating_positions=(
        C4ExonSignaturePosition(offset=2, a_base="A", b_base="G"),
        C4ExonSignaturePosition(offset=5, a_base="A", b_base="G"),
        C4ExonSignaturePosition(offset=8, a_base="A", b_base="G"),
    ),
    intron9_exon_before=9, intron9_long_median=100.0, intron9_short_median=20.0,
    intron9_length_threshold=60.0,
)
_ANNOTATIONS = {
    "C4A*1": C4AlleleAnnotation(gene="C4A", form="long", exon26=(21, 30), intron9=None),
}


def _matrix(gene: str, allele: str, reads: list[tuple]) -> GeneAssignmentMatrix:
    """reads: (read_id, ref_start, ref_end, read_start, read_end, cigar, nm, aln_len)."""
    read_ids = tuple(r[0] for r in reads)
    aln = np.array([[r[7]] for r in reads], dtype=np.int32)
    nm = np.array([[r[6]] for r in reads], dtype=np.int32)
    matches = aln - nm
    spans = {
        (i, 0): AlignmentSpan(ref_start=r[1], ref_end=r[2], read_start=r[3], read_end=r[4], strand="+", cigar=r[5])
        for i, r in enumerate(reads)
    }
    return GeneAssignmentMatrix(
        gene=gene, read_ids=read_ids, allele_ids=(allele,),
        matches=matches, aln_block_len=aln, nm=nm, spans=spans,
    )


def _mutate(seq: str, positions: dict[int, str]) -> str:
    chars = list(seq)
    for pos, base in positions.items():
        chars[pos] = base
    return "".join(chars)


def test_classify_a_vs_b_votes_by_majority_matching_base():
    read_a = _REF_SEQ  # identical to reference -> 'A' at all 3 discriminating positions
    read_b = _mutate(_REF_SEQ, {22: "G", 25: "G", 28: "G"})
    reads_by_id = {"ra": read_a, "rb": read_b}
    matrices = {
        "C4A": _matrix("C4A", "C4A*1", [
            ("ra", 0, 40, 0, 40, "40M", 0, 40),
            ("rb", 0, 40, 0, 40, "40M", 3, 40),
        ]),
    }
    call = classify_a_vs_b(matrices, reads_by_id, _ANNOTATIONS, _SIGNATURE, {"C4A*1": _REF_SEQ})
    assert call.n_a == 1
    assert call.n_b == 1
    assert call.n_uninformative == 0
    assert call.a_fraction == 0.5


def test_classify_a_vs_b_reports_uninformative_below_coverage_majority():
    # ref_end=23 only reaches discriminating position offset 2 (pos 22);
    # min_covered = (3+1)//2 = 2, so 1 covered position isn't enough.
    reads_by_id = {"short": _REF_SEQ[:23]}
    matrices = {
        "C4A": _matrix("C4A", "C4A*1", [("short", 0, 23, 0, 23, "23M", 0, 23)]),
    }
    call = classify_a_vs_b(matrices, reads_by_id, _ANNOTATIONS, _SIGNATURE, {"C4A*1": _REF_SEQ})
    assert (call.n_a, call.n_b, call.n_uninformative) == (0, 0, 1)
    assert call.a_fraction is None


def test_classify_a_vs_b_deletion_at_a_position_is_not_counted():
    # Delete the base at 0-based ref position 25 (offset 5): 5D removes it
    # from the read entirely -- that position must not count toward either
    # side, but the other two (offsets 2, 8) still give a 2-0 majority for A.
    read = _REF_SEQ[:25] + _REF_SEQ[30:]  # 35bp: skip the deleted 5 ref bases
    reads_by_id = {"del": read}
    matrices = {
        "C4A": _matrix("C4A", "C4A*1", [("del", 0, 40, 0, 35, "25M5D15M", 5, 40)]),
    }
    call = classify_a_vs_b(matrices, reads_by_id, _ANNOTATIONS, _SIGNATURE, {"C4A*1": _REF_SEQ})
    assert (call.n_a, call.n_b, call.n_uninformative) == (1, 0, 0)


def test_classify_a_vs_b_ignores_which_gene_bucket_won():
    # Read's winning cell is in the "C4B"-named matrix, but the winning
    # allele's own annotation is what matters, not the matrix key --
    # module docstring: the coarse gene bucket is never trusted as signal.
    annotations = {
        "C4B*weird": C4AlleleAnnotation(gene="C4B", form="long", exon26=(21, 30), intron9=None),
    }
    reads_by_id = {"ra": _REF_SEQ}
    matrices = {
        "C4B": _matrix("C4B", "C4B*weird", [("ra", 0, 40, 0, 40, "40M", 0, 40)]),
    }
    call = classify_a_vs_b(matrices, reads_by_id, annotations, _SIGNATURE, {"C4B*weird": _REF_SEQ})
    assert (call.n_a, call.n_b) == (1, 0)  # matches a_base regardless of the "C4B" label


def test_best_allele_cell_picks_lowest_mismatch_across_both_matrices():
    # Same read appears (as sieved reads sometimes do) with a real
    # alignment in both C4A's and C4B's matrices; the lower-mismatch one
    # must win regardless of which matrix it's in.
    read = _REF_SEQ
    reads_by_id = {"ra": read}
    matrices = {
        "C4A": _matrix("C4A", "C4A*worse", [("ra", 0, 40, 0, 40, "40M", 5, 40)]),
        "C4B": _matrix("C4B", "C4B*better", [("ra", 0, 40, 0, 40, "40M", 0, 40)]),
    }
    annotations = {
        "C4A*worse": C4AlleleAnnotation(gene="C4A", form="long", exon26=(21, 30), intron9=None),
        "C4B*better": C4AlleleAnnotation(gene="C4B", form="long", exon26=(21, 30), intron9=None),
    }
    sequences = {"C4A*worse": read, "C4B*better": read}
    call = classify_a_vs_b(matrices, reads_by_id, annotations, _SIGNATURE, sequences)
    assert (call.n_a, call.n_b, call.n_uninformative) == (1, 0, 0)


def test_classify_long_vs_short_requires_margin_on_both_flanks():
    long_ann = {"L*1": C4AlleleAnnotation(gene="C4A", form="long", exon26=None, intron9=(51, 150))}
    short_ann = {"S*1": C4AlleleAnnotation(gene="C4B", form="short", exon26=None, intron9=(51, 70))}

    spanning_matrix = {"C4A": _matrix("C4A", "L*1", [("spans_it", 0, 200, 0, 200, "200M", 0, 200)])}
    call = classify_long_vs_short(spanning_matrix, long_ann, margin_bp=5)
    assert (call.n_long, call.n_short, call.n_uninformative) == (1, 0, 0)

    short_matrix = {"C4B": _matrix("C4B", "S*1", [("spans_it", 0, 200, 0, 200, "200M", 0, 200)])}
    call = classify_long_vs_short(short_matrix, short_ann, margin_bp=5)
    assert (call.n_long, call.n_short, call.n_uninformative) == (0, 1, 0)

    not_spanning_matrix = {"C4A": _matrix("C4A", "L*1", [("stops_short", 0, 100, 0, 100, "100M", 0, 100)])}
    call = classify_long_vs_short(not_spanning_matrix, long_ann, margin_bp=5)
    assert (call.n_long, call.n_short, call.n_uninformative) == (0, 0, 1)
    assert call.long_fraction is None


def test_classify_long_vs_short_uninformative_with_no_form_or_intron9():
    no_form_ann = {"X*1": C4AlleleAnnotation(gene="C4A", form=None, intron9=(51, 150), exon26=None)}
    matrix = {"C4A": _matrix("C4A", "X*1", [("r", 0, 200, 0, 200, "200M", 0, 200)])}
    call = classify_long_vs_short(matrix, no_form_ann, margin_bp=5)
    assert call.n_uninformative == 1

    no_intron_ann = {"Y*1": C4AlleleAnnotation(gene="C4A", form="long", intron9=None, exon26=None)}
    matrix2 = {"C4A": _matrix("C4A", "Y*1", [("r", 0, 200, 0, 200, "200M", 0, 200)])}
    call2 = classify_long_vs_short(matrix2, no_intron_ann, margin_bp=5)
    assert call2.n_uninformative == 1


def test_estimate_c4_per_gene_cn_splits_total_by_vote_ratio():
    # HG00097 numbers: total 4.05, 13 A / 24 B (truth 2/2); checks the math.
    total = _c4_gene_cn(4.050515059272021, ci_low=2.6465380651197754, ci_high=5.454492053424267)
    ab = C4AbCall(n_a=13, n_b=24, n_uninformative=80, a_fraction=13 / 37)
    result = estimate_c4_per_gene_cn(total, ab)

    assert result.cn_a == pytest.approx(1.4231539397442237)
    assert result.cn_b == pytest.approx(2.6273611195277975)
    assert result.cn_a + result.cn_b == pytest.approx(total.cn_estimate)  # split, not independent
    # Both CIs are real ranges that contain the true 2/2, even though the
    # point estimates don't land exactly on it.
    assert result.cn_a_ci_low < 2.0 < result.cn_a_ci_high
    assert result.cn_b_ci_low < 2.0 < result.cn_b_ci_high


def test_estimate_c4_per_gene_cn_none_when_total_cn_unavailable():
    total = _c4_gene_cn(None)
    ab = C4AbCall(n_a=5, n_b=5, n_uninformative=0, a_fraction=0.5)
    result = estimate_c4_per_gene_cn(total, ab)
    assert all(
        v is None for v in (result.cn_a, result.cn_a_ci_low, result.cn_a_ci_high,
                             result.cn_b, result.cn_b_ci_low, result.cn_b_ci_high)
    )


def test_estimate_c4_per_gene_cn_none_when_no_informative_votes():
    total = _c4_gene_cn(4.0, ci_low=2.0, ci_high=6.0)
    ab = C4AbCall(n_a=0, n_b=0, n_uninformative=117, a_fraction=None)
    result = estimate_c4_per_gene_cn(total, ab)
    assert result.cn_a is None and result.cn_b is None


def test_estimate_c4_per_gene_cn_handles_unanimous_votes_without_crashing():
    # a_fraction=1.0 (all votes A) -- se_p is 0 at this boundary (a known,
    # documented simplification, not a Wilson interval), must not divide
    # by zero or produce a nonsensical negative-width CI.
    total = _c4_gene_cn(4.0, ci_low=3.0, ci_high=5.0)
    ab = C4AbCall(n_a=10, n_b=0, n_uninformative=0, a_fraction=1.0)
    result = estimate_c4_per_gene_cn(total, ab)
    assert result.cn_a == pytest.approx(4.0)
    assert result.cn_b == pytest.approx(0.0)
    assert result.cn_b_ci_low == 0.0
