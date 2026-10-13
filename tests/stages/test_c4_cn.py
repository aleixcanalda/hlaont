"""Unit tests for stages/c4_cn.py's Phase 1 total-C4-CN estimate (DESIGN.md
S2f). Reuses stages/test_cn.py's manifest/fixture conventions.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
import yaml

from hlaont.stages.assign import GeneAssignmentMatrix
from hlaont.stages.c4_cn import c4_total_depth, calibrator_genes_for_c4, estimate_c4_cn


def _write_length_meta(path: Path, rows: list[tuple[str, int]]) -> Path:
    """Minimal meta.tsv.gz -- _median_template_length() only reads the
    `gene`/`length` columns via csv.DictReader, so a full production-shaped
    file isn't needed here."""
    with gzip.open(path, "wt") as fh:
        fh.write("gene\tlength\n")
        for gene, length in rows:
            fh.write(f"{gene}\t{length}\n")
    return path


_MANIFEST = {
    "genes": [
        {"gene": "TESTA", "class": "I", "segment": None, "calibrator": "secondary"},
        {"gene": "TESTE", "class": "I", "segment": None, "calibrator": "primary"},
        {"gene": "TESTDRA", "class": "II", "segment": None, "calibrator": "primary"},
        {"gene": "C4A", "class": "III", "segment": "rccx", "calibrator": False, "typing_module": "structural"},
        {"gene": "C4B", "class": "III", "segment": "rccx", "calibrator": False, "typing_module": "structural"},
    ]
}


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "genes.yaml"
    path.write_text(yaml.safe_dump(_MANIFEST))
    return path


def _matrix(gene: str, read_ids: tuple[str, ...], rate: float = 0.0) -> GeneAssignmentMatrix:
    n = len(read_ids)
    if n == 0:
        matches = np.zeros((0, 1), dtype=np.int32)
        aln_len = np.zeros((0, 1), dtype=np.int32)
        nm = np.zeros((0, 1), dtype=np.int32)
    else:
        aln_len = np.full((n, 1), 100, dtype=np.int32)
        nm = np.full((n, 1), round(rate * 100), dtype=np.int32)
        matches = aln_len - nm
    return GeneAssignmentMatrix(
        gene=gene, read_ids=read_ids, allele_ids=(f"{gene}*asm1",),
        matches=matches, aln_block_len=aln_len, nm=nm,
    )


def _classical_matrices(depths: dict[str, int]) -> dict[str, GeneAssignmentMatrix]:
    return {
        gene: _matrix(gene, tuple(f"{gene}_r{i}" for i in range(n)))
        for gene, n in depths.items() if n > 0
    }


def test_calibrator_genes_for_c4_is_class_i_only(tmp_path):
    """Class I calibrator set, shared with pipeline's per-haplotype check."""
    manifest_path = _write_manifest(tmp_path)
    assert calibrator_genes_for_c4(manifest_path) == ["TESTA", "TESTE"]


def test_c4_total_depth_sums_both_genes_without_overlap():
    c4_matrices = {
        "C4A": _matrix("C4A", ("r0", "r1", "r2")),
        "C4B": _matrix("C4B", ("r3", "r4")),
    }
    assert c4_total_depth(c4_matrices) == 5


def test_c4_total_depth_dedupes_a_read_seen_in_both_matrices():
    # r0 has a real alignment to both C4A and C4B's pool (plausible given
    # how similar the two genes are) -- best_gene_per_read() must resolve
    # it to exactly one, not double-count it in the combined total.
    c4_matrices = {
        "C4A": _matrix("C4A", ("r0", "r1"), rate=0.01),
        "C4B": _matrix("C4B", ("r0", "r2"), rate=0.05),
    }
    assert c4_total_depth(c4_matrices) == 3  # r0, r1, r2 -- not 4


def test_estimate_c4_cn_uses_class_i_calibrator_not_class_ii(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    c4_matrices = {"C4A": _matrix("C4A", tuple(f"a{i}" for i in range(6)))}
    # class I calibrator = TESTA(6), TESTE(3) -> median 4.5; class II's
    # TESTDRA(100) must be ignored even though it's present and huge.
    classical = _classical_matrices({"TESTA": 6, "TESTE": 3, "TESTDRA": 100})

    result = estimate_c4_cn(c4_matrices, classical, genes_yaml=genes_yaml)

    assert result.gene == "C4"
    assert result.gene_class == "III"
    assert result.depth == 6
    assert result.calibrator_median == 4.5
    assert result.cn_estimate == pytest.approx(2.0 * 6 / 4.5)
    assert not result.is_zero_call
    assert result.corroborated is False


def test_estimate_c4_cn_zero_depth_is_flagged_not_hidden(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    classical = _classical_matrices({"TESTA": 6, "TESTE": 3})

    result = estimate_c4_cn({}, classical, genes_yaml=genes_yaml)

    assert result.depth == 0
    assert result.is_zero_call is True
    assert result.cn_estimate == 0.0


def test_estimate_c4_cn_no_calibrator_signal_returns_none_not_zero(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    c4_matrices = {"C4A": _matrix("C4A", ("a0", "a1"))}
    # No class I calibrator gene observed at all this run.
    classical: dict[str, GeneAssignmentMatrix] = {}

    result = estimate_c4_cn(c4_matrices, classical, genes_yaml=genes_yaml)

    assert result.depth == 2
    assert result.calibrator_median is None
    assert result.cn_estimate is None
    assert result.ci_low is None and result.ci_high is None


def test_estimate_c4_cn_is_deterministic(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    c4_matrices = {
        "C4A": _matrix("C4A", ("a0", "a1", "a2")),
        "C4B": _matrix("C4B", ("b0",)),
    }
    classical = _classical_matrices({"TESTA": 6, "TESTE": 3})

    a = estimate_c4_cn(c4_matrices, classical, genes_yaml=genes_yaml)
    b = estimate_c4_cn(c4_matrices, classical, genes_yaml=genes_yaml)
    assert a == b


def _reads_of_length(prefix: str, n: int, length: int) -> dict[str, str]:
    return {f"{prefix}{i}": "A" * length for i in range(n)}


def test_estimate_c4_cn_length_correction_pulls_estimate_down_for_a_long_target(tmp_path):
    """Long C4 target (20kb, 12kb reads: factor 2.667) vs short calibrators
    (factor ~1.1): the correction lowers cn_estimate."""
    genes_yaml = _write_manifest(tmp_path)
    c4_matrices = {"C4A": _matrix("C4A", tuple(f"c{i}" for i in range(6)))}
    classical = _classical_matrices({"TESTA": 6, "TESTE": 3})

    classical_pool_meta = _write_length_meta(tmp_path / "classical_meta.tsv.gz", [("TESTA", 3000), ("TESTE", 3200)])
    c4_pool_meta = _write_length_meta(tmp_path / "c4_meta.tsv.gz", [("C4A", 20000), ("C4B", 20000)])

    reads_by_id = {
        **_reads_of_length("TESTA_r", 6, 30000),  # TESTA's own recruited reads
        **_reads_of_length("TESTE_r", 3, 30000),  # TESTE's own recruited reads
        **_reads_of_length("c", 6, 12000),  # C4A's own recruited reads
    }

    raw_result = estimate_c4_cn(c4_matrices, classical, genes_yaml=genes_yaml)
    corrected_result = estimate_c4_cn(
        c4_matrices, classical, reads_by_id,
        genes_yaml=genes_yaml, c4_pool_meta=c4_pool_meta, classical_pool_meta=classical_pool_meta,
    )

    assert raw_result.cn_estimate == pytest.approx(2.0 * 6 / 4.5)  # unchanged baseline (2.667)
    assert corrected_result.cn_estimate is not None
    assert corrected_result.cn_estimate < raw_result.cn_estimate
    assert corrected_result.cn_estimate == pytest.approx(1.1024, abs=1e-3)

    # depth/calibrator_median stay the real, raw observed values -- only
    # the point estimate (and CI, derived from it) reflect the correction.
    assert corrected_result.depth == raw_result.depth == 6
    assert corrected_result.calibrator_median == raw_result.calibrator_median == 4.5


def test_estimate_c4_cn_length_correction_is_a_noop_without_matching_meta_rows(tmp_path):
    """Reads but no matching meta row -> factor 1.0, no crash."""
    genes_yaml = _write_manifest(tmp_path)
    c4_matrices = {"C4A": _matrix("C4A", tuple(f"c{i}" for i in range(6)))}
    classical = _classical_matrices({"TESTA": 6, "TESTE": 3})

    # Neither meta file has any rows at all -- every gene falls back to
    # factor 1.0, so this must reproduce the raw/uncorrected result exactly.
    empty_meta = _write_length_meta(tmp_path / "empty_meta.tsv.gz", [])
    reads_by_id = {
        **_reads_of_length("TESTA_r", 6, 30000),
        **_reads_of_length("TESTE_r", 3, 30000),
        **_reads_of_length("c", 6, 12000),
    }

    raw_result = estimate_c4_cn(c4_matrices, classical, genes_yaml=genes_yaml)
    result = estimate_c4_cn(
        c4_matrices, classical, reads_by_id,
        genes_yaml=genes_yaml, c4_pool_meta=empty_meta, classical_pool_meta=empty_meta,
    )
    assert result.cn_estimate == pytest.approx(raw_result.cn_estimate)


def test_estimate_c4_cn_length_correction_handles_zero_c4_depth(tmp_path):
    """reads_by_id supplied, but C4 itself has zero depth -- must not
    divide by zero computing C4's own mean read length/factor."""
    genes_yaml = _write_manifest(tmp_path)
    classical = _classical_matrices({"TESTA": 6, "TESTE": 3})
    classical_pool_meta = _write_length_meta(tmp_path / "classical_meta.tsv.gz", [("TESTA", 3000), ("TESTE", 3200)])
    reads_by_id = {**_reads_of_length("TESTA_r", 6, 30000), **_reads_of_length("TESTE_r", 3, 30000)}

    result = estimate_c4_cn(
        {}, classical, reads_by_id,
        genes_yaml=genes_yaml, classical_pool_meta=classical_pool_meta,
    )
    assert result.depth == 0
    assert result.is_zero_call is True
    assert result.cn_estimate == 0.0
