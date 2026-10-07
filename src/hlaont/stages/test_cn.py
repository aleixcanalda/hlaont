"""Unit tests for Stage 1 CN estimation (DESIGN.md S1); pure Python, no
minimap2."""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from cn import estimate_cn, load_gene_manifest, stage0_depth_by_gene  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "input"))
from fastq import AnchorHit, CandidateRead  # noqa: E402

_MANIFEST = {
    "genes": [
        {"gene": "TESTA", "class": "I", "segment": None, "calibrator": "secondary",
         "cn_model": "diploid_typical"},
        {"gene": "TESTE", "class": "I", "segment": None, "calibrator": "primary"},
        # TESTH/TESTY share a segment label -- documentary metadata only
        # now (cn.py's docstring), must NOT be pooled into one CN.
        {"gene": "TESTH", "class": "I", "segment": "alpha_block_TEST", "calibrator": False},
        {"gene": "TESTY", "class": "I", "segment": "alpha_block_TEST", "calibrator": False},
        {"gene": "TESTDRA", "class": "II", "segment": None, "calibrator": "primary"},
        {"gene": "TESTNOCAL", "class": "II", "segment": None, "calibrator": False},
        {"gene": "TESTDRB3", "class": "II", "segment": None, "calibrator": False, "low_cn_floor": 0.9},
        {"gene": "C4X", "class": "III", "segment": "rccx_test", "calibrator": False, "typing_module": "structural"},
        {"gene": "LEGACY", "class": "I", "segment": None, "calibrator": False, "excluded": True},
    ]
}

# Control depth used throughout; expected CN = 2 * depth / this.
_CONTROL_DEPTH = 4.5

_ANCHOR_LEN = 500


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "genes.yaml"
    path.write_text(yaml.safe_dump(_MANIFEST))
    return path


def _write_anchors_meta(tmp_path: Path, lengths: dict[str, int]) -> Path:
    """lengths: {anchor_allele_name: length} -- one target-role row per
    entry, gene inferred as the part before "-anchor" in the allele name
    (matches this file's own _read()/_hit() naming convention below)."""
    path = tmp_path / "anchors.meta.tsv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("gene\tallele\trole\tsource\tcluster_size\tlength\n")
        for allele, length in lengths.items():
            gene = allele.split("-anchor")[0]
            fh.write(f"{gene}\t{allele}\ttarget\ttest\t1\t{length}\n")
    return path


def _write_empty_anchors_meta(tmp_path: Path) -> Path:
    path = tmp_path / "anchors.meta.tsv.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("gene\tallele\trole\tsource\tcluster_size\tlength\n")
    return path


def _hit(gene: str, allele: str, coverage: float, mismatch_rate: float = 0.0) -> AnchorHit:
    return AnchorHit(gene=gene, allele=allele, coverage=coverage, mismatch_rate=mismatch_rate)


def _read(read_id: str, hits: tuple[AnchorHit, ...]) -> CandidateRead:
    return CandidateRead(read_id=read_id, sequence="A" * 100, quality=None, hits=hits)


def _full_coverage_reads(gene: str, n: int) -> tuple[list[CandidateRead], dict[str, int]]:
    """n reads, each an exact (coverage=1.0) full-length hit against
    `{gene}-anchor` -- real aligned bases == anchor length, so depth ==
    read count exactly, the base case every test in this file builds on."""
    anchor = f"{gene}-anchor"
    reads = [_read(f"{gene}_r{i}", (_hit(gene, anchor, 1.0),)) for i in range(n)]
    return reads, {anchor: _ANCHOR_LEN}


def test_load_gene_manifest_excludes_legacy_and_structural(tmp_path):
    manifest = load_gene_manifest(_write_manifest(tmp_path))
    assert set(manifest) == {
        "TESTA", "TESTE", "TESTH", "TESTY", "TESTDRA", "TESTNOCAL", "TESTDRB3",
    }


def test_load_gene_manifest_loads_low_cn_floor(tmp_path):
    manifest = load_gene_manifest(_write_manifest(tmp_path))
    assert manifest["TESTDRB3"].low_cn_floor == 0.9
    assert manifest["TESTA"].low_cn_floor is None


def test_load_gene_manifest_loads_cn_model(tmp_path):
    manifest = load_gene_manifest(_write_manifest(tmp_path))
    assert manifest["TESTA"].cn_model == "diploid_typical"
    assert manifest["TESTE"].cn_model is None


def test_shared_segment_label_does_not_pool_depth_across_genes(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    a_reads, a_lengths = _full_coverage_reads("TESTA", 6)
    e_reads, e_lengths = _full_coverage_reads("TESTE", 3)
    h_reads, h_lengths = _full_coverage_reads("TESTH", 2)
    anchors_meta = _write_anchors_meta(tmp_path, {**a_lengths, **e_lengths, **h_lengths})
    reads = a_reads + e_reads + h_reads

    results = estimate_cn(reads, _CONTROL_DEPTH, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta)

    testh = results["TESTH"]
    assert testh.depth == pytest.approx(2.0)  # TESTH's own depth only, not summed with TESTY
    assert testh.gene_class == "I"
    assert testh.calibrator_median == _CONTROL_DEPTH  # holds control_depth (cn.py's docstring)
    assert testh.cn_estimate == pytest.approx(2.0 * 2 / _CONTROL_DEPTH)
    assert not testh.is_zero_call
    assert testh.corroborated is False  # invariant 4 -- never silently confirmed

    testy = results["TESTY"]
    assert testy.depth == 0  # TESTY's own (zero) depth, not TESTH's
    assert testy.is_zero_call is True


def test_zero_depth_gene_is_flagged_not_silently_dropped(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    a_reads, a_lengths = _full_coverage_reads("TESTA", 6)
    e_reads, e_lengths = _full_coverage_reads("TESTE", 3)
    anchors_meta = _write_anchors_meta(tmp_path, {**a_lengths, **e_lengths})
    reads = a_reads + e_reads  # TESTH unobserved

    results = estimate_cn(reads, _CONTROL_DEPTH, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta)

    testh = results["TESTH"]
    assert testh.depth == 0
    assert testh.is_zero_call is True
    assert testh.cn_estimate == 0.0
    assert testh.corroborated is False


def test_every_manifest_gene_gets_its_own_entry(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    a_reads, a_lengths = _full_coverage_reads("TESTA", 6)
    anchors_meta = _write_anchors_meta(tmp_path, a_lengths)

    results = estimate_cn(a_reads, _CONTROL_DEPTH, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta)

    assert set(results) == {
        "TESTA", "TESTE", "TESTH", "TESTY", "TESTDRA", "TESTNOCAL", "TESTDRB3",
    }


def test_no_control_signal_returns_none_not_zero(tmp_path):
    """No control depth -> cn_estimate None, not 0 (invariant 4)."""
    genes_yaml = _write_manifest(tmp_path)
    reads, lengths = _full_coverage_reads("TESTNOCAL", 5)
    anchors_meta = _write_anchors_meta(tmp_path, lengths)

    for missing_control_depth in (None, 0.0):
        results = estimate_cn(
            reads, missing_control_depth, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta,
        )
        gene = results["TESTNOCAL"]
        assert gene.depth == pytest.approx(5.0)
        assert gene.calibrator_median is None
        assert gene.cn_estimate is None
        assert gene.ci_low is None and gene.ci_high is None
        assert gene.corroborated is False


def test_no_class_split_same_control_depth_normalises_class_i_and_ii(tmp_path):
    """Class I and II genes use the same control depth."""
    genes_yaml = _write_manifest(tmp_path)
    a_reads, a_lengths = _full_coverage_reads("TESTA", 6)
    dra_reads, dra_lengths = _full_coverage_reads("TESTDRA", 6)
    anchors_meta = _write_anchors_meta(tmp_path, {**a_lengths, **dra_lengths})

    results = estimate_cn(a_reads + dra_reads, _CONTROL_DEPTH, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta)

    assert results["TESTA"].gene_class == "I"
    assert results["TESTDRA"].gene_class == "II"
    assert results["TESTA"].cn_estimate == results["TESTDRA"].cn_estimate == pytest.approx(2.0 * 6 / _CONTROL_DEPTH)


def test_c4_and_excluded_genes_produce_no_entries(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    a_reads, a_lengths = _full_coverage_reads("TESTA", 6)
    anchors_meta = _write_anchors_meta(tmp_path, a_lengths)

    results = estimate_cn(a_reads, _CONTROL_DEPTH, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta)

    assert "C4X" not in results
    assert "LEGACY" not in results


def test_estimate_cn_is_deterministic(tmp_path):
    genes_yaml = _write_manifest(tmp_path)
    a_reads, a_lengths = _full_coverage_reads("TESTA", 6)
    e_reads, e_lengths = _full_coverage_reads("TESTE", 3)
    anchors_meta = _write_anchors_meta(tmp_path, {**a_lengths, **e_lengths})
    reads = a_reads + e_reads

    a = estimate_cn(reads, _CONTROL_DEPTH, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta)
    b = estimate_cn(reads, _CONTROL_DEPTH, genes_yaml=genes_yaml, anchors_meta_path=anchors_meta)
    assert a == b


def test_stage0_depth_is_real_aligned_bases_over_mean_anchor_length(tmp_path):
    """Full-length exact hits: depth equals read count."""
    h_reads, h_lengths = _full_coverage_reads("TESTH", 5)
    y_reads, y_lengths = _full_coverage_reads("TESTY", 3)
    anchors_meta = _write_anchors_meta(tmp_path, {**h_lengths, **y_lengths})
    depth = stage0_depth_by_gene(h_reads + y_reads, anchors_meta_path=anchors_meta)
    assert depth["TESTH"] == pytest.approx(5.0)
    assert depth["TESTY"] == pytest.approx(3.0)


def test_stage0_depth_counts_ambiguous_read_fully_toward_every_gene_it_hits(tmp_path):
    """No cross-gene resolution is attempted at all (module docstring) --
    a read genuinely sieved to two genes counts fully toward both."""
    anchors_meta = _write_anchors_meta(tmp_path, {"TESTH-anchor": 500, "TESTY-anchor": 500})
    reads = [
        _read("r1", (_hit("TESTH", "TESTH-anchor", 1.0),)),
        _read("r2", (_hit("TESTH", "TESTH-anchor", 1.0), _hit("TESTY", "TESTY-anchor", 1.0))),
    ]
    depth = stage0_depth_by_gene(reads, anchors_meta_path=anchors_meta)
    assert depth["TESTH"] == pytest.approx(2.0)
    assert depth["TESTY"] == pytest.approx(1.0)


def test_stage0_depth_dedupes_within_gene_across_multiple_anchor_hits(tmp_path):
    """A read hitting two anchors of one gene counts once (best hit)."""
    anchors_meta = _write_anchors_meta(tmp_path, {"TESTH-anchor-1": 500, "TESTH-anchor-2": 500})
    hits = (
        _hit("TESTH", "TESTH-anchor-1", 1.0),
        _hit("TESTH", "TESTH-anchor-2", 0.9),  # same read, second qualifying hit, same gene
    )
    reads = [_read("r1", hits)]
    depth = stage0_depth_by_gene(reads, anchors_meta_path=anchors_meta)
    # Best hit only (coverage=1.0 * 500 = 500 bases) -- not both summed
    # (which would give 500 + 450 = 950 bases -> depth 1.9, not 1.0).
    assert depth["TESTH"] == pytest.approx(1.0)


def test_stage0_depth_pools_by_mean_not_sum_of_anchor_lengths(tmp_path):
    """Depth divides by the mean, not the sum, of the gene's anchor lengths."""
    anchors_meta = _write_anchors_meta(tmp_path, {
        "TESTH-anchor-1": 500, "TESTH-anchor-2": 500, "TESTH-anchor-3": 500,
    })
    # 4 reads, all against anchor-1 only -- anchor-2/3 never hit at all.
    reads = [_read(f"r{i}", (_hit("TESTH", "TESTH-anchor-1", 1.0),)) for i in range(4)]
    depth = stage0_depth_by_gene(reads, anchors_meta_path=anchors_meta)
    # aligned bases = 4*500=2000; mean length = 500 (not 1500) -> depth 4.0
    assert depth["TESTH"] == pytest.approx(4.0)


def test_stage0_depth_partial_coverage_scales_aligned_bases(tmp_path):
    """A read covering only half its anchor contributes half the bases --
    real aligned-bases arithmetic, not a binary keep/drop."""
    anchors_meta = _write_anchors_meta(tmp_path, {"TESTH-anchor": 500})
    reads = [_read("r1", (_hit("TESTH", "TESTH-anchor", 0.5),))]
    depth = stage0_depth_by_gene(reads, anchors_meta_path=anchors_meta)
    assert depth["TESTH"] == pytest.approx(0.5)  # 250 bases / 500 mean length


def test_stage0_depth_no_data_for_gene_missing_from_anchors_meta(tmp_path):
    """A gene without anchor lengths contributes nothing (no crash)."""
    anchors_meta = _write_anchors_meta(tmp_path, {"TESTE-anchor": 500})  # TESTH absent
    reads = [_read("r1", (_hit("TESTH", "TESTH-anchor", 1.0),))]
    depth = stage0_depth_by_gene(reads, anchors_meta_path=anchors_meta)
    assert "TESTH" not in depth


def test_stage0_depth_empty_reads_returns_empty(tmp_path):
    anchors_meta = _write_empty_anchors_meta(tmp_path)
    depth = stage0_depth_by_gene([], anchors_meta_path=anchors_meta)
    assert depth == {}
