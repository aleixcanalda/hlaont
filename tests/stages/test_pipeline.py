"""End-to-end tests for stages/pipeline.py on a synthetic two-gene manifest
(a target gene and its calibrator)."""
from __future__ import annotations

import dataclasses
import gzip
import json
import random
import tempfile
from pathlib import Path

import pysam
import pytest
import yaml

from hlaont.stages import pipeline
from hlaont.stages.pipeline import run_sample, run_sample_phased

from hlaont.input.fastq import AnchorHit, CandidateRead
from hlaont.input.phasing import ReadPhase


def _write_c4_pool_files(directory: Path) -> tuple[Path, Path]:
    """Tiny synthetic C4A/C4B pool, so tests don't align against the slow
    production C4 bundle. Meta has the full production columns (c4_form.py
    reads description/ipd_id); synthetic ipd_ids match no real model, so
    Phase 2 reports every read uninformative."""
    fasta = directory / "c4_pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(">C4A*asmtest\n" + "ACGT" * 20 + "\n")
        fh.write(">C4B*asmtest\n" + "TGCA" * 20 + "\n")
    meta = directory / "c4_pool.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tipd_id\tallele\tfields\tlength\tgenomic\tsource\tsource_version\tdescription\tnearest_ipd_match\n")
        fh.write("C4A\tTESTC4A1\tC4A*asmtest\t1\t80\tTrue\ttest\t1\ttest fixture, no real form\t\n")
        fh.write("C4B\tTESTC4B1\tC4B*asmtest\t1\t80\tTrue\ttest\t1\ttest fixture, no real form\t\n")
    return fasta, meta


# Built once at import time (content is identical for every test in this
# file, and the one in stages/test_output.py that imports it) rather than
# per-test -- cheap, and avoids ~15 redundant tiny-file writes.
_TINY_C4_DIR = Path(tempfile.mkdtemp(prefix="hlaont_test_c4_"))
TINY_C4_FASTA, TINY_C4_META = _write_c4_pool_files(_TINY_C4_DIR)

# Extra C4 rows in calls.tsv; shared with test_output.py and test_cli.py to
# filter them out.
C4_EXTRA_CALLS_TSV_GENES = {"C4", "C4AL", "C4AS", "C4BL", "C4BS"}

# Injected chr17 control depth (equals the old calibrator median, CALIB1's
# 20 reads), so tests need no chr17 fixture.
_TEST_CONTROL_DEPTH = 20.0


def _random_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutate(seq: str, n_subs: int, rng: random.Random) -> str:
    bases = list(seq)
    for p in rng.sample(range(len(bases)), n_subs):
        alts = [b for b in "ACGT" if b != bases[p]]
        bases[p] = rng.choice(alts)
    return "".join(bases)


_GENE_MODEL_SHAPE = {
    "codon_start": 1,
    "cds_exons": [[101, 400], [501, 700]],
    "utrs": [[1, 100]],
    "exons": [[1, 101, 400], [2, 501, 700]],
    "introns": [[1, 401, 500]],
}


def _write_pool(tmp_path, records: dict[str, tuple[str, str]]):
    """records: allele -> (gene, 700bp sequence)."""
    fasta = tmp_path / "pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        for allele, (_, seq) in records.items():
            fh.write(f">{allele}\n{seq}\n")

    meta = tmp_path / "pool.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tallele\n")
        for allele, (gene, _) in records.items():
            fh.write(f"{gene}\t{allele}\n")

    models = tmp_path / "models.jsonl.gz"
    with gzip.open(models, "wt") as fh:
        for i, (allele, (gene, seq)) in enumerate(records.items()):
            obj = {"gene": gene, "ipd_id": f"TEST{i:05d}", "allele": allele,
                   "length": len(seq), **_GENE_MODEL_SHAPE}
            fh.write(json.dumps(obj) + "\n")

    cds_catalogue = tmp_path / "cds_catalogue.tsv.gz"
    with gzip.open(cds_catalogue, "wt") as fh:
        fh.write("gene\tipd_id\tallele\tgenomic\tfields\tcds_length\tcds_sequence\n")

    return fasta, meta, models, cds_catalogue


def _write_anchors(tmp_path, anchors: dict[str, tuple[str, str]]):
    """anchors: anchor_name -> (gene, sequence), all role=target. `length`
    must be the real sequence length or CN comes back zero."""
    fasta = tmp_path / "anchors.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        for name, (_, seq) in anchors.items():
            fh.write(f">{name}\n{seq}\n")

    meta = tmp_path / "anchors.meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("allele\tgene\trole\tlength\n")
        for name, (gene, seq) in anchors.items():
            fh.write(f"{name}\t{gene}\ttarget\t{len(seq)}\n")

    return fasta, meta


def _write_fastq(tmp_path, reads: list[tuple[str, str]]):
    path = tmp_path / "reads.fastq"
    with open(path, "w") as fh:
        for read_id, seq in reads:
            fh.write(f"@{read_id}\n{seq}\n+\n{'I' * len(seq)}\n")
    return path


def _write_chr17_control_fixture(tmp_path) -> tuple[Path, str]:
    """Tiny synthetic chr17 control reference (own seed, so it never aligns
    to GENE1/CALIB1). Only test_cli.py needs it: the CLI can't inject
    control_depth."""
    seq = "".join(random.Random(31337).choice("ACGT") for _ in range(300))
    path = tmp_path / "chr17_control_test.fa.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(f">chr17_control_test\n{seq}\n")
    return path, seq


def _write_genes_yaml(tmp_path):
    path = tmp_path / "genes.yaml"
    path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "GENE1", "class": "I", "tier": 1, "calibrator": False,
             "typing_module": "template_polish"},
            {"gene": "CALIB1", "class": "I", "tier": 1, "calibrator": "primary",
             "typing_module": "template_polish"},
        ],
    }))
    return path


def test_run_sample_end_to_end(tmp_path):
    rng = random.Random(123)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(1))
    calib_a = _random_seq(700, random.Random(55))
    calib_b = _mutate(calib_a, 5, random.Random(2))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)

    reads = (
        [(f"a{i}", allele_a) for i in range(10)]
        + [(f"b{i}", allele_b) for i in range(10)]
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    result = run_sample(
        "sample1", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    assert result.failures == ()
    assert result.sieve_stats.n_kept == 40

    by_gene: dict[str, list] = {}
    for hc in result.haplotype_calls:
        by_gene.setdefault(hc.gene, []).append(hc)
    assert set(by_gene) == {"GENE1", "CALIB1"}

    gene1_calls = by_gene["GENE1"]
    assert len(gene1_calls) == 2
    assert all(c.cn == 2 for c in gene1_calls)
    assert {c.called_allele for c in gene1_calls} == {
        "GENE1*01:01:01:01", "GENE1*01:02:01:01",
    }
    assert all(c.taxonomy == "exact" for c in gene1_calls)
    assert all(c.n_reads_assigned == 10 for c in gene1_calls)

    calib_calls = by_gene["CALIB1"]
    assert len(calib_calls) == 2
    assert all(c.cn == 2 for c in calib_calls)

    assert len(result.consensus_records) == 4
    assert len(result.novelty_calls) == 4


def test_run_sample_reports_absent_gene_without_crashing(tmp_path):
    """A gene with no reads is a recorded GeneFailure with CN 0.0, not a
    crash or a silent omission."""
    rng = random.Random(7)
    calib_a = _random_seq(700, rng)
    calib_b = _mutate(calib_a, 5, random.Random(8))
    absent_allele = _random_seq(700, random.Random(9))  # never actually read from

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", absent_allele),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", absent_allele),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)

    reads = (
        [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    result = run_sample(
        "sample2", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    gene1_failures = [f for f in result.failures if f.gene == "GENE1"]
    assert len(gene1_failures) == 1
    # Zero depth against a working calibrator computes a real CN estimate
    # of 0.0 (not "no calibrator signal") -- correctly reported as an
    # unconfirmed absence (invariant 4), not silently dropped.
    assert "rounds to 0" in gene1_failures[0].reason


def test_run_sample_caps_cn_at_max_cn(tmp_path):
    """Excess depth clamps the search-sizing cn_int to max_cn; the raw
    cn_estimate stays uncapped in gene_cn."""
    rng = random.Random(321)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(11))
    calib_a = _random_seq(700, random.Random(66))
    calib_b = _mutate(calib_a, 5, random.Random(12))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)

    # GENE1 gets 4x CALIB1's per-allele depth -> raw cn_estimate ~4.0.
    reads = (
        [(f"a{i}", allele_a) for i in range(20)]
        + [(f"b{i}", allele_b) for i in range(20)]
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    result = run_sample(
        "sample3", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    assert result.gene_cn["GENE1"].cn_estimate > 2.0
    gene1_calls = [hc for hc in result.haplotype_calls if hc.gene == "GENE1"]
    assert len(gene1_calls) == 2
    assert all(c.cn == 2 for c in gene1_calls)


def test_run_sample_collapses_weak_lrt_to_homozygous_call(tmp_path, monkeypatch):
    """A weak LRT collapses to one haplotype built from all reads. The
    fixture is heterozygous; the LRT is forced low to isolate the collapse."""
    rng = random.Random(123)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(1))
    calib_a = _random_seq(700, random.Random(55))
    calib_b = _mutate(calib_a, 5, random.Random(2))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)

    reads = (
        [(f"a{i}", allele_a) for i in range(10)]
        + [(f"b{i}", allele_b) for i in range(10)]
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    real_select = pipeline.select_template_set_ont_r10

    def _fake_select(*args, **kwargs):
        call = real_select(*args, **kwargs)
        return dataclasses.replace(call, lrt_statistic=0.5)

    monkeypatch.setattr(pipeline, "select_template_set_ont_r10", _fake_select)

    result = run_sample(
        "sample4", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    gene1_calls = [hc for hc in result.haplotype_calls if hc.gene == "GENE1"]
    assert len(gene1_calls) == 2  # one row per real copy, not one collapsed row
    assert {hc.haplotype_index for hc in gene1_calls} == {0, 1}
    for hc in gene1_calls:
        assert hc.cn == 2  # true estimated copy number, still reported
        assert hc.n_reads_assigned == 20  # every read, not split 10/10
        assert hc.called_allele == gene1_calls[0].called_allele  # both rows identical


def test_run_sample_collapses_zero_read_haplotype_despite_high_lrt(tmp_path, monkeypatch):
    """A second allele with one read (HLA-B shape) collapses despite a high
    LRT, via min_reads_per_haplotype. One read (not zero) keeps allele_b a
    Stage 1 candidate at n_best=1. LRT forced high to isolate this check."""
    rng = random.Random(321)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(1))  # same distance as the weak-LRT fixture
    calib_a = _random_seq(700, random.Random(55))
    calib_b = _mutate(calib_a, 5, random.Random(2))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)

    reads = (
        [(f"a{i}", allele_a) for i in range(20)]
        + [("b0", allele_b)]  # exactly 1 real read -- clears n_best=1's own-best-hit
        # bar so allele_b survives as a real Stage 1 candidate at all, while
        # staying far below min_reads_per_haplotype for the collapse check
        # this test targets (module docstring above).
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    real_select = pipeline.select_template_set_ont_r10

    def _fake_select(*args, **kwargs):
        call = real_select(*args, **kwargs)
        return dataclasses.replace(call, lrt_statistic=999.0)

    monkeypatch.setattr(pipeline, "select_template_set_ont_r10", _fake_select)

    result = run_sample(
        "sample5", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    gene1_calls = [hc for hc in result.haplotype_calls if hc.gene == "GENE1"]
    assert len(gene1_calls) == 2  # one row per real copy, not one collapsed row
    assert {hc.haplotype_index for hc in gene1_calls} == {0, 1}
    for hc in gene1_calls:
        assert hc.cn == 2  # true estimated copy number, still reported
        assert hc.n_reads_assigned == 21  # all 21 real reads, once collapsed to the one dominant candidate


def test_run_sample_collapses_low_ratio_haplotype_despite_high_lrt(tmp_path, monkeypatch):
    """A minor haplotype at exactly the absolute floor (3 reads) but 3/40 =
    0.075 < min_haplotype_read_ratio collapses. LRT forced high."""
    rng = random.Random(321)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(1))
    calib_a = _random_seq(700, random.Random(55))
    calib_b = _mutate(calib_a, 5, random.Random(2))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)

    reads = (
        [(f"a{i}", allele_a) for i in range(40)]
        + [(f"b{i}", allele_b) for i in range(3)]  # clears the absolute floor, fails the ratio
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    real_select = pipeline.select_template_set_ont_r10

    def _fake_select(*args, **kwargs):
        call = real_select(*args, **kwargs)
        return dataclasses.replace(call, lrt_statistic=999.0)

    monkeypatch.setattr(pipeline, "select_template_set_ont_r10", _fake_select)

    result = run_sample(
        "sample6", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    gene1_calls = [hc for hc in result.haplotype_calls if hc.gene == "GENE1"]
    assert len(gene1_calls) == 2  # one row per real copy, not one collapsed row
    assert {hc.haplotype_index for hc in gene1_calls} == {0, 1}
    for hc in gene1_calls:
        assert hc.cn == 2  # true estimated copy number, still reported
        assert hc.n_reads_assigned == 43  # every read, not split 40/3
        assert hc.called_allele == "GENE1*01:01:01:01"  # the dominant allele, homozygous


def _manifest_entry_for_raw_cn_int(
    cn_model: str | None = None, low_cn_floor: float | None = None,
) -> pipeline.GeneManifestEntry:
    return pipeline.GeneManifestEntry(
        gene="GENE1", gene_class="I", segment=None, calibrator=False,
        low_cn_floor=low_cn_floor, cn_model=cn_model,
    )


def test_raw_cn_int_diploid_typical_no_longer_forces_max_cn():
    """diploid_typical genes are no longer forced to max_cn: an estimate of
    0.8 rounds by the ordinary thresholds, so a real deletion isn't
    masked."""
    entry = _manifest_entry_for_raw_cn_int(cn_model="diploid_typical")
    assert pipeline._raw_cn_int_for_gene(entry, 0.8, pipeline._round_cn_estimate, max_cn=2) == 1


def test_raw_cn_int_diploid_typical_still_capped_at_max_cn():
    """max_cn stays a hard ceiling (invariant 11 -- an unfitted noise guard
    against a stray high estimate forcing a 3rd/4th-allele search), the one
    part of the old behaviour this revision keeps."""
    entry = _manifest_entry_for_raw_cn_int(cn_model="diploid_typical")
    assert pipeline._raw_cn_int_for_gene(entry, 5.0, pipeline._round_cn_estimate, max_cn=2) == 2


def test_raw_cn_int_low_cn_floor_still_forces_zero():
    """genes.yaml's low_cn_floor (DRB3/4/5) is untouched by this revision --
    it runs before the (now-removed) diploid_typical branch ever would
    have, and still forces CN=0 below the floor."""
    entry = _manifest_entry_for_raw_cn_int(low_cn_floor=0.9)
    assert pipeline._raw_cn_int_for_gene(entry, 0.47, pipeline._round_cn_estimate, max_cn=2) == 0


def _write_genes_yaml_variable_cn_model(tmp_path):
    path = tmp_path / "genes.yaml"
    path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "CALIB1", "class": "I", "tier": 1, "calibrator": "primary",
             "typing_module": "template_polish"},
            {"gene": "VARGENE", "class": "I", "tier": 2, "calibrator": False,
             "typing_module": "template_polish", "cn_model": "variable"},
        ],
    }))
    return path


def test_run_sample_asymmetric_cn_rounding_calls_presence_below_round_half(tmp_path):
    """cn_estimate 0.5 (5 reads vs 20 calibrator) -> CN 1 (>0.25 presence
    bar), where round() gives 0 (HLA-U at 0.41)."""
    rng = random.Random(2023)
    var_allele = _random_seq(700, rng)
    calib_a = _random_seq(700, random.Random(55))
    calib_b = _mutate(calib_a, 5, random.Random(2))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "VARGENE*01:01:01:01": ("VARGENE", var_allele),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "VARGENE-anchor": ("VARGENE", var_allele),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml_variable_cn_model(tmp_path)

    reads = (
        [(f"v{i}", var_allele) for i in range(5)]
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    result = run_sample(
        "sample8", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    assert result.gene_cn["VARGENE"].cn_estimate == pytest.approx(0.5)
    assert not any(f.gene == "VARGENE" for f in result.failures)
    var_calls = [hc for hc in result.haplotype_calls if hc.gene == "VARGENE"]
    assert len(var_calls) == 1
    assert var_calls[0].cn == 1
    assert var_calls[0].called_allele == "VARGENE*01:01:01:01"


def test_run_sample_asymmetric_cn_rounding_sizes_search_at_two_above_1_25(tmp_path):
    """cn_estimate 1.3 -> cn 2 (>1.25), where round() gives 1. Checks only
    the reported `cn`; collapse is tested elsewhere."""
    rng = random.Random(2024)
    var_allele_a = _random_seq(700, rng)
    var_allele_b = _mutate(var_allele_a, 5, random.Random(1))
    calib_a = _random_seq(700, random.Random(55))
    calib_b = _mutate(calib_a, 5, random.Random(2))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "VARGENE*01:01:01:01": ("VARGENE", var_allele_a),
        "VARGENE*01:02:01:01": ("VARGENE", var_allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "VARGENE-anchor": ("VARGENE", var_allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml_variable_cn_model(tmp_path)

    reads = (
        [(f"v{i}", var_allele_a) for i in range(12)]
        + [("v_b0", var_allele_b)]  # exactly 1 real read -- clears n_best=1's own-
        # best-hit bar so allele_b survives as a real Stage 1 candidate at all
        # (cn=2 needs >=2 candidates to form a template set), while staying far
        # below min_reads_per_haplotype for the collapse this test expects.
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    result = run_sample(
        "sample9", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    assert result.gene_cn["VARGENE"].cn_estimate == pytest.approx(1.3)
    var_calls = [hc for hc in result.haplotype_calls if hc.gene == "VARGENE"]
    assert var_calls  # the gene was searched/typed at all, not reported as a failure
    assert {hc.cn for hc in var_calls} == {2}  # search was sized at 2, not 1, per the new rule


def _write_genes_yaml_with_low_cn_floor(tmp_path):
    path = tmp_path / "genes.yaml"
    path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "CALIB1", "class": "I", "tier": 1, "calibrator": "primary",
             "typing_module": "template_polish"},
            {"gene": "DRB3TEST", "class": "I", "tier": 2, "calibrator": False,
             "typing_module": "template_polish", "low_cn_floor": 0.9},
        ],
    }))
    return path


def test_run_sample_applies_low_cn_floor(tmp_path):
    """A gene with low_cn_floor (DRB3/4/5) below its floor is CN 0, even
    though round() would give 1."""
    rng = random.Random(999)
    calib_a = _random_seq(700, rng)
    calib_b = _mutate(calib_a, 5, random.Random(1000))
    drb3_allele = _random_seq(700, random.Random(1001))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
        "DRB3TEST*01:01:01:01": ("DRB3TEST", drb3_allele),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "CALIB1-anchor": ("CALIB1", calib_a),
        "DRB3TEST-anchor": ("DRB3TEST", drb3_allele),
    })
    genes_yaml = _write_genes_yaml_with_low_cn_floor(tmp_path)

    # Calibrator depth 20 (10+10) -> median 20; DRB3TEST depth 7 -> raw
    # cn_estimate = 2*7/20 = 0.7 -- rounds to 1 under plain round(), but
    # must be forced to CN=0 by low_cn_floor=0.9.
    reads = (
        [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
        + [(f"d{i}", drb3_allele) for i in range(7)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    result = run_sample(
        "sample5", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    assert result.gene_cn["DRB3TEST"].cn_estimate == pytest.approx(0.7, abs=0.05)
    drb3_failures = [f for f in result.failures if f.gene == "DRB3TEST"]
    assert len(drb3_failures) == 1
    assert "rounds to 0" in drb3_failures[0].reason
    assert not any(hc.gene == "DRB3TEST" for hc in result.haplotype_calls)


def _drb_gene_cn(gene: str, cn_estimate: float | None) -> pipeline.GeneCN:
    return pipeline.GeneCN(
        gene=gene, gene_class="II", depth=0, calibrator_median=44.0,
        cn_estimate=cn_estimate, ci_low=None, ci_high=None,
        is_zero_call=False, corroborated=False,
    )


def test_drb345_mutual_exclusivity_pulls_ambiguous_gene_down_to_one():
    """HG00097 shape: DRB3 at 2 (raw 1.49) and DRB5 at 1 (raw 1.02) sum to 3;
    DRB3 is pulled down to 1."""
    cn_int = {"HLA-DRB3": 2, "HLA-DRB4": 0, "HLA-DRB5": 1}
    gene_cn = {
        "HLA-DRB3": _drb_gene_cn("HLA-DRB3", 1.49),
        "HLA-DRB4": _drb_gene_cn("HLA-DRB4", 0.13),
        "HLA-DRB5": _drb_gene_cn("HLA-DRB5", 1.02),
    }
    out = pipeline._resolve_drb345_mutual_exclusivity(cn_int, gene_cn)
    assert out == {"HLA-DRB3": 1, "HLA-DRB4": 0, "HLA-DRB5": 1}


def test_drb345_mutual_exclusivity_leaves_non_conflicting_sum_alone():
    """Sum already <= 2 (the ordinary case) -- no gene should be touched,
    even though DRB3 alone is in the same (1.2, 2.0) ambiguous band."""
    cn_int = {"HLA-DRB3": 2, "HLA-DRB4": 0, "HLA-DRB5": 0}
    gene_cn = {
        "HLA-DRB3": _drb_gene_cn("HLA-DRB3", 1.8),
        "HLA-DRB4": _drb_gene_cn("HLA-DRB4", 0.1),
        "HLA-DRB5": _drb_gene_cn("HLA-DRB5", 0.05),
    }
    out = pipeline._resolve_drb345_mutual_exclusivity(cn_int, gene_cn)
    assert out == cn_int


def test_drb345_mutual_exclusivity_requires_a_confident_partner():
    """Two genes at 2 with no confident 1 partner: neither is changed."""
    cn_int = {"HLA-DRB3": 2, "HLA-DRB4": 0, "HLA-DRB5": 2}
    gene_cn = {
        "HLA-DRB3": _drb_gene_cn("HLA-DRB3", 1.4),
        "HLA-DRB4": _drb_gene_cn("HLA-DRB4", 0.1),
        "HLA-DRB5": _drb_gene_cn("HLA-DRB5", 1.6),
    }
    out = pipeline._resolve_drb345_mutual_exclusivity(cn_int, gene_cn)
    assert out == cn_int


def test_drb345_mutual_exclusivity_ignores_genes_outside_the_ambiguous_band():
    """A gene at cn_int=2 with a raw estimate >= 2.0 (confidently two
    copies on its own, not the borderline-rounding case the rule targets)
    must not be pulled down just because a partner is at 1."""
    cn_int = {"HLA-DRB3": 2, "HLA-DRB4": 0, "HLA-DRB5": 1}
    gene_cn = {
        "HLA-DRB3": _drb_gene_cn("HLA-DRB3", 2.4),
        "HLA-DRB4": _drb_gene_cn("HLA-DRB4", 0.1),
        "HLA-DRB5": _drb_gene_cn("HLA-DRB5", 1.02),
    }
    out = pipeline._resolve_drb345_mutual_exclusivity(cn_int, gene_cn)
    assert out == cn_int


# ---------------------------------------------------------------------------
# DESIGN.md S10: phased pipeline (run_sample_phased)
# ---------------------------------------------------------------------------

def _write_mini_bam_with_phase_tags(tmp_path: Path, tags: dict[str, tuple[int, int]]) -> Path:
    """Synthetic mini-BAM carrying only read-name HP/PS tags (coordinates
    are irrelevant to run_sample_phased())."""
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr6", "LN": 1000}]}
    bam_path = tmp_path / "phased_mini.bam"
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bf:
        for i, (name, (hp, ps)) in enumerate(sorted(tags.items())):
            a = pysam.AlignedSegment(header=bf.header)
            a.query_name = name
            a.query_sequence = "ACGT" * 10
            a.query_qualities = pysam.qualitystring_to_array("I" * 40)
            a.flag = 0
            a.reference_id = 0
            a.reference_start = i
            a.mapping_quality = 60
            a.cigartuples = [(0, 40)]
            a.set_tags([("HP", hp), ("PS", ps)])
            bf.write(a)
    return bam_path


def test_run_sample_phased_splits_cleanly_phased_gene_by_real_haplotype(tmp_path):
    """GENE1 (clean HP split, one PS) gets two calls labelled by HP; CALIB1
    (untagged) falls back to the unphased path."""
    rng = random.Random(123)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(1))
    calib_a = _random_seq(700, random.Random(55))
    calib_b = _mutate(calib_a, 5, random.Random(2))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)

    reads = (
        [(f"a{i}", allele_a) for i in range(10)]
        + [(f"b{i}", allele_b) for i in range(10)]
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    phase_tags = {f"a{i}": (1, 100) for i in range(10)}
    phase_tags.update({f"b{i}": (2, 100) for i in range(10)})
    # ca*/cb* (CALIB1) deliberately get no entry at all -- untagged/absent
    # from the mini-BAM, same as a real read that never made it through
    # phasing.
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sample1", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    assert result.phasing_routing is not None
    gene1_routing = result.phasing_routing["GENE1"]
    assert gene1_routing.routing == "phased"
    assert gene1_routing.ps_values_seen == (100,)
    assert gene1_routing.n_reads_hp1 == 10
    assert gene1_routing.n_reads_hp2 == 10
    assert gene1_routing.n_reads_untagged == 0
    assert sorted(gene1_routing.haplotypes_called) == [1, 2]

    calib1_routing = result.phasing_routing["CALIB1"]
    assert calib1_routing.routing == "unphased_fallback"
    assert "no HP-tagged reads" in calib1_routing.reason
    assert calib1_routing.haplotypes_called == ()

    by_gene: dict[str, list] = {}
    for hc in result.haplotype_calls:
        by_gene.setdefault(hc.gene, []).append(hc)

    gene1_calls = by_gene["GENE1"]
    assert len(gene1_calls) == 2
    assert {c.haplotype_index for c in gene1_calls} == {1, 2}
    assert {c.called_allele for c in gene1_calls} == {
        "GENE1*01:01:01:01", "GENE1*01:02:01:01",
    }

    # CALIB1 came from the untouched unphased fallback -- still two calls,
    # still with the pre-existing arbitrary 0/1 haplotype_index scheme.
    calib_calls = by_gene["CALIB1"]
    assert len(calib_calls) == 2
    assert {c.haplotype_index for c in calib_calls} == {0, 1}

    assert len(result.consensus_records) == 4
    assert len(result.novelty_calls) == 4


def test_run_sample_phased_conflicting_ps_blocks_falls_back(tmp_path):
    """A gene whose recruited reads span two distinct PS values cannot be
    cleanly split by HP (DESIGN.md S10.3) -- must fall back, not guess."""
    rng = random.Random(7)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(9))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
    })
    genes_yaml_path = tmp_path / "genes.yaml"
    genes_yaml_path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "GENE1", "class": "I", "tier": 1, "calibrator": False,
             "typing_module": "template_polish"},
        ],
    }))

    reads = [(f"a{i}", allele_a) for i in range(10)] + [(f"b{i}", allele_b) for i in range(10)]
    fastq_path = _write_fastq(tmp_path, reads)

    # Same HP labels, but split across two different PS blocks -- a real
    # phase-block-boundary conflict, not a clean two-haplotype split.
    phase_tags = {f"a{i}": (1, 100) for i in range(5)}
    phase_tags.update({f"a{i}": (1, 200) for i in range(5, 10)})
    phase_tags.update({f"b{i}": (2, 100) for i in range(10)})
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sample1", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml_path,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    routing = result.phasing_routing["GENE1"]
    assert routing.routing == "unphased_fallback"
    assert "distinct PS phase" in routing.reason
    assert routing.ps_values_seen == (100, 200)
    assert routing.haplotypes_called == ()


def test_run_sample_phased_low_tagged_fraction_falls_back(tmp_path):
    """HP tags in one PS block but too few reads tagged (HLA-C: 11/72) ->
    unphased fallback via min_phased_tagged_fraction."""
    rng = random.Random(42)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(43))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
    })
    genes_yaml_path = tmp_path / "genes.yaml"
    genes_yaml_path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "GENE1", "class": "I", "tier": 1, "calibrator": False,
             "typing_module": "template_polish"},
        ],
    }))

    reads = [(f"a{i}", allele_a) for i in range(20)] + [(f"b{i}", allele_b) for i in range(20)]
    fastq_path = _write_fastq(tmp_path, reads)

    # Only 2 of each allele's 20 reads carry a tag -- 4/40 = 10% tagged,
    # well below the 50% floor, even though the tagged ones agree on one PS.
    phase_tags = {"a0": (1, 100), "a1": (1, 100), "b0": (2, 100), "b1": (2, 100)}
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sample1", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml_path,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    routing = result.phasing_routing["GENE1"]
    assert routing.routing == "unphased_fallback"
    assert "usable HP tag" in routing.reason
    assert routing.n_reads_hp1 == 2
    assert routing.n_reads_hp2 == 2
    assert routing.haplotypes_called == ()


def test_run_sample_phased_thin_haplotype_skipped_not_called(tmp_path):
    """A haplotype below min_reads_per_haplotype is a failure, while its
    well-supported sibling is still called."""
    rng = random.Random(55)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(56))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
    })
    genes_yaml_path = tmp_path / "genes.yaml"
    genes_yaml_path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "GENE1", "class": "I", "tier": 1, "calibrator": False,
             "typing_module": "template_polish"},
        ],
    }))

    # hap2 has 2 reads (< 3); all tagged, so only the per-haplotype floor
    # applies.
    reads = [(f"a{i}", allele_a) for i in range(10)] + [(f"b{i}", allele_b) for i in range(2)]
    fastq_path = _write_fastq(tmp_path, reads)
    phase_tags = {f"a{i}": (1, 100) for i in range(10)}
    phase_tags.update({f"b{i}": (2, 100) for i in range(2)})
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sample1", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml_path,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    routing = result.phasing_routing["GENE1"]
    assert routing.routing == "phased"
    assert routing.haplotypes_called == (1,)

    gene1_calls = [c for c in result.haplotype_calls if c.gene == "GENE1"]
    assert len(gene1_calls) == 1
    assert gene1_calls[0].haplotype_index == 1

    assert any(f.gene == "GENE1#hp2" and "too little evidence" in f.reason for f in result.failures)


# ---------------------------------------------------------------------------
# _decide_phase_routing
# ---------------------------------------------------------------------------

def _fake_read(read_id: str, gene: str = "GENE1") -> CandidateRead:
    return CandidateRead(
        read_id=read_id, sequence="ACGT", quality=None,
        hits=(AnchorHit(gene=gene, allele=f"{gene}*test", coverage=1.0, mismatch_rate=0.0),),
    )


def test_decide_phase_routing_no_reads_falls_back():
    routing = pipeline._decide_phase_routing(
        "GENE1", [], lambda r: ReadPhase(hp=None, ps=None), 0.5,
    )
    assert routing.routing == "unphased_fallback"
    assert routing.reason == "no reads recruited to this gene"


def test_decide_phase_routing_no_tagged_reads_falls_back():
    reads = [_fake_read(f"r{i}") for i in range(5)]
    routing = pipeline._decide_phase_routing(
        "GENE1", reads, lambda r: ReadPhase(hp=None, ps=None), 0.5,
    )
    assert routing.routing == "unphased_fallback"
    assert "no HP-tagged reads" in routing.reason


def test_decide_phase_routing_ps_conflict_falls_back():
    reads = [_fake_read(f"r{i}") for i in range(4)]
    phase_map = {
        "r0": ReadPhase(hp=1, ps=100), "r1": ReadPhase(hp=2, ps=100),
        "r2": ReadPhase(hp=1, ps=200), "r3": ReadPhase(hp=2, ps=200),
    }
    routing = pipeline._decide_phase_routing(
        "GENE1", reads, lambda r: phase_map[r.read_id], 0.5,
    )
    assert routing.routing == "unphased_fallback"
    assert "distinct PS phase" in routing.reason
    assert routing.ps_values_seen == (100, 200)


def test_decide_phase_routing_low_tagged_fraction_falls_back():
    reads = [_fake_read(f"r{i}") for i in range(10)]
    phase_map = {
        f"r{i}": (ReadPhase(hp=1, ps=100) if i < 2 else ReadPhase(hp=None, ps=None))
        for i in range(10)
    }
    routing = pipeline._decide_phase_routing(
        "GENE1", reads, lambda r: phase_map[r.read_id], 0.5,
    )
    assert routing.routing == "unphased_fallback"
    assert "usable HP tag" in routing.reason
    assert routing.n_reads_hp1 == 2
    assert routing.n_reads_hp2 == 0
    assert routing.n_reads_untagged == 8


def test_decide_phase_routing_clean_split_routes_phased():
    reads = [_fake_read(f"r{i}") for i in range(10)]
    phase_map = {f"r{i}": ReadPhase(hp=1 if i < 5 else 2, ps=100) for i in range(10)}
    routing = pipeline._decide_phase_routing(
        "GENE1", reads, lambda r: phase_map[r.read_id], 0.5,
    )
    assert routing.routing == "phased"
    assert routing.n_reads_hp1 == 5
    assert routing.n_reads_hp2 == 5
    assert routing.ps_values_seen == (100,)
    assert routing.haplotypes_called == ()  # filled in by the caller afterward


def _fake_routing(gene: str, routing: str, ps_values_seen: tuple[int, ...]) -> "pipeline.GenePhaseRouting":
    return pipeline.GenePhaseRouting(
        gene=gene, routing=routing, reason="test", ps_values_seen=ps_values_seen,
        n_reads_hp1=0, n_reads_hp2=0, n_reads_untagged=0, haplotypes_called=(),
    )


# ---------------------------------------------------------------------------
# _c4_calibrator_reads_for_haplotype
# ---------------------------------------------------------------------------

def test_c4_calibrator_reads_for_haplotype_matches_shared_ps_block():
    calib_reads = [_fake_read(f"c{i}", gene="CALIB1") for i in range(4)]
    reads_by_gene = {"CALIB1": calib_reads}
    phase_map = {f"c{i}": ReadPhase(hp=1 if i < 2 else 2, ps=100) for i in range(4)}
    routing = {"CALIB1": _fake_routing("CALIB1", "phased", (100,))}
    result = pipeline._c4_calibrator_reads_for_haplotype(
        1, (100,), ["CALIB1"], routing, reads_by_gene, lambda r: phase_map[r.read_id],
    )
    assert {r.read_id for r in result} == {"c0", "c1"}


def test_c4_calibrator_reads_for_haplotype_excludes_ps_mismatch():
    """S10.3: a calibrator gene phased under a *different* PS block than
    C4's own can't be trusted as evidence about the same chromosome."""
    calib_reads = [_fake_read(f"c{i}", gene="CALIB1") for i in range(2)]
    reads_by_gene = {"CALIB1": calib_reads}
    phase_map = {f"c{i}": ReadPhase(hp=1, ps=200) for i in range(2)}
    routing = {"CALIB1": _fake_routing("CALIB1", "phased", (200,))}
    result = pipeline._c4_calibrator_reads_for_haplotype(
        1, (100,), ["CALIB1"], routing, reads_by_gene, lambda r: phase_map[r.read_id],
    )
    assert result == []


def test_c4_calibrator_reads_for_haplotype_excludes_unphased_gene():
    calib_reads = [_fake_read(f"c{i}", gene="CALIB1") for i in range(2)]
    reads_by_gene = {"CALIB1": calib_reads}
    phase_map = {f"c{i}": ReadPhase(hp=1, ps=100) for i in range(2)}
    routing = {"CALIB1": _fake_routing("CALIB1", "unphased_fallback", (100,))}
    result = pipeline._c4_calibrator_reads_for_haplotype(
        1, (100,), ["CALIB1"], routing, reads_by_gene, lambda r: phase_map[r.read_id],
    )
    assert result == []


def test_c4_calibrator_reads_for_haplotype_no_c4_ps_block_returns_empty():
    """C4's own routing carrying zero or multiple PS values (should not
    happen once it's already "phased", but checked defensively) means
    nothing can ever match, by construction -- not a special case."""
    calib_reads = [_fake_read("c0", gene="CALIB1")]
    reads_by_gene = {"CALIB1": calib_reads}
    routing = {"CALIB1": _fake_routing("CALIB1", "phased", (100,))}
    result = pipeline._c4_calibrator_reads_for_haplotype(
        1, (), ["CALIB1"], routing, reads_by_gene, lambda r: ReadPhase(hp=1, ps=100),
    )
    assert result == []


# ---------------------------------------------------------------------------
# DESIGN.md S10.6: C4A/C4B phasing (Phase 2 -- A-vs-B/long-vs-short form)
# ---------------------------------------------------------------------------

def test_run_sample_phased_c4_routes_and_runs_form_classifiers(tmp_path):
    """C4 reads split by HP route "phased" and get per-haplotype form calls
    (all uninformative with the synthetic pool); checks the wiring only."""
    rng = random.Random(9)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(10))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
    })
    # 500bp local C4 pool: minimap2 reports nothing under ~350bp with these
    # scoring args, so the shared 80bp pool would never align.
    c4a_seq = _random_seq(500, random.Random(20))
    c4b_seq = _random_seq(500, random.Random(21))
    c4_pool_dir = tmp_path / "c4_pool_long"
    c4_pool_dir.mkdir()
    local_c4_fasta = c4_pool_dir / "c4_pool.fa.gz"
    with gzip.open(local_c4_fasta, "wt") as fh:
        fh.write(f">C4A*asmtest\n{c4a_seq}\n")
        fh.write(f">C4B*asmtest\n{c4b_seq}\n")
    local_c4_meta = c4_pool_dir / "c4_pool.meta.tsv.gz"
    with gzip.open(local_c4_meta, "wt") as fh:
        fh.write("gene\tipd_id\tallele\tfields\tlength\tgenomic\tsource\tsource_version\tdescription\tnearest_ipd_match\n")
        fh.write(f"C4A\tTESTC4A1\tC4A*asmtest\t1\t{len(c4a_seq)}\tTrue\ttest\t1\ttest fixture, no real form\t\n")
        fh.write(f"C4B\tTESTC4B1\tC4B*asmtest\t1\t{len(c4b_seq)}\tTrue\ttest\t1\ttest fixture, no real form\t\n")
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "C4A-anchor": ("C4A", c4a_seq),
        "C4B-anchor": ("C4B", c4b_seq),
    })
    genes_yaml_path = tmp_path / "genes.yaml"
    genes_yaml_path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "GENE1", "class": "I", "tier": 1, "calibrator": False,
             "typing_module": "template_polish"},
        ],
    }))

    reads = (
        [(f"a{i}", allele_a) for i in range(10)]
        + [(f"b{i}", allele_b) for i in range(10)]
        + [(f"c4a{i}", c4a_seq) for i in range(10)]
        + [(f"c4b{i}", c4b_seq) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    phase_tags = {f"a{i}": (1, 100) for i in range(10)}
    phase_tags.update({f"b{i}": (2, 100) for i in range(10)})
    phase_tags.update({f"c4a{i}": (1, 100) for i in range(10)})
    phase_tags.update({f"c4b{i}": (2, 100) for i in range(10)})
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sample1", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml_path,
        c4_pool_fasta=local_c4_fasta, c4_pool_meta=local_c4_meta, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    c4_routing = result.phasing_routing["C4"]
    assert c4_routing.routing == "phased"
    assert sorted(c4_routing.haplotypes_called) == [1, 2]
    assert c4_routing.n_reads_hp1 == 10
    assert c4_routing.n_reads_hp2 == 10

    assert result.c4_phased_ab_calls is not None
    assert set(result.c4_phased_ab_calls) == {1, 2}
    for hp in (1, 2):
        ab = result.c4_phased_ab_calls[hp]
        assert ab.n_uninformative == 10
        assert ab.n_a == ab.n_b == 0
    assert result.c4_phased_form_calls is not None
    for hp in (1, 2):
        form = result.c4_phased_form_calls[hp]
        assert form.n_uninformative == 10


def test_run_sample_phased_c4_per_haplotype_cn_when_calibrator_shares_ps_block(tmp_path):
    """CALIB1 phased in C4's PS block -> per-haplotype C4 cn_estimate."""
    calib_a = _random_seq(700, random.Random(30))
    calib_b = _mutate(calib_a, 5, random.Random(31))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    c4a_seq = _random_seq(500, random.Random(32))
    c4b_seq = _random_seq(500, random.Random(33))
    c4_pool_dir = tmp_path / "c4_pool_long"
    c4_pool_dir.mkdir()
    local_c4_fasta = c4_pool_dir / "c4_pool.fa.gz"
    with gzip.open(local_c4_fasta, "wt") as fh:
        fh.write(f">C4A*asmtest\n{c4a_seq}\n")
        fh.write(f">C4B*asmtest\n{c4b_seq}\n")
    local_c4_meta = c4_pool_dir / "c4_pool.meta.tsv.gz"
    with gzip.open(local_c4_meta, "wt") as fh:
        fh.write("gene\tipd_id\tallele\tfields\tlength\tgenomic\tsource\tsource_version\tdescription\tnearest_ipd_match\n")
        fh.write(f"C4A\tTESTC4A1\tC4A*asmtest\t1\t{len(c4a_seq)}\tTrue\ttest\t1\ttest fixture, no real form\t\n")
        fh.write(f"C4B\tTESTC4B1\tC4B*asmtest\t1\t{len(c4b_seq)}\tTrue\ttest\t1\ttest fixture, no real form\t\n")
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "CALIB1-anchor": ("CALIB1", calib_a),
        "C4A-anchor": ("C4A", c4a_seq),
        "C4B-anchor": ("C4B", c4b_seq),
    })
    genes_yaml_path = tmp_path / "genes.yaml"
    genes_yaml_path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "CALIB1", "class": "I", "tier": 1, "calibrator": "primary",
             "typing_module": "template_polish"},
        ],
    }))

    reads = (
        [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
        + [(f"c4a{i}", c4a_seq) for i in range(10)]
        + [(f"c4b{i}", c4b_seq) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    # CALIB1's own reads share C4's exact PS value (100) -- the real,
    # checked condition _c4_calibrator_reads_for_haplotype() requires.
    phase_tags = {f"ca{i}": (1, 100) for i in range(10)}
    phase_tags.update({f"cb{i}": (2, 100) for i in range(10)})
    phase_tags.update({f"c4a{i}": (1, 100) for i in range(10)})
    phase_tags.update({f"c4b{i}": (2, 100) for i in range(10)})
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sample1", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml_path,
        c4_pool_fasta=local_c4_fasta, c4_pool_meta=local_c4_meta, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    assert result.phasing_routing["CALIB1"].routing == "phased"
    assert result.phasing_routing["CALIB1"].ps_values_seen == (100,)
    assert result.phasing_routing["C4"].ps_values_seen == (100,)

    assert result.c4_phased_cn is not None
    assert set(result.c4_phased_cn) == {1, 2}
    for hp in (1, 2):
        hp_cn = result.c4_phased_cn[hp]
        assert hp_cn.depth == 10
        assert hp_cn.cn_estimate is not None
        assert hp_cn.cn_estimate > 0
    assert result.c4_phased_per_gene_cn is not None
    assert set(result.c4_phased_per_gene_cn) == {1, 2}


def test_run_sample_phased_c4_per_haplotype_cn_none_when_calibrator_ps_mismatched(tmp_path):
    """CALIB1 phased in a different PS block -> per-haplotype cn_estimate
    None, depth still reported."""
    calib_a = _random_seq(700, random.Random(40))
    calib_b = _mutate(calib_a, 5, random.Random(41))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    c4a_seq = _random_seq(500, random.Random(42))
    c4b_seq = _random_seq(500, random.Random(43))
    c4_pool_dir = tmp_path / "c4_pool_long"
    c4_pool_dir.mkdir()
    local_c4_fasta = c4_pool_dir / "c4_pool.fa.gz"
    with gzip.open(local_c4_fasta, "wt") as fh:
        fh.write(f">C4A*asmtest\n{c4a_seq}\n")
        fh.write(f">C4B*asmtest\n{c4b_seq}\n")
    local_c4_meta = c4_pool_dir / "c4_pool.meta.tsv.gz"
    with gzip.open(local_c4_meta, "wt") as fh:
        fh.write("gene\tipd_id\tallele\tfields\tlength\tgenomic\tsource\tsource_version\tdescription\tnearest_ipd_match\n")
        fh.write(f"C4A\tTESTC4A1\tC4A*asmtest\t1\t{len(c4a_seq)}\tTrue\ttest\t1\ttest fixture, no real form\t\n")
        fh.write(f"C4B\tTESTC4B1\tC4B*asmtest\t1\t{len(c4b_seq)}\tTrue\ttest\t1\ttest fixture, no real form\t\n")
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "CALIB1-anchor": ("CALIB1", calib_a),
        "C4A-anchor": ("C4A", c4a_seq),
        "C4B-anchor": ("C4B", c4b_seq),
    })
    genes_yaml_path = tmp_path / "genes.yaml"
    genes_yaml_path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "CALIB1", "class": "I", "tier": 1, "calibrator": "primary",
             "typing_module": "template_polish"},
        ],
    }))

    reads = (
        [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
        + [(f"c4a{i}", c4a_seq) for i in range(10)]
        + [(f"c4b{i}", c4b_seq) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    # CALIB1 phased under PS=999 -- a different block than C4's own PS=100.
    phase_tags = {f"ca{i}": (1, 999) for i in range(10)}
    phase_tags.update({f"cb{i}": (2, 999) for i in range(10)})
    phase_tags.update({f"c4a{i}": (1, 100) for i in range(10)})
    phase_tags.update({f"c4b{i}": (2, 100) for i in range(10)})
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sample1", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml_path,
        c4_pool_fasta=local_c4_fasta, c4_pool_meta=local_c4_meta, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    assert result.phasing_routing["CALIB1"].ps_values_seen == (999,)
    assert result.phasing_routing["C4"].ps_values_seen == (100,)

    assert result.c4_phased_cn is not None
    for hp in (1, 2):
        hp_cn = result.c4_phased_cn[hp]
        assert hp_cn.depth == 10  # real reads still counted
        assert hp_cn.cn_estimate is None  # but no cross-PS calibrator borrowed


def test_run_sample_phased_c4_falls_back_when_no_reads_recruited(tmp_path):
    """No C4 reads: "C4" still gets a routing entry."""
    rng = random.Random(11)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(12))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
    })
    genes_yaml_path = tmp_path / "genes.yaml"
    genes_yaml_path.write_text(yaml.safe_dump({
        "manifest_version": 1,
        "genes": [
            {"gene": "GENE1", "class": "I", "tier": 1, "calibrator": False,
             "typing_module": "template_polish"},
        ],
    }))

    reads = [(f"a{i}", allele_a) for i in range(10)] + [(f"b{i}", allele_b) for i in range(10)]
    fastq_path = _write_fastq(tmp_path, reads)
    phase_tags = {f"a{i}": (1, 100) for i in range(10)}
    phase_tags.update({f"b{i}": (2, 100) for i in range(10)})
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sample1", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml_path,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )

    c4_routing = result.phasing_routing["C4"]
    assert c4_routing.routing == "unphased_fallback"
    assert c4_routing.reason == "no reads recruited to C4A/C4B"
    assert result.c4_phased_ab_calls is None
    assert result.c4_phased_form_calls is None
