"""End-to-end test for the orchestration pipeline (stages/pipeline.py):
FASTQ -> sieve -> assign -> cn -> genotype -> consensus -> hla_typing ->
novelty, over a small fully-synthetic two-gene manifest (a target gene and
its calibrator) so it needs neither the real 173-anchor set nor the real
23,671-allele IPD pool.
"""
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

import pipeline
from pipeline import run_sample, run_sample_phased

from fastq import AnchorHit, CandidateRead  # noqa: E402 -- pipeline's own sys.path.insert
from phasing import ReadPhase  # noqa: E402


def _write_c4_pool_files(directory: Path) -> tuple[Path, Path]:
    """Tiny synthetic C4A/C4B pool (stages/c4_cn.py, DESIGN.md S2f) --
    run_sample() now always builds a second, separate matrix against a C4
    pool by default; every test in this file must not silently hit the
    real, much larger production resources/templates/assembly_bundle/
    c4.fa.gz just because it only cares about GENE1/CALIB1 -- confirmed
    directly that minimap2 is real slow on that pool's real ~20kb x
    362-sequence, highly repetitive RCCX content, not a guess.

    2026-08-26 Phase 2 addition: the meta.tsv now carries the full
    production column set (stages/c4_form.py's load_c4_allele_annotations()
    reads `description`/`ipd_id` via csv.DictReader -- a 2-column
    gene/allele-only file would KeyError). `c4_models_path` is left at its
    real production default in every test call site (not overridden here):
    these synthetic ipd_ids never match a real one in that file, so every
    annotation legitimately comes back with exon26/intron9 both None
    (correctly reported as uninformative by both Phase 2 classifiers, not a
    crash) -- loading that ~362-line JSON file costs nothing like the
    minimap2-against-362-real-sequences cost this fixture exists to avoid.
    """
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

# The extra calls.tsv row labels output.py's write_calls_tsv() can emit for
# C4 (2026-08-26/27, user-requested, revised repeatedly -- see that
# function's own docstring) -- shared by this file, test_output.py, and
# test_cli.py so all three filter them out identically when comparing
# against len(result.haplotype_calls). 2026-09-01: "C4A"/"C4B" (the
# proportional vote-ratio split) dropped from calls.tsv entirely,
# superseded by the directly-measured "C4AL"/"C4AS"/"C4BL"/"C4BS" rows --
# see write_calls_tsv()'s own docstring for the full rationale.
C4_EXTRA_CALLS_TSV_GENES = {"C4", "C4AL", "C4AS", "C4BL", "C4BS"}

# 2026-09-01 CN-normalisation revision (stages/cn.py's own docstring):
# every run_sample()/run_sample_phased() call in this file injects
# control_depth directly rather than aligning against a real chr17 control
# reference -- these tests exercise unrelated pipeline behaviour (LRT
# collapse, phasing, CN rounding thresholds, ...) and would otherwise need
# a real, matching synthetic chr17 fixture just to get a nonzero control
# signal. 20.0 reproduces exactly what every one of this file's fixtures'
# old calibrator-panel median was (CALIB1's own depth: 10 calib_a + 10
# calib_b reads = 20, its sole class-I "primary" calibrator role) -- every
# existing cn_estimate/threshold assertion in this file was derived
# against that same number, so this preserves them unchanged.
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
    """anchors: anchor_name -> (gene, sequence), all role=target.

    `length` (2026-09-01, stages/cn.py's own sieve-reuse depth revision):
    stage0_depth_by_gene() now reconstructs real aligned bases as
    AnchorHit.coverage * this column's own value, straight off the real
    Stage 0 sieve alignment against this same anchors.fa.gz -- must be the
    anchor's own real sequence length (len(seq)), not omitted, or every
    gene's CN silently comes back zero (no length -> no depth contribution
    at all, same graceful-degradation contract stage0_depth_by_gene() uses
    for a genuinely absent gene, which a length-less synthetic fixture is
    indistinguishable from)."""
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
    """Tiny synthetic chr17-control reference (stages/chr17_control.py,
    2026-09-01 CN-normalisation revision) -- unrelated in sequence to every
    other synthetic fixture in this file (a different, fixed seed), so it
    never coincidentally aligns against GENE1/CALIB1's own anchors/pool.
    Only test_cli.py needs this: it drives the real argparse CLI path,
    which has no --control-depth escape hatch (deliberately -- that is a
    test-only override, not something a real user should be able to fake),
    so it points --chr17-control-fasta at this instead of the real 4.5Mb
    resource. Every other test in this file/test_output.py injects
    control_depth directly via run_sample()/run_sample_phased()'s own
    keyword argument and does not need this."""
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
    """A gene with zero of its own reads, alongside a working calibrator,
    must show up as a recorded GeneFailure (CN estimate of exactly 0.0,
    invariant 4's "unconfirmed absence") rather than silently vanishing or
    crashing the run."""
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
    """A gene recruiting far more depth than its calibrator (real HG00097
    finding: HLA-C/MICA/HLA-DRB1 all rounded above 2) must have its
    *search-sizing* cn_int clamped to max_cn -- the raw cn_estimate stays
    honest/uncapped in gene_cn for diagnosis."""
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
    """When Stage 2's LRT says weak evidence for two distinct alleles,
    pipeline.py must collapse to a single haplotype call built from every
    read in the gene's matrix (not split across two clusters where one is
    noise) -- stages/consensus.py's own docstring names this gap and how to
    close it; this exercises the orchestration-layer half. The real fixture
    here is genuinely heterozygous (two distinct real alleles) -- lrt_statistic
    is forced low via monkeypatch specifically to isolate the collapse
    *mechanism* from whether real ONT data naturally produces a low LRT."""
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
    """Real HLA-B finding: min_heterozygous_lrt alone doesn't catch a
    "distinct" second allele with essentially no reads behind it. Fixture
    gives allele_b exactly one real read (the real HLA-B shape: one
    genuinely divergent read, not zero) so it survives stage1_n_best=1's
    own-best-hit restriction as a real Stage 1 candidate at all -- a fully
    zero-read allele_b is invisible to candidate_alleles() under
    production's n_best=1 default and select_template_set_ont_r10() raises
    before any collapse logic runs (2026-08-07 finding, this fixture
    previously predated that default). build_gene_consensus()'s real hard
    read partition should still send only that 1 read to the second,
    merely-plausible candidate the cn=2 combination search has to name --
    far below min_reads_per_haplotype. lrt_statistic is forced high via
    monkeypatch specifically to isolate this check from the LRT check
    above it (same style as the weak-LRT test above, opposite direction)."""
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
    """User-requested check, distinct from min_reads_per_haplotype: a minor
    haplotype can clear the absolute read-count floor (3) yet still be a
    small fraction of the major haplotype's depth -- e.g. 3 reads vs 40 --
    which is the signature of a real, dominant homozygous call with a
    trickle of cross-mapped/misassigned reads, not a genuine second
    haplotype. Fixture keeps the minor count exactly at the absolute floor
    (3, so min_reads_per_haplotype alone does not fire) while the ratio
    (3/40 = 0.075) sits below min_haplotype_read_ratio's default 0.1.
    lrt_statistic is forced high via monkeypatch, same style as the
    zero-read test above, to isolate this check from the other two."""
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
    """2026-08-31 revision, user-directed: `diploid_typical` used to force
    cn_int to max_cn unconditionally whenever the raw estimate rounded to
    >=1, on the reasoning that the old matrix-derived depth (assign.py's
    best_gene_per_read()) was too noisy to trust a genuine CN=1 reading.
    stages/cn.py's depth source changed the same session specifically
    because it was shown, on real data, to systematically *undercount* a
    different gene cluster (HLA-H/K/T/U/W) -- and real HG00097/HG00320
    checks found every diploid_typical gene's raw estimate already lands
    close to 2 under the new signal without forcing. Forcing is removed:
    a real, low estimate (0.8, which used to force cn_int=2 regardless
    since the old branch only checked "did this round to >=1") must now
    round via the ordinary >0.25/>1.25 thresholds honestly, the same rule
    every other gene already uses -- otherwise a genuine single-copy
    deletion in a normally-diploid gene is silently masked."""
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
    """2026-08-23, user-requested: `_round_cn_estimate`'s presence bar
    (>0.25) is below plain round()'s (>=0.5) -- real HG00097 motivating
    case was HLA-U (cn_estimate 0.41, present on ~86.5% of real hla_seq/
    haplotypes) rounding to 0 and dropping out of calls.tsv entirely.
    Fixture: 5 VARGENE reads against a 20-read CALIB1 calibrator ->
    cn_estimate = 2*5/20 = 0.5 exactly, which plain round() sends to 0
    (Python's round-half-to-even) but the new rule sends to 1."""
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
    """Upper threshold of the same rule: cn_estimate 1.3 (2*13/20) rounds
    to 1 under plain round() but must round to 2 under the new >1.25 rule
    -- checked via the reported `cn` field (the search-sizing integer) only.
    Deliberately not asserting on row count/allele recovery: with only a
    single real read supporting the second candidate, whether the
    downstream LRT/read-floor/ratio collapse checks fold this back to one
    reported row is a separate, already-covered question (the collapse
    tests elsewhere in this file) -- this test isolates the rounding rule
    itself, which is reflected in `cn` on every row regardless."""
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
    """A gene with genes.yaml's low_cn_floor set (DRB3/4/5 in the real
    manifest) must be reported as CN=0 when its raw estimate falls below
    that floor -- even though plain round() would round it up to 1 and
    attempt a real single-allele call. User's own hypothesis: for genes
    sharing a physical block with an always-present neighbour (DRB1), a
    thin estimate is more plausibly cross-mapped read leakage than genuine
    partial presence."""
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
    """User-requested, 2026-08-26: real HG00097 shape -- DRB3 rounds to 2
    (raw estimate 1.49, in the ambiguous (1.2, 2.0) band above the ordinary
    >1.25 cutoff), DRB5 independently rounds to a confident 1 (raw 1.02).
    Summed, that's 3 copies at a locus where a real haplotype carries at
    most one of DRB3/DRB4/DRB5 -- DRB3 must be pulled down to 1 (the two
    genes read as heterozygous, one haplotype each), not left at 2."""
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
    """Two genes both at cn_int=2 (e.g. two independently ambiguous
    estimates) with no gene confidently at 1 -- there's no basis in the
    user's requested rule to prefer pulling one down over the other, so
    neither should be touched even though the group sum (4) is impossible."""
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
    """A synthetic mini-BAM carrying only what run_sample_phased() actually
    reads from it: read name -> (HP, PS) tags (input/phasing.py). Coordinates
    are irrelevant here -- Stage 0 extraction (DESIGN.md S3.1) is assumed to
    have already happened by the time this function's caller (cli.py) hands
    a mini-BAM to run_sample_phased(), so this fixture skips straight to
    "a mini-BAM with real phase tags on it", not a full extraction round-trip
    (that mechanism is already covered by input/test_bam.py)."""
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
    """GENE1's reads are cleanly split HP=1/HP=2 under one PS block: the
    phased path should replace its unphased calls with two per-haplotype
    calls labelled by the real HP value. CALIB1 carries no phase tags at
    all, so it must fall back to the ordinary unphased path untouched --
    exercising both routings in the same run, per the user's request that
    the two are clearly distinguishable."""
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
    """Real HG00097 finding (2026-08-27): a gene can have HP-tagged reads
    that agree on one PS block, yet still be unsafe to phase if only a
    small minority of its recruited reads are actually tagged (HLA-C: 11 of
    72). min_phased_tagged_fraction (default 0.5) gates on this directly."""
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
    """A haplotype with a nonzero but tiny tagged-read count (below
    min_reads_per_haplotype) must not get a full Stage 2-5 call -- it should
    be recorded as a failure, while a well-supported sibling haplotype for
    the same gene still gets called normally."""
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

    # hap1: 10 well-supported reads. hap2: only 2 reads -- below the
    # default min_reads_per_haplotype=3 floor. All tagged (fraction=1.0),
    # so this exercises the per-haplotype floor specifically, not the
    # gene-level tagged-fraction gate above.
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
# _decide_phase_routing (DESIGN.md S10.3), factored out of the per-gene loop
# and the C4A/C4B pseudo-gene check so the routing decision is unit-testable
# without any real alignment work.
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
# DESIGN.md S10.5/S10.6: per-haplotype C4 CN's own PS-matched calibrator
# panel (_c4_calibrator_reads_for_haplotype) -- pure-function tests, no
# alignment needed.
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
    """C4A/C4B reads that split cleanly by HP should route "phased" and get
    real per-haplotype C4AbCall/C4FormLengthCall objects -- using this
    file's TINY_C4_FASTA/META fixture, whose synthetic ipd_ids never match
    a real resources/models/... models file entry, every read is correctly
    "uninformative" for both classifiers (module docstring on
    _write_c4_pool_files) -- this test checks the phased wiring runs to
    completion and produces that expected shape, not the A/B signal itself
    (already covered by stages/test_c4_form.py's classifier-level tests)."""
    rng = random.Random(9)
    allele_a = _random_seq(700, rng)
    allele_b = _mutate(allele_a, 5, random.Random(10))

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
    })
    # A dedicated, longer local C4 pool -- NOT the shared module-level
    # TINY_C4_FASTA/META (only 80bp): input/fastq.py's own docstring notes
    # minimap2's -A1 -s350 scoring reports nothing below ~350bp of matched
    # sequence, so an 80bp anchor/read pair never produces a real alignment
    # through either the anchor sieve or assign.py's pool alignment. 500bp
    # clears that floor.
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
    """DESIGN.md S10.5/S10.6, added 2026-08-28: when a class I calibrator
    gene (CALIB1) is *also* cleanly phased under the identical PS block as
    C4's own reads, its HP-split depth is trusted as C4's per-haplotype
    calibrator panel and a real cn_estimate comes back for each haplotype
    -- not just the sample-wide aggregate."""
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
    """DESIGN.md S10.3's stitching warning, made concrete: CALIB1 is
    cleanly phased, but under a *different* PS block than C4's own --
    its HP-split depth must not be borrowed as C4's calibrator panel.
    cn_estimate comes back None (honest, invariant 4's "loud, not silent"),
    while the real per-haplotype read depth is still reported."""
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
    """Baseline: existing tests never add a C4A/C4B anchor at all, so no
    read is ever recruited to C4 -- confirms the "C4" routing key still
    appears (satisfying the "well annotated" requirement) even when there
    is nothing to phase."""
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
