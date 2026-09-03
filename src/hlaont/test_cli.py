"""End-to-end test for cli.py, reusing stages/test_pipeline.py's synthetic
two-gene fixture builders. Colocated at repo root (alongside cli.py itself)
per the development notes' "unit tests alongside code" convention -- tests/ is
reserved for the not-yet-existing HPRC integration fixtures.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import pysam

import cli  # adds stages/ and input/ to sys.path as an import side effect
from regions import RegionSet  # noqa: E402 -- cli's sys.path.insert makes this resolvable

sys.path.insert(0, str(Path(__file__).parent / "stages"))
from test_pipeline import (  # noqa: E402
    C4_EXTRA_CALLS_TSV_GENES,
    TINY_C4_FASTA,
    TINY_C4_META,
    _write_anchors,
    _write_chr17_control_fixture,
    _write_fastq,
    _write_genes_yaml,
    _write_pool,
)


def _build_fixture(tmp_path):
    rng = random.Random(2024)
    allele_a = "".join(rng.choice("ACGT") for _ in range(700))
    bases = list(allele_a)
    for p in random.Random(1).sample(range(len(bases)), 5):
        alts = [b for b in "ACGT" if b != bases[p]]
        bases[p] = random.Random(p + 1).choice(alts)
    allele_b = "".join(bases)
    calib_a = "".join(random.Random(77).choice("ACGT") for _ in range(700))
    calib_b = calib_a[:200] + ("G" if calib_a[200] != "G" else "T") + calib_a[201:]

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
    chr17_control_fasta, chr17_control_seq = _write_chr17_control_fixture(tmp_path)
    reads = (
        [(f"a{i}", allele_a) for i in range(10)]
        + [(f"b{i}", allele_b) for i in range(10)]
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
        # A handful of reads matching the synthetic chr17 control fixture --
        # this test drives the real CLI path (--chr17-control-fasta below),
        # so it needs real, nonzero control signal in the FASTQ itself to
        # get past cn.py's 2026-09-01 "no control signal" loud-None branch
        # and exercise real Stage 2-5 typing, not just a graceful failure.
        + [(f"chr17_{i}", chr17_control_seq) for i in range(5)]
    )
    fastq_path = _write_fastq(tmp_path, reads)
    return {
        "fastq": fastq_path, "pool_fasta": pool_fasta, "pool_meta": pool_meta,
        "models_path": models_path, "cds_catalogue_path": cds_catalogue,
        "genes_yaml": genes_yaml, "anchors_path": anchors_path,
        "anchors_meta_path": anchors_meta_path,
        "c4_pool_fasta": TINY_C4_FASTA, "c4_pool_meta": TINY_C4_META,
        "chr17_control_fasta": chr17_control_fasta,
    }


def test_cli_end_to_end_writes_all_outputs(tmp_path):
    fx = _build_fixture(tmp_path)
    out_dir = tmp_path / "out"

    exit_code = cli.main([
        "run", "sampleCLI", str(fx["fastq"]), "--out-dir", str(out_dir), "--n-threads", "2",
        "--pool-fasta", str(fx["pool_fasta"]),
        "--pool-meta", str(fx["pool_meta"]),
        "--models-path", str(fx["models_path"]),
        "--cds-catalogue-path", str(fx["cds_catalogue_path"]),
        "--genes-yaml", str(fx["genes_yaml"]),
        "--anchors-path", str(fx["anchors_path"]),
        "--anchors-meta-path", str(fx["anchors_meta_path"]),
        "--c4-pool-fasta", str(fx["c4_pool_fasta"]),
        "--c4-pool-meta", str(fx["c4_pool_meta"]),
        "--chr17-control-fasta", str(fx["chr17_control_fasta"]),
        "--dorado-version", "7.1.0",
    ])
    assert exit_code == 0

    for name in ("calls.tsv", "consensus.fa", "consensus.qv.tsv", "variants.vcf", "qc.json"):
        assert (out_dir / f"sampleCLI.{name}").exists(), f"sampleCLI.{name} was not written"

    with open(out_dir / "sampleCLI.calls.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    # +5 rows for C4 (no per-haplotype call for any of them -- output.py's
    # write_calls_tsv docstring, 2026-08-26 user-requested addition).
    haplotype_rows = [r for r in rows if r["gene"] not in C4_EXTRA_CALLS_TSV_GENES]
    assert len(haplotype_rows) == 4
    assert {r["gene"] for r in haplotype_rows} == {"GENE1", "CALIB1"}
    assert {r["gene"] for r in rows if r["gene"] in C4_EXTRA_CALLS_TSV_GENES} == C4_EXTRA_CALLS_TSV_GENES


def test_cli_errors_loudly_on_missing_fastq(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.fastq"
    try:
        cli.main(["run", "sampleX", str(missing), "--out-dir", str(tmp_path / "out")])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def _write_bam_and_reference(
    tmp_path, reads: list[tuple[str, str]],
    *, phase_tags: dict[str, tuple[int, int]] | None = None,
):
    """A source BAM of unmapped reads only -- extract_mini_bam pulls every
    unmapped read via samtools' "*" pseudo-region regardless of RegionSet
    contents (input/test_bam.py's own extraction tests cover the mapped/
    ALT/decoy region-matching logic already), so this exercises cli.py's
    BAM->mini-BAM->FASTQ wiring without needing a real chromosome-scale
    reference to satisfy derive_regions' build-identification check.

    `phase_tags` (DESIGN.md S10), read name -> (HP, PS), is optional and
    defaults to no tags at all -- unphased callers (most tests in this
    file) get exactly the same BAM as before this parameter existed.
    """
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    bam_path = tmp_path / "source.bam"
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bf:
        for name, seq in reads:
            a = pysam.AlignedSegment(header=bf.header)
            a.query_name = name
            a.query_sequence = seq
            a.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
            a.flag = 4
            a.reference_id = -1
            a.reference_start = -1
            if phase_tags is not None and name in phase_tags:
                hp, ps = phase_tags[name]
                a.set_tags([("HP", hp), ("PS", ps)])
            bf.write(a)
    pysam.index(str(bam_path))

    reference_path = tmp_path / "ref.fa"
    with open(reference_path, "w") as fh:
        fh.write(">chr1\n" + "N" * 1000 + "\n")
    pysam.faidx(str(reference_path))
    return bam_path, reference_path


def test_cli_bam_path_extracts_and_writes_all_outputs(tmp_path, monkeypatch):
    fx = _build_fixture(tmp_path)
    # Reuse the same reads _build_fixture wrote to FASTQ, sourced from the
    # FASTQ file itself so the two paths see byte-identical read content.
    with pysam.FastxFile(str(fx["fastq"])) as fq:
        reads = [(entry.name, entry.sequence) for entry in fq]

    bam_path, reference_path = _write_bam_and_reference(tmp_path, reads)
    monkeypatch.setattr(
        cli, "derive_regions_from_header",
        lambda header, **kwargs: RegionSet(build="TEST", regions=()),
    )

    out_dir = tmp_path / "out"
    exit_code = cli.main([
        "run", "sampleBAM", "--bam", str(bam_path), "--reference", str(reference_path),
        "--out-dir", str(out_dir), "--n-threads", "2",
        "--pool-fasta", str(fx["pool_fasta"]),
        "--pool-meta", str(fx["pool_meta"]),
        "--models-path", str(fx["models_path"]),
        "--cds-catalogue-path", str(fx["cds_catalogue_path"]),
        "--genes-yaml", str(fx["genes_yaml"]),
        "--anchors-path", str(fx["anchors_path"]),
        "--anchors-meta-path", str(fx["anchors_meta_path"]),
        "--c4-pool-fasta", str(fx["c4_pool_fasta"]),
        "--c4-pool-meta", str(fx["c4_pool_meta"]),
    ])
    assert exit_code == 0

    assert (out_dir / "sampleBAM.stage0.mini.bam").exists()
    assert (out_dir / "sampleBAM.stage0.fastq").exists()
    for name in ("calls.tsv", "consensus.fa", "consensus.qv.tsv", "variants.vcf", "qc.json"):
        assert (out_dir / f"sampleBAM.{name}").exists(), f"sampleBAM.{name} was not written"

    with open(out_dir / "sampleBAM.calls.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    haplotype_rows = [r for r in rows if r["gene"] not in C4_EXTRA_CALLS_TSV_GENES]
    assert len(haplotype_rows) == 4
    assert {r["gene"] for r in haplotype_rows} == {"GENE1", "CALIB1"}


def test_cli_phased_end_to_end_writes_haplotype_column_and_qc_phasing(tmp_path, monkeypatch):
    """DESIGN.md S10: --phased with --bam should route GENE1 (cleanly
    HP/PS-tagged) through the per-haplotype path and CALIB1 (no tags) through
    the unphased fallback, both reported in qc.json's "phasing" block."""
    fx = _build_fixture(tmp_path)
    with pysam.FastxFile(str(fx["fastq"])) as fq:
        reads = [(entry.name, entry.sequence) for entry in fq]

    phase_tags = {name: (1, 100) for name, _ in reads if name.startswith("a")}
    phase_tags.update({name: (2, 100) for name, _ in reads if name.startswith("b")})
    # ca*/cb* (CALIB1) deliberately left untagged -- must fall back.
    bam_path, reference_path = _write_bam_and_reference(tmp_path, reads, phase_tags=phase_tags)
    monkeypatch.setattr(
        cli, "derive_regions_from_header",
        lambda header, **kwargs: RegionSet(build="TEST", regions=()),
    )

    out_dir = tmp_path / "out"
    exit_code = cli.main([
        "run", "samplePhased", "--bam", str(bam_path), "--reference", str(reference_path),
        "--phased", "--out-dir", str(out_dir), "--n-threads", "2",
        "--pool-fasta", str(fx["pool_fasta"]),
        "--pool-meta", str(fx["pool_meta"]),
        "--models-path", str(fx["models_path"]),
        "--cds-catalogue-path", str(fx["cds_catalogue_path"]),
        "--genes-yaml", str(fx["genes_yaml"]),
        "--anchors-path", str(fx["anchors_path"]),
        "--anchors-meta-path", str(fx["anchors_meta_path"]),
        "--c4-pool-fasta", str(fx["c4_pool_fasta"]),
        "--c4-pool-meta", str(fx["c4_pool_meta"]),
    ])
    assert exit_code == 0

    with open(out_dir / "samplePhased.calls.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert "haplotype" in rows[0]
    gene1_haplotypes = {r["haplotype"] for r in rows if r["gene"] == "GENE1"}
    assert gene1_haplotypes == {"1", "2"}
    calib1_haplotypes = {r["haplotype"] for r in rows if r["gene"] == "CALIB1"}
    assert calib1_haplotypes == {"unphased"}

    qc = json.loads((out_dir / "samplePhased.qc.json").read_text())
    assert qc["phasing"]["GENE1"]["routing"] == "phased"
    assert qc["phasing"]["CALIB1"]["routing"] == "unphased_fallback"


def test_cli_errors_when_phased_given_without_bam(tmp_path):
    fx = _build_fixture(tmp_path)
    try:
        cli.main([
            "run", "sampleX", str(fx["fastq"]), "--phased", "--out-dir", str(tmp_path / "out"),
        ])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_cli_errors_when_both_fastq_and_bam_given(tmp_path):
    fx = _build_fixture(tmp_path)
    bam_path, reference_path = _write_bam_and_reference(tmp_path, [("r1", "ACGT")])
    try:
        cli.main([
            "run", "sampleX", str(fx["fastq"]), "--bam", str(bam_path),
            "--reference", str(reference_path), "--out-dir", str(tmp_path / "out"),
        ])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_cli_errors_when_cram_given_without_reference(tmp_path):
    """2026-08-29: --reference is only mandatory for a .cram --bam input
    (CRAM cannot decode without it) -- a plain .bam needs no reference at
    all, see test_cli_bam_path_works_without_reference below. The .cram
    path here need not exist; the suffix check happens before any
    existence check (cli.py's own ordering)."""
    cram_path = tmp_path / "source.cram"
    try:
        cli.main(["run", "sampleX", "--bam", str(cram_path), "--out-dir", str(tmp_path / "out")])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_cli_bam_path_works_without_reference(tmp_path, monkeypatch):
    """2026-08-29, real-data need: a real whole-genome ONT BAM with no
    accompanying reference FASTA on disk. A plain .bam already carries its
    own sequences and needs no reference to derive the region set from its
    header or to extract regions via samtools view -- see
    bam.extract_mini_bam()'s own docstring. Otherwise identical to
    test_cli_bam_path_extracts_and_writes_all_outputs, just without
    --reference."""
    fx = _build_fixture(tmp_path)
    with pysam.FastxFile(str(fx["fastq"])) as fq:
        reads = [(entry.name, entry.sequence) for entry in fq]

    bam_path, _reference_path = _write_bam_and_reference(tmp_path, reads)
    monkeypatch.setattr(
        cli, "derive_regions_from_header",
        lambda header, **kwargs: RegionSet(build="TEST", regions=()),
    )

    out_dir = tmp_path / "out"
    exit_code = cli.main([
        "run", "sampleBAMNoRef", "--bam", str(bam_path),
        "--out-dir", str(out_dir), "--n-threads", "2",
        "--pool-fasta", str(fx["pool_fasta"]),
        "--pool-meta", str(fx["pool_meta"]),
        "--models-path", str(fx["models_path"]),
        "--cds-catalogue-path", str(fx["cds_catalogue_path"]),
        "--genes-yaml", str(fx["genes_yaml"]),
        "--anchors-path", str(fx["anchors_path"]),
        "--anchors-meta-path", str(fx["anchors_meta_path"]),
        "--c4-pool-fasta", str(fx["c4_pool_fasta"]),
        "--c4-pool-meta", str(fx["c4_pool_meta"]),
    ])
    assert exit_code == 0
    assert (out_dir / "sampleBAMNoRef.stage0.mini.bam").exists()
    assert (out_dir / "sampleBAMNoRef.stage0.fastq").exists()
    assert (out_dir / "sampleBAMNoRef.calls.tsv").exists()


_HLA_DAT_HEADER = """\
ID   HLA99999; SV 1; standard; DNA; HUM; {length} BP.
XX
DE   {allele}, Human MHC test sequence
XX
CC   IPD-IMGT/HLA Release Version 3.65.0
XX
"""

_GENOMIC_FT = """\
FT   source          1..1000
FT   CDS             join(301..373,504..600,700..750)
FT                   /codon_start=1
FT                   /gene="{gene}"
FT   UTR             1..300
FT   exon            301..373
FT                   /number="1"
FT   intron          374..503
FT                   /number="1"
FT   exon            504..600
FT                   /number="2"
FT   intron          601..699
FT                   /number="2"
FT   exon            700..750
FT                   /number="3"
FT   UTR             751..1000
"""


def _write_update_db_fixture(tmp_path):
    dat_path = tmp_path / "hla.dat"
    seq = "ACGT" * 250
    with open(dat_path, "w") as fh:
        fh.write(_HLA_DAT_HEADER.format(length=len(seq), allele="HLA-TEST*01:01"))
        fh.write(_GENOMIC_FT.format(gene="HLA-TEST"))
        fh.write("SQ   Sequence\n" + seq + "\n//\n")

    genes_yaml = tmp_path / "genes.yaml"
    genes_yaml.write_text("""\
manifest_version: 1
genes:
  - gene: HLA-TEST
    class: I
    segment: null
    cn_model: diploid_typical
    calibrator: false
    typing_module: template_polish
    tier: 1
    audit_genomic_v3.65.0: 1
""")
    hla_seq_dir = tmp_path / "hla_seq"
    hla_seq_dir.mkdir()  # no gtf.gz/fasta needed -- HLA-TEST has no assembly_extraction field
    return dat_path, genes_yaml, hla_seq_dir


def test_update_db_errors_loudly_on_missing_hla_seq_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    try:
        cli.main(["update-db", "--hla-seq-dir", str(missing)])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_update_db_rebuilds_pool_end_to_end(tmp_path):
    dat_path, genes_yaml, hla_seq_dir = _write_update_db_fixture(tmp_path)
    out_dir = tmp_path / "pool"

    exit_code = cli.main([
        "update-db", "--hla-seq-dir", str(hla_seq_dir),
        "--genes-yaml", str(genes_yaml), "--raw-dir", str(tmp_path / "raw"),
        "--out-dir", str(out_dir), "--ipd-dat", str(dat_path),
    ])
    assert exit_code == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "ref.gene.fa.gz").exists()


def test_update_db_works_without_hla_seq_dir(tmp_path):
    """2026-08-25: --hla-seq-dir is optional -- update-db must still run to
    completion without it (falling back to the checked-in assembly bundle
    for any assembly_extraction gene; this fixture's HLA-TEST has none, so
    it's IPD-only either way, but the CLI wiring itself must not require
    the flag anymore)."""
    dat_path, genes_yaml, _hla_seq_dir = _write_update_db_fixture(tmp_path)
    out_dir = tmp_path / "pool"

    exit_code = cli.main([
        "update-db",
        "--genes-yaml", str(genes_yaml), "--raw-dir", str(tmp_path / "raw"),
        "--out-dir", str(out_dir), "--ipd-dat", str(dat_path),
        "--assembly-bundle-dir", str(tmp_path / "no_such_bundle"),
    ])
    assert exit_code == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "ref.gene.fa.gz").exists()
