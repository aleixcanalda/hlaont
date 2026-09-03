"""End-to-end tests for build.py's multi-source wiring (2026-08-22): IPD +
AssemblyGTFSource combined, cross-source dedup, and manifest provenance --
using tiny synthetic fixtures (hla.dat, genes.yaml, hla_seq_dir), not the
real IPD/hla_seq datasets. The pre-existing IPD-only path (gene selection,
CDS catalogue, the missing_model hard-stop) is untouched by this change and
already covered by test_sources.py; this file covers only what's new.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pysam

import build
import build_assembly_bundle

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


def _write_hla_dat(tmp_path: Path, alleles: dict[str, str]) -> Path:
    """alleles: allele_name -> 1000bp sequence (must have >=2 UTR + >=1
    intron to qualify as "genomic" under build.py's require_genomic=True
    bar -- this module's own single-exon fixture would silently yield zero
    pool records, which is exactly what happened before this comment was
    added)."""
    path = tmp_path / "hla.dat"
    with open(path, "w") as fh:
        for allele, seq in alleles.items():
            gene = allele.split("*")[0]
            fh.write(_HLA_DAT_HEADER.format(length=len(seq), allele=allele))
            fh.write(_GENOMIC_FT.format(gene=gene))
            fh.write("SQ   Sequence\n" + seq + "\n//\n")
    return path


def _write_genes_yaml(tmp_path: Path, extra_test_gene_fields: str = "") -> Path:
    path = tmp_path / "genes.yaml"
    path.write_text(f"""\
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
    {extra_test_gene_fields}
  - gene: HLA-OTHER
    class: I
    segment: null
    cn_model: diploid_typical
    calibrator: false
    typing_module: template_polish
    tier: 1
    audit_genomic_v3.65.0: 1
""")
    return path


def _write_hla_seq_fixture(tmp_path: Path, gene: str, haplotype_id: str,
                            contig: str, seq: str) -> None:
    # AssemblyGTFSource's default all_fasta_path is hla_seq_dir/all_HLA.fasta.gz,
    # bgzip-compressed (matching real hla_seq/'s convention, .fai requires
    # proper BGZF, not plain gzip) -- accumulate a plain staging fasta across
    # calls, then (re)compress+index the whole thing each time.
    staging_path = tmp_path / "all_HLA.staging.fasta"
    with open(staging_path, "a") as fh:
        fh.write(f">{contig}\n{seq}\n")

    gz_path = tmp_path / "all_HLA.fasta.gz"
    for p in (gz_path, Path(str(gz_path) + ".fai"), Path(str(gz_path) + ".gzi")):
        if p.exists():
            p.unlink()
    pysam.tabix_compress(str(staging_path), str(gz_path), force=True)
    pysam.faidx(str(gz_path))

    seqname = f"{contig}:1-1000"
    row_tmpl = "{seqname}\tTEST\t{feature}\t{start}\t{end}\t.\t+\t{frame}\t{attrs}\n"

    def attrs(extra=""):
        return f'gene_id "G1"; transcript_id "T1"; gene_name "{gene}"; {extra}'

    rows = [
        row_tmpl.format(seqname=seqname, feature="gene", start=1, end=len(seq), frame=".", attrs=attrs()),
        row_tmpl.format(seqname=seqname, feature="transcript", start=1, end=len(seq), frame=".", attrs=attrs()),
        # Fixed coordinates matching this test file's fixed 400bp sequences
        # (mirrors _GENOMIC_FT's own IPD-side numbering, not derived from
        # len(seq) generically).
        row_tmpl.format(seqname=seqname, feature="UTR", start=1, end=49, frame=".", attrs=attrs()),
        row_tmpl.format(seqname=seqname, feature="exon", start=50, end=349, frame=".", attrs=attrs('exon_number "1";')),
        row_tmpl.format(seqname=seqname, feature="CDS", start=50, end=349, frame="0", attrs=attrs()),
        row_tmpl.format(seqname=seqname, feature="UTR", start=350, end=400, frame=".", attrs=attrs()),
    ]
    gtf_path = tmp_path / f"{haplotype_id}_HLA.gtf.gz"
    with gzip.open(gtf_path, "wt") as fh:
        fh.write("## format: gtf\n")
        fh.writelines(rows)


def test_build_merges_ipd_and_assembly_sources(tmp_path: Path) -> None:
    ipd_seq = "ACGT" * 250  # 1000bp, matches _GENOMIC_FT's coordinates
    dat_path = _write_hla_dat(tmp_path, {"HLA-TEST*01:01": ipd_seq, "HLA-OTHER*01:01": ipd_seq})
    genes_yaml = _write_genes_yaml(tmp_path, extra_test_gene_fields="assembly_extraction: gtf_coordinate")

    hla_seq_dir = tmp_path / "hla_seq"
    hla_seq_dir.mkdir()
    distinct_seq = "TGCA" * 250  # a real, distinct-from-IPD assembly sequence
    duplicate_seq = ipd_seq  # byte-identical to IPD's own record -- should dedup away
    _write_hla_seq_fixture(hla_seq_dir, "HLA-TEST", "SAMPLE1_1", "ctg1", distinct_seq)
    _write_hla_seq_fixture(hla_seq_dir, "HLA-TEST", "SAMPLE2_1", "ctg2", duplicate_seq)

    out_dir = tmp_path / "pool"
    build.main([
        "--genes-yaml", str(genes_yaml), "--raw-dir", str(tmp_path / "raw"),
        "--out-dir", str(out_dir), "--ipd-dat", str(dat_path),
        "--hla-seq-dir", str(hla_seq_dir),
    ])

    manifest = json.loads((out_dir / "manifest.json").read_text())
    source_names = {s["name"] for s in manifest["sources"]}
    assert source_names == {"IPD-IMGT/HLA", "hla_seq-assemblies-gtf"}

    # HLA-TEST: 1 real IPD record + 1 distinct assembly record (the
    # duplicate assembly sequence deduped away against IPD's own) = 2.
    assert manifest["records_per_gene"]["HLA-TEST"] == 2
    # HLA-OTHER has no assembly_extraction field -- untouched, IPD-only.
    assert manifest["records_per_gene"]["HLA-OTHER"] == 1

    meta = gzip.open(out_dir / "ref.gene.meta.tsv.gz", "rt").read()
    assert "IPD-IMGT/HLA" in meta
    assert "hla_seq-assemblies-gtf" in meta


def test_build_without_hla_seq_dir_or_bundle_is_ipd_only(tmp_path: Path) -> None:
    """No --hla-seq-dir and no bundle at --assembly-bundle-dir (a from-scratch
    setup with neither available) -- falls back to IPD-only, same as before
    the 2026-08-25 bundle-fallback revision, not a silent skip."""
    ipd_seq = "ACGT" * 250
    dat_path = _write_hla_dat(tmp_path, {"HLA-TEST*01:01": ipd_seq})
    genes_yaml = _write_genes_yaml(tmp_path, extra_test_gene_fields="assembly_extraction: gtf_coordinate")

    out_dir = tmp_path / "pool"
    build.main([
        "--genes-yaml", str(genes_yaml), "--raw-dir", str(tmp_path / "raw"),
        "--out-dir", str(out_dir), "--ipd-dat", str(dat_path),
        "--assembly-bundle-dir", str(tmp_path / "no_such_bundle"),
    ])

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert {s["name"] for s in manifest["sources"]} == {"IPD-IMGT/HLA"}
    assert manifest["records_per_gene"]["HLA-TEST"] == 1


def test_build_falls_back_to_bundled_assembly_source_without_hla_seq_dir(tmp_path: Path) -> None:
    """2026-08-25: --hla-seq-dir is optional -- when omitted, build.py reads
    a pre-extracted bundle (build_assembly_bundle.py's own output) instead of
    requiring the real, external hla_seq/ directory on disk."""
    ipd_seq = "ACGT" * 250
    dat_path = _write_hla_dat(tmp_path, {"HLA-TEST*01:01": ipd_seq})
    genes_yaml = _write_genes_yaml(tmp_path, extra_test_gene_fields="assembly_extraction: gtf_coordinate")

    hla_seq_dir = tmp_path / "hla_seq"
    hla_seq_dir.mkdir()
    distinct_seq = "TGCA" * 250
    _write_hla_seq_fixture(hla_seq_dir, "HLA-TEST", "SAMPLE1_1", "ctg1", distinct_seq)

    bundle_dir = tmp_path / "bundle"
    build_assembly_bundle.main([
        "--hla-seq-dir", str(hla_seq_dir), "--genes-yaml", str(genes_yaml),
        "--out-dir", str(bundle_dir),
    ])

    out_dir = tmp_path / "pool"
    build.main([
        "--genes-yaml", str(genes_yaml), "--raw-dir", str(tmp_path / "raw"),
        "--out-dir", str(out_dir), "--ipd-dat", str(dat_path),
        "--assembly-bundle-dir", str(bundle_dir),
        # deliberately no --hla-seq-dir
    ])

    manifest = json.loads((out_dir / "manifest.json").read_text())
    source_names = {s["name"] for s in manifest["sources"]}
    assert source_names == {"IPD-IMGT/HLA", "hla_seq-assemblies-bundle"}
    # IPD's own record + the one distinct bundled assembly record
    assert manifest["records_per_gene"]["HLA-TEST"] == 2

    meta = gzip.open(out_dir / "ref.gene.meta.tsv.gz", "rt").read()
    # per-record source/source_version stay the bundle's real origin, not
    # the loading mechanism's own identity (only manifest.json's top-level
    # "sources" list distinguishes bundle-loaded from freshly-parsed).
    assert "hla_seq-assemblies-gtf" in meta
    assert "hla_seq-assemblies-bundle" not in meta
