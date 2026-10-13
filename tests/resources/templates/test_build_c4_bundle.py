"""Unit tests for build_c4_bundle.py's C4AL/AS/BL/BS -> C4A/C4B collapse and
manifest."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pysam

from hlaont.resources.templates import build_c4_bundle

_GTF_HEADER = "## format: gtf\n"


def _gtf_row(seqname, feature, start, end, attrs) -> str:
    attrs_str = "; ".join(f'{k} "{v}"' for k, v in attrs.items()) + ";"
    return f"{seqname}\tTEST\t{feature}\t{start}\t{end}\t.\t+\t.\t{attrs_str}\n"


def _write_hla_seq_fixture(hla_seq_dir: Path, gene: str, haplotype_id: str, contig: str, seq: str) -> None:
    staging_path = hla_seq_dir / "all_HLA.staging.fasta"
    with open(staging_path, "a") as fh:
        fh.write(f">{contig}\n{seq}\n")
    gz_path = hla_seq_dir / "all_HLA.fasta.gz"
    for p in (gz_path, Path(str(gz_path) + ".fai"), Path(str(gz_path) + ".gzi")):
        if p.exists():
            p.unlink()
    pysam.tabix_compress(str(staging_path), str(gz_path), force=True)
    pysam.faidx(str(gz_path))

    seqname = f"{contig}:1-1000"

    def attrs(extra=""):
        return {"gene_id": "G1", "transcript_id": "T1", "gene_name": gene, **({} if not extra else extra)}

    rows = [
        _gtf_row(seqname, "gene", 1, len(seq), attrs()),
        _gtf_row(seqname, "transcript", 1, len(seq), attrs()),
        _gtf_row(seqname, "exon", 1, len(seq), {**attrs(), "exon_number": "1"}),
    ]
    gtf_path = hla_seq_dir / f"{haplotype_id}_HLA.gtf.gz"
    with gzip.open(gtf_path, "wt") as fh:
        fh.write(_GTF_HEADER)
        fh.writelines(rows)


def _contig_seq(length: int, seed: str) -> str:
    bases = "ACGT"
    return "".join(bases[(i + len(seed)) % 4] for i in range(length))


def test_collapses_all_four_forms_to_two_genes(tmp_path: Path) -> None:
    hla_seq_dir = tmp_path / "hla_seq"
    hla_seq_dir.mkdir()
    forms = {"C4AL": 400, "C4AS": 300, "C4BL": 410, "C4BS": 310}
    for i, (form, length) in enumerate(forms.items()):
        _write_hla_seq_fixture(hla_seq_dir, form, f"SAMPLE{i}_1", f"ctg{i}", _contig_seq(length, form))

    out_dir = tmp_path / "bundle"
    build_c4_bundle.main(["--hla-seq-dir", str(hla_seq_dir), "--out-dir", str(out_dir)])

    manifest = json.loads((out_dir / "c4_bundle_manifest.json").read_text())
    assert manifest["records_per_gene"] == {"C4A": 2, "C4B": 2}
    assert manifest["records_per_real_form"] == {"C4AL": 1, "C4AS": 1, "C4BL": 1, "C4BS": 1}
    assert manifest["total_records"] == 4

    meta = gzip.open(out_dir / "c4.meta.tsv.gz", "rt").read()
    assert "\nC4A\t" in meta or meta.startswith("C4A\t")
    assert "C4AL" not in meta.split("\n")[0]  # header row has no gene name
    genes_seen = {line.split("\t")[0] for line in meta.strip().splitlines()[1:]}
    assert genes_seen == {"C4A", "C4B"}
    assert "form=long (C4AL)" in meta
    assert "form=short (C4AS)" in meta
    assert "form=long (C4BL)" in meta
    assert "form=short (C4BS)" in meta

    # allele names carry the collapsed gene, not the real form
    fasta_text = gzip.open(out_dir / "c4.fa.gz", "rt").read()
    headers = [line for line in fasta_text.splitlines() if line.startswith(">")]
    assert all(h.startswith(">C4A*asm") or h.startswith(">C4B*asm") for h in headers)


def test_writes_gene_models_keyed_consistently_with_meta(tmp_path: Path) -> None:
    """c4.models.jsonl.gz is keyed by the same ipd_ids as the meta."""
    hla_seq_dir = tmp_path / "hla_seq"
    hla_seq_dir.mkdir()
    _write_hla_seq_fixture(hla_seq_dir, "C4AL", "SAMPLE0_1", "ctg0", _contig_seq(400, "C4AL"))
    _write_hla_seq_fixture(hla_seq_dir, "C4BL", "SAMPLE1_1", "ctg1", _contig_seq(410, "C4BL"))

    out_dir = tmp_path / "bundle"
    build_c4_bundle.main(["--hla-seq-dir", str(hla_seq_dir), "--out-dir", str(out_dir)])

    models_path = out_dir / "c4.models.jsonl.gz"
    assert models_path.exists()
    models = [json.loads(line) for line in gzip.open(models_path, "rt")]
    assert {m["gene"] for m in models} == {"C4A", "C4B"}
    assert all(m["exons"] for m in models)  # the fixture's single exon survives the collapse

    meta = gzip.open(out_dir / "c4.meta.tsv.gz", "rt").read()
    meta_ipd_ids = {line.split("\t")[1] for line in meta.strip().splitlines()[1:]}
    assert {m["ipd_id"] for m in models} == meta_ipd_ids


def test_errors_loudly_on_zero_records(tmp_path: Path) -> None:
    hla_seq_dir = tmp_path / "hla_seq"
    hla_seq_dir.mkdir()
    # A valid (empty-of-C4) fasta but no GTFs at all -- AssemblyGTFSource's
    # own glob finds nothing to extract.
    staging_path = hla_seq_dir / "all_HLA.staging.fasta"
    staging_path.write_text(">unused\nACGT\n")
    pysam.tabix_compress(str(staging_path), str(hla_seq_dir / "all_HLA.fasta.gz"), force=True)
    pysam.faidx(str(hla_seq_dir / "all_HLA.fasta.gz"))
    try:
        build_c4_bundle.main(["--hla-seq-dir", str(hla_seq_dir), "--out-dir", str(tmp_path / "bundle")])
        raised = False
    except SystemExit:
        raised = True
    assert raised
