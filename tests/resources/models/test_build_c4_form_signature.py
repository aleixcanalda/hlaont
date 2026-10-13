"""Unit tests for build_c4_form_signature.py on a tiny synthetic bundle."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from hlaont.resources.models import build_c4_form_signature as fitter


def _write_bundle(
    tmp_path: Path,
    records: list[tuple[str, str, str, str, dict]],
) -> Path:
    """records: (gene, ipd_id, allele, sequence, model_dict)."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    with gzip.open(bundle_dir / "c4.fa.gz", "wt") as fh:
        for gene, ipd_id, allele, seq, _model in records:
            fh.write(f">{allele}\n{seq}\n")

    with gzip.open(bundle_dir / "c4.meta.tsv.gz", "wt") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["gene", "ipd_id", "allele", "fields", "length", "genomic",
                          "source", "source_version", "description", "nearest_ipd_match"])
        for gene, ipd_id, allele, seq, model in records:
            form = model["_form"]
            writer.writerow([gene, ipd_id, allele, "1", str(len(seq)), "True",
                              "test", "1", f"form={form} ({gene}L); test", ""])

    with gzip.open(bundle_dir / "c4.models.jsonl.gz", "wt") as fh:
        for gene, ipd_id, allele, seq, model in records:
            obj = {k: v for k, v in model.items() if not k.startswith("_")}
            obj.update({"gene": gene, "ipd_id": ipd_id, "allele": allele, "length": len(seq)})
            fh.write(json.dumps(obj) + "\n")

    return bundle_dir


def _model(exon26: tuple[int, int] | None, intron9: tuple[int, int] | None, form: str) -> dict:
    exons = [[26, exon26[0], exon26[1]]] if exon26 else []
    introns = [[9, intron9[0], intron9[1]]] if intron9 else []
    return {"codon_start": 1, "cds_exons": [], "utrs": [], "exons": exons, "introns": introns, "_form": form}


def test_exon26_signature_finds_discriminating_positions(tmp_path: Path):
    # exon26 spans seq[9:19] (1-based 10-19), identical between A/B except
    # one position (index 5 within the exon -> global index 14).
    a_seq = "N" * 9 + "AAAAAXAAAA" + "N" * 20
    b_seq = "N" * 9 + "AAAAAYAAAA" + "N" * 20
    records = [
        ("C4A", "IPD_A1", "C4A*asm1", a_seq, _model((10, 19), None, "long")),
        ("C4A", "IPD_A2", "C4A*asm2", a_seq, _model((10, 19), None, "long")),
        ("C4B", "IPD_B1", "C4B*asm1", b_seq, _model((10, 19), None, "long")),
        ("C4B", "IPD_B2", "C4B*asm2", b_seq, _model((10, 19), None, "long")),
    ]
    bundle_dir = _write_bundle(tmp_path, records)
    rows, sequences, models = fitter._load_bundle(bundle_dir)
    result = fitter._exon26_signature(rows, sequences, models)
    assert result["exon_length"] == 10
    assert result["discriminating_positions"] == [{"offset": 5, "a_base": "X", "b_base": "Y"}]
    assert result["n_a_records"] == 2
    assert result["n_b_records"] == 2


def test_exon26_signature_excludes_non_modal_length_records(tmp_path: Path):
    a_seq_ok = "N" * 9 + "AAAAAXAAAA" + "N" * 20
    a_seq_short = "N" * 9 + "AAAAAX" + "N" * 20  # 6bp exon, not the modal 10bp
    b_seq_ok = "N" * 9 + "AAAAAYAAAA" + "N" * 20
    records = [
        ("C4A", "IPD_A1", "C4A*asm1", a_seq_ok, _model((10, 19), None, "long")),
        ("C4A", "IPD_A2", "C4A*asm2", a_seq_short, _model((10, 15), None, "long")),
        ("C4B", "IPD_B1", "C4B*asm1", b_seq_ok, _model((10, 19), None, "long")),
        ("C4B", "IPD_B2", "C4B*asm2", b_seq_ok, _model((10, 19), None, "long")),
    ]
    bundle_dir = _write_bundle(tmp_path, records)
    rows, sequences, models = fitter._load_bundle(bundle_dir)
    result = fitter._exon26_signature(rows, sequences, models)
    assert result["n_a_records"] == 1
    assert result["n_excluded_wrong_length"] == 1


def test_intron9_signature_splits_by_form_and_computes_threshold(tmp_path: Path):
    seq = "N" * 200
    records = [
        ("C4A", "IPD_A1", "C4A*asm1", seq, _model(None, (10, 109), "long")),   # len 100
        ("C4B", "IPD_B1", "C4B*asm1", seq, _model(None, (10, 111), "long")),   # len 102
        ("C4A", "IPD_A2", "C4A*asm2", seq, _model(None, (10, 29), "short")),   # len 20
        ("C4B", "IPD_B2", "C4B*asm2", seq, _model(None, (10, 31), "short")),   # len 22
    ]
    bundle_dir = _write_bundle(tmp_path, records)
    rows, sequences, models = fitter._load_bundle(bundle_dir)
    result = fitter._intron9_signature(rows, models)
    assert result["long_length_median"] == 101.0
    assert result["short_length_median"] == 21.0
    assert result["length_threshold"] == 61.0
    assert result["n_long_records"] == 2
    assert result["n_short_records"] == 2
