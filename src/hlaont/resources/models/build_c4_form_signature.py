#!/usr/bin/env python3
"""Fit resources/models/c4_form_signature.yaml from the C4 bundle
(invariant 10).

Exon 26: per-position majority consensus for A and B records (modal-length
exons only); positions where they differ are the discriminating positions.
Intron 9: length distribution per long/short form label (expected ~6.4kb
apart, the HERV-K insertion).

    python build_c4_form_signature.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import pysam
import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DEFAULT_BUNDLE_DIR = REPO_ROOT / "resources" / "templates" / "assembly_bundle"
DEFAULT_OUT_PATH = HERE / "c4_form_signature.yaml"

_EXON26_NUMBER = 26
_INTRON9_EXON_BEFORE = 9
_FORM_RE = re.compile(r"^form=(long|short) \(")

# Published isotype peptide (Yu 1991): a printed sanity check only.
_KNOWN_A_MOTIF = "PCPVLD"
_KNOWN_B_MOTIF = "LSPVIH"

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _translate(nt: str) -> str:
    return "".join(_CODON_TABLE.get(nt[i:i + 3], "X") for i in range(0, len(nt) - len(nt) % 3, 3))


def _load_bundle(bundle_dir: Path) -> tuple[list[dict], dict[str, str], dict[str, dict]]:
    meta_path = bundle_dir / "c4.meta.tsv.gz"
    fasta_path = bundle_dir / "c4.fa.gz"
    models_path = bundle_dir / "c4.models.jsonl.gz"
    for p in (meta_path, fasta_path, models_path):
        if not p.exists():
            sys.exit(f"[build_c4_form_signature] ERROR: missing {p} -- run "
                      f"build_c4_bundle.py first (writes c4.models.jsonl.gz as of "
                      f"2026-08-26).")

    with gzip.open(meta_path, "rt") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    sequences: dict[str, str] = {}
    with pysam.FastxFile(str(fasta_path)) as fh:
        for rec in fh:
            if rec.name is None or rec.sequence is None:
                raise ValueError(f"{fasta_path}: record with no name/sequence")
            sequences[rec.name] = rec.sequence

    models: dict[str, dict] = {}
    with gzip.open(models_path, "rt") as fh:
        for line in fh:
            obj = json.loads(line)
            models[obj["ipd_id"]] = obj

    return rows, sequences, models


def _exon26_signature(rows: list[dict], sequences: dict[str, str], models: dict[str, dict]) -> dict:
    by_gene: dict[str, list[str]] = collections.defaultdict(list)
    n_no_exon26 = 0
    n_wrong_length = 0
    lengths: list[int] = []
    extracted: list[tuple[str, str]] = []  # (gene, exon26_seq)

    for row in rows:
        model = models.get(row["ipd_id"])
        if model is None:
            continue
        exon26 = next((e for e in model["exons"] if e[0] == _EXON26_NUMBER), None)
        if exon26 is None:
            n_no_exon26 += 1
            continue
        _, s, e = exon26
        seq = sequences[row["allele"]][s - 1:e]
        lengths.append(len(seq))
        extracted.append((row["gene"], seq))

    if not lengths:
        sys.exit("[build_c4_form_signature] ERROR: zero records with an exon 26 -- "
                 "cannot derive an A-vs-B signature.")

    modal_length = collections.Counter(lengths).most_common(1)[0][0]
    for gene, seq in extracted:
        if len(seq) != modal_length:
            n_wrong_length += 1
            continue
        by_gene[gene].append(seq)

    if "C4A" not in by_gene or "C4B" not in by_gene:
        sys.exit(f"[build_c4_form_signature] ERROR: modal-length exon 26 records missing "
                 f"for one of C4A/C4B (have: {sorted(by_gene)}) -- cannot derive a "
                 f"two-sided signature.")

    def consensus(seqs: list[str]) -> str:
        return "".join(
            collections.Counter(s[i] for s in seqs).most_common(1)[0][0]
            for i in range(modal_length)
        )

    consensus_a = consensus(by_gene["C4A"])
    consensus_b = consensus(by_gene["C4B"])
    positions = [
        {"offset": i, "a_base": consensus_a[i], "b_base": consensus_b[i]}
        for i in range(modal_length) if consensus_a[i] != consensus_b[i]
    ]

    # Exon 26's reading frame is unknown here; try all three.
    found_frame = None
    for frame in (0, 1, 2):
        pa, pb = _translate(consensus_a[frame:]), _translate(consensus_b[frame:])
        if _KNOWN_A_MOTIF in pa and _KNOWN_B_MOTIF in pb:
            found_frame = frame
            break
    print(f"[build_c4_form_signature] exon26: modal_length={modal_length}bp, "
          f"{len(by_gene['C4A'])} C4A + {len(by_gene['C4B'])} C4B records used "
          f"({n_wrong_length} excluded for non-modal length, {n_no_exon26} with no "
          f"exon 26 at all), {len(positions)} discriminating nt position(s)", file=sys.stderr)
    if found_frame is not None:
        print(f"[build_c4_form_signature] sanity check PASSED at frame offset {found_frame}: "
              f"data-derived consensus contains the published Yu 1991 isotype tetrapeptide "
              f"for both forms.", file=sys.stderr)
    else:
        print(f"[build_c4_form_signature] sanity check DID NOT find the published Yu 1991 "
              f"motif in any of the 3 frames (A={_translate(consensus_a)!r} "
              f"B={_translate(consensus_b)!r} at frame 0) -- not fatal (the discriminating "
              f"*positions* below are measured directly from real labelled data, not from "
              f"this literature check), but flagged for a human to look at.", file=sys.stderr)

    return {
        "exon_number": _EXON26_NUMBER,
        "exon_length": modal_length,
        "discriminating_positions": positions,
        "n_a_records": len(by_gene["C4A"]),
        "n_b_records": len(by_gene["C4B"]),
        "n_excluded_wrong_length": n_wrong_length,
        "n_excluded_no_exon26": n_no_exon26,
    }


def _intron9_signature(rows: list[dict], models: dict[str, dict]) -> dict:
    lengths_by_form: dict[str, list[int]] = collections.defaultdict(list)
    n_no_intron9 = 0
    n_no_form_tag = 0

    for row in rows:
        model = models.get(row["ipd_id"])
        if model is None:
            continue
        intron9 = next((iv for iv in model["introns"] if iv[0] == _INTRON9_EXON_BEFORE), None)
        if intron9 is None:
            n_no_intron9 += 1
            continue
        m = _FORM_RE.match(row["description"])
        if m is None:
            n_no_form_tag += 1
            continue
        _, s, e = intron9
        lengths_by_form[m.group(1)].append(e - s + 1)

    if "long" not in lengths_by_form or "short" not in lengths_by_form:
        sys.exit(f"[build_c4_form_signature] ERROR: intron 9 lengths missing for one of "
                 f"long/short (have: {sorted(lengths_by_form)}) -- cannot derive a "
                 f"length threshold.")

    long_median = statistics.median(lengths_by_form["long"])
    short_median = statistics.median(lengths_by_form["short"])
    threshold = (long_median + short_median) / 2
    print(f"[build_c4_form_signature] intron9: long median={long_median}bp "
          f"(n={len(lengths_by_form['long'])}), short median={short_median}bp "
          f"(n={len(lengths_by_form['short'])}), threshold={threshold}bp, "
          f"observed HERV-K delta={long_median - short_median}bp "
          f"({n_no_intron9} with no intron 9, {n_no_form_tag} with no form tag)", file=sys.stderr)

    return {
        "exon_before": _INTRON9_EXON_BEFORE,
        "long_length_median": long_median,
        "short_length_median": short_median,
        "length_threshold": threshold,
        "n_long_records": len(lengths_by_form["long"]),
        "n_short_records": len(lengths_by_form["short"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = ap.parse_args(argv)

    rows, sequences, models = _load_bundle(args.bundle_dir)
    exon26 = _exon26_signature(rows, sequences, models)
    intron9 = _intron9_signature(rows, models)

    doc = {
        "version": 1,
        "source": (
            "Fitted from resources/templates/assembly_bundle/c4.{fa,meta.tsv,models.jsonl}.gz "
            "(build_c4_form_signature.py) -- real per-haplotype Immuannot A/B and long/short "
            "form labels from hla_seq/, not literature coordinates. See DESIGN.md S2f."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exon26": exon26,
        "intron9": intron9,
    }
    args.out.write_text(yaml.dump(doc, sort_keys=False, default_flow_style=False))
    print(f"[build_c4_form_signature] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
