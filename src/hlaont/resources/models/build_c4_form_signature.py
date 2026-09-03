#!/usr/bin/env python3
"""Fit resources/models/c4_form_signature.yaml from the real C4 bundle
(resources/templates/assembly_bundle/c4.{fa,meta.tsv,models.jsonl}.gz) --
DESIGN.md S2f's "own module" for C4A/C4B, Phase 2 (A vs B / long vs short
form). Invariant 10: models are fitted offline and shipped as versioned
parameter files, never hardcoded in a stage module.

**Why this can be data-derived at all, not literature-guessed.** DESIGN.md
S2f names Immuannot's real approach: A vs B by key exon 26 peptides, long vs
short by intron 9 length. hla_seq/'s own GTFs already carry Immuannot's own
A/B (C4AL/C4AS/C4BL/C4BS -> collapsed to C4A/C4B by build_c4_bundle.py) and
long/short (the "form=" tag in each record's own description) calls for 362
real per-haplotype sequences -- so the actual discriminating nucleotide
positions and intron 9 length distinction can be *measured* directly from
this project's own already-labelled real data, the same way every other
fitted model in this directory is derived, rather than copied from a paper's
stated genomic coordinates (which would also assume a specific reference
build's numbering -- invariant 2's spirit, even though this isn't a `@SQ`
coordinate in the literal sense).

**Exon 26 signature.** For every real record with a gene_model exon
numbered 26, extract that exon's own nucleotide substring (this sequence's
own local coordinates, `resources/templates/sources.py`'s GeneModel).
Records whose exon 26 isn't the modal length are excluded (logged, not
silently dropped) -- a length-changing indel there is a genuine rare-allele
case this signature isn't trying to model. Position-wise majority vote
gives a consensus sequence per gene bucket (A, B); positions where the two
consensuses disagree are the discriminating positions a real read can be
scored against.

**Intron 9 length.** For every real record with a gene_model intron
numbered 9 (the gap between exon 9 and exon 10), intron length grouped by
the record's own real form label (long/short, from build_c4_bundle.py's
description tag) gives the two length populations directly -- expected
around DESIGN.md's cited ~6.4kb HERV-K separation, confirmed rather than
assumed by printing both medians.

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

# Real, published C4A/C4B isotype-determining tetrapeptide (Yu 1991 et al.)
# -- used *only* as a printed sanity check against the data-derived
# consensus below, never as the source of truth for the shipped signature.
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

    # Exon 26 doesn't necessarily start on a codon boundary (an internal
    # exon inherits whatever frame the upstream exons left it in) -- this
    # sanity check doesn't need the exact frame from the full CDS
    # (informational only, never load-bearing), so it just tries all 3 and
    # reports whichever one turns up the published motif, rather than
    # assuming frame 0 and silently failing a real match on a frame error.
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
            "form labels from hla_seq/, not literature coordinates. See DESIGN.md S2f and "
            "the 2026-08-26 Phase 2 development notes."
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
