#!/usr/bin/env python3
"""Extract C4 templates from hla_seq/ into assembly_bundle/c4.{fa,meta.tsv,
models.jsonl}.gz: the C4 alignment pool (stages/c4_cn.py), anchor
candidates (prepare_candidates.py) and exon 26 / intron 9 coordinates
(c4_form.py). Kept apart from the star-allele pool (DESIGN.md S2f).

Immuannot labels C4AL/C4AS/C4BL/C4BS; these are collapsed to genes C4A/C4B
(gene and allele renamed, form kept in `description`) to match genes.yaml.

    python build_c4_bundle.py --hla-seq-dir /path/to/hla_seq
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from hlaont.resources.templates.assembly_sources import HLA_SEQ_SNAPSHOT_VERSION, AssemblyGTFSource, DEFAULT_BUNDLE_DIR
from hlaont.resources.templates.build import write_fasta_gz, write_gene_models_jsonl_gz, write_meta_tsv_gz

# Real gene_name values in hla_seq/'s own GTFs (confirmed directly against
# the real dataset, not guessed) -- long/short form x A/B.
_C4_FORM_GENES = {"C4AL": "C4A", "C4AS": "C4A", "C4BL": "C4B", "C4BS": "C4B"}
_FORM_LABEL = {"C4AL": "long", "C4AS": "short", "C4BL": "long", "C4BS": "short"}


def _collapse_to_c4a_c4b(records):
    """Rename C4AL/AS/BL/BS records to C4A/C4B; form goes in description."""
    out = []
    for r in records:
        bucket = _C4_FORM_GENES[r.gene]
        form = _FORM_LABEL[r.gene]
        assert r.allele.startswith(r.gene), f"unexpected allele shape: {r.allele!r}"
        new_allele = bucket + r.allele[len(r.gene):]
        out.append(dataclasses.replace(
            r, gene=bucket, allele=new_allele,
            description=f"form={form} ({r.gene}); {r.description}",
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hla-seq-dir", type=Path, required=True,
                     help="Real per-individual phased assemblies (each a "
                          "{sample}_{1,2}_HLA.gtf.gz + all_HLA.fasta.gz pair)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_BUNDLE_DIR,
                     help="Same directory as build_assembly_bundle.py's NPTUWYR "
                          "bundle, but distinct filenames (c4.*) -- kept separate "
                          "so build.py's BundledAssemblySource (fixed bundle.* "
                          "filenames) never picks these up.")
    args = ap.parse_args(argv)

    src = AssemblyGTFSource(genes=set(_C4_FORM_GENES), hla_seq_dir=args.hla_seq_dir)
    raw_records = list(src.records())
    if not raw_records:
        sys.exit(f"[build_c4_bundle] ERROR: zero records extracted from {args.hla_seq_dir} "
                  f"for {sorted(_C4_FORM_GENES)} — loud stop, not a silently empty bundle.")

    records = sorted(_collapse_to_c4a_c4b(raw_records), key=lambda r: (r.gene, r.allele, r.ipd_id))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_fasta_gz(args.out_dir / "c4.fa.gz", records)
    write_meta_tsv_gz(args.out_dir / "c4.meta.tsv.gz", records)
    write_gene_models_jsonl_gz(args.out_dir / "c4.models.jsonl.gz", records)

    counts: dict[str, int] = {}
    for r in records:
        counts[r.gene] = counts.get(r.gene, 0) + 1
    forms: dict[str, int] = {}
    for raw in raw_records:
        forms[raw.gene] = forms.get(raw.gene, 0) + 1
    for gene in sorted(counts):
        print(f"[build_c4_bundle] {gene}: {counts[gene]} records", file=sys.stderr)

    manifest = {
        "hla_seq_snapshot_version": HLA_SEQ_SNAPSHOT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_hla_seq_dir_name": args.hla_seq_dir.name,
        "records_per_gene": dict(sorted(counts.items())),
        "records_per_real_form": dict(sorted(forms.items())),
        "total_records": len(records),
        "note": "gene/allele collapsed C4AL/C4AS->C4A, C4BL/C4BS->C4B (see script "
                "docstring); real form kept in each record's own description only.",
    }
    (args.out_dir / "c4_bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[build_c4_bundle] wrote {len(records)} records across {len(counts)} genes "
          f"to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
