#!/usr/bin/env python3
"""Extract HLA-N/P/T/U/W/Y/R templates from hla_seq/ (AssemblyGTFSource)
into the checked-in bundle assembly_bundle/bundle.{fa,meta.tsv,models.jsonl}.gz
used by build.py by default. Run only to regenerate the bundle.

    python build_assembly_bundle.py --hla-seq-dir /path/to/hla_seq
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from assembly_sources import (  # noqa: E402
    HLA_SEQ_SNAPSHOT_VERSION,
    AssemblyGTFSource,
    DEFAULT_BUNDLE_DIR,
)
from build import (  # noqa: E402
    load_assembly_extraction_genes,
    write_fasta_gz,
    write_gene_models_jsonl_gz,
    write_meta_tsv_gz,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    ap.add_argument("--hla-seq-dir", type=Path, required=True,
                     help="Real per-individual phased assemblies (each a "
                          "{sample}_{1,2}_HLA.gtf.gz + all_HLA.fasta.gz pair)")
    ap.add_argument("--genes-yaml", type=Path, default=here.parent / "genes.yaml")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    args = ap.parse_args(argv)

    gtf_genes = load_assembly_extraction_genes(args.genes_yaml)
    if not gtf_genes:
        sys.exit("[build_assembly_bundle] ERROR: no assembly_extraction genes in genes.yaml — nothing to bundle.")

    src = AssemblyGTFSource(genes=gtf_genes, hla_seq_dir=args.hla_seq_dir)
    records = sorted(src.records(), key=lambda r: (r.gene, r.allele, r.ipd_id))
    if not records:
        sys.exit(f"[build_assembly_bundle] ERROR: zero records extracted from {args.hla_seq_dir} "
                  f"for {sorted(gtf_genes)} — loud stop, not a silently empty bundle.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_fasta_gz(args.out_dir / "bundle.fa.gz", records)
    write_meta_tsv_gz(args.out_dir / "bundle.meta.tsv.gz", records)
    write_gene_models_jsonl_gz(args.out_dir / "bundle.models.jsonl.gz", records)

    counts: dict[str, int] = {}
    for r in records:
        counts[r.gene] = counts.get(r.gene, 0) + 1
    for gene in sorted(gtf_genes):
        n = counts.get(gene, 0)
        print(f"[build_assembly_bundle] {gene}: {n} records"
              + ("" if n else " (expected — no real hla_seq annotation for this gene yet)"),
              file=sys.stderr)

    manifest = {
        "hla_seq_snapshot_version": HLA_SEQ_SNAPSHOT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_hla_seq_dir_name": args.hla_seq_dir.name,  # basename only -- no local absolute path in a checked-in artifact
        "genes": sorted(gtf_genes),
        "records_per_gene": dict(sorted(counts.items())),
        "total_records": len(records),
        "provenance": "HPRC phased assemblies + CHM13/GRCh38/chimp references, Immuannot-annotated",
    }
    (args.out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[build_assembly_bundle] wrote {len(records)} records across {len(counts)} genes "
          f"to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
