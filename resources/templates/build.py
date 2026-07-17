#!/usr/bin/env python3
"""Build the versioned HLA template pool (DESIGN.md S2d).

Multi-source by design: today this wires up IPD-IMGT/HLA only. Immuannot
HPRC/CPC assemblies and MHC ALT haplotypes are added later as additional
TemplateSource instances in `sources`, below — not a refactor.

The pool is restricted to "genomic" records: both UTRs plus at least one
intron present (DESIGN.md S2b), matching ipd_audit.py's definition. This is
stricter than FuFiHLA's `grep -v partial`, which keeps exon-only records that
cannot template-and-polish a full gene.

The gene list comes from resources/genes.yaml (development notes invariant 5) — never
hardcode a gene list here. A gene qualifies for the IPD template pool if it is
not `excluded` and its `typing_module` is not `structural` (C4A/C4B route
through DESIGN.md S2f instead, not the IPD star-allele pool).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from sources import IPD_DEFAULT_TAG, IPDSource, TemplateRecord  # noqa: E402

POOL_SCHEMA_VERSION = 1


def load_target_genes(genes_yaml: Path) -> tuple[set[str], dict[str, int]]:
    manifest = yaml.safe_load(genes_yaml.read_text())
    target = set()
    tier_by_gene: dict[str, int] = {}
    for g in manifest["genes"]:
        if g.get("excluded"):
            continue
        if g.get("typing_module") == "structural":
            continue
        target.add(g["gene"])
        tier = g.get("tier")
        if isinstance(tier, int):
            tier_by_gene[g["gene"]] = tier
    return target, tier_by_gene


def write_fasta_gz(path: Path, records: list[TemplateRecord]) -> None:
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for r in records:
                header = (
                    f">{r.allele} {r.ipd_id} {r.source}:{r.source_version} "
                    f"{r.description} len={len(r.sequence)}bp\n"
                )
                gz.write(header.encode())
                seq = r.sequence
                for i in range(0, len(seq), 80):
                    gz.write(seq[i:i + 80].encode())
                    gz.write(b"\n")


def write_meta_tsv_gz(path: Path, records: list[TemplateRecord]) -> None:
    cols = ["gene", "ipd_id", "allele", "fields", "length", "genomic",
            "source", "source_version", "description"]
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(("\t".join(cols) + "\n").encode())
            for r in records:
                row = [r.gene, r.ipd_id, r.allele, str(r.fields),
                       str(len(r.sequence)), str(r.genomic), r.source,
                       r.source_version, r.description]
                gz.write(("\t".join(row) + "\n").encode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    ap.add_argument("--genes-yaml", type=Path, default=here.parent / "genes.yaml")
    ap.add_argument("--raw-dir", type=Path, default=here / "raw")
    ap.add_argument("--out-dir", type=Path, default=here / "pool")
    ap.add_argument("--ipd-tag", default=IPD_DEFAULT_TAG,
                     help="Pinned IMGTHLA git tag (DESIGN.md: 'Pin it').")
    ap.add_argument("--ipd-dat", type=Path, default=None,
                     help="Use a local hla.dat instead of downloading.")
    args = ap.parse_args()

    target_genes, tier_by_gene = load_target_genes(args.genes_yaml)
    if not target_genes:
        sys.exit("[build] ERROR: empty target gene set from genes.yaml — loud stop, not a silent pass.")

    ipd = IPDSource(genomic_genes=target_genes, raw_dir=args.raw_dir,
                     tag=args.ipd_tag, dat_path=args.ipd_dat)
    sources = [ipd]
    # Later: sources.append(ImmuannotSource(...)); sources.append(MHCAltSource(...))

    all_records: list[TemplateRecord] = []
    for src in sources:
        all_records.extend(src.records())

    all_records.sort(key=lambda r: (r.gene, r.allele, r.ipd_id))

    counts: dict[str, int] = {}
    for r in all_records:
        counts[r.gene] = counts.get(r.gene, 0) + 1

    for gene in sorted(target_genes):
        n = counts.get(gene, 0)
        tier = tier_by_gene.get(gene)
        if n == 0:
            if tier is not None and tier <= 3:
                print(f"[build] WARNING: {gene} (tier {tier}) has zero genomic "
                      f"templates — expected nonzero, check the source.", file=sys.stderr)
            else:
                print(f"[build] {gene}: 0 templates (expected — deferred, "
                      f"needs S2d augmentation)", file=sys.stderr)
        else:
            print(f"[build] {gene}: {n} templates", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = args.out_dir / "ref.gene.fa.gz"
    meta_path = args.out_dir / "ref.gene.meta.tsv.gz"
    write_fasta_gz(fasta_path, all_records)
    write_meta_tsv_gz(meta_path, all_records)

    manifest = {
        "pool_schema_version": POOL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "genes_yaml_manifest_version": yaml.safe_load(args.genes_yaml.read_text())["manifest_version"],
        "target_genes": sorted(target_genes),
        "total_records": len(all_records),
        "records_per_gene": dict(sorted(counts.items())),
        "sources": [
            {
                "name": src.name,
                "version": src.version,
                "dat_sha256": getattr(src, "dat_sha256", None),
                "tag": getattr(src, "tag", None),
            }
            for src in sources
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[build] wrote {len(all_records)} templates across {len(counts)} genes "
          f"to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
