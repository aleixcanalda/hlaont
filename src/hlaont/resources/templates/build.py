#!/usr/bin/env python3
"""Build the versioned HLA template pool (DESIGN.md S2d) from IPD-IMGT/HLA
plus assembly-derived templates.

Only "genomic" IPD records (both UTRs and >= 1 intron, DESIGN.md S2b) enter
the pool. Genes come from resources/genes.yaml: not excluded and not
`typing_module: structural` (C4).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hlaont.resources.templates.assembly_sources import (
    DEFAULT_BUNDLE_DIR,
    AssemblyGTFSource,
    BundledAssemblySource,
)
from hlaont.resources.templates.sources import (
    IPD_DEFAULT_TAG,
    IPDSource,
    TemplateRecord,
    cds_exon_numbers,
    cds_nucleotide_sequence,
)

# Schema history: 6 cds_catalogue gained partial + exon_numbers (and partial
# records); 5 meta gained gtf_template_allele; 4 meta gained
# nearest_ipd_match; 3 added ref.gene.cds_catalogue.tsv.gz.
POOL_SCHEMA_VERSION = 6


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


def load_assembly_extraction_genes(genes_yaml: Path) -> set[str]:
    """Genes with genes.yaml `assembly_extraction: gtf_coordinate`."""
    manifest = yaml.safe_load(genes_yaml.read_text())
    return {g["gene"] for g in manifest["genes"] if g.get("assembly_extraction") == "gtf_coordinate"}


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
            "source", "source_version", "description", "nearest_ipd_match",
            "gtf_template_allele"]
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(("\t".join(cols) + "\n").encode())
            for r in records:
                row = [r.gene, r.ipd_id, r.allele, str(r.fields),
                       str(len(r.sequence)), str(r.genomic), r.source,
                       r.source_version, r.description, r.nearest_ipd_match or "",
                       r.gtf_template_allele or ""]
                gz.write(("\t".join(row) + "\n").encode())


def write_gene_models_jsonl_gz(path: Path, records: list[TemplateRecord]) -> None:
    """One JSON object per allele with a gene model, keyed by ipd_id;
    coordinates 1-based inclusive in ref.gene.fa.gz numbering."""
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for r in records:
                if r.gene_model is None:
                    continue
                gm = r.gene_model
                obj = {
                    "gene": r.gene,
                    "ipd_id": r.ipd_id,
                    "allele": r.allele,
                    "length": len(r.sequence),
                    "codon_start": gm.codon_start,
                    "cds_exons": [list(x) for x in gm.cds_exons],
                    "utrs": [list(x) for x in gm.utrs],
                    "exons": [list(x) for x in gm.exons],
                    "introns": [list(x) for x in gm.introns],
                }
                gz.write((json.dumps(obj, sort_keys=True) + "\n").encode())


def write_cds_catalogue_tsv_gz(path: Path, records: list[TemplateRecord]) -> None:
    """CDS sequence of every allele with a parseable CDS (genomic or not),
    for `extension` detection against exon-only alleles. `partial` rows
    carry `exon_numbers` (transcript order; empty if unresolved) for the
    partial-extension check."""
    cols = ["gene", "ipd_id", "allele", "genomic", "partial", "fields",
            "cds_length", "cds_sequence", "exon_numbers"]
    n_skipped = 0
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(("\t".join(cols) + "\n").encode())
            for r in records:
                if r.gene_model is None or not r.gene_model.cds_exons:
                    n_skipped += 1
                    continue
                cds = cds_nucleotide_sequence(r.sequence, r.gene_model)
                exon_numbers = cds_exon_numbers(r.gene_model)
                exon_numbers_str = ",".join(str(n) for n in exon_numbers) if exon_numbers else ""
                row = [r.gene, r.ipd_id, r.allele, str(r.genomic), str(r.partial),
                       str(r.fields), str(len(cds)), cds, exon_numbers_str]
                gz.write(("\t".join(row) + "\n").encode())
    if n_skipped:
        print(f"[build] CDS catalogue: skipped {n_skipped} record(s) with no "
              f"parseable CDS (e.g. unannotated pseudogene ORFs)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    ap.add_argument("--genes-yaml", type=Path, default=here.parent / "genes.yaml")
    ap.add_argument("--raw-dir", type=Path, default=here / "raw")
    ap.add_argument("--out-dir", type=Path, default=here / "pool")
    ap.add_argument("--ipd-tag", default=IPD_DEFAULT_TAG,
                     help="Pinned IMGTHLA git tag (DESIGN.md: 'Pin it').")
    ap.add_argument("--ipd-dat", type=Path, default=None,
                     help="Use a local hla.dat instead of downloading.")
    ap.add_argument(
        "--hla-seq-dir", type=Path, default=None,
        help="Real per-individual phased assemblies (232 samples / 461 haplotypes, "
             "each a {sample}_{1,2}_HLA.gtf.gz + all_HLA.fasta.gz pair) used to "
             "extract templates for genes.yaml's assembly_extraction genes "
             "(HLA-N/P/T/U/W/Y/R) via a fresh AssemblyGTFSource parse. Optional: "
             "when omitted, falls back to --assembly-bundle-dir's checked-in "
             "snapshot of the same extraction, so this dataset never needs to be "
             "on disk just to get these seven genes. Pass this only to regenerate "
             "from a newer/larger hla_seq/ (then also rerun build_assembly_bundle.py "
             "to update the bundle other users get by default).",
    )
    ap.add_argument(
        "--assembly-bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR,
        help="Pre-extracted HLA-N/P/T/U/W/Y/R snapshot (build_assembly_bundle.py's "
             "output) used for these genes when --hla-seq-dir is not given.",
    )
    args = ap.parse_args(argv)

    target_genes, tier_by_gene = load_target_genes(args.genes_yaml)
    if not target_genes:
        sys.exit("[build] ERROR: empty target gene set from genes.yaml — loud stop, not a silent pass.")

    ipd = IPDSource(genomic_genes=target_genes, raw_dir=args.raw_dir,
                     tag=args.ipd_tag, dat_path=args.ipd_dat)
    sources: list = [ipd]

    gtf_genes = load_assembly_extraction_genes(args.genes_yaml)

    if args.hla_seq_dir is not None:
        if gtf_genes:
            sources.append(AssemblyGTFSource(genes=gtf_genes, hla_seq_dir=args.hla_seq_dir))
    elif gtf_genes:
        if args.assembly_bundle_dir.exists():
            sources.append(BundledAssemblySource(genes=gtf_genes, bundle_dir=args.assembly_bundle_dir))
        else:
            print(f"[build] WARNING: --hla-seq-dir not given and no bundle at "
                  f"{args.assembly_bundle_dir}; {sorted(gtf_genes)} get zero non-IPD "
                  f"templates this build.", file=sys.stderr)

    all_records: list[TemplateRecord] = []
    seen: set[tuple[str, str]] = set()  # (gene, sequence) -- first-source-wins dedup,
                                          # same pattern as resources/anchors/build.py's add()
    for src in sources:
        for r in src.records():
            key = (r.gene, r.sequence)
            if key in seen:
                continue
            seen.add(key)
            all_records.append(r)

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

    missing_model = [r.allele for r in all_records if r.gene_model is None]
    if missing_model:
        sys.exit(
            f"[build] ERROR: {len(missing_model)} genomic record(s) with no "
            f"parsed gene model (e.g. {missing_model[:5]}) — FT feature-table "
            f"parsing has drifted from a real IPD record shape; loud stop, "
            f"not a silently incomplete resources/templates/pool/ref.gene.models.jsonl.gz"
        )

    # Separate non-genomic parse (incl. partials) for the CDS catalogue.
    cds_source = IPDSource(genomic_genes=target_genes, raw_dir=args.raw_dir,
                            tag=args.ipd_tag, dat_path=args.ipd_dat, require_genomic=False,
                            drop_partial=False)
    cds_records = sorted(cds_source.records(), key=lambda r: (r.gene, r.allele, r.ipd_id))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = args.out_dir / "ref.gene.fa.gz"
    meta_path = args.out_dir / "ref.gene.meta.tsv.gz"
    models_path = args.out_dir / "ref.gene.models.jsonl.gz"
    cds_catalogue_path = args.out_dir / "ref.gene.cds_catalogue.tsv.gz"
    write_fasta_gz(fasta_path, all_records)
    write_meta_tsv_gz(meta_path, all_records)
    write_gene_models_jsonl_gz(models_path, all_records)
    write_cds_catalogue_tsv_gz(cds_catalogue_path, cds_records)

    manifest = {
        "pool_schema_version": POOL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "genes_yaml_manifest_version": yaml.safe_load(args.genes_yaml.read_text())["manifest_version"],
        "target_genes": sorted(target_genes),
        "total_records": len(all_records),
        "records_per_gene": dict(sorted(counts.items())),
        "cds_catalogue_records": len(cds_records),
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
