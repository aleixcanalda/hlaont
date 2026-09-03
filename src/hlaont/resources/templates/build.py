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
from assembly_sources import (  # noqa: E402
    DEFAULT_BUNDLE_DIR,
    AssemblyGTFSource,
    BundledAssemblySource,
)
from sources import (  # noqa: E402
    IPD_DEFAULT_TAG,
    IPDSource,
    TemplateRecord,
    cds_nucleotide_sequence,
)

POOL_SCHEMA_VERSION = 5  # 5: ref.gene.meta.tsv.gz gained gtf_template_allele (2026-09-03,
                          # user-requested: calls.tsv's template_allele column showed a raw
                          # content-hash accession, e.g. "HLA-W*asm3e9f0510aa", for the
                          # assembly-extracted genes -- this is the source hla_seq/ GTF's own
                          # gene-line template_allele attribute instead, e.g.
                          # "HLA-W*01:01:01:05") -- empty string for every IPD-sourced record.
                          # 4: gained nearest_ipd_match (assembly-derived genes' display naming,
                          # stages/novelty.py) -- empty string for every IPD-sourced record.
                          # 3: added ref.gene.cds_catalogue.tsv.gz (DESIGN.md S5 "extension"
                          # detection)


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
    """Genes with genes.yaml's `assembly_extraction: gtf_coordinate` field
    set (2026-08-22) -- read by build.py to route them through
    AssemblyGTFSource (real per-haplotype sequences extracted from
    hla_seq/'s GTF coordinates), never branched on by gene name in code
    (invariant 5)."""
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
    """One JSON object per allele, keyed by ipd_id (stable across nomenclature
    renames, unlike `allele`) -- DESIGN.md S5's novelty classifier needs the
    exact CDS/exon/intron/UTR coordinates of whichever template an allele was
    selected against (stages/genotype.py), not just genomic/non-genomic.
    Coordinates are 1-based inclusive, in this record's own sequence
    numbering (sources.py's GeneModel docstring) -- the same numbering
    ref.gene.fa.gz's sequence is in, so no coordinate translation is needed
    downstream. Records with no gene_model (should not occur for a genomic
    record, see build.py's own validation pass) are skipped, not written as
    a null placeholder -- absence is the loud signal, not a written-out None."""
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
    """Every non-partial allele's CDS nucleotide sequence, genomic or not --
    DESIGN.md S5's `extension` class ("CDS matches a known allele; IPD
    holds no full-length record. Not a novel allele.") needs to check a
    consensus's CDS against alleles IPD only ever recorded as exon-only,
    which never enter resources/templates/pool/ref.gene.fa.gz's genomic-only
    pool (invariant 7: getting `extension` vs `novel` wrong "is the mistake
    most likely to embarrass us"). Records with no parseable CDS (e.g. a
    pseudogene record with a disrupted/unannotated ORF) are skipped, not
    written with an empty sequence -- absence from the catalogue, not a
    false empty match, same "loud, not silent" stance as elsewhere here."""
    cols = ["gene", "ipd_id", "allele", "genomic", "fields", "cds_length", "cds_sequence"]
    n_skipped = 0
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(("\t".join(cols) + "\n").encode())
            for r in records:
                if r.gene_model is None or not r.gene_model.cds_exons:
                    n_skipped += 1
                    continue
                cds = cds_nucleotide_sequence(r.sequence, r.gene_model)
                row = [r.gene, r.ipd_id, r.allele, str(r.genomic), str(r.fields),
                       str(len(cds)), cds]
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

    # A separate, non-genomic-inclusive parse pass for the CDS catalogue
    # (DESIGN.md S5 `extension` detection) -- deliberately not folded into
    # the genomic-only `all_records` above (require_genomic=True there is
    # load-bearing for the pool itself, S2b), and deliberately not the same
    # IPDSource instance re-used with a flag flip, since .records() commits
    # to require_genomic at construction (this module's own existing
    # design, untouched here).
    cds_source = IPDSource(genomic_genes=target_genes, raw_dir=args.raw_dir,
                            tag=args.ipd_tag, dat_path=args.ipd_dat, require_genomic=False)
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
