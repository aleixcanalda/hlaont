#!/usr/bin/env python3
"""Stage per-gene candidate FASTAs for the anchor-set cd-hit-est rebuild, and
emit the exact commands to run it (DESIGN.md S2a/S9).

Two asset classes, kept separate here on purpose:

1. New target genes -- the 30 v0 target genes (tier 1-3, template_polish)
   NOT already covered as targets in FuFiHLA's ported 136-anchor set (the
   original six: A, B, C, DRB1, DQA1, DQB1). These need dense, tight-threshold
   clustering so a target gene's own divergent alleles can still find a close
   anchor -- DESIGN.md's "too sparse to reliably recruit divergent alleles
   once a gene becomes a target" critique of the 0.95 decoy threshold.
   Sequences come from the already-built template pool (genomic only).

2. New decoy genes -- tier-4 genes with IPD sequence but no entry at all in
   the ported 136 set (DRB2, DRB7 have exactly one non-partial record each;
   DRB6/DRB8/DRB9 have zero and are flagged unresolvable, not silently
   dropped). These stay at cd-hit-est's original decoy role: soak up stray
   reads, never typed. Single-sequence "clusters" don't need cd-hit at all,
   so they are written straight through and excluded from run_cdhit.sh.

The ported 136-anchor set itself (resources/anchors/vendor/136ref.fa.gz) is
NOT touched or re-clustered -- DESIGN.md: re-deriving it "wastes weeks and
yields something slightly different and worse."

cd-hit-est isn't available on the login node here (module load only, and the
login node shouldn't run CPU-heavy jobs). This script only prepares inputs
and prints/writes the commands; run_cdhit.sh is meant to run in an
interactive SLURM allocation, not here.
"""
from __future__ import annotations

import gzip
import sys
import zlib
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "resources" / "templates"))
from sources import IPDSource  # noqa: E402

GENES_YAML = REPO_ROOT / "resources" / "genes.yaml"
POOL_FASTA = REPO_ROOT / "resources" / "templates" / "pool" / "ref.gene.fa.gz"
RAW_DAT = REPO_ROOT / "resources" / "templates" / "raw" / "hla-v3.65.0-alpha.dat"
C4_BUNDLE_FASTA = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.fa.gz"

STAGING = HERE / "build_staging"
CANDIDATES_DIR = STAGING / "candidates"       # -> cd-hit-est input, one gene per file
SINGLETON_DIR = STAGING / "singletons"        # -> no clustering needed, copied straight through
CLUSTERED_DIR = STAGING / "clustered"         # -> cd-hit-est output lands here

# Tighter than FuFiHLA's 0.95 decoy threshold (DESIGN.md S2a/S9). Provisional:
# no ONT/HiFi recruitment experiment has validated this value yet -- it is a
# documented, versioned starting point for milestone 1/2 (DESIGN.md S6), not
# a fitted constant. Revisit before trusting the read-recruitment numbers.
TARGET_IDENTITY = 0.98
TARGET_WORD_SIZE = 10  # cd-hit-est: n=10-11 recommended for c in [0.95, 1.0]

FUFIHLA_PORTED_TARGETS = {"HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1"}


def read_gzip_first_member_text(path: Path) -> str:
    """FuFiHLA's 136ref.fa.gz has trailing bytes after the first gzip member
    (upstream artifact -- content-length matches exactly, so it's not a
    truncated download; GNU `gzip -t` tolerates it and warns, Python's gzip
    module raises on it). Decompress only the first member and stop there.
    """
    d = zlib.decompressobj(zlib.MAX_WBITS | 16)
    chunks = []
    with open(path, "rb") as fh:
        while not d.eof:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            chunks.append(d.decompress(chunk))
    return b"".join(chunks).decode("utf-8", errors="replace")


def load_manifest():
    return yaml.safe_load(GENES_YAML.read_text())["genes"]


def read_pool_fasta(path: Path) -> dict[str, list[tuple[str, str]]]:
    """gene -> list of (header, sequence) from the already-built template pool."""
    by_gene: dict[str, list[tuple[str, str]]] = {}
    header = None
    seq_lines: list[str] = []

    def flush():
        if header is None:
            return
        gene = header[1:].split(" ", 1)[0].split("*")[0]
        by_gene.setdefault(gene, []).append((header, "".join(seq_lines)))

    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        flush()
    return by_gene


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(header + "\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")


def main() -> int:
    manifest = load_manifest()
    by_gene_name = {g["gene"]: g for g in manifest}

    target_genes_v0 = {
        g["gene"] for g in manifest
        if not g.get("excluded") and g.get("typing_module") == "template_polish"
    }
    new_targets = sorted(target_genes_v0 - FUFIHLA_PORTED_TARGETS)

    # C4A/C4B (2026-08-25, DESIGN.md S2f): typing_module: structural, not
    # template_polish -- no star-allele template pool exists for them at all,
    # so their candidates can't come from POOL_FASTA the way every other
    # target gene's do. genes.yaml's own anchor_source: c4_bundle field
    # (data-driven, invariant 5 -- not a hardcoded "C4A"/"C4B" check) points
    # at resources/templates/build_c4_bundle.py's separate real hla_seq/
    # extraction instead. Same cd-hit-est clustering treatment either way --
    # C4's own real diversity (362 distinct sequences before clustering,
    # spanning both long/short form and the A/B split) is exactly the kind
    # of redundancy this clustering step exists to collapse for every other
    # target gene here.
    c4_targets = sorted(
        g["gene"] for g in manifest
        if not g.get("excluded") and g.get("anchor_source") == "c4_bundle"
    )

    print(f"[anchors] {len(new_targets)} new target genes need tight-threshold "
          f"clustering: {new_targets}", file=sys.stderr)
    print(f"[anchors] {len(c4_targets)} C4 gene(s) need clustering from the "
          f"c4_bundle source: {c4_targets}", file=sys.stderr)

    pool_by_gene = read_pool_fasta(POOL_FASTA)
    c4_bundle_by_gene: dict[str, list[tuple[str, str]]] = {}
    if C4_BUNDLE_FASTA.exists():
        c4_bundle_by_gene = read_pool_fasta(C4_BUNDLE_FASTA)
    elif c4_targets:
        print(f"[anchors] WARNING: {C4_BUNDLE_FASTA} missing -- run "
              f"resources/templates/build_c4_bundle.py first. {c4_targets} get "
              f"zero anchors this pass.", file=sys.stderr)

    cdhit_commands = []
    for gene in new_targets + c4_targets:
        source_by_gene = c4_bundle_by_gene if gene in c4_targets else pool_by_gene
        records = source_by_gene.get(gene, [])
        if not records:
            print(f"[anchors] WARNING: {gene} is a v0 target with zero genomic "
                  f"templates in its candidate source -- skipping, check the build.",
                  file=sys.stderr)
            continue
        in_path = CANDIDATES_DIR / f"{gene}.fa"
        out_path = CLUSTERED_DIR / f"{gene}.fa"
        write_fasta(in_path, records)
        if len(records) == 1:
            # cd-hit is a no-op on a single sequence; write straight through
            # and don't waste a cluster job on it.
            write_fasta(out_path, records)
            print(f"[anchors] {gene}: 1 candidate, no clustering needed", file=sys.stderr)
            continue
        cdhit_commands.append(
            f"cd-hit-est -i {in_path} -o {out_path} "
            f"-c {TARGET_IDENTITY} -n {TARGET_WORD_SIZE} -g 1 -d 0 -M 0 -T 12"
        )

    # New decoy genes: tier-4 genes absent from the ported 136 set but with
    # >=1 non-partial IPD record. Single-record genes need no clustering.
    ported_genes = set()
    vendor = REPO_ROOT / "resources" / "anchors" / "vendor" / "136ref.fa.gz"
    for line in read_gzip_first_member_text(vendor).splitlines():
        if line.startswith(">"):
            ported_genes.add(line[1:].split(" ", 1)[0].split("*")[0])

    tier4_genes = {g["gene"] for g in manifest if g.get("tier") == 4}
    missing_decoys = sorted(tier4_genes - ported_genes)
    print(f"[anchors] tier-4 genes with no ported decoy anchor: {missing_decoys}",
          file=sys.stderr)

    if missing_decoys:
        ipd = IPDSource(genomic_genes=set(missing_decoys), raw_dir=RAW_DAT.parent,
                         dat_path=RAW_DAT, require_genomic=False)
        decoy_records: dict[str, list[tuple[str, str]]] = {}
        for rec in ipd.records():
            header = f">{rec.allele} {rec.ipd_id} {rec.source}:{rec.source_version} {rec.description} len={len(rec.sequence)}bp"
            decoy_records.setdefault(rec.gene, []).append((header, rec.sequence))

        for gene in missing_decoys:
            recs = decoy_records.get(gene, [])
            if not recs:
                print(f"[anchors] {gene}: 0 non-partial records in IPD -- "
                      f"UNRESOLVABLE from IPD, needs a non-IPD source or stays undecoyed.",
                      file=sys.stderr)
                continue
            write_fasta(SINGLETON_DIR / f"{gene}.fa", recs)
            print(f"[anchors] {gene}: {len(recs)} candidate(s) written straight "
                  f"through (singleton decoy, no clustering)", file=sys.stderr)

    STAGING.mkdir(parents=True, exist_ok=True)
    script_path = STAGING / "run_cdhit.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "module load GCC/11.3.0\n"
        "module load CD-HIT/4.8.1\n"
        f"mkdir -p {CLUSTERED_DIR}\n\n"
        + "\n".join(cdhit_commands) + "\n"
    )
    script_path.chmod(0o755)

    print(f"\n[anchors] wrote {len(cdhit_commands)} cd-hit-est commands to {script_path}",
          file=sys.stderr)
    print("[anchors] run on an interactive node, e.g.:", file=sys.stderr)
    print("  sinteractive --time=01:00:00 --cpus-per-task=12 --mem=10GB", file=sys.stderr)
    print(f"  bash {script_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
