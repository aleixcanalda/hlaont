#!/usr/bin/env python3
"""Stage per-gene candidate FASTAs for the cd-hit-est anchor rebuild and
write the commands (DESIGN.md S2a/S9).

1. Target genes without ported anchors (all but A/B/C/DRB1/DQA1/DQB1):
   genomic pool sequences, clustered at TARGET_IDENTITY so divergent
   alleles still have a close anchor.
2. Tier-4 decoys absent from the ported set: DRB2/DRB7 (one record each,
   written straight through); DRB6/8/9 have none.

The ported 136 anchors are not re-clustered. Run run_cdhit.sh on a compute
node, not the login node.
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

# Tighter than FuFiHLA's 0.95 decoy threshold. Provisional (not validated).
TARGET_IDENTITY = 0.98
TARGET_WORD_SIZE = 10  # cd-hit-est: n=10-11 recommended for c in [0.95, 1.0]

FUFIHLA_PORTED_TARGETS = {"HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1"}


def read_gzip_first_member_text(path: Path) -> str:
    """Decompress only the first gzip member (136ref.fa.gz has trailing
    bytes that Python's gzip rejects)."""
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

    # C4A/C4B (genes.yaml anchor_source: c4_bundle) take candidates from the
    # C4 bundle instead of the template pool.
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
