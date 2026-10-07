#!/usr/bin/env python3
"""Consolidate the anchor set (DESIGN.md S2a/S9) from:

1. FuFiHLA's 136 anchors, ported verbatim (resources/anchors/vendor/, MIT).
2. cd-hit-est representatives for target genes without ported anchors,
   including C4A/C4B from the C4 bundle (see prepare_candidates.py).
3. Singleton decoys for tier-4 genes with one non-partial IPD record (DRB2,
   DRB7).

Role comes from resources/genes.yaml (invariant 5); structural genes (C4)
are targets so Stage 0 recruits their reads. HLA-DRB6/8/9 have no anchor
(no non-partial IPD record).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import zlib
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent

GENES_YAML = REPO_ROOT / "resources" / "genes.yaml"
VENDOR_FASTA = HERE / "vendor" / "136ref.fa.gz"
STAGING = HERE / "build_staging"
CANDIDATES_DIR = STAGING / "candidates"
SINGLETON_DIR = STAGING / "singletons"
CLUSTERED_DIR = STAGING / "clustered"

OUT_FASTA = HERE / "anchors.fa.gz"
OUT_META = HERE / "anchors.meta.tsv.gz"
OUT_MANIFEST = HERE / "anchors_manifest.json"

TARGET_IDENTITY = 0.98
TARGET_WORD_SIZE = 10

KNOWN_UNRESOLVED = ["HLA-DRB6", "HLA-DRB8", "HLA-DRB9"]  # zero non-partial IPD records


def read_gzip_first_member_text(path: Path) -> str:
    """See prepare_candidates.py: 136ref.fa.gz has trailing bytes after the
    first gzip member (upstream artifact, not a truncated download)."""
    d = zlib.decompressobj(zlib.MAX_WBITS | 16)
    chunks = []
    with open(path, "rb") as fh:
        while not d.eof:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            chunks.append(d.decompress(chunk))
    return b"".join(chunks).decode("utf-8", errors="replace")


def parse_fasta_text(text: str):
    header = None
    seq_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_lines)
            header = line[1:]
            seq_lines = []
        elif header is not None:
            seq_lines.append(line.strip())
    if header is not None:
        yield header, "".join(seq_lines)


def parse_fasta_file(path: Path):
    yield from parse_fasta_text(path.read_text())


def allele_and_gene(header: str) -> tuple[str, str]:
    allele = header.split(" ", 1)[0]
    gene = allele.split("*")[0]
    return allele, gene


def cluster_size(clstr_path: Path, representative_allele: str) -> int:
    """Count members of the cluster whose representative (the '*' entry in
    cd-hit's .clstr) matches representative_allele."""
    if not clstr_path.exists():
        return 1
    text = clstr_path.read_text()
    for block in text.split(">Cluster ")[1:]:
        lines = block.strip().splitlines()[1:]
        members = [ln for ln in lines if ln.strip()]
        for ln in members:
            if ln.rstrip().endswith("*") and representative_allele in ln:
                return len(members)
    return 1


def main() -> int:
    manifest = yaml.safe_load(GENES_YAML.read_text())["genes"]
    role_by_gene = {}
    for g in manifest:
        if g.get("excluded"):
            continue
        # Targets: template_polish and structural (C4) genes.
        role_by_gene[g["gene"]] = (
            "target" if g.get("typing_module") in ("template_polish", "structural") else "decoy"
        )

    rows = []  # (gene, allele, role, source, cluster_size)
    seen: set[tuple[str, str]] = set()

    def add(header: str, source: str, cluster_size_val: int | None, seq: str):
        allele, gene = allele_and_gene(header)
        if gene not in role_by_gene:
            return  # not a manifest gene, or excluded -- not our concern here
        key = (gene, allele)
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "gene": gene, "allele": allele, "role": role_by_gene[gene],
            "source": source, "cluster_size": cluster_size_val, "sequence": seq,
        })

    # 1. Ported anchors; cluster_size unknown (no .clstr shipped), so None.
    n_ported = 0
    for header, seq in parse_fasta_text(read_gzip_first_member_text(VENDOR_FASTA)):
        add(header, "FuFiHLA-ported", None, seq)
        n_ported += 1

    # 2. cd-hit-est representatives for the 30 new target genes.
    n_new_target = 0
    target_genes_v0 = sorted(g for g, r in role_by_gene.items() if r == "target")
    fufihla_targets = {"HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1"}
    for gene in target_genes_v0:
        if gene in fufihla_targets:
            continue
        fa = CLUSTERED_DIR / f"{gene}.fa"
        clstr = CLUSTERED_DIR / f"{gene}.fa.clstr"
        if not fa.exists():
            print(f"[anchors] WARNING: expected {fa}, missing -- gene has no "
                  f"anchor from this pass.", file=sys.stderr)
            continue
        for header, seq in parse_fasta_file(fa):
            allele, _ = allele_and_gene(header)
            csize = cluster_size(clstr, allele)
            add(header, "cd-hit-derived", csize, seq)
            n_new_target += 1

    # 3. Singleton decoys for tier-4 genes absent from the ported set.
    n_new_decoy = 0
    if SINGLETON_DIR.exists():
        for fa in sorted(SINGLETON_DIR.glob("*.fa")):
            for header, seq in parse_fasta_file(fa):
                add(header, "ipd-singleton", 1, seq)
                n_new_decoy += 1

    rows.sort(key=lambda r: (r["role"], r["gene"], r["allele"]))

    def csize_str(v):
        return "unknown" if v is None else str(v)

    with open(OUT_FASTA, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for r in rows:
                gz.write(f">{r['allele']} role={r['role']} source={r['source']} "
                         f"cluster_size={csize_str(r['cluster_size'])}\n".encode())
                seq = r["sequence"]
                for i in range(0, len(seq), 80):
                    gz.write(seq[i:i + 80].encode())
                    gz.write(b"\n")

    with open(OUT_META, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(b"gene\tallele\trole\tsource\tcluster_size\tlength\n")
            for r in rows:
                gz.write(f"{r['gene']}\t{r['allele']}\t{r['role']}\t{r['source']}\t"
                          f"{csize_str(r['cluster_size'])}\t{len(r['sequence'])}\n".encode())

    counts_per_gene: dict[str, dict[str, int]] = {}
    for r in rows:
        d = counts_per_gene.setdefault(r["gene"], {"target": 0, "decoy": 0})
        d[r["role"]] += 1

    covered_genes = set(counts_per_gene)
    active_genes = set(target_genes_v0)
    missing_target_anchors = sorted(active_genes - covered_genes)

    manifest_out = {
        "sources": {
            "FuFiHLA-ported": {
                "records": n_ported,
                "provenance": "github.com/jingqing-hu/FuFiHLA v0.2.3, "
                               "share/fufihla/dps-dat/136ref.fa.gz, MIT licensed "
                               "(see vendor/FUFIHLA_LICENSE). Not re-derived.",
            },
            "cd-hit-derived": {
                "records": n_new_target,
                "params": f"cd-hit-est -c {TARGET_IDENTITY} -n {TARGET_WORD_SIZE} -g 1",
                "note": "Provisional threshold, not yet validated against a real "
                        "ONT/HiFi recruitment experiment (DESIGN.md S6).",
            },
            "ipd-singleton": {"records": n_new_decoy},
        },
        "records_per_gene": counts_per_gene,
        "total_records": len(rows),
        "known_unresolved_genes": KNOWN_UNRESOLVED,
        "unresolved_reason": "Zero non-partial IPD records; absent from the "
                              "FuFiHLA-ported set too. No anchor exists for these "
                              "genes -- needs a non-IPD source (DESIGN.md S2d).",
        "missing_target_anchors": missing_target_anchors,
        "c4a_c4b_note": "2026-08-25: C4A/C4B anchors are now built (cd-hit-derived, "
                         "sourced from resources/templates/assembly_bundle/c4.fa.gz, "
                         "real hla_seq/ per-haplotype extraction -- see "
                         "prepare_candidates.py). They still have no star-allele "
                         "template pool (DESIGN.md S2f) -- these anchors exist purely "
                         "to recruit reads for stages/c4_cn.py's depth-ratio total CN, "
                         "not for typing.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")

    if missing_target_anchors:
        print(f"[anchors] WARNING: v0 target genes with zero anchors: "
              f"{missing_target_anchors}", file=sys.stderr)

    print(f"[anchors] {len(rows)} total anchors "
          f"({n_ported} ported + {n_new_target} cd-hit + {n_new_decoy} singleton) "
          f"across {len(counts_per_gene)} genes -> {OUT_FASTA}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
