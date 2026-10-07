"""Stage 1 -- per-gene copy number (DESIGN.md S1).

CN(gene) = 2 x depth(gene) / control_depth, where:

- depth(gene): aligned bases from Stage 0's sieve hits on the gene's
  target anchors, divided by the gene's mean anchor length
  (`stage0_depth_by_gene`). A read ambiguous between paralogues (e.g.
  HLA-H/K/T/U/W, >99% identical) counts toward each; per-read gene
  resolution scattered depth across the cluster on noise-level margins.
- control_depth: samtools depth over the segdup-free chr17 interval
  (`chr17_control.py`), computed once per sample by the caller.

CN is per gene, not pooled by genes.yaml `segment`: pooling made HLA-A
inherit HLA-H's copies (CN=3). C4A/C4B (`typing_module: structural`) are
handled by c4_cn.py / c4_chr17_cn.py. CN=0 is never a bare deletion
(invariant 4): `corroborated` is always False until the BAM-path
presence check exists.
"""
from __future__ import annotations

import csv
import gzip
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from assign import GeneAssignmentMatrix, best_gene_per_read

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_GENES_YAML = REPO_ROOT / "resources" / "genes.yaml"

sys.path.insert(0, str(REPO_ROOT / "input"))
from fastq import CandidateRead, DEFAULT_ANCHORS_META  # noqa: E402


@dataclass(frozen=True)
class GeneManifestEntry:
    gene: str
    gene_class: str  # "I" / "II" / "III"
    segment: str | None
    calibrator: str | bool  # "primary" | "secondary" | False
    low_cn_floor: float | None  # see genes.yaml's HLA-DRB3 entry for rationale
    cn_model: str | None  # "diploid_typical" | "variable_0_2" | "variable" | ...


def load_gene_manifest(path: Path = DEFAULT_GENES_YAML) -> dict[str, GeneManifestEntry]:
    """Genes eligible for per-gene CN: not excluded, not structural (C4).
    `segment`, `low_cn_floor` and `cn_model` are not read by estimate_cn()."""
    manifest = yaml.safe_load(path.read_text())
    out: dict[str, GeneManifestEntry] = {}
    for g in manifest["genes"]:
        if g.get("excluded") or g.get("typing_module") == "structural":
            continue
        out[g["gene"]] = GeneManifestEntry(
            gene=g["gene"],
            gene_class=g["class"],
            segment=g.get("segment"),
            calibrator=g.get("calibrator", False),
            low_cn_floor=g.get("low_cn_floor"),
            cn_model=g.get("cn_model"),
        )
    return out


@dataclass(frozen=True)
class GeneCN:
    gene: str
    gene_class: str
    depth: float
    calibrator_median: float | None  # chr17 control depth here; calibrator median in c4_cn.py
    cn_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    is_zero_call: bool
    corroborated: bool  # always False in v0 -- see module docstring


def depth_by_gene(matrices: dict[str, GeneAssignmentMatrix]) -> dict[str, float]:
    """Reads per gene from assign.best_gene_per_read(); a read tied between
    genes is split fractionally, so its total credit is 1.0."""
    counts: dict[str, float] = {}
    for gene_weights in best_gene_per_read(matrices).values():
        for gene, weight in gene_weights.items():
            counts[gene] = counts.get(gene, 0.0) + weight
    return counts


def _load_target_anchor_lengths(anchors_meta_path: Path) -> dict[str, dict[str, int]]:
    """gene -> {anchor_allele: length} for target-role anchors; rows without
    a `length` column are skipped."""
    lengths: dict[str, dict[str, int]] = {}
    with gzip.open(anchors_meta_path, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["role"] != "target":
                continue
            length_str = row.get("length")
            if length_str is None:
                continue
            lengths.setdefault(row["gene"], {})[row["allele"]] = int(length_str)
    return lengths


def stage0_depth_by_gene(
    reads: Sequence[CandidateRead],
    *,
    anchors_meta_path: Path = DEFAULT_ANCHORS_META,
) -> dict[str, float]:
    """Per-gene depth from Stage 0's sieve hits (no realignment).

    Per read and gene, keep the best hit (a read can hit several of a gene's
    near-identical anchors); aligned bases = coverage x anchor length. Sum
    per gene and divide by the mean length of the gene's target anchors."""
    if not reads:
        return {}
    anchor_lengths = _load_target_anchor_lengths(anchors_meta_path)

    aligned_bases: dict[str, float] = {}
    for r in reads:
        best_bases_by_gene: dict[str, float] = {}
        for h in r.hits:
            length = anchor_lengths.get(h.gene, {}).get(h.allele)
            if length is None:
                continue
            bases = h.coverage * length
            if bases > best_bases_by_gene.get(h.gene, -1.0):
                best_bases_by_gene[h.gene] = bases
        for gene, bases in best_bases_by_gene.items():
            aligned_bases[gene] = aligned_bases.get(gene, 0.0) + bases

    depth: dict[str, float] = {}
    for gene, bases in aligned_bases.items():
        lengths = list(anchor_lengths.get(gene, {}).values())
        if lengths:
            depth[gene] = bases / statistics.mean(lengths)
    return depth


def estimate_cn(
    reads: Sequence[CandidateRead],
    control_depth: float | None,
    *,
    genes_yaml: Path = DEFAULT_GENES_YAML,
    anchors_meta_path: Path = DEFAULT_ANCHORS_META,
) -> dict[str, GeneCN]:
    """Per-gene CN keyed by gene (see module docstring). `control_depth` is
    required so every caller is explicit about the normaliser."""
    manifest = load_gene_manifest(genes_yaml)
    depth = stage0_depth_by_gene(reads, anchors_meta_path=anchors_meta_path)

    results: dict[str, GeneCN] = {}
    for gene in sorted(manifest):
        entry = manifest[gene]
        gene_class = entry.gene_class
        gene_depth = depth.get(gene, 0)

        if not control_depth:
            # No control depth: CN is None, not a silent 0 (invariant 4).
            results[gene] = GeneCN(
                gene=gene, gene_class=gene_class,
                depth=gene_depth, calibrator_median=None, cn_estimate=None,
                ci_low=None, ci_high=None,
                is_zero_call=(gene_depth == 0), corroborated=False,
            )
            continue

        cn_estimate = 2.0 * gene_depth / control_depth

        # Poisson error propagation on the ratio; provisional until a
        # cohort-fitted noise model exists.
        if gene_depth > 0:
            rel_se = (1.0 / gene_depth + 1.0 / control_depth) ** 0.5
        else:
            rel_se = (1.0 / control_depth) ** 0.5
        results[gene] = GeneCN(
            gene=gene, gene_class=gene_class,
            depth=gene_depth, calibrator_median=float(control_depth),
            cn_estimate=cn_estimate,
            ci_low=cn_estimate * max(0.0, 1 - 1.96 * rel_se),
            ci_high=cn_estimate * (1 + 1.96 * rel_se),
            is_zero_call=(gene_depth == 0), corroborated=False,
        )

    return results
