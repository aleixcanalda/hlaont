"""C4A/C4B total copy number by depth ratio (DESIGN.md S2f, Phase 1).

C4 has no star-allele pool, so the Stage 0 reads are aligned a second time
(build_gene_matrices) against the C4 bundle
(resources/templates/assembly_bundle/c4.*; long/short forms collapsed to
C4A/C4B). Total depth = reads resolved to C4A or C4B, deduped. CN uses
cn.py's 2 x depth / calibrator median formula with class I's calibrator
panel (nearest to RCCX; not validated). No A/B depth split is reported:
that bucket is dominated by noise outside the isotype bases (see
c4_form.py).

Length-inflation correction (when `reads_by_id` is given): a read count over
a gene body overstates per-base depth by (L + R) / R for target length L
and mean read length R. C4 (~20.6kb) gives ~1.69 vs 1.09-1.33 for the
calibrators; correcting both sides moved HG00097 from 5.32 to ~4 (truth 4).
Falls back to factor 1.0 when lengths are unavailable. Provisional.
"""
from __future__ import annotations

import csv
import gzip
import statistics
from pathlib import Path

from hlaont.stages.assign import DEFAULT_POOL_META, GeneAssignmentMatrix, best_gene_per_read
from hlaont.stages.cn import DEFAULT_GENES_YAML, GeneCN, depth_by_gene, load_gene_manifest

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_C4_POOL_FASTA = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.fa.gz"
DEFAULT_C4_POOL_META = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.meta.tsv.gz"

C4_GENES = ("C4A", "C4B")
C4_REPORTED_GENE = "C4"  # the combined-segment key this module writes into SampleResult.gene_cn
_CALIBRATOR_CLASS = "I"  # see module docstring for why class I, not class II
_CALIBRATOR_ROLES = ("primary", "secondary")


def c4_total_depth(c4_matrices: dict[str, GeneAssignmentMatrix]) -> int:
    """Distinct reads resolved to C4A or C4B (one entry per read in
    best_gene_per_read, so no double count)."""
    return len(best_gene_per_read(c4_matrices))


def _median_template_length(meta_path: Path, genes: set[str] | None = None) -> dict[str, float]:
    """Per-gene median template length from the pool meta `length` column;
    rows without one are skipped (no correction)."""
    lengths_by_gene: dict[str, list[int]] = {}
    with gzip.open(meta_path, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            gene = row.get("gene")
            length_str = row.get("length")
            if gene is None or length_str is None:
                continue
            if genes is not None and gene not in genes:
                continue
            lengths_by_gene.setdefault(gene, []).append(int(length_str))
    return {g: statistics.median(ls) for g, ls in lengths_by_gene.items()}


def _mean_read_length_by_gene(
    matrices: dict[str, GeneAssignmentMatrix], reads_by_id: dict[str, str],
) -> dict[str, float]:
    """Mean full read length of reads resolved to each gene; a tied read
    counts fully toward each gene."""
    lengths_by_gene: dict[str, list[int]] = {}
    for read_id, gene_weights in best_gene_per_read(matrices).items():
        seq = reads_by_id.get(read_id)
        if seq is None:
            continue
        for gene in gene_weights:
            lengths_by_gene.setdefault(gene, []).append(len(seq))
    return {g: statistics.mean(ls) for g, ls in lengths_by_gene.items()}


def _mean_length_of_assigned_reads(
    matrices: dict[str, GeneAssignmentMatrix], reads_by_id: dict[str, str],
) -> float | None:
    """Mean read length over all genes in `matrices` pooled (C4A + C4B)."""
    lengths = [len(reads_by_id[rid]) for rid in best_gene_per_read(matrices) if rid in reads_by_id]
    return statistics.mean(lengths) if lengths else None


def _length_inflation_factor(target_length: float | None, mean_read_length: float | None) -> float:
    """(L + R) / R; 1.0 if either input is missing or non-positive."""
    if not target_length or not mean_read_length or mean_read_length <= 0:
        return 1.0
    return (target_length + mean_read_length) / mean_read_length


def calibrator_genes_for_c4(genes_yaml: Path = DEFAULT_GENES_YAML) -> list[str]:
    """Class I calibrator genes used to normalise C4 depth (also used by
    run_sample_phased() to check per-gene phasing)."""
    manifest = load_gene_manifest(genes_yaml)
    return sorted(
        g for g, e in manifest.items()
        if e.gene_class == _CALIBRATOR_CLASS and e.calibrator in _CALIBRATOR_ROLES
    )


def estimate_c4_cn(
    c4_matrices: dict[str, GeneAssignmentMatrix],
    classical_matrices: dict[str, GeneAssignmentMatrix],
    reads_by_id: dict[str, str] | None = None,
    *,
    genes_yaml: Path = DEFAULT_GENES_YAML,
    c4_pool_meta: Path = DEFAULT_C4_POOL_META,
    classical_pool_meta: Path = DEFAULT_POOL_META,
) -> GeneCN:
    """Total C4 CN keyed C4_REPORTED_GENE. `classical_matrices` supplies
    calibrator depths; `c4_matrices` are the C4 pool alignments.
    `reads_by_id` enables the length correction; `depth` and
    `calibrator_median` stay uncorrected, only the CN and CI change."""
    depth = c4_total_depth(c4_matrices)

    calibrator_genes = calibrator_genes_for_c4(genes_yaml)
    classical_depth = depth_by_gene(classical_matrices)
    calibrator_depths = [classical_depth[g] for g in calibrator_genes if classical_depth.get(g, 0) > 0]

    if not calibrator_depths:
        # No calibrator signal this run -- cannot normalise. Loud (None CN),
        # not a silent 0, matching stages/cn.py's own convention.
        return GeneCN(
            gene=C4_REPORTED_GENE, gene_class="III",
            depth=depth, calibrator_median=None, cn_estimate=None,
            ci_low=None, ci_high=None,
            is_zero_call=(depth == 0), corroborated=False,
        )

    calibrator_median = statistics.median(calibrator_depths)

    if reads_by_id is None:
        normalized_c4_depth = float(depth)
        normalized_calibrator_median = float(calibrator_median)
    else:
        calibrator_template_lengths = _median_template_length(classical_pool_meta, set(calibrator_genes))
        calibrator_read_lengths = _mean_read_length_by_gene(classical_matrices, reads_by_id)
        normalized_calibrator_depths = []
        for g in calibrator_genes:
            raw = classical_depth.get(g, 0)
            if raw <= 0:
                continue
            f = _length_inflation_factor(calibrator_template_lengths.get(g), calibrator_read_lengths.get(g))
            normalized_calibrator_depths.append(raw / f)
        normalized_calibrator_median = statistics.median(normalized_calibrator_depths)

        c4_lengths_by_gene = _median_template_length(c4_pool_meta, set(C4_GENES))
        c4_target_length = statistics.median(c4_lengths_by_gene.values()) if c4_lengths_by_gene else None
        c4_mean_read_length = _mean_length_of_assigned_reads(c4_matrices, reads_by_id)
        c4_f = _length_inflation_factor(c4_target_length, c4_mean_read_length)
        normalized_c4_depth = depth / c4_f

    cn_estimate = 2.0 * normalized_c4_depth / normalized_calibrator_median

    # Poisson CI on the raw (uncorrected) counts.
    if depth > 0:
        rel_se = (1.0 / depth + 1.0 / calibrator_median) ** 0.5
    else:
        rel_se = (1.0 / calibrator_median) ** 0.5

    return GeneCN(
        gene=C4_REPORTED_GENE, gene_class="III",
        depth=depth, calibrator_median=float(calibrator_median),
        cn_estimate=cn_estimate,
        ci_low=cn_estimate * max(0.0, 1 - 1.96 * rel_se),
        ci_high=cn_estimate * (1 + 1.96 * rel_se),
        is_zero_call=(depth == 0), corroborated=False,
    )
