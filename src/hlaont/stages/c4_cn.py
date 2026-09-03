"""C4A/C4B total copy number (DESIGN.md S2f) -- Phase 1 only, user-scoped
2026-08-25: "total CN via depth ratio", not the full three-part module.

DESIGN.md S2f is explicit that C4A/C4B have no star-allele nomenclature and
must not be routed through stages/assign.py's ordinary per-gene template
pool (S2a-2e) -- "there is no template pool to expand." stages/cn.py's own
docstring already excludes `typing_module: structural` genes for the same
reason. This module is the "own module" S2f calls for.

**Mechanism.** resources/templates/build_c4_bundle.py extracts real C4A/C4B
gene bodies (both long and short form) from hla_seq/'s real per-haplotype
GTFs into a small, separate, non-nomenclature pool
(resources/templates/assembly_bundle/c4.{fa,meta.tsv}.gz) -- gene C4A/C4B
only, the real long/short form collapsed away at that step (see that
script's own docstring for why). resources/anchors/ carries cd-hit-derived
target-role anchors for both genes (2 representatives each, one per real
form -- confirmed directly: cd-hit-est's own 0.98-identity clustering split
each gene into exactly a long-form and a short-form representative, since
the ~6.4kb HERV-K intron-9 insertion is a length difference far outside
that threshold), so Stage 0's sieve recruits real C4 reads instead of
silently dropping them as off-target (resources/anchors/build.py's role
assignment now treats `typing_module: structural` as target, not decoy).
stages/assign.build_gene_matrices() is reused completely unchanged, called
a second time on the same Stage 0-sieved reads but against this separate C4
pool -- exactly DESIGN.md S1's "second readout" idea, just pointed at a
different reference set than the classical genes' star-allele pool.

**Total depth** is every read `best_gene_per_read()` resolves to *either*
C4A or C4B's matrix (deduped by read_id, same mechanism stages/cn.py uses
for every other gene) -- the segment-level RCCX-module read count DESIGN.md
S2f's "Total C4 CN" bullet means. This is deliberately not a per-gene (A vs
B) count; see "Not attempted" below for why that split is not reported.

**Calibration.** Reuses class I's existing calibrator panel (E/F primary,
A/B/C secondary, resources/genes.yaml) and stages/cn.py's exact CN formula
(2 x depth / calibrator median) plus Poisson-based CI propagation, for
reporting consistency with every other gene's GeneCN. DESIGN.md S2f itself
just says "the calibrator panel" without naming a class -- class I is
chosen here because it is the closer physical neighbour of the class III
RCCX module in the real MHC (chromosome order is class II - class III -
class I; HLA-B sits immediately adjacent to the complement gene cluster),
so its local coverage is the more plausible depth proxy of the two existing
panels. This is a flagged design choice (invariant 11), not a DESIGN.md
instruction -- not validated against a real depth-correlation check.

**Not attempted here -- real follow-up work, not silently solved:**
- A vs B discrimination and long vs short form calling from spanning reads
  are now built (stages/c4_form.py, DESIGN.md S2f's other two named
  sub-problems, added 2026-08-26) -- but as separate, orthogonal read
  tallies alongside this module's total-CN estimate, not folded into it.
  This module still never reports a per-C4A/C4B *depth* split (the raw
  matrix-level A-vs-B bucket a read's best overall alignment lands in is
  dominated by noise elsewhere in the ~20kb template, not the handful of
  real isotype bases -- stages/c4_form.py's own docstring has the full
  reasoning for why that bucket is never trusted as a discrimination
  signal).

**Length-inflation correction, added 2026-08-26, user-directed after a
real HG00097 root-cause investigation (the development notes has the
full derivation).** `depth` (this module's and stages/cn.py's shared
metric) is a *distinct-read count* across a gene's whole template pool, not
a per-base pileup. That equivalence -- "read count over the whole gene body
tracks local per-base depth" -- only holds when the gene body is short
relative to the sequencing read length, which is true for every classical
calibrator gene (~3-4kb genomic templates) but false for C4 (~20.6kb bundle
records, comparable to or longer than a large fraction of this project's
own ONT read lengths). A locus that long needs proportionally more distinct
reads to tile across it at a *fixed* true per-base depth, independent of
real copy number -- confirmed on real HG00097 data: C4's own recruited-read
mean length (~29.8kb) combined with its ~20.6kb template gives an inflation
factor `(target_length + mean_read_length) / mean_read_length` of ~1.69,
versus ~1.09-1.33 for the five classical calibrator genes; correcting for
this (dividing each gene's raw depth by its own factor before the ratio)
moved C4's estimate from 5.32 to ~3.8-4.2 against a real ground truth of 4,
without touching stages/cn.py's own formula (used by all 40+ other genes,
whose own factors are much closer to each other and to 1 -- DESIGN.md
frames C4 as "its own module", and this correction stays scoped to it
rather than risking every other gene's already-shipped, tested CN
behaviour on a single sample's worth of validation). `_length_inflation_factor()`
is the shared helper; it is a physically-derived geometric correction (not
a fitted/guessed constant -- invariant 10 doesn't apply), but is still
provisional (invariant 11: one real sample, not a cohort) and falls back to
a no-op (factor 1.0, i.e. today's original formula) whenever real
length/read data isn't available -- every pre-existing caller/test that
doesn't supply `reads_by_id` gets the untouched original behaviour.
"""
from __future__ import annotations

import csv
import gzip
import statistics
from pathlib import Path

from assign import DEFAULT_POOL_META, GeneAssignmentMatrix, best_gene_per_read
from cn import DEFAULT_GENES_YAML, GeneCN, depth_by_gene, load_gene_manifest

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_C4_POOL_FASTA = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.fa.gz"
DEFAULT_C4_POOL_META = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.meta.tsv.gz"

C4_GENES = ("C4A", "C4B")
C4_REPORTED_GENE = "C4"  # the combined-segment key this module writes into SampleResult.gene_cn
_CALIBRATOR_CLASS = "I"  # see module docstring for why class I, not class II
_CALIBRATOR_ROLES = ("primary", "secondary")


def c4_total_depth(c4_matrices: dict[str, GeneAssignmentMatrix]) -> int:
    """Reads assigned to *either* C4A or C4B's own matrix, deduped by
    read_id -- best_gene_per_read() returns exactly one dict entry per
    read_id regardless of how its 1.0 depth credit is split across genes
    (2026-08-31 margin-gated resolution, that function's own docstring), so
    counting its keys is still correct, not a double-count of a read that
    hit both."""
    return len(best_gene_per_read(c4_matrices))


def _median_template_length(meta_path: Path, genes: set[str] | None = None) -> dict[str, float]:
    """Per-gene median template length from a pool's own meta.tsv.gz
    `length` column (already computed by resources/templates/build.py's
    write_meta_tsv_gz for every real pool this project ships) -- a real,
    directly-measured property of the reference sequences, not derived
    from alignment behaviour. A row with no `length` column at all (every
    synthetic test pool's minimal `gene\tallele`-only meta.tsv, since this
    correction was retrofitted onto callers/fixtures that predate it) is
    skipped, not a crash -- same "opportunistic when real data supports
    it, otherwise the untouched original formula" contract as the missing-
    `reads_by_id`/missing-gene-row fallbacks elsewhere in this module,
    not the "silent failure worse than a crash" case invariant 4 warns
    about (no CN gets misreported here either way -- this only ever
    affects whether the length correction itself fires)."""
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
    """Mean full-read length (not aligned-block length) of the reads
    best_gene_per_read() resolves to each gene -- the same read set
    depth_by_gene() counts. A read margin-gated (tied) across more than one
    gene contributes its full length to each tied gene's own list,
    unweighted by its fractional depth credit -- this function estimates a
    length *distribution* per gene, not a depth total, so a tied read is
    still real information about that gene's own recruited-read lengths
    regardless of how much depth credit it carries there."""
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
    """Combined mean read length across every gene in `matrices` put
    together (module docstring's C4 case: C4A+C4B pooled, matching how
    c4_total_depth() itself pools them via best_gene_per_read(), not a
    per-gene breakdown)."""
    lengths = [len(reads_by_id[rid]) for rid in best_gene_per_read(matrices) if rid in reads_by_id]
    return statistics.mean(lengths) if lengths else None


def _length_inflation_factor(target_length: float | None, mean_read_length: float | None) -> float:
    """(target_length + mean_read_length) / mean_read_length -- module
    docstring's derivation of how much a distinct-read COUNT over a whole
    gene body is inflated, relative to true local per-base depth, purely
    as a function of the target's length relative to the reads recruited
    to it. Falls back to 1.0 (no correction) whenever either input is
    missing or non-positive."""
    if not target_length or not mean_read_length or mean_read_length <= 0:
        return 1.0
    return (target_length + mean_read_length) / mean_read_length


def calibrator_genes_for_c4(genes_yaml: Path = DEFAULT_GENES_YAML) -> list[str]:
    """The class I calibrator panel this module normalises C4 depth against
    (module docstring: chosen for chromosomal proximity to the RCCX module,
    not a DESIGN.md instruction). Factored out of estimate_c4_cn() so a
    per-haplotype caller (stages/pipeline.py's run_sample_phased()) can
    check each calibrator gene's own phasing routing against this exact
    same gene set before trusting its HP-split depth -- see that module's
    docstring for why (S10.3: HP labels aren't comparable across genes
    unless they share a PS phase block)."""
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
    """Total C4 copy number (module docstring) -- keyed C4_REPORTED_GENE,
    not "C4A"/"C4B" individually. `classical_matrices` is whatever
    stages/assign.build_gene_matrices() already produced for the ordinary
    star-allele genes in the same run (only used here to read class I's
    calibrator depths); `c4_matrices` is a second, separate
    build_gene_matrices() call against DEFAULT_C4_POOL_FASTA/META.

    `reads_by_id` (read_id -> full sequence, the same mapping
    stages/c4_form.py's estimate_c4_form() already takes) is optional and
    keyword-compatible with every pre-existing caller -- when omitted (the
    default), this function is byte-for-byte the original, uncorrected
    formula. When supplied, the module docstring's length-inflation
    correction is applied to both C4's own depth and each calibrator gene's
    own depth (independently, using each gene's own template length and
    own mean recruited-read length) before the ratio -- `depth` and
    `calibrator_median` on the returned GeneCN still report the real,
    uncorrected observed values (facts, not adjusted numbers); only
    `cn_estimate`/`ci_low`/`ci_high` reflect the correction."""
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

    # Same first-cut Poisson error-propagation formula as stages/cn.py's
    # estimate_cn() (module docstring there) -- not a fitted constant.
    # Deliberately uses the *raw*, uncorrected counts (not the
    # length-normalized ones) -- real Poisson counting noise is governed by
    # how many reads were actually observed, not by a post-hoc length
    # adjustment to the point estimate.
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
