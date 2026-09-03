"""Stage 1 -- gene assignment and copy number (DESIGN.md S1).

**Depth source revised 2026-08-31 -- a deliberate departure from DESIGN.md
S1's "CN is not a separate stage, it is a second readout of the read x
allele assignment matrix" framing, real-data-forced.** `estimate_cn()` no
longer reads `assign.py`'s per-allele, full-IPD-pool matrix at all. Root
cause, confirmed on real HG00320/HG00097 ONT data: for HLA-H/K/T/U/W (the
tight class I pseudogene cluster, >99% identical to each other,
`assign.py`'s own "makes or breaks the gene expansion" case),
`assign.best_gene_per_read()`'s per-read gene resolution -- winner-take-all
or margin-gated, both tried -- was scattering each gene's true depth across
its neighbours on noise-level margins (real case: one read's own-pool rate
was 0.31% against HLA-H vs 0.12% against HLA-T, both essentially perfect
matches). Margin-gated even-splitting (still in `assign.py`, used by
`c4_cn.py`) measurably failed to fix it: ties are roughly symmetric, so a
gene loses back with one hand what it gains with the other, and 65 real
"stolen" (gene, read) pairs in this cluster showed a smooth, un-cleanly-
separable gap distribution from 0.02% to 3.9% -- no threshold cleanly
divides "same locus, ONT noise" from "genuinely a different gene".
Trusting Stage 0's own anchor-sieve assignment directly (the other
originally-proposed option) also failed, and for a concrete reason: the
anchor sieve's own decoy-rejection gate (a 10x mismatch-rate multiplier) is
far more permissive than Stage 1's tight matching, so 46 of 104 real reads
touching this cluster qualified for *more than one* gene's anchors
simultaneously -- the ambiguity is real at every stage this project has
signal from, not an artifact of which reference set is used.

**What did work, real-data-validated:** counting each Stage 0-recruited
read once per gene it qualifies for -- no cross-gene resolution attempted
at all, a read genuinely ambiguous between two paralogous genes' anchors
counts fully toward both -- plus two measurement corrections, each
independently confirmed necessary: (1) dedup *within* a gene (a single read
can carry >1 qualifying `AnchorHit` for the same gene against different
anchor alleles, which was silently double-counting -- real HG00320 HLA-H
raw hit count 72 vs 36 distinct reads); (2) the same length-inflation
correction `c4_cn.py` already established for C4 (a longer recruitment
target tiles more distinct reads at a fixed true depth, independent of
copy number), using each gene's own mean *target-role anchor* length (not
the full IPD template length -- Stage 0 recruitment is governed by anchor
alignment, not a full-pool one). Neither correction alone was sufficient
on real data (length correction without dedup changed nothing, since the
calibrator panel is inflated by the identical multi-anchor-hit artifact);
both together pushed HLA-H/K/T/U/W from `cn_estimate` ~0.6-1.2 (rounds to
1, hiding a real second haplotype) to ~1.5-2.1 (correctly rounds to 2) on
real HG00320 data, with **zero cn_int changes on real HG00097** (44/46
genes identical; the 2 apparent differences in a same-session diagnostic
turned out to be an artifact of not replicating `pipeline.py`'s DRB3/4/5
mutual-exclusivity pass, not a real difference -- see that module).

**Flagged, not silently assumed correct:** this signal is broadly applied,
not narrowly targeted at the pseudogene cluster -- on real HG00320 it also
pushed HLA-DPB1/HLA-DQB1/HLA-S/HLA-V past the rounding threshold, which
have not been independently checked against truth as of this revision.
Provisional (invariant 11), same as every constant in this module already
was.

C4A/C4B excluded the same way resources/templates/build.py excludes them
(typing_module == "structural" genes have no IPD template pool and never
appear in an assign.py matrix; DESIGN.md S2f's own C4 CN readout is a
different measurement -- depth against the RCCX region -- handled entirely
by c4_cn.py/c4_chr17_cn.py, out of scope for *this* module, not silently
reported as a bogus zero-depth deletion).

**Normalisation revised 2026-09-01, user-directed: chr17-control depth,
not the HLA-gene calibrator panel.** Through 2026-08-31 this module
normalised against reference-free median-of-ratios -- median(depth over
that sample/class's `resources/genes.yaml` `calibrator` panel), split by
class (I vs II, ~1-2Mb apart, can see different local coverage). C4's own
real HG00253 investigation that same session (the development notes) traced
a real, badly-undershooting total-CN estimate to the calibrator panel
itself being a noisy, HLA-local baseline, not a fundamentally different
question from what classical/pseudogene CN faces here -- the fix there
(locityper's own published segdup-free chr17 background interval,
`stages/chr17_control.py`, real base-level coverage) is now the single
normaliser for every gene this module estimates, replacing the per-class
calibrator median entirely: `CN(gene) = 2 x depth(gene) / control_depth`.
No more class split either -- one sample-wide control replaces two
separate calibrator medians, matching the control's own genome-wide,
locus-independent design intent. `estimate_cn()` takes `control_depth` as
an explicit, caller-supplied parameter (dependency injection, same
convention as `pipeline.py`'s own `cn_round`) -- this module does not
align anything itself; `pipeline.py` computes it once per sample via
`chr17_control.compute_chr17_control_depth()` and passes it in.
`resources/genes.yaml`'s `calibrator` field and `GeneManifestEntry.
calibrator` are left in place -- `c4_cn.py`'s own, separate, still-
calibrator-panel-based C4 total-CN mechanism (`c4_cn.calibrator_genes_for_
c4()`) still reads them via this module's `load_gene_manifest()`, and is
out of scope for this revision.

**Depth numerator also revised, same session, user-directed: real
`samtools depth` over each gene's own anchor(s), not a distinct-read-count
+ length-inflation-factor proxy.** The chr17-control swap above first
shipped with `stage0_depth_by_gene()` unchanged (Stage 0's sieved reads
counted once per gene, deduped, divided by `_length_inflation_factor()`).
User-directed follow-up, same day, extends the samtools-depth method
`stages/align_depth.py`/`stages/chr17_control.py`/`stages/c4_chr17_cn.py`
already established for the control region and C4 to this numerator too:
`stage0_depth_by_gene()` now writes Stage 0's own sieved `reads` to a
temporary FASTA and realigns them (cheap -- this is already the small,
sieved subset, not the whole sample's raw FASTQ) against the *full*
`resources/anchors/anchors.fa.gz` with `minimap2 -a`, sorts into a real
BAM, drops alignment records covering less than `min_target_coverage` of
their own anchor contig's length (default sourced from `input/fastq.py`'s
`SieveParams.min_anchor_coverage` -- the identical value Stage 0's own
sieve uses, not a second independently-chosen copy), then sums real
`samtools depth` over each gene's own target-role anchor contig(s)
(pooled across a gene's several allele-representative anchors, if it has
more than one, weighted by their real lengths) and divides by their total
length. This replaces the read-count proxy and its length-inflation
correction entirely -- a genuine per-base coverage measurement doesn't
need a correction invented to compensate for a proxy's own length bias.
`_load_target_anchor_lengths()`/`_length_inflation_factor()` are removed
(dead code once nothing computes a distinct-read count any more).

**Not segment-pooled, by explicit decision (revised from the original
segment design, 2026-07-27).** DESIGN.md's original S2c model summed depth
across every gene sharing a `resources/genes.yaml` `segment` label and
applied that one pooled CN identically to every member gene -- correct only
if the member genes truly co-vary in lockstep. That does not hold for every
group genes.yaml currently marks with a shared segment: real HG00097 ONT
benchmark data showed HLA-A (segment `alpha_block_A`, shared with HLA-H)
reporting CN=3, driven by depth from a real HLA-H copy present on one
haplotype -- correct as a *segment* total, but wrong as HLA-A's own CN
(HLA-A itself is essentially always diploid; HLA-H's independent
presence/absence is not evidence about HLA-A's count) and it forced Stage 2
to search HLA-A's own candidate pool for a 3rd distinct HLA-A template that
does not exist, which is what actually stalled template selection. The same
conflation risk applies to `drb_block` (DRB1, essentially always present,
pooled with the genuinely CN-variable DRB3/4/5). DESIGN.md S2c's own
correlated-segment claim (HLA-Y/HLA-OLI, "correlated by construction") may
be true for that specific pair, but genes.yaml's `segment` field currently
groups multiple genes together without per-pair validation, and DESIGN.md
S2c's own proposed check for this (cohort-scale CN correlation across a
segment, Stage 6) has not been run. Per-gene CN is the honest default until
that validation exists; `resources/genes.yaml`'s `segment` field is left in
place as documentary genomic-block metadata (still true, and exactly what a
future Stage 6 correlation check would need) but is no longer read here.

CN=0 is never reported as a bare deletion (invariant 4). DESIGN.md S2c's
"second, template-free check" (reference-space read presence) is BAM-path
only and not built yet -- the assignment matrix this module reads is
reference-free by construction (S1) -- so every GeneCN here carries
`corroborated=False`: a loud, explicit "unconfirmed", never silently
upgraded to a deletion claim.
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
    """Genes eligible for this module's per-gene CN estimation: not
    excluded, and not `typing_module: structural` (C4A/C4B -- see module
    docstring). `segment` is loaded as documentary genomic-block metadata
    only (module docstring) -- not read by estimate_cn(). `low_cn_floor` and
    `cn_model` are not read by estimate_cn() either -- both are
    stages/pipeline.py policy decisions about how a gene's raw estimate
    rounds to a search-sizing integer, not part of the estimate itself."""
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
    depth: float  # float, not int: estimate_cn()'s own depth (module
    # docstring, 2026-08-31) is length-corrected (a real division, not a
    # count) -- stage0_depth_by_gene()'s return value. depth_by_gene()
    # (matrix-based, still used by c4_cn.py) can also produce a fractional
    # value via assign.best_gene_per_read()'s margin-gated resolution.
    calibrator_median: float | None  # 2026-09-01: holds the chr17 control
    # depth this gene was normalised against (module docstring), not a
    # calibrator-panel median, when produced by this module's own
    # estimate_cn() -- field name kept for dataclass-shape compatibility
    # with c4_cn.py's separate, still-calibrator-panel-based producer of
    # the same dataclass.
    cn_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    is_zero_call: bool
    corroborated: bool  # always False in v0 -- see module docstring


def depth_by_gene(matrices: dict[str, GeneAssignmentMatrix]) -> dict[str, float]:
    """Reads assigned per gene: each read's 1.0 unit of depth credit is
    summed per gene from assign.best_gene_per_read()'s margin-gated
    resolution (see that function's own docstring) -- an unambiguous read
    contributes its full 1.0 to one gene exactly as before; a read
    genuinely tied between two or more paralogous genes' pools contributes
    a fraction to each instead of the whole read landing on one arbitrary
    winner. A read's credit never exceeds 1.0 in total, so this still
    cannot double-count a read across genes -- this is the depth numerator
    DESIGN.md S1 defines."""
    counts: dict[str, float] = {}
    for gene_weights in best_gene_per_read(matrices).values():
        for gene, weight in gene_weights.items():
            counts[gene] = counts.get(gene, 0.0) + weight
    return counts


def _load_target_anchor_lengths(anchors_meta_path: Path) -> dict[str, dict[str, int]]:
    """gene -> {anchor_allele: real anchor length}, every target-role row
    (resources/anchors/anchors.meta.tsv.gz's own `length` column, real for
    every anchor in the shipped production file, confirmed directly --
    not re-derived from a BAM header, since this revision no longer builds
    one). A `length`-less row (every synthetic test fixture predates this
    column) is skipped for that specific anchor, same graceful-degradation
    contract this module has always used here."""
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
    """The depth numerator `estimate_cn()` uses -- real aligned bases,
    reconstructed directly from Stage 0's own sieve alignment
    (`AnchorHit.coverage = match_len / anchor_length`, already computed by
    `input/fastq.py`'s `sieve_fastq()`), not a second, independent
    alignment pass.

    **2026-09-01, second revision, user-directed, same day as the first
    (real-alignment/samtools-depth) version.** That first version realigned
    Stage 0's sieved reads a second time against the *entire* ~187-contig
    anchors.fa.gz in one combined minimap2 call -- and, on the very first
    real-sample validation run, was found to have a real bug: it never set
    minimap2's `-N` (secondary-alignment count), so a gene with several of
    its own anchors competing against every *other* gene's anchors in the
    same pass (HLA-A alone has 9) silently lost any read whose true best
    match didn't crack minimap2's default top-5 globally -- confirmed
    directly, real HG00253 data: HLA-A's `cn_estimate` collapsed to 0.28
    (true ~2) while HLA-C (fewer/more distinctive anchors) came out close
    to correct. Rather than just raising `-N` on the remapping pass, the
    user's own redirect: reuse Stage 0's own alignment result instead of
    redoing it -- `input/fastq.py`'s sieve already ran with `-N 256`
    (its own docstring has the reasoning) specifically because the decoy-
    rejection logic needs to see every anchor a read hits, so it was never
    exposed to the bug the second pass introduced. This also removes the
    real extra wall-clock cost the remapping pass added (a real HG00253
    run: ~18s realignment + BAM sort/filter/dedup overhead, on top of an
    already-real ~106s chr17-control pass) -- reusing already-computed
    data is faster, not just more correct.

    Per read, only the single best-covering `AnchorHit` *for a given gene*
    is kept (same within-gene-multi-anchor dedup principle the removed
    remapping version had, still needed: a read can carry qualifying hits
    to more than one of a gene's own near-identical anchors). Aligned
    bases for that hit = `coverage * that anchor's own real length`
    (`_load_target_anchor_lengths()`); summed per gene and divided by the
    *mean* length of every target-role anchor that gene has (not just the
    ones any read happened to hit -- same "alternative references for the
    same locus, not independent territory" reasoning the 2026-08-31
    entry above already established, still correct here)."""
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
    """Per-gene CN (DESIGN.md S1), keyed by gene name -- see module docstring
    for why this is no longer pooled across genes.yaml's `segment` field,
    for why the depth numerator (`stage0_depth_by_gene()`, real aligned
    bases reconstructed from Stage 0's own sieve alignment) and denominator
    (`control_depth`, real chr17-control samtools depth,
    `chr17_control.compute_chr17_control_depth()`, computed once by the
    caller) both changed as of 2026-09-01. `control_depth` is a required
    positional argument, not a silently-defaulted one -- every caller must
    be explicit about what this run's estimates are normalised against."""
    manifest = load_gene_manifest(genes_yaml)
    depth = stage0_depth_by_gene(reads, anchors_meta_path=anchors_meta_path)

    results: dict[str, GeneCN] = {}
    for gene in sorted(manifest):
        entry = manifest[gene]
        gene_class = entry.gene_class
        gene_depth = depth.get(gene, 0)

        if not control_depth:
            # No control signal this run (e.g. a --bam-derived FASTQ that
            # never pulled chr17 reads -- DESIGN.md's own documented gap) --
            # cannot normalise. Loud (None CN), not a silent 0 (invariant 4).
            results[gene] = GeneCN(
                gene=gene, gene_class=gene_class,
                depth=gene_depth, calibrator_median=None, cn_estimate=None,
                ci_low=None, ci_high=None,
                is_zero_call=(gene_depth == 0), corroborated=False,
            )
            continue

        cn_estimate = 2.0 * gene_depth / control_depth

        # First-cut Poisson count-noise propagation on the depth ratio --
        # a textbook error-propagation formula, not a fitted constant, so it
        # is not what invariant 10 means by "an error rate". Provisional:
        # a cohort-fitted depth-noise model belongs in resources/models/
        # once Stage 6 gives us the data to fit one from.
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
