"""C4A/C4B form discrimination (DESIGN.md S2f, Phase 2) -- A-vs-B by exon 26
key peptide positions, long-vs-short by intron 9 span length, both observed
directly from real reads. User-scoped follow-up to stages/c4_cn.py's Phase 1
(total CN via depth ratio, unchanged and untouched by this module).

**Why this is a separate, honest readout from Phase 1's total depth, not a
replacement for it.** stages/c4_cn.py's own docstring already flags that
which of the two build_gene_matrices() matrices (C4A vs C4B) a read's *best
overall alignment* lands in is not a validated A-vs-B call -- C4A/C4B are
near-identical outside exon 26, so a read's coarse best-gene bucket over the
whole ~20kb template is dominated by noise elsewhere in the sequence, not by
the handful of real isotype-determining bases. This module never uses that
bucket assignment as the A/B signal. Instead, for every C4 read it finds
its single best-matching *allele* across the combined C4A+C4B candidate
pool (`_best_allele_cell_per_read`, one shared reduction feeding both
classifiers below) purely to get accurate alignment coordinates/CIGAR, then:

- **A vs B**: reads the read's own bases at `resources/models/
  c4_form_signature.yaml`'s data-derived exon 26 discriminating positions
  (fitted from 362 real per-haplotype Immuannot-labelled sequences,
  `resources/models/build_c4_form_signature.py` -- see that script's own
  docstring) via the same PAF-CIGAR alignment reconstruction
  stages/error_model.py's context scorer uses, and votes A/B by which
  consensus base the read actually carries. A read whose alignment doesn't
  reach a majority of the (5, in the real fitted signature) discriminating
  positions is reported uninformative, not guessed at.
- **Long vs short**: checks whether the read's winning candidate's own
  alignment span geometrically covers intron 9's boundary (in that
  candidate's own local coordinates, from `c4.models.jsonl.gz`) with a
  margin on both flanks, and if so reports that candidate's own real form
  label. No new alignment work -- if the read's true form didn't match the
  winning candidate's form, minimap2 would not have produced a clean,
  fully-spanning low-mismatch alignment across a real ~6.4kb length
  difference in the first place (this pipeline's existing
  max_read_mismatch_rate gate, stages/assign.py, already screens those out
  upstream), so the geometric-overlap gate is what separates "read actually
  crossed the insertion site" from "read landed entirely outside it and
  says nothing about form" -- DESIGN.md's own framing ("observed directly
  from reads that span it").

Both classifiers report an explicit uninformative bucket rather than forcing
every read into a call (same "loud, not silent" stance as everywhere else in
this project) and are reported as raw counts/fractions, not a hard genotype
-- a factual tally of what the reads show, not a claim of a validated A/B or
long/short genotype call.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pysam
import yaml

from assign import AlignmentSpan, GeneAssignmentMatrix
from cn import GeneCN
from error_model import build_aligned_strings

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_SIGNATURE_PATH = REPO_ROOT / "resources" / "models" / "c4_form_signature.yaml"
DEFAULT_C4_POOL_FASTA = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.fa.gz"
DEFAULT_C4_POOL_META = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.meta.tsv.gz"
DEFAULT_C4_MODELS = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.models.jsonl.gz"

_FORM_RE = re.compile(r"^form=(long|short) \(")

# Reused from input/fastq.py's own edge-proximity margin convention (its
# 200bp "close to an anchor's end" rescue clause), not independently
# re-derived for this specific evidentiary bar -- same "reused for
# consistency with an already-accepted provisional call" pattern
# stages/genotype.py's narrow_candidates_by_mismatch_rate() already uses
# (invariant 11: flagged, not presented as fitted).
_INTRON9_MARGIN_BP = 200


@dataclass(frozen=True)
class C4ExonSignaturePosition:
    offset: int  # 0-based, relative to the exon's own start
    a_base: str
    b_base: str


@dataclass(frozen=True)
class C4FormSignature:
    exon26_number: int
    exon26_length: int
    discriminating_positions: tuple[C4ExonSignaturePosition, ...]
    intron9_exon_before: int
    intron9_long_median: float
    intron9_short_median: float
    intron9_length_threshold: float


def load_c4_form_signature(path: Path = DEFAULT_SIGNATURE_PATH) -> C4FormSignature:
    doc = yaml.safe_load(path.read_text())
    exon26 = doc["exon26"]
    intron9 = doc["intron9"]
    return C4FormSignature(
        exon26_number=exon26["exon_number"],
        exon26_length=exon26["exon_length"],
        discriminating_positions=tuple(
            C4ExonSignaturePosition(offset=p["offset"], a_base=p["a_base"], b_base=p["b_base"])
            for p in exon26["discriminating_positions"]
        ),
        intron9_exon_before=intron9["exon_before"],
        intron9_long_median=intron9["long_length_median"],
        intron9_short_median=intron9["short_length_median"],
        intron9_length_threshold=intron9["length_threshold"],
    )


@dataclass(frozen=True)
class C4AlleleAnnotation:
    """Per-allele real structure needed by both classifiers below, read
    from the C4 bundle's own meta/models files (resources/templates/
    build_c4_bundle.py) -- `exon26`/`intron9` are 1-based inclusive in this
    specific allele's own local sequence numbering (sources.py's GeneModel
    convention), not a shared reference coordinate; `form` is parsed from
    the bundle's own "form=long/short (...)" description tag."""

    gene: str
    form: str | None
    exon26: tuple[int, int] | None
    intron9: tuple[int, int] | None


def load_c4_allele_annotations(
    meta_path: Path = DEFAULT_C4_POOL_META,
    models_path: Path = DEFAULT_C4_MODELS,
    *,
    exon_number: int = 26,
    intron_exon_before: int = 9,
) -> dict[str, C4AlleleAnnotation]:
    models: dict[str, dict] = {}
    with gzip.open(models_path, "rt") as fh:
        for line in fh:
            obj = json.loads(line)
            models[obj["ipd_id"]] = obj

    out: dict[str, C4AlleleAnnotation] = {}
    with gzip.open(meta_path, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            m = _FORM_RE.match(row["description"])
            form = m.group(1) if m else None
            model = models.get(row["ipd_id"])
            exon26 = intron9 = None
            if model is not None:
                e = next((e for e in model["exons"] if e[0] == exon_number), None)
                if e is not None:
                    exon26 = (e[1], e[2])
                iv = next((iv for iv in model["introns"] if iv[0] == intron_exon_before), None)
                if iv is not None:
                    intron9 = (iv[1], iv[2])
            out[row["allele"]] = C4AlleleAnnotation(
                gene=row["gene"], form=form, exon26=exon26, intron9=intron9,
            )
    return out


def load_c4_allele_sequences(pool_fasta: Path = DEFAULT_C4_POOL_FASTA) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with pysam.FastxFile(str(pool_fasta)) as fh:
        for rec in fh:
            if rec.name is None or rec.sequence is None:
                raise ValueError(f"{pool_fasta}: record with no name/sequence")
            sequences[rec.name] = rec.sequence
    return sequences


def _best_allele_cell_per_read(
    c4_matrices: dict[str, GeneAssignmentMatrix],
) -> dict[str, tuple[str, str, AlignmentSpan]]:
    """Every C4 read's single best-matching (gene, allele) cell across BOTH
    C4A and C4B matrices combined, by lowest mismatch rate -- ties broken
    alphabetically by (gene, allele) (invariant 9). Iterates only
    `spans` (real alignments), never the dense zero-filled matrix cells.
    Module docstring: this is an *alignment-coordinate* anchor for both
    classifiers below, not an A-vs-B/long-vs-short signal on its own."""
    candidates: dict[str, list[tuple[float, str, str, AlignmentSpan]]] = {}
    for gene in sorted(c4_matrices):
        m = c4_matrices[gene]
        if not m.read_ids:
            continue
        rates = m.mismatch_rate()
        for (read_idx, allele_idx), span in m.spans.items():
            read_id = m.read_ids[read_idx]
            allele = m.allele_ids[allele_idx]
            candidates.setdefault(read_id, []).append(
                (float(rates[read_idx, allele_idx]), gene, allele, span)
            )
    out: dict[str, tuple[str, str, AlignmentSpan]] = {}
    for read_id, cells in candidates.items():
        _rate, gene, allele, span = min(cells, key=lambda c: (c[0], c[1], c[2]))
        out[read_id] = (gene, allele, span)
    return out


def _ref_positions_to_read_bases(
    aligned_ref: str, aligned_read: str, ref_start: int, positions: list[int],
) -> dict[int, str | None]:
    """Map absolute (0-based) reference positions to the read's own base at
    each, via a single pass over the gapped alignment strings
    (stages/error_model.py's build_aligned_strings) -- None where that
    position falls in a read-side deletion (no base to vote with)."""
    wanted = {p - ref_start: p for p in positions}
    out: dict[int, str | None] = {}
    consumed = 0
    for i, ref_char in enumerate(aligned_ref):
        if ref_char == "-":
            continue
        if consumed in wanted:
            read_char = aligned_read[i]
            out[wanted[consumed]] = None if read_char == "-" else read_char
        consumed += 1
    return out


@dataclass(frozen=True)
class C4AbCall:
    n_a: int
    n_b: int
    n_uninformative: int
    a_fraction: float | None  # n_a / (n_a + n_b), None if no informative reads at all


def classify_a_vs_b(
    c4_matrices: dict[str, GeneAssignmentMatrix],
    reads_by_id: dict[str, str],
    annotations: dict[str, C4AlleleAnnotation],
    signature: C4FormSignature,
    allele_sequences: dict[str, str],
) -> C4AbCall:
    """Module docstring. A read needs its winning candidate's own exon 26
    to exist, its alignment span to cover a majority of the signature's
    discriminating positions, and a real (non-deletion) base at a majority
    of those covered positions favouring one side -- anything short of
    that is uninformative, not forced into a call."""
    best = _best_allele_cell_per_read(c4_matrices)
    n_positions = len(signature.discriminating_positions)
    min_covered = (n_positions + 1) // 2
    n_a = n_b = n_uninformative = 0

    for read_id, (_gene, allele, span) in best.items():
        ann = annotations.get(allele)
        read_seq = reads_by_id.get(read_id)
        ref_seq = allele_sequences.get(allele)
        if ann is None or ann.exon26 is None or read_seq is None or ref_seq is None:
            n_uninformative += 1
            continue

        exon_start0 = ann.exon26[0] - 1
        by_pos = {exon_start0 + p.offset: p for p in signature.discriminating_positions}
        covered = [pos for pos in by_pos if span.ref_start <= pos < span.ref_end]
        if len(covered) < min_covered:
            n_uninformative += 1
            continue

        aligned_ref, aligned_read = build_aligned_strings(
            ref_seq, read_seq,
            ref_start=span.ref_start, ref_end=span.ref_end,
            read_start=span.read_start, read_end=span.read_end,
            strand=span.strand, cigar=span.cigar,
        )
        bases = _ref_positions_to_read_bases(aligned_ref, aligned_read, span.ref_start, covered)

        a_votes = b_votes = 0
        for pos, base in bases.items():
            if base is None:
                continue
            sig = by_pos[pos]
            if base == sig.a_base:
                a_votes += 1
            elif base == sig.b_base:
                b_votes += 1
        if a_votes > b_votes:
            n_a += 1
        elif b_votes > a_votes:
            n_b += 1
        else:
            n_uninformative += 1

    total = n_a + n_b
    return C4AbCall(
        n_a=n_a, n_b=n_b, n_uninformative=n_uninformative,
        a_fraction=(n_a / total) if total else None,
    )


@dataclass(frozen=True)
class C4PerGeneCN:
    """CN_A/CN_B (user-requested, 2026-08-27): a *proportional split* of
    the combined Phase 1 total-C4 CN by the Phase 2 exon 26 vote ratio, not
    an independently-measured per-gene depth. See estimate_c4_per_gene_cn()
    for why a real per-gene depth-ratio estimate isn't used instead."""
    cn_a: float | None
    cn_a_ci_low: float | None
    cn_a_ci_high: float | None
    cn_b: float | None
    cn_b_ci_low: float | None
    cn_b_ci_high: float | None


def _implied_relative_se(cn_estimate: float, ci_high: float) -> float:
    """Recover the relative SE that produced `ci_high = cn_estimate * (1 +
    1.96*rel_se)` (stages/cn.py's and stages/c4_cn.py's own CI convention)
    -- uses the *upper* bound, not the lower, because `ci_low` is clipped
    at 0 (`max(0.0, ...)`) and so can't always be inverted cleanly."""
    if cn_estimate <= 0:
        return 0.0
    return (ci_high / cn_estimate - 1.0) / 1.96


def estimate_c4_per_gene_cn(total_c4_cn: GeneCN, ab_call: C4AbCall) -> C4PerGeneCN:
    """CN_A = total_CN x a_fraction, CN_B = total_CN x (1 - a_fraction).

    **Why a split of the total, not an independent per-gene depth-ratio
    estimate.** `ab_call.n_a`/`n_b` are read counts at one narrow ~157bp
    exon (26), not distinct-read counts over the whole ~20kb gene body the
    way stages/c4_cn.py's `depth` is defined -- and classify_a_vs_b()'s own
    evidentiary gate (a read must cover a majority of the signature's
    discriminating positions with real, non-deletion bases) systematically
    excludes real reads that do overlap the site but don't clear that bar.
    Plugging `n_a`/`n_b` directly into the same depth-ratio formula would
    therefore underestimate both genes by an uncalibrated amount. Splitting
    the already length-corrected combined total by the *ratio* n_a/(n_a+n_b)
    sidesteps that: the undercounting should affect A- and B-origin reads
    symmetrically (the two genes are >99% identical outside the signature's
    own 5 positions, so nothing about the gate's coverage requirement
    should favour one over the other) -- flagged as an assumption, not
    independently verified, invariant 11.

    **CI, real caveats.** Combines two normal-approximation relative SEs in
    quadrature: `total_c4_cn`'s own (backed out from its CI via
    `_implied_relative_se()`) and the vote fraction's own binomial SE
    (`sqrt(p(1-p)/n)`, n = n_a+n_b) -- treating the two as independent is
    an approximation (they're computed from overlapping but not identical
    read sets), and the binomial SE formula understates uncertainty right
    at n_a=0 or n_b=0 (a Wilson interval would not, but this project's own
    CI convention elsewhere is already this same first-cut normal
    approximation, not Wilson -- consistent, not gold-standard). At the
    read counts this project expects at UKB's target 15X, `n_a+n_b` can be
    small enough that this split has real, sometimes wide, uncertainty --
    report the CI, never just the point estimate.

    Returns all-None fields when either input lacks a usable estimate
    (`total_c4_cn.cn_estimate is None` or `ab_call.a_fraction is None`,
    i.e. zero informative reads) -- no guess in place of missing data."""
    if total_c4_cn.cn_estimate is None or ab_call.a_fraction is None:
        return C4PerGeneCN(
            cn_a=None, cn_a_ci_low=None, cn_a_ci_high=None,
            cn_b=None, cn_b_ci_low=None, cn_b_ci_high=None,
        )

    total_cn = total_c4_cn.cn_estimate
    rel_se_total = 0.0
    if total_c4_cn.ci_high is not None:
        rel_se_total = _implied_relative_se(total_cn, total_c4_cn.ci_high)

    n = ab_call.n_a + ab_call.n_b
    p = ab_call.a_fraction
    se_p = (p * (1 - p) / n) ** 0.5 if n > 0 else 0.0

    def _split(fraction: float, se_fraction: float) -> tuple[float, float, float]:
        cn = total_cn * fraction
        rel_se_fraction = (se_fraction / fraction) if fraction > 0 else 0.0
        rel_se = (rel_se_total ** 2 + rel_se_fraction ** 2) ** 0.5
        return cn, cn * max(0.0, 1 - 1.96 * rel_se), cn * (1 + 1.96 * rel_se)

    cn_a, cn_a_lo, cn_a_hi = _split(p, se_p)
    cn_b, cn_b_lo, cn_b_hi = _split(1 - p, se_p)

    return C4PerGeneCN(
        cn_a=cn_a, cn_a_ci_low=cn_a_lo, cn_a_ci_high=cn_a_hi,
        cn_b=cn_b, cn_b_ci_low=cn_b_lo, cn_b_ci_high=cn_b_hi,
    )


@dataclass(frozen=True)
class C4FormLengthCall:
    n_long: int
    n_short: int
    n_uninformative: int
    long_fraction: float | None  # n_long / (n_long + n_short), None if no informative reads


def classify_long_vs_short(
    c4_matrices: dict[str, GeneAssignmentMatrix],
    annotations: dict[str, C4AlleleAnnotation],
    *,
    margin_bp: int = _INTRON9_MARGIN_BP,
) -> C4FormLengthCall:
    """Module docstring: a read is only informative if its winning
    candidate's alignment span geometrically clears both flanks of that
    candidate's own intron 9 window by `margin_bp` -- a read that never
    reaches the insertion site says nothing about which form it carries."""
    best = _best_allele_cell_per_read(c4_matrices)
    n_long = n_short = n_uninformative = 0

    for _read_id, (_gene, allele, span) in best.items():
        ann = annotations.get(allele)
        if ann is None or ann.form is None or ann.intron9 is None:
            n_uninformative += 1
            continue
        intron_start0 = ann.intron9[0] - 1
        intron_end0 = ann.intron9[1]
        spans_intron = (
            span.ref_start <= intron_start0 - margin_bp
            and span.ref_end >= intron_end0 + margin_bp
        )
        if not spans_intron:
            n_uninformative += 1
        elif ann.form == "long":
            n_long += 1
        else:
            n_short += 1

    total = n_long + n_short
    return C4FormLengthCall(
        n_long=n_long, n_short=n_short, n_uninformative=n_uninformative,
        long_fraction=(n_long / total) if total else None,
    )


def estimate_c4_form(
    c4_matrices: dict[str, GeneAssignmentMatrix],
    reads_by_id: dict[str, str],
    *,
    signature_path: Path = DEFAULT_SIGNATURE_PATH,
    meta_path: Path = DEFAULT_C4_POOL_META,
    models_path: Path = DEFAULT_C4_MODELS,
    pool_fasta: Path = DEFAULT_C4_POOL_FASTA,
) -> tuple[C4AbCall, C4FormLengthCall]:
    """Convenience entry point stages/pipeline.py calls: loads the shipped
    signature/annotations/sequences and runs both classifiers over the same
    Stage 1 C4 matrices stages/c4_cn.py already builds (no new alignment
    work)."""
    signature = load_c4_form_signature(signature_path)
    annotations = load_c4_allele_annotations(meta_path, models_path)
    allele_sequences = load_c4_allele_sequences(pool_fasta)
    ab_call = classify_a_vs_b(c4_matrices, reads_by_id, annotations, signature, allele_sequences)
    form_call = classify_long_vs_short(c4_matrices, annotations)
    return ab_call, form_call
