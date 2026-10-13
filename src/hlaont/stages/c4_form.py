"""C4A/C4B form discrimination from reads (DESIGN.md S2f, Phase 2).

The C4A-vs-C4B matrix a read lands in is noise-dominated (the genes differ
almost only in exon 26), so it is never used as the A/B signal. Each read's
best allele across both pools (`_best_allele_cell_per_read`) only provides
alignment coordinates; then:

- A vs B: the read's bases at the fitted exon-26 discriminating positions
  (resources/models/c4_form_signature.yaml) vote A or B. Reads covering
  fewer than a majority of positions are uninformative.
- Long vs short: if the alignment spans the candidate's intron 9 with a
  margin on both sides, report the candidate's form (a clean alignment
  can't cross the ~6.4kb HERV length difference with the wrong form).

Both report raw counts with an uninformative bucket, not a genotype.
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

from hlaont.stages.assign import AlignmentSpan, GeneAssignmentMatrix
from hlaont.stages.cn import GeneCN
from hlaont.stages.error_model import build_aligned_strings

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_SIGNATURE_PATH = REPO_ROOT / "resources" / "models" / "c4_form_signature.yaml"
DEFAULT_C4_POOL_FASTA = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.fa.gz"
DEFAULT_C4_POOL_META = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.meta.tsv.gz"
DEFAULT_C4_MODELS = REPO_ROOT / "resources" / "templates" / "assembly_bundle" / "c4.models.jsonl.gz"

_FORM_RE = re.compile(r"^form=(long|short) \(")

# Flank margin around intron 9, reused from fastq.py's 200bp edge rule.
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
    """Per-allele exon 26 / intron 9 (1-based inclusive, allele-local
    coordinates) and form from the C4 bundle."""

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
    """Each read's best (gene, allele) cell across C4A and C4B by mismatch
    rate (ties alphabetical); an alignment anchor, not a call."""
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
    """Read base at each 0-based reference position, None in a deletion."""
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
    """A read votes only if its candidate has exon 26, it covers a majority
    of the discriminating positions, and a majority of covered real bases
    favour one side; otherwise uninformative."""
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
    """CN_A/CN_B: total C4 CN split by the Phase 2 vote ratio."""
    cn_a: float | None
    cn_a_ci_low: float | None
    cn_a_ci_high: float | None
    cn_b: float | None
    cn_b_ci_low: float | None
    cn_b_ci_high: float | None


def _implied_relative_se(cn_estimate: float, ci_high: float) -> float:
    """Relative SE from `ci_high = cn * (1 + 1.96 * rel_se)` (ci_low is
    clipped at 0)."""
    if cn_estimate <= 0:
        return 0.0
    return (ci_high / cn_estimate - 1.0) / 1.96


def estimate_c4_per_gene_cn(total_c4_cn: GeneCN, ab_call: C4AbCall) -> C4PerGeneCN:
    """CN_A = total x a_fraction, CN_B = total x (1 - a_fraction).

    The vote counts come from one ~157bp exon behind a strict coverage gate,
    so they undercount; their ratio is used instead, assuming the gate hits
    A and B reads alike. The CI combines the total's relative SE and the
    binomial SE of the fraction in quadrature (approximate; wide at low
    counts). All None if either input is unavailable."""
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
    """A read is informative only if its alignment clears both flanks of its
    candidate's intron 9 by `margin_bp`."""
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
    """Load the shipped signature/annotations/sequences and run both
    classifiers on the existing C4 matrices."""
    signature = load_c4_form_signature(signature_path)
    annotations = load_c4_allele_annotations(meta_path, models_path)
    allele_sequences = load_c4_allele_sequences(pool_fasta)
    ab_call = classify_a_vs_b(c4_matrices, reads_by_id, annotations, signature, allele_sequences)
    form_call = classify_long_vs_short(c4_matrices, annotations)
    return ab_call, form_call
