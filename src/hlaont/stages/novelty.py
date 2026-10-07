"""Stage 5 -- novelty classification (DESIGN.md S5).

Diffs the consensus against `called_allele` using Stage 4's captured CIGAR
and locates each variant in the allele's CDS/intron/UTR (gene models from
ref.gene.models.jsonl.gz).

Taxonomy: exact | extension | novel:intronic | novel:utr |
novel:synonymous | novel:nonsynonymous | novel:null, plus
- extension:partial: all CDS differences match an allele IPD holds only as
  a partial (exon-subset) record (`_partial_extension_match`);
- novel:extraction: zero variants against an assembly-extracted allele
  whose Immuannot nearest match is itself `*new`.

`extension` (invariant 7): the edited CDS exactly matches another known
allele that IPD has only as exon-only, so it was never in the genomic pool
(ref.gene.cds_catalogue.tsv.gz). A CDS match to a different *genomic*
allele is not re-routed.

Variants within 2bp of an intron end (GT...AG) are treated as
splice-disrupting -> novel:null. Names follow Immuannot's `new` tag: the
agreeing fields, then `:new` (e.g. HLA-A*24:02:01:new).
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "input"))

from assign import DEFAULT_POOL_FASTA, AlignmentSpan  # noqa: E402
from consensus import ConsensusVariant  # noqa: E402
from error_model import build_aligned_strings  # noqa: E402
from genotype import load_allele_sequences  # noqa: E402
from hla_typing import TypingCall  # noqa: E402

DEFAULT_POOL_MODELS = REPO_ROOT / "resources" / "templates" / "pool" / "ref.gene.models.jsonl.gz"
DEFAULT_CDS_CATALOGUE = REPO_ROOT / "resources" / "templates" / "pool" / "ref.gene.cds_catalogue.tsv.gz"
DEFAULT_POOL_META = REPO_ROOT / "resources" / "templates" / "pool" / "ref.gene.meta.tsv.gz"

# Canonical splice-site window (bp at each intron end).
_SPLICE_SITE_WINDOW = 2

# Variants with read support below this fraction are ignored for the
# taxonomy; a haplotype is novel if any variant survives. Consensus QV
# saturates at 60 even under real dissent, so it can't gate novelty. On 99
# labelled calls: precision 64.7% -> 80.2%, recall 90.0% -> 77.0% (MICA
# loses 3 TPs, accepted). Provisional.
_MIN_VARIANT_SUPPORT = 0.8


def _read_support(cv: ConsensusVariant) -> float:
    if cv.depth == 0:
        return 0.0
    return (cv.depth - cv.dissent) / cv.depth


def _filter_by_read_support(
    raw_variants: list["_RawVariant"], consensus_variants: Sequence[ConsensusVariant]
) -> list["_RawVariant"]:
    """Drop variants below `_MIN_VARIANT_SUPPORT`. No-op without
    `consensus_variants`; variants with no matching entry are kept."""
    if not consensus_variants:
        return raw_variants
    lookup = {
        (cv.pos, cv.ref == "" and cv.alt != ""): cv for cv in consensus_variants
    }
    kept = []
    for rv in raw_variants:
        cv = lookup.get((rv.pos, rv.is_insertion))
        if cv is None or _read_support(cv) >= _MIN_VARIANT_SUPPORT:
            kept.append(rv)
    return kept

_STANDARD_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass(frozen=True)
class GeneModelLite:
    cds_exons: tuple[tuple[int, int], ...]
    codon_start: int
    utrs: tuple[tuple[int, int], ...]
    exons: tuple[tuple[int | None, int, int], ...]
    introns: tuple[tuple[int | None, int, int], ...]


@dataclass(frozen=True)
class NoveltyVariant:
    pos: int
    ref: str
    alt: str
    location: str  # "cds" | "intron" | "utr" | "unannotated"
    splice_site: bool


@dataclass(frozen=True)
class NoveltyCall:
    gene: str
    called_allele: str
    taxonomy: str
    proposed_name: str
    candidate_null: bool
    extension_allele: str | None
    coding_effect: str | None  # "synonymous" | "nonsynonymous" | "frameshift" | "premature_stop" | None
    variants: tuple[NoveltyVariant, ...]


def load_gene_model(allele: str, models_path: Path = DEFAULT_POOL_MODELS) -> GeneModelLite:
    with gzip.open(models_path, "rt") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj["allele"] == allele:
                return GeneModelLite(
                    cds_exons=tuple((s, e) for s, e in obj["cds_exons"]),
                    codon_start=obj["codon_start"],
                    utrs=tuple((s, e) for s, e in obj["utrs"]),
                    exons=tuple((n, s, e) for n, s, e in obj["exons"]),
                    introns=tuple((n, s, e) for n, s, e in obj["introns"]),
                )
    raise ValueError(f"{allele!r}: no gene model in {models_path}")


def load_nearest_ipd_match(allele: str, pool_meta: Path = DEFAULT_POOL_META) -> str | None:
    """Immuannot's nearest IPD match for an assembly-extracted `allele`
    (meta `nearest_ipd_match`); None otherwise."""
    with gzip.open(pool_meta, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["allele"] == allele:
                return row.get("nearest_ipd_match") or None
    return None


def load_gtf_template_allele(allele: str, pool_meta: Path = DEFAULT_POOL_META) -> str | None:
    """Source GTF `template_allele` for an assembly-extracted `allele` (meta
    `gtf_template_allele`, never `:new`); None otherwise. Used to display a
    real name instead of the content-hash accession."""
    with gzip.open(pool_meta, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["allele"] == allele:
                return row.get("gtf_template_allele") or None
    return None


def load_cds_catalogue_for_gene(
    gene: str, catalogue_path: Path = DEFAULT_CDS_CATALOGUE
) -> dict[str, list[tuple[str, bool]]]:
    """cds_sequence -> [(allele, genomic), ...] for `gene`'s non-partial
    records (the `extension` lookup)."""
    out: dict[str, list[tuple[str, bool]]] = {}
    with gzip.open(catalogue_path, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["gene"] != gene or row.get("partial") == "True":
                continue
            out.setdefault(row["cds_sequence"], []).append(
                (row["allele"], row["genomic"] == "True")
            )
    return out


@dataclass(frozen=True)
class PartialCdsRecord:
    allele: str
    cds_sequence: str
    exon_numbers: frozenset[int]
    n_fields: int


def load_partial_cds_records_for_gene(
    gene: str, catalogue_path: Path = DEFAULT_CDS_CATALOGUE
) -> list[PartialCdsRecord]:
    """`gene`'s partial (exon-subset) CDS records with known exon numbers."""
    out: list[PartialCdsRecord] = []
    with gzip.open(catalogue_path, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["gene"] != gene or row.get("partial") != "True":
                continue
            if not row["exon_numbers"]:
                continue
            out.append(PartialCdsRecord(
                allele=row["allele"],
                cds_sequence=row["cds_sequence"],
                exon_numbers=frozenset(int(x) for x in row["exon_numbers"].split(",")),
                n_fields=len(row["allele"].partition("*")[2].split(":")),
            ))
    return out


def _cds_exon_numbers(gm: GeneModelLite) -> list[int] | None:
    """IMGT exon number of each CDS segment, or None if any is unresolved.
    Duplicates sources.cds_exon_numbers()."""
    numbers: list[int] = []
    for cs, ce in gm.cds_exons:
        match = None
        for num, es, ee in gm.exons:
            if num is not None and es <= cs and ce <= ee:
                match = num
                break
        if match is None:
            return None
        numbers.append(match)
    return numbers


def _partial_extension_match(
    gm: GeneModelLite,
    ref_seq: str,
    consensus_seq: str,
    span: AlignmentSpan,
    cds_variants: Sequence[NoveltyVariant],
    partial_records: Sequence[PartialCdsRecord],
) -> PartialCdsRecord | None:
    """First partial record whose covered exons (a) contain every CDS
    variant position and (b) match the edited sequence over that range
    exactly. Both are required; there is no partial credit. None otherwise."""
    segment_exon_numbers = _cds_exon_numbers(gm)
    if segment_exon_numbers is None:
        return None
    for rec in partial_records:
        covered_segments = [
            seg for seg, num in zip(gm.cds_exons, segment_exon_numbers)
            if num in rec.exon_numbers
        ]
        if not covered_segments:
            continue
        if any(not _in_any_range(v.pos, covered_segments) for v in cds_variants):
            continue  # a real CDS difference falls outside what this record covers
        alt_covered = _edited_cds(ref_seq, consensus_seq, span, gm, ranges=covered_segments)
        if alt_covered == rec.cds_sequence:
            return rec
    return None


def _in_any_range(pos1: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(s <= pos1 <= e for s, e in ranges)


def _in_any_numbered_range(pos1: int, ranges: Sequence[tuple[int | None, int, int]]) -> bool:
    return any(s <= pos1 <= e for _, s, e in ranges)


def _splice_site(pos1: int, introns: tuple[tuple[int | None, int, int], ...]) -> bool:
    for _, s, e in introns:
        if s <= pos1 <= s + _SPLICE_SITE_WINDOW - 1 or e - _SPLICE_SITE_WINDOW + 1 <= pos1 <= e:
            return True
    return False


def _locate(pos1: int, gm: GeneModelLite) -> str:
    if _in_any_range(pos1, gm.cds_exons):
        return "cds"
    if _in_any_numbered_range(pos1, gm.introns):
        return "intron"
    if _in_any_range(pos1, gm.utrs):
        return "utr"
    return "unannotated"


@dataclass(frozen=True)
class _RawVariant:
    pos: int  # 1-based, called_allele's own numbering
    ref: str
    alt: str
    is_insertion: bool


def _diff_against_reference(
    ref_seq: str, consensus_seq: str, span: AlignmentSpan
) -> list[_RawVariant]:
    """Diff consensus_seq against ref_seq via the captured CIGAR, in ref_seq's
    1-based coordinates."""
    aligned_ref, aligned_consensus = build_aligned_strings(
        ref_seq, consensus_seq,
        ref_start=span.ref_start, ref_end=span.ref_end,
        read_start=span.read_start, read_end=span.read_end,
        strand=span.strand, cigar=span.cigar,
    )
    variants: list[_RawVariant] = []
    ref_pos = span.ref_start
    i, n = 0, len(aligned_ref)
    while i < n:
        r, q = aligned_ref[i], aligned_consensus[i]
        if r != "-":
            if q != r:
                variants.append(_RawVariant(pos=ref_pos + 1, ref=r, alt=(q if q != "-" else ""), is_insertion=False))
            ref_pos += 1
            i += 1
        else:
            j = i
            buf = []
            while j < n and aligned_ref[j] == "-":
                buf.append(aligned_consensus[j])
                j += 1
            variants.append(_RawVariant(pos=ref_pos + 1, ref="", alt="".join(buf), is_insertion=True))
            i = j
    return variants


def _edited_cds(ref_seq: str, consensus_seq: str, span: AlignmentSpan, gm: GeneModelLite,
                 ranges: Sequence[tuple[int, int]] | None = None) -> str:
    """Edited CDS of the consensus: consensus bases at ref positions inside
    `ranges` (default gm.cds_exons); an insertion counts if the ref base it
    follows is inside. Passing a subset of exons gives an indel-safe
    partial CDS."""
    if ranges is None:
        ranges = gm.cds_exons
    aligned_ref, aligned_consensus = build_aligned_strings(
        ref_seq, consensus_seq,
        ref_start=span.ref_start, ref_end=span.ref_end,
        read_start=span.read_start, read_end=span.read_end,
        strand=span.strand, cigar=span.cigar,
    )
    out: list[str] = []
    ref_pos = span.ref_start
    last_pos_in_cds = False
    i, n = 0, len(aligned_ref)
    while i < n:
        r, q = aligned_ref[i], aligned_consensus[i]
        if r != "-":
            in_cds = _in_any_range(ref_pos + 1, ranges)
            if in_cds and q != "-":
                out.append(q)
            last_pos_in_cds = in_cds
            ref_pos += 1
            i += 1
        else:
            j = i
            buf = []
            while j < n and aligned_ref[j] == "-":
                buf.append(aligned_consensus[j])
                j += 1
            if last_pos_in_cds:
                out.append("".join(buf))
            i = j
    return "".join(out)


def _translate(cds: str, codon_start: int) -> str:
    coding = cds[codon_start - 1:]
    coding = coding[: len(coding) - len(coding) % 3]
    protein = []
    for k in range(0, len(coding), 3):
        codon = coding[k:k + 3]
        aa = _STANDARD_CODON_TABLE.get(codon, "X")
        protein.append(aa)
        if aa == "*":
            break
    return "".join(protein)


def _coding_effect(ref_protein: str, alt_protein: str, ref_cds_len: int, alt_cds_len: int) -> str:
    """Compare stop positions (_translate() stops at the first stop codon,
    so a stop is always last)."""
    if (alt_cds_len - ref_cds_len) % 3 != 0:
        return "frameshift"
    ref_has_stop = ref_protein.endswith("*")
    alt_has_stop = alt_protein.endswith("*")
    if alt_has_stop and ref_has_stop and len(alt_protein) < len(ref_protein):
        return "premature_stop"
    if alt_has_stop != ref_has_stop:
        # Stop gained or lost: both novel:null.
        return "premature_stop"
    if alt_protein != ref_protein:
        return "nonsynonymous"
    return "synonymous"


def _novel_name(display_allele: str, n_known_fields: int) -> str:
    gene, _, fields = display_allele.partition("*")
    parts = fields.split(":")
    prefix = parts[: max(n_known_fields, 0)]
    if not prefix:
        return f"{gene}*new"
    if prefix[-1] == "new":
        # Already a `*new` name; don't append another `:new`.
        return f"{gene}*{':'.join(prefix)}"
    return f"{gene}*{':'.join(prefix)}:new"


def _is_immuannot_novel(name: str) -> bool:
    return name.endswith(":new") or name.endswith("*new")


def classify(
    call: TypingCall,
    consensus_sequence: str,
    *,
    consensus_variants: Sequence[ConsensusVariant] = (),
    pool_fasta: Path = DEFAULT_POOL_FASTA,
    pool_meta: Path = DEFAULT_POOL_META,
    models_path: Path = DEFAULT_POOL_MODELS,
    cds_catalogue_path: Path = DEFAULT_CDS_CATALOGUE,
) -> NoveltyCall:
    """Stage 5 entry point. `consensus_sequence` is what Stage 4 typed.

    `consensus_variants`: Stage 3 variants, used for the read-support filter
    (omit to skip it).

    For assembly-extracted genes the displayed name is Immuannot's
    `nearest_ipd_match`; the diff is still against `called_allele`. If that
    match is itself `*new`, zero variants gives `novel:extraction`, not
    `exact`.
    """
    ref_seq = load_allele_sequences(pool_fasta, (call.called_allele,))[call.called_allele]
    gm = load_gene_model(call.called_allele, models_path)
    nearest_ipd_match = load_nearest_ipd_match(call.called_allele, pool_meta)
    display_allele = nearest_ipd_match or call.called_allele

    raw_variants = _diff_against_reference(ref_seq, consensus_sequence, call.called_allele_span)
    raw_variants = _filter_by_read_support(raw_variants, consensus_variants)

    if not raw_variants:
        if nearest_ipd_match is not None and _is_immuannot_novel(nearest_ipd_match):
            return NoveltyCall(
                gene=call.gene, called_allele=call.called_allele, taxonomy="novel:extraction",
                proposed_name=display_allele, candidate_null=False,
                extension_allele=None, coding_effect=None, variants=(),
            )
        return NoveltyCall(
            gene=call.gene, called_allele=call.called_allele, taxonomy="exact",
            proposed_name=display_allele, candidate_null=False,
            extension_allele=None, coding_effect=None, variants=(),
        )

    variants = tuple(
        NoveltyVariant(
            pos=v.pos, ref=v.ref, alt=v.alt,
            location=_locate(v.pos, gm),
            splice_site=(_locate(v.pos, gm) == "intron" and _splice_site(v.pos, gm.introns)),
        )
        for v in raw_variants
    )

    cds_variants = [v for v in variants if v.location == "cds"]
    n_fields = len(display_allele.partition("*")[2].split(":"))

    if cds_variants:
        alt_cds = _edited_cds(ref_seq, consensus_sequence, call.called_allele_span, gm)
        # Reference CDS is simpler to get directly from gm (no alignment
        # needed -- it's ref_seq's own coordinates).
        ref_cds = "".join(ref_seq[s - 1:e] for s, e in gm.cds_exons)
        ref_protein = _translate(ref_cds, gm.codon_start)
        alt_protein = _translate(alt_cds, gm.codon_start)
        effect = _coding_effect(ref_protein, alt_protein, len(ref_cds), len(alt_cds))

        catalogue = load_cds_catalogue_for_gene(call.gene, cds_catalogue_path)
        matches = catalogue.get(alt_cds, [])
        non_genomic_matches = sorted(a for a, genomic in matches if not genomic)
        if non_genomic_matches:
            extension_allele = non_genomic_matches[0]
            return NoveltyCall(
                gene=call.gene, called_allele=call.called_allele, taxonomy="extension",
                proposed_name=extension_allele, candidate_null=False,
                extension_allele=extension_allele, coding_effect=effect, variants=variants,
            )

        # Partial-CDS extension: every CDS difference must fall inside and
        # exactly match a partial record; else plain novel:*.
        partial_records = load_partial_cds_records_for_gene(call.gene, cds_catalogue_path)
        if partial_records:
            partial_match = _partial_extension_match(
                gm, ref_seq, consensus_sequence, call.called_allele_span,
                cds_variants, partial_records,
            )
            if partial_match is not None:
                return NoveltyCall(
                    gene=call.gene, called_allele=call.called_allele, taxonomy="extension:partial",
                    proposed_name=_novel_name(partial_match.allele, partial_match.n_fields),
                    candidate_null=False, extension_allele=partial_match.allele,
                    coding_effect=effect, variants=variants,
                )

        if effect in ("frameshift", "premature_stop"):
            return NoveltyCall(
                gene=call.gene, called_allele=call.called_allele, taxonomy="novel:null",
                proposed_name=_novel_name(display_allele, 1), candidate_null=True,
                extension_allele=None, coding_effect=effect, variants=variants,
            )
        if effect == "nonsynonymous":
            return NoveltyCall(
                gene=call.gene, called_allele=call.called_allele, taxonomy="novel:nonsynonymous",
                proposed_name=_novel_name(display_allele, 1), candidate_null=False,
                extension_allele=None, coding_effect=effect, variants=variants,
            )
        # Synonymous: still check intron/UTR below, but at most 2 fields agree.
        n_fields_agree = min(2, n_fields)
        if any(v.splice_site for v in variants if v.location == "intron"):
            return NoveltyCall(
                gene=call.gene, called_allele=call.called_allele, taxonomy="novel:null",
                proposed_name=_novel_name(display_allele, 1), candidate_null=True,
                extension_allele=None, coding_effect="splice_disrupting", variants=variants,
            )
        return NoveltyCall(
            gene=call.gene, called_allele=call.called_allele, taxonomy="novel:synonymous",
            proposed_name=_novel_name(display_allele, n_fields_agree), candidate_null=False,
            extension_allele=None, coding_effect=effect, variants=variants,
        )

    # No CDS-region variants at all: purely intronic/UTR (or unannotated).
    if any(v.splice_site for v in variants if v.location == "intron"):
        return NoveltyCall(
            gene=call.gene, called_allele=call.called_allele, taxonomy="novel:null",
            proposed_name=_novel_name(display_allele, 1), candidate_null=True,
            extension_allele=None, coding_effect="splice_disrupting", variants=variants,
        )
    if any(v.location == "intron" for v in variants):
        return NoveltyCall(
            gene=call.gene, called_allele=call.called_allele, taxonomy="novel:intronic",
            proposed_name=_novel_name(display_allele, min(3, n_fields)), candidate_null=False,
            extension_allele=None, coding_effect=None, variants=variants,
        )
    return NoveltyCall(
        gene=call.gene, called_allele=call.called_allele, taxonomy="novel:utr",
        proposed_name=_novel_name(display_allele, min(3, n_fields)), candidate_null=False,
        extension_allele=None, coding_effect=None, variants=variants,
    )
