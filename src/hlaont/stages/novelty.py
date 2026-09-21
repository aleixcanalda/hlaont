"""Stage 5 -- novelty classification (DESIGN.md S5).

Classifies a Stage 4 TypingCall's consensus against its own `called_allele`
(the nearest known IPD allele) using the exact CDS/exon/intron/UTR
coordinates resources/templates/build.py now captures per allele
(resources/templates/pool/ref.gene.models.jsonl.gz -- see
resources/templates/sources.py's GeneModel). The diff itself reuses
`called_allele_span` (stages/hla_typing.py's captured consensus-vs-
called_allele CIGAR) through `error_model.build_aligned_strings`, the same
alignment-expansion helper stages/consensus.py uses -- no new alignment
step.

Mandatory taxonomy (DESIGN.md S5):
    exact | extension | novel:intronic | novel:utr |
    novel:synonymous | novel:nonsynonymous | novel:null

Plus two project-added tags. `novel:extraction` (2026-08-23, user-requested):
scoped to assembly-extraction genes only (HLA-N/P/T/U/W/Y/R, resources/
templates/assembly_sources.py) -- see `classify()`'s own docstring for why
a real diff of zero variants against Immuannot's own `*new`-tagged guess
cannot honestly be reported as plain `exact`. `extension:partial`
(2026-09-20, user-specified gate, real HG03009/HLA-C*07:457 case): a
CDS-level novel:* call whose entire set of CDS differences from
called_allele exactly matches an IPD allele that IPD only ever recorded as
a *partial* (exon-subset, e.g. exon 2-3 only) submission -- see
`_partial_extension_match()`'s own docstring for the exact all-or-nothing
gate (every differing position must fall inside, and exactly match, the
partial record's own covered exon range; any leftover difference outside
that range, or any mismatch inside it, and the call stays plain novel:*,
same called_allele -- never a partial-credit rename).

**`extension` vs `novel` is invariant 7's named risk** ("getting this wrong
inflates the headline number and is the mistake most likely to embarrass
us"). `extension` means the consensus's CDS nucleotide sequence exactly
matches some *other* already-known IPD allele that IPD only ever recorded
as exon-only (no genomic template of its own, so it could never enter
`called_allele`'s genomic-only candidate pool in the first place) --
resources/templates/pool/ref.gene.cds_catalogue.tsv.gz (built from a
require_genomic=False parse, DESIGN.md S5) is the lookup this needs.
`extension:partial` is the same idea one step weaker: IPD's own record of
that other allele doesn't even cover the whole CDS, only some of its
exons, so only that fragment can honestly be confirmed as "known" -- see
the module-level note above and `_partial_extension_match()`.
**Scope note, not silently glossed over:** if the edited CDS instead
exactly matches a *different genomic* allele than `called_allele`, this
module does not re-route to that allele -- Stage 4's own candidate search
already had that allele available and chose `called_allele` by whole-
sequence likelihood; reconciling the two is deferred, not guessed at here.

Splice-site handling is a bounded heuristic, not a splice-effect predictor:
a variant in the first/last 2bp of an intron (the canonical GT...AG donor/
acceptor dinucleotides) is treated as splice-disrupting and escalated to
`novel:null`, per DESIGN.md S5's "an intronic variant at a 3' acceptor is
the difference between a curiosity and a null allele."

Naming follows Immuannot's `new`-tag convention (DESIGN.md S5): the called
allele's fields up to the point they still agree, then literal `:new` --
e.g. a field-4-only difference on a 4-field called_allele keeps 3 fields
(DESIGN.md's own `HLA-A*24:02:01:new` example). This project's own
convention, not independently re-verified against Immuannot's source (cite
regardless, per DESIGN.md S9's provenance stance) -- a `candidate_null`
flag stands in for the "N suffix" DESIGN.md mentions rather than guessing
at Immuannot's exact null-suffix assignment rule.
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

# Splice-site proximity window (canonical GT...AG donor/acceptor
# dinucleotides) -- see module docstring; a bounded heuristic, not a fitted
# or re-derived-for-ONT threshold (nothing to fit it against), so exempt
# from invariant 11 in the same sense a structural biology convention is,
# not a statistical one.
_SPLICE_SITE_WINDOW = 2

# Read-support floor for a variant to count toward the novelty taxonomy
# decision (2026-09-13, user-directed, following the 2026-09-11/12
# consensus-QV investigation logged in development notes). stages/consensus.py's
# per-position qv saturates at 60 regardless of real read disagreement
# (proven on real data: a genuine 21-vs-9 dissenting vote still reports
# qv=60 at every tested error rate), so it cannot gate novelty on its own.
# The raw read-support fraction -- (depth-dissent)/depth, unsmoothed --
# was checked instead against 99 real labeled calls across 7 samples
# (`.debug/sanity_check_dissent_qv_v3.tsv`): dropping a variant below this
# floor, then only still calling a haplotype novel if at least one variant
# survives (OR-across-positions, not AND -- an AND-across-every-position
# gate was tried first and found to punish real multi-variant TPs for
# having more chances to fail), moved novelty precision 64.7% -> 80.2% at
# a recall cost of 90.0% -> 77.0% on that dataset. Uniform across genes by
# user's explicit choice, despite a known, accepted regression on MICA
# specifically (4 TP -> 1 TP there) -- not re-derived/fitted per invariant
# 11, the user's own chosen value inside the range checked.
_MIN_VARIANT_SUPPORT = 0.8


def _read_support(cv: ConsensusVariant) -> float:
    if cv.depth == 0:
        return 0.0
    return (cv.depth - cv.dissent) / cv.depth


def _filter_by_read_support(
    raw_variants: list["_RawVariant"], consensus_variants: Sequence[ConsensusVariant]
) -> list["_RawVariant"]:
    """Drops a raw variant below `_MIN_VARIANT_SUPPORT` read support.
    `consensus_variants` empty (the default, and every pre-2026-09-13
    caller) is a no-op -- exact backward compatibility for anything not
    yet passing Stage 3's `HaplotypeConsensus.variants` through. A raw
    variant with no matching `ConsensusVariant` (positions should always
    line up, both built from the same consensus sequence, but this is
    defensive) is kept rather than dropped -- "no information, not a
    zero", the same stance `_haplotype_pileup()` already takes."""
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
    """Immuannot's own nearest-known-IPD-match guess for `allele`
    (ref.gene.meta.tsv.gz's `nearest_ipd_match` column, resources/templates/
    assembly_sources.py) -- None for every IPD-sourced allele (empty string
    in the file) and for any allele not found at all (the ordinary case for
    every synthetic test fixture's placeholder gene, matching this module's
    own load_gene_model precedent of not guessing at a missing lookup)."""
    with gzip.open(pool_meta, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["allele"] == allele:
                return row.get("nearest_ipd_match") or None
    return None


def load_gtf_template_allele(allele: str, pool_meta: Path = DEFAULT_POOL_META) -> str | None:
    """The source hla_seq/ GTF's own gene-line `template_allele` attribute
    for `allele` (ref.gene.meta.tsv.gz's `gtf_template_allele` column,
    resources/templates/assembly_sources.py) -- None for every IPD-sourced
    allele and for any allele not found, same precedent as
    load_nearest_ipd_match. Unlike nearest_ipd_match (sourced from the same
    GTF's transcript-line `consensus` attribute), this is never `:new`-
    suffixed -- stages/output.py uses it purely to show a real,
    human-readable nearest-allele name in calls.tsv's `template_allele`
    column instead of an assembly-extracted record's content-hashed
    accession (e.g. "HLA-W*asm3e9f0510aa")."""
    with gzip.open(pool_meta, "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["allele"] == allele:
                return row.get("gtf_template_allele") or None
    return None


def load_cds_catalogue_for_gene(
    gene: str, catalogue_path: Path = DEFAULT_CDS_CATALOGUE
) -> dict[str, list[tuple[str, bool]]]:
    """cds_sequence -> [(allele, genomic), ...] for every non-partial IPD
    record of `gene` (build.py's require_genomic=False pass) -- the
    extension-detection lookup (module docstring). Rows with `partial=True`
    (2026-09-20) are excluded here on purpose, even though harmless in
    practice (a partial record's cds_sequence is a strict fragment of any
    real multi-exon CDS, so it could never collide with a full `alt_cds`
    dict lookup) -- kept out anyway so this function's behaviour is exactly
    what it was before partial rows existed in the catalogue at all.
    `_partial_extension_match()` reads partial rows separately, via
    `load_partial_cds_records_for_gene()`."""
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
    """Partial-CDS IPD records for `gene` (ref.gene.cds_catalogue.tsv.gz's
    `partial=True` rows, build.py) -- real IPD alleles whose only
    catalogued sequence is an incomplete (exon-subset) submission, so they
    never enter the genomic pool and can never satisfy the plain exact-
    full-CDS `extension` lookup above either. Rows with no resolvable
    `exon_numbers` (build.py already leaves this column empty for those,
    see write_cds_catalogue_tsv_gz) are skipped -- `_partial_extension_match()`
    cannot safely use a record whose own coverage isn't known."""
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
    """The IMGT `/number` for each of gm.cds_exons's segments, in order,
    resolved by containment against gm.exons's own numbered FT ranges. None
    if any segment can't be resolved. Deliberately duplicated from
    resources/templates/sources.py's cds_exon_numbers() (same shape, same
    logic) rather than imported -- this module already keeps its own
    GeneModelLite rather than importing sources.py's GeneModel, same
    precedent."""
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
    """First partial_records entry whose own covered exon numbers (a)
    contain every cds_variants position and (b) reproduce that exact
    covered range's edited sequence, byte for byte -- both required. (a)
    alone is not enough: a partial record could cover some of the varying
    positions without covering all of them, which would silently ignore a
    real difference outside its own coverage. (b) alone is not enough
    either: it would let a short fragment match by coincidence against an
    arbitrary substring rather than the specific exon range it actually
    claims to cover. This is the user-specified gate, precisely: "zero
    mismatch, complete match [over the partial's own covered fields], or
    leave the novel call and same template" -- there is no partial-credit
    return value, only a full match or None.

    None if this called_allele's own exon numbering can't be resolved
    (_cds_exon_numbers() returns None) or no partial record satisfies both
    conditions -- the existing exact-full-CDS `extension` tier and plain
    novel:* taxonomy are both unaffected either way; this is purely
    additive."""
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
    """Diff consensus_seq against ref_seq using the already-captured CIGAR
    (span), in ref_seq's own 1-based coordinates -- no per-read voting
    needed (unlike stages/consensus.py's multi-read pileup): there is only
    one sequence being compared, the already-polished consensus."""
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
    """Reconstruct the consensus's own CDS nucleotide sequence by walking
    the same gapped alignment _diff_against_reference() uses, keeping only
    consensus bases anchored at (or, for an insertion, immediately after) a
    ref_seq position inside `ranges` (default gm.cds_exons) -- an insertion
    is attributed to the CDS iff the ref position it sits after is itself
    in the CDS (module docstring: a simple, documented convention for the
    exon/intron boundary case, not a splice-aware placement).

    `ranges` (2026-09-20): `_partial_extension_match()` calls this a second
    time with a *subset* of gm.cds_exons (just the segments a partial IPD
    record's own coverage corresponds to) to reconstruct only that portion
    of the edited CDS -- indel-safe (unlike slicing the already-built full
    alt_cds by a precomputed reference-length offset, which would drift at
    any indel inside the covered range), by construction, since it reuses
    this same alignment walk rather than doing post-hoc arithmetic on it."""
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
    """_translate() stops at the first in-frame stop codon it finds, so a
    stop in its output is always the *last* character -- there is no
    "stop buried mid-string" case to check for. Compares stop *position*
    (present/absent, and how early) between ref and alt instead."""
    if (alt_cds_len - ref_cds_len) % 3 != 0:
        return "frameshift"
    ref_has_stop = ref_protein.endswith("*")
    alt_has_stop = alt_protein.endswith("*")
    if alt_has_stop and ref_has_stop and len(alt_protein) < len(ref_protein):
        return "premature_stop"
    if alt_has_stop != ref_has_stop:
        # A stop gained where the reference had none, or lost entirely --
        # both a disrupted reading-frame termination. DESIGN.md S5 has one
        # `novel:null` bucket for "frameshift / PTC / splice-disrupting",
        # not finer subtypes, so both collapse to the same label here.
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
        # display_allele is already an Immuannot `*new`-tagged guess
        # (assembly-derived genes, module docstring) -- appending a second
        # `:new` would read as "HLA-P*new:new", not a real convention.
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
    """Stage 5 entry point. `consensus_sequence` is the same
    HaplotypeConsensus.sequence Stage 4 typed (call.called_allele_span was
    computed against it).

    `consensus_variants` (optional): the same HaplotypeConsensus.variants
    Stage 3 built alongside `consensus_sequence`, used only to apply
    `_MIN_VARIANT_SUPPORT`'s read-support filter (see that constant's own
    comment) before a raw diff position counts toward the taxonomy
    decision below. Omitting it (the default) reproduces this function's
    pre-2026-09-13 behaviour exactly.

    **`display_allele` (2026-08-23, user-requested):** for assembly-
    extraction genes (HLA-N/P/T/U/W/Y/R, resources/templates/
    assembly_sources.py), `call.called_allele` is a content-hash accession
    (`HLA-N*asm<sha256>`) -- a real, stable identity, but not a name that
    looks like every other gene's output. `ref.gene.meta.tsv.gz`'s
    `nearest_ipd_match` column (Immuannot's own nearest-known-IPD-match
    guess for this exact sequence) is used for *display* only when present
    -- `taxonomy`/`variants` still come from a real diff against
    `called_allele`'s own sequence, never assumed correct just because a
    friendlier name is shown (the guess is not treated as authoritative,
    same stance assembly_sources.py's own docstring takes). If Immuannot's
    own guess is itself `*new`-tagged (no real IPD catalog hit at all), a
    zero-variant diff against our own extracted reference cannot honestly
    be reported as `exact` -- that would claim a catalogued match that
    doesn't exist -- so it gets the dedicated `novel:extraction` taxonomy
    instead (this project's own addition to DESIGN.md S5's six mandatory
    tags, scoped to exactly this case). Ordinary IPD-sourced genes have no
    `nearest_ipd_match` at all (empty column), so `display_allele` is just
    `called_allele` and every existing code path/output is unchanged.
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

        # Partial-CDS extension (2026-09-20, real HG03009/HLA-C*07:457 case,
        # user-specified gate): a CDS-level novel:* call (fields 1-3 -- this
        # branch is already scoped to cds_variants, so novel:intronic/utr
        # calls below never reach here) whose *entire* set of CDS
        # differences from called_allele falls inside, and exactly matches,
        # some IPD allele that only exists as a partial (exon-subset)
        # record. Fields beyond what that partial record itself specifies
        # stay `:new` -- this never claims more than the partial record
        # actually confirms. Any leftover difference outside the partial's
        # own coverage, or any mismatch within it, and this returns None --
        # falls straight through to the ordinary novel:* taxonomy below,
        # unchanged, same called_allele.
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
        # Synonymous CDS change -- fall through to also check intron/UTR
        # variants below, but a coding (field-3) difference already sets
        # the floor for how many fields can be reported "known".
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
