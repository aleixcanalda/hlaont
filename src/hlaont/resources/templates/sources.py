"""Template pool sources.

A source yields full-length TemplateRecords for genes in the manifest. IPD is
the only source wired up today; Immuannot HPRC/CPC assemblies and MHC ALT
haplotypes (DESIGN.md S2d) are added later as new Source implementations, not
by changing this shape.
"""
from __future__ import annotations

import hashlib
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

# IMGTHLA tags its releases "vX.Y.Z-alpha" regardless of release maturity —
# that suffix is IMGTHLA's naming convention, not a quality signal.
IPD_DEFAULT_VERSION = "3.65.0"
IPD_DEFAULT_TAG = "v3.65.0-alpha"
IPD_DAT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ANHIG/IMGTHLA/{tag}/hla.dat.zip"
)


@dataclass(frozen=True)
class GeneModel:
    """Per-allele gene structure, coordinates 1-based inclusive in this
    record's own sequence numbering (DESIGN.md S5 novelty classification
    needs this: mapping a consensus variant to intronic/UTR/synonymous/
    nonsynonymous/splice requires the exact feature boundaries of the
    template that consensus was built against, not just presence/absence
    of an intron the way build.py's genomic-record filter already checks).

    `cds_exons` is the CDS's `join(...)` ranges in transcript order (or a
    single range for a one-exon CDS) -- concatenating the substrings they
    address and translating from `codon_start` gives the protein IPD's own
    /translation qualifier records, which is how a nonsynonymous/synonymous
    call is made without re-deriving a translation table.
    `exons`/`introns` are (number, start, end) from the FT table's own
    /number qualifiers (None if a record lacks one -- not expected in
    practice but not assumed). `utrs` is in position order (typically
    5' then 3'; a record can have zero, one, or two)."""

    cds_exons: tuple[tuple[int, int], ...]
    codon_start: int
    utrs: tuple[tuple[int, int], ...]
    exons: tuple[tuple[int | None, int, int], ...]
    introns: tuple[tuple[int | None, int, int], ...]


def cds_nucleotide_sequence(sequence: str, gene_model: GeneModel) -> str:
    """Concatenate `gene_model.cds_exons` ranges (1-based inclusive, this
    record's own numbering) out of `sequence` in transcript order --
    resources/templates/build.py's CDS catalogue (DESIGN.md S5's
    `extension` taxonomy: "CDS matches a known allele; IPD holds no
    full-length record") and stages/novelty.py both need this, so it lives
    here rather than being duplicated in either."""
    return "".join(sequence[s - 1:e] for s, e in gene_model.cds_exons)


def cds_exon_numbers(gene_model: GeneModel) -> list[int] | None:
    """The IMGT `/number` for each of `gene_model.cds_exons`'s segments, in
    order, resolved by containment against `gene_model.exons`'s own
    numbered FT ranges. None if any segment can't be resolved (an exon
    feature with no `/number` qualifier) -- the partial-CDS extension-
    matching gate (stages/novelty.py's `_partial_extension_match()`) must
    know exactly which exon numbers a fragment covers before treating it as
    confirming anything; an unresolvable segment means that gate silently
    doesn't apply to this record, not a guessed coverage."""
    numbers: list[int] = []
    for cs, ce in gene_model.cds_exons:
        match = None
        for num, es, ee in gene_model.exons:
            if num is not None and es <= cs and ce <= ee:
                match = num
                break
        if match is None:
            return None
        numbers.append(match)
    return numbers


@dataclass(frozen=True)
class TemplateRecord:
    gene: str
    ipd_id: str
    allele: str
    description: str
    sequence: str
    genomic: bool
    fields: int
    source: str
    source_version: str
    # True iff IPD's own FT table marks this record `/partial` (a genuinely
    # truncated CDS, e.g. exon 2-3 only, not just "not genomic") -- dropped
    # from every pool/catalogue by default (drop_partial=True below); a
    # dedicated require_genomic=False, drop_partial=False parse.py pass
    # (resources/templates/build.py) keeps these specifically to feed
    # stages/novelty.py's partial-CDS extension-matching gate, which needs
    # to see them, never to feed the actual template/alignment pool.
    partial: bool = False
    gene_model: GeneModel | None = field(default=None)
    # Immuannot's own nearest-known-IPD-match guess for this exact sequence
    # (assembly_sources.py's AssemblyGTFSource only -- None for every IPD-
    # sourced record). Not authoritative (assembly_sources.py's own
    # docstring: this is Immuannot's guess, not a curated database entry),
    # but real HLA-N/P/T/U/W/Y/R output has no other real nomenclature to
    # display at all (IPD has zero genomic records for them) -- stages/
    # novelty.py uses this purely for display naming, still computing
    # taxonomy/variants from a real diff against this record's own sequence,
    # never assuming the guess is correct.
    nearest_ipd_match: str | None = field(default=None)
    # The source GTF's own gene-line `template_allele` attribute (real
    # per-haplotype hla_seq/ assemblies only, assembly_sources.py's
    # AssemblyGTFSource) -- the exact nearest-IPD-match name Immuannot
    # aligned this haplotype's real sequence against, *never* carrying a
    # `:new` suffix (unlike nearest_ipd_match, sourced from the same GTF's
    # transcript-line `consensus` attribute, which does). Display-only, same
    # non-authoritative caveat as nearest_ipd_match: stages/output.py uses
    # this to show a real, human-readable name (e.g. "HLA-W*01:01:01:05")
    # in calls.tsv's `template_allele` column instead of this record's own
    # content-hashed `allele` accession -- it never substitutes for `allele`
    # anywhere a real pool lookup happens.
    gtf_template_allele: str | None = field(default=None)


class TemplateSource(Protocol):
    name: str
    version: str

    def records(self) -> Iterator[TemplateRecord]: ...


def fetch_ipd_dat(dest_dir: Path, tag: str = IPD_DEFAULT_TAG) -> Path:
    """Download and unpack hla.dat for a pinned IMGTHLA tag. Cached in dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dat_path = dest_dir / f"hla-{tag}.dat"
    if dat_path.exists():
        return dat_path

    zip_path = dest_dir / f"hla-{tag}.dat.zip"
    url = IPD_DAT_URL_TEMPLATE.format(tag=tag)
    print(f"[sources] downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("hla.dat")]
        if len(names) != 1:
            raise RuntimeError(
                f"expected exactly one hla.dat in {zip_path}, found {names}"
            )
        with zf.open(names[0]) as src, open(dat_path, "wb") as dst:
            dst.write(src.read())

    return dat_path


_ID_RE = re.compile(r"^ID\s+\S+;.*?(\d+)\s+BP\.")
_DE_RE = re.compile(r"^DE\s+(\S+?),\s*(.*)")
_CC_VERSION_RE = re.compile(r"Release Version (\S+)")
_GENE_FT_RE = re.compile(r'/gene="(.*)"')

# FT feature table parsing (GeneModel -- see its docstring for why). A
# feature-start line has its key immediately after the fixed "FT   " (2
# chars + 3 spaces) prefix; a continuation/qualifier line indents further
# (see resources/templates/sources.py's module notes / the raw hla.dat
# format), so anchoring the key group right after those 3 literal spaces
# is what tells the two apart -- a continuation line has a space there
# instead and fails this match.
_FT_FEATURE_RE = re.compile(r"^FT   (\S+)\s+(.*)$")
_RANGE_RE = re.compile(r"(\d+)\.\.(\d+)")
_NUMBER_QUAL_RE = re.compile(r'/number="?(\d+)"?')
_CODON_START_RE = re.compile(r"/codon_start=(\d+)")


def _finalize_cds(cur: dict) -> None:
    combined = "".join(cur["_cds_buf"])
    cur["cds_exons"] = [(int(a), int(b)) for a, b in _RANGE_RE.findall(combined)]
    cur["_loc_mode"] = None
    cur["_cds_buf"] = []


def _handle_ft_line(cur: dict, line: str) -> None:
    """Feed one FT line into `cur`'s in-progress GeneModel accumulators
    (utrs/exons/introns/cds_exons/codon_start), alongside the pre-existing
    partial/utr-count/intron-count bookkeeping `_flush` already used for
    the genomic-completeness check (kept as-is, untouched by this)."""
    if "/partial" in line:
        cur["partial"] = True

    # A /number qualifier (or its absence) resolves whatever exon/intron
    # feature is pending from the immediately preceding feature-start line
    # -- checked first, on every line, so it applies regardless of what
    # this line turns out to be otherwise (new feature, CDS continuation,
    # or an unrelated qualifier).
    pending = cur.get("_pending_feature")
    if pending is not None:
        kind, start, end = pending
        num_m = _NUMBER_QUAL_RE.search(line)
        number = int(num_m.group(1)) if num_m else None
        cur[f"{kind}s"].append((number, start, end))
        cur["_pending_feature"] = None

    cs_m = _CODON_START_RE.search(line)
    if cs_m:
        cur["codon_start"] = int(cs_m.group(1))

    m = _FT_FEATURE_RE.match(line)
    if m:
        key, rest = m.group(1), m.group(2).strip()
        if key == "UTR":
            cur["utr"] += 1
            rm = _RANGE_RE.search(rest)
            if rm:
                cur["utrs"].append((int(rm.group(1)), int(rm.group(2))))
        elif key == "exon":
            rm = _RANGE_RE.search(rest)
            if rm:
                cur["_pending_feature"] = ("exon", int(rm.group(1)), int(rm.group(2)))
        elif key == "intron":
            cur["intron"] += 1
            rm = _RANGE_RE.search(rest)
            if rm:
                cur["_pending_feature"] = ("intron", int(rm.group(1)), int(rm.group(2)))
        elif key == "CDS":
            cur["_loc_mode"] = "cds"
            cur["_cds_buf"] = [rest]
            if not rest.startswith("join(") or rest.endswith(")"):
                _finalize_cds(cur)
        return

    # Continuation line (no key at the anchored position): only meaningful
    # while accumulating a multi-line CDS join(...) location.
    if cur.get("_loc_mode") == "cds" and not line.strip().startswith("/"):
        cur["_cds_buf"].append(line.strip())
        if line.strip().endswith(")"):
            _finalize_cds(cur)


def _build_gene_model(cur: dict) -> GeneModel | None:
    if not cur["cds_exons"] and not cur["exons"] and not cur["introns"] and not cur["utrs"]:
        return None
    return GeneModel(
        cds_exons=tuple(cur["cds_exons"]),
        codon_start=cur["codon_start"],
        utrs=tuple(cur["utrs"]),
        exons=tuple(cur["exons"]),
        introns=tuple(cur["introns"]),
    )


def _flush(cur: dict | None, genomic_genes: set[str], source_version: str,
           require_genomic: bool = True, drop_partial: bool = True) -> TemplateRecord | None:
    if cur is None or cur["gene"] is None:
        return None
    if drop_partial and cur["partial"]:
        return None
    if cur["gene"] not in genomic_genes:
        return None
    genomic = cur["utr"] >= 2 and cur["intron"] > 0
    if require_genomic and not genomic:
        return None
    seq = "".join(cur["seq_lines"])
    if not seq:
        return None
    return TemplateRecord(
        gene=cur["gene"],
        ipd_id=cur["id"],
        allele=cur["allele"],
        description=cur["description"],
        sequence=seq,
        genomic=genomic,
        fields=cur["fields"],
        source="IPD-IMGT/HLA",
        source_version=source_version,
        partial=cur["partial"],
        gene_model=_build_gene_model(cur),
    )


def parse_ipd_dat(dat_path: Path, genomic_genes: set[str],
                   require_genomic: bool = True, drop_partial: bool = True) -> Iterator[TemplateRecord]:
    """Stream hla.dat, yielding one TemplateRecord per allele whose gene is
    in genomic_genes.

    require_genomic=True (default, template pool use): additionally require
    "genomic" -- both UTRs plus at least one intron present (DESIGN.md S2b).
    Same criterion as ipd_audit.py, not FuFiHLA's coarser `grep -v partial` on
    the description line.

    require_genomic=False (anchor decoy use, and the CDS-catalogue build):
    any record, since a decoy anchor only needs to represent a paralog's
    sequence well enough to catch stray reads -- it is never typed against.

    drop_partial=True (default): drop records IPD's own FT table marks
    `/partial` (a genuinely truncated CDS, not just non-genomic). Only
    resources/templates/build.py's CDS-catalogue pass sets this False --
    stages/novelty.py's partial-CDS extension-matching gate needs to see
    these specifically; the real template/alignment pool never does.
    """
    cur: dict | None = None
    in_seq = False
    source_version = "unknown"

    with open(dat_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("ID   "):
                rec = _flush(cur, genomic_genes, source_version, require_genomic, drop_partial)
                if rec is not None:
                    yield rec
                m = _ID_RE.match(line)
                cur = {
                    "id": None, "gene": None, "allele": None, "description": "",
                    "partial": False, "utr": 0, "intron": 0,
                    "len": int(m.group(1)) if m else 0, "fields": 0,
                    "seq_lines": [],
                    "cds_exons": [], "codon_start": 1, "utrs": [],
                    "exons": [], "introns": [],
                    "_loc_mode": None, "_cds_buf": [], "_pending_feature": None,
                }
                in_seq = False
                id_m = re.match(r"^ID\s+(\S+?);", line)
                if id_m:
                    cur["id"] = id_m.group(1)
                continue

            if cur is None:
                continue

            if line.startswith("CC") and "Release Version" in line:
                vm = _CC_VERSION_RE.search(line)
                if vm:
                    source_version = vm.group(1)
                continue

            if line.startswith("DE   ") and cur["allele"] is None:
                m = _DE_RE.match(line)
                if m:
                    name, desc = m.group(1), m.group(2).strip()
                    cur["allele"] = name
                    cur["description"] = desc
                    if "*" in name:
                        cur["gene"] = name.split("*")[0]
                        cur["fields"] = len(name.split("*")[1].split(":"))
                continue

            if line.startswith("FT"):
                _handle_ft_line(cur, line)
                continue

            if line.startswith("SQ"):
                in_seq = True
                continue

            if line.startswith("//"):
                in_seq = False
                continue

            if in_seq:
                cleaned = re.sub(r"[\s\d]", "", line).upper()
                cur["seq_lines"].append(cleaned)

    rec = _flush(cur, genomic_genes, source_version, require_genomic, drop_partial)
    if rec is not None:
        yield rec


class IPDSource:
    """IPD-IMGT/HLA, pinned by GitHub tag (DESIGN.md: "Pin it")."""

    name = "IPD-IMGT/HLA"

    def __init__(self, genomic_genes: set[str], raw_dir: Path, tag: str = IPD_DEFAULT_TAG,
                 dat_path: Path | None = None, require_genomic: bool = True,
                 drop_partial: bool = True):
        self.genomic_genes = genomic_genes
        self.tag = tag
        self.raw_dir = raw_dir
        self._dat_path = dat_path
        self.require_genomic = require_genomic
        self.drop_partial = drop_partial
        self.version = IPD_DEFAULT_VERSION if tag == IPD_DEFAULT_TAG else tag
        self.dat_sha256: str | None = None

    def _resolve_dat(self) -> Path:
        if self._dat_path is not None:
            return self._dat_path
        return fetch_ipd_dat(self.raw_dir, self.tag)

    def records(self) -> Iterator[TemplateRecord]:
        dat_path = self._resolve_dat()
        h = hashlib.sha256()
        with open(dat_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        self.dat_sha256 = h.hexdigest()
        yield from parse_ipd_dat(dat_path, self.genomic_genes, self.require_genomic, self.drop_partial)
