"""Assembly-derived template source (sources.py's TemplateSource protocol),
for HLA genes IPD-IMGT/HLA has no usable genomic template for at all
(HLA-N/P/T/U/W) or only one, ancestry-skewed template for (HLA-Y --
DESIGN.md's own flagged risk: a single European-derived template fails to
recruit reads from the ~20% of haplotypes, mostly non-European, that
actually carry HLA-Y, producing an ancestry-structured false-negative CN=0).

Real per-individual phased genome assemblies (hla_seq/, 232 samples / 461
haplotype copies) already carry an Immuannot-style GTF annotation with exact
gene coordinates on a matching per-haplotype assembly contig -- this module
extracts the gene span plus its exon/CDS/UTR structure directly from that
real genomic DNA, converts it into the same TemplateRecord/GeneModel shape
sources.py's IPD parser produces, and lets build.py fold it into the pool
as a second source.

Deliberately does NOT use the GTF's own `consensus`/`template_allele`
attributes as the accession name -- those are Immuannot's own nearest-known-
IPD-match guess, not an authoritative catalogued name, and treating them as
one would misrepresent a real individual's sequence as a curated database
entry (invariant 7: extension != novel; this is the same mistake in a
different guise). Accessions are instead content-addressed (sha256 of the
extracted sequence), so a later rerun that adds more hla_seq samples never
renumbers an existing accession the way a sort-then-enumerate scheme would.

GTF quirks below are all confirmed against real hla_seq files, not assumed:
- No `intron` feature rows exist at all -- introns are derived as gaps
  between consecutively-*present* `exon_number` rows, never synthesized for
  a missing leading exon (a real, partial pseudogene annotation -- e.g. some
  HLA-W haplotypes are missing exons 1-2 -- must not get a fabricated
  intron inserted before the first present exon).
- On minus-strand genes, `exon_number "1"` is already the highest-genomic-
  coordinate row (i.e. numbering already matches transcript order) -- only
  coordinate *values* need the revcomp/flip transform, not the ordinals.
- GTF's `frame` column (0/1/2, GFF convention: bases to trim from this
  feature's start to reach the next full codon) is the inverse sense from
  IPD's own `codon_start` (1/2/3, bases already consumed): frame=0 means no
  bases skipped (codon_start=1), so `codon_start = frame + 1`.
- A gene with zero UTR rows, or missing start/stop codons, is real
  pseudogene degeneracy confirmed on real HLA-N (single-exon, no introns)
  and HLA-W (partial, no UTRs, no start/stop codon) records -- not
  annotation failure, never papered over or fabricated.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import pysam

sys.path.insert(0, str(Path(__file__).parent))
from sources import GeneModel, TemplateRecord  # noqa: E402

# Fixed pin (mirrors sources.py's IPD_DEFAULT_TAG) -- hla_seq/ has no
# upstream release tag of its own to pin against; chosen from this
# snapshot's real on-disk file mtimes. Bumped 2026-09-03, user-requested:
# hla_seq/ gained a full re-annotation pass (`*_HLA_v2.gtf.gz`, 462 real
# files, ## date: 2026-09-01/02) alongside the original `*_HLA.gtf.gz`
# (461 files, Mar 2025) -- confirmed by direct grep this is not a cosmetic
# rerun: the v2 pass finds real HLA-R gene annotation on 90/462 haplotypes,
# where the original snapshot had zero (genes.yaml's own HLA-R comment
# previously said so, now stale -- update it alongside this). `records()`
# below now prefers `*_HLA_v2.gtf.gz` when present, falling back to the
# legacy `*_HLA.gtf.gz` naming only when it isn't (keeps every existing
# unit test's `_write_gtf()`-fixture naming working unchanged).
HLA_SEQ_SNAPSHOT_VERSION = "hla_seq-2026-09-v2"

_V2_GTF_SUFFIX = "_HLA_v2.gtf.gz"
_LEGACY_GTF_SUFFIX = "_HLA.gtf.gz"

# Pre-extracted, checked-in snapshot of AssemblyGTFSource's own output
# (BundledAssemblySource below) -- lets build.py cover HLA-N/P/T/U/W/Y/R
# without a caller needing real hla_seq/ phased assemblies on disk at all.
DEFAULT_BUNDLE_DIR = Path(__file__).parent / "assembly_bundle"

_SEQNAME_SUFFIX_RE = re.compile(r":\d+-\d+$")
_ATTR_RE = re.compile(r'(\S+) "([^"]*)"')
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _parse_attrs(field9: str) -> dict[str, str]:
    return dict(_ATTR_RE.findall(field9))


def _open_maybe_gz(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else open(path, "r")


def _iter_gtf_rows(gtf_path: Path) -> Iterator[tuple[str, str, int, int, str, str, dict]]:
    """Yield (seqname, feature, start, end, strand, frame, attrs) per row."""
    with _open_maybe_gz(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            seqname, _src, feature, start, end, _score, strand, frame, attrs_str = f[:9]
            yield seqname, feature, int(start), int(end), strand, frame, _parse_attrs(attrs_str)


def _contig_name(seqname: str) -> str:
    """The GTF seqname carries a `:start-end` suffix the matching FASTA
    header does not (confirmed real: GTF seqname
    `HG00097#1#JBIRDD010000044.1:28564748-33733244` vs. FASTA header
    `HG00097#1#JBIRDD010000044.1`) -- strip it to get the real contig name."""
    return _SEQNAME_SUFFIX_RE.sub("", seqname)


class AssemblyGTFSource:
    """Coordinate-lookup extraction for genes with real GTF gene-level
    annotation in hla_seq/ (HLA-N, HLA-P, HLA-T, HLA-U, HLA-W, HLA-Y, and
    HLA-R since the 2026-09-03 v2 re-annotation (module docstring) --
    driven entirely by the `genes` argument, itself driven by genes.yaml's
    `assembly_extraction: gtf_coordinate` field, never hardcoded here per
    invariant 5)."""

    def __init__(self, genes: set[str], hla_seq_dir: Path,
                 all_fasta_path: Path | None = None,
                 version: str = HLA_SEQ_SNAPSHOT_VERSION):
        self.name = "hla_seq-assemblies-gtf"
        self.version = version
        self.genes = genes
        self.hla_seq_dir = hla_seq_dir
        self.all_fasta_path = all_fasta_path or (hla_seq_dir / "all_HLA.fasta.gz")

    def records(self) -> Iterator[TemplateRecord]:
        gtf_paths = sorted(self.hla_seq_dir.glob(f"*{_V2_GTF_SUFFIX}"))
        suffix = _V2_GTF_SUFFIX
        if not gtf_paths:
            gtf_paths = sorted(self.hla_seq_dir.glob(f"*{_LEGACY_GTF_SUFFIX}"))
            suffix = _LEGACY_GTF_SUFFIX
        by_gene_seq: dict[str, dict[str, dict]] = {g: {} for g in self.genes}

        with pysam.FastaFile(str(self.all_fasta_path)) as fasta:
            for gtf_path in gtf_paths:
                haplotype_id = gtf_path.name[: -len(suffix)]
                self._extract_from_gtf(gtf_path, haplotype_id, fasta, by_gene_seq)

        for gene in sorted(self.genes):
            for seq in sorted(by_gene_seq[gene]):
                yield self._make_record(gene, seq, by_gene_seq[gene][seq])

    def _extract_from_gtf(self, gtf_path: Path, haplotype_id: str, fasta,
                           by_gene_seq: dict[str, dict[str, dict]]) -> None:
        genes_rows: dict[tuple[str, str], dict] = {}
        for seqname, feature, start, end, strand, frame, attrs in _iter_gtf_rows(gtf_path):
            gene_name = attrs.get("gene_name")
            if gene_name not in self.genes:
                continue
            gene_id = attrs.get("gene_id", gene_name)
            key = (gene_name, gene_id)
            row = genes_rows.setdefault(key, {
                "seqname": None, "gene_start": None, "gene_end": None, "strand": strand,
                "exons": [], "cds": [], "utrs": [], "consensus": None, "template_allele": None,
            })
            if feature == "gene":
                row["seqname"] = seqname
                row["gene_start"], row["gene_end"] = start, end
                row["strand"] = strand
                row["template_allele"] = attrs.get("template_allele")
            elif feature == "exon":
                # Real quirk, confirmed on HLA-P: a purely-UTR trailing exon
                # (no CDS content of its own) is numbered "utr", not an
                # integer -- GeneModel.exons' own type (int | None) already
                # anticipates this; treat it as unnumbered rather than
                # guessing an ordinal, and it simply doesn't participate in
                # intron-gap derivation below (no fabricated adjacency).
                num_str = attrs.get("exon_number", "")
                num = int(num_str) if num_str.isdigit() else None
                row["exons"].append((num, start, end))
            elif feature == "CDS":
                row["cds"].append((start, end, frame))
            elif feature == "UTR":
                row["utrs"].append((start, end))
            elif feature == "transcript":
                row["consensus"] = attrs.get("consensus")

        for (gene_name, _gene_id), row in genes_rows.items():
            if row["seqname"] is None:
                continue  # a gene row itself missing/malformed -- skip, don't guess
            self._build_and_store(gene_name, haplotype_id, row, fasta, by_gene_seq)

    def _build_and_store(self, gene_name: str, haplotype_id: str, row: dict, fasta,
                          by_gene_seq: dict[str, dict[str, dict]]) -> None:
        contig = _contig_name(row["seqname"])
        gene_start, gene_end, strand = row["gene_start"], row["gene_end"], row["strand"]
        raw_seq = fasta.fetch(contig, gene_start - 1, gene_end).upper()
        sequence = _revcomp(raw_seq) if strand == "-" else raw_seq

        def shift(s: int, e: int) -> tuple[int, int]:
            if strand == "-":
                return (gene_end - e + 1, gene_end - s + 1)
            return (s - gene_start + 1, e - gene_start + 1)

        utrs = tuple(sorted(shift(s, e) for s, e in row["utrs"]))
        exons_shifted = sorted(
            ((num, *shift(s, e)) for num, s, e in row["exons"]),
            key=lambda x: x[1],
        )

        introns: list[tuple[int, int, int]] = []
        by_number = {num: (s, e) for num, s, e in exons_shifted if num is not None}
        numbers = sorted(by_number)
        for a, b in zip(numbers, numbers[1:]):
            if b == a + 1:
                s, e = by_number[a][1] + 1, by_number[b][0] - 1
                if e >= s:
                    introns.append((a, s, e))

        cds_shifted = sorted(shift(s, e) for s, e, _frame in row["cds"])
        codon_start = 1
        if row["cds"]:
            first = min(row["cds"], key=lambda c: shift(c[0], c[1])[0])
            codon_start = int(first[2]) + 1  # GTF frame -> IPD codon_start

        gene_model = None
        if cds_shifted or exons_shifted or introns or utrs:
            gene_model = GeneModel(
                cds_exons=tuple(cds_shifted),
                codon_start=codon_start,
                utrs=utrs,
                exons=tuple((num, s, e) for num, s, e in exons_shifted),
                introns=tuple(introns),
            )

        entry = by_gene_seq[gene_name].setdefault(sequence, {
            "haplotypes": [], "consensus_names": set(), "gene_model": gene_model,
            "template_allele_names": set(),
        })
        entry["haplotypes"].append(haplotype_id)
        if row["consensus"]:
            entry["consensus_names"].add(row["consensus"])
        if row["template_allele"]:
            entry["template_allele_names"].add(row["template_allele"])

    def _make_record(self, gene: str, seq: str, info: dict) -> TemplateRecord:
        h = hashlib.sha256(seq.encode()).hexdigest()
        allele = f"{gene}*asm{h[:10]}"
        ipd_id = f"ASM{h[:12]}"
        n_haps = len(info["haplotypes"])
        example = info["haplotypes"][0]
        # Every haplotype contributing to this exact-sequence group went
        # through the same real Immuannot alignment against the same IPD
        # reference, so this should always be a single guess in practice;
        # a real, non-unanimous split (multiple distinct guesses for byte-
        # identical DNA) is kept, not silently collapsed -- sorted()[0] is
        # the deterministic pick (invariant 9), and the full set stays in
        # `description` for anyone auditing the call.
        consensus_names = sorted(info["consensus_names"])
        consensus_note = ",".join(consensus_names) or "n/a"
        nearest_ipd_match = consensus_names[0] if consensus_names else None
        # Same deterministic-pick/audit-trail treatment as consensus_names
        # above, but sourced from the GTF's gene-line `template_allele`
        # attribute rather than the transcript-line `consensus` attribute --
        # the exact nearest-match name, never `:new`-suffixed (module
        # docstring, TemplateRecord.gtf_template_allele's own docstring).
        template_allele_names = sorted(info["template_allele_names"])
        template_allele_note = ",".join(template_allele_names) or "n/a"
        gtf_template_allele = template_allele_names[0] if template_allele_names else None
        description = (
            f"assembly-derived ({self.version}); observed on {n_haps} "
            f"haplotype(s), e.g. {example}; Immuannot nearest-match "
            f"(not authoritative): {consensus_note}; GTF template_allele: "
            f"{template_allele_note}"
        )
        return TemplateRecord(
            gene=gene, ipd_id=ipd_id, allele=allele, description=description,
            sequence=seq, genomic=True, fields=1,
            source=self.name, source_version=self.version,
            gtf_template_allele=gtf_template_allele,
            gene_model=info["gene_model"], nearest_ipd_match=nearest_ipd_match,
        )


class BundledAssemblySource:
    """Reads back a bundle written by build_assembly_bundle.py
    (bundle.fa.gz / bundle.meta.tsv.gz / bundle.models.jsonl.gz) --
    reconstructs the identical TemplateRecords AssemblyGTFSource would
    yield from the real hla_seq/ directory the bundle was generated from,
    without needing that (large, external) directory on disk at all. Each
    record's own `source`/`source_version` fields are read verbatim from
    the bundle's meta row (i.e. still "hla_seq-assemblies-gtf" /
    HLA_SEQ_SNAPSHOT_VERSION, the real origin) -- this class's own
    `name`/`version` identify the *loading mechanism* for manifest.json's
    provenance, not a different data source.

    Use AssemblyGTFSource directly (build.py's --hla-seq-dir) instead when
    a fresher or larger hla_seq/ snapshot is available to extract from;
    regenerate this bundle from that via build_assembly_bundle.py to update
    what ships in the repo."""

    def __init__(self, genes: set[str], bundle_dir: Path = DEFAULT_BUNDLE_DIR):
        self.name = "hla_seq-assemblies-bundle"
        self.version = HLA_SEQ_SNAPSHOT_VERSION
        self.genes = genes
        self.bundle_dir = bundle_dir

    def records(self) -> Iterator[TemplateRecord]:
        meta_path = self.bundle_dir / "bundle.meta.tsv.gz"
        models_path = self.bundle_dir / "bundle.models.jsonl.gz"
        fasta_path = self.bundle_dir / "bundle.fa.gz"
        for p in (meta_path, models_path, fasta_path):
            if not p.exists():
                raise ValueError(
                    f"assembly bundle missing {p} -- rerun build_assembly_bundle.py "
                    f"against a real hla_seq/ directory, or pass --hla-seq-dir directly"
                )

        gene_models: dict[str, GeneModel] = {}
        with gzip.open(models_path, "rt") as fh:
            for line in fh:
                obj = json.loads(line)
                gene_models[obj["ipd_id"]] = GeneModel(
                    cds_exons=tuple(tuple(x) for x in obj["cds_exons"]),
                    codon_start=obj["codon_start"],
                    utrs=tuple(tuple(x) for x in obj["utrs"]),
                    exons=tuple(tuple(x) for x in obj["exons"]),
                    introns=tuple(tuple(x) for x in obj["introns"]),
                )

        sequences: dict[str, str] = {}
        with pysam.FastxFile(str(fasta_path)) as fh:
            for rec in fh:
                if rec.name is None or rec.sequence is None:
                    raise ValueError(f"{fasta_path}: record {rec.name!r} has no name/sequence")
                sequences[rec.name] = rec.sequence

        with gzip.open(meta_path, "rt") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in fh]

        for row in sorted(rows, key=lambda r: (r["gene"], r["allele"], r["ipd_id"])):
            if row["gene"] not in self.genes:
                continue
            allele = row["allele"]
            if allele not in sequences:
                raise ValueError(f"{fasta_path}: missing sequence for {allele!r}")
            yield TemplateRecord(
                gene=row["gene"], ipd_id=row["ipd_id"], allele=allele,
                description=row["description"], sequence=sequences[allele],
                genomic=row["genomic"] == "True", fields=int(row["fields"]),
                source=row["source"], source_version=row["source_version"],
                gene_model=gene_models.get(row["ipd_id"]),
                nearest_ipd_match=row["nearest_ipd_match"] or None,
                # .get(), not [...]: a bundle written before this column
                # existed must still load (graceful degradation, same
                # precedent as load_nearest_ipd_match's own missing-lookup
                # handling), not raise a KeyError on every old checked-in
                # bundle until it happens to be regenerated.
                gtf_template_allele=row.get("gtf_template_allele") or None,
            )

