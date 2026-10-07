"""Assembly-derived template source for genes IPD lacks usable genomic
templates for (HLA-N/P/T/U/W/R; HLA-Y has a single ancestry-skewed one).

Extracts gene spans and exon/CDS/UTR structure from hla_seq/'s phased
assemblies using their Immuannot GTFs, as TemplateRecord/GeneModel like the
IPD parser. Accessions are content hashes of the sequence (stable across
reruns); Immuannot's nearest-match names are kept only for display, never
as the accession (invariant 7).

GTF quirks (checked on real files):
- No intron rows: introns are gaps between consecutive present exons, never
  invented before a missing leading exon.
- Minus-strand exon_number already follows transcript order; only the
  coordinates are flipped.
- codon_start = frame + 1.
- Missing UTRs or start/stop codons are real pseudogene degeneracy.
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

# hla_seq/ has no release tag; pinned here. records() prefers the v2
# re-annotation (*_HLA_v2.gtf.gz, which adds HLA-R) and falls back to
# *_HLA.gtf.gz.
HLA_SEQ_SNAPSHOT_VERSION = "hla_seq-2026-09-v2"

_V2_GTF_SUFFIX = "_HLA_v2.gtf.gz"
_LEGACY_GTF_SUFFIX = "_HLA.gtf.gz"

# Checked-in snapshot of AssemblyGTFSource output, so no hla_seq/ is needed.
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
    """Strip the GTF seqname's `:start-end` suffix to get the FASTA contig."""
    return _SEQNAME_SUFFIX_RE.sub("", seqname)


class AssemblyGTFSource:
    """Extracts genes listed via genes.yaml `assembly_extraction:
    gtf_coordinate` from hla_seq/ GTF annotations."""

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
                # Pure-UTR trailing exons are numbered "utr" (HLA-P): keep
                # them unnumbered, outside intron derivation.
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
                # Identical sequences should share one Immuannot guess; if
                # not, take sorted()[0] and keep all in `description`.
        consensus_names = sorted(info["consensus_names"])
        consensus_note = ",".join(consensus_names) or "n/a"
        nearest_ipd_match = consensus_names[0] if consensus_names else None
                # Same, from the gene-line `template_allele` (never `:new`).
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
    """Loads the checked-in bundle written by build_assembly_bundle.py,
    yielding the same TemplateRecords AssemblyGTFSource would. Records keep
    their original source/version; this class's name/version only identify
    the loader."""

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
                # .get(): older bundles lack this column.
                gtf_template_allele=row.get("gtf_template_allele") or None,
            )

