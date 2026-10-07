"""Unit tests for AssemblyGTFSource on synthetic hla_seq-shaped fixtures:
strands, single-exon HLA-N, missing-leading-exon HLA-W, frame ->
codon_start, dedup, determinism."""
from __future__ import annotations

from pathlib import Path

import pysam
import pytest

from assembly_sources import AssemblyGTFSource, BundledAssemblySource
from build import write_fasta_gz, write_gene_models_jsonl_gz, write_meta_tsv_gz

_GTF_HEADER = "## format: gtf\n## date: 2025-01-01\n"


def _gtf_row(seqname, feature, start, end, strand, frame, attrs) -> str:
    attrs_str = "; ".join(f'{k} "{v}"' for k, v in attrs.items()) + ";"
    return f"{seqname}\tIPD-IMGT/HLA-TEST\t{feature}\t{start}\t{end}\t.\t{strand}\t{frame}\t{attrs_str}\n"


def _write_fasta(tmp_path: Path, contigs: dict[str, str]) -> Path:
    fasta_path = tmp_path / "all_HLA.fasta"
    with open(fasta_path, "w") as fh:
        for name, seq in contigs.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")
    pysam.faidx(str(fasta_path))
    return fasta_path


def _write_gtf(tmp_path: Path, haplotype_id: str, rows: list[str]) -> Path:
    path = tmp_path / f"{haplotype_id}_HLA.gtf.gz"
    import gzip
    with gzip.open(path, "wt") as fh:
        fh.write(_GTF_HEADER)
        fh.writelines(rows)
    return path


def _contig_seq(length: int, seed: str = "A") -> str:
    bases = "ACGT"
    return "".join(bases[(i + len(seed)) % 4] for i in range(length))


def test_forward_strand_multi_exon_extraction(tmp_path: Path) -> None:
    # contig: 5'UTR(1-50) exon1/CDS(51-150) intron(151-200) exon2/CDS(201-260) 3'UTR(261-300)
    seq = _contig_seq(300, "fwd")
    contig = "SAMPLE1#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig: seq})
    gene_start, gene_end = 1, 300  # gene row spans the full UTR-to-UTR extent, matching real hla_seq shape
    seqname = f"{contig}:1-1000"
    rows = [
        _gtf_row(seqname, "gene", gene_start, gene_end, "+", ".",
                 {"gene_id": "G1", "template_allele": "HLA-TEST*01:new", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "transcript", gene_start, gene_end, "+", ".",
                 {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "consensus": "HLA-TEST*01:new"}),
        _gtf_row(seqname, "UTR", 1, 50, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "exon", 51, 150, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "1"}),
        _gtf_row(seqname, "CDS", 51, 150, "+", "0", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "exon", 201, 260, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "2"}),
        _gtf_row(seqname, "CDS", 201, 260, "+", "2", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "UTR", 261, 300, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
    ]
    _write_gtf(tmp_path, "SAMPLE1_1", rows)

    src = AssemblyGTFSource(genes={"HLA-TEST"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs = list(src.records())
    assert len(recs) == 1
    r = recs[0]
    assert r.gene == "HLA-TEST"
    assert r.sequence == seq
    assert r.genomic is True
    assert r.fields == 1
    assert r.allele.startswith("HLA-TEST*asm")
    gm = r.gene_model
    assert gm is not None
    assert gm.utrs == ((1, 50), (261, 300))
    assert gm.cds_exons == ((51, 150), (201, 260))
    assert gm.exons == ((1, 51, 150), (2, 201, 260))
    assert gm.introns == ((1, 151, 200),)
    # frame "0" on the first CDS segment -> codon_start = 0 + 1 = 1
    assert gm.codon_start == 1
    assert "not authoritative" in r.description
    assert "HLA-TEST*01:new" not in r.allele


def test_reverse_strand_extraction_revcomps_and_flips_coordinates(tmp_path: Path) -> None:
    seq = _contig_seq(200, "rev")
    contig = "SAMPLE2#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig: seq})
    seqname = f"{contig}:1-1000"
    # minus-strand gene spans 51..150; exon_number 1 is the highest-coordinate
    # row (confirmed real convention), exon_number 2 the lowest.
    rows = [
        _gtf_row(seqname, "gene", 51, 150, "-", ".", {"gene_id": "G1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "transcript", 51, 150, "-", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "exon", 111, 150, "-", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "1"}),
        _gtf_row(seqname, "CDS", 111, 150, "-", "0", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "exon", 51, 90, "-", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "2"}),
        _gtf_row(seqname, "CDS", 51, 90, "-", "1", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
    ]
    _write_gtf(tmp_path, "SAMPLE2_1", rows)

    src = AssemblyGTFSource(genes={"HLA-TEST"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs = list(src.records())
    assert len(recs) == 1
    r = recs[0]

    import pysam as _pysam
    raw = _pysam.FastaFile(str(fasta_path)).fetch(contig, 50, 150)
    comp = str.maketrans("ACGT", "TGCA")
    expected = raw.translate(comp)[::-1]
    assert r.sequence == expected
    gm = r.gene_model
    assert gm is not None
    # exon_number 1 (genomic 111-150) is nearest the gene's 3' end on the +
    # strand, i.e. becomes the LOWEST shifted coordinate after revcomp flip
    # (gene_end=150): shift(111,150) -> (150-150+1, 150-111+1) = (1, 40)
    assert gm.exons[0] == (1, 1, 40)
    # exon_number 2 (genomic 51-90) -> shift(51,90) -> (150-90+1,150-51+1) = (61,100)
    assert gm.exons[1] == (2, 61, 100)
    # single intron between exon 1 (ends at 40) and exon 2 (starts at 61)
    assert gm.introns == ((1, 41, 60),)
    # codon_start comes from exon 1's CDS (frame 0) despite its highest
    # coordinate on the - strand: 0 + 1 = 1.
    assert gm.codon_start == 1


def test_single_exon_intronless_shape_like_real_hla_n(tmp_path: Path) -> None:
    seq = _contig_seq(150, "n")
    contig = "SAMPLE3#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig: seq})
    seqname = f"{contig}:1-1000"
    rows = [
        _gtf_row(seqname, "gene", 1, 150, "+", ".", {"gene_id": "G1", "gene_name": "HLA-N"}),
        _gtf_row(seqname, "transcript", 1, 150, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-N"}),
        _gtf_row(seqname, "UTR", 1, 40, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-N"}),
        _gtf_row(seqname, "exon", 1, 150, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-N", "exon_number": "1"}),
        _gtf_row(seqname, "CDS", 41, 120, "+", "0", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-N"}),
        _gtf_row(seqname, "UTR", 121, 150, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-N"}),
    ]
    _write_gtf(tmp_path, "SAMPLE3_1", rows)

    src = AssemblyGTFSource(genes={"HLA-N"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs = list(src.records())
    assert len(recs) == 1
    gm = recs[0].gene_model
    assert gm is not None
    assert gm.introns == ()  # real: single-exon pseudogenes have no introns, none fabricated
    assert gm.exons == ((1, 1, 150),)


def test_utr_only_exon_number_like_real_hla_p(tmp_path: Path) -> None:
    # Real quirk confirmed on real HLA-P records: a purely-UTR trailing
    # exon (no CDS content of its own) is numbered "utr", not an integer.
    seq = _contig_seq(160, "p")
    contig = "SAMPLE10#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig: seq})
    seqname = f"{contig}:1-1000"
    rows = [
        _gtf_row(seqname, "gene", 1, 160, "+", ".", {"gene_id": "G1", "gene_name": "HLA-P"}),
        _gtf_row(seqname, "transcript", 1, 160, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-P"}),
        _gtf_row(seqname, "exon", 1, 60, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-P", "exon_number": "1"}),
        _gtf_row(seqname, "CDS", 1, 60, "+", "0", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-P"}),
        _gtf_row(seqname, "UTR", 101, 160, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-P"}),
        _gtf_row(seqname, "exon", 101, 160, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-P", "exon_number": "utr"}),
    ]
    _write_gtf(tmp_path, "SAMPLE10_1", rows)

    src = AssemblyGTFSource(genes={"HLA-P"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs = list(src.records())
    assert len(recs) == 1
    gm = recs[0].gene_model
    assert gm is not None
    # both exon rows are kept, the unnumbered one as (None, ...) -- no
    # fabricated ordinal, matching GeneModel.exons' own (int | None) type.
    assert gm.exons == ((1, 1, 60), (None, 101, 160))
    # the None-numbered exon doesn't pair into intron derivation -- the
    # real gap between exon 1 and the UTR-only exon is simply not captured
    # as an intron, rather than guessed.
    assert gm.introns == ()


def test_partial_annotation_missing_leading_exons_like_real_hla_w(tmp_path: Path) -> None:
    seq = _contig_seq(200, "w")
    contig = "SAMPLE4#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig: seq})
    seqname = f"{contig}:1-1000"
    # Only exon_number 3 and 4 are present (1-2 missing, real degenerate
    # pseudogene shape) -- no UTR rows at all, matching real HLA-W.
    rows = [
        _gtf_row(seqname, "gene", 1, 200, "+", ".", {"gene_id": "G1", "gene_name": "HLA-W"}),
        _gtf_row(seqname, "transcript", 1, 200, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-W"}),
        _gtf_row(seqname, "exon", 1, 60, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-W", "exon_number": "3"}),
        _gtf_row(seqname, "CDS", 1, 60, "+", "0", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-W"}),
        _gtf_row(seqname, "exon", 101, 200, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-W", "exon_number": "4"}),
        _gtf_row(seqname, "CDS", 101, 200, "+", "2", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-W"}),
    ]
    _write_gtf(tmp_path, "SAMPLE4_1", rows)

    src = AssemblyGTFSource(genes={"HLA-W"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs = list(src.records())
    assert len(recs) == 1
    gm = recs[0].gene_model
    assert gm is not None
    assert gm.utrs == ()  # real: no UTR rows at all on a partial annotation
    # intron only between exon 3 and exon 4 (consecutive present numbers) --
    # nothing fabricated for the missing exons 1-2 before exon 3.
    assert gm.introns == ((3, 61, 100),)
    assert gm.exons == ((3, 1, 60), (4, 101, 200))


def test_dedup_identical_sequences_across_haplotypes(tmp_path: Path) -> None:
    seq = _contig_seq(100, "dup")
    contig_a, contig_b = "SAMPLE5#1#ctg1", "SAMPLE6#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig_a: seq, contig_b: seq})

    def rows_for(contig):
        seqname = f"{contig}:1-1000"
        return [
            _gtf_row(seqname, "gene", 1, 100, "+", ".", {"gene_id": "G1", "gene_name": "HLA-TEST"}),
            _gtf_row(seqname, "transcript", 1, 100, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "consensus": "HLA-TEST*01:new"}),
            _gtf_row(seqname, "exon", 1, 100, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "1"}),
        ]

    _write_gtf(tmp_path, "SAMPLE5_1", rows_for(contig_a))
    _write_gtf(tmp_path, "SAMPLE6_1", rows_for(contig_b))

    src = AssemblyGTFSource(genes={"HLA-TEST"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs = list(src.records())
    assert len(recs) == 1  # byte-identical sequence collapses to one record
    assert "2 haplotype" in recs[0].description


def test_one_bp_difference_yields_two_distinct_records(tmp_path: Path) -> None:
    seq_a = _contig_seq(100, "diff")
    seq_b = seq_a[:50] + ("A" if seq_a[50] != "A" else "C") + seq_a[51:]
    contig_a, contig_b = "SAMPLE7#1#ctg1", "SAMPLE8#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig_a: seq_a, contig_b: seq_b})

    def rows_for(contig):
        seqname = f"{contig}:1-1000"
        return [
            _gtf_row(seqname, "gene", 1, 100, "+", ".", {"gene_id": "G1", "gene_name": "HLA-TEST"}),
            _gtf_row(seqname, "transcript", 1, 100, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
            _gtf_row(seqname, "exon", 1, 100, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "1"}),
        ]

    _write_gtf(tmp_path, "SAMPLE7_1", rows_for(contig_a))
    _write_gtf(tmp_path, "SAMPLE8_1", rows_for(contig_b))

    src = AssemblyGTFSource(genes={"HLA-TEST"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs = list(src.records())
    assert len(recs) == 2
    assert recs[0].allele != recs[1].allele
    assert len({r.sequence for r in recs}) == 2


def test_contig_name_suffix_is_stripped(tmp_path: Path) -> None:
    seq = _contig_seq(80, "suffix")
    contig = "SAMPLE9#1#ctg1"  # FASTA header has NO :start-end suffix
    fasta_path = _write_fasta(tmp_path, {contig: seq})
    seqname = f"{contig}:12345-67890"  # GTF seqname DOES carry one
    rows = [
        _gtf_row(seqname, "gene", 1, 80, "+", ".", {"gene_id": "G1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "transcript", 1, 80, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "exon", 1, 80, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "1"}),
    ]
    _write_gtf(tmp_path, "SAMPLE9_1", rows)

    src = AssemblyGTFSource(genes={"HLA-TEST"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs = list(src.records())
    assert len(recs) == 1
    assert recs[0].sequence == seq


def _write_bundle(tmp_path: Path, records) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    write_fasta_gz(bundle_dir / "bundle.fa.gz", records)
    write_meta_tsv_gz(bundle_dir / "bundle.meta.tsv.gz", records)
    write_gene_models_jsonl_gz(bundle_dir / "bundle.models.jsonl.gz", records)
    return bundle_dir


def test_bundled_assembly_source_round_trips_a_written_bundle(tmp_path: Path) -> None:
    # Write a real AssemblyGTFSource record as a bundle and read it back.
    seq = _contig_seq(300, "fwd")
    contig = "SAMPLE1#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig: seq})
    seqname = f"{contig}:1-1000"
    rows = [
        _gtf_row(seqname, "gene", 1, 300, "+", ".",
                 {"gene_id": "G1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "transcript", 1, 300, "+", ".",
                 {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "consensus": "HLA-TEST*01:new"}),
        _gtf_row(seqname, "UTR", 1, 50, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "exon", 51, 150, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "1"}),
        _gtf_row(seqname, "CDS", 51, 150, "+", "0", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "exon", 201, 260, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "2"}),
        _gtf_row(seqname, "CDS", 201, 260, "+", "2", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
        _gtf_row(seqname, "UTR", 261, 300, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
    ]
    _write_gtf(tmp_path, "SAMPLE1_1", rows)
    original = list(AssemblyGTFSource(genes={"HLA-TEST"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path).records())
    assert len(original) == 1

    bundle_dir = _write_bundle(tmp_path, original)
    reloaded = list(BundledAssemblySource(genes={"HLA-TEST"}, bundle_dir=bundle_dir).records())
    assert len(reloaded) == 1

    o, r = original[0], reloaded[0]
    assert (r.gene, r.allele, r.ipd_id, r.sequence, r.genomic, r.fields) == \
           (o.gene, o.allele, o.ipd_id, o.sequence, o.genomic, o.fields)
    assert r.description == o.description
    assert r.nearest_ipd_match == o.nearest_ipd_match
    assert r.gene_model == o.gene_model
    # source/source_version are the bundle's real origin, not a "bundle" tag
    assert r.source == "hla_seq-assemblies-gtf"


def test_bundled_assembly_source_filters_by_requested_genes(tmp_path: Path) -> None:
    seq_a = _contig_seq(100, "gA")
    seq_b = _contig_seq(100, "gB")
    fasta_path = _write_fasta(tmp_path, {"CTGA#1#c1": seq_a, "CTGB#1#c1": seq_b})

    def rows_for(contig, gene):
        seqname = f"{contig}:1-1000"
        return [
            _gtf_row(seqname, "gene", 1, 100, "+", ".", {"gene_id": "G1", "gene_name": gene}),
            _gtf_row(seqname, "transcript", 1, 100, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": gene}),
            _gtf_row(seqname, "exon", 1, 100, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": gene, "exon_number": "1"}),
        ]

    _write_gtf(tmp_path, "CTGA_1", rows_for("CTGA#1#c1", "HLA-GENEA"))
    _write_gtf(tmp_path, "CTGB_1", rows_for("CTGB#1#c1", "HLA-GENEB"))
    records = list(AssemblyGTFSource(genes={"HLA-GENEA", "HLA-GENEB"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path).records())
    assert {r.gene for r in records} == {"HLA-GENEA", "HLA-GENEB"}

    bundle_dir = _write_bundle(tmp_path, records)
    only_a = list(BundledAssemblySource(genes={"HLA-GENEA"}, bundle_dir=bundle_dir).records())
    assert {r.gene for r in only_a} == {"HLA-GENEA"}


def test_bundled_assembly_source_missing_bundle_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="assembly bundle missing"):
        list(BundledAssemblySource(genes={"HLA-TEST"}, bundle_dir=tmp_path / "does_not_exist").records())


def test_determinism_across_reruns(tmp_path: Path) -> None:
    seq_a = _contig_seq(100, "det1")
    seq_b = _contig_seq(100, "det2")
    contig_a, contig_b = "SAMPLEA#1#ctg1", "SAMPLEB#1#ctg1"
    fasta_path = _write_fasta(tmp_path, {contig_a: seq_a, contig_b: seq_b})

    def rows_for(contig):
        seqname = f"{contig}:1-1000"
        return [
            _gtf_row(seqname, "gene", 1, 100, "+", ".", {"gene_id": "G1", "gene_name": "HLA-TEST"}),
            _gtf_row(seqname, "transcript", 1, 100, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST"}),
            _gtf_row(seqname, "exon", 1, 100, "+", ".", {"gene_id": "G1", "transcript_id": "T1", "gene_name": "HLA-TEST", "exon_number": "1"}),
        ]

    _write_gtf(tmp_path, "SAMPLEA_1", rows_for(contig_a))
    _write_gtf(tmp_path, "SAMPLEB_1", rows_for(contig_b))

    src = AssemblyGTFSource(genes={"HLA-TEST"}, hla_seq_dir=tmp_path, all_fasta_path=fasta_path)
    recs1 = [(r.allele, r.sequence, r.description) for r in src.records()]
    recs2 = [(r.allele, r.sequence, r.description) for r in src.records()]
    assert recs1 == recs2
