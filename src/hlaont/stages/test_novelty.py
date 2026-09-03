"""Unit tests for Stage 5 novelty classification (stages/novelty.py)."""
from __future__ import annotations

import gzip
import json

from assign import AlignmentSpan
from hla_typing import TypingCall
from novelty import classify

GENE = "GENE1"
CALLED_ALLELE = "GENE1*01:01:01:01"

# Synthetic 150bp gene: UTR1(1-30) exon1/cds(31-60) intron1(61-90)
# exon2/cds(91-120) UTR2(121-150). CDS = "CTT"*20 (60bp, 20 Leu codons,
# in-frame from codon_start=1) -- chosen because every CTx codon (T/C/A/G
# 3rd base) is still Leu, giving an easy synonymous-substitution site.
_UTR1 = "A" * 30
_EXON1 = "CTT" * 10          # pos 31-60
_INTRON1 = "GT" + "C" * 26 + "AG"  # pos 61-90, canonical-looking donor/acceptor
_EXON2 = "CTT" * 10          # pos 91-120
_UTR2 = "T" * 30
REF_SEQ = _UTR1 + _EXON1 + _INTRON1 + _EXON2 + _UTR2
assert len(REF_SEQ) == 150

GENE_MODEL_OBJ = {
    "gene": GENE, "ipd_id": "TEST00001", "allele": CALLED_ALLELE, "length": 150,
    "codon_start": 1,
    "cds_exons": [[31, 60], [91, 120]],
    "utrs": [[1, 30], [121, 150]],
    "exons": [[1, 31, 60], [2, 91, 120]],
    "introns": [[1, 61, 90]],
}


def _write_resources(tmp_path, cds_catalogue_rows=(), nearest_ipd_match=""):
    fasta = tmp_path / "pool.fa.gz"
    with gzip.open(fasta, "wt") as fh:
        fh.write(f">{CALLED_ALLELE}\n{REF_SEQ}\n")

    models = tmp_path / "models.jsonl.gz"
    with gzip.open(models, "wt") as fh:
        fh.write(json.dumps(GENE_MODEL_OBJ) + "\n")

    catalogue = tmp_path / "cds_catalogue.tsv.gz"
    with gzip.open(catalogue, "wt") as fh:
        fh.write("gene\tipd_id\tallele\tgenomic\tfields\tcds_length\tcds_sequence\n")
        for row in cds_catalogue_rows:
            fh.write("\t".join(row) + "\n")

    # nearest_ipd_match empty by default -- ordinary IPD-sourced allele
    # shape (resources/templates/build.py's own convention), so
    # display_allele falls back to called_allele and every pre-existing
    # test's behaviour is unchanged.
    meta = tmp_path / "meta.tsv.gz"
    with gzip.open(meta, "wt") as fh:
        fh.write("gene\tipd_id\tallele\tfields\tlength\tgenomic\tsource\tsource_version\tdescription\tnearest_ipd_match\n")
        fh.write(f"{GENE}\tTEST00001\t{CALLED_ALLELE}\t4\t{len(REF_SEQ)}\tTrue\ttest\tv1\ttest\t{nearest_ipd_match}\n")

    return fasta, models, catalogue, meta


def _call(span: AlignmentSpan) -> TypingCall:
    return TypingCall(
        gene=GENE, consensus_template_allele=CALLED_ALLELE, called_allele=CALLED_ALLELE,
        n_fields=4, field_gq=(60.0, 60.0, 60.0, 60.0), posterior={CALLED_ALLELE: 1.0},
        ambiguity_set=(CALLED_ALLELE,), called_allele_span=span,
    )


def _no_indel_span() -> AlignmentSpan:
    return AlignmentSpan(ref_start=0, ref_end=150, read_start=0, read_end=150,
                          strand="+", cigar="150M")


def test_exact_match(tmp_path):
    fasta, models, catalogue, meta = _write_resources(tmp_path)
    result = classify(_call(_no_indel_span()), REF_SEQ,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "exact"
    assert result.proposed_name == CALLED_ALLELE
    assert result.variants == ()


def test_novel_utr(tmp_path):
    fasta, models, catalogue, meta = _write_resources(tmp_path)
    consensus = REF_SEQ[:9] + ("C" if REF_SEQ[9] != "C" else "G") + REF_SEQ[10:]
    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:utr"
    assert result.proposed_name == "GENE1*01:01:01:new"
    assert len(result.variants) == 1 and result.variants[0].location == "utr"


def test_novel_intronic_non_splice(tmp_path):
    fasta, models, catalogue, meta = _write_resources(tmp_path)
    pos0 = 74  # 0-based index for 1-based pos 75, well inside the intron
    consensus = REF_SEQ[:pos0] + ("T" if REF_SEQ[pos0] != "T" else "A") + REF_SEQ[pos0 + 1:]
    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:intronic"
    assert not result.variants[0].splice_site


def test_intronic_splice_site_escalates_to_null(tmp_path):
    fasta, models, catalogue, meta = _write_resources(tmp_path)
    pos0 = 60  # 0-based index for 1-based pos 61 -- first base of the intron
    consensus = REF_SEQ[:pos0] + "C" + REF_SEQ[pos0 + 1:]
    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:null"
    assert result.candidate_null
    assert result.coding_effect == "splice_disrupting"
    assert result.variants[0].splice_site


def test_novel_synonymous(tmp_path):
    fasta, models, catalogue, meta = _write_resources(tmp_path)
    pos0 = 32  # 0-based index for 1-based pos 33 -- 3rd base of first CDS codon "CTT"
    assert REF_SEQ[pos0] == "T"
    consensus = REF_SEQ[:pos0] + "C" + REF_SEQ[pos0 + 1:]  # CTT -> CTC, still Leu
    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:synonymous"
    assert result.coding_effect == "synonymous"
    assert result.proposed_name == "GENE1*01:01:new"


def test_novel_nonsynonymous(tmp_path):
    fasta, models, catalogue, meta = _write_resources(tmp_path)
    pos0 = 30  # 0-based index for 1-based pos 31 -- 1st base of first CDS codon
    assert REF_SEQ[pos0] == "C"
    consensus = REF_SEQ[:pos0] + "A" + REF_SEQ[pos0 + 1:]  # CTT -> ATT, Leu -> Ile
    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:nonsynonymous"
    assert result.coding_effect == "nonsynonymous"
    assert result.proposed_name == "GENE1*01:new"


def test_novel_null_premature_stop_no_frameshift(tmp_path):
    # Mutate the first CDS codon "CTT" (pos 31-33) to the stop codon "TAA"
    # -- no length change at all, so this must be caught by the stop-codon
    # comparison, not the frameshift (length-mod-3) check.
    fasta, models, catalogue, meta = _write_resources(tmp_path)
    consensus = REF_SEQ[:30] + "TAA" + REF_SEQ[33:]
    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:null"
    assert result.candidate_null
    assert result.coding_effect == "premature_stop"


def test_novel_null_frameshift(tmp_path):
    fasta, models, catalogue, meta = _write_resources(tmp_path)
    # One extra base inserted mid-exon1 (after ref pos 45, 0-based index 45):
    # not a multiple of 3, shifts every downstream codon.
    consensus = REF_SEQ[:45] + "A" + REF_SEQ[45:]
    span = AlignmentSpan(ref_start=0, ref_end=150, read_start=0, read_end=151,
                          strand="+", cigar="45M1I105M")
    result = classify(_call(span), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:null"
    assert result.candidate_null
    assert result.coding_effect == "frameshift"


def test_extension_overrides_novel_when_cds_matches_known_nongenomic_allele(tmp_path):
    pos0 = 30
    consensus = REF_SEQ[:pos0] + "A" + REF_SEQ[pos0 + 1:]  # same edit as nonsynonymous test
    alt_cds = "ATT" + "CTT" * 19  # the resulting edited CDS, exon1+exon2 concatenated
    known_allele = "GENE1*01:99"
    rows = [(GENE, "TESTKNOWN1", known_allele, "False", "4", str(len(alt_cds)), alt_cds)]
    fasta, models, catalogue, meta = _write_resources(tmp_path, cds_catalogue_rows=rows)

    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "extension"
    assert result.proposed_name == known_allele
    assert result.extension_allele == known_allele


def test_genomic_only_cds_catalogue_match_does_not_trigger_extension(tmp_path):
    pos0 = 30
    consensus = REF_SEQ[:pos0] + "A" + REF_SEQ[pos0 + 1:]
    alt_cds = "ATT" + "CTT" * 19
    other_genomic_allele = "GENE1*01:98"
    rows = [(GENE, "TESTKNOWN2", other_genomic_allele, "True", "4", str(len(alt_cds)), alt_cds)]
    fasta, models, catalogue, meta = _write_resources(tmp_path, cds_catalogue_rows=rows)

    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:nonsynonymous"
    assert result.extension_allele is None


def test_nearest_ipd_match_used_for_display_name_when_exact(tmp_path):
    """Assembly-extraction genes (module docstring): a real catalogued
    Immuannot guess (no `:new` tag) is shown as the exact-match name in
    place of the internal content-hash accession, taxonomy unaffected."""
    fasta, models, catalogue, meta = _write_resources(
        tmp_path, nearest_ipd_match="GENE1*01:01:01:05",
    )
    result = classify(_call(_no_indel_span()), REF_SEQ,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "exact"
    assert result.proposed_name == "GENE1*01:01:01:05"


def test_nearest_ipd_match_already_new_becomes_novel_extraction(tmp_path):
    """Immuannot's own guess is itself `:new`-tagged (no real IPD catalog
    hit at all) -- a zero-variant diff against our own extracted reference
    must not be reported as plain `exact` (would claim a catalogued match
    that doesn't exist), so it gets the dedicated `novel:extraction` tag."""
    fasta, models, catalogue, meta = _write_resources(
        tmp_path, nearest_ipd_match="GENE1*01:01:01:new",
    )
    result = classify(_call(_no_indel_span()), REF_SEQ,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:extraction"
    assert result.proposed_name == "GENE1*01:01:01:new"
    assert result.variants == ()


def test_nearest_ipd_match_bare_new_does_not_double_suffix(tmp_path):
    """A record with no matching prefix at all (Immuannot's own bare
    `GENE*new`, real HLA-P shape) must not come out as `GENE1*new:new`
    once a real coding difference routes it through `_novel_name()`."""
    fasta, models, catalogue, meta = _write_resources(
        tmp_path, nearest_ipd_match="GENE1*new",
    )
    pos0 = 30  # 1st base of first CDS codon -- same edit as the nonsynonymous test
    consensus = REF_SEQ[:pos0] + "A" + REF_SEQ[pos0 + 1:]
    result = classify(_call(_no_indel_span()), consensus,
                       pool_fasta=fasta, pool_meta=meta, models_path=models, cds_catalogue_path=catalogue)
    assert result.taxonomy == "novel:nonsynonymous"
    assert result.proposed_name == "GENE1*new"
