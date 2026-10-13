"""Tests for stages/output.py's writers, driven off a real run_sample()
result on the same synthetic two-gene fixture test_pipeline.py uses."""
from __future__ import annotations

import csv
import dataclasses
import json
import random

import pysam
from hlaont.stages.c4_form import C4AbCall, C4FormLengthCall, C4PerGeneCN
from hlaont.stages.cn import GeneCN
from hlaont.stages.output import write_calls_tsv, write_consensus_fasta, write_qc_json, write_variants_vcf
from hlaont.stages.pipeline import GenePhaseRouting, HaplotypeCallResult, run_sample, run_sample_phased
from tests.stages.test_pipeline import (
    C4_EXTRA_CALLS_TSV_GENES,
    TINY_C4_FASTA,
    TINY_C4_META,
    _TEST_CONTROL_DEPTH,
    _write_anchors,
    _write_fastq,
    _write_genes_yaml,
    _write_mini_bam_with_phase_tags,
    _write_pool,
)


def _run(tmp_path):
    rng = random.Random(321)
    allele_a = "".join(rng.choice("ACGT") for _ in range(700))
    bases = list(allele_a)
    for p in random.Random(11).sample(range(len(bases)), 5):
        alts = [b for b in "ACGT" if b != bases[p]]
        bases[p] = random.Random(p).choice(alts)
    allele_b = "".join(bases)
    calib_a = "".join(random.Random(99).choice("ACGT") for _ in range(700))
    calib_b = calib_a[:350] + ("A" if calib_a[350] != "A" else "C") + calib_a[351:]

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)
    reads = (
        [(f"a{i}", allele_a) for i in range(10)]
        + [(f"b{i}", allele_b) for i in range(10)]
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)

    return run_sample(
        "sampleX", fastq_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    ), pool_fasta


def test_write_calls_tsv(tmp_path):
    result, _ = _run(tmp_path)
    out = tmp_path / "calls.tsv"
    write_calls_tsv(result, out)

    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    # Only one extra row, "C4" (chr17-control disabled in this fixture).
    assert "haplotype_index" not in rows[0]
    haplotype_rows = [r for r in rows if r["gene"] not in C4_EXTRA_CALLS_TSV_GENES]
    c4_rows_by_gene = {r["gene"]: r for r in rows if r["gene"] in C4_EXTRA_CALLS_TSV_GENES}
    assert set(c4_rows_by_gene) == {"C4"}
    assert len(haplotype_rows) == len(result.haplotype_calls) == 4
    genes = {r["gene"] for r in haplotype_rows}
    assert genes == {"GENE1", "CALIB1"}
    for r in haplotype_rows:
        assert r["cn"] == "2"
        assert r["taxonomy"] == "exact"
        assert len(r["field_gq"].split(",")) == int(r["n_fields"])
        assert r["c4_form_note"] == ""

    assert result.c4_ab_call is not None
    assert result.c4_form_call is not None
    c4_row = c4_rows_by_gene["C4"]
    assert c4_row["cn"] == f"{result.gene_cn['C4'].cn_estimate:.2f}"
    assert c4_row["n_reads_assigned"] == str(result.gene_cn["C4"].depth)


def test_write_consensus_fasta(tmp_path):
    result, _ = _run(tmp_path)
    fasta_path = tmp_path / "consensus.fa"
    qv_path = tmp_path / "consensus.qv.tsv"
    write_consensus_fasta(result, fasta_path, qv_path)

    with pysam.FastxFile(str(fasta_path)) as fh:
        records = list(fh)
    assert len(records) == 4
    names = {r.name for r in records}
    assert names == {"GENE1_hap0", "GENE1_hap1", "CALIB1_hap0", "CALIB1_hap1"}
    for r in records:
        assert len(r.sequence) == 700

    with open(qv_path) as fh:
        qv_rows = list(csv.DictReader(fh, delimiter="\t"))
    # 4 records * 700 bases each, no deletions expected for exact matches
    assert len(qv_rows) == 4 * 700
    assert {r["record_id"] for r in qv_rows} == names


def test_write_variants_vcf_no_variants_for_exact_matches(tmp_path):
    result, pool_fasta = _run(tmp_path)
    vcf_path = tmp_path / "variants.vcf"
    write_variants_vcf(result, vcf_path, pool_fasta=pool_fasta)

    lines = vcf_path.read_text().splitlines()
    header_lines = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if not line.startswith("#")]
    assert header_lines[-1] == "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
    # Every haplotype in this fixture is an exact match to its called
    # allele (no injected mutations), so there should be zero variant rows.
    assert data_lines == []


def test_write_qc_json(tmp_path):
    result, _ = _run(tmp_path)
    qc_path = tmp_path / "qc.json"
    write_qc_json(result, qc_path, dorado_version="7.1.0", reference_build_md5=None)

    obj = json.loads(qc_path.read_text())
    assert obj["sample_id"] == "sampleX"
    assert obj["sieve_stats"]["n_kept"] == 40
    assert obj["dorado_version"] == "7.1.0"
    assert obj["reference_build_md5"] is None
    # "C4" is always in gene_cn (zero depth here).
    assert set(obj["gene_cn"]) == {"GENE1", "CALIB1", "C4"}
    assert obj["gene_cn"]["GENE1"]["cn_estimate"] == 2.0
    assert obj["gene_cn"]["C4"]["depth"] == 0
    assert obj["failures"] == []
    # 2026-08-27, user-requested: CN_A/CN_B always present as keys in
    # a_vs_b (None here since this fixture never recruits a real C4 read,
    # same reasoning as test_write_calls_tsv above).
    a_vs_b = obj["c4_form"]["a_vs_b"]
    assert set(a_vs_b) >= {"cn_a", "cn_a_ci_low", "cn_a_ci_high", "cn_b", "cn_b_ci_low", "cn_b_ci_high"}
    assert a_vs_b["cn_a"] is None and a_vs_b["cn_b"] is None


def test_write_qc_json_phasing_is_null_for_unphased_run(tmp_path):
    """DESIGN.md S10.7: an ordinary run_sample() result must not carry a
    "phasing" block implying real phase that was never asserted."""
    result, _ = _run(tmp_path)
    qc_path = tmp_path / "qc.json"
    write_qc_json(result, qc_path)

    obj = json.loads(qc_path.read_text())
    assert obj["phasing"] is None


def _run_phased(tmp_path):
    rng = random.Random(321)
    allele_a = "".join(rng.choice("ACGT") for _ in range(700))
    bases = list(allele_a)
    for p in random.Random(11).sample(range(len(bases)), 5):
        alts = [b for b in "ACGT" if b != bases[p]]
        bases[p] = random.Random(p).choice(alts)
    allele_b = "".join(bases)
    calib_a = "".join(random.Random(99).choice("ACGT") for _ in range(700))
    calib_b = calib_a[:350] + ("A" if calib_a[350] != "A" else "C") + calib_a[351:]

    pool_fasta, pool_meta, models_path, cds_catalogue = _write_pool(tmp_path, {
        "GENE1*01:01:01:01": ("GENE1", allele_a),
        "GENE1*01:02:01:01": ("GENE1", allele_b),
        "CALIB1*01:01:01:01": ("CALIB1", calib_a),
        "CALIB1*01:02:01:01": ("CALIB1", calib_b),
    })
    anchors_path, anchors_meta_path = _write_anchors(tmp_path, {
        "GENE1-anchor": ("GENE1", allele_a),
        "CALIB1-anchor": ("CALIB1", calib_a),
    })
    genes_yaml = _write_genes_yaml(tmp_path)
    reads = (
        [(f"a{i}", allele_a) for i in range(10)]
        + [(f"b{i}", allele_b) for i in range(10)]
        + [(f"ca{i}", calib_a) for i in range(10)]
        + [(f"cb{i}", calib_b) for i in range(10)]
    )
    fastq_path = _write_fastq(tmp_path, reads)
    phase_tags = {f"a{i}": (1, 100) for i in range(10)}
    phase_tags.update({f"b{i}": (2, 100) for i in range(10)})
    # CALIB1 stays untagged -- must fall back to the unphased path.
    mini_bam_path = _write_mini_bam_with_phase_tags(tmp_path, phase_tags)

    result = run_sample_phased(
        "sampleX", fastq_path, mini_bam_path,
        pool_fasta=pool_fasta, pool_meta=pool_meta, models_path=models_path,
        cds_catalogue_path=cds_catalogue, genes_yaml=genes_yaml,
        c4_pool_fasta=TINY_C4_FASTA, c4_pool_meta=TINY_C4_META, control_depth=_TEST_CONTROL_DEPTH, compute_c4_chr17_control=False,
        anchors_path=anchors_path, anchors_meta_path=anchors_meta_path,
        n_threads=2,
    )
    return result, pool_fasta


def test_write_calls_tsv_haplotype_column_phased_mode(tmp_path):
    result, _ = _run_phased(tmp_path)
    out = tmp_path / "calls_phased.tsv"
    write_calls_tsv(result, out)

    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert "haplotype" in rows[0]

    gene1_rows = {r["haplotype"]: r for r in rows if r["gene"] == "GENE1"}
    assert set(gene1_rows) == {"1", "2"}

    calib1_rows = [r for r in rows if r["gene"] == "CALIB1"]
    assert len(calib1_rows) == 2
    assert all(r["haplotype"] == "unphased" for r in calib1_rows)

    # Sample-wide C4 rows are labelled "unphased"; no C4 reads here, so no
    # per-haplotype rows.
    c4_rows = [r for r in rows if r["gene"] in C4_EXTRA_CALLS_TSV_GENES]
    assert all(r["haplotype"] == "unphased" for r in c4_rows)


def test_write_qc_json_phasing_block(tmp_path):
    result, _ = _run_phased(tmp_path)
    qc_path = tmp_path / "qc_phased.json"
    write_qc_json(result, qc_path)

    obj = json.loads(qc_path.read_text())
    phasing = obj["phasing"]
    assert phasing is not None
    assert phasing["GENE1"]["routing"] == "phased"
    assert phasing["GENE1"]["n_reads_hp1"] == 10
    assert phasing["GENE1"]["n_reads_hp2"] == 10
    assert sorted(phasing["GENE1"]["haplotypes_called"]) == [1, 2]
    assert phasing["CALIB1"]["routing"] == "unphased_fallback"
    assert phasing["CALIB1"]["haplotypes_called"] == []


def _low_qv_call(gene: str = "DRB4LIKE", haplotype_index: int = 1) -> HaplotypeCallResult:
    """Synthetic noise call (HLA-DRB4*01:new shape: 4 reads, mean_qv 5.0)
    for the min_mean_qv filter."""
    return HaplotypeCallResult(
        gene=gene, haplotype_index=haplotype_index, cn=1,
        called_allele=f"{gene}*01:new", proposed_name=f"{gene}*01:new",
        taxonomy="novel:null", extension_allele=None, candidate_null=True,
        field_gq=(60.0, 60.0, 0.0, -0.0), ambiguity_set=(),
        n_reads_assigned=4, consensus_length=15346,
        mean_qv=5.0, min_qv=0, n_variants=37,
    )


def _extension_partial_call(gene: str = "HLA-C", haplotype_index: int = 1) -> HaplotypeCallResult:
    """HLA-C*07:457 shape: template HLA-C*07:02:01:01, CDS matching the
    partial allele HLA-C*07:457."""
    return HaplotypeCallResult(
        gene=gene, haplotype_index=haplotype_index, cn=2,
        called_allele=f"{gene}*07:02:01:01", proposed_name=f"{gene}*07:457:new",
        taxonomy="extension:partial", extension_allele=f"{gene}*07:457",
        candidate_null=False,
        field_gq=(60.0, 12.0, 10.8, 2.5), ambiguity_set=(),
        n_reads_assigned=22, consensus_length=4318,
        mean_qv=60.0, min_qv=60, n_variants=1,
    )


def test_write_calls_tsv_extension_partial_shows_extension_allele_as_template(tmp_path):
    """extension:partial shows `extension_allele` as template_allele;
    allele/taxonomy unchanged."""
    result, _ = _run(tmp_path)
    call = _extension_partial_call()
    result = dataclasses.replace(result, haplotype_calls=result.haplotype_calls + (call,))

    out = tmp_path / "calls_extension_partial.tsv"
    write_calls_tsv(result, out)
    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    row = next(r for r in rows if r["gene"] == "HLA-C")
    assert row["template_allele"] == "HLA-C*07:457"
    assert row["allele"] == "HLA-C*07:457:new"
    assert row["taxonomy"] == "extension:partial"


def test_write_calls_tsv_excludes_low_mean_qv_row(tmp_path):
    """User-requested, 2026-08-27: a row with mean_qv below the default
    30.0 floor must not appear in calls.tsv at all -- real HG00097 finding
    (HLA-DRB4) reproduced here as a synthetic injected call."""
    result, _ = _run(tmp_path)
    noisy_call = _low_qv_call()
    result = dataclasses.replace(result, haplotype_calls=result.haplotype_calls + (noisy_call,))

    out = tmp_path / "calls_with_noise.tsv"
    write_calls_tsv(result, out)
    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert not any(r["gene"] == "DRB4LIKE" for r in rows)
    # Everything else (mean_qv 60.0 in this fixture) is untouched.
    haplotype_rows = [r for r in rows if r["gene"] not in C4_EXTRA_CALLS_TSV_GENES]
    assert len(haplotype_rows) == len(result.haplotype_calls) - 1 == 4


def test_write_calls_tsv_keeps_row_at_or_above_custom_threshold(tmp_path):
    result, _ = _run(tmp_path)
    noisy_call = _low_qv_call()
    result = dataclasses.replace(result, haplotype_calls=result.haplotype_calls + (noisy_call,))

    out = tmp_path / "calls_custom_threshold.tsv"
    write_calls_tsv(result, out, min_mean_qv=0.0)
    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert any(r["gene"] == "DRB4LIKE" for r in rows)


def test_write_qc_json_reports_low_mean_qv_filtered_calls(tmp_path):
    """Never silently dropped: the row excluded from calls.tsv must still
    be visible somewhere (development notes: "silent failure... is worse than a
    crash")."""
    result, _ = _run(tmp_path)
    noisy_call = _low_qv_call()
    result = dataclasses.replace(result, haplotype_calls=result.haplotype_calls + (noisy_call,))

    qc_path = tmp_path / "qc_with_noise.json"
    write_qc_json(result, qc_path)
    obj = json.loads(qc_path.read_text())

    filtered = obj["calls_filtered_low_mean_qv"]
    assert len(filtered) == 1
    assert filtered[0]["gene"] == "DRB4LIKE"
    assert filtered[0]["mean_qv"] == 5.0


def test_write_qc_json_filtered_calls_empty_when_nothing_filtered(tmp_path):
    result, _ = _run(tmp_path)
    qc_path = tmp_path / "qc_clean.json"
    write_qc_json(result, qc_path)
    obj = json.loads(qc_path.read_text())
    assert obj["calls_filtered_low_mean_qv"] == []


def test_write_calls_tsv_no_per_haplotype_c4_ab_rows(tmp_path):
    """Per-haplotype A/B calls produce no "C4A"/"C4B" rows."""
    result, _ = _run_phased(tmp_path)
    c4_routing = GenePhaseRouting(
        gene="C4", routing="phased", reason="single PS phase block, HP-tagged reads present",
        ps_values_seen=(100,), n_reads_hp1=12, n_reads_hp2=9, n_reads_untagged=0,
        haplotypes_called=(1, 2),
    )
    routing = dict(result.phasing_routing)
    routing["C4"] = c4_routing
    result = dataclasses.replace(
        result, phasing_routing=routing,
        c4_phased_ab_calls={
            1: C4AbCall(n_a=8, n_b=2, n_uninformative=2, a_fraction=0.8),
            2: C4AbCall(n_a=1, n_b=7, n_uninformative=1, a_fraction=0.125),
        },
        c4_phased_form_calls={
            1: C4FormLengthCall(n_long=10, n_short=0, n_uninformative=2, long_fraction=1.0),
            2: C4FormLengthCall(n_long=0, n_short=8, n_uninformative=1, long_fraction=0.0),
        },
    )

    out = tmp_path / "calls_c4_phased.tsv"
    write_calls_tsv(result, out)
    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert not any(r["gene"] in ("C4A", "C4B") for r in rows)
    # Aggregate "C4" row present; no per-haplotype "C4" row without c4_phased_cn.
    aggregate_c4_rows = [r for r in rows if r["gene"] in C4_EXTRA_CALLS_TSV_GENES and r["haplotype"] == "unphased"]
    assert {r["gene"] for r in aggregate_c4_rows} == {"C4"}


def test_write_calls_tsv_per_haplotype_c4_cn_rows(tmp_path):
    """c4_phased_cn adds a per-haplotype "C4" row; c4_phased_per_gene_cn
    produces no "C4A"/"C4B" rows."""
    result, _ = _run_phased(tmp_path)
    c4_routing = GenePhaseRouting(
        gene="C4", routing="phased", reason="single PS phase block, HP-tagged reads present",
        ps_values_seen=(100,), n_reads_hp1=12, n_reads_hp2=9, n_reads_untagged=0,
        haplotypes_called=(1, 2),
    )
    routing = dict(result.phasing_routing)
    routing["C4"] = c4_routing
    result = dataclasses.replace(
        result, phasing_routing=routing,
        c4_phased_ab_calls={
            1: C4AbCall(n_a=8, n_b=2, n_uninformative=2, a_fraction=0.8),
        },
        c4_phased_form_calls={
            1: C4FormLengthCall(n_long=10, n_short=0, n_uninformative=2, long_fraction=1.0),
        },
        c4_phased_cn={
            1: GeneCN(gene="C4", gene_class="III", depth=10, calibrator_median=5.0,
                      cn_estimate=4.0, ci_low=2.5, ci_high=5.5, is_zero_call=False, corroborated=False),
        },
        c4_phased_per_gene_cn={
            1: C4PerGeneCN(cn_a=3.2, cn_a_ci_low=1.0, cn_a_ci_high=5.0,
                            cn_b=0.8, cn_b_ci_low=0.1, cn_b_ci_high=2.0),
        },
    )

    out = tmp_path / "calls_c4_phased_cn.tsv"
    write_calls_tsv(result, out)
    with open(out) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    hap1_c4_row = next(r for r in rows if r["gene"] == "C4" and r["haplotype"] == "1")
    assert hap1_c4_row["cn"] == "4.00"
    assert hap1_c4_row["n_reads_assigned"] == "10"

    assert not any(r["gene"] in ("C4A", "C4B") for r in rows)


def test_write_qc_json_c4_per_haplotype_block(tmp_path):
    result, _ = _run_phased(tmp_path)
    routing = dict(result.phasing_routing)
    routing["C4"] = GenePhaseRouting(
        gene="C4", routing="phased", reason="single PS phase block, HP-tagged reads present",
        ps_values_seen=(100,), n_reads_hp1=12, n_reads_hp2=9, n_reads_untagged=0,
        haplotypes_called=(1, 2),
    )
    result = dataclasses.replace(
        result, phasing_routing=routing,
        c4_phased_ab_calls={1: C4AbCall(n_a=8, n_b=2, n_uninformative=2, a_fraction=0.8)},
        c4_phased_form_calls={1: C4FormLengthCall(n_long=10, n_short=0, n_uninformative=2, long_fraction=1.0)},
    )

    qc_path = tmp_path / "qc_c4_phased.json"
    write_qc_json(result, qc_path)
    obj = json.loads(qc_path.read_text())

    per_hap = obj["c4_form"]["per_haplotype"]
    assert per_hap is not None
    assert per_hap["1"]["n_a"] == 8
    assert per_hap["1"]["n_b"] == 2
    assert per_hap["1"]["n_long"] == 10
    assert per_hap["1"]["cn"] is None  # not injected in this test -- honest, not a guess


def test_write_qc_json_c4_per_haplotype_cn_block(tmp_path):
    """DESIGN.md S10.5/S10.6, added 2026-08-28: qc.json's per-haplotype C4
    block gains real cn/cn_a/cn_b (+ CIs) when c4_phased_cn/
    c4_phased_per_gene_cn are populated."""
    result, _ = _run_phased(tmp_path)
    routing = dict(result.phasing_routing)
    routing["C4"] = GenePhaseRouting(
        gene="C4", routing="phased", reason="single PS phase block, HP-tagged reads present",
        ps_values_seen=(100,), n_reads_hp1=12, n_reads_hp2=9, n_reads_untagged=0,
        haplotypes_called=(1, 2),
    )
    result = dataclasses.replace(
        result, phasing_routing=routing,
        c4_phased_ab_calls={1: C4AbCall(n_a=8, n_b=2, n_uninformative=2, a_fraction=0.8)},
        c4_phased_form_calls={1: C4FormLengthCall(n_long=10, n_short=0, n_uninformative=2, long_fraction=1.0)},
        c4_phased_cn={
            1: GeneCN(gene="C4", gene_class="III", depth=10, calibrator_median=5.0,
                      cn_estimate=4.0, ci_low=2.5, ci_high=5.5, is_zero_call=False, corroborated=False),
        },
        c4_phased_per_gene_cn={
            1: C4PerGeneCN(cn_a=3.2, cn_a_ci_low=1.0, cn_a_ci_high=5.0,
                            cn_b=0.8, cn_b_ci_low=0.1, cn_b_ci_high=2.0),
        },
    )

    qc_path = tmp_path / "qc_c4_phased_cn.json"
    write_qc_json(result, qc_path)
    obj = json.loads(qc_path.read_text())

    per_hap = obj["c4_form"]["per_haplotype"]
    assert per_hap["1"]["cn"] == 4.0
    assert per_hap["1"]["cn_ci_low"] == 2.5
    assert per_hap["1"]["cn_ci_high"] == 5.5
    assert per_hap["1"]["cn_depth"] == 10
    assert per_hap["1"]["cn_a"] == 3.2
    assert per_hap["1"]["cn_b"] == 0.8


def test_write_qc_json_c4_per_haplotype_none_when_not_phased(tmp_path):
    """Unphased C4 -> per_haplotype is None, not an empty dict."""
    result, _ = _run_phased(tmp_path)
    qc_path = tmp_path / "qc_no_c4_phasing.json"
    write_qc_json(result, qc_path)
    obj = json.loads(qc_path.read_text())
    assert obj["c4_form"]["per_haplotype"] is None
