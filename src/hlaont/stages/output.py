"""Output writers for a SampleResult (DESIGN.md S5): calls.tsv,
consensus.fa + consensus.qv.tsv, variants.vcf, qc.json.

Not built yet: annotations.gtf (needs a per-haplotype coordinate remap) and
sufficient_stats.parquet (no schema defined).

variants.vcf uses VCF anchor-base indels in `called_allele`'s own
coordinates (CHROM = allele name). QUAL is always "." because novelty
variants carry no per-base QV.
"""
from __future__ import annotations

import json
from pathlib import Path

from hlaont.stages.c4_cn import C4_REPORTED_GENE
from hlaont.stages.c4_form import C4AbCall, C4FormLengthCall, C4PerGeneCN
from hlaont.stages.cn import GeneCN
from hlaont.stages.genotype import load_allele_sequences
from hlaont.stages.novelty import DEFAULT_POOL_META, NoveltyVariant, load_gtf_template_allele
from hlaont.stages.pipeline import DEFAULT_POOL_FASTA, HaplotypeCallResult, SampleResult

# calls.tsv excludes haplotype calls below this mean QV (e.g. DRB4 at 4
# reads, QV 5, 37 variants); qc.json still lists them. Provisional.
DEFAULT_MIN_MEAN_QV = 30.0


def _split_by_mean_qv(
    calls: tuple[HaplotypeCallResult, ...], min_mean_qv: float,
) -> tuple[list[HaplotypeCallResult], list[HaplotypeCallResult]]:
    """(kept, filtered) by mean QV; filtered calls are reported in qc.json."""
    kept = [c for c in calls if c.mean_qv >= min_mean_qv]
    filtered = [c for c in calls if c.mean_qv < min_mean_qv]
    return kept, filtered


def write_calls_tsv(
    result: SampleResult, path: Path, *, min_mean_qv: float = DEFAULT_MIN_MEAN_QV,
    pool_meta: Path = DEFAULT_POOL_META,
) -> None:
    """One row per haplotype call, then C4 rows.

    `template_allele` is the nearest catalogued allele (Stage 4's
    called_allele, never `:new`), shown as the source GTF's real name for
    assembly-extracted genes and as the matched partial allele for
    `extension:partial`; `allele` is Stage 5's name. Calls with mean QV
    below `min_mean_qv` are omitted (listed in qc.json).

    `haplotype` column (phased runs only): the HP value for phased genes,
    "unphased" otherwise.

    C4 rows fill only gene/cn/n_reads_assigned/c4_form_note
    (`long=<n>,short=<n>`):
    - "C4", then C4AL/C4AS/C4BL/C4BS from the chr17-control mechanism
      (integer joint call if available, else continuous estimates);
    - if no chr17-control result, a single "C4" row from gene_cn["C4"];
    - phased runs: an extra per-haplotype "C4" row when that haplotype has
      a C4 CN.
    The A/B vote-ratio split is reported only in qc.json."""
    phased_mode = result.phasing_routing is not None
    cols = [
        "gene", "cn", "template_allele", "allele", "taxonomy", "n_fields",
        "field_gq", "n_reads_assigned", "consensus_length", "mean_qv",
        "n_variants", "c4_form_note",
    ]
    if phased_mode:
        cols.append("haplotype")
    kept_calls, _filtered_calls = _split_by_mean_qv(result.haplotype_calls, min_mean_qv)
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for c in kept_calls:
            if c.taxonomy == "extension:partial" and c.extension_allele is not None:
                # CDS matches a partially catalogued allele: display it.
                display_template_allele = c.extension_allele
            else:
                display_template_allele = load_gtf_template_allele(c.called_allele, pool_meta) \
                    or c.called_allele
            row = [
                c.gene, str(c.cn), display_template_allele, c.proposed_name,
                c.taxonomy, str(len(c.field_gq)),
                ",".join(f"{gq:.1f}" for gq in c.field_gq),
                str(c.n_reads_assigned), str(c.consensus_length),
                f"{c.mean_qv:.1f}", str(c.n_variants), "",
            ]
            if phased_mode:
                routing = result.phasing_routing.get(c.gene)  # type: ignore[union-attr]
                if routing is not None and routing.routing == "phased":
                    row.append(str(c.haplotype_index))
                else:
                    row.append("unphased")
            fh.write("\t".join(row) + "\n")

        c4_cn = result.gene_cn.get(C4_REPORTED_GENE)
        form = result.c4_form_call
        chr17 = result.c4_chr17_control
        form_note = "" if form is None else f"long={form.n_long},short={form.n_short}"
        c4_extra_col = ["unphased"] if phased_mode else []

        if chr17 is not None:
            # chr17-control total (integer joint call when available).
            joint = result.c4_joint_call
            if joint is not None:
                cn_total = str(joint.total)
            else:
                cn_total = "" if chr17.total_cn_estimate is None else f"{chr17.total_cn_estimate:.2f}"
            n_reads_total = sum(f.n_reads for f in chr17.forms.values())
            row = [C4_REPORTED_GENE, cn_total, "", "", "", "", "", str(n_reads_total), "", "", "", form_note] + c4_extra_col
            fh.write("\t".join(row) + "\n")
            for form_name in ("C4AL", "C4AS", "C4BL", "C4BS"):
                form_cn = chr17.forms.get(form_name)
                if form_cn is None:
                    continue
                if joint is not None and form_name in joint.forms:
                    cn_str = str(joint.forms[form_name])
                else:
                    cn_str = "" if form_cn.cn_estimate is None else f"{form_cn.cn_estimate:.2f}"
                row = [form_name, cn_str, "", "", "", "", "", str(form_cn.n_reads), "", "", "", ""] + c4_extra_col
                fh.write("\t".join(row) + "\n")
        elif c4_cn is not None:
            # Fallback, no chr17_control computed for this run (module
            # docstring) -- the pre-2026-09-01 shape, unchanged.
            cn_total = "" if c4_cn.cn_estimate is None else f"{c4_cn.cn_estimate:.2f}"
            row = [C4_REPORTED_GENE, cn_total, "", "", "", "", "", str(c4_cn.depth), "", "", "", form_note] + c4_extra_col
            fh.write("\t".join(row) + "\n")

        # Per-haplotype "C4" rows (phased mode).
        if phased_mode and result.c4_phased_ab_calls:
            for hp_val in sorted(result.c4_phased_ab_calls):
                hp_form = (result.c4_phased_form_calls or {}).get(hp_val)
                hp_form_note = "" if hp_form is None else f"long={hp_form.n_long},short={hp_form.n_short}"
                hp_cn = (result.c4_phased_cn or {}).get(hp_val)
                if hp_cn is not None:
                    cn_total = "" if hp_cn.cn_estimate is None else f"{hp_cn.cn_estimate:.2f}"
                    row = [C4_REPORTED_GENE, cn_total, "", "", "", "", "", str(hp_cn.depth), "", "", "", hp_form_note, str(hp_val)]
                    fh.write("\t".join(row) + "\n")


def write_consensus_fasta(result: SampleResult, fasta_path: Path, qv_path: Path) -> None:
    """One FASTA record per haplotype copy; QVs go to a long-format TSV
    (record_id, pos, qv) rather than a FASTQ quality string.
    `haplotype_calls` and `consensus_records` are parallel lists."""
    with open(fasta_path, "w") as fasta_fh, open(qv_path, "w") as qv_fh:
        qv_fh.write("record_id\tpos\tqv\n")
        for call, hap in zip(result.haplotype_calls, result.consensus_records):
            record_id = f"{call.gene}_hap{call.haplotype_index}"
            fasta_fh.write(f">{record_id} template={hap.template_allele} n_reads={hap.n_reads_assigned}\n")
            seq = hap.sequence
            for i in range(0, len(seq), 80):
                fasta_fh.write(seq[i:i + 80] + "\n")
            for pos, qv in enumerate(hap.qv, start=1):
                qv_fh.write(f"{record_id}\t{pos}\t{qv}\n")


def _vcf_records_for_haplotype(
    gene: str, haplotype_index: int, called_allele: str, ref_seq: str,
    variants: tuple[NoveltyVariant, ...],
) -> list[str]:
    lines: list[str] = []
    for v in variants:
        info = f"GENE={gene};HAP={haplotype_index}"
        if v.ref == "" and v.alt != "":  # insertion
            anchor_pos = v.pos - 1
            anchor_base = ref_seq[anchor_pos - 1] if anchor_pos >= 1 else "N"
            pos, ref, alt = anchor_pos if anchor_pos >= 1 else 1, anchor_base, anchor_base + v.alt
        elif v.alt == "" and v.ref != "":  # deletion
            anchor_pos = v.pos - 1
            anchor_base = ref_seq[anchor_pos - 1] if anchor_pos >= 1 else "N"
            pos, ref, alt = anchor_pos if anchor_pos >= 1 else 1, anchor_base + v.ref, anchor_base
        else:  # substitution
            pos, ref, alt = v.pos, v.ref, v.alt
        # QUAL is "." (no per-base QV on novelty variants).
        lines.append(f"{called_allele}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}")
    return lines


def write_variants_vcf(
    result: SampleResult, path: Path, *, pool_fasta: Path = DEFAULT_POOL_FASTA
) -> None:
    called_alleles = {
        nc.called_allele for nc in result.novelty_calls.values()
    }
    ref_seqs = load_allele_sequences(pool_fasta, tuple(sorted(called_alleles))) if called_alleles else {}

    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(
            "##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene name\">\n"
            "##INFO=<ID=HAP,Number=1,Type=Integer,Description=\"Haplotype copy index\">\n"
        )
        fh.write(
            "## NOTE: CHROM is the nearest known IPD allele's own name, POS is in "
            "that allele's own coordinate numbering (matches "
            "ref.gene.models.jsonl.gz) -- not a genome coordinate "
            "(DESIGN.md S5: \"relative to closest reference allele\").\n"
        )
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for (gene, h_idx), nc in sorted(result.novelty_calls.items()):
            ref_seq = ref_seqs[nc.called_allele]
            for line in _vcf_records_for_haplotype(gene, h_idx, nc.called_allele, ref_seq, nc.variants):
                fh.write(line + "\n")


def _c4_per_haplotype_entry(
    ab: C4AbCall, form: C4FormLengthCall | None,
    cn: GeneCN | None = None, per_gene_cn: C4PerGeneCN | None = None,
) -> dict:
    return {
        "n_a": ab.n_a, "n_b": ab.n_b,
        "n_uninformative": ab.n_uninformative, "a_fraction": ab.a_fraction,
        "n_long": None if form is None else form.n_long,
        "n_short": None if form is None else form.n_short,
        "long_fraction": None if form is None else form.long_fraction,
        # None when no PS-matched calibrator gene was available.
        "cn": None if cn is None else cn.cn_estimate,
        "cn_ci_low": None if cn is None else cn.ci_low,
        "cn_ci_high": None if cn is None else cn.ci_high,
        "cn_depth": None if cn is None else cn.depth,
        "cn_a": None if per_gene_cn is None else per_gene_cn.cn_a,
        "cn_a_ci_low": None if per_gene_cn is None else per_gene_cn.cn_a_ci_low,
        "cn_a_ci_high": None if per_gene_cn is None else per_gene_cn.cn_a_ci_high,
        "cn_b": None if per_gene_cn is None else per_gene_cn.cn_b,
        "cn_b_ci_low": None if per_gene_cn is None else per_gene_cn.cn_b_ci_low,
        "cn_b_ci_high": None if per_gene_cn is None else per_gene_cn.cn_b_ci_high,
    }


def write_qc_json(
    result: SampleResult,
    path: Path,
    *,
    dorado_version: str | None = None,
    reference_build_md5: str | None = None,
    min_mean_qv: float = DEFAULT_MIN_MEAN_QV,
) -> None:
    _kept_calls, filtered_calls = _split_by_mean_qv(result.haplotype_calls, min_mean_qv)
    obj = {
        "sample_id": result.sample_id,
        "sieve_stats": {
            "n_reads": result.sieve_stats.n_reads,
            "n_kept": result.sieve_stats.n_kept,
            "n_dropped_no_qualifying_alignment": result.sieve_stats.n_dropped_no_qualifying_alignment,
            "n_dropped_decoy_or_score_filter": result.sieve_stats.n_dropped_decoy_or_score_filter,
            "kept_per_gene": dict(sorted(result.sieve_stats.kept_per_gene.items())),
        },
        "gene_cn": {
            gene: {
                "depth": cn.depth,
                "cn_estimate": cn.cn_estimate,
                "ci_low": cn.ci_low,
                "ci_high": cn.ci_high,
                "is_zero_call": cn.is_zero_call,
                "corroborated": cn.corroborated,
            }
            for gene, cn in sorted(result.gene_cn.items())
        },
        "failures": [
            {"gene": f.gene, "reason": f.reason} for f in result.failures
        ],
        # Calls omitted from calls.tsv by the mean-QV floor.
        "calls_filtered_low_mean_qv": [
            {
                "gene": c.gene, "haplotype_index": c.haplotype_index,
                "mean_qv": c.mean_qv, "n_reads_assigned": c.n_reads_assigned,
            }
            for c in filtered_calls
        ],
        # Per-gene phasing routing; None for unphased runs.
        "phasing": None if result.phasing_routing is None else {
            gene: {
                "routing": r.routing,
                "reason": r.reason,
                "ps_values_seen": list(r.ps_values_seen),
                "n_reads_hp1": r.n_reads_hp1,
                "n_reads_hp2": r.n_reads_hp2,
                "n_reads_untagged": r.n_reads_untagged,
                "haplotypes_called": list(r.haplotypes_called),
            }
            for gene, r in sorted(result.phasing_routing.items())
        },
        "c4_form": {
            "a_vs_b": None if result.c4_ab_call is None else {
                "n_a": result.c4_ab_call.n_a,
                "n_b": result.c4_ab_call.n_b,
                "n_uninformative": result.c4_ab_call.n_uninformative,
                "a_fraction": result.c4_ab_call.a_fraction,
        # CN_A/CN_B: gene_cn["C4"] split by a_fraction.
                "cn_a": None if result.c4_per_gene_cn is None else result.c4_per_gene_cn.cn_a,
                "cn_a_ci_low": None if result.c4_per_gene_cn is None else result.c4_per_gene_cn.cn_a_ci_low,
                "cn_a_ci_high": None if result.c4_per_gene_cn is None else result.c4_per_gene_cn.cn_a_ci_high,
                "cn_b": None if result.c4_per_gene_cn is None else result.c4_per_gene_cn.cn_b,
                "cn_b_ci_low": None if result.c4_per_gene_cn is None else result.c4_per_gene_cn.cn_b_ci_low,
                "cn_b_ci_high": None if result.c4_per_gene_cn is None else result.c4_per_gene_cn.cn_b_ci_high,
            },
            "long_vs_short": None if result.c4_form_call is None else {
                "n_long": result.c4_form_call.n_long,
                "n_short": result.c4_form_call.n_short,
                "n_uninformative": result.c4_form_call.n_uninformative,
                "long_fraction": result.c4_form_call.long_fraction,
            },
            # Per-HP C4 tallies and CN; None unless C4 routed phased. A
            # haplotype with an RCCX duplication can show both A and B.
            "per_haplotype": (
                None if not result.c4_phased_ab_calls
                else {
                    str(hp): _c4_per_haplotype_entry(
                        ab, (result.c4_phased_form_calls or {}).get(hp),
                        (result.c4_phased_cn or {}).get(hp),
                        (result.c4_phased_per_gene_cn or {}).get(hp),
                    )
                    for hp, ab in sorted(result.c4_phased_ab_calls.items())
                }
            ),
        },
        # chr17-control per-form C4 CN and control depth; None if not computed.
        "c4_chr17_control": None if result.c4_chr17_control is None else {
            "control_depth": result.c4_chr17_control.control_depth,
            "control_n_reads": result.c4_chr17_control.control_n_reads,
            "total_cn_estimate": result.c4_chr17_control.total_cn_estimate,
            # Count-based total and inputs; depth-based numbers kept as a
            # cross-check.
            "depth_total_cn_estimate": result.c4_chr17_control.depth_total_cn_estimate,
            "n_reads_5p_window": result.c4_chr17_control.n_reads_5p_window,
            "reads_per_copy": result.c4_chr17_control.reads_per_copy,
            "herv_spanning": {
                "long": result.c4_chr17_control.herv_long_spanning,
                "short": result.c4_chr17_control.herv_short_spanning,
                "long_exposure": result.c4_chr17_control.herv_long_exposure,
                "short_exposure": result.c4_chr17_control.herv_short_exposure,
            },
            "forms": {
                form_name: {
                    "n_reads": form_cn.n_reads,
                    "depth": form_cn.depth,
                    "cn_estimate": form_cn.cn_estimate,
                    "depth_cn_estimate": form_cn.depth_cn_estimate,
                }
                for form_name, form_cn in sorted(result.c4_chr17_control.forms.items())
            },
        },
        # 2026-09-25: integer per-form call (stages/c4_chr17_cn.py
        # call_c4_forms_jointly); None when no count-based total exists.
        "c4_joint_call": None if result.c4_joint_call is None else {
            "forms": dict(result.c4_joint_call.forms),
            "total": result.c4_joint_call.total,
            "gq": result.c4_joint_call.gq,
            "log_likelihood": result.c4_joint_call.log_likelihood,
        },
        "dorado_version": dorado_version,
        "reference_build_md5": reference_build_md5,
    }
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
