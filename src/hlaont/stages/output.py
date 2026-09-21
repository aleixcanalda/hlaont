"""Output writers for a SampleResult (DESIGN.md S5).

Implements four of the six named outputs: `calls.tsv`, `consensus.fa` (+ a
`consensus.qv.tsv` companion), `variants.vcf`, `qc.json`. **Not yet built,
flagged rather than faked:** `annotations.gtf` (needs a real per-haplotype
coordinate remap of GeneModelLite's exon/intron/UTR ranges onto the
consensus's own coordinates -- a design task of its own, not a trivial
format conversion) and `sufficient_stats.parquet` (DESIGN.md names it as
"the Python calibration layer's input" but does not define a schema; there
is no calib/ consumer yet to design one against). Building either as a
placeholder would be worse than omitting it -- the development notes' "no half-
finished implementations."

`variants.vcf` reconstructs proper VCF anchor-base convention for indels
(an anchor base immediately before the event, REF/ALT both including it) --
DESIGN.md S5 deliberately left this to "an output-writing concern" when
`stages/consensus.py` and `stages/novelty.py` were built; this is that
concern. Positions are in `called_allele`'s own coordinate numbering (same
numbering as resources/templates/pool/ref.gene.models.jsonl.gz), not a
genome coordinate -- CHROM is the called allele's own name, per DESIGN.md's
"variants.vcf -- relative to closest reference allele."

**QUAL is always "."** -- `stages/novelty.py`'s `NoveltyVariant` carries no
per-base QV (`classify()` only ever receives the raw consensus string, not
`stages/consensus.py`'s `HaplotypeConsensus.qv` array it was built from).
Wiring the real per-base QV through requires either novelty.py taking the
full HaplotypeConsensus, or this writer re-deriving a ref-to-consensus
coordinate map itself (nontrivial once the alignment is reverse-stranded).
Flagged as a real, bounded follow-up, not silently faked with a placeholder
number.
"""
from __future__ import annotations

import json
from pathlib import Path

from c4_cn import C4_REPORTED_GENE
from c4_form import C4AbCall, C4FormLengthCall, C4PerGeneCN
from cn import GeneCN
from genotype import load_allele_sequences
from novelty import DEFAULT_POOL_META, NoveltyVariant, load_gtf_template_allele
from pipeline import DEFAULT_POOL_FASTA, HaplotypeCallResult, SampleResult

# User-requested, 2026-08-27: a haplotype call this thin on consensus
# confidence (real HG00097 finding: HLA-DRB4, 4 reads, mean_qv 5.0, 37
# variants -- passed the read-count floor but is still noise) is excluded
# from calls.tsv's primary summary table entirely, not merely flagged.
# Provisional (invariant 11): the user's own stated threshold, not fitted.
DEFAULT_MIN_MEAN_QV = 30.0


def _split_by_mean_qv(
    calls: tuple[HaplotypeCallResult, ...], min_mean_qv: float,
) -> tuple[list[HaplotypeCallResult], list[HaplotypeCallResult]]:
    """(kept, filtered) -- filtered calls are never silently dropped: they
    are still reported, by qc.json (write_qc_json's own filter, same
    threshold), just not in the primary calls.tsv table."""
    kept = [c for c in calls if c.mean_qv >= min_mean_qv]
    filtered = [c for c in calls if c.mean_qv < min_mean_qv]
    return kept, filtered


def write_calls_tsv(
    result: SampleResult, path: Path, *, min_mean_qv: float = DEFAULT_MIN_MEAN_QV,
    pool_meta: Path = DEFAULT_POOL_META,
) -> None:
    """User-requested trimmed schema (2026-08-22): `sample_id` is dropped
    since output filenames are now sample-prefixed (see cli.py); `allele` is
    Stage 5's own name (novelty tag included) rather than Stage 4's
    `called_allele`; `candidate_null`, `ambiguity_set`, and `min_qv` are
    dropped from this file -- still available on `HaplotypeCallResult`
    itself for any caller that needs them.

    **`template_allele` column, added 2026-09-03, user-requested.**
    Stage 4's own `called_allele` (`HaplotypeCallResult.called_allele`) --
    the nearest catalogued IPD allele, *never* carrying a `:new` suffix --
    exposed as its own column, immediately before `allele`. Motivation
    (user's own, echoing invariant 8): Immuannot's own novelty call (the
    `:new` tag in this project's truth GTFs) is itself an inference, not
    perfect ground truth -- benchmarking against `allele` alone conflates
    "did we pick the right nearest allele" with "did we agree with
    Immuannot's own novelty judgment" into one pass/fail. `template_allele`
    lets a scorer compare template-to-template (nearest-allele accuracy,
    independent of any novelty-calling disagreement between hlaont and
    Immuannot) and separately compare `allele`/`taxonomy` against truth's
    own `:new` tag (novelty-calling accuracy) rather than only the
    conflated version. For an `exact` call, `template_allele` and `allele`
    are identical by construction (stages/novelty.py's own `proposed_name`
    is `display_allele` unchanged when there are zero variants) -- this
    column only diverges from `allele` when a real difference triggered a
    `novel:*`/`extension` tag. Blank for the C4/C4A/C4B/C4AL/C4AS/C4BL/C4BS
    rows below, same as every other per-haplotype-only column.

    **Assembly-extracted genes (HLA-N/P/T/U/W/Y/R), fixed same day, same
    user report.** `called_allele` for these genes is a content-hashed pool
    accession (resources/templates/assembly_sources.py, e.g.
    "HLA-W*asm3e9f0510aa") -- there is no real IPD catalogue entry to name
    it after. `load_gtf_template_allele()` (stages/novelty.py) looks up
    ref.gene.meta.tsv.gz's `gtf_template_allele` column -- the *source*
    hla_seq/ GTF's own gene-line `template_allele` attribute for the exact
    real haplotype sequence this accession was extracted from (e.g.
    "HLA-W*01:01:01:05") -- and that real name is written into this column
    instead, falling back to `called_allele` unchanged when no such lookup
    exists (every ordinary IPD-sourced allele, and any assembly-extracted
    record built before this column existed). Purely a display swap: the
    accession itself (`called_allele`) is still what every other stage
    used internally, and `allele`/`taxonomy` are computed exactly as
    before -- this only changes what this one column prints.

    **`extension:partial` calls, added 2026-09-21, user-requested.** Same
    display-swap idea, one step further: when Stage 5 classifies a
    haplotype as `extension:partial` (stages/novelty.py's
    `_partial_extension_match()` -- the consensus's CDS exactly matches a
    *different* IPD allele than `called_allele`, one only ever catalogued
    as a partial/exon-subset record, e.g. real HG03009/HLA-C*07:457), this
    column shows that matched allele (`extension_allele`) instead of the
    genomic template Stage 2/4 actually aligned/built consensus against.
    `called_allele` itself, and everything Stage 3/4 computed from it, are
    unchanged -- only this display column and (already, unrelated to this
    change) `allele`/`taxonomy` reflect the partial match.

    **`haplotype_index` removed, 2026-08-27, user-requested.** This
    pipeline never phases reads -- Stage 2's per-haplotype split is an
    arbitrary assignment order out of template-set selection, not a real
    maternal/paternal phase, so a column named `haplotype_index` implying
    real phase would misrepresent what the tool actually did. Real phasing
    is tracked as a future, separate pipeline (see the development notes' "Next"
    list), not something this column can honestly claim in the meantime.

    **User-requested, 2026-08-26/27: C4/C4A/C4B rows, revised repeatedly
    per follow-up feedback (see the development notess for the full
    back-and-forth).** C4A/C4B never produce a `HaplotypeCallResult`
    (DESIGN.md S2f: no star-allele nomenclature, no template pool, no
    per-haplotype consensus to report -- `typing_module: structural` genes
    are excluded from the manifest the Stage 2-5 loop walks), so most of
    this schema's columns -- built around a per-haplotype template call --
    simply don't apply. **Final shape:** three extra rows --
    `C4_REPORTED_GENE` ("C4", stages/c4_cn.py's combined, length-corrected
    total CN) and `"C4A"`/`"C4B"` (stages/c4_form.py's
    `estimate_c4_per_gene_cn()` -- a proportional split of that total by
    the exon 26 vote ratio, not an independent per-gene measurement; that
    function's own docstring has the full reasoning and precision
    caveats) -- each leaving every per-haplotype column empty except `cn`,
    `n_reads_assigned`, and `c4_form_note`. `n_reads_assigned` is
    `c4_cn.depth` for the "C4" row and `C4AbCall.n_a`/`n_b` for "C4A"/
    "C4B" -- it stays the real read tally regardless of whether `cn` could
    be computed, since the two columns answer different questions and one
    being unavailable must not blank the other. `c4_form_note` carries
    stages/c4_form.py's long-vs-short read tally (`long=<n_long>,
    short=<n_short>`, identical across all three rows since it's a
    locus-wide measurement, not something resolved per A/B) so a reader
    can see at a glance whether the sample's C4 reads are all long, all
    short, or a real mix -- full uninformative-read counts and fractions
    for both axes stay in `qc.json` only.

    **`haplotype` column, phased mode only (DESIGN.md S10.7).** Added
    *only* when `result.phasing_routing is not None` (i.e. the sample was
    run through `run_sample_phased()`) -- an unphased run's schema is
    byte-identical to before this existed, preserving the same reasoning
    that removed `haplotype_index` from this file in the first place: a
    column implying real phase must not appear unless the phase is real.
    When present, it carries the real `HP` tag value (1/2) for a gene
    routed "phased" (`HaplotypeCallResult.haplotype_index` is set to that
    same real value by `run_sample_phased()`), or the literal string
    `"unphased"` for a gene that fell back to the ordinary combinatorial
    path -- never a fabricated label for a real haplotype the tool cannot
    actually back up. The whole-sample "C4"/"C4A"/"C4B" rows carry
    `"unphased"` too in phased mode (Phase 1/2's aggregate computation is
    the same "unphased" computation regardless of whether C4 also got a
    per-haplotype breakdown, see below) -- never blank, so a reader can
    filter on this column uniformly.

    **`min_mean_qv` (default 30.0), user-requested 2026-08-27.** A
    haplotype call whose `mean_qv` falls below this is excluded from this
    table entirely -- real HG00097 finding: `HLA-DRB4*01:new` at 4 reads,
    `mean_qv` 5.0, 37 variants, cleared every read-count floor upstream but
    is still noise, not a call worth surfacing in the primary summary.
    Never silently dropped project-wide: `write_qc_json()` reports the
    same excluded calls (gene, haplotype label, mean_qv) under
    `calls_filtered_low_mean_qv` with the identical threshold, and
    `consensus.fa`/`variants.vcf` are untouched (still every haplotype,
    unfiltered) -- this filter is calls.tsv-only, a display decision, not
    a claim that the underlying consensus/variant calls don't exist.

    **User-requested, 2026-08-26/27, per-haplotype C4A/C4B added
    2026-08-27 (DESIGN.md S10.6), revised repeatedly per follow-up feedback
    (see the development notess for the full back-and-forth).** C4A/C4B
    never produce a `HaplotypeCallResult` (DESIGN.md S2f: no star-allele
    nomenclature, no template pool, no per-haplotype consensus to report),
    so most of this schema's columns -- built around a per-haplotype
    template call -- simply don't apply. **Base shape (both modes):**
    three rows -- `C4_REPORTED_GENE` ("C4", stages/c4_cn.py's combined,
    length-corrected total CN) and `"C4A"`/`"C4B"` (stages/c4_form.py's
    `estimate_c4_per_gene_cn()` -- a proportional split of that total by
    the exon 26 vote ratio, not an independent per-gene measurement) --
    each leaving every per-haplotype column empty except `cn`,
    `n_reads_assigned`, and `c4_form_note`. `n_reads_assigned` is
    `c4_cn.depth` for the "C4" row and `C4AbCall.n_a`/`n_b` (sample-wide)
    for "C4A"/"C4B". **Phased mode, extra rows when
    `result.c4_phased_ab_calls` is populated** (i.e. C4A/C4B's own
    `GenePhaseRouting.routing == "phased"`, S10.6): one additional `"C4A"`
    and `"C4B"` row per haplotype that cleared the routing floor,
    `n_reads_assigned` that haplotype's own vote count, `c4_form_note` that
    haplotype's own long-vs-short tally, `haplotype` the real HP value --
    additive to, not a replacement for, the sample-wide aggregate rows
    above (a reader filtering on `haplotype != "unphased"` sees only the
    genuinely phase-resolved rows). **`cn`, added 2026-08-28 (DESIGN.md
    S10.5/S10.6, Phase 1 per-haplotype), populated only when this
    haplotype's own PS-matched calibrator panel produced a real estimate**
    (`stages/pipeline.py`'s `run_sample_phased()` docstring has the full
    PS-matching mechanism) -- blank, never a guess, otherwise. When
    populated, an extra per-haplotype `"C4"` row also appears (mirroring
    the sample-wide aggregate row's shape: `cn` = that haplotype's own
    length-corrected total, `n_reads_assigned` = that haplotype's own C4
    depth); `"C4A"`/`"C4B"`'s own `cn` is that total split by the same
    haplotype's own A-vs-B vote ratio (`estimate_c4_per_gene_cn()`,
    unchanged, just called per haplotype). `min_mean_qv` does not apply to
    any C4 row (no `mean_qv` is ever computed for them).

    **2026-09-01, user-directed: the chr17-control mechanism
    (`stages/c4_chr17_cn.py`) is now production for C4's sample-wide
    (unphased) numbers.** When `result.c4_chr17_control` is populated
    (every unphased run computes it; phased runs also carry it forward on
    the shared aggregate, since the per-haplotype C4 path has not been
    moved onto this mechanism yet -- module docstring on `SampleResult.
    c4_chr17_control`), the `"C4"` row's `cn` comes from `total_cn_
    estimate` (real per-base `samtools depth` against a real, external
    chr17 control region -- the development notes' own session log found this
    measurably closer to real truth than the calibrator-panel total below)
    rather than `gene_cn["C4"]` (`stages/c4_cn.py`'s older mechanism,
    still computed every run and still visible in `qc.json`'s own
    `"gene_cn"` block as a live cross-check, not silently dropped -- just
    no longer what `calls.tsv` reports as the headline number). Four new
    rows, `"C4AL"`/`"C4AS"`/`"C4BL"`/`"C4BS"`, report each real form's own
    directly-measured `cn`/`n_reads_assigned` (`stages/c4_chr17_cn.py`'s
    own per-form BAM regions, not a proportional split like `"C4A"`/
    `"C4B"` above) -- additive to, not a replacement for, the existing
    `"C4A"`/`"C4B"` rows, since the two answer different questions (a
    combined-form vote-ratio split vs. a directly-measured per-form
    count). Falls back to the pre-2026-09-01 `gene_cn["C4"]`-based `"C4"`
    row (no C4AL/C4AS/C4BL/C4BS rows at all) when `c4_chr17_control` is
    `None` -- a caller that never computed it (e.g. a test fixture
    predating this revision), not a silent gap in production.

    **2026-09-01, same day, user-directed: `"C4A"`/`"C4B"` rows dropped
    from this file entirely, sample-wide and per-haplotype alike.** Every
    paragraph above describing a `"C4A"`/`"C4B"` row is historical --
    superseded by this note, not deleted, per this project's own
    conventions. Rationale: those rows were a proportional split of the
    `"C4"` total by `estimate_c4_per_gene_cn()`'s exon-26 vote ratio, not
    an independent measurement -- once `"C4AL"`/`"C4AS"`/`"C4BL"`/
    `"C4BS"` existed as real, directly-measured, chr17-control-normalised
    per-form numbers (this file's own paragraph immediately above), the
    coarser A-vs-B split answered a strictly weaker version of the same
    question (`cn_a` is recoverable as `C4AL + C4AS`, `cn_b` as
    `C4BL + C4BS`, when both per-form estimates exist). `result.
    c4_ab_call`/`c4_per_gene_cn`/`c4_phased_ab_calls`/
    `c4_phased_per_gene_cn` are untouched on `SampleResult` and still
    populated every run (`stages/pipeline.py` unchanged) -- only this
    file's row output changed; `qc.json` (`write_qc_json()`) still reports
    the vote-ratio split in full, so the underlying measurement is not
    lost, just no longer duplicated into `calls.tsv`. The per-haplotype
    `"C4"` row (phased mode) is otherwise unchanged."""
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
                # The real genomic template used to align/build consensus
                # (c.called_allele) is Stage 2/4's own alignment anchor, not
                # the actual known allele -- stages/novelty.py's
                # extension:partial gate already confirmed the consensus's
                # CDS exactly matches a *different*, real (if only
                # partially-catalogued) IPD allele over its own covered
                # range. Same display-swap precedent as the assembly-
                # extraction gtf_template_allele case below: purely a
                # display column, called_allele is still what every other
                # stage used internally, and allele/taxonomy are computed
                # exactly as before -- this only changes what this one
                # column prints.
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
            # 2026-09-01: chr17-control total supersedes gene_cn["C4"] here
            # (module docstring) -- the older mechanism stays computed and
            # visible in qc.json's own "gene_cn" block, just not repeated
            # in this row.
            cn_total = "" if chr17.total_cn_estimate is None else f"{chr17.total_cn_estimate:.2f}"
            n_reads_total = sum(f.n_reads for f in chr17.forms.values())
            row = [C4_REPORTED_GENE, cn_total, "", "", "", "", "", str(n_reads_total), "", "", "", form_note] + c4_extra_col
            fh.write("\t".join(row) + "\n")
            for form_name in ("C4AL", "C4AS", "C4BL", "C4BS"):
                form_cn = chr17.forms.get(form_name)
                if form_cn is None:
                    continue
                cn_str = "" if form_cn.cn_estimate is None else f"{form_cn.cn_estimate:.2f}"
                row = [form_name, cn_str, "", "", "", "", "", str(form_cn.n_reads), "", "", "", ""] + c4_extra_col
                fh.write("\t".join(row) + "\n")
        elif c4_cn is not None:
            # Fallback, no chr17_control computed for this run (module
            # docstring) -- the pre-2026-09-01 shape, unchanged.
            cn_total = "" if c4_cn.cn_estimate is None else f"{c4_cn.cn_estimate:.2f}"
            row = [C4_REPORTED_GENE, cn_total, "", "", "", "", "", str(c4_cn.depth), "", "", "", form_note] + c4_extra_col
            fh.write("\t".join(row) + "\n")

        # DESIGN.md S10.5/S10.6: per-haplotype "C4" row, phased mode only,
        # additive to the aggregate rows above. Only appears when this
        # haplotype's PS-matched calibrator panel produced a real estimate
        # (stages/pipeline.py's run_sample_phased() docstring) -- blank, not
        # a guess, otherwise. 2026-09-01, user-requested: "C4A"/"C4B" rows
        # (the proportional A-vs-B/long-vs-short vote-ratio split) dropped
        # from calls.tsv entirely, sample-wide and per-haplotype alike --
        # "C4AL"/"C4AS"/"C4BL"/"C4BS" (directly measured, chr17-control-
        # normalised) already answer the same question with a real
        # per-form measurement instead of a split of the "C4" total, so the
        # older, coarser breakdown is redundant. `c4_form_call`/
        # `c4_per_gene_cn`/`c4_phased_ab_calls`/`c4_phased_per_gene_cn`
        # remain on `SampleResult` and in qc.json -- only this file's own
        # rows change.
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
    """`fasta_path` gets one record per haplotype copy (`sequence`);
    `qv_path` is a long-format companion (record_id, pos, qv) rather than a
    FASTQ quality string -- avoids phred+33 ASCII encoding entirely
    (consensus.py's QV can be int-valued past the printable-ASCII-safe
    range some tools assume) and is trivial to parse without a FASTQ
    library.

    `result.haplotype_calls` and `result.consensus_records` are appended in
    lockstep, one pair per haplotype, inside stages/pipeline.py's own loop
    -- zipping them (rather than re-deriving a haplotype index from
    `.index()`, which would give the wrong, flat-across-all-genes position
    and rely on dataclass equality) is what recovers each consensus's real
    per-gene haplotype_index."""
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
        # QUAL is always "." -- NoveltyVariant carries no per-base QV
        # (novelty.classify() only receives the raw consensus string, not
        # stages/consensus.py's HaplotypeConsensus.qv array it came from).
        # Flagged, not faked: see module docstring's deferred-work note.
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
        # DESIGN.md S10.5/S10.6, added 2026-08-28: per-haplotype total C4
        # CN (Phase 1) -- None when this haplotype's own PS-matched
        # calibrator panel had no usable gene (stages/pipeline.py's
        # run_sample_phased() docstring), not a guess.
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
        # User-requested, 2026-08-27: calls.tsv's own mean_qv floor
        # (write_calls_tsv, same min_mean_qv/DEFAULT_MIN_MEAN_QV) excludes
        # a row entirely rather than merely flagging it -- reported here so
        # a low-confidence call never simply vanishes with no record
        # anywhere (development notes: "silent failure... is worse than a crash").
        "calls_filtered_low_mean_qv": [
            {
                "gene": c.gene, "haplotype_index": c.haplotype_index,
                "mean_qv": c.mean_qv, "n_reads_assigned": c.n_reads_assigned,
            }
            for c in filtered_calls
        ],
        # DESIGN.md S10.7/user-requested: which genes went through the
        # phased per-haplotype path vs the unphased fallback, and why --
        # None for an ordinary run_sample() (unphased) call, one entry per
        # resources/genes.yaml manifest gene for a run_sample_phased() call.
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
                # 2026-08-27, user-requested: CN_A/CN_B, a proportional
                # split of gene_cn["C4"]'s total by a_fraction -- see
                # stages/c4_form.py's estimate_c4_per_gene_cn() docstring
                # for the derivation and its real precision caveats.
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
            # DESIGN.md S10.6, user-requested 2026-08-27: per-haplotype
            # A-vs-B/long-vs-short vote tallies, keyed by real HP value --
            # None unless C4_REPORTED_GENE's own GenePhaseRouting.routing
            # (see "phasing" above) is "phased". A haplotype can carry both
            # A and B (or long and short) votes if it has its own internal
            # RCCX duplication (stages/pipeline.py's own docstring) -- not
            # a hard per-haplotype genotype call, the same honest-tally
            # stance as the sample-wide a_vs_b/long_vs_short blocks above.
            # `cn`/`cn_a`/`cn_b` (+ CIs), added 2026-08-28 (DESIGN.md
            # S10.5/S10.6): this haplotype's own total/per-gene C4 CN, None
            # when its PS-matched calibrator panel had no usable gene
            # (run_sample_phased()'s own docstring has the full mechanism).
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
        # 2026-09-01, user-directed: the chr17-control mechanism
        # (stages/c4_chr17_cn.py), now production for calls.tsv's own "C4"
        # row (see write_calls_tsv's own docstring) -- real per-form
        # samtools-depth CN/read counts plus the external control's own
        # depth/read count, for full traceability of what the calls.tsv
        # numbers were computed from. None for a run that never computed
        # it (module docstring on SampleResult.c4_chr17_control).
        "c4_chr17_control": None if result.c4_chr17_control is None else {
            "control_depth": result.c4_chr17_control.control_depth,
            "control_n_reads": result.c4_chr17_control.control_n_reads,
            "total_cn_estimate": result.c4_chr17_control.total_cn_estimate,
            "forms": {
                form_name: {
                    "n_reads": form_cn.n_reads,
                    "depth": form_cn.depth,
                    "cn_estimate": form_cn.cn_estimate,
                }
                for form_name, form_cn in sorted(result.c4_chr17_control.forms.items())
            },
        },
        "dorado_version": dorado_version,
        "reference_build_md5": reference_build_md5,
    }
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
