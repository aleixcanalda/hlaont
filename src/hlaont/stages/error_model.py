"""Context-aware ONT read error model, ported from Badread (DESIGN.md S2e).

Replaces the flat per-base `edit_error_rate` this project started with
(a made-up constant, exactly what invariant 10/11 warn against) with a real
empirical model: Badread's `nanopore2023` per-7-mer error table, trained on
real ONT R10.4.1 alignments (`resources/models/vendor/`, ported verbatim,
GPLv3, see that directory's NOTICE.md). This is the same operating point as
invariant 1 -- unlike the earlier placeholder, this is not a guess.

**Badread uses this table generatively; we use it as a scorer, which is a
different problem and needs a different windowing scheme -- read this
before changing either.** Badread's own code
(`github.com/rrwick/Badread/blob/main/badread/error_model.py`,
`make_error_model()`) builds the table by sliding a window over an aligned
read/reference pair *one reference base at a time*, so a single true error
gets counted into ~7 overlapping training windows. That is correct for
accumulating statistics, but multiplying ~7x-overlapping per-window
probabilities to score one real alignment would raise every base's
contribution to roughly the 7th power and is not a real likelihood.
`context_log_likelihood()` instead advances by whole, non-overlapping
7-base chunks of the reference -- an approximation (a homopolymer run or
indel straddling a chunk boundary can get mis-attributed to the wrong side
of it), but a tractable, honest one, flagged here rather than silently
assumed. Revisit if milestone-1 calibration shows this costs real accuracy.

Only ACGT k-mers are ever queried in practice: `resources/models/vendor/`
holds an entry for all 4**7 = 16384 possible 7-mers (confirmed empirically,
not assumed), so the `default_remainder` fallback below exists for
defensiveness (e.g. an ambiguity-code base slipping through), not because
it is expected to fire on real IPD template sequences.
"""
from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DEFAULT_NANOPORE2023_MODEL = REPO_ROOT / "resources" / "models" / "vendor" / "nanopore2023.gz"

_FLOOR_PROB = 1e-12
_CIGAR_OP_RE = re.compile(r"(\d+)([MIDNSHP=X])")
_REVCOMP_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(_REVCOMP_TABLE)[::-1]


@dataclass(frozen=True)
class KmerErrorModel:
    kmer_size: int
    # kmer -> {observed_kmer: probability}, including the self ("correctly
    # sequenced") entry. Values for one kmer sum to <= 1 (Badread truncates
    # the alternative list; see resources/models/vendor/NOTICE.md).
    probs: dict[str, dict[str, float]]
    # kmer -> 1 - sum(probs[kmer].values()): the probability mass of some
    # alternative *not* explicitly listed for that kmer.
    remainder: dict[str, float]
    # Fallback for a query kmer entirely absent from the table (see module
    # docstring -- not expected to fire on real ACGT template sequences).
    default_remainder: float


def load_kmer_error_model(path: Path = DEFAULT_NANOPORE2023_MODEL) -> KmerErrorModel:
    """Parse a Badread-format k-mer error model file (gzip or plain text).
    Each line is `kmer,p;alt,p;alt,p;...`; the first entry is always the
    kmer itself (P(correctly sequenced)). See
    resources/models/vendor/NOTICE.md for the full format description and
    provenance."""
    opener = gzip.open if str(path).endswith(".gz") else open
    probs: dict[str, dict[str, float]] = {}
    remainder: dict[str, float] = {}
    kmer_size: int | None = None
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries = [e.split(",") for e in line.split(";") if e]
            kmer = entries[0][0]
            if kmer_size is None:
                kmer_size = len(kmer)
            elif len(kmer) != kmer_size:
                raise ValueError(f"{path}: inconsistent k-mer length at {kmer!r}")
            row = {alt: float(p) for alt, p in entries}
            probs[kmer] = row
            remainder[kmer] = max(0.0, 1.0 - sum(row.values()))
    if kmer_size is None:
        raise ValueError(f"{path}: no k-mer entries found")
    default_remainder = sum(remainder.values()) / len(remainder)
    return KmerErrorModel(
        kmer_size=kmer_size, probs=probs, remainder=remainder,
        default_remainder=default_remainder,
    )


def log_prob(model: KmerErrorModel, ref_kmer: str, obs_kmer: str) -> float:
    """log P(obs_kmer | ref_kmer). Falls back to `remainder[ref_kmer]` (the
    probability of *some* unlisted alternative) if `obs_kmer` isn't one of
    the explicitly trained alternatives for `ref_kmer` -- crediting one
    specific unlisted outcome with the whole unlisted bucket's probability
    is generous (the bucket can represent many possible k-mers), but there
    is no finer-grained information in the table to split it further."""
    row = model.probs.get(ref_kmer)
    if row is None:
        return math.log(max(model.default_remainder, _FLOOR_PROB))
    p = row.get(obs_kmer)
    if p is None:
        p = model.remainder[ref_kmer]
    return math.log(max(p, _FLOOR_PROB))


def _expand_cigar(ref_span: str, read_span: str, cigar: str) -> tuple[str, str]:
    """ref_span/read_span must already be sliced to exactly the aligned
    region, same strand orientation as the CIGAR. Returns gapped
    (aligned_ref, aligned_read), '-' padding each other out across indels.
    """
    aligned_ref: list[str] = []
    aligned_read: list[str] = []
    ri = qi = 0
    for length_str, op in _CIGAR_OP_RE.findall(cigar):
        n = int(length_str)
        if op in "M=X":
            aligned_ref.append(ref_span[ri:ri + n])
            aligned_read.append(read_span[qi:qi + n])
            ri += n
            qi += n
        elif op == "D":
            aligned_ref.append(ref_span[ri:ri + n])
            aligned_read.append("-" * n)
            ri += n
        elif op == "I":
            aligned_ref.append("-" * n)
            aligned_read.append(read_span[qi:qi + n])
            qi += n
        elif op == "S":
            # Not expected inside a PAF cg:Z: block (minimap2's qstart/qend
            # already exclude soft-clipped bases), guarded rather than
            # assumed away.
            qi += n
        else:
            raise ValueError(f"unsupported/unexpected CIGAR op {op!r} in {cigar!r}")
    return "".join(aligned_ref), "".join(aligned_read)


def context_log_likelihood(model: KmerErrorModel, aligned_ref: str, aligned_read: str) -> float:
    """Sum of per-window log P(observed | reference) over disjoint
    `model.kmer_size`-base chunks of the alignment (see module docstring
    for why this must be disjoint, not Badread's own overlapping-by-1
    stepping)."""
    k = model.kmer_size
    total = 0.0
    start = 0
    n = len(aligned_ref)
    while start < n:
        end = start
        ref_bases = 0
        while end < n and ref_bases < k:
            if aligned_ref[end] != "-":
                ref_bases += 1
            end += 1
        ref_kmer = aligned_ref[start:end].replace("-", "")
        read_kmer = aligned_read[start:end].replace("-", "")
        if (
            len(ref_kmer) == k and len(read_kmer) >= 1
            and ref_kmer[0] == read_kmer[0] and ref_kmer[-1] == read_kmer[-1]
        ):
            total += log_prob(model, ref_kmer, read_kmer)
        elif len(ref_kmer) == k:
            # An edit landed on this window's own anchor base (or the whole
            # window was deleted) -- Badread's own training loop excludes
            # these cases too (`len(read_kmer) > 1` guard in
            # make_error_model), so there is no trained entry to look up.
            # Score as one average-unlikely event rather than silently
            # skipping (which would hide the very edit we're trying to
            # detect) or crashing.
            total += math.log(max(model.default_remainder, _FLOOR_PROB))
        # else: trailing partial window (< k reference bases left) at the
        # alignment's end -- nothing more to score.
        start = end
    return total


def build_aligned_strings(
    ref_full_seq: str,
    read_full_seq: str,
    *,
    ref_start: int,
    ref_end: int,
    read_start: int,
    read_end: int,
    strand: str,
    cigar: str,
) -> tuple[str, str]:
    """ref_full_seq/read_full_seq are the *full* allele/read sequences
    (not pre-sliced); start/end are PAF-convention 0-based half-open
    coordinates. Query coordinates in PAF are always given on the forward
    read strand regardless of `strand` -- when strand == '-', the cg:Z:
    CIGAR describes the alignment against the reverse complement of that
    query span (PAF spec), so the read span is revcomp'd here before the
    CIGAR is applied.
    """
    ref_span = ref_full_seq[ref_start:ref_end]
    read_span = read_full_seq[read_start:read_end]
    if strand == "-":
        read_span = reverse_complement(read_span)
    return _expand_cigar(ref_span, read_span, cigar)


def read_allele_log_likelihood(
    model: KmerErrorModel,
    ref_full_seq: str,
    read_full_seq: str,
    *,
    ref_start: int,
    ref_end: int,
    read_start: int,
    read_end: int,
    strand: str,
    cigar: str,
) -> float:
    """log P(read | allele) for one (read, allele) alignment under the
    context model -- the function stages/genotype.py calls per candidate
    (read, allele) pair."""
    aligned_ref, aligned_read = build_aligned_strings(
        ref_full_seq, read_full_seq,
        ref_start=ref_start, ref_end=ref_end,
        read_start=read_start, read_end=read_end,
        strand=strand, cigar=cigar,
    )
    return context_log_likelihood(model, aligned_ref, aligned_read)
