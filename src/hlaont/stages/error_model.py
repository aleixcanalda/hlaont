"""Context-aware ONT read error model from Badread's nanopore2023 per-7-mer
table (ONT R10.4.1; resources/models/vendor/, GPLv3) (DESIGN.md S2e).

Badread slides its window one base at a time to build the table; scoring an
alignment that way would count each base ~7 times. `context_log_likelihood`
instead uses disjoint 7-base chunks, an approximation (events straddling a
chunk boundary can be misattributed).
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
    # kmer -> {observed_kmer: probability}, including the kmer itself; may
    # sum to < 1 (Badread truncates alternatives).
    probs: dict[str, dict[str, float]]
    # kmer -> 1 - sum(probs[kmer].values()): the probability mass of some
    # alternative *not* explicitly listed for that kmer.
    remainder: dict[str, float]
    # Fallback for a query kmer entirely absent from the table (see module
    # docstring -- not expected to fire on real ACGT template sequences).
    default_remainder: float


def load_kmer_error_model(path: Path = DEFAULT_NANOPORE2023_MODEL) -> KmerErrorModel:
    """Parse a Badread k-mer model (gzip or plain): lines of
    `kmer,p;alt,p;...`, the first entry being the kmer itself."""
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
    """log P(obs_kmer | ref_kmer); unlisted alternatives get the kmer's
    whole unlisted remainder."""
    row = model.probs.get(ref_kmer)
    if row is None:
        return math.log(max(model.default_remainder, _FLOOR_PROB))
    p = row.get(obs_kmer)
    if p is None:
        p = model.remainder[ref_kmer]
    return math.log(max(p, _FLOOR_PROB))


def _expand_cigar(ref_span: str, read_span: str, cigar: str) -> tuple[str, str]:
    """Gapped (aligned_ref, aligned_read) from spans already sliced to the
    aligned region, in CIGAR orientation."""
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
            # Soft clips shouldn't occur inside PAF cg:Z:; guarded anyway.
            qi += n
        else:
            raise ValueError(f"unsupported/unexpected CIGAR op {op!r} in {cigar!r}")
    return "".join(aligned_ref), "".join(aligned_read)


def context_log_likelihood(model: KmerErrorModel, aligned_ref: str, aligned_read: str) -> float:
    """Sum of log P(observed | reference) over disjoint k-base chunks."""
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
            # Edit on the window's anchor base (or whole window deleted):
            # Badread has no entry; score as an average unlisted event.
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
    """Gapped strings from full sequences and PAF 0-based half-open
    coordinates. PAF query coordinates are forward-strand; for strand '-'
    the read span is reverse-complemented before applying the CIGAR."""
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
    """log P(read | allele) for one alignment under the context model."""
    aligned_ref, aligned_read = build_aligned_strings(
        ref_full_seq, read_full_seq,
        ref_start=ref_start, ref_end=ref_end,
        read_start=read_start, read_end=read_end,
        strand=strand, cigar=cigar,
    )
    return context_log_likelihood(model, aligned_ref, aligned_read)
