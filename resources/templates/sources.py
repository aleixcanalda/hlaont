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
from dataclasses import dataclass
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


def _flush(cur: dict | None, genomic_genes: set[str], source_version: str) -> TemplateRecord | None:
    if cur is None or cur["gene"] is None or cur["partial"]:
        return None
    if cur["gene"] not in genomic_genes:
        return None
    genomic = cur["utr"] >= 2 and cur["intron"] > 0
    if not genomic:
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
        genomic=True,
        fields=cur["fields"],
        source="IPD-IMGT/HLA",
        source_version=source_version,
    )


def parse_ipd_dat(dat_path: Path, genomic_genes: set[str]) -> Iterator[TemplateRecord]:
    """Stream hla.dat, yielding one TemplateRecord per full-length ("genomic":
    both UTRs plus at least one intron present, DESIGN.md S2b) allele whose
    gene is in genomic_genes. Same criterion as ipd_audit.py, not FuFiHLA's
    coarser `grep -v partial` on the description line.
    """
    cur: dict | None = None
    in_seq = False
    source_version = "unknown"

    with open(dat_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("ID   "):
                rec = _flush(cur, genomic_genes, source_version)
                if rec is not None:
                    yield rec
                m = _ID_RE.match(line)
                cur = {
                    "id": None, "gene": None, "allele": None, "description": "",
                    "partial": False, "utr": 0, "intron": 0,
                    "len": int(m.group(1)) if m else 0, "fields": 0,
                    "seq_lines": [],
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
                if "/partial" in line:
                    cur["partial"] = True
                elif line.startswith("FT   UTR "):
                    cur["utr"] += 1
                elif line.startswith("FT   intron "):
                    cur["intron"] += 1
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

    rec = _flush(cur, genomic_genes, source_version)
    if rec is not None:
        yield rec


class IPDSource:
    """IPD-IMGT/HLA, pinned by GitHub tag (DESIGN.md: "Pin it")."""

    name = "IPD-IMGT/HLA"

    def __init__(self, genomic_genes: set[str], raw_dir: Path, tag: str = IPD_DEFAULT_TAG,
                 dat_path: Path | None = None):
        self.genomic_genes = genomic_genes
        self.tag = tag
        self.raw_dir = raw_dir
        self._dat_path = dat_path
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
        yield from parse_ipd_dat(dat_path, self.genomic_genes)
