#!/usr/bin/env python3
"""Audit IPD-IMGT/HLA per-gene full-length coverage.

Streams hla.dat (EMBL flat file) and tabulates, per gene:
  - total alleles
  - non-partial ("full") alleles       <- what FuFiHLA's `grep -v partial` keeps
  - alleles with both 5'UTR and 3'UTR  <- true full-gene templates
  - allele-name field depth (how many are 4-field)
"""
import re
import sys
from collections import defaultdict

stats = defaultdict(lambda: {
    "total": 0, "full": 0, "genomic": 0, "f4": 0, "lens": []
})

cur = None


def flush():
    global cur
    if cur is None:
        return
    g = cur["gene"]
    s = stats[g]
    s["total"] += 1
    if not cur["partial"]:
        s["full"] += 1
        s["lens"].append(cur["len"])
        if cur["utr"] >= 2 and cur["intron"] > 0:
            s["genomic"] += 1
        if cur["fields"] >= 4:
            s["f4"] += 1
    cur = None


id_re = re.compile(r"^ID\s+\S+;.*?(\d+)\s+BP\.")
de_re = re.compile(r"^DE\s+(\S+?),")

with open("hla.dat", "r", errors="replace") as fh:
    for line in fh:
        if line.startswith("ID   "):
            flush()
            m = id_re.match(line)
            cur = {"gene": None, "partial": False, "utr": 0,
                   "intron": 0, "len": int(m.group(1)) if m else 0,
                   "fields": 0}
        elif cur is not None and line.startswith("DE   ") and cur["gene"] is None:
            m = de_re.match(line)
            if m:
                name = m.group(1)
                cur["gene"] = name.split("*")[0]
                if "*" in name:
                    cur["fields"] = len(name.split("*")[1].split(":"))
        elif cur is not None and line.startswith("FT"):
            if "/partial" in line:
                cur["partial"] = True
            elif line.startswith("FT   UTR "):
                cur["utr"] += 1
            elif line.startswith("FT   intron "):
                cur["intron"] += 1
flush()

rows = []
for g, s in stats.items():
    if not g:
        continue
    lens = sorted(s["lens"])
    med = lens[len(lens) // 2] if lens else 0
    rows.append((g, s["total"], s["full"], s["genomic"], s["f4"], med))

rows.sort(key=lambda r: -r[3])

print(f"{'gene':<12}{'total':>8}{'full':>8}{'genomic':>10}{'4-field':>9}{'med_bp':>9}")
print("-" * 56)
for r in rows:
    print(f"{r[0]:<12}{r[1]:>8}{r[2]:>8}{r[3]:>10}{r[4]:>9}{r[5]:>9}")

tot = sum(r[1] for r in rows)
full = sum(r[2] for r in rows)
utr = sum(r[3] for r in rows)
print("-" * 56)
print(f"{'TOTAL':<12}{tot:>8}{full:>8}{utr:>10}")
print(f"\ngenes: {len(rows)}   full/total: {full/tot:.1%}   genomic/total: {utr/tot:.1%}")
