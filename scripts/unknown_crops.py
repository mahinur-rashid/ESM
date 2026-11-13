"""Which target crops can the source model not name? Unknown-crop share by HCAT taxonomy level (the paper's taxonomy table).

    python scripts/unknown_crops.py --data /path/to/preprocess.zip

Reads only the file names in the archive. A crop is known to the source model if, after the same selection as in
the experiments (crops with >= 100 parcels per country, at most 1,500 sampled per crop and country), it has at least
300 source parcels. HCAT codes are merged to coarser levels by keeping their first 10, 8, 6 or 4 digits.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cropconf.data import find_archive, index_archive, open_archive  # noqa: E402

PAIRS = [("LV_to_EE", ["LV"], "EE"), ("EE_to_LV", ["EE"], "LV"), ("PT_to_LV", ["PT"], "LV"),
         ("LV_to_PT", ["LV"], "PT"), ("LVEE_to_PT", ["LV", "EE"], "PT")]
CAP, MIN_GROUP, MIN_SRC = 1500, 100, 300
NAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hcat_names.csv")


def parent(code, level):
    """Truncate a 10-digit HCAT code to its first `level` digits (pairs of digits = taxonomy levels)."""
    return code[:level].ljust(10, "0")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    names = pd.read_csv(NAMES, dtype=str).set_index("code")["name"].to_dict()
    label = lambda code: names.get(code, "?")                                       # noqa: E731
    counts = index_archive(open_archive(find_archive(args.data))).groupby(["cls", "country"]).size().unstack(fill_value=0)
    for pname, src, tgt in PAIRS:
        sampled = sum(counts[c].where(counts[c] >= MIN_GROUP, 0).clip(upper=CAP) for c in src)
        known = set(sampled.index[sampled >= MIN_SRC])
        tgt_counts = counts[tgt][counts[tgt] > 0].sort_values(ascending=False)
        tot = tgt_counts.sum()
        print(f"## {pname}: target {tgt} has {tot} parcels in {len(tgt_counts)} crop codes; source model knows {len(known)} codes")
        for level in (10, 8, 6, 4):
            known_par = {parent(c, level) for c in known}
            unk = sum(v for c, v in tgt_counts.items() if parent(c, level) not in known_par) / tot
            print(f"   unknown share with codes merged to {level} digits: {unk:.3f}")
        print("   top target crops (share, known?, nearest known code at 8 digits):")
        for c, v in tgt_counts.head(12).items():
            sib = [k for k in known if parent(k, 8) == parent(c, 8)]
            print(f"     {v / tot:6.1%}  {c} {label(c):38s} known={c in known}  same-family known: "
                  + (", ".join(label(k) for k in sib[:3]) or "-"))


if __name__ == "__main__":
    main()
