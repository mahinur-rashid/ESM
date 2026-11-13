"""Run one experiment of the paper and write its tables.

    python scripts/run_experiment.py --experiment main     --data /path/to/preprocess.zip --out outputs
    python scripts/run_experiment.py --experiment controls --data /path/to/preprocess.zip --out outputs

Outputs (under --out): runs_<experiment>/ (one JSON per run; a re-run resumes from them), tables_<experiment>/ (CSV
tables read by make_paper_tables.py), and the parcel cache.
"""
import argparse
import glob
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cropconf.config import EXPERIMENTS, make_config  # noqa: E402
from cropconf.data import Data, find_archive, md5sum  # noqa: E402
from cropconf.models import Predictor  # noqa: E402
from cropconf.runner import aggregate, run_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--data", default=None, help="EuroCropsML preprocess.zip (or an unpacked folder); "
                                                 "default: $ECML_ZIP or a copy under /kaggle/input")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--cache-search", nargs="*", default=["/kaggle/input"],
                    help="folders searched for an existing parcel cache")
    ap.add_argument("--presto-dir", default=None, help="folder with single_file_presto.py and default_model.pt "
                                                       "(default: /kaggle/input, else download from GitHub)")
    ap.add_argument("--fast-dev", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--check-md5", action="store_true", help="verify the archive against the Zenodo checksum")
    args = ap.parse_args()

    cfg = make_config(args.experiment, fast_dev=args.fast_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    archive = find_archive(args.data)
    print(f"experiment {args.experiment} | models {cfg['archs']} | device {device} | data {archive}")
    if args.check_md5 and os.path.isfile(archive):
        ok = md5sum(archive) == cfg["zip_md5"]
        print("md5", "matches Zenodo record 15095445" if ok else "DIFFERS from Zenodo record 15095445")

    data = Data(archive, cfg, cache_dir=args.out, cache_search_dirs=args.cache_search)
    predictor = Predictor(data, cfg, device, work_dir=args.out, presto_dir=args.presto_dir)
    run_dir = os.path.join(args.out, f"runs_{cfg['tag']}")
    os.makedirs(run_dir, exist_ok=True)
    for root in args.cache_search:            # resume: finished runs from a previous session's output
        for f in glob.glob(os.path.join(glob.escape(root), "**", f"runs_{cfg['tag']}", "*.json"), recursive=True):
            dst = os.path.join(run_dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
    run_grid(data, predictor, cfg, run_dir, device)
    tables = aggregate(cfg, run_dir, os.path.join(args.out, f"tables_{cfg['tag']}"))
    print(f"{len(tables['run_info'])} runs aggregated into {os.path.join(args.out, 'tables_' + cfg['tag'])} "
          f"({(time.time() - t0) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
