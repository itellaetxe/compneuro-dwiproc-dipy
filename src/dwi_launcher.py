#!/usr/bin/env python3
"""
dwi_launcher.py
===============

Runs the whole DIPY pipeline participant by participant -- the counterpart of
``compneuro-dwiproc/src/dwi_launcher.sh``:

* participants come from ``--participant-label``, else ``participants.tsv``,
  else every ``sub-*`` folder (or the folder itself if ``--bids-dir`` is a
  single subject);
* a participant whose ``<parc>_det.csv`` already exists is skipped
  ("already processed"), as in the original;
* preprocessing is skipped if it already finished with the same correction
  route ("Dwi data already cleaned"); a run done with another route (e.g. the
  earlier DIPY-only version) is redone, reusing finished steps such as
  denoising; unfinished steps resume where they stopped; tractography results
  older than the preprocessed data are recomputed;
* one global log ``<output-dir>/log/Dwipreproc_<date_time>.txt`` (like the
  original) plus one log per participant/run in ``<run dir>/log.txt``;
* an error in one participant is logged and the launcher moves on; the
  failed participants are listed at the end.

Example (test subject sub-6175)::

    ~/venvs/neurovenv/bin/python src/dwi_launcher.py \\
        --bids-dir ~/Desktop/sub-6175 \\
        --output-dir ~/Desktop/sub-6175/derivatives/dwiproc_dipy \\
        --fs-dir ~/Desktop/sub-6175/derivatives/Recon \\
        --parcellation Desikan_genetics_2mm --n-jobs 8
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwiproc_common as dc  # noqa: E402
import dwi_fsl  # noqa: E402
import dwi_preproc  # noqa: E402
import dwi_tract  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DIPY DWI pipeline launcher (preprocessing + "
                                            "tractography/connectomes).",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    dc.add_common_arguments(p)
    p.add_argument("--stages", nargs="+", choices=["preproc", "tract"],
                   default=["preproc", "tract"], help="Stages to run (default: both).")
    dwi_preproc.add_preproc_arguments(p)
    dwi_fsl.add_fsl_arguments(p)
    dwi_tract.add_tract_arguments(p)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if "tract" in args.stages and not args.parcellation:
        parser.error("--parcellation is required when the 'tract' stage is run")
    stamp = time.strftime("%Y%m%d_%H%M")
    main_log = args.output_dir / "log" / f"Dwipreproc_{stamp}.txt"
    logger = dc.get_logger(main_log)
    parc_name = dc.parcellation_name(dc.resolve_template(args.parcellation)) \
        if "tract" in args.stages else None
    participants = dc.list_participants(args.bids_dir, args.participant_label)
    logger.info(f"compneuro-dwiproc-dipy | {len(participants)} participant(s) | "
                f"stages: {' + '.join(args.stages)} | {dc.software_versions()}")
    failed, skipped = [], []

    for participant in participants:
        try:
            runs = dc.find_dwi_runs(args.bids_dir, participant)
        except FileNotFoundError as e:
            logger.warning(f"{participant}: {e} - skipped")
            skipped.append(participant)
            continue
        if not runs:
            logger.warning(f"{participant}: no DWI run with .bval/.bvec found - skipped")
            skipped.append(participant)
            continue
        for run in runs:
            n_same = sum(r["session"] == run["session"] for r in runs)
            out = dc.run_output_dir(args.output_dir, participant, run, n_same)
            label = participant + (f" {run['session']}" if run["session"] else "") + \
                (f" {run['stem']}" if n_same > 1 else "")
            pre_ok, pre_why = dwi_preproc.preproc_status(out, run, participant, args) \
                if (out / dwi_preproc.PREPROC_DONE).is_file() else (False, "not preprocessed yet")
            det_csv = out / f"{parc_name}_det.csv" if parc_name else None
            if "tract" in args.stages and det_csv.is_file() and not args.overwrite and pre_ok \
                    and det_csv.stat().st_mtime >= (out / "dwi_clean.nii.gz").stat().st_mtime:
                logger.info(f"{label} already processed")
                continue
            logger.info("*********************")
            logger.info(label)
            logger.info("*********************")
            run_log = out / "log.txt"
            dc.get_logger(run_log)
            try:
                if "preproc" in args.stages:
                    if pre_ok:
                        logger.info(f"Dwi data already cleaned: {pre_why}")
                    else:
                        logger.info(f"Preprocessing: {pre_why}")
                        dwi_preproc.run_preproc(run, out, participant, args, logger)
                if "tract" in args.stages:
                    dwi_tract.run_tract(out, participant, args, logger)
            except Exception as e:  # keep going with the next participant
                logger.error(f"{label} FAILED: {e}\n{traceback.format_exc()}")
                failed.append(label)
            finally:
                dc.remove_file_handler(logger, run_log)

    logger.info(f"Finished. Failed: {failed or 'none'}. Skipped (no DWI): {skipped or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
