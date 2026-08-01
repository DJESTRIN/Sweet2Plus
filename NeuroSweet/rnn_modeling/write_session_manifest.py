#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: write_session_manifest.py
Description: Small helper for the SLURM pipeline -- writes a plain-text session manifest
    (one obj*.json path per line) for a given data_directory, and prints the total session
    count, so submit_rnn_pipeline.sh can size its --array without needing to import any heavy
    NeuroSweet code inside the submission shell script itself.
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import os
import sys
import argparse

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from NeuroSweet.rnn_modeling.rnn_data import list_sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--out_manifest", type=str, required=True)
    args = parser.parse_args()

    manifest = list_sessions(args.data_directory)
    with open(args.out_manifest, "w") as f:
        for path in manifest["path"]:
            f.write(path + "\n")
    print(f"n_sessions={len(manifest)}")
    print(f"Wrote manifest -> {args.out_manifest}")


if __name__ == "__main__":
    main()
