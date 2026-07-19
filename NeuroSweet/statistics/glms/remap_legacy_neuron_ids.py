#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: remap_legacy_neuron_ids.py
Description: One-off recovery tool for engelhardglm.py results produced *before* the
    2026-07-17 local_neuron_id fix (commit 0dbb6cb, "Add SharedTransformerNN/dropout support,
    local neuron id tracking, and decoder-encoder comparison tooling").

    Prior to that fix, engelhardglm.py named each neuron's output file using a *global*
    neuron index (its position across the entire chunked/concatenated dataset passed to that
    SLURM array task), not the per-recording-local index (0..n_neurons_in_recording-1) that
    circuit_regression's decoder output uses for its `nuid` column. That mismatch makes
    compare_decoder_encoder.py fail to find any overlapping (nuid, stimulus) rows.

    Recovery is possible because the global neuron index is still monotonically increasing
    within a single recording (day, cage, mouse) -- engelhardglm/currate_data process each
    recording's neurons in a single contiguous block, never interleaved with another
    recording's neurons. So for any (day, cage, mouse) group whose neuron files are complete
    (no gaps in the global index run), re-ranking by the old global index recovers the exact
    local index the decoder expects.

    This script copies (never moves) legacy-named `.pkl.gz` files into a new directory with
    corrected `_N<local_id>.pkl.gz` suffixes, and reports which (day, cage, mouse) groups were
    remapped vs. skipped (because their neuron run isn't contiguous yet -- e.g. still syncing
    down from the cluster).

Usage:
    python remap_legacy_neuron_ids.py --input_directory /path/to/temp --output_directory /path/to/remapped
"""

import argparse
import os
import re
import shutil
from collections import defaultdict

FILENAME_RE = re.compile(r'^D(?P<day>[^_]+)_C(?P<cage>[^_]+)_M(?P<mouse>[^_]+)_G(?P<group>[^_]+)_N(?P<neuronid>\d+)\.pkl\.gz$')


def scan_legacy_files(input_directory):
    """ Group legacy-numbered .pkl.gz files by (day, cage, mouse, group). Returns a dict:
    (day, cage, mouse, group) -> sorted list of (old_global_id, filepath). """
    groups = defaultdict(list)
    for fname in os.listdir(input_directory):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        key = (m.group('day'), m.group('cage'), m.group('mouse'), m.group('group'))
        groups[key].append((int(m.group('neuronid')), os.path.join(input_directory, fname)))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def is_contiguous(global_ids):
    """ A recording's neuron run is only safely re-rankable if every neuron between the
    minimum and maximum observed global id is present -- i.e. no gaps from files that
    haven't finished downloading/syncing yet. """
    return len(global_ids) > 0 and (global_ids[-1] - global_ids[0] + 1) == len(global_ids)


def remap(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    groups = scan_legacy_files(input_directory)

    remapped_groups, skipped_groups = 0, 0
    remapped_files = 0
    for (day, cage, mouse, group), entries in groups.items():
        global_ids = [g for g, _ in entries]
        if not is_contiguous(global_ids):
            skipped_groups += 1
            print(f"Skipping D{day}_C{cage}_M{mouse}_G{group}: {len(entries)} files present, "
                  f"global id range {global_ids[0]}-{global_ids[-1]} has gaps (still syncing?).")
            continue

        for local_id, (_, src_path) in enumerate(entries):
            dst_name = f"D{day}_C{cage}_M{mouse}_G{group}_N{local_id}.pkl.gz"
            shutil.copy2(src_path, os.path.join(output_directory, dst_name))
            remapped_files += 1
        remapped_groups += 1

    print(f"Remapped {remapped_files} files across {remapped_groups} complete recordings "
          f"into {output_directory}; skipped {skipped_groups} incomplete recordings.")


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_directory', type=str, required=True,
                         help='Directory containing legacy-numbered engelhardglm .pkl.gz files')
    parser.add_argument('--output_directory', type=str, required=True,
                         help='Directory to write re-numbered copies into')
    return parser.parse_args()


if __name__ == '__main__':
    args = cli_parser()
    remap(args.input_directory, args.output_directory)
