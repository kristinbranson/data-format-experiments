"""
Convert the CA1 geometric-deformation dataset of Lee, Keinath, Cianfarano & Brandon (2025),
"Identifying representational structure in CA1 to benchmark theoretical models of cognitive
mapping" (Neuron 113:307-320), into the decoder format expected by train_decoder.py.

Decoder task
------------
    inputs  : environment geometry (which of the 3x3 partitions of the arena are blocked),
              static within a session/trial
    outputs : the mouse's position discretised into the 3 x 3 = 9 spatial partitions,
              time-varying

Source data
-----------
Per-animal joblib files in /app/data (identical content to the .mat files, produced by the
paper's own `mat2joblib`).  Fields used (see the repository README):

    trace    : (n_days, n_cells, n_frames) binarised rising phase of calcium transients,
               NaN for a cell that was not registered on that day.  The paper treats this
               binary vector as the firing rate in every analysis.
    position : (n_days, 2, n_frames) head position from DeepLabCut, in cm, 0-75 cm in both
               x and y, already in a common frame across days.
    envs     : (n_days, 1) geometry name for each day.
    blocked  : per day, the indices of the occluded partitions in the 3x3 layout
               [[0, 1, 2], [3, 4, 5], [6, 7, 8]], or -1 when nothing is blocked.

Processing decisions (and where they come from)
-----------------------------------------------
* Neural signal.  The rising-phase binary trace is used, as in the paper ("This binary
  vector was treated as the firing rate in all further analyses").  It is smoothed and
  temporally binned exactly as in the paper's own position decoder
  (`utils.fit_decoder`/`test_decoder`): gaussian_filter1d along time with sigma = 3 frames,
  then average pooling over 3 frames.  At the 30 Hz acquisition rate of both the miniscope
  and the behavioural camera this gives 100 ms bins.  The pooled value is multiplied by the
  frame rate so that the stored quantity is an event rate in Hz, the same units the paper
  uses for its rate maps (`utils.get_rate_maps`).

* Temporal alignment.  The two streams are acquired by the same DAQ at 30 Hz and are already
  frame-aligned in the released arrays (trace and position have identical frame counts), so
  position is pooled over the *same* 3-frame windows as the neural data and no further
  alignment is needed.

* Trials.  Each session is a single continuous 40 min foraging session with no trial
  structure, so it is cut into consecutive 1 min trials (600 bins of 100 ms), as instructed.
  A trailing partial minute is dropped so that every trial has the same length.

* Output.  Position is binned with the paper's rule
  (`floor(position / ((max_position + buffer) / n_bins))`, `utils.decode_position_within`)
  with n_bins = 3 instead of 15, and with the scale taken from the maximum over *all* days
  of an animal, again as in `decode_position_within`.  This makes one bin exactly one 25 cm
  partition of the 75 cm arena, so the 9 classes are the 9 partitions of the paper's design.
  The partition index is 3 * y_bin + x_bin, which is the indexing of the `blocked` field
  (verified empirically: with this convention the occupancy of blocked partitions is 0.003%
  of frames on average).  The residual handful of samples that fall inside a blocked
  partition are tracking noise at partition walls; they are reassigned to the nearest open
  partition, mirroring the "cleaning" of actual and predicted bins to the nearest valid bin
  in `decode_position_within`.

* Input.  A 9-dimensional binary vector, 1 for a blocked partition and 0 for an open one,
  taken from the `blocked` field of the session.  `blocked` is used rather than
  `utils.get_env_mat(env_name)` because several geometries were run in a vertically mirrored
  version for some animals, and `blocked` records the configuration actually used.

* Curation.  All 7 mice, all 207 sessions and all 10 geometries are kept.  Within a session
  only cells registered on that day are kept (an unregistered cell is NaN, and NaNs are not
  allowed by the format); across the dataset that is the 69,744 cell-sessions the paper
  reports.  Cells with 5 or fewer transients in the session are then dropped, which is the
  `cell_threshold = 5` sparsity criterion of the paper's decoder (~1% of cells).  No
  place-cell/spatial-reliability selection is applied, matching the paper's decoding
  analysis, which uses all sufficiently active cells.

* Running speed.  The paper's Bayesian decoder additionally restricts fitting and testing to
  frames faster than 5 cm/s.  That criterion belongs to their decoding analysis rather than
  to the preparation of the dataset, and applying it here would delete ~48% of the recording
  and leave ragged, non-contiguous 1 min trials, so all time points are kept and the choice
  of whether to condition on running is left to the analysis.

Usage:  python convert_data.py [--out /app/converted_data.pkl] [--animals A B ...]
        [--max-sessions N]
"""

import argparse
import os
import pickle

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------
DATA_DIR = "/app/data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

FPS = 30                     # miniscope and behaviour camera acquisition rate (Hz)
FRAMES_PER_BIN = 3           # paper's temporal_bin_size in utils.fit_decoder -> 100 ms bins
TIME_BIN_MS = 1000.0 * FRAMES_PER_BIN / FPS
TRIAL_SECONDS = 60.0         # length of a trial
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FPS / FRAMES_PER_BIN))
SMOOTH_SIGMA_FRAMES = FRAMES_PER_BIN   # gaussian smoothing of traces, as in fit_decoder
N_SPATIAL_BINS = 3           # 3 x 3 partition design of the arena
POSITION_BUFFER = 1e-15      # buffer used when binning position in decode_position_within
CELL_EVENT_THRESHOLD = 5     # cell_threshold in decode_position_within
ARENA_SIZE_CM = 75.0

# centres of the 9 partitions, used to snap stray samples to the nearest open partition
_PART_CENTRES = np.array([[i % N_SPATIAL_BINS, i // N_SPATIAL_BINS]
                          for i in range(N_SPATIAL_BINS ** 2)], dtype=float)


def partition_names():
    """Human-readable name of each of the 9 spatial bins (x = column, y = row)."""
    return [f"x{i % N_SPATIAL_BINS}y{i // N_SPATIAL_BINS}"
            for i in range(N_SPATIAL_BINS ** 2)]


def blocked_indices(blocked_entry):
    """Indices of the occluded partitions of one session, as an integer array (empty for none)."""
    vals = np.atleast_1d(np.asarray(blocked_entry[0], dtype=float)).ravel()
    vals = vals[vals >= 0]           # -1 codes "no partition blocked" (the full square)
    return vals.astype(int)


def pool_mean(x, factor):
    """Average-pool the last axis of x in non-overlapping windows, dropping the remainder.

    Equivalent to torch.nn.AvgPool1d(kernel_size=factor, stride=factor), which is what the
    paper uses to temporally bin traces and position.
    """
    n = (x.shape[-1] // factor) * factor
    return x[..., :n].reshape(*x.shape[:-1], n // factor, factor).mean(axis=-1)


def convert(animals, out_path, max_sessions=None):
    neural_all, input_all, output_all = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []

    for subject, animal in enumerate(animals):
        print(f"Loading {animal} ...", flush=True)
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        trace, position = dat["trace"], dat["position"]
        envs = [str(e) for e in np.asarray(dat["envs"]).ravel()]
        blocked_all = dat["blocked"]
        n_days = trace.shape[0]

        # Spatial bin size, from the maximum position over all days of this animal, exactly
        # as in utils.decode_position_within.  This is ~25 cm, i.e. one partition.
        bin_size = (np.nanmax(position) + POSITION_BUFFER) / N_SPATIAL_BINS

        days = range(n_days) if max_sessions is None else range(min(n_days, max_sessions))
        for day in days:
            blocked = blocked_indices(blocked_all[day])
            open_parts = np.setdiff1d(np.arange(N_SPATIAL_BINS ** 2), blocked)

            # ---- neural -------------------------------------------------------------
            tr = trace[day]
            registered = ~np.isnan(tr[:, 0])          # a cell is NaN for the whole day or not
            tr = tr[registered]
            n_registered = int(registered.sum())
            # sparsity criterion of the paper's decoder: drop near-silent cells
            active = tr.sum(axis=1) > CELL_EVENT_THRESHOLD
            tr = tr[active]

            # smooth along time, then average pool into 100 ms bins, and express in Hz
            tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
            rates = (pool_mean(tr, FRAMES_PER_BIN) * FPS).astype(np.float32)

            # ---- output: 3x3 partition occupied at each time bin --------------------
            pos = pool_mean(position[day], FRAMES_PER_BIN)          # (2, n_time_bins)
            xb = np.clip((pos[0] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
            yb = np.clip((pos[1] // bin_size).astype(int), 0, N_SPATIAL_BINS - 1)
            part = N_SPATIAL_BINS * yb + xb

            # tracking noise can put a few samples inside a wall; snap them to the nearest
            # open partition, as decode_position_within snaps bins to the nearest valid one
            stray = np.isin(part, blocked)
            if stray.any():
                d = np.linalg.norm(_PART_CENTRES[part[stray]][:, None, :]
                                   - _PART_CENTRES[open_parts][None, :, :], axis=2)
                part[stray] = open_parts[np.argmin(d, axis=1)]

            n_bins = min(rates.shape[1], part.shape[0])
            n_trials = n_bins // BINS_PER_TRIAL     # drop a trailing partial minute

            # ---- input: geometry of the environment ---------------------------------
            geom = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
            geom[blocked] = 1.0

            neural_trials, input_trials, output_trials = [], [], []
            for t in range(n_trials):
                sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
                neural_trials.append(np.ascontiguousarray(rates[:, sl]))
                input_trials.append(geom.copy())
                output_trials.append(part[sl].astype(np.int64)[None, :])

            neural_all.append(neural_trials)
            input_all.append(input_trials)
            output_all.append(output_trials)
            subject_idx.append(subject)
            brain_region_idx.append(np.zeros(rates.shape[0], dtype=np.int64))
            session_info.append({
                "subject": animal,
                "day": int(day),
                "environment": envs[day],
                "blocked_partitions": [int(b) for b in blocked],
                "n_neurons": int(rates.shape[0]),
                "n_neurons_registered": n_registered,
                "n_trials": int(n_trials),
            })
            print(f"  day {day:2d} {envs[day]:>10s}  cells {rates.shape[0]:4d}"
                  f" (registered {n_registered:4d})  trials {n_trials}", flush=True)

        del dat, trace, position

    data = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,

        "subjects": list(animals),
        "subject_idx": np.array(subject_idx, dtype=np.int64),

        "brain_regions": ["CA1"],
        "brain_region_idx": brain_region_idx,

        "input_names": [f"blocked_{n}" for n in partition_names()],
        "output_names": ["position_bin"],
        "output_values": [partition_names()],

        "metadata": {
            "task_description":
                "Mice freely foraged for 40 min in a 75 x 75 cm arena whose geometry was "
                "changed across days by occluding partitions of a 3 x 3 grid (10 geometries, "
                "each sequence repeated up to three times).  The decoder receives CA1 "
                "population activity plus the geometry of the environment (which of the 9 "
                "partitions are blocked) and predicts which of the 9 partitions the mouse "
                "occupies at each time bin.",
            "time_bin_size": float(TIME_BIN_MS),
            "temporal_alignment_event":
                "start of each 1-min trial; the session is a single continuous recording with "
                "no trial structure, cut into consecutive 1-min segments from recording onset",
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
            "source":
                "Lee, Keinath, Cianfarano & Brandon (2025), Identifying representational "
                "structure in CA1 to benchmark theoretical models of cognitive mapping, "
                "Neuron 113(2):307-320.  Data: doi:10.5281/zenodo.13993254",
            "recording_modality":
                "one-photon miniscope calcium imaging of dorsal CA1, 30 Hz",
            "neural_signal":
                "binarised rising phase of calcium transients (the paper's firing-rate "
                "surrogate), gaussian-smoothed along time with sigma = 3 frames and "
                "average-pooled over 3 frames, expressed as an event rate in Hz",
            "behaviour":
                "head position from DeepLabCut, averaged over the same 3-frame windows and "
                "assigned to one of the 9 25 x 25 cm partitions of the arena",
            "arena_size_cm": ARENA_SIZE_CM,
            "spatial_bin_size_cm": ARENA_SIZE_CM / N_SPATIAL_BINS,
            "spatial_bin_layout":
                "bin index = 3 * y_bin + x_bin, i.e. [[0, 1, 2], [3, 4, 5], [6, 7, 8]], the "
                "partition indexing of the paper's 'blocked' field",
            "environments": sorted({s["environment"] for s in session_info}),
            "sampling_rate_hz": float(FPS),
            "neuron_selection":
                "cells registered on that day with more than 5 transients in the session",
            "session_info": session_info,
        },
    }

    n_trials = sum(len(s) for s in neural_all)
    n_neurons = sum(len(b) for b in brain_region_idx)
    print(f"\n{len(neural_all)} sessions, {n_trials} trials, {n_neurons} neuron-sessions")

    print(f"Writing {out_path} ...", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"done ({os.path.getsize(out_path) / 1e9:.2f} GB)")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/app/converted_data.pkl")
    ap.add_argument("--animals", nargs="+", default=ANIMALS)
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="only convert the first N sessions of each animal (for testing)")
    a = ap.parse_args()
    convert(a.animals, a.out, a.max_sessions)
