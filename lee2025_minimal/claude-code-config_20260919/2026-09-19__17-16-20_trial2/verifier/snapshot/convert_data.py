"""
Convert the CA1 geometric-deformation dataset of

    "Identifying representational structure in CA1 to benchmark theoretical
     models of cognitive mapping"  (georepca1)

into the nested dict / pickle format used by ``train_decoder.py``.

Decoder task
------------
    input  : environment geometry (which of the 3 x 3 arena partitions are
             blocked), static within a trial   -> 9 binary values
    output : the mouse's position discretised into the same 3 x 3 = 9 spatial
             bins, time varying                -> 1 categorical output

Source data
-----------
``data/<animal>.mat`` is a MATLAB v7.3 file holding, for one mouse:

    envs      (n_days,)             geometry name of each daily session
    blocked   (n_days,)             0-based flat indices of the 3x3 partitions
                                    that were walled off that day (-1 = none)
    position  (n_days,)             (n_frames, 2) head position in cm, in the
                                    coordinate frame of the full 75 x 75 cm
                                    square, sampled at 30 Hz (DeepLabCut)
    trace     (n_days,)             (n_frames, n_cells) binarised rising phase
                                    of the calcium transients -- the quantity
                                    the paper treats as the firing rate.  Cells
                                    that were not registered on a given day are
                                    all-NaN columns of that day's matrix.

Position and trace share the frame index (the DAQ timestamped the behavioural
and cellular imaging streams together and the paper's pipeline resamples them
onto a common clock), so no further temporal alignment is required here.

Conventions taken from the reference code (``src/utils.py``)
------------------------------------------------------------
``get_euclidean_similarity_partitioned`` orients a geometry for comparison with
the rate maps as ``np.flipud(env_mat).T.ravel()``; ``get_rate_maps`` indexes the
maps as ``[x_bin, y_bin]`` with ``x``/``y`` taken from columns 0/1 of
``position``.  Composing the two makes the flat index stored in ``blocked``

    flat = 3 * y_bin + x_bin .

This was verified empirically: binning every frame of all 207 sessions onto a
fixed 25 cm grid puts essentially zero occupancy (< 0.01 % of frames overall,
attributable to tracking jitter at the partition walls) in the bins that
``blocked`` marks as walled off.  The same flat index is therefore used for both
the geometry input and the position output, so input dimension *i* says whether
output class *i* was reachable at all.
"""

import os
import pickle

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "converted_data.pkl")

ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

FPS = 30                 # acquisition rate of both imaging streams (Hz)
ARENA_SIZE = 75.0        # side length of the full square arena (cm)
N_SPATIAL_BINS = 3       # 3 x 3 partitioning of the arena, as in the paper

# Temporal binning.  The paper's own position decoder (``fit_decoder`` /
# ``test_decoder`` in src/utils.py) smooths the traces with a Gaussian of
# sigma = temporal_bin_size frames and then average-pools over the same number
# of frames.  That recipe is kept, but with a wider bin: the binarised
# transients are extremely sparse (~0.16 events/s/cell), so 100 ms bins leave
# ~1.6 % of bins non-zero, and the full dataset at 100 ms would be ~6.7 GB.
# 500 ms integrates enough events to estimate a rate while still resolving
# position well below the 25 cm scale of a spatial bin (the median running
# speed is ~5 cm/s, i.e. ~2.5 cm per bin).
BIN_FRAMES = 15                                  # 15 frames @ 30 Hz = 500 ms
BIN_SIZE_MS = 1000.0 * BIN_FRAMES / FPS
SMOOTH_SIGMA_FRAMES = BIN_FRAMES                 # sigma = bin width, as in the paper

TRIAL_SECONDS = 60.0                             # 1-minute trials, as specified
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / BIN_FRAMES))   # 120 bins per trial


def bin_name(flat_idx):
    """Human readable name of the 3x3 partition with flat index ``3*y + x``."""
    return f"x{flat_idx % N_SPATIAL_BINS}y{flat_idx // N_SPATIAL_BINS}"


BIN_NAMES = [bin_name(i) for i in range(N_SPATIAL_BINS ** 2)]


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #

def _read_string(f, ref):
    """Decode a MATLAB char array stored behind an HDF5 object reference."""
    return "".join(chr(c) for c in f[ref][:].flatten())


def load_session_list(f):
    """Return (env_names, blocked_flat_indices) for every day of one animal."""
    envs = [_read_string(f, ref) for ref in f["envs"][0]]
    blocked = []
    for ref in f["blocked"][0]:
        idx = f[ref][:].ravel().astype(int)
        blocked.append(idx[idx >= 0])   # 'square' is stored as -1 (nothing blocked)
    return envs, blocked


def geometry_vector(blocked_flat):
    """9-dim binary vector, 1 where a 3x3 partition is walled off."""
    geo = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    geo[blocked_flat] = 1.0
    return geo


# --------------------------------------------------------------------------- #
# Per-session processing
# --------------------------------------------------------------------------- #

def process_position(position):
    """Discretise head position into the 3x3 partition grid, one label per frame.

    The arena is a fixed 75 x 75 cm square and ``position`` is already expressed
    in centimetres in that frame, so the partition boundaries are the physical
    ones (25 cm, 50 cm) rather than a per-session rescaling of the observed
    range.  This keeps the labels comparable across sessions and makes them
    agree with the geometry stored in ``blocked``.
    """
    bins = np.floor(position / (ARENA_SIZE / N_SPATIAL_BINS)).astype(int)
    bins = np.clip(bins, 0, N_SPATIAL_BINS - 1)
    return bins[:, 1] * N_SPATIAL_BINS + bins[:, 0]      # flat = 3*y + x


def bin_labels(frame_labels, n_bins):
    """Majority label of each ``BIN_FRAMES``-frame window.

    The mode is used rather than the mean position: averaging coordinates across
    a window can place the animal inside a partition it never entered (e.g. the
    walled-off centre of the 'o' geometry), whereas the mode is always a bin the
    animal actually occupied.
    """
    chunks = frame_labels[:n_bins * BIN_FRAMES].reshape(n_bins, BIN_FRAMES)
    counts = np.zeros((n_bins, N_SPATIAL_BINS ** 2), dtype=np.int32)
    for k in range(N_SPATIAL_BINS ** 2):
        counts[:, k] = (chunks == k).sum(axis=1)
    return counts.argmax(axis=1).astype(np.int64)


def bin_traces(trace, n_bins):
    """Smooth and average-pool the binarised transients into firing rates (Hz)."""
    smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA_FRAMES, axis=0)
    chunks = smoothed[:n_bins * BIN_FRAMES].reshape(n_bins, BIN_FRAMES, trace.shape[1])
    return (chunks.mean(axis=1) * FPS).astype(np.float32)   # (n_bins, n_cells)


def process_animal(path, animal):
    """Convert every session of one animal into per-trial arrays."""
    sessions = []
    with h5py.File(path, "r") as f:
        envs, blocked = load_session_list(f)
        for day, (env, blk) in enumerate(zip(envs, blocked)):
            position = f[f["position"][day, 0]][:]
            trace = f[f["trace"][day, 0]][:]

            # Keep the cells registered on this day.  Unregistered cells are
            # stored as all-NaN columns; no further neuron curation is applied,
            # matching the paper, whose headline counts (5,413 neurons,
            # 69,744 rate maps over 207 sessions) are exactly the registered
            # cell-days of these files.
            registered = ~np.all(np.isnan(trace), axis=0)
            trace = trace[:, registered]
            assert not np.isnan(trace).any(), f"{animal} day {day}: partial NaN trace"
            assert not np.isnan(position).any(), f"{animal} day {day}: NaN position"

            n_frames = min(trace.shape[0], position.shape[0])
            n_bins = n_frames // BIN_FRAMES

            rates = bin_traces(trace, n_bins)                      # (n_bins, n_cells)
            labels = bin_labels(process_position(position), n_bins)  # (n_bins,)

            # Split the continuous recording into consecutive, non-overlapping
            # 1-minute trials; the trailing fragment (< 1 min) is dropped so that
            # every trial covers the same amount of time.
            n_trials = n_bins // TRIAL_BINS
            geo = geometry_vector(blk)

            neural, inputs, outputs = [], [], []
            for t in range(n_trials):
                sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
                neural.append(np.ascontiguousarray(rates[sl].T))    # (n_cells, T)
                inputs.append(geo.copy())                           # (9,) static
                outputs.append(labels[sl][np.newaxis, :])           # (1, T)

            sessions.append({
                "neural": neural,
                "input": inputs,
                "output": outputs,
                "n_neurons": int(registered.sum()),
                "info": {
                    "animal": animal,
                    "day": day,
                    "environment": env,
                    "blocked_bins": [BIN_NAMES[i] for i in blk],
                    "sequence": day // len(set(envs)),   # 10 geometries per sequence
                    "n_neurons": int(registered.sum()),
                    "n_trials": n_trials,
                },
            })
            print(f"  {animal} day {day:2d} ({env:>10s}): "
                  f"{registered.sum():4d} neurons, {n_trials} trials")
    return sessions


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    data = {
        "neural": [], "input": [], "output": [],
        "subjects": ANIMALS, "subject_idx": [],
        "brain_regions": ["CA1"], "brain_region_idx": [],
        "input_names": [f"blocked_{n}" for n in BIN_NAMES],
        "output_names": ["position_bin"],
        "output_values": [BIN_NAMES],
    }
    session_info = []

    for a, animal in enumerate(ANIMALS):
        path = os.path.join(DATA_DIR, f"{animal}.mat")
        print(f"Processing {animal} ...")
        for session in process_animal(path, animal):
            data["neural"].append(session["neural"])
            data["input"].append(session["input"])
            data["output"].append(session["output"])
            data["subject_idx"].append(a)
            data["brain_region_idx"].append(np.zeros(session["n_neurons"], dtype=np.int64))
            session_info.append(session["info"])

    data["subject_idx"] = np.array(data["subject_idx"], dtype=np.int64)
    data["metadata"] = {
        "task_description":
            "Mice freely foraged for 40 min/day in a 75 x 75 cm arena partitioned "
            "into a 3 x 3 grid, with a different subset of the 9 partitions walled "
            "off each day (10 geometries, each repeated up to 3 times). CA1 "
            "populations were recorded with miniscope calcium imaging. The decoder "
            "receives the binarised calcium-transient rates of the CA1 population "
            "plus the geometry of the environment (which of the 9 partitions are "
            "blocked) and predicts which of the 9 partitions the mouse occupies.",
        "time_bin_size": BIN_SIZE_MS,
        "temporal_alignment_event":
            "start of the trial, i.e. of a consecutive non-overlapping 1-minute "
            "window of the continuous free-exploration session",
        "off_start": 0.0,
        "off_end": TRIAL_SECONDS,
        "neural_signal":
            "Binarised rising phase of the calcium transients (the paper's firing "
            "rate), smoothed with a Gaussian of sigma = 500 ms and averaged within "
            "each 500 ms bin, expressed in Hz.",
        "input_description":
            "Binary indicator per 3 x 3 partition, 1 = walled off in this session, "
            "0 = accessible. Constant within a session; dimension i corresponds to "
            "output class i.",
        "output_description":
            "Index of the occupied 3 x 3 partition, flat index 3*y + x with x, y "
            "the 25 cm bins of the position coordinates; the majority bin over "
            "each 500 ms window.",
        "exclusions":
            "None beyond dropping, per session, the cells not registered that day "
            "and the trailing < 1 min fragment of the recording. No velocity "
            "threshold is applied: unlike the paper's Bayesian decoding analysis, "
            "which scores decoding error during locomotion only, the position "
            "label here is defined at every time bin and every bin is scored.",
        "sampling_rate_hz": FPS,
        "arena_size_cm": ARENA_SIZE,
        "session_info": session_info,
    }

    n_trials = sum(len(s) for s in data["neural"])
    n_timepoints = sum(t.shape[1] for s in data["neural"] for t in s)
    print(f"\n{len(data['neural'])} sessions, {n_trials} trials, "
          f"{n_timepoints} timepoints, "
          f"{sum(s[0].shape[0] for s in data['neural'])} neuron-sessions")

    with open(OUT_FILE, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
