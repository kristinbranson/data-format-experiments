#!/usr/bin/env python3
"""Convert the Hasnain/Birnbaum two-context ALM data for neural decoding.

The conversion follows the supplied MATLAB analysis code where it applies:

* the two-context cohort and ALM probe numbers come from the Figure 8 scripts;
* trials with early licks or optogenetic stimulation are excluded;
* clusters tagged garbage/gabrga/noisy/real? are excluded, followed by the
  paper's >1 Hz firing-rate criterion;
* spikes are aligned to ``bp.ev.goCue``, binned at 10 ms, and smoothed with
  the repository's 15-bin causal Gaussian kernel and reflect boundary mode;
* camera and motion-energy streams use the repository's per-session video
  offset before interpolation onto the neural time axis.

The requested ignore outcome is deliberately retained (unlike analyses in the
paper that only compare hits/misses).  Trial-level labels are repeated in time
so they can coexist with the requested time-varying movement outputs.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter


DATA_DIR = Path(__file__).resolve().parent / "data" / "Ephys_Behavior"
OUTPUT_PATH = Path(__file__).resolve().parent / "converted_data.pkl"

# ALM probe selection copied from code/DataLoadingScripts/Recording and video.
# These are the sessions loaded by both supplied Figure 8 two-context scripts.
ALM_PROBE = {
    "JEB6": 2,
    "JEB7": 1,
    "EKH1": 2,
    "EKH3": 2,
    "JGR2": 1,
    "JGR3": 1,
    "JEB19": 1,
}

TMIN = -2.5
TMAX = 2.5
DT = 0.01
SMOOTH_BINS = 15
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
N_TIME = TIME.size
REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}


def matlab_char(h5: h5py.File, ref: h5py.Reference) -> str:
    """Decode a MATLAB v7.3 char array referenced from a cell/struct."""
    values = np.asarray(h5[ref]).ravel()
    return "".join(chr(int(v)) for v in values if int(v) != 0).strip()


def referenced_array(h5: h5py.File, ref: h5py.Reference) -> np.ndarray:
    return np.asarray(h5[ref])


def matlab_cellstr(h5: h5py.File, ref: h5py.Reference) -> list[str]:
    cell = h5[ref]
    return [matlab_char(h5, item) for item in np.asarray(cell).ravel()]


def session_files(data_dir: Path) -> list[tuple[Path, str, str]]:
    sessions = []
    pattern = re.compile(r"data_structure_([^_]+)_(\d{4}-\d{2}-\d{2})\.mat$")
    for path in sorted(data_dir.glob("data_structure_*.mat")):
        match = pattern.match(path.name)
        if match and match.group(1) in ALM_PROBE:
            sessions.append((path, match.group(1), match.group(2)))
    if len(sessions) != 12:
        raise RuntimeError(f"Expected 12 two-context sessions, found {len(sessions)}")
    return sessions


def gaussian_causal_kernel(n: int = SMOOTH_BINS) -> np.ndarray:
    """Reproduce MATLAB gausswin(n), then the causalization in mySmooth.m."""
    # MATLAB gausswin's default alpha is 2.5.
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2
    kernel = np.exp(-0.5 * (2.5 * x / ((n - 1) / 2)) ** 2)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    # With conv(..., 'same'), the center coefficient acts on the current bin.
    return kernel[n // 2 :].astype(np.float32)


CAUSAL_KERNEL = gaussian_causal_kernel()


def smooth_rates(counts: np.ndarray) -> np.ndarray:
    """Apply mySmooth(..., 15, 'reflect') along the final dimension."""
    # mySmooth prepends the first N samples in their original order.  Although
    # called "reflect" in the source, this exact behavior is what we reproduce.
    padded = np.concatenate((counts[..., :SMOOTH_BINS], counts), axis=-1)
    filtered = lfilter(CAUSAL_KERNEL, [1.0], padded, axis=-1)
    return (filtered[..., SMOOTH_BINS:] / DT).astype(np.float32)


def histogram_unit(
    aligned_time: np.ndarray,
    spike_trial: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    """Bin one unit's aligned spikes as trial x time spike counts."""
    result = np.zeros((n_trials, N_TIME), dtype=np.float32)
    bins = np.floor((aligned_time - TMIN) / DT).astype(np.int64)
    valid = (
        (spike_trial >= 0)
        & (spike_trial < n_trials)
        & (bins >= 0)
        & (bins < N_TIME)
    )
    np.add.at(result, (spike_trial[valid], bins[valid]), 1.0)
    return result


def unit_mean_rate(
    aligned_time: np.ndarray,
    spike_trial: np.ndarray,
    condition_masks: list[np.ndarray],
) -> float:
    """Match removeLowFRClusters using the Figure 8 condition PSTHs."""
    condition_means = []
    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    for mask in condition_masks:
        n_trials = int(mask.sum())
        selected = mask[spike_trial] if spike_trial.size else np.zeros(0, bool)
        hist = np.histogram(aligned_time[selected], bins=edges)[0].astype(np.float32)
        if n_trials:
            psth = smooth_rates(hist[None, :])[0] / n_trials
        else:
            psth = np.zeros(N_TIME, dtype=np.float32)
        condition_means.append(float(psth.mean()))
    return float(np.mean(condition_means))


def nearest_fill(values: np.ndarray) -> np.ndarray:
    """One-dimensional equivalent of MATLAB fillmissing(..., 'nearest')."""
    values = np.asarray(values, dtype=np.float64).copy()
    good = np.flatnonzero(np.isfinite(values))
    if not good.size:
        return values
    bad = np.flatnonzero(~np.isfinite(values))
    if not bad.size:
        return values
    positions = np.searchsorted(good, bad)
    left_pos = np.maximum(positions - 1, 0)
    right_pos = np.minimum(positions, good.size - 1)
    left = good[left_pos]
    right = good[right_pos]
    choose_right = np.abs(right - bad) < np.abs(bad - left)
    nearest = np.where(choose_right, right, left)
    values[bad] = values[nearest]
    return values


def interp_preserving_missing(
    source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    """Linear interpolation that preserves NaN intervals and extrapolates NaN."""
    source_time = np.asarray(source_time, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    valid_time = np.isfinite(source_time)
    source_time, values = source_time[valid_time], values[valid_time]
    if source_time.size < 2:
        return np.full(target_time.shape, np.nan, dtype=np.float64)
    order = np.argsort(source_time)
    source_time, values = source_time[order], values[order]
    return np.interp(target_time, source_time, values, left=np.nan, right=np.nan)


def video_offset(h5: h5py.File) -> float:
    bit_start = np.asarray(h5["obj/bp/ev/bitStart"]).ravel()
    video_bit_start = np.asarray(h5["obj/sglx/bitcode/bitstart"]).ravel()
    fs = float(np.asarray(h5["obj/sglx/fs"]).squeeze())
    return float(np.nanmedian(video_bit_start / fs) - np.nanmedian(bit_start))


def find_feature_indices(
    h5: h5py.File, trajectory_group: h5py.Group, wanted: str
) -> int:
    for ref in np.asarray(trajectory_group["featNames"]).ravel():
        try:
            names = matlab_cellstr(h5, ref)
        except (KeyError, TypeError, ValueError):
            continue
        if wanted in names:
            return names.index(wanted)
    raise RuntimeError(f"Could not find DLC feature {wanted!r}")


def trajectory_signal(
    h5: h5py.File,
    trajectory_group: h5py.Group,
    trial: int,
    feature_index: int,
    go_cue: float,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return aligned x/y positions and frame times for one feature/trial."""
    dropped = referenced_array(h5, trajectory_group["NdroppedFrames"][trial, 0]).squeeze()
    if np.size(dropped) == 0 or not np.all(np.isfinite(dropped)):
        return None
    frame_time = referenced_array(h5, trajectory_group["frameTimes"][trial, 0]).ravel()
    raw = referenced_array(h5, trajectory_group["ts"][trial, 0])
    if raw.ndim != 3 or raw.shape[1] < 2 or feature_index >= raw.shape[0]:
        return None
    # HDF5 dimensions are reversed relative to MATLAB: feature x xyz x frame.
    x_raw = raw[feature_index, 0, :]
    y_raw = raw[feature_index, 1, :]
    n = min(frame_time.size, x_raw.size, y_raw.size)
    aligned_frame_time = frame_time[:n] - offset - go_cue
    x = interp_preserving_missing(aligned_frame_time, x_raw[:n], TIME)
    y = interp_preserving_missing(aligned_frame_time, y_raw[:n], TIME)
    return x, y, aligned_frame_time


def speed_from_position(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    speed = np.hypot(np.gradient(x), np.gradient(y))
    speed[~(np.isfinite(x) & np.isfinite(y))] = np.nan
    return speed


def load_motion_energy(path: Path) -> list[np.ndarray]:
    mat = loadmat(path, simplify_cells=True)
    raw = mat["me"]["data"]
    if isinstance(raw, dict):
        raw = raw["data"]
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        return [np.asarray(item, dtype=np.float64).ravel() for item in raw.ravel()]
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        return [np.asarray(raw[:, i], dtype=np.float64) for i in range(raw.shape[1])]
    raise RuntimeError(f"Unsupported motion-energy layout in {path}")


def categorize_session_signal(signals: list[np.ndarray]) -> tuple[list[np.ndarray], float | None]:
    finite_parts = [x[np.isfinite(x)] for x in signals if np.any(np.isfinite(x))]
    if not finite_parts:
        return [np.full(N_TIME, 2, dtype=np.int64) for _ in signals], None
    threshold = float(np.percentile(np.concatenate(finite_parts), 50))
    categorical = []
    for signal in signals:
        out = np.full(N_TIME, 2, dtype=np.int64)
        visible = np.isfinite(signal)
        out[visible] = (signal[visible] >= threshold).astype(np.int64)
        categorical.append(out)
    return categorical, threshold


def process_session(path: Path, animal: str, date: str) -> dict:
    motion_path = path.with_name(path.name.replace("data_structure_", "motionEnergy_"))
    motion_trials = load_motion_energy(motion_path)

    with h5py.File(path, "r") as h5:
        bp = h5["obj/bp"]
        go = np.asarray(bp["ev/goCue"]).ravel().astype(np.float64)
        n_trials = go.size
        hit = np.asarray(bp["hit"]).ravel().astype(bool)
        miss = np.asarray(bp["miss"]).ravel().astype(bool)
        ignore = np.asarray(bp["no"]).ravel().astype(bool)
        early = np.asarray(bp["early"]).ravel().astype(bool)
        autowater = np.asarray(bp["autowater"]).ravel().astype(bool)
        right_target = np.asarray(bp["R"]).ravel().astype(bool)
        stim = np.asarray(bp["stim/enable"]).ravel().astype(bool)

        # Paper analyses use control trials and omit early licks.  Ignore trials
        # remain because ignore is a required decoder output in this task.
        trial_mask = ~early & ~stim & np.isfinite(go) & (hit | miss | ignore)
        kept_trials = np.flatnonzero(trial_mask)
        if kept_trials.size < 2:
            raise RuntimeError(f"{animal} {date} has fewer than two usable trials")

        # Figure 8 condition definitions used by the source low-FR filter.
        condition_masks = [
            hit | miss | ignore,
            hit & ~autowater,
            hit & autowater,
            miss & ~autowater,
            miss & autowater,
            hit & ~autowater & ~early,
            hit & autowater & ~early,
        ]

        probe_number = ALM_PROBE[animal]
        cluster_group = h5[h5["obj/clu"][probe_number - 1, 0]]
        selected_units: list[tuple[np.ndarray, np.ndarray, str, float]] = []
        for unit in range(cluster_group["quality"].shape[0]):
            quality = matlab_char(h5, cluster_group["quality"][unit, 0])
            # Deliberately case-sensitive, matching findClusters.m/isMember.
            if quality in REJECTED_QUALITIES:
                continue
            spike_trial = referenced_array(h5, cluster_group["trial"][unit, 0]).ravel().astype(np.int64) - 1
            trial_time = referenced_array(h5, cluster_group["trialtm"][unit, 0]).ravel().astype(np.float64)
            valid = (spike_trial >= 0) & (spike_trial < n_trials)
            spike_trial, trial_time = spike_trial[valid], trial_time[valid]
            aligned_time = trial_time - go[spike_trial]
            mean_rate = unit_mean_rate(aligned_time, spike_trial, condition_masks)
            if mean_rate > 1.0:
                selected_units.append((aligned_time, spike_trial, quality, mean_rate))

        if len(selected_units) < 10:
            raise RuntimeError(f"{animal} {date} has only {len(selected_units)} curated units")

        original_to_kept = np.full(n_trials, -1, dtype=np.int64)
        original_to_kept[kept_trials] = np.arange(kept_trials.size)
        counts = np.zeros((kept_trials.size, len(selected_units), N_TIME), dtype=np.float32)
        for unit_index, (aligned_time, spike_trial, _, _) in enumerate(selected_units):
            mapped = original_to_kept[spike_trial]
            use = mapped >= 0
            unit_counts = histogram_unit(aligned_time[use], mapped[use], kept_trials.size)
            counts[:, unit_index, :] = unit_counts
        rates = smooth_rates(counts)

        trajectory_refs = np.asarray(h5["obj/traj"]).ravel()
        side = h5[trajectory_refs[0]]
        bottom = h5[trajectory_refs[1]]
        tongue_index = find_feature_indices(h5, side, "tongue")
        paw_index = find_feature_indices(h5, bottom, "top_paw")
        offset = video_offset(h5)

        tongue_velocity: list[np.ndarray] = []
        paw_velocity: list[np.ndarray] = []
        motion_energy: list[np.ndarray] = []
        for trial in kept_trials:
            tongue = trajectory_signal(h5, side, int(trial), tongue_index, go[trial], offset)
            paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
            tongue_velocity.append(
                np.full(N_TIME, np.nan) if tongue is None else speed_from_position(tongue[0], tongue[1])
            )
            paw_velocity.append(
                np.full(N_TIME, np.nan) if paw is None else speed_from_position(paw[0], paw[1])
            )

            if tongue is None or trial >= len(motion_trials):
                motion_energy.append(np.full(N_TIME, np.nan))
            else:
                raw_me = motion_trials[int(trial)]
                # Motion energy and the side camera have one value per frame.
                n = min(raw_me.size, tongue[2].size)
                if n < 2 or not np.any(np.isfinite(raw_me[:n])):
                    motion_energy.append(np.full(N_TIME, np.nan))
                else:
                    me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
                    motion_energy.append(nearest_fill(me))

    tongue_cat, tongue_threshold = categorize_session_signal(tongue_velocity)
    paw_cat, paw_threshold = categorize_session_signal(paw_velocity)
    motion_cat, motion_threshold = categorize_session_signal(motion_energy)

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    time_input = TIME.astype(np.float32)[None, :]
    for local_trial, original_trial in enumerate(kept_trials):
        neural_trials.append(np.ascontiguousarray(rates[local_trial], dtype=np.float32))
        input_trials.append(time_input.copy())

        if ignore[original_trial]:
            lick_direction = 2  # none
            outcome = 2
        elif hit[original_trial]:
            lick_direction = 1 if right_target[original_trial] else 0
            outcome = 1
        else:
            # R/L identifies instructed/reward side, so an incorrect response is
            # the opposite side and is the animal's actual lick direction.
            lick_direction = 0 if right_target[original_trial] else 1
            outcome = 0
        context = 0 if autowater[original_trial] else 1

        output = np.empty((6, N_TIME), dtype=np.int64)
        output[0] = lick_direction
        output[1] = context
        output[2] = outcome
        output[3] = tongue_cat[local_trial]
        output[4] = paw_cat[local_trial]
        output[5] = motion_cat[local_trial]
        output_trials.append(output)

    return {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "n_units": len(selected_units),
        "n_original_trials": n_trials,
        "n_trials": int(kept_trials.size),
        "probe": probe_number,
        "video_offset_s": offset,
        "tongue_threshold": tongue_threshold,
        "paw_threshold": paw_threshold,
        "motion_threshold": motion_threshold,
        "qualities": [item[2] for item in selected_units],
        "mean_rates_hz": [float(item[3]) for item in selected_units],
    }


def convert(data_dir: Path = DATA_DIR, output_path: Path = OUTPUT_PATH) -> dict:
    files = session_files(data_dir)
    subjects = sorted({animal for _, animal, _ in files})
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}

    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for number, (path, animal, date) in enumerate(files, start=1):
        print(f"[{number:02d}/{len(files)}] {animal} {date}", flush=True)
        result = process_session(path, animal, date)
        neural.append(result.pop("neural"))
        inputs.append(result.pop("input"))
        outputs.append(result.pop("output"))
        subject_idx.append(subject_lookup[animal])
        brain_region_idx.append(np.zeros(result["n_units"], dtype=np.int64))
        result.pop("qualities")
        result.pop("mean_rates_hz")
        session_info.append({"subject": animal, "date": date, "source_file": path.name, **result})
        print(f"    {result['n_trials']} trials, {result['n_units']} ALM units", flush=True)

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["ALM"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_go_cue_s"],
        "output_names": [
            "lick_direction",
            "behavioral_context",
            "outcome",
            "tongue_velocity",
            "paw_velocity",
            "motion_energy",
        ],
        "output_values": [
            ["left", "right", "none"],
            ["WC", "DR"],
            ["incorrect", "correct", "ignore"],
            ["below_session_median", "at_or_above_session_median", "not_visible"],
            ["below_session_median", "at_or_above_session_median", "not_visible"],
            ["below_session_median", "at_or_above_session_median", "no_video"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice alternated between delayed-response (DR) and water-cued "
                "(WC) directional licking; labels include actual lick direction, context, "
                "outcome, and median-discretized video-derived movement variables."
            ),
            "time_bin_size": DT * 1000,
            "temporal_alignment_event": (
                "Bpod goCue onset (auditory go cue in DR; corresponding water-delivery "
                "event in WC), using obj.bp.ev.goCue"
            ),
            "off_start": TMIN,
            "off_end": TMAX,
            "neural_representation": "10 ms firing rates smoothed with the source 15-bin causal Gaussian kernel",
            "trial_filter": "exclude early-lick and photostimulation trials; retain hit, miss, and ignore trials",
            "unit_filter": (
                "manifest-selected ALM probe; exclude exact quality labels garbage, gabrga, "
                "noisy, real?; retain units with source-style mean firing rate >1 Hz"
            ),
            "video_processing": (
                "DLC tongue (side camera) and top-paw (bottom camera) Euclidean frame-to-frame "
                "velocity, aligned with the source video offset; per-session visible-sample medians"
            ),
            "cohort": "12-session two-context electrophysiology cohort loaded by the supplied Figure 8 scripts",
            "session_info": session_info,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {output_path} ({output_path.stat().st_size / 2**20:.1f} MiB)")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    convert(args.data_dir, args.output)


if __name__ == "__main__":
    main()
