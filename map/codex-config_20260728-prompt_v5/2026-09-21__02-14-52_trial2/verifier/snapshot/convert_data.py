#!/usr/bin/env python3
"""Convert MAP NWB sessions into decoder-ready trial tensors.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from collections import Counter

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = "/app/data"
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES_REL = np.linspace(WINDOW_START, WINDOW_END, NBINS + 1, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2

TONGUE_LIKELIHOOD_THRESHOLD = 0.9
TONGUE_VELOCITY_SIGMA = 5.0
RESPONSE_WINDOW = 1.5

INPUT_NAMES = ["time_from_tone_onset_s", "photostimulation_on"]
OUTPUT_NAMES = ["choice", "outcome", "early_lick", "tongue_y"]
OUTPUT_VALUES = [
    ["left", "right", "no lick"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["<40th percentile", "40th-60th percentile", ">60th percentile", "not visible"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MAP NWB files to decoder format.")
    parser.add_argument("outpicklefile", type=str, help="Where to save the converted pickle.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all curated sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 curated sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def decode_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "decode"):
        return value.decode()
    return str(value)


def decode_vector(values) -> list[str]:
    return [decode_scalar(v) for v in values]


def ragged_rows(flat: np.ndarray, index: np.ndarray, row: int) -> np.ndarray:
    start = 0 if row == 0 else int(index[row - 1])
    end = int(index[row])
    return flat[start:end]


def time_key(value: float) -> int:
    """Stable key for matching NWB trial intervals stored at 0.1 ms precision."""
    return int(round(float(value) * 10000.0))


def map_obs_intervals_to_trial_indices(
    trial_start: np.ndarray,
    trial_stop: np.ndarray,
    obs_intervals: np.ndarray,
) -> np.ndarray:
    trial_lookup = {
        (time_key(start), time_key(stop)): idx
        for idx, (start, stop) in enumerate(zip(trial_start, trial_stop, strict=False))
    }
    mapped = []
    for start, stop in obs_intervals:
        key = (time_key(start), time_key(stop))
        if key not in trial_lookup:
            raise ValueError(f"Could not map obs interval {(start, stop)} onto trial table.")
        mapped.append(trial_lookup[key])
    return np.asarray(mapped, dtype=np.int64)


def list_session_files() -> list[str]:
    files = []
    for dirpath, _, filenames in os.walk(DATA_ROOT):
        for name in filenames:
            if name.endswith(".nwb"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def session_has_good_units(path: str) -> bool:
    with h5py.File(path, "r") as h5:
        cls = decode_vector(h5["units"]["classification"][:])
    return "good" in cls


def discover_curated_sessions() -> list[str]:
    files = list_session_files()
    curated = [path for path in files if session_has_good_units(path)]
    return curated


def parse_target_region(location_json: str) -> str:
    try:
        return json.loads(location_json).get("brain_regions", "").strip()
    except Exception:
        return location_json.strip()


def get_session_id(path: str) -> str:
    return os.path.basename(path).replace("_behavior+ecephys+ogen.nwb", "").replace("_behavior+ecephys.nwb", "")


def get_subject_id(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def velocity_outlier_mask(xy: np.ndarray, dt: float) -> np.ndarray:
    """Flag tracking outliers using a 5-sigma velocity rule."""
    if len(xy) < 3:
        return np.zeros(len(xy), dtype=bool)
    diffs = np.diff(xy, axis=0)
    speed = np.linalg.norm(diffs, axis=1) / max(dt, 1e-9)
    mu = float(np.mean(speed))
    sigma = float(np.std(speed))
    thresh = mu + TONGUE_VELOCITY_SIGMA * sigma
    bad = np.zeros(len(xy), dtype=bool)
    if sigma == 0.0:
        return bad
    flagged = np.where(speed > thresh)[0]
    bad[flagged] = True
    bad[flagged + 1] = True
    return bad


def derive_choice_per_trial(
    go_times: np.ndarray,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> np.ndarray:
    """Choice from the first lick after go cue within the 1.5 s response window."""
    choice = np.full(len(go_times), 2, dtype=np.int64)  # default no lick
    for i, go in enumerate(go_times):
        end = go + RESPONSE_WINDOW
        left = left_lick_times[(left_lick_times >= go) & (left_lick_times < end)]
        right = right_lick_times[(right_lick_times >= go) & (right_lick_times < end)]
        if len(left) == 0 and len(right) == 0:
            continue
        if len(left) > 0 and (len(right) == 0 or left[0] < right[0]):
            choice[i] = 0
        elif len(right) > 0 and (len(left) == 0 or right[0] < left[0]):
            choice[i] = 1
    return choice


def last_event_before(times: np.ndarray, lo: float, hi: float) -> float | None:
    """Latest event in [lo, hi], else None."""
    idx = np.searchsorted(times, hi, side="right") - 1
    if idx < 0:
        return None
    t = float(times[idx])
    if t < lo or t > hi:
        return None
    return t


def build_tongue_session_stats(
    timestamps: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xy = np.asarray(data[:, :2], dtype=np.float64)
    likelihood = np.asarray(data[:, 2], dtype=np.float64)
    dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0034
    outliers = velocity_outlier_mask(xy, dt)
    visible = np.isfinite(xy[:, 1]) & np.isfinite(likelihood) & (~outliers) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    visible_y = xy[visible, 1]
    if len(visible_y) == 0:
        raise ValueError("No visible tongue samples remain after filtering.")
    q40, q60 = np.quantile(visible_y, [0.4, 0.6])
    return xy, likelihood, visible, float(q40), float(q60)


def discretize_tongue_bins(
    frame_timestamps: np.ndarray,
    tongue_xy: np.ndarray,
    tongue_likelihood: np.ndarray,
    tongue_visible_mask: np.ndarray,
    q40: float,
    q60: float,
    trial_edges_abs: np.ndarray,
) -> np.ndarray:
    """Reference-style sample-and-hold: use the last frame within each bin."""
    out = np.full(NBINS, 3, dtype=np.int64)
    frame_idx = np.searchsorted(frame_timestamps, trial_edges_abs)
    for b in range(NBINS):
        start = int(frame_idx[b])
        end = int(frame_idx[b + 1])
        if end <= start:
            continue
        last = end - 1
        if not tongue_visible_mask[last]:
            continue
        y = float(tongue_xy[last, 1])
        if y < q40:
            out[b] = 0
        elif y <= q60:
            out[b] = 1
        else:
            out[b] = 2
    return out


def bin_spike_rates(spike_times_abs: np.ndarray, trial_edges_abs: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(spike_times_abs, trial_edges_abs, side="left")
    counts = np.diff(idx)
    return counts.astype(np.float32) / BIN_SIZE


def plot_processing_figure(
    session_id: str,
    session_summary: dict,
    example_trial: dict,
    outpath: str,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    ax = axes.ravel()

    # Neural heatmap
    neural = example_trial["neural"]
    nplot = min(40, neural.shape[0])
    if nplot > 0:
        ax[0].imshow(neural[:nplot], aspect="auto", interpolation="nearest", cmap="viridis")
    ax[0].set_title(f"{session_id}: example neural trial")
    ax[0].set_ylabel("Good units")
    ax[0].set_xlabel("50 ms bins")

    # Input traces
    ax[1].plot(BIN_CENTERS_REL, example_trial["input"][0], label="time from tone onset (s)")
    ax[1].plot(BIN_CENTERS_REL, example_trial["input"][1], label="photostim on")
    ax[1].axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax[1].set_title("Inputs")
    ax[1].legend(loc="best", fontsize=8)

    # Raw tongue signal around example trial
    trial_mask = (
        (session_summary["tongue_timestamps"] >= example_trial["go_time"] + WINDOW_START)
        & (session_summary["tongue_timestamps"] < example_trial["go_time"] + WINDOW_END)
    )
    rel_t = session_summary["tongue_timestamps"][trial_mask] - example_trial["go_time"]
    y = session_summary["tongue_xy"][trial_mask, 1]
    like = session_summary["tongue_likelihood"][trial_mask]
    vis = session_summary["tongue_visible"][trial_mask]
    ax[2].plot(rel_t, y, color="0.8", linewidth=1, label="all y")
    if np.any(vis):
        ax[2].scatter(rel_t[vis], y[vis], s=6, c="tab:blue", label="visible", alpha=0.7)
    if np.any(~vis):
        ax[2].scatter(rel_t[~vis], y[~vis], s=6, c="tab:red", label="not visible", alpha=0.5)
    ax[2].axhline(session_summary["tongue_q40"], color="tab:green", linestyle="--", linewidth=1)
    ax[2].axhline(session_summary["tongue_q60"], color="tab:orange", linestyle="--", linewidth=1)
    ax[2].set_title("Tongue y / visibility")
    ax[2].set_xlabel("Time from go (s)")
    ax[2].legend(loc="best", fontsize=8)

    # Binned tongue classes
    ax[3].step(BIN_CENTERS_REL, example_trial["output"][3], where="mid")
    ax[3].set_title("Binned tongue_y classes")
    ax[3].set_ylim(-0.2, 3.2)
    ax[3].set_xlabel("Time from go (s)")

    # Sample onset / go timing summary
    sample_rel = np.asarray(session_summary["sample_onsets_rel"], dtype=np.float64)
    ax[4].hist(sample_rel, bins=30, color="tab:purple", alpha=0.8)
    ax[4].axvline(-1.85, color="k", linestyle="--", linewidth=1)
    ax[4].set_title("Final sample onset relative to go")
    ax[4].set_xlabel("Seconds")

    # Text summary
    ax[5].axis("off")
    lines = [
        f"Kept trials: {session_summary['n_kept_trials']}",
        f"Good units: {session_summary['n_good_units']}",
        f"Choice counts: {session_summary['choice_counts']}",
        f"Outcome counts: {session_summary['outcome_counts']}",
        f"Early lick counts: {session_summary['early_counts']}",
        f"Tongue visible frac: {session_summary['tongue_visible_frac']:.3f}",
    ]
    ax[5].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def process_session(path: str, make_plot: bool = False) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    session_id = get_session_id(path)
    subject = get_subject_id(path)
    print(f"[session] {session_id}: loading")

    with h5py.File(path, "r") as h5:
        # Trials
        trials = h5["intervals"]["trials"]
        trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
        trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
        outcome_text = decode_vector(trials["outcome"][:])
        early_text = decode_vector(trials["early_lick"][:])
        trial_instruction = decode_vector(trials["trial_instruction"][:])
        auto_water = np.asarray(trials["auto_water"][:], dtype=np.int64)
        free_water = np.asarray(trials["free_water"][:], dtype=np.int64)
        photo_onset_text = decode_vector(trials["photostim_onset"][:])
        photo_dur_text = decode_vector(trials["photostim_duration"][:])

        # Events
        ev = h5["acquisition"]["BehavioralEvents"]
        go_times = np.asarray(ev["go_start_times"]["timestamps"][:], dtype=np.float64)
        left_lick_times = np.asarray(ev["left_lick_times"]["timestamps"][:], dtype=np.float64)
        right_lick_times = np.asarray(ev["right_lick_times"]["timestamps"][:], dtype=np.float64)
        sample_start_times = np.asarray(ev["sample_start_times"]["timestamps"][:], dtype=np.float64)

        # Tracking
        tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
        tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
        tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
        tongue_xy, tongue_likelihood, tongue_visible_mask, tongue_q40, tongue_q60 = build_tongue_session_stats(
            tongue_timestamps,
            tongue_data,
        )

        # Units
        units = h5["units"]
        classification = decode_vector(units["classification"][:])
        anno_name = decode_vector(units["anno_name"][:])
        spike_flat = np.asarray(units["spike_times"][:], dtype=np.float64)
        spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
        obs_intervals_flat = np.asarray(units["obs_intervals"][:], dtype=np.float64)
        obs_intervals_index = np.asarray(units["obs_intervals_index"][:], dtype=np.int64)
        electrodes = np.asarray(units["electrodes"][:], dtype=np.int64)
        electrode_locations = decode_vector(h5["general"]["extracellular_ephys"]["electrodes"]["location"][:])

        good_units = [i for i, c in enumerate(classification) if c == "good"]
        if not good_units:
            raise ValueError(f"{session_id}: no classifier-good units")

        # `obs_intervals` gives the subset of behavioral trials with ephys coverage.
        coverage_obs = ragged_rows(obs_intervals_flat, obs_intervals_index, good_units[0])
        recorded_trial_idx = map_obs_intervals_to_trial_indices(trial_start, trial_stop, coverage_obs)
        recorded_trial_set = set(recorded_trial_idx.tolist())

        # Per-unit region labels
        region_labels = []
        for unit_idx in good_units:
            unit_region = anno_name[unit_idx].split(",")[0].strip()
            if not unit_region:
                target = parse_target_region(electrode_locations[int(electrodes[unit_idx])])
                unit_region = target if target else "Unknown"
            region_labels.append(unit_region)

        # Per-trial derived variables
        choice = derive_choice_per_trial(go_times, left_lick_times, right_lick_times)
        outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
        early_map = {"no early": 0, "early": 1}
        outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int64)
        early = np.array([early_map[x] for x in early_text], dtype=np.int64)

        candidate_trial_idx = []
        sample_onsets_abs = []
        candidate_sample_onsets_rel = []
        candidate_photo_intervals_abs = []
        for i in range(len(trial_start)):
            if i not in recorded_trial_set:
                continue
            if auto_water[i] == 1 or free_water[i] == 1:
                continue
            go = float(go_times[i])
            edges_abs = go + BIN_EDGES_REL
            if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]:
                continue
            sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
            if sample_start is None:
                continue

            if photo_onset_text[i] != "N/A" and photo_dur_text[i] != "N/A":
                stim_on = float(trial_start[i]) + float(photo_onset_text[i])
                stim_off = stim_on + float(photo_dur_text[i])
            else:
                stim_on = math.nan
                stim_off = math.nan

            candidate_trial_idx.append(i)
            sample_onsets_abs.append(sample_start)
            candidate_sample_onsets_rel.append(sample_start - go)
            candidate_photo_intervals_abs.append((stim_on, stim_off))

        if len(candidate_trial_idx) < 2:
            raise ValueError(f"{session_id}: fewer than 2 trials remain after filtering")

        # Session-level outputs
        neural_trials = []
        input_trials = []
        output_trials = []
        trial_keep = []
        sample_onsets_rel = []
        dropped_all_zero_neural = 0

        # Pre-slice spike times once per good unit
        unit_spike_times = [ragged_rows(spike_flat, spike_index, unit_idx) for unit_idx in good_units]

        for kept_idx, trial_idx in enumerate(candidate_trial_idx):
            go = float(go_times[trial_idx])
            edges_abs = go + BIN_EDGES_REL

            # Neural
            neural = np.empty((len(good_units), NBINS), dtype=np.float32)
            for unit_row, spikes_abs in enumerate(unit_spike_times):
                neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
            if not np.any(neural):
                dropped_all_zero_neural += 1
                continue

            # Inputs
            tone_on_rel = float(candidate_sample_onsets_rel[kept_idx])
            input_arr = np.empty((2, NBINS), dtype=np.float32)
            input_arr[0] = (BIN_CENTERS_REL - tone_on_rel).astype(np.float32)

            stim_on, stim_off = candidate_photo_intervals_abs[kept_idx]
            if math.isnan(stim_on):
                input_arr[1] = 0.0
            else:
                abs_centers = go + BIN_CENTERS_REL
                input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off)).astype(np.float32)

            # Outputs
            tongue_disc = discretize_tongue_bins(
                tongue_timestamps,
                tongue_xy,
                tongue_likelihood,
                tongue_visible_mask,
                tongue_q40,
                tongue_q60,
                edges_abs,
            )
            output_arr = np.empty((4, NBINS), dtype=np.int64)
            output_arr[0] = choice[trial_idx]
            output_arr[1] = outcome[trial_idx]
            output_arr[2] = early[trial_idx]
            output_arr[3] = tongue_disc

            trial_keep.append(trial_idx)
            sample_onsets_rel.append(candidate_sample_onsets_rel[kept_idx])
            neural_trials.append(neural)
            input_trials.append(input_arr)
            output_trials.append(output_arr)

        if len(trial_keep) < 2:
            raise ValueError(f"{session_id}: fewer than 2 trials remain after neural validation")

        region_counts = Counter(region_labels)
        session_summary = {
            "session_id": session_id,
            "subject": subject,
            "n_good_units": len(good_units),
            "n_recorded_trials": len(recorded_trial_idx),
            "n_candidate_trials": len(candidate_trial_idx),
            "n_kept_trials": len(trial_keep),
            "n_dropped_all_zero_neural": dropped_all_zero_neural,
            "choice_counts": dict(Counter(choice[trial_keep].tolist())),
            "outcome_counts": dict(Counter(outcome[trial_keep].tolist())),
            "early_counts": dict(Counter(early[trial_keep].tolist())),
            "sample_onsets_rel": sample_onsets_rel,
            "tongue_q40": tongue_q40,
            "tongue_q60": tongue_q60,
            "tongue_visible_frac": float(np.mean(tongue_visible_mask)),
            "region_counts": dict(region_counts),
            "tongue_timestamps": tongue_timestamps,
            "tongue_xy": tongue_xy,
            "tongue_likelihood": tongue_likelihood,
            "tongue_visible": tongue_visible_mask,
        }

        example_trial_idx = 0
        if make_plot:
            example = {
                "go_time": float(go_times[trial_keep[example_trial_idx]]),
                "neural": neural_trials[example_trial_idx],
                "input": input_trials[example_trial_idx],
                "output": output_trials[example_trial_idx],
            }
            outpath = f"/app/processing_{session_id}.png"
            plot_processing_figure(session_id, session_summary, example, outpath)

    elapsed = time.perf_counter() - t0
    print(
        f"[session] {session_id}: kept {len(trial_keep)} trials, "
        f"{len(good_units)} good units in {elapsed:.2f}s"
    )

    return (
        {
            "session_id": session_id,
            "subject": subject,
            "neural": neural_trials,
            "input": input_trials,
            "output": output_trials,
            "region_labels": region_labels,
        },
        session_summary,
    )


def build_dataset(session_dicts: list[dict], session_summaries: list[dict]) -> dict:
    subjects = sorted({sess["subject"] for sess in session_dicts})
    subject_to_idx = {sub: i for i, sub in enumerate(subjects)}

    all_regions = sorted({region for sess in session_dicts for region in sess["region_labels"]})
    region_to_idx = {region: i for i, region in enumerate(all_regions)}

    data = {
        "neural": [sess["neural"] for sess in session_dicts],
        "input": [sess["input"] for sess in session_dicts],
        "output": [sess["output"] for sess in session_dicts],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[sess["subject"]] for sess in session_dicts], dtype=np.int64),
        "brain_regions": all_regions,
        "brain_region_idx": [
            np.array([region_to_idx[label] for label in sess["region_labels"]], dtype=np.int64)
            for sess in session_dicts
        ],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Auditory delayed-response task aligned to go cue; decoder predicts "
                "choice, outcome, early lick, and discretized tongue y-position."
            ),
            "time_bin_size": 50.0,
            "temporal_alignment_event": "Go cue onset",
            "off_start": WINDOW_START,
            "off_end": WINDOW_END,
            "n_timepoints": NBINS,
            "bin_centers_rel_go_s": BIN_CENTERS_REL.astype(np.float32),
            "source_dataset": "Mesoscale Activity Map Dataset (NWB bundle in /app/data)",
            "source_file_format": "NWB",
            "session_filter": "sessions with at least one units.classification == good",
            "trial_filter": (
                "keep only trials covered by units.obs_intervals, then exclude auto_water and "
                "free_water; keep early, ignore, and photostim trials"
            ),
            "unit_filter": "units.classification == good",
            "tongue_visibility_rule": (
                f"Camera0_side_TongueTracking likelihood >= {TONGUE_LIKELIHOOD_THRESHOLD} "
                f"and not a {TONGUE_VELOCITY_SIGMA}-sigma velocity outlier"
            ),
            "session_info": [
                {
                    "session_id": summary["session_id"],
                    "subject": summary["subject"],
                    "n_good_units": summary["n_good_units"],
                    "n_kept_trials": summary["n_kept_trials"],
                }
                for summary in session_summaries
            ],
        },
    }
    return data


def main() -> None:
    args = parse_args()
    t_all = time.perf_counter()

    curated_sessions = discover_curated_sessions()
    print(f"[setup] curated sessions with good units: {len(curated_sessions)}")
    if args.sample:
        curated_sessions = curated_sessions[:2]
        print(f"[setup] sample mode: processing {len(curated_sessions)} sessions")
    else:
        print(f"[setup] full mode: processing {len(curated_sessions)} sessions")

    session_dicts = []
    session_summaries = []
    plot_ids = set(get_session_id(p) for p in curated_sessions[:2]) if args.show_processing else set()

    for path in curated_sessions:
        session_id = get_session_id(path)
        session_data, session_summary = process_session(path, make_plot=session_id in plot_ids)
        session_dicts.append(session_data)
        session_summaries.append(session_summary)

    dataset = build_dataset(session_dicts, session_summaries)

    with open(args.outpicklefile, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_trials = sum(len(sess["neural"]) for sess in session_dicts)
    n_units = sum(len(sess["region_labels"]) for sess in session_dicts)
    elapsed = time.perf_counter() - t_all
    print(
        f"[done] saved {args.outpicklefile} with {len(session_dicts)} sessions, "
        f"{n_trials} trials, {n_units} good units in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
