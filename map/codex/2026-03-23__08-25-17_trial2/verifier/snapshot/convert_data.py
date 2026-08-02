#!/usr/bin/env python3
"""Convert Mesoscale Activity Map NWB sessions into decoder-ready pickle format."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gc

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("/app/data")
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_WIDTH_S / 2.0
N_BINS = len(BIN_CENTERS_REL)
LIKELIHOOD_THRESHOLD = 0.1

CHOICE_MAP = {"left": 0, "right": 1}
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
EARLY_MAP = {"no early": 0, "early": 1}


@dataclass
class SessionResult:
    session_id: str
    subject_id: str
    neural: list[np.ndarray]
    input: list[np.ndarray]
    output: list[np.ndarray]
    brain_region_labels: list[str]
    n_trials: int
    n_good_units: int
    n_ignore_choice_fallback: int
    n_missing_sample_onset_fallback: int
    stats: dict
    plot_payload: dict | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MAP NWB files into decoder-ready pickle format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Process all sessions (default).")
    group.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save detailed processing plots for up to 2 sessions.",
    )
    return parser.parse_args()


def decode_strings(dataset: h5py.Dataset) -> np.ndarray:
    try:
        return dataset.asstr()[:]
    except Exception:
        arr = dataset[:]
        return np.array(
            [x.decode() if isinstance(x, bytes) else str(x) for x in arr],
            dtype=object,
        )


def parse_optional_float(value: str) -> float | None:
    if value in {"N/A", "", "nan", "None"}:
        return None
    return float(value)


def interval_values(times: np.ndarray, start: float, stop: float) -> np.ndarray:
    lo = np.searchsorted(times, start, side="left")
    hi = np.searchsorted(times, stop, side="right")
    return times[lo:hi]


def split_ragged(flat: np.ndarray, index: np.ndarray) -> list[np.ndarray]:
    starts = np.concatenate(([0], index[:-1]))
    return [flat[s:e] for s, e in zip(starts, index, strict=True)]


def process_tongue_trace(
    tongue_xyzl: np.ndarray, tongue_timestamps: np.ndarray
) -> tuple[np.ndarray, dict]:
    xy = tongue_xyzl[:, :2].astype(np.float64, copy=False)
    y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
    likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)

    dt = np.diff(tongue_timestamps)
    dt[dt == 0] = np.nan
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    valid_velocity = np.isfinite(velocity)
    if np.any(valid_velocity):
        vel_mean = np.nanmean(velocity[valid_velocity])
        vel_std = np.nanstd(velocity[valid_velocity])
        vel_threshold = vel_mean + 5.0 * vel_std
    else:
        vel_threshold = np.inf

    outlier_mask = np.zeros(len(y), dtype=bool)
    outlier_mask[1:] = np.isfinite(velocity) & (velocity > vel_threshold)
    low_likelihood_mask = likelihood < LIKELIHOOD_THRESHOLD

    good_mask = ~(outlier_mask | low_likelihood_mask | ~np.isfinite(y))
    if not np.any(good_mask):
        session_mean_y = float(np.nanmean(y[np.isfinite(y)]))
        if not math.isfinite(session_mean_y):
            session_mean_y = 0.0
        processed_y = np.full_like(y, session_mean_y, dtype=np.float64)
    else:
        session_mean_y = float(np.mean(y[good_mask]))
        processed_y = y.copy()
        processed_y[low_likelihood_mask] = session_mean_y
        interp_mask = outlier_mask | ~np.isfinite(processed_y)
        keep_mask = ~interp_mask
        if np.any(interp_mask):
            processed_y[interp_mask] = np.interp(
                tongue_timestamps[interp_mask],
                tongue_timestamps[keep_mask],
                processed_y[keep_mask],
            )

    p40, p60 = np.percentile(processed_y, [40.0, 60.0])
    info = {
        "session_mean_y": session_mean_y,
        "p40": float(p40),
        "p60": float(p60),
        "n_low_likelihood": int(low_likelihood_mask.sum()),
        "n_outlier_frames": int(outlier_mask.sum()),
        "velocity_threshold": float(vel_threshold),
    }
    return processed_y.astype(np.float32), info


def infer_choice(
    instruction: str,
    outcome: str,
    trial_start: float,
    trial_stop: float,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> tuple[int, str]:
    if outcome == "hit":
        return CHOICE_MAP[instruction], "instruction+outcome"
    if outcome == "miss":
        opposite = "right" if instruction == "left" else "left"
        return CHOICE_MAP[opposite], "instruction+outcome"

    left_trial = interval_values(left_lick_times, trial_start, trial_stop)
    right_trial = interval_values(right_lick_times, trial_start, trial_stop)
    if len(left_trial) and len(right_trial):
        side = "left" if left_trial[0] <= right_trial[0] else "right"
        return CHOICE_MAP[side], "ignore:first_lick_in_trial"
    if len(left_trial):
        return CHOICE_MAP["left"], "ignore:first_lick_in_trial"
    if len(right_trial):
        return CHOICE_MAP["right"], "ignore:first_lick_in_trial"
    return CHOICE_MAP[instruction], "ignore:instruction_fallback"


def build_region_labels(anno_name: np.ndarray, good_unit_mask: np.ndarray) -> list[str]:
    labels = [str(x) for x in anno_name[good_unit_mask].tolist()]
    if any((label == "" or label.lower() == "nan") for label in labels):
        raise ValueError("Good units unexpectedly contain empty anatomical annotations.")
    return labels


def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))


def process_session(path: Path, make_plot: bool = False) -> SessionResult | None:
    session_start = time.time()
    with h5py.File(path, "r") as f:
        classification = decode_strings(f["units"]["classification"])
        good_unit_mask = classification == "good"
        n_good_units = int(good_unit_mask.sum())
        if n_good_units == 0:
            return None

        anno_name = decode_strings(f["units"]["anno_name"])
        brain_region_labels = build_region_labels(anno_name, good_unit_mask)

        subject_id = str(f["general"]["subject"]["subject_id"][()])
        if subject_id.startswith("b'"):
            subject_id = subject_id[2:-1]
        subject_id = f"sub-{subject_id}"
        session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
            "_behavior+ecephys", ""
        )

        spike_times_flat = f["units"]["spike_times"][:]
        spike_times_index = f["units"]["spike_times_index"][:]
        spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
        good_unit_indices = np.flatnonzero(good_unit_mask)
        good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
        obs_intervals = f["units"]["obs_intervals"][:].astype(np.float64)
        obs_intervals_index = f["units"]["obs_intervals_index"][:]
        first_good_unit = int(good_unit_indices[0])
        obs_start = 0 if first_good_unit == 0 else int(obs_intervals_index[first_good_unit - 1])
        obs_stop = int(obs_intervals_index[first_good_unit])
        session_obs_intervals = obs_intervals[obs_start:obs_stop]

        trials = f["intervals"]["trials"]
        trial_start = trials["start_time"][:].astype(np.float64)
        trial_stop = trials["stop_time"][:].astype(np.float64)
        trial_instruction = decode_strings(trials["trial_instruction"])
        trial_outcome = decode_strings(trials["outcome"])
        trial_early = decode_strings(trials["early_lick"])
        trial_photostim_onset = decode_strings(trials["photostim_onset"])
        trial_photostim_duration = decode_strings(trials["photostim_duration"])
        trial_photostim_power = decode_strings(trials["photostim_power"])

        events = f["acquisition"]["BehavioralEvents"]
        go_times = events["go_start_times"]["timestamps"][:].astype(np.float64)
        sample_start_times = events["sample_start_times"]["timestamps"][:].astype(np.float64)
        left_lick_times = events["left_lick_times"]["timestamps"][:].astype(np.float64)
        right_lick_times = events["right_lick_times"]["timestamps"][:].astype(np.float64)

        tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
        tongue_data = tongue_group["data"][:].astype(np.float64)
        tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
        processed_tongue_y, tongue_info = process_tongue_trace(tongue_data, tongue_timestamps)

    trial_records: list[dict] = []
    choice_sources = Counter()
    n_missing_sample_onset_fallback = 0
    n_trials_dropped_outside_obs = 0

    for trial_idx in range(len(trial_start)):
        start = trial_start[trial_idx]
        stop = trial_stop[trial_idx]
        go_candidates = interval_values(go_times, start, stop)
        if len(go_candidates) == 0:
            raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
        go_time = float(go_candidates[-1])
        window_start = go_time + WINDOW_START_S
        window_end = go_time + WINDOW_END_S
        covered = np.any(
            (window_start >= session_obs_intervals[:, 0])
            & (window_end <= session_obs_intervals[:, 1])
        )
        if not covered:
            n_trials_dropped_outside_obs += 1
            continue

        sample_candidates = interval_values(sample_start_times, start, go_time)
        if len(sample_candidates) == 0:
            sample_onset = go_time - 1.85
            n_missing_sample_onset_fallback += 1
        else:
            sample_onset = float(sample_candidates[-1])

        bin_centers_abs = go_time + BIN_CENTERS_REL
        time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)

        photostim_row = np.zeros(N_BINS, dtype=np.float32)
        stim_onset = parse_optional_float(trial_photostim_onset[trial_idx])
        stim_dur = parse_optional_float(trial_photostim_duration[trial_idx])
        stim_power = parse_optional_float(trial_photostim_power[trial_idx])
        if stim_power is not None and stim_onset is not None and stim_dur is not None:
            stim_start_abs = start + stim_onset
            stim_stop_abs = stim_start_abs + stim_dur
            photostim_row = (
                (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
            ).astype(np.float32)

        choice_code, choice_source = infer_choice(
            instruction=trial_instruction[trial_idx],
            outcome=trial_outcome[trial_idx],
            trial_start=start,
            trial_stop=stop,
            left_lick_times=left_lick_times,
            right_lick_times=right_lick_times,
        )
        choice_sources[choice_source] += 1
        outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
        early_code = EARLY_MAP[trial_early[trial_idx]]

        tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
        tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
        tongue_y = processed_tongue_y[tongue_frame_idx]
        tongue_bin = np.zeros(N_BINS, dtype=np.int16)
        tongue_bin[tongue_y >= tongue_info["p40"]] = 1
        tongue_bin[tongue_y > tongue_info["p60"]] = 2

        output_row = np.empty((4, N_BINS), dtype=np.int16)
        output_row[0, :] = choice_code
        output_row[1, :] = outcome_code
        output_row[2, :] = early_code
        output_row[3, :] = tongue_bin
        trial_records.append(
            {
                "raw_trial_idx": trial_idx,
                "go_time": go_time,
                "sample_onset": sample_onset,
                "time_from_tone": time_from_tone,
                "photostim_on": photostim_row,
                "output": output_row,
            }
        )

    n_trials = len(trial_records)
    if n_trials == 0:
        raise ValueError(f"{path.name}: all trials were excluded by observation-interval coverage.")

    trial_edge_matrix = np.empty((n_trials, N_BINS + 1), dtype=np.float64)
    input_tensor = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
    output_tensor = np.empty((n_trials, 4, N_BINS), dtype=np.int16)
    go_per_trial = np.empty(n_trials, dtype=np.float64)
    sample_on_per_trial = np.empty(n_trials, dtype=np.float64)
    for keep_idx, rec in enumerate(trial_records):
        go_per_trial[keep_idx] = rec["go_time"]
        sample_on_per_trial[keep_idx] = rec["sample_onset"]
        trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
        input_tensor[keep_idx, 0, :] = rec["time_from_tone"]
        input_tensor[keep_idx, 1, :] = rec["photostim_on"]
        output_tensor[keep_idx] = rec["output"]

    neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
    for unit_idx, spikes in enumerate(good_spike_times):
        edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
        counts = np.diff(edge_idx, axis=1)
        neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)

    nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
    n_trials_dropped_all_zero = int((~nonzero_trial_mask).sum())
    if n_trials_dropped_all_zero:
        neural_tensor = neural_tensor[nonzero_trial_mask]
        input_tensor = input_tensor[nonzero_trial_mask]
        output_tensor = output_tensor[nonzero_trial_mask]
        go_per_trial = go_per_trial[nonzero_trial_mask]
        sample_on_per_trial = sample_on_per_trial[nonzero_trial_mask]
        trial_records = [rec for rec, keep in zip(trial_records, nonzero_trial_mask, strict=True) if keep]
        n_trials = len(trial_records)

    neural_trials = [neural_tensor[i] for i in range(n_trials)]
    input_trials = [input_tensor[i] for i in range(n_trials)]
    output_trials = [output_tensor[i] for i in range(n_trials)]

    plot_payload = None
    if make_plot:
        # Pick a representative non-early, non-ignore trial when possible.
        candidate = np.where(
            (output_tensor[:, 1, 0] != OUTCOME_MAP["ignore"])
            & (output_tensor[:, 2, 0] == EARLY_MAP["no early"])
        )[0]
        trial_idx = int(candidate[0] if len(candidate) else 0)
        raw_trial_idx = int(trial_records[trial_idx]["raw_trial_idx"])
        plot_payload = {
            "trial_idx": raw_trial_idx,
            "trial_start": float(trial_start[raw_trial_idx]),
            "trial_stop": float(trial_stop[raw_trial_idx]),
            "go_time": float(go_per_trial[trial_idx]),
            "sample_onset": float(sample_on_per_trial[trial_idx]),
            "bin_centers_rel": BIN_CENTERS_REL.copy(),
            "time_from_tone": input_tensor[trial_idx, 0].copy(),
            "photostim_on": input_tensor[trial_idx, 1].copy(),
            "tongue_y_raw": tongue_data[:, 1].astype(np.float32),
            "tongue_y_processed": processed_tongue_y.copy(),
            "tongue_timestamps": tongue_timestamps.copy(),
            "tongue_p40": tongue_info["p40"],
            "tongue_p60": tongue_info["p60"],
            "tongue_bins": output_tensor[trial_idx, 3].copy(),
            "neural_trial": neural_tensor[trial_idx].astype(np.float32),
            "choice": int(output_tensor[trial_idx, 0, 0]),
            "outcome": int(output_tensor[trial_idx, 1, 0]),
            "early": int(output_tensor[trial_idx, 2, 0]),
        }

    stats = {
        "n_trials": n_trials,
        "n_good_units": n_good_units,
        "choice_sources": dict(choice_sources),
        "n_missing_sample_onset_fallback": n_missing_sample_onset_fallback,
        "n_trials_dropped_outside_obs": n_trials_dropped_outside_obs,
        "n_trials_dropped_all_zero": n_trials_dropped_all_zero,
        "tongue_info": tongue_info,
        "stim_trials": int(np.sum(input_tensor[:, 1, :].any(axis=1))),
        "session_seconds": float(time.time() - session_start),
    }

    return SessionResult(
        session_id=session_id,
        subject_id=subject_id,
        neural=neural_trials,
        input=input_trials,
        output=output_trials,
        brain_region_labels=brain_region_labels,
        n_trials=n_trials,
        n_good_units=n_good_units,
        n_ignore_choice_fallback=int(
            choice_sources["ignore:first_lick_in_trial"] + choice_sources["ignore:instruction_fallback"]
        ),
        n_missing_sample_onset_fallback=n_missing_sample_onset_fallback,
        stats=stats,
        plot_payload=plot_payload,
    )


def plot_processing(result: SessionResult, out_path: Path) -> None:
    payload = result.plot_payload
    if payload is None:
        return

    fig, ax = plt.subplots(3, 2, figsize=(16, 12))

    trial_idx = payload["trial_idx"]
    trial_mask = (payload["tongue_timestamps"] >= payload["trial_start"]) & (
        payload["tongue_timestamps"] <= payload["trial_stop"]
    )
    tongue_t_rel = payload["tongue_timestamps"][trial_mask] - payload["go_time"]

    ax[0, 0].plot(
        tongue_t_rel,
        payload["tongue_y_raw"][trial_mask],
        color="tab:gray",
        linewidth=1,
        label="raw tongue y",
    )
    ax[0, 0].plot(
        tongue_t_rel,
        payload["tongue_y_processed"][trial_mask],
        color="tab:red",
        linewidth=1,
        label="processed tongue y",
    )
    ax[0, 0].axvline(payload["sample_onset"] - payload["go_time"], color="tab:blue", linestyle="--", label="tone onset")
    ax[0, 0].axvline(0.0, color="k", linestyle="--", label="go cue")
    ax[0, 0].set_title(f"{result.session_id} trial {trial_idx}: tongue preprocessing")
    ax[0, 0].set_xlabel("time to go (s)")
    ax[0, 0].set_ylabel("tongue y")
    ax[0, 0].legend(loc="best", fontsize=8)

    ax[0, 1].plot(payload["bin_centers_rel"], payload["time_from_tone"], color="tab:blue")
    ax[0, 1].step(payload["bin_centers_rel"], payload["photostim_on"], where="mid", color="tab:orange")
    ax[0, 1].axvline(0.0, color="k", linestyle="--")
    ax[0, 1].set_title("time_from_tone and photostim_on inputs")
    ax[0, 1].set_xlabel("time to go (s)")

    sample_neurons = min(40, payload["neural_trial"].shape[0])
    im = ax[1, 0].imshow(
        payload["neural_trial"][:sample_neurons],
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[WINDOW_START_S, WINDOW_END_S, 0, sample_neurons],
    )
    ax[1, 0].axvline(0.0, color="w", linestyle="--")
    ax[1, 0].set_title("example neural firing rates (Hz)")
    ax[1, 0].set_xlabel("time to go (s)")
    ax[1, 0].set_ylabel("good units")
    fig.colorbar(im, ax=ax[1, 0], shrink=0.8)

    ax[1, 1].step(payload["bin_centers_rel"], payload["tongue_bins"], where="mid", color="tab:green")
    ax[1, 1].axhline(1, color="tab:gray", linestyle=":")
    ax[1, 1].axhline(2, color="tab:gray", linestyle=":")
    ax[1, 1].set_title("discretized tongue y output")
    ax[1, 1].set_xlabel("time to go (s)")
    ax[1, 1].set_ylabel("class")

    ax[2, 0].hist(payload["tongue_y_processed"], bins=100, color="tab:purple", alpha=0.8)
    ax[2, 0].axvline(payload["tongue_p40"], color="tab:blue", linestyle="--", label="40th pct")
    ax[2, 0].axvline(payload["tongue_p60"], color="tab:red", linestyle="--", label="60th pct")
    ax[2, 0].set_title("session tongue y percentiles")
    ax[2, 0].set_xlabel("tongue y")
    ax[2, 0].legend(loc="best", fontsize=8)

    text = [
        f"session: {result.session_id}",
        f"subject: {result.subject_id}",
        f"good units: {result.n_good_units}",
        f"trials: {result.n_trials}",
        f"choice: {result.stats['choice_sources']}",
        f"missing sample fallback: {result.n_missing_sample_onset_fallback}",
        f"choice fallback ignore: {result.n_ignore_choice_fallback}",
        f"tongue low-likelihood frames: {result.stats['tongue_info']['n_low_likelihood']}",
        f"tongue outlier frames: {result.stats['tongue_info']['n_outlier_frames']}",
        f"trial labels: choice={payload['choice']} outcome={payload['outcome']} early={payload['early']}",
    ]
    ax[2, 1].axis("off")
    ax[2, 1].text(0.0, 1.0, "\n".join(text), va="top", family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_dataset(results: list[SessionResult], excluded_sessions: list[str]) -> dict:
    subjects: list[str] = []
    subject_to_idx: dict[str, int] = {}
    brain_regions: list[str] = []
    region_to_idx: dict[str, int] = {}

    neural = []
    decoder_input = []
    output = []
    subject_idx = []
    brain_region_idx = []

    total_trials = 0
    total_units = 0
    total_ignore_choice_fallback = 0
    total_missing_sample_onset_fallback = 0

    for result in results:
        if result.subject_id not in subject_to_idx:
            subject_to_idx[result.subject_id] = len(subjects)
            subjects.append(result.subject_id)
        subject_idx.append(subject_to_idx[result.subject_id])

        session_region_idx = np.empty(len(result.brain_region_labels), dtype=np.int32)
        for i, label in enumerate(result.brain_region_labels):
            if label not in region_to_idx:
                region_to_idx[label] = len(brain_regions)
                brain_regions.append(label)
            session_region_idx[i] = region_to_idx[label]

        neural.append(result.neural)
        decoder_input.append(result.input)
        output.append(result.output)
        brain_region_idx.append(session_region_idx)

        total_trials += result.n_trials
        total_units += result.n_good_units
        total_ignore_choice_fallback += result.n_ignore_choice_fallback
        total_missing_sample_onset_fallback += result.n_missing_sample_onset_fallback

    metadata = {
        "task_description": (
            "Auditory delayed-response task. Decoder predicts choice, outcome, early lick, "
            "and discretized tongue y-position from go-cue aligned neural firing rates."
        ),
        "time_bin_size": 50.0,
        "temporal_alignment_event": "Go cue onset",
        "off_start": WINDOW_START_S,
        "off_end": WINDOW_END_S,
        "n_sessions": len(results),
        "n_trials": total_trials,
        "n_good_units": total_units,
        "session_ids": [r.session_id for r in results],
        "excluded_sessions": excluded_sessions,
        "unit_filter": 'units.classification == "good"',
        "brain_region_field": "units.anno_name",
        "tone_onset_definition": "last sample_start event before go cue within the trial",
        "photostim_definition": "binary at 50 ms bin centers using trial-relative photostim onset/duration",
        "choice_note": (
            "choice inferred from instruction+outcome on hit/miss trials; ignore trials use "
            "earliest lick in trial when present, otherwise instructed side as placeholder"
        ),
        "tongue_processing": {
            "likelihood_threshold": LIKELIHOOD_THRESHOLD,
            "outlier_rule": "5-sigma threshold on frame-to-frame velocity",
            "low_likelihood_fill": "session mean tongue y",
            "outlier_fill": "linear interpolation",
            "discretization_percentiles": [40, 60],
        },
        "n_ignore_choice_fallback": total_ignore_choice_fallback,
        "n_missing_sample_onset_fallback": total_missing_sample_onset_fallback,
        "bin_centers_rel_s": BIN_CENTERS_REL.astype(np.float32),
    }

    return {
        "neural": neural,
        "input": decoder_input,
        "output": output,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_tone_onset_s", "photostim_on"],
        "output_names": ["choice", "outcome", "early_lick", "tongue_y_bin"],
        "output_values": [
            ["left", "right"],
            ["ignore", "miss", "hit"],
            ["no", "yes"],
            ["<40th_pct", "40th_to_60th_pct", ">60th_pct"],
        ],
        "metadata": metadata,
    }


def select_files(all_files: list[Path], sample_mode: bool) -> tuple[list[Path], list[str]]:
    selected: list[Path] = []
    excluded: list[str] = []
    for path in all_files:
        with h5py.File(path, "r") as f:
            classification = decode_strings(f["units"]["classification"])
            if np.sum(classification == "good") == 0:
                excluded.append(path.name)
                continue
        selected.append(path)
    if sample_mode:
        return selected[:2], excluded
    return selected, excluded


def main() -> None:
    args = parse_args()
    sample_mode = args.sample

    overall_start = time.time()
    all_files = load_candidate_files()
    session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)

    print(f"Found {len(all_files)} NWB files under {DATA_DIR}")
    print(f"Excluded {len(excluded_sessions)} sessions with zero classifier-good units")
    print(f"Processing {len(session_files)} sessions ({'sample' if sample_mode else 'full'})")
    print(
        f"Neural window: [{WINDOW_START_S:.1f}, {WINDOW_END_S:.1f}) s relative to go, "
        f"{N_BINS} bins at {BIN_WIDTH_S * 1000:.0f} ms"
    )

    results: list[SessionResult] = []
    session_times = []
    for i, path in enumerate(session_files, start=1):
        make_plot = args.show_processing and len(results) < 2
        print(f"[{i}/{len(session_files)}] {path.name}")
        t0 = time.time()
        result = process_session(path, make_plot=make_plot)
        if result is None:
            print("  skipped (no good units)")
            continue
        results.append(result)
        elapsed = time.time() - t0
        session_times.append(elapsed)
        print(
            f"  kept {result.n_good_units} good units, {result.n_trials} trials, "
            f"{elapsed:.2f}s"
        )
        if make_plot:
            plot_path = Path(f"processing_{result.session_id}.png")
            plot_processing(result, plot_path)
            print(f"  wrote {plot_path.name}")
        gc.collect()

    dataset = build_dataset(results, excluded_sessions)

    with open(args.outpicklefile, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_elapsed = time.time() - overall_start
    out_size_mb = Path(args.outpicklefile).stat().st_size / (1024 * 1024)
    mean_session_s = float(np.mean(session_times)) if session_times else 0.0
    est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))

    print("\nConversion complete.")
    print(f"  sessions: {len(dataset['neural'])}")
    print(f"  subjects: {len(dataset['subjects'])}")
    print(f"  total trials: {dataset['metadata']['n_trials']}")
    print(f"  total good units: {dataset['metadata']['n_good_units']}")
    print(f"  brain region labels: {len(dataset['brain_regions'])}")
    print(f"  output file: {args.outpicklefile} ({out_size_mb:.1f} MB)")
    print(f"  elapsed: {total_elapsed / 60.0:.2f} min")
    print(f"  mean session time: {mean_session_s:.2f}s")
    if sample_mode:
        print(f"  projected full-conversion time: {est_full_s / 60.0:.2f} min")


if __name__ == "__main__":
    main()
