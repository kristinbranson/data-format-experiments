#!/usr/bin/env python3
"""Convert the staged IBL brain-wide-map release to decoder format.

Usage
-----
python -u /app/convert_data.py OUTPUT [--full | --sample] [--show-processing]

The implementation follows code_zhang2025's common cache representation:
stimulus-onset alignment, [-0.5, 1.5) s windows, 20 ms spike-count bins,
all Kilosort clusters, and SessionLoader-equivalent wheel processing.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from brainbox.behavior.wheel import interpolate_position, velocity_filtered


APP = Path("/app")
DATA_ROOT = APP / "data" / "one_cache"
FREEZE_CSV = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
SPIKE_BIN_CHUNK_ELEMENTS = 25_000_000  # <=200 MB temporary int64 bincount array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output pickle path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="Process two sessions")
    parser.add_argument(
        "--show-processing", action="store_true",
        help="Save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


def session_path(row: pd.Series) -> Path:
    return (
        DATA_ROOT / str(row["lab"]) / "Subjects" / str(row["subject"])
        / str(row["date"]) / f"{int(row['session_number']):03d}"
    )


def revision_key(path: Path) -> tuple[str, str]:
    """Sort unrevisioned paths before ISO-date revision folders."""
    revisions = [p[1:-1] for p in path.parts if p.startswith("#") and p.endswith("#")]
    return (max(revisions, default=""), str(path))


def newest_file(base: Path, pattern: str, preferred_revision: str | None = None) -> Path | None:
    paths = sorted(base.glob(f"**/{pattern}"), key=revision_key)
    if not paths:
        return None
    if preferred_revision:
        preferred = [p for p in paths if preferred_revision in p.parts]
        if preferred:
            return preferred[-1]
    return paths[-1]


def load_freeze() -> tuple[pd.DataFrame, list[dict]]:
    freeze = pd.read_csv(FREEZE_CSV)
    freeze = freeze.loc[:, ~freeze.columns.str.startswith("Unnamed:")]
    required = {"pid", "eid", "probe_name", "session_number", "date", "subject", "lab"}
    missing = required - set(freeze.columns)
    if missing:
        raise ValueError(f"Freeze CSV missing columns: {sorted(missing)}")

    sessions: list[dict] = []
    # sort=False preserves the paper freeze order and probe order.
    for eid, group in freeze.groupby("eid", sort=False):
        first = group.iloc[0]
        sessions.append({
            "eid": str(eid),
            "lab": str(first["lab"]),
            "subject": str(first["subject"]),
            "date": str(first["date"]),
            "session_number": int(first["session_number"]),
            "pids": group["pid"].astype(str).tolist(),
            "probe_names": group["probe_name"].astype(str).tolist(),
        })
    return freeze, sessions


def load_trials(info: dict) -> tuple[pd.DataFrame, Path]:
    alf = session_path(pd.Series(info)) / "alf"
    path = newest_file(alf, "_ibl_trials.table.pqt", "#2025-03-03#")
    if path is None:
        raise FileNotFoundError(f"No trial table for {info['eid']}")
    trials = pd.read_parquet(path)
    return trials, path


def trial_mask(trials: pd.DataFrame) -> np.ndarray:
    """Reproduce load_trials_and_mask(..., max_trial_len=10)."""
    required = [
        "stimOn_times", "choice", "feedback_times", "probabilityLeft",
        "firstMovement_times", "feedbackType",
    ]
    missing = [c for c in required if c not in trials]
    if missing:
        raise ValueError(f"Trial table missing required columns: {missing}")

    good = np.ones(len(trials), dtype=bool)
    for column in required:
        good &= trials[column].notna().to_numpy()
    reaction_time = (
        trials["firstMovement_times"].to_numpy(dtype=float)
        - trials["stimOn_times"].to_numpy(dtype=float)
    )
    good &= reaction_time >= 0.08
    good &= reaction_time <= 2.0
    good &= trials["choice"].to_numpy() != 0
    # Reference query excludes only durations >10; NaN go cues are not explicitly excluded.
    if "goCue_times" in trials:
        duration = (
            trials["feedback_times"].to_numpy(dtype=float)
            - trials["goCue_times"].to_numpy(dtype=float)
        )
        good &= ~(duration > 10.0)
    return good


def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    changes = np.ones(len(probability_left), dtype=bool)
    if len(probability_left) > 1:
        changes[1:] = ~np.isclose(
            probability_left[1:], probability_left[:-1], rtol=0.0, atol=1e-8,
            equal_nan=False,
        )
    starts = np.maximum.accumulate(np.where(changes, np.arange(len(probability_left)), 0))
    return (np.arange(len(probability_left)) - starts).astype(np.float32)


def _linear_interp_extrapolate(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Equivalent to scipy interp1d(..., linear, extrapolate) for 1-D data."""
    out = np.interp(query, x, y).astype(np.float64, copy=False)
    left = query < x[0]
    right = query > x[-1]
    if np.any(left):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        out[left] = y[0] + slope * (query[left] - x[0])
    if np.any(right):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        out[right] = y[-1] + slope * (query[right] - x[-1])
    return out


def sample_behavior(
    times: np.ndarray,
    values: np.ndarray,
    stim_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce get_behavior_per_interval at the reference bin-right-edge grid."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    stim_times = np.asarray(stim_times, dtype=np.float64)
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values):
        raise ValueError("Behavior times/values must be same-length 1-D arrays")
    if len(times) < 2 or np.any(np.diff(times) < 0):
        raise ValueError("Behavior timestamps must be sorted and contain >=2 samples")

    beginnings = stim_times + OFF_START
    endings = stim_times + OFF_END
    ibeg = np.searchsorted(times, beginnings, side="right")
    iend = np.searchsorted(times, endings, side="left")
    sampled = np.full((len(stim_times), N_BINS), np.nan, dtype=np.float32)
    good = np.zeros(len(stim_times), dtype=bool)

    for trial in range(len(stim_times)):
        lo, hi = int(ibeg[trial]), int(iend[trial])
        if hi - lo < 2:
            continue
        tx = times[lo:hi]
        vy = values[lo:hi]
        if abs(beginnings[trial] - tx[0]) > BIN_SIZE:
            continue
        if abs(endings[trial] - tx[-1]) > BIN_SIZE:
            continue
        query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
        sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
        good[trial] = True
    return sampled, good


def load_wheel_samples(
    alf: Path, stim_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    timestamp_path = newest_file(alf, "_ibl_wheel.timestamps.npy")
    position_path = newest_file(alf, "_ibl_wheel.position.npy")
    if timestamp_path is None or position_path is None:
        raise FileNotFoundError("Wheel timestamps or position missing")
    raw_times = np.load(timestamp_path, mmap_mode="r")
    raw_position = np.load(position_path, mmap_mode="r")
    if len(raw_times) != len(raw_position):
        raise ValueError("Wheel timestamps and position length mismatch")

    position, times = interpolate_position(raw_times, raw_position, freq=1000)
    velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
    speed = np.abs(velocity)
    sampled, good = sample_behavior(times, speed, stim_times)
    aux = {
        "times": times,
        "values": speed,
        "timestamp_path": str(timestamp_path),
        "position_path": str(position_path),
    }
    return sampled, good, aux


def load_motion_samples(
    alf: Path, stim_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    failures: list[str] = []
    for view in ("left", "right"):
        value_path = newest_file(alf, f"{view}Camera.ROIMotionEnergy.npy")
        time_path = newest_file(alf, f"_ibl_{view}Camera.times.npy")
        if value_path is None or time_path is None:
            failures.append(f"{view}: files missing")
            continue
        values = np.load(value_path, mmap_mode="r")
        times = np.load(time_path, mmap_mode="r")
        if len(times) < len(values):
            failures.append(f"{view}: timestamps shorter than data")
            continue
        if len(times) > len(values):
            times = times[-len(values):]
        sampled, good = sample_behavior(times, values, stim_times)
        aux = {
            "times": times,
            "values": values,
            "view": view,
            "timestamp_path": str(time_path),
            "value_path": str(value_path),
        }
        return sampled, good, aux
    raise FileNotFoundError("No usable whisker motion stream; " + "; ".join(failures))


def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Continuous behavior has no finite values")
    thresholds = np.quantile(values[finite], [1 / 3, 2 / 3]).astype(np.float64)
    median = float(np.median(values[finite]))
    imputed = int(np.size(values) - np.count_nonzero(finite))
    clean = np.where(finite, values, median)
    categories = np.digitize(clean, thresholds, right=False).astype(np.int8)
    return categories, thresholds, imputed, median


def probe_directory(alf: Path, probe_name: str) -> Path:
    base = alf / probe_name / "pykilosort"
    revised = base / "#2024-05-06#"
    return revised if revised.is_dir() else base


def load_probe_metadata(alf: Path, probe_name: str) -> dict:
    directory = probe_directory(alf, probe_name)
    metrics_path = directory / "clusters.metrics.pqt"
    channels_path = directory / "clusters.channels.npy"
    atlas_path = directory / "channels.brainLocationIds_ccf_2017.npy"
    spikes_times_path = directory / "spikes.times.npy"
    spikes_clusters_path = directory / "spikes.clusters.npy"
    for path in (metrics_path, channels_path, atlas_path, spikes_times_path, spikes_clusters_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metrics = pd.read_parquet(metrics_path)
    cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)
    n_clusters = len(metrics)
    if len(np.unique(cluster_ids)) != n_clusters:
        raise ValueError(f"Duplicate cluster IDs in {metrics_path}")
    cluster_channels = np.load(channels_path)
    channel_atlas_ids = np.load(atlas_path)
    if len(cluster_channels) != n_clusters:
        raise ValueError(f"Cluster channel length mismatch in {directory}")
    if np.any(cluster_channels < 0) or np.any(cluster_channels >= len(channel_atlas_ids)):
        raise ValueError(f"Out-of-range cluster channel in {directory}")

    regions = BrainRegions()
    allen = regions.id2acronym(channel_atlas_ids[cluster_channels.astype(int)])
    beryl = regions.acronym2acronym(allen, mapping="Beryl").astype(str)
    return {
        "directory": directory,
        "n_clusters": n_clusters,
        "cluster_ids": cluster_ids,
        "regions": beryl,
        "n_good": int((metrics["label"].to_numpy() >= 1).sum()),
        "spikes_times_path": spikes_times_path,
        "spikes_clusters_path": spikes_clusters_path,
    }


def map_spike_clusters(raw_clusters: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    if np.array_equal(cluster_ids, np.arange(len(cluster_ids))):
        mapped = raw_clusters.astype(np.int64, copy=False)
        if len(mapped) and (mapped.min() < 0 or mapped.max() >= len(cluster_ids)):
            raise ValueError("Spike cluster index outside cluster table")
        return mapped
    order = np.argsort(cluster_ids)
    sorted_ids = cluster_ids[order]
    positions = np.searchsorted(sorted_ids, raw_clusters)
    if np.any(positions >= len(sorted_ids)) or np.any(sorted_ids[positions] != raw_clusters):
        raise ValueError("Spike cluster ID absent from cluster table")
    return order[positions]


def bin_probe(
    probe: dict,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Bin one probe into trial x cluster x time with bounded temporaries."""
    spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
    spike_clusters = np.load(probe["spikes_clusters_path"], mmap_mode="r")
    if len(spike_times) != len(spike_clusters):
        raise ValueError(f"Spike times/clusters length mismatch: {probe['directory']}")
    n_trials = len(starts)
    n_clusters = probe["n_clusters"]
    output = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    trials_per_chunk = max(1, SPIKE_BIN_CHUNK_ELEMENTS // max(1, n_clusters * N_BINS))

    for first in range(0, n_trials, trials_per_chunk):
        last = min(n_trials, first + trials_per_chunk)
        encoded: list[np.ndarray] = []
        for local_trial, trial in enumerate(range(first, last)):
            lo = int(np.searchsorted(spike_times, starts[trial], side="left"))
            hi = int(np.searchsorted(spike_times, ends[trial], side="left"))
            if hi <= lo:
                continue
            times = np.asarray(spike_times[lo:hi])
            raw_clusters = np.asarray(spike_clusters[lo:hi])
            clusters = map_spike_clusters(raw_clusters, probe["cluster_ids"])
            bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
            keep = (bins >= 0) & (bins < N_BINS)
            if not np.all(keep):
                bins = bins[keep]
                clusters = clusters[keep]
            code = ((local_trial * n_clusters + clusters) * N_BINS + bins).astype(np.int64)
            encoded.append(code)
        if encoded:
            flat = np.concatenate(encoded)
            counts = np.bincount(
                flat, minlength=(last - first) * n_clusters * N_BINS,
            ).reshape(last - first, n_clusters, N_BINS)
            output[first:last] = counts
    return output


def make_plot_payload(
    eid: str,
    source_indices: np.ndarray,
    neural: np.ndarray,
    wheel_continuous: np.ndarray,
    motion_continuous: np.ndarray,
    wheel_categories: np.ndarray,
    motion_categories: np.ndarray,
    choices: np.ndarray,
    priors: np.ndarray,
    block_numbers: np.ndarray,
    wheel_thresholds: np.ndarray,
    motion_thresholds: np.ndarray,
    wheel_aux: dict,
    motion_aux: dict,
    stim_times: np.ndarray,
    raw_trials: int,
    code_valid_trials: int,
) -> dict:
    trial = 0
    stim = stim_times[trial]
    wheel_mask = (wheel_aux["times"] >= stim + OFF_START - 0.1) & (
        wheel_aux["times"] <= stim + OFF_END + 0.1
    )
    motion_mask = (motion_aux["times"] >= stim + OFF_START - 0.1) & (
        motion_aux["times"] <= stim + OFF_END + 0.1
    )
    wheel_idx = np.flatnonzero(wheel_mask)
    motion_idx = np.flatnonzero(motion_mask)
    if len(wheel_idx) > 5000:
        wheel_idx = wheel_idx[:: max(1, len(wheel_idx) // 5000)]
    if len(motion_idx) > 5000:
        motion_idx = motion_idx[:: max(1, len(motion_idx) // 5000)]
    return {
        "eid": eid,
        "source_trial": int(source_indices[trial]),
        "neural": neural[trial, : min(100, neural.shape[1])].copy(),
        "wheel_continuous": wheel_continuous[trial].copy(),
        "motion_continuous": motion_continuous[trial].copy(),
        "wheel_categories": wheel_categories[trial].copy(),
        "motion_categories": motion_categories[trial].copy(),
        "choice": int(choices[trial]),
        "prior": int(priors[trial]),
        "block_number": float(block_numbers[trial]),
        "wheel_thresholds": wheel_thresholds.copy(),
        "motion_thresholds": motion_thresholds.copy(),
        "wheel_hist": wheel_continuous[np.isfinite(wheel_continuous)][::10].copy(),
        "motion_hist": motion_continuous[np.isfinite(motion_continuous)][::10].copy(),
        "wheel_raw_t": np.asarray(wheel_aux["times"])[wheel_idx] - stim,
        "wheel_raw_v": np.asarray(wheel_aux["values"])[wheel_idx],
        "motion_raw_t": np.asarray(motion_aux["times"])[motion_idx] - stim,
        "motion_raw_v": np.asarray(motion_aux["values"])[motion_idx],
        "motion_view": motion_aux["view"],
        "raw_trials": raw_trials,
        "code_valid_trials": code_valid_trials,
        "retained_trials": len(source_indices),
    }


def process_session(index: int, info: dict, want_plot: bool = False) -> dict:
    started = time.perf_counter()
    try:
        path = session_path(pd.Series(info))
        alf = path / "alf"
        trials, trial_path = load_trials(info)
        code_mask = trial_mask(trials)
        code_indices = np.flatnonzero(code_mask)
        if len(code_indices) < 2:
            raise ValueError(f"Only {len(code_indices)} code-valid trials")

        raw_block_numbers = trial_number_in_block(trials["probabilityLeft"].to_numpy())
        stim_code = trials["stimOn_times"].to_numpy(dtype=float)[code_indices]

        # Retain only windows covered by every recorded probe.  Without this check,
        # trailing behavioral trials can be represented as population-wide zero
        # activity after an electrophysiology recording has already stopped.
        probes = [load_probe_metadata(alf, name) for name in info["probe_names"]]
        probe_starts: list[float] = []
        probe_ends: list[float] = []
        for probe in probes:
            spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
            if len(spike_times) == 0:
                raise ValueError(f"Probe has no spikes: {probe['directory']}")
            probe_starts.append(float(spike_times[0]))
            probe_ends.append(float(spike_times[-1]))
        neural_coverage_start = max(probe_starts)
        neural_coverage_end = min(probe_ends)
        neural_good = (
            (stim_code + OFF_START >= neural_coverage_start)
            & (stim_code + OFF_END <= neural_coverage_end)
        )

        t0 = time.perf_counter()
        wheel_values, wheel_good, wheel_aux = load_wheel_samples(alf, stim_code)
        motion_values, motion_good, motion_aux = load_motion_samples(alf, stim_code)
        stream_good = wheel_good & motion_good & neural_good
        keep_in_code = np.flatnonzero(stream_good)
        source_indices = code_indices[keep_in_code]
        if len(source_indices) < 2:
            raise ValueError(f"Only {len(source_indices)} trials after stream coverage")
        wheel_values = wheel_values[keep_in_code]
        motion_values = motion_values[keep_in_code]
        behavior_seconds = time.perf_counter() - t0

        wheel_categories, wheel_thresholds, wheel_imputed, wheel_median = discretize_tertiles(
            wheel_values
        )
        motion_categories, motion_thresholds, motion_imputed, motion_median = discretize_tertiles(
            motion_values
        )

        raw_choice = trials["choice"].to_numpy()[source_indices]
        if not np.all(np.isin(raw_choice, [-1, 1])):
            raise ValueError("Unexpected retained choice value")
        # IBL convention: +1 is a leftward choice and -1 is rightward.
        # Target convention required here: left=0, right=1.
        choices = (raw_choice == -1).astype(np.int8)
        raw_prior = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
        priors = np.full(len(raw_prior), -1, dtype=np.int8)
        for value, category in ((0.2, 0), (0.5, 1), (0.8, 2)):
            priors[np.isclose(raw_prior, value, rtol=0.0, atol=1e-8)] = category
        if np.any(priors < 0):
            raise ValueError(f"Unexpected probabilityLeft values: {np.unique(raw_prior[priors < 0])}")
        block_numbers = raw_block_numbers[source_indices]
        stim_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
        starts = stim_times + OFF_START
        ends = stim_times + OFF_END

        t1 = time.perf_counter()
        n_neurons = sum(p["n_clusters"] for p in probes)
        neural = np.zeros((len(source_indices), n_neurons, N_BINS), dtype=np.float32)
        all_regions: list[np.ndarray] = []
        offset = 0
        for probe in probes:
            n = probe["n_clusters"]
            neural[:, offset:offset + n] = bin_probe(probe, starts, ends)
            all_regions.append(probe["regions"])
            offset += n
        region_names = np.concatenate(all_regions).astype(str)
        neural_seconds = time.perf_counter() - t1

        time_input = np.broadcast_to(REL_SAMPLE_TIMES, (len(source_indices), N_BINS))
        block_input = np.broadcast_to(block_numbers[:, None], (len(source_indices), N_BINS))
        inputs = np.stack((time_input, block_input), axis=1).astype(np.float32, copy=True)
        choice_output = np.broadcast_to(choices[:, None], (len(source_indices), N_BINS))
        prior_output = np.broadcast_to(priors[:, None], (len(source_indices), N_BINS))
        outputs = np.stack(
            (choice_output, prior_output, wheel_categories, motion_categories), axis=1,
        ).astype(np.int8, copy=False)

        if neural.shape != (len(source_indices), n_neurons, N_BINS):
            raise AssertionError("Unexpected neural shape")
        if inputs.shape != (len(source_indices), 2, N_BINS):
            raise AssertionError("Unexpected input shape")
        if outputs.shape != (len(source_indices), 4, N_BINS):
            raise AssertionError("Unexpected output shape")
        if not np.all(np.isfinite(neural)) or not np.all(np.isfinite(inputs)):
            raise ValueError("Non-finite neural/input values")
        if outputs.min() < 0 or outputs.max() > 2:
            raise ValueError("Output category outside expected range")

        plot_payload = None
        if want_plot:
            plot_payload = make_plot_payload(
                info["eid"], source_indices, neural, wheel_values, motion_values,
                wheel_categories, motion_categories, choices, priors, block_numbers,
                wheel_thresholds, motion_thresholds, wheel_aux, motion_aux, stim_times,
                len(trials), int(code_mask.sum()),
            )

        session_info = {
            "eid": info["eid"],
            "pids": info["pids"],
            "probe_names": info["probe_names"],
            "lab": info["lab"],
            "subject": info["subject"],
            "date": info["date"],
            "session_number": info["session_number"],
            "source_path": str(path),
            "trial_table_path": str(trial_path),
            "whisker_camera": motion_aux["view"],
            "n_raw_trials": int(len(trials)),
            "n_code_valid_trials": int(code_mask.sum()),
            "n_retained_trials": int(len(source_indices)),
            "n_neural_coverage_excluded_trials": int(np.count_nonzero(~neural_good)),
            "source_trial_indices": source_indices.astype(int).tolist(),
            "n_neurons": int(n_neurons),
            "n_good_label_clusters": int(sum(p["n_good"] for p in probes)),
            "wheel_tertiles": wheel_thresholds.tolist(),
            "whisker_tertiles": motion_thresholds.tolist(),
            "wheel_imputed_samples": wheel_imputed,
            "whisker_imputed_samples": motion_imputed,
            "wheel_imputation_median": wheel_median,
            "whisker_imputation_median": motion_median,
            "neural_coverage_start_s": neural_coverage_start,
            "neural_coverage_end_s": neural_coverage_end,
            "behavior_processing_seconds": behavior_seconds,
            "neural_processing_seconds": neural_seconds,
        }
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "index": index,
            "info": session_info,
            "neural": [neural[i] for i in range(len(neural))],
            "input": [inputs[i] for i in range(len(inputs))],
            "output": [outputs[i] for i in range(len(outputs))],
            "region_names": region_names,
            "plot": plot_payload,
            "elapsed": elapsed,
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": index,
            "eid": info["eid"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed": time.perf_counter() - started,
        }


def save_processing_plot(payload: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = REL_SAMPLE_TIMES
    fig, axes = plt.subplots(4, 2, figsize=(16, 15))
    ax = axes.ravel()
    image = ax[0].imshow(
        payload["neural"], aspect="auto", interpolation="nearest",
        extent=[t[0], t[-1], payload["neural"].shape[0], 0], cmap="viridis",
    )
    ax[0].axvline(0, color="w", linestyle="--", linewidth=1)
    ax[0].set(title="20-ms spike counts (first 100 neurons)", xlabel="time from stimulus (s)", ylabel="neuron")
    fig.colorbar(image, ax=ax[0], label="spikes/bin")

    ax[1].plot(payload["wheel_raw_t"], payload["wheel_raw_v"], color="0.6", label="1-kHz filtered speed")
    ax[1].plot(t, payload["wheel_continuous"], ".-", label="20-ms right-edge samples")
    ax[1].axvline(0, color="k", linestyle="--")
    ax[1].set(title="Wheel processing and alignment", xlabel="time from stimulus (s)", ylabel="speed")
    ax[1].legend()

    ax[2].plot(payload["motion_raw_t"], payload["motion_raw_v"], color="0.6", label=f"{payload['motion_view']} camera raw")
    ax[2].plot(t, payload["motion_continuous"], ".-", label="20-ms interpolated")
    ax[2].axvline(0, color="k", linestyle="--")
    ax[2].set(title="Whisker motion-energy processing", xlabel="time from stimulus (s)", ylabel="motion energy")
    ax[2].legend()

    ax[3].hist(payload["wheel_hist"], bins=80, color="C0", alpha=0.8)
    for threshold in payload["wheel_thresholds"]:
        ax[3].axvline(threshold, color="k", linestyle="--")
    ax[3].set(title="Wheel tertile discretization", xlabel="speed", ylabel="samples")

    ax[4].hist(payload["motion_hist"], bins=80, color="C1", alpha=0.8)
    for threshold in payload["motion_thresholds"]:
        ax[4].axvline(threshold, color="k", linestyle="--")
    ax[4].set(title="Whisker tertile discretization", xlabel="motion energy", ylabel="samples")

    ax[5].step(t, payload["wheel_categories"], where="mid", label="wheel class")
    ax[5].step(t, payload["motion_categories"] + 3, where="mid", label="whisker class (+3)")
    ax[5].axvline(0, color="k", linestyle="--")
    ax[5].set(title="Time-varying categorical outputs", xlabel="time from stimulus (s)", yticks=range(6))
    ax[5].legend()

    ax[6].plot(t, t, label="time input")
    ax[6].plot(t, np.full_like(t, payload["block_number"]), label="trial-in-block input")
    ax[6].axvline(0, color="k", linestyle="--")
    ax[6].set(title="Decoder inputs", xlabel="time from stimulus (s)")
    ax[6].legend()

    ax[7].axis("off")
    ax[7].text(
        0.02, 0.98,
        "\n".join([
            f"EID: {payload['eid']}",
            f"source trial: {payload['source_trial']}",
            f"choice class: {payload['choice']} (0 left, 1 right)",
            f"prior class: {payload['prior']} (0=.2, 1=.5, 2=.8)",
            f"raw trials: {payload['raw_trials']}",
            f"code-valid trials: {payload['code_valid_trials']}",
            f"stream-valid retained: {payload['retained_trials']}",
            f"neural shape shown: {payload['neural'].shape}",
            "Window: [-0.5, +1.5) s; 20 ms bins",
            "Behavior samples label the preceding spike-count bin end.",
        ]),
        va="top", family="monospace",
    )
    fig.suptitle(f"Conversion processing audit: {payload['eid']}")
    fig.tight_layout()
    output = APP / f"processing_{payload['eid']}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def assemble_dataset(results: list[dict], skipped: list[dict]) -> dict:
    results = sorted(results, key=lambda r: r["index"])
    subjects = sorted({r["info"]["subject"] for r in results})
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    brain_regions = sorted({str(region) for r in results for region in r["region_names"]})
    region_lookup = {region: i for i, region in enumerate(brain_regions)}

    neural = [r["neural"] for r in results]
    inputs = [r["input"] for r in results]
    outputs = [r["output"] for r in results]
    subject_idx = np.asarray(
        [subject_lookup[r["info"]["subject"]] for r in results], dtype=np.int32,
    )
    brain_region_idx = [
        np.asarray([region_lookup[name] for name in r["region_names"]], dtype=np.int32)
        for r in results
    ]

    total_trials = sum(len(x) for x in neural)
    total_neurons = sum(x[0].shape[0] for x in neural)
    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_since_stimulus_onset_s", "trial_number_in_block"],
        "output_names": [
            "choice", "prior_probability_left", "wheel_speed", "whisker_motion_energy",
        ],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL visual two-alternative choice task; decode left/right choice, block "
                "probabilityLeft, and tertile-discretized wheel speed and whisker-pad motion energy."
            ),
            "time_bin_size": 20.0,
            "time_bin_size_units": "ms",
            "temporal_alignment_event": "visual stimulus onset (stimOn_times)",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "n_timepoints": N_BINS,
            "neural_representation": "raw spike counts in half-open 20-ms bins",
            "behavior_sample_convention": "right edge of each neural time bin",
            "trial_filter": (
                "finite stimOn/choice/feedback/probabilityLeft/firstMovement/feedbackType; "
                "choice != 0; stimulus-to-first-movement 0.08-2.00 s inclusive; "
                "goCue-to-feedback <=10 s; valid wheel and whisker window coverage"
            ),
            "neuron_filter": "all Kilosort clusters (qc=None), matching decoder reference code",
            "brain_region_mapping": "Allen CCF cluster channel -> Beryl acronym",
            "dynamic_output_discretization": (
                "within-session empirical 1/3 and 2/3 quantiles over retained finite samples"
            ),
            "trial_number_in_block_definition": (
                "zero-based index in contiguous raw probabilityLeft run, computed before trial filtering"
            ),
            "source_release": "code_zhang2025/data/bwm_release.csv; staged Brainwidemap ALF data",
            "session_info": [r["info"] for r in results],
            "skipped_sessions": skipped,
            "summary": {
                "n_sessions": len(results),
                "n_subjects": len(subjects),
                "n_trials": total_trials,
                "sum_neurons_across_sessions": total_neurons,
            },
        },
    }
    return data


def main() -> None:
    args = parse_args()
    overall = time.perf_counter()
    freeze, sessions = load_freeze()
    print(
        f"Freeze inventory: {len(sessions)} sessions, {len(freeze)} probes, "
        f"{freeze['subject'].nunique()} subjects", flush=True,
    )
    if len(sessions) != 459 or len(freeze) != 699 or freeze["subject"].nunique() != 139:
        raise RuntimeError("Freeze identity does not match the paper release (459/699/139)")

    if args.sample:
        selected = sessions[:2]
    else:
        selected = sessions
    workers_default = 2 if args.sample else min(24, os.cpu_count() or 1)
    workers = max(1, int(os.environ.get("CONVERSION_WORKERS", workers_default)))
    print(f"Mode: {'sample' if args.sample else 'full'}; workers={workers}", flush=True)

    results: list[dict] = []
    skipped: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_session, i, info, args.show_processing and i < 2): (i, info)
            for i, info in enumerate(selected)
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result["ok"]:
                results.append(result)
                print(
                    f"[{completed}/{len(selected)}] {result['info']['eid']}: "
                    f"{result['info']['n_retained_trials']} trials, "
                    f"{result['info']['n_neurons']} neurons, {result['elapsed']:.1f}s "
                    f"(behavior {result['info']['behavior_processing_seconds']:.1f}s, "
                    f"neural {result['info']['neural_processing_seconds']:.1f}s)",
                    flush=True,
                )
            else:
                skipped.append({
                    "eid": result["eid"], "reason": result["error"],
                    "freeze_index": result["index"],
                })
                print(
                    f"[{completed}/{len(selected)}] SKIP {result['eid']}: {result['error']}",
                    flush=True,
                )
                print(result["traceback"], flush=True)

    if args.sample and len(results) != 2:
        raise RuntimeError(f"Sample mode requires 2 successful sessions, got {len(results)}")
    if not results:
        raise RuntimeError("No sessions converted")

    if args.show_processing:
        for result in sorted(results, key=lambda r: r["index"])[:2]:
            if result["plot"] is not None:
                path = save_processing_plot(result["plot"])
                print(f"Saved processing plot: {path}", flush=True)
            result["plot"] = None

    print("Assembling global subject/region indices...", flush=True)
    data = assemble_dataset(results, skipped)
    print(
        f"Converted: {len(data['neural'])} sessions, {len(data['subjects'])} subjects, "
        f"{sum(map(len, data['neural']))} trials, "
        f"{sum(s[0].shape[0] for s in data['neural'])} session-neurons, "
        f"{len(data['brain_regions'])} Beryl regions; skipped {len(skipped)} sessions",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    print(f"Writing {args.output} ...", flush=True)
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, args.output)
    size_gb = args.output.stat().st_size / 1e9
    elapsed = time.perf_counter() - overall
    print(f"Wrote {args.output} ({size_gb:.3f} GB) in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
