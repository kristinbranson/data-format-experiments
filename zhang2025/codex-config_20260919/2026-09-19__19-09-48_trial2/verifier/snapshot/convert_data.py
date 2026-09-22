#!/usr/bin/env python3
"""Convert the IBL Brain Wide Map release to the requested decoder format.

Usage
-----
python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
from iblatlas.regions import BrainRegions


APP = Path("/app")
DATA_ROOT = APP / "data" / "one_cache"
FREEZE = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)


@dataclass
class BehaviorStream:
    times: np.ndarray
    values: np.ndarray
    source: str


def preferred(paths) -> Path | None:
    """Choose the newest explicit revision, or the unrevised file if unique."""
    paths = list(paths)
    if not paths:
        return None
    return sorted(paths, key=lambda p: ("#" in str(p), str(p)))[-1]


def session_path(row: pd.Series) -> Path:
    return (
        DATA_ROOT
        / str(row.lab)
        / "Subjects"
        / str(row.subject)
        / str(row.date)
        / f"{int(row.session_number):03d}"
    )


def load_trials(path: Path) -> tuple[pd.DataFrame, Path]:
    trial_file = preferred(path.glob("alf/**/_ibl_trials.table.pqt"))
    if trial_file is None:
        raise FileNotFoundError(f"No trial table under {path}")
    return pd.read_parquet(trial_file), trial_file


def reference_trial_mask(trials: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Reproduce the supplied load_trials_and_mask call used by prepare_data."""
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
    ]
    present = np.ones(len(trials), dtype=bool)
    for name in required:
        present &= trials[name].notna().to_numpy()
    rt = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    rt_ok = (rt >= 0.08) & (rt <= 2.00)
    duration = (trials["feedback_times"] - trials["goCue_times"]).to_numpy()
    duration_ok = duration <= 10.0
    choice_ok = trials["choice"].to_numpy() != 0
    mask = present & rt_ok & duration_ok & choice_ok
    audit = {
        "native": int(len(trials)),
        "required_events": int(present.sum()),
        "reaction_time": int((present & rt_ok).sum()),
        "duration": int((present & rt_ok & duration_ok).sum()),
        "choice": int(mask.sum()),
    }
    return mask, audit


def block_trial_numbers(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based native trial number within each probability block."""
    out = np.zeros(len(probability_left), dtype=np.float32)
    count = 0
    for i in range(1, len(probability_left)):
        same = np.isfinite(probability_left[i]) and np.isfinite(probability_left[i - 1])
        same = same and np.isclose(probability_left[i], probability_left[i - 1])
        count = count + 1 if same else 0
        out[i] = count
    return out


def load_motion_energy(path: Path) -> BehaviorStream | None:
    """Match SessionLoader correction and reference left-first/right-fallback rule."""
    for side in ("left", "right"):
        values_path = preferred(path.glob(f"alf/**/{side}Camera.ROIMotionEnergy.npy"))
        times_path = preferred(path.glob(f"alf/**/_ibl_{side}Camera.times.npy"))
        if values_path is None or times_path is None:
            continue
        values = np.asarray(np.load(values_path, mmap_mode="r"), dtype=np.float64)
        times = np.asarray(np.load(times_path, mmap_mode="r"), dtype=np.float64)
        if len(times) < len(values) or len(values) < 2:
            continue
        if len(times) > len(values):
            times = times[-len(values) :]
        if not np.all(np.diff(times) > 0):
            continue
        return BehaviorStream(times, values, side)
    return None


def load_wheel_speed(path: Path) -> tuple[BehaviorStream, np.ndarray, np.ndarray]:
    times_path = preferred(path.glob("alf/**/_ibl_wheel.timestamps.npy"))
    pos_path = preferred(path.glob("alf/**/_ibl_wheel.position.npy"))
    if times_path is None or pos_path is None:
        raise FileNotFoundError(f"Missing wheel object under {path}")
    raw_times = np.asarray(np.load(times_path, mmap_mode="r"), dtype=np.float64)
    raw_pos = np.asarray(np.load(pos_path, mmap_mode="r"), dtype=np.float64)
    if len(raw_times) != len(raw_pos) or len(raw_times) < 20:
        raise ValueError("Invalid wheel position/timestamp dimensions")
    pos_1khz, times_1khz = interpolate_position(raw_times, raw_pos, freq=1000)
    velocity, _ = velocity_filtered(pos_1khz, fs=1000, corner_frequency=20, order=8)
    return (
        BehaviorStream(times_1khz, np.abs(velocity), "wheel"),
        raw_times,
        raw_pos,
    )


def coverage_mask(stream: BehaviorStream, begins: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Reference coverage checks used before interpolation."""
    ib = np.searchsorted(stream.times, begins, side="right")
    ie = np.searchsorted(stream.times, ends, side="left")
    ok = (ie - ib) >= 2
    first = np.full(len(begins), np.nan)
    last = np.full(len(begins), np.nan)
    idx = np.flatnonzero(ok)
    first[idx] = stream.times[ib[idx]]
    last[idx] = stream.times[ie[idx] - 1]
    ok &= np.abs(begins - first) <= BIN_SIZE
    ok &= np.abs(ends - last) <= BIN_SIZE
    return ok


def interpolate_trials(
    stream: BehaviorStream, begins: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Fast equivalent of reference per-interval scipy interp1d + extrapolation."""
    out = np.empty((len(begins), N_BINS), dtype=np.float32)
    steps = BIN_SIZE * np.arange(1, N_BINS + 1)
    ibs = np.searchsorted(stream.times, begins, side="right")
    ies = np.searchsorted(stream.times, ends, side="left")
    for i, (beg, end, ib, ie) in enumerate(zip(begins, ends, ibs, ies)):
        t = stream.times[ib:ie]
        v = stream.values[ib:ie]
        x = beg + steps
        y = np.interp(x, t, v)
        # scipy.interpolate.interp1d(fill_value='extrapolate') in the reference
        # extrapolates the final right-edge sample from the last two in-window points.
        slope = (v[-1] - v[-2]) / (t[-1] - t[-2])
        y[-1] = v[-1] + (end - t[-1]) * slope
        out[i] = y
    if not np.all(np.isfinite(out)):
        raise ValueError(f"Non-finite interpolated values in {stream.source}")
    return out


def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    flat = values.ravel()
    thresholds = np.quantile(flat, [1 / 3, 2 / 3])
    if thresholds[0] < thresholds[1]:
        labels = np.searchsorted(thresholds, values, side="right").astype(np.int64)
        return labels, thresholds.astype(float), "value_quantiles"
    # Deterministic equal-frequency fallback for a degenerate/tied signal.
    order = np.argsort(flat, kind="stable")
    ranked = np.empty(flat.size, dtype=np.int64)
    ranked[order] = np.minimum(2, (np.arange(flat.size) * 3) // flat.size)
    return ranked.reshape(values.shape), thresholds.astype(float), "stable_rank_quantiles"


def load_probe_units(probe_path: Path, brain_regions: BrainRegions) -> dict:
    metrics_path = preferred(probe_path.glob("**/clusters.metrics.pqt"))
    channels_path = preferred(probe_path.glob("**/clusters.channels.npy"))
    atlas_path = preferred(probe_path.glob("electrodeSites.brainLocationIds_ccf_2017.npy"))
    if atlas_path is None:
        atlas_path = preferred(probe_path.glob("**/channels.brainLocationIds_ccf_2017.npy"))
    spike_times_path = preferred(probe_path.glob("**/spikes.times.npy"))
    spike_clusters_path = preferred(probe_path.glob("**/spikes.clusters.npy"))
    required = [metrics_path, channels_path, atlas_path, spike_times_path, spike_clusters_path]
    if any(x is None for x in required):
        raise FileNotFoundError(f"Incomplete spike sorting under {probe_path}")

    metrics = pd.read_parquet(metrics_path)
    good_rows = np.flatnonzero(metrics["label"].to_numpy() >= 1)
    cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)[good_rows]
    cluster_channels = np.load(channels_path, mmap_mode="r")
    atlas_ids = np.load(atlas_path, mmap_mode="r")
    peak_channels = np.asarray(cluster_channels[good_rows], dtype=np.int64)
    if np.any(peak_channels < 0) or np.any(peak_channels >= len(atlas_ids)):
        raise ValueError(f"Peak-channel index outside atlas array in {probe_path}")
    allen = brain_regions.id2acronym(np.asarray(atlas_ids[peak_channels], dtype=np.int64))
    beryl = brain_regions.acronym2acronym(allen, mapping="Beryl").astype(str)
    return {
        "raw_units": int(len(metrics)),
        "good_rows": good_rows,
        "cluster_ids": cluster_ids,
        "regions": beryl,
        "times_path": spike_times_path,
        "clusters_path": spike_clusters_path,
    }


def bin_probe(
    info: dict, begins: np.ndarray, ends: np.ndarray, destination: np.ndarray
) -> None:
    """Bin one probe directly from sorted spike memmaps into a trial×unit×time view."""
    times = np.load(info["times_path"], mmap_mode="r")
    clusters = np.load(info["clusters_path"], mmap_mode="r")
    good_ids = np.asarray(info["cluster_ids"], dtype=np.int64)
    max_id = int(max(len(info["good_rows"]) + info["raw_units"], good_ids.max(initial=0) + 1))
    remap = np.full(max_id, -1, dtype=np.int32)
    remap[good_ids] = np.arange(len(good_ids), dtype=np.int32)
    left = np.searchsorted(times, begins, side="left")
    right = np.searchsorted(times, ends, side="left")
    width = destination.shape[1] * N_BINS
    for trial, (beg, lo, hi) in enumerate(zip(begins, left, right)):
        raw_c = np.asarray(clusters[lo:hi], dtype=np.int64)
        in_range = raw_c < len(remap)
        mapped = np.full(len(raw_c), -1, dtype=np.int32)
        mapped[in_range] = remap[raw_c[in_range]]
        keep = mapped >= 0
        if not np.any(keep):
            continue
        rel_t = np.asarray(times[lo:hi], dtype=np.float64)[keep] - beg
        bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)
        valid = (bins >= 0) & (bins < N_BINS)
        flat = mapped[keep][valid].astype(np.int64) * N_BINS + bins[valid]
        destination[trial] = np.bincount(flat, minlength=width).reshape(
            destination.shape[1], N_BINS
        )


def plot_processing(
    eid: str,
    trials: pd.DataFrame,
    trial_indices: np.ndarray,
    audit: dict[str, int],
    raw_wheel_times: np.ndarray,
    raw_wheel_pos: np.ndarray,
    wheel_cont: np.ndarray,
    motion_cont: np.ndarray,
    wheel_labels: np.ndarray,
    motion_labels: np.ndarray,
    thresholds: dict,
    neural_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    motion_side: str,
) -> None:
    """Visual audit of loading, filtering, alignment, binning and discretization."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    ax = axes.ravel()
    stages = list(audit)
    ax[0].bar(stages, [audit[k] for k in stages])
    ax[0].tick_params(axis="x", rotation=35)
    ax[0].set_title("Trial curation by stage")
    ax[0].set_ylabel("trials retained")

    ti = 0
    raw_i = int(trial_indices[ti])
    stim = float(trials.iloc[raw_i].stimOn_times)
    view = (raw_wheel_times >= stim + OFF_START) & (raw_wheel_times <= stim + OFF_END)
    ax[1].plot(raw_wheel_times[view] - stim, raw_wheel_pos[view], lw=0.8)
    ax[1].axvline(0, color="k", ls="--")
    ax[1].set_title("Raw wheel position around stimulus")
    ax[1].set_xlabel("time from stimulus (s)")

    ax[2].plot(TIME_GRID, wheel_cont[ti], label="speed")
    ax[2].step(TIME_GRID, wheel_labels[ti], where="mid", label="class")
    ax[2].axvline(0, color="k", ls="--")
    ax[2].legend()
    ax[2].set_title("Wheel processing/alignment")

    ax[3].plot(TIME_GRID, motion_cont[ti], label=f"{motion_side} ME")
    ax[3].step(TIME_GRID, motion_labels[ti], where="mid", label="class")
    ax[3].axvline(0, color="k", ls="--")
    ax[3].legend()
    ax[3].set_title("Whisker ME processing/alignment")

    ax[4].hist(wheel_cont.ravel(), bins=80, alpha=0.8)
    for q in thresholds["wheel"]:
        ax[4].axvline(q, color="r", ls="--")
    ax[4].set_title("Wheel tertile thresholds")
    ax[5].hist(motion_cont.ravel(), bins=80, alpha=0.8)
    for q in thresholds["motion_energy"]:
        ax[5].axvline(q, color="r", ls="--")
    ax[5].set_title("Whisker tertile thresholds")

    nshow = min(50, neural_trials[ti].shape[0])
    ax[6].imshow(neural_trials[ti][:nshow], aspect="auto", interpolation="nearest", extent=[OFF_START, OFF_END, nshow, 0])
    ax[6].axvline(0, color="w", ls="--")
    ax[6].set_title("Binned good-unit spike counts")
    ax[6].set_xlabel("time from stimulus (s)")

    ax[7].imshow(output_trials[ti], aspect="auto", interpolation="nearest", extent=[OFF_START, OFF_END, 4, 0])
    ax[7].axvline(0, color="w", ls="--")
    ax[7].set_yticks(np.arange(0.5, 4, 1), ["choice", "prior", "wheel", "whisker"])
    ax[7].set_title("Final categorical outputs")

    mean_rate = np.mean(np.stack(neural_trials), axis=(0, 1)) / BIN_SIZE
    ax[8].plot(TIME_GRID, mean_rate, label="mean population rate")
    ax[8].axvline(0, color="k", ls="--", label="stimulus")
    ax[8].set_title("Session trial/unit mean alignment")
    ax[8].set_xlabel("time from stimulus (s)")
    ax[8].set_ylabel("spikes/s")
    ax[8].legend()
    fig.suptitle(f"Conversion processing audit: {eid}")
    fig.tight_layout()
    fig.savefig(APP / f"processing_{eid}.png", dpi=140)
    plt.close(fig)


def process_session(
    eid: str,
    rows: pd.DataFrame,
    brain_regions: BrainRegions,
    show_processing: bool,
) -> dict | None:
    t0 = time.perf_counter()
    first = rows.iloc[0]
    path = session_path(first)
    trials, trial_file = load_trials(path)
    base_mask, audit = reference_trial_mask(trials)
    begins_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_START
    ends_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_END

    motion = load_motion_energy(path)
    if motion is None:
        print(f"SKIP {eid}: no valid left/right whisker motion-energy stream", flush=True)
        return None
    motion_ok = coverage_mask(motion, begins_all, ends_all)

    wheel, raw_wheel_times, raw_wheel_pos = load_wheel_speed(path)
    wheel_ok = coverage_mask(wheel, begins_all, ends_all)
    valid = base_mask & motion_ok & wheel_ok
    trial_indices = np.flatnonzero(valid)
    audit["motion_coverage"] = int((base_mask & motion_ok).sum())
    audit["joint_coverage"] = int(valid.sum())
    if len(trial_indices) < 2:
        print(f"SKIP {eid}: {len(trial_indices)} jointly valid trials", flush=True)
        return None

    begins = begins_all[trial_indices]
    ends = ends_all[trial_indices]
    wheel_cont = interpolate_trials(wheel, begins, ends)
    motion_cont = interpolate_trials(motion, begins, ends)
    wheel_labels, wheel_q, wheel_method = discretize_tertiles(wheel_cont)
    motion_labels, motion_q, motion_method = discretize_tertiles(motion_cont)

    unit_infos = []
    raw_units = 0
    probe_names = []
    for _, row in rows.sort_values("probe_name").iterrows():
        probe_names.append(str(row.probe_name))
        probe_path = path / "alf" / str(row.probe_name)
        info = load_probe_units(probe_path, brain_regions)
        raw_units += info["raw_units"]
        info["probe_name"] = str(row.probe_name)
        unit_infos.append(info)
    n_units = sum(len(x["cluster_ids"]) for x in unit_infos)
    if n_units == 0:
        print(f"SKIP {eid}: no label>=1 units", flush=True)
        return None

    # uint16 is ample for a single unit's 20-ms count and halves peak construction memory.
    counts = np.zeros((len(trial_indices), n_units, N_BINS), dtype=np.uint16)
    region_labels = []
    unit_probe_names = []
    unit_cluster_ids = []
    offset = 0
    for info in unit_infos:
        n = len(info["cluster_ids"])
        bin_probe(info, begins, ends, counts[:, offset : offset + n, :])
        region_labels.extend(info["regions"].tolist())
        unit_probe_names.extend([info["probe_name"]] * n)
        unit_cluster_ids.extend(info["cluster_ids"].tolist())
        offset += n

    native_prob = trials["probabilityLeft"].to_numpy(dtype=float)
    block_number = block_trial_numbers(native_prob)[trial_indices]
    choice_native = trials["choice"].to_numpy(dtype=float)[trial_indices]
    choice = np.where(choice_native == -1, 0, 1).astype(np.int64)
    prior_native = native_prob[trial_indices]
    prior = np.full(len(trial_indices), -1, dtype=np.int64)
    for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
        prior[np.isclose(prior_native, value)] = label
    if np.any(prior < 0):
        raise ValueError(f"Unexpected probabilityLeft in {eid}")

    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    neural_trials: list[np.ndarray] = []
    for i in range(len(trial_indices)):
        inp = np.vstack(
            (TIME_GRID, np.full(N_BINS, block_number[i], dtype=np.float32))
        ).astype(np.float32, copy=False)
        out = np.vstack(
            (
                np.full(N_BINS, choice[i], dtype=np.int64),
                np.full(N_BINS, prior[i], dtype=np.int64),
                wheel_labels[i],
                motion_labels[i],
            )
        )
        neural_trials.append(counts[i].astype(np.float32))
        input_trials.append(inp)
        output_trials.append(out)

    thresholds = {"wheel": wheel_q, "motion_energy": motion_q}
    if show_processing:
        plot_processing(
            eid,
            trials,
            trial_indices,
            audit,
            raw_wheel_times,
            raw_wheel_pos,
            wheel_cont,
            motion_cont,
            wheel_labels,
            motion_labels,
            thresholds,
            neural_trials,
            output_trials,
            motion.source,
        )

    elapsed = time.perf_counter() - t0
    print(
        f"DONE {eid}: {len(trial_indices)}/{len(trials)} trials, "
        f"{n_units}/{raw_units} good/raw units, {elapsed:.2f}s",
        flush=True,
    )
    info = {
        "eid": eid,
        "subject": str(first.subject),
        "lab": str(first.lab),
        "date": str(first.date),
        "session_number": int(first.session_number),
        "probe_names": probe_names,
        "trial_table": str(trial_file.relative_to(APP)),
        "original_trial_indices": trial_indices.astype(np.int64),
        "motion_energy_camera": motion.source,
        "raw_unit_count": int(raw_units),
        "good_unit_count": int(n_units),
        "unit_probe_names": unit_probe_names,
        "unit_cluster_ids": np.asarray(unit_cluster_ids, dtype=np.int64),
        "unit_region_labels": list(region_labels),
        "trial_filter_counts": audit,
        "wheel_tertile_thresholds": wheel_q.tolist(),
        "wheel_discretization": wheel_method,
        "whisker_motion_energy_tertile_thresholds": motion_q.tolist(),
        "whisker_motion_energy_discretization": motion_method,
        "processing_seconds": elapsed,
    }
    return {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "region_labels": list(region_labels),
        "session_info": info,
    }


def release_preflight(freeze: pd.DataFrame) -> dict[str, int]:
    raw = 0
    good = 0
    for _, row in freeze.iterrows():
        probe = session_path(row) / "alf" / str(row.probe_name)
        metrics_path = preferred(probe.glob("**/clusters.metrics.pqt"))
        if metrics_path is None:
            raise FileNotFoundError(f"Missing metrics for release probe {row.pid}")
        metrics = pd.read_parquet(metrics_path, columns=["label"])
        raw += len(metrics)
        good += int((metrics["label"] >= 1).sum())
    expected = {"sessions": 459, "probes": 699, "raw_units": 621733, "good_units": 75708}
    observed = {
        "sessions": int(freeze.eid.nunique()),
        "probes": int(len(freeze)),
        "raw_units": int(raw),
        "good_units": int(good),
    }
    if observed != expected:
        raise AssertionError(f"Release preflight mismatch: {observed} != {expected}")
    print(f"Release preflight passed: {observed}", flush=True)
    return observed


def validate_before_save(data: dict) -> None:
    ns = len(data["neural"])
    assert ns == len(data["input"]) == len(data["output"])
    assert len(data["subject_idx"]) == ns == len(data["brain_region_idx"])
    for s in range(ns):
        nt = len(data["neural"][s])
        assert nt >= 2 and nt == len(data["input"][s]) == len(data["output"][s])
        assert len(data["brain_region_idx"][s]) == data["neural"][s][0].shape[0]
        for n, i, o in zip(data["neural"][s], data["input"][s], data["output"][s]):
            assert n.ndim == 2 and n.shape[1] == N_BINS
            assert i.shape == (2, N_BINS) and o.shape == (4, N_BINS)
            assert n.dtype == np.float32 and i.dtype == np.float32
            assert np.issubdtype(o.dtype, np.integer)
            assert np.all(np.isfinite(n)) and np.all(n >= 0)
            assert np.all(np.isfinite(i)) and np.all(np.isfinite(o))
        assert set(np.unique(np.concatenate([x[0] for x in data["output"][s]]))).issubset({0, 1})
        for d in (1, 2, 3):
            assert set(np.unique(np.concatenate([x[d] for x in data["output"][s]]))).issubset({0, 1, 2})


def convert(outfile: Path, sample: bool, show_processing: bool) -> None:
    started = time.perf_counter()
    freeze = pd.read_csv(FREEZE, index_col=0)
    release_stats = release_preflight(freeze)
    groups = list(freeze.groupby("eid", sort=False))
    brain_atlas = BrainRegions()
    converted = []
    if not sample and not show_processing:
        # Session processing is dominated by independent mounted-file reads plus
        # NumPy/SciPy kernels that release the GIL. A small thread pool overlaps
        # cold-cache latency without copying multi-megabyte session results through
        # multiprocessing pipes. executor.map preserves publication-freeze order.
        def run_group(group):
            eid, rows = group
            return process_session(str(eid), rows, brain_atlas, False)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(run_group, groups):
                if result is not None:
                    converted.append(result)
    else:
        for eid, rows in groups:
            want_plot = show_processing and len(converted) < 2
            result = process_session(str(eid), rows, brain_atlas, want_plot)
            if result is not None:
                converted.append(result)
                if sample and len(converted) == 2:
                    break

    if sample and len(converted) != 2:
        raise RuntimeError(f"Sample mode found only {len(converted)} usable sessions")
    if not converted:
        raise RuntimeError("No sessions were converted")

    subjects = list(dict.fromkeys(x["session_info"]["subject"] for x in converted))
    subject_lookup = {name: i for i, name in enumerate(subjects)}
    all_regions = sorted({r for x in converted for r in x["region_labels"]})
    region_lookup = {name: i for i, name in enumerate(all_regions)}
    brain_region_idx = [
        np.asarray([region_lookup[r] for r in x["region_labels"]], dtype=np.int64)
        for x in converted
    ]
    session_info = [x["session_info"] for x in converted]
    data = {
        "neural": [x["neural"] for x in converted],
        "input": [x["input"] for x in converted],
        "output": [x["output"] for x in converted],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[x["session_info"]["subject"]] for x in converted], dtype=np.int64
        ),
        "brain_regions": all_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_since_stimulus_onset", "trial_number_in_block"],
        "output_names": [
            "choice",
            "prior_probability_left",
            "wheel_speed",
            "whisker_motion_energy",
        ],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL visual two-alternative forced-choice task; decode choice, block prior, "
                "wheel-speed tertile, and whisker-motion-energy tertile from trial-aligned spikes."
            ),
            "time_bin_size": 20.0,
            "time_bin_size_units": "ms",
            "temporal_alignment_event": "visual stimulus onset (stimOn_times)",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "time_coordinate": "right edge of each half-open neural spike-count bin, seconds from stimulus onset",
            "neural_representation": "raw spike counts from label>=1 well-isolated units",
            "source_release": "IBL Brain Wide Map 2025 publication freeze (bwm_release.csv)",
            "release_preflight": release_stats,
            "trial_filter": (
                "required events; 0.08<=firstMovement-stimOn<=2.0 s; "
                "feedback-goCue<=10 s; nonzero choice; complete wheel and whisker coverage"
            ),
            "continuous_output_discretization": (
                "within-session tertiles over all retained 20-ms samples; right-edge search; stable-rank fallback on tied thresholds"
            ),
            "trial_number_in_block_definition": (
                "zero-based index in native probabilityLeft block, computed before trial filtering"
            ),
            "session_info": session_info,
            "conversion_mode": "sample" if sample else "full",
        },
    }
    validate_before_save(data)
    total_trials = sum(len(x) for x in data["neural"])
    total_units = sum(x[0].shape[0] for x in data["neural"])
    print(
        f"Validated converted structure: {len(converted)} sessions, {len(subjects)} subjects, "
        f"{total_trials} trials, {total_units} session-units, {len(all_regions)} Beryl regions",
        flush=True,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.perf_counter()
    with outfile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Saved {outfile} ({outfile.stat().st_size / 2**30:.3f} GiB) in "
        f"{time.perf_counter() - save_start:.2f}s; total {time.perf_counter() - started:.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all usable release sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process the first two usable sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<eid>.png diagnostics for up to two sessions",
    )
    args = parser.parse_args()
    convert(args.outpicklefile, sample=args.sample, show_processing=args.show_processing)


if __name__ == "__main__":
    main()
