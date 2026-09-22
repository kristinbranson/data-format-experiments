#!/usr/bin/env python3
"""Convert the Sosa et al. NWB files to the shared neural-decoder format.

The conversion intentionally starts from raw ROI and neuropil fluorescence.  It
reproduces the paper's trial-wise maximin dF/F and OASIS event extraction rather
than using the NWB Suite2P ``Deconvolved`` series, which is a different stage of
processing.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from suite2p.extraction import dcnv


DATA_ROOT = Path("/app/data")
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
NEUROPIL_COEF = 0.7
BASELINE_SMOOTH_SIGMA = 15.0
BASELINE_WINDOW_SAMPLES = 300
DFF_SMOOTH_SIGMA = 2.0
OASIS_TAU_SECONDS = 0.7
LICK_ERROR_FRACTION = 0.30
INTERNEURON_SPEED_R = 0.50

ZONE_BOUNDS = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}


def _decode(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else str(value)


def _session_sort_key(path: str) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", os.path.basename(path))
    if match is None:
        raise ValueError(f"Unexpected NWB filename: {path}")
    return int(match.group(1)), int(match.group(2))


def discover_files(sample: bool) -> list[str]:
    files = sorted(
        glob.glob(str(DATA_ROOT / "sub-*" / "*_behavior+ophys.nwb")),
        key=_session_sort_key,
    )
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    return files[:2] if sample else files


def parse_scene_zones(scene: str) -> list[str]:
    """Return chronological A/B/C zone labels encoded in a scene name."""
    labels = re.findall(r"(?:Location)?([ABC])(?=_to|$)", scene)
    if len(labels) not in (1, 2):
        raise ValueError(f"Could not parse one or two reward zones from {scene!r}")
    return labels


def zone_for_trial(labels: list[str], raw_trial_number: int) -> str:
    if len(labels) == 1 or raw_trial_number < 30:
        return labels[0]
    return labels[1]


def distance_classes(distance: np.ndarray) -> np.ndarray:
    """Discretize signed point-to-zone distance exactly as specified."""
    out = np.empty(distance.shape, dtype=np.int8)
    out[distance < -50.0] = 0
    out[(distance >= -50.0) & (distance < -10.0)] = 1
    out[(distance >= -10.0) & (distance < 0.0)] = 2
    out[distance == 0.0] = 3
    out[(distance > 0.0) & (distance <= 10.0)] = 4
    out[(distance > 10.0) & (distance <= 50.0)] = 5
    out[distance > 50.0] = 6
    return out


def position_classes(position: np.ndarray) -> np.ndarray:
    """Use explicit inequalities, including 360 cm in class 3 as specified."""
    out = np.empty(position.shape, dtype=np.int8)
    out[position < 90.0] = 0
    out[(position >= 90.0) & (position < 180.0)] = 1
    out[(position >= 180.0) & (position < 270.0)] = 2
    out[(position >= 270.0) & (position <= 360.0)] = 3
    out[position > 360.0] = 4
    return out


def speed_classes(speed: np.ndarray) -> np.ndarray:
    """Use explicit inequalities, including 40 cm/s in class 3 as specified."""
    out = np.empty(speed.shape, dtype=np.int8)
    out[speed < 2.0] = 0
    out[(speed >= 2.0) & (speed < 10.0)] = 1
    out[(speed >= 10.0) & (speed < 20.0)] = 2
    out[(speed >= 20.0) & (speed <= 40.0)] = 3
    out[speed > 40.0] = 4
    return out


def compute_dff_and_events(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    compute_events: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    """Reference-style processing for one trial, arrays neuron x time."""
    corrected = fluorescence - NEUROPIL_COEF * neuropil
    corrected += NEUROPIL_COEF * np.mean(neuropil, axis=1, keepdims=True)

    # The reference TwoPUtils call uses a 2-D sigma [0, 15], smoothing time only.
    baseline_seed = gaussian_filter(
        corrected, sigma=(0.0, BASELINE_SMOOTH_SIGMA), mode="reflect"
    )
    baseline = minimum_filter1d(
        baseline_seed, BASELINE_WINDOW_SAMPLES, axis=1, mode="reflect"
    )
    baseline = maximum_filter1d(
        baseline, BASELINE_WINDOW_SAMPLES, axis=1, mode="reflect"
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        dff_unsmoothed = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff_unsmoothed, DFF_SMOOTH_SIGMA, axis=1)

    events = None
    if compute_events:
        events = dcnv.oasis(
            np.asarray(dff, dtype=np.float32),
            2000,
            OASIS_TAU_SECONDS,
            FRAME_RATE_HZ,
        )
        events = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
        events = np.asarray(events, dtype=np.float32)

    trace = {
        "corrected": corrected,
        "baseline": baseline,
        "dff_unsmoothed": dff_unsmoothed,
        "dff": dff,
    }
    return dff, events, trace


def _update_corr_sums(
    sums: dict[str, np.ndarray | float | int], dff: np.ndarray, speed: np.ndarray
) -> None:
    finite = np.isfinite(dff)
    # Merge centered batch moments (parallel Welford). Direct raw sums lose
    # precision for long fluorescence traces and can yield impossible |r| > 1.
    x = np.where(finite, dff, 0.0).astype(np.float64, copy=False)
    y = np.asarray(speed, dtype=np.float64)[None, :]
    n_batch = finite.sum(axis=1).astype(np.float64)
    safe_n = np.maximum(n_batch, 1.0)
    mean_x_batch = x.sum(axis=1) / safe_n
    mean_y_batch = (finite * y).sum(axis=1) / safe_n
    dx_batch = np.where(finite, x - mean_x_batch[:, None], 0.0)
    dy_batch = np.where(finite, y - mean_y_batch[:, None], 0.0)
    m2x_batch = np.square(dx_batch).sum(axis=1)
    m2y_batch = np.square(dy_batch).sum(axis=1)
    cxy_batch = (dx_batch * dy_batch).sum(axis=1)

    n_old = sums["n"]
    n_new = n_old + n_batch
    valid = n_batch > 0
    cross_weight = np.zeros_like(n_new)
    cross_weight[valid] = n_old[valid] * n_batch[valid] / n_new[valid]
    delta_x = mean_x_batch - sums["mean_x"]
    delta_y = mean_y_batch - sums["mean_y"]
    sums["m2x"] += m2x_batch + np.square(delta_x) * cross_weight
    sums["m2y"] += m2y_batch + np.square(delta_y) * cross_weight
    sums["cxy"] += cxy_batch + delta_x * delta_y * cross_weight
    sums["mean_x"][valid] += delta_x[valid] * n_batch[valid] / n_new[valid]
    sums["mean_y"][valid] += delta_y[valid] * n_batch[valid] / n_new[valid]
    sums["n"] = n_new


def _finalize_corr(sums: dict[str, np.ndarray]) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = sums["cxy"] / np.sqrt(sums["m2x"] * sums["m2y"])
    # Tiny floating error at perfect correlation must not escape Pearson bounds.
    return np.clip(corr, -1.0, 1.0)


def _reward_outcomes(
    timestamps: np.ndarray,
    starts: np.ndarray,
    teleports: np.ndarray,
    reward_times: np.ndarray,
    reward_zone: np.ndarray,
) -> np.ndarray:
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        lo = np.searchsorted(reward_times, timestamps[start], side="left")
        hi = np.searchsorted(reward_times, timestamps[stop], side="left")
        outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
    return outcomes


def convert_session(path: str, show_processing: bool) -> dict:
    started = time.perf_counter()
    with h5py.File(path, "r") as nwb:
        subject = _decode(nwb["general/subject/subject_id"])
        day = int(_decode(nwb["general/session_id"]))
        scene = _decode(nwb["identifier"]).rstrip("/").split("/")[-1]
        session_id = f"{subject}_ses-{day:02d}"

        behavior = nwb["processing/behavior/BehavioralTimeSeries"]
        neural_group = nwb["processing/ophys"]
        fluorescence_ds = neural_group["Fluorescence/plane0/data"]
        neuropil_ds = neural_group["Neuropil/plane0/data"]
        common_length = min(
            fluorescence_ds.shape[0],
            *(behavior[name]["data"].shape[0] for name in behavior if name != "Reward"),
        )

        dense_names = [
            "position", "speed", "lick", "environment", "trial number",
            "reward_zone", "trial_start", "teleport",
        ]
        dense = {
            name: np.asarray(behavior[name]["data"][:common_length])
            for name in dense_names
        }
        timestamps = np.asarray(behavior["position/timestamps"][:common_length])
        if not all(
            np.allclose(
                behavior[name]["timestamps"][:common_length], timestamps,
                rtol=0.0, atol=1e-9,
            )
            for name in dense_names
        ):
            raise ValueError(f"Dense behavior timestamps differ in {session_id}")
        if not np.allclose(np.diff(timestamps), 1.0 / FRAME_RATE_HZ, atol=1e-9):
            raise ValueError(f"Nonuniform timestamps in {session_id}")

        starts = np.flatnonzero(dense["trial_start"] > 0)
        teleports = np.flatnonzero(dense["teleport"] > 0)
        if len(starts) != len(teleports) or not np.all(teleports > starts):
            raise ValueError(f"Unpaired or reversed trial bounds in {session_id}")

        labels = parse_scene_zones(scene)
        raw_trial_numbers = np.rint(dense["trial number"][starts]).astype(int)
        if len(np.unique(raw_trial_numbers)) != len(raw_trial_numbers):
            raise ValueError(f"Repeated raw trial numbers in {session_id}")
        outcomes = _reward_outcomes(
            timestamps,
            starts,
            teleports,
            np.asarray(behavior["Reward/timestamps"][:]),
            dense["reward_zone"],
        )
        lick_bad = np.array([
            np.mean(dense["lick"][start:stop] > 2) > LICK_ERROR_FRACTION
            for start, stop in zip(starts, teleports)
        ])

        segmentation = neural_group["ImageSegmentation/PlaneSegmentation"]
        plane_index = np.asarray(segmentation["planeIdx"][:])
        iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
        plane0_iscell = iscell[plane_index == 0]
        if fluorescence_ds.shape[1] != len(plane0_iscell):
            raise ValueError(
                f"plane0 response/segmentation mismatch in {session_id}: "
                f"{fluorescence_ds.shape[1]} vs {len(plane0_iscell)}"
            )
        roi_columns = np.flatnonzero(plane0_iscell)
        n_curated = len(roi_columns)

        corr_sums = {
            key: np.zeros(n_curated, dtype=np.float64)
            for key in ("n", "mean_x", "mean_y", "m2x", "m2y", "cxy")
        }
        kept_events: list[np.ndarray] = []
        kept_inputs: list[np.ndarray] = []
        kept_outputs: list[np.ndarray] = []
        kept_raw_trials: list[int] = []
        plot_payload = None

        for i, (start, stop) in enumerate(zip(starts, teleports)):
            # Reading the short contiguous row interval first is substantially
            # faster for contiguous NWB datasets than HDF5 fancy column reads.
            fluorescence = np.asarray(
                fluorescence_ds[start:stop, :], dtype=np.float32
            )[:, roi_columns].T
            neuropil = np.asarray(
                neuropil_ds[start:stop, :], dtype=np.float32
            )[:, roi_columns].T
            dff, events, processing_trace = compute_dff_and_events(
                fluorescence, neuropil, compute_events=not lick_bad[i]
            )
            speed = np.asarray(dense["speed"][start:stop], dtype=np.float32)
            _update_corr_sums(corr_sums, dff, speed)

            if lick_bad[i]:
                continue
            if events is None:
                raise AssertionError("Events unexpectedly missing for retained trial")

            position = np.asarray(dense["position"][start:stop], dtype=np.float32)
            lick = np.asarray(dense["lick"][start:stop])
            env_values = np.unique(dense["environment"][start:stop])
            if len(env_values) != 1 or env_values[0] not in (0, 1):
                raise ValueError(
                    f"Environment must be a constant 0/1 in {session_id} trial {i}: "
                    f"{env_values}"
                )
            raw_trial = int(raw_trial_numbers[i])
            zone_label = zone_for_trial(labels, raw_trial)
            zone_start, zone_end = ZONE_BOUNDS[zone_label]
            distance = position - np.clip(position, zone_start, zone_end)
            T = stop - start

            input_data = np.vstack((
                timestamps[start:stop] - timestamps[start],
                np.full(T, env_values[0]),
                np.full(T, raw_trial),
                np.full(T, outcomes[i - 1] if i > 0 else 0),
            )).astype(np.float32)
            output_data = np.vstack((
                distance_classes(distance),
                position_classes(position),
                speed_classes(speed),
                (lick > 0).astype(np.int8),
                np.full(T, ZONE_TO_CLASS[zone_label], dtype=np.int8),
                np.full(T, outcomes[i], dtype=np.int8),
            )).astype(np.int8)

            if events.shape[1] != T or input_data.shape != (4, T) or output_data.shape != (6, T):
                raise AssertionError(f"Trial shape mismatch in {session_id} trial {i}")
            kept_events.append(events)
            kept_inputs.append(input_data)
            kept_outputs.append(output_data)
            kept_raw_trials.append(raw_trial)

            if show_processing and plot_payload is None:
                example_neuron = int(np.nanargmax(np.nanmax(dff, axis=1)))
                plot_payload = {
                    "trial": raw_trial,
                    "neuron": example_neuron,
                    "fluorescence": fluorescence[example_neuron].copy(),
                    "neuropil": neuropil[example_neuron].copy(),
                    "processing": {k: v[example_neuron].copy() for k, v in processing_trace.items()},
                    "events": events[example_neuron].copy(),
                    "position": position.copy(),
                    "distance": distance.copy(),
                    "speed": speed.copy(),
                    "lick_raw": lick.copy(),
                    "input": input_data.copy(),
                    "output": output_data.copy(),
                    "zone": zone_label,
                }

        speed_corr = _finalize_corr(corr_sums)
        is_interneuron = np.isfinite(speed_corr) & (speed_corr > INTERNEURON_SPEED_R)
        neuron_keep = ~is_interneuron
        if np.count_nonzero(neuron_keep) == 0:
            raise ValueError(f"Interneuron filter removed every neuron in {session_id}")
        kept_events = [np.asarray(x[neuron_keep], dtype=np.float32) for x in kept_events]

        if len(kept_events) < 2:
            raise ValueError(f"Fewer than two retained trials in {session_id}")
        if any(not np.all(np.isfinite(x)) for x in kept_events + kept_inputs):
            raise ValueError(f"Nonfinite converted data in {session_id}")
        if any(np.min(x) < -1e-6 for x in kept_events):
            raise ValueError(f"Negative OASIS activity in {session_id}")

        elapsed = time.perf_counter() - started
        info = {
            "session_id": session_id,
            "source_file": os.path.relpath(path, "/app"),
            "subject": subject,
            "session_day": day,
            "scene": scene,
            "source_common_frames": int(common_length),
            "raw_trial_count": int(len(starts)),
            "kept_trial_count": int(len(kept_events)),
            "kept_raw_trial_numbers": kept_raw_trials,
            "dropped_bad_lick_trial_numbers": raw_trial_numbers[lick_bad].tolist(),
            "plane0_roi_candidates": int(fluorescence_ds.shape[1]),
            "manual_curated_cells": int(n_curated),
            "speed_correlated_cells_removed": int(np.count_nonzero(is_interneuron)),
            "dff_speed_correlation_range": [
                float(np.nanmin(speed_corr)), float(np.nanmax(speed_corr))
            ],
            "dff_speed_correlation_percentiles": np.nanpercentile(
                speed_corr, [1, 50, 99]
            ).tolist(),
            "retained_neurons": int(np.count_nonzero(neuron_keep)),
            "rewarded_raw_trials": int(outcomes.sum()),
            "processing_seconds": elapsed,
        }

    if show_processing and plot_payload is not None:
        make_processing_plot(session_id, plot_payload, speed_corr, is_interneuron)

    print(
        f"[{session_id}] {len(starts)} raw -> {len(kept_events)} trials; "
        f"{n_curated} curated -> {np.count_nonzero(neuron_keep)} neurons; "
        f"speed-r [{np.nanmin(speed_corr):.3f}, {np.nanmax(speed_corr):.3f}]; "
        f"{elapsed:.2f} s",
        flush=True,
    )
    return {
        "neural": kept_events,
        "input": kept_inputs,
        "output": kept_outputs,
        "subject": subject,
        "brain_region_idx": np.zeros(np.count_nonzero(neuron_keep), dtype=np.int64),
        "info": info,
    }


def make_processing_plot(
    session_id: str,
    payload: dict,
    speed_corr: np.ndarray,
    is_interneuron: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = payload["input"][0]
    fig, axes = plt.subplots(5, 2, figsize=(18, 18), sharex=False)
    ax = axes.ravel()
    ax[0].plot(t, payload["fluorescence"], label="ROI F", lw=1)
    ax[0].plot(t, 0.7 * payload["neuropil"], label="0.7 × neuropil", lw=1)
    ax[0].set_title("1. Raw fluorescence and neuropil")
    ax[0].legend()

    ax[1].plot(t, payload["processing"]["corrected"], label="corrected F", lw=1)
    ax[1].plot(t, payload["processing"]["baseline"], label="maximin baseline", lw=2)
    ax[1].set_title("2. Neuropil correction and trial-wise baseline")
    ax[1].legend()

    ax[2].plot(t, payload["processing"]["dff_unsmoothed"], label="raw dF/F", alpha=.6)
    ax[2].plot(t, payload["processing"]["dff"], label="2-sample smoothed", lw=2)
    ax[2].set_title("3. dF/F processing")
    ax[2].legend()

    ax[3].plot(t, payload["events"], color="black")
    ax[3].set_title("4. OASIS deconvolved events")

    ax[4].plot(t, payload["position"], label="position (cm)")
    zs, ze = ZONE_BOUNDS[payload["zone"]]
    ax[4].axhspan(zs, ze, color="gold", alpha=.25, label=f"zone {payload['zone']}")
    ax[4].set_title("5. Trial alignment and absolute position")
    ax[4].legend()

    ax[5].plot(t, payload["distance"], label="signed distance (cm)")
    ax[5].step(t, payload["output"][0] * 10, where="mid", label="distance class ×10")
    ax[5].axhline(0, color="gray", ls=":")
    ax[5].set_title("6. Reward-zone distance discretization")
    ax[5].legend()

    ax[6].plot(t, payload["speed"], label="speed (cm/s)")
    ax[6].step(t, payload["output"][2] * 10, where="mid", label="speed class ×10")
    ax[6].set_title("7. Speed discretization")
    ax[6].legend()

    ax[7].plot(t, payload["lick_raw"], label="raw cumulative count")
    ax[7].step(t, payload["output"][3], where="mid", label="binary lick")
    ax[7].set_title("8. Lick binarization")
    ax[7].legend()

    for row, name in enumerate(("time", "environment", "trial", "previous outcome")):
        ax[8].plot(t, payload["input"][row] + row * 2, label=name)
    ax[8].set_title("9. Four decoder inputs")
    ax[8].legend(ncol=2)

    finite_corr = speed_corr[np.isfinite(speed_corr)]
    ax[9].hist(finite_corr, bins=40, color="steelblue")
    ax[9].axvline(INTERNEURON_SPEED_R, color="red", ls="--")
    ax[9].set_title(
        f"10. dF/F–speed curation: {np.count_nonzero(is_interneuron)} removed"
    )
    ax[9].set_xlabel("Pearson r")

    for a in ax[:9]:
        a.set_xlabel("seconds from trial start")
    fig.suptitle(
        f"{session_id}, raw trial {payload['trial']}, example curated ROI {payload['neuron']}"
    )
    fig.tight_layout()
    fig.savefig(f"/app/processing_{session_id}.png", dpi=140)
    plt.close(fig)


def validate_converted(data: dict) -> None:
    nsessions = len(data["neural"])
    if not (nsessions == len(data["input"]) == len(data["output"])):
        raise AssertionError("Session counts differ")
    if len(data["subject_idx"]) != nsessions or len(data["brain_region_idx"]) != nsessions:
        raise AssertionError("Session metadata lengths differ")
    for s in range(nsessions):
        if len(data["neural"][s]) < 2:
            raise AssertionError(f"Session {s} has fewer than two trials")
        if not (len(data["neural"][s]) == len(data["input"][s]) == len(data["output"][s])):
            raise AssertionError(f"Trial counts differ in session {s}")
        n_neurons = len(data["brain_region_idx"][s])
        for n, x, y in zip(data["neural"][s], data["input"][s], data["output"][s]):
            if n.shape[0] != n_neurons or x.shape[0] != 4 or y.shape[0] != 6:
                raise AssertionError(f"Feature dimensions differ in session {s}")
            if not (n.shape[1] == x.shape[1] == y.shape[1]):
                raise AssertionError(f"Time dimensions differ in session {s}")
            if not np.allclose(x[0], np.arange(x.shape[1]) / FRAME_RATE_HZ, atol=2e-4):
                raise AssertionError(f"Time axis differs in session {s}")
            limits = (7, 5, 5, 2, 3, 2)
            for row, limit in enumerate(limits):
                if y[row].min() < 0 or y[row].max() >= limit:
                    raise AssertionError(f"Output {row} out of range in session {s}")


def build_dataset(files: list[str], show_processing: bool) -> dict:
    converted = []
    all_started = time.perf_counter()
    for i, path in enumerate(files):
        print(f"Converting session {i + 1}/{len(files)}: {path}", flush=True)
        converted.append(convert_session(path, show_processing and i < 2))

    subjects = sorted({x["subject"] for x in converted}, key=lambda s: int(s[1:]))
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    data = {
        "neural": [x["neural"] for x in converted],
        "input": [x["input"] for x in converted],
        "output": [x["output"] for x in converted],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[x["subject"]] for x in converted], dtype=np.int64
        ),
        "brain_regions": ["CA1"],
        "brain_region_idx": [x["brain_region_idx"] for x in converted],
        "input_names": [
            "time from trial start (s)",
            "environment (ENV1=0, ENV2=1)",
            "trial number (zero-based)",
            "previous trial outcome (omitted=0, rewarded=1)",
        ],
        "output_names": [
            "distance to reward zone",
            "absolute corridor position",
            "speed",
            "lick",
            "reward zone location",
            "reward outcome",
        ],
        "output_values": [
            ["< -50 cm", "-50 to < -10 cm", "-10 to < 0 cm", "inside zone (0 cm)",
             "> 0 to 10 cm", "> 10 to 50 cm", "> 50 cm"],
            ["< 90 cm", "90 to < 180 cm", "180 to < 270 cm", "270 to 360 cm", "> 360 cm"],
            ["< 2 cm/s", "2 to < 10 cm/s", "10 to < 20 cm/s", "20 to 40 cm/s", "> 40 cm/s"],
            ["no lick", "lick"],
            ["A", "B", "C"],
            ["omitted/no reward", "rewarded"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice navigate a 450 cm virtual corridor in ENV1 or ENV2, "
                "with a hidden reward zone at A, B, or C and approximately 15% reward omissions. "
                "The six categorical targets describe zone-relative distance, absolute position, "
                "speed, licking, reward-zone identity, and trial reward outcome."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": "entry into the 450 cm track (trial_start)",
            "off_start": 0.0,
            "off_end": None,
            "sampling_rate_hz": FRAME_RATE_HZ,
            "trial_window": "NWB rows [trial_start, teleport); teleport/ITI excluded",
            "neural_signal": "trial-wise maximin dF/F followed by OASIS deconvolved calcium events",
            "neural_processing": {
                "neuropil_coefficient": NEUROPIL_COEF,
                "baseline_smoothing_sigma_samples": BASELINE_SMOOTH_SIGMA,
                "maximin_window_samples": BASELINE_WINDOW_SAMPLES,
                "dff_smoothing_sigma_samples": DFF_SMOOTH_SIGMA,
                "oasis_tau_seconds": OASIS_TAU_SECONDS,
                "nonfinite_events_replaced_with_zero": True,
            },
            "curation": {
                "manual_suite2p_iscell": True,
                "putative_interneuron_dff_speed_r_threshold": INTERNEURON_SPEED_R,
                "bad_lick_trial_rule": ">30% of frames have cumulative lick count >2",
                "place_cell_filter": False,
            },
            "source_limitation": (
                "m17/m18 segmentation contains two planes, but NWB RoiResponseSeries data are "
                "available only for plane0; unavailable plane1 cells are not included."
            ),
            "session_info": [x["info"] for x in converted],
        },
    }
    validate_converted(data)
    elapsed = time.perf_counter() - all_started
    total_trials = sum(len(x) for x in data["neural"])
    total_neurons = sum(len(x) for x in data["brain_region_idx"])
    neural_bytes = sum(a.nbytes for session in data["neural"] for a in session)
    print(
        f"Converted {len(files)} sessions, {total_trials} trials, "
        f"{total_neurons} session-neurons in {elapsed:.2f} s; "
        f"neural arrays {neural_bytes / 2**30:.3f} GiB",
        flush=True,
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", help="Output pickle path")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full", action="store_true", help="Process all sessions (default)")
    modes.add_argument("--sample", action="store_true", help="Process only two sessions")
    parser.add_argument(
        "--show-processing", action="store_true",
        help="Save processing_<session_id>.png for up to two sessions",
    )
    args = parser.parse_args()

    files = discover_files(sample=args.sample)
    mode = "sample" if args.sample else "full"
    print(f"Mode: {mode}; found {len(files)} NWB files", flush=True)
    data = build_dataset(files, args.show_processing)

    output = Path(args.outpicklefile)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_started = time.perf_counter()
    with output.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Wrote {output} ({output.stat().st_size / 2**30:.3f} GiB) "
        f"in {time.perf_counter() - dump_started:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
