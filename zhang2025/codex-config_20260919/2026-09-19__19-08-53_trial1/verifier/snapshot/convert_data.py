#!/usr/bin/env python3
"""Convert the frozen IBL brain-wide-map release to decoder format.

Usage
-----
python -u /app/convert_data.py OUTPUT.pkl [--full | --sample] [--show-processing]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from scipy.interpolate import interp1d

# Use the implementation bundled with the reference code, not a reimplementation.
import sys
sys.path.insert(0, "/app/code/ibllib")
from brainbox.behavior.wheel import interpolate_position, velocity_filtered


APP = Path("/app")
DATA_ROOT = APP / "data" / "one_cache"
RELEASE_CSV = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
# The reference behavior code labels a count bin by its right edge.
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)

EXPECTED_RELEASE = {
    "subjects": 139,
    "sessions": 459,
    "probes": 699,
    "clusters": 621_733,
    "good_clusters": 75_708,
}


@dataclass(frozen=True)
class ProbeFiles:
    name: str
    spikes_times: Path
    spikes_clusters: Path
    clusters_channels: Path
    channel_ids: Path


@dataclass(frozen=True)
class SessionSpec:
    eid: str
    subject: str
    lab: str
    date: str
    number: int
    session_dir: Path
    trial_table: Path
    wheel_times: Path
    wheel_position: Path
    probes: tuple[ProbeFiles, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all valid frozen-release sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process the first two valid sessions.")
    parser.add_argument("--show-processing", action="store_true", help="Save processing plots for up to two sessions.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Concurrent session spike-binning workers (default: min(8, CPUs)).")
    return parser.parse_args()


def _find_file(root: Path, name: str, preferred_revision: str | None = None) -> Path:
    candidates = sorted(p for p in root.glob(f"**/{name}") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"Missing {name} beneath {root}")
    if preferred_revision:
        preferred = [p for p in candidates if preferred_revision in p.parts]
        if preferred:
            return preferred[-1]
    # Most attributes have one physical version. For multiples, a revision directory
    # sorts deterministically; explicit preferences above cover trials and spikes.
    return candidates[-1]


def _optional_file(root: Path, name: str, preferred_revision: str | None = None) -> Path | None:
    try:
        return _find_file(root, name, preferred_revision)
    except FileNotFoundError:
        return None


def build_release_specs() -> tuple[list[SessionSpec], dict[str, int]]:
    release = pd.read_csv(RELEASE_CSV, dtype={"date": str, "subject": str, "lab": str})
    assert release["pid"].nunique() == EXPECTED_RELEASE["probes"]
    assert release["eid"].nunique() == EXPECTED_RELEASE["sessions"]
    assert release["subject"].nunique() == EXPECTED_RELEASE["subjects"]

    specs: list[SessionSpec] = []
    cluster_inventory = 0
    # groupby(sort=False) preserves the release order.
    for eid, rows in release.groupby("eid", sort=False):
        first = rows.iloc[0]
        number = int(first["session_number"])
        session_dir = (DATA_ROOT / first["lab"] / "Subjects" / first["subject"] /
                       first["date"] / f"{number:03d}")
        alf = session_dir / "alf"
        probes: list[ProbeFiles] = []
        for row in rows.itertuples(index=False):
            probe_root = alf / row.probe_name / "pykilosort"
            pf = ProbeFiles(
                name=row.probe_name,
                spikes_times=_find_file(probe_root, "spikes.times.npy", "#2024-05-06#"),
                spikes_clusters=_find_file(probe_root, "spikes.clusters.npy", "#2024-05-06#"),
                clusters_channels=_find_file(probe_root, "clusters.channels.npy", "#2024-05-06#"),
                channel_ids=_find_file(probe_root, "channels.brainLocationIds_ccf_2017.npy", "#2024-05-06#"),
            )
            cluster_inventory += int(np.load(pf.clusters_channels, mmap_mode="r").shape[0])
            probes.append(pf)
        specs.append(SessionSpec(
            eid=str(eid), subject=str(first["subject"]), lab=str(first["lab"]),
            date=str(first["date"]), number=number, session_dir=session_dir,
            trial_table=_find_file(alf, "_ibl_trials.table.pqt", "#2025-03-03#"),
            wheel_times=_find_file(alf, "_ibl_wheel.timestamps.npy"),
            wheel_position=_find_file(alf, "_ibl_wheel.position.npy"),
            probes=tuple(probes),
        ))
    assert cluster_inventory == EXPECTED_RELEASE["clusters"], (
        f"Frozen cluster inventory mismatch: {cluster_inventory} != {EXPECTED_RELEASE['clusters']}")
    return specs, {**EXPECTED_RELEASE, "clusters_from_headers": cluster_inventory}


def reference_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    mask = trials[required].notna().all(axis=1).to_numpy().copy()
    rt = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    duration = (trials["feedback_times"] - trials["goCue_times"]).to_numpy()
    mask &= (rt >= 0.08) & (rt <= 2.0)
    mask &= ~(duration > 10.0)  # matches the reference query, including NaN semantics
    mask &= trials["choice"].to_numpy() != 0
    return mask


def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    """Zero-based block position on the unfiltered source sequence."""
    n = probability_left.size
    starts = np.r_[0, np.flatnonzero(probability_left[1:] != probability_left[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    out = np.empty(n, dtype=np.int32)
    for start, end in zip(starts, ends):
        out[start:end] = np.arange(end - start, dtype=np.int32)
    return out


def choose_motion_stream(alf: Path) -> tuple[str, Path, Path] | None:
    for view, revision in (("left", "#2025-05-29#"), ("right", "#2025-05-31#")):
        times = _optional_file(alf, f"_ibl_{view}Camera.times.npy")
        values = _optional_file(alf, f"{view}Camera.ROIMotionEnergy.npy", revision)
        if times is not None and values is not None:
            return view, times, values
    return None


def coverage_mask(times: np.ndarray, begins: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implement get_behavior_per_interval boundary validity checks."""
    ib = np.searchsorted(times, begins, side="right")
    ie = np.searchsorted(times, ends, side="left")
    good = (ie - ib >= 2) & (ib < len(times)) & (ie > 1)
    idx = np.flatnonzero(good)
    good[idx] &= np.abs(begins[idx] - times[ib[idx]]) <= BIN_SIZE
    good[idx] &= np.abs(ends[idx] - times[ie[idx] - 1]) <= BIN_SIZE
    return good, ib, ie


def interval_interpolate(times: np.ndarray, values: np.ndarray, targets: np.ndarray,
                         interval_ends: np.ndarray, ie: np.ndarray) -> np.ndarray:
    """Reference-equivalent linear interpolation, including last-bin extrapolation."""
    result = np.interp(targets.ravel(), times, values).reshape(targets.shape)
    # get_behavior_per_interval excludes samples at/after interval end and uses
    # fill_value='extrapolate'. Reproduce that subtle behavior for the final bin.
    i1 = ie - 1
    i0 = ie - 2
    x0, x1 = times[i0], times[i1]
    y0, y1 = values[i0], values[i1]
    result[:, -1] = y1 + (interval_ends - x1) * (y1 - y0) / (x1 - x0)
    return result


def prepare_behavior(spec: SessionSpec, capture_plot: bool = False) -> tuple[dict[str, Any] | None, str | None]:
    t0 = time.perf_counter()
    trials = pd.read_parquet(spec.trial_table)
    raw_n = len(trials)
    ref_mask = reference_trial_mask(trials)
    raw_idx = np.flatnonzero(ref_mask)
    if raw_idx.size < 2:
        return None, "fewer than two reference-valid trials"

    motion = choose_motion_stream(spec.session_dir / "alf")
    if motion is None:
        return None, "missing paired left and right whisker-motion streams"
    motion_view, motion_times_path, motion_values_path = motion
    motion_times = np.asarray(np.load(motion_times_path, mmap_mode="r"), dtype=np.float64)
    motion_values = np.asarray(np.load(motion_values_path, mmap_mode="r"), dtype=np.float64)
    if motion_times.shape != motion_values.shape or motion_times.size < 2:
        return None, "invalid whisker-motion stream shape"
    if not (np.all(np.diff(motion_times) > 0) and np.all(np.isfinite(motion_values))):
        return None, "nonmonotonic/nonfinite whisker-motion stream"

    wheel_raw_times = np.asarray(np.load(spec.wheel_times, mmap_mode="r"), dtype=np.float64)
    wheel_raw_position = np.asarray(np.load(spec.wheel_position, mmap_mode="r"), dtype=np.float64)
    if wheel_raw_times.shape != wheel_raw_position.shape or wheel_raw_times.size < 2:
        return None, "invalid wheel stream shape"
    wheel_position, wheel_times = interpolate_position(wheel_raw_times, wheel_raw_position, freq=1000)
    wheel_velocity, _ = velocity_filtered(wheel_position, fs=1000, corner_frequency=20, order=8)
    wheel_speed = np.abs(wheel_velocity)

    stim = trials["stimOn_times"].to_numpy(dtype=np.float64)[raw_idx]
    begins, ends = stim + OFF_START, stim + OFF_END
    wheel_good, _, wheel_ie = coverage_mask(wheel_times, begins, ends)
    motion_good, _, motion_ie = coverage_mask(motion_times, begins, ends)
    stream_good = wheel_good & motion_good
    raw_idx = raw_idx[stream_good]
    if raw_idx.size < 2:
        return None, "fewer than two trials with complete wheel/whisker coverage"

    stim = stim[stream_good]
    begins, ends = begins[stream_good], ends[stream_good]
    wheel_ie = wheel_ie[stream_good]
    motion_ie = motion_ie[stream_good]
    targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
    wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
    whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
    if not (np.all(np.isfinite(wheel)) and np.all(np.isfinite(whisker))):
        return None, "nonfinite aligned behavior"

    probs_all = trials["probabilityLeft"].to_numpy(dtype=np.float64)
    block_no_all = trial_number_in_block(probs_all)
    choices = trials["choice"].to_numpy(dtype=np.float64)[raw_idx]
    priors = probs_all[raw_idx]
    if not np.all(np.isin(choices, (-1.0, 1.0))):
        raise ValueError(f"{spec.eid}: unexpected retained choice values")
    if not np.all(np.isin(priors, (0.2, 0.5, 0.8))):
        raise ValueError(f"{spec.eid}: unexpected retained prior values")

    result: dict[str, Any] = {
        "spec": spec,
        "raw_trial_count": raw_n,
        "reference_valid_count": int(ref_mask.sum()),
        "source_trial_idx": raw_idx.astype(np.int32),
        "stim_times": trials["stimOn_times"].to_numpy(dtype=np.float64)[raw_idx],
        "choice": (choices == 1).astype(np.int8),  # -1 left -> 0; +1 right -> 1
        "prior": np.searchsorted(np.array([0.2, 0.5, 0.8]), priors).astype(np.int8),
        "block_trial": block_no_all[raw_idx].astype(np.float32),
        "wheel": wheel.astype(np.float32),
        "whisker": whisker.astype(np.float32),
        "motion_view": motion_view,
        "behavior_seconds": time.perf_counter() - t0,
    }
    if capture_plot:
        j = 0
        lo, hi = begins[j] - 0.1, ends[j] + 0.1
        wi = (wheel_times >= lo) & (wheel_times <= hi)
        mi = (motion_times >= lo) & (motion_times <= hi)
        result["plot_capture"] = {
            "wheel_times": wheel_times[wi] - stim[j],
            "wheel_position": wheel_position[wi],
            "wheel_speed": wheel_speed[wi],
            "motion_times": motion_times[mi] - stim[j],
            "motion_values": motion_values[mi],
        }
    return result, None


def bin_spikes_for_session(behavior: dict[str, Any]) -> dict[str, Any]:
    spec: SessionSpec = behavior["spec"]
    t0 = time.perf_counter()
    stim = behavior["stim_times"]
    begins, ends = stim + OFF_START, stim + OFF_END
    n_trials = len(stim)
    n_per_probe = [int(np.load(p.clusters_channels, mmap_mode="r").shape[0]) for p in spec.probes]
    n_neurons = int(sum(n_per_probe))
    neural = [np.zeros((n_neurons, N_BINS), dtype=np.float32) for _ in range(n_trials)]
    region_names: list[str] = []
    br = BrainRegions()
    offset = 0

    for probe, n_clusters in zip(spec.probes, n_per_probe):
        spike_times = np.load(probe.spikes_times, mmap_mode="r")
        spike_clusters = np.load(probe.spikes_clusters, mmap_mode="r")
        if spike_times.shape != spike_clusters.shape or np.any(np.diff(spike_times[: min(len(spike_times), 1_000_000)]) < 0):
            raise ValueError(f"{spec.eid}/{probe.name}: invalid spike arrays")
        cluster_channels = np.load(probe.clusters_channels, mmap_mode="r").astype(np.int64, copy=False)
        channel_ids = np.load(probe.channel_ids, mmap_mode="r")
        if cluster_channels.min() < 0 or cluster_channels.max() >= len(channel_ids):
            raise ValueError(f"{spec.eid}/{probe.name}: invalid cluster channel index")
        native = br.id2acronym(channel_ids[cluster_channels])
        region_names.extend(br.acronym2acronym(native, mapping="Beryl").tolist())

        left = np.searchsorted(spike_times, begins, side="left")
        right = np.searchsorted(spike_times, ends, side="left")
        for j, (i0, i1, begin) in enumerate(zip(left, right, begins)):
            if i1 <= i0:
                continue
            clusters = np.asarray(spike_clusters[i0:i1], dtype=np.int64)
            if clusters.min() < 0 or clusters.max() >= n_clusters:
                raise ValueError(f"{spec.eid}/{probe.name}: spike cluster outside cluster table")
            bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
            # Floating-point arithmetic at exact edges may yield N_BINS; the
            # half-open interval guarantees such points belong outside.
            keep = (bins >= 0) & (bins < N_BINS)
            flat = clusters[keep] * N_BINS + bins[keep]
            counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
            neural[j][offset:offset + n_clusters] = counts
        offset += n_clusters

    assert offset == n_neurons == len(region_names)
    # Direct independent spot check against original data for one count.
    p0 = spec.probes[0]
    st = np.load(p0.spikes_times, mmap_mode="r")
    sc = np.load(p0.spikes_clusters, mmap_mode="r")
    direct = np.sum((st >= begins[0]) & (st < begins[0] + BIN_SIZE) & (sc == 0))
    if not np.allclose(neural[0][0, 0], direct):
        raise AssertionError(f"{spec.eid}: raw neural spot check failed")

    behavior["neural"] = neural
    behavior["region_names"] = np.asarray(region_names, dtype=object)
    behavior["n_neurons"] = n_neurons
    behavior["spike_seconds"] = time.perf_counter() - t0
    print(f"Binned {spec.eid}: {n_trials} trials, {n_neurons} neurons, "
          f"behavior {behavior['behavior_seconds']:.2f}s, spikes {behavior['spike_seconds']:.2f}s",
          flush=True)
    return behavior


def make_processing_plot(session: dict[str, Any], thresholds: dict[str, list[float]]) -> None:
    spec: SessionSpec = session["spec"]
    cap = session.get("plot_capture", {})
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
    ax = axes.ravel()
    ax[0].bar(["raw", "reference valid", "stream valid"],
              [session["raw_trial_count"], session["reference_valid_count"], len(session["neural"])])
    ax[0].set_title("Trial filtering")

    if cap:
        ax[1].plot(cap["wheel_times"], cap["wheel_position"], lw=0.8)
        ax[1].axvline(0, color="k", ls="--"); ax[1].set_title("Wheel position (stimulus at 0)")
        ax[2].plot(cap["wheel_times"], cap["wheel_speed"], alpha=0.6, lw=0.8)
        ax[2].plot(RELATIVE_BIN_ENDS, session["wheel"][0], "o-", ms=2)
        ax[2].set_title("Filtered wheel speed and 20 ms samples")
        ax[3].plot(cap["motion_times"], cap["motion_values"], alpha=0.65, lw=0.8)
        ax[3].plot(RELATIVE_BIN_ENDS, session["whisker"][0], "o-", ms=2)
        ax[3].set_title(f"{session['motion_view']} whisker motion energy and samples")

    wheel_cls = np.searchsorted(np.asarray(thresholds["wheel_speed"]), session["wheel"][0], side="right")
    whisk_cls = np.searchsorted(np.asarray(thresholds["whisker_motion_energy"]), session["whisker"][0], side="right")
    ax[4].plot(RELATIVE_BIN_ENDS, session["wheel"][0], label="continuous")
    ax4 = ax[4].twinx(); ax4.step(RELATIVE_BIN_ENDS, wheel_cls, where="mid", color="C1", label="class")
    ax[4].set_title("Wheel discretization (global tertiles)")
    ax[5].plot(RELATIVE_BIN_ENDS, session["whisker"][0], label="continuous")
    ax5 = ax[5].twinx(); ax5.step(RELATIVE_BIN_ENDS, whisk_cls, where="mid", color="C1", label="class")
    ax[5].set_title("Whisker discretization (global tertiles)")

    raster = session["neural"][0][: min(50, session["n_neurons"])]
    ax[6].imshow(raster, aspect="auto", interpolation="nearest", extent=[OFF_START, OFF_END, raster.shape[0], 0])
    ax[6].axvline(0, color="w", ls="--"); ax[6].set_title("Aligned 20 ms spike counts (first 50 units)")
    ax[6].set_xlabel("s from stimulus onset")
    ax[7].plot(RELATIVE_BIN_ENDS, RELATIVE_BIN_ENDS, label="time input")
    ax[7].plot(RELATIVE_BIN_ENDS, np.full(N_BINS, session["block_trial"][0]), label="block trial input")
    ax[7].step(RELATIVE_BIN_ENDS, wheel_cls, label="wheel class", alpha=0.8)
    ax[7].step(RELATIVE_BIN_ENDS, whisk_cls, label="whisker class", alpha=0.8)
    ax[7].legend(fontsize=8); ax[7].set_title("Final aligned inputs/outputs")
    fig.suptitle(f"Processing audit: {spec.eid} ({spec.subject})")
    out = APP / f"processing_{spec.eid}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def assemble_data(sessions: list[dict[str, Any]], inventory: dict[str, int],
                  excluded: list[dict[str, str]], thresholds: dict[str, list[float]]) -> dict[str, Any]:
    brain_regions = sorted({str(r) for s in sessions for r in s["region_names"]})
    region_lookup = {r: i for i, r in enumerate(brain_regions)}
    subjects = sorted({s["spec"].subject for s in sessions})
    subject_lookup = {s: i for i, s in enumerate(subjects)}

    neural_all: list[list[np.ndarray]] = []
    input_all: list[list[np.ndarray]] = []
    output_all: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict[str, Any]] = []

    qwheel = np.asarray(thresholds["wheel_speed"])
    qwhisker = np.asarray(thresholds["whisker_motion_energy"])
    for s in sessions:
        n_trials = len(s["neural"])
        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []
        for j in range(n_trials):
            inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                                     np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
            outputs.append(np.vstack((
                np.full(N_BINS, s["choice"][j], dtype=np.int8),
                np.full(N_BINS, s["prior"][j], dtype=np.int8),
                np.searchsorted(qwheel, s["wheel"][j], side="right").astype(np.int8),
                np.searchsorted(qwhisker, s["whisker"][j], side="right").astype(np.int8),
            )))
        neural_all.append(s["neural"])
        input_all.append(inputs)
        output_all.append(outputs)
        brain_region_idx.append(np.fromiter((region_lookup[str(x)] for x in s["region_names"]),
                                            dtype=np.int32, count=len(s["region_names"])))
        spec: SessionSpec = s["spec"]
        session_info.append({
            "eid": spec.eid, "subject": spec.subject, "lab": spec.lab,
            "date": spec.date, "session_number": spec.number,
            "probe_names": [p.name for p in spec.probes],
            "motion_energy_view": s["motion_view"],
            "raw_trials": s["raw_trial_count"],
            "reference_valid_trials": s["reference_valid_count"],
            "converted_trials": n_trials,
            "source_trial_indices": s["source_trial_idx"].astype(int).tolist(),
            "n_neurons": s["n_neurons"],
        })

    data = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": np.asarray([subject_lookup[s["spec"].subject] for s in sessions], dtype=np.int32),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_since_stimulus_onset", "trial_number_in_block"],
        "output_names": ["choice", "prior_probability_left", "wheel_speed", "whisker_motion_energy"],
        "output_values": [
            ["left", "right"],
            ["0.2", "0.5", "0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": ("IBL visual decision task; decode left/right choice, block prior, "
                                 "wheel-speed tertile, and whisker-motion-energy tertile from "
                                 "stimulus-aligned brain-wide Neuropixels spike counts."),
            "time_bin_size": 20.0,
            "time_bin_size_units": "ms",
            "temporal_alignment_event": "visual stimulus onset (trials.stimOn_times)",
            "off_start": OFF_START,
            "off_end": OFF_END,
            "n_time_bins": N_BINS,
            "time_coordinate": "right edge of each half-open spike-count bin, seconds from stimulus onset",
            "neural_representation": "all-unit Kilosort 2.5 spike counts in half-open 20 ms bins, float32",
            "source_release": "code/code_zhang2025/data/bwm_release.csv (data-paper freeze)",
            "source_release_inventory": inventory,
            "trial_filter": ("nonmissing stimOn/choice/feedback/probabilityLeft/firstMovement/feedbackType; "
                             "0.08 <= firstMovement-stimOn <= 2 s; choice != 0; feedback-goCue <= 10 s; "
                             "complete wheel and whisker coverage"),
            "choice_mapping": "source -1 (left) -> 0; source +1 (right) -> 1",
            "prior_mapping": {"0.2": 0, "0.5": 1, "0.8": 2},
            "trial_number_in_block": "zero-based position in the original unfiltered probabilityLeft run",
            "continuous_output_discretization": "global tertiles over all retained aligned time samples",
            "discretization_thresholds": thresholds,
            "motion_energy_view_policy": "left preferred; right fallback",
            "excluded_sessions": excluded,
            "session_info": session_info,
        },
    }
    return data


def internal_validate(data: dict[str, Any]) -> dict[str, Any]:
    assert len(data["neural"]) == len(data["input"]) == len(data["output"]) >= 1
    class_counts = [np.zeros(n, dtype=np.int64) for n in (2, 3, 3, 3)]
    total_trials = total_neurons = 0
    for si, (neural, inputs, outputs) in enumerate(zip(data["neural"], data["input"], data["output"])):
        assert len(neural) == len(inputs) == len(outputs) >= 2
        assert len(data["brain_region_idx"][si]) == neural[0].shape[0]
        total_trials += len(neural)
        total_neurons += neural[0].shape[0]
        for x, u, y in zip(neural, inputs, outputs):
            assert x.shape[1] == u.shape[1] == y.shape[1] == N_BINS
            assert u.shape == (2, N_BINS) and y.shape == (4, N_BINS)
            assert x.dtype == np.float32 and u.dtype == np.float32 and y.dtype == np.int8
            assert np.all(np.isfinite(x)) and np.all(np.isfinite(u)) and np.all(np.isfinite(y))
            assert np.all(x >= 0) and np.allclose(x, np.round(x))
            for k, counts in enumerate(class_counts):
                counts += np.bincount(y[k], minlength=len(counts))
    for counts in class_counts:
        assert np.all(counts > 0), f"Empty categorical class: {counts}"
    return {
        "sessions": len(data["neural"]), "trials": total_trials,
        "session_neuron_sum": total_neurons,
        "class_counts": [x.tolist() for x in class_counts],
        "class_fractions": [(x / x.sum()).tolist() for x in class_counts],
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    print("Loading frozen release inventory...", flush=True)
    specs, inventory = build_release_specs()
    print(json.dumps(inventory, indent=2), flush=True)

    behavior_sessions: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    wanted = 2 if args.sample else None
    for spec in specs:
        capture = args.show_processing and len(behavior_sessions) < 2
        try:
            behavior, reason = prepare_behavior(spec, capture_plot=capture)
        except Exception as exc:
            behavior, reason = None, f"behavior processing error: {type(exc).__name__}: {exc}"
        if behavior is None:
            excluded.append({"eid": spec.eid, "reason": str(reason)})
            print(f"Excluded {spec.eid}: {reason}", flush=True)
            continue
        behavior_sessions.append(behavior)
        print(f"Prepared behavior {spec.eid}: {len(behavior['source_trial_idx'])}/"
              f"{behavior['raw_trial_count']} trials ({behavior['behavior_seconds']:.2f}s)", flush=True)
        if wanted is not None and len(behavior_sessions) >= wanted:
            break
    if not behavior_sessions:
        raise RuntimeError("No valid sessions")

    wheel_values = np.concatenate([s["wheel"].ravel() for s in behavior_sessions])
    whisker_values = np.concatenate([s["whisker"].ravel() for s in behavior_sessions])
    wheel_q = np.quantile(wheel_values, [1 / 3, 2 / 3])
    whisker_q = np.quantile(whisker_values, [1 / 3, 2 / 3])
    del wheel_values, whisker_values
    if not (wheel_q[0] < wheel_q[1] and whisker_q[0] < whisker_q[1]):
        raise ValueError(f"Non-distinct discretization thresholds: wheel={wheel_q}, whisker={whisker_q}")
    thresholds = {
        "wheel_speed": [float(x) for x in wheel_q],
        "whisker_motion_energy": [float(x) for x in whisker_q],
    }
    print(f"Global tertile thresholds: {thresholds}", flush=True)

    workers = max(1, min(args.workers, len(behavior_sessions)))
    print(f"Binning spikes with {workers} session worker(s)...", flush=True)
    if workers == 1:
        sessions = [bin_spikes_for_session(s) for s in behavior_sessions]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sessions = list(pool.map(bin_spikes_for_session, behavior_sessions))

    if args.show_processing:
        for session in sessions[:2]:
            make_processing_plot(session, thresholds)

    data = assemble_data(sessions, inventory, excluded, thresholds)
    stats = internal_validate(data)
    print("Internal validation:", json.dumps(stats, indent=2), flush=True)
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    write_start = time.perf_counter()
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - started
    gib = args.outpicklefile.stat().st_size / 1024 ** 3
    print(f"Wrote {args.outpicklefile} ({gib:.3f} GiB) in {write_seconds:.2f}s", flush=True)
    print(f"Total conversion time: {total_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
