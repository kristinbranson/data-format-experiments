#!/usr/bin/env python3
"""Convert IBL brain-wide map data to the decoder pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from iblutil.numerical import bincount2D
from scipy.interpolate import interp1d

# Reuse lightweight reference utilities from the bundled ibllib source tree.
sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "ibllib"))
from brainbox.behavior.wheel import interpolate_position, velocity_filtered  # noqa: E402


ROOT = Path("/app")
DATA_ROOT = ROOT / "data" / "one_cache"
MANIFEST_FILES = [
    DATA_ROOT / "2025_Q3_IBL_et_al_BWM" / "sessions.pqt",
    DATA_ROOT / "Brainwidemap" / "sessions.pqt",
    DATA_ROOT / "2022_Q4_IBL_et_al_BWM" / "sessions.pqt",
]

TARGET_DATASET_NAME = "IBL brain-wide map release from local cache"
TIME_WINDOW = (-0.5, 1.5)
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))
TRIAL_MASK_RT = (0.08, 2.0)
MAX_TRIAL_LEN = 10.0
GOOD_CLUSTER_LABEL = 1
MAX_PROCESSING_PLOTS = 2


@dataclass
class SessionSpec:
    eid: str
    lab: str
    subject: str
    date: str
    session_number: int
    session_path: Path

    @property
    def session_number_str(self) -> str:
        return f"{self.session_number:03d}"


@dataclass
class ProcessedSession:
    eid: str
    subject: str
    neural: list[np.ndarray]
    inputs: list[np.ndarray]
    choice: list[np.ndarray]
    prior: list[np.ndarray]
    wheel_cont: list[np.ndarray]
    whisker_cont: list[np.ndarray]
    cluster_regions: np.ndarray
    kept_trial_indices: np.ndarray
    whisker_source: str
    n_clusters_total: int
    n_clusters_good: int
    raw_n_trials: int
    kept_n_trials: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IBL data to decoder format.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process the full dataset (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions.",
    )
    return parser.parse_args()


def load_session_manifest() -> pd.DataFrame:
    for manifest in MANIFEST_FILES:
        if manifest.exists():
            return pd.read_parquet(manifest)
    raise FileNotFoundError("No session manifests found in data cache")


def resolve_session_specs() -> tuple[list[SessionSpec], list[str]]:
    manifest = load_session_manifest()
    specs: list[SessionSpec] = []
    missing: list[str] = []
    for eid, row in manifest.iterrows():
        session_path = (
            DATA_ROOT
            / row["lab"]
            / "Subjects"
            / row["subject"]
            / str(row["date"])
            / f"{int(row['number']):03d}"
        )
        if not session_path.exists():
            missing.append(eid)
            continue
        specs.append(
            SessionSpec(
                eid=eid,
                lab=str(row["lab"]),
                subject=str(row["subject"]),
                date=str(row["date"]),
                session_number=int(row["number"]),
                session_path=session_path,
            )
        )
    return specs, missing


def version_key(path: Path) -> tuple[int, str]:
    revision = ""
    for part in path.parts:
        if part.startswith("#") and part.endswith("#"):
            revision = part.strip("#")
    is_versioned = 1 if revision else 0
    return (is_versioned, revision)


def pick_one_file(base: Path, pattern: str) -> Path | None:
    matches = list(base.rglob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: (version_key(p), str(p)))
    return matches[-1]


def load_trials_table(session_path: Path) -> pd.DataFrame:
    trial_file = pick_one_file(session_path / "alf", "_ibl_trials.table.pqt")
    if trial_file is None:
        raise FileNotFoundError(f"Missing trial table for {session_path}")
    return pd.read_parquet(trial_file)


def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
    ]
    mask = np.ones(len(trials), dtype=bool)
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask &= rt >= TRIAL_MASK_RT[0]
    mask &= rt <= TRIAL_MASK_RT[1]
    mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN
    mask &= trials["choice"] != 0
    for col in required:
        mask &= trials[col].notna().to_numpy()
    return mask


def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    counters = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return counters
    count = 1
    counters[0] = count
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            count += 1
        else:
            count = 1
        counters[i] = count
    return counters


def load_probe_spikes_and_regions(
    probe_path: Path,
    br: BrainRegions,
    label_threshold: int = GOOD_CLUSTER_LABEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    pykilo_path = probe_path / "pykilosort"
    spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
    spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
    metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
    clusters_channels_file = pick_one_file(pykilo_path, "clusters.channels.npy")
    channels_ids_file = pick_one_file(pykilo_path, "channels.brainLocationIds_ccf_2017.npy")

    required = [
        spikes_times_file,
        spikes_clusters_file,
        metrics_file,
        clusters_channels_file,
        channels_ids_file,
    ]
    if any(x is None for x in required):
        raise FileNotFoundError(f"Missing spike sorting assets under {probe_path}")

    metrics = pd.read_parquet(metrics_file, columns=["label"])
    cluster_labels = metrics["label"].to_numpy()
    good_mask = cluster_labels >= label_threshold
    selected_cluster_ids = np.flatnonzero(good_mask)
    if selected_cluster_ids.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=object),
            int(len(cluster_labels)),
            0,
        )

    spikes_times = np.load(spikes_times_file, mmap_mode="r")
    spikes_clusters = np.load(spikes_clusters_file, mmap_mode="r")
    spike_keep = good_mask[spikes_clusters]

    remap = np.full(cluster_labels.shape[0], -1, dtype=np.int32)
    remap[selected_cluster_ids] = np.arange(selected_cluster_ids.size, dtype=np.int32)
    selected_times = np.asarray(spikes_times[spike_keep], dtype=np.float64)
    selected_clusters = remap[np.asarray(spikes_clusters[spike_keep], dtype=np.int64)]

    cluster_channels = np.asarray(np.load(clusters_channels_file), dtype=np.int64)
    channel_ids = np.asarray(np.load(channels_ids_file), dtype=np.int64)
    cluster_channels = cluster_channels[selected_cluster_ids]
    valid_channel = (cluster_channels >= 0) & (cluster_channels < len(channel_ids))
    cluster_region_ids = np.zeros(selected_cluster_ids.size, dtype=np.int64)
    cluster_region_ids[valid_channel] = channel_ids[cluster_channels[valid_channel]]
    cluster_regions = br.id2acronym(cluster_region_ids)

    return (
        selected_times,
        selected_clusters,
        np.asarray(cluster_regions, dtype=object),
        int(len(cluster_labels)),
        int(selected_cluster_ids.size),
    )


def load_session_spikes(
    session_path: Path,
    br: BrainRegions,
    label_threshold: int = GOOD_CLUSTER_LABEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    probe_paths = sorted(
        [
            p
            for p in (session_path / "alf").glob("probe*")
            if p.is_dir() and p.name.startswith("probe")
        ]
    )
    merged_times = []
    merged_clusters = []
    merged_regions = []
    cluster_offset = 0
    total_clusters = 0
    good_clusters = 0
    for probe_path in probe_paths:
        times, clusters, regions, total_here, good_here = load_probe_spikes_and_regions(
            probe_path, br, label_threshold=label_threshold
        )
        total_clusters += total_here
        good_clusters += good_here
        if good_here == 0:
            continue
        merged_times.append(times)
        merged_clusters.append(clusters + cluster_offset)
        merged_regions.append(regions)
        cluster_offset += good_here

    if cluster_offset == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=object),
            total_clusters,
            0,
        )

    spike_times = np.concatenate(merged_times)
    spike_clusters = np.concatenate(merged_clusters)
    cluster_regions = np.concatenate(merged_regions)
    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]
    return spike_times, spike_clusters, cluster_regions, total_clusters, good_clusters


def bin_spikes_for_trials(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    n_clusters: int,
    align_times: np.ndarray,
    window: tuple[float, float] = TIME_WINDOW,
    binsize: float = BINSIZE_S,
) -> list[np.ndarray]:
    interval_len = window[1] - window[0]
    n_bins = int(np.ceil(interval_len / binsize))
    cluster_ids = np.arange(n_clusters, dtype=np.int32)
    intervals = np.c_[align_times + window[0], align_times + window[1]]
    results: list[np.ndarray] = []
    idx_starts = np.searchsorted(spike_times, intervals[:, 0], side="left")
    idx_ends = np.searchsorted(spike_times, intervals[:, 1], side="left")
    for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
        trial_counts = np.zeros((n_clusters, n_bins), dtype=np.float16)
        if idx1 > idx0:
            counts, _, cluster_idx = bincount2D(
                spike_times[idx0:idx1],
                spike_clusters[idx0:idx1],
                xbin=binsize,
                xlim=[start, end],
            )
            if counts.size:
                counts = counts[:, :n_bins]
                trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
        results.append(trial_counts)
    return results


def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wheel_pos_file = pick_one_file(session_path / "alf", "_ibl_wheel.position.npy")
    wheel_ts_file = pick_one_file(session_path / "alf", "_ibl_wheel.timestamps.npy")
    if wheel_pos_file is None or wheel_ts_file is None:
        raise FileNotFoundError(f"Missing wheel files for {session_path}")
    pos = np.asarray(np.load(wheel_pos_file), dtype=np.float64)
    ts = np.asarray(np.load(wheel_ts_file), dtype=np.float64)
    if ts.ndim == 2 and ts.shape[1] == 2:
        ts = ts.mean(axis=1)
    pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
    vel, _ = velocity_filtered(pos_interp, 1000)
    return ts_interp, np.abs(vel)


def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    for camera in ("left", "right"):
        me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
        times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
        if me_file is None or times_file is None:
            continue
        values = np.asarray(np.load(me_file), dtype=np.float64)
        times = np.asarray(np.load(times_file), dtype=np.float64)
        if len(values) != len(times):
            n = min(len(values), len(times))
            values = values[:n]
            times = times[:n]
        return times, values, camera
    raise FileNotFoundError(f"Missing whisker motion energy files for {session_path}")


def interpolate_behavior_trials(
    target_times: np.ndarray,
    target_values: np.ndarray,
    align_times: np.ndarray,
    window: tuple[float, float] = TIME_WINDOW,
    binsize: float = BINSIZE_S,
) -> tuple[list[np.ndarray | None], np.ndarray]:
    valid_source = np.isfinite(target_times) & np.isfinite(target_values)
    target_times = target_times[valid_source]
    target_values = target_values[valid_source]
    interval_len = window[1] - window[0]
    n_bins = int(np.ceil(interval_len / binsize))
    x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
    outputs: list[np.ndarray | None] = []
    mask = np.zeros(len(align_times), dtype=bool)

    idx_beg = np.searchsorted(target_times, align_times + window[0], side="right")
    idx_end = np.searchsorted(target_times, align_times + window[1], side="left")
    for i, align_time in enumerate(align_times):
        start = align_time + window[0]
        end = align_time + window[1]
        vals = target_values[idx_beg[i] : idx_end[i]]
        ts = target_times[idx_beg[i] : idx_end[i]]
        if len(vals) == 0:
            outputs.append(None)
            continue
        if np.abs(start - ts[0]) > binsize or np.abs(end - ts[-1]) > binsize:
            outputs.append(None)
            continue
        interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
        outputs.append(interp(align_time + x_rel).astype(np.float32))
        mask[i] = True
    return outputs, mask


def map_choice_to_binary(choice_values: np.ndarray) -> np.ndarray:
    mapped = np.full(choice_values.shape, -1, dtype=np.int16)
    mapped[choice_values == 1] = 0   # left
    mapped[choice_values == -1] = 1  # right
    if np.any(mapped < 0):
        raise ValueError("Unexpected choice values after masking")
    return mapped


def map_prior_to_categorical(prob_left: np.ndarray) -> np.ndarray:
    mapped = np.full(prob_left.shape, -1, dtype=np.int16)
    mapped[np.isclose(prob_left, 0.2)] = 0
    mapped[np.isclose(prob_left, 0.5)] = 1
    mapped[np.isclose(prob_left, 0.8)] = 2
    if np.any(mapped < 0):
        vals = np.unique(prob_left[mapped < 0])
        raise ValueError(f"Unexpected probabilityLeft values: {vals}")
    return mapped


def make_time_input() -> np.ndarray:
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)


def compute_tertile_edges(values: Iterable[np.ndarray]) -> tuple[float, float]:
    flat_values = [np.asarray(v, dtype=np.float64).ravel() for v in values if len(v)]
    if not flat_values:
        raise ValueError("Cannot compute tertile edges from empty values")
    concat = np.concatenate(flat_values)
    q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        mn = float(np.min(concat))
        mx = float(np.max(concat))
        if mn == mx:
            eps = np.finfo(np.float32).eps
            return mn + eps, mn + 2 * eps
        step = (mx - mn) / 3.0
        return mn + step, mn + 2.0 * step
    return float(q1), float(q2)


def discretize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m {sec:.1f}s"


def process_session(spec: SessionSpec, br: BrainRegions) -> ProcessedSession | None:
    t0 = time.perf_counter()
    trials = load_trials_table(spec.session_path)
    raw_n_trials = len(trials)
    trial_mask = compute_trial_mask(trials)
    block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())

    spike_times, spike_clusters, cluster_regions, n_clusters_total, n_clusters_good = load_session_spikes(
        spec.session_path, br
    )
    if n_clusters_good == 0:
        print(f"[skip] {spec.eid}: no good clusters after QC")
        return None

    try:
        wheel_times, wheel_speed = load_wheel_speed(spec.session_path)
        whisk_times, whisk_values, whisk_source = load_whisker_motion_energy(spec.session_path)
    except FileNotFoundError as exc:
        print(f"[skip] {spec.eid}: {exc}")
        return None

    masked_trials = trials.loc[trial_mask].reset_index(drop=False)
    if len(masked_trials) < 2:
        print(f"[skip] {spec.eid}: fewer than 2 trials after trial mask")
        return None

    align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
    neural_trials = bin_spikes_for_trials(
        spike_times,
        spike_clusters,
        n_clusters=n_clusters_good,
        align_times=align_times,
    )
    wheel_trials, wheel_mask = interpolate_behavior_trials(wheel_times, wheel_speed, align_times)
    whisk_trials, whisk_mask = interpolate_behavior_trials(whisk_times, whisk_values, align_times)
    neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
    combined_mask = wheel_mask & whisk_mask & neural_mask

    if combined_mask.sum() < 2:
        print(f"[skip] {spec.eid}: fewer than 2 trials after behavior/neural alignment")
        return None

    neural_keep = [neural_trials[i] for i, keep in enumerate(combined_mask) if keep]
    wheel_keep = [wheel_trials[i] for i, keep in enumerate(combined_mask) if keep]
    whisk_keep = [whisk_trials[i] for i, keep in enumerate(combined_mask) if keep]
    masked_keep = masked_trials.loc[combined_mask].reset_index(drop=True)

    time_input = make_time_input()
    inputs = []
    choice = []
    prior = []
    choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
    prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
    block_vals = block_trial_number[masked_keep["index"].to_numpy()]

    for block_num, choice_val, prior_val in zip(block_vals, choice_vals, prior_vals, strict=True):
        input_trial = np.vstack(
            [
                time_input,
                np.full(N_BINS, block_num, dtype=np.float32),
            ]
        ).astype(np.float32)
        inputs.append(input_trial)
        choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
        prior.append(np.full(N_BINS, prior_val, dtype=np.int16))

    elapsed = time.perf_counter() - t0
    print(
        f"[ok] {spec.eid}: raw_trials={raw_n_trials}, kept_trials={len(neural_keep)}, "
        f"good_clusters={n_clusters_good}, whisker={whisk_source}, elapsed={format_seconds(elapsed)}"
    )
    return ProcessedSession(
        eid=spec.eid,
        subject=spec.subject,
        neural=neural_keep,
        inputs=inputs,
        choice=choice,
        prior=prior,
        wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep],
        whisker_cont=[np.asarray(x, dtype=np.float32) for x in whisk_keep],
        cluster_regions=np.asarray(cluster_regions, dtype=object),
        kept_trial_indices=masked_keep["index"].to_numpy(dtype=np.int32),
        whisker_source=whisk_source,
        n_clusters_total=n_clusters_total,
        n_clusters_good=n_clusters_good,
        raw_n_trials=raw_n_trials,
        kept_n_trials=len(neural_keep),
    )


def process_session_worker(spec: SessionSpec) -> ProcessedSession | None:
    return process_session(spec, BrainRegions())


def plot_processing_summary(
    session: ProcessedSession,
    wheel_edges: tuple[float, float],
    whisker_edges: tuple[float, float],
    outdir: Path,
) -> None:
    if session.kept_n_trials == 0:
        return
    trial_idx = 0
    neural = session.neural[trial_idx].astype(np.float32)
    input_trial = session.inputs[trial_idx]
    wheel_cont = session.wheel_cont[trial_idx]
    whisk_cont = session.whisker_cont[trial_idx]
    wheel_disc = discretize_three_bins(wheel_cont, wheel_edges)
    whisk_disc = discretize_three_bins(whisk_cont, whisker_edges)
    sample_neurons = min(40, neural.shape[0])
    neural_sample = neural[:sample_neurons]
    t = input_trial[0]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex="col")
    ax = axes.ravel()

    im = ax[0].imshow(neural_sample, aspect="auto", interpolation="nearest", cmap="viridis")
    ax[0].set_title("Binned Spike Counts")
    ax[0].set_ylabel("Neuron")
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    ax[1].plot(t, input_trial[0], label="time since stim")
    ax[1].plot(t, input_trial[1], label="trial number in block")
    ax[1].axvline(0.0, color="k", linestyle="--", alpha=0.5)
    ax[1].set_title("Decoder Inputs")
    ax[1].legend(loc="upper left")

    ax[2].plot(t, wheel_cont, color="tab:blue", label="wheel speed")
    ax[2].axhline(wheel_edges[0], color="tab:blue", linestyle=":")
    ax[2].axhline(wheel_edges[1], color="tab:blue", linestyle=":")
    ax[2].axvline(0.0, color="k", linestyle="--", alpha=0.5)
    ax[2].set_title("Wheel Speed Continuous")
    ax[2].set_ylabel("speed")

    ax[3].step(t, wheel_disc, where="mid", color="tab:blue")
    ax[3].axvline(0.0, color="k", linestyle="--", alpha=0.5)
    ax[3].set_title("Wheel Speed Discretized")
    ax[3].set_ylim(-0.2, 2.2)

    ax[4].plot(t, whisk_cont, color="tab:orange", label=f"whisker ({session.whisker_source})")
    ax[4].axhline(whisker_edges[0], color="tab:orange", linestyle=":")
    ax[4].axhline(whisker_edges[1], color="tab:orange", linestyle=":")
    ax[4].axvline(0.0, color="k", linestyle="--", alpha=0.5)
    ax[4].set_title("Whisker Motion Energy Continuous")
    ax[4].set_xlabel("Time from stimulus onset (s)")
    ax[4].set_ylabel("motion energy")

    ax[5].step(t, whisk_disc, where="mid", color="tab:orange")
    ax[5].axvline(0.0, color="k", linestyle="--", alpha=0.5)
    ax[5].set_title("Whisker Motion Energy Discretized")
    ax[5].set_xlabel("Time from stimulus onset (s)")
    ax[5].set_ylim(-0.2, 2.2)

    fig.suptitle(f"Processing summary: {session.eid}")
    fig.tight_layout()
    fig.savefig(outdir / f"processing_{session.eid}.png", dpi=150)
    plt.close(fig)


def build_data_dict(
    sessions: list[ProcessedSession],
    wheel_edges: tuple[float, float],
    whisker_edges: tuple[float, float],
) -> dict:
    subjects = sorted({s.subject for s in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    brain_regions = sorted({str(region) for s in sessions for region in s.cluster_regions})
    brain_region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16),
        "brain_regions": brain_regions,
        "brain_region_idx": [],
        "input_names": ["time_since_stimulus_onset", "trial_number_in_block"],
        "output_names": [
            "choice",
            "prior_probability_of_left",
            "wheel_speed_bin",
            "whisker_motion_energy_bin",
        ],
        "output_values": [
            ["left", "right"],
            ["p_left_0.2", "p_left_0.5", "p_left_0.8"],
            ["low", "medium", "high"],
            ["low", "medium", "high"],
        ],
        "metadata": {
            "task_description": (
                "IBL visual decision task; decode choice, prior probability of left, "
                "wheel speed bin, and whisker motion energy bin from stimulus-onset-aligned "
                "spike counts in the IBL brain-wide map release."
            ),
            "source_dataset": TARGET_DATASET_NAME,
            "time_bin_size": 20.0,
            "temporal_alignment_event": "stimulus onset (stimOn_times)",
            "off_start": TIME_WINDOW[0],
            "off_end": TIME_WINDOW[1],
            "neuron_filter": "clusters.metrics.label >= 1",
            "trial_filter": (
                "required events present; 0.08 <= firstMovement_times - stimOn_times <= 2.0 s; "
                "choice != 0; feedback_times - goCue_times <= 10 s"
            ),
            "reference_manifest": "2025_Q3_IBL_et_al_BWM/sessions.pqt",
            "wheel_speed_discretization_edges": [float(wheel_edges[0]), float(wheel_edges[1])],
            "whisker_motion_energy_discretization_edges": [
                float(whisker_edges[0]),
                float(whisker_edges[1]),
            ],
            "session_eids": [s.eid for s in sessions],
        },
    }

    for session in sessions:
        data["neural"].append([trial.astype(np.float16) for trial in session.neural])
        data["input"].append([trial.astype(np.float32) for trial in session.inputs])
        session_output = []
        for choice, prior, wheel_cont, whisk_cont in zip(
            session.choice,
            session.prior,
            session.wheel_cont,
            session.whisker_cont,
            strict=True,
        ):
            output_trial = np.vstack(
                [
                    choice,
                    prior,
                    discretize_three_bins(wheel_cont, wheel_edges),
                    discretize_three_bins(whisk_cont, whisker_edges),
                ]
            ).astype(np.int16)
            session_output.append(output_trial)
        data["output"].append(session_output)
        data["brain_region_idx"].append(
            np.array([brain_region_to_idx[str(r)] for r in session.cluster_regions], dtype=np.int16)
        )
    return data


def summarize_sessions(sessions: list[ProcessedSession]) -> None:
    if not sessions:
        return
    total_trials = sum(s.kept_n_trials for s in sessions)
    total_good_clusters = sum(s.n_clusters_good for s in sessions)
    total_raw_clusters = sum(s.n_clusters_total for s in sessions)
    print("Processed dataset summary:")
    print(f"  sessions: {len(sessions)}")
    print(f"  subjects: {len({s.subject for s in sessions})}")
    print(f"  trials: {total_trials}")
    print(f"  good clusters: {total_good_clusters}")
    print(f"  raw clusters before QC: {total_raw_clusters}")
    print(f"  mean trials/session: {np.mean([s.kept_n_trials for s in sessions]):.1f}")
    print(f"  mean good clusters/session: {np.mean([s.n_clusters_good for s in sessions]):.1f}")


def choose_num_workers(mode: str, nsessions: int) -> int:
    if nsessions < 2:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count, nsessions))


def main() -> None:
    args = parse_args()
    mode = "sample" if args.sample else "full"
    t_start = time.perf_counter()

    specs, missing_eids = resolve_session_specs()
    if missing_eids:
        print(f"Missing or unavailable local sessions from manifest: {len(missing_eids)}")
        print("  " + ", ".join(missing_eids[:10]) + (" ..." if len(missing_eids) > 10 else ""))
    if args.sample:
        specs = specs[:2]

    print(f"Requested mode: {mode}")
    print(f"Candidate local sessions from manifest: {len(specs)}")
    processed: list[ProcessedSession] = []
    num_workers = choose_num_workers(mode, len(specs))
    print(f"Session workers: {num_workers}")
    if num_workers == 1:
        for spec in specs:
            session = process_session_worker(spec)
            if session is not None:
                processed.append(session)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for session in executor.map(process_session_worker, specs):
                if session is not None:
                    processed.append(session)

    if len(processed) < 2:
        raise RuntimeError("Fewer than 2 sessions survived processing; cannot build dataset")

    wheel_edges = compute_tertile_edges(
        trial_values for session in processed for trial_values in session.wheel_cont
    )
    whisker_edges = compute_tertile_edges(
        trial_values for session in processed for trial_values in session.whisker_cont
    )
    print(f"Wheel speed tertile edges: {wheel_edges}")
    print(f"Whisker motion energy tertile edges: {whisker_edges}")

    data = build_data_dict(processed, wheel_edges, whisker_edges)
    summarize_sessions(processed)

    if args.show_processing:
        outdir = Path(args.outpicklefile).resolve().parent
        for session in processed[:MAX_PROCESSING_PLOTS]:
            plot_processing_summary(session, wheel_edges, whisker_edges, outdir)

    outpath = Path(args.outpicklefile).resolve()
    with outpath.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - t_start
    print(f"Saved converted dataset to {outpath}")
    print(f"Total elapsed time: {format_seconds(elapsed)}")


if __name__ == "__main__":
    main()
