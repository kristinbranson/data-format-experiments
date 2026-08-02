#!/usr/bin/env python3
"""Convert IBL brain-wide map data into the decoder pickle format."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "ibllib"))
from brainbox.behavior.wheel import interpolate_position, velocity_filtered  # noqa: E402


RELEASE_CSV = Path("code/code_zhang2025/data/bwm_release.csv")
DATA_ROOT = Path("data/one_cache")

ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
COMMON_RELATIVE_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS, dtype=np.float32)

INPUT_NAMES = [
    "time_since_stimulus_onset_s",
    "trial_number_in_block",
]
OUTPUT_NAMES = [
    "choice",
    "prior_probability_of_left",
    "wheel_speed_bin",
    "whisker_motion_energy_bin",
]
OUTPUT_VALUES = [
    ["left", "right"],
    ["0.2", "0.5", "0.8"],
    ["low", "medium", "high"],
    ["low", "medium", "high"],
]


@dataclass(frozen=True)
class SessionSpec:
    eid: str
    subject: str
    lab: str
    date: str
    session_number: int
    probe_names: tuple[str, ...]

    @property
    def session_path(self) -> Path:
        return DATA_ROOT / self.lab / "Subjects" / self.subject / self.date / f"{self.session_number:03d}"


@dataclass
class PreparedSession:
    spec: SessionSpec
    trials: pd.DataFrame
    keep_mask: np.ndarray
    trial_number_in_block: np.ndarray
    choice_raw: np.ndarray
    prior_raw: np.ndarray
    wheel_cont: np.ndarray
    whisker_cont: np.ndarray
    n_good_units: int
    whisker_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IBL BWM data into decoder pickle format.")
    parser.add_argument("outpicklefile", type=Path, help="Output pickle path.")
    parser.add_argument("--full", action="store_true", help="Process all eligible sessions (default).")
    parser.add_argument("--sample", action="store_true", help="Process only 2 eligible sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save detailed processing figures for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def choose_n_workers(mode: str, show_processing: bool) -> int:
    if mode == "sample" or show_processing:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(32, cpu_count))


def resolve_latest(base: Path, pattern: str) -> Path:
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {pattern} under {base}")
    return matches[-1]


def load_release_sessions() -> list[SessionSpec]:
    bwm = pd.read_csv(RELEASE_CSV, index_col=0)
    grouped = bwm.groupby("eid", sort=False)
    sessions: list[SessionSpec] = []
    for eid, df in grouped:
        row = df.iloc[0]
        probe_names = tuple(df["probe_name"].tolist())
        sessions.append(
            SessionSpec(
                eid=eid,
                subject=str(row["subject"]),
                lab=str(row["lab"]),
                date=str(row["date"]),
                session_number=int(row["session_number"]),
                probe_names=probe_names,
            )
        )
    return sessions


def load_trials_table(session_path: Path) -> pd.DataFrame:
    path = resolve_latest(session_path / "alf", "**/_ibl_trials.table.pqt")
    return pd.read_parquet(path)


def compute_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    rt = trials["firstMovement_times"] - trials["stimOn_times"]
    mask = (
        ~trials["stimOn_times"].isnull()
        & ~trials["choice"].isnull()
        & ~trials["feedback_times"].isnull()
        & ~trials["probabilityLeft"].isnull()
        & ~trials["firstMovement_times"].isnull()
        & ~trials["feedbackType"].isnull()
        & (rt >= 0.08)
        & (rt <= 2.0)
        & ((trials["feedback_times"] - trials["goCue_times"]) <= 10.0)
        & (trials["choice"] != 0)
    )
    return mask.to_numpy(dtype=bool)


def compute_trial_number_in_block(prob_left: np.ndarray) -> np.ndarray:
    out = np.zeros(len(prob_left), dtype=np.int16)
    prev = None
    counter = 0
    for i, val in enumerate(prob_left):
        current = None if pd.isna(val) else float(val)
        if i == 0 or current != prev:
            counter = 1
        else:
            counter += 1
        out[i] = counter
        prev = current
    return out


def count_good_units(spec: SessionSpec) -> int:
    total = 0
    for probe_name in spec.probe_names:
        probe_dir = spec.session_path / "alf" / probe_name / "pykilosort"
        metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
        metrics = pd.read_parquet(metrics_path, columns=["label"])
        total += int((metrics["label"].to_numpy() >= 1).sum())
    return total


def load_wheel_speed(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    alf_path = session_path / "alf"
    timestamps = np.load(resolve_latest(alf_path, "**/_ibl_wheel.timestamps.npy"))
    position = np.load(resolve_latest(alf_path, "**/_ibl_wheel.position.npy"))
    if timestamps.shape[0] != position.shape[0]:
        raise ValueError(f"Wheel timestamp/position length mismatch for {session_path}")
    interp_pos, interp_times = interpolate_position(timestamps, position, freq=1000)
    velocity, _ = velocity_filtered(interp_pos, fs=1000, corner_frequency=20, order=8)
    return interp_times.astype(np.float64), np.abs(velocity).astype(np.float32)


def check_video_timestamps(view: str, video_timestamps: np.ndarray, video_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if video_timestamps.shape[0] < video_data.shape[0]:
        if video_timestamps.shape[0] == 0:
            raise ValueError(f"Camera times empty for {view}Camera.")
        raise ValueError(f"Camera times are shorter than video data for {view}Camera.")
    if video_timestamps.shape[0] > video_data.shape[0]:
        video_timestamps = video_timestamps[-video_data.shape[0]:]
    return video_timestamps, video_data


def load_whisker_motion_energy(session_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    alf_path = session_path / "alf"
    for view in ("left", "right"):
        me_candidates = sorted(alf_path.glob(f"**/{view}Camera.ROIMotionEnergy.npy"))
        ts_candidates = sorted(alf_path.glob(f"**/_ibl_{view}Camera.times.npy"))
        if not me_candidates or not ts_candidates:
            continue
        motion_energy = np.load(me_candidates[-1])
        timestamps = np.load(ts_candidates[-1])
        timestamps, motion_energy = check_video_timestamps(view, timestamps, motion_energy)
        return timestamps.astype(np.float64), motion_energy.astype(np.float32), view
    raise FileNotFoundError(f"No whisker motion energy stream found for {session_path}")


def interpolate_behavior_per_trial(
    target_times: np.ndarray,
    target_vals: np.ndarray,
    align_times: np.ndarray,
    binsize: float = BINSIZE,
    time_window: tuple[float, float] = TIME_WINDOW,
) -> tuple[list[np.ndarray | None], np.ndarray]:
    interval_begs = align_times + time_window[0]
    interval_ends = align_times + time_window[1]
    n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))

    idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
    idxs_end = np.searchsorted(target_times, interval_ends, side="left")

    outputs: list[np.ndarray | None] = [None] * len(align_times)
    mask = np.zeros(len(align_times), dtype=bool)

    for i in range(len(align_times)):
        t_beg = interval_begs[i]
        t_end = interval_ends[i]
        if np.isnan(t_beg) or np.isnan(t_end):
            continue

        curr_times = target_times[idxs_beg[i]:idxs_end[i]]
        curr_vals = target_vals[idxs_beg[i]:idxs_end[i]]
        if curr_vals.shape[0] == 0:
            continue
        if np.isnan(curr_vals).any():
            continue
        if abs(t_beg - curr_times[0]) > binsize:
            continue
        if abs(t_end - curr_times[-1]) > binsize:
            continue

        x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
        y_interp = np.interp(x_interp, curr_times, curr_vals)
        outputs[i] = y_interp.astype(np.float32)
        mask[i] = True

    return outputs, mask


def build_session_behavior(spec: SessionSpec) -> PreparedSession | None:
    trials = load_trials_table(spec.session_path)
    trial_mask = compute_trial_mask(trials)
    trial_number_in_block = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
    n_good_units = count_good_units(spec)
    if n_good_units == 0:
        return None

    wheel_times, wheel_speed = load_wheel_speed(spec.session_path)
    whisker_times, whisker_motion, whisker_source = load_whisker_motion_energy(spec.session_path)

    align_times = trials[ALIGN_EVENT].to_numpy(dtype=np.float64)
    wheel_interp, wheel_mask = interpolate_behavior_per_trial(wheel_times, wheel_speed, align_times)
    whisker_interp, whisker_mask = interpolate_behavior_per_trial(whisker_times, whisker_motion, align_times)

    keep_mask = trial_mask & wheel_mask & whisker_mask
    if keep_mask.sum() < 2:
        return None

    wheel_cont = np.stack([wheel_interp[i] for i in np.where(keep_mask)[0]], axis=0)
    whisker_cont = np.stack([whisker_interp[i] for i in np.where(keep_mask)[0]], axis=0)

    return PreparedSession(
        spec=spec,
        trials=trials,
        keep_mask=keep_mask,
        trial_number_in_block=trial_number_in_block[keep_mask],
        choice_raw=trials.loc[keep_mask, "choice"].to_numpy(),
        prior_raw=trials.loc[keep_mask, "probabilityLeft"].to_numpy(),
        wheel_cont=wheel_cont,
        whisker_cont=whisker_cont,
        n_good_units=n_good_units,
        whisker_source=whisker_source,
    )


def build_session_behavior_safe(spec: SessionSpec) -> tuple[str, PreparedSession | None, str | None]:
    try:
        prepared = build_session_behavior(spec)
        if prepared is None:
            return spec.eid, None, "no_good_units_or_too_few_valid_trials"
        return spec.eid, prepared, None
    except FileNotFoundError:
        return spec.eid, None, "missing_required_stream"
    except Exception as exc:  # pragma: no cover - defensive for raw-data irregularities
        return spec.eid, None, f"{type(exc).__name__}: {exc}"


def robust_tertile_edges(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 0.0
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    if q2 > q1:
        return float(q1), float(q2)

    sorted_vals = np.sort(finite)
    i1 = max(0, min(sorted_vals.size - 1, sorted_vals.size // 3))
    i2 = max(0, min(sorted_vals.size - 1, (2 * sorted_vals.size) // 3))
    q1 = float(sorted_vals[i1])
    q2 = float(sorted_vals[i2])
    if q2 <= q1:
        larger = sorted_vals[sorted_vals > q1]
        q2 = float(larger[0]) if larger.size else q1
    return q1, q2


def digitize_three_bins(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low, high = edges
    if high <= low:
        return np.zeros(values.shape, dtype=np.int8)
    return np.digitize(values, bins=np.array([low, high], dtype=np.float32), right=False).astype(np.int8)


def load_good_spikes_and_regions(spec: SessionSpec, brain_regions: BrainRegions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spike_times_all: list[np.ndarray] = []
    spike_clusters_all: list[np.ndarray] = []
    cluster_regions_all: list[np.ndarray] = []
    cluster_offset = 0

    for probe_name in spec.probe_names:
        probe_dir = spec.session_path / "alf" / probe_name / "pykilosort"
        metrics_path = resolve_latest(probe_dir, "**/clusters.metrics.pqt")
        clusters_channels_path = resolve_latest(probe_dir, "**/clusters.channels.npy")
        channels_region_ids_path = resolve_latest(probe_dir, "**/channels.brainLocationIds_ccf_2017.npy")
        spikes_times_path = resolve_latest(probe_dir, "**/spikes.times.npy")
        spikes_clusters_path = resolve_latest(probe_dir, "**/spikes.clusters.npy")

        metrics = pd.read_parquet(metrics_path, columns=["label"])
        good_rows = metrics["label"].to_numpy(copy=False) >= 1
        if not np.any(good_rows):
            continue

        spikes_times = np.load(spikes_times_path, mmap_mode="r")
        spikes_clusters = np.load(spikes_clusters_path, mmap_mode="r")
        n_clusters = good_rows.shape[0]
        valid_spikes = spikes_clusters < n_clusters
        if np.all(valid_spikes):
            spike_mask = good_rows[spikes_clusters]
        else:
            spike_mask = np.zeros(spikes_clusters.shape[0], dtype=bool)
            spike_mask[valid_spikes] = good_rows[spikes_clusters[valid_spikes]]
        remap = np.full(n_clusters, -1, dtype=np.int32)
        remap[good_rows] = np.arange(int(good_rows.sum()), dtype=np.int32)
        good_spike_times = np.asarray(spikes_times[spike_mask], dtype=np.float64)
        good_spike_clusters = remap[spikes_clusters[spike_mask]] + cluster_offset

        cluster_channels = np.load(clusters_channels_path, mmap_mode="r")[good_rows]
        channel_region_ids = np.load(channels_region_ids_path, mmap_mode="r")
        region_ids = np.zeros(cluster_channels.shape[0], dtype=np.int64)
        valid_channel = (cluster_channels >= 0) & (cluster_channels < channel_region_ids.shape[0])
        region_ids[valid_channel] = channel_region_ids[cluster_channels[valid_channel]]
        cluster_regions = brain_regions.id2acronym(region_ids, mapping="Beryl").astype(str)

        spike_times_all.append(good_spike_times)
        spike_clusters_all.append(good_spike_clusters)
        cluster_regions_all.append(cluster_regions)
        cluster_offset += cluster_regions.shape[0]

    if cluster_offset == 0:
        raise ValueError(f"No good units remained for {spec.eid}")

    merged_spike_times = np.concatenate(spike_times_all)
    merged_spike_clusters = np.concatenate(spike_clusters_all)
    order = np.argsort(merged_spike_times)
    merged_spike_times = merged_spike_times[order]
    merged_spike_clusters = merged_spike_clusters[order]
    merged_cluster_regions = np.concatenate(cluster_regions_all)
    return merged_spike_times, merged_spike_clusters, merged_cluster_regions


def bin_spikes_by_trial(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    align_times: np.ndarray,
    n_units: int,
    binsize: float = BINSIZE,
    time_window: tuple[float, float] = TIME_WINDOW,
) -> list[np.ndarray]:
    interval_begs = align_times + time_window[0]
    interval_ends = align_times + time_window[1]
    n_bins = int(np.ceil((time_window[1] - time_window[0]) / binsize))
    start_idx = np.searchsorted(spike_times, interval_begs, side="left")
    end_idx = np.searchsorted(spike_times, interval_ends, side="left")

    out: list[np.ndarray] = []
    for i in range(len(align_times)):
        i0 = start_idx[i]
        i1 = end_idx[i]
        if i1 <= i0:
            out.append(np.zeros((n_units, n_bins), dtype=np.float16))
            continue

        rel = spike_times[i0:i1] - interval_begs[i]
        bin_idx = np.floor(rel / binsize).astype(np.int64)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        if not np.any(valid):
            out.append(np.zeros((n_units, n_bins), dtype=np.float16))
            continue

        flat = spike_clusters[i0:i1][valid] * n_bins + bin_idx[valid]
        counts = np.bincount(flat, minlength=n_units * n_bins).reshape(n_units, n_bins)
        out.append(counts.astype(np.float16))
    return out


def map_choice(raw_choice: np.ndarray) -> np.ndarray:
    mapped = np.empty(raw_choice.shape[0], dtype=np.int8)
    mapped[raw_choice == 1] = 0
    mapped[raw_choice == -1] = 1
    if not np.all(np.isin(raw_choice, [-1, 1])):
        raise ValueError("Unexpected choice values encountered after filtering.")
    return mapped


def map_prior(raw_prior: np.ndarray) -> np.ndarray:
    out = np.empty(raw_prior.shape[0], dtype=np.int8)
    mapper = {0.2: 0, 0.5: 1, 0.8: 2}
    for i, val in enumerate(raw_prior):
        key = round(float(val), 1)
        if key not in mapper:
            raise ValueError(f"Unexpected probabilityLeft value {val}")
        out[i] = mapper[key]
    return out


def build_session_payload(
    prepared: PreparedSession,
    wheel_edges: tuple[float, float],
    whisker_edges: tuple[float, float],
) -> tuple[str, list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, float]:
    session_start = time.time()
    brain_regions = BrainRegions()
    kept_align_times = prepared.trials.loc[prepared.keep_mask, ALIGN_EVENT].to_numpy(dtype=np.float64)
    spike_times, spike_clusters, cluster_regions = load_good_spikes_and_regions(prepared.spec, brain_regions)
    neural_trials = bin_spikes_by_trial(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        align_times=kept_align_times,
        n_units=cluster_regions.shape[0],
    )

    choice = map_choice(prepared.choice_raw)
    prior = map_prior(prepared.prior_raw)
    wheel_bins = digitize_three_bins(prepared.wheel_cont, wheel_edges)
    whisker_bins = digitize_three_bins(prepared.whisker_cont, whisker_edges)

    session_input: list[np.ndarray] = []
    session_output: list[np.ndarray] = []
    for trial_idx in range(len(neural_trials)):
        inp = np.vstack(
            [
                COMMON_RELATIVE_TIMES,
                np.full(NBINS, prepared.trial_number_in_block[trial_idx], dtype=np.float32),
            ]
        ).astype(np.float32)
        out = np.vstack(
            [
                np.full(NBINS, choice[trial_idx], dtype=np.int8),
                np.full(NBINS, prior[trial_idx], dtype=np.int8),
                wheel_bins[trial_idx],
                whisker_bins[trial_idx],
            ]
        ).astype(np.int8)
        session_input.append(inp)
        session_output.append(out)

    elapsed = time.time() - session_start
    return prepared.spec.eid, neural_trials, session_input, session_output, cluster_regions.astype(str), elapsed


def make_processing_plot(
    prepared: PreparedSession,
    neural_trials: list[np.ndarray],
    wheel_edges: tuple[float, float],
    whisker_edges: tuple[float, float],
) -> None:
    trial_idx = 0
    wheel_bins = digitize_three_bins(prepared.wheel_cont[trial_idx], wheel_edges)
    whisker_bins = digitize_three_bins(prepared.whisker_cont[trial_idx], whisker_edges)
    neural = neural_trials[trial_idx]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(f"Processing review for session {prepared.spec.eid}")

    axes[0].imshow(neural[: min(60, neural.shape[0]), :], aspect="auto", cmap="viridis")
    axes[0].set_title(f"Stimulus-aligned spike counts, trial 0 ({neural.shape[0]} good units)")
    axes[0].set_ylabel("Neuron")

    axes[1].plot(COMMON_RELATIVE_TIMES, prepared.wheel_cont[trial_idx], color="black", lw=1.2)
    axes[1].step(COMMON_RELATIVE_TIMES, wheel_bins, where="mid", color="tab:red", alpha=0.7)
    axes[1].axvline(0.0, color="tab:blue", ls="--", lw=1)
    axes[1].set_title(f"Wheel speed interpolation and discretization (edges={wheel_edges})")
    axes[1].set_ylabel("Speed / bin")

    axes[2].plot(COMMON_RELATIVE_TIMES, prepared.whisker_cont[trial_idx], color="black", lw=1.2)
    axes[2].step(COMMON_RELATIVE_TIMES, whisker_bins, where="mid", color="tab:green", alpha=0.7)
    axes[2].axvline(0.0, color="tab:blue", ls="--", lw=1)
    axes[2].set_title(
        f"Whisker motion energy ({prepared.whisker_source}) interpolation and discretization "
        f"(edges={whisker_edges})"
    )
    axes[2].set_ylabel("Energy / bin")

    choice = int(map_choice(prepared.choice_raw[[trial_idx]])[0])
    prior = int(map_prior(prepared.prior_raw[[trial_idx]])[0])
    block_num = int(prepared.trial_number_in_block[trial_idx])
    axes[3].axis("off")
    axes[3].text(
        0.01,
        0.95,
        "\n".join(
            [
                f"Kept trials: {prepared.keep_mask.sum()} / {len(prepared.keep_mask)}",
                f"Good units (label>=1): {prepared.n_good_units}",
                f"Trial 0 choice bin: {choice} ({OUTPUT_VALUES[0][choice]})",
                f"Trial 0 prior bin: {prior} ({OUTPUT_VALUES[1][prior]})",
                f"Trial 0 trial_number_in_block: {block_num}",
                f"Alignment window: {TIME_WINDOW} s, binsize={BINSIZE:.3f} s",
            ]
        ),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(f"processing_{prepared.spec.eid}.png", dpi=150)
    plt.close(fig)


def build_dataset(
    prepared_sessions: list[PreparedSession],
    wheel_edges: tuple[float, float],
    whisker_edges: tuple[float, float],
    show_processing: bool,
    n_workers: int,
) -> dict[str, Any]:
    neural_all: list[list[np.ndarray]] = []
    input_all: list[list[np.ndarray]] = []
    output_all: list[list[np.ndarray]] = []
    subject_names = sorted({ps.spec.subject for ps in prepared_sessions})
    subject_to_idx = {subject: i for i, subject in enumerate(subject_names)}
    subject_idx_list: list[int] = []

    all_region_names: set[str] = set()
    session_region_names: list[np.ndarray] = []
    session_ids: list[str] = []
    excluded_session_notes: list[dict[str, Any]] = []

    t0 = time.time()
    if n_workers == 1:
        for session_i, prepared in enumerate(prepared_sessions, start=1):
            eid, neural_trials, session_input, session_output, cluster_regions, elapsed = build_session_payload(
                prepared, wheel_edges, whisker_edges
            )
            neural_all.append(neural_trials)
            input_all.append(session_input)
            output_all.append(session_output)
            session_region_names.append(cluster_regions)
            all_region_names.update(cluster_regions.tolist())
            session_ids.append(eid)
            subject_idx_list.append(subject_to_idx[prepared.spec.subject])

            print(
                f"[build] {session_i:03d}/{len(prepared_sessions)} {eid} "
                f"trials={len(neural_trials)} neurons={cluster_regions.shape[0]} "
                f"time={elapsed:.2f}s"
            )

            if show_processing and session_i <= 2:
                make_processing_plot(prepared, neural_trials, wheel_edges, whisker_edges)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_index = {
                pool.submit(build_session_payload, prepared, wheel_edges, whisker_edges): idx
                for idx, prepared in enumerate(prepared_sessions)
            }
            for completed_i, future in enumerate(as_completed(future_to_index), start=1):
                idx = future_to_index[future]
                prepared = prepared_sessions[idx]
                eid, neural_trials, session_input, session_output, cluster_regions, elapsed = future.result()

                neural_all.append(neural_trials)
                input_all.append(session_input)
                output_all.append(session_output)
                session_region_names.append(cluster_regions)
                all_region_names.update(cluster_regions.tolist())
                session_ids.append(eid)
                subject_idx_list.append(subject_to_idx[prepared.spec.subject])

                print(
                    f"[build] {completed_i:03d}/{len(prepared_sessions)} {eid} "
                    f"trials={len(neural_trials)} neurons={cluster_regions.shape[0]} "
                    f"time={elapsed:.2f}s"
                )

    brain_region_list = sorted(all_region_names)
    brain_region_to_idx = {name: i for i, name in enumerate(brain_region_list)}
    brain_region_idx = [
        np.array([brain_region_to_idx[name] for name in session_regions], dtype=np.int64)
        for session_regions in session_region_names
    ]

    print(f"[build] completed in {time.time() - t0:.2f}s")

    data = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subject_names,
        "subject_idx": np.array(subject_idx_list, dtype=np.int64),
        "brain_regions": brain_region_list,
        "brain_region_idx": brain_region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Stimulus-onset-aligned IBL brain-wide map decoding task with well-isolated ephys units; "
                "decode choice, block prior, wheel speed bin, and whisker motion energy bin."
            ),
            "time_bin_size": 20.0,
            "temporal_alignment_event": "stimulus onset (stimOn_times)",
            "off_start": TIME_WINDOW[0],
            "off_end": TIME_WINDOW[1],
            "session_ids": session_ids,
            "source_release": "code/code_zhang2025/data/bwm_release.csv",
            "neuron_qc_rule": "clusters.metrics.label >= 1",
            "trial_filtering": {
                "required_non_nan": [
                    "stimOn_times",
                    "choice",
                    "feedback_times",
                    "probabilityLeft",
                    "firstMovement_times",
                    "feedbackType",
                ],
                "reaction_time_s": [0.08, 2.0],
                "max_trial_len_s": 10.0,
                "exclude_nochoice": True,
            },
            "dynamic_output_bin_edges": {
                "wheel_speed_bin": [float(wheel_edges[0]), float(wheel_edges[1])],
                "whisker_motion_energy_bin": [float(whisker_edges[0]), float(whisker_edges[1])],
            },
            "dynamic_output_binning": "global tertile bins over all kept timepoints",
            "common_timepoints_relative_to_alignment_s": COMMON_RELATIVE_TIMES.astype(np.float32),
            "excluded_session_notes": excluded_session_notes,
            "n_workers": n_workers,
        },
    }
    return data


def summarize_prepared(prepared_sessions: list[PreparedSession]) -> None:
    n_trials = np.array([ps.keep_mask.sum() for ps in prepared_sessions], dtype=np.int64)
    n_neurons = np.array([ps.n_good_units for ps in prepared_sessions], dtype=np.int64)
    print(
        "[summary] sessions={} subjects={} trials_total={} trials_mean={:.2f} neurons_total={} neurons_mean={:.2f}".format(
            len(prepared_sessions),
            len({ps.spec.subject for ps in prepared_sessions}),
            int(n_trials.sum()),
            float(n_trials.mean()) if n_trials.size else 0.0,
            int(n_neurons.sum()),
            float(n_neurons.mean()) if n_neurons.size else 0.0,
        )
    )


def main() -> None:
    args = parse_args()
    mode = "sample" if args.sample else "full"
    n_workers = choose_n_workers(mode, args.show_processing)

    sessions = load_release_sessions()
    print(f"[setup] loaded {len(sessions)} release sessions from {RELEASE_CSV}")
    print(f"[setup] using {n_workers} worker(s)")

    if args.sample:
        target_sessions = sessions
    else:
        target_sessions = sessions

    prepared_sessions: list[PreparedSession] = []
    excluded: list[tuple[str, str]] = []

    first_pass_start = time.time()
    if n_workers == 1:
        results_iter = (build_session_behavior_safe(spec) for spec in target_sessions)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_iter = pool.map(build_session_behavior_safe, target_sessions)

    for idx, (eid, prepared, error) in enumerate(results_iter, start=1):
        if prepared is None:
            excluded.append((eid, error or "unknown"))
        else:
            prepared_sessions.append(prepared)
            print(
                f"[pass1] {idx:03d}/{len(target_sessions)} {eid} "
                f"kept_trials={prepared.keep_mask.sum()} good_units={prepared.n_good_units} "
                f"whisker={prepared.whisker_source}"
            )
            if args.sample and len(prepared_sessions) >= 2:
                break

    if not prepared_sessions:
        raise RuntimeError("No sessions passed preprocessing.")

    print(f"[pass1] completed in {time.time() - first_pass_start:.2f}s")
    print(f"[pass1] excluded sessions: {len(excluded)}")
    summarize_prepared(prepared_sessions)

    all_wheel = np.concatenate([ps.wheel_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
    all_whisker = np.concatenate([ps.whisker_cont.reshape(-1) for ps in prepared_sessions]).astype(np.float32)
    wheel_edges = robust_tertile_edges(all_wheel)
    whisker_edges = robust_tertile_edges(all_whisker)
    print(f"[bins] wheel tertile edges={wheel_edges}")
    print(f"[bins] whisker tertile edges={whisker_edges}")

    data = build_dataset(
        prepared_sessions=prepared_sessions,
        wheel_edges=wheel_edges,
        whisker_edges=whisker_edges,
        show_processing=args.show_processing,
        n_workers=n_workers,
    )
    data["metadata"]["excluded_sessions"] = excluded
    data["metadata"]["mode"] = mode

    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] wrote {args.outpicklefile}")
    print(
        "[done] final sessions={} trials_total={} neurons_total={}".format(
            len(data["neural"]),
            sum(len(x) for x in data["neural"]),
            sum(len(x) for x in data["brain_region_idx"]),
        )
    )


if __name__ == "__main__":
    main()
