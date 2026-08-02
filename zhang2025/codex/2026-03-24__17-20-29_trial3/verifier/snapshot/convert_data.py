#!/usr/bin/env python3
"""Convert IBL ephys data into the decoder pickle format."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent
CODE_SRC = ROOT / "code" / "code_zhang2025" / "src"
if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from one.api import ONE
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
from iblatlas.atlas import BrainRegions
from iblutil.numerical import bincount2D, ismember
from utils.ibl_data_utils import merge_probes


BWM_RELEASE_CSV = ROOT / "code" / "code_zhang2025" / "data" / "bwm_release.csv"
ONE_CACHE_DIR = ROOT / "data" / "one_cache"

PARAMS = {
    "interval_len": 2.0,
    "binsize": 0.02,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
}

INPUT_NAMES = [
    "time_since_stimulus_onset_s",
    "trial_number_in_block",
]

OUTPUT_NAMES = [
    "choice",
    "prior_probability_left",
    "wheel_speed_bin",
    "whisker_motion_energy_bin",
]

OUTPUT_VALUES = [
    ["left", "right"],
    ["0.2", "0.5", "0.8"],
    ["low", "medium", "high"],
    ["low", "medium", "high"],
]

PRIOR_MAP = {
    0.2: 0,
    0.5: 1,
    0.8: 2,
}

BRAIN_REGIONS = BrainRegions()


@dataclass
class ProcessedSession:
    release_index: int
    eid: str
    subject: str
    lab: str
    date: str
    session_number: int
    neural: list[np.ndarray]
    input_data: list[np.ndarray]
    choice_labels: np.ndarray
    prior_labels: np.ndarray
    wheel_continuous: list[np.ndarray]
    whisker_continuous: list[np.ndarray]
    cluster_regions: list[str]
    cluster_good: np.ndarray
    raw_trial_count: int
    kept_trial_count: int
    kept_trial_indices: np.ndarray
    skipped_trial_count: int
    timing: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IBL sessions into the decoder pickle format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle filename.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Process all sessions.")
    group.add_argument(
        "--sample",
        action="store_true",
        help="Process only 2 successfully converted sessions.",
    )
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 converted sessions.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 8)),
        help="Workers used inside per-session spike/behavior alignment.",
    )
    parser.add_argument(
        "--session-workers",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 12)),
        help="Parallel workers across sessions for --full mode.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def is_likely_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_markers = [
        "expecting value",
        "jsondecodeerror",
        "read timed out",
        "connection aborted",
        "connection reset",
        "remote disconnected",
        "bad gateway",
        "service unavailable",
        "too many requests",
        "502",
        "503",
        "504",
    ]
    return any(marker in message for marker in transient_markers)


def build_one() -> ONE:
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        cache_dir=str(ONE_CACHE_DIR),
        cache_rest=None,
    )


def load_release_sessions() -> tuple[pd.DataFrame, pd.DataFrame]:
    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    sessions_df = (
        bwm_df[["eid", "lab", "subject", "date", "session_number"]]
        .drop_duplicates(subset=["eid"], keep="first")
        .reset_index(drop=True)
    )
    probe_map = (
        bwm_df[["eid", "pid", "probe_name"]]
        .astype({"pid": str, "probe_name": str})
        .groupby("eid")
        .apply(
            lambda frame: [
                (str(pid), str(probe_name))
                for pid, probe_name in zip(frame["pid"], frame["probe_name"])
            ],
            include_groups=False,
        )
        .to_dict()
    )
    sessions_df["probe_info"] = sessions_df["eid"].map(probe_map)
    return bwm_df, sessions_df


def build_time_axis() -> np.ndarray:
    start, end = PARAMS["time_window"]
    binsize = PARAMS["binsize"]
    n_bins = int(math.ceil((end - start) / binsize))
    # Match the reference behavior interpolation grid: bin end times.
    return np.linspace(start + binsize, end, n_bins, dtype=np.float32)


def choice_to_label(choice_values: np.ndarray) -> np.ndarray:
    choice_values = np.asarray(choice_values)
    if not np.all(np.isin(choice_values, [-1, 1])):
        bad = np.unique(choice_values[~np.isin(choice_values, [-1, 1])])
        raise ValueError(f"Unexpected choice values after filtering: {bad.tolist()}")
    # Official IBL docs: choice == -1 means chose right, choice == +1 means chose left.
    return (choice_values == -1).astype(np.int64)


def prior_to_label(probability_left: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(probability_left, dtype=float), 1)
    unknown = sorted(set(rounded.tolist()) - set(PRIOR_MAP))
    if unknown:
        raise ValueError(f"Unexpected probabilityLeft values: {unknown}")
    return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)


def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    out = np.zeros(probability_left.shape[0], dtype=np.float32)
    current = 0
    prev = None
    for idx, value in enumerate(probability_left):
        if idx == 0 or not np.isclose(value, prev):
            current = 1
        else:
            current += 1
        out[idx] = current
        prev = value
    return out


def find_latest_file(base_dir: Path, pattern: str) -> Path:
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {base_dir}")
    return matches[-1]


def get_spike_data_per_interval(
    times: np.ndarray,
    clusters: np.ndarray,
    interval_begs: np.ndarray,
    interval_ends: np.ndarray,
    interval_len: float,
    binsize: float,
) -> np.ndarray:
    n_intervals = len(interval_begs)
    n_bins = int(np.ceil(interval_len / binsize))
    cluster_ids = np.unique(clusters)
    n_clusters = len(cluster_ids)
    binned_spikes = np.zeros((n_intervals, n_clusters, n_bins), dtype=np.float32)

    for interval_idx, (t_beg, t_end) in enumerate(zip(interval_begs, interval_ends)):
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]
        if times_curr.shape[0] == 0:
            continue
        binned_tmp, _, cluster_idxs = bincount2D(
            times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end]
        )
        _, target_indices, _ = np.intersect1d(
            cluster_ids, cluster_idxs, return_indices=True
        )
        binned_spikes[interval_idx, target_indices, :] = binned_tmp[:, :n_bins]
    return binned_spikes


def bin_spiking_data_current(
    reg_clu_ids: np.ndarray,
    neural_dict: dict[str, Any],
    trials_df: pd.DataFrame,
) -> np.ndarray:
    intervals = np.vstack(
        [
            trials_df[PARAMS["align_time"]] + PARAMS["time_window"][0],
            trials_df[PARAMS["align_time"]] + PARAMS["time_window"][1],
        ]
    ).T
    spikemask = np.isin(neural_dict["spike_clusters"], reg_clu_ids)
    regspikes = neural_dict["spike_times"][spikemask]
    regclu = neural_dict["spike_clusters"][spikemask]
    interval_len = PARAMS["time_window"][1] - PARAMS["time_window"][0]
    binned_array = get_spike_data_per_interval(
        regspikes,
        regclu,
        interval_begs=intervals[:, 0],
        interval_ends=intervals[:, 1],
        interval_len=interval_len,
        binsize=PARAMS["binsize"],
    )
    return np.array([x.T for x in binned_array], dtype=np.float32)


def load_spiking_data_current(
    session_path: Path,
    probe_name: str,
    qc: int | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    sort_dir = find_latest_file(
        session_path / "alf" / probe_name / "pykilosort",
        "**/spikes.times.npy",
    ).parent
    spikes = {
        "times": np.load(sort_dir / "spikes.times.npy"),
        "clusters": np.load(sort_dir / "spikes.clusters.npy").astype(np.int32),
    }
    clusters_labeled = pd.read_parquet(sort_dir / "clusters.metrics.pqt")
    if "cluster_id" in clusters_labeled.columns:
        clusters_labeled = clusters_labeled.set_index("cluster_id", drop=False)
    cluster_channels = np.load(sort_dir / "clusters.channels.npy").astype(np.int64)
    channel_region_ids = np.load(sort_dir / "channels.brainLocationIds_ccf_2017.npy").astype(
        np.int64
    )
    good_channel_idx = np.clip(cluster_channels, 0, len(channel_region_ids) - 1)
    cluster_region_ids = channel_region_ids[good_channel_idx]
    clusters_labeled["acronym"] = BRAIN_REGIONS.id2acronym(cluster_region_ids)
    if qc is None:
        return spikes, clusters_labeled

    iok = clusters_labeled["label"] >= qc
    selected_clusters = clusters_labeled[iok].copy()
    spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index.to_numpy())
    selected_clusters.reset_index(drop=True, inplace=True)
    selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
    selected_spikes["clusters"] = selected_clusters.index.to_numpy()[ib].astype(np.int32)
    return selected_spikes, selected_clusters


def load_trials_and_mask_current(
    session_path: Path,
    min_rt: float = 0.08,
    max_rt: float = 2.0,
    max_trial_len: float | None = 10.0,
    exclude_nochoice: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    trials_path = find_latest_file(session_path / "alf", "**/_ibl_trials.table.pqt")
    trials = pd.read_parquet(trials_path).copy()

    query_parts = []
    if min_rt is not None:
        query_parts.append(f"(firstMovement_times - stimOn_times < {min_rt})")
    if max_rt is not None:
        query_parts.append(f"(firstMovement_times - stimOn_times > {max_rt})")
    if max_trial_len is not None:
        query_parts.append(f"(feedback_times - goCue_times > {max_trial_len})")
    for event in [
        "stimOn_times",
        "choice",
        "feedback_times",
        "probabilityLeft",
        "firstMovement_times",
        "feedbackType",
    ]:
        query_parts.append(f"{event}.isnull()")
    if exclude_nochoice:
        query_parts.append("(choice == 0)")

    if query_parts:
        query = " | ".join(query_parts)
        mask = ~trials.eval(query).to_numpy()
    else:
        mask = np.ones(len(trials), dtype=bool)
    return trials.reset_index(drop=True), mask


def load_target_behavior_current(session_path: Path, target: str) -> dict[str, Any]:
    alf_path = session_path / "alf"
    try:
        if target == "wheel-speed":
            wheel_timestamps = np.load(find_latest_file(alf_path, "**/_ibl_wheel.timestamps.npy"))
            wheel_position = np.load(find_latest_file(alf_path, "**/_ibl_wheel.position.npy"))
            if wheel_position.shape[0] != wheel_timestamps.shape[0]:
                raise ValueError("Length mismatch between wheel.position and wheel.timestamps")
            position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
            velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
            return {
                "times": np.asarray(times, dtype=np.float32),
                "values": np.abs(np.asarray(velocity, dtype=np.float32)),
            }
        if target == "left-whisker-motion-energy":
            times = np.load(find_latest_file(alf_path, "**/_ibl_leftCamera.times.npy"))
            values = np.load(find_latest_file(alf_path, "**/leftCamera.ROIMotionEnergy.npy"))
            if times.shape[0] < values.shape[0]:
                raise ValueError("Camera times are shorter than video data for leftCamera.")
            if times.shape[0] > values.shape[0]:
                times = times[-values.shape[0] :]
            return {
                "times": np.asarray(times, dtype=np.float32),
                "values": np.asarray(values, dtype=np.float32),
            }
        if target == "right-whisker-motion-energy":
            times = np.load(find_latest_file(alf_path, "**/_ibl_rightCamera.times.npy"))
            values = np.load(find_latest_file(alf_path, "**/rightCamera.ROIMotionEnergy.npy"))
            if times.shape[0] < values.shape[0]:
                raise ValueError("Camera times are shorter than video data for rightCamera.")
            if times.shape[0] > values.shape[0]:
                times = times[-values.shape[0] :]
            return {
                "times": np.asarray(times, dtype=np.float32),
                "values": np.asarray(values, dtype=np.float32),
            }
    except BaseException as exc:  # noqa: BLE001
        return {"times": None, "values": None, "skip": True, "error": str(exc)}

    raise NotImplementedError(target)


def get_behavior_per_interval_current(
    target_times: np.ndarray | None,
    target_vals: np.ndarray | None,
    trials_df: pd.DataFrame,
    allow_nans: bool = True,
) -> tuple[list[np.ndarray | None], np.ndarray, list[str | None]]:
    binsize = PARAMS["binsize"]
    start, end = PARAMS["time_window"]
    interval_len = end - start
    align_times = trials_df[PARAMS["align_time"]].to_numpy()
    interval_begs = align_times + start
    interval_ends = align_times + end
    n_intervals = len(interval_begs)
    n_bins = int(np.ceil(interval_len / binsize))

    if target_times is None or target_vals is None:
        return [None] * n_intervals, np.zeros(n_intervals, dtype=bool), ["missing"] * n_intervals

    idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
    idxs_end = np.searchsorted(target_times, interval_ends, side="left")
    target_times_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    interpolated = [None] * n_intervals
    good = np.zeros(n_intervals, dtype=bool)
    reasons: list[str | None] = [None] * n_intervals

    for idx, (t_seg, v_seg) in enumerate(zip(target_times_list, target_vals_list)):
        if len(v_seg) == 0:
            reasons[idx] = "target data not present"
            continue
        if np.sum(np.isnan(v_seg)) > 0 and not allow_nans:
            reasons[idx] = "nans in target data"
            continue
        if np.isnan(interval_begs[idx]) or np.isnan(interval_ends[idx]):
            reasons[idx] = "bad interval data"
            continue
        if np.abs(interval_begs[idx] - t_seg[0]) > binsize:
            reasons[idx] = "target data starts too late"
            continue
        if np.abs(interval_ends[idx] - t_seg[-1]) > binsize:
            reasons[idx] = "target data ends too early"
            continue

        x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
        if np.ndim(v_seg) > 1 and v_seg.shape[1] > 1:
            y_list = []
            for col in range(v_seg.shape[1]):
                fn = interp1d(t_seg, v_seg[:, col], kind="linear", fill_value="extrapolate")
                y_list.append(fn(x_interp))
            y_interp = np.column_stack(y_list)
        else:
            fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
            y_interp = fn(x_interp)

        interpolated[idx] = np.asarray(y_interp, dtype=np.float32)
        good[idx] = True

    return interpolated, good, reasons


def align_continuous_behavior(
    session_path: Path,
    behavior_name: str,
    trials_df: pd.DataFrame,
) -> tuple[list[np.ndarray | None], np.ndarray]:
    if behavior_name == "whisker-motion-energy":
        target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
        if target.get("skip"):
            target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
    else:
        target = load_target_behavior_current(session_path, behavior_name)
    traces, good_mask, _ = get_behavior_per_interval_current(
        target.get("times"),
        target.get("values"),
        trials_df=trials_df,
        allow_nans=True,
    )
    return traces, good_mask


def release_session_path(row: pd.Series) -> Path:
    return (
        ONE_CACHE_DIR
        / row["lab"]
        / "Subjects"
        / row["subject"]
        / row["date"]
        / f"{int(row['session_number']):03d}"
    )


def session_has_local_modalities(row: pd.Series) -> bool:
    alf = release_session_path(row) / "alf"
    if not alf.exists():
        return False
    has_trials = any(alf.glob("**/_ibl_trials.table.pqt"))
    has_wheel = any(alf.glob("**/_ibl_wheel.timestamps.npy"))
    has_whisk = any(alf.glob("**/leftCamera.ROIMotionEnergy.npy")) or any(
        alf.glob("**/rightCamera.ROIMotionEnergy.npy")
    )
    has_spikes = any(alf.glob("**/spikes.times.npy")) and any(
        alf.glob("**/spikes.clusters.npy")
    )
    return has_trials and has_wheel and has_whisk and has_spikes


def load_session_neural(
    session_path: Path,
    probe_info: list[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(probe_info) == 0:
        raise RuntimeError("No probe insertions found")

    spikes_list = []
    clusters_list = []
    for pid, probe_name in probe_info:
        spikes, clusters = load_spiking_data_current(
            session_path,
            probe_name=probe_name,
            qc=1,
        )
        if len(clusters) == 0:
            continue
        clusters["pid"] = str(pid)
        spikes_list.append(spikes)
        clusters_list.append(clusters)

    if not spikes_list:
        raise RuntimeError("No good clusters after QC filtering")

    spikes, clusters = merge_probes(spikes_list, clusters_list)
    neural_dict = {
        "spike_times": spikes["times"],
        "spike_clusters": spikes["clusters"],
        "cluster_regions": clusters["acronym"].to_numpy(),
    }
    meta = {
        "cluster_regions": list(clusters["acronym"]),
        "good_clusters": (clusters["label"] >= 1).to_numpy(dtype=np.int8),
        "cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()},
    }
    return neural_dict, meta


def process_one_session(
    row: pd.Series,
    time_axis: np.ndarray,
) -> ProcessedSession:
    eid = row["eid"]
    t0 = time.time()
    session_path = release_session_path(row)

    trials_df, trials_mask = load_trials_and_mask_current(session_path)
    t_trials = time.time()

    neural_dict, meta = load_session_neural(session_path, row["probe_info"])
    reg_clu_ids = np.arange(len(meta["cluster_regions"]), dtype=np.int64)
    binned_spikes = bin_spiking_data_current(reg_clu_ids, neural_dict, trials_df)
    t_spikes = time.time()

    wheel_traces, wheel_mask = align_continuous_behavior(session_path, "wheel-speed", trials_df)
    whisker_traces, whisker_mask = align_continuous_behavior(
        session_path, "whisker-motion-energy", trials_df
    )
    t_behavior = time.time()

    keep_mask = np.asarray(trials_mask, dtype=bool) & wheel_mask & whisker_mask
    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size < 2:
        raise RuntimeError(f"Only {keep_idx.size} valid trials after filtering")

    prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
    block_num_all = trial_number_in_block(prob_left_all)
    choice_all = trials_df["choice"].to_numpy()

    neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
    input_trials = [
        np.vstack(
            [
                time_axis,
                np.full(time_axis.shape[0], block_num_all[i], dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)
        for i in keep_idx
    ]
    wheel_aligned = [np.asarray(wheel_traces[i], dtype=np.float32) for i in keep_idx]
    whisker_aligned = [np.asarray(whisker_traces[i], dtype=np.float32) for i in keep_idx]

    if not all(x.shape == time_axis.shape for x in wheel_aligned):
        raise RuntimeError("Wheel trace shape mismatch after alignment")
    if not all(x.shape == time_axis.shape for x in whisker_aligned):
        raise RuntimeError("Whisker trace shape mismatch after alignment")

    choice_labels = choice_to_label(choice_all[keep_idx])
    prior_labels = prior_to_label(prob_left_all[keep_idx])

    return ProcessedSession(
        release_index=int(row["release_index"]),
        eid=eid,
        subject=row["subject"],
        lab=row["lab"],
        date=row["date"],
        session_number=int(row["session_number"]),
        neural=neural_trials,
        input_data=input_trials,
        choice_labels=choice_labels,
        prior_labels=prior_labels,
        wheel_continuous=wheel_aligned,
        whisker_continuous=whisker_aligned,
        cluster_regions=meta["cluster_regions"],
        cluster_good=np.asarray(meta["good_clusters"], dtype=np.int8),
        raw_trial_count=len(trials_df),
        kept_trial_count=len(keep_idx),
        kept_trial_indices=keep_idx.astype(np.int64),
        skipped_trial_count=int(len(trials_df) - len(keep_idx)),
        timing={
            "trials_s": t_trials - t0,
            "spikes_s": t_spikes - t_trials,
            "behavior_s": t_behavior - t_spikes,
            "total_s": t_behavior - t0,
        },
    )


def process_one_session_worker(row_dict: dict[str, Any], time_axis: np.ndarray) -> ProcessedSession:
    row = pd.Series(row_dict)
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            if attempt > 1:
                time.sleep(0.5 * attempt + random.random() * 0.5)
            return process_one_session(row, time_axis)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= 3 or not is_likely_transient_error(exc):
                raise
    assert last_exc is not None
    raise last_exc


def compute_thresholds(processed_sessions: list[ProcessedSession]) -> dict[str, tuple[float, float]]:
    wheel_values = np.concatenate(
        [np.concatenate(session.wheel_continuous) for session in processed_sessions]
    )
    whisker_values = np.concatenate(
        [np.concatenate(session.whisker_continuous) for session in processed_sessions]
    )
    thresholds = {}
    for name, values in [
        ("wheel_speed", wheel_values),
        ("whisker_motion_energy", whisker_values),
    ]:
        q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
        thresholds[name] = (float(q1), float(q2))
    return thresholds


def discretize(values: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    q1, q2 = thresholds
    return np.digitize(values, bins=np.array([q1, q2]), right=False).astype(np.int64)


def ordered_unique(strings: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in strings:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def to_beryl_acronyms(acronyms: list[str] | np.ndarray) -> np.ndarray:
    return np.asarray(BRAIN_REGIONS.acronym2acronym(list(acronyms), mapping="Beryl"))


def assemble_dataset(
    processed_sessions: list[ProcessedSession],
    thresholds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    subjects = ordered_unique([session.subject for session in processed_sessions])
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    session_regions_beryl = [
        to_beryl_acronyms(session.cluster_regions) for session in processed_sessions
    ]
    brain_regions = ordered_unique(
        [region for regions in session_regions_beryl for region in regions.tolist()]
    )
    region_to_idx = {region: idx for idx, region in enumerate(brain_regions)}

    neural = []
    input_data = []
    output_data = []
    subject_idx = []
    brain_region_idx = []

    for session, session_regions in zip(processed_sessions, session_regions_beryl):
        neural.append(session.neural)
        input_data.append(session.input_data)
        outputs_session = []
        for trial_idx in range(session.kept_trial_count):
            T = session.neural[trial_idx].shape[1]
            outputs_session.append(
                np.vstack(
                    [
                        np.full(T, session.choice_labels[trial_idx], dtype=np.int64),
                        np.full(T, session.prior_labels[trial_idx], dtype=np.int64),
                        discretize(
                            session.wheel_continuous[trial_idx],
                            thresholds["wheel_speed"],
                        ),
                        discretize(
                            session.whisker_continuous[trial_idx],
                            thresholds["whisker_motion_energy"],
                        ),
                    ]
                )
            )
        output_data.append(outputs_session)
        subject_idx.append(subject_to_idx[session.subject])
        brain_region_idx.append(
            np.array(
                [region_to_idx[region] for region in session_regions.tolist()],
                dtype=np.int64,
            )
        )

    metadata = {
        "task_description": (
            "IBL visual decision task converted to a stimulus-onset aligned trial dataset. "
            "Neural inputs are well-isolated spike counts; decoder targets are choice, prior "
            "probability of left, wheel-speed bins, and whisker-motion-energy bins."
        ),
        "time_bin_size": 20.0,
        "temporal_alignment_event": "stimulus onset (stimOn_times)",
        "off_start": float(PARAMS["time_window"][0]),
        "off_end": float(PARAMS["time_window"][1]),
        "time_axis_definition": "bin end times relative to stimulus onset",
        "source_release": "code/code_zhang2025/data/bwm_release.csv",
        "choice_mapping": {"1": "left", "-1": "right"},
        "prior_mapping": {"0.2": 0, "0.5": 1, "0.8": 2},
        "wheel_speed_thresholds": thresholds["wheel_speed"],
        "whisker_motion_energy_thresholds": thresholds["whisker_motion_energy"],
        "brain_region_mapping": "Beryl",
        "n_sessions_attempted": len(processed_sessions),
        "session_eids": [session.eid for session in processed_sessions],
    }

    return {
        "neural": neural,
        "input": input_data,
        "output": output_data,
        "subjects": subjects,
        "subject_idx": np.array(subject_idx, dtype=np.int64),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": metadata,
    }


def plot_processing(session: ProcessedSession, thresholds: dict[str, tuple[float, float]]) -> None:
    example_trial = 0
    neural_trial = session.neural[example_trial]
    input_trial = session.input_data[example_trial]
    wheel_trial = session.wheel_continuous[example_trial]
    whisker_trial = session.whisker_continuous[example_trial]
    wheel_bins = discretize(wheel_trial, thresholds["wheel_speed"])
    whisker_bins = discretize(whisker_trial, thresholds["whisker_motion_energy"])

    neuron_sample = min(50, neural_trial.shape[0])
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))

    axes[0, 0].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"EID: {session.eid}",
                f"Subject: {session.subject}",
                f"Trials kept: {session.kept_trial_count} / {session.raw_trial_count}",
                f"Neurons: {neural_trial.shape[0]}",
                f"Choice label (trial 0): {session.choice_labels[example_trial]}",
                f"Prior label (trial 0): {session.prior_labels[example_trial]}",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
    )
    axes[0, 0].axis("off")

    axes[0, 1].plot(input_trial[0], label="time since stim onset")
    axes[0, 1].plot(input_trial[1], label="trial number in block")
    axes[0, 1].set_title("Input Channels")
    axes[0, 1].legend(loc="best")

    im = axes[1, 0].imshow(
        neural_trial[:neuron_sample],
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title("Neural Spike Counts")
    axes[1, 0].set_xlabel("Time bin")
    axes[1, 0].set_ylabel("Neuron")

    axes[1, 1].hist(np.concatenate(session.wheel_continuous), bins=80, alpha=0.7)
    axes[1, 1].axvline(thresholds["wheel_speed"][0], color="r", linestyle="--")
    axes[1, 1].axvline(thresholds["wheel_speed"][1], color="r", linestyle="--")
    axes[1, 1].set_title("Wheel-Speed Thresholds")

    axes[2, 0].plot(wheel_trial, label="continuous")
    axes[2, 0].step(np.arange(len(wheel_bins)), wheel_bins, where="mid", label="bin")
    axes[2, 0].set_title("Wheel Speed Alignment / Discretization")
    axes[2, 0].legend(loc="best")

    axes[2, 1].hist(np.concatenate(session.whisker_continuous), bins=80, alpha=0.7)
    axes[2, 1].axvline(thresholds["whisker_motion_energy"][0], color="r", linestyle="--")
    axes[2, 1].axvline(thresholds["whisker_motion_energy"][1], color="r", linestyle="--")
    axes[2, 1].set_title("Whisker Thresholds")

    axes[3, 0].plot(whisker_trial, label="continuous")
    axes[3, 0].step(
        np.arange(len(whisker_bins)),
        whisker_bins,
        where="mid",
        label="bin",
    )
    axes[3, 0].set_title("Whisker Motion-Energy Alignment / Discretization")
    axes[3, 0].legend(loc="best")

    axes[3, 1].plot(
        session.prior_labels.astype(float),
        label="prior label",
        alpha=0.8,
    )
    axes[3, 1].plot(
        session.choice_labels.astype(float),
        label="choice label",
        alpha=0.8,
    )
    axes[3, 1].set_title("Per-Trial Labels Across Kept Trials")
    axes[3, 1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(ROOT / f"processing_{session.eid}.png", dpi=150)
    plt.close(fig)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def main() -> None:
    args = parse_args()
    mode = "sample" if args.sample else "full"
    time_axis = build_time_axis()

    bwm_df, sessions_df = load_release_sessions()
    log(f"Loaded release roster: {len(sessions_df)} sessions from {BWM_RELEASE_CSV.name}")
    sessions_df = sessions_df.reset_index(names="release_index")

    if mode == "sample":
        target_successes = 2
        candidate_rows = sessions_df.itertuples(index=False)
    else:
        target_successes = None
        candidate_rows = sessions_df.itertuples(index=False)

    processed_sessions: list[ProcessedSession] = []
    skipped_sessions: list[dict[str, Any]] = []

    total_start = time.time()
    if mode == "sample":
        for idx, row in enumerate(candidate_rows, start=1):
            row_series = pd.Series(row._asdict())
            log(
                f"[{idx}/{len(sessions_df)}] Processing {row_series['eid']} "
                f"({row_series['subject']} {row_series['date']} #{int(row_series['session_number'])})"
            )
            try:
                session = process_one_session(row_series, time_axis)
                processed_sessions.append(session)
                log(
                    f"  kept {session.kept_trial_count}/{session.raw_trial_count} trials, "
                    f"{len(session.cluster_regions)} neurons, total {session.timing['total_s']:.1f}s"
                )
                if target_successes is not None and len(processed_sessions) >= target_successes:
                    break
            except Exception as exc:  # noqa: BLE001
                skipped_sessions.append(
                    {
                        "eid": row_series["eid"],
                        "subject": row_series["subject"],
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "has_local_modalities": session_has_local_modalities(row_series),
                    }
                )
                log(f"  skipped: {exc}")
    else:
        records = sessions_df.to_dict(orient="records")
        retry_records: list[dict[str, Any]] = []
        log(f"Using {args.session_workers} session workers for full conversion")
        with ProcessPoolExecutor(max_workers=args.session_workers) as executor:
            future_map = {
                executor.submit(process_one_session_worker, row_dict, time_axis): row_dict
                for row_dict in records
            }
            completed = 0
            total = len(future_map)
            for future in as_completed(future_map):
                row_dict = future_map[future]
                completed += 1
                try:
                    session = future.result()
                    processed_sessions.append(session)
                    log(
                        f"[{completed}/{total}] kept {session.eid}: "
                        f"{session.kept_trial_count}/{session.raw_trial_count} trials, "
                        f"{len(session.cluster_regions)} neurons, {session.timing['total_s']:.1f}s"
                    )
                except Exception as exc:  # noqa: BLE001
                    retry_records.append(row_dict)
                    log(
                        f"[{completed}/{total}] worker failed {row_dict['eid']}: {exc} "
                        f"(queued for sequential retry)"
                    )

        if retry_records:
            log(f"Sequential retry pass for {len(retry_records)} session(s)")
            for idx, row_dict in enumerate(retry_records, start=1):
                row_series = pd.Series(row_dict)
                try:
                    session = process_one_session(row_series, time_axis)
                    processed_sessions.append(session)
                    log(
                        f"[retry {idx}/{len(retry_records)}] kept {session.eid}: "
                        f"{session.kept_trial_count}/{session.raw_trial_count} trials, "
                        f"{len(session.cluster_regions)} neurons, {session.timing['total_s']:.1f}s"
                    )
                except Exception as exc:  # noqa: BLE001
                    skipped_sessions.append(
                        {
                            "eid": row_dict["eid"],
                            "subject": row_dict["subject"],
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "has_local_modalities": session_has_local_modalities(pd.Series(row_dict)),
                        }
                    )
                    log(f"[retry {idx}/{len(retry_records)}] skipped {row_dict['eid']}: {exc}")

    if not processed_sessions:
        raise RuntimeError("No sessions were successfully converted")

    processed_sessions.sort(key=lambda session: session.release_index)

    thresholds = compute_thresholds(processed_sessions)
    log(
        "Discretization thresholds: "
        f"wheel={thresholds['wheel_speed']}, "
        f"whisker={thresholds['whisker_motion_energy']}"
    )

    dataset = assemble_dataset(processed_sessions, thresholds)
    dataset["metadata"]["mode"] = mode
    dataset["metadata"]["n_sessions_attempted"] = (
        len(processed_sessions) + len(skipped_sessions)
    )
    dataset["metadata"]["n_sessions_converted"] = len(processed_sessions)
    dataset["metadata"]["skipped_sessions"] = [
        {
            "eid": item["eid"],
            "subject": item["subject"],
            "error": item["error"],
            "has_local_modalities": item["has_local_modalities"],
        }
        for item in skipped_sessions
    ]

    if args.show_processing:
        for session in processed_sessions[:2]:
            plot_processing(session, thresholds)

    out_path = Path(args.outpicklefile)
    with out_path.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - total_start
    log(
        f"Saved {out_path} with {len(processed_sessions)} sessions in {elapsed:.1f}s "
        f"({format_size(out_path.stat().st_size)})"
    )


if __name__ == "__main__":
    main()
