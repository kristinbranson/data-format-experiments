#!/usr/bin/env python3
"""Convert the Zhong et al. dataset into the decoder format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--show-processing]
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEC_PER_DAY = 24.0 * 3600.0
MS_PER_DAY = SEC_PER_DAY * 1000.0
MIN_TRIAL_TIMEPOINTS = 5
MAX_RETAINED_TRIAL_DURATION_S = 60.0
MAX_RETAINED_INTERFRAME_GAP_S = 10.0
INPUT_NAMES = [
    "time_to_sound_cue_s",
    "training_day",
    "time_since_trial_start_s",
    "reward_available",
]
OUTPUT_NAMES = [
    "visual_stimulus",
    "licking",
    "position_bin",
    "running_speed_bin",
]
POSITION_OUTPUT_VALUES = ["0-1m", "1-2m", "2-3m", "3-4m"]
RUNNING_SPEED_OUTPUT_VALUES = ["q1", "q2", "q3", "q4"]
BRAIN_REGION_NAMES = ["V1", "mHV", "lHV", "aHV", "other"]


@dataclass
class SessionSpec:
    base: str
    key: str
    record: dict
    subject: str
    date: datetime
    blk: str
    source_files: set[str] = field(default_factory=set)
    all_keys: set[str] = field(default_factory=set)
    experiment_types: set[str] = field(default_factory=set)
    training_day: float = 0.0
    estimated_nfr: int = 0
    trial_frame_indices: list[np.ndarray] = field(default_factory=list)
    kept_trial_mask: np.ndarray | None = None
    median_frame_dt_ms: float = np.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset to decoder format.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    parser.add_argument("--full", action="store_true", help="Process all sessions (default).")
    parser.add_argument("--sample", action="store_true", help="Process only 2 sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save diagnostic plots for up to 2 processed sessions.",
    )
    return parser.parse_args()


def parse_base(base: str) -> tuple[str, datetime, str]:
    parts = base.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected recording base format: {base}")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    blk = parts[4]
    return subject, date, blk


def session_key_score(key: str, record: dict) -> tuple[int, int, int, str]:
    stim_id = np.asarray(record.get("stim_id", []), dtype=float)
    non_nan_stim = int(np.isfinite(stim_id).sum()) if stim_id.size else 0
    unique_walls = int(len(np.unique(record.get("WallName", []))))
    swap_penalty = 1 if "swap" in key else 0
    return (swap_penalty, -non_nan_stim, -unique_walls, key)


def load_experiment_type_map(root: Path) -> dict[str, set[str]]:
    exp_info = np.load(root / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
    base_to_experiments: dict[str, set[str]] = {}
    for exp_type, records in exp_info.items():
        for rec in records:
            base = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
            base_to_experiments.setdefault(base, set()).add(exp_type)
    return base_to_experiments


def select_representative_sessions(root: Path) -> list[SessionSpec]:
    beh_dir = root / "data" / "beh"
    base_to_experiments = load_experiment_type_map(root)
    selected: dict[str, SessionSpec] = {}

    for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
        exp_type = beh_path.stem.replace("Beh_", "")
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for key, record in beh_dict.items():
            base = "_".join(key.split("_")[:5])
            subject, date, blk = parse_base(base)
            if base not in selected:
                selected[base] = SessionSpec(
                    base=base,
                    key=key,
                    record=record,
                    subject=subject,
                    date=date,
                    blk=blk,
                )
            else:
                current = selected[base]
                if session_key_score(key, record) < session_key_score(current.key, current.record):
                    current.key = key
                    current.record = record
            selected[base].source_files.add(beh_path.name)
            selected[base].all_keys.add(key)
            selected[base].experiment_types.add(exp_type)

    for base, spec in selected.items():
        spec.experiment_types.update(base_to_experiments.get(base, set()))

    specs = sorted(selected.values(), key=lambda s: (s.subject, s.date, int(s.blk)))
    first_date_by_subject: dict[str, datetime] = {}
    for spec in specs:
        first_date_by_subject.setdefault(spec.subject, spec.date)
        spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
    return specs


def choose_sample_specs(specs: list[SessionSpec], nsample: int = 2) -> list[SessionSpec]:
    rewarded = [spec for spec in specs if int(np.sum(spec.record["isRew"])) > 0]
    unrewarded = [spec for spec in specs if int(np.sum(spec.record["isRew"])) == 0]
    chosen: list[SessionSpec] = []
    if rewarded:
        chosen.append(rewarded[0])
    if unrewarded:
        for spec in unrewarded:
            if spec.base not in {item.base for item in chosen}:
                chosen.append(spec)
                break
    if len(chosen) < nsample:
        for spec in specs:
            if spec.base not in {item.base for item in chosen}:
                chosen.append(spec)
            if len(chosen) >= nsample:
                break
    return chosen[:nsample]


def estimate_behavior_frame_count(record: dict) -> int:
    frame_keys = ["ft", "ft_trInd", "ft_move", "ft_CorrSpc", "ft_Pos", "ft_RunSpeed"]
    return min(int(np.asarray(record[key]).shape[0]) for key in frame_keys)


def build_trial_frame_indices(record: dict, nfr: int) -> list[np.ndarray]:
    ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
    valid = np.isfinite(ft_tr)
    valid_idx = np.flatnonzero(valid)
    trial_ids = ft_tr[valid].astype(int)
    keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
    keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0

    ntrials = int(record["ntrials"])
    frame_indices: list[np.ndarray] = []
    for trial in range(ntrials):
        trial_frames = valid_idx[keep & (trial_ids == trial)]
        frame_indices.append(trial_frames.astype(np.int32, copy=False))
    return frame_indices


def prepare_session_specs(specs: list[SessionSpec]) -> tuple[np.ndarray, np.ndarray]:
    all_speeds = []
    frame_dt_medians = []
    for spec in specs:
        record = spec.record
        spec.estimated_nfr = estimate_behavior_frame_count(record)
        spec.trial_frame_indices = build_trial_frame_indices(record, spec.estimated_nfr)
        spec.kept_trial_mask = np.array(
            [len(frame_idx) >= MIN_TRIAL_TIMEPOINTS for frame_idx in spec.trial_frame_indices],
            dtype=bool,
        )
        ft = np.asarray(record["ft"][:spec.estimated_nfr], dtype=float)
        dt_ms = np.diff(ft) * MS_PER_DAY
        spec.median_frame_dt_ms = float(np.median(dt_ms))
        frame_dt_medians.append(spec.median_frame_dt_ms)
        run_speed = np.asarray(record["ft_RunSpeed"][:spec.estimated_nfr], dtype=np.float32)
        for trial_idx, (keep_trial, frame_idx) in enumerate(zip(spec.kept_trial_mask, spec.trial_frame_indices)):
            if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
                all_speeds.append(run_speed[frame_idx])

    if not all_speeds:
        raise RuntimeError("No valid trials found after applying the running-corridor mask.")
    speed_values = np.concatenate(all_speeds).astype(np.float32, copy=False)
    speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
    return speed_edges, np.asarray(frame_dt_medians, dtype=np.float32)


def collect_visual_categories(specs: list[SessionSpec]) -> list[str]:
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
    return categories


def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)


def load_region_index(root: Path, base: str, nneurons: int) -> np.ndarray:
    subject, date, _ = parse_base(base)
    ret_path = root / "data" / "retinotopy" / f"{subject}_{date.strftime('%Y_%m_%d')}_trans.npz"
    ret = np.load(ret_path, allow_pickle=True)
    iarea = np.asarray(ret["iarea"], dtype=float)
    if iarea.shape[0] != nneurons:
        raise ValueError(
            f"Retinotopy neuron count mismatch for {base}: iarea={iarea.shape[0]}, spikes={nneurons}"
        )
    region_idx = np.full(nneurons, 4, dtype=np.int16)
    region_idx[iarea == 8] = 0
    region_idx[np.isin(iarea, [0, 1, 2, 9])] = 1
    region_idx[np.isin(iarea, [5, 6])] = 2
    region_idx[np.isin(iarea, [3, 4])] = 3
    return region_idx


def choose_reference_stimuli(record: dict) -> tuple[str, str] | None:
    uniq_walls = np.asarray(record["UniqWalls"]).astype(str)
    stim_id = np.asarray(record.get("stim_id", []), dtype=float)
    preferred_pairs = [(2, 0), (3, 0), (2, 3)]
    for left_id, right_id in preferred_pairs:
        if np.any(stim_id == left_id) and np.any(stim_id == right_id):
            stim_left = str(uniq_walls[np.where(stim_id == left_id)[0][0]])
            stim_right = str(uniq_walls[np.where(stim_id == right_id)[0][0]])
            return stim_left, stim_right
    return None


def dprime(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    u1 = np.nanmean(x1, axis=1)
    u2 = np.nanmean(x2, axis=1)
    s1 = np.nanstd(x1, axis=1)
    s2 = np.nanstd(x2, axis=1)
    denom = s1 + s2
    out = np.zeros_like(u1, dtype=np.float32)
    valid = denom > 0
    out[valid] = 2.0 * (u1[valid] - u2[valid]) / denom[valid]
    return out


def select_decoder_neurons(
    spk: np.ndarray,
    record: dict,
    region_idx: np.ndarray,
) -> np.ndarray:
    """Select a compact, paper-grounded neuron subset for decoding.

    Uses the same 5% positive / 5% negative selectivity logic as the reference
    coding-direction analysis, computed on running frames in the textured corridor.
    """
    nfr = spk.shape[1]
    stim_pair = choose_reference_stimuli(record)
    if stim_pair is None:
        return np.arange(spk.shape[0], dtype=np.int32)

    ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
    valid = np.isfinite(ft_tr)
    corr_mask = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool) & valid
    gray_mask = np.asarray(record["ft_GraySpc"][:nfr], dtype=bool) & valid
    move_mask = np.asarray(record["ft_move"][:nfr], dtype=float) > 0
    corr_mask &= move_mask
    gray_mask &= move_mask

    wall_id = np.asarray(record["ft_WallID"][:nfr]).astype(str)
    stim1, stim2 = stim_pair
    stim1_mask = corr_mask & (wall_id == stim1)
    stim2_mask = corr_mask & (wall_id == stim2)
    if stim1_mask.sum() < 10 or stim2_mask.sum() < 10:
        mapped = region_idx < 4
        return np.flatnonzero(mapped).astype(np.int32)

    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    corr_neu = (
        (np.mean(spk[:, stim1_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
        | (np.mean(spk[:, stim2_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
    )
    selected = np.zeros(spk.shape[0], dtype=bool)
    for area in range(4):
        candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
        if candidates.sum() == 0:
            continue
        if candidates.sum() < 20:
            selected[candidates] = True
            continue
        hi, lo = np.nanpercentile(dp[candidates], [95, 5])
        selected |= candidates & ((dp >= hi) | (dp <= lo))

    if not np.any(selected):
        selected = (region_idx < 4) & corr_neu & np.isfinite(dp)
    if not np.any(selected):
        selected = region_idx < 4
    return np.flatnonzero(selected).astype(np.int32)


def build_lick_frame_lookup(record: dict, nfr: int) -> dict[int, np.ndarray]:
    lick_frames = np.asarray(record["LickFr"], dtype=float)
    lick_trial = np.asarray(record["LickTrind"], dtype=float)
    valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
    lick_frames = lick_frames[valid].astype(int)
    lick_trial = lick_trial[valid].astype(int)
    valid = (lick_frames >= 0) & (lick_frames < nfr)
    lick_frames = lick_frames[valid]
    lick_trial = lick_trial[valid]
    lookup: dict[int, np.ndarray] = {}
    for trial in np.unique(lick_trial):
        lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
    return lookup


def digitize_speed(speed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)


def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)


def session_trial_info(record: dict, trial: int, frame_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
    time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
    return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)


def trial_passes_quality_filters(
    record: dict,
    trial_idx: int,
    frame_idx: np.ndarray,
    nfr: int,
) -> tuple[bool, str | None]:
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    frame_idx = frame_idx[frame_idx < nfr]
    if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
        return False, "too_few_timepoints"

    frame_times = np.asarray(record["ft"][:nfr], dtype=float)[frame_idx]
    retained_duration_s = float((frame_times[-1] - float(record["Trial_start_time"][trial_idx])) * SEC_PER_DAY)
    if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
        return False, "retained_duration_gt_60s"

    if frame_idx.size > 1:
        max_gap_s = float(np.max(np.diff(frame_times)) * SEC_PER_DAY)
        if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
            return False, "interframe_gap_gt_10s"

    return True, None


def make_processing_plot(
    session_base: str,
    trial_idx: int,
    record: dict,
    frame_idx: np.ndarray,
    neural_trial: np.ndarray,
    input_trial: np.ndarray,
    output_trial: np.ndarray,
    speed_edges: np.ndarray,
    visual_categories: list[str],
) -> None:
    pos = np.asarray(record["ft_Pos"], dtype=float)[frame_idx] / 10.0
    time_since = input_trial[2]
    stim_name = visual_categories[int(output_trial[0, 0])]
    lick = output_trial[1].astype(float)
    pos_bin = output_trial[2].astype(float)
    speed_bin = output_trial[3].astype(float)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)

    axes[0, 0].plot(time_since, pos, color="k", lw=1.5, label="position (m)")
    axes[0, 0].scatter(time_since, np.full_like(time_since, float(record["SoundPos"][trial_idx]) / 10.0), s=8, color="purple", label="cue pos")
    axes[0, 0].set_title(f"{session_base} trial {trial_idx}: raw retained frames")
    axes[0, 0].set_xlabel("time since trial start (s)")
    axes[0, 0].set_ylabel("position (m)")
    axes[0, 0].legend(loc="upper left")

    nneu_plot = min(100, neural_trial.shape[0])
    axes[0, 1].imshow(neural_trial[:nneu_plot], aspect="auto", interpolation="nearest", cmap="gray_r")
    axes[0, 1].set_title(f"Neural activity ({nneu_plot} / {neural_trial.shape[0]} neurons)")
    axes[0, 1].set_xlabel("retained frame")
    axes[0, 1].set_ylabel("neuron")

    axes[1, 0].plot(input_trial[2], label="time since start (s)")
    axes[1, 0].plot(input_trial[0], label="time to cue (s)")
    axes[1, 0].plot(input_trial[1], label="training day")
    axes[1, 0].plot(input_trial[3], label="reward available")
    axes[1, 0].set_title("Decoder inputs")
    axes[1, 0].set_xlabel("retained frame")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].step(np.arange(lick.size), lick, where="mid", label="licking")
    axes[1, 1].step(np.arange(pos_bin.size), pos_bin, where="mid", label="position bin")
    axes[1, 1].step(np.arange(speed_bin.size), speed_bin, where="mid", label="speed bin")
    axes[1, 1].set_title(f"Decoder outputs (stimulus={stim_name})")
    axes[1, 1].set_xlabel("retained frame")
    axes[1, 1].legend(loc="upper right", fontsize=8)

    speed = np.asarray(record["ft_RunSpeed"], dtype=float)[frame_idx]
    axes[2, 0].hist(speed, bins=30, color="0.7", edgecolor="0.2")
    for edge in speed_edges:
        axes[2, 0].axvline(float(edge), color="r", ls="--", lw=1)
    axes[2, 0].set_title("Running-speed discretization")
    axes[2, 0].set_xlabel("ft_RunSpeed")
    axes[2, 0].set_ylabel("count")

    text = [
        f"trial frames kept: {frame_idx.size}",
        f"stimulus: {stim_name}",
        f"reward_available: {int(input_trial[3, 0] > 0)}",
        f"cue pos (m): {float(record['SoundPos'][trial_idx]) / 10.0:.3f}",
        f"speed quartiles: {speed_edges.tolist()}",
    ]
    axes[2, 1].axis("off")
    axes[2, 1].text(0.0, 1.0, "\n".join(text), va="top", family="monospace")

    out_name = f"processing_{session_base}.png"
    fig.savefig(out_name, dpi=150)
    plt.close(fig)


def process_sessions(
    root: Path,
    specs: list[SessionSpec],
    visual_categories: list[str],
    speed_edges: np.ndarray,
    frame_dt_medians: np.ndarray,
    show_processing: bool,
) -> tuple[dict, dict]:
    visual_to_idx = {name: idx for idx, name in enumerate(visual_categories)}

    neural_data: list[list[np.ndarray]] = []
    input_data: list[list[np.ndarray]] = []
    output_data: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    subject_names = sorted({spec.subject for spec in specs})
    subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
    subject_idx = []
    removed_sessions = []
    removed_trials = []

    processing_plots_remaining = 2 if show_processing else 0
    total_start = time.perf_counter()

    for session_idx, spec in enumerate(specs, start=1):
        session_start = time.perf_counter()
        print(f"[{session_idx:03d}/{len(specs):03d}] Loading {spec.base} ({spec.key})")

        spk = load_spike_matrix(root, spec.base)
        nneurons, nfr = spk.shape
        region_idx = load_region_index(root, spec.base, nneurons)
        selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
        spk = spk[selected_neurons]
        region_idx = region_idx[selected_neurons]
        nneurons = spk.shape[0]
        lick_lookup = build_lick_frame_lookup(spec.record, nfr)

        session_neural: list[np.ndarray] = []
        session_input: list[np.ndarray] = []
        session_output: list[np.ndarray] = []
        kept_trial_indices: list[int] = []

        for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
            frame_idx = frame_idx[frame_idx < nfr]
            keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
            if not keep_trial:
                if frame_idx.size > 0:
                    removed_trials.append((spec.base, int(trial_idx), int(frame_idx.size), remove_reason))
                continue

            neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
            time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
            reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
            training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
            input_trial = np.vstack(
                [time_to_cue, training_day, time_since_start, reward_available]
            ).astype(np.float32, copy=False)

            lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
            licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
            position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
            speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
            stim_idx = np.full(
                frame_idx.size,
                visual_to_idx[str(spec.record["WallName"][trial_idx])],
                dtype=np.int16,
            )
            output_trial = np.vstack([stim_idx, licking, position_bin, speed_bin]).astype(np.int16, copy=False)

            session_neural.append(neural_trial)
            session_input.append(input_trial)
            session_output.append(output_trial)
            kept_trial_indices.append(trial_idx)

        if len(session_neural) < 2:
            removed_sessions.append((spec.base, len(session_neural), "fewer_than_two_valid_trials"))
            del spk
            continue

        neural_data.append(session_neural)
        input_data.append(session_input)
        output_data.append(session_output)
        brain_region_idx.append(region_idx)
        subject_idx.append(subject_to_idx[spec.subject])

        if processing_plots_remaining > 0:
            plot_trial_idx = len(session_neural) // 2
            raw_trial_idx = kept_trial_indices[plot_trial_idx]
            make_processing_plot(
                session_base=spec.base,
                trial_idx=raw_trial_idx,
                record=spec.record,
                frame_idx=spec.trial_frame_indices[raw_trial_idx][spec.trial_frame_indices[raw_trial_idx] < nfr],
                neural_trial=session_neural[plot_trial_idx],
                input_trial=session_input[plot_trial_idx],
                output_trial=session_output[plot_trial_idx],
                speed_edges=speed_edges,
                visual_categories=visual_categories,
            )
            processing_plots_remaining -= 1

        elapsed = time.perf_counter() - session_start
        print(
            f"    kept {len(session_neural)} trials, {nneurons} neurons, "
            f"session time {elapsed:.1f}s"
        )
        del spk

    total_elapsed = time.perf_counter() - total_start
    kept_sessions = len(neural_data)
    kept_trials = sum(len(session_trials) for session_trials in neural_data)
    print(
        f"Processed {kept_sessions} sessions / {len(specs)} requested, "
        f"{kept_trials} kept trials in {total_elapsed:.1f}s"
    )

    metadata = {
        "task_description": (
            "Head-fixed mouse virtual-reality corridor imaging. Neural inputs are deconvolved "
            "calcium traces on retained running frames inside the 0-4 m corridor; outputs decode "
            "visual stimulus category, licking, 1 m position bin, and running-speed quartile."
        ),
        "time_bin_size": float(np.median(frame_dt_medians)),
        "temporal_alignment_event": "corridor entry / trial start",
        "off_start": 0.0,
        "off_end": None,
        "frame_bin_source": "native imaging frame timestamps; no temporal resampling",
        "frame_mask": "ft_CorrSpc & (ft_move > 0)",
        "min_trial_timepoints": MIN_TRIAL_TIMEPOINTS,
        "max_retained_trial_duration_s": MAX_RETAINED_TRIAL_DURATION_S,
        "max_retained_interframe_gap_s": MAX_RETAINED_INTERFRAME_GAP_S,
        "training_day_definition": "elapsed days since the subject's first retained imaging session",
        "position_units": "decimeters in raw data; converted position bins cover 0-4 m corridor",
        "running_speed_bin_edges": [float(x) for x in speed_edges],
        "visual_stimulus_values": visual_categories,
        "source_recording_bases": [spec.base for spec in specs],
        "source_behavior_keys": [spec.key for spec in specs],
        "source_experiment_types": {
            spec.base: sorted(spec.experiment_types) for spec in specs
        },
        "removed_sessions": removed_sessions,
        "removed_trials": removed_trials[:1000],
        "removed_trials_truncated": len(removed_trials) > 1000,
    }

    data = {
        "neural": neural_data,
        "input": input_data,
        "output": output_data,
        "subjects": subject_names,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": BRAIN_REGION_NAMES,
        "brain_region_idx": brain_region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": [
            visual_categories,
            ["no_lick", "lick"],
            POSITION_OUTPUT_VALUES,
            RUNNING_SPEED_OUTPUT_VALUES,
        ],
        "metadata": metadata,
    }

    summary = {
        "requested_sessions": len(specs),
        "kept_sessions": len(neural_data),
        "kept_trials": sum(len(session_trials) for session_trials in neural_data),
        "removed_sessions": removed_sessions,
        "removed_trial_count": len(removed_trials),
    }
    return data, summary


def write_pickle(path: Path, data: dict) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    specs = select_representative_sessions(root)
    mode = "sample" if args.sample else "full"
    if args.sample:
        specs = choose_sample_specs(specs, nsample=2)

    print(f"Selected {len(specs)} sessions for mode={mode}")
    speed_edges, frame_dt_medians = prepare_session_specs(specs)
    visual_categories = collect_visual_categories(specs)
    print(f"Visual categories ({len(visual_categories)}): {visual_categories}")
    print(f"Running-speed quartile edges: {speed_edges.tolist()}")
    print(
        f"Median frame dt across selected sessions: "
        f"{float(np.min(frame_dt_medians)):.3f} .. "
        f"{float(np.median(frame_dt_medians)):.3f} .. "
        f"{float(np.max(frame_dt_medians)):.3f} ms"
    )

    data, summary = process_sessions(
        root=root,
        specs=specs,
        visual_categories=visual_categories,
        speed_edges=speed_edges,
        frame_dt_medians=frame_dt_medians,
        show_processing=args.show_processing,
    )

    out_path = Path(args.outpicklefile)
    write_start = time.perf_counter()
    write_pickle(out_path, data)
    write_elapsed = time.perf_counter() - write_start
    size_gb = out_path.stat().st_size / (1024 ** 3)

    print(
        f"Wrote {out_path} ({size_gb:.2f} GiB) in {write_elapsed:.1f}s. "
        f"Summary: {summary}"
    )


if __name__ == "__main__":
    main()
