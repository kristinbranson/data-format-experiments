#!/usr/bin/env python3
"""Convert Zhong et al. 2025 imaging data into decoder-ready trial tensors."""

from __future__ import annotations

import argparse
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BEH_DIR = "/app/data/beh"
SPK_DIR = "/app/data/spk"
RET_DIR = "/app/data/retinotopy"
EXP_INFO_PATH = "/app/data/beh/Imaging_Exp_info.npy"

SECONDS_PER_DAY = 24.0 * 3600.0
REGION_NAMES = ["V1", "mHV", "lHV", "aHV"]
N_PER_REGION = 64


@dataclass(frozen=True)
class SessionView:
    exp_type: str
    key: str
    triplet: tuple[str, str, str]
    has_stimtype: bool
    beh_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset to decoder format.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Plot processing diagnostics for up to 2 sessions.",
    )
    return parser.parse_args()


def load_exp_info() -> dict[str, list[dict[str, Any]]]:
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()


def parse_date(datexp: str) -> date:
    year, month, day = map(int, datexp.split("_"))
    return date(year, month, day)


def behavior_view_priority(beh: dict[str, Any], has_stimtype: bool) -> tuple[int, int, int]:
    stim_id = np.asarray(beh["stim_id"], dtype=float)
    nfinite = int(np.isfinite(stim_id).sum())
    n_walls = int(len(beh["UniqWalls"]))
    return (nfinite, n_walls, 0 if has_stimtype else 1)


def collect_session_views(exp_info: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str, str], list[SessionView]]:
    views: dict[tuple[str, str, str], list[SessionView]] = defaultdict(list)
    for exp_type, records in exp_info.items():
        beh_path = os.path.join(BEH_DIR, f"Beh_{exp_type}.npy")
        beh_all = np.load(beh_path, allow_pickle=True).item()
        for rec in records:
            triplet = (rec["mname"], rec["datexp"], rec["blk"])
            key = "_".join(triplet)
            has_stimtype = "stimtype" in rec
            if has_stimtype:
                key = f"{key}_{rec['stimtype']}"
            if key not in beh_all:
                raise KeyError(f"Missing behavior key {key} in {beh_path}")
            views[triplet].append(
                SessionView(
                    exp_type=exp_type,
                    key=key,
                    triplet=triplet,
                    has_stimtype=has_stimtype,
                    beh_path=beh_path,
                )
            )
    return views


def choose_canonical_view(
    views: dict[tuple[str, str, str], list[SessionView]]
) -> list[tuple[tuple[str, str, str], SessionView, list[str]]]:
    out = []
    for triplet in sorted(views.keys(), key=lambda x: (x[0], parse_date(x[1]), int(x[2]))):
        candidates = []
        exp_types = []
        for view in views[triplet]:
            beh = np.load(view.beh_path, allow_pickle=True).item()[view.key]
            candidates.append((behavior_view_priority(beh, view.has_stimtype), view))
            exp_types.append(view.exp_type)
        candidates.sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[1].exp_type, x[1].key), reverse=True)
        out.append((triplet, candidates[0][1], sorted(set(exp_types))))
    return out


def subject_day_map(
    sessions: list[tuple[tuple[str, str, str], SessionView, list[str]]]
) -> dict[tuple[str, str, str], float]:
    per_subject: dict[str, list[tuple[date, int, tuple[str, str, str]]]] = defaultdict(list)
    for triplet, _view, _exp_types in sessions:
        mouse, datexp, blk = triplet
        per_subject[mouse].append((parse_date(datexp), int(blk), triplet))
    out: dict[tuple[str, str, str], float] = {}
    for mouse, entries in per_subject.items():
        entries.sort()
        first_date = entries[0][0]
        for dt, blk, triplet in entries:
            out[triplet] = float((dt - first_date).days + 0.01 * (blk - 1))
    return out


def load_behavior(view: SessionView) -> dict[str, Any]:
    return np.load(view.beh_path, allow_pickle=True).item()[view.key]


def load_spike_planes(triplet: tuple[str, str, str]) -> list[np.ndarray]:
    mouse, datexp, blk = triplet
    path = os.path.join(SPK_DIR, f"{mouse}_{datexp}_{blk}_neural_data.npy")
    obj = np.load(path, allow_pickle=True).item()
    return [np.asarray(x) for x in obj["spks"]]


def load_iarea(triplet: tuple[str, str, str]) -> np.ndarray:
    mouse, datexp, _blk = triplet
    path = os.path.join(RET_DIR, f"{mouse}_{datexp}_trans.npz")
    return np.load(path, allow_pickle=True)["iarea"]


def area_masks(iarea: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "V1": iarea == 8,
        "mHV": (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9),
        "lHV": (iarea == 5) | (iarea == 6),
        "aHV": (iarea == 3) | (iarea == 4),
    }


def trim_behavior_framewise(beh: dict[str, Any], nfr: int) -> dict[str, Any]:
    trimmed = dict(beh)
    for key, value in beh.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] >= nfr:
            if key.startswith("ft_") or key in {"ft", "AftCueFr", "BefCueFr"}:
                trimmed[key] = value[:nfr]
    return trimmed


def get_primary_stimulus_names(beh: dict[str, Any]) -> tuple[str, str]:
    stim_id = np.asarray(beh["stim_id"], dtype=float)
    uniq = np.asarray(beh["UniqWalls"])
    if np.any(stim_id == 2) and np.any(stim_id == 0):
        stim_a = str(uniq[np.where(stim_id == 2)[0][0]])
        stim_b = str(uniq[np.where(stim_id == 0)[0][0]])
        return stim_a, stim_b
    finite = np.where(np.isfinite(stim_id))[0]
    if len(finite) < 2:
        raise ValueError("Need at least two finite stim_id entries to define a primary stimulus pair.")
    order = np.argsort(stim_id[finite])
    first, second = finite[order[0]], finite[order[1]]
    return str(uniq[first]), str(uniq[second])


def compute_dprime(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    u1 = np.nanmean(x1, axis=1)
    u2 = np.nanmean(x2, axis=1)
    s1 = np.nanstd(x1, axis=1)
    s2 = np.nanstd(x2, axis=1)
    denom = s1 + s2
    out = np.zeros_like(u1, dtype=np.float32)
    valid = denom > 0
    out[valid] = (2.0 * (u1[valid] - u2[valid]) / denom[valid]).astype(np.float32)
    return out


def select_neurons(
    planes: list[np.ndarray], beh: dict[str, Any], iarea: np.ndarray
) -> tuple[list[tuple[int, np.ndarray, int]], np.ndarray, dict[str, int], dict[str, Any]]:
    nfr = planes[0].shape[1]
    ft_wall = np.asarray(beh["ft_WallID"][:nfr])
    ft_move = np.asarray(beh["ft_move"][:nfr], dtype=float) > 0
    ft_corr = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    ft_gray = np.asarray(beh["ft_GraySpc"][:nfr], dtype=bool)
    stim_a, stim_b = get_primary_stimulus_names(beh)
    stim_a_fr = (ft_wall == stim_a) & ft_corr & ft_move
    stim_b_fr = (ft_wall == stim_b) & ft_corr & ft_move
    gray_fr = ft_gray & ft_move
    if stim_a_fr.sum() == 0 or stim_b_fr.sum() == 0 or gray_fr.sum() == 0:
        raise ValueError("Reference stimulus or gray-space frames are empty for neuron selection.")

    candidate_by_region: dict[str, list[tuple[float, int, int]]] = {name: [] for name in REGION_NAMES}
    total_corr_neu = 0
    total_abs_dp = 0
    total_neurons = 0
    offset = 0
    for plane_idx, plane in enumerate(planes):
        plane_iarea = iarea[offset : offset + plane.shape[0]]
        dp = compute_dprime(plane[:, stim_a_fr], plane[:, stim_b_fr])
        corr_neu = (plane[:, stim_a_fr].mean(axis=1) > plane[:, gray_fr].mean(axis=1)) | (
            plane[:, stim_b_fr].mean(axis=1) > plane[:, gray_fr].mean(axis=1)
        )
        masks = area_masks(plane_iarea)
        total_corr_neu += int(corr_neu.sum())
        total_abs_dp += int((np.abs(dp) >= 0.3).sum())
        total_neurons += int(plane.shape[0])
        for region_name in REGION_NAMES:
            region_mask = masks[region_name]
            region_pool = region_mask & corr_neu & np.isfinite(dp)
            strong_pool = region_pool & (np.abs(dp) >= 0.3)
            chosen_pool = strong_pool if strong_pool.any() else region_pool
            local_idx = np.flatnonzero(chosen_pool)
            if local_idx.size == 0:
                continue
            scores = np.abs(dp[local_idx])
            candidate_by_region[region_name].extend(
                (float(score), plane_idx, int(idx)) for score, idx in zip(scores.tolist(), local_idx.tolist())
            )
        offset += plane.shape[0]

    selected_blocks: list[tuple[int, np.ndarray, int]] = []
    selected_regions = []
    counts: dict[str, int] = {}
    for region_idx, region_name in enumerate(REGION_NAMES):
        candidates = candidate_by_region[region_name]
        if not candidates:
            counts[region_name] = 0
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[:N_PER_REGION]
        per_plane: dict[int, list[int]] = defaultdict(list)
        for _score, plane_idx, local_idx in chosen:
            per_plane[plane_idx].append(local_idx)
        n_selected_region = 0
        for plane_idx in sorted(per_plane):
            local_idx = np.asarray(per_plane[plane_idx], dtype=np.int64)
            selected_blocks.append((plane_idx, local_idx, region_idx))
            selected_regions.append(np.full(local_idx.size, region_idx, dtype=np.int16))
            n_selected_region += int(local_idx.size)
        counts[region_name] = n_selected_region
    if not selected_blocks:
        raise ValueError("No neurons selected in any target visual region.")
    region_idx = np.concatenate(selected_regions).astype(np.int16)
    meta = {
        "stimulus_pair_for_selection": [stim_a, stim_b],
        "corridor_responsive_fraction": float(total_corr_neu / max(1, total_neurons)),
        "abs_dprime_ge_0p3_fraction": float(total_abs_dp / max(1, total_neurons)),
        "selected_fraction": float(region_idx.size / max(1, total_neurons)),
    }
    return selected_blocks, region_idx, counts, meta


def family_from_wall_name(name: str) -> str:
    lowered = name.lower()
    for prefix in ("circle", "leaf", "rock", "wood", "brick"):
        if lowered.startswith(prefix):
            return prefix
    match = re.match(r"([a-zA-Z]+)", lowered)
    if match:
        return match.group(1)
    return lowered


def build_lick_frame_vector(beh: dict[str, Any], nfr: int) -> np.ndarray:
    lick_vec = np.zeros(nfr, dtype=np.int8)
    lick_fr = np.asarray(beh["LickFr"], dtype=float)
    lick_idx = np.asarray(np.floor(lick_fr), dtype=int)
    valid = (lick_idx >= 0) & (lick_idx < nfr)
    lick_vec[lick_idx[valid]] = 1
    return lick_vec


def compute_frame_timing(
    sessions: list[tuple[tuple[str, str, str], SessionView, list[str]]]
) -> tuple[float, list[float]]:
    dts = []
    for triplet, view, _exp_types in sessions:
        beh = load_behavior(view)
        ft = np.asarray(beh["ft"], dtype=float)
        dts.append(float(np.nanmedian(np.diff(ft)) * SECONDS_PER_DAY * 1000.0))
    return float(np.median(dts)), dts


def choose_sample_sessions(
    sessions: list[tuple[tuple[str, str, str], SessionView, list[str]]], n_sample: int = 2
) -> list[tuple[tuple[str, str, str], SessionView, list[str]]]:
    scored = []
    for triplet, view, exp_types in sessions:
        beh = load_behavior(view)
        reward_frac = float(np.mean(np.asarray(beh["isRew"], dtype=float)))
        lick_count = int(len(beh["LickFr"]))
        n_families = len({family_from_wall_name(str(x)) for x in beh["WallName"]})
        is_task = any(exp_type.startswith("sup_") for exp_type in exp_types)
        has_mixed_reward = reward_frac > 0.0 and reward_frac < 1.0
        score = (
            1 if is_task else 0,
            1 if has_mixed_reward else 0,
            1 if lick_count > 0 else 0,
            n_families,
            lick_count,
            reward_frac,
        )
        scored.append((score, triplet, view, exp_types))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(triplet, view, exp_types) for _score, triplet, view, exp_types in scored[:n_sample]]


def position_to_bin(pos: np.ndarray) -> np.ndarray:
    bins = np.floor(np.asarray(pos, dtype=float) / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)


def assign_rank_speed_bins(all_speed: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    all_speed = np.asarray(all_speed, dtype=np.float32)
    if all_speed.ndim != 1 or all_speed.size == 0:
        raise ValueError("Need a non-empty 1D speed vector to assign quartile bins.")
    order = np.argsort(all_speed, kind="mergesort")
    speed_bins = np.empty(all_speed.size, dtype=np.int16)
    for bin_idx, idx_chunk in enumerate(np.array_split(order, 4)):
        speed_bins[idx_chunk] = bin_idx
    edges = np.quantile(all_speed, [0.25, 0.5, 0.75]).astype(np.float32)
    value_ranges = []
    for bin_idx in range(4):
        vals = all_speed[speed_bins == bin_idx]
        value_ranges.append([float(vals.min()), float(vals.max())])
    return speed_bins, edges, value_ranges


def make_processing_plot(
    session_id: str,
    neural_trials: list[np.ndarray],
    input_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    region_counts: dict[str, int],
    selection_meta: dict[str, Any],
    speed_binning_summary: dict[str, Any],
    day_value: float,
) -> None:
    if not neural_trials:
        return
    sample_trial = min(2, len(neural_trials) - 1)
    neural = neural_trials[sample_trial]
    inputs = input_trials[sample_trial]
    outputs = output_trials[sample_trial]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].bar(REGION_NAMES, [region_counts.get(k, 0) for k in REGION_NAMES], color="tab:blue")
    axes[0].set_title("Selected Neurons per Region")
    axes[0].set_ylabel("Count")

    axes[1].plot(inputs[2], label="time_since_start_s")
    axes[1].plot(inputs[0], label="time_to_sound_cue_s")
    axes[1].axhline(0, color="k", linewidth=0.8, alpha=0.5)
    axes[1].set_title(f"Timing Inputs (day={day_value:.2f})")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].imshow(neural[: min(32, neural.shape[0])], aspect="auto", cmap="magma")
    axes[2].set_title("Sample Neural Trial (first 32 selected neurons)")
    axes[2].set_xlabel("Frame in corridor")
    axes[2].set_ylabel("Neuron")

    axes[3].plot(outputs[1], label="lick")
    axes[3].plot(outputs[2], label="position_bin")
    axes[3].plot(outputs[3], label="speed_bin")
    axes[3].set_title(
        "Outputs / Speed Quartiles\n"
        f"pair={selection_meta['stimulus_pair_for_selection']}, "
        f"ranges={np.round(np.asarray(speed_binning_summary['value_ranges']), 2).tolist()}"
    )
    axes[3].legend(loc="best", fontsize=8)

    fig.suptitle(f"Processing summary: {session_id}")
    fig.tight_layout()
    fig.savefig(f"processing_{session_id}.png", dpi=150)
    plt.close(fig)


def convert_dataset(
    sessions: list[tuple[tuple[str, str, str], SessionView, list[str]]],
    day_map: dict[tuple[str, str, str], float],
    time_bin_ms: float,
    show_processing: bool,
) -> dict[str, Any]:
    subjects = sorted({triplet[0] for triplet, _view, _exp_types in sessions})
    subject_lookup = {name: idx for idx, name in enumerate(subjects)}

    family_values = ["circle", "leaf", "rock", "wood", "brick"]
    family_values = [x for x in family_values if x != "brick"]
    family_lookup = {name: idx for idx, name in enumerate(family_values)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": list(REGION_NAMES),
        "brain_region_idx": [],
        "input_names": [
            "time_to_sound_cue_s",
            "day_of_training",
            "time_since_trial_start_s",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus_category",
            "licking",
            "position_bin",
            "running_speed_bin",
        ],
        "output_values": [
            family_values,
            ["no_lick", "lick"],
            ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"],
            ["q1", "q2", "q3", "q4"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice traversing 4 m visual texture corridors with cue-linked reward availability; "
                "decoder predicts stimulus family, licking, position bin, and running-speed quartile."
            ),
            "time_bin_size": float(time_bin_ms),
            "temporal_alignment_event": "corridor entry (trial start)",
            "off_start": 0.0,
            "off_end": None,
            "corridor_only": True,
            "corridor_length_m": 4.0,
            "gray_space_excluded_m": 2.0,
            "speed_binning_method": "global_rank_quartiles_over_all_converted_corridor_frames",
            "neuron_selection": {
                "method": "top_abs_dprime_per_region",
                "primary_pair_stim_ids": [2, 0],
                "dprime_threshold": 0.3,
                "corridor_responsive_required": True,
                "regions": REGION_NAMES,
                "max_neurons_per_region": N_PER_REGION,
            },
            "session_info": [],
        },
    }

    total_kept_trials = 0
    total_raw_trials = 0
    session_plot_info: list[dict[str, Any]] = []
    speed_value_trials_per_session: list[list[np.ndarray]] = []
    for session_idx, (triplet, view, exp_types) in enumerate(sessions):
        session_start = time.time()
        mouse, datexp, blk = triplet
        session_id = "_".join(triplet)
        beh = load_behavior(view)
        planes = load_spike_planes(triplet)
        iarea = load_iarea(triplet)
        nfr = planes[0].shape[1]
        total_neurons_raw = sum(plane.shape[0] for plane in planes)
        if total_neurons_raw != iarea.shape[0]:
            raise ValueError(f"Neuron count mismatch for {session_id}: spk={total_neurons_raw}, iarea={iarea.shape[0]}")
        beh = trim_behavior_framewise(beh, nfr)
        selected_blocks, region_idx, region_counts, selection_meta = select_neurons(planes, beh, iarea)
        selected_plane_arrays = [
            np.asarray(planes[plane_idx][local_idx], dtype=np.float32)
            for plane_idx, local_idx, _region_id in selected_blocks
        ]
        del planes

        ft = np.asarray(beh["ft"], dtype=float)[:nfr]
        ft_pos = np.asarray(beh["ft_Pos"], dtype=float)[:nfr]
        ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)[:nfr]
        ft_corr = np.asarray(beh["ft_CorrSpc"], dtype=bool)[:nfr]
        frame_trial_raw = np.asarray(beh["ft_trInd"], dtype=float)[:nfr]
        frame_trial = np.full(frame_trial_raw.shape, -1, dtype=int)
        valid_frame_trial = np.isfinite(frame_trial_raw)
        frame_trial[valid_frame_trial] = np.rint(frame_trial_raw[valid_frame_trial]).astype(int)
        lick_vec = build_lick_frame_vector(beh, nfr)

        ntrials = int(beh["ntrials"])
        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        speed_value_trials: list[np.ndarray] = []
        skipped_trials = 0
        day_value = day_map[triplet]

        for trial_idx in range(ntrials):
            mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
            if mask.sum() < 2:
                skipped_trials += 1
                continue

            wall_name = str(beh["WallName"][trial_idx])
            family = family_from_wall_name(wall_name)
            if family not in family_lookup:
                skipped_trials += 1
                continue

            trial_times = ft[mask]
            cue_time = float(beh["SoundTime"][trial_idx])
            start_time = float(beh["Trial_start_time"][trial_idx])
            if not np.isfinite(cue_time) or not np.isfinite(start_time):
                skipped_trials += 1
                continue

            neural_trial = np.concatenate(
                [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
                axis=0,
            ).astype(np.float16, copy=False)
            input_trial = np.vstack(
                [
                    (cue_time - trial_times) * SECONDS_PER_DAY,
                    np.full(mask.sum(), day_value, dtype=np.float32),
                    (trial_times - start_time) * SECONDS_PER_DAY,
                    np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32),
                ]
            ).astype(np.float32)

            lick_out = lick_vec[mask].astype(np.int16)
            pos_out = position_to_bin(ft_pos[mask])
            speed_values = np.asarray(ft_speed[mask], dtype=np.float32)
            stim_out = np.full(mask.sum(), family_lookup[family], dtype=np.int16)
            output_trial = np.vstack([stim_out, lick_out, pos_out]).astype(np.int16)

            if not (
                np.all(np.isfinite(neural_trial))
                and np.all(np.isfinite(input_trial))
                and np.all(np.isfinite(output_trial))
                and np.all(np.isfinite(speed_values))
            ):
                skipped_trials += 1
                continue

            neural_trials.append(neural_trial)
            input_trials.append(input_trial)
            output_trials.append(output_trial)
            speed_value_trials.append(speed_values)

        if len(neural_trials) < 2:
            raise ValueError(f"Session {session_id} retained fewer than 2 usable trials.")

        total_kept_trials += len(neural_trials)
        total_raw_trials += ntrials
        data["neural"].append(neural_trials)
        data["input"].append(input_trials)
        data["output"].append(output_trials)
        speed_value_trials_per_session.append(speed_value_trials)
        data["subject_idx"].append(subject_lookup[mouse])
        data["brain_region_idx"].append(region_idx.astype(np.int16))
        data["metadata"]["session_info"].append(
            {
                "session_id": session_id,
                "subject": mouse,
                "datexp": datexp,
                "blk": blk,
                "canonical_behavior_key": view.key,
                "source_experiment_types": exp_types,
                "day_of_training": day_value,
                "raw_ntrials": ntrials,
                "kept_ntrials": len(neural_trials),
                "skipped_ntrials": skipped_trials,
                "selected_neurons": int(region_idx.size),
                "selected_neurons_per_region": region_counts,
                "selection_meta": selection_meta,
            }
        )
        session_plot_info.append(
            {
                "session_id": session_id,
                "region_counts": region_counts,
                "selection_meta": selection_meta,
                "day_value": day_value,
            }
        )

        elapsed = time.time() - session_start
        print(
            f"[{session_idx + 1:03d}/{len(sessions):03d}] {session_id}: "
            f"selected {region_idx.size} neurons, kept {len(neural_trials)}/{ntrials} trials, "
            f"{elapsed:.2f}s",
            flush=True,
        )

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int16)
    all_speed_values = np.concatenate(
        [trial_speed for session_speed in speed_value_trials_per_session for trial_speed in session_speed]
    )
    all_speed_bins, speed_edges, speed_value_ranges = assign_rank_speed_bins(all_speed_values)
    cursor = 0
    for session_idx, session_speed in enumerate(speed_value_trials_per_session):
        for trial_idx, trial_speed in enumerate(session_speed):
            n_time = int(trial_speed.size)
            trial_speed_bins = all_speed_bins[cursor : cursor + n_time]
            cursor += n_time
            data["output"][session_idx][trial_idx] = np.vstack(
                [data["output"][session_idx][trial_idx], trial_speed_bins[np.newaxis, :]]
            ).astype(np.int16)
    if cursor != int(all_speed_bins.size):
        raise ValueError("Global speed-bin assignment did not consume all frames.")

    data["metadata"]["speed_bin_edges"] = speed_edges.astype(float).tolist()
    data["metadata"]["speed_bin_value_ranges"] = speed_value_ranges
    data["metadata"]["speed_bin_counts"] = [
        int(np.sum(all_speed_bins == bin_idx)) for bin_idx in range(4)
    ]

    if show_processing:
        speed_binning_summary = {
            "edges": speed_edges.astype(float).tolist(),
            "value_ranges": speed_value_ranges,
        }
        for session_idx, plot_info in enumerate(session_plot_info[:2]):
            make_processing_plot(
                session_id=plot_info["session_id"],
                neural_trials=data["neural"][session_idx],
                input_trials=data["input"][session_idx],
                output_trials=data["output"][session_idx],
                region_counts=plot_info["region_counts"],
                selection_meta=plot_info["selection_meta"],
                speed_binning_summary=speed_binning_summary,
                day_value=plot_info["day_value"],
            )
    data["metadata"]["n_sessions"] = len(data["neural"])
    data["metadata"]["n_trials_total_kept"] = int(total_kept_trials)
    data["metadata"]["n_trials_total_raw"] = int(total_raw_trials)
    return data


def main() -> None:
    args = parse_args()
    overall_start = time.time()
    exp_info = load_exp_info()
    session_views = collect_session_views(exp_info)
    canonical_sessions = choose_canonical_view(session_views)
    if args.sample:
        canonical_sessions = choose_sample_sessions(canonical_sessions, n_sample=2)
    time_bin_ms, frame_dt_ms_all = compute_frame_timing(canonical_sessions)
    day_map = subject_day_map(canonical_sessions)

    print(f"Sessions to process: {len(canonical_sessions)}", flush=True)
    print("Running-speed bins: global rank-based quartiles across converted corridor frames", flush=True)
    print(
        f"Median imaging frame interval: {time_bin_ms:.3f} ms "
        f"(session medians range {min(frame_dt_ms_all):.3f}-{max(frame_dt_ms_all):.3f} ms)",
        flush=True,
    )

    data = convert_dataset(
        sessions=canonical_sessions,
        day_map=day_map,
        time_bin_ms=time_bin_ms,
        show_processing=args.show_processing,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.outpicklefile)), exist_ok=True)
    with open(args.outpicklefile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - overall_start
    out_size_mb = os.path.getsize(args.outpicklefile) / (1024.0 * 1024.0)
    print(f"Saved converted dataset to {args.outpicklefile}", flush=True)
    print(f"Output size: {out_size_mb:.2f} MB", flush=True)
    print(f"Total run time: {elapsed:.2f} s", flush=True)


if __name__ == "__main__":
    main()
