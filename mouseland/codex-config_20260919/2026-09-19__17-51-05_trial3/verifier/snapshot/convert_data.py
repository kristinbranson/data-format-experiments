#!/usr/bin/env python3
"""Convert Zhong et al. (2025) imaging data to decoder format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample]
                                              [--show-processing]

The conversion preserves released Suite2p deconvolved activity at its native
3.17 Hz imaging frames.  Following the paper, only moving frames in the 0--4 m
textured corridor are retained.  Trials are aligned to corridor entry through
the frame-aligned behavior streams and exact timestamps.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np


APP_ROOT = Path("/app")
DATA_ROOT = APP_ROOT / "data"
BEH_ROOT = DATA_ROOT / "beh"
SPK_ROOT = DATA_ROOT / "spk"
RET_ROOT = DATA_ROOT / "retinotopy"

IMAGING_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / IMAGING_RATE_HZ
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV"]
INPUT_NAMES = [
    "time_to_sound_cue_s",
    "day_of_training",
    "time_since_trial_start_s",
    "reward_available",
]
OUTPUT_NAMES = [
    "visual_stimulus_category",
    "licking",
    "position_1m_bin",
    "running_speed_quartile",
]
OUTPUT_VALUES = [
    ["circle", "leaf", "rock", "wood"],
    ["not licking", "licking"],
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    ["0-25%", "25-50%", "50-75%", "75-100%"],
]
STIMULUS_PREFIX_TO_CLASS = {"circle": 0, "leaf": 1, "rock": 2, "wood": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process the first two sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


def session_id(db: dict) -> str:
    return f"{db['mname']}_{db['datexp']}_{db['blk']}"


def build_session_records() -> list[dict]:
    """Return one deterministic record per physical neural recording."""
    exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
    records: list[dict] = []
    seen: set[str] = set()
    for group, descriptors in exp_info.items():
        for db0 in descriptors:
            db = dict(db0)
            sid = session_id(db)
            if sid in seen:
                continue
            seen.add(sid)
            behavior_key = sid
            if "stimtype" in db:
                behavior_key = f"{sid}_{db['stimtype']}"
            if "days" in db:
                day_value = float(db["days"])
                day_source = "days"
            elif "sess#" in db:
                day_value = float(db["sess#"])
                day_source = "sess#"
            else:
                raise ValueError(f"{sid}: neither 'days' nor 'sess#' is available")
            records.append(
                {
                    "session_id": sid,
                    "group": group,
                    "behavior_key": behavior_key,
                    "db": db,
                    "day_value": day_value,
                    "day_source": day_source,
                }
            )
    if len(records) != 89:
        raise AssertionError(f"Expected 89 physical sessions, found {len(records)}")
    return records


def records_by_group(records: list[dict]) -> OrderedDict[str, list[dict]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        grouped.setdefault(record["group"], []).append(record)
    return grouped


def get_behavior(behavior_dict: dict, record: dict) -> dict:
    key = record["behavior_key"]
    if key in behavior_dict:
        return behavior_dict[key]
    matches = [k for k in behavior_dict if k.startswith(record["session_id"])]
    if len(matches) != 1:
        raise KeyError(
            f"{record['session_id']}: expected one behavior key, got {matches}"
        )
    record["behavior_key"] = matches[0]
    return behavior_dict[matches[0]]


def validate_behavior_shapes(beh: dict, sid: str) -> None:
    ntrials = int(beh["ntrials"])
    trial_fields = [
        "trInd", "Trial_start_time", "Trial_end_time", "Gray_space_time",
        "SoundPos", "SoundTime", "RewTime", "RewPos", "isRew", "WallName",
        "RewardFr", "StartFr", "GrayFr", "EndFr", "SoundFr",
    ]
    frame_fields = [
        "ft", "ft_trInd", "ft_Pos", "ft_move", "ft_CorrSpc", "ft_RunSpeed",
    ]
    for field in trial_fields:
        if len(beh[field]) != ntrials:
            raise ValueError(
                f"{sid}: {field} length {len(beh[field])} != ntrials {ntrials}"
            )
    nframes = len(beh["ft"])
    for field in frame_fields:
        if len(beh[field]) != nframes:
            raise ValueError(
                f"{sid}: {field} length {len(beh[field])} != ft length {nframes}"
            )
    if tuple(np.asarray(beh["run_pos"]).shape) != (ntrials, 60):
        raise ValueError(f"{sid}: unexpected run_pos shape {np.asarray(beh['run_pos']).shape}")
    if not (
        float(beh["Corridor_Length"]) == 60.0
        and float(beh["Texture_Length"]) == 40.0
        and float(beh["Gray_Space_length"]) == 20.0
    ):
        raise ValueError(f"{sid}: unexpected corridor geometry")


def selected_frames_by_trial(beh: dict, sid: str) -> list[np.ndarray]:
    """Apply the paper's moving, textured-corridor frame mask."""
    trial_id = np.asarray(beh["ft_trInd"])
    valid = (
        np.isfinite(trial_id)
        & np.asarray(beh["ft_CorrSpc"], dtype=bool)
        & (np.asarray(beh["ft_move"]) > 0)
    )
    selected = np.flatnonzero(valid)
    selected_trial = trial_id[selected].astype(np.int64)
    ntrials = int(beh["ntrials"])
    if np.any((selected_trial < 0) | (selected_trial >= ntrials)):
        raise ValueError(f"{sid}: selected frames contain invalid trial IDs")
    frames = [selected[selected_trial == trial] for trial in range(ntrials)]
    lengths = np.asarray([len(x) for x in frames])
    if np.any(lengths < 2):
        bad = np.flatnonzero(lengths < 2).tolist()
        raise ValueError(f"{sid}: trials with fewer than two valid frames: {bad}")
    positions = np.asarray(beh["ft_Pos"])[selected]
    if not np.all(np.isfinite(positions)) or np.any((positions < 0) | (positions >= 40)):
        raise ValueError(f"{sid}: selected positions are outside [0, 40)")
    speed = np.asarray(beh["ft_RunSpeed"])[selected]
    if not np.all(np.isfinite(speed)):
        raise ValueError(f"{sid}: selected running speed contains NaN/Inf")
    return frames


def speed_quartile_prepass(records: list[dict]) -> tuple[np.ndarray, dict]:
    """Validate behavior and get global speed quartile edges."""
    t0 = time.perf_counter()
    speed_chunks: list[np.ndarray] = []
    trial_lengths: list[int] = []
    total_trials = 0
    rewarded_deliveries = 0
    rewarded_availability = 0
    for group, group_records in records_by_group(records).items():
        behavior_dict = np.load(
            BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True
        ).item()
        for record in group_records:
            beh = get_behavior(behavior_dict, record)
            sid = record["session_id"]
            validate_behavior_shapes(beh, sid)
            frames = selected_frames_by_trial(beh, sid)
            for idx in frames:
                speed_chunks.append(np.asarray(beh["ft_RunSpeed"])[idx])
                trial_lengths.append(len(idx))
            total_trials += int(beh["ntrials"])
            is_rewarded = np.asarray(beh["isRew"], dtype=bool)
            rewarded_deliveries += int(is_rewarded.sum())
            reward_walls = np.unique(np.asarray(beh["WallName"])[is_rewarded])
            if len(reward_walls) > 1:
                raise ValueError(f"{sid}: multiple rewarded walls {reward_walls.tolist()}")
            if len(reward_walls) == 1:
                rewarded_availability += int(
                    np.sum(np.asarray(beh["WallName"]) == reward_walls[0])
                )
        del behavior_dict
        gc.collect()
    speeds = np.concatenate(speed_chunks).astype(np.float64, copy=False)
    thresholds = np.quantile(speeds, [0.25, 0.50, 0.75])
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError(f"Non-unique running-speed quartile thresholds: {thresholds}")
    lengths = np.asarray(trial_lengths)
    summary = {
        "n_trials": int(total_trials),
        "n_timepoints": int(len(speeds)),
        "trial_length_min": int(lengths.min()),
        "trial_length_mean": float(lengths.mean()),
        "trial_length_median": float(np.median(lengths)),
        "trial_length_max": int(lengths.max()),
        "speed_min": float(speeds.min()),
        "speed_max": float(speeds.max()),
        "reward_deliveries": int(rewarded_deliveries),
        "reward_availability_trials": int(rewarded_availability),
    }
    print(
        f"Behavior prepass: {total_trials:,} trials, {len(speeds):,} selected "
        f"timepoints, speed edges {thresholds.tolist()} "
        f"({time.perf_counter() - t0:.2f} s)",
        flush=True,
    )
    return thresholds, summary


def raw_area_to_region_idx(iarea: np.ndarray, sid: str) -> tuple[np.ndarray, np.ndarray]:
    """Return named-area keep mask and target indices, matching reference code."""
    iarea = np.asarray(iarea).astype(np.int64, copy=False)
    region = np.full(len(iarea), -1, dtype=np.int16)
    region[iarea == 8] = 0
    region[np.isin(iarea, [0, 1, 2, 9])] = 1
    region[np.isin(iarea, [5, 6])] = 2
    region[np.isin(iarea, [3, 4])] = 3
    keep = region >= 0
    unexpected = np.unique(iarea[(~keep) & (~np.isin(iarea, [-1, 7]))])
    if len(unexpected):
        raise ValueError(f"{sid}: unmapped retinotopy codes {unexpected.tolist()}")
    if not np.any(keep):
        raise ValueError(f"{sid}: no cells in named visual regions")
    return keep, region[keep]


def load_filtered_neural(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one neural session and retain named visual-area cells."""
    sid = record["session_id"]
    db = record["db"]
    ret_path = RET_ROOT / f"{db['mname']}_{db['datexp']}_trans.npz"
    with np.load(ret_path, allow_pickle=True) as ret:
        iarea = np.asarray(ret["iarea"])
    keep, region_idx = raw_area_to_region_idx(iarea, sid)

    spk_path = SPK_ROOT / f"{sid}_neural_data.npy"
    raw = np.load(spk_path, allow_pickle=True).item()
    if set(raw) != {"spks"} or not isinstance(raw["spks"], list):
        raise ValueError(f"{sid}: unexpected neural file structure")
    planes = raw["spks"]
    if not planes:
        raise ValueError(f"{sid}: empty spks list")
    nframes_set = {int(x.shape[1]) for x in planes}
    if len(nframes_set) != 1:
        raise ValueError(f"{sid}: imaging planes have different frame counts")
    nframes = nframes_set.pop()
    raw_neurons = sum(int(x.shape[0]) for x in planes)
    if raw_neurons != len(iarea):
        raise ValueError(
            f"{sid}: spks has {raw_neurons} cells but retinotopy has {len(iarea)}"
        )

    # Fill the retained matrix directly to avoid concatenating every raw cell first.
    activity = np.empty((int(keep.sum()), nframes), dtype=np.float32)
    raw_offset = 0
    kept_offset = 0
    for plane in planes:
        nplane = int(plane.shape[0])
        plane_keep = keep[raw_offset : raw_offset + nplane]
        nkeep = int(plane_keep.sum())
        activity[kept_offset : kept_offset + nkeep] = plane[plane_keep]
        raw_offset += nplane
        kept_offset += nkeep
    del raw, planes
    gc.collect()
    if not np.all(np.isfinite(activity)):
        raise ValueError(f"{sid}: retained neural activity contains NaN/Inf")
    return activity, region_idx, iarea


def stimulus_class(wall_name: str) -> int:
    for prefix, value in STIMULUS_PREFIX_TO_CLASS.items():
        if str(wall_name).startswith(prefix):
            return value
    raise ValueError(f"Unrecognized WallName {wall_name!r}")


def rewarded_wall(beh: dict, sid: str) -> str | None:
    walls = np.asarray(beh["WallName"])
    reward_walls = np.unique(walls[np.asarray(beh["isRew"], dtype=bool)])
    if len(reward_walls) > 1:
        raise ValueError(f"{sid}: multiple rewarded wall names {reward_walls.tolist()}")
    return None if len(reward_walls) == 0 else str(reward_walls[0])


def make_processing_plot(
    record: dict,
    beh: dict,
    frames_by_trial: list[np.ndarray],
    activity: np.ndarray,
    iarea: np.ndarray,
    region_idx: np.ndarray,
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
    speed_edges: np.ndarray,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sid = record["session_id"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))

    # 1: region curation.
    raw_counts = Counter(np.asarray(iarea, dtype=int).tolist())
    axes[0, 0].bar([str(k) for k in sorted(raw_counts)], [raw_counts[k] for k in sorted(raw_counts)])
    axes[0, 0].set_title(f"Raw area codes ({len(iarea):,})\nkept named areas: {len(region_idx):,}")
    axes[0, 0].set_xlabel("raw iarea")
    axes[0, 0].set_ylabel("cells")

    # 2: native mean trace and selected-frame mask for a short interval.
    selected = np.concatenate(frames_by_trial)
    show_n = min(activity.shape[1], max(1000, int(selected[min(len(selected) - 1, 500)] + 1)))
    mean_trace = activity[: min(200, activity.shape[0]), :show_n].mean(axis=0)
    axes[0, 1].plot(mean_trace, lw=0.7, color="0.3", label="native frames")
    selected_short = selected[selected < show_n]
    axes[0, 1].scatter(selected_short, mean_trace[selected_short], s=5, color="tab:red", label="retained")
    axes[0, 1].set_title("Native neural timeline and retained frames")
    axes[0, 1].set_xlabel("imaging frame")
    axes[0, 1].legend(fontsize=8)

    # 3: retained neural activity after alignment.
    trial_show = min(12, len(frames_by_trial))
    cols = np.concatenate(frames_by_trial[:trial_show])
    image = activity[: min(80, activity.shape[0]), cols]
    vmax = float(np.percentile(image, 99)) if image.size else 1.0
    axes[0, 2].imshow(image, aspect="auto", interpolation="none", vmin=0, vmax=max(vmax, 1e-6))
    axes[0, 2].set_title(f"Retained activity, first {trial_show} trials")
    axes[0, 2].set_xlabel("selected timepoints")
    axes[0, 2].set_ylabel("sample neurons")

    # 4: explicit trial alignment and cue/lick locations.
    for trial in range(min(5, len(inputs))):
        t = inputs[trial][2]
        position_m = np.asarray(beh["ft_Pos"])[frames_by_trial[trial]] / 10.0
        line = axes[1, 0].plot(t, position_m, marker=".", ms=3, label=f"trial {trial}")[0]
        cue_t = float((beh["SoundTime"][trial] - beh["Trial_start_time"][trial]) * 86400.0)
        axes[1, 0].axvline(cue_t, color=line.get_color(), alpha=0.25, ls="--")
        lick = outputs[trial][1].astype(bool)
        axes[1, 0].scatter(t[lick], position_m[lick], marker="x", color=line.get_color(), s=25)
    axes[1, 0].set_title("Corridor-entry alignment\ndashed=cue, x=lick")
    axes[1, 0].set_xlabel("time since trial start (s)")
    axes[1, 0].set_ylabel("position (m)")
    axes[1, 0].legend(fontsize=7)

    # 5: speed discretization.
    speeds = np.concatenate([np.asarray(beh["ft_RunSpeed"])[x] for x in frames_by_trial])
    axes[1, 1].hist(speeds, bins=80, color="0.65")
    for edge in speed_edges:
        axes[1, 1].axvline(edge, color="tab:red", ls="--")
    axes[1, 1].set_title("Running-speed quartile thresholds")
    axes[1, 1].set_xlabel("native speed")

    # 6: physical position bins.
    positions = np.concatenate([np.asarray(beh["ft_Pos"])[x] / 10.0 for x in frames_by_trial])
    axes[1, 2].hist(positions, bins=np.linspace(0, 4, 41), color="0.65")
    for edge in [1, 2, 3]:
        axes[1, 2].axvline(edge, color="tab:blue", ls="--")
    axes[1, 2].set_title("Four equal 1-m position bins")
    axes[1, 2].set_xlabel("position (m)")

    # 7: all decoder inputs for an example trial, normalized only for display.
    example = 0
    inp = inputs[example]
    for row, name in enumerate(INPUT_NAMES):
        y = inp[row]
        span = float(y.max() - y.min())
        yn = (y - y.min()) / span if span else np.zeros_like(y)
        axes[2, 0].plot(yn + row, label=name)
    axes[2, 0].set_yticks(np.arange(len(INPUT_NAMES)) + 0.5, INPUT_NAMES, fontsize=8)
    axes[2, 0].set_title("Final decoder inputs (trial 0)")
    axes[2, 0].set_xlabel("selected timepoint")

    # 8: all final categorical outputs for the same trial.
    out = outputs[example]
    for row, name in enumerate(OUTPUT_NAMES):
        denom = max(1, len(OUTPUT_VALUES[row]) - 1)
        axes[2, 1].step(np.arange(out.shape[1]), out[row] / denom + row, where="mid")
    axes[2, 1].set_yticks(np.arange(len(OUTPUT_NAMES)) + 0.5, OUTPUT_NAMES, fontsize=8)
    axes[2, 1].set_title("Final decoder outputs (trial 0)")
    axes[2, 1].set_xlabel("selected timepoint")

    # 9: per-trial stimulus and reward availability.
    stim = np.asarray([x[0, 0] for x in outputs], dtype=int)
    reward = np.asarray([x[3, 0] for x in inputs], dtype=int)
    stim_counts = np.bincount(stim, minlength=4)
    axes[2, 2].bar(np.arange(4) - 0.18, stim_counts, width=0.36, label="all trials")
    axes[2, 2].bar(
        np.arange(4) + 0.18,
        np.bincount(stim[reward == 1], minlength=4),
        width=0.36,
        label="reward available",
    )
    axes[2, 2].set_xticks(np.arange(4), OUTPUT_VALUES[0])
    axes[2, 2].set_title("Trial categories and reward context")
    axes[2, 2].legend(fontsize=8)

    fig.suptitle(
        f"Processing audit: {sid} | day={record['day_value']:g} ({record['day_source']})",
        fontsize=14,
    )
    fig.tight_layout()
    path = APP_ROOT / f"processing_{sid}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def convert(args: argparse.Namespace) -> dict:
    all_records = build_session_records()
    records = all_records[:2] if args.sample else all_records
    mode_name = "sample" if args.sample else "full"
    print(f"Mode: {mode_name}; sessions: {len(records)}", flush=True)

    speed_edges, prepass = speed_quartile_prepass(records)
    subjects = sorted({r["db"]["mname"] for r in records})
    subject_lookup = {name: i for i, name in enumerate(subjects)}

    neural_sessions: list[list[np.ndarray]] = []
    input_sessions: list[list[np.ndarray]] = []
    output_sessions: list[list[np.ndarray]] = []
    brain_region_sessions: list[np.ndarray] = []
    subject_idx: list[int] = []
    session_info: list[dict] = []
    raw_total = 0
    retained_total = 0
    plot_count = 0
    conversion_start = time.perf_counter()

    session_number = 0
    for group, group_records in records_by_group(records).items():
        behavior_dict = np.load(
            BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True
        ).item()
        for record in group_records:
            t0 = time.perf_counter()
            sid = record["session_id"]
            beh = get_behavior(behavior_dict, record)
            frames_by_trial = selected_frames_by_trial(beh, sid)
            activity, region_idx, iarea = load_filtered_neural(record)
            nframes_neural = activity.shape[1]
            max_selected = max(int(x[-1]) for x in frames_by_trial)
            if max_selected >= nframes_neural:
                raise ValueError(
                    f"{sid}: selected frame {max_selected} exceeds neural length {nframes_neural}"
                )

            rw_wall = rewarded_wall(beh, sid)
            walls = np.asarray(beh["WallName"])
            reward_available = (
                np.zeros(len(walls), dtype=np.float32)
                if rw_wall is None
                else (walls == rw_wall).astype(np.float32)
            )

            lick_fr = np.asarray(beh["LickFr"])
            lick_tr = np.asarray(beh["LickTrind"])
            finite_lick = np.isfinite(lick_fr) & np.isfinite(lick_tr)
            lick_fr_int = lick_fr[finite_lick].astype(np.int64)
            lick_tr_int = lick_tr[finite_lick].astype(np.int64)

            trial_neural: list[np.ndarray] = []
            trial_inputs: list[np.ndarray] = []
            trial_outputs: list[np.ndarray] = []
            for trial, frames in enumerate(frames_by_trial):
                # Explicit C-order copy lets the full session matrix be released.
                ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
                ft = np.asarray(beh["ft"])[frames]
                time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
                time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
                T = len(frames)
                inp = np.vstack(
                    [
                        time_to_cue,
                        np.full(T, record["day_value"]),
                        time_since_start,
                        np.full(T, reward_available[trial]),
                    ]
                ).astype(np.float32, copy=False)

                stim = stimulus_class(str(walls[trial]))
                trial_lick_frames = lick_fr_int[lick_tr_int == trial]
                lick = np.isin(frames, trial_lick_frames).astype(np.int16)
                position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
                position = np.clip(position, 0, 3).astype(np.int16)
                speed = np.asarray(beh["ft_RunSpeed"])[frames]
                speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
                out = np.vstack(
                    [
                        np.full(T, stim, dtype=np.int16),
                        lick,
                        position,
                        speed_bin,
                    ]
                )

                if ntrial.shape[1] != inp.shape[1] or inp.shape[1] != out.shape[1]:
                    raise AssertionError(f"{sid} trial {trial}: time dimension mismatch")
                if not np.all(np.isfinite(inp)) or not np.all(np.isfinite(ntrial)):
                    raise ValueError(f"{sid} trial {trial}: NaN/Inf after conversion")
                trial_neural.append(ntrial)
                trial_inputs.append(inp)
                trial_outputs.append(out)

            if args.show_processing and plot_count < 2:
                plot_path = make_processing_plot(
                    record,
                    beh,
                    frames_by_trial,
                    activity,
                    iarea,
                    region_idx,
                    trial_inputs,
                    trial_outputs,
                    speed_edges,
                )
                print(f"Saved {plot_path}", flush=True)
                plot_count += 1

            raw_n = int(len(iarea))
            kept_n = int(len(region_idx))
            raw_total += raw_n
            retained_total += kept_n
            neural_sessions.append(trial_neural)
            input_sessions.append(trial_inputs)
            output_sessions.append(trial_outputs)
            brain_region_sessions.append(region_idx.astype(np.int16, copy=False))
            subject_idx.append(subject_lookup[record["db"]["mname"]])
            lengths = [len(x) for x in frames_by_trial]
            session_info.append(
                {
                    "session_id": sid,
                    "experiment_group": group,
                    "behavior_key": record["behavior_key"],
                    "subject": record["db"]["mname"],
                    "date": record["db"]["datexp"],
                    "block": str(record["db"]["blk"]),
                    "day_of_training": float(record["day_value"]),
                    "day_source_field": record["day_source"],
                    "reward_mode": str(beh["Reward_Mode"]),
                    "rewarded_wall": rw_wall,
                    "n_trials": int(beh["ntrials"]),
                    "n_timepoints": int(sum(lengths)),
                    "trial_timepoints_min": int(min(lengths)),
                    "trial_timepoints_max": int(max(lengths)),
                    "raw_neurons": raw_n,
                    "retained_neurons": kept_n,
                    "neural_frames": int(nframes_neural),
                }
            )

            # Trial matrices now own their data.
            del activity, iarea, frames_by_trial
            gc.collect()
            session_number += 1
            elapsed = time.perf_counter() - t0
            total_elapsed = time.perf_counter() - conversion_start
            rate = total_elapsed / session_number
            eta = rate * (len(records) - session_number)
            print(
                f"[{session_number:02d}/{len(records):02d}] {sid}: "
                f"{len(trial_neural)} trials, {kept_n:,}/{raw_n:,} cells, "
                f"{sum(lengths):,} timepoints, {elapsed:.2f} s; ETA {eta/60:.1f} min",
                flush=True,
            )
        del behavior_dict
        gc.collect()

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": BRAIN_REGIONS,
        "brain_region_idx": brain_region_sessions,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Decode visual texture family, binary licking, 1-m corridor position, "
                "and running-speed quartile from visual-cortex deconvolved activity."
            ),
            "time_bin_size": float(TIME_BIN_MS),
            "temporal_alignment_event": "trial start (entry into the 4-m textured corridor)",
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": (
                "Released Suite2p non-negative deconvolved fluorescence/spks; "
                "0.75-s decay; no additional normalization or interpolation."
            ),
            "sampling_rate_hz": float(IMAGING_RATE_HZ),
            "frame_selection": "ft_trInd == trial AND ft_CorrSpc AND ft_move > 0",
            "stationary_frames_removed": True,
            "timepoints_may_have_gaps": True,
            "corridor_m": 4.0,
            "gray_space_m": 2.0,
            "position_source_units_per_m": 10.0,
            "running_speed_quartile_edges": [float(x) for x in speed_edges],
            "running_speed_units": "native ft_RunSpeed units",
            "stimulus_category_mapping": {
                "circle*": 0,
                "leaf*": 1,
                "rock*": 2,
                "wood*": 3,
            },
            "reward_availability_definition": (
                "All trials whose WallName equals the unique wall identified by any "
                "isRew=True trial; not merely trials where reward was delivered."
            ),
            "brain_region_mapping": {
                "V1": [8],
                "mHV": [0, 1, 2, 9],
                "lHV": [5, 6],
                "aHV": [3, 4],
                "excluded_outside_visual_cortex": [-1, 7],
            },
            "conversion_mode": mode_name,
            "n_sessions": int(len(records)),
            "n_trials": int(prepass["n_trials"]),
            "n_selected_timepoints": int(prepass["n_timepoints"]),
            "raw_neurons_in_converted_sessions": int(raw_total),
            "retained_neurons_in_converted_sessions": int(retained_total),
            "reference_full_dataset": {
                "recordings": 89,
                "subjects": 19,
                "trials": 38110,
                "raw_neuron_session_units": 4691034,
                "retained_named_visual_area_neuron_session_units": 4105393,
                "paper_raw_neurons_per_recording_min": 20547,
                "paper_raw_neurons_per_recording_max": 89577,
            },
            "behavior_prepass": prepass,
            "session_info": session_info,
        },
    }
    return data


def main() -> None:
    args = parse_args()
    args.outpicklefile = args.outpicklefile.resolve()
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    data = convert(args)
    conversion_elapsed = time.perf_counter() - total_start
    print(
        f"Conversion in memory complete in {conversion_elapsed:.2f} s; "
        f"writing {args.outpicklefile}",
        flush=True,
    )
    write_start = time.perf_counter()
    with open(args.outpicklefile, "wb", buffering=16 * 1024 * 1024) as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_elapsed = time.perf_counter() - write_start
    size_gb = args.outpicklefile.stat().st_size / (1024 ** 3)
    print(
        f"Wrote {args.outpicklefile} ({size_gb:.3f} GiB) in {write_elapsed:.2f} s; "
        f"total {time.perf_counter() - total_start:.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
