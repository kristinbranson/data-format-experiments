#!/usr/bin/env python3
"""Convert NWB sessions from Sosa et al. into decoder-ready pickle format."""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DATA_ROOT = Path("/app/data")
SESSIONS_DICT_PATH = Path("/app/code/src/reward_relative/sessions_dict.py")
TRACK_LENGTH_CM = 450.0
COMMON_FRAME_RATE_HZ = 15.5078125
COMMON_FRAME_DT_S = 1.0 / COMMON_FRAME_RATE_HZ
TIME_BIN_SIZE_MS = COMMON_FRAME_DT_S * 1000.0
LICK_ERROR_THRESHOLD = 0.30
SAMPLE_SHOW_MAX = 2

REWARD_ZONE_COORDS = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
RZ_CODE = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class SessionMeta:
    date: str
    scene: str
    session: int
    scan: float | int
    exp_day: int


def load_sessions_dict() -> dict[tuple[str, int], SessionMeta]:
    spec = importlib.util.spec_from_file_location("sessions_dict_local", SESSIONS_DICT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sessions dict from {SESSIONS_DICT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mapping: dict[tuple[str, int], SessionMeta] = {}
    for collection_name in ("single_plane", "multi_plane"):
        collection = getattr(module, collection_name)
        for animal, entries in collection.items():
            for entry in entries:
                meta = SessionMeta(
                    date=entry["date"],
                    scene=entry["scene"],
                    session=int(entry["session"]),
                    scan=entry["scan"],
                    exp_day=int(entry["exp_day"]),
                )
                mapping[(animal, meta.exp_day)] = meta
    return mapping


def subject_to_gcamp(subject_id: str) -> str:
    if subject_id.startswith("m"):
        return f"GCAMP{subject_id[1:]}"
    raise ValueError(f"Unexpected subject id: {subject_id}")


def scene_reward_labels(scene: str, n_trials: int, switch_trial: int = 30) -> list[str]:
    if scene in {"Env1_LocationA", "Env2_LocationA", "Env3_LocationA"}:
        return ["A"] * n_trials
    if scene in {"Env1_LocationB", "Env2_LocationB", "Env3_LocationB"}:
        return ["B"] * n_trials
    if scene in {"Env1_LocationC", "Env2_LocationC", "Env3_LocationC"}:
        return ["C"] * n_trials

    transitions = [
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("B", "C"),
        ("C", "B"),
    ]
    for src, dst in transitions:
        if f"{src}_to" in scene and scene.endswith(dst):
            return [src] * min(switch_trial, n_trials) + [dst] * max(0, n_trials - switch_trial)

    raise ValueError(f"Unsupported scene for reward-zone mapping: {scene}")


def pair_trial_events(trial_start: np.ndarray, teleport: np.ndarray, scanning: np.ndarray) -> list[tuple[int, int]]:
    start_idx = np.where((trial_start > 0) & (scanning > 0))[0]
    end_idx = np.where((teleport > 0) & (scanning > 0))[0]

    pairs: list[tuple[int, int]] = []
    end_ptr = 0
    for start in start_idx:
        while end_ptr < len(end_idx) and end_idx[end_ptr] <= start:
            end_ptr += 1
        if end_ptr >= len(end_idx):
            break
        end = end_idx[end_ptr]
        if end > start:
            pairs.append((int(start), int(end)))
        end_ptr += 1
    return pairs


def detect_bad_lick_trials(lick: np.ndarray, trial_pairs: list[tuple[int, int]]) -> np.ndarray:
    bad = np.zeros(len(trial_pairs), dtype=bool)
    for i, (start, end) in enumerate(trial_pairs):
        trial_lick = lick[start:end]
        if trial_lick.size == 0:
            bad[i] = True
            continue
        bad[i] = np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD
    return bad


def discretize_reward_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, 6, dtype=np.int16)
    out[distance_cm < -50] = 0
    out[(distance_cm >= -50) & (distance_cm < -10)] = 1
    out[(distance_cm >= -10) & (distance_cm < 0)] = 2
    out[distance_cm == 0] = 3
    out[(distance_cm > 0) & (distance_cm <= 10)] = 4
    out[(distance_cm > 10) & (distance_cm <= 50)] = 5
    return out


def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    out = np.zeros(position_cm.shape, dtype=np.int16)
    out[(position_cm >= 90) & (position_cm < 180)] = 1
    out[(position_cm >= 180) & (position_cm < 270)] = 2
    out[(position_cm >= 270) & (position_cm <= 360)] = 3
    out[position_cm > 360] = 4
    return out


def discretize_speed(speed_cm_s: np.ndarray) -> np.ndarray:
    out = np.zeros(speed_cm_s.shape, dtype=np.int16)
    out[(speed_cm_s >= 2) & (speed_cm_s < 10)] = 1
    out[(speed_cm_s >= 10) & (speed_cm_s < 20)] = 2
    out[(speed_cm_s >= 20) & (speed_cm_s <= 40)] = 3
    out[speed_cm_s > 40] = 4
    return out


def reward_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist = np.zeros(position_cm.shape, dtype=np.float32)
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
    return dist


def repeated_row(value: float | int, T: int, dtype: np.dtype) -> np.ndarray:
    return np.full((T,), value, dtype=dtype)


def median_step_seconds(timestamps: np.ndarray) -> float:
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return COMMON_FRAME_DT_S
    return float(np.median(diffs))


def select_sample_files(all_files: list[Path]) -> list[Path]:
    if len(all_files) <= 2:
        return all_files
    first = all_files[0]
    two_plane = next((f for f in all_files if "/sub-m17/" in str(f) or "/sub-m18/" in str(f)), None)
    if two_plane is None or two_plane == first:
        return all_files[:2]
    return [first, two_plane]


def make_processing_plot(
    session_tag: str,
    timestamps: np.ndarray,
    position: np.ndarray,
    speed: np.ndarray,
    lick: np.ndarray,
    environment: np.ndarray,
    trial_pairs: list[tuple[int, int]],
    bad_lick_trials: np.ndarray,
    reward_labels_kept: list[str],
    reward_outcomes_kept: list[int],
    neural_trials: list[np.ndarray],
    input_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not trial_pairs:
        return

    rep_trials = [0, min(29, len(neural_trials) - 1), len(neural_trials) - 1]
    rep_trials = sorted(set(rep_trials))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    ax = axes.ravel()

    # Raw session overview around first two trial boundaries.
    window_start = max(0, trial_pairs[0][0] - 40)
    window_end = min(len(position), trial_pairs[min(1, len(trial_pairs) - 1)][1] + 60)
    ax[0].plot(timestamps[window_start:window_end], position[window_start:window_end], color="black", lw=1)
    for i, (start, end) in enumerate(trial_pairs[:2]):
        ax[0].axvline(timestamps[start], color="green", alpha=0.7, lw=1)
        ax[0].axvline(timestamps[end], color="red", alpha=0.7, lw=1)
        ax[0].text(timestamps[start], 460, f"trial {i}", fontsize=8, rotation=90, va="top")
    ax[0].set_title("Raw position around trial starts/teleports")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (cm)")

    # Representative trial-aligned position and reward-distance bins.
    for idx in rep_trials:
        T = input_trials[idx].shape[1]
        ax[1].plot(input_trials[idx][0], output_trials[idx][1], lw=1, label=f"trial {idx}")
        ax[1].scatter(input_trials[idx][0], output_trials[idx][0], s=4)
        ax[1].text(input_trials[idx][0][-1], output_trials[idx][0][-1], reward_labels_kept[idx], fontsize=8)
    ax[1].set_title("Trial-aligned position bins and reward-distance bins")
    ax[1].set_xlabel("Time from trial start (s)")
    ax[1].set_ylabel("Output code")
    ax[1].legend(fontsize=8, loc="upper right")

    # Lick correction and speed on representative trials.
    for idx in rep_trials:
        ax[2].plot(input_trials[idx][0], output_trials[idx][2] + 0.15 * idx, lw=1, label=f"speed bin {idx}")
        ax[2].step(input_trials[idx][0], output_trials[idx][3] + 0.15 * idx, where="mid", lw=1)
    bad_text = ", ".join(str(i) for i in np.where(bad_lick_trials)[0][:8])
    if not bad_text:
        bad_text = "none"
    ax[2].set_title(f"Speed and lick bins; bad lick trials: {bad_text}")
    ax[2].set_xlabel("Time from trial start (s)")
    ax[2].set_ylabel("Output code")

    # Neural heatmap for one representative trial.
    neural_show = neural_trials[rep_trials[0]][: min(60, neural_trials[rep_trials[0]].shape[0]), :]
    vmax = np.percentile(neural_show, 99) if neural_show.size else 1.0
    ax[3].imshow(neural_show, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
    ax[3].set_title(f"Neural activity heatmap ({neural_show.shape[0]} neurons shown)")
    ax[3].set_xlabel("Time bins")
    ax[3].set_ylabel("Neuron")

    # Trial metadata summary.
    trial_env = [int(input_trials[i][1, 0]) for i in range(len(input_trials))]
    zone_counts = {lab: reward_labels_kept.count(lab) for lab in ("A", "B", "C")}
    ax[4].bar(["ENV1", "ENV2"], [trial_env.count(0), trial_env.count(1)], color=["#4c78a8", "#f58518"])
    ax[4].set_title("Environment trials")
    ax[4].set_ylabel("Count")
    ax4b = ax[4].twinx()
    ax4b.plot(["A", "B", "C"], [zone_counts["A"], zone_counts["B"], zone_counts["C"]], color="black", marker="o")
    ax4b.set_ylabel("Reward-zone count")

    # Reward outcomes and trial lengths.
    trial_lengths = [arr.shape[1] for arr in neural_trials]
    ax[5].scatter(
        np.arange(len(trial_lengths)),
        trial_lengths,
        s=10,
        c=np.asarray(reward_outcomes_kept),
        cmap="coolwarm",
        vmin=0,
        vmax=1,
    )
    ax[5].set_title("Trial lengths (color = reward outcome)")
    ax[5].set_xlabel("Trial index")
    ax[5].set_ylabel("Time bins")

    fig.suptitle(f"Processing summary: {session_tag}")
    fig.tight_layout()
    fig.savefig(f"processing_{session_tag}.png", dpi=150)
    plt.close(fig)


def convert_session(
    nwb_path: Path,
    sessions_dict: dict[tuple[str, int], SessionMeta],
    make_plot: bool,
) -> tuple[dict, dict]:
    start_time = time.perf_counter()
    session_tag = nwb_path.stem.replace("sub-", "").replace("_behavior+ophys", "")

    with h5py.File(nwb_path, "r") as h5:
        subject_id = h5["general/subject/subject_id"][()].decode()
        exp_day = int(h5["general/session_id"][()].decode())
        gcamp = subject_to_gcamp(subject_id)
        session_meta = sessions_dict[(gcamp, exp_day)]

        beh_root = h5["processing/behavior/BehavioralTimeSeries"]
        frame_timestamps = beh_root["position"]["timestamps"][()].astype(np.float64)
        position = beh_root["position"]["data"][()].astype(np.float32)
        environment = beh_root["environment"]["data"][()].astype(np.float32)
        lick = beh_root["lick"]["data"][()].astype(np.float32)
        speed = beh_root["speed"]["data"][()].astype(np.float32)
        trial_number = beh_root["trial number"]["data"][()].astype(np.int32)
        trial_start = beh_root["trial_start"]["data"][()].astype(np.float32)
        teleport = beh_root["teleport"]["data"][()].astype(np.float32)
        scanning = beh_root["scanning"]["data"][()].astype(np.float32)
        reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)

        seg = h5["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        iscell = seg["iscell"][()]
        plane_idx = seg["planeIdx"][()].astype(np.int16)
        keep_mask = iscell[:, 0] > 0.5
        keep_indices = np.flatnonzero(keep_mask)
        brain_region_idx = np.zeros(len(keep_indices), dtype=np.int16)
        deconv_root = h5["processing/ophys/Deconvolved"]
        unique_planes = np.unique(plane_idx)
        plane_frame_counts = [int(deconv_root[f"plane{int(plane)}"]["data"].shape[0]) for plane in unique_planes]
        min_frames = min([frame_timestamps.shape[0], *plane_frame_counts])
        frame_timestamps = frame_timestamps[:min_frames]
        position = position[:min_frames]
        environment = environment[:min_frames]
        lick = lick[:min_frames]
        speed = speed[:min_frames]
        trial_number = trial_number[:min_frames]
        trial_start = trial_start[:min_frames]
        teleport = teleport[:min_frames]
        scanning = scanning[:min_frames]

        deconv = np.empty((min_frames, keep_indices.size), dtype=np.float16)
        local_index = np.empty(plane_idx.shape[0], dtype=np.int32)
        for plane in unique_planes:
            plane_mask = plane_idx == plane
            local_index[plane_mask] = np.arange(np.sum(plane_mask), dtype=np.int32)
        kept_plane_idx = plane_idx[keep_indices]
        for plane in np.unique(kept_plane_idx):
            kept_positions = np.where(kept_plane_idx == plane)[0]
            local_cols = local_index[keep_indices[kept_positions]]
            plane_data = deconv_root[f"plane{int(plane)}"]["data"][:min_frames, local_cols]
            deconv[:, kept_positions] = plane_data.astype(np.float16)

    frame_dt = median_step_seconds(frame_timestamps)
    if not np.isclose(frame_dt, COMMON_FRAME_DT_S, atol=1e-6):
        raise RuntimeError(f"{session_tag}: unexpected frame dt {frame_dt}")

    trial_pairs = pair_trial_events(trial_start, teleport, scanning)
    bad_lick_trials = detect_bad_lick_trials(lick, trial_pairs)
    reward_labels_all = scene_reward_labels(session_meta.scene, len(trial_pairs))

    reward_outcomes_all = np.zeros(len(trial_pairs), dtype=np.int16)
    for i, (start, end) in enumerate(trial_pairs):
        start_ts = frame_timestamps[start]
        end_ts = frame_timestamps[end]
        reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    kept_reward_labels: list[str] = []
    kept_reward_outcomes: list[int] = []
    kept_trial_numbers: list[int] = []

    for trial_idx, ((start, end), bad_lick, reward_label) in enumerate(
        zip(trial_pairs, bad_lick_trials, reward_labels_all, strict=True)
    ):
        if bad_lick:
            continue

        trial_slice = slice(start, end)  # exclude teleport frame itself
        pos_trial = position[trial_slice]
        if pos_trial.size == 0:
            continue

        T = pos_trial.shape[0]
        t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
        env_trial_vals = environment[trial_slice]
        env_trial_vals = env_trial_vals[env_trial_vals >= 0]
        if env_trial_vals.size == 0:
            raise RuntimeError(f"{session_tag}: no valid environment values in trial {trial_idx}")
        env_trial = int(np.rint(np.median(env_trial_vals)))

        trial_num = int(trial_number[start])
        prev_reward = int(reward_outcomes_all[trial_idx - 1]) if trial_idx > 0 else 0
        reward_outcome = int(reward_outcomes_all[trial_idx])

        zone_start, zone_end = REWARD_ZONE_COORDS[reward_label]
        reward_dist = reward_distance_to_zone(pos_trial, zone_start, zone_end)
        reward_dist_bin = discretize_reward_distance(reward_dist)
        pos_bin = discretize_absolute_position(pos_trial)
        speed_bin = discretize_speed(speed[trial_slice])
        lick_bin = (lick[trial_slice] > 0).astype(np.int16)

        neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
        input_trial = np.ascontiguousarray(
            np.vstack(
                [
                    t_trial.astype(np.float32),
                    repeated_row(env_trial, T, np.float32),
                    repeated_row(trial_num, T, np.float32),
                    repeated_row(prev_reward, T, np.float32),
                ]
            ),
            dtype=np.float32,
        )
        output_trial = np.ascontiguousarray(
            np.vstack(
                [
                    reward_dist_bin,
                    pos_bin,
                    speed_bin,
                    lick_bin,
                    repeated_row(RZ_CODE[reward_label], T, np.int16),
                    repeated_row(reward_outcome, T, np.int16),
                ]
            ),
            dtype=np.int16,
        )

        if neural_trial.shape[1] != input_trial.shape[1] or neural_trial.shape[1] != output_trial.shape[1]:
            raise RuntimeError(f"{session_tag}: time dimension mismatch in trial {trial_idx}")
        if neural_trial.shape[0] != brain_region_idx.shape[0]:
            raise RuntimeError(f"{session_tag}: neuron dimension mismatch")
        if np.isnan(neural_trial).any() or np.isnan(input_trial).any() or np.isnan(output_trial).any():
            raise RuntimeError(f"{session_tag}: NaN values detected after conversion")

        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)
        kept_reward_labels.append(reward_label)
        kept_reward_outcomes.append(reward_outcome)
        kept_trial_numbers.append(trial_num)

    if len(neural_trials) < 2:
        raise RuntimeError(f"{session_tag}: fewer than 2 valid trials after filtering")

    if make_plot:
        make_processing_plot(
            session_tag=session_tag,
            timestamps=frame_timestamps,
            position=position,
            speed=speed,
            lick=lick,
            environment=environment,
            trial_pairs=trial_pairs,
            bad_lick_trials=bad_lick_trials,
            reward_labels_kept=kept_reward_labels,
            reward_outcomes_kept=kept_reward_outcomes,
            neural_trials=neural_trials,
            input_trials=input_trials,
            output_trials=output_trials,
        )

    summary = {
        "session_tag": session_tag,
        "subject_id": subject_id,
        "exp_day": exp_day,
        "scene": session_meta.scene,
        "n_rois_total": int(iscell.shape[0]),
        "n_rois_kept": int(keep_indices.size),
        "n_trials_complete": int(len(trial_pairs)),
        "n_trials_bad_lick": int(np.sum(bad_lick_trials)),
        "n_trials_kept": int(len(neural_trials)),
        "mean_trial_len": float(np.mean([arr.shape[1] for arr in neural_trials])),
        "frame_dt_s": frame_dt,
        "n_frames_used": int(frame_timestamps.shape[0]),
        "plane_values": sorted(np.unique(plane_idx[keep_mask]).astype(int).tolist()),
        "rate_attr_note": "Ignored NWB deconvolved rate attribute; used behavior timestamps as master clock.",
        "elapsed_s": time.perf_counter() - start_time,
    }

    converted = {
        "subject_id": subject_id,
        "brain_region_idx": brain_region_idx,
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
        "summary": summary,
    }
    return converted, summary


def build_dataset(converted_sessions: list[dict], session_summaries: list[dict]) -> dict:
    subjects = sorted({sess["subject_id"] for sess in converted_sessions})
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}

    data = {
        "neural": [sess["neural_trials"] for sess in converted_sessions],
        "input": [sess["input_trials"] for sess in converted_sessions],
        "output": [sess["output_trials"] for sess in converted_sessions],
        "subjects": subjects,
        "subject_idx": np.asarray([subject_lookup[sess["subject_id"]] for sess in converted_sessions], dtype=np.int16),
        "brain_regions": ["CA1"],
        "brain_region_idx": [sess["brain_region_idx"] for sess in converted_sessions],
        "input_names": [
            "time_from_trial_start_s",
            "environment_type",
            "trial_number",
            "previous_trial_outcome",
        ],
        "output_names": [
            "distance_to_reward_zone",
            "absolute_position",
            "speed",
            "lick",
            "reward_zone_location",
            "reward_outcome",
        ],
        "output_values": [
            ["lt_-50", "-50_to_-10", "-10_to_lt0", "0", "gt0_to_10", "gt10_to_50", "gt50"],
            ["lt_90", "90_to_180", "180_to_270", "270_to_360", "gt_360"],
            ["lt_2", "2_to_10", "10_to_20", "20_to_40", "gt_40"],
            ["no", "yes"],
            ["A", "B", "C"],
            ["no", "yes"],
        ],
        "metadata": {
            "task_description": (
                "CA1 two-photon imaging during virtual-reality navigation on a 450 cm track; "
                "decoder predicts reward-zone distance, absolute position, speed, lick, reward-zone label, "
                "and reward outcome from deconvolved neural activity."
            ),
            "time_bin_size": TIME_BIN_SIZE_MS,
            "temporal_alignment_event": "trial_start pulse / entry to linear track",
            "off_start": 0.0,
            "off_end": None,
            "source_dataset": "DANDI 001361 NWB export of Sosa, Plitt, Giocomo 2025",
            "track_length_cm": TRACK_LENGTH_CM,
            "reward_zone_coords_cm": REWARD_ZONE_COORDS,
            "trial_window_rule": "frames from trial_start through frame before teleport",
            "neural_signal": (
                "deconvolved calcium activity from NWB processing/ophys/Deconvolved/plane* "
                "datasets, pooled across imaging planes by ROI plane index"
            ),
            "roi_filter": "PlaneSegmentation iscell[:,0] == 1",
            "lick_fault_rule": (
                "Trials removed if >30% of trial frames had cumulative lick count > 2, "
                "matching manuscript lick-sensor QC logic."
            ),
            "session_info": session_summaries,
        },
    }
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Sosa et al. NWB data to decoder format.")
    parser.add_argument("outpicklefile", type=Path, help="Output pickle file path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", dest="mode", action="store_const", const="full", default="full",
                      help="Process all sessions (default).")
    mode.add_argument("--sample", dest="mode", action="store_const", const="sample",
                      help="Process only 2 representative sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not all_files:
        raise RuntimeError(f"No NWB files found under {DATA_ROOT}")

    selected_files = all_files if args.mode == "full" else select_sample_files(all_files)
    sessions_dict = load_sessions_dict()

    print(f"Mode: {args.mode}")
    print(f"Found {len(all_files)} total NWB files; processing {len(selected_files)}")
    print(f"Common frame dt from reference behavior timestamps: {COMMON_FRAME_DT_S:.9f} s")

    t0 = time.perf_counter()
    converted_sessions: list[dict] = []
    session_summaries: list[dict] = []
    plots_remaining = SAMPLE_SHOW_MAX if args.show_processing else 0

    for file_idx, nwb_path in enumerate(selected_files, start=1):
        do_plot = plots_remaining > 0
        print(f"[{file_idx}/{len(selected_files)}] Processing {nwb_path.name} ...", flush=True)
        converted, summary = convert_session(nwb_path, sessions_dict, do_plot)
        converted_sessions.append(converted)
        session_summaries.append(summary)
        if do_plot:
            plots_remaining -= 1
        print(
            f"  kept {summary['n_rois_kept']} neurons, "
            f"{summary['n_trials_kept']} trials "
            f"({summary['n_trials_bad_lick']} bad-lick trials removed), "
            f"mean T={summary['mean_trial_len']:.1f}, "
            f"elapsed={summary['elapsed_s']:.2f}s",
            flush=True,
        )

    data = build_dataset(converted_sessions, session_summaries)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_elapsed = time.perf_counter() - t0
    total_trials = sum(len(sess["neural_trials"]) for sess in converted_sessions)
    total_neurons = sum(int(sess["brain_region_idx"].shape[0]) for sess in converted_sessions)
    print(f"Saved {args.outpicklefile}")
    print(f"Sessions: {len(converted_sessions)}")
    print(f"Trials kept: {total_trials}")
    print(f"Neurons kept across sessions: {total_neurons}")
    print(f"Total elapsed: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
