#!/usr/bin/env python3
"""Convert the Zhong et al. 2025 dataset into decoder format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("/app/data")
BEH_ROOT = DATA_ROOT / "beh"
SPK_ROOT = DATA_ROOT / "spk"
RETINO_ROOT = DATA_ROOT / "retinotopy"

FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ

INPUT_NAMES = [
    "time_to_sound_cue_s",
    "day_of_training",
    "time_since_trial_start_s",
    "reward_availability",
]

OUTPUT_NAMES = [
    "visual_stimulus_category",
    "licking",
    "position_bin",
    "running_speed_bin",
]

BRAIN_REGIONS = [
    "V1",
    "mHV",
    "lHV",
    "aHV",
    "unassigned_7",
    "outside_visual",
]


@dataclass(frozen=True)
class SessionInfo:
    raw_id: str
    subject: str
    dateexp: str
    block: str
    behavior_exp_type: str
    behavior_key: str
    aliases: tuple[str, ...]
    exp_types: tuple[str, ...]

    @property
    def date_obj(self):
        return datetime.strptime(self.dateexp, "%Y_%m_%d").date()

    @property
    def spk_path(self) -> Path:
        return SPK_ROOT / f"{self.raw_id}_neural_data.npy"

    @property
    def retino_path(self) -> Path:
        return RETINO_ROOT / f"{self.subject}_{self.dateexp}_trans.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Zhong et al. 2025 data.")
    parser.add_argument("outpicklefile", type=Path, help="Output pickle path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions for testing.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing visualizations for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def canonical_raw_id(entry: dict[str, Any]) -> str:
    return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"


def behavior_key_for_entry(entry: dict[str, Any]) -> str:
    raw_id = canonical_raw_id(entry)
    if "stimtype" in entry:
        return f"{raw_id}_{entry['stimtype']}"
    return raw_id


def load_experiment_index() -> dict[str, list[dict[str, Any]]]:
    return np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()


@lru_cache(maxsize=2)
def load_behavior_file(exp_type: str) -> dict[str, dict[str, Any]]:
    return np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()


def build_canonical_sessions() -> list[SessionInfo]:
    exp_info = load_experiment_index()
    per_raw: dict[str, dict[str, Any]] = {}

    for exp_type, entries in exp_info.items():
        beh_file = load_behavior_file(exp_type)
        for entry in entries:
            raw_id = canonical_raw_id(entry)
            beh_key = behavior_key_for_entry(entry)

            rec = per_raw.setdefault(
                raw_id,
                {
                    "subject": entry["mname"],
                    "dateexp": entry["datexp"],
                    "block": entry["blk"],
                    "behavior_exp_type": exp_type,
                    "behavior_key": beh_key,
                    "aliases": set(),
                    "exp_types": set(),
                },
            )
            rec["aliases"].add(beh_key)
            rec["exp_types"].add(exp_type)

            # Keep the first-seen canonical behavior source; Step 4 verified that
            # alias keys for the same raw session are duplicates on core arrays.
            if rec["behavior_key"] not in beh_file and beh_key in beh_file:
                rec["behavior_exp_type"] = exp_type
                rec["behavior_key"] = beh_key

    sessions = [
        SessionInfo(
            raw_id=raw_id,
            subject=rec["subject"],
            dateexp=rec["dateexp"],
            block=rec["block"],
            behavior_exp_type=rec["behavior_exp_type"],
            behavior_key=rec["behavior_key"],
            aliases=tuple(sorted(rec["aliases"])),
            exp_types=tuple(sorted(rec["exp_types"])),
        )
        for raw_id, rec in sorted(per_raw.items())
    ]

    return sessions


def get_behavior(session: SessionInfo) -> dict[str, Any]:
    beh_file = load_behavior_file(session.behavior_exp_type)
    return beh_file[session.behavior_key]


def compute_day_offsets(sessions: list[SessionInfo]) -> dict[str, float]:
    subject_first_date = {}
    for session in sessions:
        subject_first_date[session.subject] = min(
            subject_first_date.get(session.subject, session.date_obj),
            session.date_obj,
        )
    return {
        session.raw_id: float((session.date_obj - subject_first_date[session.subject]).days)
        for session in sessions
    }


def map_iarea_to_region_idx(iarea: np.ndarray) -> np.ndarray:
    out = np.empty(iarea.shape[0], dtype=np.int16)
    out.fill(-1)
    out[iarea == 8] = 0  # V1
    out[np.isin(iarea, [0, 1, 2, 9])] = 1  # mHV
    out[np.isin(iarea, [5, 6])] = 2  # lHV
    out[np.isin(iarea, [3, 4])] = 3  # aHV
    out[iarea == 7] = 4  # unassigned_7
    out[iarea == -1] = 5  # outside_visual
    if np.any(out < 0):
        raise ValueError(f"Found unexpected iarea labels: {sorted(np.unique(iarea[out < 0]).tolist())}")
    return out


def trial_frame_groups(beh: dict[str, Any]) -> list[np.ndarray]:
    ft_tr = np.asarray(beh["ft_trInd"])
    ft_corr = np.asarray(beh["ft_CorrSpc"], dtype=bool)
    ft_move = np.asarray(beh["ft_move"]) > 0

    finite = np.isfinite(ft_tr)
    tr_int = np.full(ft_tr.shape, -1, dtype=np.int32)
    tr_int[finite] = ft_tr[finite].astype(np.int32)

    keep = finite & ft_corr & ft_move
    frame_idx = np.flatnonzero(keep)
    trial_ids = tr_int[keep]
    ntrials = int(beh["ntrials"])
    groups = [np.empty(0, dtype=np.int32) for _ in range(ntrials)]

    if frame_idx.size == 0:
        return groups

    changes = np.flatnonzero(np.diff(trial_ids)) + 1
    split_frames = np.split(frame_idx, changes)
    split_trials = np.split(trial_ids, changes)
    for frames, tids in zip(split_frames, split_trials):
        groups[int(tids[0])] = frames.astype(np.int32, copy=False)
    return groups


def build_lick_binary(beh: dict[str, Any], nframes: int) -> np.ndarray:
    lick_binary = np.zeros(nframes, dtype=np.int8)
    lick_frames = np.asarray(beh["LickFr"])
    if lick_frames.size == 0:
        return lick_binary
    finite = np.isfinite(lick_frames)
    lick_idx = lick_frames[finite].astype(np.int64, copy=False)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
    if lick_idx.size:
        lick_binary[np.unique(lick_idx)] = 1
    return lick_binary


def position_to_bin(pos: np.ndarray) -> np.ndarray:
    bins = np.floor(np.clip(pos, 0.0, 39.999999) / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)


def stabilized_quantile_edges(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(values, [0.25, 0.50, 0.75]).astype(np.float32)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf, dtype=np.float32)
    return edges


def speed_to_bin(speed: np.ndarray, speed_edges: np.ndarray) -> np.ndarray:
    return np.digitize(speed, speed_edges, right=False).astype(np.int16)


def gather_stimulus_vocabulary(sessions: list[SessionInfo]) -> list[str]:
    vocab = set()
    for session in sessions:
        beh = get_behavior(session)
        vocab.update(map(str, np.unique(np.asarray(beh["WallName"]))))
    return sorted(vocab)


def gather_speed_edges(sessions: list[SessionInfo]) -> np.ndarray:
    t0 = time.time()
    speeds = []
    total_frames = 0
    for idx, session in enumerate(sessions, start=1):
        beh = get_behavior(session)
        groups = trial_frame_groups(beh)
        ft_run_speed = np.asarray(beh["ft_RunSpeed"], dtype=np.float32)
        for frames in groups:
            if frames.size == 0:
                continue
            vals = ft_run_speed[frames]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                speeds.append(vals)
                total_frames += vals.size
        if idx % 20 == 0 or idx == len(sessions):
            print(
                f"[speed-scan] processed {idx}/{len(sessions)} sessions, "
                f"retained frames so far: {total_frames}",
                flush=True,
            )
    if not speeds:
        raise RuntimeError("No running-speed samples found while computing quartile edges.")
    all_speeds = np.concatenate(speeds)
    edges = stabilized_quantile_edges(all_speeds)
    print(
        f"[speed-scan] finished in {time.time() - t0:.2f}s, "
        f"frames={all_speeds.size}, edges={edges.tolist()}",
        flush=True,
    )
    return edges


def choose_sample_sessions(sessions: list[SessionInfo], nsample: int = 2) -> list[SessionInfo]:
    representative = []
    for session in sessions:
        beh = get_behavior(session)
        is_rew = np.asarray(beh["isRew"], dtype=bool)
        has_both_reward_states = bool(np.any(is_rew)) and bool(np.any(~is_rew))
        has_any_lick = np.asarray(beh["LickFr"]).size > 0
        if has_both_reward_states and has_any_lick:
            representative.append(session)
        if len(representative) >= nsample:
            break
    if len(representative) >= nsample:
        return representative[:nsample]
    return sessions[:nsample]


def convert_session(
    session: SessionInfo,
    stim_to_idx: dict[str, int],
    day_offsets: dict[str, float],
    speed_edges: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, Any]]:
    t0 = time.time()
    beh = get_behavior(session)
    spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
    retino = np.load(session.retino_path, allow_pickle=True)

    iarea = np.asarray(retino["iarea"], dtype=np.float32)
    brain_region_idx = map_iarea_to_region_idx(iarea)
    total_rows = sum(part.shape[0] for part in spk_parts)
    if total_rows != brain_region_idx.shape[0]:
        raise ValueError(
            f"{session.raw_id}: neural rows ({total_rows}) do not match retinotopy rows ({brain_region_idx.shape[0]})"
        )

    ft = np.asarray(beh["ft"], dtype=np.float64)
    lick_binary = build_lick_binary(beh, nframes=ft.shape[0])
    groups = trial_frame_groups(beh)

    wall_name = np.asarray(beh["WallName"])
    is_rew = np.asarray(beh["isRew"], dtype=bool)
    trial_start_time = np.asarray(beh["Trial_start_time"], dtype=np.float64)
    sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
    ft_pos = np.asarray(beh["ft_Pos"], dtype=np.float32)
    ft_run_speed = np.asarray(beh["ft_RunSpeed"], dtype=np.float32)

    session_neural: list[np.ndarray] = []
    session_input: list[np.ndarray] = []
    session_output: list[np.ndarray] = []

    day_value = np.float32(day_offsets[session.raw_id])

    for trial, frames in enumerate(groups):
        if frames.size == 0:
            continue

        neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)

        frame_times = ft[frames]
        time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
        time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
        reward_availability = np.full(frames.size, float(is_rew[trial]), dtype=np.float32)
        day_of_training = np.full(frames.size, day_value, dtype=np.float32)

        input_trial = np.stack(
            [time_to_cue, day_of_training, time_since_start, reward_availability],
            axis=0,
        )

        stimulus_idx = stim_to_idx[str(wall_name[trial])]
        stimulus_out = np.full(frames.size, stimulus_idx, dtype=np.int16)
        lick_out = lick_binary[frames].astype(np.int16, copy=False)
        pos_out = position_to_bin(ft_pos[frames])
        speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)

        output_trial = np.stack([stimulus_out, lick_out, pos_out, speed_out], axis=0)

        session_neural.append(neural_trial)
        session_input.append(input_trial)
        session_output.append(output_trial)

    session_meta = {
        "raw_id": session.raw_id,
        "subject": session.subject,
        "dateexp": session.dateexp,
        "block": session.block,
        "behavior_exp_type": session.behavior_exp_type,
        "behavior_key": session.behavior_key,
        "aliases": list(session.aliases),
        "exp_types": list(session.exp_types),
        "n_trials_original": int(beh["ntrials"]),
        "n_trials_converted": len(session_neural),
        "day_of_training": float(day_offsets[session.raw_id]),
        "reward_mode": str(beh["Reward_Mode"]),
        "stimuli_present": sorted(map(str, np.unique(wall_name))),
        "n_rows_neural": int(total_rows),
        "conversion_seconds": time.time() - t0,
    }

    return session_neural, session_input, session_output, brain_region_idx, session_meta


def plot_processing_summary(
    session: SessionInfo,
    session_meta: dict[str, Any],
    beh: dict[str, Any],
    neural_trials: list[np.ndarray],
    input_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    speed_edges: np.ndarray,
) -> None:
    out_path = Path(f"processing_{session.raw_id}.png")

    groups = trial_frame_groups(beh)
    ft_tr = np.asarray(beh["ft_trInd"])
    finite = np.isfinite(ft_tr)
    tr_int = np.full(ft_tr.shape, -1, dtype=np.int32)
    tr_int[finite] = ft_tr[finite].astype(np.int32)
    corr_mask = finite & np.asarray(beh["ft_CorrSpc"], dtype=bool)
    run_mask = corr_mask & (np.asarray(beh["ft_move"]) > 0)

    fig, ax = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f"Processing Summary: {session.raw_id}")

    # Mask effect on position samples.
    pos = np.asarray(beh["ft_Pos"], dtype=np.float32)
    ax[0, 0].hist(pos[corr_mask], bins=40, alpha=0.6, label="corridor")
    ax[0, 0].hist(pos[run_mask], bins=40, alpha=0.6, label="corridor+running")
    ax[0, 0].set_title("Position Samples Before/After Running Mask")
    ax[0, 0].set_xlabel("Native position units (0.1 m)")
    ax[0, 0].legend()

    # Running speed quartiles.
    speed = np.asarray(beh["ft_RunSpeed"], dtype=np.float32)[run_mask]
    ax[0, 1].hist(speed, bins=60, color="0.4")
    for edge in speed_edges:
        ax[0, 1].axvline(edge, color="tab:red", linestyle="--")
    ax[0, 1].set_title("Running Speed With Quartile Edges")
    ax[0, 1].set_xlabel("ft_RunSpeed")

    # Stimulus counts across trials.
    wall = np.asarray(beh["WallName"])
    stim_names, stim_counts = np.unique(wall, return_counts=True)
    ax[1, 0].bar(range(len(stim_names)), stim_counts, color="tab:blue")
    ax[1, 0].set_xticks(range(len(stim_names)))
    ax[1, 0].set_xticklabels(stim_names, rotation=45, ha="right")
    ax[1, 0].set_title("Trial Counts by Stimulus")

    # Cue positions.
    cue_pos = np.asarray(beh["SoundPos"], dtype=np.float32)
    ax[1, 1].hist(cue_pos[np.isfinite(cue_pos)], bins=30, color="tab:green")
    ax[1, 1].set_title("Cue Position Distribution")
    ax[1, 1].set_xlabel("Position (native units)")

    # Example trial neural heatmap.
    trial_idx = 0
    if session_meta["n_trials_converted"] > 1:
        for i, out_trial in enumerate(output_trials):
            if np.any(out_trial[1] == 1):
                trial_idx = i
                break
    neural_trial = neural_trials[trial_idx]
    nneu_show = min(200, neural_trial.shape[0])
    ax[2, 0].imshow(neural_trial[:nneu_show], aspect="auto", interpolation="nearest", cmap="viridis")
    ax[2, 0].set_title(f"Example Trial Neural Activity (first {nneu_show} rows)")
    ax[2, 0].set_xlabel("Retained frame")
    ax[2, 0].set_ylabel("Neural row")

    # Example inputs.
    input_trial = input_trials[trial_idx]
    for row in range(input_trial.shape[0]):
        ax[2, 1].plot(input_trial[row], label=INPUT_NAMES[row])
    ax[2, 1].set_title("Example Trial Inputs")
    ax[2, 1].set_xlabel("Retained frame")
    ax[2, 1].legend(fontsize=8)

    # Example outputs.
    output_trial = output_trials[trial_idx]
    for row in range(output_trial.shape[0]):
        ax[3, 0].plot(output_trial[row], label=OUTPUT_NAMES[row])
    ax[3, 0].set_title("Example Trial Outputs")
    ax[3, 0].set_xlabel("Retained frame")
    ax[3, 0].legend(fontsize=8)

    # Converted frame counts across trials.
    kept_counts = np.array([frames.size for frames in groups], dtype=np.int32)
    ax[3, 1].plot(kept_counts, color="tab:purple")
    ax[3, 1].set_title("Retained Corridor-Running Frames per Trial")
    ax[3, 1].set_xlabel("Trial")
    ax[3, 1].set_ylabel("Frames kept")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}", flush=True)


def build_dataset(sessions: list[SessionInfo], show_processing: bool) -> dict[str, Any]:
    overall_t0 = time.time()
    print(f"[setup] converting {len(sessions)} sessions", flush=True)

    day_offsets = compute_day_offsets(sessions)
    stim_vocab = gather_stimulus_vocabulary(sessions)
    stim_to_idx = {stim: idx for idx, stim in enumerate(stim_vocab)}
    speed_edges = gather_speed_edges(sessions)

    subjects = sorted({session.subject for session in sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    neural_all = []
    input_all = []
    output_all = []
    subject_idx = []
    brain_region_idx = []
    session_info = []

    plots_remaining = 2 if show_processing else 0

    for idx, session in enumerate(sessions, start=1):
        print(f"[convert] {idx}/{len(sessions)} {session.raw_id}", flush=True)
        neural_trials, input_trials, output_trials, region_idx, meta = convert_session(
            session=session,
            stim_to_idx=stim_to_idx,
            day_offsets=day_offsets,
            speed_edges=speed_edges,
        )

        if len(neural_trials) < 2:
            print(
                f"[convert] skipping {session.raw_id}: only {len(neural_trials)} converted trial(s)",
                flush=True,
            )
            continue

        neural_all.append(neural_trials)
        input_all.append(input_trials)
        output_all.append(output_trials)
        subject_idx.append(subject_to_idx[session.subject])
        brain_region_idx.append(region_idx)
        session_info.append(meta)

        print(
            f"[convert] kept {len(neural_trials)}/{meta['n_trials_original']} trials, "
            f"rows={meta['n_rows_neural']}, elapsed={meta['conversion_seconds']:.2f}s",
            flush=True,
        )

        if plots_remaining > 0:
            beh = get_behavior(session)
            plot_processing_summary(session, meta, beh, neural_trials, input_trials, output_trials, speed_edges)
            plots_remaining -= 1

    dataset = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
        "brain_regions": BRAIN_REGIONS,
        "brain_region_idx": brain_region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": [
            stim_vocab,
            ["no_lick", "lick"],
            ["0-1m", "1-2m", "2-3m", "3-4m"],
            ["q1", "q2", "q3", "q4"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice running through 4 m virtual-reality texture corridors; "
                "decoder predicts stimulus identity, licking, 1 m position bin, and running-speed quartile."
            ),
            "time_bin_size": float(TIME_BIN_MS),
            "source_frame_rate_hz": float(FRAME_RATE_HZ),
            "temporal_alignment_event": "corridor entry / trial start",
            "off_start": 0.0,
            "off_end": None,
            "frame_selection": "native imaging frames within corridor while running (ft_CorrSpc & ft_move > 0)",
            "corridor_length_m": 4.0,
            "gray_space_length_m": 2.0,
            "native_position_unit_m": 0.1,
            "speed_bin_edges": [float(x) for x in speed_edges.tolist()],
            "stimulus_alias_note": "paper prose refers to a rock/brick pair; raw files use rock/wood labels",
            "source_behavior_aliases_collapsed": True,
            "session_info": session_info,
            "conversion_seconds_total": time.time() - overall_t0,
        },
    }

    print(
        f"[done] built dataset with {len(dataset['neural'])} sessions in "
        f"{dataset['metadata']['conversion_seconds_total']:.2f}s",
        flush=True,
    )
    return dataset


def main():
    args = parse_args()
    mode = "sample" if args.sample else "full"

    start = time.time()
    sessions = build_canonical_sessions()
    print(f"[setup] discovered {len(sessions)} canonical sessions", flush=True)

    if mode == "sample":
        sessions = choose_sample_sessions(sessions, nsample=2)
        print(f"[setup] sample mode active: using {[s.raw_id for s in sessions]}", flush=True)

    dataset = build_dataset(sessions, show_processing=args.show_processing)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_gb = args.outpicklefile.stat().st_size / (1024 ** 3)
    print(
        f"[save] wrote {args.outpicklefile} ({size_gb:.3f} GiB) in {time.time() - start:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
