#!/usr/bin/env python3
"""Convert Zhong et al. imaging data into the decoder training format."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("/app/data")
BEH_DIR = DATA_ROOT / "beh"
SPK_DIR = DATA_ROOT / "spk"
RETINO_DIR = DATA_ROOT / "retinotopy"
EXP_INFO_PATH = BEH_DIR / "Imaging_Exp_info.npy"

NOMINAL_FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / NOMINAL_FRAME_RATE_HZ
CORRIDOR_SPEED_DM_PER_S = 6.0

CANONICAL_STIM_BY_ID = {
    0: "circle1",
    1: "circle2",
    2: "leaf1",
    3: "leaf2",
    4: "leaf3",
    5: "leaf1_swap1",
    6: "leaf1_swap2",
}

OUTPUT_STIMULI = [
    "circle1",
    "circle2",
    "circle3",
    "leaf1",
    "leaf2",
    "leaf3",
    "leaf1_swap1",
    "leaf1_swap2",
]
OUTPUT_STIM_TO_IDX = {name: i for i, name in enumerate(OUTPUT_STIMULI)}

BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV", "other"]


@dataclass
class BehaviorView:
    exp_type: str
    file_name: str
    full_key: str
    beh: dict[str, Any]


@dataclass
class SessionSpec:
    raw_key: str
    subject: str
    date: dt.date
    block: int
    behavior_views: list[BehaviorView]
    behavior_ref: dict[str, Any]
    wall_to_stimulus: dict[str, str]
    training_day: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert imaging data into decoder format.")
    parser.add_argument("outpicklefile", type=str, help="Output pickle file path.")
    parser.add_argument("--full", action="store_true", help="Process all sessions (default behavior).")
    parser.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing diagnostics for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def parse_raw_session_key(full_key: str) -> str:
    parts = full_key.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected session key format: {full_key}")
    return "_".join(parts[:5])


def parse_date_and_block(raw_key: str) -> tuple[dt.date, int]:
    parts = raw_key.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected raw key format: {raw_key}")
    date = dt.date(int(parts[1]), int(parts[2]), int(parts[3]))
    block = int(parts[4])
    return date, block


def load_exp_info() -> dict[str, list[dict[str, Any]]]:
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()


def iter_behavior_files() -> list[Path]:
    files = []
    for path in sorted(BEH_DIR.glob("Beh_*.npy")):
        if path.name in {
            "Beh_no_pretrain.npy",
            "Beh_pretrain_on_grat_image.npy",
            "Beh_pretrain_on_nat_image.npy",
        }:
            continue
        files.append(path)
    return files


def arrays_match(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in {"U", "S", "O"} or b.dtype.kind in {"U", "S", "O"}:
        return np.array_equal(a, b)
    return np.allclose(a, b, equal_nan=True)


def validate_duplicate_behavior_views(raw_key: str, views: list[BehaviorView]) -> None:
    if len(views) < 2:
        return

    ref = views[0].beh
    keys_to_match = [
        "WallName",
        "SoundPos",
        "StartFr",
        "EndFr",
        "isRew",
        "SoundFr",
        "RewardFr",
        "ft_trInd",
        "ft_CorrSpc",
        "ft_Pos",
        "ft_RunSpeed",
    ]
    for view in views[1:]:
        for key in keys_to_match:
            if not arrays_match(np.asarray(ref[key]), np.asarray(view.beh[key])):
                raise ValueError(
                    f"Duplicate behavior views disagree for {raw_key}: "
                    f"{views[0].file_name}/{views[0].full_key} vs {view.file_name}/{view.full_key} on {key}"
                )


def build_wall_to_stimulus_map(raw_key: str, views: list[BehaviorView]) -> dict[str, str]:
    label_map: dict[str, str] = {}

    for view in views:
        uniq_walls = list(map(str, np.asarray(view.beh["UniqWalls"])))
        stim_ids = np.asarray(view.beh["stim_id"])
        for wall, stim_id in zip(uniq_walls, stim_ids):
            if np.isnan(stim_id):
                continue
            canonical = CANONICAL_STIM_BY_ID[int(stim_id)]
            prev = label_map.get(wall)
            if prev is not None and prev != canonical:
                raise ValueError(
                    f"Conflicting stimulus mapping inside raw session {raw_key}: {wall} -> {prev} vs {canonical}"
                )
            label_map[wall] = canonical

    present_walls = set(map(str, np.unique(views[0].beh["WallName"])))
    missing = sorted(present_walls - set(label_map))
    for wall in list(missing):
        if wall in OUTPUT_STIMULI:
            label_map[wall] = wall
    missing = sorted(present_walls - set(label_map))
    if missing:
        raise ValueError(f"Missing label mapping for session {raw_key}: {missing}")

    return label_map


def choose_behavior_reference(views: list[BehaviorView]) -> dict[str, Any]:
    def score(view: BehaviorView) -> tuple[int, str, str]:
        non_nan = int(np.count_nonzero(~np.isnan(np.asarray(view.beh["stim_id"], dtype=float))))
        return (non_nan, view.file_name, view.full_key)

    return max(views, key=score).beh


def build_session_specs() -> list[SessionSpec]:
    exp_info = load_exp_info()
    raw_to_subject: dict[str, str] = {}
    for records in exp_info.values():
        for rec in records:
            raw_key = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
            raw_to_subject[raw_key] = rec["mname"]

    raw_to_views: dict[str, list[BehaviorView]] = defaultdict(list)
    for beh_path in iter_behavior_files():
        exp_type = beh_path.stem.removeprefix("Beh_")
        beh_all = np.load(beh_path, allow_pickle=True).item()
        for full_key, beh in beh_all.items():
            raw_key = parse_raw_session_key(full_key)
            raw_to_views[raw_key].append(
                BehaviorView(exp_type=exp_type, file_name=beh_path.name, full_key=full_key, beh=beh)
            )

    subject_dates: dict[str, list[tuple[dt.date, int]]] = defaultdict(list)
    raw_date_block: dict[str, tuple[dt.date, int]] = {}
    for raw_key in raw_to_views:
        date, block = parse_date_and_block(raw_key)
        raw_date_block[raw_key] = (date, block)
        subject_dates[raw_to_subject[raw_key]].append((date, block))

    subject_first_date = {
        subject: min(date for date, _ in date_blocks)
        for subject, date_blocks in subject_dates.items()
    }

    specs: list[SessionSpec] = []
    for raw_key in sorted(raw_to_views):
        views = raw_to_views[raw_key]
        validate_duplicate_behavior_views(raw_key, views)
        subject = raw_to_subject[raw_key]
        date, block = raw_date_block[raw_key]
        training_day = float((date - subject_first_date[subject]).days) + 0.01 * (block - 1)
        specs.append(
            SessionSpec(
                raw_key=raw_key,
                subject=subject,
                date=date,
                block=block,
                behavior_views=views,
                behavior_ref=choose_behavior_reference(views),
                wall_to_stimulus=build_wall_to_stimulus_map(raw_key, views),
                training_day=training_day,
            )
        )

    return specs


def build_brain_region_idx(iarea: np.ndarray) -> np.ndarray:
    idx = np.full(iarea.shape, 4, dtype=np.int8)
    idx[iarea == 8] = 0
    idx[np.isin(iarea, [0, 1, 2, 9])] = 1
    idx[np.isin(iarea, [5, 6])] = 2
    idx[np.isin(iarea, [3, 4])] = 3
    return idx


def get_trial_frame_indices(beh: dict[str, Any], nframes: int) -> list[np.ndarray]:
    ft_trind = np.asarray(beh["ft_trInd"][:nframes])
    ft_corr = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
    ft_move = np.asarray(beh["ft_move"][:nframes]) > 0
    ntrials = int(beh["ntrials"])
    frame_sets: list[np.ndarray] = []

    for trial in range(ntrials):
        idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
        if idx.size == 0:
            frame_sets.append(idx)
            continue
        frame_sets.append(idx.astype(np.int64, copy=False))

    return frame_sets


def load_spike_planes(raw_key: str) -> list[np.ndarray]:
    path = SPK_DIR / f"{raw_key}_neural_data.npy"
    obj = np.load(path, allow_pickle=True).item()
    return obj["spks"]


def load_retinotopy(raw_key: str) -> np.ndarray:
    mouse_date = "_".join(raw_key.split("_")[:4])
    path = RETINO_DIR / f"{mouse_date}_trans.npz"
    return np.load(path, allow_pickle=True)["iarea"]


def position_to_bins(ft_pos_dm: np.ndarray) -> np.ndarray:
    pos_dm = np.asarray(ft_pos_dm, dtype=np.float32)
    bins = np.floor(pos_dm / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)


def binarize_licks(lick_frames: np.ndarray, trial_lick_mask: np.ndarray, frame_idx: np.ndarray) -> np.ndarray:
    out = np.zeros(frame_idx.size, dtype=np.uint8)
    if frame_idx.size == 0:
        return out

    frames = np.asarray(lick_frames[trial_lick_mask], dtype=np.int64)
    if frames.size == 0:
        return out

    kept = np.intersect1d(frames, frame_idx, assume_unique=False)
    if kept.size == 0:
        return out

    offsets = np.searchsorted(frame_idx, kept)
    valid = (offsets >= 0) & (offsets < frame_idx.size) & (frame_idx[offsets] == kept)
    if np.any(valid):
        out[offsets[valid]] = 1
    return out


def make_processing_plot(
    session_idx: int,
    session_id: str,
    beh: dict[str, Any],
    neural_trials: list[np.ndarray],
    input_trials: list[np.ndarray],
    output_trials: list[np.ndarray],
    speed_trials: list[np.ndarray],
    speed_edges: np.ndarray,
) -> None:
    trial_count = len(neural_trials)
    if trial_count == 0:
        return

    example_trial = min(5, trial_count - 1)
    example_neural = neural_trials[example_trial]
    example_input = input_trials[example_trial]
    example_output = output_trials[example_trial]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    nshow_trials = min(80, int(beh["ntrials"]))
    cue_pos = np.asarray(beh["SoundPos"][:nshow_trials], dtype=float) / 10.0
    rew_pos = np.asarray(beh["RewPos"][:nshow_trials], dtype=float) / 10.0
    is_rew = np.asarray(beh["isRew"][:nshow_trials], dtype=bool)
    axes[0, 0].scatter(cue_pos, np.arange(nshow_trials), s=6, c="purple", label="cue")
    axes[0, 0].scatter(rew_pos[is_rew], np.flatnonzero(is_rew), s=6, c="tab:blue", label="reward")
    axes[0, 0].set_xlim(0, 4)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Corridor position (m)")
    axes[0, 0].set_ylabel("Trial")
    axes[0, 0].set_title("Cue / reward positions")
    axes[0, 0].legend(loc="upper right")

    nneurons_plot = min(64, example_neural.shape[0])
    vmax = np.percentile(example_neural[:nneurons_plot], 99)
    axes[0, 1].imshow(example_neural[:nneurons_plot], aspect="auto", interpolation="nearest", vmax=vmax)
    axes[0, 1].set_title(f"Example neural trial {example_trial}")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Neuron")

    axes[1, 0].plot(example_input[0], label="time_to_sound")
    axes[1, 0].plot(example_input[2], label="time_since_start")
    axes[1, 0].plot(example_input[3], label="reward_available")
    axes[1, 0].set_title("Decoder inputs")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].plot(example_output[1], label="licking")
    axes[1, 1].plot(example_output[2], label="position_bin")
    axes[1, 1].plot(example_output[3], label="speed_bin")
    axes[1, 1].set_title("Decoder outputs")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].legend(loc="upper right")

    all_speeds = np.concatenate(speed_trials) if speed_trials else np.array([])
    if all_speeds.size == 0:
        all_speeds = np.zeros(1)
    axes[2, 0].hist(all_speeds, bins=50, color="0.5")
    for edge in speed_edges[1:-1]:
        axes[2, 0].axvline(edge, color="tab:red", linestyle="--", linewidth=1)
    axes[2, 0].set_title("Running speed distribution in session")
    axes[2, 0].set_xlabel("Running speed")
    axes[2, 0].set_ylabel("Count")

    pos_bins, pos_counts = np.unique(example_output[2], return_counts=True)
    axes[2, 1].bar(pos_bins, pos_counts, color="tab:green")
    axes[2, 1].set_title("Example trial position bins")
    axes[2, 1].set_xlabel("Position bin")
    axes[2, 1].set_ylabel("Frames")

    fig.suptitle(f"Processing diagnostics: session {session_idx} ({session_id})")
    fig.tight_layout()
    fig.savefig(f"processing_{session_id}.png", dpi=150)
    plt.close(fig)


def process_session(spec: SessionSpec, session_idx: int) -> dict[str, Any]:
    t0 = time.time()
    planes = load_spike_planes(spec.raw_key)
    nframes = int(planes[0].shape[1])
    if any(int(arr.shape[1]) != nframes for arr in planes):
        raise ValueError(f"Plane frame-count mismatch in session {spec.raw_key}")

    beh = spec.behavior_ref
    trial_frame_sets = get_trial_frame_indices(beh, nframes)
    iarea = load_retinotopy(spec.raw_key)
    total_neurons = int(sum(arr.shape[0] for arr in planes))
    if total_neurons != int(iarea.shape[0]):
        raise ValueError(
            f"Neuron count mismatch in session {spec.raw_key}: planes sum to {total_neurons}, retinotopy has {iarea.shape[0]}"
        )

    sound_pos = np.asarray(beh["SoundPos"][: int(beh["ntrials"])], dtype=np.float32)
    wall_name = np.asarray(beh["WallName"][: int(beh["ntrials"])], dtype=object)
    is_rew = np.asarray(beh["isRew"][: int(beh["ntrials"])], dtype=np.int8)
    ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float32)
    lick_frames = np.asarray(beh["LickFr"], dtype=int)
    lick_trials = np.asarray(beh["LickTrind"], dtype=int)

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    speed_trials: list[np.ndarray] = []

    for trial, frame_idx in enumerate(trial_frame_sets):
        if frame_idx.size == 0:
            continue

        literal_wall = str(wall_name[trial])
        canonical_wall = spec.wall_to_stimulus[literal_wall]
        stimulus_idx = OUTPUT_STIM_TO_IDX[canonical_wall]

        neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0).astype(np.float16, copy=False)
        trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
        t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
        cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
        reward_trial = np.full(frame_idx.size, float(is_rew[trial]), dtype=np.float16)
        day_trial = np.full(frame_idx.size, spec.training_day, dtype=np.float16)
        speed_trial = ft_speed[frame_idx].astype(np.float32, copy=False)

        input_trial = np.vstack(
            [
                cue_sec.astype(np.float16, copy=False),
                day_trial,
                t_sec.astype(np.float16, copy=False),
                reward_trial,
            ]
        )

        lick_trial_mask = lick_trials == trial
        licking = binarize_licks(lick_frames, lick_trial_mask, frame_idx)
        pos_bins = position_to_bins(ft_pos[frame_idx]).astype(np.uint8, copy=False)
        output_trial = np.vstack(
            [
                np.full(frame_idx.size, stimulus_idx, dtype=np.uint8),
                licking,
                pos_bins,
                np.zeros(frame_idx.size, dtype=np.uint8),  # placeholder for speed bins
            ]
        )

        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)
        speed_trials.append(speed_trial)

    if len(neural_trials) < 2:
        raise ValueError(f"Session {spec.raw_key} has fewer than 2 valid trials after processing")

    elapsed = time.time() - t0
    print(
        f"[{session_idx:03d}] {spec.raw_key}: trials={len(neural_trials)} "
        f"neurons={total_neurons} frames/session={nframes} time={elapsed:.2f}s",
        flush=True,
    )

    return {
        "session_id": spec.raw_key,
        "subject": spec.subject,
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "speed_trials": speed_trials,
        "brain_region_idx": build_brain_region_idx(iarea),
        "wall_to_stimulus": dict(spec.wall_to_stimulus),
        "reward_mode": str(beh["Reward_Mode"]),
        "nframes": nframes,
    }


def apply_speed_bins(processed_sessions: list[dict[str, Any]]) -> np.ndarray:
    all_speeds = np.concatenate(
        [speed for session in processed_sessions for speed in session["speed_trials"]],
        axis=0,
    )
    quantiles = np.quantile(all_speeds, [0.0, 0.25, 0.5, 0.75, 1.0])
    quantiles[0] = min(quantiles[0], np.min(all_speeds))
    quantiles[-1] = max(quantiles[-1], np.max(all_speeds))

    for session in processed_sessions:
        for trial_idx, speed_trial in enumerate(session["speed_trials"]):
            bins = np.digitize(speed_trial, quantiles[1:-1], right=False).astype(np.int16)
            session["output"][trial_idx][3] = bins

    return quantiles.astype(np.float32)


def build_metadata(
    selected_specs: list[SessionSpec],
    processed_sessions: list[dict[str, Any]],
    speed_edges: np.ndarray,
) -> dict[str, Any]:
    return {
        "task_description": (
            "Two-photon mesoscope recordings during visual virtual-reality corridor behavior; "
            "decoder predicts canonical visual stimulus identity, licking, 1 m position bin, and running-speed quartile "
            "from deconvolved neural activity plus task-context inputs."
        ),
        "time_bin_size": TIME_BIN_MS,
        "temporal_alignment_event": "first imaging frame assigned to the current corridor trial (corridor entry)",
        "off_start": 0.0,
        "off_end": None,
        "nominal_frame_rate_hz": NOMINAL_FRAME_RATE_HZ,
        "corridor_length_m": 4.0,
        "gray_space_length_m": 2.0,
        "frame_selection": "frames where ft_trInd == trial, ft_CorrSpc is True, and ft_move > 0",
        "speed_bin_edges": speed_edges.tolist(),
        "visual_stimulus_values": OUTPUT_STIMULI,
        "session_ids": [spec.raw_key for spec in selected_specs],
        "session_label_maps": {
            spec.raw_key: dict(processed_sessions[i]["wall_to_stimulus"])
            for i, spec in enumerate(selected_specs)
        },
        "reward_modes_present": sorted({session["reward_mode"] for session in processed_sessions}),
    }


def save_processing_plots(
    selected_specs: list[SessionSpec],
    processed_sessions: list[dict[str, Any]],
    speed_edges: np.ndarray,
) -> None:
    for session_idx, (spec, session) in enumerate(zip(selected_specs[:2], processed_sessions[:2])):
        make_processing_plot(
            session_idx=session_idx,
            session_id=spec.raw_key,
            beh=spec.behavior_ref,
            neural_trials=session["neural"],
            input_trials=session["input"],
            output_trials=session["output"],
            speed_trials=session["speed_trials"],
            speed_edges=speed_edges,
        )


def count_trials_with_corridor_licks(beh: dict[str, Any]) -> int:
    frame_sets = get_trial_frame_indices(beh, len(beh["ft"]))
    lick_frames = np.asarray(beh["LickFr"], dtype=int)
    lick_trials = np.asarray(beh["LickTrind"], dtype=int)
    lick_any = 0
    for trial, frame_idx in enumerate(frame_sets):
        if frame_idx.size == 0:
            continue
        mask = (lick_trials == trial) & np.isin(lick_frames, frame_idx, assume_unique=False)
        if np.any(mask):
            lick_any += 1
    return lick_any


def get_session_neuron_count(raw_key: str) -> int:
    return int(load_retinotopy(raw_key).shape[0])


def select_sample_specs(specs: list[SessionSpec], nsample: int = 2) -> list[SessionSpec]:
    primary = []
    secondary = []
    for spec in specs:
        beh = spec.behavior_ref
        ntrials = int(beh["ntrials"])
        reward_states = int(np.unique(np.asarray(beh["isRew"][:ntrials], dtype=int)).size)
        lick_any = count_trials_with_corridor_licks(beh)
        nstim = int(len(np.unique(beh["WallName"])))
        nneurons = get_session_neuron_count(spec.raw_key)
        cost = nneurons * ntrials
        item = (
            0 if nstim >= 4 else 1,
            cost,
            -lick_any,
            spec.raw_key,
        )
        if reward_states > 1 and lick_any > 0:
            primary.append((item, spec))
        else:
            secondary.append((item, spec))

    primary.sort(key=lambda item: item[0])
    secondary.sort(key=lambda item: item[0])

    selected: list[SessionSpec] = []
    used_subjects: set[str] = set()

    for _, spec in primary:
        if spec.subject in used_subjects:
            continue
        selected.append(spec)
        used_subjects.add(spec.subject)
        if len(selected) == nsample:
            return selected

    if len(selected) < nsample:
        for _, spec in primary:
            if spec in selected:
                continue
            selected.append(spec)
            if len(selected) == nsample:
                return selected

    if len(selected) < nsample:
        for _, spec in secondary:
            if spec in selected:
                continue
            selected.append(spec)
            if len(selected) == nsample:
                return selected

    return selected


def process_all_sessions(selected_specs: list[SessionSpec], max_workers: int) -> list[dict[str, Any]]:
    if max_workers <= 1:
        return [process_session(spec, i) for i, spec in enumerate(selected_specs)]

    results: list[dict[str, Any] | None] = [None] * len(selected_specs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_session, spec, i) for i, spec in enumerate(selected_specs)]
        for i, future in enumerate(futures):
            results[i] = future.result()
    return [result for result in results if result is not None]


def main() -> None:
    args = parse_args()
    out_path = Path(args.outpicklefile)

    t0 = time.time()
    specs = build_session_specs()
    selected_specs = select_sample_specs(specs, nsample=2) if args.sample else specs
    print(f"Selected {len(selected_specs)} of {len(specs)} sessions", flush=True)
    max_workers = 1 if args.sample else min(4, os.cpu_count() or 1)
    print(f"Processing with {max_workers} worker(s)", flush=True)

    processed_sessions = process_all_sessions(selected_specs, max_workers=max_workers)

    speed_edges = apply_speed_bins(processed_sessions)

    if args.show_processing:
        save_processing_plots(selected_specs, processed_sessions, speed_edges)

    subjects = sorted({spec.subject for spec in selected_specs})
    subject_to_idx = {subject: i for i, subject in enumerate(subjects)}

    data = {
        "neural": [session["neural"] for session in processed_sessions],
        "input": [session["input"] for session in processed_sessions],
        "output": [session["output"] for session in processed_sessions],
        "subjects": subjects,
        "subject_idx": np.array([subject_to_idx[spec.subject] for spec in selected_specs], dtype=np.int8),
        "brain_regions": BRAIN_REGIONS,
        "brain_region_idx": [session["brain_region_idx"] for session in processed_sessions],
        "input_names": [
            "time_to_sound_cue_s",
            "training_day",
            "time_since_trial_start_s",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus",
            "licking",
            "position_bin",
            "speed_bin",
        ],
        "output_values": [
            OUTPUT_STIMULI,
            ["no_lick", "lick"],
            ["0-1m", "1-2m", "2-3m", "3-4m"],
            [
                f"q1_[{speed_edges[0]:.3f},{speed_edges[1]:.3f})",
                f"q2_[{speed_edges[1]:.3f},{speed_edges[2]:.3f})",
                f"q3_[{speed_edges[2]:.3f},{speed_edges[3]:.3f})",
                f"q4_[{speed_edges[3]:.3f},{speed_edges[4]:.3f}]",
            ],
        ],
        "metadata": build_metadata(selected_specs, processed_sessions, speed_edges),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_trials = sum(len(session["neural"]) for session in processed_sessions)
    total_neurons = sum(session["brain_region_idx"].shape[0] for session in processed_sessions)
    elapsed = time.time() - t0
    print(
        f"Saved {out_path} with {len(selected_specs)} sessions, {total_trials} trials, "
        f"{total_neurons} neurons in {elapsed / 60:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
