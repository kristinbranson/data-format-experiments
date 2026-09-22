from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from suite2p.extraction.dcnv import preprocess


DATA_ROOT = Path("/app/data")
DEFAULT_OUTPUT = Path("/app/converted_data.pkl")

FRAME_RATE_HZ = 30.0
DENOISE_BIN_FRAMES = 10
TRIAL_SECONDS = 60
MOTION_NBINS = 5


def sorted_subjects(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir())


def sorted_sessions(subject_dir: Path) -> list[Path]:
    return sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()
    )


def reconstruct_motion_energy(session_dir: Path, target_len: int) -> np.ndarray:
    motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy").astype(
        np.float32
    )
    if motion.shape[0] == target_len:
        return motion

    interframe = np.load(session_dir / "move_deve" / "interframe_int.npy")
    if interframe.shape[0] != motion.shape[0] - 1:
        raise ValueError(
            f"{session_dir}: unexpected interframe length {interframe.shape[0]} "
            f"for motion length {motion.shape[0]}"
        )

    median_dt = float(np.median(interframe))
    if median_dt <= 0:
        raise ValueError(f"{session_dir}: non-positive median interframe interval")

    frame_steps = np.rint(interframe / median_dt).astype(np.int64)
    frame_steps = np.maximum(frame_steps, 1)
    known_positions = np.concatenate([[0], np.cumsum(frame_steps)])

    if known_positions.shape[0] != motion.shape[0]:
        raise ValueError(f"{session_dir}: failed to map motion samples to frame indices")
    if int(known_positions[-1]) != target_len - 1:
        raise ValueError(
            f"{session_dir}: reconstructed motion spans {int(known_positions[-1]) + 1} "
            f"frames but neural data has {target_len}"
        )

    full_motion = np.full(target_len, np.nan, dtype=np.float32)
    full_motion[known_positions] = motion
    valid = np.flatnonzero(~np.isnan(full_motion))
    return np.interp(
        np.arange(target_len, dtype=np.float64), valid, full_motion[valid]
    ).astype(np.float32)


def suite2p_dff(session_dir: Path) -> np.ndarray:
    plane_dir = session_dir / "suite2p" / "plane0"
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(plane_dir / "iscell.npy")
    if not np.all(iscell[:, 1] > 0.5):
        raise ValueError(f"{session_dir}: found tracked ROIs below the 0.5 iscell threshold")

    f = np.load(plane_dir / "F.npy").astype(np.float32)
    fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32)

    # The paper states that downstream analyses used Suite2p baseline-corrected
    # fluorescence traces with default parameters. Suite2p performs this on
    # neuropil-subtracted fluorescence before deconvolution.
    fc = f - float(ops.get("neucoeff", 0.7)) * fneu

    dff = preprocess(
        fc.copy(),
        baseline=ops.get("baseline", "maximin"),
        win_baseline=float(ops.get("win_baseline", 60.0)),
        sig_baseline=float(ops.get("sig_baseline", 10.0)),
        fs=float(ops.get("fs", FRAME_RATE_HZ)),
        prctile_baseline=float(ops.get("prctile_baseline", 8)),
        batch_size=100,
        device=torch.device("cpu"),
    )
    return dff.astype(np.float32, copy=False)


def mean_bin_1d(x: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (x.shape[0] // bin_size) * bin_size
    if usable == 0:
        raise ValueError("cannot bin an empty array")
    x = x[:usable]
    return x.reshape(-1, bin_size).mean(axis=1)


def mean_bin_2d(x: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (x.shape[1] // bin_size) * bin_size
    if usable == 0:
        raise ValueError("cannot bin an empty array")
    x = x[:, :usable]
    return x.reshape(x.shape[0], -1, bin_size).mean(axis=2)


def discretize_equal_percentile(values: np.ndarray, nbins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, nbins + 1)[1:-1])
    if np.unique(edges).shape[0] == edges.shape[0]:
        return np.digitize(values, edges, right=False).astype(np.int64)

    # Fallback that still guarantees exactly nbins equal-frequency classes.
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(values.shape[0], dtype=np.int64)
    for bin_idx, idx in enumerate(np.array_split(order, nbins)):
        bins[idx] = bin_idx
    return bins


def session_to_trials(session_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    neural = suite2p_dff(session_dir)
    raw_frames = neural.shape[1]
    motion = reconstruct_motion_energy(session_dir, raw_frames)

    neural_binned = mean_bin_2d(neural, DENOISE_BIN_FRAMES)
    motion_binned = mean_bin_1d(motion, DENOISE_BIN_FRAMES).astype(np.float32)

    frame_times_s = np.arange(raw_frames, dtype=np.float32) / FRAME_RATE_HZ
    time_binned = mean_bin_1d(frame_times_s, DENOISE_BIN_FRAMES).astype(np.float32)
    motion_bins = discretize_equal_percentile(motion_binned, MOTION_NBINS)

    trial_bins = int(TRIAL_SECONDS * FRAME_RATE_HZ / DENOISE_BIN_FRAMES)
    n_complete_trials = neural_binned.shape[1] // trial_bins
    if n_complete_trials < 2:
        raise ValueError(f"{session_dir}: fewer than two complete 60-second trials")

    usable = n_complete_trials * trial_bins
    neural_binned = neural_binned[:, :usable]
    motion_bins = motion_bins[:usable]
    time_binned = time_binned[:usable]

    neural_trials = []
    input_trials = []
    output_trials = []
    for trial_idx in range(n_complete_trials):
        start = trial_idx * trial_bins
        end = start + trial_bins
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(time_binned[np.newaxis, start:end].astype(np.float32, copy=False))
        output_trials.append(motion_bins[np.newaxis, start:end].astype(np.int64, copy=False))

    return neural_trials, input_trials, output_trials, motion_bins


def build_dataset(data_root: Path) -> dict:
    subjects = [subject_dir.name for subject_dir in sorted_subjects(data_root)]
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}

    neural_all: list[list[np.ndarray]] = []
    input_all: list[list[np.ndarray]] = []
    output_all: list[list[np.ndarray]] = []
    subject_idx = []
    brain_region_idx = []
    session_ids = []
    trial_counts = []

    for subject_dir in sorted_subjects(data_root):
        for session_dir in sorted_sessions(subject_dir):
            print(f"Converting {subject_dir.name}/{session_dir.name}...")
            neural_trials, input_trials, output_trials, _ = session_to_trials(session_dir)
            neural_all.append(neural_trials)
            input_all.append(input_trials)
            output_all.append(output_trials)
            subject_idx.append(subject_lookup[subject_dir.name])
            brain_region_idx.append(
                np.zeros(neural_trials[0].shape[0], dtype=np.int64)
            )
            session_ids.append(f"{subject_dir.name}/{session_dir.name}")
            trial_counts.append(len(neural_trials))

    dataset = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["barrel cortex L2/3"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_session_start_s"],
        "output_names": ["motion_energy_quintile"],
        "output_values": [[
            "lowest_20pct",
            "20_40pct",
            "40_60pct",
            "60_80pct",
            "highest_20pct",
        ]],
        "metadata": {
            "task_description": (
                "Decode 5-bin motion-energy state from longitudinal barrel-cortex "
                "population activity during spontaneous behavior."
            ),
            "time_bin_size": 1000.0 * DENOISE_BIN_FRAMES / FRAME_RATE_HZ,
            "temporal_alignment_event": "start of each consecutive 60-second trial",
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
            "input_time_reference": (
                "Time input is elapsed seconds from session start at each 10-frame "
                "averaged bin."
            ),
            "neural_trace_processing": (
                "Track2p-tracked Suite2p F/Fneu traces with iscell > 0.5, "
                "neuropil subtraction using ops['neucoeff'], Suite2p maximin "
                "baseline correction, then non-overlapping 10-frame averaging."
            ),
            "behavior_processing": (
                "Motion energy aligned to neural frames; missing camera frames were "
                "reinserted from interframe intervals and linearly interpolated "
                "before 10-frame averaging and per-session quintile binning."
            ),
            "trial_definition": "Consecutive non-overlapping 60-second windows.",
            "session_ids": session_ids,
            "trial_counts": trial_counts,
            "source_data_root": str(data_root),
        },
    }
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = build_dataset(args.data_root)

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved converted dataset to {args.output}")
    print(f"Sessions: {len(dataset['neural'])}")
    print(f"Trials per session: {[len(x) for x in dataset['neural']]}")


if __name__ == "__main__":
    main()
