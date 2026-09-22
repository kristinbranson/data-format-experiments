from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from suite2p.extraction import dcnv


DATA_ROOT = Path("/app/data")
OUTPUT_PATH = Path("/app/converted_data.pkl")
BIN_SIZE_FRAMES = 10
TRIAL_DURATION_SEC = 60.0


def list_subjects(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.iterdir() if path.is_dir())


def list_sessions(subject_dir: Path) -> list[Path]:
    return sorted(path for path in subject_dir.iterdir() if path.is_dir())


def repair_motion_energy(session_dir: Path, n_frames: int) -> tuple[np.ndarray, int]:
    move_dir = session_dir / "move_deve"
    motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float32, copy=False)

    if len(motion) == n_frames:
        return motion, 0

    tstamps = np.load(move_dir / "tstamps.npy")
    if len(tstamps) != len(motion):
        raise ValueError(
            f"{session_dir}: motion timestamps length {len(tstamps)} does not match "
            f"motion length {len(motion)}"
        )

    diffs = np.diff(tstamps)
    frame_dt = float(np.median(diffs))
    missing_counts = np.maximum(np.round(diffs / frame_dt).astype(int) - 1, 0)
    expected_len = int(len(motion) + missing_counts.sum())
    if expected_len != n_frames:
        raise ValueError(
            f"{session_dir}: repaired motion length would be {expected_len}, "
            f"but neural data has {n_frames} frames"
        )

    repaired = np.full(n_frames, np.nan, dtype=np.float32)
    positions = np.arange(len(motion), dtype=np.int64)
    positions[1:] += np.cumsum(missing_counts)
    repaired[positions] = motion

    missing_mask = ~np.isfinite(repaired)
    valid_idx = np.flatnonzero(~missing_mask)
    repaired[missing_mask] = np.interp(
        np.flatnonzero(missing_mask),
        valid_idx,
        repaired[valid_idx],
    ).astype(np.float32)

    return repaired, int(missing_mask.sum())


def preprocess_neural(session_dir: Path) -> tuple[np.ndarray, float]:
    plane_dir = session_dir / "suite2p" / "plane0"
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    F = np.load(plane_dir / "F.npy").astype(np.float32, copy=False)
    Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32, copy=False)

    fs = float(ops["fs"])
    neuropil_corrected = F - float(ops.get("neucoeff", 0.7)) * Fneu
    baseline_corrected = dcnv.preprocess(
        neuropil_corrected.copy(),
        baseline=ops.get("baseline", "maximin"),
        win_baseline=float(ops.get("win_baseline", 60.0)),
        sig_baseline=float(ops.get("sig_baseline", 10.0)),
        fs=fs,
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=64,
        device=torch.device("cpu"),
    )
    return baseline_corrected.astype(np.float32, copy=False), fs


def mean_bin_2d(array: np.ndarray, bin_size: int) -> np.ndarray:
    n_rows, n_frames = array.shape
    usable = (n_frames // bin_size) * bin_size
    if usable != n_frames:
        array = array[:, :usable]
    return array.reshape(n_rows, -1, bin_size).mean(axis=2, dtype=np.float32)


def mean_bin_1d(array: np.ndarray, bin_size: int) -> np.ndarray:
    usable = (len(array) // bin_size) * bin_size
    if usable != len(array):
        array = array[:usable]
    return array.reshape(-1, bin_size).mean(axis=1, dtype=np.float32)


def session_motion_bins(motion_binned: np.ndarray) -> np.ndarray:
    thresholds = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(motion_binned, thresholds, right=False).astype(np.int64)


def split_into_trials(
    neural_binned: np.ndarray,
    time_binned_sec: np.ndarray,
    motion_classes: np.ndarray,
    bins_per_trial: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    total_bins = neural_binned.shape[1]
    n_trials = total_bins // bins_per_trial
    usable = n_trials * bins_per_trial

    neural_binned = neural_binned[:, :usable]
    time_binned_sec = time_binned_sec[:usable]
    motion_classes = motion_classes[:usable]

    neural_trials = []
    input_trials = []
    output_trials = []

    for trial_idx in range(n_trials):
        start = trial_idx * bins_per_trial
        end = start + bins_per_trial
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(time_binned_sec[start:end][None, :].astype(np.float32, copy=False))
        output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))

    return neural_trials, input_trials, output_trials


def build_dataset(data_root: Path = DATA_ROOT) -> dict:
    subjects = [subject_dir.name for subject_dir in list_subjects(data_root)]
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}

    neural_sessions: list[list[np.ndarray]] = []
    input_sessions: list[list[np.ndarray]] = []
    output_sessions: list[list[np.ndarray]] = []
    subject_idx = []
    brain_region_idx = []
    session_info = []

    brain_regions = ["barrel cortex L2/3"]

    common_fs = None
    for subject_dir in list_subjects(data_root):
        for session_dir in list_sessions(subject_dir):
            neural_trace, fs = preprocess_neural(session_dir)
            if common_fs is None:
                common_fs = fs
            elif not np.isclose(common_fs, fs):
                raise ValueError(f"Sampling rate mismatch: expected {common_fs}, got {fs}")

            n_neurons, n_frames = neural_trace.shape
            motion_trace, n_motion_repaired = repair_motion_energy(session_dir, n_frames)

            time_sec = np.arange(n_frames, dtype=np.float32) / np.float32(fs)
            neural_binned = mean_bin_2d(neural_trace, BIN_SIZE_FRAMES)
            motion_binned = mean_bin_1d(motion_trace, BIN_SIZE_FRAMES)
            time_binned_sec = mean_bin_1d(time_sec, BIN_SIZE_FRAMES)
            motion_classes = session_motion_bins(motion_binned)

            bins_per_trial = int(round(TRIAL_DURATION_SEC / (BIN_SIZE_FRAMES / fs)))
            neural_trials, input_trials, output_trials = split_into_trials(
                neural_binned,
                time_binned_sec,
                motion_classes,
                bins_per_trial,
            )

            neural_sessions.append(neural_trials)
            input_sessions.append(input_trials)
            output_sessions.append(output_trials)
            subject_idx.append(subject_to_idx[subject_dir.name])
            brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))

            session_info.append(
                {
                    "subject": subject_dir.name,
                    "session": session_dir.name,
                    "n_neurons": int(n_neurons),
                    "n_frames_raw": int(n_frames),
                    "n_motion_frames_repaired": int(n_motion_repaired),
                    "n_time_bins": int(neural_binned.shape[1]),
                    "n_trials": int(len(neural_trials)),
                    "duration_sec": float(n_frames / fs),
                }
            )

    time_bin_size_ms = 1000.0 * BIN_SIZE_FRAMES / float(common_fs)

    return {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": brain_regions,
        "brain_region_idx": brain_region_idx,
        "input_names": ["time_from_session_start_sec"],
        "output_names": ["motion_energy_bin"],
        "output_values": [["lowest", "low", "middle", "high", "highest"]],
        "metadata": {
            "task_description": (
                "Decode per-time-bin spontaneous motion energy quintiles from "
                "baseline-corrected barrel-cortex calcium activity."
            ),
            "time_bin_size": float(time_bin_size_ms),
            "temporal_alignment_event": "session start",
            "off_start": 0.0,
            "off_end": float(TRIAL_DURATION_SEC),
            "bin_size_frames": int(BIN_SIZE_FRAMES),
            "trial_duration_sec": float(TRIAL_DURATION_SEC),
            "neural_trace_description": (
                "Suite2p baseline-corrected fluorescence after neuropil subtraction "
                "using the saved ops parameters."
            ),
            "behavior_description": (
                "Global motion energy repaired for missing camera frames by timestamp-based "
                "NaN insertion and linear interpolation when needed, then averaged in "
                "10-frame bins and discretized into session-wise quintiles."
            ),
            "session_info": session_info,
        },
    }


def main() -> None:
    dataset = build_dataset()
    with OUTPUT_PATH.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved converted dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
