#!/usr/bin/env python3
"""Convert the Majnik et al. Track2p release for motion decoding.

Decisions tied to the reference material
-----------------------------------------
* Every dated directory is a session.  The supplied Suite2p files contain only
  cells classified by Suite2p and successfully tracked across every day for a
  subject, so no second cell filter is applied.
* Neural activity is the baseline-corrected F trace used by the paper.  This is
  reproduced from ``track2p/gui/data_management.py::F_processing``: a 10-frame
  Gaussian followed by 60-second rolling minimum/maximum filters.  Track2p's
  dF/F0 call uses its function default ``neucoeff=0.0``.
* The paper averages neural and motion traces in non-overlapping 10-frame bins.
  At 30 Hz this gives 333.333 ms samples and exactly 180 samples per 60-second
  trial.
* Camera frames missing from a handful of recordings are located from doubled
  timestamp intervals and linearly interpolated, as suggested by data/README.
* Motion quintile thresholds are computed separately for each complete session
  after alignment and 10-frame averaging, as requested by the decoder task.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d


FRAME_RATE_HZ = 30.0
AVERAGE_FRAMES = 10
TRIAL_SECONDS = 60
TRIAL_FRAMES = int(FRAME_RATE_HZ * TRIAL_SECONDS)
TRIAL_BINS = TRIAL_FRAMES // AVERAGE_FRAMES


def align_motion_to_imaging(
    motion: np.ndarray, timestamps: np.ndarray, n_imaging_frames: int
) -> np.ndarray:
    """Place camera motion samples on imaging-frame indices.

    Normal timestamp intervals count as one frame and doubled intervals locate
    dropped camera frames.  The release's short streams infer exactly the number
    of imaging frames this way.  Full-length streams are already one-to-one and
    are left untouched; occasional long intervals in those streams do not imply
    an absent array element.
    """
    motion = np.asarray(motion, dtype=np.float32).reshape(-1)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if motion.size != timestamps.size:
        raise ValueError("motion_energy_glob and tstamps lengths differ")
    if motion.size == n_imaging_frames:
        return motion
    if motion.size < 2 or motion.size > n_imaging_frames:
        raise ValueError(
            f"cannot align {motion.size} motion samples to {n_imaging_frames} frames"
        )

    intervals = np.diff(timestamps)
    typical_interval = np.median(intervals)
    if not np.isfinite(typical_interval) or typical_interval <= 0:
        raise ValueError("invalid camera timestamps")
    frame_steps = np.maximum(1, np.rint(intervals / typical_interval).astype(np.int64))
    camera_frame_idx = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(frame_steps))
    )
    inferred_frames = int(camera_frame_idx[-1] + 1)
    if inferred_frames != n_imaging_frames:
        raise ValueError(
            "timestamp gaps imply "
            f"{inferred_frames} frames, expected {n_imaging_frames}"
        )

    imaging_frame_idx = np.arange(n_imaging_frames, dtype=np.float64)
    return np.interp(imaging_frame_idx, camera_frame_idx, motion).astype(np.float32)


def baseline_correct_fluorescence(F: np.ndarray, fs: float) -> np.ndarray:
    """Reproduce Track2p's Suite2p-style default maximin correction."""
    corrected = gaussian_filter1d(
        np.asarray(F, dtype=np.float32), sigma=10.0, axis=1
    )
    window = int(60.0 * fs)
    corrected = minimum_filter1d(corrected, size=window, axis=1)
    corrected = maximum_filter1d(corrected, size=window, axis=1)
    # Subtract in a fresh float32 array so the raw memory-mapped F is unchanged.
    return np.subtract(F, corrected, dtype=np.float32)


def average_in_bins(values: np.ndarray, bin_size: int = AVERAGE_FRAMES) -> np.ndarray:
    """Average the last axis in consecutive, non-overlapping bins."""
    usable = values.shape[-1] - values.shape[-1] % bin_size
    values = values[..., :usable]
    new_shape = values.shape[:-1] + (usable // bin_size, bin_size)
    return values.reshape(new_shape).mean(axis=-1, dtype=np.float32)


def quintile_labels(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discretize one session into five session-specific percentile bins."""
    thresholds = np.percentile(motion, [20, 40, 60, 80])
    labels = np.digitize(motion, thresholds, right=False).astype(np.int64)
    return labels, thresholds


def session_directories(data_root: Path) -> list[tuple[str, Path]]:
    sessions: list[tuple[str, Path]] = []
    for subject_dir in sorted(data_root.glob("jm*")):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.glob("*_a")):
            plane_dir = session_dir / "suite2p" / "plane0"
            motion_dir = session_dir / "move_deve"
            required = [
                plane_dir / "F.npy",
                plane_dir / "iscell.npy",
                plane_dir / "ops.npy",
                motion_dir / "motion_energy_glob.npy",
                motion_dir / "tstamps.npy",
            ]
            if not all(path.is_file() for path in required):
                raise FileNotFoundError(f"incomplete session: {session_dir}")
            sessions.append((subject_dir.name, session_dir))
    if not sessions:
        raise FileNotFoundError(f"no sessions found under {data_root}")
    return sessions


def convert(data_root: Path) -> dict:
    session_paths = session_directories(data_root)
    subjects = sorted({subject for subject, _ in session_paths})
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}

    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    output: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for session_number, (subject, session_dir) in enumerate(session_paths, start=1):
        plane_dir = session_dir / "suite2p" / "plane0"
        motion_dir = session_dir / "move_deve"
        F = np.load(plane_dir / "F.npy", mmap_mode="r")
        iscell = np.load(plane_dir / "iscell.npy")
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
        fs = float(ops.get("fs", FRAME_RATE_HZ))
        if not np.isclose(fs, FRAME_RATE_HZ):
            raise ValueError(f"unexpected sampling rate {fs} in {session_dir}")
        if F.ndim != 2 or iscell.shape != (F.shape[0], 2):
            raise ValueError(f"unexpected Suite2p shapes in {session_dir}")
        if not np.all((iscell[:, 0] == 1) & (iscell[:, 1] > 0.5)):
            raise ValueError(
                f"{session_dir} contains an ROI outside the paper's cell criterion"
            )
        if F.shape[1] % TRIAL_FRAMES:
            raise ValueError(f"session is not a whole number of 60-second trials: {session_dir}")

        motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
        timestamps = np.load(motion_dir / "tstamps.npy")
        motion_aligned = align_motion_to_imaging(motion_raw, timestamps, F.shape[1])

        neural_binned = average_in_bins(baseline_correct_fluorescence(F, fs))
        motion_binned = average_in_bins(motion_aligned)
        if neural_binned.shape[1] != motion_binned.size:
            raise RuntimeError(f"aligned streams still differ in {session_dir}")
        labels, thresholds = quintile_labels(motion_binned)

        n_trials = F.shape[1] // TRIAL_FRAMES
        session_neural: list[np.ndarray] = []
        session_input: list[np.ndarray] = []
        session_output: list[np.ndarray] = []
        # Mean times of the ten original frames represented by each sample.
        elapsed_seconds = (
            np.arange(neural_binned.shape[1], dtype=np.float32) * AVERAGE_FRAMES
            + (AVERAGE_FRAMES - 1) / 2
        ) / np.float32(fs)

        for trial in range(n_trials):
            start = trial * TRIAL_BINS
            stop = start + TRIAL_BINS
            session_neural.append(
                np.ascontiguousarray(neural_binned[:, start:stop], dtype=np.float32)
            )
            session_input.append(
                np.ascontiguousarray(elapsed_seconds[None, start:stop], dtype=np.float32)
            )
            session_output.append(
                np.ascontiguousarray(labels[None, start:stop], dtype=np.int64)
            )

        neural.append(session_neural)
        decoder_input.append(session_input)
        output.append(session_output)
        subject_idx.append(subject_lookup[subject])
        brain_region_idx.append(np.zeros(F.shape[0], dtype=np.int64))
        session_info.append(
            {
                "subject": subject,
                "session": session_dir.name,
                "n_neurons": int(F.shape[0]),
                "n_trials": n_trials,
                "n_imaging_frames": int(F.shape[1]),
                "n_camera_frames": int(motion_raw.size),
                "motion_quintile_thresholds": thresholds.tolist(),
            }
        )
        print(
            f"[{session_number:02d}/{len(session_paths)}] {subject}/{session_dir.name}: "
            f"{F.shape[0]} neurons, {n_trials} trials"
        )

    return {
        "neural": neural,
        "input": decoder_input,
        "output": output,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["barrel cortex L2/3"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["elapsed session time (s)"],
        "output_names": ["motion energy quintile"],
        "output_values": [[
            "lowest 0-20%",
            "low 20-40%",
            "middle 40-60%",
            "high 60-80%",
            "highest 80-100%",
        ]],
        "metadata": {
            "task_description": (
                "Decode session-specific motion-energy quintile from layer 2/3 "
                "barrel-cortex calcium activity during spontaneous behavior."
            ),
            "time_bin_size": 1000.0 * AVERAGE_FRAMES / FRAME_RATE_HZ,
            "temporal_alignment_event": "start of each contiguous 60-second session block",
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
            "source": "Majnik et al. (2025), Track2p longitudinal dataset",
            "neural_signal": (
                "Track2p-matched Suite2p F, maximin baseline corrected and averaged "
                "over 10 frames"
            ),
            "motion_processing": (
                "Missing camera frames linearly interpolated from timestamp gaps; "
                "motion energy averaged over 10 frames and discretized by per-session "
                "20th/40th/60th/80th percentiles"
            ),
            "original_frame_rate_hz": FRAME_RATE_HZ,
            "trial_duration_seconds": TRIAL_SECONDS,
            "session_info": session_info,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/app/data"))
    parser.add_argument("--output", type=Path, default=Path("/app/converted_data.pkl"))
    args = parser.parse_args()

    data = convert(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(data['neural'])} sessions to {args.output}")


if __name__ == "__main__":
    main()
