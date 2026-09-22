#!/usr/bin/env python3
"""Convert the Majnik et al. Track2p dataset for motion decoding.

Decisions implemented here follow the supplied paper, data README, and Track2p
code:

* Every dated directory is a session.  The supplied Suite2p files already contain
  only neurons tracked through every day for that animal, in matched row order;
  consequently no additional ROI filter or rematching is applied.
* Neural activity is the repository's ``dF/F0`` representation: fluorescence
  minus the maximin baseline (Gaussian sigma 10 frames and a 60 s min/max
  window).  Track2p's implementation calls this routine with its default
  neuropil coefficient of zero, so this converter does the same.
* Camera motion is interpolated from its recorded timestamps onto the regular
  two-photon clock.  This handles the missing camera frames documented in the
  data README while retaining the complete neural recording.
* As in the paper's decoding analysis, both streams are averaged over 10
  consecutive 30 Hz samples.  The resulting time bin is 1/3 second.
* Sessions are cut into consecutive, non-overlapping 60 s trials.  Motion-energy
  quintile cut points are estimated independently from each complete, aligned,
  downsampled session, as requested by the decoder task.

Run from any directory with, for example::

    python /app/convert_data.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d


FS_HZ = 30.0
AVERAGE_FRAMES = 10
TRIAL_SECONDS = 60
BASELINE_SIGMA_FRAMES = 10.0
BASELINE_WINDOW_SECONDS = 60.0


def baseline_correct_fluorescence(fluorescence: np.ndarray) -> np.ndarray:
    """Reproduce Track2p ``DataManagement.F_processing`` for dF/F0.

    Despite the GUI label, the reference implementation returns F - Flow (it
    does not divide by Flow).  Its call site leaves ``neucoeff`` at the function
    default of 0.0.  Reproducing those details avoids silently substituting a
    different definition of dF/F.
    """
    smoothed = gaussian_filter(
        fluorescence, sigma=(0.0, BASELINE_SIGMA_FRAMES), mode="reflect"
    )
    window = int(BASELINE_WINDOW_SECONDS * FS_HZ)
    baseline = minimum_filter1d(smoothed, size=window, axis=1, mode="reflect")
    baseline = maximum_filter1d(baseline, size=window, axis=1, mode="reflect")
    return fluorescence - baseline


def align_motion_to_imaging(
    motion: np.ndarray, timestamps: np.ndarray, n_imaging_frames: int
) -> tuple[np.ndarray, int]:
    """Interpolate camera samples to the regular 30 Hz imaging-frame clock.

    Timestamp units are irrelevant here: their median step defines one frame.
    Large steps reveal dropped camera frames.  ``np.interp`` also behaves well
    for sessions without drops and avoids extrapolation at the first sample.
    """
    motion = np.asarray(motion, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if motion.ndim != 1 or timestamps.ndim != 1 or len(motion) != len(timestamps):
        raise ValueError("Motion energy and timestamps must be equal-length 1-D arrays")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Camera timestamps must be strictly increasing")

    intervals = np.diff(timestamps)
    frame_step = float(np.median(intervals))
    missing_frames = int(
        np.maximum(np.rint(intervals / frame_step).astype(np.int64) - 1, 0).sum()
    )
    imaging_clock = timestamps[0] + np.arange(n_imaging_frames) * frame_step
    aligned = np.interp(imaging_clock, timestamps, motion)
    return aligned, missing_frames


def average_consecutive(x: np.ndarray, frames: int, axis: int = -1) -> np.ndarray:
    """Average non-overlapping groups of ``frames`` along one axis."""
    x = np.asarray(x)
    axis = axis % x.ndim
    usable = (x.shape[axis] // frames) * frames
    if usable != x.shape[axis]:
        index = [slice(None)] * x.ndim
        index[axis] = slice(0, usable)
        x = x[tuple(index)]
    shape = list(x.shape)
    shape[axis : axis + 1] = [usable // frames, frames]
    return x.reshape(shape).mean(axis=axis + 1)


def discover_sessions(data_root: Path) -> list[Path]:
    sessions = sorted(
        p
        for p in data_root.glob("jm*/20*")
        if (p / "suite2p/plane0/F.npy").is_file()
        and (p / "move_deve/motion_energy_glob.npy").is_file()
    )
    if not sessions:
        raise FileNotFoundError(f"No complete sessions found below {data_root}")
    return sessions


def convert(data_root: Path) -> dict:
    sessions = discover_sessions(data_root)
    subjects = sorted({session.parent.name for session in sessions})
    subject_to_idx = {subject: i for i, subject in enumerate(subjects)}

    neural_sessions: list[list[np.ndarray]] = []
    input_sessions: list[list[np.ndarray]] = []
    output_sessions: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []
    subject_idx: list[int] = []

    trial_frames = int(TRIAL_SECONDS * FS_HZ)
    trial_bins = trial_frames // AVERAGE_FRAMES

    for session_i, session_dir in enumerate(sessions, start=1):
        plane_dir = session_dir / "suite2p/plane0"
        motion_dir = session_dir / "move_deve"

        fluorescence = np.load(plane_dir / "F.npy")
        iscell = np.load(plane_dir / "iscell.npy")
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
        if fluorescence.ndim != 2:
            raise ValueError(f"Expected a neuron x time F array in {session_dir}")
        if len(iscell) != fluorescence.shape[0] or not np.all(iscell[:, 0] == 1):
            raise ValueError(
                f"{session_dir} is not the expected prefiltered Track2p output"
            )
        if not np.isclose(float(ops["fs"]), FS_HZ):
            raise ValueError(f"Unexpected sampling rate {ops['fs']} in {session_dir}")

        n_neurons, n_frames = fluorescence.shape
        if n_frames % trial_frames:
            raise ValueError(
                f"{session_dir} has {n_frames} frames, not an integer number of trials"
            )

        # Process before splitting, so the baseline filter is continuous across
        # artificial trial boundaries, exactly as for the full paper sessions.
        neural_binned = average_consecutive(
            baseline_correct_fluorescence(fluorescence), AVERAGE_FRAMES, axis=1
        ).astype(np.float32, copy=False)

        raw_motion = np.load(motion_dir / "motion_energy_glob.npy")
        timestamps = np.load(motion_dir / "tstamps.npy")
        aligned_motion, n_missing_camera_frames = align_motion_to_imaging(
            raw_motion, timestamps, n_frames
        )
        motion_binned = average_consecutive(aligned_motion, AVERAGE_FRAMES)

        # Equal-percentile categories are session-specific.  searchsorted gives
        # labels 0..4; ties at a cut point consistently enter the upper bin.
        quintile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
        motion_classes = np.searchsorted(
            quintile_edges, motion_binned, side="right"
        ).astype(np.int64)

        n_trials = n_frames // trial_frames
        elapsed_seconds = (
            np.arange(neural_binned.shape[1], dtype=np.float32)
            * (AVERAGE_FRAMES / FS_HZ)
        )
        session_neural: list[np.ndarray] = []
        session_input: list[np.ndarray] = []
        session_output: list[np.ndarray] = []
        for trial in range(n_trials):
            start = trial * trial_bins
            stop = start + trial_bins
            # Slices are views into per-session arrays; pickle preserves the
            # values and the lists satisfy the requested session/trial structure.
            session_neural.append(neural_binned[:, start:stop])
            session_input.append(elapsed_seconds[None, start:stop])
            session_output.append(motion_classes[None, start:stop])

        subject = session_dir.parent.name
        neural_sessions.append(session_neural)
        input_sessions.append(session_input)
        output_sessions.append(session_output)
        subject_idx.append(subject_to_idx[subject])
        brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
        session_info.append(
            {
                "subject": subject,
                "session": session_dir.name,
                "n_neurons": int(n_neurons),
                "n_trials": int(n_trials),
                "source_frames": int(n_frames),
                "camera_samples": int(len(raw_motion)),
                "inferred_missing_camera_frames": n_missing_camera_frames,
                "motion_energy_quintile_edges": quintile_edges.tolist(),
            }
        )
        print(
            f"[{session_i:02d}/{len(sessions)}] {subject}/{session_dir.name}: "
            f"{n_neurons} neurons, {n_trials} trials, "
            f"{n_missing_camera_frames} inferred missing camera frames",
            flush=True,
        )

        # Make peak memory independent of the number of raw sessions.
        del fluorescence, neural_binned, raw_motion, aligned_motion, motion_binned

    return {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["barrel cortex L2/3"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time elapsed from session start (s)"],
        "output_names": ["motion energy quintile"],
        "output_values": [
            ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        ],
        "metadata": {
            "task_description": (
                "Decode session-specific motion-energy quintiles from Track2p-matched "
                "barrel-cortex calcium activity, with elapsed session time as context."
            ),
            "time_bin_size": 1000.0 * AVERAGE_FRAMES / FS_HZ,
            "temporal_alignment_event": "start of each consecutive 60-second trial",
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
            "sampling_rate_hz": FS_HZ,
            "averaged_source_frames_per_bin": AVERAGE_FRAMES,
            "trial_duration_seconds": TRIAL_SECONDS,
            "neural_processing": (
                "Track2p dF/F0 implementation: F minus a maximin baseline "
                "(Gaussian sigma 10 frames; 60 s minimum then maximum filters; "
                "neuropil coefficient 0), followed by non-overlapping 10-frame means."
            ),
            "behavior_processing": (
                "Motion energy linearly interpolated from camera timestamps to the "
                "30 Hz imaging clock, averaged in 10-frame bins, then discretized "
                "by per-session 20th/40th/60th/80th percentiles."
            ),
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
    with args.output.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(data['neural'])} sessions to {args.output}")


if __name__ == "__main__":
    main()
