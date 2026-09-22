#!/usr/bin/env python3
"""Convert the Track2p longitudinal barrel-cortex data for decoding.

Processing decisions follow Majnik et al. and the supplied Track2p code:

* sessions and subjects are sorted chronologically/lexicographically;
* ROIs are retained at the Track2p/Suite2p probability threshold > 0.5;
* fluorescence is baseline-corrected with ``DataManagement.F_processing``'s
  maximin implementation and defaults (including ``neucoeff=0.0``);
* neural and motion traces are averaged in non-overlapping 10-frame bins;
* synchronized camera frames missing from a few sessions are interpolated at
  the trigger positions recovered from the supplied camera timestamps;
* averaged motion energy is split by within-session 20/40/60/80 percentiles;
* the continuous recordings are split into non-overlapping 60-second trials.

The source arrays have already been reduced to the Track2p populations that
are present on every day of a subject: each subject has the same ROI count and
cell order on all of its sessions.  No additional cross-day matching is needed.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d


ISCELL_THRESHOLD = 0.5
SOURCE_FRAMES_PER_BIN = 10
TRIAL_SECONDS = 60
EXPECTED_SOURCE_FS = 30.0
MOTION_PERCENTILES = (20.0, 40.0, 60.0, 80.0)


def find_sessions(data_dir: Path) -> list[Path]:
    """Return source session directories in stable subject/date order."""
    sessions = sorted(
        path
        for path in data_dir.glob("*/20*_*")
        if (path / "suite2p" / "plane0" / "F.npy").is_file()
        and (path / "move_deve" / "motion_energy_glob.npy").is_file()
    )
    if not sessions:
        raise FileNotFoundError(f"No complete sessions found below {data_dir}")
    return sessions


def baseline_correct_fluorescence(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Reproduce Track2p GUI ``F_processing`` with its decoding defaults.

    Despite the GUI label ``dF/F0``, the supplied function returns ``Fc-Flow``
    (baseline-subtracted fluorescence), without division by Flow.  Its default
    neuropil coefficient is explicitly zero, so this intentionally differs
    from applying Suite2p's ops-level ``neucoeff`` value.
    """
    neucoeff = 0.0
    sigma_frames = 10.0
    baseline_window_seconds = 60.0

    corrected = fluorescence - neucoeff * neuropil
    baseline = gaussian_filter(corrected, [0.0, sigma_frames])
    window_frames = int(baseline_window_seconds * sampling_rate_hz)
    baseline = minimum_filter1d(baseline, window_frames)
    baseline = maximum_filter1d(baseline, window_frames)
    return corrected - baseline


def align_motion_to_neural_frames(
    motion: np.ndarray,
    timestamps: np.ndarray,
    neural_frame_count: int,
) -> tuple[np.ndarray, int]:
    """Place motion samples on microscope-trigger indices and fill frame drops.

    The camera was triggered by the microscope.  Normally the two arrays have
    identical lengths and therefore align one-to-one.  For the nine supplied
    sessions with camera drops, each missing sample appears as an integer-
    multiple gap in ``tstamps``.  We recover those trigger indices and linearly
    interpolate only the absent motion samples.  This preserves complete trials
    rather than losing up to 60 seconds because of a single missing frame.
    """
    motion = np.asarray(motion, dtype=np.float64).reshape(-1)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if motion.size != timestamps.size:
        raise ValueError(
            f"Motion/timestamp length mismatch: {motion.size} vs {timestamps.size}"
        )
    if motion.size > neural_frame_count:
        raise ValueError(
            f"Motion has {motion.size} samples but neural has {neural_frame_count}"
        )
    if motion.size == neural_frame_count:
        return motion, 0
    if motion.size < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Motion timestamps must be strictly increasing")

    intervals = np.diff(timestamps)
    nominal_interval = np.median(intervals)
    missing_after = np.maximum(
        np.rint(intervals / nominal_interval).astype(np.int64) - 1,
        0,
    )
    trigger_indices = np.arange(motion.size, dtype=np.int64)
    trigger_indices[1:] += np.cumsum(missing_after)

    missing_count = neural_frame_count - motion.size
    if int(missing_after.sum()) != missing_count:
        raise ValueError(
            "Timestamp-inferred camera drops do not match neural/motion length "
            f"difference ({int(missing_after.sum())} vs {missing_count})"
        )
    if trigger_indices[-1] != neural_frame_count - 1:
        raise ValueError(
            f"Recovered final trigger {trigger_indices[-1]}, expected "
            f"{neural_frame_count - 1}"
        )

    aligned = np.interp(
        np.arange(neural_frame_count, dtype=np.float64),
        trigger_indices,
        motion,
    )
    return aligned, missing_count


def average_consecutive_bins(values: np.ndarray, bin_frames: int) -> np.ndarray:
    """Average the final axis in non-overlapping, complete frame bins."""
    complete_frames = values.shape[-1] // bin_frames * bin_frames
    if complete_frames != values.shape[-1]:
        values = values[..., :complete_frames]
    new_shape = values.shape[:-1] + (complete_frames // bin_frames, bin_frames)
    return values.reshape(new_shape).mean(axis=-1)


def motion_quintile_labels(
    binned_motion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize motion with four within-session percentile boundaries."""
    boundaries = np.percentile(binned_motion, MOTION_PERCENTILES)
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError(f"Non-unique motion percentile boundaries: {boundaries}")
    labels = np.digitize(binned_motion, boundaries, right=False).astype(np.int64)
    return labels, boundaries


def convert_session(session_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and process one recording session."""
    plane_dir = session_dir / "suite2p" / "plane0"
    motion_dir = session_dir / "move_deve"

    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    sampling_rate_hz = float(ops["fs"])
    if not np.isclose(sampling_rate_hz, EXPECTED_SOURCE_FS):
        raise ValueError(
            f"Unexpected sampling rate {sampling_rate_hz} Hz in {session_dir}"
        )

    iscell = np.load(plane_dir / "iscell.npy")
    cell_mask = iscell[:, 1] > ISCELL_THRESHOLD
    if not np.any(cell_mask):
        raise ValueError(f"No ROIs pass iscell>{ISCELL_THRESHOLD} in {session_dir}")

    fluorescence_all = np.load(plane_dir / "F.npy")
    neuropil_all = np.load(plane_dir / "Fneu.npy")
    if fluorescence_all.shape != neuropil_all.shape:
        raise ValueError(f"F/Fneu shape mismatch in {session_dir}")
    if fluorescence_all.shape[0] != iscell.shape[0]:
        raise ValueError(f"F/iscell ROI mismatch in {session_dir}")

    # Boolean indexing copies the entire recording.  The distributed Track2p
    # arrays are already threshold-filtered (all masks are true), so retain the
    # original buffers in that common case while preserving correct behavior
    # for an unfiltered source directory.
    if np.all(cell_mask):
        fluorescence = fluorescence_all
        neuropil = neuropil_all
    else:
        fluorescence = fluorescence_all[cell_mask]
        neuropil = neuropil_all[cell_mask]
    neural_frame_count = fluorescence.shape[1]

    corrected = baseline_correct_fluorescence(
        fluorescence,
        neuropil,
        sampling_rate_hz,
    )
    binned_neural = average_consecutive_bins(
        corrected,
        SOURCE_FRAMES_PER_BIN,
    ).astype(np.float32, copy=False)

    motion_raw = np.load(motion_dir / "motion_energy_glob.npy")
    timestamps = np.load(motion_dir / "tstamps.npy")
    aligned_motion, interpolated_frames = align_motion_to_neural_frames(
        motion_raw,
        timestamps,
        neural_frame_count,
    )
    binned_motion = average_consecutive_bins(
        aligned_motion,
        SOURCE_FRAMES_PER_BIN,
    )
    motion_labels, percentile_boundaries = motion_quintile_labels(binned_motion)

    # Mean raw-frame times make each decoder input the temporal center of its
    # 10-frame bin.  Values remain elapsed time from session start, not time
    # relative to the start of each 60-second trial.
    frame_times = np.arange(neural_frame_count, dtype=np.float64) / sampling_rate_hz
    binned_time = average_consecutive_bins(
        frame_times,
        SOURCE_FRAMES_PER_BIN,
    ).astype(np.float32, copy=False)

    if not (
        binned_neural.shape[1] == binned_motion.size == motion_labels.size == binned_time.size
    ):
        raise ValueError(f"Binned streams do not align in {session_dir}")

    binned_rate_hz = sampling_rate_hz / SOURCE_FRAMES_PER_BIN
    bins_per_trial = int(round(TRIAL_SECONDS * binned_rate_hz))
    if not np.isclose(bins_per_trial / binned_rate_hz, TRIAL_SECONDS):
        raise ValueError("Trial duration is not an integer number of time bins")
    trial_count = binned_time.size // bins_per_trial
    if trial_count < 2:
        raise ValueError(f"Fewer than two complete trials in {session_dir}")
    used_bins = trial_count * bins_per_trial

    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    for trial_index in range(trial_count):
        start = trial_index * bins_per_trial
        stop = start + bins_per_trial
        neural_trials.append(np.ascontiguousarray(binned_neural[:, start:stop]))
        input_trials.append(np.ascontiguousarray(binned_time[None, start:stop]))
        output_trials.append(np.ascontiguousarray(motion_labels[None, start:stop]))

    session_id = f"{session_dir.parent.name}/{session_dir.name}"
    result = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros(int(cell_mask.sum()), dtype=np.int64),
    }
    session_info = {
        "session_id": session_id,
        "subject": session_dir.parent.name,
        "source_sampling_rate_hz": sampling_rate_hz,
        "source_neural_frames": int(neural_frame_count),
        "source_motion_frames": int(motion_raw.size),
        "interpolated_motion_frames": int(interpolated_frames),
        "retained_neurons": int(cell_mask.sum()),
        "source_rois": int(iscell.shape[0]),
        "complete_trials": int(trial_count),
        "discarded_binned_samples": int(binned_time.size - used_bins),
        "motion_quintile_boundaries": percentile_boundaries.tolist(),
    }
    return result, session_info


def convert_dataset(data_dir: Path) -> dict[str, Any]:
    """Convert every supplied session to the requested nested dictionary."""
    session_dirs = find_sessions(data_dir)
    subjects = sorted({path.parent.name for path in session_dirs})
    subject_to_index = {subject: index for index, subject in enumerate(subjects)}

    neural: list[list[np.ndarray]] = []
    decoder_input: list[list[np.ndarray]] = []
    output: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    subject_idx: list[int] = []
    session_info: list[dict[str, Any]] = []

    for session_number, session_dir in enumerate(session_dirs, start=1):
        print(
            f"[{session_number:02d}/{len(session_dirs):02d}] "
            f"processing {session_dir.parent.name}/{session_dir.name}",
            flush=True,
        )
        converted, info = convert_session(session_dir)
        neural.append(converted["neural"])
        decoder_input.append(converted["input"])
        output.append(converted["output"])
        brain_region_idx.append(converted["brain_region_idx"])
        subject_idx.append(subject_to_index[session_dir.parent.name])
        session_info.append(info)

    time_bin_ms = 1000.0 * SOURCE_FRAMES_PER_BIN / EXPECTED_SOURCE_FS
    return {
        "neural": neural,
        "input": decoder_input,
        "output": output,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["barrel cortex (L2/3)"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time elapsed from session start (s)"],
        "output_names": ["motion energy quintile"],
        "output_values": [[
            "lowest (0-20th percentile)",
            "low (20-40th percentile)",
            "middle (40-60th percentile)",
            "high (60-80th percentile)",
            "highest (80-100th percentile)",
        ]],
        "metadata": {
            "task_description": (
                "Decode within-session motion-energy quintile from Track2p-matched "
                "layer 2/3 barrel-cortex calcium activity, with elapsed session "
                "time supplied as a decoder covariate."
            ),
            "time_bin_size": time_bin_ms,
            "temporal_alignment_event": (
                "start of each non-overlapping 60-second trial; decoder time input "
                "remains elapsed time from session start"
            ),
            "off_start": 0.0,
            "off_end": float(TRIAL_SECONDS),
            "source_frame_rate_hz": EXPECTED_SOURCE_FS,
            "frames_averaged_per_bin": SOURCE_FRAMES_PER_BIN,
            "trial_duration_seconds": float(TRIAL_SECONDS),
            "neural_signal": (
                "Track2p/Suite2p fluorescence, iscell probability > 0.5, "
                "baseline-corrected as F-F0 by the repository maximin routine "
                "(sigma 10 frames, 60 s window, neuropil coefficient 0.0), then "
                "averaged over 10 frames"
            ),
            "motion_processing": (
                "Supplied global consecutive-frame squared-pixel-difference motion "
                "energy; missing triggered camera samples interpolated; averaged "
                "over 10 frames; discretized at per-session percentiles"
            ),
            "motion_percentiles": list(MOTION_PERCENTILES),
            "session_info": session_info,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing subject/session folders (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "converted_data.pkl",
        help="Output pickle path (default: ./converted_data.pkl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = convert_dataset(args.data_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
