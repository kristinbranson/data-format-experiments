#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from suite2p.extraction import dcnv


DATA_ROOT = Path("/app/data")
DEFAULT_BIN_FRAMES = 10
DEFAULT_TRIAL_SECONDS = 60
DEFAULT_FS = 30.0
KS_TO_SECONDS = 1000.0


@dataclass(frozen=True)
class SessionPath:
    subject: str
    session: str
    path: Path

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.session}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Track2p longitudinal imaging data into decoder format."
    )
    parser.add_argument("outpicklefile", type=str, help="Output pickle path.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Process all sessions (default).")
    mode.add_argument("--sample", action="store_true", help="Process only 2 sessions.")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="Save processing plots for up to 2 sessions as processing_<session_id>.png.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def list_sessions() -> list[SessionPath]:
    sessions: list[SessionPath] = []
    for subject_dir in sorted(DATA_ROOT.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("jm"):
            continue
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir() or not session_dir.name[:4].isdigit():
                continue
            sessions.append(
                SessionPath(
                    subject=subject_dir.name,
                    session=session_dir.name,
                    path=session_dir,
                )
            )
    return sessions


def choose_sample_sessions(sessions: list[SessionPath]) -> list[SessionPath]:
    if len(sessions) <= 2:
        return sessions

    first = sessions[0]
    mismatch = None
    for sess in sessions:
        motion = np.load(sess.path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
        ops = np.load(sess.path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()
        if not (len(motion) == int(ops["nframes"])):
            mismatch = sess
            break

    if mismatch is None or mismatch.session_id == first.session_id:
        return sessions[:2]
    return [first, mismatch]


def load_ops(session_path: Path) -> dict:
    return np.load(session_path / "suite2p" / "plane0" / "ops.npy", allow_pickle=True).item()


def load_neural_arrays(session_path: Path) -> tuple[np.ndarray, np.ndarray]:
    f = np.load(session_path / "suite2p" / "plane0" / "F.npy", allow_pickle=True).astype(np.float32)
    fneu = np.load(session_path / "suite2p" / "plane0" / "Fneu.npy", allow_pickle=True).astype(np.float32)
    return f, fneu


def load_behavior_arrays(session_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion = np.load(session_path / "move_deve" / "motion_energy_glob.npy", allow_pickle=True)
    tstamps = np.load(session_path / "move_deve" / "tstamps.npy", allow_pickle=True)
    interframe = np.load(session_path / "move_deve" / "interframe_int.npy", allow_pickle=True)
    return motion.astype(np.float64), tstamps.astype(np.float64), interframe.astype(np.float64)


def get_brain_region_idx(n_neurons: int) -> np.ndarray:
    return np.zeros(n_neurons, dtype=np.int64)


def infer_timestamp_scale_seconds(interframe: np.ndarray, fs: float) -> float:
    if interframe.size == 0:
        return 1.0
    nominal = 1.0 / fs
    median_dt = float(np.median(interframe))
    ratio = nominal / median_dt if median_dt > 0 else 1.0
    if abs(ratio - 1000.0) < abs(ratio - 1.0):
        return KS_TO_SECONDS
    return 1.0


def full_frame_times_seconds(n_frames: int, fs: float) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float64) / fs


def align_motion_to_imaging(
    motion: np.ndarray,
    tstamps: np.ndarray,
    interframe: np.ndarray,
    n_frames: int,
    fs: float,
) -> tuple[np.ndarray, dict]:
    scale = infer_timestamp_scale_seconds(interframe, fs)
    motion_times = tstamps * scale
    imaging_times = full_frame_times_seconds(n_frames, fs)
    if motion_times.size == 0:
        raise ValueError("No motion timestamps available.")

    aligned = np.interp(imaging_times, motion_times, motion.astype(np.float64))
    nominal_dt = 1.0 / fs
    gaps = np.where(np.diff(motion_times) > nominal_dt * 1.5)[0]
    gap_sizes = []
    for idx in gaps:
        missing = int(round((motion_times[idx + 1] - motion_times[idx]) * fs)) - 1
        gap_sizes.append(max(missing, 0))

    info = {
        "timestamp_scale_to_seconds": scale,
        "motion_len_raw": int(len(motion)),
        "motion_len_aligned": int(len(aligned)),
        "missing_motion_frames": int(n_frames - len(motion)),
        "gap_indices_raw": gaps.astype(np.int64),
        "gap_sizes_frames": np.array(gap_sizes, dtype=np.int64),
        "motion_times_seconds": motion_times,
        "imaging_times_seconds": imaging_times,
    }
    return aligned.astype(np.float32), info


def preprocess_neural(F: np.ndarray, Fneu: np.ndarray, ops: dict) -> tuple[np.ndarray, dict]:
    fs = float(ops["fs"])
    fc = F - np.float32(ops["neucoeff"]) * Fneu
    corrected = dcnv.preprocess(
        fc.copy(),
        baseline=ops["baseline"],
        win_baseline=float(ops["win_baseline"]),
        sig_baseline=float(ops["sig_baseline"]),
        fs=fs,
        prctile_baseline=float(ops.get("prctile_baseline", 8.0)),
        batch_size=int(ops.get("batch_size", 2000)),
        device=torch.device("cpu"),
    )
    info = {
        "fs": fs,
        "neucoeff": float(ops["neucoeff"]),
        "baseline": ops["baseline"],
        "win_baseline": float(ops["win_baseline"]),
        "sig_baseline": float(ops["sig_baseline"]),
        "prctile_baseline": float(ops.get("prctile_baseline", 8.0)),
        "batch_size": int(ops.get("batch_size", 2000)),
    }
    return corrected.astype(np.float32), info


def mean_bin_2d(arr: np.ndarray, bin_frames: int) -> np.ndarray:
    n_rows, n_frames = arr.shape
    usable = (n_frames // bin_frames) * bin_frames
    if usable == 0:
        raise ValueError("Array is too short for requested bin size.")
    return arr[:, :usable].reshape(n_rows, usable // bin_frames, bin_frames).mean(axis=2)


def mean_bin_1d(arr: np.ndarray, bin_frames: int) -> np.ndarray:
    usable = (arr.shape[0] // bin_frames) * bin_frames
    if usable == 0:
        raise ValueError("Array is too short for requested bin size.")
    return arr[:usable].reshape(usable // bin_frames, bin_frames).mean(axis=1)


def build_time_input(n_frames: int, fs: float, bin_frames: int) -> np.ndarray:
    frame_times = full_frame_times_seconds(n_frames, fs)
    binned = mean_bin_1d(frame_times.astype(np.float32), bin_frames)
    return binned.astype(np.float32)


def equal_frequency_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 1:
        raise ValueError("Values for discretization must be 1D.")
    n = values.shape[0]
    if n == 0:
        raise ValueError("Cannot discretize an empty array.")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    bins = (ranks * n_bins) // n
    bins = np.minimum(bins, n_bins - 1).astype(np.int64)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    return bins, quantiles.astype(np.float64)


def split_trials_2d(arr: np.ndarray, trial_bins: int) -> list[np.ndarray]:
    n_rows, n_time = arr.shape
    if n_time % trial_bins != 0:
        raise ValueError(f"Time dimension {n_time} is not divisible by trial_bins {trial_bins}.")
    n_trials = n_time // trial_bins
    reshaped = arr.reshape(n_rows, n_trials, trial_bins)
    return [reshaped[:, i, :].astype(arr.dtype, copy=False) for i in range(n_trials)]


def split_trials_1d(arr: np.ndarray, trial_bins: int) -> list[np.ndarray]:
    if arr.shape[0] % trial_bins != 0:
        raise ValueError(f"Length {arr.shape[0]} is not divisible by trial_bins {trial_bins}.")
    n_trials = arr.shape[0] // trial_bins
    reshaped = arr.reshape(n_trials, trial_bins)
    return [reshaped[i][None, :].astype(arr.dtype, copy=False) for i in range(n_trials)]


def trial_count_from_frames(n_frames: int, fs: float, bin_frames: int, trial_seconds: int) -> int:
    bins_per_trial = int(round(trial_seconds * fs / bin_frames))
    return (n_frames // bin_frames) // bins_per_trial


def build_processing_plot(
    out_path: Path,
    session: SessionPath,
    raw_F: np.ndarray,
    corrected_F: np.ndarray,
    raw_motion: np.ndarray,
    aligned_motion: np.ndarray,
    binned_motion: np.ndarray,
    motion_bins: np.ndarray,
    time_binned: np.ndarray,
    gap_sizes: np.ndarray,
    quantiles: np.ndarray,
    bin_frames: int,
    fs: float,
) -> None:
    example_neurons = np.arange(min(3, raw_F.shape[0]))
    snippet = min(int(fs * 120), raw_F.shape[1])
    snippet_bins = min(int(120 / (bin_frames / fs)), binned_motion.shape[0])

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    for idx in example_neurons:
        ax1.plot(raw_F[idx, :snippet], alpha=0.8, lw=1)
    ax1.set_title("Raw F traces (first 120 s)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Fluorescence")

    for idx in example_neurons:
        ax2.plot(corrected_F[idx, :snippet], alpha=0.8, lw=1)
    ax2.set_title("Baseline-corrected traces (first 120 s)")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Corrected signal")

    raw_motion_x = np.arange(raw_motion.shape[0], dtype=np.float64) / fs
    aligned_motion_x = np.arange(aligned_motion.shape[0], dtype=np.float64) / fs
    ax3.plot(raw_motion_x[:snippet], raw_motion[:snippet], lw=1, label="raw motion")
    ax3.plot(aligned_motion_x[:snippet], aligned_motion[:snippet], lw=1, label="aligned motion", alpha=0.7)
    ax3.set_title("Motion alignment (first 120 s)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Motion energy")
    ax3.legend(loc="upper right")

    ax4.plot(time_binned[:snippet_bins], binned_motion[:snippet_bins], lw=1)
    ax4.set_title("10-frame binned motion (first 120 s)")
    ax4.set_xlabel("Session time (s)")
    ax4.set_ylabel("Binned motion energy")

    ax5.hist(binned_motion, bins=50, color="0.6")
    for q in quantiles[1:-1]:
        ax5.axvline(q, color="tab:red", lw=1)
    ax5.set_title("Motion distribution with session quantiles")
    ax5.set_xlabel("Binned motion energy")
    ax5.set_ylabel("Count")

    counts = np.bincount(motion_bins, minlength=5)
    ax6.bar(np.arange(5), counts, color="tab:blue")
    gap_text = "none" if gap_sizes.size == 0 else ", ".join(str(int(x)) for x in gap_sizes[:10])
    ax6.set_title(f"Motion quintiles / missing-frame gaps: {gap_text}")
    ax6.set_xlabel("Motion quintile")
    ax6.set_ylabel("Time bins")
    ax6.set_xticks(np.arange(5))

    fig.suptitle(session.session_id)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def convert_session(
    session: SessionPath,
    bin_frames: int,
    trial_seconds: int,
    make_plot: bool,
) -> tuple[dict, dict]:
    start = time.perf_counter()

    ops = load_ops(session.path)
    fs = float(ops["fs"])
    n_frames = int(ops["nframes"])
    raw_F, raw_Fneu = load_neural_arrays(session.path)
    raw_motion, tstamps, interframe = load_behavior_arrays(session.path)

    corrected_F, neural_info = preprocess_neural(raw_F, raw_Fneu, ops)
    aligned_motion, motion_info = align_motion_to_imaging(raw_motion, tstamps, interframe, n_frames, fs)

    neural_binned = mean_bin_2d(corrected_F, bin_frames).astype(np.float32)
    motion_binned = mean_bin_1d(aligned_motion.astype(np.float32), bin_frames).astype(np.float32)
    time_binned = build_time_input(n_frames, fs, bin_frames)

    motion_labels, motion_quantiles = equal_frequency_bins(motion_binned.astype(np.float64), 5)
    motion_labels = motion_labels.astype(np.int64)

    bins_per_trial = int(round(trial_seconds * fs / bin_frames))
    neural_trials = split_trials_2d(neural_binned, bins_per_trial)
    input_trials = split_trials_1d(time_binned, bins_per_trial)
    output_trials = split_trials_1d(motion_labels.astype(np.int64), bins_per_trial)

    if not (len(neural_trials) == len(input_trials) == len(output_trials)):
        raise RuntimeError(f"Trial count mismatch for {session.session_id}.")

    duration = time.perf_counter() - start
    summary = {
        "subject": session.subject,
        "session": session.session,
        "session_id": session.session_id,
        "n_neurons": int(raw_F.shape[0]),
        "n_frames": n_frames,
        "n_motion_raw": int(raw_motion.shape[0]),
        "n_trials": int(len(neural_trials)),
        "trial_timepoints": int(bins_per_trial),
        "bin_frames": int(bin_frames),
        "fs": fs,
        "duration_seconds": duration,
        "motion_missing_frames": int(motion_info["missing_motion_frames"]),
        "motion_gap_sizes_frames": motion_info["gap_sizes_frames"].tolist(),
        "motion_quantiles": motion_quantiles.tolist(),
        "time_start_s": float(time_binned[0]),
        "time_end_s": float(time_binned[-1]),
    }

    if make_plot:
        build_processing_plot(
            Path(f"/app/processing_{session.session_id}.png"),
            session,
            raw_F,
            corrected_F,
            raw_motion,
            aligned_motion,
            motion_binned,
            motion_labels,
            time_binned,
            motion_info["gap_sizes_frames"],
            motion_quantiles,
            bin_frames,
            fs,
        )

    session_payload = {
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
        "brain_region_idx": get_brain_region_idx(raw_F.shape[0]),
        "session_summary": summary,
        "neural_info": neural_info,
        "motion_info": {
            "timestamp_scale_to_seconds": float(motion_info["timestamp_scale_to_seconds"]),
            "motion_len_raw": int(motion_info["motion_len_raw"]),
            "motion_len_aligned": int(motion_info["motion_len_aligned"]),
            "missing_motion_frames": int(motion_info["missing_motion_frames"]),
        },
    }
    return session_payload, summary


def assemble_dataset(processed_sessions: list[tuple[SessionPath, dict]]) -> dict:
    subjects = sorted({session.subject for session, _ in processed_sessions})
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}

    session_info = []
    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.array([subject_lookup[s.subject] for s, _ in processed_sessions], dtype=np.int64),
        "brain_regions": ["barrel cortex"],
        "brain_region_idx": [],
        "input_names": ["time_from_session_start_s"],
        "output_names": ["motion_energy_quintile"],
        "output_values": [[
            "lowest_20pct",
            "lowmid_20pct",
            "middle_20pct",
            "highmid_20pct",
            "highest_20pct",
        ]],
        "metadata": {},
    }

    for session, payload in processed_sessions:
        data["neural"].append(payload["neural_trials"])
        data["input"].append(payload["input_trials"])
        data["output"].append(payload["output_trials"])
        data["brain_region_idx"].append(payload["brain_region_idx"])
        session_info.append(
            {
                "subject": session.subject,
                "session": session.session,
                **payload["session_summary"],
                "neural_preprocessing": payload["neural_info"],
                "motion_alignment": payload["motion_info"],
            }
        )

    data["metadata"] = {
        "task_description": (
            "Decode session-specific motion-energy quintile from longitudinal barrel cortex neural activity. "
            "Sessions are split into non-overlapping 60-second windows after 10-frame temporal averaging."
        ),
        "time_bin_size": 1000.0 * DEFAULT_BIN_FRAMES / DEFAULT_FS,
        "temporal_alignment_event": "start of each derived 60-second trial window",
        "off_start": 0.0,
        "off_end": float(DEFAULT_TRIAL_SECONDS),
        "original_frame_rate_hz": DEFAULT_FS,
        "bin_frames": DEFAULT_BIN_FRAMES,
        "trace_representation": (
            "Neuropil-subtracted, Suite2p-style baseline-corrected fluorescence reconstructed from F/Fneu/ops."
        ),
        "behavior_representation": (
            "Global motion energy aligned to imaging frames and averaged in non-overlapping 10-frame bins."
        ),
        "output_definition": (
            "Per-session equal-frequency quintiles of 10-frame-binned motion energy."
        ),
        "brain_region": "barrel cortex layer 2/3",
        "session_info": session_info,
    }
    return data


def print_dataset_summary(processed_summaries: list[dict]) -> None:
    trial_counts = [summary["n_trials"] for summary in processed_summaries]
    neuron_counts = [summary["n_neurons"] for summary in processed_summaries]
    missing_counts = [summary["motion_missing_frames"] for summary in processed_summaries]
    total_trials = sum(trial_counts)
    total_neurons_session = sum(neuron_counts)
    log("Conversion summary:")
    log(f"  Sessions: {len(processed_summaries)}")
    log(f"  Trials total: {total_trials}")
    log(f"  Neuron-session rows total: {total_neurons_session}")
    log(f"  Trials/session range: {min(trial_counts)} - {max(trial_counts)}")
    log(f"  Neurons/session range: {min(neuron_counts)} - {max(neuron_counts)}")
    log(f"  Sessions with missing motion frames: {sum(x > 0 for x in missing_counts)}")


def main() -> None:
    args = parse_args()
    mode_full = args.full or not args.sample

    total_start = time.perf_counter()
    sessions = list_sessions()
    if not mode_full:
        sessions = choose_sample_sessions(sessions)
        log("Sample mode selected.")
    log(f"Processing {len(sessions)} session(s).")

    processed_sessions: list[tuple[SessionPath, dict]] = []
    processed_summaries: list[dict] = []
    durations = []

    plot_budget = 2
    for idx, session in enumerate(sessions, start=1):
        session_start = time.perf_counter()
        make_plot = args.show_processing and idx <= plot_budget
        payload, summary = convert_session(
            session,
            bin_frames=DEFAULT_BIN_FRAMES,
            trial_seconds=DEFAULT_TRIAL_SECONDS,
            make_plot=make_plot,
        )
        processed_sessions.append((session, payload))
        processed_summaries.append(summary)
        elapsed = time.perf_counter() - session_start
        durations.append(elapsed)
        remaining = len(sessions) - idx
        eta = np.mean(durations) * remaining if remaining > 0 else 0.0
        log(
            f"[{idx}/{len(sessions)}] {session.session_id}: "
            f"{summary['n_neurons']} neurons, {summary['n_trials']} trials, "
            f"missing motion frames={summary['motion_missing_frames']}, "
            f"time={elapsed:.2f}s, eta={eta:.2f}s"
        )

    data = assemble_dataset(processed_sessions)
    with open(args.outpicklefile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print_dataset_summary(processed_summaries)
    total_elapsed = time.perf_counter() - total_start
    log(f"Saved converted dataset to {args.outpicklefile}")
    log(f"Total conversion time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
