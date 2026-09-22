#!/usr/bin/env python3
"""Convert the released Track2p dataset to decoder-compatible trials.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full|--sample]
                                               [--show-processing]
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d


DATA_ROOT = Path("/app/data")
RAW_FS_HZ = 30.0
ANALYSIS_FRAMES = 36_000  # first 20 min, matching the paper
FRAMES_PER_BIN = 10
TRIAL_SECONDS = 60
BINNED_FS_HZ = RAW_FS_HZ / FRAMES_PER_BIN
TRIAL_BINS = int(TRIAL_SECONDS * BINNED_FS_HZ)
N_TRIALS = ANALYSIS_FRAMES // (FRAMES_PER_BIN * TRIAL_BINS)
CHUNK_NEURONS = 64


def discover_sessions() -> list[tuple[str, Path]]:
    """Return all (subject, session path) pairs in chronological subject order."""
    sessions: list[tuple[str, Path]] = []
    for subject_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            if (session_dir / "suite2p" / "plane0" / "F.npy").exists():
                sessions.append((subject_dir.name, session_dir))
    return sessions


def select_sessions(
    sessions: list[tuple[str, Path]], sample: bool
) -> list[tuple[str, Path]]:
    """Select all sessions or two late sessions from the paper's example mouse."""
    if not sample:
        return sessions
    example = [(s, p) for s, p in sessions if s == "jm039"]
    if len(example) < 2:
        raise RuntimeError("Sample mode requires at least two jm039 sessions")
    return example[-2:]


def load_and_align_motion(session_dir: Path) -> tuple[np.ndarray, dict]:
    """Interpolate camera motion onto expected imaging-trigger frame ordinals.

    Camera timestamps have an arbitrary scale, so only ratios to the median
    inter-frame interval are used. Normal intervals map to one trigger step;
    gaps map to two or more steps and therefore expose dropped camera frames.
    """
    move_dir = session_dir / "move_deve"
    motion = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
    timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
    intervals_stored = np.load(move_dir / "interframe_int.npy").astype(np.float64)

    if motion.ndim != 1 or timestamps.ndim != 1:
        raise ValueError(f"Motion/timestamps must be 1D in {session_dir}")
    if len(motion) != len(timestamps):
        raise ValueError(f"Motion/timestamp length mismatch in {session_dir}")
    if len(intervals_stored) != len(timestamps) - 1:
        raise ValueError(f"Inter-frame interval length mismatch in {session_dir}")
    if not np.allclose(intervals_stored, np.diff(timestamps), rtol=1e-8, atol=1e-12):
        raise ValueError(f"Stored inter-frame intervals disagree with timestamps in {session_dir}")

    median_interval = float(np.median(intervals_stored))
    trigger_steps = np.rint(intervals_stored / median_interval).astype(np.int64)
    if np.any(trigger_steps < 1):
        raise ValueError(f"Non-positive inferred camera trigger step in {session_dir}")
    trigger_positions = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(trigger_steps, dtype=np.int64)]
    )
    target_positions = np.arange(ANALYSIS_FRAMES, dtype=np.float64)
    if trigger_positions[-1] < ANALYSIS_FRAMES - 1:
        raise ValueError(
            f"Behavior ends at trigger {trigger_positions[-1]} before retained neural data "
            f"ends at {ANALYSIS_FRAMES - 1} in {session_dir}"
        )

    aligned = np.interp(target_positions, trigger_positions, motion)
    gaps_in_window = (trigger_steps > 1) & (trigger_positions[:-1] < ANALYSIS_FRAMES)
    missing_in_window = int(np.sum(trigger_steps[gaps_in_window] - 1))
    identity_alignment = bool(
        len(motion) >= ANALYSIS_FRAMES
        and np.array_equal(trigger_positions[:ANALYSIS_FRAMES], np.arange(ANALYSIS_FRAMES))
    )
    if identity_alignment and not np.allclose(aligned, motion[:ANALYSIS_FRAMES]):
        raise AssertionError("Gap-free behavior alignment unexpectedly changed values")

    info = {
        "source_motion_frames": int(len(motion)),
        "median_timestamp_interval_stored_units": median_interval,
        "inferred_missing_camera_frames_retained_window": missing_in_window,
        "last_inferred_camera_trigger": int(trigger_positions[-1]),
        "identity_alignment": identity_alignment,
    }
    return aligned, info


def process_fluorescence(
    session_dir: Path, keep_debug: bool = False
) -> tuple[np.ndarray, dict | None, dict]:
    """Apply Suite2p-default baseline correction and non-overlapping 10-frame means."""
    plane_dir = session_dir / "suite2p" / "plane0"
    fluorescence = np.load(plane_dir / "F.npy", mmap_mode="r")
    neuropil = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()

    if fluorescence.shape != neuropil.shape or fluorescence.ndim != 2:
        raise ValueError(f"F/Fneu shape mismatch in {session_dir}")
    if fluorescence.shape[1] < ANALYSIS_FRAMES:
        raise ValueError(f"Fewer than {ANALYSIS_FRAMES} neural frames in {session_dir}")
    fs = float(ops["fs"])
    if not np.isclose(fs, RAW_FS_HZ):
        raise ValueError(f"Unexpected imaging rate {fs} Hz in {session_dir}")

    neucoeff = float(ops.get("neucoeff", 0.7))
    baseline = ops.get("baseline", "maximin")
    sig_baseline = float(ops.get("sig_baseline", 10.0))
    win_baseline_s = float(ops.get("win_baseline", 60.0))
    if baseline != "maximin":
        raise ValueError(f"Unsupported baseline mode {baseline!r} in {session_dir}")
    win_frames = int(win_baseline_s * fs)

    n_neurons, source_frames = fluorescence.shape
    n_bins = ANALYSIS_FRAMES // FRAMES_PER_BIN
    binned = np.empty((n_neurons, n_bins), dtype=np.float32)
    debug = None

    for start in range(0, n_neurons, CHUNK_NEURONS):
        stop = min(start + CHUNK_NEURONS, n_neurons)
        raw_f = np.asarray(fluorescence[start:stop], dtype=np.float32)
        raw_fneu = np.asarray(neuropil[start:stop], dtype=np.float32)
        corrected_neuropil = raw_f - np.float32(neucoeff) * raw_fneu
        flow = gaussian_filter1d(
            corrected_neuropil, sigma=sig_baseline, axis=1, mode="reflect"
        )
        flow = minimum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
        flow = maximum_filter1d(flow, size=win_frames, axis=1, mode="reflect")
        baseline_corrected = corrected_neuropil - flow
        retained = baseline_corrected[:, :ANALYSIS_FRAMES]
        binned[start:stop] = retained.reshape(stop - start, n_bins, FRAMES_PER_BIN).mean(
            axis=2
        )

        if keep_debug and start == 0:
            n_debug = min(5, stop - start)
            debug = {
                "raw_f": raw_f[:n_debug, :ANALYSIS_FRAMES].copy(),
                "raw_fneu": raw_fneu[:n_debug, :ANALYSIS_FRAMES].copy(),
                "neuropil_corrected": corrected_neuropil[
                    :n_debug, :ANALYSIS_FRAMES
                ].copy(),
                "baseline": flow[:n_debug, :ANALYSIS_FRAMES].copy(),
                "baseline_corrected": retained[:n_debug].copy(),
            }

    if not np.isfinite(binned).all():
        raise ValueError(f"Non-finite processed fluorescence in {session_dir}")
    info = {
        "source_neural_frames": int(source_frames),
        "n_neurons": int(n_neurons),
        "neuropil_coefficient": neucoeff,
        "baseline_method": baseline,
        "baseline_gaussian_sigma_frames": sig_baseline,
        "baseline_window_seconds": win_baseline_s,
    }
    return binned, debug, info


def discretize_motion(aligned_motion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average ten frames and assign session-specific percentile quintiles."""
    if aligned_motion.shape != (ANALYSIS_FRAMES,):
        raise ValueError(f"Unexpected aligned motion shape {aligned_motion.shape}")
    binned = aligned_motion.reshape(-1, FRAMES_PER_BIN).mean(axis=1)
    edges = np.quantile(binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.searchsorted(edges, binned, side="right").astype(np.int64)
    if labels.min() < 0 or labels.max() > 4:
        raise AssertionError("Motion labels outside 0..4")
    return binned, edges, labels


def make_processing_plot(
    session_id: str,
    debug: dict,
    raw_motion: np.ndarray,
    aligned_motion: np.ndarray,
    binned_motion: np.ndarray,
    edges: np.ndarray,
    labels: np.ndarray,
    elapsed_time: np.ndarray,
) -> Path:
    """Plot every material conversion stage for one session."""
    display_frames = min(3_600, ANALYSIS_FRAMES)
    raw_t = np.arange(display_frames) / RAW_FS_HZ
    bin_t = elapsed_time[: display_frames // FRAMES_PER_BIN]
    fig, axes = plt.subplots(5, 2, figsize=(18, 18), constrained_layout=True)

    axes[0, 0].plot(raw_t, debug["raw_f"][0, :display_frames], lw=0.6, label="F")
    axes[0, 0].plot(
        raw_t, debug["raw_fneu"][0, :display_frames], lw=0.6, label="Fneu"
    )
    axes[0, 0].set_title("1. Source fluorescence and neuropil")
    axes[0, 0].legend()

    axes[0, 1].plot(
        raw_t, debug["neuropil_corrected"][0, :display_frames], lw=0.6
    )
    axes[0, 1].set_title("2. Neuropil-corrected F - 0.7×Fneu")

    axes[1, 0].plot(
        raw_t, debug["neuropil_corrected"][0, :display_frames], lw=0.5, label="Fc"
    )
    axes[1, 0].plot(
        raw_t, debug["baseline"][0, :display_frames], lw=1.0, label="maximin baseline"
    )
    axes[1, 0].set_title("3. Suite2p maximin baseline")
    axes[1, 0].legend()

    axes[1, 1].plot(
        raw_t, debug["baseline_corrected"][0, :display_frames], lw=0.6
    )
    axes[1, 1].set_title("4. Baseline-corrected neural activity")

    neural_small = debug["baseline_corrected"][:, :display_frames]
    neural_binned = neural_small.reshape(neural_small.shape[0], -1, FRAMES_PER_BIN).mean(2)
    axes[2, 0].imshow(neural_binned, aspect="auto", extent=[bin_t[0], bin_t[-1], 5, 0])
    axes[2, 0].set_title("5. Ten-frame averaged neural activity")
    axes[2, 0].set_ylabel("example neuron")

    raw_n = min(len(raw_motion), display_frames)
    axes[2, 1].plot(np.arange(raw_n) / RAW_FS_HZ, raw_motion[:raw_n], lw=0.5, label="camera samples")
    axes[2, 1].plot(raw_t, aligned_motion[:display_frames], lw=0.5, alpha=0.8, label="aligned")
    axes[2, 1].set_title("6. Timestamp-aware motion alignment")
    axes[2, 1].legend()

    axes[3, 0].plot(bin_t, binned_motion[: len(bin_t)], lw=0.8)
    axes[3, 0].set_title("7. Ten-frame averaged motion energy")

    axes[3, 1].hist(binned_motion, bins=80, color="0.5")
    for edge in edges:
        axes[3, 1].axvline(edge, color="C3", lw=1)
    axes[3, 1].set_title("8. Session percentile thresholds")

    axes[4, 0].step(bin_t, labels[: len(bin_t)], where="mid")
    axes[4, 0].set_yticks(range(5))
    axes[4, 0].set_title("9. Five-class motion output")

    axes[4, 1].plot(elapsed_time, elapsed_time, lw=1)
    for boundary in np.arange(TRIAL_SECONDS, 20 * 60, TRIAL_SECONDS):
        axes[4, 1].axvline(boundary, color="0.8", lw=0.5)
    axes[4, 1].set_title("10. Elapsed-time input and 60-s trial boundaries")
    axes[4, 1].set_ylabel("input time (s)")

    for ax in axes.flat:
        ax.set_xlabel("seconds" if ax is not axes[3, 1] else "motion energy")
    fig.suptitle(f"Conversion processing: {session_id}", fontsize=16)
    out = Path(f"/app/processing_{session_id}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def validate_session_arrays(
    neural_binned: np.ndarray, elapsed_time: np.ndarray, labels: np.ndarray
) -> None:
    expected_bins = N_TRIALS * TRIAL_BINS
    if neural_binned.shape[1] != expected_bins:
        raise AssertionError(f"Neural bins {neural_binned.shape[1]} != {expected_bins}")
    if elapsed_time.shape != (expected_bins,) or labels.shape != (expected_bins,):
        raise AssertionError("Input/output binned lengths do not match neural data")
    if not np.isfinite(neural_binned).all() or not np.isfinite(elapsed_time).all():
        raise AssertionError("Non-finite neural/input values")


def convert(out_path: Path, sample: bool, show_processing: bool) -> None:
    started = time.perf_counter()
    selected = select_sessions(discover_sessions(), sample=sample)
    if not selected:
        raise RuntimeError("No source sessions found")
    subjects = sorted({subject for subject, _ in selected})
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[subject] for subject, _ in selected], dtype=np.int64
        ),
        "brain_regions": ["S1 barrel cortex (L2/3)"],
        "brain_region_idx": [],
        "input_names": ["time_elapsed_from_session_start_s"],
        "output_names": ["motion_energy_quintile"],
        "output_values": [["lowest", "low", "middle", "high", "highest"]],
        "metadata": {
            "task_description": (
                "Decode session-specific quintiles of spontaneous whole-body motion energy "
                "from longitudinally tracked layer 2/3 barrel-cortex calcium activity."
            ),
            "time_bin_size": 1000.0 * FRAMES_PER_BIN / RAW_FS_HZ,
            "temporal_alignment_event": (
                "Session start; trials are consecutive non-overlapping 60-second windows."
            ),
            "off_start": None,
            "off_end": None,
            "raw_sampling_rate_hz": RAW_FS_HZ,
            "frames_per_time_bin": FRAMES_PER_BIN,
            "trial_duration_seconds": TRIAL_SECONDS,
            "retained_session_duration_seconds": ANALYSIS_FRAMES / RAW_FS_HZ,
            "neural_signal": (
                "Suite2p-default neuropil- and maximin-baseline-corrected fluorescence, "
                "averaged over 10 frames"
            ),
            "behavior_alignment": (
                "Camera trigger ordinals reconstructed from timestamp interval ratios; "
                "missing samples linearly interpolated onto imaging frame indices."
            ),
            "output_discretization": (
                "20/40/60/80th percentiles of aligned 10-frame-mean motion, separately "
                "for each session; labels 0 (lowest) through 4 (highest)."
            ),
            "source_dataset": "Majnik et al. 2025 Track2p developmental barrel-cortex data",
            "session_info": [],
        },
    }

    elapsed_time = (
        np.arange(ANALYSIS_FRAMES, dtype=np.float64)
        .reshape(-1, FRAMES_PER_BIN)
        .mean(axis=1)
        / RAW_FS_HZ
    ).astype(np.float32)
    plotted = 0

    for session_index, (subject, session_dir) in enumerate(selected):
        session_started = time.perf_counter()
        session_id = f"{subject}_{session_dir.name}"
        print(
            f"[{session_index + 1}/{len(selected)}] Processing {session_id}", flush=True
        )
        aligned_motion, motion_info = load_and_align_motion(session_dir)
        binned_motion, edges, labels = discretize_motion(aligned_motion)
        want_debug = show_processing and plotted < 2
        neural_binned, debug, neural_info = process_fluorescence(
            session_dir, keep_debug=want_debug
        )
        validate_session_arrays(neural_binned, elapsed_time, labels)

        neural_trials = [
            x.copy()
            for x in np.split(neural_binned, N_TRIALS, axis=1)
        ]
        input_trials = [
            x[np.newaxis, :].copy()
            for x in np.split(elapsed_time, N_TRIALS)
        ]
        output_trials = [
            x[np.newaxis, :].copy()
            for x in np.split(labels, N_TRIALS)
        ]
        if not (
            len(neural_trials) == len(input_trials) == len(output_trials) == N_TRIALS
        ):
            raise AssertionError("Trial count mismatch")

        data["neural"].append(neural_trials)
        data["input"].append(input_trials)
        data["output"].append(output_trials)
        data["brain_region_idx"].append(
            np.zeros(neural_binned.shape[0], dtype=np.int64)
        )
        counts = np.bincount(labels, minlength=5)
        session_info = {
            "session_id": session_id,
            "subject": subject,
            **neural_info,
            **motion_info,
            "retained_neural_frames": ANALYSIS_FRAMES,
            "motion_quintile_edges": edges.tolist(),
            "motion_class_counts": counts.tolist(),
        }
        data["metadata"]["session_info"].append(session_info)

        if want_debug:
            raw_motion = np.load(session_dir / "move_deve" / "motion_energy_glob.npy")
            plot_path = make_processing_plot(
                session_id,
                debug,
                raw_motion,
                aligned_motion,
                binned_motion,
                edges,
                labels,
                elapsed_time,
            )
            print(f"  Saved {plot_path}", flush=True)
            plotted += 1

        elapsed = time.perf_counter() - session_started
        print(
            f"  neurons={neural_binned.shape[0]}, trials={N_TRIALS}, "
            f"class_counts={counts.tolist()}, missing_camera_frames="
            f"{motion_info['inferred_missing_camera_frames_retained_window']}, "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    total_elapsed = time.perf_counter() - started
    print(
        f"Saved {len(selected)} sessions / {len(selected) * N_TRIALS} trials to "
        f"{out_path} ({out_path.stat().st_size / 1024**2:.1f} MiB) in "
        f"{total_elapsed:.2f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full", action="store_true", help="process all sessions (default)")
    modes.add_argument("--sample", action="store_true", help="process two late example sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    convert(
        out_path=arguments.outpicklefile,
        sample=arguments.sample,
        show_processing=arguments.show_processing,
    )
