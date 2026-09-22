#!/usr/bin/env python3
"""Convert the released Track2p dataset to the neural-decoder format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample]
                                      [--show-processing]

The conversion follows the paper's fluorescence preprocessing and paired
10-frame averaging, then applies the task-specific 60 s segmentation and
per-session motion-energy quintiles.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d


APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT / "data"
NATIVE_FS_HZ = 30
DOWNSAMPLE = 10
OUTPUT_FS_HZ = NATIVE_FS_HZ / DOWNSAMPLE
TIME_BIN_MS = 1000.0 / OUTPUT_FS_HZ
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * OUTPUT_FS_HZ)
NEUROPIL_COEFF = 0.7
BASELINE_SIGMA_FRAMES = 10.0
BASELINE_WINDOW_SECONDS = 60.0
NEURON_CHUNK = 64


@dataclass(frozen=True)
class SessionRef:
    subject: str
    session_id: str
    path: Path


def discover_sessions() -> tuple[list[str], list[SessionRef]]:
    """Return subjects and sessions in deterministic chronological order."""
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    sessions: list[SessionRef] = []
    for subject in subjects:
        subject_dir = DATA_ROOT / subject
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            required = (
                session_dir / "suite2p" / "plane0" / "F.npy",
                session_dir / "suite2p" / "plane0" / "Fneu.npy",
                session_dir / "move_deve" / "motion_energy_glob.npy",
            )
            if not all(p.exists() for p in required):
                raise FileNotFoundError(f"Incomplete session directory: {session_dir}")
            sessions.append(SessionRef(subject, session_dir.name, session_dir))
    return subjects, sessions


def validate_source_session(ref: SessionRef) -> tuple[int, int, dict]:
    """Validate packaged Track2p/Suite2p row and frame invariants."""
    plane = ref.path / "suite2p" / "plane0"
    f = np.load(plane / "F.npy", mmap_mode="r")
    fneu = np.load(plane / "Fneu.npy", mmap_mode="r")
    spks = np.load(plane / "spks.npy", mmap_mode="r")
    iscell = np.load(plane / "iscell.npy", mmap_mode="r")
    stat = np.load(plane / "stat.npy", allow_pickle=True)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()

    if f.ndim != 2 or f.shape != fneu.shape or f.shape != spks.shape:
        raise ValueError(f"Neural stream mismatch in {ref.path}: "
                         f"F={f.shape}, Fneu={fneu.shape}, spks={spks.shape}")
    n_neurons, n_frames = f.shape
    if iscell.shape != (n_neurons, 2) or len(stat) != n_neurons:
        raise ValueError(f"ROI row mismatch in {ref.path}")
    if not np.all(iscell[:, 0] == 1) or not np.all(iscell[:, 1] > 0.5):
        raise ValueError(f"Packaged neurons do not all pass Track2p curation: {ref.path}")
    if int(ops["fs"]) != NATIVE_FS_HZ or int(ops["nframes"]) != n_frames:
        raise ValueError(f"Unexpected sampling metadata in {ref.path}")
    if n_frames % (DOWNSAMPLE * TRIAL_BINS) != 0:
        raise ValueError(f"Session does not divide into full 60-s trials: {ref.path}")
    return n_neurons, n_frames, ops


def baseline_correct_and_bin(
    plane_dir: Path,
    n_neurons: int,
    n_frames: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply Suite2p-style baseline subtraction and non-overlapping 10-frame means.

    Neurons are processed in chunks to bound peak memory. Filtering is performed
    over the full continuous session, before trial splitting.
    """
    f = np.load(plane_dir / "F.npy", mmap_mode="r")
    fneu = np.load(plane_dir / "Fneu.npy", mmap_mode="r")
    n_binned = n_frames // DOWNSAMPLE
    result = np.empty((n_neurons, n_binned), dtype=np.float32)
    example: dict[str, np.ndarray] = {}
    baseline_window = int(BASELINE_WINDOW_SECONDS * NATIVE_FS_HZ)

    for start in range(0, n_neurons, NEURON_CHUNK):
        stop = min(start + NEURON_CHUNK, n_neurons)
        raw_f = np.asarray(f[start:stop], dtype=np.float32)
        raw_fneu = np.asarray(fneu[start:stop], dtype=np.float32)
        corrected_neuropil = raw_f - NEUROPIL_COEFF * raw_fneu
        baseline = gaussian_filter(
            corrected_neuropil, sigma=(0.0, BASELINE_SIGMA_FRAMES)
        )
        baseline = minimum_filter1d(
            baseline, size=baseline_window, axis=1
        )
        baseline = maximum_filter1d(
            baseline, size=baseline_window, axis=1
        )
        activity = corrected_neuropil - baseline
        result[start:stop] = activity.reshape(
            stop - start, n_binned, DOWNSAMPLE
        ).mean(axis=2)

        if start == 0:
            example = {
                "raw_f": raw_f[0].copy(),
                "raw_fneu": raw_fneu[0].copy(),
                "neuropil_corrected": corrected_neuropil[0].copy(),
                "baseline": baseline[0].copy(),
                "activity": activity[0].copy(),
                "activity_binned": result[0].copy(),
            }

    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite processed neural values in {plane_dir}")
    return result, example


def align_motion(
    move_dir: Path, expected_frames: int
) -> tuple[np.ndarray, dict[str, np.ndarray | int | float]]:
    """Align motion to neural frame indices, interpolating documented dropped frames."""
    raw = np.load(move_dir / "motion_energy_glob.npy").astype(np.float64)
    timestamps = np.load(move_dir / "tstamps.npy").astype(np.float64)
    intervals = np.load(move_dir / "interframe_int.npy").astype(np.float64)
    if len(raw) != len(timestamps) or len(intervals) != len(raw) - 1:
        raise ValueError(f"Malformed motion arrays in {move_dir}")
    if len(raw) > expected_frames:
        raise ValueError(f"Motion has more frames than neural data in {move_dir}")

    positive = intervals[intervals > 0]
    if len(positive) == 0:
        raise ValueError(f"No positive camera intervals in {move_dir}")
    median_interval = float(np.median(positive))
    interval_steps = np.maximum(1, np.rint(intervals / median_interval).astype(int))
    timestamp_gap_count = int(np.sum(interval_steps - 1))
    missing = expected_frames - len(raw)

    if missing:
        observed_positions = np.concatenate(([0], np.cumsum(interval_steps)))
        if observed_positions[-1] != expected_frames - 1:
            raise ValueError(
                f"Timestamp gaps do not explain frame deficit in {move_dir}: "
                f"last reconstructed index {observed_positions[-1]}, expected "
                f"{expected_frames - 1}"
            )
        aligned = np.interp(
            np.arange(expected_frames, dtype=np.float64), observed_positions, raw
        )
        interpolated_mask = np.ones(expected_frames, dtype=bool)
        interpolated_mask[observed_positions] = False
    else:
        # The release README defines dropped frames by a length mismatch. A few
        # full-length jm046 timestamp streams contain clock anomalies; no values
        # are inserted when no camera samples are absent.
        aligned = raw.copy()
        observed_positions = np.arange(expected_frames, dtype=int)
        interpolated_mask = np.zeros(expected_frames, dtype=bool)

    if len(aligned) != expected_frames or not np.all(np.isfinite(aligned)):
        raise ValueError(f"Failed motion alignment in {move_dir}")
    info: dict[str, np.ndarray | int | float] = {
        "raw": raw,
        "timestamps": timestamps,
        "observed_positions": observed_positions,
        "interpolated_mask": interpolated_mask,
        "missing_frames": missing,
        "timestamp_gap_count": timestamp_gap_count,
        "median_timestamp_interval": median_interval,
    }
    return aligned, info


def bin_and_discretize_motion(
    aligned_motion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average motion in 10-frame bins and assign session-specific quintiles."""
    motion_binned = aligned_motion.reshape(-1, DOWNSAMPLE).mean(axis=1)
    quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    labels = np.searchsorted(quantile_edges, motion_binned, side="right").astype(
        np.int64
    )
    counts = np.bincount(labels, minlength=5)
    expected = len(labels) // 5
    if not np.array_equal(counts, np.full(5, expected)):
        raise ValueError(f"Motion quintiles are not equal: counts={counts.tolist()}")
    return motion_binned, quantile_edges, labels


def make_processing_plot(
    ref: SessionRef,
    neural_binned: np.ndarray,
    trace_example: dict[str, np.ndarray],
    aligned_motion: np.ndarray,
    motion_info: dict[str, np.ndarray | int | float],
    motion_binned: np.ndarray,
    quantile_edges: np.ndarray,
    labels: np.ndarray,
) -> Path:
    """Plot every transformation stage for one session."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_time = np.arange(len(aligned_motion)) / NATIVE_FS_HZ
    bin_time = (np.arange(len(motion_binned)) * DOWNSAMPLE + 4.5) / NATIVE_FS_HZ
    first_two_min = min(len(aligned_motion), 2 * 60 * NATIVE_FS_HZ)
    first_five_min_bins = min(len(motion_binned), 5 * 60 * int(OUTPUT_FS_HZ))

    fig, axes = plt.subplots(4, 2, figsize=(16, 15))
    ax = axes.ravel()
    ax[0].plot(raw_time[:first_two_min], trace_example["raw_f"][:first_two_min], label="F", lw=0.7)
    ax[0].plot(raw_time[:first_two_min], trace_example["raw_fneu"][:first_two_min], label="Fneu", lw=0.7)
    ax[0].set(title="1. Raw fluorescence streams", xlabel="Time (s)")
    ax[0].legend()

    ax[1].plot(raw_time[:first_two_min], trace_example["neuropil_corrected"][:first_two_min], label="F - 0.7 Fneu", lw=0.7)
    ax[1].plot(raw_time[:first_two_min], trace_example["baseline"][:first_two_min], label="maximin baseline", lw=1.0)
    ax[1].set(title="2. Neuropil correction and baseline estimate", xlabel="Time (s)")
    ax[1].legend()

    ax[2].plot(raw_time[:first_two_min], trace_example["activity"][:first_two_min], label="baseline-subtracted", alpha=0.55, lw=0.6)
    bshow = first_two_min // DOWNSAMPLE
    ax[2].plot(bin_time[:bshow], trace_example["activity_binned"][:bshow], label="10-frame mean", lw=1.2)
    ax[2].set(title="3. Neural temporal averaging", xlabel="Time (s)")
    ax[2].legend()

    heat_n = min(40, neural_binned.shape[0])
    heat_t = min(360, neural_binned.shape[1])
    z = neural_binned[:heat_n, :heat_t]
    z = (z - z.mean(axis=1, keepdims=True)) / np.maximum(z.std(axis=1, keepdims=True), 1e-6)
    ax[3].imshow(z, aspect="auto", cmap="gray_r", vmin=-1, vmax=3,
                 extent=(bin_time[0], bin_time[heat_t - 1], heat_n, 0))
    ax[3].set(title="4. Processed neural raster (row z-scores)", xlabel="Time (s)", ylabel="Neuron")

    raw_motion = np.asarray(motion_info["raw"])
    obs = np.asarray(motion_info["observed_positions"])
    ax[4].plot(obs / NATIVE_FS_HZ, raw_motion, label="observed", lw=0.7, alpha=0.7)
    ax[4].plot(raw_time, aligned_motion, label="aligned/interpolated", lw=0.6, alpha=0.7)
    interp = np.asarray(motion_info["interpolated_mask"])
    if np.any(interp):
        ax[4].scatter(raw_time[interp], aligned_motion[interp], s=7, c="red", label="inserted")
    ax[4].set(title=f"5. Motion alignment ({int(motion_info['missing_frames'])} missing frames)", xlabel="Time (s)")
    ax[4].legend()

    neural_mean = neural_binned.mean(axis=0)[:first_five_min_bins]
    neural_mean = (neural_mean - neural_mean.mean()) / max(neural_mean.std(), 1e-6)
    mot = motion_binned[:first_five_min_bins]
    mot = (mot - mot.mean()) / max(mot.std(), 1e-6)
    ax[5].plot(bin_time[:first_five_min_bins], neural_mean, label="population activity (z)", lw=0.7)
    ax[5].plot(bin_time[:first_five_min_bins], mot, label="motion (z)", lw=0.7)
    ax[5].set(title="6. Paired 10-frame alignment", xlabel="Time (s)")
    ax[5].legend()

    ax[6].hist(motion_binned, bins=100, color="0.5")
    for edge in quantile_edges:
        ax[6].axvline(edge, color="C3", lw=1)
    ax[6].set(title="7. Per-session motion quintile thresholds", xlabel="10-frame mean motion energy", ylabel="Count")

    ax[7].step(bin_time[:first_five_min_bins], labels[:first_five_min_bins], where="mid", lw=0.8)
    for boundary in np.arange(0, 5 * 60 + 1, TRIAL_SECONDS):
        ax[7].axvline(boundary, color="C3", ls="--", lw=0.8)
    ax[7].set(title="8. Categorical output and 60-s trial boundaries", xlabel="Time (s)", ylabel="Motion class", yticks=range(5))

    fig.suptitle(f"Processing audit: {ref.subject}/{ref.session_id}")
    fig.tight_layout()
    out = APP_ROOT / f"processing_{ref.subject}_{ref.session_id}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def convert_session(ref: SessionRef, show_processing: bool) -> tuple[dict, dict]:
    """Convert and validate one continuous recording session."""
    started = time.perf_counter()
    n_neurons, n_frames, ops = validate_source_session(ref)
    plane = ref.path / "suite2p" / "plane0"

    t0 = time.perf_counter()
    neural_binned, trace_example = baseline_correct_and_bin(
        plane, n_neurons, n_frames
    )
    neural_seconds = time.perf_counter() - t0

    aligned_motion, motion_info = align_motion(ref.path / "move_deve", n_frames)
    motion_binned, quantile_edges, labels = bin_and_discretize_motion(aligned_motion)
    n_binned = n_frames // DOWNSAMPLE
    elapsed_time = (
        np.arange(n_binned, dtype=np.float32) * DOWNSAMPLE + np.float32(4.5)
    ) / np.float32(NATIVE_FS_HZ)

    if neural_binned.shape != (n_neurons, n_binned):
        raise AssertionError("Processed neural shape mismatch")
    if len(motion_binned) != n_binned or len(labels) != n_binned:
        raise AssertionError("Processed behavior length mismatch")

    n_trials = n_binned // TRIAL_BINS
    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []
    for trial in range(n_trials):
        sl = slice(trial * TRIAL_BINS, (trial + 1) * TRIAL_BINS)
        neural_trial = np.ascontiguousarray(neural_binned[:, sl], dtype=np.float32)
        input_trial = np.ascontiguousarray(elapsed_time[sl][None, :], dtype=np.float32)
        output_trial = np.ascontiguousarray(labels[sl][None, :], dtype=np.int64)
        if neural_trial.shape != (n_neurons, TRIAL_BINS):
            raise AssertionError("Neural trial shape mismatch")
        if input_trial.shape != (1, TRIAL_BINS) or output_trial.shape != (1, TRIAL_BINS):
            raise AssertionError("Input/output trial shape mismatch")
        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)

    plot_path = None
    if show_processing:
        plot_path = make_processing_plot(
            ref,
            neural_binned,
            trace_example,
            aligned_motion,
            motion_info,
            motion_binned,
            quantile_edges,
            labels,
        )

    class_counts = np.bincount(labels, minlength=5)
    total_seconds = time.perf_counter() - started
    session_data = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "brain_region_idx": np.zeros(n_neurons, dtype=np.int64),
    }
    session_info = {
        "session_id": f"{ref.subject}/{ref.session_id}",
        "subject": ref.subject,
        "date": ref.session_id.removesuffix("_a"),
        "n_neurons": n_neurons,
        "native_frames": n_frames,
        "native_duration_seconds": n_frames / NATIVE_FS_HZ,
        "original_motion_frames": len(np.asarray(motion_info["raw"])),
        "interpolated_motion_frames": int(motion_info["missing_frames"]),
        "timestamp_gap_count": int(motion_info["timestamp_gap_count"]),
        "suite2p_badframes": int(np.count_nonzero(ops["badframes"])),
        "n_trials": n_trials,
        "quantile_edges": quantile_edges.astype(float).tolist(),
        "class_counts": class_counts.astype(int).tolist(),
        "processing_plot": None if plot_path is None else plot_path.name,
        "neural_processing_seconds": neural_seconds,
        "total_processing_seconds": total_seconds,
    }
    print(
        f"Converted {session_info['session_id']}: {n_neurons} neurons, "
        f"{n_frames} native frames -> {n_trials} trials x {TRIAL_BINS} bins; "
        f"motion missing={session_info['interpolated_motion_frames']}; "
        f"neural={neural_seconds:.2f}s total={total_seconds:.2f}s",
        flush=True,
    )
    return session_data, session_info


def build_dataset(sample: bool, show_processing: bool) -> dict:
    subjects, session_refs = discover_sessions()
    if sample:
        session_refs = session_refs[:2]

    neural: list[list[np.ndarray]] = []
    inputs: list[list[np.ndarray]] = []
    outputs: list[list[np.ndarray]] = []
    brain_region_idx: list[np.ndarray] = []
    subject_idx: list[int] = []
    session_info: list[dict] = []

    for index, ref in enumerate(session_refs):
        converted, info = convert_session(
            ref, show_processing=show_processing and index < 2
        )
        neural.append(converted["neural"])
        inputs.append(converted["input"])
        outputs.append(converted["output"])
        brain_region_idx.append(converted["brain_region_idx"])
        subject_idx.append(subjects.index(ref.subject))
        session_info.append(info)

    total_trials = sum(len(x) for x in neural)
    total_timepoints = sum(a.shape[1] for session in neural for a in session)
    if not sample:
        expected_neurons = 20445
        expected_unique_tracks = 2998
        if len(neural) != 41 or total_trials != 1090 or total_timepoints != 196200:
            raise AssertionError(
                f"Full-data totals mismatch: sessions={len(neural)}, "
                f"trials={total_trials}, timepoints={total_timepoints}"
            )
        if sum(x[0].shape[0] for x in neural) != expected_neurons:
            raise AssertionError("Session-neuron total mismatch")
        unique_tracks = sum(
            neural[next(i for i, r in enumerate(session_refs) if r.subject == subject)][0].shape[0]
            for subject in subjects
        )
        if unique_tracks != expected_unique_tracks:
            raise AssertionError("Longitudinal-track total mismatch")

    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["barrel cortex (S1), layer 2/3"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time elapsed from session start (s)"],
        "output_names": ["motion energy percentile bin"],
        "output_values": [[
            "0-20% (lowest)",
            "20-40%",
            "40-60%",
            "60-80%",
            "80-100% (highest)",
        ]],
        "metadata": {
            "task_description": (
                "Decode session-specific motion-energy quintile from population "
                "activity during spontaneous behavior."
            ),
            "time_bin_size": TIME_BIN_MS,
            "temporal_alignment_event": (
                "start of each non-overlapping 60-second segment"
            ),
            "off_start": 0.0,
            "off_end": 60.0,
            "trial_duration_seconds": TRIAL_SECONDS,
            "native_sampling_rate_hz": NATIVE_FS_HZ,
            "converted_sampling_rate_hz": OUTPUT_FS_HZ,
            "temporal_downsampling": (
                "non-overlapping means of 10 consecutive native timestamps"
            ),
            "neural_representation": (
                "Suite2p-style baseline-subtracted fluorescence: "
                "F - 0.7*Fneu minus maximin baseline (sigma=10 frames, "
                "window=60 s), followed by 10-frame means"
            ),
            "motion_alignment": (
                "Direct frame alignment; sessions shorter than neural data were "
                "expanded at timestamp-identified gaps by linear interpolation"
            ),
            "motion_discretization": (
                "20th, 40th, 60th, and 80th percentiles of aligned, "
                "10-frame-averaged motion energy, independently per session"
            ),
            "source_brain_region": (
                "Layer 2/3 mouse barrel cortex, 100-200 um below pia"
            ),
            "session_order": "subjects lexical; sessions and trials chronological",
            "session_info": session_info,
            "reference_discrepancies": (
                "The released tracked-cell counts differ from Figure 5 for four "
                "mice, and four mice contain 30-minute rather than the paper's "
                "stated 20-minute sessions; all valid released rows/times are retained."
            ),
        },
    }
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process only two sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    mode = "sample" if args.sample else "full"
    print(f"Starting {mode} conversion", flush=True)
    data = build_dataset(sample=args.sample, show_processing=args.show_processing)
    conversion_seconds = time.perf_counter() - started
    print(f"Conversion computations finished in {conversion_seconds:.2f}s", flush=True)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    with args.outpicklefile.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    save_seconds = time.perf_counter() - save_started
    size_mb = args.outpicklefile.stat().st_size / 1024**2
    print(
        f"Saved {args.outpicklefile} ({size_mb:.1f} MiB) in {save_seconds:.2f}s; "
        f"total elapsed {time.perf_counter() - started:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
