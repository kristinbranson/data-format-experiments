#!/usr/bin/env python3
"""Convert the Track2p developmental barrel-cortex dataset for decoding.

Usage:
    python -u /app/convert_data.py OUTPUT [--full | --sample] [--show-processing]

The paper's neural preprocessing is reproduced from Suite2p/Track2p: neuropil
correction, maximin baseline subtraction, and non-overlapping 10-frame means.
Behavior is repaired only at documented dropped-camera-frame locations, averaged
over the same frames, and discretized into within-session quintiles.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d


APP_DIR = Path("/app")
DATA_DIR = APP_DIR / "data"
RAW_FRAMES_PER_BIN = 10
TRIAL_SECONDS = 60
NEURAL_CHUNK_SIZE = 64
SAMPLE_SESSION_IDS = {
    "jm031/2023-10-22_a",  # 20 min; 116 missing camera frames
    "jm039/2024-05-04_a",  # 30 min; 1 missing camera frame
}


def discover_sessions() -> list[tuple[str, Path]]:
    """Return all dated session directories in subject/date order."""
    found: list[tuple[str, Path]] = []
    for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            plane = session_dir / "suite2p" / "plane0"
            behavior = session_dir / "move_deve"
            required = (
                plane / "F.npy",
                plane / "Fneu.npy",
                plane / "iscell.npy",
                plane / "ops.npy",
                behavior / "motion_energy_glob.npy",
                behavior / "tstamps.npy",
                behavior / "interframe_int.npy",
            )
            if all(path.exists() for path in required):
                found.append((subject_dir.name, session_dir))
    if not found:
        raise RuntimeError(f"No sessions found under {DATA_DIR}")
    return found


def load_ops_parameters(session_dir: Path) -> dict[str, Any]:
    """Load only the Suite2p settings needed for faithful processing."""
    ops_path = session_dir / "suite2p" / "plane0" / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    result = {
        "fs": float(ops["fs"]),
        "nframes": int(ops["nframes"]),
        "neucoeff": float(ops.get("neucoeff", 0.7)),
        "baseline": str(ops.get("baseline", "maximin")),
        "win_baseline": float(ops.get("win_baseline", 60.0)),
        "sig_baseline": float(ops.get("sig_baseline", 10.0)),
        "prctile_baseline": float(ops.get("prctile_baseline", 8.0)),
        "nchannels": int(ops.get("nchannels", 1)),
        "nplanes": int(ops.get("nplanes", 1)),
        "badframes": int(np.count_nonzero(ops.get("badframes", []))),
    }
    del ops
    return result


def baseline_correct_and_bin(
    session_dir: Path,
    ops: dict[str, Any],
    collect_diagnostics: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply Suite2p maximin baseline correction and 10-frame averaging.

    This mirrors ``DataManagement.F_processing`` in the reference repository,
    while using the actual Suite2p ``neucoeff`` stored in ops (0.7 here).
    Neurons are processed in chunks to keep peak memory bounded.
    """
    plane = session_dir / "suite2p" / "plane0"
    fluorescence = np.load(plane / "F.npy", mmap_mode="r")
    neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
    if fluorescence.shape != neuropil.shape:
        raise ValueError(f"F/Fneu shape mismatch in {session_dir}")
    n_neurons, n_frames = fluorescence.shape
    if n_frames != ops["nframes"]:
        raise ValueError(
            f"ops nframes={ops['nframes']} but F has {n_frames} in {session_dir}"
        )
    if n_frames % RAW_FRAMES_PER_BIN:
        raise ValueError(f"Frame count is not divisible by 10 in {session_dir}")

    n_bins = n_frames // RAW_FRAMES_PER_BIN
    binned = np.empty((n_neurons, n_bins), dtype=np.float32)
    diagnostics: dict[str, np.ndarray] = {}
    baseline_mode = ops["baseline"]
    baseline_window = max(1, int(ops["win_baseline"] * ops["fs"]))
    sigma = ops["sig_baseline"]
    neucoeff = np.float32(ops["neucoeff"])

    for start in range(0, n_neurons, NEURAL_CHUNK_SIZE):
        stop = min(start + NEURAL_CHUNK_SIZE, n_neurons)
        raw_f = np.asarray(fluorescence[start:stop], dtype=np.float32)
        raw_fneu = np.asarray(neuropil[start:stop], dtype=np.float32)
        corrected = raw_f - neucoeff * raw_fneu

        if baseline_mode == "maximin":
            flow = gaussian_filter(corrected, sigma=(0.0, sigma))
            flow = minimum_filter1d(flow, size=baseline_window, axis=1)
            flow = maximum_filter1d(flow, size=baseline_window, axis=1)
        elif baseline_mode == "constant":
            smoothed = gaussian_filter(corrected, sigma=(0.0, sigma))
            flow = np.full_like(corrected, np.amin(smoothed))
        elif baseline_mode == "constant_prctile":
            flow = np.percentile(
                corrected, ops["prctile_baseline"], axis=1, keepdims=True
            ).astype(np.float32)
        else:
            flow = np.zeros((corrected.shape[0], 1), dtype=np.float32)

        processed = corrected - flow
        binned[start:stop] = processed.reshape(
            stop - start, n_bins, RAW_FRAMES_PER_BIN
        ).mean(axis=2)

        if collect_diagnostics and start == 0:
            nplot_frames = min(n_frames, 6000)
            nplot_neurons = min(3, stop - start)
            nplot_bins = nplot_frames // RAW_FRAMES_PER_BIN
            diagnostics = {
                "raw_f": raw_f[:nplot_neurons, :nplot_frames].copy(),
                "raw_fneu": raw_fneu[:nplot_neurons, :nplot_frames].copy(),
                "neuropil_corrected": corrected[
                    :nplot_neurons, :nplot_frames
                ].copy(),
                "baseline": np.broadcast_to(flow, corrected.shape)[
                    :nplot_neurons, :nplot_frames
                ].copy(),
                "baseline_corrected": processed[
                    :nplot_neurons, :nplot_frames
                ].copy(),
                "neural_binned": binned[
                    :nplot_neurons, :nplot_bins
                ].copy(),
            }

    if not np.isfinite(binned).all():
        raise ValueError(f"Non-finite processed neural values in {session_dir}")
    return binned, diagnostics


def repair_motion(
    session_dir: Path, n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align motion to neural frames and interpolate documented missing frames.

    Observed frame positions are recovered by rounding each timestamp interval
    relative to that session's median interval. This operation is used only when
    the behavior array is shorter than the neural array. It must infer exactly
    the same number of missing frames as the source length deficit.
    """
    behavior_dir = session_dir / "move_deve"
    raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
    timestamps = np.load(behavior_dir / "tstamps.npy")
    intervals = np.load(behavior_dir / "interframe_int.npy")
    if raw_motion.ndim != 1 or timestamps.shape != raw_motion.shape:
        raise ValueError(f"Invalid motion/timestamp shapes in {session_dir}")
    if intervals.shape != (raw_motion.size - 1,):
        raise ValueError(f"Invalid interframe interval shape in {session_dir}")
    if not np.allclose(intervals, np.diff(timestamps), rtol=1e-10, atol=1e-12):
        raise ValueError(f"interframe_int != diff(tstamps) in {session_dir}")
    if raw_motion.size > n_frames:
        raise ValueError(f"More behavior frames than neural frames in {session_dir}")

    raw_float = raw_motion.astype(np.float64)
    missing_mask = np.zeros(n_frames, dtype=bool)
    if raw_motion.size == n_frames:
        return raw_float, missing_mask, raw_float.copy(), np.arange(n_frames)

    median_interval = float(np.median(intervals))
    frame_steps = np.maximum(np.rint(intervals / median_interval).astype(np.int64), 1)
    observed_idx = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(frame_steps, dtype=np.int64))
    )
    deficit = n_frames - raw_motion.size
    inferred = int(np.sum(frame_steps - 1))
    if inferred != deficit or observed_idx[-1] != n_frames - 1:
        raise ValueError(
            f"Timestamp gaps infer {inferred} missing frames, expected {deficit}, "
            f"in {session_dir}"
        )
    missing_mask[:] = True
    missing_mask[observed_idx] = False
    repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)
    if not np.allclose(repaired[observed_idx], raw_float):
        raise AssertionError(f"Motion repair altered observed values in {session_dir}")
    return repaired, missing_mask, raw_float, observed_idx


def split_trials(array: np.ndarray, bins_per_trial: int) -> list[np.ndarray]:
    """Split a (features,time) array into independent contiguous trial arrays."""
    if array.ndim != 2 or array.shape[1] % bins_per_trial:
        raise ValueError(f"Cannot split array of shape {array.shape}")
    return [
        array[:, start : start + bins_per_trial].copy()
        for start in range(0, array.shape[1], bins_per_trial)
    ]


def convert_session(
    subject: str,
    session_dir: Path,
    collect_diagnostics: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any], dict[str, Any]]:
    """Convert one continuous recording into 60-second decoder trials."""
    ops = load_ops_parameters(session_dir)
    if ops["fs"] <= 0 or ops["nplanes"] != 1:
        raise ValueError(f"Unexpected acquisition parameters in {session_dir}: {ops}")
    frames_per_trial_float = TRIAL_SECONDS * ops["fs"]
    if not float(frames_per_trial_float).is_integer():
        raise ValueError("Trial duration is not an integer number of frames")
    frames_per_trial = int(frames_per_trial_float)
    if frames_per_trial % RAW_FRAMES_PER_BIN:
        raise ValueError("Trial frame count is not divisible by temporal bin size")
    bins_per_trial = frames_per_trial // RAW_FRAMES_PER_BIN

    t_neural = time.perf_counter()
    neural_binned, neural_diag = baseline_correct_and_bin(
        session_dir, ops, collect_diagnostics
    )
    neural_seconds = time.perf_counter() - t_neural
    n_neurons, n_bins = neural_binned.shape
    n_frames = n_bins * RAW_FRAMES_PER_BIN
    if n_frames % frames_per_trial:
        raise ValueError(f"Session does not divide into complete 60-s trials: {session_dir}")

    repaired_motion, missing_mask, raw_motion, observed_idx = repair_motion(
        session_dir, n_frames
    )
    motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
    quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
    if np.unique(quantile_edges).size != 4:
        raise ValueError(f"Non-distinct motion quintile thresholds in {session_dir}")
    motion_classes = np.searchsorted(
        quantile_edges, motion_binned, side="right"
    ).astype(np.int64)
    fractions = np.bincount(motion_classes, minlength=5) / motion_classes.size
    if not np.allclose(fractions, 0.2, atol=1.0 / motion_classes.size):
        raise AssertionError(f"Unexpected quintile fractions {fractions} in {session_dir}")

    frame_centers_s = (
        np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
        + (RAW_FRAMES_PER_BIN - 1) / 2.0
    ) / ops["fs"]
    input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
    output_binned = motion_classes[np.newaxis, :]

    neural_trials = split_trials(neural_binned, bins_per_trial)
    input_trials = split_trials(input_binned, bins_per_trial)
    output_trials = split_trials(output_binned, bins_per_trial)
    if not (
        len(neural_trials) == len(input_trials) == len(output_trials) >= 2
    ):
        raise AssertionError(f"Trial count mismatch in {session_dir}")
    for neural, decoder_input, output in zip(
        neural_trials, input_trials, output_trials
    ):
        if not (
            neural.shape == (n_neurons, bins_per_trial)
            and decoder_input.shape == (1, bins_per_trial)
            and output.shape == (1, bins_per_trial)
        ):
            raise AssertionError(f"Trial shape mismatch in {session_dir}")

    session_id = f"{subject}/{session_dir.name}"
    info = {
        "session_id": session_id,
        "subject": subject,
        "date": session_dir.name.removesuffix("_a"),
        "source_frames": n_frames,
        "source_motion_frames": int(raw_motion.size),
        "missing_motion_frames": int(np.count_nonzero(missing_mask)),
        "sampling_rate_hz": ops["fs"],
        "duration_seconds_nominal": n_frames / ops["fs"],
        "n_trials": len(neural_trials),
        "n_neurons": n_neurons,
        "suite2p_badframes": ops["badframes"],
        "motion_quintile_edges": [float(x) for x in quantile_edges],
        "motion_class_fractions": [float(x) for x in fractions],
        "neural_processing_seconds": neural_seconds,
    }
    diagnostics: dict[str, Any] = {}
    if collect_diagnostics:
        diagnostics = {
            **neural_diag,
            "raw_motion": raw_motion,
            "observed_idx": observed_idx,
            "repaired_motion": repaired_motion,
            "missing_mask": missing_mask,
            "motion_binned": motion_binned,
            "motion_classes": motion_classes,
            "quantile_edges": quantile_edges,
            "elapsed_time": input_binned[0],
            "fs": ops["fs"],
        }
    return neural_trials, input_trials, output_trials, info, diagnostics


def plot_processing(session_id: str, diagnostics: dict[str, Any]) -> Path:
    """Plot every temporal processing stage for one session."""
    safe_id = session_id.replace("/", "_")
    output_path = APP_DIR / f"processing_{safe_id}.png"
    fs = diagnostics["fs"]
    raw_f = diagnostics["raw_f"]
    n_frames = raw_f.shape[1]
    raw_t = np.arange(n_frames) / fs
    bin_t = diagnostics["elapsed_time"][: diagnostics["neural_binned"].shape[1]]

    fig, axes = plt.subplots(6, 1, figsize=(16, 20), constrained_layout=True)
    axes[0].plot(raw_t, raw_f[0], lw=0.5, label="F")
    axes[0].plot(raw_t, diagnostics["raw_fneu"][0], lw=0.5, label="Fneu")
    axes[0].set_title("1. Raw fluorescence and neuropil")
    axes[0].legend(loc="upper right")

    axes[1].plot(raw_t, diagnostics["neuropil_corrected"][0], lw=0.5, label="F - 0.7 Fneu")
    axes[1].plot(raw_t, diagnostics["baseline"][0], lw=0.7, label="maximin baseline")
    axes[1].set_title("2. Neuropil correction and baseline estimate")
    axes[1].legend(loc="upper right")

    axes[2].plot(raw_t, diagnostics["baseline_corrected"][0], lw=0.45, label="baseline corrected")
    axes[2].plot(bin_t, diagnostics["neural_binned"][0], lw=0.8, label="10-frame mean")
    axes[2].set_title("3. Baseline-corrected neural activity and temporal averaging")
    axes[2].legend(loc="upper right")

    repaired = diagnostics["repaired_motion"]
    motion_t = np.arange(repaired.size) / fs
    axes[3].plot(
        diagnostics["observed_idx"] / fs,
        diagnostics["raw_motion"],
        lw=0.35,
        label="observed motion",
    )
    if diagnostics["missing_mask"].any():
        missing = np.flatnonzero(diagnostics["missing_mask"])
        axes[3].scatter(
            missing / fs,
            repaired[missing],
            s=10,
            color="red",
            label="interpolated missing frames",
            zorder=3,
        )
    axes[3].set_title("4. Frame alignment and sparse missing-frame interpolation")
    axes[3].legend(loc="upper right")

    binned_motion = diagnostics["motion_binned"]
    axes[4].plot(diagnostics["elapsed_time"], binned_motion, lw=0.5, color="0.35")
    for q, value in zip((20, 40, 60, 80), diagnostics["quantile_edges"]):
        axes[4].axhline(value, ls="--", lw=0.8, label=f"q{q}={value:.0f}")
    axes[4].set_title("5. Ten-frame motion mean and within-session quintile thresholds")
    axes[4].legend(ncol=4, fontsize=8)

    axes[5].step(
        diagnostics["elapsed_time"],
        diagnostics["motion_classes"],
        where="mid",
        label="motion quintile (output)",
    )
    elapsed_scaled = 4 * diagnostics["elapsed_time"] / diagnostics["elapsed_time"][-1]
    axes[5].plot(
        diagnostics["elapsed_time"],
        elapsed_scaled,
        alpha=0.7,
        label="session elapsed time, scaled (input)",
    )
    axes[5].set_yticks(range(5))
    axes[5].set_ylim(-0.2, 4.2)
    axes[5].set_title("6. Final aligned decoder input and categorical output")
    axes[5].set_xlabel("Time from session start (s)")
    axes[5].legend(loc="upper right")
    fig.suptitle(f"Conversion processing: {session_id}", fontsize=16)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def validate_converted(data: dict[str, Any]) -> None:
    """Fail early on structural or statistical conversion mistakes."""
    nsessions = len(data["neural"])
    if not (
        nsessions
        == len(data["input"])
        == len(data["output"])
        == len(data["subject_idx"])
        == len(data["brain_region_idx"])
    ):
        raise AssertionError("Top-level session lengths differ")
    for session in range(nsessions):
        ntrials = len(data["neural"][session])
        if ntrials < 2 or not (
            ntrials == len(data["input"][session]) == len(data["output"][session])
        ):
            raise AssertionError(f"Invalid trial lists for session {session}")
        nneurons = data["neural"][session][0].shape[0]
        if data["brain_region_idx"][session].shape != (nneurons,):
            raise AssertionError(f"Invalid brain-region vector for session {session}")
        labels = []
        for neural, decoder_input, output in zip(
            data["neural"][session], data["input"][session], data["output"][session]
        ):
            if neural.dtype != np.float32 or decoder_input.dtype != np.float32:
                raise AssertionError("Neural/input data are not float32")
            if output.dtype != np.int64:
                raise AssertionError("Output data are not int64")
            if not (
                neural.ndim == decoder_input.ndim == output.ndim == 2
                and neural.shape[1] == decoder_input.shape[1] == output.shape[1]
            ):
                raise AssertionError("Trial time dimensions differ")
            if not (
                np.isfinite(neural).all()
                and np.isfinite(decoder_input).all()
                and np.isfinite(output).all()
            ):
                raise AssertionError("Converted arrays contain NaN/Inf")
            labels.append(output.ravel())
        labels_concat = np.concatenate(labels)
        if not np.array_equal(np.unique(labels_concat), np.arange(5)):
            raise AssertionError(f"Session {session} does not contain labels 0..4")
        if not np.allclose(
            np.bincount(labels_concat, minlength=5) / labels_concat.size, 0.2
        ):
            raise AssertionError(f"Session {session} quintiles are imbalanced")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full", action="store_true", help="process all sessions (default)")
    modes.add_argument("--sample", action="store_true", help="process two representative sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<session_id>.png for up to two sessions",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    all_sessions = discover_sessions()
    if args.sample:
        selected = [
            item
            for item in all_sessions
            if f"{item[0]}/{item[1].name}" in SAMPLE_SESSION_IDS
        ]
        if len(selected) != 2:
            raise RuntimeError(f"Expected two sample sessions, found {len(selected)}")
        mode = "sample"
    else:
        selected = all_sessions
        mode = "full"
    print(f"Mode: {mode}; processing {len(selected)} of {len(all_sessions)} sessions")

    subjects = sorted({subject for subject, _ in selected})
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
    neural_all: list[list[np.ndarray]] = []
    input_all: list[list[np.ndarray]] = []
    output_all: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict[str, Any]] = []

    for index, (subject, session_dir) in enumerate(selected):
        session_id = f"{subject}/{session_dir.name}"
        session_started = time.perf_counter()
        make_plot = args.show_processing and index < 2
        neural, decoder_input, output, info, diagnostics = convert_session(
            subject, session_dir, make_plot
        )
        neural_all.append(neural)
        input_all.append(decoder_input)
        output_all.append(output)
        subject_idx.append(subject_lookup[subject])
        brain_region_idx.append(np.zeros(neural[0].shape[0], dtype=np.int64))
        session_info.append(info)
        if make_plot:
            plot_path = plot_processing(session_id, diagnostics)
            print(f"  saved {plot_path}")
        elapsed = time.perf_counter() - session_started
        print(
            f"[{index + 1:02d}/{len(selected):02d}] {session_id}: "
            f"{len(neural)} trials, {neural[0].shape[0]} neurons, "
            f"{info['missing_motion_frames']} missing motion frames, {elapsed:.2f} s",
            flush=True,
        )

    data: dict[str, Any] = {
        "neural": neural_all,
        "input": input_all,
        "output": output_all,
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int64),
        "brain_regions": ["barrel cortex (S1), layer 2/3"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["session_elapsed_time_s"],
        "output_names": ["motion_energy_quintile"],
        "output_values": [[
            "lowest motion (0-20%)",
            "low motion (20-40%)",
            "middle motion (40-60%)",
            "high motion (60-80%)",
            "highest motion (80-100%)",
        ]],
        "metadata": {
            "task_description": (
                "Decode frame-synchronous whole-body motion-energy quintile from "
                "baseline-corrected population calcium activity during spontaneous behavior."
            ),
            "time_bin_size": 1000.0 * RAW_FRAMES_PER_BIN / 30.0,
            "temporal_alignment_event": "start of each non-overlapping 60-second block",
            "off_start": 0.0,
            "off_end": 60.0,
            "input_time_reference": "absolute elapsed time from session start at bin centers",
            "trial_duration_seconds": TRIAL_SECONDS,
            "source_sampling_rate_hz": 30.0,
            "temporal_averaging_frames": RAW_FRAMES_PER_BIN,
            "neural_signal": (
                "Suite2p F - neucoeff*Fneu, maximin baseline subtracted, then 10-frame mean"
            ),
            "motion_signal": (
                "global sum of squared pixel differences, missing frames linearly "
                "interpolated, then 10-frame mean"
            ),
            "motion_discretization": (
                "five equal-percentile bins selected independently over each full session"
            ),
            "neuron_curation": (
                "distributed Track2p complete tracks; source iscell probability >0.5"
            ),
            "brain_region": "mouse primary somatosensory barrel cortex, layer 2/3",
            "session_info": session_info,
        },
    }
    validate_converted(data)

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    with args.outpicklefile.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    save_seconds = time.perf_counter() - save_started
    total_seconds = time.perf_counter() - started
    total_trials = sum(len(session) for session in neural_all)
    print(f"Validated {len(selected)} sessions and {total_trials} trials")
    print(
        f"Saved {args.outpicklefile} ({args.outpicklefile.stat().st_size / 2**20:.1f} MiB) "
        f"in {save_seconds:.2f} s"
    )
    print(f"Total conversion time: {total_seconds:.2f} s", flush=True)


if __name__ == "__main__":
    main()
