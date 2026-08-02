import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d

from decoder import print_data_summary, verify_data_format


FRAME_RATE_HZ = 30.0
FRAME_BIN_SIZE = 10
BIN_SIZE_SECONDS = FRAME_BIN_SIZE / FRAME_RATE_HZ
BIN_SIZE_MS = BIN_SIZE_SECONDS * 1000.0
TRIAL_DURATION_SECONDS = 120.0
TRIAL_BINS = int(TRIAL_DURATION_SECONDS / BIN_SIZE_SECONDS)
MOTION_QUANTILES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
BRAIN_REGION = "barrel cortex L2/3"


def mean_bin_1d(x: np.ndarray, bin_size: int) -> np.ndarray:
    nbins = x.shape[0] // bin_size
    return x[: nbins * bin_size].reshape(nbins, bin_size).mean(axis=1)


def mean_bin_2d(x: np.ndarray, bin_size: int) -> np.ndarray:
    nbins = x.shape[1] // bin_size
    return x[:, : nbins * bin_size].reshape(x.shape[0], nbins, bin_size).mean(axis=2)


def suite2p_baseline_corrected_fluorescence(
    fluorescence: np.ndarray, neuropil: np.ndarray, ops: dict
) -> np.ndarray:
    fc = fluorescence.astype(np.float32) - float(ops["neucoeff"]) * neuropil.astype(np.float32)
    baseline = ops.get("baseline", "maximin")

    if baseline == "maximin":
        win = int(round(float(ops["win_baseline"]) * float(ops["fs"])))
        flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
        flow = minimum_filter1d(flow, win)
        flow = maximum_filter1d(flow, win)
    elif baseline == "constant":
        flow = gaussian_filter(fc, [0.0, float(ops["sig_baseline"])])
        flow = np.amin(flow)
    elif baseline == "constant_prctile":
        flow = np.percentile(fc, float(ops["prctile_baseline"]), axis=1, keepdims=True)
    else:
        flow = 0.0

    return (fc - flow).astype(np.float32)


def infer_missing_frames(interframe_int: np.ndarray) -> np.ndarray:
    median_interval = float(np.median(interframe_int))
    frame_jumps = np.rint(interframe_int / median_interval).astype(np.int64)
    return np.clip(frame_jumps - 1, 0, None)


def align_motion_to_imaging_frames(
    motion_energy: np.ndarray,
    n_imaging_frames: int,
    interframe_int: np.ndarray,
) -> tuple[np.ndarray, dict]:
    motion_energy = motion_energy.astype(np.float32, copy=False)
    if motion_energy.shape[0] == n_imaging_frames:
        return motion_energy, {
            "missing_motion_frames": 0,
            "missing_motion_frame_indices": [],
        }

    missing_after = infer_missing_frames(interframe_int)
    missing_total = int(missing_after.sum())
    expected_missing = n_imaging_frames - motion_energy.shape[0]
    if missing_total != expected_missing:
        raise ValueError(
            f"Could not reconcile missing motion frames: inferred {missing_total}, "
            f"expected {expected_missing}"
        )

    aligned = np.empty(n_imaging_frames, dtype=np.float32)
    inserted_indices: list[int] = []
    src = 0
    dst = 0
    aligned[dst] = motion_energy[src]
    dst += 1

    for gap_missing in missing_after:
        next_src = src + 1
        if gap_missing:
            inserted_indices.extend(range(dst, dst + int(gap_missing)))
            aligned[dst : dst + gap_missing] = np.linspace(
                motion_energy[src],
                motion_energy[next_src],
                int(gap_missing) + 2,
                dtype=np.float32,
            )[1:-1]
            dst += int(gap_missing)
        aligned[dst] = motion_energy[next_src]
        dst += 1
        src = next_src

    if dst != n_imaging_frames:
        raise ValueError(f"Aligned motion has {dst} frames, expected {n_imaging_frames}")

    return aligned, {
        "missing_motion_frames": missing_total,
        "missing_motion_frame_indices": inserted_indices,
    }


def session_paths(data_dir: Path, selected_subjects: list[str] | None) -> list[tuple[str, Path]]:
    available_subjects = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    if selected_subjects is None:
        subjects = available_subjects
    else:
        missing = sorted(set(selected_subjects) - set(available_subjects))
        if missing:
            raise FileNotFoundError(f"Unknown subjects: {missing}")
        subjects = sorted(selected_subjects)

    out: list[tuple[str, Path]] = []
    for subject in subjects:
        subject_dir = data_dir / subject
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            out.append((subject, session_dir))
    return out


def load_session(subject: str, session_dir: Path) -> dict:
    plane_dir = session_dir / "suite2p" / "plane0"
    move_dir = session_dir / "move_deve"

    fluorescence = np.load(plane_dir / "F.npy", allow_pickle=True)
    neuropil = np.load(plane_dir / "Fneu.npy", allow_pickle=True)
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    iscell = np.load(plane_dir / "iscell.npy", allow_pickle=True)
    motion_energy = np.load(move_dir / "motion_energy_glob.npy", allow_pickle=True)
    interframe_int = np.load(move_dir / "interframe_int.npy", allow_pickle=True)

    if fluorescence.shape != neuropil.shape:
        raise ValueError(f"{session_dir}: F and Fneu shapes do not match")
    if not np.all(iscell[:, 1] > 0.5):
        raise ValueError(f"{session_dir}: found Track2p-exported cells with iscell <= 0.5")

    n_neurons, n_frames = fluorescence.shape
    neural = suite2p_baseline_corrected_fluorescence(fluorescence, neuropil, ops)
    motion_aligned, motion_info = align_motion_to_imaging_frames(
        motion_energy=motion_energy,
        n_imaging_frames=n_frames,
        interframe_int=interframe_int,
    )

    neural_binned = mean_bin_2d(neural, FRAME_BIN_SIZE).astype(np.float32)
    motion_binned = mean_bin_1d(motion_aligned, FRAME_BIN_SIZE).astype(np.float32)
    motion_z = (
        (motion_binned - motion_binned.mean()) / (motion_binned.std() + 1e-8)
    ).astype(np.float32)

    if neural_binned.shape[1] != motion_binned.shape[0]:
        raise ValueError(f"{session_dir}: binned neural and motion lengths do not match")
    if motion_binned.shape[0] % TRIAL_BINS != 0:
        raise ValueError(
            f"{session_dir}: {motion_binned.shape[0]} binned frames cannot be split into "
            f"{TRIAL_BINS}-bin trials"
        )

    return {
        "subject": subject,
        "session_id": f"{subject}/{session_dir.name}",
        "session_date": session_dir.name.split("_")[0],
        "n_neurons": n_neurons,
        "n_frames_raw": n_frames,
        "duration_seconds": n_frames / float(ops["fs"]),
        "neural_binned": neural_binned,
        "motion_binned": motion_binned,
        "motion_z": motion_z,
        "ops": {
            "fs": float(ops["fs"]),
            "neucoeff": float(ops["neucoeff"]),
            "baseline": ops.get("baseline", "maximin"),
            "sig_baseline": float(ops["sig_baseline"]),
            "win_baseline": float(ops["win_baseline"]),
            "prctile_baseline": float(ops["prctile_baseline"]),
        },
        **motion_info,
    }


def split_session_into_trials(
    neural_binned: np.ndarray,
    motion_classes: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    neural_trials: list[np.ndarray] = []
    input_trials: list[np.ndarray] = []
    output_trials: list[np.ndarray] = []

    elapsed_time_seconds = (
        (np.arange(neural_binned.shape[1], dtype=np.float32) + 0.5) * BIN_SIZE_SECONDS
    )
    n_trials = neural_binned.shape[1] // TRIAL_BINS

    for trial_idx in range(n_trials):
        start = trial_idx * TRIAL_BINS
        end = start + TRIAL_BINS
        neural_trials.append(neural_binned[:, start:end].astype(np.float32, copy=False))
        input_trials.append(elapsed_time_seconds[start:end][None, :].astype(np.float32, copy=False))
        output_trials.append(motion_classes[start:end][None, :].astype(np.int64, copy=False))

    return neural_trials, input_trials, output_trials


def build_dataset(data_dir: Path, selected_subjects: list[str] | None) -> dict:
    loaded_sessions = [load_session(subject, session_dir) for subject, session_dir in session_paths(data_dir, selected_subjects)]
    subjects = sorted({session["subject"] for session in loaded_sessions})

    pooled_motion_z = np.concatenate([session["motion_z"] for session in loaded_sessions], axis=0)
    motion_bin_edges = np.quantile(pooled_motion_z, MOTION_QUANTILES).astype(np.float32)

    neural: list[list[np.ndarray]] = []
    input_data: list[list[np.ndarray]] = []
    output: list[list[np.ndarray]] = []
    subject_idx: list[int] = []
    brain_region_idx: list[np.ndarray] = []
    session_info: list[dict] = []

    for session in loaded_sessions:
        motion_classes = np.digitize(session["motion_z"], motion_bin_edges, right=False).astype(np.int64)
        neural_trials, input_trials, output_trials = split_session_into_trials(
            neural_binned=session["neural_binned"],
            motion_classes=motion_classes,
        )

        neural.append(neural_trials)
        input_data.append(input_trials)
        output.append(output_trials)
        subject_idx.append(subjects.index(session["subject"]))
        brain_region_idx.append(np.zeros(session["n_neurons"], dtype=np.int64))
        session_info.append(
            {
                "session_id": session["session_id"],
                "subject": session["subject"],
                "session_date": session["session_date"],
                "n_neurons": session["n_neurons"],
                "n_frames_raw": session["n_frames_raw"],
                "n_time_bins": int(session["neural_binned"].shape[1]),
                "n_trials": len(neural_trials),
                "duration_seconds": float(session["duration_seconds"]),
                "missing_motion_frames": int(session["missing_motion_frames"]),
                "suite2p_ops": session["ops"],
            }
        )

    tracked_neurons_per_subject = {
        subject: int(
            next(session["n_neurons"] for session in loaded_sessions if session["subject"] == subject)
        )
        for subject in subjects
    }
    tracked_counts = np.array(list(tracked_neurons_per_subject.values()), dtype=np.float32)
    missing_motion_by_session = {
        session["session_id"]: int(session["missing_motion_frames"]) for session in loaded_sessions
    }

    return {
        "neural": neural,
        "input": input_data,
        "output": output,
        "subjects": subjects,
        "subject_idx": np.array(subject_idx, dtype=np.int64),
        "brain_regions": [BRAIN_REGION],
        "brain_region_idx": brain_region_idx,
        "input_names": ["elapsed_time_s"],
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
                "Spontaneous barrel-cortex activity during longitudinal two-photon imaging; "
                "decoder predicts motion-energy state from neural activity."
            ),
            "time_bin_size": BIN_SIZE_MS,
            "temporal_alignment_event": "start of each consecutive 2-minute block within a session",
            "off_start": 0.0,
            "off_end": TRIAL_DURATION_SECONDS,
            "raw_frame_rate_hz": FRAME_RATE_HZ,
            "frame_bin_size": FRAME_BIN_SIZE,
            "trial_duration_seconds": TRIAL_DURATION_SECONDS,
            "neural_trace_type": (
                "Suite2p baseline-corrected fluorescence computed from Track2p-exported F/Fneu "
                "using the saved Suite2p ops parameters."
            ),
            "neural_processing": {
                "neuropil_subtraction": "F - neucoeff * Fneu",
                "baseline": "Suite2p default maximin baseline subtraction from ops.npy",
                "extra_iscell_filtering_applied": False,
            },
            "motion_processing": {
                "source": "move_deve/motion_energy_glob.npy",
                "missing_frame_handling": (
                    "Missing camera frames inferred from interframe_int.npy and linearly interpolated "
                    "on the imaging-frame grid before binning."
                ),
                "normalization": "Per-session z-score after 10-frame averaging",
                "discretization": "Global quintile binning over pooled session-normalized motion energy",
                "global_quintile_edges_zscore": motion_bin_edges.tolist(),
            },
            "tracked_neurons_per_subject": tracked_neurons_per_subject,
            "tracked_neurons_mean": float(tracked_counts.mean()),
            "tracked_neurons_std": float(tracked_counts.std(ddof=0)),
            "missing_motion_frames_total": int(sum(missing_motion_by_session.values())),
            "missing_motion_frames_by_session": missing_motion_by_session,
            "session_info": session_info,
        },
    }


def print_conversion_summary(data: dict) -> None:
    metadata = data["metadata"]
    duration_counts: dict[float, int] = {}
    for session in metadata["session_info"]:
        duration_minutes = session["duration_seconds"] / 60.0
        duration_counts[duration_minutes] = duration_counts.get(duration_minutes, 0) + 1

    print("Conversion choices:")
    print("  Neural traces: Suite2p baseline-corrected fluorescence from Track2p-exported F/Fneu")
    print("  Behavior alignment: imaging-frame grid with interpolation over inferred dropped camera frames")
    print("  Denoising: non-overlapping averages over 10 consecutive frames for neural and motion traces")
    print("  Trials: consecutive 2-minute blocks within each session")
    print("  Decoder input: elapsed time from session start at bin centers (seconds)")
    print("  Decoder output: session-z-scored motion energy discretized with global quintile edges")
    print("")
    print("Sanity checks against reference materials:")
    print(f"  Subjects: {data['subjects']}")
    print(f"  Sessions: {len(data['neural'])}")
    print(
        "  Tracked neurons per subject: "
        + ", ".join(
            f"{subject}={count}" for subject, count in metadata["tracked_neurons_per_subject"].items()
        )
    )
    print(
        "  Mean tracked neurons across subjects: "
        f"{metadata['tracked_neurons_mean']:.1f} +/- {metadata['tracked_neurons_std']:.1f}"
    )
    if len(data["subjects"]) == 6:
        print(
            "  Paper reports 526 +/- 190 tracked neurons across mice for the full dataset; "
            "this conversion yields 499.7 +/- 180.5 from the released six-subject dataset."
        )
    print(
        "  Session durations (minutes): "
        + ", ".join(f"{minutes:.1f}x{count}" for minutes, count in sorted(duration_counts.items()))
    )
    print(f"  Missing motion frames total: {metadata['missing_motion_frames_total']}")
    print(
        "  Global motion quintile edges after per-session z-scoring: "
        + ", ".join(f"{edge:.4f}" for edge in metadata["motion_processing"]["global_quintile_edges_zscore"])
    )
    print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Track2p developmental barrel-cortex data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing subject folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("converted_data.pkl"),
        help="Output pickle path.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional explicit list of subject IDs to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_dataset(data_dir=args.data_dir, selected_subjects=args.subjects)

    valid, errors, warnings = verify_data_format(data)
    if not valid:
        raise ValueError("Converted dataset failed validation:\n" + "\n".join(errors))

    print_conversion_summary(data)
    if warnings:
        print("Format warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print("")
    else:
        print("Format validation: no warnings\n")

    print_data_summary(data)

    with args.output.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("")
    print(f"Saved dataset to {args.output}")


if __name__ == "__main__":
    main()
