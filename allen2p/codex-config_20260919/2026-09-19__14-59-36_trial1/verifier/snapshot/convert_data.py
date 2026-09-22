#!/usr/bin/env python3
"""Convert Allen VisualBehavior 2P NWBs to decoder-compatible trials.

Usage
-----
python -u /app/convert_data.py <outpicklefile> [--full|--sample]
                                            [--show-processing]

The conversion intentionally reads the released, already-QC'd L0 calcium
events directly from NWB.  All behavioral streams are aligned by timestamp to
the microscope (ophys) timestamps; sample indices from different clocks are
never equated.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


APP_ROOT = Path("/app")
DATA_ROOT = APP_ROOT / "data" / "visual-behavior-ophys-1.1.0"
NWB_ROOT = DATA_ROOT / "behavior_ophys_experiments"
EXPERIMENT_TABLE = DATA_ROOT / "project_metadata" / "ophys_experiment_table.csv"

MISSING_EYE_EXPERIMENTS = {795953296, 806456687, 833631914}
IMAGE_VALUES = [
    "gray",
    "im000",
    "im031",
    "im035",
    "im045",
    "im054",
    "im061",
    "im062",
    "im063",
    "im065",
    "im066",
    "im069",
    "im073",
    "im075",
    "im077",
    "im085",
    "im106",
]
IMAGE_TO_CODE = {name: idx for idx, name in enumerate(IMAGE_VALUES)}
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
QUINTILE_NAMES = ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full", action="store_true", help="process all sessions (default)")
    modes.add_argument("--sample", action="store_true", help="process two representative sessions")
    parser.add_argument(
        "--show-processing",
        action="store_true",
        help="save processing_<experiment_id>.png for up to two sessions",
    )
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    """Decode an NWB variable-length string dataset to a unicode array."""
    return np.asarray(
        [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values]
    )


def eligible_experiments(sample: bool) -> pd.DataFrame:
    """Return exact-project active sessions with all required source streams."""
    table = pd.read_csv(EXPERIMENT_TABLE)
    selected = table[
        table["project_code"].eq("VisualBehavior")
        & ~table["passive"].astype(bool)
        & ~table["ophys_experiment_id"].isin(MISSING_EYE_EXPERIMENTS)
    ].copy()
    selected["nwb_path"] = selected["ophys_experiment_id"].map(
        lambda eid: NWB_ROOT / f"behavior_ophys_experiment_{int(eid)}.nwb"
    )
    missing_files = selected.loc[~selected["nwb_path"].map(Path.exists)]
    if not missing_files.empty:
        raise FileNotFoundError(
            "Missing selected NWBs: "
            + ", ".join(map(str, missing_files["ophys_experiment_id"].tolist()))
        )
    selected.sort_values("ophys_experiment_id", inplace=True)

    if sample:
        # The first and second active sessions of mouse 403491 use image sets A
        # and B, respectively, so the two-session sample exercises all 16 image
        # codes as well as both GCaMP6f event matrices.
        preferred = selected[selected["mouse_id"].eq(403491)]
        if len(preferred) >= 2:
            selected = preferred.iloc[[0, -1]].copy()
        else:
            selected = selected.iloc[:2].copy()
    return selected


def find_image_presentations(intervals: h5py.Group) -> h5py.Group:
    candidates = [
        group
        for name, group in intervals.items()
        if name != "trials" and isinstance(group, h5py.Group) and "image_name" in group
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one natural-image presentation table, found {len(candidates)}")
    return candidates[0]


def quintile_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return value-threshold percentile bins and their six boundary values."""
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Quintile source must be a nonempty finite 1-D array")
    edges = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    codes = np.searchsorted(edges[1:-1], values, side="right").astype(np.int16)
    return codes, edges


def interpolate_finite(
    source_t: np.ndarray, source_x: np.ndarray, target_t: np.ndarray, label: str
) -> np.ndarray:
    good = np.isfinite(source_t) & np.isfinite(source_x)
    if np.count_nonzero(good) < 2:
        raise ValueError(f"Insufficient finite {label} samples")
    t = source_t[good]
    x = source_x[good]
    if np.any(np.diff(t) <= 0):
        order = np.argsort(t, kind="stable")
        t, x = t[order], x[order]
        unique = np.concatenate(([True], np.diff(t) > 0))
        t, x = t[unique], x[unique]
    return np.interp(target_t, t, x)


def presentation_outputs(
    group: h5py.Group, target_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    starts = group["start_time"][:]
    stops = group["stop_time"][:]
    names = decode_strings(group["image_name"][:])
    active = group["active"][:].astype(bool)
    omitted = np.nan_to_num(group["omitted"][:], nan=0.0).astype(bool)
    is_change = np.nan_to_num(group["is_change"][:], nan=0.0).astype(bool)

    order = np.argsort(starts, kind="stable")
    starts, stops = starts[order], stops[order]
    names, active, omitted, is_change = (
        names[order],
        active[order],
        omitted[order],
        is_change[order],
    )

    idx = np.searchsorted(starts, target_t, side="right") - 1
    nonnegative = idx >= 0
    safe_idx = np.maximum(idx, 0)
    known_image = np.fromiter(
        (name in IMAGE_TO_CODE and name != "gray" for name in names[safe_idx]),
        dtype=bool,
        count=len(target_t),
    )
    shown = (
        nonnegative
        & np.isfinite(stops[safe_idx])
        & (target_t < stops[safe_idx])
        & active[safe_idx]
        & ~omitted[safe_idx]
        & known_image
    )
    image = np.zeros(len(target_t), dtype=np.int16)
    if np.any(shown):
        image[shown] = np.fromiter(
            (IMAGE_TO_CODE[name] for name in names[safe_idx[shown]]),
            dtype=np.int16,
            count=np.count_nonzero(shown),
        )
    change = (shown & is_change[safe_idx]).astype(np.int16)
    return image, change


def plot_processing(context: dict, outpath: Path) -> None:
    """Plot all transformations for one representative retained trial."""
    t = context["t"]
    rel_t = t - t[0]
    neural = context["neural"]
    output = context["output"]
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)

    nshow = min(50, neural.shape[0])
    axes[0].imshow(
        neural[:nshow],
        aspect="auto",
        interpolation="nearest",
        extent=[rel_t[0], rel_t[-1], nshow, 0],
        cmap="magma",
    )
    axes[0].set_ylabel("valid cells")
    axes[0].set_title(
        f"Experiment {context['experiment_id']}, raw L0 events sliced on ophys timestamps\n"
        f"raw trial id {context['trial_id']}, outcome={OUTCOME_NAMES[context['outcome']]}"
    )

    axes[1].step(rel_t, output[0], where="post", label="image code")
    axes[1].step(rel_t, output[1] * (len(IMAGE_VALUES) - 1), where="post", label="change (scaled)")
    axes[1].set_ylabel("stimulus")
    axes[1].legend(loc="upper right")

    axes[2].plot(rel_t, context["running"], color="tab:blue", label="interpolated cm/s")
    for edge in context["running_edges"][1:-1]:
        axes[2].axhline(edge, color="gray", alpha=0.35, linewidth=0.8)
    axes[2].set_ylabel("run cm/s")
    axes[2].legend(loc="upper right")

    axes[3].step(rel_t, output[2], where="post", color="tab:blue")
    axes[3].set_ylabel("run quintile")
    axes[3].set_yticks(range(5))

    axes[4].plot(rel_t, context["pupil"], color="tab:green", label="blink-masked/interpolated diameter")
    for edge in context["pupil_edges"][1:-1]:
        axes[4].axhline(edge, color="gray", alpha=0.35, linewidth=0.8)
    axes[4].set_ylabel("pupil px")
    axes[4].legend(loc="upper right")

    axes[5].step(rel_t, output[3], where="post", color="tab:green", label="pupil quintile")
    axes[5].step(rel_t, output[4], where="post", color="tab:red", label="static outcome")
    axes[5].set_ylabel("category")
    axes[5].set_xlabel("seconds from native trial start frame")
    axes[5].set_yticks(range(5))
    axes[5].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def convert_session(row: pd.Series, make_plot: bool) -> tuple[dict, dict | None]:
    eid = int(row["ophys_experiment_id"])
    path = Path(row["nwb_path"])
    t0 = time.perf_counter()
    with h5py.File(path, "r") as nwb:
        raw_subject = nwb["general/subject/subject_id"][()]
        raw_subject = raw_subject.decode() if isinstance(raw_subject, bytes) else str(raw_subject)
        if raw_subject != str(int(row["mouse_id"])):
            raise ValueError(f"Mouse mismatch in experiment {eid}: {raw_subject} vs {row['mouse_id']}")

        trials = nwb["intervals/trials"]
        go = trials["go"][:].astype(bool)
        catch = trials["catch"][:].astype(bool)
        aborted = trials["aborted"][:].astype(bool)
        auto_rewarded = trials["auto_rewarded"][:].astype(bool)
        keep = (go | catch) & ~aborted & ~auto_rewarded
        raw_rows = np.flatnonzero(keep)
        if len(raw_rows) < 2:
            raise ValueError(f"Experiment {eid} has fewer than two eligible trials")

        outcome_flags = np.column_stack([trials[name][:].astype(bool) for name in OUTCOME_NAMES])
        if not np.all(outcome_flags[raw_rows].sum(axis=1) == 1):
            raise ValueError(f"Experiment {eid} has non-exclusive eligible outcomes")
        outcomes = np.argmax(outcome_flags[raw_rows], axis=1).astype(np.int16)

        event_group = nwb["processing/ophys/event_detection"]
        event_data = event_group["data"]
        ophys_t = event_group["timestamps"][:]
        if event_data.shape[0] != len(ophys_t):
            raise ValueError(f"Experiment {eid}: event/timestamp length mismatch")
        if np.any(np.diff(ophys_t) <= 0):
            raise ValueError(f"Experiment {eid}: non-monotonic ophys timestamps")

        starts = trials["start_time"][:][raw_rows]
        stops = trials["stop_time"][:][raw_rows]
        left = np.searchsorted(ophys_t, starts, side="left")
        right = np.searchsorted(ophys_t, stops, side="left")
        if np.any(right <= left):
            raise ValueError(f"Experiment {eid}: empty aligned trial")
        if np.any(left[1:] < right[:-1]):
            raise ValueError(f"Experiment {eid}: overlapping eligible trial frame slices")
        lengths = right - left
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        target_t = np.concatenate([ophys_t[a:b] for a, b in zip(left, right)])

        running = nwb["processing/running/speed"]
        running_aligned = interpolate_finite(
            running["timestamps"][:], running["data"][:], target_t, "running speed"
        )
        running_codes, running_edges = quintile_codes(running_aligned)

        pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
        pupil_area = pupil["area"][:]
        pupil_diameter = np.full(pupil_area.shape, np.nan, dtype=np.float64)
        nonnegative_area = np.isfinite(pupil_area) & (pupil_area >= 0)
        pupil_diameter[nonnegative_area] = 2.0 * np.sqrt(pupil_area[nonnegative_area] / np.pi)
        pupil_aligned = interpolate_finite(
            pupil["timestamps"][:], pupil_diameter, target_t, "pupil diameter"
        )
        pupil_codes, pupil_edges = quintile_codes(pupil_aligned)

        presentations = find_image_presentations(nwb["intervals"])
        image_codes, change_codes = presentation_outputs(presentations, target_t)

        block_left, block_right = int(left.min()), int(right.max())
        event_block = np.empty(
            (block_right - block_left, event_data.shape[1]), dtype=np.float32
        )
        event_data.read_direct(event_block, source_sel=np.s_[block_left:block_right, :])
        if not np.all(np.isfinite(event_block)):
            raise ValueError(f"Experiment {eid}: event data contain NaN/Inf")

        neural_trials: list[np.ndarray] = []
        input_trials: list[np.ndarray] = []
        output_trials: list[np.ndarray] = []
        trial_ids = trials["id"][:][raw_rows].astype(np.int64)
        plot_context = None
        for j, (a, b) in enumerate(zip(left, right)):
            lo, hi = int(offsets[j]), int(offsets[j + 1])
            neural = event_block[a - block_left : b - block_left].T.copy()
            output = np.empty((5, b - a), dtype=np.int16)
            output[0] = image_codes[lo:hi]
            output[1] = change_codes[lo:hi]
            output[2] = running_codes[lo:hi]
            output[3] = pupil_codes[lo:hi]
            output[4].fill(outcomes[j])
            neural_trials.append(neural)
            input_trials.append(np.empty((0, b - a), dtype=np.float32))
            output_trials.append(output)

            if make_plot and j == 0:
                plot_context = {
                    "experiment_id": eid,
                    "trial_id": int(trial_ids[j]),
                    "outcome": int(outcomes[j]),
                    "t": target_t[lo:hi].copy(),
                    "neural": neural,
                    "output": output,
                    "running": running_aligned[lo:hi].copy(),
                    "pupil": pupil_aligned[lo:hi].copy(),
                    "running_edges": running_edges.copy(),
                    "pupil_edges": pupil_edges.copy(),
                }

        # Data-level assertions catch mapping regressions before serialization.
        if sum(x.shape[1] for x in neural_trials) != len(target_t):
            raise AssertionError("Trial frame count mismatch")
        if not all(n.shape[1] == i.shape[1] == o.shape[1] for n, i, o in zip(neural_trials, input_trials, output_trials)):
            raise AssertionError("Within-trial time dimension mismatch")
        if set(np.unique(image_codes)) - set(range(len(IMAGE_VALUES))):
            raise AssertionError("Unknown image output code")
        if set(np.unique(change_codes)) - {0, 1}:
            raise AssertionError("Unknown change output code")
        if np.min(running_codes) != 0 or np.max(running_codes) != 4:
            raise AssertionError("Running quintiles do not span 0..4")
        if np.min(pupil_codes) != 0 or np.max(pupil_codes) != 4:
            raise AssertionError("Pupil quintiles do not span 0..4")

        cell_table = nwb["processing/ophys/image_segmentation/cell_specimen_table"]
        if "valid_roi" in cell_table and not np.all(cell_table["valid_roi"][:]):
            raise ValueError(f"Experiment {eid}: unexpected invalid stored ROI")
        if len(cell_table["id"]) != event_data.shape[1]:
            raise ValueError(f"Experiment {eid}: cell/event column mismatch")
        cell_specimen_ids = (
            cell_table["cell_specimen_id"][:].astype(np.int64).tolist()
            if "cell_specimen_id" in cell_table
            else []
        )

    info = {
        "ophys_experiment_id": eid,
        "ophys_session_id": int(row["ophys_session_id"]),
        "behavior_session_id": int(row["behavior_session_id"]),
        "mouse_id": raw_subject,
        "session_type": str(row["session_type"]),
        "experience_level": str(row["experience_level"]),
        "image_set": str(row["image_set"]),
        "cre_line": str(row["cre_line"]),
        "indicator": str(row["indicator"]),
        "imaging_depth_um": int(row["imaging_depth"]),
        "targeted_structure": str(row["targeted_structure"]),
        "n_neurons": int(event_data.shape[1]),
        "n_trials": len(neural_trials),
        "n_trial_frames": int(len(target_t)),
        "trial_ids": trial_ids.tolist(),
        "cell_specimen_ids": cell_specimen_ids,
        "ophys_median_dt_s": float(np.median(np.diff(ophys_t))),
        "running_quintile_edges_cm_s": running_edges.tolist(),
        "pupil_quintile_edges_px": pupil_edges.tolist(),
        "image_on_fraction": float(np.mean(image_codes != 0)),
        "image_change_fraction": float(np.mean(change_codes)),
        "outcome_trial_counts": {
            name: int(np.count_nonzero(outcomes == idx)) for idx, name in enumerate(OUTCOME_NAMES)
        },
        "conversion_seconds": float(time.perf_counter() - t0),
    }
    converted = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "subject": raw_subject,
        "brain_region_idx": np.zeros(event_data.shape[1], dtype=np.int16),
        "info": info,
    }
    return converted, plot_context


def main() -> None:
    args = parse_args()
    sample = bool(args.sample)
    mode = "sample" if sample else "full"
    selected = eligible_experiments(sample=sample)
    print(f"Mode: {mode}; selected {len(selected)} sessions", flush=True)
    print(
        "Excluded passive replay sessions and eye-absent experiments: "
        + ", ".join(map(str, sorted(MISSING_EYE_EXPERIMENTS))),
        flush=True,
    )

    neural_sessions = []
    input_sessions = []
    output_sessions = []
    session_subjects = []
    brain_region_idx = []
    session_info = []
    conversion_start = time.perf_counter()
    for session_num, (_, row) in enumerate(selected.iterrows(), start=1):
        eid = int(row["ophys_experiment_id"])
        make_plot = args.show_processing and session_num <= 2
        converted, plot_context = convert_session(row, make_plot=make_plot)
        neural_sessions.append(converted["neural"])
        input_sessions.append(converted["input"])
        output_sessions.append(converted["output"])
        session_subjects.append(converted["subject"])
        brain_region_idx.append(converted["brain_region_idx"])
        session_info.append(converted["info"])
        if plot_context is not None:
            plot_path = APP_ROOT / f"processing_{eid}.png"
            plot_processing(plot_context, plot_path)
            print(f"  saved {plot_path}", flush=True)
        elapsed = time.perf_counter() - conversion_start
        mean_s = elapsed / session_num
        eta = mean_s * (len(selected) - session_num)
        print(
            f"[{session_num:3d}/{len(selected)}] {eid}: "
            f"{converted['info']['n_neurons']} neurons, "
            f"{converted['info']['n_trials']} trials, "
            f"{converted['info']['n_trial_frames']} frames; "
            f"session {converted['info']['conversion_seconds']:.2f}s, "
            f"elapsed {elapsed:.1f}s, ETA {eta:.1f}s",
            flush=True,
        )

    subjects = sorted(set(session_subjects), key=int)
    subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
    subject_idx = np.asarray([subject_lookup[x] for x in session_subjects], dtype=np.int32)
    median_dt_ms = float(np.median([x["ophys_median_dt_s"] for x in session_info]) * 1000.0)
    total_trials = sum(len(x) for x in neural_sessions)
    total_neurons = sum(len(x) for x in brain_region_idx)
    total_frames = sum(x["n_trial_frames"] for x in session_info)

    data = {
        "neural": neural_sessions,
        "input": input_sessions,
        "output": output_sessions,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": ["VISp"],
        "brain_region_idx": brain_region_idx,
        "input_names": [],
        "output_names": [
            "image_identity",
            "image_change",
            "running_speed_quintile",
            "pupil_diameter_quintile",
            "trial_outcome",
        ],
        "output_values": [
            IMAGE_VALUES,
            ["no_change", "change"],
            QUINTILE_NAMES,
            QUINTILE_NAMES,
            OUTCOME_NAMES,
        ],
        "metadata": {
            "task_description": (
                "Decode flashed natural-image identity, immediate image change, "
                "running-speed quintile, pupil-diameter quintile, and go/catch "
                "trial outcome from valid-cell L0 calcium event magnitudes during "
                "the active Allen VisualBehavior change-detection task."
            ),
            "time_bin_size": median_dt_ms,
            "temporal_alignment_event": (
                "Native ophys frame timestamps within each SDK trial; trials begin "
                "at the first ophys timestamp >= trial start_time."
            ),
            "off_start": 0.0,
            "off_end": None,
            "source_release": "visual-behavior-ophys-1.1.0",
            "project_code": "VisualBehavior",
            "neural_signal": "unfiltered FastLZero L0 calcium event magnitude",
            "trial_interval_convention": "start_time <= ophys_timestamp < stop_time",
            "trial_filter": "(go OR catch) AND NOT aborted AND NOT auto_rewarded",
            "session_filter": "active behavior, exact project code, required pupil data present",
            "excluded_eye_absent_experiment_ids": sorted(MISSING_EYE_EXPERIMENTS),
            "behavior_alignment": "linear timestamp interpolation to native ophys frames",
            "quintile_definition": (
                "within-session 20/40/60/80 percentiles over retained eligible-trial ophys frames"
            ),
            "pupil_definition": (
                "blink-masked major-axis diameter = 2*sqrt(released pupil_area/pi), in pixels"
            ),
            "image_change_definition": (
                "1 during the on-screen changed-image presentation interval, otherwise 0"
            ),
            "gray_definition": (
                "normal inter-stimulus gray plus omitted-flash intervals and any other non-image time"
            ),
            "n_sessions": len(neural_sessions),
            "n_trials": total_trials,
            "n_session_neurons": total_neurons,
            "n_trial_frames": total_frames,
            "session_info": session_info,
        },
    }

    if not sample:
        expected = (165, 42470, 28821, 11192974, 37)
        observed = (len(neural_sessions), total_trials, total_neurons, total_frames, len(subjects))
        if observed != expected:
            raise AssertionError(f"Full-cohort totals changed: observed {observed}, expected {expected}")

    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Serializing {len(neural_sessions)} sessions, {total_trials} trials, "
        f"{total_neurons} session-neurons, {total_frames} trial frames to "
        f"{args.outpicklefile}",
        flush=True,
    )
    write_start = time.perf_counter()
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    total_elapsed = time.perf_counter() - conversion_start
    print(
        f"Wrote {args.outpicklefile.stat().st_size / 2**30:.3f} GiB in "
        f"{time.perf_counter() - write_start:.2f}s; total conversion {total_elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
