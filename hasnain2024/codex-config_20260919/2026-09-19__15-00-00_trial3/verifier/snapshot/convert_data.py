#!/usr/bin/env python3
"""Convert Hasnain/Birnbaum ALM two-context sessions for neural decoding.

Run as:
    python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scipy_io


APP = Path("/app")
DATA_DIR = APP / "data" / "Ephys_Behavior"
TMIN = -2.5
TMAX = 2.5
DT = 0.010
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
N_TIME = TIME.size
EDGES = TMIN + np.arange(N_TIME + 1, dtype=np.float64) * DT
SMOOTH_N = 15
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}

# Exact order and ALM probes in the supplied two-context analysis scripts.
SESSIONS = [
    ("JEB6", "2021-04-18", (2,)),
    ("JEB7", "2021-04-29", (1,)),
    ("JEB7", "2021-04-30", (1,)),
    ("EKH1", "2021-08-07", (2,)),
    ("EKH3", "2021-08-11", (2,)),
    ("JGR2", "2021-11-16", (1,)),
    ("JGR2", "2021-11-17", (1,)),
    ("JGR3", "2021-11-18", (1,)),
    ("JEB19", "2023-04-21", (1,)),
    ("JEB19", "2023-04-20", (1,)),
    ("JEB19", "2023-04-19", (1,)),
    ("JEB19", "2023-04-18", (1,)),
]


def matlab_gausswin_causal(n: int = SMOOTH_N, alpha: float = 2.5) -> np.ndarray:
    """Match MATLAB gausswin(N), then the causal modification in mySmooth.m."""
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2
    # MATLAB gausswin uses exp(-0.5*(alpha*n/(N/2))^2).
    kernel = np.exp(-0.5 * (alpha * x / (n / 2)) ** 2)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    return kernel


KERNEL = matlab_gausswin_causal()


def h5_vector(f: h5py.File, obj_or_ref, dtype=np.float64) -> np.ndarray:
    obj = f[obj_or_ref] if isinstance(obj_or_ref, h5py.Reference) else obj_or_ref
    if not isinstance(obj, h5py.Dataset) or obj.size == 0:
        return np.empty(0, dtype=dtype)
    return np.asarray(obj[()]).ravel().astype(dtype, copy=False)


def h5_string(f: h5py.File, ref) -> str:
    values = h5_vector(f, ref, np.uint16)
    return "".join(chr(int(x)) for x in values).strip("\x00 ")


def h5_cellstr(f: h5py.File, ref) -> list[str]:
    obj = f[ref]
    if not isinstance(obj, h5py.Dataset) or obj.dtype.kind != "O":
        return []
    return [h5_string(f, r) for r in obj[()].ravel(order="F")]


def matlab_mode(x: np.ndarray) -> float:
    finite = np.asarray(x)[np.isfinite(x)]
    if not finite.size:
        return np.nan
    values, counts = np.unique(finite, return_counts=True)
    return float(values[np.argmax(counts)])


def smooth_reference(counts: np.ndarray) -> np.ndarray:
    """Reference mySmooth(x, 15, 'reflect') along the last dimension."""
    prefix = counts[..., :SMOOTH_N]
    padded = np.concatenate((prefix, counts), axis=-1)
    half = KERNEL.size // 2
    pad_width = [(0, 0)] * padded.ndim
    pad_width[-1] = (half, half)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(padded, pad_width, mode="constant"), KERNEL.size, axis=-1
    )
    # Convolution reverses the kernel; einsum handles every leading dimension.
    smoothed = np.einsum("...k,k->...", windows, KERNEL[::-1], optimize=True)
    return smoothed[..., SMOOTH_N:]


def bin_one_unit(trials: np.ndarray, trial_times: np.ndarray, go: np.ndarray,
                 n_trials: int) -> np.ndarray:
    """Return trial x time spike counts, matching histc edge semantics."""
    counts = np.zeros((n_trials, N_TIME), dtype=np.float32)
    valid_trial = (trials >= 1) & (trials <= n_trials) & np.isfinite(trial_times)
    tr0 = trials[valid_trial].astype(np.int64) - 1
    aligned = trial_times[valid_trial] - go[tr0]
    # MATLAB histc assigns an observation equal to an edge to that edge's bin.
    # Explicit search avoids floor roundoff for spikes exactly on decimal edges.
    bins = np.searchsorted(EDGES, aligned, side="right") - 1
    valid = np.isfinite(aligned) & (bins >= 0) & (bins < N_TIME)
    np.add.at(counts, (tr0[valid], bins[valid]), 1)
    return counts


def load_neural(f: h5py.File, probes: tuple[int, ...], go: np.ndarray,
                n_trials: int) -> tuple[np.ndarray, list[dict], dict]:
    """Load, quality-curate, bin, smooth, and firing-rate-curate units."""
    all_rates = []
    all_meta = []
    quality_counts = Counter()
    for probe_index, probe_ref in enumerate(f["obj/clu"][:, 0], start=1):
        if probe_index not in probes:
            continue
        group = f[probe_ref]
        if not isinstance(group, h5py.Group):
            continue
        for cluster_index in range(group["quality"].shape[0]):
            quality = h5_string(f, group["quality"][cluster_index, 0]).strip()
            quality_counts[quality or "<empty>"] += 1
            if quality.lower() in BAD_QUALITIES:
                continue
            trials = h5_vector(f, group["trial"][cluster_index, 0], np.float64)
            trial_times = h5_vector(f, group["trialtm"][cluster_index, 0], np.float64)
            if trials.size != trial_times.size:
                raise ValueError(f"Spike trial/time length mismatch, probe {probe_index}, cluster {cluster_index + 1}")
            counts = bin_one_unit(trials, trial_times, go, n_trials)
            rates = smooth_reference(counts / DT).astype(np.float32)
            mean_rate = float(rates.mean())
            if mean_rate > 1.0:
                all_rates.append(rates)
                all_meta.append({
                    "probe": probe_index,
                    "source_cluster_index_1based": cluster_index + 1,
                    "quality": quality,
                    "mean_firing_rate_hz": mean_rate,
                })
    if not all_rates:
        raise ValueError("No units survived quality and firing-rate curation")
    # trial x neuron x time
    neural = np.stack(all_rates, axis=1)
    stats = {
        "raw_quality_counts_selected_probes": dict(quality_counts),
        "quality_pass_count": int(sum(v for k, v in quality_counts.items() if k.lower() not in BAD_QUALITIES)),
        "retained_unit_count": len(all_rates),
    }
    return neural, all_meta, stats


def fill_nearest(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    good = np.flatnonzero(np.isfinite(values))
    if not good.size:
        return values
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        pos = np.searchsorted(good, bad)
        left = good[np.maximum(pos - 1, 0)]
        right = good[np.minimum(pos, good.size - 1)]
        choose_right = np.abs(right - bad) < np.abs(bad - left)
        nearest = np.where(choose_right, right, left)
        values[bad] = values[nearest]
    return values


def interp_matlab(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Linear interpolation with NaN outside source coverage, like interp1."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    finite_x = np.isfinite(x)
    x, y = x[finite_x], y[finite_x]
    if x.size < 2 or np.any(np.diff(x) <= 0):
        return np.full(target.shape, np.nan, dtype=np.float64)
    return np.interp(target, x, y, left=np.nan, right=np.nan)


def velocity_from_xy(x: np.ndarray, y: np.ndarray, tongue: bool) -> tuple[np.ndarray, np.ndarray]:
    visible = np.isfinite(x) & np.isfinite(y)
    if tongue:
        xf, yf = x.copy(), y.copy()
    else:
        xf, yf = fill_nearest(x), fill_nearest(y)
    if np.sum(np.isfinite(xf) & np.isfinite(yf)) < 2:
        return np.full(x.shape, np.nan), visible
    xv = np.gradient(xf)
    yv = np.gradient(yf)
    if tongue:
        xv[~np.isfinite(xv)] = 0
        yv[~np.isfinite(yv)] = 0
    else:
        # Match findVelocity.m, including its subtraction of basederiv(1)
        # from both coordinate gradients.
        baseline_x = np.nanmedian(np.diff(np.column_stack((xf, yf)), axis=0), axis=0)[0]
        xv -= baseline_x
        yv -= baseline_x
    speed = np.hypot(xv, yv)
    speed[~visible] = np.nan
    return speed, visible


def get_traj_group(f: h5py.File, view_index_zero: int):
    ref = f["obj/traj"][view_index_zero, 0]
    obj = f[ref]
    return obj if isinstance(obj, h5py.Group) else None


def traj_feature_names(f: h5py.File, traj: h5py.Group) -> list[str]:
    for ref in traj["featNames"][:, 0]:
        names = h5_cellstr(f, ref)
        if names:
            return names
    return []


def load_trial_traj(f: h5py.File, traj: h5py.Group, trial: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if traj is None or trial >= traj["ts"].shape[0]:
        return None, None
    if "NdroppedFrames" in traj:
        dropped = h5_vector(f, traj["NdroppedFrames"][trial, 0])
        if dropped.size and np.isnan(dropped[0]):
            return None, None
    ts_obj = f[traj["ts"][trial, 0]]
    if not isinstance(ts_obj, h5py.Dataset) or ts_obj.size == 0:
        return None, None
    ts = np.asarray(ts_obj[()]).T  # MATLAB frames x (x,y,p) x features
    ft_obj = f[traj["frameTimes"][trial, 0]] if "frameTimes" in traj else None
    if isinstance(ft_obj, h5py.Dataset) and ft_obj.size:
        frame_times = np.asarray(ft_obj[()]).ravel().astype(np.float64)
    else:
        frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    if frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
        return None, None
    return ts, frame_times


def load_motion_file(path: Path, n_trials: int) -> list[np.ndarray | None]:
    loaded = scipy_io.loadmat(path, simplify_cells=True)["me"]
    raw = loaded["data"]
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    raw = np.atleast_1d(raw)
    out = []
    for trial in range(n_trials):
        if trial >= raw.size:
            out.append(None)
            continue
        arr = np.asarray(raw[trial], dtype=np.float64).ravel()
        out.append(arr if arr.size else None)
    return out


def load_behavior_streams(f: h5py.File, motion_path: Path, go: np.ndarray,
                          n_trials: int, vidshift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return continuous tongue speed, paw speed, and motion energy (trial x time)."""
    tongue = np.full((n_trials, N_TIME), np.nan, dtype=np.float64)
    paw = np.full_like(tongue, np.nan)
    motion = np.full_like(tongue, np.nan)

    side = get_traj_group(f, 0)
    bottom = get_traj_group(f, 1)
    side_names = traj_feature_names(f, side) if side is not None else []
    bottom_names = traj_feature_names(f, bottom) if bottom is not None else []
    if "tongue" not in side_names:
        raise ValueError(f"Side-view tongue feature unavailable: {side_names}")
    tongue_ix = side_names.index("tongue")
    paw_indices = [bottom_names.index(x) for x in ("top_paw", "bottom_paw") if x in bottom_names]
    if not paw_indices:
        raise ValueError(f"Bottom-view paw features unavailable: {bottom_names}")
    motion_trials = load_motion_file(motion_path, n_trials)

    for trial in range(n_trials):
        target_absolute = TIME
        side_ts, side_ft = load_trial_traj(f, side, trial)
        if side_ts is not None and tongue_ix < side_ts.shape[2]:
            aligned_ft = side_ft - vidshift - go[trial]
            x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], target_absolute)
            y = interp_matlab(aligned_ft, side_ts[:, 1, tongue_ix], target_absolute)
            tongue[trial], _ = velocity_from_xy(x, y, tongue=True)

        bottom_ts, bottom_ft = load_trial_traj(f, bottom, trial)
        paw_speeds = []
        if bottom_ts is not None:
            aligned_ft = bottom_ft - vidshift - go[trial]
            for feature_ix in paw_indices:
                if feature_ix >= bottom_ts.shape[2]:
                    continue
                x = interp_matlab(aligned_ft, bottom_ts[:, 0, feature_ix], target_absolute)
                y = interp_matlab(aligned_ft, bottom_ts[:, 1, feature_ix], target_absolute)
                speed, _ = velocity_from_xy(x, y, tongue=False)
                paw_speeds.append(speed)
        if paw_speeds:
            stack = np.stack(paw_speeds)
            count = np.sum(np.isfinite(stack), axis=0)
            summed = np.nansum(stack, axis=0)
            paw[trial] = np.divide(summed, count, out=np.full(N_TIME, np.nan), where=count > 0)

        me = motion_trials[trial]
        # Reference uses side-camera frame times for motion-energy alignment.
        if me is not None and side_ft is not None:
            aligned_ft = side_ft - vidshift - go[trial]
            motion[trial] = interp_matlab(aligned_ft, me, target_absolute)
        elif me is not None:
            fallback = np.arange(1, me.size + 1, dtype=np.float64) / 400.0
            motion[trial] = interp_matlab(fallback - 0.5 - go[trial], me, target_absolute)

    return tongue, paw, motion, {
        "side_features": side_names,
        "bottom_features": bottom_names,
    }


def discretize_session(values: np.ndarray, retained: np.ndarray) -> tuple[np.ndarray, float]:
    subset = values[retained]
    finite = np.isfinite(subset)
    if not finite.any():
        return np.full(subset.shape, 2, dtype=np.int64), np.nan
    threshold = float(np.nanpercentile(subset, 50))
    labels = np.full(subset.shape, 2, dtype=np.int64)
    labels[finite & (subset < threshold)] = 0
    labels[finite & (subset >= threshold)] = 1
    return labels, threshold


def trial_labels(bp: dict[str, np.ndarray], trial: int) -> tuple[int, int, int]:
    right, left = bool(bp["R"][trial]), bool(bp["L"][trial])
    hit, miss, no = bool(bp["hit"][trial]), bool(bp["miss"][trial]), bool(bp["no"][trial])
    outcome = 1 if hit else (0 if miss else 2)
    if no:
        lick = 2
    elif hit:
        lick = 1 if right else 0
    else:  # incorrect response is opposite the instructed/reward side
        lick = 0 if right else 1
    context = 0 if bool(bp["autowater"][trial]) else 1
    return lick, context, outcome


def processing_plot(session_id: str, neural: np.ndarray, retained_trials: np.ndarray,
                    continuous: tuple[np.ndarray, np.ndarray, np.ndarray],
                    labels: tuple[np.ndarray, np.ndarray, np.ndarray], thresholds: tuple[float, float, float],
                    output_path: Path) -> None:
    trial_pos = 0
    source_trial = int(retained_trials[trial_pos])
    tongue, paw, motion = continuous
    tongue_lab, paw_lab, motion_lab = labels
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    ax = axes[0, 0]
    image = ax.imshow(neural[trial_pos], aspect="auto", extent=(TMIN, TMAX, neural.shape[1], 0))
    ax.axvline(0, color="w", linestyle="--", linewidth=1)
    ax.set(title=f"{session_id}: smoothed neural rates, source trial {source_trial + 1}", ylabel="retained unit")
    ax.set_xlabel("time from go cue / water (s)")
    fig.colorbar(image, ax=ax, label="spikes/s")

    ax = axes[0, 1]
    ax.plot(TIME, tongue[source_trial], label="tongue speed", linewidth=1)
    ax.plot(TIME, paw[source_trial], label="paw speed", linewidth=1)
    ax.axhline(thresholds[0], color="C0", linestyle=":", label="tongue median")
    ax.axhline(thresholds[1], color="C1", linestyle=":", label="paw median")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set(title="Clock-corrected kinematics and thresholds", xlabel="time (s)", ylabel="pixels / 10-ms bin")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(TIME, motion[source_trial], color="C2", linewidth=1)
    ax.axhline(thresholds[2], color="k", linestyle=":", label="session median")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set(title="Aligned motion energy", xlabel="time (s)", ylabel="a.u.")
    ax.legend()

    ax = axes[1, 1]
    for row, (name, arr, color) in enumerate(zip(
            ("tongue", "paw", "motion"),
            (tongue_lab[trial_pos], paw_lab[trial_pos], motion_lab[trial_pos]),
            ("C0", "C1", "C2"))):
        ax.step(TIME, arr + row * 3, where="mid", label=name, color=color)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set(title="Discretized outputs (0 low, 1 high, 2 unavailable)", xlabel="time (s)", yticks=[])
    ax.legend()

    ax = axes[2, 0]
    for vals, threshold, name, color in zip((tongue, paw, motion), thresholds,
                                             ("tongue", "paw", "motion"), ("C0", "C1", "C2")):
        finite = vals[retained_trials][np.isfinite(vals[retained_trials])]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [1, 99])
            ax.hist(finite[(finite >= lo) & (finite <= hi)], bins=60, density=True,
                    histtype="step", label=name, color=color)
            ax.axvline(threshold, color=color, linestyle=":")
    ax.set(title="Session distributions and median cutoffs", xlabel="raw value", ylabel="density")
    ax.legend()

    ax = axes[2, 1]
    missing = [np.mean(x == 2) for x in (tongue_lab, paw_lab, motion_lab)]
    low = [np.mean(x == 0) for x in (tongue_lab, paw_lab, motion_lab)]
    high = [np.mean(x == 1) for x in (tongue_lab, paw_lab, motion_lab)]
    xx = np.arange(3)
    ax.bar(xx, low, label="low")
    ax.bar(xx, high, bottom=low, label="high")
    ax.bar(xx, missing, bottom=np.asarray(low) + high, label="unavailable")
    ax.set(title="Categorical fractions", xticks=xx, xticklabels=["tongue", "paw", "motion"], ylim=(0, 1))
    ax.legend()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def process_session(animal: str, date: str, probes: tuple[int, ...], show_plot: bool):
    session_id = f"{animal}_{date}"
    data_path = DATA_DIR / f"data_structure_{session_id}.mat"
    motion_path = DATA_DIR / f"motionEnergy_{session_id}.mat"
    started = time.perf_counter()
    with h5py.File(data_path, "r") as f:
        n_trials = int(np.asarray(f["obj/bp/Ntrials"])[0, 0])
        fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
        bp = {name: np.asarray(f[f"obj/bp/{name}"]).ravel().astype(bool) for name in fields}
        bp["stim"] = np.asarray(f["obj/bp/stim/enable"]).ravel().astype(bool)
        go = np.asarray(f["obj/bp/ev/goCue"]).ravel().astype(np.float64)
        for name, values in bp.items():
            if values.size != n_trials:
                raise ValueError(f"{session_id}: {name} has {values.size}, expected {n_trials}")
        if go.size != n_trials:
            raise ValueError(f"{session_id}: goCue length mismatch")

        outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
        side_sum = bp["R"].astype(int) + bp["L"].astype(int)
        retained = np.flatnonzero(~bp["early"] & ~bp["stim"] & np.isfinite(go)
                                  & (outcome_sum == 1) & (side_sum == 1))
        if retained.size < 2:
            raise ValueError(f"{session_id}: fewer than two retained trials")

        neural_all, unit_info, neural_stats = load_neural(f, probes, go, n_trials)
        bit_start = np.asarray(f["obj/bp/ev/bitStart"]).ravel()
        sglx_start = np.asarray(f["obj/sglx/bitcode/bitstart"]).ravel()
        fs = float(np.asarray(f["obj/sglx/fs"])[0, 0])
        vidshift = matlab_mode(sglx_start) / fs - matlab_mode(bit_start)
        tongue, paw, motion, behavior_info = load_behavior_streams(
            f, motion_path, go, n_trials, vidshift
        )

    tongue_labels, tongue_threshold = discretize_session(tongue, retained)
    paw_labels, paw_threshold = discretize_session(paw, retained)
    motion_labels, motion_threshold = discretize_session(motion, retained)
    thresholds = (tongue_threshold, paw_threshold, motion_threshold)

    neural_trials = []
    input_trials = []
    output_trials = []
    time_input = TIME.astype(np.float32)[None, :]
    for pos, source_trial in enumerate(retained):
        neural_trial = np.ascontiguousarray(neural_all[source_trial], dtype=np.float32)
        lick, context, outcome = trial_labels(bp, int(source_trial))
        output = np.empty((6, N_TIME), dtype=np.int64)
        output[0] = lick
        output[1] = context
        output[2] = outcome
        output[3] = tongue_labels[pos]
        output[4] = paw_labels[pos]
        output[5] = motion_labels[pos]
        if not np.isfinite(neural_trial).all():
            raise ValueError(f"{session_id}: non-finite neural data")
        neural_trials.append(neural_trial)
        input_trials.append(time_input.copy())
        output_trials.append(output)

    elapsed = time.perf_counter() - started
    session_info = {
        "session_id": session_id,
        "animal": animal,
        "date": date,
        "source_file": str(data_path),
        "motion_energy_file": str(motion_path),
        "selected_probes_1based": list(probes),
        "native_trial_count": n_trials,
        "retained_trial_count": int(retained.size),
        "retained_source_trial_indices_1based": (retained + 1).tolist(),
        "excluded_early_count": int(bp["early"].sum()),
        "excluded_stimulation_count": int(bp["stim"].sum()),
        "video_shift_seconds": float(vidshift),
        "tongue_velocity_median": tongue_threshold,
        "paw_velocity_median": paw_threshold,
        "motion_energy_median": motion_threshold,
        "units": unit_info,
        **neural_stats,
        **behavior_info,
        "conversion_seconds": elapsed,
    }
    if show_plot:
        processing_plot(
            session_id, neural_all[retained], retained,
            (tongue, paw, motion), (tongue_labels, paw_labels, motion_labels), thresholds,
            APP / f"processing_{session_id}.png",
        )
    print(
        f"[{session_id}] native trials={n_trials}, retained={retained.size}, "
        f"units={neural_all.shape[1]}, thresholds={thresholds}, time={elapsed:.2f}s",
        flush=True,
    )
    return neural_trials, input_trials, output_trials, session_info


def validate_built(data: dict) -> None:
    n_sessions = len(data["neural"])
    assert n_sessions == len(data["input"]) == len(data["output"])
    assert data["subject_idx"].shape == (n_sessions,)
    assert len(data["brain_region_idx"]) == n_sessions
    for s in range(n_sessions):
        n_trials = len(data["neural"][s])
        assert n_trials >= 2 and n_trials == len(data["input"][s]) == len(data["output"][s])
        n_neurons = data["neural"][s][0].shape[0]
        assert data["brain_region_idx"][s].shape == (n_neurons,)
        for neural, inp, out in zip(data["neural"][s], data["input"][s], data["output"][s]):
            assert neural.shape == (n_neurons, N_TIME) and neural.dtype == np.float32
            assert inp.shape == (1, N_TIME) and inp.dtype == np.float32
            assert out.shape == (6, N_TIME) and np.issubdtype(out.dtype, np.integer)
            assert np.isfinite(neural).all() and np.isfinite(inp).all()
            for row, values in enumerate(data["output_values"]):
                assert out[row].min() >= 0 and out[row].max() < len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outpicklefile", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="process all sessions (default)")
    mode.add_argument("--sample", action="store_true", help="process only the first two sessions")
    parser.add_argument("--show-processing", action="store_true")
    args = parser.parse_args()
    selected = SESSIONS[:2] if args.sample else SESSIONS
    overall = time.perf_counter()

    subjects = []
    neural, inputs, outputs, session_info = [], [], [], []
    for index, (animal, date, probes) in enumerate(selected):
        if animal not in subjects:
            subjects.append(animal)
        show = args.show_processing and index < 2
        n, i, o, info = process_session(animal, date, probes, show)
        neural.append(n)
        inputs.append(i)
        outputs.append(o)
        session_info.append(info)

    subject_idx = np.asarray([subjects.index(animal) for animal, _, _ in selected], dtype=np.int64)
    brain_region_idx = [np.zeros(trials[0].shape[0], dtype=np.int64) for trials in neural]
    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": subject_idx,
        "brain_regions": ["ALM"],
        "brain_region_idx": brain_region_idx,
        "input_names": ["time from go cue onset (s)"],
        "output_names": [
            "lick direction", "behavioral context", "outcome",
            "tongue velocity", "paw velocity", "motion energy",
        ],
        "output_values": [
            ["left", "right", "none"],
            ["WC", "DR"],
            ["incorrect", "correct", "ignore"],
            ["<50th percentile", ">=50th percentile", "not visible"],
            ["<50th percentile", ">=50th percentile", "not visible"],
            ["<50th percentile", ">=50th percentile", "no video"],
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice alternated between delayed-response (DR) and water-cued (WC) "
                "directional licking blocks; ALM activity predicts response, context, outcome, and movement."
            ),
            "time_bin_size": 10.0,
            "temporal_alignment_event": "go cue onset (water presentation in WC trials)",
            "off_start": TMIN,
            "off_end": TMAX,
            "neural_representation": "spikes/s, 10-ms bins, 15-sample causal Gaussian smoothing",
            "neuron_curation": "reference-selected ALM probes; explicit bad qualities removed; mean firing rate >1 Hz",
            "trial_curation": "early-lick and stimulation trials excluded; valid correct, incorrect, and ignore trials retained",
            "video_frame_rate_hz": 400.0,
            "velocity_units": "DLC pixels per 10-ms aligned sample (categorization is session-percentile based)",
            "session_threshold_scope": "median of finite visible samples over retained trials and full aligned window",
            "time_axis_seconds": TIME.astype(np.float32),
            "session_info": session_info,
        },
    }
    validate_built(data)
    args.outpicklefile.parent.mkdir(parents=True, exist_ok=True)
    with args.outpicklefile.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    total_seconds = time.perf_counter() - overall
    total_trials = sum(map(len, neural))
    total_units = sum(x[0].shape[0] for x in neural)
    print(
        f"Saved {args.outpicklefile} ({len(neural)} sessions, {total_trials} trials, "
        f"{total_units} session-units) in {total_seconds:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
