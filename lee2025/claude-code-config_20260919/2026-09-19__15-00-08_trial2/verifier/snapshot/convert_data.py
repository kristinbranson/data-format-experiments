#!/usr/bin/env python3
"""
Convert the Lee, Keinath, Cianfarano & Brandon (2025) CA1 geometric-deformation dataset
into the decoder-compatible pickle format.

    Lee JQ, Keinath AT, Cianfarano E, Brandon MP (2025).
    Identifying representational structure in CA1 to benchmark theoretical models of
    cognitive mapping. Neuron 113(2):307-320.   Data: 10.5281/zenodo.13993254

Decoder task
------------
    input  : environment geometry -- 9-dim binary vector, 1 = that 3x3 partition is
             blocked off on this session.  Static within a trial.
    output : the mouse's position, discretized into the 3 x 3 = 9 partitions of the
             arena.  Time-varying, one categorical value (0..8) per time bin.
    neural : CA1 calcium event rate (the paper's binarized rising-phase "trace",
             treated as the firing rate), temporally binned.

Processing follows the reference implementation of the paper's own within-session
position decoding, `decode_position_within` / `fit_decoder` in
code/georepca1/src/utils.py:
  * spatial binning        `bin_down = (max position over all days and both axes + buffer) / n_bins`
  * running-speed filter   speed = gaussian_filter1d(|dposition|*fps, sigma=5 frames) > 5 cm/s
  * cell filter            > 5 events among the retained (running) samples
  * temporal binning       gaussian_filter1d(trace, sigma=bin) then average pooling by `bin`
See CONVERSION_NOTES.md for the full list of decisions and where/why this deviates.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time
from collections import OrderedDict

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------------------------------------------
# Constants (values taken from the paper / reference code unless noted)
# --------------------------------------------------------------------------------------
DATA_DIR = "/app/data"

ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]          # main.py

FPS = 30.0                 # methods: "acquired behavioral and cellular imaging streams at 30 Hz"
TRIAL_SECONDS = 60.0       # decoder task: 1-minute trials
TRIAL_FRAMES = int(round(TRIAL_SECONDS * FPS))          # 1800
TEMPORAL_BIN_FRAMES = 15   # 500 ms; reference uses 3 (see CONVERSION_NOTES Step 5 decision 4)
BINS_PER_TRIAL = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES    # 120
TIME_BIN_MS = TEMPORAL_BIN_FRAMES / FPS * 1000.0        # 500.0

V_FILT_SIGMA = 5.0         # decode_position_within(v_filt_size=5) frames
V_THRESH = 5.0             # decode_position_within(v_thresh=5) cm/s
CELL_EVENT_THRESHOLD = 5   # decode_position_within(cell_threshold=5) events
SPATIAL_BINS = 3           # decoder task: 3 x 3 partitions (reference uses 15 for 5 cm bins)
POS_BUFFER = 1e-5          # get_rate_maps(buffer=1e-5)

MIN_BINS_PER_TRIAL = 10    # 5 s of running; trials with less are dropped
MIN_TRIALS_PER_SESSION = 2 # train_validate_decoder needs >= 2 trials to split a session

# Two sessions used by --sample, chosen from two different animals and two geometries
# so that the sample exercises subject indexing and a non-square geometry.
SAMPLE_SESSIONS = [("QLAK-CA1-08", 0), ("QLAK-CA1-30", 3)]

BRAIN_REGIONS = ["CA1"]    # paper: GRIN lens targeting dorsal CA1

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def parse_blocked(blocked_day):
    """Return the 9-dim binary geometry vector for one session.

    The dataset's `blocked` field lists the indices of blocked (occluded) partitions in
    the 3x3 grid, numbered [[0,1,2],[3,4,5],[6,7,8]] (dataset README), or -1 when nothing
    is blocked (the square).  Verified in Step 4 to equal `flipud(get_env_mat(env)) == 0`
    for every session, and to coincide exactly with the partitions the animal never
    visits.

    Returns:
        geometry: (9,) float32, 1.0 where the partition is blocked.
    """
    idx = np.atleast_1d(np.asarray(blocked_day, dtype=float).ravel())
    geometry = np.zeros(SPATIAL_BINS * SPATIAL_BINS, dtype=np.float32)
    if idx.size == 1 and idx[0] < 0:          # -1 => open square, nothing blocked
        return geometry
    geometry[idx.astype(int)] = 1.0
    return geometry


def compute_speed(position_xy):
    """Running speed in cm/s, exactly as in `decode_position_within`.

    Args:
        position_xy: (2, T) position in cm.
    Returns:
        (T,) speed; element 0 is 0 and is always excluded downstream, matching the
        reference where `vel_idx[0]` is left False.
    """
    step = np.linalg.norm(np.diff(position_xy, axis=1), axis=0) * FPS   # (T-1,) cm/s
    speed = np.zeros(position_xy.shape[1], dtype=np.float64)
    speed[1:] = gaussian_filter1d(step, sigma=V_FILT_SIGMA)
    return speed


def bin_time(x, n_trials, axis=-1, reduce="mean"):
    """Average-pool the last axis into (n_trials, BINS_PER_TRIAL) blocks of
    TEMPORAL_BIN_FRAMES frames.  Equivalent to torch AvgPool1d(kernel=stride=bin) applied
    within each 1-minute trial."""
    x = np.moveaxis(x, axis, -1)
    x = x[..., : n_trials * TRIAL_FRAMES]
    shp = x.shape[:-1] + (n_trials, BINS_PER_TRIAL, TEMPORAL_BIN_FRAMES)
    out = x.reshape(shp).mean(axis=-1)
    return out


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------


def convert_session(trace_day, position_day, blocked_day, env_name, bin_down,
                    session_id, show_processing=False, plot_dir="."):
    """Convert a single animal-day recording into lists of per-trial arrays.

    Args:
        trace_day: (n_cells_total, T) binarized calcium events, NaN for cells that were
                   not registered on this day.
        position_day: (2, T) x-y position in cm.
        blocked_day: the session's entry of the dataset `blocked` field.
        env_name: geometry name (for bookkeeping / plots).
        bin_down: spatial bin width in cm (shared across all sessions of the animal).
        session_id: string used for plot filenames.
    Returns:
        dict with 'neural', 'input', 'output' (lists of per-trial arrays), plus
        bookkeeping fields, or None if the session does not survive curation.
    """
    n_cells_total, n_frames = trace_day.shape
    n_trials = n_frames // TRIAL_FRAMES          # complete 1-minute trials only
    if n_trials < MIN_TRIALS_PER_SESSION:
        return None
    n_used = n_trials * TRIAL_FRAMES

    # ---- 1. cells registered on this day ------------------------------------------
    registered = ~np.isnan(trace_day[:, 0])
    raw = trace_day[registered][:, :n_used].astype(np.float32)   # (n_reg, n_used), {0,1}

    # ---- 2. running speed and its temporal binning ---------------------------------
    speed = compute_speed(position_day)
    speed_binned = bin_time(speed[None, :], n_trials)[0]         # (n_trials, BINS_PER_TRIAL)

    # ---- 3. position -> 3x3 partition index ----------------------------------------
    pos_binned = bin_time(position_day, n_trials)                # (2, n_trials, BINS)
    xy_bin = np.clip((pos_binned / bin_down).astype(int), 0, SPATIAL_BINS - 1)
    partition = xy_bin[1] * SPATIAL_BINS + xy_bin[0]             # class = y_bin*3 + x_bin

    # ---- 4. geometry (decoder input) ------------------------------------------------
    geometry = parse_blocked(blocked_day)                        # (9,), 1 = blocked

    # ---- 5. sample mask: running, and not in a physically blocked partition ---------
    #   The second term removes rare tracking artifacts: a handful of frames (and, on
    #   two sessions, a few hundred) are tracked inside a partition that is walled off,
    #   which cannot happen physically.  These would be unlearnable label noise.
    moving = speed_binned > V_THRESH
    reachable = geometry[partition] == 0
    keep = moving & reachable
    keep[0, 0] = False          # frame/bin 0 has no defined speed (reference excludes it)

    # ---- 6. cell activity filter ------------------------------------------------------
    #   decode_position_within: cells with more than `cell_threshold` events among the
    #   retained samples.  Counted on the raw binary events of the retained frames.
    frame_keep = np.repeat(keep.ravel(), TEMPORAL_BIN_FRAMES)
    n_events = raw[:, frame_keep].sum(axis=1)
    active = n_events > CELL_EVENT_THRESHOLD
    if active.sum() == 0:
        return None

    # ---- 7. temporal binning of the neural data --------------------------------------
    #   Reference recipe: gaussian smooth along time with sigma = bin size, then average
    #   pool.  Expressed in Hz (x FPS) as in get_rate_maps.
    smoothed = gaussian_filter1d(raw[active], sigma=TEMPORAL_BIN_FRAMES, axis=-1)
    neural_binned = bin_time(smoothed, n_trials) * FPS           # (n_act, n_trials, BINS)
    del smoothed

    # ---- 8. assemble per-trial arrays -------------------------------------------------
    neural_trials, input_trials, output_trials, trial_index = [], [], [], []
    for t in range(n_trials):
        sel = keep[t]
        if sel.sum() < MIN_BINS_PER_TRIAL:
            continue
        neural_trials.append(np.ascontiguousarray(neural_binned[:, t, sel], dtype=np.float32))
        input_trials.append(geometry.copy())
        output_trials.append(partition[t, sel][None, :].astype(np.int64))
        trial_index.append(t)

    if len(neural_trials) < MIN_TRIALS_PER_SESSION:
        return None

    result = {
        "neural": neural_trials,
        "input": input_trials,
        "output": output_trials,
        "n_neurons": int(active.sum()),
        "n_registered": int(registered.sum()),
        "n_trials_possible": n_trials,
        "n_frames_used": n_used,
        "n_frames_total": n_frames,
        "n_bins_total": int(keep.size),
        "n_bins_kept": int(keep.sum()),
        "n_bins_moving": int(moving.sum()),
        "n_bins_unreachable": int((moving & ~reachable).sum()),
        "trial_index": trial_index,
        "env": env_name,
        "geometry": geometry,
        "bin_down": bin_down,
    }

    if show_processing:
        _plot_processing(session_id, raw, active, neural_binned, position_day, pos_binned,
                         xy_bin, partition, speed, speed_binned, keep, geometry, env_name,
                         bin_down, n_trials, plot_dir)
    return result


# --------------------------------------------------------------------------------------
# Diagnostic plots
# --------------------------------------------------------------------------------------


def _plot_processing(session_id, raw, active, neural_binned, position, pos_binned, xy_bin,
                     partition, speed, speed_binned, keep, geometry, env_name, bin_down,
                     n_trials, plot_dir):
    """Plot every processing step for one session so it can be checked by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # pick a trial with a decent amount of running for the zoomed panels
    t_show = int(np.argmax(keep.sum(axis=1)))
    frames = np.arange(t_show * TRIAL_FRAMES, (t_show + 1) * TRIAL_FRAMES)
    t_frames = np.arange(TRIAL_FRAMES) / FPS
    t_bins = (np.arange(BINS_PER_TRIAL) + 0.5) * TEMPORAL_BIN_FRAMES / FPS

    fig, ax = plt.subplots(7, 1, figsize=(17, 26))

    # (1) whole session position, with trial boundaries
    tt = np.arange(position.shape[1]) / FPS / 60.0
    ax[0].plot(tt, position[0], lw=.4, label="x (cm)")
    ax[0].plot(tt, position[1], lw=.4, label="y (cm)")
    for b in range(n_trials + 1):
        ax[0].axvline(b, color="k", lw=.3, alpha=.3)
    for c in (bin_down, 2 * bin_down):
        ax[0].axhline(c, color="r", ls="--", lw=.8)
    ax[0].set_xlabel("time (min)  |  vertical lines = 1-min trial boundaries")
    ax[0].set_ylabel("position (cm)")
    ax[0].set_title(f"{session_id}  env='{env_name}'   (1) raw position, full session; "
                    f"red = 3x3 partition borders at {bin_down:.2f} cm")
    ax[0].legend(loc="upper right")

    # (2) one trial: raw vs binned position, and the discretization
    ax[1].plot(t_frames, position[0, frames], color="C0", lw=.8, label="x raw 30 Hz")
    ax[1].plot(t_bins, pos_binned[0, t_show], "o-", color="navy", ms=3, lw=.8,
               label="x binned (500 ms mean)")
    ax[1].plot(t_frames, position[1, frames], color="C1", lw=.8, label="y raw 30 Hz")
    ax[1].plot(t_bins, pos_binned[1, t_show], "o-", color="darkred", ms=3, lw=.8,
               label="y binned (500 ms mean)")
    for c in (bin_down, 2 * bin_down):
        ax[1].axhline(c, color="r", ls="--", lw=.8)
    axb = ax[1].twinx()
    axb.step(t_bins, xy_bin[0, t_show], where="mid", color="navy", alpha=.35, lw=2)
    axb.step(t_bins, xy_bin[1, t_show], where="mid", color="darkred", alpha=.35, lw=2)
    axb.set_ylabel("x/y bin index (0-2, shaded)")
    axb.set_ylim(-0.2, 2.2)
    ax[1].set_ylabel("position (cm)")
    ax[1].set_xlabel("time in trial (s)")
    ax[1].set_title(f"(2) trial {t_show}: temporal binning + spatial discretization "
                    f"(bin index changes exactly when the trace crosses a red line)")
    ax[1].legend(loc="upper right", fontsize=8)

    # (3) speed and the running filter
    ax[2].plot(t_frames, speed[frames], color=".5", lw=.8, label="smoothed speed (cm/s)")
    ax[2].plot(t_bins, speed_binned[t_show], "o-", color="C2", ms=3, lw=.8,
               label="binned speed")
    ax[2].axhline(V_THRESH, color="r", ls="--", label=f"{V_THRESH:g} cm/s threshold")
    for i, k in enumerate(keep[t_show]):
        if k:
            ax[2].axvspan(t_bins[i] - .25, t_bins[i] + .25, color="g", alpha=.12, lw=0)
    ax[2].set_ylabel("speed (cm/s)")
    ax[2].set_xlabel("time in trial (s)")
    ax[2].set_title("(3) running-speed filter; green shading = bins kept "
                    f"({keep[t_show].sum()}/{BINS_PER_TRIAL} in this trial)")
    ax[2].legend(loc="upper right", fontsize=8)

    # (4) raw binary events for a sample of cells
    idx_act = np.where(active)[0]
    show = idx_act[: 30]
    for i, c in enumerate(show):
        ev = np.where(raw[c, frames] > 0)[0]
        ax[3].vlines(ev / FPS, i, i + .85, lw=.7, color="k")
    ax[3].set_ylim(0, max(len(show), 1))
    ax[3].set_ylabel("cell")
    ax[3].set_xlabel("time in trial (s)")
    ax[3].set_xlim(0, TRIAL_SECONDS)
    ax[3].set_title("(4) raw binarized rising-phase events (30 Hz) for 30 example cells")

    # (5) the same cells after smoothing + binning
    rows = np.searchsorted(idx_act, show)
    im = ax[4].imshow(neural_binned[rows, t_show, :], aspect="auto", origin="lower",
                      extent=[0, TRIAL_SECONDS, 0, len(show)], cmap="magma")
    ax[4].set_ylabel("cell")
    ax[4].set_xlabel("time in trial (s)")
    ax[4].set_title("(5) same cells, gaussian-smoothed and 500 ms average-pooled (Hz) "
                    "- events in (4) line up with bright bins here")
    fig.colorbar(im, ax=ax[4], fraction=.02)

    # (6) output partition index for the trial + what is kept
    ax[5].step(t_bins, partition[t_show], where="mid", color="k", lw=1.2,
               label="partition index (output)")
    ax[5].plot(t_bins[keep[t_show]], partition[t_show][keep[t_show]], "go", ms=4,
               label="kept samples")
    ax[5].set_yticks(range(9))
    ax[5].set_ylabel("partition (y*3+x)")
    ax[5].set_xlabel("time in trial (s)")
    ax[5].set_title("(6) decoder output for this trial")
    ax[5].legend(loc="upper right", fontsize=8)

    # (7) occupancy of the converted output vs the geometry input
    counts = np.bincount(partition[keep].ravel(), minlength=9).astype(float)
    ax[6].imshow(counts.reshape(3, 3), cmap="Blues")
    for p in range(9):
        r, c = divmod(p, 3)
        blocked = geometry[p] > 0
        ax[6].text(c, r, f"p{p}\nn={int(counts[p])}\n"
                         f"{'BLOCKED' if blocked else 'open'}\ninput={int(geometry[p])}",
                   ha="center", va="center",
                   color="red" if blocked else "black", fontsize=9, fontweight="bold")
    ax[6].set_xticks([0, 1, 2], ["x bin 0", "x bin 1", "x bin 2"])
    ax[6].set_yticks([0, 1, 2], ["y bin 0", "y bin 1", "y bin 2"])
    ax[6].set_aspect("equal")
    ax[6].set_title("(7) kept-sample occupancy per partition (output classes) with the "
                    "geometry input overlaid:\nevery BLOCKED partition must have 0 samples")

    fig.tight_layout()
    out = os.path.join(plot_dir, f"processing_{session_id}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    wrote {out}")


# --------------------------------------------------------------------------------------
# Per-animal driver
# --------------------------------------------------------------------------------------


def convert_animal(animal, days=None, show_processing=False, plot_dir="."):
    """Load one animal's joblib file and convert the requested days."""
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    t_load = time.time() - t0

    trace = dat["trace"]                      # (n_days, n_cells, T)
    position = dat["position"]                # (n_days, 2, T)
    envs = np.asarray(dat["envs"]).ravel()
    blocked = dat["blocked"]
    n_days = trace.shape[0]

    # One isotropic spatial bin size per animal, from the maximum position across all
    # days and both axes -- exactly decode_position_within's `bin_down`.
    bin_down = (np.nanmax(position) + POS_BUFFER) / SPATIAL_BINS

    if days is None:
        days = range(n_days)

    t1 = time.time()
    out = []
    for d in days:
        sid = f"{animal}_day{d:02d}_{str(envs[d]).replace(' ', '')}"
        res = convert_session(trace[d], position[d], blocked[d], str(envs[d]), bin_down,
                              sid, show_processing=show_processing, plot_dir=plot_dir)
        if res is None:
            print(f"  {sid}: DROPPED (did not survive curation)")
            continue
        res["animal"] = animal
        res["day"] = int(d)
        res["session_id"] = sid
        # The reference (`get_rsm_partitioned_sequences`) defines a sequence as running
        # square-to-square inclusive, so there are n_days // 10 sequences and the final
        # closing square belongs to the last one.
        res["sequence"] = int(min(d // 10, max(n_days // 10 - 1, 0)) + 1)
        res["n_days"] = int(n_days)
        out.append(res)
    t_proc = time.time() - t1

    total_cells = trace.shape[1]
    del dat, trace, position
    print(f"  {animal}: {len(out)} sessions, {total_cells} registered cells total, "
          f"load {t_load:.1f}s, process {t_proc:.1f}s ({t_proc / max(len(list(days)), 1):.2f}s/session)",
          flush=True)
    return out, total_cells


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def build_dataset(sessions, subjects):
    """Assemble the final dictionary in the target format."""
    subj_index = {s: i for i, s in enumerate(subjects)}

    data = {
        "neural": [s["neural"] for s in sessions],
        "input": [s["input"] for s in sessions],
        "output": [s["output"] for s in sessions],
        "subjects": list(subjects),
        "subject_idx": np.array([subj_index[s["animal"]] for s in sessions], dtype=np.int64),
        "brain_regions": list(BRAIN_REGIONS),
        "brain_region_idx": [np.zeros(s["n_neurons"], dtype=np.int64) for s in sessions],
        "input_names": [f"geometry_partition_{p}_blocked" for p in range(9)],
        "output_names": ["position_partition"],
        "output_values": [[f"p{p}_x{p % 3}_y{p // 3}" for p in range(9)]],
    }

    session_info = []
    for s in sessions:
        session_info.append({
            "session_id": s["session_id"],
            "animal": s["animal"],
            "day": s["day"],
            "environment": s["env"],
            "sequence": s["sequence"],               # 10 geometries per sequence
            "blocked_partitions": [int(p) for p in np.where(s["geometry"] > 0)[0]],
            "n_neurons": s["n_neurons"],
            "n_registered_cells": s["n_registered"],
            "n_trials": len(s["neural"]),
            "spatial_bin_cm": float(s["bin_down"]),
        })

    data["metadata"] = {
        "task_description":
            "Mice freely foraged for 40 min/day in a 75 x 75 cm arena partitioned into a "
            "3 x 3 grid, where subsets of the 9 partitions were walled off to create 10 "
            "distinct geometries presented in a randomized sequence repeated up to three "
            "times (one session per day). CA1 populations were recorded with miniscope "
            "calcium imaging. The decoder receives the CA1 event rates plus the trial's "
            "environment geometry (which of the 9 partitions are blocked) and must "
            "predict which of the 9 partitions the mouse currently occupies.",
        "time_bin_size": float(TIME_BIN_MS),
        "temporal_alignment_event":
            "start of each 1-minute trial; trials are consecutive non-overlapping 60 s "
            "segments of the continuous 40-min free-foraging session (the experiment has "
            "no discrete trial structure or task events)",
        "off_start": 0.0,
        "off_end": float(TRIAL_SECONDS),
        "neural_signal":
            "binarized rising phase of calcium transients (the paper's 'trace'; 1 = "
            "significant event onset), gaussian-smoothed along time with sigma = one "
            "time bin, average-pooled into 500 ms bins, expressed in Hz",
        "sampling_rate_hz": FPS,
        "temporal_bin_frames": TEMPORAL_BIN_FRAMES,
        "spatial_discretization":
            "floor(position / bin_down) per axis, clipped to [0,2]; partition index = "
            "y_bin * 3 + x_bin, matching the dataset's `blocked` partition numbering "
            "[[0,1,2],[3,4,5],[6,7,8]]. bin_down = (max position over all sessions of the "
            "animal + 1e-5) / 3 = 25 cm, as in decode_position_within.",
        "curation":
            f"cells not registered on a session are dropped (NaN traces); cells with "
            f"<= {CELL_EVENT_THRESHOLD} events among retained samples are dropped "
            f"(decode_position_within cell_threshold); samples with running speed "
            f"<= {V_THRESH:g} cm/s are dropped (decode_position_within v_thresh, speed "
            f"smoothed with sigma = {V_FILT_SIGMA:g} frames); samples tracked inside a "
            f"physically blocked partition are dropped as tracking artifacts; trials with "
            f"< {MIN_BINS_PER_TRIAL} retained bins and sessions with "
            f"< {MIN_TRIALS_PER_SESSION} trials are dropped. No place-cell selection: the "
            f"paper includes all cells.",
        "reference":
            "Lee, Keinath, Cianfarano & Brandon (2025) Neuron 113(2):307-320; "
            "data 10.5281/zenodo.13993254",
        "session_info": session_info,
    }
    return data


def print_summary(data, sessions, total_cells_per_animal):
    n_sessions = len(sessions)
    n_trials = sum(len(s["neural"]) for s in sessions)
    n_bins = sum(int(a.shape[1]) for s in sessions for a in s["neural"])
    neurons = np.array([s["n_neurons"] for s in sessions])
    registered = np.array([s["n_registered"] for s in sessions])
    kept = np.array([s["n_bins_kept"] for s in sessions], dtype=float)
    tot = np.array([s["n_bins_total"] for s in sessions], dtype=float)
    unreach = sum(s["n_bins_unreachable"] for s in sessions)
    counts = np.bincount(np.concatenate([o.ravel() for s in sessions for o in s["output"]]),
                         minlength=9)

    print("\n" + "=" * 78)
    print("CONVERSION SUMMARY")
    print("=" * 78)
    print(f"subjects                      : {len(data['subjects'])}  {data['subjects']}")
    print(f"unique registered cells       : {sum(total_cells_per_animal.values())}"
          f"  {dict(total_cells_per_animal)}")
    print(f"sessions                      : {n_sessions}")
    print(f"registered cell-sessions      : {registered.sum()} "
          f"(mean {registered.mean():.1f}/session)")
    print(f"neurons after curation        : {neurons.sum()} "
          f"(mean {neurons.mean():.1f}/session, min {neurons.min()}, max {neurons.max()})")
    n_trials_possible = sum(s["n_trials_possible"] for s in sessions)
    print(f"trials                        : {n_trials} of {n_trials_possible} possible "
          f"(mean {n_trials / n_sessions:.2f}/session)")
    print(f"500 ms bins (all)             : {int(tot.sum())}")
    print(f"  dropped as non-running      : {int(tot.sum() - kept.sum() - unreach)} "
          f"({(tot.sum() - kept.sum() - unreach) / tot.sum() * 100:.1f}%)")
    print(f"  dropped as unreachable      : {unreach} "
          f"({unreach / tot.sum() * 100:.3f}% - tracking artifacts inside blocked partitions)")
    print(f"  dropped with short trials   : {int(kept.sum()) - n_bins} in "
          f"{n_trials_possible - n_trials} trials with < {MIN_BINS_PER_TRIAL} running bins")
    print(f"time bins retained            : {n_bins} "
          f"({n_bins / tot.sum() * 100:.1f}% of all 500 ms bins)")
    print(f"mean bins/trial               : {n_bins / n_trials:.1f} of {BINS_PER_TRIAL}")
    print(f"recording time retained       : {n_bins * TIME_BIN_MS / 1000 / 3600:.2f} h "
          f"of {tot.sum() * TIME_BIN_MS / 1000 / 3600:.2f} h")
    print(f"output class distribution     : {np.round(counts / counts.sum(), 4)}")
    print(f"mean neural value (Hz)        : "
          f"{np.mean([a.mean() for s in sessions for a in s['neural']]):.4f}")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outfile", help="output pickle path")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true", default=True, help="process all sessions (default)")
    g.add_argument("--sample", action="store_true", help="process only 2 sessions")
    ap.add_argument("--show-processing", action="store_true",
                    help="write processing_<session_id>.png for up to 2 sessions")
    ap.add_argument("--plot-dir", default="/app", help="where to write processing plots")
    args = ap.parse_args()

    t_start = time.time()

    if args.sample:
        plan = OrderedDict()
        for animal, day in SAMPLE_SESSIONS:
            plan.setdefault(animal, []).append(day)
        print(f"SAMPLE mode: {SAMPLE_SESSIONS}")
    else:
        plan = OrderedDict((a, None) for a in ANIMALS)
        print(f"FULL mode: {len(ANIMALS)} animals")

    n_plots_left = 2 if args.show_processing else 0
    sessions, total_cells = [], OrderedDict()
    for animal, days in plan.items():
        show = n_plots_left > 0
        out, ncells = convert_animal(animal, days=days, show_processing=show,
                                     plot_dir=args.plot_dir)
        # only plot the first session of each animal we visit, up to 2 in total
        if show:
            n_plots_left -= 1
        sessions.extend(out)
        total_cells[animal] = int(ncells)

    subjects = list(total_cells.keys())
    data = build_dataset(sessions, subjects)
    print_summary(data, sessions, total_cells)

    t = time.time()
    with open(args.outfile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(args.outfile) / 1e6
    print(f"wrote {args.outfile} ({size_mb:.1f} MB) in {time.time() - t:.1f}s")
    print(f"total time {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
