#!/usr/bin/env python3
"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal VR dataset (DANDI 001361, NWB)
into the decoder-ready pickle format.

Reference paper : "A flexible hippocampal population code for experience relative to reward"
Reference code  : /app/code  (package `reward_relative`, notebooks)
Reference data  : /app/data  (152 NWB files, 11 mice)

Processing follows the paper's own pipeline (see CONVERSION_NOTES.md):
  * ROIs: suite2p manual curation (`iscell==1`), planes pooled
  * dF/F : neuropil subtraction (0.7) -> per-trial `maximin` baseline (20 s) ->
           (F-base)/|base| -> Gaussian smooth (sigma = 2 frames)
  * events: OASIS deconvolution of dF/F (tau = 0.7 s, fs = 15.5078125 Hz)
  * putative interneurons removed: Pearson r(dF/F, speed) > 0.5
  * trials = laps, sample range [trial_start, teleport)
  * trials with lick-sensor failure removed (>30% of frames with cumulative lick count > 2)

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample]
                                   [--show-processing] [--signal dff|events]
                                   [--workers N]
"""

import argparse
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

# ----------------------------------------------------------------------------------
# Constants taken from the reference code / methods
# ----------------------------------------------------------------------------------
DATA_ROOT = "/app/data"

FRAME_RATE = 15.5078125          # Hz, per plane; identical for every session
FRAME_PERIOD = 1.0 / FRAME_RATE  # s
NEU_COEF = 0.7                   # reward_relative.utilities.default_dff_method
TAU = 0.7                        # suite2p ops in notebooks/suite2p_notebooks/example_m12.ipynb
BASELINE_SMOOTH_SIGMA = 15       # frames; preprocessing.dff `nansmooth(f_, [0, 15])`
BASELINE_FILTER_WIN = 300        # frames ~= 19.3 s  ("20 s sliding window")
DFF_SMOOTH_SIGMA = 2             # frames (~0.129 s) ; methods "two-sample s.d. Gaussian kernel"
INTERNEURON_SPEED_R = 0.5        # methods: Pearson r(dF/F, speed) > 0.5 -> putative interneuron
LICK_ERROR_FRAC = 0.30           # methods: >30% of frames with cumulative lick count > 2
TRACK_LENGTH = 450.0             # cm
SWITCH_TRIAL = 30                # behavior.get_reward_zones(change_trial=30)

# behavior.reward_zone_dict, resolved through behavior.map_labels (A->'X', B->'Y', C->'Z')
REWARD_ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
ZONE_TO_IDX = {"A": 0, "B": 1, "C": 2}

# teleport_metadata.teleport_sessions: animal -> experiment days on which the laser was
# NOT blanked during the inter-trial teleport period (so the dF/F baseline segment spans it).
TELEPORT_SESSIONS = {
    "m10": [1, 7, 8, 14, 15],
    "m11": [1, 7, 8, 14, 15],
    "m12": [1, 7, 8, 14, 15],
    "m13": [1, 7, 8, 14, 15],
    "m14": [1, 7, 8, 14, 15],
    "m15": [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    "m17": [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    "m18": [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    "m19": [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
}

INPUT_NAMES = [
    "time_from_trial_start_s",
    "environment",
    "trial_number",
    "previous_trial_outcome",
]

OUTPUT_NAMES = [
    "distance_to_reward_zone",
    "track_position",
    "speed",
    "lick",
    "reward_zone_location",
    "reward_outcome",
]

OUTPUT_VALUES = [
    ["< -50 cm", "-50 to -10 cm", "-10 to 0 cm", "0 cm (in reward zone)",
     "0 to +10 cm", "+10 to +50 cm", "> +50 cm"],
    ["< 90 cm", "90-180 cm", "180-270 cm", "270-360 cm", "> 360 cm"],
    ["< 2 cm/s", "2-10 cm/s", "10-20 cm/s", "20-40 cm/s", "> 40 cm/s"],
    ["no lick", "lick"],
    ["A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)"],
    ["omitted", "rewarded"],
]


# ----------------------------------------------------------------------------------
# Session metadata / reward-zone logic (ports of reward_relative.behavior)
# ----------------------------------------------------------------------------------
def parse_scene(identifier):
    """Scene name (reward-zone condition) out of the NWB `identifier` path string."""
    return identifier.strip("/").split("/")[-1]


def reward_zone_labels(scene, ntrials, change_trial=SWITCH_TRIAL):
    """Per-trial reward-zone label ('A'/'B'/'C').

    Port of `reward_relative.behavior.get_reward_zones`: the active zone is given by the
    VR scene name, and on a 'switch' scene the zone changes after `change_trial` laps.
    Scene grammars present in this dataset:
        Env<n>_Location<Z>              -- constant zone Z
        Env<n>_Location<Z0>_to_<Z1>     -- zone Z0 for laps [0, 30), then Z1
        Env<n>_<Z0>_to_Env<m>_<Z1>      -- as above, and the environment changes too
    """
    parts = scene.split("_")
    labels = np.empty(ntrials, dtype="<U1")

    if len(parts) == 2 and parts[1].startswith("Location"):          # Env1_LocationA
        labels[:] = parts[1][-1]
    elif len(parts) == 4 and parts[1].startswith("Location") and parts[2] == "to":
        labels[:change_trial] = parts[1][-1]                          # Env1_LocationA_to_B
        labels[change_trial:] = parts[3]
    elif len(parts) == 5 and parts[2] == "to":                        # Env1_B_to_Env2_C
        labels[:change_trial] = parts[1]
        labels[change_trial:] = parts[4]
    else:
        raise ValueError("Unrecognised scene name: %r" % scene)

    unknown = set(labels) - set(REWARD_ZONES)
    if unknown:
        raise ValueError("Unknown reward zone label(s) %s in scene %r" % (unknown, scene))
    return labels


# ----------------------------------------------------------------------------------
# dF/F (port of reward_relative.preprocessing.dff, maximin baseline, single channel)
# ----------------------------------------------------------------------------------
def compute_dff(F, Fneu, segments, deconvolve=True):
    """Per-trial maximin dF/F, exactly as in `reward_relative.preprocessing.dff`.

    Args:
        F, Fneu: (n_neurons, T) raw suite2p fluorescence / neuropil, float32.
        segments: list of (lo, hi) half-open sample ranges over which the baseline is
            computed. Samples outside every segment stay NaN (as in the reference, which
            blanks out inter-trial data before computing baselines).
        deconvolve: also return OASIS-deconvolved "events".

    Returns:
        dff, spks (spks is None if deconvolve is False), both (n_neurons, T) float32
        with NaN outside the segments.
    """
    from suite2p.extraction import dcnv

    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for lo, hi in segments:
        f_[:, lo:hi] = F[:, lo:hi]
        fneu_[:, lo:hi] = Fneu[:, lo:hi]

    # neuropil_method == 'subtract'
    f_ -= NEU_COEF * fneu_

    dff = np.full(F.shape, np.nan, dtype=np.float32)
    spks = np.full(F.shape, np.nan, dtype=np.float32) if deconvolve else None

    for lo, hi in segments:
        # add the segment's mean neuropil back so dF/F is not divided by a tiny baseline
        x = f_[:, lo:hi] + NEU_COEF * np.nanmean(fneu_[:, lo:hi], axis=1, keepdims=True)

        # maximin baseline: smooth -> minimum filter (erosion) -> maximum filter (dilation)
        flow = gaussian_filter1d(x, BASELINE_SMOOTH_SIGMA, axis=1)
        flow = minimum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)
        flow = maximum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)

        d = (x - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SMOOTH_SIGMA, axis=1)
        dff[:, lo:hi] = d
        if deconvolve:
            spks[:, lo:hi] = dcnv.oasis(np.ascontiguousarray(d), 2000, TAU, FRAME_RATE)

    return dff, spks


# ----------------------------------------------------------------------------------
# Discretisation of the continuous decoder outputs
# ----------------------------------------------------------------------------------
def bin_distance_to_reward(dist):
    """7 classes; see the Decoder Task specification.

    0: < -50 | 1: [-50,-10) | 2: [-10,0) | 3: 0 (inside zone) | 4: (0,10] | 5: (10,50] | 6: > 50
    """
    out = np.empty(dist.shape, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out


def bin_position(pos):
    """5 equal 90 cm bins spanning the 450 cm track."""
    return np.clip(np.digitize(pos, [90.0, 180.0, 270.0, 360.0]), 0, 4).astype(np.int64)


def bin_speed(speed):
    """5 classes: <2, 2-10, 10-20, 20-40, >40 cm/s."""
    return np.clip(np.digitize(speed, [2.0, 10.0, 20.0, 40.0]), 0, 4).astype(np.int64)


def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) to the nearest point of the reward zone; 0 while inside it."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d


# ----------------------------------------------------------------------------------
# Loading one NWB session
# ----------------------------------------------------------------------------------
def load_session(path):
    """Read everything we need out of one NWB file.

    Returns a dict with raw (n_neurons, T) fluorescence, the per-frame behaviour
    columns, trial start/teleport indices and session metadata.
    """
    with h5py.File(path, "r") as f:
        subject = f["general/subject/subject_id"][()].decode()
        exp_day = int(f["general/session_id"][()].decode())
        scene = parse_scene(f["identifier"][()].decode())
        date = f["identifier"][()].decode().strip("/").split("/")[-2]
        region = f["general/optophysiology/ImagingPlane/location"][()].decode()

        seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        iscell = seg["iscell"][:, 0] > 0
        plane_idx_all = seg["planeIdx"][:]

        # Rows of PlaneSegmentation are ordered plane0 then plane1; pool the planes
        # ("planes were pooled for all analyses", Methods).
        F_parts, Fneu_parts, plane_parts, roi_parts = [], [], [], []
        offset = 0
        for plane in sorted(f["processing/ophys/Fluorescence"].keys()):
            dset = f["processing/ophys/Fluorescence"][plane]["data"]
            nroi = dset.shape[1]
            keep = np.where(iscell[offset:offset + nroi])[0]
            F_parts.append(dset[:, :][:, keep].T.astype(np.float32))
            Fneu_parts.append(
                f["processing/ophys/Neuropil"][plane]["data"][:, :][:, keep].T.astype(np.float32))
            plane_parts.append(plane_idx_all[offset:offset + nroi][keep])
            roi_parts.append(offset + keep)   # row index into PlaneSegmentation
            offset += nroi
        assert offset == len(iscell), "ROI count mismatch between planes and segmentation"

        F = np.concatenate(F_parts, axis=0)
        Fneu = np.concatenate(Fneu_parts, axis=0)
        plane_of_cell = np.concatenate(plane_parts, axis=0)
        roi_index = np.concatenate(roi_parts, axis=0)

        b = f["processing/behavior/BehavioralTimeSeries"]
        beh = {k: b[k]["data"][:] for k in
               ["position", "speed", "lick", "reward_zone", "environment",
                "trial number", "autoreward", "scanning"]}
        timestamps = b["position"]["timestamps"][:]
        trial_start = np.where(b["trial_start"]["data"][:] > 0)[0]
        teleport = np.where(b["teleport"]["data"][:] > 0)[0]
        reward_times = f["processing/behavior/BehavioralTimeSeries/Reward"]["timestamps"][:]

    # A handful of sessions store one extra imaging frame; align to the common length.
    T = min(F.shape[1], len(beh["position"]))
    F, Fneu = F[:, :T], Fneu[:, :T]
    beh = {k: v[:T] for k, v in beh.items()}
    timestamps = timestamps[:T]

    assert len(trial_start) == len(teleport), "trial_start/teleport count mismatch"
    assert np.all(teleport > trial_start), "teleport must follow its trial start"
    assert np.all(trial_start[1:] > teleport[:-1]), "trials must not overlap"
    assert teleport[-1] <= T, "trial extends past the imaging data"

    reward_idx = np.searchsorted(timestamps, reward_times)

    return dict(path=path, subject=subject, exp_day=exp_day, scene=scene, date=date,
                region=region, F=F, Fneu=Fneu, plane_of_cell=plane_of_cell,
                roi_index=roi_index,
                beh=beh, timestamps=timestamps, trial_start=trial_start,
                teleport=teleport, reward_idx=reward_idx, n_roi_iscell=int(iscell.sum()),
                n_roi_total=int(len(iscell)))


# ----------------------------------------------------------------------------------
# Full per-session conversion
# ----------------------------------------------------------------------------------
def convert_session(path, signal="dff", show_processing=False, plot_dir="/app"):
    """Load one NWB session and produce its per-trial neural / input / output arrays."""
    t_load = time.time()
    S = load_session(path)
    t_load = time.time() - t_load

    si, ti = S["trial_start"], S["teleport"]
    ntrials_raw = len(si)
    beh = S["beh"]
    pos, speed, lick, rzone = beh["position"], beh["speed"], beh["lick"], beh["reward_zone"]

    # --- baseline segments for dF/F -------------------------------------------------
    # Reference `preprocessing.dff`: by default the baseline is computed on each lap
    # separately; for the animal/day combinations in `teleport_metadata` the laser stayed
    # on through the teleport period and the segment spans ITI + next lap.
    keep_teleports = S["exp_day"] in TELEPORT_SESSIONS.get(S["subject"], [])
    if keep_teleports:
        segments = [(si[0], ti[0])]
        segments += [(ti[k - 1] + 1, ti[k]) for k in range(1, ntrials_raw)]
    else:
        segments = list(zip(si.tolist(), ti.tolist()))

    # --- dF/F and deconvolved events ------------------------------------------------
    t_dff = time.time()
    dff, spks = compute_dff(S["F"], S["Fneu"], segments,
                            deconvolve=(signal == "events") or show_processing)
    t_dff = time.time() - t_dff

    # in-trial sample mask (what actually gets exported)
    in_trial = np.zeros(len(pos), dtype=bool)
    for lo, hi in zip(si, ti):
        in_trial[lo:hi] = True

    # --- neuron curation: drop putative interneurons --------------------------------
    d = dff[:, in_trial]
    v = speed[in_trial].astype(np.float64)
    dc = d - d.mean(axis=1, keepdims=True)
    vc = v - v.mean()
    denom = np.sqrt((dc.astype(np.float64) ** 2).sum(axis=1)) * np.sqrt((vc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r_speed = (dc.astype(np.float64) @ vc) / denom
    r_speed = np.nan_to_num(r_speed, nan=0.0)
    keep_cells = r_speed <= INTERNEURON_SPEED_R
    n_interneurons = int((~keep_cells).sum())
    del d, dc

    activity = spks if signal == "events" else dff
    activity = activity[keep_cells]

    # --- per-trial task variables ---------------------------------------------------
    zone_labels = reward_zone_labels(S["scene"], ntrials_raw)
    zone_start = np.array([REWARD_ZONES[l][0] for l in zone_labels])
    zone_end = np.array([REWARD_ZONES[l][1] for l in zone_labels])
    zone_idx = np.array([ZONE_TO_IDX[l] for l in zone_labels], dtype=np.int64)

    # behavior.get_trial_types: rewarded iff a reward was delivered AND the reward zone
    # was entered on that lap (an omission lap never sets the rzone flag).
    is_reward = np.array([
        bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))
        for lo, hi in zip(si, ti)], dtype=np.int64)

    environment = np.array([int(np.round(np.median(beh["environment"][lo:hi])))
                            for lo, hi in zip(si, ti)], dtype=np.int64)
    trial_number = np.arange(ntrials_raw, dtype=np.int64)

    # First lap of a session: the immediately preceding (unrecorded) warm-up laps were
    # rewarded unless randomly omitted, so 1 is the expected value.
    prev_outcome = np.empty(ntrials_raw, dtype=np.int64)
    prev_outcome[0] = 1
    prev_outcome[1:] = is_reward[:-1]

    # behavior.correct_lick_sensor_error -- trials whose lick sensor was stuck
    lick_error = np.array([(lick[lo:hi] > 2).sum() / (hi - lo) > LICK_ERROR_FRAC
                           for lo, hi in zip(si, ti)], dtype=bool)

    # --- assemble trials ------------------------------------------------------------
    neural, inputs, outputs = [], [], []
    kept_trials = []
    for k, (lo, hi) in enumerate(zip(si, ti)):
        if lick_error[k]:
            continue
        T = hi - lo
        act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)
        if not np.all(np.isfinite(act)):
            raise RuntimeError("non-finite activity in %s trial %d" % (path, k))

        p = pos[lo:hi]
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD
        inp[1] = environment[k]
        inp[2] = trial_number[k]
        inp[3] = prev_outcome[k]

        out = np.empty((6, T), dtype=np.int64)
        out[0] = bin_distance_to_reward(signed_distance_to_zone(p, zone_start[k], zone_end[k]))
        out[1] = bin_position(p)
        out[2] = bin_speed(speed[lo:hi])
        out[3] = (lick[lo:hi] > 0).astype(np.int64)
        out[4] = zone_idx[k]
        out[5] = is_reward[k]

        neural.append(act)
        inputs.append(inp)
        outputs.append(out)
        kept_trials.append(k)

    n_neurons = activity.shape[0]
    info = dict(
        session_id=os.path.basename(path).replace("_behavior+ophys.nwb", ""),
        subject=S["subject"], exp_day=S["exp_day"], scene=S["scene"], date=S["date"],
        region=S["region"], n_roi_total=S["n_roi_total"], n_roi_iscell=S["n_roi_iscell"],
        n_interneurons_removed=n_interneurons, n_neurons=n_neurons,
        n_trials_raw=ntrials_raw, n_trials=len(neural),
        n_trials_lick_error=int(lick_error.sum()),
        keep_teleports=bool(keep_teleports),
        frac_rewarded=float(is_reward.mean()),
        reward_zones=sorted(set(zone_labels.tolist())),
        environments=sorted(set(environment.tolist())),
        n_planes=int(S["plane_of_cell"].max()) + 1,
        kept_trial_indices=np.asarray(kept_trials, dtype=np.int64),
        roi_index=S["roi_index"][keep_cells],
        plane_of_neuron=S["plane_of_cell"][keep_cells],
        load_time_s=round(t_load, 2), dff_time_s=round(t_dff, 2),
    )

    if show_processing:
        try:
            plot_processing(S, dff, spks, keep_cells, r_speed, zone_start, zone_end,
                            zone_labels, is_reward, environment, lick_error,
                            neural, inputs, outputs, kept_trials, signal, info, plot_dir)
        except Exception:
            traceback.print_exc()

    return dict(neural=neural, input=inputs, output=outputs,
                brain_region_idx=np.zeros(n_neurons, dtype=np.int64), info=info)


# ----------------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------------
def plot_processing(S, dff, spks, keep_cells, r_speed, zone_start, zone_end, zone_labels,
                    is_reward, environment, lick_error, neural, inputs, outputs,
                    kept_trials, signal, info, plot_dir):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    si, ti = S["trial_start"], S["teleport"]
    beh = S["beh"]
    ntr = min(6, len(si))
    lo, hi = si[0], ti[ntr - 1]
    tt = np.arange(lo, hi) * FRAME_PERIOD

    fig, ax = plt.subplots(9, 1, figsize=(18, 26), sharex=True)

    # 1. raw F and neuropil for 3 example cells
    cells = np.argsort(-np.nanvar(dff[:, lo:hi], axis=1))[:3]
    for c in cells:
        ax[0].plot(tt, S["F"][c, lo:hi], lw=0.6, label="F cell %d" % c)
        ax[0].plot(tt, S["Fneu"][c, lo:hi], lw=0.6, ls=":", label="Fneu cell %d" % c)
    ax[0].set_ylabel("raw F / Fneu")
    ax[0].legend(fontsize=6, ncol=3)
    ax[0].set_title("%s  (%s, day %d, %s)  -- steps of the conversion, first %d trials"
                    % (info["session_id"], S["subject"], S["exp_day"], S["scene"], ntr))

    # 2. dF/F for the same cells
    for c in cells:
        ax[1].plot(tt, dff[c, lo:hi], lw=0.7, label="dF/F cell %d" % c)
    ax[1].set_ylabel("dF/F")
    ax[1].legend(fontsize=6, ncol=3)

    # 3. deconvolved events
    if spks is not None:
        for c in cells:
            ax[2].plot(tt, spks[c, lo:hi], lw=0.7, label="events cell %d" % c)
        ax[2].set_ylabel("OASIS events")
        ax[2].legend(fontsize=6, ncol=3)
    else:
        ax[2].text(0.5, 0.5, "events not computed (signal=dff)", ha="center",
                   transform=ax[2].transAxes)

    # 4. exported neural raster (what lands in the pickle), trial by trial
    exported = [k for k in kept_trials if k < ntr]
    if exported:
        # pick the 60 most variable cells once, then draw each exported trial at the
        # session time it actually occupies (gaps = inter-trial intervals, which are
        # not exported).
        ref = np.concatenate([neural[kept_trials.index(k)] for k in exported], axis=1)
        sub = np.argsort(-ref.var(axis=1))[:60]
        vmax = np.percentile(ref[sub], 99.5)
        for k in exported:
            mat = neural[kept_trials.index(k)][sub]
            ax[3].imshow(mat, aspect="auto", interpolation="nearest", cmap="magma",
                         vmin=0 if signal == "events" else np.percentile(ref[sub], 1),
                         vmax=vmax,
                         extent=[si[k] * FRAME_PERIOD, ti[k] * FRAME_PERIOD, 60, 0])
        ax[3].set_xlim(lo * FRAME_PERIOD, hi * FRAME_PERIOD)
        ax[3].set_ylim(60, 0)
    ax[3].set_ylabel("exported neural\n(60 cells, %s)" % signal)

    # 5. position with reward zones
    ax[4].plot(tt, beh["position"][lo:hi], "k", lw=1)
    for k in range(ntr):
        ax[4].axvspan(si[k] * FRAME_PERIOD, ti[k] * FRAME_PERIOD, color="0.9", zorder=0)
        ax[4].hlines([zone_start[k], zone_end[k]], si[k] * FRAME_PERIOD,
                     ti[k] * FRAME_PERIOD, color="tab:green", lw=2)
        ax[4].text(si[k] * FRAME_PERIOD, 460, "trial %d\nzone %s\n%s\nENV%d" %
                   (k, zone_labels[k], "rew" if is_reward[k] else "OMIT", environment[k] + 1),
                   fontsize=6, va="bottom")
    ax[4].set_ylabel("position (cm)\ngrey = in-trial")

    # 6. signed distance to reward zone + its discretisation
    dist = np.concatenate([signed_distance_to_zone(beh["position"][si[k]:ti[k]],
                                                   zone_start[k], zone_end[k])
                           for k in range(ntr)])
    tdist = np.concatenate([np.arange(si[k], ti[k]) for k in range(ntr)]) * FRAME_PERIOD
    ax[5].plot(tdist, dist, "k", lw=1)
    for edge in (-50, -10, 0, 10, 50):
        ax[5].axhline(edge, color="tab:red", lw=0.5, ls="--")
    ax2 = ax[5].twinx()
    ax2.plot(tdist, bin_distance_to_reward(dist), "tab:blue", lw=1, drawstyle="steps-post")
    ax2.set_ylabel("class", color="tab:blue")
    ax[5].set_ylabel("signed distance to\nreward zone (cm)")

    # 7. speed + discretisation
    spd = np.concatenate([beh["speed"][si[k]:ti[k]] for k in range(ntr)])
    ax[6].plot(tdist, spd, "k", lw=1)
    for edge in (2, 10, 20, 40):
        ax[6].axhline(edge, color="tab:red", lw=0.5, ls="--")
    ax3 = ax[6].twinx()
    ax3.plot(tdist, bin_speed(spd), "tab:blue", lw=1, drawstyle="steps-post")
    ax3.set_ylabel("class", color="tab:blue")
    ax[6].set_ylabel("speed (cm/s)")

    # 8. licks (raw cumulative count and the exported binary) + reward times
    lk = np.concatenate([beh["lick"][si[k]:ti[k]] for k in range(ntr)])
    ax[7].plot(tdist, lk, "k", lw=1, label="cumulative lick count / frame")
    ax[7].plot(tdist, (lk > 0).astype(float), "tab:blue", lw=1, alpha=0.7,
               drawstyle="steps-post", label="exported binary lick")
    rew = S["reward_idx"][(S["reward_idx"] >= lo) & (S["reward_idx"] < hi)]
    ax[7].plot(rew * FRAME_PERIOD, np.full(len(rew), -0.5), "rv", ms=6, label="reward")
    ax[7].set_ylabel("licks")
    ax[7].legend(fontsize=6)

    # 9. the exported decoder inputs
    for k in exported:
        idx = kept_trials.index(k)
        tk = np.arange(si[k], si[k] + inputs[idx].shape[1]) * FRAME_PERIOD
        for j, nm in enumerate(INPUT_NAMES):
            ax[8].plot(tk, inputs[idx][j], lw=1,
                       color="C%d" % j, label=nm if k == exported[0] else None)
    ax[8].set_ylabel("decoder inputs")
    ax[8].set_xlabel("time in session (s)")
    ax[8].legend(fontsize=6)

    for a in ax:
        for k in range(ntr):
            a.axvline(si[k] * FRAME_PERIOD, color="tab:green", lw=0.8)
            a.axvline(ti[k] * FRAME_PERIOD, color="tab:orange", lw=0.8)

    fig.tight_layout()
    out1 = os.path.join(plot_dir, "processing_%s.png" % info["session_id"])
    fig.savefig(out1, dpi=110)
    plt.close(fig)

    # --- second figure: curation + per-trial summary + output distributions ----------
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    ax[0, 0].hist(r_speed, bins=60, color="0.6")
    ax[0, 0].axvline(INTERNEURON_SPEED_R, color="r")
    ax[0, 0].set_title("r(dF/F, speed) per cell\n%d/%d removed as putative interneurons"
                       % ((~keep_cells).sum(), len(keep_cells)))

    ax[0, 1].plot(is_reward, "o-", ms=3, label="rewarded")
    ax[0, 1].plot(environment, "s-", ms=3, label="environment")
    ax[0, 1].plot([ZONE_TO_IDX[l] for l in zone_labels], "^-", ms=3, label="zone (0=A,1=B,2=C)")
    ax[0, 1].plot(np.where(lick_error)[0], np.full(lick_error.sum(), -0.4), "kx",
                  label="dropped (lick error)")
    ax[0, 1].set_xlabel("trial"); ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title("per-trial task variables")

    # empirical reward-zone entry position vs. the zone assigned from the scene name
    emp = []
    for k, (l, h) in enumerate(zip(si, ti)):
        w = np.where(beh["reward_zone"][l:h] > 0)[0]
        emp.append(beh["position"][l:h][w[0]] if len(w) else np.nan)
    ax[0, 2].plot(emp, "k.", label="1st reward-zone flag position")
    ax[0, 2].plot(zone_start, "g-", label="assigned zone start")
    ax[0, 2].plot(zone_end, "g--", label="assigned zone end")
    ax[0, 2].set_xlabel("trial"); ax[0, 2].set_ylabel("cm"); ax[0, 2].legend(fontsize=7)
    ax[0, 2].set_title("SANITY: zone from scene name vs. data")

    allout = np.concatenate(outputs, axis=1)
    for j, (nm, vals) in enumerate(zip(OUTPUT_NAMES[:3], OUTPUT_VALUES[:3])):
        counts = np.bincount(allout[j], minlength=len(vals)) / allout.shape[1]
        ax[1, j].bar(np.arange(len(vals)), counts)
        ax[1, j].set_xticks(np.arange(len(vals)))
        ax[1, j].set_xticklabels(vals, rotation=45, ha="right", fontsize=7)
        ax[1, j].set_title("%s distribution" % nm)

    fig.suptitle("%s: curation, per-trial variables and output distributions"
                 % info["session_id"])
    fig.tight_layout()
    out2 = os.path.join(plot_dir, "processing_%s_summary.png" % info["session_id"])
    fig.savefig(out2, dpi=110)
    plt.close(fig)
    print("  wrote %s and %s" % (out1, out2), flush=True)


# ----------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------
def list_sessions(sample=False, session_list=None):
    paths = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        subdir = os.path.join(DATA_ROOT, sub)
        if not os.path.isdir(subdir):
            continue
        for fn in sorted(os.listdir(subdir)):
            if fn.endswith(".nwb"):
                paths.append(os.path.join(subdir, fn))

    # sort by (subject number, session number) so sessions are grouped per animal
    def key(p):
        base = os.path.basename(p)
        sub = int(base.split("_")[0].replace("sub-m", ""))
        ses = int(base.split("_")[1].replace("ses-", ""))
        return (sub, ses)
    paths.sort(key=key)
    if session_list:
        wanted = [s.strip() for s in session_list.split(",")]
        return [p for p in paths if any(w in os.path.basename(p) for w in wanted)]
    if sample:
        # one small single-plane session and one large two-plane session
        chosen = [p for p in paths if "sub-m11_ses-03" in p or "sub-m18_ses-03" in p]
        return chosen
    return paths


def _worker(args):
    path, signal, show, plot_dir = args
    t0 = time.time()
    res = convert_session(path, signal=signal, show_processing=show, plot_dir=plot_dir)
    res["info"]["total_time_s"] = round(time.time() - t0, 2)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outfile", help="output pickle path")
    ap.add_argument("--full", action="store_true", default=True,
                    help="process all sessions (default)")
    ap.add_argument("--sample", action="store_true",
                    help="process only 2 sessions, for testing")
    ap.add_argument("--show-processing", action="store_true",
                    help="save per-step diagnostic plots for up to 2 sessions")
    ap.add_argument("--signal", choices=["dff", "events"], default="dff",
                    help="neural signal to export (default: dF/F; see CONVERSION_NOTES.md "
                         "Step 7 for the dF/F-vs-events comparison)")
    ap.add_argument("--session-list", default=None,
                    help="comma-separated session ids to convert (overrides --sample/--full)")
    ap.add_argument("--workers", type=int, default=10,
                    help="parallel session workers (default 10)")
    args = ap.parse_args()

    paths = list_sessions(sample=args.sample, session_list=args.session_list)
    print("Converting %d sessions (signal=%s, workers=%d)" %
          (len(paths), args.signal, args.workers), flush=True)

    plot_dir = os.path.dirname(os.path.abspath(args.outfile)) or "."
    show_for = set(paths[:2]) if args.show_processing else set()
    jobs = [(p, args.signal, p in show_for, plot_dir) for p in paths]

    t_start = time.time()
    results = [None] * len(paths)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, res in enumerate(pool.map(_worker, jobs)):
                results[i] = res
                inf = res["info"]
                print("[%3d/%3d] %-22s %4d neurons  %3d trials  (load %.1fs dff %.1fs "
                      "total %.1fs)  elapsed %.0fs"
                      % (i + 1, len(paths), inf["session_id"], inf["n_neurons"],
                         inf["n_trials"], inf["load_time_s"], inf["dff_time_s"],
                         inf["total_time_s"], time.time() - t_start), flush=True)
    else:
        for i, job in enumerate(jobs):
            results[i] = _worker(job)
            inf = results[i]["info"]
            print("[%3d/%3d] %-22s %4d neurons  %3d trials  (total %.1fs)  elapsed %.0fs"
                  % (i + 1, len(paths), inf["session_id"], inf["n_neurons"],
                     inf["n_trials"], inf["total_time_s"], time.time() - t_start), flush=True)

    print("conversion loop finished in %.1f s" % (time.time() - t_start), flush=True)

    subjects = sorted({r["info"]["subject"] for r in results},
                      key=lambda s: int(s.replace("m", "")))
    data = {
        "neural": [r["neural"] for r in results],
        "input": [r["input"] for r in results],
        "output": [r["output"] for r in results],
        "subjects": subjects,
        "subject_idx": np.array([subjects.index(r["info"]["subject"]) for r in results],
                                dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": [r["brain_region_idx"] for r in results],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Head-fixed mice run laps on a 450 cm virtual linear track (ENV1 or ENV2) with a "
                "hidden 50 cm reward zone at one of three locations (A 80-130, B 200-250, "
                "C 320-370 cm). Licking inside the zone delivers sucrose water; reward is randomly "
                "omitted on ~15% of laps. On 'switch' days the zone moves to a new location after "
                "30 laps, and on day 8 the environment changes as well. From CA1 two-photon "
                "calcium activity the decoder predicts, at every imaging frame, the signed distance "
                "to the reward zone (7 classes), the absolute track position (5 classes), running "
                "speed (5 classes) and licking (2 classes), plus the per-lap reward-zone identity "
                "(3 classes) and reward outcome (2 classes)."),
            "time_bin_size": 1000.0 / FRAME_RATE,
            "temporal_alignment_event": (
                "start of trial: the imaging frame at which the mouse enters the linear track at "
                "0 cm (NWB `trial_start` flag). Each trial runs to the `teleport` flag at the end "
                "of the lap, so trials have variable length."),
            "off_start": 0.0,
            "off_end": None,
            "neural_signal": (
                "OASIS-deconvolved calcium events" if args.signal == "events" else "dF/F"),
            "neural_processing": (
                "suite2p F and neuropil -> neuropil subtraction (0.7) -> per-trial maximin "
                "baseline (Gaussian sigma 15 frames, then 300-frame min and max filters, ~19.3 s) "
                "-> dF/F = (F - baseline)/|baseline| -> Gaussian smoothing (sigma 2 frames, "
                "~0.129 s)" + (" -> OASIS deconvolution (tau = 0.7 s, fs = 15.5078125 Hz)"
                               if args.signal == "events" else "")),
            "neuron_curation": (
                "suite2p manual curation (iscell == 1); putative interneurons removed by "
                "Pearson r(dF/F, running speed) > 0.5"),
            "trial_curation": (
                "trials with lick-sensor failure removed (>30%% of frames with cumulative lick "
                "count > 2), matching the reference; %d trials dropped"
                % sum(r["info"]["n_trials_lick_error"] for r in results)),
            "sampling_rate_hz": FRAME_RATE,
            "recording_modality": "two-photon calcium imaging, GCaMP7f, dorsal CA1",
            "species": "Mus musculus",
            "reference": ("Sosa, Plitt & Giocomo (2025) Nature Neuroscience, "
                          "'A flexible hippocampal population code for experience relative to "
                          "reward'; data DANDI:001361"),
            "session_info": [r["info"] for r in results],
        },
    }

    # ---- summary / sanity checks ----------------------------------------------------
    n_trials = sum(len(s) for s in data["neural"])
    n_neurons = sum(r["info"]["n_neurons"] for r in results)
    n_samples = sum(t.shape[1] for s in data["neural"] for t in s)
    print("\n=== conversion summary ===")
    print("sessions           : %d" % len(results))
    print("subjects           : %d %s" % (len(subjects), subjects))
    print("neurons (total)    : %d   (per session %d-%d, mean %.1f)"
          % (n_neurons, min(r["info"]["n_neurons"] for r in results),
             max(r["info"]["n_neurons"] for r in results), n_neurons / len(results)))
    print("iscell ROIs        : %d ; interneurons removed: %d (%.2f%%)"
          % (sum(r["info"]["n_roi_iscell"] for r in results),
             sum(r["info"]["n_interneurons_removed"] for r in results),
             100 * sum(r["info"]["n_interneurons_removed"] for r in results)
             / max(1, sum(r["info"]["n_roi_iscell"] for r in results))))
    tr = np.array([r["info"]["n_trials"] for r in results])
    print("trials             : %d kept of %d (%d dropped for lick-sensor error); "
          "per session %.1f +/- %.1f"
          % (n_trials, sum(r["info"]["n_trials_raw"] for r in results),
             sum(r["info"]["n_trials_lick_error"] for r in results), tr.mean(), tr.std()))
    print("timepoints (total) : %d  (%.2f GB float32 neural)"
          % (n_samples, sum(t.nbytes for s in data["neural"] for t in s) / 1e9))
    fr = np.array([r["info"]["frac_rewarded"] for r in results])
    print("fraction rewarded  : %.4f (mean over sessions)" % fr.mean())

    allout = np.concatenate([t for s in data["output"] for t in s], axis=1)
    print("\noutput class distributions (fraction of timepoints):")
    for j, nm in enumerate(OUTPUT_NAMES):
        counts = np.bincount(allout[j], minlength=len(OUTPUT_VALUES[j]))
        print("  %-26s %s" % (nm, np.round(counts / counts.sum(), 4).tolist()))
    allinp = np.concatenate([t for s in data["input"] for t in s], axis=1)
    print("\ninput ranges:")
    for j, nm in enumerate(INPUT_NAMES):
        print("  %-26s [%.3f, %.3f]" % (nm, allinp[j].min(), allinp[j].max()))
    del allout, allinp

    t0 = time.time()
    with open(args.outfile, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nwrote %s (%.2f GB) in %.1f s"
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t0))
    print("TOTAL %.1f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
