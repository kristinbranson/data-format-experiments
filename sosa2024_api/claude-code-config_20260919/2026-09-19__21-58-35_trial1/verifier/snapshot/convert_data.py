#!/usr/bin/env python3
"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal CA1 VR dataset (DANDI:001361)
from NWB into the decoder-ready pickle format.

Reference paper:  "A flexible hippocampal population code for experience relative
                   to reward", Nature Neuroscience 2025.
Reference code:   /app/code  (GiocomoLab/Sosa_et_al_2024)

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

    --full             process all sessions (default)
    --sample           process only 2 sessions (for testing)
    --show-processing  save per-step diagnostic plots for up to 2 sessions
"""

import argparse
import glob
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy as sp
import scipy.ndimage

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- #
# Constants taken from the reference code / paper
# ----------------------------------------------------------------------------- #

# reward_relative/behavior.py :: reward_zone_dict  (map_labels A->X, B->Y, C->Z)
# Paper: "zone A, 80-130 cm; zone B, 200-250 cm; zone C, 320-370 cm"
REWARD_ZONES = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
ZONE_NAMES = ["A", "B", "C"]

# reward_relative/behavior.py :: get_reward_zones(..., change_trial=30)
# Paper: "Each switch occurred after 30 trials."
CHANGE_TRIAL = 30

# reward_relative/utilities.py :: default_dff_method
NEU_COEF = 0.7                 # neuropil subtraction coefficient
BASELINE_FILTER_LEN = 300      # maximin window, 300 samples @15.5 Hz ~= 19.3 s ("20 s")
BASELINE_SMOOTH_SIGMA = 15     # pre-maximin Gaussian smoothing sigma (samples)
DFF_SMOOTH_SIGMA = 2           # post-dF/F Gaussian smoothing sigma (~0.129 s)

# Paper: putative interneurons excluded if corr(dF/F, speed) > 0.5
SPEED_CORR_THRESH = 0.5

# Paper: lick-sensor failure detection (~0.65% of trials, n = 81 / 12,376)
LICK_ERR_COUNT_THRESH = 2      # cumulative lick count per imaging frame
LICK_ERR_FRAC_THRESH = 0.3     # fraction of frames in the trial

TRACK_LENGTH = 450.0           # cm
POS_BIN_EDGES = [90.0, 180.0, 270.0, 360.0]
SPEED_BIN_EDGES = [2.0, 10.0, 20.0, 40.0]

OUTPUT_NAMES = [
    "reward_zone_distance",
    "position",
    "speed",
    "lick",
    "reward_zone_location",
    "reward_outcome",
]
OUTPUT_VALUES = [
    ["< -50 cm", "-50 to -10 cm", "-10 to <0 cm", "0 cm (in reward zone)",
     ">0 to +10 cm", "+10 to +50 cm", "> +50 cm"],
    ["< 90 cm", "90-180 cm", "180-270 cm", "270-360 cm", "> 360 cm"],
    ["< 2 cm/s", "2-10 cm/s", "10-20 cm/s", "20-40 cm/s", "> 40 cm/s"],
    ["no lick", "lick"],
    ["zone A (80-130 cm)", "zone B (200-250 cm)", "zone C (320-370 cm)"],
    ["omitted", "rewarded"],
]
INPUT_NAMES = [
    "time_from_trial_start_s",
    "environment",          # 0 = ENV1, 1 = ENV2
    "trial_number",
    "previous_trial_reward",
]


# ----------------------------------------------------------------------------- #
# Helpers copied / adapted from the reference code
# ----------------------------------------------------------------------------- #

def nansmooth(a, sig, axis=-1):
    """Gaussian smoothing that does not propagate NaNs.

    Verbatim port of reward_relative/utilities.py :: nansmooth (and the
    equivalent TwoPUtils.utilities.nansmooth used for the dF/F baseline).
    """
    nan_inds = np.isnan(a)
    a_nanless = np.copy(a)
    a_nanless[nan_inds] = 0
    one = np.ones(a.shape)
    one[nan_inds] = 0.001
    a_nanless = sp.ndimage.gaussian_filter1d(a_nanless, sig, axis=axis)
    one = sp.ndimage.gaussian_filter1d(one, sig, axis=axis)
    return a_nanless / one


def compute_dff(F, Fneu, trial_starts, trial_ends):
    """Compute dF/F exactly as in reward_relative/preprocessing.py :: dff().

    Steps (keep_teleports=False, neuropil_method='subtract',
    baseline_method='maximin', subtract_baseline=True):
      1. keep only within-trial samples (everything else NaN)
      2. F  <- F - 0.7 * Fneu
      3. add back 0.7 * <Fneu> of the trial so dF/F is close to true dF/F
      4. baseline = maximin (min then max filter, 300 samples ~ 20 s) of the
         Gaussian-smoothed (sigma 15) signal, computed within each trial
      5. dF/F = (F - baseline) / |baseline|
      6. smooth dF/F with a 2-sample s.d. Gaussian kernel, within each trial

    Args:
        F:     (n_neurons, n_frames) ROI fluorescence
        Fneu:  (n_neurons, n_frames) neuropil fluorescence
        trial_starts, trial_ends: frame indices, trial i spans [start, end)

    Returns:
        dff: (n_neurons, n_frames) with NaN outside of trials
        in_trial: boolean (n_frames,) mask of within-trial samples
    """
    f_ = np.full(F.shape, np.nan)
    fneu_ = np.full(F.shape, np.nan)
    for s, e in zip(trial_starts, trial_ends):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]

    in_trial = ~np.isnan(f_[0, :])

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan)
    for s, e in zip(trial_starts, trial_ends):
        # add back the trial-mean neuropil so we never divide by a small number
        f_[:, s:e] = f_[:, s:e] + NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1,
                                                        keepdims=True)
        tmp = nansmooth(f_[:, s:e], BASELINE_SMOOTH_SIGMA, axis=1)
        tmp = sp.ndimage.minimum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)
        flow[:, s:e] = sp.ndimage.maximum_filter1d(tmp, BASELINE_FILTER_LEN, axis=-1)

    dff = np.full(F.shape, np.nan)
    dff[:, in_trial] = ((f_[:, in_trial] - flow[:, in_trial])
                        / np.abs(flow[:, in_trial]))

    for s, e in zip(trial_starts, trial_ends):
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)

    return dff, in_trial


def reward_zone_labels_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    """Per-trial reward-zone label, following behavior.py :: get_reward_zones().

    Scene strings look like 'Env1_LocationA'          (no switch)
                            'Env1_LocationB_to_A'      (zone switch)
                            'Env1_B_to_Env2_C'         (zone + environment switch)
    The zone switches after `change_trial` trials.

    Returns list of 'A'/'B'/'C' of length n_trials.
    """
    # no-switch scenes: Env{1,2,3}_Location{A,B,C}
    for z in ZONE_NAMES:
        if scene.endswith("_Location" + z):
            return [z] * n_trials

    # switch scenes: '<zone0>_to' ... ends with '<zone1>'
    zone1 = scene[-1]
    zone0 = None
    for z in ZONE_NAMES:
        if (z + "_to") in scene:
            zone0 = z
            break
    if zone0 is None or zone1 not in ZONE_NAMES:
        raise NotImplementedError(f"Cannot parse reward zones for scene '{scene}'")
    return [zone0] * min(change_trial, n_trials) + \
           [zone1] * max(n_trials - change_trial, 0)


def discretize_reward_distance(pos, zone_start, zone_end):
    """Signed distance to the nearest point of the reward zone -> 7 classes.

    d < 0   : animal is before the zone (d = pos - zone_start)
    d = 0   : animal is inside the zone
    d > 0   : animal is past the zone (d = pos - zone_end)
    """
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end

    out = np.full(pos.shape, 3, dtype=np.int64)     # 3 == inside the zone
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
    return out, d


# ----------------------------------------------------------------------------- #
# Session processing
# ----------------------------------------------------------------------------- #

def load_session(path):
    """Read everything we need out of one NWB file using pynwb."""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwb = io.read()

        subject = nwb.subject.subject_id
        session_id = nwb.session_id
        # nwb.identifier == '/data/<...>/<animal>/<date>/<scene>'
        ident_parts = nwb.identifier.rstrip("/").split("/")
        scene = ident_parts[-1]
        date = ident_parts[-2]

        ophys = nwb.processing["ophys"]
        plane_seg = (ophys.data_interfaces["ImageSegmentation"]
                     .plane_segmentations["PlaneSegmentation"])
        # suite2p iscell.npy, column 0 == (manually curated) cell/not-cell label
        iscell = np.asarray(plane_seg["iscell"].data)[:, 0].astype(bool)
        plane_idx_all = np.asarray(plane_seg["planeIdx"].data).astype(int)

        # Multi-plane animals (m17, m18) store one RoiResponseSeries per imaging
        # plane; the PlaneSegmentation table concatenates the planes' ROIs. Planes
        # are pooled, as in the paper ("planes were pooled for all analyses except
        # those in Extended Data Fig. 7").
        fluo = ophys.data_interfaces["Fluorescence"].roi_response_series
        neuro = ophys.data_interfaces["Neuropil"].roi_response_series
        plane_keys = sorted(fluo.keys(), key=lambda k: int(k.replace("plane", "")))
        n_roi_total = len(plane_idx_all)
        n_frames = fluo[plane_keys[0]].data.shape[0]
        F = np.empty((n_roi_total, n_frames), dtype=np.float64)
        Fneu = np.empty((n_roi_total, n_frames), dtype=np.float64)
        for k in plane_keys:
            p = int(k.replace("plane", ""))
            rows = np.where(plane_idx_all == p)[0]
            assert len(rows) == fluo[k].data.shape[1], (
                f"{k}: {len(rows)} ROIs in table vs {fluo[k].data.shape[1]} traces")
            F[rows] = np.asarray(fluo[k].data[:]).T
            Fneu[rows] = np.asarray(neuro[k].data[:]).T
        rate = float(fluo[plane_keys[0]].rate)
        plane_idx = plane_idx_all[iscell]
        F = F[iscell]
        Fneu = Fneu[iscell]

        bts = (nwb.processing["behavior"]
               .data_interfaces["BehavioralTimeSeries"].time_series)
        beh = {k: np.asarray(v.data[:]) for k, v in bts.items() if k != "Reward"}
        timestamps = np.asarray(bts["position"].timestamps[:])
        reward_times = np.asarray(bts["Reward"].timestamps[:])

    n_planes = int(plane_idx_all.max()) + 1

    # In 10 of the 2-plane sessions the fluorescence series has exactly one frame
    # more than the VR-aligned behaviour: TwoPUtils' vr_align_to_2P builds
    # int(max_idx / n_planes) rows, which truncates the final volume. The extra
    # frame is at the end of the recording, so drop it.
    n_extra = F.shape[1] - len(timestamps)
    assert 0 <= n_extra <= 1, (
        f"fluorescence has {F.shape[1]} frames, behaviour {len(timestamps)}")
    if n_extra:
        F = F[:, :len(timestamps)]
        Fneu = Fneu[:, :len(timestamps)]
    return dict(subject=subject, session_id=session_id, scene=scene, date=date,
                iscell=iscell, plane_idx=plane_idx, F=F, Fneu=Fneu, rate=rate,
                beh=beh, timestamps=timestamps, reward_times=reward_times,
                n_planes=n_planes, n_roi_total=int(n_roi_total))


def process_session(path, show_processing=False, plot_prefix=None, verbose=True):
    """Convert one NWB session into (neural, input, output, info) lists."""
    t_load = time.time()
    S = load_session(path)
    t_load = time.time() - t_load

    t_proc = time.time()
    beh = S["beh"]
    ts = S["timestamps"]
    pos = beh["position"]
    speed = beh["speed"]
    lick = beh["lick"]
    env_ts = beh["environment"]
    rzone_ts = beh["reward_zone"]
    scanning = beh["scanning"]
    trial_num_ts = beh["trial number"]

    starts = np.where(beh["trial_start"] > 0)[0]
    ends = np.where(beh["teleport"] > 0)[0]
    assert len(starts) == len(ends), "trial_start / teleport count mismatch"
    assert np.all(ends > starts), "teleport must follow its trial_start"
    n_trials_raw = len(starts)

    # ---------------- neural: dF/F, then interneuron exclusion --------------- #
    dff, in_trial = compute_dff(S["F"], S["Fneu"], starts, ends)

    # Paper: "Additional putative interneurons were detected for exclusion ...
    #         by a Pearson correlation of >0.5 between their dF/F timeseries and
    #         the animal's running speed"
    d = dff[:, in_trial]
    sp_ = speed[in_trial]
    d0 = d - d.mean(axis=1, keepdims=True)
    s0 = sp_ - sp_.mean()
    denom = np.sqrt((d0 ** 2).sum(axis=1) * (s0 ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        speed_corr = (d0 @ s0) / denom
    speed_corr = np.nan_to_num(speed_corr, nan=0.0)
    keep_cells = speed_corr <= SPEED_CORR_THRESH
    n_interneuron = int((~keep_cells).sum())
    dff_all = dff
    dff = dff[keep_cells]
    plane_idx = S["plane_idx"][keep_cells]
    n_neurons = dff.shape[0]

    # ---------------- per-trial behavioural variables ------------------------ #
    # reward delivery: the 'Reward' TimeSeries holds one timestamp per delivery
    rew_frames = np.searchsorted(ts, S["reward_times"])

    zone_labels = reward_zone_labels_from_scene(S["scene"], n_trials_raw)

    trial_rewarded = np.zeros(n_trials_raw, dtype=np.int64)
    trial_env = np.zeros(n_trials_raw, dtype=np.int64)
    trial_lick_error = np.zeros(n_trials_raw, dtype=bool)
    trial_unscanned = np.zeros(n_trials_raw, dtype=bool)
    observed_zone = np.full(n_trials_raw, -1, dtype=np.int64)

    for i, (s, e) in enumerate(zip(starts, ends)):
        # reference behavior.py :: get_trial_types
        #   isreward = any(reward > 0) and any(rzone > 0) within the trial
        in_zone = rzone_ts[s:e] > 0
        got_reward = np.any((rew_frames >= s) & (rew_frames < e))
        trial_rewarded[i] = int(got_reward and np.any(in_zone))

        ev = np.unique(env_ts[s:e])
        ev = ev[ev >= 0]
        trial_env[i] = int(ev[0]) if ev.size else -1

        # lick-sensor failure, exactly as described in the paper
        trial_lick_error[i] = (np.mean(lick[s:e] > LICK_ERR_COUNT_THRESH)
                               > LICK_ERR_FRAC_THRESH)
        trial_unscanned[i] = np.any(scanning[s:e] < 0)

        if np.any(in_zone):
            p = pos[s:e][np.where(in_zone)[0][0]]
            observed_zone[i] = int(np.argmin([abs(p - REWARD_ZONES[z][0])
                                              for z in ZONE_NAMES]))

    # sanity check: where the animal actually triggered the reward zone, the
    # zone derived from the scene name must agree
    scene_zone = np.array([ZONE_NAMES.index(z) for z in zone_labels])
    seen = observed_zone >= 0
    n_zone_mismatch = int(np.sum(observed_zone[seen] != scene_zone[seen]))

    # previous-trial outcome. Trial 0 of a session is preceded (within minutes)
    # by ~30 rewarded warm-up trials of the same task, so it is assigned 1.
    prev_rewarded = np.empty(n_trials_raw, dtype=np.int64)
    prev_rewarded[0] = 1
    prev_rewarded[1:] = trial_rewarded[:-1]

    # ---------------- trial curation ---------------------------------------- #
    trial_len = ends - starts
    too_short = trial_len < 2
    keep_trial = ~(trial_lick_error | trial_unscanned | too_short)

    # ---------------- build per-trial arrays -------------------------------- #
    neural_trials, input_trials, output_trials = [], [], []
    kept_idx = []
    for i in np.where(keep_trial)[0]:
        s, e = starts[i], ends[i]
        T = e - s

        neural_trials.append(np.ascontiguousarray(dff[:, s:e], dtype=np.float32))

        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = ts[s:e] - ts[s]           # time from trial start (s)
        inp[1] = trial_env[i]              # 0 = ENV1, 1 = ENV2
        inp[2] = i                         # trial number within session
        inp[3] = prev_rewarded[i]
        input_trials.append(inp)

        zs, ze = REWARD_ZONES[zone_labels[i]]
        p = pos[s:e]
        out = np.empty((6, T), dtype=np.int64)
        out[0], _ = discretize_reward_distance(p, zs, ze)
        out[1] = np.clip(np.digitize(p, POS_BIN_EDGES), 0, 4)
        out[2] = np.clip(np.digitize(speed[s:e], SPEED_BIN_EDGES), 0, 4)
        out[3] = (lick[s:e] > 0).astype(np.int64)
        out[4] = scene_zone[i]
        out[5] = trial_rewarded[i]
        output_trials.append(out)
        kept_idx.append(int(i))

    info = dict(
        file=os.path.basename(path), subject=S["subject"], session_id=S["session_id"],
        exp_day=int(S["session_id"]), scene=S["scene"], date=S["date"],
        n_roi_total=S["n_roi_total"], n_iscell=int(S["iscell"].sum()),
        n_interneuron_excluded=n_interneuron, n_neurons=n_neurons,
        n_planes=S["n_planes"], rate=S["rate"],
        n_trials_raw=n_trials_raw, n_trials_kept=len(kept_idx),
        n_lick_error=int(trial_lick_error.sum()),
        n_unscanned=int(trial_unscanned.sum()), n_too_short=int(too_short.sum()),
        n_zone_mismatch=n_zone_mismatch, n_zone_observed=int(seen.sum()),
        frac_rewarded=float(trial_rewarded.mean()),
        kept_trial_idx=kept_idx,
        zone_labels=[zone_labels[i] for i in kept_idx],
        dt=float(np.median(np.diff(ts))),
    )
    t_proc = time.time() - t_proc
    info["t_load"] = t_load
    info["t_proc"] = t_proc

    if verbose:
        print(f"  {info['file']}: {n_neurons} neurons "
              f"({info['n_iscell']} iscell - {n_interneuron} interneuron), "
              f"{len(kept_idx)}/{n_trials_raw} trials, scene={S['scene']}, "
              f"zone-mismatch={n_zone_mismatch}/{info['n_zone_observed']}, "
              f"load={t_load:.1f}s proc={t_proc:.1f}s", flush=True)

    if show_processing:
        make_processing_plots(S, dff_all, keep_cells, speed_corr, starts, ends,
                              zone_labels, scene_zone, trial_rewarded, trial_env,
                              neural_trials, input_trials, output_trials,
                              kept_idx, plot_prefix)

    return neural_trials, input_trials, output_trials, plane_idx, info


# ----------------------------------------------------------------------------- #
# Diagnostic plots
# ----------------------------------------------------------------------------- #

def make_processing_plots(S, dff, keep_cells, speed_corr, starts, ends,
                          zone_labels, scene_zone, trial_rewarded, trial_env,
                          neural_trials, input_trials, output_trials,
                          kept_idx, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    beh, ts = S["beh"], S["timestamps"]
    pos, speed, lick = beh["position"], beh["speed"], beh["lick"]

    ntr_show = min(6, len(starts))
    i0, i1 = starts[0], ends[ntr_show - 1]
    sl = slice(i0, i1)
    t = ts[sl]

    fig, ax = plt.subplots(9, 1, figsize=(16, 24), sharex=True)

    # (1) raw fluorescence + neuropil for 3 example cells
    cells = np.arange(min(3, S["F"].shape[0]))
    for c in cells:
        ax[0].plot(t, S["F"][c, sl], lw=0.6, label=f"F cell {c}")
        ax[0].plot(t, S["Fneu"][c, sl], lw=0.6, ls=":", label=f"Fneu cell {c}")
    ax[0].set_ylabel("raw F / Fneu")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title(f"{S['subject']} ses-{S['session_id']} ({S['scene']}) "
                    f"- first {ntr_show} trials\nStep 1: raw suite2p traces "
                    f"(iscell ROIs only)")

    # (2) dF/F for the same cells
    for c in cells:
        ax[1].plot(t, dff[c, sl], lw=0.7, label=f"dF/F cell {c}")
    ax[1].set_ylabel("dF/F")
    ax[1].legend(fontsize=6, ncol=3)
    ax[1].set_title("Step 2: dF/F (neuropil-subtracted, maximin baseline, "
                    "2-sample Gaussian). NaN between trials.")

    # (3) population raster of the kept neurons
    kept = dff[keep_cells]
    ax[2].imshow(kept[:, sl], aspect="auto", origin="lower",
                 extent=[t[0], t[-1], 0, kept.shape[0]],
                 vmin=0, vmax=np.nanpercentile(kept[:, sl], 99), cmap="magma")
    ax[2].set_ylabel("neuron")
    ax[2].set_title(f"Step 3: dF/F population, {kept.shape[0]} neurons kept "
                    f"({int((~keep_cells).sum())} excluded, speed corr > 0.5)")

    # (4) position + reward zone
    ax[3].plot(t, pos[sl], "k", lw=1)
    for j in range(ntr_show):
        s, e = starts[j], ends[j]
        zs, ze = REWARD_ZONES[zone_labels[j]]
        ax[3].fill_between([ts[s], ts[e - 1]], zs, ze, color="g", alpha=0.25)
        ax[3].axvline(ts[s], color="b", lw=1)
        ax[3].axvline(ts[e - 1], color="r", lw=1)
        ax[3].text(ts[s], 460, f"t{j} {zone_labels[j]}"
                               f"{' R' if trial_rewarded[j] else ' omit'}",
                   fontsize=7)
    ax[3].set_ylabel("position (cm)")
    ax[3].set_title("Step 4: position, reward zone (green), trial start (blue) "
                    "/ teleport (red)")

    # (5) discretized position
    ax[4].step(t, np.clip(np.digitize(pos[sl], POS_BIN_EDGES), 0, 4), where="post")
    ax[4].set_ylabel("position bin")
    ax[4].set_title("Step 5: discretized position (5 bins of 90 cm)")

    # (6) distance to reward zone (continuous + discretized)
    dist = np.full(t.shape, np.nan)
    dcls = np.full(t.shape, np.nan)
    for j in range(ntr_show):
        s, e = starts[j], ends[j]
        zs, ze = REWARD_ZONES[zone_labels[j]]
        cls, dd = discretize_reward_distance(pos[s:e], zs, ze)
        dist[s - i0:e - i0] = dd
        dcls[s - i0:e - i0] = cls
    ax[5].plot(t, dist, "k", lw=1)
    for y in [-50, -10, 0, 10, 50]:
        ax[5].axhline(y, color="gray", lw=0.5, ls="--")
    ax[5].set_ylabel("dist to RZ (cm)")
    ax[5].set_title("Step 6: signed distance to nearest point of reward zone")
    ax[6].step(t, dcls, where="post")
    ax[6].set_ylabel("RZ dist bin")
    ax[6].set_title("Step 7: discretized reward-zone distance (7 bins)")

    # (8) speed
    ax[7].plot(t, speed[sl], "k", lw=0.8)
    for y in SPEED_BIN_EDGES:
        ax[7].axhline(y, color="gray", lw=0.5, ls="--")
    ax2 = ax[7].twinx()
    ax2.step(t, np.clip(np.digitize(speed[sl], SPEED_BIN_EDGES), 0, 4),
             where="post", color="C1", lw=0.8)
    ax2.set_ylabel("speed bin", color="C1")
    ax[7].set_ylabel("speed (cm/s)")
    ax[7].set_title("Step 8: speed and its discretization")

    # (9) licks
    ax[8].step(t, lick[sl], where="post", color="k", lw=0.8, label="lick count")
    ax[8].step(t, (lick[sl] > 0).astype(float), where="post", color="C3", lw=0.8,
               label="lick binary (output)")
    ax[8].set_ylabel("licks")
    ax[8].set_xlabel("time (s)")
    ax[8].legend(fontsize=7)
    ax[8].set_title("Step 9: lick counts per imaging frame -> binary output")

    fig.tight_layout()
    fig.savefig(f"{prefix}.png", dpi=110)
    plt.close(fig)

    # --- second figure: final per-trial arrays, to prove alignment ---------- #
    fig, ax = plt.subplots(4, 4, figsize=(20, 12))
    for k in range(min(4, len(neural_trials))):
        n, inp, out = neural_trials[k], input_trials[k], output_trials[k]
        tt = inp[0]
        ax[0, k].imshow(n, aspect="auto", origin="lower",
                        extent=[tt[0], tt[-1], 0, n.shape[0]],
                        vmin=0, vmax=np.percentile(n, 99), cmap="magma")
        ax[0, k].set_title(f"trial {kept_idx[k]} dF/F ({n.shape[0]} neurons)")
        for r in range(4):
            ax[1, k].plot(tt, inp[r], label=INPUT_NAMES[r])
        ax[1, k].legend(fontsize=6)
        ax[1, k].set_title("inputs")
        for r in range(4):
            ax[2, k].step(tt, out[r], where="post", label=OUTPUT_NAMES[r])
        ax[2, k].legend(fontsize=6)
        ax[2, k].set_title("time-varying outputs")
        ax[3, k].step(tt, out[4], where="post", label=OUTPUT_NAMES[4])
        ax[3, k].step(tt, out[5], where="post", label=OUTPUT_NAMES[5])
        ax[3, k].legend(fontsize=6)
        ax[3, k].set_xlabel("time from trial start (s)")
        ax[3, k].set_title("per-trial outputs")
    fig.suptitle(f"{S['subject']} ses-{S['session_id']} - converted trials")
    fig.tight_layout()
    fig.savefig(f"{prefix}_trials.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# Driver
# ----------------------------------------------------------------------------- #

def _worker(args):
    path, show, prefix = args
    return process_session(path, show_processing=show, plot_prefix=prefix)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outfile")
    ap.add_argument("--full", action="store_true", default=True)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--show-processing", action="store_true")
    ap.add_argument("--data-dir", default="/app/data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "sub-*", "*.nwb")))
    if args.sample:
        # one single-plane and one multi-plane animal, one switch session each
        preferred = ["sub-m11_ses-03", "sub-m17_ses-05"]
        chosen = []
        for p in preferred:
            m = [f for f in files if p in f]
            if m:
                chosen.append(m[0])
        files = chosen if len(chosen) == 2 else files[:2]
    print(f"Processing {len(files)} sessions with {args.workers} workers", flush=True)

    t0 = time.time()
    jobs = []
    for i, f in enumerate(files):
        show = args.show_processing and i < 2
        sid = os.path.basename(f).replace("_behavior+ophys.nwb", "")
        jobs.append((f, show, f"processing_{sid}"))

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_worker, jobs):
                results.append(r)
                done = len(results)
                el = time.time() - t0
                print(f"[{done}/{len(files)}] elapsed {el:.0f}s, "
                      f"est. total {el / done * len(files):.0f}s", flush=True)
    else:
        for j in jobs:
            results.append(_worker(j))

    # ------------------------------ assemble ------------------------------- #
    neural, inputs, outputs, region_idx, infos = [], [], [], [], []
    subjects = []
    subject_idx = []
    for n, i_, o, pidx, info in results:
        if len(n) < 2:
            print(f"  WARNING: dropping {info['file']} with {len(n)} trials")
            continue
        neural.append(n)
        inputs.append(i_)
        outputs.append(o)
        region_idx.append(np.zeros(n[0].shape[0], dtype=np.int64))
        infos.append(info)
        if info["subject"] not in subjects:
            subjects.append(info["subject"])
        subject_idx.append(subjects.index(info["subject"]))

    dt = float(np.median([i["dt"] for i in infos]))
    data = {
        "neural": neural,
        "input": inputs,
        "output": outputs,
        "subjects": subjects,
        "subject_idx": np.array(subject_idx, dtype=np.int64),
        "brain_regions": ["CA1"],
        "brain_region_idx": region_idx,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": OUTPUT_VALUES,
        "metadata": {
            "task_description": (
                "Head-fixed mice run laps on a 450 cm virtual linear track with a "
                "hidden 50 cm reward zone at one of three locations (A: 80-130 cm, "
                "B: 200-250 cm, C: 320-370 cm) in one of two visually distinct "
                "environments (ENV1/ENV2). Sucrose reward is delivered operantly "
                "for licking inside the zone and is randomly omitted on ~15% of "
                "trials. On 'switch' days the reward zone moves to a new location "
                "after 30 trials, sometimes together with a switch of environment. "
                "Decoded outputs: signed distance to the reward zone, absolute "
                "track position, running speed, licking, reward-zone identity and "
                "reward outcome. Decoder inputs: time from trial start, "
                "environment, trial number and previous trial's reward outcome."),
            "time_bin_size": dt * 1000.0,
            "temporal_alignment_event": (
                "trial start = VR 'trial_start' event, i.e. entry onto the linear "
                "track at position 0 cm after the intertrial teleport period"),
            "off_start": 0.0,
            "off_end": None,
            "trial_end_event": "teleport (end of lap, entry into the intertrial interval)",
            "neural_signal": (
                "dF/F of suite2p ROIs, computed as in "
                "reward_relative/preprocessing.py::dff(): neuropil subtraction "
                "(0.7 * Fneu) with the trial-mean neuropil added back, per-trial "
                "maximin baseline (Gaussian sigma=15 samples, then 300-sample "
                "~20 s min/max filters), dF/F = (F-F0)/|F0|, smoothed with a "
                "2-sample (~0.129 s) s.d. Gaussian kernel."),
            "neuron_curation": (
                "suite2p iscell == 1 (manual curation in the suite2p GUI, as in "
                "the paper) and Pearson correlation between the cell's dF/F and "
                "the animal's running speed <= 0.5 (putative interneuron "
                "exclusion, as in the paper)."),
            "trial_curation": (
                "trials from trial_start to teleport. Excluded: trials flagged as "
                "lick-sensor failures (>30% of imaging frames with a cumulative "
                "lick count > 2; the paper removes the same 81 trials), trials "
                "overlapping periods without 2P scanning, and trials shorter than "
                "2 imaging frames."),
            "sampling_rate_hz": 1.0 / dt,
            "imaging": ("2-photon calcium imaging (GCaMP7f) of hippocampal CA1, "
                        "~15.5 Hz per imaging plane; 2 planes pooled in m17/m18."),
            "session_info": infos,
            "n_sessions": len(neural),
            "n_trials_total": int(sum(len(n) for n in neural)),
            "n_neurons_total": int(sum(n[0].shape[0] for n in neural)),
            "source": "DANDI:001361 (Sosa, Plitt & Giocomo 2025, Nat Neurosci)",
        },
    }

    print(f"\nSessions: {len(neural)}  trials: {data['metadata']['n_trials_total']}"
          f"  neurons: {data['metadata']['n_neurons_total']}"
          f"  subjects: {len(subjects)}")
    print(f"time bin: {data['metadata']['time_bin_size']:.4f} ms")
    nbytes = sum(t.nbytes for s in neural for t in s)
    print(f"neural size: {nbytes / 1e9:.2f} GB")

    t1 = time.time()
    with open(args.outfile, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Wrote {args.outfile} in {time.time() - t1:.0f}s "
          f"({os.path.getsize(args.outfile) / 1e9:.2f} GB)")
    print(f"Total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
