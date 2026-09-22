#!/usr/bin/env python3
"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal CA1 2-photon dataset
(DANDI 001361, NWB format) into the decoder-training dictionary format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the authors' reference code (/app/code, `reward_relative` package):
  * dF/F is recomputed from the raw `Fluorescence` / `Neuropil` traces with
    `preprocessing.dff`'s exact algorithm (neuropil subtraction with coefficient 0.7,
    per-trial `maximin` baseline over a 20 s window, 2-sample Gaussian smoothing).
    The OASIS-deconvolved "events" of the same function are also computed and can be
    stored instead with `--signal events`; dF/F is the default (see CONVERSION_NOTES).
  * Trials are delimited by the `[trial_start - 1, teleport - 1)` frame window used
    throughout the reference code (`preprocessing.dff`, `glmUtils.get_timeseries_data`).
  * Per-trial task variables (reward zone identity, environment, reward outcome) are
    derived exactly as in `behavior.get_reward_zones` / `behavior.get_trial_types`.
"""

import argparse
import glob
import os
import pickle
import sys
import time
import warnings
import multiprocessing as mp

import numpy as np
import scipy as sp
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter1d, minimum_filter1d

from pynwb import NWBHDF5IO

DATA_ROOT = '/app/data'

# ----------------------------------------------------------------------------------
# Constants taken from the reference code / methods
# ----------------------------------------------------------------------------------

# reward_relative.behavior.reward_zone_dict: 'X'->A, 'Y'->B, 'Z'->C (start, end in cm)
REWARD_ZONE_DICT = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_ORDER = ['A', 'B', 'C']
CHANGE_TRIAL = 30            # "Each switch occurred after 30 trials"
TRACK_LENGTH = 450.0         # cm

# reward_relative.utilities.default_dff_method + preprocessing.dff defaults
NEU_COEF = 0.7               # neuropil subtraction coefficient
BASELINE_WIN = 300           # frames (~19.4 s) for the maximin filter ("20 s sliding window")
BASELINE_SMOOTH_SIG = 15     # frames, Gaussian sigma along time before the maximin filter
DFF_SMOOTH_SIG = 2           # frames (~0.129 s) Gaussian smoothing of dF/F
OASIS_TAU = 0.7              # preprocessing.dff default (suite2p tau for fast GCaMP)
OASIS_BATCH = 2000

# Methods: "putative interneurons ... Pearson correlation of >0.5 between their dF/F
# timeseries and the animal's running speed"
INTERNEURON_SPEED_CORR_THR = 0.5

# Methods: lick-sensor error trials -- ">30% of the 0.0645 s imaging frame samples in the
# trial containing a cumulative lick count >2"
LICK_ERROR_FRAC_THR = 0.3
LICK_ERROR_COUNT_THR = 2

# Decoder output discretisation
DIST_EDGES = (-50.0, -10.0, 0.0, 10.0, 50.0)
POS_EDGES = (90.0, 180.0, 270.0, 360.0)
SPEED_EDGES = (2.0, 10.0, 20.0, 40.0)

INPUT_NAMES = ['time_from_trial_start', 'environment', 'trial_number',
               'previous_trial_rewarded']
OUTPUT_NAMES = ['reward_zone_distance', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm (in reward zone)',
     '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
    ['omitted', 'rewarded'],
]


# ----------------------------------------------------------------------------------
# Signal processing helpers (ports of the reference code)
# ----------------------------------------------------------------------------------

def nansmooth(a, sig, axis=None):
    """NaN-tolerant Gaussian smoothing (TwoPUtils.utilities.nansmooth / ut.nansmooth).

    `sig` may be a scalar (smooth along `axis`) or a per-axis list (n-d smoothing).
    """
    nan_inds = np.isnan(a)
    a_nanless = np.where(nan_inds, 0.0, a)
    one = np.ones(a.shape, dtype=a.dtype)
    one[nan_inds] = 0.001
    if np.isscalar(sig):
        a_nanless = gaussian_filter1d(a_nanless, sig, axis=axis)
        one = gaussian_filter1d(one, sig, axis=axis)
    else:
        a_nanless = gaussian_filter(a_nanless, sig)
        one = gaussian_filter(one, sig)
    return a_nanless / one


def dff_and_events(f, f_neu, trial_starts, teleports, frame_rate,
                   neu_coef=NEU_COEF, tau=OASIS_TAU):
    """Port of `reward_relative.preprocessing.dff` for the single-channel,
    `neuropil_method='subtract'`, `baseline_method='maximin'`, `deconvolve=True`,
    `keep_teleports=False` configuration used for this dataset.

    Args:
        f:        (n_cells, n_frames) raw ROI fluorescence
        f_neu:    (n_cells, n_frames) neuropil fluorescence
        trial_starts, teleports: frame indices of trial starts / teleports
        frame_rate: imaging rate per plane (Hz)

    Returns:
        dff:   (n_cells, n_frames) float32, NaN outside trials
        spks:  (n_cells, n_frames) float32 deconvolved "events", NaN outside trials

    suite2p's OASIS implementation -- the one the reference code calls -- is imported
    lazily so the parent process never initialises its OpenMP runtime before forking.
    """
    from suite2p.extraction import dcnv

    f = f.astype(np.float32, copy=False)
    f_neu = f_neu.astype(np.float32, copy=False)

    f_ = np.full(f.shape, np.nan, dtype=np.float32)
    f_neu_ = np.full(f_neu.shape, np.nan, dtype=np.float32)

    # keep_teleports=False -> only on-track samples, window [start-1, stop-1)
    slices = [slice(s - 1, t - 1) for s, t in zip(trial_starts, teleports)]
    for sl in slices:
        f_[:, sl] = f[:, sl]
        f_neu_[:, sl] = f_neu[:, sl]

    nanmask = ~np.isnan(f_[0, :])

    # neuropil correction
    f_ -= neu_coef * f_neu_

    flow = np.full(f_.shape, np.nan, dtype=np.float32)
    spks = np.full(f_.shape, np.nan, dtype=np.float32)

    for sl in slices:
        # add the per-trial neuropil mean back in so dF/F is not divided by a small number
        f_[:, sl] = f_[:, sl] + neu_coef * np.nanmean(f_neu_[:, sl], axis=1, keepdims=True)
        # maximin baseline: smooth, 20 s minimum filter, then same-window maximum filter
        base = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH_SIG])
        base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
        base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
        flow[:, sl] = base

    dff = np.full(f_.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    for sl in slices:
        smoothed = nansmooth(dff[:, sl], DFF_SMOOTH_SIG, axis=1)
        dff[:, sl] = smoothed
        spks[:, sl] = dcnv.oasis(np.ascontiguousarray(smoothed, dtype=np.float32),
                                 OASIS_BATCH, tau, frame_rate)

    return dff, spks


# ----------------------------------------------------------------------------------
# Task-variable helpers (ports of the reference code)
# ----------------------------------------------------------------------------------

def zone_labels_from_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Port of `reward_relative.behavior.get_reward_zones` label logic.

    Scene names are either '<Env>_Location<Z>' (fixed zone) or a switch scene
    containing '<Z1>_to' and ending in '<Z2>' (e.g. 'Env1_LocationB_to_A',
    'Env1_C_to_Env2_B'), in which case the zone changes after `change_trial` trials.
    """
    for z in ZONE_ORDER:
        if scene.endswith('Location' + z):
            return np.array([z] * ntrials)
    first = None
    for z in ZONE_ORDER:
        if z + '_to' in scene:
            first = z
    last = scene[-1]
    if first is None or last not in ZONE_ORDER:
        raise ValueError(f'Unrecognised scene name: {scene}')
    n0 = min(change_trial, ntrials)
    return np.array([first] * n0 + [last] * (ntrials - n0))


def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) from `pos` to the nearest point of [zone_start, zone_end].

    0 inside the zone, negative before it, positive after it.
    """
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d


def discretize_distance(d):
    """7 classes, exactly as specified by the decoder task."""
    out = np.full(d.shape, 3, dtype=np.int64)     # d == 0 -> in the reward zone
    out[d < 0] = 2                                # -10 <= d < 0
    out[d < DIST_EDGES[1]] = 1                    # -50 <= d < -10
    out[d < DIST_EDGES[0]] = 0                    # d < -50
    out[d > 0] = 4                                # 0 < d <= 10
    out[d > DIST_EDGES[3]] = 5                    # 10 < d <= 50
    out[d > DIST_EDGES[4]] = 6                    # d > 50
    return out


# ----------------------------------------------------------------------------------
# Per-session conversion
# ----------------------------------------------------------------------------------

def read_session(path):
    """Read everything needed from one NWB file using pynwb.

    Returns a dict of raw (unprocessed) arrays.
    """
    with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        scene = nwb.identifier.split('/')[-1]
        subject = nwb.subject.subject_id
        session_id = nwb.session_id

        b = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        ts = np.asarray(b['position'].timestamps[:], dtype=np.float64)
        beh = {k: np.asarray(b[k].data[:], dtype=np.float64) for k in
               ['position', 'speed', 'lick', 'reward_zone', 'environment',
                'trial number', 'trial_start', 'teleport', 'scanning']}
        reward_times = np.asarray(b['Reward'].timestamps[:], dtype=np.float64)

        oph = nwb.processing['ophys'].data_interfaces
        seg = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
        iscell = np.asarray(seg['iscell'].data[:])[:, 0].astype(bool)
        plane_idx = np.asarray(seg['planeIdx'].data[:]).astype(int)

        # Concatenate the per-plane RoiResponseSeries; their ROI ids index the shared
        # PlaneSegmentation table in order, so plane0 then plane1 reproduces its row order.
        plane_names = sorted(oph['Fluorescence'].roi_response_series.keys())
        f_list, fneu_list, roi_ids = [], [], []
        for pn in plane_names:
            rrs = oph['Fluorescence'].roi_response_series[pn]
            rid = np.asarray(rrs.rois.data[:], dtype=int)
            sel = iscell[rid]                       # only load curated cells
            roi_ids.append(rid[sel])
            f_list.append(np.asarray(rrs.data[:, :], dtype=np.float32)[:, sel].T)
            nrs = oph['Neuropil'].roi_response_series[pn]
            fneu_list.append(np.asarray(nrs.data[:, :], dtype=np.float32)[:, sel].T)
        f = np.concatenate(f_list, axis=0)
        f_neu = np.concatenate(fneu_list, axis=0)
        roi_ids = np.concatenate(roi_ids)
        imaging_rate = float(nwb.imaging_planes['ImagingPlane'].imaging_rate)
        location = nwb.imaging_planes['ImagingPlane'].location

    # A handful of the 2-plane files (10 of 152) carry one more imaging frame than the
    # 2P-aligned VR data. The VR alignment defines the common sample grid, so trim the
    # trailing frames of every stream to the shorter length.
    n = min(len(ts), f.shape[1])
    n_trimmed = max(len(ts), f.shape[1]) - n
    ts = ts[:n]
    beh = {k: v[:n] for k, v in beh.items()}
    f, f_neu = f[:, :n], f_neu[:, :n]

    return dict(path=path, scene=scene, subject=subject, session_id=session_id,
                ts=ts, reward_times=reward_times, f=f, f_neu=f_neu,
                roi_ids=roi_ids, plane_of_cell=plane_idx[roi_ids], n_trimmed=n_trimmed,
                imaging_rate=imaging_rate, location=location, **beh)


def convert_session(path, show_processing=False, plot_dir='/app', signal='dff'):
    """Convert one NWB session into per-trial neural / input / output arrays."""
    t0 = time.time()
    raw = read_session(path)
    t_read = time.time() - t0

    ts = raw['ts']
    pos = raw['position']
    speed = raw['speed']
    lick = raw['lick']
    rzone_flag = raw['reward_zone']
    morph = raw['environment']
    scanning = raw['scanning']

    si = np.where(raw['trial_start'] == 1)[0]
    ti = np.where(raw['teleport'] == 1)[0]
    assert len(si) == len(ti), f'{path}: {len(si)} starts vs {len(ti)} teleports'
    # guard against a trial starting on frame 0 (window would wrap around)
    keep = (si >= 1) & (ti <= len(ts))
    n_trials_out_of_range = int((~keep).sum())
    si, ti = si[keep], ti[keep]
    assert np.all(ti > si), f'{path}: teleport before trial start'
    ntrials_all = len(si)

    # The frame period is the sampling interval of the (already 2P-aligned) VR data.
    dt = float(np.median(np.diff(ts)))
    frame_rate = 1.0 / dt

    # ---- neural: dF/F + deconvolved events (reference preprocessing.dff) -------------
    t1 = time.time()
    dff, events = dff_and_events(raw['f'], raw['f_neu'], si, ti, frame_rate)
    t_dff = time.time() - t1

    valid = ~np.isnan(dff[0, :])          # frames inside a trial window

    # ---- neuron curation: putative interneurons (r(dF/F, speed) > 0.5) ---------------
    sp_v = speed[valid].astype(np.float32)
    d_v = dff[:, valid]
    sp_c = sp_v - sp_v.mean()
    d_c = d_v - d_v.mean(axis=1, keepdims=True)
    denom = np.sqrt((d_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        r_speed = (d_c @ sp_c) / denom
    r_speed = np.nan_to_num(r_speed, nan=0.0)
    keep_cells = r_speed <= INTERNEURON_SPEED_CORR_THR
    n_interneuron = int((~keep_cells).sum())
    dff = dff[keep_cells]
    events = events[keep_cells]
    plane_of_cell = raw['plane_of_cell'][keep_cells]

    # ---- per-trial task variables (reference behavior.get_trial_types/_reward_zones) --
    reward_idx = np.searchsorted(ts, raw['reward_times'])
    zone_lab = zone_labels_from_scene(raw['scene'], ntrials_all)
    zone_start = np.array([REWARD_ZONE_DICT[z][0] for z in zone_lab])
    zone_end = np.array([REWARD_ZONE_DICT[z][1] for z in zone_lab])
    zone_code = np.array([ZONE_ORDER.index(z) for z in zone_lab], dtype=np.int64)

    isreward = np.zeros(ntrials_all, dtype=np.int64)
    trial_morph = np.zeros(ntrials_all, dtype=np.int64)
    lick_error = np.zeros(ntrials_all, dtype=bool)
    scan_bad = np.zeros(ntrials_all, dtype=bool)
    for i, (s, e) in enumerate(zip(si, ti)):
        has_reward = np.any((reward_idx >= s) & (reward_idx < e))
        isreward[i] = int(has_reward and np.any(rzone_flag[s:e] > 0))
        mvals = np.unique(morph[s:e])
        trial_morph[i] = int(mvals[0]) if len(mvals) == 1 else int(np.round(np.median(morph[s:e])))
        L = lick[s:e]
        lick_error[i] = (np.sum(L > LICK_ERROR_COUNT_THR) / len(L)) > LICK_ERROR_FRAC_THR
        scan_bad[i] = np.any(scanning[s - 1:e - 1] != 1)

    # previous-trial outcome; index 0 is undefined -> 0 (see CONVERSION_NOTES Step 5)
    prev_reward = np.zeros(ntrials_all, dtype=np.int64)
    prev_reward[1:] = isreward[:-1]

    # ---- assemble per-trial arrays ---------------------------------------------------
    neural_trials, input_trials, output_trials = [], [], []
    kept_trial_idx = []
    n_dropped_lick, n_dropped_scan, n_dropped_nan = 0, 0, 0
    for i, (s, e) in enumerate(zip(si, ti)):
        if lick_error[i]:
            n_dropped_lick += 1
            continue
        if scan_bad[i]:
            n_dropped_scan += 1
            continue
        sl = slice(s - 1, e - 1)
        act = (events if signal == 'events' else dff)[:, sl]
        if act.shape[1] < 2 or not np.all(np.isfinite(act)):
            n_dropped_nan += 1
            continue

        T = act.shape[1]
        p = pos[sl]
        sp_t = speed[sl]
        lk = lick[sl]

        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = ts[sl] - ts[s]                 # t = 0 at the trial_start frame
        inp[1] = trial_morph[i]
        inp[2] = i
        inp[3] = prev_reward[i]

        out = np.empty((6, T), dtype=np.int64)
        out[0] = discretize_distance(signed_distance_to_zone(p, zone_start[i], zone_end[i]))
        out[1] = np.digitize(p, POS_EDGES)
        out[2] = np.digitize(sp_t, SPEED_EDGES)
        out[3] = (lk > 0).astype(np.int64)
        out[4] = zone_code[i]
        out[5] = isreward[i]

        neural_trials.append(np.ascontiguousarray(act, dtype=np.float32))
        input_trials.append(inp)
        output_trials.append(out)
        kept_trial_idx.append(i)

    info = dict(
        path=path, subject=raw['subject'], session_id=raw['session_id'],
        scene=raw['scene'], location=raw['location'],
        n_trials_all=ntrials_all, n_trials_kept=len(neural_trials),
        n_dropped_lick=n_dropped_lick, n_dropped_scan=n_dropped_scan,
        n_dropped_nan=n_dropped_nan,
        n_cells_iscell=int(raw['f'].shape[0]), n_interneuron=n_interneuron,
        n_frames_trimmed=int(raw['n_trimmed']),
        n_trials_out_of_range=n_trials_out_of_range,
        n_cells=int(dff.shape[0]), dt=dt, imaging_rate=raw['imaging_rate'],
        n_omission=int((isreward == 0).sum()),
        zone_labels=''.join(zone_lab), morph=trial_morph.tolist(),
        t_read=t_read, t_dff=t_dff, t_total=time.time() - t0,
        dff_median=float(np.nanmedian(dff[:, valid])),
        dff_p99=float(np.nanpercentile(dff[:, valid], 99)),
        events_frac_zero=float(np.mean(events[:, valid] == 0)),
    )

    if show_processing:
        plot_processing(raw, dff, events, si, ti, valid, zone_start, zone_end,
                        zone_lab, isreward, kept_trial_idx, neural_trials,
                        input_trials, output_trials, r_speed, keep_cells,
                        plot_dir, signal)

    brain_region_idx = np.zeros(dff.shape[0], dtype=np.int64)
    return dict(neural=neural_trials, input=input_trials, output=output_trials,
                brain_region_idx=brain_region_idx, info=info,
                plane_of_cell=plane_of_cell)


# ----------------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------------

def plot_processing(raw, dff, events, si, ti, valid, zone_start, zone_end, zone_lab,
                    isreward, kept_trial_idx, neural_trials, input_trials,
                    output_trials, r_speed, keep_cells, plot_dir, signal='dff'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tag = f"{raw['subject']}_ses-{raw['session_id']}"
    ts, pos, speed, lick = raw['ts'], raw['position'], raw['speed'], raw['lick']

    fig, axes = plt.subplots(9, 1, figsize=(20, 26))

    # --- 1. raw F / neuropil / baseline-corrected dF/F for one example cell ----------
    cell = int(np.nanargmax(np.nanstd(dff, axis=1)))
    w = slice(si[0] - 1, ti[4] - 1)          # first five trials
    ax = axes[0]
    ax.plot(ts[w], raw['f'][cell, w], lw=0.7, label='raw F')
    ax.plot(ts[w], raw['f_neu'][cell, w], lw=0.7, label='neuropil')
    for s, e in zip(si[:5], ti[:5]):
        ax.axvspan(ts[s], ts[e - 1], color='0.9', zorder=0)
    ax.set_title(f'{tag}: raw fluorescence, example cell {cell} (grey = trials)')
    ax.legend(loc='upper right')
    ax.set_ylabel('a.u.')

    ax = axes[1]
    ax.plot(ts[w], dff[cell, w], lw=0.8, label='dF/F (smoothed)')
    ax.plot(ts[w], events[cell, w], lw=0.8, label='events (OASIS)')
    for s, e in zip(si[:5], ti[:5]):
        ax.axvspan(ts[s], ts[e - 1], color='0.9', zorder=0)
    ax.set_title('dF/F and deconvolved events (NaN between trials = gaps)')
    ax.legend(loc='upper right')

    # --- 2. dF/F distribution and interneuron filter ---------------------------------
    ax = axes[2]
    ax.hist(dff[:, valid].ravel()[::101], bins=200, log=True)
    ax.set_title(f'dF/F distribution (in-trial samples), median='
                 f'{np.nanmedian(dff[:, valid]):.3f}')
    ax.set_xlabel('dF/F')

    ax = axes[3]
    ax.hist(r_speed, bins=100)
    ax.axvline(INTERNEURON_SPEED_CORR_THR, color='r')
    ax.set_title(f'Pearson r(dF/F, speed) per cell; {int((~keep_cells).sum())} of '
                 f'{len(keep_cells)} excluded as putative interneurons (r > 0.5)')

    # --- 3. behaviour with trial boundaries and reward zone --------------------------
    w2 = slice(si[0] - 1, ti[7] - 1)
    ax = axes[4]
    ax.plot(ts[w2], pos[w2], lw=1, color='k', label='position')
    for j, (s, e) in enumerate(zip(si[:8], ti[:8])):
        ax.axvline(ts[s], color='g', lw=0.8)
        ax.axvline(ts[e - 1], color='r', lw=0.8)
        ax.fill_between([ts[s], ts[e - 1]], zone_start[j], zone_end[j],
                        color='orange', alpha=.3)
    ax.set_ylabel('position (cm)')
    ax.set_title('position, trial starts (green) / ends (red), reward zone (orange)')

    ax = axes[5]
    ax.plot(ts[w2], speed[w2], lw=1, label='speed (cm/s)')
    ax.plot(ts[w2], lick[w2] * 5, lw=1, label='lick count x5')
    rt = raw['reward_times']
    rt = rt[(rt >= ts[w2.start]) & (rt <= ts[w2.stop])]
    ax.plot(rt, np.zeros_like(rt) + 50, 'v', color='m', label='reward')
    ax.legend(loc='upper right')
    ax.set_title('speed / licks / reward delivery')

    # --- 4. discretisation check: continuous vs class, one trial ---------------------
    tr = min(5, len(neural_trials) - 1)
    orig = kept_trial_idx[tr]
    sl = slice(si[orig] - 1, ti[orig] - 1)
    tt = input_trials[tr][0]
    d = signed_distance_to_zone(pos[sl], zone_start[orig], zone_end[orig])
    ax = axes[6]
    ax.plot(tt, pos[sl], 'k', label='position (cm)')
    ax.plot(tt, d, 'b', label='distance to reward zone (cm)')
    for y in (-50, -10, 0, 10, 50):
        ax.axhline(y, color='b', ls=':', lw=.5)
    ax2 = ax.twinx()
    ax2.step(tt, output_trials[tr][0], 'r', where='post', label='distance class')
    ax2.step(tt, output_trials[tr][1], 'g', where='post', label='position class')
    ax2.set_ylabel('class')
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    ax.set_title(f'trial {orig}: discretisation of distance-to-zone and position '
                 f'(zone {zone_lab[orig]}, rewarded={isreward[orig]})')
    ax.set_xlabel('time from trial start (s)')

    ax = axes[7]
    ax.plot(tt, speed[sl], 'k', label='speed (cm/s)')
    for y in (2, 10, 20, 40):
        ax.axhline(y, color='b', ls=':', lw=.5)
    ax2 = ax.twinx()
    ax2.step(tt, output_trials[tr][2], 'r', where='post', label='speed class')
    ax2.step(tt, output_trials[tr][3], 'g', where='post', label='lick class')
    ax.plot(tt, lick[sl], 'm', lw=.8, label='lick count')
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    ax.set_title('discretisation of speed and licking')
    ax.set_xlabel('time from trial start (s)')

    # --- 5. final aligned trial: neural raster + outputs ------------------------------
    ax = axes[8]
    act = neural_trials[tr]
    order = np.argsort(np.argmax(act, axis=1))
    ax.imshow(act[order][:200], aspect='auto', interpolation='nearest',
              extent=[tt[0], tt[-1], 200, 0], cmap='magma',
              vmax=np.percentile(act, 99.5))
    ax2 = ax.twinx()
    ax2.plot(tt, pos[sl], 'c', lw=1.5)
    ax2.set_ylabel('position (cm)', color='c')
    ax.set_xlabel('time from trial start (s)')
    ax.set_ylabel('neuron (sorted)')
    ax.set_title(f'converted trial: neural data as stored ({signal}, first 200 cells) with '
                 'position overlaid -- checks temporal alignment')

    fig.tight_layout()
    out = os.path.join(plot_dir, f'processing_{tag}.png')
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f'  wrote {out}', flush=True)


# ----------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------

def _worker(args):
    path, show, plot_dir, signal = args
    warnings.filterwarnings('ignore')
    try:
        return convert_session(path, show_processing=show, plot_dir=plot_dir,
                               signal=signal)
    except Exception as exc:          # keep the run alive, report at the end
        import traceback
        return dict(error=f'{path}: {exc}\n{traceback.format_exc()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nproc', type=int, default=16)
    # dF/F is the default neural signal: it is the paper's own `preprocessing.dff`
    # output, and it decodes markedly better than the OASIS-deconvolved "events" for a
    # per-timepoint linear decoder (see CONVERSION_NOTES Step 7).
    ap.add_argument('--signal', choices=['dff', 'events'], default='dff')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*.nwb')))
    if args.sample:
        # two sessions from different mice, one single-plane and one 2-plane
        files = [f for f in files if 'sub-m11_ses-03' in f or 'sub-m17_ses-08' in f]
    print(f'Converting {len(files)} sessions with {args.nproc} workers', flush=True)

    show_n = 2 if args.show_processing else 0
    tasks = [(f, i < show_n, os.path.dirname(os.path.abspath(args.outfile)) or '.',
              args.signal) for i, f in enumerate(files)]

    t0 = time.time()
    results = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(args.nproc, len(files))) as pool:
        for k, res in enumerate(pool.imap(_worker, tasks)):
            if 'error' in res:
                print('ERROR', res['error'], flush=True)
                raise RuntimeError(res['error'])
            i = res['info']
            el = time.time() - t0
            print(f"[{k+1}/{len(files)}] {i['subject']} ses-{i['session_id']} "
                  f"{i['scene']}: {i['n_cells']} cells "
                  f"({i['n_cells_iscell']} iscell, {i['n_interneuron']} interneurons), "
                  f"{i['n_trials_kept']}/{i['n_trials_all']} trials "
                  f"(dropped lick={i['n_dropped_lick']} scan={i['n_dropped_scan']} "
                  f"nan={i['n_dropped_nan']}) | read {i['t_read']:.1f}s "
                  f"dff {i['t_dff']:.1f}s total {i['t_total']:.1f}s | "
                  f"elapsed {el/60:.1f} min", flush=True)
            results.append(res)

    # ------------------------------------------------------------------ assemble
    subjects = sorted({r['info']['subject'] for r in results},
                      key=lambda s: int(s[1:]))
    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['info']['subject']) for r in results],
                                dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': [r['brain_region_idx'] for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
    }

    dts = np.array([r['info']['dt'] for r in results])
    ntr = np.array([r['info']['n_trials_kept'] for r in results])
    ncell = np.array([r['info']['n_cells'] for r in results])
    trial_durs = np.concatenate([[inp[0, -1] for inp in r['input']] for r in results])

    data['metadata'] = {
        'task_description': (
            'Head-fixed mice run laps on a 450 cm virtual linear track for a hidden 50 cm '
            'reward zone at one of three locations (A 80-130 cm, B 200-250 cm, C 320-370 cm) '
            'in one of two visual environments (ENV1/ENV2). The reward zone is moved to a new '
            'location after 30 trials on "switch" days, and reward is randomly omitted on ~15% '
            'of trials. Outputs to decode are the signed distance to the reward zone, absolute '
            'track position, running speed, licking, which reward zone is active, and whether '
            'the trial was rewarded. Inputs are time within the trial, the environment, the '
            'trial number and the previous trial\'s reward outcome.'),
        'temporal_alignment_event': (
            'start of trial: the VR "trial_start" flag, i.e. entry into the linear track at '
            'position 0 cm'),
        'off_start': float(-np.median(dts)),
        'off_end': None,
        'off_end_note': (
            'Trials end at the teleport out of the track, so trial duration varies: '
            f'median {np.median(trial_durs):.1f} s, 5-95 pct '
            f'{np.percentile(trial_durs, 5):.1f}-{np.percentile(trial_durs, 95):.1f} s, '
            f'max {trial_durs.max():.1f} s.'),
        'time_bin_size': float(np.median(dts) * 1000.0),
        'neural_signal': (
            'dF/F computed from the raw suite2p fluorescence exactly as in '
            'reward_relative.preprocessing.dff: neuropil subtraction (coefficient 0.7, '
            'per-trial neuropil mean added back), per-trial "maximin" baseline (Gaussian '
            'sigma 15 frames, then a 300-frame ~20 s minimum filter followed by a '
            '300-frame maximum filter), dF/F = (F - baseline)/|baseline|, smoothed with a '
            '2-sample (~0.129 s) Gaussian'
            if args.signal == 'dff' else
            'deconvolved calcium "events": the dF/F above, further deconvolved with '
            'suite2p OASIS (tau=0.7), as in reward_relative.preprocessing.dff'),
        'recording_modality': 'two-photon calcium imaging (GCaMP7f) of dorsal CA1',
        'neuron_curation': (
            'suite2p manual curation (iscell == 1); putative interneurons removed by '
            'Pearson r(dF/F, running speed) > 0.5'),
        'trial_curation': (
            'trials with lick-sensor errors (>30% of frames with a cumulative lick count > 2) '
            'removed, as in the paper; only on-track samples are kept '
            '(frames [trial_start-1, teleport-1), the reference code window), so the '
            'inter-trial teleport period is excluded'),
        'session_info': [
            {k: r['info'][k] for k in
             ['subject', 'session_id', 'scene', 'n_cells', 'n_cells_iscell',
              'n_interneuron', 'n_trials_all', 'n_trials_kept', 'n_dropped_lick',
              'n_omission', 'zone_labels', 'dt']}
            for r in results],
        'n_sessions': len(results),
        'n_trials_total': int(ntr.sum()),
        'n_neurons_total': int(ncell.sum()),
        'source': ('Sosa, Plitt & Giocomo (2025) Nature Neuroscience, '
                   '"A flexible hippocampal population code for experience relative to '
                   'reward"; DANDI dandiset 001361'),
    }

    print(f'\nWriting {args.outfile} ...', flush=True)
    t1 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f'wrote {os.path.getsize(args.outfile)/1e9:.2f} GB in {time.time()-t1:.0f}s')

    # ------------------------------------------------------------------ summary
    print('\n==================== SUMMARY ====================')
    print(f'sessions            : {len(results)}')
    print(f'subjects            : {len(subjects)} {subjects}')
    print(f'neurons total       : {ncell.sum()}  (mean {ncell.mean():.1f}, '
          f'min {ncell.min()}, max {ncell.max()})')
    print(f'iscell total        : {sum(r["info"]["n_cells_iscell"] for r in results)}')
    n_int = sum(r['info']['n_interneuron'] for r in results)
    n_isc = sum(r['info']['n_cells_iscell'] for r in results)
    frac_int = np.array([r['info']['n_interneuron'] / max(r['info']['n_cells_iscell'], 1)
                         for r in results])
    print(f'interneurons removed: {n_int} ({100*n_int/n_isc:.2f}% of iscell; '
          f'per-session mean {100*frac_int.mean():.2f} +- {100*frac_int.std():.2f}%)')
    print(f'trials kept         : {ntr.sum()} of '
          f'{sum(r["info"]["n_trials_all"] for r in results)} '
          f'(mean/session {ntr.mean():.2f} +- {ntr.std():.2f})')
    print(f'trials dropped lick : {sum(r["info"]["n_dropped_lick"] for r in results)}')
    print(f'trials dropped scan : {sum(r["info"]["n_dropped_scan"] for r in results)}')
    print(f'trials dropped nan  : {sum(r["info"]["n_dropped_nan"] for r in results)}')
    print(f'trials out of range : {sum(r["info"]["n_trials_out_of_range"] for r in results)}')
    print(f'frames trimmed      : {sum(r["info"]["n_frames_trimmed"] for r in results)} '
          f'(sessions with an extra ophys frame: '
          f'{sum(1 for r in results if r["info"]["n_frames_trimmed"])})')
    print(f'time bin            : {np.median(dts)*1000:.4f} ms '
          f'(all sessions equal: {np.allclose(dts, dts[0])})')
    print(f'timepoints total    : {sum(sum(t.shape[1] for t in r["neural"]) for r in results)}')

    # output distributions
    counts = [np.zeros(len(v), dtype=np.int64) for v in OUTPUT_VALUES]
    for r in results:
        for out in r['output']:
            for d in range(len(OUTPUT_VALUES)):
                c = np.bincount(out[d], minlength=len(OUTPUT_VALUES[d]))
                counts[d] += c
    print('\noutput distributions (fraction of timepoints):')
    for d, name in enumerate(OUTPUT_NAMES):
        frac = counts[d] / counts[d].sum()
        print(f'  {name}: ' + ', '.join(f'{v}={f:.4f}'
                                        for v, f in zip(OUTPUT_VALUES[d], frac)))
    # per-trial fractions for the per-trial outputs
    zl = ''.join(r['info']['zone_labels'] for r in results)
    print(f'\nper-trial reward outcome: rewarded fraction = '
          f'{np.mean([o[5, 0] for r in results for o in r["output"]]):.4f}')
    print(f'per-trial environment ENV2 fraction = '
          f'{np.mean([i[1, 0] for r in results for i in r["input"]]):.4f}')
    print(f'zone label counts (all trials incl. dropped): '
          f'A={zl.count("A")} B={zl.count("B")} C={zl.count("C")}')
    print('\ninput ranges:')
    for d, name in enumerate(INPUT_NAMES):
        lo = min(float(i[d].min()) for r in results for i in r['input'])
        hi = max(float(i[d].max()) for r in results for i in r['input'])
        print(f'  {name}: [{lo:.4f}, {hi:.4f}]')
    print(f'\ntotal time {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
