"""
Convert the Sosa, Plitt & Giocomo (2025) CA1 2-photon VR dataset (DANDI 001361, NWB)
into the decoder-ready pickle format.

Processing follows the reference code (https://github.com/GiocomoLab/Sosa_et_al_2024,
copy in /app/code):
  * neural activity  = OASIS deconvolution of the per-trial maximin dF/F
                       (reward_relative.preprocessing.dff, deconvolve=True -> 'events')
  * trials           = [trial_start-1, teleport-1)  (reward_relative.glmUtils.get_timeseries_data)
  * reward zones     = reward_relative.behavior.get_reward_zones (scene name + switch at trial 30)
  * rewarded trials  = reward_relative.behavior.get_trial_types
  * cell curation    = suite2p iscell + putative-interneuron exclusion
                       (reward_relative.spatial.is_putative_interneuron, r(dF/F, speed) > 0.5)

Usage:  python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import (gaussian_filter, gaussian_filter1d, maximum_filter1d,
                           minimum_filter1d)

from pynwb import NWBHDF5IO
from suite2p.extraction import dcnv

# ----------------------------------------------------------------------------------
# constants taken from the reference code / methods
# ----------------------------------------------------------------------------------
DATA_DIR = '/app/data'
# reward_relative.behavior.reward_zone_dict: 'X'/'Y'/'Z' are the zones used in this task
ZONE_DICT = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
CHANGE_TRIAL = 30           # reward zone switches after 30 trials (methods)
TRACK_LENGTH = 450.0        # cm
NEU_COEF = 0.7              # neuropil coefficient (preprocessing.dff)
TAU = 0.7                   # OASIS calcium decay constant (suite2p ops in reference notebook)
BASELINE_SMOOTH = 15        # samples, gaussian sigma for baseline smoothing (preprocessing.dff)
BASELINE_WINDOW = 300       # samples (~20 s at 15.5 Hz) min/max filter (preprocessing.dff)
DFF_SMOOTH = 2              # samples, gaussian sigma applied to dF/F (preprocessing.dff, methods)
INTERNEURON_R_THRESH = 0.5  # r(dF/F, speed) > 0.5 -> putative interneuron (methods)
LICK_ERROR_FRAC = 0.30      # >30% of frames with cumulative lick > 2 -> lick sensor error (methods)
LICK_ERROR_COUNT = 2

# output discretisation (from the Decoder Task specification)
RZ_DIST_EDGES = [-50.0, -10.0, 0.0]      # handled explicitly, see discretize_reward_distance
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
SPEED_EDGES = [2.0, 10.0, 20.0, 40.0]

# Animals/days for which the laser was NOT blanked during the teleport period, so the reference
# code computes dF/F with keep_teleports=True (reward_relative.teleport_metadata.teleport_sessions,
# and methods: "except for mice m11-m14 on task days 1, 7, 8 and 14, and mice m15-m19 on day 1 and
# all switch days, for which laser power was maintained and imaging continued throughout the teleport").
TELEPORT_SESSIONS = {
    'm10': [1, 7, 8, 14, 15],
    'm11': [1, 7, 8, 14, 15],
    'm12': [1, 7, 8, 14, 15],
    'm13': [1, 7, 8, 14, 15],
    'm14': [1, 7, 8, 14, 15],
    'm15': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm17': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm18': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm19': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
}

INPUT_NAMES = ['time_from_trial_start', 'environment', 'trial_number',
               'previous_trial_outcome']
OUTPUT_NAMES = ['reward_zone_distance', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
     '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
    ['omitted', 'rewarded'],
]


# ----------------------------------------------------------------------------------
# helpers ported from the reference code
# ----------------------------------------------------------------------------------
def nansmooth(a, sig, axis=-1):
    """NaN-aware gaussian smoothing (TwoPUtils.utilities.nansmooth / ut.nansmooth)."""
    V = a.copy()
    V[np.isnan(a)] = 0
    W = np.ones_like(a)
    W[np.isnan(a)] = 0
    if np.iterable(sig):
        VV = gaussian_filter(V, sig)
        WW = gaussian_filter(W, sig)
    else:
        VV = gaussian_filter1d(V, sig, axis=axis)
        WW = gaussian_filter1d(W, sig, axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        return VV / WW


def get_reward_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label, replicating reward_relative.behavior.get_reward_zones."""
    if scene.endswith('LocationA'):
        return ['A'] * ntrials
    if scene.endswith('LocationB'):
        return ['B'] * ntrials
    if scene.endswith('LocationC'):
        return ['C'] * ntrials
    for first in ('A', 'B', 'C'):
        if f'{first}_to' in scene:
            second = scene[-1]
            if second not in ZONE_DICT:
                raise ValueError(f'unrecognised switch scene {scene}')
            n0 = min(change_trial, ntrials)
            return [first] * n0 + [second] * (ntrials - n0)
    raise ValueError(f'unrecognised scene {scene}')


def baseline_segments(starts, stops, keep_teleports):
    """Segments over which the per-trial baseline/deconvolution are computed.

    Port of the start_inds/stop_inds logic in reward_relative.preprocessing.dff():
    with keep_teleports=True the segment for each trial starts 1 sample after the previous
    teleport (the ITI was imaged), otherwise it starts at the trial start.
    """
    if keep_teleports:
        seg_starts = np.append(starts[0], np.asarray(stops[:-1]) + 2)
    else:
        seg_starts = np.asarray(starts)
    return np.maximum(seg_starts, 1), np.asarray(stops)


def compute_events(F, Fneu, starts, stops, fs, keep_teleports=False):
    """dF/F + OASIS deconvolution, port of reward_relative.preprocessing.dff().

    F, Fneu : (n_roi, n_frames) float32
    starts, stops : trial_start / teleport frame indices
    Returns (dff, events), both (n_roi, n_frames) with NaN outside trials.
    """
    starts, stops = baseline_segments(starts, stops, keep_teleports)
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s - 1:e - 1] = F[:, s - 1:e - 1]
        fneu_[:, s - 1:e - 1] = Fneu[:, s - 1:e - 1]

    nanmask = ~np.isnan(f_[0, :])

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        sl = slice(s - 1, e - 1)
        # add back the per-trial neuropil mean so dF/F is not divided by tiny baselines
        f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
        seg = nansmooth(f_[:, sl], [0, BASELINE_SMOOTH])
        seg = minimum_filter1d(seg, BASELINE_WINDOW, axis=-1)
        seg = maximum_filter1d(seg, BASELINE_WINDOW, axis=-1)
        flow[:, sl] = seg

    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        sl = slice(s - 1, e - 1)
        dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
        events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl]), 2000, TAU, fs)
    return dff, events


def discretize_reward_distance(pos, zone_start, zone_end):
    """Signed distance to the nearest point of the reward zone -> 7 categories."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start          # negative: before the zone
    d[after] = pos[after] - zone_end              # positive: past the zone
    cat = np.zeros(pos.shape, dtype=np.int64)
    cat[d < -50] = 0
    cat[(d >= -50) & (d < -10)] = 1
    cat[(d >= -10) & (d < 0)] = 2
    cat[d == 0] = 3
    cat[(d > 0) & (d <= 10)] = 4
    cat[(d > 10) & (d <= 50)] = 5
    cat[d > 50] = 6
    return cat, d


# ----------------------------------------------------------------------------------
# per-session conversion
# ----------------------------------------------------------------------------------
def read_nwb_session(path):
    """Read everything needed from one NWB file using pynwb."""
    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        oph = nwb.processing['ophys'].data_interfaces

        d = {}
        d['subject'] = nwb.subject.subject_id
        d['identifier'] = nwb.identifier
        d['scene'] = nwb.identifier.split('/')[-1]
        d['date'] = nwb.identifier.split('/')[-2]
        d['session_start_time'] = str(nwb.session_start_time)
        d['brain_region'] = nwb.imaging_planes['ImagingPlane'].location

        d['time'] = np.asarray(beh['position'].timestamps[:], dtype=np.float64)
        d['pos'] = np.asarray(beh['position'].data[:], dtype=np.float64)
        d['speed'] = np.asarray(beh['speed'].data[:], dtype=np.float64)
        d['lick'] = np.asarray(beh['lick'].data[:], dtype=np.float64)
        d['rzone'] = np.asarray(beh['reward_zone'].data[:], dtype=np.float64)
        d['env'] = np.asarray(beh['environment'].data[:], dtype=np.float64)
        d['trialnum'] = np.asarray(beh['trial number'].data[:], dtype=np.float64)
        d['scanning'] = np.asarray(beh['scanning'].data[:], dtype=np.float64)
        d['trial_starts'] = np.where(np.asarray(beh['trial_start'].data[:]) > 0)[0]
        d['teleports'] = np.where(np.asarray(beh['teleport'].data[:]) > 0)[0]
        d['reward_times'] = np.asarray(beh['Reward'].timestamps[:], dtype=np.float64)

        planes = sorted(oph['Fluorescence'].roi_response_series.keys())
        Fs, Fneus, roi_idx = [], [], []
        for pl in planes:
            rrs = oph['Fluorescence'].roi_response_series[pl]
            Fs.append(np.asarray(rrs.data[:], dtype=np.float32))
            Fneus.append(np.asarray(oph['Neuropil'].roi_response_series[pl].data[:],
                                    dtype=np.float32))
            roi_idx.append(np.asarray(rrs.rois.data[:], dtype=np.int64))
        d['scan_rate'] = float(oph['Fluorescence'].roi_response_series[planes[0]].rate)
        d['n_planes'] = len(planes)
        d['fs'] = d['scan_rate'] / len(planes)     # per-plane sampling rate
        d['F'] = np.concatenate(Fs, axis=1).T      # (n_roi, n_frames)
        d['Fneu'] = np.concatenate(Fneus, axis=1).T
        roi_idx = np.concatenate(roi_idx)
        ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
        iscell = np.asarray(ps['iscell'].data[:])[:, 0]
        d['iscell'] = iscell[roi_idx] == 1
        d['planeIdx'] = np.asarray(ps['planeIdx'].data[:])[roi_idx]

    # In 10 sessions (all 2-plane m17/m18 recordings) the behavioural series are exactly one
    # sample shorter than the ophys series -- the same "one frame correction ... scan stopping
    # mid frame" that the reference VR-alignment code warns about. Truncate every stream to the
    # common length so neural and behaviour stay sample-aligned. All trial_start/teleport indices
    # lie within the shorter length, so no trial is affected.
    nB = len(d['time'])
    nF = d['F'].shape[1]
    n = min(nB, nF)
    d['n_trunc'] = max(nB, nF) - n
    if d['n_trunc'] > 0:
        for k in ('time', 'pos', 'speed', 'lick', 'rzone', 'env', 'trialnum', 'scanning'):
            d[k] = d[k][:n]
        d['F'] = d['F'][:, :n]
        d['Fneu'] = d['Fneu'][:, :n]
        d['trial_starts'] = d['trial_starts'][d['trial_starts'] < n]
        d['teleports'] = d['teleports'][d['teleports'] < n]
        d['trial_starts'] = d['trial_starts'][:len(d['teleports'])]
    return d


def convert_session(path, show_processing=False, verbose=True):
    """Convert one NWB session into (neural, input, output, info) lists."""
    t_start = time.time()
    raw = read_nwb_session(path)
    t_load = time.time() - t_start

    starts = raw['trial_starts']
    stops = raw['teleports']
    assert len(starts) == len(stops), 'trial_start / teleport count mismatch'
    assert np.all(stops > starts), 'teleport before trial start'
    # the reference indexes [start-1:stop-1]; guard against start == 0
    starts = np.maximum(starts, 1)
    ntrials_all = len(starts)

    # --- neural: iscell ROIs, dF/F + OASIS (reference preprocessing.dff) --------------
    t0 = time.time()
    F = raw['F'][raw['iscell']]
    Fneu = raw['Fneu'][raw['iscell']]
    # experiment day = session number in the file name (e.g. sub-m11_ses-03 -> day 3)
    exp_day = int(os.path.basename(path).split('_ses-')[1].split('_')[0])
    keep_teleports = exp_day in TELEPORT_SESSIONS.get(raw['subject'], [])
    dff, events = compute_events(F, Fneu, starts, stops, raw['fs'],
                                 keep_teleports=keep_teleports)
    t_dff = time.time() - t0

    # --- putative interneuron exclusion (spatial.is_putative_interneuron) -------------
    nanmask = ~np.isnan(dff[0, :])
    speed_all = raw['speed']
    sp = speed_all[nanmask]
    dm = dff[:, nanmask]
    dm_c = dm - dm.mean(axis=1, keepdims=True)
    sp_c = sp - sp.mean()
    denom = np.sqrt((dm_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_corr = (dm_c @ sp_c) / denom
    is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH
    keep_cells = ~is_int
    events = events[keep_cells]
    n_neurons = int(keep_cells.sum())

    # --- per-trial behavioural variables ----------------------------------------------
    pos = raw['pos']
    lick = raw['lick']
    rzone = raw['rzone']
    env = raw['env']
    time_v = raw['time']
    rew_frames = np.searchsorted(time_v, raw['reward_times'])
    rew_frames = rew_frames[rew_frames < len(time_v)]

    labels = get_reward_zone_labels(raw['scene'], ntrials_all)
    zone_code = {'A': 0, 'B': 1, 'C': 2}

    # rewarded / omission per trial (behavior.get_trial_types)
    isreward = np.zeros(ntrials_all, dtype=np.int64)
    lick_error = np.zeros(ntrials_all, dtype=bool)
    env_trial = np.zeros(ntrials_all, dtype=np.int64)
    for i, (s, e) in enumerate(zip(starts, stops)):
        sl = slice(s - 1, e - 1)
        has_rew = np.any((rew_frames >= s - 1) & (rew_frames < e - 1))
        in_zone = np.any(rzone[sl] > 0)
        isreward[i] = int(bool(has_rew) and bool(in_zone))
        lk = lick[sl]
        lick_error[i] = lk.size > 0 and (np.sum(lk > LICK_ERROR_COUNT) / lk.size) > LICK_ERROR_FRAC
        ev = env[sl]
        ev = ev[ev >= 0]
        env_trial[i] = int(np.round(np.median(ev))) if ev.size else 0

    # --- assemble trials ---------------------------------------------------------------
    neural, inputs, outputs = [], [], []
    kept_trials = []
    for i, (s, e) in enumerate(zip(starts, stops)):
        if lick_error[i]:
            continue                      # lick sensor error (methods): licks unusable
        sl = slice(s - 1, e - 1)
        T = (e - 1) - (s - 1)
        if T < 2:
            continue
        ev_trial = events[:, sl]
        if not np.all(np.isfinite(ev_trial)):
            continue                      # should not happen; guard against bad frames

        p = pos[sl]
        sp_t = speed_all[sl]
        lk = lick[sl]

        zlab = labels[i]
        z0, z1 = ZONE_DICT[zlab]

        # inputs
        t_rel = time_v[sl] - time_v[s - 1]
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = t_rel
        inp[1] = env_trial[i]
        inp[2] = i
        inp[3] = isreward[i - 1] if i > 0 else 0

        # outputs
        rz_cat, _ = discretize_reward_distance(p, z0, z1)
        pos_cat = np.digitize(p, POS_EDGES).astype(np.int64)
        spd_cat = np.digitize(sp_t, SPEED_EDGES).astype(np.int64)
        lick_cat = (lk > 0).astype(np.int64)
        out = np.empty((6, T), dtype=np.int64)
        out[0] = rz_cat
        out[1] = pos_cat
        out[2] = spd_cat
        out[3] = lick_cat
        out[4] = zone_code[zlab]
        out[5] = isreward[i]

        neural.append(np.ascontiguousarray(ev_trial, dtype=np.float32))
        inputs.append(inp)
        outputs.append(out)
        kept_trials.append(i)

    info = dict(
        file=os.path.basename(path), subject=raw['subject'], scene=raw['scene'],
        date=raw['date'], identifier=raw['identifier'],
        session_start_time=raw['session_start_time'],
        brain_region=raw['brain_region'], n_planes=raw['n_planes'], fs=raw['fs'],
        n_roi_total=int(raw['iscell'].size), n_iscell=int(raw['iscell'].sum()),
        n_interneurons=int(is_int.sum()), n_neurons=n_neurons,
        n_trials_total=ntrials_all, n_trials_kept=len(neural),
        n_lick_error=int(lick_error.sum()), n_trunc=int(raw['n_trunc']),
        exp_day=exp_day, keep_teleports=bool(keep_teleports),
        frac_rewarded=float(isreward.mean()),
        is_switch='_to_' in raw['scene'],
        kept_trials=kept_trials,
        t_load=t_load, t_dff=t_dff, t_total=time.time() - t_start,
    )
    if verbose:
        print(f"  {info['file']}: {info['n_neurons']} neurons "
              f"({info['n_iscell']} iscell, {info['n_interneurons']} interneurons), "
              f"{info['n_trials_kept']}/{info['n_trials_total']} trials, "
              f"load {t_load:.1f}s dff {t_dff:.1f}s total {info['t_total']:.1f}s", flush=True)

    if show_processing:
        make_processing_plots(path, raw, dff, events, keep_cells, starts, stops,
                              labels, isreward, neural, inputs, outputs, kept_trials, info)
    return neural, inputs, outputs, info


# ----------------------------------------------------------------------------------
# processing visualisation
# ----------------------------------------------------------------------------------
def make_processing_plots(path, raw, dff, events, keep_cells, starts, stops,
                          labels, isreward, neural, inputs, outputs, kept_trials, info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = os.path.basename(path).replace('.nwb', '')
    fig, axs = plt.subplots(8, 1, figsize=(16, 22), sharex=False)

    # show the first ~8 trials
    ntr = min(8, len(starts))
    i0, i1 = starts[0] - 1, stops[ntr - 1] - 1
    tt = raw['time'][i0:i1]
    tstarts = raw['time'][starts[:ntr] - 1]

    # 1. raw F / Fneu of one example cell
    cell_ids = np.where(raw['iscell'])[0]
    c = 0
    ax = axs[0]
    ax.plot(tt, raw['F'][cell_ids[c], i0:i1], label='F (raw)', lw=0.7)
    ax.plot(tt, raw['Fneu'][cell_ids[c], i0:i1], label='Fneu', lw=0.7)
    for x in tstarts:
        ax.axvline(x, color='k', ls='--', lw=0.5)
    ax.set_title(f'{sid}: raw fluorescence, example iscell ROI {cell_ids[c]} '
                 f'(dashed = trial starts, gaps = excluded ITI/teleport)')
    ax.legend(loc='upper right', fontsize=8)

    # 2. dF/F and deconvolved events for same cell
    ax = axs[1]
    ax.plot(tt, dff[c, i0:i1], label='dF/F (per-trial maximin baseline, smoothed)', lw=0.8)
    ax.plot(tt, events[c, i0:i1] if keep_cells[c] else np.full(i1 - i0, np.nan),
            label='deconvolved events (OASIS)', lw=0.8)
    for x in tstarts:
        ax.axvline(x, color='k', ls='--', lw=0.5)
    ax.set_title('dF/F and deconvolved activity (NaN outside trials -> excluded)')
    ax.legend(loc='upper right', fontsize=8)

    # 3. population raster of converted neural data for the first trials
    ax = axs[2]
    cat = np.concatenate(neural[:ntr], axis=1)
    vmax = np.percentile(cat, 99.5) if cat.size else 1
    ax.imshow(cat, aspect='auto', vmin=0, vmax=vmax, cmap='magma',
              interpolation='nearest')
    bnds = np.cumsum([0] + [n.shape[1] for n in neural[:ntr]])
    for b in bnds:
        ax.axvline(b, color='w', ls='--', lw=0.5)
    ax.set_title('converted neural matrix (neurons x time), first kept trials, '
                 'white lines = trial boundaries')

    # 4. behaviour: position with reward zone
    ax = axs[3]
    ax.plot(tt, raw['pos'][i0:i1], 'k', lw=0.8, label='position (cm)')
    for i in range(ntr):
        z0, z1 = ZONE_DICT[labels[i]]
        ax.fill_between(raw['time'][starts[i] - 1:stops[i] - 1], z0, z1,
                        color='g', alpha=0.3)
    rf = np.searchsorted(raw['time'], raw['reward_times'])
    rf = rf[(rf >= i0) & (rf < i1)]
    ax.plot(raw['time'][rf], raw['pos'][rf], 'rv', ms=6, label='reward delivery')
    for x in tstarts:
        ax.axvline(x, color='k', ls='--', lw=0.5)
    ax.set_title('position, reward zone (green) and reward deliveries (red) - '
                 'rewards must fall inside the zone')
    ax.legend(loc='upper right', fontsize=8)

    # 5. inputs
    ax = axs[4]
    icat = np.concatenate(inputs[:ntr], axis=1)
    for k, nm in enumerate(INPUT_NAMES):
        ax.plot(icat[k], label=nm, lw=0.8)
    for b in bnds:
        ax.axvline(b, color='k', ls='--', lw=0.5)
    ax.set_title('decoder inputs (concatenated kept trials)')
    ax.legend(loc='upper right', fontsize=8)

    # 6. reward-zone distance: continuous vs discretised
    ax = axs[5]
    ocat = np.concatenate(outputs[:ntr], axis=1)
    pcat = np.concatenate([raw['pos'][starts[i] - 1:stops[i] - 1] for i in kept_trials[:ntr]])
    dcont = []
    for i in kept_trials[:ntr]:
        z0, z1 = ZONE_DICT[labels[i]]
        p = raw['pos'][starts[i] - 1:stops[i] - 1]
        _, d = discretize_reward_distance(p, z0, z1)
        dcont.append(d)
    dcont = np.concatenate(dcont)
    ax.plot(dcont, 'k', lw=0.8, label='distance to reward zone (cm)')
    ax2 = ax.twinx()
    ax2.step(np.arange(ocat.shape[1]), ocat[0], color='r', lw=0.8,
             label='discretised category')
    for thr in [-50, -10, 0, 10, 50]:
        ax.axhline(thr, color='b', ls=':', lw=0.5)
    ax.set_title('output 0: reward-zone distance, continuous (black) vs category (red), '
                 'blue = bin edges')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # 7. position + speed discretisation
    ax = axs[6]
    ax.plot(pcat, 'k', lw=0.8, label='position (cm)')
    ax2 = ax.twinx()
    ax2.step(np.arange(ocat.shape[1]), ocat[1], color='r', lw=0.8, label='position bin')
    for thr in POS_EDGES:
        ax.axhline(thr, color='b', ls=':', lw=0.5)
    ax.set_title('output 1: position, continuous (black) vs category (red)')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    ax = axs[7]
    scat = np.concatenate([raw['speed'][starts[i] - 1:stops[i] - 1] for i in kept_trials[:ntr]])
    lcat = np.concatenate([raw['lick'][starts[i] - 1:stops[i] - 1] for i in kept_trials[:ntr]])
    ax.plot(scat, 'k', lw=0.8, label='speed (cm/s)')
    ax2 = ax.twinx()
    ax2.step(np.arange(ocat.shape[1]), ocat[2], color='r', lw=0.8, label='speed bin')
    ax2.step(np.arange(ocat.shape[1]), ocat[3] - 3, color='g', lw=0.8, label='lick (shifted)')
    ax.plot(lcat, color='c', lw=0.6, alpha=0.7, label='raw lick count')
    for thr in SPEED_EDGES:
        ax.axhline(thr, color='b', ls=':', lw=0.5)
    ax.set_title('outputs 2,3: speed and lick, continuous (black/cyan) vs category (red/green)')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fname = f'/app/processing_{sid}.png'
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print(f'  saved {fname}', flush=True)


# ----------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------
def _worker(path):
    try:
        return convert_session(path, show_processing=False, verbose=True)
    except Exception:
        traceback.print_exc()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', help='output pickle path')
    ap.add_argument('--full', action='store_true', default=True, help='process all sessions')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save processing plots for up to 2 sessions')
    ap.add_argument('--nworkers', type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', 'sub-*.nwb')))
    print(f'found {len(files)} NWB files')
    if args.sample:
        # one single-plane and one 2-plane session so both code paths are exercised
        files = [f for f in files if 'sub-m11_ses-03' in f or 'sub-m17_ses-03' in f]
        print(f'--sample: processing {len(files)} sessions: {[os.path.basename(f) for f in files]}')

    t0 = time.time()
    results = []
    if args.show_processing:
        plot_files = files[:2]
        for f in plot_files:
            print(f'processing (with plots) {os.path.basename(f)}')
            results.append(convert_session(f, show_processing=True))
        remaining = files[2:]
    else:
        remaining = files

    if remaining:
        nw = min(args.nworkers, len(remaining))
        print(f'processing {len(remaining)} sessions with {nw} workers')
        # 'spawn' avoids 'fork() called from a process already using GNU OpenMP' aborts
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
            for res in ex.map(_worker, remaining):
                results.append(res)
    print(f'all sessions processed in {time.time() - t0:.1f}s')

    # ---- assemble the output dictionary ----
    neural, inputs, outputs, infos = [], [], [], []
    for (n, i, o, info) in results:
        if len(n) < 2:
            print(f"  SKIPPING {info['file']}: only {len(n)} usable trials")
            continue
        neural.append(n)
        inputs.append(i)
        outputs.append(o)
        infos.append(info)

    subjects = sorted({inf['subject'] for inf in infos}, key=lambda s: int(s[1:]))
    subject_idx = np.array([subjects.index(inf['subject']) for inf in infos], dtype=np.int64)
    brain_regions = ['CA1']
    brain_region_idx = [np.zeros(inf['n_neurons'], dtype=np.int64) for inf in infos]

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice navigate a 450 cm virtual linear track (ENV1 or ENV2) with a hidden '
                '50 cm reward zone at one of three locations (A 80-130, B 200-250, C 320-370 cm). '
                'Sucrose reward is delivered operantly for licking in the zone and is randomly omitted '
                'on ~15% of trials. On switch sessions the zone moves to a new location after 30 trials. '
                'Decoded outputs: distance to the reward zone, absolute track position, running speed, '
                'licking, reward zone identity and trial reward outcome. '
                'Neural data are two-photon calcium imaging of dorsal CA1 pyramidal cells (GCaMP7f), '
                'expressed as deconvolved activity (OASIS on per-trial maximin dF/F).'),
            'time_bin_size': float(1000.0 / infos[0]['fs']),   # ms
            'temporal_alignment_event': 'trial start (teleport into the track at position 0 cm)',
            'off_start': 0.0,
            'off_end': None,
            'neural_signal': ('deconvolved calcium activity (suite2p OASIS, tau=0.7 s) applied to '
                              'per-trial maximin dF/F, as in Sosa et al. preprocessing.dff()'),
            'sampling_rate_hz': float(infos[0]['fs']),
            'trial_window': 'frames [trial_start-1, teleport-1); inter-trial/teleport period excluded',
            'neuron_curation': ('suite2p iscell==1 (manual curation) minus putative interneurons '
                                '(Pearson r between dF/F and running speed > 0.5)'),
            'trial_curation': ('trials with lick-sensor errors excluded (>30% of frames with cumulative '
                               'lick count > 2, as defined in the paper methods)'),
            'reward_zones_cm': {k: list(v) for k, v in ZONE_DICT.items()},
            'track_length_cm': TRACK_LENGTH,
            'session_info': infos,
            'source': ('DANDI:001361 - Sosa, Plitt & Giocomo (2025) Nature Neuroscience, '
                       '"A flexible hippocampal population code for experience relative to reward"'),
        },
    }

    # ---- summary / sanity checks ----
    ntr = sum(len(n) for n in neural)
    nneur = sum(inf['n_neurons'] for inf in infos)
    print('\n==== conversion summary ====')
    print(f'sessions: {len(neural)}   subjects: {len(subjects)} {subjects}')
    print(f'trials kept: {ntr} (of {sum(inf["n_trials_total"] for inf in infos)})')
    print(f'trials/session: mean {ntr / len(neural):.1f} '
          f'sd {np.std([len(n) for n in neural]):.1f} '
          f'min {min(len(n) for n in neural)} max {max(len(n) for n in neural)}')
    print(f'neurons total: {nneur}; per session mean {nneur / len(neural):.1f} '
          f'min {min(inf["n_neurons"] for inf in infos)} max {max(inf["n_neurons"] for inf in infos)}')
    n_int = sum(inf['n_interneurons'] for inf in infos)
    n_isc = sum(inf['n_iscell'] for inf in infos)
    print(f'iscell ROIs: {n_isc}; putative interneurons excluded: {n_int} '
          f'({100.0 * n_int / n_isc:.2f}%)')
    print(f'lick-error trials excluded: {sum(inf["n_lick_error"] for inf in infos)}')
    print(f'switch sessions: {sum(inf["is_switch"] for inf in infos)}')
    frac_rew = np.mean(np.concatenate([np.concatenate([o[5, :1] for o in outs])
                                       for outs in outputs]))
    print(f'fraction of kept trials rewarded: {frac_rew:.4f}')
    tot_time = sum(n.shape[1] for outs in neural for n in outs)
    print(f'total time bins: {tot_time} ({tot_time / infos[0]["fs"] / 3600:.2f} h)')

    # output distributions
    print('\noutput value distributions (time-weighted):')
    for k, nm in enumerate(OUTPUT_NAMES):
        counts = np.zeros(len(OUTPUT_VALUES[k]), dtype=np.int64)
        for outs in outputs:
            for o in outs:
                c = np.bincount(o[k], minlength=len(OUTPUT_VALUES[k]))
                counts[:len(c)] += c
        frac = counts / counts.sum()
        print(f'  {nm}: ' + ', '.join(f'{v}={f:.3f}' for v, f in zip(OUTPUT_VALUES[k], frac)))

    print('\ninput ranges:')
    for k, nm in enumerate(INPUT_NAMES):
        lo = min(float(i[k].min()) for inps in inputs for i in inps)
        hi = max(float(i[k].max()) for inps in inputs for i in inps)
        print(f'  {nm}: [{lo:.3f}, {hi:.3f}]')

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nwrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
