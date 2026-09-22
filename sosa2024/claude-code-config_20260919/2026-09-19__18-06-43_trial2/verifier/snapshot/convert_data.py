#!/usr/bin/env python3
"""
Convert the Sosa, Plitt & Giocomo (2025) CA1 2P/VR dataset (DANDI:001361, NWB files in
/app/data) into the decoder-compatible pickle format.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Options
-------
    --full             process all 152 sessions (default)
    --sample           process only 2 sessions (for testing)
    --show-processing  save per-step diagnostic plots for up to 2 sessions as
                       processing_<session_id>.png
    --workers N        number of worker processes (default: auto)

Processing follows the reference repository `Sosa_et_al_2024`
(`src/reward_relative/{preprocessing,behavior,rewardAnalysis,spatial,glmUtils}.py`) and the
paper Methods. See CONVERSION_NOTES.md for the full mapping and for every place where this
script deliberately differs from the reference.
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from collections import OrderedDict

import h5py
import numpy as np
import scipy as sp
import scipy.ndimage

# ---------------------------------------------------------------------------------------
# Constants taken from the reference code / paper Methods
# ---------------------------------------------------------------------------------------

DATA_ROOT = '/app/data'

# behavior.reward_zone_dict: scene "LocationA" -> zone 'X', "LocationB" -> 'Y', "LocationC" -> 'Z'
REWARD_ZONE_CM = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
REWARD_ZONE_LABELS = ['A', 'B', 'C']
CHANGE_TRIAL = 30            # behavior.get_reward_zones default; "Each switch occurred after 30 trials"
TRACK_LENGTH_CM = 450.0

NEU_COEF = 0.7               # pp.dff / make_multi_anim_sess dff_method['neu_coef']
TAU = 0.7                    # pp.dff default (suite2p ops['tau'])
BASELINE_SMOOTH_SIGMA = 15   # pp.dff maximin: nansmooth(f, [0, 15])
BASELINE_FILTER_WIN = 300    # pp.dff maximin: min/max filter over 300 samples (~20 s)
DFF_SMOOTH_SIGMA = 2         # paper: "smoothed with a two-sample (~0.129 s) s.d. Gaussian kernel"
INTERNEURON_R_THRESH = 0.5   # Methods: "Pearson correlation of >0.5 between dF/F and running speed"
LICK_ERROR_FRAC_THRESH = 0.3  # Methods: ">30% of the ... samples in the trial containing a cumulative lick count >2"
LICK_ERROR_COUNT_THRESH = 2

# teleport_metadata.teleport_sessions: animal -> experiment days on which the laser was NOT
# blanked during the teleport period, i.e. pp.dff(keep_teleports=True).
KEEP_TELEPORT_DAYS = {
    'm11': [1, 7, 8, 14, 15],
    'm12': [1, 7, 8, 14, 15],
    'm13': [1, 7, 8, 14, 15],
    'm14': [1, 7, 8, 14, 15],
    'm15': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm17': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm18': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm19': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
}

INPUT_NAMES = ['time_from_trial_start_s', 'environment', 'trial_number', 'previous_trial_outcome']
OUTPUT_NAMES = ['distance_to_reward_zone', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm (in zone)',
     '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
    ['omitted', 'rewarded'],
]


# ---------------------------------------------------------------------------------------
# Helpers copied / adapted from the reference repository
# ---------------------------------------------------------------------------------------

def nansmooth(a, sig, axis=-1):
    """Gaussian smoothing of an array containing NaNs without propagating them.

    Verbatim port of `reward_relative.utilities.nansmooth` (same implementation as
    `TwoPUtils.utilities.nansmooth`, which `preprocessing.dff` imports).
    """
    nan_inds = np.isnan(a)
    a_nanless = np.copy(a)
    a_nanless[nan_inds] = 0
    one = np.ones(a.shape, dtype=a.dtype)
    one[nan_inds] = 0.001
    a_nanless = sp.ndimage.gaussian_filter1d(a_nanless, sig, axis=axis)
    one = sp.ndimage.gaussian_filter1d(one, sig, axis=axis)
    return a_nanless / one


def zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    """Reward-zone label ('A'/'B'/'C') for each trial, from the VR scene name.

    Port of `behavior.get_reward_zones` for the scenes present in this dataset:
    'Env1_LocationB', 'Env1_LocationB_to_A', 'Env1_B_to_Env2_C', ...
    """
    for first in REWARD_ZONE_LABELS:
        if f'{first}_to' in scene:
            second = scene[-1]
            if second in REWARD_ZONE_LABELS and second != first:
                labels = np.array([first] * min(change_trial, n_trials)
                                  + [second] * max(0, n_trials - change_trial))
                return labels
    for zone in REWARD_ZONE_LABELS:
        if scene.endswith('Location' + zone):
            return np.array([zone] * n_trials)
    raise NotImplementedError(f'Reward zone not defined for scene {scene!r}')


def dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports,
                   fs, neu_coef=NEU_COEF, tau=TAU):
    """deltaF/F and deconvolved 'events' for one session.

    Faithful port of `reward_relative.preprocessing.dff` with
    ``neuropil_method='subtract'``, ``baseline_method='maximin'``,
    ``subtract_baseline=True``, ``deconvolve=True``.

    The one intentional difference is the trial slicing convention: the reference slices
    ``[start-1:stop-1]``; here we slice ``[start:stop]`` so that the kept samples are exactly
    the on-track frames of the lap (the frame at the teleport index carries a corrupted,
    interpolated position and is excluded by both conventions). See CONVERSION_NOTES Step 4.

    Args:
        F: (n_cells, n_frames) raw suite2p fluorescence.
        Fneu: (n_cells, n_frames) neuropil fluorescence.
        trial_starts, teleports: (n_trials,) frame indices.
        keep_teleports: if True, baseline windows span trial + preceding ITI.
        fs: imaging rate per plane (Hz), for OASIS.

    Returns:
        (dff, events) both (n_cells, n_frames) float32, NaN outside the kept segments.
    """
    from suite2p.extraction import dcnv

    if keep_teleports:
        # Keep everything from the start of imaging except the teleport sample itself,
        # whose interpolated position is unreliable (cf. pp.dff docstring).
        start_inds = [int(trial_starts[0])] + (np.asarray(teleports[:-1]) + 1).tolist()
        stop_inds = np.asarray(teleports).tolist()
    else:
        start_inds = np.asarray(trial_starts).tolist()
        stop_inds = np.asarray(teleports).tolist()

    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    f_neu_ = np.full(Fneu.shape, np.nan, dtype=np.float32)
    for start, stop in zip(start_inds, stop_inds):
        f_[:, start:stop] = F[:, start:stop]
        f_neu_[:, start:stop] = Fneu[:, start:stop]

    nanmask = ~np.isnan(f_[0, :])

    # neuropil correction
    f_ -= neu_coef * f_neu_

    # maximin baseline, computed independently within each segment
    flow = np.full(f_.shape, np.nan, dtype=np.float32)
    for start, stop in zip(start_inds, stop_inds):
        seg = f_[:, start:stop]
        # add back the per-trial neuropil mean so dF/F is close to true deltaF/F
        seg += neu_coef * np.nanmean(f_neu_[:, start:stop], axis=1, keepdims=True)
        f_[:, start:stop] = seg
        base = nansmooth(seg, BASELINE_SMOOTH_SIGMA, axis=-1)
        base = sp.ndimage.minimum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
        base = sp.ndimage.maximum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
        flow[:, start:stop] = base

    dff = np.full(f_.shape, np.nan, dtype=np.float32)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    events = np.full(f_.shape, np.nan, dtype=np.float32)
    for start, stop in zip(start_inds, stop_inds):
        smoothed = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=-1)
        dff[:, start:stop] = smoothed
        events[:, start:stop] = dcnv.oasis(np.ascontiguousarray(smoothed), 2000, tau, fs)

    return dff, events


def digitize_distance_to_reward(d):
    """Discretise signed distance (cm) to the nearest point of the reward zone.

    0: < -50 | 1: [-50,-10) | 2: [-10,0) | 3: == 0 (inside the zone)
    4: (0,10] | 5: (10,50] | 6: > 50
    """
    out = np.empty(d.shape, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out


def digitize_position(pos):
    """5 equal 90 cm bins over the 450 cm track."""
    return np.clip((pos // 90.0), 0, 4).astype(np.int64)


def digitize_speed(speed):
    """<2, 2-10, 10-20, 20-40, >40 cm/s."""
    return np.digitize(speed, [2.0, 10.0, 20.0, 40.0]).astype(np.int64)


# ---------------------------------------------------------------------------------------
# Per-session conversion
# ---------------------------------------------------------------------------------------

def load_behavior(f):
    """Read every behaviour stream (already on the imaging-frame grid) from an open NWB file."""
    b = f['processing/behavior/BehavioralTimeSeries']
    beh = {
        'time': b['position/timestamps'][:],
        'position': b['position/data'][:],
        'speed': b['speed/data'][:],
        'lick': b['lick/data'][:],
        'environment': b['environment/data'][:],
        'reward_zone': b['reward_zone/data'][:],
        'trial_number': b['trial number/data'][:],
        'trial_start': b['trial_start/data'][:],
        'teleport': b['teleport/data'][:],
        'scanning': b['scanning/data'][:],
        'reward_times': b['Reward/timestamps'][:],
    }
    return beh


def load_fluorescence(f):
    """Concatenate planes and keep only manually curated cells (`iscell==1`).

    Mirrors TwoPUtils' `load_suite2p_data`, which keeps `iscell[:, 0].astype(bool)`.
    Planes are pooled, as the paper does for all analyses except Extended Data Fig. 7.
    """
    seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:, 0].astype(bool)
    plane_idx = seg['planeIdx'][:]

    Fs, Fneus, keep = [], [], []
    offset = 0
    for plane in sorted(f['processing/ophys/Fluorescence'].keys()):
        Fp = f[f'processing/ophys/Fluorescence/{plane}/data']
        Np = f[f'processing/ophys/Neuropil/{plane}/data']
        n_roi = Fp.shape[1]
        sel = iscell[offset:offset + n_roi]
        # read then subset: column fancy-indexing through hdf5 is far slower
        Fs.append(np.asarray(Fp[:, :], dtype=np.float32)[:, sel].T)
        Fneus.append(np.asarray(Np[:, :], dtype=np.float32)[:, sel].T)
        keep.append(plane_idx[offset:offset + n_roi][sel])
        offset += n_roi
    assert offset == len(iscell), 'plane ROI counts do not cover PlaneSegmentation'
    return np.concatenate(Fs, axis=0), np.concatenate(Fneus, axis=0), np.concatenate(keep)


def process_session(path, show_processing=False):
    """Convert one NWB session. Returns a dict (or raises)."""
    t0 = time.time()
    timing = {}
    with h5py.File(path, 'r') as f:
        subject = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        exp_day = int(session_id)
        identifier = f['identifier'][()].decode()
        scene = identifier.split('/')[-1]
        imaging_rate = float(f['general/optophysiology/ImagingPlane/imaging_rate'][()])
        region = f['general/optophysiology/ImagingPlane/location'][()].decode()

        beh = load_behavior(f)
        t1 = time.time(); timing['read_behavior'] = t1 - t0
        F, Fneu, plane_of_cell = load_fluorescence(f)
        t2 = time.time(); timing['read_fluorescence'] = t2 - t1

    n_frames = len(beh['time'])
    assert F.shape[1] == n_frames, f'{path}: F has {F.shape[1]} frames, behavior has {n_frames}'
    dt = float(np.median(np.diff(beh['time'])))
    fs = 1.0 / dt

    # ---- trial boundaries -------------------------------------------------------------
    trial_starts = np.where(beh['trial_start'] > 0)[0]
    teleports = np.where(beh['teleport'] > 0)[0]
    # robustness: drop an unpaired leading teleport / trailing trial start
    if len(teleports) and len(trial_starts) and teleports[0] < trial_starts[0]:
        teleports = teleports[1:]
    if len(trial_starts) > len(teleports):
        trial_starts = trial_starts[:len(teleports)]
    if len(teleports) > len(trial_starts):
        teleports = teleports[:len(trial_starts)]
    assert np.all(teleports > trial_starts), f'{path}: teleport before trial start'
    n_trials = len(trial_starts)

    # ---- per-trial task variables (behavior.get_trial_types / get_reward_zones) --------
    reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
    reward_frames = np.clip(reward_frames, 0, n_frames - 1)

    zone_labels = zones_from_scene(scene, n_trials)
    zone_starts = np.array([REWARD_ZONE_CM[z][0] for z in zone_labels])
    zone_stops = np.array([REWARD_ZONE_CM[z][1] for z in zone_labels])

    is_reward = np.zeros(n_trials, dtype=np.int64)
    environment = np.zeros(n_trials, dtype=np.int64)
    lick_error = np.zeros(n_trials, dtype=bool)
    for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
        # behavior.get_trial_types: reward delivered AND the reward zone was active
        any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
        any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
        is_reward[i] = int(bool(any_reward and any_rzone))
        env_vals = np.unique(beh['environment'][st:sp_])
        env_vals = env_vals[env_vals >= 0]
        assert len(env_vals) == 1, f'{path}: trial {i} has environment values {env_vals}'
        environment[i] = int(env_vals[0])
        # behavior.correct_lick_sensor_error, threshold from Methods (>30%)
        seg = beh['lick'][st:sp_]
        lick_error[i] = (np.sum(seg > LICK_ERROR_COUNT_THRESH) / len(seg)) > LICK_ERROR_FRAC_THRESH

    prev_outcome = np.empty(n_trials, dtype=np.int64)
    prev_outcome[1:] = is_reward[:-1]
    prev_outcome[0] = 1  # see CONVERSION_NOTES Step 5, decision 8

    # ---- dF/F and deconvolved events ---------------------------------------------------
    keep_teleports = exp_day in KEEP_TELEPORT_DAYS.get(subject, [])
    t3 = time.time()
    dff, events = dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports, fs)
    timing['dff_events'] = time.time() - t3

    # ---- neuron curation: putative interneurons (spatial.is_putative_interneuron) -------
    valid = ~np.isnan(dff[0, :])
    speed_valid = beh['speed'][valid]
    dv = dff[:, valid]
    dv = dv - dv.mean(axis=1, keepdims=True)
    sv = speed_valid - speed_valid.mean()
    denom = np.sqrt((dv ** 2).sum(axis=1) * (sv ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_corr = (dv @ sv) / denom
    is_interneuron = speed_corr > INTERNEURON_R_THRESH

    finite_cells = np.all(np.isfinite(events[:, valid]), axis=1) & np.isfinite(speed_corr)
    keep_cells = (~is_interneuron) & finite_cells
    n_dropped_nonfinite = int(np.sum(~finite_cells))
    events = events[keep_cells]
    dff_kept = dff[keep_cells]
    plane_of_cell = plane_of_cell[keep_cells]

    # ---- build per-trial neural / input / output ---------------------------------------
    neural, inputs, outputs = [], [], []
    kept_trials = []
    for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
        if lick_error[i]:
            continue
        pos = beh['position'][st:sp_]
        speed = beh['speed'][st:sp_]
        lick = beh['lick'][st:sp_]
        tt = beh['time'][st:sp_] - beh['time'][st]
        T = sp_ - st

        act = events[:, st:sp_]
        if not np.all(np.isfinite(act)):
            # should never happen; guard so the decoder never sees NaN
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

        # distance to the nearest point of the reward zone (0 while inside it)
        d = np.zeros(T)
        before = pos < zone_starts[i]
        after = pos > zone_stops[i]
        d[before] = pos[before] - zone_starts[i]
        d[after] = pos[after] - zone_stops[i]

        inp = np.empty((len(INPUT_NAMES), T), dtype=np.float32)
        inp[0] = tt
        inp[1] = environment[i]
        inp[2] = i
        inp[3] = prev_outcome[i]

        out = np.empty((len(OUTPUT_NAMES), T), dtype=np.int64)
        out[0] = digitize_distance_to_reward(d)
        out[1] = digitize_position(pos)
        out[2] = digitize_speed(speed)
        out[3] = (lick > 0).astype(np.int64)
        out[4] = REWARD_ZONE_LABELS.index(zone_labels[i])
        out[5] = is_reward[i]

        neural.append(np.ascontiguousarray(act, dtype=np.float32))
        inputs.append(inp)
        outputs.append(out)
        kept_trials.append(i)

    result = {
        'session_key': f'{subject}_ses-{session_id}',
        'subject': subject,
        'exp_day': exp_day,
        'scene': scene,
        'identifier': identifier,
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'brain_region_idx': np.zeros(events.shape[0], dtype=np.int64),
        'n_neurons': int(events.shape[0]),
        'n_iscell': int(F.shape[0]),
        'n_interneurons': int(np.sum(is_interneuron)),
        'n_nonfinite_cells': n_dropped_nonfinite,
        'n_trials_raw': n_trials,
        'n_trials_kept': len(neural),
        'n_lick_error_trials': int(np.sum(lick_error)),
        'kept_trials': kept_trials,
        'is_reward': is_reward,
        'zone_labels': zone_labels.tolist(),
        'environment': environment,
        'dt': dt,
        'imaging_rate': imaging_rate,
        'n_planes': int(len(np.unique(plane_of_cell))) if len(plane_of_cell) else 1,
        'region': region,
        'keep_teleports': bool(keep_teleports),
        'timing': timing,
        'total_time': time.time() - t0,
    }

    if show_processing:
        plot_processing(result, beh, F, Fneu, dff_kept, events,
                        trial_starts, teleports, zone_starts, zone_stops, lick_error)

    return result


# ---------------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------------

def plot_processing(res, beh, F, Fneu, dff, events, trial_starts, teleports,
                    zone_starts, zone_stops, lick_error):
    """Visualise every processing step for a handful of trials."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_i, t_j = 5, 10                      # window of trials to display
    t_j = min(t_j, len(trial_starts))
    i0, i1 = trial_starts[t_i], teleports[t_j - 1]
    sl = slice(i0, i1)
    tt = beh['time'][sl]

    fig, ax = plt.subplots(8, 1, figsize=(18, 24), sharex=True)

    cells = np.arange(min(4, dff.shape[0]))
    for c in cells:
        ax[0].plot(tt, F[c, sl], lw=0.7, label=f'F cell {c}')
        ax[0].plot(tt, Fneu[c, sl], lw=0.5, ls=':', alpha=0.6)
    ax[0].set_ylabel('raw F (solid)\nFneu (dotted)')
    ax[0].legend(fontsize=7, ncol=4)
    ax[0].set_title(f"{res['session_key']}  {res['scene']}  "
                    f"(keep_teleports={res['keep_teleports']}, trials {t_i}-{t_j-1})")

    for c in cells:
        ax[1].plot(tt, dff[c, sl], lw=0.8)
    ax[1].set_ylabel('dF/F\n(neuropil-sub., maximin,\nsigma=2 smoothed)')

    for c in cells:
        ax[2].plot(tt, events[c, sl], lw=0.8)
    ax[2].set_ylabel('deconvolved events\n(OASIS)')

    im = ax[3].imshow(events[:min(150, events.shape[0]), sl], aspect='auto',
                      extent=[tt[0], tt[-1], min(150, events.shape[0]), 0],
                      vmin=0, vmax=np.nanpercentile(events[:, sl], 99), cmap='magma')
    ax[3].set_ylabel('events\n(first 150 cells)')

    pos = beh['position'][sl]
    ax[4].plot(tt, pos, 'k', lw=1, label='position')
    for k in range(t_i, t_j):
        s, e = trial_starts[k], teleports[k]
        ax[4].plot([beh['time'][s], beh['time'][e - 1]],
                   [zone_starts[k], zone_starts[k]], 'g-', lw=2)
        ax[4].plot([beh['time'][s], beh['time'][e - 1]],
                   [zone_stops[k], zone_stops[k]], 'g-', lw=2)
    ax[4].set_ylabel('position (cm)\ngreen = reward zone')
    ax[4].legend(fontsize=7)

    # discretised outputs, reassembled from the exported trials for this window
    keyed = {k: n for n, k in enumerate(res['kept_trials'])}
    for k in range(t_i, t_j):
        if k not in keyed:
            continue
        n = keyed[k]
        tk = beh['time'][trial_starts[k]:teleports[k]]
        out = res['output'][n]
        ax[5].step(tk, out[0], where='post', color='C0')
        ax[5].step(tk, out[1], where='post', color='C1', alpha=0.7)
        ax[6].step(tk, out[2], where='post', color='C2')
        ax[6].step(tk, out[3] * 4, where='post', color='C3', alpha=0.6)
        inp = res['input'][n]
        ax[7].plot(tk, inp[0], color='C4')
        ax[7].plot(tk, inp[1] * 10, color='C5')
        ax[7].plot(tk, inp[3] * 5, color='C6')
    ax[5].set_ylabel('out0 dist-to-zone (C0)\nout1 position bin (C1)')
    ax[6].set_ylabel('out2 speed bin (C2)\nout3 lick x4 (C3)')
    ax[7].set_ylabel('in0 t-from-start (C4)\nin1 env x10 (C5)\nin3 prev-rew x5 (C6)')
    ax[7].set_xlabel('session time (s)')

    for a in ax:
        for k in range(t_i, t_j):
            a.axvline(beh['time'][trial_starts[k]], color='g', lw=0.8, alpha=0.6)
            a.axvline(beh['time'][teleports[k]], color='r', lw=0.8, alpha=0.6)

    fig.tight_layout()
    fname = f"processing_{res['session_key']}.png"
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print(f'  wrote {fname}', flush=True)

    # --- second figure: verification of the discretisations against the raw streams -----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    allpos, alld, allspeed = [], [], []
    for n, k in enumerate(res['kept_trials']):
        s, e = trial_starts[k], teleports[k]
        p = beh['position'][s:e]
        allpos.append(p)
        d = np.zeros(len(p))
        d[p < zone_starts[k]] = p[p < zone_starts[k]] - zone_starts[k]
        d[p > zone_stops[k]] = p[p > zone_stops[k]] - zone_stops[k]
        alld.append(d)
        allspeed.append(beh['speed'][s:e])
    allpos = np.concatenate(allpos)
    alld = np.concatenate(alld)
    allspeed = np.concatenate(allspeed)
    allout = np.concatenate(res['output'], axis=1)
    ax[0].scatter(alld, allout[0], s=1, alpha=0.2)
    ax[0].set_xlabel('distance to reward zone (cm)'); ax[0].set_ylabel('output 0 bin')
    ax[1].scatter(allpos, allout[1], s=1, alpha=0.2)
    ax[1].set_xlabel('position (cm)'); ax[1].set_ylabel('output 1 bin')
    ax[2].scatter(allspeed, allout[2], s=1, alpha=0.2)
    ax[2].set_xlabel('speed (cm/s)'); ax[2].set_ylabel('output 2 bin')
    ax[2].set_xlim(-10, 80)
    fig.suptitle(f"{res['session_key']}: discretisation check")
    fig.tight_layout()
    fname = f"processing_{res['session_key']}_discretisation.png"
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print(f'  wrote {fname}', flush=True)


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------

def list_sessions():
    files = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        d = os.path.join(DATA_ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.nwb'):
                files.append(os.path.join(d, fn))
    # deterministic order: subject number, then session number
    def key(p):
        base = os.path.basename(p)
        sub = base.split('_')[0].replace('sub-m', '')
        ses = base.split('_')[1].replace('ses-', '')
        return (int(sub), int(ses))
    return sorted(files, key=key)


def _worker(args):
    path, show = args
    warnings.simplefilter('ignore')
    try:
        return process_session(path, show_processing=show)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return {'error': f'{path}: {exc}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outfile', type=str)
    parser.add_argument('--full', action='store_true', help='process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true',
                        help='save per-step plots for up to 2 sessions')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    files = list_sessions()
    if args.sample:
        # one single-plane and one two-plane session, one with keep_teleports=True
        picks = ['sub-m11/sub-m11_ses-03_behavior+ophys.nwb',
                 'sub-m17/sub-m17_ses-01_behavior+ophys.nwb']
        files = [os.path.join(DATA_ROOT, p) for p in picks]
    print(f'Processing {len(files)} sessions', flush=True)

    n_show = 2 if args.show_processing else 0
    jobs = [(p, i < n_show) for i, p in enumerate(files)]

    workers = args.workers
    if workers is None:
        workers = min(10, len(files), max(1, os.cpu_count() // 4))

    t_start = time.time()
    results = []
    if workers <= 1 or len(files) == 1:
        for j in jobs:
            results.append(_worker(j))
            r = results[-1]
            print(f"[{len(results)}/{len(files)}] {r.get('session_key', r.get('error'))} "
                  f"{r.get('total_time', 0):.1f}s", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, maxtasksperchild=2) as pool:
            for r in pool.imap(_worker, jobs):
                results.append(r)
                elapsed = time.time() - t_start
                rate = elapsed / len(results)
                print(f"[{len(results)}/{len(files)}] {r.get('session_key', r.get('error'))} "
                      f"cells={r.get('n_neurons')} trials={r.get('n_trials_kept')} "
                      f"t={r.get('total_time', 0):.1f}s "
                      f"| elapsed {elapsed/60:.1f} min, ETA {(len(files)-len(results))*rate/60:.1f} min",
                      flush=True)

    errors = [r for r in results if 'error' in r]
    if errors:
        for e in errors:
            print('ERROR:', e['error'])
        raise SystemExit('conversion failed')

    results.sort(key=lambda r: (int(r['subject'][1:]), r['exp_day']))

    subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
    sub_index = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([sub_index[r['subject']] for r in results], dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': [r['brain_region_idx'] for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
    }

    dts = np.array([r['dt'] for r in results])
    n_trials = sum(r['n_trials_kept'] for r in results)
    n_neurons = sum(r['n_neurons'] for r in results)

    data['metadata'] = {
        'task_description':
            'Head-fixed mice run laps on a 450 cm virtual linear track (ENV 1 or ENV 2) with a '
            'hidden 50 cm reward zone at one of three locations (A: 80-130, B: 200-250, '
            'C: 320-370 cm). Licking inside the zone delivers sucrose water; reward is randomly '
            'omitted on ~15% of trials. On switch sessions the zone moves to a new location after '
            'trial 30 (0-indexed trial 30), sometimes together with a switch of environment. '
            'Decoded outputs are the signed distance to the reward zone, absolute track position, '
            'running speed, licking, the active reward-zone location, and whether the trial was '
            'rewarded.',
        'time_bin_size': float(np.median(dts) * 1000.0),
        'time_bin_size_range_ms': [float(dts.min() * 1000), float(dts.max() * 1000)],
        'temporal_alignment_event':
            'trial start (entry to the linear track at position 0 cm, NWB "trial_start" event)',
        'off_start': 0.0,
        'off_end': None,
        'trial_window':
            'each trial spans [trial_start, teleport), i.e. the whole on-track lap; laps have '
            'variable duration (median ~12.2 s) so off_end is not a fixed value',
        'neural_signal':
            'deconvolved calcium activity ("events"): raw suite2p F, neuropil-subtracted '
            '(coef 0.7), per-trial maximin dF/F baseline (Gaussian sigma=15 frames, then '
            'min/max filter over 300 frames ~20 s), dF/F = (F-b)/|b|, smoothed with a '
            '2-frame s.d. Gaussian, then OASIS deconvolution (tau=0.7) - exactly '
            'reward_relative.preprocessing.dff(neuropil_method="subtract", '
            'baseline_method="maximin", deconvolve=True)',
        'neuron_curation':
            'suite2p manual curation (iscell==1), then putative interneurons removed '
            '(Pearson r(dF/F, speed) > 0.5)',
        'trial_curation':
            'trials with a lick-sensor error (>30% of frames with cumulative lick count > 2) '
            'are removed, as in the paper Methods',
        'source': 'DANDI:001361 - Sosa, Plitt & Giocomo 2025, Nature Neuroscience',
        'brain_region_detail': 'dorsal hippocampus CA1 (2-photon, GCaMP7f); imaging planes pooled',
        'input_descriptions': [
            'time since trial start (s)',
            'environment: 0 = ENV 1, 1 = ENV 2 (constant within a trial)',
            'trial number within the session (0-indexed, constant within a trial)',
            'outcome of the previous trial: 0 = omitted, 1 = rewarded; the first trial of each '
            'session is set to 1 (undefined; the modal outcome)',
        ],
        'output_descriptions': [
            'signed distance (cm) to the nearest point of the active reward zone, 0 inside it',
            'absolute position on the 450 cm track',
            'running speed (cm/s)',
            'any lick detected in the imaging frame',
            'active reward-zone location (constant within a trial)',
            'trial rewarded or omitted (constant within a trial)',
        ],
        'session_info': [
            {'session_key': r['session_key'], 'subject': r['subject'], 'exp_day': r['exp_day'],
             'scene': r['scene'], 'n_neurons': r['n_neurons'], 'n_iscell': r['n_iscell'],
             'n_interneurons_removed': r['n_interneurons'],
             'n_nonfinite_cells_removed': r['n_nonfinite_cells'],
             'n_trials_raw': r['n_trials_raw'], 'n_trials_kept': r['n_trials_kept'],
             'n_lick_error_trials': r['n_lick_error_trials'],
             'n_rewarded': int(np.sum(r['is_reward'])),
             'keep_teleports': r['keep_teleports'], 'dt_s': r['dt'],
             'n_planes': r['n_planes']}
            for r in results
        ],
        'n_sessions': len(results),
        'n_trials': n_trials,
        'n_neurons_total': n_neurons,
    }

    print('\n--- conversion summary ---')
    print(f'sessions: {len(results)}  subjects: {len(subjects)}  trials: {n_trials}  '
          f'neurons: {n_neurons}')
    print(f"cells: iscell={sum(r['n_iscell'] for r in results)}, "
          f"interneurons removed={sum(r['n_interneurons'] for r in results)} "
          f"({100*sum(r['n_interneurons'] for r in results)/sum(r['n_iscell'] for r in results):.2f}%), "
          f"non-finite removed={sum(r['n_nonfinite_cells'] for r in results)}")
    print(f"lick-error trials removed: {sum(r['n_lick_error_trials'] for r in results)} / "
          f"{sum(r['n_trials_raw'] for r in results)}")
    per_sess_int = np.array([100 * r['n_interneurons'] / max(r['n_iscell'], 1) for r in results])
    print(f'interneuron %% per session: mean {per_sess_int.mean():.2f} '
          f'sd {per_sess_int.std(ddof=1):.2f} (paper: 0.42 +/- 0.85)')
    ntr = np.array([r['n_trials_kept'] for r in results])
    print(f'trials/session: mean {ntr.mean():.1f} sd {ntr.std(ddof=1):.1f} '
          f'min {ntr.min()} max {ntr.max()} (paper: 80.5 +/- 7.4)')
    nn = np.array([r['n_neurons'] for r in results])
    print(f'neurons/session: mean {nn.mean():.0f} min {nn.min()} max {nn.max()} '
          f'(paper: 155-2172)')
    rew = np.array([np.sum(r['is_reward']) for r in results])
    print(f"rewarded fraction: {rew.sum()/sum(r['n_trials_raw'] for r in results):.4f} "
          f'(paper: ~85%, i.e. ~15% omission)')
    print(f"time bin: {data['metadata']['time_bin_size']:.3f} ms")
    print(f'total wall time: {(time.time()-t_start)/60:.1f} min')

    t_w = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t_w:.1f}s')


if __name__ == '__main__':
    main()
