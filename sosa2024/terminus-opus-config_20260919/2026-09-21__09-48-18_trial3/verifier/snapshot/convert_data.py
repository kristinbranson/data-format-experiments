"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal CA1 VR dataset (DANDI 001361)
into the decoder-compatible pickle format.

Processing follows the reference code in /app/code (repo Sosa_et_al_2024):
  * neural signal  : dF/F computed per trial with a maximin baseline from the raw
                     suite2p F / Fneu traces (reward_relative.preprocessing.dff),
                     smoothed, then deconvolved with OASIS -> 'events'
  * cell curation  : suite2p manual curation flag (iscell) + exclusion of putative
                     interneurons (Pearson r > 0.5 between dF/F and running speed,
                     reward_relative.spatial.is_putative_interneuron)
  * trial window   : [trial_start-1, teleport-1) as in preprocessing.dff /
                     glmUtils.get_timeseries_data
  * reward zones   : scene name + switch after 30 trials
                     (reward_relative.behavior.get_reward_zones)
  * trial types    : rewarded if a reward was delivered while the reward zone was
                     active (reward_relative.behavior.get_trial_types)
  * lick curation  : trials with >30% of frames having cumulative lick count > 2 are
                     invalid (reward_relative.behavior.correct_lick_sensor_error and
                     Methods: 81/12,376 trials)

Usage:  python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""
import argparse
import glob
import os
import pickle
import time
import traceback

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from suite2p.extraction import dcnv

# ----------------------------------------------------------------------------------
# Constants taken from the reference code / paper methods
# ----------------------------------------------------------------------------------
# reward_relative.behavior.reward_zone_dict (X/Y/Z are the internal names of A/B/C)
REWARD_ZONE_DICT = {'A': [80.0, 130.0], 'B': [200.0, 250.0], 'C': [320.0, 370.0],
                    'T': [275.0, 325.0]}
CHANGE_TRIAL = 30           # behavior.get_reward_zones(change_trial=30)
NEU_COEF = 0.7              # utilities.default_dff_method['neu_coef']
TAU = 0.7                   # suite2p ops['tau'] in the reference suite2p notebook
BASELINE_SMOOTH_SIGMA = 15  # frames, preprocessing.dff maximin
MAXIMIN_WINDOW = 300        # frames (~20 s), preprocessing.dff maximin
DFF_SMOOTH_SIGMA = 2        # frames (~0.129 s), preprocessing.dff
OASIS_BATCH = 2000          # preprocessing.dff
INTERNEURON_R_THRESH = 0.5  # spatial.is_putative_interneuron(r_thresh=0.5)
# extra numerical-stability rule (not in the reference): drop cells whose per-trial
# maximin baseline falls below this fraction of the cell median raw fluorescence
BASELINE_MIN_FRAC = 0.05
LICK_ERROR_THRESH = 0.30    # Methods: >30% of frames with cumulative lick count > 2
TRACK_LENGTH = 450.0        # cm

# output discretisation edges (np.digitize, right=False unless noted)
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
SPEED_EDGES = [2.0, 10.0, 20.0, 40.0]
RZ_LABELS = ['A', 'B', 'C']

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
# helpers copied / adapted from the reference code
# ----------------------------------------------------------------------------------
def nansmooth(a, sig, axis=-1):
    """reward_relative.utilities.nansmooth: gaussian smoothing that ignores NaNs."""
    nan_inds = np.isnan(a)
    a_nanless = np.where(nan_inds, 0.0, a)
    one = np.ones(a.shape)
    one[nan_inds] = 0.001
    a_nanless = gaussian_filter1d(a_nanless, sig, axis=axis)
    one = gaussian_filter1d(one, sig, axis=axis)
    return a_nanless / one


def scene_reward_zones(scene, n_trials):
    """Per-trial reward zone coordinates and labels from the VR scene name.

    Mirrors reward_relative.behavior.get_reward_zones: a constant zone for
    'Env<n>_Location<X>' scenes, and a switch after CHANGE_TRIAL trials for the
    'Env<n>_Location<X>_to_<Y>' and 'Env<n>_<X>_to_Env<m>_<Y>' switch scenes.
    """
    s = scene
    if '_to_' not in s:
        # e.g. Env1_LocationA / Env2_LocationC
        label = s[-1]
        labels = [label] * n_trials
    else:
        pre, post = s.split('_to_')
        first = pre[-1]              # ...LocationX  or  Env1_X
        second = post[-1]            # Y  or  Env2_Y
        labels = [first] * min(CHANGE_TRIAL, n_trials)
        if n_trials > CHANGE_TRIAL:
            labels += [second] * (n_trials - CHANGE_TRIAL)
    if any(l not in REWARD_ZONE_DICT for l in labels):
        raise ValueError('unrecognised scene %s' % scene)
    coords = np.array([REWARD_ZONE_DICT[l] for l in labels], dtype=float)
    return coords, np.array(labels)


def compute_dff_and_events(F, Fneu, starts, stops, fs):
    """Reference reward_relative.preprocessing.dff with
    neuropil_method='subtract', baseline_method='maximin', subtract_baseline=True,
    keep_teleports=False, deconvolve=True.

    F, Fneu: (n_cells, n_frames) raw suite2p traces
    starts, stops: trial_start / teleport frame indices
    fs: frames per second per plane
    Returns dff, events (both NaN outside of trials).
    """
    f_ = np.full(F.shape, np.nan)
    fneu_ = np.full(F.shape, np.nan)
    for start, stop in zip(starts, stops):
        f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
        fneu_[:, start - 1:stop - 1] = Fneu[:, start - 1:stop - 1]

    nanmask = ~np.isnan(f_[0, :])

    # neuropil correction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan)
    events = np.full(F.shape, np.nan)

    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        # add back the per-trial neuropil mean so dF/F is not divided by small numbers
        f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
        # maximin baseline within the trial
        base = gaussian_filter1d(f_[:, sl], BASELINE_SMOOTH_SIGMA, axis=-1)
        base = minimum_filter1d(base, MAXIMIN_WINDOW, axis=-1)
        base = maximum_filter1d(base, MAXIMIN_WINDOW, axis=-1)
        flow[:, sl] = base

    dff = np.full(F.shape, np.nan)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    # Cells whose maximin baseline collapses towards zero (this happens when the
    # neuropil trace is much larger than the ROI trace, so the neuropil-corrected
    # signal crosses zero) give a numerically meaningless dF/F, because dF/F divides
    # by |baseline|. Flag them so they can be excluded from the converted data.
    with np.errstate(invalid='ignore'):
        min_base = np.nanmin(flow, axis=1)
        scale = np.nanmedian(np.where(np.isnan(flow), np.nan, F), axis=1)
    bad_baseline = ~(min_base > BASELINE_MIN_FRAC * scale)

    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH_SIGMA, axis=1)
        events[:, sl] = dcnv.oasis(
            np.ascontiguousarray(dff[:, sl], dtype=np.float32), OASIS_BATCH, TAU, fs)

    return dff, events, bad_baseline


def reward_zone_distance(pos, rz_start, rz_end):
    """Signed distance (cm) to the nearest point of the reward zone.

    0 anywhere inside the zone, negative before the zone, positive after it.
    """
    d = np.zeros_like(pos)
    before = pos < rz_start
    after = pos > rz_end
    d[before] = pos[before] - rz_start
    d[after] = pos[after] - rz_end
    return d


def digitize_rz_distance(d):
    """7-way discretisation of the reward-zone distance (see task description)."""
    out = np.full(d.shape, 3, dtype=np.int64)   # exactly 0 -> in the zone
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
    return out


# ----------------------------------------------------------------------------------
# per-session conversion
# ----------------------------------------------------------------------------------
def load_session(fn):
    """Read everything needed from one NWB file (h5py, no pynwb needed)."""
    with h5py.File(fn, 'r') as f:
        beh = f['processing/behavior/BehavioralTimeSeries']
        g = lambda k: beh[k]['data'][()].astype(np.float64)
        d = dict(
            file=fn,
            subject=f['general/subject/subject_id'][()].decode(),
            session_id=f['general/session_id'][()].decode(),
            scene=f['identifier'][()].decode().split('/')[-1],
            date=f['identifier'][()].decode().split('/')[-2],
            imaging_rate=float(f['general/optophysiology/ImagingPlane/imaging_rate'][()]),
            location=f['general/optophysiology/ImagingPlane/location'][()].decode(),
            pos=g('position'), speed=g('speed'), lick=g('lick'),
            rzone_flag=g('reward_zone'), env=g('environment'),
            trialnum=g('trial number'), autoreward=g('autoreward'),
            scanning=g('scanning'),
            t=beh['position']['timestamps'][()].astype(np.float64),
            reward_t=beh['Reward']['timestamps'][()].astype(np.float64),
            starts=np.where(g('trial_start') > 0)[0],
            stops=np.where(g('teleport') > 0)[0],
        )
        ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = ps['iscell'][()][:, 0].astype(bool)
        planes = sorted(int(p) for p in np.unique(ps['planeIdx'][()]))
        Fl, Nl = [], []
        for p in planes:
            rois = f['processing/ophys/Fluorescence/plane%d/rois' % p][()]
            keep = iscell[rois]
            Fl.append(f['processing/ophys/Fluorescence/plane%d/data' % p][:, keep].T)
            Nl.append(f['processing/ophys/Neuropil/plane%d/data' % p][:, keep].T)
        d['F'] = np.concatenate(Fl, axis=0).astype(np.float64)
        d['Fneu'] = np.concatenate(Nl, axis=0).astype(np.float64)
        # In 10 sessions (m17, m18) the ophys series contain exactly one frame more
        # than the behavioural series. Both streams start at t = 0 (ophys
        # starting_time = 0 and behaviour timestamps[0] = 0), so the extra imaging
        # frame is at the end: truncate to the common length so that frame i of the
        # neural data corresponds to frame i of the behaviour.
        n_beh = d['pos'].shape[0]
        if d['F'].shape[1] != n_beh:
            print('  NOTE %s: ophys has %d frames, behaviour %d; truncating to %d'
                  % (os.path.basename(fn), d['F'].shape[1], n_beh, min(d['F'].shape[1], n_beh)))
            n = min(d['F'].shape[1], n_beh)
            d['F'] = d['F'][:, :n]
            d['Fneu'] = d['Fneu'][:, :n]
            for k in ['pos', 'speed', 'lick', 'rzone_flag', 'env', 'trialnum',
                      'autoreward', 'scanning', 't']:
                d[k] = d[k][:n]
            d['starts'] = d['starts'][d['starts'] < n - 1]
            d['stops'] = d['stops'][:len(d['starts'])]
            d['starts'] = d['starts'][:len(d['stops'])]
        d['n_planes'] = len(planes)
        d['n_iscell'] = int(iscell.sum())
        d['n_roi_total'] = int(iscell.size)
    return d


def process_session(fn, show=False, plotdir='/app'):
    """Convert one NWB session. Returns (session_dict, stats_dict)."""
    t0 = time.time()
    S = load_session(fn)
    t_load = time.time() - t0

    starts, stops = S['starts'], S['stops']
    n_trials = len(starts)
    assert n_trials == len(stops) and np.all(starts < stops) and starts[0] >= 1

    fs = S['imaging_rate'] / S['n_planes']
    dt = float(np.median(np.diff(S['t'])))

    # ---- neural: dF/F + OASIS events (reference preprocessing.dff) ----
    t1 = time.time()
    dff, events, bad_baseline = compute_dff_and_events(S['F'], S['Fneu'], starts, stops, fs)
    t_dff = time.time() - t1

    # ---- cell curation: putative interneurons (dF/F vs speed r > 0.5) ----
    t2 = time.time()
    valid = ~np.isnan(dff[0, :])
    sp = S['speed'][valid]
    X = dff[:, valid]
    Xc = X - X.mean(axis=1, keepdims=True)
    spc = sp - sp.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=1) * (spc ** 2).sum())
    speed_corr = (Xc @ spc) / np.where(denom == 0, np.nan, denom)
    is_int = np.nan_to_num(speed_corr, nan=0.0) > INTERNEURON_R_THRESH
    keep_cells = (~is_int) & (~bad_baseline)
    t_int = time.time() - t2

    # ---- trial-level task variables ----
    rz_coords, rz_labels = scene_reward_zones(S['scene'], n_trials)
    reward_frames = np.searchsorted(S['t'], S['reward_t'])

    isreward = np.zeros(n_trials, dtype=np.int64)
    env_trial = np.zeros(n_trials, dtype=np.int64)
    lick_error = np.zeros(n_trials, dtype=bool)
    rz_onset_pos = np.full(n_trials, np.nan)
    for i, (s, e) in enumerate(zip(starts, stops)):
        sl = slice(s - 1, e - 1)
        rzflag = S['rzone_flag'][sl] > 0
        got_reward = np.any((reward_frames >= s - 1) & (reward_frames < e - 1))
        # behavior.get_trial_types: reward delivered AND reward zone active
        isreward[i] = int(got_reward and np.any(rzflag))
        env_trial[i] = int(np.round(np.median(S['env'][sl])))
        L = S['lick'][sl]
        lick_error[i] = (np.sum(L > 2) / len(L)) > LICK_ERROR_THRESH
        if np.any(rzflag):
            rz_onset_pos[i] = S['pos'][sl][np.argmax(rzflag)]

    # sanity check: scene-derived zone vs zone entry position measured in the data
    obs = rz_onset_pos[~np.isnan(rz_onset_pos)]
    exp = rz_coords[~np.isnan(rz_onset_pos), 0]
    zone_err = float(np.nanmedian(obs - exp)) if len(obs) else np.nan
    if len(obs) and not (-1.0 <= zone_err <= 15.0):
        print('  WARNING %s: reward-zone mismatch, median(onset-zone_start)=%.1f cm'
              % (os.path.basename(fn), zone_err))

    # ---- build per-trial arrays ----
    neural, inp, out = [], [], []
    kept_trials = []
    for i, (s, e) in enumerate(zip(starts, stops)):
        if i == 0:
            continue                      # previous-trial outcome undefined
        if lick_error[i]:
            continue                      # invalid lick data (reference sets to NaN)
        sl = slice(s - 1, e - 1)
        T = (e - 1) - (s - 1)
        ev = events[np.ix_(keep_cells, np.arange(s - 1, e - 1))]
        if not np.all(np.isfinite(ev)):
            print('  WARNING %s trial %d: non-finite events, trial dropped'
                  % (os.path.basename(fn), i))
            continue

        pos = S['pos'][sl]
        speed = S['speed'][sl]
        lick = S['lick'][sl]
        tt = S['t'][sl] - S['t'][s]       # time from trial start (s)

        x = np.empty((4, T), dtype=np.float32)
        x[0] = tt
        x[1] = env_trial[i]
        x[2] = i                           # lap index within the session
        x[3] = isreward[i - 1]             # previous trial outcome

        d_rz = reward_zone_distance(pos, rz_coords[i, 0], rz_coords[i, 1])
        y = np.empty((6, T), dtype=np.int64)
        y[0] = digitize_rz_distance(d_rz)
        y[1] = np.digitize(pos, POSITION_EDGES)
        y[2] = np.digitize(speed, SPEED_EDGES)
        y[3] = (lick >= 1).astype(np.int64)
        y[4] = RZ_LABELS.index(rz_labels[i])
        y[5] = isreward[i]

        neural.append(ev.astype(np.float32))
        inp.append(x)
        out.append(y)
        kept_trials.append(i)

    sess = dict(
        neural=neural, input=inp, output=out,
        subject=S['subject'], session_id=S['session_id'], scene=S['scene'],
        date=S['date'], location=S['location'],
        n_neurons=int(keep_cells.sum()), kept_trials=np.array(kept_trials),
        dt=dt,
    )
    stats = dict(
        file=os.path.basename(fn), subject=S['subject'], session=S['session_id'],
        scene=S['scene'], n_roi=S['n_roi_total'], n_iscell=S['n_iscell'],
        n_interneuron=int(is_int.sum()), n_bad_baseline=int(bad_baseline.sum()),
        n_neurons=int(keep_cells.sum()),
        n_trials_raw=n_trials, n_trials_kept=len(neural),
        n_lick_error=int(lick_error.sum()), n_rewarded=int(isreward.sum()),
        frac_rewarded=float(isreward.mean()), zone_err=zone_err,
        dff_min=float(np.nanmin(dff[keep_cells])), dff_max=float(np.nanmax(dff[keep_cells])),
        ev_max=float(np.nanmax(events[keep_cells])), dt=dt,
        t_load=t_load, t_dff=t_dff, t_int=t_int, t_total=time.time() - t0,
    )

    if show:
        try:
            make_processing_plots(S, dff, events, keep_cells, rz_coords, rz_labels,
                                 isreward, sess, plotdir)
        except Exception:
            traceback.print_exc()

    return sess, stats


# ----------------------------------------------------------------------------------
# visualisation of every processing step (--show-processing)
# ----------------------------------------------------------------------------------
def make_processing_plots(S, dff, events, keep_cells, rz_coords, rz_labels,
                          isreward, sess, plotdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = '%s_ses-%s' % (S['subject'], S['session_id'])
    starts, stops = S['starts'], S['stops']
    # a window covering trials 1-4 (0-indexed), which are all kept trials
    i0, i1 = 1, min(5, len(starts) - 1)
    a, b = starts[i0] - 1, stops[i1 - 1] - 1
    tt = S['t'][a:b]
    cells = np.where(keep_cells)[0][:5]

    fig, ax = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
    for c in cells:
        ax[0].plot(tt, S['F'][c, a:b], lw=0.7)
    ax[0].set_ylabel('raw F')
    ax[0].set_title('%s (%s): raw fluorescence of 5 example curated cells' % (sid, S['scene']))
    for c in cells:
        ax[1].plot(tt, S['Fneu'][c, a:b], lw=0.7)
    ax[1].set_ylabel('neuropil F')
    for c in cells:
        ax[2].plot(tt, dff[c, a:b], lw=0.7)
    ax[2].set_ylabel('dF/F')
    for c in cells:
        ax[3].plot(tt, events[c, a:b], lw=0.7)
    ax[3].set_ylabel('events (OASIS)')
    ax[4].plot(tt, S['pos'][a:b], 'k', lw=1)
    for i in range(i0, i1):
        s, e = starts[i] - 1, stops[i] - 1
        ax[4].axhspan(rz_coords[i, 0], rz_coords[i, 1], xmin=(s - a) / (b - a),
                      xmax=(e - a) / (b - a), color='g', alpha=0.2)
    ax[4].set_ylabel('position (cm)')
    ax[5].plot(tt, S['speed'][a:b], 'b', lw=1)
    ax[5].axhline(2, color='gray', ls=':')
    ax[5].set_ylabel('speed (cm/s)')
    ax[6].plot(tt, S['lick'][a:b], 'm', lw=1, label='cumulative lick count')
    rf = np.searchsorted(S['t'], S['reward_t'])
    rf = rf[(rf >= a) & (rf < b)]
    ax[6].plot(S['t'][rf], np.ones(len(rf)) * 0.5, 'r*', ms=12, label='reward')
    ax[6].set_ylabel('licks / reward')
    ax[6].legend(loc='upper right', fontsize=7)
    ax[6].set_xlabel('session time (s)')
    for axi in ax:
        for i in range(i0, i1 + 1):
            axi.axvline(S['t'][starts[i] - 1], color='g', lw=0.8)
            axi.axvline(S['t'][stops[i] - 1], color='r', lw=0.8, ls='--')
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, 'processing_%s_timeseries.png' % sid), dpi=110)
    plt.close(fig)

    # ---- discretisation checks + per-trial alignment of converted arrays ----
    k = 0  # first kept trial
    x, y = sess['input'][k], sess['output'][k]
    trial = sess['kept_trials'][k]
    s, e = starts[trial] - 1, stops[trial] - 1
    pos = S['pos'][s:e]
    d_rz = reward_zone_distance(pos, rz_coords[trial, 0], rz_coords[trial, 1])

    fig, ax = plt.subplots(4, 2, figsize=(14, 12))
    ax[0, 0].plot(x[0], pos, 'k')
    ax2 = ax[0, 0].twinx()
    ax2.step(x[0], y[1], 'r', where='post')
    ax[0, 0].axhline(rz_coords[trial, 0], color='g', ls=':')
    ax[0, 0].axhline(rz_coords[trial, 1], color='g', ls=':')
    ax[0, 0].set_xlabel('time from trial start (s)')
    ax[0, 0].set_ylabel('position (cm)')
    ax2.set_ylabel('position bin (red)')
    ax[0, 0].set_title('%s trial %d: position and its discretisation' % (sid, trial))

    ax[0, 1].plot(x[0], d_rz, 'k')
    ax3 = ax[0, 1].twinx()
    ax3.step(x[0], y[0], 'r', where='post')
    ax[0, 1].set_xlabel('time from trial start (s)')
    ax[0, 1].set_ylabel('distance to reward zone (cm)')
    ax3.set_ylabel('distance bin (red)')
    ax[0, 1].set_title('distance to reward zone (0 inside zone %s)' % rz_labels[trial])

    ax[1, 0].plot(x[0], S['speed'][s:e], 'b')
    ax4 = ax[1, 0].twinx()
    ax4.step(x[0], y[2], 'r', where='post')
    ax[1, 0].set_ylabel('speed (cm/s)')
    ax4.set_ylabel('speed bin (red)')
    ax[1, 0].set_xlabel('time from trial start (s)')

    ax[1, 1].plot(x[0], S['lick'][s:e], 'm', label='cumulative count')
    ax[1, 1].step(x[0], y[3], 'r', where='post', label='lick output')
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_xlabel('time from trial start (s)')

    # discretisation scatter over the whole session
    allpos = np.concatenate([S['pos'][starts[i] - 1:stops[i] - 1] for i in sess['kept_trials']])
    allposbin = np.concatenate([o[1] for o in sess['output']])
    ax[2, 0].plot(allpos, allposbin, '.', ms=1)
    ax[2, 0].set_xlabel('position (cm)'); ax[2, 0].set_ylabel('bin')
    allsp = np.concatenate([S['speed'][starts[i] - 1:stops[i] - 1] for i in sess['kept_trials']])
    allspbin = np.concatenate([o[2] for o in sess['output']])
    ax[2, 1].plot(allsp, allspbin, '.', ms=1)
    ax[2, 1].set_xlabel('speed (cm/s)'); ax[2, 1].set_ylabel('bin')
    alld = np.concatenate([reward_zone_distance(S['pos'][starts[i] - 1:stops[i] - 1],
                                                rz_coords[i, 0], rz_coords[i, 1])
                           for i in sess['kept_trials']])
    alldbin = np.concatenate([o[0] for o in sess['output']])
    ax[3, 0].plot(alld, alldbin, '.', ms=1)
    ax[3, 0].set_xlabel('distance to reward zone (cm)'); ax[3, 0].set_ylabel('bin')

    # neural raster of the converted trial next to position, to check alignment
    ev = sess['neural'][k]
    ax[3, 1].imshow(ev, aspect='auto', vmin=0, vmax=np.percentile(ev, 99.5),
                    extent=[x[0][0], x[0][-1], ev.shape[0], 0], cmap='gray_r')
    ax5 = ax[3, 1].twinx()
    ax5.plot(x[0], pos, 'r', lw=1)
    ax[3, 1].set_xlabel('time from trial start (s)')
    ax[3, 1].set_ylabel('cell')
    ax5.set_ylabel('position (cm)')
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, 'processing_%s_outputs.png' % sid), dpi=110)
    plt.close(fig)

    # ---- per-trial summary: all trials of the session ----
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    trials = sess['kept_trials']
    ax[0].plot(trials, [rz_coords[i, 0] for i in trials], 'g.', label='zone start')
    ax[0].plot(trials, [rz_coords[i, 1] for i in trials], 'r.', label='zone end')
    ax[0].plot(trials, [np.nan if np.isnan(0) else 0 for i in trials], alpha=0)
    ax[0].set_xlabel('lap'); ax[0].set_ylabel('reward zone (cm)'); ax[0].legend(fontsize=7)
    ax[1].plot(trials, [o[5][0] for o in sess['output']], 'k.')
    ax[1].set_xlabel('lap'); ax[1].set_ylabel('reward outcome')
    ax[2].plot(trials, [x[3][0] for x in sess['input']], 'b.')
    ax[2].set_xlabel('lap'); ax[2].set_ylabel('previous outcome (input)')
    ax[3].plot(trials, [x[1][0] for x in sess['input']], 'm.')
    ax[3].set_xlabel('lap'); ax[3].set_ylabel('environment (input)')
    fig.suptitle('%s per-trial variables' % sid)
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, 'processing_%s_trials.png' % sid), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
def _worker(args):
    fn, show, plotdir = args
    try:
        return process_session(fn, show=show, plotdir=plotdir)
    except Exception:
        traceback.print_exc()
        return None, dict(file=os.path.basename(fn), error=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nproc', type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
    if args.sample:
        # two sessions from two different subjects (smallest files, for speed)
        chosen, subs = [], set()
        for fn in sorted(files, key=os.path.getsize):
            sub = os.path.basename(fn).split('_')[0]
            if sub in subs:
                continue
            chosen.append(fn); subs.add(sub)
            if len(chosen) == 2:
                break
        files = sorted(chosen)
    print('processing %d sessions' % len(files))

    show_files = set(files[:2]) if args.show_processing else set()
    t0 = time.time()
    results = []
    if args.nproc > 1 and len(files) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(min(args.nproc, len(files)), maxtasksperchild=1) as pool:
            for res in pool.imap(_worker, [(fn, fn in show_files, '/app') for fn in files]):
                results.append(res)
                st = res[1]
                print('[%5.1f s] %s: %d cells (%d iscell, %d interneurons), '
                      '%d/%d trials kept  (load %.1fs dff %.1fs)'
                      % (time.time() - t0, st.get('file'), st.get('n_neurons', -1),
                         st.get('n_iscell', -1), st.get('n_interneuron', -1),
                         st.get('n_trials_kept', -1), st.get('n_trials_raw', -1),
                         st.get('t_load', 0), st.get('t_dff', 0)))
    else:
        for fn in files:
            res = _worker((fn, fn in show_files, '/app'))
            results.append(res)
            st = res[1]
            print('[%5.1f s] %s: %d cells, %d/%d trials kept'
                  % (time.time() - t0, st.get('file'), st.get('n_neurons', -1),
                     st.get('n_trials_kept', -1), st.get('n_trials_raw', -1)))

    sessions = [r[0] for r in results if r[0] is not None and len(r[0]['neural']) >= 2]
    stats = [r[1] for r in results]
    print('kept %d sessions with >= 2 trials' % len(sessions))

    subjects = sorted({s['subject'] for s in sessions}, key=lambda x: int(x[1:]))
    brain_regions = ['CA1']
    data = dict(
        neural=[s['neural'] for s in sessions],
        input=[s['input'] for s in sessions],
        output=[s['output'] for s in sessions],
        subjects=subjects,
        subject_idx=np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64),
        brain_regions=brain_regions,
        brain_region_idx=[np.zeros(s['n_neurons'], dtype=np.int64) for s in sessions],
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        output_values=OUTPUT_VALUES,
        metadata=dict(
            task_description=(
                'Head-fixed mice ran laps on a 450 cm virtual linear track for water '
                'reward delivered in a hidden 50 cm reward zone (location A: 80-130 cm, '
                'B: 200-250 cm, C: 320-370 cm). The reward zone was moved to a new '
                'location after 30 laps on switch days, and on one switch day the '
                'virtual environment also changed (ENV1 vs ENV2). Reward was omitted on '
                'a subset of laps. Two-photon calcium imaging of hippocampal CA1 '
                'pyramidal neurons (GCaMP7f). The decoder predicts, from CA1 '
                'deconvolved calcium activity: distance to the reward zone, absolute '
                'track position, running speed, licking, the active reward zone '
                'location and the trial reward outcome. (Sosa, Plitt & Giocomo 2025, '
                'Nature Neuroscience; DANDI 001361)'),
            time_bin_size=float(np.mean([s['dt'] for s in sessions]) * 1000.0),
            temporal_alignment_event='trial start (teleport into the linear track, position 0 cm)',
            off_start=-float(np.mean([s['dt'] for s in sessions])),
            off_end=None,
            neural_signal=('deconvolved calcium events (OASIS) computed from per-trial '
                           'maximin dF/F of neuropil-corrected suite2p ROI fluorescence, '
                           'exactly as in reward_relative.preprocessing.dff'),
            neuron_curation=('suite2p manual curation (iscell), exclusion of putative '
                             'interneurons with Pearson r > 0.5 between dF/F and running speed, '
                             'and exclusion of cells whose per-trial maximin baseline falls '
                             'below 5%% of their median raw fluorescence (dF/F numerically '
                             'undefined for these cells)'),
            trial_curation=('complete laps only ([trial_start-1, teleport-1) frames); the first '
                            'lap of each session is dropped because previous-trial outcome is '
                            'undefined; laps with lick-sensor errors (>30% of frames with '
                            'cumulative lick count > 2) are dropped'),
            deviations_from_reference=('the reference GLM/decoder analyses discard samples with '
                                       'speed < 2 cm/s; all within-trial samples are kept here '
                                       'because speed (including a < 2 cm/s class) is a decoder '
                                       'output and the decoder requires contiguous trial time series'),
            session_info=[dict(subject=s['subject'], session=s['session_id'], scene=s['scene'],
                               date=s['date'], n_neurons=s['n_neurons'],
                               n_trials=len(s['neural']),
                               laps=s['kept_trials'].tolist()) for s in sessions],
            source='DANDI dandiset 001361 (NWB), Sosa, Plitt & Giocomo 2025',
        ),
    )

    # ------------------------- summary / sanity checks -------------------------
    ntr = sum(len(s['neural']) for s in sessions)
    nneu = sum(s['n_neurons'] for s in sessions)
    nfr = sum(int(a.shape[1]) for s in sessions for a in s['neural'])
    print('\n=== conversion summary ===')
    print('sessions %d, subjects %d, trials %d, neurons %d, frames %d'
          % (len(sessions), len(subjects), ntr, nneu, nfr))
    ok = [s for s in stats if not s.get('error')]
    print('iscell total %d, interneurons excluded %d (%.2f%%), unstable-baseline cells '
          'excluded %d (%.3f%%), neurons kept %d'
          % (sum(s['n_iscell'] for s in ok), sum(s['n_interneuron'] for s in ok),
             100 * sum(s['n_interneuron'] for s in ok) / max(1, sum(s['n_iscell'] for s in ok)),
             sum(s['n_bad_baseline'] for s in ok),
             100 * sum(s['n_bad_baseline'] for s in ok) / max(1, sum(s['n_iscell'] for s in ok)),
             sum(s['n_neurons'] for s in ok)))
    print('trials raw %d, lick-error %d, kept %d'
          % (sum(s['n_trials_raw'] for s in ok), sum(s['n_lick_error'] for s in ok),
             sum(s['n_trials_kept'] for s in ok)))
    print('reward rate %.3f' % (sum(s['n_rewarded'] for s in ok) / max(1, sum(s['n_trials_raw'] for s in ok))))
    print('median reward-zone onset error %.2f cm (expect 0-15)'
          % np.nanmedian([s['zone_err'] for s in ok]))
    print('dF/F range [%.2f, %.2f], max event %.2f'
          % (min(s['dff_min'] for s in ok), max(s['dff_max'] for s in ok),
             max(s['ev_max'] for s in ok)))
    allout = np.concatenate([o for s in sessions for o in s['output']], axis=1)
    for i, name in enumerate(OUTPUT_NAMES):
        vals, cnt = np.unique(allout[i], return_counts=True)
        print('output %-22s distribution %s' % (name, np.round(cnt / cnt.sum(), 4).tolist()))
    allin = np.concatenate([x for s in sessions for x in s['input']], axis=1)
    for i, name in enumerate(INPUT_NAMES):
        print('input  %-22s range [%.3f, %.3f]' % (name, allin[i].min(), allin[i].max()))
    print('time_bin_size %.4f ms, off_start %.4f s' % (data['metadata']['time_bin_size'],
                                                       data['metadata']['off_start']))

    t1 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('wrote %s (%.2f GB) in %.1f s; total %.1f s'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t1,
             time.time() - t0))

    import pandas as pd
    pd.DataFrame(stats).to_csv(os.path.splitext(args.outfile)[0] + '_session_stats.csv', index=False)


if __name__ == '__main__':
    main()
