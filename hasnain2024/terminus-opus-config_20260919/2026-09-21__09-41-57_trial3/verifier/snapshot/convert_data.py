#!/usr/bin/env python3
"""Convert the Hasnain, Birnbaum et al. (Nat Neurosci 2024) ALM dataset into the
decoder-ready pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

The processing follows the authors' MATLAB pipeline (`DataLoadingScripts/`,
`funcs/kinematics/`) as closely as possible:
  * sessions and the ALM probe(s) of each session are taken from the reference meta
    scripts `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`
  * units with quality garbage/noisy/real? are dropped (findClusters.m)
  * spikes are aligned to the go cue (alignSpikes.m), binned at dt = 10 ms from
    -2.5 s to +2.5 s and smoothed with a causal Gaussian (getSeq.m + mySmooth.m)
  * units with mean firing rate <= 1 Hz are dropped (removeLowFRClusters.m)
  * early-lick and optogenetic-stim trials are dropped (params.condition strings)
  * DLC kinematics and motion energy are interpolated onto the same time axis after
    correcting the video/ephys clock offset (findVideoOffset.m, findPosition.m,
    loadMotionEnergy.m)
"""
import argparse
import os
import pickle
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matio import (load_obj, as_vector, get_clusters, get_traj_view,
                   get_feat_names, load_motion_energy, video_offset)
from session_meta import get_sessions

# ----------------------------------------------------------------- params ---
# identical to WorkingWithDataObjs.m part 2.1
PARAMS = dict(
    alignEvent='goCue',
    tmin=-2.5,
    tmax=2.5,
    dt=0.01,             # 10 ms bins
    smooth=15,           # causal gaussian window, in bins
    bctype='reflect',
    lowFR=1.0,           # Hz
    min_units=10,        # "sessions were included only if they had at least 10 units"
    advance_movement=0.0,
)
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}
VIDEO_FS = 400.0         # Hz (paper)
DEFAULT_VIDSHIFT = 0.5   # s, fallback used by the reference code

OUTPUT_NAMES = ['lick_direction', 'context', 'outcome',
                'tongue_velocity', 'paw_velocity', 'motion_energy']
OUTPUT_VALUES = [
    ['left', 'right', 'none'],
    ['WC', 'DR'],
    ['incorrect', 'correct', 'ignore'],
    ['below_median', 'above_median', 'not_visible'],
    ['below_median', 'above_median', 'not_visible'],
    ['below_median', 'above_median', 'no_video'],
]
INPUT_NAMES = ['time_from_go_cue']


def time_axis():
    """getSeq.m: edges = tmin:dt:tmax; time = edges + dt/2, dropping the last edge."""
    edges = np.arange(PARAMS['tmin'], PARAMS['tmax'] + 1e-12, PARAMS['dt'])
    tm = edges[:-1] + PARAMS['dt'] / 2
    return edges, tm


# --------------------------------------------------------------- smoothing --
def _gausswin(N, alpha=2.5):
    """MATLAB gausswin(N, alpha)."""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * 2 * n / (N - 1)) ** 2)


def causal_gauss_kernel(N=PARAMS['smooth']):
    """mySmooth.m kernel: gausswin with the first half zeroed (causal), normalised."""
    k = _gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()


def my_smooth(x, N=PARAMS['smooth'], bctype=PARAMS['bctype']):
    """Port of utils/mySmooth.m. Operates along the first axis of x (time)."""
    if N in (0, 1):
        return x
    k = causal_gauss_kernel(N)
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[:, None]
    if bctype == 'reflect':
        xf = np.concatenate([x[:N, :], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        xf = np.concatenate([np.zeros((N, x.shape[1])), x], axis=0)
        trim = N
    else:
        xf = x
        trim = 0
    out = np.empty_like(xf)
    for j in range(xf.shape[1]):
        out[:, j] = np.convolve(xf[:, j], k, mode='same')
    out = out[trim:, :]
    return out[:, 0] if single else out


# ------------------------------------------------------------------ neural --
def probe_regions(obj, probes):
    """Brain region label for each requested probe, from obj.ex/obj.meta probe.loc.

    The reference meta scripts (`load<ANM>_ALMVideo.m`) select the probe(s) to analyse;
    `obj.ex.probe.loc` gives the recording location of each probe (e.g. 'L ALM', 'R M1TJ').
    Older sessions (JEB7, JGR2, JGR3) have no `ex`/`meta` field; the paper states that all
    of these recordings were made in ALM, so ALM is used as the default.
    """
    locs = None
    for key in ('ex', 'meta'):
        d = obj.get(key)
        if isinstance(d, dict) and isinstance(d.get('probe'), dict) and 'loc' in d['probe']:
            L = d['probe']['loc']
            locs = [L] if isinstance(L, str) else [str(x) for x in np.atleast_1d(L).ravel()]
            break
    out = []
    for p in probes:
        name = 'ALM'
        if locs is not None and len(locs) >= p:
            l = locs[p - 1].upper()
            if 'M1TJ' in l or 'TJM1' in l:
                name = 'tjM1'
            elif 'ALM' in l:
                name = 'ALM'
        out.append(name)
    return out


def bin_spikes(obj, probes, trials, align_times, edges):
    """Return (rates, quality) where rates is (ntrials, nbins, nunits) in spikes/s.

    Replicates alignSpikes.m + getSeq.m: spike times within a trial minus the
    alignment event, histogrammed on `edges`, divided by dt, then smoothed with the
    causal Gaussian kernel.
    """
    nb = len(edges) - 1
    ntr = len(trials)
    trial_pos = -np.ones(int(max(trials.max(), 0)) + 2, dtype=np.int64)
    trial_pos[trials] = np.arange(ntr)

    all_counts = []
    qualities = []
    spikes_per_trial = np.zeros(ntr, dtype=np.int64)
    units_per_probe = []
    for p in probes:
        clusters = get_clusters(obj, p)
        keep = [i for i, c in enumerate(clusters)
                if str(c.get('quality', '')).strip().lower() not in BAD_QUALITY]
        if not keep:
            continue
        counts = np.zeros((ntr, nb, len(keep)), dtype=np.float32)
        for j, ci in enumerate(keep):
            c = clusters[ci]
            qualities.append(str(c.get('quality', '')).strip().lower())
            tr = np.asarray(c['trial'], dtype=np.int64).ravel()
            tm = np.asarray(c['trialtm'], dtype=float).ravel()
            if tr.size == 0:
                continue
            ok = (tr >= 0) & (tr < trial_pos.size)
            tr, tm = tr[ok], tm[ok]
            pos = trial_pos[tr]
            sel = pos >= 0
            if not np.any(sel):
                continue
            pos = pos[sel]
            np.add.at(spikes_per_trial, pos, 1)
            aligned = tm[sel] - align_times[pos]
            b = np.floor((aligned - edges[0]) / PARAMS['dt']).astype(np.int64)
            good = (b >= 0) & (b < nb)
            np.add.at(counts[:, :, j], (pos[good], b[good]), 1.0)
        all_counts.append(counts)
        units_per_probe.append((p, len(keep)))
    if not all_counts:
        return np.zeros((ntr, nb, 0), dtype=np.float32), [], spikes_per_trial, units_per_probe
    counts = np.concatenate(all_counts, axis=2)
    rates = counts / PARAMS['dt']
    # smooth along time for every (trial, unit)
    flat = rates.transpose(1, 0, 2).reshape(nb, -1)
    sm = my_smooth(flat)
    rates = sm.reshape(nb, ntr, -1).transpose(1, 0, 2).astype(np.float32)
    return rates, qualities, spikes_per_trial, units_per_probe


# ------------------------------------------------------------------- video --
def get_vidshift(obj):
    try:
        vs = video_offset(obj)
        if np.isfinite(vs):
            return float(vs)
    except Exception:
        pass
    return DEFAULT_VIDSHIFT


def interp_positions(traj_trials, featix, trials, align_times, taxis, vidshift):
    """findPosition.m: interpolate DLC x/y of one feature onto taxis for each trial.

    Returns x, y of shape (ntrials, ntimepoints); NaN where the feature is not
    visible or the video does not cover that time.
    """
    ntr = len(trials)
    X = np.full((ntr, taxis.size), np.nan)
    Y = np.full((ntr, taxis.size), np.nan)
    for i, tr in enumerate(trials):
        t = traj_trials[tr - 1]
        ts = np.asarray(t['ts'], dtype=float)
        if ts.ndim != 3 or ts.shape[0] < 2:
            continue
        ft = np.asarray(t.get('frameTimes', np.nan), dtype=float).ravel()
        if ft.size != ts.shape[0] or not np.any(np.isfinite(ft)):
            # reference fallback: frameTimes = (1:nframes)/400, shift 0.5 s
            ft = (np.arange(1, ts.shape[0] + 1)) / VIDEO_FS
            tt = ft - DEFAULT_VIDSHIFT - align_times[i]
        else:
            tt = ft - vidshift - align_times[i]
        for k, arr in ((0, X), (1, Y)):
            v = ts[:, k, featix]
            arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
            # np.interp does not propagate NaNs, do it explicitly
            nanmask = np.interp(taxis, tt, np.isnan(v).astype(float),
                                left=1.0, right=1.0) > 0
            arr[i][nanmask] = np.nan
    return X, Y


def _gradient_nan(A):
    """Central difference along axis 1, falling back to a one-sided difference when
    one neighbour is missing (e.g. at the first/last visible frame of a lick).

    findVelocity.m uses MATLAB `gradient`, which is the central difference; that
    returns NaN next to NaNs, which would incorrectly mark the first and last
    timepoint of every lick as 'not visible'. The one-sided fallback keeps the
    velocity defined wherever the position is.
    """
    with np.errstate(invalid='ignore'):
        fwd = np.full_like(A, np.nan)
        bwd = np.full_like(A, np.nan)
        fwd[:, :-1] = A[:, 1:] - A[:, :-1]
        bwd[:, 1:] = A[:, 1:] - A[:, :-1]
        central = 0.5 * (fwd + bwd)
        out = np.where(np.isfinite(central), central,
                       np.where(np.isfinite(fwd), fwd, bwd))
        out[~np.isfinite(A)] = np.nan
    return out


def speed_from_positions(X, Y):
    """findVelocity.m: velocity = gradient of the interpolated position; speed = |v|."""
    with np.errstate(invalid='ignore'):
        vx = _gradient_nan(X)
        vy = _gradient_nan(Y)
        return np.sqrt(vx ** 2 + vy ** 2)


def discretize(values, visible):
    """0 = < session median, 1 = >= session median, 2 = not visible/no video."""
    out = np.full(values.shape, 2, dtype=np.int64)
    vis = visible & np.isfinite(values)
    if np.any(vis):
        thr = np.percentile(values[vis], 50)
        out[vis] = (values[vis] >= thr).astype(np.int64)
    else:
        thr = np.nan
    return out, thr


def motion_energy_on_axis(me_trials, traj_trials, trials, align_times, taxis, vidshift):
    """loadMotionEnergy.m: interpolate the 400 Hz motion energy onto taxis."""
    ntr = len(trials)
    ME = np.full((ntr, taxis.size), np.nan)
    if me_trials is None:
        return ME
    for i, tr in enumerate(trials):
        if tr - 1 >= len(me_trials):
            continue
        d = np.asarray(me_trials[tr - 1], dtype=float).ravel()
        if d.size < 2:
            continue
        t = traj_trials[tr - 1]
        ft = np.asarray(t.get('frameTimes', np.nan), dtype=float).ravel()
        if ft.size != d.size or not np.any(np.isfinite(ft)):
            ft = np.arange(1, d.size + 1) / VIDEO_FS
            tt = ft - DEFAULT_VIDSHIFT - align_times[i]
        else:
            tt = ft - vidshift - align_times[i]
        ME[i] = np.interp(taxis, tt, d, left=np.nan, right=np.nan)
    return ME


# --------------------------------------------------------------- per session -
def process_session(sess, show_processing=False, outdir='/app'):
    t0 = time.time()
    timing = {}
    obj = load_obj(sess['datafile'])
    timing['load'] = time.time() - t0

    bp = obj['bp']
    N = int(as_vector(bp['Ntrials'])[0])
    hit = as_vector(bp['hit'])[:N].astype(bool)
    miss = as_vector(bp['miss'])[:N].astype(bool)
    no = as_vector(bp['no'])[:N].astype(bool)
    R = as_vector(bp['R'])[:N].astype(bool)
    L = as_vector(bp['L'])[:N].astype(bool)
    early = as_vector(bp['early'])[:N].astype(bool)
    aw = as_vector(bp['autowater'])[:N].astype(bool)
    stim = as_vector(bp['stim']['enable'])[:N]
    stim = np.nan_to_num(stim).astype(bool)
    gocue = as_vector(bp['ev'][PARAMS['alignEvent']])[:N]

    # --- trial curation: exclude early-lick and optogenetic stim trials
    keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
    trials = np.where(keep)[0] + 1          # 1-based trial numbers, as in MATLAB
    ntr = trials.size
    align_times = gocue[trials - 1]

    edges, taxis = time_axis()

    # --- neural
    t1 = time.time()
    rates, quals, spk_per_trial, units_per_probe = bin_spikes(obj, sess['probe'], trials, align_times, edges)
    timing['spikes'] = time.time() - t1

    # Some sessions' ephys recording stops before the behavioural session ends; those
    # trials contain no spikes from any unit and carry no neural information, so they
    # are dropped (data-quality curation).
    recorded = spk_per_trial > 0
    n_not_recorded = int((~recorded).sum())
    if n_not_recorded:
        trials = trials[recorded]
        align_times = align_times[recorded]
        rates = rates[recorded]
        ntr = trials.size

    # low firing rate filter (removeLowFRClusters.m; methods: FR > 1 Hz)
    if rates.shape[2]:
        meanfr = rates.mean(axis=(0, 1))
        keep_u = meanfr > PARAMS['lowFR']
    else:
        meanfr = np.zeros(0)
        keep_u = np.zeros(0, dtype=bool)
    regions = probe_regions(obj, [p for p, n in units_per_probe])
    unit_region = np.array([reg for reg, (p, n) in zip(regions, units_per_probe)
                            for _ in range(n)], dtype=object)
    rates = rates[:, :, keep_u]
    unit_region = unit_region[keep_u] if unit_region.size else unit_region
    nunits = rates.shape[2]

    # --- video kinematics
    t1 = time.time()
    vidshift = get_vidshift(obj)
    side = get_traj_view(obj, 1)
    bottom = get_traj_view(obj, 2)
    side_feats = get_feat_names(side[0])
    bot_feats = get_feat_names(bottom[0])

    tongue_ix = side_feats.index('tongue')
    Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
    tongue_speed = speed_from_positions(Xt, Yt)
    tongue_visible = np.isfinite(Xt) & np.isfinite(Yt)
    # findVelocity.m sets the tongue velocity to 0 wherever it cannot be computed;
    # here that only happens for an isolated visible frame (no neighbour to difference
    # against). Those timepoints stay 'visible' with zero speed.
    tongue_speed = np.where(tongue_visible & ~np.isfinite(tongue_speed), 0.0, tongue_speed)

    paw_speeds = []
    paw_vis = []
    for f in ('top_paw', 'bottom_paw'):
        ix = bot_feats.index(f)
        Xp, Yp = interp_positions(bottom, ix, trials, align_times, taxis, vidshift)
        paw_speeds.append(speed_from_positions(Xp, Yp))
        paw_vis.append(np.isfinite(Xp) & np.isfinite(Yp))
    paw_vis = np.any(np.stack(paw_vis), axis=0)
    with np.errstate(invalid='ignore'):
        paw_speed = np.nanmean(np.stack(paw_speeds), axis=0)
    paw_speed = np.where(paw_vis & ~np.isfinite(paw_speed), 0.0, paw_speed)
    timing['kinematics'] = time.time() - t1

    # --- motion energy
    t1 = time.time()
    me_trials = None
    if sess.get('mefile'):
        try:
            me_trials, _thr = load_motion_energy(sess['mefile'])
        except Exception:
            me_trials = None
    ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
    me_visible = np.isfinite(ME)
    timing['motion_energy'] = time.time() - t1

    # --- discretise the three movement variables with a per-session median
    tongue_cls, tongue_thr = discretize(tongue_speed, tongue_visible)
    paw_cls, paw_thr = discretize(paw_speed, paw_vis)
    me_cls, me_thr = discretize(ME, me_visible)

    # --- per-trial categorical outputs
    tr0 = trials - 1
    right = (R[tr0] & hit[tr0]) | (L[tr0] & miss[tr0])
    lick = np.where(no[tr0], 2, np.where(right, 1, 0))          # 0 left, 1 right, 2 none
    context = np.where(aw[tr0], 0, 1)                            # 0 WC, 1 DR
    outcome = np.where(no[tr0], 2, np.where(hit[tr0], 1, 0))     # 0 incorrect,1 correct,2 ignore

    # --- assemble per-trial arrays
    neural, inp, outp = [], [], []
    T = taxis.size
    tin = taxis.astype(np.float32)[None, :]
    for i in range(ntr):
        neural.append(np.ascontiguousarray(rates[i].T))           # (nunits, T)
        inp.append(tin.copy())
        o = np.empty((6, T), dtype=np.int64)
        o[0] = lick[i]
        o[1] = context[i]
        o[2] = outcome[i]
        o[3] = tongue_cls[i]
        o[4] = paw_cls[i]
        o[5] = me_cls[i]
        outp.append(o)

    info = dict(
        session=sess['sessid'], anm=sess['anm'], date=sess['date'],
        dataset=sess['dataset'], probes=sess['probe'], vidshift=vidshift,
        ntrials_total=N, ntrials_used=ntr,
        n_early=int(early.sum()), n_stim=int(stim.sum()),
        n_not_recorded=n_not_recorded,
        nunits_quality=int(len(quals)), nunits=int(nunits),
        regions={str(r): int(np.sum(unit_region == r)) for r in set(unit_region)} if nunits else {},
        n_single_units=int(sum(1 for q, k in zip(quals, keep_u)
                               if k and q in ('excellent', 'great', 'good'))),
        tongue_thr=float(tongue_thr) if np.isfinite(tongue_thr) else None,
        paw_thr=float(paw_thr) if np.isfinite(paw_thr) else None,
        me_thr=float(me_thr) if np.isfinite(me_thr) else None,
        has_me=me_trials is not None,
        frac_tongue_visible=float(tongue_visible.mean()) if ntr else 0.0,
        frac_paw_visible=float(paw_vis.mean()) if ntr else 0.0,
        frac_me_visible=float(me_visible.mean()) if ntr else 0.0,
        timing=timing,
    )

    if show_processing and ntr:
        try:
            plot_processing(sess, taxis, rates, tongue_speed, tongue_visible, tongue_cls,
                            paw_speed, paw_vis, paw_cls, ME, me_visible, me_cls,
                            lick, context, outcome, bp, trials, align_times, outdir)
        except Exception:
            traceback.print_exc()

    info['total_s'] = time.time() - t0
    return dict(neural=neural, input=inp, output=outp, info=info,
                unit_region=list(unit_region))


# ------------------------------------------------------------------- plots ---
def plot_processing(sess, taxis, rates, tongue_speed, tongue_vis, tongue_cls,
                    paw_speed, paw_vis, paw_cls, ME, me_vis, me_cls,
                    lick, context, outcome, bp, trials, align_times, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    sid = sess['sessid']

    # 1. population raster (mean rate) and a few single units
    ax = axs[0, 0]
    m = rates.mean(axis=0)  # (T, units)
    if m.shape[1]:
        ax.imshow(m.T, aspect='auto', origin='lower',
                  extent=[taxis[0], taxis[-1], 0, m.shape[1]], cmap='magma')
    ax.axvline(0, color='c', lw=2)
    ax.set_title(f'{sid}: trial-averaged rate (spks/s), {rates.shape[2]} units')
    ax.set_xlabel('time from go cue (s)')
    ax.set_ylabel('unit')

    ax = axs[0, 1]
    for u in range(min(4, rates.shape[2])):
        ax.plot(taxis, rates[:, :, u].mean(axis=0), label=f'unit {u}')
    ax.axvline(0, color='k', ls='--')
    ax.set_title('example unit PSTHs (aligned to go cue)')
    ax.set_xlabel('time from go cue (s)')
    ax.legend(fontsize=7)

    # 2. tongue speed + visibility + discretisation
    ax = axs[1, 0]
    ax.imshow(tongue_cls, aspect='auto', origin='lower', interpolation='nearest',
              extent=[taxis[0], taxis[-1], 0, tongue_cls.shape[0]], cmap='viridis')
    ax.axvline(0, color='r', lw=1)
    ax.set_title('tongue velocity class (0 low, 1 high, 2 not visible)')
    ax.set_xlabel('time from go cue (s)'); ax.set_ylabel('trial')

    ax = axs[1, 1]
    ax.plot(taxis, np.nanmean(tongue_vis, axis=0), label='P(tongue visible)')
    ax.plot(taxis, np.nanmean(me_vis, axis=0), label='P(ME available)')
    ax.plot(taxis, np.nanmean(paw_vis, axis=0), label='P(paw visible)')
    ax.axvline(0, color='k', ls='--')
    ax.set_title('visibility vs time (tongue should rise after the go cue)')
    ax.set_xlabel('time from go cue (s)'); ax.legend(fontsize=7)

    # 3. single trial example: tongue speed with threshold
    ax = axs[2, 0]
    ex = int(np.argmax(np.sum(tongue_vis, axis=1)))
    ax.plot(taxis, tongue_speed[ex], 'k', label='tongue speed')
    vis = tongue_vis[ex]
    thr = np.percentile(tongue_speed[tongue_vis], 50) if np.any(tongue_vis) else np.nan
    ax.axhline(thr, color='r', ls='--', label='session median')
    ax.plot(taxis[~vis], np.zeros((~vis).sum()), '.', color='gray', ms=2, label='not visible')
    ax.axvline(0, color='b')
    ax.set_title(f'trial {trials[ex]}: tongue speed and discretisation threshold')
    ax.set_xlabel('time from go cue (s)'); ax.legend(fontsize=7)

    ax = axs[2, 1]
    ax.plot(taxis, ME[ex], 'k', label='motion energy')
    thrme = np.percentile(ME[me_vis], 50) if np.any(me_vis) else np.nan
    ax.axhline(thrme, color='r', ls='--', label='session median')
    ax.axvline(0, color='b')
    ax.set_title(f'trial {trials[ex]}: motion energy')
    ax.set_xlabel('time from go cue (s)'); ax.legend(fontsize=7)

    # 4. per-trial outputs and sanity: mean rate aligned to go cue by lick direction
    ax = axs[3, 0]
    ax.plot(lick, '.', label='lick dir (0 L,1 R,2 none)')
    ax.plot(context + 0.05, '.', label='context (0 WC,1 DR)')
    ax.plot(outcome + 0.1, '.', label='outcome (0 inc,1 cor,2 ign)')
    ax.set_xlabel('trial (curated)'); ax.legend(fontsize=7)
    ax.set_title('per-trial outputs across the session (blocks visible in context)')

    ax = axs[3, 1]
    for cls, nm in ((0, 'left'), (1, 'right'), (2, 'none')):
        sel = lick == cls
        if sel.sum() and rates.shape[2]:
            ax.plot(taxis, rates[sel].mean(axis=(0, 2)), label=f'{nm} (n={sel.sum()})')
    ax.axvline(0, color='k', ls='--')
    ax.set_title('population mean rate by lick direction')
    ax.set_xlabel('time from go cue (s)'); ax.legend(fontsize=7)

    fig.tight_layout()
    fn = os.path.join(outdir, f'processing_{sid}.png')
    fig.savefig(fn, dpi=90)
    plt.close(fig)
    print(f'  wrote {fn}')


def _worker(args):
    sess, show, outdir = args
    try:
        return process_session(sess, show, outdir)
    except Exception:
        traceback.print_exc()
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nworkers', type=int, default=8)
    args = ap.parse_args()

    sessions = get_sessions()
    if args.sample:
        # one fixed-delay (two-context) and one randomized-delay session
        pick = [s for s in sessions if s['sessid'] == 'JEB13_2022-09-13']
        pick += [s for s in sessions if s['sessid'] == 'JEB24_2023-10-24']
        sessions = pick if len(pick) == 2 else sessions[:2]
    print(f'Processing {len(sessions)} sessions')

    t0 = time.time()
    outdir = os.path.dirname(os.path.abspath(args.outfile)) or '.'
    jobs = [(s, args.show_processing, outdir) for s in sessions]
    nw = min(args.nworkers, len(jobs))
    if nw > 1:
        with Pool(nw) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    data = dict(neural=[], input=[], output=[], subjects=[], subject_idx=[],
                brain_regions=[], brain_region_idx=[],
                input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                output_values=OUTPUT_VALUES, metadata={})
    session_info = []
    nskipped = []
    for sess, res in zip(sessions, results):
        if res is None:
            nskipped.append((sess['sessid'], 'error'))
            continue
        info = res['info']
        if info['nunits'] < PARAMS['min_units']:
            nskipped.append((sess['sessid'], f"only {info['nunits']} units"))
            print(f"  SKIP {sess['sessid']}: {info['nunits']} units < {PARAMS['min_units']}")
            continue
        if info['ntrials_used'] < 2:
            nskipped.append((sess['sessid'], 'fewer than 2 trials'))
            continue
        anm = sess['anm']
        if anm not in data['subjects']:
            data['subjects'].append(anm)
        data['subject_idx'].append(data['subjects'].index(anm))
        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        for reg in [str(r) for r in res['unit_region']]:
            if reg not in data['brain_regions']:
                data['brain_regions'].append(reg)
        data['brain_region_idx'].append(
            np.array([data['brain_regions'].index(str(r)) for r in res['unit_region']], dtype=np.int64))
        session_info.append(info)
        print(f"  {sess['sessid']:22s} trials {info['ntrials_used']:4d}/{info['ntrials_total']:4d} "
              f"units {info['nunits']:4d} (SU {info['n_single_units']:3d}) "
              f"tongue_vis {info['frac_tongue_visible']:.2f} me {info['frac_me_visible']:.2f} "
              f"[{info['total_s']:.1f}s]", flush=True)

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    _, tm = time_axis()
    data['metadata'] = dict(
        task_description=(
            'Head-fixed mice performed two directional licking tasks that alternated in blocks '
            'within a session (Hasnain, Birnbaum et al., Nat Neurosci 2024). In the delayed-response '
            '(DR) context an auditory tone indicated the rewarded lickport, and after a delay an '
            'auditory go cue instructed the animal to lick. In the water-cued (WC) context all '
            'auditory cues were omitted and a water drop was delivered at a random time and port. '
            'Decoder outputs: lick direction (left/right/none), behavioural context (WC/DR), trial '
            'outcome (incorrect/correct/ignore), and three discretised movement variables '
            '(tongue velocity, paw velocity, motion energy; 0 = below the session median, '
            '1 = at or above it, 2 = feature not visible / no video).'),
        time_bin_size=PARAMS['dt'] * 1000.0,
        temporal_alignment_event=('go cue onset (auditory go cue on DR trials, water-drop '
                                  'presentation on WC trials; obj.bp.ev.goCue)'),
        off_start=PARAMS['tmin'],
        off_end=PARAMS['tmax'],
        time_axis=tm.astype(np.float32),
        smoothing=('causal Gaussian kernel over 15 bins (150 ms), reflect boundary, '
                   'as in utils/mySmooth.m'),
        neural_units='spikes/s (smoothed firing rate)',
        neuron_curation=('quality label not in {garbage, noisy, real?} and mean firing rate > 1 Hz; '
                         'only probes located in ALM'),
        trial_curation='early-lick trials and optogenetic stimulation trials excluded',
        session_curation='sessions listed in the reference meta scripts with >= 10 curated units',
        video_sampling_rate_hz=VIDEO_FS,
        source='Hasnain, Birnbaum et al., Nature Neuroscience 2024 (Zenodo 10.5281/zenodo.13941415)',
        session_info=session_info,
        skipped_sessions=nskipped,
    )

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    ntrials = sum(len(x) for x in data['neural'])
    nunits = sum(x[0].shape[0] for x in data['neural'])
    print(f'\nWrote {args.outfile}: {len(data["neural"])} sessions, {len(data["subjects"])} subjects, '
          f'{ntrials} trials, {nunits} units, T={tm.size} bins')
    print(f'skipped: {nskipped}')
    print(f'total time {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
