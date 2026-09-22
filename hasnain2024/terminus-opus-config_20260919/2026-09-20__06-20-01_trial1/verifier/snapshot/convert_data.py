#!/usr/bin/env python3
"""Convert Hasnain, Birnbaum et al. 2024 (Nat Neurosci) ALM ephys data to the
decoder-compatible pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference MATLAB pipeline in /app/code:
  loadObjs.m -> processData.m -> findTrials.m / findClusters.m / alignSpikes.m /
  getSeq.m / removeLowFRClusters.m, plus funcs/kinematics/* and
  DataLoadingScripts/loadMotionEnergy.m for the video-derived signals.

See /app/CONVERSION_NOTES.md for the full mapping and justification.
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import h5py
import scipy.io as sio

# ----------------------------------------------------------------------------
# Parameters (mirrors params in the reference scripts, e.g. Scripts/Figure 3/Figure3c.m)
# ----------------------------------------------------------------------------
ALIGN_EVENT = 'goCue'
TMIN = -2.5           # params.tmin
TMAX = 2.5            # params.tmax
DT = 0.03             # params.dt = (1/100)*3  (30 ms bins)
SMOOTH = 15           # params.smooth (causal gaussian kernel width in bins)
LOW_FR = 1.0          # params.lowFR (spikes/s); paper: "units with firing rates exceeding 1 Hz"
MIN_UNITS = 10        # paper: "sessions were included ... only if they had at least 10 units"
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}   # findClusters.m, quality='all'

# ----------------------------------------------------------------------------
# Reference session list, transcribed from
# /app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m
# (commented-out entries in those files are excluded)
# ----------------------------------------------------------------------------
FIXED = [
    ('Ephys_Behavior', 'JEB6',  '2021-04-18', [2]),
    ('Ephys_Behavior', 'JEB7',  '2021-04-29', [1]),
    ('Ephys_Behavior', 'JEB7',  '2021-04-30', [1]),
    ('Ephys_Behavior', 'EKH1',  '2021-08-07', [2]),
    ('Ephys_Behavior', 'EKH3',  '2021-08-11', [2]),
    ('Ephys_Behavior', 'JGR2',  '2021-11-16', [1]),
    ('Ephys_Behavior', 'JGR2',  '2021-11-17', [1]),
    ('Ephys_Behavior', 'JGR3',  '2021-11-18', [1]),
    ('Ephys_Behavior', 'JEB13', '2022-09-13', [2]),
    ('Ephys_Behavior', 'JEB13', '2022-09-14', [2]),
    ('Ephys_Behavior', 'JEB13', '2022-09-21', [1]),
    ('Ephys_Behavior', 'JEB13', '2022-09-24', [1]),
    ('Ephys_Behavior', 'JEB13', '2022-09-25', [1]),
    ('Ephys_Behavior', 'JEB14', '2022-08-22', [1]),
    ('Ephys_Behavior', 'JEB14', '2022-08-23', [1]),
    ('Ephys_Behavior', 'JEB14', '2022-08-24', [1]),
    ('Ephys_Behavior', 'JEB14', '2022-08-25', [1]),
    ('Ephys_Behavior', 'JEB15', '2022-07-26', [1, 2]),
    ('Ephys_Behavior', 'JEB15', '2022-07-27', [1, 2]),
    ('Ephys_Behavior', 'JEB15', '2022-07-28', [1, 2]),
    ('Ephys_Behavior', 'JEB15', '2022-07-29', [2]),
    ('Ephys_Behavior', 'JEB19', '2023-04-18', [1]),
    ('Ephys_Behavior', 'JEB19', '2023-04-19', [1]),
    ('Ephys_Behavior', 'JEB19', '2023-04-20', [1]),
    ('Ephys_Behavior', 'JEB19', '2023-04-21', [1]),
]
RANDOM = [
    ('RandomizedDelay_Ephys_Behavior', 'JEB11', '2022-05-10', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB11', '2022-05-11', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB12', '2022-05-12', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB12', '2022-05-13', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-10', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-11', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-12', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-13', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-18', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-19', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB23', '2023-10-21', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-23', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-24', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-25', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-26', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-27', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-10-31', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-02', [1]),
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]
SESSIONS = FIXED + RANDOM
DATA_ROOT = '/app/data'


def data_path(entry):
    d, anm, date, _ = entry
    return f'{DATA_ROOT}/{d}/data_structure_{anm}_{date}.mat'


def me_path(entry):
    d, anm, date, _ = entry
    return f'{DATA_ROOT}/{d}/motionEnergy_{anm}_{date}.mat'


# ----------------------------------------------------------------------------
# Unified MAT v7 / v7.3 loader for the data objects
# ----------------------------------------------------------------------------
def is_hdf5(fn):
    with open(fn, 'rb') as f:
        return b'7.3' in f.read(120)


def _h5str(f, ref):
    o = f[ref] if isinstance(ref, h5py.Reference) else ref
    a = np.asarray(o[()]).ravel()
    try:
        return ''.join(chr(int(c)) for c in a)
    except Exception:
        return ''


class Session:
    """Accessor for one `data_structure_<ANM>_<DATE>.mat` file (obj struct)."""

    def __init__(self, fn):
        self.fn = fn
        self.hdf5 = is_hdf5(fn)
        if self.hdf5:
            self.f = h5py.File(fn, 'r')
            self.obj = self.f['obj']
        else:
            self.obj = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)['obj']

    def bpvec(self, name):
        if self.hdf5:
            return np.asarray(self.obj['bp/' + name][()]).ravel()
        o = self.obj.bp
        for part in name.split('/'):
            o = getattr(o, part)
        return np.asarray(o).ravel()

    def ev(self, name):
        if self.hdf5:
            return np.asarray(self.obj['bp/ev/' + name][()]).ravel()
        return np.asarray(getattr(self.obj.bp.ev, name)).ravel()

    def ev_cell(self, name):
        out = []
        if self.hdf5:
            for r in self.obj['bp/ev/' + name][()].ravel():
                d = self.f[r]
                if d.attrs.get('MATLAB_empty', 0) == 1:
                    out.append(np.array([]))
                else:
                    out.append(np.asarray(d[()], dtype=float).ravel())
        else:
            for x in getattr(self.obj.bp.ev, name):
                out.append(np.atleast_1d(np.asarray(x, dtype=float)).ravel())
        return out

    @property
    def ntrials(self):
        if self.hdf5:
            return int(np.asarray(self.obj['bp/Ntrials'][()]).ravel()[0])
        return int(self.obj.bp.Ntrials)

    def has_bp_field(self, name):
        if self.hdf5:
            return name in self.obj['bp']
        return name in self.obj.bp._fieldnames

    def nprobes(self):
        if self.hdf5:
            return 0 if 'clu' not in self.obj else self.obj['clu'][()].shape[0]
        clu = self.obj.clu
        if isinstance(clu, np.ndarray) and clu.dtype == object and clu.size:
            if isinstance(np.atleast_1d(clu)[0], np.ndarray):
                return len(clu)
        return 1

    def clusters(self, prb):
        """List of dicts (quality, trial, trialtm) for 0-based probe index prb."""
        out = []
        if self.hdf5:
            if 'clu' not in self.obj:
                return out
            o = self.f[self.obj['clu'][()][prb, 0]]
            if not isinstance(o, h5py.Group):
                return out
            qual, trial, trialtm = o['quality'][()].ravel(), o['trial'][()].ravel(), o['trialtm'][()].ravel()
            for i in range(len(qual)):
                out.append(dict(quality=_h5str(self.f, qual[i]),
                                trial=np.asarray(self.f[trial[i]][()]).ravel(),
                                trialtm=np.asarray(self.f[trialtm[i]][()]).ravel()))
        else:
            clu = self.obj.clu
            arr = clu[prb] if self.nprobes() > 1 else clu
            for c in np.atleast_1d(arr):
                out.append(dict(quality=str(c.quality),
                                trial=np.atleast_1d(np.asarray(c.trial)).ravel(),
                                trialtm=np.atleast_1d(np.asarray(c.trialtm, dtype=float)).ravel()))
        return out

    def feat_names(self, view):
        if self.hdf5:
            g = self.f[self.obj['traj'][()][view, 0]]
            first = self.f[g['featNames'][()].ravel()[0]][()].ravel()
            return [_h5str(self.f, r) for r in first]
        return [str(x) for x in np.atleast_1d(self.obj.traj[view][0].featNames)]

    def traj_trial(self, view, trix):
        """(ts (frames,3,feat), frameTimes (frames,), NdroppedFrames)."""
        if self.hdf5:
            g = self.f[self.obj['traj'][()][view, 0]]
            ts = np.asarray(self.f[g['ts'][()][trix, 0]][()])
            ts = np.transpose(ts, (2, 1, 0)) if ts.ndim == 3 else np.full((0, 3, 0), np.nan)
            ft = np.asarray(self.f[g['frameTimes'][()][trix, 0]][()], dtype=float).ravel()
            nd = np.asarray(self.f[g['NdroppedFrames'][()][trix, 0]][()]).ravel()
        else:
            t = self.obj.traj[view][trix]
            ts = np.asarray(t.ts, dtype=float)
            ft = np.atleast_1d(np.asarray(t.frameTimes, dtype=float)).ravel()
            nd = np.atleast_1d(np.asarray(t.NdroppedFrames)).ravel()
        return ts, ft, (nd[0] if nd.size else np.nan)

    def ntraj_trials(self, view=0):
        if self.hdf5:
            return self.f[self.obj['traj'][()][view, 0]]['ts'].shape[0]
        return len(self.obj.traj[view])

    def vidshift(self):
        """funcs/findVideoOffset.m: mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)"""
        from scipy.stats import mode as _mode
        bitStart = _mode(self.ev('bitStart'), keepdims=False).mode
        if self.hdf5:
            bs = np.asarray(self.obj['sglx/bitcode/bitstart'][()]).ravel()
            fs = float(np.asarray(self.obj['sglx/fs'][()]).ravel()[0])
        else:
            bs = np.atleast_1d(np.asarray(self.obj.sglx.bitcode.bitstart, dtype=float)).ravel()
            fs = float(np.asarray(self.obj.sglx.fs).ravel()[0])
        bs = bs[~np.isnan(bs)]
        return float(_mode(bs, keepdims=False).mode / fs - bitStart)

    def probe_locs(self):
        out = []
        try:
            if self.hdf5:
                d = self.obj['ex/probe/loc']
                if d.dtype == object:
                    pl = d[()]
                    out = [_h5str(self.f, pl[i, 0]).strip() for i in range(pl.shape[0])]
                else:
                    out = [''.join(chr(int(c)) for c in np.asarray(d[()]).ravel()).strip()]
            else:
                loc = self.obj.ex.probe.loc
                out = [str(x).strip() for x in np.atleast_1d(loc)]
        except Exception:
            pass
        return out

    def close(self):
        if self.hdf5:
            self.f.close()


def load_motion_energy(fn):
    """Return (list of per-trial motion-energy arrays, moveThresh or None).

    Three file variants exist in the dataset: me struct with cell .data, me struct
    whose .data is itself a struct (JEB15), and a bare cell array (some JEB23).
    """
    me = sio.loadmat(fn)['me']
    thresh = [None]

    def unpack(x):
        if isinstance(x, np.ndarray) and x.dtype.names is not None and 'data' in x.dtype.names:
            if 'moveThresh' in x.dtype.names:
                t = x['moveThresh']
                t = t[0, 0] if t.dtype == object else t
                t = np.asarray(t).ravel()
                if t.size:
                    thresh[0] = float(t[0])
            d = x['data']
            d = d[0, 0] if d.dtype == object and d.shape == (1, 1) else d
            return unpack(d)
        return x

    data = np.asarray(unpack(me))
    if data.dtype == object:
        trials = [np.asarray(data.ravel()[i], dtype=float).ravel() for i in range(data.size)]
    else:
        trials = [np.asarray(data, dtype=float).ravel()]
    return trials, thresh[0]


# ----------------------------------------------------------------------------
# Signal processing helpers (ports of utils/mySmooth.m and DataLoadingScripts/getSeq.m)
# ----------------------------------------------------------------------------
def time_axis():
    """getSeq.m: edges = tmin:dt:tmax; time = edges + dt/2 with the last dropped."""
    nedges = int(np.floor((TMAX - TMIN) / DT)) + 1
    edges = TMIN + DT * np.arange(nedges)
    t = edges + DT / 2.0
    return edges, t[:-1]


def gausswin(N, alpha=2.5):
    """MATLAB gausswin(N, alpha)."""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def causal_kernel(N=SMOOTH):
    """mySmooth.m kernel: gausswin with the first half zeroed (causal), normalized."""
    k = gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()


def mysmooth(x, N=SMOOTH, bctype='reflect'):
    """Port of utils/mySmooth.m; smooths along axis 0."""
    if N <= 1:
        return x
    x = np.asarray(x, dtype=np.float64)
    if bctype == 'reflect':
        xf = np.concatenate([x[:N], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        xf = np.concatenate([np.zeros((N,) + x.shape[1:]), x], axis=0)
        trim = N
    else:
        xf, trim = x, 0
    k = causal_kernel(N)
    from scipy.signal import fftconvolve
    if xf.ndim == 1:
        out = fftconvolve(xf, k, mode='same')
    else:
        out = fftconvolve(xf, k.reshape((-1,) + (1,) * (xf.ndim - 1)), mode='same', axes=0)
    return out[trim:]


def bin_spikes(clusters, gocue, ntrials, edges):
    """alignSpikes.m + getSeq.m: aligned spike counts -> smoothed rates.

    Returns
        rates: (ntrials, nbins, nunits) smoothed firing rates in spikes/s
        meanfr: (nunits,) mean firing rate over the whole window and all trials,
                computed from the raw (unsmoothed) counts. This is the quantity
                thresholded by removeLowFRClusters.m (whose first PSTH condition,
                '(hit|miss|no)', covers every trial) and by the paper's
                "units with firing rates exceeding 1 Hz" criterion.
    """
    nbins = len(edges) - 1
    rates = np.zeros((ntrials, nbins, len(clusters)), dtype=np.float32)
    meanfr = np.zeros(len(clusters))
    trial_edges = np.arange(0.5, ntrials + 1.5, 1.0)  # trial numbers are 1-based
    window = edges[-1] - edges[0]
    for i, c in enumerate(clusters):
        tr = np.asarray(c['trial'], dtype=float).ravel()
        tt = np.asarray(c['trialtm'], dtype=float).ravel()
        n = min(tr.size, tt.size)
        tr, tt = tr[:n], tt[:n]
        ok = (tr >= 1) & (tr <= ntrials) & np.isfinite(tt)
        tr, tt = tr[ok], tt[ok]
        aligned = tt - gocue[(tr - 1).astype(int)]     # trialtm_aligned
        cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
        meanfr[i] = cnt.sum() / (ntrials * window)
        rates[:, :, i] = mysmooth(cnt.T / DT).T.astype(np.float32)
    return rates, meanfr


def interp_to_axis(src_t, src_y, taxis):
    """Linear interpolation like MATLAB interp1 (NaN outside the source range)."""
    src_t = np.asarray(src_t, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    good = np.isfinite(src_t)
    if good.sum() < 2:
        return np.full((taxis.size,) + src_y.shape[1:], np.nan)
    st = src_t[good]
    sy = src_y[good]
    order = np.argsort(st)
    st, sy = st[order], sy[order]
    inside = (taxis >= st[0]) & (taxis <= st[-1])
    if sy.ndim == 1:
        out = np.full(taxis.size, np.nan)
        out[inside] = np.interp(taxis[inside], st, sy)
    else:
        out = np.full((taxis.size, sy.shape[1]), np.nan)
        for j in range(sy.shape[1]):
            out[inside, j] = np.interp(taxis[inside], st, sy[:, j])
    return out


def speed_from_xy(xy):
    """Speed of a DLC feature, computed like findVelocity.m (gradient of position),
    but evaluated separately within each contiguous run of frames in which the
    feature is visible.

    The reference code fills the tongue's missing frames with a baseline position
    and sets its velocity to 0 where it is not visible. Here 'not visible' is its
    own output class, so the velocity is left undefined (NaN) on those frames and
    computed from neighbouring visible frames only; np.gradient uses one-sided
    differences at the ends of each run, so every visible frame gets a value and
    no NaN leaks into visible frames.
    """
    vis = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    spd = np.full(xy.shape[0], np.nan)
    if not vis.any():
        return spd
    idx = np.flatnonzero(vis)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [idx.size - 1]])
    for a, b in zip(starts, stops):
        seg = idx[a:b + 1]
        if seg.size == 1:
            spd[seg] = 0.0                      # single visible frame: no motion estimate
            continue
        vx = np.gradient(xy[seg, 0])
        vy = np.gradient(xy[seg, 1])
        spd[seg] = np.sqrt(vx ** 2 + vy ** 2)
    return spd


def discretize(values, visible, thresh):
    """0 = below threshold, 1 = at/above threshold, 2 = not visible / no video."""
    out = np.full(values.shape, 2, dtype=np.int64)
    v = visible & np.isfinite(values)
    out[v & (values < thresh)] = 0
    out[v & (values >= thresh)] = 1
    return out


# ----------------------------------------------------------------------------
# Per-session conversion
# ----------------------------------------------------------------------------
def region_from_loc(loc):
    """Map obj.ex.probe.loc strings to a brain-region name."""
    u = (loc or '').upper()
    if 'M1TJ' in u or 'TJM1' in u:
        return 'tjM1'
    return 'ALM'        # '', 'DUMMY', 'R ALM', 'L ALM' -> ALM (all reference probes are ALM recordings)


def process_session(entry, want_debug=False):
    """Convert one session. Returns a dict with neural/input/output and metadata."""
    t0 = time.time()
    d, anm, date, probes = entry
    sess_id = f'{anm}_{date}'
    s = Session(data_path(entry))
    timing = {}

    edges, taxis = time_axis()
    nbins = taxis.size
    ntrials = s.ntrials
    gocue = s.ev(ALIGN_EVENT)

    # ---------------- trial curation (reference condition strings) -------------
    stim = s.bpvec('stim/enable') if s.has_bp_field('stim') else np.zeros(ntrials)
    early = s.bpvec('early')
    hit = s.bpvec('hit')
    miss = s.bpvec('miss')
    no = s.bpvec('no')
    autowater = s.bpvec('autowater')
    for v in (stim, early, hit, miss, no, autowater):
        v[np.isnan(v)] = 0
    keep = (stim[:ntrials] == 0) & (early[:ntrials] == 0) & np.isfinite(gocue[:ntrials])
    # a trial must be one of hit / miss / ignore
    keep &= (hit[:ntrials] + miss[:ntrials] + no[:ntrials]) > 0
    trials = np.where(keep)[0]

    # ---------------- neural ---------------------------------------------------
    t1 = time.time()
    clusters, qualities, cl_region = [], [], []
    for p in probes:
        loc_list = s.probe_locs()
        loc = loc_list[p - 1] if len(loc_list) >= p else ''
        reg = region_from_loc(loc)
        for c in s.clusters(p - 1):
            q = c['quality'].strip().lower()
            if q in BAD_QUALITY:
                continue                      # findClusters.m with quality = 'all'
            clusters.append(c)
            qualities.append(q)
            cl_region.append(reg)
    rates, meanfr = bin_spikes(clusters, gocue, ntrials, edges)  # (ntrials, nbins, nunits)
    timing['neural_bin'] = time.time() - t1

    # removeLowFRClusters.m: mean rate must exceed lowFR (1 spk/s)
    keep_units = meanfr > LOW_FR
    rates = rates[:, :, keep_units]
    cl_region = [r for r, k in zip(cl_region, keep_units) if k]
    qualities = [q for q, k in zip(qualities, keep_units) if k]
    nunits = rates.shape[2]

    # Trials in which no curated unit fired a single spike in the whole 5 s window
    # have no ephys coverage (in 2 JEB24 sessions the recording stopped before the
    # behavioural session ended); drop them.
    if nunits > 0:
        has_spikes = rates[trials].sum(axis=(1, 2)) > 0
        n_nospk = int((~has_spikes).sum())
        trials = trials[has_spikes]
    else:
        n_nospk = 0

    # ---------------- behavior (per-trial outputs) -----------------------------
    lickL, lickR = s.ev_cell('lickL'), s.ev_cell('lickR')
    lick_dir = np.full(ntrials, 2, dtype=np.int64)          # 2 = none
    for i in range(ntrials):
        tl = np.concatenate([lickL[i] - gocue[i], lickR[i] - gocue[i]])
        side = np.concatenate([np.zeros(lickL[i].size), np.ones(lickR[i].size)])
        m = tl > 0                                           # firstLickTime.m keeps post-go-cue licks
        if m.any():
            lick_dir[i] = int(side[m][np.argmin(tl[m])])     # 0 = left, 1 = right
    context = np.where(autowater[:ntrials] > 0, 0, 1).astype(np.int64)   # 0 = WC, 1 = DR
    outcome = np.full(ntrials, 2, dtype=np.int64)            # 2 = ignore
    outcome[miss[:ntrials] > 0] = 0                          # 0 = incorrect
    outcome[hit[:ntrials] > 0] = 1                           # 1 = correct

    # ---------------- video-derived outputs ------------------------------------
    t1 = time.time()
    vidshift = s.vidshift()
    fn0, fn1 = s.feat_names(0), s.feat_names(1)
    i_tongue = fn0.index('tongue')
    i_paws = [fn1.index(f) for f in ('top_paw', 'bottom_paw') if f in fn1]
    me_trials, _ = load_motion_energy(me_path(entry))
    nvid = min(s.ntraj_trials(0), s.ntraj_trials(1))

    tongue_spd = np.full((ntrials, nbins), np.nan)
    tongue_vis = np.zeros((ntrials, nbins), dtype=bool)
    paw_spd = np.full((ntrials, nbins), np.nan)
    paw_vis = np.zeros((ntrials, nbins), dtype=bool)
    me_binned = np.full((ntrials, nbins), np.nan)
    has_video = np.zeros(ntrials, dtype=bool)

    bin_edges = edges
    for i in trials:
        if i >= nvid:
            continue
        ts0, ft0, nd0 = s.traj_trial(0, i)
        ts1, ft1, nd1 = s.traj_trial(1, i)
        if ts0.size == 0 or ft0.size == 0 or np.all(~np.isfinite(ft0)) or not np.isfinite(nd0):
            continue                                        # findPosition.m skips these trials
        # video clock -> time from the alignment event
        vt0 = ft0 - vidshift - gocue[i]
        dtf = np.median(np.diff(ft0)) if ft0.size > 1 else 1 / 400.0
        has_video[i] = True
        idx = np.digitize(vt0, bin_edges) - 1               # bin index for each frame
        inb = (idx >= 0) & (idx < nbins)

        # --- tongue (side cam) ---
        xy = ts0[:, :2, i_tongue]
        vis = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        spd = speed_from_xy(xy) / dtf                        # pixels / s
        _accumulate(idx, inb, spd, vis, tongue_spd, tongue_vis, i, nbins)

        # --- paws (bottom cam) ---
        if ts1.size and ft1.size and np.isfinite(nd1):
            vt1 = ft1 - vidshift - gocue[i]
            idx1 = np.digitize(vt1, bin_edges) - 1
            inb1 = (idx1 >= 0) & (idx1 < nbins)
            dtf1 = np.median(np.diff(ft1)) if ft1.size > 1 else 1 / 400.0
            spds, viss = [], []
            for j in i_paws:
                pxy = ts1[:, :2, j]
                spds.append(speed_from_xy(pxy) / dtf1)
                viss.append(np.isfinite(pxy[:, 0]) & np.isfinite(pxy[:, 1]))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                spd_p = np.nanmean(np.vstack(spds), axis=0)
            vis_p = np.any(np.vstack(viss), axis=0)
            _accumulate(idx1, inb1, spd_p, vis_p, paw_spd, paw_vis, i, nbins)

        # --- motion energy (loadMotionEnergy.m: interp to taxis, fill nearest) ---
        if i < len(me_trials):
            mev = me_trials[i]
            n = min(mev.size, vt0.size)
            if n > 1:
                y = interp_to_axis(vt0[:n], mev[:n], taxis)
                y = _fill_nearest(y)
                me_binned[i] = y

    timing['video'] = time.time() - t1

    # ---------------- discretization (per-session 50th percentile) -------------
    sel = np.zeros(ntrials, dtype=bool)
    sel[trials] = True
    m_t = sel[:, None] & tongue_vis & np.isfinite(tongue_spd)
    m_p = sel[:, None] & paw_vis & np.isfinite(paw_spd)
    m_m = sel[:, None] & np.isfinite(me_binned)
    thr_t = np.nanpercentile(tongue_spd[m_t], 50) if m_t.any() else np.nan
    thr_p = np.nanpercentile(paw_spd[m_p], 50) if m_p.any() else np.nan
    thr_m = np.nanpercentile(me_binned[m_m], 50) if m_m.any() else np.nan

    tongue_cls = discretize(tongue_spd, tongue_vis, thr_t)
    paw_cls = discretize(paw_spd, paw_vis, thr_p)
    me_cls = discretize(me_binned, has_video[:, None] & np.isfinite(me_binned), thr_m)

    # ---------------- assemble -------------------------------------------------
    neural, inp, outp = [], [], []
    tin = taxis.astype(np.float32).reshape(1, -1)
    for i in trials:
        neural.append(np.ascontiguousarray(rates[i].T))           # (nunits, nbins)
        inp.append(tin.copy())
        o = np.empty((6, nbins), dtype=np.int64)
        o[0] = lick_dir[i]
        o[1] = context[i]
        o[2] = outcome[i]
        o[3] = tongue_cls[i]
        o[4] = paw_cls[i]
        o[5] = me_cls[i]
        outp.append(o)

    res = dict(
        session_id=sess_id, animal=anm, date=date, group=('fixed' if entry in FIXED else 'randomized'),
        neural=neural, input=inp, output=outp,
        regions=cl_region, qualities=qualities, nunits=nunits,
        ntrials_total=ntrials, ntrials_kept=len(trials), trials=trials,
        thresholds=dict(tongue=float(thr_t), paw=float(thr_p), me=float(thr_m)),
        vidshift=vidshift, timing=timing, elapsed=time.time() - t0,
        n_novideo=int(np.sum(~has_video[trials])), n_nospike=n_nospk,
    )
    if want_debug:
        res['debug'] = dict(taxis=taxis, rates=rates, tongue_spd=tongue_spd, tongue_vis=tongue_vis,
                            paw_spd=paw_spd, paw_vis=paw_vis, me=me_binned, has_video=has_video,
                            lick_dir=lick_dir, context=context, outcome=outcome, gocue=gocue,
                            lickL=lickL, lickR=lickR, clusters_kept=int(nunits))
    s.close()
    return res


def _accumulate(idx, inb, spd, vis, spd_out, vis_out, trial, nbins):
    """Average per-frame speed within each time bin; a bin is visible if any frame is."""
    good = inb & vis & np.isfinite(spd)
    if good.any():
        sums = np.bincount(idx[good], weights=spd[good], minlength=nbins)
        cnts = np.bincount(idx[good], minlength=nbins)
        with np.errstate(invalid='ignore', divide='ignore'):
            vals = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        spd_out[trial] = vals
        vis_out[trial] = cnts > 0


def _fill_nearest(y):
    """MATLAB fillmissing(y,'nearest') for a 1-D array."""
    y = np.asarray(y, dtype=float).copy()
    good = np.isfinite(y)
    if not good.any() or good.all():
        return y
    gi = np.flatnonzero(good)
    bi = np.flatnonzero(~good)
    pos = np.searchsorted(gi, bi)
    left = np.clip(pos - 1, 0, gi.size - 1)
    right = np.clip(pos, 0, gi.size - 1)
    take_left = np.abs(bi - gi[left]) <= np.abs(gi[right] - bi)
    nearest = np.where(take_left, gi[left], gi[right])
    y[bi] = y[nearest]
    return y


# ----------------------------------------------------------------------------
# Visualisation of every processing step (--show-processing)
# ----------------------------------------------------------------------------
def plot_processing(res, outfn):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dbg = res['debug']
    taxis = dbg['taxis']
    trials = res['trials']
    # pick a trial with a lick and visible tongue
    pick = None
    for i in trials:
        if dbg['lick_dir'][i] != 2 and dbg['tongue_vis'][i].any():
            pick = i
            break
    if pick is None:
        pick = trials[0]

    fig, ax = plt.subplots(7, 2, figsize=(18, 22))

    # 1. raster + binned rates for the example trial
    rates = dbg['rates'][pick]
    ax[0, 0].imshow(rates.T, aspect='auto', origin='lower',
                    extent=[taxis[0], taxis[-1], 0, rates.shape[1]], cmap='magma')
    ax[0, 0].set_title(f"{res['session_id']} trial {pick}: binned+smoothed rates (spk/s)")
    ax[0, 0].set_xlabel('time from go cue (s)'); ax[0, 0].set_ylabel('unit')
    ax[0, 0].axvline(0, color='c', lw=1)

    # 2. population mean rate aligned to go cue, all kept trials
    m = dbg['rates'][trials].mean(axis=(0, 2))
    ax[0, 1].plot(taxis, m)
    ax[0, 1].axvline(0, color='k', ls='--')
    ax[0, 1].set_title('population mean firing rate (all kept trials)')
    ax[0, 1].set_xlabel('time from go cue (s)'); ax[0, 1].set_ylabel('spk/s')

    # 3. lick raster vs. alignment (checks temporal alignment of behaviour)
    for k, i in enumerate(trials[:150]):
        l = dbg['lickL'][i] - dbg['gocue'][i]
        r = dbg['lickR'][i] - dbg['gocue'][i]
        ax[1, 0].plot(l, np.full(l.size, k), 'r.', ms=2)
        ax[1, 0].plot(r, np.full(r.size, k), 'b.', ms=2)
    ax[1, 0].axvline(0, color='k'); ax[1, 0].set_xlim(taxis[0], taxis[-1])
    ax[1, 0].set_title('lick raster (red=left, blue=right), aligned to go cue')
    ax[1, 0].set_xlabel('time from go cue (s)'); ax[1, 0].set_ylabel('trial')

    # 4. tongue visibility raster (should start right after the go cue)
    vis = dbg['tongue_vis'][trials[:150]]
    ax[1, 1].imshow(vis, aspect='auto', origin='lower', extent=[taxis[0], taxis[-1], 0, vis.shape[0]])
    ax[1, 1].axvline(0, color='c')
    ax[1, 1].set_title('tongue visible (DLC non-NaN)')
    ax[1, 1].set_xlabel('time from go cue (s)')

    # 5. tongue speed + discretization for the example trial
    spd = dbg['tongue_spd'][pick]
    cls = res['output'][list(trials).index(pick)][3]
    ax[2, 0].plot(taxis, spd, '.-')
    ax[2, 0].axhline(res['thresholds']['tongue'], color='r', ls='--', label='session median')
    ax[2, 0].set_title('tongue speed (px/s), example trial'); ax[2, 0].legend()
    ax[2, 1].step(taxis, cls, where='mid')
    ax[2, 1].set_yticks([0, 1, 2]); ax[2, 1].set_yticklabels(['<50%', '>=50%', 'not visible'])
    ax[2, 1].set_title('discretized tongue velocity')

    # 6. paw speed + discretization
    spd = dbg['paw_spd'][pick]
    cls = res['output'][list(trials).index(pick)][4]
    ax[3, 0].plot(taxis, spd, '.-')
    ax[3, 0].axhline(res['thresholds']['paw'], color='r', ls='--')
    ax[3, 0].set_title('paw speed (px/s), example trial')
    ax[3, 1].step(taxis, cls, where='mid')
    ax[3, 1].set_yticks([0, 1, 2]); ax[3, 1].set_yticklabels(['<50%', '>=50%', 'not visible'])
    ax[3, 1].set_title('discretized paw velocity')

    # 7. motion energy + discretization
    me = dbg['me'][pick]
    cls = res['output'][list(trials).index(pick)][5]
    ax[4, 0].plot(taxis, me, '.-')
    ax[4, 0].axhline(res['thresholds']['me'], color='r', ls='--')
    ax[4, 0].set_title('motion energy, example trial')
    ax[4, 1].step(taxis, cls, where='mid')
    ax[4, 1].set_yticks([0, 1, 2]); ax[4, 1].set_yticklabels(['<50%', '>=50%', 'no video'])
    ax[4, 1].set_title('discretized motion energy')

    # 8. trial-averaged motion energy split by context
    ctx = dbg['context'][trials]
    for c, lab in [(0, 'WC'), (1, 'DR')]:
        if (ctx == c).any():
            ax[5, 0].plot(taxis, np.nanmean(dbg['me'][trials[ctx == c]], axis=0), label=lab)
    ax[5, 0].axvline(0, color='k', ls='--'); ax[5, 0].legend()
    ax[5, 0].set_title('mean motion energy by context')

    # 9. distribution of the discretized classes
    allout = np.stack([o for o in res['output']], axis=0)  # (ntrials, 6, nbins)
    names = ['lick_direction', 'context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy']
    fr = [[np.mean(allout[:, k] == v) for v in range(3)] for k in range(6)]
    ax[5, 1].imshow(np.array(fr), aspect='auto', cmap='viridis', vmin=0, vmax=1)
    for k in range(6):
        for v in range(3):
            ax[5, 1].text(v, k, f'{fr[k][v]:.2f}', ha='center', va='center', color='w')
    ax[5, 1].set_yticks(range(6)); ax[5, 1].set_yticklabels(names)
    ax[5, 1].set_xticks(range(3)); ax[5, 1].set_xticklabels(['class 0', 'class 1', 'class 2'])
    ax[5, 1].set_title('fraction of timepoints per class')

    # 10. tongue-visibility-triggered check: mean rate for right vs left licks
    ld = dbg['lick_dir'][trials]
    for c, lab, col in [(0, 'left lick', 'r'), (1, 'right lick', 'b'), (2, 'no lick', 'k')]:
        if (ld == c).any():
            ax[6, 0].plot(taxis, dbg['rates'][trials[ld == c]].mean(axis=(0, 2)), col, label=lab)
    ax[6, 0].axvline(0, color='k', ls='--'); ax[6, 0].legend()
    ax[6, 0].set_title('population rate by lick direction')

    # 11. input: time from go cue
    ax[6, 1].plot(taxis, res['input'][0][0])
    ax[6, 1].set_title('decoder input: time from go cue (s)')
    ax[6, 1].set_xlabel('bin centre (s)')

    fig.suptitle(f"Processing steps: {res['session_id']} "
                 f"({res['nunits']} units, {res['ntrials_kept']}/{res['ntrials_total']} trials)")
    fig.tight_layout()
    fig.savefig(outfn, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def _worker(args):
    entry, want_debug = args
    try:
        return process_session(entry, want_debug)
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        return dict(session_id=f'{entry[1]}_{entry[2]}', error=str(exc))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save processing plots for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=8)
    args = ap.parse_args()

    sessions = SESSIONS
    if args.sample:
        sessions = [FIXED[0], RANDOM[0]]
    print(f'Converting {len(sessions)} sessions -> {args.outfile}', flush=True)

    t0 = time.time()
    want_debug = [args.show_processing and i < 2 for i in range(len(sessions))]
    nproc = min(args.nproc, len(sessions))
    if nproc > 1:
        with Pool(nproc) as pool:
            results = pool.map(_worker, list(zip(sessions, want_debug)))
    else:
        results = [_worker(a) for a in zip(sessions, want_debug)]
    print(f'All sessions processed in {time.time() - t0:.1f}s', flush=True)

    data = dict(neural=[], input=[], output=[], subjects=[], subject_idx=[],
                brain_regions=['ALM', 'tjM1'], brain_region_idx=[],
                input_names=['time_from_go_cue'],
                output_names=['lick_direction', 'context', 'outcome',
                              'tongue_velocity', 'paw_velocity', 'motion_energy'],
                output_values=[['left', 'right', 'none'],
                               ['WC', 'DR'],
                               ['incorrect', 'correct', 'ignore'],
                               ['below_median', 'above_median', 'not_visible'],
                               ['below_median', 'above_median', 'not_visible'],
                               ['below_median', 'above_median', 'no_video']])
    session_info = []
    subjects = []
    for r in results:
        if 'error' in r:
            print(f"  SKIP {r['session_id']}: {r['error']}")
            continue
        if r['nunits'] < MIN_UNITS:
            print(f"  SKIP {r['session_id']}: only {r['nunits']} units (< {MIN_UNITS})")
            continue
        if r['ntrials_kept'] < 2:
            print(f"  SKIP {r['session_id']}: only {r['ntrials_kept']} trials")
            continue
        if r['animal'] not in subjects:
            subjects.append(r['animal'])
        data['neural'].append(r['neural'])
        data['input'].append(r['input'])
        data['output'].append(r['output'])
        data['subject_idx'].append(subjects.index(r['animal']))
        data['brain_region_idx'].append(
            np.array([data['brain_regions'].index(x) for x in r['regions']], dtype=np.int64))
        session_info.append(dict(session_id=r['session_id'], animal=r['animal'], date=r['date'],
                                 task=r['group'], nunits=r['nunits'],
                                 ntrials=r['ntrials_kept'], ntrials_raw=r['ntrials_total'],
                                 thresholds=r['thresholds'], vidshift=r['vidshift'],
                                 trials_without_video=r['n_novideo'],
                                 trials_dropped_no_ephys=r['n_nospike'],
                                 regions=sorted(set(r['regions']))))
        print(f"  {r['session_id']:<18} units={r['nunits']:<4} trials={r['ntrials_kept']}/{r['ntrials_total']}"
              f"  novideo={r['n_novideo']:<3} t={r['elapsed']:.1f}s", flush=True)

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    edges, taxis = time_axis()
    data['metadata'] = dict(
        task_description=(
            'Head-fixed mice performed a delayed-response (DR) directional licking task with '
            'interleaved blocks of an autowater/water-cued (WC) task, while activity was recorded '
            'with silicon probes in anterior lateral motor cortex (ALM). Decoder outputs are the '
            'lick direction of the first lick after the go cue (left/right/none), the behavioral '
            'context (WC/DR), the trial outcome (incorrect/correct/ignore), and three discretized '
            'movement signals (tongue velocity, paw velocity, video motion energy; each split at '
            'the session median with a separate class for timepoints where the feature is not '
            'visible / has no video). The decoder input is the time from go-cue onset. '
            'Data from Hasnain, Birnbaum et al., Nature Neuroscience 2024.'),
        time_bin_size=DT * 1000.0,
        temporal_alignment_event='go cue onset (water drop onset on water-cued trials); obj.bp.ev.goCue',
        off_start=TMIN, off_end=TMAX,
        smoothing='causal Gaussian kernel, 15 bins (450 ms) wide, reflect boundary (utils/mySmooth.m)',
        neural_units='firing rate, spikes/s',
        neuron_curation=('cluster quality not in {garbage, noisy, real?} (findClusters.m) and mean '
                         'firing rate over the -2.5..2.5 s window > 1 spk/s (removeLowFRClusters.m, '
                         'paper: units with firing rates exceeding 1 Hz)'),
        trial_curation=('excluded photoinactivation trials (bp.stim.enable) and early-lick trials '
                        '(bp.early), as in every reference condition string, and trials with no '
                        'spikes at all (ephys recording ended before the behavioural session); '
                        'ignore trials retained because ignore/none are required output classes'),
        session_curation=f'sessions kept if >= {MIN_UNITS} curated units (paper criterion)',
        time_axis=taxis.astype(np.float32),
        source='Zenodo 10.5281/zenodo.13941415; code https://github.com/economolab',
        session_info=session_info,
    )

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    nses = len(data['neural'])
    ntr = sum(len(x) for x in data['neural'])
    nun = sum(len(x) for x in data['brain_region_idx'])
    print(f'\nSaved {args.outfile}: {nses} sessions, {ntr} trials, {nun} neurons, '
          f'{len(subjects)} subjects')
    print(f'Total time {time.time() - t0:.1f}s')

    if args.show_processing:
        for r in results[:2]:
            if 'debug' in r:
                fn = f"processing_{r['session_id']}.png"
                plot_processing(r, fn)
                print('wrote', fn)


if __name__ == '__main__':
    main()
