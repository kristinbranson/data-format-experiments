#!/usr/bin/env python3
"""
Convert the ALM electrophysiology + behavior dataset of

    Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the
    behaving mouse", Nature Neuroscience 2024

into the decoder-ready pickle format described in the task specification.

The processing follows the authors' own MATLAB pipeline
(/app/code/DataLoadingScripts + /app/code/funcs/kinematics), in particular:

  loadObjs.m / WorkingWithDataObjs.m  -> which sessions / probes to load
  findClusters.m                      -> cluster quality curation
  alignSpikes.m                       -> spikes aligned to params.alignEvent ('goCue')
  getSeq.m                            -> binning (dt = 10 ms, [-2.5, 2.5] s) and
                                         causal gaussian smoothing (mySmooth.m)
  removeLowFRClusters.m               -> drop units with mean FR <= 1 Hz
  findPosition.m / findVelocity.m     -> DLC kinematics, video/ephys alignment
  loadMotionEnergy.m                  -> motion energy, video/ephys alignment
  findVideoOffset.m                   -> video <-> spikeGLX clock offset

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import h5py
import scipy.io as sio
from scipy import stats
from scipy.signal import lfilter

# --------------------------------------------------------------------------- #
#  Parameters (mirrors params in WorkingWithDataObjs.m / getDefaultParams.m)
# --------------------------------------------------------------------------- #
ALIGN_EVENT = 'goCue'      # params.alignEvent
TMIN, TMAX = -2.5, 2.5     # params.tmin, params.tmax  (s, relative to go cue)
DT = 0.01                  # params.dt = 1/100 s  -> 10 ms bins
SMOOTH_N = 15              # params.smooth  (gaussian window length, in bins)
SMOOTH_BC = 'reflect'      # params.bctype
LOW_FR = 1.0               # params.lowFR (Hz); paper: "All units with firing
                           # rates exceeding 1 Hz were included"
ADVANCE_MOVEMENT = 0.0     # params.advance_movement

# findClusters.m, params.quality = {'all'}: everything except these labels
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
# WorkingWithDataObjs.m: "excellent, great, good treated as single units"
SINGLE_UNIT_QUALITY = {'excellent', 'great', 'good'}

DATA_FIXED = '/app/data/Ephys_Behavior'
DATA_RAND = '/app/data/RandomizedDelay_Ephys_Behavior'

# Sessions + probes, transcribed verbatim from the reference session lists in
# /app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m
# (commented-out sessions in those files are excluded here as well).
SESSIONS = [
    # (animal, date, probes (1-based), data directory, task)
    ('EKH1',  '2021-08-07', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ('EKH3',  '2021-08-11', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB6',  '2021-04-18', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB7',  '2021-04-29', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB7',  '2021-04-30', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JGR2',  '2021-11-16', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JGR2',  '2021-11-17', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JGR3',  '2021-11-18', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB13', '2022-09-13', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB13', '2022-09-14', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB13', '2022-09-21', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB13', '2022-09-24', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB13', '2022-09-25', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB14', '2022-08-22', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB14', '2022-08-23', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB14', '2022-08-24', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB14', '2022-08-25', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB15', '2022-07-26', [1, 2], DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB15', '2022-07-27', [1, 2], DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB15', '2022-07-28', [1, 2], DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB15', '2022-07-29', [2],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB19', '2023-04-18', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB19', '2023-04-19', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB19', '2023-04-20', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB19', '2023-04-21', [1],    DATA_FIXED, 'DR/WC fixed delay'),
    ('JEB11', '2022-05-10', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB11', '2022-05-11', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB12', '2022-05-12', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB12', '2022-05-13', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB23', '2023-10-10', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB23', '2023-10-11', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB23', '2023-10-12', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB23', '2023-10-13', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB23', '2023-10-18', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB23', '2023-10-19', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB23', '2023-10-21', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB24', '2023-10-23', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB24', '2023-10-24', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB24', '2023-10-25', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB24', '2023-10-26', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB24', '2023-10-27', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB24', '2023-10-31', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB24', '2023-11-02', [1],    DATA_RAND,  'DR randomized delay'),
    ('JEB24', '2023-11-03', [1],    DATA_RAND,  'DR randomized delay'),
]

# DLC features used for the discretised kinematic outputs.
# view 0 = side camera (cam0), view 1 = bottom camera (cam1).
TONGUE_VIEW, TONGUE_FEAT = 0, 'tongue'
PAW_VIEW, PAW_FEAT = 1, 'top_paw'   # paws are only tracked from the bottom cam;
                                    # 'top_paw_*_view2' is the paw feature used
                                    # in the reference figure code (Figure1e.m)

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
INPUT_NAMES = ['time_from_go_cue_s']


# --------------------------------------------------------------------------- #
#  MATLAB file readers (data objects are saved as both v7.3/HDF5 and v7/MAT5)
# --------------------------------------------------------------------------- #
def _chars(a):
    """MATLAB char array -> python str"""
    a = np.array(a).ravel()
    if a.dtype.kind in 'US':
        return ''.join(str(x) for x in a)
    return ''.join(chr(int(c)) for c in a)


def _is_hdf5(path):
    with open(path, 'rb') as fh:
        return b'MATLAB 7.3' in fh.read(128)


class _H5Session:
    """reader for data_structure_*.mat saved as MATLAB v7.3 (HDF5)"""
    fmt = 'v7.3'

    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, 'r')
        self.o = self.f['obj']

    def close(self):
        self.f.close()

    # -- bpod ------------------------------------------------------------
    def bp(self, name):
        g = self.o['bp']
        for part in name.split('.'):
            g = g[part]
        return np.array(g).ravel().astype(float)

    def has_bp(self, name):
        g = self.o['bp']
        for part in name.split('.'):
            if part not in g:
                return False
            g = g[part]
        return True

    @property
    def Ntrials(self):
        return int(np.array(self.o['bp']['Ntrials']).ravel()[0])

    def licks(self, side):
        refs = np.array(self.o['bp']['ev']['lick' + side]).ravel()
        out = []
        for r in refs:
            a = np.array(self.f[r])
            out.append(np.asarray(a, float).ravel() if a.dtype == np.float64
                       else np.zeros(0))
        return out

    # -- clusters --------------------------------------------------------
    @property
    def nprobes(self):
        return self.o['clu'].shape[0] if 'clu' in self.o else 0

    def _clu(self, prb):
        g = self.f[self.o['clu'][prb - 1, 0]]
        return g if isinstance(g, h5py.Group) else None

    def qualities(self, prb):
        g = self._clu(prb)
        if g is None:
            return []
        q = g['quality']
        out = []
        for i in range(q.shape[0]):
            try:
                out.append(_chars(self.f[q[i, 0]]).replace('\x00', '').strip())
            except Exception:
                out.append('')
        return out

    def spikes(self, prb, i):
        g = self._clu(prb)
        return (np.array(self.f[g['trialtm'][i, 0]]).ravel().astype(np.float64),
                np.array(self.f[g['trial'][i, 0]]).ravel().astype(np.int64))

    # -- video -----------------------------------------------------------
    def featnames(self, view):
        g = self.f[self.o['traj'][view, 0]]
        return [_chars(self.f[r]) for r in np.array(self.f[g['featNames'][0, 0]]).ravel()]

    def traj_xy(self, view, trix, featix):
        """(x, y) DLC traces for one feature, plus frame times (video clock)."""
        g = self.f[self.o['traj'][view, 0]]
        try:
            ds = self.f[g['ts'][trix, 0]]
            if ds.ndim != 3:
                return None, None
            xy = np.array(ds[featix, 0:2, :], dtype=float)   # (2, nframes)
        except Exception:
            return None, None
        ft = None
        if 'frameTimes' in g:
            d = np.array(self.f[g['frameTimes'][trix, 0]])
            if d.dtype == np.float64 and d.size:
                ft = d.ravel().astype(float)
        return xy, ft

    def frame_times(self, view, trix):
        g = self.f[self.o['traj'][view, 0]]
        if 'frameTimes' not in g:
            return None
        d = np.array(self.f[g['frameTimes'][trix, 0]])
        if d.dtype != np.float64 or d.size == 0:
            return None
        return d.ravel().astype(float)

    def ndropped(self, view, trix):
        g = self.f[self.o['traj'][view, 0]]
        if 'NdroppedFrames' not in g:
            return np.nan
        d = np.array(self.f[g['NdroppedFrames'][trix, 0]]).ravel()
        return float(d[0]) if (d.size and d.dtype == np.float64) else np.nan

    # -- misc ------------------------------------------------------------
    def sglx_fs(self):
        return float(np.array(self.o['sglx']['fs']).ravel()[0])

    def sglx_bitstart(self):
        return np.array(self.o['sglx']['bitcode']['bitstart']).ravel().astype(float)

    def probe_loc(self, prb):
        try:
            return _chars(self.f[self.o['ex']['probe']['loc'][prb - 1, 0]]
                          ).replace('\x00', '').strip()
        except Exception:
            return ''


class _V7Session:
    """reader for data_structure_*.mat saved as MATLAB v7 (MAT5)"""
    fmt = 'v7'

    def __init__(self, path):
        self.path = path
        d = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
        self.o = d['obj'][0, 0]
        self._bp = self.o.bp[0, 0]

    def close(self):
        self.o = None
        self._bp = None

    def bp(self, name):
        g, parts = self._bp, name.split('.')
        for p in parts[:-1]:
            g = getattr(g, p)[0, 0]
        return np.array(getattr(g, parts[-1])).ravel().astype(float)

    def has_bp(self, name):
        g, parts = self._bp, name.split('.')
        for p in parts[:-1]:
            if not hasattr(g, p):
                return False
            g = getattr(g, p)[0, 0]
        return hasattr(g, parts[-1])

    @property
    def Ntrials(self):
        return int(np.array(self._bp.Ntrials).ravel()[0])

    def licks(self, side):
        c = getattr(self._bp.ev[0, 0], 'lick' + side)
        return [np.asarray(x, float).ravel() for x in c.ravel()]

    @property
    def nprobes(self):
        return self.o.clu.size if hasattr(self.o, 'clu') else 0

    def _clu(self, prb):
        c = self.o.clu.ravel()[prb - 1]
        return c.ravel() if c.size else None

    def qualities(self, prb):
        c = self._clu(prb)
        return [] if c is None else [_chars(x.quality).replace('\x00', '').strip()
                                     for x in c]

    def spikes(self, prb, i):
        c = self._clu(prb)
        return (np.asarray(c[i].trialtm, float).ravel(),
                np.asarray(c[i].trial, np.int64).ravel())

    def featnames(self, view):
        t = self.o.traj.ravel()[view].ravel()
        return [_chars(x) for x in np.array(t[0].featNames).ravel()]

    def traj_xy(self, view, trix, featix):
        t = self.o.traj.ravel()[view].ravel()[trix]
        ts = np.asarray(t.ts, float)
        if ts.ndim != 3:
            return None, None
        xy = ts[:, 0:2, featix].T                      # (2, nframes)
        ft = np.asarray(t.frameTimes, float).ravel() if hasattr(t, 'frameTimes') else None
        if ft is not None and ft.size == 0:
            ft = None
        return xy, ft

    def frame_times(self, view, trix):
        t = self.o.traj.ravel()[view].ravel()[trix]
        if not hasattr(t, 'frameTimes'):
            return None
        ft = np.asarray(t.frameTimes, float).ravel()
        return None if ft.size == 0 else ft

    def ndropped(self, view, trix):
        t = self.o.traj.ravel()[view].ravel()[trix]
        if not hasattr(t, 'NdroppedFrames'):
            return np.nan
        a = np.asarray(t.NdroppedFrames, float).ravel()
        return float(a[0]) if a.size else np.nan

    def sglx_fs(self):
        return float(np.array(self.o.sglx[0, 0].fs).ravel()[0])

    def sglx_bitstart(self):
        return np.array(self.o.sglx[0, 0].bitcode[0, 0].bitstart).ravel().astype(float)

    def probe_loc(self, prb):
        try:
            return _chars(self.o.ex[0, 0].probe.ravel()[prb - 1].loc
                          ).replace('\x00', '').strip()
        except Exception:
            return ''


def open_session(path):
    return _H5Session(path) if _is_hdf5(path) else _V7Session(path)


def load_motion_energy(path):
    """motionEnergy_<anm>_<date>.mat -> list of per-trial 400 Hz traces.

    Mirrors the layouts handled by loadMotionEnergy.m ('if isstruct(me.data)').
    All motion-energy files in this dataset are MATLAB v7 (MAT5).
    """
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
    me = d['me']
    if isinstance(me, np.ndarray) and me.dtype == object and not hasattr(me[0, 0], '_fieldnames'):
        cells = me            # me is itself the cell array of trials
    else:
        me = me[0, 0]
        cells = me.data
        if hasattr(cells[0, 0], '_fieldnames'):   # me.data is a struct -> me.data.data
            cells = cells[0, 0].data
    return [np.asarray(x, float).ravel() for x in cells.ravel()]


# --------------------------------------------------------------------------- #
#  Signal processing helpers (ports of the reference MATLAB utilities)
# --------------------------------------------------------------------------- #
def gausswin(N, alpha=2.5):
    """MATLAB gausswin(N, alpha)"""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def _causal_kernel(N=SMOOTH_N):
    """mySmooth.m: gaussian window, first floor(N/2) taps zeroed, normalised."""
    k = gausswin(N)
    k[:N // 2] = 0.0
    k = k / k.sum()
    # conv(.,'same') with a kernel whose first N//2 taps are zero is the causal
    # FIR filter  y[i] = sum_j k[N//2 + j] * x[i - j].
    return k[N // 2:]


_FIR = _causal_kernel()


def my_smooth(x, N=SMOOTH_N, bctype=SMOOTH_BC):
    """mySmooth(x, N, bctype) - operates on the first (time) dimension."""
    if N <= 1:
        return x
    if bctype == 'reflect':
        xp = np.concatenate([x[:N], x], axis=0)
        trim = N
    elif bctype == 'zeropad':
        xp = np.concatenate([np.zeros((N,) + x.shape[1:], x.dtype), x], axis=0)
        trim = N
    else:
        xp, trim = x, 0
    y = lfilter(_FIR, [1.0], xp, axis=0)
    return y[trim:]


def interp_matlab(xq, xp, fp):
    """interp1(xp, fp, xq) with MATLAB semantics: NaN outside the data range and
    NaN propagated from NaN samples."""
    return np.interp(xq, xp, fp, left=np.nan, right=np.nan)


def nan_gradient(x):
    """np.gradient-like first derivative that tolerates NaN gaps.

    Central difference where both neighbours are finite, one-sided at the edges
    of a valid run, 0 for isolated samples. For gap-free signals this is exactly
    MATLAB's gradient() (unit spacing), which is what findVelocity.m uses.
    """
    x = np.asarray(x, float)
    n = x.size
    fwd = np.full(n, np.nan)
    bwd = np.full(n, np.nan)
    fwd[:-1] = x[1:] - x[:-1]
    bwd[1:] = x[1:] - x[:-1]
    both = np.isfinite(fwd) & np.isfinite(bwd)
    g = np.where(both, (fwd + bwd) / 2.0,
                 np.where(np.isfinite(fwd), fwd,
                          np.where(np.isfinite(bwd), bwd, 0.0)))
    g[~np.isfinite(x)] = np.nan
    return g


# --------------------------------------------------------------------------- #
#  Per-session conversion
# --------------------------------------------------------------------------- #
def time_axis():
    """getSeq.m: edges = tmin:dt:tmax, obj.time = edges + dt/2 (last dropped)"""
    edges = np.arange(TMIN, TMAX + DT / 2, DT)
    edges = edges[edges <= TMAX + 1e-9]
    t = edges + DT / 2
    return edges, t[:-1]


EDGES, TAXIS = time_axis()
NT = TAXIS.size


def bin_spikes(obj, probes, keep_clu, gocue, ntrials):
    """Binned, smoothed single-trial firing rates.

    Port of alignSpikes.m + getSeq.m: spike times are expressed relative to the
    alignment event and histogrammed into dt bins, divided by dt to give
    spikes/s, then smoothed with the causal gaussian kernel.

    Returns (ntrials, NT, nunits) float32 array of firing rates.
    """
    nunits = sum(len(k) for k in keep_clu.values())
    out = np.zeros((NT, nunits, ntrials), dtype=np.float32)
    col = 0
    for prb in probes:
        for i in keep_clu[prb]:
            tt, tr = obj.spikes(prb, i)
            ok = (tr >= 1) & (tr <= ntrials)
            tt, tr = tt[ok], tr[ok]
            al = tt - gocue[tr - 1]                      # trialtm_aligned
            b = np.floor((al - TMIN) / DT).astype(np.int64)
            m = (b >= 0) & (b < NT)
            if m.any():
                idx = (tr[m] - 1) * NT + b[m]
                cnt = np.bincount(idx, minlength=ntrials * NT)
                out[:, col, :] = cnt.reshape(ntrials, NT).T.astype(np.float32)
            col += 1
    out /= DT                                            # spikes / s
    sm = my_smooth(out.reshape(NT, -1)).astype(np.float32)
    return sm.reshape(NT, nunits, ntrials)


def mean_firing_rates(obj, probes, keep_clu, gocue, ntrials):
    """Mean FR (Hz) over the aligned analysis window, across all trials.

    removeLowFRClusters.m: "all those firing less than lowFR spikes per second
    on average across all trials".
    """
    frs = []
    for prb in probes:
        for i in keep_clu[prb]:
            tt, tr = obj.spikes(prb, i)
            ok = (tr >= 1) & (tr <= ntrials)
            al = tt[ok] - gocue[tr[ok] - 1]
            n = np.count_nonzero((al >= TMIN) & (al < TMAX))
            frs.append(n / ((TMAX - TMIN) * ntrials))
    return np.array(frs)


def video_offset(obj):
    """findVideoOffset.m"""
    bitstart = obj.sglx_bitstart()
    fs = obj.sglx_fs()
    bp_bitstart = obj.bp('ev.bitStart')
    bp_bitstart = bp_bitstart[np.isfinite(bp_bitstart)]
    return (stats.mode(bitstart, keepdims=False).mode / fs
            - stats.mode(bp_bitstart, keepdims=False).mode)


def kinematic_speed(obj, view, featname, gocue, ntrials, vidshift):
    """Interpolated DLC speed for one feature, aligned to the go cue.

    Returns
        speed   (ntrials, NT) float, |d(position)/dt| in px/bin
        visible (ntrials, NT) bool, True where DLC tracked the feature
    Mirrors findPosition.m / findVelocity.m: interpolate the raw DLC trace onto
    the neural time axis (video clock shifted by findVideoOffset and the
    alignment event), then differentiate.
    """
    feats = obj.featnames(view)
    featix = feats.index(featname)
    speed = np.full((ntrials, NT), np.nan)
    visible = np.zeros((ntrials, NT), dtype=bool)
    taxis = TAXIS + ADVANCE_MOVEMENT
    for t in range(ntrials):
        xy, ft = obj.traj_xy(view, t, featix)
        if xy is None or not np.isfinite(obj.ndropped(view, t)):
            continue                                   # no usable video
        # findPosition.m: if frameTimes is missing/all NaN, fall back to a
        # nominal 400 Hz frame clock
        if ft is None or not np.all(np.isfinite(ft)) or ft.size != xy.shape[1]:
            ft = np.arange(1, xy.shape[1] + 1) / 400.0 - 0.5 + vidshift
        tv = ft - vidshift - gocue[t]
        x = interp_matlab(taxis, tv, xy[0])
        y = interp_matlab(taxis, tv, xy[1])
        vis = np.isfinite(x) & np.isfinite(y)
        if not vis.any():
            continue
        vx = nan_gradient(x)
        vy = nan_gradient(y)
        speed[t] = np.sqrt(vx ** 2 + vy ** 2)
        visible[t] = vis & np.isfinite(speed[t])
    return speed, visible


def motion_energy_trace(obj, me_cells, gocue, ntrials, vidshift):
    """loadMotionEnergy.m: interpolate 400 Hz motion energy onto the neural axis."""
    out = np.full((ntrials, NT), np.nan)
    taxis = TAXIS + ADVANCE_MOVEMENT
    for t in range(min(ntrials, len(me_cells))):
        m = me_cells[t]
        if m.size == 0:
            continue
        ft = obj.frame_times(0, t)
        if ft is None or ft.size != m.size or not np.all(np.isfinite(ft)):
            ft = (np.arange(1, m.size + 1) / 400.0) - 0.5 + vidshift  # fallback
        out[t] = interp_matlab(taxis, ft - vidshift - gocue[t], m)
    return out


def discretize(values, valid, thresh=None):
    """0 = < median, 1 = >= median, 2 = invalid (not visible / no video)."""
    v = np.full(values.shape, 2, dtype=np.int8)
    if thresh is None:
        thresh = np.nanpercentile(values[valid], 50) if valid.any() else np.nan
    if valid.any():
        v[valid] = (values[valid] >= thresh).astype(np.int8)
    return v, thresh


def process_session(anm, date, probes, datadir, task, show=False, outdir='/app'):
    """Load + process one session. Returns a dict of converted arrays."""
    t0 = time.time()
    fn = os.path.join(datadir, f'data_structure_{anm}_{date}.mat')
    obj = open_session(fn)
    timing = {}

    ntrials = obj.Ntrials
    gocue = obj.bp('ev.goCue')
    sample = obj.bp('ev.sample')
    delay = obj.bp('ev.delay')
    hit, miss, no = obj.bp('hit'), obj.bp('miss'), obj.bp('no')
    R, L = obj.bp('R'), obj.bp('L')
    autowater = obj.bp('autowater')
    early = obj.bp('early')
    stim = obj.bp('stim.enable') if obj.has_bp('stim.enable') else np.zeros(ntrials)

    # ---- cluster curation (findClusters.m, case-insensitive) --------------
    keep_clu, qualities = {}, {}
    for prb in probes:
        q = obj.qualities(prb)
        qualities[prb] = q
        keep_clu[prb] = [i for i, qq in enumerate(q) if qq.lower() not in BAD_QUALITY]
    n_quality = sum(len(v) for v in keep_clu.values())

    # ---- low firing rate removal (removeLowFRClusters.m) ------------------
    fr = mean_firing_rates(obj, probes, keep_clu, gocue, ntrials)
    use = fr > LOW_FR
    off = 0
    for prb in probes:
        k = len(keep_clu[prb])
        keep_clu[prb] = [c for c, u in zip(keep_clu[prb], use[off:off + k]) if u]
        off += k
    nunits = sum(len(v) for v in keep_clu.values())
    timing['curation'] = time.time() - t0

    # per-unit metadata
    unit_quality, unit_region, unit_probe = [], [], []
    for prb in probes:
        loc = obj.probe_loc(prb)
        region = 'tjM1' if 'M1TJ' in loc.upper() else 'ALM'
        for i in keep_clu[prb]:
            unit_quality.append(qualities[prb][i].lower())
            unit_region.append(region)
            unit_probe.append((prb, loc))

    # ---- neural: binned, smoothed single trial firing rates ---------------
    t1 = time.time()
    trialdat = bin_spikes(obj, probes, keep_clu, gocue, ntrials)   # (NT, nunits, ntrials)
    timing['neural'] = time.time() - t1

    # ---- video derived outputs -------------------------------------------
    t1 = time.time()
    vidshift = video_offset(obj)
    tongue_speed, tongue_vis = kinematic_speed(obj, TONGUE_VIEW, TONGUE_FEAT,
                                               gocue, ntrials, vidshift)
    paw_speed, paw_vis = kinematic_speed(obj, PAW_VIEW, PAW_FEAT,
                                         gocue, ntrials, vidshift)
    mefn = os.path.join(datadir, f'motionEnergy_{anm}_{date}.mat')
    me_cells = load_motion_energy(mefn)
    me = motion_energy_trace(obj, me_cells, gocue, ntrials, vidshift)
    me_vis = np.isfinite(me)
    timing['video'] = time.time() - t1

    # ---- trial curation ---------------------------------------------------
    # early-lick and photoinactivation trials are excluded from all analyses in
    # the reference (params.condition: '~stim.enable & ~early'; Methods:
    # "Trials in which the animal contacted the lickport before the reward
    # ('early lick') were omitted from analyses").
    keep = (early < 0.5) & (stim < 0.5) & np.isfinite(gocue)
    # trials with no spikes at all on any unit = no ephys coverage (the
    # recording ended before the behavioural session did)
    spk_per_trial = trialdat.sum(axis=(0, 1))
    no_ephys = spk_per_trial <= 0
    keep &= ~no_ephys
    ntrials_keep = int(keep.sum())

    # ---- outputs ----------------------------------------------------------
    # lick direction (getPrevChoice.m: choice = (R&hit) | (L&miss))
    lickdir = np.full(ntrials, 2, dtype=np.int8)                 # none
    lickdir[((R > 0.5) & (hit > 0.5)) | ((L > 0.5) & (miss > 0.5))] = 1   # right
    lickdir[((L > 0.5) & (hit > 0.5)) | ((R > 0.5) & (miss > 0.5))] = 0   # left
    lickdir[no > 0.5] = 2

    context = np.where(autowater > 0.5, 0, 1).astype(np.int8)    # 0=WC, 1=DR

    outcome = np.full(ntrials, -1, dtype=np.int8)
    outcome[miss > 0.5] = 0                                      # incorrect
    outcome[hit > 0.5] = 1                                       # correct
    outcome[no > 0.5] = 2                                        # ignore

    # per-session median thresholds, computed on the retained trials only
    tng_d, tng_thr = discretize(tongue_speed[keep], tongue_vis[keep])
    paw_d, paw_thr = discretize(paw_speed[keep], paw_vis[keep])
    me_d, me_thr = discretize(me[keep], me_vis[keep])

    kt = np.flatnonzero(keep)
    neural, inputs, outputs = [], [], []
    inp = TAXIS.astype(np.float32)[None, :]
    for j, t in enumerate(kt):
        neural.append(np.ascontiguousarray(trialdat[:, :, t].T))       # (nunits, NT)
        inputs.append(inp.copy())
        out = np.empty((6, NT), dtype=np.int8)
        out[0] = lickdir[t]
        out[1] = context[t]
        out[2] = outcome[t]
        out[3] = tng_d[j]
        out[4] = paw_d[j]
        out[5] = me_d[j]
        outputs.append(out)

    info = dict(
        animal=anm, date=date, task=task, probes=[int(p) for p in probes],
        probe_locations=sorted(set(p[1] for p in unit_probe)),
        file_format=obj.fmt,
        ntrials_total=int(ntrials), ntrials_kept=ntrials_keep,
        n_early=int(np.sum(early > 0.5)), n_stim=int(np.sum(stim > 0.5)),
        n_no_ephys=int(np.sum(no_ephys)),
        nunits_all=sum(len(qualities[p]) for p in probes),
        nunits_quality=n_quality, nunits=nunits,
        # provenance: 0-based indices of the retained trials and of the retained
        # clusters (index into obj.clu{probe}) in the original data object
        trial_indices=np.flatnonzero(keep).astype(np.int32),
        cluster_indices={int(p): np.asarray(keep_clu[p], dtype=np.int32) for p in probes},
        unit_quality=list(unit_quality),
        n_single_units=int(sum(q in SINGLE_UNIT_QUALITY for q in unit_quality)),
        video_offset_s=float(vidshift),
        median_delay_s=float(np.nanmedian(gocue - delay)),
        median_sample_s=float(np.nanmedian(delay - sample)),
        tongue_speed_threshold=float(tng_thr), paw_speed_threshold=float(paw_thr),
        motion_energy_threshold=float(me_thr),
        frac_context_WC=float(np.mean(context[keep] == 0)),
        frac_outcome=[float(np.mean(outcome[keep] == c)) for c in range(3)],
        frac_lickdir=[float(np.mean(lickdir[keep] == c)) for c in range(3)],
        frac_tongue_notvisible=float(np.mean(tng_d == 2)),
        frac_paw_notvisible=float(np.mean(paw_d == 2)),
        frac_me_novideo=float(np.mean(me_d == 2)),
        mean_firing_rate=float(np.mean([f for f, u in zip(fr, use) if u])) if nunits else 0.0,
    )

    if outcome[keep].min() < 0:
        raise RuntimeError(f'{anm} {date}: trial with no hit/miss/no outcome')

    res = dict(neural=neural, input=inputs, output=outputs,
               region=unit_region, quality=unit_quality, info=info,
               subject=anm)

    if show:
        _plot_processing(outdir, anm, date, obj, gocue, sample, delay, keep, kt,
                         trialdat, tongue_speed, tongue_vis, tng_thr,
                         paw_speed, paw_vis, paw_thr, me, me_vis, me_thr,
                         lickdir, context, outcome, unit_region)

    timing['total'] = time.time() - t0
    info['timing'] = {k: round(v, 2) for k, v in timing.items()}
    obj.close()
    return res


# --------------------------------------------------------------------------- #
#  Diagnostic plots
# --------------------------------------------------------------------------- #
def _plot_processing(outdir, anm, date, obj, gocue, sample, delay, keep, kt,
                     trialdat, tongue_speed, tongue_vis, tng_thr,
                     paw_speed, paw_vis, paw_thr, me, me_vis, me_thr,
                     lickdir, context, outcome, unit_region):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ev_sample = np.median(sample - gocue)
    ev_delay = np.median(delay - gocue)
    fig, ax = plt.subplots(4, 2, figsize=(18, 20))

    # (1) raster + smoothed rate for the most active unit, first kept trials
    rates = trialdat[:, :, keep].mean(axis=(0, 2))
    u = int(np.argmax(rates))
    axx = ax[0, 0]
    ntr = min(40, len(kt))
    lL, lR = obj.licks('L'), obj.licks('R')
    for j, t in enumerate(kt[:ntr]):
        axx.plot(TAXIS, trialdat[:, u, t] / max(1e-9, trialdat[:, u, kt[:ntr]].max()) * 0.9 + j,
                 lw=0.5, color='k')
        for lt, c in ((lL[t], 'r'), (lR[t], 'b')):
            lt = np.asarray(lt).ravel() - gocue[t]
            lt = lt[(lt > TMIN) & (lt < TMAX)]
            axx.plot(lt, np.full(lt.size, j + 0.5), '.', color=c, ms=2)
    for e, c in ((0, 'k'), (ev_sample, 'g'), (ev_delay, 'm')):
        axx.axvline(e, color=c, ls='--', lw=1)
    axx.set_title(f'{anm} {date}: unit {u} smoothed rate per trial (black)\n'
                  'licks: red=left blue=right; dashed: sample/delay/go cue')
    axx.set_xlabel('time from go cue (s)'); axx.set_ylabel('trial')

    # (2) population PSTH split by lick direction
    axx = ax[0, 1]
    for d, c, lab in ((0, 'r', 'left'), (1, 'b', 'right'), (2, 'gray', 'none')):
        m = keep & (lickdir == d)
        if m.sum() > 2:
            axx.plot(TAXIS, trialdat[:, :, m].mean(axis=(1, 2)), color=c, label=lab)
    for e, c in ((0, 'k'), (ev_sample, 'g'), (ev_delay, 'm')):
        axx.axvline(e, color=c, ls='--', lw=1)
    axx.legend(); axx.set_title('population mean firing rate by lick direction')
    axx.set_xlabel('time from go cue (s)'); axx.set_ylabel('spk/s')

    # (3) tongue speed + discretisation
    for row, (sp, vis, thr, name) in enumerate(
            [(tongue_speed, tongue_vis, tng_thr, 'tongue speed'),
             (paw_speed, paw_vis, paw_thr, 'paw speed'),
             (me, me_vis, me_thr, 'motion energy')]):
        axx = ax[row + 1, 0]
        for j, t in enumerate(kt[:6]):
            v = sp[t].copy()
            axx.plot(TAXIS, v + j * 3 * (thr if np.isfinite(thr) else 1), 'k', lw=0.8)
            iv = ~vis[t]
            axx.plot(TAXIS[iv], np.full(iv.sum(), j * 3 * thr), 'r.', ms=2)
            axx.axhline(j * 3 * thr + thr, color='b', lw=0.5)
        axx.axvline(0, color='k', ls='--')
        axx.set_title(f'{name}: traces for 6 trials (blue = per-session median '
                      f'{thr:.3g}, red = invalid)')
        axx.set_xlabel('time from go cue (s)')

        axx = ax[row + 1, 1]
        vals = sp[keep][vis[keep]]
        axx.hist(vals, bins=100)
        axx.axvline(thr, color='b')
        axx.set_yscale('log')
        axx.set_title(f'{name} distribution over valid samples; median={thr:.3g}; '
                      f'frac invalid={1 - np.mean(vis[keep]):.3f}')

    fig.tight_layout()
    fn = os.path.join(outdir, f'processing_{anm}_{date}.png')
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print(f'  wrote {fn}')


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save per-step diagnostic plots for up to 2 sessions')
    args = ap.parse_args()

    sessions = SESSIONS
    if args.sample:
        # one fixed-delay two-context session and one randomized-delay session
        sessions = [SESSIONS[2], SESSIONS[29]]

    t_start = time.time()
    data = dict(neural=[], input=[], output=[], subjects=[], subject_idx=[],
                brain_regions=[], brain_region_idx=[],
                input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                output_values=OUTPUT_VALUES, metadata={})
    subjects, regions, session_info = [], [], []
    nshown = 0
    for k, (anm, date, probes, datadir, task) in enumerate(sessions):
        show = args.show_processing and nshown < 2
        res = process_session(anm, date, probes, datadir, task, show=show,
                              outdir=os.path.dirname(os.path.abspath(args.outfile)) or '.')
        nshown += int(show)
        if anm not in subjects:
            subjects.append(anm)
        for r in res['region']:
            if r not in regions:
                regions.append(r)
        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        data['subject_idx'].append(subjects.index(anm))
        data['brain_region_idx'].append(np.array([regions.index(r) for r in res['region']],
                                                 dtype=np.int64))
        session_info.append(res['info'])
        i = res['info']
        print(f"[{k + 1}/{len(sessions)}] {anm} {date} ({i['file_format']}) "
              f"units {i['nunits_all']}->{i['nunits_quality']}->{i['nunits']} "
              f"(SU {i['n_single_units']}) | trials {i['ntrials_total']}->{i['ntrials_kept']} "
              f"(early {i['n_early']}, stim {i['n_stim']}, no-ephys {i['n_no_ephys']}) | "
              f"WC {i['frac_context_WC']:.2f} | invalid tongue/paw/me "
              f"{i['frac_tongue_notvisible']:.2f}/{i['frac_paw_notvisible']:.2f}/"
              f"{i['frac_me_novideo']:.2f} | {i['timing']['total']:.1f}s "
              f"{i['timing']}", flush=True)

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['brain_regions'] = regions
    data['metadata'] = dict(
        task_description=(
            'Head-fixed mice performed directional licking tasks. In delayed-response '
            '(DR) trials an auditory tone (1.3 s sample epoch) instructed a left or right '
            'lick, which had to be withheld through a delay epoch (0.9 s fixed, or '
            'randomized 0.3-3.6 s in the randomized-delay sessions) until an auditory go '
            'cue; correct licks were rewarded with water. In water-cued (WC) trials, '
            'interleaved in blocks of 10-25 trials in a subset of sessions, all auditory '
            'cues were omitted and water was delivered at a random port at a random time; '
            'mice had to infer the context. Neural activity was recorded extracellularly '
            'with silicon probes in anterior lateral motor cortex (ALM; a few probes were '
            'located in tongue-jaw motor cortex, tjM1). Decoded variables are the '
            "animal's lick direction, the behavioural context, the trial outcome, and "
            'discretised tongue speed, paw speed and whole-frame motion energy from '
            'high-speed video.'),
        time_bin_size=DT * 1000.0,
        temporal_alignment_event=('go cue onset (DR trials) / water drop presentation '
                                  '(WC trials); obj.bp.ev.goCue'),
        off_start=TMIN,
        off_end=TMAX,
        neural_units='spikes/s (spike counts in 10 ms bins, divided by the bin width and '
                     'smoothed with the causal gaussian kernel of mySmooth.m, N=15 bins, '
                     'reflect boundary condition)',
        neuron_curation=('clusters whose manual quality label is garbage/gabrga/noisy/real? '
                         'are discarded (findClusters.m), then units with mean firing rate '
                         '<= 1 Hz over the -2.5..2.5 s window across all trials are '
                         'discarded (removeLowFRClusters.m, params.lowFR = 1)'),
        trial_curation=('early-lick trials (bp.early) and photoinactivation trials '
                        '(bp.stim.enable) are excluded, as in the reference conditions '
                        "'~stim.enable&~early'; trials with no spikes on any unit (ephys "
                        'recording ended before the behavioural session) are excluded'),
        video='DeepLabCut, 400 Hz, two cameras; aligned to the ephys clock with '
              'findVideoOffset.m and resampled onto the neural time axis',
        output_discretisation=('tongue/paw speed and motion energy are split at the 50th '
                               'percentile of all valid samples of that session; timepoints '
                               'where DeepLabCut did not track the feature (tongue, paw) or '
                               'where no video frames cover the timepoint (motion energy) '
                               'get the third class'),
        source='Hasnain, Birnbaum et al., Nat Neurosci 2024; Zenodo 10.5281/zenodo.13941415',
        session_info=session_info,
    )

    # ---- summary ---------------------------------------------------------
    ntr = [len(s) for s in data['neural']]
    nun = [s[0].shape[0] for s in data['neural']]
    print(f'\nsessions {len(ntr)} | subjects {len(subjects)} | '
          f'trials {sum(ntr)} | units {sum(nun)} | '
          f'regions {Counter([regions[i] for s in data["brain_region_idx"] for i in s])}')
    print(f'total time {time.time() - t_start:.1f}s')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB)')


if __name__ == '__main__':
    main()
