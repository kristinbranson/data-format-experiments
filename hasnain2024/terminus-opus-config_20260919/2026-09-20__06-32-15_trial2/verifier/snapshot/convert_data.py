#!/usr/bin/env python3
"""
Convert the Hasnain, Birnbaum et al. (Nat Neurosci 2024) ALM electrophysiology +
high-speed-video dataset into the decoder-compatible pickle format.

Processing follows the authors' MATLAB pipeline (DataLoadingScripts/, funcs/):
  * trials aligned to the go cue (= water-drop time on water-cued trials)
  * window [-2.5, 2.5] s, 10 ms bins, causal Gaussian smoothing (mySmooth, N=15)
  * cluster quality filter (findClusters) + 1 Hz firing-rate filter (removeLowFRClusters)
  * photostim and early-lick trials removed (reference condition strings)
  * video aligned with the reference clock offset (findVideoOffset)

Usage:  python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""
import argparse, glob, os, pickle, sys, time
import numpy as np
import h5py
import scipy.io as sio
from scipy.interpolate import interp1d
from multiprocessing import Pool

# --------------------------------------------------------------------------------------
# Parameters (mirroring params in the reference figure scripts, e.g. Scripts/Figure 3/Figure3c.m)
# --------------------------------------------------------------------------------------
ALIGN_EVENT = 'goCue'
TMIN, TMAX = -2.5, 2.5      # params.tmin / params.tmax
DT_FINE = 0.01              # params.dt (1/100 s) used for binning + smoothing
SMOOTH_N = 15               # params.smooth, causal gaussian window (bins)
BCTYPE = 'reflect'          # params.bctype
LOW_FR = 1.0                # params.lowFR (Hz)
DOWNSAMPLE = 5              # 10 ms -> 50 ms decoder bins
DT_OUT = DT_FINE * DOWNSAMPLE
MIN_UNITS = 10              # paper: sessions included only if they had at least 10 units
MIN_TRIALS = 2              # decoder requirement
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}   # findClusters.m
VIDEO_FS = 400.0

DATA_FIXED = '/app/data/Ephys_Behavior'
DATA_RAND = '/app/data/RandomizedDelay_Ephys_Behavior'

# Sessions + ALM probe(s), transcribed from DataLoadingScripts/'Recording and video'/load*_ALMVideo.m
# (commented-out entries in those files are excluded, as are JEB4/JEB5 whose data are not distributed)
SESSIONS = [
    # (animal, date, probes, directory, task)
    ('JEB6',  '2021-04-18', [2],    DATA_FIXED, 'fixed delay'),
    ('JEB7',  '2021-04-29', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB7',  '2021-04-30', [1],    DATA_FIXED, 'fixed delay'),
    ('EKH1',  '2021-08-07', [2],    DATA_FIXED, 'fixed delay'),
    ('EKH3',  '2021-08-11', [2],    DATA_FIXED, 'fixed delay'),
    ('JGR2',  '2021-11-16', [1],    DATA_FIXED, 'fixed delay'),
    ('JGR2',  '2021-11-17', [1],    DATA_FIXED, 'fixed delay'),
    ('JGR3',  '2021-11-18', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB15', '2022-07-26', [1, 2], DATA_FIXED, 'fixed delay'),
    ('JEB15', '2022-07-27', [1, 2], DATA_FIXED, 'fixed delay'),
    ('JEB15', '2022-07-28', [1, 2], DATA_FIXED, 'fixed delay'),
    ('JEB15', '2022-07-29', [2],    DATA_FIXED, 'fixed delay'),
    ('JEB14', '2022-08-22', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB14', '2022-08-23', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB14', '2022-08-24', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB14', '2022-08-25', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB13', '2022-09-13', [2],    DATA_FIXED, 'fixed delay'),
    ('JEB13', '2022-09-14', [2],    DATA_FIXED, 'fixed delay'),
    ('JEB13', '2022-09-21', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB13', '2022-09-24', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB13', '2022-09-25', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB19', '2023-04-18', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB19', '2023-04-19', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB19', '2023-04-20', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB19', '2023-04-21', [1],    DATA_FIXED, 'fixed delay'),
    ('JEB11', '2022-05-10', [1],    DATA_RAND,  'randomized delay'),
    ('JEB11', '2022-05-11', [1],    DATA_RAND,  'randomized delay'),
    ('JEB12', '2022-05-12', [1],    DATA_RAND,  'randomized delay'),
    ('JEB12', '2022-05-13', [1],    DATA_RAND,  'randomized delay'),
    ('JEB23', '2023-10-10', [1],    DATA_RAND,  'randomized delay'),
    ('JEB23', '2023-10-11', [1],    DATA_RAND,  'randomized delay'),
    ('JEB23', '2023-10-12', [1],    DATA_RAND,  'randomized delay'),
    ('JEB23', '2023-10-13', [1],    DATA_RAND,  'randomized delay'),
    ('JEB23', '2023-10-18', [1],    DATA_RAND,  'randomized delay'),
    ('JEB23', '2023-10-19', [1],    DATA_RAND,  'randomized delay'),
    ('JEB23', '2023-10-21', [1],    DATA_RAND,  'randomized delay'),
    ('JEB24', '2023-10-23', [1],    DATA_RAND,  'randomized delay'),
    ('JEB24', '2023-10-24', [1],    DATA_RAND,  'randomized delay'),
    ('JEB24', '2023-10-25', [1],    DATA_RAND,  'randomized delay'),
    ('JEB24', '2023-10-26', [1],    DATA_RAND,  'randomized delay'),
    ('JEB24', '2023-10-27', [1],    DATA_RAND,  'randomized delay'),
    ('JEB24', '2023-10-31', [1],    DATA_RAND,  'randomized delay'),
    ('JEB24', '2023-11-02', [1],    DATA_RAND,  'randomized delay'),
    ('JEB24', '2023-11-03', [1],    DATA_RAND,  'randomized delay'),
]

TONGUE_FEAT = ('side', 'tongue')            # obj.traj{1} 'tongue'
PAW_FEATS = [('bottom', 'top_paw'), ('bottom', 'bottom_paw')]   # obj.traj{2}, paws only in bottom view

# --------------------------------------------------------------------------------------
# MATLAB file loading (handles both v7 and v7.3 files)
# --------------------------------------------------------------------------------------
def is_v73(fn):
    with open(fn, 'rb') as fh:
        return b'MATLAB 7.3' in fh.read(128)

def _h5str(f, ref):
    a = np.array(f[ref]).ravel()
    return ''.join(chr(int(c)) for c in a)

def _v7str(x):
    if isinstance(x, str):
        return x
    a = np.array(x).ravel()
    if a.size == 0:
        return ''
    if a.dtype.kind in 'US':
        return str(a[0])
    return ''.join(chr(int(c)) for c in a)

def _load_v73(fn, probes, feats):
    """Load only what we need from a v7.3 (HDF5) data_structure file."""
    f = h5py.File(fn, 'r')
    o = f['obj']
    out = {}
    bp = o['bp']
    B = {k: np.array(bp[k]).ravel().astype(float) for k in
         ['Ntrials', 'L', 'R', 'hit', 'miss', 'no', 'early', 'autowater'] if k in bp}
    B['stim_enable'] = (np.array(bp['stim']['enable']).ravel().astype(float)
                        if 'stim' in bp else None)
    ev = {k: np.array(bp['ev'][k]).ravel().astype(float)
          for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward'] if k in bp['ev']}
    for k in ['lickL', 'lickR']:
        refs = bp['ev'][k]
        n = refs.shape[1] if refs.ndim == 2 else refs.shape[0]
        lst = []
        for i in range(n):
            r = refs[0, i] if refs.ndim == 2 else refs[i]
            try:
                lst.append(np.array(f[r]).ravel().astype(float))
            except Exception:
                lst.append(np.array([]))
        ev[k] = lst
    B['ev'] = ev
    out['bp'] = B
    # clusters (only requested probes)
    clu = {}
    for p in probes:
        g = f[o['clu'][p - 1, 0]] if 'clu' in o and o['clu'].shape[0] >= p else None
        if isinstance(g, h5py.Group) and 'quality' in g:
            n = g['quality'].shape[0]
            clu[p] = dict(
                quality=[_h5str(f, g['quality'][i, 0]).strip() for i in range(n)],
                trialtm=[np.array(f[g['trialtm'][i, 0]]).ravel().astype(np.float64) for i in range(n)],
                trial=[np.array(f[g['trial'][i, 0]]).ravel().astype(np.int64) for i in range(n)])
        else:
            clu[p] = None
    out['clu'] = clu
    # trajectories: only the requested features
    traj = {}
    viewmap = {'side': 0, 'bottom': 1}
    for viewname, featname in feats:
        v = viewmap[viewname]
        g = f[o['traj'][v, 0]]
        featNames = [_h5str(f, x) for x in np.array(f[g['featNames'][0, 0]]).ravel()]
        if featname not in featNames:
            traj[(viewname, featname)] = None
            continue
        fi = featNames.index(featname)
        ntr = g['ts'].shape[0]
        xy, ft, nd = [], [], []
        for i in range(ntr):
            ds = f[g['ts'][i, 0]]               # (nfeat, 3, nframes)
            xy.append(np.array(ds[fi, :2, :]).T.astype(np.float64))   # (nframes, 2)
            ft.append(np.array(f[g['frameTimes'][i, 0]]).ravel().astype(np.float64))
            try:
                nd.append(float(np.array(f[g['NdroppedFrames'][i, 0]]).ravel()[0]))
            except Exception:
                nd.append(np.nan)
        traj[(viewname, featname)] = dict(xy=xy, frameTimes=ft, NdroppedFrames=np.array(nd))
    out['traj'] = traj
    # side-cam frame times are also used for motion-energy alignment
    g = f[o['traj'][0, 0]]
    out['frameTimes'] = [np.array(f[g['frameTimes'][i, 0]]).ravel().astype(np.float64)
                         for i in range(g['frameTimes'].shape[0])]
    out['nframes'] = [f[g['ts'][i, 0]].shape[2] for i in range(g['ts'].shape[0])]
    out['NdroppedFrames'] = np.array([
        (float(np.array(f[g['NdroppedFrames'][i, 0]]).ravel()[0])
         if g['NdroppedFrames'][i, 0] else np.nan)
        for i in range(g['NdroppedFrames'].shape[0])])
    out['sglx'] = dict(fs=float(np.array(o['sglx']['fs']).ravel()[0]),
                       bitstart=np.array(o['sglx']['bitcode']['bitstart']).ravel().astype(float))
    locs = []
    try:
        pr = o['ex']['probe']['loc']
        for i in range(pr.shape[0]):
            locs.append(_h5str(f, pr[i, 0]).replace('\x00', '').strip())
    except Exception:
        locs = []
    out['probe_loc'] = locs
    f.close()
    return out

def _load_v7(fn, probes, feats):
    m = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)
    o = m['obj']
    out = {}
    bp = o.bp
    B = {k: np.atleast_1d(np.array(getattr(bp, k)).ravel()).astype(float)
         for k in ['Ntrials', 'L', 'R', 'hit', 'miss', 'no', 'early', 'autowater'] if hasattr(bp, k)}
    B['stim_enable'] = (np.array(bp.stim.enable).ravel().astype(float)
                        if hasattr(bp, 'stim') else None)
    ev = {k: np.array(getattr(bp.ev, k)).ravel().astype(float)
          for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward'] if hasattr(bp.ev, k)}
    for k in ['lickL', 'lickR']:
        raw = getattr(bp.ev, k)
        ev[k] = [np.atleast_1d(np.array(raw[i]).ravel()).astype(float) for i in range(len(raw))]
    B['ev'] = ev
    out['bp'] = B

    def parse_probe(sa):
        cc = np.atleast_1d(sa)
        n = cc.size
        if n == 0:
            return None
        return dict(quality=[_v7str(cc[i].quality).strip() for i in range(n)],
                    trialtm=[np.atleast_1d(np.array(cc[i].trialtm).ravel()).astype(np.float64) for i in range(n)],
                    trial=[np.atleast_1d(np.array(cc[i].trial).ravel()).astype(np.int64) for i in range(n)])
    clu = {}
    if hasattr(o, 'clu'):
        c = o.clu
        first = np.atleast_1d(c).ravel()[0] if np.size(c) > 0 else None
        single = first is not None and hasattr(first, 'quality')
        for p in probes:
            if single:
                clu[p] = parse_probe(c) if p == 1 else None
            else:
                arr = np.atleast_1d(c).ravel()
                clu[p] = parse_probe(arr[p - 1]) if len(arr) >= p else None
    else:
        clu = {p: None for p in probes}
    out['clu'] = clu

    viewmap = {'side': 0, 'bottom': 1}
    traj = {}
    for viewname, featname in feats:
        tv = np.atleast_1d(o.traj[viewmap[viewname]])
        featNames = [_v7str(x) for x in np.atleast_1d(tv[0].featNames)]
        if featname not in featNames:
            traj[(viewname, featname)] = None
            continue
        fi = featNames.index(featname)
        xy, ft, nd = [], [], []
        for i in range(tv.size):
            a = np.array(tv[i].ts, dtype=float)      # (nframes, 3, nfeat)
            xy.append(a[:, :2, fi])
            ft.append(np.atleast_1d(np.array(tv[i].frameTimes).ravel()).astype(float))
            try:
                nd.append(float(np.array(tv[i].NdroppedFrames).ravel()[0]))
            except Exception:
                nd.append(np.nan)
        traj[(viewname, featname)] = dict(xy=xy, frameTimes=ft, NdroppedFrames=np.array(nd))
    out['traj'] = traj
    tv = np.atleast_1d(o.traj[0])
    out['frameTimes'] = [np.atleast_1d(np.array(tv[i].frameTimes).ravel()).astype(float)
                         for i in range(tv.size)]
    out['nframes'] = [np.array(tv[i].ts).shape[0] for i in range(tv.size)]
    nd = []
    for i in range(tv.size):
        try:
            nd.append(float(np.array(tv[i].NdroppedFrames).ravel()[0]))
        except Exception:
            nd.append(np.nan)
    out['NdroppedFrames'] = np.array(nd)
    out['sglx'] = dict(fs=float(np.array(o.sglx.fs).ravel()[0]),
                       bitstart=np.array(o.sglx.bitcode.bitstart).ravel().astype(float))
    locs = []
    try:
        for x in np.atleast_1d(o.ex.probe.loc):
            locs.append(_v7str(x).replace('\x00', '').strip())
    except Exception:
        locs = []
    out['probe_loc'] = locs
    return out

def load_session(fn, probes, feats):
    return _load_v73(fn, probes, feats) if is_v73(fn) else _load_v7(fn, probes, feats)

def load_motion_energy(fn):
    """Load motionEnergy_ANM_DATE.mat.

    Three layouts occur in the released data (all handled by loadMotionEnergy.m, which does
    `me = temp.me; if isstruct(me.data), me.data = me.data.data; end`):
      1. `me` struct with fields {data (cell of per-trial 400 Hz vectors), moveThresh}
      2. `me` struct whose `data` field is itself a struct with {data, moveThresh}
      3. `me` saved directly as the cell array of per-trial vectors (no moveThresh)
    """
    m = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)
    me = m['me']
    thresh = np.nan
    raw = me
    # descend through nested structs until we reach the cell array of trials
    while hasattr(raw, '_fieldnames'):
        if 'moveThresh' in raw._fieldnames:
            try:
                thresh = float(np.array(raw.moveThresh).ravel()[0])
            except Exception:
                pass
        if 'data' not in raw._fieldnames:
            break
        raw = raw.data
    data = [np.atleast_1d(np.array(d).ravel()).astype(float) for d in np.atleast_1d(raw)]
    return dict(data=data, moveThresh=thresh)

# --------------------------------------------------------------------------------------
# Reference processing helpers
# --------------------------------------------------------------------------------------
def gausswin(N, alpha=2.5):
    """MATLAB gausswin(N, alpha) (default alpha = 2.5)."""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)

def causal_kernel(N=SMOOTH_N):
    """Kernel used by utils/mySmooth.m: gausswin with the first floor(N/2) taps zeroed."""
    k = gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()

KERN = causal_kernel()

def my_smooth(x, kern=KERN, bctype=BCTYPE):
    """utils/mySmooth.m, operating on the LAST axis (time).

    MATLAB: x_filt = cat(1, x(1:N,:), x); conv(...,'same'); out = out(N+1:end,:)
    """
    from scipy.signal import fftconvolve
    N = len(kern)
    if bctype == 'reflect':
        xf = np.concatenate([x[..., :N], x], axis=-1)
        trim = N
    else:
        xf = x
        trim = 0
    out = fftconvolve(xf, kern.reshape((1,) * (xf.ndim - 1) + (N,)), mode='same', axes=-1)
    return out[..., trim:]

def time_axes():
    """Fine (10 ms) bin edges/centres and the down-sampled decoder bin centres."""
    edges = np.round(np.arange(TMIN, TMAX + 1e-9, DT_FINE), 6)
    centres = edges[:-1] + DT_FINE / 2.0
    nfine = len(centres)
    nout = nfine // DOWNSAMPLE
    centres_out = centres[:nout * DOWNSAMPLE].reshape(nout, DOWNSAMPLE).mean(axis=1)
    return edges, centres, centres_out

EDGES, TCENT, TOUT = time_axes()
NFINE, NOUT = len(TCENT), len(TOUT)

def downsample_mean(x, k=DOWNSAMPLE, nanmean=False):
    """Average k consecutive samples along the last axis."""
    n = (x.shape[-1] // k) * k
    y = x[..., :n].reshape(x.shape[:-1] + (n // k, k))
    if nanmean:
        with np.errstate(invalid='ignore'):
            return np.nanmean(y, axis=-1)
    return y.mean(axis=-1)

def video_offset(sess):
    """funcs/findVideoOffset.m: mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)."""
    from scipy.stats import mode as smode
    bs = sess['sglx']['bitstart']
    bs = bs[~np.isnan(bs)]
    bstart = sess['bp']['ev']['bitStart']
    bstart = bstart[~np.isnan(bstart)]
    if bs.size == 0 or bstart.size == 0:
        return 0.5
    vid_file_offset = float(smode(bs, keepdims=False).mode) / sess['sglx']['fs']
    return vid_file_offset - float(smode(np.round(bstart, 6), keepdims=False).mode)

def frame_times_for_trial(sess, trial, vidshift, align_t):
    """Frame times relative to the alignment event (reference fallback if frameTimes bad)."""
    ft = sess['frameTimes'][trial]
    if ft.size == 0 or np.all(np.isnan(ft)):
        ft = (np.arange(1, sess['nframes'][trial] + 1) / VIDEO_FS)
        return ft - 0.5 - align_t
    return ft - vidshift - align_t

# --------------------------------------------------------------------------------------
# Neural data
# --------------------------------------------------------------------------------------
def bin_spikes(clu, probes, align_times, keep_trials, ntrials):
    """Bin + smooth spikes exactly like DataLoadingScripts/getSeq.m + alignSpikes.m.

    Returns (rates (ntrials_kept, nunits, NOUT) in spikes/s, list of (probe, cluster idx)).
    """
    rates, ids = [], []
    keep_idx = {t: i for i, t in enumerate(keep_trials)}   # 0-based trial -> row
    nkeep = len(keep_trials)
    for p in probes:
        c = clu.get(p)
        if c is None:
            continue
        good = [i for i, q in enumerate(c['quality'])
                if q.strip().lower() not in BAD_QUALITY]          # findClusters.m
        for i in good:
            tr = c['trial'][i].astype(np.int64) - 1               # -> 0-based
            tm = c['trialtm'][i]
            ok = np.array([t in keep_idx for t in tr]) if tr.size else np.zeros(0, bool)
            if tr.size:
                row = np.full(tr.shape, -1, dtype=np.int64)
                for j in np.nonzero(ok)[0]:
                    row[j] = keep_idx[tr[j]]
            else:
                row = np.zeros(0, dtype=np.int64)
            counts = np.zeros((nkeep, NFINE), dtype=np.float32)
            if tr.size:
                sel = ok & np.isfinite(tm)
                if sel.any():
                    aligned = tm[sel] - align_times[tr[sel]]
                    b = np.floor((aligned - TMIN) / DT_FINE).astype(np.int64)
                    inwin = (b >= 0) & (b < NFINE)
                    if inwin.any():
                        flat = row[sel][inwin] * NFINE + b[inwin]
                        np.add.at(counts.reshape(-1), flat, 1.0)
            rate = counts / DT_FINE                                # spikes / s
            rate = my_smooth(rate)                                 # causal gaussian
            rates.append(downsample_mean(rate).astype(np.float32))
            ids.append((p, i))
    if not rates:
        return np.zeros((nkeep, 0, NOUT), np.float32), []
    return np.stack(rates, axis=1), ids

# --------------------------------------------------------------------------------------
# Kinematics / motion energy
# --------------------------------------------------------------------------------------
def feature_speed(sess, key, keep_trials, vidshift, align_times, fill_missing):
    """Speed of a DLC feature on the fine time axis, NaN where the feature is not visible.

    Mirrors funcs/kinematics/findPosition.m + findVelocity.m:
      * interp1(frameTimes - vidshift - alignEvent, [x y], taxis)
      * tongue: no smoothing, NaNs kept (not visible)
      * other features: nearest-fill + baseline-derivative subtraction
    """
    tr = sess['traj'].get(key)
    nkeep = len(keep_trials)
    speed = np.full((nkeep, NFINE), np.nan)
    visible = np.zeros((nkeep, NFINE), bool)
    if tr is None:
        return speed, visible
    nd = tr['NdroppedFrames']
    for r, t in enumerate(keep_trials):
        if t >= len(tr['xy']):
            continue
        if t < len(nd) and np.isnan(nd[t]):      # reference skips these trials
            continue
        ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
        xy = tr['xy'][t]
        if ft.size != xy.shape[0] or ft.size < 2:
            m = min(ft.size, xy.shape[0])
            if m < 2:
                continue
            ft, xy = ft[:m], xy[:m]
        valid = np.isfinite(ft)
        if valid.sum() < 2:
            continue
        pos = np.empty((NFINE, 2))
        for d in range(2):
            pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], left=np.nan, right=np.nan)
        vis = np.isfinite(pos).all(axis=1)
        if fill_missing:
            # nearest-neighbour fill (fillmissing(...,'nearest')) as in findPosition.m
            if vis.any():
                idx = np.arange(NFINE)
                pos = np.stack([np.interp(idx, idx[vis], pos[vis, d]) for d in range(2)], axis=1)
            else:
                continue
        vel = np.gradient(pos, axis=0)
        if fill_missing:
            base = np.nanmedian(np.diff(pos, axis=0), axis=0)   # findVelocity.m
            vel = vel - base[0]
        sp = np.hypot(vel[:, 0], vel[:, 1])
        sp[~vis] = np.nan                                       # not visible
        speed[r] = sp
        visible[r] = vis
    return speed, visible

def motion_energy_trials(me, sess, keep_trials, vidshift, align_times):
    """DataLoadingScripts/loadMotionEnergy.m: interpolate 400 Hz ME onto the trial time axis."""
    nkeep = len(keep_trials)
    out = np.full((nkeep, NFINE), np.nan)
    if me is None:
        return out
    for r, t in enumerate(keep_trials):
        if t >= len(me['data']):
            continue
        y = me['data'][t]
        ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
        m = min(ft.size, y.size)
        if m < 2:
            continue
        ft, y = ft[:m], y[:m]
        valid = np.isfinite(ft) & np.isfinite(y)
        if valid.sum() < 2:
            continue
        v = np.interp(TCENT, ft[valid], y[valid], left=np.nan, right=np.nan)
        # fillmissing(...,'nearest')
        ok = np.isfinite(v)
        if not ok.all() and ok.any():
            idx = np.arange(NFINE)
            v = np.interp(idx, idx[ok], v[ok])
        out[r] = v
    return out

def discretize(values, nan_class=2):
    """0 = < 50th percentile, 1 = >= 50th percentile (per session), 2 = not visible/no video."""
    finite = np.isfinite(values)
    cls = np.full(values.shape, nan_class, dtype=np.int64)
    if finite.any():
        thr = np.percentile(values[finite], 50)
        cls[finite & (values >= thr)] = 1
        cls[finite & (values < thr)] = 0
    else:
        thr = np.nan
    return cls, thr

# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def region_of_probe(sess, probe):
    """Brain region label for a probe, from obj.ex.probe.loc (default ALM: all sessions
    come from the reference's *_ALMVideo loaders)."""
    locs = sess.get('probe_loc', [])
    if len(locs) >= probe:
        s = locs[probe - 1].upper()
        if 'ALM' in s:
            return 'ALM'
        if 'M1TJ' in s or 'TJM1' in s:
            return 'tjM1'
    return 'ALM'

def process_session(args):
    anm, date, probes, ddir, task, show = args
    t0 = time.time()
    fn = os.path.join(ddir, 'data_structure_%s_%s.mat' % (anm, date))
    feats = [TONGUE_FEAT] + PAW_FEATS
    sess = load_session(fn, probes, feats)
    t_load = time.time() - t0

    bp = sess['bp']
    N = int(bp['Ntrials'][0])
    ev = bp['ev']
    align = ev[ALIGN_EVENT].astype(float)
    hit = np.nan_to_num(bp['hit']).astype(bool)
    miss = np.nan_to_num(bp['miss']).astype(bool)
    no = np.nan_to_num(bp['no']).astype(bool)
    R = np.nan_to_num(bp['R']).astype(bool)
    L = np.nan_to_num(bp['L']).astype(bool)
    early = np.nan_to_num(bp['early']).astype(bool)
    aw = np.nan_to_num(bp['autowater']).astype(bool)
    stim = (np.nan_to_num(bp['stim_enable']).astype(bool)
            if bp['stim_enable'] is not None else np.zeros(N, bool))
    for a in (hit, miss, no, R, L, early, aw, stim):
        assert a.size >= N

    # ---- trial curation: reference conditions use ~stim.enable & ~early -------------
    valid_align = np.isfinite(align[:N])
    keep = (~stim[:N]) & (~early[:N]) & valid_align
    keep_trials = np.nonzero(keep)[0]
    if keep_trials.size < MIN_TRIALS:
        return dict(ok=False, reason='too few trials', sess='%s_%s' % (anm, date))

    # ---- neural --------------------------------------------------------------------
    t1 = time.time()
    rates, ids = bin_spikes(sess['clu'], probes, align, keep_trials, N)
    t_neural = time.time() - t1
    if rates.shape[1] == 0:
        return dict(ok=False, reason='no units', sess='%s_%s' % (anm, date))
    # Drop trials with no ephys coverage: in a few sessions (e.g. JEB24_2023-10-23) the
    # SpikeGLX recording ends before the behavioural session, so the trailing trials contain
    # no spikes from ANY unit (impossible over a 5 s window with tens of simultaneously
    # recorded units unless the probe was no longer recording).
    has_spikes = rates.sum(axis=(1, 2)) > 0
    n_noephys = int((~has_spikes).sum())
    if n_noephys:
        rates = rates[has_spikes]
        keep_trials = keep_trials[has_spikes]
        if keep_trials.size < MIN_TRIALS:
            return dict(ok=False, reason='too few trials with ephys', sess='%s_%s' % (anm, date))
    # removeLowFRClusters.m: mean firing rate across trials & time must exceed lowFR
    mean_fr = rates.mean(axis=(0, 2))
    use = mean_fr > LOW_FR
    rates = rates[:, use, :]
    ids = [c for c, u in zip(ids, use) if u]
    if rates.shape[1] < MIN_UNITS:
        return dict(ok=False, reason='fewer than %d units (%d)' % (MIN_UNITS, rates.shape[1]),
                    sess='%s_%s' % (anm, date))

    # ---- per-trial task variables ---------------------------------------------------
    kt = keep_trials
    lick_dir = np.full(kt.size, 2, dtype=np.int64)          # 2 = none (ignore trials)
    right_lick = (R[kt] & hit[kt]) | (L[kt] & miss[kt])     # funcs/getPrevChoice.m
    left_lick = (L[kt] & hit[kt]) | (R[kt] & miss[kt])
    lick_dir[right_lick] = 1
    lick_dir[left_lick] = 0
    context = np.where(aw[kt], 0, 1).astype(np.int64)       # 0 = WC (autowater), 1 = DR
    outcome = np.full(kt.size, 2, dtype=np.int64)           # 2 = ignore
    outcome[hit[kt]] = 1
    outcome[miss[kt]] = 0

    # ---- video-derived outputs ------------------------------------------------------
    t2 = time.time()
    vidshift = video_offset(sess)
    tongue_sp, _ = feature_speed(sess, TONGUE_FEAT, kt, vidshift, align, fill_missing=False)
    paw_sps = [feature_speed(sess, k, kt, vidshift, align, fill_missing=True)[0] for k in PAW_FEATS]
    paw_sp = np.nanmean(np.stack(paw_sps, axis=0), axis=0) if paw_sps else np.full_like(tongue_sp, np.nan)
    mefn = os.path.join(ddir, 'motionEnergy_%s_%s.mat' % (anm, date))
    me = load_motion_energy(mefn) if os.path.exists(mefn) else None
    me_fine = motion_energy_trials(me, sess, kt, vidshift, align)
    t_video = time.time() - t2

    tongue_ds = downsample_mean(tongue_sp, nanmean=True)
    paw_ds = downsample_mean(paw_sp, nanmean=True)
    me_ds = downsample_mean(me_fine, nanmean=True)
    tongue_cls, tongue_thr = discretize(tongue_ds)
    paw_cls, paw_thr = discretize(paw_ds)
    me_cls, me_thr = discretize(me_ds)

    # ---- assemble -------------------------------------------------------------------
    neural = [np.ascontiguousarray(rates[i]).astype(np.float32) for i in range(kt.size)]
    tin = TOUT.astype(np.float32)[None, :]
    inputs = [tin.copy() for _ in range(kt.size)]
    outputs = []
    for i in range(kt.size):
        o = np.empty((6, NOUT), dtype=np.int64)
        o[0] = lick_dir[i]
        o[1] = context[i]
        o[2] = outcome[i]
        o[3] = tongue_cls[i]
        o[4] = paw_cls[i]
        o[5] = me_cls[i]
        outputs.append(o)
    regions = [region_of_probe(sess, p) for p, _ in ids]

    info = dict(ok=True, sess='%s_%s' % (anm, date), anm=anm, date=date, task=task,
                probes=probes, ntrials_total=N, ntrials_kept=int(kt.size),
                nunits_prefilter=int(use.size), nunits=int(rates.shape[1]),
                nstim=int(stim[:N].sum()), nearly=int(early[:N].sum()), n_noephys=n_noephys,
                vidshift=float(vidshift), me_thresh_session=(None if me is None else me['moveThresh']),
                thr=dict(tongue=float(tongue_thr) if np.isfinite(tongue_thr) else None,
                         paw=float(paw_thr) if np.isfinite(paw_thr) else None,
                         me=float(me_thr) if np.isfinite(me_thr) else None),
                frac_hit=float(hit[kt].mean()), frac_wc=float(aw[kt].mean()),
                timing=dict(load=t_load, neural=t_neural, video=t_video, total=time.time() - t0))

    result = dict(ok=True, neural=neural, input=inputs, output=outputs,
                  regions=regions, anm=anm, info=info)
    if show:
        plot_processing(sess, kt, align, rates, tongue_sp, paw_sp, me_fine,
                        tongue_ds, paw_ds, me_ds, tongue_cls, paw_cls, me_cls,
                        lick_dir, context, outcome, '%s_%s' % (anm, date), me)
    return result

# --------------------------------------------------------------------------------------
# Diagnostic plots (--show-processing)
# --------------------------------------------------------------------------------------
def plot_processing(sess, kt, align, rates, tongue_sp, paw_sp, me_fine,
                    tongue_ds, paw_ds, me_ds, tongue_cls, paw_cls, me_cls,
                    lick_dir, context, outcome, sessname, me):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bp = sess['bp']
    ev = bp['ev']
    fig, ax = plt.subplots(5, 2, figsize=(16, 18))

    # 1. spike raster + binned rate for one neuron/trial
    itr = 0
    tnum = kt[itr]
    probe = list(sess['clu'].keys())[0]
    c = sess['clu'][probe]
    iu = 0
    for i, q in enumerate(c['quality']):
        if q.strip().lower() not in BAD_QUALITY:
            iu = i
            break
    trm = c['trial'][iu].astype(int) - 1
    tm = c['trialtm'][iu]
    spk = tm[trm == tnum] - align[tnum]
    ax[0, 0].eventplot([spk], colors='k')
    ax[0, 0].set_title('%s: aligned spikes, probe %d unit %d, trial %d' % (sessname, probe, iu, tnum))
    ax[0, 0].set_xlim(TMIN, TMAX); ax[0, 0].axvline(0, color='r')
    ax[0, 0].set_xlabel('time from go cue (s)')

    ax[0, 1].plot(TOUT, rates[itr].mean(axis=0), 'k')
    ax[0, 1].axvline(0, color='r')
    ax[0, 1].set_title('population mean rate (spk/s), trial %d, %d units' % (tnum, rates.shape[1]))
    ax[0, 1].set_xlabel('time from go cue (s)')

    # 2. PSTH by lick direction (sanity: choice selectivity)
    for lab, name, col in [(1, 'right', 'b'), (0, 'left', 'r')]:
        m = lick_dir == lab
        if m.sum() > 2:
            ax[1, 0].plot(TOUT, rates[m].mean(axis=(0, 1)), col, label='%s (n=%d)' % (name, m.sum()))
    ax[1, 0].legend(); ax[1, 0].axvline(0, color='k', ls='--')
    ax[1, 0].set_title('population PSTH by lick direction'); ax[1, 0].set_xlabel('time from go cue (s)')

    # 3. lick raster vs tongue visibility
    nshow = min(40, kt.size)
    for r in range(nshow):
        t = kt[r]
        for arr, col in [(ev['lickR'][t], 'b'), (ev['lickL'][t], 'r')]:
            if arr.size:
                x = arr - align[t]
                x = x[(x > TMIN) & (x < TMAX)]
                ax[1, 1].plot(x, np.full(x.size, r), col + '.', ms=3)
        vis = np.isfinite(tongue_sp[r])
        ax[1, 1].plot(TCENT[vis], np.full(vis.sum(), r), 'g.', ms=1, alpha=0.3)
    ax[1, 1].axvline(0, color='k')
    ax[1, 1].set_title('lick times (blue=R, red=L) and tongue visible (green)')
    ax[1, 1].set_xlabel('time from go cue (s)'); ax[1, 1].set_xlim(TMIN, TMAX)

    # 4. tongue speed: fine vs binned vs discretized
    ax[2, 0].plot(TCENT, tongue_sp[itr], 'k.-', ms=2, label='fine (10 ms)')
    ax[2, 0].plot(TOUT, tongue_ds[itr], 'c.-', label='binned (50 ms)')
    ax[2, 0].axvline(0, color='r')
    ax[2, 0].legend(); ax[2, 0].set_title('tongue speed, trial %d (NaN = not visible)' % tnum)
    ax[2, 1].imshow(tongue_cls[:nshow], aspect='auto', interpolation='nearest',
                    extent=[TOUT[0], TOUT[-1], nshow, 0], vmin=0, vmax=2)
    ax[2, 1].set_title('tongue_velocity class (0 slow, 1 fast, 2 not visible)')

    # 5. paw + ME
    ax[3, 0].plot(TCENT, paw_sp[itr], 'k', label='paw speed (fine)')
    ax[3, 0].plot(TOUT, paw_ds[itr], 'c.-', label='binned')
    ax[3, 0].axvline(0, color='r'); ax[3, 0].legend(); ax[3, 0].set_title('paw speed, trial %d' % tnum)
    ax[3, 1].imshow(paw_cls[:nshow], aspect='auto', interpolation='nearest',
                    extent=[TOUT[0], TOUT[-1], nshow, 0], vmin=0, vmax=2)
    ax[3, 1].set_title('paw_velocity class')

    ax[4, 0].plot(TCENT, me_fine[itr], 'k', label='motion energy (fine)')
    ax[4, 0].plot(TOUT, me_ds[itr], 'c.-', label='binned')
    if me is not None:
        ax[4, 0].axhline(me['moveThresh'], color='m', ls=':', label='session moveThresh')
    ax[4, 0].axvline(0, color='r'); ax[4, 0].legend(); ax[4, 0].set_title('motion energy, trial %d' % tnum)
    ax[4, 1].imshow(me_cls[:nshow], aspect='auto', interpolation='nearest',
                    extent=[TOUT[0], TOUT[-1], nshow, 0], vmin=0, vmax=2)
    ax[4, 1].set_title('motion_energy class')
    for a in ax.ravel():
        a.set_xlabel('time from go cue (s)')
    fig.suptitle('Processing steps: %s  (outputs: lickdir %s, context %s, outcome %s)'
                 % (sessname, np.bincount(lick_dir, minlength=3),
                    np.bincount(context, minlength=2), np.bincount(outcome, minlength=3)))
    fig.tight_layout()
    fig.savefig('processing_%s.png' % sessname, dpi=110)
    plt.close(fig)

    # second figure: event-time sanity (sample / delay / go cue) and trial-average traces
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    smp = ev['sample'][kt] - align[kt]
    dly = ev['delay'][kt] - align[kt]
    ax[0].plot(smp, np.arange(kt.size), 'g.', ms=3, label='sample onset')
    ax[0].plot(dly, np.arange(kt.size), 'b.', ms=3, label='delay onset')
    ax[0].axvline(0, color='r', label='go cue (alignment)')
    ax[0].set_xlim(TMIN, TMAX); ax[0].legend(); ax[0].set_xlabel('time from go cue (s)')
    ax[0].set_ylabel('trial'); ax[0].set_title('event times relative to alignment event')
    ax[1].plot(TOUT, np.nanmean(me_ds, axis=0), 'k', label='mean motion energy')
    ax[1].plot(TOUT, 100 * np.mean(tongue_cls != 2, axis=0), 'g', label='% trials tongue visible')
    ax[1].axvline(0, color='r'); ax[1].legend(); ax[1].set_xlabel('time from go cue (s)')
    ax[1].set_title('trial-averaged movement signals')
    fig.tight_layout()
    fig.savefig('processing_%s_alignment.png' % sessname, dpi=110)
    plt.close(fig)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nproc', type=int, default=12)
    args = ap.parse_args()

    sessions = SESSIONS[:2] if args.sample else SESSIONS
    show = args.show_processing
    jobs = [(a, d, p, dd, tk, show and i < 2) for i, (a, d, p, dd, tk) in enumerate(sessions)]

    print('Converting %d sessions (%s mode)' % (len(jobs), 'sample' if args.sample else 'full'))
    t0 = time.time()
    if args.nproc > 1 and len(jobs) > 1:
        with Pool(min(args.nproc, len(jobs))) as pool:
            results = pool.map(process_session, jobs)
    else:
        results = [process_session(j) for j in jobs]
    print('All sessions processed in %.1f s' % (time.time() - t0))

    data = dict(neural=[], input=[], output=[], subjects=[], subject_idx=[],
                brain_regions=['ALM', 'tjM1'], brain_region_idx=[])
    session_info = []
    for r in results:
        if not r.get('ok'):
            print('  EXCLUDED %s: %s' % (r['sess'], r['reason']))
            continue
        info = r['info']
        print('  %s (%s): %d units, %d/%d trials, load %.1fs neural %.1fs video %.1fs total %.1fs'
              % (info['sess'], info['task'], info['nunits'], info['ntrials_kept'],
                 info['ntrials_total'], info['timing']['load'], info['timing']['neural'],
                 info['timing']['video'], info['timing']['total']))
        if r['anm'] not in data['subjects']:
            data['subjects'].append(r['anm'])
        data['subject_idx'].append(data['subjects'].index(r['anm']))
        data['neural'].append(r['neural'])
        data['input'].append(r['input'])
        data['output'].append(r['output'])
        data['brain_region_idx'].append(
            np.array([data['brain_regions'].index(x) for x in r['regions']], dtype=np.int64))
        session_info.append(info)

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['input_names'] = ['time_from_go_cue']
    data['output_names'] = ['lick_direction', 'context', 'outcome',
                            'tongue_velocity', 'paw_velocity', 'motion_energy']
    data['output_values'] = [
        ['left', 'right', 'none'],
        ['WC', 'DR'],
        ['incorrect', 'correct', 'ignore'],
        ['below_median', 'above_median', 'not_visible'],
        ['below_median', 'above_median', 'not_visible'],
        ['below_median', 'above_median', 'no_video'],
    ]
    data['metadata'] = dict(
        task_description=(
            'Head-fixed mice alternate block-wise between a delayed-response (DR) task '
            '(auditory sample tone instructs left/right lick direction, delay epoch, auditory go cue) '
            'and a water-cued (WC) task (water delivered at a random time at a random port, no auditory cues). '
            'Neural activity is single-unit spiking recorded with Neuropixels/H2 probes in ALM (and tjM1 on a '
            'few dual-probe sessions). Decoder outputs: lick direction (left/right/none), behavioral context '
            '(WC/DR), trial outcome (incorrect/correct/ignore), and median-split discretized tongue speed, '
            'paw speed and motion energy (class 2 = feature not visible / no video).'),
        time_bin_size=DT_OUT * 1000.0,
        temporal_alignment_event='go cue onset (auditory go cue on DR trials, water drop on WC trials)',
        off_start=TMIN,
        off_end=TMIN + NOUT * DT_OUT,
        dataset='Hasnain, Birnbaum et al., Nature Neuroscience 2024 - '
                '"Separating cognitive and motor processes in the behaving mouse"',
        neural_processing=('spikes aligned to the go cue, binned at %d ms, converted to spikes/s and smoothed '
                           'with a causal Gaussian kernel (mySmooth, N=%d), then averaged into %d ms bins'
                           % (DT_FINE * 1000, SMOOTH_N, DT_OUT * 1000)),
        neuron_curation=('cluster quality not in %s (findClusters.m) and mean firing rate > %g Hz '
                         '(removeLowFRClusters.m); sessions require >= %d units'
                         % (sorted(BAD_QUALITY), LOW_FR, MIN_UNITS)),
        trial_curation='photostimulation trials (stim.enable) and early-lick trials removed; '
                       'ignore (no-response) trials retained as an output class',
        video='DeepLabCut tracking at 400 Hz aligned with findVideoOffset.m; motion energy from '
              'motionEnergy_*.mat interpolated onto the same time axis',
        discretization='tongue/paw speed and motion energy split at the per-session 50th percentile of '
                       'visible values; not-visible/no-video bins assigned class 2',
        session_info=session_info,
    )

    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh)

    nses = len(data['neural'])
    ntr = sum(len(s) for s in data['neural'])
    nun = sum(s[0].shape[0] for s in data['neural'])
    print('\nSaved %s' % args.outfile)
    print('  sessions %d, subjects %d, trials %d, neurons %d, timepoints %d, bin %g ms'
          % (nses, len(data['subjects']), ntr, nun, NOUT, DT_OUT * 1000))
    print('  total time %.1f s' % (time.time() - t0))

if __name__ == '__main__':
    main()
