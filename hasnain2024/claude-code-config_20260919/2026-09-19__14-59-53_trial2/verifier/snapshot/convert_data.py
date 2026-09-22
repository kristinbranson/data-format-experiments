#!/usr/bin/env python3
"""
Convert the ALM electrophysiology + high-speed-video dataset of

    Hasnain, Birnbaum et al. (2024/2025)
    "Separating cognitive and motor processes in the behaving mouse"
    Nature Neuroscience, data DOI 10.5281/zenodo.13941415

into the pickle format expected by ``train_decoder.py`` / ``decoder.py``.

The processing deliberately mirrors the authors' MATLAB pipeline in ``/app/code``
(see CONVERSION_NOTES.md for a function-by-function mapping):

  loadObjs        -> ``load_obj``
  findClusters    -> ``select_clusters``     (drop garbage / noisy clusters)
  findTrials      -> ``select_trials``       (drop early-lick and photostim trials)
  alignSpikes     -> spike times minus ``bp.ev.goCue``
  getSeq          -> ``bin_and_smooth``      (10 ms bins, [-2.5, 2.5] s, spikes/s)
  mySmooth        -> ``causal_gaussian``     (gausswin(15), causal, 'reflect')
  removeLowFRClu  -> mean rate > 1 Hz
  findVideoOffset -> ``video_shift``
  findPosition    -> ``interp_feature``
  findVelocity    -> ``feature_speed``
  loadMotionEnergy-> ``load_motion_energy`` + ``interp_trace``

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Reference parameters (identical to every analysis script in /app/code)
# --------------------------------------------------------------------------------------
ALIGN_EVENT = "goCue"
TMIN, TMAX, DT = -2.5, 2.5, 1.0 / 100.0   # params.tmin / params.tmax / params.dt
SMOOTH_N = 15                              # params.smooth (samples, causal gaussian)
BCTYPE = "reflect"                         # params.bctype
LOW_FR = 1.0                               # params.lowFR (Hz)
ADVANCE_MOVEMENT = 0.0                     # params.advance_movement
MIN_UNITS = 10                             # Methods: sessions need >= 10 units
QUALITY_REJECT = ("garbage", "gabrga", "noisy", "real?")  # findClusters(..., 'all')

EDGES = np.round(np.arange(TMIN, TMAX + DT / 2.0, DT), 10)   # 501 bin edges
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)            # 500 bin centres
NT = TIME.size

# DLC features used for the discretised kinematic outputs
TONGUE_VIEW, TONGUE_FEAT = 0, "tongue"            # side camera
PAW_VIEW, PAW_FEATS = 1, ("top_paw", "bottom_paw")  # bottom camera (paws only tracked there)

BRAIN_REGIONS = ["ALM"]

OUTPUT_NAMES = [
    "lick_direction",
    "context",
    "outcome",
    "tongue_velocity",
    "paw_velocity",
    "motion_energy",
]
OUTPUT_VALUES = [
    ["left", "right", "none"],
    ["WC", "DR"],
    ["incorrect", "correct", "ignore"],
    ["below_median", "above_median", "not_visible"],
    ["below_median", "above_median", "not_visible"],
    ["below_median", "above_median", "no_video"],
]
INPUT_NAMES = ["time_from_go_cue"]

# --------------------------------------------------------------------------------------
# Session list -- taken verbatim from DataLoadingScripts/'Recording and video'/load*_ALMVideo.m
# (anm, date, ALM probe number(s) (1-based), task)
# --------------------------------------------------------------------------------------
FIXED_DELAY = [
    ("JEB6", "2021-04-18", [2]),
    ("JEB7", "2021-04-29", [1]),
    ("JEB7", "2021-04-30", [1]),
    ("EKH1", "2021-08-07", [2]),
    ("EKH3", "2021-08-11", [2]),
    ("JGR2", "2021-11-16", [1]),
    ("JGR2", "2021-11-17", [1]),
    ("JGR3", "2021-11-18", [1]),
    ("JEB19", "2023-04-18", [1]),
    ("JEB19", "2023-04-19", [1]),
    ("JEB19", "2023-04-20", [1]),
    ("JEB19", "2023-04-21", [1]),
    ("JEB13", "2022-09-13", [2]),
    ("JEB13", "2022-09-14", [2]),
    ("JEB13", "2022-09-21", [1]),
    ("JEB13", "2022-09-24", [1]),
    ("JEB13", "2022-09-25", [1]),
    ("JEB14", "2022-08-22", [1]),
    ("JEB14", "2022-08-23", [1]),
    ("JEB14", "2022-08-24", [1]),
    ("JEB14", "2022-08-25", [1]),
    ("JEB15", "2022-07-26", [1, 2]),
    ("JEB15", "2022-07-27", [1, 2]),
    ("JEB15", "2022-07-28", [1, 2]),
    ("JEB15", "2022-07-29", [2]),
]
RANDOMIZED_DELAY = (
    [("JEB11", "2022-05-10", [1]), ("JEB11", "2022-05-11", [1]),
     ("JEB12", "2022-05-12", [1]), ("JEB12", "2022-05-13", [1])]
    + [("JEB23", d, [1]) for d in
       ("2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13",
        "2023-10-18", "2023-10-19", "2023-10-21")]
    + [("JEB24", d, [1]) for d in
       ("2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26",
        "2023-10-27", "2023-10-31", "2023-11-02", "2023-11-03")]
)
SESSIONS = ([(a, d, p, "fixed") for a, d, p in FIXED_DELAY]
            + [(a, d, p, "randomized") for a, d, p in RANDOMIZED_DELAY])

DATA_DIR = {"fixed": "/app/data/Ephys_Behavior",
            "randomized": "/app/data/RandomizedDelay_Ephys_Behavior"}

# Two-context (DR + water-cued blocks) sessions, identified from the block structure of
# bp.autowater and cross-checked with the paper (12 sessions).  Recorded for bookkeeping only.
TWO_CONTEXT = {
    ("JEB6", "2021-04-18"), ("JEB7", "2021-04-29"), ("JEB7", "2021-04-30"),
    ("EKH1", "2021-08-07"), ("EKH3", "2021-08-11"),
    ("JGR2", "2021-11-16"), ("JGR2", "2021-11-17"), ("JGR3", "2021-11-18"),
    ("JEB19", "2023-04-18"), ("JEB19", "2023-04-19"),
    ("JEB19", "2023-04-20"), ("JEB19", "2023-04-21"),
}


# --------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------
def data_path(anm, date, task):
    return os.path.join(DATA_DIR[task], f"data_structure_{anm}_{date}.mat")


def me_path(anm, date, task):
    return os.path.join(DATA_DIR[task], f"motionEnergy_{anm}_{date}.mat")


def load_obj(anm, date, task):
    """loadObjs.m -- read one session's data object (handles MAT v7 and v7.3)."""
    from pymatreader import read_mat
    return read_mat(data_path(anm, date, task))["obj"]


def vec(x):
    """Flatten a MATLAB numeric field to a 1-D float array."""
    return np.ravel(np.asarray(x, dtype=np.float64))


def mode_of(a):
    """MATLAB mode() for a numeric vector (smallest most-frequent value), NaNs dropped."""
    a = a[~np.isnan(a)]
    vals, counts = np.unique(a, return_counts=True)
    return float(vals[np.argmax(counts)])


def causal_gaussian(n=SMOOTH_N, alpha=2.5):
    """mySmooth.m kernel: gausswin(n) with the first floor(n/2) taps zeroed, normalised."""
    k = np.arange(n)
    w = np.exp(-0.5 * (alpha * (k - (n - 1) / 2.0) / ((n - 1) / 2.0)) ** 2)
    w[: n // 2] = 0.0
    return w / w.sum()


KERNEL = causal_gaussian()


def my_smooth(x, kern=KERNEL, bctype=BCTYPE):
    """mySmooth(x, N, bctype) for x of shape (T, M); operates on the first dimension.

    Reproduces `conv(..., 'same')` with an odd, causal kernel: because taps 0..n//2-1 are
    zero, the 'same' convolution reduces to a causal FIR filter with n - n//2 taps, which
    is evaluated directly (exact, no FFT round-off).
    """
    n = kern.size
    half = (n - 1) // 2
    if bctype == "reflect":
        xp = np.concatenate([x[:n], x], axis=0)
        trim = n
    elif bctype == "zeropad":
        xp = np.concatenate([np.zeros((n,) + x.shape[1:], dtype=x.dtype), x], axis=0)
        trim = n
    else:
        xp, trim = x, 0
    out = np.zeros_like(xp, dtype=np.float64)
    # conv(...,'same')[t] = sum_k xp[t + half - k] * kern[k]; kern[k]==0 for k < half
    for k in range(half, n):
        w = kern[k]
        if w == 0.0:
            continue
        j = k - half                      # delay in samples (>= 0)
        if j == 0:
            out += w * xp
        else:
            out[j:] += w * xp[:-j]
    return out[trim:]


def bin_index(t):
    """histc(t, EDGES) bin index; -1 (or >= NT) when outside the analysis window."""
    idx = np.searchsorted(EDGES, t, side="right") - 1
    idx[(idx < 0) | (idx >= NT)] = -1
    return idx


def video_shift(obj):
    """findVideoOffset.m -- seconds to subtract from traj frameTimes to get trial time."""
    bit_start = mode_of(vec(obj["bp"]["ev"]["bitStart"]))
    fs = float(vec(obj["sglx"]["fs"])[0])
    vid_file_offset = mode_of(vec(obj["sglx"]["bitcode"]["bitstart"])) / fs
    return vid_file_offset - bit_start


def as_cell_list(x, n):
    """Normalise a MATLAB cell-array field to a python list with n entries.

    pymatreader returns a list for a cell array with >1 element but the bare content
    when the cell array has exactly one element, so single-trial / single-cluster
    fields have to be wrapped.
    """
    if isinstance(x, list):
        return x
    return [x] * n if n == 1 else [x] + [None] * (n - 1)


# kept for readability at the call sites
as_trial_list = as_cell_list


def frame_times(obj, view, itrial):
    """frameTimes for one trial, or None when the video timing is unusable."""
    ft = obj["traj"][view]["frameTimes"]
    ft = ft[itrial] if isinstance(ft, list) else ft
    if ft is None:
        return None
    a = np.ravel(np.asarray(ft, dtype=np.float64))
    if a.size < 2 or not np.all(np.isfinite(a)):
        return None
    return a


def load_motion_energy(anm, date, task, obj, ntrials):
    """loadMotionEnergy.m -- list of per-trial 400 Hz motion-energy traces (or None)."""
    from pymatreader import read_mat

    me = None
    p = me_path(anm, date, task)
    if os.path.exists(p):
        me = read_mat(p)["me"]
    elif "me" in obj:
        me = obj["me"]
    if me is None:
        return [None] * ntrials
    if isinstance(me, dict):                      # struct with .data (+ .moveThresh)
        me = me["data"]
        if isinstance(me, dict):                  # me.data.data (older objects)
            me = me["data"]
    me = as_trial_list(me, ntrials)
    out = []
    for i in range(ntrials):
        if i >= len(me) or me[i] is None:
            out.append(None)
            continue
        a = np.ravel(np.asarray(me[i], dtype=np.float64))
        out.append(a if a.size > 1 else None)
    return out


# --------------------------------------------------------------------------------------
# Curation (findClusters.m / findTrials.m conditions)
# --------------------------------------------------------------------------------------
def select_clusters(obj, probes):
    """findClusters(quality, {'all'}): keep everything except garbage / noisy clusters.

    Returns a list of (probe_index, cluster_index) for the designated ALM probe(s).
    The label comparison is case-sensitive, exactly as in the MATLAB `ismember` calls.
    """
    clu = obj["clu"]
    if isinstance(clu, dict):
        clu = [clu]
    keep = []
    for p in probes:
        c = clu[p - 1]
        if not isinstance(c, dict) or "quality" not in c:
            continue
        quals = c["quality"]
        if not isinstance(quals, list):
            quals = [quals]
        for i, q in enumerate(quals):
            q = str(q).strip() if isinstance(q, (str, np.str_)) else ""
            if q not in QUALITY_REJECT:
                keep.append((p - 1, i))
    return keep


def trials_with_ephys(obj, probes, ntrials):
    """Boolean mask of trials for which spikes were actually recorded.

    In a few sessions the SpikeGLX recording stops before the behavioural session does
    (JEB24 2023-10-23 and 2023-11-03 lose the last ~30 trials, JEB12 2022-05-12 loses one
    trial, which `obj.trials.bp.haveEphys` also flags).  Those trials carry an all-zero
    neural matrix, which is not data, so they are dropped.  Detection uses *every* cluster
    on the probe, including the ones the quality filter rejects, so a trial is only
    declared ephys-less when literally no spike was sorted on it.
    """
    clu = obj["clu"]
    if isinstance(clu, dict):
        clu = [clu]
    has = np.zeros(ntrials, dtype=bool)
    for p in probes:
        c = clu[p - 1]
        if not isinstance(c, dict) or "trial" not in c:
            continue
        trialno = c["trial"] if isinstance(c["trial"], list) else [c["trial"]]
        for t in trialno:
            if t is None:
                continue
            a = np.ravel(np.asarray(t, dtype=np.float64))
            a = a[np.isfinite(a)].astype(np.int64) - 1
            a = a[(a >= 0) & (a < ntrials)]
            has[a] = True
    return has


def select_trials(bp, obj=None, probes=()):
    """Trial curation.

    - `findTrials` conditions `~stim.enable & ~early`: drop photoinactivation and
      early-lick trials (Methods: early-lick trials "were omitted from analyses").
    - drop trials with no electrophysiology (see `trials_with_ephys`).
    hit / miss / ignore and DR / WC trials are all kept, as the decoder outputs require.
    """
    n = int(vec(bp["Ntrials"])[0])
    early = vec(bp["early"])[:n] > 0
    stim = vec(bp["stim"]["enable"])[:n] > 0
    keep = ~(early | stim)
    if obj is not None:
        keep &= trials_with_ephys(obj, probes, n)
    return np.flatnonzero(keep), n


# --------------------------------------------------------------------------------------
# Neural
# --------------------------------------------------------------------------------------
def bin_and_smooth(obj, cluster_ids, align_times, ntrials):
    """getSeq.m -- (ntrials, nclusters, NT) smoothed firing rates in spikes/s."""
    clu = obj["clu"]
    if isinstance(clu, dict):
        clu = [clu]
    counts = np.zeros((len(cluster_ids), ntrials, NT), dtype=np.float32)
    for ci, (p, i) in enumerate(cluster_ids):
        c = clu[p]
        trialtm = c["trialtm"] if isinstance(c["trialtm"], list) else [c["trialtm"]]
        trialno = c["trial"] if isinstance(c["trial"], list) else [c["trial"]]
        tm = np.ravel(np.asarray(trialtm[i], dtype=np.float64))
        tr = np.ravel(np.asarray(trialno[i], dtype=np.float64)).astype(np.int64) - 1
        ok = (tr >= 0) & (tr < ntrials) & np.isfinite(tm)
        tm, tr = tm[ok], tr[ok]
        if tm.size == 0:
            continue
        aligned = tm - align_times[tr]              # alignSpikes.m
        b = bin_index(aligned)
        ok = b >= 0
        if not np.any(ok):
            continue
        flat = tr[ok] * NT + b[ok]
        counts[ci] = np.bincount(flat, minlength=ntrials * NT).reshape(ntrials, NT)
    rate = counts / DT                               # spikes/s
    # smooth along time: my_smooth wants (T, M)
    m = rate.reshape(-1, NT).T                       # (NT, nclu*ntrials)
    m = my_smooth(m)
    return m.T.reshape(len(cluster_ids), ntrials, NT)


# --------------------------------------------------------------------------------------
# Video-derived outputs
# --------------------------------------------------------------------------------------
def interp_trace(src_t, src_y, taxis):
    """interp1(src_t, src_y, taxis) -- linear, NaN outside the source range."""
    return np.interp(taxis, src_t, src_y, left=np.nan, right=np.nan)


def interp_feature(ts, featix, ft, align_time, taxis):
    """findPosition.m -- interpolate one DLC feature's (x, y) onto the aligned time base."""
    src_t = ft - align_time
    x = interp_trace(src_t, ts[:, 0, featix], taxis)
    y = interp_trace(src_t, ts[:, 1, featix], taxis)
    return x, y


def feature_speed(x, y, subtract_baseline):
    """findVelocity.m -- |d(pos)/dbin| at every timepoint where the feature is visible.

    The derivative is evaluated separately on each contiguous visible run (np.gradient,
    i.e. MATLAB's gradient()), so a gap never contaminates a neighbouring sample.
    Returns NaN where the feature is not visible.
    """
    visible = np.isfinite(x) & np.isfinite(y)
    vx = np.full(x.shape, np.nan)
    vy = np.full(y.shape, np.nan)
    idx = np.flatnonzero(visible)
    if idx.size == 0:
        return np.full(x.shape, np.nan)
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], brk + 1))
    stops = np.concatenate((brk, [idx.size - 1]))
    for s, e in zip(starts, stops):
        seg = idx[s:e + 1]
        if seg.size == 1:
            vx[seg] = 0.0
            vy[seg] = 0.0
        else:
            vx[seg] = np.gradient(x[seg])
            vy[seg] = np.gradient(y[seg])
    if subtract_baseline:
        # findVelocity.m: basederiv = median(diff(pos),'omitnan'); xvel/yvel -= basederiv(1)
        d = np.diff(np.stack([x, y], axis=1), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = np.nanmedian(d, axis=0)
        if np.isfinite(base[0]):
            vx = vx - base[0]
            vy = vy - base[0]
    return np.hypot(vx, vy)


def feature_index(obj, view, name):
    """findDLCFeatIndex.m -- index of a DLC feature in obj.traj{view}.featNames.

    Scans trials until one has a usable featNames list (early trials can hold dummy data).
    """
    fn = obj["traj"][view]["featNames"]
    cands = fn if isinstance(fn, list) else [fn]
    for names in cands:
        if names is None:
            continue
        names = [str(s) for s in np.ravel(np.asarray(names, dtype=object))]
        if name in names:
            return names.index(name)
    raise ValueError(f"DLC feature {name!r} not found in view {view + 1}")


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def process_session(args):
    anm, date, probes, task = args
    t0 = time.time()
    obj = load_obj(anm, date, task)
    t_load = time.time() - t0

    bp = obj["bp"]
    trials, ntrials_all = select_trials(bp, obj, probes)
    gocue = vec(bp["ev"][ALIGN_EVENT])[:ntrials_all]

    # ---- neural -----------------------------------------------------------------
    t1 = time.time()
    cluster_ids = select_clusters(obj, probes)
    rates = bin_and_smooth(obj, cluster_ids, gocue, ntrials_all)      # (nclu, ntr, NT)
    rates = rates[:, trials, :]
    n_quality = len(cluster_ids)
    if n_quality:
        mean_fr = rates.reshape(n_quality, -1).mean(axis=1)           # removeLowFRClusters.m
        keep_unit = mean_fr > LOW_FR
        rates = rates[keep_unit]
    else:
        keep_unit = np.zeros(0, dtype=bool)
    nunits = rates.shape[0]
    t_neural = time.time() - t1

    # ---- per-trial outputs -------------------------------------------------------
    R = vec(bp["R"])[:ntrials_all] > 0
    L = vec(bp["L"])[:ntrials_all] > 0
    hit = vec(bp["hit"])[:ntrials_all] > 0
    miss = vec(bp["miss"])[:ntrials_all] > 0
    no = vec(bp["no"])[:ntrials_all] > 0
    autowater = vec(bp["autowater"])[:ntrials_all] > 0

    lick = np.full(ntrials_all, 2, dtype=np.int8)                     # 2 = none
    lick[(L & hit) | (R & miss)] = 0                                  # left
    lick[(R & hit) | (L & miss)] = 1                                  # right
    lick[no] = 2                                                      # getPrevChoice: NaN on ignore
    context = np.where(autowater, 0, 1).astype(np.int8)               # 0 = WC, 1 = DR
    outcome = np.full(ntrials_all, 0, dtype=np.int8)                  # 0 = incorrect
    outcome[hit] = 1                                                  # 1 = correct
    outcome[no] = 2                                                   # 2 = ignore

    # ---- video-derived, time-varying outputs -------------------------------------
    t2 = time.time()
    vidshift = video_shift(obj)
    taxis = TIME + ADVANCE_MOVEMENT
    me_traces = load_motion_energy(anm, date, task, obj, ntrials_all)
    ts_side = as_trial_list(obj["traj"][TONGUE_VIEW]["ts"], ntrials_all)
    ts_bot = as_trial_list(obj["traj"][PAW_VIEW]["ts"], ntrials_all)
    i_tongue = feature_index(obj, TONGUE_VIEW, TONGUE_FEAT)
    i_paws = [feature_index(obj, PAW_VIEW, f) for f in PAW_FEATS]

    ntr = trials.size
    tongue_v = np.full((ntr, NT), np.nan)
    paw_v = np.full((ntr, NT), np.nan)
    me_v = np.full((ntr, NT), np.nan)
    for k, it in enumerate(trials):
        align = gocue[it] + vidshift
        ft_side = frame_times(obj, TONGUE_VIEW, it)
        ft_bot = frame_times(obj, PAW_VIEW, it)
        if ft_side is not None:
            ts = np.asarray(ts_side[it], dtype=np.float64)
            if ts.ndim == 3 and ts.shape[0] == ft_side.size:
                x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)
                tongue_v[k] = feature_speed(x, y, subtract_baseline=False)
        if ft_bot is not None:
            ts = np.asarray(ts_bot[it], dtype=np.float64)
            if ts.ndim == 3 and ts.shape[0] == ft_bot.size:
                sp = []
                for ip in i_paws:
                    x, y = interp_feature(ts, ip, ft_bot, align, taxis)
                    sp.append(feature_speed(x, y, subtract_baseline=True))
                sp = np.stack(sp, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    paw_v[k] = np.nanmean(sp, axis=0)   # mean over the visible paw(s)
        tr = me_traces[it]
        if tr is not None and ft_side is not None and tr.size == ft_side.size:
            me_v[k] = interp_trace(ft_side - align, tr, taxis)
    t_video = time.time() - t2

    # ---- discretise with the per-session 50th percentile --------------------------
    def discretise(v):
        valid = np.isfinite(v)
        out = np.full(v.shape, 2, dtype=np.int8)
        if valid.any():
            thr = float(np.median(v[valid]))
            out[valid] = (v[valid] >= thr).astype(np.int8)
        else:
            thr = np.nan
        return out, thr

    tongue_c, thr_tongue = discretise(tongue_v)
    paw_c, thr_paw = discretise(paw_v)
    me_c, thr_me = discretise(me_v)

    # ---- assemble ----------------------------------------------------------------
    neural, inputs, outputs = [], [], []
    input_trial = np.ascontiguousarray(TIME[None, :], dtype=np.float32)   # shared, read-only
    for k, it in enumerate(trials):
        neural.append(np.ascontiguousarray(rates[:, k, :], dtype=np.float32))
        inputs.append(input_trial)
        out = np.empty((6, NT), dtype=np.int8)
        out[0] = lick[it]
        out[1] = context[it]
        out[2] = outcome[it]
        out[3] = tongue_c[k]
        out[4] = paw_c[k]
        out[5] = me_c[k]
        outputs.append(out)

    info = dict(
        anm=anm, date=date, task=task, probes=list(probes),
        session=f"{anm}_{date}",
        two_context=(anm, date) in TWO_CONTEXT,
        ntrials_raw=ntrials_all, ntrials_kept=int(ntr),
        trial_index=trials.astype(np.int32),   # 0-based index into the raw session's trials
        cluster_index=np.array([(p + 1, i) for p, i in cluster_ids], dtype=np.int32
                               )[keep_unit] if n_quality else np.zeros((0, 2), np.int32),
        n_units_quality=n_quality, n_units=int(nunits),
        n_trials_early=int(np.sum(vec(bp["early"])[:ntrials_all] > 0)),
        n_trials_stim=int(np.sum(vec(bp["stim"]["enable"])[:ntrials_all] > 0)),
        n_trials_no_ephys=int(np.sum(~trials_with_ephys(obj, probes, ntrials_all))),
        n_autowater_kept=int(np.sum(autowater[trials])),
        vidshift=float(vidshift),
        thr_tongue=thr_tongue, thr_paw=thr_paw, thr_me=thr_me,
        timing=dict(load=t_load, neural=t_neural, video=t_video,
                    total=time.time() - t0),
    )
    return dict(neural=neural, input=inputs, output=outputs, info=info)


# --------------------------------------------------------------------------------------
# Diagnostic plots (--show-processing)
# --------------------------------------------------------------------------------------
def show_processing(args, result, outdir="/app"):
    """Plot every processing step for one session so the conversion can be eyeballed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anm, date, probes, task = args
    sid = f"{anm}_{date}"
    obj = load_obj(anm, date, task)
    bp = obj["bp"]
    trials, ntrials_all = select_trials(bp, obj, probes)
    gocue = vec(bp["ev"][ALIGN_EVENT])[:ntrials_all]
    cluster_ids = select_clusters(obj, probes)
    vidshift = video_shift(obj)

    neural = result["neural"]
    outputs = result["output"]
    info = result["info"]
    nunits = neural[0].shape[0]

    fig, ax = plt.subplots(4, 2, figsize=(20, 20))

    # (1) raw spike raster + binned + smoothed rate for one unit / one trial ------------
    k = min(5, len(trials) - 1)
    it = trials[k]
    clu = obj["clu"]
    if isinstance(clu, dict):
        clu = [clu]
    # pick the cluster with the most spikes so the panel is informative
    nspk = [np.size(clu[p]["trialtm"][i]) for p, i in cluster_ids]
    p, i = cluster_ids[int(np.argmax(nspk))]
    tm = np.ravel(np.asarray(clu[p]["trialtm"][i], float))
    tr = np.ravel(np.asarray(clu[p]["trial"][i], float)).astype(int) - 1
    st = tm[tr == it] - gocue[it]
    st = st[(st >= TMIN) & (st < TMAX)]
    cnt = np.histogram(st, bins=EDGES)[0] / DT
    sm = my_smooth(cnt[:, None])[:, 0]
    a = ax[0, 0]
    a.eventplot(st, lineoffsets=-2, linelengths=3, color="k")
    a.plot(TIME, cnt, color="0.7", lw=0.8, label="binned rate (10 ms)")
    a.plot(TIME, sm, color="C3", lw=2, label="causal-gaussian smoothed")
    a.axvline(0, color="b", ls="--", label="go cue")
    a.set_xlabel("time from go cue (s)")
    a.set_ylabel("spikes/s")
    a.set_title(f"{sid}: raster -> binning -> smoothing (trial {it}, unit p{p+1}#{i})")
    a.legend(fontsize=8)

    # (2) population PSTH: alignment check --------------------------------------------
    allrates = np.stack([n.mean(axis=0) for n in neural], axis=0)  # (ntrials, NT)
    a = ax[0, 1]
    a.plot(TIME, allrates.mean(axis=0), "k", lw=2)
    a.fill_between(TIME,
                   allrates.mean(0) - allrates.std(0) / np.sqrt(len(neural)),
                   allrates.mean(0) + allrates.std(0) / np.sqrt(len(neural)),
                   color="k", alpha=0.25)
    a.axvline(0, color="b", ls="--")
    a.set_xlabel("time from go cue (s)")
    a.set_ylabel("population mean rate (spikes/s)")
    a.set_title(f"{sid}: go-cue-aligned population PSTH ({nunits} units, {len(neural)} trials)")

    # (3) neural heat map for one trial ------------------------------------------------
    a = ax[1, 0]
    im = a.imshow(neural[k], aspect="auto", origin="lower",
                  extent=[TIME[0], TIME[-1], 0, nunits], cmap="magma")
    a.axvline(0, color="c", ls="--")
    a.set_xlabel("time from go cue (s)")
    a.set_ylabel("unit")
    a.set_title(f"{sid}: neural matrix, trial index {k}")
    fig.colorbar(im, ax=a, label="spikes/s")

    # (4) tongue trace: raw video vs interpolated + discretisation ---------------------
    ft = frame_times(obj, TONGUE_VIEW, it)
    ts = np.asarray(as_trial_list(obj["traj"][TONGUE_VIEW]["ts"], ntrials_all)[it], float)
    ifeat = feature_index(obj, TONGUE_VIEW, TONGUE_FEAT)
    a = ax[1, 1]
    if ft is not None:
        a.plot(ft - vidshift - gocue[it], ts[:, 1, ifeat], ".", ms=2, color="0.6",
               label="raw DLC tongue y (400 Hz)")
        x, y = interp_feature(ts, ifeat, ft, gocue[it] + vidshift, TIME)
        a.plot(TIME, y, "-", color="C0", lw=1.5, label="interpolated (100 Hz)")
    a2 = a.twinx()
    a2.plot(TIME, outputs[k][3], color="C3", lw=1.5, drawstyle="steps-mid",
            label="tongue_velocity class")
    a2.set_ylabel("class (0/1/2)")
    a2.set_yticks([0, 1, 2])
    a.axvline(0, color="b", ls="--")
    a.set_xlim(TMIN, TMAX)
    a.set_xlabel("time from go cue (s)")
    a.set_ylabel("tongue y (px)")
    a.set_title(f"{sid}: tongue alignment + discretisation (trial index {k})")
    a.legend(fontsize=8, loc="upper left")

    # (5) motion energy: raw vs interpolated + median threshold ------------------------
    me_traces = load_motion_energy(anm, date, task, obj, ntrials_all)
    a = ax[2, 0]
    if ft is not None and me_traces[it] is not None and me_traces[it].size == ft.size:
        a.plot(ft - vidshift - gocue[it], me_traces[it], color="0.6", lw=0.8,
               label="raw motion energy (400 Hz)")
        a.plot(TIME, interp_trace(ft - vidshift - gocue[it], me_traces[it], TIME),
               color="C0", lw=1.5, label="interpolated (100 Hz)")
    a.axhline(info["thr_me"], color="C2", ls=":", label="session median")
    a2 = a.twinx()
    a2.plot(TIME, outputs[k][5], color="C3", lw=1.5, drawstyle="steps-mid")
    a2.set_ylabel("class (0/1/2)")
    a2.set_yticks([0, 1, 2])
    a.axvline(0, color="b", ls="--")
    a.set_xlim(TMIN, TMAX)
    a.set_xlabel("time from go cue (s)")
    a.set_ylabel("motion energy")
    a.set_title(f"{sid}: motion energy alignment + discretisation")
    a.legend(fontsize=8, loc="upper left")

    # (6) paw speed distribution and threshold -----------------------------------------
    a = ax[2, 1]
    for name, thr, dim in (("tongue", info["thr_tongue"], 3),
                           ("paw", info["thr_paw"], 4),
                           ("motion energy", info["thr_me"], 5)):
        cls = np.concatenate([o[dim] for o in outputs])
        frac = [np.mean(cls == c) for c in (0, 1, 2)]
        a.bar([f"{name}\n{c}" for c in (0, 1, 2)], frac, color=["C0", "C1", "0.7"])
    a.set_ylabel("fraction of timepoints")
    a.set_title(f"{sid}: class fractions of the discretised outputs")
    a.tick_params(axis="x", labelsize=7)

    # (7) per-trial outputs across the session ------------------------------------------
    a = ax[3, 0]
    per_trial = np.stack([o[:3, 0] for o in outputs], axis=1)
    a.plot(per_trial[0], ".", label="lick_direction (0 L, 1 R, 2 none)")
    a.plot(per_trial[1] + 3.5, ".", label="context (0 WC, 1 DR)")
    a.plot(per_trial[2] + 7, ".", label="outcome (0 err, 1 correct, 2 ignore)")
    a.set_xlabel("kept trial index")
    a.set_title(f"{sid}: per-trial outputs (offset for display)")
    a.legend(fontsize=8)

    # (8) mean motion-energy class vs time, split by context ----------------------------
    a = ax[3, 1]
    ctx = per_trial[1]
    for c, lab, col in ((0, "WC", "C1"), (1, "DR", "C4")):
        m = ctx == c
        if m.sum() == 0:
            continue
        mv = np.stack([outputs[j][5] for j in np.flatnonzero(m)])
        a.plot(TIME, np.mean(mv == 1, axis=0), color=col, label=f"{lab} (n={m.sum()})")
    a.axvline(0, color="b", ls="--")
    a.set_xlabel("time from go cue (s)")
    a.set_ylabel("P(motion energy above median)")
    a.set_title(f"{sid}: movement locked to the go cue, by context")
    a.legend(fontsize=8)

    fig.tight_layout()
    fn = os.path.join(outdir, f"processing_{sid}.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print(f"  wrote {fn}", flush=True)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outfile", help="output pickle path")
    ap.add_argument("--full", action="store_true", default=True, help="process all sessions")
    ap.add_argument("--sample", action="store_true", help="process only 2 sessions")
    ap.add_argument("--show-processing", action="store_true",
                    help="plot every processing step for up to 2 sessions")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    sessions = SESSIONS
    if args.sample:
        # one two-context fixed-delay session (JEB19 2023-04-18, has both DR and WC blocks)
        # and one randomized-delay session (JEB11 2022-05-10)
        sessions = [SESSIONS[8], SESSIONS[25]]
    print(f"Converting {len(sessions)} session(s) with {args.workers} workers", flush=True)

    t_start = time.time()
    results = []
    if args.workers > 1 and len(sessions) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(process_session, sessions):
                results.append(r)
                i = r["info"]
                print(f"  [{len(results):2d}/{len(sessions)}] {i['session']:>18s} "
                      f"({i['task'][:5]}) units {i['n_units']:4d}/{i['n_units_quality']:4d}  "
                      f"trials {i['ntrials_kept']:4d}/{i['ntrials_raw']:4d}  "
                      f"t={i['timing']['total']:.1f}s "
                      f"(load {i['timing']['load']:.1f}, neural {i['timing']['neural']:.1f}, "
                      f"video {i['timing']['video']:.1f})", flush=True)
    else:
        for s in sessions:
            r = process_session(s)
            results.append(r)
            i = r["info"]
            print(f"  [{len(results):2d}/{len(sessions)}] {i['session']:>18s} "
                  f"({i['task'][:5]}) units {i['n_units']:4d}/{i['n_units_quality']:4d}  "
                  f"trials {i['ntrials_kept']:4d}/{i['ntrials_raw']:4d}  "
                  f"t={i['timing']['total']:.1f}s "
                  f"(load {i['timing']['load']:.1f}, neural {i['timing']['neural']:.1f}, "
                  f"video {i['timing']['video']:.1f})", flush=True)
    t_proc = time.time() - t_start
    print(f"Processing took {t_proc:.1f} s "
          f"({t_proc / max(len(sessions), 1):.1f} s/session)", flush=True)

    # ---- session-level curation ------------------------------------------------------
    kept, dropped = [], []
    for s, r in zip(sessions, results):
        i = r["info"]
        if i["n_units"] < MIN_UNITS:
            dropped.append((i["session"], f"only {i['n_units']} units (< {MIN_UNITS})"))
        elif i["ntrials_kept"] < 2:
            dropped.append((i["session"], f"only {i['ntrials_kept']} trials"))
        else:
            kept.append((s, r))
    for sid, why in dropped:
        print(f"  DROPPED session {sid}: {why}", flush=True)

    # ---- assemble the dataset dict ---------------------------------------------------
    subjects = sorted({r["info"]["anm"] for _, r in kept})
    data = {
        "neural": [r["neural"] for _, r in kept],
        "input": [r["input"] for _, r in kept],
        "output": [r["output"] for _, r in kept],
        "subjects": subjects,
        "subject_idx": np.array([subjects.index(r["info"]["anm"]) for _, r in kept],
                                dtype=np.int64),
        "brain_regions": list(BRAIN_REGIONS),
        "brain_region_idx": [np.zeros(r["neural"][0].shape[0], dtype=np.int64)
                             for _, r in kept],
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "output_values": [list(v) for v in OUTPUT_VALUES],
        "metadata": {
            "task_description":
                "Head-fixed mice performed two directional-licking tasks that alternated "
                "block-wise within a session: a delayed-response (DR) task, in which an "
                "auditory tone indicated the rewarded lickport and an auditory go cue "
                "(after a delay) instructed movement, and a water-cued (WC) task, in which "
                "all auditory cues were omitted and a water drop was delivered at a random "
                "time from a randomly chosen lickport. A subset of sessions used a "
                "randomized delay duration (0.3-3.6 s). Neural activity is extracellular "
                "spiking recorded in anterior lateral motor cortex (ALM) with silicon "
                "probes. The decoder predicts, from ALM activity plus the time relative to "
                "the go cue: the direction of the animal's lick (left/right/none), the "
                "behavioural context (WC/DR), the trial outcome "
                "(incorrect/correct/ignore), and three discretised measures of uninstructed "
                "movement from high-speed video (tongue velocity, paw velocity and whole-"
                "frame motion energy, each split at that session's 50th percentile with a "
                "third class for timepoints where the feature/video is unavailable).",
            "time_bin_size": DT * 1000.0,
            "temporal_alignment_event":
                "go cue onset (bp.ev.goCue; in water-cued trials this is the time the water "
                "drop is presented)",
            "off_start": TMIN,
            "off_end": TMAX,
            "time_bin_centers_s": TIME.astype(np.float32),
            "neural_units": "spikes/s (spike counts in 10 ms bins, divided by the bin width "
                            "and smoothed with a causal Gaussian kernel, gausswin(15), "
                            "'reflect' boundary) -- identical to obj.trialdat in the "
                            "reference MATLAB pipeline",
            "smoothing": {"kernel": "causal gaussian (gausswin(15), first 7 taps zeroed)",
                          "window_samples": SMOOTH_N, "boundary": BCTYPE},
            "neuron_curation": "ALM probe(s) as designated in the reference "
                               "load<ANM>_ALMVideo.m scripts; clusters labelled "
                               "garbage/gabrga/noisy/real? removed (findClusters 'all'); "
                               f"mean firing rate > {LOW_FR} Hz over the analysis window "
                               "(removeLowFRClusters).",
            "trial_curation": "early-lick trials (bp.early) and photoinactivation trials "
                              "(bp.stim.enable) removed, as in every params.condition of the "
                              "reference code; hit / miss / ignore and DR / WC trials kept.",
            "session_curation": f"sessions with >= {MIN_UNITS} curated units (Methods).",
            "video": {"frame_rate_hz": 400.0,
                      "alignment": "frameTimes - videoOffset - goCue, with videoOffset from "
                                   "findVideoOffset.m; linear interpolation onto the 10 ms "
                                   "neural time base; no neural/video lag "
                                   "(params.advance_movement = 0)",
                      "tongue_feature": f"traj view 1 (side camera) '{TONGUE_FEAT}'",
                      "paw_feature": "traj view 2 (bottom camera), mean speed over the "
                                     "visible subset of ('top_paw', 'bottom_paw')"},
            "discretization": "per session, the 50th percentile of the valid (visible / "
                              "video-covered) timepoints of all kept trials; class 2 marks "
                              "timepoints where the feature is not tracked or the video does "
                              "not cover the time bin.",
            "reference": "Hasnain, Birnbaum et al., Nature Neuroscience 2025, "
                         "doi:10.1038/s41593-024-01859-1; data doi:10.5281/zenodo.13941415",
            "session_info": [r["info"] for _, r in kept],
            "sessions_dropped": dropped,
        },
    }

    # ---- summary ---------------------------------------------------------------------
    ntr = [len(n) for n in data["neural"]]
    nun = [n[0].shape[0] for n in data["neural"]]
    print("\n=== converted dataset ===")
    print(f"sessions           : {len(ntr)}")
    print(f"subjects           : {len(subjects)} {subjects}")
    print(f"trials             : {sum(ntr)} (min {min(ntr)}, max {max(ntr)}, "
          f"mean {np.mean(ntr):.1f})")
    print(f"units              : {sum(nun)} (min {min(nun)}, max {max(nun)}, "
          f"mean {np.mean(nun):.1f})")
    print(f"timepoints/trial   : {NT} at {DT*1000:.0f} ms")
    print(f"input range        : [{TIME[0]:.3f}, {TIME[-1]:.3f}] s")
    for d, name in enumerate(OUTPUT_NAMES):
        cls = np.concatenate([o[d] for sess in data["output"] for o in sess])
        frac = [float(np.mean(cls == c)) for c in range(len(OUTPUT_VALUES[d]))]
        print(f"output {d} {name:<16s}: "
              + ", ".join(f"{v}={f:.3f}" for v, f in zip(OUTPUT_VALUES[d], frac)))

    t_w = time.time()
    with open(args.outfile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nwrote {args.outfile} "
          f"({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t_w:.1f} s")

    if args.show_processing:
        print("\nplotting processing steps...", flush=True)
        for s, r in kept[:2]:
            show_processing(s, r)

    print(f"total time {time.time()-t_start:.1f} s")


if __name__ == "__main__":
    main()
