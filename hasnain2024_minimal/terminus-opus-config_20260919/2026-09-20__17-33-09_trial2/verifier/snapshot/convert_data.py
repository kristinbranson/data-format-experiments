"""
Convert data from Hasnain, Birnbaum et al. (Nat Neurosci 2024),
"Separating cognitive and motor processes in the behaving mouse",
into the decoder dataset format.

Processing follows the paper's own MATLAB pipeline
(code/DataLoadingScripts/{loadSessionData,processData,getSeq,alignSpikes,
removeLowFRClusters,findClusters}.m, code/funcs/kinematics/*, and the
parameters used in code/Scripts/*):

  * sessions: every ALM electrophysiology + video session listed in
    code/DataLoadingScripts/'Recording and video'/load<ANM>_ALMVideo.m for
    which a data object is present in /app/data (fixed-delay/two-context
    sessions in Ephys_Behavior and randomized-delay sessions in
    RandomizedDelay_Ephys_Behavior).  The probe number given in those files
    selects the ALM probe(s) of each recording.
  * temporal alignment: go cue (obj.bp.ev.goCue; in the water-cued context
    this field holds the water-drop time), params.alignEvent = 'goCue'.
  * time window/bin: params.tmin = -2.5 s, params.tmax = 2.5 s, params.dt = 10 ms.
  * neural: spikes binned at dt and converted to a firing rate, smoothed with
    the causal half-Gaussian kernel of utils/mySmooth.m (params.smooth = 15,
    params.bctype = 'reflect').
  * units: cluster qualities except garbage/noisy/'real?' (params.quality =
    'all'), then units with mean firing rate <= 1 Hz removed (params.lowFR = 1).
  * trials: early-lick and photoinactivation trials removed (the paper removes
    both from all analyses).  Ignore ('no') trials are kept because 'ignore' is
    one of the requested outcome categories.
  * video: DeepLabCut traces and motion energy are interpolated onto the same
    time axis, after correcting the video/ephys offset exactly as in
    funcs/findVideoOffset.m (vidshift = mode(sglx.bitcode.bitstart)/fs -
    mode(bp.ev.bitStart)).

Additional decisions made for this decoding dataset:

  * trials with no spikes at all from any unit are dropped: in two sessions the
    ephys recording stopped before the behavioral session ended.
  * tongue velocity is the speed (sqrt(vx^2+vy^2)) of the side-camera 'tongue'
    DLC marker, paw velocity the mean speed of the bottom-camera 'top_paw' and
    'bottom_paw' markers; velocities are the gradient of the interpolated
    position, as in funcs/kinematics/findVelocity.m, but expressed as a speed so
    that a single scalar per time point can be discretized.  Following the
    reference code, tongue positions are NaN whenever DeepLabCut does not detect
    the tongue, and those time points are labelled 'not visible' (category 2)
    rather than being filled in.  Time points that fall outside the video
    recording of a trial are likewise labelled 'not visible'/'no video'.
  * each of the three movement variables is discretized at the median of all of
    that session's valid (visible) samples, as requested by the decoder task.
  * inputs: time from the go cue in seconds, time-varying, one value per bin.
"""

import os
import re
import glob
import pickle
import argparse
import numpy as np
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# Loading of the MATLAB data objects (both v7 and v7.3 files are present in
# the released dataset).
# ---------------------------------------------------------------------------
import numpy as np, h5py, scipy.io as sio


def is_v73(fn):
    with open(fn, 'rb') as f:
        return b'MATLAB 7.3' in f.read(128)


def _h5str(f, ref):
    a = np.array(f[ref]).ravel()
    return ''.join(chr(int(c)) for c in a)


def load_session(fn, probes, traj_feats):
    """Load the fields needed for conversion.

    probes: list of 1-based probe numbers (ALM probes for this session)
    traj_feats: {view_index(0/1): [featnames]} features to extract
    Returns dict.
    """
    if is_v73(fn):
        return _load_v73(fn, probes, traj_feats)
    return _load_v7(fn, probes, traj_feats)


QUAL_EXCLUDE = ('garbage', 'gabrga', 'noisy', 'real?')


def _load_v73(fn, probes, traj_feats):
    out = {}
    with h5py.File(fn, 'r') as f:
        o = f['obj']
        bp = o['bp']
        N = int(np.array(bp['Ntrials']).ravel()[0])
        out['Ntrials'] = N
        for k in ['hit', 'miss', 'no', 'early', 'autowater', 'R', 'L']:
            out[k] = np.array(bp[k]).ravel().astype(float)
        out['stim'] = np.array(bp['stim']['enable']).ravel().astype(float)
        for k in ['goCue', 'sample', 'delay', 'bitStart']:
            out[k] = np.array(bp['ev'][k]).ravel().astype(float)
        out['has_fidx'] = 'fidx' in bp
        # spike data
        clu = o['clu']
        units = []
        for p in probes:
            g = f[clu[p - 1, 0]]
            nclu = g['quality'].shape[0]
            for i in range(nclu):
                q = _h5str(f, g['quality'][i, 0]).strip()
                if q in QUAL_EXCLUDE:
                    continue
                trial = np.array(f[g['trial'][i, 0]]).ravel().astype(int)
                trialtm = np.array(f[g['trialtm'][i, 0]]).ravel().astype(float)
                units.append({'quality': q, 'trial': trial, 'trialtm': trialtm, 'probe': p})
        out['units'] = units
        # sglx
        out['fs'] = float(np.array(o['sglx']['fs']).ravel()[0])
        out['bitstart_sglx'] = np.array(o['sglx']['bitcode']['bitstart']).ravel().astype(float)
        # trajectories
        traj = {}
        for v, feats in traj_feats.items():
            g = f[o['traj'][v, 0]]
            names = [_h5str(f, r) for r in np.array(f[g['featNames'][0, 0]]).ravel()]
            fidx = [names.index(ft) for ft in feats]
            per_trial = []
            for t in range(g['ts'].shape[0]):
                ts = np.array(f[g['ts'][t, 0]])  # (nfeat,3,nframes)
                ft_ = np.array(f[g['frameTimes'][t, 0]]).ravel().astype(float)
                nd = np.array(f[g['NdroppedFrames'][t, 0]]).ravel()
                if ts.ndim != 3:
                    per_trial.append(None)
                    continue
                xy = np.stack([ts[i, 0:2, :].T for i in fidx], axis=-1)  # (nframes,2,nfeat)
                per_trial.append({'xy': xy.astype(float), 'frameTimes': ft_,
                                  'Ndropped': nd})
            traj[v] = {'feats': feats, 'trials': per_trial}
        out['traj'] = traj
        # region
        out['loc'] = _probe_locs_v73(f, o, probes)
    return out


def _probe_locs_v73(f, o, probes):
    locs = []
    try:
        pr = o['ex']['probe']
        if isinstance(pr, h5py.Group) and 'loc' in pr:
            s = ''.join(chr(int(c)) for c in np.array(pr['loc']).ravel())
            locs = [s] * len(probes)
        else:
            arr = np.array(pr).ravel()
            for p in probes:
                g = f[arr[p - 1]]
                locs.append(''.join(chr(int(c)) for c in np.array(g['loc']).ravel()))
    except Exception:
        locs = ['ALM'] * len(probes)
    return locs


def _load_v7(fn, probes, traj_feats):
    m = sio.loadmat(fn, squeeze_me=True, struct_as_record=False)
    o = m['obj']
    bp = o.bp
    out = {}
    out['Ntrials'] = int(bp.Ntrials)
    for k in ['hit', 'miss', 'no', 'early', 'autowater', 'R', 'L']:
        out[k] = np.atleast_1d(getattr(bp, k)).astype(float)
    out['stim'] = np.atleast_1d(bp.stim.enable).astype(float)
    for k in ['goCue', 'sample', 'delay', 'bitStart']:
        out[k] = np.atleast_1d(getattr(bp.ev, k)).astype(float)
    out['has_fidx'] = hasattr(bp, 'fidx')
    clu = o.clu
    units = []
    for p in probes:
        if isinstance(clu, np.ndarray) and clu.dtype == object and clu.ndim == 1 and \
                hasattr(clu[0], '_fieldnames') and 'quality' in clu[0]._fieldnames:
            # single probe stored as struct array of clusters
            cl = clu if len(probes) == 1 else clu
        else:
            cl = clu[p - 1]
        cl = np.atleast_1d(cl)
        for c in cl:
            q = str(c.quality).strip()
            if q in QUAL_EXCLUDE:
                continue
            units.append({'quality': q,
                          'trial': np.atleast_1d(c.trial).astype(int),
                          'trialtm': np.atleast_1d(c.trialtm).astype(float),
                          'probe': p})
    out['units'] = units
    out['fs'] = float(o.sglx.fs)
    out['bitstart_sglx'] = np.atleast_1d(o.sglx.bitcode.bitstart).astype(float)
    traj = {}
    for v, feats in traj_feats.items():
        tv = np.atleast_1d(o.traj[v])
        names = list(np.atleast_1d(tv[0].featNames))
        fidx = [names.index(ft) for ft in feats]
        per_trial = []
        for t in range(len(tv)):
            ts = tv[t].ts
            ft_ = np.atleast_1d(tv[t].frameTimes).astype(float)
            nd = np.atleast_1d(tv[t].NdroppedFrames).astype(float)
            ts = np.asarray(ts)
            if ts.ndim != 3:
                per_trial.append(None)
                continue
            xy = np.stack([ts[:, 0:2, i] for i in fidx], axis=-1)
            per_trial.append({'xy': xy.astype(float), 'frameTimes': ft_, 'Ndropped': nd})
        traj[v] = {'feats': feats, 'trials': per_trial}
    out['traj'] = traj
    locs = []
    try:
        pr = o.ex.probe
        if hasattr(pr, 'loc'):
            locs = [str(pr.loc)] * len(probes)
        else:
            pr = np.atleast_1d(pr)
            locs = [str(pr[p - 1].loc) for p in probes]
    except Exception:
        locs = ['ALM'] * len(probes)
    out['loc'] = locs
    return out


def load_motion_energy(fn):
    m = sio.loadmat(fn, squeeze_me=True, struct_as_record=False)
    me = m['me']
    data = me.data
    if hasattr(data, '_fieldnames'):
        data = data.data
    out = [np.atleast_1d(np.asarray(d, dtype=float)).ravel() for d in np.atleast_1d(data)]
    return out



# ----------------------------------------------------------------------------
# parameters (from the paper's params struct)
# ----------------------------------------------------------------------------
ALIGN_EVENT = 'goCue'
TMIN = -2.5           # s, params.tmin
TMAX = 2.5            # s, params.tmax
DT = 0.01             # s, params.dt = 1/100
SMOOTH = 15           # params.smooth, width of causal gaussian kernel (bins)
BCTYPE = 'reflect'    # params.bctype
LOW_FR = 1.0          # Hz, params.lowFR
MIN_UNITS = 10        # paper: sessions included only if >= 10 units
MIN_TRIALS = 2        # decoder requirement

DATA_DIRS = ['/app/data/Ephys_Behavior', '/app/data/RandomizedDelay_Ephys_Behavior']
LOADSCRIPT_DIR = '/app/code/DataLoadingScripts/Recording and video'

# DeepLabCut features used for the requested kinematic outputs.
# The tongue is tracked in the side camera (view 1) and the paws only in the
# bottom camera (view 2), so each is taken from the camera that tracks it.
TONGUE_VIEW, TONGUE_FEATS = 0, ['tongue']
PAW_VIEW, PAW_FEATS = 1, ['top_paw', 'bottom_paw']

EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]     # bin centres, obj.time
NT = len(TAXIS)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def gausswin(N, alpha=2.5):
    """MATLAB gausswin(N, alpha)."""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def my_smooth(x, N=SMOOTH, bctype=BCTYPE):
    """Port of utils/mySmooth.m: causal half-Gaussian smoothing along axis 0."""
    x = np.asarray(x, dtype=float)
    if bctype == 'reflect':
        xf = np.concatenate([x[:N], x], axis=0)
        trim = N
    else:
        xf = x
        trim = 0
    kern = gausswin(N)
    kern[:N // 2] = 0          # causal
    kern = kern / kern.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xf)
    return out[trim:]


def parse_sessions():
    """Read the paper's per-animal loading scripts -> [(anm, date, [probes])]."""
    out = []
    for fn in sorted(glob.glob(os.path.join(LOADSCRIPT_DIR, '*.m'))):
        anm, cur = None, None
        for line in open(fn):
            s = line.strip()
            if s.startswith('%'):
                continue          # commented-out sessions are excluded by the authors
            m = re.search(r"\.anm\s*=\s*'([^']+)'", s)
            if m:
                anm = m.group(1)
            m = re.search(r"\.date\s*=\s*'([^']+)'", s)
            if m:
                if cur is not None:
                    out.append(cur)
                cur = {'anm': anm, 'date': m.group(1), 'probe': [1]}
            m = re.search(r"\.probe\s*=\s*\[?([0-9 ,]+)\]?\s*;", s)
            if m and cur is not None:
                cur['probe'] = [int(x) for x in re.findall(r'\d+', m.group(1))]
        if cur is not None:
            out.append(cur)
    return out


def find_files(anm, date):
    for d in DATA_DIRS:
        f = os.path.join(d, 'data_structure_%s_%s.mat' % (anm, date))
        if os.path.exists(f):
            me = os.path.join(d, 'motionEnergy_%s_%s.mat' % (anm, date))
            return f, (me if os.path.exists(me) else None), os.path.basename(d)
    return None, None, None


def interp_nan(xnew, xp, fp):
    """Linear interpolation that returns NaN outside the sampled range and
    propagates NaNs in fp, like MATLAB's interp1."""
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    ok = np.isfinite(xp)
    xp, fp = xp[ok], fp[ok]
    if xp.size < 2:
        return np.full(len(xnew), np.nan)
    order = np.argsort(xp)
    xp, fp = xp[order], fp[order]
    idx = np.searchsorted(xp, xnew) - 1
    out = np.full(len(xnew), np.nan)
    valid = (idx >= 0) & (idx < len(xp) - 1)
    i = idx[valid]
    x0, x1 = xp[i], xp[i + 1]
    y0, y1 = fp[i], fp[i + 1]
    w = (xnew[valid] - x0) / np.where(x1 > x0, x1 - x0, np.nan)
    out[valid] = y0 + w * (y1 - y0)
    # exact right edge
    edge = xnew == xp[-1]
    out[edge] = fp[-1]
    return out


# ----------------------------------------------------------------------------
# per-session conversion
# ----------------------------------------------------------------------------
def process_session(sess):
    anm, date, probes = sess['anm'], sess['date'], sess['probe']
    fn, me_fn, dset = find_files(anm, date)
    if fn is None:
        return None

    traj_feats = {TONGUE_VIEW: list(TONGUE_FEATS), PAW_VIEW: list(PAW_FEATS)}

    d = load_session(fn, probes, traj_feats)
    N = d['Ntrials']
    gocue = d[ALIGN_EVENT]

    # ---- trial selection ----------------------------------------------------
    keep = (d['early'] == 0) & (d['stim'] == 0) & np.isfinite(gocue)
    ntraj = len(d['traj'][TONGUE_VIEW]['trials'])
    keep = keep[:N]
    if ntraj < N:
        keep[ntraj:] = False
    trials = np.where(keep)[0]
    if len(trials) < MIN_TRIALS:
        return None

    # ---- neural -------------------------------------------------------------
    # bin spikes aligned to the go cue and convert to smoothed firing rates
    units = d['units']
    rates = np.zeros((NT, len(units), len(trials)), dtype=np.float32)
    for iu, u in enumerate(units):
        tt = u['trial']
        tm = u['trialtm'] - gocue[np.clip(tt - 1, 0, N - 1)]   # trialtm_aligned
        order = np.argsort(tt, kind='stable')
        tt, tm = tt[order], tm[order]
        starts = np.searchsorted(tt, trials + 1, 'left')
        ends = np.searchsorted(tt, trials + 1, 'right')
        cnt = np.zeros((NT, len(trials)))
        for j in range(len(trials)):
            if ends[j] > starts[j]:
                cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
        rates[:, iu, :] = my_smooth(cnt / DT).astype(np.float32)

    # remove low firing rate units (params.lowFR)
    mean_fr = rates.mean(axis=(0, 2))
    use = mean_fr > LOW_FR
    if use.sum() < MIN_UNITS:
        return None
    rates = rates[:, use, :]
    units = [u for u, k in zip(units, use) if k]

    # Drop trials in which no unit fired a single spike.  In a couple of sessions
    # the ephys recording stopped before the behavioral session ended, leaving a
    # tail of trials with no neural data at all.
    nonempty = np.any(rates != 0, axis=(0, 1))
    n_no_ephys = int((~nonempty).sum())
    if n_no_ephys:
        rates = rates[:, :, nonempty]
        trials = trials[nonempty]
        if len(trials) < MIN_TRIALS:
            return None

    # ---- behaviour / task variables ----------------------------------------
    hit, miss, no = d['hit'] > 0, d['miss'] > 0, d['no'] > 0
    R, L = d['R'] > 0, d['L'] > 0
    aw = d['autowater'] > 0

    # lick direction: right if (right trial & correct) or (left trial & error)
    # (identical to funcs/getPrevChoice.m); ignore trials have no lick
    right_lick = (R & hit) | (L & miss)
    lick_dir = np.where(no, 2, np.where(right_lick, 1, 0))       # 0 left,1 right,2 none
    context = np.where(aw, 0, 1)                                 # 0 WC, 1 DR
    outcome = np.where(no, 2, np.where(hit, 1, 0))               # 0 incorrect,1 correct,2 ignore

    # ---- video: kinematics and motion energy --------------------------------
    # video/ephys offset, funcs/findVideoOffset.m
    vidshift = (np.median(d['bitstart_sglx']) / d['fs']) - np.median(d['bitStart'])

    me_raw = load_motion_energy(me_fn) if me_fn else None

    tongue_speed = np.full((NT, len(trials)), np.nan)
    paw_speed = np.full((NT, len(trials)), np.nan)
    me_t = np.full((NT, len(trials)), np.nan)

    tv = d['traj'][TONGUE_VIEW]
    pv = d['traj'][PAW_VIEW]
    t_idx = [tv['feats'].index(f) for f in TONGUE_FEATS]
    p_idx = [pv['feats'].index(f) for f in PAW_FEATS]

    for j, t in enumerate(trials):
        for view, feats_idx, dest in ((tv, t_idx, 'tongue'), (pv, p_idx, 'paw')):
            tr = view['trials'][t] if t < len(view['trials']) else None
            if tr is None or tr['Ndropped'].size == 0 or np.any(~np.isfinite(tr['Ndropped'])):
                continue          # video for this trial is unusable
            ft = tr['frameTimes']
            if ft.size == 0 or not np.any(np.isfinite(ft)):
                ft = (np.arange(tr['xy'].shape[0]) + 1) / 400.0
            tt = ft - vidshift - gocue[t]
            sp = []
            for fi in feats_idx:
                x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
                y = interp_nan(TAXIS, tt, tr['xy'][:, 1, fi])
                vx = np.gradient(x) / DT
                vy = np.gradient(y) / DT
                sp.append(np.sqrt(vx ** 2 + vy ** 2))
            sp = np.vstack(sp)
            with np.errstate(invalid='ignore'):
                spd = np.where(np.all(np.isnan(sp), axis=0), np.nan, np.nanmean(sp, axis=0))
            if dest == 'tongue':
                tongue_speed[:, j] = spd
            else:
                paw_speed[:, j] = spd
        # motion energy (sampled with the video frames)
        if me_raw is not None and t < len(me_raw):
            tr = tv['trials'][t] if t < len(tv['trials']) else None
            if tr is not None and tr['frameTimes'].size == len(me_raw[t]):
                tt = tr['frameTimes'] - vidshift - gocue[t]
                me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])
            elif tr is not None:
                ftimes = (np.arange(len(me_raw[t])) + 1) / 400.0
                me_t[:, j] = interp_nan(TAXIS, ftimes - 0.5 - gocue[t], me_raw[t])

    # ---- discretise the kinematic outputs with per-session thresholds -------
    def discretise(x):
        vis = np.isfinite(x)
        out = np.full(x.shape, 2, dtype=np.int64)          # 2 = not visible / no video
        if vis.sum() > 0:
            thr = np.percentile(x[vis], 50)
            out[vis] = (x[vis] >= thr).astype(np.int64)
        return out, (float(np.percentile(x[vis], 50)) if vis.sum() else None)

    tongue_cat, tongue_thr = discretise(tongue_speed)
    paw_cat, paw_thr = discretise(paw_speed)
    me_cat, me_thr = discretise(me_t)

    # ---- assemble -----------------------------------------------------------
    neural, inputs, outputs = [], [], []
    time_in = TAXIS.astype(np.float32)[None, :]
    for j, t in enumerate(trials):
        neural.append(np.ascontiguousarray(rates[:, :, j].T))             # (nneurons, T)
        inputs.append(time_in.copy())
        out = np.empty((6, NT), dtype=np.int64)
        out[0, :] = lick_dir[t]
        out[1, :] = context[t]
        out[2, :] = outcome[t]
        out[3, :] = tongue_cat[:, j]
        out[4, :] = paw_cat[:, j]
        out[5, :] = me_cat[:, j]
        outputs.append(out)

    info = {
        'animal': anm, 'date': date, 'dataset': dset,
        'task': ('randomized delay' if 'Randomized' in dset else 'fixed delay'),
        'probes': probes, 'probe_loc': d.get('loc'),
        'n_trials_total': int(N), 'n_trials_used': int(len(trials)),
        'n_trials_without_ephys': n_no_ephys,
        'n_units': int(len(units)),
        'n_single_units': int(sum(1 for u in units
                                  if u['quality'].strip() in ('excellent', 'great', 'good'))),
        'n_WC_trials': int(np.sum(context[trials] == 0)),
        'n_DR_trials': int(np.sum(context[trials] == 1)),
        'vidshift': float(vidshift),
        'tongue_speed_threshold': tongue_thr,
        'paw_speed_threshold': paw_thr,
        'motion_energy_threshold': me_thr,
    }
    return {'anm': anm, 'neural': neural, 'input': inputs, 'output': outputs,
            'nunits': len(units), 'info': info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--nproc', type=int, default=12)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    sessions = parse_sessions()
    sessions = [s for s in sessions if find_files(s['anm'], s['date'])[0] is not None]
    if args.limit:
        sessions = sessions[:args.limit]
    print('processing %d sessions' % len(sessions))

    if args.nproc > 1:
        with Pool(args.nproc, maxtasksperchild=1) as p:
            res = p.map(process_session, sessions, chunksize=1)
    else:
        res = [process_session(s) for s in sessions]
    res = [r for r in res if r is not None]
    print('kept %d sessions' % len(res))

    subjects = sorted({r['anm'] for r in res})
    data = {
        'neural': [r['neural'] for r in res],
        'input': [r['input'] for r in res],
        'output': [r['output'] for r in res],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['anm']) for r in res]),
        'brain_regions': ['ALM'],
        'brain_region_idx': [np.zeros(r['nunits'], dtype=np.int64) for r in res],
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'context', 'outcome',
                         'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [['left', 'right', 'none'],
                          ['WC', 'DR'],
                          ['incorrect', 'correct', 'ignore'],
                          ['below_median', 'above_median', 'not_visible'],
                          ['below_median', 'above_median', 'not_visible'],
                          ['below_median', 'above_median', 'no_video']],
        'metadata': {
            'task_description':
                'Head-fixed mice performed two directional licking tasks that alternated '
                'block-wise within a session. In the delayed-response (DR) task an auditory '
                'sample tone indicated the rewarded lick port, and after a delay an auditory '
                'go cue instructed the mouse to lick. In the water-cued (WC) task all auditory '
                'cues were omitted and a water drop was presented at a random time at a randomly '
                'chosen port. Decoded variables are the direction of the instructed lick, the '
                'behavioral context (WC/DR), the trial outcome, and three uninstructed-movement '
                'variables measured from high-speed video (tongue speed, paw speed and whole-frame '
                'motion energy), each discretized at its per-session median.',
            'time_bin_size': DT * 1000.0,
            'temporal_alignment_event':
                'go cue onset (obj.bp.ev.goCue); in WC blocks this is the time of water presentation',
            'off_start': TMIN,
            'off_end': TMAX,
            'smoothing': 'causal half-Gaussian kernel, 15 bins (150 ms) wide (mySmooth.m)',
            'neural_units': 'spikes/s (firing rate)',
            'brain_area': 'anterior lateral motor cortex (ALM), left or right hemisphere',
            'trial_exclusion': 'early-lick trials and optogenetic photoinactivation trials excluded',
            'unit_selection':
                'clusters labelled garbage/noisy/real? excluded; units with mean firing rate <= 1 Hz '
                'removed; sessions with fewer than 10 remaining units excluded',
            'paper': 'Hasnain, Birnbaum et al., Nature Neuroscience 2024, '
                     '"Separating cognitive and motor processes in the behaving mouse"',
            'session_info': [r['info'] for r in res],
        },
    }

    with open(args.out, 'wb') as f:
        pickle.dump(data, f)
    print('saved', args.out)
    print('sessions', len(data['neural']),
          'trials', sum(len(s) for s in data['neural']),
          'neurons', sum(len(b) for b in data['brain_region_idx']))


if __name__ == '__main__':
    main()
