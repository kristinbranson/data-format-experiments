"""
Convert the ALM electrophysiology + high-speed-video dataset of
Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the behaving mouse"
into the decoder dataset format (/app/converted_data.pkl).

Decisions, all following the authors' MATLAB pipeline (code/DataLoadingScripts, code/funcs,
WorkingWithDataObjs.m) unless the decoder task requires otherwise:

Sessions
  * data/Ephys_Behavior: the fixed-delay recordings used for the paper's main neural analyses
    (25 sessions / 9 mice, 12 of which are two-context DR+WC sessions).  The
    randomized-delay dataset (data/RandomizedDelay_Ephys_Behavior) is a separate cohort
    analysed separately in the paper (Fig. 8) with a different trial structure, and its
    sessions contain essentially no water-cued trials, so it is not merged here.
  * The probes used in each session are taken from the authors' loadANM_ALMVideo.m meta
    scripts (sessions/probes that the authors commented out are excluded).
  * Sessions with fewer than 10 usable units are excluded (paper inclusion criterion).

Units
  * Clusters whose quality label is garbage/noisy/real? are excluded (findClusters.m with
    params.quality = 'all').
  * Units with mean firing rate <= 1 Hz are excluded (paper: "All units with firing rates
    exceeding 1 Hz were included in all other analyses"; removeLowFRClusters.m).
  * The brain region of every unit is read from the probe location stored in the data object
    (obj.ex.probe.loc / obj.meta.probe.loc): ALM for all probes except the second JEB15 probe,
    which is in tjM1.

Trials
  * Completed trials only (hit | miss | no).
  * Early-lick trials are excluded (paper: "excluding early lick and ignore trials, which were
    omitted from all analyses").  Ignore (no-response) trials are kept, because "ignore" is one
    of the outcome categories that the decoder must predict.
  * Optogenetic photoinactivation trials (stim.enable) are excluded so that the neural data
    reflect unperturbed activity.

Neural data
  * Spikes are aligned to the go cue (which, on water-cued trials, is the time of water
    presentation) -- params.alignEvent = 'goCue' in the paper's scripts.
  * Binned from -2.5 s to +2.5 s in 10 ms bins (params.tmin/tmax/dt), converted to firing rate
    and smoothed with the authors' causal Gaussian kernel (mySmooth.m, window 15 bins,
    'reflect' boundary handling) -> (n_units, 500) per trial.

Decoder input
  * Time from go cue onset (s), the bin centres of the time axis above.

Decoder outputs (integer categories, time-varying, constant within a trial where the variable
is a per-trial quantity)
  * lick_direction: side of the first lick-port contact after the go cue (left/right/none).
  * context: DR vs WC block, from obj.bp.autowater (the authors use autowater as the proxy for
    water-cued blocks, see WorkingWithDataObjs.m).
  * outcome: incorrect (miss) / correct (hit) / ignore (no).
  * tongue_velocity, paw_velocity: speed of the DeepLabCut-tracked tongue (side camera) and
    paws (bottom camera, mean of top_paw and bottom_paw), computed on the neural time axis
    after interpolating the trajectories with the authors' video/ephys offset
    (findVideoOffset.m, findPosition.m/findVelocity.m).  Discretized per session at the 50th
    percentile of the finite values (0 = below, 1 = at/above); timepoints where the feature is
    not tracked (tongue not out of the mouth, paw not visible) are class 2.
  * motion_energy: frame-to-frame motion energy from the motionEnergy_*.mat files interpolated
    onto the same axis (loadMotionEnergy.m), discretized the same way; trials/sessions with no
    video get class 2.
"""

import os
import re
import glob
import pickle
import numpy as np
import h5py
import scipy.io as sio

DATA_DIR = '/app/data/Ephys_Behavior'
OUT_FILE = '/app/converted_data.pkl'

# ---- analysis parameters (match params in the paper's scripts) ----
TMIN = -2.5      # s relative to go cue
TMAX = 2.5
DT = 0.01        # 10 ms bins
SMOOTH = 15      # causal gaussian kernel window (bins)
LOW_FR = 1.0     # Hz, minimum mean firing rate of a unit
MIN_UNITS = 10   # minimum number of units for a session to be included

EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
NT = len(TAXIS)

BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}


# --------------------------------------------------------------------------
# helpers for reading MATLAB v7.3 files
# --------------------------------------------------------------------------
def mstr(f, ref):
    """read a MATLAB char array (possibly behind a reference)"""
    d = np.array(f[ref]).ravel()
    return ''.join(chr(int(c)) for c in d)


def gausswin(N, alpha=2.5):
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def causal_kernel(N=SMOOTH):
    """mySmooth.m: gaussian window, first floor(N/2) coefficients zeroed (causal)"""
    k = gausswin(N)
    k[:N // 2] = 0
    return k / k.sum()


KERN = causal_kernel()


def smooth_causal(x):
    """smooth along last axis, replicating mySmooth(x,15,'reflect')"""
    N = len(KERN)
    xp = np.concatenate([x[..., :N], x], axis=-1)
    out = np.empty_like(xp)
    flat = xp.reshape(-1, xp.shape[-1])
    o = out.reshape(-1, xp.shape[-1])
    for i in range(flat.shape[0]):
        o[i] = np.convolve(flat[i], KERN, mode='same')
    return out[..., N:]


# --------------------------------------------------------------------------
# session list / probe assignment from the authors' meta scripts
# --------------------------------------------------------------------------
def parse_meta_scripts(meta_dir='/app/code/DataLoadingScripts/Recording and video'):
    sessions = {}
    for fn in sorted(glob.glob(os.path.join(meta_dir, '*.m'))):
        anm = None
        cur_date = None
        for line in open(fn).read().split('\n'):
            s = line.strip()
            if s.startswith('%'):
                continue
            m = re.match(r"meta\(.*\)\.anm\s*=\s*'([^']+)'", s)
            if m:
                anm = m.group(1)
            m = re.match(r"meta\(.*\)\.date\s*=\s*'([^']+)'", s)
            if m:
                cur_date = m.group(1)
            m = re.match(r"meta\(.*\)\.probe\s*=\s*\[?([0-9 ,]+)\]?", s)
            if m and anm is not None and cur_date is not None:
                probes = [int(p) for p in m.group(1).replace(',', ' ').split()]
                sessions[(anm, cur_date)] = probes
    return sessions


# --------------------------------------------------------------------------
# per session conversion
# --------------------------------------------------------------------------

def get_probe_locs(f):
    """probe locations, stored in obj.ex.probe.loc (newer objs) or obj.meta.probe.loc"""
    for key in ['obj/ex/probe/loc', 'obj/meta/probe/loc']:
        if key in f:
            dset = f[key]
            if dset.dtype == h5py.special_dtype(ref=h5py.Reference):
                return [mstr(f, r) for r in np.array(dset).ravel()]
            # single probe sessions store the location string directly
            return [''.join(chr(int(c)) for c in np.array(dset).ravel())]
    return None



def probe_region(loc):
    """map a probe location string (e.g. 'R ALM', 'L M1TJ') onto a brain region name"""
    u = loc.upper().replace('_', ' ')
    if 'ALM' in u:
        return 'ALM'
    if 'M1TJ' in u or 'TJM1' in u:
        return 'tjM1'
    # the authors' meta scripts only list ALM (and, for JEB15, tjM1) probes, so an
    # unlabelled probe listed there is an ALM probe
    return 'ALM'


def load_session(dsfile, probes, verbose=True):
    f = h5py.File(dsfile, 'r')
    bp = f['obj/bp']
    get = lambda k: np.array(bp[k]).ravel()
    ntrials = int(get('Ntrials')[0])
    hit, miss, no = get('hit'), get('miss'), get('no')
    early, autowater = get('early'), get('autowater')
    R, L = get('R'), get('L')
    stim = np.array(bp['stim/enable']).ravel()
    gocue = np.array(bp['ev/goCue']).ravel()

    # ---- trial selection ----
    keep = ((hit == 1) | (miss == 1) | (no == 1)) & (early != 1) & (stim != 1) & ~np.isnan(gocue)
    trials = np.where(keep)[0]            # 0-based trial indices
    if len(trials) < 2:
        f.close()
        return None

    # ---- neural data ----
    probe_locs = get_probe_locs(f)
    rates = []
    regions = []
    for prb in probes:
        # the authors' meta scripts list the probes used for each session; the brain
        # region of each probe is read from the data object (obj.ex/meta.probe.loc)
        loc = probe_locs[prb - 1].strip() if probe_locs is not None else ''
        region = probe_region(loc)
        clu = f[f['obj/clu'][prb - 1, 0]]
        qual = [mstr(f, r).strip().lower() for r in np.array(clu['quality']).ravel()]
        cluid = [i for i, q in enumerate(qual) if q not in BAD_QUALITY]
        trialtm_refs = np.array(clu['trialtm']).ravel()
        trial_refs = np.array(clu['trial']).ravel()

        trial_pos = -np.ones(ntrials, dtype=np.int64)
        trial_pos[trials] = np.arange(len(trials))

        sess_rates = np.zeros((len(cluid), len(trials), NT), dtype=np.float32)
        for ui, cid in enumerate(cluid):
            tt = np.array(f[trialtm_refs[cid]]).ravel()
            tr = np.array(f[trial_refs[cid]]).ravel().astype(np.int64) - 1  # 0-based
            ok = (tr >= 0) & (tr < ntrials)
            tt, tr = tt[ok], tr[ok]
            ta = tt - gocue[tr]                     # align to go cue
            pos = trial_pos[tr]
            b = np.floor((ta - TMIN) / DT).astype(np.int64)
            good = (pos >= 0) & (b >= 0) & (b < NT)
            if not np.any(good):
                continue
            idx = pos[good] * NT + b[good]
            counts = np.bincount(idx, minlength=len(trials) * NT).reshape(len(trials), NT)
            sess_rates[ui] = counts / DT
        sess_rates = smooth_causal(sess_rates)
        # remove low firing rate units
        mfr = sess_rates.mean(axis=(1, 2))
        use = mfr > LOW_FR
        sess_rates = sess_rates[use]
        rates.append(sess_rates)
        regions += [region] * int(use.sum())
        if verbose:
            print(f'   probe {prb} (loc={loc!r} -> {region}): {len(qual)} clusters, '
                  f'{len(cluid)} non-garbage, {int(use.sum())} with FR>{LOW_FR}Hz')

    if len(rates) == 0:
        f.close()
        return None
    rates = np.concatenate(rates, axis=0)       # (nunits, ntrials, nt)
    if rates.shape[0] < MIN_UNITS:
        if verbose:
            print(f'   session dropped: only {rates.shape[0]} units')
        f.close()
        return None

    # ---- behavioral outputs derived from bpod ----
    # lick direction: first lick port contact after the go cue
    lickL = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
             for r in np.array(bp['ev/lickL']).ravel()]
    lickR = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
             for r in np.array(bp['ev/lickR']).ravel()]
    lick_dir = np.full(len(trials), 2, dtype=np.int64)   # 2 = none
    for i, t in enumerate(trials):
        gc = gocue[t]
        lt = lickL[t][lickL[t] > gc] if lickL[t].size else np.array([])
        rt = lickR[t][lickR[t] > gc] if lickR[t].size else np.array([])
        fl = lt.min() if lt.size else np.inf
        fr = rt.min() if rt.size else np.inf
        if np.isinf(fl) and np.isinf(fr):
            lick_dir[i] = 2
        elif fl <= fr:
            lick_dir[i] = 0          # left
        else:
            lick_dir[i] = 1          # right

    context = (autowater[trials] == 1).astype(np.int64)   # 0 = DR, 1 = WC
    outcome = np.full(len(trials), 0, dtype=np.int64)     # 0 incorrect
    outcome[hit[trials] == 1] = 1                         # 1 correct
    outcome[no[trials] == 1] = 2                          # 2 ignore

    # ---- video based outputs ----
    tongue, paw = video_speeds(f, trials, gocue)
    me = motion_energy(dsfile, f, trials, gocue)
    f.close()
    return dict(rates=rates, regions=regions, trials=trials,
                lick_dir=lick_dir, context=context, outcome=outcome,
                tongue=tongue, paw=paw, me=me)


def video_offset(f):
    """findVideoOffset.m : offset between neural file start and video file start"""
    def mode(x):
        x = x[~np.isnan(x)]
        vals, cnt = np.unique(x, return_counts=True)
        return vals[np.argmax(cnt)]
    fs = np.array(f['obj/sglx/fs']).ravel()[0]
    bitstart_sglx = np.array(f['obj/sglx/bitcode/bitstart']).ravel().astype(float)
    bitstart_bp = np.array(f['obj/bp/ev/bitStart']).ravel().astype(float)
    return mode(bitstart_sglx) / fs - mode(bitstart_bp)


def interp_nan(xnew, x, y):
    """linear interpolation, NaN outside of range (like MATLAB interp1)"""
    out = np.interp(xnew, x, y, left=np.nan, right=np.nan)
    # np.interp ignores NaNs in y in a different way than MATLAB: redo bounding check
    idx = np.searchsorted(x, xnew) - 1
    idx = np.clip(idx, 0, len(x) - 2)
    inrange = (xnew >= x[0]) & (xnew <= x[-1])
    bad = np.isnan(y[idx]) | np.isnan(y[idx + 1])
    out[inrange & bad] = np.nan
    return out


def video_speeds(f, trials, gocue):
    """speed of the tongue (side cam) and of the paws (bottom cam), on the neural time axis.

    Returns arrays (ntrials, NT) of speed with NaN where the feature is not visible.
    """
    vidshift = video_offset(f)
    traj = f['obj/traj']
    nviews = traj.shape[0]
    views = []
    for v in range(nviews):
        g = f[traj[v, 0]]
        names = [mstr(f, r) for r in np.array(f[np.array(g['featNames']).ravel()[0]]).ravel()]
        views.append((g, names))

    def feat_speed(view, featnames, t):
        """mean speed over the listed features for trial t (NaN where not visible)"""
        g, names = views[view]
        idxs = [names.index(n) for n in featnames if n in names]
        if len(idxs) == 0:
            return np.full(NT, np.nan)
        ft = np.array(f[np.array(g['frameTimes']).ravel()[t]]).ravel().astype(float)
        ts = np.array(f[np.array(g['ts']).ravel()[t]])       # (feat, 3, frames)
        if ft.size < 2 or np.all(np.isnan(ft)):
            return np.full(NT, np.nan)
        tt = ft - vidshift - gocue[t]
        sp = []
        for fi in idxs:
            x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
            y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
            vx = np.gradient(x)
            vy = np.gradient(y)
            sp.append(np.sqrt(vx ** 2 + vy ** 2))
        sp = np.array(sp)
        with np.errstate(invalid='ignore'):
            out = np.nanmean(sp, axis=0) if sp.shape[0] > 1 else sp[0]
        out[np.all(np.isnan(sp), axis=0)] = np.nan
        return out

    tongue = np.full((len(trials), NT), np.nan)
    paw = np.full((len(trials), NT), np.nan)
    # side cam tongue, bottom cam paws (params.traj_features in the paper's scripts)
    for i, t in enumerate(trials):
        tongue[i] = feat_speed(0, ['tongue'], t)
        if nviews > 1:
            paw[i] = feat_speed(1, ['top_paw', 'bottom_paw'], t)
    return tongue, paw


def motion_energy(dsfile, f, trials, gocue):
    """loadMotionEnergy.m : interpolate motion energy onto the neural time axis"""
    mefile = dsfile.replace('data_structure', 'motionEnergy')
    if not os.path.exists(mefile):
        return np.full((len(trials), NT), np.nan)
    me = sio.loadmat(mefile)['me']
    data = me['data'][0, 0]
    # some sessions store me.data as a struct with its own .data field (loadMotionEnergy.m)
    if data.dtype.names is not None and 'data' in data.dtype.names:
        data = data['data'][0, 0]
    if data.dtype != object:
        return np.full((len(trials), NT), np.nan)
    vidshift = video_offset(f)
    g = f[f['obj/traj'][0, 0]]
    ftrefs = np.array(g['frameTimes']).ravel()
    out = np.full((len(trials), NT), np.nan)
    for i, t in enumerate(trials):
        if t >= data.shape[0]:
            continue
        d = np.array(data[t, 0]).ravel().astype(float)
        ft = np.array(f[ftrefs[t]]).ravel().astype(float)
        if d.size < 2 or ft.size != d.size or np.all(np.isnan(ft)):
            continue
        tt = ft - vidshift - gocue[t]
        out[i] = interp_nan(TAXIS, tt, d)
        # fill edge NaNs (time points outside of the video) with nearest value,
        # as in loadMotionEnergy.m
        v = out[i]
        if np.any(np.isnan(v)) and np.any(~np.isnan(v)):
            good = np.where(~np.isnan(v))[0]
            idx = np.clip(np.searchsorted(good, np.arange(NT)), 0, len(good) - 1)
            prev = np.maximum(idx - 1, 0)
            choose = np.where(np.abs(good[idx] - np.arange(NT)) <= np.abs(good[prev] - np.arange(NT)),
                              good[idx], good[prev])
            out[i] = np.where(np.isnan(v), v[choose], v)
    return out


def discretize(x):
    """per-session median split; NaN (not visible / no video) -> 2"""
    out = np.full(x.shape, 2, dtype=np.int64)
    finite = np.isfinite(x)
    if np.any(finite):
        thresh = np.percentile(x[finite], 50)
        out[finite & (x < thresh)] = 0
        out[finite & (x >= thresh)] = 1
    return out


def main():
    meta = parse_meta_scripts()
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_structure_*.mat')))

    data = {'neural': [], 'input': [], 'output': [], 'subjects': [],
            'subject_idx': [], 'brain_regions': [], 'brain_region_idx': [],
            'input_names': ['time_from_go_cue'],
            'output_names': ['lick_direction', 'context', 'outcome',
                             'tongue_velocity', 'paw_velocity', 'motion_energy'],
            'output_values': [['left', 'right', 'none'],
                              ['DR', 'WC'],
                              ['incorrect', 'correct', 'ignore'],
                              ['below_median', 'above_median', 'not_visible'],
                              ['below_median', 'above_median', 'not_visible'],
                              ['below_median', 'above_median', 'no_video']],
            'metadata': {}}
    session_info = []
    subjects = []

    time_input = TAXIS.astype(np.float32).reshape(1, NT)

    for dsfile in files:
        base = os.path.basename(dsfile)
        m = re.match(r'data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat', base)
        anm, date = m.group(1), m.group(2)
        probes = meta.get((anm, date))
        if probes is None:
            print(f'{base}: no meta entry, skipping')
            continue
        print(f'{base}: probes {probes}')
        res = load_session(dsfile, probes)
        if res is None:
            continue

        rates = res['rates']
        ntr = rates.shape[1]
        tongue = discretize(res['tongue'])
        paw = discretize(res['paw'])
        me = discretize(res['me'])

        neural_sess = [np.ascontiguousarray(rates[:, i, :]) for i in range(ntr)]
        input_sess = [time_input.copy() for _ in range(ntr)]
        output_sess = []
        for i in range(ntr):
            out = np.empty((6, NT), dtype=np.int64)
            out[0] = res['lick_dir'][i]
            out[1] = res['context'][i]
            out[2] = res['outcome'][i]
            out[3] = tongue[i]
            out[4] = paw[i]
            out[5] = me[i]
            output_sess.append(out)

        if anm not in subjects:
            subjects.append(anm)
        data['neural'].append(neural_sess)
        data['input'].append(input_sess)
        data['output'].append(output_sess)
        data['subject_idx'].append(subjects.index(anm))
        for rg in res['regions']:
            if rg not in data['brain_regions']:
                data['brain_regions'].append(rg)
        data['brain_region_idx'].append(
            np.array([data['brain_regions'].index(rg) for rg in res['regions']], dtype=np.int64))
        session_info.append({'animal': anm, 'date': date, 'probes': probes,
                             'n_units': int(rates.shape[0]), 'n_trials': int(ntr)})
        print(f'   -> {rates.shape[0]} units, {ntr} trials')

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description': (
            'Head-fixed mice performed alternating blocks of a delayed-response (DR) task '
            '(auditory cue indicates reward port, go cue instructs directional lick after a delay) '
            'and a water-cued (WC) task (water delivered at a random time and port, no auditory cues). '
            'Decoder predicts lick direction, behavioral context (DR/WC), trial outcome, and '
            'discretized tongue speed, paw speed and motion energy from ALM population activity.'),
        'time_bin_size': DT * 1000,
        'temporal_alignment_event': 'go cue onset (water drop presentation on water-cued trials)',
        'off_start': TMIN,
        'off_end': TMAX,
        'source': ('Hasnain, Birnbaum et al. 2024, Separating cognitive and motor processes in the '
                   'behaving mouse; ALM silicon probe recordings with high-speed video '
                   '(data/Ephys_Behavior)'),
        'neural_preprocessing': (f'spike counts in {DT*1000:.0f} ms bins converted to firing rate and '
                                 f'smoothed with a causal Gaussian kernel (window {SMOOTH} bins), as in '
                                 'the authors\' mySmooth.m/getSeq.m'),
        'unit_curation': (f'clusters labelled garbage/noisy excluded; units with mean firing rate '
                          f'<= {LOW_FR} Hz excluded; sessions with < {MIN_UNITS} units excluded'),
        'trial_curation': ('completed trials only (hit|miss|no); early-lick trials and optogenetic '
                           'photoinactivation trials excluded'),
        'session_info': session_info,
    }

    with open(OUT_FILE, 'wb') as fh:
        pickle.dump(data, fh)
    print(f'saved {OUT_FILE}: {len(data["neural"])} sessions, '
          f'{sum(len(s) for s in data["neural"])} trials')


if __name__ == '__main__':
    main()
