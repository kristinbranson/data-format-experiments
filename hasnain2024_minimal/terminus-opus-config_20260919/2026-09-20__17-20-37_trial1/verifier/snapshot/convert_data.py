"""
Convert ALM electrophysiology + behavior data from Bhandari, Birnbaum et al.,
"Separating cognitive and motor processes in the behaving mouse" (Nat Neurosci)
into the decoder dataset format.

Processing follows the paper's MATLAB pipeline (/app/code):
  - sessions/probes: the ALM recording sessions listed in
    code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m (25 sessions,
    10 mice), i.e. the two-context (DR + WC) / DR ephys dataset used in the
    main figures.
  - alignment: params.alignEvent = 'goCue' (in WC blocks this event is the
    water drop), params.tmin = -2.5 s, params.tmax = 2.5 s, params.dt = 1/100 s
    (getDefaultParams.m / Scripts/Figure*/*.m).
  - neural: spikes binned at dt and smoothed with a causal Gaussian kernel
    (mySmooth.m, params.smooth = 15, params.bctype = 'reflect'), giving
    firing rate (spikes/s) per unit and bin (getSeq.m).
  - unit curation: clusters labelled 'garbage'/'noisy'/'real?' are dropped
    (findClusters.m) and units with mean firing rate <= 1 Hz are removed
    (removeLowFRClusters.m with params.lowFR = 1, as in the figure scripts and
    the Methods: "All units with firing rates exceeding 1 Hz were included").
  - trial curation: early-lick trials and photoinactivation trials are excluded
    (all params.condition strings contain ~early & ~stim.enable; the Methods
    state early-lick trials were omitted from all analyses). Ignore (no-response)
    trials are kept because 'ignore' is one of the outcome classes to decode.
  - session curation: sessions need >= 10 units after curation (Methods).
  - video: DLC trajectories are aligned with
    frameTimes - vidshift - alignTime, vidshift = mode(sglx.bitcode.bitstart)/fs
    - mode(bp.ev.bitStart) (findVideoOffset.m, findPosition.m), and motion
    energy is interpolated onto the same time axis (loadMotionEnergy.m).
"""

import os
import pickle
import sys

import numpy as np
import scipy.io as sio
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sessionio import Session

DATA_DIR = '/app/data/Ephys_Behavior'
OUT_FILE = '/app/converted_data.pkl'

# ---------------------------------------------------------------- parameters
TMIN = -2.5           # s relative to go cue (params.tmin)
TMAX = 2.5            # s relative to go cue (params.tmax)
DT = 0.01             # s, bin size (params.dt = 1/100)
SMOOTH_N = 15         # params.smooth
LOW_FR = 1.0          # Hz, params.lowFR used in all figure scripts
MIN_UNITS = 10        # Methods: sessions need at least 10 units
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')  # findClusters.m

# ALM sessions and probes, transcribed from
# code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m
# (only the sessions that are NOT commented out there).
SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ('JEB7',  '2021-04-29', [1]),
    ('JEB7',  '2021-04-30', [1]),
    ('EKH1',  '2021-08-07', [2]),
    ('EKH3',  '2021-08-11', [2]),
    ('JGR2',  '2021-11-16', [1]),
    ('JGR2',  '2021-11-17', [1]),
    ('JGR3',  '2021-11-18', [1]),
    ('JEB13', '2022-09-13', [2]),
    ('JEB13', '2022-09-14', [2]),
    ('JEB13', '2022-09-21', [1]),
    ('JEB13', '2022-09-24', [1]),
    ('JEB13', '2022-09-25', [1]),
    ('JEB14', '2022-08-22', [1]),
    ('JEB14', '2022-08-23', [1]),
    ('JEB14', '2022-08-24', [1]),
    ('JEB14', '2022-08-25', [1]),
    ('JEB15', '2022-07-26', [1, 2]),
    ('JEB15', '2022-07-27', [1, 2]),
    ('JEB15', '2022-07-28', [1, 2]),
    ('JEB15', '2022-07-29', [2]),
    ('JEB19', '2023-04-18', [1]),
    ('JEB19', '2023-04-19', [1]),
    ('JEB19', '2023-04-20', [1]),
    ('JEB19', '2023-04-21', [1]),
]

EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2          # bin centres, as in getSeq.m
NT = len(TIME)


def causal_gaussian_kernel(n=SMOOTH_N):
    """MATLAB gausswin(n) (alpha=2.5) with the acausal half zeroed, normalised."""
    k = np.arange(n)
    alpha = 2.5
    w = np.exp(-0.5 * (alpha * (k - (n - 1) / 2) / ((n - 1) / 2)) ** 2)
    w[:n // 2] = 0.0                # 'causal' step in mySmooth.m
    return w / w.sum()


KERN = causal_gaussian_kernel()


def smooth_causal(x):
    """mySmooth(x, 15, 'reflect') along the last axis."""
    pad = x[..., :SMOOTH_N]
    xf = np.concatenate([pad, x], axis=-1)
    out = convolve1d(xf, KERN, axis=-1, mode='constant', cval=0.0)
    return out[..., SMOOTH_N:]


def region_from_loc(loc):
    """Map obj.ex.probe.loc strings onto brain region names."""
    if loc is None:
        return 'ALM'
    s = loc.upper()
    if 'ALM' in s:
        return 'ALM'
    if 'M1TJ' in s or 'TJM1' in s:
        return 'tjM1'
    return 'ALM'   # all load*_ALMVideo sessions are ALM recordings


def interp_to_axis(src_t, src_y, dst_t):
    """Linear interpolation (NaN outside range / across NaNs, like MATLAB interp1)."""
    f = interp1d(src_t, src_y, kind='linear', bounds_error=False,
                 fill_value=np.nan, axis=0, assume_sorted=False)
    return f(dst_t)


def fill_nearest(y):
    """fillmissing(y,'nearest') for a 1-d array."""
    y = np.asarray(y, float)
    good = np.isfinite(y)
    if not good.any():
        return y
    idx = np.arange(len(y))
    return np.interp(idx, idx[good], y[good])


def nan_gradient(pos):
    """Per-sample velocity of a (T,2) position trace that contains NaNs.

    Central difference where both neighbours are tracked, one-sided at the
    edges of a visibility bout, NaN where the marker is not tracked. This is
    np.gradient for fully tracked stretches but does not let a single missing
    frame erase the velocity of its tracked neighbours (which would
    artificially inflate the 'not visible' class).
    """
    T = pos.shape[0]
    vel = np.full_like(pos, np.nan)
    fwd = np.full_like(pos, np.nan)
    bwd = np.full_like(pos, np.nan)
    fwd[:-1] = pos[1:] - pos[:-1]
    bwd[1:] = pos[1:] - pos[:-1]
    both = np.isfinite(fwd) & np.isfinite(bwd)
    vel[both] = 0.5 * (fwd[both] + bwd[both])
    only_f = np.isfinite(fwd) & ~np.isfinite(bwd)
    vel[only_f] = fwd[only_f]
    only_b = np.isfinite(bwd) & ~np.isfinite(fwd)
    vel[only_b] = bwd[only_b]
    # tracked sample with no tracked neighbour: no velocity estimate, call it 0
    isolated = np.isfinite(pos) & ~np.isfinite(vel)
    vel[isolated] = 0.0
    return vel


def process_session(anm, date, probes, verbose=True):
    path = os.path.join(DATA_DIR, 'data_structure_%s_%s.mat' % (anm, date))
    s = Session(path)

    ntrials_all = s.ntrials
    gocue = s.ev('goCue')
    early = s.bp_flag('early')
    stim = s.stim_enable()
    hit = s.bp_flag('hit')
    miss = s.bp_flag('miss')
    no = s.bp_flag('no')
    R = s.bp_flag('R')
    L = s.bp_flag('L')
    autowater = s.bp_flag('autowater')   # 1 in water-cued (WC) blocks

    # ------------------------------------------------ trial curation
    keep = (~early) & (~stim) & np.isfinite(gocue)
    trials = np.where(keep)[0]

    # ------------------------------------------------ neural data
    fr_list = []
    region_list = []
    locs = s.probe_locs()
    for prb in probes:
        qual = s.qualities(prb)
        cluix = [i for i, q in enumerate(qual) if q.lower() not in BAD_QUALITY]
        if len(cluix) == 0:
            continue
        counts = np.zeros((len(cluix), len(trials), NT), dtype=np.float32)
        trial_pos = -np.ones(ntrials_all + 1, dtype=int)
        trial_pos[trials + 1] = np.arange(len(trials))   # bp trials are 1-indexed
        for ii, ci in enumerate(cluix):
            tr, ttm = s.clu_spikes(prb, ci)
            aligned = ttm - gocue[tr - 1]                 # alignSpikes.m
            bi = np.floor((aligned - TMIN) / DT).astype(int)
            ti = trial_pos[tr]
            ok = (bi >= 0) & (bi < NT) & (ti >= 0)
            np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
        rate = smooth_causal(counts / DT)                 # getSeq.m
        fr_list.append(rate)
        loc = locs[prb - 1] if len(locs) >= prb else None
        region_list += [region_from_loc(loc)] * len(cluix)

    if len(fr_list) == 0:
        s.close()
        return None
    fr = np.concatenate(fr_list, axis=0)                  # (units, trials, time)
    regions = np.array(region_list)

    # low firing rate units (removeLowFRClusters.m)
    meanfr = fr.mean(axis=(1, 2))
    use = meanfr > LOW_FR
    fr = fr[use]
    regions = regions[use]
    if fr.shape[0] < MIN_UNITS or len(trials) < 2:
        if verbose:
            print('  skipping %s %s: %d units, %d trials' %
                  (anm, date, fr.shape[0], len(trials)))
        s.close()
        return None

    # ------------------------------------------------ video kinematics
    vidshift = s.video_offset()
    feats0 = s.traj_featnames(0)     # side cam
    feats1 = s.traj_featnames(1)     # bottom cam
    tongue_ix = feats0.index('tongue')
    paw_ix = feats1.index('top_paw')
    ndrop = s.ndropped(0)

    tongue_speed = np.full((len(trials), NT), np.nan)
    paw_speed = np.full((len(trials), NT), np.nan)
    has_video = np.zeros(len(trials), dtype=bool)

    for k, tr in enumerate(trials):
        if ndrop is not None and not np.isfinite(ndrop[tr]):
            continue                                        # bad video trial
        taxis = TIME + gocue[tr] + vidshift                 # into video clock
        ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
        if ft is None or xy is None or not np.isfinite(ft).any():
            continue
        has_video[k] = True
        pos = interp_to_axis(ft, xy, taxis)
        vel = nan_gradient(pos)
        tongue_speed[k] = np.hypot(vel[:, 0], vel[:, 1])
        ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
        if ft1 is None or xy1 is None or not np.isfinite(ft1).any():
            continue
        pos1 = interp_to_axis(ft1, xy1, taxis)
        vel1 = nan_gradient(pos1)
        # findVelocity.m subtracts the baseline (median) frame-to-frame
        # displacement for all non-tongue features, removing slow drift
        base = np.nanmedian(np.diff(pos1, axis=0), axis=0)
        vel1 = vel1 - base[None, :]
        paw_speed[k] = np.hypot(vel1[:, 0], vel1[:, 1])

    # ------------------------------------------------ motion energy
    me_arr = np.full((len(trials), NT), np.nan)
    mepath = os.path.join(DATA_DIR, 'motionEnergy_%s_%s.mat' % (anm, date))
    if os.path.exists(mepath):
        medat = sio.loadmat(mepath)['me']['data'][0, 0]
        # loadMotionEnergy.m: 'if isstruct(me.data), me.data = me.data.data'
        while medat.dtype.names is not None and 'data' in medat.dtype.names:
            medat = medat['data'][0, 0]
        medat = medat.ravel()
        for k, tr in enumerate(trials):
            if not has_video[k] or tr >= len(medat):
                continue
            m = np.asarray(medat[tr], float).ravel()
            ft, _ = s.traj_feat_trial(0, tr, tongue_ix)
            if ft is None or len(ft) != len(m):
                continue
            taxis = TIME + gocue[tr] + vidshift
            vals = interp_to_axis(ft, m, taxis)
            me_arr[k] = fill_nearest(vals)   # loadMotionEnergy.m fills nans

    # ------------------------------------------------ discretise movement
    def discretise(x):
        """0: < session median, 1: >= session median, 2: not visible / no video."""
        out = np.full(x.shape, 2, dtype=np.int64)
        vis = np.isfinite(x)
        if vis.any():
            thr = np.percentile(x[vis], 50)
            out[vis] = (x[vis] >= thr).astype(np.int64)
        return out

    tongue_d = discretise(tongue_speed)
    paw_d = discretise(paw_speed)
    me_d = discretise(me_arr)

    # ------------------------------------------------ per-trial variables
    # lick direction: 0 left, 1 right, 2 none (no response / ignore trial)
    lick_dir = np.full(len(trials), 2, dtype=np.int64)
    lick_dir[hit[trials] & R[trials]] = 1
    lick_dir[hit[trials] & L[trials]] = 0
    lick_dir[miss[trials] & R[trials]] = 0     # error trial: licked other way
    lick_dir[miss[trials] & L[trials]] = 1
    # context: 0 WC (autowater block), 1 DR
    context = np.where(autowater[trials], 0, 1).astype(np.int64)
    # outcome: 0 incorrect, 1 correct, 2 ignore
    outcome = np.full(len(trials), 0, dtype=np.int64)
    outcome[hit[trials]] = 1
    outcome[no[trials]] = 2

    # ------------------------------------------------ assemble
    neural = [np.ascontiguousarray(fr[:, k, :]) for k in range(len(trials))]
    inp = [np.asarray(TIME, dtype=np.float32).reshape(1, NT)] * len(trials)
    outputs = []
    for k in range(len(trials)):
        o = np.empty((6, NT), dtype=np.int64)
        o[0] = lick_dir[k]
        o[1] = context[k]
        o[2] = outcome[k]
        o[3] = tongue_d[k]
        o[4] = paw_d[k]
        o[5] = me_d[k]
        outputs.append(o)

    info = dict(animal=anm, date=date, probes=list(probes),
                n_units=int(fr.shape[0]), n_trials=int(len(trials)),
                n_trials_total=int(ntrials_all),
                n_wc_trials=int(np.sum(context == 0)),
                n_dr_trials=int(np.sum(context == 1)),
                n_video_trials=int(has_video.sum()))
    if verbose:
        print('  %s %s: %d units, %d/%d trials (WC %d), video %d' %
              (anm, date, fr.shape[0], len(trials), ntrials_all,
               info['n_wc_trials'], info['n_video_trials']), flush=True)
    s.close()
    return dict(neural=neural, input=inp, output=outputs, regions=regions,
                subject=anm, info=info)


def main(limit=None):
    sessions = SESSIONS if limit is None else SESSIONS[:limit]
    data = dict(neural=[], input=[], output=[], subjects=[], subject_idx=[],
                brain_regions=['ALM', 'tjM1'], brain_region_idx=[])
    session_info = []
    for anm, date, probes in sessions:
        print('processing %s %s' % (anm, date), flush=True)
        res = process_session(anm, date, probes)
        if res is None:
            continue
        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        if res['subject'] not in data['subjects']:
            data['subjects'].append(res['subject'])
        data['subject_idx'].append(data['subjects'].index(res['subject']))
        data['brain_region_idx'].append(
            np.array([data['brain_regions'].index(r) for r in res['regions']],
                     dtype=np.int64))
        session_info.append(res['info'])

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
    data['metadata'] = {
        'task_description':
            'Head-fixed mice performed two directional licking tasks that '
            'alternated block-wise: a delayed-response (DR) task in which an '
            'auditory sample cue instructed the reward side and an auditory go '
            'cue instructed movement after a delay, and a water-cued (WC) task '
            'in which water was delivered at a random time and random port with '
            'no auditory cues. Decoder predicts lick direction, behavioral '
            'context (WC/DR), trial outcome, and discretized tongue velocity, '
            'paw velocity and facial motion energy from ALM (and tjM1) '
            'population activity.',
        'time_bin_size': DT * 1000.0,
        'temporal_alignment_event':
            'go cue onset (auditory go cue in DR blocks, water drop in WC blocks)',
        'off_start': TMIN,
        'off_end': TMAX,
        'neural_data_description':
            'Single-trial firing rate (spikes/s): spikes binned at 10 ms and '
            'smoothed with a causal Gaussian kernel (15-bin window), as in the '
            "paper's getSeq.m/mySmooth.m.",
        'unit_curation':
            "Clusters labelled 'garbage'/'noisy'/'real?' removed (findClusters.m); "
            'units with mean firing rate <= 1 Hz removed (params.lowFR = 1); '
            'sessions with fewer than 10 remaining units excluded.',
        'trial_curation':
            'Early-lick trials and optogenetic photoinactivation trials excluded '
            '(all params.condition strings use ~early & ~stim.enable). Ignore '
            'trials are retained because they form one of the outcome classes.',
        'output_description': {
            'lick_direction': 'direction of the first lick after the go cue '
                              '(none on ignore trials)',
            'context': 'block type, WC = water-cued (bp.autowater), DR = delayed response',
            'outcome': 'incorrect (error lick), correct (rewarded), ignore (no lick)',
            'tongue_velocity': 'speed of the side-camera DLC tongue marker, '
                               'split at the session 50th percentile over '
                               'timepoints where the tongue is visible; class 2 '
                               'when the tongue is not visible',
            'paw_velocity': 'speed of the bottom-camera DLC top_paw marker, '
                            'split at the session 50th percentile over visible '
                            'timepoints; class 2 when the paw is not visible',
            'motion_energy': 'frame-to-frame motion energy interpolated onto the '
                             'neural time axis, split at the session 50th '
                             'percentile; class 2 when no video is available',
        },
        'video_alignment':
            'DLC/motion-energy frame times shifted by '
            'mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart) '
            '(findVideoOffset.m) and aligned to the go cue.',
        'session_info': session_info,
        'source': 'Bhandari, Birnbaum et al., Separating cognitive and motor '
                  'processes in the behaving mouse; ALM ephys + video sessions '
                  '(load<ANM>_ALMVideo.m).',
    }

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    ntr = sum(len(x) for x in data['neural'])
    nun = sum(len(x) for x in data['brain_region_idx'])
    print('saved %s: %d sessions, %d trials, %d units, %d subjects' %
          (OUT_FILE, len(data['neural']), ntr, nun, len(data['subjects'])))


if __name__ == '__main__':
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lim)
