"""
Convert the Sosa, Plitt & Giocomo (2024/2025) hippocampal CA1 2P imaging dataset
("A flexible hippocampal population code for experience relative to reward",
DANDI:001361) into the decoder-training format described in the task.

The conversion reproduces the preprocessing of the paper / reference repository
(`/app/code`, module `reward_relative`):

  * neural signal  : per-trial maximin dF/F (neuropil-subtracted, neu_coef 0.7,
                     20 s min/max window, 2-sample Gaussian smoothing) followed by
                     OASIS deconvolution -- i.e. the "events" timeseries that the
                     paper uses for place-cell detection, position decoding and the
                     GLM (reward_relative.preprocessing.dff with the parameters set
                     in notebooks/make_multi_anim_sess.md).
  * cell curation  : suite2p manual curation (iscell) + exclusion of putative
                     interneurons with Pearson r > 0.5 between their dF/F and the
                     animal's running speed (reward_relative.spatial.is_putative_interneuron
                     with the int_thresh = 0.5 used in dayData).
  * trial structure: one trial = one lap, from the VR "trial_start" frame to the
                     "teleport" frame, sliced exactly as the reference dff() does.
  * behaviour      : lick-sensor artefact trials removed as in
                     reward_relative.behavior.correct_lick_sensor_error.

Usage:  python /app/convert_data.py [--out /app/converted_data.pkl] [--workers N]
"""

import argparse
import os
import pickle
import re
import sys
import traceback
from multiprocessing import get_context

import h5py
import numpy as np
import scipy.ndimage as ndi
from suite2p.extraction import dcnv

# --------------------------------------------------------------------------------------
# Constants taken from the paper / reference code
# --------------------------------------------------------------------------------------

DATA_ROOT = '/app/data'
OUT_PATH = '/app/converted_data.pkl'
CACHE_DIR = '/app/.session_cache'

# reward_relative.behavior.reward_zone_dict: the three hidden 50 cm zones (keys X, Y, Z)
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_LABELS = ['A', 'B', 'C']

TRACK_LENGTH = 450.0          # cm
SWITCH_TRIAL = 30             # reward zone is moved after 30 trials (behavior.get_reward_zones)
FRAME_RATE = 15.5078125       # Hz, per imaging plane
DT = 1.0 / FRAME_RATE         # s

# dff parameters (utilities.default_dff_method / notebooks/make_multi_anim_sess.md)
NEU_COEF = 0.7
TAU = 0.7                     # suite2p ops['tau'], GCaMP indicator decay
BASELINE_WIN = 300            # samples ~= 20 s minimum/maximum filter window
BASELINE_SMOOTH_SIG = 15      # samples, Gaussian sigma used before the maximin filter
DFF_SMOOTH_SIG = 2            # samples (~0.129 s) Gaussian sigma on dF/F

INT_R_THRESH = 0.5            # speed-correlation threshold for putative interneurons
# Methods: trials in which >30% of the imaging frames contain a cumulative lick count > 2
# are lick-sensor artefacts (this reproduces the paper's count of 81 affected trials).
LICK_ERROR_THRESH = 0.30

# teleport_metadata.teleport_sessions: (animal, experiment day) pairs for which the laser
# was NOT blanked during the inter-trial teleport period, so the dF/F baseline is computed
# over trial+teleport rather than over the trial alone.
TELEPORT_SESSIONS = {
    'm10': [1, 7, 8, 14, 15],
    'm11': [1, 7, 8, 14, 15],
    'm12': [1, 7, 8, 14, 15],
    'm13': [1, 7, 8, 14, 15],
    'm14': [1, 7, 8, 14, 15],
    'm15': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm17': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm18': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
    'm19': [1, 3, 5, 7, 8, 10, 12, 14, 15, 16, 17],
}

# Output discretisation (edges are used with np.digitize, i.e. left-closed intervals)
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
SPEED_EDGES = [2.0, 10.0, 20.0, 40.0]

INPUT_NAMES = ['time_from_trial_start',
               'environment',
               'trial_number',
               'previous_trial_outcome']

OUTPUT_NAMES = ['reward_zone_distance',
                'position',
                'speed',
                'lick',
                'reward_zone_location',
                'reward_outcome']

OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm (inside reward zone)',
     '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
    ['omitted', 'rewarded'],
]


# --------------------------------------------------------------------------------------
# dF/F + deconvolution, following reward_relative.preprocessing.dff
# --------------------------------------------------------------------------------------

def compute_dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports):
    """Maximin dF/F and OASIS-deconvolved "events", as in reward_relative.preprocessing.dff.

    F, Fneu : (ncells, nframes) raw suite2p ROI and neuropil fluorescence.
    trial_starts, teleports : frame indices of lap start / teleport (int arrays).
    keep_teleports : if True the baseline segment spans trial + preceding teleport
        period (sessions in which the laser was not blanked between laps).

    Returns (dff, events), both (ncells, nframes), NaN outside the processed segments.
    """
    if keep_teleports:
        start_inds = [int(trial_starts[0])] + (np.asarray(teleports[:-1]) + 2).tolist()
    else:
        start_inds = [int(s) for s in trial_starts]
    stop_inds = [int(s) for s in teleports]
    segments = [(a - 1, b - 1) for a, b in zip(start_inds, stop_inds)]

    f_ = np.full(F.shape, np.nan, dtype=np.float64)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float64)
    for a, b in segments:
        f_[:, a:b] = F[:, a:b]
        fneu_[:, a:b] = Fneu[:, a:b]

    nanmask = ~np.isnan(f_[0, :])

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan, dtype=np.float64)
    for a, b in segments:
        # add the trial's mean neuropil back so dF/F is not divided by a near-zero baseline
        f_[:, a:b] += NEU_COEF * np.nanmean(fneu_[:, a:b], axis=1, keepdims=True)
        seg = ndi.gaussian_filter1d(f_[:, a:b], BASELINE_SMOOTH_SIG, axis=1)
        seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
        seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
        flow[:, a:b] = seg

    dff = np.full(F.shape, np.nan, dtype=np.float64)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    events = np.full(F.shape, np.nan, dtype=np.float32)
    for a, b in segments:
        dff[:, a:b] = ndi.gaussian_filter1d(dff[:, a:b], DFF_SMOOTH_SIG, axis=1)
        events[:, a:b] = dcnv.oasis(np.ascontiguousarray(dff[:, a:b], dtype=np.float32),
                                    2000, TAU, FRAME_RATE)
    return dff, events


# --------------------------------------------------------------------------------------
# per-session conversion
# --------------------------------------------------------------------------------------

def scene_reward_zones(scene, ntrials):
    """Per-trial reward-zone label from the VR scene name (behavior.get_reward_zones).

    Scenes are either 'Env<n>_Location<Z>' (one zone all session), 'Env<n>_Location<Z1>_to_<Z2>'
    (zone switch after 30 trials) or 'Env<n>_<Z1>_to_Env<m>_<Z2>' (zone + environment switch).
    """
    m = re.fullmatch(r'Env\d_Location([ABC])', scene)
    if m:
        return [m.group(1)] * ntrials
    m = (re.fullmatch(r'Env\d_Location([ABC])_to_([ABC])', scene)
         or re.fullmatch(r'Env\d_([ABC])_to_Env\d_([ABC])', scene))
    if m:
        return [m.group(1)] * SWITCH_TRIAL + [m.group(2)] * (ntrials - SWITCH_TRIAL)
    raise ValueError(f'unrecognised scene name: {scene}')


def reward_zone_distance_bin(pos, zstart, zstop):
    """Signed distance to the nearest point of the reward zone, discretised into 7 bins."""
    d = np.zeros_like(pos)
    before = pos < zstart
    after = pos > zstop
    d[before] = pos[before] - zstart
    d[after] = pos[after] - zstop
    out = np.full(pos.shape, 3, dtype=np.int64)          # 3: inside the zone (d == 0)
    out[d < -50.0] = 0
    out[(d >= -50.0) & (d < -10.0)] = 1
    out[(d >= -10.0) & (d < 0.0)] = 2
    out[(d > 0.0) & (d <= 10.0)] = 4
    out[(d > 10.0) & (d <= 50.0)] = 5
    out[d > 50.0] = 6
    return out


def convert_session(path):
    """Convert a single NWB session file. Returns a dict (or None if nothing survives)."""
    with h5py.File(path, 'r') as f:
        subject = f['general/subject/subject_id'][()].decode()
        exp_day = int(f['general/session_id'][()].decode())
        scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]

        beh = f['processing/behavior/BehavioralTimeSeries']
        ts = beh['position/timestamps'][:]
        nframes = len(ts)
        pos = beh['position/data'][:]
        speed = beh['speed/data'][:]
        lick = beh['lick/data'][:]
        env_ts = beh['environment/data'][:]
        scanning = beh['scanning/data'][:]
        trial_starts = np.flatnonzero(beh['trial_start/data'][:] > 0)
        teleports = np.flatnonzero(beh['teleport/data'][:] > 0)
        reward_ts = beh['Reward/timestamps'][:]

        # ophys: pool imaging planes (the paper pools deep/superficial planes)
        plane_names = sorted(f['processing/ophys/Fluorescence'].keys())
        F = np.concatenate([f['processing/ophys/Fluorescence'][p]['data'][:nframes, :]
                            for p in plane_names], axis=1).T
        Fneu = np.concatenate([f['processing/ophys/Neuropil'][p]['data'][:nframes, :]
                               for p in plane_names], axis=1).T
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] > 0

    assert F.shape[1] == nframes and len(trial_starts) == len(teleports)

    # ---- cell curation: suite2p manual curation --------------------------------------
    F = np.ascontiguousarray(F[iscell], dtype=np.float64)
    Fneu = np.ascontiguousarray(Fneu[iscell], dtype=np.float64)
    n_curated = F.shape[0]

    keep_teleports = exp_day in TELEPORT_SESSIONS.get(subject, [])
    dff, events = compute_dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports)
    del F, Fneu

    # ---- cell curation: drop putative interneurons (speed-correlated) ----------------
    valid = ~np.isnan(dff[0, :])
    sp_v = speed[valid]
    dv = dff[:, valid]
    dv = dv - dv.mean(axis=1, keepdims=True)
    spc = sp_v - sp_v.mean()
    denom = np.sqrt((dv ** 2).sum(axis=1) * (spc ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_corr = (dv @ spc) / denom
    is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
    keep_cells = ~is_int
    events = events[keep_cells]
    del dff, dv

    n_neurons = int(keep_cells.sum())
    if n_neurons == 0:
        return None

    # ---- per-trial task variables -----------------------------------------------------
    ntrials = len(trial_starts)
    zone_labels = scene_reward_zones(scene, ntrials)

    # reward delivery: map the reward event timestamps onto imaging frames
    rew_frames = np.searchsorted(ts, reward_ts)
    rewarded = np.zeros(ntrials, dtype=np.int64)
    environment = np.zeros(ntrials, dtype=np.int64)
    lick_error = np.zeros(ntrials, dtype=bool)
    for i, (a, b) in enumerate(zip(trial_starts, teleports)):
        rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
        env_vals = np.unique(env_ts[a:b])
        assert len(env_vals) == 1, f'{path}: trial {i} has environments {env_vals}'
        environment[i] = int(env_vals[0])
        # behavior.correct_lick_sensor_error: capacitive sensor stuck on
        lick_error[i] = np.mean(lick[a:b] > 2) > LICK_ERROR_THRESH

    # ---- assemble trials ---------------------------------------------------------------
    neural_trials, input_trials, output_trials = [], [], []
    kept_trials = []
    for i in range(ntrials):
        # The first imaged lap has no known preceding outcome (the ~30 un-imaged warm-up
        # laps are not in the file), so it cannot carry the "previous trial outcome" input.
        if i == 0:
            continue
        if lick_error[i]:
            continue
        # slice exactly as the reference dff() does: [trial_start-1, teleport-1)
        a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
        if b - a < 2:
            continue
        assert np.all(scanning[a:b] > 0)

        neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
        if not np.all(np.isfinite(neu)):
            continue
        T = b - a

        t_rel = (ts[a:b] - ts[a]).astype(np.float32)
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = t_rel
        inp[1] = environment[i]
        inp[2] = i
        inp[3] = rewarded[i - 1]

        zstart, zstop = REWARD_ZONES[zone_labels[i]]
        p = pos[a:b]
        out = np.empty((6, T), dtype=np.int64)
        out[0] = reward_zone_distance_bin(p, zstart, zstop)
        out[1] = np.digitize(p, POSITION_EDGES)
        out[2] = np.digitize(speed[a:b], SPEED_EDGES)
        out[3] = (lick[a:b] > 0).astype(np.int64)
        out[4] = ZONE_LABELS.index(zone_labels[i])
        out[5] = rewarded[i]

        neural_trials.append(neu)
        input_trials.append(inp)
        output_trials.append(out)
        kept_trials.append(i)

    if len(neural_trials) < 2:
        return None

    info = {
        'subject': subject,
        'experiment_day': exp_day,
        'scene': scene,
        'environments': sorted({int(environment[i]) for i in kept_trials}),
        'reward_zones': sorted({zone_labels[i] for i in kept_trials}),
        'reward_switch_trial': SWITCH_TRIAL if len(set(zone_labels)) > 1 else None,
        'n_trials': len(neural_trials),
        'n_trials_recorded': ntrials,
        'n_trials_lick_artifact': int(lick_error[1:].sum()),
        'n_neurons': n_neurons,
        'n_neurons_curated': n_curated,
        'n_neurons_putative_interneuron': int(is_int.sum()),
        'imaged_teleport_period': bool(keep_teleports),
        'frac_rewarded': float(np.mean([rewarded[i] for i in kept_trials])),
    }
    return {'neural': neural_trials, 'input': input_trials, 'output': output_trials,
            'subject': subject, 'n_neurons': n_neurons, 'info': info}


def _worker(path):
    cache = os.path.join(CACHE_DIR, os.path.basename(path).replace('.nwb', '.pkl'))
    if os.path.exists(cache):
        return path, cache, None
    try:
        res = convert_session(path)
    except Exception:
        return path, None, traceback.format_exc()
    if res is None:
        return path, None, 'no usable trials'
    tmp = cache + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache)
    return path, cache, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    files = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        d = os.path.join(DATA_ROOT, sub)
        if os.path.isdir(d):
            files += [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.endswith('.nwb')]
    # order sessions by subject number then experiment day
    files.sort(key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                              int(re.search(r'ses-(\d+)', p).group(1))))
    if args.limit:
        files = files[:args.limit]
    print(f'{len(files)} session files', flush=True)

    results = {}
    ctx = get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        for n, (path, cache, err) in enumerate(pool.imap_unordered(_worker, files), 1):
            if err:
                print(f'[{n}/{len(files)}] SKIP {os.path.basename(path)}: {err}', flush=True)
            else:
                results[path] = cache
                print(f'[{n}/{len(files)}] done {os.path.basename(path)}', flush=True)

    data = {'neural': [], 'input': [], 'output': [], 'brain_region_idx': []}
    subject_idx, session_info = [], []
    subjects = sorted({re.search(r'sub-(m\d+)', p).group(1) for p in files},
                      key=lambda s: int(s[1:]))

    for path in files:
        cache = results.get(path)
        if cache is None:
            continue
        with open(cache, 'rb') as fh:
            res = pickle.load(fh)
        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        data['brain_region_idx'].append(np.zeros(res['n_neurons'], dtype=np.int64))
        subject_idx.append(subjects.index(res['subject']))
        session_info.append(res['info'])

    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
    data['brain_regions'] = ['CA1']
    data['input_names'] = INPUT_NAMES
    data['output_names'] = OUTPUT_NAMES
    data['output_values'] = OUTPUT_VALUES

    ntrials = sum(len(s) for s in data['neural'])
    nsamples = sum(t.shape[1] for s in data['neural'] for t in s)
    data['metadata'] = {
        'task_description': (
            'Head-fixed mice run laps on a 450 cm virtual linear track for a hidden 50 cm '
            'reward zone at one of three locations (A 80-130 cm, B 200-250 cm, C 320-370 cm) '
            'in one of two visually distinct environments (ENV 1 / ENV 2). Sucrose reward is '
            'delivered operantly for licking inside the zone and is randomly omitted on ~15% '
            'of laps. On "switch" days the zone moves to a new location after 30 laps, and on '
            'day 8 the switch coincides with a change of environment. From the CA1 population '
            'activity the decoder predicts, at each imaging frame, the signed distance to the '
            'reward zone (7 bins), the absolute track position (5 bins), running speed (5 bins) '
            'and licking (binary), plus the per-lap reward zone identity (A/B/C) and whether '
            'the lap was rewarded.'),
        'time_bin_size': 1000.0 / FRAME_RATE,
        'temporal_alignment_event': (
            'trial (lap) start: the VR frame at which the mouse enters the linear track at '
            'position 0 cm, after the inter-trial teleport period'),
        'off_start': 0.0,
        'off_end': None,
        'trial_duration_note': (
            'Trials are self-paced laps and therefore have variable length; each trial runs '
            'from lap start to the teleport out of the track (median ~12.3 s, no truncation).'),
        'recording': ('two-photon calcium imaging of dorsal CA1 pyramidal neurons expressing '
                      'GCaMP7f, ~15.5 Hz per plane'),
        'neural_signal': (
            'OASIS-deconvolved calcium activity ("events") computed from per-trial maximin '
            'dF/F (neuropil coefficient 0.7, 20 s min/max baseline window, dF/F smoothed with '
            'a 2-sample Gaussian), as in reward_relative.preprocessing.dff with tau=0.7'),
        'neuron_curation': (
            'suite2p manual curation (iscell) followed by removal of putative interneurons '
            'with Pearson r > 0.5 between dF/F and running speed'),
        'trial_curation': (
            'the first imaged lap of each session is dropped because the outcome of the '
            'preceding (un-imaged warm-up) lap is unknown, and laps flagged by the paper\'s '
            'lick-sensor artefact criterion (>30% of frames with a cumulative lick count > 2) '
            'are dropped'),
        'input_descriptions': {
            'time_from_trial_start': 'seconds since the lap started (time-varying)',
            'environment': 'virtual environment, 0 = ENV 1, 1 = ENV 2 (per trial)',
            'trial_number': '0-based index of the lap within the session (per trial)',
            'previous_trial_outcome': 'reward on the preceding lap, 0 = omitted, 1 = rewarded',
        },
        'output_descriptions': {
            'reward_zone_distance': ('signed distance from the animal to the nearest point of '
                                     'the active reward zone (0 while inside the zone)'),
            'position': 'absolute position on the 450 cm track, 5 equal bins',
            'speed': 'running speed',
            'lick': 'at least one lick detected in the imaging frame',
            'reward_zone_location': 'identity of the active reward zone on that lap',
            'reward_outcome': 'whether reward was delivered on that lap',
        },
        'dataset': 'DANDI:001361 - Sosa, Plitt & Giocomo, "A flexible hippocampal population '
                   'code for experience relative to reward"',
        'n_sessions': len(data['neural']),
        'n_trials': ntrials,
        'n_timepoints': nsamples,
        'session_info': session_info,
    }

    print(f"sessions={len(data['neural'])} trials={ntrials} samples={nsamples} "
          f"neurons={sum(len(b) for b in data['brain_region_idx'])}", flush=True)

    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB)')


if __name__ == '__main__':
    sys.exit(main())
