"""
Convert Sosa, Plitt & Giocomo (2025) CA1 2P imaging + VR behavior data (DANDI 001361)
into the decoder dataset format.

Processing follows the reference paper/code (Sosa_et_al_2024 repo) wherever applicable:

* Trials are laps of the 450 cm virtual linear track, from `trial_start` (entry to the
  track) to `teleport` (entry to the inter-trial interval), as in the sess class used
  throughout the repo.  All VR streams in the NWB files are already synchronized to the
  ~15.5 Hz imaging frames, so a frame is the natural time bin (64.5 ms) and no
  resampling is needed to align neural, input and output streams.
* Neural activity: suite2p ROIs curated as cells (`iscell`), neuropil-subtracted
  (F - 0.7*Fneu), dF/F with a per-trial maximin baseline (20 s window) and 2-sample
  Gaussian smoothing -- i.e. reward_relative
  `preprocessing.dff(..., baseline_method='maximin', neu_coef=0.7)`.  The same call
  optionally returns OASIS-deconvolved 'events' (tau = 0.7), and both signals are
  used in the paper.  dF/F is used here (`--signal dff`, the default) because it is
  the signal the paper treats as closest to the raw data (e.g. spatial peak firing,
  sequence analyses) and it retains the graded amplitude information that a linear
  decoder can exploit; deconvolution in the paper served analyses that needed the
  asymmetric calcium kinetics removed (spatial information, RR-position decoding).
  Deconvolved events can be produced instead with `--signal events`; they decode the
  same variables but less accurately (e.g. position 0.51 vs 0.72 balanced accuracy
  on a 4-session test).
* Putative interneurons (Pearson r > 0.5 between dF/F and running speed) are excluded,
  as in `spatial.is_putative_interneuron`.
* Reward zones: A 80-130, B 200-250, C 320-370 cm, switching after 30 trials on switch
  days, as in `behavior.get_reward_zones`/`reward_zone_dict`.
* Reward outcome: reward delivered while in the reward zone, as in
  `behavior.get_trial_types`.
* Trials with lick-sensor errors (>30% of frames with a cumulative lick count > 2) are
  dropped, as these licks are unreliable (`behavior.correct_lick_sensor_error`; the
  paper NaNs them out -- 81 trials here, matching the paper's 81/12,376).
"""

import argparse
import glob
import os
import pickle
import re
import sys
import time

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d


# ---------------------------------------------------------------- constants

REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_NAMES = ['A', 'B', 'C']
SWITCH_TRIAL = 30           # reward zone moves after 30 trials on switch days
NEU_COEF = 0.7              # neuropil coefficient (repo default)
TAU = 0.7                   # GCaMP7f decay constant used for suite2p/OASIS
BASELINE_WIN = 300          # ~20 s at 15.5 Hz, maximin baseline window
BASELINE_SMOOTH = 15        # samples, smoothing before the min/max filter
DFF_SMOOTH = 2              # samples (~0.129 s) Gaussian smoothing of dF/F
INT_R_THRESH = 0.5          # speed-correlation threshold for putative interneurons
LICK_ERR_FRAC = 0.3         # >30% of frames with cumulative lick count > 2
LICK_ERR_COUNT = 2
TRACK_LENGTH = 450.0


def nansmooth(arr, sigma):
    """Gaussian smoothing that ignores NaNs (reward_relative.utilities.nansmooth)."""
    out = np.copy(arr)
    nanmask = np.isnan(out)
    out[nanmask] = 0.0
    num = gaussian_filter(out, sigma)
    den = gaussian_filter((~nanmask).astype(out.dtype), sigma)
    with np.errstate(invalid='ignore', divide='ignore'):
        res = num / den
    res[nanmask] = np.nan
    return res


def scene_reward_zones(scene, ntrials, change_trial=SWITCH_TRIAL):
    """Reward zone label for each trial, from the VR scene name.

    Mirrors reward_relative.behavior.get_reward_zones: on switch scenes
    (`..._X_to_Y`) the zone moves after `change_trial` trials.
    """
    m = re.match(r'^Env\d_Location([ABC])$', scene)
    if m:
        return [m.group(1)] * ntrials
    m = re.match(r'^Env\d_Location([ABC])_to_([ABC])$', scene)
    if m is None:
        m = re.match(r'^Env\d_([ABC])_to_Env\d_([ABC])$', scene)
    if m:
        return [m.group(1)] * change_trial + [m.group(2)] * (ntrials - change_trial)
    raise ValueError('unrecognized scene name: %s' % scene)


def load_fluorescence(f):
    """Return (F, Fneu, iscell, planeIdx) with ROIs in PlaneSegmentation order.

    Multi-plane mice (m17, m18) have one RoiResponseSeries per plane; planes are
    pooled, as in the paper (ROIs identified per plane, pooled for all analyses).
    """
    ophys = f['processing/ophys']
    seg = ophys['ImageSegmentation/PlaneSegmentation']
    nroi = seg['id'].shape[0]
    nframes = ophys['Fluorescence/plane0/data'].shape[0]
    F = np.empty((nroi, nframes), dtype=np.float32)
    Fneu = np.empty((nroi, nframes), dtype=np.float32)
    for plane in sorted(ophys['Fluorescence'].keys()):
        rois = ophys['Fluorescence'][plane]['rois'][:]
        F[rois] = ophys['Fluorescence'][plane]['data'][:].T
        Fneu[rois] = ophys['Neuropil'][plane]['data'][:].T
    iscell = seg['iscell'][:, 0] > 0
    plane_idx = seg['planeIdx'][:]
    return F, Fneu, iscell, plane_idx


def compute_activity(F, Fneu, starts, stops, frame_rate):
    """dF/F and deconvolved events, computed within each trial (paper pipeline)."""
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]
    valid = ~np.isnan(f_[0])

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        seg = nansmooth(f_[:, s:e], [0, BASELINE_SMOOTH])
        win = min(BASELINE_WIN, max(1, e - s))
        seg = minimum_filter1d(seg, win, axis=-1)
        seg = maximum_filter1d(seg, win, axis=-1)
        # add back the trial's mean neuropil so dF/F is not divided by tiny baselines
        offset = NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        f_[:, s:e] = f_[:, s:e] + offset
        flow[:, s:e] = seg + offset

    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])

    import suite2p.extraction.dcnv as dcnv
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        dff[:, s:e] = nansmooth(dff[:, s:e], [0, DFF_SMOOTH])
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]),
                                    2000, TAU, frame_rate)
    return dff, events, valid


def discretize_reward_distance(d):
    """Signed distance (cm) to the nearest point of the reward zone -> 7 bins."""
    out = np.full(d.shape, 6, dtype=np.int64)
    out[d > 50] = 6
    out[(d > 10) & (d <= 50)] = 5
    out[(d > 0) & (d <= 10)] = 4
    out[d == 0] = 3
    out[(d >= -10) & (d < 0)] = 2
    out[(d >= -50) & (d < -10)] = 1
    out[d < -50] = 0
    return out


def process_session(path, signal='dff'):
    f = h5py.File(path, 'r')
    ident = f['identifier'][()].decode()
    scene = ident.split('/')[-1]
    subject = f['general/subject/subject_id'][()].decode()
    exp_day = int(f['general/session_id'][()])
    date = ident.split('/')[-2]

    b = f['processing/behavior/BehavioralTimeSeries']
    get = lambda k: b[k + '/data'][:]
    tstamps = b['position/timestamps'][:]
    pos = get('position')
    speed = get('speed')
    lick = get('lick')
    morph = get('environment')
    trialnum = get('trial number')
    starts = np.where(get('trial_start') > 0)[0]
    stops = np.where(get('teleport') > 0)[0]
    reward_frames = np.searchsorted(tstamps, b['Reward/timestamps'][:])
    rzone_entry = get('reward_zone')
    ntrials = len(starts)
    assert len(stops) == ntrials

    frame_rate = 1.0 / np.median(np.diff(tstamps))

    F, Fneu, iscell, plane_idx = load_fluorescence(f)
    F = F[iscell]
    Fneu = Fneu[iscell]
    plane_idx = plane_idx[iscell]
    f.close()

    # A few two-plane sessions have one more imaging frame than behavior samples
    # (the last frame of the interleaved scan); truncate to the common length so
    # that neural and behavioral streams stay sample-aligned.
    nsamp = min(F.shape[1], len(pos))
    if F.shape[1] != len(pos):
        F = F[:, :nsamp]
        Fneu = Fneu[:, :nsamp]
        pos = pos[:nsamp]; speed = speed[:nsamp]; lick = lick[:nsamp]
        morph = morph[:nsamp]; trialnum = trialnum[:nsamp]
        rzone_entry = rzone_entry[:nsamp]; tstamps = tstamps[:nsamp]
        keep = stops <= nsamp
        starts = starts[keep]; stops = stops[keep]
        ntrials = len(starts)

    dff, events, valid = compute_activity(F, Fneu, starts, stops, frame_rate)
    del F, Fneu

    # exclude putative interneurons: dF/F correlated with running speed (r > 0.5)
    spd_valid = speed[valid]
    dff_valid = dff[:, valid]
    dv = dff_valid - dff_valid.mean(axis=1, keepdims=True)
    sv = spd_valid - spd_valid.mean()
    denom = np.sqrt((dv ** 2).sum(axis=1) * (sv ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_corr = (dv @ sv) / denom
    is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
    keep_cells = ~is_int
    del dff_valid, dv

    activity = events if signal == 'events' else dff
    activity = activity[keep_cells]
    plane_idx = plane_idx[keep_cells]

    # ---- per-trial task variables
    zone_labels = scene_reward_zones(scene, ntrials)
    env = np.array([1.0 if np.nanmax(morph[s:e]) > 0.5 else 0.0
                    for s, e in zip(starts, stops)])
    rewarded = np.zeros(ntrials, dtype=np.int64)
    lick_error = np.zeros(ntrials, dtype=bool)
    for i, (s, e) in enumerate(zip(starts, stops)):
        got_reward = np.any((reward_frames >= s) & (reward_frames < e))
        in_zone = np.any(rzone_entry[s:e] > 0)
        rewarded[i] = int(bool(got_reward) and bool(in_zone))
        seg = lick[s:e]
        lick_error[i] = (np.sum(seg > LICK_ERR_COUNT) / len(seg)) > LICK_ERR_FRAC

    # previous trial outcome; for the first imaged trial of a session the previous
    # trial is one of the ~30 un-imaged warm-up trials run in the same condition
    # immediately before imaging, which were rewarded on ~85% of trials, so it is
    # coded as rewarded.
    prev_rewarded = np.concatenate([[1], rewarded[:-1]]).astype(np.float64)

    neural_trials, input_trials, output_trials = [], [], []
    kept_trials = []
    for i, (s, e) in enumerate(zip(starts, stops)):
        if lick_error[i]:
            continue
        act = activity[:, s:e]
        if not np.all(np.isfinite(act)):
            continue
        p = np.clip(pos[s:e], 0.0, None)
        spd = speed[s:e]
        lk = lick[s:e]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(spd))
                and np.all(np.isfinite(lk))):
            continue
        T = e - s
        t_rel = tstamps[s:e] - tstamps[s]

        zstart, zend = REWARD_ZONES[zone_labels[i]]
        dist = np.where(p < zstart, p - zstart, np.where(p > zend, p - zend, 0.0))

        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = t_rel
        inp[1] = env[i]
        inp[2] = float(trialnum[s])
        inp[3] = prev_rewarded[i]

        out = np.empty((6, T), dtype=np.int64)
        out[0] = discretize_reward_distance(dist)
        out[1] = np.clip(np.floor(p / 90.0), 0, 4).astype(np.int64)
        out[2] = np.digitize(spd, [2.0, 10.0, 20.0, 40.0])
        out[3] = (lk > 0).astype(np.int64)
        out[4] = ZONE_NAMES.index(zone_labels[i])
        out[5] = rewarded[i]

        neural_trials.append(np.ascontiguousarray(act, dtype=np.float32))
        input_trials.append(inp)
        output_trials.append(out)
        kept_trials.append(i)

    info = {
        'subject': subject,
        'exp_day': exp_day,
        'date': date,
        'scene': scene,
        'environment': 'ENV2' if env.mean() > 0.5 else 'ENV1',
        'is_switch_session': len(set(zone_labels)) > 1,
        'reward_zones': sorted(set(zone_labels)),
        'n_trials_recorded': int(ntrials),
        'n_trials_kept': len(kept_trials),
        'n_trials_dropped_lick_error': int(lick_error.sum()),
        'n_rois_curated': int(iscell.sum()),
        'n_putative_interneurons_excluded': int(is_int.sum()),
        'n_neurons': int(keep_cells.sum()),
        'frame_rate_hz': float(frame_rate),
    }
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'plane_idx': plane_idx,
        'info': info,
    }


def worker(args):
    path, signal, outdir = args
    tag = os.path.basename(path).replace('.nwb', '')
    outfn = os.path.join(outdir, tag + '.pkl')
    if os.path.exists(outfn):
        return outfn
    t0 = time.time()
    res = process_session(path, signal=signal)
    with open(outfn, 'wb') as fh:
        pickle.dump(res, fh, protocol=4)
    print('%s: %d neurons, %d trials (%.1fs)' % (tag, res['info']['n_neurons'],
                                                 len(res['neural']), time.time() - t0),
          flush=True)
    return outfn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/app/data')
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--cache-dir', default='/tmp/conv')
    ap.add_argument('--signal', default='dff', choices=['events', 'dff'])
    ap.add_argument('--limit', type=int, default=None,
                    help='only process the first N sessions (for testing)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))
    if args.limit:
        paths = paths[:args.limit]
    os.makedirs(args.cache_dir, exist_ok=True)

    jobs = [(p, args.signal, args.cache_dir) for p in paths]
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
            files = pool.map(worker, jobs, chunksize=1)
    else:
        files = [worker(j) for j in jobs]

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': [], 'subject_idx': [],
        'brain_regions': ['CA1'], 'brain_region_idx': [],
        'input_names': ['time_from_trial_start_s', 'environment',
                        'trial_number', 'previous_trial_outcome'],
        'output_names': ['reward_zone_distance', 'position', 'speed', 'lick',
                         'reward_zone_location', 'reward_outcome'],
        'output_values': [
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
             '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
            ['omitted', 'rewarded'],
        ],
        'metadata': {},
    }

    subjects = []
    session_info = []
    for fn in files:
        with open(fn, 'rb') as fh:
            res = pickle.load(fh)
        if len(res['neural']) < 2:
            print('skipping %s: fewer than 2 usable trials' % fn)
            continue
        sub = res['info']['subject']
        if sub not in subjects:
            subjects.append(sub)
        data['neural'].append(res['neural'])
        data['input'].append(res['input'])
        data['output'].append(res['output'])
        data['subject_idx'].append(subjects.index(sub))
        data['brain_region_idx'].append(np.zeros(res['neural'][0].shape[0], dtype=np.int64))
        session_info.append(res['info'])

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    bin_ms = float(np.mean([1000.0 / s['frame_rate_hz'] for s in session_info]))
    data['metadata'] = {
        'task_description': (
            'Head-fixed mice ran laps on a 450 cm virtual linear track with a hidden '
            '50 cm reward zone at one of three locations (A 80-130, B 200-250, '
            'C 320-370 cm) in one of two visually distinct environments (ENV1/ENV2). '
            'Reward was delivered operantly for licking in the zone and randomly '
            'omitted on ~15% of trials; the zone moved after 30 trials on switch days. '
            'Decoder predicts, from CA1 population activity, the distance to the '
            'reward zone, absolute track position, running speed, licking, the '
            'identity of the active reward zone and the trial reward outcome.'),
        'time_bin_size': bin_ms,
        'temporal_alignment_event': ('start of trial: teleport into the start of the '
                                     'virtual linear track (trial_start)'),
        'off_start': 0.0,
        'off_end': None,
        'trial_definition': ('trial_start (entry to the track) to teleport (entry to '
                             'the inter-trial interval); trials have variable length '
                             'because mice run at different speeds'),
        'neural_signal': ('OASIS-deconvolved calcium events from per-trial maximin dF/F '
                          'of suite2p-curated CA1 pyramidal ROIs'
                          if args.signal == 'events' else
                          'per-trial maximin dF/F of suite2p-curated CA1 pyramidal ROIs'),
        'sampling': ('native two-photon frame rate (~15.5 Hz); all VR behavior streams '
                     'in the NWB files are already synchronized to imaging frames'),
        'exclusions': ('ROIs not curated as cells by suite2p manual curation; putative '
                       'interneurons (dF/F-speed Pearson r > 0.5); trials with lick '
                       'sensor errors (>30% of frames with cumulative lick count > 2)'),
        'species': 'Mus musculus',
        'source': ('Sosa, Plitt & Giocomo 2025, Nature Neuroscience; DANDI:001361 '
                   '(11 switch-condition mice, 14 experiment days each)'),
        'session_info': session_info,
    }

    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    ntrials = sum(len(s) for s in data['neural'])
    nneurons = sum(s[0].shape[0] for s in data['neural'])
    print('sessions: %d, trials: %d, neurons: %d' % (len(data['neural']), ntrials, nneurons))
    print('wrote %s (%.2f GB)' % (args.out, os.path.getsize(args.out) / 1e9))


if __name__ == '__main__':
    main()
