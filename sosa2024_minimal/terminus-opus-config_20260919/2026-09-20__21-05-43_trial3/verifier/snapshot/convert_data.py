"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal VR dataset (DANDI 001361)
into the decoder dictionary format.

Processing follows the paper / the authors' `reward_relative` code:
  * ROIs: suite2p `iscell` curated ROIs only; planes pooled for the two
    dual-plane mice (m17, m18), as in the paper.
  * dF/F: neuropil subtraction (0.7 * Fneu, trial-mean neuropil added back),
    baseline per trial with a maximin procedure (Gaussian sd 15 frames, then
    20 s minimum + maximum filter = 300 frames), dF/F = (F-F0)/|F0|,
    smoothed with a 2-frame s.d. Gaussian (preprocessing.dff).
  * Deconvolution: OASIS (suite2p dcnv.oasis, tau = 0.7, fs = frame rate per
    plane) run per trial, as in the paper ('events').  The deconvolved trace
    is what the paper uses for its own decoding analyses, so it is used as the
    neural input here.
  * Putative interneurons (Pearson r > 0.5 between dF/F and running speed) are
    excluded (spatial.is_putative_interneuron / dayData defaults).
  * Trials are laps of the 450 cm track, from the `trial_start` frame to the
    `teleport` frame (the teleport/ITI period is excluded, as in the paper,
    where dF/F is only computed on the track).
  * Trials with lick-sensor errors (>30% of frames with a cumulative lick
    count > 2) are dropped, as in the paper's licking analyses.
  * Reward zone per trial comes from the session scene name (A: 80-130 cm,
    B: 200-250 cm, C: 320-370 cm), switching after trial 30 on switch
    sessions; verified against the reward-zone entry flag in the data.
"""

import os
import glob
import pickle
import numpy as np
import scipy.ndimage as ndi
import h5py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from suite2p.extraction import dcnv

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

B = 'processing/behavior/BehavioralTimeSeries/'

# reward zone coordinates (cm), from reward_relative.behavior.reward_zone_dict
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_IDX = {'A': 0, 'B': 1, 'C': 2}

SWITCH_TRIAL = 30       # reward zone switches after 30 trials (Methods)
NEU_COEF = 0.7          # neuropil coefficient
TAU = 0.7               # GCaMP7f decay constant used in the paper's suite2p ops
BASELINE_SMOOTH = 15    # frames, s.d. of Gaussian before maximin filtering
BASELINE_WIN = 300      # frames (~20 s at 15.5 Hz) maximin window
DFF_SMOOTH = 2          # frames, s.d. of Gaussian smoothing of dF/F
INT_R_THRESH = 0.5      # speed-correlation threshold for putative interneurons
LICK_ERROR_FRAC = 0.3   # fraction of frames with lick count > 2 -> sensor error
TRACK_LENGTH = 450.0


def zone_labels_for_session(scene, ntrials):
    """Reward zone label for each trial, from the scene/session name."""
    if '_to_' in scene:
        pre, post = scene.split('_to_')
        z0, z1 = pre[-1], post[-1]
        labels = [z0] * min(SWITCH_TRIAL, ntrials) + [z1] * max(0, ntrials - SWITCH_TRIAL)
    else:
        labels = [scene[-1]] * ntrials
    return labels


def signed_distance_to_zone(pos, zone):
    """Signed distance (cm) to the nearest point of the reward zone.
    Negative before the zone, 0 inside it, positive past it."""
    start, stop = zone
    d = np.zeros_like(pos)
    d[pos < start] = pos[pos < start] - start
    d[pos > stop] = pos[pos > stop] - stop
    return d


def bin_reward_distance(d):
    # 0: < -50 | 1: -50..-10 | 2: -10..<0 | 3: 0 | 4: >0..+10 | 5: +10..+50 | 6: >+50
    out = np.zeros(d.shape, dtype=np.int16)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out


def bin_position(pos):
    # 5 equal bins over the 450 cm track
    edges = np.array([90.0, 180.0, 270.0, 360.0])
    return np.digitize(pos, edges).astype(np.int16)


def bin_speed(speed):
    # 0: <2 | 1: 2-10 | 2: 10-20 | 3: 20-40 | 4: >40 cm/s
    edges = np.array([2.0, 10.0, 20.0, 40.0])
    return np.digitize(speed, edges).astype(np.int16)


def process_session(fn):
    with h5py.File(fn, 'r') as f:
        scene = f['identifier'][()].decode().split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        planes = sorted(f['processing/ophys/Fluorescence'].keys())
        nplanes = len(planes)
        rate = float(f['processing/ophys/Fluorescence/' + planes[0] +
                       '/starting_time'].attrs['rate'])
        fs = rate / nplanes   # sampling rate per plane (~15.5 Hz)

        F = np.concatenate([f['processing/ophys/Fluorescence/' + p + '/data'][:].T
                            for p in planes], axis=0)
        Fneu = np.concatenate([f['processing/ophys/Neuropil/' + p + '/data'][:].T
                               for p in planes], axis=0)
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] > 0

        pos = f[B + 'position/data'][:]
        speed = f[B + 'speed/data'][:]
        lick = f[B + 'lick/data'][:]
        env = f[B + 'environment/data'][:]
        rzone_flag = f[B + 'reward_zone/data'][:]
        tstart = np.where(f[B + 'trial_start/data'][:] > 0)[0]
        teleport = np.where(f[B + 'teleport/data'][:] > 0)[0]
        ts = f[B + 'position/timestamps'][:]
        reward_ts = f[B + 'Reward/timestamps'][:]

    # suite2p curated ROIs only
    F = F[iscell].astype(np.float32)
    Fneu = Fneu[iscell].astype(np.float32)

    # a few dual-plane sessions have one more imaging frame than VR samples;
    # truncate to the samples that have both imaging and behaviour
    nframes = min(F.shape[1], len(pos))
    F = F[:, :nframes]
    Fneu = Fneu[:, :nframes]
    pos = pos[:nframes]; speed = speed[:nframes]; lick = lick[:nframes]
    env = env[:nframes]; rzone_flag = rzone_flag[:nframes]; ts = ts[:nframes]
    npairs = min(len(tstart), len(teleport))
    tstart, teleport = tstart[:npairs], teleport[:npairs]
    keep_tr = teleport < nframes
    tstart, teleport = tstart[keep_tr], teleport[keep_tr]

    ntrials = min(len(tstart), len(teleport))
    tstart = tstart[:ntrials]
    teleport = teleport[:ntrials]

    # ---- dF/F and deconvolution, computed within each trial ----
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(tstart, teleport):
        fseg = F[:, s:e] - NEU_COEF * Fneu[:, s:e]
        # add the trial-mean neuropil back so dF/F is not divided by small numbers
        fseg = fseg + NEU_COEF * np.mean(Fneu[:, s:e], axis=1, keepdims=True)
        flow = ndi.gaussian_filter1d(fseg, BASELINE_SMOOTH, axis=1)
        flow = ndi.minimum_filter1d(flow, BASELINE_WIN, axis=-1)
        flow = ndi.maximum_filter1d(flow, BASELINE_WIN, axis=-1)
        d = (fseg - flow) / np.abs(flow)
        d = ndi.gaussian_filter1d(d, DFF_SMOOTH, axis=1).astype(np.float32)
        dff[:, s:e] = d
        events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)

    # ---- exclude putative interneurons (dF/F correlated with running speed) ----
    on_track = ~np.isnan(dff[0, :])
    sp = speed[on_track]
    dsub = dff[:, on_track]
    dsub = dsub - dsub.mean(axis=1, keepdims=True)
    spc = sp - sp.mean()
    denom = (np.sqrt((dsub ** 2).sum(axis=1)) * np.sqrt((spc ** 2).sum()))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (dsub @ spc) / denom
    is_int = np.nan_to_num(r, nan=0.0) > INT_R_THRESH
    keep = ~is_int
    events = events[keep]

    # ---- per-trial behaviour ----
    zone_lab = zone_labels_for_session(scene, ntrials)

    rewarded = np.zeros(ntrials, dtype=np.int16)
    lick_error = np.zeros(ntrials, dtype=bool)
    env_trial = np.zeros(ntrials, dtype=np.int16)
    for i, (s, e) in enumerate(zip(tstart, teleport)):
        # rewarded == reward delivered while in the reward zone (behavior.get_trial_types)
        got_reward = np.any((reward_ts >= ts[s]) & (reward_ts <= ts[e]))
        in_zone = np.any(rzone_flag[s:e] > 0)
        rewarded[i] = int(got_reward and in_zone)
        seg = lick[s:e]
        lick_error[i] = (np.sum(seg > 2) / len(seg)) > LICK_ERROR_FRAC
        env_trial[i] = int(np.round(np.median(env[s:e])))

    neural_trials, input_trials, output_trials = [], [], []
    trials_kept = []
    # first trial of each session is dropped: the previous trial's outcome
    # (a decoder input) is unknown for it
    for i in range(1, ntrials):
        if lick_error[i]:
            continue
        if env_trial[i] not in (0, 1):
            continue
        s, e = tstart[i], teleport[i]
        T = e - s
        if T < 2:
            continue
        ev = events[:, s:e]
        if not np.all(np.isfinite(ev)):
            continue

        p = pos[s:e]
        sp_t = speed[s:e]
        lk = (lick[s:e] > 0).astype(np.int16)
        zone = REWARD_ZONES[zone_lab[i]]
        dist = signed_distance_to_zone(p, zone)

        t_in_trial = np.arange(T, dtype=np.float32) / fs
        inp = np.stack([
            t_in_trial,
            np.full(T, env_trial[i], dtype=np.float32),
            np.full(T, i, dtype=np.float32),
            np.full(T, rewarded[i - 1], dtype=np.float32),
        ], axis=0)

        outp = np.stack([
            bin_reward_distance(dist),
            bin_position(p),
            bin_speed(sp_t),
            lk,
            np.full(T, ZONE_IDX[zone_lab[i]], dtype=np.int16),
            np.full(T, rewarded[i], dtype=np.int16),
        ], axis=0).astype(np.int16)

        neural_trials.append(ev.astype(np.float32))
        input_trials.append(inp)
        output_trials.append(outp)
        trials_kept.append(i)

    info = {
        'subject': subject,
        'session_id': session_id,
        'scene': scene,
        'n_planes': nplanes,
        'fs': fs,
        'n_rois_total': int(iscell.size),
        'n_iscell': int(iscell.sum()),
        'n_interneurons_excluded': int(is_int.sum()),
        'n_neurons': int(keep.sum()),
        'n_trials_total': int(ntrials),
        'n_trials_kept': len(neural_trials),
        'n_lick_error_trials': int(lick_error.sum()),
        'file': os.path.basename(fn),
    }
    return neural_trials, input_trials, output_trials, info


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    print(f'{len(files)} sessions found')

    results = []
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=12, mp_context=ctx) as ex:
        for fn, res in zip(files, ex.map(process_session, files)):
            results.append(res)
            info = res[3]
            print(f"{info['file']}: {info['n_neurons']} neurons "
                  f"({info['n_iscell']} iscell, {info['n_interneurons_excluded']} int excluded), "
                  f"{info['n_trials_kept']}/{info['n_trials_total']} trials", flush=True)

    subjects = sorted({r[3]['subject'] for r in results},
                      key=lambda s: int(s[1:]))
    data = {
        'neural': [r[0] for r in results],
        'input': [r[1] for r in results],
        'output': [r[2] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r[3]['subject']) for r in results]),
        'brain_regions': ['CA1'],
        'brain_region_idx': [np.zeros(r[3]['n_neurons'], dtype=int) for r in results],
        'input_names': ['time_from_trial_start', 'environment', 'trial_number',
                        'previous_trial_outcome'],
        'output_names': ['reward_zone_distance', 'position', 'speed', 'lick',
                         'reward_zone_location', 'reward_outcome'],
        'output_values': [
            ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm (in zone)',
             '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
            ['omitted', 'rewarded'],
        ],
        'metadata': {
            'task_description': (
                'Head-fixed mice run laps on a 450 cm virtual linear track with a hidden '
                '50 cm reward zone (A: 80-130, B: 200-250, C: 320-370 cm) in one of two '
                'visual environments (ENV1/ENV2). Sucrose reward is delivered operantly '
                'for licking in the zone and randomly omitted on ~15% of trials. On switch '
                'sessions the zone moves to a new location after trial 30. Decoder predicts '
                'distance to the reward zone, track position, running speed, licking, the '
                'active reward zone and the trial reward outcome from CA1 two-photon '
                'calcium activity.'),
            'time_bin_size': 1000.0 / (15.5078125),
            'temporal_alignment_event': 'trial start (entry into the linear track at 0 cm)',
            'off_start': 0.0,
            'off_end': None,
            'neural_signal': ('deconvolved calcium events (OASIS, tau=0.7) computed from '
                              'per-trial maximin-baselined, neuropil-subtracted dF/F, as in '
                              'the paper'),
            'trial_definition': ('each trial is one lap from the trial_start frame to the '
                                 'teleport frame; the inter-trial teleport period is excluded'),
            'session_info': [r[3] for r in results],
            'dataset': ('Sosa, Plitt & Giocomo 2025, Nat Neurosci; DANDI 001361 '
                        '(two-photon CA1 imaging, GCaMP7f)'),
            'exclusions': ('non-cell ROIs (suite2p iscell), putative interneurons '
                           '(dF/F-speed r > 0.5), trials with lick sensor errors '
                           '(>30% of frames with cumulative lick count > 2), and the first '
                           'trial of each session (previous trial outcome undefined)'),
        },
    }

    ntrials = sum(len(x) for x in data['neural'])
    nneurons = sum(x[3]['n_neurons'] for x in results)
    print(f'total sessions: {len(results)}, trials: {ntrials}, neurons: {nneurons}')

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('saved', OUT_FILE)


if __name__ == '__main__':
    main()
