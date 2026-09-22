"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal VR dataset (DANDI 001361, NWB)
into the decoder-ready dictionary format described in the task.

Processing follows the paper's own pipeline (reward_relative / TwoPUtils code in
/app/code) wherever it applies:

  * Neurons: suite2p ROIs that passed the authors' manual curation (`iscell`),
    with additional putative interneurons removed by Pearson r > 0.5 between the
    cell's dF/F and the animal's running speed (Methods, "Calcium data processing").
    For the two multi-plane mice (m17, m18) the planes are pooled, as in the paper.
  * dF/F: re-computed exactly as in `reward_relative.preprocessing.dff`
    (neuropil subtraction with 0.7 coefficient, per-trial maximin baseline with a
    ~20 s window, division by |baseline|, 2-sample Gaussian smoothing), then
    deconvolved with OASIS (suite2p `dcnv.oasis`, tau = 0.7) to give the
    "deconvolved calcium activity" that the paper uses for decoding and GLMs.
    Baselines are computed within each trial only (the ITI/teleport period is
    excluded), as in the paper.
  * Trials: one lap of the virtual track, from the `trial_start` frame
    (inclusive) to the `teleport` frame (exclusive), i.e. the ITI/teleport period
    is not included.
  * Behaviour: position, speed, licks and reward-zone occupancy are the VR streams
    already interpolated to the imaging frame clock in the NWB files (~15.5 Hz).
  * Reward zone identity per trial comes from the session's scene name with the
    switch after trial 30 (`reward_relative.behavior.get_reward_zones`); this was
    verified against the reward-zone occupancy signal in every trial that has one.

Outputs /app/converted_data.pkl.
"""

import os
import re
import glob
import pickle
import argparse
import multiprocessing as mp
import numpy as np
import h5py

from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from suite2p.extraction import dcnv

from concurrent.futures import ProcessPoolExecutor

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

# ---------------------------------------------------------------------------
# constants taken from the paper / reference code
# ---------------------------------------------------------------------------

# reward zone [start, stop] in cm; reward_relative.behavior.reward_zone_dict
# ('X', 'Y', 'Z' there are the A, B, C zones of the switch task)
REWARD_ZONES = {'A': (80., 130.), 'B': (200., 250.), 'C': (320., 370.)}
ZONE_ORDER = ['A', 'B', 'C']

TRACK_LENGTH = 450.       # cm
CHANGE_TRIAL = 30         # reward zone switches after 30 trials (Methods)

NEU_COEF = 0.7            # neuropil coefficient
BASELINE_WIN = 300        # samples (~20 s at 15.5 Hz) for the maximin baseline
BASELINE_SMOOTH = 15      # samples, pre-baseline smoothing (reward_relative.preprocessing.dff)
DFF_SMOOTH = 2            # samples s.d. Gaussian on dF/F (~0.129 s)
TAU = 0.7                 # calcium kernel decay for OASIS deconvolution
INTERNEURON_R = 0.5       # speed/dF/F correlation above which a cell is dropped

LICK_ERROR_FRAC = 0.3     # >30% of samples in a trial with cumulative lick count > 2
LICK_ERROR_COUNT = 2      # => stuck lick sensor, trial discarded


# ---------------------------------------------------------------------------
# session metadata from the scene name
# ---------------------------------------------------------------------------

def parse_scene(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label and environment (0 = ENV 1, 1 = ENV 2).

    Mirrors reward_relative.behavior.get_reward_zones / get_trial_types: on switch
    sessions the reward zone (and, on day 8, the environment) changes after
    `change_trial` trials.
    """
    m = re.fullmatch(r'Env(\d)_Location([ABC])', scene)
    if m:
        env = int(m.group(1)) - 1
        return [m.group(2)] * ntrials, [env] * ntrials
    m = re.fullmatch(r'Env(\d)_Location([ABC])_to_([ABC])', scene)
    if m:
        env = int(m.group(1)) - 1
        zones = [m.group(2)] * change_trial + [m.group(3)] * (ntrials - change_trial)
        return zones[:ntrials], [env] * ntrials
    m = re.fullmatch(r'Env(\d)_([ABC])_to_Env(\d)_([ABC])', scene)
    if m:
        zones = [m.group(2)] * change_trial + [m.group(4)] * (ntrials - change_trial)
        envs = ([int(m.group(1)) - 1] * change_trial
                + [int(m.group(3)) - 1] * (ntrials - change_trial))
        return zones[:ntrials], envs[:ntrials]
    raise ValueError(f'unrecognized scene name: {scene}')


# ---------------------------------------------------------------------------
# neural processing
# ---------------------------------------------------------------------------

def compute_dff_events(F, Fneu, starts, stops, fs):
    """dF/F and deconvolved activity, following reward_relative.preprocessing.dff.

    Everything outside of a trial (the ITI / teleport period) is left out of the
    computation, so baselines are per-trial and are not contaminated by the
    periods in which the laser power was blanked.

    Args:
        F, Fneu: (ncells, nframes) raw suite2p ROI and neuropil fluorescence.
        starts, stops: trial start / teleport frame indices.
        fs: imaging rate per plane (Hz).

    Returns:
        dff, events: (ncells, nframes) float arrays, NaN/0 outside of trials.
    """
    f = np.full(F.shape, np.nan, dtype=np.float64)
    fneu = np.full(F.shape, np.nan, dtype=np.float64)
    for s, e in zip(starts, stops):
        f[:, s:e] = F[:, s:e]
        fneu[:, s:e] = Fneu[:, s:e]

    # neuropil subtraction
    f -= NEU_COEF * fneu

    dff = np.zeros(F.shape, dtype=np.float64)
    events = np.zeros(F.shape, dtype=np.float32)
    for s, e in zip(starts, stops):
        # add the trial's mean neuropil back in so that dF/F is not divided by a
        # near-zero baseline (as in the reference implementation)
        trial = f[:, s:e] + NEU_COEF * np.nanmean(fneu[:, s:e], axis=1, keepdims=True)
        # maximin baseline: smooth, minimum filter, then maximum filter
        base = gaussian_filter1d(trial, BASELINE_SMOOTH, axis=1)
        base = minimum_filter1d(base, BASELINE_WIN, axis=-1)
        base = maximum_filter1d(base, BASELINE_WIN, axis=-1)
        d = (trial - base) / np.abs(base)
        d = gaussian_filter1d(d, DFF_SMOOTH, axis=1)
        dff[:, s:e] = d
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(d, dtype=np.float32),
                                    2000, TAU, fs)
    return dff, events


# ---------------------------------------------------------------------------
# discretization of the decoder outputs
# ---------------------------------------------------------------------------

def bin_reward_distance(pos, zone):
    """Signed distance to the nearest point of the reward zone, discretized."""
    start, stop = REWARD_ZONES[zone]
    d = np.zeros_like(pos)
    d[pos < start] = pos[pos < start] - start
    d[pos > stop] = pos[pos > stop] - stop
    out = np.full(pos.shape, 3, dtype=np.int64)     # 3: inside the zone (d == 0)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out


def bin_position(pos):
    """Track position into 5 equal bins of 90 cm spanning the 450 cm track."""
    return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)


def bin_speed(speed):
    return np.digitize(speed, [2., 10., 20., 40.]).astype(np.int64)


# ---------------------------------------------------------------------------
# per-session conversion
# ---------------------------------------------------------------------------

def convert_session(path):
    with h5py.File(path, 'r') as f:
        scene = f['identifier'][()].decode().split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        day = int(f['general/session_id'][()].decode())
        region = f['general/optophysiology/ImagingPlane/location'][()].decode()

        beh = f['processing/behavior/BehavioralTimeSeries']
        pos = beh['position/data'][:]
        speed = beh['speed/data'][:]
        lick = beh['lick/data'][:]
        rzone = beh['reward_zone/data'][:]
        tstart = beh['trial_start/data'][:]
        teleport = beh['teleport/data'][:]
        stamps = beh['position/timestamps'][:]
        reward_times = beh['Reward/timestamps'][:]

        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = seg['iscell'][:, 0].astype(bool)

        planes = sorted(f['processing/ophys/Fluorescence'].keys())
        Fs, Fneus, roi_ids = [], [], []
        for p in planes:
            Fs.append(f[f'processing/ophys/Fluorescence/{p}/data'][:].T)
            Fneus.append(f[f'processing/ophys/Neuropil/{p}/data'][:].T)
            roi_ids.append(f[f'processing/ophys/Fluorescence/{p}/rois'][:])
        # per-plane frame rate: the NWB `rate` attribute is the scan rate over all
        # planes, so divide by the number of simultaneously imaged planes
        fs = float(f[f'processing/ophys/Fluorescence/{planes[0]}/starting_time']
                   .attrs['rate']) / len(planes)

    F = np.concatenate(Fs, axis=0)
    Fneu = np.concatenate(Fneus, axis=0)
    roi_ids = np.concatenate(roi_ids)

    # behaviour and imaging occasionally differ by one frame; use the overlap
    nframes = min(F.shape[1], len(pos))
    F, Fneu = F[:, :nframes], Fneu[:, :nframes]

    # curated cells only (suite2p manual curation by the authors)
    keep = iscell[roi_ids]
    F, Fneu = F[keep], Fneu[keep]

    starts = np.where(tstart == 1)[0]
    stops = np.where(teleport == 1)[0]
    assert len(starts) == len(stops) and np.all(stops > starts)
    stops = np.minimum(stops, nframes)
    ntrials = len(starts)

    dff, events = compute_dff_events(F, Fneu, starts, stops, fs)

    # drop putative interneurons: dF/F correlated with running speed
    inside = np.zeros(nframes, dtype=bool)
    for s, e in zip(starts, stops):
        inside[s:e] = True
    d = dff[:, inside]
    sp = speed[:nframes][inside]
    d = d - d.mean(axis=1, keepdims=True)
    spc = sp - sp.mean()
    denom = np.sqrt((d ** 2).sum(axis=1) * (spc ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (d @ spc) / denom
    good = ~(r > INTERNEURON_R)
    events = events[good]
    n_interneurons = int((~good).sum())

    zones, envs = parse_scene(scene, ntrials)

    # per-trial reward: a reward was delivered inside the (active) reward zone,
    # following reward_relative.behavior.get_trial_types
    rewarded = np.zeros(ntrials, dtype=np.int64)
    for t, (s, e) in enumerate(zip(starts, stops)):
        in_zone = np.any(rzone[s:e] > 0)
        got = np.any((reward_times >= stamps[s]) & (reward_times < stamps[e]))
        rewarded[t] = int(in_zone and got)

    neural, inputs, outputs, kept_trials = [], [], [], []
    n_lick_error = 0
    for t, (s, e) in enumerate(zip(starts, stops)):
        licks = lick[s:e]
        # trials with a stuck lick sensor (Methods, "Quantification of licking
        # behavior") -- licking is a decoder output, so the trial is discarded
        if np.mean(licks > LICK_ERROR_COUNT) > LICK_ERROR_FRAC:
            n_lick_error += 1
            continue

        T = e - s
        p = pos[s:e]
        sp_t = speed[s:e]

        out = np.empty((6, T), dtype=np.int64)
        out[0] = bin_reward_distance(p, zones[t])
        out[1] = bin_position(p)
        out[2] = bin_speed(sp_t)
        out[3] = (licks > 0).astype(np.int64)
        out[4] = ZONE_ORDER.index(zones[t])
        out[5] = rewarded[t]

        # previous trial outcome; on the first imaged trial of a session the
        # preceding (warm-up) trial is not in the file, so it is set to rewarded,
        # the outcome of ~85% of trials
        prev = 1 if t == 0 else int(rewarded[t - 1])

        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = np.arange(T, dtype=np.float32) / fs
        inp[1] = envs[t]
        inp[2] = t
        inp[3] = prev

        neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))
        inputs.append(inp)
        outputs.append(out)
        kept_trials.append(t)

    info = {
        'subject': subject,
        'exp_day': day,
        'scene': scene,
        'environment': 'ENV2' if envs[0] else 'ENV1',
        'reward_zones': sorted(set(zones)),
        'switch_session': len(set(zones)) > 1,
        'n_trials': len(neural),
        'n_trials_total': ntrials,
        'n_trials_lick_error': n_lick_error,
        'n_neurons': int(events.shape[0]),
        'n_interneurons_excluded': n_interneurons,
        'n_rois_curated': int(keep.sum()),
        'n_planes': len(planes),
        'frame_rate': fs,
        'brain_region': region,
        'trial_indices': kept_trials,
    }
    return dict(neural=neural, input=inputs, output=outputs,
                subject=subject, region=region, info=info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_FILE)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=None,
                    help='only convert the first N sessions (for testing)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')),
                   key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                                  int(re.search(r'ses-(\d+)', p).group(1))))
    if args.limit:
        files = files[:args.limit]
    print(f'converting {len(files)} sessions')

    results = []
    # spawn rather than fork: suite2p's numba-threaded OASIS deconvolution is not
    # fork-safe and kills the worker
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as ex:
        for i, res in enumerate(ex.map(convert_session, files)):
            print(f"[{i + 1}/{len(files)}] {res['info']['subject']} day "
                  f"{res['info']['exp_day']} {res['info']['scene']}: "
                  f"{res['info']['n_trials']} trials, {res['info']['n_neurons']} neurons",
                  flush=True)
            results.append(res)

    subjects = sorted({r['subject'] for r in results},
                      key=lambda s: int(re.sub(r'\D', '', s)))
    regions = sorted({r['region'] for r in results})

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['subject']) for r in results],
                                dtype=np.int64),
        'brain_regions': regions,
        'brain_region_idx': [np.full(r['info']['n_neurons'],
                                     regions.index(r['region']), dtype=np.int64)
                             for r in results],
        'input_names': [
            'time from trial start (s)',
            'environment (0=ENV1, 1=ENV2)',
            'trial number',
            'previous trial rewarded',
        ],
        'output_names': [
            'distance to reward zone',
            'position in corridor',
            'speed',
            'lick',
            'reward zone location',
            'reward outcome',
        ],
        'output_values': [
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
             '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
            ['no reward', 'rewarded'],
        ],
        'metadata': {
            'task_description': (
                'Head-fixed mice run laps on a 450 cm virtual linear track for a hidden '
                '50 cm reward zone at one of three locations (A: 80-130 cm, B: 200-250 cm, '
                'C: 320-370 cm) in one of two visually distinct environments (ENV1/ENV2). '
                'Sucrose reward is delivered operantly for licking in the zone and is '
                'randomly omitted on ~15% of trials. On switch sessions the reward zone '
                '(and on day 8 also the environment) moves to a new location after 30 '
                'trials. Each lap ends in a variable-length gray teleport period, which is '
                'excluded here. The decoder predicts, from CA1 two-photon calcium activity, '
                'the distance to the reward zone, the absolute track position, running '
                'speed and licking at each imaging frame, plus the reward zone identity and '
                'whether the trial was rewarded.'),
            'time_bin_size': 1000.0 / 15.5078125,
            'temporal_alignment_event': 'trial start (entry into the virtual linear track)',
            'off_start': 0.0,
            'off_end': None,  # trials run until teleport, so their length varies
            'neural_signal': ('deconvolved calcium activity (OASIS, tau=0.7) computed from '
                              'dF/F as in Sosa et al. 2025'),
            'sampling_rate_hz': 15.5078125,
            'species': 'Mus musculus',
            'imaging': 'two-photon calcium imaging of GCaMP7f in dorsal CA1 pyramidal cells',
            'source': ('Sosa, Plitt & Giocomo, "A flexible hippocampal population code for '
                       'experience relative to reward", DANDI 001361'),
            'session_info': [r['info'] for r in results],
        },
    }

    ntrials = sum(len(r['neural']) for r in results)
    nneurons = sum(r['info']['n_neurons'] for r in results)
    print(f'{len(results)} sessions, {ntrials} trials, {nneurons} neurons, '
          f'{len(subjects)} subjects')
    print(f'writing {args.out}')
    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print('done')


if __name__ == '__main__':
    main()
