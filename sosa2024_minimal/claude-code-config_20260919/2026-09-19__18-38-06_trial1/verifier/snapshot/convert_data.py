"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal CA1 2P imaging dataset
(DANDI:001361, /app/data) into the decoder format described in the task.

Processing follows the reference paper / code (github.com/GiocomoLab/Sosa_et_al_2024,
copy in /app/code) as closely as the decoding task allows:

Neural data
-----------
* ROIs: only suite2p-curated cells (`iscell == 1` in the NWB PlaneSegmentation),
  which is the manual curation described in "Calcium data processing".
  For the two multi-plane mice (m17, m18) the planes are pooled, as in the paper.
* dF/F is recomputed exactly as in `reward_relative.preprocessing.dff`
  (`utilities.multi_anim_sess` defaults: neuropil subtraction with coefficient 0.7,
  'maximin' baseline within each trial using a 20 s (300 sample) sliding window,
  dF/F = (F - baseline)/|baseline|, smoothed with a 2-sample s.d. Gaussian).
  The NWB file's "Deconvolved" series is raw suite2p `spks` (non-zero during the
  inter-trial interval, i.e. not from the paper's within-trial dF/F), so it is NOT
  used; F and Fneu are re-processed instead.
* The dF/F is deconvolved with OASIS (suite2p `dcnv.oasis`, tau = 0.7, per-plane
  frame rate), as the paper does for all spatial/decoding analyses. These
  deconvolved "events" are the neural data handed to the decoder.
* Putative interneurons are excluded (Pearson r > 0.5 between a cell's dF/F and
  running speed over all within-trial samples; `spatial.is_putative_interneuron`
  with the `dayData` default int_thresh = 0.5).

Trials / alignment
------------------
* A trial is one lap of the 450 cm virtual track: samples
  [trial_start - 1, teleport - 1), exactly the window used by
  `preprocessing.dff` (it drops the teleport sample, whose position is
  interpolated across the jump from the end of the track to the teleport zone).
  Time bins are the imaging frames (64.484 ms, ~15.5 Hz).
* Trials are aligned to their own start: t = 0 at the first sample of the lap.
  The inter-trial ("teleport") period is excluded, both because the decoded
  variables (track position, distance to the reward zone) are undefined there and
  because the laser was blanked during the ITI on most sessions.

Trial curation
--------------
* Trials with a stuck lick sensor are dropped: >30% of imaging samples in the
  trial with a cumulative lick count > 2 ("Quantification of licking behavior";
  `behavior.correct_lick_sensor_error`). This flags 81 trials, matching the 81
  trials reported in the paper. The paper NaNs their licking; since licking is a
  decoder output here, the whole trial is dropped instead.
* The first trial of each session is dropped because "previous trial outcome" (a
  required decoder input) is undefined for it.
* Everything else is kept: all 11 mice, all 152 sessions, all remaining laps. No
  speed threshold is applied (unlike the paper's place-cell analyses) because
  speed itself is a decoder output.

Per-trial task variables
------------------------
* Reward zone: A = 80-130 cm, B = 200-250 cm, C = 320-370 cm
  (`behavior.reward_zone_dict` entries 'X', 'Y', 'Z'). The zone of each trial is
  read from the data (the positions at which the VR reward-zone flag is on) and,
  on switch sessions, the switch is placed at trial 30, the protocol value used by
  `behavior.get_reward_zones` (verified against the flagged trials in every
  session).
* Rewarded: reward delivered inside the reward zone on that trial
  (`behavior.get_trial_types`: reward AND reward-zone flag), i.e. 0 on the ~15%
  randomly omitted trials.
* Environment: the VR 'environment' (morph) value of the trial, 0 = ENV 1, 1 = ENV 2.

Usage: python convert_data.py [--out /app/converted_data.pkl] [--nsessions N]
"""

import argparse
import glob
import multiprocessing
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from suite2p.extraction import dcnv

DATA_ROOT = '/app/data'

# --- paper constants -------------------------------------------------------
NEU_COEF = 0.7          # neuropil subtraction coefficient (dff default)
TAU = 0.7               # GCaMP7f decay constant used for OASIS (suite2p ops)
BASELINE_WIN = 300      # ~20 s maximin window at 15.5 Hz
BASELINE_SIG = 15       # s.d. (samples) of the Gaussian pre-smoothing for baseline
DFF_SIG = 2             # s.d. (samples) of the Gaussian smoothing of dF/F
INT_R_THRESH = 0.5      # speed correlation above which a cell is a putative interneuron
LICK_ERR_FRAC = 0.3     # >30% of samples with cumulative lick count > 2 => sensor error
SWITCH_TRIAL = 30       # reward zone moves after 30 trials on switch sessions
REWARD_ZONES = {0: (80.0, 130.0), 1: (200.0, 250.0), 2: (320.0, 370.0)}  # A, B, C

# --- output discretisation (distance bins are handled explicitly below) ----
POS_EDGES = [90.0, 180.0, 270.0, 360.0]
SPEED_EDGES = [2.0, 10.0, 20.0, 40.0]

OUTPUT_NAMES = ['distance_to_reward_zone', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
     '0 to 10 cm', '10 to 50 cm', '> 50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
    ['omitted', 'rewarded'],
]
INPUT_NAMES = ['time_from_trial_start', 'environment', 'trial_number',
               'previous_trial_rewarded']


def zone_label(zone_start):
    """Map an observed reward-zone start position to the zone label A/B/C -> 0/1/2."""
    if zone_start < 150:
        return 0
    if zone_start < 280:
        return 1
    return 2


def discretize_distance(d):
    """Signed distance (cm) to the nearest point of the reward zone -> 7 bins."""
    out = np.empty(d.shape, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out


def compute_dff_and_events(F, Fneu, starts, stops, frame_rate):
    """Paper's dF/F (reward_relative.preprocessing.dff) + OASIS deconvolution.

    F, Fneu: (n_rois, n_frames). starts/stops: per-trial sample windows
    (already offset as in the reference code). Samples outside trials stay NaN.
    """
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    events = np.full(F.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        f = F[:, start:stop] - NEU_COEF * Fneu[:, start:stop]
        # add the trial's mean neuropil back so dF/F is not divided by a tiny baseline
        f = f + NEU_COEF * np.nanmean(Fneu[:, start:stop], axis=1, keepdims=True)
        flow = gaussian_filter1d(f, BASELINE_SIG, axis=-1)
        flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
        flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
        d = (f - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SIG, axis=-1)
        dff[:, start:stop] = d
        events[:, start:stop] = dcnv.oasis(d, 2000, TAU, frame_rate)
    return dff, events


def load_session(path):
    """Read one NWB session and return the per-trial neural/input/output arrays."""
    with h5py.File(path, 'r') as f:
        ident = f['identifier'][()].decode()
        scene = ident.split('/')[-1]
        date = ident.split('/')[-2]
        subject = f['general/subject/subject_id'][()].decode()
        day = int(f['general/session_id'][()].decode())

        beh = f['processing/behavior/BehavioralTimeSeries']
        pos = beh['position/data'][:]
        speed = beh['speed/data'][:]
        lick = beh['lick/data'][:]
        rzone = beh['reward_zone/data'][:]
        env = beh['environment/data'][:]
        tstamps = beh['position/timestamps'][:]
        reward_t = beh['Reward/timestamps'][:]
        trial_starts = np.nonzero(beh['trial_start/data'][:])[0]
        teleports = np.nonzero(beh['teleport/data'][:])[0]

        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = seg['iscell'][:, 0].astype(bool)
        plane_idx = seg['planeIdx'][:]
        planes = sorted(f['processing/ophys/Fluorescence'].keys())
        imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
        frame_rate = float(imaging_rate) / len(planes)

        # trial windows, matching preprocessing.dff: [start-1, stop-1)
        starts = trial_starts - 1
        stops = teleports - 1
        assert np.all(starts[1:] > stops[:-1]) and starts[0] >= 0

        F_list, Fneu_list, cellplane = [], [], []
        for pi, plane in enumerate(planes):
            rois = f['processing/ophys/Fluorescence'][plane]['rois'][:]
            keep = iscell[rois]
            assert np.all(plane_idx[rois] == pi)
            F_list.append(f['processing/ophys/Fluorescence'][plane]['data'][:, :].T[keep])
            Fneu_list.append(f['processing/ophys/Neuropil'][plane]['data'][:, :].T[keep])
            cellplane.append(np.full(int(keep.sum()), pi, dtype=np.int64))
        F = np.concatenate(F_list, axis=0)
        Fneu = np.concatenate(Fneu_list, axis=0)
        cellplane = np.concatenate(cellplane)
        del F_list, Fneu_list

    # a few multi-plane sessions have one imaging frame more than the VR data it
    # was aligned to (the "one frame correction" in TwoPUtils' vr_align_to_2P);
    # that trailing frame has no behaviour and falls after the last teleport
    nframes = len(pos)
    assert 0 <= F.shape[1] - nframes <= 1, (path, F.shape, nframes)
    assert stops[-1] <= nframes
    F = F[:, :nframes]
    Fneu = Fneu[:, :nframes]

    dt = float(np.median(np.diff(tstamps)))
    dff, events = compute_dff_and_events(F, Fneu, starts, stops, frame_rate)
    del F, Fneu

    # --- exclude putative interneurons (dF/F vs speed correlation > 0.5) -----
    valid = ~np.isnan(dff[0])
    dff_v = dff[:, valid]
    spd_v = speed[valid]
    dff_c = dff_v - dff_v.mean(axis=1, keepdims=True)
    spd_c = spd_v - spd_v.mean()
    denom = np.sqrt((dff_c ** 2).sum(axis=1) * (spd_c ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_corr = (dff_c @ spd_c) / denom
    is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
    del dff, dff_v, dff_c
    keep_cells = ~is_int
    events = events[keep_cells]
    cellplane = cellplane[keep_cells]

    # --- per-trial task variables -------------------------------------------
    ntrials = len(starts)
    observed_zone = np.full(ntrials, np.nan)
    rewarded = np.zeros(ntrials, dtype=np.int64)
    envs = np.zeros(ntrials, dtype=np.int64)
    lick_error = np.zeros(ntrials, dtype=bool)
    for i, (s, t) in enumerate(zip(starts, stops)):
        in_zone = rzone[s:t] > 0
        got_reward = np.any((reward_t >= tstamps[s]) & (reward_t < tstamps[t]))
        rewarded[i] = int(in_zone.any() and got_reward)
        if in_zone.any():
            observed_zone[i] = zone_label(pos[s:t][in_zone].min())
        env_vals = np.unique(env[s:t])
        assert len(env_vals) == 1, f'{path}: trial {i} spans environments {env_vals}'
        envs[i] = int(env_vals[0])
        lick_error[i] = np.mean(lick[s:t] > 2) > LICK_ERR_FRAC

    # reward zone label per trial: observed where available, switch at trial 30
    known = np.nonzero(~np.isnan(observed_zone))[0]
    labels = observed_zone[known]
    zone_of_trial = np.empty(ntrials, dtype=np.int64)
    if np.all(labels == labels[0]):
        zone_of_trial[:] = int(labels[0])
        switch_trial = None
    else:
        changes = np.nonzero(np.diff(labels))[0]
        assert len(changes) == 1, f'{path}: more than one reward zone switch'
        last_before = known[changes[0]]
        first_after = known[changes[0] + 1]
        # the protocol switches after 30 trials; fall back on the data if the
        # observed bracket does not contain it (it always does in this dataset)
        switch_trial = (SWITCH_TRIAL if last_before < SWITCH_TRIAL <= first_after
                        else first_after)
        zone_of_trial[:switch_trial] = int(labels[0])
        zone_of_trial[switch_trial:] = int(labels[-1])
    assert np.all(zone_of_trial[known] == observed_zone[known])

    # --- build per-trial arrays ---------------------------------------------
    neural, inputs, outputs = [], [], []
    trial_ids = []
    for i in range(ntrials):
        if i == 0:                      # previous trial outcome undefined
            continue
        if lick_error[i]:               # stuck lick sensor
            continue
        s, t = starts[i], stops[i]
        T = t - s
        p = pos[s:t]
        v = speed[s:t]
        lk = (lick[s:t] > 0).astype(np.int64)
        z0, z1 = REWARD_ZONES[zone_of_trial[i]]
        dist = np.where(p < z0, p - z0, np.where(p > z1, p - z1, 0.0))

        out = np.empty((6, T), dtype=np.int64)
        out[0] = discretize_distance(dist)
        out[1] = np.digitize(p, POS_EDGES)
        out[2] = np.digitize(v, SPEED_EDGES)
        out[3] = lk
        out[4] = zone_of_trial[i]
        out[5] = rewarded[i]

        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = np.arange(T, dtype=np.float32) * dt
        inp[1] = envs[i]
        inp[2] = i
        inp[3] = rewarded[i - 1]

        neural.append(np.ascontiguousarray(events[:, s:t], dtype=np.float32))
        inputs.append(inp)
        outputs.append(out)
        trial_ids.append(i)

    info = {
        'subject': subject,
        'exp_day': day,
        'scene': scene,
        'date': date,
        'environment': ['ENV1', 'ENV2'][int(envs[0])] if len(np.unique(envs)) == 1
                       else 'ENV%d_to_ENV%d' % (envs[0] + 1, envs[-1] + 1),
        'reward_zones': [['A', 'B', 'C'][z] for z in np.unique(zone_of_trial)],
        'switch_trial': switch_trial,
        'n_trials_total': int(ntrials),
        'n_trials_kept': len(neural),
        'n_trials_lick_error': int(lick_error.sum()),
        'n_rois_curated': int(len(is_int)),
        'n_putative_interneurons': int(is_int.sum()),
        'n_neurons': int(keep_cells.sum()),
        'n_planes': len(planes),
        'n_neurons_per_plane': np.bincount(cellplane, minlength=len(planes)).tolist(),
        'frame_rate_hz': frame_rate,
        'trial_indices': trial_ids,
        'rewarded_fraction': float(np.mean(rewarded)),
    }
    return dict(neural=neural, input=inputs, output=outputs, subject=subject,
                nneurons=int(keep_cells.sum()), dt=dt, info=info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--nsessions', type=int, default=None,
                    help='only convert the first N sessions (for testing)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')),
                   key=lambda p: (int(os.path.basename(p).split('_')[0][5:]),
                                  os.path.basename(p)))
    if args.nsessions is not None:
        files = files[:args.nsessions]
    print(f'converting {len(files)} sessions', flush=True)

    t0 = time.time()
    results = []
    # spawn: the OASIS deconvolution is numba-jitted and does not survive a fork
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
        for i, res in enumerate(pool.map(load_session, files)):
            results.append(res)
            print(f'[{i + 1}/{len(files)}] {res["info"]["subject"]} '
                  f'day {res["info"]["exp_day"]}: {res["nneurons"]} neurons, '
                  f'{len(res["neural"])} trials ({time.time() - t0:.0f} s)', flush=True)

    subjects = sorted({r['subject'] for r in results},
                      key=lambda s: int(s[1:]))
    dts = {round(r['dt'], 6) for r in results}
    assert len(dts) == 1, f'inconsistent time bins: {dts}'

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['subject']) for r in results],
                                dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': [np.zeros(r['nneurons'], dtype=np.int64) for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice run laps on a 450 cm virtual linear track with a '
                'hidden 50 cm reward zone at one of three locations (A: 80-130 cm, '
                'B: 200-250 cm, C: 320-370 cm) in one of two visually distinct '
                'environments (ENV1/ENV2). Sucrose reward is delivered operantly for '
                'licking in the zone and randomly omitted on ~15% of trials. On switch '
                'sessions the zone moves to a new location after 30 trials. Decoded '
                'outputs are the distance from the animal to the reward zone, the '
                'absolute track position, running speed and licking (all time-varying), '
                'plus the reward zone identity and whether the trial was rewarded '
                '(per trial). Neural data are OASIS-deconvolved dF/F events from '
                'two-photon imaging of CA1 pyramidal neurons.'),
            'time_bin_size': float(round(list(dts)[0] * 1000, 4)),
            'temporal_alignment_event': (
                'start of the trial (lap): the imaging frame at which the animal '
                'enters the virtual track at position 0 cm'),
            'off_start': 0.0,
            'off_end': None,
            'trial_definition': (
                'one lap, from the trial-start frame to the frame before teleport '
                '(the inter-trial teleport period is excluded); trial durations vary, '
                'so the number of time bins differs across trials'),
            'neural_signal': ('deconvolved calcium events (OASIS, tau=0.7) from dF/F '
                              'computed per trial with a 20 s maximin baseline after '
                              'neuropil subtraction (0.7), as in Sosa et al. 2025'),
            'recording_modality': 'two-photon calcium imaging (GCaMP7f)',
            'exclusions': (
                'non-curated suite2p ROIs; putative interneurons (dF/F-speed r > 0.5); '
                'trials with a stuck lick sensor (>30% of samples with cumulative lick '
                'count > 2); the first trial of each session (no previous outcome)'),
            'paper': ('Sosa, Plitt & Giocomo (2025), A flexible hippocampal population '
                      'code for experience relative to reward, Nature Neuroscience; '
                      'DANDI:001361'),
            'session_info': [r['info'] for r in results],
        },
    }

    ntrials = sum(len(r['neural']) for r in results)
    nneurons = sum(r['nneurons'] for r in results)
    nsamples = sum(x.shape[1] for r in results for x in r['neural'])
    print(f'{len(results)} sessions, {ntrials} trials, {nneurons} neurons, '
          f'{nsamples} time bins', flush=True)

    with open(args.out, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB) in '
          f'{time.time() - t0:.0f} s')


if __name__ == '__main__':
    sys.exit(main())
