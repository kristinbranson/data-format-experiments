"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal CA1 2p imaging + VR dataset
(DANDI 001361, NWB files in /app/data) into the decoder data format.

Processing follows the reference paper/code (GiocomoLab/Sosa_et_al_2024):
  * Trials run from trial_start to teleport (the VR linear track lap); the
    inter-trial 'teleport' period is excluded, as in preprocessing.dff
    (keep_teleports=False, the default used for the paper).
  * Neural activity = deconvolved calcium 'events' computed exactly as in
    reward_relative.preprocessing.dff: neuropil subtraction (0.7), per-trial
    maximin baseline (smooth sigma=15 samples, 300-sample min then max filter
    ~ 20 s), dF/F = (F-F0)/|F0|, 2-sample Gaussian smoothing, then OASIS
    deconvolution (tau=0.7, per-plane frame rate ~15.5 Hz).  This 'events'
    timeseries is what the paper used for decoding (glmUtils.get_timeseries_data).
  * Cells: suite2p iscell==1 (manually curated in the paper), minus putative
    interneurons (Pearson r between dF/F and running speed > 0.5, as in
    dayData int_thresh=0.5 / spatial.is_putative_interneuron).
  * Reward zones: A=[80,130], B=[200,250], C=[320,370] cm (behavior.reward_zone_dict),
    assigned from the scene name with the switch at trial index 30
    (behavior.get_reward_zones change_trial=30).
  * Trials with lick-sensor errors (>30% of imaging samples in the trial with a
    cumulative lick count > 2) are dropped, as those lick data are unusable
    (the paper sets them to NaN; 81/12216 trials here, matching the paper).
  * The first trial of each session is dropped because the decoder input
    'previous trial outcome' is undefined for it.

Deliberate deviations from the paper's own decoding analysis, required by the
decoder task specification here:
  * The paper's decoder (glmUtils.get_timeseries_data, use_speed_thr=2) discards
    samples with running speed < 2 cm/s.  Running speed is a decoder output here
    (with a dedicated '< 2 cm/s' bin) and trials must stay contiguous in time, so
    all in-trial samples are kept.
  * The paper decoded reward-relative position in circular (radian) coordinates;
    here the task specifies signed linear distance (cm) to the reward zone,
    discretised into the seven bins listed in the task description.
"""

import os
import sys
import glob
import pickle
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from suite2p.extraction import dcnv
import multiprocessing as mp

DATA_DIR = '/app/data'
OUT_DIR = '/app/sessions_out'
OUT_FILE = '/app/converted_data.pkl'

ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_IDX = {'A': 0, 'B': 1, 'C': 2}
CHANGE_TRIAL = 30          # reward zone switches on trial 30 (0-indexed)
NEU_COEF = 0.7             # neuropil coefficient
TAU = 0.7                  # GCaMP7f decay constant used in the paper
BASELINE_SMOOTH = 15       # samples, baseline pre-smoothing (paper: nansmooth [0,15])
BASELINE_WIN = 300         # samples ~ 20 s maximin window
DFF_SMOOTH = 2             # samples, dF/F smoothing sigma
INT_R_THRESH = 0.5         # speed-correlation threshold for putative interneurons
LICK_ERR_THRESH = 0.3      # fraction of samples with cumulative lick count > 2
TRACK_LENGTH = 450.0


def scene_zones(scene):
    """Reward zone label(s) for a scene name, e.g. Env1_LocationB_to_A or Env1_C_to_Env2_A."""
    if '_to_' in scene:
        left, right = scene.split('_to_')
        return left[-1], right[-1]
    return scene[-1], scene[-1]


def compute_events(F, Fneu, starts, stops, fs):
    """Per-trial dF/F and OASIS-deconvolved events, following preprocessing.dff."""
    n_cells, n_frames = F.shape
    dff = np.full((n_cells, n_frames), np.nan, dtype=np.float32)
    events = np.full((n_cells, n_frames), np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f = F[:, s:e].astype(np.float64)
        fneu = Fneu[:, s:e].astype(np.float64)
        # neuropil subtraction, then add back the trial mean neuropil so that
        # dF/F stays close to true dF/F (as in preprocessing.dff)
        f = f - NEU_COEF * fneu + NEU_COEF * np.mean(fneu, axis=1, keepdims=True)
        # maximin baseline
        flow = gaussian_filter1d(f, BASELINE_SMOOTH, axis=1)
        flow = minimum_filter1d(flow, BASELINE_WIN, axis=-1)
        flow = maximum_filter1d(flow, BASELINE_WIN, axis=-1)
        d = (f - flow) / np.abs(flow)
        d = gaussian_filter1d(d, DFF_SMOOTH, axis=1)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        dff[:, s:e] = d
        events[:, s:e] = dcnv.oasis(d, 2000, TAU, fs)
    return dff, events


def process_session(fn):
    with h5py.File(fn, 'r') as f:
        ident = f['identifier'][()].decode()
        scene = ident.split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        b = f['processing/behavior/BehavioralTimeSeries']
        get = lambda k: b[k + '/data'][()]
        pos = get('position')
        speed = get('speed')
        lick = get('lick')
        env = get('environment')
        rzone_ts = get('reward_zone')
        trial_start = get('trial_start')
        teleport = get('teleport')
        tstamps = b['position/timestamps'][()]
        reward_times = b['Reward/timestamps'][()]

        # imaging: concatenate planes (pooled for all analyses in the paper)
        planes = sorted(f['processing/ophys/Fluorescence'].keys())
        F = np.concatenate([f['processing/ophys/Fluorescence/%s/data' % p][()] for p in planes], axis=1).T
        Fneu = np.concatenate([f['processing/ophys/Neuropil/%s/data' % p][()] for p in planes], axis=1).T
        ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = ps['iscell'][:, 0] > 0

    # a few sessions have one extra imaging frame relative to the VR timeseries;
    # truncate to the common length so imaging and behavior stay aligned
    n_common = min(F.shape[1], len(pos))
    F = F[:, :n_common]
    Fneu = Fneu[:, :n_common]

    dt = float(np.median(np.diff(tstamps)))
    fs = 1.0 / dt
    starts = np.where(trial_start > 0)[0]
    stops = np.where(teleport > 0)[0]
    assert len(starts) == len(stops) and np.all(stops > starts)
    n_trials = len(starts)

    # keep curated cells only
    F = F[iscell]
    Fneu = Fneu[iscell]

    dff, events = compute_events(F, Fneu, starts, stops, fs)
    del F, Fneu

    # exclude putative interneurons: dF/F correlated with running speed
    in_trial = np.zeros(len(pos), dtype=bool)
    for s, e in zip(starts, stops):
        in_trial[s:e] = True
    sp_ = speed[in_trial]
    d_ = dff[:, in_trial]
    d_c = d_ - d_.mean(axis=1, keepdims=True)
    s_c = sp_ - sp_.mean()
    denom = np.sqrt((d_c ** 2).sum(axis=1) * (s_c ** 2).sum())
    r = np.divide((d_c * s_c).sum(axis=1), denom, out=np.zeros(d_.shape[0]), where=denom > 0)
    keep_cells = r <= INT_R_THRESH
    events = events[keep_cells]
    del dff, d_, d_c

    # per-trial variables
    z0, z1 = scene_zones(scene)
    zone_label = [z0 if t < CHANGE_TRIAL else z1 for t in range(n_trials)]
    reward_idx = np.searchsorted(tstamps, reward_times)
    trials = []
    n_lick_err = 0
    n_zone_mismatch = 0
    for t in range(n_trials):
        s, e = starts[t], stops[t]
        lk = lick[s:e]
        lick_err = np.sum(lk > 2) / len(lk) > LICK_ERR_THRESH
        n_lick_err += int(lick_err)
        rewarded = int(np.any((reward_idx >= s) & (reward_idx < e)))
        # sanity check the scene-derived zone against reward-zone entry times
        m = rzone_ts[s:e] > 0
        if m.sum() > 0:
            p_in = pos[s:e][m]
            lo, hi = ZONES[zone_label[t]]
            if not (lo - 15 <= p_in.min() <= hi + 15):
                n_zone_mismatch += 1
        trials.append(dict(s=int(s), e=int(e), lick_err=bool(lick_err), rewarded=rewarded,
                           zone=zone_label[t], env=int(np.round(np.median(env[s:e])))))

    # build decoder arrays
    neural, inp, out = [], [], []
    kept_trials = []
    for t in range(n_trials):
        tr = trials[t]
        if t == 0 or tr['lick_err']:
            continue  # no previous-trial outcome / unusable lick data
        s, e = tr['s'], tr['e']
        T = e - s
        neural.append(np.ascontiguousarray(events[:, s:e], dtype=np.float32))

        time_s = np.arange(T, dtype=np.float32) * dt
        inp.append(np.stack([
            time_s,
            np.full(T, tr['env'], dtype=np.float32),
            np.full(T, t, dtype=np.float32),
            np.full(T, trials[t - 1]['rewarded'], dtype=np.float32),
        ]).astype(np.float32))

        p = pos[s:e]
        lo, hi = ZONES[tr['zone']]
        d = np.zeros(T)
        d[p < lo] = p[p < lo] - lo
        d[p > hi] = p[p > hi] - hi
        # 0: <-50, 1: [-50,-10), 2: [-10,0), 3: 0, 4: (0,10], 5: (10,50], 6: >50
        rd = np.full(T, 3, dtype=np.int64)
        rd[d < -50] = 0
        rd[(d >= -50) & (d < -10)] = 1
        rd[(d >= -10) & (d < 0)] = 2
        rd[(d > 0) & (d <= 10)] = 4
        rd[(d > 10) & (d <= 50)] = 5
        rd[d > 50] = 6

        pbin = np.digitize(np.clip(p, 0, TRACK_LENGTH - 1e-6), [90.0, 180.0, 270.0, 360.0]).astype(np.int64)
        sp = speed[s:e]
        sbin = np.digitize(sp, [2.0, 10.0, 20.0, 40.0]).astype(np.int64)
        lk = (lick[s:e] > 0).astype(np.int64)

        out.append(np.stack([
            rd, pbin, sbin, lk,
            np.full(T, ZONE_IDX[tr['zone']], dtype=np.int64),
            np.full(T, tr['rewarded'], dtype=np.int64),
        ]).astype(np.int64))
        kept_trials.append(t)

    meta = dict(file=os.path.basename(fn), subject=subject, session_id=session_id,
                exp_day=int(session_id), scene=scene, identifier=ident,
                n_trials_total=n_trials, n_trials_kept=len(kept_trials),
                n_lick_error_trials=n_lick_err, n_zone_mismatch=n_zone_mismatch,
                n_cells_iscell=int(iscell.sum()), n_cells_kept=int(keep_cells.sum()),
                n_interneurons=int((~keep_cells).sum()), dt=dt, kept_trials=kept_trials,
                n_planes=len(planes))
    res = dict(neural=neural, input=inp, output=out, meta=meta)
    with open(os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl')), 'wb') as fo:
        pickle.dump(res, fo, protocol=4)
    print('done', os.path.basename(fn), meta['n_cells_kept'], 'cells,', len(kept_trials), 'trials,',
          'lick_err', n_lick_err, 'zone_mismatch', n_zone_mismatch, flush=True)
    return meta


def assemble():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
    data = dict(neural=[], input=[], output=[], subjects=[], subject_idx=[],
                brain_regions=['CA1'], brain_region_idx=[],
                input_names=['time_from_trial_start', 'environment', 'trial_number',
                             'previous_trial_outcome'],
                output_names=['reward_zone_distance', 'position', 'speed', 'lick',
                              'reward_zone_location', 'reward_outcome'],
                output_values=[
                    ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
                     '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
                    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
                    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
                    ['no lick', 'lick'],
                    ['zone A (80-130 cm)', 'zone B (200-250 cm)', 'zone C (320-370 cm)'],
                    ['omitted', 'rewarded'],
                ],
                metadata={})
    subjects = []
    session_info = []
    for fn in files:
        pkl = os.path.join(OUT_DIR, os.path.basename(fn).replace('.nwb', '.pkl'))
        if not os.path.exists(pkl):
            print('missing', pkl)
            continue
        with open(pkl, 'rb') as fi:
            r = pickle.load(fi)
        if len(r['neural']) < 2:
            print('skipping session with <2 trials:', pkl)
            continue
        m = r['meta']
        if m['subject'] not in subjects:
            subjects.append(m['subject'])
        data['neural'].append(r['neural'])
        data['input'].append(r['input'])
        data['output'].append(r['output'])
        data['subject_idx'].append(subjects.index(m['subject']))
        data['brain_region_idx'].append(np.zeros(r['neural'][0].shape[0], dtype=np.int64))
        session_info.append(m)
    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    dts = [s['dt'] for s in session_info]
    data['metadata'] = dict(
        task_description=(
            'Head-fixed mice ran laps on a 450 cm virtual linear track (ENV 1 or ENV 2) '
            'with a hidden 50 cm reward zone at one of three locations (A: 80-130 cm, '
            'B: 200-250 cm, C: 320-370 cm); licking in the zone delivered sucrose water, '
            'randomly omitted on ~15% of trials. On switch sessions the reward zone moved '
            'to a new location on trial 30. Decoder predicts, from CA1 population activity: '
            'signed distance to the reward zone, absolute track position, running speed, '
            'licking, the active reward zone location, and whether the trial was rewarded.'),
        time_bin_size=float(np.mean(dts)) * 1000.0,
        temporal_alignment_event='trial start (entry into the virtual linear track at 0 cm)',
        off_start=0.0,
        off_end=None,
        trial_definition=('each trial spans trial_start to teleport (one lap of the track); '
                          'trials have variable duration so off_end is not fixed'),
        neural_signal=('deconvolved calcium events (OASIS, tau=0.7 s) computed from per-trial '
                       'dF/F (neuropil-subtracted with 0.7 coefficient, maximin baseline over a '
                       '20 s window, 2-sample Gaussian smoothing), as in the reference paper'),
        neuron_curation=('suite2p iscell==1 (manually curated putative pyramidal cells); '
                         'putative interneurons excluded when dF/F-speed Pearson r > 0.5'),
        trial_curation=('trials with lick-sensor errors (>30% of samples with cumulative lick '
                        'count > 2) excluded; first trial of each session excluded because the '
                        'previous-trial outcome input is undefined'),
        imaging_rate_hz=float(1.0 / np.mean(dts)),
        deviations_from_paper=('the paper masked out samples with speed < 2 cm/s for its '
                               'decoding/spatial analyses; all in-trial samples are kept here '
                               'because speed is a decoder output and trials must be contiguous'),
        session_info=session_info,
        n_sessions=len(session_info),
        source='DANDI:001361, Sosa, Plitt & Giocomo 2025, Nature Neuroscience',
    )
    with open(OUT_FILE, 'wb') as fo:
        pickle.dump(data, fo, protocol=4)
    print('saved', OUT_FILE, 'sessions', len(data['neural']),
          'trials', sum(len(s) for s in data['neural']))


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    args = sys.argv[1:]
    if 'assemble' in args:
        assemble()
        sys.exit(0)
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
    if '--limit' in args:
        files = files[:int(args[args.index('--limit') + 1])]
    nproc = 12
    if '--nproc' in args:
        nproc = int(args[args.index('--nproc') + 1])
    ctx = mp.get_context('spawn')
    with ctx.Pool(nproc, maxtasksperchild=1) as p:
        for _ in p.imap_unordered(process_session, files):
            pass
    assemble()
