"""
Convert the Zhong et al. (2025) "Unsupervised pretraining in biological neural
networks" dataset into the decoder format described in the task instructions.

Experiment (see /app/methods.txt and the paper):
  Head-fixed mice run on an air-floating ball through 4 m long virtual-reality
  corridors whose walls are covered with one of several 'frozen' naturalistic
  texture patterns ('leaf1', 'circle1', ... -- canonical names used by the paper
  even when the physical textures were rock/brick/wood).  Corridors are separated
  by 2 m of grey space and are presented in pseudo-random order.  A sound cue is
  played at a random position (0.5-3.5 m) inside every corridor.  For the task
  ('sup') cohort, water is available after the cue in the rewarded corridor only;
  the unsupervised and naive cohorts run through the same corridors without any
  reward.  Neural activity is 2-photon mesoscope calcium imaging across visual
  cortex (V1 + higher visual areas), deconvolved with Suite2p at ~3.17 Hz.

Conversion decisions (all follow the paper / the released analysis code unless
the decoder task specification requires otherwise) are documented inline and
summarised in the module docstring of each step below.

Usage:
    python convert_data.py [--out /app/converted_data.pkl] [--nsub 2000]
                           [--max-sessions N] [--workers K]
"""

import argparse
import collections
import datetime
import os
import pickle

import numpy as np

DATA_ROOT = '/app/data'

# Canonical stimulus names, indexed by beh['stim_id'] (see data_process_script.ipynb:
# "0:circle1, 1:circle2, 2:leaf1, 3:leaf2, 4:leaf3, 5:leaf1_swap1, 6:leaf1_swap2").
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']

# Visual areas, exactly as utils.neu_area_ID groups the retinotopy 'iarea' codes.
# iarea values -1 and 7 are not part of any of the four groups; the paper's code
# (utils.Get_density_map) explicitly calls these "neurons from outside of visual
# cortex" and excludes them, so we do the same.
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
AREA_CODE_TO_REGION = {8: 0,                      # V1
                       0: 1, 1: 1, 2: 1, 9: 1,    # medial HV
                       5: 2, 6: 2,                # lateral HV
                       3: 3, 4: 3}                # anterior HV

# Corridor geometry (constant across all 89 recordings: Corridor_Length 60,
# Texture_Length 40, Gray_Space_length 20, in decimetres).
TEXTURE_LENGTH_DM = 40.0
POSITION_BIN_DM = 10.0          # 1 m spatial bins -> 4 bins over the 4 m corridor
N_POSITION_BINS = 4
N_SPEED_BINS = 4

# A trial is kept only if the animal actually traversed the corridor while the
# imaging was running; 4 m of virtual corridor at the fixed VR speed takes ~21
# imaging frames, so this only removes truncated (start/end of recording) trials.
MIN_FRAMES_PER_TRIAL = 5

SEED = 0


# -----------------------------------------------------------------------------
# loading helpers (mirroring code/utils.py)
# -----------------------------------------------------------------------------

def load_exp_info():
    return np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()


def load_all_behavior(exp_info):
    """Group every behaviour entry by the recording (mouse, date, block) it came from.

    The same recording is listed under several 'experiment types' (e.g. a session
    is both 'unsup_test1' and 'unsup_train2_before_learning', and the two
    leaf1-swap stimuli of one session are listed as two entries 'swap1'/'swap2').
    Those entries share the same behaviour, but each one only names the stimuli
    that are relevant for that particular comparison in the paper -- the others
    are left as the placeholder string 'stimulus_of_trial' in beh['TrialStim'].
    We therefore merge the UniqWalls -> stim_id maps of all entries of a
    recording to recover the stimulus identity of every trial, and convert each
    recording exactly once.
    """
    beh_files = {et: np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % et),
                             allow_pickle=True).item() for et in exp_info}
    sessions = collections.OrderedDict()
    for exp_type, db in exp_info.items():
        for ndb in db:
            rec = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
            key = rec + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
            entry = sessions.setdefault(rec, {
                'mname': ndb['mname'], 'datexp': ndb['datexp'], 'blk': ndb['blk'],
                'exp_types': [], 'cohorts': set(), 'wall2stim': {}, 'beh': None})
            entry['exp_types'].append(exp_type)
            if ndb.get('exptype'):
                entry['cohorts'].add(ndb['exptype'])
            beh = beh_files[exp_type][key]
            entry['beh'] = beh
            for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
                if not np.isnan(float(sid)):
                    entry['wall2stim'][str(wall)] = int(sid)
    return sessions


def load_spikes(rec):
    """Deconvolved activity, neurons x frames, planes concatenated (utils.load_spk)."""
    path = os.path.join(DATA_ROOT, 'spk', '%s_neural_data.npy' % rec)
    return np.concatenate(np.load(path, allow_pickle=True).item()['spks'], 0)


def load_areas(mname, datexp):
    """Visual-area id of every neuron, in the same order as the activity matrix."""
    path = os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp))
    return np.load(path, allow_pickle=True)['iarea']


# -----------------------------------------------------------------------------
# behaviour -> trial structure
# -----------------------------------------------------------------------------

def frame_mask(beh, nframes):
    """Imaging frames that belong to a corridor traversal and are used for analysis.

    Three conditions, all taken from the paper:
      * beh['ft_CorrSpc']: the animal is inside the textured part of the corridor
        (the 4 m over which the visual stimulus is shown).  The 2 m of grey space
        between corridors is excluded, which also makes the trial window match
        the 4 x 1 m position bins requested by the decoder task.
      * beh['ft_move'] > 0: the virtual reality advanced during this frame, i.e.
        the animal was running above the 6 cm/s threshold.  "We only considered
        timepoints during running for analysis, which removed time periods when
        the task mice stopped to collect water rewards" (methods.txt); the
        released code uses exactly this mask (`VRmove = beh['ft_move']>0`,
        `corr_fr = beh['ft_CorrSpc'] & VRmove`).
      * the frame is assigned to a trial (ft_trInd is not NaN, i.e. it falls
        inside the behaviourally recorded period).

    All behaviour arrays are truncated to the number of imaging frames, as in
    utils.Get_dprime_selective_neuron (`nfr = spk.shape[1]`; the behaviour files
    can contain a few frames more than the imaging files).
    """
    corridor = beh['ft_CorrSpc'][:nframes]
    running = beh['ft_move'][:nframes] > 0
    in_trial = ~np.isnan(beh['ft_trInd'][:nframes])
    return corridor & running & in_trial


def trial_frames(beh, nframes):
    """dict trial index -> array of imaging-frame indices, ordered in time."""
    mask = frame_mask(beh, nframes)
    frames = np.where(mask)[0]
    trials = beh['ft_trInd'][:nframes][frames].astype(int)
    order = np.argsort(trials, kind='stable')
    frames, trials = frames[order], trials[order]
    splits = np.searchsorted(trials, np.arange(int(beh['ntrials']) + 1))
    return {t: frames[splits[t]:splits[t + 1]] for t in range(int(beh['ntrials']))}


def lick_frames(beh, nframes):
    """Boolean per imaging frame: did the animal lick during this frame?"""
    licks = np.zeros(nframes, dtype=bool)
    lick_fr = np.asarray(beh['LickFr'], dtype=float)
    lick_fr = lick_fr[np.isfinite(lick_fr)]
    if lick_fr.size:
        idx = np.round(lick_fr).astype(int)
        idx = idx[(idx >= 0) & (idx < nframes)]
        licks[idx] = True
    return licks


def session_stimulus_names(entry):
    """Canonical stimulus name of each trial, or None where the paper gives none."""
    wall2stim = entry['wall2stim']
    return [STIM_NAMES[wall2stim[str(w)]] if str(w) in wall2stim else None
            for w in entry['beh']['WallName']]


def day_of_training(entry, first_day):
    """Days elapsed since the first recording of this mouse.

    The dataset does not record the calendar date on which each animal started
    its virtual-reality training, so the first imaging session of a mouse is used
    as its reference day.  For every mouse that has a 'before learning' session
    this is literally day 0 of training / of unsupervised exposure; for the naive
    cohort it is the day of the naive recording.
    """
    date = datetime.date(*map(int, entry['datexp'].split('_')))
    return float((date - first_day).days)


# -----------------------------------------------------------------------------
# per-session conversion
# -----------------------------------------------------------------------------

def collect_behavior(sessions):
    """First pass (behaviour only): trial structure, and the global speed quartiles.

    The running-speed output has to be binned into four bins "each corresponding
    to 25% of the data", so the bin edges are the quartiles of the running speed
    over every timepoint that ends up in the dataset (pooled over all sessions,
    which is what makes each bin hold 25% of the data).
    """
    speeds = []
    per_session = {}
    for rec, entry in sessions.items():
        beh = entry['beh']
        nframes = len(beh['ft'])
        frames = trial_frames(beh, nframes)
        names = session_stimulus_names(entry)
        keep = [t for t in range(int(beh['ntrials']))
                if len(frames[t]) >= MIN_FRAMES_PER_TRIAL and names[t] is not None]
        per_session[rec] = keep
        for t in keep:
            speeds.append(beh['ft_RunSpeed'][:nframes][frames[t]])
    speeds = np.concatenate(speeds)
    edges = np.percentile(speeds, [25, 50, 75])
    return per_session, edges


def convert_session(rec, entry, speed_edges, day, nsub, session_index):
    """Build the neural / input / output lists of one recording."""
    beh = entry['beh']

    # --- neurons -------------------------------------------------------------
    spk = load_spikes(rec)
    nneurons_total, nframes = spk.shape
    iarea = load_areas(entry['mname'], entry['datexp'])
    assert len(iarea) == nneurons_total, rec

    # Keep only neurons that the paper assigns to one of the four visual-cortex
    # area groups (utils.neu_area_ID); iarea == -1 / 7 lie outside visual cortex.
    region = np.full(nneurons_total, -1, dtype=np.int64)
    for code, ridx in AREA_CODE_TO_REGION.items():
        region[iarea == code] = ridx
    sd = spk.std(axis=1)
    usable = np.where((region >= 0) & (sd > 0))[0]   # silent neurons cannot be z-scored

    # Sub-sample neurons.  Sessions contain 20k-90k neurons; keeping all of them
    # would make the dataset ~150 GB while the decoder compresses each session to
    # 100 components (and its own SVD initialisation already random-projects any
    # session with more than 2000 neurons).  A uniform random subset preserves the
    # relative area composition of the recording in expectation.
    rng = np.random.default_rng(SEED + session_index)
    if nsub is not None and len(usable) > nsub:
        usable = np.sort(rng.choice(usable, size=nsub, replace=False))

    # z-score every neuron over the whole recording, as the paper's code does
    # (utils.get_kfold_reward_response: `spk = stats.zscore(spk, axis=1)`).  This
    # puts sessions with very different deconvolved amplitudes on a common scale.
    activity = spk[usable]
    activity = (activity - activity.mean(axis=1, keepdims=True)) / sd[usable][:, None]
    activity = activity.astype(np.float32)
    del spk

    # --- trials --------------------------------------------------------------
    frames = trial_frames(beh, nframes)
    names = session_stimulus_names(entry)
    licks = lick_frames(beh, nframes)
    ft = beh['ft'][:nframes]
    pos = beh['ft_Pos'][:nframes]
    speed = beh['ft_RunSpeed'][:nframes]

    neural, inputs, outputs, trial_ids = [], [], [], []
    for t in range(int(beh['ntrials'])):
        fr = frames[t]
        if len(fr) < MIN_FRAMES_PER_TRIAL or names[t] is None:
            continue

        # inputs: seconds are derived from the frame timestamps (MATLAB datenum
        # -> seconds), so gaps created by removing non-running frames are kept.
        t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
        t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
        inp = np.empty((4, len(fr)), dtype=np.float32)
        inp[0] = t_to_cue
        inp[1] = day
        inp[2] = t_since_start
        inp[3] = float(beh['isRew'][t])

        out = np.empty((4, len(fr)), dtype=np.int64)
        out[0] = STIM_NAMES.index(names[t])
        out[1] = licks[fr]
        out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
        out[3] = np.searchsorted(speed_edges, speed[fr], side='right')

        neural.append(np.ascontiguousarray(activity[:, fr]))
        inputs.append(inp)
        outputs.append(out)
        trial_ids.append(t)

    info = {
        'session': rec,
        'mouse': entry['mname'],
        'date': entry['datexp'],
        'block': entry['blk'],
        'experiment_types': sorted(set(entry['exp_types'])),
        'cohort': ('sup' if 'sup' in entry['cohorts'] else
                   'unsup' if 'unsup' in entry['cohorts'] else 'naive'),
        'day_of_training': day,
        'rewarded_session': bool(np.any(beh['isRew'])),
        'n_trials_recorded': int(beh['ntrials']),
        'n_trials_kept': len(neural),
        'n_neurons_recorded': int(nneurons_total),
        'n_neurons_kept': int(len(usable)),
        'n_imaging_frames': int(nframes),
        'frame_period_s': float(np.median(np.diff(ft)) * 86400.0),
        'trial_indices': trial_ids,
    }
    return neural, inputs, outputs, region[usable].astype(np.int64), info


def _worker(args):
    return convert_session(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='/app/converted_data.pkl')
    parser.add_argument('--nsub', type=int, default=2000,
                        help='max neurons kept per session (0 = keep all)')
    parser.add_argument('--max-sessions', type=int, default=None)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    nsub = args.nsub if args.nsub > 0 else None

    exp_info = load_exp_info()
    sessions = load_all_behavior(exp_info)
    recs = list(sessions)
    if args.max_sessions:
        recs = recs[:args.max_sessions]
        sessions = collections.OrderedDict((r, sessions[r]) for r in recs)

    kept_trials, speed_edges = collect_behavior(sessions)
    print('speed quartile edges (cm/s):', speed_edges)
    print('%d recordings, %d trials' % (len(recs), sum(len(v) for v in kept_trials.values())))

    first_day = {}
    for entry in sessions.values():
        date = datetime.date(*map(int, entry['datexp'].split('_')))
        m = entry['mname']
        first_day[m] = min(first_day.get(m, date), date)

    jobs = [(rec, sessions[rec], speed_edges,
             day_of_training(sessions[rec], first_day[sessions[rec]['mname']]),
             nsub, i) for i, rec in enumerate(recs)]

    results = [None] * len(jobs)
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap(_worker, jobs)):
                results[i] = res
                print('[%d/%d] %s: %d trials, %d neurons'
                      % (i + 1, len(jobs), res[4]['session'], res[4]['n_trials_kept'],
                         res[4]['n_neurons_kept']), flush=True)
    else:
        for i, job in enumerate(jobs):
            results[i] = _worker(job)
            print('[%d/%d] %s' % (i + 1, len(jobs), results[i][4]['session']), flush=True)

    subjects = sorted({s['mname'] for s in sessions.values()})
    session_info = [r[4] for r in results]
    data = {
        'neural': [r[0] for r in results],
        'input': [r[1] for r in results],
        'output': [r[2] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(i['mouse']) for i in session_info],
                                dtype=np.int64),
        'brain_regions': BRAIN_REGIONS,
        'brain_region_idx': [r[3] for r in results],
        'input_names': ['time_to_sound_cue', 'day_of_training',
                        'time_since_trial_start', 'reward_available'],
        'output_names': ['stimulus', 'licking', 'position_bin', 'speed_bin'],
        'output_values': [
            list(STIM_NAMES),
            ['no_lick', 'lick'],
            ['0-1m', '1-2m', '2-3m', '3-4m'],
            ['speed_Q1', 'speed_Q2', 'speed_Q3', 'speed_Q4'],
        ],
        'metadata': {
            'task_description':
                'Head-fixed mice run through 4 m virtual-reality corridors whose walls '
                'show one of several frozen naturalistic textures (canonical names '
                'circle1/circle2/leaf1/leaf2/leaf3/leaf1_swap1/leaf1_swap2); corridors '
                'are separated by 2 m of grey space and presented in pseudo-random '
                'order. A sound cue is played at a random position in every corridor; '
                'for the task ("sup") cohort water is available after the cue in the '
                'rewarded corridor only, while the unsupervised and naive cohorts run '
                'through the same corridors with no reward. From two-photon mesoscope '
                'recordings of visual cortex the decoder predicts, at every imaging '
                'frame of a corridor traversal: which texture is on the walls, whether '
                'the animal is licking, which 1 m bin of the 4 m corridor it is in, and '
                'which quartile of the running-speed distribution its speed falls in.',
            'time_bin_size': float(np.median([i['frame_period_s'] for i in session_info]) * 1000.0),
            'temporal_alignment_event':
                'trial start = corridor entry (the frame at which the animal enters the '
                'textured corridor, beh["Trial_start_time"]/beh["StartFr"])',
            'off_start': 0.0,
            'off_end': None,
            'trial_window':
                'From corridor entry to corridor exit (4 m later). Only frames during '
                'which the virtual reality advanced (the animal ran faster than the '
                '6 cm/s threshold) are kept, following the paper ("We only considered '
                'timepoints during running for analysis"), so trials have a variable '
                'number of frames (median 21, i.e. ~6.6 s of running) and off_end is '
                'not a fixed time.',
            'recording': 'two-photon mesoscope calcium imaging, Suite2p deconvolved '
                         'traces (0.75 s decay timescale), ~3.17 Hz, z-scored per neuron '
                         'over the whole recording',
            'brain_region_note':
                'V1 and the medial (mHV), lateral (lHV) and anterior (aHV) higher visual '
                'area groups of utils.neu_area_ID; neurons outside visual cortex '
                '(retinotopy iarea -1 and 7) are excluded.',
            'input_units': {'time_to_sound_cue': 'seconds until the sound cue (negative '
                                                 'after the cue)',
                            'day_of_training': 'days since the first imaging session of '
                                               'this mouse',
                            'time_since_trial_start': 'seconds since corridor entry',
                            'reward_available': '1 if this corridor is the rewarded one '
                                                '(task cohort only), else 0'},
            'speed_bin_edges_cm_per_s': [float(e) for e in speed_edges],
            'neurons_subsampled_per_session': nsub,
            'session_info': session_info,
        },
    }

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    nbytes = sum(t.nbytes for s in data['neural'] for t in s)
    print('wrote %s (neural %.2f GB, %d sessions, %d trials)'
          % (args.out, nbytes / 1e9, len(data['neural']),
             sum(len(s) for s in data['neural'])))


if __name__ == '__main__':
    main()
