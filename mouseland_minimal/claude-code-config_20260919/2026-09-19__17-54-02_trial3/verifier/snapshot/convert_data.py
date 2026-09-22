"""
Convert the Zhong et al. (2025) "Unsupervised pretraining in biological neural networks"
two-photon dataset into the decoder format described in the task.

Experiment (see /app/methods.txt and code/data_process_script.ipynb):
  Head-fixed mice run on a ball through 4 m long virtual-reality corridors whose walls
  are covered with one of several naturalistic textures ("leaf"/"circle"/"rock"/"brick").
  Corridors are separated by 2 m of grey space.  A sound cue is played at a random
  position (0.5-3.5 m) inside every corridor; for the task ("sup") cohort the cue marks
  the beginning of the reward zone in the rewarded corridor.  Unsupervised ("unsup") and
  naive mice see the same corridors and cues but never get water.  Neural activity is
  deconvolved calcium (Suite2p, 2-photon mesoscope, ~3.18 Hz) from tens of thousands of
  neurons spread over V1 and higher visual areas.

Conversion decisions (see README-style notes in the code below):
  * one session == one recording (mname_datexp_blk); 89 recordings from 19 mice.
  * one trial   == one corridor traversal, aligned to corridor entry.
  * timepoints  == the native imaging frames of that trial that are inside the textured
                   corridor and during running, exactly the frames the paper analyses.
  * neurons     == neurons assigned to V1/mHV/lHV/aHV by the retinotopy files, randomly
                   subsampled to at most NEURONS_PER_SESSION per session.
"""

import os
import pickle
import collections
import datetime

import numpy as np

ROOT = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

# Maximum number of neurons kept per session.  The recordings contain 20,547-89,577
# neurons each; keeping all of them would make the dataset several hundred GB.  2000 is
# also the largest number of neurons for which the reference decoder computes an exact
# SVD initialisation of the per-session projection (decoder.svd_max_neurons), so nothing
# is gained by keeping more.  Neurons are drawn uniformly at random (fixed seed), which
# keeps the relative representation of the four visual areas unbiased.
NEURONS_PER_SESSION = 2000
SEED = 0

# Areas defined by utils.neu_area_ID; iarea values -1 and 7 are not assigned to any of
# the four areas analysed in the paper and those neurons are dropped.
AREA_NAMES = ['V1', 'mHV', 'lHV', 'aHV']
AREA_CODES = {'V1': (8,), 'mHV': (0, 1, 2, 9), 'lHV': (5, 6), 'aHV': (3, 4)}

# Canonical stimulus categories used throughout the paper (beh['stim_id']).  Different
# mice were trained with different texture pairs (leaf/circle, rock/brick, ...); the
# paper maps each mouse's stimuli onto this common set of roles so that stimuli are
# comparable across mice, and we decode that common set.
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']

POSITION_EDGES = [10.0, 20.0, 30.0]      # corridor is 40 units (=4 m) long -> 1 m bins
N_SPEED_BINS = 4


# ----------------------------------------------------------------------------------
# Pass 1: behaviour
# ----------------------------------------------------------------------------------
def load_behaviour():
    """Return an ordered dict recording-key -> compact behaviour dict.

    A recording can appear in several experiment types (e.g. a session is both
    'unsup_test1' and 'unsup_train2_before_learning', and test3 sessions appear twice
    with stimtype 'swap1'/'swap2').  The behaviour arrays are identical in all copies;
    only `stim_id` differs, because it holds the subset of stimuli used by that
    particular analysis (NaN = not used there).  We therefore read the behaviour arrays
    once per recording and merge `stim_id` over all experiment types so that every
    stimulus that the paper ever labels gets its canonical category.
    """
    exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=1).item()

    recordings = collections.OrderedDict()
    for exp_type, db in exp_info.items():
        Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                      allow_pickle=1).item()
        for d in db:
            key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
            behkey = key + ('_%s' % d['stimtype'] if 'stimtype' in d else '')
            beh = Beh[behkey]

            rec = recordings.setdefault(key, {'exp_types': [], 'wallmap': {}})
            rec['exp_types'].append(exp_type)
            for wall, cat in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], float)):
                if not np.isnan(cat):
                    rec['wallmap'][str(wall)] = int(cat)

            if 'ft' in rec:
                continue
            rec.update(dict(
                mname=d['mname'], datexp=d['datexp'], blk=d['blk'],
                # cohort: 'sup' (rewarded task), 'unsup' (VR exposure, no reward),
                # 'naive' (first ever exposure to these stimuli)
                cohort=d.get('exptype', 'naive'),
                ntrials=int(beh['ntrials']),
                ft=beh['ft'],
                ft_trInd=beh['ft_trInd'],
                ft_CorrSpc=beh['ft_CorrSpc'],
                ft_move=beh['ft_move'],
                ft_Pos=beh['ft_Pos'],
                ft_RunSpeed=beh['ft_RunSpeed'],
                LickFr=beh['LickFr'],
                StartFr=beh['StartFr'],
                SoundFr=beh['SoundFr'],
                isRew=beh['isRew'],
                WallName=np.asarray(beh['WallName'], dtype=str),
                texture_length=float(beh['Texture_Length']),
                reward_mode=str(beh['Reward_Mode']),
            ))
        del Beh
    return recordings


def add_frame_selection(rec, nframes):
    """Frames of this recording that belong to a trial, and their trial index.

    `nframes` is the number of imaging frames actually present in the spike file; the
    behaviour arrays are sometimes one frame longer (the paper's code truncates the same
    way, e.g. utils.Get_dprime_selective_neuron).

    A timepoint is kept when it is (a) assigned to a trial, (b) inside the textured part
    of the corridor (the 4 m the mouse discriminates in; the 2 m grey gap between
    corridors is not part of a trial) and (c) during running, i.e. the virtual reality
    was moving.  (c) is the paper's standard selection: "We only considered timepoints
    during running for analysis, which removed time periods when the task mice stopped
    to collect water rewards."  Without it, trial length varies from 11 to >600 frames
    depending on how long the mouse idled; with it, corridor traversals are 11-30 frames.
    """
    n = min(nframes, len(rec['ft']))
    trind = rec['ft_trInd'][:n]
    keep = (rec['ft_CorrSpc'][:n] & (rec['ft_move'][:n] > 0) & ~np.isnan(trind))
    frames = np.nonzero(keep)[0]
    return frames, trind[frames].astype(int)


# ----------------------------------------------------------------------------------
# neurons
# ----------------------------------------------------------------------------------
def select_neurons(rec, nneurons, rng):
    """Indices (sorted) of the neurons kept for this recording + their area index."""
    fn = os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (rec['mname'], rec['datexp']))
    iarea = np.load(fn, allow_pickle=True)['iarea']
    assert len(iarea) == nneurons, (fn, len(iarea), nneurons)

    area_idx = np.full(len(iarea), -1, dtype=np.int64)
    for a, name in enumerate(AREA_NAMES):
        area_idx[np.isin(iarea, AREA_CODES[name])] = a

    valid = np.nonzero(area_idx >= 0)[0]
    if len(valid) > NEURONS_PER_SESSION:
        valid = np.sort(rng.choice(valid, NEURONS_PER_SESSION, replace=False))
    return valid, area_idx[valid]


def load_spikes(key, neuron_getter):
    """Load deconvolved activity, selecting neurons plane by plane.

    utils.load_spk concatenates the per-plane blocks in order, and the retinotopy
    `iarea` follows that same order, so a global neuron index maps to (plane, row)
    by cumulative plane size.  Selecting inside the loop avoids ever materialising the
    full (nneurons x nframes) matrix, which can be 10 GB.
    """
    planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                     allow_pickle=True).item()['spks']
    sizes = [p.shape[0] for p in planes]
    nneurons, nframes = int(np.sum(sizes)), planes[0].shape[1]
    sel, area = neuron_getter(nneurons, nframes)

    offsets = np.concatenate([[0], np.cumsum(sizes)])
    out = np.empty((len(sel), nframes), dtype=np.float32)
    for p, plane in enumerate(planes):
        m = (sel >= offsets[p]) & (sel < offsets[p + 1])
        if m.any():
            out[m] = plane[sel[m] - offsets[p]]
    del planes
    return out, area, nframes


# ----------------------------------------------------------------------------------
# main conversion
# ----------------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    recordings = load_behaviour()
    keys = sorted(recordings.keys(),
                  key=lambda k: (recordings[k]['mname'], recordings[k]['datexp'],
                                 recordings[k]['blk']))
    print('%d recordings, %d mice' % (
        len(keys), len({recordings[k]['mname'] for k in keys})))

    # --- day of training -----------------------------------------------------------
    # Mice were trained/exposed for ~2 weeks and imaged repeatedly along the way; the
    # date of each recording is the only training-time information in the metadata.
    # We express it as days elapsed since that mouse's first recording (= its first
    # session in the virtual-reality corridors), which is 0 for every "naive"/"before
    # learning" session and grows through training.
    date = {k: datetime.date(*map(int, recordings[k]['datexp'].split('_'))) for k in keys}
    first = {}
    for k in keys:
        m = recordings[k]['mname']
        first[m] = min(first.get(m, date[k]), date[k])
    for k in keys:
        recordings[k]['day'] = float((date[k] - first[recordings[k]['mname']]).days)

    # --- running-speed quartiles ---------------------------------------------------
    # Running speed is discretised into 4 bins each holding 25% of the data.  The bin
    # edges are computed once over all timepoints that enter the dataset, so a speed bin
    # means the same thing in every session.
    speeds = []
    for k in keys:
        rec = recordings[k]
        frames, _ = add_frame_selection(rec, len(rec['ft']))
        speeds.append(rec['ft_RunSpeed'][frames])
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed quartile edges:', speed_edges,
          '(n=%d timepoints, using all frames of the behaviour files)' % len(speeds))

    data = {'neural': [], 'input': [], 'output': [], 'brain_region_idx': []}
    subjects, subject_idx, session_info = [], [], []
    n_dropped_stim = 0
    dts = []

    for k in keys:
        rec = recordings[k]

        neuron_getter = lambda nn, nf: select_neurons(rec, nn, rng)
        spk, area, nframes = load_spikes(k, neuron_getter)
        frames, trials = add_frame_selection(rec, nframes)

        dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)  # datenum -> s
        dts.append(dt)

        # stimulus category of each trial in the paper's canonical labelling
        wallmap = rec['wallmap']
        stim_of_trial = np.array([wallmap.get(w, -1) for w in rec['WallName']])

        # reward availability == "is this the rewarded corridor?".  beh['isRew'] marks
        # the trials on which water was actually delivered, which in the 'active after
        # cue' sessions requires the mouse to have licked -- using it directly would
        # leak the licking output into the decoder's input.  Instead we take the one
        # corridor in which reward was ever available in this session (always a single
        # texture, always canonical category 2 = the trained/rewarded stimulus) and mark
        # every trial of that corridor.  Unsupervised and naive mice never get water, so
        # reward availability is 0 for all of their trials.
        rewarded_walls = set(rec['WallName'][rec['isRew']])
        assert len(rewarded_walls) <= 1, (k, rewarded_walls)
        rew_of_trial = np.isin(rec['WallName'], list(rewarded_walls))

        # licking: a binary time series on the imaging frame grid
        licks = np.zeros(nframes, dtype=bool)
        lick_fr = np.round(rec['LickFr']).astype(int)
        lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]
        licks[lick_fr] = True

        pos = rec['ft_Pos']
        speed = rec['ft_RunSpeed']

        neural_s, input_s, output_s = [], [], []
        for t in range(rec['ntrials']):
            if stim_of_trial[t] < 0:
                # A few naive/unsupervised sessions also showed a third variant
                # ("circle3") that the paper never assigns a category to; those trials
                # cannot be labelled on the common scale and are dropped.
                n_dropped_stim += 1
                continue
            f = frames[trials == t]
            if len(f) == 0:
                continue

            # ---- inputs -----------------------------------------------------------
            # time to the sound cue: positive before the cue, negative after it
            time_to_cue = (rec['SoundFr'][t] - f) * dt
            time_since_start = (f - rec['StartFr'][t]) * dt
            day = np.full(len(f), rec['day'])
            rew = np.full(len(f), 1.0 if rew_of_trial[t] else 0.0)
            input_s.append(np.stack([time_to_cue, day, time_since_start, rew]
                                    ).astype(np.float32))

            # ---- outputs ----------------------------------------------------------
            out = np.empty((4, len(f)), dtype=np.int64)
            out[0] = stim_of_trial[t]
            out[1] = licks[f]
            out[2] = np.digitize(pos[f], POSITION_EDGES)
            out[3] = np.digitize(speed[f], speed_edges)
            output_s.append(out)

            neural_s.append(spk[:, f])

        del spk

        data['neural'].append(neural_s)
        data['input'].append(input_s)
        data['output'].append(output_s)
        data['brain_region_idx'].append(area)
        if rec['mname'] not in subjects:
            subjects.append(rec['mname'])
        subject_idx.append(subjects.index(rec['mname']))
        session_info.append(dict(
            session=k, mouse=rec['mname'], date=rec['datexp'], block=rec['blk'],
            cohort=rec['cohort'], experiment_types=sorted(set(rec['exp_types'])),
            day_of_training=rec['day'], reward_mode=rec['reward_mode'],
            ntrials=len(neural_s), nneurons=len(area),
            ntimepoints=int(sum(x.shape[1] for x in neural_s)),
            frame_rate_hz=1.0 / dt,
        ))
        print('%-22s cohort=%-5s day=%3d trials=%4d neurons=%4d timepoints=%6d'
              % (k, rec['cohort'], rec['day'], len(neural_s), len(area),
                 session_info[-1]['ntimepoints']), flush=True)

    data['subjects'] = subjects
    data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
    data['brain_regions'] = AREA_NAMES
    data['input_names'] = ['time_to_sound_cue', 'day_of_training',
                           'time_since_trial_start', 'reward_availability']
    data['output_names'] = ['stimulus_category', 'licking', 'position_bin',
                            'running_speed_bin']
    data['output_values'] = [
        STIM_NAMES,
        ['no_lick', 'lick'],
        ['0-1m', '1-2m', '2-3m', '3-4m'],
        ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4'],
    ]
    data['metadata'] = {
        'task_description':
            'Head-fixed mice run through 4 m virtual-reality corridors whose walls show '
            'one of several naturalistic textures; a sound cue is played at a random '
            'position in every corridor and, for the task ("sup") cohort only, marks the '
            'start of the reward zone in the rewarded corridor.  From the deconvolved '
            'activity of visual-cortex neurons the decoder predicts the stimulus '
            'category of the corridor, whether the mouse is licking, its position in the '
            'corridor (four 1 m bins) and its running speed (four quartile bins).',
        'time_bin_size': float(np.mean(dts)) * 1000.0,
        'temporal_alignment_event':
            'entry into the virtual-reality corridor (start of the trial)',
        'off_start': 0.0,
        'off_end': None,
        'trial_definition':
            'One corridor traversal.  Timepoints are the native two-photon frames from '
            'corridor entry to the end of the textured corridor (4 m) that were acquired '
            'while the mouse was running, i.e. while the virtual reality was moving, as '
            'in the paper.  Trials therefore start at the alignment event (off_start=0) '
            'but have variable length (11-30 frames, median 21), so off_end is N/A.',
        'input_descriptions': {
            'time_to_sound_cue': 'seconds until the sound cue (positive before the cue, '
                                 'negative after it)',
            'day_of_training': 'days elapsed since that mouse\'s first recording session',
            'time_since_trial_start': 'seconds since entry into the corridor',
            'reward_availability': '1 if this corridor is the one in which water was '
                                   'available in this session (task cohort only), '
                                   '0 otherwise',
        },
        'output_descriptions': {
            'stimulus_category': 'wall texture of the corridor, mapped onto the canonical '
                                 'categories used in the paper (beh["stim_id"]); mice '
                                 'trained on rock/brick are labelled by the matching role',
            'licking': '1 if at least one lick was detected in this imaging frame',
            'position_bin': 'position along the 4 m corridor in 1 m bins',
            'running_speed_bin': 'running speed quartile, edges %s (a.u., ball tracking)'
                                 % np.round(speed_edges, 3).tolist(),
        },
        'speed_bin_edges': speed_edges.tolist(),
        'neural_signal': 'deconvolved two-photon calcium fluorescence (Suite2p), '
                         'sampled at ~3.18 Hz, not normalised',
        'neurons_per_session': NEURONS_PER_SESSION,
        'neuron_selection':
            'neurons assigned to V1/mHV/lHV/aHV by the retinotopic map '
            '(utils.neu_area_ID), randomly subsampled to at most %d per session'
            % NEURONS_PER_SESSION,
        'ntrials_dropped_unlabelled_stimulus': n_dropped_stim,
        'session_info': session_info,
        'source': 'Zhong et al., Unsupervised pretraining in biological neural networks',
    }

    ntr = sum(len(s) for s in data['neural'])
    ntp = sum(x.shape[1] for s in data['neural'] for x in s)
    print('\n%d sessions, %d trials, %d timepoints, %.1f GB'
          % (len(keys), ntr, ntp,
             sum(x.nbytes for s in data['neural'] for x in s) / 1e9))
    print('dropped %d trials with an unlabelled stimulus' % n_dropped_stim)

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('wrote', OUT_FILE)


if __name__ == '__main__':
    main()
