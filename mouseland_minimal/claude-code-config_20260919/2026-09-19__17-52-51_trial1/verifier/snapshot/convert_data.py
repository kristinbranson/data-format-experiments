"""
Convert the Zhong et al. 2025 ("Unsupervised pretraining in biological neural networks")
two-photon dataset into the trial-structured dictionary expected by train_decoder.py.

Data source (/app/data):
  beh/Imaging_Exp_info.npy        experiment table, one entry per (recording, analysis type)
  beh/Beh_<exp_type>.npy          behaviour dict per recording, keyed '<mouse>_<date>_<block>'
  spk/<mouse>_<date>_<blk>_neural_data.npy   deconvolved calcium traces, list of imaging planes
  retinotopy/<mouse>_<date>_trans.npz        per-neuron visual-area assignment (`iarea`)

Processing follows the reference code in /app/code/utils.py (see README of decisions in
the accompanying report):

  * one session per unique recording (mouse, date, block); the 89 recordings appear
    several times in Imaging_Exp_info under different analysis names but the behaviour
    dictionaries are identical, so they are de-duplicated.
  * a trial is one traversal of the 4 m textured corridor, from corridor entry
    (`Trial_start_time` / `StartFr`) to entry into the 2 m grey space (`GrayFr`).
  * frames are kept only when the mouse was running, i.e. when the virtual reality was
    advancing (`ft_move > 0`) and the mouse was inside the textured part of the corridor
    (`ft_CorrSpc`). This is exactly the frame selection used throughout utils.py
    (`fr_valid = VRmove & isCorridor`) and described in the paper ("We only considered
    timepoints during running for analysis").
  * neurons are restricted to the four visual-area groups used in the paper
    (V1, mHV, lHV, aHV; utils.neu_area_ID) and randomly subsampled to --max-neurons per
    session.
"""

import argparse
import datetime
import os
import pickle
from collections import Counter, OrderedDict
from multiprocessing import Pool

import numpy as np

# canonical stimulus identities used by the paper / utils.py (`all_stim`).
# stim_id in Imaging_Exp_info indexes into this list. Mice trained on the rock/brick/wood
# texture pairs are mapped onto the same identities by the authors' own stim_id tables.
# 'circle3' is the one wall type that never receives a stim_id (it is never analysed in
# the paper); it is kept as an extra category so that no trial has to be discarded.
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']
EXTRA_STIM = ['circle3']
STIM_VALUES = CANONICAL_STIM + EXTRA_STIM

BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']

INPUT_NAMES = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start',
               'reward_available']
OUTPUT_NAMES = ['stimulus', 'lick', 'position_bin', 'speed_bin']

POSITION_VALUES = ['0-1m', '1-2m', '2-3m', '3-4m']
LICK_VALUES = ['no_lick', 'lick']
SPEED_VALUES = ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']

# behaviour fields needed downstream (keeps the in-memory copy of Beh_*.npy small)
BEH_FIELDS = ['ntrials', 'Trial_start_time', 'SoundTime', 'isRew', 'WallName',
              'UniqWalls', 'LickFr', 'ft', 'ft_trInd', 'ft_Pos', 'ft_move',
              'ft_CorrSpc', 'ft_RunSpeed', 'Corridor_Length', 'Texture_Length',
              'Reward_Mode']


def neu_area_idx(iarea):
    """Region index per neuron, following utils.neu_area_ID. -1 = not one of the four
    area groups analysed in the paper (unassigned neurons, iarea==-1, and iarea==7)."""
    idx = np.full(len(iarea), -1, dtype=np.int64)
    idx[iarea == 8] = 0                                              # V1
    idx[np.isin(iarea, [0, 1, 2, 9])] = 1                            # medial HVAs
    idx[np.isin(iarea, [5, 6])] = 2                                  # lateral HVAs
    idx[np.isin(iarea, [3, 4])] = 3                                  # anterior HVAs
    return idx


def session_key(d):
    return '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])


def beh_key(d):
    return session_key(d) + (('_' + d['stimtype']) if 'stimtype' in d else '')


def collect_sessions(root):
    """One entry per unique recording, with merged stimulus-name -> canonical-id map."""
    exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    sessions = OrderedDict()
    for exp_type, db in exp_info.items():
        for d in db:
            kn = session_key(d)
            s = sessions.setdefault(kn, {
                'kn': kn, 'mname': d['mname'], 'datexp': d['datexp'], 'blk': d['blk'],
                'exp_types': [], 'stim_map': {}, 'beh_source': (exp_type, beh_key(d)),
            })
            if exp_type not in s['exp_types']:
                s['exp_types'].append(exp_type)
            s['stim_id_entries'] = s.get('stim_id_entries', []) + \
                [(exp_type, beh_key(d), np.asarray(d['stim_id'], dtype=float))]
    return sessions


def load_behaviour(root, sessions):
    """Load each Beh_<exp_type>.npy once and keep only the fields we need."""
    by_exp = {}
    for kn, s in sessions.items():
        by_exp.setdefault(s['beh_source'][0], []).append(kn)
    beh = {}
    for exp_type, kns in by_exp.items():
        B = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type),
                    allow_pickle=True).item()
        for kn in kns:
            b = B[sessions[kn]['beh_source'][1]]
            beh[kn] = {f: b[f] for f in BEH_FIELDS}
        del B
    return beh


def build_stim_map(session, beh):
    """Map each wall name in the session onto a canonical stimulus index.

    The stim_id vectors in Imaging_Exp_info align with the (sorted) UniqWalls of the
    session; each analysis entry only labels the stimuli it uses, so the entries for a
    recording are merged. Wall types that are never labelled ('circle3') get their own
    category."""
    uniq = list(beh['UniqWalls'])
    mapping = {}
    for exp_type, key, stim_id in session['stim_id_entries']:
        if len(stim_id) != len(uniq):
            raise ValueError('stim_id length mismatch for %s (%s)' % (session['kn'], exp_type))
        for wall, sid in zip(uniq, stim_id):
            if not np.isnan(sid):
                sid = int(sid)
                if wall in mapping and mapping[wall] != sid:
                    raise ValueError('conflicting stim_id for %s / %s' % (session['kn'], wall))
                mapping[wall] = sid
    for wall in uniq:
        if wall not in mapping:
            if wall not in EXTRA_STIM:
                raise ValueError('unlabelled wall type %s in %s' % (wall, session['kn']))
            mapping[wall] = len(CANONICAL_STIM) + EXTRA_STIM.index(wall)
    return mapping


def trial_frames(beh, nfr):
    """Frame indices per trial: running frames inside the textured corridor.

    Returns (list of index arrays, trial index array) with one entry per trial."""
    ft_trInd = beh['ft_trInd'][:nfr]
    valid = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr] & ~np.isnan(ft_trInd)
    frames = np.nonzero(valid)[0]
    tr = ft_trInd[frames].astype(np.int64)
    order = np.argsort(tr, kind='stable')
    frames, tr = frames[order], tr[order]
    bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
    return [frames[bounds[i]:bounds[i + 1]] for i in range(int(beh['ntrials']))]


def session_behaviour_arrays(session, beh, nfr, day_of_training, speed_edges=None):
    """Per-trial inputs/outputs (everything except the neural data)."""
    stim_map = build_stim_map(session, beh)
    per_trial = trial_frames(beh, nfr)

    ft = beh['ft'][:nfr]
    pos = beh['ft_Pos'][:nfr]
    speed = beh['ft_RunSpeed'][:nfr]
    lick = np.zeros(nfr, dtype=bool)
    if len(beh['LickFr']) > 0:
        lf = np.round(np.asarray(beh['LickFr'], dtype=float))
        lf = lf[np.isfinite(lf)].astype(np.int64)
        lick[lf[(lf >= 0) & (lf < nfr)]] = True

    texture_len = float(beh['Texture_Length'])          # 40 units == 4 m
    bin_len = texture_len / len(POSITION_VALUES)

    keep, inputs, outputs, frames_out = [], [], [], []
    for trial, fr in enumerate(per_trial):
        if len(fr) < 2:
            continue
        t0 = beh['Trial_start_time'][trial]
        tcue = beh['SoundTime'][trial]
        wall = beh['WallName'][trial]
        if not np.isfinite(t0) or not np.isfinite(tcue):
            continue
        t = (ft[fr] - t0) * 86400.0                      # seconds since corridor entry
        to_cue = (tcue - ft[fr]) * 86400.0               # seconds until the sound cue
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(to_cue))):
            continue
        p = pos[fr]
        s = speed[fr]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(s))):
            continue
        pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)

        inp = np.empty((len(INPUT_NAMES), len(fr)), dtype=np.float32)
        inp[0] = to_cue
        inp[1] = day_of_training
        inp[2] = t
        inp[3] = 1.0 if beh['isRew'][trial] else 0.0

        out = np.empty((len(OUTPUT_NAMES), len(fr)), dtype=np.int64)
        out[0] = stim_map[wall]
        out[1] = lick[fr].astype(np.int64)
        out[2] = pos_bin
        if speed_edges is not None:
            out[3] = np.searchsorted(speed_edges, s, side='right')
        else:
            out[3] = 0

        keep.append(trial)
        inputs.append(inp)
        outputs.append(out)
        frames_out.append(fr)
    return dict(trials=keep, input=inputs, output=outputs, frames=frames_out,
                speeds=np.concatenate([speed[f] for f in frames_out]) if frames_out
                else np.zeros(0), stim_map=stim_map)


def process_session(job):
    """Worker: extract the neural data for one session.

    The spk file stores one array per imaging plane; neurons are indexed by concatenating
    the planes in order (utils.load_spk)."""
    (root, session, beh, day_of_training, speed_edges, max_neurons, seed) = job
    kn = session['kn']
    ret = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' %
                               (session['mname'], session['datexp'])), allow_pickle=True)
    region = neu_area_idx(np.asarray(ret['iarea']))
    n_total = len(region)

    candidates = np.nonzero(region >= 0)[0]
    rng = np.random.default_rng(seed)
    if max_neurons is not None and len(candidates) > max_neurons:
        rows = np.sort(rng.choice(candidates, size=max_neurons, replace=False))
    else:
        rows = candidates

    # the number of imaging frames in the recording bounds the behaviour arrays
    # (utils.py slices the behaviour with spk.shape[1] in the same way)
    spk_path = os.path.join(root, 'spk', '%s_neural_data.npy' % kn)
    spks = np.load(spk_path, allow_pickle=True).item()['spks']
    nfr = min(p.shape[1] for p in spks)
    n_file = sum(p.shape[0] for p in spks)
    if n_file != n_total:
        raise ValueError('%s: %d neurons in spk file but %d in retinotopy'
                         % (kn, n_file, n_total))

    beh_arrays = session_behaviour_arrays(session, beh, nfr, day_of_training, speed_edges)
    frames = np.concatenate(beh_arrays['frames']) if beh_arrays['frames'] else np.zeros(0, int)

    out = np.empty((len(rows), len(frames)), dtype=np.float32)
    offset = 0
    for plane in spks:
        n = plane.shape[0]
        sel = (rows >= offset) & (rows < offset + n)
        if sel.any():
            out[sel] = plane[rows[sel] - offset][:, frames]
        offset += n
    del spks

    neural = []
    start = 0
    for fr in beh_arrays['frames']:
        neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
        start += len(fr)
    del out

    return dict(kn=kn, neural=neural, input=beh_arrays['input'],
                output=beh_arrays['output'], trials=beh_arrays['trials'],
                brain_region_idx=region[rows].astype(np.int64),
                n_neurons_total=int(n_total), n_neurons_in_areas=int(len(candidates)),
                n_neurons=int(len(rows)), nfr=int(nfr), stim_map=beh_arrays['stim_map'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/app/data')
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--max-neurons', type=int, default=2000,
                    help='random subsample of neurons per session (0 = keep all)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit-sessions', type=int, default=0,
                    help='only convert the first N sessions (debugging)')
    args = ap.parse_args()

    max_neurons = args.max_neurons if args.max_neurons > 0 else None

    sessions = collect_sessions(args.root)
    if args.limit_sessions:
        sessions = OrderedDict(list(sessions.items())[:args.limit_sessions])
    print('%d unique recordings' % len(sessions), flush=True)

    beh = load_behaviour(args.root, sessions)
    print('behaviour loaded', flush=True)

    # day of training: days since the first recording of that mouse (the released data
    # does not contain the absolute start of training)
    first_date = {}
    for kn, s in sessions.items():
        d = datetime.date(*map(int, s['datexp'].split('_')))
        first_date[s['mname']] = min(first_date.get(s['mname'], d), d)
    day_of_training = {kn: (datetime.date(*map(int, s['datexp'].split('_')))
                            - first_date[s['mname']]).days
                       for kn, s in sessions.items()}

    # pass 1 (behaviour only): global running-speed quartiles over all kept timepoints
    speeds = []
    frame_rates = []
    for kn, s in sessions.items():
        b = beh[kn]
        nfr = len(b['ft'])
        arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
        speeds.append(arrays['speeds'])
        frame_rates.append(1.0 / (np.median(np.diff(b['ft'])) * 86400.0))
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed quartile edges (cm/s):', np.round(speed_edges, 3), flush=True)
    time_bin_size = 1000.0 / float(np.median(frame_rates))
    print('time bin: %.2f ms' % time_bin_size, flush=True)

    jobs = [(args.root, s, beh[kn], day_of_training[kn], speed_edges,
             max_neurons, args.seed + i)
            for i, (kn, s) in enumerate(sessions.items())]

    results = {}
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
            results[res['kn']] = res
            print('[%d/%d] %s: %d neurons, %d trials, %d timepoints'
                  % (i + 1, len(jobs), res['kn'], res['n_neurons'], len(res['neural']),
                     sum(x.shape[1] for x in res['neural'])), flush=True)

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': [], 'subject_idx': [],
        'brain_regions': BRAIN_REGIONS, 'brain_region_idx': [],
        'input_names': INPUT_NAMES, 'output_names': OUTPUT_NAMES,
        'output_values': [STIM_VALUES, LICK_VALUES, POSITION_VALUES, SPEED_VALUES],
        'metadata': {},
    }
    subjects = sorted({s['mname'] for s in sessions.values()})
    session_info = []
    for kn, s in sessions.items():
        r = results[kn]
        data['neural'].append(r['neural'])
        data['input'].append(r['input'])
        data['output'].append(r['output'])
        data['subject_idx'].append(subjects.index(s['mname']))
        data['brain_region_idx'].append(r['brain_region_idx'])
        stim_counts = Counter(STIM_VALUES[int(o[0, 0])] for o in r['output'])
        session_info.append({
            'session': kn, 'subject': s['mname'], 'date': s['datexp'], 'block': s['blk'],
            'experiment_types': s['exp_types'],
            'day_of_training': day_of_training[kn],
            'n_trials': len(r['neural']),
            'n_timepoints': int(sum(x.shape[1] for x in r['neural'])),
            'n_neurons': r['n_neurons'],
            'n_neurons_recorded': r['n_neurons_total'],
            'n_neurons_in_visual_areas': r['n_neurons_in_areas'],
            'wall_name_to_stimulus': {w: STIM_VALUES[i] for w, i in r['stim_map'].items()},
            'stimulus_trial_counts': dict(stim_counts),
            'rewarded_trials': int(sum(int(x[3, 0]) for x in r['input'])),
            'reward_mode': str(beh[kn]['Reward_Mode']),
        })
    data['subjects'] = subjects
    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)

    data['metadata'] = {
        'task_description':
            'Head-fixed mice run through 4 m long virtual-reality corridors whose walls '
            'are covered with naturalistic texture patterns (leaf/circle, or rock/brick '
            'for some mice). A sound cue occurs at a random position in each corridor; '
            'for the trained ("task") mice licking after the cue in the rewarded '
            'corridor delivers water, while the unsupervised and naive cohorts run '
            'through the same corridors without reward. Neurons in V1 and higher visual '
            'areas were recorded with a two-photon mesoscope. The decoder predicts, from '
            'deconvolved calcium activity plus task context, (0) the visual stimulus '
            'shown in the corridor, (1) whether the mouse is licking, (2) which 1 m bin '
            'of the corridor it is in, and (3) which quartile of the running-speed '
            'distribution its speed falls in.',
        'time_bin_size': float(time_bin_size),
        'temporal_alignment_event':
            'trial start = entry into the textured virtual-reality corridor '
            '(beh["Trial_start_time"] / "StartFr")',
        'off_start': 0.0,
        'off_end': None,
        'trial_end_event':
            'exit from the textured corridor into the 2 m grey space (beh["GrayFr"]); '
            'trial duration is behaviour-dependent so off_end varies per trial',
        'neural_signal': 'Suite2p deconvolved fluorescence (non-negative deconvolution, '
                         '0.75 s decay), one value per imaging frame, no normalisation',
        'frame_selection':
            'only frames inside the textured corridor (ft_CorrSpc) during which the '
            'virtual reality was advancing, i.e. the mouse was running (ft_move > 0); '
            'this is the frame selection used throughout the reference code '
            '(fr_valid = VRmove & isCorridor) and in the paper',
        'neuron_selection':
            'neurons assigned to one of the four visual-area groups analysed in the '
            'paper (V1, mHV, lHV, aHV; utils.neu_area_ID), randomly subsampled to at '
            'most %s per session' % (max_neurons if max_neurons else 'all'),
        'max_neurons_per_session': max_neurons,
        'neuron_subsample_seed': args.seed,
        'input_descriptions': {
            'time_to_sound_cue': 'seconds until the sound cue of this trial '
                                 '(positive before the cue, negative after it)',
            'day_of_training': 'days between this recording and the first recording of '
                               'the same mouse',
            'time_since_trial_start': 'seconds since entry into the corridor (real time, '
                                      'including periods when the mouse was not running)',
            'reward_available': '1 if this corridor is the rewarded one for this mouse '
                                '(beh["isRew"]), 0 otherwise; always 0 for the '
                                'unsupervised and naive cohorts',
        },
        'output_descriptions': {
            'stimulus': 'canonical stimulus identity of the corridor (paper stim_id; '
                        'rock/brick/wood stimuli are mapped onto the equivalent '
                        'identities by the authors\' stim_id tables, see '
                        'session_info["wall_name_to_stimulus"])',
            'lick': 'at least one lick was detected in this imaging frame',
            'position_bin': 'position along the 4 m textured corridor in 1 m bins',
            'speed_bin': 'quartile of the running speed (ft_RunSpeed), quartiles '
                         'computed over all timepoints in the dataset',
        },
        'speed_bin_edges_cm_per_s': [float(x) for x in speed_edges],
        'frame_rate_hz': float(np.median(frame_rates)),
        'corridor_length_m': 4.0,
        'session_info': session_info,
        'source': 'Zhong et al. 2025, Unsupervised pretraining in biological neural '
                  'networks (doi:10.25378/janelia.28811129.v1)',
    }

    ntrials = sum(len(x) for x in data['neural'])
    ntime = sum(x.shape[1] for s in data['neural'] for x in s)
    print('sessions %d, trials %d, timepoints %d, neurons %d'
          % (len(data['neural']), ntrials, ntime,
             sum(len(x) for x in data['brain_region_idx'])), flush=True)

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('wrote %s (%.2f GB)' % (args.out, os.path.getsize(args.out) / 1e9))


if __name__ == '__main__':
    main()
