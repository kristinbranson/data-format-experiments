"""
Convert the data from Zhong et al., "Unsupervised pretraining in biological neural
networks" into the decoder dataset format.

Experiment (see /app/methods.txt and the paper): head-fixed mice run through 4-m-long
virtual-reality corridors whose walls are covered with one of several 'frozen' natural
texture patterns.  Corridors are separated by 2 m of grey space.  A sound cue is played
at a random position inside every corridor; for the task ('sup') mice the cue signals
the start of the reward zone in the rewarded corridor only.  Neural activity of
20k-90k neurons across visual cortical areas was recorded with a two-photon mesoscope
(~3.17 Hz) and deconvolved with suite2p.

Processing decisions (following the paper / its code in /app/code/utils.py):
  * Trials are corridor traversals, temporally aligned to corridor entry.  Only the
    4-m textured part of the corridor is kept (beh['ft_CorrSpc']), which is exactly
    the part that the decoder's position output refers to.
  * Only frames in which the animal was running are kept (beh['ft_move'] > 0, i.e. the
    VR was moving): "We only considered timepoints during running for analysis"
    (methods.txt); the same mask (VRmove & ft_CorrSpc) is used throughout utils.py.
  * Neural data are the raw deconvolved traces ('spks'), one time bin per imaging
    frame (~315 ms).  Planes are concatenated as in utils.load_spk.
  * Neurons are kept only if the retinotopy file assigns them to one of the four
    visual-area groups used in the paper (V1, mHV, lHV, aHV; utils.neu_area_ID).
    For tractability a random subset of at most NNEURONS neurons per session is kept
    (the raw data are 405 GB).
  * Stimulus labels use the canonical/functional names of the paper
    ('circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2'), which
    are stored per trial in beh['TrialStim'].  This is essential because the physical
    textures differ between mice (leaf/circle vs rock/brick...) while their role in
    the experiment is the same; the decoder is shared across sessions.
"""

import os
import pickle
import datetime
import argparse
import numpy as np

ROOT = '/app/data'
OUT = '/app/converted_data.pkl'
SEED = 0
NNEURONS = 1000            # max neurons kept per session
SEC_PER_DAY = 86400.0      # beh times are MATLAB datenums (days)

# canonical stimulus names, in the order used by the paper (stim_id 0..6)
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
AREA_NAMES = ['V1', 'mHV', 'lHV', 'aHV']


def area_index(iarea):
    """Map suite2p/retinotopy area ids onto the 4 area groups of utils.neu_area_ID."""
    out = -np.ones(len(iarea), dtype=int)
    out[iarea == 8] = 0                                   # V1
    out[np.isin(iarea, [0, 1, 2, 9])] = 1                  # medial HV
    out[np.isin(iarea, [5, 6])] = 2                        # lateral HV
    out[np.isin(iarea, [3, 4])] = 3                        # anterior HV
    return out


def load_sessions():
    """Return {session_key: {'beh','db','stim_map'}} for the 89 imaging sessions.

    The same recording appears in several Beh_<exptype>.npy files (once per analysis
    it is used for); the behaviour is identical in all of them (verified), but each
    file only names the stimuli that belong to that analysis, so the canonical names
    are collected across all files.
    """
    exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    sessions = {}
    for exp_type in exp_info:
        Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                      allow_pickle=True).item()
        for db in exp_info[exp_type]:
            key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
            bkey = key + ('_' + db['stimtype'] if 'stimtype' in db else '')
            beh = Beh[bkey]
            s = sessions.setdefault(key, {'beh': beh, 'db': db, 'stim_map': {},
                                          'exp_types': []})
            s['exp_types'].append(exp_type)
            # wall name -> canonical stimulus name
            for w, c in zip(beh['WallName'], beh['TrialStim']):
                if str(c) != 'stimulus_of_trial':
                    s['stim_map'][str(w)] = str(c)
    return sessions


def trial_frames(beh, nfr):
    """Indices of the analysed frames of every trial (running, inside the texture)."""
    ft_tr = beh['ft_trInd'][:nfr]
    keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
    idx = np.where(keep)[0]
    tr = ft_tr[idx]
    ok = ~np.isnan(tr)
    idx, tr = idx[ok], tr[ok].astype(int)
    order = np.argsort(tr, kind='stable')
    idx, tr = idx[order], tr[order]
    bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
    return [idx[bounds[t]:bounds[t + 1]] for t in range(int(beh['ntrials']))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsessions', type=int, default=None)
    ap.add_argument('--nneurons', type=int, default=NNEURONS)
    ap.add_argument('--out', type=str, default=OUT)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    sessions = load_sessions()
    keys = list(sessions.keys())
    if args.nsessions is not None:
        keys = keys[:args.nsessions]
    print('%d sessions' % len(keys))

    # ---- pass 1 (behaviour only): global running-speed quartiles, day of training
    speeds = []
    first_date = {}
    for k in keys:
        beh, db = sessions[k]['beh'], sessions[k]['db']
        nfr = len(beh['ft'])
        keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
        speeds.append(beh['ft_RunSpeed'][:nfr][keep])
        d = datetime.date(*map(int, db['datexp'].split('_')))
        if db['mname'] not in first_date or d < first_date[db['mname']]:
            first_date[db['mname']] = d
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed quartile edges (cm/s):', speed_edges)

    subjects = sorted(set(sessions[k]['db']['mname'] for k in keys))

    data = {'neural': [], 'input': [], 'output': [], 'subjects': subjects,
            'subject_idx': [], 'brain_regions': AREA_NAMES, 'brain_region_idx': [],
            'input_names': ['time_to_sound_cue', 'day_of_training',
                            'time_since_trial_start', 'reward_available'],
            'output_names': ['stimulus', 'lick', 'position_bin', 'speed_bin'],
            'output_values': [STIM_NAMES,
                              ['no_lick', 'lick'],
                              ['0-1m', '1-2m', '2-3m', '3-4m'],
                              ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']],
            'metadata': {}}
    session_info = []
    dts = []

    for si, k in enumerate(keys):
        beh, db = sessions[k]['beh'], sessions[k]['db']
        smap = sessions[k]['stim_map']

        # ---- neurons: area labels from the retinotopy file
        ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                                   % (db['mname'], db['datexp'])), allow_pickle=True)
        aidx = area_index(ret['iarea'])

        spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
                       allow_pickle=True).item()['spks']
        nneu_all = sum(s.shape[0] for s in spks)
        assert nneu_all == len(aidx), (k, nneu_all, len(aidx))

        valid = np.where(aidx >= 0)[0]
        if len(valid) > args.nneurons:
            sel = np.sort(rng.choice(valid, args.nneurons, replace=False))
        else:
            sel = valid
        # gather the selected neurons plane by plane, then free the full arrays
        spk = np.empty((len(sel), spks[0].shape[1]), dtype=np.float32)
        off, o2 = 0, 0
        for p in range(len(spks)):
            n = spks[p].shape[0]
            take = sel[(sel >= off) & (sel < off + n)] - off
            spk[o2:o2 + len(take)] = spks[p][take]
            o2 += len(take)
            off += n
        del spks

        nfr = min(spk.shape[1], len(beh['ft']))
        ft = beh['ft'][:nfr]
        pos = beh['ft_Pos'][:nfr]
        speed = beh['ft_RunSpeed'][:nfr]
        lickfr = np.round(beh['LickFr']).astype(int) if len(beh['LickFr']) else np.zeros(0, int)
        lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
        is_lick = np.zeros(nfr, dtype=np.int64)
        is_lick[lickfr] = 1

        day = float((datetime.date(*map(int, db['datexp'].split('_')))
                     - first_date[db['mname']]).days)

        tidx = trial_frames(beh, nfr)
        neural_s, input_s, output_s = [], [], []
        nskip = 0
        for t, idx in enumerate(tidx):
            if len(idx) < 2:
                nskip += 1
                continue
            name = smap.get(str(beh['WallName'][t]), None)
            if name is None or name not in STIM_NAMES:
                nskip += 1
                continue
            T = len(idx)
            tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
            tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
            inp = np.empty((4, T), dtype=np.float32)
            inp[0] = tcue
            inp[1] = day
            inp[2] = tt
            inp[3] = float(beh['isRew'][t])

            out = np.empty((4, T), dtype=np.int64)
            out[0] = STIM_NAMES.index(name)
            out[1] = is_lick[idx]
            out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)  # 4 x 1 m bins
            out[3] = np.searchsorted(speed_edges, speed[idx])

            neural_s.append(np.ascontiguousarray(spk[:, idx]))
            input_s.append(inp)
            output_s.append(out)
        del spk

        dts.append(np.median(np.diff(ft)) * SEC_PER_DAY)
        data['neural'].append(neural_s)
        data['input'].append(input_s)
        data['output'].append(output_s)
        data['subject_idx'].append(subjects.index(db['mname']))
        data['brain_region_idx'].append(aidx[sel].astype(int))
        session_info.append({'session': k, 'mouse': db['mname'], 'date': db['datexp'],
                             'block': db['blk'],
                             'exp_types': sorted(set(sessions[k]['exp_types'])),
                             'cohort': str(db.get('exptype', 'naive')),
                             'reward_type': str(db.get('rewType', 'None')),
                             'day_of_training': day,
                             'ntrials': len(neural_s),
                             'nneurons': len(sel),
                             'nneurons_recorded': int(nneu_all)})
        print('[%d/%d] %s  neurons %d/%d  trials %d (skipped %d)'
              % (si + 1, len(keys), k, len(sel), nneu_all, len(neural_s), nskip),
              flush=True)

    data['subject_idx'] = np.array(data['subject_idx'], dtype=int)
    data['metadata'] = {
        'task_description':
            'Head-fixed mice run through 4-m virtual-reality corridors whose walls show '
            'one of several frozen natural-texture patterns, separated by 2 m of grey '
            'space.  A sound cue is played at a random position in every corridor; for '
            'the task (supervised) mice it marks the start of the reward zone in the '
            'rewarded corridor, where licking delivers water.  From the deconvolved '
            'two-photon activity of visual cortex we decode (1) the visual stimulus '
            'category of the corridor, (2) whether the animal licks, (3) its position '
            'in the corridor (4 x 1 m bins) and (4) its running speed (quartile bins).',
        'time_bin_size': float(np.median(dts) * 1000.0),
        'temporal_alignment_event': 'trial start = entry into the textured corridor',
        'off_start': 0.0,
        'off_end': None,
        'trial_definition':
            'one corridor traversal, from corridor entry until the animal leaves the '
            '4-m textured section and enters the grey space.  Trials have variable '
            'length; only frames in which the animal was running (the VR was moving, '
            'beh["ft_move"]>0) are kept, as in the paper.',
        'neural_data': 'suite2p deconvolved calcium activity (non-negative '
                       'deconvolution, 0.75 s decay), one bin per imaging frame',
        'sampling_rate_hz': float(1000.0 / (np.median(dts) * 1000.0)),
        'neuron_selection': 'neurons assigned to V1/mHV/lHV/aHV by the retinotopy '
                            'mapping; random subset of at most %d per session' % args.nneurons,
        'speed_bin_edges_cm_per_s': [float(x) for x in speed_edges],
        'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
        'input_descriptions': {
            'time_to_sound_cue': 'seconds until the sound cue (negative after the cue)',
            'day_of_training': 'days elapsed since the first imaging session of that mouse',
            'time_since_trial_start': 'seconds since entry into the corridor',
            'reward_available': '1 if this corridor is the rewarded one (task mice), else 0'},
        'output_descriptions': {
            'stimulus': 'canonical (functional) identity of the wall texture of the corridor',
            'lick': '1 if the animal licked during this time bin',
            'position_bin': 'position along the 4 m corridor in 4 x 1 m bins',
            'speed_bin': 'running-speed quartile (edges from all analysed frames)'},
        'session_info': session_info,
        'source': 'Zhong et al., Unsupervised pretraining in biological neural networks',
    }

    ntr = sum(len(s) for s in data['neural'])
    nt = sum(x.shape[1] for s in data['neural'] for x in s)
    print('sessions %d, trials %d, timepoints %d' % (len(data['neural']), ntr, nt))
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('saved', args.out)


if __name__ == '__main__':
    main()
