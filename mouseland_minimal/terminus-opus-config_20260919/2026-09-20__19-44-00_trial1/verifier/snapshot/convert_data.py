"""Convert Zhong et al. 2025 virtual-reality imaging data to decoder format.

Data/processing decisions follow the paper and /app/code/utils.py:
  * Sessions: the 89 unique two-photon recordings (mouse_date_block).  The Beh_*.npy
    files contain several *views* of the same recording (e.g. '..._swap1'/'..._swap2'
    and re-listing under several experiment types); these are deduplicated so each
    recording contributes exactly one session, matching the 89 spk files 1:1.
  * Stimulus label: the canonical role from UniqWalls + stim_id
    (0 circle1, 1 circle2, 2 leaf1, 3 leaf2, 4 leaf3, 5 leaf1_swap1, 6 leaf1_swap2).
    Physical texture names differ between mice (leaf/circle, rock/brick, wood), and
    for some mice the families are swapped, so the role is the cross-mouse label.
    TrialStim is NOT used: in a swap view it contains the placeholder
    'stimulus_of_trial' for trials outside that view's stimulus set.
  * Neurons: Suite2p deconvolved traces, restricted to neurons with a retinotopic
    area assignment (V1/mHV/lHV/aHV as in utils.neu_area_ID; iarea -1 and 7 are
    unassigned and are excluded, as in the paper's analyses), z-scored per neuron
    across the whole recording (as utils.get_kfold_reward_response does), then a
    random subset of at most MAX_NEURONS is kept per session because the full
    dataset is 405 GB.
  * Timepoints: only frames while the mouse was running, i.e. the virtual reality
    was moving (ft_move > 0, the VRmove filter of data_process_script.ipynb).  The
    paper: "We only considered timepoints during running for analysis".
  * Trials: aligned to corridor entry and restricted to the 4 m textured corridor
    (ft_CorrSpc, position 0-40 dm); the following 2 m of grey space is excluded.
    This is the window the 4 x 1 m position bins of the decoder task describe.
"""

import os
import glob
import pickle
import datetime
import collections
import argparse
import numpy as np

ROOT = '/app/data'
OUT = '/app/converted_data.pkl'

STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
AREA_NAMES = ['V1', 'mHV', 'lHV', 'aHV']
# utils.neu_area_ID: V1 = 8; mHV = 0,1,2,9; lHV = 5,6; aHV = 3,4
IAREA_TO_AREA = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}

MAX_NEURONS = 2000      # per session; the decoder's SVD init also caps at 2000
MIN_FRAMES_PER_TRIAL = 5
CORRIDOR_BINS = 4       # 4 x 1 m bins over the 4 m textured corridor
SEED = 0


def collect_recordings():
    """Deduplicate the Beh_*.npy views into one entry per recording."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'beh', 'Beh_*.npy'))):
        exp = os.path.basename(f)[4:-4]
        B = np.load(f, allow_pickle=True).item()
        for key, beh in B.items():
            parts = key.split('_')
            base = '_'.join(parts[:5]) if parts[-1].startswith('swap') else key
            d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
            d['exps'].append(exp)
            # merge the wall -> canonical role maps of all views of this recording
            for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
                if not np.isnan(sid):
                    d['walls'][str(wall)] = int(sid)
            if d['beh'] is None:
                d['beh'] = beh
        print('  read', os.path.basename(f), flush=True)
    return recs


def trial_frames(beh, nfr):
    """Frames of each trial: running, inside the textured corridor."""
    tr = beh['ft_trInd'][:nfr]
    corr = beh['ft_CorrSpc'][:nfr].astype(bool)
    moving = beh['ft_move'][:nfr] > 0
    ok = corr & moving & ~np.isnan(tr)
    order = np.argsort(tr[ok], kind='stable')
    idx = np.where(ok)[0][order]
    groups = collections.defaultdict(list)
    for i in idx:
        groups[int(tr[i])].append(i)
    return {t: np.sort(np.array(v)) for t, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', type=str, default=OUT)
    args = ap.parse_args()

    print('collecting behaviour ...', flush=True)
    recs = collect_recordings()
    names = sorted(recs)
    print('unique recordings:', len(names), flush=True)

    # ---- pass 1: behaviour only, to get global running-speed quartiles ----
    speeds = []
    for name in names:
        beh = recs[name]['beh']
        nfr = len(beh['ft'])
        frames = trial_frames(beh, nfr)
        spd = beh['ft_RunSpeed'][:nfr]
        for t, fr in frames.items():
            if len(fr) >= MIN_FRAMES_PER_TRIAL and str(beh['WallName'][t]) in recs[name]['walls']:
                speeds.append(spd[fr])
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed quartile edges (cm/s):', speed_edges.round(3), flush=True)

    # day of training: days elapsed since each mouse's first recording
    def date_of(name):
        p = name.split('_')
        return datetime.date(int(p[1]), int(p[2]), int(p[3]))
    first = {}
    for name in names:
        m = name.split('_')[0]
        first[m] = min(first.get(m, date_of(name)), date_of(name))

    data = {'neural': [], 'input': [], 'output': [],
            'subjects': [], 'subject_idx': [], 'brain_regions': AREA_NAMES,
            'brain_region_idx': [],
            'input_names': ['time_to_sound_cue', 'day_of_training',
                            'time_since_trial_start', 'reward_available'],
            'output_names': ['stimulus', 'licking', 'position_bin', 'speed_bin'],
            'output_values': [STIM_NAMES,
                              ['no_lick', 'lick'],
                              ['0-1m', '1-2m', '2-3m', '3-4m'],
                              ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest']],
            'metadata': {}}
    subjects = []
    session_info = []
    if args.limit:
        names = names[:args.limit]

    bin_ms = []
    for si, name in enumerate(names):
        beh = recs[name]['walls'], recs[name]['beh']
        walls, beh = beh
        mouse = name.split('_')[0]
        date = '_'.join(name.split('_')[1:4])
        blk = name.split('_')[4]

        # ---- neurons: retinotopy -> area, then subsample ----
        ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mouse, date)),
                      allow_pickle=True)
        iarea = ret['iarea']
        area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
        valid = np.where(area >= 0)[0]
        rng = np.random.default_rng(SEED + si)
        if len(valid) > MAX_NEURONS:
            sel = np.sort(rng.choice(valid, MAX_NEURONS, replace=False))
        else:
            sel = valid

        spkfile = os.path.join(ROOT, 'spk', '%s_neural_data.npy' % name)
        blocks = np.load(spkfile, allow_pickle=True).item()['spks']
        counts = [b.shape[0] for b in blocks]
        assert sum(counts) == len(iarea), (name, sum(counts), len(iarea))
        nfr = blocks[0].shape[1]
        offs = np.concatenate([[0], np.cumsum(counts)])
        spk = np.empty((len(sel), nfr), dtype=np.float32)
        for bi, blk_arr in enumerate(blocks):
            m = (sel >= offs[bi]) & (sel < offs[bi + 1])
            if m.any():
                spk[m] = blk_arr[sel[m] - offs[bi]]
        del blocks
        # z-score each neuron over the whole recording (as in utils.py)
        mu = spk.mean(axis=1, keepdims=True)
        sd = spk.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        spk = (spk - mu) / sd

        # ---- behaviour ----
        frames = trial_frames(beh, nfr)
        ft = beh['ft'][:nfr] * 86400.0
        bin_s = float(np.median(np.diff(ft)))
        pos = beh['ft_Pos'][:nfr]
        spd = beh['ft_RunSpeed'][:nfr]
        lick = np.zeros(nfr, dtype=bool)
        lf = np.floor(beh['LickFr']).astype(int)
        lf = lf[(lf >= 0) & (lf < nfr)]
        lick[lf] = True
        day = float((date_of(name) - first[mouse]).days)

        neural_s, input_s, output_s = [], [], []
        for t in sorted(frames):
            fr = frames[t]
            wall = str(beh['WallName'][t])
            if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
                continue    # unassigned stimulus (circle3) or too little running
            neural_s.append(spk[:, fr].astype(np.float32))
            # Times are measured on the retained (running) frames, i.e. in
            # accumulated running time since corridor entry.  Only running frames
            # are kept (see module docstring), so this is the clock of the neural
            # sequence the decoder actually sees; wall-clock time would contain
            # large invisible jumps whenever the mouse stopped inside the corridor
            # (pauses of up to ~30 min occur), which are not represented in the data.
            k = np.arange(len(fr), dtype=np.float32) * bin_s
            rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
            inp = np.empty((4, len(fr)), dtype=np.float32)
            inp[0] = k - rank_cue
            inp[1] = day
            inp[2] = k
            inp[3] = float(beh['isRew'][t])
            input_s.append(inp)
            out = np.empty((4, len(fr)), dtype=np.int64)
            out[0] = walls[wall]
            out[1] = lick[fr].astype(np.int64)
            out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
            out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
            output_s.append(out)
        bin_ms.append(np.median(np.diff(ft)) * 1000.0)

        if mouse not in subjects:
            subjects.append(mouse)
        data['neural'].append(neural_s)
        data['input'].append(input_s)
        data['output'].append(output_s)
        data['subject_idx'].append(subjects.index(mouse))
        data['brain_region_idx'].append(area[sel].astype(np.int64))
        session_info.append({'session': name, 'mouse': mouse, 'date': date, 'block': blk,
                             'experiment_types': sorted(set(recs[name]['exps'])),
                             'day_of_training': day, 'n_trials': len(neural_s),
                             'n_neurons': len(sel),
                             'n_neurons_recorded': int(len(iarea))})
        print('[%d/%d] %s  trials=%d neurons=%d' % (si + 1, len(names), name,
              len(neural_s), len(sel)), flush=True)
        del spk

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description':
            'Head-fixed mice ran through 4 m virtual-reality corridors whose walls '
            'were one of several naturalistic texture patterns; a sound cue occurred '
            'at a random position in the corridor and, in task mice, predicted water '
            'reward in the rewarded corridor. From two-photon population activity '
            '(Suite2p deconvolved, z-scored) the decoder predicts the visual stimulus '
            'category of the corridor, whether the mouse licked, its position in the '
            'corridor (4 x 1 m bins) and its running speed (quartile bins).',
        'time_bin_size': float(np.median(bin_ms)),
        'temporal_alignment_event': 'trial start = entry into the virtual corridor',
        'off_start': 0.0,
        'off_end': None,
        'trial_window':
            'corridor entry to the end of the 4 m textured corridor; only frames '
            'while the mouse was running (virtual reality moving) are kept, so trials '
            'have variable numbers of time bins (median 21)',
        'neural_data': 'Suite2p deconvolved calcium traces (0.75 s decay), z-scored '
                       'per neuron over the whole recording',
        'neuron_subsampling':
            'neurons with a retinotopic area assignment (V1/mHV/lHV/aHV); at most '
            '%d randomly chosen per session (full dataset is 405 GB)' % MAX_NEURONS,
        'speed_bin_edges_cm_per_s': speed_edges.tolist(),
        'time_convention':
            'time_since_trial_start and time_to_sound_cue are accumulated '
            'running time (retained frames x frame period) relative to corridor '
            'entry and to the sound cue; periods when the mouse stopped are '
            'excluded from the data and therefore from the clock',
        'excluded': 'trials whose wall texture had no canonical stimulus role '
                    '(circle3) and trials with fewer than %d running frames'
                    % MIN_FRAMES_PER_TRIAL,
        'session_info': session_info,
    }

    ntr = sum(len(s) for s in data['neural'])
    print('sessions %d, trials %d' % (len(data['neural']), ntr))
    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('saved', args.out)


if __name__ == '__main__':
    main()
