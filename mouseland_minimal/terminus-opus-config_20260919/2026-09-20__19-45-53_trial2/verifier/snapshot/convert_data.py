"""
Convert data from Zhong et al. 2025 ("Unsupervised pretraining in biological neural
networks") into the decoder dataset format.

Experiment (see /app/methods.txt and the paper): head-fixed mice run through 4 m long
virtual-reality corridors whose walls are covered with one of several 'frozen' texture
patterns (leaf/circle/rock/brick/wood variants).  Corridors are separated by 2 m of grey
space.  A sound cue is played at a random position inside every corridor; in the task
(supervised) cohort licking after the cue in the rewarded corridor delivers water.
Neural activity is two-photon mesoscope calcium imaging of visual cortex (V1 + higher
visual areas), deconvolved with Suite2p (fs ~ 3.18 Hz).

Conversion decisions (kept as close to the reference paper/code as possible):

* Sessions          : all 89 imaging recordings listed in beh/Imaging_Exp_info.npy
                      (19 mice, task / unsupervised / naive / grating cohorts).
* Trial             : one corridor traversal, aligned to corridor entry
                      (beh['StartFr'] / beh['Trial_start_time']), ending when the animal
                      leaves the 4 m texture area (beh['ft_CorrSpc']).
* Timepoints        : the native imaging frames (no re-binning), restricted to frames in
                      which the virtual reality was moving, i.e. the animal was running
                      (beh['ft_move']>0).  The paper states: "We only considered
                      timepoints during running for analysis, which removed time periods
                      when the task mice stopped to collect water rewards"; the reference
                      code applies exactly this mask (VRmove = beh['ft_move']>0).
* Neurons           : deconvolved traces (utils.load_spk concatenates the imaging planes).
                      Only neurons that the retinotopic maps assign to one of the four
                      area groups used in the paper (V1, mHV, lHV, aHV; utils.neu_area_ID)
                      are kept, and a fixed random subset of NNEURONS neurons per session
                      is stored to keep the dataset tractable (recordings contain
                      20,547-89,577 neurons).  Each neuron is z-scored over the whole
                      recording, as done in the reference code (utils.get_kfold_reward_response).
* Trial curation    : trials whose wall texture has no canonical stimulus id in
                      Imaging_Exp_info.npy (a third 'circle3' texture shown in 4 sessions)
                      are dropped, because their stimulus category is undefined.

Outputs are the categorical variables requested by the decoder task.
"""

import os
import pickle
import datetime
import numpy as np

ROOT = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

NNEURONS = 1000          # neurons kept per session (random subset, fixed seed)
SEED = 0
POS_BIN_EDGES = [10., 20., 30.]      # corridor position bins (dm) -> 4 x 1 m bins
CORRIDOR_TEXTURE_LEN = 40.           # dm (4 m)

# canonical stimulus ids used throughout the paper's code
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
# area groups, as in utils.neu_area_ID
AREA_GROUPS = {'V1': [8], 'mHV': [0, 1, 2, 9], 'lHV': [5, 6], 'aHV': [3, 4]}
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']


def load_behavior():
    """Collect the behavioural variables of every unique recording.

    Recordings can appear in several 'experiment types' (e.g. a session used both as
    'unsup_test1' and 'unsup_train2_before_learning', or with the 'swap1'/'swap2' stimulus
    subsets); the behaviour is identical, only the subset of stimuli that is analysed
    differs.  We therefore keep each recording once and merge the wall-name -> canonical
    stimulus-id maps of all experiment types it belongs to.
    """
    exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    recs = {}
    for exp_type, db in exp_info.items():
        Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                      allow_pickle=True).item()
        for ndb in db:
            rid = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
            key = rid + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
            beh = Beh[key]
            if rid not in recs:
                recs[rid] = dict(
                    rid=rid, mname=ndb['mname'], datexp=ndb['datexp'], blk=ndb['blk'],
                    exp_types=[], stim_map={}, cohorts=set(),
                    ntrials=int(beh['ntrials']),
                    ft=np.asarray(beh['ft'], dtype=np.float64),
                    ft_trInd=np.asarray(beh['ft_trInd'], dtype=np.float64),
                    ft_CorrSpc=np.asarray(beh['ft_CorrSpc']).astype(bool),
                    ft_move=np.asarray(beh['ft_move'], dtype=np.float64),
                    ft_Pos=np.asarray(beh['ft_Pos'], dtype=np.float64),
                    ft_RunSpeed=np.asarray(beh['ft_RunSpeed'], dtype=np.float64),
                    LickFr=np.asarray(beh['LickFr'], dtype=np.float64),
                    WallName=np.array([str(w) for w in beh['WallName']]),
                    isRew=np.asarray(beh['isRew']).astype(bool),
                    SoundTime=np.asarray(beh['SoundTime'], dtype=np.float64),
                    Trial_start_time=np.asarray(beh['Trial_start_time'], dtype=np.float64),
                )
            r = recs[rid]
            r['exp_types'].append(exp_type)
            r['cohorts'].add(str(ndb.get('exptype', 'None')))
            if 'grating' in exp_type:
                r['cohorts'].add('VRgrating')
            for wall, sid in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], dtype=float)):
                if not np.isnan(sid):
                    r['stim_map'][str(wall)] = int(sid)
        del Beh
        print('loaded behavior:', exp_type, flush=True)
    return recs


def trial_frames(rec, nfr_spk):
    """Frame indices of every trial: running frames inside the texture corridor."""
    nfr = min(len(rec['ft']), nfr_spk)
    keep = rec['ft_CorrSpc'][:nfr] & (rec['ft_move'][:nfr] > 0)
    tr = rec['ft_trInd'][:nfr]
    idx = np.where(keep)[0]
    tr_idx = tr[idx]
    return [idx[tr_idx == i] for i in range(rec['ntrials'])], nfr


def main():
    rng = np.random.default_rng(SEED)
    recs = load_behavior()
    rids = sorted(recs.keys(), key=lambda r: (recs[r]['mname'], recs[r]['datexp'], recs[r]['blk']))

    # day of training: days elapsed since the first recording day of that mouse
    first_day = {}
    for rid in rids:
        r = recs[rid]
        d = datetime.date(*[int(x) for x in r['datexp'].split('_')])
        r['date'] = d
        if r['mname'] not in first_day or d < first_day[r['mname']]:
            first_day[r['mname']] = d
    for rid in rids:
        r = recs[rid]
        r['day'] = float((r['date'] - first_day[r['mname']]).days)

    # ---- global running-speed quartiles (4 bins with 25% of the data each) ----
    speeds = []
    for rid in rids:
        r = recs[rid]
        frames, _ = trial_frames(r, len(r['ft']))
        ii = np.concatenate([f for f in frames if len(f)])
        speeds.append(r['ft_RunSpeed'][ii])
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed bin edges (cm/s):', speed_edges, flush=True)

    subjects = sorted(set(recs[r]['mname'] for r in rids))
    data = dict(neural=[], input=[], output=[], subjects=subjects,
                subject_idx=[], brain_regions=BRAIN_REGIONS, brain_region_idx=[],
                input_names=['time_to_sound_cue', 'day_of_training',
                             'time_since_trial_start', 'reward_availability'],
                output_names=['stimulus_category', 'licking', 'position_bin', 'speed_bin'],
                output_values=[STIM_NAMES,
                               ['no_lick', 'lick'],
                               ['0-1m', '1-2m', '2-3m', '3-4m'],
                               ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']],
                metadata={})
    session_info = []
    dts = []
    n_dropped_stim = 0

    for si, rid in enumerate(rids):
        r = recs[rid]
        spk_file = os.path.join(ROOT, 'spk', '%s_neural_data.npy' % rid)
        spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
        nneu_all, nfr_spk = spk.shape

        ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                                   % (r['mname'], r['datexp'])), allow_pickle=True)
        iarea = np.asarray(ret['iarea'], dtype=float)
        assert len(iarea) == nneu_all, (rid, len(iarea), nneu_all)

        # keep neurons assigned to one of the four area groups analysed in the paper
        region_of_neuron = np.full(nneu_all, -1, dtype=int)
        for ri, reg in enumerate(BRAIN_REGIONS):
            m = np.isin(iarea, AREA_GROUPS[reg])
            region_of_neuron[m] = ri
        valid = np.where(region_of_neuron >= 0)[0]
        sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))

        X = spk[sel].astype(np.float32)
        del spk
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        good = (sd[:, 0] > 0)
        X = (X[good] - mu[good]) / sd[good]          # z-score over the whole recording
        sel = sel[good]
        region_idx = region_of_neuron[sel].astype(int)

        frames, nfr = trial_frames(r, nfr_spk)
        dts.append(np.median(np.diff(r['ft'][:nfr])) * 86400.)

        lick = np.zeros(nfr, dtype=bool)
        lf = np.round(r['LickFr']).astype(int)
        lick[lf[(lf >= 0) & (lf < nfr)]] = True

        neural_s, input_s, output_s = [], [], []
        for ti in range(r['ntrials']):
            ii = frames[ti]
            if len(ii) == 0:
                continue
            sid = r['stim_map'].get(r['WallName'][ti], None)
            if sid is None:                     # undefined stimulus category
                n_dropped_stim += 1
                continue
            t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.   # s since corridor entry
            t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.      # s until sound cue
            inp = np.stack([t_cue,
                            np.full(len(ii), r['day']),
                            t,
                            np.full(len(ii), float(r['isRew'][ti]))]).astype(np.float32)
            pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
            pos_bin = np.clip(pos_bin, 0, 3)
            spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
            out = np.stack([np.full(len(ii), sid),
                            lick[ii].astype(int),
                            pos_bin,
                            spd_bin]).astype(np.int64)
            neural_s.append(np.ascontiguousarray(X[:, ii]))
            input_s.append(inp)
            output_s.append(out)
        del X

        data['neural'].append(neural_s)
        data['input'].append(input_s)
        data['output'].append(output_s)
        data['brain_region_idx'].append(region_idx)
        data['subject_idx'].append(subjects.index(r['mname']))
        session_info.append(dict(session_id=rid, mouse=r['mname'], date=r['datexp'],
                                 block=r['blk'], experiment_types=sorted(set(r['exp_types'])),
                                 cohort=sorted(c for c in r['cohorts'] if c != 'None'),
                                 day_of_training=r['day'], n_trials=len(neural_s),
                                 n_neurons_recorded=int(nneu_all),
                                 n_neurons_saved=int(len(sel))))
        print('[%d/%d] %s: %d neurons (of %d), %d trials'
              % (si + 1, len(rids), rid, len(sel), nneu_all, len(neural_s)), flush=True)

    data['subject_idx'] = np.array(data['subject_idx'], dtype=int)
    data['metadata'] = dict(
        task_description=(
            'Mice run through 4 m long virtual-reality corridors with naturalistic texture '
            'patterns on the walls (leaf/circle/rock/brick/wood variants, denoted by the '
            'canonical names circle1, circle2, leaf1, leaf2, leaf3 and spatially shuffled '
            'leaf1_swap1/2 used in the paper).  A sound cue is played at a random position '
            'in every corridor; in the task cohort, licking after the cue in the rewarded '
            'corridor triggers water.  From visual-cortex population activity we decode the '
            'visual stimulus category of the corridor, whether the animal is licking, the '
            'position of the animal in the corridor (4 x 1 m bins) and its running speed '
            '(4 bins containing 25% of the timepoints each).  Decoder inputs are the signed '
            'time to the sound cue, the time since corridor entry, the day of training and '
            'whether the corridor is the rewarded one.'),
        time_bin_size=float(np.mean(dts) * 1000.),
        temporal_alignment_event='trial start = entry into the virtual-reality corridor',
        off_start=0.0,
        off_end=None,
        trial_definition=('all two-photon frames from corridor entry until the animal leaves '
                          'the 4 m texture corridor, keeping only frames in which the animal '
                          'was running (virtual reality moving, beh["ft_move"]>0), as in the '
                          'paper; trial length is therefore variable (median ~21 frames)'),
        neural_data=('Suite2p deconvolved calcium traces (0.75 s decay), z-scored per neuron '
                     'over the whole recording; %d randomly chosen neurons per session out of '
                     'the 20,547-89,577 recorded, restricted to neurons assigned to V1/mHV/'
                     'lHV/aHV by the retinotopic maps' % NNEURONS),
        sampling_rate_hz=float(1000. / (np.mean(dts) * 1000.)),
        speed_bin_edges_cm_s=[float(x) for x in speed_edges],
        position_bin_edges_m=[1.0, 2.0, 3.0],
        licking_note=('licks were only recorded in the 28 water-restricted task sessions; '
                      'mice of the unsupervised, naive and grating cohorts were not water '
                      'restricted, received no reward and have no licks (licking = 0)'),
        n_trials_dropped_unknown_stimulus=int(n_dropped_stim),
        session_info=session_info,
        source='Zhong et al. 2025, Unsupervised pretraining in biological neural networks',
    )

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('saved', OUT_FILE)


if __name__ == '__main__':
    main()
