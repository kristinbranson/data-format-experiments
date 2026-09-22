"""Convert Zhong et al. 2025 VR-corridor imaging data to decoder format.

Usage: python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference code in /app/code (utils.py, data_process_script.ipynb):
  neural      : deconvolved Suite2p traces, planes concatenated (utils.load_spk)
  neurons     : iarea in {-1,7} excluded (outside visual cortex); V1/mHV/lHV/aHV per utils.neu_area_ID
  frames      : fr_valid = (ft_move>0) & ft_CorrSpc  (utils.Get_dprime_selective_neuron)
  alignment   : corridor entry (beh['Trial_start_time'] / beh['StartFr'])
  stim labels : canonical role-based ids from beh['stim_id'] / beh['UniqWalls']
"""

import argparse
import glob
import os
import pickle
import time
from collections import defaultdict
from datetime import date

import numpy as np

DATA_ROOT = '/app/data'
SPK_DIR = os.path.join(DATA_ROOT, 'spk')
BEH_DIR = os.path.join(DATA_ROOT, 'beh')
RET_DIR = os.path.join(DATA_ROOT, 'retinotopy')

CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']

AREA_NAMES = ['V1', 'mHV', 'lHV', 'aHV']
AREA_OF_IAREA = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}

N_NEURONS_PER_SESSION = 1000
MIN_FRAMES_PER_TRIAL = 5
RNG_SEED = 2025
POS_BIN_EDGES_DM = np.array([10.0, 20.0, 30.0])
SEC_PER_DAY = 24.0 * 3600.0


def session_key(mname, datexp, blk):
    return '%s_%s_%s' % (mname, datexp, blk)


def load_exp_info():
    """Unique sessions from beh/Imaging_Exp_info.npy -> dict key -> info dict."""
    exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    sessions = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = session_key(db['mname'], db['datexp'], db['blk'])
            rec = sessions.setdefault(key, {'mname': db['mname'], 'datexp': db['datexp'],
                                            'blk': db['blk'], 'exp_types': [],
                                            'exptype': None, 'rewType': db.get('rewType', None)})
            rec['exp_types'].append(exp_type)
            if db.get('exptype', None) is not None:
                rec['exptype'] = db['exptype']
    return sessions


def load_behaviour(sessions, verbose=True):
    """One pass over the 23 Beh_<exp_type>.npy files.

    Returns beh_by_session (key -> behaviour dict) and stim_map_by_session
    (key -> {wall name: canonical stimulus id}) merged over every experiment type in
    which the session appears (the _swap1/_swap2 duplicates label complementary subsets).
    """
    beh_by_session = {}
    stim_map = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(BEH_DIR, 'Beh_*.npy'))):
        t0 = time.time()
        B = np.load(f, allow_pickle=True).item()
        for k, beh in B.items():
            base = '_'.join(k.split('_')[:5])
            if base not in sessions:
                continue
            if base not in beh_by_session:
                beh_by_session[base] = beh
            sid = np.asarray(beh['stim_id'], dtype=float)
            for w, s in zip(list(beh['UniqWalls']), sid):
                if not np.isnan(s):
                    stim_map[base][str(w)] = int(s)
        if verbose:
            print('  loaded %-45s (%.1f s)' % (os.path.basename(f), time.time() - t0), flush=True)
    return beh_by_session, dict(stim_map)


def select_neurons(iarea, n_target, rng):
    """Visual-cortex neurons only, subsampled proportionally across the four areas."""
    area_idx = np.full(len(iarea), -1, dtype=np.int64)
    for ia, a in AREA_OF_IAREA.items():
        area_idx[iarea == ia] = a
    valid = np.where(area_idx >= 0)[0]
    if len(valid) <= n_target:
        return valid, area_idx[valid]
    counts = np.array([(area_idx[valid] == a).sum() for a in range(len(AREA_NAMES))])
    alloc = np.floor(counts / counts.sum() * n_target).astype(int)
    while alloc.sum() < n_target:
        alloc[np.argmax(counts - alloc)] += 1
    chosen = []
    for a in range(len(AREA_NAMES)):
        pool = valid[area_idx[valid] == a]
        if alloc[a] > 0:
            chosen.append(rng.choice(pool, size=min(alloc[a], len(pool)), replace=False))
    sel = np.sort(np.concatenate(chosen))
    return sel, area_idx[sel]


def load_spk_rows(key, rows):
    """Load only the requested neurons (rows of the concatenated matrix) of a session."""
    path = os.path.join(SPK_DIR, '%s_neural_data.npy' % key)
    planes = np.load(path, allow_pickle=True).item()['spks']
    sizes = np.array([p.shape[0] for p in planes])
    nfr = min(p.shape[1] for p in planes)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    out = np.empty((len(rows), nfr), dtype=np.float32)
    for i in range(len(planes)):
        m = (rows >= offsets[i]) & (rows < offsets[i + 1])
        if m.any():
            out[m] = planes[i][rows[m] - offsets[i], :nfr]
        planes[i] = None
    del planes
    return out, int(offsets[-1])


_BEH = None
_STIM_MAP = None
_SESSIONS = None


def _init_worker(beh, stim_map, sessions):
    global _BEH, _STIM_MAP, _SESSIONS
    _BEH, _STIM_MAP, _SESSIONS = beh, stim_map, sessions


def process_session(key):
    """Process one session; returns per-trial raw streams (neural + behaviour)."""
    t0 = time.time()
    beh = _BEH[key]
    smap = _STIM_MAP.get(key, {})
    info = _SESSIONS[key]

    ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (info['mname'], info['datexp'])),
                  allow_pickle=True)
    iarea = ret['iarea']
    n_neurons_total = len(iarea)

    rng = np.random.default_rng(RNG_SEED + (abs(hash(key)) % (2 ** 31)))
    rows, area_idx = select_neurons(iarea, N_NEURONS_PER_SESSION, rng)
    spk, n_spk_neurons = load_spk_rows(key, rows)
    assert n_spk_neurons == n_neurons_total, 'neuron count mismatch %s: %d vs %d' % (
        key, n_spk_neurons, n_neurons_total)
    t_load = time.time() - t0

    nfr = min(spk.shape[1], len(beh['ft']))
    spk = spk[:, :nfr]
    ft = np.asarray(beh['ft'], dtype=float)[:nfr]
    ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
    ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
    ft_trind = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
    ft_corr = np.asarray(beh['ft_CorrSpc'], dtype=bool)[:nfr]
    ft_move = np.asarray(beh['ft_move'], dtype=float)[:nfr] > 0

    valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
             & np.isfinite(ft_speed))

    mu = spk[:, valid].mean(axis=1, keepdims=True)
    sd = spk[:, valid].std(axis=1, keepdims=True)
    keep_neu = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
    spk = (spk - mu) / np.where(sd > 0, sd, 1.0)
    spk = spk[keep_neu]
    area_idx = area_idx[keep_neu]

    lick_fr = np.atleast_1d(np.asarray(beh['LickFr'], dtype=float))
    lick_bin = np.zeros(nfr, dtype=np.int64)
    if lick_fr.size:
        li = np.floor(lick_fr[np.isfinite(lick_fr)]).astype(int)
        li = li[(li >= 0) & (li < nfr)]
        lick_bin[li] = 1

    ntrials = int(beh['ntrials'])
    wall = np.asarray(beh['WallName'])
    is_rew = np.asarray(beh['isRew'], dtype=bool)
    t_start = np.asarray(beh['Trial_start_time'], dtype=float)
    t_sound = np.asarray(beh['SoundTime'], dtype=float)

    idx_valid = np.where(valid)[0]
    trials_out = []
    n_drop_short = n_drop_stim = n_drop_cue = 0
    if idx_valid.size:
        tr_valid = ft_trind[idx_valid].astype(int)
        order = np.argsort(tr_valid, kind='stable')
        tr_sorted = tr_valid[order]
        idx_sorted = idx_valid[order]
        bounds = np.where(np.diff(tr_sorted) != 0)[0] + 1
        for fr, tt in zip(np.split(idx_sorted, bounds), np.split(tr_sorted, bounds)):
            tr = int(tt[0])
            if tr < 0 or tr >= ntrials:
                continue
            fr = np.sort(fr)
            if len(fr) < MIN_FRAMES_PER_TRIAL:
                n_drop_short += 1
                continue
            wname = str(wall[tr])
            if wname not in smap:
                n_drop_stim += 1
                continue
            if not np.isfinite(t_sound[tr]) or not np.isfinite(t_start[tr]):
                n_drop_cue += 1
                continue
            trials_out.append({
                'frames': fr,
                'neural': np.ascontiguousarray(spk[:, fr]),
                'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
                'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
                'pos_dm': ft_pos[fr],
                'speed': ft_speed[fr],
                'lick': lick_bin[fr],
                'stim': smap[wname],
                'wall': wname,
                'is_rew': int(is_rew[tr]),
                'trial_index': tr,
            })

    out = {
        'key': key, 'mname': info['mname'], 'datexp': info['datexp'],
        'exp_types': sorted(set(info['exp_types'])), 'exptype': info['exptype'],
        'trials': trials_out, 'area_idx': area_idx,
        'n_neurons_recorded': int(n_neurons_total), 'n_neurons_used': int(spk.shape[0]),
        'n_frames': int(nfr), 'n_valid_frames': int(valid.sum()),
        'ntrials_raw': ntrials, 'n_drop_short': n_drop_short,
        'n_drop_stim': n_drop_stim, 'n_drop_cue': n_drop_cue,
        'dt': float(np.median(np.diff(ft)) * SEC_PER_DAY),
        'load_time': t_load, 'total_time': time.time() - t0,
    }
    print('  %-22s neu %5d/%6d  fr %6d (valid %6d)  trials %4d/%4d '
          '(short %3d, nostim %3d, nocue %3d)  [%.0fs load, %.0fs tot]'
          % (key, out['n_neurons_used'], out['n_neurons_recorded'], out['n_frames'],
             out['n_valid_frames'], len(trials_out), ntrials, n_drop_short, n_drop_stim,
             n_drop_cue, t_load, out['total_time']), flush=True)
    return out


def parse_date(datexp):
    y, m, d = datexp.split('_')
    return date(int(y), int(m), int(d))


def assemble(results):
    """Build the final data dictionary from the per-session raw streams."""
    results = [r for r in results if len(r['trials']) >= 2]

    first_day = {}
    for r in results:
        d = parse_date(r['datexp'])
        if r['mname'] not in first_day or d < first_day[r['mname']]:
            first_day[r['mname']] = d

    all_speed = np.concatenate([np.concatenate([t['speed'] for t in r['trials']])
                                for r in results])
    speed_edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
    print('\nGlobal running-speed quartile edges (cm/s): %s'
          % np.round(speed_edges, 3), flush=True)

    subjects = sorted(set(r['mname'] for r in results))
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64),
        'brain_regions': AREA_NAMES,
        'brain_region_idx': [r['area_idx'].astype(np.int64) for r in results],
        'input_names': ['time_to_sound_cue', 'sound_cue_onset', 'day_of_training',
                        'time_since_trial_start', 'reward_availability'],
        'output_names': ['stimulus_category', 'licking', 'position_bin', 'speed_bin'],
        'output_values': [
            CANONICAL_STIM,
            ['no_lick', 'lick'],
            ['0-1m', '1-2m', '2-3m', '3-4m'],
            ['q1_slowest', 'q2', 'q3', 'q4_fastest'],
        ],
    }

    session_info = []
    for r in results:
        day = (parse_date(r['datexp']) - first_day[r['mname']]).days
        neural_s, input_s, output_s = [], [], []
        for t in r['trials']:
            T = len(t['frames'])
            cue_onset = np.zeros(T, dtype=np.float32)
            after = np.where(t['time_to_cue'] <= 0)[0]
            if after.size:
                cue_onset[after[0]] = 1.0
            inp = np.stack([
                t['time_to_cue'].astype(np.float32),
                cue_onset,
                np.full(T, float(day), dtype=np.float32),
                t['time_since_start'].astype(np.float32),
                np.full(T, float(t['is_rew']), dtype=np.float32),
            ])
            pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
            spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
            outp = np.stack([
                np.full(T, t['stim'], dtype=np.int64),
                t['lick'].astype(np.int64),
                pos_bin.astype(np.int64),
                spd_bin.astype(np.int64),
            ])
            neural_s.append(t['neural'].astype(np.float32))
            input_s.append(inp)
            output_s.append(outp)
        data['neural'].append(neural_s)
        data['input'].append(input_s)
        data['output'].append(output_s)
        session_info.append({
            'session_key': r['key'], 'mouse': r['mname'], 'date': r['datexp'],
            'experiment_types': r['exp_types'], 'cohort': r['exptype'],
            'day_of_training': day, 'n_trials': len(r['trials']),
            'n_trials_recorded': r['ntrials_raw'],
            'n_neurons_recorded': r['n_neurons_recorded'],
            'n_neurons_used': r['n_neurons_used'],
            'n_frames_recorded': r['n_frames'], 'n_frames_used': r['n_valid_frames'],
            'frame_interval_s': r['dt'],
        })

    dt_ms = float(np.mean([r['dt'] for r in results]) * 1000.0)
    durations = np.concatenate([[len(t['frames']) * r['dt'] for t in r['trials']]
                                for r in results])
    data['metadata'] = {
        'task_description': (
            'Head-fixed mice ran through 4 m long virtual-reality corridors whose walls showed one of '
            'several naturalistic texture patterns (leaf/circle families, also rock/brick in some mice, '
            'and spatially swapped versions of the rewarded texture). A sound cue was played at a random '
            'position (0.5-3.5 m) in every corridor; in task (water-restricted) mice licking after the '
            'cue in the rewarded corridor delivered water. Decoder outputs: visual stimulus category, '
            'licking (binary, per frame), position in the corridor (4 x 1 m bins) and running speed '
            '(4 quartile bins). Decoder inputs: signed time to the sound cue, binary cue onset, day of '
            'training, time since corridor entry and reward availability.'),
        'time_bin_size': dt_ms,
        'temporal_alignment_event': 'corridor entry (trial start; beh["Trial_start_time"]/beh["StartFr"])',
        'off_start': 0.0,
        'off_end': None,
        'off_end_note': ('trials end when the mouse leaves the 4 m textured corridor, so trial length '
                         'varies with running speed: mean %.2f s, median %.2f s, min %.2f s, max %.2f s'
                         % (durations.mean(), np.median(durations), durations.min(), durations.max())),
        'neural_data_type': ('deconvolved calcium traces (Suite2p, non-negative deconvolution, 0.75 s '
                             'decay), z-scored per neuron over the included frames'),
        'neuron_selection': ('neurons outside visual cortex (iarea in {-1,7}) excluded, then up to %d '
                             'neurons per session sampled proportionally across V1/mHV/lHV/aHV '
                             '(the full dataset is 405 GB)' % N_NEURONS_PER_SESSION),
        'frame_selection': ('frames inside the 4 m textured corridor while the mouse was running '
                            '(reference mask fr_valid = (ft_move>0) & ft_CorrSpc)'),
        'input_descriptions': {
            'time_to_sound_cue': 'seconds until the sound cue (positive before, negative after)',
            'sound_cue_onset': 'binary, 1 on the first frame at/after the sound cue',
            'day_of_training': 'days elapsed since the first imaging session of that mouse',
            'time_since_trial_start': 'seconds since corridor entry',
            'reward_availability': '1 if the corridor is the rewarded one, 0 otherwise',
        },
        'output_descriptions': {
            'stimulus_category': 'canonical texture identity (%s)' % ', '.join(CANONICAL_STIM),
            'licking': '1 if at least one lick occurred in that imaging frame',
            'position_bin': 'position in the 4 m corridor, 4 equal 1 m bins',
            'speed_bin': 'running speed, 4 bins each holding 25%% of all timepoints',
        },
        'speed_bin_edges_cm_per_s': speed_edges.tolist(),
        'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
        'source': 'Zhong et al. 2025, Unsupervised pretraining in biological neural networks',
        'session_info': session_info,
    }
    return data, speed_edges


def show_processing(result, speed_edges):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    key = result['key']
    trials = result['trials'][:6]
    fig, axs = plt.subplots(6, 1, figsize=(16, 21))

    beh = _BEH[key]
    nfr = result['n_frames']
    n_show = min(600, nfr)
    pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr][:n_show]
    corr = np.asarray(beh['ft_CorrSpc'], dtype=bool)[:nfr][:n_show]
    mv = np.asarray(beh['ft_move'], dtype=float)[:nfr][:n_show] > 0
    ax = axs[0]
    ax.plot(pos, 'k-', lw=0.6, label='ft_Pos (dm)')
    ax.plot(np.where(corr & mv)[0], pos[corr & mv], 'g.', ms=4,
            label='included: corridor & running')
    ax.plot(np.where(corr & ~mv)[0], pos[corr & ~mv], 'r.', ms=4,
            label='corridor, not running (excluded)')
    ax.plot(np.where(~corr)[0], pos[~corr], 'b.', ms=2, label='grey space (excluded)')
    ax.axhline(40, color='b', ls='--', lw=0.8)
    ax.set_title('%s step 1: frame curation, first %d frames (dashed = end of 4 m texture area)'
                 % (key, n_show))
    ax.set_xlabel('neural frame'); ax.set_ylabel('position (dm)'); ax.legend(fontsize=7)

    ax = axs[1]
    cat = np.concatenate([t['neural'][:60] for t in trials], axis=1)
    ax.imshow(cat, aspect='auto', vmin=-1, vmax=4, cmap='gray_r', interpolation='nearest')
    b = 0
    for t in trials:
        b += t['neural'].shape[1]
        ax.axvline(b - 0.5, color='r', lw=1)
    ax.set_title('step 2: z-scored deconvolved activity (60 example neurons), first %d trials '
                 '(red = trial boundaries)' % len(trials))
    ax.set_xlabel('timepoint'); ax.set_ylabel('neuron')

    ax = axs[2]
    b = 0
    for i, t in enumerate(trials):
        T = len(t['frames'])
        x = np.arange(b, b + T)
        ax.plot(x, t['time_to_cue'], 'b.-', lw=0.8, label='time to cue (s)' if i == 0 else None)
        ax.plot(x, t['time_since_start'], 'g.-', lw=0.8,
                label='time since trial start (s)' if i == 0 else None)
        ax.plot(x, np.full(T, t['is_rew']), 'm-', lw=1.5,
                label='reward availability' if i == 0 else None)
        cue = np.where(t['time_to_cue'] <= 0)[0]
        if cue.size:
            ax.axvline(b + cue[0], color='c', ls=':', lw=1.2,
                       label='cue onset frame' if i == 0 else None)
        b += T
        ax.axvline(b - 0.5, color='r', lw=1)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('step 3: decoder inputs')
    ax.set_xlabel('timepoint'); ax.legend(fontsize=7)

    ax = axs[3]
    b = 0
    for i, t in enumerate(trials):
        T = len(t['frames'])
        x = np.arange(b, b + T)
        ax.plot(x, t['pos_dm'] / 10.0, 'k.-', lw=0.8, label='position (m)' if i == 0 else None)
        pb = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
        ax.step(x, pb, 'r-', where='mid', label='position bin' if i == 0 else None)
        b += T
        ax.axvline(b - 0.5, color='b', lw=1)
    for e in [1, 2, 3]:
        ax.axhline(e, color='grey', lw=0.5, ls=':')
    ax.set_title('step 4: output position and its 4 x 1 m discretisation')
    ax.set_xlabel('timepoint'); ax.legend(fontsize=7)

    ax = axs[4]
    b = 0
    for i, t in enumerate(trials):
        T = len(t['frames'])
        x = np.arange(b, b + T)
        ax.plot(x, t['speed'], 'k.-', lw=0.8, label='speed (cm/s)' if i == 0 else None)
        sb = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
        ax.step(x, sb * 10, 'r-', where='mid', label='speed bin x10' if i == 0 else None)
        ax.step(x, t['lick'] * 5, 'g-', where='mid', label='licking x5' if i == 0 else None)
        ax.text(b + 0.5, np.nanmax(t['speed']) if T else 0,
                '%s%s' % (CANONICAL_STIM[t['stim']], ' (rew)' if t['is_rew'] else ''),
                fontsize=7, color='m')
        b += T
        ax.axvline(b - 0.5, color='b', lw=1)
    for e in speed_edges:
        ax.axhline(e, color='grey', lw=0.5, ls=':')
    ax.set_title('step 5: output running speed (grey dotted = global quartile edges), licking, '
                 'and per-trial stimulus label')
    ax.set_xlabel('timepoint'); ax.legend(fontsize=7)

    ax = axs[5]
    allspd = np.concatenate([t['speed'] for t in result['trials']])
    ax.hist(allspd, bins=100, color='k')
    for e in speed_edges:
        ax.axvline(e, color='r', ls='--')
    stims = np.array([t['stim'] for t in result['trials']])
    counts = ', '.join('%s:%d' % (CANONICAL_STIM[s], int((stims == s).sum()))
                       for s in np.unique(stims))
    ax.set_title('step 6: session speed distribution with global quartile edges | trials per '
                 'stimulus: %s' % counts)
    ax.set_xlabel('running speed (cm/s)')

    fig.tight_layout()
    fn = '/app/processing_%s.png' % key
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    print('  wrote %s' % fn, flush=True)


def sanity_checks(data):
    print('\n==== SANITY CHECKS ====', flush=True)
    nses = len(data['neural'])
    ntr = sum(len(s) for s in data['neural'])
    T = np.concatenate([[t.shape[1] for t in s] for s in data['neural']])
    nneu = [s[0].shape[0] for s in data['neural']]
    rec = [si['n_neurons_recorded'] for si in data['metadata']['session_info']]
    dt = data['metadata']['time_bin_size'] / 1000.0
    print('sessions: %d | subjects: %d | trials: %d (mean %.1f/session)'
          % (nses, len(data['subjects']), ntr, ntr / nses))
    print('neurons used/session: min %d max %d | recorded/session: min %d max %d '
          '(paper: 20,547-89,577)' % (min(nneu), max(nneu), min(rec), max(rec)))
    print('timepoints/trial: mean %.1f median %d min %d max %d -> mean duration %.2f s '
          '(expect ~6.7 s for 4 m at 60 cm/s)'
          % (T.mean(), np.median(T), T.min(), T.max(), T.mean() * dt))
    inp = np.concatenate([np.concatenate(s, axis=1) for s in data['input']], axis=1)
    out = np.concatenate([np.concatenate(s, axis=1) for s in data['output']], axis=1)
    for i, n in enumerate(data['input_names']):
        print('input  %d %-24s range [%.3f, %.3f] mean %.3f'
              % (i, n, inp[i].min(), inp[i].max(), inp[i].mean()))
    for i, n in enumerate(data['output_names']):
        vals, cnt = np.unique(out[i], return_counts=True)
        print('output %d %-24s %s'
              % (i, n, dict(zip(vals.tolist(), np.round(cnt / cnt.sum(), 4).tolist()))))
    cue_pos = (inp[3] + inp[0]) * 0.6
    print('implied cue position (m, VR 60 cm/s): 1st %.2f, median %.2f, 99th %.2f '
          '(paper: uniform 0.5-3.5 m)' % tuple(np.percentile(cue_pos, [1, 50, 99])))
    print('fraction of trials in rewarded corridor: %.3f'
          % np.mean([t[4, 0] for s in data['input'] for t in s]))
    print('fraction of timepoints with licking: %.4f' % out[1].mean())
    print('=======================\n', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Convert Zhong et al. 2025 data.')
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nworkers', type=int, default=6)
    args = ap.parse_args()

    t_start = time.time()
    print('Loading experiment info ...', flush=True)
    sessions = load_exp_info()
    print('  %d unique sessions, %d mice'
          % (len(sessions), len(set(s['mname'] for s in sessions.values()))), flush=True)

    print('Loading behaviour files ...', flush=True)
    t0 = time.time()
    beh, stim_map = load_behaviour(sessions)
    print('  behaviour loaded in %.1f s (%d sessions)' % (time.time() - t0, len(beh)), flush=True)

    keys = sorted(beh.keys())
    if args.sample:
        preferred = [k for k in ['TX108_2023_03_25_1', 'TX83_2022_08_31_1'] if k in beh]
        keys = preferred if len(preferred) == 2 else keys[:2]
        print('SAMPLE MODE: processing %s' % keys, flush=True)

    _init_worker(beh, stim_map, sessions)

    print('\nProcessing %d sessions with %d workers ...' % (len(keys), args.nworkers), flush=True)
    t0 = time.time()
    if args.nworkers > 1 and len(keys) > 1:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(args.nworkers) as pool:
            results = pool.map(process_session, keys, chunksize=1)
    else:
        results = [process_session(k) for k in keys]
    t_proc = time.time() - t0
    print('Session processing: %.1f s total, %.1f s/session'
          % (t_proc, t_proc / len(keys)), flush=True)

    n_drop_sess = sum(1 for r in results if len(r['trials']) < 2)
    if n_drop_sess:
        print('WARNING: dropping %d sessions with < 2 usable trials' % n_drop_sess, flush=True)

    data, speed_edges = assemble(results)
    sanity_checks(data)

    if args.show_processing:
        for r in [r for r in results if len(r['trials']) >= 2][:2]:
            show_processing(r, speed_edges)

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('Wrote %s (%.2f GB) in %.1f s'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t0), flush=True)
    print('TOTAL TIME: %.1f s' % (time.time() - t_start), flush=True)


if __name__ == '__main__':
    main()
