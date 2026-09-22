"""Convert the IBL brain-wide map dataset into the decoder's pickle format.

The processing follows the reference pipeline of Zhang et al. 2025
(`/app/code/code_zhang2025/src/0_data_caching.py` and `utils/ibl_data_utils.py`),
which is itself built on the IBL brain-wide-map release described in
"A brain-wide map of neural activity during complex behaviour".

Decisions, and where they come from
-----------------------------------
* Sessions: the 459 eids of the brain-wide map freeze shipped with the reference
  repo (`data/bwm_release.csv`).  These are exactly the sessions that passed the
  data paper's session/insertion inclusion criteria.
* Probes: all insertions of a session are merged (`merge_probes`), because the
  data paper states probes from one session "are not independent" and must be
  combined.
* Neurons: every Kilosort cluster is kept.  `prepare_data` calls
  `load_spiking_data` with the default `qc=None`, and the methods paper says
  "we bin spike counts using all neurons, sorted by Kilosort 2.5, from each
  session".  Clusters that emit no spike at all in the session are dropped,
  which is what `get_spike_data_per_interval` does implicitly.
* Trials: `load_trials_and_mask(..., max_trial_len=10.0)`, i.e. the data paper's
  trial exclusions -- no NaN in stimOn_times / choice / feedback_times /
  probabilityLeft / firstMovement_times / feedbackType, reaction time in
  [0.08, 2] s, trial shorter than 10 s, and a response was made.  Trials whose
  behavioural traces do not cover the decoding window are dropped as well, which
  is what `get_behavior_per_interval` + `align_spike_behavior` do.
* Alignment / binning: stimulus onset, window (-0.5, 1.5) s, 20 ms
  non-overlapping bins -> T = 100, exactly the `params` dict of
  `0_data_caching.py`.
* Behaviour: wheel speed is |velocity| of the SessionLoader wheel trace and
  whisker motion energy is the left camera trace (right camera as fallback),
  both linearly interpolated onto the bin grid, as in `load_target_behavior` /
  `get_behavior_per_interval`.

Decoder inputs  : time since stimulus onset, trial number within block.
Decoder outputs : choice, prior probability of left, wheel speed (3 bins),
                  whisker motion energy (3 bins).
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.insert(0, '/app/code/code_zhang2025/src')

warnings.filterwarnings('ignore')

from one.api import ONE                                    # noqa: E402
from brainbox.io.one import SessionLoader, SpikeSortingLoader   # noqa: E402
from iblatlas.regions import BrainRegions                  # noqa: E402
from utils.ibl_data_utils import load_trials_and_mask, merge_probes  # noqa: E402

# ---------------------------------------------------------------- parameters
CACHE_DIR = '/app/data/one_cache'
FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'
OUT_FILE = '/app/converted_data.pkl'
TMP_DIR = Path('/app/work/sessions')

ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to stimulus onset
BINSIZE = 0.02                     # seconds
NBINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
MAX_TRIAL_LEN = 10.0
NBEHBINS = 3                       # tertiles for the continuous behaviours

# bin right edges relative to the alignment event; this is the grid that the
# reference code interpolates the behavioural traces onto
BIN_TIMES = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], NBINS)


# ------------------------------------------------------------- ONE cache
def rebuild_cache_tables():
    """Regenerate the ONE cache tables from the files staged on disk.

    The container ships the released ONE tables for the tags 2022_Q4_IBL_et_al_BWM,
    Brainwidemap and 2025_Q3_IBL_et_al_BWM, but the data staged under
    /app/data/one_cache is *newer* than any of them: the trials tables sit under
    revision #2025-03-03#, the camera motion energy under #2025-05-xx# and the
    spike sorting under #2024-05-06#, none of which appear in the shipped tables.
    With no network ONE cannot refresh them, and it then silently resolves only
    the handful of unrevisioned datasets (a trials object with a single column,
    no motion energy).  So we rebuild the tables by walking the staged `alf`
    trees, keeping the released session UUIDs and marking the newest revision of
    each dataset as the default one -- which is what ONE would have been told by
    Alyx.  Nothing else about the loading is changed: the files read are the ones
    the reference pipeline reads.
    """
    import uuid
    from iblutil.io import parquet
    from one.alf.path import rel_path_parts

    root = Path(CACHE_DIR)
    tags = ['2022_Q4_IBL_et_al_BWM', 'Brainwidemap', '2025_Q3_IBL_et_al_BWM']
    ns = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

    sessions = pd.concat([pd.read_parquet(root / t / 'sessions.pqt')
                          for t in tags if (root / t / 'sessions.pqt').exists()])
    sessions = sessions[~sessions.index.duplicated(keep='last')].sort_index()

    rows = []
    for eid, r in sessions.iterrows():
        d = root / f"{r.lab}/Subjects/{r.subject}/{r.date}/{int(r.number):03d}"
        if not (d / 'alf').exists():
            continue
        for f in (d / 'alf').rglob('*'):
            if not f.is_file():
                continue
            rel = str(f.relative_to(d))
            try:
                rev = rel_path_parts(rel, assert_valid=True)[1]
            except Exception:
                continue
            base = rel.replace(f'#{rev}#/', '') if rev else rel
            rows.append((eid, rel, base, rev or '', f.stat().st_size))

    ds = pd.DataFrame(rows, columns=['eid', 'rel_path', 'base', 'revision', 'file_size'])
    newest = ds.groupby(['eid', 'base'])['revision'].transform('max')
    ds['default_revision'] = ds['revision'].values == newest.values
    ds['id'] = [str(uuid.uuid5(ns, e + '/' + p)) for e, p in zip(ds.eid, ds.rel_path)]
    ds['hash'] = ''
    ds['exists'] = True
    ds['qc'] = pd.Categorical(['NOT_SET'] * len(ds),
                             categories=['NOT_SET', 'PASS', 'WARNING', 'FAIL', 'CRITICAL'],
                             ordered=True)
    ds['file_size'] = ds['file_size'].astype('UInt64')
    ds = ds.set_index(['eid', 'id']).sort_index()
    ds = ds[['file_size', 'hash', 'default_revision', 'qc', 'exists', 'rel_path']]

    meta = {'date_created': '2025-12-11 12:26', 'origin': 'public',
            'min_api_version': '2.10', 'database_tags': tags}
    parquet.save(root / 'datasets.pqt', ds, meta)
    parquet.save(root / 'sessions.pqt', sessions, meta)
    with open(root / 'cache_info.json', 'w') as fh:
        json.dump({**meta, 'tables': {'sessions': {'nrecs': len(sessions), 'size': 0},
                                      'datasets': {'nrecs': len(ds), 'size': 0}}}, fh)
    print(f'rebuilt ONE cache: {len(sessions)} sessions, {len(ds)} datasets', flush=True)


def get_one():
    """ONE client in local (offline) mode against the staged cache."""
    return ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
               cache_dir=CACHE_DIR, mode='local')


# ---------------------------------------------------------------- behaviour
def interp_behavior(target_times, target_vals, interval_begs, interval_ends):
    """Interpolate a continuous behavioural trace onto each trial's bin grid.

    Mirrors `utils.ibl_data_utils.get_behavior_per_interval`: samples strictly
    inside the interval are used, a trial is rejected if the trace does not
    cover the interval to within one bin, and the values are linearly
    interpolated onto `linspace(beg + binsize, end, nbins)`.

    Returns
    -------
    values : (ntrials, NBINS) float array, NaN for rejected trials
    good   : (ntrials,) bool array
    """
    ntrials = len(interval_begs)
    values = np.full((ntrials, NBINS), np.nan)
    good = np.zeros(ntrials, dtype=bool)
    if target_times is None or target_vals is None:
        return values, good

    idx_beg = np.searchsorted(target_times, interval_begs, side='right')
    idx_end = np.searchsorted(target_times, interval_ends, side='left')

    for k in range(ntrials):
        if np.isnan(interval_begs[k]) or np.isnan(interval_ends[k]):
            continue
        tt = target_times[idx_beg[k]:idx_end[k]]
        vv = target_vals[idx_beg[k]:idx_end[k]]
        if len(vv) == 0 or np.any(np.isnan(vv)):
            continue
        if np.abs(interval_begs[k] - tt[0]) > BINSIZE:      # starts too late
            continue
        if np.abs(interval_ends[k] - tt[-1]) > BINSIZE:     # ends too early
            continue
        x = interval_begs[k] + BIN_TIMES - TIME_WINDOW[0]
        values[k] = interp1d(tt, vv, kind='linear', fill_value='extrapolate')(x)
        good[k] = True
    return values, good


def discretize(values, nbins=NBEHBINS):
    """Discretise a continuous behaviour into `nbins` equal-occupancy bins.

    Quantiles are computed within a session.  Wheel speed and, above all,
    whisker motion energy are in units that are not comparable across sessions
    (motion energy depends on camera, illumination and ROI size), so a single
    global threshold would put whole sessions into one class.  Equal-occupancy
    (tertile) edges keep the three classes usable in every session.
    """
    edges = np.nanquantile(values, np.arange(1, nbins) / nbins)
    edges = np.unique(edges)
    return np.digitize(values, edges).astype(np.int64)


def trial_in_block(pleft):
    """Index of each trial within its block of constant probabilityLeft."""
    out = np.zeros(len(pleft), dtype=np.int64)
    count = 0
    for i in range(len(pleft)):
        if i > 0 and pleft[i] != pleft[i - 1]:
            count = 0
        out[i] = count
        count += 1
    return out


# ---------------------------------------------------------------- one session
def process_session(eid, sub_df):
    one = get_one()
    brainreg = BrainRegions()

    # ---- spikes, all probes of the session merged -------------------------
    spikes_list, clusters_list = [], []
    for _, r in sub_df.iterrows():
        ssl = SpikeSortingLoader(pid=r.pid, one=one, eid=eid, pname=r.probe_name)
        sp, cl, ch = ssl.load_spike_sorting()
        if len(sp) == 0:
            continue
        cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
        spikes_list.append(sp)
        clusters_list.append(cld)
    if len(spikes_list) == 0:
        raise RuntimeError('no spike sorting')
    spikes, clusters = merge_probes(spikes_list, clusters_list)

    # ---- trials -----------------------------------------------------------
    sess_loader = SessionLoader(one=one, eid=eid)
    sess_loader.load_trials()
    trials_df, trials_mask = load_trials_and_mask(
        one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
    trials_mask = trials_mask.to_numpy()

    align = trials_df[ALIGN_TIME].to_numpy(dtype=float)
    beg = align + TIME_WINDOW[0]
    end = align + TIME_WINDOW[1]

    # ---- behaviour --------------------------------------------------------
    sess_loader.load_wheel()
    wheel_speed, wheel_ok = interp_behavior(
        sess_loader.wheel['times'].to_numpy(),
        np.abs(sess_loader.wheel['velocity'].to_numpy()), beg, end)

    me_vals, me_ok, me_view = None, None, None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            df = sess_loader.motion_energy[cam]
            v, ok = interp_behavior(df['times'].to_numpy(),
                                    df['whiskerMotionEnergy'].to_numpy(), beg, end)
        except Exception:
            continue
        if ok.sum() > 0:
            me_vals, me_ok, me_view = v, ok, view
            break
    if me_vals is None:
        raise RuntimeError('no whisker motion energy')

    keep = trials_mask & wheel_ok & me_ok & ~np.isnan(align)
    if keep.sum() < 2:
        raise RuntimeError(f'only {keep.sum()} usable trials')
    kidx = np.flatnonzero(keep)

    # ---- bin spikes -------------------------------------------------------
    st = spikes['times']
    sc = spikes['clusters']
    finite = np.isfinite(st) & np.isfinite(sc)
    st, sc = st[finite], sc[finite].astype(np.int64)
    order = np.argsort(st, kind='stable')
    st, sc = st[order], sc[order]

    # clusters that fire at least once, as in get_spike_data_per_interval
    used = np.unique(sc)
    remap = np.full(len(clusters), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nneurons = len(used)

    binned = np.zeros((len(kidx), nneurons, NBINS), dtype=np.float32)
    for j, k in enumerate(kidx):
        i0 = np.searchsorted(st, beg[k], side='left')
        i1 = np.searchsorted(st, end[k], side='left')
        if i1 <= i0:
            continue
        b = ((st[i0:i1] - beg[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        np.add.at(binned[j], (remap[sc[i0:i1]], b), 1.0)

    # ---- decoder inputs ---------------------------------------------------
    tib = trial_in_block(trials_df['probabilityLeft'].to_numpy())[kidx]
    inputs = np.empty((len(kidx), 2, NBINS), dtype=np.float32)
    inputs[:, 0, :] = BIN_TIMES[None, :]
    inputs[:, 1, :] = tib[:, None]

    # ---- decoder outputs --------------------------------------------------
    # IBL convention: choice == +1 -> stimulus/response on the LEFT,
    # choice == -1 -> RIGHT (verified against contrastLeft on correct trials).
    choice = trials_df['choice'].to_numpy()[kidx]
    choice_out = (choice < 0).astype(np.int64)             # left 0, right 1
    pleft = trials_df['probabilityLeft'].to_numpy()[kidx]
    prior_out = np.select([np.isclose(pleft, 0.2), np.isclose(pleft, 0.5),
                           np.isclose(pleft, 0.8)], [0, 1, 2], default=-1)
    if np.any(prior_out < 0):
        raise RuntimeError('unexpected probabilityLeft value')

    wheel_bin = discretize(wheel_speed[kidx])
    me_bin = discretize(me_vals[kidx])

    outputs = np.empty((len(kidx), 4, NBINS), dtype=np.int64)
    outputs[:, 0, :] = choice_out[:, None]
    outputs[:, 1, :] = prior_out[:, None]
    outputs[:, 2, :] = wheel_bin
    outputs[:, 3, :] = me_bin

    # ---- brain regions (Beryl) -------------------------------------------
    acronyms = clusters['acronym'].to_numpy()[used]
    beryl = np.asarray(brainreg.acronym2acronym(acronyms, mapping='Beryl'))

    meta = dict(eid=eid, subject=sub_df.subject.iloc[0], lab=sub_df.lab.iloc[0],
                date=str(sub_df.date.iloc[0]), n_probes=len(sub_df),
                motion_energy_view=me_view, ntrials=int(len(kidx)),
                ntrials_total=int(len(trials_df)), nneurons=int(nneurons))
    return binned, inputs, outputs, beryl, meta


def _worker(args):
    eid, sub_df = args
    try:
        binned, inputs, outputs, beryl, meta = process_session(eid, sub_df)
    except Exception as e:                                  # noqa: BLE001
        return eid, None, f'{type(e).__name__}: {e}'
    f = TMP_DIR / f'{eid}.npz'
    np.savez(f, neural=binned, input=inputs, output=outputs, regions=beryl)
    return eid, meta, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sessions', type=int, default=None)
    ap.add_argument('--n_workers', type=int, default=16)
    ap.add_argument('--out', type=str, default=OUT_FILE)
    ap.add_argument('--skip_processing', action='store_true')
    args = ap.parse_args()

    if not (Path(CACHE_DIR) / 'datasets.pqt').exists():
        rebuild_cache_tables()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    bwm_df = pd.read_csv(FREEZE_FILE, index_col=0)

    # only keep sessions whose raw data is staged locally
    one = get_one()
    available = set(one._cache['sessions'].index.astype(str))
    eids = [e for e in bwm_df.eid.unique() if e in available]
    eids.sort()
    if args.n_sessions:
        eids = eids[:args.n_sessions]
    print(f'{len(eids)} sessions to process', flush=True)

    jobs = [(e, bwm_df[bwm_df.eid == e]) for e in eids]
    results = {}
    if not args.skip_processing:
        import multiprocessing as mp
        with mp.Pool(args.n_workers, maxtasksperchild=4) as pool:
            for i, (eid, meta, err) in enumerate(pool.imap_unordered(_worker, jobs)):
                if err:
                    print(f'[{i+1}/{len(jobs)}] skipped {eid}: {err}', flush=True)
                else:
                    results[eid] = meta
                    print(f"[{i+1}/{len(jobs)}] {eid} "
                          f"{meta['ntrials']} trials x {meta['nneurons']} neurons",
                          flush=True)
        pd.DataFrame(results).T.to_csv(TMP_DIR / 'meta.csv')
    else:
        results = pd.read_csv(TMP_DIR / 'meta.csv', index_col=0).to_dict('index')

    # ------------------------------------------------------------ assemble
    eids = [e for e in eids if e in results]
    subjects, subject_idx = [], []
    brain_regions, brain_region_idx = [], []
    neural, inputs, outputs, session_info = [], [], [], []
    reg_lookup = {}

    for eid in eids:
        meta = results[eid]
        z = np.load(TMP_DIR / f'{eid}.npz', allow_pickle=True)
        nb, ib, ob, regs = z['neural'], z['input'], z['output'], z['regions']
        neural.append([nb[i] for i in range(nb.shape[0])])
        inputs.append([ib[i] for i in range(ib.shape[0])])
        outputs.append([ob[i] for i in range(ob.shape[0])])

        sub = str(meta['subject'])
        if sub not in subjects:
            subjects.append(sub)
        subject_idx.append(subjects.index(sub))

        idx = np.empty(len(regs), dtype=np.int64)
        for i, a in enumerate(regs):
            a = str(a)
            if a not in reg_lookup:
                reg_lookup[a] = len(brain_regions)
                brain_regions.append(a)
            idx[i] = reg_lookup[a]
        brain_region_idx.append(idx)
        session_info.append(dict(meta))

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_stimulus_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left',
                         'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],
            ['0.2', '0.5', '0.8'],
            ['low', 'medium', 'high'],
            ['low', 'medium', 'high'],
        ],
        'metadata': {
            'task_description':
                'IBL decision-making task: a Gabor patch appears at +-35 deg azimuth '
                'with one of five contrasts and the mouse turns a wheel to bring it to '
                'the centre. After 90 unbiased trials the stimulus side is drawn from '
                'blocks with prior probability of left 0.2 or 0.8. Decode, from '
                'Neuropixels population spike counts, the choice and the block prior '
                '(per trial) and the tertile-discretised wheel speed and whisker '
                'motion energy (per 20 ms bin).',
            'time_bin_size': BINSIZE * 1000.0,
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'dataset': 'IBL brain-wide map (bwm_release freeze, 459 sessions)',
            'spike_sorting': 'Kilosort 2.5 / pykilosort, all clusters, probes merged '
                             'within a session',
            'neural_units': 'spike counts per 20 ms bin',
            'trial_inclusion':
                'load_trials_and_mask(max_trial_len=10 s): no NaN in stimOn_times, '
                'choice, feedback_times, probabilityLeft, firstMovement_times or '
                'feedbackType; 0.08 s <= firstMovement_times - stimOn_times <= 2 s; '
                'goCue-to-feedback < 10 s; a choice was made. Trials whose wheel or '
                'whisker traces do not cover the decoding window are also dropped.',
            'input_descriptions': {
                'time_from_stimulus_onset':
                    'seconds from stimulus onset to the right edge of each 20 ms bin, '
                    'from -0.48 to 1.5 s',
                'trial_number_in_block':
                    'index of the trial within its block of constant probabilityLeft '
                    '(0-based), constant within a trial',
            },
            'output_descriptions': {
                'choice': 'side the mouse reported; IBL choice +1 (left) -> 0, '
                          '-1 (right) -> 1; constant within a trial',
                'prior_probability_left': 'block probabilityLeft, 0.2 -> 0, 0.5 -> 1, '
                                          '0.8 -> 2; constant within a trial',
                'wheel_speed': 'abs(wheel velocity) interpolated to each bin then '
                               'split at the within-session tertiles',
                'whisker_motion_energy': 'whisker-pad motion energy of the left camera '
                                         '(right camera if left is unavailable), '
                                         'interpolated to each bin then split at the '
                                         'within-session tertiles',
            },
            'behaviour_discretisation': 'equal-occupancy tertiles computed per session',
            'session_info': session_info,
        },
    }

    ntr = sum(len(s) for s in neural)
    nn = sum(s[0].shape[0] for s in neural if len(s))
    print(f'{len(neural)} sessions, {ntr} trials, {nn} neurons, '
          f'{len(subjects)} subjects, {len(brain_regions)} regions', flush=True)

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('wrote', args.out, os.path.getsize(args.out) / 1e9, 'GB', flush=True)


if __name__ == '__main__':
    main()
