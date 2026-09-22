"""Convert the IBL brain-wide-map dataset into the neural-decoder data format.

All loading / curation / alignment / binning decisions follow the reference
implementation of the methods paper (Zhang et al., "Exploiting correlations across
trials and behavioral sessions to improve neural decoding"), vendored in
/app/code/code_zhang2025 -- specifically src/0_data_caching.py and
src/utils/ibl_data_utils.py.

Sessions: all sessions of the BWM public release freeze shipped with the reference code
  (data/bwm_release.csv; 459 sessions, 139 mice), in order of first appearance in that
  freeze. 443 sessions / 136 mice survive (188,554 trials, 599,213 neurons, 280 Beryl
  regions), comparable to the 433 sessions and 270 regions the methods paper reports.
Neural: pykilosort spike-sorted spikes of ALL clusters (reference prepare_data calls
  load_spiking_data with qc=None, i.e. no unit-quality threshold; methods paper: "we bin
  spike counts using all neurons ... from each session"). All probes of a session are
  merged as if recorded by one probe (merge_probes), as both papers do, because probes in
  one session are not statistically independent.
Alignment: stimOn_times, window (-0.5, +1.5) s, non-overlapping 20 ms bins -> T = 100.
  These are exactly the params of 0_data_caching.py and match the methods paper (2-s
  trials in 20-ms bins, T = 100; for choice "from 0.5 s before to 1.5 s post-onset").
  The 20 ms bin size (rather than the 50 ms the paper uses for the purely static
  choice/prior targets) is required here because this task also asks for time-varying
  wheel-speed and whisker-motion-energy outputs, for which paper and code use 20 ms.
Trial curation: load_trials_and_mask(max_trial_len=10.0), exactly as reference
  prepare_data. Drops reaction times < 80 ms or > 2 s, trials longer than 10 s, trials
  with a NaN in stimOn_times/choice/feedback_times/probabilityLeft/firstMovement_times/
  feedbackType, and no-choice trials.
Behaviour curation: as in align_spike_behavior, trials whose behavioural trace could not
  be interpolated across the whole window are dropped. Trials whose wheel or whisker
  trace still contains a NaN are also dropped: the reference runs with allow_nans=True,
  but a NaN cannot be assigned a category, and the decoder rejects non-finite values.
Regions: Beryl-mapped acronym of each cluster (list_brain_regions).
Skipped sessions: a session is dropped when it has no wheel or no whisker-motion-energy
  trace at all (both required decoder outputs; the reference 0_data_caching.py skips
  these sessions too, via its per-session try/except) or when fewer than 2 trials
  survive curation, or when one of its probe insertions has no spike sorting at all.

Decoder inputs (d_input = 2, time-varying, (2, 100)):
  0 time_from_stim_onset   signed time (s) of the end of each 20 ms bin relative to
                           stimulus onset: -0.48 ... +1.50. This is the same time grid
                           the reference code interpolates the behaviour onto.
  1 trial_number_in_block  0-based index of the trial within its block of constant
                           probabilityLeft; constant within a trial, broadcast in time.

Decoder outputs (d_output = 4, (4, 100)):
  0 choice                 0 = left, 1 = right. IBL convention: choice == +1 is a
                           leftward wheel turn, choice == -1 a rightward one.
  1 prior_prob_left        probabilityLeft, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.
  2 wheel_speed            |wheel velocity| discretised into 3 bins.
  3 whisker_motion_energy  whisker-pad motion energy discretised into 3 bins.
  The two per-trial variables are broadcast across the 100 bins so one array holds both
  static and time-varying targets.
  Discretisation uses the WITHIN-SESSION tertiles (33.3 / 66.7 percentiles) of all
  retained samples. Both signals are heavy-tailed and in session-specific units (wheel
  speed in rad/s; motion energy in arbitrary camera units that depend on the camera,
  illumination and ROI of that session), so fixed global thresholds would collapse whole
  sessions into one class. Session tertiles give three roughly equinumerous,
  interpretable low/medium/high classes in every session.
"""

import os
import sys
import pickle
import argparse
import traceback

import numpy as np
import pandas as pd

REPO = '/app/code/code_zhang2025/src'
sys.path.insert(0, REPO)

# ibllib 4.0.1 turned brainbox.io.one.SessionLoader into a keyword-only dataclass, while
# the reference code (written against ibllib 2.x) calls SessionLoader(one, eid)
# positionally. Patch it so the vendored reference utilities can be reused unchanged.
import brainbox.io.one as _bbone  # noqa: E402

_SessionLoader = _bbone.SessionLoader


class SessionLoaderCompat(_SessionLoader):
    def __init__(self, one=None, eid='', **kwargs):
        super().__init__(one=one, eid=eid, **kwargs)


import utils.ibl_data_utils as U  # noqa: E402

U.SessionLoader = SessionLoaderCompat
_bbone.SessionLoader = SessionLoaderCompat

from one.api import ONE  # noqa: E402
from brainbox.io.one import SpikeSortingLoader  # noqa: E402

# --- configuration: identical to code_zhang2025/src/0_data_caching.py -----------------
PARAMS = {
    'interval_len': 2,
    'binsize': 0.02,
    'single_region': False,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
NBINS = int(round((PARAMS['time_window'][1] - PARAMS['time_window'][0])
                  / PARAMS['binsize']))
BWM_RELEASE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT = '/app/data/DATALIMIT_SUBSET.csv'
OUT_PATH = '/app/converted_data.pkl'
MIN_TRIALS = 2
BEH_NAMES = ['wheel-speed', 'whisker-motion-energy']

INPUT_NAMES = ['time_from_stim_onset', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['0.2', '0.5', '0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}


def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant probabilityLeft."""
    p = np.asarray(prob_left, dtype=float)
    idx = np.zeros(len(p), dtype=np.float32)
    counter = 0
    for i in range(len(p)):
        if i > 0 and np.isclose(p[i], p[i - 1]):
            counter += 1
        else:
            counter = 0
        idx[i] = counter
    return idx


def discretize_tertiles(values):
    """Map a (ntrials, T) float array to {0,1,2} using this session's pooled tertiles."""
    edges = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
    # searchsorted keeps the mapping monotone even when the two edges coincide (e.g. a
    # session in which more than a third of the wheel samples are exactly zero).
    return np.searchsorted(edges, values, side='right').astype(np.int64)


def load_clusters_and_spikes(one, pid, eid, pname):
    """Spike sorting of one probe insertion, all clusters (no quality threshold).

    Same as utils.ibl_data_utils.load_spiking_data with qc=None, minus its call to
    SpikeSortingLoader.raw_electrophysiology(band='ap'), which is only there to report
    the AP sampling frequency. That value is unused by this conversion, but the call
    raises ALFObjectNotFound when a session's raw-ephys metadata is not in the cache,
    so dropping the call means a session is only ever skipped for a reason that
    actually affects the data.
    """
    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    if clusters is None:
        # No spike sorting exists for this insertion at all. The reference
        # load_spiking_data also fails on such an insertion (for these sessions the raw
        # ephys metadata it reads the sampling frequency from is missing too), and
        # 0_data_caching.py then skips the whole session; raised explicitly here so the
        # skip is reported for what it is.
        raise RuntimeError(f'no spike sorting for probe insertion {pid} ({pname})')
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=False).to_df()
    return spikes, clusters_labeled


def load_session(one, eid, n_workers):
    """Load, curate, align and bin one session. Returns a dict, or raises on failure."""
    # --- spikes: all probes of the session merged (reference prepare_data) -----------
    pids, probe_names = one.eid2pid(eid)
    if len(pids) == 0:
        raise RuntimeError('no probe insertions')
    spikes_list, clusters_list = [], []
    for pid, probe_name in zip(pids, probe_names):
        sp, cl = load_clusters_and_spikes(one, str(pid), eid, probe_name)
        spikes_list.append(sp)
        clusters_list.append(cl)
    spikes, clusters = U.merge_probes(spikes_list, clusters_list)
    neural_dict = {
        'spike_times': spikes['times'],
        'spike_clusters': spikes['clusters'],
        'cluster_regions': clusters['acronym'].to_numpy(),
    }

    # --- trials and the reference trial-quality mask ---------------------------------
    trials_df, trials_mask = U.load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0)
    trials_mask = np.asarray(trials_mask).astype(bool)

    # --- Beryl brain region of every cluster -----------------------------------------
    regions, beryl_reg = U.list_brain_regions(neural_dict, **PARAMS)
    reg_clu_ids = U.select_brain_regions(neural_dict, beryl_reg, regions[0], **PARAMS)

    # --- bin spikes into (ntrials, T, nneurons) --------------------------------------
    binned_spikes, clusters_used = U.bin_spiking_data(
        reg_clu_ids, neural_dict, trials_df=trials_df, n_workers=n_workers, **PARAMS)
    cluster_regions = np.asarray(beryl_reg)[clusters_used]
    if binned_spikes.shape[2] == 0:
        raise RuntimeError('no neurons')

    # --- behaviour availability ------------------------------------------------------
    # A handful of BWM sessions have no whisker motion energy at all (neither the left
    # nor the right camera ROIMotionEnergy exists). load_target_behavior returns
    # {'times': None, 'values': None, 'skip': True} for them and the reference
    # bin_behaviors then crashes inside np.searchsorted; 0_data_caching.py catches that
    # and skips the session. Whisker motion energy is a required decoder output here, so
    # such a session cannot be used either way - checked up front so the skip is
    # reported for what it is.
    for beh, targets in (('wheel-speed', ['wheel-speed']),
                         ('whisker-motion-energy', ['left-whisker-motion-energy',
                                                    'right-whisker-motion-energy'])):
        if all('skip' in U.load_target_behavior(one, eid, t) for t in targets):
            raise RuntimeError(f'{beh} is not available for this session')

    # --- bin the behavioural traces onto the same time grid --------------------------
    binned_beh, _ = U.bin_behaviors(
        one, eid, BEH_NAMES, trials_df=trials_df, allow_nans=True,
        n_workers=n_workers, **PARAMS)

    # --- keep trials passing the reference mask AND having usable behaviour ----------
    good = trials_mask.copy()
    for beh in BEH_NAMES:
        traces = binned_beh[beh]
        ok = np.array([
            (tr is not None) and (np.asarray(tr).size == NBINS)
            and bool(np.all(np.isfinite(np.asarray(tr, dtype=float))))
            for tr in traces])
        good &= ok
    keep = np.flatnonzero(good)
    if len(keep) < MIN_TRIALS:
        raise RuntimeError(f'only {len(keep)} trials survive curation')

    spk = binned_spikes[keep]                                 # (ntrials, T, nneurons)
    wheel = np.stack([np.asarray(binned_beh['wheel-speed'][i], dtype=float).ravel()
                      for i in keep])
    whisk = np.stack([np.asarray(binned_beh['whisker-motion-energy'][i],
                                 dtype=float).ravel() for i in keep])

    choice = trials_df['choice'].to_numpy()[keep]
    pleft = trials_df['probabilityLeft'].to_numpy()[keep]
    tinb = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())[keep]

    # IBL convention: choice == +1 -> leftward turn, choice == -1 -> rightward turn.
    choice_out = np.where(choice == -1, 1, 0).astype(np.int64)
    prior_out = np.array([PRIOR_MAP[round(float(p), 1)] for p in pleft], dtype=np.int64)
    wheel_out = discretize_tertiles(wheel)
    whisk_out = discretize_tertiles(whisk)

    # time grid = end of each 20 ms bin relative to stimulus onset (-0.48 ... 1.50 s),
    # the grid the reference code interpolates the behavioural traces onto.
    tgrid = (np.arange(1, NBINS + 1) * PARAMS['binsize']
             + PARAMS['time_window'][0]).astype(np.float32)

    neural, inputs, outputs = [], [], []
    for i in range(len(keep)):
        neural.append(np.ascontiguousarray(spk[i].T, dtype=np.float32))
        inputs.append(np.stack([tgrid, np.full(NBINS, tinb[i], dtype=np.float32)]))
        outputs.append(np.stack([
            np.full(NBINS, choice_out[i], dtype=np.int64),
            np.full(NBINS, prior_out[i], dtype=np.int64),
            wheel_out[i],
            whisk_out[i],
        ]))

    return {
        'neural': neural, 'input': inputs, 'output': outputs,
        'cluster_regions': cluster_regions,
        'n_trials_total': int(len(trials_df)),
        'n_trials_kept': int(len(keep)),
        'n_neurons': int(spk.shape[2]),
        'n_probes': int(len(pids)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_sessions', type=int, default=None)
    ap.add_argument('--n_workers', type=int, default=16)
    ap.add_argument('--out', type=str, default=OUT_PATH)
    args = ap.parse_args()

    one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True,
              cache_dir='/app/data/one_cache')

    bwm_df = pd.read_csv(BWM_RELEASE, index_col=0)
    # session order = order of first appearance in the public release freeze
    eids = list(dict.fromkeys(bwm_df.eid.tolist()))
    if os.path.exists(DATALIMIT):
        subset = pd.read_csv(DATALIMIT)
        col = 'eid' if 'eid' in subset.columns else subset.columns[0]
        allowed = set(subset[col].astype(str))
        eids = [e for e in eids if e in allowed]
        print(f'DATALIMIT_SUBSET.csv found: restricting to {len(eids)} sessions')
    if args.n_sessions is not None:
        eids = eids[:args.n_sessions]
    eid2subject = dict(zip(bwm_df.eid, bwm_df.subject))
    eid2lab = dict(zip(bwm_df.eid, bwm_df.lab))

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': [], 'subject_idx': [],
        'brain_regions': [], 'brain_region_idx': [],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {},
    }
    subjects, region_names, session_info, skipped = [], [], [], []

    for k, eid in enumerate(eids):
        print(f'=== [{k + 1}/{len(eids)}] {eid} ({eid2subject[eid]})', flush=True)
        try:
            sess = load_session(one, eid, args.n_workers)
        except Exception as exc:
            print(f'  SKIPPED {eid}: {exc!r}', flush=True)
            skipped.append({'eid': eid, 'reason': repr(exc)})
            continue

        sub = eid2subject[eid]
        if sub not in subjects:
            subjects.append(sub)

        data['neural'].append(sess['neural'])
        data['input'].append(sess['input'])
        data['output'].append(sess['output'])
        data['subject_idx'].append(subjects.index(sub))
        region_names.append(sess['cluster_regions'])
        session_info.append({
            'eid': eid, 'subject': sub, 'lab': eid2lab[eid],
            'n_probes': sess['n_probes'], 'n_neurons': sess['n_neurons'],
            'n_trials_total': sess['n_trials_total'],
            'n_trials_kept': sess['n_trials_kept'],
            'regions': sorted(set(sess['cluster_regions'].tolist())),
        })
        print(f"  kept {sess['n_trials_kept']}/{sess['n_trials_total']} trials, "
              f"{sess['n_neurons']} neurons", flush=True)

    brain_regions = sorted({r for names in region_names for r in names})
    lut = {r: i for i, r in enumerate(brain_regions)}
    data['brain_regions'] = brain_regions
    data['brain_region_idx'] = [np.array([lut[r] for r in names], dtype=np.int64)
                                for names in region_names]
    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    data['metadata'] = {
        'task_description': (
            'IBL decision-making task: a visual grating of one of five contrasts '
            '(100, 25, 12.5, 6.25, 0%) appears left or right of a screen and the mouse '
            'turns a wheel to bring it to the centre for a water reward. After 90 '
            'unbiased trials the prior probability that the stimulus appears on the left '
            'alternates between 0.2 and 0.8 in uncued blocks of 20-100 trials. From the '
            'binned spike counts of all recorded neurons the decoder predicts the '
            "mouse's choice (left/right), the block prior probability of left "
            '(0.2/0.5/0.8), and the wheel speed and whisker-pad motion energy, each '
            'discretised into 3 bins, in every 20 ms time bin.'),
        'time_bin_size': PARAMS['binsize'] * 1000.0,
        'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
        'off_start': float(PARAMS['time_window'][0]),
        'off_end': float(PARAMS['time_window'][1]),
        'n_timepoints': NBINS,
        'neural_units': 'spike counts per 20 ms bin',
        'dataset': ('IBL brain-wide map public release freeze '
                    '(code_zhang2025/data/bwm_release.csv)'),
        'spike_sorting': 'pykilosort, all clusters (no unit-quality threshold, qc=None)',
        'probes': 'all probes of a session merged as if recorded by a single probe',
        'region_mapping': 'Beryl (iblatlas.regions.BrainRegions, mapping="Beryl")',
        'trial_curation': (
            'load_trials_and_mask(max_trial_len=10.0) as in the reference prepare_data: '
            'reaction time (firstMovement_times - stimOn_times) in [0.08, 2] s, trial '
            'length (feedback_times - goCue_times) <= 10 s, no NaN in stimOn_times, '
            'choice, feedback_times, probabilityLeft, firstMovement_times or '
            'feedbackType, and choice != 0. Additionally trials whose wheel-speed or '
            'whisker-motion-energy trace does not cover the whole window or contains a '
            'NaN are dropped, and sessions with fewer than 2 remaining trials are '
            'dropped.'),
        'output_discretization': (
            'wheel speed and whisker motion energy are binned into 3 classes at the '
            'within-session 33.3rd and 66.7th percentiles of all retained samples'),
        'input_units': {
            'time_from_stim_onset': 'seconds (end of each 20 ms bin, -0.48 to 1.50)',
            'trial_number_in_block': 'trials elapsed since the start of the current '
                                     'probabilityLeft block (0-based)'},
        'reference_code': 'code_zhang2025/src/0_data_caching.py',
        'n_sessions': len(data['neural']),
        'session_info': session_info,
        'skipped_sessions': skipped,
    }

    ntr = sum(len(s) for s in data['neural'])
    nneu = sum(s[0].shape[0] for s in data['neural'] if len(s))
    print(f'\n{len(data["neural"])} sessions, {len(subjects)} subjects, '
          f'{ntr} trials, {nneu} neurons, {len(brain_regions)} regions')
    print(f'skipped {len(skipped)} sessions')

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
