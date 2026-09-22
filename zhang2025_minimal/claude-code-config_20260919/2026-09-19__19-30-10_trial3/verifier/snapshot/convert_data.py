"""Convert the IBL brain-wide map (BWM) dataset into the decoder's pickle format.

The processing follows the reference pipeline of Zhang et al. ("Exploiting
correlations across trials and behavioral sessions to improve neural decoding",
`/app/code/code_zhang2025`) and the inclusion criteria of the IBL brain-wide-map
data paper ("A brain-wide map of neural activity during complex behaviour").

Summary of the choices (see README-style notes next to each step):

  * Sessions      : the 459 eids of the BWM public release (`data/bwm_release.csv`),
                    i.e. the sessions that already passed the release criteria
                    (>=250 trials, >=90% correct on 100% contrast, hardware QC,
                    resolved histology alignment).
  * Insertions    : the released probe insertions for each eid, merged into a single
                    population per session (reference: `merge_probes`).
  * Neurons       : well-isolated units only (`label == 1`, i.e. the three RIGOR
                    single-unit metrics of the data paper: amplitude > 50 uV,
                    noise cut-off < 20 uV, refractory-period violation), restricted
                    to grey matter (Beryl acronym not in {root, void}).
  * Trials        : reference `load_trials_and_mask(..., max_trial_len=10.)` --
                    no NaN in stimOn_times / choice / feedback_times /
                    probabilityLeft / firstMovement_times / feedbackType,
                    0.08 s <= firstMovement - stimOn <= 2.0 s,
                    feedback - goCue <= 10 s, and a response was made.
                    Trials whose wheel / whisker traces do not cover the decoding
                    window (or are NaN in it) are dropped as well.
  * Alignment     : stimulus onset (`stimOn_times`), window (-0.5, 1.5) s,
                    20 ms non-overlapping bins -> 100 timepoints per trial.
                    These are exactly the reference caching parameters
                    (`src/0_data_caching.py`: interval_len 2, binsize 0.02,
                    align_time 'stimOn_times', time_window (-.5, 1.5)).
  * Behaviour     : wheel speed = |velocity| from the ibllib wheel loader and
                    whisker motion energy (left camera, right camera as fallback),
                    linearly interpolated onto the right edge of each 20 ms bin,
                    exactly as in the reference `get_behavior_per_interval`.

Usage:
    python convert_data.py [--out /app/converted_data.pkl] [--n-sessions N]
                           [--workers W]
"""

import argparse
import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONE_CACHE_DIR = '/app/data/one_cache'
ONE_BASE_URL = 'https://openalyx.internationalbrainlab.org'
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
# Present only in the reduced ("datalimit") variant of the task; when it exists the
# conversion is restricted to the subset of sessions that ships with it.
DATALIMIT_SUBSET_CSV = '/app/DATALIMIT_SUBSET.csv'

# Trial-aligned dataset parameters, identical to src/0_data_caching.py of the
# reference repository.
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100

# Reference trial-mask parameters (load_trials_and_mask defaults + max_trial_len=10).
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0

# Regions that the Beryl atlas mapping assigns to non-grey-matter / unassigned
# locations; the data paper restricts analyses to grey matter.
NON_GREY_MATTER = ('root', 'void')

# A session must retain at least this many trials to be usable: the decoder is
# trained on 80% of each session's trials and validated on the rest, so a session
# with a single trial cannot contribute to both splits.
MIN_TRIALS_PER_SESSION = 2

N_OUTPUT_BINS = 3  # tertiles for the two continuous behaviours

INPUT_NAMES = ['time_from_stimulus_onset', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p_left=0.2', 'p_left=0.5', 'p_left=0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]


# ---------------------------------------------------------------------------
# ONE helpers
# ---------------------------------------------------------------------------

_ONE = None


def get_one():
    """Return a (cached) ONE client pointed at the local cache.

    No password is passed: the container ships an auth token in ~/.one, and the
    Alyx host is unreachable, so building the client with a password would fail.
    """
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ONE_BASE_URL, silent=True, cache_dir=ONE_CACHE_DIR)
    return _ONE


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------

def load_trials_and_mask(one, eid):
    """Reference `load_trials_and_mask(one, eid, max_trial_len=10.)`.

    Reimplemented here only because the repository's copy calls
    `SessionLoader(one, eid)` positionally, which the installed ibllib (4.0.1) no
    longer accepts. The query and its defaults are unchanged.
    """
    from brainbox.io.one import SessionLoader

    nan_exclude = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
                   'firstMovement_times', 'feedbackType']

    sess_loader = SessionLoader(one=one, eid=eid)
    sess_loader.load_trials()
    trials = sess_loader.trials

    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'          # no response

    mask = ~trials.eval(query)
    return trials, mask.to_numpy().astype(bool)


def trial_number_in_block(trials):
    """0-based index of each trial within its block of constant probabilityLeft.

    Computed on the *full* trials table, before any trial is excluded, so that the
    value reflects the animal's actual position in the block rather than a position
    in the filtered sequence.
    """
    p_left = trials['probabilityLeft'].to_numpy(dtype=float)
    # NaN != NaN, so a NaN run is broken up; that is fine, those trials are excluded.
    new_block = np.ones(len(p_left), dtype=bool)
    new_block[1:] = ~(p_left[1:] == p_left[:-1])
    block_id = np.cumsum(new_block) - 1
    block_start = np.flatnonzero(new_block)
    idx_in_block = np.arange(len(p_left)) - block_start[block_id]
    return idx_in_block.astype(np.float32)


# ---------------------------------------------------------------------------
# Spikes
# ---------------------------------------------------------------------------

def load_session_units(one, eid, probe_rows, brain_regions):
    """Load, quality-filter and merge the released insertions of one session.

    Returns
    -------
    spike_times : (n_spikes,) float, sorted
    spike_units : (n_spikes,) int, index into the returned unit table
    acronyms    : (n_units,) str, Beryl acronym of each retained unit
    """
    from brainbox.io.one import SpikeSortingLoader

    times_list, units_list, acronym_list = [], [], []
    offset = 0
    for _, row in probe_rows.iterrows():
        ssl = SpikeSortingLoader(pid=str(row['pid']), one=one, eid=eid,
                                 pname=row['probe_name'])
        spikes, clusters, channels = ssl.load_spike_sorting()
        if len(spikes) == 0 or len(clusters) == 0:
            continue
        clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()

        beryl = np.asarray(brain_regions.acronym2acronym(
            clusters['acronym'].to_numpy(), mapping='Beryl'))

        # Data-paper inclusion criteria for neurons: well isolated (all three RIGOR
        # single-unit metrics passed, which ibllib encodes as label == 1) and located
        # in grey matter.
        keep = (clusters['label'].to_numpy() >= 1) & ~np.isin(beryl, NON_GREY_MATTER)
        keep_ids = np.flatnonzero(keep)
        if keep_ids.size == 0:
            continue

        # Map original cluster ids -> compact 0..n_keep-1 indices, dropping spikes of
        # discarded clusters.
        lut = np.full(int(clusters.shape[0]), -1, dtype=np.int64)
        lut[keep_ids] = np.arange(keep_ids.size) + offset
        sc = np.asarray(spikes['clusters'], dtype=np.int64)
        st = np.asarray(spikes['times'], dtype=np.float64)
        ok = (sc >= 0) & (sc < lut.size)
        sc, st = sc[ok], st[ok]
        mapped = lut[sc]
        ok = mapped >= 0
        # Spike times can contain NaN for samples outside the sync range.
        ok &= np.isfinite(st)

        times_list.append(st[ok])
        units_list.append(mapped[ok])
        acronym_list.append(beryl[keep_ids])
        offset += keep_ids.size

    if offset == 0:
        return None, None, None

    spike_times = np.concatenate(times_list)
    spike_units = np.concatenate(units_list)
    acronyms = np.concatenate(acronym_list)

    order = np.argsort(spike_times, kind='stable')
    return spike_times[order], spike_units[order], acronyms


def bin_spikes(spike_times, spike_units, n_units, interval_begs):
    """Spike counts in `N_BINS` non-overlapping `BINSIZE` bins after each interval start.

    Bin i of a trial covers [beg + i*BINSIZE, beg + (i+1)*BINSIZE), which is the
    binning the reference performs with `bincount2D(..., xbin=binsize,
    xlim=[t_beg, t_end])` followed by keeping the first `N_BINS` columns.
    """
    n_trials = len(interval_begs)
    out = np.zeros((n_trials, n_units, N_BINS), dtype=np.float32)
    interval_ends = interval_begs + N_BINS * BINSIZE
    i0 = np.searchsorted(spike_times, interval_begs, side='left')
    i1 = np.searchsorted(spike_times, interval_ends, side='left')
    flat = n_units * N_BINS
    for k in range(n_trials):
        t = spike_times[i0[k]:i1[k]]
        if t.size == 0:
            continue
        u = spike_units[i0[k]:i1[k]]
        b = np.floor((t - interval_begs[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, N_BINS - 1, out=b)
        counts = np.bincount(u * N_BINS + b, minlength=flat)
        out[k] = counts.reshape(n_units, N_BINS)
    return out


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

def load_behaviour_traces(one, eid):
    """Wheel speed and whisker motion energy for the whole session.

    Follows the reference `load_target_behavior` / `bin_behaviors`: wheel speed is
    the absolute value of the Gaussian-smoothed wheel velocity, whisker motion
    energy is taken from the left camera with the right camera as a fallback.
    """
    from brainbox.io.one import SessionLoader

    traces = {}

    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    traces['wheel-speed'] = (sl.wheel['times'].to_numpy(dtype=np.float64),
                             np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64)))

    whisker = None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sl_me = SessionLoader(one=one, eid=eid)
            sl_me.load_motion_energy(views=[view])
            df = sl_me.motion_energy[cam]
            whisker = (df['times'].to_numpy(dtype=np.float64),
                       df['whiskerMotionEnergy'].to_numpy(dtype=np.float64))
            break
        except Exception:
            continue
    if whisker is None:
        raise RuntimeError('no whisker motion energy available')
    traces['whisker-motion-energy'] = whisker
    return traces


def bin_behaviour(times, values, interval_begs):
    """Interpolate a behavioural trace onto the bins of each interval.

    The sample time of bin i is `beg + (i+1) * BINSIZE`, i.e. the right edge of the
    bin, matching the reference `get_behavior_per_interval`
    (`x_interp = linspace(beg + binsize, end, n_bins)`).

    Returns the binned values and a boolean mask of intervals for which the trace
    actually covers the window (within one bin at either end) -- the reference's
    'target data not present' / 'starts too late' / 'ends too early' checks.
    """
    n_trials = len(interval_begs)
    interval_ends = interval_begs + N_BINS * BINSIZE
    offsets = (np.arange(N_BINS) + 1) * BINSIZE
    sample_times = interval_begs[:, None] + offsets[None, :]

    binned = np.interp(sample_times.ravel(), times, values).reshape(n_trials, N_BINS)

    i0 = np.searchsorted(times, interval_begs, side='right')
    i1 = np.searchsorted(times, interval_ends, side='left')
    good = i1 > i0
    idx = np.flatnonzero(good)
    good[idx] &= np.abs(interval_begs[idx] - times[i0[idx]]) <= BINSIZE
    idx = np.flatnonzero(good)
    good[idx] &= np.abs(interval_ends[idx] - times[i1[idx] - 1]) <= BINSIZE
    good &= np.all(np.isfinite(binned), axis=1)
    return binned, good


def discretize_tertiles(values):
    """Map a (n_trials, N_BINS) trace to 3 equally-populated levels.

    Thresholds are the 1/3 and 2/3 quantiles of that session's own values. Both
    behaviours are in arbitrary, session-dependent units -- whisker motion energy
    depends on the camera (left at 60 Hz vs right at 150 Hz), the illumination and
    the size of the bounding box, and wheel speed on how vigorously that mouse
    turns -- so a global threshold would mostly encode which session a trial came
    from. Per-session tertiles make the three levels mean "slow / medium / fast for
    this animal" in every session and keep the classes balanced, which is what the
    balanced-accuracy score the decoder is graded with assumes.
    """
    lo, hi = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not hi > lo:  # degenerate (e.g. a mostly constant trace): keep bins ordered
        hi = np.nextafter(lo, np.inf)
    return np.digitize(values, [lo, hi]).astype(np.int64), (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Per-session conversion
# ---------------------------------------------------------------------------

def convert_session(eid, probe_rows, subject, lab):
    """Convert one session; returns a dict or None if the session is unusable."""
    from iblatlas.regions import BrainRegions

    one = get_one()
    brain_regions = BrainRegions()

    trials, trials_mask = load_trials_and_mask(one, eid)
    if trials_mask.sum() < MIN_TRIALS_PER_SESSION:
        return None, 'too few valid trials'

    align_times = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
    interval_begs_all = align_times + TIME_WINDOW[0]

    # Block identity, one of the three probabilities the task defines.
    p_left_all = trials['probabilityLeft'].to_numpy(dtype=float)
    prior_all = np.full(len(trials), -1, dtype=np.int64)
    for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
        prior_all[np.isclose(p_left_all, value)] = code

    keep = trials_mask & np.isfinite(interval_begs_all) & (prior_all >= 0)

    # --- behaviour, on all trials so the masks line up with the trials table ------
    traces = load_behaviour_traces(one, eid)
    beh_binned = {}
    safe_begs = np.where(np.isfinite(interval_begs_all), interval_begs_all, 0.0)
    for name, (t, v) in traces.items():
        binned, good = bin_behaviour(t, v, safe_begs)
        beh_binned[name] = binned
        keep &= good

    if keep.sum() < MIN_TRIALS_PER_SESSION:
        return None, 'too few trials with usable behaviour'

    # --- neural ------------------------------------------------------------------
    spike_times, spike_units, acronyms = load_session_units(
        one, eid, probe_rows, brain_regions)
    if spike_times is None:
        return None, 'no well-isolated grey-matter units'
    n_units = len(acronyms)

    interval_begs = interval_begs_all[keep]
    binned_spikes = bin_spikes(spike_times, spike_units, n_units, interval_begs)

    # --- inputs ------------------------------------------------------------------
    # Time since stimulus onset at the centre of each neural bin.
    bin_centres = TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE
    n_in_block = trial_number_in_block(trials)[keep]

    n_trials = int(keep.sum())
    inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
    inputs[:, 0, :] = bin_centres[None, :]
    inputs[:, 1, :] = n_in_block[:, None]

    # --- outputs -----------------------------------------------------------------
    # IBL codes choice as +1 when the mouse reported the stimulus on the LEFT
    # (verified: trials with contrastLeft > 0 and feedbackType == +1 all have
    # choice == +1) and -1 for a right report. The task asks for left = 0, right = 1.
    choice = (trials['choice'].to_numpy()[keep] == -1).astype(np.int64)
    prior = prior_all[keep]

    wheel_lvl, wheel_thr = discretize_tertiles(beh_binned['wheel-speed'][keep])
    whisk_lvl, whisk_thr = discretize_tertiles(beh_binned['whisker-motion-energy'][keep])

    outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)
    outputs[:, 0, :] = choice[:, None]
    outputs[:, 1, :] = prior[:, None]
    outputs[:, 2, :] = wheel_lvl
    outputs[:, 3, :] = whisk_lvl

    return {
        'eid': eid,
        'subject': subject,
        'lab': lab,
        'neural': [binned_spikes[k] for k in range(n_trials)],
        'input': [inputs[k] for k in range(n_trials)],
        'output': [outputs[k] for k in range(n_trials)],
        'acronyms': acronyms,
        'n_trials': n_trials,
        'n_trials_total': int(len(trials)),
        'n_units': int(n_units),
        'n_probes': int(len(probe_rows)),
        'wheel_speed_thresholds': wheel_thr,
        'whisker_motion_energy_thresholds': whisk_thr,
    }, None


def _worker(task):
    eid, probe_records, subject, lab = task
    warnings.filterwarnings('ignore')
    probe_rows = pd.DataFrame(probe_records)
    try:
        result, reason = convert_session(eid, probe_rows, subject, lab)
        return eid, result, reason
    except Exception as exc:  # noqa: BLE001 - a bad session must not stop the run
        return eid, None, f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/app/converted_data.pkl')
    ap.add_argument('--n-sessions', type=int, default=None,
                    help='process only the first N sessions (debugging)')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)

    eids = list(dict.fromkeys(bwm_df['eid'].tolist()))
    if os.path.exists(DATALIMIT_SUBSET_CSV):
        subset = pd.read_csv(DATALIMIT_SUBSET_CSV)
        col = 'eid' if 'eid' in subset.columns else subset.columns[0]
        allowed = set(subset[col].astype(str))
        eids = [e for e in eids if e in allowed]
        print(f'DATALIMIT subset present: restricting to {len(eids)} sessions')
    if args.n_sessions is not None:
        eids = eids[:args.n_sessions]
    print(f'Converting {len(eids)} sessions from the BWM release', flush=True)

    by_eid = bwm_df.groupby('eid')
    tasks = []
    for eid in eids:
        rows = by_eid.get_group(eid)
        tasks.append((eid, rows[['pid', 'probe_name']].to_dict('records'),
                      str(rows['subject'].iloc[0]), str(rows['lab'].iloc[0])))

    # Make sure the Allen atlas files are cached before the workers start, otherwise
    # every worker downloads them at once.
    from iblatlas.regions import BrainRegions
    BrainRegions()

    results, skipped = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, task): task[0] for task in tasks}
        for n, future in enumerate(as_completed(futures), start=1):
            eid, result, reason = future.result()
            if result is None:
                skipped.append((eid, reason))
                print(f'[{n}/{len(tasks)}] skipped {eid}: '
                      f'{str(reason).splitlines()[0]}', flush=True)
            else:
                results.append(result)
                print(f'[{n}/{len(tasks)}] {eid}: {result["n_trials"]} trials, '
                      f'{result["n_units"]} units  ({time.time() - t0:.0f}s)',
                      flush=True)

    # Deterministic session order, independent of completion order.
    results.sort(key=lambda r: r['eid'])

    subjects = sorted({r['subject'] for r in results})
    subject_index = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({a for r in results for a in np.unique(r['acronyms'])})
    region_index = {a: i for i, a in enumerate(brain_regions)}

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[r['subject']] for r in results],
                                dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([region_index[a] for a in r['acronyms']],
                                      dtype=np.int64) for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'IBL decision-making task: a Gabor patch of one of five contrasts '
                '(100, 25, 12.5, 6.25, 0%) appears at +/-35 degrees azimuth and the '
                'mouse turns a wheel to bring it to the centre for a water reward. '
                'After 90 unbiased trials (p(left) = 0.5) the stimulus side is drawn '
                'in blocks of 20-100 trials with p(left) = 0.2 or 0.8; block switches '
                'are uncued. Decoded from the spiking population of each session are '
                "the mouse's choice (left/right), the prior probability that the "
                'stimulus appears on the left (0.2 / 0.5 / 0.8, i.e. the block '
                'identity), and the wheel speed and whisker-pad motion energy, each '
                'discretised into three equally populated levels per session.'),
            'time_bin_size': BINSIZE * 1000.0,
            'temporal_alignment_event': 'stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': N_BINS,
            'neural_units': 'spike count per 20 ms bin',
            'dataset': 'International Brain Laboratory brain-wide map (public release)',
            'alignment_note': (
                'Neural bin i spans [stimOn - 0.5 + 0.02*i, stimOn - 0.5 + 0.02*(i+1)). '
                'time_from_stimulus_onset is the centre of that bin; wheel speed and '
                'whisker motion energy are linearly interpolated onto its right edge, '
                'as in the reference pipeline.'),
            'neuron_inclusion': (
                'Well-isolated units only (ibllib label == 1: amplitude > 50 uV, '
                'noise cut-off < 20 uV, refractory-period violation passed), located '
                'in grey matter (Beryl acronym not root/void). Units from all probe '
                'insertions of a session are merged into one population.'),
            'trial_inclusion': (
                'No NaN in stimOn_times, choice, feedback_times, probabilityLeft, '
                'firstMovement_times or feedbackType; 0.08 s <= firstMovement - '
                'stimOn <= 2.0 s; feedback - goCue <= 10 s; a response was made; and '
                'wheel and whisker traces cover the whole decoding window without NaN.'),
            'output_discretization': (
                'wheel_speed and whisker_motion_energy are split at the 1/3 and 2/3 '
                'quantiles of that session; thresholds are in session_info.'),
            'session_info': [
                {'eid': r['eid'], 'subject': r['subject'], 'lab': r['lab'],
                 'n_probes': r['n_probes'], 'n_neurons': r['n_units'],
                 'n_trials': r['n_trials'], 'n_trials_recorded': r['n_trials_total'],
                 'brain_regions': sorted(set(r['acronyms'].tolist())),
                 'wheel_speed_thresholds': r['wheel_speed_thresholds'],
                 'whisker_motion_energy_thresholds':
                     r['whisker_motion_energy_thresholds']}
                for r in results],
            'skipped_sessions': [{'eid': e, 'reason': str(m).splitlines()[0]}
                                 for e, m in sorted(skipped)],
        },
    }

    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = sum(len(idx) for idx in data['brain_region_idx'])
    print(f'\n{len(results)} sessions, {len(subjects)} subjects, {n_trials} trials, '
          f'{n_neurons} neurons, {len(brain_regions)} brain regions '
          f'({len(skipped)} sessions skipped)')

    with open(args.out, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.out} '
          f'({os.path.getsize(args.out) / 1e9:.2f} GB) in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    sys.exit(main())
