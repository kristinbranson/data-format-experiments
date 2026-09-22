"""Convert the IBL Brain-Wide Map dataset to the decoder-ready pickle format.

Data are read exclusively through the ONE API (`one.api.ONE`) with the brainbox loaders
(`SessionLoader`, `SpikeSortingLoader`), in `mode='remote'`, which resolves dataset revisions
from the REST responses cached in the staged ONE cache.

Processing follows the reference pipeline of Zhang et al. (`/app/code/code_zhang2025`,
`src/0_data_caching.py` + `src/utils/ibl_data_utils.py`) and the inclusion criteria of the IBL
Brain-Wide Map paper:

  * trials aligned to `stimOn_times`, window (-0.5, +1.5) s, 20 ms bins  -> T = 100
  * probes of a session merged into one population
  * only well-isolated units (IBL unit QC `label == 1`) in grey matter (Beryl region not void/root)
  * trials filtered with the reference `load_trials_and_mask` criteria
  * wheel speed = |wheel velocity|, whisker motion energy from the left camera (right as fallback),
    linearly interpolated onto the trial bin grid exactly as `get_behavior_per_interval` does

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""
import argparse
import os
import pickle
import time
import traceback
import multiprocessing as mp

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Parameters -- identical to the reference `params` in src/0_data_caching.py
# ----------------------------------------------------------------------------------
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to the alignment event
BINSIZE = 0.02                     # seconds
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100

MIN_RT, MAX_RT = 0.08, 2.0         # reference load_trials_and_mask defaults
MAX_TRIAL_LEN = 10.0               # reference prepare_data passes max_trial_len=10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

QC_LABEL = 1.0                     # IBL unit QC: 1 == passes all three RIGOR single-unit metrics
EXCLUDE_REGIONS = ('void', 'root')  # not grey matter / outside the brain
MIN_NEURONS = 5      # BWM paper: analyses require >= 5 well-isolated neurons
MIN_TRIALS = 2

BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
ONE_BASE_URL = 'https://openalyx.internationalbrainlab.org'

INPUT_NAMES = ['time_from_stim_on', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['0.2', '0.5', '0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]


def get_one():
    """ONE instance backed by the staged local cache (works fully offline)."""
    from one.api import ONE
    return ONE(base_url=ONE_BASE_URL, mode='remote')


# ----------------------------------------------------------------------------------
# Trial curation -- copy of reference `load_trials_and_mask` query (ibl_data_utils.py)
# ----------------------------------------------------------------------------------
def compute_trials_mask(trials, min_rt=MIN_RT, max_rt=MAX_RT, max_trial_len=MAX_TRIAL_LEN,
                        nan_exclude=NAN_EXCLUDE, exclude_nochoice=True):
    query = f'(firstMovement_times - stimOn_times < {min_rt})'
    query += f' | (firstMovement_times - stimOn_times > {max_rt})'
    if max_trial_len is not None:
        query += f' | (feedback_times - goCue_times > {max_trial_len})'
    for event in nan_exclude:
        query += f' | {event}.isnull()'
    if exclude_nochoice:
        query += ' | (choice == 0)'
    return ~trials.eval(query)


# ----------------------------------------------------------------------------------
# Neural binning -- equivalent to reference bin_spiking_data / bincount2D
# ----------------------------------------------------------------------------------
def bin_spikes(spike_times, spike_clusters, n_clusters, align_times,
               t_start=TIME_WINDOW[0], t_end=TIME_WINDOW[1], binsize=BINSIZE, n_bins=NBINS):
    """Spike counts in (n_trials, n_clusters, n_bins).

    Bin j of a trial covers [align + t_start + j*binsize, align + t_start + (j+1)*binsize),
    the same half-open convention as `iblutil.numerical.bincount2D(xlim=[t_beg, t_end])`.
    `spike_clusters` must already be re-indexed to 0..n_clusters-1.
    """
    n_trials = len(align_times)
    out = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    begs = align_times + t_start
    ends = align_times + t_end
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')
    for k in range(n_trials):
        s, e = i0[k], i1[k]
        if e <= s:
            continue
        b = ((spike_times[s:e] - begs[k]) / binsize).astype(np.int64)
        np.clip(b, 0, n_bins - 1, out=b)
        idx = spike_clusters[s:e].astype(np.int64) * n_bins + b
        cnt = np.bincount(idx, minlength=n_clusters * n_bins)
        out[k] = cnt.reshape(n_clusters, n_bins)
    return out


# ----------------------------------------------------------------------------------
# Behaviour binning -- equivalent to reference get_behavior_per_interval
# ----------------------------------------------------------------------------------
def bin_behavior(times, values, align_times, t_start=TIME_WINDOW[0], t_end=TIME_WINDOW[1],
                 binsize=BINSIZE, n_bins=NBINS):
    """Interpolate a continuous behavioural signal onto the trial bin grid.

    Returns (n_trials, n_bins) array and a boolean validity mask.  As in the reference, the
    interpolation grid is `np.linspace(beg + binsize, end, n_bins)` (the right edge of every bin)
    and a trial is invalid if the signal does not cover the window (starts too late / ends too
    early, by more than one bin) or contains NaNs inside the window.
    """
    begs = align_times + t_start
    ends = align_times + t_end
    n_trials = len(align_times)
    out = np.zeros((n_trials, n_bins), dtype=np.float64)
    valid = np.ones(n_trials, dtype=bool)

    idx_beg = np.searchsorted(times, begs, side='right')
    idx_end = np.searchsorted(times, ends, side='left')

    grid = (np.linspace(binsize, t_end - t_start, n_bins)[None, :] + begs[:, None])
    for k in range(n_trials):
        ib, ie = idx_beg[k], idx_end[k]
        if ie <= ib:
            valid[k] = False
            continue
        tt = times[ib:ie]
        vv = values[ib:ie]
        if np.abs(begs[k] - tt[0]) > binsize or np.abs(ends[k] - tt[-1]) > binsize:
            valid[k] = False      # target data starts too late / ends too early
            continue
        if np.any(np.isnan(vv)):
            valid[k] = False      # NaNs would break the decoder
            continue
        out[k] = np.interp(grid[k], tt, vv)
    return out, valid


def discretize_tertiles(x, valid_rows):
    """Discretize a (n_trials, n_bins) signal into 3 classes using per-session tertiles."""
    ref = x[valid_rows].ravel()
    q1, q2 = np.quantile(ref, [1. / 3., 2. / 3.])
    lab = np.zeros(x.shape, dtype=np.int64)
    lab[x > q1] = 1
    lab[x > q2] = 2
    return lab, (float(q1), float(q2))


def trial_number_in_block(prob_left):
    """0-based index of each trial within its block, computed on the raw trial sequence."""
    pl = np.asarray(prob_left, dtype=float)
    newblock = np.ones(len(pl), dtype=bool)
    newblock[1:] = pl[1:] != pl[:-1]
    idx = np.zeros(len(pl), dtype=np.int64)
    c = 0
    for i in range(len(pl)):
        c = 0 if newblock[i] else c + 1
        idx[i] = c
    return idx


# ----------------------------------------------------------------------------------
# Per-session conversion
# ----------------------------------------------------------------------------------
def convert_session(eid, show_processing=False, outdir='/app'):
    """Convert one session; returns a dict or None if the session is unusable."""
    from brainbox.io.one import SessionLoader, SpikeSortingLoader
    from iblatlas.regions import BrainRegions

    t_start_session = time.time()
    timing = {}
    one = get_one()
    br = BrainRegions()

    # ---------------- trials ----------------
    t0 = time.time()
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    trials = sl.trials
    mask = compute_trials_mask(trials).to_numpy()
    timing['trials'] = time.time() - t0
    if mask.sum() < MIN_TRIALS:
        return dict(eid=eid, skipped='too few trials after mask')

    # ---------------- spikes ----------------
    t0 = time.time()
    pids, pnames = one.eid2pid(eid)
    spike_times_l, spike_clu_l, acronyms_l, labels_l = [], [], [], []
    probe_t0, probe_t1 = [], []          # spike-sorted recording coverage per probe
    offset = 0
    for pid, pname in zip(pids, pnames):
        ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        # merge probes as in reference `merge_probes`: offset cluster ids so they stay unique
        probe_t0.append(np.nanmin(spikes['times']))
        probe_t1.append(np.nanmax(spikes['times']))
        spike_times_l.append(spikes['times'])
        spike_clu_l.append(spikes['clusters'] + offset)
        acronyms_l.append(clu['acronym'].to_numpy())
        labels_l.append(clu['label'].to_numpy())
        offset += int(clu.index.max()) + 1
    spike_times = np.concatenate(spike_times_l)
    spike_clusters = np.concatenate(spike_clu_l)
    acronyms = np.concatenate(acronyms_l)
    labels = np.concatenate(labels_l)
    srt = np.argsort(spike_times, kind='stable')
    spike_times = spike_times[srt]
    spike_clusters = spike_clusters[srt]
    timing['spikes'] = time.time() - t0

    # ---------------- neuron curation ----------------
    beryl = br.acronym2acronym(acronyms, mapping='Beryl')
    keep = (labels >= QC_LABEL) & (~np.isin(beryl, EXCLUDE_REGIONS))
    keep_ids = np.where(keep)[0]
    if len(keep_ids) < MIN_NEURONS:
        return dict(eid=eid, skipped='too few good neurons')
    remap = -np.ones(offset, dtype=np.int64)
    remap[keep_ids] = np.arange(len(keep_ids))
    sel = np.isin(spike_clusters, keep_ids)
    sp_t = spike_times[sel]
    sp_c = remap[spike_clusters[sel]]
    regions = beryl[keep_ids]

    # ---------------- behaviour ----------------
    t0 = time.time()
    sl.load_wheel()
    wheel_times = sl.wheel['times'].to_numpy()
    wheel_speed_raw = np.abs(sl.wheel['velocity'].to_numpy())

    me_times = me_vals = None
    camera_used = None
    for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sl.load_motion_energy(views=[view])
            df = sl.motion_energy[key]
            me_times = df['times'].to_numpy()
            me_vals = df['whiskerMotionEnergy'].to_numpy()
            camera_used = view
            break
        except Exception:
            continue
    if me_times is None:
        return dict(eid=eid, skipped='no whisker motion energy')
    timing['behavior_load'] = time.time() - t0

    # ---------------- bin everything on the masked trials ----------------
    t0 = time.time()
    align_all = trials[ALIGN_TIME].to_numpy()
    tnb_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())
    # A trial is only usable if its whole 2 s window is inside the spike-sorted recording of
    # every probe: in a few sessions the ephys recording stops before the behaviour does, which
    # would otherwise produce trials with no spikes at all.
    rec_t0, rec_t1 = max(probe_t0), min(probe_t1)
    covered = (align_all + TIME_WINDOW[0] >= rec_t0) & (align_all + TIME_WINDOW[1] <= rec_t1)
    mask = mask & covered
    if mask.sum() < MIN_TRIALS:
        return dict(eid=eid, skipped='too few trials inside the ephys recording')
    idx_trials = np.where(mask)[0]
    align = align_all[idx_trials]

    wheel_binned, wheel_valid = bin_behavior(wheel_times, wheel_speed_raw, align)
    me_binned, me_valid = bin_behavior(me_times, me_vals, align)
    good = wheel_valid & me_valid
    if good.sum() < MIN_TRIALS:
        return dict(eid=eid, skipped='too few trials with behaviour coverage')
    idx_trials = idx_trials[good]
    align = align[good]
    wheel_binned = wheel_binned[good]
    me_binned = me_binned[good]
    timing['behavior_bin'] = time.time() - t0

    t0 = time.time()
    binned_spikes = bin_spikes(sp_t, sp_c, len(keep_ids), align)
    timing['spike_bin'] = time.time() - t0

    # Drop trials in which not a single neuron fired anywhere in the 2 s window: they carry no
    # neural information (they occur in a few low-yield sessions and in silent stretches of a
    # recording) and the decoder flags them as degenerate.
    nonzero = binned_spikes.sum(axis=(1, 2)) > 0
    if nonzero.sum() < MIN_TRIALS:
        return dict(eid=eid, skipped='too few trials with any spikes')
    if not np.all(nonzero):
        binned_spikes = binned_spikes[nonzero]
        idx_trials = idx_trials[nonzero]
        align = align[nonzero]
        wheel_binned = wheel_binned[nonzero]
        me_binned = me_binned[nonzero]

    # ---------------- outputs ----------------
    allrows = np.ones(len(align), dtype=bool)
    wheel_lab, wheel_q = discretize_tertiles(wheel_binned, allrows)
    me_lab, me_q = discretize_tertiles(me_binned, allrows)

    choice_raw = trials['choice'].to_numpy()[idx_trials]         # +1 left, -1 right
    choice_lab = (choice_raw < 0).astype(np.int64)               # left -> 0, right -> 1
    pleft_raw = trials['probabilityLeft'].to_numpy()[idx_trials]
    prior_lab = np.select([pleft_raw == 0.2, pleft_raw == 0.5, pleft_raw == 0.8], [0, 1, 2],
                          default=-1).astype(np.int64)
    if np.any(prior_lab < 0):
        keep_pl = prior_lab >= 0
        idx_trials, align = idx_trials[keep_pl], align[keep_pl]
        binned_spikes = binned_spikes[keep_pl]
        wheel_lab, me_lab = wheel_lab[keep_pl], me_lab[keep_pl]
        wheel_binned, me_binned = wheel_binned[keep_pl], me_binned[keep_pl]
        choice_lab, prior_lab = choice_lab[keep_pl], prior_lab[keep_pl]
    n_trials = len(idx_trials)
    if n_trials < MIN_TRIALS:
        return dict(eid=eid, skipped='too few trials after all filters')

    # ---------------- inputs ----------------
    bin_centres = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)   # -0.49 ... 1.49
    tnb = tnb_all[idx_trials].astype(np.float32)

    neural, inputs, outputs = [], [], []
    for k in range(n_trials):
        neural.append(binned_spikes[k])
        inputs.append(np.stack([bin_centres.astype(np.float32),
                                np.full(NBINS, tnb[k], dtype=np.float32)], axis=0))
        outputs.append(np.stack([np.full(NBINS, choice_lab[k], dtype=np.int64),
                                 np.full(NBINS, prior_lab[k], dtype=np.int64),
                                 wheel_lab[k], me_lab[k]], axis=0))

    result = dict(
        eid=eid,
        neural=neural,
        input=inputs,
        output=outputs,
        regions=list(regions),
        n_trials=n_trials,
        n_neurons=len(keep_ids),
        camera=camera_used,
        wheel_quantiles=wheel_q,
        me_quantiles=me_q,
        trial_idx=idx_trials,
        n_trials_raw=len(trials),
        n_trials_mask=int(mask.sum()),
        timing=timing,
        total_time=time.time() - t_start_session,
    )

    if show_processing:
        try:
            plot_processing(eid, trials, mask, idx_trials, align, sp_t, sp_c,
                            binned_spikes, wheel_times, wheel_speed_raw, wheel_binned, wheel_lab,
                            wheel_q, me_times, me_vals, me_binned, me_lab, me_q,
                            bin_centres, tnb, choice_lab, prior_lab, outdir)
        except Exception:
            traceback.print_exc()
    return result


# ----------------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------------
def plot_processing(eid, trials, mask, idx_trials, align, sp_t, sp_c, binned_spikes,
                    wheel_times, wheel_raw, wheel_binned, wheel_lab, wheel_q,
                    me_times, me_raw, me_binned, me_lab, me_q,
                    bin_centres, tnb, choice_lab, prior_lab, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    k = min(3, len(align) - 1)             # example trial
    t0 = align[k] + TIME_WINDOW[0]
    t1 = align[k] + TIME_WINDOW[1]
    fig, ax = plt.subplots(4, 2, figsize=(18, 16))

    # 1. raw spike raster for the example trial with the bin grid
    sel = (sp_t >= t0) & (sp_t < t1)
    ax[0, 0].plot(sp_t[sel] - align[k], sp_c[sel], '.', ms=1)
    for e in np.arange(TIME_WINDOW[0], TIME_WINDOW[1] + 1e-9, 0.2):
        ax[0, 0].axvline(e, color='k', lw=0.2)
    ax[0, 0].axvline(0, color='r')
    ax[0, 0].set_title(f'{eid}\nraw spikes, trial {k} (red = stimOn)')
    ax[0, 0].set_xlabel('time from stim on (s)'); ax[0, 0].set_ylabel('neuron')

    # 2. binned spikes for the same trial
    im = ax[0, 1].imshow(binned_spikes[k], aspect='auto', interpolation='none',
                         extent=[TIME_WINDOW[0], TIME_WINDOW[1], binned_spikes.shape[1], 0])
    ax[0, 1].axvline(0, color='r')
    ax[0, 1].set_title('binned spike counts (20 ms bins)')
    ax[0, 1].set_xlabel('time from stim on (s)'); ax[0, 1].set_ylabel('neuron')
    plt.colorbar(im, ax=ax[0, 1])

    # 3. wheel speed: raw vs interpolated vs discretized
    sel = (wheel_times >= t0) & (wheel_times < t1)
    ax[1, 0].plot(wheel_times[sel] - align[k], wheel_raw[sel], 'k-', label='raw |velocity|')
    ax[1, 0].plot(bin_centres + 0.5 * BINSIZE, wheel_binned[k], 'o-', ms=3, label='interpolated')
    ax[1, 0].axhline(wheel_q[0], color='g', ls='--', label='tertile 1')
    ax[1, 0].axhline(wheel_q[1], color='m', ls='--', label='tertile 2')
    ax[1, 0].axvline(0, color='r'); ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title('wheel speed, trial %d' % k); ax[1, 0].set_xlabel('time from stim on (s)')
    ax[1, 1].step(bin_centres, wheel_lab[k], where='mid')
    ax[1, 1].set_title('wheel speed class (0/1/2)'); ax[1, 1].set_xlabel('time from stim on (s)')

    # 4. whisker ME: raw vs interpolated vs discretized
    sel = (me_times >= t0) & (me_times < t1)
    ax[2, 0].plot(me_times[sel] - align[k], me_raw[sel], 'k-', label='raw ME')
    ax[2, 0].plot(bin_centres + 0.5 * BINSIZE, me_binned[k], 'o-', ms=3, label='interpolated')
    ax[2, 0].axhline(me_q[0], color='g', ls='--'); ax[2, 0].axhline(me_q[1], color='m', ls='--')
    ax[2, 0].axvline(0, color='r'); ax[2, 0].legend(fontsize=7)
    ax[2, 0].set_title('whisker motion energy, trial %d' % k)
    ax[2, 1].step(bin_centres, me_lab[k], where='mid')
    ax[2, 1].set_title('whisker ME class (0/1/2)')

    # 5. inputs / per-trial outputs across trials
    ax[3, 0].plot(tnb, '.-', ms=3)
    ax[3, 0].set_title('input 1: trial number in block (retained trials)')
    ax[3, 0].set_xlabel('trial')
    pl = trials['probabilityLeft'].to_numpy()[idx_trials]
    ax[3, 1].plot(pl, 'k.-', ms=3, label='probabilityLeft')
    ax[3, 1].plot(prior_lab / 2. * 0.6 + 0.2, 'r.', ms=3, label='prior class (rescaled)')
    ax[3, 1].plot(choice_lab * 0.1 + 0.05, 'b.', ms=3, label='choice class (rescaled)')
    ax[3, 1].legend(fontsize=7); ax[3, 1].set_title('outputs: prior and choice')
    ax[3, 1].set_xlabel('trial')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'processing_{eid}.png'), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------
def _worker(args):
    eid, show = args
    try:
        return convert_session(eid, show_processing=show)
    except Exception as e:
        traceback.print_exc()
        return dict(eid=eid, skipped=f'exception: {e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--n-workers', type=int, default=12)
    args = ap.parse_args()

    bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    eid2subject = bwm.groupby('eid').subject.first().to_dict()
    eid2lab = bwm.groupby('eid').lab.first().to_dict()
    eids = list(bwm.eid.unique())
    if args.sample:
        eids = eids[:2]
    print(f'Converting {len(eids)} sessions', flush=True)

    t0 = time.time()
    results = []
    if args.show_processing or len(eids) <= 2:
        for i, eid in enumerate(eids):
            r = _worker((eid, args.show_processing and i < 2))
            results.append(r)
            print(f'[{i+1}/{len(eids)}] {eid} '
                  + (r.get('skipped') or f"ntrials={r['n_trials']} nneurons={r['n_neurons']} "
                     f"time={r['total_time']:.1f}s timing={ {k: round(v,2) for k,v in r['timing'].items()} }"),
                  flush=True)
    else:
        with mp.Pool(args.n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, [(e, False) for e in eids])):
                results.append(r)
                if (i + 1) % 20 == 0 or i == 0:
                    el = time.time() - t0
                    print(f'[{i+1}/{len(eids)}] elapsed {el:.0f}s, '
                          f'est total {el/(i+1)*len(eids):.0f}s', flush=True)

    ok = [r for r in results if 'neural' in r]
    skipped = [r for r in results if 'neural' not in r]
    print(f'\nConverted {len(ok)} sessions, skipped {len(skipped)}')
    for r in skipped:
        print('  skipped', r['eid'], r.get('skipped'))

    ok.sort(key=lambda r: r['eid'])

    subjects = sorted({eid2subject[r['eid']] for r in ok})
    subj_index = {s: i for i, s in enumerate(subjects)}
    all_regions = sorted({reg for r in ok for reg in r['regions']})
    reg_index = {s: i for i, s in enumerate(all_regions)}

    data = {
        'neural': [r['neural'] for r in ok],
        'input': [r['input'] for r in ok],
        'output': [r['output'] for r in ok],
        'subjects': subjects,
        'subject_idx': np.array([subj_index[eid2subject[r['eid']]] for r in ok], dtype=np.int64),
        'brain_regions': all_regions,
        'brain_region_idx': [np.array([reg_index[x] for x in r['regions']], dtype=np.int64)
                             for r in ok],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'IBL decision-making task (Brain-Wide Map). Mice turn a wheel to move a visual '
                'grating of varying contrast, appearing on the left or right, to the centre of a '
                'screen. Stimulus side probability is constant within blocks of trials '
                '(0.2/0.5/0.8 probability of left). Decoded outputs: the animal\'s choice '
                '(left/right), the block prior probability of a left stimulus (0.2/0.5/0.8), '
                'wheel speed discretized into 3 per-session tertile bins, and whisker motion '
                'energy discretized into 3 per-session tertile bins.'),
            'time_bin_size': BINSIZE * 1000.,
            'temporal_alignment_event': 'stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': NBINS,
            'neural_data_type': 'spike counts per 20 ms bin (not normalised)',
            'neuron_curation': ('IBL unit QC label == 1 (well-isolated: amplitude > 50 uV, '
                                'noise cut-off < 20 uV, refractory-period violation passed); '
                                'Beryl region not void/root; probes of a session merged'),
            'trial_curation': ('reference load_trials_and_mask: no NaN in stimOn_times, choice, '
                               'feedback_times, probabilityLeft, firstMovement_times, feedbackType; '
                               '0.08 s <= firstMovement_times - stimOn_times <= 2 s; '
                               'feedback_times - goCue_times <= 10 s; choice != 0; plus full '
                               'coverage of the 2 s window by the wheel and whisker traces'),
            'input_descriptions': [
                'time of the bin centre relative to stimulus onset, in seconds (-0.49 ... 1.49)',
                'index of the trial within its block (0-based, counted over all raw trials)'],
            'output_descriptions': [
                'choice: 0 = left (trials.choice == +1), 1 = right (trials.choice == -1)',
                'prior probability of left (trials.probabilityLeft): 0.2 -> 0, 0.5 -> 1, 0.8 -> 2',
                'wheel speed |velocity| discretized by per-session tertiles: 0 low, 1 medium, 2 high',
                'whisker motion energy (left camera, right as fallback) discretized by per-session '
                'tertiles: 0 low, 1 medium, 2 high'],
            'session_info': [
                dict(eid=r['eid'], subject=eid2subject[r['eid']], lab=eid2lab[r['eid']],
                     n_trials=r['n_trials'], n_neurons=r['n_neurons'], camera=r['camera'],
                     n_trials_raw=r['n_trials_raw'], n_trials_after_mask=r['n_trials_mask'],
                     wheel_tertiles=r['wheel_quantiles'], me_tertiles=r['me_quantiles'])
                for r in ok],
            'source': ('IBL Brain-Wide Map public data release, loaded with the ONE API; '
                       'processing follows Zhang et al. src/0_data_caching.py '
                       '(align stimOn_times, window (-0.5, 1.5) s, 20 ms bins)'),
        },
    }

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    ntr = [len(x) for x in data['neural']]
    nn = [x.shape[0] for x in [s[0] for s in data['neural']]]
    print(f'\nSaved {args.outfile}: {len(ntr)} sessions, {sum(ntr)} trials, '
          f'{sum(nn)} neurons, {len(subjects)} subjects, {len(all_regions)} regions')
    print(f'Total conversion time {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
