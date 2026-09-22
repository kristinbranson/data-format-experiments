#!/usr/bin/env python3
"""
Convert the Mesoscale Activity Map dataset (DANDI:000363, Chen et al. 2024) from NWB
into the decoder-ready pickle format.

Processing follows the reference pipeline of Wang, Kurgyis et al. 2025
(`/app/code`, module `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` and
`Sherlock/align_markers.py`):

  * only units passing the spike-sorting quality-control classifier are used
    (`units.classification == 'good'`, equivalent to the reference `goodunits` lists),
  * everything is aligned to the **go cue**,
  * spike trains are binned into firing rates (spikes / s) like `sliding_histogram(..., rate=True)`,
  * neurons are labelled by hemisphere (CCF ML midline at 5700 um) + coarse CCF region,
    exactly as in `helper_get_neuron_id_area`,
  * side-view DeepLabCut markers (300 Hz) are aligned to the go cue as in `align_markers.py`.

Deviations required by the decoder task are documented in /app/CONVERSION_NOTES.md.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import sys
import time
import json
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------- config
DATA_DIR = '/app/data'

T_START = -2.5           # s relative to go cue (task specification)
T_END = 1.5              # s relative to go cue
BIN_SIZE = 0.05          # s (task specification: 50 ms bins)
N_BINS = int(round((T_END - T_START) / BIN_SIZE))          # 80
BIN_EDGES = T_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2

VIDEO_DT = 0.0034        # s, 300 Hz side-view camera (reference `align_markers.py`)
LIKELIHOOD_THRESH = 0.9  # DLC likelihood above which the tongue counts as visible
MIN_VIDEO_COVERAGE = 0.9 # fraction of the window that must contain video frames
MIN_SESSION_VIDEO_FRAC = 0.5  # a session must have video for at least this fraction of trials
ML_MIDLINE = 5700.0      # um, CCF ML midline (reference `helper_get_neuron_id_area`)

# session selection criteria of the data paper (Chen et al., STAR Methods)
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50

INPUT_NAMES = ['time_from_tone_onset', 'photostim_on']
OUTPUT_NAMES = ['lick_direction_choice', 'outcome', 'early_lick', 'tongue_y_position']
OUTPUT_VALUES = [['left', 'right', 'no lick'],
                 ['ignore', 'miss', 'hit'],
                 ['no', 'yes'],
                 ['<40th pct', '40-60th pct', '>60th pct', 'not visible']]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_map import coarse_region     # noqa: E402


# ------------------------------------------------------------------- helper functions
def bin_spike_times(spike_times_rel, n_neurons):
    """Bin go-cue-aligned spike times into firing rates.

    Equivalent to the reference `sliding_histogram(..., rate=True)` with
    bin_width == stride == BIN_SIZE.

    Args:
        spike_times_rel: list (len n_neurons) of 1-D arrays of spike times relative to
            the go cue, already restricted to [T_START, T_END].
        n_neurons: number of neurons.
    Returns:
        (n_neurons, N_BINS) float32 array of firing rates in spikes / s.
    """
    out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
    for i, st in enumerate(spike_times_rel):
        if st.size:
            idx = np.floor((st - T_START) / BIN_SIZE).astype(np.int64)
            np.clip(idx, 0, N_BINS - 1, out=idx)
            out[i] = np.bincount(idx, minlength=N_BINS).astype(np.float32)
    out /= BIN_SIZE
    return out


def choice_from_trial(outcome, instruction):
    """Lick direction the animal actually chose: 0 left, 1 right, 2 no lick."""
    if outcome == 'ignore':
        return 2
    right = (instruction == 'right')
    if outcome == 'miss':          # error trial -> licked the other spout
        right = not right
    return 1 if right else 0


OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}


def process_session(args):
    """Load, curate and convert one NWB session. Returns a dict or None."""
    path, show_processing, plot_dir = args
    from pynwb import NWBHDF5IO

    t_open = time.time()
    timings = {}
    with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
        nwb = io.read()
        sess_name = nwb.identifier
        subject = nwb.subject.description or nwb.subject.subject_id
        timings['open'] = time.time() - t_open

        # ------------------------------------------------ trials table & task events
        t0 = time.time()
        trials = nwb.trials
        n_trials = len(trials)
        start_time = np.asarray(trials['start_time'].data[:])
        stop_time = np.asarray(trials['stop_time'].data[:])
        outcome = np.asarray(trials['outcome'].data[:])
        instruction = np.asarray(trials['trial_instruction'].data[:])
        early_lick = np.asarray(trials['early_lick'].data[:])
        auto_water = np.asarray(trials['auto_water'].data[:])
        free_water = np.asarray(trials['free_water'].data[:])

        bev = nwb.acquisition['BehavioralEvents'].time_series
        go_times_all = np.asarray(bev['go_start_times'].timestamps[:])
        sample_times = np.asarray(bev['sample_start_times'].timestamps[:])
        photostim_on = np.asarray(bev['photostim_start_times'].timestamps[:])
        photostim_off = np.asarray(bev['photostim_stop_times'].timestamps[:])
        timings['trials'] = time.time() - t0

        # one go cue per trial (verified for every session in the dataset)
        go = np.full(n_trials, np.nan)
        gi = np.searchsorted(start_time, go_times_all, side='right') - 1
        ok = (gi >= 0) & (gi < n_trials)
        go[gi[ok]] = go_times_all[ok]
        if np.any(np.isnan(go)):
            return dict(sess=sess_name, skipped='missing go cue')

        # --------------------------------------------- session-level behaviour checks
        hit = outcome == 'hit'
        miss = outcome == 'miss'
        ctrl = ((early_lick == 'no early') & (auto_water == 0) & (free_water == 0)
                & ~np.isin(np.arange(n_trials), np.searchsorted(start_time, photostim_on, side='right') - 1))
        responded = (hit | miss) & ctrl
        performance = hit[responded].sum() / max(responded.sum(), 1)
        n_correct_left = int((hit & ctrl & (instruction == 'left')).sum())
        n_correct_right = int((hit & ctrl & (instruction == 'right')).sum())

        # ------------------------------------------------------------- unit curation
        t0 = time.time()
        units = nwb.units
        classification = np.asarray(units['classification'].data[:])
        good = np.where(classification == 'good')[0]
        n_good = len(good)
        info = dict(sess=sess_name, subject=subject, n_trials=n_trials, n_good=n_good,
                    performance=float(performance), n_correct_left=n_correct_left,
                    n_correct_right=n_correct_right)
        if n_good == 0:
            info['skipped'] = 'no good units'
            return info
        if not (performance > MIN_PERFORMANCE
                and n_correct_left >= MIN_CORRECT_PER_DIRECTION
                and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
            info['skipped'] = 'behavioural session criteria'
            return info

        anno = np.asarray(units['anno_name'].data[:])[good]
        electrodes = nwb.electrodes
        erows = np.asarray(units['electrodes'].target.data[:])[good]
        ex = np.asarray(electrodes['x'].data[:])[erows]
        egroup = np.asarray(electrodes['group_name'].data[:])[erows]
        target_of_group = {}
        for gname, grp in nwb.electrode_groups.items():
            try:
                target_of_group[gname] = json.loads(grp.location)['brain_regions']
            except Exception:
                target_of_group[gname] = ''
        region_labels = []
        for a, x, g in zip(anno, ex, egroup):
            reg = coarse_region(str(a), target_of_group.get(g, ''))
            side = 'left' if (np.isfinite(x) and x >= ML_MIDLINE) else 'right'
            region_labels.append(f'{side} {reg}')
        timings['units_meta'] = time.time() - t0

        # --------------------------------------------------------------- video (tongue)
        t0 = time.time()
        bts = nwb.acquisition['BehavioralTimeSeries'].time_series
        tongue = bts['Camera0_side_TongueTracking']
        vtime = np.asarray(tongue.timestamps[:])
        vdata = np.asarray(tongue.data[:])            # (n_frames, 3): x, y, likelihood
        tongue_y = vdata[:, 1]
        tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
        timings['video_read'] = time.time() - t0

        # ---------------------------------------------------------------- spike times
        t0 = time.time()
        spikes = [np.asarray(units['spike_times'][int(i)]) for i in good]
        timings['spikes_read'] = time.time() - t0

        # ------------------------------------------------- observed (recorded) trials
        # Spikes are only stored inside the NWB trial intervals and each unit carries
        # `obs_intervals` listing the trials during which it was recorded. In 8 sessions
        # the ephys covers only part of the behavioural session, so trials outside
        # `obs_intervals` contain no neural data at all and must be dropped.
        t0 = time.time()
        observed = np.ones(n_trials, dtype=bool)
        for i in good:
            obs = np.asarray(units['obs_intervals'][int(i)])
            oi = np.searchsorted(start_time, obs[:, 0], side='right') - 1
            oi = oi[(oi >= 0) & (oi < n_trials)]
            m = np.zeros(n_trials, dtype=bool)
            m[oi] = True
            observed &= m
        timings['obs_intervals'] = time.time() - t0
        info['n_trials_observed'] = int(observed.sum())

        # ------------------------------------------------------------- per-trial loop
        t0 = time.time()
        # tone onset = last sample-epoch onset before the go cue (epoch is replayed
        # after an early lick, so the last one is the instructing tone)
        tone_onset = np.full(n_trials, np.nan)
        si = np.searchsorted(start_time, sample_times, side='right') - 1
        for tr_i, s in zip(si, sample_times):
            if 0 <= tr_i < n_trials and s < go[tr_i]:
                if np.isnan(tone_onset[tr_i]) or s > tone_onset[tr_i]:
                    tone_onset[tr_i] = s

        # photostim windows per trial (relative to the go cue)
        stim_on = np.full(n_trials, np.nan)
        stim_off = np.full(n_trials, np.nan)
        pi = np.searchsorted(start_time, photostim_on, side='right') - 1
        for tr_i, on, off in zip(pi, photostim_on, photostim_off):
            if 0 <= tr_i < n_trials:
                stim_on[tr_i] = on - go[tr_i]
                stim_off[tr_i] = off - go[tr_i]

        win0 = go + T_START
        win1 = go + T_END

        # video coverage of the window, used for trial curation
        vlo = np.searchsorted(vtime, win0)
        vhi = np.searchsorted(vtime, win1)
        coverage = (vhi - vlo) / ((T_END - T_START) / VIDEO_DT)

        # ------------------------------------------------- session-level video check
        # In 6 sessions the side-view video stops at (or before) the go cue, so the
        # tongue output would be undefined for the whole response period.
        if np.mean(coverage[observed] >= MIN_VIDEO_COVERAGE) < MIN_SESSION_VIDEO_FRAC:
            info['skipped'] = 'video does not cover the analysis window'
            return info

        # ------------------------------------------------- trial selection mask
        keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
        n_dropped_unobserved = int((~observed).sum())
        n_dropped_video = int((observed & (coverage < MIN_VIDEO_COVERAGE)).sum())
        keep_trials = np.where(keep_mask)[0]
        if len(keep_trials) == 0:
            info['skipped'] = 'no usable trials'
            return info
        gk = go[keep_trials]
        nk = len(keep_trials)

        # ------------------------------------------------- neural: vectorised binning
        # For every unit, all trial windows are extracted at once with searchsorted and
        # binned with a single bincount (equivalent to the reference sliding_histogram
        # with bin_width == stride == BIN_SIZE and rate=True).
        neural_all = np.zeros((nk, n_good, N_BINS), dtype=np.float32)
        w0 = gk + T_START
        w1 = gk + T_END
        for ui, st in enumerate(spikes):
            if st.size == 0:
                continue
            lo = np.searchsorted(st, w0)
            hi = np.searchsorted(st, w1)
            cnt = hi - lo
            tot = int(cnt.sum())
            if tot == 0:
                continue
            trial_ids = np.repeat(np.arange(nk), cnt)
            offsets = np.repeat(lo - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
            pos = np.arange(tot) + offsets
            rel = st[pos] - gk[trial_ids]
            bidx = np.floor((rel - T_START) / BIN_SIZE).astype(np.int64)
            np.clip(bidx, 0, N_BINS - 1, out=bidx)
            flat = trial_ids * N_BINS + bidx
            counts = np.bincount(flat, minlength=nk * N_BINS).reshape(nk, N_BINS)
            neural_all[:, ui, :] = counts
        neural_all /= BIN_SIZE      # spikes / s

        # trials without a single spike from any good unit: recording interrupted
        nonempty = neural_all.any(axis=(1, 2))
        n_dropped_empty = int((~nonempty).sum())
        keep_trials = keep_trials[nonempty]
        neural_all = neural_all[nonempty]
        gk = gk[nonempty]
        nk = len(keep_trials)
        if nk < 2:
            info['skipped'] = 'fewer than 2 usable trials'
            return info

        # ------------------------------------------------------------------- inputs
        tone_rel = tone_onset[keep_trials] - gk
        input_all = np.zeros((nk, 2, N_BINS), dtype=np.float32)
        input_all[:, 0, :] = (BIN_CENTERS[None, :] - tone_rel[:, None]).astype(np.float32)
        son = stim_on[keep_trials]
        soff = stim_off[keep_trials]
        has_stim = np.isfinite(son)
        if has_stim.any():
            input_all[has_stim, 1, :] = (
                (BIN_CENTERS[None, :] >= son[has_stim][:, None])
                & (BIN_CENTERS[None, :] <= soff[has_stim][:, None])).astype(np.float32)

        # ------------------------------------------------- tongue y position per bin
        tongue_bin_y = np.full((nk, N_BINS), np.nan)
        for k in range(nk):
            g = gk[k]
            lo = np.searchsorted(vtime, g + T_START)
            hi = np.searchsorted(vtime, g + T_END)
            if hi <= lo:
                continue
            sel = tongue_vis[lo:hi]
            if not sel.any():
                continue
            vt = vtime[lo:hi][sel] - g
            vy = tongue_y[lo:hi][sel]
            bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
            np.clip(bidx, 0, N_BINS - 1, out=bidx)
            sums = np.bincount(bidx, weights=vy, minlength=N_BINS)
            cnts = np.bincount(bidx, minlength=N_BINS)
            nz = cnts > 0
            tongue_bin_y[k, nz] = sums[nz] / cnts[nz]

        # how much of the window lies after the end of the recorded trial interval
        n_unobserved = int(np.sum(BIN_CENTERS[None, :] > (stop_time[keep_trials] - gk)[:, None]))

        neural_trials = [neural_all[k] for k in range(nk)]
        input_trials = [input_all[k] for k in range(nk)]
        timings['bin_trials'] = time.time() - t0


        # ------------------------------------------- per-session tongue discretization
        visible = np.isfinite(tongue_bin_y)
        if visible.sum() >= 10:
            p40, p60 = np.percentile(tongue_bin_y[visible], [40, 60])
        else:
            p40 = p60 = np.nan
        tongue_class = np.full(tongue_bin_y.shape, 3, dtype=np.int64)   # 3 = not visible
        if np.isfinite(p40):
            tongue_class[visible & (tongue_bin_y < p40)] = 0
            tongue_class[visible & (tongue_bin_y >= p40) & (tongue_bin_y <= p60)] = 1
            tongue_class[visible & (tongue_bin_y > p60)] = 2

        # --------------------------------------------------------------- outputs
        output_trials = []
        for k, tr in enumerate(keep_trials):
            out = np.zeros((4, N_BINS), dtype=np.int64)
            out[0] = choice_from_trial(outcome[tr], instruction[tr])
            out[1] = OUTCOME_CODE[outcome[tr]]
            out[2] = 1 if early_lick[tr] == 'early' else 0
            out[3] = tongue_class[k]
            output_trials.append(out)

        res = dict(
            sess=sess_name, subject=subject, path=path,
            neural=neural_trials, input=input_trials, output=output_trials,
            region_labels=region_labels,
            n_trials_total=n_trials, n_trials_kept=len(keep_trials), n_good=n_good,
            performance=float(performance), n_correct_left=n_correct_left,
            n_correct_right=n_correct_right,
            trial_idx=keep_trials,
            frac_unobserved_bins=float(n_unobserved / (len(keep_trials) * N_BINS)),
            tongue_p40=float(p40), tongue_p60=float(p60),
            n_photostim=int(np.isfinite(stim_on[keep_trials]).sum()),
            n_trials_observed=int(observed.sum()),
            n_dropped_video=int(n_dropped_video), n_dropped_unobserved=int(n_dropped_unobserved),
            n_dropped_empty=int(n_dropped_empty),
            n_auto_water=int(auto_water[keep_trials].sum()),
            n_free_water=int(free_water[keep_trials].sum()),
            timings=timings,
        )

        if show_processing:
            plot_session(nwb, res, go, tone_onset, stim_on, stim_off, vtime=vtime,
                         tongue_y=tongue_y, tongue_vis=tongue_vis, spikes=spikes,
                         tongue_bin_y=tongue_bin_y, plot_dir=plot_dir)
        return res


# ------------------------------------------------------------------------- plotting
def plot_session(nwb, res, go, tone_onset, stim_on, stim_off, vtime, tongue_y,
                 tongue_vis, spikes, tongue_bin_y, plot_dir):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sess = res['sess']
    keep = res['trial_idx']
    ntr_plot = min(3, len(keep))
    # prefer trials with photostim / visible tongue for the illustration
    order = np.argsort(-np.array([res['output'][k][3].min() < 3 for k in range(len(keep))], dtype=float))
    sel = order[:ntr_plot]

    fig, axes = plt.subplots(5, ntr_plot, figsize=(6 * ntr_plot, 16), squeeze=False)
    for j, k in enumerate(sel):
        tr = keep[k]
        g = go[tr]
        # 1. spike raster (raw times minus go cue) for up to 40 neurons
        ax = axes[0, j]
        for i, st in enumerate(spikes[:40]):
            lo = np.searchsorted(st, g + T_START); hi = np.searchsorted(st, g + T_END)
            s = st[lo:hi] - g
            ax.plot(s, np.full(s.size, i), '|', color='k', markersize=2)
        ax.axvline(0, color='r', label='go cue')
        ax.axvline(tone_onset[tr] - g, color='g', label='tone onset')
        if np.isfinite(stim_on[tr]):
            ax.axvspan(stim_on[tr], stim_off[tr], color='b', alpha=0.2, label='photostim')
        ax.set_title(f'{sess}\ntrial {tr}: raw spikes aligned to go cue')
        ax.set_xlim(T_START, T_END); ax.legend(fontsize=7)
        ax.set_ylabel('neuron')

        # 2. binned firing rates
        ax = axes[1, j]
        im = ax.imshow(res['neural'][k][:40], aspect='auto', origin='lower',
                       extent=[T_START, T_END, 0, min(40, res['n_good'])], cmap='viridis')
        ax.axvline(0, color='r')
        ax.set_title('binned firing rate (spikes/s), 50 ms bins')
        ax.set_ylabel('neuron')
        plt.colorbar(im, ax=ax)

        # 3. inputs
        ax = axes[2, j]
        ax.plot(BIN_CENTERS, res['input'][k][0], label='time from tone onset (s)')
        ax.plot(BIN_CENTERS, res['input'][k][1], label='photostim on')
        ax.axvline(0, color='r'); ax.axvline(tone_onset[tr] - g, color='g')
        ax.set_title('inputs'); ax.legend(fontsize=7)

        # 4. raw tongue trace vs binned class
        ax = axes[3, j]
        lo = np.searchsorted(vtime, g + T_START); hi = np.searchsorted(vtime, g + T_END)
        t = vtime[lo:hi] - g
        yy = tongue_y[lo:hi].copy().astype(float)
        yy[~tongue_vis[lo:hi]] = np.nan
        ax.plot(t, yy, 'k.', markersize=1, label='tongue y (visible frames)')
        ax.plot(BIN_CENTERS, tongue_bin_y[k], 'o-', color='C1', markersize=3, label='binned mean y')
        ax.axhline(res['tongue_p40'], color='C2', ls='--', label='40th pct')
        ax.axhline(res['tongue_p60'], color='C3', ls='--', label='60th pct')
        ax.axvline(0, color='r')
        ax.set_title('tongue y position'); ax.legend(fontsize=7)

        # 5. outputs
        ax = axes[4, j]
        for i, name in enumerate(OUTPUT_NAMES):
            ax.step(BIN_CENTERS, res['output'][k][i], where='mid', label=name)
        ax.axvline(0, color='r')
        ax.set_ylim(-0.5, 3.5); ax.set_xlabel('time from go cue (s)')
        ax.set_title('outputs (class indices)'); ax.legend(fontsize=7)

    fig.tight_layout()
    fname = os.path.join(plot_dir, f'processing_{sess}.png')
    fig.savefig(fname, dpi=110)
    plt.close(fig)

    # session-level summary figure: tongue percentile discretization + coverage
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    vis = np.isfinite(tongue_bin_y)
    axes[0].hist(tongue_bin_y[vis], bins=60, color='k')
    axes[0].axvline(res['tongue_p40'], color='C2', ls='--', label='40th pct')
    axes[0].axvline(res['tongue_p60'], color='C3', ls='--', label='60th pct')
    axes[0].set_title(f'{sess}: binned tongue y (visible bins)'); axes[0].legend()
    cls = np.stack([o[3] for o in res['output']])
    for c in range(4):
        axes[1].plot(BIN_CENTERS, (cls == c).mean(axis=0), label=f'class {c}')
    axes[1].axvline(0, color='r'); axes[1].set_title('tongue class fraction vs time'); axes[1].legend()
    axes[2].plot(BIN_CENTERS, np.mean([n.mean(axis=0) for n in res['neural']], axis=0))
    axes[2].axvline(0, color='r'); axes[2].set_title('session-mean population rate (spikes/s)')
    axes[2].set_xlabel('time from go cue (s)')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'processing_{sess}_summary.png'), dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nproc', type=int, default=16)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    if args.sample:
        # keep the first files needed to obtain 2 sessions that pass curation
        args.nproc = 1
    print(f'Processing {len(files)} NWB files with {args.nproc} workers', flush=True)

    plot_dir = os.path.dirname(os.path.abspath(args.outfile))
    jobs = [(f, args.show_processing and (args.sample or i < 2), plot_dir) for i, f in enumerate(files)]

    t_start = time.time()
    results = []
    if args.nproc > 1:
        with Pool(args.nproc) as pool:
            for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
                results.append(res)
                el = time.time() - t_start
                print(f'[{i+1}/{len(files)}] {res.get("sess","?"):32s} '
                      f'{res.get("skipped", "kept %d trials, %d units" % (res.get("n_trials_kept",0), res.get("n_good",0)))}'
                      f'  | elapsed {el:.0f}s, eta {el/(i+1)*(len(files)-i-1):.0f}s', flush=True)
    else:
        n_kept = 0
        for i, job in enumerate(jobs):
            res = process_session(job)
            results.append(res)
            el = time.time() - t_start
            print(f'[{i+1}/{len(files)}] {res.get("sess","?"):32s} '
                  f'{res.get("skipped", "kept %d trials, %d units" % (res.get("n_trials_kept",0), res.get("n_good",0)))}'
                  f'  | elapsed {el:.0f}s', flush=True)
            print('    timings: ' + ', '.join(f'{k}={v:.2f}s' for k, v in res.get('timings', {}).items()), flush=True)
            if 'skipped' not in res:
                n_kept += 1
            if args.sample and n_kept >= 2:
                break

    kept = [r for r in results if r is not None and 'skipped' not in r]
    skipped = [r for r in results if r is not None and 'skipped' in r]
    kept.sort(key=lambda r: (r['subject'], r['sess']))
    print(f'\nSessions kept: {len(kept)}; skipped: {len(skipped)}')
    for r in skipped:
        print(f'  skipped {r["sess"]}: {r["skipped"]}')

    # --------------------------------------------------------------- assemble dataset
    subjects = sorted({r['subject'] for r in kept})
    subject_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
    all_regions = sorted({lab for r in kept for lab in r['region_labels']})
    brain_region_idx = [np.array([all_regions.index(l) for l in r['region_labels']], dtype=np.int64)
                        for r in kept]

    n_trials_total = sum(len(r['neural']) for r in kept)
    n_units = sum(r['n_good'] for r in kept)
    frac_unobs = float(np.mean([r['frac_unobserved_bins'] for r in kept]))

    data = {
        'neural': [r['neural'] for r in kept],
        'input': [r['input'] for r in kept],
        'output': [r['output'] for r in kept],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': all_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice performed an auditory delayed-response (memory-guided movement) task: '
                'a 3 kHz or 12 kHz tone during the sample epoch instructed licking left or right, a 1.2 s '
                'delay epoch required withholding, and an auditory go cue opened a 1.5 s answer epoch. '
                'Decoded outputs are the animal lick-direction choice, the trial outcome (ignore/miss/hit), '
                'whether the animal licked early, and the discretized side-view tongue y-position per time bin. '
                'Decoder inputs are the time since the instructing tone onset and whether ALM photoinhibition '
                'was on at each time point.'),
            'time_bin_size': BIN_SIZE * 1000.0,   # ms
            'temporal_alignment_event': 'auditory go cue onset (BehavioralEvents/go_start_times)',
            'off_start': T_START,
            'off_end': T_END,
            'neural_units': 'firing rate, spikes/s (spike count per 50 ms bin / 0.05 s)',
            'dataset': 'DANDI:000363 Mesoscale Activity Map (Chen, Nguyen, Li, Svoboda 2023)',
            'papers': ['Chen et al., Brain-wide neural activity underlying memory-guided movement',
                       'Wang, Kurgyis et al., Brain-wide analysis reveals movement encoding structured '
                       'across and within brain areas'],
            'neuron_curation': ("units.classification == 'good' (spike-sorting quality-control classifier "
                                'of Chen, Liu et al. 2023, the same unit list used in both papers)'),
            'session_curation': (f'>=1 good unit; behavioural criteria of the data paper (performance > '
                                 f'{MIN_PERFORMANCE:.0%} on control non-early trials and >= '
                                 f'{MIN_CORRECT_PER_DIRECTION} correct lick-left and lick-right trials); '
                                 'side-view video covering the analysis window'),
            'trial_curation': ('all trial types kept (photostimulation, early lick, ignore, auto/free water) '
                               'because photostimulation is a decoder input and early lick / ignore are '
                               'decoder outputs; trials dropped only when the video covers < 90 % of the '
                               'analysis window'),
            'video': f'DeepLabCut side-view tongue marker at 300 Hz (dt={VIDEO_DT}s); visible if likelihood > {LIKELIHOOD_THRESH}',
            'tongue_discretization': ('per-session 40th/60th percentiles of the binned, visible tongue '
                                      'y-position; bins with no visible frame are class 3 (not visible)'),
            'input_descriptions': {
                'time_from_tone_onset': 'seconds from the onset of the instructing sample tone (last sample '
                                        'epoch onset before the go cue) to the centre of the time bin',
                'photostim_on': '1 if ALM photoinhibition was on during the time bin, else 0'},
            'n_sessions': len(kept),
            'n_subjects': len(subjects),
            'n_trials': n_trials_total,
            'n_neurons': n_units,
            'frac_unobserved_neural_bins': frac_unobs,
            'note_unobserved': ('spikes are stored only within the NWB trial intervals; on error (miss) '
                                'trials the interval ends ~0.8 s after the go cue, so later bins contain '
                                'no spikes'),
            'session_info': [{'session': r['sess'], 'subject': r['subject'],
                              'n_trials_total': r['n_trials_total'], 'n_trials_kept': r['n_trials_kept'],
                              'n_trials_observed': r['n_trials_observed'],
                              'n_dropped_video': r['n_dropped_video'],
                              'n_dropped_unobserved': r['n_dropped_unobserved'],
                              'n_dropped_empty': r['n_dropped_empty'],
                              'n_neurons': r['n_good'], 'performance': r['performance'],
                              'n_photostim_trials': r['n_photostim'],
                              'tongue_p40': r['tongue_p40'], 'tongue_p60': r['tongue_p60'],
                              'frac_unobserved_bins': r['frac_unobserved_bins']} for r in kept],
        },
    }

    print(f'\nTotal: {len(kept)} sessions, {len(subjects)} subjects, {n_trials_total} trials, '
          f'{n_units} neurons, {len(all_regions)} brain-region labels')
    print(f'Mean fraction of unobserved neural bins: {frac_unobs:.4f}')

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t0:.0f}s')
    print(f'Total run time {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
