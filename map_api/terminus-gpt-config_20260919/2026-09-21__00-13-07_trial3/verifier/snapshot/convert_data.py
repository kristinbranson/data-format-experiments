#!/usr/bin/env python3
"""Convert MAP NWB sessions to decoder-compatible pickle format.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
All NWB access is through pynwb; h5py is intentionally not imported.
"""
import argparse, json, pickle, time
from pathlib import Path
from collections import Counter
import numpy as np
from pynwb import NWBHDF5IO

DATA_ROOT = Path('/app/data')
BIN = 0.05
OFF_START, OFF_END = -2.5, 1.5
EDGES_REL = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
N_TIME = 80
LIKELIHOOD_THRESHOLD = 0.9
MAX_VIDEO_GAP = 0.010


def selected_unit_indices(units):
    """Paper-matched classifier-good units valid on every trial."""
    classification = np.asarray(units['classification'][:]).astype(str)
    candidates = np.flatnonzero(classification == 'good')
    keep = []
    for j in candidates:
        if np.asarray(units['is_good_trials'][int(j)], dtype=bool).all():
            keep.append(int(j))
    return np.asarray(keep, dtype=np.int64), len(candidates)


def fully_observed_trial_mask(units, unit_inds, go):
    """Trials whose complete requested window is in every selected unit's observation intervals."""
    lo, hi = go + OFF_START, go + OFF_END
    keep = np.ones(len(go), dtype=bool)
    for j in unit_inds:
        obs = np.asarray(units['obs_intervals'][int(j)], dtype=np.float64).reshape(-1, 2)
        covered = np.zeros(len(go), dtype=bool)
        if len(obs):
            oi = np.searchsorted(obs[:, 0], lo, side='right') - 1
            valid = oi >= 0
            covered[valid] = obs[oi[valid], 1] >= hi[valid]
        keep &= covered
    return keep


def bin_spikes(units, unit_inds, go):
    """Return trial x neuron x time firing rates using nonoverlapping windows.

    Go spacings in this release exceed 4 s, so trial windows do not overlap.
    Each unit's spikes are assigned to the latest sorted window start and then
    accumulated with bincount. This is equivalent to per-trial histograms.
    """
    starts = go + OFF_START
    ntr, nneu = len(go), len(unit_inds)
    rates = np.zeros((ntr, nneu, N_TIME), dtype=np.float32)
    for k, unit_i in enumerate(unit_inds):
        spikes = np.asarray(units['spike_times'][int(unit_i)], dtype=np.float64)
        trial_i = np.searchsorted(starts, spikes, side='right') - 1
        valid = trial_i >= 0
        if not valid.any():
            continue
        sp = spikes[valid]
        ti = trial_i[valid]
        rel = sp - starts[ti]
        valid2 = (rel >= 0) & (rel < (OFF_END - OFF_START))
        if not valid2.any():
            continue
        ti = ti[valid2]
        bi = np.floor(rel[valid2] / BIN).astype(np.int64)
        flat = ti * N_TIME + bi
        counts = np.bincount(flat, minlength=ntr * N_TIME).reshape(ntr, N_TIME)
        rates[:, k, :] = counts.astype(np.float32) / BIN
    return rates


def nearest_values(query, timestamps, values):
    """Nearest timestamp indices and gaps for sorted timestamps."""
    if len(timestamps) < 2:
        shape = query.shape
        return np.full(shape, np.nan), np.full(shape, np.inf), np.zeros(shape, dtype=np.int64)
    ix = np.searchsorted(timestamps, query)
    ix = np.clip(ix, 1, len(timestamps)-1)
    prev = ix - 1
    choose = np.where(np.abs(timestamps[prev]-query) <= np.abs(timestamps[ix]-query), prev, ix)
    return values[choose], np.abs(timestamps[choose]-query), choose


def coarse_regions(nwb, unit_inds):
    """Read coarse lateralized recording target through unit electrode links."""
    out = []
    for j in unit_inds:
        electrode_rows = nwb.units['electrodes'][int(j)]  # pynwb-dereferenced DataFrame
        if len(electrode_rows) == 0:
            out.append('unknown')
            continue
        loc = str(electrode_rows.iloc[0]['location'])
        try:
            loc = str(json.loads(loc).get('brain_regions', loc))
        except (json.JSONDecodeError, TypeError):
            pass
        out.append(loc)
    return out


def process_session(path, make_plot=False):
    t0 = time.perf_counter()
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        unit_inds, classifier_good = selected_unit_indices(nwb.units)
        if len(unit_inds) == 0:
            return None
        trials = nwb.trials
        events = nwb.acquisition['BehavioralEvents'].time_series
        go_all = np.asarray(events['go_start_times'].timestamps[:], dtype=np.float64)
        assert len(go_all) == len(trials)
        # obs_intervals encode a paper-analysis subset that strongly excludes error
        # trials, not general acquisition validity. Preserve all released trial rows;
        # true simultaneous acquisition gaps are detected from zero total raw spikes.
        trial_keep = np.ones(len(go_all), dtype=bool)
        original_trial_indices = np.arange(len(go_all), dtype=np.int64)
        go = go_all
        assert np.all(np.diff(go) > (OFF_END-OFF_START)), 'overlapping requested windows'

        sample = np.asarray(events['sample_start_times'].timestamps[:], dtype=np.float64)
        sample_i = np.searchsorted(sample, go, side='right') - 1
        assert np.all(sample_i >= 0)
        tone = sample[sample_i]
        trial_start_all = np.asarray(trials['start_time'][:], dtype=np.float64)
        trial_start = trial_start_all[trial_keep]
        assert np.all((tone >= trial_start) & (tone <= go))

        # Inputs at bin centers.
        absolute_centers = go[:, None] + CENTERS_REL[None, :]
        time_from_tone = (absolute_centers - tone[:, None]).astype(np.float32)
        photostim = np.zeros((len(go), N_TIME), dtype=np.float32)
        ps = np.asarray(events['photostim_start_times'].timestamps[:], dtype=np.float64)
        pe = np.asarray(events['photostim_stop_times'].timestamps[:], dtype=np.float64)
        assert len(ps) == len(pe)
        if len(ps):
            event_i = np.searchsorted(ps, absolute_centers, side='right') - 1
            valid_event = event_i >= 0
            safe_i = np.maximum(event_i, 0)
            photostim = (valid_event & (absolute_centers < pe[safe_i])).astype(np.float32)

        # Trial labels, restricted to complete neural observation periods.
        instruction = np.asarray(trials['trial_instruction'][:]).astype(str)[trial_keep]
        outcome_s = np.asarray(trials['outcome'][:]).astype(str)[trial_keep]
        early_s = np.asarray(trials['early_lick'][:]).astype(str)[trial_keep]
        choice = np.full(len(go), 2, dtype=np.int64)  # no lick
        choice[(outcome_s == 'hit') & (instruction == 'left')] = 0
        choice[(outcome_s == 'hit') & (instruction == 'right')] = 1
        choice[(outcome_s == 'miss') & (instruction == 'left')] = 1
        choice[(outcome_s == 'miss') & (instruction == 'right')] = 0
        outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
        outcome = np.asarray([outcome_map[x] for x in outcome_s], dtype=np.int64)
        early = (early_s == 'early').astype(np.int64)

        # Tongue thresholds use all visible samples over the session.
        tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
        tongue_t = np.asarray(tongue_ts.timestamps[:], dtype=np.float64)
        tongue_data = np.asarray(tongue_ts.data[:], dtype=np.float64)
        y_all, like_all = tongue_data[:, 1], tongue_data[:, 2]
        visible_all = np.isfinite(y_all) & np.isfinite(like_all) & (like_all >= LIKELIHOOD_THRESHOLD)
        if visible_all.any():
            p40, p60 = np.percentile(y_all[visible_all], [40, 60])
        else:
            p40 = p60 = np.nan
        query = absolute_centers
        y, gaps, nearest_i = nearest_values(query, tongue_t, y_all)
        nearest_like = like_all[nearest_i] if len(tongue_t) >= 2 else np.full(query.shape, np.nan)
        visible = (gaps <= MAX_VIDEO_GAP) & np.isfinite(y) & np.isfinite(nearest_like) & (nearest_like >= LIKELIHOOD_THRESHOLD)
        tongue_class = np.full(query.shape, 3, dtype=np.int64)
        tongue_class[visible & (y < p40)] = 0
        tongue_class[visible & (y >= p40) & (y <= p60)] = 1
        tongue_class[visible & (y > p60)] = 2

        rates = bin_spikes(nwb.units, unit_inds, go)
        # Simultaneous zero spikes across all selected units for the full 4-s window
        # indicates an acquisition gap not represented reliably by obs_intervals.
        activity_keep = np.any(rates != 0, axis=(1, 2))
        excluded_zero_activity_trials = int((~activity_keep).sum())
        if activity_keep.sum() < 2:
            return None
        rates = rates[activity_keep]
        go = go[activity_keep]
        tone = tone[activity_keep]
        time_from_tone = time_from_tone[activity_keep]
        photostim = photostim[activity_keep]
        choice = choice[activity_keep]
        outcome = outcome[activity_keep]
        early = early[activity_keep]
        tongue_class = tongue_class[activity_keep]
        y = y[activity_keep]
        nearest_like = nearest_like[activity_keep]
        original_trial_indices = original_trial_indices[activity_keep]
        regions = coarse_regions(nwb, unit_inds)
        subject = str(nwb.subject.subject_id)
        identifier = str(nwb.identifier)
        fine_regions = np.asarray(nwb.units['anno_name'][:]).astype(str)[unit_inds]
        unit_ids = np.asarray(nwb.units.id[:])[unit_inds].tolist()

        inputs = [np.stack((time_from_tone[j], photostim[j]), axis=0).astype(np.float32)
                  for j in range(len(go))]
        outputs = [np.stack((np.full(N_TIME, choice[j]), np.full(N_TIME, outcome[j]),
                             np.full(N_TIME, early[j]), tongue_class[j]), axis=0).astype(np.int8)
                   for j in range(len(go))]
        neural = [rates[j] for j in range(len(go))]

        assert all(x.shape == (len(unit_inds), N_TIME) for x in neural)
        assert all(x.shape == (2, N_TIME) for x in inputs)
        assert all(x.shape == (4, N_TIME) for x in outputs)
        assert np.isfinite(rates).all() and (rates >= 0).all()

        plot_info = None
        if make_plot:
            plot_info = dict(go=go, tone=tone, centers=CENTERS_REL, rates=rates,
                             photostim=photostim, time_from_tone=time_from_tone,
                             tongue_y=y, tongue_like=nearest_like, tongue_class=tongue_class,
                             p40=p40, p60=p60, identifier=identifier)
        info = dict(identifier=identifier, source_file=str(path), subject=subject,
                    n_trials=len(go), source_trials=len(trials), original_trial_indices=original_trial_indices.tolist(),
                    excluded_unobserved_trials=int(len(trials)-trial_keep.sum()),
                    excluded_zero_activity_trials=excluded_zero_activity_trials, classifier_good_units=classifier_good,
                    selected_units=len(unit_inds), excluded_bad_trial_units=classifier_good-len(unit_inds),
                    unit_ids=unit_ids, fine_region_labels=fine_regions.tolist(),
                    tongue_p40=float(p40), tongue_p60=float(p60),
                    tongue_visible_fraction=float(np.mean(tongue_class != 3)),
                    tone_to_go_min=float(np.min(go-tone)), tone_to_go_max=float(np.max(go-tone)))
    return dict(neural=neural, inputs=inputs, outputs=outputs, regions=regions,
                subject=subject, info=info, plot_info=plot_info,
                elapsed=time.perf_counter()-t0)


def save_processing_plot(info, outpath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = info['rates']; trial = min(5, r.shape[0]-1); t = info['centers']
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    mean_rate = r[trial].mean(axis=0)
    ax[0].plot(t, mean_rate); ax[0].set_ylabel('Mean rate (Hz)'); ax[0].set_title(info['identifier'])
    ax[1].plot(t, info['time_from_tone'][trial]); ax[1].set_ylabel('Time from tone (s)')
    ax[2].step(t, info['photostim'][trial], where='mid'); ax[2].set_ylabel('Photostim')
    ax[3].plot(t, info['tongue_y'][trial], label='y'); ax[3].axhline(info['p40'], ls='--'); ax[3].axhline(info['p60'], ls='--'); ax[3].set_ylabel('Tongue y')
    ax[4].plot(t, info['tongue_like'][trial], label='likelihood'); ax[4].step(t, info['tongue_class'][trial]/3, where='mid', label='class/3'); ax[4].axhline(LIKELIHOOD_THRESHOLD, ls='--', color='k'); ax[4].legend(); ax[4].set_ylabel('Visibility/class'); ax[4].set_xlabel('Time from go (s)')
    for a in ax: a.axvline(0, color='r', lw=1); a.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='process first 2 usable sessions')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()
    files = sorted(DATA_ROOT.rglob('*.nwb'))
    target = 2 if args.sample else None
    print(f'Found {len(files)} NWB files; mode={"sample" if args.sample else "full"}', flush=True)

    results=[]; subjects=[]; all_regions=set(); t0=time.perf_counter(); skipped=[]
    for p in files:
        make_plot=args.show_processing and len(results)<2
        res=process_session(p, make_plot)
        if res is None:
            skipped.append(str(p)); print(f'SKIP no selected units: {p.name}', flush=True); continue
        results.append(res); subjects.append(res['subject']); all_regions.update(res['regions'])
        i=len(results); inf=res['info']
        print(f'[{i}] {inf["identifier"]}: trials={inf["n_trials"]} neurons={inf["selected_units"]} '
              f'visible={inf["tongue_visible_fraction"]:.3f} time={res["elapsed"]:.2f}s', flush=True)
        if make_plot:
            save_processing_plot(res['plot_info'], f'/app/processing_{inf["identifier"]}.png')
            res['plot_info']=None
        if target and len(results)>=target: break

    unique_subjects=sorted(set(subjects)); subject_lookup={x:i for i,x in enumerate(unique_subjects)}
    brain_regions=sorted(all_regions); region_lookup={x:i for i,x in enumerate(brain_regions)}
    data={
      'neural':[r['neural'] for r in results],
      'input':[r['inputs'] for r in results],
      'output':[r['outputs'] for r in results],
      'subjects':unique_subjects,
      'subject_idx':np.asarray([subject_lookup[r['subject']] for r in results],dtype=np.int64),
      'brain_regions':brain_regions,
      'brain_region_idx':[np.asarray([region_lookup[x] for x in r['regions']],dtype=np.int64) for r in results],
      'input_names':['time from tone onset','photostimulation on'],
      'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
      'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
      'metadata':{
        'task_description':'Auditory delayed-response task; decode actual lick choice, outcome, early lick, and time-varying tongue y category from go-aligned neural activity.',
        'time_bin_size':50.0,
        'temporal_alignment_event':'auditory go cue onset (BehavioralEvents/go_start_times)',
        'off_start':OFF_START,'off_end':OFF_END,
        'neural_measure':'firing rate (Hz) from non-overlapping spike-count bins',
        'time_bin_centers_seconds':CENTERS_REL.astype(np.float32),
        'unit_filter':'NWB classification == good; is_good_trials true for every session trial; exclude trials with zero total spikes across all selected units (acquisition gaps)',
        'tongue_visibility_likelihood_threshold':LIKELIHOOD_THRESHOLD,
        'tongue_max_nearest_sample_gap_seconds':MAX_VIDEO_GAP,
        'tongue_percentile_scope':'all visible Camera0 side tongue y samples within each session',
        'session_info':[r['info'] for r in results],
        'skipped_files':skipped,
        'source_format':'NWB loaded with pynwb',
      }
    }
    total_trials=sum(map(len, data['neural'])); total_neurons=sum(session[0].shape[0] for session in data['neural'] if session)
    print(f'Converted sessions={len(results)} subjects={len(unique_subjects)} trials={total_trials} neurons={total_neurons} regions={len(brain_regions)}',flush=True)
    out=Path(args.outpicklefile); out.parent.mkdir(parents=True,exist_ok=True)
    ts=time.perf_counter();
    with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {out} ({out.stat().st_size/1e9:.3f} GB), pickle time={time.perf_counter()-ts:.2f}s, total={time.perf_counter()-t0:.2f}s',flush=True)

if __name__=='__main__': main()
