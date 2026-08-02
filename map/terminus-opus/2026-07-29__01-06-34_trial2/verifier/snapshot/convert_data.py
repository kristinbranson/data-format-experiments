#!/usr/bin/env python3
"""Convert MAP dataset NWB files to decoder-compatible format."""

import os, sys, glob, json, time, pickle, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import pynwb
except ImportError:
    print("ERROR: pynwb not installed"); sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', default=True)
    group.add_argument('--sample', action='store_true')
    parser.add_argument('--show-processing', action='store_true')
    return parser.parse_args()


def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))


def get_brain_region(eg):
    loc = json.loads(eg.location)
    region = loc['brain_regions']
    for prefix in ['left ', 'right ']:
        if region.startswith(prefix):
            return region[len(prefix):]
    return region


def get_recorded_trial_indices(nwb, good_indices):
    """Return array of trial indices that have neural recordings."""
    obs = nwb.units['obs_intervals'][good_indices[0]]
    n_obs = obs.shape[0]
    n_trials = len(nwb.trials)
    
    if n_obs == n_trials:
        return np.arange(n_trials)
    
    trial_starts = nwb.trials['start_time'][:]
    trial_stops = nwb.trials['stop_time'][:]
    obs_durs = obs[:, 1] - obs[:, 0]
    trial_durs = trial_stops[:n_obs] - trial_starts[:n_obs]
    
    if np.allclose(obs_durs, trial_durs, atol=0.01) and np.allclose(obs[:, 0], trial_starts[:n_obs], atol=0.01):
        return np.arange(n_obs)
    
    recorded = []
    for i in range(n_obs):
        diffs = np.abs(trial_starts - obs[i, 0])
        best = np.argmin(diffs)
        if diffs[best] < 0.1:
            recorded.append(best)
    return np.array(recorded)


def bin_spikes_all_trials(spike_times_list, go_times_arr, t_start, t_end, bin_size):
    n_bins = int(round((t_end - t_start) / bin_size))
    n_neurons = len(spike_times_list)
    result = []
    for go in go_times_arr:
        bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, st in enumerate(spike_times_list):
            if len(st) == 0:
                continue
            mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
            st_w = st[mask]
            if len(st_w) > 0:
                counts, _ = np.histogram(st_w, bins=bin_edges)
                fr[i] = counts.astype(np.float32) / bin_size
        result.append(fr)
    return result


def get_sample_start_for_trial(sample_start_times, trial_start, trial_stop):
    mask = (sample_start_times >= trial_start) & (sample_start_times <= trial_stop)
    return sample_start_times[mask][0] if np.any(mask) else None


def process_session(nwb_path, t_start=-2.5, t_end=1.5, bin_size=0.05,
                   show_processing=False, session_idx=0):
    t0 = time.time()
    basename = os.path.basename(nwb_path)
    print(f"\n{'='*60}")
    print(f"Processing: {basename}")
    
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()
        subject_id = nwb.subject.subject_id
        n_trials = len(nwb.trials)
        
        trial_instruction = nwb.trials['trial_instruction'][:]
        outcome = nwb.trials['outcome'][:]
        early_lick = nwb.trials['early_lick'][:]
        auto_water = nwb.trials['auto_water'][:]
        free_water = nwb.trials['free_water'][:]
        trial_starts = nwb.trials['start_time'][:]
        trial_stops = nwb.trials['stop_time'][:]
        
        photostim_power_raw = nwb.trials['photostim_power'][:]
        photostim_onset_raw = nwb.trials['photostim_onset'][:]
        photostim_dur_raw = nwb.trials['photostim_duration'][:]
        has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
        
        be = nwb.acquisition['BehavioralEvents']
        go_times = be.time_series['go_start_times'].timestamps[:]
        sample_start_times = be.time_series['sample_start_times'].timestamps[:]
        
        # === Session-level filtering (on ALL trials, not just recorded) ===
        # Performance on control trials (no photostim, no early lick)
        mask_no_early = (early_lick == 'no early')
        mask_no_photostim = ~has_photostim
        control = mask_no_early & mask_no_photostim
        ctrl_out = outcome[control]
        n_hit = np.sum(ctrl_out == 'hit')
        n_miss = np.sum(ctrl_out == 'miss')
        perf = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0.0
        
        ctrl_responded = control & (outcome != 'ignore')
        correct_left = np.sum((trial_instruction[ctrl_responded] == 'left') & (outcome[ctrl_responded] == 'hit'))
        correct_right = np.sum((trial_instruction[ctrl_responded] == 'right') & (outcome[ctrl_responded] == 'hit'))
        
        print(f"  Performance: {perf:.1%} ({n_hit}/{n_hit+n_miss}), Correct L={correct_left}, R={correct_right}")
        
        
        # === Good neurons ===
        classifications = nwb.units['classification'][:]
        good_mask = np.array(classifications) == 'good'
        good_indices = np.where(good_mask)[0]
        n_good = len(good_indices)
        if n_good == 0:
            print(f"  SKIPPING: No good neurons"); return None
        
        # === Recorded trials ===
        recorded_trials = get_recorded_trial_indices(nwb, good_indices)
        
        # === Trial filtering (get_regular_trial_mask) ===
        mask_no_auto = (auto_water == 0)
        mask_no_free = (free_water == 0)
        mask_no_ignore = (outcome != 'ignore')
        regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim
        
        recorded_set = set(recorded_trials)
        valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
        
        print(f"  Trials: {n_trials} total, {len(recorded_trials)} recorded, {len(valid_trials)} valid")
        if len(valid_trials) < 2:
            print(f"  SKIPPING: <2 valid trials"); return None
        print(f"  Good neurons: {n_good} / {len(classifications)}")
        
        # === Brain regions ===
        neuron_regions = [get_brain_region(nwb.units['electrode_group'][idx]) for idx in good_indices]
        
        # === Spike times ===
        t_load = time.time()
        spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
        print(f"  Loaded spike times ({time.time()-t_load:.1f}s)")
        
        # === Tongue tracking ===
        bts = nwb.acquisition['BehavioralTimeSeries']
        tongue_ts = tongue_y_arr = tongue_lk_arr = None
        if 'Camera0_side_TongueTracking' in bts.time_series:
            tt_obj = bts.time_series['Camera0_side_TongueTracking']
            tongue_ts = tt_obj.timestamps[:]
            td = tt_obj.data[:]
            tongue_y_arr = td[:, 1]
            tongue_lk_arr = td[:, 2]
        
        n_bins = int(round((t_end - t_start) / bin_size))
        bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
        
        # === Tongue y percentiles ===
        tongue_y_p40 = tongue_y_p60 = None
        if tongue_ts is not None:
            all_ty = []
            for ti in valid_trials:
                go = go_times[ti]
                i_lo = np.searchsorted(tongue_ts, go + t_start)
                i_hi = np.searchsorted(tongue_ts, go + t_end)
                if i_hi > i_lo:
                    lk = tongue_lk_arr[i_lo:i_hi]
                    good = lk > 0.9
                    if np.any(good):
                        all_ty.append(tongue_y_arr[i_lo:i_hi][good])
            if all_ty:
                concat = np.concatenate(all_ty)
                tongue_y_p40 = np.percentile(concat, 40)
                tongue_y_p60 = np.percentile(concat, 60)
                print(f"  Tongue: p40={tongue_y_p40:.1f}, p60={tongue_y_p60:.1f}")
        
        # === Process trials ===
        t_fr = time.time()
        go_valid = go_times[valid_trials]
        neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
        
        input_trials = []; output_trials = []
        half_bin = bin_size / 2.0
        
        for t_idx, trial_idx in enumerate(valid_trials):
            go = go_times[trial_idx]
            
            # Input 0: time from tone onset
            ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
            if ss is not None:
                tft = bin_centers - (ss - go)
            else:
                tft = np.full(n_bins, np.nan, dtype=np.float32)
            
            # Input 1: photostim
            ps_on = np.zeros(n_bins, dtype=np.float32)
            if has_photostim[trial_idx]:
                o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
                d = float(photostim_dur_raw[trial_idx])
                ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
            
            input_trials.append(np.stack([tft.astype(np.float32), ps_on]))
            
            # Outputs
            choice = 1 if trial_instruction[trial_idx] == 'right' else 0
            out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
            out_val = out_map.get(outcome[trial_idx], 0)
            early_val = 1 if early_lick[trial_idx] == 'early' else 0
            
            ty = np.ones(n_bins, dtype=np.int64)
            if tongue_ts is not None and tongue_y_p40 is not None:
                for b in range(n_bins):
                    bc = go + bin_centers[b]
                    il = np.searchsorted(tongue_ts, bc - half_bin)
                    ih = np.searchsorted(tongue_ts, bc + half_bin)
                    if ih > il:
                        lk = tongue_lk_arr[il:ih]
                        gd = lk > 0.9
                        if np.any(gd):
                            my = np.mean(tongue_y_arr[il:ih][gd])
                            ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
            
            out = np.zeros((4, n_bins), dtype=np.int64)
            out[0, :] = choice; out[1, :] = out_val; out[2, :] = early_val; out[3, :] = ty
            output_trials.append(out)
        
        print(f"  Processed {len(valid_trials)} trials ({time.time()-t_fr:.1f}s, {(time.time()-t_fr)/len(valid_trials)*1000:.0f}ms/trial)")
        print(f"  Session done ({time.time()-t0:.1f}s)")
        
        if show_processing:
            try:
                plot_processing(basename, neural_trials, input_trials, output_trials,
                              bin_centers, tongue_y_p40, tongue_y_p60, session_idx)
            except Exception as e:
                print(f"  Plot error: {e}"); import traceback; traceback.print_exc()
        
        return {
            'neural': neural_trials, 'input': input_trials, 'output': output_trials,
            'subject_id': subject_id, 'neuron_regions': neuron_regions,
            'n_good': n_good, 'n_total': len(classifications),
            'n_trials_total': n_trials, 'n_trials_recorded': len(recorded_trials),
            'n_trials_valid': len(valid_trials), 'performance': perf,
            'session_name': basename,
        }


def plot_processing(basename, neural_trials, input_trials, output_trials,
                   bin_centers, tongue_y_p40, tongue_y_p60, session_idx):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f'Processing: {basename}', fontsize=14)
    nt = len(neural_trials); nn = neural_trials[0].shape[0] if nt else 0
    
    ax = axes[0, 0]
    for i in range(min(5, nt)):
        ax.plot(bin_centers, np.mean(neural_trials[i], axis=0), alpha=0.7)
    ax.axvline(0, color='k', ls='--', alpha=0.5); ax.set_title('Mean FR')
    
    ax = axes[0, 1]
    if nt and nn:
        fr0 = neural_trials[0]
        im = ax.imshow(fr0[np.argsort(np.argmax(fr0, axis=1))], aspect='auto',
                      extent=[bin_centers[0], bin_centers[-1], nn, 0], cmap='viridis')
        plt.colorbar(im, ax=ax); ax.set_title('FR heatmap')
    
    ax = axes[1, 0]
    for i in range(min(5, nt)):
        ax.plot(bin_centers, input_trials[i][0], alpha=0.7)
    ax.set_title('Time from tone'); ax.axvline(0, color='k', ls='--', alpha=0.5)
    
    ax = axes[1, 1]
    for i in range(min(nt, 20)):
        if np.any(input_trials[i][1] > 0): ax.plot(bin_centers, input_trials[i][1], alpha=0.7)
    ax.set_title('Photostim')
    
    ax = axes[2, 0]
    ch = [int(output_trials[i][0, 0]) for i in range(nt)]
    ax.bar([0, 1], [ch.count(0), ch.count(1)], tick_label=['L', 'R']); ax.set_title('Choice')
    
    ax = axes[2, 1]
    oc = [int(output_trials[i][1, 0]) for i in range(nt)]
    ax.bar([0, 1, 2], [oc.count(0), oc.count(1), oc.count(2)], tick_label=['Ign', 'Miss', 'Hit'])
    ax.set_title('Outcome')
    
    ax = axes[3, 0]
    for i in range(min(5, nt)): ax.plot(bin_centers, output_trials[i][3], alpha=0.7)
    ax.set_yticks([0, 1, 2]); ax.set_title('Tongue y')
    
    ax = axes[3, 1]; ax.axis('off')
    ax.text(0.1, 0.5, f'N={nn}, T={nt}\np40={tongue_y_p40}\np60={tongue_y_p60}',
            transform=ax.transAxes, fontsize=12, va='center', family='monospace')
    plt.tight_layout()
    plt.savefig(f'processing_{session_idx}.png', dpi=100, bbox_inches='tight')
    plt.close(); print(f"  Saved: processing_{session_idx}.png")


def main():
    args = parse_args()
    t_total = time.time()
    T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
    
    nwb_files = get_nwb_files()
    print(f"Found {len(nwb_files)} NWB files")
    if args.sample:
        nwb_files = [nwb_files[2], nwb_files[4]]  # Pick sessions likely to pass
        print(f"Sample mode: {len(nwb_files)} sessions")
    
    all_sessions = []; skipped = 0
    for i, path in enumerate(nwb_files):
        r = process_session(path, T_START, T_END, BIN_SIZE,
                           show_processing=args.show_processing and i < 2, session_idx=i)
        if r: all_sessions.append(r)
        else: skipped += 1
        elapsed = time.time() - t_total
        if i > 0:
            rate = elapsed / (i + 1)
            print(f"  Progress: {i+1}/{len(nwb_files)}, {elapsed:.0f}s, ~{rate*(len(nwb_files)-i-1):.0f}s left")
    
    print(f"\n{'='*60}")
    print(f"Processed {len(all_sessions)}, skipped {skipped}")
    if not all_sessions: print("ERROR: No sessions!"); sys.exit(1)
    
    subjects = sorted(set(s['subject_id'] for s in all_sessions))
    regions = sorted(set(r for s in all_sessions for r in s['neuron_regions']))
    
    neural=[]; inputs=[]; outputs=[]; subj_idx=[]; br_idx=[]
    tot_n=0; tot_t=0
    for s in all_sessions:
        neural.append(s['neural']); inputs.append(s['input']); outputs.append(s['output'])
        subj_idx.append(subjects.index(s['subject_id']))
        br_idx.append(np.array([regions.index(r) for r in s['neuron_regions']]))
        tot_n += s['n_good']; tot_t += s['n_trials_valid']
    
    data = {
        'neural': neural, 'input': inputs, 'output': outputs,
        'subjects': subjects, 'subject_idx': np.array(subj_idx),
        'brain_regions': regions, 'brain_region_idx': br_idx,
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y'],
        'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['low','mid','high']],
        'metadata': {
            'task_description': 'Auditory delayed response: tones during sample, 1.2s delay, lick L/R after go cue',
            'time_bin_size': BIN_SIZE * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START, 'off_end': T_END,
            'n_sessions': len(all_sessions), 'n_sessions_skipped': skipped,
            'total_neurons': tot_n, 'total_trials': tot_t,
            'session_info': [{'name': s['session_name'], 'subject_id': s['subject_id'],
                'n_neurons': s['n_good'], 'n_trials': s['n_trials_valid'],
                'performance': s['performance']} for s in all_sessions],
        }
    }
    
    print(f"\nSUMMARY: {len(all_sessions)} sessions, {len(subjects)} subjects")
    print(f"Regions: {regions}")
    print(f"Neurons: {tot_n}, Trials: {tot_t}")
    print(f"Mean: {tot_n/len(all_sessions):.0f} neurons/sess, {tot_t/len(all_sessions):.0f} trials/sess")
    
    with open(args.output, 'wb') as f: pickle.dump(data, f)
    print(f"Saved: {os.path.getsize(args.output)/(1024**2):.1f} MB, Time: {time.time()-t_total:.1f}s")

if __name__ == '__main__':
    main()
