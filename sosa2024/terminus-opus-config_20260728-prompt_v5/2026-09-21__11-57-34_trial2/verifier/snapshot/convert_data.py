#!/usr/bin/env python3
"""Convert hippocampal reward-relative data from NWB to decoder format."""
import sys, os, time, argparse
import numpy as np
import pickle, h5py, glob
import scipy.ndimage

# Constants from reference paper/code
REWARD_ZONE_DICT = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
TRACK_LENGTH = 450.0
SWITCH_TRIAL = 30
NEU_COEF = 0.7
BASELINE_WINDOW = 300  # ~20s at 15.5 Hz
SMOOTH_SIGMA = 2
TARGET_RATE_HZ = 15.5078125

def parse_scene_name(identifier):
    return identifier.split('/')[-1]

def get_reward_zones_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    zone_to_coords = {'A': [80,130], 'B': [200,250], 'C': [320,370]}
    rz_coords = np.zeros((n_trials, 2))
    rz_labels = np.empty(n_trials, dtype='U1')
    
    if 'Training' in scene or 'RunningTraining' in scene:
        rz_coords[:] = [275, 325]; rz_labels[:] = 'T'
        return rz_coords, rz_labels
    
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        pre_zone = parts[0][-1]
        post_zone = parts[1][-1]
        ct = min(change_trial, n_trials)
        rz_coords[:ct] = zone_to_coords[pre_zone]; rz_labels[:ct] = pre_zone
        rz_coords[ct:] = zone_to_coords[post_zone]; rz_labels[ct:] = post_zone
        return rz_coords, rz_labels
    
    if '_to_' in scene:
        parts = scene.split('_to_')
        pre_zone = parts[0][-1]
        post_zone = parts[1]
        ct = min(change_trial, n_trials)
        rz_coords[:ct] = zone_to_coords[pre_zone]; rz_labels[:ct] = pre_zone
        rz_coords[ct:] = zone_to_coords[post_zone]; rz_labels[ct:] = post_zone
        return rz_coords, rz_labels
    
    if 'Location' in scene:
        zone = scene.split('Location')[-1][0]
        rz_coords[:] = zone_to_coords[zone]; rz_labels[:] = zone
        return rz_coords, rz_labels
    
    raise ValueError(f"Cannot parse scene: {scene}")

def nansmooth(a, sig, axis=-1):
    nan_mask = np.isnan(a)
    a_filled = np.where(nan_mask, 0, a)
    weights = np.where(nan_mask, 0, 1.0)
    smoothed = scipy.ndimage.gaussian_filter1d(a_filled, sig, axis=axis)
    weight_smoothed = scipy.ndimage.gaussian_filter1d(weights, sig, axis=axis)
    return np.where(weight_smoothed > 0, smoothed / weight_smoothed, np.nan)

def compute_dff(F, Fneu, trial_start_inds, teleport_inds):
    """Compute dF/F: neuropil subtract, maximin baseline, smooth."""
    n_tp, n_neu = F.shape
    f_ = (F.T - NEU_COEF * Fneu.T).astype(np.float64)  # (neurons, timepoints)
    
    nanmask = np.zeros(n_tp, dtype=bool)
    for s, e in zip(trial_start_inds, teleport_inds):
        if s < e: nanmask[s:e] = True
    
    flow = np.full_like(f_, np.nan)
    for s, e in zip(trial_start_inds, teleport_inds):
        if s >= e: continue
        w = min(BASELINE_WINDOW, e - s)
        flow[:, s:e] = scipy.ndimage.minimum_filter1d(f_[:, s:e], w, axis=-1)
        flow[:, s:e] = scipy.ndimage.maximum_filter1d(flow[:, s:e], w, axis=-1)
    
    dff = np.full_like(f_, np.nan)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    
    for s, e in zip(trial_start_inds, teleport_inds):
        if s >= e: continue
        dff[:, s:e] = nansmooth(dff[:, s:e], SMOOTH_SIGMA, axis=1)
    
    return dff

def discretize_distance(d):
    r = np.zeros_like(d, dtype=np.int64)
    r[d < -50] = 0
    r[(d >= -50) & (d < -10)] = 1
    r[(d >= -10) & (d < 0)] = 2
    r[d == 0] = 3
    r[(d > 0) & (d <= 10)] = 4
    r[(d > 10) & (d <= 50)] = 5
    r[d > 50] = 6
    return r

def discretize_position(p):
    r = np.zeros_like(p, dtype=np.int64)
    r[p < 90] = 0
    r[(p >= 90) & (p < 180)] = 1
    r[(p >= 180) & (p < 270)] = 2
    r[(p >= 270) & (p < 360)] = 3
    r[p >= 360] = 4
    return r

def discretize_speed(s):
    r = np.zeros_like(s, dtype=np.int64)
    r[s < 2] = 0
    r[(s >= 2) & (s < 10)] = 1
    r[(s >= 10) & (s < 20)] = 2
    r[(s >= 20) & (s < 40)] = 3
    r[s >= 40] = 4
    return r

def load_neural_data(f):
    """Load F, Fneu for cells (iscell=1) from all planes, concatenated."""
    fluor = f['processing/ophys/Fluorescence']
    neu = f['processing/ophys/Neuropil']
    seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell = seg['iscell'][:]
    plane_idx = seg['planeIdx'][:]
    planes = sorted(fluor.keys())
    
    F_list, Fneu_list = [], []
    for pname in planes:
        pnum = int(pname.replace('plane', ''))
        pmask = plane_idx == pnum
        pcell = iscell[pmask, 0] == 1
        F_list.append(fluor[pname]['data'][:, pcell])
        Fneu_list.append(neu[pname]['data'][:, pcell])
    
    return np.concatenate(F_list, axis=1), np.concatenate(Fneu_list, axis=1), len(planes)

def process_session(nwb_path, show_processing=False):
    t0 = time.time()
    
    with h5py.File(nwb_path, 'r') as f:
        subject_id = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        identifier = f['identifier'][()].decode()
        scene = parse_scene_name(identifier)
        
        print(f"  Processing {subject_id} ses-{session_id} ({scene})...")
        
        F, Fneu, n_planes = load_neural_data(f)
        n_cells = F.shape[1]
        
        # Behavioral data
        behav = f['processing/behavior/BehavioralTimeSeries']
        position = behav['position/data'][:]
        speed = behav['speed/data'][:]
        lick = behav['lick/data'][:]
        environment = behav['environment/data'][:]
        trial_start_sig = behav['trial_start/data'][:]
        teleport_sig = behav['teleport/data'][:]
        reward_zone_raw = behav['reward_zone/data'][:]
        
        # Timestamps for time alignment
        frame_timestamps = behav['position/timestamps'][:]
        reward_timestamps = behav['Reward/timestamps'][:]
    
    n_timepoints = F.shape[0]
    dt = np.median(np.diff(frame_timestamps))
    effective_rate = 1.0 / dt
    print(f"    {n_cells} cells, {n_planes} planes, effective_rate={effective_rate:.1f}Hz")
    
    # Trial boundaries
    trial_start_inds = np.where(trial_start_sig == 1)[0]
    teleport_inds = np.where(teleport_sig == 1)[0]
    n_trials = min(len(trial_start_inds), len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]
    
    if n_trials < 2:
        print(f"    Skipping: {n_trials} trials"); return None
    
    # Compute dF/F
    t_dff = time.time()
    dff = compute_dff(F, Fneu, trial_start_inds, teleport_inds)
    print(f"    dF/F: {time.time()-t_dff:.1f}s, shape={dff.shape}")
    
    # Reward zones per trial
    rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
    
    # Reward outcome per trial (using actual timestamps)
    isreward = np.zeros(n_trials, dtype=np.int64)
    for ti in range(n_trials):
        ts, te = trial_start_inds[ti], teleport_inds[ti]
        has_rzone = np.any(reward_zone_raw[ts:te] > 0)
        t_start = frame_timestamps[ts]
        t_end = frame_timestamps[min(te, n_timepoints-1)]
        has_reward = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
        if has_reward and has_rzone:
            isreward[ti] = 1
    
    # Environment per trial
    trial_env = np.zeros(n_trials, dtype=np.int64)
    for ti in range(n_trials):
        ts, te = trial_start_inds[ti], teleport_inds[ti]
        env_vals = environment[ts:te]
        valid = env_vals[env_vals >= 0]
        if len(valid) > 0:
            trial_env[ti] = int(np.round(np.median(valid)))
    
    # Build trial data
    neural_trials, input_trials, output_trials = [], [], []
    
    for ti in range(n_trials):
        ts, te = trial_start_inds[ti], teleport_inds[ti]
        nf = te - ts
        if nf < 2: continue
        
        # Neural
        trial_neural = np.nan_to_num(dff[:, ts:te], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # Time from trial start (using actual timestamps)
        time_from_start = (frame_timestamps[ts:te] - frame_timestamps[ts]).astype(np.float32)
        
        # Behavioral
        trial_pos = np.clip(position[ts:te], 0, TRACK_LENGTH)
        trial_speed = speed[ts:te]
        trial_lick = lick[ts:te]
        
        # Distance to reward zone
        rz_s, rz_e = rz_coords[ti]
        dist = np.where(trial_pos < rz_s, trial_pos - rz_s,
                       np.where(trial_pos > rz_e, trial_pos - rz_e, 0.0))
        
        # Discretize outputs
        dist_disc = discretize_distance(dist)
        pos_disc = discretize_position(trial_pos)
        speed_disc = discretize_speed(np.abs(trial_speed))
        lick_bin = (trial_lick > 0).astype(np.int64)
        rz_label_int = {'A': 0, 'B': 1, 'C': 2}.get(rz_labels[ti], 0)
        prev_outcome = int(isreward[ti-1]) if ti > 0 else 0
        
        # Input (4, nf)
        inp = np.zeros((4, nf), dtype=np.float32)
        inp[0] = time_from_start
        inp[1] = trial_env[ti]
        inp[2] = ti  # trial number
        inp[3] = prev_outcome
        
        # Output (6, nf)
        out = np.zeros((6, nf), dtype=np.int64)
        out[0] = dist_disc
        out[1] = pos_disc
        out[2] = speed_disc
        out[3] = lick_bin
        out[4] = rz_label_int
        out[5] = isreward[ti]
        
        neural_trials.append(trial_neural)
        input_trials.append(inp)
        output_trials.append(out)
    
    if len(neural_trials) < 2:
        print(f"    Skipping: {len(neural_trials)} valid trials"); return None
    
    print(f"    Done: {len(neural_trials)} trials, {n_cells} cells, {time.time()-t0:.1f}s")
    
    if show_processing:
        _plot_processing(subject_id, session_id, scene, neural_trials, input_trials, 
                        output_trials, rz_coords, rz_labels, position, speed,
                        trial_start_inds, teleport_inds, frame_timestamps)
    
    return {
        'neural_trials': neural_trials, 'input_trials': input_trials,
        'output_trials': output_trials, 'subject_id': subject_id,
        'session_id': session_id, 'scene': scene, 'n_cells': n_cells,
        'n_trials': len(neural_trials), 'effective_rate': effective_rate,
    }

def _plot_processing(subject_id, session_id, scene, neural_trials, input_trials,
                    output_trials, rz_coords, rz_labels, position, speed,
                    trial_start_inds, teleport_inds, frame_timestamps):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(6, 2, figsize=(20, 24))
        fig.suptitle(f'{subject_id} ses-{session_id} ({scene})', fontsize=14)
        
        for col, tidx in enumerate([0, min(len(neural_trials)-1, 35)]):
            t = input_trials[tidx][0]
            axes[0,col].plot(t, np.mean(neural_trials[tidx], axis=0))
            axes[0,col].set_title(f'Trial {int(input_trials[tidx][2,0])}: Mean dF/F')
            
            ti = int(input_trials[tidx][2,0])
            ts, te = trial_start_inds[ti], teleport_inds[ti]
            axes[1,col].plot(t, np.clip(position[ts:te], 0, TRACK_LENGTH))
            axes[1,col].set_title('Position (cm)')
            
            axes[2,col].plot(t, speed[ts:te])
            axes[2,col].set_title('Speed (cm/s)')
            
            rz_s, rz_e = rz_coords[ti]
            tp = np.clip(position[ts:te], 0, TRACK_LENGTH)
            d = np.where(tp < rz_s, tp - rz_s, np.where(tp > rz_e, tp - rz_e, 0.0))
            axes[3,col].plot(t, d); axes[3,col].axhline(0, color='r', ls='--', alpha=0.5)
            axes[3,col].set_title(f'Dist to RZ {rz_labels[ti]}:[{rz_s:.0f},{rz_e:.0f}]')
            
            axes[4,col].plot(t, output_trials[tidx][0], label='dist_rz', alpha=0.7)
            axes[4,col].plot(t, output_trials[tidx][1], label='pos', alpha=0.7)
            axes[4,col].plot(t, output_trials[tidx][2], label='speed', alpha=0.7)
            axes[4,col].plot(t, output_trials[tidx][3], label='lick', alpha=0.7)
            axes[4,col].legend(fontsize=8); axes[4,col].set_title('Outputs')
            
            axes[5,col].plot(t, input_trials[tidx][0], label='time')
            axes[5,col].set_title(f'Inputs (env={input_trials[tidx][1,0]:.0f}, prev={input_trials[tidx][3,0]:.0f})')
            axes[5,col].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig(f'/app/processing_{subject_id}_ses{session_id}.png', dpi=100)
        plt.close()
        print(f"    Saved processing plot")
    except Exception as e:
        print(f"    Warning: plot failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile')
    parser.add_argument('--full', action='store_true', default=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--show-processing', action='store_true')
    args = parser.parse_args()
    if args.sample: args.full = False
    
    nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
    print(f"Found {len(nwb_files)} NWB files")
    
    if args.sample:
        sample = [f for f in nwb_files if 'sub-m11_ses-03' in f or 'sub-m17_ses-09' in f]
        nwb_files = sample[:2] if sample else nwb_files[:2]
        print(f"Sample mode: {len(nwb_files)} sessions")
    
    all_sessions = []
    t_total = time.time()
    
    for i, nwb_path in enumerate(nwb_files):
        print(f"\n[{i+1}/{len(nwb_files)}] {os.path.basename(nwb_path)}")
        t0 = time.time()
        sess = process_session(nwb_path, show_processing=args.show_processing and i < 2)
        if sess: all_sessions.append(sess)
        dt = time.time() - t0
        rem = (len(nwb_files) - i - 1) * dt
        print(f"    Time: {dt:.1f}s, Est remaining: {rem/60:.1f}min")
    
    print(f"\nTotal: {(time.time()-t_total)/60:.1f}min, {len(all_sessions)} sessions")
    
    # Build output structure
    subjects = sorted(set(s['subject_id'] for s in all_sessions))
    sub2idx = {s: i for i, s in enumerate(subjects)}
    
    neural, inputs, outputs, subject_idx, br_idx = [], [], [], [], []
    for s in all_sessions:
        neural.append(s['neural_trials'])
        inputs.append(s['input_trials'])
        outputs.append(s['output_trials'])
        subject_idx.append(sub2idx[s['subject_id']])
        br_idx.append(np.zeros(s['n_cells'], dtype=np.int64))
    
    time_bin_ms = 1000.0 / TARGET_RATE_HZ
    
    data = {
        'neural': neural, 'input': inputs, 'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': br_idx,
        'input_names': ['time_from_trial_start', 'environment_type', 'trial_number', 'previous_trial_outcome'],
        'output_names': ['distance_to_reward_zone', 'absolute_position', 'speed', 'lick', 'reward_zone_location', 'reward_outcome'],
        'output_values': [
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '>0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no_lick', 'lick'],
            ['zone_A', 'zone_B', 'zone_C'],
            ['no_reward', 'reward'],
        ],
        'metadata': {
            'task_description': 'VR linear track (450cm) with hidden reward zone at A(80-130), B(200-250), or C(320-370). Zone switches after 30 trials. Two environments. ~15% omission.',
            'time_bin_size': time_bin_ms,
            'temporal_alignment_event': 'trial_start (entry to linear track)',
            'off_start': 0.0, 'off_end': None,
            'imaging_rate_hz': TARGET_RATE_HZ,
            'brain_region': 'hippocampus CA1',
            'indicator': 'GCaMP7f',
            'neural_signal': 'dF/F (F - 0.7*Fneu, maximin baseline, Gaussian smooth sigma=2)',
            'track_length_cm': TRACK_LENGTH,
            'reward_zones': REWARD_ZONE_DICT,
        }
    }
    
    # Summary stats
    tot_trials = sum(len(s) for s in neural)
    tot_neurons = sum(s[0].shape[0] for s in neural if s)
    print(f"\n=== Summary ===")
    print(f"Subjects: {len(subjects)}, Sessions: {len(neural)}")
    print(f"Trials: {tot_trials}, Neurons: {tot_neurons}")
    print(f"Mean trials/sess: {tot_trials/len(neural):.1f}, Mean neurons/sess: {tot_neurons/len(neural):.1f}")
    print(f"Time bin: {time_bin_ms:.2f}ms")
    
    all_rew = [o[5,0] for outs in outputs for o in outs]
    print(f"Reward rate: {np.mean(all_rew):.3f} (expected ~0.85)")
    
    for oi, on in enumerate(data['output_names']):
        vals = np.concatenate([o[oi].flatten() for outs in outputs for o in outs])
        u, c = np.unique(vals, return_counts=True)
        print(f"  {on}: {dict(zip(u.astype(int), np.round(c/c.sum(), 3)))}")
    
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as pf:
        pickle.dump(data, pf, protocol=4)
    print(f"Saved: {os.path.getsize(args.outfile)/(1024**2):.1f} MB")
    print("Done!")

if __name__ == '__main__':
    main()
