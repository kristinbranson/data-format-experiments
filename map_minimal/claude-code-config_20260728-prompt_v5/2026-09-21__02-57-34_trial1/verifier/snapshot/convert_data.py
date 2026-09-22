"""
Convert MAP (Memory-guided Action Planning) NWB data to the standardized decoder format.

Data source: "Brain-wide neural activity underlying memory-guided movement" (Chen et al.)
Analysis reference: "Brain-wide analysis reveals movement encoding structured across and within brain areas"

Processing decisions:
- Unit filtering: classification == 'good' (QC classifier approach from the paper)
- Trial filtering: only trials within recording observation intervals are included
- Session filtering: >65% correct on control (non-photostim, non-early-lick) trials,
  and at least 50 correct lick-left and 50 correct lick-right control trials
- Temporal alignment: Go cue onset (from BehavioralEvents/go_start_times)
- Window: -2.5s to +1.5s around go cue
- Bin width: 50ms non-overlapping bins -> firing rates (spikes/s)
- Brain regions: derived from electrode_group location metadata (probe target region),
  with left/right prefix removed
- Tongue y-position: discretized per session from Camera0_side_TongueTracking,
  using DLC likelihood threshold of 0.9 for visibility
- All trial types included (early lick, ignore, miss, hit) since these are decoder outputs
"""

import os
import glob
import json
import pickle
import numpy as np
import pynwb

# Constants
BIN_WIDTH = 0.05  # 50 ms
T_START = -2.5    # seconds relative to go cue
T_END = 1.5       # seconds relative to go cue
N_BINS = int(round((T_END - T_START) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
MIN_CORRECT_PER_SIDE = 50
MIN_PERFORMANCE = 0.65


def get_brain_region(electrode_group):
    """Extract simplified brain region name from electrode group location."""
    loc = json.loads(electrode_group.location)
    region = loc.get('brain_regions', '')
    for prefix in ['left ', 'right ']:
        if region.startswith(prefix):
            region = region[len(prefix):]
    return region


def get_valid_trial_mask(obs_intervals, go_cue_times):
    """
    Determine which trials have go cues within the observation intervals.
    A trial is valid if its go cue time falls within any observation interval,
    and the full extraction window [-2.5, 1.5] around the go cue is within
    the observation range.
    """
    n_trials = len(go_cue_times)
    valid = np.zeros(n_trials, dtype=bool)

    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        # Check if go cue falls within any obs interval
        for obs_start, obs_end in obs_intervals:
            if go >= obs_start and go <= obs_end:
                valid[t_i] = True
                break

    return valid


def compute_firing_rates_all(spike_times_list, go_cue_times):
    """
    Compute firing rates for all units across given trials.
    Returns array of shape (n_units, n_trials, N_BINS).
    """
    n_units = len(spike_times_list)
    n_trials = len(go_cue_times)
    fr_all = np.zeros((n_units, n_trials, N_BINS), dtype=np.float32)

    for u_i, spike_times in enumerate(spike_times_list):
        if len(spike_times) == 0:
            continue
        st = np.sort(spike_times)
        for t_i in range(n_trials):
            go = go_cue_times[t_i]
            i_lo = np.searchsorted(st, go + T_START, side='left')
            i_hi = np.searchsorted(st, go + T_END, side='left')
            if i_hi > i_lo:
                rel = st[i_lo:i_hi] - go
                counts, _ = np.histogram(rel, bins=BIN_EDGES)
                fr_all[u_i, t_i] = counts / BIN_WIDTH
    return fr_all


def discretize_tongue_y(tongue_timestamps, tongue_data, go_cue_times):
    """
    Discretize tongue y-position per trial per time bin.
    Returns: list of arrays, each shape (N_BINS,)
    """
    tongue_y_vals = tongue_data[:, 1]
    tongue_lk_vals = tongue_data[:, 2]
    n_trials = len(go_cue_times)

    # First pass: collect visible y values for percentiles
    all_visible_y = []
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        i_start = np.searchsorted(tongue_timestamps, go + T_START, side='left')
        i_end = np.searchsorted(tongue_timestamps, go + T_END, side='left')
        if i_end > i_start:
            lk = tongue_lk_vals[i_start:i_end]
            vis = lk >= TONGUE_LIKELIHOOD_THRESHOLD
            if np.any(vis):
                all_visible_y.append(tongue_y_vals[i_start:i_end][vis])

    if len(all_visible_y) > 0:
        all_vis = np.concatenate(all_visible_y)
        p40 = np.percentile(all_vis, 40)
        p60 = np.percentile(all_vis, 60)
    else:
        p40 = p60 = 0.0

    # Second pass: discretize per bin
    result = []
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        ty_binned = np.full(N_BINS, 3, dtype=np.int64)
        abs_edges = go + BIN_EDGES
        edge_indices = np.searchsorted(tongue_timestamps, abs_edges, side='left')

        for b_i in range(N_BINS):
            i_start = edge_indices[b_i]
            i_end = edge_indices[b_i + 1]
            if i_end > i_start:
                lk = tongue_lk_vals[i_start:i_end]
                vis = lk >= TONGUE_LIKELIHOOD_THRESHOLD
                if np.any(vis):
                    mean_y = np.mean(tongue_y_vals[i_start:i_end][vis])
                    if mean_y < p40:
                        ty_binned[b_i] = 0
                    elif mean_y <= p60:
                        ty_binned[b_i] = 1
                    else:
                        ty_binned[b_i] = 2
        result.append(ty_binned)
    return result


def process_session(nwb_path):
    """Process a single NWB session file. Returns dict or None if filtered."""
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    subject_id = str(nwb.subject.subject_id) if nwb.subject else 'unknown'
    trials = nwb.trials
    n_trials_total = len(trials)
    units = nwb.units

    # Get behavioral events
    be = nwb.acquisition['BehavioralEvents']
    go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
    sample_start_times = be.time_series['sample_start_times'].timestamps[:]

    if len(go_cue_times_all) != n_trials_total:
        io.close()
        return None

    # Get trial info (all trials)
    outcomes_all = trials['outcome'][:]
    early_licks_all = trials['early_lick'][:]
    instructions_all = trials['trial_instruction'][:]
    photostim_onset_all = trials['photostim_onset'][:]
    photostim_duration_all = trials['photostim_duration'][:]
    trial_starts_all = trials['start_time'][:]

    # --- Filter good units ---
    classifications = units['classification'][:]
    good_idx = np.where(classifications == 'good')[0]

    if len(good_idx) == 0:
        io.close()
        return None

    # --- Determine valid trials based on obs_intervals ---
    # Use the obs_intervals from the first good unit (all units in a session
    # share the same obs_intervals since they come from the same probe set)
    obs_intervals = units['obs_intervals'][int(good_idx[0])]
    valid_trial_mask = get_valid_trial_mask(obs_intervals, go_cue_times_all)

    # Filter to valid trials only
    valid_idx = np.where(valid_trial_mask)[0]
    if len(valid_idx) < 2:
        io.close()
        return None

    go_cue_times = go_cue_times_all[valid_idx]
    outcomes = outcomes_all[valid_idx]
    early_licks = early_licks_all[valid_idx]
    instructions = instructions_all[valid_idx]
    photostim_onset = photostim_onset_all[valid_idx]
    photostim_duration = photostim_duration_all[valid_idx]
    trial_starts = trial_starts_all[valid_idx]
    n_trials = len(valid_idx)

    # --- Session selection criteria (on valid trials only) ---
    is_control = np.array([po == 'N/A' for po in photostim_onset])
    is_no_early = early_licks == 'no early'
    is_responded = outcomes != 'ignore'
    control_responded = is_control & is_no_early & is_responded

    if np.sum(control_responded) == 0:
        io.close()
        return None

    n_correct_control = np.sum((outcomes == 'hit') & is_control & is_no_early)
    performance = n_correct_control / np.sum(control_responded)

    correct_left = np.sum((outcomes == 'hit') & (instructions == 'left') & is_control & is_no_early)
    correct_right = np.sum((outcomes == 'hit') & (instructions == 'right') & is_control & is_no_early)

    if performance < MIN_PERFORMANCE:
        io.close()
        return None
    if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
        io.close()
        return None

    # --- Get brain regions ---
    regions = []
    for idx in good_idx:
        eg = units['electrode_group'][int(idx)]
        regions.append(get_brain_region(eg))

    # --- Compute firing rates ---
    spike_times_list = [units['spike_times'][int(idx)] for idx in good_idx]
    fr_all = compute_firing_rates_all(spike_times_list, go_cue_times)

    # --- Tone onset per trial ---
    sorted_ss = np.sort(sample_start_times)
    tone_onset_rel_go = np.full(n_trials, -1.85)
    for t_i in range(n_trials):
        go = go_cue_times[t_i]
        # Find the original trial index to get the previous go cue correctly
        orig_idx = valid_idx[t_i]
        prev_go = go_cue_times_all[orig_idx - 1] if orig_idx > 0 else 0
        i_lo = np.searchsorted(sorted_ss, prev_go, side='right')
        i_hi = np.searchsorted(sorted_ss, go, side='right')
        if i_hi > i_lo:
            tone_onset_rel_go[t_i] = sorted_ss[i_hi - 1] - go

    # --- Photostim binary per trial ---
    photostim_on_all = np.zeros((n_trials, N_BINS), dtype=np.float32)
    for t_i in range(n_trials):
        if photostim_onset[t_i] != 'N/A':
            ps_onset_abs = trial_starts[t_i] + float(photostim_onset[t_i])
            ps_dur = float(photostim_duration[t_i])
            ps_onset_rel = ps_onset_abs - go_cue_times[t_i]
            ps_offset_rel = ps_onset_rel + ps_dur
            photostim_on_all[t_i] = ((BIN_CENTERS >= ps_onset_rel) & (BIN_CENTERS < ps_offset_rel)).astype(np.float32)

    # --- Tongue tracking ---
    bts = nwb.acquisition.get('BehavioralTimeSeries', None)
    has_tongue = (bts is not None and 'Camera0_side_TongueTracking' in bts.time_series)

    if has_tongue:
        tongue_ts = bts.time_series['Camera0_side_TongueTracking']
        tongue_data = tongue_ts.data[:]
        tongue_timestamps = tongue_ts.timestamps[:]
        tongue_y_per_trial = discretize_tongue_y(tongue_timestamps, tongue_data, go_cue_times)
    else:
        tongue_y_per_trial = [np.full(N_BINS, 3, dtype=np.int64) for _ in range(n_trials)]

    # --- Assemble per-trial data ---
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}

    neural_trials = []
    input_trials = []
    output_trials = []

    for t_i in range(n_trials):
        neural_trials.append(fr_all[:, t_i, :])

        time_from_tone = (BIN_CENTERS - tone_onset_rel_go[t_i]).astype(np.float32)
        inp = np.stack([time_from_tone, photostim_on_all[t_i]], axis=0)
        input_trials.append(inp)

        o = outcomes[t_i]
        inst = instructions[t_i]
        if o == 'ignore':
            choice = 2
        elif o == 'hit':
            choice = 0 if inst == 'left' else 1
        elif o == 'miss':
            choice = 1 if inst == 'left' else 0
        else:
            choice = 2

        out = np.stack([
            np.full(N_BINS, choice, dtype=np.int64),
            np.full(N_BINS, outcome_map.get(o, 0), dtype=np.int64),
            np.full(N_BINS, 1 if early_licks[t_i] == 'early' else 0, dtype=np.int64),
            tongue_y_per_trial[t_i].astype(np.int64)
        ], axis=0)
        output_trials.append(out)

    io.close()

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'subject_id': subject_id,
        'regions': regions,
        'n_trials': n_trials,
        'n_neurons': len(good_idx),
        'performance': performance,
    }


def main():
    nwb_files = sorted(glob.glob('/app/data/sub-*/sub-*.nwb'))
    print(f"Found {len(nwb_files)} NWB files")

    all_sessions = []
    all_subjects = set()
    all_brain_regions = set()

    for i, nwb_path in enumerate(nwb_files):
        basename = os.path.basename(nwb_path)
        print(f"[{i+1}/{len(nwb_files)}] {basename}...", end=' ', flush=True)

        try:
            result = process_session(nwb_path)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        if result is None:
            print("FILTERED")
            continue

        print(f"OK ({result['n_neurons']}n, {result['n_trials']}t, p={result['performance']:.2f})")
        all_sessions.append(result)
        all_subjects.add(result['subject_id'])
        all_brain_regions.update(result['regions'])

    print(f"\n{len(all_sessions)} sessions passed filtering")
    print(f"{len(all_subjects)} subjects")
    print(f"Brain regions: {sorted(all_brain_regions)}")

    if len(all_sessions) == 0:
        print("ERROR: No sessions passed filtering!")
        return

    subjects = sorted(list(all_subjects))
    brain_regions = sorted(list(all_brain_regions))
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[s['subject_id']] for s in all_sessions], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [],
        'input_names': ['time_from_tone_onset', 'photostim_on'],
        'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position'],
        'output_values': [
            ['left', 'right', 'no_lick'],
            ['ignore', 'miss', 'hit'],
            ['no', 'yes'],
            ['low', 'mid', 'high', 'not_visible'],
        ],
        'metadata': {
            'task_description': 'Auditory delayed response task: mice hear tones during sample epoch, wait through delay, then lick left or right after go cue',
            'time_bin_size': BIN_WIDTH * 1000,  # in ms
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'dataset': 'MAP (Memory-guided Action Planning)',
            'species': 'Mus musculus',
            'n_sessions': len(all_sessions),
        }
    }

    for sess in all_sessions:
        data['neural'].append(sess['neural'])
        data['input'].append(sess['input'])
        data['output'].append(sess['output'])
        region_idx = np.array([region_to_idx[r] for r in sess['regions']], dtype=np.int64)
        data['brain_region_idx'].append(region_idx)

    output_path = '/app/converted_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved to {output_path}")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {data['subjects']}")
    print(f"Brain regions: {data['brain_regions']}")
    total_trials = sum(len(s) for s in data['neural'])
    total_neurons = sum(s[0].shape[0] for s in data['neural'] if len(s) > 0)
    print(f"Total trials: {total_trials}, Total neurons: {total_neurons}")


if __name__ == '__main__':
    main()
