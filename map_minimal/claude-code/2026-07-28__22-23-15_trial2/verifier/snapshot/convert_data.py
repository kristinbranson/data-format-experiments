"""
Convert NWB data from the MAP dataset to the standardized decoder format.

Reference: "Brain-wide analysis reveals movement encoding structured across and within brain areas"
Data: "Brain-wide neural activity underlying memory-guided movement"

Processing:
- Temporal alignment: Go cue onset
- Time window: -2.5s to +1.5s relative to go cue (4s total)
- Bin size: 50ms (non-overlapping) -> 80 time bins
- Neural data: spike counts per bin, converted to firing rates (spikes/s)
- Unit filtering: classification == 'good' and non-empty anno_name
- Trial coverage: only include trials covered by neural recordings (obs_intervals)
- Session filtering: >65% correct on control trials, >=50 correct left + right control trials
- Trial filtering: exclude auto_water and free_water trials (keep early lick and ignore)
"""

import os
import sys
import glob
import json
import pickle
import argparse
import numpy as np
from collections import OrderedDict
from pynwb import NWBHDF5IO
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5  # seconds relative to go cue
END_TIME = 1.5    # seconds relative to go cue
TONE_ONSET_REL = -1.85  # tone onset relative to go cue (sample 0.65s + delay 1.2s)
MIN_CORRECT_PER_SIDE = 50  # minimum correct trials per side for session inclusion
MIN_PERFORMANCE = 0.65  # minimum performance for session inclusion

# Time bin edges and centers
N_BINS = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))  # 80
BIN_EDGES = np.linspace(BEGIN_TIME, END_TIME, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

# Precompute time_from_tone (same for all trials)
TIME_FROM_TONE = (BIN_CENTERS - TONE_ONSET_REL).astype(np.float32)


def map_anno_to_region(anno_name):
    """Map a CCF annotation name to one of 14 broad brain regions."""
    if not anno_name or anno_name.strip() == '':
        return None

    al = anno_name.strip().lower()

    # ALM: Secondary motor area (= Anterior Lateral Motor cortex)
    if 'secondary motor area' in al:
        return 'ALM'

    # Orbital: Orbital areas and Frontal pole
    if 'orbital area' in al or 'frontal pole' in al:
        return 'Orbital'

    # Striatum
    if al in ('caudoputamen', 'striatum', 'fundus of striatum', 'nucleus accumbens'):
        return 'Striatum'

    # Pallidum (including septal nuclei, bed nuclei, substantia innominata per Allen CCF)
    if ('globus pallidus' in al or al == 'pallidum' or
        'septal nucleus' in al or 'septofimbrial' in al or
        'triangular nucleus of septum' in al or
        'substantia innominata' in al or
        'bed nuclei of the stria terminalis' in al):
        return 'Pallidum'

    # Hypothalamus (including zona incerta, subthalamic, fields of forel)
    if ('hypothalamus' in al or 'hypothalamic' in al or
        'zona incerta' in al or 'fields of forel' in al or
        'subthalamic' in al or 'lateral preoptic' in al or
        'tuberomammillary' in al or 'parasubthalamic' in al or
        'posterior hypothalamic' in al or 'lateral hypothalamic' in al):
        return 'Hypothalamus'

    # Thalamus (check AFTER hypothalamus to avoid matching "hypothalamus")
    if ('thalamus' in al or 'habenula' in al or
        'lateral geniculate' in al or 'medial geniculate' in al or
        'suprageniculate' in al or
        'paracentral nucleus' in al or 'parafascicular nucleus' in al or
        'anterodorsal nucleus' in al or 'anteromedial nucleus' in al or
        'rhomboid nucleus' in al or 'perireunensis nucleus' in al):
        return 'Thalamus'

    # Cerebellum
    if ('cerebellum' in al or 'lobule' in al or 'lobules' in al or
        'simple lobule' in al or 'copula' in al or 'nodulus' in al or
        'uvula' in al or 'pyramus' in al or 'declive' in al or
        'paramedian lobule' in al or 'crus ' in al or 'lingula' in al or
        'fastigial' in al or 'interposed nucleus' in al or
        'infracerebellar' in al or 'dentate nucleus' in al):
        return 'Cerebellum'

    # Midbrain
    if ('midbrain' in al or 'superior colliculus' in al or
        'inferior colliculus' in al or 'substantia nigra' in al or
        'red nucleus' in al or 'ventral tegmental' in al or
        'pretectal' in al or 'pedunculopontine' in al or
        'periaqueductal' in al or 'peripeduncular' in al or
        'nucleus of the optic tract' in al or 'nucleus sagulum' in al or
        'subparafascicular nucleus' in al or 'subparafascicular area' in al or
        'accessory optic tract' in al):
        return 'Midbrain'

    # Pons
    if ('pontine' in al or al == 'pons' or 'parabrachial' in al or
        'tegmental reticular' in al or 'koelliker' in al or
        'locus ceruleus' in al or 'lateral lemniscus' in al or
        'brachium of the inferior colliculus' in al):
        return 'Pons'

    # Medulla
    if ('medulla' in al or 'medullary' in al or
        'gigantocellular' in al or 'magnocellular' in al or
        'parvicellular' in al or 'paragigantocellular' in al or
        'intermediate reticular' in al or 'inferior olivary' in al or
        'solitary tract' in al or 'vagus' in al or
        'vestibular' in al or 'facial motor' in al or
        'trigeminal' in al or 'hypoglossal' in al or
        'external cuneate' in al or 'lateral reticular nucleus' in al or
        'parasolitary' in al or 'parapyramidal' in al or
        'nucleus raphe' in al or 'nucleus of roller' in al or
        'nucleus x' in al):
        return 'Medulla'

    # Hippocampus
    if ('field ca' in al or 'dentate gyrus' in al or
        'subiculum' in al or 'postsubiculum' in al or
        'hippocampal' in al or 'entorhinal' in al):
        return 'Hippocampus'

    # Olfactory
    if ('olfactory' in al or 'piriform' in al or 'taenia tecta' in al):
        return 'Olfactory'

    # Cortical subplate (amygdala, claustrum, endopiriform)
    if ('amygdal' in al or 'claustrum' in al or 'endopiriform' in al or
        'cortical subplate' in al or 'intercalated' in al):
        return 'CorticalSubplate'

    # OtherCortex: remaining cortical areas
    cortex_keywords = [
        'primary motor', 'primary somatosensory', 'primary visual',
        'primary auditory', 'supplemental somatosensory',
        'gustatory', 'visceral area', 'retrosplenial',
        'temporal association', 'ectorhinal', 'perirhinal',
        'anterior cingulate', 'prelimbic', 'infralimbic',
        'agranular insular', 'dorsal peduncular',
        'dorsal auditory', 'ventral auditory',
        'posteromedial visual', 'anteromedial visual',
    ]
    for kw in cortex_keywords:
        if kw in al:
            return 'OtherCortex'

    # If contains "layer" it's likely cortex
    if 'layer' in al:
        return 'OtherCortex'

    # Fallback
    print(f"  WARNING: Unmapped annotation: '{anno_name}'")
    return None


def process_session(nwb_file, verbose=True):
    """
    Process a single NWB file (session) and return formatted data.

    Returns:
        session_data dict or None if session doesn't meet criteria.
    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb = io.read()

    sess_name = os.path.basename(nwb_file)
    subject_id = nwb.subject.subject_id
    subject_desc = nwb.subject.description  # e.g. 'SC015'

    # --- Get trial info ---
    trials = nwb.trials.to_dataframe()
    n_trials_total = len(trials)

    # --- Get go cue times ---
    be = nwb.acquisition['BehavioralEvents']
    go_start_times = be.time_series['go_start_times'].timestamps[:]
    assert len(go_start_times) == n_trials_total

    # --- Extract trial-level arrays ---
    trial_instruction = trials['trial_instruction'].values
    early_lick = trials['early_lick'].values
    outcome = trials['outcome'].values
    photostim_power = trials['photostim_power'].values
    auto_water = trials['auto_water'].values
    free_water = trials['free_water'].values

    no_photostim = np.array([str(p) == 'N/A' or str(p) == 'nan' or
                             (isinstance(p, (int, float)) and p == 0)
                             for p in photostim_power])

    # --- Session filtering: compute performance on ALL control trials (entire session) ---
    # Control = no photostim, no auto_water, no free_water, no early lick, responded
    all_valid = (auto_water == 0) & (free_water == 0)
    control_mask_all = (all_valid &
                        (early_lick == 'no early') &
                        (outcome != 'ignore') &
                        no_photostim)

    n_control = control_mask_all.sum()
    if n_control == 0:
        if verbose:
            print(f"  SKIP: no control trials")
        io.close()
        return None

    n_correct = ((outcome == 'hit') & control_mask_all).sum()
    performance = n_correct / n_control

    correct_left = ((outcome == 'hit') & (trial_instruction == 'left') & control_mask_all).sum()
    correct_right = ((outcome == 'hit') & (trial_instruction == 'right') & control_mask_all).sum()

    if performance < MIN_PERFORMANCE:
        if verbose:
            print(f"  SKIP: perf {performance:.1%} < {MIN_PERFORMANCE:.0%}")
        io.close()
        return None

    if correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
        if verbose:
            print(f"  SKIP: correct L={correct_left}, R={correct_right} (need {MIN_CORRECT_PER_SIDE})")
        io.close()
        return None

    # --- Get unit info ---
    units = nwb.units
    n_units_total = len(units)
    classification = units['classification'].data[:]
    anno_names = units['anno_name'].data[:]

    # Filter to good units with valid annotations
    good_indices = []
    unit_regions = []
    for ui in range(n_units_total):
        if classification[ui] != 'good':
            continue
        region = map_anno_to_region(anno_names[ui])
        if region is not None:
            good_indices.append(ui)
            unit_regions.append(region)

    n_good = len(good_indices)
    if n_good == 0:
        if verbose:
            print(f"  SKIP: no good units with valid regions")
        io.close()
        return None

    # --- Determine trial coverage from obs_intervals ---
    # obs_intervals correspond to consecutive trials, but may not start at trial 0.
    # Match obs_intervals to trial indices by start time.
    obs_0 = units.get_unit_obs_intervals(good_indices[0])
    n_obs_trials = len(obs_0)

    for ui in good_indices[1:]:
        oi = units.get_unit_obs_intervals(ui)
        if len(oi) != n_obs_trials:
            n_obs_trials = min(n_obs_trials, len(oi))

    # Find which trial index the first obs_interval corresponds to
    trial_starts = trials['start_time'].values
    first_obs_start = obs_0[0, 0]
    first_recorded_trial = np.argmin(np.abs(trial_starts - first_obs_start))

    # Recorded trials: first_recorded_trial to first_recorded_trial + n_obs_trials - 1
    recorded_trial_indices = np.arange(first_recorded_trial, first_recorded_trial + n_obs_trials)
    recorded_trial_indices = recorded_trial_indices[recorded_trial_indices < n_trials_total]

    # Only use recorded trials that pass trial-level filters (no auto_water, no free_water)
    valid_trial_mask = np.zeros(n_trials_total, dtype=bool)
    for ti in recorded_trial_indices:
        if auto_water[ti] == 0 and free_water[ti] == 0:
            valid_trial_mask[ti] = True

    valid_indices = np.where(valid_trial_mask)[0]
    n_trials = len(valid_indices)

    if n_trials < 2:
        if verbose:
            print(f"  SKIP: only {n_trials} valid trials")
        io.close()
        return None

    # --- Pre-load all spike times for good units (faster than per-unit reads) ---
    all_spike_times = []
    for ui in good_indices:
        st = units.get_unit_spike_times(ui)
        all_spike_times.append(st)

    # --- Compute firing rates for each trial ---
    neural_data = []
    for trial_idx in valid_indices:
        go_time = go_start_times[trial_idx]
        trial_fr = np.zeros((n_good, N_BINS), dtype=np.float32)
        abs_start = go_time + BEGIN_TIME
        abs_end = go_time + END_TIME

        for i, st in enumerate(all_spike_times):
            # Fast filter to window
            lo = np.searchsorted(st, abs_start)
            hi = np.searchsorted(st, abs_end)
            if hi > lo:
                rel_spikes = st[lo:hi] - go_time
                counts, _ = np.histogram(rel_spikes, bins=BIN_EDGES)
                trial_fr[i, :] = counts / BIN_WIDTH

        neural_data.append(trial_fr)

    # --- Get photostim times ---
    photostim_start_ts = be.time_series['photostim_start_times'].timestamps[:]
    photostim_stop_ts = be.time_series['photostim_stop_times'].timestamps[:]

    # --- Get tongue tracking data ---
    bt = nwb.acquisition['BehavioralTimeSeries']
    tongue_ts = bt.time_series['Camera0_side_TongueTracking']
    tongue_y_all_data = tongue_ts.data[:, 1]  # y column only
    tongue_timestamps = tongue_ts.timestamps[:]

    # Compute tongue y percentiles over entire session
    tongue_y_p40 = np.percentile(tongue_y_all_data, 40)
    tongue_y_p60 = np.percentile(tongue_y_all_data, 60)

    # --- Compute inputs and outputs per trial ---
    input_data = []
    output_data = []

    for trial_idx in valid_indices:
        go_time = go_start_times[trial_idx]

        # --- Input 1: Time from tone onset (same for all trials) ---
        # --- Input 2: Photostimulation on/off ---
        photostim_binary = np.zeros(N_BINS, dtype=np.float32)
        trial_start_abs = go_time + BEGIN_TIME
        trial_end_abs = go_time + END_TIME

        for ps_start, ps_stop in zip(photostim_start_ts, photostim_stop_ts):
            if ps_stop > trial_start_abs and ps_start < trial_end_abs:
                rel_start = ps_start - go_time
                rel_stop = ps_stop - go_time
                # Vectorized bin marking
                mask = (BIN_CENTERS >= rel_start) & (BIN_CENTERS < rel_stop)
                photostim_binary[mask] = 1.0

        input_trial = np.stack([TIME_FROM_TONE, photostim_binary], axis=0)
        input_data.append(input_trial)

        # --- Outputs ---
        instr = trial_instruction[trial_idx]
        outc = outcome[trial_idx]

        # Choice: for hit, choice = instruction; for miss, choice = opposite; for ignore, use instruction
        if outc == 'hit':
            choice = 0 if instr == 'left' else 1
        elif outc == 'miss':
            choice = 1 if instr == 'left' else 0
        else:  # ignore
            choice = 0 if instr == 'left' else 1

        outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outc]
        early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0

        # Tongue y-position (time-varying, discretized)
        # Vectorized: find closest tongue timestamp for each bin center
        abs_times = go_time + BIN_CENTERS
        tongue_indices = np.searchsorted(tongue_timestamps, abs_times)
        tongue_indices = np.clip(tongue_indices, 0, len(tongue_timestamps) - 1)
        # Check if previous index is closer
        prev_indices = np.clip(tongue_indices - 1, 0, len(tongue_timestamps) - 1)
        dist_curr = np.abs(tongue_timestamps[tongue_indices] - abs_times)
        dist_prev = np.abs(tongue_timestamps[prev_indices] - abs_times)
        use_prev = dist_prev < dist_curr
        tongue_indices[use_prev] = prev_indices[use_prev]

        tongue_y_values = tongue_y_all_data[tongue_indices]
        tongue_y_disc = np.zeros(N_BINS, dtype=np.float32)
        tongue_y_disc[tongue_y_values >= tongue_y_p40] = 1
        tongue_y_disc[tongue_y_values > tongue_y_p60] = 2

        # Stack per-trial scalar outputs (broadcast to time bins) and time-varying outputs
        # Use int type for categorical outputs
        output_trial = np.stack([
            np.full(N_BINS, choice, dtype=np.int64),
            np.full(N_BINS, outcome_val, dtype=np.int64),
            np.full(N_BINS, early_lick_val, dtype=np.int64),
            tongue_y_disc.astype(np.int64),
        ], axis=0)
        output_data.append(output_trial)

    if verbose:
        print(f"  OK: {n_good} units, {n_trials} trials (of {n_trials_total}, rec={n_obs_trials}), "
              f"perf={performance:.1%}, L={correct_left}, R={correct_right}, sub={subject_desc}")

    io.close()

    return {
        'neural': neural_data,
        'input': input_data,
        'output': output_data,
        'unit_regions': unit_regions,
        'subject_id': subject_id,
        'subject_desc': subject_desc,
        'n_good': n_good,
        'n_trials': n_trials,
        'performance': performance,
        'correct_left': correct_left,
        'correct_right': correct_right,
        'sess_name': sess_name,
    }


def convert_all(data_dir='data', output_file='converted_data.pkl', sample_output_file='sample_data.pkl',
                max_sessions=None, verbose=True):
    """Convert all NWB files to the standardized format."""

    nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*/sub-*_ses-*.nwb')))
    print(f"Found {len(nwb_files)} NWB files")

    if max_sessions is not None:
        nwb_files = nwb_files[:max_sessions]
        print(f"Processing first {max_sessions} files")

    all_sessions = []
    all_subjects = OrderedDict()
    all_regions = OrderedDict()

    for i, nwb_file in enumerate(nwb_files):
        print(f"[{i+1}/{len(nwb_files)}] {os.path.basename(nwb_file)}")
        try:
            result = process_session(nwb_file, verbose=verbose)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        if result is not None:
            all_sessions.append(result)
            sub_desc = result['subject_desc']
            if sub_desc not in all_subjects:
                all_subjects[sub_desc] = result['subject_id']
            for region in result['unit_regions']:
                if region not in all_regions:
                    all_regions[region] = len(all_regions)

    print(f"\nProcessed {len(all_sessions)} valid sessions from {len(all_subjects)} subjects")
    print(f"Brain regions ({len(all_regions)}): {list(all_regions.keys())}")

    subjects = list(all_subjects.keys())
    brain_regions = list(all_regions.keys())

    neural = []
    inputs = []
    outputs = []
    subject_idx = []
    brain_region_idx = []
    total_units = 0
    total_trials = 0

    for sess in all_sessions:
        neural.append(sess['neural'])
        inputs.append(sess['input'])
        outputs.append(sess['output'])
        subject_idx.append(subjects.index(sess['subject_desc']))
        region_indices = np.array([all_regions[r] for r in sess['unit_regions']], dtype=np.int64)
        brain_region_idx.append(region_indices)
        total_units += sess['n_good']
        total_trials += sess['n_trials']

    subject_idx = np.array(subject_idx, dtype=np.int64)

    input_names = ['time_from_tone_onset', 'photostimulation']
    output_names = ['choice', 'outcome', 'early_lick', 'tongue_y_position']
    output_values = [
        ['left', 'right'],
        ['ignore', 'miss', 'hit'],
        ['no', 'yes'],
        ['low', 'mid', 'high'],
    ]

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Auditory delayed response task: mice report tone frequency (3 or 12 kHz) by licking left or right after a delay period.',
            'time_bin_size': BIN_WIDTH * 1000,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': BEGIN_TIME,
            'off_end': END_TIME,
            'tone_onset_relative_to_go': TONE_ONSET_REL,
            'sample_period': 0.65,
            'delay_period': 1.2,
            'n_sessions': len(all_sessions),
            'n_subjects': len(subjects),
            'n_brain_regions': len(brain_regions),
            'total_units': total_units,
            'total_trials': total_trials,
            'session_filtering': f'Performance > {MIN_PERFORMANCE:.0%}, >= {MIN_CORRECT_PER_SIDE} correct trials per lick direction on control trials',
            'unit_filtering': "classification == 'good' and non-empty CCF annotation",
            'trial_filtering': 'Excluded auto_water and free_water trials. Kept early lick and no-response trials (decoder outputs).',
        },
    }

    # Print summary
    print(f"\n=== Data Summary ===")
    print(f"Sessions: {len(all_sessions)}")
    print(f"Subjects: {len(subjects)}: {subjects}")
    print(f"Total units: {total_units}")
    print(f"Total trials: {total_trials}")
    print(f"Brain regions: {brain_regions}")
    print(f"Time bins: {N_BINS} ({BIN_WIDTH*1000:.0f} ms bins)")
    print(f"Time window: [{BEGIN_TIME}, {END_TIME}] s relative to go cue")

    region_counts = {r: 0 for r in brain_regions}
    for sess_br in brain_region_idx:
        for idx in sess_br:
            region_counts[brain_regions[idx]] += 1
    print(f"\nUnits per region:")
    for r, c in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")

    # Save full data
    print(f"\nSaving full data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved ({os.path.getsize(output_file) / 1e6:.1f} MB)")

    # Save sample data (first 5 sessions)
    n_sample = min(5, len(all_sessions))
    sample_data = {
        'neural': neural[:n_sample],
        'input': inputs[:n_sample],
        'output': outputs[:n_sample],
        'subjects': subjects,
        'subject_idx': subject_idx[:n_sample],
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx[:n_sample],
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': data['metadata'].copy(),
    }
    sample_data['metadata']['n_sessions'] = n_sample
    sample_data['metadata']['note'] = f'Sample of first {n_sample} sessions'

    print(f"Saving sample data to {sample_output_file}...")
    with open(sample_output_file, 'wb') as f:
        pickle.dump(sample_data, f)
    print(f"Saved ({os.path.getsize(sample_output_file) / 1e6:.1f} MB)")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing NWB files')
    parser.add_argument('--output', type=str, default='converted_data.pkl', help='Output pickle file')
    parser.add_argument('--sample-output', type=str, default='sample_data.pkl', help='Sample output pickle file')
    parser.add_argument('--max-sessions', type=int, default=None, help='Max sessions to process')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    args = parser.parse_args()

    convert_all(
        data_dir=args.data_dir,
        output_file=args.output,
        sample_output_file=args.sample_output,
        max_sessions=args.max_sessions,
        verbose=not args.quiet,
    )
