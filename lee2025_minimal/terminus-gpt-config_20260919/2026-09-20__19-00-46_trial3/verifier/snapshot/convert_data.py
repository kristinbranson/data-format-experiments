#!/usr/bin/env python3
"""Convert the CA1 geometric-deformation data for position decoding.

Decisions:
* Each of the 207 curated daily recordings is a session. Neurons are meaningful
  only within a recording for decoding; globally tracked rows which are all NaN
  on that day are absent and are removed.
* The supplied `trace` is used directly: it is the paper's thresholded binary
  rising-phase calcium-event vector, aligned sample-for-sample with `position`.
* The paper specifies 40-minute recordings. Arrays differ slightly around 30 Hz
  after timestamp alignment, so all samples are partitioned into 2400 equal
  one-second bins (rather than assuming exactly 30 frames or dropping data).
  Binary events are summed in each bin; aligned positions are averaged.
* The 75 cm square was designed as a 3x3 grid. Position is discretized with
  25-cm edges, clipping the occasional exact 75-cm coordinate into the last bin.
* Source `blocked` indices are converted to nine blocked/not-blocked channels.
  They are repeated over trial time because decoder inputs are represented with
  a common time axis, although geometry is static within each recording.
"""
import gc
import os
import pickle
import joblib
import numpy as np

DATA_DIR = '/app/data'
OUT = '/app/converted_data.pkl'
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
N_SECONDS = 40 * 60
TRIAL_SECONDS = 60


def aggregate_equal_bins(x, edges, reducer='sum'):
    """Aggregate final-axis samples using integer boundaries in edges."""
    # reduceat is unsafe for repeated final boundaries; lengths here are ~30,
    # but explicit loop is memory-efficient and unambiguous.
    if reducer == 'sum':
        return np.stack([x[..., edges[i]:edges[i+1]].sum(axis=-1)
                         for i in range(len(edges)-1)], axis=-1)
    return np.stack([x[..., edges[i]:edges[i+1]].mean(axis=-1)
                     for i in range(len(edges)-1)], axis=-1)


def main():
    neural, inputs, outputs = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []

    for subj_i, subject in enumerate(SUBJECTS):
        print('Loading', subject, flush=True)
        source = joblib.load(os.path.join(DATA_DIR, subject))[subject]
        traces = source['trace']       # session x tracked cell x aligned frame
        positions = source['position'] # session x (x,y) x aligned frame
        envs = np.asarray(source['envs']).reshape(-1)
        blocked = source['blocked']

        for day in range(traces.shape[0]):
            tr = traces[day]
            pos = positions[day]
            # In source files a cell is either present for every sample or NaN
            # for every sample in a given daily recording.
            valid = ~np.isnan(tr).any(axis=1)
            tr = tr[valid]
            if not len(tr):
                raise ValueError(f'No valid neurons: {subject}, day {day}')
            if np.isnan(pos).any():
                raise ValueError(f'NaN position: {subject}, day {day}')

            nframes = tr.shape[1]
            edges = np.linspace(0, nframes, N_SECONDS + 1).astype(np.int64)
            counts = aggregate_equal_bins(tr, edges, 'sum').astype(np.uint8)
            xy = aggregate_equal_bins(pos, edges, 'mean')

            col = np.clip(np.floor(xy[0] / 25.0).astype(np.int8), 0, 2)
            row = np.clip(np.floor(xy[1] / 25.0).astype(np.int8), 0, 2)
            spatial_bin = (row * 3 + col).astype(np.int8)[None, :]

            geometry = np.zeros(9, dtype=np.float32)
            blocked_idx = np.asarray(blocked[day]).reshape(-1).astype(int)
            blocked_idx = blocked_idx[blocked_idx >= 0]  # square uses sentinel -1
            geometry[blocked_idx] = 1.0
            geometry_ts = np.repeat(geometry[:, None], TRIAL_SECONDS, axis=1)

            ses_neural, ses_input, ses_output = [], [], []
            for start in range(0, N_SECONDS, TRIAL_SECONDS):
                stop = start + TRIAL_SECONDS
                ses_neural.append(counts[:, start:stop].copy())
                ses_input.append(geometry_ts.copy())
                ses_output.append(spatial_bin[:, start:stop].copy())
            neural.append(ses_neural)
            inputs.append(ses_input)
            outputs.append(ses_output)
            subject_idx.append(subj_i)
            brain_region_idx.append(np.zeros(counts.shape[0], dtype=np.int64))
            session_info.append({
                'subject': subject,
                'recording_index': day,
                'environment': str(envs[day]),
                'blocked_spatial_bins': blocked_idx.tolist(),
                'source_frames': int(nframes),
                'n_neurons': int(counts.shape[0]),
                'n_trials': 40,
            })
        del source, traces, positions
        gc.collect()

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': SUBJECTS,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx,
        'input_names': [f'blocked spatial bin {i}' for i in range(9)],
        'output_names': ['mouse position'],
        'output_values': [[f'row {r}, column {c}' for r in range(3) for c in range(3)]],
        'metadata': {
            'task_description': ('Decode the mouse location in a 3 x 3 spatial grid from '
                                 'CA1 binary calcium-event activity while accounting for '
                                 'the static blocked-compartment geometry.'),
            'time_bin_size': 1000.0,
            'temporal_alignment_event': 'start of each consecutive one-minute segment',
            'off_start': 0.0,
            'off_end': 60.0,
            'trial_duration_seconds': 60.0,
            'source_sampling_rate_hz': 'approximately 30; streams timestamp-aligned by source authors',
            'neural_representation': ('counts per 1-second bin of supplied binary rising-phase '
                                      'calcium transient vectors'),
            'position_binning': ('row-major 3x3 bins; x defines column, y defines row; '
                                 '25 cm bin width in 75 x 75 cm arena'),
            'input_encoding': '1 means source spatial compartment is blocked; 0 means accessible',
            'session_definition': 'one 40-minute daily recording',
            'session_info': session_info,
        }
    }
    print(f'Saving {len(neural)} sessions to {OUT}', flush=True)
    with open(OUT, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done:', os.path.getsize(OUT), 'bytes', flush=True)


if __name__ == '__main__':
    main()
