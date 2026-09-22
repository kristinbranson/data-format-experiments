#!/usr/bin/env python3
"""Convert the CA1 geometry data to the neural-decoder interchange format.

Decisions
---------
* The repository's joblib files are used rather than reprocessing calcium video.
  They contain the paper's aligned 30-Hz binary rising-transient traces and DLC
  position, produced with the processing described in the paper.
* A recording day is a session.  Neurons tracked across animals' days occupy a
  union array; an all-NaN row means that cell was not observed that day.  Such
  rows are removed session-wise.  This gives the reported 69,744 rate maps.
* Complete, non-overlapping 60-s windows are trials.  A short trailing fragment
  is omitted so all trials have identical duration.
* Thirty synchronized frames are aggregated per 1-s decoder bin.  Binary neural
  events are summed (event counts), while x/y is averaged.  This preserves event
  mass and avoids an impractically large 30-Hz decoder dataset.
* The 75-cm square is divided at 25 and 50 cm, exactly matching the experiment's
  3x3 construction.  Class is row*3+column, with x selecting column and y row.
* Geometry is the supplied `blocked` list, encoded as nine binary indicators.
  The square's sentinel -1 means no blocked cells.
"""
from pathlib import Path
import gc
import pickle
import joblib
import numpy as np

DATA_DIR = Path('/app/data')
OUT = Path('/app/converted_data.pkl')
SUBJECTS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50',
            'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74',
            'QLAK-CA1-75']
FS = 30
BIN_FRAMES = 30
TRIAL_SECONDS = 60
BINS_PER_TRIAL = TRIAL_SECONDS
TRIAL_FRAMES = FS * TRIAL_SECONDS


def scalar_env(v):
    """Normalize the (1,) numpy string used by the converted source."""
    a = np.asarray(v).reshape(-1)
    return str(a[0])


def main():
    neural, inputs, outputs = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []
    total_valid = 0

    for subj_i, subject in enumerate(SUBJECTS):
        print(f'Loading {subject}', flush=True)
        source = joblib.load(DATA_DIR / subject)[subject]
        traces = source['trace']       # day x union-neuron x frame
        positions = source['position'] # day x (x,y) x frame
        envs = source['envs']
        blocked = source['blocked']

        for day in range(traces.shape[0]):
            tr = np.asarray(traces[day])
            pos = np.asarray(positions[day])
            # Source conversion puts features first. Be defensive about raw layout.
            if pos.shape[0] != 2 and pos.shape[1] == 2:
                pos = pos.T
            if tr.shape[1] != pos.shape[1] and tr.shape[0] == pos.shape[1]:
                tr = tr.T
            if tr.shape[1] != pos.shape[1]:
                raise ValueError(f'unaligned streams: {subject} day {day}')

            finite = np.isfinite(tr)
            valid = finite.all(axis=1)
            if np.any(finite.any(axis=1) != valid):
                raise ValueError(f'partially missing neuron: {subject} day {day}')
            tr = tr[valid]
            n_neurons, n_frames = tr.shape
            n_trials = n_frames // TRIAL_FRAMES
            if n_trials < 2:
                raise ValueError(f'too few complete trials: {subject} day {day}')
            used = n_trials * TRIAL_FRAMES

            # (neuron, trial, second, frame-within-second), then trial first.
            counts = tr[:, :used].reshape(
                n_neurons, n_trials, BINS_PER_TRIAL, BIN_FRAMES
            ).sum(axis=3, dtype=np.float32).transpose(1, 0, 2)
            mean_pos = pos[:, :used].reshape(
                2, n_trials, BINS_PER_TRIAL, BIN_FRAMES
            ).mean(axis=3)
            xbin = np.clip(np.floor(mean_pos[0] / 25.0), 0, 2).astype(np.int64)
            ybin = np.clip(np.floor(mean_pos[1] / 25.0), 0, 2).astype(np.int64)
            spatial_class = ybin * 3 + xbin

            geometry = np.zeros(9, dtype=np.float32)
            block_values = np.asarray(blocked[day], dtype=float).reshape(-1)
            for b in block_values:
                if np.isfinite(b) and 0 <= int(b) < 9:
                    geometry[int(b)] = 1.0

            neural.append([counts[t] for t in range(n_trials)])
            inputs.append([geometry.copy() for _ in range(n_trials)])
            outputs.append([spatial_class[t][None, :] for t in range(n_trials)])
            subject_idx.append(subj_i)
            brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
            env = scalar_env(envs[day])
            session_info.append({
                'subject': subject,
                'source_session_index': day,
                'environment': env,
                'blocked_grid_indices': [int(b) for b in block_values if b >= 0],
                'source_frames': int(n_frames),
                'used_frames': int(used),
                'n_complete_one_minute_trials': int(n_trials),
                'n_neurons_present': int(n_neurons),
            })
            total_valid += n_neurons

        del source, traces, positions
        gc.collect()

    if len(neural) != 207 or total_valid != 69744:
        raise RuntimeError(f'unexpected source totals: {len(neural)} sessions, '
                           f'{total_valid} session-neurons')

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': SUBJECTS,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx,
        'input_names': [f'blocked_spatial_bin_{i}' for i in range(9)],
        'output_names': ['spatial_bin'],
        'output_values': [[f'row_{r}_column_{c}' for r in range(3) for c in range(3)]],
        'metadata': {
            'task_description': ('Decode mouse location in a 3 x 3 grid from CA1 '
                                 'binary calcium-event activity; arena geometry is '
                                 'provided as blocked-cell indicators.'),
            'time_bin_size': 1000.0,
            'temporal_alignment_event': 'start of each non-overlapping one-minute window',
            'off_start': 0.0,
            'off_end': 60.0,
            'source_sampling_rate_hz': 30.0,
            'neural_measurement': ('count per 1-s bin of paper-preprocessed binary '
                                   'rising-phase calcium transient events'),
            'position_processing': ('mean aligned DLC x/y per 1-s bin, discretized '
                                    'at 25 and 50 cm into class y_bin*3+x_bin'),
            'trial_definition': ('complete non-overlapping 60-s windows; trailing '
                                 'partial minute omitted'),
            'geometry_encoding': ('nine binary blocked-cell indicators in the same '
                                  'row-major index convention as source blocked lists'),
            'neuron_filter': ('session-wise removal only of all-NaN rows, which '
                              'represent tracked cells absent from that recording'),
            'session_info': session_info,
        },
    }
    print(f'Writing {OUT}: {len(neural)} sessions, '
          f'{sum(map(len, neural))} trials, {total_valid} session-neurons', flush=True)
    with OUT.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done', flush=True)


if __name__ == '__main__':
    main()
