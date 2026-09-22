"""Convert the Lee et al. (2025) CA1 geometry dataset for neural decoding.

Decisions:
* Use the released ``trace`` arrays.  These are the manually curated, binary
  rising phases of calcium transients (z > 2.5), which the paper treats as
  firing rate; we do not redo calcium extraction or select only place cells.
* A registered cell is included in a recording session exactly when its trace
  on that day is not all NaN.  NaN rows denote cells absent on that day.
* Recordings are nominally 40 min at 30 Hz but aligned arrays differ slightly
  from 72,000 frames.  We use all aligned frames and proportional integer
  edges to form exactly 2,400 one-second bins.  Events are summed (event
  counts), while position is averaged.  This avoids inventing/dropping a final
  minute and preserves the session duration specified in the paper.
* The resulting series is split into 40 contiguous one-minute trials.
* Position is categorized on the physical 3 x 3, 75 cm arena grid.  Labels use
  the repository's x-major convention: label = 3*x_bin + y_bin.
* Inputs are the nine accessible-region indicators for the day's geometry,
  in the same x-major flattened order.  They are static per trial as requested.
"""
from pathlib import Path
import gc
import pickle
import joblib
import numpy as np

DATA_DIR = Path('/app/data')
OUT = Path('/app/converted_data.pkl')
ANIMALS = [
    'QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
    'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75'
]

# Exact get_env_mat definitions in georepca1/src/utils.py.  Axis 0 is x and
# axis 1 is y in the paper code's rate-map construction.
ENV_MATS = {
    'square':    [[1,1,1], [1,1,1], [1,1,1]],
    'o':         [[1,1,1], [1,0,1], [1,1,1]],
    't':         [[0,1,0], [0,1,0], [1,1,1]],
    'u':         [[1,1,1], [1,0,0], [1,1,1]],
    'rectangle': [[0,1,1], [0,1,1], [0,1,1]],
    '+':         [[0,1,0], [1,1,1], [0,1,0]],
    'i':         [[1,1,1], [0,1,0], [1,1,1]],
    'l':         [[1,1,1], [1,0,0], [1,0,0]],
    'bit donut': [[1,1,1], [1,0,1], [0,1,1]],
    'glenn':     [[1,1,0], [1,1,1], [0,1,1]],
}

N_SECONDS = 40 * 60
TRIAL_SECONDS = 60


def aggregate_session(trace, position):
    """Aggregate one aligned session to event counts and mean positions."""
    n_frames = trace.shape[1]
    if position.shape != (2, n_frames):
        raise ValueError(f'alignment mismatch: trace {trace.shape}, position {position.shape}')

    # A neuron is either finite for the complete day or all NaN in released data.
    keep = np.any(np.isfinite(trace), axis=1)
    tr = trace[keep]
    if not np.all(np.isfinite(tr)):
        raise ValueError('partially non-finite trace found in an included neuron')

    # np.linspace makes adjacent bins exhaustive and nonoverlapping despite the
    # small animal-specific differences in aligned recording length.
    edges = np.linspace(0, n_frames, N_SECONDS + 1, dtype=np.int64)
    starts, widths = edges[:-1], np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError('recording too short for one-second aggregation')

    # Released events are 0/1. reduceat sums each variable-width (~30 frame) bin.
    counts = np.add.reduceat(tr, starts, axis=1)
    counts = counts[:, :N_SECONDS].astype(np.float32, copy=False)
    pos_sum = np.add.reduceat(position, starts, axis=1)[:, :N_SECONDS]
    pos_mean = pos_sum / widths[None, :]

    # Physical grid boundaries are 0,25,50,75 cm. clip handles exact wall values.
    xy_bin = np.floor(pos_mean / 25.0).astype(np.int64)
    np.clip(xy_bin, 0, 2, out=xy_bin)
    labels = (3 * xy_bin[0] + xy_bin[1]).astype(np.int64)
    return counts, labels, int(keep.sum())


def convert():
    neural, inputs, outputs = [], [], []
    subject_idx, region_idx, session_info = [], [], []

    for subj_i, animal in enumerate(ANIMALS):
        print(f'Loading {animal}...', flush=True)
        outer = joblib.load(DATA_DIR / animal)
        dat = outer[animal]
        traces = dat['trace']
        positions = dat['position']
        envs = np.asarray(dat['envs']).reshape(-1)
        if not (len(traces) == len(positions) == len(envs)):
            raise ValueError(f'inconsistent session count for {animal}')

        for day, env_raw in enumerate(envs):
            env = str(env_raw)
            counts, labels, n_cells = aggregate_session(traces[day], positions[day])
            geom = np.asarray(ENV_MATS[env], dtype=np.float32).reshape(9)

            sess_neural, sess_input, sess_output = [], [], []
            for start in range(0, N_SECONDS, TRIAL_SECONDS):
                stop = start + TRIAL_SECONDS
                # Copies keep the pickle independent of the full-session buffers.
                sess_neural.append(np.ascontiguousarray(counts[:, start:stop]))
                sess_input.append(geom.copy())
                sess_output.append(labels[None, start:stop].copy())
            neural.append(sess_neural)
            inputs.append(sess_input)
            outputs.append(sess_output)
            subject_idx.append(subj_i)
            region_idx.append(np.zeros(n_cells, dtype=np.int64))
            session_info.append({
                'subject': animal, 'day_index': int(day), 'environment': env,
                'source_frames': int(traces.shape[-1]),
                'n_neurons': n_cells, 'n_trials': 40,
            })

        del outer, dat, traces, positions
        gc.collect()

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': ANIMALS,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': region_idx,
        'input_names': [f'accessible_x{x}_y{y}' for x in range(3) for y in range(3)],
        'output_names': ['position_bin'],
        'output_values': [[f'x{x}_y{y}' for x in range(3) for y in range(3)]],
        'metadata': {
            'task_description': 'Decode mouse position in the 3 x 3 arena grid from CA1 calcium-event counts; geometry accessibility is supplied as context.',
            'time_bin_size': 1000.0,
            'temporal_alignment_event': 'start of each contiguous one-minute segment of a 40-minute free-exploration session',
            'off_start': 0.0,
            'off_end': 60.0,
            'source_sampling_rate_hz': 30.0,
            'neural_representation': 'Counts per 1 s of released binary significant calcium-transient rising phases (z > 2.5)',
            'position_processing': 'Mean aligned x/y position per 1 s, discretized at 25 and 50 cm; label = 3*x_bin + y_bin',
            'input_description': 'Nine static binary indicators (1 accessible, 0 blocked), x-major flattening of repository get_env_mat geometry',
            'trial_definition': '40 contiguous one-minute trials per nominal 40-minute recording; all aligned source frames distributed across 2400 bins',
            'session_info': session_info,
        },
    }
    with OUT.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {OUT}: {len(neural)} sessions, {sum(map(len, neural))} trials')


if __name__ == '__main__':
    convert()
