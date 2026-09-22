"""
Convert the Lee, Keinath, Cianfarano & Brandon (2025) CA1 geometric-deformation dataset
(https://doi.org/10.5281/zenodo.13993254) into the decoder-compatible pickle format.

Decoder task: predict which of the 3x3 = 9 partitions of the arena the mouse occupies
(time-varying, 100 ms resolution) from CA1 calcium event activity, given the environment
geometry (which partitions are blocked) as a static per-trial input.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference code base (georepca1, src/utils.py):
  * `trace` is the already-binarized rising-phase calcium event vector (30 Hz) and is used
    directly as the firing rate (paper: "This binary vector was treated as the firing rate
    in all further analyses").
  * Only cells registered in a given session (non-NaN trace) are included -> 69,744 cell-sessions.
  * Neural temporal binning is exactly the reference decoder's (`fit_decoder`, utils.py:1776):
    gaussian_filter1d(sigma=3 frames) along time followed by AvgPool1d(kernel=3, stride=3),
    i.e. 100 ms bins. Rates are expressed in events/s (x fps).
  * Position is averaged over the same 100 ms bins and discretized on the fixed 25 cm x 25 cm
    3x3 grid of the 75 x 75 cm arena. Partition index p = 3*floor(y/25) + floor(x/25), which is
    the indexing used by the dataset's `blocked` field (verified for all 207 sessions).
"""

import argparse
import os
import pickle
import sys
import time

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------------
# Constants (all taken from the reference code / paper)
# ----------------------------------------------------------------------------------
DATA_DIR = "/app/data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
FPS = 30                      # acquisition rate of both behaviour and calcium streams
TEMPORAL_BIN_FRAMES = 3       # reference `temporal_bin_size` -> 100 ms bins
SMOOTH_SIGMA_FRAMES = 3       # reference gaussian_filter1d(sigma=temporal_bin_size)
TRIAL_SECONDS = 60            # task specification: 1-minute trials
TRIAL_FRAMES = TRIAL_SECONDS * FPS                     # 1800 frames
TRIAL_BINS = TRIAL_FRAMES // TEMPORAL_BIN_FRAMES       # 600 bins
MIN_PARTIAL_TRIAL_BINS = TRIAL_BINS // 2               # keep a trailing trial if >= 30 s
ARENA_CM = 75.0               # 75 x 75 cm square arena
N_PART = 3                    # 3 x 3 partitions
PART_CM = ARENA_CM / N_PART   # 25 cm partitions

# 3x3 environment matrices from the reference `get_env_mat` (1 = open, 0 = blocked).
# Rows are printed top-to-bottom; the dataset's `blocked` indices refer to np.flipud() of
# this matrix raveled in C order, i.e. index = 3*floor(y/25) + floor(x/25).
ENV_MATS = {
    'square':    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    'o':         [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    't':         [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
    'u':         [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
    'rectangle': [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
    '+':         [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    'i':         [[1, 1, 1], [0, 1, 0], [1, 1, 1]],
    'l':         [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
    'bit donut': [[1, 1, 1], [1, 0, 1], [0, 1, 1]],
    'glenn':     [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
}

# centre of each partition in cm, indexed by partition id p = 3*row + col
PART_CENTERS = np.array([[PART_CM * (p % N_PART) + PART_CM / 2,
                          PART_CM * (p // N_PART) + PART_CM / 2] for p in range(9)])


def env_blocked_vector(env_name):
    """Binary length-9 vector, 1 where the partition is blocked, using dataset indexing."""
    mat = np.array(ENV_MATS[env_name], dtype=float)
    open_flat = np.flipud(mat).ravel()      # index p -> 1 if open
    return (open_flat == 0).astype(np.float32)


def bin_time_series(x, kernel=TEMPORAL_BIN_FRAMES):
    """Average-pool the last axis in non-overlapping windows of `kernel` frames.

    Equivalent to torch.nn.AvgPool1d(kernel_size=kernel, stride=kernel) used by the
    reference `fit_decoder` / `test_decoder`.
    """
    n = (x.shape[-1] // kernel) * kernel
    x = x[..., :n]
    return x.reshape(*x.shape[:-1], n // kernel, kernel).mean(axis=-1)


def position_to_partition(pos_binned, blocked_vec):
    """Map binned (x, y) positions in cm to a 3x3 partition index 0-8.

    pos_binned: (2, nbins) array of x, y in cm.
    blocked_vec: (9,) binary vector, 1 = partition blocked in this session.

    Frames tracked inside a blocked partition (tracking noise; 0.003% of all frames) are
    re-assigned to the nearest open partition, mirroring the reference decoder's step that
    snaps actual/predicted positions to valid spatial bins (`decode_position_within`).
    """
    col = np.clip((pos_binned[0] // PART_CM).astype(int), 0, N_PART - 1)
    row = np.clip((pos_binned[1] // PART_CM).astype(int), 0, N_PART - 1)
    p = (N_PART * row + col).astype(np.int64)
    bad = blocked_vec[p] > 0
    if np.any(bad):
        open_ids = np.flatnonzero(blocked_vec == 0)
        d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None], axis=2)
        p[bad] = open_ids[np.argmin(d, axis=1)]
    return p, int(np.sum(bad))


def trial_slices(n_bins):
    """Non-overlapping 1-minute trials; a trailing partial trial is kept if >= 30 s."""
    slices = []
    n_full = n_bins // TRIAL_BINS
    for t in range(n_full):
        slices.append((t * TRIAL_BINS, (t + 1) * TRIAL_BINS))
    rem = n_bins - n_full * TRIAL_BINS
    if rem >= MIN_PARTIAL_TRIAL_BINS:
        slices.append((n_full * TRIAL_BINS, n_bins))
    return slices


def process_session(trace_day, position_day, env_name, blocked_field):
    """Process a single session (animal-day).

    trace_day: (n_cells_all, n_frames) binary events, NaN rows for unregistered cells
    position_day: (2, n_frames) x, y position in cm
    Returns dict with neural (list of trials), output (list), input vector, and diagnostics.
    """
    # --- neuron curation: keep cells registered in this session (non-NaN) ---
    registered = ~np.isnan(trace_day[:, 0])
    assert np.all(np.isnan(trace_day[~registered]).all(axis=1)), \
        "unregistered cells must be NaN for the entire session"
    tr = trace_day[registered].astype(np.float32)
    assert np.all(np.isin(tr, (0.0, 1.0))), "trace must be binary for registered cells"

    # --- neural temporal processing (reference fit_decoder) ---
    tr_s = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA_FRAMES, axis=1)
    rates = (bin_time_series(tr_s) * FPS).astype(np.float32)      # events / s, (n_cells, n_bins)

    # --- behaviour: same temporal bins, then spatial discretization ---
    pos_b = bin_time_series(position_day.astype(np.float64))       # (2, n_bins)
    blocked_vec = env_blocked_vector(env_name)

    # consistency check against the dataset's own `blocked` field
    bl = np.atleast_1d(np.asarray(blocked_field[0]).ravel()).astype(int)
    from_field = np.zeros(9, dtype=np.float32)
    if not (bl.size == 1 and bl[0] == -1):
        from_field[bl] = 1
    assert np.array_equal(from_field, blocked_vec), \
        f"blocked field {bl} disagrees with env matrix for '{env_name}'"

    part, n_snapped = position_to_partition(pos_b, blocked_vec)

    n_bins = rates.shape[1]
    assert pos_b.shape[1] == n_bins

    neural, output, inputs = [], [], []
    for (a, b) in trial_slices(n_bins):
        neural.append(np.ascontiguousarray(rates[:, a:b]))
        output.append(part[a:b][None, :].astype(np.int64))
        inputs.append(blocked_vec.copy())

    return {
        'neural': neural, 'output': output, 'input': inputs,
        'n_neurons': int(registered.sum()),
        'n_bins': n_bins,
        'n_snapped': n_snapped,
        'blocked_vec': blocked_vec,
        'rates': rates, 'pos_b': pos_b, 'part': part, 'tr': tr,
    }


def plot_processing(session_id, res, position_day, trace_day, outdir='/app'):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rates, pos_b, part = res['rates'], res['pos_b'], res['part']
    tr = res['tr']
    fig, ax = plt.subplots(6, 1, figsize=(16, 20))

    # 1. raw binary events (raster) for the first 60 s and first 40 neurons
    nshow = min(40, tr.shape[0])
    ax[0].imshow(tr[:nshow, :TRIAL_FRAMES], aspect='auto', cmap='Greys', interpolation='nearest',
                 extent=[0, TRIAL_SECONDS, nshow, 0])
    ax[0].set_title(f'{session_id}: raw binarized calcium events (30 Hz), trial 0, first {nshow} neurons')
    ax[0].set_xlabel('time in trial (s)'); ax[0].set_ylabel('neuron')

    # 2. processed rates for the same window
    ax[1].imshow(rates[:nshow, :TRIAL_BINS], aspect='auto', cmap='viridis',
                 extent=[0, TRIAL_SECONDS, nshow, 0])
    ax[1].set_title('converted neural: gaussian(sigma=3 frames) + 3-frame average pooling, events/s (100 ms bins)')
    ax[1].set_xlabel('time in trial (s)'); ax[1].set_ylabel('neuron')

    # 3. alignment check: one neuron, raw vs processed on the same time axis
    c = int(np.argmax(tr[:, :TRIAL_FRAMES].sum(axis=1)))
    t_raw = np.arange(TRIAL_FRAMES) / FPS
    t_bin = (np.arange(TRIAL_BINS) + 0.5) * TEMPORAL_BIN_FRAMES / FPS
    ax[2].vlines(t_raw[tr[c, :TRIAL_FRAMES] > 0], 0, 1, color='k', lw=0.8, label='raw events')
    ax[2].plot(t_bin, rates[c, :TRIAL_BINS] / max(rates[c, :TRIAL_BINS].max(), 1e-9),
               'r-', lw=1, label='processed rate (normalized)')
    ax[2].set_title(f'temporal alignment check, neuron {c}')
    ax[2].set_xlabel('time in trial (s)'); ax[2].legend(loc='upper right')

    # 4. position and discretization for the first 3 trials
    nb = min(3 * TRIAL_BINS, pos_b.shape[1])
    tt = np.arange(nb) * TEMPORAL_BIN_FRAMES / FPS
    raw_t = np.arange(nb * TEMPORAL_BIN_FRAMES) / FPS
    ax[3].plot(raw_t, position_day[0, :nb * TEMPORAL_BIN_FRAMES], color='0.7', lw=0.5, label='raw x (30 Hz)')
    ax[3].plot(raw_t, position_day[1, :nb * TEMPORAL_BIN_FRAMES], color='0.85', lw=0.5, label='raw y (30 Hz)')
    ax[3].plot(tt, pos_b[0, :nb], 'b-', lw=1, label='binned x')
    ax[3].plot(tt, pos_b[1, :nb], 'g-', lw=1, label='binned y')
    for v in (25, 50):
        ax[3].axhline(v, color='k', ls=':', lw=0.8)
    for tb in range(1, 3):
        ax[3].axvline(tb * TRIAL_SECONDS, color='r', ls='--', lw=1)
    ax[3].set_title('position (cm) with 25 cm partition boundaries (dotted) and trial boundaries (red)')
    ax[3].set_xlabel('time in session (s)'); ax[3].legend(loc='upper right', ncol=2)

    # 5. resulting discrete output
    ax[4].step(tt, part[:nb], where='post', color='m')
    ax[4].set_yticks(range(9)); ax[4].set_ylim(-0.5, 8.5)
    for tb in range(1, 3):
        ax[4].axvline(tb * TRIAL_SECONDS, color='r', ls='--', lw=1)
    ax[4].set_title('converted output: partition index p = 3*floor(y/25) + floor(x/25)')
    ax[4].set_xlabel('time in session (s)'); ax[4].set_ylabel('partition')

    # 6. occupancy over partitions vs blocked input
    occ = np.bincount(part, minlength=9) / part.size
    ax[5].bar(np.arange(9) - 0.2, occ, width=0.4, label='occupancy fraction')
    ax[5].bar(np.arange(9) + 0.2, res['blocked_vec'], width=0.4, label='decoder input (blocked)')
    ax[5].set_xticks(range(9)); ax[5].set_xlabel('partition'); ax[5].legend()
    ax[5].set_title('occupancy per partition vs blocked-partition input vector '
                    f"(n time bins snapped out of blocked partitions: {res['n_snapped']})")

    fig.tight_layout()
    fn = os.path.join(outdir, f'processing_{session_id}.png')
    fig.savefig(fn, dpi=100)
    plt.close(fig)
    print(f'    wrote {fn}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    args = ap.parse_args()
    sample = args.sample and not args.full

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': [], 'subject_idx': [],
        'brain_regions': ['CA1'], 'brain_region_idx': [],
        'input_names': [f'blocked_partition_{p}' for p in range(9)],
        'output_names': ['position_partition'],
        'output_values': [[f'p{p} (x bin {p % 3}, y bin {p // 3})' for p in range(9)]],
        'metadata': {},
    }
    session_info = []
    n_plotted = 0
    t_start = time.time()
    total_cell_sessions = 0
    total_snapped = 0
    total_bins = 0

    animals = ANIMALS[3:4] if sample else ANIMALS   # sample: the smallest animal
    for ai, animal in enumerate(animals):
        t0 = time.time()
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        t_load = time.time() - t0
        envs = np.asarray(dat['envs']).ravel()
        n_days = dat['trace'].shape[0]
        if sample:
            n_days = 2
        subject_idx = len(data['subjects'])
        data['subjects'].append(animal)
        print(f'{animal}: loaded in {t_load:.1f}s, {n_days} sessions, '
              f'{dat["trace"].shape[1]} registered-or-not cells, {dat["trace"].shape[2]} frames')

        t1 = time.time()
        for d in range(n_days):
            res = process_session(dat['trace'][d], dat['position'][d], envs[d], dat['blocked'][d])
            data['neural'].append(res['neural'])
            data['input'].append(res['input'])
            data['output'].append(res['output'])
            data['subject_idx'].append(subject_idx)
            data['brain_region_idx'].append(np.zeros(res['n_neurons'], dtype=np.int64))
            session_id = f'{animal}_day{d:02d}_{envs[d].replace(" ", "-")}'
            session_info.append({
                'session_id': session_id,
                'subject': animal,
                'day': int(d),
                'environment': str(envs[d]),
                'blocked_partitions': [int(x) for x in np.flatnonzero(res['blocked_vec'])],
                'n_neurons': res['n_neurons'],
                'n_trials': len(res['neural']),
                'n_time_bins': res['n_bins'],
            })
            total_cell_sessions += res['n_neurons']
            total_snapped += res['n_snapped']
            total_bins += res['n_bins']
            if args.show_processing and n_plotted < 2:
                plot_processing(session_id, res, dat['position'][d], dat['trace'][d])
                n_plotted += 1
            del res
        print(f'  processed {n_days} sessions in {time.time() - t1:.1f}s '
              f'({(time.time() - t1) / n_days:.2f}s/session)')
        del dat

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    data['metadata'] = {
        'task_description': (
            'Mice freely foraged for 40 min/day in a 75 x 75 cm arena partitioned into a 3 x 3 grid; '
            'across days the geometry was changed by blocking subsets of the 9 partitions (10 distinct '
            'geometries, sequence repeated up to 3 times per mouse). The decoder receives CA1 calcium '
            'event rates plus the static environment geometry (which of the 9 partitions are blocked) and '
            'predicts which of the 9 partitions the animal occupies at each 100 ms time bin.'),
        'time_bin_size': 1000.0 * TEMPORAL_BIN_FRAMES / FPS,   # 100.0 ms
        'temporal_alignment_event': (
            'start of each 1-minute trial; trial k starts 60*k s after the start of the continuous '
            '40-min recording session (there is no explicit task event - free foraging)'),
        'off_start': 0.0,
        'off_end': float(TRIAL_SECONDS),
        'session_info': session_info,
        'n_sessions': len(data['neural']),
        'n_subjects': len(data['subjects']),
        'recording_modality': 'one-photon miniscope calcium imaging (UCLA miniscope v3), 30 Hz',
        'neural_units': 'calcium events per second (binarized transient rising phases, '
                        'gaussian-smoothed with sigma = 3 frames then averaged in 3-frame/100 ms bins)',
        'neural_preprocessing': (
            'Authors\' preprocessed binary rising-phase event trains (trace field) for all cells '
            'registered in that session (non-NaN). Temporal processing follows the reference decoder '
            '(georepca1 src/utils.py fit_decoder): gaussian_filter1d(sigma=3 frames) then '
            'AvgPool1d(kernel_size=3, stride=3); values multiplied by 30 to express events/s.'),
        'output_discretization': (
            'x,y head position (DeepLabCut) averaged within each 100 ms bin, then assigned to a '
            '25 cm x 25 cm partition: p = 3*floor(y/25) + floor(x/25) (same indexing as the dataset '
            'blocked field). Bins landing in a walled-off partition (0.003% of frames, tracking noise) '
            'are snapped to the nearest open partition.'),
        'spatial_bin_size_cm': PART_CM,
        'arena_size_cm': ARENA_CM,
        'fps': FPS,
        'source': ('Lee, Keinath, Cianfarano & Brandon (2025) Neuron 113:307-320, '
                   'dataset https://doi.org/10.5281/zenodo.13993254'),
    }

    # ---------------- sanity checks ----------------
    ns = len(data['neural'])
    assert len(data['input']) == ns and len(data['output']) == ns
    assert len(data['brain_region_idx']) == ns and len(data['subject_idx']) == ns
    ntrials_all = 0
    for s in range(ns):
        nn = data['neural'][s][0].shape[0]
        assert len(data['input'][s]) == len(data['neural'][s]) == len(data['output'][s])
        assert len(data['neural'][s]) >= 2, f'session {s} has < 2 trials'
        for tr_i in range(len(data['neural'][s])):
            nrl, out, inp = data['neural'][s][tr_i], data['output'][s][tr_i], data['input'][s][tr_i]
            assert nrl.shape[0] == nn and nrl.dtype == np.float32
            assert out.shape == (1, nrl.shape[1]) and out.dtype == np.int64
            assert inp.shape == (9,)
            assert out.min() >= 0 and out.max() <= 8
            assert np.all(inp[out[0]] == 0), 'output partition must never be a blocked one'
            assert np.isfinite(nrl).all()
            ntrials_all += 1

    print('\n=== conversion summary ===')
    print(f'sessions: {ns}, subjects: {len(data["subjects"])}, trials: {ntrials_all}')
    print(f'registered cell-sessions (= number of rate maps in paper): {total_cell_sessions}')
    nneur = [data['neural'][s][0].shape[0] for s in range(ns)]
    print(f'neurons/session: mean {np.mean(nneur):.1f}, min {np.min(nneur)}, max {np.max(nneur)}')
    print(f'total time bins: {total_bins} ({total_bins * 0.1 / 3600:.2f} h), '
          f'time bins snapped out of blocked partitions: {total_snapped} '
          f'({100.0 * total_snapped / max(total_bins, 1):.4f}%)')
    occ = np.zeros(9)
    for s in range(ns):
        for o in data['output'][s]:
            occ += np.bincount(o[0], minlength=9)
    print('output distribution over partitions:', np.round(occ / occ.sum(), 4))
    allr = np.concatenate([data['neural'][s][0].ravel() for s in range(ns)])
    print(f'neural rate range: [{allr.min():.3f}, {allr.max():.3f}] events/s, mean {allr.mean():.3f}')

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')
    print(f'total elapsed {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
