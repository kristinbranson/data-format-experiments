#!/usr/bin/env python
"""
Convert the CA1 geometric-deformation dataset of

  Lee, Keinath, Cianfarano & Brandon (2025)
  "Identifying representational structure in CA1 to benchmark theoretical models
   of cognitive mapping", Neuron 113(2):307-320

into the decoder-compatible pickle format described in the task.

Decoder task
------------
  inputs  : environment geometry = which of the 9 partitions of the 3x3 arena are
            blocked (static, one value per trial)
  outputs : the mouse position discretised into the 3 x 3 = 9 spatial bins
            (time varying, one class label per time bin)

Processing follows the reference code (`/app/code/georepca1/src/utils.py`):
  * neural signal  : the distributed binarised rising-phase `trace` (30 Hz),
                     smoothed with gaussian_filter1d(sigma=3 frames) and averaged
                     into 100 ms bins (AvgPool1d(kernel=3, stride=3)), as in
                     `fit_decoder`/`test_decoder`, then scaled to events/s.
  * neuron curation: cells registered on that day and with > 5 events during
                     locomotion, as in `decode_position_within` (cell_threshold=5).
  * timepoint curation: only periods with smoothed speed > 5 cm/s are kept,
                     as in `decode_position_within` (v_thresh=5, v_filt_size=5).
  * spatial binning: absolute bins of 75 cm / n_bins (verified to reproduce the
                     dataset's own occupancy and rate maps exactly).
  * partition index: 3 * ybin + xbin, the indexing used by the `blocked` field
                     (verified against all 207 sessions).

Usage
-----
  python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------------------------------- #
# constants (all taken from the reference code / methods)
# --------------------------------------------------------------------------- #
DATA_DIR = '/app/data'
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

FPS = 30.0                 # acquisition rate of both streams (methods)
ARENA_SIZE = 75.0          # cm, side of the square arena (methods)
N_PART = 3                 # 3 x 3 partition grid (decoder task)
TEMPORAL_BIN_FRAMES = 3    # AvgPool1d(kernel_size=3) in fit_decoder  -> 100 ms
TRACE_SMOOTH_SIGMA = 3     # gaussian_filter1d(sigma=3) in fit_decoder
V_FILT_SIGMA = 5           # v_filt_size in decode_position_within (frames)
V_THRESH = 5.0             # v_thresh in decode_position_within (cm/s)
CELL_THRESHOLD = 5         # cell_threshold in decode_position_within (events)
TRIAL_SEC = 60.0           # 1 minute trials (decoder task)
MIN_TRIAL_FRAC = 0.5       # keep the final partial block if >= 50% of a trial
MIN_BINS_PER_TRIAL = 30    # >= 3 s of running data required to keep a trial

BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS      # 100 ms
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN_FRAMES))   # 600

# The 10 geometries, as in utils.get_env_mat (matrix as drawn in the paper).
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


def env_blocked_vector(env_name):
    """9-dim binary vector, 1 = partition blocked, indexed 3*ybin + xbin.

    Derived from utils.get_env_mat / utils.clean_rate_maps: the map mask is
    ``np.fliplr(get_env_mat(env).T)`` so mask[x, y] = env_mat[2-y, x], i.e. the
    partition at (xbin, ybin) is ``np.flipud(env_mat)[ybin, xbin]``.
    Verified against the dataset's own `blocked` field for all 207 sessions.
    """
    m = np.array(ENV_MATS[env_name], dtype=float)
    return (np.flipud(m).ravel() == 0).astype(np.float32)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def open_animal(animal):
    """Open the MATLAB v7.3 file of one animal (lazy, HDF5)."""
    return h5py.File(os.path.join(DATA_DIR, f'{animal}.mat'), 'r')


def read_meta(f):
    """Return (envs, blocked) for every day of an opened animal file."""
    envs = [''.join(chr(c) for c in f[r][:].ravel()) for r in f['envs'][0]]
    blocked = [np.atleast_1d(f[r][:].ravel()).astype(int) for r in f['blocked'][0]]
    return envs, blocked


def read_session(f, day):
    """Read one session. Returns trace (T, n_cells) and position (T, 2), both 30 Hz."""
    trace = f[f['trace'][day, 0]][:]          # (T, n_cells) float64, NaN if unregistered
    position = f[f['position'][day, 0]][:]    # (T, 2) float64, cm in [0, 75]
    return trace, position


# --------------------------------------------------------------------------- #
# processing
# --------------------------------------------------------------------------- #
def compute_speed(position):
    """Smoothed speed in cm/s, exactly as in utils.decode_position_within."""
    speed = np.zeros(position.shape[0])
    speed[1:] = gaussian_filter1d(
        np.linalg.norm(np.diff(position, axis=0) * FPS, axis=1),
        axis=0, sigma=V_FILT_SIGMA)
    return speed


def pool_mean(x, k=TEMPORAL_BIN_FRAMES):
    """Average non-overlapping blocks of k samples along axis 0 (== AvgPool1d)."""
    n = (x.shape[0] // k) * k
    x = x[:n]
    return x.reshape((n // k, k) + x.shape[1:]).mean(axis=1)


def position_to_class(position, blocked_vec=None):
    """3x3 partition index for each timepoint: 3*ybin + xbin (`blocked` convention).

    If ``blocked_vec`` is given, samples that land inside a partition that is
    physically blocked in this geometry (DeepLabCut tracking noise near the
    partition walls; 0.005% of all samples) are snapped to the nearest open
    partition. This is the 3x3 analogue of the cleaning step of
    ``utils.decode_position_within``, which moves both the true and the predicted
    position to the nearest bin that actually belongs to the environment
    (``true_bins[np.argmin(actual_norms)]``).
    """
    part = ARENA_SIZE / N_PART
    bins = np.clip((position / part).astype(int), 0, N_PART - 1)
    cls = (N_PART * bins[:, 1] + bins[:, 0]).astype(np.int64)
    if blocked_vec is not None and blocked_vec.any():
        blocked_ids = np.where(blocked_vec > 0)[0]
        bad = np.isin(cls, blocked_ids)
        if bad.any():
            open_ids = np.where(blocked_vec == 0)[0]
            centres = np.stack([(open_ids % N_PART) * part + part / 2,
                                (open_ids // N_PART) * part + part / 2], axis=1)
            d = np.linalg.norm(position[bad][:, None, :] - centres[None], axis=2)
            cls[bad] = open_ids[np.argmin(d, axis=1)]
    return cls


def process_session(trace, position, env_name, diagnostics=None):
    """Convert one recording day into a list of trials.

    Returns
        neural_trials : list of (n_neurons, n_bins) float32 arrays (events/s)
        input_trials  : list of (9,) float32 arrays (blocked partitions)
        output_trials : list of (1, n_bins) int64 arrays (3x3 position class)
        keep_cells    : boolean mask of the cells kept (length = n_cells in file)
    """
    n_frames = trace.shape[0]

    # ---- 1. speed and locomotion mask at 30 Hz (reference: v_thresh = 5 cm/s)
    speed = compute_speed(position)
    moving = speed > V_THRESH

    # ---- 2. neuron curation (reference: registered AND > 5 events while moving)
    registered = ~np.isnan(trace[0, :])
    events_moving = np.nansum(trace[moving, :], axis=0)
    keep_cells = registered & (events_moving > CELL_THRESHOLD)

    tr = trace[:, keep_cells]
    if tr.size == 0:
        return [], [], [], keep_cells

    # ---- 3. temporal smoothing + 100 ms binning of the neural data
    #         (reference fit_decoder: gaussian_filter1d(sigma=3) then AvgPool1d(3))
    tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SMOOTH_SIGMA, axis=0)
    neural = pool_mean(tr_smooth).astype(np.float32) * FPS     # -> events / s

    # ---- 4. behaviour binned on exactly the same 100 ms grid
    pos_binned = pool_mean(position)
    speed_binned = pool_mean(speed[:, None])[:, 0]
    n_bins = neural.shape[0]
    assert pos_binned.shape[0] == n_bins == speed_binned.shape[0]

    # ---- 5. static per-trial input: the blocked partitions of this geometry
    input_vec = env_blocked_vector(env_name)

    out_class = position_to_class(pos_binned, blocked_vec=input_vec)
    moving_binned = speed_binned > V_THRESH

    # ---- 6. cut into 1-minute trials and drop non-locomotion bins
    neural_trials, input_trials, output_trials = [], [], []
    trial_slices = []
    start = 0
    while start < n_bins:
        stop = min(start + BINS_PER_TRIAL, n_bins)
        if (stop - start) < MIN_TRIAL_FRAC * BINS_PER_TRIAL:
            break                      # drop a too-short remainder
        sel = np.where(moving_binned[start:stop])[0] + start
        if sel.size >= MIN_BINS_PER_TRIAL:
            neural_trials.append(np.ascontiguousarray(neural[sel, :].T))
            input_trials.append(input_vec.copy())
            output_trials.append(out_class[sel][None, :].copy())
            trial_slices.append((start, stop, sel))
        start = stop

    if diagnostics is not None:
        diagnostics.update(dict(speed=speed, moving=moving, neural=neural,
                                pos_binned=pos_binned, out_class=out_class,
                                moving_binned=moving_binned, trace=trace,
                                position=position, keep_cells=keep_cells,
                                trial_slices=trial_slices, n_frames=n_frames,
                                input_vec=input_vec))
    return neural_trials, input_trials, output_trials, keep_cells


# --------------------------------------------------------------------------- #
# plotting for --show-processing
# --------------------------------------------------------------------------- #
def plot_processing(diag, session_id, env_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(6, 1, figsize=(16, 18))
    t_frames = np.arange(diag['n_frames']) / FPS
    t_bins = (np.arange(diag['neural'].shape[0]) + 0.5) * TEMPORAL_BIN_FRAMES / FPS
    w = slice(0, int(120 * FPS))           # first 2 minutes at 30 Hz
    wb = slice(0, int(120 * FPS / TEMPORAL_BIN_FRAMES))

    # 1. raw binary events of the kept cells
    tr = diag['trace'][:, diag['keep_cells']]
    show = tr[w, :min(60, tr.shape[1])]
    ax[0].imshow(show.T, aspect='auto', cmap='Greys', interpolation='nearest',
                 extent=[t_frames[w][0], t_frames[w][-1], show.shape[1], 0])
    ax[0].set_title(f'{session_id} ({env_name}): 1) raw binarised events, 30 Hz '
                    f'(first 60 kept cells of {tr.shape[1]})')
    ax[0].set_ylabel('cell')

    # 2. smoothed + 100 ms pooled neural data (same cells, same time window)
    nb = diag['neural'][wb, :min(60, tr.shape[1])]
    ax[1].imshow(nb.T, aspect='auto', cmap='viridis', interpolation='nearest',
                 extent=[t_bins[wb][0], t_bins[wb][-1], nb.shape[1], 0])
    ax[1].set_title('2) after gaussian smoothing (sigma=3 frames) and 100 ms '
                    'average pooling [events/s]')
    ax[1].set_ylabel('cell')

    # 3. speed and the locomotion mask
    ax[2].plot(t_frames[w], diag['speed'][w], 'k-', lw=0.6, label='speed 30 Hz')
    ax[2].plot(t_bins[wb], np.where(diag['moving_binned'][wb], 1, 0) * V_THRESH,
               'r-', lw=1.0, alpha=0.6, label='kept bins (speed > 5 cm/s)')
    ax[2].axhline(V_THRESH, color='b', ls='--', lw=0.8, label='5 cm/s threshold')
    ax[2].set_ylabel('cm/s')
    ax[2].legend(loc='upper right', fontsize=8)
    ax[2].set_title('3) speed filtering')

    # 4. position and its 100 ms binned version - checks temporal alignment
    ax[3].plot(t_frames[w], diag['position'][w, 0], 'k-', lw=0.6, label='x 30 Hz')
    ax[3].plot(t_bins[wb], diag['pos_binned'][wb, 0], 'r.', ms=2, label='x 100 ms bins')
    ax[3].plot(t_frames[w], diag['position'][w, 1], 'g-', lw=0.6, label='y 30 Hz')
    ax[3].plot(t_bins[wb], diag['pos_binned'][wb, 1], 'm.', ms=2, label='y 100 ms bins')
    for b in [25, 50]:
        ax[3].axhline(b, color='b', ls=':', lw=0.8)
    ax[3].set_ylabel('cm')
    ax[3].legend(loc='upper right', fontsize=8, ncol=2)
    ax[3].set_title('4) position, 30 Hz vs 100 ms bins (dotted = partition borders '
                    'at 25 and 50 cm)')

    # 5. discretisation check: class label vs position
    ax[4].step(t_bins[wb], diag['out_class'][wb], 'k-', where='mid', lw=0.8,
               label='output class = 3*ybin + xbin')
    xb = np.clip((diag['pos_binned'][wb, 0] / 25).astype(int), 0, 2)
    yb = np.clip((diag['pos_binned'][wb, 1] / 25).astype(int), 0, 2)
    ax[4].step(t_bins[wb], 3 * yb + xb, 'r--', where='mid', lw=1.2, alpha=0.6,
               label='recomputed from binned position (before out-of-geometry cleaning)')
    ax[4].set_ylabel('class 0-8')
    ax[4].legend(loc='upper right', fontsize=8)
    ax[4].set_title('5) discretisation of position into 3x3 partitions')

    # 6. trajectory, partition grid, occupancy per class and the input vector
    ax[5].remove()
    gs = fig.add_gridspec(6, 3)
    a0 = fig.add_subplot(gs[5, 0])
    a0.plot(diag['position'][:, 0], diag['position'][:, 1], 'k-', lw=0.2, alpha=0.5)
    for b in [25, 50]:
        a0.axvline(b, color='r', lw=1)
        a0.axhline(b, color='r', lw=1)
    for p in range(9):
        a0.text((p % 3) * 25 + 12, (p // 3) * 25 + 12, str(p), color='b',
                ha='center', va='center', fontsize=14)
    a0.set_xlim(0, 75); a0.set_ylim(0, 75); a0.set_aspect('equal')
    a0.set_title('6a) trajectory + partition ids')

    a1 = fig.add_subplot(gs[5, 1])
    frac = np.bincount(diag['out_class'][diag['moving_binned']], minlength=9) / \
        max(1, diag['moving_binned'].sum())
    a1.bar(np.arange(9), frac, color='k')
    a1.set_xticks(np.arange(9))
    a1.set_title('6b) output class distribution (kept bins)')

    a2 = fig.add_subplot(gs[5, 2])
    a2.imshow(diag['input_vec'].reshape(3, 3), origin='lower', cmap='Reds',
              vmin=0, vmax=1)
    for p in range(9):
        a2.text(p % 3, p // 3, f'{p}\n{int(diag["input_vec"][p])}', ha='center',
                va='center')
    a2.set_title(f'6c) decoder input: blocked partitions ({env_name})')

    fig.tight_layout()
    out = f'/app/processing_{session_id}.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'    wrote {out}')


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
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
        'subjects': list(ANIMALS), 'subject_idx': [],
        'brain_regions': ['CA1'], 'brain_region_idx': [],
        'input_names': [f'blocked_partition_{i}' for i in range(9)],
        'output_names': ['position_3x3'],
        'output_values': [[f'partition_{i}_(x{i % 3},y{i // 3})' for i in range(9)]],
        'metadata': {},
    }
    session_info = []
    n_plotted = 0
    t_start = time.time()
    timings = []

    # In sample mode process 2 sessions from 2 different animals, chosen so that
    # one has no blocked partitions (square) and one has several (t), giving
    # variation in the decoder input.
    SAMPLE_SESSIONS = {'QLAK-CA1-51': [0], 'QLAK-CA1-56': [9]}
    animals = list(SAMPLE_SESSIONS.keys()) if sample else ANIMALS
    for ai, animal in enumerate(animals):
        subj_idx = ANIMALS.index(animal)
        t_open = time.time()
        f = open_animal(animal)
        envs, blocked = read_meta(f)
        n_days = len(envs)
        print(f'{animal}: {n_days} sessions (file opened in '
              f'{time.time() - t_open:.2f}s)', flush=True)

        days = SAMPLE_SESSIONS[animal] if sample else range(n_days)
        for day in days:
            t0 = time.time()
            trace, position = read_session(f, day)
            t_read = time.time() - t0

            # consistency check: the geometry name and the `blocked` field must agree
            expected = np.where(env_blocked_vector(envs[day]))[0]
            got = np.sort(blocked[day][blocked[day] >= 0])
            assert np.array_equal(expected, got), \
                f'{animal} day {day}: blocked {got} != {expected} for env {envs[day]}'

            diag = {} if (args.show_processing and n_plotted < 2) else None
            t1 = time.time()
            ntr, inp, outp, keep = process_session(trace, position, envs[day],
                                                   diagnostics=diag)
            t_proc = time.time() - t1

            sid = f'{animal}_day{day:02d}'
            if len(ntr) < 2:
                print(f'  SKIP {sid}: only {len(ntr)} usable trials', flush=True)
                continue

            data['neural'].append(ntr)
            data['input'].append(inp)
            data['output'].append(outp)
            data['subject_idx'].append(subj_idx)
            data['brain_region_idx'].append(np.zeros(int(keep.sum()), dtype=np.int64))
            session_info.append({
                'session_id': sid, 'animal': animal, 'day': day,
                'environment': envs[day], 'sequence': day // 10,
                'n_neurons': int(keep.sum()),
                'n_cells_registered': int((~np.isnan(trace[0, :])).sum()),
                'n_cells_file': int(trace.shape[1]),
                'n_trials': len(ntr),
                'n_frames_raw': int(trace.shape[0]),
                'n_bins_kept': int(sum(t.shape[1] for t in ntr)),
            })
            timings.append((t_read, t_proc))
            print(f'  {sid} env={envs[day]:<10s} cells {keep.sum():4d}/'
                  f'{(~np.isnan(trace[0, :])).sum():4d} reg  trials {len(ntr):3d}  '
                  f'bins {sum(t.shape[1] for t in ntr):6d}  '
                  f'(read {t_read:.2f}s proc {t_proc:.2f}s)', flush=True)

            if diag is not None:
                plot_processing(diag, sid, envs[day])
                n_plotted += 1
            del trace, position
        f.close()

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    n_sessions = len(data['neural'])
    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = sum(s['n_neurons'] for s in session_info)
    data['metadata'] = {
        'task_description':
            'Mice freely explore a 75 x 75 cm arena partitioned into a 3 x 3 grid; '
            'across days the arena takes one of 10 geometries created by blocking '
            'partitions. The decoder receives CA1 calcium event rates plus the static '
            'environment geometry (which of the 9 partitions are blocked) and must '
            'predict which of the 9 partitions the mouse occupies at each time bin.',
        'time_bin_size': BIN_MS,
        'temporal_alignment_event':
            'Start of each 60 s trial, measured from the start of the recording '
            'session (free exploration; the task has no discrete events). Behaviour '
            'and calcium were acquired simultaneously at 30 Hz by the same DAQ and '
            'are frame-aligned in the source data.',
        'off_start': 0.0,
        'off_end': TRIAL_SEC,
        'source': 'Lee, Keinath, Cianfarano & Brandon (2025) Neuron 113(2):307-320; '
                  'Zenodo doi:10.5281/zenodo.13993254',
        'recording_modality': 'one-photon miniscope calcium imaging (UCLA Miniscope v3)',
        'neural_signal': 'binarised rising phase of calcium transients (the dataset '
                         '`trace` field), smoothed with a gaussian (sigma = 3 frames '
                         '= 100 ms) and averaged into 100 ms bins, in events/s',
        'sampling_rate_hz': FPS,
        'arena_size_cm': ARENA_SIZE,
        'partition_size_cm': ARENA_SIZE / N_PART,
        'output_encoding': 'class = 3 * ybin + xbin with xbin, ybin in {0,1,2} '
                           '(the partition indexing of the dataset `blocked` field); '
                           'the 0.005% of samples that tracking noise places inside a '
                           'blocked partition are snapped to the nearest open partition, '
                           'as in the cleaning step of utils.decode_position_within',
        'speed_threshold_cm_s': V_THRESH,
        'neuron_inclusion': 'registered on that day and > 5 events during locomotion '
                            '(cell_threshold of utils.decode_position_within)',
        'trial_definition': f'consecutive {TRIAL_SEC:.0f} s blocks of the ~40 min '
                            'session; bins with speed <= 5 cm/s removed',
        'geometries': sorted(ENV_MATS.keys()),
        'n_sessions': n_sessions,
        'n_trials': n_trials,
        'n_neurons_total': n_neurons,
        'session_info': session_info,
    }

    print(f'\nSummary: {n_sessions} sessions, {n_trials} trials, '
          f'{n_neurons} neuron-sessions, '
          f'{len(set(data["subject_idx"].tolist()))} subjects')
    if timings:
        tr_ = np.array(timings)
        print(f'Mean per session: read {tr_[:, 0].mean():.2f}s, '
              f'process {tr_[:, 1].mean():.2f}s')
        print(f'Estimated full-dataset time: '
              f'{tr_.sum(axis=1).mean() * 207 / 60:.1f} min')

    t0 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')
    print(f'Total time {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
