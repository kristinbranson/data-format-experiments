#!/usr/bin/env python3
"""
Convert the CA1 geometric-deformation dataset of Lee, Keinath, Cianfarano & Brandon (2025)
("Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping",
Neuron 113:307-320; data: Zenodo 14867736; code: github.com/jquinnlee/georepca1) into the decoder format
described in the task specification.

Decoder task
------------
    inputs  : environment geometry - which of the 3x3 partitions of the arena are blocked (static per trial)
    outputs : the mouse's position discretised into the 3x3 = 9 partitions (time-varying)
    neural  : CA1 population activity, binarised calcium-transient rising-phase events

Processing follows the reference implementation of the paper's own position-decoding analysis
(`decode_position_within` / `fit_decoder` / `test_decoder` in georepca1/src/utils.py):

    * 100 ms time bins   : traces smoothed with a 3-frame Gaussian then average-pooled over 3 frames at 30 Hz
                           (`fit_decoder(..., temporal_bin_size=3)`), expressed in events/s
    * speed filter       : only bins with speed > 5 cm/s are kept (`v_thresh=5`, speed smoothed with a
                           5-frame Gaussian, `v_filt_size=5`)
    * cell curation      : cells registered on that day with > 5 events during the retained bins
                           (`cell_threshold=5`); NaN (unregistered) cells are therefore dropped
    * spatial binning    : fixed physical grid of the 75 x 75 cm arena -> 25 x 25 cm partitions,
                           partition = 3*floor(y/25) + floor(x/25) (verified against the `blocked` field)

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing] [--jobs N]
"""

import argparse
import os
import pickle
import time

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------------------------------
# Constants (all taken from the reference code / methods)
# ----------------------------------------------------------------------------------------------------
DATA_DIR = '/app/data'
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

FPS = 30.0             # acquisition rate of both the imaging and the behaviour stream (methods)
TBIN = 3               # frames per time bin -> 100 ms  (fit_decoder: temporal_bin_size=3)
TRACE_SIGMA = 3.0      # frames, Gaussian smoothing of the traces (fit_decoder)
V_SIGMA = 5.0          # frames, Gaussian smoothing of the speed trace (decode_position_within: v_filt_size)
V_THRESH = 5.0         # cm/s, speed threshold (decode_position_within: v_thresh)
CELL_THRESH = 5        # events during the retained bins (decode_position_within: cell_threshold)

ARENA_SIZE = 75.0      # cm, side of the square arena (methods)
N_GRID = 3             # 3 x 3 partitions
PART_SIZE = ARENA_SIZE / N_GRID   # 25 cm

TRIAL_SECONDS = 60.0                                   # 1-minute trials (task specification)
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TBIN))    # 600 bins of 100 ms
MIN_TRIAL_BINS = 30                                    # drop trials with < 3 s of movement

OUTPUT_VALUES = [f'r{r}c{c}' for r in range(N_GRID) for c in range(N_GRID)]
INPUT_NAMES = [f'blocked_{v}' for v in OUTPUT_VALUES]

# centre of each partition in cm, indexed by partition id = 3*row + col
PART_CENTERS = np.array([[(p % N_GRID) * PART_SIZE + PART_SIZE / 2,      # x (dim 0, West->East, column)
                          (p // N_GRID) * PART_SIZE + PART_SIZE / 2]     # y (dim 1, North->South, row)
                         for p in range(N_GRID * N_GRID)])


# ----------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------
def blocked_vector(blocked_entry):
    """
    Convert one entry of the dataset's `blocked` field into a 9-d binary vector.

    `blocked` holds the indices of the occluded partitions of the 3x3 grid, laid out as
    [[0,1,2],[3,4,5],[6,7,8]] (code/README.md).  A value of -1 means nothing is blocked (square).

    Returns (vector float32 of shape (9,), sorted tuple of blocked partition ids).
    """
    e = blocked_entry
    if isinstance(e, (list, tuple)):
        e = e[0]
    idx = np.atleast_1d(np.asarray(e, dtype=float)).ravel()
    idx = idx[idx >= 0].astype(int)
    vec = np.zeros(N_GRID * N_GRID, dtype=np.float32)
    vec[idx] = 1.0
    return vec, tuple(sorted(idx.tolist()))


def bin_time(x, nbins, how='mean'):
    """Bin the last axis of `x` into `nbins` non-overlapping bins of TBIN frames."""
    x = x[..., :nbins * TBIN]
    x = x.reshape(x.shape[:-1] + (nbins, TBIN))
    return x.mean(axis=-1) if how == 'mean' else x.sum(axis=-1)


def compute_speed(position):
    """
    Speed in cm/s for every frame, exactly as in decode_position_within:
        vel[1:] = gaussian_filter1d(norm(diff(position) * fps), sigma=v_filt_size)
    Frame 0 has no speed estimate and is left at 0 (the reference leaves vel_idx[0] False).
    """
    speed = np.zeros(position.shape[1], dtype=np.float64)
    speed[1:] = gaussian_filter1d(np.linalg.norm(np.diff(position, axis=1) * FPS, axis=0),
                                  sigma=V_SIGMA)
    return speed


def discretize_position(pos_binned, blocked_ids):
    """
    Map binned x-y position (cm) onto the 3x3 partition index.

    partition = 3 * floor(y / 25) + floor(x / 25), with x = position dim 0 (West->East) and
    y = position dim 1 (North->South).  This convention was verified against the dataset's `blocked`
    field: the occupancy of every blocked partition is zero (see CONVERSION_NOTES.md, Step 4).

    The handful of samples that still land in a blocked partition (tracking noise at a partition wall;
    0.0003% of frames) are reassigned to the nearest open partition, mirroring the reference decoder's
    "cleaning" of positions onto valid bins.
    """
    xb = np.clip(np.floor(pos_binned[0] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
    yb = np.clip(np.floor(pos_binned[1] / PART_SIZE), 0, N_GRID - 1).astype(np.int64)
    part = N_GRID * yb + xb

    n_fixed = 0
    if len(blocked_ids):
        bad = np.isin(part, blocked_ids)
        n_fixed = int(bad.sum())
        if n_fixed:
            open_ids = np.array([p for p in range(N_GRID * N_GRID) if p not in blocked_ids])
            d = np.linalg.norm(pos_binned[:, bad].T[:, None, :] - PART_CENTERS[open_ids][None, :, :],
                               axis=2)
            part[bad] = open_ids[np.argmin(d, axis=1)]
    return part, n_fixed


# ----------------------------------------------------------------------------------------------------
# Per-session processing
# ----------------------------------------------------------------------------------------------------
def process_session(trace, position, blocked_entry, env, session_id, show_processing=False,
                    plot_dir='/app'):
    """
    Convert one recording session (one day of one animal) into a list of trials.

    Inputs
        trace         : (n_cells, n_frames) binarised event trains, NaN for cells not registered that day
        position      : (2, n_frames) head position in cm
        blocked_entry : the session's entry of the dataset's `blocked` field
        env           : geometry name (string)

    Returns a dict with the session's trials and bookkeeping information.
    """
    n_cells, n_frames = trace.shape
    nb = n_frames // TBIN                      # number of complete 100 ms bins
    assert position.shape[1] == n_frames       # imaging and behaviour streams are frame-aligned

    blocked_vec, blocked_ids = blocked_vector(blocked_entry)

    # ---- behaviour: speed, binning, spatial discretisation ------------------------------------------
    speed = compute_speed(position)
    speed_b = bin_time(speed, nb)                                  # (nb,)
    moving = speed_b > V_THRESH                                    # velocity filter (reference: v_thresh=5)
    pos_b = bin_time(position, nb)                                 # (2, nb), cm
    part, n_fixed = discretize_position(pos_b, blocked_ids)        # (nb,) partition index 0..8

    # ---- neural: registered cells, smoothing, temporal binning ---------------------------------------
    registered = ~np.isnan(trace[:, 0])                            # NaN is all-or-none per (cell, day)
    raw = trace[registered][:, :nb * TBIN].astype(np.float32)      # (n_reg, nb*TBIN) binary
    events_b = bin_time(raw, nb, how='sum')                        # events per 100 ms bin (for curation)

    # smooth the continuous session trace, then average-pool 3 frames and convert to events/s
    smoothed = gaussian_filter1d(raw, sigma=TRACE_SIGMA, axis=1)
    neural = (bin_time(smoothed, nb) * FPS).astype(np.float32)     # (n_reg, nb) in events/s

    # cell curation: > 5 events during the retained (moving) bins  (reference: cell_threshold=5)
    active = events_b[:, moving].sum(axis=1) > CELL_THRESH
    cell_idx = np.where(registered)[0][active]                     # indices into the animal's cell list
    neural = neural[active]

    # ---- cut into 1-minute trials, keeping only moving bins ------------------------------------------
    trials_neural, trials_input, trials_output, trial_bin_idx = [], [], [], []
    n_dropped = 0
    for start in range(0, nb, TRIAL_BINS):
        idx = np.arange(start, min(start + TRIAL_BINS, nb))
        sel = idx[moving[idx]]
        if len(sel) < MIN_TRIAL_BINS:
            n_dropped += 1
            continue
        trials_neural.append(np.ascontiguousarray(neural[:, sel]))
        trials_input.append(blocked_vec.copy())
        trials_output.append(part[sel][np.newaxis, :].astype(np.int64))
        trial_bin_idx.append(sel)

    result = {
        'session_id': session_id,
        'env': env,
        'blocked_ids': blocked_ids,
        'neural': trials_neural,
        'input': trials_input,
        'output': trials_output,
        'n_neurons': int(neural.shape[0]),
        'n_registered': int(registered.sum()),
        'cell_idx': cell_idx,
        'n_bins': int(nb),
        'n_moving': int(moving.sum()),
        'n_dropped_trials': n_dropped,
        'n_blocked_fixed': n_fixed,
    }

    if show_processing:
        _plot_processing(result, trace, position, speed, speed_b, moving, pos_b, part,
                         registered, active, neural, trial_bin_idx, plot_dir)
    return result


# ----------------------------------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------------------------------
def _plot_processing(res, trace, position, speed, speed_b, moving, pos_b, part,
                     registered, active, neural, trial_bin_idx, plot_dir):
    """Visualise every processing step of one session (saved to processing_<session_id>.png)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nb = res['n_bins']
    t_frames = np.arange(trace.shape[1]) / FPS
    t_bins = (np.arange(nb) * TBIN + (TBIN - 1) / 2.) / FPS      # bin centres in seconds
    w0, w1 = 0.0, 60.0                                           # 1-minute zoom window (first trial)
    fw = (t_frames >= w0) & (t_frames < w1)
    bw = (t_bins >= w0) & (t_bins < w1)

    fig, ax = plt.subplots(5, 2, figsize=(22, 22))

    # (1) raw vs binned position ----------------------------------------------------------------------
    for k, c in enumerate(['tab:blue', 'tab:orange']):
        ax[0, 0].plot(t_frames[fw], position[k, fw], color=c, alpha=.4,
                      label=f'raw dim{k} (30 Hz)')
        ax[0, 0].plot(t_bins[bw], pos_b[k, bw], '.', color=c, ms=3,
                      label=f'binned dim{k} (100 ms)')
    for g in (PART_SIZE, 2 * PART_SIZE):
        ax[0, 0].axhline(g, color='k', ls=':', lw=1)
    ax[0, 0].set(xlabel='time (s)', ylabel='position (cm)',
                 title='1. position: raw 30 Hz vs 100 ms bins (dotted = 25 cm partition edges)')
    ax[0, 0].legend(fontsize=7, ncol=2)

    # (2) discretisation: position -> partition -------------------------------------------------------
    ax[0, 1].plot(t_bins[bw], part[bw], drawstyle='steps-mid', color='k', label='partition id')
    ax[0, 1].plot(t_bins[bw], np.floor(pos_b[0, bw] / PART_SIZE), '--', color='tab:blue',
                  label='col = floor(x/25)')
    ax[0, 1].plot(t_bins[bw], np.floor(pos_b[1, bw] / PART_SIZE), '--', color='tab:orange',
                  label='row = floor(y/25)')
    ax[0, 1].set(xlabel='time (s)', ylabel='partition / row / col',
                 title='2. output = 3*row + col')
    ax[0, 1].legend(fontsize=7)

    # (3) speed and the velocity filter ---------------------------------------------------------------
    ax[1, 0].plot(t_frames[fw], speed[fw], color='gray', alpha=.6, label='speed (30 Hz, smoothed)')
    ax[1, 0].plot(t_bins[bw], speed_b[bw], '.', color='tab:red', ms=3, label='speed (100 ms bins)')
    ax[1, 0].axhline(V_THRESH, color='k', ls='--', label=f'{V_THRESH:g} cm/s threshold')
    ax[1, 0].fill_between(t_bins[bw], 0, speed_b[bw].max() * 1.05, where=moving[bw],
                          color='tab:green', alpha=.15, step='mid', label='retained bins')
    ax[1, 0].set(xlabel='time (s)', ylabel='speed (cm/s)',
                 title=f'3. velocity filter: {100 * moving.mean():.1f}% of bins retained')
    ax[1, 0].legend(fontsize=7)

    # (4) trajectory coloured by assigned partition ---------------------------------------------------
    sc = ax[1, 1].scatter(pos_b[0], pos_b[1], c=part, cmap='tab10', s=.5, vmin=-.5, vmax=9.5)
    for g in (PART_SIZE, 2 * PART_SIZE):
        ax[1, 1].axvline(g, color='k', lw=1)
        ax[1, 1].axhline(g, color='k', lw=1)
    for p in res['blocked_ids']:
        ax[1, 1].add_patch(plt.Rectangle(((p % N_GRID) * PART_SIZE, (p // N_GRID) * PART_SIZE),
                                         PART_SIZE, PART_SIZE, color='k', alpha=.25))
    for p in range(9):
        ax[1, 1].text(PART_CENTERS[p, 0], PART_CENTERS[p, 1], str(p), ha='center', va='center')
    ax[1, 1].invert_yaxis()
    ax[1, 1].set(xlabel='position dim 0 (cm, West->East)', ylabel='position dim 1 (cm, North->South)',
                 title=f"4. env='{res['env']}', blocked={list(res['blocked_ids'])} (grey) - "
                       f'trajectory coloured by output label')
    plt.colorbar(sc, ax=ax[1, 1], label='partition')

    # (5) raw binary events vs smoothed/binned rate for 3 example cells --------------------------------
    raw_reg = trace[registered][active]     # same row order as `neural`
    for i in range(min(3, raw_reg.shape[0])):
        ax[2, 0].plot(t_frames[fw], raw_reg[i, fw] * 0.8 + i, color='k', lw=.7)
        ax[2, 0].plot(t_bins[bw], neural[i, bw] / max(neural[i].max(), 1e-9) * 0.8 + i,
                      color='tab:red', lw=1)
    ax[2, 0].set(xlabel='time (s)', ylabel='example cells',
                 title='5. neural: raw binary events (black) vs smoothed+binned rate (red, normalised)')

    # (6) population raster before / after binning ----------------------------------------------------
    ax[2, 1].imshow(raw_reg[:60][:, fw], aspect='auto', cmap='Greys',
                    extent=[w0, w1, 60, 0], interpolation='nearest')
    ax[2, 1].set(xlabel='time (s)', ylabel='cell', title='6. raw binary events, 60 cells (30 Hz)')

    ax[3, 0].imshow(neural[:60][:, bw], aspect='auto', cmap='magma',
                    extent=[w0, w1, 60, 0], interpolation='nearest')
    ax[3, 0].set(xlabel='time (s)', ylabel='cell',
                 title='7. converted neural (events/s, 100 ms bins), same 60 cells')

    # (7) cell curation --------------------------------------------------------------------------------
    ev_total = np.nansum(trace, axis=1)
    ax[3, 1].hist([ev_total[registered][active], ev_total[registered][~active]], bins=40, stacked=True,
                  label=['kept', f'dropped (<= {CELL_THRESH} events while moving)'], color=['tab:green', 'tab:red'])
    ax[3, 1].set(xlabel='total events in session', ylabel='# cells', yscale='log',
                 title=f"8. cell curation: {res['n_neurons']} kept of {res['n_registered']} registered "
                       f'({trace.shape[0]} in animal)')
    ax[3, 1].legend(fontsize=7)

    # (8) trial structure ------------------------------------------------------------------------------
    for i, sel in enumerate(trial_bin_idx):
        ax[4, 0].plot(sel * TBIN / FPS / 60., np.full(len(sel), i), '.', ms=.5)
    ax[4, 0].set(xlabel='time in session (min)', ylabel='trial index',
                 title=f'9. trials: {len(trial_bin_idx)} kept of {int(np.ceil(nb / TRIAL_BINS))} '
                       f"1-min blocks ({res['n_dropped_trials']} dropped)")

    # (9) output distribution and geometry input -------------------------------------------------------
    kept = np.concatenate(trial_bin_idx) if trial_bin_idx else np.array([], dtype=int)
    frac = np.bincount(part[kept], minlength=9) / max(len(kept), 1)
    ax[4, 1].bar(np.arange(9), frac, color=['k' if p in res['blocked_ids'] else 'tab:blue' for p in range(9)])
    for p in res['blocked_ids']:
        ax[4, 1].text(p, 0.01, 'blocked', rotation=90, fontsize=7)
    ax[4, 1].set(xlabel='partition (output value)', ylabel='fraction of retained bins',
                 xticks=np.arange(9), xticklabels=OUTPUT_VALUES,
                 title='10. output distribution; black = blocked partitions (input = 1)')

    fig.suptitle(f"Processing steps - session {res['session_id']} (env '{res['env']}')", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .98])
    out = os.path.join(plot_dir, f"processing_{res['session_id']}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  wrote {out}')


# ----------------------------------------------------------------------------------------------------
# Per-animal processing
# ----------------------------------------------------------------------------------------------------
def process_animal(animal, days=None, show_processing=False, plot_dir='/app', verbose=True):
    """Load one animal's joblib file and convert the requested sessions (all of them if days is None)."""
    t0 = time.time()
    dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
    t_load = time.time() - t0
    trace, position, envs, blocked = dat['trace'], dat['position'], dat['envs'].ravel(), dat['blocked']
    if days is None:
        days = list(range(trace.shape[0]))
    if verbose:
        print(f'[{animal}] loaded in {t_load:.1f}s: {trace.shape[0]} sessions, {trace.shape[1]} cells, '
              f'{trace.shape[2]} frames ({trace.shape[2] / FPS / 60:.1f} min)', flush=True)

    sessions = []
    t1 = time.time()
    for day in days:
        res = process_session(trace[day], position[day], blocked[day], str(envs[day]),
                              session_id=f'{animal}_day{day:02d}',
                              show_processing=show_processing,
                              plot_dir=plot_dir)
        res['animal'] = animal
        res['day'] = day
        sessions.append(res)
        if verbose:
            print(f"  [{animal}] day {day:02d} env={res['env']:<10s} neurons={res['n_neurons']:4d}/"
                  f"{res['n_registered']:4d}  bins={res['n_bins']}  moving={res['n_moving']} "
                  f"({100 * res['n_moving'] / res['n_bins']:.0f}%)  trials={len(res['neural'])}"
                  f"{'  fixed=%d' % res['n_blocked_fixed'] if res['n_blocked_fixed'] else ''}", flush=True)
    t_proc = time.time() - t1
    if verbose:
        print(f'[{animal}] processed {len(days)} sessions in {t_proc:.1f}s '
              f'({t_proc / max(len(days), 1):.2f}s/session), load {t_load:.1f}s', flush=True)
    del dat, trace, position
    return sessions


# ----------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', help='output pickle file')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions (for testing)')
    ap.add_argument('--show-processing', action='store_true',
                    help='save diagnostic plots of every processing step for up to 2 sessions')
    ap.add_argument('--jobs', type=int, default=7, help='number of animals to process in parallel')
    args = ap.parse_args()

    t_start = time.time()
    plot_dir = os.path.dirname(os.path.abspath(args.outfile)) or '.'

    if args.sample:
        # 2 sessions, one from each of two animals (exercises the multi-animal bookkeeping):
        # QLAK-CA1-08 day 2 is the 't' geometry (4 blocked partitions), QLAK-CA1-51 day 0 is the square
        jobs = [(ANIMALS[0], [2]), (ANIMALS[3], [0])]
        print('SAMPLE mode: 2 sessions (1 from each of 2 animals)')
    else:
        jobs = [(a, None) for a in ANIMALS]
        print(f'FULL mode: all {len(ANIMALS)} animals')

    n_jobs = max(1, min(args.jobs, len(jobs)))
    if n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_animal)(a, n, args.show_processing, plot_dir) for a, n in jobs)
    else:
        results = [process_animal(a, n, args.show_processing, plot_dir) for a, n in jobs]

    # ---- assemble the target structure ----------------------------------------------------------------
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': list(ANIMALS), 'subject_idx': [],
        'brain_regions': ['CA1'], 'brain_region_idx': [],
        'input_names': INPUT_NAMES,
        'output_names': ['position_bin'],
        'output_values': [OUTPUT_VALUES],
        'metadata': {},
    }
    session_info = []
    for sessions in results:
        for s in sessions:
            if len(s['neural']) < 2:
                print(f"  !! skipping session {s['session_id']}: only {len(s['neural'])} usable trials")
                continue
            data['neural'].append(s['neural'])
            data['input'].append(s['input'])
            data['output'].append(s['output'])
            data['subject_idx'].append(ANIMALS.index(s['animal']))
            data['brain_region_idx'].append(np.zeros(s['n_neurons'], dtype=np.int64))
            session_info.append({
                'session_id': s['session_id'], 'animal': s['animal'], 'day': int(s['day']),
                'env': s['env'], 'blocked': list(s['blocked_ids']),
                'n_neurons': s['n_neurons'], 'n_registered': s['n_registered'],
                'n_bins_total': s['n_bins'], 'n_bins_kept': s['n_moving'],
                'n_trials': len(s['neural']),
                'cell_idx': s['cell_idx'].tolist(),
            })
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    data['metadata'] = {
        'task_description':
            'Mice freely explore a 75 x 75 cm arena that is partitioned into a 3 x 3 grid of 25 x 25 cm '
            'partitions; on each recording day a different subset of partitions is blocked off with walls, '
            'creating 10 distinct geometries presented in a randomised sequence repeated up to 3 times '
            '(one 40-min session per day). There is no explicit trial structure and no reward; the decoder '
            'predicts which of the 9 partitions the mouse occupies at each time point from CA1 population '
            'activity, given the geometry of the environment as input.',
        'time_bin_size': 1000.0 * TBIN / FPS,          # 100.0 ms
        'temporal_alignment_event':
            'Start of each 1-minute trial. Sessions are continuous 40-min recordings cut into consecutive '
            '1-min blocks; within a block only the time bins in which the mouse ran faster than 5 cm/s are '
            'kept, so trials contain <= 600 time bins.',
        'off_start': 0.0,
        'off_end': TRIAL_SECONDS,
        'dataset': 'Lee, Keinath, Cianfarano & Brandon (2025) Neuron 113:307-320; Zenodo 10.5281/zenodo.13993254',
        'recording_modality': '1-photon miniscope calcium imaging (GCaMP6f) of dorsal CA1',
        'neural_units': 'events/s (binarised calcium-transient rising phases averaged in each 100 ms bin, x30)',
        'neural_processing':
            'Binarised rising-phase event trains (provided with the dataset) smoothed with a 3-frame '
            'Gaussian and average-pooled over 3 frames (100 ms) at 30 Hz, as in the reference decoder '
            '(fit_decoder, temporal_bin_size=3), expressed in events/s.',
        'neuron_curation':
            'Cells registered on that session (non-NaN trace) with more than 5 events during the retained '
            'moving bins (reference decode_position_within: cell_threshold=5).',
        'trial_curation':
            'Time bins with speed <= 5 cm/s are discarded (reference v_thresh=5, speed smoothed with a '
            '5-frame Gaussian); 1-min blocks retaining fewer than 30 bins (3 s) are dropped.',
        'input_description':
            'Environment geometry: 9-d binary vector, 1 if that partition of the 3 x 3 grid is blocked off '
            'on this session (from the dataset field "blocked"), 0 if it is open. Static within a trial.',
        'output_description':
            'Index of the 3 x 3 partition occupied by the mouse, 3*row + column with column = '
            'floor(x/25 cm) (West to East) and row = floor(y/25 cm) (North to South), at each 100 ms bin.',
        'environment_size_cm': ARENA_SIZE,
        'spatial_bin_size_cm': PART_SIZE,
        'fps': FPS,
        'geometries': sorted({tuple(si['blocked']): si['env'] for si in session_info}.values()),
        'session_info': session_info,
    }

    # ---- summary --------------------------------------------------------------------------------------
    ntr = [len(t) for t in data['neural']]
    nneur = [data['neural'][s][0].shape[0] for s in range(len(data['neural']))]
    Ts = np.concatenate([[t.shape[1] for t in sess] for sess in data['neural']])
    allout = np.concatenate([t.ravel() for sess in data['output'] for t in sess])
    nbytes = sum(t.nbytes for sess in data['neural'] for t in sess)
    print(f'\nSessions: {len(data["neural"])}  trials: {sum(ntr)}  '
          f'trials/session: {np.mean(ntr):.1f} [{min(ntr)}, {max(ntr)}]')
    print(f'Neurons/session: mean {np.mean(nneur):.1f} [{min(nneur)}, {max(nneur)}]  total {sum(nneur)}')
    print(f'T/trial: mean {Ts.mean():.1f} [{Ts.min()}, {Ts.max()}]  total timepoints {Ts.sum()}')
    print(f'Output distribution: {np.round(np.bincount(allout, minlength=9) / len(allout), 4).tolist()}')
    print(f'Neural array size: {nbytes / 1e9:.2f} GB')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB) '
          f'in {time.time() - t_start:.1f}s total')


if __name__ == '__main__':
    main()
