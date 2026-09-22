#!/usr/bin/env python3
"""
Convert the Lee, Keinath, Cianfarano & Brandon (2025) CA1 geometric-deformation dataset
(Neuron 113:307-320; Zenodo 10.5281/zenodo.13993254) into the decoder-compatible pickle format.

Decoder task
------------
    inputs  : environment geometry -- which of the 3x3 arena partitions are blocked (static per trial)
    outputs : the mouse's position, discretised into the 3x3 = 9 spatial bins (time-varying)

Processing follows the reference implementation of the paper's own within-session position decoder,
`decode_position_within` / `fit_decoder` in georepca1/src/utils.py:

    * binarised transient rising-phase vector used directly as the firing rate (no dF/F to compute --
      it is already binarised in the distributed dataset)
    * cells kept if registered that day AND with > 5 events while the animal is running
    * frames kept if running speed > 5 cm/s, speed = gaussian_filter1d(|d position|*30, sigma=5 frames)
    * traces gaussian-smoothed with sigma = 3 frames then average-pooled 3 frames -> 100 ms bins
    * position average-pooled over the same 3 frames and floored into spatial bins of 75 cm / n_bins

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------------------------------
# Constants -- all taken from the reference code / methods
# ----------------------------------------------------------------------------------------------------
DATA_DIR = "/app/data"

ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

FPS = 30                 # acquisition rate of both imaging and behaviour streams (methods.txt)
BIN_FRAMES = 3           # fit_decoder(temporal_bin_size=3)  ->  100 ms bins
SMOOTH_SIGMA = 3         # fit_decoder: gaussian_filter1d(traces, sigma=temporal_bin_size)
ARENA_CM = 75.0          # "open square (75 x 75 cm)"
NBINS = 3                # Decoder Task: 3 x 3 = 9 spatial bins
V_FILT_SIZE = 5          # decode_position_within(v_filt_size=5)
V_THRESH = 5.0           # decode_position_within(v_thresh=5) -- cm/s
CELL_THRESH = 5          # decode_position_within(cell_threshold=5) -- events while running

TRIAL_SEC = 60.0         # Decoder Task: "split into 1-minute trials within each session"
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / BIN_FRAMES))   # 600 bins of 100 ms
MIN_TAIL_BINS = 100      # keep the trailing partial block only if it covers >= 10 s of recording
MIN_TRIAL_BINS = 30      # drop a trial left with < 3 s of running

BIN_SIZE_CM = ARENA_CM / NBINS      # 25 cm
TIME_BIN_MS = 1000.0 * BIN_FRAMES / FPS  # 100 ms

# Grid indexing convention of the dataset README: "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
# which corresponds to bin = 3 * ybin + xbin (verified against `blocked`, see CONVERSION_NOTES.md).
BIN_NAMES = [f"x{i % NBINS}y{i // NBINS}" for i in range(NBINS * NBINS)]

GEOMETRIES = ["square", "o", "t", "u", "rectangle", "+", "i", "l", "bit donut", "glenn"]


# ----------------------------------------------------------------------------------------------------
# Reference helper (copied from georepca1/src/utils.py:215) -- used only to cross-check `blocked`
# ----------------------------------------------------------------------------------------------------
def get_env_mat(env):
    """Binary 3x3 matrix for a geometry name, 0 = omitted (blocked) partition. From utils.get_env_mat."""
    mats = {
        'square':     [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        'o':          [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        't':          [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        'u':          [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
        'rectangle':  [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
        '+':          [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        'i':          [[1, 1, 1], [0, 1, 0], [1, 1, 1]],
        'l':          [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
        'bit donut':  [[1, 1, 1], [1, 0, 1], [0, 1, 1]],
        'glenn':      [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
    }
    if env not in mats:
        return np.full((3, 3), np.nan)
    return np.array(mats[env]).astype(float)


def blocked_from_env(env):
    """Blocked-partition indices implied by the geometry name, in the dataset's `blocked` convention.

    `blocked` indexes the grid as 3*row + col of np.flipud(get_env_mat(env)); this is the algebraic
    equivalent of the mask `np.fliplr(get_env_mat(env).T)` that utils.clean_rate_maps applies to rate
    maps indexed [xbin, ybin].
    """
    return np.flatnonzero(np.flipud(get_env_mat(env)).ravel() == 0)


# ----------------------------------------------------------------------------------------------------
# Raw data access
# ----------------------------------------------------------------------------------------------------
def _h5_str(f, ref):
    """Decode a MATLAB char array stored behind an object reference."""
    return ''.join(chr(c) for c in f[ref][:].ravel())


def read_session(f, day):
    """Read one session (day) from an open HDF5 (MATLAB v7.3) animal file.

    HDF5 reverses every dimension relative to the joblib arrays used by the reference code, so
    f[trace_ref] is (T, n_cells) and f[position_ref] is (T, 2) -- already the orientation
    `decode_position_within` works in (`dat[animal]['trace'].T`, `dat[animal]['position'].T`).

    Returns
    -------
    position : (T, 2) float64, head position in cm, columns [x, y]
    trace    : (T, n_cells) float32, binarised transient rising phases; all-NaN column = unregistered cell
    env      : str, geometry name
    blocked  : (n_blocked,) int, indices of blocked partitions (empty for the open square)
    """
    position = f[f['position'][day, 0]][()].astype(np.float64)
    trace = f[f['trace'][day, 0]][()].astype(np.float32)
    env = _h5_str(f, f['envs'][0, day])
    blocked = np.atleast_1d(f[f['blocked'][0, day]][()].ravel())
    blocked = blocked[blocked >= 0].astype(int)     # -1 means "nothing blocked" (the square)
    return position, trace, env, blocked


# ----------------------------------------------------------------------------------------------------
# Processing of one session
# ----------------------------------------------------------------------------------------------------
def running_speed(position):
    """Running speed in cm/s at 30 Hz, exactly as decode_position_within computes it.

    The reference computes `gaussian_filter1d(norm(diff(behav)*fps), sigma=v_filt_size)` in *bin units*
    and compares with `v_thresh / bin_down`; that is identical to computing the speed in cm/s and
    comparing with `v_thresh`. Frame 0 has no velocity estimate and is always excluded
    (`vel_idx[d, 1:] = ...` leaves `vel_idx[d, 0] = False`).
    """
    speed = np.zeros(position.shape[0])
    speed[1:] = gaussian_filter1d(np.linalg.norm(np.diff(position, axis=0) * FPS, axis=1),
                                  sigma=V_FILT_SIZE)
    return speed


def position_to_bin(position_cm):
    """Discretise (N, 2) positions in cm into the 3x3 grid index 3*ybin + xbin.

    The bin size is fixed at 75 cm / 3 = 25 cm from the arena size rather than taken from the
    per-session maximum: in several geometries whole columns of the arena are walled off, so the
    per-session range is not the arena range. Reproducing the dataset's own stored occupancy maps
    requires exactly this fixed 75 cm scaling (see CONVERSION_NOTES.md Step 4).
    """
    b = np.clip(np.floor(position_cm / BIN_SIZE_CM).astype(int), 0, NBINS - 1)
    return NBINS * b[:, 1] + b[:, 0]


def snap_to_open_bins(pos_bin, blocked):
    """Move any sample that landed in a blocked partition to the nearest open partition.

    DeepLabCut occasionally tracks the head a centimetre or so past a partition wall (and the 3-frame
    average of a boundary crossing can land just inside one), so a handful of samples per dataset fall
    in a partition the animal cannot occupy. The reference decoder does exactly this cleaning:
    `decode_position_within` snaps both the true and the predicted bin to the nearest entry of
    `temp_maps`, the set of spatial bins with valid rate-map values, before scoring.

    Distance is Euclidean in (xbin, ybin); ties go to the lower bin index. Affects ~0.005% of samples.
    """
    if blocked.size == 0:
        return pos_bin
    open_bins = np.setdiff1d(np.arange(NBINS * NBINS), blocked)
    coords = np.stack([np.arange(NBINS * NBINS) % NBINS, np.arange(NBINS * NBINS) // NBINS], axis=1)
    lut = np.arange(NBINS * NBINS)
    for b in blocked:
        d = np.linalg.norm(coords[open_bins] - coords[b], axis=1)
        lut[b] = open_bins[np.argmin(d)]
    return lut[pos_bin]


def process_session(position, trace, env, blocked, session_id, show_processing=False):
    """Convert one 40-min session into a list of 1-minute trials.

    Returns a dict with 'neural', 'input', 'output' (lists over trials) plus bookkeeping fields,
    or None if the session yields fewer than two usable trials.
    """
    n_frames_raw, n_cells_total = trace.shape
    assert position.shape == (n_frames_raw, 2), "position and trace must have the same number of frames"

    # --- neuron curation 1: cells registered on this day (trace is all-NaN otherwise) -------------
    registered = ~np.all(np.isnan(trace), axis=0)
    trace = trace[:, registered]
    assert not np.any(np.isnan(trace)), "a registered cell has partial NaNs -- unexpected"

    # --- running speed and immobility mask (30 Hz) ------------------------------------------------
    speed = running_speed(position)
    moving = speed > V_THRESH

    # --- neuron curation 2: > 5 binarised events while running ------------------------------------
    events_running = trace[moving].sum(axis=0)
    keep_cell = events_running > CELL_THRESH
    trace = trace[:, keep_cell]
    n_neurons = trace.shape[1]

    # --- temporal smoothing + binning to 100 ms (fit_decoder) -------------------------------------
    # Smoothing is applied to the intact, contiguous 30 Hz session (the reference smooths after
    # discarding immobility frames, which convolves across temporal discontinuities).
    n_bins = n_frames_raw // BIN_FRAMES
    n_use = n_bins * BIN_FRAMES
    smoothed = gaussian_filter1d(trace, sigma=SMOOTH_SIGMA, axis=0)
    neural = smoothed[:n_use].reshape(n_bins, BIN_FRAMES, n_neurons).mean(axis=1).T.astype(np.float32)

    # position: average-pooled over the same frames, then discretised
    position_binned_cm = position[:n_use].reshape(n_bins, BIN_FRAMES, 2).mean(axis=1)
    pos_bin = position_to_bin(position_binned_cm).astype(np.int64)
    n_snapped = int(np.isin(pos_bin, blocked).sum())
    pos_bin = snap_to_open_bins(pos_bin, blocked)

    # immobility mask downsampled by majority vote over the 3 frames of each bin
    moving_bins = moving[:n_use].reshape(n_bins, BIN_FRAMES).mean(axis=1) >= 0.5

    # --- decoder input: 9-dim binary geometry vector, 1 = partition blocked ------------------------
    geometry = np.zeros(NBINS * NBINS, dtype=np.float32)
    geometry[blocked] = 1.0

    # --- split into contiguous 1-minute trials -----------------------------------------------------
    starts = list(range(0, n_bins - BINS_PER_TRIAL + 1, BINS_PER_TRIAL))
    bounds = [(s, s + BINS_PER_TRIAL) for s in starts]
    tail_start = len(starts) * BINS_PER_TRIAL
    if n_bins - tail_start >= MIN_TAIL_BINS:
        bounds.append((tail_start, n_bins))

    neural_trials, input_trials, output_trials, trial_bounds = [], [], [], []
    for (s, e) in bounds:
        idx = np.flatnonzero(moving_bins[s:e]) + s
        if idx.size < MIN_TRIAL_BINS:
            continue
        neural_trials.append(np.ascontiguousarray(neural[:, idx]))
        input_trials.append(geometry.copy())
        output_trials.append(pos_bin[idx][np.newaxis, :].copy())
        trial_bounds.append((s, e, idx.size))

    if len(neural_trials) < 2:
        return None

    result = {
        'session_id': session_id,
        'env': env,
        'blocked': blocked,
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_neurons,
        'n_registered': int(registered.sum()),
        'n_cells_total': n_cells_total,
        'n_frames': n_frames_raw,
        'n_bins': n_bins,
        'frac_moving': float(moving_bins.mean()),
        'n_snapped': n_snapped,
        'trial_bounds': trial_bounds,
    }
    if show_processing:
        result['_plot'] = dict(position=position, speed=speed, moving=moving,
                               neural=neural, pos_bin=pos_bin, moving_bins=moving_bins,
                               position_binned_cm=position_binned_cm)
    return result


# ----------------------------------------------------------------------------------------------------
# Visualisation of every processing step (--show-processing)
# ----------------------------------------------------------------------------------------------------
def plot_processing(session, raw_position, raw_trace, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pl = session['_plot']
    env, blocked = session['env'], session['blocked']

    # a 120 s window at 30 Hz / 10 Hz for the time-series panels
    t0_frame, n_win_frames = 0, 120 * FPS
    t0_bin, n_win_bins = 0, 120 * FPS // BIN_FRAMES
    tf = np.arange(t0_frame, t0_frame + n_win_frames) / FPS
    tb = (np.arange(t0_bin, t0_bin + n_win_bins) * BIN_FRAMES) / FPS

    fig, ax = plt.subplots(7, 1, figsize=(16, 26))

    # 1. raw position
    ax[0].plot(tf, raw_position[t0_frame:t0_frame + n_win_frames, 0], lw=.8, label='x (cm)')
    ax[0].plot(tf, raw_position[t0_frame:t0_frame + n_win_frames, 1], lw=.8, label='y (cm)')
    for g in np.arange(0, ARENA_CM + 1, BIN_SIZE_CM):
        ax[0].axhline(g, color='k', ls=':', lw=.5)
    ax[0].set_ylabel('raw position')
    ax[0].set_title(f"{session['session_id']}  env='{env}'  blocked={[int(b) for b in blocked]}  "
                    f"(step 1: raw 30 Hz position, dotted lines = 25 cm bin edges)")
    ax[0].legend(loc='upper right')

    # 2. speed + immobility mask
    ax[1].plot(tf, pl['speed'][t0_frame:t0_frame + n_win_frames], lw=.8, color='k')
    ax[1].axhline(V_THRESH, color='r', ls='--', label=f'{V_THRESH:g} cm/s')
    ax[1].fill_between(tf, 0, ax[1].get_ylim()[1],
                       where=pl['moving'][t0_frame:t0_frame + n_win_frames],
                       color='g', alpha=.15, step='mid', label='running (kept)')
    ax[1].set_ylabel('speed (cm/s)')
    ax[1].set_title('step 2: running speed, gaussian sigma=5 frames; shaded = frames retained')
    ax[1].legend(loc='upper right')

    # 3. raw binarised events for a sample of cells
    nshow = min(40, raw_trace.shape[1])
    cells = np.linspace(0, raw_trace.shape[1] - 1, nshow).astype(int)
    ev = raw_trace[t0_frame:t0_frame + n_win_frames][:, cells]
    ax[2].imshow(ev.T, aspect='auto', cmap='Greys', interpolation='nearest',
                 extent=[tf[0], tf[-1], nshow, 0])
    ax[2].set_ylabel('cell (sample)')
    ax[2].set_title('step 3: raw binarised transient rising phases (30 Hz), registered cells')

    # 4. smoothed + pooled neural
    nshow2 = min(40, pl['neural'].shape[0])
    cells2 = np.linspace(0, pl['neural'].shape[0] - 1, nshow2).astype(int)
    ax[3].imshow(pl['neural'][cells2][:, t0_bin:t0_bin + n_win_bins], aspect='auto', cmap='magma',
                 interpolation='nearest', extent=[tb[0], tb[-1], nshow2, 0])
    ax[3].set_ylabel('cell (sample)')
    ax[3].set_title('step 4: gaussian sigma=3 frames + average-pool 3 frames -> 100 ms bins (final `neural`)')

    # 5. binned position vs discretised output bin -- verifies the discretisation
    axb = ax[4]
    axb.plot(tb, pl['position_binned_cm'][t0_bin:t0_bin + n_win_bins, 0] / BIN_SIZE_CM,
             lw=.8, label='x / 25 cm')
    axb.plot(tb, pl['position_binned_cm'][t0_bin:t0_bin + n_win_bins, 1] / BIN_SIZE_CM,
             lw=.8, label='y / 25 cm')
    axb2 = axb.twinx()
    axb2.step(tb, pl['pos_bin'][t0_bin:t0_bin + n_win_bins], where='mid', color='k', lw=1.2,
              label='output bin = 3*ybin+xbin')
    axb2.set_ylabel('output bin (0-8)')
    axb2.set_ylim(-0.5, 8.5)
    axb.set_ylabel('position / bin size')
    axb.set_title('step 5: discretisation check -- black trace = 3*floor(y/25)+floor(x/25) '
                 '(except the ~0.005% of samples tracked inside a wall, snapped to the nearest open bin)')
    axb.legend(loc='upper left')
    axb2.legend(loc='upper right')

    # 6. trial segmentation + retained bins (temporal alignment check)
    ax[5].step(tb, pl['pos_bin'][t0_bin:t0_bin + n_win_bins], where='mid', color='k', lw=1.,
               label='output bin')
    ax[5].fill_between(tb, -0.5, 8.5, where=pl['moving_bins'][t0_bin:t0_bin + n_win_bins],
                       color='g', alpha=.15, step='mid', label='bins kept')
    for (s, e, _n) in session['trial_bounds']:
        if s * BIN_FRAMES / FPS <= tb[-1]:
            ax[5].axvline(s * BIN_FRAMES / FPS, color='b', lw=1.5)
    ax[5].set_ylim(-0.5, 8.5)
    ax[5].set_xlabel('time in session (s)')
    ax[5].set_ylabel('output bin')
    ax[5].set_title('step 6: blue lines = 1-min trial boundaries; shaded = 100 ms bins written out')
    ax[5].legend(loc='upper right')

    # 7. occupancy of the 9 bins in the converted output vs the blocked partitions (input)
    allout = np.concatenate([o.ravel() for o in session['output']])
    occ = np.bincount(allout, minlength=9) / allout.size
    ax[6].bar(np.arange(9), occ, color=['r' if i in set(blocked.tolist()) else 'steelblue'
                                        for i in range(9)])
    ax[6].set_xticks(np.arange(9))
    ax[6].set_xticklabels([f'{i}\n{BIN_NAMES[i]}' for i in range(9)])
    ax[6].set_ylabel('fraction of converted timepoints')
    ax[6].set_title('step 7: occupancy of the 9 output bins; red bars = partitions marked blocked in `input` '
                    '(must be exactly zero)')

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    del session['_plot']


# ----------------------------------------------------------------------------------------------------
# Per-animal driver
# ----------------------------------------------------------------------------------------------------
def process_animal(task):
    """Process the requested days of one animal.

    `task` = (animal, day_list or None, plot_days) where `plot_days` is the set of days for which a
    full `processing_<session_id>.png` diagnostic figure should be written.
    """
    animal, days, plot_days = task
    plot_days = set(plot_days or ())
    t_start = time.time()
    sessions = []
    with h5py.File(os.path.join(DATA_DIR, f"{animal}.mat"), 'r') as f:
        n_days = f['trace'].shape[0]
        day_list = range(n_days) if days is None else days
        for day in day_list:
            t0 = time.time()
            position, trace, env, blocked = read_session(f, day)

            # cross-check the stored `blocked` list against the geometry name (reference get_env_mat)
            expected = blocked_from_env(env)
            assert np.array_equal(np.sort(blocked), expected), \
                f"{animal} day {day}: blocked {blocked} != {expected} implied by env '{env}'"

            show_processing = day in plot_days
            sess = process_session(position, trace, env, blocked,
                                   session_id=f"{animal}_day{day:02d}",
                                   show_processing=show_processing)
            if sess is None:
                print(f"  [{animal} day {day}] SKIPPED (fewer than 2 usable trials)", flush=True)
                continue
            sess['animal'] = animal
            sess['day'] = day
            if show_processing:
                plot_processing(sess, position, trace[:, ~np.all(np.isnan(trace), axis=0)],
                                f"processing_{sess['session_id']}.png")
                print(f"  wrote processing_{sess['session_id']}.png", flush=True)
            sessions.append(sess)
            print(f"  [{animal} day {day:2d}] env={env:<10s} cells {sess['n_cells_total']}"
                  f" -> reg {sess['n_registered']} -> kept {sess['n_neurons']}"
                  f" | {sess['n_bins']} bins, {sess['frac_moving']*100:.1f}% running"
                  f" | {len(sess['neural'])} trials"
                  f"{f' | {sess[chr(39)+chr(39)]}' if False else ''}"
                  f" | {time.time()-t0:.2f}s", flush=True)
    print(f"[{animal}] {len(sessions)} sessions in {time.time()-t_start:.1f}s", flush=True)
    return sessions


# ----------------------------------------------------------------------------------------------------
# Assembly into the target format
# ----------------------------------------------------------------------------------------------------
def build_dataset(all_sessions):
    subjects = sorted({s['animal'] for s in all_sessions})
    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(s['animal']) for s in all_sessions], dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': [],
        'input_names': [f'blocked_{n}' for n in BIN_NAMES],
        'output_names': ['position_bin'],
        'output_values': [list(BIN_NAMES)],
        'metadata': {},
    }
    session_info = []
    for s in all_sessions:
        data['neural'].append(s['neural'])
        data['input'].append(s['input'])
        data['output'].append(s['output'])
        data['brain_region_idx'].append(np.zeros(s['n_neurons'], dtype=np.int64))
        session_info.append({
            'session_id': s['session_id'],
            'subject': s['animal'],
            'day': int(s['day']),
            'environment': s['env'],
            'blocked_partitions': [int(b) for b in s['blocked']],
            'n_neurons': int(s['n_neurons']),
            'n_registered_cells': int(s['n_registered']),
            'n_cells_in_registry': int(s['n_cells_total']),
            'n_frames_30hz': int(s['n_frames']),
            'n_trials': len(s['neural']),
            'frac_bins_running': round(s['frac_moving'], 4),
            'n_samples_snapped_to_open_bin': int(s['n_snapped']),
        })

    data['metadata'] = {
        'task_description':
            'Mice freely foraged for 40 min/day in a 75 x 75 cm arena partitioned into a 3 x 3 grid, '
            'in which different subsets of the 9 partitions were blocked off with 25 cm walls to create '
            '10 distinct geometries (square, o, t, u, rectangle, +, i, l, bit donut, glenn). One geometry '
            'was presented per day; the random per-animal geometry sequence started and ended with the '
            'square and was repeated up to 3 times (31 days). CA1 populations were imaged with a '
            'miniscope. The decoder receives the arena geometry (which partitions are blocked) plus CA1 '
            'population activity and predicts which of the 9 spatial bins the animal occupies.',
        'time_bin_size': TIME_BIN_MS,
        'temporal_alignment_event':
            'Start of each 1-minute trial block, measured from the start of the (continuous, 40 min) '
            'recording session. The task is continuous free foraging, so there is no stimulus or '
            'behavioural event to align to; imaging and behaviour were acquired on the same DAQ at '
            '30 Hz and are frame-aligned in the source data.',
        'off_start': 0.0,
        'off_end': TRIAL_SEC,
        'dataset':
            'Lee JQ, Keinath AT, Cianfarano E, Brandon MP (2025). Identifying representational structure '
            'in CA1 to benchmark theoretical models of cognitive mapping. Neuron 113(2):307-320. '
            'doi:10.1016/j.neuron.2024.10.027; data doi:10.5281/zenodo.13993254',
        'recording_modality': 'one-photon miniscope calcium imaging of dorsal CA1 (GCaMP6f, GRIN lens)',
        'neural_units':
            'Rate of binarised calcium-transient rising phases. The distributed `trace` is the binary '
            'event vector the paper treats as the firing rate; it is smoothed with a gaussian of '
            'sigma = 3 frames and average-pooled over 3 frames, exactly as in the reference decoder '
            '(georepca1/src/utils.py:fit_decoder), giving the mean event occupancy per 100 ms bin.',
        'sampling_rate_hz_raw': FPS,
        'arena_size_cm': ARENA_CM,
        'spatial_bin_size_cm': BIN_SIZE_CM,
        'spatial_bin_convention':
            'bin = 3 * floor(y / 25 cm) + floor(x / 25 cm), matching the dataset\'s own '
            '[[0,1,2],[3,4,5],[6,7,8]] layout for `blocked`.',
        'input_description':
            'Nine binary values, one per arena partition: 1 = partition blocked off in this session, '
            '0 = partition open. Constant within a session (and therefore within a trial).',
        'output_description':
            'Index (0-8) of the 3 x 3 spatial bin containing the animal in each 100 ms time bin.',
        'curation':
            'Cells: registered on that session (trace not all-NaN) and > 5 binarised events while '
            'running (decode_position_within cell_threshold=5). No place-cell selection -- the paper '
            'includes all cells. Timepoints: only bins in which the animal ran faster than 5 cm/s '
            '(decode_position_within v_thresh=5, velocity smoothed with sigma=5 frames) are retained, '
            'as in the paper\'s own position decoder. Trials retaining < 3 s are dropped; sessions with '
            '< 2 usable trials are dropped. All 7 animals and all 207 sessions are used.',
        'trial_definition':
            'Contiguous 60 s blocks (600 x 100 ms bins) of each 40 min session; the trailing partial '
            'block is kept when it spans >= 10 s. Within a trial only running bins are written, so '
            'trials have unequal lengths.',
        'session_info': session_info,
    }
    return data


def summarise(data):
    n_sessions = len(data['neural'])
    n_trials = sum(len(t) for t in data['neural'])
    n_neurons = [n[0].shape[0] for n in data['neural']]
    tp = [t.shape[1] for s in data['neural'] for t in s]
    allout = np.concatenate([t.ravel() for s in data['output'] for t in s])
    occ = np.bincount(allout, minlength=9) / allout.size
    print("\n" + "=" * 90)
    print(f"Subjects                : {len(data['subjects'])}  {data['subjects']}")
    print(f"Sessions                : {n_sessions}")
    print(f"Trials                  : {n_trials}  (mean {n_trials/n_sessions:.1f} per session)")
    print(f"Neurons / session       : mean {np.mean(n_neurons):.1f}  min {min(n_neurons)}  "
          f"max {max(n_neurons)}  total(cell-sessions) {sum(n_neurons)}")
    print(f"Timepoints / trial      : mean {np.mean(tp):.1f}  min {min(tp)}  max {max(tp)}")
    print(f"Total timepoints        : {int(np.sum(tp))}  ({np.sum(tp)*TIME_BIN_MS/1000/3600:.2f} h)")
    print(f"Time bin                : {TIME_BIN_MS:g} ms")
    print(f"Output bin distribution : {np.round(occ, 4).tolist()}")
    print(f"Output range            : [{allout.min()}, {allout.max()}]")
    inp = np.array([t for s in data['input'] for t in s])
    print(f"Input range             : [{inp.min():g}, {inp.max():g}]   "
          f"mean blocked partitions/trial {inp.sum(1).mean():.3f}")
    print("=" * 90 + "\n")


# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', help='path of the output pickle file')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', default=True, help='process all sessions (default)')
    g.add_argument('--sample', action='store_true', help='process only 2 sessions, for testing')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=7, help='parallel worker processes (one per animal)')
    args = ap.parse_args()

    t_all = time.time()

    plot = [5] if args.show_processing else []
    if args.sample:
        # one session from each of two animals: two subjects and two different geometries
        tasks = [(ANIMALS[0], [5], plot),
                 (ANIMALS[1], [5], plot)]
    else:
        # in --full mode only the first two sessions of the first animal are plotted
        tasks = [(a, None, []) for a in ANIMALS]
        if args.show_processing:
            tasks[0] = (ANIMALS[0], None, [0, 5])

    print(f"Converting {'2 sample sessions' if args.sample else 'all sessions'} from "
          f"{len(tasks)} animal file(s)...", flush=True)

    if len(tasks) > 1 and args.nproc > 1:
        with Pool(min(args.nproc, len(tasks))) as pool:
            results = pool.map(process_animal, tasks)
    else:
        results = [process_animal(t) for t in tasks]

    all_sessions = [s for r in results for s in r]
    all_sessions.sort(key=lambda s: (s['animal'], s['day']))
    print(f"\nProcessed {len(all_sessions)} sessions in {time.time()-t_all:.1f}s", flush=True)

    data = build_dataset(all_sessions)
    summarise(data)

    t0 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) "
          f"in {time.time()-t0:.1f}s", flush=True)
    print(f"Total time {time.time()-t_all:.1f}s", flush=True)


if __name__ == '__main__':
    sys.exit(main())
