"""
Convert the Lee, Keinath, Cianfarano & Brandon (2025, Neuron) CA1 geometric
deformation dataset (georepca1) into the decoder-compatible pickle format.

Decoder task:
    inputs  : environment geometry (which of the 3x3 partitions are blocked), static per trial
    outputs : mouse position discretized into 3 x 3 = 9 spatial bins, time-varying

Processing follows the reference code (/app/code/georepca1/src/utils.py):
  * neural data are the released binary rising-phase ("event") traces, treated as
    firing rate (paper: "This binary vector was treated as the firing rate").
  * temporal binning identical to `fit_decoder`/`test_decoder`:
        gaussian_filter1d(trace, sigma=3 frames) then AvgPool1d(kernel=3, stride=3)
    i.e. 100 ms bins at the 30 Hz acquisition rate.
  * frames with speed <= 5 cm/s are excluded, as in `decode_position_within`
    (v_filt_size=5 frames gaussian smoothing of speed, v_thresh=5 cm/s).
  * cells are kept if registered on that day (trace not NaN) and if they have
    more than 5 events during moving frames (`cell_threshold=5`).
  * position is binned by mean position within each temporal bin then floor
    division by (max position + buffer)/n_bins, with n_bins=3 (task requirement)
    instead of the reference n_bins=15.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import time

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------- constants
DATA_DIR = '/app/data'
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

FPS = 30.0                 # acquisition rate of both imaging and behaviour streams
TEMPORAL_BIN_FRAMES = 3    # reference `temporal_bin_size=3` -> 100 ms bins
TRACE_SIGMA_FRAMES = 3     # reference gaussian_filter1d(traces, sigma=temporal_bin_size)
V_FILT_SIGMA = 5           # reference `v_filt_size=5`
V_THRESH = 5.0             # reference `v_thresh=5` cm/s
CELL_THRESHOLD = 5         # reference `cell_threshold=5` events while moving
N_SPATIAL_BINS = 3         # task requirement: 3 x 3 = 9 spatial bins
ARENA_SIZE = 75.0          # cm (paper: 75 x 75 cm)
BUFFER = 1e-15             # reference buffer in decode_position_within
TRIAL_SECONDS = 60.0       # task requirement: 1-minute trials
TRIAL_BINS = int(round(TRIAL_SECONDS * FPS / TEMPORAL_BIN_FRAMES))   # 600 bins
MIN_TRIAL_BINS = 30        # drop trials with < 3 s of moving data

TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS  # 100 ms


# ----------------------------------------------------------------------------- helpers
def bin_frames(x, nframes_per_bin=TEMPORAL_BIN_FRAMES):
    """Average-pool the last axis of x in non-overlapping windows (drops the remainder).

    Equivalent to torch.nn.AvgPool1d(kernel_size=k, stride=k) used in the reference
    `fit_decoder`, but vectorised in numpy.
    """
    T = x.shape[-1]
    nbins = T // nframes_per_bin
    x = x[..., :nbins * nframes_per_bin]
    return x.reshape(*x.shape[:-1], nbins, nframes_per_bin).mean(axis=-1)


def compute_speed(position_xy):
    """Speed in cm/s, smoothed exactly as in `decode_position_within`.

    position_xy: (2, T) array in cm.  Returns (T,) array; the first frame is 0
    (the reference leaves vel_idx[0] = False).
    """
    d = np.linalg.norm(np.diff(position_xy, axis=1), axis=0) * FPS
    speed = np.zeros(position_xy.shape[1], dtype=np.float64)
    speed[1:] = gaussian_filter1d(d, sigma=V_FILT_SIGMA)
    return speed


def blocked_to_vector(blocked_day):
    """Convert the per-session `blocked` entry into a 9-d binary vector.

    1 = partition blocked (occluded), 0 = open.  Partition index p = 3*ybin + xbin,
    matching the `[[0, 1, 2], [3, 4, 5], [6, 7, 8]]` layout documented in the code
    README (verified against occupancy: mice are never in a blocked partition).
    """
    b = np.atleast_1d(np.asarray(blocked_day[0]).ravel()).astype(float)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if b.size and b[0] >= 0:
        vec[b.astype(int)] = 1.0
    return vec


def position_to_class(position_binned_xy):
    """Discretize (2, nbins) position in cm into 9 classes (3*ybin + xbin)."""
    bin_down = (ARENA_SIZE + BUFFER) / N_SPATIAL_BINS  # 25 cm, = reference bin_down formula
    xy = np.floor(position_binned_xy / bin_down).astype(int)
    xy = np.clip(xy, 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)


# ----------------------------------------------------------------------------- per-session conversion
def process_session(trace_day, position_day, blocked_day, verbose=False):
    """Process a single recording day.

    trace_day: (ncells, T) binary trace with NaN rows for unregistered cells
    position_day: (2, T) position in cm
    blocked_day: entry of the `blocked` list for this day

    Returns dict with lists of per-trial neural / input / output arrays plus stats.
    """
    stats = {}
    T = position_day.shape[1]
    assert trace_day.shape[1] == T, 'trace and position frame counts differ'

    # ---- curation: registered cells --------------------------------------------------
    nan_any = np.isnan(trace_day).any(axis=1)
    nan_all = np.isnan(trace_day).all(axis=1)
    assert np.array_equal(nan_any, nan_all), 'found partially-NaN cells'
    registered = ~nan_all

    # ---- speed filter (reference decode_position_within) ------------------------------
    speed = compute_speed(position_day)
    moving = speed > V_THRESH

    # ---- curation: cells with > 5 events while moving ---------------------------------
    events_moving = np.nansum(trace_day[:, moving], axis=1)
    keep_cells = registered & (events_moving > CELL_THRESHOLD)
    stats['n_registered'] = int(registered.sum())
    stats['n_kept_cells'] = int(keep_cells.sum())
    stats['frac_moving_frames'] = float(moving.mean())

    tr = trace_day[keep_cells].astype(np.float32)

    # ---- temporal binning (reference fit_decoder) -------------------------------------
    # smooth the continuous session trace, then average-pool 3 frames -> 100 ms bins
    tr_smooth = gaussian_filter1d(tr, sigma=TRACE_SIGMA_FRAMES, axis=1)
    neural_binned = bin_frames(tr_smooth).astype(np.float32) * FPS  # event rate in Hz
    pos_binned = bin_frames(position_day)                            # (2, nbins), cm
    speed_binned = bin_frames(speed[np.newaxis, :])[0]               # (nbins,), cm/s
    nbins = neural_binned.shape[1]

    moving_bins = speed_binned > V_THRESH
    out_class = position_to_class(pos_binned)

    stats['n_bins_total'] = int(nbins)
    stats['frac_moving_bins'] = float(moving_bins.mean())

    geometry = blocked_to_vector(blocked_day)

    # ---- cut into 1-minute trials -----------------------------------------------------
    neural_trials, input_trials, output_trials = [], [], []
    n_full_trials = nbins // TRIAL_BINS       # drop trailing partial minute
    trial_bin_counts = []
    for t in range(n_full_trials):
        sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
        m = moving_bins[sl]
        if m.sum() < MIN_TRIAL_BINS:
            continue
        neural_trials.append(np.ascontiguousarray(neural_binned[:, sl][:, m]))
        output_trials.append(out_class[sl][m][np.newaxis, :].astype(np.int64))
        input_trials.append(geometry.copy())
        trial_bin_counts.append(int(m.sum()))

    stats['n_trials'] = len(neural_trials)
    stats['n_timepoints'] = int(np.sum(trial_bin_counts)) if trial_bin_counts else 0
    stats['geometry'] = geometry

    aux = dict(speed=speed, moving=moving, speed_binned=speed_binned,
               moving_bins=moving_bins, pos_binned=pos_binned, out_class=out_class,
               neural_binned=neural_binned, keep_cells=keep_cells, registered=registered)
    return neural_trials, input_trials, output_trials, stats, aux


# ----------------------------------------------------------------------------- plotting
def plot_processing(session_id, position_day, trace_day, aux, stats, outdir='/app'):
    """Plot every processing step for one session to convince the reader it is correct."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    speed = aux['speed']; moving = aux['moving']
    pos_binned = aux['pos_binned']; out_class = aux['out_class']
    neural_binned = aux['neural_binned']; moving_bins = aux['moving_bins']
    geometry = stats['geometry']
    nbins = neural_binned.shape[1]
    tbin = np.arange(nbins) * TIME_BIN_MS / 1000.0

    fig, ax = plt.subplots(5, 2, figsize=(20, 22))

    # (0,0) trajectory with partition grid
    ax[0, 0].plot(position_day[0], position_day[1], lw=0.2, color='k')
    for v in (25, 50):
        ax[0, 0].axvline(v, color='r'); ax[0, 0].axhline(v, color='r')
    ax[0, 0].set_title(f'{session_id}: raw trajectory (cm), red = 3x3 partition edges')
    ax[0, 0].set_xlabel('x (cm)'); ax[0, 0].set_ylabel('y (cm)')
    ax[0, 0].set_xlim(0, 75); ax[0, 0].set_ylim(0, 75)

    # (0,1) geometry input vector as a 3x3 image + occupancy
    occ = np.zeros(9)
    for c in range(9):
        occ[c] = (out_class == c).mean()
    im = ax[0, 1].imshow(occ.reshape(3, 3), origin='lower', cmap='viridis')
    for c in range(9):
        y, x = divmod(c, 3)
        ax[0, 1].text(x, y, f'p{c}\nocc={occ[c]:.3f}\n' + ('BLOCKED' if geometry[c] else 'open'),
                      ha='center', va='center', color='w')
    ax[0, 1].set_title('occupancy of each partition (rows = y) vs input geometry vector')
    plt.colorbar(im, ax=ax[0, 1])

    # (1,0) speed and threshold, first 2 minutes of frames
    n = int(120 * FPS)
    ax[1, 0].plot(np.arange(n) / FPS, speed[:n], lw=0.7)
    ax[1, 0].axhline(V_THRESH, color='r', ls='--', label='5 cm/s threshold')
    ax[1, 0].fill_between(np.arange(n) / FPS, 0, speed[:n].max(), where=moving[:n],
                          color='g', alpha=0.15, label='moving (kept)')
    ax[1, 0].legend(); ax[1, 0].set_xlabel('time (s)'); ax[1, 0].set_ylabel('speed (cm/s)')
    ax[1, 0].set_title(f'speed filter, frac moving frames = {stats["frac_moving_frames"]:.3f}')

    # (1,1) raw vs binned position (first 2 min) - temporal alignment check
    nb = int(120 * FPS / TEMPORAL_BIN_FRAMES)
    ax[1, 1].plot(np.arange(int(120 * FPS)) / FPS, position_day[0, :int(120 * FPS)],
                  lw=0.7, label='raw x')
    ax[1, 1].plot(tbin[:nb], pos_binned[0, :nb], lw=1.2, label='binned x (100 ms)')
    ax[1, 1].plot(np.arange(int(120 * FPS)) / FPS, position_day[1, :int(120 * FPS)],
                  lw=0.7, label='raw y')
    ax[1, 1].plot(tbin[:nb], pos_binned[1, :nb], lw=1.2, label='binned y (100 ms)')
    for v in (25, 50):
        ax[1, 1].axhline(v, color='r', ls=':')
    ax[1, 1].legend(); ax[1, 1].set_xlabel('time (s)'); ax[1, 1].set_ylabel('position (cm)')
    ax[1, 1].set_title('raw vs temporally binned position (no lag)')

    # (2,0) discretization check: binned x/y vs class
    ax[2, 0].plot(tbin[:nb], pos_binned[0, :nb] / 25.0, label='x / 25 cm')
    ax[2, 0].plot(tbin[:nb], pos_binned[1, :nb] / 25.0, label='y / 25 cm')
    ax[2, 0].step(tbin[:nb], out_class[:nb] / 3.0, where='mid', color='k',
                  label='output class / 3 (= ybin + xbin/3)')
    ax[2, 0].legend(); ax[2, 0].set_xlabel('time (s)')
    ax[2, 0].set_title('discretization of position into 9 classes')

    # (2,1) scatter of position colored by assigned class
    sc = ax[2, 1].scatter(pos_binned[0], pos_binned[1], c=out_class, cmap='tab10', s=1)
    for v in (25, 50):
        ax[2, 1].axvline(v, color='k'); ax[2, 1].axhline(v, color='k')
    ax[2, 1].set_title('binned position colored by output class (should tile the 3x3 grid)')
    plt.colorbar(sc, ax=ax[2, 1])

    # (3,0) raw binary raster of 30 cells (first 2 min)
    keep_idx = np.where(aux['keep_cells'])[0][:30]
    raster = trace_day[keep_idx][:, :int(120 * FPS)]
    ax[3, 0].imshow(raster, aspect='auto', cmap='Greys', interpolation='nearest',
                    extent=[0, 120, 0, len(keep_idx)])
    ax[3, 0].set_title('raw binary event trace (30 kept cells, first 2 min)')
    ax[3, 0].set_xlabel('time (s)')

    # (3,1) binned neural rates for the same cells
    ax[3, 1].imshow(neural_binned[np.arange(len(keep_idx)), :][:, :nb], aspect='auto',
                    cmap='viridis', interpolation='nearest', extent=[0, 120, 0, len(keep_idx)])
    ax[3, 1].set_title('binned event rate (Hz), same cells and window')
    ax[3, 1].set_xlabel('time (s)')

    # (4,0) trial structure: which bins are kept
    ax[4, 0].plot(tbin, moving_bins.astype(float), lw=0.3)
    for t in range(0, nbins // TRIAL_BINS + 1):
        ax[4, 0].axvline(t * TRIAL_BINS * TIME_BIN_MS / 1000.0, color='r', lw=0.5)
    ax[4, 0].set_title(f'kept (moving) bins and 1-min trial boundaries; '
                       f'{stats["n_trials"]} trials, {stats["n_timepoints"]} timepoints')
    ax[4, 0].set_xlabel('time (s)')

    # (4,1) population mean rate and output class over one trial
    sl = slice(0, TRIAL_BINS)
    ax2 = ax[4, 1].twinx()
    ax[4, 1].plot(tbin[sl], neural_binned[:, sl].mean(axis=0), color='b', lw=0.8)
    ax[4, 1].set_ylabel('population mean rate (Hz)', color='b')
    ax2.step(tbin[sl], out_class[sl], where='mid', color='r', lw=0.8)
    ax2.set_ylabel('output class', color='r')
    ax[4, 1].set_title('trial 0: neural and output on the same time base')
    ax[4, 1].set_xlabel('time (s)')

    fig.tight_layout()
    fn = os.path.join(outdir, f'processing_{session_id}.png')
    fig.savefig(fn, dpi=90)
    plt.close(fig)
    print(f'  wrote {fn}')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    args = ap.parse_args()
    sample = args.sample and not args.full

    animals = ANIMALS[:1] if sample else ANIMALS

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': ANIMALS if not sample else ANIMALS[:1],
        'subject_idx': [],
        'brain_regions': ['CA1'],
        'brain_region_idx': [],
        'input_names': [f'blocked_partition_{i}' for i in range(9)],
        'output_names': ['position_bin'],
        'output_values': [[f'x{c % 3}y{c // 3}' for c in range(9)]],
        'metadata': {},
    }
    session_info = []
    nplotted = 0
    t_start = time.time()
    per_session_times = []

    for ai, animal in enumerate(animals):
        t0 = time.time()
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        t_load = time.time() - t0
        traces = dat['trace']
        positions = dat['position']
        envs = np.array(dat['envs']).ravel()
        blocked = dat['blocked']
        ndays = positions.shape[0]
        if sample:
            ndays = 2
        print(f'{animal}: loaded in {t_load:.1f} s, {ndays} sessions, '
              f'{traces.shape[1]} registered-or-not cells', flush=True)

        for d in range(ndays):
            ts = time.time()
            trace_day = np.asarray(traces[d])
            position_day = np.asarray(positions[d])
            neural_trials, input_trials, output_trials, stats, aux = process_session(
                trace_day, position_day, blocked[d])

            if len(neural_trials) < 2:
                print(f'  WARNING session {animal} day {d} has < 2 trials, skipped')
                continue

            data['neural'].append(neural_trials)
            data['input'].append(input_trials)
            data['output'].append(output_trials)
            data['subject_idx'].append(ai)
            data['brain_region_idx'].append(np.zeros(stats['n_kept_cells'], dtype=int))
            session_info.append(dict(animal=animal, day=int(d), env=str(envs[d]),
                                     blocked=np.where(stats['geometry'] > 0)[0].tolist(),
                                     n_registered=stats['n_registered'],
                                     n_neurons=stats['n_kept_cells'],
                                     n_trials=stats['n_trials'],
                                     n_timepoints=stats['n_timepoints'],
                                     frac_moving=stats['frac_moving_frames']))
            per_session_times.append(time.time() - ts)
            print(f'  {animal} day {d:2d} env={envs[d]:10s} '
                  f'cells {stats["n_registered"]}->{stats["n_kept_cells"]} '
                  f'trials {stats["n_trials"]} bins {stats["n_timepoints"]} '
                  f'({time.time() - ts:.1f} s)', flush=True)

            if args.show_processing and nplotted < 2:
                plot_processing(f'{animal}_day{d}', position_day, trace_day, aux, stats)
                nplotted += 1

        del dat, traces, positions

    data['subject_idx'] = np.array(data['subject_idx'], dtype=int)

    # ---- metadata ---------------------------------------------------------------------
    data['metadata'] = {
        'task_description': (
            'Mice freely forage for 40 min/day in a 75 x 75 cm arena partitioned into a '
            '3 x 3 grid of 25 cm partitions; across days the geometry is changed by '
            'blocking subsets of the 9 partitions (10 distinct geometries, sequence '
            'repeated up to 3 times per mouse). CA1 population activity is recorded with '
            'miniscope calcium imaging. Decoder task: predict which of the 9 spatial '
            'partitions the mouse occupies from CA1 activity, given the environment '
            'geometry (which partitions are blocked) as input.'),
        'time_bin_size': TIME_BIN_MS,
        'temporal_alignment_event': (
            'Start of each 1-minute trial window, measured from the start of the '
            'continuous 40-min recording session (there are no experimenter-defined '
            'trials in this free-foraging task).'),
        'off_start': 0.0,
        'off_end': TRIAL_SECONDS,
        'session_info': session_info,
        'source': ('Lee, Keinath, Cianfarano & Brandon (2025) Neuron 113(2):307-320, '
                   'georepca1 dataset (Zenodo 10.5281/zenodo.13993254)'),
        'neural_data_type': ('event rate (Hz) of rise-extracted calcium transients: the '
                             'released binary rising-phase vector smoothed with '
                             'gaussian_filter1d(sigma=3 frames) and average-pooled over '
                             '3 frames (100 ms), multiplied by 30 Hz'),
        'recording_rate_hz': FPS,
        'curation': {
            'neurons': ('registered on that day (non-NaN trace) AND > 5 events during '
                        'moving frames (reference decode_position_within cell_threshold=5)'),
            'timepoints': ('bins with mean speed <= 5 cm/s excluded (reference '
                           'v_thresh=5 cm/s, speed smoothed with gaussian sigma=5 frames)'),
            'trials': ('consecutive 1-min windows of session time; trailing partial '
                       'window dropped; trials with < 30 moving bins (3 s) dropped'),
            'sessions': 'all 207 sessions of all 7 mice retained',
        },
        'arena_size_cm': ARENA_SIZE,
        'spatial_bin_size_cm': ARENA_SIZE / N_SPATIAL_BINS,
        'output_encoding': 'class = 3 * ybin + xbin, matching the blocked-partition indexing',
    }

    # ---- sanity checks ----------------------------------------------------------------
    nsessions = len(data['neural'])
    ntrials = sum(len(s) for s in data['neural'])
    ntimepoints = sum(tr.shape[1] for s in data['neural'] for tr in s)
    nneurons = sum(s[0].shape[0] for s in data['neural'])
    print('\n==== conversion summary ====')
    print(f'sessions           : {nsessions}')
    print(f'trials             : {ntrials} (mean {ntrials / nsessions:.1f} per session)')
    print(f'timepoints         : {ntimepoints} ({ntimepoints * TIME_BIN_MS / 1000 / 60:.1f} min)')
    print(f'neurons (cell-sessions): {nneurons} (mean {nneurons / nsessions:.1f} per session)')
    print(f'registered cell-sessions: {sum(si["n_registered"] for si in session_info)}')

    for si_i, sess in enumerate(data['neural']):
        assert len(sess) == len(data['input'][si_i]) == len(data['output'][si_i])
        n = sess[0].shape[0]
        for tr_i, tr in enumerate(sess):
            assert tr.shape[0] == n
            assert tr.shape[1] == data['output'][si_i][tr_i].shape[1]
            assert np.isfinite(tr).all()
        # output classes must never fall in a blocked partition
        geom = data['input'][si_i][0]
        classes = np.unique(np.concatenate([o.ravel() for o in data['output'][si_i]]))
        bad = [c for c in classes if geom[c] > 0]
        if bad:
            print(f'  WARNING session {si_i}: output classes {bad} are blocked partitions '
                  f'(occupancy {[float(np.mean(np.concatenate([o.ravel() for o in data["output"][si_i]]) == c)) for c in bad]})')
    allout = np.concatenate([o.ravel() for s in data['output'] for o in s])
    print('output class distribution:',
          np.round(np.bincount(allout, minlength=9) / allout.size, 4))
    allneural_mean = np.mean([tr.mean() for s in data['neural'] for tr in s])
    print(f'mean neural value (Hz): {allneural_mean:.4f}')

    if per_session_times:
        print(f'mean processing time per session: {np.mean(per_session_times):.2f} s')
    print(f'total elapsed: {time.time() - t_start:.1f} s')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB)')


if __name__ == '__main__':
    main()
