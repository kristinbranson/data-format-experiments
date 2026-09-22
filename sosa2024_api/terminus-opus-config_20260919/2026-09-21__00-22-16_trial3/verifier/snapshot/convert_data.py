"""
Convert the Sosa, Plitt & Giocomo (2025) CA1 VR reward-switch dataset (DANDI 001361)
into the decoder-compatible pickle format.

Processing follows the reference code (https://github.com/GiocomoLab/Sosa_et_al_2024,
copy in /app/code) and paper methods:
  * trials are the on-track laps, from `trial_start` to `teleport` (ITI excluded;
    laser power was blanked during the ITI in most sessions)
  * neural signal = OASIS-deconvolved "events" computed from the reference dF/F
    (neuropil subtraction 0.7, per-trial maximin baseline, 2-sample Gaussian smoothing),
    i.e. `reward_relative.preprocessing.dff(..., deconvolve=True)`
  * ROIs: suite2p `iscell`, pooled across planes, minus putative interneurons
    (Pearson r(dF/F, speed) > 0.5)
  * behavior streams are already interpolated onto the imaging frame clock in the NWB

Usage:  python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import re
import sys
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from pynwb import NWBHDF5IO
from suite2p.extraction import dcnv

# ----------------------------------------------------------------------------- params
FRAME_RATE = 15.5078125          # Hz, per imaging plane (from the NWB)
TIME_BIN_MS = 1000.0 / FRAME_RATE  # 64.48 ms
NEU_COEF = 0.7                   # neuropil coefficient (reference pp.dff)
TAU = 0.7                        # GCaMP7f decay constant used by the reference code
BASELINE_SIGMA = 15              # frames, gaussian smoothing before maximin
BASELINE_WIN = 300               # frames (~19.3 s ~ the paper's 20 s maximin window)
DFF_SMOOTH_SIGMA = 2             # frames (~0.129 s), paper's 2-sample gaussian
SPEED_CORR_THR = 0.5             # putative interneuron exclusion
LICK_ERROR_FRAC = 0.30           # >30% of frames with cumulative lick count > 2
MIN_TRIAL_FRAMES = 10            # drop pathologically short trials
TRACK_LENGTH = 450.0

# reward zones, in cm (paper: zone A 80-130, B 200-250, C 320-370)
REWARD_ZONES = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
CHANGE_TRIAL = 30                # reference behavior.get_reward_zones(change_trial=30)

POS_EDGES = [90.0, 180.0, 270.0, 360.0]     # 5 equal bins over the 450 cm track
SPEED_EDGES = [2.0, 10.0, 20.0, 40.0]       # cm/s

NEURAL_SIGNAL_DESC = {
    'dff': ('dF/F computed as in Sosa et al. reward_relative.preprocessing.dff: '
            'neuropil subtraction (coefficient 0.7, per-trial neuropil mean added back), '
            'per-trial maximin baseline (~19.3 s window), dF/F=(F-baseline)/|baseline|, '
            'smoothed with a 2-frame (~0.13 s) Gaussian'),
    'events': ('OASIS-deconvolved "events" from the same dF/F (suite2p dcnv.oasis, '
               'tau=0.7, fs=15.5078 Hz), i.e. the activity rate used by the paper\'s '
               'circular-linear RR-position decoder and GLM'),
}

INPUT_NAMES = ['time_from_trial_start', 'environment', 'trial_number', 'prev_trial_outcome']
OUTPUT_NAMES = ['dist_to_reward_zone', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm (in zone)',
     '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
    ['omitted', 'rewarded'],
]


# ------------------------------------------------------------------- reference helpers
def scene_reward_zones(scene, n_trials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label from the scene name.

    Re-implements reward_relative.behavior.get_reward_zones: non-switch scenes
    (`Env<k>_Location<Z>`) use one zone for all trials; switch scenes
    (`..._<Z1>_to_..._<Z2>`) change zone after `change_trial` trials.
    """
    m = re.fullmatch(r'Env\d_Location([ABC])', scene)
    if m:
        return np.array([m.group(1)] * n_trials, dtype='<U1')
    # switch scenes: 'Env1_LocationA_to_C', 'Env1_A_to_Env2_B', 'Env2_B_to_Env1_A', ...
    m = re.search(r'([ABC])_to_(?:Env\d_)?(?:Location)?([ABC])$', scene)
    if m:
        labels = np.array([m.group(1)] * n_trials, dtype='<U1')
        labels[change_trial:] = m.group(2)
        return labels
    raise ValueError(f'Unrecognized scene name: {scene}')


def compute_dff_and_events(F, Fneu, starts, stops):
    """Reference dF/F (preprocessing.dff) + OASIS deconvolution, within trials only.

    F, Fneu: (n_rois, n_frames) raw suite2p traces
    starts, stops: trial start / teleport frame indices
    Returns (dff, events), both (n_rois, n_frames) with NaN outside trials.
    """
    f_ = np.full(F.shape, np.nan, dtype=np.float32)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        f_[:, s:e] = F[:, s:e]
        fneu_[:, s:e] = Fneu[:, s:e]

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        # add back the per-trial neuropil mean so dF/F is not divided by ~0
        f_[:, s:e] += NEU_COEF * np.nanmean(fneu_[:, s:e], axis=1, keepdims=True)
        seg = gaussian_filter1d(f_[:, s:e], BASELINE_SIGMA, axis=-1)
        seg = minimum_filter1d(seg, BASELINE_WIN, axis=-1)
        seg = maximum_filter1d(seg, BASELINE_WIN, axis=-1)
        flow[:, s:e] = seg

    valid = ~np.isnan(f_[0])
    dff = np.full(F.shape, np.nan, dtype=np.float32)
    dff[:, valid] = (f_[:, valid] - flow[:, valid]) / np.abs(flow[:, valid])

    events = np.full(F.shape, np.nan, dtype=np.float32)
    for s, e in zip(starts, stops):
        dff[:, s:e] = gaussian_filter1d(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=-1)
        events[:, s:e] = dcnv.oasis(np.ascontiguousarray(dff[:, s:e]), 2000, TAU, FRAME_RATE)
    return dff, events


def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) to the nearest point of the reward zone; 0 inside."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d


def discretize_distance(d):
    """7-way discretization of the signed distance to the reward zone."""
    b = np.full(d.shape, 3, dtype=np.int64)       # 3: exactly 0 (inside the zone)
    b[(d < 0) & (d >= -10)] = 2
    b[(d < -10) & (d >= -50)] = 1
    b[d < -50] = 0
    b[(d > 0) & (d <= 10)] = 4
    b[(d > 10) & (d <= 50)] = 5
    b[d > 50] = 6
    return b


# ------------------------------------------------------------------ session conversion
def process_session(path, show_processing=False, plot_dir='/app', signal='dff'):
    t_start = time.time()
    timing = {}
    with NWBHDF5IO(path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        subject = nwb.subject.subject_id
        session_id = nwb.session_id
        identifier = nwb.identifier
        scene = identifier.split('/')[-1]
        animal_orig = identifier.split('/')[-3]
        date = identifier.split('/')[-2]
        region = nwb.imaging_planes['ImagingPlane'].location

        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        g = lambda k: np.asarray(beh[k].data[:])
        t = np.asarray(beh['position'].timestamps[:])
        pos = g('position')
        speed = g('speed')
        lick = g('lick')
        env = g('environment')
        rz_series = g('reward_zone')
        scanning = g('scanning')
        starts = np.where(g('trial_start') > 0)[0]
        stops = np.where(g('teleport') > 0)[0]
        reward_times = np.asarray(beh['Reward'].timestamps[:])

        t_load0 = time.time()
        oph = nwb.processing['ophys'].data_interfaces
        ps = oph['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
        iscell = np.asarray(ps['iscell'].data[:])[:, 0] == 1
        plane_idx = np.asarray(ps['planeIdx'].data[:]).astype(int)
        F_list, Fneu_list, keep_list = [], [], []
        for plane in sorted(np.unique(plane_idx)):
            name = f'plane{plane}'
            Fp = np.asarray(oph['Fluorescence'].roi_response_series[name].data[:]).T
            Np = np.asarray(oph['Neuropil'].roi_response_series[name].data[:]).T
            keep = iscell[plane_idx == plane]
            F_list.append(Fp[keep])
            Fneu_list.append(Np[keep])
            keep_list.append(np.full(int(keep.sum()), plane))
        F = np.concatenate(F_list, axis=0).astype(np.float32)
        Fneu = np.concatenate(Fneu_list, axis=0).astype(np.float32)
        del F_list, Fneu_list

        # A few multi-plane sessions (m17/m18) store one extra imaging frame relative
        # to the behavior stream. Truncate everything to the common length; the last
        # teleport always falls before this boundary, so no trial data is lost.
        n_frames = min(F.shape[1], len(t))
        n_frames_trunc = (F.shape[1] - n_frames, len(t) - n_frames)
        if F.shape[1] != n_frames:
            F = F[:, :n_frames]
            Fneu = Fneu[:, :n_frames]
        if len(t) != n_frames:
            t = t[:n_frames]; pos = pos[:n_frames]; speed = speed[:n_frames]
            lick = lick[:n_frames]; env = env[:n_frames]
            rz_series = rz_series[:n_frames]; scanning = scanning[:n_frames]
        timing['load_ophys'] = time.time() - t_load0

    n_trials_raw = len(starts)
    assert len(stops) == n_trials_raw, 'trial_start/teleport count mismatch'
    assert np.all(stops > starts), 'teleport before trial_start'
    assert stops[-1] <= F.shape[1], 'teleport index beyond available frames'
    n_rois_iscell = F.shape[0]

    # ----- neural processing (reference pp.dff + OASIS)
    t0 = time.time()
    dff, events = compute_dff_and_events(F, Fneu, starts, stops)
    timing['dff_oasis'] = time.time() - t0

    # ----- putative interneuron exclusion: Pearson r(dF/F, speed) > 0.5 on valid frames
    valid = ~np.isnan(dff[0])
    sp_v = speed[valid]
    dff_v = dff[:, valid]
    sp_c = sp_v - sp_v.mean()
    dff_c = dff_v - dff_v.mean(axis=1, keepdims=True)
    denom = np.sqrt((dff_c ** 2).sum(axis=1) * (sp_c ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        r_speed = (dff_c @ sp_c) / denom
    keep_cells = ~(r_speed > SPEED_CORR_THR)
    keep_cells &= np.isfinite(r_speed)          # drop dead ROIs (zero variance)
    n_interneurons = int(np.sum(r_speed > SPEED_CORR_THR))
    if signal == 'dff':
        events = dff          # optional alternative neural signal (for comparison tests)
    events = events[keep_cells]
    dff_keep = dff[keep_cells]
    plane_of_cell = np.concatenate(keep_list)[keep_cells]
    n_cells = events.shape[0]

    # ----- per-trial task variables
    zone_labels = scene_reward_zones(scene, n_trials_raw)
    reward_idx = np.searchsorted(t, reward_times)
    rewarded = np.zeros(n_trials_raw, dtype=int)
    env_trial = np.zeros(n_trials_raw, dtype=int)
    lick_error = np.zeros(n_trials_raw, dtype=bool)
    for i, (s, e) in enumerate(zip(starts, stops)):
        got_reward = np.any((reward_idx >= s) & (reward_idx < e))
        in_zone = np.any(rz_series[s:e] > 0)
        rewarded[i] = int(got_reward and in_zone)     # behavior.get_trial_types
        env_vals = np.unique(env[s:e])
        env_trial[i] = int(env_vals[0]) if len(env_vals) == 1 else int(np.round(np.median(env[s:e])))
        lick_error[i] = (lick[s:e] > 2).mean() > LICK_ERROR_FRAC

    # sanity check: scene-derived zone matches the recorded reward-zone occupancy
    zone_mismatch = 0
    for i, (s, e) in enumerate(zip(starts, stops)):
        inz = np.where(rz_series[s:e] > 0)[0]
        if len(inz):
            zstart = pos[s + inz].min()
            data_lab = min(REWARD_ZONES, key=lambda k: abs(REWARD_ZONES[k][0] - zstart))
            zone_mismatch += int(data_lab != zone_labels[i])

    # ----- build trials
    neural, inputs, outputs = [], [], []
    kept_trials = []
    n_drop_first = 0
    n_drop_lick = 0
    n_drop_short = 0
    n_drop_nan = 0
    n_drop_scan = 0
    for i, (s, e) in enumerate(zip(starts, stops)):
        if i == 0:
            n_drop_first += 1                      # previous-trial outcome undefined
            continue
        if lick_error[i]:
            n_drop_lick += 1                       # lick-sensor error (paper: NaN'd)
            continue
        if (e - s) < MIN_TRIAL_FRAMES:
            n_drop_short += 1
            continue
        if np.any(scanning[s:e] != 1):
            n_drop_scan += 1                       # no 2P scanning -> invalid fluorescence
            continue
        ev = events[:, s:e]
        p = pos[s:e]
        sp = speed[s:e]
        lk = lick[s:e]
        tt = t[s:e] - t[s]
        if (np.any(~np.isfinite(ev)) or np.any(~np.isfinite(p)) or
                np.any(~np.isfinite(sp)) or np.any(~np.isfinite(lk))):
            n_drop_nan += 1
            continue

        zlab = zone_labels[i]
        z0, z1 = REWARD_ZONES[zlab]
        d = signed_distance_to_zone(p, z0, z1)

        T = e - s
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = tt
        inp[1] = env_trial[i]
        inp[2] = i
        inp[3] = rewarded[i - 1]

        out = np.empty((6, T), dtype=np.int64)
        out[0] = discretize_distance(d)
        out[1] = np.digitize(p, POS_EDGES)
        out[2] = np.digitize(sp, SPEED_EDGES)
        out[3] = (lk > 0).astype(np.int64)
        out[4] = ZONE_TO_IDX[zlab]
        out[5] = rewarded[i]

        neural.append(np.ascontiguousarray(ev, dtype=np.float32))
        inputs.append(inp)
        outputs.append(out)
        kept_trials.append(i)

    result = dict(
        file=path, subject=subject, session_id=session_id, animal_orig=animal_orig,
        date=date, scene=scene, region=region, neural=neural, input=inputs, output=outputs,
        plane_of_cell=plane_of_cell, n_cells=n_cells, n_rois_iscell=n_rois_iscell,
        n_interneurons=n_interneurons, n_trials_raw=n_trials_raw,
        kept_trials=np.array(kept_trials), rewarded=rewarded, env_trial=env_trial,
        zone_labels=zone_labels, zone_mismatch=zone_mismatch,
        drops=dict(first=n_drop_first, lick=n_drop_lick, short=n_drop_short,
                   nan=n_drop_nan, scanning=n_drop_scan),
        timing=timing, n_frames_trunc=n_frames_trunc,
    )
    result['timing']['total'] = time.time() - t_start

    if show_processing:
        try:
            make_processing_plot(result, dff_keep, events, F, Fneu, pos, speed, lick,
                                 t, starts, stops, zone_labels, plot_dir)
        except Exception as exc:                    # plotting must never break conversion
            warnings.warn(f'plotting failed for {path}: {exc}')
    return result


def make_processing_plot(res, dff, events, F, Fneu, pos, speed, lick, t,
                         starts, stops, zone_labels, plot_dir):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sess_tag = f"{res['subject']}_ses-{res['session_id']}"
    fig, axes = plt.subplots(8, 1, figsize=(16, 20), sharex=False)

    # show 4 trials worth of frames
    i0 = 2
    s0, s1 = starts[i0], stops[i0 + 3]
    fr = np.arange(s0, s1)
    tt = t[fr] - t[s0]

    ax = axes[0]
    for c in range(min(3, F.shape[0])):
        ax.plot(tt, F[c, fr], lw=0.6, label=f'F cell{c}')
        ax.plot(tt, Fneu[c, fr], lw=0.4, alpha=0.5, label=f'Fneu cell{c}')
    ax.set_ylabel('raw F')
    ax.legend(fontsize=6, ncol=3)
    ax.set_title(f'{sess_tag} ({res["scene"]}): raw suite2p fluorescence (4 trials)')

    ax = axes[1]
    for c in range(min(3, dff.shape[0])):
        ax.plot(tt, dff[c, fr], lw=0.7, label=f'dF/F cell{c}')
    ax.set_ylabel('dF/F')
    ax.legend(fontsize=6, ncol=3)

    ax = axes[2]
    for c in range(min(3, events.shape[0])):
        ax.plot(tt, events[c, fr], lw=0.7, label=f'events cell{c}')
    ax.set_ylabel('OASIS events')
    ax.legend(fontsize=6, ncol=3)

    ax = axes[3]
    ax.plot(tt, pos[fr], 'k', lw=0.8)
    for i in range(i0, i0 + 4):
        z0, z1 = REWARD_ZONES[zone_labels[i]]
        ax.axhline(z0, color='g', ls=':', lw=0.6)
        ax.axhline(z1, color='g', ls=':', lw=0.6)
        ax.axvline(t[starts[i]] - t[s0], color='b', lw=0.8)
        ax.axvline(t[stops[i]] - t[s0], color='r', lw=0.8)
    ax.set_ylabel('position (cm)\nblue=start red=teleport')

    # discretization checks for one trial
    k = 0
    trial_i = res['kept_trials'][k]
    inp = res['input'][k]
    out = res['output'][k]
    s, e = starts[trial_i], stops[trial_i]
    ttr = inp[0]
    p = pos[s:e]
    z0, z1 = REWARD_ZONES[zone_labels[trial_i]]
    d = signed_distance_to_zone(p, z0, z1)

    ax = axes[4]
    ax.plot(ttr, p, 'k', label='position (cm)')
    ax.step(ttr, out[1] * 90 + 45, 'r', where='mid', label='position bin (x90+45)')
    ax.axhline(z0, color='g', ls=':'); ax.axhline(z1, color='g', ls=':')
    ax.set_ylabel(f'trial {trial_i}\nposition')
    ax.legend(fontsize=6)

    ax = axes[5]
    ax.plot(ttr, d, 'k', label='signed distance to zone (cm)')
    ax2 = ax.twinx()
    ax2.step(ttr, out[0], 'r', where='mid', label='distance bin')
    ax2.set_ylabel('bin', color='r')
    ax.axhline(0, color='g', ls=':')
    ax.set_ylabel('distance (cm)')
    ax.legend(fontsize=6, loc='upper left')

    ax = axes[6]
    ax.plot(ttr, speed[s:e], 'k', label='speed (cm/s)')
    ax2 = ax.twinx()
    ax2.step(ttr, out[2], 'r', where='mid', label='speed bin')
    ax.set_ylabel('speed (cm/s)')
    ax.legend(fontsize=6, loc='upper left')

    ax = axes[7]
    ax.plot(ttr, lick[s:e], 'k', label='cumulative lick count')
    ax.step(ttr, out[3], 'r', where='mid', label='lick bin')
    ax.set_xlabel('time from trial start (s)')
    ax.set_ylabel('licks')
    ax.legend(fontsize=6)
    ax.set_title(f'inputs: env={inp[1,0]:.0f} trial={inp[2,0]:.0f} prev_outcome={inp[3,0]:.0f}; '
                 f'outputs: zone={out[4,0]} rewarded={out[5,0]}')

    fig.tight_layout()
    outpath = os.path.join(plot_dir, f'processing_{sess_tag}.png')
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f'  wrote {outpath}', flush=True)


def _worker(args):
    path, show, plot_dir, signal = args
    return process_session(path, show_processing=show, plot_dir=plot_dir, signal=signal)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--nworkers', type=int, default=8)
    ap.add_argument('--datadir', default='/app/data')
    ap.add_argument('--signal', choices=['events', 'dff'], default='dff',
                    help="neural signal: OASIS-deconvolved events (paper's decoder input) or dF/F")
    ap.add_argument('--nsessions', type=int, default=None, help='limit number of sessions (testing)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.datadir, 'sub-*', '*.nwb')))
    if args.nsessions:
        files = files[::max(1, len(files)//args.nsessions)][:args.nsessions]
    if args.sample:
        files = [f for f in files if 'sub-m3_ses-03' in f or 'sub-m15_ses-08' in f]
        if len(files) < 2:
            files = sorted(glob.glob(os.path.join(args.datadir, 'sub-*', '*.nwb')))[:2]
    signal_label = args.signal
    print(f'Converting {len(files)} sessions with {args.nworkers} workers (signal={signal_label})', flush=True)

    plot_files = set(files[:2]) if args.show_processing else set()
    tasks = [(f, f in plot_files, os.path.dirname(os.path.abspath(args.outfile)) or '.', args.signal)
             for f in files]

    t0 = time.time()
    results = []
    if args.nworkers > 1 and len(files) > 1:
        ctx = mp.get_context('spawn')  # fork is unsafe with OpenMP (suite2p/numba)
        with ProcessPoolExecutor(max_workers=args.nworkers, mp_context=ctx) as pool:
            for i, res in enumerate(pool.map(_worker, tasks)):
                results.append(res)
                print(f"[{i+1}/{len(files)}] {os.path.basename(res['file'])} "
                      f"cells={res['n_cells']} (iscell {res['n_rois_iscell']}, "
                      f"interneurons {res['n_interneurons']}) "
                      f"trials={len(res['neural'])}/{res['n_trials_raw']} "
                      f"drops={res['drops']} zone_mismatch={res['zone_mismatch']} "
                      f"t={res['timing']['total']:.1f}s", flush=True)
    else:
        for i, task in enumerate(tasks):
            res = _worker(task)
            results.append(res)
            print(f"[{i+1}/{len(files)}] {os.path.basename(res['file'])} "
                  f"cells={res['n_cells']} (iscell {res['n_rois_iscell']}, "
                  f"interneurons {res['n_interneurons']}) "
                  f"trials={len(res['neural'])}/{res['n_trials_raw']} "
                  f"drops={res['drops']} zone_mismatch={res['zone_mismatch']} "
                  f"timing={ {k: round(v,2) for k,v in res['timing'].items()} }", flush=True)
    elapsed = time.time() - t0
    print(f'Session processing took {elapsed:.1f}s ({elapsed/len(files):.1f}s/session)', flush=True)

    # ------------------------------------------------------------------ assemble output
    results = [r for r in results if len(r['neural']) >= 2]
    subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
    subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
    brain_regions = ['CA1']

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': [np.zeros(r['n_cells'], dtype=np.int64) for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice run laps on a 450 cm virtual linear track with a hidden '
                '50 cm reward zone (A 80-130, B 200-250, C 320-370 cm) that switches '
                'location after 30 trials on switch days, across two visually distinct '
                'environments (ENV1/ENV2). Reward is delivered operantly for licking in '
                'the zone and randomly omitted on ~15% of trials. Decoder predicts, from '
                'CA1 two-photon calcium activity: signed distance to the reward zone, '
                'absolute track position, running speed, licking, the active reward zone '
                'location and the trial reward outcome.'),
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'trial start (entry to the linear track at 0 cm)',
            'off_start': 0.0,
            'off_end': None,
            'off_end_note': ('trials end at the teleport frame (entry to the intertrial '
                             'interval); trial durations vary (mean ~11.5 s)'),
            'neural_signal_type': signal_label,
            'neural_signal': NEURAL_SIGNAL_DESC[signal_label],
            'sampling_rate_hz': FRAME_RATE,
            'neuron_curation': ('suite2p iscell==1 (manually curated) minus putative '
                                'interneurons with Pearson r(dF/F, speed) > 0.5'),
            'trial_curation': ('on-track laps only (trial_start to teleport); first trial of '
                               'each session dropped (previous-trial outcome undefined); '
                               'lick-sensor-error trials dropped (>30% of frames with '
                               'cumulative lick count > 2); trials with <10 frames, '
                               'non-finite samples, or frames without 2P scanning dropped'),
            'source': ('DANDI 001361; Sosa, Plitt & Giocomo 2025, Nature Neuroscience, '
                       '"A flexible hippocampal population code for experience relative to reward"'),
            'session_info': [
                dict(subject=r['subject'], session_id=r['session_id'], date=r['date'],
                     scene=r['scene'], animal_orig=r['animal_orig'],
                     n_cells=int(r['n_cells']), n_trials=len(r['neural']),
                     n_trials_raw=int(r['n_trials_raw']),
                     planes=sorted(set(int(p) for p in r['plane_of_cell'])),
                     brain_region=r['region'])
                for r in results
            ],
        },
    }

    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_gb = os.path.getsize(args.outfile) / 1e9

    # ------------------------------------------------------------------------ summary
    n_trials = sum(len(s) for s in data['neural'])
    n_cells = [r['n_cells'] for r in results]
    n_time = sum(tr.shape[1] for s in data['neural'] for tr in s)
    print('\n==== CONVERSION SUMMARY ====')
    print(f'sessions: {len(results)}   subjects: {len(subjects)} {subjects}')
    print(f'trials: {n_trials} (raw trial_start events: {sum(r["n_trials_raw"] for r in results)})')
    print(f'trials/session: mean {n_trials/len(results):.1f}')
    print(f'neurons: total {sum(n_cells)}, per session min {min(n_cells)} max {max(n_cells)} '
          f'mean {np.mean(n_cells):.1f}')
    print(f'iscell ROIs: {sum(r["n_rois_iscell"] for r in results)}, '
          f'excluded interneurons: {sum(r["n_interneurons"] for r in results)} '
          f'({100*sum(r["n_interneurons"] for r in results)/sum(r["n_rois_iscell"] for r in results):.2f}%)')
    print(f'total timepoints: {n_time} ({n_time*TIME_BIN_MS/1000/3600:.2f} h of recording)')
    drops = {}
    for r in results:
        for k, v in r['drops'].items():
            drops[k] = drops.get(k, 0) + v
    print(f'dropped trials: {drops}')
    print(f'zone-label mismatches vs data: {sum(r["zone_mismatch"] for r in results)}')
    rew = np.concatenate([r["rewarded"][r["kept_trials"]] for r in results])
    print(f'rewarded fraction (kept trials): {rew.mean():.4f}')
    envs = np.concatenate([r["env_trial"][r["kept_trials"]] for r in results])
    print(f'ENV1/ENV2 trial fractions: {(envs==0).mean():.3f}/{(envs==1).mean():.3f}')
    outs = np.concatenate([o for s in data['output'] for o in s], axis=1)
    for i, name in enumerate(OUTPUT_NAMES):
        vals, counts = np.unique(outs[i], return_counts=True)
        frac = counts / counts.sum()
        print(f'  output {name}: ' + ', '.join(f'{int(v)}:{f:.3f}' for v, f in zip(vals, frac)))
    ins = np.concatenate([o for s in data['input'] for o in s], axis=1)
    for i, name in enumerate(INPUT_NAMES):
        print(f'  input {name}: min {ins[i].min():.3f} max {ins[i].max():.3f} mean {ins[i].mean():.3f}')
    print(f'wrote {args.outfile} ({size_gb:.2f} GB) in {time.time()-t0:.1f}s total')


if __name__ == '__main__':
    main()
