"""
Convert the Sosa, Plitt & Giocomo (2025) hippocampal CA1 2P dataset (DANDI 001361)
into the decoder-compatible pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference code in /app/code/src/reward_relative:
  * neural: dF/F computed as in `preprocessing.dff` (neuropil subtraction with
    coefficient 0.7, per-trial maximin baseline, 2-frame Gaussian smoothing),
    then OASIS deconvolution (suite2p `dcnv.oasis`, tau=0.7) -> "events".
  * trials: window [trial_start-1, teleport-1) as in the reference code.
  * trial variables: `behavior.get_trial_types` (isreward, morph) and
    `behavior.get_reward_zones` (reward zone coords/labels, switch at trial 30).
"""
import argparse, os, pickle, re, sys, time
import numpy as np
import h5py
from scipy import ndimage
from suite2p.extraction import dcnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

DATA_ROOT = '/app/data'

# ---------------------------------------------------------------- constants --
# reward zone coordinates (cm), from reward_relative/behavior.py reward_zone_dict
# (keys 'X','Y','Z' correspond to labels A, B, C)
REWARD_ZONES = {'A': (80., 130.), 'B': (200., 250.), 'C': (320., 370.)}
ZONE_TO_IDX = {'A': 0, 'B': 1, 'C': 2}
CHANGE_TRIAL = 30            # behavior.get_reward_zones(change_trial=30)
NEU_COEF = 0.7               # preprocessing.dff neu_coef
TAU = 0.7                    # suite2p ops['tau'] for GCaMP7f in the repo
BASELINE_WIN = 300           # samples for maximin min/max filters (~20 s @15.5Hz)
BASELINE_SIGMA = 15          # frames, nansmooth sigma before the min filter
DFF_SMOOTH = 2               # frames, dF/F smoothing (paper: 2-sample s.d.)
INT_R_THRESH = 0.5           # putative interneuron speed-correlation threshold
LICK_ERR_FRAC = 0.30         # >30% of samples with cumulative lick > 2 -> error
TRACK_LENGTH = 450.

OUTPUT_NAMES = ['reward_zone_distance', 'position', 'speed', 'lick',
                'reward_zone_location', 'reward_outcome']
OUTPUT_VALUES = [
    ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in reward zone (0 cm)',
     '0 to 10 cm', '10 to 50 cm', '> 50 cm'],
    ['< 90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '> 360 cm'],
    ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
    ['no lick', 'lick'],
    ['A (80-130 cm)', 'B (200-250 cm)', 'C (320-370 cm)'],
    ['omitted', 'rewarded'],
]
INPUT_NAMES = ['time_from_trial_start', 'environment', 'trial_number',
               'previous_trial_outcome']


# ---------------------------------------------------------------- utilities --
def nansmooth(a, sig, axis=-1):
    """Gaussian smoothing ignoring NaNs (copy of reward_relative.utilities.nansmooth)."""
    nan_inds = np.isnan(a)
    a_nanless = np.copy(a)
    a_nanless[nan_inds] = 0
    one = np.ones(a.shape)
    one[nan_inds] = 0.001
    a_nanless = ndimage.gaussian_filter1d(a_nanless, sig, axis=axis)
    one = ndimage.gaussian_filter1d(one, sig, axis=axis)
    return a_nanless / one


def scene_zone_labels(scene, ntrials, change_trial=CHANGE_TRIAL):
    """Per-trial reward zone label from the scene name (behavior.get_reward_zones)."""
    m = re.match(r'Env\d_Location([ABC])$', scene)
    if m:
        return np.array([m.group(1)] * ntrials)
    m = re.match(r'Env\d_Location([ABC])_to_([ABC])$', scene)
    if m is None:
        m = re.match(r'Env\d_([ABC])_to_Env\d_([ABC])$', scene)
    if m:
        z0, z1 = m.group(1), m.group(2)
        n0 = min(change_trial, ntrials)
        return np.array([z0] * n0 + [z1] * (ntrials - n0))
    raise ValueError(f'unhandled scene name: {scene}')


def signed_distance_to_zone(pos, zone_start, zone_end):
    """Signed distance (cm) to the nearest point of the reward zone.
    Negative before the zone, 0 inside, positive after."""
    d = np.zeros_like(pos)
    before = pos < zone_start
    after = pos > zone_end
    d[before] = pos[before] - zone_start
    d[after] = pos[after] - zone_end
    return d


def discretize_distance(d):
    """0: <-50, 1: [-50,-10), 2: [-10,0), 3: 0, 4: (0,10], 5: (10,50], 6: >50"""
    out = np.full(d.shape, 3, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out


def discretize_position(pos):
    """5 equal bins over the 450 cm track."""
    return np.clip((pos // 90).astype(np.int64), 0, 4)


def discretize_speed(speed):
    """0: <2, 1: [2,10), 2: [10,20), 3: [20,40), 4: >=40 cm/s"""
    return np.digitize(speed, [2., 10., 20., 40.]).astype(np.int64)


# ------------------------------------------------------------ dF/F + events --
def compute_events(F, Fneu, starts, stops, frame_rate):
    """dF/F and OASIS-deconvolved events, following preprocessing.dff().

    F, Fneu: (n_cells, n_samples) raw suite2p traces
    starts, stops: trial_start / teleport sample indices
    Returns dff, events (both (n_cells, n_samples), NaN outside trials).
    """
    f_ = np.full(F.shape, np.nan, dtype=np.float64)
    fneu_ = np.full(F.shape, np.nan, dtype=np.float64)
    for start, stop in zip(starts, stops):
        f_[:, start - 1:stop - 1] = F[:, start - 1:stop - 1]
        fneu_[:, start - 1:stop - 1] = Fneu[:, start - 1:stop - 1]

    nanmask = ~np.isnan(f_[0, :])

    # neuropil subtraction
    f_ -= NEU_COEF * fneu_

    flow = np.full(F.shape, np.nan)
    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        # add back the per-trial neuropil mean (reference code)
        f_[:, sl] = f_[:, sl] + NEU_COEF * np.nanmean(fneu_[:, sl], axis=1, keepdims=True)
        # maximin baseline
        flow[:, sl] = nansmooth(f_[:, sl], BASELINE_SIGMA)
        flow[:, sl] = ndimage.minimum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)
        flow[:, sl] = ndimage.maximum_filter1d(flow[:, sl], BASELINE_WIN, axis=-1)

    dff = np.full(F.shape, np.nan)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    events = np.full(F.shape, np.nan, dtype=np.float32)
    for start, stop in zip(starts, stops):
        sl = slice(start - 1, stop - 1)
        dff[:, sl] = nansmooth(dff[:, sl], DFF_SMOOTH, axis=1)
        events[:, sl] = dcnv.oasis(np.ascontiguousarray(dff[:, sl], dtype=np.float32),
                                   2000, TAU, frame_rate)
    return dff, events


# ------------------------------------------------------------ session loader --
def load_session(path):
    """Read everything needed from one NWB file."""
    with h5py.File(path, 'r') as f:
        ident = f['identifier'][()].decode()
        scene = ident.split('/')[-1]
        subject = f['general/subject/subject_id'][()].decode()
        day = f['general/session_id'][()].decode()
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = seg['iscell'][:, 0] > 0
        plane_idx = seg['planeIdx'][:]
        fl = f['processing/ophys/Fluorescence']
        planes = sorted(fl.keys())
        Fl, Fnl = [], []
        for p in planes:
            sel = iscell[plane_idx == int(p.replace('plane', ''))]
            Fl.append(f[f'processing/ophys/Fluorescence/{p}/data'][:, :].T[sel])
            Fnl.append(f[f'processing/ophys/Neuropil/{p}/data'][:, :].T[sel])
        F = np.concatenate(Fl, axis=0)
        Fneu = np.concatenate(Fnl, axis=0)
        b = f['processing/behavior/BehavioralTimeSeries']
        g = lambda k: b[f'{k}/data'][:]
        beh = dict(pos=g('position'), speed=g('speed'), lick=g('lick'),
                   rzone=g('reward_zone'), env=g('environment'),
                   autoreward=g('autoreward'), scanning=g('scanning'),
                   trialnum=g('trial number'))
        time = b['position/timestamps'][:]
        starts = np.where(g('trial_start') > 0)[0]
        stops = np.where(g('teleport') > 0)[0]
        reward_t = b['Reward/timestamps'][:]
        # frame rate: use the median sampling interval of the behavior timestamps,
        # which equals the per-plane imaging rate (the `rate` attribute of the
        # ophys series is the raw scan rate, i.e. 2x for the 2-plane mice)
        frame_rate = 1.0 / np.median(np.diff(time))
    # A few 2-plane sessions have one more imaging frame than behavior samples
    # (the 'one frame correction' noted in TwoPUtils/reward_relative preprocessing).
    # Truncate every stream to the common length so neural and behavior stay aligned.
    n = min([F.shape[1], Fneu.shape[1], len(time)] + [len(v) for v in beh.values()])
    if F.shape[1] != n or len(time) != n:
        print(f'  {os.path.basename(path)}: truncating streams to {n} samples '
              f'(ophys {F.shape[1]}, behavior {len(time)})', flush=True)
    F = F[:, :n]; Fneu = Fneu[:, :n]; time = time[:n]
    beh = {k: v[:n] for k, v in beh.items()}
    starts = starts[starts < n]
    stops = stops[stops <= n]
    ntr = min(len(starts), len(stops))
    starts, stops = starts[:ntr], stops[:ntr]

    reward = np.zeros_like(beh['pos'])
    if len(reward_t):
        idx = np.clip(np.searchsorted(time, reward_t), 0, len(reward) - 1)
        reward[idx] = 1
    beh['reward'] = reward
    return dict(scene=scene, subject=subject, day=day, F=F, Fneu=Fneu, beh=beh,
                time=time, starts=starts, stops=stops, frame_rate=frame_rate,
                ident=ident, path=path)


def process_session(path, show_processing=False, plot_dir='/app', signal='events'):
    """Convert one NWB session into per-trial neural / input / output arrays."""
    t0 = time.time()
    S = load_session(path)
    beh, starts, stops = S['beh'], S['starts'], S['stops']
    ntrials_raw = len(starts)
    t_load = time.time() - t0

    # ---- neural: dF/F + deconvolved events (reference preprocessing.dff) ----
    t0 = time.time()
    dff, events = compute_events(S['F'], S['Fneu'], starts, stops, S['frame_rate'])
    t_dff = time.time() - t0

    # ---- neuron curation: putative interneurons (dF/F vs speed r > 0.5) ----
    nanmask = ~np.isnan(dff[0, :])
    sp = beh['speed'][nanmask]
    D = dff[:, nanmask]
    Dz = D - D.mean(axis=1, keepdims=True)
    spz = sp - sp.mean()
    denom = (np.sqrt((Dz ** 2).sum(axis=1)) * np.sqrt((spz ** 2).sum()))
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_corr = (Dz @ spz) / denom
    is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
    keep_cells = ~is_int
    events = events[keep_cells]
    dff_kept = dff[keep_cells]
    if signal == 'dff':
        # option used only for the control analysis in Step 12
        events = dff_kept.astype(np.float32)

    # ---- per-trial variables (behavior.get_trial_types / get_reward_zones) ----
    zone_labels = scene_zone_labels(S['scene'], ntrials_raw)
    isreward = np.zeros(ntrials_raw, dtype=np.int64)
    morph = np.zeros(ntrials_raw, dtype=np.int64)
    lick_error = np.zeros(ntrials_raw, dtype=bool)
    for i, (s, e) in enumerate(zip(starts, stops)):
        sl = slice(s, e)   # get_trial_types uses [start:stop]
        isreward[i] = int(np.any(beh['reward'][sl] > 0) and np.any(beh['rzone'][sl] > 0))
        u = np.unique(beh['env'][sl])
        morph[i] = int(u[0])
        # lick sensor error (paper: >30% of samples in the trial with cum lick > 2)
        lick_tr = beh['lick'][s - 1:e - 1]
        lick_error[i] = (np.sum(lick_tr > 2) / max(1, len(lick_tr))) > LICK_ERR_FRAC

    # ---- assemble trials ----
    neural, inputs, outputs = [], [], []
    kept_trials = []
    drop = dict(lick_error=0, too_short=0, nan_events=0, pre_sync=0)
    for i, (s, e) in enumerate(zip(starts, stops)):
        if lick_error[i]:
            drop['lick_error'] += 1       # lick is an output; cannot be NaN
            continue
        sl = slice(s - 1, e - 1)          # reference trial window
        ev = events[:, sl]
        if ev.shape[1] < 2:
            drop['too_short'] += 1
            continue
        if np.any(np.isnan(ev)):
            drop['nan_events'] += 1
            continue
        pos = beh['pos'][sl]
        if np.any(pos < -100):            # pre-TTL-sync samples (pos = -500)
            drop['pre_sync'] += 1
            continue
        tt = S['time'][sl] - S['time'][s - 1]
        zone = zone_labels[i]
        z0, z1 = REWARD_ZONES[zone]
        dist = signed_distance_to_zone(pos, z0, z1)
        speed = beh['speed'][sl]
        lick = (beh['lick'][sl] > 0).astype(np.int64)
        T = ev.shape[1]
        inp = np.stack([tt,
                        np.full(T, float(morph[i])),
                        np.full(T, float(i)),
                        np.full(T, float(isreward[i - 1]) if i > 0 else 1.0)]).astype(np.float32)
        out = np.stack([discretize_distance(dist),
                        discretize_position(pos),
                        discretize_speed(speed),
                        lick,
                        np.full(T, ZONE_TO_IDX[zone], dtype=np.int64),
                        np.full(T, isreward[i], dtype=np.int64)]).astype(np.int64)
        neural.append(ev.astype(np.float32))
        inputs.append(inp)
        outputs.append(out)
        kept_trials.append(i)

    info = dict(subject=S['subject'], day=S['day'], scene=S['scene'],
                ntrials_raw=ntrials_raw, ntrials=len(neural),
                nneurons=int(events.shape[0]),
                nneurons_iscell=int(S['F'].shape[0]),
                n_interneurons=int(is_int.sum()),
                n_lick_error=int(lick_error.sum()),
                drop_reasons=drop,
                frame_rate=S['frame_rate'],
                rewarded_frac=float(np.mean(isreward[kept_trials])) if kept_trials else np.nan,
                t_load=t_load, t_dff=t_dff)

    if show_processing:
        make_processing_plot(S, dff_kept, events, beh, starts, stops, kept_trials,
                            zone_labels, isreward, morph, neural, inputs, outputs,
                            plot_dir)
    return neural, inputs, outputs, info


# ------------------------------------------------------------------- plots ---
def make_processing_plot(S, dff, events, beh, starts, stops, kept_trials,
                         zone_labels, isreward, morph, neural, inputs, outputs,
                         plot_dir):
    """Visualise every processing step for one session."""
    sid = f"{S['subject']}_ses-{S['day']}"
    fig, axes = plt.subplots(7, 1, figsize=(16, 20), sharex=False)
    t = S['time']
    n0, n1 = starts[0] - 1, stops[4] - 1          # first 5 trials
    tt = t[n0:n1]
    ax = axes[0]
    ax.plot(tt, beh['pos'][n0:n1], 'k')
    for i in range(5):
        ax.axvline(t[starts[i] - 1], color='g')
        ax.axvline(t[stops[i] - 1], color='r')
        z0, z1 = REWARD_ZONES[zone_labels[i]]
        ax.hlines([z0, z1], t[starts[i] - 1], t[stops[i] - 1], color='b', ls='--')
    ax.set_ylabel('position (cm)')
    ax.set_title(f'{sid} {S["scene"]}: green=trial start, red=teleport, blue=reward zone')

    ax = axes[1]
    for c in range(min(5, dff.shape[0])):
        ax.plot(tt, dff[c, n0:n1] + c * 2, lw=.5)
    ax.set_ylabel('dF/F (5 cells)')

    ax = axes[2]
    for c in range(min(5, events.shape[0])):
        ax.plot(tt, events[c, n0:n1] + c * 1, lw=.5)
    ax.set_ylabel('deconvolved events')

    # trial-aligned check: outputs of the first kept trial
    k = 0
    tr = kept_trials[k]
    s, e = starts[tr] - 1, stops[tr] - 1
    pos = beh['pos'][s:e]
    z0, z1 = REWARD_ZONES[zone_labels[tr]]
    dist = signed_distance_to_zone(pos, z0, z1)
    tt2 = inputs[k][0]
    ax = axes[3]
    ax.plot(tt2, dist, 'k', label='signed distance to zone (cm)')
    ax2 = ax.twinx()
    ax2.step(tt2, outputs[k][0], 'r', where='post', label='binned')
    ax.axhline(0, color='b', ls=':')
    ax.set_ylabel('distance to RZ (cm)'); ax2.set_ylabel('bin')
    ax.legend(loc='upper left'); ax2.legend(loc='lower right')
    ax.set_title('Trial %d: distance to reward zone and its discretization' % tr)

    ax = axes[4]
    ax.plot(tt2, pos, 'k')
    ax2 = ax.twinx(); ax2.step(tt2, outputs[k][1], 'r', where='post')
    ax.set_ylabel('position (cm)'); ax2.set_ylabel('position bin')

    ax = axes[5]
    ax.plot(tt2, beh['speed'][s:e], 'k')
    ax2 = ax.twinx(); ax2.step(tt2, outputs[k][2], 'r', where='post')
    ax.set_ylabel('speed (cm/s)'); ax2.set_ylabel('speed bin')

    ax = axes[6]
    ax.plot(tt2, beh['lick'][s:e], 'k', label='cum lick count')
    ax.step(tt2, outputs[k][3], 'r', where='post', label='lick output')
    ax.set_xlabel('time from trial start (s)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'processing_{sid}.png'), dpi=110)
    plt.close(fig)

    # second figure: trial raster of outputs across the session
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    maxT = max(o.shape[1] for o in outputs)
    for j, name in enumerate(['reward_zone_distance', 'position', 'speed', 'lick']):
        M = np.full((len(outputs), maxT), np.nan)
        for i, o in enumerate(outputs):
            M[i, :o.shape[1]] = o[j]
        im = axes[j].imshow(M, aspect='auto', interpolation='none')
        axes[j].set_title(name); axes[j].set_xlabel('time bin'); axes[j].set_ylabel('trial')
        plt.colorbar(im, ax=axes[j])
    fig.suptitle(f'{sid}: outputs across trials (aligned to trial start)')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'processing_{sid}_outputs.png'), dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------- main ---
def session_files():
    import glob
    return sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', 'sub-*.nwb')))


def _worker(args):
    path, show, plot_dir, signal = args
    return process_session(path, show_processing=show, plot_dir=plot_dir, signal=signal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--signal', choices=['events', 'dff'], default='dff',
                    help="neural signal: dF/F (default; both are computed with the paper's "
                         'preprocessing.dff pipeline, dF/F decodes better - see CONVERSION_NOTES Step 12) '
                         'or the OASIS-deconvolved events used for the spatial analyses in the paper')
    ap.add_argument('--nsessions', type=int, default=None,
                    help='process only the first N sessions (control analyses)')
    args = ap.parse_args()

    files = session_files()
    if args.sample:
        files = [f for f in files if 'sub-m11_ses-03' in f or 'sub-m17_ses-08' in f]
    if args.nsessions:
        files = files[:args.nsessions]
    print(f'Processing {len(files)} sessions', flush=True)

    show = args.show_processing
    t_start = time.time()
    results = []
    if args.workers > 1 and not show:
        ctx = mp.get_context('spawn')  # suite2p uses OpenMP; fork is unsafe
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            for i, r in enumerate(ex.map(_worker, [(f, False, '/app', args.signal) for f in files])):
                results.append(r)
                info = r[3]
                print(f"[{i+1}/{len(files)}] {info['subject']} day {info['day']} "
                      f"{info['scene']}: {info['ntrials']}/{info['ntrials_raw']} trials, "
                      f"{info['nneurons']} neurons ({info['n_interneurons']} interneurons dropped), "
                      f"drops={info['drop_reasons']}, "
                      f"elapsed {time.time()-t_start:.1f}s", flush=True)
    else:
        for i, f in enumerate(files):
            r = process_session(f, show_processing=show, plot_dir='/app', signal=args.signal)
            results.append(r)
            info = r[3]
            print(f"[{i+1}/{len(files)}] {info['subject']} day {info['day']} "
                  f"{info['scene']}: {info['ntrials']}/{info['ntrials_raw']} trials, "
                  f"{info['nneurons']} neurons ({info['n_interneurons']} interneurons dropped), "
                  f"load {info['t_load']:.1f}s dff {info['t_dff']:.1f}s, "
                  f"elapsed {time.time()-t_start:.1f}s", flush=True)

    neural = [r[0] for r in results]
    inputs = [r[1] for r in results]
    outputs = [r[2] for r in results]
    infos = [r[3] for r in results]

    subjects = sorted({i['subject'] for i in infos}, key=lambda x: int(x[1:]))
    subject_idx = np.array([subjects.index(i['subject']) for i in infos], dtype=np.int64)
    brain_region_idx = [np.zeros(i['nneurons'], dtype=np.int64) for i in infos]

    frame_rates = np.array([i['frame_rate'] for i in infos])
    bin_ms = float(1000.0 / np.median(frame_rates))

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'Head-fixed mice run laps on a 450 cm virtual linear track with a hidden '
                '50 cm reward zone at one of three locations (A 80-130, B 200-250, C 320-370 cm) '
                'in one of two visually distinct environments (ENV1/ENV2). The reward zone is '
                'moved to a new location after 30 trials on switch days; reward is randomly '
                'omitted on ~15% of trials. Decoded outputs: distance to the reward zone, '
                'absolute track position, running speed, licking, reward zone identity and '
                'trial reward outcome. Neural data: deconvolved calcium events from '
                'hippocampal CA1 pyramidal cells (2-photon imaging, GCaMP7f).'),
            'time_bin_size': bin_ms,
            'temporal_alignment_event': 'start of trial (entry to the linear track at position 0 cm)',
            'off_start': 0.0,
            'off_end': None,
            'off_end_note': ('trials have variable duration; each trial runs from trial start '
                             'to the teleport (end of track), median 12.3 s'),
            'neural_signal_type': args.signal,
            'neural_signal': ('OASIS-deconvolved calcium events (tau=0.7 s) computed from dF/F '
                              '(neuropil coefficient 0.7, per-trial maximin baseline with 20 s '
                              'window, 2-frame Gaussian smoothing), as in Sosa et al. '
                              'preprocessing.dff()'),
            'neuron_curation': ('suite2p manual curation (iscell==1) and exclusion of putative '
                                'interneurons (Pearson r > 0.5 between dF/F and running speed)'),
            'trial_curation': ('trials with lick-sensor errors (>30% of samples with cumulative '
                               'lick count > 2) excluded, as in the paper'),
            'source': 'DANDI 001361; Sosa, Plitt & Giocomo 2025, Nature Neuroscience',
            'session_info': [
                {k: v for k, v in i.items() if k not in ('t_load', 't_dff')} for i in infos],
        },
    }

    ntr = sum(len(n) for n in neural)
    nneu = sum(i['nneurons'] for i in infos)
    print(f'\nTotal: {len(neural)} sessions, {ntr} trials, {nneu} neurons, '
          f'{len(subjects)} subjects', flush=True)
    print(f'Time bin: {bin_ms:.3f} ms', flush=True)
    t0 = time.time()
    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile)/1e9:.2f} GB) in '
          f'{time.time()-t0:.1f}s; total {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
