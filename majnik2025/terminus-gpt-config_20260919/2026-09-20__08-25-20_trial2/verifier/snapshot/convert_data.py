#!/usr/bin/env python3
"""Convert the Majnik et al. Track2p dataset to decoder-compatible trials.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d

DATA_ROOT = Path('/app/data')
NATIVE_HZ = 30
AVERAGE_FRAMES = 10
ANALYSIS_HZ = NATIVE_HZ / AVERAGE_FRAMES
TRIAL_SECONDS = 60
TRIAL_T = int(TRIAL_SECONDS * ANALYSIS_HZ)
CHUNK_NEURONS = 64


def discover_sessions():
    """Return dated session paths in subject/date order."""
    return sorted(p.parents[2] for p in DATA_ROOT.glob('*/*/suite2p/plane0/F.npy'))


def repair_motion(session, n_neural):
    """Repair camera samples missing at internal timestamp gaps.

    Acquisition is trigger synchronized.  We therefore retain frame order and only
    insert samples when behavior is shorter than neural data and local timestamp
    gaps account exactly for that deficit. Equal-length streams are left unchanged
    even if their clock timestamps contain isolated pauses.
    """
    motion = np.load(session/'move_deve/motion_energy_glob.npy').astype(np.float64)
    stamps = np.load(session/'move_deve/tstamps.npy').astype(np.float64)
    if motion.ndim != 1 or stamps.ndim != 1 or len(motion) != len(stamps):
        raise ValueError(f'Invalid behavior arrays in {session}')
    if not (np.isfinite(motion).all() and np.isfinite(stamps).all()):
        raise ValueError(f'Non-finite behavior in {session}')
    inserted = 0
    if len(motion) < n_neural:
        dt = np.diff(stamps)
        med = float(np.median(dt))
        steps = np.ones(len(dt), dtype=np.int64)
        large = dt > 1.5 * med
        steps[large] = np.maximum(1, np.rint(dt[large] / med).astype(np.int64))
        positions = np.r_[0, np.cumsum(steps)]
        inserted = int(positions[-1] + 1 - len(motion))
        deficit = n_neural - len(motion)
        if inserted != deficit or positions[-1] != n_neural - 1:
            raise ValueError(f'Behavior gaps do not explain deficit in {session}: '
                             f'deficit={deficit}, inferred={inserted}')
        repaired = np.interp(np.arange(n_neural), positions, motion)
        # Original values must be preserved exactly at their reconstructed indices.
        if not np.allclose(repaired[positions], motion, rtol=0, atol=0):
            raise AssertionError('Behavior repair changed observed samples')
    elif len(motion) == n_neural:
        repaired = motion
    else:
        # Defensive handling; no supplied session takes this branch.
        repaired = motion[:n_neural]
    return repaired.astype(np.float32), motion, stamps, inserted


def baseline_correct_and_bin(session, ops, show_details=False):
    """Reproduce Track2p GUI/Suite2p baseline subtraction, then average 10 frames."""
    F = np.load(session/'suite2p/plane0/F.npy', mmap_mode='r')
    Fneu = np.load(session/'suite2p/plane0/Fneu.npy', mmap_mode='r')
    iscell = np.load(session/'suite2p/plane0/iscell.npy', mmap_mode='r')
    if F.shape != Fneu.shape or iscell.shape[0] != F.shape[0]:
        raise ValueError(f'Neural shape mismatch in {session}')
    if not np.all(iscell[:, 1] > 0.5):
        raise ValueError(f'Distributed rows fail reference iscell criterion in {session}')
    n_neurons, n_frames = F.shape
    if n_frames % AVERAGE_FRAMES:
        raise ValueError(f'Frame count not divisible by {AVERAGE_FRAMES}: {session}')
    fs = float(ops['fs'])
    if not np.isclose(fs, NATIVE_HZ):
        raise ValueError(f'Unexpected sampling rate {fs} in {session}')
    neucoeff = float(ops.get('neucoeff', 0.7))
    baseline = ops.get('baseline', 'maximin')
    sigma = float(ops.get('sig_baseline', 10.0))
    win = int(float(ops.get('win_baseline', 60.0)) * fs)
    if baseline != 'maximin':
        raise ValueError(f'Expected maximin baseline, got {baseline}')

    out = np.empty((n_neurons, n_frames // AVERAGE_FRAMES), dtype=np.float32)
    details = None
    for lo in range(0, n_neurons, CHUNK_NEURONS):
        hi = min(n_neurons, lo + CHUNK_NEURONS)
        fc = np.asarray(F[lo:hi], dtype=np.float32) - neucoeff * np.asarray(Fneu[lo:hi], dtype=np.float32)
        flow = gaussian_filter1d(fc, sigma=sigma, axis=1, mode='reflect')
        flow = minimum_filter1d(flow, size=win, axis=1, mode='reflect')
        flow = maximum_filter1d(flow, size=win, axis=1, mode='reflect')
        corrected = fc - flow
        out[lo:hi] = corrected.reshape(hi-lo, -1, AVERAGE_FRAMES).mean(axis=2)
        if show_details and details is None:
            details = {
                'raw_F': np.asarray(F[lo], dtype=np.float32).copy(),
                'raw_Fneu': np.asarray(Fneu[lo], dtype=np.float32).copy(),
                'Fc': fc[0].copy(), 'Flow': flow[0].copy(),
                'corrected': corrected[0].copy(), 'binned': out[lo].copy()
            }
    if not np.isfinite(out).all():
        raise ValueError(f'Non-finite processed neural data in {session}')
    return out, details


def make_processing_plot(session_id, details, motion_raw, motion_repaired,
                         motion_binned, edges, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nshow = min(1800, len(details['raw_F']))
    x = np.arange(nshow) / NATIVE_HZ
    fig, ax = plt.subplots(5, 1, figsize=(14, 15), constrained_layout=True)
    ax[0].plot(x, details['raw_F'][:nshow], label='F', lw=.7)
    ax[0].plot(x, .7*details['raw_Fneu'][:nshow], label='0.7 Fneu', lw=.7)
    ax[0].plot(x, details['Fc'][:nshow], label='Fc', lw=.8)
    ax[0].set_title('Step 1: raw fluorescence and neuropil subtraction'); ax[0].legend(ncol=3)
    ax[1].plot(x, details['Fc'][:nshow], label='Fc', lw=.7)
    ax[1].plot(x, details['Flow'][:nshow], label='maximin baseline', lw=1)
    ax[1].plot(x, details['corrected'][:nshow], label='Fc - baseline', lw=.7)
    ax[1].set_title('Step 2: reference maximin baseline correction'); ax[1].legend(ncol=3)
    xb = (np.arange(min(180, len(details['binned']))) + .5) / ANALYSIS_HZ
    ax[2].plot(xb, details['binned'][:len(xb)], lw=1)
    ax[2].set_title('Step 3: neural activity after non-overlapping 10-frame means (3 Hz)')
    mr = min(1800, len(motion_repaired))
    ax[3].plot(np.arange(min(mr, len(motion_raw)))/NATIVE_HZ,
               motion_raw[:min(mr, len(motion_raw))], label='observed', alpha=.65)
    ax[3].plot(np.arange(mr)/NATIVE_HZ, motion_repaired[:mr], '--', label='repaired/aligned', lw=.8)
    ax[3].set_title('Step 4: motion frame alignment / missing-frame interpolation'); ax[3].legend()
    nt = min(540, len(motion_binned)); xx=(np.arange(nt)+.5)/ANALYSIS_HZ
    ax[4].plot(xx, motion_binned[:nt], color='k', lw=.8, label='10-frame mean motion')
    for e in edges: ax[4].axhline(e, ls=':', alpha=.6)
    ax2=ax[4].twinx(); ax2.step(xx, labels[:nt], where='mid', color='tab:red', alpha=.55, label='class 0-4')
    ax[4].set_title('Steps 5-7: binned motion, session quantile edges, categorical aligned output')
    ax[4].set_xlabel('seconds from session start'); ax[4].set_ylabel('motion energy'); ax2.set_ylabel('class')
    safe=session_id.replace('/','_')
    fig.suptitle(session_id)
    fig.savefig(f'/app/processing_{safe}.png', dpi=130)
    plt.close(fig)


def convert_session(session, show_processing=False):
    t0 = time.perf_counter()
    sid = str(session.relative_to(DATA_ROOT))
    ops = np.load(session/'suite2p/plane0/ops.npy', allow_pickle=True).item()
    neural_binned, details = baseline_correct_and_bin(session, ops, show_processing)
    n_neurons = neural_binned.shape[0]
    n_native = int(np.load(session/'suite2p/plane0/F.npy', mmap_mode='r').shape[1])
    motion_repaired, motion_raw, stamps, inserted = repair_motion(session, n_native)
    motion_binned = motion_repaired.reshape(-1, AVERAGE_FRAMES).mean(axis=1, dtype=np.float64).astype(np.float32)
    if neural_binned.shape[1] != len(motion_binned):
        raise AssertionError(f'Post-bin alignment mismatch in {sid}')
    edges = np.quantile(motion_binned.astype(np.float64), [.2, .4, .6, .8])
    labels = np.searchsorted(edges, motion_binned, side='right').astype(np.int64)
    n_bins = neural_binned.shape[1]
    n_trials = n_bins // TRIAL_T
    use = n_trials * TRIAL_T
    if use != n_bins:
        print(f'  WARNING dropping {n_bins-use} incomplete analysis bins from {sid}')
    neural_trials=[]; input_trials=[]; output_trials=[]
    elapsed = ((np.arange(n_bins, dtype=np.float64)*AVERAGE_FRAMES + (AVERAGE_FRAMES-1)/2) / NATIVE_HZ).astype(np.float32)
    for tr in range(n_trials):
        sl=slice(tr*TRIAL_T,(tr+1)*TRIAL_T)
        neural_trials.append(np.ascontiguousarray(neural_binned[:,sl], dtype=np.float32))
        input_trials.append(np.ascontiguousarray(elapsed[None,sl], dtype=np.float32))
        output_trials.append(np.ascontiguousarray(labels[None,sl], dtype=np.int64))
    for n,x,y in zip(neural_trials,input_trials,output_trials):
        assert n.shape==(n_neurons,TRIAL_T) and x.shape==(1,TRIAL_T) and y.shape==(1,TRIAL_T)
        assert np.isfinite(n).all() and np.isfinite(x).all() and np.isfinite(y).all()
    if show_processing:
        make_processing_plot(sid, details, motion_raw, motion_repaired, motion_binned, edges, labels)
    counts=np.bincount(labels,minlength=5)
    info={'session_id':sid, 'subject':session.parent.name, 'native_frames':n_native,
          'native_behavior_samples':int(len(motion_raw)), 'inserted_behavior_samples':int(inserted),
          'analysis_bins':int(n_bins), 'n_trials':int(n_trials), 'n_neurons':int(n_neurons),
          'motion_quantile_edges':[float(v) for v in edges],
          'motion_class_counts':[int(v) for v in counts],
          'duration_seconds_nominal':float(n_native/NATIVE_HZ)}
    dt=time.perf_counter()-t0
    print(f'[{sid}] neurons={n_neurons} native={n_native} repaired={inserted} '
          f'trials={n_trials} classes={counts.tolist()} time={dt:.2f}s')
    return neural_trials,input_trials,output_trials,info


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--full',action='store_true',help='process all sessions (default)')
    mode.add_argument('--sample',action='store_true',help='process first 2 sessions')
    ap.add_argument('--show-processing',action='store_true',help='plot every processing step for up to 2 sessions')
    args=ap.parse_args()
    sessions=discover_sessions()
    if args.sample: sessions=sessions[:2]
    print(f'Processing {len(sessions)} sessions; 30 Hz -> 3 Hz; {TRIAL_T} bins/60-s trial')
    subjects=sorted({s.parent.name for s in sessions})
    neural=[]; inputs=[]; outputs=[]; infos=[]; subject_idx=[]; region_idx=[]
    start=time.perf_counter()
    for i,s in enumerate(sessions):
        n,x,y,info=convert_session(s,args.show_processing and i<2)
        neural.append(n); inputs.append(x); outputs.append(y); infos.append(info)
        subject_idx.append(subjects.index(s.parent.name))
        region_idx.append(np.zeros(info['n_neurons'],dtype=np.int64))
    data={'neural':neural,'input':inputs,'output':outputs,
          'subjects':subjects,'subject_idx':np.asarray(subject_idx,dtype=np.int64),
          'brain_regions':['barrel cortex (L2/3)'],'brain_region_idx':region_idx,
          'input_names':['time_from_session_start_seconds'],
          'output_names':['motion_energy_bin'],
          'output_values':[['lowest (0-20%)','low (20-40%)','middle (40-60%)',
                            'high (60-80%)','highest (80-100%)']],
          'metadata':{
              'task_description':'Decode five within-session equal-percentile bins of global mouse motion energy from baseline-corrected calcium activity.',
              'time_bin_size':1000.0/ANALYSIS_HZ,
              'temporal_alignment_event':'session start; trials are consecutive non-overlapping 60-second windows',
              'off_start':0.0,'off_end':60.0,
              'native_sampling_rate_hz':NATIVE_HZ,'analysis_sampling_rate_hz':ANALYSIS_HZ,
              'neural_processing':'F-0.7*Fneu; Suite2p maximin baseline (sigma 10 frames, 60 s min/max window) subtracted; mean of 10 frames',
              'behavior_processing':'timestamp-gap repair when required; mean of 10 frames; session-specific 20/40/60/80 percentiles',
              'trial_duration_seconds':TRIAL_SECONDS,'session_info':infos,
              'source':'Majnik et al. 2025 Track2p longitudinal developing mouse barrel cortex dataset'}}
    out=Path(args.outpicklefile); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    elapsed=time.perf_counter()-start
    print(f'Saved {out} ({out.stat().st_size/1024**2:.1f} MiB) in {elapsed:.2f}s')
    print(f'Summary: subjects={len(subjects)} sessions={len(sessions)} trials={sum(map(len,neural))} '
          f'session-neurons={sum(x[0].shape[0] for x in neural)}')

if __name__=='__main__': main()
