#!/usr/bin/env python3
"""Convert Lee et al. CA1 geometry data to decoder-compatible format.

Usage: python -u /app/convert_data.py OUTFILE [--full|--sample] [--show-processing]
"""
import argparse
import gc
import os
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path('/app/data')
FPS = 30
POOL = 3
BIN_MS = 100.0
TRIAL_SECONDS = 60
TRIAL_BINS = int(TRIAL_SECONDS * 1000 / BIN_MS)
RAW_TRIAL_FRAMES = FPS * TRIAL_SECONDS
INPUT_NAMES = ['blocked_NW','blocked_N','blocked_NE','blocked_W','blocked_center',
               'blocked_E','blocked_SW','blocked_S','blocked_SE']
OUTPUT_VALUES = ['NW','N','NE','W','center','E','SW','S','SE']


def animal_files():
    """Return primary joblib animal files, excluding larger .mat duplicates."""
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.is_file() and p.name.startswith('QLAK-CA1-') and p.suffix != '.mat')


def blocked_mask(raw):
    """Expand MATLAB-style blocked indices to a static row-major 3x3 mask."""
    idx = np.asarray(raw).ravel().astype(np.int64)
    mask = np.zeros(9, dtype=np.float32)
    idx = idx[idx >= 0]  # -1 denotes no blocked partition
    if idx.size:
        if np.any(idx > 8):
            raise ValueError(f'Invalid blocked indices: {idx}')
        mask[idx] = 1.0
    return mask


def select_cells(position, trace):
    """Match reference position decoder's moving-frame activity curation."""
    present = np.isfinite(trace).any(axis=1)
    # First velocity sample is false, as in decode_position_within.
    displacement_speed = np.linalg.norm(np.diff(position.T, axis=0) * FPS, axis=1)
    moving = np.zeros(position.shape[1], dtype=bool)
    moving[1:] = gaussian_filter1d(displacement_speed, sigma=5, axis=0) > 5.0
    event_sum = np.nansum(trace[:, moving], axis=1)
    keep = present & (event_sum > 5)
    return keep, moving, event_sum


def process_session(position, trace, blocked, subject, source_day, environment):
    """Curate cells, smooth/pool aligned streams, discretize, and split trials."""
    if position.ndim != 2 or position.shape[0] != 2:
        raise ValueError(f'{subject} day {source_day}: bad position shape {position.shape}')
    if trace.ndim != 2 or trace.shape[1] != position.shape[1]:
        raise ValueError(f'{subject} day {source_day}: trace/position mismatch')
    if not np.isfinite(position).all():
        raise ValueError(f'{subject} day {source_day}: nonfinite position')

    keep, moving, event_sum = select_cells(position, trace)
    selected = np.asarray(trace[keep], dtype=np.float32)
    if selected.shape[0] < 1 or not np.isfinite(selected).all():
        raise ValueError(f'{subject} day {source_day}: no valid selected cells')

    n_trials = position.shape[1] // RAW_TRIAL_FRAMES
    used_raw = n_trials * RAW_TRIAL_FRAMES
    if n_trials < 2:
        raise ValueError(f'{subject} day {source_day}: fewer than two complete trials')

    # Match fit_decoder: smooth event vectors continuously, then non-overlapping
    # average pooling in groups of three frames. Smooth before session cropping so
    # minute boundaries do not introduce edge artifacts.
    smooth = gaussian_filter1d(selected, sigma=POOL, axis=1).astype(np.float32, copy=False)
    neural_pooled = smooth[:, :used_raw].reshape(selected.shape[0], -1, POOL).mean(axis=2,
                                                                                   dtype=np.float32)
    pos_pooled = position[:, :used_raw].reshape(2, -1, POOL).mean(axis=2)
    xybin = np.clip(np.floor(pos_pooled / 25.0).astype(np.int64), 0, 2)
    labels = (xybin[1] * 3 + xybin[0]).astype(np.int64)
    geometry = blocked_mask(blocked)

    neural_trials, input_trials, output_trials = [], [], []
    for tr in range(n_trials):
        sl = slice(tr * TRIAL_BINS, (tr + 1) * TRIAL_BINS)
        neural_trials.append(np.ascontiguousarray(neural_pooled[:, sl], dtype=np.float32))
        input_trials.append(geometry.copy())
        output_trials.append(np.ascontiguousarray(labels[sl][None, :], dtype=np.int64))

    assert all(x.shape == (selected.shape[0], TRIAL_BINS) for x in neural_trials)
    assert all(x.shape == (9,) for x in input_trials)
    assert all(x.shape == (1, TRIAL_BINS) for x in output_trials)
    assert all(np.isfinite(x).all() and np.min(x) >= 0 for x in neural_trials)
    assert labels.min() >= 0 and labels.max() <= 8

    info = {
        'subject': subject,
        'source_day_index': int(source_day),
        'environment': str(environment),
        'blocked_indices': [int(x) for x in np.asarray(blocked).ravel() if x >= 0],
        'source_frames': int(position.shape[1]),
        'used_source_frames': int(used_raw),
        'discarded_tail_frames': int(position.shape[1] - used_raw),
        'n_trials': int(n_trials),
        'source_present_neurons': int(np.isfinite(trace).any(axis=1).sum()),
        'retained_neurons': int(selected.shape[0]),
        'moving_frame_fraction': float(moving.mean()),
    }
    plot_payload = (pos_pooled[:, :TRIAL_BINS], labels[:TRIAL_BINS],
                    neural_pooled[:min(30, selected.shape[0]), :TRIAL_BINS], geometry)
    return neural_trials, input_trials, output_trials, info, plot_payload


def make_processing_plot(payload, session_id):
    """Visualize alignment, output discretization, geometry, and neural processing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pos, labels, neural, geometry = payload
    t = np.arange(labels.size) * BIN_MS / 1000.0
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    ax[0].plot(t, pos[0], label='x (cm)', lw=1)
    ax[0].plot(t, pos[1], label='y (cm)', lw=1)
    ax[0].step(t, labels * 8.0, where='mid', label='position class ×8', alpha=.7)
    ax[0].axhline(25, color='k', ls=':', lw=.8); ax[0].axhline(50, color='k', ls=':', lw=.8)
    ax[0].set(xlabel='time (s)', ylabel='position / class', title='Aligned pooled position and 3×3 output')
    ax[0].legend(ncol=3)
    im = ax[1].imshow(neural, aspect='auto', interpolation='nearest', extent=[0,60,neural.shape[0],0])
    ax[1].set(xlabel='time (s)', ylabel='retained neuron', title='Gaussian-smoothed, 100-ms pooled neural activity')
    fig.colorbar(im, ax=ax[1], label='mean event activity')
    ax[2].plot(pos[0], pos[1], lw=.6, color='0.5')
    sc=ax[2].scatter(pos[0], pos[1], c=labels, s=8, cmap='tab10', vmin=0, vmax=9)
    for z in (25,50): ax[2].axvline(z,color='k',lw=.8); ax[2].axhline(z,color='k',lw=.8)
    for k,b in enumerate(geometry):
        if b:
            x=(k%3)*25; y=(k//3)*25
            ax[2].add_patch(plt.Rectangle((x,y),25,25,color='black',alpha=.25))
    ax[2].set(xlim=(0,75),ylim=(75,0),aspect='equal',xlabel='x (cm)',ylabel='y (cm)',
              title='Trajectory, categorical bins, and blocked geometry (shaded)')
    fig.colorbar(sc,ax=ax[2],label='class')
    out=Path(f'/app/processing_{session_id}.png')
    fig.savefig(out,dpi=140); plt.close(fig)
    print(f'  saved {out}', flush=True)


def validate_complete(data):
    ns=len(data['neural'])
    if not (ns == len(data['input']) == len(data['output']) == len(data['subject_idx']) == len(data['brain_region_idx'])):
        raise AssertionError('session-list lengths mismatch')
    for s in range(ns):
        if not (len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s]) >= 2):
            raise AssertionError(f'trial-list mismatch session {s}')
        n=data['neural'][s][0].shape[0]
        if len(data['brain_region_idx'][s]) != n:
            raise AssertionError(f'brain-region mismatch session {s}')
        for a,b,c in zip(data['neural'][s],data['input'][s],data['output'][s]):
            if a.shape != (n,TRIAL_BINS) or b.shape != (9,) or c.shape != (1,TRIAL_BINS):
                raise AssertionError(f'bad trial shape session {s}: {a.shape}, {b.shape}, {c.shape}')
    return True


def convert(outfile, sample=False, show_processing=False):
    start=time.time(); files=animal_files(); subjects=[p.name for p in files]
    neural=[]; inputs=[]; outputs=[]; subject_idx=[]; region_idx=[]; session_info=[]
    class_counts=np.zeros(9,dtype=np.int64); total_present=total_retained=total_trials=0
    plots=0; stop=False
    for si,f in enumerate(files):
        load_t=time.time(); raw=joblib.load(f)[f.name]
        print(f'Loaded {f.name} in {time.time()-load_t:.2f}s',flush=True)
        for day in range(raw['trace'].shape[0]):
            st=time.time()
            ntr,itr,otr,info,payload=process_session(
                raw['position'][day],raw['trace'][day],raw['blocked'][day],f.name,day,raw['envs'].ravel()[day])
            neural.append(ntr); inputs.append(itr); outputs.append(otr); subject_idx.append(si)
            region_idx.append(np.zeros(info['retained_neurons'],dtype=np.int64)); session_info.append(info)
            total_present += info['source_present_neurons']; total_retained += info['retained_neurons']; total_trials += info['n_trials']
            class_counts += np.bincount(np.concatenate([x.ravel() for x in otr]),minlength=9)
            sid=f'{f.name}_day{day:02d}'
            print(f"  {sid}: neurons {info['source_present_neurons']}->{info['retained_neurons']}, "
                  f"trials {info['n_trials']}, {time.time()-st:.2f}s",flush=True)
            if show_processing and plots < 2:
                make_processing_plot(payload,sid); plots += 1
            if sample and len(neural) >= 2:
                stop=True; break
        del raw; gc.collect()
        if stop: break

    metadata={
        'task_description':'Decode mouse location as one of nine 25×25 cm bins in a 3×3 partition of a 75×75 cm arena from CA1 calcium-event activity; environment blocked geometry is supplied as static decoder input.',
        'time_bin_size':BIN_MS,
        'temporal_alignment_event':'Start of each non-overlapping 1-minute segment within a continuous animal-day recording',
        'off_start':0.0,
        'off_end':60.0,
        'source_frame_rate_hz':FPS,
        'source_trial_definition':'Continuous 40-minute animal-day sessions segmented from time zero into complete 60-second trials; incomplete tail discarded.',
        'neural_processing':'Paper-curated binary rising-phase calcium events; day-present cells with >5 events during reference-defined moving frames; Gaussian sigma=3 source-frame smoothing and non-overlapping 3-frame mean pooling.',
        'position_processing':'Aligned x/y averaged over the same 3 source frames, then fixed 25-cm bins; row-major class = y_bin*3+x_bin.',
        'input_encoding':'Nine row-major binary indicators; 1 means the corresponding arena partition is blocked.',
        'session_info':session_info,
        'source_reference':'Lee, Keinath, Cianfarano & Brandon, Neuron 2025',
    }
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,
          'subject_idx':np.asarray(subject_idx,dtype=np.int64),'brain_regions':['CA1'],
          'brain_region_idx':region_idx,'input_names':INPUT_NAMES,
          'output_names':['position_bin'],'output_values':[OUTPUT_VALUES],'metadata':metadata}
    validate_complete(data)
    print(f'Summary: sessions={len(neural)}, trials={total_trials}, source_present_cells={total_present}, retained_cells={total_retained}',flush=True)
    print(f'Output counts={class_counts.tolist()}, fractions={(class_counts/class_counts.sum()).round(6).tolist()}',flush=True)
    tmp=Path(str(outfile)+'.tmp')
    save_t=time.time()
    with open(tmp,'wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,outfile)
    print(f'Saved {outfile} ({Path(outfile).stat().st_size/2**30:.3f} GiB) in {time.time()-save_t:.2f}s; total {time.time()-start:.2f}s',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--full',action='store_true',help='process all sessions (default)')
    mode.add_argument('--sample',action='store_true',help='process only the first two sessions')
    ap.add_argument('--show-processing',action='store_true',help='save processing plots for up to two sessions')
    args=ap.parse_args()
    convert(args.outpicklefile,sample=args.sample,show_processing=args.show_processing)

if __name__=='__main__': main()
