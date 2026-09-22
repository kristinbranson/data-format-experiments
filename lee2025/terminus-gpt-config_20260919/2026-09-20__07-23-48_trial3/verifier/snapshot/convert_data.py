#!/usr/bin/env python3
"""Convert Lee et al. CA1 geometry data to decoder-compatible trials."""
import argparse, gc, os, pickle, time
from pathlib import Path
import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

ANIMALS = ['QLAK-CA1-08','QLAK-CA1-30','QLAK-CA1-50','QLAK-CA1-51','QLAK-CA1-56','QLAK-CA1-74','QLAK-CA1-75']
FPS = 30
POOL = 3
BIN_MS = 100.0
TRIAL_FRAMES = 60 * FPS
TRIAL_BINS = TRIAL_FRAMES // POOL


def blocked_vector(blocked):
    """Return row-major 3x3 mask (1 blocked, 0 accessible)."""
    out = np.zeros(9, dtype=np.float32)
    idx = np.asarray(blocked).reshape(-1).astype(int)
    idx = idx[idx >= 0]
    if len(idx):
        if np.any(idx > 8):
            raise ValueError(f'Invalid blocked indices: {idx}')
        out[idx] = 1
    return out


def pool_position(position, nframes):
    p = np.asarray(position, dtype=np.float64)[:, :nframes]
    if p.shape != (2, nframes) or not np.isfinite(p).all():
        raise ValueError(f'Invalid position shape/values: {p.shape}')
    return p.reshape(2, -1, POOL).mean(axis=2)


def position_labels(xy, blocked_mask):
    """Discretize pooled x-y coordinates and snap rare blocked labels to nearest open cell."""
    col = np.clip(np.floor(xy[0] / 25.0), 0, 2).astype(np.int64)
    row = np.clip(np.floor(xy[1] / 25.0), 0, 2).astype(np.int64)
    labels = row * 3 + col
    invalid = blocked_mask[labels].astype(bool)
    nfixed = int(invalid.sum())
    if nfixed:
        open_labels = np.flatnonzero(blocked_mask == 0)
        centers = np.column_stack(((open_labels % 3 + 0.5) * 25.0,
                                   (open_labels // 3 + 0.5) * 25.0))
        pts = xy[:, invalid].T
        nearest = np.argmin(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        labels[invalid] = open_labels[nearest]
    return labels.astype(np.uint8), nfixed


def process_session(trace, position, blocked, animal, day):
    trace = np.asarray(trace)
    position = np.asarray(position)
    if trace.ndim != 2 or position.ndim != 2 or position.shape[0] != 2:
        raise ValueError(f'{animal} day {day}: bad trace/position dimensions')
    if trace.shape[1] != position.shape[1]:
        raise ValueError(f'{animal} day {day}: neural-position length mismatch')
    finite_all = np.all(np.isfinite(trace), axis=1)
    finite_any = np.any(np.isfinite(trace), axis=1)
    if np.any(finite_any != finite_all):
        raise ValueError(f'{animal} day {day}: partially missing neural row')
    raw = trace[finite_all]
    vals = np.unique(raw)
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(f'{animal} day {day}: nonbinary finite trace values {vals[:10]}')
    ntrials = trace.shape[1] // TRIAL_FRAMES
    nframes = ntrials * TRIAL_FRAMES
    if ntrials < 2:
        raise ValueError(f'{animal} day {day}: fewer than two complete trials')

    # Reference decoder: gaussian_filter1d sigma=temporal_bin_size, then AvgPool1d(3,3).
    smooth = gaussian_filter1d(raw.astype(np.float32), sigma=POOL, axis=1)
    pooled = smooth[:, :nframes].reshape(raw.shape[0], -1, POOL).mean(axis=2)
    pooled = pooled.astype(np.float32)
    xy = pool_position(position, nframes)
    bmask = blocked_vector(blocked)
    labels, nfixed = position_labels(xy, bmask)
    if np.any(bmask[labels]):
        raise AssertionError('Blocked output remained after correction')

    neural_trials = [np.ascontiguousarray(pooled[:, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
    input_trials = [bmask.copy() for _ in range(ntrials)]
    output_trials = [np.ascontiguousarray(labels[None, i*TRIAL_BINS:(i+1)*TRIAL_BINS]) for i in range(ntrials)]
    return neural_trials, input_trials, output_trials, int(finite_all.sum()), nfixed, trace.shape[1]-nframes


def plot_processing(session_id, raw_trace, raw_position, neural_trials, output_trials, bmask, outdir='/app'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nshow = min(1800, raw_position.shape[1])
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), constrained_layout=True)
    ax[0].plot(np.arange(nshow)/FPS, raw_position[0,:nshow], label='x'); ax[0].plot(np.arange(nshow)/FPS, raw_position[1,:nshow], label='y')
    ax[0].set_ylabel('position (cm)'); ax[0].legend(); ax[0].set_title(f'{session_id}: aligned source streams')
    reg = np.all(np.isfinite(raw_trace),axis=1)
    ax[1].imshow(raw_trace[reg][:min(50,reg.sum()),:nshow], aspect='auto', interpolation='nearest', cmap='gray_r', extent=[0,nshow/FPS,min(50,reg.sum()),0])
    ax[1].set_ylabel('cells'); ax[1].set_title('binary source events (first 50 registered cells)')
    ax[2].imshow(neural_trials[0][:min(50,neural_trials[0].shape[0])], aspect='auto', interpolation='nearest', extent=[0,60,min(50,neural_trials[0].shape[0]),0])
    ax[2].set_ylabel('cells'); ax[2].set_title('Gaussian-smoothed, 100 ms pooled neural trial')
    t=np.arange(TRIAL_BINS)*0.1
    ax[3].step(t, output_trials[0][0], where='post'); ax[3].set_ylim(-.5,8.5); ax[3].set_yticks(range(9)); ax[3].set_xlabel('trial time (s)'); ax[3].set_ylabel('3x3 class')
    ax[3].set_title('Aligned discretized position; blocked='+','.join(map(str,np.flatnonzero(bmask))))
    path=os.path.join(outdir,f'processing_{session_id}.png'); fig.savefig(path,dpi=140); plt.close(fig)
    print(f'  saved {path}', flush=True)


def convert(outfile, sample=False, show_processing=False):
    t_all=time.time(); selected=ANIMALS[:2] if sample else ANIMALS
    neural=[]; inputs=[]; outputs=[]; subject_idx=[]; region_idx=[]; session_info=[]
    total_trials=total_session_neurons=total_fixed=0; plot_count=0
    for subj,animal in enumerate(selected):
        t0=time.time(); root=joblib.load('/app/data/'+animal); dat=root[animal]
        print(f'Loaded {animal} in {time.time()-t0:.1f}s ({len(dat["trace"])} sessions)',flush=True)
        for day,(tr,pos,blk,env) in enumerate(zip(dat['trace'],dat['position'],dat['blocked'],dat['envs'])):
            ts=time.time()
            nt,it,ot,ncells,nfixed,tail=process_session(tr,pos,blk,animal,day)
            neural.append(nt); inputs.append(it); outputs.append(ot); subject_idx.append(subj)
            region_idx.append(np.zeros(ncells,dtype=np.uint8))
            env_name=str(np.asarray(env).reshape(-1)[0])
            session_info.append({'session_id':f'{animal}_day{day:02d}','subject':animal,'source_day_index':day,
                                 'environment':env_name,'blocked_indices':np.flatnonzero(blocked_vector(blk)).tolist(),
                                 'source_frames':int(np.asarray(pos).shape[1]),'discarded_tail_frames':int(tail),
                                 'n_neurons':ncells,'n_trials':len(nt),'corrected_blocked_bins':nfixed})
            total_trials += len(nt); total_session_neurons += ncells; total_fixed += nfixed
            if show_processing and plot_count < 2:
                plot_processing(f'{animal}_day{day:02d}',np.asarray(tr),np.asarray(pos),nt,ot,blocked_vector(blk))
                plot_count += 1
            print(f'  day {day:02d} {env_name:10s}: cells={ncells:3d} trials={len(nt):2d} tail={tail:4d} fixed={nfixed:2d} ({time.time()-ts:.2f}s)',flush=True)
        del root,dat; gc.collect()

    data={'neural':neural,'input':inputs,'output':outputs,
          'subjects':selected,'subject_idx':np.asarray(subject_idx,dtype=np.int64),
          'brain_regions':['CA1'],'brain_region_idx':region_idx,
          'input_names':[f'blocked_row{r}_col{c}' for r in range(3) for c in range(3)],
          'output_names':['position_3x3'],
          'output_values':[[f'row{r}_col{c}' for r in range(3) for c in range(3)]],
          'metadata':{'task_description':'Decode mouse position in nine 25 cm spatial bins from dorsal CA1 binary calcium-event activity; geometry inputs mark blocked arena partitions.',
                      'time_bin_size':BIN_MS,'temporal_alignment_event':'start of each consecutive non-overlapping 60-second segment within a recording session',
                      'off_start':0.0,'off_end':60.0,'source_sampling_rate_hz':FPS,'source_session_duration_minutes':40,
                      'neural_processing':'Released binary rising-phase events; Gaussian smoothing sigma=3 source frames followed by non-overlapping 3-frame mean pooling.',
                      'position_processing':'Aligned x-y position mean-pooled over the same 3 frames, discretized into row-major 3x3 bins; rare blocked labels snapped to nearest accessible bin.',
                      'trial_duration_seconds':60.0,'spatial_bin_size_cm':25.0,'session_info':session_info,
                      'discard_incomplete_final_minute':True}}
    # Structural invariants before serialization.
    ns=len(neural)
    assert ns==len(inputs)==len(outputs)==len(subject_idx)==len(region_idx)==len(session_info)
    for sidx in range(ns):
        assert len(neural[sidx])==len(inputs[sidx])==len(outputs[sidx])>=2
        assert neural[sidx][0].shape[0]==len(region_idx[sidx])
        for n,x,y in zip(neural[sidx],inputs[sidx],outputs[sidx]):
            assert n.shape[1]==y.shape[1]==TRIAL_BINS and x.shape==(9,) and y.shape==(1,TRIAL_BINS)
            assert np.isfinite(n).all() and y.min()>=0 and y.max()<=8
    print(f'Writing {outfile}: sessions={ns}, trials={total_trials}, session-neurons={total_session_neurons}, corrected bins={total_fixed}',flush=True)
    with open(outfile,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done in {time.time()-t_all:.1f}s; size={os.path.getsize(outfile)/2**30:.3f} GiB',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group(); mode.add_argument('--full',action='store_true'); mode.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true')
    args=ap.parse_args()
    convert(args.outpicklefile,sample=args.sample,show_processing=args.show_processing)

if __name__=='__main__': main()
