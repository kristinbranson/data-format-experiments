#!/usr/bin/env python3
"""Convert Lee et al. CA1 geometry data to decoder-compatible trials."""
import argparse, gc, pickle, sys, time
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
OUTPUT_LABELS = [
    'bottom-left', 'bottom-center', 'bottom-right',
    'middle-left', 'middle-center', 'middle-right',
    'top-left', 'top-center', 'top-right'
]

def animal_files():
    return sorted(p for p in DATA_DIR.iterdir()
                  if p.name.startswith('QLAK-CA1-') and not p.suffix and '_' not in p.name)

def blocked_indices(raw):
    vals = np.asarray(raw[0]).ravel().astype(int)
    return vals[vals >= 0]

def geometry_vector(raw):
    geom = np.ones(9, dtype=np.float32)
    geom[blocked_indices(raw)] = 0
    return geom

def nearest_accessible(classes, xbin, ybin, geom):
    """Snap classes in blocked partitions to nearest accessible grid center."""
    bad = geom[classes] == 0
    if not np.any(bad):
        return classes, 0
    valid = np.flatnonzero(geom).astype(np.int8)
    vx, vy = valid % 3, valid // 3
    bx, by = xbin[bad, None], ybin[bad, None]
    nearest = valid[np.argmin((bx-vx[None, :])**2 + (by-vy[None, :])**2, axis=1)]
    out = classes.copy()
    out[bad] = nearest
    return out, int(np.count_nonzero(bad))

def process_day(trace_day, position_day, blocked_raw):
    """Reference-style temporal processing for one aligned recording day."""
    present = np.isfinite(trace_day[:, 0])
    if not np.array_equal(present, np.isfinite(trace_day[:, -1])):
        raise ValueError('Cell registration mask changes within a session')
    raw = np.asarray(trace_day[present].T, dtype=np.float32)  # time x neurons
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite value in a day-present neural trace')
    n_pool = raw.shape[0] // POOL
    n_used = n_pool * POOL
    # Match reference fit_decoder/test_decoder: temporal Gaussian sigma=3 then AvgPool stride=3.
    smooth = gaussian_filter1d(raw, sigma=POOL, axis=0)
    pooled_neural = smooth[:n_used].reshape(n_pool, POOL, raw.shape[1]).mean(axis=1, dtype=np.float32)
    pos = np.asarray(position_day[:, :n_used], dtype=np.float64)
    pooled_pos = pos.reshape(2, n_pool, POOL).mean(axis=2)
    xbin = np.clip((pooled_pos[0] / 25.0).astype(np.int8), 0, 2)
    ybin = np.clip((pooled_pos[1] / 25.0).astype(np.int8), 0, 2)
    classes = (ybin * 3 + xbin).astype(np.int8)
    geom = geometry_vector(blocked_raw)
    classes, n_corrected = nearest_accessible(classes, xbin, ybin, geom)
    return pooled_neural.T, classes[None, :], geom, present, n_corrected

def plot_processing(outpath, animal, day, env, raw_trace, raw_pos, neural, output, geom):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0,0].plot(raw_pos[0], raw_pos[1], lw=.25)
    ax[0,0].set(title=f'{animal} day {day}: raw trajectory ({env})', xlim=(0,75), ylim=(0,75), aspect='equal')
    ax[0,1].imshow(geom.reshape(3,3), origin='lower', vmin=0, vmax=1, cmap='gray')
    ax[0,1].set(title='Geometry (white=accessible)')
    nshow=min(40,raw_trace.shape[0]); tshow=min(1800,raw_trace.shape[1])
    ax[0,2].imshow(raw_trace[:nshow,:tshow], aspect='auto', interpolation='none', cmap='binary')
    ax[0,2].set(title='Raw binary events (first minute)', xlabel='30 Hz frame', ylabel='neuron')
    ax[1,0].imshow(neural[:nshow,:TRIAL_BINS], aspect='auto', interpolation='none', cmap='viridis')
    ax[1,0].set(title='Gaussian-smoothed + 3-frame mean', xlabel='100 ms bin', ylabel='neuron')
    ax[1,1].plot(output[0,:TRIAL_BINS], lw=.7)
    ax[1,1].set(title='Aligned 3x3 position class', xlabel='100 ms bin', ylim=(-.5,8.5), yticks=range(9))
    counts=np.bincount(output.ravel(),minlength=9)
    ax[1,2].bar(range(9),counts, color=['C0' if geom[i] else 'C3' for i in range(9)])
    ax[1,2].set(title='Output counts (red=blocked input)', xlabel='class', ylabel='bins', xticks=range(9))
    fig.tight_layout(); fig.savefig(outpath,dpi=140); plt.close(fig)

def convert(outfile, sample=False, show_processing=False):
    t_all=time.time(); files=animal_files()
    subjects=[p.name for p in files]
    neural=[]; inputs=[]; outputs=[]; subject_idx=[]; region_idx=[]; session_info=[]
    plot_count=0; session_count=0; total_trials=0; corrected_total=0
    stop=False
    for si, f in enumerate(files):
        t_load=time.time(); print(f'Loading {f.name}...', flush=True)
        d=joblib.load(f)[f.name]
        print(f'  loaded in {time.time()-t_load:.2f}s', flush=True)
        for day in range(d['trace'].shape[0]):
            if sample and session_count >= 2:
                stop=True; break
            t0=time.time()
            nmat, out, geom, present, ncorrect = process_day(
                d['trace'][day], d['position'][day], d['blocked'][day])
            ntrials=nmat.shape[1]//TRIAL_BINS
            if ntrials < 2:
                print(f'WARNING skipping {f.name} day {day}: only {ntrials} full trials')
                continue
            keep=ntrials*TRIAL_BINS
            ns=[np.ascontiguousarray(x, dtype=np.float32) for x in np.split(nmat[:,:keep],ntrials,axis=1)]
            os=[np.ascontiguousarray(x, dtype=np.int8) for x in np.split(out[:,:keep],ntrials,axis=1)]
            ins=[geom.copy() for _ in range(ntrials)]
            neural.append(ns); outputs.append(os); inputs.append(ins)
            subject_idx.append(si); region_idx.append(np.zeros(nmat.shape[0],dtype=np.int8))
            env=str(np.asarray(d['envs']).ravel()[day])
            info=dict(session_id=f'{f.name}_day{day:02d}', subject=f.name, source_day=day,
                      environment=env, blocked=blocked_indices(d['blocked'][day]).tolist(),
                      native_frames=int(d['trace'].shape[2]), present_neurons=int(present.sum()),
                      n_trials=int(ntrials), retained_time_bins=int(keep),
                      discarded_native_equivalent_frames=int(d['trace'].shape[2]-keep*POOL),
                      corrected_blocked_pooled_bins=int(ncorrect))
            session_info.append(info); total_trials+=ntrials; corrected_total+=ncorrect
            if show_processing and plot_count < 2:
                plot_processing(Path('/app')/f'processing_{info["session_id"]}.png', f.name, day, env,
                                d['trace'][day,present],d['position'][day],nmat,out,geom)
                plot_count+=1
            print(f'  session {session_count:03d} {info["session_id"]}: {present.sum()} neurons, '
                  f'{ntrials} trials, corrected={ncorrect}, {time.time()-t0:.2f}s', flush=True)
            session_count+=1
        del d; gc.collect()
        if stop: break
    used_subjects=sorted(set(subject_idx))
    # Keep full subject vocabulary as required IDs of all subjects; sample may reference only index 0.
    data=dict(
        neural=neural, input=inputs, output=outputs,
        subjects=subjects, subject_idx=np.asarray(subject_idx,dtype=np.int64),
        brain_regions=['CA1'], brain_region_idx=region_idx,
        input_names=[f'accessible_{name}' for name in OUTPUT_LABELS],
        output_names=['position_bin'], output_values=[OUTPUT_LABELS],
        metadata=dict(
            task_description='Decode mouse position in a 3x3 arena grid from dorsal CA1 calcium-event activity; arena accessibility is static context.',
            time_bin_size=BIN_MS,
            temporal_alignment_event='Start of each contiguous non-overlapping 1-minute segment within a recording day',
            off_start=0.0, off_end=60.0, source_frame_rate_hz=FPS,
            neural_processing='Deposited binary rising-phase events; Gaussian smoothing sigma=3 native frames followed by non-overlapping 3-frame mean pooling.',
            position_processing='Aligned 3-frame mean position, discretized into 25 cm bins; class=y_bin*3+x_bin; rare blocked-bin points snapped to nearest accessible bin.',
            trialization='Complete 60 s segments only; trailing sub-minute data discarded.',
            geometry_encoding='Nine row-major values: 1=accessible, 0=blocked.',
            session_info=session_info,
            conversion_mode='sample' if sample else 'full'))
    # Internal structural checks.
    assert len(neural)==len(inputs)==len(outputs)==len(subject_idx)==len(region_idx)==len(session_info)
    for sidx in range(len(neural)):
        assert len(neural[sidx])==len(inputs[sidx])==len(outputs[sidx])>=2
        nn=neural[sidx][0].shape[0]
        for n,i,o in zip(neural[sidx],inputs[sidx],outputs[sidx]):
            assert n.shape==(nn,TRIAL_BINS) and i.shape==(9,) and o.shape==(1,TRIAL_BINS)
            assert np.isfinite(n).all() and np.all((o>=0)&(o<9)) and np.all(i[o]==1)
    print(f'Saving {len(neural)} sessions, {total_trials} trials to {outfile}...',flush=True)
    t_save=time.time()
    with open(outfile,'wb') as fh: pickle.dump(data,fh,protocol=5)
    print(f'Saved in {time.time()-t_save:.2f}s; total {time.time()-t_all:.2f}s; '
          f'blocked-bin corrections before trial-tail removal={corrected_total}',flush=True)
    print(f'Output size: {Path(outfile).stat().st_size/1024**2:.2f} MiB',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group(); mode.add_argument('--full',action='store_true'); mode.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true')
    a=ap.parse_args(); convert(a.outpicklefile,sample=a.sample,show_processing=a.show_processing)
if __name__=='__main__': main()
