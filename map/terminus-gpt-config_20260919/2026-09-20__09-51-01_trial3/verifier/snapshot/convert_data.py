#!/usr/bin/env python3
"""Convert MAP NWB files to decoder-compatible go-cue-aligned trial data.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse, pickle, time, warnings
from pathlib import Path
from collections import Counter
import h5py
import numpy as np

DATA_ROOT = Path('/app/data')
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
N_BINS = int(round((OFF_END-OFF_START)/BIN_S))
REL_EDGES = OFF_START + np.arange(N_BINS+1, dtype=np.float64)*BIN_S
REL_CENTERS = (REL_EDGES[:-1]+REL_EDGES[1:])/2
VISIBILITY_THRESHOLD = 0.9


def decode_strings(dataset):
    """Robustly decode an HDF5 string-like 1-D dataset."""
    a = dataset[:]
    out=[]
    for x in a:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode('utf-8', errors='replace').strip())
        elif isinstance(x, str):
            out.append(x.strip())
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def eligible_files():
    """Return files with the custom classifier output used by the papers."""
    kept=[]; skipped=[]
    for p in sorted(DATA_ROOT.rglob('*.nwb')):
        with h5py.File(p, 'r') as f:
            d=f['units/classification']
            if d.dtype.kind in 'OSU': kept.append(p)
            else: skipped.append((p.name, 'missing classifier labels'))
    return kept, skipped


def nearest_indices(sorted_t, query):
    idx=np.searchsorted(sorted_t, query)
    idx=np.clip(idx, 1, len(sorted_t)-1)
    left=idx-1
    choose_left=(query-sorted_t[left]) <= (sorted_t[idx]-query)
    return np.where(choose_left, left, idx)


def unit_spike_slice(spikes, ends, unit_id):
    start=0 if unit_id==0 else int(ends[unit_id-1])
    return spikes[start:int(ends[unit_id])]


def bin_selected_units(f, unit_ids, go_kept):
    """Vectorized half-open spike counting, output trials x units x bins."""
    edges=go_kept[:,None]+REL_EDGES[None,:]
    flat=edges.ravel()
    spikes=f['units/spike_times']; ends=f['units/spike_times_index'][:]
    counts=np.empty((len(unit_ids), len(go_kept), N_BINS), dtype=np.float32)
    for j,u in enumerate(unit_ids):
        st=np.asarray(unit_spike_slice(spikes, ends, int(u)), dtype=np.float64)
        cumulative=np.searchsorted(st, flat, side='left').reshape(edges.shape)
        counts[j]=np.diff(cumulative, axis=1)/BIN_S
    return counts.transpose(1,0,2)  # trials, neurons, time


def session_tone_onsets(f, trial_starts, go):
    sample=f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
    pos=np.searchsorted(sample, go, side='left')-1
    if np.any(pos<0): raise ValueError('go cue without preceding sample onset')
    tone=sample[pos]
    if np.any(tone<trial_starts): raise ValueError('last pre-go sample is outside trial')
    return tone


def photostim_series(f, absolute_centers):
    ev=f['acquisition/BehavioralEvents']
    on=ev['photostim_start_times/timestamps'][:]
    off=ev['photostim_stop_times/timestamps'][:]
    if len(on)!=len(off): raise ValueError('photostim on/off count mismatch')
    result=np.zeros(absolute_centers.shape, dtype=bool)
    for a,b in zip(on,off): result |= ((absolute_centers>=a)&(absolute_centers<b))
    return result.astype(np.float32)


def process_session(path, show_processing=False):
    t0=time.perf_counter()
    with h5py.File(path,'r') as f:
        subject=(f['general/subject/subject_id'].asstr()[()]
                 if f['general/subject/subject_id'].dtype.kind in 'OSU'
                 else path.parent.name.replace('sub-',''))
        trials=f['intervals/trials']; nt=len(trials['id'])
        go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
        if len(go)!=nt: raise ValueError(f'{path.name}: {len(go)} go events != {nt} trials')
        starts=trials['start_time'][:]
        next_starts=np.r_[starts[1:], np.inf]
        if np.any(go+OFF_END>next_starts): raise ValueError('requested window crosses next trial')

        # NWB units are trialized: obs_intervals and is_good_trials columns identify
        # behavioral trials that were actually recorded electrophysiologically.
        good_trials=f['units/is_good_trials'][:]
        obs=f['units/obs_intervals']; obs_end=f['units/obs_intervals_index'][:]
        obs_start=np.r_[0,obs_end[:-1]]
        interval_counts=obs_end-obs_start
        if not np.all(interval_counts==good_trials.shape[1]):
            raise ValueError(f'{path.name}: inconsistent per-unit observation interval counts')
        ref_intervals=obs[obs_start[0]:obs_end[0]]
        recorded_idx=np.searchsorted(starts,ref_intervals[:,0])
        if np.any(recorded_idx>=nt) or not np.allclose(starts[recorded_idx],ref_intervals[:,0],atol=1e-6):
            raise ValueError(f'{path.name}: cannot map neural observation intervals to trials')
        recorded_mask=np.zeros(nt,dtype=bool); recorded_mask[recorded_idx]=True
        neural_window_mask=np.zeros(nt,dtype=bool)
        neural_window_mask[recorded_idx]=((go[recorded_idx]+OFF_START>=ref_intervals[:,0]) &
                                          (go[recorded_idx]+OFF_END<=ref_intervals[:,1]))

        # Complete video coverage is required for every retained output sample.
        tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
        video_t=tg['timestamps'][:]
        video_mask=(go+OFF_START>=video_t[0]) & (go+OFF_END<=video_t[-1])
        keep_trial=recorded_mask & video_mask
        kept=np.flatnonzero(keep_trial)
        if len(kept)<2: raise ValueError(f'{path.name}: fewer than two complete neural+video trials')
        go_k=go[kept]; centers=go_k[:,None]+REL_CENTERS[None,:]

        classification=decode_strings(f['units/classification'])
        stable=good_trials.all(axis=1)
        unit_ids=np.flatnonzero((classification=='good') & stable)
        if len(unit_ids)==0: raise ValueError(f'{path.name}: no stable classifier-good units')
        neural_cube=bin_selected_units(f,unit_ids,go_k)
        # Exclude mapped trials having no spikes from any retained unit anywhere
        # in the requested window; these have no usable neural decoder input.
        neural_nonzero=np.any(neural_cube!=0,axis=(1,2))
        excluded_all_zero=int(np.sum(~neural_nonzero))
        if excluded_all_zero:
            kept=kept[neural_nonzero]
            go_k=go_k[neural_nonzero]
            centers=centers[neural_nonzero]
            neural_cube=neural_cube[neural_nonzero]
        if len(kept)<2:
            raise ValueError(f'{path.name}: fewer than two usable neural trials')

        tone_all=session_tone_onsets(f, starts, go)
        time_from_tone=(centers-tone_all[kept,None]).astype(np.float32)
        stim=photostim_series(f,centers)
        input_cube=np.stack((time_from_tone,stim),axis=1).astype(np.float32)

        outcome=decode_strings(trials['outcome'])[kept]
        instruction=decode_strings(trials['trial_instruction'])[kept]
        early=decode_strings(trials['early_lick'])[kept]
        choice=np.where(outcome=='ignore',2,np.where(outcome=='hit',
                         np.where(instruction=='left',0,1),
                         np.where(instruction=='left',1,0))).astype(np.int8)
        outcome_code=np.array([{'ignore':0,'miss':1,'hit':2}[x] for x in outcome],dtype=np.int8)
        early_code=np.array([{'no early':0,'early':1}[x] for x in early],dtype=np.int8)

        tongue=np.asarray(tg['data'][:],dtype=np.float64)
        visible_session=tongue[:,2]>=VISIBILITY_THRESHOLD
        if visible_session.sum()<2: raise ValueError(f'{path.name}: insufficient visible tongue frames')
        q40,q60=np.percentile(tongue[visible_session,1],[40,60])
        vi=nearest_indices(video_t,centers.ravel()).reshape(centers.shape)
        y=tongue[vi,1]; likelihood=tongue[vi,2]
        tongue_code=np.full(y.shape,3,dtype=np.int8)
        vis=likelihood>=VISIBILITY_THRESHOLD
        tongue_code[vis & (y<q40)]=0
        tongue_code[vis & (y>=q40) & (y<=q60)]=1
        tongue_code[vis & (y>q60)]=2

        output_cube=np.empty((len(kept),4,N_BINS),dtype=np.int8)
        output_cube[:,0,:]=choice[:,None]
        output_cube[:,1,:]=outcome_code[:,None]
        output_cube[:,2,:]=early_code[:,None]
        output_cube[:,3,:]=tongue_code

        anno=decode_strings(f['units/anno_name'])[unit_ids].tolist()
        # Empty/missing native annotation gets an explicit name rather than an invalid index.
        anno=[x if x and x.lower() not in ('nan','none') else 'Unknown' for x in anno]
        session_info={
            'session_id':path.stem, 'source_file':str(path), 'subject':str(subject),
            'source_trials':int(nt), 'neural_recorded_trials':int(recorded_mask.sum()),
            'retained_trials':int(len(kept)),
            'excluded_unrecorded_neural_trials':int((~recorded_mask).sum()),
            'excluded_incomplete_neural_window_trials':int(np.sum(recorded_mask & ~neural_window_mask)),
            'excluded_incomplete_video_trials':int(np.sum(recorded_mask & ~video_mask)),
            'excluded_all_zero_neural_trials':excluded_all_zero,
            'source_units':int(len(classification)), 'classifier_good_units':int(np.sum(classification=='good')),
            'retained_stable_good_units':int(len(unit_ids)),
            'tongue_y_percentile_40':float(q40), 'tongue_y_percentile_60':float(q60),
            'processing_seconds':float(time.perf_counter()-t0)
        }

    assert neural_cube.shape==(len(kept),len(unit_ids),N_BINS)
    assert input_cube.shape==(len(kept),2,N_BINS)
    assert output_cube.shape==(len(kept),4,N_BINS)
    assert np.isfinite(neural_cube).all() and np.isfinite(input_cube).all()

    if show_processing:
        plot_processing(path.stem, REL_CENTERS, neural_cube, input_cube, output_cube,
                        q40, q60, choice, outcome_code, early_code)
    return {
        'subject':str(subject), 'neural':[neural_cube[i] for i in range(len(kept))],
        'input':[input_cube[i] for i in range(len(kept))],
        'output':[output_cube[i] for i in range(len(kept))],
        'regions':anno, 'info':session_info
    }


def plot_processing(session_id,t,neural,inp,out,q40,q60,choice,outcome,early):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(4,2,figsize=(14,11),constrained_layout=True)
    ax[0,0].imshow(neural[0,:min(100,neural.shape[1])],aspect='auto',extent=[t[0],t[-1],min(100,neural.shape[1]),0]); ax[0,0].set_title('Trial 1 firing rates (up to 100 neurons)')
    ax[0,1].plot(t,neural[0].mean(0)); ax[0,1].axvline(0,color='k'); ax[0,1].set_title('Population mean aligned to go=0')
    ax[1,0].plot(t,inp[0,0],label='time from tone'); ax[1,0].plot(t,inp[0,1],label='photostim'); ax[1,0].legend(); ax[1,0].set_title('Decoder inputs, trial 1')
    ax[1,1].imshow(out[0],aspect='auto',extent=[t[0],t[-1],4,0]); ax[1,1].set_yticks(np.arange(4)+.5,['choice','outcome','early','tongue']); ax[1,1].set_title('Decoder outputs, trial 1')
    ax[2,0].hist(out[:,3,:].ravel(),bins=np.arange(5)-.5); ax[2,0].set_xticks(range(4)); ax[2,0].set_title(f'Tongue classes (q40={q40:.2f}, q60={q60:.2f})')
    ax[2,1].bar(range(3),np.bincount(choice,minlength=3)); ax[2,1].set_xticks(range(3),['left','right','no lick']); ax[2,1].set_title('Choice')
    ax[3,0].bar(range(3),np.bincount(outcome,minlength=3)); ax[3,0].set_xticks(range(3),['ignore','miss','hit']); ax[3,0].set_title('Outcome')
    ax[3,1].bar(range(2),np.bincount(early,minlength=2)); ax[3,1].set_xticks(range(2),['no','yes']); ax[3,1].set_title('Early lick')
    for a in ax.ravel(): a.set_xlabel('time from go (s)' if a in ax[:2].ravel() else '')
    fig.suptitle(session_id)
    fig.savefig(f'/app/processing_{session_id}.png',dpi=130); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--full',action='store_true',help='process all sessions (default)')
    mode.add_argument('--sample',action='store_true',help='process first two eligible sessions')
    ap.add_argument('--show-processing',action='store_true',help='save processing plots for up to two sessions')
    ap.add_argument('outpicklefile')
    args=ap.parse_args()
    files,skipped=eligible_files()
    if args.sample: files=files[:2]
    print(f'Eligible sessions: {len(files)}; skipped source files: {len(skipped)}',flush=True)
    for x in skipped: print('SKIP',x,flush=True)
    sessions=[]; t0=time.perf_counter()
    for i,p in enumerate(files):
        ts=time.perf_counter(); print(f'[{i+1}/{len(files)}] {p.name}',flush=True)
        sess=process_session(p,show_processing=args.show_processing and i<2)
        sessions.append(sess)
        print('  trials={retained_trials}/{source_trials}, neurons={retained_stable_good_units}, time={:.2f}s'.format(time.perf_counter()-ts,**sess['info']),flush=True)

    subjects=sorted(set(x['subject'] for x in sessions)); smap={x:i for i,x in enumerate(subjects)}
    regions=sorted(set(r for x in sessions for r in x['regions'])); rmap={x:i for i,x in enumerate(regions)}
    data={
      'neural':[x['neural'] for x in sessions], 'input':[x['input'] for x in sessions],
      'output':[x['output'] for x in sessions], 'subjects':subjects,
      'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int32),
      'brain_regions':regions,
      'brain_region_idx':[np.asarray([rmap[r] for r in x['regions']],dtype=np.int32) for x in sessions],
      'input_names':['time from tone onset','photostimulation on'],
      'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
      'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
      'metadata':{
        'task_description':'Auditory delayed-response task; decode actual lick choice, outcome, early licking, and time-varying tongue y-position from neural activity.',
        'time_bin_size':50.0, 'time_bin_size_units':'ms',
        'temporal_alignment_event':'go cue onset (BehavioralEvents/go_start_times)',
        'off_start':OFF_START, 'off_end':OFF_END, 'n_time_bins':N_BINS,
        'bin_centers_seconds':REL_CENTERS.astype(np.float32),
        'spike_binning':'half-open [left,right) bins; counts divided by 0.05 s (Hz)',
        'neuron_filter':"units/classification == 'good' and is_good_trials true for every source trial",
        'session_filter':'requires paper custom classifier labels; one all-NaN-classification source session excluded',
        'trial_filter':'retain trials represented in units/obs_intervals with complete tongue-video coverage; exclude wholly zero neural windows',
        'tongue_visibility_likelihood_threshold':VISIBILITY_THRESHOLD,
        'tongue_discretization':'session visible-frame y percentiles: <40, 40-60 inclusive, >60; likelihood<0.9 is not visible',
        'choice_derivation':'ignore=no lick; hit=instruction side; miss=opposite side, using curated NWB trial labels',
        'tone_onset_definition':'last sample_start_times event after trial start and before go (handles replay)',
        'session_info':[x['info'] for x in sessions],
        'source_dataset':'Mesoscale Activity Map NWB files provided in /app/data'
      }
    }
    out=Path(args.outpicklefile); out.parent.mkdir(parents=True,exist_ok=True)
    print(f'Writing {out} ...',flush=True)
    with out.open('wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
    elapsed=time.perf_counter()-t0
    print(f'DONE sessions={len(sessions)} subjects={len(subjects)} trials={sum(map(len,data["neural"]))} neurons={sum(len(x) for x in data["brain_region_idx"])} regions={len(regions)} size={out.stat().st_size/1e9:.3f}GB elapsed={elapsed:.1f}s',flush=True)

if __name__=='__main__': main()
