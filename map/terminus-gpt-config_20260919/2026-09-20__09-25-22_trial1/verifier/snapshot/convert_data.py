#!/usr/bin/env python3
"""Convert Li et al. brain-wide NWB files to decoder-compatible pickle."""
import argparse, pickle, re, time
from pathlib import Path
import h5py
import numpy as np

DATA_ROOT = Path('/app/data')
OFF_START, OFF_END, BIN_S = -2.5, 1.5, 0.05
EDGES_REL = np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL = (EDGES_REL[:-1] + EDGES_REL[1:]) / 2
NT = len(CENTERS_REL)
assert NT == 80

def decode(a):
    return np.asarray([x.decode(errors='replace') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in a])

def ragged_bounds(index):
    index = np.asarray(index, dtype=np.int64)
    return np.r_[0, index[:-1]], index

def map_observed_trials(trial_starts, intervals):
    j = np.searchsorted(trial_starts, intervals[:, 0])
    j = np.minimum(j, len(trial_starts)-1)
    alt = np.maximum(j-1, 0)
    j = np.where(np.abs(trial_starts[alt]-intervals[:,0]) < np.abs(trial_starts[j]-intervals[:,0]), alt, j)
    if len(j) and np.max(np.abs(trial_starts[j]-intervals[:,0])) > 1e-5:
        raise ValueError('Observation intervals do not map to trial starts')
    return j

def curate(f):
    u=f['units']; good=np.where(decode(u['classification'][()]) == 'good')[0]
    if len(good)==0: return good, np.zeros(len(f['intervals/trials/id']), bool)
    trial_st=f['intervals/trials/start_time'][()]
    go=f['acquisition/BehavioralEvents/go_start_times/timestamps'][()]
    flat=u['obs_intervals'][()]; a,b=ragged_bounds(u['obs_intervals_index'][()]); manual=u['is_good_trials'][()]
    valid=np.ones(len(go), bool)
    for ui in good:
        z=flat[a[ui]:b[ui]]; m=manual[ui]
        if len(z)!=len(m): raise ValueError('Validity/interval length mismatch')
        jj=map_observed_trials(trial_st,z)
        uv=np.zeros(len(go), bool)
        uv[jj]=m  # obs_intervals rows identify recorded trials; their stop times are behavioral trial ends, not recording ends
        valid &= uv
    return good, valid

def final_tone_onsets(f, trial_indices, go):
    starts=f['intervals/trials/start_time'][()]
    sample=np.sort(f['acquisition/BehavioralEvents/sample_start_times/timestamps'][()])
    out=np.empty(len(trial_indices), np.float64)
    for k,i in enumerate(trial_indices):
        lo=np.searchsorted(sample, starts[i]-1e-8, 'left'); hi=np.searchsorted(sample, go[i]+1e-8, 'right')
        if hi<=lo: raise ValueError(f'No sample onset for trial {i}')
        out[k]=sample[hi-1]
    return out

def spike_rates(f, units, go_valid):
    """Vectorized within each unit; output trial x neuron x time float32."""
    ntr, nn=len(go_valid),len(units)
    counts=np.zeros((ntr, nn, NT), dtype=np.uint16)
    starts=go_valid+OFF_START; ends=go_valid+OFF_END
    u=f['units']; flat=u['spike_times']; a,b=ragged_bounds(u['spike_times_index'][()])
    for col,ui in enumerate(units):
        sp=np.asarray(flat[a[ui]:b[ui]])
        # windows are ordered and non-overlapping in this task
        ti=np.searchsorted(starts, sp, side='right')-1
        ok=(ti>=0)
        ti=ti[ok]; ss=sp[ok]
        ok=ss < ends[ti]
        ti=ti[ok]; ss=ss[ok]
        bi=np.floor((ss-starts[ti])/BIN_S).astype(np.int64)
        ok=(bi>=0)&(bi<NT); code=ti[ok]*NT+bi[ok]
        if len(code): counts[:,col,:]=np.bincount(code,minlength=ntr*NT).reshape(ntr,NT)
    return counts.astype(np.float32).transpose(0,1,2) / np.float32(BIN_S)

def stim_series(f, abs_centers):
    be=f['acquisition/BehavioralEvents']; out=np.zeros(abs_centers.shape, dtype=bool)
    starts=be['photostim_start_times/timestamps'][()]; stops=be['photostim_stop_times/timestamps'][()]
    for a,b in zip(starts,stops): out |= ((abs_centers>=a)&(abs_centers<b))
    return out.astype(np.float32)

def tongue_outputs(f, abs_centers):
    ts=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
    tt=ts['timestamps'][()]; d=ts['data'][()]; y=d[:,1]; likelihood=d[:,2]
    visible=likelihood>=0.9
    if visible.sum()<2: raise ValueError('Insufficient visible tongue frames')
    p40,p60=np.percentile(y[visible],[40,60])
    flat=abs_centers.ravel(); yi=np.interp(flat,tt,y,left=np.nan,right=np.nan).reshape(abs_centers.shape)
    li=np.interp(flat,tt,likelihood,left=0,right=0).reshape(abs_centers.shape)
    cat=np.full(abs_centers.shape,3,dtype=np.int64)
    v=(li>=0.9)&np.isfinite(yi)
    cat[v & (yi<p40)]=0; cat[v & (yi>=p40) & (yi<=p60)]=1; cat[v & (yi>p60)]=2
    return cat,p40,p60

def convert_session(path, make_plot=False):
    t0=time.time()
    with h5py.File(path,'r') as f:
        good,valid=curate(f)
        if not len(good): return None
        inds=np.where(valid)[0]
        if len(inds)<2: return None
        be=f['acquisition/BehavioralEvents']; go_all=be['go_start_times/timestamps'][()]; go=go_all[inds]
        abs_centers=go[:,None]+CENTERS_REL[None,:]
        tone=final_tone_onsets(f,inds,go_all)
        inp=np.stack([abs_centers-tone[:,None],stim_series(f,abs_centers)],axis=1).astype(np.float32)
        neural=spike_rates(f,good,go)
        # A completely silent population across four seconds indicates a missing neural segment.
        has_neural=np.any(neural != 0, axis=(1,2))
        inds=inds[has_neural]; go=go[has_neural]; abs_centers=abs_centers[has_neural]
        tone=tone[has_neural]; inp=inp[has_neural]; neural=neural[has_neural]
        if len(inds)<2: return None
        trial=f['intervals/trials']; instruction=decode(trial['trial_instruction'][()])[inds]
        outcome_s=decode(trial['outcome'][()])[inds]; early_s=decode(trial['early_lick'][()])[inds]
        choice=np.empty(len(inds),np.int64)
        for k,(ins,o) in enumerate(zip(instruction,outcome_s)):
            if o=='ignore': choice[k]=2
            elif o=='hit': choice[k]=0 if ins=='left' else 1
            elif o=='miss': choice[k]=1 if ins=='left' else 0
            else: raise ValueError(o)
        omap={'ignore':0,'miss':1,'hit':2}; outcome=np.array([omap[x] for x in outcome_s],np.int64)
        early=np.array([1 if x=='early' else 0 for x in early_s],np.int64)
        tongue,p40,p60=tongue_outputs(f,abs_centers)
        output=np.stack([np.repeat(choice[:,None],NT,1),np.repeat(outcome[:,None],NT,1),np.repeat(early[:,None],NT,1),tongue],axis=1)
        regions=decode(f['units/anno_name'][()])[good].tolist()
        subject=re.search(r'sub-([^_]+)',path.name).group(1)
        assert neural.shape==(len(inds),len(good),NT) and inp.shape==(len(inds),2,NT) and output.shape==(len(inds),4,NT)
        assert np.isfinite(neural).all() and np.isfinite(inp).all()
        info={'file':path.name,'source_trial_indices':inds.tolist(),'n_source_trials':len(valid),'n_retained_trials':len(inds),'n_neurons':len(good),'tongue_p40':float(p40),'tongue_p60':float(p60),'seconds':time.time()-t0}
        if make_plot:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True)
            ax[0].imshow(neural[0,:min(100,len(good))],aspect='auto',extent=[OFF_START,OFF_END,min(100,len(good)),0]); ax[0].set_ylabel('neurons')
            ax[1].plot(CENTERS_REL,inp[0,0],label='time from tone'); ax[1].plot(CENTERS_REL,inp[0,1],label='photostim'); ax[1].legend()
            ax[2].step(CENTERS_REL,tongue[0],where='mid'); ax[2].set_ylabel('tongue class')
            ax[3].hist(neural.ravel(),bins=50); ax[3].set_xlabel('firing rate (Hz)')
            fig.suptitle(path.stem); fig.tight_layout(); fig.savefig('/app/processing_'+path.stem+'.png',dpi=130); plt.close(fig)
        return neural,inp,output,regions,subject,info

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    files=sorted(DATA_ROOT.rglob('*.nwb')); files=files[:2] if args.sample else files
    results=[]
    for i,p in enumerate(files):
        r=convert_session(p,args.show_processing and len(results)<2)
        if r is not None: results.append(r); print(f'[{i+1}/{len(files)}] {p.name}: trials={r[5]["n_retained_trials"]}, neurons={r[5]["n_neurons"]}, seconds={r[5]["seconds"]:.2f}',flush=True)
        else: print(f'[{i+1}/{len(files)}] skipped {p.name}',flush=True)
    all_regions=sorted({x for r in results for x in r[3]}); rmap={x:i for i,x in enumerate(all_regions)}
    subjects=sorted({r[4] for r in results}); smap={x:i for i,x in enumerate(subjects)}
    data={'neural':[[x for x in r[0]] for r in results],
          'input':[[x for x in r[1]] for r in results],
          'output':[[x for x in r[2]] for r in results],
          'subjects':subjects,'subject_idx':np.array([smap[r[4]] for r in results],np.int64),
          'brain_regions':all_regions,'brain_region_idx':[np.array([rmap[x] for x in r[3]],np.int64) for r in results],
          'input_names':['time from tone onset (s)','photostimulation on'],
          'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
          'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
          'metadata':{'task_description':'Auditory delayed-response task; decode lick choice, outcome, early licking, and time-varying tongue y category from go-aligned population activity.','time_bin_size':50.0,'temporal_alignment_event':'auditory Go cue onset','off_start':OFF_START,'off_end':OFF_END,'neural_measure':'50-ms spike-count firing rate (Hz)','tongue_visibility_likelihood_threshold':0.9,'neuron_filter':'NWB units classification == good','trial_filter':'observation-interval row present and manually good for every retained unit','session_info':[r[5] for r in results]}}
    with open(args.outpicklefile,'wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
    print('WROTE',args.outpicklefile,'sessions',len(results),'trials',sum(len(x) for x in data['neural']),'neurons',sum(x.shape[0] for x in [s[0] for s in data['neural']]),flush=True)
if __name__=='__main__': main()
