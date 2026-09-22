#!/usr/bin/env python3
"""Convert local Allen Visual Behavior Ophys NWB experiments to decoder format."""
import argparse, pickle, time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

DATA_ROOT = Path('/app/data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_ROOT / 'behavior_ophys_experiments'
META_DIR = DATA_ROOT / 'project_metadata'
DT = 0.1
IMAGE_NAMES = ['gray','im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_INT = {x:i for i,x in enumerate(IMAGE_NAMES)}
OUTCOME_KEYS = ['hit','miss','false_alarm','correct_reject']
NO_PUPIL = {795953296,806456687,833631914}


def exp_id(path): return int(path.stem.rsplit('_', 1)[1])


def interp_matrix(ts, data, q):
    """Interpolate time x feature data; q must lie within ts."""
    idx = np.searchsorted(ts, q, side='left')
    idx = np.clip(idx, 1, len(ts)-1)
    lo, hi = idx-1, idx
    w = ((q-ts[lo])/(ts[hi]-ts[lo])).astype(np.float32)
    return data[lo] + (data[hi]-data[lo])*w[:,None]


def interp_vector_finite(ts, values, q):
    ok = np.isfinite(ts) & np.isfinite(values)
    if ok.sum() < 2 or q[0] < ts[ok][0] or q[-1] > ts[ok][-1]:
        return None
    return np.interp(q, ts[ok], values[ok])


def quantile_codes(values, valid_mask, n=5):
    """Session-level equal-frequency bins, robust to repeated quantiles."""
    fit = values[valid_mask]
    if fit.size == 0 or not np.all(np.isfinite(fit)):
        raise ValueError('No finite values for quantile fit')
    edges = np.quantile(fit, np.arange(1,n)/n)
    return np.searchsorted(edges, values, side='right').astype(np.int16), edges


def stimulus_info(f):
    pg = next(iter(f['stimulus/presentation'].values()))
    tg = next(iter(f['stimulus/templates'].values()))
    desc = [x.decode() if isinstance(x,bytes) else str(x) for x in tg['control_description'][:]]
    controls = np.asarray(tg['control'][:], int)
    lookup = {int(c): IMAGE_TO_INT[d] for c,d in zip(controls,desc)}
    if len(lookup) != 8 or any(v == 0 for v in lookup.values()): raise ValueError('Unexpected stimulus template/control mapping')
    return np.asarray(pg['timestamps'][:],float), np.asarray(pg['data'][:],int), lookup


def image_labels(q, stim_ts, stim_val, lookup):
    out = np.zeros(len(q), dtype=np.int16)
    j = np.searchsorted(stim_ts, q, side='right')-1
    ok = (j>=0) & ((q-stim_ts[np.clip(j,0,len(stim_ts)-1)]) < 0.250)
    jj = j[ok]
    out[ok] = np.array([lookup.get(int(v),0) for v in stim_val[jj]], dtype=np.int16)
    return out


def trial_grid(start, stop):
    # Half-open trial, 100 ms centers. Avoid floating endpoint ambiguity.
    n = int(np.floor((stop-start)/DT + 1e-9))
    return start + DT/2 + DT*np.arange(n, dtype=float)


def process_experiment(path, meta_row, make_plot=False):
    eid=exp_id(path); t0=time.perf_counter(); dropped={'support':0,'nonfinite':0}
    with h5py.File(path,'r') as f:
        tr=f['intervals/trials']
        eligible=(~np.asarray(tr['aborted'][:],bool) & ~np.asarray(tr['auto_rewarded'][:],bool)
                  & (np.asarray(tr['go'][:],bool)|np.asarray(tr['catch'][:],bool)))
        inds=np.flatnonzero(eligible)
        nts=np.asarray(f['processing/ophys/dff/traces/timestamps'][:],float)
        nds=np.asarray(f['processing/ophys/dff/traces/data'][:],dtype=np.float32)
        rts=np.asarray(f['processing/running/speed/timestamps'][:],float)
        rvs=np.asarray(f['processing/running/speed/data'][:],float)
        ets=np.asarray(f['acquisition/EyeTracking/eye_tracking/timestamps'][:],float)
        area=np.asarray(f['acquisition/EyeTracking/pupil_tracking/area'][:],float)
        diam=2*np.sqrt(np.maximum(area,0)/np.pi)
        sts,sval,lookup=stimulus_info(f)
        starts=np.asarray(tr['start_time'][:],float); stops=np.asarray(tr['stop_time'][:],float)
        changes=np.asarray(tr['change_time'][:],float)
        is_changes=np.asarray(tr['is_change'][:],bool)
        outcome_flags=np.vstack([np.asarray(tr[k][:],bool) for k in OUTCOME_KEYS])
        records=[]; allrun=[]; allpupil=[]
        for ii in inds:
            q=trial_grid(starts[ii],stops[ii])
            if len(q)<2 or q[0]<nts[0] or q[-1]>nts[-1] or q[0]<rts[0] or q[-1]>rts[-1]:
                dropped['support']+=1; continue
            run=interp_vector_finite(rts,rvs,q); pup=interp_vector_finite(ets,diam,q)
            if run is None or pup is None or not np.all(np.isfinite(run)) or not np.all(np.isfinite(pup)):
                dropped['nonfinite']+=1; continue
            flags=outcome_flags[:,ii]
            if flags.sum()!=1: raise AssertionError(f'{eid} trial {ii}: outcome not exclusive')
            records.append((ii,q,run,pup,int(np.argmax(flags))))
            allrun.append(run); allpupil.append(pup)
        if len(records)<2: raise ValueError(f'{eid}: fewer than 2 usable trials')
        rv=np.concatenate(allrun); pv=np.concatenate(allpupil)
        rcodes,redges=quantile_codes(rv,np.ones(rv.size,bool)); pcodes,pedges=quantile_codes(pv,np.ones(pv.size,bool))
        neural=[]; inputs=[]; outputs=[]; raw_plot=None; roff=poff=0
        for ii,q,run,pup,outcome in records:
            n=len(q); neu=interp_matrix(nts,nds,q).T.astype(np.float32)
            img=image_labels(q,sts,sval,lookup)
            ch=np.zeros(n,dtype=np.int16)
            if is_changes[ii] and np.isfinite(changes[ii]):
                k=np.searchsorted(q,changes[ii],side='left')
                if k<n: ch[k]=1
            if not is_changes[ii]: assert ch.sum()==0
            rr=rcodes[roff:roff+n]; pp=pcodes[poff:poff+n]; roff+=n; poff+=n
            out=np.vstack([img,ch,rr,pp,np.full(n,outcome,dtype=np.int16)])
            assert neu.shape[1]==out.shape[1]==n
            assert img.min()>=0 and img.max()<len(IMAGE_NAMES) and rr.min()>=0 and rr.max()<5 and pp.min()>=0 and pp.max()<5
            neural.append(neu); inputs.append(np.empty((0,n),dtype=np.float32)); outputs.append(out)
            if raw_plot is None: raw_plot=(q,neu,img,ch,run,pup,rr,pp)
    if make_plot:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        q,neu,img,ch,run,pup,rr,pp=raw_plot; x=q-q[0]
        fig,ax=plt.subplots(6,1,figsize=(12,12),sharex=True)
        ax[0].imshow(neu[:min(30,len(neu))],aspect='auto',extent=[x[0],x[-1],min(30,len(neu)),0]); ax[0].set_ylabel('dF/F cells')
        ax[1].step(x,img,where='mid'); ax[1].set_ylabel('image class')
        ax[2].step(x,ch,where='mid'); ax[2].set_ylabel('change')
        ax[3].plot(x,run); ax[3].step(x,rr,where='mid'); ax[3].set_ylabel('run / quintile')
        ax[4].plot(x,pup); ax[4].step(x,pp,where='mid'); ax[4].set_ylabel('pupil / quintile')
        ax[5].plot(np.diff(q)*1000); ax[5].set_ylabel('grid dt ms'); ax[5].set_xlabel('trial time bin')
        fig.suptitle(f'Experiment {eid}: raw/interpolated streams and categorical outputs'); fig.tight_layout()
        fig.savefig(f'/app/processing_{eid}.png',dpi=130); plt.close(fig)
    info={'ophys_experiment_id':eid,'ophys_session_id':str(meta_row.ophys_session_id),'mouse_id':str(meta_row.mouse_id),
          'session_type':str(meta_row.session_type),'targeted_structure':str(meta_row.targeted_structure),
          'imaging_depth':int(meta_row.imaging_depth),'n_neurons':neural[0].shape[0],
          'source_eligible_trials':int(eligible.sum()),'retained_trials':len(records),'dropped_trials':dropped,
          'running_quintile_edges':redges.tolist(),'pupil_diameter_quintile_edges':pedges.tolist(),
          'conversion_seconds':round(time.perf_counter()-t0,3)}
    print(f"{eid}: cells={info['n_neurons']} trials={len(records)}/{eligible.sum()} dropped={dropped} time={info['conversion_seconds']}s",flush=True)
    return neural,inputs,outputs,info


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); g=ap.add_mutually_exclusive_group()
    g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true')
    args=ap.parse_args(); start=time.perf_counter()
    paths=sorted(NWB_DIR.glob('*.nwb'),key=exp_id); paths=[p for p in paths if exp_id(p) not in NO_PUPIL]
    # Sample one fast single-plane and one slow multiplane experiment, both with pupil.
    if args.sample:
        scan=pd.read_csv('/tmp/nwb_scan.csv') if Path('/tmp/nwb_scan.csv').exists() else None
        if scan is not None:
            valid=scan[~scan.experiment.isin(NO_PUPIL)].copy(); valid['rate_group']=np.where(valid.dt_ms<60,'fast','slow'); ids=[int(valid[valid.rate_group==g].sort_values(['neurons','experiment'],ascending=[False,True]).iloc[0].experiment) for g in ['fast','slow']]
            paths=[NWB_DIR/f'behavior_ophys_experiment_{i}.nwb' for i in dict.fromkeys(ids)]
        else: paths=paths[:2]
    table=pd.read_csv(META_DIR/'ophys_experiment_table.csv').set_index('ophys_experiment_id')
    neural=[]; inputs=[]; outputs=[]; infos=[]; regions=[]; mice=[]
    for si,p in enumerate(paths):
        eid=exp_id(p); row=table.loc[eid]
        n,i,o,info=process_experiment(p,row,args.show_processing and si<2)
        neural.append(n); inputs.append(i); outputs.append(o); infos.append(info); regions.append(str(row.targeted_structure)); mice.append(str(row.mouse_id))
    subjects=sorted(set(mice)); brain_regions=sorted(set(regions))
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,
          'subject_idx':np.asarray([subjects.index(x) for x in mice],dtype=np.int64),
          'brain_regions':brain_regions,
          'brain_region_idx':[np.full(sess[0].shape[0],brain_regions.index(r),dtype=np.int64) for sess,r in zip(neural,regions)],
          'input_names':[],
          'output_names':['image_identity','image_change','running_speed_quintile','pupil_diameter_quintile','trial_outcome'],
          'output_values':[IMAGE_NAMES,['no_change','change'],['Q1_lowest','Q2','Q3','Q4','Q5_highest'],
                           ['Q1_smallest','Q2','Q3','Q4','Q5_largest'],OUTCOME_KEYS],
          'metadata':{'task_description':'Decode visible image identity, image changes, running-speed quintile, pupil-diameter quintile, and trial outcome from released dF/F during valid go/catch trials.',
                      'time_bin_size':100.0,'temporal_alignment_event':'native trial start time on absolute ophys clock',
                      'off_start':0.0,'off_end':None,'output_static':[False,False,False,False,True],
                      'neural_signal':'Allen released dF/F','source_release':'visual-behavior-ophys-1.1.0',
                      'trial_filter':'(go OR catch) AND NOT aborted AND NOT auto_rewarded',
                      'grid':'100 ms centers in half-open [trial_start, trial_stop)',
                      'pupil_transform':'equivalent-circle diameter = 2*sqrt(processed_area/pi)',
                      'excluded_no_pupil_experiments':sorted(NO_PUPIL),'session_info':infos}}
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print(f'Saved {args.outpicklefile}: sessions={len(neural)} trials={sum(map(len,neural))} neurons={sum(x[0].shape[0] for x in neural)} size={Path(args.outpicklefile).stat().st_size} bytes total_time={time.perf_counter()-start:.2f}s',flush=True)

if __name__=='__main__': main()
