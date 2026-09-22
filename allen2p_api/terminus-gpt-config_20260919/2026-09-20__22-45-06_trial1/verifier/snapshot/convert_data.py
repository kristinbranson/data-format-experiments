#!/usr/bin/env python3
"""Convert cached Allen Visual Behavior ophys experiments to decoder format.

All NWB-backed data access is exclusively through VisualBehaviorOphysProjectCache.
"""
import argparse, pickle, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path('/app/data')
BIN_S = 0.100
IMAGE_VALUES = ['gray','im000','im031','im035','im045','im054','im061','im062',
                'im063','im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_ID = {v:i for i,v in enumerate(IMAGE_VALUES)}
OUTCOME_COLS = ['hit','miss','false_alarm','correct_reject']
REGIONS = ['VISp','VISl']


def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))


def local_experiment_ids():
    paths = CACHE_DIR.glob('visual-behavior-ophys-*/behavior_ophys_experiments/behavior_ophys_experiment_*.nwb')
    return sorted(int(re.search(r'(\d+)\.nwb$', p.name).group(1)) for p in paths)


def selected_table(cache):
    tab = cache.get_ophys_experiment_table()
    ids = tab.index.intersection(local_experiment_ids())
    tab = tab.loc[ids]
    tab = tab[tab['behavior_type'].eq('active_behavior')].sort_index()
    return tab


def interp_valid(times, values, query):
    times = np.asarray(times, float); values = np.asarray(values, float)
    good = np.isfinite(times) & np.isfinite(values)
    if good.sum() < 2:
        return None
    t, idx = np.unique(times[good], return_index=True)
    v = values[good][idx]
    if len(t) < 2:
        return None
    out = np.interp(query, t, v, left=np.nan, right=np.nan)
    return out


def quintile_edges(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return np.nanpercentile(v, [20,40,60,80])


def labels_from_edges(values, edges, fill):
    v = np.asarray(values, float).copy()
    missing = ~np.isfinite(v)
    v[missing] = fill
    return np.digitize(v, edges, right=False).astype(np.int64), int(missing.sum())


def prepare_trials(exp):
    tr = exp.trials.copy()
    mask = (tr['go'].astype(bool) | tr['catch'].astype(bool))
    mask &= ~tr['aborted'].astype(bool) & ~tr['auto_rewarded'].astype(bool)
    tr = tr.loc[mask].copy()
    exact = tr[OUTCOME_COLS].astype(int).sum(axis=1).eq(1)
    if (~exact).any():
        print(f'  warning: dropping {(~exact).sum()} trials without exactly one outcome')
        tr = tr.loc[exact]
    return tr


def make_grids(trials):
    grids=[]
    for idx,row in trials.iterrows():
        n = int(np.floor((float(row.stop_time)-float(row.start_time))/BIN_S + 1e-9))
        if n < 1: continue
        edges = float(row.start_time) + np.arange(n+1)*BIN_S
        centers = edges[:-1] + BIN_S/2
        grids.append((idx, edges, centers))
    return grids


def cumulative_events(event_matrix):
    # One cumulative array per experiment; avoids recomputing it for every trial.
    cs = np.empty((event_matrix.shape[0], event_matrix.shape[1]+1), dtype=np.float64)
    cs[:,0] = 0.0
    # float64 prevents cancellation error when late-bin sums subtract large cumulative totals.
    np.cumsum(event_matrix, axis=1, dtype=np.float64, out=cs[:,1:])
    return cs

def bin_events(event_times, event_cumsum, edges):
    # Sum each cell's detected event magnitudes in [edge_i, edge_i+1).
    lo = np.searchsorted(event_times, edges[:-1], side='left')
    hi = np.searchsorted(event_times, edges[1:], side='left')
    return (event_cumsum[:,hi] - event_cumsum[:,lo]).astype(np.float32, copy=False)


def image_and_change(sp, centers, edges):
    image = np.zeros(len(centers), dtype=np.int64)
    change = np.zeros(len(centers), dtype=np.int64)
    # Presentation intervals are non-overlapping; omitted rows stay gray.
    for r in sp.itertuples():
        name = r.image_name
        omitted = bool(r.omitted) if pd.notna(r.omitted) else False
        if not omitted and pd.notna(name) and str(name) in IMAGE_TO_ID:
            a=np.searchsorted(centers, float(r.start_time), side='left')
            b=np.searchsorted(centers, float(r.end_time), side='left')
            image[a:b] = IMAGE_TO_ID[str(name)]
        if bool(r.is_change):
            # 'Right after change': label the 400 ms post-onset response window
            # used by the reference paper's image-change decoder.
            a=np.searchsorted(centers, float(r.start_time), side='left')
            b=np.searchsorted(centers, float(r.start_time)+0.400, side='left')
            change[a:b]=1
    return image,change


def process_experiment(cache, eid, row, show=False):
    t0=time.time(); exp=cache.get_behavior_ophys_experiment(int(eid))
    trials=prepare_trials(exp); grids=make_grids(trials)
    if len(grids)<2: return None, f'fewer than 2 eligible trials ({len(grids)})'
    ts=np.asarray(exp.ophys_timestamps,float)
    evdf=exp.events
    cell_ids=exp.cell_specimen_table.index.to_numpy()
    evdf=evdf.loc[cell_ids]
    events=np.stack(evdf['events'].to_numpy()).astype(np.float32)
    if events.shape != (len(cell_ids),len(ts)):
        return None, f'event shape mismatch {events.shape} vs {len(cell_ids),len(ts)}'
    event_cs=cumulative_events(events)
    del events
    sp=exp.stimulus_presentations
    sp=sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection',na=False)].copy()
    run=exp.running_speed
    eye=exp.eye_tracking
    pupil=2.0*np.maximum(eye['pupil_width'].to_numpy(float),eye['pupil_height'].to_numpy(float))
    all_centers=np.concatenate([g[2] for g in grids])
    all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
    all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
    if all_run is None: return None,'insufficient running data'
    if all_pupil is None: return None,'insufficient processed pupil data'
    run_edges=quintile_edges(all_run); pupil_edges=quintile_edges(all_pupil)
    if run_edges is None or pupil_edges is None: return None,'cannot estimate quintiles'
    run_fill=float(np.nanmedian(all_run)); pupil_fill=float(np.nanmedian(all_pupil))
    neural=[]; inputs=[]; outputs=[]; plot_info=[]; pos=0; miss_run=miss_pupil=0
    for tidx,edges,centers in grids:
        T=len(centers); rowt=trials.loc[tidx]
        n=bin_events(ts,event_cs,edges)
        im,ch=image_and_change(sp,centers,edges)
        rv=all_run[pos:pos+T]; pv=all_pupil[pos:pos+T]; pos+=T
        rb,mr=labels_from_edges(rv,run_edges,run_fill)
        pb,mp=labels_from_edges(pv,pupil_edges,pupil_fill)
        miss_run+=mr; miss_pupil+=mp
        outcome=int(np.flatnonzero(rowt[OUTCOME_COLS].to_numpy(bool))[0])
        out=np.vstack([im,ch,rb,pb,np.full(T,outcome,dtype=np.int64)])
        assert n.shape[1]==T and out.shape==(5,T) and np.isfinite(n).all()
        neural.append(n); inputs.append(np.empty((0,T),dtype=np.float32)); outputs.append(out)
        if len(plot_info)<3: plot_info.append((centers,n,out,rv,pv))
    result=dict(neural=neural,input=inputs,output=outputs,cell_ids=cell_ids,
                run_edges=run_edges,pupil_edges=pupil_edges,missing_run=miss_run,
                missing_pupil=miss_pupil,plot_info=plot_info,
                n_bins=sum(x.shape[1] for x in neural),load_seconds=time.time()-t0)
    if show: plot_processing(eid,result)
    return result,None


def plot_processing(eid,res):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(len(res['plot_info']),4,figsize=(16,3.2*len(res['plot_info'])),squeeze=False)
    for i,(t,n,o,rv,pv) in enumerate(res['plot_info']):
        rel=t-t[0]
        axs[i,0].imshow(n,aspect='auto',interpolation='nearest',extent=[rel[0],rel[-1],n.shape[0],0]); axs[i,0].set_title('event magnitude')
        axs[i,1].step(rel,o[0],where='mid',label='image'); axs[i,1].step(rel,o[1]*16,where='mid',label='change x16'); axs[i,1].legend(); axs[i,1].set_title('stimulus labels')
        axs[i,2].plot(rel,rv,label='speed'); axs[i,2].step(rel,o[2],where='mid',label='quintile'); axs[i,2].legend(); axs[i,2].set_title('running')
        axs[i,3].plot(rel,pv,label='diameter'); axs[i,3].step(rel,o[3],where='mid',label='quintile'); axs[i,3].legend(); axs[i,3].set_title(f'pupil; outcome={o[4,0]}')
        for ax in axs[i]: ax.set_xlabel('seconds from trial start')
    fig.suptitle(f'Processing experiment {eid}'); fig.tight_layout(); fig.savefig(f'/app/processing_{eid}.png',dpi=130); plt.close(fig)


_WORKER_CACHE = None

def init_worker():
    global _WORKER_CACHE
    warnings.filterwarnings('ignore')
    _WORKER_CACHE = get_cache()

def worker_process(task):
    eid, row_dict = task
    row = pd.Series(row_dict)
    res, err = process_experiment(_WORKER_CACHE, eid, row, show=False)
    return eid, row_dict, res, err


def validate(data):
    ns=len(data['neural'])
    assert ns==len(data['input'])==len(data['output'])==len(data['subject_idx'])==len(data['brain_region_idx'])
    assert data['input_names']==[] and len(data['output_names'])==5
    for s in range(ns):
        assert len(data['neural'][s])>=2
        assert len(data['neural'][s])==len(data['input'][s])==len(data['output'][s])
        assert len(data['brain_region_idx'][s])==data['neural'][s][0].shape[0]
        for n,i,o in zip(data['neural'][s],data['input'][s],data['output'][s]):
            T=n.shape[1]; assert i.shape==(0,T) and o.shape==(5,T)
            assert np.isfinite(n).all() and np.isfinite(o).all()
            assert o[0].min()>=0 and o[0].max()<17 and o[1].min()>=0 and o[1].max()<2
            assert o[2].min()>=0 and o[2].max()<5 and o[3].min()>=0 and o[3].max()<5
            assert o[4].min()>=0 and o[4].max()<4 and np.all(o[4]==o[4,0])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group(); mode.add_argument('--full',action='store_true'); mode.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true')
    args=ap.parse_args()
    warnings.filterwarnings('ignore',category=FutureWarning)
    cache=get_cache(); tab=selected_table(cache)
    if args.sample: tab=tab.iloc[:2]
    print(f'Selected {len(tab)} active local experiments ({"sample" if args.sample else "full"})')
    subjects=sorted(tab.mouse_id.astype(str).unique()); subjmap={x:i for i,x in enumerate(subjects)}
    data={'neural':[],'input':[],'output':[],'subjects':subjects,'subject_idx':[],
          'brain_regions':REGIONS,'brain_region_idx':[],'input_names':[],
          'output_names':['image_identity','image_change','running_speed_bin','pupil_diameter_bin','trial_outcome'],
          'output_values':[IMAGE_VALUES,['no_change','change'],['Q1','Q2','Q3','Q4','Q5'],['Q1','Q2','Q3','Q4','Q5'],OUTCOME_COLS],
          'metadata':{'task_description':'Visual change-detection: decode image identity, real image changes, running/pupil quintiles, and go/catch trial outcome from detected calcium events.',
                      'time_bin_size':100.0,'temporal_alignment_event':'Native SDK trial start; 100 ms bins defined in synchronized ophys timestamp coordinates.',
                      'off_start':None,'off_end':None,'neural_signal':'AllenSDK detected calcium event magnitude summed per bin',
                      'trial_filter':'go or catch; aborted and auto-rewarded excluded','stimulus_block':'change_detection only',
                      'image_values':IMAGE_VALUES,'outcome_values':OUTCOME_COLS,'session_info':[]}}
    total_t=time.time(); skipped=[]
    tasks=[(int(eid),row.to_dict()) for eid,row in tab.iterrows()]
    pool=None
    if args.sample:
        result_iter=((eid,rowdict,*process_experiment(cache,eid,pd.Series(rowdict),show=args.show_processing and k<=2))
                     for k,(eid,rowdict) in enumerate(tasks,1))
    else:
        import multiprocessing as mp
        nworkers=min(8,len(tasks))
        print(f'Using {nworkers} experiment workers',flush=True)
        pool=mp.get_context('spawn').Pool(nworkers,initializer=init_worker)
        result_iter=pool.imap(worker_process,tasks,chunksize=1)
    try:
        for k,(eid,rowdict,res,err) in enumerate(result_iter,1):
            row=pd.Series(rowdict)
            print(f'[{k}/{len(tab)}] experiment {eid}',flush=True)
            if err:
                print('  SKIP:',err); skipped.append((int(eid),err)); continue
            data['neural'].append(res['neural']); data['input'].append(res['input']); data['output'].append(res['output'])
            data['subject_idx'].append(subjmap[str(row.mouse_id)])
            ridx=REGIONS.index(str(row.targeted_structure)); data['brain_region_idx'].append(np.full(len(res['cell_ids']),ridx,dtype=np.int64))
            outcomes=np.bincount([int(o[4,0]) for o in res['output']],minlength=4).tolist()
            info={'ophys_experiment_id':int(eid),'ophys_session_id':int(row.ophys_session_id),'behavior_session_id':int(row.behavior_session_id),
                  'mouse_id':str(row.mouse_id),'cre_line':str(row.cre_line),'targeted_structure':str(row.targeted_structure),
                  'session_type':str(row.session_type),'experience_level':str(row.experience_level),'n_cells':len(res['cell_ids']),
                  'n_trials':len(res['neural']),'n_bins':res['n_bins'],'outcome_counts':outcomes,
                  'running_quintile_edges':res['run_edges'].tolist(),'pupil_quintile_edges':res['pupil_edges'].tolist(),
                  'imputed_running_bins':res['missing_run'],'imputed_pupil_bins':res['missing_pupil'],'processing_seconds':res['load_seconds']}
            data['metadata']['session_info'].append(info)
            print(f"  cells={info['n_cells']} trials={info['n_trials']} bins={info['n_bins']} pupil_imputed={info['imputed_pupil_bins']} sec={info['processing_seconds']:.1f}",flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    data['subject_idx']=np.asarray(data['subject_idx'],dtype=np.int64)
    data['metadata']['skipped_experiments']=skipped
    # Remove subjects not represented after any skips and remap indices.
    used=sorted(set(data['subject_idx'].tolist()))
    remap={old:new for new,old in enumerate(used)}
    data['subjects']=[subjects[i] for i in used]
    data['subject_idx']=np.asarray([remap[i] for i in data['subject_idx']],dtype=np.int64)
    validate(data)
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    ntr=sum(map(len,data['neural'])); nc=sum(x[0].shape[0] for x in data['neural'])
    print(f'Wrote {args.outpicklefile}: sessions={len(data["neural"])} trials={ntr} summed_cells={nc} size={Path(args.outpicklefile).stat().st_size/1e6:.1f} MB total_sec={time.time()-total_t:.1f}')

if __name__=='__main__': main()
