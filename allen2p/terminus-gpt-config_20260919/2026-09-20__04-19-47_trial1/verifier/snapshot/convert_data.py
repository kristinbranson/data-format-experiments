#!/usr/bin/env python3
"""Convert Allen Visual Behavior Ophys NWBs to decoder-compatible trials."""
import argparse, pickle, time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

ROOT = Path('/app/data/visual-behavior-ophys-1.1.0')
NWBDIR = ROOT / 'behavior_ophys_experiments'
META = ROOT / 'project_metadata' / 'ophys_experiment_table.csv'
HZ = 30.0
DT = 1.0 / HZ
IMAGE_NAMES = ['im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_CODE = {x:i+1 for i,x in enumerate(IMAGE_NAMES)}
OUTCOMES = ['hit','miss','false_alarm','correct_reject']

def dec(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)

def interp_rows(times, values, grid):
    """Vectorized linear interpolation of time x feature matrix."""
    j = np.searchsorted(times, grid, side='left')
    j = np.clip(j, 1, len(times)-1)
    lo, hi = j-1, j
    den = times[hi]-times[lo]
    a = np.divide(grid-times[lo], den, out=np.zeros_like(grid), where=den != 0)
    return values[lo] + (values[hi]-values[lo]) * a[:, None]

def interp_1d_finite(times, values, grid):
    ok = np.isfinite(times) & np.isfinite(values)
    if ok.sum() < 2:
        raise ValueError('fewer than two finite samples for interpolation')
    return np.interp(grid, times[ok], values[ok]).astype(np.float32)

def stimulus_table(h):
    for name, g in h['intervals'].items():
        if all(c in g for c in ('image_name','start_time','stop_time','is_change','omitted')):
            return g
    raise KeyError('active natural-image stimulus presentation table not found')

def quintile(values):
    allv = np.concatenate(values)
    edges = np.percentile(allv, [20,40,60,80])
    return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges

def process_experiment(path, row, show=False):
    t0 = time.time()
    with h5py.File(path, 'r') as h:
        if 'acquisition/EyeTracking/pupil_tracking/area' not in h:
            return None, 'missing pupil tracking'
        evg = h['processing/ophys/event_detection']
        ots = np.asarray(evg['timestamps'][:], dtype=np.float64)
        events = np.asarray(evg['data'][:], dtype=np.float32)
        cells = h['processing/ophys/image_segmentation/cell_specimen_table']
        valid = np.asarray(cells['valid_roi'][:], bool) if 'valid_roi' in cells else np.ones(events.shape[1], bool)
        if len(valid) != events.shape[1]:
            raise ValueError(f'ROI/event mismatch in {path.name}')
        events = events[:, valid]
        tr = h['intervals/trials']
        elig = np.asarray(tr['go'][:], bool) | np.asarray(tr['catch'][:], bool)
        trial_idx = np.flatnonzero(elig)
        starts = np.asarray(tr['start_time'][:], float)
        stops = np.asarray(tr['stop_time'][:], float)
        # Filter only impossible non-overlap edge cases; normal data have none.
        trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
        rg = h['processing/running/speed']
        rts, rsp = np.asarray(rg['timestamps'][:],float), np.asarray(rg['data'][:],float)
        eg = h['acquisition/EyeTracking']
        ets = np.asarray(eg['eye_tracking/timestamps'][:],float)
        area = np.asarray(eg['pupil_tracking/area'][:],float)
        diameter = 2.0*np.sqrt(np.maximum(area,0.0)/np.pi)
        sg = stimulus_table(h)
        ss = np.asarray(sg['start_time'][:],float); se = np.asarray(sg['stop_time'][:],float)
        simg = np.array([dec(x) for x in sg['image_name'][:]], object)
        schange = np.asarray(sg['is_change'][:],float) == 1
        somit = np.asarray(sg['omitted'][:],float) == 1
        neural=[]; grids=[]; image_rows=[]; change_rows=[]; run_cont=[]; pupil_cont=[]; outcomes=[]
        origin = ots[0]
        for ti in trial_idx:
            a = max(starts[ti], ots[0]); b = min(stops[ti], ots[-1] + DT)
            k0 = int(np.ceil((a-origin)*HZ - 1e-9)); k1 = int(np.ceil((b-origin)*HZ - 1e-9))
            grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
            if len(grid) < 2: continue
            n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
            image = np.zeros(len(grid), dtype=np.int16)
            change = np.zeros(len(grid), dtype=np.int16)
            # presentations that overlap this trial
            p0 = np.searchsorted(se, grid[0], side='right')
            p1 = np.searchsorted(ss, grid[-1], side='right')
            for pi in range(max(0,p0), min(len(ss),p1+1)):
                if not somit[pi] and simg[pi] in IMAGE_TO_CODE:
                    mask=(grid >= ss[pi]) & (grid < se[pi])
                    image[mask] = IMAGE_TO_CODE[simg[pi]]
                if schange[pi] and not somit[pi] and starts[ti] <= ss[pi] < stops[ti]:
                    q=np.searchsorted(grid,ss[pi],side='left')
                    if q < len(grid): change[q]=1
            out = next((oi for oi,name in enumerate(OUTCOMES) if bool(tr[name][ti])), None)
            if out is None: raise ValueError(f'no outcome for eligible trial {ti} in {path.name}')
            neural.append(n); grids.append(grid); image_rows.append(image); change_rows.append(change)
            run_cont.append(interp_1d_finite(rts,rsp,grid))
            pupil_cont.append(interp_1d_finite(ets,diameter,grid))
            outcomes.append(out)
        run_bins, redges = quintile(run_cont); pupil_bins, pedges = quintile(pupil_cont)
        outputs=[]; inputs=[]
        for i,g in enumerate(grids):
            outcome=np.full(len(g), outcomes[i], dtype=np.int16)
            outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
            inputs.append(np.empty((0,len(g)),dtype=np.float32))
        assert len(neural)==len(outputs)==len(inputs)>=2
        for n,o,inp in zip(neural,outputs,inputs):
            assert n.shape[1]==o.shape[1]==inp.shape[1] and n.shape[0]==valid.sum()
            assert np.isfinite(n).all() and np.isfinite(o).all()
        info=dict(ophys_experiment_id=int(row.ophys_experiment_id),
                  ophys_session_id=int(row.ophys_session_id), behavior_session_id=int(row.behavior_session_id),
                  mouse_id=str(row.mouse_id), targeted_structure=str(row.targeted_structure),
                  imaging_depth=int(row.imaging_depth), cre_line=str(row.cre_line),
                  experience_level=str(row.experience_level), session_type=str(row.session_type),
                  n_neurons=int(valid.sum()), n_trials=len(neural),
                  running_quintile_edges=redges.tolist(), pupil_diameter_quintile_edges=pedges.tolist(),
                  source_file=path.name, seconds=round(time.time()-t0,3))
        plot_payload=(grids[0],neural[0],run_cont[0],pupil_cont[0],outputs[0]) if show else None
        return (neural,inputs,outputs,info,plot_payload), None

def make_plot(payload, sid):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t,n,r,p,o=payload; x=t-t[0]
    fig,ax=plt.subplots(5,1,figsize=(12,10),sharex=True)
    ax[0].imshow(n,aspect='auto',extent=[x[0],x[-1],n.shape[0],0]); ax[0].set_ylabel('event cells')
    ax[1].step(x,o[0],where='post'); ax[1].set_ylabel('image class')
    ax[2].step(x,o[1],where='post'); ax[2].set_ylabel('change')
    ax[3].plot(x,r,label='speed'); ax[3].step(x,o[2],label='quintile'); ax[3].legend(); ax[3].set_ylabel('running')
    ax[4].plot(x,p,label='diameter'); ax[4].step(x,o[3],label='quintile'); ax[4].legend(); ax[4].set_ylabel('pupil'); ax[4].set_xlabel('seconds from trial start')
    fig.suptitle(f'Processing alignment, experiment {sid}'); fig.tight_layout()
    fig.savefig(f'/app/processing_{sid}.png',dpi=140); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--full',action='store_true'); mode.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    meta=pd.read_csv(META); fmap={int(f.stem.rsplit('_',1)[1]):f for f in NWBDIR.glob('*.nwb')}
    meta=meta[meta.ophys_experiment_id.isin(fmap)].sort_values('ophys_experiment_id')
    # deterministic sample with adequate but modest cell counts
    if args.sample: meta=meta.head(2)
    neural=[]; inputs=[]; outputs=[]; infos=[]; excluded=[]; subjects=[]; regions=[]; subject_idx=[]; region_idx=[]
    overall=time.time()
    for j,row in enumerate(meta.itertuples(index=False),1):
        eid=int(row.ophys_experiment_id); print(f'[{j}/{len(meta)}] experiment {eid}',flush=True)
        result,reason=process_experiment(fmap[eid],row,args.show_processing and j<=2)
        if result is None:
            print(f'  EXCLUDED: {reason}',flush=True); excluded.append({'ophys_experiment_id':eid,'reason':reason}); continue
        nn,ii,oo,info,plot=result
        neural.append(nn); inputs.append(ii); outputs.append(oo); infos.append(info)
        mouse=str(row.mouse_id); region=str(row.targeted_structure)
        if mouse not in subjects: subjects.append(mouse)
        if region not in regions: regions.append(region)
        subject_idx.append(subjects.index(mouse)); region_idx.append(np.full(nn[0].shape[0],regions.index(region),dtype=np.int16))
        print(f"  neurons={info['n_neurons']} trials={info['n_trials']} time={info['seconds']}s",flush=True)
        if info['n_trials'] != len(nn): raise AssertionError('trial count mismatch')
        if args.show_processing and j<=2: make_plot(plot,eid)
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,
          'subject_idx':np.asarray(subject_idx,dtype=np.int32),'brain_regions':regions,
          'brain_region_idx':region_idx,'input_names':[],
          'output_names':['image_identity','image_change','running_speed_quintile','pupil_diameter_quintile','trial_outcome'],
          'output_values':[['gray']+IMAGE_NAMES,['no_change','change'],
                           ['Q1_slowest','Q2','Q3','Q4','Q5_fastest'],
                           ['Q1_smallest','Q2','Q3','Q4','Q5_largest'],OUTCOMES],
          'metadata':{'task_description':'Decode visual image identity/change, running-speed and pupil-diameter quintiles, and go/catch trial outcome from detected calcium events.',
                      'time_bin_size':1000.0/HZ,'temporal_alignment_event':'Native trial start on a 30 Hz grid anchored to the experiment ophys timestamps',
                      'off_start':0.0,'off_end':None,'neural_signal':'AllenSDK detected calcium event magnitude',
                      'trial_filter':'go OR catch; aborted and auto-rewarded excluded',
                      'session_unit':'one ophys experiment/imaging plane','session_info':infos,
                      'excluded_sessions':excluded,'image_classes':['gray']+IMAGE_NAMES,
                      'continuous_discretization':'within-experiment quintiles over all retained trial time bins'}}
    print(f'Writing {args.outpicklefile}',flush=True)
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Done: sessions={len(neural)} trials={sum(map(len,neural))} neurons={sum(x[0].shape[0] for x in neural)} elapsed={time.time()-overall:.1f}s',flush=True)
if __name__=='__main__': main()
