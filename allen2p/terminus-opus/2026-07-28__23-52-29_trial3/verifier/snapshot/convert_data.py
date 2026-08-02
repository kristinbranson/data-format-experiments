#!/usr/bin/env python3
"""Convert Allen Brain Observatory Visual Behavior 2P data to decoder format."""
import sys, os
sys.path.insert(0, 'code')
import argparse, time, glob, pickle, gc, warnings, h5py
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pynwb
from scipy import interpolate

TARGET_BIN_SIZE = 0.093

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('output_file', type=str)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', default=True)
    g.add_argument('--sample', action='store_true')
    p.add_argument('--show-processing', action='store_true')
    return p.parse_args()

def get_experiment_list(sample=False):
    et = pd.read_csv('data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
    nwb_ids = set(int(f.split('experiment_')[1].split('.nwb')[0]) 
                  for f in glob.glob('data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb'))
    m = et[et['ophys_experiment_id'].isin(nwb_ids)]
    a = m[m['passive']==False].copy().sort_values('ophys_experiment_id').reset_index(drop=True)
    if sample: a = a.head(2)
    print(f"Selected {len(a)} experiments, {a['mouse_id'].nunique()} mice")
    print(f"  Equipment: {a['equipment_name'].value_counts().to_dict()}")
    return a

def fast_collect_stats(exp_list):
    """Use h5py for fast collection of image names and running/pupil stats."""
    img_set = set()
    rs_all, pa_all = [], []
    for i, (_, row) in enumerate(exp_list.iterrows()):
        eid = row['ophys_experiment_id']
        path = f'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_{int(eid)}.nwb'
        with h5py.File(path, 'r') as f:
            # Image names
            for k in f['intervals']:
                if 'image_name' in f['intervals'][k]:
                    names = f['intervals'][k]['image_name'][:]
                    names = [x.decode() if isinstance(x, bytes) else x for x in names]
                    img_set.update(x for x in names if x != 'omitted' and x != '' and not pd.isna(x))
            # Running speed (subsample)
            rs = f['processing']['running']['speed']['data'][::10]
            rs_all.append(rs)
            # Pupil area from NWB
            try:
                pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
                pa_all.append(np.array(pa, dtype=np.float64))
            except:
                pass
        if (i+1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(exp_list)}] stats collected")
    
    all_img = sorted(img_set)
    rs_cat = np.concatenate(rs_all).astype(np.float64)
    pa_cat = np.concatenate(pa_all).astype(np.float64) if pa_all else np.array([0.0])
    return all_img, rs_cat, pa_cat

def load_experiment(eid):
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment
    path = f'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_{int(eid)}.nwb'
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        ds = BehaviorOphysExperiment.from_nwb(nwbfile=nwb)
        return {
            'ophys_timestamps': ds.ophys_timestamps.copy(),
            'events': np.vstack(ds.events.events.values).astype(np.float32),
            'trials': ds.trials.copy(),
            'stimulus_presentations': ds.stimulus_presentations.copy(),
            'running_speed': ds.running_speed.copy(),
            'eye_tracking': ds.eye_tracking.copy(),
            'metadata': dict(ds.metadata),
        }

def resample_session(events, ophys_ts, bc):
    h = TARGET_BIN_SIZE / 2
    nn, nb = events.shape[0], len(bc)
    out = np.zeros((nn, nb), dtype=np.float32)
    li = np.searchsorted(ophys_ts, bc - h, side='left')
    ri = np.searchsorted(ophys_ts, bc + h, side='left')
    for b in range(nb):
        if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
        elif b > 0: out[:, b] = out[:, b-1]
    return out

def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    if len(v2) == 0: return np.linspace(-1, 1, n+1)
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e

def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)

def process_experiment(ed, imn, rbe, pbe):
    ots = ed['ophys_timestamps']
    ev = ed['events']
    tr = ed['trials']
    sp = ed['stimulus_presentations']
    run = ed['running_speed']
    eye = ed['eye_tracking']
    nn = ev.shape[0]
    n2i = {n: i for i, n in enumerate(imn)}
    
    vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
    if len(vt) == 0: return [], [], nn
    
    s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
    bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
    nb = len(bc)
    if nb < 2: return [], [], nn
    
    nr = resample_session(ev, ots, bc)
    
    # Running
    rts, rsp = run['timestamps'].values, run['speed'].values
    vm = ~np.isnan(rsp)
    ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
    rb = dig(ri, rbe)
    
    # Pupil
    ets, pa = eye['timestamps'].values, eye['pupil_area'].values
    vm2 = ~np.isnan(pa)
    pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
    pb = dig(pi, pbe)
    
    # Image identity
    cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
    cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
    ist = cdi['start_time'].values
    iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
    sidx = np.searchsorted(ist, bc, side='right') - 1
    ib = np.zeros(nb, dtype=np.int64)
    m = sidx >= 0
    ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
    
    # Change
    cts = cd[cd['is_change']==True]['start_time'].values
    cb = np.zeros(nb, dtype=np.int64)
    for ct in cts:
        cb[(bc>=ct)&(bc<ct+0.750)] = 1
    
    ntl, otl = [], []
    for _, t in vt.iterrows():
        tm = (bc>=t['start_time'])&(bc<t['stop_time'])
        ti = np.where(tm)[0]
        if len(ti) < 2: continue
        oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
        ntl.append(nr[:, ti])
        otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
    return ntl, otl, nn

def plot_processing(ed, ntl, otl, eid, imn):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'Exp {eid}', fontsize=16)
    ev, ots = ed['events'], ed['ophys_timestamps']
    tr = ed['trials']
    vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
    for i in range(min(5, ev.shape[0])):
        ax[0,0].plot(ots[:5000], ev[i,:5000]+i*0.5, alpha=0.7, lw=0.5)
    ax[0,0].set_title('Neural events')
    for _, t in vt.head(20).iterrows():
        c = 'green' if t['hit'] else ('red' if t['miss'] else ('orange' if t['false_alarm'] else 'blue'))
        ax[0,1].axvspan(t['start_time'], t['stop_time'], alpha=0.3, color=c)
    ax[0,1].set_title('Trials')
    if len(ntl) >= 2:
        for ti in range(min(2, len(ntl))):
            nt, ot = ntl[ti], otl[ti]
            ta = np.arange(nt.shape[1]) * TARGET_BIN_SIZE
            ax[1,ti].plot(ta, nt[:min(5,nt.shape[0])].T, alpha=0.7, lw=0.5); ax[1,ti].set_title(f'T{ti}: Neural')
            ax[2,ti].plot(ta, ot[0], 'b-'); ax[2,ti].set_title(f'T{ti}: ImgID')
            ax[3,ti].plot(ta, ot[1], 'r-'); ax[3,ti].set_title(f'T{ti}: Change')
            ax[4,ti].plot(ta, ot[2], 'g-'); ax[4,ti].set_title(f'T{ti}: Running')
            ax[5,ti].plot(ta, ot[3], 'm-'); ax[5,ti].set_title(f'T{ti}: Pupil')
    plt.tight_layout()
    fig.savefig(f'processing_{eid}.png', dpi=100); plt.close(fig)
    print(f'  Saved processing_{eid}.png')

def main():
    args = parse_args()
    t0g = time.time()
    sample = args.sample
    show = args.show_processing
    outf = args.output_file
    print(f"Mode: {'sample' if sample else 'full'}, Output: {outf}, Bin: {TARGET_BIN_SIZE*1000:.1f}ms")
    
    el = get_experiment_list(sample=sample)
    subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
    regions = sorted(set(r['targeted_structure'] for _, r in el.iterrows()))
    s2i = {s: i for i, s in enumerate(subjects)}
    r2i = {r: i for i, r in enumerate(regions)}
    print(f"Subjects: {len(subjects)}, Regions: {regions}")
    
    # Fast stats collection with h5py
    print(f"\n=== Fast stats collection ===")
    t_stats = time.time()
    imn, rs_all, pa_all = fast_collect_stats(el)
    rbe = pct_bins(rs_all, 5)
    pbe = pct_bins(pa_all, 5)
    del rs_all, pa_all
    print(f"Images ({len(imn)}): {imn}")
    print(f"Running bins: {rbe}")
    print(f"Pupil bins: {pbe}")
    print(f"Stats time: {time.time()-t_stats:.1f}s")
    
    # Process all experiments
    print(f"\n=== Processing experiments ===")
    an, ai, ao, asi, abri = [], [], [], [], []
    skipped = 0
    
    for i, (_, row) in enumerate(el.iterrows()):
        eid = row['ophys_experiment_id']
        t0 = time.time()
        ed = load_experiment(eid)
        tl = time.time()
        ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
        tp = time.time()
        if show and len(an) < 2:
            plot_processing(ed, ntl, otl, eid, imn)
        del ed; gc.collect()
        if len(ntl) < 2:
            print(f"  [{i+1}/{len(el)}] Exp {eid}: SKIPPED")
            skipped += 1; continue
        it = [np.zeros((0,), dtype=np.float32) for _ in ntl]
        an.append(ntl); ai.append(it); ao.append(otl)
        asi.append(s2i[str(row['mouse_id'])])
        abri.append(np.full(nn, r2i[row['targeted_structure']], dtype=np.int64))
        print(f"  [{i+1}/{len(el)}] Exp {eid}: {nn}n {len(ntl)}t load={tl-t0:.1f}s proc={tp-tl:.1f}s")
    
    on = ['image_identity','image_change','running_speed','pupil_diameter','trial_outcome']
    ov = [imn, ['no_change','change'], [f'bin_{i}' for i in range(5)],
          [f'bin_{i}' for i in range(5)], ['hit','miss','false_alarm','correct_reject']]
    data = {
        'neural': an, 'input': ai, 'output': ao,
        'subjects': subjects, 'subject_idx': np.array(asi, dtype=np.int64),
        'brain_regions': regions, 'brain_region_idx': abri,
        'input_names': [], 'output_names': on, 'output_values': ov,
        'metadata': {
            'task_description': 'Visual Behavior change detection: detect image identity changes, lick for reward',
            'time_bin_size': TARGET_BIN_SIZE * 1000,
            'temporal_alignment_event': 'Trial start time',
            'off_start': 0.0, 'off_end': None,
            'dataset': 'Allen Brain Observatory Visual Behavior 2P',
            'neural_data_type': 'calcium_events',
            'n_sessions': len(an), 'n_subjects': len(subjects),
            'bin_size_seconds': TARGET_BIN_SIZE,
        }
    }
    tt = sum(len(s) for s in an)
    tn = sum(an[s][0].shape[0] for s in range(len(an)))
    print(f"\n=== Summary ===")
    print(f"Sessions: {len(an)} (skipped: {skipped}), Trials: {tt}, Neurons: {tn}")
    print(f"Subjects: {len(subjects)}, Regions: {regions}")
    print(f"\nSaving to {outf}...")
    with open(outf, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved {outf} ({os.path.getsize(outf)/(1024*1024):.1f} MB)")
    print(f"Total time: {time.time()-t0g:.1f}s")

if __name__ == '__main__':
    main()
