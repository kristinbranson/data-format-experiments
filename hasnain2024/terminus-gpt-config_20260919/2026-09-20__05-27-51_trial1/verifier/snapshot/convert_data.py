#!/usr/bin/env python3
"""Convert the paper's two-context ALM ephys/video sessions to decoder format."""
import argparse, pickle, time, warnings
from pathlib import Path
import h5py, numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d

ROOT=Path('/app/data/Ephys_Behavior')
SESSIONS=[('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),('JGR2','2021-11-16',1),
 ('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),('JEB19','2023-04-18',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-20',1),('JEB19','2023-04-21',1)]
DT=.01; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT); TIME=(EDGES[:-1]+EDGES[1:])/2
ARTIFACT={'garbage','','noisy','real?'}

def chars(f,x):
    if isinstance(x,h5py.Reference): x=f[x]
    return ''.join(chr(int(v)) for v in x[()].ravel(order='F') if v).strip()
def deref_array(f,ds,i): return np.asarray(f[ds[()].ravel(order='F')[i]][()])
def vec(ds,dtype=None):
    a=np.asarray(ds[()]).ravel(order='F')
    return a.astype(dtype) if dtype else a

def smooth_rates(x,width=15):
    # Reference mySmooth is a centered boxcar; nearest avoids artificial zero edges.
    return uniform_filter1d(x,size=width,axis=1,mode='nearest').astype(np.float32)

def feature_trial(f,view,trial,name):
    names_obj=f[view['featNames'][()].ravel(order='F')[trial]]
    names=[chars(f,r) for r in names_obj[()].ravel(order='F')]
    if name not in names:return None,None,None
    a=deref_array(f,view['ts'],trial)
    # MATLAB (frames, coordinates, features) appears reversed by h5py: feature,coord,frame.
    fi=names.index(name); x=np.asarray(a[fi,0,:],float); y=np.asarray(a[fi,1,:],float)
    likelihood=np.asarray(a[fi,2,:],float) if a.shape[1]>2 else np.ones_like(x)
    ft=deref_array(f,view['frameTimes'],trial).ravel(order='F').astype(float)
    n=min(len(ft),len(x)); return ft[:n],np.c_[x[:n],y[:n]],likelihood[:n]

def velocity_on_grid(ft,pos,likelihood,go):
    out=np.full(TIME.shape,np.nan,float)
    if ft is None or len(ft)<2:return out
    visible=np.isfinite(pos).all(1)&np.isfinite(ft)&np.isfinite(likelihood)&(likelihood>0.9)
    # Reference smooths positions over 21 frames before finite differencing.
    pp=pos.copy()
    for j in range(2):
        good=np.isfinite(pp[:,j])
        if good.sum()>=2:
            fill=np.interp(np.arange(len(pp)),np.flatnonzero(good),pp[good,j])
            pp[:,j]=uniform_filter1d(fill,size=21,mode='nearest')
    dt=np.diff(ft); speed=np.r_[np.nan,np.sqrt(np.sum(np.diff(pp,axis=0)**2,axis=1))/np.where(dt>0,dt,np.nan)]
    speed[~visible]=np.nan; rel=ft-go
    good=np.isfinite(speed)&np.isfinite(rel)
    if good.sum()>=2:
        inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
        out[inside]=np.interp(TIME[inside],rel[good],speed[good])
        # Restore visibility gaps by nearest-frame visibility.
        ix=np.searchsorted(rel,TIME[inside]).clip(1,len(rel)-1)
        left=ix-1; near=np.where(abs(TIME[inside]-rel[left])<=abs(rel[ix]-TIME[inside]),left,ix)
        out[np.flatnonzero(inside)[~visible[near]]]=np.nan
    return out

def load_motion(subject,date,ntrials):
    p=ROOT/f'motionEnergy_{subject}_{date}.mat'
    if not p.exists():return None
    try:
        m=loadmat(p,squeeze_me=True,struct_as_record=False)['me']
        raw=np.asarray(m.data,dtype=object).ravel(order='F')
        if len(raw)!=ntrials:return None
        return [np.asarray(x,float).ravel(order='F') for x in raw]
    except Exception as e:
        warnings.warn(f'Could not load motion energy {p.name}: {e}');return None

def convert_session(subject,date,probe,show=False):
    t0=time.time(); path=ROOT/f'data_structure_{subject}_{date}.mat'
    with h5py.File(path,'r') as f:
        b=f['obj/bp']; ntr=int(np.asarray(b['Ntrials'][()]).squeeze())
        get=lambda k:vec(b[k],bool)
        R,L,hit,miss,no,early,aw=map(get,['R','L','hit','miss','no','early','autowater'])
        go=vec(b['ev/goCue'],float)
        stim=np.zeros(ntr,bool)
        if isinstance(b.get('stim'),h5py.Group) and 'enable' in b['stim']:stim=vec(b['stim/enable'],bool)
        keep_trials=np.flatnonzero((~stim)&np.isfinite(go))
        clu=f['obj/clu']; cg=f[clu[()].ravel(order='F')[probe-1]]
        qualities=np.array([chars(f,r).lower() for r in cg['quality'][()].ravel(order='F')])
        trialrefs=cg['trial'][()].ravel(order='F'); tmrefs=cg['trialtm'][()].ravel(order='F')
        unit_counts=[]; unit_rates=[]
        for q,rr,rt in zip(qualities,trialrefs,tmrefs):
            tr=np.asarray(f[rr][()]).ravel(order='F').astype(int)-1
            tm=np.asarray(f[rt][()]).ravel(order='F').astype(float)
            valid=(tr>=0)&(tr<ntr); tr=tr[valid]; tm=tm[valid]
            use=np.isin(tr,keep_trials)
            # Reference low-FR curation is based on all native valid trials, before
            # the decoder-specific exclusion of stimulation trials.
            aligned_all=tm-go[tr]
            rate=np.sum((aligned_all>=TMIN)&(aligned_all<TMAX))/(ntr*(TMAX-TMIN))
            if q not in ARTIFACT and rate>1:
                mats=np.zeros((len(keep_trials),len(TIME)),np.float32)
                remap=np.full(ntr,-1,int);remap[keep_trials]=np.arange(len(keep_trials))
                for old in np.unique(tr[use]):
                    z=tm[(tr==old)]-go[old]
                    mats[remap[old]]=np.histogram(z,EDGES)[0]/DT
                unit_counts.append(mats);unit_rates.append(rate)
        if not unit_counts:raise ValueError(f'No retained units in {subject} {date}')
        # Stack directly as trial x neuron x time and smooth only the time axis.
        neural=np.stack(unit_counts,axis=1).astype(np.float32)
        neural=uniform_filter1d(neural,size=15,axis=2,mode='nearest').astype(np.float32)
        # Video continuous values before session median threshold.
        views=[f[r] for r in f['obj/traj'][()].ravel(order='F')]
        tongue=np.full((len(keep_trials),len(TIME)),np.nan);paw=tongue.copy()
        video_times=[]
        for j,tr in enumerate(keep_trials):
            ft,pos,lk=feature_trial(f,views[0],tr,'tongue');tongue[j]=velocity_on_grid(ft,pos,lk,go[tr])
            ft,pos,lk=feature_trial(f,views[1],tr,'top_paw');paw[j]=velocity_on_grid(ft,pos,lk,go[tr])
            video_times.append(deref_array(f,views[0]['frameTimes'],tr).ravel(order='F').astype(float))
    me_raw=load_motion(subject,date,ntr)
    me=np.full((len(keep_trials),len(TIME)),np.nan)
    if me_raw is not None:
        # Motion-energy samples correspond one-to-one with camera frames.
        for j,tr in enumerate(keep_trials):
            y=me_raw[tr]; ft=video_times[j]; n=min(len(y),len(ft)); y=y[:n]; rel=ft[:n]-go[tr]
            good=np.isfinite(y)&np.isfinite(rel)
            if good.sum()>1:
                inside=(TIME>=rel[good].min())&(TIME<=rel[good].max())
                me[j,inside]=np.interp(TIME[inside],rel[good],y[good])
    def disc(a):
        out=np.full(a.shape,2,np.int64); finite=np.isfinite(a)
        if finite.any():
            med=float(np.median(a[finite]));out[finite]=(a[finite]>=med).astype(np.int64)
        else:med=np.nan
        return out,med
    td,tmed=disc(tongue);pd,pmed=disc(paw);md,mmed=disc(me)
    outputs=[]
    for j,tr in enumerate(keep_trials):
        if no[tr]:lick=2
        elif hit[tr]:lick=1 if R[tr] else 0
        elif miss[tr]:lick=0 if R[tr] else 1
        else:lick=2
        outcome=1 if hit[tr] else (0 if miss[tr] else 2)
        context=0 if aw[tr] else 1
        y=np.vstack([np.full(len(TIME),lick),np.full(len(TIME),context),np.full(len(TIME),outcome),td[j],pd[j],md[j]]).astype(np.int64)
        outputs.append(y)
    info={'session_id':f'{subject}_{date}','native_trials':ntr,'retained_trials':len(keep_trials),'excluded_stim_trials':int(stim.sum()),'n_neurons':len(unit_counts),'mean_rates_hz':unit_rates,'thresholds':{'tongue_velocity':tmed,'paw_velocity':pmed,'motion_energy':mmed},'probe':probe}
    print(f"{info['session_id']}: {ntr}->{len(keep_trials)} trials, {len(unit_counts)} neurons, {time.time()-t0:.2f}s",flush=True)
    return [x.astype(np.float32) for x in neural], [TIME[None,:].astype(np.float32).copy() for _ in keep_trials], outputs, info, (tongue,paw,me)

def plot_processing(sid,cont,out):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
    names=['tongue velocity','paw velocity','motion energy']
    for i,(a,n) in enumerate(zip(cont,names)):
        ax[i].plot(TIME,a[:min(20,len(a))].T,alpha=.2);ax[i].axvline(0,color='k');ax[i].set_ylabel(n)
    ax[-1].set_xlabel('time from go cue (s)');fig.tight_layout();fig.savefig(f'/app/processing_{sid}.png',dpi=140);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('outpicklefile');g=ap.add_mutually_exclusive_group();g.add_argument('--full',action='store_true');g.add_argument('--sample',action='store_true');ap.add_argument('--show-processing',action='store_true');args=ap.parse_args()
    sessions=SESSIONS[:2] if args.sample else SESSIONS
    neural=[];inputs=[];outputs=[];infos=[];subjects=[];sidx=[];bridx=[]
    for i,(sub,date,probe) in enumerate(sessions):
        n,x,y,info,cont=convert_session(sub,date,probe,args.show_processing);neural.append(n);inputs.append(x);outputs.append(y);infos.append(info)
        if sub not in subjects:subjects.append(sub)
        sidx.append(subjects.index(sub));bridx.append(np.zeros(n[0].shape[0],dtype=np.int64))
        if args.show_processing and i<2:plot_processing(info['session_id'],cont,y)
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,'subject_idx':np.asarray(sidx,dtype=np.int64),'brain_regions':['ALM'],'brain_region_idx':bridx,'input_names':['time from go cue (s)'],'output_names':['lick direction','behavioral context','outcome','tongue velocity','paw velocity','motion energy'],'output_values':[['left','right','none'],['WC','DR'],['incorrect','correct','ignore'],['below session median','at or above session median','not visible'],['below session median','at or above session median','not visible'],['below session median','at or above session median','no video']], 'metadata':{'task_description':'Decode choice, WC/DR context, outcome, and discretized video behavior from ALM firing rates.','time_bin_size':10.0,'temporal_alignment_event':'Bpod goCue onset','off_start':-2.5,'off_end':2.5,'neural_units':'Hz, 15-bin centered smoothing','session_info':infos,'trial_filter':'finite goCue and no photostimulation','source':'Separating cognitive and motor processes in the behaving mouse'}}
    with open(args.outpicklefile,'wb') as f:pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print(f'Saved {args.outpicklefile}: {len(neural)} sessions, {sum(map(len,neural))} trials, {sum(x[0].shape[0] for x in neural)} session-neurons')
if __name__=='__main__':main()
