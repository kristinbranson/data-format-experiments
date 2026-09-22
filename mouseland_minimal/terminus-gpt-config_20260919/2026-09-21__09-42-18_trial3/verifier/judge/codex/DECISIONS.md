# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, loads each existing `Beh_<group>.npy`, deduplicates behavior by base session ID, intersects those IDs with all `*_neural_data.npy` files, and loads retinotopy per session. Neural files are loaded sequentially.

ii. `info=np.load(os.path.join(ROOT,'beh','Imaging_Exp_info.npy'),allow_pickle=True).item()`; `bd=np.load(bf,allow_pickle=True).item()`; `sessions=sorted(set(beh)&set(spkfiles))`; `raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']`

iii. The trajectory says the index, behavior schema, repository loaders, and large object-array neural files were inspected first. Sequential loading was chosen because the source neural files total hundreds of GB and cannot be memory-mapped.

## 1-b. How are the data split into subjects?

i. Subject is the mouse prefix/`mname`; unique subject names are sorted and every retained session receives an index into that list.

ii. `subj.append(s.split('_')[0])`; `subjects=sorted(set(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int16)`

iii. The AI treated the mouse name in the experiment index/session ID as the authoritative subject identifier.

## 1-c. How are the data split into sessions?

i. A session is the base ID `mname_datexp_blk`. Duplicate listings across experiment groups are collapsed by preserving the first behavior record, and only sessions having both behavior and neural files are used.

ii. `def base_sid(r): return f"{r['mname']}_{r['datexp']}_{r['blk']}"`; `if key in bd and base not in beh: beh[base]=bd[key]`; `sessions=sorted(set(beh)&set(spkfiles))`

iii. The trajectory notes that the master index repeats some recordings under multiple experiment types, so deduplication was intentional.

## 1-d. How are the data split into trials?

i. The AI loops over `ntrials`/`TrialStim` and selects frames whose `ft_trInd` equals the trial and which are in `ft_CorrSpc`, finite, nonnegative, and below `Texture_Length`. Trials remain variable length before 1-second aggregation.

ii. `for tr in range(min(ntr,len(trialstim))):`; `ix=np.where((tri==tr)&np.asarray(b['ft_CorrSpc'][:nfr],dtype=bool)&np.isfinite(pos)&(pos>=0)&(pos<float(b['Texture_Length'])))[0]`

iii. The AI identified corridor entry through exit of the visual texture as the requested trial interval and later corrected an initial inclusion of gray-space frames.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than two selected frames are discarded; sessions with fewer than two retained trials are discarded. There is no long/stationary-trial outlier filter.

ii. `if ix.size<2: continue`; `if len(ns)<2: print('skip',s,'too few trials'); continue`

iii. The trajectory describes these as “all valid visual-corridor trials” and reports no malformed trials. It does not justify omitting the long-trial quality control used by the human solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from every array in the `spks` list of each session neural file, concatenated along the neuron dimension. `iarea` from the corresponding retinotopy file supplies region metadata.

ii. `raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']; spk=np.concatenate(raw,axis=0)`; `a=np.asarray(np.load(f)['iarea']).ravel()`

iii. Repository `utils.load_spk` was inspected, and the AI explicitly chose to concatenate all supplied planes and use the paper’s nonnegative Suite2p deconvolved traces.

## 2-b. How is the `neural` data processed?

i. For each trial, traces are averaged within non-overlapping 1-second bins and stored as float32. Empty bins are forward-filled (or zero-filled at the start).

ii. `N[:,j]=spk[:,jj].mean(1,dtype=np.float32)`; `N[:,j]=N[:,j-1] if j else 0`

iii. The AI judged a native-frame full copy too large and chose 1-second means as a tractability compromise intended to retain temporal cue and lick structure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. All concatenated Suite2p traces are retained, including neurons labeled unassigned. Region codes are heuristically remapped and length mismatches are padded/truncated.

ii. `out=np.zeros(len(a),dtype=np.int16)`; `out[a==1]=1; out[np.isin(a,[2,3])]=2; ...`; `z=np.zeros(n,dtype=np.int16); z[:min(n,len(out))]=out[:min(n,len(out))]`

iii. The AI reasoned that Suite2p cell classification was already curated and that the repository loader concatenated all traces, so it applied no post-hoc selectivity filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The first retained visual-corridor frame is treated as time zero. Selected frames are assigned to elapsed 1-second bins from that point, through the end of the texture corridor.

ii. `t0=tsec[ix[0]]; rel=tsec[ix]-t0`; `bins=np.minimum((rel/BIN_S).astype(int),nb-1)`

iii. The AI states that bins are aligned to corridor entry/trial start and corrected the conversion to exclude the subsequent gray-space segment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 1,000 ms. Native imaging at about 3.2 Hz is rebinned into non-overlapping 1-second means.

ii. `ROOT='/app/data'; BIN_S=1.0`; `'time_bin_size':1000.0`; `'temporal_binning':'non-overlapping 1 s means from corridor entry to exit'`

iii. Rebinning was explicitly chosen to reduce an otherwise very large output while preserving coarse temporal order.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and the session frame timestamps `ft`.

ii. `ft=np.asarray(b['ft'][:nfr])`; `sound=np.asarray(b['SoundFr'])`; `cue_rel=(np.interp(sound[tr],np.arange(nfr),tsec)-t0)`

iii. The AI recognized `SoundFr` as a fractional global imaging-frame index and interpolated it onto elapsed timestamp seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. MATLAB datenums are converted to seconds, the cue frame is interpolated, and cue-relative time is cue time minus each 1-second bin center. Missing cues produce NaNs.

ii. `tsec=(ft-ft[0])*86400.0`; `elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S`; `tcue=(cue_rel-elapsed)... else np.full(nb,np.nan,np.float32)`

iii. The AI intended positive values before and negative values after the cue, on the same rebinned time grid as neural data.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. One value is computed at the center of every neural 1-second bin, using the same number of bins.

ii. `elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S`; `inp=np.vstack([tcue,day,elapsed,rewarded])`

iii. The common bin axis was chosen to keep all streams temporally aligned.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `datexp` for each session and the earliest included date for that mouse.

ii. `d=date_ordinal(rowmap[s]['datexp'])`; `subject_day0[mouse]=min(...)`

iii. After validation, the AI changed an absolute date ordinal to a per-subject relative training-day value.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days since the subject’s first included imaging date are computed, with 1 added so the first date is day 1, then broadcast across all bins.

ii. `day=np.full(nb,date_ordinal(rowmap[s]['datexp'])-subject_day0[s.split('_')[0]]+1.0,np.float32)`

iii. The AI considered relative date more meaningful than an absolute ordinal, but did not distinguish recorded-session count from elapsed calendar days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `ft` timestamps and the first selected `ft_trInd`/`ft_CorrSpc` frame, not directly from `StartFr`.

ii. `t0=tsec[ix[0]]; rel=tsec[ix]-t0`; `elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S`

iii. The AI treated the first visual-corridor frame as corridor entry/time zero.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It creates regularly spaced bin-center times of 0.5, 1.5, 2.5 seconds, etc., rather than retaining measured frame times.

ii. `elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S`

iii. This follows directly from the chosen 1-second aggregation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Each value is the center time of the corresponding aggregated neural bin.

ii. `inp=np.vstack([tcue,day,elapsed,rewarded]).astype(np.float32)`

iii. The AI used a shared bin count and axis for inputs, outputs, and neural activity.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is inferred from `TrialStim` and finite `RewardFr` values: the stimulus with the largest number of finite rewards is designated the session’s rewarded stimulus.

ii. `counts={x:int(np.isfinite(rw[st==x]).sum()) for x in np.unique(st)}`; `reward_stim[s]=max(counts,key=counts.get)...`

iii. The AI reasoned that reward is an experimental corridor property and that naive/unsupervised sessions with no rewards should be all zero.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A constant binary series is made per trial by comparing that trial’s stringified `TrialStim` to the inferred rewarded-stimulus value.

ii. `rewarded=np.full(nb,int(reward_stim[s] is not None and str(trialstim[tr])==reward_stim[s]),np.float32)`

iii. The intent was to label the rewarded corridor, rather than merely bins in which a reward was delivered.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived directly from `TrialStim` across all sessions.

ii. `stim_names=sorted({str(x) for s in sessions for x in np.unique(beh[s]['TrialStim'])})`

iii. The trajectory does not document consideration of `WallName` or the masked values in swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct string representation of `TrialStim` becomes a global category index, broadcast across a trial’s bins.

ii. `stim_id={x:i for i,x in enumerate(stim_names)}`; `cat=np.full(nb,stim_id[str(trialstim[tr])],np.int16)`

iii. The AI treated stored trial-stimulus identities as the desired visual categories without collapsing variants to four base textures.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, with `ft` used to place lick frames in elapsed seconds.

ii. `lick=np.asarray(b['LickFr'])`; `lt=np.interp(lf,np.arange(nfr),tsec)-t0`

iii. The AI identified lick events as fractional imaging-frame indices.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Finite lick frames are interpolated to time, converted to 1-second bin indices, restricted to the trial range, and each bin containing one or more licks is set to 1.

ii. `lb=(lt/BIN_S).astype(int); lb=lb[(lb>=0)&(lb<nb)]; L[np.unique(lb)]=1`

iii. The AI explicitly chose “any lick in a temporal bin” as the binary output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are expressed relative to the same trial start and assigned to the same 1-second indices as neural means.

ii. `lt=np.interp(lf,np.arange(nfr),tsec)-t0`; `L[np.unique(lb)]=1`; `out=np.vstack([cat,L,pcat,vcat])`

iii. Shared trial-relative bins were intended to align every stream.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise `ft_Pos`, gated by `ft_CorrSpc` and `Texture_Length`.

ii. `pos=np.asarray(b['ft_Pos'][:nfr])`; `(pos>=0)&(pos<float(b['Texture_Length']))`

iii. The AI corrected its initial interpretation from the full 60-unit corridor to the 40-unit/4 m visual texture corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Frame positions are averaged within each 1-second bin, with empty bins forward-filled, then scaled relative to `Texture_Length`.

ii. `P[j]=np.nanmean(pos[jj])`; `P[j]=P[j-1] if j else 0`; `P/float(b['Texture_Length'])*4`

iii. The AI intended a linear mapping from stored corridor units to the requested four meters.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The scaled bin-mean position is truncated to an integer and clipped to category 0–3, corresponding to four equal 1 m bins.

ii. `pcat=np.clip((P/float(b['Texture_Length'])*4).astype(int),0,3).astype(np.int16)`

iii. Equal-length categories were chosen because the decoder specification explicitly requests four 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is averaged over exactly the frame indices used for each neural average.

ii. `jj=ix[bins==j]`; `N[:,j]=spk[:,jj].mean(...)`; `P[j]=np.nanmean(pos[jj])`

iii. The same selected frames and bin assignments provide alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed` in valid visual-corridor frames.

ii. `v=np.asarray(b['ft_RunSpeed'])`; `ok=...&np.asarray(b['ft_CorrSpc'],dtype=bool)...`

iii. The AI used the directly available running-speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global value thresholds are computed from all valid corridor-frame speeds. Within each trial, speed is averaged per 1-second bin before categorization; empty bins are forward-filled.

ii. `q=np.quantile(np.concatenate(speeds),[.25,.5,.75])`; `V[j]=np.nanmean(speed[jj])`

iii. The AI chose global quartiles to avoid loading neural sessions twice and to provide common thresholds across the dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Each bin-mean speed is digitized against the three global quartile values, producing 0–3.

ii. `vcat=np.digitize(V,q,right=False).astype(np.int16)`

iii. This was intended to implement the requested four 25%-of-data bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is averaged over the identical frame subset used for each neural bin.

ii. `jj=ix[bins==j]`; `N[:,j]=...`; `V[j]=np.nanmean(speed[jj])`

iii. Common bin membership provides temporal alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Streams are truncated to their common available frame count; nonfinite positions are excluded; missing cue values become NaN; bins without frames are forward-filled; malformed region-vector lengths are padded/truncated; short trials and sessions are skipped.

ii. `nfr=min(spk.shape[1],len(b['ft_trInd']))`; `if ix.size<2: continue`; `else: N[:,j]=N[:,j-1] if j else 0`; `if len(out)!=n: ...`

iii. The AI added guards to let the full conversion complete despite mismatched metadata or missing events; the trajectory reports no runtime alignment failures.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating hundreds of GB of object-array neural files, looping through all trial bins to average tens of thousands of neurons, and serializing the roughly 95 GB pickle dominate runtime.

ii. `raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']; spk=np.concatenate(raw,axis=0)`; `for tr ... for j in range(nb):`; `pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)`

iii. The trajectory repeatedly identifies neural I/O and final serialization as lengthy, with two full conversions required after corrections.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial full-array search and inner per-bin aggregation loop could be replaced by grouped indices/reductions; subject indexing also uses repeated linear searches.

ii. `ix=np.where((tri==tr)&...)[0]`; `for j in range(nb): jj=ix[bins==j]`; `[subjects.index(x) for x in subj]`

iii. The trajectory does not explicitly justify these loops; they are straightforward implementations chosen while prioritizing completion of very large sequential session processing.

## 12-c. What processing does the code repeat multiple times?

i. Trial masks repeatedly convert/slice `ft_CorrSpc` and scan `tri`; each bin repeatedly scans the trial’s `bins` array. The conversion itself was run twice after validation exposed metadata/scaling issues.

ii. `np.asarray(b['ft_CorrSpc'][:nfr],dtype=bool)` inside the trial loop; `jj=ix[bins==j]` inside the bin loop

iii. No explicit justification was given for repeated mask construction; the second full run was justified by correcting corridor, region-file, and training-day errors.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads and retains unassigned neurons even though the reference analysis keeps only four mapped visual regions. It also computes native-frame detail only to average it into 1-second bins, and the preliminary global speed pass builds a large list that is discarded after quantile calculation.

ii. `speeds.append(v[ok].astype(np.float32))`; `q=np.quantile(np.concatenate(speeds),...); del speeds`; `spk=np.concatenate(raw,axis=0)` with no neuron filter

iii. The AI intentionally retained all Suite2p neurons as a fidelity choice and accepted the preliminary speed pass as a way to establish global thresholds before neural conversion.
