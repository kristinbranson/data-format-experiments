# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the BWM dataset/session parquet indexes directly, selects a deterministic ten-session subset from `bwm_release.csv`, and downloads the required ALF payloads from public S3 into a UUID-keyed cache. Thus `--full` means ten selected sessions, not all eligible released sessions.

ii. `x=pd.read_csv(RELEASE,index_col=0); np.random.seed(42)`; `subs=np.random.choice(np.unique(x.subject),10,replace=False)`; `q=pd.read_parquet(QPATH)`; `urllib.request.urlretrieve(url,tmp)`; `n=2 if args.sample else 10`.

iii. The notes justify ten sessions as a deterministic methods-paper/README use case and as limiting downloads to about 5.4 GB instead of hundreds of GB. They describe exact cache metadata, UUIDs, sizes, and public-S3 provenance.

## 1-b. How are the data split into subjects?

i. Subject labels come from `bwm_release.csv`. Ten unique subjects are randomly selected without replacement, one first listed session is chosen for each, and output `subject_idx` indexes the encounter-ordered unique subject list.

ii. `subs=np.random.choice(np.unique(x.subject),10,replace=False)`; `eids=[str(x.iloc[by[s][0]].eid) for s in subs]`; `subject_idx=np.array([subjects.index(x['subject']) for x in sessions],np.int64)`.

iii. The notes say this seeded one-session-per-subject selection makes a reproducible, tractable ten-session subset.

## 1-c. How are the data split into sessions?

i. Each EID is already a session. `load_session` processes one selected EID, and its result becomes one element of each session-level output list.

ii. `for subject,eid in chosen: sessions.append(load_session(eid,subject,q,ss.loc[eid]))`; `data['neural'].append(x['neural'])`.

iii. The notes treat EID/session boundaries from the ONE metadata as authoritative.

## 1-d. How are the data split into trials?

i. Rows of `trials.table` define trials. For each retained row, stimulus onset defines a common window and one neural/input/output array is appended in chronological order.

ii. `stim=np.asarray(trials.stimOn_times,float)`; `for ti in idx:`; `times=stim[ti]+CENTERS`; `neural.append(np.concatenate(mats))`.

iii. The notes state that trial tables are authoritative and that chronological ordering is preserved.

## 1-e. How are trials filtered based on quality controls?

i. Trials require finite stimulus onset, choice in {-1,+1}, rounded prior in {0.2,0.5,0.8}, full wheel/camera endpoint coverage, and finite interpolated traces. Sessions must retain at least two trials. No 80 ms–2 s reaction-time filter is applied.

ii. `valid=np.isfinite(stim)&np.isin(choice,[-1,1])&np.isin(np.round(prior,1),[.2,.5,.8])`; `valid &= (stim+OFF0>=wtime[0])...`; `if not (np.isfinite(ww).all() and np.isfinite(mm).all()): continue`; `if len(kept)<2: raise ValueError(...)`.

iii. The notes justify a common validity mask and full stream coverage, but do not justify omitting the paper/reference reaction-time criterion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from each probe's pykilosort `spikes.times`, `spikes.clusters`, `clusters.metrics`, and `clusters.channels`; channel atlas IDs supply anatomy.

ii. `st=...spikes.times`; `sc=...spikes.clusters`; `met=...clusters.metrics`; `ch=...clusters.channels`.

iii. The notes identify spike times and cluster assignments as the activity source, with metrics/anatomy used for curation and metadata.

## 2-b. How is the `neural` data processed?

i. For every trial and probe, spikes are sliced to the stimulus-relative window, assigned to 20 ms bins, counted by good cluster, and probe matrices are concatenated. Values remain spike counts (`float32`), not firing rates.

ii. `bins=np.floor((rel-OFF0)/DT).astype(int)`; `M=np.zeros((len(good),len(CENTERS)),np.float32)`; `M[j,b]+=1`; `neural.append(np.concatenate(mats))`.

iii. The notes claim fixed-bin spike counting matches reference `bincount2D` semantics and explicitly describe the representation as spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters with `metrics.label >= 1` are retained. All such clusters are kept even when anatomy falls back to or maps to `void`; no explicit in-brain/void exclusion is made.

ii. `labels=np.asarray(met['label'],float); good=np.flatnonzero(labels>=1)`; `ID2AC.get(...,'void')`.

iii. The notes justify the threshold as the reference `good_clusters` criterion and report that the final selected data happened to contain zero `void` neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike times are shifted by `stimOn_times` and binned from −0.6 through +1.5 s relative to visual stimulus onset.

ii. `OFF0,OFF1,DT=-.6,1.5,.02`; `lo=np.searchsorted(st,stim[ti]+OFF0)`; `rel=st[lo:hi]-stim[ti]`.

iii. The notes say stimulus onset is explicitly required and use the union of target-specific IBL choice and prior windows to make one shared tensor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The final bin size is 20 ms, yielding 105 bins over −0.6 to +1.5 s. Spikes are directly counted in those bins; there is no later rebinning.

ii. `DT=.02`; `EDGES=np.arange(OFF0,OFF1+DT/2,DT)`; metadata stores `'time_bin_size':20.0`.

iii. The notes choose 20 ms to support shared dynamic outputs, while acknowledging that target-specific paper analyses used differing windows/bin sizes.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the chosen offset, bin width, and the trials' `stimOn_times` alignment event; the same relative centers are used for every trial.

ii. `CENTERS=(EDGES[:-1]+EDGES[1:])/2`; `times=stim[ti]+CENTERS`.

iii. The notes describe this as the fixed stimulus-relative grid required by the decoder task.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Adjacent bin edges are averaged to form centers from −0.59 to +1.49 s, then copied into every trial input.

ii. `CENTERS=(EDGES[:-1]+EDGES[1:])/2`; `np.vstack([CENTERS,np.full(len(CENTERS),b)])`.

iii. The notes emphasize bin centers to avoid boundary ambiguity.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the centers of the exact same edges used to bin stimulus-relative spikes, so columns align one-to-one.

ii. `bins=np.floor((rel-OFF0)/DT).astype(int)` and `ins.append(np.vstack([CENTERS,...]))`.

iii. The notes report independent reconstruction and exact comparison of the time grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from ordered `trials.probabilityLeft`, rounded to one decimal place; a change denotes a new block.

ii. `pr=np.round(prior,1)`; `blocknum[i]=0 if pr[i]!=pr[i-1] else blocknum[i-1]+1`.

iii. The notes identify block-local trial count as a task-required derived input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Before trial filtering, the first trial is zero; subsequent trials reset to zero when the rounded prior changes and otherwise increment. The retained value is broadcast across time.

ii. `blocknum=np.zeros(len(trials),np.float32)`; `for i in range(1,len(trials))...`; `np.full(len(CENTERS),b)`.

iii. Computing before filtering preserves the true position in the original behavioral block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`, restricted to −1 and +1.

ii. `choice=np.asarray(trials.choice,float)`; `np.isin(choice,[-1,1])`.

iii. The notes describe raw −1/+1 as the reference task variable.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps −1 to 0 and +1 to 1, then broadcasts the category across all time bins. This is opposite the required/reference left=0, right=1 mapping because IBL uses +1 for left and −1 for right.

ii. `cc=0 if c==-1 else 1`; `np.full(len(CENTERS),cc)`.

iii. The notes say labels were checked and static outputs were broadcast for a uniform tensor, but they do not recognize the direction reversal.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii. `prior=np.asarray(trials.probabilityLeft,float)`; `pr=np.round(prior,1)`.

iii. The notes identify the raw block prior as the required target.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to one decimal, mapped 0.2→0, 0.5→1, 0.8→2, and broadcast across time.

ii. `pp={.2:0,.5:1,.8:2}[float(p)]`; `np.full(len(CENTERS),pp)`.

iii. The notes say static outputs are broadcast to meet the decoder's uniform time-varying tensor format.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from `_ibl_wheel.timestamps` and `_ibl_wheel.position`.

ii. `wt=...wheel.timestamps`; `wp=...wheel.position`; `wpos,wtime=interpolate_position(wt,wp,freq=1000)`.

iii. The notes explicitly choose the official IBL wheel processing path.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, filtered velocity is calculated with IBL utilities, absolute value gives speed, and `np.interp` samples it at neural-bin centers.

ii. `wvel,_=velocity_filtered(wpos,fs=1000)`; `wspeed=np.abs(wvel)`; `ww=np.interp(times,wtime,wspeed)`.

iii. The notes state that this matches `SessionLoader`/IBL's recommended velocity computation.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. One pair of 1/3 and 2/3 quantiles is computed after pooling every aligned wheel value from all selected sessions; `np.digitize` creates low/medium/high classes.

ii. `wheel=np.concatenate([np.concatenate(x['wheel']) for x in sessions])`; `wq=np.quantile(wheel,[1/3,2/3])`; `np.digitize(w,wq)`.

iii. The notes prefer fixed pooled thresholds so categories are dataset-wide comparable and globally balanced.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at `stimOn_times + CENTERS`, the same centers represented by neural columns.

ii. `times=stim[ti]+CENTERS`; `ww=np.interp(times,wtime,wspeed)`.

iii. The notes describe common-clock, bin-center alignment and independently validate it.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It derives from left-camera times and `leftCamera.ROIMotionEnergy`, with right-camera fallback if left is absent or insufficiently finite.

ii. `for v in ('left','right'):`; `ct=...f'{v}Camera.times'`; `cm=...f'{v}Camera.ROIMotionEnergy'`.

iii. The notes say this follows reference left preference and uses released ROI motion energy rather than recomputing video features.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Timestamp and motion arrays are truncated to their common length, checked for >95% finite data, and linearly interpolated at neural-bin centers; no filtering or normalization is applied.

ii. `n=min(len(ct),len(cm))`; `cm=np.asarray(cm[:n],float)`; `np.isfinite(cm).mean()>.95`; `mm=np.interp(times,ct,cm)`.

iii. The notes state that official motion energy is already processed and should be used directly.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. One pair of pooled 1/3 and 2/3 quantiles across all selected sessions is used with `np.digitize`.

ii. `whisk=np.concatenate([np.concatenate(x['whisk']) for x in sessions])`; `mq=np.quantile(whisk,[1/3,2/3])`; `np.digitize(m,mq)`.

iii. The notes justify pooled fixed thresholds as dataset-wide comparable and globally balanced.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated at stimulus onset plus the same bin centers as the neural data.

ii. `times=stim[ti]+CENTERS`; `mm=np.interp(times,ct,cm)`.

iii. The notes rely on synchronized IBL timestamps and report an independent alignment check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Downloads retry and validate expected byte size; camera length mismatches are truncated; right camera is a fallback; absent probe records are skipped; anatomy can fall back to `void`; invalid trials are omitted. Conversely, no valid camera, no good neurons, fewer than two trials, or any selected-session exception aborts the whole conversion.

ii. `for attempt in range(5)`; `n=min(len(ct),len(cm))`; `except KeyError: continue`; `except Exception: atlas=np.zeros(...)`; `except Exception as e: ... raise`.

iii. The notes emphasize exact source validation, caching, fallbacks, and common masking, but claim no unresolved issue after the chosen ten sessions succeeded.

## 10-a. What are the most time-consuming steps of the code?

i. Initial multi-gigabyte S3 downloads dominate uncached execution; loading and trial-binning large spike arrays is the main cached processing cost.

ii. `urllib.request.urlretrieve(url,tmp)`; per trial/probe `np.searchsorted`, followed by the per-spike loop.

iii. The notes estimate ~5.4 GB canonical downloads and 5–7 seconds cached processing per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop, probe loop, per-spike `(cluster, bin)` counting loop, trial-number loop, and final trial assembly loop could be partly vectorized. The per-spike dictionary loop is the clearest hotspot and could use remapped IDs plus `np.add.at`/`bincount`.

ii. `for ti in idx:`; `for st,sc,good in probe_data:`; `for c,b in zip(cid,bins): ... M[j,b]+=1`; `for i in range(1,len(trials))`; `for c,p,b,w,m in zip(...)`.

iii. The notes say binary-search slicing limits Python work to each trial window and call behavior interpolation/masks vectorized, but the implementation still loops over individual spikes.

## 10-c. What processing does the code repeat multiple times?

i. Dataset-record filtering and download/loading occur separately for every requested object; every trial repeats time-vector construction, interpolation, spike slicing, cluster lookup construction per probe, and matrix allocation. Final assembly loops over all trials again.

ii. Repeated calls to `load_record(...)`; inside `for ti in idx`, `times=stim[ti]+CENTERS`, `lut={...}`, and `M=np.zeros(...)`.

iii. The notes justify caching downloads and slicing windows, but do not discuss rebuilding the identical cluster LUT for every trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It retains `labels`, `trial_idx`, raw continuous wheel/whisker traces, and camera side in intermediate session objects; most are discarded after thresholds/metadata are built. It also hashes imports and creates `ThreadPoolExecutor` imports that are unused, and downloads/loads anatomy metadata needed only for labels, not decoding.

ii. `return dict(...labels=np.array(labels_all),trial_idx=kept,...wheel=rawwheel,whisk=rawwhisk...)`; only selected fields are copied into `data`.

iii. The notes use these intermediates for sanity checks, plotting, provenance, and threshold construction, so some are diagnostically useful even though they are absent from the final decoder arrays.
