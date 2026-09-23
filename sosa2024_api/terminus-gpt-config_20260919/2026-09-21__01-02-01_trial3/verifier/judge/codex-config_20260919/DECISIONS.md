# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every NWB file under `/app/data`, sorts the paths, and processes one file as one session with `pynwb.NWBHDF5IO`. In full/default mode it uses all 152 files; `--sample` restricts this to two.

ii. `files=sorted(Path('/app/data').rglob('*.nwb')); files=files[:2] if a.sample else files` and `with NWBHDF5IO(str(fn),'r',load_namespaces=True) as io: nwb=io.read()`.

iii. The notes justify this as retaining every released, nonduplicate NWB session and satisfying the pynwb-only requirement. They report 11 subjects and 152 sessions.

## 1-b. How are the data split into subjects?

i. Subject identity comes from `nwb.subject.subject_id`; a first-seen unique subject list is built, and each session receives the corresponding index.

ii. `return neural_trials,inputs,outputs,nwb.subject.subject_id,info` and `if sub not in subjects: subjects.append(sub); sidx.append(subjects.index(sub))`.

iii. The notes treat NWB subject metadata as the authoritative mouse identifier and validate 11 unique subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file becomes one session in the outer lists. Session metadata combines subject and NWB session/file identity.

ii. `for i,fn in enumerate(files): N,I,O,sub,info=process_file(fn,...); neural.append(N); inputs.append(I); outputs.append(O)`.

iii. The agent states that each released NWB has a native session identity and should be retained rather than applying analysis utilities that sometimes select one session per day.

## 1-d. How are the data split into trials?

i. Frame indices with positive `trial_start` are paired with the first positive `teleport` at or after the start and before the next start. Bin centers span the paired start/teleport timestamps.

ii. `starts=np.flatnonzero(B['trial_start']>0); tele=np.flatnonzero(B['teleport']>0)`; `e0=tele[tele>=s]`; `pairs.append((int(s),int(e0[0])))`.

iii. The notes say the NWBs lack a trials table, explicit start/teleport pulses define complete traversals, and inter-trial sentinel periods must be excluded.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than two complete trials fail. Trials are retained only if their start and teleport timestamps are fully covered by every neural plane. There is no minimum-duration filter; one terminal trial was removed.

ii. `pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]` and `if len(pairs)<2: raise ValueError(...)`.

iii. A validator warning exposed an all-zero extrapolated trial, so the agent added complete neural-coverage filtering. It explicitly retained an unusually long but valid trial rather than imposing an unsupported duration rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each plane of `processing['ophys']['Deconvolved']`, restricted using its ROI table region and Suite2p `iscell` flag.

ii. `dec=nwb.processing['ophys']['Deconvolved'].roi_response_series`; `neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]`.

iii. The agent reasoned that the stored Deconvolved series is equivalent to the paper session object's `timeseries['events']`, so recomputing dF/F from Fluorescence and Neuropil was unnecessary.

## 2-b. How is the `neural` data processed?

i. Curated deconvolved samples are averaged into common 100-ms temporal bins independently per plane, then planes are concatenated neuron-wise. No dF/F, neuropil correction, smoothing, or new deconvolution is performed.

ii. `xp[:,k]=neural_all[a:b].mean(0)` and `X=np.concatenate(binned_planes,axis=0)`.

iii. The notes claim the NWB already contains author-processed deconvolved events and choose mean binning to put unequal native sampling grids on a common decoder grid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `iscell[:,0] > 0` are retained. All planes are included. No speed-correlated putative-interneuron filter, place-cell filter, or running-time filter is applied.

ii. `iscell=np.asarray(table['iscell'].data[:])[:,0]>0`; `valid=iscell[region]`.

iii. The agent identifies `iscell` as Suite2p curation and argues that place/reward-cell and speed filters are analysis-specific and could remove useful decoder information.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial grid is defined relative to the explicit `trial_start` timestamp; neural plane timestamps are assigned to 100-ms bins centered at 0.05, 0.15 seconds, and so on after that event.

ii. `centers=t_start+DT/2+np.arange(...)*DT`; `ni=np.searchsorted(neural_t,centers-DT/2)`.

iii. The agent emphasizes timestamp alignment because neural and behavior streams can have different lengths and sampling grids.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 100 ms. Native approximately 15.5-Hz neural events are temporally rebinned by taking the mean of samples in each bin; empty bins use the nearest sample.

ii. `DT=0.1`; `metadata['time_bin_size']=DT*1000`; `if b>a: ...mean(0) else: ...nearest_idx(...)`.

iii. The notes justify 100 ms as a common, fine temporal grid across sessions with differing acquisition configurations.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior `trial_start` timestamp and the generated 100-ms bin centers.

ii. `t_start,t_end=bt[s],bt[e]` and `centers-t_start`.

iii. The notes describe it as continuous time relative to the explicit trial-start pulse.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial-start time is subtracted from each generated bin center, producing values beginning at 0.05 s in 0.1-s increments.

ii. `inp=np.vstack([centers-t_start, ...]).astype(np.float32)`.

iii. This follows from using bin centers rather than native behavior samples.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses exactly the same `centers` array that defines the neural bins, so its columns correspond one-to-one with neural columns.

ii. `assert X.shape[1]==inp.shape[1]==out.shape[1]`.

iii. The agent reports independent raw-to-converted alignment checks and strictly increasing 100-ms time vectors.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the frame-sampled behavior `environment` value at the trial-start index.

ii. `env=int(round(float(B['environment'][s])))`.

iii. The agent observed the native variable is the required binary ENV1/ENV2 context.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The value is rounded, converted to integer, clipped to 0/1, and repeated at all trial time points.

ii. `env=max(0,min(1,env))`; `np.full(len(centers),env)`.

iii. Repetition makes all inputs consistently shaped `(variables, time)` while preserving a per-trial label.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is taken from behavior `trial number` at the trial-start frame.

ii. `trialnum=float(B['trial number'][s])`.

iii. The mapping plan calls the native value the continuous, session-local trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No numerical transformation is applied; the start-frame value is repeated across the trial and cast as float32 with the input matrix.

ii. `np.full(len(centers),trialnum)`.

iii. The agent says repeating per-trial variables avoids ambiguous mixed input shapes.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward` timestamps by first determining whether each retained trial contains a reward event.

ii. `rew_t=np.asarray(beh['Reward'].timestamps[:],float)` and `rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))`.

iii. The agent notes Reward is event-based and must be matched directly by time rather than treated as a frame-sampled stream.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first retained trial gets 0. Later trials receive the immediately preceding retained trial's binary outcome, repeated across time.

ii. `prev=outcomes[-1] if outcomes else 0; outcomes.append(rewarded)`.

iii. This implements omitted=0/rewarded=1 and the agent validated trial-level constancy.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from interpolated behavior `position` and an inferred active A/B/C zone. Zone evidence is the position where behavior `reward_zone` is positive; omission labels are filled from the nearest labeled trial. The hard-coded zones are A 80–100, B 200–220, C 320–340 cm.

ii. `ZONES=np.array([[80.,100.],[200.,220.],[320.,340.]])`; `hit=np.flatnonzero(B['reward_zone'][s:e+1]>0)`.

iii. The agent says `reward_zone` is a transient signal rather than an A/B/C code and assumes contiguous task blocks permit nearest-trial filling.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is linearly interpolated to bin centers. Signed distance is negative before the zone, zero within it, and positive after its far edge.

ii. `pos=np.interp(centers,bt,B['position'])`; `dist=np.where(pos<lo,pos-lo,np.where(pos>hi,pos-hi,0.))`.

iii. The notes describe distance to the closest point of the active fixed zone, making every in-zone sample exactly zero.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven requested classes at -50, -10, 0, 10, and 50 cm.

ii. `y[d < -50]=0; y[(d>=-50)&(d<-10)]=1; ...; y[d>50]=6`.

iii. The thresholds are presented as a direct implementation of the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is interpolated at the same 100-ms centers used for neural binning; the derived distance therefore has identical columns.

ii. `pos=np.interp(centers,bt,B['position'])` and the shared-length assertion.

iii. The notes report exact independent checks of interpolated position and derived output classes.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavior `position` and behavior timestamps.

ii. `pos=np.interp(centers,bt,B['position']).astype(np.float32)`.

iii. The position stream records location along the 450-cm virtual corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is linearly interpolated to the common 100-ms bin centers, then discretized. It is not smoothed or clipped.

ii. `pos=np.interp(centers,bt,B['position'])`.

iii. The agent chose linear interpolation for continuous position to align independent sampling grids.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with thresholds 90, 180, 270, and 360 cm yields five integer classes.

ii. `poscls=np.digitize(pos,[90,180,270,360],right=False).astype(np.int64)`.

iii. These are the requested five equal 90-cm bins over the 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is evaluated at the same bin centers as neural activity and checked to have the same number of columns.

ii. `np.interp(centers,bt,B['position'])` and `assert X.shape[1]==...==out.shape[1]`.

iii. Timestamp-based interpolation was chosen instead of assuming index alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavior `lick` stream and behavior timestamps.

ii. `jj=nearest_idx(bt,centers); lick=(B['lick'][jj]>0).astype(np.int64)`.

iii. The agent identifies lick as a discrete/count-like stream and uses nearest-sample alignment.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The nearest native sample is selected at each bin center and any positive value is binarized to 1.

ii. `lick=(B['lick'][jj]>0).astype(np.int64)`.

iii. Binarization matches the required no/yes output and, according to the notes, the paper's pre-smoothing binary lick representation.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Nearest behavior samples are selected at the same centers used for neural bins.

ii. `jj=nearest_idx(bt,centers)`.

iii. The agent uses nearest-neighbor rather than linear interpolation because lick is discrete.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from `reward_zone` activation together with `position`; trials without activation inherit the nearest trial's label.

ii. `raw_zone.append(zone_for_trial(float(np.median(B['position'][s+hit]))))`; `zones[bad]=zones[good[np.argmin(...)] ]`.

iii. The agent argues that the native signal is transient and task blocks are contiguous, so neighboring trials supply omission labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Median activation position is assigned to the nearest hard-coded zone center, encoded A/B/C as 0/1/2, nearest-trial-filled if absent, and repeated across time.

ii. `return int(np.argmin(np.abs(ZONES.mean(1)-pos)))`; `np.full(len(centers),z)`.

iii. Balanced full-data counts and presence of all three classes were used as sanity checks.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes solely from the sparse behavior `Reward` event timestamps.

ii. `rew_t=np.asarray(beh['Reward'].timestamps[:],float)`.

iii. The agent treats delivered Reward events as authoritative rather than using omission flags.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 when any reward timestamp lies inclusively between its start and teleport timestamps, otherwise 0; this value is repeated for every bin.

ii. `rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))`; `np.full(len(centers),rewarded)`.

iii. The notes report independent checks on rewarded, omission, and multi-plane trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid sessions/trials raise errors; missing reward-zone activation is nearest-neighbor-filled across trials; behavior beyond neural coverage is excluded; empty neural bins use the nearest neural sample; shape and finiteness assertions guard output. No general NaN imputation is done.

ii. `if not len(good): raise ValueError(...)`; `if b>a: ... else: xp[:,k]=neural_all[nearest_idx(...)]`; `assert ... and np.isfinite(X).all()`.

iii. These choices arose from multi-plane and incomplete-imaging failures found during validation; the notes document rerunning full conversion and checks after fixes.

## 13-a. What are the most time-consuming steps of the code?

i. Reading complete deconvolved matrices, per-bin neural means inside every trial/plane, serializing the 5.89-GiB pickle, and downstream decoder training are the costly operations. The conversion notes report a corrected full run of 124.5 seconds.

ii. `neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]`; nested `for ti...`, `for plane...`, `for k...`; `pickle.dump(...)`.

iii. The agent says one read per session and vectorized timestamp searches keep conversion under a few minutes; large NWB arrays and serialization dominate.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over temporal bins that computes means could be replaced by grouped/reduce operations. Trial and plane loops are natural but parts of zone extraction and binning could also be batched.

ii. `for k,(a,b) in enumerate(zip(ni,nj)): if b>a: xp[:,k]=neural_all[a:b].mean(0)`.

iii. The notes acknowledge the small per-bin loop but characterize it as bounded by trial duration; timestamp searches and interpolation were already vectorized.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly performs `searchsorted` and per-bin averaging for every plane and trial, creates repeated per-trial constant vectors, and scans each trial's reward-zone slice. It avoids repeated file reads by loading each session matrix once.

ii. `np.searchsorted(neural_t,centers-DT/2)` inside the trial/plane loops and multiple `np.full(len(centers),...)` calls.

iii. The agent specifically highlights one neural read per session and one-time curated-column extraction as implemented speedups.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `process_file` accepts an unused `show` argument and stores/prints extensive session metadata; `plane_name` is carried through tuples but unused in conversion. Plotting is optional and its figures are not consumed by the decoder. The required input variables themselves are saved although the provided decoder predicts only outputs from neural data.

ii. `def process_file(fn, show=False)`; `plane_data.append((plane_name,neural_t,neural_all))`; optional `plot_session(...)`.

iii. Plots and metadata were retained for auditability and visual sanity checking, not decoder computation.
