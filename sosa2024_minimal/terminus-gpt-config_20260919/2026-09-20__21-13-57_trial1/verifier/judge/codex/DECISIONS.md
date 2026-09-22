# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively identifies every `*_behavior+ophys.nwb` under subject directories, sorts the paths, and opens each with `h5py`. It loads behavior arrays eagerly but reads selected neural trial slices from HDF5.

ii. `files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*_behavior+ophys.nwb')))` and `with h5py.File(path, 'r') as f:`

iii. The trajectory says it inspected the HDF5 structure lazily because the NWBs are large, then reports processing all 152 files/sessions.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from each file's parent `sub-*` directory, deduplicated and sorted; each retained session gets the corresponding integer `subject_idx`.

ii. `subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-','') for p in files})` and `data['subject_idx'].append(subj_to_idx[subj])`

iii. The directory layout and filenames were inspected and treated as authoritative subject identifiers.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and contributes one entry to each session-level list, unless it has fewer than two retained trials.

ii. `for i,p in enumerate(files): nt, it, ot, nc, scene, rate, block = convert_session(p)`

iii. The trajectory explicitly describes all 152 NWBs as sessions and reports retaining all 152.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples in `trial_start`; each end is the first subsequent positive `teleport` sample, excluded from the slice. A start with no subsequent teleport is discarded.

ii. `starts = np.flatnonzero(trial_start > 0)` and `tq = np.flatnonzero(teleport[s:] > 0); e = s + int(tq[0]) if len(tq) else len(pos)`

iii. The agent stated that laps should start at the NWB `trial_start` pulse/0-cm crossing and stop at teleport onset because teleport activity is outside a corridor trial.

## 1-e. How are trials filtered based on quality controls?

i. It discards nonpositive spans, clipped final trials lacking teleport, trials yielding fewer than two rebinned samples, and entire sessions with fewer than two retained trials.

ii. `if e <= s: continue`, `if not len(tq): continue`, `if len(offsets) < 2: continue`, and `if len(nt) < 2: ... continue`

iii. The rationale is defensive handling of clipped/too-short data and compliance with the decoder's two-trial session minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from each plane in `processing/ophys/Deconvolved`, restricted using `PlaneSegmentation/iscell` and `planeIdx`.

ii. `dg = f['processing/ophys/Deconvolved']` and `iscell = seg['iscell'][:,0] == 1`

iii. The agent decided the supplied signal was already OASIS-deconvolved and therefore suitable, rather than rebuilding it from fluorescence and neuropil.

## 2-b. How is the `neural` data processed?

i. Accepted cells are selected plane-by-plane, concatenated across planes, averaged in blocks of four aligned frames, transposed to neuron-by-time, and stored as `float16`.

ii. `raw = np.concatenate(raw_parts, axis=1)` and `nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)`

iii. The agent aimed to harmonize acquisition metadata and reduce size with a common 257.934-ms bin; the trajectory shows it corrected an initial mistaken use of 31-Hz metadata by auditing timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains only `iscell == 1` ROIs, checks per-plane ROI counts and plane-rate agreement, but does not remove speed-correlated putative interneurons.

ii. `mask = iscell[plane_idx == pi]` and `if dset.shape[1] != len(mask): raise ValueError(...)`

iii. The agent interpreted `iscell` as Suite2p/manual curation. No trajectory justification was given for omitting the paper's interneuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays share behavioral frame indices, so each slice begins at `trial_start`; bin zero begins at that sample.

ii. `raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]`

iii. The agent identified `trial_start` as the requested event and described the streams as index-aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target is `4 / 15.5078125 = 0.257934...` s (257.934 ms). Four frame-aligned samples are averaged per bin; the final partial block is also averaged despite the helper comment saying it is dropped.

ii. `TARGET_DT = 4 / 15.5078125`, `block = int(round(TARGET_DT / dt))`, and `x[s:min(s+block, len(x))].mean(axis=0)`

iii. The trajectory documents an audit showing all aligned timestamps are effectively 15.5078125 Hz, including 28 files whose ophys attribute says 31 Hz; timestamps were made authoritative to ensure a common bin duration.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The emitted values are synthetic bin indices multiplied by `TARGET_DT`; raw position timestamps are used indirectly to establish the common sample interval and trial reward window.

ii. `np.arange(T, dtype=np.float32) * np.float32(TARGET_DT)`

iii. Uniform, trial-start-aligned bins made an arithmetic time vector natural; the agent's timestamp audit established the interval.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. It creates `0, TARGET_DT, 2*TARGET_DT, ...` after rebinning, with no subtraction of actual per-trial timestamps.

ii. `inp = np.vstack([np.arange(T, dtype=np.float32) * np.float32(TARGET_DT), ...])`

iii. The trial begins at time zero and sampling is treated as uniform.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It has exactly one value per neural block, starts at zero, and uses the same `T = len(offsets)`.

ii. `T = len(offsets)` followed by construction of both `nbin` and the length-`T` input.

iii. The agent relied on common frame indices and common block starts.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data` within the trial.

ii. `env = b['environment/data'][:]`

iii. The agent notes that the NWB stream directly encodes ENV1/ENV2 and is constant per trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The trial median is rounded, converted to 0 if nonpositive or 1 if positive, then repeated at every rebinned time point.

ii. `ev = int(round(float(np.median(env[s:e])))); ev = 1 if ev > 0 else 0`

iii. This enforces the requested binary per-trial representation.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the median of the NWB `trial number/data` samples within each trial.

ii. `trial_num = b['trial number/data'][:]` and `tr = int(round(float(np.median(trial_num[s:e]))))`

iii. The agent treated the stored trial-number stream as authoritative and later sorted records by it to guarantee chronological order.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The within-trial median is rounded to an integer and repeated over all bins; it is also used for sorting and reward-zone switching.

ii. `np.full(T, tr, np.float32)` and `trial_records.sort(key=lambda x: x[0])`

iii. Median aggregation is intended to robustly obtain a per-trial constant.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from NWB `Reward/timestamps`, current trial timestamp windows, and the preceding record in stored trial-number order.

ii. `reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)`

iii. The agent stated that previous outcome must follow actual chronological order, not incidental list order.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current reward is whether any reward timestamp lies in `[trial_start_time, trial_end_time)`. After sorting trials, the first previous outcome is 0 and later trials receive the preceding trial's reward flag, repeated through time.

ii. `rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))` and `inp[3,:] = prev; prev = rewarded`

iii. This implements omitted=0/rewarded=1 and explicitly handles the first trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses block-averaged `position/data` plus zone identities parsed from the NWB root `identifier`; fixed A/B/C coordinates are hard-coded. On switch sessions it selects the final parsed zone for stored trial numbers 30 onward.

ii. `ZONE_COORDS = {'A': (80.,130.), 'B': (200.,250.), 'C': (320.,370.)}` and `zone = z1 if (switched and tr >= 30) else z0`

iii. The agent says these coordinates and the trial-30 switch follow `reward_relative.behavior` and repository defaults, while identifiers preserve scene names.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is averaged within temporal blocks. Signed distance is position minus the lower edge before the zone, zero inside it, or position minus the upper edge after it.

ii. `pbin = binned_mean(pos[s:e], ...)` and `d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))`

iii. This represents distance to the nearest reward-zone location with sign indicating before/after.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven requested categories at -50, -10, 0, 10, and 50 cm.

ii. `y[d < -50] = 0`, `y[(d >= -50) & (d < -10)] = 1`, ..., `y[d > 50] = 6`

iii. The boundaries were copied from the decoder specification, with exact zero reserved for inside-zone class 3.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural samples use the same trial slice and block starts; both are averaged per block before distance categorization.

ii. `nbin = binned_mean(raw, np.arange(0, e-s, block), block)` and `pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)`

iii. The agent relies on shared frame indices and timestamps.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. `pos = b['position/data'][:]`

iii. This is the direct VR corridor position stream.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is averaged in the same temporal blocks as neural activity and then discretized.

ii. `pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)`

iii. Averaging was chosen as the continuous-variable aggregation under common temporal rebinning.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` uses thresholds 90, 180, 270, and 360 cm (`right=False`), producing classes 0--4.

ii. `pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)`

iii. These are five equal 90-cm divisions of the 450-cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial indices and block boundaries as neural activity.

ii. Both calls use `np.arange(0, e-s, block)`.

iii. Shared frame indexing was considered authoritative.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. `lick = b['lick/data'][:]`

iii. The agent identified this as the direct lick stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Each output bin is 1 if any raw lick sample in that block is positive, otherwise 0.

ii. `lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)`

iii. The agent chose an any-event reduction so temporal averaging would not erase sparse licks.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick blocks begin at the same absolute `offsets` as neural blocks and terminate at the same trial end.

ii. `offsets = np.arange(s, e, block, dtype=int)` and the `lbin` comprehension above.

iii. Alignment follows the shared frame index.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from reward-location letters parsed from the NWB `identifier`, fixed coordinate/ID maps, and the stored trial number on switch sessions; the raw `reward_zone` stream is not used.

ii. `scene, z0, z1, switched = scene_info(f['identifier'][()])` and `ZONE_ID = {'A':0, 'B':1, 'C':2}`

iii. The agent believed identifiers and repository switch conventions reliably encoded the scene and zone.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex extracts A/B/C occurrences. The first/last become pre/post-switch zones; at trial 30 the code switches if they differ, maps the letter to 0/1/2, and repeats it through time.

ii. `locs = re.findall(r'(?:Location)?([ABC])', scene)` and `np.full(T, ZONE_ID[zone], np.int8)`

iii. The logic was said to cover ordinary, within-environment switch, and cross-environment identifier forms.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from the timestamps in the behavioral `Reward` TimeSeries and the position timestamp interval of each trial.

ii. `rg = b['Reward']` and `reward_times = rg['timestamps'][:]`

iii. The agent correctly recognized Reward as an event series with separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is rewarded if any reward timestamp is at least its start timestamp and before one effective sample after its last included frame; the binary value is repeated for all bins.

ii. `t0, t1 = timestamps[s], timestamps[e-1] + 1/rate` and `np.full(T, rewarded, np.int8)`

iii. This implements the requested per-trial event-presence label.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files cause failure; plane-rate or ROI-count inconsistencies raise errors. A missing Reward timestamp dataset is treated as no rewards. Trials without teleport and trials shorter than two output bins are skipped. Sessions below two trials are skipped. The median timestamp interval works around incorrect 31-Hz ophys metadata. No NaN repair, neural/behavior cropping, or noisy/missing reward-zone inference is implemented.

ii. `reward_times = ... if 'timestamps' in rg else np.empty(0)`, `if not len(tq): continue`, and the explicit `raise ValueError(...)` checks.

iii. The trajectory highlights the 28-file rate-metadata audit and correction; other checks are defensive consistency measures.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large per-trial deconvolved HDF5 slices, repeatedly averaging variable-length blocks, serializing the roughly 1.15-GiB pickle, and verification are the likely dominant steps.

ii. `raw_parts = [dset[s:e, :][:, idx] ...]`, `binned_mean(...)`, and `pickle.dump(...)`

iii. The trajectory repeatedly notes the large NWBs and long full-dataset conversion/verification, though it provides no benchmark breakdown.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops over trials, block starts in `binned_mean`, lick blocks, and planes could partly be vectorized for full blocks; variable trial lengths and final partial blocks complicate this.

ii. `[np.asarray(x[s:min(s+block, len(x))]).mean(axis=0) for s in starts]` and `[np.any(lick[q:min(q+block,e)] > 0) for q in offsets]`

iii. No explicit trajectory justification addresses vectorization; the chosen loops make variable boundaries straightforward.

## 13-c. What processing does the code repeat multiple times?

i. It recreates the same block-start arrays for neural, position, and speed, scans `teleport[s:]` anew for every trial, and repeatedly slices each plane per trial.

ii. Three calls use `np.arange(0, e-s, block)`, and every trial runs `np.flatnonzero(teleport[s:] > 0)`.

iii. The trajectory does not discuss these repetitions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads speed and computes/bins/discretizes it because speed is a requested output, so that work is retained. More arguably unnecessary work includes loading full-session behavior arrays, calculating/checking ophys `rates` before replacing them with timestamp-derived rate, and computing metadata fields not consumed by decoding.

ii. `rates = [...]`, then later `rate = 1.0 / dt`; also `speed = b['speed/data'][:]` and `speed_cls = np.digitize(...)` (retained output).

iii. No trajectory rationale identifies discarded processing; most work contributes to output or validation metadata.
