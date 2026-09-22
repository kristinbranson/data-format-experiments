# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively gathers every `/app/data/sub-*/*.nwb` file, sorts them, and opens each directly with `h5py`. It retains sessions having at least two trials and one selected cell.

ii. `files = sorted(glob.glob(DATA_ROOT + '/sub-*/*.nwb'))` and `with h5py.File(f,'r') as h:`.

iii. The trajectory says the release contains 152 NWBs and concludes that the paper's 77-session count is a task subset, not a general quality filter. It chose selective HDF5 reads because the release is about 87 GB.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from `sub-*` parent directories, naturally sorted, and each session is mapped using the NWB `subject_id`.

ii. `subjects = sorted({Path(f).parent.name.removeprefix('sub-') for f in files}, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)`; `subject = text(h['general/subject/subject_id'][()])`.

iii. The agent observed 11 subject directories and treated directory and NWB metadata as consistent identifiers.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; one list element is appended to `neural`, `input`, and `output` per retained file.

ii. `for si,f in enumerate(files): ... neural.append(ns); inputs.append(xs); outputs.append(ys)`.

iii. The trajectory identifies 152 NWB session files and uses their session IDs only as metadata, preserving the file as the session unit.

## 1-d. How are the data split into trials?

i. Trial starts are every nonzero `trial_start` sample and ends are every nonzero `teleport` sample. Invalid pairs where end is not after start are dropped. Each trial is the half-open interval `[start, end)`.

ii. `starts = np.flatnonzero(b['trial_start/data'][:])`; `ends = np.flatnonzero(b['teleport/data'][:])`; `valid = (ends > starts)`; `sl = slice(int(a),int(e))`.

iii. The agent found no NWB trials table and traced repository trial matrices to `trial_start`/`teleport`; it chose complete laps from start through, but excluding, teleport.

## 1-e. How are trials filtered based on quality controls?

i. No duration/quality threshold is applied. Only boundary pairs with `end > start` survive; a whole session is skipped if it has fewer than two trials or no selected cells.

ii. `valid = (ends > starts)` and `if len(ns) < 2 or n_cells==0: continue`.

iii. The agent argued that the paper's speed exclusion applied only to particular spatial analyses, not this decoder. It did not discuss or implement the reference's removal of trials shorter than 50 frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from every response series under `processing/ophys/Deconvolved`, with ROI selection from `ImageSegmentation/PlaneSegmentation/iscell`.

ii. `deconv = h['processing/ophys/Deconvolved']`; `event_sets = [deconv[q]['data'] for q in plane_names]`; `iscell = ...['iscell'][:,0] > 0`.

iii. The agent interpreted the methods and repository as using deconvolved activity and believed the NWB Suite2p events were the appropriate paper-aligned signal.

## 2-b. How is the `neural` data processed?

i. Stored deconvolved values are selected by trial and curated ROI, transposed to neuron-by-time, and concatenated across planes. The code performs no fluorescence correction, dF/F baseline calculation, smoothing, or new deconvolution.

ii. `n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T for d,ids in zip(event_sets,plane_cell_ids)],axis=0)`.

iii. The trajectory explicitly chose the stored `Deconvolved` interface as the paper's event representation and focused on correctly concatenating the 28 dual-plane sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains only ROIs with the first `iscell` column positive. It does not remove putative interneurons using dF/F–speed correlation.

ii. `iscell = ...[:,0] > 0`; `plane_cell_ids.append(np.flatnonzero(iscell[off:off+width]))`.

iii. The agent regarded `iscell` as Suite2p/manual cell curation. Its trajectory does not justify omitting the paper's interneuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural frames are sliced from the trial-start frame inclusive, making column zero the alignment event. No further shift is applied.

ii. `sl = slice(int(a),int(e))` followed by selection of each deconvolved series with `d[sl,ids]`.

iii. The agent found behavior and neural streams share a frame clock and aligned them by common frame index.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Metadata declares `1000/15.5078125 = 64.4839 ms` per frame.

ii. `'time_bin_size':1000.0/15.5078125`.

iii. The trajectory measured the behavior interval and concluded the streams already use the native common imaging rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps` and the trial-start index.

ii. `ts = b['position/timestamps'][:]`; `t = np.asarray(ts[sl]-ts[a], dtype=np.float32)`.

iii. The agent found the behavior series' timestamps mutually consistent and used position timestamps as the shared clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial's first frame is subtracted from every timestamp in that trial and values are cast to float32.

ii. `t = np.asarray(ts[sl]-ts[a], dtype=np.float32)`.

iii. This directly realizes time since the alignment event without resampling.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same half-open frame slice is used for timestamps and neural arrays. A neural stream may have one unused terminal frame, which is ignored.

ii. `sl = slice(int(a),int(e))`; both `ts[sl]` and `d[sl,ids]` use it.

iii. The agent inspected mismatched dual-plane files and found exactly one extra terminal neural frame in ten files, with shared start/rate and all trials ending earlier.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is parsed from the NWB `identifier` scene string, not read from the framewise raw `environment` series.

ii. `scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])`; regex `r'Env([12])(?:_Location)?([ABC])'`.

iii. The agent reasoned that scene identity is encoded in the identifier and used it to support fixed- and cross-environment conditions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `Env1`/`Env2` strings become 0/1. For scenes containing `_to_`, the code switches from the first to second parsed environment at trial-loop index 30, then repeats the scalar across all trial frames.

ii. `after = switched and ti >= 30`; `env = e1 if after else e0`; `np.full(ntime,env,np.float32)`.

iii. The trajectory inferred trial 30 as the repository's default switch point and required a binary input.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It normally uses `behavior/trial number/data` at the trial-start frame, falling back to the zero-based trial-loop index for a nonfinite or negative value.

ii. `trialnum_all = b['trial number/data'][:]`; `trnum = float(trialnum_all[a])`; `if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)`.

iii. The trajectory identified the raw trial-number stream as available and retained it, adding a defensive fallback.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. One scalar is sampled at trial start, conditionally repaired, cast to float, and repeated over the trial.

ii. `np.full(ntime,trnum,np.float32)`.

iii. This makes the per-trial variable compatible with the time-varying input matrix.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward/timestamps` and position timestamps defining each trial interval.

ii. `reward_times = b['Reward/timestamps'][:]`; `np.any((reward_times >= ts[a]) & (reward_times < ts[e]))`.

iii. The agent treated actual reward delivery inside the lap as the binary outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current-trial outcomes are precomputed by interval membership. Trial zero gets 0; later trials receive the preceding binary outcome repeated over time.

ii. `prev = float(outcomes[ti-1] if ti else 0)`; `np.full(ntime,prev,np.float32)`.

iii. This directly implements omitted=0/rewarded=1 and defines the unavailable first previous outcome as 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw framewise position plus a zone label parsed from the NWB identifier. It does not derive the active zone from raw `reward_zone` behavior.

ii. `pos_all = b['position/data'][:]`; `scene,e0,e1,z0,z1 = parse_scene(...)`; `ZONE_COORDS = {'A': (80.0,130.0), ...}`.

iii. The agent mapped task labels A/B/C to the repository's physical zones and assumed identifier plus the trial-30 switch fully specifies the active zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position before the interval is expressed relative to its lower edge, position after it relative to its upper edge, and positions inside are zero; the result is immediately categorized.

ii. `d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))`.

iii. The agent describes this as signed distance to the nearest point in the active zone: negative before, zero within, positive after.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit Boolean masks implement the requested bounds, including -50 in class 1, -10 in class 2, exactly zero in class 3, and +10 in class 4.

ii. `y[d < -50]=0`; `y[(d >= -50)&(d < -10)]=1`; `y[(d >= -10)&(d < 0)]=2`; `y[d==0]=3`; `y[(d>0)&(d<=10)]=4`; `y[(d>10)&(d<=50)]=5`; `y[d>50]=6`.

iii. The masks were written to mirror the decoder specification literally.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity use the identical per-trial frame slice, so categories align column-for-column.

ii. `pos = np.asarray(pos_all[sl], ...)` and neural uses `d[sl,ids]`.

iii. The agent established that behavior had already been resampled to imaging frames.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from `behavior/position/data`.

ii. `pos_all = b['position/data'][:]`; `pos = np.asarray(pos_all[sl], dtype=np.float32)`.

iii. The agent identified this as the centimeter coordinate on the 450 cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial raw values are sliced and categorized with four internal cut points; there is no smoothing or clipping.

ii. `np.digitize(pos,[90,180,270,360],right=False).astype(np.int8)`.

iii. Four thresholds create five equal 90 cm bins as required.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Values below 90 map to 0; thresholds 90, 180, 270, and 360 begin classes 1–4 respectively.

ii. `np.digitize(pos,[90,180,270,360],right=False)`.

iii. This directly follows the specified five equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural arrays are indexed by the same trial slice.

ii. `pos_all[sl]` and `d[sl,ids]`.

iii. Framewise behavior and imaging events were found to share indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `behavior/lick/data`.

ii. `lick_all = b['lick/data'][:]`.

iii. The agent observed that this stream contains per-frame lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive count becomes 1 and all other values become 0, cast to int8.

ii. `(np.asarray(lick_all[sl])>0).astype(np.int8)`.

iii. Binarization is needed because the requested decoder output is no/yes rather than a count.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks and neural data use the same half-open frame slice.

ii. `lick_all[sl]` and `d[sl,ids]`.

iii. The raw lick series was found already aligned to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived solely from A/B/C labels parsed from the NWB identifier, rather than the raw reward-zone/position streams.

ii. `scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])` and `ZONE_INDEX = {'A':0,'B':1,'C':2}`.

iii. The agent believed the scene identifier was a reliable declarative encoding of task and zone identity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code selects the first parsed zone before trial 30 and the second thereafter in switch scenes, maps A/B/C to 0/1/2, and repeats it over time.

ii. `zone = z1 if after else z0`; `np.full(ntime,ZONE_INDEX[zone],np.int8)`.

iii. The trial-30 choice came from the repository's default switch convention.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from sparse raw `Reward/timestamps` and the position-series timestamps at trial boundaries.

ii. `reward_times = b['Reward/timestamps'][:]`; interval comparison against `ts[a]` and `ts[e]`.

iii. The agent defined outcome as actual delivery during the trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp lies in `[trial start, teleport)`, otherwise 0; the value is repeated across the trial.

ii. `outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))`; `np.full(ntime,outcomes[ti],np.int8)`.

iii. The binary trial-level result matches the requested no/yes outcome and avoids inventing frame-level reward timing.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files and unmatched boundary counts raise errors. Nonpositive trial intervals are dropped. Invalid trial numbers fall back to the loop index. Plane/ROI dimensions are checked. Neural streams may exceed behavior by at most one terminal frame; that frame is ignored. Sessions with fewer than two trials or zero cells are skipped. Other missing values are not imputed.

ii. `if len(starts) != len(ends): raise ValueError(...)`; `valid = (ends > starts)`; `if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)`; `if any(d.shape[0] < len(ts) or d.shape[0]-len(ts) > 1 ...): raise ValueError(...)`.

iii. The agent investigated all frame mismatches and found ten dual-plane files with exactly one extra terminal neural frame, so it allowed only that observed case while rejecting broader mismatches.

## 13-a. What are the most time-consuming steps of the code?

i. Reading deconvolved arrays from 152 large NWBs, materializing a separate neural matrix for every trial, and serializing the roughly 9.7 GB pickle dominate runtime and I/O.

ii. `np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T ...])` inside the trial loop and `pickle.dump(data,fp,protocol=4)`.

iii. The trajectory repeatedly notes the 87 GB source, selective reading, conversion polling, and potentially lengthy 9.73 GB serialization.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Reward outcome interval checks and per-trial construction/slicing could be partly vectorized or computed once at session scale; the trial loop remains useful because trials have variable lengths. Plane concatenation is also repeated per trial.

ii. `for a,e in zip(starts,ends): outcomes.append(...)`; `for ti,(a,e) in enumerate(zip(starts,ends)):`; list comprehension across `event_sets`.

iii. The trajectory prioritizes streaming/selective access and correctness across variable-length trials; it does not explicitly discuss vectorization.

## 13-c. What processing does the code repeat multiple times?

i. It makes one pass over trials to compute outcomes and a second to build arrays. In every trial it allocates constant environment, trial-number, previous-outcome, zone, and outcome vectors and reselects/concatenates neural plane data.

ii. The two `for ... zip(starts,ends)` loops and repeated `np.full(...)`/`np.concatenate(...)` calls.

iii. The first pass is justified by needing the preceding trial's result, though outcomes could have been computed more compactly before assembly.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It converts continuous position and speed to float32 trial copies even though only categorical outputs are retained; it also constructs detailed session metadata not used by decoder training. Most work otherwise feeds the saved dataset.

ii. `pos = np.asarray(..., dtype=np.float32)`; `speed = np.asarray(..., dtype=np.float32)`; `session_info.append({...})`.

iii. The agent retained metadata for auditability and intermediate continuous arrays for straightforward binning; it did not identify these as discarded work in the trajectory.
