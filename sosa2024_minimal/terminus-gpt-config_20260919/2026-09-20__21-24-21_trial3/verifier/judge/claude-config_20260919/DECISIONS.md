# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `*.nwb` file under every `sub-*` directory of `/app/data` is globbed and processed in sorted order, one file per session; all 152 released NWB files are used. Files are opened directly with `h5py` (not `pynwb`) and only the specific HDF5 datasets that are needed are read (behaviour time series, `Deconvolved` event arrays, `iscell`, and a few scalar metadata fields). There is a single pass over the data: no separate survey/inspection pass and no caching.

ii.
```python
DATA_ROOT = '/app/data'
...
def convert():
    files = sorted(glob.glob(DATA_ROOT + '/sub-*/*.nwb'))
    if not files:
        raise FileNotFoundError('No NWB files found')
    ...
    for si,f in enumerate(files):
        with h5py.File(f,'r') as h:
            subject = text(h['general/subject/subject_id'][()])
            session_id = text(h['general/session_id'][()])
            scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])
            b = h['processing/behavior/BehavioralTimeSeries']
            ...
            deconv = h['processing/ophys/Deconvolved']
```

iii. From the trajectory: the agent first inventoried the DANDI/BIDS hierarchy and found "152 NWB session files across 11 subjects", with file sizes from ~90 MB to >800 MB, and concluded that "conversion must read only processed ROI/time-series datasets, not image stacks". It explicitly abandoned `pynwb` after observing that "opening 152 large NWBs through PyNWB is unnecessarily slow" and switched to raw `h5py` reads. It also considered whether the paper's *n* = 77 figure implied a session subset, inventoried the scene names, found "exactly 77 reward-switch sessions and 75 fixed-location sessions, totaling 152", and concluded the 77 is "the switch-session cohort for remapping analyses, not a general quality exclusion", so all 152 sessions are retained.

## 1-b. How are the data split into subjects?

i. Subjects are the parent directory names of each NWB file with the `sub-` prefix stripped (`m3`, `m4`, `m7`, `m11`, …). The unique set is sorted numerically by the integer after the leading `m`, giving the 11 subjects. Each session's `subject_idx` is looked up from that map; the subject id stored in the file (`general/subject/subject_id`) is what is actually used as the lookup key.

ii.
```python
subjects = sorted({Path(f).parent.name.removeprefix('sub-') for f in files},
                  key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
subj_map = {s:i for i,s in enumerate(subjects)}
...
subject = text(h['general/subject/subject_id'][()])
...
subject_idx.append(subj_map[subject])
```

iii. Not discussed explicitly in the trajectory beyond the inventory step, where the agent noted "The DANDI release has 152 sessions: 14 for every mouse except m11 (12)" and "a large (87 GB) DANDI/BIDS-style hierarchy with 11 mice", i.e. the directory layout was taken at face value as the subject split.

## 1-c. How are the data split into sessions?

i. One session per NWB file. No merging across days and no cross-day ROI alignment is attempted. `general/session_id` is recorded in `metadata['session_info']` for traceability. A session is dropped only if it ends up with fewer than two trials or zero curated cells (which never happens in this dataset).

ii.
```python
    for si,f in enumerate(files):
        with h5py.File(f,'r') as h:
            ...
            session_id = text(h['general/session_id'][()])
...
            if len(ns) < 2 or n_cells==0:
                continue
            neural.append(ns); inputs.append(xs); outputs.append(ys)
            ...
            session_info.append({'file':os.path.relpath(f,DATA_ROOT),'subject':subject,
                'session_id':session_id,'scene':scene,'n_trials':len(ns),
                'n_cells':n_cells,'switch_trial':30 if switched else None})
```

iii. The agent's docstring states: "Every released session is retained. The 77-session number in the paper is the reward-switch subset (there are 77 switch and 75 fixed-zone NWBs), not a general quality filter." The trajectory adds that "the task asks for the full converted dataset and fixed-location sessions contain all required variables, [so] all 152 should be retained".

## 1-d. How are the data split into trials?

i. Trial boundaries come from the two impulse channels in `BehavioralTimeSeries`: a trial runs from a `trial_start` frame (inclusive) to the next `teleport` frame (exclusive), i.e. the half-open interval `[trial_start, teleport)`. Starts and ends are obtained as `np.flatnonzero` of each channel; the code asserts the two counts are equal and keeps only pairs with `end > start`. This yields 12,216 trials over the 152 sessions.

ii.
```python
            starts = np.flatnonzero(b['trial_start/data'][:])
            ends = np.flatnonzero(b['teleport/data'][:])
            if len(starts) != len(ends):
                raise ValueError(f'Unmatched boundaries in {f}')
            valid = (ends > starts)
            starts, ends = starts[valid], ends[valid]
...
            for ti,(a,e) in enumerate(zip(starts,ends)):
                sl = slice(int(a),int(e))
                ntime = e-a
```

iii. The agent read the paper's repository and reported that "trial matrices are bounded by `trial_start_inds` and `teleport_inds`", and that `trial_start`/`teleport` in the NWB are "impulses" rather than state channels. Its docstring records: "Complete trials are [trial_start, teleport), matching get_trial_types and trial-matrix calls in the supplied repository."

## 1-e. How are trials filtered based on quality controls?

i. Essentially no trial-level quality control is applied. The only trial rejection is the structural `ends > starts` mask (which removes nothing in this dataset). There is no minimum-trial-length filter and, deliberately, no running-speed filter. Sessions with fewer than two trials are dropped (none are).

ii.
```python
            valid = (ends > starts)
            starts, ends = starts[valid], ends[valid]
...
            if len(ns) < 2 or n_cells==0:
                continue
```

iii. The agent explicitly reasoned about the paper's <2 cm/s exclusion and decided it does not apply here: "movement below 2 cm/s was excluded only for specific spatial/time-warp neural analyses, not universally; therefore it should not be applied to this time-aligned decoder." Its docstring repeats: "No speed filtering is used: the <2 cm/s exclusion in the paper applies only to specified spatial maps." It gives no justification for omitting a minimum-length trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are read verbatim from `processing/ophys/Deconvolved/plane<N>/data` — Suite2p's own deconvolved traces stored in the NWB. `Fluorescence` (F) and `Neuropil` (Fneu) are never read. Columns are restricted to ROIs flagged by `ImageSegmentation/PlaneSegmentation/iscell[:,0]`, and for the 28 two-plane sessions the single concatenated `iscell` vector is split by plane widths (plane0 block then plane1 block) and the per-plane selections are concatenated along the neuron axis.

ii.
```python
            deconv = h['processing/ophys/Deconvolved']
            plane_names = sorted(deconv.keys(), key=lambda q: int(re.search(r'([0-9]+)$',q).group(1)))
            event_sets = [deconv[q]['data'] for q in plane_names]
...
            iscell = h['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:,0] > 0
            widths = [d.shape[1] for d in event_sets]
            if sum(widths) != len(iscell):
                raise ValueError(f'Plane ROI widths do not match segmentation in {f}')
            plane_cell_ids=[]; off=0
            for width in widths:
                plane_cell_ids.append(np.flatnonzero(iscell[off:off+width]))
                off += width
...
                n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. The agent's docstring justifies this as "Suite2p deconvolved events are used, as in the paper's population/GLM analyses. Only ROIs for which Suite2p `iscell[:, 0]` is true are retained." In the trajectory it repeatedly equated the stored `Deconvolved` array with the paper's signal ("dayData documentation confirms analyses use deconvolved activity", "GLM code uses deconvolved `events`") and then wrote "use raw deconvolved events at native 15.5078125 Hz". It never opened `preprocessing.py::dff` or compared the stored array against the paper's own dF/F→OASIS pipeline.

## 2-b. How is the `neural` data processed?

i. No processing at all beyond ROI selection, dtype cast to `float32`, and transposition to (n_neurons, n_timepoints). No neuropil subtraction, no maximin baseline, no dF/F normalisation, no Gaussian smoothing, no OASIS deconvolution, no per-plane frame-rate handling of the kernel — the stored Suite2p values are used as-is.

ii.
```python
                # HDF5 selection is time x selected ROI; transpose to neuron x time.
                n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. Implicit in 2-a: the agent believed the stored `Deconvolved` array already *is* the paper's event signal, so no further processing was thought necessary. The trajectory contains no discussion of the Methods paragraph describing the per-trial maximin baseline, the dF/F formula, the two-sample Gaussian, or the OASIS `tau = 0.7` deconvolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: the Suite2p `iscell` classification (`iscell[:,0] > 0`), applied per plane. This keeps 138,678 cells across the 152 sessions. The paper's second curation step — excluding putative interneurons whose dF/F correlates with running speed at *r* > 0.5 — is not applied. All retained cells are labelled `CA1`.

ii.
```python
            iscell = h['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:,0] > 0
...
            n_cells = int(sum(map(len,plane_cell_ids)))
...
            region_idx.append(np.zeros(n_cells,dtype=np.int16))
...
      'brain_regions': ['CA1'], 'brain_region_idx': region_idx,
```

iii. The docstring justifies `iscell` as the paper's manual curation: "Only ROIs for which Suite2p `iscell[:, 0]` is true are retained." The agent noted in the trajectory that "Keeping every classified cell would produce ~13.3 GB of neural float32 data before pickle overhead, suggesting careful confirmation of paper session/cell curation is important", but then concluded "I will retain Suite2p `iscell` ROIs, matching the paper's cell curation". The interneuron exclusion is never mentioned anywhere in the trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires nothing beyond the trial slicing itself: neural and behavioural streams are both stored at one sample per imaging frame on a common clock starting at *t* = 0, so the same frame index `[trial_start, teleport)` is applied to both. `metadata['temporal_alignment_event']` is set to "start of trial (entry to linear track)" with `off_start = 0.0` and `off_end = None`. Ten (dual-plane) sessions have exactly one *extra* trailing neural frame; the code permits a surplus of at most one frame and ignores it, and raises if a neural stream is ever *shorter* than the behaviour stream.

ii.
```python
            # Ten dual-plane files contain one unused trailing neural frame.
            # Both streams start at t=0 at the imaging rate; align by frame and
            # ignore that terminal sample (all trial teleports precede it).
            if any(d.shape[0] < len(ts) or d.shape[0]-len(ts) > 1 for d in event_sets):
                raise ValueError(f'Behavior/neural frame mismatch in {f}')
...
                sl = slice(int(a),int(e))
...
        'temporal_alignment_event':'start of trial (entry to linear track)',
        'off_start':0.0, 'off_end':None,
```

iii. The agent verified alignment empirically: "Neural response series use `starting_time` rather than explicit timestamps, while behavior starts at 0 with the exact native 15.5078125 Hz interval. In mismatched sessions neural has one extra trailing sample, and all trial slices are safely within both streams. The correct alignment is therefore by shared frame index while ignoring the unused extra terminal neural sample." It also checked that "Only 10 sessions mismatch, and in every case both neural planes have exactly one extra frame while all 80 complete behavior trials fit inside the neural arrays."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame is kept: 1 / 15.5078125 s = 64.4836 ms per bin, written into `metadata['time_bin_size']` as a hard-coded literal. No rebinning, resampling, interpolation or smoothing is performed, on either the neural or the behavioural streams. The value is not read from the file, so the two-plane sessions (whose stored `rate` is the 31.015625 Hz scanner rate) are implicitly assumed to have a 15.5078125 Hz per-plane rate.

ii.
```python
        'time_bin_size':1000.0/15.5078125,
```

iii. The agent established the rate from the methods and from the behaviour timestamps: "The methods establish the native common sampling rate (~15.5 Hz), deconvolved calcium events as the paper's neural response, and behavioral streams sampled at imaging frames", and later "behavior starts at 0 with the exact native 15.5078125 Hz interval". It planned to "use raw deconvolved events at native 15.5078125 Hz".

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From `processing/behavior/BehavioralTimeSeries/position/timestamps`, the per-frame behaviour clock in seconds (identical to the timestamps attached to every other behaviour series).

ii.
```python
            ts = b['position/timestamps'][:]
...
                t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
```

iii. Not discussed separately; the agent established generally that "all streams are already imaging-frame aligned" and that behaviour timestamps run from 0 at the exact imaging interval, so any behaviour series' timestamps are interchangeable.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the timestamp of the trial's first frame from the whole trial slice, then cast to `float32`. Each trial therefore starts at exactly 0.0 s and increments by ~0.0645 s per sample.

ii.
```python
                t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
                ...
                x = np.vstack((t,
                    np.full(ntime,env,np.float32),
                    np.full(ntime,trnum,np.float32),
                    np.full(ntime,prev,np.float32)))
```

iii. Implied by the required alignment ("Temporally align based on start of the trial"); no separate justification is given in the trajectory.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced with the same frame indices `[a, e)` used for the neural matrix, so the two are aligned by construction and have identical `n_timepoints`. No resampling or interpolation is needed.

ii.
```python
                sl = slice(int(a),int(e))
                ntime = e-a
                ...
                t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
                ...
                n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```

iii. "Trials can be sliced directly from start through teleport, all streams are already imaging-frame aligned" (trajectory, step 20).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the NWB root `identifier` string, e.g. `/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A`. The last path component ("scene") is parsed with regular expressions to recover the environment (`Env1` → 0, `Env2` → 1) and the pre-/post-switch reward-zone labels. The per-frame `environment` behaviour time series, which is also present in every file, is deliberately *not* used.

ii.
```python
def parse_scene(identifier):
    """Return scene, environment (0/1), first/second zone labels."""
    scene = text(identifier).rstrip('/').split('/')[-1]
    # Environment type requested is binary. Cross-environment sessions switch
    # environment along with zone; ordinary sessions have one EnvN prefix.
    pairs = re.findall(r'Env([12])(?:_Location)?([ABC])', scene)
    if len(pairs) >= 2:
        (e0,z0),(e1,z1) = pairs[0],pairs[1]
    else:
        em = re.search(r'Env([12])', scene)
        if not em:
            raise ValueError(f'Cannot parse environment from {scene}')
        e0=e1=em.group(1)
        ...
    return scene, int(e0)-1, int(e1)-1, z0, z1
```

iii. The agent decided against the stored channel on the basis of a value survey: "environment is mostly 0 with -1 events … We must decode per-trial environment and reward-zone from event semantics or descriptions rather than treating their raw frame values as state", and then "Environment input likewise should come from the scene/task condition, not the sparse raw `environment` channel." Once it found the scene in the identifier it wrote: "The root `identifier` contains the complete scene (`Env1_LocationB_to_A`), solving both environment and reward-zone labeling."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The stated processing is: parse the scene; if the scene names two `Env<digit><Zone>` pairs (the cross-environment "day 8" sessions, e.g. `Env1_B_to_Env2_C`) use the first environment for trials 0–29 and the second from trial 30 onward; otherwise the single `EnvN` applies to the whole session. The value is broadcast as a constant across all timepoints of the trial.

   In practice the regex `Env([12])(?:_Location)?([ABC])` cannot match `Env1_B` (there is an underscore between the digit and the zone letter, and `(?:_Location)?` does not accept a bare `_`). `pairs` therefore never reaches length 2, every scene falls into the `else` branch, and `e1` is always set equal to `e0`. The environment consequently never switches: in the 11 cross-environment sessions all 570 post-switch trials are labelled with the pre-switch environment. (The zone labels `z0`/`z1` are unaffected — the `else` branch's second regex recovers them correctly, so reward-zone labelling is right in all 12,216 trials.)

ii.
```python
    pairs = re.findall(r'Env([12])(?:_Location)?([ABC])', scene)
    if len(pairs) >= 2:
        (e0,z0),(e1,z1) = pairs[0],pairs[1]
    else:
        em = re.search(r'Env([12])', scene)
        ...
        e0=e1=em.group(1)
        # Handles LocationA, LocationA_to_B, and A_to_Env1_C forms.
        zs = re.findall(r'(?:Location)?([ABC])', scene)
        ...
        z0, z1 = zs[0], (zs[-1] if '_to_' in scene else zs[0])
    return scene, int(e0)-1, int(e1)-1, z0, z1
```
```python
                after = switched and ti >= 30
                env = e1 if after else e0
                zone = z1 if after else z0
...
                x = np.vstack((t,
                    np.full(ntime,env,np.float32),
```

iii. The docstring states the intent: "Cross-environment sessions switch environment along with zone; ordinary sessions have one EnvN prefix", and "Paper code maps task labels A/B/C to physical zones X/Y/Z … and uses trial 30 as the default switch." The trajectory confirms the trial-30 constant was taken from the paper's code and spot-checked on one session: "The example switches from physical zone Y (label B, 200–250 cm) to X (label A, 80–130 cm) exactly at trial 30, consistent with paper code." The agent never validated the cross-environment branch of its own parser.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the `trial number` behaviour time series, sampled at the trial's first frame. A fallback to the loop index is provided if the stored value is non-finite or negative (it never triggers — the stored value equals the 0-based within-session trial index at every one of the 12,216 trial-start frames).

ii.
```python
            trialnum_all = b['trial number/data'][:]
...
                trnum = float(trialnum_all[a])
                if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
```

iii. The agent's value survey found "trial number is framewise with -1 outside" trials, i.e. valid inside a lap and sentinel-valued between laps, which is why it samples at the trial-start frame and guards against negatives.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond the sampling and the negative/NaN guard: the scalar is broadcast as a `float32` constant over the trial's timepoints. Values run 0…(n_trials − 1) within each session (0–99 over the dataset); no normalisation or cross-session offsetting.

ii.
```python
                trnum = float(trialnum_all[a])
                if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
                ...
                    np.full(ntime,trnum,np.float32),
```

iii. Same as 5-a; the agent treated it as a straightforward per-trial scalar to "broadcast per-trial variables to framewise matrices".

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the sparse `Reward` event series' `timestamps`, compared against the behaviour clock `position/timestamps`. Reward *amounts* are not used, nor is the `autoreward` channel, nor the `reward_zone` channel.

ii.
```python
            # Sparse reward-delivery timestamps (seconds on same clock).
            reward_times = b['Reward/timestamps'][:]
...
            outcomes = []
            # Determine outcomes first so previous outcome can be assigned.
            for a,e in zip(starts,ends):
                outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
```

iii. The agent distinguished reward-zone activation from actual delivery: "omission trials are unrewarded trials with no active reward-zone signal, while other unrewarded trials are lapsed (the zone activated but the animal failed to trigger reward). For the requested binary reward outcome, both should be 0 because actual sparse reward delivery is authoritative."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A per-trial outcome vector is built first, in a pre-pass over all trials of the session, by testing whether any reward timestamp falls in the half-open time window `[ts[start], ts[end])`. The input for trial *i* is then `outcomes[i-1]`, and 0 for the first trial of a session. The value is broadcast as a constant across the trial.

ii.
```python
            for a,e in zip(starts,ends):
                outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
                prev = float(outcomes[ti-1] if ti else 0)
                ...
                    np.full(ntime,prev,np.float32)))
```

iii. The pre-pass is explained inline: "Determine outcomes first so previous outcome can be assigned." Treating the first trial as 0 (omitted) follows the instruction's binary coding; the agent gives no further comment.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behaviour time series (cm along the corridor) combined with the active reward zone's fixed physical coordinates, looked up from the scene-derived A/B/C label: A = [80, 130], B = [200, 250], C = [320, 370] cm. The `reward_zone` behaviour channel is not used.

ii.
```python
ZONE_COORDS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
...
                zone = z1 if after else z0
                lo,hi = ZONE_COORDS[zone]
                pos = np.asarray(pos_all[sl], dtype=np.float32)
```

iii. The agent extracted the coordinates from the paper repository: "Reward-zone coordinates are now known: task labels A/B/C map to physical zones X=[80,130], Y=[200,250], Z=[320,370] cm." It rejected the raw channel: "Reward-zone raw values are lick-like sensor/event amplitudes and should not be used as labels", and "The framewise `reward_zone` stream is only zone-entry signaling and cannot itself identify A/B/C."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. A signed distance to the *nearest point of the interval*: `pos − lo` when the animal is before the zone (negative), `pos − hi` when past it (positive), and exactly 0.0 anywhere inside `[lo, hi]`.

ii.
```python
def distance_class(pos, lo, hi):
    d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))
```

iii. The docstring: "Distance is signed to the nearest point in the active zone: negative before it, zero within it, positive after it", which the agent restated in the trajectory as "signed distance is negative before zone start, zero inside, positive after zone end."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks into an `int8` array: `d < −50` → 0; `−50 ≤ d < −10` → 1; `−10 ≤ d < 0` → 2; `d == 0` → 3; `0 < d ≤ 10` → 4; `10 < d ≤ 50` → 5; `d > 50` → 6. The exact-zero class 3 corresponds to being inside the reward zone. Note the boundary convention at ±10 and ±50 is closed-on-the-positive-side (d = 10 → class 4, d = 50 → class 5), the mirror of how the negative side is handled.

ii.
```python
def distance_class(pos, lo, hi):
    d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))
    y = np.empty(d.shape, np.int8)
    y[d < -50] = 0
    y[(d >= -50) & (d < -10)] = 1
    y[(d >= -10) & (d < 0)] = 2
    y[d == 0] = 3
    y[(d > 0) & (d <= 10)] = 4
    y[(d > 10) & (d <= 50)] = 5
    y[d > 50] = 6
    return y
```
```python
      'output_values': [
        ['< -50 cm','-50 to -10 cm','-10 to <0 cm','in reward zone','>0 to +10 cm','+10 to +50 cm','> +50 cm'],
```

iii. The bin edges are taken directly from the Decoder Task specification in the instructions; the agent's plan step says it will "discretize outputs at specified boundaries". No further comment.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. `position` is sliced with the same `[a, e)` frame indices as the neural matrix, so alignment is automatic and the time axis lengths match exactly.

ii.
```python
                sl = slice(int(a),int(e))
                pos = np.asarray(pos_all[sl], dtype=np.float32)
                y = np.vstack((
                    distance_class(pos,lo,hi),
                    ...
```

iii. "all streams are already imaging-frame aligned" (trajectory, step 20).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Directly from the `position` behaviour time series (cm), the same array used for distance-to-reward-zone.

ii.
```python
            pos_all = b['position/data'][:]
...
                pos = np.asarray(pos_all[sl], dtype=np.float32)
```

iii. The agent's value survey noted that "position uses -500 outside valid trials", i.e. it is valid within `[trial_start, teleport)` and sentinel-valued elsewhere — which is consistent with slicing only complete laps.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice and the cast to `float32`; the raw cm value is discretised directly.

ii.
```python
                    np.digitize(pos,[90,180,270,360],right=False).astype(np.int8),
```

iii. Not discussed; implied by the instruction to bin the 450 cm track into 5 equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with interior edges `[90, 180, 270, 360]` and `right=False`, giving 5 classes of 90 cm each: <90 → 0, [90,180) → 1, [180,270) → 2, [270,360) → 3, ≥360 → 4. The outer bins are open, so the handful of samples marginally below 0 cm or above 450 cm fall into classes 0 and 4 rather than forming extra classes.

ii.
```python
                    np.digitize(pos,[90,180,270,360],right=False).astype(np.int8),
```
```python
        ['<90 cm','90-180 cm','180-270 cm','270-360 cm','>360 cm'],
```

iii. Taken from the Decoder Task specification ("Discretized into 5 equal-sized bins spanning the 450 cm track"); the agent gives no separate rationale.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame slice `[a, e)` as the neural matrix — no separate alignment step.

ii.
```python
                sl = slice(int(a),int(e))
                pos = np.asarray(pos_all[sl], dtype=np.float32)
```

iii. "all streams are already imaging-frame aligned" (trajectory, step 20).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. From the `lick` behaviour time series, which stores a lick *count* per imaging frame (observed range 0–6).

ii.
```python
            lick_all = b['lick/data'][:]
...
                    (np.asarray(lick_all[sl])>0).astype(np.int8),
```

iii. "Lick is a count per imaging frame (0–6), so output should be binarized as >0" (trajectory, step 13).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation: any frame with a count greater than zero becomes 1, all others 0, stored as `int8`. No smoothing, debouncing or onset detection.

ii.
```python
                    (np.asarray(lick_all[sl])>0).astype(np.int8),
```
```python
        ['no','yes'],
```

iii. The docstring: "Licks are frame counts and are converted to presence/absence", matching the instruction's binary 0/1 lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame slice `[a, e)` as the neural matrix.

ii.
```python
                sl = slice(int(a),int(e))
                ...
                    (np.asarray(lick_all[sl])>0).astype(np.int8),
```

iii. "all streams are already imaging-frame aligned" (trajectory, step 20).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the scene string in the NWB root `identifier` (see 4-a): the zone letters are recovered from the scene, and for `_to_` sessions the second letter takes over from trial 30 onward. It is not derived from the `reward_zone` behaviour channel.

ii.
```python
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}
...
        zs = re.findall(r'(?:Location)?([ABC])', scene)
        ...
        z0, z1 = zs[0], (zs[-1] if '_to_' in scene else zs[0])
...
                switched = '_to_' in scene
                after = switched and ti >= 30
                zone = z1 if after else z0
```

iii. "Paper code defines fixed reward-zone coordinates from session scene and a scene-dependent label, with within-session switches at `change_reward_trial`. The framewise `reward_zone` stream is only zone-entry signaling and cannot itself identify A/B/C. Therefore scene metadata is essential."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The letter is mapped to an integer through `ZONE_INDEX` (A → 0, B → 1, C → 2) and broadcast as an `int8` constant across all timepoints of the trial, so the per-trial variable is stored as a time-varying row.

ii.
```python
                    np.full(ntime,ZONE_INDEX[zone],np.int8),
```
```python
        ['A','B','C'],
```

iii. Per the plan: "apply the paper's default switch trial 30 (and parse cross-environment scene variants)" and "broadcast per-trial variables to framewise matrices."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the sparse `Reward` event `timestamps` only (the same `outcomes` vector used for the previous-trial input). Reward magnitude, `autoreward`, and `reward_zone` activation are all ignored.

ii.
```python
            reward_times = b['Reward/timestamps'][:]
...
            for a,e in zip(starts,ends):
                outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
```

iii. The docstring: "Reward outcome is actual reward delivery within the trial." The trajectory adds the distinction between omission and lapsed trials, concluding both should be 0 "because actual sparse reward delivery is authoritative."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, test whether any reward timestamp lies in the half-open interval `[ts[trial_start], ts[teleport])` on the behaviour clock; the resulting 0/1 is broadcast as an `int8` constant over the trial's timepoints. No nearest-frame index mapping or tolerance check is performed — the comparison is done directly in seconds.

ii.
```python
                outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
                    np.full(ntime,outcomes[ti],np.int8)))
```
```python
        ['no reward','rewarded']],
```

iii. As in 11-a. The agent notes that the reward series is "Sparse reward-delivery timestamps (seconds on same clock)", which is why it compares in the time domain rather than by frame index.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code is mostly *fail-fast* rather than tolerant; the only irregularities it accommodates are ones the agent first quantified across the whole dataset:
   - **Unequal start/teleport counts** → hard error (`raise ValueError`).
   - **Trials with `end <= start`** → silently dropped by the `valid` mask (never triggers here).
   - **Neural/behaviour frame mismatch** → a trailing surplus of exactly one neural frame is allowed and ignored (the 10 affected dual-plane sessions); any shortfall, or a surplus of ≥2 frames, is a hard error.
   - **Multi-plane ROI bookkeeping** → the concatenated `iscell` length must equal the sum of per-plane ROI widths, else hard error.
   - **Bad trial number** → non-finite or negative stored `trial number` falls back to the loop index.
   - **Degenerate sessions** → sessions with <2 trials or 0 curated cells are skipped.
   - **Out-of-range position/speed** → absorbed by the open outer bins of `np.digitize`; small negative speeds are explicitly noted as belonging in the `<2 cm/s` class.
   - **Atomic write** → the pickle is written to `converted_data.pkl.tmp` and `os.replace`d, so a crash cannot leave a truncated output.

ii.
```python
            if len(starts) != len(ends):
                raise ValueError(f'Unmatched boundaries in {f}')
            valid = (ends > starts)
            starts, ends = starts[valid], ends[valid]
...
            if any(d.shape[0] < len(ts) or d.shape[0]-len(ts) > 1 for d in event_sets):
                raise ValueError(f'Behavior/neural frame mismatch in {f}')
...
            if sum(widths) != len(iscell):
                raise ValueError(f'Plane ROI widths do not match segmentation in {f}')
...
                # Small negative values are numerical differentiation noise and
                # properly fall in the specified <2 cm/s category.
...
                trnum = float(trialnum_all[a])
                if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
...
            if len(ns) < 2 or n_cells==0:
                continue
...
    with open(OUT+'.tmp','wb') as fp: pickle.dump(data,fp,protocol=4)
    os.replace(OUT+'.tmp',OUT)
```

iii. Each of these guards was added reactively after a crash, and only after the agent measured the scope of the problem. On the dual-plane failure: "conversion failed at the first m17 session because the selected `iscell` indices exceed the number of columns in `Deconvolved/plane0` … The converter must concatenate cells from all planes while preserving frame alignment." On the frame surplus: "Only 10 sessions mismatch, and in every case both neural planes have exactly one extra frame while all 80 complete behavior trials fit inside the neural arrays. This is a known edge-type mismatch rather than truncation", and it deliberately kept the check tight — "It permits only the empirically observed one-sample terminal neural surplus and still rejects genuine truncation/misalignment."

## 13-a. What are the most time-consuming steps of the code?

i. Three steps dominate:
   1. **Reading the ophys arrays off disk.** The `Deconvolved` datasets are contiguous, uncompressed `float32` of shape (n_frames, n_ROIs) and total tens of GB across the 152 files; because the per-trial read selects a scattered subset of columns, the underlying I/O effectively touches whole rows.
   2. **The per-trial HDF5 read itself.** `d[sl, ids]` is a fancy-indexed dataset read issued once per trial *per plane* — about 12,500 separate HDF5 selection operations, each with selection-construction and buffer-allocation overhead, instead of one read per session.
   3. **Serialising the 9.7 GB pickle** at the end, which also requires the entire dataset to be resident in RAM first.

   The behavioural processing (`distance_class`, two `np.digitize` calls, the lick threshold, the `np.full` broadcasts) is negligible by comparison.

ii.
```python
                n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
```
```python
    with open(OUT+'.tmp','wb') as fp: pickle.dump(data,fp,protocol=4)
    os.replace(OUT+'.tmp',OUT)
```

iii. The agent anticipated the I/O and size costs up front — "a large (87 GB) DANDI/BIDS-style hierarchy", "conversion must read only processed ROI/time-series datasets, not image stacks", "Keeping every classified cell would produce ~13.3 GB of neural float32 data before pickle overhead" — and structured the run around them: it used `h5py` instead of `pynwb` for speed, kept `event_sets` as lazy HDF5 dataset handles rather than materialising whole arrays, chose `protocol=4`, and explicitly planned to "poll rather than over-wait" through the multi-minute conversion and "potentially lengthy ~13 GB pickle serialization". It did not, however, discuss the cost of the per-trial fancy-indexed read.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops are candidates:
   - **The outcome pre-pass** (`for a,e in zip(starts,ends)`) rescans the full `reward_times` array once per trial. `np.searchsorted(reward_times, ts[starts])` vs `ts[ends]` would give all per-trial outcomes in two vectorised calls.
   - **The main per-trial loop**: `distance_class` and both `np.digitize` calls are pure element-wise functions of session-long arrays. They could be computed once per session over the whole recording (the zone only changes at the switch trial, so two whole-session passes would suffice) and then sliced per trial, instead of being re-invoked 12,216 times on short segments.
   - **The per-trial, per-plane neural read**: reading `d[:, ids]` once per plane per session into memory and then slicing views per trial replaces ~12,500 HDF5 selection calls with 180.

   That said, the loop is intrinsically ragged (variable-length trials) and the per-call arrays are short, so the remaining Python overhead is real but modest relative to I/O.

ii.
```python
            outcomes = []
            # Determine outcomes first so previous outcome can be assigned.
            for a,e in zip(starts,ends):
                outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
            for ti,(a,e) in enumerate(zip(starts,ends)):
                ...
                y = np.vstack((
                    distance_class(pos,lo,hi),
                    np.digitize(pos,[90,180,270,360],right=False).astype(np.int8),
                    np.digitize(speed,[2,10,20,40],right=False).astype(np.int8),
```

iii. The agent does not discuss vectorisation anywhere in the trajectory. Its efficiency reasoning is confined to file-level I/O (dropping `pynwb`, avoiding image stacks, not materialising the full neural arrays) and output size.

## 13-c. What processing does the code repeat multiple times?

i. Comparatively little — the design is a single pass, with each NWB opened exactly once and each session's behaviour arrays read exactly once. The repetition that does exist is:
   - `reward_times` is scanned once per trial in the outcome pre-pass (see 13-b).
   - The reward-outcome vector is computed once but consumed twice (as the trial's own `reward outcome` output and as the next trial's `previous trial outcome` input) — this is a deliberate, correct reuse rather than duplicated work.
   - `parse_scene`, the `'_to_' in scene` test and the `ZONE_COORDS` lookup are re-evaluated per trial although the result changes only at the switch trial.
   - Per-trial `np.asarray(...)` calls re-copy already-in-memory slices of `pos_all`, `speed_all`, `lick_all`.

   Notably, the code does **not** repeat the expensive whole-dataset pass (unlike a survey-then-convert design), because it derives its labels from the scene string rather than from a data-driven pre-survey.

ii.
```python
            outcomes = []
            for a,e in zip(starts,ends):
                outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
...
                after = switched and ti >= 30
                env = e1 if after else e0
                zone = z1 if after else z0
                lo,hi = ZONE_COORDS[zone]
                pos = np.asarray(pos_all[sl], dtype=np.float32)
                speed = np.asarray(speed_all[sl], dtype=np.float32)
```

iii. The single-pass structure was a conscious response to scale: after seeing that repeated whole-dataset scans were slow ("The PyNWB scan remains active and has not produced output; opening 152 large NWBs through PyNWB is unnecessarily slow. It should be interrupted"), the agent moved straight to "implement conversion efficiently, likely using references/slices and protocol 4 pickle". The `outcomes` pre-pass is justified inline as necessary ordering, not as an optimisation: "Determine outcomes first so previous outcome can be assigned."

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Very little is computed and thrown away; the main items are:
   - **`e1` / `z1` are computed for every session** including the 75 fixed-zone sessions, where `switched` is `False` and they are never read.
   - **`session_info`** accumulates a per-session dict (file path, session id, scene, counts, switch trial) that the decoder never reads; it is diagnostic metadata only. It is tiny, and the instructions invite extra metadata fields.
   - **The per-trial progress `print`** (one formatted line per session) and the `text()` decoding of `session_id`.
   - **The `trnum` non-finite/negative fallback**, which never fires.
   - Arguably the largest waste is not computation but storage: the neural data is written as `float32` for all 138,678 `iscell` ROIs with no interneuron exclusion, producing a 9.7 GB pickle that must be fully loaded by the decoder.

   Conversely, nothing that the decoder needs is computed and discarded, and the code avoids the obvious trap of materialising the full (n_frames × n_ROIs) neural arrays.

ii.
```python
            session_info.append({'file':os.path.relpath(f,DATA_ROOT),'subject':subject,
                'session_id':session_id,'scene':scene,'n_trials':len(ns),
                'n_cells':n_cells,'switch_trial':30 if switched else None})
        print(f'[{si+1:3d}/{len(files)}] {subject} ses-{session_id}: {scene}, {len(ns)} trials, {n_cells} cells',flush=True)
```
```python
                trnum = float(trialnum_all[a])
                if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
```

iii. The agent does not raise this question directly. Its only related comment is on output size: "Keeping every classified cell would produce ~13.3 GB of neural float32 data before pickle overhead, suggesting careful confirmation of paper session/cell curation is important", followed by "Disk and RAM are ample. I will retain Suite2p `iscell` ROIs". The per-session print was added deliberately so the long run could be monitored: "I will poll rather than over-wait."
