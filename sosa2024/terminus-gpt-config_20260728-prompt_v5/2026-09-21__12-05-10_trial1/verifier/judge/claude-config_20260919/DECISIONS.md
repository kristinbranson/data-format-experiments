# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `*.nwb` file under `/app/data` is found recursively with `Path.rglob` and sorted; each file is one session. Files are read with `pynwb.NWBHDF5IO(..., load_namespaces=True)` inside a `with` block, and everything needed is pulled eagerly into memory before the file is closed. From each file the AI reads:
- the subject id from `nwb.subject.subject_id` (falling back to the parent directory name),
- the session id from `nwb.session_id`,
- **only** `nwb.processing['ophys']['Deconvolved'].roi_response_series['plane0']` as neural data — `plane1` is never read, so on the 28 two-plane sessions (all of `m17` and `m18`) roughly half of each session's ROIs are silently discarded (52,019 of 312,110 ROIs archive-wide),
- ten named behavior time series from `processing['behavior']['BehavioralTimeSeries']`, each passed through `align_series_to_frame` so that it comes back on the imaging-frame grid,
- `frame_timestamps` taken from the `trial_start` time series' own `timestamps`.

Trials are then reconstructed inside `process_session` from the behavior streams (there is no `nwb.trials` table). All 152 sessions, 11 subjects and 12,216 trials end up in the output. `--sample` takes `files[:2]`, which are two sessions of the same mouse (m11).

There is no check that the neural array and the behavior timestamps have the same length. Ten sessions (m18 ses-05…14) have one more imaging frame than behavior sample; the code happens to be safe because the neural stream is the longer one there, but the reverse case would raise an `IndexError`.

ii.
```python
def list_sessions(data_root: Path):
    return sorted(data_root.rglob('*.nwb'))

def read_session(nwb_path: Path):
    with NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        subj = nwb.subject.subject_id if nwb.subject is not None else nwb_path.parent.name.replace('sub-', '')
        sess_id = getattr(nwb, 'session_id', nwb_path.stem)

        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries']
        ophys = nwb.processing['ophys']
        deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
        roi_count = deconv.shape[1]

        frame_timestamps = np.asarray(beh.time_series['trial_start'].timestamps[:], dtype=np.float64)
        ts_names = ['Reward', 'environment', 'lick', 'position', 'reward_zone', 'scanning', 'speed', 'teleport', 'trial number', 'trial_start']
        beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
        return {...}
```

iii. From CONVERSION_NOTES.md Step 5: *"Use `Deconvolved` ROIResponseSeries as neural input: This is the most directly usable processed neural activity stream in NWB"*; *"Segment trials from behavior streams rather than NWB trial tables: `nwb.trials` is absent, but `trial_start` and `trial number` are present for every frame"*; *"Use imaging-frame timestamps as the common time base"*; *"Retain full archive sessions initially: Archive has 152 sessions; later consistency checks can determine whether filtering to a paper-like subset is necessary."* The choice of `plane0` is never justified — the trajectory (step 22) records only that *"`Deconvolved/plane0` has shape (19818, 349)"* for the single-plane session the AI inspected, and the possibility of a second plane is never revisited.

## 1-b. How are the data split into subjects?

i. The subject of a session is `nwb.subject.subject_id` read from the NWB file (with the `sub-<id>` directory name as a fallback). Subjects are accumulated in first-encountered order into `data['subjects']`, and `data['subject_idx']` stores the index of the owning subject for each kept session. Because the file list is sorted by path, the order is alphabetical by directory: `['m11','m12','m13','m14','m15','m17','m18','m19','m3','m4','m7']` — 11 subjects, which matches the reference.

ii.
```python
subj = nwb.subject.subject_id if nwb.subject is not None else nwb_path.parent.name.replace('sub-', '')
...
if sess['subject'] not in subject_to_idx:
    subject_to_idx[sess['subject']] = len(subjects)
    subjects.append(sess['subject'])
...
subject_idx.append(subject_to_idx[sess['subject']])
```

iii. CONVERSION_NOTES.md Step 2 records *"11 subject folders and 152 session NWB files total"*; the subject id is taken from the file's own metadata rather than parsed from the path, which the AI treated as the authoritative source.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are processed independently and appended to `neural`/`input`/`output` in sorted-path order. A session is kept only if it yields at least two trials (the decoder's requirement); in practice all 152 sessions are kept. No session-level quality filtering (e.g. to the paper's analysis subsets) is applied.

ii.
```python
files = list_sessions(data_root)
for j, f in enumerate(files):
    sess = read_session(f)
    ntr, itr, otr, bri = process_session(sess, make_plot=show_processing and j < 2)
    if len(ntr) >= 2:
        neural_all.append(ntr)
        input_all.append(itr)
        output_all.append(otr)
        subject_idx.append(subject_to_idx[sess['subject']])
        brain_region_idx.append(bri)
        session_info.append({'path': sess['path'], 'session_id': sess['session_id'], ...})
```

iii. CONVERSION_NOTES.md Step 4: *"Treat each NWB file as a session and reconstruct session-level trial data from NWB contents."* Step 5 decision 5: *"Retain full archive sessions initially: Archive has 152 sessions; later consistency checks can determine whether filtering to a paper-like subset is necessary."* The AI noted the paper's *"50 out of 77 sessions"* remapping subset but concluded those are analysis-specific inclusion criteria, not a definition of the decoder dataset.

## 1-d. How are the data split into trials?

i. Trial starts are the frames where the `trial_start` pulse is positive **and** the frame is "valid". A frame is valid when `trial number >= 0` and `scanning > 0`. Each trial then runs from its start to the **next trial start** (the last trial runs to the end of the session), and within that window only the valid frames are kept:

```
trial_i = [ trial_start_i , trial_start_{i+1} )   ∩  {trial number >= 0 and scanning > 0}
```

The `teleport` stream is loaded but never used to end a trial. Because `trial number` stays `>= 0` and `scanning` stays `1` through the inter-trial interval, the "valid" mask does not remove it: each converted trial therefore contains the lap **plus the following teleport / inter-trial period**, during which the animal is held off the track at position ≈ −50 cm.

The consequences are measurable. For `sub-m11_ses-03` the trial-start→teleport laps are 137–325 frames (mean 178.5) while the AI's trials are 157–392 frames (mean 245.8). Across the whole dataset the AI's mean trial length is 297.8 frames against 216.8 for the reference, and the longest "trial" lasts 653 s (10,132 frames). The extra samples are all off-track and they dominate several output distributions (see 7-b, 8-b).

ii.
```python
valid = np.isfinite(t) & (b['trial number'] >= 0)
if 'scanning' in b:
    valid &= (b['scanning'] > 0)

trial_start_idx = np.flatnonzero((b['trial_start'] > 0) & valid)
...
segs = contiguous_segments(trial_start_idx, len(t))
```
```python
def contiguous_segments(start_idxs, n_time):
    starts = list(start_idxs)
    segs = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n_time
        if e > s:
            segs.append((s, e))
    return segs
```
```python
for i, (s, e) in enumerate(kept_segs):
    mask = valid[s:e]
    idx = np.flatnonzero(mask) + s
```

iii. Trajectory step 26: *"`trial_start` is a binary pulse with 80 positive events in the inspected session, and `trial number` is constant within each trial and increments from 0 to 80, with −1 during off-track periods. Position ranges roughly from −500 to 450 cm, indicating off-track or teleport periods exist and must be handled … The script should probably use only samples where `scanning` is on and trial number >= 0."* The AI therefore believed the `trial number >= 0` mask removed the off-track period; it never verified that belief against the `teleport` stream, and CONVERSION_NOTES.md Step 5 simply records *"Trials are laps on the track."*

## 1-e. How are trials filtered based on quality controls?

i. Three filters, all applied in `process_session`:
1. a segment is dropped if it has fewer than 5 valid frames (`mask.sum() < 5`);
2. a segment is dropped if fewer than 5 of its valid frames have an in-track position (`0 <= position <= 450`);
3. the whole session is dropped if fewer than 2 trials survive (`len(kept_segs) < 2`, and again `len(ntr) >= 2` in `convert`).

No trial is rejected for missing reward-zone information, missing reward, or abnormal length. In practice nothing is removed by the length rule (the shortest kept trial is 104 frames); 12,216 trials are kept versus 12,210 for the reference.

ii.
```python
for (s, e), trn in zip(segs, trial_nums):
    mask = valid[s:e]
    if mask.sum() < 5:
        continue
    pos = np.asarray(b['position'][s:e])[mask]
    if np.sum((pos >= 0) & (pos <= 450)) < 5:
        continue
    ...
    kept_segs.append((s, e))

if len(kept_segs) < 2:
    return [], [], [], np.zeros(sess['roi_count'], dtype=np.int64)
```

iii. Not discussed explicitly in CONVERSION_NOTES.md. The thresholds appear in the first draft of the script (trajectory step 26) as defensive minimums, motivated by the observation that position can run to −500 cm off-track and that the decoder requires *"at least two trials within each session"*.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is the NWB `ophys/Deconvolved` ROIResponseSeries, `plane0` only, transposed from (time × ROI) to (ROI × time) per trial and cast to `float32`. Neither `Fluorescence` (F) nor `Neuropil` (Fneu) is read, so the paper's own dF/F + OASIS pipeline (`src/reward_relative/preprocessing.py::dff`, which the paper stores as `sess.timeseries['events']`) is never reproduced — the stored suite2p deconvolution is used instead. `plane1` of the 28 two-plane sessions is not read at all.

ii.
```python
deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
roi_count = deconv.shape[1]
...
nn = neural[idx, :].T.astype(np.float32)
...
neural_trials.append(nn)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 1: *"Use `Deconvolved` ROIResponseSeries as neural input: This is the most directly usable processed neural activity stream in NWB and is closer to event-like activity than raw fluorescence."* The README repeats *"Neural signal: `ophys/Deconvolved/plane0`"*. The reference code's `dff`, `nansmooth` and OASIS deconvolution are never mentioned anywhere in the notes or trajectory, even though the AI printed the first 260 lines of every `src/reward_relative/*.py` file in Step 1.

## 2-b. How is the `neural` data processed?

i. No processing at all. The deconvolved array is read, cast to `float32`, indexed by the trial's valid frames and transposed. There is no neuropil subtraction (`F - 0.7*Fneu`), no maximin baseline over a 20 s window, no dF/F normalisation, no 2-sample Gaussian smoothing, no OASIS deconvolution at `tau = 0.7`, and no per-plane rate correction — i.e. none of the steps the Methods describe.

ii.
```python
nn = neural[idx, :].T.astype(np.float32)
...
neural_trials.append(nn)
```

iii. Implied by the Step 5 decision that `Deconvolved` is *"the most directly usable processed neural activity stream"*, i.e. the AI treated the stored array as already-processed and did nothing further. No justification is offered for skipping the paper's processing, because the AI never identified that the paper has its own.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no neural quality control whatsoever. Every ROI in `plane0` is kept:
- the suite2p `iscell` curation column (available on `plane_data.rois[:]`) is never read, so non-cell ROIs are included — in `sub-m11_ses-03` that is 349 ROIs kept where only 155 are `iscell`; in `sub-m12_ses-01`, 3,779 kept versus 1,780 cells;
- the paper's putative-interneuron exclusion (dF/F vs running-speed Pearson r > 0.5) is not applied;
- no ROI is dropped for NaNs, low activity, or any other criterion.

The result is 260,091 stored "neurons" (mean 1,711 per session, max 3,934) against the reference's 138,298 (mean 910, max 2,320) — and the AI's count is *still* low in the two-plane sessions because `plane1` is missing.

ii. There is no filtering code. The only related line is the one that defines the per-neuron region index over the raw ROI count:
```python
roi_count = deconv.shape[1]
...
brain_region_idx = np.zeros(sess['roi_count'], dtype=np.int64)
```

iii. None given. CONVERSION_NOTES.md Step 3 says only *"Neuron curation rules: Not fully determined from text alone; likely depends on ROI/cell definitions in the ophys processing outputs and code"*, and the question is never returned to in Steps 4, 5, 10 or 12. Step 9's consistency table records *"Total neurons … Reference Paper: unresolved from paper text … Converted Data: 260091 … data/code match"*, i.e. the count was checked only against the AI's own reading of the data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is trial start, and no offset is applied: the trial's frame indices begin at the frame carrying the `trial_start` pulse, and the same index array `idx` is used to slice the neural array and every behavior stream, so all streams share one time base. Metadata records `temporal_alignment_event = 'trial_start (entry to the linear track)'`, `off_start = 0.0`, `off_end = None`. No pre-event window is included.

ii.
```python
trial_start_idx = np.flatnonzero((b['trial_start'] > 0) & valid)
...
idx = np.flatnonzero(mask) + s
tt = t[idx] - t[idx[0]]
nn = neural[idx, :].T.astype(np.float32)
pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 3: *"Use imaging-frame timestamps as the common time base: Behavioral time series appear sampled on the same frame grid as deconvolved activity (same session length), minimizing interpolation complexity."* Step 10 check 1 reports a raw-data spot check: *"converted neural data exactly matches the raw deconvolved ROIResponseSeries for the sampled trial, trial-aligned time input matches raw timestamps."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling or interpolation of the neural data — the native imaging-frame grid is kept. `metadata['time_bin_size']` is the median over sessions of the per-session median inter-frame interval of the behavior timestamps, in ms. That value is 64.4836 ms (≈15.5 Hz) and is identical for every session, including the two-plane ones (where the series' stored `rate` is 31.0156 Hz but the per-plane sampling is half that), so it agrees with the reference's `nplanes/rate*1000`. No assertion checks that all sessions share the bin size.

ii.
```python
dts = np.diff(sess['timestamps'])
dts = dts[np.isfinite(dts) & (dts > 0)]
if len(dts):
    dt_list.append(float(np.median(dts)))
...
time_bin = float(np.median(dt_list) * 1000.0) if dt_list else np.nan
```

iii. Follows from Key Decision 3 (frame timestamps as the common time base). The AI deduced the bin size from the behavior timestamps rather than from the series `rate`, which sidesteps the two-plane rate trap without ever discussing it.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The `timestamps` attribute of the `trial_start` behavior time series, which is the common frame clock used for all streams (the reference used the identical clock, taken from the `trial number` series instead).

ii.
```python
frame_timestamps = np.asarray(beh.time_series['trial_start'].timestamps[:], dtype=np.float64)
...
t = sess['timestamps']
tt = t[idx] - t[idx[0]]
```

iii. CONVERSION_NOTES.md Step 5 maps *"`trial_start` timestamps relative to trial onset → input[0] = time from start of trial, continuous per-timepoint seconds from trial onset"*, justified by Key Decision 3 that all behavior streams sit on the same frame grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Only subtraction of the trial's first timestamp, so every trial starts at 0.0 s. Stored as `float32`. No clipping or rescaling. Note that because trials extend to the next trial start (1-d), the value runs over the inter-trial interval too: the observed range is [0, 653.3] s against [0, 216.5] s for the reference.

ii.
```python
tt = t[idx] - t[idx[0]]
inp = np.vstack([
    tt.astype(np.float32),
    ...
])
```

iii. Straightforward implementation of the decoder specification ("Time from start of trial in seconds"); no further rationale is recorded.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: `tt` is computed from the same `idx` used to slice the neural array, so input, output and neural rows are the same frames in the same order and have the same length. `align_series_to_frame` guarantees every behavior stream has been placed on the frame grid first. No cross-stream timestamp assertion is made (unlike the reference, which asserts `np.allclose` between each stream's timestamps).

ii.
```python
idx = np.flatnonzero(mask) + s
tt = t[idx] - t[idx[0]]
nn = neural[idx, :].T.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 10 check 1: *"converted neural data exactly matches the raw deconvolved ROIResponseSeries for the sampled trial, trial-aligned time input matches raw timestamps … This is strong evidence the conversion is correctly aligned for at least one trial."*

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavior time series, after frame alignment.

ii.
```python
ts_names = ['Reward', 'environment', 'lick', 'position', 'reward_zone', 'scanning', 'speed', 'teleport', 'trial number', 'trial_start']
beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}
...
env = np.asarray(b['environment'][s:e])[mask]
env_per_trial.append(np.nanmedian(env))
```

iii. CONVERSION_NOTES.md Step 5: *"`environment` → input[1] = environment type, binary per-trial or broadcast across time … Map NWB environment values to ENV1 vs ENV2 after inspecting coding."*

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the nanmedian of the valid `environment` samples is taken. Those per-trial values are then recoded **relative to the values present in that session**: the unique values are sorted and the first two are mapped to 0 and 1; anything else maps to 0. The code is broadcast constant across the trial's timepoints.

This makes the code session-relative rather than absolute. In the raw data a session's in-lap `environment` value is a global label: 0 or 1 (in the 40 sessions I checked, 19 are entirely 0, 18 are entirely 1, and 3 — the day-8 "switch" sessions — contain both). A session that is entirely ENV2 has the single unique value `1.0`, which the mapping sends to `0` — the same code as an ENV1 session. The verification log confirms this: the per-session `environment_type` range is `[0,0]` for almost every session and `[0,1]` only for the handful of mixed sessions, so ENV1 and ENV2 are not distinguishable in the converted data. The reference instead writes the raw value through unchanged.

ii.
```python
def map_binary_environment(env_trial_vals):
    vals = env_trial_vals[~np.isnan(env_trial_vals)]
    uniq = np.unique(vals)
    if len(uniq) == 0:
        return np.zeros_like(env_trial_vals, dtype=np.int64), {}
    uniq_sorted = sorted(uniq.tolist())
    mapping = {v: i for i, v in enumerate(uniq_sorted[:2])}
    out = np.array([mapping.get(v, 0) for v in env_trial_vals], dtype=np.int64)
    return out, mapping
```
```python
env_codes, env_map = map_binary_environment(np.asarray(env_per_trial, dtype=float))
...
np.full(len(idx), env_codes[i], dtype=np.float32),
```

iii. Trajectory step 31: *"`environment` uses values `-1` and `0` in these sessions, so our binary environment mapping on the sample collapsed to a constant 0 because only one in-track value may be present after filtering."* Step 32: *"within valid trials in the sample sessions it is always 0, and −1 appears only outside valid trial periods. So the constant `environment_type` in the sample is not necessarily a bug; those two sessions likely belong to one environment."* The AI diagnosed the symptom, accepted the constant as benign, and never checked whether a session recorded entirely in the other environment would also be coded 0.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. As implemented, from nothing in the raw data: `input[2]` is `i`, the loop index of the trial within the session's kept segments (0, 1, 2, …). The stored `trial number` stream *is* read and summarised into a per-segment array `trial_nums` (the rounded median of the non-negative `trial number` samples in each segment), but that array is only used to zip alongside the segments and is never written into the output — it is dead. The CONVERSION_NOTES.md mapping table nevertheless states the source is the stored stream. The reference also uses the loop index, explicitly rejecting the stored `trial number`, so the implemented values agree with the reference (both 0…79 for a standard session; the verification log shows `trial_number` ranging to 99 on the 100-lap sessions).

ii.
```python
trial_nums = []
for s, e in segs:
    tr = b['trial number'][s:e]
    tr_valid = tr[tr >= 0]
    if len(tr_valid) == 0:
        continue
    trial_nums.append(int(np.round(np.median(tr_valid))))
...
for (s, e), trn in zip(segs, trial_nums):   # trn never used
    ...
inp = np.vstack([
    tt.astype(np.float32),
    np.full(len(idx), env_codes[i], dtype=np.float32),
    np.full(len(idx), i, dtype=np.float32),        # <- trial number = loop index
    np.full(len(idx), prev_rew[i], dtype=np.float32),
])
```

iii. CONVERSION_NOTES.md Step 5: *"`trial number` → input[2] = trial number, continuous per-trial or broadcast across time … Use within-session trial number from behavior stream."* Key Decision 4: *"Broadcast per-trial variables across all timepoints in a trial: This keeps all decoder inputs/outputs in consistent (n_var, n_timepoints) arrays."* No note explains the switch to the loop index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the integer index across the trial's timepoints as `float32`. Numbering restarts at 0 in each session and is compacted over dropped segments (a trial rejected by the 1-e filters does not consume a number).

ii.
```python
np.full(len(idx), i, dtype=np.float32),
```

iii. Key Decision 4 (broadcast per-trial variables across all timepoints).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The `Reward` behavior time series. `Reward` is a sparse event series (e.g. 74 events in a 19,818-frame session) with its own timestamps, so `align_series_to_frame` detects `len(data) < len(frame_timestamps)` and scatters each reward amount onto the **nearest** frame (comparing the `searchsorted` insertion point with its left neighbour), leaving zeros elsewhere. The resulting dense array is what the trial logic sees.

ii.
```python
if len(data) < len(frame_timestamps):
    out = np.zeros(len(frame_timestamps), dtype=np.float64)
    idx = np.searchsorted(frame_timestamps, ts_t)
    idx = np.clip(idx, 0, len(frame_timestamps) - 1)
    left = np.maximum(idx - 1, 0)
    choose_left = np.abs(frame_timestamps[left] - ts_t) < np.abs(frame_timestamps[idx] - ts_t)
    idx[choose_left] = left[choose_left]
    out[idx] = data
    return out
```
```python
rw = np.asarray(b['Reward'][s:e])[mask]
...
rew_per_trial.append(int(np.any(rw > 0)))
```

iii. CONVERSION_NOTES.md Step 10: *"Sparse `Reward` event stream not frame-aligned: fixed by aligning sparse event timestamps onto frame timestamps before trial segmentation."* Trajectory step 31: *"`Reward` is a sparse event series of length 74/62, confirming our alignment fix was necessary."*

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A per-trial rewarded flag is built first (`1` if any aligned reward amount is positive inside the trial window, else `0`), then shifted by one over the list of kept trials, with `0` prepended for the first trial of each session. The value is broadcast across the trial's timepoints. Because the trial window extends to the next trial start (1-d), a reward delivered during the teleport is attributed to the trial that just ended.

ii.
```python
rew_per_trial.append(int(np.any(rw > 0)))
...
prev_rew = np.array([0] + rew_per_trial[:-1], dtype=np.int64)
...
np.full(len(idx), prev_rew[i], dtype=np.float32),
```

iii. Follows the decoder specification ("Previous trial outcome, binary, omitted = 0, rewarded = 1, per trial"); CONVERSION_NOTES.md Step 5 maps *"previous trial `Reward` outcome → input[3] … Derive from prior segmented trial reward delivery/omission."*

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From `position` and `reward_zone`. For each trial the AI infers a reward-zone interval `[lo, hi]` as the min and max **position at the frames where `reward_zone > 0`** (restricted to in-track positions), and then measures signed distance to that interval.

The problem is what `reward_zone > 0` actually marks. It is not zone occupancy over the whole 50 cm zone: in the sessions I inspected it fires on only 1–6 frames per trial, clustered at the animal's entry into the zone (e.g. trial 0 of `sub-m11_ses-03`: 5 frames spanning positions 203.3–210.3 cm; trial 3: a single frame at 199.6 cm). So the inferred `[lo, hi]` is a ~1–10 cm sliver at the zone entrance rather than the paper's 50 cm zone (`A [80,130]`, `B [200,250]`, `C [320,370]`, from `src/reward_relative/behavior.py`). Every position past the entrance therefore scores as *after* the zone.

When no frame in the trial has `reward_zone > 0` — 6 to 17 trials per session in my spot checks, ≈13% overall — the code falls back to a window of ±10 cm around the trial's **median position**, which has nothing to do with the reward zone.

The downstream distortion is large: the exactly-0 ("in zone") class holds 5.2% of samples in the AI's data versus 23.7% in the reference, and the two extreme classes hold 40.9% and 26.2% versus 25.3% and 24.3%.

ii.
```python
def infer_zone_interval_and_location(rz_vals, pos_vals):
    m = (rz_vals > 0) & np.isfinite(pos_vals) & (pos_vals >= 0) & (pos_vals <= 450)
    if np.any(m):
        lo = float(np.min(pos_vals[m]))
        hi = float(np.max(pos_vals[m]))
    else:
        c = float(np.nanmedian(pos_vals[np.isfinite(pos_vals)]))
        lo, hi = c - 10.0, c + 10.0
    center = 0.5 * (lo + hi)
    ...
    return lo, hi, loc
```
```python
zlo, zhi, zloc = infer_zone_interval_and_location(rz, pos)
rz_interval_per_trial.append((zlo, zhi))
```

iii. Trajectory step 34: *"raw `reward_zone` is sparse and only nonzero near positions around ~200–225 cm in these sessions … it is more like a sparse zone-local annotation around the reward location. Using it directly as `distance_to_reward_zone` is incorrect … we should instead compute signed distance from position to the reward-zone *interval*. We can infer the interval from positions where `reward_zone > 0` within each session/trial."* Step 35: *"The interval-based reward-zone distance fix substantially improved the sample output distribution: all 7 distance classes are now present with a plausible bimodal distribution."* The AI validated the fix on the shape of the histogram, not against the zone coordinates in the reference code or paper, which it never looked up.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The trial's positions are first clipped to `[0, 450]`, then the signed distance to the inferred interval is taken: `pos - lo` where `pos < lo` (negative, before the zone), `pos - hi` where `pos > hi` (positive, past the zone), and exactly `0` inside. The result is discretised (see 7-c).

The clip matters here because of the trial definition (1-d): the inter-trial samples sit at position ≈ −50 cm and are clipped to 0, which then scores as roughly −80 cm from a zone-A interval and −200 cm from a zone-B one, i.e. they pile into the "< −50 cm" class. That class holds 40.9% of all samples.

ii.
```python
pos_clip = np.clip(pos, 0, 450)
zlo, zhi = rz_interval_per_trial[i]
dist = signed_distance_to_interval(pos_clip, zlo, zhi)
```
```python
def signed_distance_to_interval(pos, lo, hi):
    dist = np.zeros_like(pos, dtype=np.float32)
    dist[pos < lo] = pos[pos < lo] - lo
    dist[pos > hi] = pos[pos > hi] - hi
    return dist
```

iii. Trajectory step 34: *"compute signed distance to the nearest point in that interval: negative before zone, 0 inside zone, positive after zone. This should produce all 7 bins more sensibly."* The clipping is justified in the metadata note as *"position clipped to [0,450] for discretization."*

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes by explicit boolean masks, matching the specification edge for edge: `< −50 → 0`, `[−50, −10) → 1`, `[−10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`. (The reference's `np.digitize` with edges `[-inf,-50,-10,0,1e-6,10,50,inf]` differs only in which side of exactly +10 and +50 the boundary falls on.) `output_values` names the seven classes accordingly.

ii.
```python
def bin_distance(dist):
    out = np.full(dist.shape, 6, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. Direct transcription of the Decoder Output specification. CONVERSION_NOTES.md Step 5: *"compute signed distance to nearest reward-zone location then discretize into 7 bins."* Trajectory step 30 notes that the exact-0 class must exist: *"`distance_to_reward_zone` lacks class 3 (exactly 0) … it hints that the current distance definition uses zone center rather than distance to any location in the reward zone."*

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices as the neural data — `position` is sliced with the same `idx`, so no resampling or shifting is needed. The zone interval is a per-trial scalar pair and carries no timing.

ii.
```python
idx = np.flatnonzero(mask) + s
nn = neural[idx, :].T.astype(np.float32)
pos = np.asarray(b['position'][idx], dtype=np.float32)
...
out = np.vstack([bin_distance(dist), ...])
```

iii. CONVERSION_NOTES.md Step 10 check 2: *"distance-bin match fraction 1.0"* against raw data for session 0 trial 0 — i.e. the AI re-derived its own bins from raw `position` and `reward_zone` and confirmed they agree, which tests the alignment but not the zone definition.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (cm along the VR corridor), frame-aligned and sliced by the trial's `idx`.

ii.
```python
pos = np.asarray(b['position'][idx], dtype=np.float32)
pos_clip = np.clip(pos, 0, 450)
```

iii. CONVERSION_NOTES.md Step 5: *"`position` → output[1] absolute position, discretize 0–450 cm into 5 bins, time-varying per sample."*

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Only a clip to `[0, 450]` before binning. This is effectively equivalent to the reference's open end bins (`[-inf, 90)` and `[360, inf)`), which also absorb out-of-range samples into the end classes. What differs is the population being binned: because the trials include the inter-trial interval (1-d), ~25% of the samples sit off-track at ≈ −50 cm and land in class 0. Class 0 therefore holds 42.1% of samples versus 21.1% in the reference, and the other four classes are correspondingly depleted.

ii.
```python
pos_clip = np.clip(pos, 0, 450)
...
out = np.vstack([
    bin_distance(dist),
    bin_position(pos_clip),
    ...
])
```

iii. Metadata note: *"position clipped to [0,450] for discretization."* Trajectory step 26 motivates it: *"Position ranges roughly from −500 to 450 cm, indicating off-track or teleport periods exist and must be handled … likely clip or exclude off-track position values outside the track when constructing outputs."*

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins spanning the 450 cm track, by boolean masks: `< 90 → 0`, `[90,180) → 1`, `[180,270) → 2`, `[270,360) → 3`, `>= 360 → 4`. Identical to the reference's `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]`.

ii.
```python
def bin_position(pos):
    out = np.full(pos.shape, 0, dtype=np.int64)
    out[(pos >= 90) & (pos < 180)] = 1
    out[(pos >= 180) & (pos < 270)] = 2
    out[(pos >= 270) & (pos < 360)] = 3
    out[pos >= 360] = 4
    return out
```

iii. Direct transcription of the Decoder Output specification ("Discretized into 5 equal-sized bins spanning the 450 cm track"). CONVERSION_NOTES.md Step 10 check 1 reports *"position-bin match fraction 1.0"* against bins re-derived from raw position.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as the neural data; no separate alignment step.

ii.
```python
idx = np.flatnonzero(mask) + s
nn = neural[idx, :].T.astype(np.float32)
pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. Key Decision 3 (frame timestamps as the common time base) plus the Step 10 raw-data spot check.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series, which carries per-frame lick counts (observed unique values 0–6), frame-aligned and sliced by `idx`.

ii.
```python
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 5: *"`lick` → output[3] lick, binarize to 0/1, time-varying per sample."* Trajectory step 31: *"`lick` also has values >1, meaning it may be counts per frame rather than binary, though binarization is still acceptable for the decoder output."*

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarisation at `> 0`, identical to the reference. No smoothing, no debouncing, no spatial binning. The resulting class balance is 81.7% no / 18.3% yes (reference: 77.0% / 23.0%; the difference comes from the extra non-licking inter-trial samples).

ii.
```python
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
...
out = np.vstack([..., lick, ...])
```

iii. The decoder specification requires a binary lick output ("0 = no, 1 = yes"), and the AI noted the raw values exceed 1.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as the neural data; `lick` is a dense per-frame stream so `align_series_to_frame` returns it untouched.

ii.
```python
lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
```

iii. Key Decision 3 (shared frame grid).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the same per-trial inferred interval as 7-a (`position` at frames where `reward_zone > 0`): the interval's centre is thresholded into A/B/C with fixed cut-points at 200 cm and 300 cm. When the trial has no `reward_zone > 0` frame, the centre is the trial's median position, and the label is whatever that happens to fall into.

ii.
```python
def infer_zone_interval_and_location(rz_vals, pos_vals):
    m = (rz_vals > 0) & np.isfinite(pos_vals) & (pos_vals >= 0) & (pos_vals <= 450)
    if np.any(m):
        lo = float(np.min(pos_vals[m])); hi = float(np.max(pos_vals[m]))
    else:
        c = float(np.nanmedian(pos_vals[np.isfinite(pos_vals)]))
        lo, hi = c - 10.0, c + 10.0
    center = 0.5 * (lo + hi)
    if center < 200:
        loc = 0
    elif center < 300:
        loc = 1
    else:
        loc = 2
    return lo, hi, loc
```
```python
rz_per_trial.append(zloc)
...
rz_codes = np.asarray(rz_per_trial, dtype=np.int64)
...
np.full(len(idx), rz_codes[i], dtype=np.int64),
```

iii. Trajectory step 34: *"The per-trial reward zone location A/B/C can be derived from the inferred zone center (<200 A, 200–300 B, >300 C)."* Step 33 records the earlier failed attempt (using the raw `reward_zone` codes directly as the 7 distance bins gave 98.8% in class 0) and notes the improvement: *"`reward_zone_location` now varies between A and B across the two sample sessions."*

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The threshold assignment above, done **independently for every trial** and broadcast across the trial's timepoints. There is no smoothing or block-level constraint (the reference runs a Viterbi pass with a 0.9 stay probability so that the label is stable across trials and so that trials with no zone signal inherit their neighbours' label).

The fallback branch is the visible failure mode. It fires on the ≈13% of trials with no `reward_zone > 0` sample, and those trials' median position lands below 200 cm, so they are labelled A. The converted class balance is A 0.456 / B 0.263 / C 0.282, against 0.329 / 0.337 / 0.334 for the reference — the 0.456 is almost exactly 1/3 + 0.13, which is what a systematic mislabelling of all no-signal trials as A predicts.

ii.
```python
rz_codes = np.asarray(rz_per_trial, dtype=np.int64)
rz_map = {0: 0, 1: 1, 2: 2}      # computed, never used
...
np.full(len(idx), rz_codes[i], dtype=np.int64),
```

iii. As for 10-a. CONVERSION_NOTES.md Step 12 treats the resulting decoder accuracy (0.628 validation, chance 0.333) as *"Strongly above chance and consistent with session-level context decoding"* and does not check the class balance against the three-zone design.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` behavior time series, scattered onto the nearest imaging frame by `align_series_to_frame` (see 6-a).

ii.
```python
rw = np.asarray(b['Reward'][s:e])[mask]
...
rew_per_trial.append(int(np.any(rw > 0)))
```

iii. CONVERSION_NOTES.md Step 5: *"`Reward` → output[5] reward outcome, binary per trial from any reward event within trial … Distinguish rewarded vs omitted/no reward trials."*

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. `1` if any positive reward amount falls inside the trial window, else `0`, broadcast across the trial's timepoints as a time-varying (constant) row. Identical in spirit to the reference's `np.any(isreward[idx])`. The reward-amount value itself is not used (it is a constant 0.004 mL in the raw data). The converted balance is 81.1% rewarded / 18.9% omitted, close to the reference's 84.3% / 15.7%.

ii.
```python
rew_per_trial.append(int(np.any(rw > 0)))
...
np.full(len(idx), rew_per_trial[i], dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 10 check 2: *"reward outcome exact match"* against the raw `Reward` events for session 0 trial 0.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled:
- **Streams with their own sampling** (`Reward`): `align_series_to_frame` branches on length — a sparse series is scattered onto nearest frames, an equal-length series is passed through, a longer/unsorted one is sorted and linearly interpolated, an empty one becomes zeros, and a series without timestamps is zero/NaN-padded to the frame count.
- **Off-track and non-scanning samples**: excluded from trials by `valid = (trial number >= 0) & (scanning > 0)` (though, as 1-d shows, this does not exclude the inter-trial interval it was intended to).
- **Out-of-range positions**: clipped to `[0, 450]`; a trial with fewer than 5 in-track samples is dropped.
- **Degenerate trials/sessions**: <5 valid frames → trial dropped; <2 trials → session dropped.
- **Negative speeds** (the raw stream dips to ≈ −1.4 cm/s): floored at 0 before binning.
- **Missing reward-zone annotation**: silently replaced by a ±10 cm window around the trial's median position (see 7-a/10-b) — the one case where a missing value produces a confidently wrong label rather than a flag or an abstention.

Not handled:
- **Neural / behavior length mismatch**: no check. Ten sessions (m18, ses-05…14) have exactly one more imaging frame than behavior samples; the code survives only because the neural stream is the longer one there — the reverse would raise an `IndexError` from `neural[idx, :]`. The reference crops both to the shorter length with a warning.
- **Cross-stream timestamp agreement**: never asserted (the reference asserts `np.allclose` for every behavior stream against the reference clock, and asserts the reward alignment error is within half a time bin).
- Nothing is printed or counted when a trial or session is dropped, so silent losses would not be noticed.

ii.
```python
def align_series_to_frame(ts, frame_timestamps):
    data = np.asarray(ts.data[:])
    ts_t = np.asarray(ts.timestamps[:], dtype=np.float64) if ts.timestamps is not None else None
    if data.ndim > 1:
        data = np.squeeze(data)
    if ts_t is not None and len(data) == len(frame_timestamps) and len(ts_t) == len(frame_timestamps):
        return data
    if ts_t is None:
        if len(data) == len(frame_timestamps):
            return data
        out = np.full(len(frame_timestamps), np.nan, dtype=np.float64)
        n = min(len(data), len(out))
        out[:n] = data[:n]
        return out
    if len(data) == 0:
        return np.zeros(len(frame_timestamps), dtype=np.float64)
    ...
```
```python
mask = valid[s:e]
if mask.sum() < 5:
    continue
pos = np.asarray(b['position'][s:e])[mask]
if np.sum((pos >= 0) & (pos <= 450)) < 5:
    continue
```
```python
def bin_speed(speed):
    s = np.maximum(speed, 0)
    ...
```

iii. CONVERSION_NOTES.md Step 10: *"Sparse `Reward` event stream not frame-aligned: fixed by aligning sparse event timestamps onto frame timestamps before trial segmentation."* Step 5 planned check: *"Check that each trial segment has matching lengths across deconvolved neural data and all behavioral time series"* — this check was planned but never implemented or reported.

## 13-a. What are the most time-consuming steps of the code?

i. The script prints per-session wall time and a total. The full conversion takes **98.16 s** for 152 sessions (0.20–0.98 s per session), so it is comfortably inside the 15-minute budget and roughly an order of magnitude faster than the reference, which makes two full passes over every NWB file (a `survey` pass plus the conversion pass) and additionally runs the dF/F + OASIS pipeline per plane.

The real costs, in order:
1. **NWB reads** — `deconv.data[:]` materialises the whole (T × ROI) deconvolved array, up to 3,934 ROIs × ~23,000 frames, plus ten behavior streams. This is essentially all of the per-session time and is I/O plus decompression bound.
2. **Pickling the result** — the output is 25.2 GB, written in one `pickle.dump`. This step is not timed separately (it falls inside the reported total), but it is the dominant single write and is 2.6× larger than it needs to be because no ROI curation is applied (2-c).
3. Per-trial slicing and binning — negligible by comparison.

CONVERSION_NOTES.md Step 7 estimates *"~0.5 s / session, ~1-2 min for full archive if scaling linearly"*, which the actual 98 s confirms.

ii.
```python
for j, f in enumerate(files):
    t0 = time.time()
    sess = read_session(f)
    ...
    print(f'processed {f.name}: kept_trials={len(ntr)} neurons={sess["roi_count"]} elapsed={time.time()-t0:.2f}s')
...
with open(out_path, 'wb') as f:
    pickle.dump(data, f)
print(f'total_sessions_kept={len(neural_all)} total_subjects={len(subjects)} total_time={time.time()-start:.2f}s')
```

iii. CONVERSION_NOTES.md Step 6: *"First-pass implementation loads full session arrays into memory and iterates session-by-session; may need optimization after timing sample conversion"*; Step 7 concluded no optimisation was needed. The notes never identify NWB I/O or the pickle write specifically as the bottleneck.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Candidates, none of which are material at 98 s total:
- the two per-trial Python loops in `process_session` (one to infer per-trial scalars, one to build the arrays) — vectorisable only with padding/masking given variable trial lengths, the same trade-off the reference makes;
- `map_binary_environment`'s `np.array([mapping.get(v, 0) for v in env_trial_vals])`, a Python-level comprehension over the per-trial values (short: one element per trial) that could be `np.searchsorted` on the sorted unique values;
- `contiguous_segments`, a Python loop over trial starts that is just `np.stack([starts, np.append(starts[1:], n_time)], axis=1)`;
- `align_series_to_frame` is already vectorised;
- `np.asarray(b[key][idx])` per stream per trial — the fancy-indexing copies could be hoisted out by slicing all six streams once with a single index array.

ii.
```python
out = np.array([mapping.get(v, 0) for v in env_trial_vals], dtype=np.int64)
```
```python
def contiguous_segments(start_idxs, n_time):
    starts = list(start_idxs)
    segs = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n_time
        if e > s:
            segs.append((s, e))
    return segs
```

iii. CONVERSION_NOTES.md Step 6: *"Code speedups added: Uses vectorized per-trial masking and direct NWB array extraction; avoids per-timepoint Python loops."* The AI treated the per-trial loop as acceptable and did not revisit efficiency after Step 7 confirmed the runtime estimate.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read exactly **once** — there is no survey pass, so the duplicated disk I/O of the reference implementation is avoided. What is repeated is in-memory and cheap:
- `process_session` walks the trial list twice, recomputing `mask = valid[s:e]` for every segment in both passes and re-slicing `position` (once for the ≥5-in-track-samples test and the zone inference, again for binning);
- `pos` is sliced once per pass and `pos_clip` recomputed, and the `reward_zone`/`Reward` slices of pass 1 are discarded and partly re-derived in pass 2;
- `np.asarray(...)` is called on arrays that are already ndarrays throughout, forcing extra copies on the fancy-indexed slices;
- `roi_count` is recorded per session and `np.zeros(roi_count)` is rebuilt per session even though the region is always CA1.

ii.
```python
for (s, e), trn in zip(segs, trial_nums):
    mask = valid[s:e]                       # pass 1
    ...
    pos = np.asarray(b['position'][s:e])[mask]
    ...
for i, (s, e) in enumerate(kept_segs):
    mask = valid[s:e]                       # pass 2: recomputed
    idx = np.flatnonzero(mask) + s
    pos = np.asarray(b['position'][idx], dtype=np.float32)
```

iii. Not discussed in CONVERSION_NOTES.md. The two-pass structure is required by the design: the previous-trial reward flag (`prev_rew`) and the session-relative environment mapping both need every trial's per-trial scalars before any trial's arrays can be written.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A handful of dead computations, all cheap:
- `trial_nums` — the per-segment median of the stored `trial number` is computed for every segment and then never used (the trial-number input is the loop index, 5-a);
- `env_map` and `rz_map` — returned/constructed and never read;
- `map_reward_zone` — a fully written function that is never called anywhere;
- the `teleport` and `autoreward`-adjacent streams: `teleport` is loaded and frame-aligned on every session and never used (and it is precisely the stream that would have fixed 1-d);
- `scanning` is loaded and used only in the `valid` mask, where it removes almost nothing (99.3% of frames are valid in the session I checked);
- `--show-processing` plots only trial 0 of the first two sessions and plots the binned outputs without the raw traces beside them, so the figures cannot show whether the discretisation or the alignment is right.

The costly item is not dead code but retained data: because no `iscell` curation is applied (2-c), roughly 55% of the stored neurons are non-cell ROIs, which inflates the pickle to 25.2 GB and is carried through every downstream decoder epoch.

ii.
```python
trial_nums.append(int(np.round(np.median(tr_valid))))     # never used
...
env_codes, env_map = map_binary_environment(...)           # env_map never used
rz_map = {0: 0, 1: 1, 2: 2}                                # never used

def map_reward_zone(zone_trial_vals):                      # never called
    ...
```
```python
ts_names = ['Reward', 'environment', 'lick', 'position', 'reward_zone', 'scanning', 'speed', 'teleport', 'trial number', 'trial_start']
```

iii. Not discussed in CONVERSION_NOTES.md. The dead paths are residue from the two abandoned reward-zone attempts documented in trajectory steps 32–34 (first mapping the raw `reward_zone` codes to A/B/C, then using them directly as the seven distance bins, then the interval inference that survives).
