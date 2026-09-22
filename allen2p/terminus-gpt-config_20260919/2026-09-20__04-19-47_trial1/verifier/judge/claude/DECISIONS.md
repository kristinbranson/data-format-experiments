# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK object model entirely and read the released NWB/HDF5 files directly with `h5py`. It globbed all `behavior_ophys_experiment_<id>.nwb` files in `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` (284 files), and joined them to `project_metadata/ophys_experiment_table.csv` on `ophys_experiment_id` to obtain mouse, region, depth, cre line, session type, and experience level. **No `project_code` filter was applied**, so the converted set contains both the 239 `VisualBehavior` (single-plane, ~30.95 Hz) experiments and the 45 `VisualBehaviorMultiscope` experiments (1 mouse, 8 sessions, ~11 Hz/plane) that are present on disk. Files are processed in ascending `ophys_experiment_id` order; each file is opened once and only the required datasets are read (`processing/ophys/event_detection`, `image_segmentation/cell_specimen_table`, `intervals/trials`, the natural-image presentation interval table, `processing/running/speed`, `acquisition/EyeTracking`). 281 of 284 experiments were converted (3 excluded for missing pupil arrays), yielding 84,313 trials, 41,871 neurons, 38 mice.

ii.
```python
ROOT = Path('/app/data/visual-behavior-ophys-1.1.0')
NWBDIR = ROOT / 'behavior_ophys_experiments'
META = ROOT / 'project_metadata' / 'ophys_experiment_table.csv'
...
meta=pd.read_csv(META); fmap={int(f.stem.rsplit('_',1)[1]):f for f in NWBDIR.glob('*.nwb')}
meta=meta[meta.ophys_experiment_id.isin(fmap)].sort_values('ophys_experiment_id')
if args.sample: meta=meta.head(2)
for j,row in enumerate(meta.itertuples(index=False),1):
    eid=int(row.ophys_experiment_id); print(f'[{j}/{len(meta)}] experiment {eid}',flush=True)
    result,reason=process_experiment(fmap[eid],row,args.show_processing and j<=2)
```
```python
with h5py.File(path, 'r') as h:
    evg = h['processing/ophys/event_detection']
    ots = np.asarray(evg['timestamps'][:], dtype=np.float64)
    events = np.asarray(evg['data'][:], dtype=np.float32)
    ...
    tr = h['intervals/trials']
    rg = h['processing/running/speed']
    eg = h['acquisition/EyeTracking']
    sg = stimulus_table(h)
```

iii. From CONVERSION_NOTES Step 6: "Loading complete 247 GB NWBs through high-level SDK objects would add unnecessary overhead… Direct HDF5 access reads only required datasets." Step 4 argues that the release metadata table lists 1,936 experiments while only 284 NWBs are supplied, so the supplied files are the authoritative scope: "Convert every supplied usable experiment; do not fabricate absent paper-cohort files. Report subset statistics explicitly." Step 1 documents that the HDF5 paths used correspond one-to-one to the SDK data objects (`BehaviorOphysExperiment`, `Trials`, `RunningSpeed`, `EyeTrackingTable`), and Step 10 Check 3 states "direct HDF5 paths correspond to SDK BehaviorOphysExperiment/BehaviorSession data objects."

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` strings from `ophys_experiment_table.csv`. The list is built incrementally in file-processing order (first-seen order, not sorted), and `subject_idx` records one subject index per converted session. 38 mice result (37 `VisualBehavior` + 1 `VisualBehaviorMultiscope` mouse).

ii.
```python
mouse=str(row.mouse_id); region=str(row.targeted_structure)
if mouse not in subjects: subjects.append(mouse)
...
subject_idx.append(subjects.index(mouse))
...
'subjects':subjects, 'subject_idx':np.asarray(subject_idx,dtype=np.int32),
```

iii. CONVERSION_NOTES Step 5 variable mapping: "`mouse_id` → `subjects`, `subject_idx`; String IDs and integer lookup; One subject index per experiment session." Step 2 established "38 mice" from the metadata table for the supplied subset, and Step 9's consistency table reports converted subjects = 38, matching the raw-data count.

## 1-c. How are the data split into sessions?

i. **One converted "session" = one ophys experiment = one imaging plane = one NWB file.** Imaging planes belonging to the same `ophys_session_id` are *not* merged. For the 239 single-plane `VisualBehavior` experiments this is identical to an ophys session, but the eight multiscope sessions (3–7 planes each, 45 experiments) become 45 separate sessions, so their behavioural trials are replicated once per plane. This is why the converted trial total is 84,313 (experiment-level) rather than 74,476 (unique-session-level), and why one mouse (457841) accounts for 45 of 281 sessions.

ii.
```python
for j,row in enumerate(meta.itertuples(index=False),1):
    eid=int(row.ophys_experiment_id)
    result,reason=process_experiment(fmap[eid],row,...)
    nn,ii,oo,info,plot=result
    neural.append(nn); inputs.append(ii); outputs.append(oo); infos.append(info)
```
```python
'metadata':{... 'session_unit':'one ophys experiment/imaging plane', ...}
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "Eight sessions contain 3–7 planes; plane clocks are staggered by up to 69.91 ms… Paper decoding and bootstrap unit is imaging plane → Treat each NWB experiment/plane as a target 'session'; this avoids invalid direct concatenation and matches reference analysis." Step 5 Key Decision 1 repeats: "Session unit is one experiment/imaging plane: Matches paper decoding and avoids merging multiscope planes whose timestamps are staggered." Step 4 adds that duplicating behaviour across planes "is intentional and consistent with paper decoding by plane."

## 1-d. How are the data split into trials?

i. Trials come from the native NWB trials table (`intervals/trials`). A trial is eligible if `go OR catch` is true; the window is the native `[start_time, stop_time)`, giving variable-length trials (median ~7.3 s, converted T range 210–378 bins). Within that window the code builds a uniform 30 Hz sample grid anchored to the experiment's first ophys timestamp, clipped to the recorded ophys interval.

ii.
```python
tr = h['intervals/trials']
elig = np.asarray(tr['go'][:], bool) | np.asarray(tr['catch'][:], bool)
trial_idx = np.flatnonzero(elig)
starts = np.asarray(tr['start_time'][:], float)
stops  = np.asarray(tr['stop_time'][:], float)
trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
...
for ti in trial_idx:
    a = max(starts[ti], ots[0]); b = min(stops[ti], ots[-1] + DT)
    k0 = int(np.ceil((a-origin)*HZ - 1e-9)); k1 = int(np.ceil((b-origin)*HZ - 1e-9))
    grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
    if len(grid) < 2: continue
```

iii. The instructions require "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." CONVERSION_NOTES Step 4 verified empirically that the four flags are mutually exclusive ("Every one of 148,231 unique-session rows has exactly one category"), so `go | catch` is exactly the required set. Step 5 Key Decision 11: "Variable duration: Preserve native start/stop boundaries, yielding variable time lengths but a fixed bin duration." Step 5 Key Decision 3 explains the grid anchor: "For each trial, start at the first 30 Hz grid point at or after the native trial start, relative to that experiment's first ophys timestamp; stop before native trial stop. This avoids including pre-trial samples and is explicitly tied to the ophys clock."

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: (a) `go | catch` only — aborted and auto-rewarded are dropped; (b) trials that do not overlap the ophys recording interval at all are dropped; (c) trials whose 30 Hz grid would contain fewer than 2 samples are skipped; (d) a trial with no hit/miss/false-alarm/correct-reject flag raises an error rather than being silently kept. Session-level: (e) experiments lacking `acquisition/EyeTracking/pupil_tracking/area` are excluded whole (3 of 284); (f) an assertion requires at least 2 trials per session. No further behavioural/performance filtering (e.g. d-prime) is applied — release QC is trusted. Note that passive sessions (`OPHYS_2/5_..._passive`, 82 of 284 experiments) are retained.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking/area' not in h:
    return None, 'missing pupil tracking'
...
elig = np.asarray(tr['go'][:], bool) | np.asarray(tr['catch'][:], bool)
trial_idx = np.flatnonzero(elig)
# Filter only impossible non-overlap edge cases; normal data have none.
trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
...
if len(grid) < 2: continue
...
out = next((oi for oi,name in enumerate(OUTCOMES) if bool(tr[name][ti])), None)
if out is None: raise ValueError(f'no outcome for eligible trial {ti} in {path.name}')
...
assert len(neural)==len(outputs)==len(inputs)>=2
```

iii. CONVERSION_NOTES Step 5 Key Decision 9: "Trial filtering: Include exactly `go | catch`; categories are mutually exclusive, so this already excludes aborted and auto-rewarded rows." Step 4: "Required decoder output includes pupil diameter → Exclude the three experiments because the required target cannot be constructed without unjustified imputation." Step 3 Curation: "Whitepaper release QC required peak d-prime at least 1.0"; Step 5 Key Decision 10: "no extra amplitude threshold because files already passed release QC." Step 5 Key Decision 11 notes "All supplied usable experiments have at least 39 eligible trials," which is why the ≥2-trial rule is an assertion rather than a skip.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is the **detected calcium event magnitude** trace: `processing/ophys/event_detection/data` (time × cells) with `processing/ophys/event_detection/timestamps`. dF/F (`processing/ophys/dff/traces`) is present in the files but was deliberately **not** used.

ii.
```python
evg = h['processing/ophys/event_detection']
ots = np.asarray(evg['timestamps'][:], dtype=np.float64)
events = np.asarray(evg['data'][:], dtype=np.float32)
```
```python
'metadata':{... 'neural_signal':'AllenSDK detected calcium event magnitude', ...}
```

iii. Directly grounded in the paper's methods. CONVERSION_NOTES Step 3: "The paper used detected calcium events for all neural analyses, producing per-cell events with times and magnitudes. Therefore event-detection traces, not recomputed dF/F, are the reference-consistent neural representation for this conversion." Step 4 discrepancy table: "SDK exposes dF/F and event detection | Both arrays share ophys timestamps | Paper used detected calcium events for all neural analyses | Use event-detection magnitudes." (`/app/methods.txt` line 208: "For all analysis of neural data we used the detected calcium events…"; line 179: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f.")

## 2-b. How is the `neural` data processed?

i. Three operations only: (1) restrict to columns where `cell_specimen_table/valid_roi` is True (in practice a no-op — all released ROIs are valid); (2) linearly interpolate every cell's event trace, vectorised across cells, onto the per-trial 30 Hz grid; (3) cast to `float32` and transpose to (n_neurons, n_timepoints). No smoothing, normalisation, z-scoring, baseline subtraction, deconvolution or neuron-level aggregation is applied. Neurons from different planes are never concatenated (each plane is its own session).

ii.
```python
def interp_rows(times, values, grid):
    """Vectorized linear interpolation of time x feature matrix."""
    j = np.searchsorted(times, grid, side='left')
    j = np.clip(j, 1, len(times)-1)
    lo, hi = j-1, j
    den = times[hi]-times[lo]
    a = np.divide(grid-times[lo], den, out=np.zeros_like(grid), where=den != 0)
    return values[lo] + (values[hi]-values[lo]) * a[:, None]
```
```python
events = events[:, valid]
...
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "Neural interpolation: Linear interpolation is consistent with paper event-triggered trace processing. No smoothing, normalization, or deconvolution is added." Step 5 mapping row: "Select valid ROI columns; linearly interpolate event magnitudes to a 30 Hz grid anchored to the first ophys timestamp in each native trial… Output is cells x time, float32. Detected events match paper methods." Step 10 Check 3: "Binning: linear interpolation matches paper event-triggered 30 Hz processing; no smoothing or normalization added."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the released `valid_roi` flag is used; no signal-based curation (SNR, event-rate, amplitude) is added. A shape guard raises if the ROI count and the event-trace column count disagree. A consequence documented but *not* filtered: 3,915 trials (4.64% of 84,313, spread over 122 sessions) contain an all-zero event matrix, which triggers 3,915 verifier warnings; these were verified against the original NWBs and deliberately retained.

ii.
```python
cells = h['processing/ophys/image_segmentation/cell_specimen_table']
valid = np.asarray(cells['valid_roi'][:], bool) if 'valid_roi' in cells else np.ones(events.shape[1], bool)
if len(valid) != events.shape[1]:
    raise ValueError(f'ROI/event mismatch in {path.name}')
events = events[:, valid]
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "ROI filtering: Use `valid_roi`; no extra amplitude threshold because files already passed release QC." Step 3 Curation: "Use released valid ROIs/cells and the event-detection output; do not rerun segmentation or event detection. Whitepaper experiment QC includes z-drift exclusion above 10 µm, residual-motion review, data-stream integrity checks, and temporal synchronization checks." Step 10 Check 1 on the all-zero trials: "Direct reconstruction from original event arrays confirmed these are genuine zero-event intervals (4.64% of 84,313 trials), expected for sparse detected events and low-cell-count planes. Filtering them would violate the required trial selection and bias against quiescent neural periods, so they are retained."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event is the **native trial start**. The 30 Hz sample grid for a trial consists of absolute times `ots[0] + k/30` for the integers `k` from `ceil((trial_start - ots[0])*30)` up to (exclusive) `ceil((trial_stop - ots[0])*30)`, clipped to the recorded ophys interval. Because the grid is expressed in absolute ophys-clock seconds and the neural trace is interpolated at those exact times, neural, stimulus, running and pupil all live on one common, ophys-anchored axis. Metadata records `temporal_alignment_event`, `off_start = 0.0`, `off_end = None` (variable trial length).

ii.
```python
origin = ots[0]
...
a = max(starts[ti], ots[0]); b = min(stops[ti], ots[-1] + DT)
k0 = int(np.ceil((a-origin)*HZ - 1e-9)); k1 = int(np.ceil((b-origin)*HZ - 1e-9))
grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
```
```python
'temporal_alignment_event':'Native trial start on a 30 Hz grid anchored to the experiment ophys timestamps',
'off_start':0.0,'off_end':None,
```

iii. The instructions say "Temporally align based on ophys timestamp" and "Segment each recording session into individual trials based on how they are defined in the experiment." CONVERSION_NOTES Step 4: "Paper resampled event-triggered behavior to 30 Hz; task explicitly requires ophys timestamps → Keep native ophys samples and align stimulus by intervals and continuous behavior by interpolation." Step 5 Key Decision 3 (grid anchor) and Step 10 Check 3: "Temporal alignment: all streams sampled on a 30 Hz grid anchored to ophys timestamps; independent boundary audit had zero failures."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **33.333 ms (30 Hz), uniform across every trial and session — i.e. rebinning is applied.** The native rates are ~30.95 Hz for the 239 single-plane experiments (so this is a near-identity resample with a slight drift of sampling instants) and ~11 Hz for the 45 multiscope planes (so those are **upsampled ~3×** by linear interpolation). Trial length remains variable (210–378 bins; mean 255.7).

ii.
```python
HZ = 30.0
DT = 1.0 / HZ
...
grid = origin + np.arange(k0, k1, dtype=np.float64)*DT
...
'metadata':{... 'time_bin_size':1000.0/HZ, ...}
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "Common bin is 1/30 s (33.333 ms): Papers use a common 30 Hz time series and target format requires equal bin size across sessions. Native single-plane and multiscope plane sampling rates differ, so retaining raw frame indices would violate that requirement." Step 3 cites the paper: "linearly interpolating onto a common 30hz timeseries," and Step 2/Step 3 note the hardware rates "31 Hz for single plane and… 11 Hz for each plane in multi-plane experiments."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Derived from the **natural-image stimulus presentation interval table** in `intervals/` — columns `image_name`, `start_time`, `stop_time`, `omitted` — not from the trials table. The table is discovered at runtime by required columns because its NWB name is image-set-specific (`Natural_Images_Lum_Matched_set_training_2017_presentations`, `..._ophys_6_2017_...`, `..._TRAINING_...`).

ii.
```python
def stimulus_table(h):
    for name, g in h['intervals'].items():
        if all(c in g for c in ('image_name','start_time','stop_time','is_change','omitted')):
            return g
    raise KeyError('active natural-image stimulus presentation table not found')
...
sg = stimulus_table(h)
ss = np.asarray(sg['start_time'][:],float); se = np.asarray(sg['stop_time'][:],float)
simg = np.array([dec(x) for x in sg['image_name'][:]], object)
somit = np.asarray(sg['omitted'][:],float) == 1
```

iii. CONVERSION_NOTES Step 1: "Stimulus presentations provide image identity on presentation intervals; omitted/gray intervals must be treated according to their explicit labels rather than guessed." Step 4: "NWB dynamic tables can have stimulus-specific names… Discover active natural-image interval group by columns; assign identity only on actual 250 ms non-gray start/stop intervals and gray otherwise." Step 10 recorded this as a resolved issue: "Stimulus dynamic-table name: It is image-set-specific rather than a fixed `stimulus_presentations` path. Script discovers it by required columns."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A 17-class time-varying integer row per trial. Class 0 = "gray" and is the default for every bin; a class 1–16 is written only for bins falling inside an actual `[start_time, stop_time)` presentation interval (~250 ms per flash). Omitted flashes stay gray. The image→code map is a hard-coded, alphabetically sorted list of the 16 image names (sets A and B combined), so codes are globally consistent across sessions. Resulting distribution: gray 0.670, each image ~0.020.

ii.
```python
IMAGE_NAMES = ['im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_CODE = {x:i+1 for i,x in enumerate(IMAGE_NAMES)}
...
image = np.zeros(len(grid), dtype=np.int16)
p0 = np.searchsorted(se, grid[0], side='right')
p1 = np.searchsorted(ss, grid[-1], side='right')
for pi in range(max(0,p0), min(len(ss),p1+1)):
    if not somit[pi] and simg[pi] in IMAGE_TO_CODE:
        mask=(grid >= ss[pi]) & (grid < se[pi])
        image[mask] = IMAGE_TO_CODE[simg[pi]]
```
```python
'output_values':[['gray']+IMAGE_NAMES, ...]
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Image identity: Class 0 is gray (including inter-image gray and omitted presentations); real images are globally sorted classes 1–16. This follows 'image presented during the non-grey screen.'" Step 3 supplies the timing basis: "The task presents 250 ms natural images separated by 500 ms gray, yielding one 750 ms image interval… omissions retain a nominal 750 ms interval." Step 9 checks the resulting occupancy against that 250/750 ratio: "gray 0.66975; each image ~0.0198–0.0216 | Yes."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on the same `grid` array used to interpolate the neural data, so row/column indices are identical by construction. Membership uses a half-open interval test `grid >= start_time & grid < stop_time` on absolute ophys-clock seconds.

ii.
```python
mask=(grid >= ss[pi]) & (grid < se[pi])
image[mask] = IMAGE_TO_CODE[simg[pi]]
...
outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
...
for n,o,inp in zip(neural,outputs,inputs):
    assert n.shape[1]==o.shape[1]==inp.shape[1] and n.shape[0]==valid.sum()
```

iii. Step 10 Check 3: "Temporal alignment: all streams sampled on a 30 Hz grid anchored to ophys timestamps; independent boundary audit had zero failures." Step 7's `--show-processing` plots are described as jointly displaying "event activity, non-gray image intervals, single-bin change pulses, continuous running/pupil traces, and their quintile labels on the same 30 Hz trial axis."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Derived from the stimulus presentation table's `is_change` flag together with the presentation `start_time` (and `omitted`), restricted to presentations whose onset lies inside the trial's native `[start_time, stop_time)`. It is **not** derived from the trials table's `change_time`/`go` columns. Because `is_change` is False for catch trials (which are `is_sham_change`), catch trials get an all-zero row.

ii.
```python
schange = np.asarray(sg['is_change'][:],float) == 1
...
if schange[pi] and not somit[pi] and starts[ti] <= ss[pi] < stops[ti]:
    q=np.searchsorted(grid,ss[pi],side='left')
    if q < len(grid): change[q]=1
```

iii. CONVERSION_NOTES Step 5 mapping row: "presentation `is_change` and start time → `output[1]`… Binary; omitted/sham changes are not true image changes." Step 10 Check 4 verified the consequence against raw data: "All 73,733 go trials have exactly one change pulse and all catch trials have zero."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary row, zero everywhere except a **single 33 ms bin** — the first grid sample at or after the true change onset. No widening to the flash duration or to the 750 ms presentation interval. Global rate: 0.34% of time bins are 1.

ii.
```python
change = np.zeros(len(grid), dtype=np.int16)
...
q=np.searchsorted(grid,ss[pi],side='left')
if q < len(grid): change[q]=1
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "Image change: A single sample pulse is 'right after' a true identity-change onset and avoids labeling the full changed-image interval as an event." Step 12 defends it against the near-chance decoding result: "the required image-change output is a one-bin event while calcium responses are delayed… The apparent low change score was investigated exhaustively; onset pulses exactly match all 73,733 raw go trials and no catch trials, so labels were retained rather than broadened into a biologically delayed response window."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Two categories, `['no_change','change']`, with no continuous quantity to threshold — the category is set by the boolean `is_change` flag, and the temporal extent of the "change" category is one bin. Sham (catch) changes and omissions are explicitly mapped to `no_change`.

ii.
```python
OUTCOMES = ['hit','miss','false_alarm','correct_reject']   # (for reference)
...
'output_values':[['gray']+IMAGE_NAMES,['no_change','change'], ...]
...
if schange[pi] and not somit[pi] and starts[ti] <= ss[pi] < stops[ti]:
    q=np.searchsorted(grid,ss[pi],side='left')
    if q < len(grid): change[q]=1
```

iii. Same justification as 4-b — the AI reads the instruction "Have value of 1 right after a change in image identity, otherwise 0" as literally one sample, and treats catch trials' sham changes as non-changes because `is_change` is False for them. Step 9 reports the resulting distribution and accepts it: "Image change | change onset event | `is_change` | one onset on go/change trials | pulse fraction 0.003403 | Yes."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same `grid` as the neural data; the pulse index is found with `np.searchsorted(grid, change_onset, side='left')`, i.e. the first grid bin at or after the onset time on the shared absolute ophys-clock axis. The row is then stacked into the same (5, T) output matrix whose T is asserted equal to the neural T.

ii.
```python
q=np.searchsorted(grid,ss[pi],side='left')
if q < len(grid): change[q]=1
...
outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
assert n.shape[1]==o.shape[1]==inp.shape[1]
```

iii. Step 10 Check 5: "Tested every converted trial against the ceil-based start/stop grid formula; zero length mismatches." Step 12 Check 2: "Processing plots place neural, stimulus, change, continuous behavior, and discretized labels on one ophys-anchored grid. Independent reconstruction and exhaustive boundary checks passed."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed` — the AllenSDK **filtered** running speed (10 Hz low-pass Butterworth, wrap/transient corrected) in cm/s, with its own `timestamps`. The unfiltered alternative `processing/running/speed_unfiltered` present in the same file was not used.

ii.
```python
rg = h['processing/running/speed']
rts, rsp = np.asarray(rg['timestamps'][:],float), np.asarray(rg['data'][:],float)
```

iii. CONVERSION_NOTES Step 1: "`RunningSpeed.from_nwb(..., filtered=True)` — Load the SDK-filtered running-speed time series and timestamps… SDK-filtered streams should be preferred to reproducing filters ad hoc." Step 5 mapping row: "filtered `processing/running/speed` → `output[2]`."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the filtered speed onto each trial's 30 Hz grid (via `np.interp`, which clamps — constant-extrapolates — outside the running-timestamp range), then percentile discretisation (see 5-c). The continuous values are retained only transiently for binning and plotting; only the integer quintile row is saved.

ii.
```python
def interp_1d_finite(times, values, grid):
    ok = np.isfinite(times) & np.isfinite(values)
    if ok.sum() < 2:
        raise ValueError('fewer than two finite samples for interpolation')
    return np.interp(grid, times[ok], values[ok]).astype(np.float32)
...
run_cont.append(interp_1d_finite(rts,rsp,grid))
```

iii. CONVERSION_NOTES Step 5 mapping: "Linear interpolation to trial grid, then session-wise quintile discretization." Step 4: "Streams have independent timestamps… Keep native ophys samples and align stimulus by intervals and continuous behavior by interpolation." Step 3 records that the paper itself linearly interpolates running onto a common 30 Hz series.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five bins from the 20/40/60/80th percentiles, computed **per experiment** over all time bins of all retained trials in that experiment (not globally across the dataset). `np.searchsorted(..., side='right')` assigns labels 0–4. The resulting distribution is exactly 20% per class in every session (and globally). Edges are recorded per session in `metadata['session_info'][i]['running_quintile_edges']`.

ii.
```python
def quintile(values):
    allv = np.concatenate(values)
    edges = np.percentile(allv, [20,40,60,80])
    return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges
...
run_bins, redges = quintile(run_cont); pupil_bins, pedges = quintile(pupil_cont)
```
```python
'continuous_discretization':'within-experiment quintiles over all retained trial time bins'
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "Behavior quintiles: Compute percentile edges independently per experiment from samples retained in eligible trials, preventing between-rig calibration and mouse-size differences from dominating labels. Duplicate quantile edges are handled with `searchsorted`, though continuous streams should normally yield all five classes." Step 9 confirms "each class 0.2000 | Yes by definition."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated at the very same `grid` timestamps as the neural data, inside the same per-trial loop, so it shares indices exactly; the quintile row is then stacked into the (5, T) output matrix with the shape assertion.

ii.
```python
n = interp_rows(ots, events, grid).T.astype(np.float32, copy=False)
...
run_cont.append(interp_1d_finite(rts,rsp,grid))
...
outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
assert n.shape[1]==o.shape[1]==inp.shape[1]
```

iii. Step 10 Check 3: "all streams sampled on a 30 Hz grid anchored to ophys timestamps; independent boundary audit had zero failures." Step 7's processing plots overlay the continuous speed and its quintile label on the same trial axis as the neural raster, which the notes describe as showing "no offset or length anomalies."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` — the SDK-**filtered** pupil-ellipse area — with `acquisition/EyeTracking/eye_tracking/timestamps`. The raw variant `area_raw` and the ellipse `width`/`height` columns were not used. Note that the filtered `area` is already NaN on `likely_blink` frames (verified: NaN fraction equals the blink fraction exactly), so blink frames are excluded implicitly by the finite-sample mask.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking/area' not in h:
    return None, 'missing pupil tracking'
...
eg = h['acquisition/EyeTracking']
ets = np.asarray(eg['eye_tracking/timestamps'][:],float)
area = np.asarray(eg['pupil_tracking/area'][:],float)
diameter = 2.0*np.sqrt(np.maximum(area,0.0)/np.pi)
```

iii. CONVERSION_NOTES Step 3: "Eye processing uses DeepLabCut perimeter points and ellipse fits for pupil, eye, and corneal reflection. The SDK's filtered pupil area is the reference-processed pupil-size measurement. Diameter can be derived as equivalent-circle diameter `2*sqrt(area/pi)`, which is monotonic with area and has physical 'diameter' semantics." Step 1: "Eye tracking uses z-score outlier detection and temporal dilation around outlier frames in the SDK loader; missing filtered samples must not be silently converted into a biological category."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area → equivalent-circle diameter `2*sqrt(max(area,0)/pi)`; drop non-finite samples (which are the blink/outlier frames); linearly interpolate the remaining samples onto the trial's 30 Hz grid with edge clamping; then percentile discretisation (6-c). Gaps are bridged by interpolation rather than being assigned a class.

ii.
```python
diameter = 2.0*np.sqrt(np.maximum(area,0.0)/np.pi)
...
pupil_cont.append(interp_1d_finite(ets,diameter,grid))
```
```python
def interp_1d_finite(times, values, grid):
    ok = np.isfinite(times) & np.isfinite(values)
    if ok.sum() < 2:
        raise ValueError('fewer than two finite samples for interpolation')
    return np.interp(grid, times[ok], values[ok]).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: "Pupil gaps: Interpolate using only finite positive filtered pupil values. Do not turn missing samples into a category. Exclude only sessions lacking the stream entirely." Step 5 mapping row: "Equivalent-circle diameter `2*sqrt(area/pi)`; interpolate across finite positive eye samples; session-wise quintiles… Monotonic conversion gives diameter semantics; three experiments with no pupil stream excluded."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: five bins at the 20/40/60/80th percentiles computed **per experiment** over all retained trial time bins, `searchsorted(side='right')` → labels 0–4, exactly 20% per class. Edges saved as `pupil_diameter_quintile_edges` in `session_info`.

ii.
```python
run_bins, redges = quintile(run_cont); pupil_bins, pedges = quintile(pupil_cont)
...
info=dict(... running_quintile_edges=redges.tolist(),
          pupil_diameter_quintile_edges=pedges.tolist(), ...)
```
```python
'output_values':[..., ['Q1_smallest','Q2','Q3','Q4','Q5_largest'], OUTCOMES],
```

iii. Same as 5-c — Step 5 Key Decision 7 ("Compute percentile edges independently per experiment… preventing between-rig calibration and mouse-size differences from dominating labels"), which is a stronger argument for pupil than for running because absolute pupil area depends on eye-camera geometry and per-animal calibration.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated at the same `grid` timestamps as the neural data inside the same per-trial loop; stacked as row 3 of the (5, T) output matrix with the shape/finiteness assertions.

ii.
```python
pupil_cont.append(interp_1d_finite(ets,diameter,grid))
...
outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
for n,o,inp in zip(neural,outputs,inputs):
    assert n.shape[1]==o.shape[1]==inp.shape[1] and n.shape[0]==valid.sum()
    assert np.isfinite(n).all() and np.isfinite(o).all()
```

iii. CONVERSION_NOTES Step 1 notes that the eye camera and ophys clocks are hardware-synced on a common sync board ("Temporal synchronization of all data-streams… recorded on a single NI PCI-6612 digital IO board"), so interpolation to the ophys-anchored grid is valid. Step 10 Check 2 independently reconstructed the pupil row for spot-checked trials from the original HDF5 and matched with `np.allclose`.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order. The first True flag determines the class; if none is True the code raises rather than defaulting.

ii.
```python
OUTCOMES = ['hit','miss','false_alarm','correct_reject']
...
out = next((oi for oi,name in enumerate(OUTCOMES) if bool(tr[name][ti])), None)
if out is None: raise ValueError(f'no outcome for eligible trial {ti} in {path.name}')
```

iii. CONVERSION_NOTES Step 1: "Trial outcome is represented by mutually meaningful boolean fields: hit/miss for go trials and false alarm/correct reject for catch trials." Step 4 confirms exclusivity empirically, and Step 10 Check 4 cross-checked raw vs converted counts: "Raw usable outcomes and converted labels match exactly: hit 15,682; miss 58,051; false alarm 921; correct reject 9,659."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The scalar class index 0–3 is broadcast to a constant row of length T so it can be stacked with the four time-varying rows into a single (5, T) matrix. Semantically it remains static per trial. Global time-weighted distribution: hit 0.182, miss 0.692, false alarm 0.010, correct reject 0.115 (heavily miss-dominated because the 82 passive experiments are retained and yield almost exclusively miss/correct-reject).

ii.
```python
for i,g in enumerate(grids):
    outcome=np.full(len(g), outcomes[i], dtype=np.int16)
    outputs.append(np.vstack((image_rows[i],change_rows[i],run_bins[i],pupil_bins[i],outcome)))
```

iii. CONVERSION_NOTES Step 5 mapping: "Four-class label repeated at every time bin… Semantically static per trial; repeated because validator requires a single 2-D matrix when other outputs vary in time." Step 10 records it as a resolved design issue: "Mixed static/time-varying outputs: Trainer accepts a trial output as wholly 1-D or 2-D. Repeated static outcome across time to preserve semantics while supporting the four time-varying rows."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Missing pupil stream** (3 experiments): whole experiment excluded and recorded in `metadata['excluded_sessions']` with a reason string; the conversion continues.
- **Blink / outlier pupil samples** (~9% of eye frames, NaN in the filtered `area`): removed before interpolation, so gaps are bridged by interpolation rather than becoming a pupil class.
- **Behaviour streams that do not span the whole ophys recording** (running starts ~8 s late and ends ~17 s early): `np.interp` clamps to the nearest valid sample rather than producing NaN or a spurious class.
- **Negative pupil area**: clipped at 0 before the square root.
- **Trials extending past the recording / not overlapping it**: window clipped to `[ots[0], ots[-1]+DT]`, and non-overlapping trials dropped; degenerate windows (<2 samples) skipped.
- **Degenerate interpolation** (`den == 0`, fewer than 2 finite samples): guarded with `np.divide(..., where=)` and an explicit `ValueError`.
- **Unrecognised image names / ROI-count mismatch / missing outcome flag**: the first is silently left as gray; the latter two raise.
- **All-zero neural trials** (3,915): verified genuine and deliberately retained.
Explicit assertions check per-trial shape consistency, finiteness, and a ≥2-trial minimum before a session is emitted.

ii.
```python
if 'acquisition/EyeTracking/pupil_tracking/area' not in h:
    return None, 'missing pupil tracking'
...
diameter = 2.0*np.sqrt(np.maximum(area,0.0)/np.pi)
...
ok = np.isfinite(times) & np.isfinite(values)
if ok.sum() < 2:
    raise ValueError('fewer than two finite samples for interpolation')
return np.interp(grid, times[ok], values[ok]).astype(np.float32)
...
a = np.divide(grid-times[lo], den, out=np.zeros_like(grid), where=den != 0)
...
trial_idx = trial_idx[(stops[trial_idx] >= ots[0]) & (starts[trial_idx] <= ots[-1])]
if len(grid) < 2: continue
...
assert np.isfinite(n).all() and np.isfinite(o).all()
```
```python
if result is None:
    print(f'  EXCLUDED: {reason}',flush=True); excluded.append({'ophys_experiment_id':eid,'reason':reason}); continue
```

iii. CONVERSION_NOTES Step 4: "Required decoder output includes pupil diameter → Exclude the three experiments because the required target cannot be constructed without unjustified imputation." Step 5 Key Decision 8: "Do not turn missing samples into a category." Step 1: "missing filtered samples must not be silently converted into a biological category." Step 10 Check 5: "Three entirely missing pupil sessions are documented exclusions; finite gaps in available pupil streams are interpolated from filtered valid samples."

## 9-a. What are the most time-consuming steps of the code?

i. Total full conversion was 282.2 s for 284 files. Per-experiment `process_experiment` time summed to 267.7 s (94.9%), leaving ~14 s for pickling the 13.7 GB output. Within `process_experiment` the cost is (1) reading the full `event_detection/data` matrix (up to 149,508 × 666 float32) plus the 270k-sample running and 136k-sample eye arrays from HDF5, and (2) the per-trial loop, whose dominant term is `interp_rows`, which materialises two (T × n_cells) gather arrays per trial. Timing per experiment is printed and stored in `info['seconds']`.

ii.
```python
def process_experiment(path, row, show=False):
    t0 = time.time()
    ...
    info=dict(..., seconds=round(time.time()-t0,3))
...
print(f"  neurons={info['n_neurons']} trials={info['n_trials']} time={info['seconds']}s",flush=True)
...
print(f'Done: sessions={len(neural)} trials={sum(map(len,neural))} ... elapsed={time.time()-overall:.1f}s')
```

iii. CONVERSION_NOTES Step 6: "Loading complete 247 GB NWBs through high-level SDK objects would add unnecessary overhead… Output size is inherently large because all event cells and native trial durations are retained at 30 Hz." Step 7's timing table reports "Conversion | 0.4–1.1 s computation plus serialization | Approximately 5–10 minutes… below 15 minutes," and Step 9 confirms the actual 282.2 s.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Already vectorised: neural interpolation across all cells at once (`interp_rows`), and both behaviour interpolations (`np.interp`). Remaining scalar loops that could be vectorised: (1) the outer `for ti in trial_idx` per-trial loop — the grid construction and `searchsorted` lookups could be computed for all trials at once; (2) the inner `for pi in range(p0, p1+1)` loop over stimulus presentations within each trial, which builds an O(T) boolean `mask` per presentation (~10–16 presentations per trial) when the whole image row could be produced with one `np.searchsorted` of the grid into the presentation boundaries; (3) the list comprehension inside `quintile` that calls `searchsorted` once per trial instead of once per experiment on a concatenated array; (4) `simg = np.array([dec(x) for x in sg['image_name'][:]], object)`, a Python-level decode of ~70k byte strings per file.

ii.
```python
for pi in range(max(0,p0), min(len(ss),p1+1)):
    if not somit[pi] and simg[pi] in IMAGE_TO_CODE:
        mask=(grid >= ss[pi]) & (grid < se[pi])
        image[mask] = IMAGE_TO_CODE[simg[pi]]
```
```python
return [np.searchsorted(edges, x, side='right').astype(np.int16) for x in values], edges
```
```python
simg = np.array([dec(x) for x in sg['image_name'][:]], object)
```

iii. The notes do not enumerate the residual loops; they claim the vectorisation that was done was sufficient. Step 6: "Neural interpolation is vectorized over all cells and each source dataset is loaded once per experiment. Trial stimulus lookup uses sorted interval searches." Step 7 concluded the runtime was well inside the 15-minute budget, so no further vectorisation was pursued.

## 9-c. What processing does the code repeat multiple times?

i. Genuinely repeated work, none of it flagged in the notes:
- `interp_1d_finite` recomputes `np.isfinite(times) & np.isfinite(values)` and the boolean-indexed copies `times[ok]`, `values[ok]` over the **entire** running (270,240 samples) and eye (135,981 samples) arrays **once per trial** — up to 412 times per experiment, although the mask is trial-independent.
- The outcome flags are read from HDF5 element-by-element inside the trial loop (`tr[name][ti]`, up to 4 scalar reads per trial) instead of being loaded once as arrays the way `go`, `catch`, `start_time` and `stop_time` are.
- `interp_rows` re-runs `np.searchsorted` against the full ophys timestamp vector for each trial.
- The stimulus `start_time` / `stop_time` / `image_name` arrays are re-scanned per trial via `p0`/`p1`.

ii.
```python
def interp_1d_finite(times, values, grid):
    ok = np.isfinite(times) & np.isfinite(values)      # recomputed every trial
    ...
    return np.interp(grid, times[ok], values[ok]).astype(np.float32)
...
for ti in trial_idx:
    ...
    out = next((oi for oi,name in enumerate(OUTCOMES) if bool(tr[name][ti])), None)  # per-trial HDF5 reads
    run_cont.append(interp_1d_finite(rts,rsp,grid))
    pupil_cont.append(interp_1d_finite(ets,diameter,grid))
```

iii. The notes claim the opposite. Step 6: "Direct HDF5 access reads only required datasets. Neural interpolation is vectorized over all cells and **each source dataset is loaded once per experiment**." No repetition is acknowledged anywhere in CONVERSION_NOTES; the justification offered is that the observed runtime (282 s total) was acceptable.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work performed whose result is never used, or is nullified downstream:
- **Area → equivalent-circle diameter** (`2*sqrt(max(area,0)/pi)`) is a strictly monotonic transform applied immediately before percentile binning; percentile bins are invariant under monotonic transforms, so the conversion has **literally no effect** on the saved `pupil_diameter_quintile` row.
- **Empty input arrays**: a `(0, T)` float32 array is allocated, shape-asserted and pickled for every one of the 84,313 trials, even though `input_names` is empty and the decoder has no inputs.
- **Constant outcome row**: a static per-trial scalar is expanded to a full length-T int16 row for all 84,313 trials (~21.6M redundant values).
- **Continuous running/pupil traces** (`run_cont`, `pupil_cont`) and `grids` are materialised and retained for every trial but are discarded after binning except for the single trial used in `--show-processing` plots.
- **30 Hz resampling of the 239 single-plane experiments** is a near-identity resample of ~30.95 Hz data that costs the interpolation while changing almost nothing.
- Minor: `IMAGE_TO_CODE` membership test and `dec()` decode run over all presentations including those outside every trial window.

ii.
```python
diameter = 2.0*np.sqrt(np.maximum(area,0.0)/np.pi)   # monotonic; nullified by percentile bins
...
inputs.append(np.empty((0,len(g)),dtype=np.float32))
outcome=np.full(len(g), outcomes[i], dtype=np.int16)
...
neural.append(n); grids.append(grid); image_rows.append(image); change_rows.append(change)
run_cont.append(interp_1d_finite(rts,rsp,grid))
pupil_cont.append(interp_1d_finite(ets,diameter,grid))
```

iii. CONVERSION_NOTES does not identify any of these as unnecessary. The constant outcome row is explicitly defended as a format requirement (Step 10: "Trainer accepts a trial output as wholly 1-D or 2-D… Repeated static outcome across time to preserve semantics"); the empty input arrays are defended as the literal encoding of "no inputs" (Step 5: "Decoder Task explicitly specifies no inputs"); the diameter conversion is defended on semantic grounds (Step 3: "Monotonic conversion gives diameter semantics") without noting that percentile binning makes it inert; and Step 6 acknowledges only that "Output size is inherently large because all event cells and native trial durations are retained at 30 Hz."
