# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI works entirely from the local data mount rather than the Allen S3 cache. It reads the SDK's canonical experiment metadata table (`project_metadata/ophys_experiment_table.csv`), intersects it with the set of NWB files actually present on disk (284 files, found by `glob`), and then keeps only **active (non-passive) behavior sessions** (`passive == False`), giving 202 experiments / 38 mice / 174 unique ophys sessions. It does **not** filter on `project_code`, so both `VisualBehavior` (168 active single-plane experiments, CAM2P, ~31 Hz) and `VisualBehaviorMultiscope` (34 active planes from 6 sessions of 1 mouse, MESO.1, ~11 Hz) are included. Each experiment is then loaded as a full SDK object with `pynwb.NWBHDF5IO` + `BehaviorOphysExperiment.from_nwb()`, from which only the needed streams are copied out (ophys timestamps, events, trials, stimulus presentations, running speed, eye tracking). A separate, fast "stats" pre-pass opens every NWB file directly with `h5py` to collect the global image-name set and pooled running/pupil samples used for the global discretization edges.

ii.
```python
def get_experiment_list(sample=False):
    et = pd.read_csv('data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
    nwb_ids = set(int(f.split('experiment_')[1].split('.nwb')[0])
                  for f in glob.glob('data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb'))
    m = et[et['ophys_experiment_id'].isin(nwb_ids)]
    a = m[m['passive']==False].copy().sort_values('ophys_experiment_id').reset_index(drop=True)
    if sample: a = a.head(2)
    print(f"Selected {len(a)} experiments, {a['mouse_id'].nunique()} mice")
    return a

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
```

iii. From CONVERSION_NOTES Step 1/2 and the trajectory: the AI first confirmed only 284 of the 1,936 released experiments are available locally, so the S3 cache route used by the SDK tutorials is unusable and `from_nwb()` on the local file is the equivalent entry point. It read the task line *"Collect and convert data under the 'Visual Behavior' task"* as "all sessions in which the mouse actually performed the change-detection task", and the paper's statement that *"Imaging was also performed during passive viewing of the same stimulus, which was not analyzed here"* as grounds to drop the 82 passive experiments. It explicitly decided **not** to restrict to the paper's analysis subset (familiar images, MESO rig only), reasoning that the task asks for all Visual Behavior data, not the paper's figure subset: *"The task doesn't restrict to familiar only... I should use all active (non-passive) sessions that are available."* The h5py pre-pass was justified in Step 6 as an optimisation (*"h5py for fast stats collection (0.1s vs 5s per experiment with SDK)"*) needed because percentile edges and the image-code map must be global.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the selected experiments, cast to `str` and sorted; a dict maps mouse id → index, and each emitted session records its mouse's index in `subject_idx`. Result: 38 subjects (37 single-plane mice + 1 Multiscope mouse).

ii.
```python
subjects = sorted(set(str(r['mouse_id']) for _, r in el.iterrows()))
s2i = {s: i for i, s in enumerate(subjects)}
...
asi.append(s2i[str(row['mouse_id'])])
...
'subjects': subjects, 'subject_idx': np.array(asi, dtype=np.int64),
```

iii. Documented in CONVERSION_NOTES Step 2 (*"Subjects | 38"*). `mouse_id` is the SDK's unique animal identifier in the experiment table; the AI cross-checked the count against the metadata table (38 unique mice among the 202 active experiments) and reported it in the Step 9 consistency table as matching.

## 1-c. How are the data split into sessions?

i. **One ophys *experiment* (imaging plane) = one "session"** in the output. The AI iterates over the rows of the experiment table and appends one entry to `neural`/`input`/`output` per experiment; it never groups by `ophys_session_id`. For the 168 single-plane CAM2P experiments this is a 1:1 mapping, but the 34 active Multiscope planes (which come from only 6 physical sessions of mouse 457841) become 34 separate "sessions" that share identical behavioural data, so that one mouse contributes 34 of the 202 sessions.

ii.
```python
for i, (_, row) in enumerate(el.iterrows()):
    eid = row['ophys_experiment_id']
    ed = load_experiment(eid)
    ntl, otl, nn = process_experiment(ed, imn, rbe, pbe)
    ...
    an.append(ntl); ai.append(it); ao.append(otl)
    asi.append(s2i[str(row['mouse_id'])])
    abri.append(np.full(nn, r2i[row['targeted_structure']], dtype=np.int64))
```

iii. Trajectory step 43: *"CAM2P sessions: 1 experiment per session (168 sessions); MESO sessions: 3–7 experiments per session (6 sessions). Each experiment is a separate imaging plane with its own neurons. MESO experiments share the same behavioral data but different neurons from different planes/depths. For the decoder, each experiment should be treated as a separate 'session' since each has its own set of neurons."* The justification is therefore that the target format requires a fixed neuron set per session and that the planes are distinct populations; the AI did not consider concatenating simultaneously recorded planes into one neuron dimension.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. A trial is kept if it is a go **or** catch trial and is neither aborted nor auto-rewarded. Each trial occupies the bins of the session-wide time grid whose centres fall in `[start_time, stop_time)`, i.e. variable-length trials (75–135 bins, mean 91 bins ≈ 8.5 s at 93 ms). Trials yielding fewer than 2 bins are dropped.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
if len(vt) == 0: return [], [], nn
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
for _, t in vt.iterrows():
    tm = (bc>=t['start_time'])&(bc<t['stop_time'])
    ti = np.where(tm)[0]
    if len(ti) < 2: continue
    ...
    ntl.append(nr[:, ti])
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules: Include Go + Catch, exclude Aborted + Auto-rewarded") simply follows the explicit instruction. The full `start_time`→`stop_time` window was chosen after the AI measured trial timing (trajectory step 32: *"Trial length ~7.3–12.5 s, mean ~8.6 s; change time 3–8.3 s after start; stop_time consistently ~4.2 s after change"*) and concluded the natural window contains the pre-change flashes plus the post-change response period, which is what makes the time-varying outputs (image identity, image change) informative.

## 1-e. How are trials filtered based on quality controls?

i. Quality control is: (a) exclude aborted trials (early lick, no change presented); (b) exclude auto-rewarded trials (free reward); (c) drop any trial covering fewer than 2 time bins; (d) drop any experiment that yields fewer than 2 usable trials (`if len(ntl) < 2: SKIPPED`); (e) drop passive experiments up front (see 1-a). No check on `change_time` validity is made, and no trial is rejected for degenerate neural content — 2,621 trials with all-zero event data were kept and merely noted as warnings.

ii.
```python
vt = tr[(tr['go']|tr['catch'])&~tr['aborted']&~tr['auto_rewarded']]
...
    if len(ti) < 2: continue
...
if len(ntl) < 2:
    print(f"  [{i+1}/{len(el)}] Exp {eid}: SKIPPED")
    skipped += 1; continue
```

iii. CONVERSION_NOTES Step 3/Step 10: exclusions follow the instruction verbatim and are checked ("Trial filtering: Go+Catch only, excluding Aborted and Auto-rewarded. Verified on sample experiment."). The ≥2-trial/≥2-bin rules implement the format requirement *"There needs to be at least two trials within each session"*. The all-zero-neural warnings were dismissed in Step 10 as *"expected for sparse calcium events... in MESO experiments with few neurons"* rather than filtered out.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `dataset.events.events` — the **deconvolved discrete calcium event magnitudes** (not `dff_traces`, and not the smoothed `filtered_events`), stacked into an `(n_neurons, n_frames)` float32 array, together with `dataset.ophys_timestamps` for the timebase.

ii.
```python
'ophys_timestamps': ds.ophys_timestamps.copy(),
'events': np.vstack(ds.events.events.values).astype(np.float32),
```

iii. CONVERSION_NOTES Step 1: *"Paper uses 'events' (deconvolved calcium events) for neural analysis"*, and Step 10 check 6: *"Uses events (not dff), matches paper specification."* This follows the methods text the AI read: *"For all analysis of neural data we used the detected calcium events... We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of GCaMP6f."* The AI also noted the tutorial's argument that events *"exclude prolonged Ca transients that may contaminate neural responses to subsequent stimuli"*, but did not consider the `filtered_events` variant that the same tutorial mentions.

## 2-b. How is the `neural` data processed?

i. The only processing is temporal re-binning onto a uniform 93 ms grid spanning the session's valid-trial period. For each bin centre, the event traces of all ophys frames falling in the 93 ms window centred on that bin are averaged; if a bin contains no ophys frame (possible for the ~11 Hz Multiscope data), the previous bin's values are carried forward. No dF/F computation, no smoothing, no z-scoring, no normalisation, and no merging of planes (each plane stays its own session).

ii.
```python
TARGET_BIN_SIZE = 0.093

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
```

iii. Trajectory steps 44/58: the AI found two frame rates in the selected data (CAM2P ~32 ms, MESO ~93 ms) and concluded *"the time bin size needs to be the same for all trials and sessions... Since CAM2P is ~32 ms and MESO is ~93 ms, I need to resample. The most natural approach would be to downsample CAM2P data to match the MESO rate (~93 ms), since we can't upsample MESO data without inventing data"*, adding that 93 ms ≈ 750 ms / 8 gives a whole number of bins per stimulus interval. Averaging (rather than summing) was not separately justified; CONVERSION_NOTES Step 6 lists "session-wide resampling then trial extraction" as an efficiency choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is performed. Every ROI present in the NWB `events` table is kept (29,444 neurons total; 4–666 per session, mean 145.8).

ii. None — all rows of `ds.events.events` are used verbatim:
```python
'events': np.vstack(ds.events.events.values).astype(np.float32),
```

iii. CONVERSION_NOTES Step 1/Step 3: *"Cell filtering (valid_roi) already applied in NWB files; ROI filtering uses multi-label classifier"* and *"Neuron curation: ROI filtering already applied (valid_roi=True)"*. I.e. the AI verified the published NWB files already contain only classifier-passed, valid ROIs, so additional filtering would double-filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, but indirectly: one uniform bin grid is built for the whole session, anchored at the **first valid trial's** `start_time` (`bc = arange(s0 + bin/2, s1, bin)`); a trial's data is the contiguous set of bins whose centres lie in `[start_time, stop_time)`. Neural bins are formed by averaging ophys frames inside each bin window, so the alignment to the ophys clock is exact per bin, but the phase of the grid relative to each individual trial's `start_time` can differ by up to one bin (93 ms). All output streams are sampled on exactly the same grid, so neural↔output alignment is exact by construction. The metadata declares `temporal_alignment_event = 'Trial start time'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
nr = resample_session(ev, ots, bc)          # neural on the grid
...
tm = (bc>=t['start_time'])&(bc<t['stop_time'])
ti = np.where(tm)[0]
ntl.append(nr[:, ti])
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc)]))
```

iii. The instruction "Temporally align based on ophys timestamp" is honoured by mapping every stream onto bins defined by ophys frame membership (`np.searchsorted(ophys_ts, ...)`). The AI's stated reason for a session-wide grid (CONVERSION_NOTES Step 6) is efficiency and guaranteed consistency: *"Session-wide resampling then trial extraction (vs per-trial resampling)"*, which also guarantees all five outputs and the neural data share one index set. Step 10 check 3 verified the resampling ratio: *"Original trial has 225 frames at 32.3 ms, converted has 78 bins at 93 ms. Ratio 225/78 = 2.88 ≈ 32.3/93. Correct."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 93 ms (10.75 Hz) for every trial and session; `metadata['time_bin_size'] = 93.0` ms. Yes — rebinning is applied to everything: the 31 Hz CAM2P recordings are downsampled ~2.9×, and the ~11 Hz MESO recordings are resampled onto a slightly different (nearly matched) grid. Trial lengths become 75–135 bins (mean 91).

ii.
```python
TARGET_BIN_SIZE = 0.093
...
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)
...
'time_bin_size': TARGET_BIN_SIZE * 1000,
'bin_size_seconds': TARGET_BIN_SIZE,
```

iii. Trajectory steps 44/58 and CONVERSION_NOTES Step 4 (*"Frame rate | 11 Hz + 31 Hz | 11 Hz (MESO) | Resample all to ~93 ms bins"*): because the AI kept both rigs, the format rule "time bins should be the same size for all trials and sessions" forces a common bin; 93 ms was chosen as the coarser native rate (no invented data) and because 750 ms / 8 ≈ 94 ms tiles the stimulus interval.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From `stimulus_presentations`: the `stimulus_block_name` (to keep only the `change_detection_behavior` block), `image_name`, `omitted`, and `start_time` columns. It is **not** taken from the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
cd = sp[sp['stimulus_block_name'].str.contains('change_detection')]
cdi = cd[(cd['image_name']!='omitted')&(~cd['omitted'].astype(bool))]
ist = cdi['start_time'].values
iix = np.array([n2i.get(n, 0) for n in cdi['image_name'].values])
```

iii. CONVERSION_NOTES Step 5 maps `stimulus_presentations.image_name → output[0]: image_identity (Categorical, time-varying)`. The AI inspected the stimulus table early (trajectory step 26: *"Stimulus presentations: image_name, start_time, end_time, is_change, omitted, active, stimulus_block_name; 8 unique images + 'omitted'; blocks: initial_gray_screen, change_detection_behavior, post_behavior_gray_screen, natural_movie_one"*) and used the flash table as the direct record of which image was on screen, restricting to the behavioural block so that the movie/gray blocks cannot contaminate the labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global image→code map is built in the h5py pre-pass from every image name appearing in any NWB stimulus table (excluding `omitted`/empty/NaN), sorted alphabetically — 16 codes for the union of image sets A and B. Per bin, the identity is the code of the **most recent non-omitted flash at or before the bin centre** (step-hold), so the label persists through the 500 ms grey inter-stimulus interval and through omitted flashes. Bins before any flash (in practice none within trials) default to code 0.

ii.
```python
all_img = sorted(img_set)                       # in fast_collect_stats
...
n2i = {n: i for i, n in enumerate(imn)}
sidx = np.searchsorted(ist, bc, side='right') - 1
ib = np.zeros(nb, dtype=np.int64)
m = sidx >= 0
ib[m] = iix[np.clip(sidx[m], 0, len(iix)-1)]
```

iii. CONVERSION_NOTES Step 10 check 5: *"Image identity: 16 unique images across image sets A and B. Correctly mapped."* A global (rather than per-session) code map is needed because different mice see image set A or B; sorting makes it deterministic. The step-hold convention implements the instruction "Image identity (of the image presented during the non-grey screen)" by labelling each grey gap with the image that was just shown.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated at the same bin centres `bc` used for the neural re-binning, and the same index vector `ti` slices both, so identity and neural activity are sample-for-sample aligned by construction. The switch point is the first bin centre at/after a new flash's `start_time`.

ii.
```python
sidx = np.searchsorted(ist, bc, side='right') - 1     # same bc as resample_session
...
tm = (bc>=t['start_time'])&(bc<t['stop_time']); ti = np.where(tm)[0]
ntl.append(nr[:, ti])
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. Implicit in the Step 6 design decision to resample everything onto one session-wide grid before segmenting; the `--show-processing` plots (`processing_775614751.png`, `processing_788490510.png`) were produced to visually confirm neural/image/change/running/pupil traces share a time axis with no offset.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From `stimulus_presentations.is_change` (within the change-detection block) and the corresponding flash `start_time`. Because `is_change` is only True when the displayed image actually differs from the previous flash, catch (sham-change) trials contribute no change events — I verified this directly on experiment 1007107386: 20/20 go-trial `change_time`s coincide with an `is_change` flash and 0/20 catch-trial `change_time`s do.

ii.
```python
cts = cd[cd['is_change']==True]['start_time'].values
```

iii. CONVERSION_NOTES Step 5: `stimulus_presentations.is_change → output[1]: image_change (Binary, time-varying)`. The AI took the SDK's own change flag as the definition of "a change in image identity" rather than re-deriving it from the trials table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector over the session grid is set to 1 for every bin whose centre lies in `[change_flash_start, change_flash_start + 750 ms)` — i.e. the changed image flash (250 ms) plus the following grey interval (500 ms), which is 8 bins of 93 ms (verified: 8 consecutive ones starting at the first bin ≥ `change_time`). Elsewhere it is 0.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts:
    cb[(bc>=ct)&(bc<ct+0.750)] = 1
```

iii. The 750 ms window follows the paper's unit of analysis, which the AI recorded in CONVERSION_NOTES Step 3 (*"ISI 500 ms", "Image duration 250 ms"*, *"750 ms image presentation intervals"*) from the methods text: *"By image presentation interval we refer to the 750 ms interval beginning with each image presentation."* This satisfies the instruction "value of 1 right after a change in image identity, otherwise 0" while keeping the marker a transient event rather than a whole-trial label.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — it is natively binary with value names `['no_change', 'change']`. The only implicit "threshold" is the 750 ms duration of the positive window. Resulting distribution over the whole dataset: 92.3 % `no_change`, 7.7 % `change`.

ii.
```python
cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
ov = [imn, ['no_change','change'], [f'bin_{i}' for i in range(5)],
      [f'bin_{i}' for i in range(5)], ['hit','miss','false_alarm','correct_reject']]
```

iii. CONVERSION_NOTES Step 7/9 record the check of the resulting distribution (*"Image change: 92.1 % no_change, 7.9 % change — reasonable"*), consistent with roughly one 750 ms change window per ~8.5 s trial.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same mechanism as image identity: computed on the shared bin-centre grid and sliced with the same `ti` indices as the neural matrix, so the first "1" bin is the first bin whose centre is at or after the change flash onset.

ii.
```python
cb = np.zeros(nb, dtype=np.int64)
for ct in cts: cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]))
```

iii. Same justification as 3-c (single session-wide grid → guaranteed alignment); the AI's `--show-processing` panels plot the change trace against the neural trace on a common time axis for two sessions as visual verification.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed`, i.e. the SDK's filtered linear running speed (cm/s) with its own ~60 Hz timestamps (NWB `processing/running/speed`, not `speed_unfiltered` or `dx`). The same `processing/running/speed/data` array, subsampled every 10th sample, is read via h5py in the pre-pass to build the global bin edges.

ii.
```python
rts, rsp = run['timestamps'].values, run['speed'].values
...
rs = f['processing']['running']['speed']['data'][::10]     # in fast_collect_stats
```

iii. CONVERSION_NOTES Step 5 maps `running_speed → output[2]`. The AI had inspected the stream (trajectory step 26: *"Running speed: timestamps + speed columns, sampled at ~60 Hz"*) and used the SDK's standard attribute; the whitepaper text it read notes the filtered speed is the one exposed as `running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Two steps: (1) linear interpolation of the ~60 Hz speed trace onto the 93 ms bin centres, after dropping NaN samples, with `bounds_error=False, fill_value=np.nan` (if fewer than 2 valid samples exist the whole trace becomes NaN); (2) discretisation with globally pooled quintile edges. The edges are computed **once, before the main pass**, from every 10th sample of every selected experiment's full-session speed trace (i.e. including grey-screen and movie epochs, and counting the Multiscope mouse's behaviour once per plane), with the outermost edges replaced by ±inf. No smoothing or anti-alias filtering is done before downsampling.

ii.
```python
vm = ~np.isnan(rsp)
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False,
                          fill_value=np.nan)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
rb = dig(ri, rbe)

def pct_bins(v, n=5):
    v2 = v[~np.isnan(v)]
    if len(v2) == 0: return np.linspace(-1, 1, n+1)
    e = np.percentile(v2, np.linspace(0, 100, n+1))
    e[0] = -np.inf; e[-1] = np.inf
    return e
```

iii. CONVERSION_NOTES Step 5 (*"Interpolate + 5 percentile bins"*) and Step 6 (h5py pre-pass justified as 50× faster than an SDK pass). Global rather than per-session edges keep the five categories comparable across sessions; the instruction requires "five equal percentile bins". The achieved within-trial distribution is 0.199 / 0.203 / 0.184 / 0.209 / 0.205, which the AI reported without comment.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five quintile bins of the pooled distribution (edges −∞, −0.0041, 0.336, 14.18, 33.14, +∞ cm/s), labelled `bin_0`…`bin_4`. `np.digitize` assigns each interpolated value; NaN values are assigned to the middle bin (2).

ii.
```python
def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2
    return b.astype(np.int64)
```

iii. Direct implementation of "discretized into five equal percentile bins". The ±inf outer edges guarantee every finite value lands in 0–4 (defence against values outside the pre-pass sample). Mapping NaN to the middle (median) bin was chosen as the least-distorting default; it is not discussed explicitly in CONVERSION_NOTES beyond the general "handle missing data appropriately" requirement.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated directly at the shared bin centres `bc` and sliced with the same `ti`, so it is sample-for-sample aligned with the neural matrix. Running timestamps and ophys timestamps come from the same hardware-synced clock, so no additional offset correction is applied.

ii.
```python
ri = interpolate.interp1d(rts[vm], rsp[vm], 'linear', bounds_error=False, fill_value=np.nan)(bc)
rb = dig(ri, rbe)
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], ...]))
```

iii. From the whitepaper text the AI read (trajectory step 18): *"Temporal synchronization of all data-streams (calcium imaging, visual stimulation, body and eye tracking cameras) was achieved by recording all experimental clocks on a single NI PCI-6612 digital IO board at 100 kHz"* — hence interpolating the behavioural stream onto the ophys-derived grid is legitimate.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking['pupil_area']` (with `timestamps`), not `pupil_width`. Blinks are handled implicitly: the SDK/NWB already stores NaN in `pupil_area` for `likely_blink` frames (I confirmed 12,500/12,500 blink frames are NaN in a sample file), and the AI's NaN mask removes exactly those samples. The pre-pass reads the same quantity directly as `acquisition/EyeTracking/pupil_tracking/area`.

ii.
```python
ets, pa = eye['timestamps'].values, eye['pupil_area'].values
vm2 = ~np.isnan(pa)
...
pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]    # in fast_collect_stats
```

iii. CONVERSION_NOTES Step 5 maps `eye_tracking.pupil_area → output[3]: pupil_diameter`; trajectory step 26 records the inspection of the eye-tracking table (*"Eye tracking: timestamps + pupil_area (and other columns), sampled at ~30 Hz"*), and step 89 records checking the NWB layout (`likely_blink`, `pupil_tracking/area`, `area_raw`) before choosing `area` over the raw variant. Step 10 notes a fix: *"Fixed pupil area collection in h5py (use area key, not width*height)"*. The AI never states explicitly that it is relying on the SDK's blink NaNs.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Identical pipeline to running speed: drop NaN (= blink/untracked) samples, linearly interpolate the ~30 Hz trace onto the 93 ms bin centres (NaN outside the tracked range), then apply globally pooled quintile edges computed in the pre-pass from every 10th pupil-area sample of every experiment. If an experiment has <2 valid pupil samples the trace becomes all-NaN and every bin lands in the middle category.

ii.
```python
vm2 = ~np.isnan(pa)
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False,
                          fill_value=np.nan)(bc) if vm2.sum()>=2 else np.full(nb, np.nan)
pb = dig(pi, pbe)
...
pbe = pct_bins(pa_all, 5)
```

iii. Same rationale as running speed (CONVERSION_NOTES Step 5: *"Interpolate + 5 percentile bins"*). No conversion from area to diameter is performed, since only the rank order matters after percentile binning.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five quintile bins of the pooled pupil-area distribution (edges −∞, 4372.1, 5526.1, 6745.7, 8597.6, +∞ px²), labelled `bin_0`…`bin_4`; NaN → middle bin (2). Achieved within-trial distribution: 0.209 / 0.185 / 0.209 / 0.191 / 0.206.

ii.
```python
pbe = pct_bins(pa_all, 5)
...
pb = dig(pi, pbe)
```

iii. Same `pct_bins`/`dig` helpers and same justification as running speed; the AI reported the resulting near-uniform distribution in the Step 9/verification output as evidence the discretisation is behaving as intended.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated at the shared bin centres and sliced by the same `ti` index vector as the neural data; eye-tracking timestamps are on the same hardware-synced clock as the ophys frames, so no further correction is applied.

ii.
```python
pi = interpolate.interp1d(ets[vm2], pa[vm2], 'linear', bounds_error=False, fill_value=np.nan)(bc)
pb = dig(pi, pbe)
...
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti], np.full(len(ti), oc, dtype=np.int64)]))
```

iii. Same as 5-d (single grid + hardware-synced clocks); the processing plots include the pupil trace on the same time axis as the neural trace for visual confirmation.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, and (by fallback) `correct_reject`.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
```

iii. CONVERSION_NOTES Step 5 maps `trials outcome → output[4]: trial_outcome (Per-trial categorical)`; the AI had listed the trials-table columns (trajectory steps 24–25) and verified the counts on a sample experiment (*"hit (66), miss (115), false_alarm (4), correct_reject (24)"*). These are the SDK's canonical change-detection outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Mapped to a fixed integer code (hit = 0, miss = 1, false_alarm = 2, correct_reject = 3) and broadcast as a constant row across all of the trial's time bins, so the per-trial static variable is stored in the same time-varying (5, n_bins) output matrix as the other four outputs. Value names `['hit','miss','false_alarm','correct_reject']`. Full-dataset distribution: hit 0.302, miss 0.572, false_alarm 0.017, correct_reject 0.108.

ii.
```python
oc = 0 if t['hit'] else (1 if t['miss'] else (2 if t['false_alarm'] else 3))
otl.append(np.stack([ib[ti], cb[ti], rb[ti], pb[ti],
                     np.full(len(ti), oc, dtype=np.int64)]).astype(np.int64))
```

iii. Broadcasting satisfies the format guidance *"If at all possible, make it time-varying"* while keeping the value constant within a trial as the task requires ("Static per-trial"). The `else 3` fallback is safe here because, for the retained (non-aborted, non-auto-rewarded) trials, exactly one of the four flags is true — I confirmed this on a sample experiment (0 rows with >1 flag, 0 rows with none).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The handled cases are: (a) NaN samples in running speed / pupil area (blinks, dropped frames) are excluded before interpolation; (b) values outside the tracked range interpolate to NaN and are then assigned the **middle** discretisation bin; (c) an experiment with <2 valid behavioural samples gets an all-NaN trace → all middle-bin labels rather than a crash; (d) bins containing no ophys frame reuse the previous bin's neural values; (e) bins before the first stimulus flash default to image code 0, and unknown image names map to code 0 via `n2i.get(n, 0)`; (f) trials shorter than 2 bins, and experiments with fewer than 2 usable trials, are dropped; (g) pupil extraction in the h5py pre-pass is wrapped in a bare `try/except` so an experiment lacking eye tracking is silently skipped for edge computation. There is **no** try/except around per-experiment loading/processing, so a corrupt file would abort the whole run (none occurred: 0 skipped of 202), and no explicit clipping of trials that run past the end of the ophys recording (I verified this never happens — the recording extends ~10 min past the last trial in every file checked).

ii.
```python
def dig(v, e):
    b = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2)
    b[np.isnan(v)] = (len(e) - 1) // 2      # NaN -> middle bin
    return b.astype(np.int64)
...
ri = interpolate.interp1d(...)(bc) if vm.sum()>=2 else np.full(nb, np.nan)
...
        elif b > 0: out[:, b] = out[:, b-1]   # empty bin -> hold previous
...
            try:
                pa = f['acquisition']['EyeTracking']['pupil_tracking']['area'][::10]
                pa_all.append(np.array(pa, dtype=np.float64))
            except:
                pass
...
if len(ntl) < 2:
    print(f"  [{i+1}/{len(el)}] Exp {eid}: SKIPPED"); skipped += 1; continue
```

iii. CONVERSION_NOTES Step 10 lists the missing-data-related fixes made during review (*"Fixed output dtype from float32 to int64 (required for decoder indexing)"*, *"Fixed pupil area collection in h5py"*) and documents the residual warning class (2,621 trials with all-zero event data) as *"expected for sparse calcium events"* rather than as a defect. Mapping missing behaviour to the median category is the conservative choice that avoids inventing extreme behavioural states.

## 9-a. What are the most time-consuming steps of the code?

i. By the script's own timing prints, loading each NWB through the SDK dominates: `load=3.4–5.5 s` per experiment versus `proc=0.3–0.5 s`, i.e. ~13 min of the 26 min total (1,575.9 s) is `BehaviorOphysExperiment.from_nwb()` I/O plus SDK table construction. Secondary costs are the h5py stats pre-pass (19.2 s for all 202 files), the per-bin Python loop in `resample_session`, and pickling/writing the 3 GB output.

ii.
```python
t0 = time.time(); ed = load_experiment(eid); tl = time.time()
ntl, otl, nn = process_experiment(ed, imn, rbe, pbe); tp = time.time()
...
print(f"  [{i+1}/{len(el)}] Exp {eid}: {nn}n {len(ntl)}t load={tl-t0:.1f}s proc={tp-tl:.1f}s")
```

iii. CONVERSION_NOTES Step 7 contains the projection made from the sample run (*"Load (SDK) 5.5 s avg → 18 min; Process 0.5 s avg → 1.5 min; Stats (h5py) 0.1 s → 20 s; Total ~20 min"*), and Step 6 explains the mitigation chosen: use h5py (not the SDK) wherever only raw arrays are needed, and load/process/release one experiment at a time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python-level loops: (1) `for b in range(nb)` in `resample_session`, which loops over ~37,000 bins per experiment and could be replaced by `np.add.reduceat`/cumulative-sum differencing on the frame axis (the bin boundaries are already computed vectorially by `searchsorted`); (2) `for ct in cts` building the change indicator, which creates a full-length boolean mask per change flash (~330 per session) and could be done with one `searchsorted` pair; (3) `for _, t in vt.iterrows()` over trials, which recomputes a full-length boolean mask `(bc>=start)&(bc<stop)` per trial instead of two `searchsorted` calls; (4) the `iterrows()` comprehensions in `main`/`get_experiment_list` for subject and region lists (`el['mouse_id'].unique()` would do); (5) `[n2i.get(n, 0) for n in cdi['image_name'].values]`, replaceable by `pd.Series.map`. None of these was vectorised, but all are small next to the 3.4–5.5 s SDK load per experiment.

ii.
```python
    for b in range(nb):
        if ri[b] > li[b]: out[:, b] = events[:, li[b]:ri[b]].mean(axis=1)
        elif b > 0: out[:, b] = out[:, b-1]
...
    for ct in cts:
        cb[(bc>=ct)&(bc<ct+0.750)] = 1
...
    for _, t in vt.iterrows():
        tm = (bc>=t['start_time'])&(bc<t['stop_time'])
```

iii. The AI's stated efficiency strategy (CONVERSION_NOTES Step 6) was to attack I/O rather than arithmetic: *"h5py for fast stats collection (0.1 s vs 5 s per experiment with SDK); session-wide resampling then trial extraction (vs per-trial resampling); memory-efficient: load, process, release each experiment."* The measured 0.3–0.5 s processing time per experiment confirms the loops were not the bottleneck, so it did not pursue them.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB file is opened and read **twice**: once in `fast_collect_stats` (h5py: image names, running speed subsample, pupil area subsample) and once in `load_experiment` (SDK: events, trials, stimulus presentations, running speed, eye tracking). Running speed and pupil area are therefore read from disk twice, and the image-name column is read twice (once via h5py, once inside `stimulus_presentations`). Within the main pass nothing else is recomputed — the resampled session arrays are computed once and sliced per trial. The bin-centre mask `(bc>=start)&(bc<stop)` is rebuilt over the whole session grid once per trial, which is repeated work of the same kind.

ii.
```python
imn, rs_all, pa_all = fast_collect_stats(el)      # pass 1: h5py over all files
...
for i, (_, row) in enumerate(el.iterrows()):
    ed = load_experiment(eid)                     # pass 2: SDK over all files
```

iii. This is a deliberate trade-off documented in Step 6: the global percentile edges and the global image-code map must be known **before** any trial is emitted, and the alternative (holding all 202 sessions' behavioural traces in RAM through a single pass, as the reference does) conflicted with the AI's memory-conservation strategy for a 3 GB output. The duplicate pass costs only 19 s of the 1,576 s runtime.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `resample_session`, the running/pupil interpolation, and the image-identity/change vectors are computed for **every** bin between the first trial's start and the last trial's stop, but only bins inside retained trials are kept — I measured 35–97 % bin utilisation across sample experiments (≈25 % of the resampled neural array discarded on average, more in sessions with many aborted trials). (2) `ed['metadata'] = dict(ds.metadata)` is loaded and never used. (3) Full `.copy()` of `trials` and `stimulus_presentations` (21 and ~10 columns) when only ~10 columns are needed. (4) `nn` (neuron count) is computed and returned even for experiments that are subsequently skipped. (5) An empty `np.zeros((0,))` input array is allocated per trial (required by the target format, but 52 k allocations). (6) In `--show-processing` mode a 6×2 figure is rendered per session. None of this is large next to the SDK load time.

ii.
```python
s0, s1 = vt['start_time'].min(), vt['stop_time'].max()
bc = np.arange(s0 + TARGET_BIN_SIZE/2, s1, TARGET_BIN_SIZE)   # spans gaps + aborted trials
nr = resample_session(ev, ots, bc)                            # all bins resampled...
...
    ntl.append(nr[:, ti])                                     # ...only in-trial bins kept
...
        'metadata': dict(ds.metadata),                        # never used
...
    it = [np.zeros((0,), dtype=np.float32) for _ in ntl]
```

iii. The AI justified session-wide resampling as an optimisation over per-trial resampling (CONVERSION_NOTES Step 6) — it avoids recomputing overlapping windows and guarantees one consistent grid — accepting the discarded inter-trial bins as the price. It did not flag the unused `metadata` load or the discarded-bin fraction anywhere in CONVERSION_NOTES.
