# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset (DANDI 000363) is one NWB file per session under `data/sub-<subject_id>/`. The AI finds every session with a single sorted glob over that layout and processes each file exactly once. Unlike the reference, it does **not** use `pynwb`: it opens each file directly with `h5py` and reads the raw HDF5 datasets (`intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `units`, `general/extracellular_ephys/electrodes`). Sessions are processed in parallel with a `multiprocessing.Pool` (16 workers by default), and each worker returns a per-session dict that `main()` stitches together. All 174 files are opened; curation then decides which reach the output (138 sessions).

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
print('found %d NWB files' % len(files))
...
tasks = [(fn, i < nplot) for i, fn in enumerate(files)]
...
with Pool(min(args.nproc, len(tasks))) as pool:
    for i, res in enumerate(pool.imap(process_session, tasks)):
```
```python
with h5py.File(fname, 'r') as f:
    tr = f['intervals/trials']
    start_time = tr['start_time'][:]
    ...
    be = f['acquisition/BehavioralEvents']
    go = be['go_start_times/timestamps'][:]
    ...
    u = f['units']
    classification = decode_array(u['classification'][:])
```
Ragged columns are read as one flat buffer plus an index, as in the reference:
```python
st_index = u['spike_times_index'][:]
st_all = u['spike_times'][:]
starts = np.concatenate([[0], st_index[:-1]])
```

iii. From CONVERSION_NOTES Step 10 Check 3: "(a) Data loading | reference: `preprocessing_utils.loadmat` on per-probe DataJoint `.mat` exports; probes concatenated per session | this conversion: h5py on the DANDI NWB files, which already contain all probes of a session in one `units` table | Equivalent (same underlying data, different distribution format)". Step 2 documents 174 files / 28 `sub-*` folders, cross-checked against the paper's 28 mice and 173 sessions. `h5py` was chosen over `pynwb` for speed (each array is read in bulk, "each NWB file is opened once and all needed arrays are read in bulk"), and sessions are run in parallel because "the machine has 128 CPUs"; the full conversion takes 35 s.

## 1-b. How are the data split into subjects?

i. One mouse per `sub-*` directory/file-name prefix. The AI parses the subject id out of the file name (`sub-440956_ses-...nwb` -> `'440956'`) rather than reading `general/subject/subject_id`; the two are identical because the DANDI folder name is derived from the subject field. At assembly, `subjects` is the sorted set of unique ids and `subject_idx` indexes it per session. Result: 28 subjects, 2-10 sessions each.

ii.
```python
session = {
    'file': os.path.basename(fname),
    'subject': os.path.basename(fname).split('_')[0].replace('sub-', ''),
    'session_id': os.path.basename(fname).split('_')[1].replace('ses-', ''),
```
```python
subjects = sorted({s['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "`general/subject/subject_id` (folder `sub-*`) -> `subjects`, `subject_idx` | one entry per mouse | 28 mice". The count is one of the AI's sanity checks (Step 4: "Mice | 28 | 28 `sub-*` folders | YES"), matching Fig 1J of the data paper. As in the reference, the numeric DANDI id (`440956`) is used rather than the paper's mouse name (`SC015`).

## 1-c. How are the data split into sessions?

i. One NWB file = one session; no splitting or grouping is inferred. The session key is the `ses-<timestamp>` field of the file name, and the sorted glob makes session order deterministic (chronological within subject). Unlike the reference, the AI then *rejects* sessions: 174 files -> 138 sessions, on four grounds — behavioural performance <= 65% (12 sessions), fewer than 50 correct lick-left or lick-right trials (17), no QC-good units (1, the session whose `classification` is NaN), and fewer than 10 usable trials after trial curation (6, all because the side-view video does not cover the analysis window).

ii.
```python
is_stim = np.isfinite(ps_power) & (ps_power > 0)
control = (~is_stim) & (early_lick == 'no early') & (auto_water == 0) & (free_water == 0)
n_hit = int(np.sum(control & (outcome == 'hit')))
n_miss = int(np.sum(control & (outcome == 'miss')))
performance = n_hit / max(n_hit + n_miss, 1)
n_correct_left = int(np.sum(control & (outcome == 'hit') & (instruction == 'left')))
n_correct_right = int(np.sum(control & (outcome == 'hit') & (instruction == 'right')))
if performance <= MIN_PERFORMANCE:            # 0.65
    info['reject'] = 'performance %.3f <= %.2f' % (performance, MIN_PERFORMANCE)
    return None, info
if n_correct_left < MIN_CORRECT_PER_SIDE or n_correct_right < MIN_CORRECT_PER_SIDE:   # 50
    info['reject'] = 'too few correct trials (L %d, R %d)' % (n_correct_left, n_correct_right)
    return None, info
```
```python
if len(good) == 0:
    info['reject'] = 'no QC-good units'
if len(trials) < 10:
    info['reject'] = 'fewer than 10 usable trials'
```

iii. The session filter is taken verbatim from the data paper's methods (`methods.txt`: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each"). CONVERSION_NOTES Step 5 Key Decision 4: "**Session curation** follows the data paper ... plus the session must contain at least one 'good' unit and usable video ... This reproduces the paper's mean performance of 84%." The AI verified that the retained sessions have mean performance 83.9% (range 65.8-98.9%) against the paper's "84% correct rate (range, 65-99%)", and that the one session with `classification = NaN` is the difference between the 174 files and the paper's 173 sessions.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioural trial. The AI asserts that the go-cue event stream has exactly one entry per row, which is what makes the trial<->event mapping unambiguous, and it treats `sample_start_times` as possibly multi-valued per trial (early-lick replay).

ii.
```python
tr = f['intervals/trials']
start_time = tr['start_time'][:]
stop_time = tr['stop_time'][:]
outcome = decode_array(tr['outcome'][:])
...
ntrials_all = len(start_time)

go = be['go_start_times/timestamps'][:]
assert len(go) == ntrials_all, 'go cue count != trial count'
```

iii. CONVERSION_NOTES Step 5: "`go_start_times/timestamps` | alignment event | t = 0 for every trial | one per trial in all 174 sessions". Step 4 records the discrepancy that `sample_start_times` can have up to 12 entries in a trial because "Licking early during the sample/delay epoch triggered a replay of the epoch", resolved by taking the last tone before the go cue — i.e. the trials table, not the event streams, defines trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Five filters, in this order:
1. **Water trials**: `auto_water == 0 & free_water == 0` (reward given independently of the animal's choice).
2. **Recording coverage / stability**: per unit, `units/obs_intervals` is matched to trial start times to build an `observed` matrix, and `units/is_good_trials` (expanded onto the observed trials when its column count is smaller than the trial count) gives a per-unit stability flag. A trial is kept only if at least 90% of the good units are both observed and flagged good on it (`IGT_TRIAL_FRAC = 0.9`).
3. **Video coverage**: a trial is kept only if *every one of its 80 bins* contains at least one side-camera frame.
4. **No-spike trials**: after binning, trials in which no retained unit fires a single spike anywhere in the 4 s window are removed (1 trial dataset-wide).
5. Units still flagged bad on a retained trial are dropped, and the session is rejected if fewer than 10 trials survive.

Early-lick, `ignore`/`miss` and photostimulation trials are deliberately **kept**. Totals: of the 75,711 trials in the 138 retained sessions, 2,763 are dropped as water trials, 1,465 by the `obs_intervals`/`is_good_trials` rule, 2,574 by the video rule, 1 for having no spikes; 69,074 remain.

ii.
```python
oi_index = u['obs_intervals_index'][:]
oi_all = u['obs_intervals'][:]
oi_starts = np.concatenate([[0], oi_index[:-1]])
observed = np.zeros((len(good), ntrials_all), dtype=bool)
for i, ui in enumerate(good):
    o = oi_all[oi_starts[ui]:oi_index[ui], 0]
    j = np.searchsorted(start_time, o)
    j = np.clip(j, 0, ntrials_all - 1)
    ok = np.isclose(start_time[j], o, atol=1e-6)
    observed[i, j[ok]] = True
...
usable = observed & igt
trial_stable = usable.mean(axis=0) >= IGT_TRIAL_FRAC
keep = (auto_water == 0) & (free_water == 0) & trial_stable
```
```python
edges_abs = go[:, None] + BIN_EDGES_REL[None, :]        # (ntrials, NBINS+1)
vidx = np.searchsorted(vts, edges_abs)
frames_per_bin = np.diff(vidx, axis=1)
video_ok = np.all(frames_per_bin > 0, axis=1)
keep = keep & video_ok
trials = np.where(keep)[0]
```
```python
nonempty = rates.sum(axis=(0, 2)) > 0
if not nonempty.all():
    rates = rates[:, nonempty, :]
    trials = trials[nonempty]
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "auto-water and free-water trials are removed (reward independent of the animal's choice; removed by the reference `get_regular_trial_mask` as well). Early-lick, 'ignore'/'miss' and photostimulation trials are **kept**, deviating from the reference analyses, because the decoder task explicitly requires early lick and outcome as decoder outputs and photostimulation as a decoder input." Key Decision 3 justifies `is_good_trials` ("per-probe manual annotation of stable trials ... Affects only 4/174 sessions"), and Key Decision 6 the video rule: "a trial is kept only if every one of its 80 bins contains at least one video frame; otherwise the tongue output could not be distinguished between 'tongue not visible' and 'no video recorded'. This removes ~5% of trials overall and effectively removes 6 sessions whose video stops at the go cue (the data paper also screened sessions for video artifacts)." The `obs_intervals` rule was added in response to a verification warning ("Session 0, trial N: all neural data is zero ... in 9/174 sessions the probes were recorded only during part of the behavioural session"), and the no-spike rule to kill the last remaining all-zero trial.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (with `units/spike_times_index` for the ragged layout), restricted to units with `units/classification == 'good'`. The other input is `acquisition/BehavioralEvents/go_start_times/timestamps`, which places the bin edges. Both are in session-absolute seconds.

ii.
```python
u = f['units']
classification = decode_array(u['classification'][:])
good = np.where(classification == 'good')[0]
...
st_index = u['spike_times_index'][:]
st_all = u['spike_times'][:]
starts = np.concatenate([[0], st_index[:-1]])
for i, ui in enumerate(good):
    sp = st_all[starts[ui]:st_index[ui]]
```

iii. CONVERSION_NOTES Step 5 mapping table: "`units/spike_times` (+`spike_times_index`), `units/classification=='good'` -> `neural[session][trial]` (n_neurons, 80)". Spike times are the only neural representation in the file. The AI cross-checked its own rates against the file's `units/avg_firing_rate` (trajectory step 84: "my mean rate 16.4 Hz matches the file's avg_firing_rate mean 16.19 Hz — correct").

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz with no smoothing, normalisation or baseline subtraction. For each good unit, the absolute bin edges of all retained trials are flattened into one array, a single `np.searchsorted` gives the running spike count at every edge, and `np.diff` turns those into per-bin counts; dividing by the 50 ms bin width gives Hz. Stored as `float32`. This is exactly the reference's estimator.

ii.
```python
edges_kept = edges_abs[trials]                   # (ntrials_kept, NBINS+1)
flat_edges = edges_kept.ravel()
rates = np.zeros((len(good), nkept, NBINS), dtype=np.float32)
for i, ui in enumerate(good):
    sp = st_all[starts[ui]:st_index[ui]]
    # spike_times are stored sorted per unit; make sure
    counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
    rates[i] = np.diff(counts, axis=1).astype(np.float32)
rates /= BIN_SIZE                                  # Hz
```

iii. CONVERSION_NOTES Step 5 Key Decision 9: "**Firing rates in Hz** (counts / bin width), matching the reference `sliding_histogram(rate=True)`." Step 10 Check 3(d): reference "`sliding_histogram(bin_width=0.04, stride=0.0034, rate=True)` -> Hz" vs "80 non-overlapping 50-ms bins, counts / 0.05 s -> Hz | Same units and estimator; bin width/stride prescribed by the decoder task". Step 6 notes the vectorisation: "for each unit a single `np.searchsorted` of all 81 x n_trials bin edges into that unit's spike-time array, then `np.diff` ... This replaced an O(n_units x n_trials x n_bins) loop." Independently re-verified in Step 10 Check 2 with `np.allclose` on 192 randomly chosen (neuron, bin) spot checks plus whole-trial spike totals.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept — the NWB encoding of the white paper's 15-metric, region-specific logistic-regression classifier. No thresholds on individual quality metrics and no firing-rate threshold. On top of that, units flagged bad by `units/is_good_trials` on any retained trial are dropped (8 units dataset-wide, 55,437 -> 55,429). A session with zero good units is rejected. Over all 174 files this label selects 69,453 / 272,227 units; 55,429 of those fall in the 138 retained sessions.

ii.
```python
classification = decode_array(u['classification'][:])
good = np.where(classification == 'good')[0]
info['n_units_all'] = len(classification)
info['n_units_good'] = len(good)
if len(good) == 0:
    info['reject'] = 'no QC-good units'
    return None, info
```
```python
unit_ok = usable[:, trials].all(axis=1)
if unit_ok.sum() == 0:
    info['reject'] = 'no units stable over the retained trials'
    return None, info
good = good[unit_ok]
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "this is the NWB encoding of the reference pipeline's classifier-based QC (white paper, 15 metrics, region-specific logistic regression). Gives 69,453 units vs the 69,943 reported (99.3%). Units without this label also have an empty `anno_name`, so the reference's 'unit must have histology' criterion is automatically satisfied." Key Decision 2: "**No additional firing-rate threshold**: the 2-Hz threshold in the method paper applies only to their video->firing-rate regression analyses, not to the dataset itself. Keeping all good units maximises the information available to the decoder." The AI also validated its region assignment against the paper's per-area good-unit counts (ALM 8,464 vs 8,717; thalamus 13,514 vs 12,808; etc.).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset (`go_start_times`). Because spike times, event times and camera timestamps all live on the same session-absolute clock, no resampling or offset correction is needed: the fixed relative edge grid is added to each trial's go-cue time to give absolute bin edges, and spikes are binned against those. The same `edges_abs` array is reused for the video-coverage test, so the neural and tongue grids are identical by construction.

ii.
```python
ALIGN_EVENT = 'go cue onset'
T_START = -2.5           # s relative to go cue
T_END = 1.5              # s relative to go cue
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
...
go = be['go_start_times/timestamps'][:]
edges_abs = go[:, None] + BIN_EDGES_REL[None, :]        # (ntrials, NBINS+1)
...
counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
```

iii. CONVERSION_NOTES Step 10 Check 3(c): reference "go cue (`task_cue_time[0]`); spike times in the export are already go-cue relative" vs "go cue (`BehavioralEvents/go_start_times`); spike times, photostim events and video frames are all converted to go-cue-relative time | Same". The alignment was validated three ways: raster-vs-binned-rate overlays in `processing_*.png`, `np.allclose` spot checks recomputed from the raw file, and a replication of the paper's ALM choice-decoding time course (0.78-0.90 in the late delay, 0.88-0.98 in the response epoch, vs ~0.9 / ~0.95 in the paper), which the AI treats as proof that "the neural data, the trial alignment and the behavioural labels in the converted dataset are correct."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s around the go cue, identical for every trial and session (`time_bin_size = 50.0` ms in the metadata, `off_start = -2.5`, `off_end = 1.5`). Spikes are binned once at that resolution directly from spike times — there is no intermediate representation and therefore no rebinning. The relative edge/centre grid is built once at module level. This deviates from the reference pipeline's 40 ms width / 3.4 ms stride sliding histogram, because the decoder task prescribes 50 ms bins.

ii.
```python
T_START = -2.5           # s relative to go cue
T_END = 1.5              # s relative to go cue
BIN_SIZE = 0.05          # s
NBINS = int(round((T_END - T_START) / BIN_SIZE))   # 80
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```
```python
'time_bin_size': BIN_SIZE * 1000.0,
'temporal_alignment_event': ALIGN_EVENT,
'off_start': T_START,
'off_end': T_END,
```

iii. CONVERSION_NOTES Step 4: "Neural binning | 40 ms / 3.4 ms stride | `sliding_histogram(bw=0.04, stride=0.0034)` | The decoder task **requires 50-ms bins**, so I use 50-ms non-overlapping bins (rates in Hz, same as reference)". This is listed as one of the two sanctioned deviations from the reference processing (the other being the retention of early-lick/ignore/photostim trials).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `acquisition/BehavioralEvents/sample_start_times/timestamps` (the sample-epoch/tone onsets of the session) together with the trial's go-cue time. Because an early lick replays the sample epoch, a trial can contain several sample onsets; the AI takes the **last one before the go cue**.

ii.
```python
sample_on = be['sample_start_times/timestamps'][:]
...
si = np.searchsorted(sample_on, go) - 1
tone_abs = np.where(si >= 0, sample_on[np.clip(si, 0, len(sample_on) - 1)], np.nan)
tone_rel = tone_abs - go                          # negative, ~ -1.85 s
assert np.all(np.isfinite(tone_rel[trials])), 'missing tone onset'
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "up to 12 `sample_start_times` in one trial (early-lick replay) ... Use the **last** sample (tone) onset before the go cue, i.e. the tone that actually instructed the executed trial". Sanity-checked against the task structure: "Tone onset median -1.85 s as expected (0.65 s sample + 1.2 s delay); early-lick trials have earlier/replayed tones", and "Every trial has a sample (tone) onset before the go cue".

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The value of a bin is the signed time of its centre relative to the tone, i.e. the bin centre (which is go-cue-relative) minus the tone's go-cue-relative time. It is left continuous and signed, so bins before the tone are negative (min ~-0.6 s, the first bin of a standard trial) and replay trials reach up to +11.9 s. No clipping or normalisation. Stored as `float32`.

ii.
```python
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]
...
input_list.append(np.stack([time_from_tone[k].astype(np.float32),
                            stim_input[k]]).astype(np.float32))
```

iii. Trajectory step 78: "'time_from_tone_onset': seconds since the last sample/tone onset before go cue (continuous). Before tone onset -> negative values (natural continuation) — spec says 'Time from tone onset in seconds (continuous, time-varying)'. I'll use signed time (t_bin - tone_onset), giving negative before onset." Trajectory step 84 considered and rejected capping the long values: "It's a legitimate continuous input; values up to ~7 s occur in replay trials. Keep as is". Verified in Step 10 Check 2 by recomputing `bin_centre - (last sample_start_time before the go cue - go cue)` from the raw file with `np.allclose`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined on the same grid: the values are the go-cue-relative bin centres `BIN_CENTERS_REL` (the centres of the very bins used for the firing rates) shifted by the per-trial tone offset. Bin *k* of the input therefore covers exactly the interval of bin *k* of `neural`, and no interpolation or separate time base exists.

ii.
```python
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
...
edges_abs = go[:, None] + BIN_EDGES_REL[None, :]   # used for the spike binning
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]
```

iii. Implicit in the design (one grid for every stream, Step 10 Check 3(c): "spike times, photostim events and video frames are all converted to go-cue-relative time"). The `--show-processing` figures overlay the input trace, the tone-onset marker and the spike raster on one axis so that the alignment is visually verifiable (panel 4 of `processing_<session>.png`).

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Primarily the event streams `acquisition/BehavioralEvents/photostim_start_times` and `photostim_stop_times`, each event being assigned to the trial whose `start_time` last precedes it. Where a trial is marked as stimulated (`photostim_power > 0`) but has no matching event, the AI falls back to the trials-table columns `photostim_onset` and `photostim_duration` (strings, measured from trial start) — which is the reference's source. `go` is used to re-express everything relative to the go cue.

ii.
```python
ps_onset = np.array([to_float(x) for x in tr['photostim_onset'][:]])
ps_dur = np.array([to_float(x) for x in tr['photostim_duration'][:]])
ps_power = np.array([to_float(x) for x in tr['photostim_power'][:]])
is_stim = np.isfinite(ps_power) & (ps_power > 0)
...
if 'photostim_start_times' in be:
    pstart = be['photostim_start_times/timestamps'][:]
    pstop = be['photostim_stop_times/timestamps'][:]
stim_rel = np.full((ntrials_all, 2), np.nan)
if len(pstart):
    pt = np.searchsorted(start_time, pstart, side='right') - 1
    ok = (pt >= 0) & (pt < ntrials_all)
    stim_rel[pt[ok], 0] = pstart[ok] - go[pt[ok]]
    stim_rel[pt[ok], 1] = pstop[ok] - go[pt[ok]]
# fall back to the trials table where the event stream is missing
miss = is_stim & ~np.isfinite(stim_rel[:, 0])
if miss.any():
    stim_rel[miss, 0] = start_time[miss] + ps_onset[miss] - go[miss]
    stim_rel[miss, 1] = stim_rel[miss, 0] + ps_dur[miss]
```

iii. CONVERSION_NOTES Step 5 mapping table: "`photostim_start_times` / `photostim_stop_times` (cross-checked against `intervals/trials/photostim_onset`,`photostim_duration`) -> `input[1]` = `photostim_on`", reference equivalent "`sess_dict['stimulation']` columns 2-3". Step 10 Check 2 validates the two sources against each other: "**Input** `photostim_on`: from `BehavioralEvents/photostim_start|stop_times`, and independently from `trials.photostim_onset/duration` (relative to trial start) | 12 photostim trials in 3 sessions | PASS (always exactly 10 bins = 0.5 s, always ending at or before the go cue)", consistent with the methods' "late delay (last 0.5 s), always ends before go cue".

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1 `float32`) time series, not a per-trial flag: a bin is 1 if its centre lies within `[stim_start, stim_stop]`. Trials with no stimulation keep NaN bounds and are left all-zero. Across the dataset the input is on in 2.6% of all bins (0.5 s of the 4 s window on ~21% of trials).

ii.
```python
stim_input = np.zeros((nkept, NBINS), dtype=np.float32)
...
for k, ti in enumerate(trials):
    if np.isfinite(stim_rel[ti, 0]):
        stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                         (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping: "1 if the bin centre lies in [stim start, stim stop], else 0 | binary time series", which follows the decoder-task requirement "Whether **photostimulation** is on at every time point (discrete, time-varying)". Step 12: "`photostim_on` is non-zero only between -2.18 s and -0.02 s relative to the go cue, i.e. photoinhibition always ends before the go cue, as stated in the methods."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset/offset are converted to go-cue-relative seconds (`pstart - go[trial]`), then compared against the same `BIN_CENTERS_REL` grid used for the firing rates, so no separate time base or interpolation is involved.

ii.
```python
stim_rel[pt[ok], 0] = pstart[ok] - go[pt[ok]]
stim_rel[pt[ok], 1] = pstop[ok] - go[pt[ok]]
...
stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                 (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
```

iii. Same single-grid rationale as 2-d/3-c. The `--show-processing` plot draws the raw `axvspan` of the stimulation window behind the binary input trace as an explicit alignment check, and the AI verified that the resulting window is always exactly 10 bins ending at or before the go cue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from the measured licks, not from the trial labels: `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times`, restricted to the response window `[go_cue, trials/stop_time]`. The direction of the *first* lick in that window is the choice; if there is no lick in either channel, the choice is "no lick". `trials/outcome` x `trials/trial_instruction` is used only as a cross-check (the reference derives choice from those two columns instead).

ii.
```python
left_lick = be['left_lick_times/timestamps'][:]
right_lick = be['right_lick_times/timestamps'][:]
...
choice = np.full(ntrials_all, 2, dtype=np.int64)    # 2 = no lick
for ti in trials:
    lo, hi = go[ti], stop_time[ti]
    l0 = left_lick[np.searchsorted(left_lick, lo)] if np.searchsorted(left_lick, lo) < len(left_lick) else np.inf
    r0 = right_lick[np.searchsorted(right_lick, lo)] if np.searchsorted(right_lick, lo) < len(right_lick) else np.inf
    l0 = l0 if l0 <= hi else np.inf
    r0 = r0 if r0 <= hi else np.inf
    if np.isinf(l0) and np.isinf(r0):
        choice[ti] = 2
    elif l0 <= r0:
        choice[ti] = 0
    else:
        choice[ti] = 1
# sanity: agreement with outcome x instruction
expected = np.where(outcome == 'hit', instruction,
                    np.where(instruction == 'left', 'right', 'left'))
expected = np.where(outcome == 'ignore', 'none', expected)
ch_str = np.array(['left', 'right', 'none'])[choice]
info['choice_agreement'] = float(np.mean(ch_str[trials] == expected[trials]))
```

iii. CONVERSION_NOTES Step 5 mapping: "`{left,right}_lick_times` -> `output[0]` = `choice` | first lick after the go cue (within the trial) -> 0 left, 1 right, 2 no lick | reference `behavior_lick_directions`/`lick_times` | agrees with `outcome` x `trial_instruction` in 99.8% of trials (sanity check)". Trajectory step 53/78: "Choice inferred from licks matches outcome/instruction perfectly" on the first session examined, and "Choice from first lick within [go, trial stop] agrees with outcome/instruction 99.8% — excellent". The lick-time route was preferred because the reference code's behavioural variables are the lick directions themselves; the derived agreement metric is stored per session in `converted_data.pkl.info.json`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0` = left, `1` = right, `2` = no lick, and written into row 0 of the per-trial `(4, 80)` output array, repeated across all 80 bins (a per-trial value broadcast over time). `output_values[0] = ['left', 'right', 'no lick']`. Dataset distribution: left 0.453 / right 0.443 / no lick 0.104.

ii.
```python
out = np.empty((4, NBINS), dtype=np.int64)
out[0] = choice[ti]
out[1] = outcome_code[ti]
out[2] = early_code[ti]
out[3] = tongue_class[k]
output_list.append(out)
```
```python
'output_values': [
    ['left', 'right', 'no lick'],
    ...
```

iii. The code assignment follows the decoder-task spec ("Lick direction **choice** (left, right, no lick, per-trial)"). CONVERSION_NOTES Step 5 Key Decision 8: "**Outputs are time-varying** (shape (4, 80)): choice/outcome/early lick are per-trial values broadcast across time (as allowed by the format), tongue position is genuinely time-varying", which also keeps all four outputs in one array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `intervals/trials/outcome` column, which already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'` (decoded from bytes).

ii.
```python
outcome = decode_array(tr['outcome'][:])
...
outcome_code = np.select([outcome == 'ignore', outcome == 'miss', outcome == 'hit'],
                         [0, 1, 2], default=0).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping: "`intervals/trials/outcome` -> `output[1]` = `outcome` | 'ignore'->0, 'miss'->1, 'hit'->2 | reference `behavior_report` (-1/0/1 in the .mat export) | same three categories as the reference `correctness` variable". No derivation needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to `0` ignore / `1` miss / `2` hit with `np.select`, written to row 1 of the output array and repeated across all 80 bins. Dataset distribution: ignore 0.106 / miss 0.154 / hit 0.741, i.e. hit/(hit+miss) = 82.8%.

ii.
```python
outcome_code = np.select([outcome == 'ignore', outcome == 'miss', outcome == 'hit'],
                         [0, 1, 2], default=0).astype(np.int64)
...
out[1] = outcome_code[ti]
```
```python
['ignore', 'miss', 'hit'],
```

iii. Codes follow the decoder-task spec ("**Outcome** (ignore, miss, hit, per-trial)"). Validated in Step 9: "hit 0.741, miss 0.154, ignore 0.106 (hit/(hit+miss) = 82.8%)" against the paper's 84% correct rate, and re-derived from the raw file in Step 10 Check 2.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Directly from the `intervals/trials/early_lick` column, whose values are the strings `'no early'` and `'early'`.

ii.
```python
early_lick = decode_array(tr['early_lick'][:])
...
early_code = (early_lick == 'early').astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping: "`intervals/trials/early_lick` -> `output[2]` = `early_lick` | 'no early'->0, 'early'->1 | reference `behavior_early_report`". The same column is used (inverted) in the session-performance computation, per the methods' definition of performance on non-early control trials.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A boolean comparison against `'early'` cast to int (`0` no / `1` yes), written to row 2 and repeated across all 80 bins. Dataset distribution 0.880 / 0.120, matching the 11.4% of raw trials flagged early. Note that these trials are kept rather than excluded (the reference analyses exclude them), because early lick is a required decoder output; the lick that sets the flag falls inside the -2.5 s window.

ii.
```python
early_code = (early_lick == 'early').astype(np.int64)
...
out[2] = early_code[ti]
```
```python
['no', 'yes'],
```

iii. CONVERSION_NOTES Step 4/Step 5 Key Decision 5: early-lick trials "are **kept**, deviating from the reference analyses, because the decoder task explicitly requires early lick and outcome as decoder outputs". Step 9 checks the resulting rate against the raw data ("early lick | 11.4% of raw trials | 11.4% | 0.120 | YES"), and Step 10 Check 5 notes that one retained session has no early-lick trial at all, judged acceptable because the class is present in other sessions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: `data[:, 1]` is the DeepLabCut tongue *y* coordinate, `data[:, 2]` the tracking likelihood, with matching `timestamps` (~300 Hz side-view camera). The likelihood column decides whether the tongue is visible in a frame; `y` provides the value.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vts = tongue['timestamps'][:]
vdata = tongue['data'][:]
vy = vdata[:, 1]
vlik = vdata[:, 2]
visible_all = vlik > LIKELIHOOD_THRESH      # 0.9
if visible_all.sum() < 100:
    info['reject'] = 'tongue never tracked in this session'
    return None, info
```

iii. CONVERSION_NOTES Step 5 mapping: "`Camera0_side_TongueTracking` (`data[:,1]`=y, `data[:,2]`=likelihood) -> `output[3]` = `tongue_y`". The layout was established during exploration (trajectory step 17/20: "BehavioralTimeSeries with Camera0_side Jaw/Nose/Tongue tracking (data shape (N,3) — presumably x, y, likelihood)", then confirmed) and the reference code's marker handling (`align_markers.py`) was consulted for the likelihood treatment. This is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) Frames with `likelihood <= 0.9` are treated as "tongue not visible" and excluded — the tracker still reports a position when the tongue is retracted. (2) Within each 50 ms bin of the trial window, the *mean* y of the visible frames is taken (`np.bincount` of sums / counts). (3) That mean is discretised against the session's percentiles (see 8-c); bins with no visible frame become class 3. Sessions in which the tongue is essentially never tracked are rejected.

ii.
```python
b = np.clip(((ft - T_START) / BIN_SIZE).astype(int), 0, NBINS - 1)
cnt = np.bincount(b[fvis], minlength=NBINS)
ysum = np.bincount(b[fvis], weights=fy[fvis], minlength=NBINS)
has = cnt > 0
ymean = np.zeros(NBINS)
ymean[has] = ysum[has] / cnt[has]
cls = np.full(NBINS, 3, dtype=np.int64)
cls[has] = np.digitize(ymean[has], [y_p40, y_p60])   # 0,1,2
```

iii. Trajectory step 79: "Tongue likelihood is essentially binary (0.5 vs 0.9 thresholds equivalent); I'll use >0.9" (step 53 measured "89% <0.1, 10.5% >0.9"), and 0.9 is the conventional DeepLabCut cut-off. Trajectory step 78: "use frames within the bin; if any visible frame, use the median y of visible frames; else class 3 (not visible)" — the implementation and all later documentation/validation use the **mean**, so the word "median" in the Step 5 mapping table is stale. CONVERSION_NOTES Step 10 Check 2 states the implemented rule and verifies it: "**Output** `tongue_y`: per-bin mean y of frames with likelihood > 0.9 discretised at the session's 40th/60th percentiles | 4 sessions x 3 trials (960 bins) | PASS".

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, two class edges are the 40th and 60th percentiles of the y values of **all visible frames of that session** (not of the binned means, and not restricted to the trial windows). Each bin's mean y is then `np.digitize`d against those edges into 0 (< 40th), 1 (40th-60th), 2 (> 60th); bins with no visible frame get 3 ("not visible"). The two percentile values are stored per session in the metadata. Resulting dataset distribution: 0.127 / 0.069 / 0.071 / 0.734 — i.e. among the 26.6% of bins that are visible the split is roughly 48 / 26 / 27 rather than the nominal 40 / 20 / 40.

ii.
```python
visible_all = vlik > LIKELIHOOD_THRESH
...
y_p40, y_p60 = np.percentile(vy[visible_all], [40, 60])
...
cls[has] = np.digitize(ymean[has], [y_p40, y_p60])   # 0,1,2
```
```python
'output_values': [..., ['<40th pctile', '40th-60th pctile', '>60th pctile', 'not visible']],
...
'tongue_y_percentiles': s['tongue_pctl'],
```

iii. Trajectory step 78: "Session percentile definition: '< 40th percentile of y-position over the session' — computed over the session (visible frames presumably). I'll compute percentiles over visible frames across the whole session", i.e. a literal reading of the decoder-task text, with the visibility restriction added because the tracker emits a position even when the tongue is hidden. The fourth class is the task spec's "3: not visible". The stored percentiles were re-derived from the raw file in Step 10 Check 2 ("stored tongue percentiles equal recomputed ones | PASS").

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and events, so for each trial the frame range is found with `np.searchsorted(vts, [go + T_START, go + T_END])` and each frame is assigned to a bin by its offset from `go + T_START` — the same go-cue-anchored 50 ms grid used for the firing rates, so bin *k* covers the same interval in both streams. This is the only genuinely time-varying output. The AI additionally requires (in trial curation) that every bin contain at least one frame, so partially covered windows never appear.

ii.
```python
for k, ti in enumerate(trials):
    i0, i1 = np.searchsorted(vts, [go[ti] + T_START, go[ti] + T_END])
    if i1 <= i0:
        continue
    ft = vts[i0:i1] - go[ti]
    fy = vy[i0:i1]
    fvis = visible_all[i0:i1]
    ...
    b = np.clip(((ft - T_START) / BIN_SIZE).astype(int), 0, NBINS - 1)
```
```python
vidx = np.searchsorted(vts, edges_abs)   # same edges used for the spikes
frames_per_bin = np.diff(vidx, axis=1)
video_ok = np.all(frames_per_bin > 0, axis=1)
```

iii. Step 10 Check 3(c) lists video frames among the streams "converted to go-cue-relative time", and Step 12 gives the behavioural check that confirms the alignment: "tongue class 3 ('not visible') in 92.5% of pre-go-cue bins and 41.4% of post-go-cue bins; classes 0/1/2 rise immediately after the go cue — exactly the expected licking behaviour." Panel 5 of `processing_<session>.png` overlays the raw tongue trace, the percentile lines and the go cue for visual confirmation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six distinct cases, each handled explicitly:
- **Session never quality-controlled** (`classification` is NaN for all units): `decode_array` stringifies it, no unit matches `'good'`, and the session is rejected — this is the single file separating the 174 files from the paper's 173 sessions.
- **Probes recorded for only part of a session** (9 files): trials not covered by `units/obs_intervals` are identified by matching interval starts to trial starts and removed, rather than being emitted as 4 s of silence.
- **`units/is_good_trials` with fewer columns than trials** (same 9 files, originally a crash): the array is expanded onto each unit's observed trials; if the shapes still do not line up the flag is ignored and that is recorded in `info['igt_used']`.
- **Trials with no spikes at all** (recording stopped mid-trial): removed after binning (1 trial dataset-wide).
- **Missing / non-covering video**: trials whose window is not fully covered by frames are removed; sessions with essentially no tongue tracking are rejected.
- **Frames with no tracked tongue**: excluded from the bin mean; a bin with no visible frame becomes the explicit `'not visible'` class rather than being imputed.
- Robustness helpers: `dec`/`decode_array` handle bytes-vs-str columns, `to_float` returns NaN for `'N/A'` photostim strings, and the whole per-session body is wrapped in a `try/except` that records the traceback in the `.info.json` instead of aborting the run.

ii.
```python
def to_float(x):
    try:
        return float(dec(x))
    except Exception:
        return np.nan
```
```python
igt_raw = u['is_good_trials'][:][good]
igt = np.ones((len(good), ntrials_all), dtype=bool)
if igt_raw.shape[1] == ntrials_all:
    igt = igt_raw
    info['igt_used'] = 'direct'
else:
    ok_expand = True
    for i in range(len(good)):
        obs_idx = np.where(observed[i])[0]
        if len(obs_idx) == igt_raw.shape[1]:
            igt[i, obs_idx] = igt_raw[i]
        else:
            ok_expand = False
    info['igt_used'] = 'expanded' if ok_expand else 'ignored'
```
```python
nonempty = rates.sum(axis=(0, 2)) > 0
if not nonempty.all():
    info['n_trials_drop_nospikes'] = int((~nonempty).sum())
    ...
```
```python
    except Exception as e:
        info['error'] = '%s: %s' % (type(e).__name__, e)
        info['traceback'] = traceback.format_exc()
        return None, info
```

iii. CONVERSION_NOTES Step 6 "Issues found during development" and Step 10 Check 1/Check 5 document each case and the evidence that drove it — all of them were found either by a crash or by a `verify_data_format` warning ("Session 0, trial N: all neural data is zero"), and the final state is "Data format is valid, no errors or warnings." The guiding principle stated in Step 5 Key Decisions 6-7 is that data that was never recorded must be removed rather than emitted as zeros/"not visible", whereas a measurement that legitimately has no value (retracted tongue) gets its own category. One knowingly accepted exception is documented: on `miss` trials, which end ~0.8 s after the go cue, the tail bins contain no recorded spikes and stay at 0 Hz, because "'miss' is one of the output classes".

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion takes 35.3 s wall clock: 18.7 s for all 174 sessions (16 worker processes, ~0.4-1.9 s per session single-threaded) and 16.4 s to pickle the 9.2 GB result. Within a session the cost is dominated by bulk HDF5 reads — the `spike_times` buffer (up to ~11.5 M doubles) and the tongue tracking array (~680k x 3) — followed by the per-unit `searchsorted` over 81 x n_trials edges; per-session timings for the neural / input / output blocks are recorded in `info['timing']` and written to the `.info.json`. The per-trial loops (choice, photostim, tongue) are not a measurable share.

ii.
```python
t0 = time.time()
info = {'file': os.path.basename(fname), 'timing': {}}
...
info['timing']['neural'] = time.time() - t1
...
info['timing']['input'] = time.time() - t1
...
info['timing']['output'] = time.time() - t1
...
info['timing']['total'] = time.time() - t0
```
```python
with Pool(min(args.nproc, len(tasks))) as pool:
    for i, res in enumerate(pool.imap(process_session, tasks)):
```

iii. CONVERSION_NOTES Step 6 "Efficiency" and Step 7 "Run Time Estimates": the work is I/O plus one binary search per bin edge per unit, both scaling with the data actually needed; sessions are processed in parallel because "the machine has 128 CPUs". The 35 s total is far inside the instructions' 15-minute budget, so no further optimisation was pursued.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The hot path is vectorised: the spike binning loops over units only (one `searchsorted` covering all trials and bins at once, "This replaced an O(n_units x n_trials x n_bins) loop"), and the video-coverage test is a single `searchsorted` of the whole edge matrix. Five Python loops remain, all cheap:
- the per-unit `obs_intervals` loop building `observed` (could be one flat `searchsorted` + `bincount`);
- the per-unit `is_good_trials` expansion loop;
- the per-trial `choice` loop (fully vectorisable: two `searchsorted` calls on `go` and a comparison against `stop_time`);
- the per-trial photostim loop (vectorisable exactly as the reference does it, one broadcast comparison of `BIN_CENTERS_REL` against `stim_on`/`stim_off` columns);
- the per-trial tongue loop (vectorisable with a global bin index and one `bincount`, as the reference's `_bin_mean` does per trial).
The final `for k, ti in enumerate(trials)` assembly loop is inherent to the required list-of-arrays output format.

ii.
```python
for i, ui in enumerate(good):
    o = oi_all[oi_starts[ui]:oi_index[ui], 0]
    j = np.searchsorted(start_time, o)
```
```python
for ti in trials:
    lo, hi = go[ti], stop_time[ti]
    l0 = left_lick[np.searchsorted(left_lick, lo)] if ...
```
```python
for k, ti in enumerate(trials):
    if np.isfinite(stim_rel[ti, 0]):
        stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                         (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
```

iii. CONVERSION_NOTES Step 6 "Efficiency" lists what was vectorised ("Spike binning is vectorised ... Video frames per bin are obtained with one `np.searchsorted` ... Tongue classes per trial are computed with `np.bincount` over bin indices instead of a per-bin loop") and treats the remaining per-trial loops as not worth optimising given the 35 s runtime; the per-unit spike loop cannot be collapsed further because each unit has a different number of spikes.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened once and the large arrays are read once, so nothing expensive is recomputed. Four small redundancies exist:
- in the `choice` loop, `np.searchsorted(left_lick, lo)` and `np.searchsorted(right_lick, lo)` are each evaluated **twice** per trial (once in the condition, once in the value of the conditional expression);
- `np.searchsorted` on the camera timestamps is done once for the whole edge matrix (`vidx`, for `video_ok`) and then again per trial inside the tongue loop;
- `frames_per_bin` is computed for *all* trials although only the all-bins-covered reduction is used;
- `make_processing_plot` (only under `--show-processing`) re-reads `spike_times`, `spike_times_index` and `classification` and recomputes `good_idx`, duplicating work already done in `process_session`.

ii.
```python
l0 = left_lick[np.searchsorted(left_lick, lo)] if np.searchsorted(left_lick, lo) < len(left_lick) else np.inf
r0 = right_lick[np.searchsorted(right_lick, lo)] if np.searchsorted(right_lick, lo) < len(right_lick) else np.inf
```
```python
vidx = np.searchsorted(vts, edges_abs)
frames_per_bin = np.diff(vidx, axis=1)
video_ok = np.all(frames_per_bin > 0, axis=1)
...
    i0, i1 = np.searchsorted(vts, [go[ti] + T_START, go[ti] + T_END])
```
```python
def make_processing_plot(...):
    st_index = u['spike_times_index'][:]
    st_all = u['spike_times'][:]
    starts = np.concatenate([[0], st_index[:-1]])
    classification = decode_array(u['classification'][:])
    good_idx = np.where(classification == 'good')[0]
```

iii. Not discussed in CONVERSION_NOTES beyond the general claim that "Each NWB file is opened once and all needed arrays are read in bulk"; the duplicated `searchsorted` calls appear to be unintentional. Because the conversion is a single pass and per-session percentiles are computed inside that pass, no second pass over the data is needed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little of the output is wasted, but several computations do not feed the decoder:
- the elaborate CCF keyword -> coarse-region mapping with hemisphere assignment (14 groups x 2 hemispheres, from electrode `x`/`z` coordinates) is required to fill `brain_regions`/`brain_region_idx`, but `train_decoder.py` only prints those fields — they never enter the model;
- diagnostics: `choice_agreement`, `mean_rate`, `performance`, per-region `Counter` summaries, and the whole summary block at the end of `main()`;
- per-session bookkeeping stored in the pickle's metadata that the decoder never reads: `trial_indices` and `unit_indices` (full integer lists for every session), `tone_rel`, `stim_rel`, `bin_centers_s`, plus the separate `.info.json`;
- `anno` is carried in the per-session dict but only used to build `regions`;
- `frames_per_bin` is materialised for all trials and bins to produce one boolean per trial;
- `ps_power`/`ps_onset`/`ps_dur` are parsed for every trial of every session, including the 29 sessions that are rejected before any of it is used;
- the outputs are stored as `int64` although they take values 0-3 (`int8` would be 8x smaller), which inflates the pickle.

ii.
```python
regions = np.array(['%s %s' % (h, ccf_to_region(a, p))
                    for h, a, p in zip(hemi, anno, ap_um)])
```
```python
info['choice_agreement'] = float(np.mean(ch_str[trials] == expected[trials]))
```
```python
'trial_indices': [int(x) for x in s['trial_indices']],
'unit_indices': [int(x) for x in s['unit_indices']],
'mean_firing_rate_hz': s['mean_rate']
```
```python
out = np.empty((4, NBINS), dtype=np.int64)
```

iii. The region mapping is justified by the target format ("`brain_regions`: List of names of all brain regions recorded from") and was validated against the paper's per-area unit counts (Step 4). The diagnostics and the stored indices are justified as the mechanism for the independent sanity checks: Step 10 Check 2 states "The converted pickle stores the raw trial and unit indices of every session ... so the comparison is exact." The `int64` output dtype is not discussed.
