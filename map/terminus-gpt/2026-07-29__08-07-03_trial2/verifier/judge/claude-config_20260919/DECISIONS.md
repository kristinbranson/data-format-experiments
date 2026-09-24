# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is the DANDI 000363 NWB release, one `.nwb` file per session under `data/sub-<subject_id>/`. The AI discovers every session with a single recursive glob relative to the current working directory (so the script must be run from `/app`), sorts the paths for determinism, and opens each file **once** with raw `h5py` rather than `pynwb`. Everything else (trials table, units table, behavioral time series, electrode table, subject id) is read from HDF5 paths inside that one open handle. `--sample` truncates the list to the first 2 files. All 174 files are opened and all 174 are kept in the final output.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
if args.sample:
    paths = paths[:2]
data = build_dataset(paths, show_processing=args.show_processing)
```

```python
with h5py.File(session_path, 'r') as f:
    trials = f['intervals/trials']
    n_trials = len(trials['id'])
    ...
    units = f['units']
```

```python
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
```

iii. From CONVERSION_NOTES.md Step 2/Step 4: "Data are organized as subject-specific directories under `data/`, each containing NWB session files ... Each NWB file contains standard groups including `units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, and `stimulus`." Key decision 1 in Step 5: "**Use NWB as source of truth**: Reference code operates on preprocessed arrays and external paths, so conversion will load raw NWB data directly while matching paper/code semantics for curation and alignment." The AI found (Step 1) that `/app/code` reads pre-processed pickles from `/oak/...` paths and therefore could not be reused for loading. `h5py` was chosen implicitly (the trajectory shows all exploration was done with `h5py`; `pynwb` is never mentioned). The counts it derived this way — 174 sessions, 28 subjects, 272,227 units, 94,990 trials — are recorded in Step 2.

## 1-b. How are the data split into subjects (mice)?

i. Each session's subject is read from the NWB metadata field `general/subject/subject_id` (a numeric string such as `'440956'`), with a fallback to the parent directory name (`sub-440956`) if that field is missing or unreadable. Subjects are collected in **first-encounter order** into `subjects`, and `subject_idx` holds each session's index into that list. This yields 28 subjects with 3–10 sessions each.

ii.
```python
def get_subject_id(f, path):
    try:
        sid = f['general/subject/subject_id'][()]
        return _decode(sid)
    except Exception:
        return path.parent.name
```

```python
sid = info['subject']
if sid not in subject_to_idx:
    subject_to_idx[sid] = len(subjects)
    subjects.append(sid)
subject_idx.append(subject_to_idx[sid])
```

iii. No explicit written justification; the choice follows from Step 5's "Use NWB as source of truth" decision — `subject_id` is the canonical animal identifier stored in the file, and the `sub-<id>/` directory name is derived from it, so the directory is a safe fallback. The resulting 28 subjects matched the AI's own Step 2 count from the data.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as exactly one session; no grouping or splitting is performed. The session identifier is the filename stem (e.g. `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`), stored per session in `metadata['session_info']`. Session order in the output follows the sorted file list. A session is dropped only if it yields fewer than 2 usable trials or zero good units.

ii.
```python
info = {
    'subject': get_subject_id(f, session_path),
    ...
    'session_id': session_path.stem,
    'load_seconds': time.time() - t0,
}
```

```python
if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
    print(f'Skipping {path.name}: insufficient kept trials or no good units')
    continue
```

iii. The AI documented in Step 2 that the data are "organized as subject-specific directories ... each containing NWB session files", i.e. the file boundary is the session boundary and nothing has to be inferred. The ≥2-trial minimum is the target-format requirement from the instructions. In Steps 9/10 the AI noted the resulting count (174) does not match the paper's 173 sessions and listed this as an unresolved discrepancy: "converted sessions = 174 vs paper 173".

## 1-d. How are the data split into trials?

i. Trials come directly from the NWB trials table `intervals/trials`, one row per behavioural trial. The row count is taken as `len(trials['id'])` and every per-trial column (`start_time`, `trial_instruction`, `outcome`, `early_lick`, `photostim_onset`, `photostim_duration`) is read as a full vector and indexed by row. Trial boundaries are never re-derived from the event streams, and the number of go-cue events is never cross-checked against the number of rows.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
trial_start = trials['start_time'][:].astype(float)
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START

trial_instruction = np.array([_decode(x) for x in trials['trial_instruction'][:]])
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
```

```python
for i in range(n_trials):
    if not valid_trials[i]:
        continue
```

iii. Step 5's variable-mapping table lists "Use trial fields `start_time`, `trial_instruction`, `outcome`, `early_lick`, `photostim_onset`, `photostim_duration`". The trials table is the explicit, unambiguous record of trial structure, so the AI used it directly.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied, none of them taken from the file's own observation metadata:

1. **Label validity** — the trial's `trial_instruction`, `outcome` and `early_lick` must each be one of the expected strings.
2. **Neural coverage** — the whole `[go − 2.5 s, go + 1.5 s]` window must lie inside `[spike_t_min, spike_t_max]`, the global first/last spike time pooled over all good units of that session.
3. **All-zero drop** — after binning, any trial whose entire firing-rate matrix is zero is discarded.

Sessions with fewer than 2 surviving trials or zero good units are dropped. No behavioural quality filter (early lick / ignore) is applied, and `free_water`, `auto_water` and `units/obs_intervals` are never used. Net effect: 94,990 raw trials → 90,997 kept, 174/174 sessions kept.

ii.
```python
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)
```

```python
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
valid_trials = valid_trials & window_ok
```

```python
if np.all(fr == 0):
    continue
```

iii. Filter 1 comes from Step 5 key decision 5: "**Restrict to trials with complete required variables**: Needed to avoid invalid labels or misalignment." Filters 2 and 3 were added reactively in Step 10 after the decoder verification flagged all-zero neural trials. CONVERSION_NOTES.md Step 10: "Root cause identified: in some sessions, behavioral/trial timestamps continue long after `units/spike_times` end (e.g. spikes end ~1107 s while trials continue to ~3705 s). Therefore many later trials have no concurrent neural recording. Conversion must exclude trials whose aligned window falls outside neural recording coverage." Then: "Residual issue after neural-coverage filtering: 91 sessions still contained all-zero neural trials (e.g. sessions 47-75 and others). Planned fix: explicitly drop any trial whose binned neural matrix is all zeros after histogramming." The trajectory (step 1152) confirms the coverage filter was introduced specifically to make the verification warnings disappear.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike times from the ragged NWB units table: `units/spike_times` (the flat concatenated buffer) together with `units/spike_times_index` (the per-unit end offsets), split into one array per unit. Only units passing `units/unit_quality == 'good'` contribute. The other input to the binning is the alignment time, which is **not** read from the file but computed as `trials/start_time + 1.85`.

ii.
```python
def get_unit_spike_times(units_group):
    spikes = units_group['spike_times'][:]
    index = units_group['spike_times_index'][:]
    out = []
    start = 0
    for stop in index:
        out.append(spikes[start:stop])
        start = stop
    return out
```

```python
units = f['units']
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. Step 5 mapping table: "`units/spike_times` + unit metadata in `units` → neural | Bin spikes into 50 ms bins from -2.5 s to +1.5 s around go cue; keep only curated good units". Spike times are the only neural representation in the file, so firing rates are computed from them directly.

## 2-b. How is the `neural` data processed?

i. Per trial and per good unit, spikes are masked to the trial window, histogrammed into the 80 fixed 50 ms bins, and divided by the bin width to give Hz, stored as `float32`. No smoothing, normalisation, or baseline subtraction. Units with no spikes in the window leave a row of zeros.

ii.
```python
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. Step 5 key decision 3: "**Use 50 ms bins for neural and derived time-varying variables**: Required by decoder task". Converting counts to a rate by dividing by the bin width matches the reference code's `sliding_histogram(..., rate=True)` convention. The AI noted in Step 6 that this per-neuron histogram loop is "a bottleneck on full dataset".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if `units/unit_quality == 'good'` (the alternative value being `'multi'`). No metric thresholds and no use of the `units/classification` column. This retains **154,948 of 272,227** units (57%), a mean of 890 per session. The AI compared this to the paper's reported 69,943 good units, found the mismatch, documented it, and shipped it unresolved.

ii.
```python
units = f['units']
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. Step 3 curation rules: "Use only units passing the paper's quality-control procedure, i.e. units labeled as `good` by region-specific classifiers trained on manual curation." Step 5 key decision 4: "**Filter to good units only**: Matches paper curation using region-specific QC classifiers." The AI believed `unit_quality` was that field — trajectory step 24: "Unit metadata includes `unit_quality` and `is_good_trials`; `unit_quality` has values like `good` and `multi`, which likely supports filtering to good units." It later recognised the discrepancy (Step 9/10): "unit curation from available NWB fields (`unit_quality`) yields 154,948 units, which does not match the 69,943 good units reported in the paper", and at step 1705 of the trajectory planned to "inspect the `classification` field ... to determine whether we can derive ... a stricter unit filter" — but that inspection was never carried out, and the notes conclude: "The remaining unresolved issue is a documented mismatch between NWB `unit_quality == good` counts and the paper's reported curated good-unit count."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The go cue is **not** read from the file. It is synthesised as a fixed offset from the trial table's `start_time`: `go = start_time + 1.85 s`, derived from the nominal task timing in methods.txt (3 × 150 ms tones + 2 × 100 ms gaps = 0.65 s sample epoch, + 1.2 s delay). Bin edges are then `go + [-2.5 … +1.5]` in absolute session time and spikes are histogrammed against them. `acquisition/BehavioralEvents/go_start_times`, which contains one exact go-cue timestamp per trial, is never used for alignment.

ii.
```python
# From methods.txt for the auditory delayed response task:
# sample epoch: 3 x 150 ms tones with 100 ms inter-tone intervals => 0.65 s total
# delay epoch: 1.2 s
# go cue at 1.85 s after trial start
TONE_ONSET_FROM_TRIAL_START = 0.0
GO_CUE_FROM_TRIAL_START = 1.85
```

```python
trial_start = trials['start_time'][:].astype(float)
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

```python
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
```

iii. Trajectory step 28: "The trial table lacks explicit go cue/tone timestamps, and stimulus groups are empty in the inspected file, so the most defensible approach is to derive tone and go cue timing relative to trial start from the documented task structure ... go cue onset at 0.15+0.1+0.15+0.1+0.15+1.2 = 1.85 s after trial start". Step 4 discrepancy table records the same: "If explicit event timestamps are absent, derive tone/go cue times from documented task structure relative to trial start." The AI only inspected the trials table and `stimulus/`, not `acquisition/BehavioralEvents/`, before committing to this. It later did print the `BehavioralEvents` contents (trajectory step 1149, which shows `go_start_times n 480 ... first5 [3.1786, 11.422, 18.0066, 25.4169, ...]`) while debugging all-zero trials, and even wrote "The presence of behavioral event timestamps suggests that these event timestamps may be on the same clock as spikes and should be used instead for alignment" — but it then pivoted to blaming the ephys/behaviour clock span, added the coverage filter, and never replaced the 1.85 s constant.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins per trial spanning −2.5 s to +1.5 s relative to the (estimated) go cue. The 81 relative edges and 80 centres are computed once at module level and reused for every trial and session, so `n_timepoints = 80` everywhere. No rebinning, resampling or smoothing is applied to the spikes — they are binned once at 50 ms straight from spike times. `metadata['time_bin_size']` is reported as 50.0 ms.

ii.
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

```python
'time_bin_size': 50.0,
'temporal_alignment_event': 'Go cue onset',
'off_start': WINDOW_START,
'off_end': WINDOW_END,
```

iii. Directly specified by the instructions ("Extract 2.5 s before to 1.5 s after the go cue"; "Use 50-ms-width bins"). Step 3 notes the tension with the reference: "Neural data time bin | 40 ms in visible reference analysis scripts; decoder task requires 50 ms bins", resolved in favour of the decoder spec. Step 5 key decision 3: "Required by decoder task; behavior/video streams will be resampled or rasterized onto this grid."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Only `intervals/trials/start_time`. Tone onset is defined as the trial start itself (`TONE_ONSET_FROM_TRIAL_START = 0.0`). `acquisition/BehavioralEvents/sample_start_times`, which holds the actual per-trial tone onsets, is not used.

ii.
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
```

```python
trial_start = trials['start_time'][:].astype(float)
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. Same justification as 2-d: the AI concluded that no explicit tone timestamp existed in the trials table or `stimulus/` group and fell back on the nominal task structure, assuming the sample epoch begins at trial onset. Step 4: "Trial table lacks explicit go-cue/tone columns in inspected sessions ... derive tone/go cue times from documented task structure relative to trial start."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For every trial, the absolute bin centres are differenced against that trial's tone-onset time and stored as `float32` in row 0 of the `(2, 80)` input array. Because both the tone and the go cue are fixed offsets from the same `start_time`, the gap between them is the constant 1.85 s on every trial, so the resulting vector is the *identical* ramp `−0.6 … 3.3 s` for all 90,997 trials in all 174 sessions (confirmed in `verification_full_out.txt`, where every session reports input 0 range `[-0.6, 3.3]`).

ii.
```python
# input 0: time from tone onset in seconds, time-varying
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
...
inp = np.stack([time_from_tone, photo], axis=0)
```

iii. No standalone justification beyond the derived-timing decision. Trajectory step 33 treats the constant range as confirmation rather than a warning sign: "The time_from_tone_onset range [-0.6, 3.3] is consistent with go cue at 1.85 s and window [-2.5, 1.5], so alignment math is internally consistent."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same bin grid used for the firing rates: `centers_abs = go_cue_times[i] + BIN_CENTERS`, built inside the same per-trial loop as `edges_abs`. So input bin *k* covers the same interval as neural bin *k* by construction, with no interpolation or per-stream offset. (The grid itself sits ~1.3 s earlier than the true go cue — see 2-d.)

ii.
```python
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel

# neural: spike counts per bin
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
...
# input 0: time from tone onset in seconds, time-varying
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. Step 5 key decision 2: "**Align all streams to go cue**: Required by decoder task and consistent with task epoch structure in methods", and key decision 3: "behavior/video streams will be resampled or rasterized onto this grid."

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`, both stored as strings measured from trial start, with the literal `'N/A'` on non-stimulated trials, plus `intervals/trials/start_time` to convert them to absolute session time.

ii.
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

```python
def parse_optional_time(x):
    x = _decode(x)
    if isinstance(x, str) and x == 'N/A':
        return None
    try:
        return float(x)
    except Exception:
        return None
```

iii. Trajectory step 21: "`photostim_onset`, `photostim_duration`, and `photostim_power` exist but may be `N/A` for non-photostim trials. This is exactly what we need for outputs and one input." Step 5 mapping lists these as the photostim source fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary `float32` time series rather than a per-trial flag: the stimulation onset and offset are put on the absolute clock (`start_time + onset`, `+ duration`) and every bin whose centre falls in `[on, off)` is set to 1.0, all others 0.0. Trials with `'N/A'` onset or duration (`None`) skip the branch entirely and keep an all-zero row.

ii.
```python
# input 1: photostimulation on/off, time-varying
photo = np.zeros(N_BINS, dtype=np.float32)
p_on = photostim_onset[i]
p_dur = photostim_duration[i]
if p_on is not None and p_dur is not None:
    p_start = trial_start[i] + p_on
    p_stop = p_start + p_dur
    photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Required by the instructions: "Whether **photostimulation** is on at every time point (discrete, time-varying)" and "If an input is a time such as onset of some stimulus, represent it as a binary time series." Step 3 also records from methods.txt that "Photostimulation ended before the go cue in the cited manipulation protocol", consistent with expecting the stim to land in the pre-go part of the window.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Both the stimulation interval and the bin centres are expressed in absolute session seconds and compared directly, so the photostim row is on the same 80-bin grid as the firing rates for that trial. This is algebraically the same operation the reference performs in go-cue-relative coordinates. It inherits the ~1.3 s alignment error of the window itself (2-d), so the stim epoch lands later in the extracted window than it should.

ii.
```python
p_start = trial_start[i] + p_on
p_stop = p_start + p_dur
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Same as 3-c — Step 5 key decisions 2 and 3, all streams rasterised onto the single go-cue-referenced 50 ms grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. A single column, `intervals/trials/trial_instruction` (`'left'` / `'right'`) — i.e. the side the tone *instructed*, not the side the animal actually licked. `outcome` is read but is not combined with the instruction to recover the lick direction, and no lick-time stream (`left_lick_times` / `right_lick_times`) is consulted. Only two classes are emitted; there is no "no lick" class even though 14.9% of retained trials are `ignore`.

ii.
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
```

```python
trial_instruction = np.array([_decode(x) for x in trials['trial_instruction'][:]])
```

```python
choice_val = CHOICE_MAP[trial_instruction[i]]
```

```python
'output_values': [
    ['left', 'right'],
    ...
```

iii. No explicit justification is given anywhere in CONVERSION_NOTES.md or the trajectory; the mapping table in Step 5 simply says "Keep per-trial categorical labels for choice/outcome/early lick" and lists `trial_instruction` as the choice source. Trajectory step 21 notes `trial_instruction` "exists and is a string like `right`", and it appears to have been adopted as the choice variable without further analysis. The AI did register the availability of `left_lick_times` / `right_lick_times` (step 24) but never used them, and it never revisited the two-class encoding despite the instructions listing three classes.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `'left' → 0`, `'right' → 1` via a fixed dict that accepts both `bytes` and `str` keys; the scalar is then broadcast across all 80 bins as `int64` and stacked as row 0 of the `(4, 80)` output array. Resulting distribution over the full dataset: left 0.485 / right 0.515.

ii.
```python
choice_val = CHOICE_MAP[trial_instruction[i]]
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
...
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)
```

iii. The 0 = left / 1 = right coding follows the instructions' ordering. Broadcasting a per-trial scalar across bins is done so that all four outputs share one `(n_output, n_timepoints)` array, satisfying the target format's preference: "Can be time-varying or discrete values per trial. If at all possible, make it time-varying."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `intervals/trials/outcome`, read directly; it already contains exactly the three strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
```

iii. Trajectory step 21: "`outcome` exists and is stored as strings like `ignore`". The trials table stores the outcome explicitly with exactly the three categories the instructions ask for, so no derivation is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped through a fixed dict to `ignore → 0`, `miss → 1`, `hit → 2`, broadcast across all 80 bins as row 1 of the output array. Trials whose `outcome` string is not one of the three are excluded upstream by the label-validity filter. Full-dataset distribution: ignore 0.149 / miss 0.166 / hit 0.684.

ii.
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
```

```python
outcome_val = OUTCOME_MAP[outcome[i]]
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The code assignment follows the instructions' listed order (`ignore, miss, hit`). Broadcast across bins like the other per-trial outputs for a uniform output array. The AI cross-checked the hit fraction against the paper's reported performance in Step 3 ("Reward rate | 83.2% control; 71.7% bilateral ALM photostim").

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. `intervals/trials/early_lick`, which holds the strings `'no early'` / `'early'`.

ii.
```python
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
```

iii. The trials table flags early licking explicitly, so no derivation from lick times is needed. Listed in the Step 5 mapping table among the trial fields used for outputs.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `no early → 0`, `early → 1` and broadcast across all 80 bins as row 2. Full-dataset distribution: no 0.885 / yes 0.115.

ii.
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
```

```python
early_val = EARLY_MAP[early_lick[i]]
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. Coding order follows the instructions (`no, yes`). One value per trial, repeated across bins like the other per-trial outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: its `data` array (`n_frames × 3`) and its `timestamps`. Column 1 is taken as tongue y and column 2 as the tracking likelihood. A helper picks any `*TongueTracking*` series in `BehavioralTimeSeries`, preferring one whose name starts with `Camera0`, and returns `None` if none exists.

ii.
```python
def choose_tongue_group(f):
    bts = f.get('acquisition/BehavioralTimeSeries')
    if bts is None:
        return None
    candidates = [k for k in bts.keys() if 'TongueTracking' in k]
    if not candidates:
        return None
    # Prefer Camera0 if present, otherwise first available
    candidates = sorted(candidates, key=lambda x: (not x.startswith('Camera0'), x))
    return bts[candidates[0]]
```

```python
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
# Convention in these tracking arrays is assumed [x, y, likelihood]
if tongue_data.ndim == 2 and tongue_data.shape[1] >= 2:
    tongue_y = tongue_data[:, 1].astype(float)
```

iii. Trajectory step 24: "Tongue tracking is available at `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` with `data` shape (n_frames, 3) and `timestamps`; likely columns are x, y, likelihood or similar." The column layout was *assumed* rather than read from the series' `description` attribute — the code comment says so explicitly. Step 5 was corrected mid-way after an earlier note referenced the wrong camera: "the output row still references `Camera3_side_TongueTracking` even though the confirmed path is `Camera0_side_TongueTracking`".

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood < 0.5 are set to NaN. The 40th and 60th percentiles are then taken over **all remaining raw frames of the session** (`np.nanpercentile` over the frame-level y values, not over 50 ms bin means). Per trial, each bin's value is taken as the **nearest surviving frame** to the bin centre — found with `searchsorted` and accepted only if it lies within 0.1 s of the centre — rather than the mean of the frames inside the bin. There is no averaging or interpolation.

ii.
```python
if tongue_data.shape[1] >= 3:
    lik = tongue_data[:, 2].astype(float)
    tongue_y = tongue_y.copy()
    tongue_y[lik < 0.5] = np.nan

# Session-wise discretization thresholds from all valid tongue samples
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])
else:
    q40, q60 = np.nan, np.nan
```

```python
valid_idx = np.flatnonzero(np.isfinite(tongue_y))
...
vt = tongue_t[valid_idx]
vy = tongue_y[valid_idx]
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y = y.copy()
y[far] = np.nan
```

iii. The likelihood mask and the switch to nearest-valid sampling were introduced in Step 7 after the first attempt put ~98% of bins in the middle class. Trajectory step 34: "only ~10.5% of frames have likelihood >= 0.9, so our current strategy of defaulting missing/invalid bins to the middle class causes ~98% of timepoints to be class 1 ... A better approach is to use nearest valid tongue samples only, possibly with a looser likelihood threshold". CONVERSION_NOTES.md Step 7: "Tongue discretization improved after lowering likelihood threshold and nearest-valid sampling, but remains strongly skewed toward the middle class; keep under review during decoder training."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes only. The per-bin array is **initialised to 1** and then overwritten with 0 where the sampled y is below `q40` and with 2 where it is above `q60`. Consequently class 1 means *either* "between the 40th and 60th percentile" *or* "no tongue visible near this bin centre" *or* "the session has no usable tongue data at all". The fourth class required by the instructions (`3: not visible`) is never emitted, and `output_values[3]` declares only three names. The resulting full-dataset distribution is `lt_40pct 0.080 / 40_to_60pct 0.901 / gt_60pct 0.019` instead of the ≈40/20/40 the percentile definition implies.

ii.
```python
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
if len(valid_idx) > 0:
    ...
    finite = np.isfinite(y)
    tongue_disc[finite & (y < q40)] = 0
    tongue_disc[finite & (y > q60)] = 2
else:
    tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['lt_40pct', '40_to_60pct', 'gt_60pct'],
],
```

iii. The AI never states a rationale for omitting the "not visible" class; it repeatedly recorded the resulting skew as an open concern rather than a decision. Step 7: "remains strongly skewed toward the middle class; keep under review during decoder training". Step 8 format validation: "note brain regions fallback to `unknown` and tongue output remains imbalanced." After full training it concluded only that the variable decoded above chance (0.8148 balanced accuracy) and closed Step 12 without revisiting the class definition.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Via the shared absolute-time grid: camera `timestamps` are on the same session clock as the spikes, and for each of the 80 bin centres of a trial the nearest retained camera frame is located with `searchsorted` and accepted if within 0.1 s. No interpolation, no per-stream offset correction. Note that the ±0.1 s acceptance window is four times the 50 ms bin width, so a frame from an adjacent bin can supply a bin's value; and the grid inherits the ~1.3 s go-cue error from 2-d.

ii.
```python
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y = y.copy()
y[far] = np.nan
```

iii. Step 5 key decision 3: "behavior/video streams will be resampled or rasterized onto this grid". The nearest-sample-with-tolerance scheme was chosen in Step 7 specifically to increase the number of bins with a usable tongue value (trajectory step 34), since tongue tracking is valid in only ~10% of frames.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Six distinct cases, handled by a mixture of explicit branches and blanket exception handlers:

- **`'N/A'` photostim strings** → `parse_optional_time` returns `None`; the trial gets an all-zero photostim row. A bare `except Exception` also swallows any unparseable value.
- **Missing/unreadable `subject_id`** → falls back to the `sub-<id>` directory name.
- **Missing tongue series, wrong array shape, or a session with no usable tongue frames** → `q40/q60` become NaN and every bin is emitted as class 1.
- **Low-likelihood tongue frames** → set to NaN, excluded from the percentiles, and (if no valid frame is near a bin centre) imputed as class 1 — i.e. missing data is folded into a real category rather than flagged.
- **Missing electrode/location metadata** → an outer `try/except Exception` labels every unit `'unknown'`; per-unit failures give `'unknown'` for that unit.
- **Unexpected trial label strings, trials outside spike coverage, all-zero trials, sessions with <2 trials or 0 good units** → dropped (see 1-e, 1-c).

ii.
```python
def parse_optional_time(x):
    x = _decode(x)
    if isinstance(x, str) and x == 'N/A':
        return None
    try:
        return float(x)
    except Exception:
        return None
```

```python
try:
    elec = units['electrodes'][:]
    ...
    brain_region_labels = [all_unit_regions[i] for i in np.where(good_mask)[0]]
except Exception:
    brain_region_labels = ['unknown' for _ in good_spike_times]
```

```python
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])
else:
    q40, q60 = np.nan, np.nan
```

```python
def _decode(x):
    if isinstance(x, bytes):
        return x.decode()
    return x
```

iii. The instructions call for handling missing data "appropriately (consult references, use sensible defaults, document)". The AI's stated principle is to drop trials/sessions whose required variables are absent (Step 5 key decision 5) and to use permissive defaults elsewhere; the `bytes`/`str` dual keys in the mapping dicts and `_decode` exist because HDF5 string columns may be returned either way depending on the file. The `'unknown'` region fallback was the original behaviour and was described in Step 6 as a "known limitation" before being partly fixed in Step 10 ("Brain-region mapping resolved by deriving per-unit region labels from electrode location metadata"), though the blanket `except` and the stale metadata note ("brain region metadata unavailable in inspected NWB unit tables, so fallback region label `unknown` used") remain in the shipped code.

## 10-a. What are the most time-consuming steps of the code?

i. Spike binning dominates overwhelmingly. For every trial the code loops over every good unit and runs a full boolean scan of that unit's entire spike-time array before histogramming, so the cost is O(n_trials × n_units × n_spikes_per_unit) per session. The full conversion log shows 5,730 s total (≈95 minutes) across 174 sessions, 5–143 s per session, growing with trials × units. Secondary costs are reading the whole `units/spike_times` buffer and slicing it into a Python list of per-unit arrays (done for *all* units, not just good ones) and pickling the 26.4 GB result. The AI identified the bottleneck but never fixed it, and left the Step 7 timing tables empty.

ii.
```python
for i in range(n_trials):
    ...
    fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
        if np.any(mask):
            counts, _ = np.histogram(st[mask], bins=edges_abs)
            fr[n] = counts.astype(np.float32) / BIN_SIZE
```

```python
def get_unit_spike_times(units_group):
    spikes = units_group['spike_times'][:]
    index = units_group['spike_times_index'][:]
    out = []
    start = 0
    for stop in index:
        out.append(spikes[start:stop])
        start = stop
    return out
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: Per-neuron histogram loop is a bottleneck on full dataset; observed ~17.7 s/session over first 3 sessions, implying ~50+ min for full run if unoptimized." And: "Code speedups added: Uses direct histogramming on pre-sliced spike times and skips sessions without >=2 valid trials or without good units. Need further optimization before full conversion because projected runtime exceeds 15 minutes." No further optimisation was performed; the Step 7 "Run Time Estimates" tables were left as empty templates.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested `for i in range(n_trials)` / `for n, st in enumerate(good_spike_times)` binning loop is the main candidate and is the direct cause of the 95-minute runtime. Because the bin edges of all trials can be flattened into one monotonically increasing array, the entire per-unit computation collapses to a single `np.searchsorted` over that unit's spikes followed by `np.diff` — one pass per unit instead of one full-array scan per (unit, trial). The `mask`/`np.any`/`np.histogram` triple inside the inner loop is itself three redundant passes where one would do. Other vectorizable loops: the Python list-comprehension decoding of the trial string columns and the `valid_trials` comprehension (both O(n_trials) Python-level), `get_unit_spike_times`'s per-unit slicing loop, and the per-electrode-index loop building `all_unit_regions`. The per-trial output/input construction could also be built as whole `(n_trials, ...)` arrays in one shot.

ii.
```python
for i in range(n_trials):
    if not valid_trials[i]:
        continue
    align = go_cue_times[i]
    edges_abs = align + BIN_EDGES
    ...
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
        if np.any(mask):
            counts, _ = np.histogram(st[mask], bins=edges_abs)
            fr[n] = counts.astype(np.float32) / BIN_SIZE
```

```python
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)
```

iii. The instructions required vectorised loops and a conversion under 15 minutes ("If full conversion time estimate is longer than 15 minutes, speed up the code by writing more efficient code (vectorizing loops, removing unnecessary or redundant computations) and/or including parallel processing"). The AI acknowledged the requirement in Step 6 ("Need further optimization before full conversion because projected runtime exceeds 15 minutes") but did not act on it, and Step 9's instruction to kill and re-optimise a run exceeding 1.5× the estimate was also not followed.

## 10-c. What processing does the code repeat multiple times?

i. Several recomputations, from trivial to expensive:

- **`go_cue_times` is computed twice** from the same inputs, once at the top of `extract_session` and again a few lines later before the coverage filter — a leftover from the Step 10 patch.
- **The full spike array is rescanned for every trial**: `mask = (st >= edges_abs[0]) & (st < edges_abs[-1])` touches all of a unit's spikes on each of the ~500 trials, then `np.any(mask)` walks the mask again, then `np.histogram` walks the selected spikes a third time.
- **`edges_abs` / `centers_abs` are rebuilt inside the trial loop** from constants that never change, and `BIN_CENTERS.copy()` is copied per trial.
- **All units' spike arrays are sliced out** in `get_unit_spike_times`, including the ~43% of units immediately discarded by the quality filter.
- **Per-trial constants are re-broadcast** into 80-element arrays for choice/outcome/early lick on every trial rather than once per session.

ii.
```python
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
...
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
```

```python
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
centers_rel = BIN_CENTERS.copy()
centers_abs = align + centers_rel
```

iii. Not documented — CONVERSION_NOTES.md records no analysis of repeated work, and the only efficiency claim ("Uses direct histogramming on pre-sliced spike times") is not what the shipped code does. The duplicated `go_cue_times` line is visible in the trajectory as an artefact of patching: the filter was first inserted before `spike_t_min` existed, raised `UnboundLocalError`, and was then moved lower without removing the extra assignment.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four kinds of wasted work:

- **Inputs and outputs are built for trials that are then thrown away.** The `np.all(fr == 0)` drop is the *last* statement in the trial loop, so `time_from_tone`, `photo`, the three broadcast label rows and the whole tongue nearest-neighbour lookup are all computed first and discarded for ~4,000 trials.
- **Spike arrays are materialised for every unit**, then 117,279 of 272,227 are dropped by the quality mask.
- **Whole sessions are fully processed before being skipped.** `build_dataset` only checks `len(ntr) < 2` after `extract_session` has returned, so a session that fails the minimum-trials test is binned in full first.
- **Unused / dead values**: `centers_rel = BIN_CENTERS.copy()` is copied then immediately added to `align` (the copy serves no purpose); `spike_t_min`/`spike_t_max` are recomputed even for sessions with no good units; `q40`/`q60` and the per-unit region parse run even when the session is later dropped; `info` fields `n_trials_raw`, `n_units_raw`, `q40`, `q60`, `load_seconds` are stored in `metadata['session_info']` but unused by the decoder. Separately, the `--show-processing` flag is accepted and threaded through `extract_session(..., show_processing=...)` but the parameter is never read — no plots are produced, despite being required by the instructions.

ii.
```python
ap.add_argument('--show-processing', action='store_true', help='Reserved; plotting omitted in this implementation')
```

```python
def extract_session(session_path, show_processing=False):
```

```python
out = np.stack([choice_ts, outcome_ts, early_ts, tongue_disc], axis=0)

if np.all(fr == 0):
    continue

neural_trials.append(fr)
input_trials.append(inp)
output_trials.append(out)
```

```python
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
    if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
        print(f'Skipping {path.name}: insufficient kept trials or no good units')
        continue
```

iii. Not documented. Step 6 claims the script "skips sessions without >=2 valid trials or without good units" as a speedup, but as written that check happens only after the session has been fully converted. The `--show-processing` omission is acknowledged in Step 6 as a "known limitation" ("`--show-processing` plotting not yet implemented") and was never implemented, so the Step 7 requirement to inspect processing plots was satisfied only by the note "Sample verification succeeded structurally."
