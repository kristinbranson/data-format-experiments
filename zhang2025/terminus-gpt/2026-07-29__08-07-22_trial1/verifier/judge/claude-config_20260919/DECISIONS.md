# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API at all. It reads the release session index directly as a parquet file (`data/one_cache/Brainwidemap/sessions.pqt`), and for every row reconstructs the session directory on disk from the `lab`/`subject`/`date`/`number` columns, assuming the ALF layout `data/one_cache/<lab>/Subjects/<subject>/<date>/<NNN>/alf/...`. Within a session it globs for the files it needs, handling ALF revision folders by globbing `#*#` sub-directories and taking the lexicographically last one (i.e. the latest revision date). Four streams are read per session: `_ibl_trials.table.pqt` (trials), `_ibl_wheel.position.npy` / `_ibl_wheel.timestamps.npy` (wheel), `leftCamera.ROIMotionEnergy.npy` / `rightCamera.ROIMotionEnergy.npy` (whisker), and per probe `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt` under `alf/probe*/pykilosort/#rev#/`. There is no pre-selection of sessions by dataset availability: every row of `sessions.pqt` is attempted, and a session is silently skipped only if its directory is absent, its trials table is absent, or no probe has spikes. 480 sessions were iterated and 459 written out. No parallelism is used; sessions are processed serially in a single process.

ii.
```python
def session_dir_from_row(row):
    return Path('data') / 'one_cache' / str(row['lab']) / 'Subjects' / str(row['subject']) / str(row['date']) / f"{int(row['number']):03d}"

def find_trial_table(session_dir):
    alf = session_dir / 'alf'
    cands = sorted(alf.glob('#*/_ibl_trials.table.pqt'))
    if cands:
        return cands[-1]
    p = alf / '_ibl_trials.table.pqt'
    return p if p.exists() else None

def latest_revision_dir(pykilo_dir):
    revs = sorted([p for p in pykilo_dir.glob('#*#') if p.is_dir()])
    return revs[-1] if revs else pykilo_dir
```

```python
sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
...
for i, (_, row) in enumerate(sessions.iterrows()):
    loaded = load_session(row)
    if loaded is None:
        log('  skipped: missing required data')
        continue
```

iii. From the trajectory (steps 44–45): after inspecting `sessions.pqt` (index `id`, columns `lab`, `subject`, `date`, `number`, ...) and running a filesystem search, the agent concluded "we now have the concrete filesystem layout … This is enough to replace the placeholder loader with a real first-pass implementation." CONVERSION_NOTES Step 2 records that the data are "an IBL ONE cache under `data/one_cache`" and "native IBL ALF/ONE-style cached session data". The notes never justify preferring direct file reads over the ONE/`SessionLoader`/`SpikeSortingLoader` API that the reference code uses; it appears to have been chosen simply because the file layout was discoverable.

## 1-b. How are the data split into subjects?

i. The subject name is taken verbatim from the `subject` column of `sessions.pqt`, so no parsing is needed. A subject is appended to the `subjects` list the first time it is seen (encounter order, not sorted), and `subject_idx` stores that index per session. The full run produced 139 subjects over 459 sessions.

ii.
```python
subject = str(row['subject'])
```

```python
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. Not explicitly justified in CONVERSION_NOTES beyond the Step 5 mapping table; the session index already carries a unique subject identifier, so nothing has to be derived.

## 1-c. How are the data split into sessions?

i. A session is one row of `sessions.pqt`, which maps one-to-one onto one directory `<lab>/Subjects/<subject>/<date>/<number>`. No splitting is performed. Multiple probes belonging to the same session are merged into a single population rather than being treated as separate sessions.

ii.
```python
for i, (_, row) in enumerate(sessions.iterrows()):
    log(f'processing session {i+1}/{len(sessions)}: {row["lab"]}/{row["subject"]}/{row["date"]}/{int(row["number"]):03d}')
    loaded = load_session(row)
```

iii. CONVERSION_NOTES Step 4/Step 5 decision 4: "**Merge probes within session**: Matches methods paper treatment of probes within a session", citing the methods paper statement that neurons from the same session and region are combined across probes rather than decoded separately.

## 1-d. How are the data split into trials?

i. Trials are the rows of `_ibl_trials.table.pqt`; the split is given by the data. The table is filtered (see 1-e) and re-indexed, and every downstream array is built by iterating over the surviving rows in order.

ii.
```python
def load_trials(session_dir):
    p = find_trial_table(session_dir)
    if p is None:
        return None
    return pd.read_parquet(p)
```

```python
mask = trial_mask(trials)
trials = trials.loc[mask].reset_index(drop=True)
if len(trials) < 2:
    return None
```

iii. Step 5 mapping table: trial variables are drawn from the IBL trial table, which is already one row per trial; no justification was needed.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, all weak. (1) `trial_mask` keeps rows where `stimOn_times`, `choice` and `probabilityLeft` are all finite. (2) Later, while building outputs, a trial is dropped if its mapped `choice` or `prior` is NaN — i.e. no-response trials (`choice == 0`) and any `probabilityLeft` outside {0.2, 0.5, 0.8} are removed. (3) A session with fewer than 2 surviving rows is skipped entirely. **No reaction-time filter is applied**, and **no check is made that the wheel or camera streams actually cover the trial window**. On a spot-checked session (UCLA049/2022-08-02/001) 217 of 505 trials (43%) had `firstMovement_times - stimOn_times` outside the reference's [0.08, 2] s bounds and were all retained. 294,851 trials were written in total.

ii.
```python
def trial_mask(trials):
    need = ['stimOn_times', 'choice', 'probabilityLeft']
    mask = np.ones(len(trials), dtype=bool)
    for c in need:
        if c in trials.columns:
            mask &= np.isfinite(trials[c].to_numpy())
    return mask
```

```python
for i in range(len(trials)):
    if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
        continue
    ...
    valid_trial_keep.append(i)
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules" states: "Use valid task trials from the IBL trial table and apply **the same trial mask logic as the reference code (`load_trials_and_mask`)** … Exclude trials lacking required alignment/behavioral variables for the requested decoder outputs." Step 4 repeats "Apply reference trial mask and additionally drop trials missing requested outputs." The trajectory shows the agent identified `load_trials_and_mask` by name at step 14 from a grep, but never opened its body, so the reaction-time bounds and the no-response flag it implements were never carried into the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from every `alf/probe*/pykilosort/<latest revision>/` directory of the session. `clusters.metrics.pqt` is read only to obtain the per-cluster `label` used for quality filtering. `clusters.channels.npy` is located but never read.

ii.
```python
st = base / 'spikes.times.npy'
sc = base / 'spikes.clusters.npy'
cm = base / 'clusters.metrics.pqt'
ch = base / 'clusters.channels.npy'
if not (st.exists() and sc.exists()):
    continue
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

iii. Step 5 mapping table: "`spikes.times`, `spikes.clusters`, curated `clusters` metadata → neural … reference functions `load_good_units`, `merge_probes`, `bin_spiking_data`". Step 1 concluded the conversion "should … follow the IBL ephys convention of loading trial tables, filtering valid trials, loading good units only, and binning spikes relative to trial events rather than using precomputed rates."

## 2-b. How is the `neural` data processed?

i. Spikes of the surviving clusters are counted into 20 ms bins spanning the trial window, giving one integer count per unit per bin; counts are **not** divided by the bin width, so the values are spike counts, not firing rates in Hz. No smoothing is applied. Probes of one session are pooled: each probe's surviving clusters are renumbered consecutively with a running `offset` so probe 1's units continue after probe 0's, and the two probes' spike-time arrays are concatenated. **The concatenated spike-time array is never re-sorted**, so it is piecewise- rather than globally-monotonic, while the binning routine uses `np.searchsorted` on it.

ii.
```python
uniq = np.array(sorted(np.unique(spikes_c)))
remap = {cid: i + offset for i, cid in enumerate(uniq)}
spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
offset += len(uniq)
all_times.append(spikes_t)
all_clusters.append(spikes_c)
region_names.extend(['unknown'] * len(uniq))
...
return np.concatenate(all_times), np.concatenate(all_clusters), region_names
```

```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, edges[-1], side='right')
    st = spike_times[lo:hi]
    sc = spike_clusters[lo:hi]
    ...
    tbin = np.digitize(st, edges) - 1
    good = (tbin >= 0) & (tbin < mat.shape[1])
    np.add.at(mat, (sc, tbin), 1)
    return mat
```

iii. Step 4/Step 5 decisions 1, 3 and 4: "20 ms common bin size: Matches paper analyses for wheel and population activity and simplifies alignment across streams"; "Good units only: Follows IBL QC and reference code curation"; "Merge probes within session: Matches methods paper treatment of probes within a session". No justification is given for keeping counts rather than rates; the notes simply describe "Bin spike counts in 20 ms bins per trial aligned to stimulus onset".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Per probe, `clusters.metrics.pqt` is read and only clusters whose `label` column equals exactly 1 are kept; spikes belonging to any other cluster are discarded. If the metrics file lacks a `label` column, the code falls back to `ks2_label == 'good'`; if the metrics file is missing entirely, it silently keeps **all** clusters. After masking, only clusters that still have at least one spike anywhere in the recording are numbered as units. There is no exclusion of units located outside the brain (`void` in the Beryl atlas), because no anatomical information is loaded at all. The full run reports 75,708 units, exactly the "well-isolated neurons" figure of the data paper.

ii.
```python
good_ids = None
if cm.exists():
    m = pd.read_parquet(cm)
    cols = {c.lower(): c for c in m.columns}
    if 'label' in cols:
        good_ids = np.flatnonzero(m[cols['label']].to_numpy() == 1)
    elif 'ks2_label' in cols:
        good_ids = np.flatnonzero(m[cols['ks2_label']].astype(str).str.lower().eq('good').to_numpy())
if good_ids is None:
    good_ids = np.unique(spikes_c)
keep = np.isin(spikes_c, good_ids)
```

iii. Step 3 "Neuron curation rules": "Use curated good units / passing units as in the IBL reference code and QC metadata." Step 9/trajectory step 735 records the cross-check: "`Brain region: unknown: 75708 neurons`, matching the paper's well-isolated neuron count", which the agent treated as confirmation that the `label == 1` criterion reproduced the paper's stringent QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All IBL streams share one session clock, so alignment is a subtraction. The bin edges of a trial are `stimOn_times + [-0.2, -0.18, …, 1.0]`, i.e. the fixed relative grid is shifted onto each trial's stimulus onset. The window is **-0.2 s to +1.0 s** around stimulus onset (60 bins), not the -0.5 to +1.5 s, 100-bin window of the reference. `metadata['temporal_alignment_event'] = 'stimulus onset'`, `off_start = -0.2`, `off_end = 1.0`.

ii.
```python
T_START = -0.2
T_END = 1.0

def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
```

```python
edges = stim_time + build_time_edges()
```

iii. Step 3 processing notes: "For wheel decoding in the paper, bins spanned from 200 ms before first wheel movement to 1000 ms after first wheel movement, using a causal window of W=10 bins" and "For this task, we must adapt the same general IBL processing style but align to stimulus onset". Step 5 decision 2: "**Stimulus onset alignment**: Required by the decoder task and consistent with paper analyses around stimulus processing." So the window length was borrowed from the methods paper's wheel-decoding window and re-anchored on stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms (`BIN_SIZE_S = 0.02`), reported as `metadata['time_bin_size'] = 20.0` ms. The window of -0.2 to 1.0 s gives exactly 60 bins for every trial of every session (verification confirms `T: min 60, max 60`). No resampling or interpolation of the spike data occurs; spikes are assigned to bins by `np.digitize`. The same 20 ms grid is reused for the input and both time-varying outputs, so all streams share one resolution.

ii.
```python
BIN_SIZE_S = 0.02
...
def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

```python
'time_bin_size': 20.0,
```

iii. Step 3 expected-statistics table: "Neural data time bin | 20 ms used in population/decoding analyses | 'firing rates … across all trials in 20-ms bins'" and "Behavior data time bin | 20 ms for wheel decoding in the methods paper | 'Wheel values were averaged in nonoverlapping 20-ms bins'". Step 5 decision 1 adopts 20 ms as the common bin size for all streams.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable. It is the fixed vector of bin centres of the alignment window, identical for every trial and session, spanning -0.19 s to +0.99 s. The alignment event itself comes from `trials.stimOn_times`, but the input values are defined by the window constants.

ii.
```python
centers = build_time_centers().astype(np.float32)
...
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. Step 5 mapping table: "stimulus-onset-aligned time axis → input[0] → Continuous time-since-stimulus-onset replicated across bins → trial interval construction + binning logic → Time-varying decoder input."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. The edges are `np.arange(-0.2, 1.02, 0.02)` and the values are the midpoints of consecutive edges, cast to float32. Verification reports the range as exactly [-0.2, 1.0] (printed to one decimal).

ii.
```python
def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)

def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

iii. N/A — the variable is defined by the conversion, not measured.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the neural binning grid itself. `bin_spikes_for_trial` uses `stim_time + build_time_edges()` and the input uses the centres of those same edges, so element *t* of the input and column *t* of the neural matrix describe the same 20 ms interval relative to the same `stimOn_times`. The same `centers` vector is also used to place the wheel and whisker samples, so all four streams share one axis.

ii.
```python
edges = stim_time + build_time_edges()          # neural
...
centers = build_time_centers().astype(np.float32)   # input
ws = bin_signal(wheel_v_ts, wheel_v, st + centers)  # outputs placed on the same centres
```

iii. Not separately justified; it follows from Step 5 decision 1 (a single common 20 ms grid for all streams).

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `trials.probabilityLeft`, which is constant within a block, so a change of its value marks a block boundary. There is no explicit block identifier in the trials table.

ii.
```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
trial_in_block = trial_number_in_block(prob_left)
```

iii. Step 5 decision 6: "**Use trial number within block as decoder input**: Derived from block prior run lengths because the requested input is not a native raw variable."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python run-length counter over the `probabilityLeft` sequence: the counter starts at **1** on the first trial and on every trial whose prior differs from the previous trial's, and increments otherwise. It is computed on the table **after** the finite-value mask but **before** the no-response/invalid-prior trials are dropped, so a dropped trial still advances the counter and the value remains the animal's true position in the block. The per-trial scalar is broadcast across all 60 bins. Verification reports the range [1.0, 99.0], consistent with the paper's 20–100 trial blocks.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return out
    run = 1
    out[0] = 1
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            run += 1
        else:
            run = 1
        out[i] = run
    return out
```

```python
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. Step 5 decisions 5 and 6: per-trial variables are "broadcast … across time bins" to keep tensor shapes consistent, and the block position is recovered from prior run lengths. Step 3 notes the task structure ("Initial 90 unbiased trials, then 20:80 or 80:20 blocks"; "Blocks lasted for between 20 and 100 trials").

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is +1, -1 or 0. +1 (leftward) maps to 0, -1 (rightward) to 1, and 0 (no response) maps to NaN and causes the trial to be dropped.

ii.
```python
def map_choice(v):
    # IBL usually: 1=CCW(left), -1=CW(right)
    return 0 if v == 1 else 1 if v == -1 else np.nan
```

```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

iii. Step 5 mapping table: "`trials.choice` → output[0] → Map left=0, right=1 → `load_trials_and_mask` → Per-trial categorical output." The inline comment records the IBL sign convention. The instructions specify left = 0, right = 1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the recoding. The scalar is cast to int and broadcast across all 60 bins as a constant row of the output matrix, stored as int64. `output_values[0] = ['left', 'right']`. The full dataset is nearly balanced: left 0.505, right 0.495.

ii.
```python
output_trials.append(np.vstack([
    np.full(centers.shape, int(choice[i]), dtype=np.int64),
    np.full(centers.shape, int(prior[i]), dtype=np.int64),
    wb,
    mb,
]).astype(np.int64))
```

iii. Step 5 decision 5: "**Broadcast per-trial variables across time bins when needed**: Keeps input/output tensor shapes consistent for the decoder." The target format also states that outputs should be made time-varying where possible.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, compared to 0.2 / 0.5 / 0.8 with `np.isclose` and recoded to 0 / 1 / 2. Any other value yields NaN and drops the trial.

ii.
```python
def map_prior(v):
    if np.isclose(v, 0.2):
        return 0
    if np.isclose(v, 0.5):
        return 1
    if np.isclose(v, 0.8):
        return 2
    return np.nan
```

```python
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

iii. Step 5 mapping table: "`trials.probabilityLeft` → output[1] → Map 0.2->0, 0.5->1, 0.8->2", i.e. the mapping is taken directly from the Decoder Task specification. Step 3 records the block structure that produces these three values.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding; the scalar is broadcast over the 60 bins as an int64 row, with `output_values[1] = ['0.2', '0.5', '0.8']`. The unbiased 0.5 block (the first ~90 trials of each session) is retained, giving the full-run distribution 0.2: 0.417, 0.5: 0.140, 0.8: 0.443.

ii.
```python
np.full(centers.shape, int(prior[i]), dtype=np.int64),
```

iii. As 5-b: per-trial variables are broadcast across bins to keep shapes uniform.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `alf/_ibl_wheel.position.npy` and `alf/_ibl_wheel.timestamps.npy`, read directly with `np.load`. If either file is missing the wheel stream is `None` for that session and no session-level exclusion follows.

ii.
```python
def load_wheel(session_dir):
    alf = session_dir / 'alf'
    pos = alf / '_ibl_wheel.position.npy'
    ts = alf / '_ibl_wheel.timestamps.npy'
    if not (pos.exists() and ts.exists()):
        return None, None
    return np.load(ts), np.load(pos)
```

iii. Step 5 mapping table: "wheel velocity / speed dataset → output[2] → Align to stimulus onset, average/bin at 20 ms, discretize into 3 bins". Step 4 resolution: "Use cached wheel/video-derived variables when available".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Speed is the absolute first difference of the raw encoder trace: `|Δposition / Δt|` evaluated at the midpoints of consecutive raw samples, keeping only samples with `Δt > 0`. There is **no** interpolation onto a regular grid and **no** low-pass filter, i.e. neither of the two steps `SessionLoader.load_wheel` performs (1 kHz `interpolate_position` followed by a 20 Hz Butterworth `velocity_filtered`). The resulting irregular samples are then averaged within each 20 ms bin by `bin_signal`; a bin containing no wheel sample is left as NaN. Because the IBL rotary encoder only emits samples while the wheel turns, this makes stationary periods NaN rather than zero — on the spot-checked session 26.3% of all bins were NaN. Those NaNs are later replaced by the session's 33rd-percentile threshold value (see 7-c).

ii.
```python
def wheel_speed(ts, pos):
    if ts is None or pos is None or len(ts) < 2:
        return None, None
    dt = np.diff(ts)
    dp = np.diff(pos)
    good = dt > 0
    v = np.zeros_like(pos, dtype=np.float32)
    mids = ts[:-1][good] + dt[good] / 2
    vv = np.abs(dp[good] / dt[good]).astype(np.float32)
    return mids, vv
```

```python
def bin_signal(ts, values, centers):
    ...
    idx = np.digitize(ts, edges) - 1
    for i in range(len(centers)):
        m = idx == i
        if np.any(m):
            out[i] = np.nanmean(values[m])
    return out
```

iii. Step 3: "Wheel values were averaged in nonoverlapping 20 ms bins in the methods paper", which is the stated basis for the per-bin averaging. CONVERSION_NOTES Step 10 records the NaN handling as a fix rather than a principled choice: "Sample-stage NaN/non-integer outputs in wheel/whisker bins: fixed by forcing categorical integer bins and filling missing values before discretization." The absence of interpolation/filtering is never mentioned or justified.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per session. All finite binned speed values across all trials of the session are pooled, the 1/3 and 2/3 quantiles are taken, and every bin is assigned with `np.digitize(x, [q1, q2], right=False)` to class 0 / 1 / 2 (`['low','mid','high']`). NaN bins are first replaced by `q1` itself, which `digitize` maps to class **1 ("mid")**, not class 0 — so stationary bins are labelled medium speed. The resulting full-dataset distribution is skewed away from the intended equal thirds: low 0.260, mid 0.423, high 0.316. If a session has no wheel data at all, the thresholds default to `[0, 0]` and every bin becomes class 2 ("high"), since `np.digitize(0.0, [0, 0])` returns 2.

ii.
```python
wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
...
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
```

iii. CONVERSION_NOTES Step 5 decision 7: "**Discretize wheel speed and whisker motion energy into 3 bins**: Required by the task; exact thresholding to be chosen after inspecting distributions", and an inline comment "session-wise thresholds for reproducibility". The NaN fill is justified only as a bug-fix for verification errors (Step 10, Step 12: "filling missing values before discretization").

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. `bin_signal` is called with query points `stimOn_times + centers`, and internally rebuilds edges as `centers ± 10 ms`, i.e. exactly the neural bin edges shifted onto the trial's stimulus onset. Each output bin is therefore the mean of the wheel samples falling inside the same 20 ms interval as the corresponding neural column. The wheel timestamps are on the same session clock as the spike times, so no further correction is applied.

ii.
```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers) if wheel_v is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

```python
edges = np.concatenate([[centers[0] - BIN_SIZE_S / 2], centers + BIN_SIZE_S / 2])
```

iii. Step 5 mapping table: "Align to stimulus onset, average/bin at 20 ms"; Step 5 decision 1 makes the 20 ms grid common to all streams, so one axis serves neural, input and output.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` **and** `rightCamera.ROIMotionEnergy.npy` (whichever are present, in the revision sub-directories of `alf/`). When both are present they are truncated to the shorter length and **element-wise averaged into a single trace**. The per-frame timestamps `_ibl_leftCamera.times.npy` / `_ibl_rightCamera.times.npy` — which are present in `alf/` for these sessions — are **never loaded**. 16 of the 461 session directories have no motion-energy file at all, and are not excluded.

ii.
```python
def load_motion_energy(session_dir):
    left_p, right_p = find_motion_energy(session_dir)
    vals = []
    for p in [left_p, right_p]:
        if p is not None and p.exists():
            vals.append(np.load(p).astype(np.float32))
    if not vals:
        return None
    n = min(len(v) for v in vals)
    arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
    return arr
```

iii. Step 5 mapping table: "whisker motion energy dataset → output[3] → Align to stimulus onset, average/bin at 20 ms, discretize into 3 bins → video/behavior dataset loading + custom alignment". No rationale is given for averaging the two cameras rather than choosing one; the notes only acknowledge (Step 7) that "whisker timestamps are approximated from available camera/session timing because explicit ROI timestamps were not yet loaded."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used unfiltered and unnormalised, apart from the left/right averaging above. Its timestamps are **fabricated** by `guess_motion_timestamps`: it looks for a `_ibl_*Camera.features.pqt` file (which does not exist in this cache), then falls back to `np.linspace` spread evenly between `min(stimOn_times) + T_START` and `max(stimOn_times) + T_END`. On the spot-checked session this produced a time base of 18.6 s – 2791.0 s for frames whose true times are 10.6 s – 4213.4 s, i.e. the trace is shifted and compressed by ~1.5×. The trace is then binned by `bin_signal` at the trial bin centres and cut into tertiles as for the wheel.

ii.
```python
def guess_motion_timestamps(session_dir, n):
    # Fallback: use right camera features if present; otherwise spread across session interval.
    alf = session_dir / 'alf'
    feats = sorted(alf.glob('_ibl_*Camera.features.pqt'))
    if feats:
        ...
    trials = load_trials(session_dir)
    if trials is not None and 'stimOn_times' in trials.columns:
        t0 = np.nanmin(trials['stimOn_times'].to_numpy()) + T_START
        t1 = np.nanmax(trials['stimOn_times'].to_numpy()) + T_END
        return np.linspace(t0, t1, n, dtype=np.float32)
    return np.arange(n, dtype=np.float32) * BIN_SIZE_S
```

```python
me = load_motion_energy(session_dir)
me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None
```

iii. CONVERSION_NOTES Step 7 states the limitation openly: "whisker timestamps are approximated from available camera/session timing because explicit ROI timestamps were not yet loaded." The trajectory (steps 58, 60, 774) shows the agent repeatedly recognised this as "scientifically imperfect" and a likely cause of the weak whisker accuracy, but chose to proceed because the format validation passed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: the 1/3 and 2/3 quantiles of all finite binned values pooled over the whole session, `np.digitize(..., right=False)` into `['low','mid','high']`, with NaN bins replaced by the lower threshold first. Because the camera stream is continuous, NaN bins are rare here, and the resulting distribution is close to equal thirds: low 0.321, mid 0.323, high 0.355. Sessions with no motion-energy file get `mq = [0, 0]` and are assigned class 2 ("high") for every bin of every trial.

ii.
```python
mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
...
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
```

iii. Step 5 decision 7 ("Discretize wheel speed and whisker motion energy into 3 bins: Required by the task"), applied with the same session-wise tertile rule as the wheel for consistency.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Nominally the same as the wheel — `bin_signal(me_ts, me, st + centers)` places each frame into the 20 ms bin of the neural grid. In practice the alignment is meaningless because `me_ts` is the synthetic `linspace` of 8-b rather than the recorded frame times, and because the averaged left/right trace mixes a 60 Hz and a 150 Hz stream that cover different spans of the recording (the first 256,086 right-camera frames cover ~1,695 s while the 256,086 left-camera frames cover ~4,203 s).

ii.
```python
ms = bin_signal(me_ts, me, st + centers) if me is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. Step 5 mapping table specifies "Align to stimulus onset, average/bin at 20 ms"; Step 7 records the approximation caveat. Step 12 of the notes links the weak result to it implicitly: "whisker_motion_energy_bin | 0.4180 | 0.4008 | Above chance but weaker", and the trajectory at step 774 attributes this to "approximate whisker timestamps/region labels".

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled by silent skipping at the session level and by silent substitution at the bin level. Sessions are skipped when the directory does not exist, the trials table is missing or lacks `stimOn_times`, fewer than 2 trials survive the finite mask, or no probe has spike files. Within a session: a missing `clusters.metrics.pqt` silently disables quality filtering and keeps every cluster; missing wheel files produce an all-NaN speed trace; missing motion energy produces an all-NaN whisker trace; missing camera times are fabricated by `linspace`. NaN behavioural bins are not dropped but replaced with the lower tertile threshold, which places them in class "mid" for the wheel and class "high" for a session with no camera data at all. Trials whose window is not covered by the wheel or camera stream are not detected or removed. 85 trials out of 294,851 ended up with an all-zero neural matrix; these were quantified and kept.

ii.
```python
if not session_dir.exists():
    return None
trials = load_trials(session_dir)
if trials is None or 'stimOn_times' not in trials.columns:
    return None
...
if len(trials) < 2:
    return None
spike_times, spike_clusters, region_names = load_spikes_and_regions(session_dir)
if spike_times is None:
    return None
```

```python
if good_ids is None:
    good_ids = np.unique(spikes_c)
```

```python
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
```

iii. CONVERSION_NOTES Step 10/12: "Sample-stage NaN/non-integer outputs in wheel/whisker bins: fixed by forcing categorical integer bins and filling missing values before discretization"; "Verification warning for all-zero neural trials: quantified at 85 all-zero trials out of 294,851 total trials (~0.029%), indicating rare no-spike windows rather than a pervasive formatting failure." The trajectory (step 51) shows the fill was chosen specifically to make `train_decoder.py --verify-only` stop reporting NaN/non-integer outputs.

## 10-a. What are the most time-consuming steps of the code?

i. The AI documented no bottleneck analysis: CONVERSION_NOTES Step 6 leaves "Code inefficiencies identified: [Note]" and "Code speedups added: [Note]" as unfilled placeholders, and Step 7 only estimates "~35-75 s/session … Full run likely several hours without optimization" without acting on it. The full conversion in fact ran from 12:20:17 to 18:46:56, about 6.5 hours, far beyond the 15-minute threshold at which the instructions require optimisation. The actual dominant costs are (1) `bin_signal`, which re-`digitize`s the entire session-length wheel array (~700k samples) once per trial and then runs a 60-iteration Python loop each building a full-length boolean mask — executed twice per trial, roughly 10^8 element operations per trial; and (2) the per-spike Python dict lookup in the cluster remap, which iterates over millions of spikes per probe. `np.add.at` and the per-trial `np.searchsorted`/`np.digitize` calls are secondary.

ii.
```python
idx = np.digitize(ts, edges) - 1
for i in range(len(centers)):
    m = idx == i
    if np.any(m):
        out[i] = np.nanmean(values[m])
```

```python
spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
```

iii. No justification is offered; the notes simply state (Step 7) "Full run likely several hours without optimization" and the agent proceeded anyway. The instructions' Step 7.2 requirement to speed the code up if the estimate exceeded 15 minutes was not acted on, and no parallelism was used (the reference runs 10 worker processes).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI identified none. Four are vectorizable: (1) the `for i in range(len(centers))` loop in `bin_signal` — a single `np.bincount(idx, weights=values)` divided by `np.bincount(idx)` replaces it; (2) the `[remap[c] for c in spikes_c]` comprehension — `np.searchsorted(uniq, spikes_c) + offset` is the vectorized equivalent; (3) `np.add.at(mat, (sc, tbin), 1)`, which is the slow unbuffered path — a single `np.bincount(sc * n_bins + tbin).reshape(...)` is far faster, as the reference does; (4) the per-trial `for i, st in enumerate(stim_times)` loop, which could be replaced by one flat-index `bincount` over all trials, and the pure-Python `trial_number_in_block` run-length loop, which is a `groupby(...).cumcount()` on a `cumsum` of prior changes.

ii.
```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
    input_trials.append(...)
    ws = bin_signal(wheel_v_ts, wheel_v, st + centers) ...
    ms = bin_signal(me_ts, me, st + centers) ...
```

```python
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        run += 1
    else:
        run = 1
    out[i] = run
```

iii. No justification given — the corresponding CONVERSION_NOTES fields were never filled in.

## 10-c. What processing does the code repeat multiple times?

i. Not documented by the AI. Repeats present in the code: `load_trials(session_dir)` is executed a second time inside `guess_motion_timestamps` for the same session that already has the table in memory (an extra parquet read per session); `build_time_edges()` and `build_time_centers()` are recomputed on every call of `bin_spikes_for_trial` and `bin_signal`, i.e. twice per trial per stream; `bin_signal` recomputes `np.digitize(ts, edges)` over the *entire* session-long wheel/camera array once per trial instead of slicing the window; `find_motion_energy` re-globs the revision directories; and `np.quantile` is computed both inside `discretize_three_bins` and again as `wq`/`mq` on the same pooled arrays.

ii.
```python
def guess_motion_timestamps(session_dir, n):
    ...
    trials = load_trials(session_dir)      # already loaded by the caller
```

```python
wheel_bins_all = discretize_three_bins(all_wheel_cont) if len(all_wheel_cont) else ...
me_bins_all = discretize_three_bins(all_me_cont) if len(all_me_cont) else ...
# session-wise thresholds for reproducibility
wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
```

iii. Not justified or mentioned anywhere in CONVERSION_NOTES.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Not documented by the AI. Concretely: `wheel_bins_all` and `me_bins_all` are computed by `discretize_three_bins` over every pooled behavioural sample in the session and then **never used** — the code re-derives the thresholds as `wq`/`mq` immediately afterwards. The neural and input matrices are built for *every* masked trial and only afterwards subset by `valid_trial_keep`, so the expensive spike binning is done for trials that are then thrown away. `region_names` is built as a list of the literal string `'unknown'` per unit purely to carry a count. `clusters.channels.npy` is resolved into the variable `ch` and never read. `latest_file()` is defined but never called, `v` inside `wheel_speed` is allocated and never used, and `n_target` in `build_dataset` is computed and never used. Finally, `--show-processing` is accepted on the command line but is never implemented — no `processing_<session_id>.png` files exist, so the required verification plots were never produced.

ii.
```python
wheel_bins_all = discretize_three_bins(all_wheel_cont) if len(all_wheel_cont) else np.array([], dtype=np.float32)
me_bins_all = discretize_three_bins(all_me_cont) if len(all_me_cont) else np.array([], dtype=np.float32)
```

```python
neural_trials = [neural_trials[i] for i in valid_trial_keep]
input_trials = [input_trials[i] for i in valid_trial_keep]
```

```python
ap.add_argument('--show-processing', action='store_true')   # parsed, never used
```

iii. Not justified or mentioned anywhere in CONVERSION_NOTES; Step 6's inefficiency fields were left as template placeholders and the Step 7 "Processing Plots Review" section describes verification output rather than any plot.
