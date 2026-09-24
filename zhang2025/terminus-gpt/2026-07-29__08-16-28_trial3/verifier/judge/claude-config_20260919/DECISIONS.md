# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the ONE API (it imports `from one.api import ONE` but never instantiates it). Instead it walks the local ONE cache directory tree directly with `pathlib.glob`, treating every path matching `data/one_cache/<lab>/Subjects/<subject>/<date>/<number>` that contains an `alf/` sub-directory as one session. 461 candidate session directories are found. Within a session, each dataset is read straight off disk with `pd.read_parquet` / `np.load`:

- trials: `alf/#<revision>#/_ibl_trials.table.pqt`, falling back to `alf/_ibl_trials.table.pqt`;
- spikes: `alf/probe*/pykilosort/[#<revision>#/]{spikes.times.npy, spikes.clusters.npy, clusters.metrics.pqt, clusters.channels.npy, channels.brainLocationIds_ccf_2017.npy}`;
- wheel: `alf/_ibl_wheel.timestamps.npy`, `alf/_ibl_wheel.position.npy` (top level **only**);
- camera: `alf/#<revision>#/_ibl_<side>Camera.times.npy` and `alf/#<revision>#/<side>Camera.ROIMotionEnergy.npy` (revision folders **only**).

Where several revision folders exist, `find_latest_file` takes the lexicographically last one. A session is dropped if any of the four streams cannot be found. In the full run this kept only **159 of 461 sessions** (55 subjects, 108,235 trials): 238 sessions were skipped for "no whisker motion energy" and 63 for "no wheel data". I verified on disk that in those sessions the files are in fact present — e.g. `NYU-12/2020-01-20/001` has `alf/_ibl_leftCamera.times.npy` at the alf root but `alf/#2025-05-31#/leftCamera.ROIMotionEnergy.npy` in a revision folder, and the camera loader only globs `#*/`, so it finds nothing; symmetrically `PL015/2022-02-20/001` has `_ibl_wheel.timestamps.npy` at the alf root but `_ibl_wheel.position.npy` in a revision folder, and the wheel loader only looks at the alf root. So roughly two thirds of the released sessions were lost to inconsistent revision handling, not to genuinely missing data.

ii.
```python
def find_latest_file(session_alf: Path, pattern: str):
    matches = sorted(session_alf.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)
```

```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    if not tp.exists() or not pp.exists():
        return None, None
    return np.load(tp), np.load(pp)


def load_motion_energy(session_alf: Path):
    left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
    right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
    left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
    right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
```

iii. From CONVERSION_NOTES.md Step 6: the first version of the script did use `ONE`/`brainbox` object loading, but `import brainbox` failed in the agent's environment and ONE-based loading pulled unneeded arrays (`spikes.samples.npy`, `spikes.templates.npy`, …) over the network, so the AI "Replaced ONE object loading with direct local `.npy`/`.pqt` reads from session ALF directories, reducing sample conversion time from ~27 s to ~3 s for 2 sessions." The mass skipping was noticed during the full run (trajectory steps 48–51: "requiring whisker motion energy may drastically reduce usable sessions … we need the final count before deciding") but the AI concluded the data were genuinely absent — "this is reasonable for the task", "the keep rate is not catastrophically low" — and never checked the file paths.

## 1-b. How are the data split into subjects?

i. One directory level of the ONE cache is the subject, so the subject name is taken from the session path (`<lab>/Subjects/<subject>/<date>/<number>`, element index 2 of the path relative to the cache root). Subjects are accumulated in first-encountered order into `subjects`, and `subject_idx` stores the index of the subject of each kept session, in the same order as the session lists. The converted file has 55 subjects for 159 sessions.

ii.
```python
rel = session_dir.relative_to(data_root)
parts = rel.parts
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
...
subj = subject
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
out['subject_idx'].append(subject_to_idx[subj])
```

iii. CONVERSION_NOTES.md Step 2: "Local data are organized as an IBL ONE cache under `data/one_cache` … Lab-specific directories", and the Step 5 mapping row "Subject/session path metadata → subjects / subject_idx: Extract unique subject IDs and session-to-subject mapping … Session order must match neural/input/output lists". The subject id is already a directory name, so nothing has to be derived.

## 1-c. How are the data split into sessions?

i. A session is one `<lab>/Subjects/<subject>/<date>/<number>` directory containing an `alf/` folder; no splitting is performed. `choose_sessions` pre-filters to directories that have a trials table, and `--sample` takes the first two of those. Everything downstream (neural / input / output / brain_region_idx / subject_idx) is appended once per kept session, so the session ordering is consistent across all fields.

ii.
```python
def choose_sessions(session_dirs, mode):
    valid = []
    for s in session_dirs:
        alf = s / 'alf'
        if find_latest_file(alf, '#*/_ibl_trials.table.pqt') or (alf / '_ibl_trials.table.pqt').exists():
            valid.append(s)
    if mode == 'sample':
        return valid[:2]
    return valid
```

iii. Implicit in the Step 2 notes: the cache is organised one directory per session, so the session boundary is given by the data and nothing has to be inferred. Multiple probes belonging to one session are merged rather than treated as separate sessions (Step 4: "Combine neurons across probes within session, preserving neuron-level region labels").

## 1-d. How are the data split into trials?

i. One row of `_ibl_trials.table.pqt` is one trial; the split is taken directly from the table. `stimOn_times`, `choice` and `probabilityLeft` are read as columns and indexed by row, and the per-trial arrays (neural, input, output) are built by iterating over the surviving row indices.

ii.
```python
trials_df, trial_path = load_trials_table(session_alf)
if 'stimOn_times' not in trials_df.columns or 'choice' not in trials_df.columns or 'probabilityLeft' not in trials_df.columns:
    print('skip missing required trial columns', session_dir)
    continue
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
choice = map_choice(trials_df['choice'].to_numpy())
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
```

iii. CONVERSION_NOTES.md Step 5 maps the trial table columns straight onto the target fields; the AI checked the required columns exist and skips a session if any is absent. No justification is needed beyond "the trials table is already one row per trial".

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality control is applied. The only trial mask is: `stimOn_times` finite **and** `choice` ∈ {+1, −1} (so no-response trials are dropped) **and** `probabilityLeft` ∈ {0.2, 0.5, 0.8}. A session is dropped if fewer than two trials survive. There is **no reaction-time filter** (the reference paper's 80 ms – 2 s window, and the reference code's `load_trials_and_mask(..., max_trial_len=10.0)`), **no** check that the wheel or camera stream actually covers the −0.5 – 1.5 s window of a trial, and **no** check that the ephys recording covers the trial. As a result the converted file keeps ~681 trials per session (108,235 / 159), against ~428 per session in the human reference, and the verification log reports trials whose neural matrix is entirely zero (session 135, trials 462–467), i.e. trials past the end of the spike recording.

ii.
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    print('skip too few valid trials', session_dir)
    continue
```
with
```python
def map_choice(vals):
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
```

iii. The AI *planned* to do more than it did. CONVERSION_NOTES.md Step 4 records "Trial filtering | `load_trials_and_mask(..., max_trial_len=10.0)` and alignment removes masked trials … Resolution: Apply explicit valid-trial mask before packaging trials", and Step 5 Key Decision 3 is "**Filter to valid trials explicitly**: Use trial masks / valid aligned intervals and implement elementwise mask conjunction, rather than reproducing the apparent Python-list `and` quirk in `align_spike_behavior`." Neither the reaction-time mask nor the behavioural-coverage mask was ever implemented, and the notes contain no explanation for the omission.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` of every `probe*/pykilosort` folder of the session (latest revision). `clusters.metrics.pqt` supplies the `label` column used for quality filtering; `clusters.channels.npy` together with `channels.brainLocationIds_ccf_2017.npy` supplies the anatomical location, which is written into `brain_regions` / `brain_region_idx` but is not used to build the neural matrix itself.

ii.
```python
for probe_dir in sorted((session_dir / 'alf').glob('probe*/pykilosort')):
    revs = sorted([d for d in probe_dir.iterdir() if d.is_dir() and d.name.startswith('#')])
    src = revs[-1] if revs else probe_dir
    st_p = src / 'spikes.times.npy'
    sc_p = src / 'spikes.clusters.npy'
    metrics_p = src / 'clusters.metrics.pqt'
    ...
    st = np.load(st_p)
    sc = np.load(sc_p).astype(int)
    metrics = pd.read_parquet(metrics_p)
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "`spikes.times.npy` + `spikes.clusters.npy` merged across probes within session → neural". Step 6 notes that the earlier ONE-based loader was replaced because it "pulled unnecessary spike/template/waveform arrays, causing major slowdowns" — reading only the two spike arrays is the deliberate narrowing.

## 2-b. How is the `neural` data processed?

i. For every kept trial, spikes are counted into 100 non-overlapping 20 ms bins spanning −0.5 s to +1.5 s relative to that trial's `stimOn_times`. The stored value is the raw **spike count** per neuron per bin, as `float32`; it is *not* divided by the bin width, so the units are counts/20 ms rather than Hz (the human reference stores Hz). No smoothing, normalisation or z-scoring is applied. Units from all probes of a session are pooled into one population: surviving clusters of probe *n* are renumbered continuously after those of probe *n−1*, and the concatenated spike train is sorted by time.

ii.
```python
def bin_spikes(spike_times, spike_clusters, n_neurons, stim_on):
    mats = []
    for t0 in stim_on:
        edges = t0 + TIME_BINS
        out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
        lo = np.searchsorted(spike_times, edges[0], side='left')
        hi = np.searchsorted(spike_times, t0 + T_END, side='left')
        ts = spike_times[lo:hi] - t0
        cl = spike_clusters[lo:hi]
        bins = np.floor((ts - T_START) / BINSIZE).astype(int)
        m = (bins >= 0) & (bins < N_BINS) & (cl >= 0) & (cl < n_neurons)
        np.add.at(out, (cl[m], bins[m]), 1)
        mats.append(out)
    return mats
```
Probe merging and renumbering:
```python
        good = labels >= 1
        offset = sum(len(c['acronym']) for c in clusters_list)
        remap = -np.ones(nclu, dtype=int)
        remap[good] = np.arange(good.sum()) + offset
        ...
        spikes_list.append((st2[keep2], remap[sc2[keep2]]))
    spike_times = np.concatenate([x[0] for x in spikes_list])
    spike_clusters = np.concatenate([x[1] for x in spikes_list])
    order = np.argsort(spike_times)
```

iii. CONVERSION_NOTES.md Step 4: "Temporal binning | `params` in `0_data_caching.py` use `binsize=0.02`, `time_window=(-0.5, 1.5)`, `align_time='stimOn_times'` … Use 20 ms bins and stimulus-onset alignment for this conversion", and "Probe/session handling | `prepare_data` merges all probes for an eid via `eid2pid` and `merge_probes` … Paper says probes in same session are not decoded separately → Combine neurons across probes within session, preserving neuron-level region labels". The README states the stored values are "spike-count matrices"; no reason is given for keeping counts rather than rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single cluster-level cut: only clusters whose `label` in `clusters.metrics.pqt` is ≥ 1 are kept, which is the IBL "stringent" QC label (all metrics passed). Spikes whose cluster index falls outside the cluster table are also dropped. That is the only filter — units placed outside the brain are **not** removed: the AI keeps neurons whose Allen CCF region id is 0 (52 such neurons appear in the verification summary as brain region "0"), whereas the human reference drops units whose Beryl acronym is `void`. Related: because `iblatlas` was not importable in the agent's environment, region labels are stored as stringified Allen CCF **ids** (`'549'`, `'496345668'`, …) rather than acronyms, and at the finest CCF granularity, giving 407 "brain regions" instead of the reference's Beryl acronym set. The filter yields a mean of 178.5 neurons per session (reference: 164.2).

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
...
good = labels >= 1
...
keep_spk = (sc >= 0) & (sc < nclu)
sc2 = sc[keep_spk]
st2 = st[keep_spk]
keep2 = good[sc2]
```
and the region labels:
```python
cluster_channels = np.load(cl_chan_p).astype(int)
channel_region_ids = np.load(ch_brain_p).astype(int)
valid_ch = (cluster_channels >= 0) & (cluster_channels < len(channel_region_ids))
region_ids = np.full(nclu, 0, dtype=int)
region_ids[valid_ch] = channel_region_ids[cluster_channels[valid_ch]]
acr = np.array([str(x) for x in region_ids], dtype=object)
```

iii. CONVERSION_NOTES.md Step 4: "Neuron quality | `good_clusters` defined as `(clusters['label'] >= 1)` … Paper references well-isolated units / canonical cell sets → Use quality-filtered units and record region/QC metadata"; Step 10: "preserved … strict cluster label filtering (`label >= 1`)". On the region labels, Step 10 records the caveat explicitly: "brain-region labels are numeric region IDs as strings rather than acronyms because atlas mapping package was unavailable in the environment; structurally valid but semantically less ideal." (In the judge's environment `iblatlas` is in fact installed; the agent's original import was `from iblatlas.atlas import BrainRegions`, which is the wrong module path — the class lives in `iblatlas.regions`.)

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to visual stimulus onset. All IBL streams already live on one synchronised session clock, so alignment is a subtraction: for each trial the spike times inside the window are shifted by that trial's `stimOn_times`, putting 0 at stimulus onset, and the window runs from −0.5 s to +1.5 s. `np.searchsorted` on the globally time-sorted spike train picks out the window before the subtraction. `metadata['temporal_alignment_event'] = 'stimulus onset'`, `off_start = -0.5`, `off_end = 1.5`.

ii.
```python
    for t0 in stim_on:
        edges = t0 + TIME_BINS
        lo = np.searchsorted(spike_times, edges[0], side='left')
        hi = np.searchsorted(spike_times, t0 + T_END, side='left')
        ts = spike_times[lo:hi] - t0
```
with the window constants
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
```

iii. CONVERSION_NOTES.md Step 4 records the reference parameters `align_time='stimOn_times'`, `time_window=(-.5, 1.5)` and resolves to "Use 20 ms bins and stimulus-onset alignment for this conversion"; the Decoder Task section of the instructions also specifies stimulus onset. Step 3 notes "all streams will be aligned to stimulus onset".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (`metadata['time_bin_size'] = 20.0` ms), 100 bins per trial covering the fixed 2 s window −0.5 – 1.5 s, identical for every trial and session. Spikes are histogrammed once onto this grid; there is no resampling or re-binning of the neural data afterwards. The behavioural traces are sampled at the centres of the same bins.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
N_BINS = len(TIME_BINS)
```
```python
        bins = np.floor((ts - T_START) / BINSIZE).astype(int)
```

iii. CONVERSION_NOTES.md Step 3: "Neural data time bin | 20 ms in reference wheel decoding … methods.txt: 'Spike counts were similarly binned' and wheel values were averaged in nonoverlapping 20-ms bins"; Step 4 resolution "Use 20 ms bins and stimulus-onset alignment"; Step 5 Key Decision 1: "**Use 20 ms bins and stimulus-onset alignment**: Matches the reference caching params (`binsize=0.02`, `align_time='stimOn_times'`) and the task requirement."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable at all: it is the fixed bin-centre vector of the analysis window, `TIME_CENTERS = [-0.49, -0.47, …, 1.49]`, which is defined by the choice of alignment event (`stimOn_times`), window (−0.5 – 1.5 s) and bin size (20 ms). The same vector is written as input channel 0 of every trial of every session, with name `time_since_stim_onset`.

ii.
```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```
```python
                inp = np.vstack([
                    TIME_CENTERS.astype(np.float32),
                    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
                ])
```

iii. CONVERSION_NOTES.md Step 5 mapping: "Trial-relative time axis → input[0] time since stimulus onset | Repeat common time vector across trials as a 1 x T input | reference params in `0_data_caching.py` | Continuous time-varying decoder input." The Decoder Task specifies this input as "continuous, time-varying", so a continuous ramp rather than a binary onset marker was used.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None. The vector is constructed analytically from the three window constants and cast to `float32`; it is identical for every trial. The verification log reports its range as [−0.5, 1.5] (actual values −0.49 … 1.49, matching the reference's [−0.49, 1.49]).

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
...
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. N/A — the variable is defined by the conversion, not measured. The AI's only stated rationale is that it is the "trial-relative time axis" from the reference caching parameters.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction it is the same grid. `bin_spikes` assigns a spike to bin `floor((t − stimOn − T_START) / BINSIZE)`, i.e. bin *k* covers [T_START + k·0.02, T_START + (k+1)·0.02) after stimulus onset; the input value in column *k* is `T_START + k·0.02 + 0.01`, the centre of that same bin. The behavioural outputs are interpolated at exactly these centres, so all four streams share one time axis, column for column.

ii.
```python
        bins = np.floor((ts - T_START) / BINSIZE).astype(int)      # neural
```
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2                              # input channel 0
```
```python
        x = t0 + TIME_CENTERS                                       # behavioural outputs
```

iii. Implicit; CONVERSION_NOTES.md Step 4 lists "Alignment implementation detail" and the AI's resolution "Implement explicit elementwise conjunction in our conversion", and Step 5 records that all streams are placed on the 20 ms stimulus-locked grid. The `--show-processing` plots (`processing_0.png`, `processing_1.png`) plot the neural raster and the two behavioural traces on a shared `TIME_CENTERS` x-axis to make the alignment visible.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table only. The IBL trials table carries no block identifier, so block boundaries are recovered from the fact that `probabilityLeft` is held constant within a block: a change of value starts a new block.

ii.
```python
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. CONVERSION_NOTES.md Step 5 mapping: "Trial table `probabilityLeft` and trial order within block → input[1] trial number in block | Compute within-block trial index from consecutive equal-probability blocks | trial table + reference trial loading | Continuous per-trial or broadcast across time."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A single pass over the raw (unfiltered) `probabilityLeft` column with a running counter that resets to 1 whenever the value changes, so the count is **1-based** (the human reference counts from 0). Crucially the count is computed on the *full* trials table before the validity mask is applied, and then indexed by the original trial row (`trial_in_block[tr]` where `tr` comes from `np.where(valid)[0]`), so a dropped trial still advances the counter and the value is the animal's true position in the block. The scalar is broadcast across all 100 time bins as input channel 1. Observed range in the converted data: [1, 99], with per-session maxima of 90–99 — consistent with the IBL task's 90-trial first block and 20–100-trial pseudo-random blocks.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return out
    cur = prob_left[0]
    c = 0
    for i, p in enumerate(prob_left):
        if i == 0 or p != cur:
            cur = p
            c = 1
        else:
            c += 1
        out[i] = c
    return out
```
```python
            for i, tr in enumerate(np.where(valid)[0]):
                inp = np.vstack([
                    TIME_CENTERS.astype(np.float32),
                    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
                ])
```

iii. CONVERSION_NOTES.md Step 5 as quoted above. The notes do not comment on the 1-based origin or on the decision to count before filtering, but the sample statistics table records the observed range "trial_number_in_block | [1, 90]" as a sanity check against the known 90-trial first block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of `_ibl_trials.table.pqt`, which takes values +1 (leftward wheel turn), −1 (rightward) and 0 (no response).

ii.
```python
choice = map_choice(trials_df['choice'].to_numpy())
```

iii. CONVERSION_NOTES.md Step 5 mapping: "Trial table `choice` → output[0] choice | Map left=0, right=1 using IBL coding after verifying sign convention | trial loading in reference code | Per-trial categorical output."

## 5-b. What processing is involved in computing `output` *Choice*?

i. A recode only: +1 → 0 (left), −1 → 1 (right); anything else (i.e. 0, no response) is set to −1 by `map_choice` and those trials are then removed by the `valid` mask, so no-response trials never reach the output. The per-trial scalar is broadcast across all 100 bins as output channel 0 and stored as `int64`; `output_values[0] = ['left', 'right']`. Converted choice fractions per session cluster around 0.5 (verification log), as expected for the balanced IBL task.

ii.
```python
def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
```
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
...
                out_trial = np.vstack([
                    np.full(N_BINS, choice[tr], dtype=np.int64),
                    ...
```

iii. The instructions state "Choice, binary, per-trial, left = 0, right = 1"; the AI's Step 5 note says the IBL sign convention was verified before mapping. Broadcasting the per-trial scalar across time follows the instruction "If at all possible, make it time-varying"; CONVERSION_NOTES.md Step 5 Key Decision 5: "**Represent per-trial categorical outputs compactly**: Choice and prior can be stored as per-trial vectors".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which takes the three block values 0.2, 0.5 and 0.8.

ii.
```python
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
```

iii. CONVERSION_NOTES.md Step 5 mapping: "Trial table `probabilityLeft` → output[1] prior probability of left | Map 0.2->0, 0.5->1, 0.8->2 | trial table | Per-trial categorical output."

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A recode only, using `np.isclose` for float-safe comparison: 0.2 → 0, 0.5 → 1, 0.8 → 2; any other value is set to −1 and the trial is dropped by the `valid` mask. The per-trial scalar is broadcast across the 100 bins as output channel 1; `output_values[1] = ['0.2','0.5','0.8']`. The converted distribution is ≈ [0.42, 0.15, 0.43] (per-session values in the verification log), closely matching the human reference's [0.418, 0.141, 0.442] — the 0.5 class is small because it occurs only in the 90-trial unbiased first block.

ii.
```python
def map_prior(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[np.isclose(vals, 0.2)] = 0
    out[np.isclose(vals, 0.5)] = 1
    out[np.isclose(vals, 0.8)] = 2
    return out
```

iii. The mapping is given verbatim in the instructions ("Prior probability of left, per-trial, 0.2 -> 0, 0.5 -> 1, 0.8 -> 2") and repeated in CONVERSION_NOTES.md Step 5. `np.isclose` was used because `probabilityLeft` is stored as float.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy` — the raw rotary-encoder timestamps and angular position in radians. Only copies sitting directly in `alf/` are looked for; if either file lives in a revision folder the session is skipped (63 sessions lost this way). No IBL/brainbox wheel helper is used.

ii.
```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    if not tp.exists() or not pp.exists():
        return None, None
    return np.load(tp), np.load(pp)
```

iii. CONVERSION_NOTES.md Step 5 mapping: "`_ibl_wheel.position.npy` + `_ibl_wheel.timestamps.npy` → output[2] wheel speed | Differentiate/interpolate to stimulus-aligned 20 ms bins, then discretize into 3 bins | `bin_behaviors`, `load_target_behavior` | Time-varying categorical output." Step 4 notes the reference code's behaviour list includes "wheel velocity/speed".

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps, none of which reproduce the IBL/reference pipeline:
1. Non-finite samples are dropped and duplicate timestamps removed (`np.unique` on the timestamps, keeping the first position of each).
2. A velocity is computed with `np.gradient(position, timestamps)` **directly on the raw, irregularly sampled encoder trace** — no interpolation onto a uniform 1 kHz grid and no 20 Hz Butterworth low-pass, which is what `brainbox.behavior.wheel` (used by the reference via `SessionLoader.load_wheel`) does. The AI removed the brainbox dependency because `import brainbox` failed in its environment.
3. The **signed** velocity is interpolated at the 100 bin centres of each trial, with `left=np.nan, right=np.nan` outside the recorded range.

No absolute value is ever taken — `np.abs` does not appear anywhere in the script — so output channel 2, named `wheel_speed_bin`, actually carries signed wheel **velocity** (direction of turning), not speed. This is visible in the conversion log, where the tertile thresholds are negative/positive pairs, e.g. `wheel_thr=(-0.0369, 0.0441)`. Compared against the reference pipeline on one session, IBL's filtered |velocity| is < 0.01 rad/s for 52% of the recording with tertiles at 0.0027 / 0.047 rad/s, i.e. a genuinely "stationary / slow / fast" split; the AI's signed gradient instead splits into "turning right / nearly still / turning left". The raw-gradient derivative is also much noisier than the filtered one, and because the encoder only emits samples when the wheel moves, interpolating across a long stationary gap invents a non-zero velocity.

ii.
```python
def interp_wheel_speed(timestamps, position, stim_on):
    ...
    uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
    timestamps = uniq_t
    position = position[uniq_idx]
    ...
    vel = np.gradient(position, timestamps)
    trials = []
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
        trials.append(y.astype(np.float32))
    return trials
```

iii. CONVERSION_NOTES.md Step 6: "Initial version used ONE object loading …"; trajectory step 36: "`brainbox` is not importable in the current Python environment … The simplest fix is to replace the `brainbox.behavior.wheel.velocity` import with a local wheel velocity computation using numpy gradients, which is sufficient for our conversion task." Step 10 records the follow-up fix: "Wheel interpolation warnings from duplicate timestamps: deduplicated wheel timestamps before `np.gradient`." No justification is given anywhere for emitting signed velocity under the name *wheel speed*, and the notes do not acknowledge the discrepancy.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into three classes at the 1/3 and 2/3 quantiles computed **per session** over all finite values of all trials pooled together (`np.quantile(allv, [1/3, 2/3])`), giving three equally sized classes within each session — the same scheme the human reference uses. Values above the first threshold become 1, above the second become 2. Non-finite values (trials or bins outside the wheel recording, and whole sessions with a degenerate wheel trace) are assigned class **0**, i.e. missing data is silently merged into the "low" class rather than excluded. `output_values[2] = ['low','mid','high']`, which mislabels the classes given that the underlying quantity is signed velocity (class 0 is fast rightward turning, not "low"). The verification log confirms the intended balance: nearly every session shows fractions 0.333/0.333/0.333.

ii.
```python
def discretize_tertiles(list_of_arrays):
    allv = np.concatenate([x[np.isfinite(x)] for x in list_of_arrays if np.isfinite(x).any()])
    q1, q2 = np.quantile(allv, [1/3, 2/3]) if len(allv) else (0.0, 1.0)
    out = []
    for x in list_of_arrays:
        y = np.zeros_like(x, dtype=np.int64)
        y[x > q1] = 1
        y[x > q2] = 2
        y[~np.isfinite(x)] = 0
        out.append(y)
    return out, (float(q1), float(q2))
```
```python
            wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "**Discretize wheel speed and whisker motion energy into 3 bins using global/session-aware quantile-style thresholds to avoid degenerate classes**: exact thresholds to be finalized during implementation after checking distributions." Step 7 notes "wheel/whisker bins are perfectly balanced due to tertile discretization" as the sanity check. The NaN → class 0 convention is not discussed anywhere in the notes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel velocity trace is evaluated by linear interpolation at absolute times `stimOn_times + TIME_CENTERS`, i.e. at the centres of exactly the same 100 bins the spikes are counted into, using the same per-trial stimulus onset. Wheel timestamps and spike times are already on the same synchronised session clock, so no further correction is applied. Column *k* of output channel 2 therefore describes the same 20 ms as column *k* of the neural matrix.

ii.
```python
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. CONVERSION_NOTES.md Step 5: "align/interpolate to 20 ms bins"; Step 4 resolution "Use wheel speed and whisker motion energy as time-varying outputs". The `--show-processing` plots draw the wheel trace on the same `TIME_CENTERS` axis as the neural raster to show there is no offset.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` together with their frame times `_ibl_leftCamera.times.npy` / `_ibl_rightCamera.times.npy` — the IBL-released motion energy of a square ROI over the whisker pad, one value per video frame. Unlike the reference (left preferred, right as fallback), the AI collects **every** available side and combines them. All four files are looked for **only** inside revision folders (`#*/`), which is why 238 sessions were skipped as having "no whisker motion energy" even though the arrays exist on disk (the camera *times* are usually at the alf root).

ii.
```python
def load_motion_energy(session_alf: Path):
    left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
    right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
    left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
    right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
    streams = []
    for tp, mp in [(left_t, left_me), (right_t, right_me)]:
        if tp is not None and mp is not None and tp.exists() and mp.exists():
            streams.append((np.load(tp), np.load(mp).astype(np.float32)))
    if not streams:
        return None
    return streams
```

iii. CONVERSION_NOTES.md Step 3: "`methods.txt` states whisker motion energy is computed as the mean absolute frame-to-frame pixel difference in whisker-pad ROIs from the left/right videos"; Step 4 notes the reference "`bin_behaviors` has a special case for `whisker-motion-energy` using left and possibly right whisker motion energy"; Step 5 mapping: "Use available whisker ROI motion energy stream(s), align/interpolate to 20 ms bins, combine left/right sensibly if both available, discretize into 3 bins."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is used as-is — no filtering, no normalisation, no rescaling. Each available camera's trace is linearly interpolated at the 100 bin centres of each trial (with the frame-time array truncated to the length of the motion-energy array, `ts[:len(me)]`, to tolerate the common off-by-a-few length mismatch), and then the left and right values are **averaged bin-by-bin** over whichever streams are finite there; a bin with no finite stream becomes NaN. Averaging the two cameras un-normalised is questionable: on a session I checked, left-camera ROI motion energy has median 2.42 and 99th percentile 15.6 while the right camera has median 0.72 and 99th percentile 3.46 — a ~4× scale difference, so the plain mean is effectively dominated by the left camera and the composite has no consistent physical meaning across sessions with only one camera versus both.

ii.
```python
def interp_motion_energy(streams, stim_on):
    trials = []
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        ys = []
        for ts, me in streams:
            ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
        arr = np.stack(ys, axis=0)
        valid = np.isfinite(arr)
        denom = valid.sum(axis=0)
        summed = np.where(valid, arr, 0.0).sum(axis=0)
        y = np.divide(summed, denom, out=np.full(arr.shape[1], np.nan, dtype=np.float32), where=denom > 0)
        trials.append(y.astype(np.float32))
    return trials
```

iii. CONVERSION_NOTES.md Step 5: "combine left/right sensibly if both available"; Step 10: "Whisker `nanmean` warnings when both streams were invalid: replaced `np.nanmean` with explicit finite-value averaging and all-NaN handling." The notes give no reason for averaging rather than preferring one camera, and do not consider the differing amplitude scales of the two cameras.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: `discretize_tertiles` pools all finite values of all trials of the session, takes the 1/3 and 2/3 quantiles, and assigns classes 0/1/2; non-finite bins are assigned class 0. `output_values[3] = ['low','mid','high']`. The per-session thresholds are printed in the conversion log (`whisk_thr=(2.26, 5.80)` etc.). Nearly all sessions come out at 0.333/0.333/0.333, but the fallout of the NaN→0 rule is visible in the verification log: one session has fractions 0.814 / 0.093 / 0.093, i.e. ~72% of its bins had no usable camera data and were labelled "low".

ii.
```python
            whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```
```python
        y = np.zeros_like(x, dtype=np.int64)
        y[x > q1] = 1
        y[x > q2] = 2
        y[~np.isfinite(x)] = 0
```

iii. Same as 7-c: CONVERSION_NOTES.md Step 5 Key Decision 6 (session-aware quantile thresholds "to avoid degenerate classes") and Step 7 ("wheel/whisker bins are perfectly balanced due to tertile discretization"). The instructions require "Whisker motion energy discretized into 3 bins, time-varying".

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera trace is interpolated at `stimOn_times + TIME_CENTERS`, the centres of the same 100 bins used for the spikes, per trial. Camera frame times are on the same synchronised session clock as the spikes, so subtracting the trial's stimulus onset is the whole alignment. Column *k* of output channel 3 corresponds to column *k* of the neural matrix.

ii.
```python
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        ys = []
        for ts, me in streams:
            ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. CONVERSION_NOTES.md Step 5: "align/interpolate to 20 ms bins"; the `--show-processing` plot draws the whisker trace on the shared `TIME_CENTERS` axis together with the neural raster and the wheel trace.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms:
- **Per-session try/except**: any exception while processing a session prints `skip session due to error` and moves on, so one bad session cannot abort the run.
- **Whole-stream absence → drop the session**: if the trials table, the spike sorting, the wheel or the motion energy cannot be found, the session is skipped. This is where most of the data went: 238 + 63 + 1 of 461 sessions were dropped, and (as shown under 1-a) almost all of those files do exist — they were simply in a revision folder the globs do not search. The AI observed the skipping live and accepted it as genuinely missing data.
- **Bad per-trial values → drop the trial**: non-finite `stimOn_times`, `choice == 0`, or `probabilityLeft` not in {0.2, 0.5, 0.8}.
- **Missing behavioural samples → class 0**: NaN wheel/whisker values (bins outside the recorded stream, all-NaN trials, degenerate wheel traces) are assigned output class 0 rather than being excluded, so genuinely missing data becomes an ordinary "low" label for the decoder. There is no check that the wheel/camera/ephys streams actually cover a trial's window, so trials past the end of the recording survive — the verification log reports six trials in session 135 whose neural matrix is entirely zero.
- Two numerical edge cases were fixed after the first full run: duplicate wheel timestamps (which made `np.gradient` divide by zero) and all-NaN whisker averages (which made `np.nanmean` warn).

ii.
```python
        except Exception as e:
            print('skip session due to error', session_dir, repr(e))
```
```python
            me_streams = load_motion_energy(session_alf)
            if me_streams is None:
                print('skip no whisker motion energy', session_dir)
                continue
```
```python
    keep = np.isfinite(timestamps) & np.isfinite(position)
    ...
    uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
```
```python
        y[~np.isfinite(x)] = 0
```

iii. CONVERSION_NOTES.md Step 10: "initial full conversion had numerical RuntimeWarnings from duplicate wheel timestamps and all-NaN whisker averaging; patched conversion removed these warnings in rerun"; "Edge-case handling: added duplicate-timestamp handling for wheel traces and explicit all-NaN handling for whisker interpolation." Trajectory step 51 on the mass skipping: "the keep rate is not catastrophically low … the dataset will include a subset of sessions from labs with the required video-derived whisker signal, which is reasonable for the task." No justification is given for coding missing behaviour as class 0.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified and removed its original bottleneck: the first version loaded through `ONE`/`SpikeSortingLoader`, which pulled `spikes.samples.npy`, `spikes.templates.npy` and other arrays it never used (and re-downloaded them), taking ~27 s for 2 sessions; reading the needed `.npy`/`.pqt` files directly brought that to ~3 s. In the final script the dominant cost is reading `spikes.times.npy` + `spikes.clusters.npy` off disk (tens to hundreds of MB per probe), followed by the per-trial `np.add.at` binning loop (`np.add.at` is an unbuffered scatter-add and is roughly an order of magnitude slower than `np.bincount` on a flattened index, which is what the reference uses). The script prints a per-session wall time (`dt=…s`); kept sessions averaged 1.57 s (max 5.7 s) and the full run took 621 s single-threaded — within budget, but the reference achieves its throughput with a 10-worker process pool that the AI never added.

ii.
```python
        st = time.time()
        ...
            st = np.load(st_p)
            sc = np.load(sc_p).astype(int)
            metrics = pd.read_parquet(metrics_p)
        ...
            print(f'kept session {kept}: {session_dir} trials={len(neural_trials)} ... dt={time.time()-st:.2f}s')
    ...
    print(f'Saved {args.outpicklefile} with {len(out["neural"])} sessions in {time.time()-t0:.2f}s')
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: Initial version used ONE object loading and pulled unnecessary spike/template/waveform arrays, causing major slowdowns. Code speedups added: Replaced ONE object loading with direct local `.npy`/`.pqt` reads from session ALF directories, reducing sample conversion time from ~27 s to ~3 s for 2 sessions." Step 7: "Sample conversion ~0.7–1.2 s/session | Full dataset likely minutes to low tens of minutes."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Four:
- `bin_spikes` loops over trials and calls `np.add.at`; the whole session could be binned in one `np.bincount` over a flattened `(trial, unit, bin)` index, and even within the loop `np.bincount(cl*N_BINS + bins).reshape(...)` (the reference's approach) would be much faster than `np.add.at`.
- `interp_wheel_speed` and `interp_motion_energy` loop over trials calling `np.interp` on a 100-point query; one `np.interp` over the concatenated query vector of all trials would do the same work once.
- `trial_number_in_block` is a pure-Python loop over every trial; the two-line vectorised form is `block = (p != shift(p)).cumsum()` followed by `groupby(block).cumcount()`.
- The brain-region index is built with a Python loop and a dict lookup per neuron.

None of these are documented in CONVERSION_NOTES.md, and none were changed. The bigger missed win is that sessions are processed strictly sequentially; they are independent and the reference runs them in a 10-worker `ProcessPoolExecutor`.

ii.
```python
    for t0 in stim_on:
        ...
        np.add.at(out, (cl[m], bins[m]), 1)
```
```python
    for i, p in enumerate(prob_left):
        if i == 0 or p != cur:
```
```python
            for reg in clusters['acronym']:
                if reg not in region_to_idx:
                    region_to_idx[reg] = len(brain_regions)
                    brain_regions.append(str(reg))
                reg_idx.append(region_to_idx[reg])
```

iii. No justification is offered — the AI's only recorded efficiency work is the I/O change in Step 6. Its Step 7 runtime estimate ("minutes to low tens of minutes") was judged acceptable, so no further optimisation was pursued; the run did finish in 621 s, so the practical cost of these loops was small.

## 10-c. What processing does the code repeat multiple times?

i. Little of consequence. `find_latest_file` re-globs the `alf/` directory once per dataset per session (five separate globs) instead of listing it once. `interp_motion_energy` and `interp_wheel_speed` re-scan the full session-length time array for every trial (`np.interp` binary-searches the whole array each call) rather than slicing the window once. `discretize_tertiles` concatenates every finite value of the session into one array to compute two quantiles — fine, but it is done twice, once per behavioural stream. At the run level, the full conversion was executed twice (`conversion_full_out.txt` and `conversion_full_out_v2.txt`, `verification_full_out.txt` and `..._v2.txt`) after the wheel/whisker NaN fixes, which is the iteration protocol working as intended rather than a defect.

ii.
```python
    left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
    right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
    left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
    right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
```
```python
        for ts, me in streams:
            ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. Not discussed in CONVERSION_NOTES.md. The repeated full run is documented in Step 10 ("Validation rerun: reconverted full dataset and reran verify-only validation successfully").

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items, none documented:
- `from one.api import ONE` is imported at module scope but the client is never constructed — a leftover of the abandoned ONE-based loader that costs a slow import on every run.
- `get_subject_from_session` and `get_session_id` are defined but `get_session_id` is used only for the plot title and `get_subject_from_session` is never called; `lab`, `date` and `number` are unpacked from the path and never used.
- `load_trials_table` returns the parquet path, which is bound to `trial_path` and never used.
- `clusters.metrics.pqt` is read in full when only the `label` column is needed.
- `discretize_tertiles` returns the threshold pair, which is only printed.
- The outputs are stored as `int64` for four channels × 100 bins × 108,235 trials, where `int8` would suffice; together with storing the neural data per trial this makes `converted_data.pkl` 8.5 GB (the reference casts outputs to `int8`).
- Time-invariant quantities (choice, prior, trial-in-block) are replicated across all 100 bins, but that is required by the target format, not waste.

ii.
```python
from one.api import ONE          # never used
```
```python
def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]
```
```python
    trials_df, trial_path = load_trials_table(session_alf)   # trial_path unused
```
```python
                out_trial = np.vstack([
                    np.full(N_BINS, choice[tr], dtype=np.int64),
                    np.full(N_BINS, prior[tr], dtype=np.int64),
                    wheel_bins[i].astype(np.int64),
                    whisk_bins[i].astype(np.int64),
                ])
```

iii. Not discussed in CONVERSION_NOTES.md. The unused `ONE` import and helper functions are residue of the Step 6 rewrite from ONE-based to direct file loading; the `int64` outputs were never revisited despite the instructions' note to "Use appropriate data types (float32 vs. float64)" — the neural array *is* `float32`, so the AI applied the idea to the large array but not to the outputs.
