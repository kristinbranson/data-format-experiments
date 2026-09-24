# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the IBL ONE API at all. It walks the on-disk ONE cache directly with `pathlib.rglob`. A "session" is discovered by finding every `_ibl_trials.table.pqt` anywhere under the relative path `data/one_cache`, walking up to the enclosing `alf` directory, and taking its parent as the session root. Every later loader is a hand-written `np.load`/`pd.read_parquet` on a path found by globbing inside that session directory. Where several candidate files exist (the IBL cache stores revisions in `#YYYY-MM-DD#` sub-folders), the AI takes `sorted(...)[-1]` and comments that this "prefers the latest versioned file". No release index, no `DATALIMIT_SUBSET.csv`, and no dataset-completeness requirement: a session is accepted as long as a trials table and at least one `clusters.metrics.pqt` exist. This yielded 461 sessions / 141 subjects / 296,209 trials (the human reference yields 444 sessions / 136 subjects).

ii.
```python
def find_sessions(base=Path('data/one_cache')):
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    sessions = []
    for p in trial_tables:
        sess = p.parent
        while sess.name != 'alf' and sess != sess.parent:
            sess = sess.parent
        if sess.name == 'alf':
            sessions.append(sess.parent)
    out = sorted(set(sessions))
    return out


def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    if not trial_files:
        return None
    # prefer latest versioned file if multiple exist
    trial_file = trial_files[-1]
    return pd.read_parquet(trial_file)
```

```python
    sessions = find_sessions()
    if args.sample:
        sessions = sessions[:2]
    processed = []
    for i, sess in enumerate(sessions, 1):
        try:
            p = process_session(sess, show_processing=args.show_processing)
        except Exception as e:
            print(f'[WARN] failed session {sess}: {e}')
            p = None
```

iii. From CONVERSION_NOTES.md Step 2/Step 4: "Data are organized as an IBL ONE cache. Session data live under `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf` ... Consistent: use session as top-level unit in converted dataset." The trajectory shows the AI inspected the cache layout empirically (Steps 5–8) and then wrote its own readers. It never recorded a reason for bypassing the ONE API; the Step 1 notes are explicitly unfinished ("The IBL library *likely* provides session loaders, trial tables, wheel interpolation, and spike/cluster loading utilities that we should mirror rather than inventing custom semantics", followed by a "Need in the next step" list that is never resolved). The trajectory shows why: at Steps 3–4 the agent queued `sed`/`grep` commands over `/app/code/code_zhang2025/src/*.py`, but the terminal output was truncated and it never actually read the reference scripts before declaring Step 1 COMPLETE.

## 1-b. How are the data split into subjects (mice)?

i. The subject is parsed positionally out of the session directory path (`.../Subjects/<subject>/<date>/<number>` → `parts[-3]`). At assembly the unique subject strings are sorted and `subject_idx` records, per session, its index into that list. 141 subjects result.

ii.
```python
    subject = session_path.parts[-3]
    ...
    return {
        'session_id': '/'.join(session_path.parts[-4:]),
        'subject': subject,
        ...
    }
```

```python
    subjects = sorted({p['subject'] for p in processed})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    ...
        'subject_idx': np.array([subject_to_idx[p['subject']] for p in processed], dtype=np.int64),
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`subject/date/session path` → `subjects`, `subject_idx` — Extract subject IDs and session ordering — One session per converted session entry." The AI verified in Step 2 that the cache uses the `lab/Subjects/subject/date/number` layout, so the path position is a stable source of subject identity.

## 1-c. How are the data split into sessions?

i. One converted session per session directory in the cache, i.e. per `_ibl_trials.table.pqt` found. The session is the top-level unit of `neural`/`input`/`output`, and its identity is stored as the trailing four path components (`Subjects/<subject>/<date>/<number>`). No `session_info` block was added to `metadata`.

ii.
```python
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    ...
        if sess.name == 'alf':
            sessions.append(sess.parent)
    out = sorted(set(sessions))
```

```python
        'metadata': {
            'task_description': 'Decode choice, prior probability of left, wheel speed bin, and whisker motion-energy bin from stimulus-aligned neural activity.',
            'time_bin_size': bin_size * 1000.0,
            'temporal_alignment_event': 'stimulus onset',
            'off_start': -0.2,
            'off_end': 1.0,
        }
```

iii. CONVERSION_NOTES.md Step 4: "Session organization ... Consistent: use session as top-level unit in converted dataset." No further decision was needed — the cache is already organised one directory per session.

## 1-d. How are the data split into trials?

i. One row of `_ibl_trials.table.pqt` is one trial; the split is taken directly from the table. Trial rows are then subset by the quality mask (1-e) and `reset_index(drop=True)`.

ii.
```python
    trials = load_trials(session_path)
    if trials is None or len(trials) < 2:
        return None
    required = ['stimOn_times', 'choice', 'probabilityLeft']
    if any(c not in trials.columns for c in required):
        return None
    valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
    trials = trials.loc[valid].reset_index(drop=True)
```

iii. CONVERSION_NOTES.md Step 4: "Trial table contains choice, stimOn_times, probabilityLeft, contrasts, feedback, goCue and response times ... Consistent: probabilityLeft maps naturally to prior-probability output; stimOn_times supports required alignment." The trials table is already one row per trial, so nothing had to be derived.

## 1-e. How are trials filtered based on quality controls?

i. Three field-validity conditions plus one data-driven filter:
   1. `stimOn_times` not NaN (a trial with no alignment event cannot be aligned);
   2. `choice ∈ {-1, +1}` (drops no-response / no-go trials, which would otherwise produce an invalid third class in a binary output);
   3. `probabilityLeft` not NaN;
   4. after spike binning, any trial whose entire `(n_neurons, 60)` matrix is exactly zero is dropped.

   There is **no reaction-time window**, **no requirement that `probabilityLeft ∈ {0.2, 0.5, 0.8}`** (only `notna`), and **no check that the wheel or camera streams actually span the trial window**. Filter 4 was added late, in Step 10, purely to remove "all-zero neural trial" warnings from `train_decoder.py`. 296,209 of 297,505 trials survive (99.6%); the human reference, which applies an 80 ms–2 s reaction-time window and a wheel/camera coverage test, keeps far fewer.

ii.
```python
    valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
    trials = trials.loc[valid].reset_index(drop=True)
    if len(trials) < 2:
        return None
```

```python
    neural, edges = bin_spikes_for_trials(spike_times, spike_clusters, trials['stimOn_times'].to_numpy(), n_neurons)
    keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
    trials = trials.loc[keep_trial].reset_index(drop=True)
    neural = [m for m, k in zip(neural, keep_trial) if k]
    if len(neural) < 2:
        return None
```

iii. CONVERSION_NOTES.md Step 10/12 "Issues Found and Resolved": "Invalid binary choice coding from raw `choice==0` trials: resolved earlier by filtering to choice in {-1, 1}" and "Full-data verification warnings about all-zero neural trials: resolved by filtering trials whose binned neural activity was entirely zero." Step 3 admits the reference trial rules were never pinned down: "Need to confirm trial inclusion rules relevant to choice / prior / wheel / whisker outputs for our task." That confirmation never happened — `min_rt`/`max_rt`/`firstMovement_times` appear nowhere in the trajectory.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` per probe, gated by a per-cluster keep mask read from `clusters.metrics.pqt`. `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` are read only to build the region labels, not the activity matrix itself.

ii.
```python
        metrics = pd.read_parquet(cand[-1])
        chan_file = sorted(probe_dir.rglob('clusters.channels.npy'))[-1]
        clu_channels = np.load(chan_file)
        reg_id_files = sorted(probe_dir.rglob('channels.brainLocationIds_ccf_2017.npy'))
        reg_ids = np.load(reg_id_files[-1]) if reg_id_files else None
        spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
        spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`spikes.times.npy` + `spikes.clusters.npy` + curated cluster list → `neural` — Bin spike counts in fixed stimulus-aligned time bins for each trial; merge probes within session — Each trial becomes neuron x time matrix." Step 5 Key Decision 2: "Use binned spike counts rather than rates/dF/F because this is electrophysiology data and the reference code/papers decode from binned spikes."

## 2-b. How is the `neural` data processed?

i. Surviving clusters are renumbered contiguously, and the renumbering continues across probes so that the second probe's units are appended after the first probe's — one pooled population per session. Spikes are then counted into 20 ms bins over the trial window with `np.add.at`, giving an integer **spike count** per unit per bin, stored as `float32`. No conversion to Hz, no smoothing, no normalisation. Verified in the output pickle: values are exactly `{0, 1, 2, 3, 4}`. Merged probes are **not** re-sorted by spike time, which is harmless here because the binner masks the full array rather than slicing a sorted range.

ii.
```python
        kept_ids = np.where(keep)[0]
        if len(kept_ids) == 0:
            continue
        remap = {old: i + offset for i, old in enumerate(kept_ids)}
        mask = np.isin(spike_clusters, kept_ids)
        sc = spike_clusters[mask]
        st = spike_times[mask]
        sc = np.array([remap[c] for c in sc], dtype=np.int64)
        all_times.append(st)
        all_clusters.append(sc)
        ...
        offset += len(kept_ids)
```

```python
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    trial_mats = []
    for s in stim_on:
        rel = spike_times - s
        mask = (rel >= t0) & (rel < t1)
        rel = rel[mask]
        clu = spike_clusters[mask]
        mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
        if len(rel):
            tb = np.floor((rel - t0) / bin_size).astype(int)
            good = (tb >= 0) & (tb < mat.shape[1]) & (clu >= 0) & (clu < n_neurons)
            np.add.at(mat, (clu[good], tb[good]), 1)
        trial_mats.append(mat)
    return trial_mats, edges
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 4: "Merge neurons across probes within a session while preserving per-neuron brain region labels", justified in Step 4 by "Papers explicitly say neurons in same session and region were combined across probes." Step 3: "Probes within the same session and region were combined rather than decoded separately."

## 2-c. How is the `neural` data filtered based on quality controls?

i. A per-cluster boolean mask from `clusters.metrics.pqt`: `label >= 1` **and** `noise_cutoff < 20`, applied conditionally on the columns being present. `label` is the IBL single-unit QC score (0, 1/3, 2/3, 1) that aggregates the three RIGOR criteria, so `label >= 1` is the paper's "stringent QC"; the extra `noise_cutoff < 20` term is one of the three criteria already folded into `label` and is therefore redundant. There is **no** anatomical filter: units whose CCF location is 0 (outside the brain / unassigned) are kept — the converted data contains a `ccf_0` "region" with 250 neurons. Result: 75,808 units, against the data paper's 75,708 well-isolated neurons.

ii.
```python
        # provisional curation using available fields; amplitude criterion may require conversion/field interpretation refinement later
        keep = np.ones(len(metrics), dtype=bool)
        if 'noise_cutoff' in metrics.columns:
            keep &= metrics['noise_cutoff'].to_numpy() < 20
        if 'label' in metrics.columns:
            keep &= metrics['label'].to_numpy() >= 1
```

iii. CONVERSION_NOTES.md Step 3 "Neuron curation rules": "Exclude units failing any of the three criteria in the paper: amplitude > 50 uV, noise cut-off < 20 uV, and refractory period violation criterion (per RIGOR / ref. 28). Well-isolated neurons are the remaining units." Step 4 discrepancy row: "Papers explicitly require amplitude > 50 uV, noise cut-off < 20 uV ... 75,708 well-isolated from 621,733 total units. Need conversion code to implement paper curation rather than using all 622,377 raw clusters." Step 5 Key Decision 3: "Apply paper-compatible well-isolated neuron filtering using available cluster metrics before conversion." (The notes render `noise_cutoff` as a µV threshold, which is a misreading of a unitless metric, but the numeric threshold used is the IBL one.)

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is done by subtracting the trial's `stimOn_times` from every spike time; all IBL streams already share one session clock, so no further synchronisation is needed. The window taken around the event is the default argument of `bin_spikes_for_trials`: **`t0 = -0.2 s` to `t1 = +1.0 s`**, i.e. 60 bins, inclusive at the start and exclusive at the end (`rel >= t0 & rel < t1`). This window is never mentioned or justified anywhere in CONVERSION_NOTES.md, README.md, or the trajectory; the human reference uses -0.5 s to +1.5 s (100 bins), which is what the method paper specifies ("2-s trials, each divided into 20-ms bins, producing T = 100 time steps"). `off_start`/`off_end` in `metadata` correctly report -0.2/1.0.

ii.
```python
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    trial_mats = []
    for s in stim_on:
        rel = spike_times - s
        mask = (rel >= t0) & (rel < t1)
```

```python
            'temporal_alignment_event': 'stimulus onset',
            'off_start': -0.2,
            'off_end': 1.0,
```

iii. CONVERSION_NOTES.md Step 4: "Alignment — Paper decoding alignment depends on target variable; wheel example aligned to first movement. Trial table contains `stimOn_times` directly. User task explicitly requires temporal alignment to stimulus onset. Intentional task-specific difference: align all converted trials to stimulus onset while matching raw loading/filtering semantics otherwise." Step 5 Key Decision 1 repeats this. The *event* is justified; the *window length* is not.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins (`bin_size=0.02`), uniform across all trials and sessions, giving 60 timepoints per trial over the -0.2…1.0 s window. `metadata['time_bin_size']` is reported as 20.0 ms. Spikes are histogrammed directly onto this grid from raw spike times — there is no intermediate binning and therefore no rebinning or resampling of the neural data. The behavioural streams are not rebinned either; they are interpolated onto the same bin centres (see 7-d, 8-d). `train_decoder.py --verify-only` confirms "Mean T: 60.0" for every session.

ii.
```python
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    ...
            tb = np.floor((rel - t0) / bin_size).astype(int)
```

```python
def build_dataset(processed, bin_size=0.02):
    ...
            'time_bin_size': bin_size * 1000.0,
```

iii. CONVERSION_NOTES.md Step 3: "Paper states decoding time windows and alignment depend on target variable; for wheel they used 20 ms nonoverlapping bins around first wheel movement", and Step 4: "Papers describe wheel speed/velocity decoding from 20 ms bins." So the 20 ms figure was taken from the papers.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable at all — it is the fixed bin-centre grid of the alignment window, identical for every trial of every session, implied by `stimOn_times` being the alignment event. The values run from -0.19 s to +0.99 s in 0.02 s steps (verified in the output pickle), and the input is named `time_since_stimulus_onset`.

ii.
```python
    centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

```python
        'input_names': ['time_since_stimulus_onset', 'trial_number_in_block'],
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`_ibl_trials.stimOn_times` / `stimOn_times` column → `input[0]` — Convert to time-since-stimulus-onset representation on a common trial time grid — Because trials are aligned to stimulus onset, this can be represented as the common per-bin relative time values."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond averaging adjacent bin edges to get centres and casting to `float32`. The same 60-element vector is stacked as row 0 of every trial's input array (a continuous, time-varying input, not a binary event indicator).

ii.
```python
    centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
    ...
    for i in range(len(trials)):
        inp = np.vstack([
            centers,
            np.full_like(centers, block_trial[i], dtype=np.float32),
        ]).astype(np.float32)
```

iii. Same as 3-a — the variable is defined by the conversion's own alignment grid, so there is nothing to process. The AI's Step 10 sanity check recorded: "input time vector matched expected stimulus-aligned bin centers exactly".

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. By construction: `centers` is computed from the very `edges` array returned by `bin_spikes_for_trials`, so input bin *k* is the centre of the neural bin *k*. `edges` is threaded out of the binner and back into both the input builder and the behavioural interpolators, so all four streams share one grid.

ii.
```python
    neural, edges = bin_spikes_for_trials(spike_times, spike_clusters, trials['stimOn_times'].to_numpy(), n_neurons)
    ...
    centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
```

iii. Not separately justified; the AI's Step 10 check ("input time vector matched expected stimulus-aligned bin centers exactly") is its evidence that the grid is shared.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From the `probabilityLeft` column of the trials table only. `probabilityLeft` is constant within a block, so a change of value is taken to start a new block. The trials table carries no explicit block identifier.

ii.
```python
    block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "trial index within block from `probabilityLeft` runs → `input[1]` ... Need to compute trial number in current block from `probabilityLeft` changes", and Key Decision 7: "Compute trial number within block from consecutive runs of constant `probabilityLeft`."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop over the `probabilityLeft` array that resets a counter to 1 whenever the value changes and increments it otherwise, so the value is the 1-based position of the trial within its block. The scalar is broadcast across all 60 bins as row 1 of the trial's input, as a raw un-normalised `float32` count (observed range 1…~100, matching IBL's 20–100-trial blocks).

   Critically, the count is computed **after** the trial quality mask *and* after the all-zero-neural mask have already removed rows from `trials`, so dropped trials do not advance the counter and the value understates the animal's true position in the block. The human reference computes the count on the unfiltered table for exactly this reason.

ii.
```python
def compute_trial_number_in_block(prob_left):
    prob_left = np.asarray(prob_left)
    out = np.zeros(len(prob_left), dtype=np.float32)
    c = 0
    prev = None
    for i, v in enumerate(prob_left):
        if i == 0 or v != prev:
            c = 1
            prev = v
        else:
            c += 1
        out[i] = c
    return out
```

```python
    trials = trials.loc[keep_trial].reset_index(drop=True)
    neural = [m for m, k in zip(neural, keep_trial) if k]
    if len(neural) < 2:
        return None
    centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
    block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. Only the Step 5 Key Decision 7 quoted above ("consecutive runs of constant `probabilityLeft`"). The ordering relative to filtering is never discussed in CONVERSION_NOTES.md or the trajectory.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column of the trials table, which is IBL-coded as +1 (leftward wheel turn), -1 (rightward) and 0 (no response). Recoded to left = 0, right = 1 as the instructions require; the 0 entries are removed with the trial in the 1-e mask, so the `-1` sentinel in `map_choice` is never reached. Verified class balance in the converted data: 50.6% / 49.4%.

ii.
```python
def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    # IBL convention often left=1 right=-1, but confirm from data later; current mapping assumes left=1 -> 0, right=-1 -> 1
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[arr == 1] = 0
    out[arr == -1] = 1
    return out
```

```python
    choice = map_choice(trials['choice'].to_numpy())
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`choice` column → `output[0]` — Map left->0, right->1 using IBL choice coding after confirming sign convention." Step 10 sanity check: "choice mapping matched raw trial table for sampled trial." Step 12: "Invalid binary choice coding from raw `choice==0` trials: resolved earlier by filtering to choice in {-1, 1}."

## 5-b. What processing is involved in computing `output` *Choice*?

i. None beyond the recoding. The per-trial scalar is broadcast over all 60 bins so the output is formally time-varying, as the target format prefers, and cast to `int64`.

ii.
```python
        out = np.vstack([
            np.full_like(centers, choice[i], dtype=np.int64),
            np.full_like(centers, prior[i], dtype=np.int64),
            discretize_with_thresholds(wheel_trials[i], wheel_q1, wheel_q2),
            discretize_with_thresholds(me_trials[i], me_q1, me_q2),
        ])
```

```python
        'output_values': [
            ['left', 'right'],
            ...
        ],
```

iii. CONVERSION_NOTES.md Step 5 mapping table calls it a "Per-trial categorical output"; the instructions' "If at all possible, make it time-varying" drove the broadcast.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, recoded 0.2 → 0, 0.5 → 1, 0.8 → 2 via a dict lookup, with a `-1` fallback for any other value. Trials are admitted on `probabilityLeft.notna()` rather than on membership in {0.2, 0.5, 0.8}, so the `-1` fallback is the only guard against an out-of-set prior; in practice no `-1` reaches the output (converted class fractions are 0.417 / 0.140 / 0.443 and sum to 1).

ii.
```python
def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

```python
    prior = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`probabilityLeft` column → `output[1]` — Map 0.2->0, 0.5->1, 0.8->2 — Per-trial categorical output", taken verbatim from the Decoder Task spec. Step 10 sanity check: "prior mapping matched raw `probabilityLeft`."

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. None beyond the recoding; the per-trial scalar is broadcast across all 60 bins, exactly as for choice.

ii.
```python
            np.full_like(centers, prior[i], dtype=np.int64),
```

```python
        'output_values': [
            ['left', 'right'],
            ['0.2', '0.5', '0.8'],
            ...
        ],
```

iii. Same as 6-a — the mapping is dictated by the Decoder Task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. Unlike every other loader in the script, `load_wheel` looks for these at the **exact** path `<session>/alf/<name>` rather than with a recursive `rglob`, so a session whose wheel datasets sit in a `#revision#` sub-folder is treated as having no wheel at all. When either file is absent the function returns `(None, None)` and the caller substitutes an array of zeros for every trial. In the delivered `converted_data.pkl` this path was taken for **63 of 461 sessions (37,427 trials, 12 subjects)**, whose `wheel_speed` output is the constant class 0 for every bin of every trial.

ii.
```python
def load_wheel(session_path: Path):
    alf = session_path / 'alf'
    posf = alf / '_ibl_wheel.position.npy'
    tsf = alf / '_ibl_wheel.timestamps.npy'
    if not (posf.exists() and tsf.exists()):
        return None, None
    return np.load(posf), np.load(tsf)
```

```python
    else:
        wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`_ibl_wheel.position.npy` + `_ibl_wheel.timestamps.npy` → `output[2]` — Differentiate/interpolate to stimulus-aligned bins, convert to speed, discretize into 3 bins — Time-varying categorical output." Step 4 records "Raw wheel position/timestamps are present for 461 sessions", i.e. the AI believed every session had wheel data. The zero-fill fallback is never mentioned in the notes.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps on the raw encoder trace: sort by timestamp; drop duplicate/non-increasing timestamps; take a plain first-order finite difference `|Δposition / Δt|` between consecutive raw samples; attribute each value to the midpoint time of its interval. The resulting midpoint series is then linearly interpolated onto the trial bin centres. There is **no** interpolation of position onto a regular grid and **no** low-pass filter, i.e. neither of the two steps `brainbox.io.one.SessionLoader.load_wheel` performs by default (`interpolate_position` at 1000 Hz followed by `velocity_filtered` with a 20 Hz Butterworth), which is what the human reference relies on. Because IBL logs the wheel only on encoder ticks, the raw difference quotient is a spiky, unfiltered estimate and long stationary gaps become a single small value linearly ramped across the gap.

ii.
```python
    if wheel_pos is not None:
        order = np.argsort(wheel_ts)
        wheel_ts = np.asarray(wheel_ts)[order]
        wheel_pos = np.asarray(wheel_pos)[order]
        uniq_mask = np.concatenate([[True], np.diff(wheel_ts) > 0])
        wheel_ts = wheel_ts[uniq_mask]
        wheel_pos = wheel_pos[uniq_mask]
        if len(wheel_ts) >= 2:
            dt = np.diff(wheel_ts)
            dp = np.diff(wheel_pos)
            speed_mid = np.abs(dp / dt).astype(np.float32)
            ts_mid = ((wheel_ts[:-1] + wheel_ts[1:]) / 2).astype(np.float64)
            wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 5: "Derive wheel speed from raw wheel position timestamps, then discretize into 3 bins for categorical decoding as required by the task." Step 10/12 "Issues Found and Resolved": "Wheel-speed RuntimeWarnings from duplicate/non-monotonic wheel timestamps: resolved by sorting timestamps, removing duplicates, computing finite-difference speed, and interpolating midpoint speeds." So the sort/dedupe was a fix for divide-by-zero warnings, not a considered signal-processing choice; the absence of the IBL filtering pipeline is never discussed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Into 3 equal-occupancy classes using the 1/3 and 2/3 quantiles computed **once per session** over the concatenation of every trial's resampled speed trace (non-finite values excluded). A value is class 0 if `<= q1`, class 1 if `> q1`, class 2 if `> q2`. If nothing finite is available the thresholds default to (0.0, 0.0). This session-level equal-tertile rule is the same as the human reference's. Across the whole dataset the classes come out at 41.8% / 29.1% / 29.1% — the excess of class 0 is entirely contributed by the 63 zero-filled sessions, where `q1 = q2 = 0` forces every bin to class 0.

ii.
```python
def tertile_thresholds(arrays):
    x = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None and len(a) > 0])
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return 0.0, 0.0
    q1, q2 = np.quantile(x[finite], [1/3, 2/3])
    return float(q1), float(q2)

def discretize_with_thresholds(x, q1, q2):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=np.int64)
    y[x > q1] = 1
    y[x > q2] = 2
    return y
```

```python
    wheel_q1, wheel_q2 = tertile_thresholds(wheel_trials)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 5 ("discretize into 3 bins ... as required by the task"). The trajectory shows the AI reconsidering the scope of the quantile at Step 49: "wheel and whisker outputs are exactly balanced because discretization is currently per-trial tertiles, which may be acceptable format-wise but is potentially questionable scientifically" — the delivered code pools across the whole session, which is the corrected version.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The midpoint speed series is evaluated with `np.interp` at `stimOn_times + centers`, i.e. the same 60 bin centres, measured from the same stimulus onset, as the neural matrix — bin *k* of the wheel row describes the same instant as bin *k* of the neural matrix. IBL streams already share a session clock, so subtraction of the onset is the whole of the alignment. Note `np.interp` clamps outside the data range rather than returning NaN, and there is no coverage test, so a trial whose window extends past the end of the wheel recording silently receives a flat extrapolated value.

ii.
```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
    out = []
    for s in stim_on:
        t = s + centers
        y = np.interp(t, timestamps, values)
        out.append(y.astype(np.float32))
    return out
```

```python
            wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
```

iii. CONVERSION_NOTES.md Step 5 mapping table ("interpolate to stimulus-aligned bins"). Step 7: "Plots show consistent stimulus-aligned trial grids and categorical outputs over time"; Step 12: "wheel_speed 0.6436 validation balanced accuracy — well above chance (0.333); indicates successful temporal alignment of wheel output."

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `leftCamera.ROIMotionEnergy.npy` / `rightCamera.ROIMotionEnergy.npy` with their frame times `_ibl_leftCamera.times.npy` / `_ibl_rightCamera.times.npy`, found recursively (so revision folders are picked up here, unlike the wheel). When **both** side cameras are present the AI interpolates both and takes their **arithmetic mean**; when one is present it uses that one; when neither is present it substitutes zeros. In the delivered dataset 16 of 461 sessions (10,081 trials) ended up on the zero-fill path and carry a constant `whisker_motion_energy` class. The human reference instead picks a single camera, preferring left.

ii.
```python
def load_motion_energy(session_path: Path):
    alf = session_path / 'alf'
    left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
    right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
    left_t = sorted(alf.rglob('_ibl_leftCamera.times.npy'))
    right_t = sorted(alf.rglob('_ibl_rightCamera.times.npy'))
    streams = []
    if left_me and left_t:
        streams.append((np.load(left_me[-1]), np.load(left_t[-1]), 'left'))
    if right_me and right_t:
        streams.append((np.load(right_me[-1]), np.load(right_t[-1]), 'right'))
    return streams
```

```python
        if len(aligned) == 1:
            me_trials = aligned[0]
        else:
            me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "Use ROI motion energy as whisker motion energy; if both left and right are available, choose a consistent rule (for example mean or preferred available side) and document it." Step 4: "Papers define whisker-pad motion energy from left/right videos at camera temporal resolution ... use ROI motion energy as whisker motion energy, likely choosing available side or combining sides sensibly." The mean was chosen but the choice between "mean" and "preferred side" is never actually argued, and the differing frame rates and image scales of the two IBL side cameras are not considered.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released ROI motion-energy trace is used as-is — no filtering, no normalisation, no baseline subtraction. It is cast to `float32`, linearly interpolated onto the 60 trial bin centres, optionally averaged across the two cameras (8-a), and then tertile-discretised (8-c).

ii.
```python
    me_streams = load_motion_energy(session_path)
    if me_streams:
        aligned = []
        for vals, ts, side in me_streams:
            aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`left/rightCamera.ROIMotionEnergy.npy` + camera times → `output[3]` — Interpolate to stimulus-aligned bins; choose whisker-related ROI motion energy stream and discretize into 3 bins — Time-varying categorical output." Step 3 notes the IBL side-camera ROI is already the whisker pad, so no further extraction was needed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to the wheel: `tertile_thresholds` over the concatenation of all of the session's resampled traces gives the 1/3 and 2/3 quantiles, and `discretize_with_thresholds` maps `<= q1 → 0`, `> q1 → 1`, `> q2 → 2`. Over the whole dataset the classes land at 35.6% / 32.2% / 32.2%, the class-0 excess again coming from the 16 zero-filled sessions.

ii.
```python
    me_q1, me_q2 = tertile_thresholds(me_trials)
    ...
            discretize_with_thresholds(me_trials[i], me_q1, me_q2),
```

iii. Same as 7-c — the Decoder Task specifies "discretized into 3 bins", and the AI applies one common equal-occupancy rule to both time-varying behavioural outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Through the same `interp_to_trial_bins` call: the camera trace is evaluated at `stimOn_times + centers`, so it shares the neural time axis bin for bin. Camera frame times are already on the session clock, so subtracting the onset is all that is required. As with the wheel, `np.interp` clamps at the edges and there is no coverage check.

ii.
```python
            aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

```python
    for s in stim_on:
        t = s + centers
        y = np.interp(t, timestamps, values)
```

iii. CONVERSION_NOTES.md Step 12: "whisker_motion_energy 0.6024 validation balanced accuracy — well above chance (0.333); indicates successful temporal alignment of motion-energy output." Step 7 processing plots were the visual check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms, of differing quality:
   - **Missing columns / too few trials / no surviving clusters** → `process_session` returns `None` and the session is skipped. In the full run this never fired: all 461 sessions were kept.
   - **Missing wheel or camera streams** → **silently replaced by an array of zeros for every trial**, and the session is still emitted. This is the single most damaging choice: 63 sessions (37,427 trials) have a constant `wheel_speed` and 16 sessions (10,081 trials) a constant `whisker_motion_energy`. These sessions are reported as successful conversions and contribute unlearnable, degenerate labels to the decoder.
   - **Duplicate / non-monotonic wheel timestamps** → sorted and de-duplicated before differencing.
   - **Any other exception** → caught by a blanket `except Exception` in `main`, printed as `[WARN]`, and the session dropped. No `[WARN]` lines appear in `conversion_full_out.txt`.

   There is no check that the wheel or camera actually spans a trial window (the human reference's `covered()` test), and `np.interp` extrapolates flat rather than flagging the gap, so out-of-range trials are silently filled with a held constant. `map_prior`'s `-1` fallback and `map_choice`'s `-1` fallback are silent sentinels that would become invalid class labels rather than errors.

ii.
```python
    wheel_pos, wheel_ts = load_wheel(session_path)
    if wheel_pos is not None:
        ...
    else:
        wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
    me_streams = load_motion_energy(session_path)
    if me_streams:
        ...
    else:
        me_trials = [np.zeros_like(centers) for _ in range(len(trials))]
```

```python
        try:
            p = process_session(sess, show_processing=args.show_processing)
        except Exception as e:
            print(f'[WARN] failed session {sess}: {e}')
            p = None
```

iii. CONVERSION_NOTES.md Step 10/12 "Issues Found and Resolved" lists only the three problems the AI actually noticed (wheel timestamp warnings, all-zero neural trials, `choice == 0`). It then concludes in Step 12: "All outputs are comfortably above chance, and train-vs-validation gaps are modest, so no immediate evidence of severe bugs, leakage, or catastrophic misalignment remains", and in Step 9: "`verification_full_out.txt`: created and clean (no errors or warnings)". The degenerate sessions are not mentioned anywhere — they pass `train_decoder.py`'s format checks precisely because class 0 is a legal value.

## 10-a. What are the most time-consuming steps of the code?

i. The AI performed no profiling and left every Step 6/Step 7 timing table in CONVERSION_NOTES.md blank ("Code inefficiencies identified: [Note]", "Code speedups added: [Note]", empty "Run Time Estimates" tables). The only instrumentation is a per-session wall-clock print. The full conversion took **14,653 s (4 h 04 min)**, against the instructions' 15-minute target; the trajectory shows the AI estimated "1.5–2 hours if unoptimized, which exceeds the 15 minute guidance" and then started the full run anyway without optimising.

   The actual dominant costs are (a) `bin_spikes_for_trials`, which for **each** trial subtracts the onset from and masks the session's **entire** spike array (tens of millions of elements), making it O(n_trials × n_spikes) — roughly 500–1000 full passes over a multi-hundred-MB array per session; (b) the Python-level cluster remapping `np.array([remap[c] for c in sc])`, one dict lookup per surviving spike; and (c) single-process execution, where the reference runs 10 sessions in parallel.

ii.
```python
    for s in stim_on:
        rel = spike_times - s
        mask = (rel >= t0) & (rel < t1)
        rel = rel[mask]
        clu = spike_clusters[mask]
```

```python
        sc = np.array([remap[c] for c in sc], dtype=np.int64)
```

```python
    for i, sess in enumerate(sessions, 1):
        st = time.time()
        ...
        print(f'processed {i}/{len(sessions)} sessions; kept {len(processed)}; dt={time.time()-st:.2f}s')
```

iii. No justification is recorded. CONVERSION_NOTES.md Step 6 and Step 7 are the sections where the instructions required an inefficiency analysis and a runtime estimate, and both were submitted with the template placeholders unfilled.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI identified none. The genuinely vectorizable loops are:
   - `bin_spikes_for_trials`'s per-trial loop: `np.searchsorted` on the (sortable) spike times would slice each trial's window in O(log n) instead of rescanning the whole array, and a single flat `np.bincount` over `unit * n_bins + bin` could fill all trials at once — this is exactly what the human reference does.
   - `sc = np.array([remap[c] for c in sc])`: replaceable by a lookup array `new_index = np.cumsum(keep) - 1; new_index[sc]`.
   - `np.isin(spike_clusters, kept_ids)`: replaceable by indexing the boolean `keep` mask directly with `spike_clusters`.
   - `interp_to_trial_bins`'s per-trial loop: one `np.interp` over the concatenated query vector `stim_on[:, None] + centers` would do all trials in one call.
   - `keep_trial = np.array([np.any(m != 0) for m in neural])` and the final per-trial input/output assembly loop, both of which could be a single stacked operation.
   - `main`'s session loop, which could use a `ProcessPoolExecutor`.

ii.
```python
    for s in stim_on:
        rel = spike_times - s
```

```python
    for i in range(len(trials)):
        inp = np.vstack([...])
        out = np.vstack([...])
        inputs.append(inp)
        outputs.append(out)
        neural[i] = neural[i].astype(np.float32)
```

iii. None given. The instructions' Step 6 ("Vectorize loops ... Use parallel processing if beneficial") and Step 7 ("If full conversion time estimate is longer than 15 minutes, speed up the code") were acknowledged in the trajectory but not acted on.

## 10-c. What processing does the code repeat multiple times?

i. The AI identified none. In fact the script repeats several things:
   - `trials['stimOn_times'].to_numpy()` is recomputed three separate times per session (spike binning, wheel interpolation, motion-energy interpolation).
   - `sorted(probe_dir.rglob(...))` is called five times per probe — five independent recursive directory walks where one listing would do.
   - `centers` is recomputed inside `interp_to_trial_bins` on every call even though the caller already holds it, and `np.zeros_like(centers)` is allocated once per trial in the fallback paths.
   - `edges`/`centers` are recomputed per session although they are global constants.
   - When both cameras exist, the full per-trial interpolation is run twice and the results immediately collapsed by averaging.
   - `region_names.index(reg)` is a linear scan inside the per-cluster loop, making region assignment quadratic in the number of distinct regions per probe.

ii.
```python
    neural, edges = bin_spikes_for_trials(spike_times, spike_clusters, trials['stimOn_times'].to_numpy(), n_neurons)
    ...
            wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
    ...
            aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

```python
        for cid in kept_ids:
            ch = int(clu_channels[cid])
            reg = f'ccf_{int(reg_ids[ch])}' if reg_ids is not None and ch < len(reg_ids) else f'channel_{ch}'
            if reg not in region_names:
                region_names.append(reg)
            neuron_region_idx.append(region_names.index(reg))
```

iii. No justification recorded — CONVERSION_NOTES.md Step 6 "Code inefficiencies identified" was left as the literal placeholder `[Note]`.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI identified none. The script does perform some work whose result is thrown away:
   - Spike binning is done for **all** trials and only afterwards are trials dropped by `keep_trial`, so the binned matrices of dropped trials are discarded.
   - `pks = sorted(probe_dir.rglob('pykilosort'))` is computed (a recursive directory walk) and then never used.
   - `interp_to_trial_bins` takes a `reducer='linear'` parameter that is never read, and `load_motion_energy` returns a `side` label that is never used.
   - `from collections import defaultdict` is imported and unused.
   - When both cameras exist, two complete interpolated traces are produced and then averaged into one, so half of that computation does not survive as an independent signal.

   The volume of this waste is small next to the O(n_trials × n_spikes) binning loop, so removing it would not have brought the 4-hour runtime down materially.

ii.
```python
        pks = sorted(probe_dir.rglob('pykilosort'))
        if not pks:
            pks = [probe_dir]
```

```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
```

```python
    neural, edges = bin_spikes_for_trials(spike_times, spike_clusters, trials['stimOn_times'].to_numpy(), n_neurons)
    keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
    neural = [m for m, k in zip(neural, keep_trial) if k]
```

iii. No justification recorded. The `keep_trial` ordering follows from the filter being introduced retroactively in Step 10 to silence verification warnings ("Full-data verification warnings about all-zero neural trials: resolved by filtering trials whose binned neural activity was entirely zero") rather than being designed into the trial mask up front.
