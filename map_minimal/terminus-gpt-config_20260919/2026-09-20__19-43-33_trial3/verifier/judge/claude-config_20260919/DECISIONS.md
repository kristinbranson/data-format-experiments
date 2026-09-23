# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the DANDI release as one NWB file per session. It recursively globs `/app/data` for `*.nwb`, sorts the paths (giving subject-major, chronological order), and opens each file once with `pynwb`, streaming one session at a time (with an explicit `gc.collect()` after each) so that only one session is in memory besides the accumulating output. Within a file it reads: the trials table (`nwb.trials`), the behavioural event streams (`nwb.acquisition['BehavioralEvents'].time_series`), the unit table columns (`unit_quality`, `anno_name`, `spike_times`), the side-camera tongue tracking series, and `nwb.subject.subject_id`. Spike times are pulled as one bulk read of the ragged `VectorData`/`VectorIndex` pair rather than one HDF5 read per unit (an optimisation the AI applied after profiling). All 174 files are processed; 173 reach the output.

ii.
```python
DATA_ROOT = Path('/app/data')
...
files = sorted(DATA_ROOT.rglob('*.nwb'))
...
for si, path in enumerate(files):
    print(f'[{si+1}/{len(files)}] {path.name}', flush=True)
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        sid = str(nwb.subject.subject_id)
        tr = nwb.trials
        ...
        events = nwb.acquisition['BehavioralEvents'].time_series
```
```python
            # Bulk-load NWB's ragged spike vector and offsets once.
            spike_col = nwb.units['spike_times']
            spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
            spike_ends = np.asarray(spike_col.data[:], dtype=np.int64)
            spike_starts = np.r_[0, spike_ends[:-1]]
            spikes = [spike_data[spike_starts[i]:spike_ends[i]] for i in kept_idx]
```

iii. From the trajectory: the AI first inventoried the data layout and NWB schema with compact `pynwb` reports ("The dataset has 174 NWB sessions, 94,990 trials, and all sessions contain tongue tracking"), then decided to "stream one NWB at a time to control memory" because the source is ~50 GB. Bulk ragged-array loading was adopted after two profiling rounds: "Per-unit HDF5 access is likely now the main cost. Bulk-loading the concatenated `spike_times` dataset and its ragged index once per session should substantially reduce I/O overhead."

## 1-b. How are the data split into subjects?

i. One subject per NWB file. The list of subjects is built up-front from the sorted file names (`sub-<id>_ses-...nwb` → `<id>`), giving 28 sorted numeric ids; each session's index into that list is looked up from the NWB metadata field `nwb.subject.subject_id`, so the filename-derived list and the in-file id must agree (they do — a mismatch would raise `KeyError`).

ii.
```python
files = sorted(DATA_ROOT.rglob('*.nwb'))
subjects = sorted({p.name.split('_')[0].replace('sub-', '') for p in files})
subject_to_idx = {s:i for i,s in enumerate(subjects)}
```
```python
            sid = str(nwb.subject.subject_id)
            ...
            subject_idx.append(subject_to_idx[sid])
```
```python
        'subjects': subjects, 'subject_idx': np.asarray(subject_idx, dtype=np.int32),
```

iii. The AI's early inspection established the `sub-*/` directory + filename convention of the dandiset and that each file carries a `subject` object; it therefore treated the numeric DANDI subject id as the animal identifier and did not attempt to map it back to the mouse names (e.g. `SC015`) used in the papers. Result: 28 subjects, matching the dandiset summary.

## 1-c. How are the data split into sessions?

i. No splitting is performed: one NWB file is one session, and sessions are emitted in sorted-filename order (subject-major, chronological within subject). A session is dropped only if it contains no unit passing the AI's unit filter; exactly one session (`sub-440958_ses-20190216T162508`) is dropped for this reason, leaving 173 sessions. Per-session provenance is stored in `metadata['session_info']` as the file name plus subject, trial count, neuron count and the session's tongue percentile thresholds.

ii.
```python
            if len(kept_idx) == 0:
                print('  skipped: no good annotated units', flush=True)
                continue
```
```python
            session_info.append({'file': path.name, 'subject': sid, 'n_trials': ntr,
                                 'n_neurons': len(kept_idx), 'tongue_q40': float(q40),
                                 'tongue_q60': float(q60), 'dlc_likelihood_threshold': DLC_THRESHOLD})
```

iii. The AI noted that the paper reports "173 sessions / 69,943 good units" while the dandiset contains 174 files, and searched the repo and `dandiset.yaml` for an explicit exclusion list. Finding none, it decided to "retain all available sessions unless an explicit exclusion is documented" — the 174→173 reduction then fell out of its unit filter, since the one unannotated/uncurated session has no qualifying units.

## 1-d. How are the data split into trials?

i. Trials are the rows of the NWB `trials` table; every row of every retained session becomes a trial (94,370 trials over 173 sessions). Rather than assuming that the *i*-th go-cue event belongs to the *i*-th trial, the AI assigns events to trials by interval membership: for each trial `[start_time, stop_time)` it takes the first `go_start_times` (and first `sample_start_times`) timestamp inside it, and hard-fails if any trial has no go cue or no tone.

ii.
```python
def first_event_in_trial(times, starts, stops):
    """First event in each NWB trial, NaN if absent."""
    times = np.asarray(times, dtype=np.float64)
    ans = np.full(len(starts), np.nan)
    j = 0
    for i, (a, b) in enumerate(zip(starts, stops)):
        j = np.searchsorted(times, a, side='left')
        if j < len(times) and times[j] < b:
            ans[i] = times[j]
    return ans
```
```python
            go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
            if np.any(~np.isfinite(go)):
                raise ValueError(f'{path}: {np.sum(~np.isfinite(go))} trials lack go cue')
```

iii. From the trajectory: "Go events are one timestamp per trial. Tone/sample events can include extra transitions, so tone onset should be assigned as the first sample-start timestamp within each trial", and "One go onset is expected in every retained trial. Assign by trial interval rather than relying on array position." The conversion ran to completion without raising, confirming exactly one go cue per trial in all 174 files.

## 1-e. How are trials filtered based on quality controls?

i. **No trial filtering at all.** Every row of the trials table is retained, subject only to the requirement (enforced as a hard error, never triggered) that it contain a go cue and a tone. The AI explicitly documents this as a decision ("all released behavioral sessions and trials are retained", `metadata['trial_filter'] = 'all trials with go cue and tone onset'`). It considered the per-unit × per-trial `is_good_trials` mask that NWB provides, inspected it on a representative session, and concluded no trial masking was needed. It did **not** consult `units/obs_intervals` (which marks which trials the ephys actually covers) and did not exclude `free_water` trials. As a consequence 3,511 retained trials contain no spikes at all from any neuron; the supplied verifier emitted a warning for each one (`'Session N, trial M: all neural data is zero'`, 3,511 warnings, up to 376 consecutive leading trials in one session), and the AI did not act on them.

ii.
```python
            starts = np.asarray(tr['start_time'][:], float)
            stops = np.asarray(tr['stop_time'][:], float)
            ntr = len(starts)
            ...
            sess_n, sess_i, sess_o = [], [], []
            for ti in range(ntr):
```
```python
            'trial_filter': 'all trials with go cue and tone onset',
```
(there is no `keep` mask anywhere in the script; `obs_intervals` and `free_water` are never read)

iii. Justification in the trajectory: "In the representative session, all selected classifier-good, histologically annotated units are valid on every trial, so no trial-specific masking is needed there" and "Labels are complete and balanced enough for decoding". Note that the AI deliberately kept early-lick and no-response trials — correct here, since both are required decoder outputs — but it never looked for trials lacking ephys coverage. After the verification run it reported only "Verification succeeded" and addressed a different warning class (the `nan` brain region), leaving the 3,511 all-zero-trial warnings unexamined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `units/spike_times` (the ragged per-unit sorted spike-time vector, in session-absolute seconds), restricted to the units passing the filter in 2-c, together with the per-trial go-cue times from `BehavioralEvents/go_start_times` which position the bin edges. `units/unit_quality` and `units/anno_name` select the units; `units/anno_name` also supplies `brain_regions` (the full CCF string, e.g. "Agranular insular area, dorsal part, layer 5", yielding 292 region labels).

ii.
```python
            spike_col = nwb.units['spike_times']
            spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
            spike_ends = np.asarray(spike_col.data[:], dtype=np.int64)
            spike_starts = np.r_[0, spike_ends[:-1]]
            spikes = [spike_data[spike_starts[i]:spike_ends[i]] for i in kept_idx]
```
```python
                sess_n.append(bin_spikes(spikes, go[ti]))
```

iii. Spike times are the only neural representation in the file; the AI's schema survey identified `spike_times`, `obs_intervals`, `is_good_trials`, `anno_name` and the quality-metric columns, and it concluded that "per-unit `anno_name` supplies histological region labels and matches the reference code's CCF annotation requirement", whereas the electrode `location` field "is only insertion-level (e.g. ALM)".

## 2-b. How is the `neural` data processed?

i. Spike counts per bin are converted to firing rates in Hz. For each trial, absolute bin edges are formed as `go + EDGES` (81 edges), and for each unit `np.searchsorted(spike_train, abs_edges, side='left')` gives the running spike total at each edge; differencing gives the per-bin count, which is divided by the 50 ms bin width. Bins are non-overlapping and half-open `[left, right)`. No smoothing, normalisation, baseline subtraction or trial-averaging is applied. Output dtype is `float32`.

ii.
```python
def bin_spikes(spike_times, go):
    # Histogram absolute spike times with the same [left,right) convention as
    # the reference sliding_histogram, then convert counts to Hz.
    out = np.empty((len(spike_times), 80), dtype=np.float32)
    abs_edges = go + EDGES
    for u, st in enumerate(spike_times):
        st = np.asarray(st)
        # NWB spike trains are sorted. Edge insertion indices give the same
        # [left,right) counts as the reference code without rescanning the
        # full session spike train once per bin.
        out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
    return out
```

iii. The AI read `sliding_histogram` in `/app/code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` and replicated its semantics: "The firing-rate section confirms reference semantics: spike times relative to go cue are clipped, then passed to `sliding_histogram` with rate output. The desired 50 ms non-overlapping bins imply bin width and stride both 0.05 s", and "The reference histogram uses half-open bins and firing rates". The `searchsorted` formulation was adopted purely as a speed optimisation over `np.histogram`, with the comment above asserting it preserves the same half-open convention.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A unit is kept if `units/unit_quality == 'good'` **and** its `anno_name` is a non-empty string that is not the literal `'nan'` (i.e. it has a CCF/histology annotation). No individual quality metric (ISI violations, presence ratio, amplitude cutoff, …) is thresholded, and the per-unit/per-trial `is_good_trials` mask is not applied. Sessions left with zero units are skipped. This retains **60,978** session-neurons (mean 352/session, range 76–844).

Note on what this actually selects: in this dataset `anno_name` is populated only for units the QC classifier labelled `'good'`, so the AI's filter is effectively `classification == 'good'` **∩** `unit_quality == 'good'` — i.e. a strict subset of the classifier-good set, 12% smaller than the 69,943 units (173 sessions) reported in `methods.txt`. The AI never read the `units/classification` column, which is the classifier verdict described in the paper and the QC white paper.

ii.
```python
            quality = np.asarray(nwb.units['unit_quality'][:], dtype=str)
            annotation = np.char.strip(np.asarray(nwb.units['anno_name'][:], dtype=str))
            keep = (quality == 'good') & (np.char.str_len(annotation) > 0) & (np.char.lower(annotation) != 'nan')
            kept_idx = np.flatnonzero(keep)
            if len(kept_idx) == 0:
                print('  skipped: no good annotated units', flush=True)
                continue
            regions = annotation[keep].tolist()
```
```python
            'unit_filter': "unit_quality == 'good' and valid nonempty histological anno_name",
```

iii. The AI's stated rationale (script docstring): "classifier-labelled good units are used; as in preprocessing_DJ_2022Aug.py, units must also have histology (a nonempty CCF annotation)". In the trajectory: "The paper explicitly says analyses use classifier-labeled `good` units; thus `unit_quality == 'good'` is the primary neuron QC" — i.e. it assumed `unit_quality` *was* the classifier label. It then noticed the resulting count did not match the paper ("There are 154,948 units marked `good`, but only 62,179 are both good and histologically annotated … The paper's 173-session/69,943-unit figure likely reflects a related release/curation stage rather than blindly using all NWBs") and proceeded anyway. The `!= 'nan'` clause was added reactively, after the verifier revealed 1,201 neurons assigned to a brain region literally named `nan`: "These units do not have valid histology and violate the reference ephys–histology intersection, so they must be excluded."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go-cue onset, done in absolute session time: all NWB streams share one clock, so the fixed relative edge grid is simply added to each trial's go-cue timestamp and the spikes are binned against those absolute edges. No resampling, interpolation or per-stream offset correction. The go-cue time itself is the first `go_start_times` timestamp falling inside the trial's `[start_time, stop_time)` interval.

ii.
```python
            go = first_event_in_trial(events['go_start_times'].timestamps[:], starts, stops)
            if np.any(~np.isfinite(go)):
                raise ValueError(f'{path}: {np.sum(~np.isfinite(go))} trials lack go cue')
```
```python
    abs_edges = go + EDGES
    for u, st in enumerate(spike_times):
        ...
        out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
```
```python
            'temporal_alignment_event': 'go cue onset',
            'off_start': START, 'off_end': END,
```

iii. The AI's schema inspection confirmed that behavioural-event `timestamps` are absolute session times on the same clock as `spike_times`, so it aligned by offsetting the bin grid rather than by shifting spike times. It chose to assign the go cue by trial-interval membership "rather than relying on array position" for robustness.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 of them, spanning −2.5 s to +1.5 s relative to the go cue (`np.linspace(-2.5, 1.5, 81)` edges; centres −2.475 … +1.475 s). The grid is a module-level constant, identical for every trial and session, so every trial is exactly 80 timepoints. The neural data is binned once at this resolution directly from spike times — there is no rebinning of an intermediate representation. The inputs and the tongue output are evaluated at the same 80 bin centres, so all three streams share the grid. `metadata['time_bin_size'] = 50.0` (ms), `off_start = -2.5`, `off_end = 1.5`.

ii.
```python
START, END, DT = -2.5, 1.5, 0.05
EDGES = np.linspace(START, END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```
```python
            'time_bin_size': 50.0,
            'bin_convention': '80 nonoverlapping [left,right) bins; centers -2.475 to 1.475 s',
```

iii. The window and bin width come straight from the task instructions. The AI reasoned about the reference `sliding_histogram`, which "treats requested bounds as bin centers", and decided: "For this task, the explicit 4.0 s window with 50 ms bins is most naturally 80 bins centered at −2.475 through 1.475 s" — i.e. it interpreted the instructed interval as bin *edges* so that the window is exactly covered by 80 non-overlapping bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. `BehavioralEvents/sample_start_times` (the sample-epoch/tone onsets), together with the trial's go-cue time. For each trial the AI takes the **first** tone timestamp falling inside `[start_time, stop_time)`, and raises an error if a trial has none.

ii.
```python
            tone = first_event_in_trial(events['sample_start_times'].timestamps[:], starts, stops)
            # Audio-delay task sessions should all contain a sample/tone onset.
            if np.any(~np.isfinite(tone)):
                raise ValueError(f'{path}: {np.sum(~np.isfinite(tone))} trials lack tone onset')
```

iii. The AI noticed that the sample stream has more events than trials (because a lick during the sample/delay epoch replays that epoch) and resolved the ambiguity by taking the first event within the trial: "Tone/sample events can include extra transitions, so tone onset should be assigned as the first sample-start timestamp within each trial."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. A continuous, time-varying value: each of the 80 bin centres, expressed in absolute time (`go + CENTERS`), minus the trial's tone onset. Stored as `float32` in row 0 of the `(2, 80)` input array under the name `'time from tone onset'`. No clipping, normalisation, or binarisation is applied. Because the first tone of a replayed trial can be many seconds before the go cue, the resulting values span −1.52 s to +11.89 s across the dataset.

ii.
```python
                bt = go[ti] + CENTERS
                ...
                time_from_tone = (bt - tone[ti]).astype(np.float32)
                ...
                sess_i.append(np.vstack((time_from_tone, photo)).astype(np.float32, copy=False))
```
```python
        'input_names': ['time from tone onset', 'photostimulation on'],
```

iii. The instructions call for a continuous, time-varying input, and the AI implemented it literally as the elapsed time since the trial's tone at each bin centre: "time from tone onset and photostimulation state at bin centers".

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same grid used to bin the spikes: `bt = go[ti] + CENTERS`, where `CENTERS` are the centres of the same 81-edge array used for `abs_edges`. Bin *k* of the input therefore covers the same interval as bin *k* of the firing rates, by construction, with no interpolation or offset correction.

ii.
```python
EDGES = np.linspace(START, END, 81, dtype=np.float64)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
```
```python
                bt = go[ti] + CENTERS
                sess_n.append(bin_spikes(spikes, go[ti]))
                time_from_tone = (bt - tone[ti]).astype(np.float32)
```

iii. N/A — a direct consequence of defining one go-cue-relative grid and reusing it for all streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` — the absolute onset/offset timestamps of every photostimulation epoch in the session — rather than the `photostim_onset`/`photostim_duration` columns of the trials table. The code asserts that the two event arrays have equal length. (Checked against the data: the event-derived intervals are numerically identical to `start_time + photostim_onset` and `+ photostim_duration` from the trials table, so the two sources are equivalent.)

ii.
```python
            # Photostimulation is represented continuously from the recorded
            # onset to offset, sampled at each neural-bin center.
            ps = np.asarray(events['photostim_start_times'].timestamps[:], float)
            pe = np.asarray(events['photostim_stop_times'].timestamps[:], float)
            if len(ps) != len(pe):
                raise ValueError(f'{path}: unequal photostim onset/offset counts')
```

iii. The AI's schema survey found that "Behavioral events provide go and photostimulation start/stop timestamps", and it preferred those absolute timestamps because they live on the same clock as the spikes and the bin grid — no string parsing or trial-start arithmetic is needed. It also read from the methods that "Photoinhibition occurs in the final 0.5 s of delay and ends before go cue", i.e. inside the extracted window.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying series: for each bin centre, 1 if the centre lies in any `[onset, offset)` stimulation interval of the session, else 0. Non-stimulated trials simply get all zeros. Stored as `float32` in row 1 of the input array. Membership is tested against *all* session epochs at once with a broadcast comparison (80 × n_epochs booleans per trial), not just the current trial's epoch.

ii.
```python
                # Interval membership, [on,off), across all stim epochs.
                photo = np.zeros(80, dtype=np.float32)
                if len(ps):
                    photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. The instructions ask for "whether photostimulation is on at every time point (discrete, time-varying)", so the AI represented it as interval membership at bin centres rather than as a per-trial flag: "Photostimulation is represented continuously from the recorded onset to offset, sampled at each neural-bin center." Using the session-wide epoch list makes the result independent of any trial-to-epoch assignment.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Both the stimulation onsets/offsets and the bin centres are absolute session times, so the comparison is done directly on the shared clock — the same `bt = go[ti] + CENTERS` grid used for the neural bins and the other input. No conversion from trial-relative to go-relative time is needed because the raw event timestamps are already absolute.

ii.
```python
                bt = go[ti] + CENTERS
                ...
                photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. N/A — alignment follows from using absolute timestamps against the go-cue-anchored bin centres.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The dataset stores no explicit lick-direction column, so choice is derived from two trials-table columns: `trial_instruction` (`'left'`/`'right'`, the tone-instructed side) and `outcome` (`'hit'`/`'miss'`/`'ignore'`). Hit ⇒ the animal licked the instructed side; miss ⇒ it licked the opposite side; ignore ⇒ it did not lick (own class).

ii.
```python
            instruction = np.asarray(tr['trial_instruction'][:], dtype=str)
            outcome = np.asarray(tr['outcome'][:], dtype=str)
```
```python
                if outcome[ti] == 'ignore':
                    choice = 2                         # no lick/no response
                elif outcome[ti] == 'hit':
                    choice = 0 if instruction[ti] == 'left' else 1
                else:                                  # miss: wrong direction
                    choice = 1 if instruction[ti] == 'left' else 0
```

iii. From the trajectory: "Choice can be mapped exactly from outcome and instruction: hit = instructed side, miss = opposite side, ignore = no lick." The AI also inspected the lick-time streams (`left_lick_times`, `right_lick_times`) and the repository's correctness coding ("1 correct/free-water, 0 error, −1 no response") before settling on this deterministic mapping.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded `0 = left`, `1 = right`, `2 = no lick`, one value per trial, broadcast across all 80 bins into row 0 of an `(4, 80)` `int8` output array so that all four outputs share one rectangular, time-varying array. `output_values[0] = ['left','right','no lick']`. Resulting distribution: 42.9% left, 42.2% right, 14.9% no lick.

ii.
```python
                # First three outputs are per-trial; tongue position is
                # time-varying. To keep a rectangular categorical output, the
                # trial labels are repeated over time, as permitted by format.
                o = np.empty((4,80), dtype=np.int8)
                o[0] = choice
```
```python
        'output_names': ['lick direction choice', 'outcome', 'early lick', 'tongue y-position'],
        'output_values': [['left','right','no lick'], ['ignore','miss','hit'], ...
```

iii. The instructions list choice as a per-trial categorical output but also ask for time-varying outputs "if at all possible"; since the tongue output is genuinely time-varying, the AI repeated the per-trial labels over bins to keep one rectangular array, noting in the comment that this is "permitted by format". Its inspection of `decoder.py` confirmed that mixed per-trial/time-varying categorical outputs are accepted.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the NWB trials table, which already contains exactly the three strings the instructions ask for (`'ignore'`, `'miss'`, `'hit'`).

ii.
```python
            outcome = np.asarray(tr['outcome'][:], dtype=str)
```

iii. "Trial labels map directly from NWB: instruction, early_lick, and outcome." The AI verified the label vocabulary with a dataset-wide census before writing the converter, so no derivation or recoding from lick times was needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A fixed dictionary maps `ignore → 0`, `miss → 1`, `hit → 2` (the order given in the instructions); the per-trial code is broadcast across all 80 bins into row 1 of the output array. `output_values[1] = ['ignore','miss','hit']`. Distribution: 14.9% ignore, 16.5% miss, 68.6% hit.

ii.
```python
            out_map = {'ignore':0, 'miss':1, 'hit':2}
```
```python
                o[1] = out_map[outcome[ti]]
```

iii. Code assignment follows the instruction ordering; the mapping is a hard dictionary lookup, so any unexpected label would raise rather than be silently mis-coded. Like the other per-trial labels it is repeated across bins for a rectangular output.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The `early_lick` column of the trials table, whose values are the strings `'no early'` and `'early'`.

ii.
```python
            early = np.asarray(tr['early_lick'][:], dtype=str)
```

iii. Same rationale as outcome: the flag is stored explicitly per trial, and the AI's label census over all 174 sessions confirmed the two-value vocabulary, so no derivation from lick timestamps was attempted.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to `0 = no`, `1 = yes` with an inline conditional, broadcast across all 80 bins into row 2 of the output array. `output_values[2] = ['no','yes']`. Distribution: 88.6% no, 11.4% yes. Note that unlike the dictionary lookups used for outcome, this uses an `else` fallback, so any unexpected string would silently become `0` (in practice only the two known values occur).

ii.
```python
                o[2] = 1 if early[ti] == 'early' else 0
```

iii. Straightforward binarisation of the stored flag, coded in the order given by the instructions ("no, yes"), and repeated across bins like the other per-trial outputs. Early-lick trials are deliberately retained in the dataset (rather than excluded as in the original paper's analyses) because this is a required decoder output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: a `(n_frames, 3)` data array whose columns are `(tongue_x, tongue_y, tongue_likelihood)` with matching absolute `timestamps` (~294 Hz). Column 1 provides the y-position; column 2 (the DeepLabCut likelihood) determines whether the tongue is visible in that frame.

ii.
```python
            tts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
            cam_t = np.asarray(tts.timestamps[:], dtype=float)
            cam = np.asarray(tts.data[:], dtype=float)
```

iii. The AI read the series' own description rather than assuming the layout: "TongueTracking columns are explicitly `(tongue_x, tongue_y, tongue_likelihood)` sampled at about 294 Hz. Thus tongue y should be binned/aligned from timestamps, and visibility should be determined from likelihood." Its census confirmed tongue tracking is present in all 174 sessions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps. (1) A frame counts as visible only if `tongue_y` and likelihood are finite and `likelihood >= 0.9`. (2) Session-level thresholds `q40`, `q60` are the 40th/60th percentiles of `tongue_y` over **all visible frames of the session** (raw frames, not binned values); if a session has no visible frame at all, both become NaN. (3) For each trial, the value used for each of the 80 bins is that of the **single camera frame nearest to the bin centre** (nearest-neighbour sampling, not an average over the bin), and that frame's own likelihood decides visibility. The per-session thresholds are recorded in `session_info`. Resulting class distribution: 6.2% / 3.2% / 6.5% / 84.2% (not visible).

ii.
```python
DLC_THRESHOLD = 0.9
```
```python
            visible_all = np.isfinite(cam[:,1]) & np.isfinite(cam[:,2]) & (cam[:,2] >= DLC_THRESHOLD)
            if not np.any(visible_all):
                q40 = q60 = np.nan
            else:
                q40, q60 = np.percentile(cam[visible_all,1], [40, 60])
```
```python
                idx = np.searchsorted(cam_t, bt)
                idx = np.clip(idx, 1, len(cam_t)-1)
                prev = idx - 1
                idx = np.where(np.abs(cam_t[prev]-bt) <= np.abs(cam_t[idx]-bt), prev, idx)
                y = cam[idx,1]
                vis = np.isfinite(y) & np.isfinite(cam[idx,2]) & (cam[idx,2] >= DLC_THRESHOLD)
```

iii. The AI searched the provided repository and the method paper for an explicit DLC confidence rule and found none, so it adopted 0.9 as the conventional DeepLabCut likelihood cut-off and documented it in the script docstring ("DLC tongue points with likelihood < .9 are treated as not visible") and in `metadata['tongue_visibility']` and `session_info['dlc_likelihood_threshold']`. Percentiles are computed only over confidently visible frames because the tracker still emits a coordinate when the tongue is retracted. Nearest-sample selection was chosen as the way to "sample" the ~294 Hz video at each neural bin centre.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Exactly the four classes specified: `0` if `y < q40`, `1` if `q40 <= y <= q60`, `2` if `y > q60`, and `3` ("not visible") for any bin whose nearest frame fails the visibility test. Class 3 is the initialised default, so NaN thresholds (a session with no visible tongue) leave every bin at 3. `output_values[3] = ['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']`.

ii.
```python
                tongue_cat = np.full(80, 3, dtype=np.int8)
                tongue_cat[vis & (y < q40)] = 0
                tongue_cat[vis & (y >= q40) & (y <= q60)] = 1
                tongue_cat[vis & (y > q60)] = 2
                ...
                o[3] = tongue_cat
```

iii. Directly implements the per-session discretisation in the task instructions, with the fourth "not visible" class used for bins where the tongue is not confidently tracked. Thresholds are per session, as instructed, and are stored in `session_info` for traceability.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same absolute clock as the spikes and the go cue, so the tongue output is evaluated at the same bin centres `bt = go[ti] + CENTERS` used for the firing rates: `np.searchsorted` locates the insertion point of each bin centre in the camera time base, and whichever of the two neighbouring frames is closer in time supplies that bin's value. There is no interpolation and no offset correction. The index is clipped to `[1, len(cam_t)-1]`, so bin centres that fall outside the camera's coverage (the video is trial-gated and off during the inter-trial interval; ~3% of trials have a go cue less than 2.5 s after trial start) are still assigned the nearest existing frame however distant it is in time — there is no maximum-gap guard.

ii.
```python
                idx = np.searchsorted(cam_t, bt)
                idx = np.clip(idx, 1, len(cam_t)-1)
                prev = idx - 1
                idx = np.where(np.abs(cam_t[prev]-bt) <= np.abs(cam_t[idx]-bt), prev, idx)
                y = cam[idx,1]
```

iii. Stated in `metadata`: "DLC likelihood >= 0.9; nearest camera sample to bin center", and in the trajectory: "Each neural bin receives the nearest camera sample." Since the camera series carries absolute timestamps on the shared clock, the AI treated alignment as a pure resampling problem onto the go-cue-anchored grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Four distinct policies, applied inconsistently:
- **Missing go cue or tone onset in a trial**: hard failure — `raise ValueError`, which would abort the whole conversion (never triggered in practice).
- **Missing histology / uncurated units**: `anno_name` entries that are empty or the literal string `'nan'` are excluded; a session where this leaves no units is skipped with a printed message (1 session, `sub-440958_ses-20190216T162508`, whose annotations are all NaN).
- **Untracked tongue frames**: non-finite coordinates or likelihood below threshold are treated as "not visible" (class 3) rather than imputed; a session with no visible frame at all yields NaN percentiles, which propagate to all-class-3 through the `vis` mask.
- **Trials with no spike data**: *not handled*. 3,511 trials with zero spikes across every neuron are emitted as 4 s of 0 Hz firing for all neurons; the verifier warned on each and the warnings were not investigated.

ii.
```python
            if np.any(~np.isfinite(go)):
                raise ValueError(f'{path}: {np.sum(~np.isfinite(go))} trials lack go cue')
```
```python
            keep = (quality == 'good') & (np.char.str_len(annotation) > 0) & (np.char.lower(annotation) != 'nan')
            kept_idx = np.flatnonzero(keep)
            if len(kept_idx) == 0:
                print('  skipped: no good annotated units', flush=True)
                continue
```
```python
            if not np.any(visible_all):
                q40 = q60 = np.nan
            ...
                vis = np.isfinite(y) & np.isfinite(cam[idx,2]) & (cam[idx,2] >= DLC_THRESHOLD)
                tongue_cat = np.full(80, 3, dtype=np.int8)
```

iii. The AI's general stance is to fail loudly on structural assumptions it believes must hold (one go cue and one tone per trial, matching photostim onset/offset counts) and to represent genuinely absent measurements as an explicit category rather than impute them. The `'nan'`-annotation exclusion was a reactive fix driven by the verifier: "its region summary exposed 1,201 neurons assigned to the literal region name `nan` … These units do not have valid histology and violate the reference ephys–histology intersection, so they must be excluded." No equivalent check was performed for missing *ephys* coverage of trials.

## 10-a. What are the most time-consuming steps of the code?

i. Three dominate. (1) NWB/HDF5 I/O: reading each session's concatenated `spike_times` buffer (up to ~10^7 doubles) and the ~680k × 3 camera tracking array; across 174 files and ~50 GB of source this is the bulk of the runtime. (2) Spike binning: `bin_spikes` is called once per trial and loops over every kept unit inside, i.e. ~33 M `searchsorted` calls over the whole dataset (94,370 trials × ~350 units). Measured on one session (410 units, 368 trials) this takes 0.7 s versus 0.2 s for a trial-vectorised formulation — about 3× the necessary cost, but still a minority of session time. (3) Building ~94k × 3 small arrays and pickling the 10.8 GB result. The whole conversion took roughly 4–5 minutes.

ii.
```python
    for u, st in enumerate(spike_times):
        st = np.asarray(st)
        out[u] = np.diff(np.searchsorted(st, abs_edges, side='left')) / DT
```
```python
            spike_data = np.asarray(spike_col.target.data[:], dtype=np.float64)
```

iii. The AI profiled empirically, interrupting two slow runs. First: "The converter is taking too long because each trial calls `np.histogram` on every unit's full-session spike train, repeatedly scanning the same spikes. This is unnecessarily expensive and should be replaced with `np.searchsorted` at bin edges, which computes counts in logarithmic time and preserves the same half-open bin convention." Then: "Per-unit HDF5 access is likely now the main cost. Bulk-loading the concatenated `spike_times` dataset and its ragged index once per session should substantially reduce I/O overhead." After both fixes it judged the remaining runtime to be I/O-bound and acceptable.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three.
- The **per-trial loop** `for ti in range(ntr)` calls `bin_spikes(spikes, go[ti])`, which re-loops over all units for each trial. Because the edge grid is a fixed offset from each trial's go cue, all trials' edges can be flattened into one array and `searchsorted` called once per unit for the whole session (`np.searchsorted(s, (go[:,None]+EDGES).ravel())`), collapsing the trial dimension; this is what the expert solution does and it is ~3× faster on the session benchmarked.
- **`first_event_in_trial`** is a Python `for` loop over trials doing a scalar `searchsorted` each iteration; the whole function is one vectorised `np.searchsorted(times, starts, 'left')` plus a bounds comparison against `stops`.
- Within the trial loop, the tongue nearest-neighbour lookup and the photostim membership test are already vectorised over bins, but are re-executed per trial rather than once per session over a flattened (n_trials × 80) grid.

ii.
```python
            for ti in range(ntr):
                bt = go[ti] + CENTERS
                sess_n.append(bin_spikes(spikes, go[ti]))
```
```python
    for i, (a, b) in enumerate(zip(starts, stops)):
        j = np.searchsorted(times, a, side='left')
        if j < len(times) and times[j] < b:
            ans[i] = times[j]
```

iii. The AI never articulated a reason for keeping the trial loop; it reached an acceptable runtime after replacing `np.histogram` with `searchsorted` and bulk-loading spikes, and stopped optimising at that point ("the optimized algorithm is appropriate; remaining runtime is dominated by reading 174 NWB files, camera streams, and serializing a multi-gigabyte pickle"). The per-trial structure also keeps the code close to the required output layout (a list of per-trial arrays), which is presumably why it was retained.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read exactly once, and all session-level quantities (unit selection, percentile thresholds, photostim epochs, camera arrays) are computed once per session. The repetition is inside the trial loop:
- every unit's spike train is searched separately for each trial (`n_trials` binary searches per unit instead of one over concatenated edges), and `np.asarray(st)` is re-applied to each already-materialised spike array on every trial;
- `abs_edges = go + EDGES` and `bt = go[ti] + CENTERS` rebuild the grid per trial;
- the photostim membership matrix is rebuilt against the full session epoch list for every trial, including the ~80% of trials with no stimulation.
None of this changes results — it is pure redundant work. Across the earlier, discarded runs the entire conversion was also repeated three times (two interrupted for optimisation, one re-run after the `nan`-annotation fix).

ii.
```python
def bin_spikes(spike_times, go):
    out = np.empty((len(spike_times), 80), dtype=np.float32)
    abs_edges = go + EDGES
    for u, st in enumerate(spike_times):
        st = np.asarray(st)
```
```python
                if len(ps):
                    photo[:] = np.any((bt[:,None] >= ps[None,:]) & (bt[:,None] < pe[None,:]), axis=1)
```

iii. The AI's stated concern was eliminating *repeated I/O* and repeated full-session scans — "repeatedly scanning the same spikes… should be replaced with `np.searchsorted`" and bulk ragged loading — which it did. The residual per-trial repetition is cheap enough (log-time searches on already-resident arrays) that it did not revisit it.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Very little is computed and then thrown away: every field the script builds is written into the pickle. The genuinely wasteful work is:
- **3,511 trials with no ephys coverage** are binned in full (~350 neurons × 80 bins each) and stored as all-zero neural matrices. This is both wasted compute/storage and actively harmful downstream — the decoder trains and is scored on ~3.7% of trials whose neural input is identically zero.
- The full-session camera array is read as three `float64` columns although only `tongue_y` and the likelihood are used, and `cam[idx,2]` is re-indexed per trial for visibility after `visible_all` was already computed for the whole session.
- Photostim membership is tested against every stimulation epoch of the session for every trial, though at most one can overlap a trial's 4 s window.
- `spike_data` is materialised for **all** units in the file (including the ~70% that fail the unit filter) before slicing out the kept ones; the reference does the same, so this is inherent to the ragged storage rather than avoidable waste.
- `import json` is unused, and `stops` is used only for event assignment.

ii.
```python
            for ti in range(ntr):     # no trial mask: zero-spike trials are binned and stored
                sess_n.append(bin_spikes(spikes, go[ti]))
```
```python
            cam = np.asarray(tts.data[:], dtype=float)   # all 3 columns, float64
```
```python
from pathlib import Path
import json, pickle, gc     # json unused
```

iii. The AI did not discuss discarding work; its efficiency reasoning was confined to the spike-binning and HDF5-read bottlenecks. The retention of zero-spike trials follows from its explicit decision that "all released behavioral sessions and trials are retained" (see 1-e), not from an oversight it identified and accepted.
