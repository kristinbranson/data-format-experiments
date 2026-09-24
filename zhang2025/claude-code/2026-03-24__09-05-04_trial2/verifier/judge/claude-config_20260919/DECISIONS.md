# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the ONE API. It walks the on-disk ONE cache with `glob`, treating
`data/one_cache/<lab>/Subjects/<subject>/<date>/001` as the session unit, and keeps a session only if
four `glob`/`os.path.exists` probes succeed (spike sorting under `alf/probe*/pykilosort/*`, a trials
parquet under `alf/*/`, `alf/_ibl_wheel.timestamps.npy`, and a camera ROIMotionEnergy + camera-times
pair). Every file is then read directly with `np.load` / `pd.read_parquet`; no ONE/`SessionLoader`/
`SpikeSortingLoader` call is made anywhere in the script. Sessions are processed serially in a single
process.

Consequences measured on the shipped data: 461 session directories exist on disk, the glob pattern
only ever considers the 411 that end in `/001`, `find_session_dirs()` accepted 393, and 335 actually
converted. 57 sessions raised `FileNotFoundError` on
`alf/_ibl_wheel.position.npy` and 1 was dropped for having no valid trials. The 57 failures are not
missing data — the file is present in a revision folder, e.g.
`.../PL050/2023-06-15/001/alf/#2024-05-06#/_ibl_wheel.position.npy`; the screening check looked at
`_ibl_wheel.timestamps.npy` (which *is* un-revisioned) while the loader hard-codes the un-revisioned
position path. Final totals: 335 sessions / 118 subjects / 146,747 trials, against the reference's
441 sessions / 136 subjects / 188,740 trials.

ii.
```python
DATA_ROOT = Path('data/one_cache')

def find_session_dirs():
    """Find all session directories with required data."""
    session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
    valid = []
    for sdir in session_dirs:
        has_spikes = len(glob.glob(os.path.join(sdir, 'alf/probe*/pykilosort/*/spikes.times.npy'))) > 0
        has_trials = len(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt'))) > 0
        has_wheel = os.path.exists(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy'))
        ...
        if has_spikes and has_trials and has_wheel and has_me:
            valid.append(sdir)
    return valid
```

```python
def load_wheel_speed(sdir):
    wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
    wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

```python
for i, sdir in enumerate(session_dirs):          # serial, one process
    result = process_session(sdir, br, ...)
```

iii. CONVERSION_NOTES.md Step 2/Step 4 records the data layout as
`data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/` and resolves the session question as "Use
all 393 sessions with complete data", noting the disk holds fewer sessions than the 459 released. The
57 crashes are documented in Step 9 as "Sessions failed (missing data) | 57 (all PL050/hausserlab -
missing wheel data)" — a diagnosis that is wrong, since the wheel position file exists in a revision
directory. No justification is given anywhere for restricting the glob to `/001`.

## 1-b. How are the data split into subjects (mice)?

i. The subject is read off the cache path: the directory immediately after `Subjects/` is the mouse
name. Subjects are accumulated in first-encounter order into `all_subjects`, and `subject_idx` is the
index of each session's subject in that list. This yields 118 subjects (reference: 136).

ii.
```python
def parse_session_info(sdir):
    """Extract lab, subject, date from session directory path."""
    parts = Path(sdir).parts
    sub_idx = parts.index('Subjects')
    lab = parts[sub_idx - 1]
    subject = parts[sub_idx + 1]
    date = parts[sub_idx + 2]
    return lab, subject, date
```

```python
if subject not in all_subjects:
    all_subjects.append(subject)
subject_per_session.append(subject)
...
subject_idx = np.array([all_subjects.index(s) for s in subject_per_session], dtype=np.int32)
```

iii. CONVERSION_NOTES.md Step 2 documents the cache layout as
`<lab>/Subjects/<subject>/<date>/001`, so the subject name is taken to be given by the directory
structure and needs no derivation. No separate rationale is recorded.

## 1-c. How are the data split into sessions?

i. One session = one session directory in the ONE cache. The AI hard-codes the session-number
component to `001`, so at most one session per (subject, date) is ever considered; the 50 directories
numbered `002`/`003`/`004`/`007`/`008` on disk are never visited. The session identifier it records is
the composite string `f"{subject}_{date}"` rather than the eid, which is unique only because of the
`/001` restriction.

ii.
```python
session_dirs = sorted(glob.glob(str(DATA_ROOT / '*/Subjects/*/*/001')))
```

```python
lab, subject, date = parse_session_info(sdir)
session_id = f"{subject}_{date}"
```

iii. Step 2 of CONVERSION_NOTES.md asserts the data structure is
`data/one_cache/<lab>/Subjects/<subject>/<date>/001/alf/`, i.e. the AI treats `001` as part of the
fixed layout. There is no note recognising that other session numbers exist, and no rationale for
excluding them.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial, so the split is taken directly
from the data. Where several revisions of the table exist under `alf/*/`, the lexicographically last
one is used. Trial windows are built from `stimOn_times` as `[stimOn-0.5, stimOn+1.5]`.

ii.
```python
def load_trials(sdir):
    """Load trials table from parquet file."""
    trial_files = sorted(glob.glob(os.path.join(sdir, 'alf/*/_ibl_trials.table.pqt')))
    if not trial_files:
        raise FileNotFoundError(f"No trials table found in {sdir}")
    # Use the most recent revision
    trials = pd.read_parquet(trial_files[-1])
    return trials
```

```python
stim_on = trials[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. Step 2 lists the trials-table columns and Step 5 maps them straight onto per-trial variables; no
decision was needed because the table is already one row per trial. The "most recent revision" comment
is the AI's own choice for resolving the duplicated revisions it saw on disk.

## 1-e. How are trials filtered based on quality controls?

i. Five criteria, combined into one mask. (1) Reaction time `firstMovement_times - stimOn_times` must
lie in [0.08, 2.0] s. (2) Trial length `feedback_times - goCue_times` must be <= 10 s. (3) No-response
trials (`choice == 0`) are dropped. (4) Any NaN in `stimOn_times, choice, feedback_times,
probabilityLeft, firstMovement_times, feedbackType` drops the trial. (5) The trial window must be
covered by both the wheel stream and the camera stream, tested per trial inside the behaviour
interpolation with the same `binsize` tolerance the reference code uses. Sessions left with fewer than
2 usable trials are dropped. This reproduces the reference code's
`load_trials_and_mask(min_rt=0.08, max_rt=2., max_trial_len=10., exclude_nochoice=True,
nan_exclude='default')` plus the coverage test from `get_behavior_per_interval`.

ii.
```python
def create_trial_mask(trials):
    mask = pd.Series(True, index=trials.index)
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    if 'goCue_times' in trials.columns and 'feedback_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    mask &= (trials['choice'] != 0)
    for col in NAN_EXCLUDE:
        if col in trials.columns:
            mask &= ~trials[col].isna()
    return mask
```

```python
        # Check coverage (matching reference code)
        if np.abs(t_beg - beh_t[0]) > BINSIZE:
            mask[trial_idx] = False
            continue
        if np.abs(t_end - beh_t[-1]) > BINSIZE:
            mask[trial_idx] = False
            continue
```

```python
combined_mask = mask.values & wheel_mask & whisker_mask
good_indices = np.where(combined_mask)[0]
if len(good_indices) < 2:
    print(f"  WARNING: Only {len(good_indices)} valid trials, skipping session", flush=True)
    return None
```

iii. Step 1 of CONVERSION_NOTES.md transcribes `load_trials_and_mask`'s defaults
("min_rt=0.08, max_rt=2.0", "max_trial_len=10.0", "Exclude no-choice trials (choice==0)", the six
NaN-exclude columns) and Step 3 cross-checks the RT bounds against the data paper's "0.08-2.00 s". The
coverage checks are explicitly annotated in the code as "matching reference code", and Step 1 records
`align_spike_behavior` as the reference function that combines the trial mask with behaviour
availability.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from every `alf/probe*/pykilosort/*` directory. Two
further arrays are read only to label the units anatomically: `clusters.channels.npy` (cluster → peak
channel) and `channels.brainLocationIds_ccf_2017.npy` (channel → Allen id), which are composed and
mapped through `BrainRegions.id2acronym` and `acronym2acronym(..., mapping='Beryl')`. The unit count
for a session is `len(clusters.channels)` — i.e. every sorted cluster, including clusters that emit no
spike in the session (these become all-zero rows).

ii.
```python
        st_file = os.path.join(pdir, 'spikes.times.npy')
        sc_file = os.path.join(pdir, 'spikes.clusters.npy')
        cc_file = os.path.join(pdir, 'clusters.channels.npy')
        cb_file = os.path.join(pdir, 'channels.brainLocationIds_ccf_2017.npy')
        ...
        spike_times = np.load(st_file).flatten()
        spike_clusters = np.load(sc_file).flatten()
        cluster_channels = np.load(cc_file).flatten()
        channel_brain_ids = np.load(cb_file).flatten()
        n_clusters = len(cluster_channels)
        valid_channels = np.clip(cluster_channels, 0, len(channel_brain_ids) - 1)
        cluster_brain_ids = channel_brain_ids[valid_channels]
        cluster_acronyms = br.id2acronym(cluster_brain_ids)
        beryl_regions = br.acronym2acronym(cluster_acronyms, mapping='Beryl')
```

iii. Step 2 of CONVERSION_NOTES.md lists exactly these four files as the per-probe contents, and
Step 5's mapping table gives "spikes.times + spikes.clusters → neural", citing the reference functions
`bin_spiking_data` / `get_spike_data_per_interval`. Beryl is chosen because Step 1 records
`list_brain_regions` as the reference function that "Maps cluster regions to Beryl atlas".

## 2-b. How is the `neural` data processed?

i. Probes are pooled into one population: each probe's `spikes.clusters` is offset by the running
cluster count, the concatenated spike times are stably sorted, and one trial's spikes are located with
`searchsorted`. Within a trial, spikes are assigned to one of 100 20-ms bins by integer division from
the window start, converted to a flat `cluster * 100 + bin` index and counted with a single
`np.bincount`. No smoothing is applied.

The stored quantity is the raw **spike count**, not a rate: the reference divides by the bin width to
get Hz, the AI does not. Counts are clipped to [0, 255] and cast to `uint8` to shrink the pickle
(`train_decoder.py --verify-only` emits a dtype warning on every trial as a result). The resulting
file is 21 GB with a mean of 1443 neurons/session (reference: 164).

ii.
```python
        spike_clusters_offset = spike_clusters + cluster_offset
        cluster_offset += n_clusters
        ...
    merged_times = np.concatenate(all_spike_times)
    merged_clusters = np.concatenate(all_spike_clusters)
    sort_idx = np.argsort(merged_times, kind='stable')
```

```python
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')
        ...
        bin_idx = np.minimum(((times_trial - t_beg) / BINSIZE).astype(np.int32), N_BINS - 1)
        flat_idx = clusters_trial * N_BINS + bin_idx
        counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
        binned[trial_idx] = counts[:n_clusters * N_BINS].reshape(n_clusters, N_BINS)
```

```python
        # Store as uint8 during accumulation to save memory (spike counts rarely exceed 255)
        # Will be converted to float32 when decoder loads it
        neural_list = [np.clip(neural_trials[i], 0, 255).astype(np.uint8) for i in range(n_trials)]
```

iii. Step 1 records `merge_probes` ("Merges spikes/clusters across probes in a session") and
`bin_spiking_data` as the reference functions being reproduced, and the binning helper is documented
in-code as "Matches reference code bin_spiking_data / get_spike_data_per_interval". Step 6 justifies
the storage type: "Neural data stored as uint8 (spike counts rarely exceed 255) to reduce memory", and
Step 9 notes "Peak memory well within 64 GB cgroup limit thanks to uint8 storage". No rationale is
given for keeping counts rather than converting to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level quality control at all.** Every sorted cluster is kept: no `label >= 1` cut, no
spike-sorting metric cut, no exclusion of units the atlas places outside the brain. The converted
dataset therefore contains 11,701 `void` units (histology placed them outside the brain) and 63,294
`root` units, and averages 1443 neurons/session against the reference's 164. `void` and `root` are
kept as ordinary entries in `brain_regions`, merely sorted to the end of the list.

A side effect surfaced at training time: the 21 GB uint8 pickle expands to ~84 GB as float32, so the
AI wrote a separate `reduce_data.py` that **randomly subsamples each session to 500 neurons** and
trained the decoder on that file instead of `converted_data.pkl` (318 of 335 sessions were
subsampled).

ii.
```python
def load_spikes(sdir):
    """Load and merge spikes from all probes in a session.

    Following reference code: no QC filtering (qc=None).
    """
```

```python
    # Build brain regions list (exclude 'root' and 'void')
    brain_regions = sorted([r for r in all_brain_regions if r not in ('root', 'void')])
    # Add root and void at end if present
    if 'void' in all_brain_regions:
        brain_regions.append('void')
    if 'root' in all_brain_regions:
        brain_regions.append('root')
```

```python
            'neuron_filter': 'All neurons (no QC filtering), matching reference code',
```

`reduce_data.py`:
```python
    idx = np.sort(np.random.choice(nneurons, max_neurons, replace=False))
    for trial in range(len(data['neural'][session])):
        data['neural'][session][trial] = data['neural'][session][trial][idx, :]
```

iii. Step 1 of CONVERSION_NOTES.md: "No QC filter applied in `prepare_data` (qc=None by default) /
ALL neurons used, not just good ones (label >= 1 not required)". Step 3 explicitly weighs the two
sources — "Paper: amplitude > 50uV, noise cutoff < 20uV, refractory period violation -> 'well-isolated'
/ Code: qc=None -> uses ALL neurons / We follow the code: use ALL neurons" — and Step 4 lists "Use ALL
neurons (not just good ones), matching code" as a key resolution decision. Step 3 also records the
data paper's 75,708 well-isolated neurons and the AI's own count of 68,213 clusters with `label >= 1`,
so the alternative was known and consciously rejected. `reduce_data.py` justifies the subsampling as
"The decoder uses PCA to 100 components, so having >500 neurons per session is wasteful and causes OOM".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is by subtraction on the shared session clock: the trial window is
`stimOn_times + (-0.5, 1.5)` in absolute seconds, spikes in that window are found by `searchsorted` on
the sorted spike times, and the bin index is computed from `times_trial - t_beg`, which puts bin 0 at
stimulus onset minus 0.5 s. Spike times, trial event times, wheel timestamps and camera times are all
used as released, with no resynchronisation, because IBL has already put them on one clock.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)  # 2s trial
...
        stim_on = trials[ALIGN_TIME].values
        interval_begs = stim_on + TIME_WINDOW[0]
        interval_ends = stim_on + TIME_WINDOW[1]
```

```python
        bin_idx = np.minimum(((times_trial - t_beg) / BINSIZE).astype(np.int32), N_BINS - 1)
```

iii. Step 1 lists `align_time: 'stimOn_times'` and `time_window: (-0.5, 1.5)` as the reference code's
parameters. Step 3/Step 4 flag a discrepancy — the methods paper aligns wheel/whisker to
`firstMovement_times`, the reference code aligns everything to `stimOn_times` — and resolve it as
"Follow code + decoder task: stimOn_times for all", pointing at the decoder-task line "Temporally
align based on stimulus onset".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, `N_BINS = ceil(2.0 / 0.02) = 100` bins per trial, identical for every trial and
session; `metadata['time_bin_size'] = 20.0` (ms). Spikes are binned once, directly from spike times —
there is no resampling or rebinning of an already-binned signal. The AI explicitly rejected the
methods paper's 50 ms bins for choice/prior in favour of the reference code's uniform 20 ms.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)  # 2s trial
BINSIZE = 0.02  # 20ms
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0s
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

```python
            'time_bin_size': BINSIZE * 1000,  # 20ms
            'n_time_bins': N_BINS,
```

iii. Step 1 records `binsize: 0.02 (20ms)` from `0_data_caching.py`. Step 3 records the papers' two
bin sizes ("50-ms non-overlapping time bins" for choice/prior, "20 ms bins" for wheel/whisker) and
Step 4 resolves: "Code uses 20ms uniformly. Follow code." Step 3 also cites "split into 2-s trials" for
the window length.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a raw variable; it is the analysis grid itself. Given the alignment event
`stimOn_times` and the window `(-0.5, 1.5)` with 20 ms bins, the input is the centre of each of the
100 neural bins, so one 100-long vector is reused for every trial of every session.

ii.
```python
        # Time since stimulus onset: center of each bin
        time_input = np.linspace(
            TIME_WINDOW[0] + BINSIZE / 2,
            TIME_WINDOW[1] - BINSIZE / 2,
            N_BINS
        ).astype(np.float32)
```

iii. Step 5's mapping table gives "Time since stimOn → input[0], Continuous, linspace(-0.5, 1.48, 100),
Reference code: N/A", i.e. the AI notes there is no reference-code counterpart and that the variable is
defined by the window and bin size it already adopted from the reference code.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. None beyond constructing the bin-centre vector. Values run from -0.49 s to +1.49 s in 0.02 s steps
(verified in the converted pickle), are stored as `float32`, and are tiled identically into row 0 of
every trial's `(2, 100)` input array.

ii.
```python
            inp = np.stack([
                time_input,
                np.full(N_BINS, trial_num_in_block[i], dtype=np.float32)
            ], axis=0)  # (2, 100)
            input_list.append(inp)
```

iii. Step 5: "`time_since_stim_onset`: shape (1, 100) per trial, values from -0.5 to 1.48 (center of
each 20ms bin)". The stated reason for using centres rather than edges is that they are the
representative time of each bin.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is the neural grid. Neural bin `i` spans `[stimOn - 0.5 + 0.02i, stimOn - 0.5 + 0.02(i+1))` and
`time_input[i] = -0.49 + 0.02i` is its centre, so column `i` of the input and column `i` of the neural
matrix describe the same 20 ms of the same trial by construction. (Note that the *behavioural outputs*
are sampled on the bin right edges instead — see 7-d — so they sit 10 ms later than this input within
each bin.)

ii.
```python
        bin_idx = np.minimum(((times_trial - t_beg) / BINSIZE).astype(np.int32), N_BINS - 1)
```
```python
        time_input = np.linspace(TIME_WINDOW[0] + BINSIZE / 2, TIME_WINDOW[1] - BINSIZE / 2, N_BINS)
```

iii. Not separately justified beyond Step 5's statement that the values are the "center of each 20ms
bin"; the AI's `--show-processing` plots draw the neural trace, the behavioural traces and a stimulus
onset marker on this shared axis to demonstrate that nothing is shifted.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. The table carries no block id, so a block boundary is
inferred wherever `probabilityLeft` changes from the previous trial.

ii.
```python
def compute_trial_num_in_block(prob_left):
    """Compute trial number within block.

    A block is a contiguous sequence of trials with the same probabilityLeft.
    Trial number resets to 1 at each block boundary.
    """
```

iii. Step 5's mapping table: "Trial number in block → input[1], Count trials since last block change,
From probabilityLeft", and Step 5's key decisions: "Block trial number: Computed as position within
contiguous block of same probabilityLeft."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A scalar Python loop over the trial sequence: the counter increments when `probabilityLeft` equals
the previous trial's value and resets to 1 otherwise. The count is 1-based, and it is broadcast to all
100 bins of the trial as a constant row of the input array.

Critically, the counter is run on the **already-filtered** trial list (`trials_good`), not on the full
trial table. Every trial removed by the RT / no-choice / NaN / coverage mask therefore fails to advance
the counter, so the value is "position among surviving trials in this block", not the animal's real
position in the block. Checked directly on the first session (NYU-11 / 2020-02-18): the AI's value
differs from the true block position by 8.8 trials on average and by up to 30. Across the dataset the
input spans [1, 91] versus [0, 98] for the reference.

ii.
```python
        # 7. Extract per-trial variables
        trials_good = trials.iloc[good_indices]
        ...
        prob_left = trials_good['probabilityLeft'].values
        ...
        # Trial number in block
        trial_num_in_block = compute_trial_num_in_block(prob_left)
```

```python
def compute_trial_num_in_block(prob_left):
    trial_nums = np.ones(len(prob_left), dtype=np.int32)
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            trial_nums[i] = trial_nums[i - 1] + 1
        else:
            trial_nums[i] = 1
    return trial_nums
```

iii. Step 5 documents only "integer count of trial within current block" / "Count trials since last
block change". Neither CONVERSION_NOTES.md nor the trajectory discusses whether the count should be
taken before or after trial filtering, and no sanity check on the block lengths (IBL blocks are drawn
from 20-100 trials) was performed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The single column `choice` of the trials table, which is +1 / -1 / 0. The 0 (no-response) trials
have already been removed by the trial mask, so only ±1 reaches the encoder.

ii.
```python
        # Choice: -1 (left) -> 0, 1 (right) -> 1
        choice = trials_good['choice'].values.copy()
        choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```

iii. Step 5's mapping table: "choice → output[0], Binary: -1(left)->0, 1(right)->1, `bin_behaviors`,
Per-trial". The sign convention is asserted, not verified; there is no sanity check in
CONVERSION_NOTES.md or in the trajectory relating `choice` to `contrastLeft`/`contrastRight` or
`feedbackType`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. A single recode, `choice == 1 → 1`, everything else → 0, then the scalar is broadcast across all
100 bins as `int64`. The output value names are declared `['left', 'right']`, i.e. 0 = left, 1 = right,
as the instructions require.

The mapping is **inverted** with respect to the IBL convention. Verified against the raw trials tables:
on correct trials (`feedbackType == 1`) with the stimulus on the left (`contrastLeft > 0`), `choice` is
always `+1`; on correct trials with the stimulus on the right, `choice` is always `-1`. So `+1` is a
leftward choice and should map to 0. The AI maps `+1 → 1` ("right"). Confirmed end-to-end in
`sample_data.pkl`: every raw `choice == +1` trial carries converted value 1. Because the split is close
to 50/50 (0.495 / 0.505), the error is invisible in the output distribution and does not change
balanced accuracy — it only mislabels which class is which.

ii.
```python
        choice_binary = np.where(choice == 1, 1, 0).astype(np.int32)
```
```python
            out = np.stack([
                np.full(N_BINS, choice_binary[i], dtype=np.int64),
                ...
            ], axis=0)  # (4, 100)
```
```python
        'output_values': [
            ['left', 'right'],           # choice: 0=left, 1=right
```

iii. Step 5 states the intended mapping as "-1(left)->0, 1(right)->1". Step 5's planned sanity checks
include "Choice distribution roughly 50/50", and Step 7/Step 10 report "left 48.2%, right 51.8%" and
"Choice: ~50/50 split (matches expectation)" — a check that cannot detect an inverted label. The
decoder-task line "Choice, binary, per-trial, left = 0, right = 1" is quoted in Step 0 but the sign of
the IBL `choice` column is never checked against it.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column of the trials table, which the reference code calls "block". Trials
with NaN `probabilityLeft` were already dropped by the trial mask.

ii.
```python
        # Prior: probabilityLeft -> 0.2->0, 0.5->1, 0.8->2
        prob_left = trials_good['probabilityLeft'].values
```

iii. Step 4's discrepancy table: "Prior/Block | Code calls it 'block' = probabilityLeft |
probabilityLeft in trials table | Prior probability | Decoder output maps: 0.2->0, 0.5->1, 0.8->2" —
the AI explicitly identified the reference code's `block` target with the task's "prior probability of
left".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A three-way recode implemented as a default-plus-overrides: the array is initialised to 1 (=0.5) and
`np.isclose(..., 0.2, atol=0.05)` / `np.isclose(..., 0.8, atol=0.05)` overwrite with 0 and 2. The
scalar is broadcast across all 100 bins. Any value that is neither ~0.2 nor ~0.8 silently becomes
class 1 rather than being rejected (the reference instead drops trials whose prior is not exactly one
of the three). In practice `probabilityLeft` only takes those three values, and the converted
distribution (0.419 / 0.140 / 0.441) matches the reference (0.418 / 0.141 / 0.442) to three decimals.

ii.
```python
        prior = np.full(len(prob_left), 1, dtype=np.int32)  # default 0.5->1
        prior[np.isclose(prob_left, 0.2, atol=0.05)] = 0
        prior[np.isclose(prob_left, 0.8, atol=0.05)] = 2
```
```python
            ['0.2', '0.5', '0.8'],       # prior: 0=0.2, 1=0.5, 2=0.8
```

iii. Step 4/Step 5 record the mapping as dictated by the decoder task ("Prior = probabilityLeft mapped
to categories: 0.2->0, 0.5->1, 0.8->2"). Step 5's planned checks include "Prior distribution roughly
matches: biased blocks have 80/20 split", and Step 10 confirms "Prior: 0.2 and 0.8 dominate, 0.5 is
minority (matches biased block structure)".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `alf/_ibl_wheel.position.npy` and `alf/_ibl_wheel.timestamps.npy`, the raw rotary-encoder position
and its timestamps. Speed is defined as the absolute value of the velocity derived from them.

ii.
```python
    wh_pos = np.load(os.path.join(sdir, 'alf/_ibl_wheel.position.npy')).flatten()
    wh_times = np.load(os.path.join(sdir, 'alf/_ibl_wheel.timestamps.npy')).flatten()
```

iii. Step 1: "Wheel speed: absolute value of wheel velocity from SessionLoader"; Step 5's mapping
table: "wheel speed → output[2], abs(velocity), discretize into 3 bins, `load_target_behavior`,
interpolate". The reference code's `load_target_behavior('wheel-speed')` returns
`np.abs(sess_loader.wheel['velocity'])`, which the AI transcribed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Four steps. (1) The encoder position is interpolated onto a uniform 1 kHz grid spanning the whole
session with `np.interp`. (2) Velocity is taken as `np.gradient(pos_interp, 0.001)`, a plain central
finite difference. (3) Speed = `np.abs(velocity)`. (4) The session-long speed trace is resampled onto
each trial's 100 sample points by `scipy.interpolate.interp1d(kind='linear',
fill_value='extrapolate')`, then discretized (7-c).

The AI reimplemented `SessionLoader.load_wheel()` from scratch rather than calling it, and the
reimplementation **omits the low-pass filter**: ibllib computes velocity with `velocity_filtered`,
which applies a zero-phase 20 Hz Butterworth to the interpolated position before differentiating. The
AI's docstring claims the opposite of what the code does — it says the velocity is computed "via
Gaussian smoothing. We replicate this" and then applies no smoothing at all, so the resulting speed
retains all the encoder-quantisation noise the reference filters out.

ii.
```python
def load_wheel_speed(sdir):
    """Load wheel speed (absolute velocity) following reference code.

    Reference: load_target_behavior for 'wheel-speed' uses SessionLoader.load_wheel()
    which returns interpolated wheel with velocity computed via Gaussian smoothing.
    We need to replicate this from raw position + timestamps.
    """
    ...
    dt = 0.001  # 1kHz
    t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
    pos_interp = np.interp(t_uniform, wh_times, wh_pos)

    # Compute velocity via finite differences (matching brainbox wheel processing)
    velocity = np.gradient(pos_interp, dt)
    speed = np.abs(velocity)
```

iii. Step 1 of CONVERSION_NOTES.md records that the reference uses `SessionLoader.load_wheel()` and
that the returned wheel frame has "velocity and acceleration computed using Gaussian smoothing"; the
implementation note is simply that this had to be replicated "from raw position + timestamps" because
the AI chose not to go through the ONE API. Step 5's only stated check is "Wheel speed values are
non-negative (absolute velocity)". No comparison of the reimplemented velocity against ibllib's was
performed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-wide terciles. All finite values of the session's `(n_good_trials, 100)` speed matrix are
pooled, the 33.3rd and 66.7th percentiles are taken, and `np.digitize` assigns each sample to class
0/1/2 ("low"/"medium"/"high"). Thresholds are therefore per session, not global, which makes the three
classes equally populated within each session; the realised distribution is 0.333 / 0.333 / 0.333.
NaNs are excluded from the percentile computation but, if any survived, `np.digitize` would place them
in the top class.

ii.
```python
def discretize_to_bins(values, n_bins=3):
    """Discretize continuous values into n_bins categories using quantiles."""
    flat = values[~np.isnan(values)].flatten()
    if len(flat) == 0:
        return np.zeros_like(values, dtype=np.int32)

    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = np.percentile(flat, quantiles)

    result = np.digitize(values, boundaries).astype(np.int32)
    return result
```
```python
        wheel_discrete = discretize_to_bins(wheel_trials, n_bins=3)
```

iii. Step 5's key decisions: "Discretization: Use session-wide terciles for wheel speed and whisker
ME", with the planned check "Wheel speed bins ~33.3% each", confirmed in Step 7 and Step 10
("balanced tercile bins (~33% each)"). The decoder task only specifies "discretized into 3 bins"; equal
occupancy was the AI's choice so that chance is exactly 1/3.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The speed trace is evaluated per trial at `np.linspace(t_beg + 0.02, t_end, 100)`, i.e. at the
**right edge** of each of the 100 neural bins, with the trial window built from the same
`stimOn_times`. This is a verbatim copy of the reference code's `get_behavior_per_interval`
(`x_interp = np.linspace(interval_begs[i] + binsize, interval_ends[i], n_bins)`), including its
`searchsorted(side='right'/'left')` slicing and its `binsize`-tolerance coverage test. Relative to the
`time_since_stim_onset` input, which uses bin centres, the behavioural samples sit 10 ms (half a bin)
later; both conventions live inside the same bin, so no sample crosses a bin boundary.

ii.
```python
        idx_beg = np.searchsorted(beh_times, t_beg, side='right')
        idx_end = np.searchsorted(beh_times, t_end, side='left')
        ...
        # Interpolation points matching reference: linspace(beg + binsize, end, n_bins)
        x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)

        interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
        values[trial_idx] = interp_func(x_interp).astype(np.float32)
```
```python
        wheel_vals, wheel_mask = interpolate_behavior_to_bins(
            wh_times, wh_speed, interval_begs, interval_ends
        )
```

iii. The helper is documented as "Matches reference code get_behavior_per_interval", with the
interpolation grid annotated line-by-line against the reference. Step 1 lists
`get_behavior_per_interval` — "Interpolates behaviors to time bins" — as the function being
reproduced. The wheel timestamps are taken to be on the same clock as the spikes, so no further
alignment is applied.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. `<side>Camera.ROIMotionEnergy.npy` with its frame times `_ibl_<side>Camera.times.npy`, the IBL
ROI motion energy over the whisker pad, used exactly as released. The left camera is preferred and the
right is the fallback. Both files are searched for in the dated revision directories `alf/*/` (and the
times file additionally at `alf/`), with the lexicographically last match taken.

ii.
```python
    # Try left camera first (matching reference code)
    left_me_files, left_time_files = _find_files(
        sdir, 'leftCamera.ROIMotionEnergy.npy', '_ibl_leftCamera.times.npy')

    if left_me_files and left_time_files:
        me = np.load(left_me_files[-1]).flatten()
        times = np.load(left_time_files[-1]).flatten()
        min_len = min(len(me), len(times))
        return times[:min_len], me[:min_len]

    # Fall back to right camera
```

iii. Step 1: "Whisker ME: tries left camera first, falls back to right"; Step 5's key decisions:
"Whisker ME source: Try left camera first, fall back to right, matching code". The reference code's
target names `left-whisker-motion-energy` / `right-whisker-motion-energy` are quoted in the trajectory
as the source of this preference order.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. None on the trace itself — no filtering, normalisation or baseline subtraction. The released values
are truncated to the length of the shorter of the value/time arrays (a guard against the two arrays
disagreeing), resampled onto each trial's 100 points with the same
`interpolate_behavior_to_bins` used for the wheel, and then discretized (8-c). Unlike the reference
code, trials whose motion-energy trace contains NaNs are not rejected (the reference's
`get_behavior_per_interval` skips them with reason "nans in target data"); such values would propagate
through `interp1d` and land in the top tercile via `np.digitize`.

ii.
```python
        me_times, me_vals_raw = load_whisker_me(sdir)
        whisker_vals, whisker_mask = interpolate_behavior_to_bins(
            me_times, me_vals_raw, interval_begs, interval_ends
        )
```
```python
        min_len = min(len(me), len(times))
        return times[:min_len], me[:min_len]
```

iii. Step 5's mapping table: "whisker ME → output[3], Discretize into 3 bins, `load_target_behavior`,
interpolate, Time-varying" — i.e. the released signal goes in untouched apart from binning and
discretization, matching `load_target_behavior`, which returns the ROIMotionEnergy array as-is.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Identically to wheel speed: the same `discretize_to_bins(..., n_bins=3)` applied to the session's
own `(n_good_trials, 100)` motion-energy matrix, cutting at that session's 33.3rd/66.7th percentiles
into "low"/"medium"/"high". Realised distribution 0.332 / 0.332 / 0.335.

ii.
```python
        whisker_discrete = discretize_to_bins(whisker_trials, n_bins=3)
```
```python
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    boundaries = np.percentile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int32)
```

iii. Step 5: "Discretization: Use session-wide terciles for wheel speed and whisker ME." Per-session
thresholds are the natural choice here because ROIMotionEnergy is in uncalibrated camera units whose
scale differs between sessions and between the left and right camera; Step 7/Step 10 verify the
resulting bins are ~33% each.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Exactly as the wheel (7-d): the camera trace is evaluated at
`np.linspace(stimOn - 0.5 + 0.02, stimOn + 1.5, 100)` — the right edges of the 100 neural bins — using
the same coverage test, so trials whose first/last camera frame inside the window is more than one bin
away from the window edge are dropped rather than extrapolated.

ii.
```python
        whisker_vals, whisker_mask = interpolate_behavior_to_bins(
            me_times, me_vals_raw, interval_begs, interval_ends
        )
```
```python
        if np.abs(t_beg - beh_t[0]) > BINSIZE:
            mask[trial_idx] = False
            continue
        if np.abs(t_end - beh_t[-1]) > BINSIZE:
            mask[trial_idx] = False
            continue
```

iii. As for the wheel: the helper is annotated "Matches reference code get_behavior_per_interval", and
the camera frame times are taken to be on the same session clock as the spikes, so evaluating the
trace on the neural grid is the whole of the alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms, of differing quality.

- *Whole sessions*: `find_session_dirs()` screens for the presence of spikes / trials / wheel / camera
  files; `process_session` is wrapped in a bare `try/except` that prints a traceback and returns
  `None`, so one bad session cannot abort the run; sessions left with `< 2` usable trials are dropped.
- *Probes*: a probe directory missing any of its four files is skipped, and the remaining probes are
  still merged.
- *Trials*: NaN `stimOn_times` (hence NaN window bounds) are skipped in both the binning and the
  interpolation; trials whose wheel or camera stream does not span the window are masked out.
- *Arrays that disagree in length*: motion energy and camera times are truncated to the shorter;
  out-of-range `clusters.channels` indices are silently `np.clip`ped into range.
- *Revisions*: for trials and motion energy the code globs `alf/*/` and takes the last match; for the
  wheel it does **not**, and this is the single largest source of loss — 57 sessions (all
  hausserlab PL0xx) passed the screen on `alf/_ibl_wheel.timestamps.npy` and then crashed on
  `alf/_ibl_wheel.position.npy`, which exists one level down in e.g. `alf/#2024-05-06#/`. The AI
  recorded these as genuinely "missing wheel data" and did not investigate further.
- Not handled: NaNs inside a behavioural trace (the reference code drops those trials).

ii.
```python
    except Exception as e:
        print(f"  ERROR processing {session_id}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None
```
```python
        if not all(os.path.exists(f) for f in [st_file, sc_file, cc_file, cb_file]):
            continue
```
```python
        if np.isnan(t_beg) or np.isnan(t_end):
            mask[trial_idx] = False
            continue
```
```python
    valid_channels = np.clip(cluster_channels, 0, len(channel_brain_ids) - 1)
```

iii. Step 6: "Graceful handling of missing data (skips sessions without wheel/whisker data)". Step 9
tabulates the losses — "Sessions failed (missing data) | 57 (all PL050/hausserlab - missing wheel
data)" and "Sessions skipped (no valid trials) | 1 (ibl_witten_19/2020-07-22)" — and Step 10 accepts
the shortfall against the paper with "Sessions | ~433 (paper) | 335 (57 missing wheel data) | OK -
data subset".

## 10-a. What are the most time-consuming steps of the code?

i. The script prints a per-session total and a dedicated timer for spike binning. Measured over the
full run (`conversion_full_out.txt`): 1485.6 s wall clock for 393 attempted sessions, 3.88 s mean per
successful session, of which spike binning is only 0.32 s. The remaining ~3.5 s/session is dominated
by file I/O — `np.load` of `spikes.times`/`spikes.clusters` (hundreds of MB per probe) plus the global
`np.argsort` over the merged spike times — followed by the two per-trial behaviour interpolation loops,
which build a fresh `scipy.interpolate.interp1d` object for every trial. The whole job runs in a single
process; no multiprocessing is used anywhere.

The AI's own attribution in CONVERSION_NOTES.md is different and is not supported by its logs: it
assigns 0.2-0.3 s/session to binning (correct) and 0.5-1 s/session to "Wheel/whisker" with a total of
"~2.5s" per session and "~16 min" overall. The actual run took 24.8 min, 1.55x the estimate, and the
unaccounted 1.4 s/session is never located.

ii.
```python
        t_bin = time.time()
        binned_spikes = bin_spikes_vectorized(...)
        print(f"  Spike binning: {time.time() - t_bin:.1f}s", flush=True)
        ...
        elapsed = time.time() - t0
        print(f"  Session {session_id}: {n_trials} trials, {n_clusters} neurons, {elapsed:.1f}s", flush=True)
```
```python
    merged_times = np.concatenate(all_spike_times)
    merged_clusters = np.concatenate(all_spike_clusters)
    sort_idx = np.argsort(merged_times, kind='stable')
```
```python
            interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
            values[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. Step 7 of CONVERSION_NOTES.md gives the run-time table ("Spike binning 0.2-0.3s | ~2 min",
"Wheel/whisker 0.5-1s | ~6 min", "Total per session ~2.5s | ~16 min"), i.e. the AI judged the estimate
to be inside the instructions' 15-minute budget and therefore did no further optimisation. Step 6
lists the optimisations it did make: "Spike binning using `np.searchsorted` + flat indexing for
efficiency", "Incremental building with `gc.collect()` after each session", "Memory monitoring via
`resource.getrusage`".

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain, plus one missing parallelisation.

- `interpolate_behavior_to_bins` loops over trials and constructs a `scipy` `interp1d` object per
  trial; `np.interp` on a precomputed `searchsorted` slice (what the reference does) avoids the object
  construction entirely, and the whole loop could be one call with a single stacked query vector.
- `bin_spikes_vectorized` loops over trials (the inner counting is already vectorized through
  `bincount`); a single `bincount` over `trial * n_units * n_bins + unit * n_bins + bin` would do all
  trials at once.
- `compute_trial_num_in_block` is a scalar Python loop over every trial; it is expressible as a
  `cumsum`/`groupby.cumcount` in two lines (as the reference writes it).
- The per-trial list comprehensions that build `neural_list`, `input_list` and `output_list` re-`stack`
  and re-`clip` trial by trial.
- Sessions are fully independent but are processed serially; the reference runs 10 worker processes,
  which is where most of the 25 minutes could have been recovered.

ii.
```python
    for trial_idx in range(n_trials):
        ...
        interp_func = interp1d(beh_t, beh_v, kind='linear', fill_value='extrapolate')
        values[trial_idx] = interp_func(x_interp).astype(np.float32)
```
```python
    for trial_idx in range(n_trials):
        ...
        flat_idx = clusters_trial * N_BINS + bin_idx
        counts = np.bincount(flat_idx, minlength=n_clusters * N_BINS)
```
```python
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            trial_nums[i] = trial_nums[i - 1] + 1
        else:
            trial_nums[i] = 1
```
```python
    for i, sdir in enumerate(session_dirs):          # no ProcessPoolExecutor
```

iii. Step 6 claims the binning loop is the one that mattered and that it was handled ("Spike binning
using `np.searchsorted` + flat indexing for efficiency"). The behaviour loop, the block-count loop and
the absence of parallelism are not mentioned in CONVERSION_NOTES.md; Step 7's conclusion that the run
would fit in ~16 min is the stated reason no further work was done.

## 10-c. What processing does the code repeat multiple times?

i. Four repeats, none of them documented.

- `BrainRegions()` — which parses the Allen/Beryl atlas tables — is constructed **once per session**
  inside `load_spikes`, in addition to the one built in `main()`. The instance `main()` builds is
  passed into `process_session` as `br` and then never used.
- The full session wheel trace is rebuilt from scratch for every session even though only the ~2 s
  windows around each stimulus onset are ever read: `np.arange` over the whole recording at 1 kHz plus
  a full-length `np.interp` and `np.gradient`.
- `interpolate_behavior_to_bins` re-runs `np.searchsorted` over the whole behaviour array twice per
  trial rather than once per session for all trials (the reference computes the slice bounds
  vectorized, once).
- `interp1d` is re-fitted per trial on overlapping data (see 10-b).

ii.
```python
def load_spikes(sdir):
    ...
    br = BrainRegions()                      # rebuilt for every session
```
```python
    br = BrainRegions()                      # built in main(), passed in, never used
    ...
        result = process_session(sdir, br, ...)
```
```python
def process_session(sdir, br, show_processing=False, session_idx=0):
    #                    ^ unused
```
```python
    t_uniform = np.arange(wh_times[0], wh_times[-1], dt)
    pos_interp = np.interp(t_uniform, wh_times, wh_pos)
```

iii. Not discussed in CONVERSION_NOTES.md. The only efficiency claims are Step 6's list of
implemented speed-ups (searchsorted binning, incremental accumulation, `gc.collect()`), and Step 7's
run-time estimate, which the AI judged good enough that no profiling of repeated work was carried out.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Three substantial items.

- **Neurons.** With QC disabled the conversion carries 1443 neurons/session on average (reference:
  164), including 11,701 `void` units the atlas places outside the brain, 63,294 `root` units, and
  every cluster with zero spikes in the session (the unit count comes from `len(clusters.channels)`,
  not from the clusters that actually fire, so all-zero rows are written out). The resulting pickle is
  21 GB and could not be loaded for training at all: the AI then wrote `reduce_data.py` to keep a
  **random** 500 neurons per session (318 of 335 sessions affected) and trained the decoder on that
  7.4 GB file — so the great majority of the binned spikes were discarded, and the ones kept are a
  random draw from an uncurated population, before the decoder's own PCA reduces each session to 100
  components anyway.
- **Trials.** `bin_spikes_vectorized` is run over *all* trials of the session and the mask is applied
  afterwards; 146,747 of ~265,000 trials survive, so roughly 45% of the spike-binning work (and the
  full `(n_trials, n_units, 100)` float32 allocation behind it) is thrown away.
- **Behaviour.** The wheel speed is reconstructed at 1 kHz for the entire session (millions of
  samples) when only 100 interpolated points per trial are used.

ii.
```python
        # 4. Bin spikes for ALL trials first (before masking)
        binned_spikes = bin_spikes_vectorized(
            spike_times, spike_clusters, n_clusters, interval_begs, interval_ends
        )
        ...
        combined_mask = mask.values & wheel_mask & whisker_mask
        good_indices = np.where(combined_mask)[0]
        neural_trials = binned_spikes[good_indices]
```
```python
        n_clusters = len(cluster_channels)       # every sorted cluster, spiking or not
```
`reduce_data.py`:
```python
max_neurons = 500
...
    idx = np.sort(np.random.choice(nneurons, max_neurons, replace=False))
```

iii. CONVERSION_NOTES.md does not flag any of this as waste. Step 11 presents the subsampling as a
memory fix — "The 21 GB uint8 pickle converts to ~84 GB float32 in `_prepare_session_data`, exceeding
64 GB cgroup limit. Solution: created `reduce_data.py` to subsample neurons to max 500 per session
(decoder uses PCA to 100 components anyway)" — and Step 12 offers the retained-but-unused neurons as
an explanation for moderate accuracy: "The moderate accuracy is expected given: all neurons (not just
well-isolated), all regions pooled, simple model."
