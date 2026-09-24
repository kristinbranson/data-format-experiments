# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from the three subfolders of `data/`: `beh/` (behavior), `spk/` (deconvolved
traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is read first
as the master index; it is a dict keyed by experiment type, each holding a list of recording entries
with `mname`, `datexp`, `blk` (and sometimes `stimtype`). From each entry the AI builds the session
key `mname_datexp_blk` and the behavior key (with `_stimtype` appended when present). All 22
`Beh_<exp_type>.npy` files are then loaded into a dict `beh_cache` and **kept resident in RAM for the
whole run**; the spike file and retinotopy file are loaded per session inside the main loop. The
trial list of a session comes from the behavior entry (`ntrials`, `StartFr`, `WallName`, ...).

ii.
```python
def get_session_list():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    sessions = {}
    for exp_type in exp_info:
        for db in exp_info[exp_type]:
            key = f'{db["mname"]}_{db["datexp"]}_{db["blk"]}'
            stimtype = db.get('stimtype', '')
            beh_key = key if not stimtype else f'{key}_{stimtype}'
```
```python
def load_spk(mname, datexp, blk, root):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(os.path.join(root, 'spk', fn), allow_pickle=True).item()
    spk = np.concatenate([s for s in spk_data['spks']], axis=0)
    return spk

def load_retino(mname, datexp, root):
    fn = f'{mname}_{datexp}_trans.npz'
    dtrans = np.load(os.path.join(root, 'retinotopy', fn), allow_pickle=True)
    return dtrans['iarea']
```
```python
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(
                os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
```

iii. From the trajectory: after reading the paper code (`utils.py`) and probing the `.npy` files, the
agent concluded "Each unique (mname, datexp, blk) combination with corresponding behavior data and
neural data" is a session, and that the experiment-type grouping in `Imaging_Exp_info.npy` is only a
labelling of the same recordings ("Good - same session data, just different experiment type
labels"). The conversion log shows all 89 sessions / 19 mice were enumerated and processed with no
session skipped.

## 1-b. How are the data split into subjects (mice)?

i. The mouse is the `mname` field of the index entry, carried on each session record. `subjects` is
the sorted set of all `mname` values over **all** sessions in the index (19 mice), and
`subject_idx` is that index per emitted session. Note the subject list is built before session
filtering, so in the `--sample` run all 19 mice are listed even though only 3 sessions are written
(verification output: "Number of subjects: 19" with 3 sessions).

ii.
```python
all_subjects = sorted(set(s['mname'] for s in sessions.values()))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx_all.append(subject_to_idx[mname])
```

iii. No explicit reasoning was recorded; the mouse name is given directly by the index, so no split
has to be derived.

## 1-c. How are the data split into sessions?

i. A session is one recording, `mname_datexp_blk`. Because the same recording is listed under
several experiment types (and sometimes with `_swap1` / `_swap2` behavior keys), the AI de-duplicates
into a dict keyed by `mname_datexp_blk` and, among the duplicate entries, keeps the one whose
`stim_id` has the most non-NaN values ("most complete stimulus coverage"). This yields 89 sessions.
A session is dropped if `ntrials < 10`, if the spike or retinotopy file fails to load, if fewer than
10 visual-cortex neurons survive, or if fewer than 10 valid trials remain; in practice the log shows
none of these fired.

ii.
```python
            stim_id = db.get('stim_id', np.array([]))
            n_stim = int(np.sum(~np.isnan(stim_id.astype(float))))

            if key not in sessions or n_stim > sessions[key]['n_stim']:
                sessions[key] = {...}
```
```python
        if beh['ntrials'] < MIN_TRIALS:
            print(f"  Skipping: only {beh['ntrials']} trials (< {MIN_TRIALS})")
            continue
```

iii. The agent explicitly checked the duplicates before deciding: "swap1 and swap2 have the same
ntrials and same WallName/UniqWalls - they just differ in which swap stimulus gets a stim_id vs nan.
The underlying trial data is identical." and "For sessions appearing in multiple experiment types
(like LZ13_2024_05_15_1 in naive_test1, naive_test2, etc.), the trials are the same but stim_id
differs. I should pick the one with the most non-nan stim_id entries."

## 1-d. How are the data split into trials?

i. Trials are the trials the behavior declares (`range(beh['ntrials'])`), and each trial is a **fixed
32-frame window starting at the rounded corridor-entry frame** `StartFr`. The per-frame trial labels
that the dataset provides (`ft_trInd`) and the in-texture mask (`ft_CorrSpc`) are **not** used to
delimit trials; `ft_move` is read but never used. Consequences measured on
`TX88_2022_06_20_1`: 20.7% of the stored bins are outside the texture corridor (grey space,
`ft_Pos` > 40 dm), 1.5% of stored bins belong to a *different* trial by `ft_trInd`, and 23% of the
real in-corridor frames (4,996 of 21,256) are truncated away because the traversal lasted more than
32 frames.

ii.
```python
N_TIMEPOINTS = 32
...
    for trial in range(ntrials):
        start_fr = int(np.round(start_frs[trial]))
        end_fr = start_fr + n_timepoints
        if start_fr < 0 or end_fr > n_frames:
            valid_mask.append(False)
            ...
            continue
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. The agent recognised the variable-length nature of the traversal ("Each trial has variable
duration since the mouse can stop running (VR pauses when mouse speed < 6 cm/s threshold)") and that
"the paper focuses on running timepoints (ft_move > 0) and corridor timepoints (ft_CorrSpc)", but
then decided: "For the decoder, I need fixed-size time bins across all trials and sessions,
temporally aligned to corridor entry, so I'll extract a consistent number of time bins per trial
regardless of how long the mouse takes to traverse." The window length was set from the VR geometry:
"At 60 cm/s and 3.17 Hz: 6m corridor takes ~32 frames".

## 1-e. How are trials filtered based on quality controls?

i. There is effectively no trial-level quality control. A trial is dropped only when its 32-frame
window does not fit inside the imaged recording (`start_fr < 0` or `end_fr > n_frames`), i.e. the
first and last trials of a session. No outlier/stopped-animal trials are removed, no minimum
traversal is required, and no check is made that the 32 frames actually belong to the trial. The
conversion log shows almost every session keeping 100% of its trials ("Valid trials: 503/503",
"431/431", ...). At the session level, `MIN_TRIALS = 10` trials are required.

ii.
```python
        if start_fr < 0 or end_fr > n_frames:
            valid_mask.append(False)
```
```python
        if len(valid_neural) < MIN_TRIALS:
            print(f"  Skipping: only {len(valid_neural)} valid trials after filtering")
            continue
```

iii. No justification for the absence of trial filtering was recorded. Implicitly the fixed 32-frame
window caps the contribution of a trial in which the animal stopped (those trials are truncated
rather than dropped), so the pathological multi-thousand-frame traversals cannot dominate the
dataset — but the agent never states this, and never inspects the trial-length distribution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<mname>_<datexp>_<blk>_neural_data.npy`, a list of (neurons x frames) arrays, one
per imaging plane, concatenated along the neuron axis. The area label of each neuron comes from
`iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
    spk = np.concatenate([s for s in spk_data['spks']], axis=0)
...
    return dtrans['iarea']
```

iii. Documented in the script header: "Neural: Deconvolved calcium traces (Suite2p), sampled at ~3.17
Hz"; the agent verified the on-disk dtype is already float32 ("the raw neural data is already
float32").

## 2-b. How is the `neural` data processed?

i. No processing at all: the deconvolved traces are sliced by the trial window and cast to float32.
No normalisation, z-scoring, smoothing or dF/F. Every trial is exactly 32 bins, so nothing is padded
and long traversals are truncated.

ii.
```python
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)  # (n_neurons, n_timepoints)
```

iii. The file already contains deconvolved traces, which the paper analyses directly. On dtype, the
agent considered the size problem explicitly ("Even with float32, 158 GB is too large for a
pickle... 219 GB is too large") but chose to keep all neurons at float32 anyway, reasoning "the task
says to match the reference paper... the paper doesn't subsample neurons" and "with 1 TB RAM, even
151 GB is fine".

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is the retinotopic area: a neuron is kept if `iarea` maps to V1 (8), mHV
(0,1,2,9), lHV (5,6) or aHV (3,4); everything else (including `iarea` -1 and 7) is dropped. Typical
retention in the log is ~90% (e.g. "Visual cortex neurons: 52246/58224"). A session with fewer than
10 surviving neurons would be skipped. No further curation (the Suite2p cell classifier has already
been applied upstream).

ii.
```python
def get_brain_region_idx(iarea):
    region_idx = np.full(len(iarea), -1, dtype=int)
    region_idx[iarea == 8] = 0  # V1
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    region_idx[(iarea == 5) | (iarea == 6)] = 2  # lHV
    region_idx[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return region_idx
...
        valid_neurons = region_idx >= 0  # Exclude iarea -1 and 7
        if valid_neurons.sum() < 10:
            continue
        spk_filtered = spk[valid_neurons]
```

iii. The header states "Brain regions: V1, mHV, lHV, aHV (from retinotopy); Exclude neurons outside
visual cortex (iarea == -1 or 7)", and the mapping is taken from the paper's `utils.py`
(`neu_area_ID`), which the agent read at the start of the run.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is on corridor entry: the window starts at `round(StartFr[trial])` and runs for exactly
32 frames (~10.1 s), so `off_start = 0.0` and `off_end = 32 / 3.17 = 10.09 s`. Every trial in every
session has T = 32 (verification output: "T: mean: 32.00, median: 32.00, min: 32, max: 32"). The
alignment event is therefore correct, but the fixed window neither ends at the traversal's end nor
is padded: ~21% of stored bins fall past the texture into the grey space (and 1.5% into the next
trial), while ~23% of real in-corridor frames are discarded.

ii.
```python
        start_fr = int(np.round(start_frs[trial]))
        end_fr = start_fr + n_timepoints
        ...
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```
```python
            'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0,
            'off_end': N_TIMEPOINTS / FS,
```

iii. "For the decoder, I need fixed-size time bins across all trials and sessions, temporally aligned
to corridor entry, so I'll extract a consistent number of time bins per trial regardless of how long
the mouse takes to traverse. 10 seconds translat[es to ~32 frames]" — i.e. the format requirement
"Time bins should be the same size for all trials and sessions" was read as a requirement of the
same *number* of bins. The 32 was chosen from the VR geometry: 6 m at 60 cm/s = 10 s at 3.17 Hz.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame. No rebinning, resampling or interpolation is done; `time_bin_size` is
1000/3.17 = 315.46 ms, and each trial holds 32 such bins.

ii.
```python
FS = 3.17  # Calcium imaging frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms per frame
...
            'time_bin_size': TIME_BIN_MS,
            'n_timepoints': N_TIMEPOINTS,
            'frame_rate_hz': FS,
```

iii. The frame rate was taken from the paper/reference notebook; the imaging frame is the finest
resolution available and all behavior streams are already sampled on the same frame grid, so nothing
needs resampling. (The paper's own position-binning into 60 spatial bins was explicitly rejected:
"The paper normalizes this by interpolating neural activity into 60 position bins per corridor, but
I need time-aligned data instead.")

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (frame of the sound cue in each trial) and `StartFr` (corridor-entry frame), plus
the nominal frame rate `FS`. The actual frame timestamps `ft` are not used.

ii.
```python
        sound_frs = beh['SoundFr']
        ...
        sound_fr_rel = sound_frs[trial] - start_frs[trial]
```

iii. Not explicitly justified; the agent verified the semantics of `SoundFr` while probing the
behavior dict and noted the cue can be very late in a trial ("some sound cues are very far from
trial start (643 frames!). This is because some trials have very long duration when mouse stops
running").

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue frame is expressed relative to the (unrounded) trial start and subtracted from the
within-trial frame index, then divided by the frame rate to get seconds:
`t[k] = (k - (SoundFr - StartFr)) / FS`. This is **time since the cue** — negative before the cue,
positive after — i.e. the opposite sign convention to the variable's name (and to the expert's
`cue - time`). Because the window is fixed while the cue can occur hundreds of frames later on slow
trials, values reach -401 s in the sample data.

ii.
```python
        sound_fr_rel = sound_frs[trial] - start_frs[trial]
        time_to_cue = np.arange(n_timepoints) - sound_fr_rel  # negative before cue, positive after
        time_to_cue_sec = time_to_cue / FS  # convert to seconds
```

iii. The agent reasoned about the extreme values and accepted them: "A sound frame 403 frames after
trial start means at frame 0, we're 403 frames away from it, which is why I'm seeing values like
-403. This is mathematically correct".

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built on the same 32-frame index grid as the neural slice of that trial (`np.arange(32)`
counted from the same `start_fr`), so element k of the input corresponds to column k of the neural
matrix. The only slippage is that the neural slice starts at `round(StartFr)` while the cue offset
is computed from the unrounded `StartFr` (a sub-frame, <160 ms, offset).

ii.
```python
        trial_neural = spk[:, start_fr:end_fr]
        ...
        time_to_cue = np.arange(n_timepoints) - sound_fr_rel
        trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. All streams in this dataset are indexed by imaging frame, so a shared frame window aligns them.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session dates: `datexp` of every session of that mouse in the de-duplicated session
index, parsed as a calendar date.

ii.
```python
def compute_day_of_training(mname, datexp, all_sessions):
    from datetime import datetime
    mouse_dates = []
    for k, s in all_sessions.items():
        if s['mname'] == mname:
            mouse_dates.append(datetime.strptime(s['datexp'], '%Y_%m_%d'))
    mouse_dates.sort()
```

iii. Not explicitly justified in the trajectory; the date string is the only field that orders a
mouse's sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The value is the number of **calendar days** between the session's date and the earliest recorded
date for that mouse (0 for the first session, up to 67 in the sample output), not the ordinal index
of the recording session. It is constant within a session and broadcast across all 32 bins of every
trial (filled in after `extract_trial_data`, overwriting a zeros placeholder).

ii.
```python
    current_date = datetime.strptime(datexp, '%Y_%m_%d')
    day_idx = (current_date - mouse_dates[0]).days
    return day_idx
```
```python
        day = compute_day_of_training(mname, datexp, sessions)
        for i in range(len(input_trials)):
            if valid_mask[i]:
                input_trials[i][1, :] = float(day)  # day_of_training row
```

iii. The agent's own instruction listed "Day of training, continuous, time-varying", and the README
documents the choice as "day_of_training: Days since first recording session for this mouse". No
discussion of the alternative (counting recorded sessions) is recorded.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the within-window frame index alone, converted to seconds with the constant frame rate `FS`.
Implicitly from `StartFr`, which defines where the window begins; `ft` is not used.

ii.
```python
        time_since_start = np.arange(n_timepoints) / FS  # in seconds
```

iii. Not explicitly justified. Sampling is regular at 3.17 Hz, so the frame index is a faithful clock.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `k / FS` for k = 0..31, i.e. 0 to 9.78 s. It is therefore identical for every trial and every
session (verification: "time_since_trial_start: [0.0, 9.8]" in all sessions), and it does not
account for the sub-frame offset between the true `StartFr` and the rounded window start.

ii.
```python
        time_since_start = np.arange(n_timepoints) / FS  # in seconds
        trial_input = np.stack([time_to_cue_sec, day_of_training, time_since_start, reward_avail])
```

iii. Not explicitly justified beyond the fixed-window design; with a fixed window the variable is
just the bin index in seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same 32-frame window as the neural slice, index-for-index, starting at corridor entry
(`off_start = 0`).

ii.
```python
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
        ...
        time_since_start = np.arange(n_timepoints) / FS
```

iii. All streams share the imaging-frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
        is_rew = beh['isRew']
        ...
        reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. Directly given by the behavior file; no reasoning needed.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float (0.0/1.0) and broadcast constant across the 32 bins of the trial.

ii.
```python
        reward_avail = np.full(n_timepoints, float(is_rew[trial]))
```

iii. None recorded. The agent did note while inspecting the sample that unsupervised/naive sessions
have `isRew` false throughout ("licking is all zeros (these are unsupervised sessions with no
rewards), and reward_availability is also all 0"), and changed the sample selection so that
supervised sessions were represented.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (the texture shown on each trial) and `UniqWalls` (the textures used in a
session), which are pooled over all sessions to build a global label set. `TrialStim` / `stim_id`
are used only to pick between duplicate behavior entries, not to label trials.

ii.
```python
        wall_names = beh['WallName']
        ...
        wall_name = wall_names[trial]
        stim_cat = wall_name  # e.g., 'circle1', 'leaf2', etc.
```
```python
            for w in beh['UniqWalls']:
                all_stim_names.add(w)
```

iii. `WallName` is the per-trial texture name; the agent used it directly rather than the
`stim_id` codes, which are NaN-masked for the stimuli not analysed in a given experiment type.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 distinct wall names found across the dataset are sorted alphabetically and each trial gets
the index of its own wall name — i.e. the **fine-grained texture instance** (`circle1`, `circle2`,
`leaf1_swap1`, ...) is the category, not the four base textures (circle / leaf / rock / wood) the
expert used. The label is per trial, broadcast across the 32 bins. An unknown wall name would
silently fall back to class 0 (`.get(..., 0)`). Because each session shows only 2-4 of the 15
labels, most classes are absent in any given session (verification: `visual_stimulus: [8.0, 10.0]`
for one session).

ii.
```python
    all_stim_sorted = sorted(str(s) for s in all_stim_names)
    stim_to_idx = {}
    for exp_type in beh_cache:
        for beh_key_inner in beh_cache[exp_type]:
            for w in beh_cache[exp_type][beh_key_inner]['UniqWalls']:
                stim_to_idx[str(w)] = all_stim_sorted.index(str(w))
```
```python
                stim_idx = stim_to_idx.get(str(stim_cats[i]), 0)
                output_trials[i][0, :] = stim_idx  # constant across time
```
```python
        [str(s) for s in all_stim_sorted],  # stimulus category names (plain strings)
```

iii. The instruction the agent received read "Visual stimulus category. e.g. circle1, leaf2, etc.,
per-trial", i.e. it named the fine-grained wall identifiers as the example categories, and the agent
followed that literally; the README records "15 stimulus categories across all sessions".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (imaging-frame number of each lick) together with `LickTrind` (the trial each lick
is assigned to).

ii.
```python
    lick_frs = beh['LickFr']
    lick_trinds = beh['LickTrind']
```

iii. Not explicitly justified; both fields are read directly from the behavior dict.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the licks whose `LickTrind` equals that trial are selected, their frame numbers
rounded to the nearest frame, converted to a window offset, and the corresponding bin set to 1
(multiple licks in a bin collapse to 1). Licks that occur inside the 32-frame window but are
attributed to a neighbouring trial are not marked. The loop over licks is a Python `for`.

ii.
```python
        lick_binary = np.zeros(n_timepoints, dtype=float)
        trial_lick_mask = lick_trinds == trial
        if trial_lick_mask.any():
            trial_lick_frs = lick_frs[trial_lick_mask]
            for lf in trial_lick_frs:
                fr_idx = int(np.round(lf)) - start_fr
                if 0 <= fr_idx < n_timepoints:
                    lick_binary[fr_idx] = 1.0
```

iii. None recorded beyond the format requirement ("Licking, binary, time-varying"). The agent did
sanity-check lick rates on the sample (0.11 of bins are licks) and re-picked sample sessions so that
licking was non-degenerate.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is in imaging-frame units, so the flag lands on the same grid as the neural columns; the
offset is taken against the same `start_fr` used to slice `spk`, giving a bin-for-bin match over the
32-frame window.

ii.
```python
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
        ...
                fr_idx = int(np.round(lf)) - start_fr
```

iii. All streams are indexed by imaging frame.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the VR position of the animal at each imaging frame, in decimeters (0-40 across the
4 m texture, 40-60 through the 2 m grey space).

ii.
```python
    ft_pos = beh['ft_Pos'][:n_frames]
    ...
        trial_pos = ft_pos[start_fr:end_fr]
```

iii. The agent read the corridor geometry out of the paper code and encoded it as constants
(`CORRIDOR_LENGTH_DM = 60`, `TEXTURE_LENGTH_DM = 40`, `GRAY_LENGTH_DM = 20`).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw decimeter position of the frames in the trial window is taken and digitized; no smoothing,
no unit change, no restriction to in-corridor frames (`ft_CorrSpc` is not used).

ii.
```python
        trial_pos = ft_pos[start_fr:end_fr]
        pos_binned = discretize_position(trial_pos)
```

iii. "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins" — taken directly
from the task specification.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `np.digitize` with boundaries at 10, 20, 30 dm, giving [0,1) m, [1,2) m, [2,3) m and
**everything >= 3 m** in the last bin. Since the fixed window runs past the texture, the grey-space
frames at 40-60 dm (4-6 m) are also assigned to the "3-4m" bin: on `TX88_2022_06_20_1` 20.7% of all
stored bins have `ft_Pos` > 40 dm and are labelled "3-4m", so the last class (36.7% of bins) is more
than half mislabelled grey-space data.

ii.
```python
def discretize_position(pos_values, n_bins=4):
    """... Values in gray space (>40dm) get assigned to the last bin."""
    boundaries = [10, 20, 30]  # in decimeters
    binned = np.digitize(pos_values, boundaries)  # 0,1,2,3
    return binned
```
```python
        ['0-1m', '1-2m', '2-3m', '3-4m'],  # position bins (4m texture corridor)
```

iii. The docstring states the choice ("Values in gray space (>40dm) get assigned to the last bin"),
but no reason is given for folding them in rather than excluding those frames; the trajectory does
not discuss it.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one sample per imaging frame and is sliced with the same `start_fr:end_fr` window as
the neural matrix, so it is bin-for-bin aligned (including the frames that belong to the grey space
or, occasionally, to the next trial).

ii.
```python
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
        trial_pos = ft_pos[start_fr:end_fr]
```

iii. All streams are on the imaging-frame grid; `ft_Pos` is additionally truncated to the number of
imaged frames (`beh['ft_Pos'][:n_frames]`).

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
    ft_speed = beh['ft_RunSpeed'][:n_frames]
    ...
            speeds = beh['ft_RunSpeed']
            valid_speed = speeds[~np.isnan(speeds)]
            all_speeds.append(valid_speed)
```

iii. Directly given; the agent only checked for NaNs.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Two passes. First, a per-trial quartile binning is computed in `extract_trial_data` (and then
thrown away). Second, before the main loop, the AI pools `ft_RunSpeed` from **every frame of every
session** — including inter-trial, grey-space and non-imaged frames that never enter the converted
dataset — and takes the global 25/50/75th percentiles, which come out at [0.0, 9.667, 31.456]. Every
trial's speeds are then re-digitized with those three global boundaries.

ii.
```python
    all_speeds = np.concatenate(all_speeds)
    speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
    print(f"Global speed quartile boundaries: {speed_quartiles}")
```
```python
                    trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
                    speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
                    output_trials[i][3, :] = speed_binned.astype(int)
```

iii. The agent chose global rather than per-session boundaries "across ALL sessions for consistent
discretization" so that a speed label means the same thing in every session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, [0.0, 9.667, 31.456])`. Because ~19% of frames sit at exactly 0 cm/s and only
~4% are negative, the first boundary of 0.0 is degenerate: bin 0 collects only the (rare) negative
speeds and all the zeros fall into bin 1. The resulting distribution is nowhere near the required
25% per bin — the sample verification reports Q1 0.072, Q2 0.582, Q3 0.304, Q4 0.042, and the
per-session numbers are worse still (one session has 0.000 in Q4).

ii.
```python
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # e.g., [25, 50, 75]
    boundaries = np.percentile(valid, percentiles)
    binned = np.digitize(speed_values, boundaries)  # 0 to n_bins-1
```
```python
        ['Q1', 'Q2', 'Q3', 'Q4'],  # speed quartiles
```

iii. The agent noticed the problem twice and did not act on it: "investigating the speed quartile
where Q1 is at 0.0—indicating many frames where the mouse isn't moving at all" and later "The speed
distribution is skewed - Q1 boundary is 0 (many stationary frames)". No tie-breaking (rank-based
split) was attempted, although the task specifies "Running speed discretized into 4 bins, each
corresponding to 25% of the data".

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one sample per imaging frame and is sliced with the same `start_fr:end_fr`
window as the neural matrix, bin for bin. (In the re-binning pass the slice is taken from the
untruncated `beh['ft_RunSpeed']`, guarded by `end_fr <= len(...)`; since the trial was only kept when
`end_fr <= n_imaged_frames`, this is the same window.)

ii.
```python
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
        ...
                    trial_speed = beh['ft_RunSpeed'][start_fr:end_fr]
```

iii. All streams are on the imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several small guards: behavior frame streams are truncated to the number of imaged frames
(`beh['ft_Pos'][:n_frames]` etc., with `n_frames = spk.shape[1]`); trials whose window does not fit
in the recording are dropped; licks falling outside the window are ignored; NaN speeds are excluded
when computing the global percentiles; failures to load a spike or retinotopy file are caught and the
session skipped; a behavior key missing from its file produces a warning and a skip; unknown wall
names silently default to class 0. Dimension assertions are run at the end. NaNs in `ft_RunSpeed`
would still be digitized into the top bin (none exist in practice). One material outcome: the
delivered `/app/converted_data.pkl` (22.8 GB) is a truncated pickle — `pickle.load` fails with
`EOFError: Ran out of input` — consistent with the full run being killed for memory (the log stops at
session 47/89, after the agent measured 237 GB of RAM in use).

ii.
```python
        n_frames = spk.shape[1]
        ft_pos = beh['ft_Pos'][:n_frames]
        ft_move = beh['ft_move'][:n_frames]
        ft_speed = beh['ft_RunSpeed'][:n_frames]
```
```python
        try:
            spk = load_spk(mname, datexp, blk, DATA_ROOT)
        except Exception as e:
            print(f"  ERROR loading spk: {e}")
            continue
```
```python
    valid = speed_values[~np.isnan(speed_values)]
    if len(valid) == 0:
        return np.zeros_like(speed_values, dtype=int)
```

iii. The agent discovered while probing that behavior arrays can be longer than the imaging
(`ft` / `ft_Pos` vs `spk.shape[1]`) and cut all frame streams to the imaged length, matching the
reference code's `beh[...][:nfr]`. The dataset is otherwise clean.

## 12-a. What are the most time-consuming steps of the code?

i. (1) Reading the per-session spike files — 89 files, several GB each (~405 GB total); the log shows
a single 4 GB file taking minutes and the agent noting "Each session loads in ~7 seconds ... the 4 GB
file is taking a while to load". (2) Loading **all 22 behavior files into `beh_cache` and keeping
them resident** (~5 GB) and then walking them twice more (once for stimulus names, once to pool every
speed sample). (3) Accumulating every session's trials in RAM before a single `pickle.dump` at the
end — this is what made the run unsustainable (237 GB resident at session 47 of 89) and left the
output pickle incomplete.

ii.
```python
            beh_cache[exp_type] = np.load(
                os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
```
```python
        spk = np.concatenate([s for s in spk_data['spks']], axis=0)
```
```python
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
```

iii. The agent repeatedly estimated the output size (151-219 GB) and decided to keep all neurons at
float32 regardless ("with 1 TB RAM, even 151 GB is fine. Let me just do it"), then at the end
concluded "237 GB used already and we're only at session 47/89 ... I need to reconsider the
approach" and killed the run.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The per-lick Python loop `for lf in trial_lick_frs` — a single `np.add.at` / fancy-index
assignment over all licks of the session would do. (2) The per-trial loop in `extract_trial_data`
itself: since every window is the same length, all windows could be gathered with one
`np.arange(32) + start_frs[:, None]` fancy index. (3) The separate re-binning loop over trials for
speed, which recomputes `start_fr`/`end_fr` and slices again trial by trial. (4)
`compute_day_of_training`, which rescans the whole session dict (and re-parses every date) once per
session. All are negligible next to the spike-file I/O.

ii.
```python
            for lf in trial_lick_frs:
                fr_idx = int(np.round(lf)) - start_fr
                if 0 <= fr_idx < n_timepoints:
                    lick_binary[fr_idx] = 1.0
```
```python
        for i in range(len(output_trials)):
            if valid_mask[i]:
                start_fr = int(np.round(beh['StartFr'][i]))
                end_fr = start_fr + N_TIMEPOINTS
```

iii. Not discussed in the trajectory.

## 12-c. What processing does the code repeat multiple times?

i. (1) Speed discretization is done twice for every trial: `discretize_speed` computes per-trial
percentile bins inside `extract_trial_data`, and the result is immediately overwritten by the
global-quartile re-binning pass. (2) The trial window (`start_fr`, `end_fr`) is recomputed in that
second pass. (3) The behavior files are traversed three times (stimulus-name pass, speed-pooling
pass, main conversion), while all of them stay in memory. (4) `stim_to_idx` is rebuilt by looping
over every behavior key and calling `all_stim_sorted.index(w)` (a linear search) for each wall name,
although the sorted list already defines the mapping. (5) `compute_day_of_training` re-derives the
mouse's date list from scratch for every session.

ii.
```python
        speed_binned = discretize_speed(trial_speed, n_bins=4)   # ... later overwritten:
                    speed_binned = np.digitize(trial_speed, speed_quartiles)  # 0-3
                    output_trials[i][3, :] = speed_binned.astype(int)
```
```python
    for exp_type in beh_cache:
        for beh_key_inner in beh_cache[exp_type]:
            for w in beh_cache[exp_type][beh_key_inner]['UniqWalls']:
                stim_to_idx[str(w)] = all_stim_sorted.index(str(w))
```

iii. Not discussed; the duplication is an artefact of the script being patched (the global-quartile
pass was added after the per-trial version had been written, without removing the latter).

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The per-trial speed quartiles computed in `extract_trial_data` are discarded wholesale.
(2) `ft_move` is loaded and sliced every session but never used (the header even claims "Only include
running frames (VR moving) per paper methods", which the code does not do). (3) The placeholder rows
for `day_of_training` and stimulus are written and then overwritten. (4) `stim_id` / `n_stim` are
carried in the session dict only for de-duplication. (5) Most consequentially, the neural data is
stored as float32 rather than float16 and for all ~46k neurons per session, doubling a file that the
decoder reduces to 100 PCs / a 2000-neuron random projection anyway — the agent verified this
("the decoder uses PCA (npcs=100) ... uses random projection", "Having 40,000+ neurons for 100 PCA
components is massive overkill") and still kept everything, which is what exhausted memory.

ii.
```python
        trial_speed = ft_speed[start_fr:end_fr]
        speed_binned = discretize_speed(trial_speed, n_bins=4)     # discarded
```
```python
    ft_move = beh['ft_move'][:n_frames]                            # never used
```
```python
        day_of_training = np.zeros(n_timepoints)  # placeholder
```
```python
        trial_neural = spk[:, start_fr:end_fr].astype(np.float32)
```

iii. On the neural dtype/size the agent reasoned explicitly and chose fidelity to the paper over
practicality: "The task says to match the reference paper... the paper doesn't subsample neurons.
However, for the decoder, using all neurons is unnecessary and would create an impractical file" —
and then kept them all. The other discarded computations are not discussed.
