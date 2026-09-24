# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the same three directories as the reference (`data/beh`, `data/spk`, `data/retinotopy`), but drives the enumeration from the **spike files** rather than from the master index: `sorted(SPK_DIR.glob("*_neural_data.npy"))` gives 89 sessions, and the session id is the filename stem (`mouse_YYYY_MM_DD_blk`). Behavior is located through a separate index built by loading *every* `Beh_*.npy` file once (excluding `Imaging_Exp_info.npy` and `example_bef_and_aft_learning_behavior.npy`) and recording, for every behavior key, summary statistics (`uniq_walls`, number of finite `stim_id`, count of `'stimulus_of_trial'` placeholders, whether the key has a `_swap1/_swap2` suffix). Behavior keys are reduced to a raw session key by stripping a `_swap*` suffix, so several behavior copies can map to one recording; `select_behavior_candidate` then picks the "richest" copy by that score. `Imaging_Exp_info.npy` is used only to attach the list of experiment types to each session as metadata, not to enumerate sessions. Retinotopy is read per session from `<mouse>_<date>_trans.npz`. The behavior file for a session is then loaded again (in full) during planning and twice more during conversion.

ii.
```python
spk_files = sorted(SPK_DIR.glob("*_neural_data.npy"))
for spk_file in spk_files:
    raw_key = spk_file.name.replace("_neural_data.npy", "")
    subject, _, block = parse_session_key(raw_key)
    date = session_date(raw_key)
    beh_fname, beh_key = select_behavior_candidate(raw_key, behavior_index[raw_key])
    beh = np.load(BEH_DIR / beh_fname, allow_pickle=True).item()[beh_key]
    ...
    retino = np.load(RETINO_DIR / f"{raw_key.rsplit('_', 1)[0]}_trans.npz", allow_pickle=True)
    region_idx = region_index_from_iarea(retino["iarea"])
```
```python
def select_behavior_candidate(raw_key, candidates_for_session):
    def score(item):
        return (item["uniq_walls"], item["n_finite_stim_id"], -item["placeholder_count"],
                -item["has_swap_suffix"], item["file"], item["key"])
    best = max(candidates_for_session, key=score)
    return best["file"], best["key"]
```
```python
def build_experiment_type_index():
    info = np.load(BEH_DIR / "Imaging_Exp_info.npy", allow_pickle=True).item()
    ...
```

iii. Stated in the trajectory: the AI found that "`Imaging_Exp_info.npy` has 142 analysis entries but only 89 unique recordings, because the same raw recording is reused across different paper figures and some `test3` sessions are split by `stimtype`", and that the `swap1/swap2` keys must be deduplicated to the full raw key including block ("the `swap1/swap2` sessions should deduplicate to the full raw key including block, not the shorter date-only key"). Enumerating from the spike directory guarantees every converted session has neural data, and scoring behavior copies by unmasked-stimulus content is meant to pick the least-masked copy of a recording (in the swap sessions `TrialStim`/`stim_id` are partially masked).

## 1-b. How are the data split into subjects?

i. The subject is the first underscore-separated field of the session key (the mouse name, identical to `mname` in the master index). Subjects are the sorted unique mouse names (19), and `subject_idx` is each session's index into that list. Sessions stay in spike-file (i.e. mouse-then-date) sorted order, so sessions of a mouse are contiguous.

ii.
```python
def parse_session_key(raw_key: str):
    parts = raw_key.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected raw session key: {raw_key}")
    subject = parts[0]
    ...
```
```python
subjects = sorted({plan.subject for plan in session_plans})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.asarray([subject_to_index[plan.subject] for plan in session_plans], dtype=np.int16),
```

iii. Not explicitly justified in the trajectory. The file-naming convention `mouse_year_month_day_block` is checked by `parse_session_key` (it raises if a key does not have exactly five fields), so the mouse identity is taken directly from the data rather than derived.

## 1-c. How are the data split into sessions?

i. One session = one spike file = one mouse/date/block triple; 89 sessions. Because the enumeration is over spike files, deduplication is automatic: a recording listed under several experiment types, or stored under several behavior keys (`..._swap1`, `..._swap2`), still yields exactly one session, with a single behavior copy chosen by `select_behavior_candidate`. All experiment types a recording appears under are stored in `metadata.session_info[i]['experiment_types']`.

ii.
```python
def raw_session_key_from_behavior_key(key: str) -> str:
    if key.endswith("_swap1") or key.endswith("_swap2"):
        return key.rsplit("_", 1)[0]
    return key
```
```python
candidates[raw_key].append({...})          # several behavior copies per recording
...
beh_fname, beh_key = select_behavior_candidate(raw_key, behavior_index[raw_key])
```
```python
"experiment_types": plan.experiment_types,
```

iii. From the trajectory: the AI explicitly counted 142 index entries vs 89 unique recordings and decided the recording (mouse, date, block) is the session unit, and separately fixed a bug where the swap keys were being reduced to a date-only key.

## 1-d. How are the data split into trials?

i. Trials are the ones the behavior declares (`ntrials`), and a trial's frames are exactly the imaging frames the behavior labels with that trial *and* marks as inside the corridor texture: `(ft_trInd == tr) & ft_CorrSpc`. This is the identical rule to the reference. Frames are not dropped inside a trial (no `ft_move` running filter is applied), trials keep their natural variable length, and nothing is padded.

ii.
```python
ft_tr = np.asarray(beh["ft_trInd"][:nfr])
corr = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
...
for tr in range(ntrials):
    idx = np.flatnonzero((ft_tr == tr) & corr)
```

iii. From the trajectory: "aligned trials to corridor entry and kept corridor frames only", and earlier "raw imaging frames, trial-aligned to corridor entry, restricted to corridor time rather than the paper's position interpolation". The paper's own analyses interpolate onto position and keep only running frames; the AI deliberately kept the native frame grid and all corridor frames so the trial time series stays temporally contiguous for a decoder.

## 1-e. How are trials filtered based on quality controls?

i. Three per-trial filters, all applied in `compute_trial_plans_from_behavior`:
  - fewer than `MIN_FRAMES_PER_TRIAL = 5` corridor frames → dropped (this also removes trials that were never imaged);
  - corridor duration `Gray_space_time - Trial_start_time > TRIAL_DURATION_MAX_S = 40 s` → dropped (animals that stopped inside the corridor);
  - largest gap between consecutive imaging frames inside the trial `> MAX_FRAME_GAP_S = 1 s` → dropped (imaging interruptions).
  The thresholds are fixed constants, not measured from the data. The filter is recorded in `metadata['trial_filter']`. In addition, a whole session with fewer than 2 surviving trials raises `RuntimeError`, which aborts the entire conversion rather than skipping that session. Result: 37,283 trials kept (reference: 37,728), longest kept trial 127 frames ≈ 40 s (reference: 238 frames ≈ 75 s), shortest 11 frames (same as reference, so the 5-frame minimum is effectively only removing un-imaged trials).

ii.
```python
TRIAL_DURATION_MAX_S = 40.0
MAX_FRAME_GAP_S = 1.0
MIN_FRAMES_PER_TRIAL = 5
```
```python
idx = np.flatnonzero((ft_tr == tr) & corr)
if idx.size < MIN_FRAMES_PER_TRIAL:
    continue
duration_s = float((gray_time[tr] - trial_start[tr]) * 24.0 * 3600.0)
max_gap_s = 0.0
if idx.size > 1:
    max_gap_s = float(np.max(np.diff(ft[idx]) * 24.0 * 3600.0))
if duration_s > TRIAL_DURATION_MAX_S or max_gap_s > MAX_FRAME_GAP_S:
    continue
```
```python
if len(trial_plans) < 2:
    raise RuntimeError(f"Session {raw_key} has fewer than 2 valid trials after filtering ...")
```

iii. From the trajectory: "Most trials have about 16 to 29 imaging frames in the corridor, but there are a few extreme outliers", "There are many pathological trials with 400 to 1700 second 'corridors,' so those can't be treated as normal task epochs", and "I've narrowed it down to isolated bad trials rather than a global file mismatch", so the AI filtered individual trials instead of discarding recordings. The final summary states: "filtered pathological trials with corridor duration `> 40 s`, frame gaps `> 1 s`, or fewer than `5` corridor frames". The AI did run threshold sweeps over the whole dataset (steps 68–69) before choosing the round numbers.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<session_id>_neural_data.npy` — a list of one (neurons × frames) array per imaging plane, concatenated along neurons in plane order (implicitly, by writing the planes one after another into the columns of the session memmap). The visual area of each neuron comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`. A hard check asserts that the total neuron count equals the length of `iarea`.

ii.
```python
spk_obj = np.load(SPK_DIR / f"{plan.raw_key}_neural_data.npy", allow_pickle=True).item()
spk_chunks = spk_obj["spks"]
...
total_neurons = int(sum(chunk.shape[0] for chunk in spk_chunks))
if total_neurons != len(plan.region_idx):
    raise RuntimeError(f"Region count mismatch for {plan.raw_key}: ...")
```
```python
retino = np.load(RETINO_DIR / f"{raw_key.rsplit('_', 1)[0]}_trans.npz", allow_pickle=True)
region_idx = region_index_from_iarea(retino["iarea"])
```

iii. The AI inspected the spike files early ("the authors' loader concatenates `spks` chunks into one neuron-by-frame matrix") and verified the plane ordering matches the retinotopy indexing by the explicit count check.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are **not** transformed at all — no dF/F, no z-scoring, no smoothing, no rebinning — only cast to `float16` and sliced to the kept corridor frames. The storage strategy is the main decision: instead of putting arrays into the pickle, each session's kept frames are written to a sidecar binary file `/app/converted_data_neural/session_<idx>_<session>.dat` of shape (n_kept_frames, n_neurons), and each trial in the pickle is a `MappedTrialArray` — an `np.ndarray` subclass whose `__reduce__` stores only `(path, base_shape, dtype, row_start, row_stop)` and which reconstructs on unpickling as a transposed view `base[row_start:row_stop, :].T`, i.e. (n_neurons, n_timepoints). `converted_data.pkl` is therefore only 39 MB and the real data is 108 GB of sidecars. I verified the contents are exact: for session 0 trials 0/5/100, the stored arrays equal `np.concatenate(spks,0)[:, frame_idx].astype(np.float16)` element for element. I also verified that the pickle **fails to load** unless `/app` is importable (`ModuleNotFoundError: No module named 'convert_data'` when unpickled from another working directory), because the reconstruction function is pickled by reference to a `convert_data` module.

ii.
```python
base = np.memmap(out_path, dtype=np.float16, mode="w+", shape=base_shape)
col_start = 0
time_block = 4096
for chunk in spk_chunks:
    n_chunk = int(chunk.shape[0])
    for c0 in range(0, keep_idx.size, time_block):
        c1 = min(c0 + time_block, keep_idx.size)
        block = chunk[:, keep_idx[c0:c1]].T
        base[c0:c1, col_start:col_start + n_chunk] = block
    col_start += n_chunk
```
```python
def rebuild_mapped_trial(path, base_shape, dtype_str, row_start, row_stop):
    base = get_cached_memmap(path, base_shape, dtype_str)
    arr = base[row_start:row_stop, :].T.view(MappedTrialArray)
    arr._mapped_meta = (path, tuple(base_shape), dtype_str, int(row_start), int(row_stop))
    return arr

class MappedTrialArray(np.ndarray):
    def __reduce__(self):
        return (rebuild_mapped_trial, self._mapped_meta)
MappedTrialArray.__module__ = "convert_data"
```

iii. From the trajectory: "The full-trial conversion is likely tens of GB even in `float16`, and `pickle.load` would try to materialize all of it. I'm testing whether `np.memmap` survives pickling by reference; if it does, I can keep the dataset verifier-compatible while avoiding an 80+ GB RAM spike on load", and "each trial pickles only `(path, shape, row range)`, and unpickling reuses one memmap per session file". The AI tested cross-process unpickling before the full run, and validated the final file with `train_decoder.py --verify-only`. No justification is given for not transforming the traces beyond the paper's statement that all analyses use deconvolved fluorescence.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron is dropped. Every Suite2p cell in the spike file is kept, including cells whose retinotopic area is outside the four visual-area groups: `iarea == 8 → V1`, `{0,1,2,9} → mHV`, `{3,4} → aHV`, `{5,6} → lHV`, and everything else (`-1` unassigned and area `7`) is labelled `"unknown"`, which is added as a fifth entry of `brain_regions`. This keeps 585,641 extra neurons (12.5% of all cells) that the reference discards; the four visual-area counts are identical to the reference (V1 1,833,035; mHV 1,108,860; aHV 668,180; lHV 495,318).

ii.
```python
REGION_NAMES = ["V1", "mHV", "aHV", "lHV", "unknown"]

def region_index_from_iarea(iarea):
    out = np.full(iarea.shape, REGION_TO_INDEX["unknown"], dtype=np.int16)
    out[iarea == 8] = REGION_TO_INDEX["V1"]
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = REGION_TO_INDEX["mHV"]
    out[(iarea == 3) | (iarea == 4)] = REGION_TO_INDEX["aHV"]
    out[(iarea == 5) | (iarea == 6)] = REGION_TO_INDEX["lHV"]
    return out
```

iii. From the trajectory: the AI enumerated the `iarea` codes present across all sessions (`[-1, 0, ..., 9]`) and decided to "use coarse paper-style region groups: `V1`, `mHV`, `aHV`, `lHV`, `unknown`". The implicit rationale is that the paper's grouping is reproduced for the mapped cells while no recorded neuron is thrown away, since extra input neurons cannot hurt a decoder. The AI does not comment on the fact that this deviates from the paper's curation, nor that `unknown` pools unassigned cells with the genuine area-7 cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is corridor entry: a trial's frame list starts at the first imaged frame inside the corridor for that trial (`ft_trInd == tr & ft_CorrSpc`) and runs to the last one, so every trial's column 0 is its own corridor entry. Trials keep their natural, variable length (median 23 frames, max 127); nothing is padded or truncated to a common window. Metadata records `temporal_alignment_event = "corridor entry (trial start)"`, `off_start = 0.0`, `off_end = None`. The neural, input and output arrays of a trial are all built from the same `frame_idx`, so they have identical length (verified: all three are (·, 74) for session 0 trial 0).

ii.
```python
idx = np.flatnonzero((ft_tr == tr) & corr)
...
keep_idx = np.concatenate([tp.frame_idx for tp in plan.trial_plans]).astype(np.int64)
```
```python
"temporal_alignment_event": "corridor entry (trial start)",
"off_start": 0.0,
"off_end": None,
```

iii. Stated repeatedly in the trajectory ("aligned trials to corridor entry and kept corridor frames only"). Variable-length trials were adopted because the decoder reads each trial's own length; the AI checked `train_decoder.py`/`decoder.py` first ("I'm reading the decoder script so I match its expectations on variable-length trials, NaN handling, and categorical outputs instead of guessing").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: one bin = one two-photon frame. The bin size written to metadata is measured from the data — the median inter-frame interval within each session, then the median across sessions — giving 314.69 ms (reference: 1000/3.17 = 315.46 ms). Sessions differ by <1 ms in frame period, so a single global bin size is legitimate.

ii.
```python
dt = np.diff(ft) * 24.0 * 3600.0
median_dt_s = float(np.median(dt)) if dt.size else np.nan
...
median_time_bin_ms = 1000.0 * float(np.median(np.asarray(time_bin_values)))
```
```python
"time_bin_size": float(median_time_bin_ms),
```

iii. From the trajectory: "I've confirmed the imaging time base is the raw `ft` stream at about 3.18 Hz", and the AI chose "a single fixed trial window without inventing extra resampling". The imaging frame is the finest resolution available and all behavior streams are already sampled on that grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime`, the absolute (MATLAB datenum) timestamp of the sound cue on each trial, and `ft`, the timestamp of every imaging frame. (The reference instead interpolates the fractional cue frame `SoundFr` onto the frame-time axis; both give the same quantity.)

ii.
```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
...
t_frame = ft[idx]
((sound_time[tr] - t_frame) * 24.0 * 3600.0).astype(np.float32)
```

iii. The AI first checked whether any trial lacks a cue, because "a few early passive-reward sessions note 'no cue in non-reward corridor'. If some trials truly have no cue, I need a deterministic way to represent `time_to_sound_cue` without introducing NaNs", and found "sessions with missing cue 0" — both `SoundTime` and `SoundFr` are finite everywhere, so using the timestamp directly is safe and avoids the interpolation step.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue time minus frame time, converted from days to seconds (×86400), evaluated at every frame of the trial. Sign convention matches the reference: positive before the cue, negative after. Stored as `float32`, row 0 of the input matrix, named `time_to_sound_cue_s`. Range over the dataset: −38.9 to +38.7 s (reference −72.2 to +73.4 s; the narrower range follows from the 40 s trial-length cap). No NaN/inf values exist anywhere in the inputs (verified over all 37,283 trials).

ii.
```python
input_trial = np.vstack([
    ((sound_time[tr] - t_frame) * 24.0 * 3600.0).astype(np.float32),
    ...
])
```

iii. Only the implicit justification above (times are MATLAB datenums, so a ×24×3600 conversion to seconds is required); the AI verified the datenum interpretation early by computing inter-frame intervals of ~0.315 s.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at `ft[idx]`, where `idx` is exactly the frame list used for the neural columns of that trial, so the two are sample-for-sample aligned and equal in length.

ii.
```python
idx = tp.frame_idx
T = idx.size
neural_trials.append(MappedTrialArray(memmap_path, base_shape, ..., tp.row_start, tp.row_stop))
t_frame = ft[idx]
```

iii. All streams in this dataset are on the imaging-frame grid, so indexing every stream with the same frame list is sufficient; the AI confirmed that `ft`, `ft_move`, `ft_trInd`, `ft_Pos`, `ft_RunSpeed` and `ft_CorrSpc` all have identical length.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the calendar date embedded in the session key (`mouse_YYYY_MM_DD_blk`), parsed into a `datetime`, together with the set of all sessions of the same mouse.

ii.
```python
def session_date(raw_key: str) -> datetime:
    _, datestr, _ = parse_session_key(raw_key)
    return datetime.strptime(datestr, "%Y_%m_%d")
```

iii. Not explicitly justified; the date string is the only field that orders sessions, as in the reference.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The value is the number of **calendar days elapsed since that mouse's first recorded session**: `(session date − first date of this mouse).days`, as a float, broadcast across every bin of every trial of the session. Two blocks recorded on the same date get the same value. The range across the dataset is 0–92 days. The reference instead uses the ordinal index of the recording day (0–7).

ii.
```python
first_date = {subject: min(plan.date for plan in session_plans if plan.subject == subject)
              for subject in subjects}
for plan in session_plans:
    plan.day_of_training = float((plan.date - first_date[plan.subject]).days)
```
```python
np.full((T,), plan.day_of_training, dtype=np.float32),
```

iii. Not explicitly justified in the trajectory. The implicit rationale is that elapsed days is the literal reading of "day of training" and preserves the real (irregular) spacing between recording days, which an ordinal index discards. Neither definition can recover the true first day of training, since behavioural training precedes the first imaging session by about two weeks.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time`, the absolute timestamp of corridor entry on each trial, and `ft`, the frame timestamps. (The reference uses the fractional entry frame `StartFr` interpolated onto the frame-time axis.)

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
...
((t_frame - trial_start[tr]) * 24.0 * 3600.0).astype(np.float32)
```

iii. The same timestamp is used for the trial-duration quality filter (`Gray_space_time - Trial_start_time`), so the AI treated `Trial_start_time` as corridor entry and `Gray_space_time` as the exit into the grey space.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame time minus the trial's entry time, ×86400 to seconds; positive after entry, matching the reference's sign convention. Stored as `float32` in row 2, named `time_since_trial_start_s`. Because the frames are those inside the corridor, the first value of each trial is slightly positive (6e-5 to ~0.3 s, i.e. under one frame) rather than exactly 0, and the maximum is 39.97 s, consistent with the 40 s trial cap.

ii.
```python
input_trial = np.vstack([
    ...,
    ((t_frame - trial_start[tr]) * 24.0 * 3600.0).astype(np.float32),
    ...
])
```

iii. Not explicitly justified; it is the direct time difference to the alignment event declared in the metadata.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as 3-c: computed at `ft[idx]` for exactly the frames used for the neural columns of that trial.

ii.
```python
t_frame = ft[idx]
```

iii. All streams are indexed on the same imaging-frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial flag marking trials run in the rewarded corridor. Identical to the reference.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
...
np.full((T,), float(is_rew[tr]), dtype=np.float32),
```

iii. Not explicitly justified; `isRew` is the field that directly encodes the requested variable.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to `float32` (0.0/1.0) and broadcasting the per-trial value over all bins of the trial. Range over the dataset is 0–1 as expected (unsupervised and naive mice contribute only zeros).

ii.
```python
np.full((T,), float(is_rew[tr]), dtype=np.float32),
```

iii. N/A — no processing is needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial name of the wall texture. `TrialStim`/`stim_id` are deliberately not used as the label source.

ii.
```python
def stimulus_series_for_trial(beh, tr: int, T: int):
    stim_idx = canonical_stimulus_index(str(beh["WallName"][tr]))
    return np.full((T,), stim_idx, dtype=np.int16)
```

iii. The AI checked all three candidate fields and found that `TrialStim` contains the placeholder `'stimulus_of_trial'` in 64 sessions and `stim_id` is `nan` for unlisted stimuli, whereas `WallName` is complete for every trial ("`TrialStim` uniq ['circle1' 'leaf1' 'stimulus_of_trial']" vs a complete `WallName`). It therefore scored behavior copies by how unmasked they are but still read the label from `WallName`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 wall names are mapped onto **8 canonical categories** — `circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2` — by (a) keeping the numbered crop identity distinct (circle1 ≠ circle2 ≠ circle3) and (b) **aliasing the second texture family onto the first by role**: `rock1→circle1`, `rock2→circle2`, `wood1→leaf1`, `wood2→leaf2`, `wood5→leaf3`, `wood1_swap1→leaf1_swap1`, `wood1_swap2→leaf1_swap2`. An unknown wall name raises `KeyError`. The label is per-trial and broadcast over the trial's bins, stored as `int16`. The resulting class distribution is very unbalanced (32.3%, 6.0%, 0.9%, 34.6%, 16.0%, 5.0%, 2.6%, 2.7%). The reference instead uses the four physical textures `circle, leaf, rock, wood`, collapsing the crop index.

ii.
```python
STIMULUS_CATEGORIES = ["circle1","circle2","circle3","leaf1","leaf2","leaf3","leaf1_swap1","leaf1_swap2"]
WALL_TO_CANONICAL = {
    "circle1": "circle1", "rock1": "circle1",
    "circle2": "circle2", "rock2": "circle2",
    "circle3": "circle3",
    "leaf1": "leaf1",     "wood1": "leaf1",
    "leaf2": "leaf2",     "wood2": "leaf2",
    "leaf3": "leaf3",     "wood5": "leaf3",
    "leaf1_swap1": "leaf1_swap1", "wood1_swap1": "leaf1_swap1",
    "leaf1_swap2": "leaf1_swap2", "wood1_swap2": "leaf1_swap2",
}
```

iii. The final summary states: "mapped raw wall names (`rock*`, `wood*`) into canonical stimulus labels (`circle*`, `leaf*`)". The supporting evidence the AI gathered: (1) the methods say "For simplicity, we denote the stimuli as 'leaf' and 'circle', even though other visual stimuli were also used in some mice ('rock' and 'bricks')" and each mouse saw only one pair; (2) the AI dumped `UniqWalls`/`stim_id` for every session and found the rock/wood sessions carry the *same* `stim_id` role indices as the circle/leaf sessions (rock1↔0 like circle1↔0, wood1↔2 like leaf1↔2), so the aliasing follows the dataset's own role enumeration; (3) no session mixes the two families, so the aliasing never merges two textures within a session.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (the fractional imaging-frame index of each lick) together with `LickTrind` (the trial each lick belongs to).

ii.
```python
lick_frames = np.asarray(np.rint(beh["LickFr"]).astype(np.int64))
lick_trials = np.asarray(np.rint(beh["LickTrind"]).astype(np.int64))
sel = lick_trials == int(tr)
```

iii. The AI checked the lick representation directly, comparing `LickTime` against the frame timestamps to decide how to bin the fractional frame indices.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Per trial: take that trial's licks, round the fractional frame index to the **nearest** frame (`np.rint`), clip into `[0, nfr-1]`, map the absolute frame to its position within the trial's frame list, and set that bin to 1; every other bin is 0. Licks that fall outside the trial's corridor frames are discarded. Stored as `int16` with values `['no_lick','lick']`. Overall lick rate 4.35% of bins (reference 4.14%). The reference differs slightly: it truncates (`astype(int)`, i.e. floor) rather than rounds, drops licks past the last imaged frame instead of clipping them to it, and does not restrict a lick to its own trial.

ii.
```python
def lick_series_for_trial(beh, tr, frame_idx, nfr):
    lick = np.zeros((frame_idx.size,), dtype=np.int16)
    ...
    lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
    frame_to_rel = np.full((nfr,), -1, dtype=np.int32)
    frame_to_rel[frame_idx] = np.arange(frame_idx.size, dtype=np.int32)
    rel = frame_to_rel[lick_frames]
    rel = rel[rel >= 0]
    if rel.size:
        lick[np.unique(rel)] = 1
    return lick
```

iii. From the trajectory: "licks are stored at fractional frame indices, so I want to confirm whether nearest-frame assignment matches the actual lick timestamps well enough for a binary imaging-frame raster". The AI measured the timing error of nearest-frame assignment against `LickTime` (median 78 ms, max 162 ms, against a 315 ms frame), and concluded nearest-frame binning is accurate enough.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the raster is already on the neural grid; the per-trial vector is built with the trial's own `frame_idx` and therefore has exactly the neural trial's length.

ii.
```python
frame_to_rel[frame_idx] = np.arange(frame_idx.size, dtype=np.int32)
...
output_trial = np.vstack([stimulus_series_for_trial(beh, tr, T), lick, pos_bins, speed_bins]).astype(np.int16)
```

iii. All streams are indexed on the same frame list as the neural columns.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the corridor position at each imaging frame, in decimetres (0–40 across the 4 m texture).

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
...
pos_bins = position_series_for_trial(pos[idx])
```

iii. Not explicitly justified; `ft_Pos` is the per-frame position stream.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to `[0, 39.999]` decimetres and integer-divided by 10, yielding four 1-m bins as `int16`. Values are `['0_to_1m','1_to_2m','2_to_3m','3_to_4m']`. Bin occupancies are 25.1 / 24.1 / 24.9 / 25.9%, essentially identical to the reference (25.4 / 24.2 / 24.7 / 25.7%).

ii.
```python
def position_series_for_trial(pos_trial: np.ndarray):
    pos_clipped = np.clip(pos_trial, 0.0, 39.999)
    return np.floor(pos_clipped / 10.0).astype(np.int16)
```

iii. Implicit: the decoder spec asks for "4 equal-length, 1-m-long spatial bins", and the corridor is 4 m; the clip guards the endpoints (the `ft_CorrSpc` mask already restricts frames to positions below 40 dm).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-length spatial thresholds at 10, 20 and 30 decimetres (1, 2, 3 m) — not data-driven quantiles. This matches the instruction and the reference exactly.

ii.
```python
"output_values": [..., ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"], ...]
```
```python
np.floor(pos_clipped / 10.0).astype(np.int16)
```

iii. Directly prescribed by the decoder task ("discretized into 4 equal-length, 1-m-long spatial bins").

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one value per imaging frame; it is indexed with the same `idx` as the neural columns (verified against the raw behavior for session 0, trials 0/5/100).

ii.
```python
pos_bins = position_series_for_trial(pos[idx])
```

iii. All streams are indexed on the same frame list.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = np.clip(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32), 0.0, None)
```

iii. Not explicitly justified; it is the per-frame speed stream. The AI inspected its distribution across sessions (min ≈ −12 cm/s, p25 ≈ 0, max ≈ 113 cm/s) before choosing the discretisation.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Negative speeds (backward ball motion) are clipped to 0. Then, during the planning pass, the speeds of **all kept frames of all 89 sessions are pooled** and the 25/50/75% quantiles are computed once, globally: `[0.2325, 16.921, 35.476]` cm/s (stored in metadata). Every trial is discretised with those global thresholds. The reference instead computes a rank-based quartile split **per session**. In practice the global split lands almost exactly on quarters of the dataset (25.26 / 24.74 / 25.00 / 25.00%, essentially the same as the reference's 25.00 / 25.00 / 25.00 / 24.997%), because the pooled 25th percentile (0.23 cm/s) sits just above the large mass of exactly-zero frames.

ii.
```python
speed_values.append(speed[idx])            # during planning, kept frames only
...
all_speed = np.concatenate(speed_values).astype(np.float32)
speed_quantiles = np.quantile(all_speed, [0.25, 0.50, 0.75]).astype(np.float32)
```
```python
"speed_bin_quantiles_cm_per_s": speed_quantiles.astype(float).tolist(),
```

iii. Implicit from the decoder spec ("4 bins, each corresponding to 25% of the data"); the AI computed the quantiles over the frames that are actually kept, so the dropped trials do not move the boundaries. It did not comment on the global-vs-per-session choice, nor on the large zero-speed tie mass (which it had measured: p25 = 0.0 in two of three sampled sessions).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, [q25, q50, q75], right=True)`, so bin 0 is `speed ≤ 0.2325`, bin 1 is `(0.2325, 16.92]`, bin 2 is `(16.92, 35.48]`, bin 3 is `> 35.48` cm/s. Value names are the generic `['q1','q2','q3','q4']`.

ii.
```python
def speed_series_for_trial(speed_trial, speed_quantiles):
    return np.digitize(speed_trial, speed_quantiles, right=True).astype(np.int16)
```

iii. Thresholding on pooled quantiles is the direct implementation of "4 bins, each corresponding to 25% of the data" at the dataset level; `right=True` puts the stationary (clipped-to-zero) frames in the lowest bin.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame and is indexed with the same trial `idx` as the neural columns, so it has the trial's length.

ii.
```python
speed_bins = speed_series_for_trial(speed[idx], speed_quantiles)
```

iii. All streams are indexed on the same frame list.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several explicit guards:
  - The behavior streams are cut with `nfr = len(beh['ft']) - 1` during planning (the usual relation between the behavior and the imaging frame count), then re-checked against the true spike frame count at conversion time: if they differ by ≤ 2 frames the trial plan is recomputed against the spike count; if they differ by more, `RuntimeError` is raised.
  - A mismatch between the neuron count in the spike file and the length of `iarea` raises `RuntimeError`.
  - Lick frames are clipped into `[0, nfr-1]` and licks outside the trial's corridor frames are dropped.
  - Negative running speeds are clipped to 0; positions are clipped to `[0, 39.999]`.
  - Trials without enough imaged frames are dropped (1-e), and missing sound cues were checked for and found not to exist (0 sessions with non-finite `SoundTime`/`SoundFr`).
  I verified that the converted inputs contain no non-finite values in any of the 37,283 trials. The important difference from the reference is the failure mode: the reference wraps each session in `try/except` and simply skips a session that fails or has <2 trials, whereas the AI raises and aborts the whole run (this did not trigger on this dataset).

ii.
```python
spk_nfr = int(spk_chunks[0].shape[1])
if spk_nfr != plan.nframes_behavior:
    if abs(spk_nfr - plan.nframes_behavior) > 2:
        raise RuntimeError(f"Frame mismatch for {plan.raw_key}: ...")
    trial_plans, median_dt_s, _ = compute_trial_plans_from_behavior(beh, spk_nfr)
    ...
    plan.nframes_behavior = spk_nfr
    plan.trial_plans = trial_plans
```
```python
lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
speed = np.clip(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32), 0.0, None)
pos_clipped = np.clip(pos_trial, 0.0, 39.999)
```

iii. From the trajectory: "The session planning pass hit a real edge case: some behavior files have one more frame than the spike files, occasionally two. I'm refactoring the trial-plan builder so a session can be re-trimmed against the actual spike frame count at conversion time instead of failing on those off-by-one mismatches." The cue check was motivated by wanting to avoid NaNs in the inputs.

## 12-a. What are the most time-consuming steps of the code?

i. I/O dominates, as in the reference, but the AI's pipeline adds two costs the reference does not have:
  1. Reading the 405 GB of `spk/*_neural_data.npy` files (unavoidable) and **writing 108 GB** of sidecar memmaps.
  2. Reading the behavior files repeatedly: `build_behavior_index` loads all 23 files (6.6 GB) once, `build_session_plans` loads the full behavior file again for each of the 89 sessions, and the conversion loop loads it twice more per session (`load_behavior` in `assemble_dataset` and again inside `write_session_neural_memmap`) — roughly 250+ full loads of files up to 430 MB.
  The chunked transpose-and-copy into the memmap is the main CPU cost.

ii.
```python
spk_obj = np.load(SPK_DIR / f"{plan.raw_key}_neural_data.npy", allow_pickle=True).item()
...
base[c0:c1, col_start:col_start + n_chunk] = block
```
```python
def load_behavior(plan: SessionPlan):
    beh = np.load(BEH_DIR / plan.behavior_file, allow_pickle=True).item()
    return beh[plan.behavior_key]
```

iii. The AI was aware of the scale ("the full-trial conversion is likely tens of GB even in `float16`") and monitored throughput during the run ("roughly 2 to 3 sessions per 20 seconds", slower for the large mesoscope sessions); the run completed over all 89 sessions. It did not comment on the redundant behavior loads.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
  - `compute_trial_plans_from_behavior` scans the full frame index once per trial (`np.flatnonzero((ft_tr == tr) & corr)`), an O(ntrials × nframes) pass that could be replaced with a single grouping of frames by `ft_trInd` (the reference has the same pattern).
  - The nested loop in `write_session_neural_memmap` over planes × 4096-frame blocks does fancy indexing block by block; it could be done with one gather per plane at the cost of memory.
  - The per-trial Python loop in `assemble_dataset` that builds inputs/outputs one trial at a time; all four input rows and four output rows could be computed session-wide once and then sliced.
  All are negligible next to the I/O.

ii.
```python
for tr in range(ntrials):
    idx = np.flatnonzero((ft_tr == tr) & corr)
```
```python
for chunk in spk_chunks:
    for c0 in range(0, keep_idx.size, time_block):
        block = chunk[:, keep_idx[c0:c1]].T
```

iii. Not discussed by the AI; the block loop is a deliberate memory-bounded write.

## 12-c. What processing does the code repeat multiple times?

i.
  - **Behavior files are loaded up to three times per session** on top of the index pass: once in `build_session_plans`, once in `assemble_dataset` (`load_behavior`), and once inside `write_session_neural_memmap` (`load_behavior` again for the same session, in the same iteration). The reference loads each behavior file exactly once and processes all its sessions from that copy.
  - `compute_trial_plans_from_behavior` is run twice for every session whose spike frame count differs from `len(ft)-1`, redoing the whole per-trial scan.
  - Region mapping and quantile statistics are computed once, which is fine, but the per-trial frame masks computed during planning are recomputed rather than reused when the re-trim path triggers.

ii.
```python
beh = load_behavior(plan)                       # in assemble_dataset
memmap_path, base_shape = write_session_neural_memmap(plan, session_idx)
...
def write_session_neural_memmap(plan, session_idx):
    beh = load_behavior(plan)                   # loaded again, same session
```
```python
trial_plans, median_dt_s, _ = compute_trial_plans_from_behavior(beh, spk_nfr)   # second time
```

iii. Not discussed. The multi-pass design is itself justified (a planning pass is needed for the global speed quantiles and for per-mouse day-of-training), but the duplicate load inside the conversion loop is pure waste.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
  - The 108 GB of sidecars include the ~585,641 neurons with no visual-area assignment (12.5% of all cells), which the paper's curation excludes — extra write and read cost that the reference avoids.
  - `build_behavior_index` computes `uniq_walls`, `n_finite_stim_id` and `placeholder_count` for *every* behavior entry of *every* file in order to pick the "richest" behavior copy, but the only label the conversion reads from behavior is `WallName`, which is identical across the copies of a recording — so the scoring work never changes the output.
  - `TrialPlan.duration_s` and `max_gap_s` are stored on every one of the ~37k trial plans but used only inside the filter.
  - Unused imports `os` and `re`.
  - Per-trial constants (stimulus category, day of training, reward availability) are materialised as full-length vectors, though this is required by the target format and the reference does the same.

ii.
```python
candidates[raw_key].append({"file": fp.name, "key": key,
    "uniq_walls": len(np.unique(dat["WallName"])),
    "n_finite_stim_id": int(np.isfinite(np.asarray(dat["stim_id"], dtype=float)).sum()),
    "placeholder_count": int(np.sum(np.asarray(dat["TrialStim"]) == "stimulus_of_trial")),
    "has_swap_suffix": int(key.endswith("_swap1") or key.endswith("_swap2"))})
```
```python
import os
import re
```

iii. Not discussed. Keeping the unmapped neurons was a deliberate choice ("used coarse paper-style region groups: `V1`, `mHV`, `aHV`, `lHV`, `unknown`"); the candidate scoring was motivated by the masked `TrialStim`/`stim_id` fields in swap sessions, which in the end were not used as the label source.
