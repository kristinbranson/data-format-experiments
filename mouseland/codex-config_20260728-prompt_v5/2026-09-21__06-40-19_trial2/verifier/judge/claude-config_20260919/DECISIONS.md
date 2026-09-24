# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads everything from the three subfolders of `/app/data`: `beh/` (behavior), `spk/` (deconvolved traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is treated as the master index; it is a dict keyed by experiment type, each value a list of entries with `mname`, `datexp`, `blk` and sometimes `stimtype`. For every entry the AI builds a raw session id `<mname>_<datexp>_<blk>` and a behavior key (the raw id, plus `_<stimtype>` when present). The 142 index entries are collapsed into 89 unique raw recordings (`build_canonical_sessions`). Behavior files `Beh_<exp_type>.npy` are loaded lazily through an `lru_cache(maxsize=2)`; the per-session spike file `spk/<raw_id>_neural_data.npy` (a dict with a single key `spks`, a list of 3 plane arrays) and the retinotopy file `retinotopy/<mouse>_<date>_trans.npz` are read once per session inside `convert_session`. The dataset is traversed three times: a stimulus-vocabulary pass and a running-speed-quartile pass over behavior only, then the main conversion pass that loads the neural data.

ii.
```python
def load_experiment_index() -> dict[str, list[dict[str, Any]]]:
    return np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()

@lru_cache(maxsize=2)
def load_behavior_file(exp_type: str) -> dict[str, dict[str, Any]]:
    return np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
@property
def spk_path(self) -> Path:
    return SPK_ROOT / f"{self.raw_id}_neural_data.npy"

@property
def retino_path(self) -> Path:
    return RETINO_ROOT / f"{self.subject}_{self.dateexp}_trans.npz"
```
```python
beh = get_behavior(session)
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
retino = np.load(session.retino_path, allow_pickle=True)
```

iii. From CONVERSION_NOTES Step 1/Step 5: the loading path is copied from the reference `utils.load_exp_beh`, `utils.load_spk` and `utils.load_retino`. The AI notes that `load_spk` concatenates `['spks']` across planes and that no dF/F is computed anywhere in the reference code, so the released `spks` are the analysis signal. The behavior-only first pass was added so that the global speed quartiles and stimulus vocabulary can be computed without touching the 405 GB of spike files.

## 1-b. How are the data split into subjects?

i. The subject is the `mname` field of the index entry, stored on every `SessionInfo`. At assembly the subject list is the sorted set of unique names and `subject_idx` is each session's index into it. This yields 19 mice, matching the paper.

ii.
```python
rec = per_raw.setdefault(raw_id, {"subject": entry["mname"], ...})
```
```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.subject])
```

iii. CONVERSION_NOTES Step 2/Step 5: the index already names the mouse for every recording, and the AI verified that the unique `mname` count is 19, matching "We performed 89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is one raw recording = one mouse + one date + one block (`<mname>_<datexp>_<blk>`), which is also the name of the spike file. The 142 experiment-index entries and 99 behavior keys are de-duplicated onto 89 canonical raw sessions with a `per_raw` dict; alias behavior keys (including `stimtype` variants such as `_swap1`/`_swap2`) are recorded in `aliases`/`exp_types` but do not create extra sessions. One behavior key is selected as the canonical source for each raw session. Sessions with fewer than 2 converted trials would be dropped (none were).

ii.
```python
def canonical_raw_id(entry): return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"

def behavior_key_for_entry(entry):
    raw_id = canonical_raw_id(entry)
    if "stimtype" in entry:
        return f"{raw_id}_{entry['stimtype']}"
    return raw_id
```
```python
rec = per_raw.setdefault(raw_id, {...,"behavior_exp_type": exp_type, "behavior_key": beh_key, ...})
rec["aliases"].add(beh_key)
rec["exp_types"].add(exp_type)

# Keep the first-seen canonical behavior source; Step 4 verified that
# alias keys for the same raw session are duplicates on core arrays.
if rec["behavior_key"] not in beh_file and beh_key in beh_file:
    rec["behavior_exp_type"] = exp_type
    rec["behavior_key"] = beh_key
```

iii. CONVERSION_NOTES Step 4: the AI explicitly investigated the 142 / 99 / 89 mismatch, compared alias entries field by field, and found that repeated behavior keys are "byte-for-byte identical on the core behavioral arrays". It therefore concluded the dataset is 89 real recordings (matching the paper) and that aliases are relabels, not extra data.

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` trials the behavior declares, indexed by the per-frame label `ft_trInd`. The frames kept for a trial are those that are simultaneously (a) labelled with that trial, (b) inside the texture corridor (`ft_CorrSpc`), and (c) **moving** (`ft_move > 0`). The third condition is an addition relative to the human reference: it discards every frame in which the VR did not advance, i.e. every frame in which the mouse was below the 6 cm/s running threshold. Across the dataset this keeps 821,579 frames; the corresponding corridor-only frame count is roughly 1.2 M, so about a third of in-corridor frames are dropped, including frames in the middle of a traversal. Trials are variable length (mean T = 22.3 bins, min 11, max 178) and are not padded. The grouping is done in one vectorized pass with `np.diff`/`np.split` on the masked trial-id vector rather than one scan per trial. Non-finite `ft_trInd` entries are handled explicitly.

ii.
```python
def trial_frame_groups(beh: dict[str, Any]) -> list[np.ndarray]:
    ft_tr = np.asarray(beh["ft_trInd"])
    ft_corr = np.asarray(beh["ft_CorrSpc"], dtype=bool)
    ft_move = np.asarray(beh["ft_move"]) > 0

    finite = np.isfinite(ft_tr)
    tr_int = np.full(ft_tr.shape, -1, dtype=np.int32)
    tr_int[finite] = ft_tr[finite].astype(np.int32)

    keep = finite & ft_corr & ft_move
    frame_idx = np.flatnonzero(keep)
    trial_ids = tr_int[keep]
    ntrials = int(beh["ntrials"])
    groups = [np.empty(0, dtype=np.int32) for _ in range(ntrials)]
    ...
    changes = np.flatnonzero(np.diff(trial_ids)) + 1
    split_frames = np.split(frame_idx, changes)
    split_trials = np.split(trial_ids, changes)
    for frames, tids in zip(split_frames, split_trials):
        groups[int(tids[0])] = frames.astype(np.int32, copy=False)
    return groups
```

iii. CONVERSION_NOTES Step 1/4/5: the AI found that the reference analysis functions (`Get_dprime_selective_neuron`, `Get_coding_direction`) restrict to `ft_CorrSpc & (ft_move > 0)`, and that the paper says "We only considered timepoints during running for analysis". Its stated rationale is that this "matches the paper's running-only corridor analysis while preserving native imaging-frame sampling", and that "time inputs will use actual timestamps so skipped stationary periods remain explicit".

## 1-e. How are trials filtered based on quality controls?

i. Almost no trial-level quality control. A trial is dropped only if it retains zero corridor-running frames, and a session is dropped only if fewer than 2 trials survive. In practice **no trial and no session was dropped**: all 38,110 raw trials of the 89 sessions are in the output (the human reference keeps 37,728). The AI explicitly measured the trial-duration distribution (median 7.1 s, 99th pct 74 s, max 1765 s) and identified the same pathological trial the reference flags (`TX88_2022_07_19_1`, trial 391: a few running frames spread over 1765 s of wall clock), but decided not to filter it because the timestamps are faithful to the source. The consequence is visible in the input ranges: `time_since_trial_start` spans [0, 1765.2] s and `time_to_sound_cue` spans [-1763.3, 723.5] s, versus [0, 74.8] and [-72.2, 73.4] for the reference.

ii.
```python
for trial, frames in enumerate(groups):
    if frames.size == 0:
        continue
```
```python
if len(neural_trials) < 2:
    print(f"[convert] skipping {session.raw_id}: only {len(neural_trials)} converted trial(s)")
    continue
```

iii. CONVERSION_NOTES Step 5 decision 9 and Step 10 check 5: "Drop trials with zero retained corridor-running frames; drop sessions with fewer than 2 remaining trials. Rationale: satisfies the decoder requirement and excludes unusable raw trials." On the 1765 s outlier: "This indicates sparse movement bouts within a single raw trial, not a conversion indexing bug. Because the timestamps and retained-frame masks are faithful to the source data, no corrective filtering was applied."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<raw_id>_neural_data.npy` — a list of three plane arrays of shape (n_neurons_plane, n_frames) — and from `iarea` in `retinotopy/<mouse>_<date>_trans.npz` for the area label of each row. The AI does not pre-concatenate the planes; it slices each plane by the trial's frame indices and concatenates the (much smaller) per-trial results. It asserts that the total plane row count equals the retinotopy row count.

ii.
```python
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
retino = np.load(session.retino_path, allow_pickle=True)
iarea = np.asarray(retino["iarea"], dtype=np.float32)
brain_region_idx = map_iarea_to_region_idx(iarea)
total_rows = sum(part.shape[0] for part in spk_parts)
if total_rows != brain_region_idx.shape[0]:
    raise ValueError(...)
```
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 1/4: "The reference code does not compute dF/F ... It loads `['spks']` directly", and the paper states "All our analyses were based on deconvolved fluorescence traces." The per-plane slicing instead of whole-session concatenation is documented in Step 6/7 as a speed-up that "avoids expensive whole-session concatenation and was benchmarked to be much faster".

## 2-b. How is the `neural` data processed?

i. No processing at all: the deconvolved traces are taken as they are, sliced to the trial's retained frames, and stored as **float32**. No dF/F, no normalization, no smoothing, no padding. Trials keep their own length.

ii.
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
...
session_neural.append(neural_trial)
```

iii. CONVERSION_NOTES Step 5 decision 2: "Use released deconvolved `spks` directly. Rationale: both the paper and reference code analyze deconvolved traces and never recompute dF/F." No rationale is given for float32 over a smaller dtype; the resulting pickle is 161.6 GiB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neurons are filtered.** Every row of `spks` is kept. The retinotopy label is mapped to six region names: the four reference areas V1 / mHV / lHV / aHV plus two extra labels, `unassigned_7` for `iarea == 7` and `outside_visual` for `iarea == -1`. All 4,691,034 rows are written, of which 585,641 (188,331 with `iarea == 7` and 397,310 with `iarea == -1`) lie outside the four visual areas that the reference retains (the human reference keeps 4,105,393). An unexpected `iarea` code raises.

ii.
```python
def map_iarea_to_region_idx(iarea: np.ndarray) -> np.ndarray:
    out = np.empty(iarea.shape[0], dtype=np.int16); out.fill(-1)
    out[iarea == 8] = 0  # V1
    out[np.isin(iarea, [0, 1, 2, 9])] = 1  # mHV
    out[np.isin(iarea, [5, 6])] = 2  # lHV
    out[np.isin(iarea, [3, 4])] = 3  # aHV
    out[iarea == 7] = 4  # unassigned_7
    out[iarea == -1] = 5  # outside_visual
    if np.any(out < 0):
        raise ValueError(...)
    return out
```
```python
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV", "unassigned_7", "outside_visual"]
```

iii. CONVERSION_NOTES Step 1 and Step 5 decision 6: the AI records that "some analyses exclude neurons outside visual cortex via retinotopy labels (`iarea == -1` or `7`)" but chooses to "keep all neural rows and provide grouped region labels plus explicit labels for rows outside the four main groups ... this preserves paper-level neuron counts and source information while still matching the region grouping used in the analysis code." Its supporting sanity check is that the retained per-session row counts reproduce the paper's exact range, min 20,547 and max 89,577. It also argues (Step 1/3) that curation is delegated to Suite2p and that the reference applies no blanket cell-quality filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry / trial start. Each trial array begins at the first retained corridor frame of that traversal and ends at the last, so trials are variable length, nothing is cut to a common window and nothing is padded. Metadata records `temporal_alignment_event = "corridor entry / trial start"`, `off_start = 0.0`, `off_end = None`. Because the `ft_move > 0` mask is applied before slicing, the retained bins of a trial are chronologically ordered but not necessarily consecutive imaging frames.

ii.
```python
"temporal_alignment_event": "corridor entry / trial start",
"off_start": 0.0,
"off_end": None,
"frame_selection": "native imaging frames within corridor while running (ft_CorrSpc & ft_move > 0)",
```
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0)  # frames from trial_frame_groups
```

iii. CONVERSION_NOTES Step 4/5 decision 4: "Align every trial to corridor entry / `Trial_start_time`. Rationale: this is required by the decoder task; cue timing, position, licking, and reward context can all be reconstructed relative to that same event from native source variables." All streams are indexed with the same `frames` vector, so alignment between streams is exact by construction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. The bin is the native imaging frame at 3.17 Hz, i.e. 315.457 ms, recorded in `metadata['time_bin_size']`. Every stream (neural, inputs, outputs) already lives on the imaging-frame grid, so nothing is interpolated. Mean trial length is 22.3 bins. Note that because non-running frames are removed, successive bins within a trial are each 315.457 ms long but are not always 315.457 ms apart.

ii.
```python
FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": float(TIME_BIN_MS),
"source_frame_rate_hz": float(FRAME_RATE_HZ),
```

iii. CONVERSION_NOTES Step 1/3: the reference notebook states the calcium frame rate is 3.17 Hz, and the paper says running speed "was interpolated to the timepoints of the imaging frames". Step 10 check 3(d): "neural data remains at native imaging frames, consistent with the source signal used throughout the reference."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From the per-trial `SoundTime` (MATLAB datenum of the cue) and `ft` (the datenum timestamp of every imaging frame). The AI uses the pre-computed cue *time* rather than interpolating the fractional cue *frame* `SoundFr`; I verified on `TX108_2023_01_05_2` that `SoundTime` equals `np.interp(SoundFr, arange(n), ft)` to within 1e-14 s, so the two routes are numerically identical.

ii.
```python
ft = np.asarray(beh["ft"], dtype=np.float64)
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
...
frame_times = ft[frames]
time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping table: source `SoundTime`, `ft`; transform `(SoundTime[trial] - ft_frame) * 86400`; "Positive before cue, negative after cue. Continuous, time-varying." Referenced against the reference's cue-aligned analyses (`spk_2_cue`) and the methods statement that the cue occurs at a uniformly random position between 0.5 m and 3.5 m.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. A single subtraction and a unit conversion: the frame timestamps of the trial are subtracted from the trial's cue timestamp and the difference, in days, is multiplied by 86,400 to give seconds. Sign convention: positive before the cue, negative after, matching the variable name "time **to** sound cue". Stored as float32, one value per retained bin. No clipping or bounding is applied, so trials in which the animal stalls produce values as extreme as -1763 s / +723 s.

ii.
```python
time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
input_trial = np.stack([time_to_cue, day_of_training, time_since_start, reward_availability], axis=0)
```

iii. Step 5: the raw times are MATLAB datenums (units of days), so the ×86400 conversion is required to report seconds. The sign convention is chosen to match the input name given in the Decoder Task specification.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on exactly the same `frames` index vector used to slice the neural columns of that trial, so it is aligned bin-for-bin by construction and has the same length as the trial's neural array.

ii.
```python
for trial, frames in enumerate(groups):
    neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0)
    frame_times = ft[frames]
    time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 10 check 2 verifies with `np.allclose()` on six hand-picked trials that `time_to_sound_cue_s == (SoundTime - ft) * 86400` on the retained frames. All streams share the frame index, which is the dataset's common clock.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the recording date `datexp` in the experiment index (parsed to a `date` object) together with the mouse name `mname`, i.e. the same information the reference uses, but consumed as a calendar date rather than as an ordering key.

ii.
```python
@property
def date_obj(self):
    return datetime.strptime(self.dateexp, "%Y_%m_%d").date()
```
```python
def compute_day_offsets(sessions):
    subject_first_date = {}
    for session in sessions:
        subject_first_date[session.subject] = min(
            subject_first_date.get(session.subject, session.date_obj), session.date_obj)
    return {session.raw_id: float((session.date_obj - subject_first_date[session.subject]).days)
            for session in sessions}
```

iii. CONVERSION_NOTES Step 5 mapping: "Chosen because `sess#` / `days` fields are inconsistent across experiment branches and duplicate aliases", i.e. the AI checked for an explicit training-day field in the source, found it unreliable, and fell back on the recording date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse the earliest recording date is found, and each session's value is the number of **calendar days** elapsed since that date. The value is a per-session scalar broadcast to all bins of every trial of that session, stored as float32. The resulting range over the dataset is 0–92 days (the reference, which counts recorded sessions rather than calendar days, gets 0–7).

ii.
```python
day_value = np.float32(day_offsets[session.raw_id])
...
day_of_training = np.full(frames.size, day_value, dtype=np.float32)
```

iii. Step 5 decision 7: "Use subject-specific calendar-day offset from first recording, repeated across each trial's timepoints. Rationale: it is continuous, defined for every session, and avoids inconsistencies between `sess#` and `days` metadata fields." The offsets are computed over whichever session list is being converted, so in `--sample` mode the baseline is the first of the two sampled sessions rather than the mouse's true first recording.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the per-trial `Trial_start_time` (MATLAB datenum of corridor entry) and `ft`. Again the AI uses the pre-computed time rather than interpolating the fractional entry frame `StartFr`; I verified on `TX108_2023_01_05_2` that `Trial_start_time` equals `np.interp(StartFr, arange(n), ft)` to within 3e-14 s, so this is numerically the same quantity the reference computes.

ii.
```python
trial_start_time = np.asarray(beh["Trial_start_time"], dtype=np.float64)
...
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. Step 5 mapping: source `Trial_start_time`, `ft`; "Trial start is corridor entry, as required by the user." The AI confirmed in Step 4 that corridor entry is the required alignment event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Subtraction plus the datenum→seconds conversion, with the opposite sign to the cue input: bin time minus trial-start time, so the value is ~0 at the first retained bin and grows through the traversal. float32, one value per bin. Values range [0, 1765.2] s because stalled trials are not filtered.

ii.
```python
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. Same rationale as 3-b: the source times are in days, and the sign follows the input name "time **since** trial start".

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from `ft[frames]` with the identical `frames` vector used for the neural slice of the trial, therefore bin-aligned and of identical length.

ii.
```python
frame_times = ft[frames]
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. Step 10 check 2 verifies `time_since_trial_start_s == (ft - Trial_start_time) * 86400` against raw loads with `np.allclose()` on six trials across two sessions; all passed. The imaging frame index is the dataset's shared time base.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial boolean `isRew`, which marks trials run in the rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
...
reward_availability = np.full(frames.size, float(is_rew[trial]), dtype=np.float32)
```

iii. Step 5 mapping: "Uses the native rewarded-corridor identity even in unsupervised sessions where water reward is absent." This matches the specification "1 if in rewarded corridor, 0 if not".

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float and a broadcast of the trial-constant value across the trial's bins. Observed range [0, 1]; whole sessions (the naive/unsupervised cohorts) are all zeros, which is expected.

ii.
```python
reward_availability = np.full(frames.size, float(is_rew[trial]), dtype=np.float32)
```

iii. No further rationale needed or given; the raw field is already the requested binary indicator. The AI's sanity checks (Step 10 check 2) compare converted values against raw `isRew`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From the per-trial `WallName` string. A global vocabulary is first collected over every session with `gather_stimulus_vocabulary`, sorted alphabetically, and used as the class index. The AI explicitly rejected `TrialStim`/`stim_id` in favour of `WallName` after finding the former masked/inconsistent in swap sessions.

ii.
```python
def gather_stimulus_vocabulary(sessions):
    vocab = set()
    for session in sessions:
        beh = get_behavior(session)
        vocab.update(map(str, np.unique(np.asarray(beh["WallName"]))))
    return sorted(vocab)
```
```python
wall_name = np.asarray(beh["WallName"])
stimulus_idx = stim_to_idx[str(wall_name[trial])]
```

iii. Step 2/4: the AI enumerated the full global vocabulary (15 names) and resolved the paper's "brick" wording against the files' `wood*` labels, concluding it should "preserve the raw names in converted outputs to avoid ambiguity".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. **The 15 raw wall names are kept as 15 separate classes** — `circle1`, `circle2`, `circle3`, `leaf1`, `leaf1_swap1`, `leaf1_swap2`, `leaf2`, `leaf3`, `rock1`, `rock2`, `wood1`, `wood1_swap1`, `wood1_swap2`, `wood2`, `wood5` — rather than being collapsed onto the four base textures (circle / leaf / rock / wood). The class index is the alphabetical position in the global vocabulary, broadcast across all bins of the trial as int16, and `output_values[0]` is the vocabulary itself. Most sessions contain only 2–5 of the 15 classes, so `output_range[0]` varies per session and the global distribution is heavily skewed (circle1 0.252, leaf1 0.264, … wood1_swap1 0.008). Uniform chance for this output is 1/15 = 0.067 and the achieved validation balanced accuracy is 0.545 (reference: 4 classes, chance 0.25, 0.664).

ii.
```python
stimulus_out = np.full(frames.size, stimulus_idx, dtype=np.int16)
output_trial = np.stack([stimulus_out, lick_out, pos_out, speed_out], axis=0)
```
```python
"output_values": [stim_vocab, ["no_lick", "lick"], ["0-1m", "1-2m", "2-3m", "3-4m"], ["q1", "q2", "q3", "q4"]],
```

iii. Step 5 decision 5: "Preserve raw stimulus names, including `wood*` and swap variants. Rationale: this avoids ambiguous relabeling and keeps the conversion faithful to the released files even where paper prose used different generic names." Step 4 also notes the paper pools swap types for statistics but the AI chose to "keep the actual per-trial stimulus labels present in the canonical raw session".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame number of every lick in the session.

ii.
```python
lick_binary = build_lick_binary(beh, nframes=ft.shape[0])
```

iii. Step 5 mapping: "Uses the same frame-index convention as the reference code (`astype(int)`-style frame assignment)", referencing the reference's `spk_2_firstLick` / `spk_2_cue`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is built once: the fractional lick frame is truncated to int, non-finite entries and indices outside `[0, nframes)` are discarded, and the remaining frames are set to 1. A bin is therefore 1 if at least one lick falls in it, 0 otherwise. Note `nframes` is taken from the length of the behavior `ft` array, not from the number of imaged frames. Global distribution: 0.963 no-lick / 0.037 lick, essentially identical to the reference's 0.959 / 0.041.

ii.
```python
def build_lick_binary(beh, nframes):
    lick_binary = np.zeros(nframes, dtype=np.int8)
    lick_frames = np.asarray(beh["LickFr"])
    if lick_frames.size == 0:
        return lick_binary
    finite = np.isfinite(lick_frames)
    lick_idx = lick_frames[finite].astype(np.int64, copy=False)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
    if lick_idx.size:
        lick_binary[np.unique(lick_idx)] = 1
    return lick_binary
```

iii. Step 5: licking must be "binary per retained frame: 1 if one or more licks map to that imaging frame, else 0", as required by the Decoder Task ("Licking, binary, time-varying"). The bounds check is the AI's defensive handling of licks recorded outside the imaging window.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` already indexes imaging frames, so the binary vector is on the neural grid; the trial's values are read with the same `frames` vector as the neural slice.

ii.
```python
lick_out = lick_binary[frames].astype(np.int16, copy=False)
```

iii. Step 10 check 2 compares the converted lick channel against a fresh `build_lick_binary(LickFr)` on raw loads for six trials, all matching. The shared frame index guarantees alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the corridor position at each imaging frame, in native units the AI established to be decimeters (0–40 across the 4 m texture, 40–60 through the 2 m grey space).

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=np.float32)
...
pos_out = position_to_bin(ft_pos[frames])
```

iii. Step 4: the AI reconciled `Corridor_Length = 60`, `Texture_Length = 40`, `Gray_Space_length = 20` with the paper's "4 m long, with 2 m of grey space" and the notebook's 60-bin/1-decimeter comment, concluding "Interpret the stored position units as decimeters."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to `[0, 39.999999]`, divided by 10 and floored, giving an integer 0–3, then clipped again to 0–3 for safety. Only corridor frames (`ft_CorrSpc`) are retained, so the clip is effectively inert. Stored as int16, one value per bin.

ii.
```python
def position_to_bin(pos: np.ndarray) -> np.ndarray:
    bins = np.floor(np.clip(pos, 0.0, 39.999999) / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. Step 5 mapping: "Discretize corridor position into 4 bins of 10 native position units each (0–10, 10–20, 20–30, 30–40), corresponding to 0–1 m, 1–2 m, 2–3 m, 3–4 m" — the literal reading of the Decoder Task requirement for "4 equal-length, 1-m-long spatial bins".

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-width spatial thresholds at 1 m, 2 m and 3 m (10, 20, 30 decimeters), identical for every trial, session and mouse; labels `['0-1m','1-2m','2-3m','3-4m']`. The realized distribution is near-uniform (0.250 / 0.249 / 0.250 / 0.252 globally, and within a few percent of uniform in every session).

ii.
```python
"output_values": [..., ["0-1m", "1-2m", "2-3m", "3-4m"], ...]
```
```python
bins = np.floor(np.clip(pos, 0.0, 39.999999) / 10.0).astype(np.int16)
```

iii. The Decoder Task prescribes equal-length 1 m bins, so no data-driven thresholding is used. The AI's `--show-processing` plot of position samples before/after the running mask was its check that the full 0–4 m range is still covered after masking.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one sample per imaging frame; the trial's values are read with the same `frames` vector as the neural slice, so it is bin-aligned and of equal length.

ii.
```python
pos_out = position_to_bin(ft_pos[frames])
output_trial = np.stack([stimulus_out, lick_out, pos_out, speed_out], axis=0)
```

iii. Step 10 check 2 verified `position_bin == floor(ft_Pos / 10)` clipped to the 4 corridor bins against raw loads on six trials. All streams share the frame index.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the mouse at each imaging frame.

ii.
```python
ft_run_speed = np.asarray(beh["ft_RunSpeed"], dtype=np.float32)
...
speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)
```

iii. Step 5 mapping: cites the paper's statement that "the running speed was interpolated to the timepoints of the imaging frames", so the per-frame speed is already on the neural grid.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behavior-only pre-pass (`gather_speed_edges`) accumulates `ft_RunSpeed` over the retained corridor-running frames of **every session in the run** (821,579 frames), drops non-finite values, and takes the 25/50/75% quantiles of that pooled distribution: edges `[12.42, 25.35, 40.85]`. A tie-breaking loop nudges duplicate edges apart with `np.nextafter`. These three **global** edges are then applied to every session. Because the `ft_move > 0` mask already removes stationary frames, the pile-up at exactly zero speed that would otherwise break quantile splitting is largely absent.

ii.
```python
def stabilized_quantile_edges(values):
    edges = np.quantile(values, [0.25, 0.50, 0.75]).astype(np.float32)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf, dtype=np.float32)
    return edges
```
```python
def gather_speed_edges(sessions):
    speeds = []
    for session in sessions:
        beh = get_behavior(session); groups = trial_frame_groups(beh)
        ft_run_speed = np.asarray(beh["ft_RunSpeed"], dtype=np.float32)
        for frames in groups:
            vals = ft_run_speed[frames]; vals = vals[np.isfinite(vals)]
            if vals.size: speeds.append(vals)
    return stabilized_quantile_edges(np.concatenate(speeds))
```

iii. Step 5 mapping: "Compute global quartile thresholds over all retained samples, then discretize each retained frame into 4 equal-frequency bins ... Quartiles are computed from the canonical full dataset after trial masking, then applied consistently to every session." This is the literal reading of "4 bins, each corresponding to 25% of the data".

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, edges, right=False)` against the three global edges, giving classes 0–3 labelled `['q1','q2','q3','q4']`. Globally the split is exact (0.250 / 0.250 / 0.250 / 0.250), but because the edges are global rather than per-session the **per-session** distributions are extremely unbalanced: the printed per-session fractions include sessions at (0.956, 0.042, 0.002, 0.000) and (0.039, 0.123, 0.213, 0.625). Roughly a dozen sessions have a near-empty top or bottom bin.

ii.
```python
def speed_to_bin(speed: np.ndarray, speed_edges: np.ndarray) -> np.ndarray:
    return np.digitize(speed, speed_edges, right=False).astype(np.int16)
```
```python
"speed_bin_edges": [float(x) for x in speed_edges.tolist()],
```

iii. Step 5 / Step 9: the AI's stated criterion is that each bin should hold 25% of the data, which it verifies globally in the Step 9 consistency table ("[q1=0.250, q2=0.250, q3=0.250, q4=0.250]"). It records the edges in the metadata for reproducibility. It does not discuss the per-session imbalance this creates.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one sample per imaging frame and the trial's values are taken with the same `frames` vector as the neural slice, so it is bin-aligned and of equal length.

ii.
```python
speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)
output_trial = np.stack([stimulus_out, lick_out, pos_out, speed_out], axis=0)
```

iii. Step 10 check 2 verified the speed channel against quartile discretization of raw `ft_RunSpeed` on six trials. All streams share the frame index.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures are present, and one gap:
- Non-finite `ft_trInd` frames (63–274 per session) are excluded from every trial group.
- Non-finite `LickFr` values, and lick frames outside `[0, nframes)`, are discarded; an empty `LickFr` returns an all-zero vector.
- Non-finite `ft_RunSpeed` values are excluded from the quartile estimation.
- Duplicate quantile edges are nudged apart so `digitize` cannot collapse bins.
- Position is clipped into `[0, 40)` before binning.
- Neural row count vs retinotopy row count is asserted, and unmapped `iarea` codes raise.
- Trials with zero retained frames are skipped; sessions with <2 trials are skipped.
- **Gap**: the behavior arrays are *not* truncated to the number of imaged frames. `nframes` is taken from `len(ft)`, and the reference truncates every stream with `beh[...][:nfr]` where `nfr = spks.shape[1]`. I confirmed from the file headers that the behavior grid is 0–3 frames *longer* than the imaging grid in roughly half the sessions (e.g. `TX124_2023_12_24_1`: 18,492 behavior frames vs 18,491 imaged). If one of those trailing frames ever fell inside a retained corridor-running window, `part[:, frames]` would raise `IndexError` and lose the whole session. It never did in this dataset — all 89 sessions and all 38,110 trials converted — but the code depends on that coincidence rather than guarding against it.

ii.
```python
finite = np.isfinite(ft_tr)
tr_int = np.full(ft_tr.shape, -1, dtype=np.int32)
tr_int[finite] = ft_tr[finite].astype(np.int32)
keep = finite & ft_corr & ft_move
```
```python
lick_idx = lick_frames[finite].astype(np.int64, copy=False)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
```
```python
ft = np.asarray(beh["ft"], dtype=np.float64)
lick_binary = build_lick_binary(beh, nframes=ft.shape[0])   # behavior length, not spks.shape[1]
```

iii. CONVERSION_NOTES Step 10 check 5 ("Check for edge cases") reports only the long-trial investigation; the AI concluded the source data are clean and that the surviving anomalies are faithful to the source. It did not report the behavior/imaging frame-count mismatch, and the notes contain no statement about truncating to imaged frames.

## 12-a. What are the most time-consuming steps of the code?

i. By far the dominant cost is reading the 405 GB of per-session spike files (`np.load` of `spk/<id>_neural_data.npy`), which is what the logged per-session times of 2–22 s are made of; total conversion was 1,144.6 s for 89 sessions. The second cost is writing the output: pickling the 161.6 GiB dictionary took a further ~177 s. The two behavior-only pre-passes are cheap (the speed scan is 4.7 s over all 89 sessions). The AI instruments this: `convert_session` records `conversion_seconds` per session, printed after each one, and the total is stored in metadata.

ii.
```python
t0 = time.time()
beh = get_behavior(session)
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
...
"conversion_seconds": time.time() - t0,
```
```python
print(f"[convert] kept {len(neural_trials)}/{meta['n_trials_original']} trials, "
      f"rows={meta['n_rows_neural']}, elapsed={meta['conversion_seconds']:.2f}s", flush=True)
```

iii. Step 6/7: the AI benchmarked whole-session concatenation against per-plane per-trial slicing and kept the latter; it estimated 17–20 min for the full run and noted this was "borderline above the 15 min target; no clearly faster safe hot-path alternative was found in a direct benchmark". The realized 19 min matched the estimate.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loops are inherent to the ragged output format (one Python iteration per trial to build the neural/input/output arrays). Two loops are genuinely redundant:
- `gather_speed_edges` iterates over trial groups and appends one array per trial, when a single boolean mask over the session's `ft_RunSpeed` would give the same pooled values in one indexing operation (it builds ~38,000 small arrays).
- `gather_stimulus_vocabulary` walks every session's behavior dict in a separate pass purely to collect 15 strings.

Notably, the per-trial frame search that the human reference identifies as its own vectorizable loop is already vectorized here: `trial_frame_groups` masks once and uses `np.diff`/`np.split` instead of scanning the frame index once per trial.

ii.
```python
for frames in groups:                      # could be ft_run_speed[keep_mask] in one step
    if frames.size == 0: continue
    vals = ft_run_speed[frames]
    vals = vals[np.isfinite(vals)]
    if vals.size: speeds.append(vals)
```
```python
changes = np.flatnonzero(np.diff(trial_ids)) + 1          # one pass instead of one scan per trial
split_frames = np.split(frame_idx, changes)
```

iii. Step 6: "Code inefficiencies identified: Full-session neural concatenation would duplicate memory unnecessarily. Re-reading behavior files many times could add overhead if not cached." The AI's stated speed-ups were per-trial slicing, an LRU cache on behavior files, and the behavior-only quartile pass. It did not flag the remaining loops, correctly judging that everything is dwarfed by spike-file I/O.

## 12-c. What processing does the code repeat multiple times?

i. Three repetitions:
- `trial_frame_groups` is recomputed for every session at least twice — once in `gather_speed_edges` and once in `convert_session` — and a third time inside `plot_processing_summary` when `--show-processing` is on.
- Behavior files are loaded through `lru_cache(maxsize=2)`, so with 23 experiment types and three separate passes (vocabulary, speed, conversion) the same `Beh_*.npy` file is unpickled repeatedly whenever the session order interleaves experiment types. The human reference instead groups sessions by behavior file and reads each exactly once.
- `get_behavior(session)` is called again in `build_dataset` just to pass `beh` to the plotting routine.
- Minor: `np.unique(lick_idx)` before an assignment that is idempotent anyway.

ii.
```python
@lru_cache(maxsize=2)
def load_behavior_file(exp_type: str) -> dict[str, dict[str, Any]]:
    return np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
speed_edges = gather_speed_edges(sessions)   # calls trial_frame_groups(beh) for every session
...
groups = trial_frame_groups(beh)             # recomputed in convert_session
```

iii. The AI's justification for the extra passes (Step 5/6) is correctness rather than speed: the stimulus vocabulary and speed quartile edges must be global and identical across sessions, which requires seeing all behavior before converting any session. It documented the LRU cache as a speed-up; it did not note that a cache of size 2 does not prevent repeated reads.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little is computed and then thrown away — every array the script builds is written to the pickle. What the code does produce unnecessarily is **volume**: 585,641 neurons with `iarea` of 7 or −1, which the reference code's analyses exclude, are carried all the way into the output, and all traces are stored as float32 rather than a half-precision type. Together these make the artifact 161.6 GiB, about 16× the reference's, without adding information the decoder task needs. Smaller items: the `session_meta` bookkeeping (`aliases`, `exp_types`, `reward_mode`, `stimuli_present`) is stored in metadata but never used; the `stabilized_quantile_edges` tie-breaking loop is a no-op on this data (the three edges are well separated); `np.unique` on the lick indices is redundant; and the `--show-processing` path recomputes the trial grouping and reloads behavior for the plots.

ii.
```python
out[iarea == 7] = 4  # unassigned_7
out[iarea == -1] = 5  # outside_visual
```
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
```
```python
for i in range(1, edges.size):
    if edges[i] <= edges[i - 1]:
        edges[i] = np.nextafter(edges[i - 1], np.inf, dtype=np.float32)
```

iii. Step 5 decision 6: keeping all rows is deliberate — "this preserves paper-level neuron counts and source information while still matching the region grouping used in the analysis code". The AI treats the reproduction of the paper's exact 20,547–89,577 per-recording range as a sanity check in Step 9/10. No rationale is offered anywhere for float32 over float16, and the notes do not discuss the size of the resulting file beyond reporting it.
