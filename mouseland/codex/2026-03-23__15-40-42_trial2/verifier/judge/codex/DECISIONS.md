# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `data/beh/Imaging_Exp_info.npy`, builds a canonical session catalog, then loads behavior per session from `Beh_<exp_type>.npy`, spikes from `data/spk` via `utils.load_spk`, and retinotopy from `data/retinotopy/*.npz`. It also rereads behavior files later to collect the stimulus vocabulary and estimate the nominal frame interval.

ii. ```python
def build_session_catalog() -> list[SessionCandidate]:
    exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
    ...
```

```python
def load_behavior(exp_type: str, beh_key: str) -> dict:
    beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    return beh[beh_key]
```

```python
beh = load_behavior(candidate.exp_type, candidate.beh_key)
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI says it should use the imaging index as the canonical source, deduplicate to unique neural recordings, and reuse the reference loaders from `code/utils.py`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by `db["mname"]`, and the final `subjects` list is the sorted unique set of mouse names. `subject_idx` is built from that list.

ii. ```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
...
"subject_idx": np.asarray(
    [subject_to_idx[s.subject] for s in processed_sessions], dtype=np.int64
),
```

iii. The notes explicitly say to use `mname` as the subject ID because `Imaging_Exp_info.npy` already identifies each mouse.

## 1-c. How are the data split into sessions?

i. A session is defined by `<mname>_<datexp>_<blk>`. Duplicate entries in the master index are grouped by this base session ID, then one "canonical" entry is chosen, preferring non-`stimtype` entries.

ii. ```python
session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
...
grouped[session_id].append(SessionCandidate(...))
...
return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]
```

```python
def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key
    return sorted(candidates, key=key_fn)[0]
```

iii. The notes say the index contains repeated analysis entries for the same neural recording, so the converter should deduplicate down to 89 unique imaging sessions.

## 1-d. How are the data split into trials?

i. Trials are split using `StartFr` and `GrayFr`. For each trial, the AI takes the frames from corridor entry to gray-space entry and then further restricts to frames with `ft_move > 0`. It keeps variable-length trials rather than forcing a fixed frame count.

ii. ```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    if frames.size == 0:
        continue
```

iii. The notes justify this as "running-only texture frames" between `StartFr` and `GrayFr`, meant to mirror the paper's running-only analysis and the 0-4 m texture segment.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped only if no running frames remain after applying the `StartFr:GrayFr` window and the `ft_move > 0` filter. There is no additional trial-level quality control.

ii. ```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
```

iii. The notes say this removes trials without usable running texture frames and matches the decision to analyze running-only timepoints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the session spike matrix loaded by `utils.load_spk(...)`, which reads the deconvolved traces from `*_neural_data.npy`. Region labels come from retinotopy `iarea`.

ii. ```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
_, region_idx_all = area_labels_from_iarea(ret["iarea"])
```

iii. The notes state that the repository already provides the deconvolved fluorescence traces used in the paper, so no new calcium preprocessing should be computed.

## 2-b. How is the `neural` data processed?

i. The AI applies neuron selection first, then slices each kept trial to running frames between `StartFr` and `GrayFr`. It stores those per-trial arrays as variable-length `float32` matrices and does not pad them to a common trial length.

ii. ```python
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
spk_sel = spk[selected_mask]
...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
neural_trials.append(neural)
```

iii. The notes justify this as a computationally tractable, paper-style curation plus running-only trial extraction, even though that is stricter than simply loading the deconvolved traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only neurons in `V1`, `mHV`, `lHV`, or `aHV`, then further filters to a "paper-style" task-relevant subset: stimulus-selective neurons with `|d'| >= 0.3`, reward-prediction neurons in `aHV`, and a fallback top-`|d'|` selection if too few neurons survive.

ii. ```python
visual_mask = region_idx_all != 4
...
dp = safe_dprime(spk[:, stim1_fr], spk[:, stim2_fr])
stim_selective = visual_mask & np.isfinite(dp) & (np.abs(dp) >= DP_THRESHOLD)
...
reward_pred = (
    ahv_mask
    & np.isfinite(reward_dp)
    & np.isfinite(dp_sound)
    & (dp_sound > DP_THRESHOLD)
    & (reward_dp >= reward_dp_thr)
)
...
if selected.sum() < MIN_NEURONS_FALLBACK:
    ...
```

iii. `CONVERSION_NOTES.md` says neuron curation should follow the paper's selectivity and reward-prediction analyses and explicitly records a later patch tightening this selection to better match those analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to corridor entry via `StartFr`, but only the subsequent running frames up to `GrayFr` are retained. The first retained neural column is therefore the first running frame after trial start, not necessarily the exact `StartFr` frame.

ii. ```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. The notes say the intended alignment event is corridor entry while restricting analysis to running-only texture frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The nominal time bin is the median imaging frame interval estimated from `beh["ft"]` across sessions and stored in metadata as `time_bin_size`. No explicit resampling or temporal rebinning is applied to the neural matrix, but the sequence only keeps retained running frames.

ii. ```python
def compute_time_bin_ms(catalog: list[SessionCandidate]) -> float:
    ...
    medians.append(float(np.median(d) * SECONDS_PER_DAY * 1000.0))
    ...
```

```python
"time_bin_size": time_bin_ms,
```

iii. The notes say the converter should use the native imaging frame scale rather than inventing a new temporal grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. In code, `time_to_sound_cue_sec` is derived from `SoundFr`, the retained per-trial frame list, and a median frame duration estimated from `ft`.

ii. ```python
ft = np.asarray(beh["ft"][:nfr], dtype=float)
...
frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
...
sound_fr = np.asarray(beh["SoundFr"], dtype=int)
...
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
```

iii. The notes say the cue input should be a continuous time-to-cue variable aligned to the retained framewise trial representation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI converts cue timing into a per-trial index position within the retained running-frame sequence and multiplies the index difference by the median frame interval. It does not interpolate the cue onto actual frame timestamps.

ii. ```python
retained_idx = np.arange(frames.size, dtype=np.float32)
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
...
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The notes claim the variable should preserve elapsed time despite frame filtering, but the actual implementation does so with an index-based approximation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned to the same retained `frames` used to slice the neural matrix for that trial, so it has the same time dimension as the per-trial neural data.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
...
input_arr = np.vstack([...]).astype(np.float32)
if input_arr.shape[1] != T or output_arr.shape[1] != T:
    raise ValueError(...)
```

iii. The notes repeatedly state that all decoder variables should be constructed on the same retained running-frame trial axis.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date string `datexp` and mouse name `mname` from the session catalog.

ii. ```python
date = datetime.strptime(cand.db["datexp"], "%Y_%m_%d")
by_subject[cand.db["mname"]].append(date)
```

iii. The notes explain that complete training-day labels are not available for all sessions, so the AI uses calendar date information instead.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the AI computes the number of elapsed calendar days since that mouse's first recording date, stores it as a float, and repeats it across all retained frames of every trial in that session.

ii. ```python
first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
return {
    cand.session_id: float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
    for cand in catalog
}
```

```python
day = np.full(frames.shape, session_day, dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says "use day-since-first-recording as the training-day proxy."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In code, `time_since_trial_start_sec` is derived from `StartFr`, the retained frame list for the trial, and a median frame duration estimated from `ft`.

ii. ```python
ft = np.asarray(beh["ft"][:nfr], dtype=float)
...
frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
...
start_fr = np.asarray(beh["StartFr"], dtype=int)
```

iii. The notes frame this variable as elapsed time from trial start on the retained framewise trial axis.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI numbers the retained running frames `0, 1, 2, ...` and multiplies that index by the median frame interval, so time since start is index-based rather than computed from the actual `ft` timestamps of each retained frame.

ii. ```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The notes say actual elapsed time should be preserved after dropping non-running frames, but the implementation approximates it with uniform retained-frame spacing.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is defined on the same retained per-trial frame axis as the neural data, so the input and neural matrices have matching time dimensions.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
t_since = (retained_idx * frame_dt).astype(np.float32)
...
if input_arr.shape[1] != T or output_arr.shape[1] != T:
    raise ValueError(...)
```

iii. The notes justify all inputs as being repeated or computed on the same retained frame grid as the neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial `isRew` flag.

ii. ```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
...
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The notes say rewarded versus non-rewarded corridor identity should be taken directly from `isRew`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI simply repeats the trial's `isRew` value across every retained frame in that trial.

ii. ```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The notes justify this as a trialwise binary contextual variable that is easiest to represent uniformly as time-varying input.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. ```python
wall_name = np.asarray(beh["WallName"]).astype(str)
...
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. The notes say `WallName` is the safest label source because swap sessions can have ambiguous or missing `stim_id` values.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI preserves the exact wall-name strings as separate output classes, collecting the full dataset-wide vocabulary and mapping each unique name to an integer. This yields 15 categories rather than collapsing to 4 broad texture families.

ii. ```python
def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    names = set()
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names)
```

```python
stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}
...
"output_values": [
    stimulus_values,
    ...
]
```

iii. `CONVERSION_NOTES.md` explicitly says to "use exact `WallName` strings as stimulus labels" to preserve swap and non-leaf/circle variants.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`, with `LickTrind` used to associate lick frames to each trial.

ii. ```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
...
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
```

iii. The notes treat licking as a framewise behavioral output projected onto the retained neural trial frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI converts lick events to integer frame indices and marks a retained frame as 1 if its frame number appears in that trial's lick list; otherwise 0.

ii. ```python
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The notes justify a binary time-varying lick output rather than trial-level lick summaries.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by checking lick-frame membership on the same `frames` array used to slice the neural data for that trial.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The notes say all decoder streams should live on the same retained running-frame axis.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
...
pos = ft_pos[frames]
```

iii. The notes say `ft_Pos` is already the framewise within-corridor position signal needed for the decoder target.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI applies `floor(pos / 10)` on the retained frame positions and clips the result to `[0, 3]`, yielding four 1 m bins.

ii. ```python
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The notes explicitly map the 0-40 dm texture segment into four equal 10 dm bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are 0-10 dm, 10-20 dm, 20-30 dm, and 30-40 dm, encoded as integer bins 0-3.

ii. ```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
...
"output_values": [
    ...,
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    ...
]
```

iii. The notes justify this as the decoder-required 4 x 1 m discretization of the texture corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is evaluated on the same retained per-trial frame list used for the neural data.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The notes justify alignment by constructing every trial variable from the same retained `frames`.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`.

ii. ```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
...
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. The notes identify `ft_RunSpeed` as the direct source of the speed target before decoder-specific discretization.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first stores raw speed on each retained frame, then after all sessions are processed it computes global quartile edges over all retained running speeds in the converted dataset.

ii. ```python
output_trials.append(
    {
        ...
        "running_speed_raw": raw_speed,
    }
)
...
all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
```

iii. The notes say quartile binning should be global over all retained running frames, not per session, to satisfy the decoder requirement.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three threshold edges are taken from the global speed quartiles. Each retained frame is assigned with `np.searchsorted(...)`, and duplicated edges are nudged upward with `np.nextafter(...)`.

ii. ```python
for i in range(1, len(speed_edges)):
    if speed_edges[i] <= speed_edges[i - 1]:
        speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
...
thresholds = speed_edges[1:-1]
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The notes justify this as global quartile discretization, and explicitly mention speed-bin edges in metadata and validation checks.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Raw speed is sampled on the same retained `frames` array as the neural trial, then binned without changing that time axis.

ii. ```python
raw_speed = ft_speed[frames].astype(np.float32)
...
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The notes say speed should be derived directly from the retained running texture frames used for neural slicing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code includes several ad hoc safeguards: canonical-session selection among duplicates, finite-value checks for `stim_id`, `SoundDelPos`, and lick arrays, skipping trials with no retained running frames, setting the reward-prediction threshold to `inf` if no finite aHV values exist, and a fallback neuron-selection rule if fewer than 64 neurons survive. It also perturbs equal speed-bin edges upward.

ii. ```python
if uniq_walls.size and stim_id.size == uniq_walls.size:
    rew_like = uniq_walls[np.isfinite(stim_id) & (stim_id == 2)]
```

```python
if frames.size == 0:
    continue
...
reward_dp_thr = np.inf
...
if selected.sum() < MIN_NEURONS_FALLBACK:
    ...
```

```python
if speed_edges[i] <= speed_edges[i - 1]:
    speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
```

iii. The notes describe these choices as edge-case handling needed to keep all 89 sessions processable and to avoid degenerate samples or accidental over-selection.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the very large spike matrices, then computing neuron-selection statistics over them, especially the reward-prediction loop over trials. The code also rereads behavior files multiple times for metadata collection.

ii. ```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
...
for trial_idx in np.flatnonzero(valid_reward_trials):
    frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
    ...
    trial_means[:, trial_idx] = np.nanmean(spk[:, frames], axis=1)
```

```python
stimulus_values = collect_stimulus_values(full_catalog)
time_bin_ms = compute_time_bin_ms(full_catalog)
```

iii. `CONVERSION_NOTES.md` calls out spike-file I/O as dominant and also notes repeated behavior reloads and neuron-curation work as identifiable overhead.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The reward-prediction loop over rewarded trials, the per-trial main conversion loop, and the sample-scoring loop all operate trial by trial and could be vectorized or restructured.

ii. ```python
for trial_idx in np.flatnonzero(valid_reward_trials):
    frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
    ...
```

```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    ...
```

```python
for trial_idx, (s, g) in enumerate(zip(start, gray)):
    frames = np.arange(s, g, dtype=int)
    ...
```

iii. The notes mention code inefficiencies around repeated per-session/per-trial processing and extra behavior reloads.

## 12-c. What processing does the code repeat multiple times?

i. The converter reloads behavior repeatedly: once when collecting the stimulus vocabulary, again when estimating the global frame interval, again in sample selection, and again during actual session processing. It also rebuilds speed-related trial lists before final binning.

ii. ```python
for cand in catalog:
    beh = load_behavior(cand.exp_type, cand.beh_key)
    names.update(...)
```

```python
for cand in catalog:
    ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
```

```python
beh = load_behavior(candidate.exp_type, candidate.beh_key)
```

iii. The notes explicitly identify repeated behavior reloads for metadata and vocabulary collection as avoidable overhead.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes plotting-only summaries (`processing_info`, `cue_positions_dm`, `example_trial`), sample-selection heuristics, and a returned `speed_concat` value from `build_trial_arrays(...)` that is not used by `process_session(...)`. These computations do not affect the saved decoder arrays.

ii. ```python
return neural_trials, input_trials, output_trials, speed_concat, summary
...
neural_trials, input_trials, output_trials, speed_values, trial_summary = build_trial_arrays(...)
```

```python
proc_info = {
    **selection_info,
    **trial_summary,
    "cue_positions_dm": cue_positions,
    "example_trial": example_trial,
}
```

```python
def sample_candidate_score(candidate: SessionCandidate) -> tuple[int, int]:
    ...
```

iii. The notes justify some of this as sanity-check and debugging support, but it is not consumed by the downstream decoder once `converted_data.pkl` is written.
