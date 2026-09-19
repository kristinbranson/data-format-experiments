# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a master session catalog from `data/beh/Imaging_Exp_info.npy`, deduplicates repeated entries down to one canonical record per base session id, then loads each session separately: behavior from `Beh_<exp_type>.npy`, spikes with `utils.load_spk(...)`, and retinotopy from the matching `*_trans.npz` file.

ii. ```python
def build_session_catalog() -> list[SessionCandidate]:
    exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
    grouped: dict[str, list[SessionCandidate]] = defaultdict(list)
    for exp_type, db_list in exp_info.items():
        for db in db_list:
            session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            ...
    return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]
```
```python
beh = load_behavior(candidate.exp_type, candidate.beh_key)
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. `CONVERSION_NOTES.md` Step 5 says the dataset should be deduplicated from repeated analysis entries to the 89 unique imaging recordings in `Imaging_Exp_info.npy`, and Step 6 says the script reuses the reference loaders from `code/utils.py`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by `mname`. The converted dataset uses the sorted unique mouse names as `subjects`, and `subject_idx` maps each processed session to its mouse.

ii. ```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
...
"subject_idx": np.asarray(
    [subject_to_idx[s.subject] for s in processed_sessions], dtype=np.int64
),
```
```python
subject=candidate.db["mname"],
```

iii. The notes state that `mname` is the subject identifier and that there are 19 unique mice, so no derived subject split is needed.

## 1-c. How are the data split into sessions?

i. A session is defined by the base id `<mouse>_<date>_<block>`. If that recording appears under multiple experiment types, the AI keeps one canonical copy, preferring entries without `stimtype`.

ii. ```python
session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
beh_key = session_id
if "stimtype" in db:
    beh_key = f"{beh_key}_{db['stimtype']}"
```
```python
def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key

    return sorted(candidates, key=key_fn)[0]
```

iii. Step 5 in the notes explicitly justifies deduplicating 142 behavior-index entries down to 89 unique neural recordings and says that for swap sessions one canonical copy is enough because the per-trial `WallName` carries the real stimulus identity.

## 1-d. How are the data split into trials?

i. Trials are built from the frame interval `StartFr:GrayFr` for each behavioral trial, then reduced to only the running frames (`ft_move > 0`) inside that interval. The AI does not use `ft_trInd` to define frame membership.

ii. ```python
start_fr = np.asarray(beh["StartFr"], dtype=int)
gray_fr = np.asarray(beh["GrayFr"], dtype=int)
...
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    if frames.size == 0:
        continue
```

iii. Step 5 says the conversion should use the texture segment only (`StartFr:GrayFr`) and should keep running frames only, because the paper’s analyses “use running timepoints” and this also avoids extremely long paused traversals.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply a dataset-wide long-trial threshold. A trial is kept if it has at least one running frame between `StartFr` and `GrayFr`; otherwise it is skipped.

ii. ```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    if frames.size == 0:
        continue
```

iii. The notes justify this indirectly: Step 5 says restricting to running frames removes the extremely long paused portions of trials, and Step 6 does not mention any additional percentile-based trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays come from the deconvolved spike-like traces loaded by `utils.load_spk(...)`, and the neuron region labels come from `iarea` in the retinotopy file.

ii. ```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
_, region_idx_all = area_labels_from_iarea(ret["iarea"])
```

iii. Step 1 of the notes says the paper’s repository already stores the processed deconvolved traces used downstream, so the conversion should load those directly rather than recomputing fluorescence preprocessing.

## 2-b. How is the `neural` data processed?

i. The AI first filters neurons with a task-driven curation rule, then slices each trial to running frames only between `StartFr` and `GrayFr`, and stores the result as `float32`.

ii. ```python
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
spk_sel = spk[selected_mask]
...
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. Step 5 argues for two processing choices: use only the texture segment and only running frames, and curate neurons with “paper-style” selectivity / reward-prediction rules to keep the dataset task-relevant and computationally tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are first mapped to `V1`, `mHV`, `lHV`, `aHV`, or `other`. The AI then keeps visual-area neurons that are either stimulus-selective with `|d'| >= 0.3`, reward-prediction neurons in aHV, or part of a fallback top-`|d'|` set if too few neurons survive.

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

selected = stim_selective | reward_pred
```
```python
if selected.sum() < MIN_NEURONS_FALLBACK:
    ...
    selected[order[:nkeep]] = True
```

iii. Step 5 and Step 6 explicitly justify this as “paper-style neuron selection”: stimulus-selective neurons plus aHV reward-prediction neurons, with a fallback to avoid ending up with too few neurons in a session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial alignment starts at corridor entry (`StartFr`), but the stored per-trial data end at `GrayFr` and include only running frames within that interval. Trials therefore share the same nominal start event but not the full frame sequence between start and gray-space entry.

ii. ```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    ...
    neural = spk_sel[:, frames].astype(np.float32, copy=False)
```
```python
"temporal_alignment_event": "corridor entry (trial start / StartFr)",
"trial_end_event": "entry into gray space (GrayFr)",
"frame_selection": "running-only frames within texture area",
```

iii. The notes repeatedly justify corridor-entry alignment and texture-only windows, while also saying running-only frames should be retained to mirror the paper’s analysis regime.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging-frame resolution. It estimates one global time bin size from the median positive `ft` difference across sessions and does not perform explicit temporal rebinning.

ii. ```python
def compute_time_bin_ms(catalog: list[SessionCandidate]) -> float:
    medians = []
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
        d = np.diff(ft)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            medians.append(float(np.median(d) * SECONDS_PER_DAY * 1000.0))
    return float(np.median(medians))
```
```python
"time_bin_size": time_bin_ms,
```

iii. The notes say the imaging frame is the relevant native time base. They also say actual frame times matter because non-running frames are dropped, although the final code uses a median frame interval rather than per-frame timestamps for the time inputs.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. In the final code it is derived mainly from `SoundFr`, the running-frame subset of `StartFr:GrayFr`, and the session-wide median frame interval estimated from `ft`. It does not use `SoundTime` or full per-frame timestamps directly.

ii. ```python
ft = np.asarray(beh["ft"][:nfr], dtype=float)
dft = np.diff(ft)
dft = dft[np.isfinite(dft) & (dft > 0)]
frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
...
sound_fr = np.asarray(beh["SoundFr"], dtype=int)
...
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
```

iii. The notes originally proposed using `beh['SoundTime']` and `ft`, but the final script instead approximates cue timing with `SoundFr` and a median frame duration.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained trial window, the AI finds where the cue frame would fall in the retained running-frame sequence with `np.searchsorted`, then subtracts the retained frame indices and scales by a constant `frame_dt` to get seconds.

ii. ```python
retained_idx = np.arange(frames.size, dtype=np.float32)
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
...
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The stated rationale in Step 5 was that time inputs should preserve elapsed time even after non-running frames are dropped, but the implemented code uses a constant frame interval on the compressed running-only frame index rather than true per-frame elapsed times.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned to the same retained running-frame sequence used for each trial’s neural array, not to the full set of imaged corridor frames.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The AI’s general alignment rule in the notes is “running-only frames within the texture segment,” so the cue-time input is forced onto that same reduced frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date string `datexp` for each mouse in the session catalog, grouped by `mname`.

ii. ```python
date = datetime.strptime(cand.db["datexp"], "%Y_%m_%d")
by_subject[cand.db["mname"]].append(date)
session_dates[cand.session_id] = date
```

iii. Step 5 says the paper does not provide a complete per-session training-day label, so the AI chose elapsed calendar days since first recording as a proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the AI finds the earliest recording date and stores each session’s day offset in calendar days from that date. The scalar is then broadcast across all retained timepoints in the trial.

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

iii. The justification is explicit in the notes: calendar-day offset was used because complete training-day annotations were not consistently available across all sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the final code it is derived from `StartFr`, the retained running-frame sequence between `StartFr` and `GrayFr`, and a median frame interval estimated from `ft`.

ii. ```python
ft = np.asarray(beh["ft"][:nfr], dtype=float)
dft = np.diff(ft)
dft = dft[np.isfinite(dft) & (dft > 0)]
frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
...
start_fr = np.asarray(beh["StartFr"], dtype=int)
...
retained_idx = np.arange(frames.size, dtype=np.float32)
```

iii. As with time-to-cue, the notes originally planned to use exact frame times from `ft`, but the final implementation uses retained running-frame counts times a single `frame_dt`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI sets the first retained running frame to time 0 and increments subsequent retained frames by `frame_dt`; non-running gaps are removed rather than counted as elapsed time.

ii. ```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The notes say elapsed time should be preserved after dropping non-running frames, but the final code instead constructs time from the compressed retained-frame index.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned to the same per-trial running-frame subset used for the neural matrix.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. This follows the AI’s global decision to make every stream share the same running-only texture-frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`, the per-trial rewarded-corridor flag.

ii. ```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
...
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. Step 5 explicitly maps `beh['isRew']` to `reward_available` and says it should be used even in unsupervised sessions where it still marks corridor identity.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation is applied beyond repeating the per-trial `isRew` value across all retained timepoints in that trial.

ii. ```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The notes justify this as a decoder input that should remain trial-constant but be represented in a time-varying array for uniformity with the other inputs.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii. ```python
wall_name = np.asarray(beh["WallName"]).astype(str)
...
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. Step 5 says `WallName` was preferred because `stim_id` has `NaN` cases in swap sessions and because `WallName` preserves rock/wood and swap identities directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI does not collapse walls into four base texture classes. Instead it builds a global vocabulary of the exact `WallName` strings present across the dataset and encodes each trial with that exact wall label, broadcast across the trial.

ii. ```python
def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    names = set()
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names)
```
```python
stimulus_values = collect_stimulus_values(full_catalog)
stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}
...
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. The notes explicitly justify preserving exact `WallName` strings rather than pooling textures, to avoid ambiguity and to keep swap-specific and non-leaf/circle stimuli distinct.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii. ```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
...
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
```

iii. The notes map licking directly from the lick-frame annotations and describe it as a binary per-frame output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI selects lick frames assigned to that trial, casts them to integers, and marks each retained running frame as 1 if its frame number appears in that lick-frame list.

ii. ```python
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The notes justify a binary time-varying licking output, but there is no separate written justification for using the running-only retained frames beyond the global frame-selection rule.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned on the same retained running-frame indices used for the trial’s neural data.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. This follows the AI’s overall “running-only frames within the texture segment” alignment policy.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
...
pos = ft_pos[frames]
```

iii. Step 5 maps `beh['ft_Pos']` directly to the position output and notes that the raw units are decimeters over the 0-4 m texture segment.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI reads `ft_Pos` on the retained frames, divides by 10 dm, floors the result, and clips to the 4 requested 1 m bins.

ii. ```python
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The notes explicitly justify this as the requested conversion from the 0-40 dm texture segment into four 1 m categories.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are fixed at 0, 10, 20, 30, and 40 dm, yielding bins 0-1 m, 1-2 m, 2-3 m, and 3-4 m.

ii. ```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```
```python
"output_values": [
    stimulus_values,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
    make_speed_bin_names(speed_edges),
],
```

iii. The notes say the decoder specification itself drove this thresholding choice.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same retained running frames that define each neural trial.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Again, alignment follows the AI’s global retained-frame policy rather than the full imaged corridor window.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. ```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
...
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. Step 5 maps `ft_RunSpeed` directly to the running-speed output and says the final categories should be quartiles over the included frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI stores raw speed values on each retained running frame during session processing, concatenates them across all sessions, computes global quartile edges with `np.quantile`, then assigns each sample to a bin with `np.searchsorted`.

ii. ```python
raw_speed = ft_speed[frames].astype(np.float32)
...
all_speed_values.append(trial["running_speed_raw"])
...
all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
```
```python
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The notes explicitly justify global quartiles as matching the decoder request for four bins with 25% of the data each, using only the retained running-only texture frames.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global empirical quartile edges of all retained running-speed samples. After a monotonicity fix for ties, `searchsorted(..., side="right")` maps each frame to one of four categories.

ii. ```python
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
for i in range(1, len(speed_edges)):
    if speed_edges[i] <= speed_edges[i - 1]:
        speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
```
```python
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The notes say this was chosen to satisfy the decoder spec’s “25% of the data” requirement.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned on the same retained running-frame subset used for the neural trial matrices.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
raw_speed = ft_speed[frames].astype(np.float32)
...
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. This is another consequence of the AI’s running-only frame-selection rule.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles some missingness defensively with `np.isfinite(...)`, skips trials that end up with zero retained frames, and ignores non-finite lick annotations. It partially trims framewise arrays to the neural frame count with `[:nfr]`, but it does not explicitly clip `StartFr`/`GrayFr` trial windows to the imaged frame count.

ii. ```python
ft = np.asarray(beh["ft"][:nfr], dtype=float)
ft_move = np.asarray(beh["ft_move"][:nfr], dtype=float)
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
```
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
...
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
```

iii. There is no strong dedicated justification section for this in the notes; the handling mostly follows from the running-only trial construction and a few defensive `np.isfinite` checks.

## 12-a. What are the most time-consuming steps of the code?

i. The main costs are loading the very large spike files and then computing neuron-selection statistics over them. The full conversion log shows per-session runtimes are dominated by this phase.

ii. ```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
...
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
```
```python
dp = safe_dprime(spk[:, stim1_fr], spk[:, stim2_fr])
...
for trial_idx in np.flatnonzero(valid_reward_trials):
    frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
    ...
    trial_means[:, trial_idx] = np.nanmean(spk[:, frames], axis=1)
```

iii. Step 6 notes identify spike loading as the dominant cost and also mention additional overhead from repeated behavior-file reloads and neuron-selection computations.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loops are the per-trial reward-prediction loop in `compute_neuron_selection`, the per-trial conversion loop in `build_trial_arrays`, and the repeated per-trial lick-count loop in `sample_candidate_score`.

ii. ```python
for trial_idx in np.flatnonzero(valid_reward_trials):
    frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
    ...
    trial_means[:, trial_idx] = np.nanmean(spk[:, frames], axis=1)
```
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    ...
    neural_trials.append(neural)
```
```python
for trial_idx, (s, g) in enumerate(zip(start, gray)):
    frames = np.arange(s, g, dtype=int)
    ...
    if np.isin(frames, lfr).any():
        lick_trials += 1
```

iii. The notes do not discuss vectorization in detail, but Step 6 does acknowledge some efficiency issues, especially repeated file loading and avoidable extra passes over behavior.

## 12-c. What processing does the code repeat multiple times?

i. The script reloads behavior files multiple times for different bookkeeping tasks: collecting stimulus labels, computing time-bin metadata, sample selection, and then the real session conversion. It also re-derives per-trial frame lists in several places.

ii. ```python
stimulus_values = collect_stimulus_values(full_catalog)
time_bin_ms = compute_time_bin_ms(full_catalog)
...
ranked = sorted(
    catalog,
    key=lambda c: (
        -sample_candidate_score(c)[0],
        session_spk_size(c),
        c.session_id,
    ),
)
```
```python
def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
```
```python
def compute_time_bin_ms(catalog: list[SessionCandidate]) -> float:
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
```

iii. Step 6 explicitly records two repeated-processing inefficiencies: behavior reloads for frame-interval metadata and another pass to collect the stimulus vocabulary.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores intermediate structures that are only used transiently: raw running speeds before later binning, detailed `processing_info` summaries, example plots, sample-session scoring, and neuron-selection diagnostics. None of these survive into the downstream decoder inputs/outputs except the final binned results and metadata summaries.

ii. ```python
output_trials.append(
    {
        "visual_stimulus_category": stim_val,
        "licking": licking,
        "position_bin": pos_bin,
        "running_speed_raw": raw_speed,
    }
)
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
if args.show_processing:
    for session in processed_sessions[:2]:
        plot_processing(session, speed_edges)
```

iii. Step 6 mentions some of this explicitly as overhead added for validation and inspection rather than the final decoder dataset itself.
