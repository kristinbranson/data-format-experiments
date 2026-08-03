# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `Imaging_Exp_info.npy`, which contains 23 experiment types with 142 total entries. It deduplicates these to 89 unique sessions by base session key `<mouse>_<date>_<block>`. For each session, behavior data is loaded from `Beh_<exp_type>.npy`, neural spike data is loaded via `utils.load_spk()` from `spk/*_neural_data.npy`, and retinotopy/area labels are loaded from `retinotopy/*_trans.npz`.

ii.
```python
def build_session_catalog() -> list[SessionCandidate]:
    exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
    grouped: dict[str, list[SessionCandidate]] = defaultdict(list)
    for exp_type, db_list in exp_info.items():
        for db in db_list:
            session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            beh_key = session_id
            if "stimtype" in db:
                beh_key = f"{beh_key}_{db['stimtype']}"
            grouped[session_id].append(...)
    return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]

def load_behavior(exp_type: str, beh_key: str) -> dict:
    beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    return beh[beh_key]

# In process_session:
beh = load_behavior(candidate.exp_type, candidate.beh_key)
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 that it followed the reference notebook's approach of using `Imaging_Exp_info.npy` as the central metadata index, `load_spk()` from `utils.py` for neural data, and `load_retino()` pattern for retinotopy. The deduplication from 142 to 89 entries was documented in Step 4 as a resolution for the fact that the same recording appears under multiple analysis labels.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in each session's database entry. A sorted list of unique `mname` values becomes the `subjects` list, and each session is mapped to its subject index.

ii.
```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
# ...
"subject_idx": np.asarray(
    [subject_to_idx[s.subject] for s in processed_sessions], dtype=np.int64
),
```

iii. The AI noted in CONVERSION_NOTES.md Step 2 that there are 19 unique `mname` values, matching the paper's "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `<mouse>_<date>_<block>` keys from `Imaging_Exp_info.npy`. When multiple experiment types map to the same base session, a canonical entry is chosen (preferring entries without `stimtype` field). This yields 89 unique sessions matching the paper.

ii.
```python
def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key
    return sorted(candidates, key=key_fn)[0]
```

iii. CONVERSION_NOTES.md Step 4 documents that the notebook iterates 89 unique imaging recordings after deduplication, matching the paper's claim. The deduplication strategy is documented in Step 5 Key Decision 1.

## 1-d. How are the data split into trials?

i. For each session, trials are iterated from index 0 to `beh['ntrials']`. Each trial spans frames from `StartFr[trial_idx]` to `GrayFr[trial_idx]` (corridor entry to gray space entry). Only frames where `ft_move > 0` (running) are retained. Trials with no running frames are skipped.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    if frames.size == 0:
        continue
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2 states: "Use the texture segment only (StartFr:GrayFr)" and Decision 3: "Keep running frames only inside the texture segment." This follows the paper's rule "We only considered timepoints during running for analysis."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by a single criterion: they must have at least one running frame within the texture segment (StartFr to GrayFr where ft_move > 0). No other trial-level quality filtering is applied (e.g., no filtering based on licking, reward, or behavioral performance).

ii.
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
```

iii. The AI justified this in CONVERSION_NOTES.md Step 5 Decision 3, noting that removing non-running frames follows the paper's approach. The total trial count of 38,110 matches the raw data count, indicating all trials with running frames are kept.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the deconvolved fluorescence traces stored in `*_neural_data.npy` files in the `spk/` directory, loaded via `utils.load_spk()` which concatenates the `['spks']` arrays into a neuron x frame matrix.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
```

iii. CONVERSION_NOTES.md Step 1 documents: "load_spk() directly loads *_neural_data.npy and concatenates ['spks'], so the saved neural signal is already the processed neural activity used downstream." Step 3 confirms: "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes: (1) neuron selection based on a custom curation combining stimulus selectivity (|d'| >= 0.3) and reward-prediction neuron criteria, with a fallback for sessions with too few selected neurons; (2) frame selection retaining only running frames (ft_move > 0) within the texture segment (StartFr:GrayFr); (3) no normalization, rebinning, or dF/F computation is applied.

ii.
```python
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
spk_sel = spk[selected_mask]
# ...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md Step 1 states: "The reference code does not compute delta-F-over-F inside this repository." Step 5 Decision 4 explains the curation approach: "Keep neurons in retinotopically assigned visual regions that are either corridor-responsive and stimulus-selective with |d'| >= 0.3 or reward-prediction neurons."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neuron filtering uses a custom two-pathway curation:
- **Stimulus-selective neurons**: Visual cortex neurons (not area "other") with |d'| >= 0.3 between the primary rewarded and non-rewarded stimuli, computed on running corridor frames (`ft_CorrSpc & ft_move > 0`).
- **Reward-prediction neurons**: aHV neurons with cue-frame stimulus d' > 0.3 and late-vs-early cue d' above the session's 95th aHV percentile, using mean activity over running frames in the 0.5-4.0m texture segment.
- **Fallback**: If fewer than 64 neurons pass, the top |d'| visual cortex neurons are selected.

ii.
```python
stim_selective = visual_mask & np.isfinite(dp) & (np.abs(dp) >= DP_THRESHOLD)
# ...
reward_pred = (
    ahv_mask & np.isfinite(reward_dp) & np.isfinite(dp_sound)
    & (dp_sound > DP_THRESHOLD)
    & (reward_dp >= reward_dp_thr)
)
selected = stim_selective | reward_pred
if selected.sum() < MIN_NEURONS_FALLBACK:
    # fallback selection
```

iii. CONVERSION_NOTES.md Step 10 documents the iteration where the curation was tightened to better match the reference code, including removing the extra gray-space gate and changing reward-neuron selection to require cue-frame stimulus selectivity plus aHV 95th-percentile threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry), which corresponds to `StartFr`. Each trial's neural data begins at `StartFr` and ends at `GrayFr`, with only running frames retained. The temporal alignment event is documented as "corridor entry (trial start / StartFr)".

ii.
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The AI's metadata records: `"temporal_alignment_event": "corridor entry (trial start / StartFr)"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate (~315 ms per frame, ~3.17 Hz). No temporal rebinning is applied. The time bin size is computed as the median inter-frame interval from the `ft` timestamps.

ii.
```python
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

iii. CONVERSION_NOTES.md Step 1 notes the calcium imaging frame rate is 3.17 Hz from `data_process_script.ipynb`. No rebinning is needed since the data is already at the native imaging rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundFr` (the frame index of the sound cue for each trial) and the retained running frame indices. The frame time interval `frame_dt` is computed from `ft` (framewise timestamps) as the median inter-frame interval.

ii.
```python
sound_fr = np.asarray(beh["SoundFr"], dtype=int)
ft = np.asarray(beh["ft"][:nfr], dtype=float)
dft = np.diff(ft)
dft = dft[np.isfinite(dft) & (dft > 0)]
frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
```

iii. CONVERSION_NOTES.md Step 5 maps this variable and references `SoundTime`, `ft`, and `SoundFr` as source fields. The cue-aligned analyses in the reference code use `SoundFr` as the alignment anchor.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound cue's position within the retained running frames is found via `np.searchsorted`. Time to cue is then computed as `(cue_idx - retained_frame_idx) * frame_dt`, where `frame_dt` is the median inter-frame interval. This means positive values are before the cue (frames approaching the cue) and negative values are after.

**Note**: This uses an approximate uniform frame spacing rather than actual `ft` timestamps. The actual clock time between retained frames may vary when non-running frames have been removed.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 states the mapping should be "(SoundTime[trial] - ft[frame]) * 86400" using actual timestamps, but the implemented code uses approximate frame-count-based timing instead.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to cue is computed for the same set of retained running frames as the neural data, so they are inherently aligned frame-by-frame.

ii.
```python
# Both neural and t_to_cue use the same `frames` array:
neural = spk_sel[:, frames].astype(np.float32, copy=False)
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The AI ensures alignment by constructing all trial variables from the same `frames` array within `build_trial_arrays()`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field in each session's database entry (recording date in `YYYY_MM_DD` format).

ii.
```python
def compute_day_offsets(catalog: list[SessionCandidate]) -> dict[str, float]:
    by_subject: dict[str, list[datetime]] = defaultdict(list)
    session_dates: dict[str, datetime] = {}
    for cand in catalog:
        date = datetime.strptime(cand.db["datexp"], "%Y_%m_%d")
        by_subject[cand.db["mname"]].append(date)
        session_dates[cand.session_id] = date
    first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
    return {
        cand.session_id: float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
        for cand in catalog
    }
```

iii. CONVERSION_NOTES.md Step 5 Decision 8 explains: "Use day-since-first-recording as the training-day proxy: The paper does not provide a complete per-session training-day label, while calendar date is available for every recording."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the earliest recording date is identified. Day of training is computed as the number of calendar days between the session's recording date and the subject's first recording date. This value is constant across all timepoints within a trial (and within a session).

ii.
```python
first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
return {
    cand.session_id: float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
    for cand in catalog
}
# In build_trial_arrays:
day = np.full(frames.shape, session_day, dtype=np.float32)
```

iii. The AI noted that some sessions have a `db['days']` field that could serve as a sanity check, but chose calendar date as the universally available proxy.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the retained running frame indices within each trial. The frame time interval `frame_dt` is used to convert frame counts to seconds.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 lists the source as "beh['ft'], StartFr" and notes this should be "(ft[frame] - ft[StartFr[trial]]) * 86400."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is computed as `retained_frame_index * frame_dt`, where `retained_frame_index` is the sequential index (0, 1, 2, ...) within the retained running frames. This approximates elapsed time by assuming uniform frame spacing at the median rate.

**Note**: This does not reflect actual elapsed clock time because gaps from dropped non-running frames are not accounted for. The actual `ft` timestamps would give exact elapsed time.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The planned mapping in CONVERSION_NOTES.md Step 5 describes using actual timestamps "(ft[frame] - ft[StartFr[trial]]) * 86400", but the implementation uses the approximate frame-count method instead.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned frame-by-frame with neural data since both are indexed by the same retained running frames array.

ii. Same `frames` array is used for both neural data and time computation within `build_trial_arrays()`.

iii. Alignment is inherent in the construction approach.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a per-trial boolean/float array indicating whether the trial is in a rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
# ...
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. CONVERSION_NOTES.md Step 5 documents: "Uses rewarded-corridor identity even in unsupervised sessions, matching reference use of isRew."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trialwise `isRew` value (0 or 1) is repeated across all retained timepoints in the trial. No additional processing is applied.

ii.
```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The AI decided to broadcast per-trial values to time-varying arrays for uniformity (Step 5 Decision 6).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, which provides the texture name for each trial (e.g., "circle1", "leaf1", "wood1_swap2").

ii.
```python
wall_name = np.asarray(beh["WallName"]).astype(str)
# ...
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 5 Decision 5 states: "Use exact WallName strings as stimulus labels: This avoids ambiguity from stim_id NaNs in swap sessions and preserves non-leaf/circle texture pairs."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique WallName strings are collected across all sessions, sorted alphabetically, and mapped to integer indices 0-14 (15 categories total). The per-trial wall name is converted to its integer index and repeated across all timepoints in the trial.

ii.
```python
def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    names = set()
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names)

stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}
```

iii. The 15 categories are: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5. The AI chose to preserve individual stimulus identities rather than grouping by reward status.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickFr']` (frame indices of lick events) and `beh['LickTrind']` (trial index for each lick event).

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
```

iii. CONVERSION_NOTES.md Step 5 maps this to `LickFr` and `LickTrind`, referencing `spk_2_firstLick` and `spk_2_cue` as the reference code functions that use these fields.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frame indices are filtered to those belonging to the current trial (matching `LickTrind`). A binary vector is created: 1 if the retained running frame matches any lick frame, 0 otherwise.

ii.
```python
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The AI uses `np.isin` to match retained imaging frames against lick event frames, producing a binary time-varying output as required.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned frame-by-frame with neural data because `np.isin(frames, lick_frames_trial)` uses the same `frames` array as the neural data indexing.

ii. Both neural and licking use the same `frames` array within `build_trial_arrays()`.

iii. Alignment is inherent in the construction.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh['ft_Pos']`, the framewise position of the mouse in the corridor (in decimeters).

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
# ...
pos = ft_pos[frames]
```

iii. CONVERSION_NOTES.md Step 5 maps this variable to `ft_Pos` on retained frames.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw position (in decimeters) at each retained running frame is discretized into 4 bins of 10 dm (1 m) each using floor division.

ii.
```python
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The AI's position binning follows the instruction's requirement of "4 equal-length, 1-m-long spatial bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position in decimeters is divided by 10 and floored, then clipped to range [0, 3]:
- Bin 0: 0-1m (0-10 dm)
- Bin 1: 1-2m (10-20 dm)
- Bin 2: 2-3m (20-30 dm)
- Bin 3: 3-4m (30-40 dm)

ii.
```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The instructions specify "4 equal-length, 1-m-long spatial bins" and the corridor texture area is 4 m (40 dm), making 10 dm per bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read at the same retained running frame indices used for neural data, providing frame-by-frame alignment.

ii.
```python
pos = ft_pos[frames]  # same `frames` used for neural = spk_sel[:, frames]
```

iii. Alignment is inherent.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, the framewise running speed.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
# ...
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 maps this to `ft_RunSpeed` on retained frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Raw running speed values at retained running frames are collected across all sessions. No additional processing (smoothing, normalization, etc.) is applied to the speed values before binning.

ii.
```python
raw_speed = ft_speed[frames].astype(np.float32)
# Later, all speeds are concatenated globally:
all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
```

iii. The AI collects raw speed values for quartile computation, which is a global statistic across all retained frames.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed from all retained running-frame speed values across all sessions. Speed is then binned into 4 categories using `np.searchsorted` against the quartile thresholds (25th, 50th, 75th percentiles).

ii.
```python
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
# ...
thresholds = speed_edges[1:-1]
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The instructions specify "4 bins, each corresponding to 25% of the data," which is exactly quartile binning. The resulting edges are [-19.17, 12.51, 25.51, 41.21, 161.08].

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read at the same retained running frame indices as neural data.

ii.
```python
raw_speed = ft_speed[frames].astype(np.float32)  # same `frames` as neural
```

iii. Alignment is inherent.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Trials with no running frames**: Skipped entirely (`if frames.size == 0: continue`).
- **Sessions with few selective neurons**: Fallback to top-|d'| visual cortex neurons when fewer than 64 pass the selectivity threshold.
- **Missing lick data**: Sessions with `LickFr.shape = (0,)` naturally produce all-zero licking outputs.
- **NaN cue positions**: `np.isfinite` checks are applied before computing reward-prediction d' thresholds.
- **Sessions without aHV reward d' data**: Threshold is set to infinity, preventing accidental over-selection.
- **Degenerate speed edges**: Adjacent equal quartile edges are nudged with `np.nextafter` to avoid zero-width bins.

ii.
```python
if selected.sum() < MIN_NEURONS_FALLBACK:
    candidate = visual_mask & np.isfinite(dp)
    # ... fallback top-|dp| selection

if frames.size == 0:
    continue

for i in range(1, len(speed_edges)):
    if speed_edges[i] <= speed_edges[i - 1]:
        speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
```

iii. CONVERSION_NOTES.md Step 10 documents edge case reviews including handling of degenerate sample sessions and sessions with no finite aHV reward d' values.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading the large spike data files (405 GB total across 89 sessions) via `utils.load_spk()`. Each session takes 2-12 seconds depending on file size. The full conversion completed in ~501 seconds (~8.4 minutes). The neuron selection computation (d-prime calculations) is also moderately expensive.

ii. From the output log, per-session times range from 1.3s to 12.7s.

iii. CONVERSION_NOTES.md Step 7 documents timing: "Full conversion estimate: ~2.10 s/GB, ~870s (~14.5 min)." The actual run was faster at 501s.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized:
- The per-trial loop in `build_trial_arrays()` that iterates through all trials to construct neural, input, and output arrays.
- The per-trial loop in `compute_neuron_selection()` that computes trial-mean activity for reward-prediction d' (lines 236-240).

ii.
```python
# Per-trial loop in build_trial_arrays:
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    # ... repeated per-trial operations

# Per-trial loop in compute_neuron_selection:
for trial_idx in np.flatnonzero(valid_reward_trials):
    frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
    frames = frames[running[frames] & (ft_pos[frames] >= 5.0) & (ft_pos[frames] <= TEXTURE_LENGTH_DM)]
    if frames.size:
        trial_means[:, trial_idx] = np.nanmean(spk[:, frames], axis=1)
```

iii. CONVERSION_NOTES.md Step 6 acknowledges: "Full-mode metadata currently recomputes frame-interval medians by reloading behavior files" and "Stimulus vocabulary collection reloads behavior dictionaries once more."

## 12-c. What processing does the code repeat multiple times?

i. Several operations are redundantly repeated:
- **Behavior file loading**: `collect_stimulus_values()` reloads all behavior files after `build_session_catalog()` already processed them, and `compute_time_bin_ms()` does the same.
- **Stimulus pair identification**: `get_reference_pair()` re-determines rewarded/non-rewarded stimuli, duplicating logic partially present in the catalog building.
- **Frame timing computation**: `frame_dt` is recomputed per-session in `build_trial_arrays()` even though it could be computed once globally (as `compute_time_bin_ms()` does).

ii.
```python
# Three separate functions that each reload behavior:
stimulus_values = collect_stimulus_values(full_catalog)  # loads all beh files
time_bin_ms = compute_time_bin_ms(full_catalog)           # loads all beh files again
# Then process_session loads each beh file a third time
```

iii. CONVERSION_NOTES.md Step 6 notes: "Full-mode metadata currently recomputes frame-interval medians by reloading behavior files" and "Stimulus vocabulary collection reloads behavior dictionaries once more after catalog creation."

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of processing are computed but not used in the final output:
- **Processing info/example trial data**: Detailed per-session processing info (trial lengths, cue positions, example trial neural/input/output plots) is computed for every session but only used for `--show-processing` mode.
- **Speed concatenation per session**: `all_speeds` is collected per session for trial summary but is redundant with the global speed collection in `main()`.
- **Full processing_info dict**: Contains detailed selection info, trial summaries, and cue positions that are stored but not included in the final pickle.

ii.
```python
# Processing info computed for every session but only used for plots:
proc_info = {
    **selection_info,
    **trial_summary,
    "cue_positions_dm": cue_positions,
    "example_trial": example_trial,
}
```

iii. The AI acknowledged in CONVERSION_NOTES.md that the processing plots and detailed info are for debugging/validation purposes rather than the final output.
