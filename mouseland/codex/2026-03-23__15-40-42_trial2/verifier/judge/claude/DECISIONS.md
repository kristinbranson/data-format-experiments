# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, groups entries by experiment type, and deduplicates to 89 unique sessions by base session ID `<mouse>_<date>_<blk>`. For each session it loads the behavior from `Beh_<exp_type>.npy`, the neural data via `utils.load_spk()` (which loads `*_neural_data.npy` and concatenates the `spks` planes), and the retinotopy from `*_trans.npz`. A canonical entry is chosen per session when duplicates exist across experiment types.

ii.
```python
exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
# ...
session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
# ...
beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. The AI follows the same loading pattern as the reference code, using `Imaging_Exp_info.npy` as the master index and loading from the same three data directories. It reuses `utils.load_spk()` from the reference code.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from `db['mname']` in each catalog entry. Subjects are the sorted unique mouse names across all sessions in the catalog.

ii.
```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
```

iii. Directly uses the `mname` field from the experiment metadata, consistent with the reference approach.

## 1-c. How are the data split into sessions?

i. A session is one unique `<mouse>_<date>_<blk>` combination. When the same recording appears under multiple experiment types, a canonical entry is chosen (preferring non-`stimtype` entries). This yields 89 unique sessions.

ii.
```python
def build_session_catalog() -> list[SessionCandidate]:
    # ...
    grouped: dict[str, list[SessionCandidate]] = defaultdict(list)
    for exp_type, db_list in exp_info.items():
        for db in db_list:
            session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            # ...
            grouped[session_id].append(SessionCandidate(...))
    return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]
```

iii. The deduplication logic ensures each physical recording is processed once. The AI uses `choose_canonical` to pick a preferred entry, which is slightly more elaborate than the reference's first-seen approach but achieves the same result: 89 unique sessions.

## 1-d. How are the data split into trials?

i. Trials are split using `StartFr` and `GrayFr` from the behavior data. For each trial index (0 to `ntrials`-1), frames from `StartFr[trial]` to `GrayFr[trial]` are selected, then further filtered to only running frames (`ft_move > 0`). Trials with no surviving running frames are dropped.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    if frames.size == 0:
        continue
```

iii. The AI uses `StartFr:GrayFr` boundaries and filters to running-only frames, citing the paper's statement "We only considered timepoints during running for analysis." The reference solution uses `ft_trInd` and `ft_CorrSpc` instead and does not filter by running status.

## 1-e. How are trials filtered based on quality controls?

i. Trials with zero running frames within `StartFr:GrayFr` are dropped. No explicit trial-length percentile filter is applied.

ii.
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
```

iii. The AI relies on the running-frame filter to implicitly remove degenerate trials. The reference applies an explicit 99th-percentile trial length filter to remove animals that stopped mid-corridor. Both approaches keep all 38,110 trials after filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `*_neural_data.npy` loaded via `utils.load_spk()`, which concatenates imaging planes. The visual area comes from `iarea` in the retinotopy files.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
_, region_idx_all = area_labels_from_iarea(ret["iarea"])
```

iii. Same source data as the reference.

## 2-b. How is the `neural` data processed?

i. After loading, neurons are selected by a paper-style d-prime curation procedure (stimulus selectivity |d'| >= 0.3 and/or reward-prediction neurons). The selected neurons' activity at retained running frames is stored as float32. Non-running frames are excluded.

ii.
```python
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
spk_sel = spk[selected_mask]
# ...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. The AI applies an analysis-specific neuron curation (d-prime selectivity) that goes beyond what the reference does. The reference simply keeps all neurons in the four visual areas without d-prime filtering.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two stages: (1) assigned to visual regions (V1, mHV, lHV, aHV, other) based on `iarea`, and (2) filtered by d-prime stimulus selectivity (|d'| >= 0.3 between primary rewarded and non-rewarded corridors) and/or reward-prediction criteria. A fallback selects the top neurons by |d'| if too few pass. This yields ~357K neurons total (mean ~4,015/session).

ii.
```python
def compute_neuron_selection(spk, beh, region_idx_all):
    # ...
    stim_selective = visual_mask & np.isfinite(dp) & (np.abs(dp) >= DP_THRESHOLD)
    # ... reward prediction neurons ...
    selected = stim_selective | reward_pred
    if selected.sum() < MIN_NEURONS_FALLBACK:
        # fallback top-|d'| selection
```

iii. The AI explicitly cites the paper's neuron selection criteria. However, the reference solution keeps all neurons in the four visual areas (~4.1M neurons), applying no d-prime filtering for the decoder task. The AI's approach dramatically reduces the neuron count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (`StartFr`). Each trial starts at `StartFr` and ends at `GrayFr`, with only running frames retained. Trials have variable length.

ii.
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
```

iii. The alignment event matches the instructions (corridor entry / trial start). The AI notes `off_start = 0.0` and `off_end = None` in metadata, consistent with variable-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is computed from the median frame interval across sessions, yielding ~315 ms (matching the 3.17 Hz imaging rate). However, because non-running frames are dropped, the retained frames are not temporally contiguous.

ii.
```python
def compute_time_bin_ms(catalog):
    medians = []
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
        d = np.diff(ft)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            medians.append(float(np.median(d) * SECONDS_PER_DAY * 1000.0))
    return float(np.median(medians))
```

iii. The reference uses `1000.0 / FRAME_RATE` directly. The AI computes it from data, which gives the same result. However, the AI's running-only frame selection means the "bins" in a trial are not temporally contiguous -- there are gaps where the mouse was not running.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame of the sound cue for each trial) and the retained frame indices within the trial.

ii.
```python
sound_fr = np.asarray(beh["SoundFr"], dtype=int)
# ...
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The AI uses `SoundFr` and `np.searchsorted` to find the cue's position among retained frames, then multiplies by frame_dt. The reference uses actual timestamps from `ft` and `np.interp` for fractional frame timing.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue frame index is located among the retained running frames via `np.searchsorted`. The time to cue is computed as `(cue_idx - retained_idx) * frame_dt`, where `frame_dt` is the median frame interval. This gives positive values before the cue and negative after, consistent with the "time TO sound cue" convention.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The reference uses actual timestamps (`ft`) and `np.interp` to get precise cue times in seconds. The AI approximates time by counting frames and multiplying by a constant frame interval, which introduces small errors when non-running frames create gaps.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same retained running-frame indices used for the neural data, so they are aligned by construction.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
# neural = spk_sel[:, frames]
# t_to_cue = ((cue_idx - retained_idx) * frame_dt)
```

iii. Aligned by using the same frame indices for all data streams.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the recording date) in the session metadata for each session and mouse.

ii.
```python
def compute_day_offsets(catalog):
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

iii. The AI computes calendar days since the mouse's first recording date, giving values like 0, 7, 9, 16, etc.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is identified. Each session's day_of_training is the number of calendar days between that session's date and the mouse's first date. This is broadcast across all frames of each trial.

ii.
```python
first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
return {
    cand.session_id: float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
    for cand in catalog
}
```

iii. The reference counts session ordinals (0, 1, 2, ...) per mouse. The AI uses calendar day offsets, which can produce much larger values (e.g., 0, 7, 9, 16, ..., up to 92). Both represent training progression but with different scales and semantics.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the retained frame indices within the trial and the median frame interval `frame_dt`.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The AI uses frame count * constant frame interval. The reference uses actual timestamps from `ft` and `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as `retained_frame_index * frame_dt`, starting from 0. Because non-running frames are dropped, the retained frame indices are contiguous (0, 1, 2, ...) even though the original frames may have gaps.

ii.
```python
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. This is an approximation: it assumes all retained frames are equally spaced. The reference uses actual timestamps for precise timing. The AI's approach effectively measures "running time since trial start" rather than "wall-clock time since trial start."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned by construction: the retained_idx array corresponds 1:1 with the retained frame indices used for neural data.

ii.
```python
# Same frames array used for neural and time_since
neural = spk_sel[:, frames]
t_since = (retained_idx * frame_dt)
```

iii. Same alignment mechanism as all other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` in the behavior data, which marks whether each trial is in the rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The `isRew` value for each trial is cast to float32 and broadcast across all retained frames. No further processing.

ii.
```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. Same approach as the reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` in the behavior data, which gives the wall texture name for each trial.

ii.
```python
wall_name = np.asarray(beh["WallName"]).astype(str)
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses the exact `WallName` strings (e.g., `circle1`, `circle2`, `leaf1`, `leaf1_swap1`) as individual stimulus categories, producing 15 distinct categories. These are sorted alphabetically and mapped to integer indices 0-14.

ii.
```python
def collect_stimulus_values(catalog):
    names = set()
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names)

stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}
```

iii. The reference maps all wall names to 4 base texture categories (circle, leaf, rock, wood) using a TEXTURE lookup dict, as described in the instructions ("Visual stimulus category. e.g. circle, leaf, etc."). The AI keeps all 15 individual variants, significantly increasing the number of output classes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (lick frame numbers) and `LickTrind` (lick trial indices) in the behavior data.

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The AI uses both `LickFr` and `LickTrind` to get per-trial lick frames. The reference uses only `LickFr` to build a session-wide lick flag array and indexes into it per trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frames matching the trial index are extracted from `LickFr`/`LickTrind`. A binary vector is created where retained frames that match any lick frame are set to 1, others to 0.

ii.
```python
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The result should be equivalent to the reference approach for most cases, though using `np.isin` on retained frames vs. indexing a full session array could differ at edge cases.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking vector is computed for the same retained frame indices as the neural data.

ii.
```python
# frames used for both:
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
neural = spk_sel[:, frames]
```

iii. Aligned by using the same frame index set.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame in decimeters.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position in decimeters is divided by 10 (converting to meters) and floored to get the bin index (0-3), then clipped to [0, 3].

ii.
```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Same binning logic as the reference (`ft_Pos // 10, clipped to 0-3`).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four 1-meter bins: 0-1m, 1-2m, 2-3m, 3-4m, created by `floor(pos_dm / 10)` clipped to [0, 3].

ii.
```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Matches the reference and the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read at the same retained frame indices as the neural data.

ii.
```python
pos = ft_pos[frames]
neural = spk_sel[:, frames]
```

iii. Aligned by using the same frame set.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. Same source as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speeds from all retained frames across all sessions are concatenated. Global quartile edges are computed using `np.quantile` at [0, 0.25, 0.5, 0.75, 1.0]. Each frame's speed is binned using `np.searchsorted` on the inner edges.

ii.
```python
all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
# ...
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The reference computes quartiles per session using rank-based binning. The AI computes global quartiles across all sessions. This creates different bin boundaries: the reference guarantees exactly 25% per bin within each session, while the AI guarantees 25% globally but not per session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins defined by global quartile edges. `np.searchsorted` assigns each speed to a bin [0, 3].

ii.
```python
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0])
thresholds = speed_edges[1:-1]
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right")
```

iii. The reference uses rank-based per-session quartiles which handles ties (many frames at speed 0) more robustly. The AI's value-based global approach may produce uneven bin sizes within sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read at the same retained frame indices as the neural data.

ii.
```python
raw_speed = ft_speed[frames].astype(np.float32)
neural = spk_sel[:, frames]
```

iii. Aligned by using the same frame set.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behavior is cut to the number of imaged frames (`nfr = spk.shape[1]`). Lick frames with `NaN` values are filtered out via `np.isfinite(lick_fr)`. Trials with no surviving running frames are dropped. The AI also has a fallback neuron selection if too few neurons pass the d-prime threshold.

ii.
```python
nfr = spk_sel.shape[1]
# ...
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
# ...
if frames.size == 0:
    continue
# ...
if selected.sum() < MIN_NEURONS_FALLBACK:
    # fallback selection
```

iii. The AI handles several edge cases (NaN lick frames, zero-running-frame trials, low neuron counts). The reference similarly cuts behavior to imaged frames and drops empty trials.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files (~405 GB total) and computing neuron selection (d-prime calculations per session). The full conversion takes ~500 seconds.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
# ... d-prime computation in compute_neuron_selection ...
```

iii. The AI's neuron selection adds significant per-session computation compared to the reference, which only does area-based filtering.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level loop in `build_trial_arrays` iterates over all trials sequentially. The d-prime computation in `compute_neuron_selection` includes per-trial loops for reward-prediction analysis.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    # ...
```

iii. The trial loop is standard; the d-prime calculations add non-trivial computation that could potentially be vectorized.

## 12-c. What processing does the code repeat multiple times?

i. The `collect_stimulus_values` function reloads all behavior files to enumerate wall names. The `compute_time_bin_ms` function reloads behavior files again. `sample_candidate_score` also reloads behavior for scoring. These are repeated loads of the same large behavior dictionaries.

ii.
```python
def collect_stimulus_values(catalog):
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        # ...

def compute_time_bin_ms(catalog):
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
```

iii. Each of these functions independently loads behavior files. The CONVERSION_NOTES.md acknowledges this inefficiency.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The d-prime neuron selection is elaborate analysis-specific processing that the reference does not perform. The `get_reference_pair` function computes primary rewarded/non-rewarded stimuli per session. The `compute_neuron_selection` function computes stimulus selectivity d-prime, reward-prediction d-prime, and cue-frame d-prime -- all for neuron filtering that the reference solution does not do.

ii.
```python
def compute_neuron_selection(spk, beh, region_idx_all):
    # ~75 lines of d-prime computation for neuron filtering
```

iii. This processing is unnecessary relative to the reference solution's simpler area-based filtering, and adds significant computation time.
