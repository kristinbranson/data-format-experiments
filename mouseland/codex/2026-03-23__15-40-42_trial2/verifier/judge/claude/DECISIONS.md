# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories: `beh/` (behavior), `spk/` (neural data), and `retinotopy/` (visual area assignments). `Imaging_Exp_info.npy` is loaded first as a master index. For each session, the behavior is loaded from `Beh_<exp_type>.npy`, neural data from `<session_id>_neural_data.npy` via `utils.load_spk()`, and retinotopy from `<mouse>_<date>_trans.npz`. The AI imports the reference `utils.py` module and uses `utils.load_spk()` directly.

ii.
```python
exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. The AI documented that it follows the reference code's loading approach, using the same `Imaging_Exp_info.npy` index and the same file naming conventions.

## 1-b. How are the data split into subjects?

i. The AI extracts subjects from `mname` in the experiment info entries. Subjects are sorted and assigned indices. 89 sessions across 19 mice are identified.

ii.
```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
```

iii. The AI noted that `mname` is the subject identifier, consistent with the paper's description of "89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is defined by `<mname>_<datexp>_<blk>`. Duplicate sessions appearing under multiple experiment types are deduplicated by choosing a "canonical" entry (preferring non-stimtype entries). This yields 89 unique sessions.

ii.
```python
session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
# deduplication via choose_canonical()
def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key
    return sorted(candidates, key=key_fn)[0]
```

iii. The AI documented that the same recording appears under multiple analysis labels and uses one canonical copy per base session ID.

## 1-d. How are the data split into trials?

i. The AI splits trials using `StartFr` (corridor entry) to `GrayFr` (grey space entry), then keeps only running frames (`ft_move > 0`). Each trial yields a variable number of running frames within the texture corridor segment. Trials with no running frames are dropped.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    if frames.size == 0:
        continue
```

iii. The AI justified keeping only running frames by citing the paper: "We only considered timepoints during running for analysis." It uses `StartFr` to `GrayFr` to match the 0-4m texture area.

## 1-e. How are trials filtered based on quality controls?

i. Trials with no running frames within the texture corridor are dropped. No other trial-level quality filtering is applied.

ii.
```python
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
```

iii. The AI documented that non-running periods are excluded per the paper's rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files, loaded via `utils.load_spk()` which concatenates planes. The visual area comes from `iarea` in the retinotopy files.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
_, region_idx_all = area_labels_from_iarea(ret["iarea"])
```

iii. The AI noted the data is already deconvolved calcium traces.

## 2-b. How is the `neural` data processed?

i. Neural data is sliced to running-only frames within the texture corridor (StartFr to GrayFr, ft_move > 0). The resulting variable-length trials are stored as float32 arrays. No padding or truncation to a fixed length is applied -- trials have variable numbers of timepoints.

ii.
```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
neural_trials.append(neural)
```

iii. The AI justified variable-length trials by noting that the running-only frame selection produces trials of different lengths.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies a complex neuron selection procedure: (1) neurons must be in visual cortex areas (V1, mHV, lHV, aHV), (2) stimulus-selective neurons with |d'| >= 0.3 between primary rewarded and non-rewarded corridors, (3) reward-prediction neurons with additional criteria in aHV, (4) a fallback selecting top-|d'| neurons if too few pass. The brain_regions list includes "other" but no "other" neurons survive selection.

ii.
```python
stim_selective = visual_mask & np.isfinite(dp) & (np.abs(dp) >= DP_THRESHOLD)
# ... reward prediction neurons ...
selected = stim_selective | reward_pred
if selected.sum() < MIN_NEURONS_FALLBACK:
    # fallback top-|d'| selection
```

iii. The AI documented following the paper's neuron curation motifs: stimulus selectivity and reward prediction. It iterated on this during Step 10 to better match the reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials start at corridor entry (`StartFr`). The AI uses running frames from `StartFr` to `GrayFr`, producing variable-length trials aligned to corridor entry. There is no fixed trial length -- trials vary in number of timepoints.

ii.
```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
```

iii. The AI set `off_start = 0.0` and `off_end = None` to indicate variable trial lengths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The AI computes the time bin size from the median frame interval across all sessions, yielding approximately 315 ms (1000/3.17 Hz).

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

iii. The AI documented that imaging frames are the native temporal resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame of the sound cue for each trial), and the retained running frame indices.

ii.
```python
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The AI converts the cue frame to a position in the retained-frame index and computes time difference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes the cue position as an index into the retained running frames using `np.searchsorted`, then multiplies the frame offset by the median frame interval (`frame_dt`). The sign convention is positive before the cue (time TO cue) and negative after.

ii.
```python
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The AI uses frame_dt (the median inter-frame interval in seconds) as a uniform time step for computing the time-to-cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same retained running-frame indices as the neural data.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. By construction, all inputs/outputs use the same `frames` array as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the recording dates in `datexp` fields of the experiment info entries.

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

iii. The AI chose calendar days since first recording as the training-day proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes calendar day offsets: for each subject, it finds the earliest recording date, then computes the number of days between each session's date and that earliest date. This is broadcast across all timepoints of the trial.

ii.
```python
float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
day = np.full(frames.shape, session_day, dtype=np.float32)
```

iii. The AI noted that a complete per-session training-day label is not available for all sessions, so it uses calendar date differences.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the retained running frame indices and the median frame interval.

ii.
```python
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The AI uses retained frame index multiplied by frame_dt.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as `retained_frame_index * frame_dt`, where `frame_dt` is the median inter-frame interval. This starts at 0 for the first retained running frame and increments uniformly. Non-running frames are excluded, so the time values reflect contiguous running frame counts rather than actual elapsed wall-clock time from corridor entry.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The AI noted it uses actual frame times rather than contiguous indices, but the code actually uses `retained_idx` (0, 1, 2, ...) * frame_dt, which is just a running-frame count scaled by a constant.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Uses the same retained running-frame indices as the neural data.

ii.
```python
retained_idx = np.arange(frames.size, dtype=np.float32)
```

iii. All streams are aligned by using the same `frames` array.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` in the behavior data, which marks rewarded corridor trials.

ii.
```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. Directly from the per-trial reward flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial binary value is broadcast across all retained frames of the trial. No further processing.

ii.
```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. N/A

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` in the behavior data, which names the texture on corridor walls for each trial.

ii.
```python
wall_name = np.asarray(beh["WallName"]).astype(str)
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. The AI uses `WallName` directly to avoid ambiguity from `stim_id` NaNs in swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects ALL unique `WallName` strings across all sessions (15 total: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5), sorts them alphabetically, and assigns each a unique integer index. This produces 15 stimulus categories rather than the 4 broad categories (circle, leaf, rock, wood) used by the reference.

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

iii. The AI justified preserving individual wall names to avoid ambiguity and preserve swap-session identities.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (lick frame numbers) and `LickTrind` (lick trial indices).

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The AI uses both `LickFr` and `LickTrind` to identify licks belonging to each trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frames are filtered to those matching the trial index. A binary per-frame indicator is created: 1 if the retained frame is in the lick frame set, 0 otherwise. Only 2 output values (no_lick, lick).

ii.
```python
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. Binary licking output from lick frame events.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is computed on the same retained running-frame indices as neural data.

ii.
```python
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. All streams share the same `frames` array.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the framewise position in the corridor (in decimeters).

ii.
```python
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Directly from the framewise position variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 (converting to meters) and floored to produce 4 bins: [0-1m), [1-2m), [2-3m), [3-4m]. Values are clipped to [0, 3].

ii.
```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Matches the instruction's request for 4 equal-length 1m spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `floor(pos_dm / 10)` produces integer bins 0-3 for positions 0-40 dm (0-4m). Values outside this range are clipped.

ii.
```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Same as above.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses `ft_Pos` at the same retained running-frame indices as neural data.

ii.
```python
pos = ft_pos[frames]
```

iii. All streams share the same `frames` array.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the framewise running speed.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. Directly from the framewise running speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Raw running speeds from all retained running frames across ALL sessions are concatenated. Global quartile edges are computed from this pooled distribution. Each frame's speed is then binned using `np.searchsorted` on the quartile thresholds.

ii.
```python
all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The AI uses global quartiles across all sessions rather than per-session quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed from all running frames across all sessions. Bins are assigned using `np.searchsorted` with `side="right"`, producing 4 bins each containing ~25% of all data globally.

ii.
```python
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right")
```

iii. The AI documented quartile binning consistent with the decoder spec.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses `ft_RunSpeed` at the same retained running-frame indices as neural data.

ii.
```python
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. All streams share the same `frames` array.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Trials with no running frames are dropped. Sessions with neurons failing d-prime computation get fallback selection. The AI handles edge cases in the neuron selection fallback (MIN_NEURONS_FALLBACK=64). Lick frames are filtered by trial index and finiteness. The AI includes "other" brain region for neurons outside visual areas, though none survive neuron selection.

ii.
```python
if frames.size == 0:
    continue
if selected.sum() < MIN_NEURONS_FALLBACK:
    # fallback selection
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
```

iii. The AI documented handling of edge cases in neuron selection and frame filtering.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (405 GB total) and computing d-prime neuron selection for each session. The d-prime computation involves computing stimulus responses across all frames, which is an additional per-session computation not present in the reference.

ii.
```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
```

iii. The AI documented that conversion completed in ~501 seconds for all 89 sessions.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_trial_arrays` iterates over all trials sequentially, constructing arrays one at a time. The lick matching (`np.isin`) per trial could potentially be vectorized with a single pass.

ii.
```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    # ... per-trial processing ...
```

iii. N/A

## 12-c. What processing does the code repeat multiple times?

i. The AI loads behavior files multiple times: once in `collect_stimulus_values()` to find all wall names, once in `compute_time_bin_ms()` to compute frame intervals, once in `sample_candidate_score()` when selecting sample sessions, and once during actual session processing. This is documented as a known inefficiency.

ii.
```python
def collect_stimulus_values(catalog):
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        ...
def compute_time_bin_ms(catalog):
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], ...)
```

iii. The AI noted this in CONVERSION_NOTES.md under "Code inefficiencies identified."

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The d-prime neuron selection (`compute_neuron_selection`) is a significant computation that performs stimulus-pair detection, d-prime calculation, and reward-prediction analysis -- all of which are analysis-specific processing from the paper that is not required for the data conversion task. The reference solution simply keeps all neurons in visual areas. Additionally, the `get_reference_pair` function has elaborate fallback logic for determining stimulus pairs that adds complexity.

ii.
```python
def compute_neuron_selection(spk, beh, region_idx_all):
    # ~80 lines of d-prime computation, stimulus pair detection, reward prediction analysis
    ...
```

iii. The AI justified this as following "paper-defined task relevance" for neuron curation, but the instructions only ask to match reference processing, which does not include analysis-specific neuron filtering beyond visual area assignment.
