# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent builds a session catalog from `data/beh/Imaging_Exp_info.npy`, deduplicates repeated entries down to one canonical record per base session ID, and then loads behavior, spike data, and retinotopy separately for each session. Trials are then constructed inside `process_session()` by iterating through `beh["ntrials"]`.

ii. ```python
exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
...
for exp_type, db_list in exp_info.items():
    for db in db_list:
        session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        beh_key = session_id
        if "stimtype" in db:
            beh_key = f"{beh_key}_{db['stimtype']}"
...
def load_behavior(exp_type: str, beh_key: str) -> dict:
    beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    return beh[beh_key]
...
beh = load_behavior(candidate.exp_type, candidate.beh_key)
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the agent says the central index is `Imaging_Exp_info.npy`, that the same recording appears under multiple analysis labels, and that the conversion should use all 89 unique imaging sessions with matched behavior, neural, and retinotopy files.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the mouse name `db["mname"]`. The output `subjects` list is the sorted unique mouse IDs, and `subject_idx` maps each processed session to one of those IDs.

ii. ```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
...
"subjects": subjects_all,
"subject_idx": np.asarray(
    [subject_to_idx[s.subject] for s in processed_sessions], dtype=np.int64
),
```

iii. The notes explicitly say to use `mname` as the subject ID because the paper reports 19 mice and the metadata show 19 unique mouse names.

## 1-c. How are the data split into sessions?

i. Sessions are split by the base key `<mouse>_<date>_<block>`. If multiple metadata entries point to the same recording, the agent keeps one canonical candidate, preferring an entry without `stimtype` when available.

ii. ```python
def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key

    return sorted(candidates, key=key_fn)[0]
...
return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]
```

iii. The notes justify this as deduplication of repeated analysis labels down to the 89 unique imaging recordings reported in the paper.

## 1-d. How are the data split into trials?

i. Each session uses the trial boundaries stored in behavior arrays. For each trial, the code takes frames from `StartFr[trial]` up to `GrayFr[trial]`, then keeps only the running frames inside that interval.

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

iii. The notes say the agent intentionally used the texture segment from corridor entry to gray-space entry because the decoder asks for four 1 m position bins and the reference analyses focus on the 0 to 4 m texture corridor.

## 1-e. How are trials filtered based on quality controls?

i. The only explicit trial-level filter is dropping trials whose `StartFr:GrayFr` interval contains no running frames after applying `ft_move > 0`. There is no additional rejection by lick count, cue validity, or reward status.

ii. ```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
```

iii. The notes justify this with the paper’s repeated statement that analyses use running-only timepoints. The agent treated this as a frame-level inclusion rule rather than a richer trial-quality screen.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the precomputed spike-like activity loaded by `utils.load_spk()` from `*_neural_data.npy`, specifically the concatenated `['spks']` arrays.

ii. ```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
...
def load_spk(db, root=''):
    fn = '%s_%s_%s_neural_data.npy'%(db['mname'],db['datexp'],db['blk'])
    spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']],0)
    return spk
```

iii. In the notes, the agent states that the reference repository already uses deconvolved fluorescence traces from the saved neural files and does not recompute dF/F inside this codebase.

## 2-b. How is the `neural` data processed?

i. The code keeps the selected neurons, slices their framewise activity into per-trial matrices, restricts each trial to running frames between `StartFr` and `GrayFr`, and stores the surviving frames at native frame resolution as `float32`. It does not spatially interpolate, z-score, or append previous-trial gray bins.

ii. ```python
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
spk_sel = spk[selected_mask]
...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
neural_session.append(neural_trial.astype(np.float32, copy=False))
```

iii. The notes say the conversion should stay framewise and trial-start aligned for the decoder, while borrowing only the paper’s running-only rule and neuron-curation ideas rather than the full spatial interpolation pipeline used in some paper analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered with a custom paper-inspired curation rule. The code keeps neurons in retinotopically labeled visual regions that are either stimulus-selective (`|d'| >= 0.3` on running corridor frames for a rewarded vs non-rewarded reference pair) or reward-prediction neurons in aHV. If fewer than 64 survive, it falls back to the top absolute-d' visual neurons.

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

if selected.sum() < MIN_NEURONS_FALLBACK:
    ...
    selected[order[:nkeep]] = True
```

iii. The notes justify this as an attempt to keep the conversion computationally tractable while matching the paper’s main neuron-selection motifs: stimulus selectivity, reward-prediction neurons, and visual-area restriction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent treats corridor entry / `StartFr` as the nominal alignment event, but the stored trial matrices begin at the first running frame within `StartFr:GrayFr`, not necessarily at the exact `StartFr` frame. Neural timepoints therefore stay synchronized to the retained running frames rather than to a padded trial-start axis.

ii. ```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
...
"temporal_alignment_event": "corridor entry (trial start / StartFr)",
"off_start": 0.0,
```

iii. The notes say the goal was trial-start alignment based on corridor entry, but also that the paper uses running-only timepoints. The code resolves that tension by dropping non-running frames instead of keeping a full trial-start time grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native imaging frame interval estimated from the median difference of `beh['ft']`, converted to milliseconds. No temporal rebinning is applied.

ii. ```python
def compute_time_bin_ms(catalog: list[SessionCandidate]) -> float:
    medians = []
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
        d = np.diff(ft)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            medians.append(float(np.median(d) * SECONDS_PER_DAY * 1000.0))
...
"time_bin_size": time_bin_ms,
```

iii. The notes cite the notebook’s approximately 3.17 Hz imaging rate and explicitly say the conversion should keep the native framewise resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. In the code, `time_to_sound_cue_sec` is derived from the trial’s `SoundFr` together with the retained per-trial running-frame indices and the estimated native frame duration `frame_dt`.

ii. ```python
sound_fr = np.asarray(beh["SoundFr"], dtype=int)
...
retained_idx = np.arange(frames.size, dtype=np.float32)
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The notes describe this input more aspirationally as coming from frame timing relative to the cue, but the implemented code actually uses cue frame index plus a scalar frame duration.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the code computes the index distance from that frame to the cue frame, multiplies by `frame_dt`, and stores the result as a continuous signed time series that is positive before the cue and negative after it.

ii. ```python
retained_idx = np.arange(frames.size, dtype=np.float32)
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The notes say the intention was to preserve elapsed time after dropping non-running frames, but the implementation uses retained-frame counts rather than the actual `ft` timestamps.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same retained frame set as the neural matrix, so each time-to-cue sample corresponds one-to-one with a stored neural time bin.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
...
input_arr = np.vstack([
    input_raw["time_to_sound_cue_sec"],
    ...
]).astype(np.float32)
if input_arr.shape[1] != T or output_arr.shape[1] != T:
    raise ValueError(...)
```

iii. The agent’s justification is that all decoder inputs should share the neural time axis. The code enforces that by constructing every input from the same retained `frames` array.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session’s recording date `db['datexp']` and mouse name `db['mname']`, not from a dedicated training-day field.

ii. ```python
def compute_day_offsets(catalog: list[SessionCandidate]) -> dict[str, float]:
    by_subject: dict[str, list[datetime]] = defaultdict(list)
    ...
    for cand in catalog:
        date = datetime.strptime(cand.db["datexp"], "%Y_%m_%d")
        by_subject[cand.db["mname"]].append(date)
...
        cand.session_id: float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
```

iii. The notes explicitly justify this as a proxy because a complete per-session training-day label was not available for every session, while recording dates were always available.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the first recording date is treated as day 0. Each session gets the integer day difference from that mouse-specific first date, and that scalar is repeated across all retained time bins in every trial from the session.

ii. ```python
day = np.full(frames.shape, session_day, dtype=np.float32)
...
input_raw["day_of_training"],
```

iii. The notes call this a “day-since-first-recording” proxy for day of training.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. It is not derived at all. The agent did not include any environment-type input in the converted dataset.

ii. ```python
"input_names": [
    "time_to_sound_cue_sec",
    "day_of_training",
    "time_since_trial_start_sec",
    "reward_available",
],
```

iii. The notes’ mapping plan and the final dataset both follow the decoder task in the instructions, which asks for four inputs and does not include environment type.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. No processing is performed because the variable is omitted.

ii. ```python
"input_names": [
    "time_to_sound_cue_sec",
    "day_of_training",
    "time_since_trial_start_sec",
    "reward_available",
],
```

iii. The omission is consistent with the notes, which never map an environment-type field into the final decoder inputs.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the code, it is derived from the retained running-frame index within each trial plus the estimated frame duration `frame_dt`, not from `ft - ft[StartFr]`.

ii. ```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The notes describe the intended variable as elapsed time from trial start, but the implementation measures elapsed retained-frame count after dropping non-running frames.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first retained running frame is assigned 0 s, the next retained frame is `frame_dt`, and so on. This produces a monotonically increasing time series on the retained frame axis.

ii. ```python
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The justification in the notes is that the decoder inputs should be time-varying and aligned to the neural frames, but the code does not use the original frame timestamps when non-running frames have been removed.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned one-to-one to the retained neural frames for each trial, because both arrays are built from the same `frames` vector.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The agent’s stated design was to keep all inputs on the same trial-by-time axis as neural activity.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the trial-level behavior variable `isRew`.

ii. ```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
...
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The notes justify this by saying the reference analyses also use rewarded vs non-rewarded corridor identity and that `isRew` is available for every session.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The scalar trial label is broadcast across all retained time bins in the trial as a binary 0/1 vector.

ii. ```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The notes say trial-constant contextual variables were intentionally repeated across time so every input trial has a uniform `(d_input, T)` layout.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the trial-level texture label `WallName`.

ii. ```python
wall_name = np.asarray(beh["WallName"]).astype(str)
...
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. The notes say `WallName` was chosen over `stim_id` so swap sessions and non-leaf/circle stimulus families would remain distinguishable.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code first collects the full global set of wall names across sessions, assigns each name an integer category, and then repeats the trial’s category ID across all retained time bins.

ii. ```python
def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    names = set()
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names)
...
stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}
```

iii. The notes explicitly justify preserving the exact `WallName` strings rather than collapsing categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from the lick-frame array `LickFr` together with trial indices `LickTrind`.

ii. ```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
...
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
```

iii. The notes say the output should be a binary time-varying lick trace aligned to neural bins.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the code marks a retained neural frame as 1 if its frame index appears in the trial’s lick-frame list, otherwise 0.

ii. ```python
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The notes justify this as the simplest frame-aligned binary licking output consistent with the decoder spec.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned to the same retained `frames` array used to slice the neural data, so the binary lick vector is timepoint-for-timepoint aligned with the stored neural matrix.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The notes repeatedly emphasize that all inputs and outputs should be generated on the same retained frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the framewise position variable `ft_Pos` on the retained frames.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
...
pos = ft_pos[frames]
```

iii. The notes map corridor position directly from framewise behavior to the decoder output because the instructions ask for a time-varying discretized position signal.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code reads `ft_Pos` at each retained frame and then converts it to a coarse integer spatial bin.

ii. ```python
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The notes justify ending trials at `GrayFr` so the retained position samples naturally live inside the 0 to 4 m textured corridor required by the decoder task.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position in decimeters is divided by 10 and floored, then clipped to the four categories `0,1,2,3`, corresponding to four equal 1 m bins.

ii. ```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
...
"output_values": [
    stimulus_values,
    ["no_lick", "lick"],
    ["0-1m", "1-2m", "2-3m", "3-4m"],
```

iii. The notes explicitly say this was chosen to match the decoder specification of four equal-length 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position bins are sampled on the same retained running frames as the neural activity.

ii. ```python
neural = spk_sel[:, frames].astype(np.float32, copy=False)
pos = ft_pos[frames]
```

iii. The notes state that framewise outputs should be constructed from the same retained frame axis as `neural`.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the framewise running-speed variable `ft_RunSpeed`.

ii. ```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
...
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. The notes say the paper represents running speed framewise and by position, and the decoder task specifically asks for a discretized speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Each trial stores raw retained-frame speeds first. After all sessions are processed, the code concatenates all retained speed samples, computes global quartile edges, and then bins each trial’s speeds using those thresholds.

ii. ```python
all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
...
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The notes justify this directly from the decoder instructions, which ask for four running-speed bins each containing 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the global quartiles of all retained running-speed samples across the converted dataset. Ties are repaired by nudging non-increasing edges upward with `np.nextafter`.

ii. ```python
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
for i in range(1, len(speed_edges)):
    if speed_edges[i] <= speed_edges[i - 1]:
        speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
```

iii. The notes say this choice was required by the user’s decoder specification rather than by a paper analysis.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The running-speed samples are taken from the same retained frame indices as the neural data before quartile binning, so the final discrete speed output remains one-to-one with neural time bins.

ii. ```python
raw_speed = ft_speed[frames].astype(np.float32)
...
if input_arr.shape[1] != T or output_arr.shape[1] != T:
    raise ValueError(...)
```

iii. The notes describe speed as another framewise output that should share the neural trial axis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses a series of fallbacks rather than a single missing-data policy. `safe_dprime()` propagates empty comparisons as NaNs; `get_reference_pair()` falls back through several ways of identifying the rewarded/non-rewarded reference stimuli; invalid cue positions are masked with `np.isfinite`; empty running trials are skipped; non-finite lick frames are ignored; and degenerate speed quartiles are repaired numerically.

ii. ```python
def safe_dprime(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if x1.size == 0 or x2.size == 0:
        return np.full((x1.shape[0] if x1.ndim == 2 else x2.shape[0],), np.nan, dtype=np.float32)
...
cue_pos = np.mod(np.asarray(beh["SoundDelPos"], dtype=float), float(beh["Corridor_Length"]))
valid_reward_trials = reward_trials & np.isfinite(cue_pos)
...
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
...
if speed_edges[i] <= speed_edges[i - 1]:
    speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
```

iii. The notes call out no-lick sessions, `stim_id` ambiguities in swap sessions, and cue-position quirks, and argue that preserving raw trial labels plus fallbacks is safer than aggressive data dropping.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading very large `spk` files session by session, computing neuron-selection statistics across many neurons and frames, and iterating through every trial to build per-trial arrays. The conversion log shows per-session runtimes of several seconds and about 500 seconds total.

ii. ```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
...
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
...
for trial_idx in range(int(beh["ntrials"])):
    ...
```

iii. The notes and run log both emphasize that the spike files dominate storage and runtime, and the agent explicitly designed neuron curation to keep the conversion computationally tractable.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_trial_arrays()` could be partly vectorized or batched; the `for trial_idx in np.flatnonzero(valid_reward_trials)` loop in `compute_neuron_selection()` computes per-trial reward statistics one trial at a time; and metadata passes like `collect_stimulus_values()` repeatedly reload behavior files.

ii. ```python
for trial_idx in range(int(beh["ntrials"])):
    ...
...
for trial_idx in np.flatnonzero(valid_reward_trials):
    frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
    ...
...
for cand in catalog:
    beh = load_behavior(cand.exp_type, cand.beh_key)
    names.update(...)
```

iii. The notes identify RAM and runtime pressure around the large neural matrices and repeated behavior-file access.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded multiple times for the same sessions during catalog construction, stimulus collection, sample scoring, time-bin estimation, and actual processing. The code also recomputes frame-level trial masks separately inside neuron selection and trial building.

ii. ```python
def load_behavior(exp_type: str, beh_key: str) -> dict:
    beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    return beh[beh_key]
...
stimulus_values = collect_stimulus_values(full_catalog)
time_bin_ms = compute_time_bin_ms(full_catalog)
...
beh = load_behavior(candidate.exp_type, candidate.beh_key)
```

iii. The notes describe many exploratory passes over the same metadata and behavior structures before final conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores intermediate curation diagnostics such as `selection_info`, trial-length summaries, cue-position summaries, and optional example-trial plotting payloads, but those are not kept in the final pickle. It also carries raw running speed through `output_raw` only to replace it with quartile bins in `finalize_io()`.

ii. ```python
proc_info = {
    **selection_info,
    **trial_summary,
    "cue_positions_dm": cue_positions,
    "example_trial": example_trial,
}
...
output_trials.append(
    {
        ...
        "running_speed_raw": raw_speed,
    }
)
...
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The notes frame these extras as sanity-check and validation aids rather than downstream model inputs.
