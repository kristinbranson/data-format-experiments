# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script builds a session catalog from `data/beh/Imaging_Exp_info.npy`, groups multiple metadata entries that point to the same physical recording, keeps one canonical entry per recording, and then processes each catalog entry by loading behavior from `Beh_<exp_type>.npy`, neural data from `data/spk/*_neural_data.npy` through `utils.load_spk`, and retinotopy from `data/retinotopy/*_trans.npz`.

ii. ```python
def build_session_catalog() -> list[SessionCandidate]:
    exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
    grouped: dict[str, list[SessionCandidate]] = defaultdict(list)
    for exp_type, db_list in exp_info.items():
        for db in db_list:
            session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            beh_key = session_id
            if "stimtype" in db:
                beh_key = f"{beh_key}_{db['stimtype']}"
            grouped[session_id].append(SessionCandidate(...))
    return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]

def load_behavior(exp_type: str, beh_key: str) -> dict:
    beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    return beh[beh_key]

def process_session(...):
    beh = load_behavior(candidate.exp_type, candidate.beh_key)
    spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
    ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. `CONVERSION_NOTES.md` says the central metadata index is `Imaging_Exp_info.npy`, behavior comes from `Beh_<exp_type>.npy`, neural activity from `spk/*_neural_data.npy`, and retinotopy from `*_trans.npz`. The trajectory also states that the converter would reuse the canonical loaders from the reference code and deduplicate the session index down to the 89 unique recordings.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the mouse name `db["mname"]`. The output `subjects` list is the sorted set of unique mouse names, and `subject_idx` maps each processed session to that subject list.

ii. ```python
subjects_all = sorted({cand.db["mname"] for cand in catalog})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
...
"subjects": subjects_all,
"subject_idx": np.asarray(
    [subject_to_idx[s.subject] for s in processed_sessions], dtype=np.int64
),
```

iii. The notes explicitly say subject count should be 19 and that `mname` should be used as subject ID; Step 4 resolves the subject-count consistency check that way.

## 1-c. How are the data split into sessions?

i. Sessions are defined by the base recording key `<mouse>_<date>_<block>`. If multiple `exp_info` entries refer to that same recording, they are grouped and one canonical metadata/behavior key is kept via `choose_canonical`. Sessions are then ordered by subject, date, and block.

ii. ```python
def session_sort_key(session_id: str) -> tuple[str, datetime, int]:
    parts = session_id.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    blk = int(parts[4])
    return subject, date, blk

def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key
    return sorted(candidates, key=key_fn)[0]
```

iii. Step 5 of the notes says the same recording appears under multiple analysis labels in `Imaging_Exp_info.npy`, so the converter intentionally deduplicates to one canonical copy per base session ID. The trajectory also says the converter would 'deduplicate the session index'.

## 1-d. How are the data split into trials?

i. Within each session, the code iterates `trial_idx` from `0` to `beh["ntrials"] - 1`. Each trial is defined as the frame interval from `StartFr` (corridor entry) to `GrayFr` (entry into the gray segment), not to `EndFr`.

ii. ```python
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    frames = frames[ft_move[frames] > 0]
    if frames.size == 0:
        continue
```

iii. The notes justify this as a deliberate choice to use only the texture segment (`StartFr:GrayFr`) because the decoder position target was defined as four 1 m bins over the 0-4 m textured corridor, and because many reference analyses focus on the texture area.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. For each trial, only frames with `ft_move > 0` are kept; if that leaves zero frames, the whole trial is dropped. There is no additional explicit trial-quality filter.

ii. ```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
if frames.size == 0:
    continue
```

iii. The notes repeatedly cite the paper rule 'we only considered timepoints during running' and use that as the main trial/timepoint curation rule. They do not describe any broader generic bad-trial rejection beyond that running-only restriction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the saved imaging signal in `*_neural_data.npy`, specifically the concatenated list under `['spks']` loaded by `utils.load_spk`.

ii. ```python
spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. The notes state that the reference repository does not recompute dF/F; it directly loads the already processed deconvolved fluorescence traces from the neural-data files. The trajectory contains the original `utils.load_spk` source showing `np.load(...).item()['spks']` concatenation.

## 2-b. How is the `neural` data processed?

i. The code keeps only a curated subset of neurons, then slices the raw deconvolved traces on retained trial frames. It does not perform position interpolation, z-scoring, coding-direction normalization, or temporal rebinning before saving `neural`.

ii. ```python
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
spk_sel = spk[selected_mask]
...
neural = spk_sel[:, frames].astype(np.float32, copy=False)
```

iii. The notes frame this as a decoder-oriented simplification: preserve the paper's deconvolved traces and paper-style neuron curation, but package the trial data as framewise running-only activity aligned to corridor entry. The trajectory says the converter would 'construct the decoder inputs/outputs on running texture frames'.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by task selectivity rather than by generic imaging quality metrics. The script keeps neurons in labeled visual areas that either have running-corridor stimulus `|d'| >= 0.3` or meet an aHV reward-prediction criterion, and if too few survive it falls back to the top `|d'|` neurons until at least 64 remain.

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

iii. Step 5 of the notes explicitly says the converter would 'curate neurons using paper-defined task relevance' and describes the stimulus-selective/reward-prediction rule plus the fallback top-`|d'|` selection as a tractability measure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry: each neural trial starts at `StartFr`. The saved sequence contains only retained running frames until `GrayFr`, so the alignment is to trial start but only over a running-only texture subset of the trial.

ii. ```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
neural = spk_sel[:, frames].astype(np.float32, copy=False)
...
"temporal_alignment_event": "corridor entry (trial start / StartFr)",
```

iii. The README and metadata both describe temporal alignment as corridor entry / `StartFr`. The notes justify the running-only texture restriction as mirroring the paper's 'running only' and 'texture area' analyses.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script reports the native imaging frame interval as the time bin size by taking the median of `np.diff(ft)` across sessions, and it applies no explicit temporal rebinning.

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
...
"time_bin_size": time_bin_ms,
```

iii. The notes cite the reference notebook's imaging frame rate of about 3.17 Hz and say no temporal rebinning should be applied. They also describe the conversion as using framewise data.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. In the final code, *Time to sound cue* is derived from `SoundFr`, the kept frame indices for the trial, and a single median frame interval `frame_dt`. It is not derived from `SoundTime` and the actual frame timestamps `ft`, even though the notes say that was the intended plan.

ii. ```python
ft = np.asarray(beh["ft"][:nfr], dtype=float)
dft = np.diff(ft)
frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
...
sound_fr = np.asarray(beh["SoundFr"], dtype=int)
...
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. Step 5 of the notes says this variable should come from `SoundTime`, `ft`, and retained frames so that dropped non-running frames do not distort elapsed time. The implementation drifted from that documented rationale.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept trial, the code computes `cue_idx = searchsorted(frames, SoundFr[trial])` and then uses `(cue_idx - retained_idx) * frame_dt` to build a signed continuous vector.

ii. ```python
retained_idx = np.arange(frames.size, dtype=np.float32)
cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. The notes justify a continuous signed time-to-cue signal, but they describe using actual frame times rather than a compressed retained-frame index. The trajectory does not contain a separate rationale for the final simplification.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time-to-cue vector has one value per retained neural frame and is stacked alongside the neural trial, so it is index-aligned to the neural matrix after running-only frame selection.

ii. ```python
input_trials.append(
    {
        "time_to_sound_cue_sec": t_to_cue,
        ...
    }
)
...
input_arr = np.vstack([input_raw["time_to_sound_cue_sec"], ...])
if input_arr.shape[1] != T or output_arr.shape[1] != T:
    raise ValueError(...)
```

iii. The notes say all decoder variables should be aligned to the kept neural timepoints. The code follows that index-level alignment but not the notes' intended true-time alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's recording date `db["datexp"]` in `Imaging_Exp_info.npy`, grouped by mouse `mname`. The explicit `days` field is not used.

ii. ```python
def compute_day_offsets(catalog: list[SessionCandidate]) -> dict[str, float]:
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

iii. The notes explicitly justify this as a proxy: a complete per-session training-day label is not available for every recording, so they use elapsed calendar days since the subject's first recording instead.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest session date is treated as day 0. Each later session gets the integer number of elapsed days since that first date, and the scalar is repeated across timepoints within each trial.

ii. ```python
first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
...
day = np.full(frames.shape, session_day, dtype=np.float32)
```

iii. This processing is described directly in Step 5 of `CONVERSION_NOTES.md` as the chosen training-day proxy.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the final code, *Time since trial start* is derived from `StartFr`, the retained frame count, and the median frame interval `frame_dt`; it is not derived from the true frame timestamps `ft` despite that being the stated plan in the notes.

ii. ```python
start_fr = np.asarray(beh["StartFr"], dtype=int)
...
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The notes say this variable should use `(ft[frame] - ft[StartFr[trial]]) * 86400` so that dropped non-running frames do not collapse time. The final code instead uses retained-frame index times a constant step.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script starts each kept trial at 0 and increments by one median frame interval per retained frame, producing a monotonic continuous vector.

ii. ```python
retained_idx = np.arange(frames.size, dtype=np.float32)
t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. The notes justify a continuous time-varying trial-start variable, but again the documented rationale expected real timestamps, not compressed retained-frame indices.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The vector is aligned one-to-one with the saved neural frames after running-only filtering, so it shares the neural trial length and indexing.

ii. ```python
input_arr = np.vstack([
    input_raw["time_to_sound_cue_sec"],
    input_raw["day_of_training"],
    input_raw["time_since_trial_start_sec"],
    input_raw["reward_available"],
]).astype(np.float32)
if input_arr.shape[1] != T or output_arr.shape[1] != T:
    raise ValueError(...)
```

iii. The notes wanted all decoder inputs time-aligned to neural samples. The code satisfies the index-alignment requirement but not the true elapsed-time requirement.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial boolean `beh["isRew"]`.

ii. ```python
is_rew = np.asarray(beh["isRew"]).astype(np.float32)
...
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The notes say this input should use rewarded-corridor identity and repeat it across timepoints. They explicitly cite `isRew` for that purpose.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial scalar is converted to float and repeated for every retained frame in the trial.

ii. ```python
reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. The notes say even trial-constant decoder inputs would be stored as time-varying arrays for uniformity with the decoder format.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the trialwise corridor label `beh["WallName"]`, not from `stim_id`.

ii. ```python
wall_name = np.asarray(beh["WallName"]).astype(str)
...
stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. The notes justify using exact `WallName` strings because `stim_id` has NaNs in swap sessions and because `WallName` preserves the full texture vocabulary, including swap-specific labels and non-leaf/circle stimuli.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The converter first collects the sorted set of all wall names across the dataset, assigns each a global integer ID, and then repeats that ID across every retained frame of each trial.

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

iii. This is described in the notes as the way to preserve the full stimulus label space across all mice and sessions.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`, i.e. the neural-frame index of each lick and the trial index of that lick.

ii. ```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
lick_tr = np.asarray(beh["LickTrind"], dtype=float)
```

iii. The notes describe licking as a framewise binary output and explicitly cite `LickFr` and `LickTrind` as the direct behavioral fields for building it.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the code finds lick frames assigned to that trial, converts them to integer frame indices, and marks each retained frame as 1 if it appears in that set, else 0.

ii. ```python
lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
lick_frames_trial = lick_frames_trial.astype(int)
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The notes justify this as a binary time-varying representation of licking suitable for the decoder format.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking vector is aligned to the retained neural frames within the trial. Any licks on frames that are not in the running-only `StartFr:GrayFr` subset are omitted from the saved output.

ii. ```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
...
licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. The notes justify alignment to the retained neural frames because the converted dataset is running-only and texture-only. They do not separately justify dropping non-running-frame licks, but that follows from the chosen frame selection.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the framewise within-corridor position `ft_Pos` on the retained frames.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
...
pos = ft_pos[frames]
```

iii. The notes map position directly from `ft_Pos` and note that the raw corridor length is 60 dm with a 40 dm texture segment, which supports the requested 1 m bins over 0-4 m.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position output is just the retained-frame `ft_Pos` values followed by discretization; there is no interpolation or smoothing beyond the earlier running-frame selection.

ii. ```python
pos = ft_pos[frames]
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The notes explicitly say to use the within-corridor raw position and discretize the 0-40 dm texture segment into four equal bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The code floors `pos / 10` and clips the result to `[0, 3]`, creating bins `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40+]` in decimeters.

ii. ```python
pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The notes justify this as the exact 4-by-1 m discretization requested by the task over the 0-4 m textured corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken on the same retained frame list used for the neural data, so it is one categorical value per neural sample.

ii. ```python
frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
frames = frames[ft_move[frames] > 0]
...
pos = ft_pos[frames]
```

iii. The notes state that all time-varying outputs should be aligned to the kept neural frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the framewise running-speed variable `ft_RunSpeed`.

ii. ```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
...
raw_speed = ft_speed[frames].astype(np.float32)
```

iii. The notes map running speed directly from `ft_RunSpeed` and cite the paper's running-speed analyses as the conceptual reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. For each trial, the raw speed values on retained frames are stored temporarily. After all sessions are processed, the converter concatenates all retained running-speed samples to define global quartile edges.

ii. ```python
output_trials.append({
    ...
    "running_speed_raw": raw_speed,
})
...
all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
```

iii. The notes say the quartiles should be computed over all included running-only texture frames so that the final categories each contain roughly 25% of the data, as required by the decoder task.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The code uses the 25th, 50th, and 75th percentiles of all retained speed values as thresholds, adjusts any ties upward with `np.nextafter`, and then bins each sample with `np.searchsorted(..., side="right")`.

ii. ```python
speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
for i in range(1, len(speed_edges)):
    if speed_edges[i] <= speed_edges[i - 1]:
        speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
...
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The notes justify the quartile binning directly from the decoder specification and mention the monotonic-edge fix as a safeguard when percentiles tie.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained frames as the neural data, then converted to quartile bins without changing its frame alignment.

ii. ```python
raw_speed = ft_speed[frames].astype(np.float32)
...
output_arr = np.vstack([
    output_raw["visual_stimulus_category"],
    output_raw["licking"],
    output_raw["position_bin"],
    speed_bin,
]).astype(np.int64)
```

iii. The notes say speed should be aligned to the same neural frames used elsewhere in the converted dataset.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is mostly by omission and fallbacks rather than explicit repair. `safe_dprime` returns NaNs when comparisons are empty or degenerate; neuron-selection masks require finite values; trials with no retained frames are skipped; `get_reference_pair` falls back through several heuristics; and tied speed-quantile edges are nudged apart with `np.nextafter`. There is no substantial imputation or explicit repair of malformed cue/time annotations.

ii. ```python
def safe_dprime(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if x1.size == 0 or x2.size == 0:
        return np.full(..., np.nan, dtype=np.float32)
    ...
    denom[denom == 0] = np.nan

if frames.size == 0:
    continue

for i in range(1, len(speed_edges)):
    if speed_edges[i] <= speed_edges[i - 1]:
        speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
```

iii. The notes emphasize sanity checks and mention no-lick trials, NaNs in swap-session `stim_id`, and the need for fallback logic when identifying reference stimulus pairs. They do not document a broad missing-data-cleaning pipeline.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are loading the very large `spk` files for each session and then running `compute_neuron_selection` plus per-trial slicing over tens of thousands of neurons and hundreds of trials. The conversion log confirms per-session runtimes of several seconds, with the largest sessions taking around 10-12 seconds.

ii. ```python
for idx, cand in enumerate(catalog, start=1):
    session = process_session(...)
...
selected_mask, selection_info = compute_neuron_selection(spk, beh, region_idx_all)
...
for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    ...
```

iii. This was not spelled out as a design goal, but `conversion_full_out.txt` shows the session-by-session timings and makes it clear that session processing dominates runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial loops in `build_trial_arrays`, the per-reward-trial loop used to populate `trial_means` inside `compute_neuron_selection`, and the per-trial loop in `sample_candidate_score` that counts lick-containing trials.

ii. ```python
for trial_idx in np.flatnonzero(valid_reward_trials):
    frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
    frames = frames[running[frames] & (ft_pos[frames] >= 5.0) & (ft_pos[frames] <= TEXTURE_LENGTH_DM)]
    if frames.size:
        trial_means[:, trial_idx] = np.nanmean(spk[:, frames], axis=1)

for trial_idx in range(int(beh["ntrials"])):
    frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
    ...

for trial_idx, (s, g) in enumerate(zip(start, gray)):
    frames = np.arange(s, g, dtype=int)
    ...
```

iii. The trajectory and notes do not discuss vectorization explicitly. This is an inference from the final code structure.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly reloads behavior dictionaries from disk in separate passes for stimulus collection, time-bin estimation, sample-session ranking, and actual session processing. It also repeatedly rebuilds per-trial frame lists in different helper functions.

ii. ```python
def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)

def compute_time_bin_ms(catalog: list[SessionCandidate]) -> float:
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)

def sample_candidate_score(candidate: SessionCandidate) -> tuple[int, int]:
    beh = load_behavior(candidate.exp_type, candidate.beh_key)
```

iii. This inefficiency is not justified in the notes. The notes focus on correctness and validation rather than caching or minimizing repeated I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several intermediate values are computed only to support later binning, logging, or plotting and are then discarded from the saved dataset: raw running speed is stored and later replaced by quartile bins; `processing_info`, `example_trial`, trial-length summaries, and cue-position summaries are built only for diagnostics; and the reward-prediction statistics used during neuron selection are not preserved after the final neuron mask is made.

ii. ```python
output_trials.append({
    ...
    "running_speed_raw": raw_speed,
})
...
proc_info = {
    **selection_info,
    **trial_summary,
    "cue_positions_dm": cue_positions,
    "example_trial": example_trial,
}
...
speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. The notes and README justify these as diagnostics and convenience outputs, not as essential parts of the final dataset.
