# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `data/beh/Imaging_Exp_info.npy`, deduplicates recordings by `(mname, datexp, blk)`, then eagerly loads every `Beh_*.npy` file to build a canonical behavior lookup keyed by the five-part session base string. For each converted session it separately loads the spike file from `data/spk/<session>_neural_data.npy` and the retinotopy file from `data/retinotopy/<mouse>_<date>_trans.npz`.

ii. 
```python
def load_exp_info(root):
    path = os.path.join(root, "data", "beh", "Imaging_Exp_info.npy")
    return np.load(path, allow_pickle=True).item()
```
```python
def behavior_files(root):
    beh_dir = os.path.join(root, "data", "beh")
    files = []
    for name in sorted(os.listdir(beh_dir)):
        if not name.startswith("Beh_") or not name.endswith(".npy"):
            continue
        full = os.path.join(beh_dir, name)
        files.append((name, np.load(full, allow_pickle=True).item()))
```
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
ret = np.load(ret_path, allow_pickle=True)
```

iii. In the trajectory the agent said it had confirmed the dataset was organized as per-session neural `.npy` files plus separate behavior and retinotopy metadata, and later said duplicates across figure groupings should be deduplicated to 89 unique recordings before conversion.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses `mname` as the subject identifier. It creates one subject per distinct `mname` and builds `subject_idx` from the order the unique mouse names are first encountered while iterating sessions.

ii. 
```python
sessions.append(
    {
        "mname": item["mname"],
        "datexp": item["datexp"],
        "blk": item["blk"],
        "key": key,
        "base": f"{item['mname']}_{item['datexp']}_{item['blk']}",
        "exp_item": item,
    }
)
```
```python
subjects = []
subject_to_idx = {}
for session in sessions:
    if session["mname"] not in subject_to_idx:
        subject_to_idx[session["mname"]] = len(subjects)
        subjects.append(session["mname"])
subject_idx = np.array([subject_to_idx[s["mname"]] for s in sessions], dtype=np.int64)
```

iii. The trajectory says the agent treated the 89 unique recordings as belonging to 19 mice and wanted the converted dataset to match those paper-level counts.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` recording. The AI constructs a base id string `<mname>_<datexp>_<blk>`, drops repeated appearances of the same triple in `Imaging_Exp_info.npy`, and uses that base id for the spike filename and behavior lookup.

ii. 
```python
key = (item["mname"], item["datexp"], item["blk"])
if key in seen:
    continue
seen.add(key)
sessions.append(
    {
        "mname": item["mname"],
        "datexp": item["datexp"],
        "blk": item["blk"],
        "base": f"{item['mname']}_{item['datexp']}_{item['blk']}",
    }
)
```
```python
def session_base_from_key(key):
    return "_".join(str(key).split("_")[:5])
```

iii. The trajectory explicitly says the agent checked duplicate figure-group assignments, concluded that `swap1/swap2` entries were analysis duplications rather than separate subsets, and therefore chose to deduplicate to 89 recordings.

## 1-d. How are the data split into trials?

i. The AI identifies trial membership from `ft_trInd`, but it does not keep all trial frames. Within each trial it keeps only frames where both `ft_CorrSpc` and `ft_move > 0` are true, then further divides those retained frame indices into consecutive chunks of `frames_per_bin` frames. Each chunk becomes one decoder time bin, so the converted trial is a sequence of averaged movement-only bins rather than the full per-frame trial.

ii. 
```python
def trial_frame_indices(beh, nfr):
    ft_trind = np.asarray(beh["ft_trInd"][:nfr])
    is_corr = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    is_move = np.asarray(beh["ft_move"][:nfr], dtype=np.float64) > 0
    ntrials = int(beh["ntrials"])
    out = []
    for tr in range(ntrials):
        mask = (ft_trind == tr) & is_corr & is_move
        out.append(np.flatnonzero(mask).astype(np.int64))
    return out
```
```python
def chunk_indices(indices, chunk_size):
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
```

iii. The trajectory says the agent believed the paper used a “running-only” restriction and planned to enforce that by frame selection, then later said it would apply “3-frame temporal binning” for tractability.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not implement the reference trial-length outlier filter. Trials are effectively retained if they still contain at least one retained movement/corridor chunk; trials with zero retained chunks are skipped. There is no explicit filtering of abnormally long trials.

ii. 
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    if not chunks:
        continue
```
```python
selected_frames = np.concatenate(frames_by_trial)
if selected_frames.size == 0:
    raise RuntimeError(f"No running corridor frames found for session {session['base']}")
```

iii. In the trajectory the agent focused on a movement-only frame filter and on making the dataset tractable, but did not mention reproducing the reference percentile-based long-trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural activity from `spks` inside the per-session neural `.npy` file, concatenating the planes along the neuron axis. It derives region labels from `iarea` in the retinotopy `.npz` file.

ii. 
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. The trajectory says the agent verified that the raw neural files already contained concatenated deconvolved traces and that retinotopy carried the neuron-to-region labels.

## 2-b. How is the `neural` data processed?

i. The AI first filters frames to movement-only corridor frames, then selects a fixed-size subset of neurons, and finally averages neural activity over consecutive 3-frame chunks. The result is stored as `float32`, one `n_neurons x n_bins` matrix per trial.

ii. 
```python
neural_bin = spk_sel[:, chunk].mean(axis=1)
trial_neural.append(neural_bin)
```
```python
neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The trajectory repeatedly states the reason for this processing: the agent concluded that a literal all-neuron, frame-by-frame conversion would be too large to pickle and too heavy for decoder training, so it added temporal reduction for tractability.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only neurons assigned to the four coarse visual regions `V1`, `mHV`, `lHV`, and `aHV`, then keeps at most 128 neurons per session. The 128 are chosen proportionally across those regions and ranked by variance over retained frames.

ii. 
```python
def coarse_region_indices(iarea):
    out = np.full(iarea.shape[0], -1, dtype=np.int64)
    out[iarea == 8] = 0
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
    out[(iarea == 5) | (iarea == 6)] = 2
    out[(iarea == 3) | (iarea == 4)] = 3
    return out
```
```python
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
spk_sel = spk[selected_neurons].astype(np.float32, copy=False)
brain_region_idx = region_idx_full[selected_neurons].astype(np.int64)
```

iii. The trajectory says the agent wanted a “fixed-size visual-cortex neuron subset per session” and later optimized the variance-ranking step because that new neuron-selection pass was a runtime hotspot.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats corridor entry / trial start as the conceptual alignment event, but in practice it stores only the movement-only corridor frames assigned to each trial and then averages them in order. It does not preserve a full contiguous frame-by-frame window from corridor entry onward.

ii. 
```python
"temporal_alignment_event": "corridor entry / trial start",
```
```python
frames_by_trial = trial_frame_indices(beh, nfr)
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
```

iii. The trajectory says the agent wanted the dataset “aligned to corridor entry” while also enforcing a running-only restriction and temporal compression, so alignment is defined by trial membership rather than by keeping every frame after entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses bins of `frames_per_bin` retained imaging frames. With the default of 3, the metadata records a time bin size equal to `median_frame_dt_ms * 3`, so the effective bin size is about 0.95 s rather than a single imaging frame. Yes, explicit temporal rebinning is applied by averaging consecutive retained frames.

ii. 
```python
parser.add_argument("--frames-per-bin", type=int, default=3, help="Number of retained imaging frames to average per decoder time bin.")
```
```python
"time_bin_size": float(frame_dt_ms * args.frames_per_bin),
"time_binning_rule": f"Average every {args.frames_per_bin} retained imaging frames into one decoder time bin.",
```

iii. The trajectory explicitly says the agent chose “3-frame temporal binning” as a tractability measure and later reported the full dataset in “decoder bins” rather than imaging frames.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this input from `SoundTime` for each trial and the frame timestamps `ft` for each retained imaging frame. It does not use `SoundFr`.

ii. 
```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
```

iii. The trajectory shows the agent inspecting both frame-alignment fields and absolute-time fields and then choosing a direct time-based computation once it committed to chunked bins.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained chunk, the AI computes the mean of `(SoundTime[trial] - ft[chunk]) * SECONDS_PER_DAY`. This yields a continuous value in seconds, positive before the cue and negative after, but averaged across the frames inside each decoder bin.

ii. 
```python
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The trajectory justifies this indirectly through the same tractability decision: once the AI chose 3-frame decoder bins, it computed bin-level summaries rather than per-frame values.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the exact same retained frame chunk used to produce each neural bin. The cue-time value is computed from `chunk_ft`, so each neural column and cue-time column refer to the same decoder bin.

ii. 
```python
chunk_ft = ft[chunk]
neural_bin = spk_sel[:, chunk].mean(axis=1)
...
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The trajectory consistently says the converter uses the paper’s frame-level alignment fields and then bins all streams together into shared decoder bins.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives day-of-training from each session’s `datexp` string, grouped by `mname`. It parses the date strings into calendar dates.

ii. 
```python
def parse_date(date_str):
    return dt.datetime.strptime(date_str, "%Y_%m_%d").date()
```
```python
for idx, session in enumerate(sessions):
    by_subject[session["mname"]].append((idx, parse_date(session["datexp"])))
```

iii. The trajectory says the agent wanted a per-mouse training-day covariate and used the session metadata already present in `Imaging_Exp_info.npy`.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the number of calendar days since the first imaging session for that mouse, as a float, and then repeats that value in every decoder bin of the trial.

ii. 
```python
first_date = min(date for _, date in entries)
for idx, date in entries:
    offsets[idx] = float((date - first_date).days)
```
```python
day_value = np.float32(training_days[session_idx])
...
np.array(
    [
        np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
        day_value,
        np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
        float(is_rew[tr]),
    ],
    dtype=np.float32,
)
```

iii. The trajectory does not mention reproducing the reference ordinal session-count convention; it presents this variable as literal elapsed days from the first session.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives this input from `Trial_start_time` and the frame timestamps `ft`. It does not use `StartFr`.

ii. 
```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
```

iii. The trajectory indicates the agent preferred direct absolute-time fields after deciding to operate on chunked bins rather than on single frames.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained chunk, the AI computes the mean of `(ft[chunk] - Trial_start_time[trial]) * SECONDS_PER_DAY`. This is a continuous time-since-start value in seconds, averaged over the frames in the decoder bin.

ii. 
```python
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The trajectory again ties this to shared time-bin summaries: once the data were rebinned, trial-start timing was also summarized per decoder bin.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by being computed from the same `chunk_ft` used for the neural average in each decoder bin.

ii. 
```python
chunk_ft = ft[chunk]
neural_bin = spk_sel[:, chunk].mean(axis=1)
...
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The trajectory says all streams were converted with the same retained frame chunks, so this variable is aligned to the neural bins by construction.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The AI derives reward availability directly from `isRew`.

ii. 
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
```

iii. The trajectory treated rewarded versus unrewarded sessions as an important axis when selecting a representative sample subset, which is consistent with taking the value straight from `isRew`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No nontrivial processing is applied. The boolean `isRew[trial]` is cast to a float and repeated in every decoder bin of the trial.

ii. 
```python
float(is_rew[tr])
```

iii. The trajectory does not describe any transformation beyond carrying the reward flag into the converted input matrix.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives the stimulus output from `WallName`.

ii. 
```python
wall_name = np.asarray(beh["WallName"])
...
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The trajectory says the agent checked the global stimulus vocabulary and wanted to preserve the actual `WallName` values it found in the behavior files.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI does not collapse wall variants to four base texture classes. Instead it collects the sorted unique `WallName` strings across the dataset, assigns each unique name an index, and then repeats that index across every decoder bin of the trial.

ii. 
```python
def output_stimulus_names(canonical_lookup, sessions):
    names = set()
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        names.update(map(str, np.unique(beh["WallName"])))
    return sorted(names)
```
```python
stim_names = output_stimulus_names(canonical_lookup, sessions)
stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
...
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The trajectory explicitly says the agent was checking whether it should “preserve whatever the actual per-trial `WallName` values are without accidentally collapsing categories,” and the final message reports that it kept the per-trial `WallName` vocabulary.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The AI derives licking from `LickFr`.

ii. 
```python
def build_lick_frame_mask(beh, nfr):
    lick_mask = np.zeros(nfr, dtype=bool)
    lick_fr = np.asarray(beh["LickFr"], dtype=np.float64)
    lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
    lick_mask[lick_idx] = True
    return lick_mask
```

iii. The trajectory notes that behavior files already contain lick frame information aligned to imaging frames, so the AI used that field directly.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI converts lick frame indices into a boolean per-frame mask, filters non-finite or out-of-range indices, and then labels a decoder bin as `1` if any retained frame in that chunk contains a lick.

ii. 
```python
lick_mask = build_lick_frame_mask(beh, nfr)
...
int(lick_mask[chunk].any())
```

iii. The trajectory’s stated reason is again the binning scheme: once the data were represented as 3-frame decoder bins, the licking target was reduced to bin-level presence or absence.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The AI aligns licking with neural activity by evaluating the lick mask on the same retained frame chunk used to compute the neural average for that decoder bin.

ii. 
```python
neural_bin = spk_sel[:, chunk].mean(axis=1)
...
int(lick_mask[chunk].any())
```

iii. The trajectory says all streams were co-binned together after frame selection, so lick labels and neural activity share the same binning and trial segmentation.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI derives position from `ft_Pos`.

ii. 
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. The trajectory lists `ft_Pos` among the core frame-aligned behavior fields the agent recovered from the behavior files.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. For each decoder bin the AI averages `ft_Pos` across the retained frames in the chunk, then converts that mean position into a 1 m corridor bin.

ii. 
```python
mean_pos = float(ft_pos[chunk].mean())
...
position_to_bin(mean_pos)
```

iii. The trajectory ties this to the same 3-frame temporal reduction used for the rest of the decoder variables.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The AI divides the position value by 10 decimeters, floors it, and clips the result to the range `0..3`, corresponding to four 1 m bins.

ii. 
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. The trajectory does not give a separate justification beyond matching the task’s request for four equal-length 1 m position bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned with the neural data by using the same retained frame chunk as the neural average for each decoder bin.

ii. 
```python
neural_bin = spk_sel[:, chunk].mean(axis=1)
mean_pos = float(ft_pos[chunk].mean())
trial_output.append(
    np.array(
        [
            stim_idx,
            int(lick_mask[chunk].any()),
            position_to_bin(mean_pos),
            speed_to_bin(mean_speed, speed_thresholds),
        ],
        dtype=np.int64,
    )
)
```

iii. The trajectory says the converter uses shared decoder bins for all streams after applying the movement/corridor frame filter.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The AI derives running speed from `ft_RunSpeed`.

ii. 
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
```

iii. The trajectory lists running speed among the frame-aligned behavior variables inspected from the source files.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes the mean speed of each retained decoder bin, then computes global quartile thresholds over all such bins across all sessions, and finally maps each bin’s mean speed to one of four categories using those global thresholds.

ii. 
```python
def compute_speed_thresholds(canonical_lookup, sessions, frames_per_bin):
    speed_values = []
    for session in sessions:
        beh = canonical_lookup[session["base"]][2]
        nfr = len(beh["ft"])
        ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
        for indices in trial_frame_indices(beh, nfr):
            for chunk in chunk_indices(indices, frames_per_bin):
                speed_values.append(float(ft_speed[chunk].mean()))
    speed_values = np.asarray(speed_values, dtype=np.float32)
    q = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return speed_values, q.astype(np.float32)
```
```python
mean_speed = float(ft_speed[chunk].mean())
...
speed_to_bin(mean_speed, speed_thresholds)
```

iii. The trajectory says the converter would “compute global running-speed quartiles” and use those thresholds after the movement-only filter and temporal binning.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The AI thresholds speed with `np.searchsorted` against the three global quartile cutoffs produced by `compute_speed_thresholds`, yielding category indices `0..3`.

ii. 
```python
def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))
```

iii. The trajectory justifies this as producing decoder-ready quartile bins over the retained dataset, rather than per-session rank quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running-speed labels are aligned with neural data by being computed from the same retained frame chunk used for each neural bin.

ii. 
```python
neural_bin = spk_sel[:, chunk].mean(axis=1)
mean_speed = float(ft_speed[chunk].mean())
...
speed_to_bin(mean_speed, speed_thresholds)
```

iii. The trajectory repeatedly states that all decoder variables were built on the same chunked frame windows.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavior-derived frame streams to `nfr = min(spk.shape[1], len(beh["ft"]))`, filters non-finite and out-of-range lick frames, checks for missing behavior entries, and records duplicate behavior mismatches. Trials with no retained chunks are silently skipped. It does not implement any imputation.

ii. 
```python
nfr = min(spk.shape[1], len(beh["ft"]))
```
```python
lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
```
```python
missing = [s["base"] for s in sessions if s["base"] not in canonical_lookup]
if missing:
    raise RuntimeError(f"Missing behavior entries for sessions: {missing[:5]}")
```

iii. The trajectory shows the agent spending time checking duplicate behavior entries and later describing the dataset as mostly clean, while still adding guardrails for mismatches, missing entries, and invalid lick indices.

## 12-a. What are the most time-consuming steps of the code?

i. In this AI-written code, the expensive steps are the extra full-dataset passes it introduced: loading all behavior files to compute global speed thresholds, loading each large spike file, and computing variance-based neuron ranking for selection. The actual per-trial conversion is nested inside those passes.

ii. 
```python
speed_values, speed_thresholds = compute_speed_thresholds(canonical_lookup, sessions, args.frames_per_bin)
```
```python
var = variance_over_columns(spk, selected_frames)
```
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```

iii. The trajectory explicitly calls out two runtime bottlenecks created by the AI’s design: the full-data reduction pass needed for global speed quartiles and the neuron-ranking step, which it later patched to use fewer frames.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code has several Python-level nested loops that could be vectorized or reorganized: trial-by-trial construction of frame masks, session/trial/chunk loops for global speed thresholds, and trial/chunk loops when building every converted session.

ii. 
```python
for tr in range(ntrials):
    mask = (ft_trind == tr) & is_corr & is_move
    out.append(np.flatnonzero(mask).astype(np.int64))
```
```python
for session in sessions:
    ...
    for indices in trial_frame_indices(beh, nfr):
        for chunk in chunk_indices(indices, frames_per_bin):
            speed_values.append(float(ft_speed[chunk].mean()))
```
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    ...
    for chunk in chunks:
        ...
```

iii. The trajectory shows the agent accepting these loops initially, then later optimizing only one hotspot after observing that the full conversion was compute-bound.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats trial/chunk segmentation logic in multiple places. `trial_frame_indices` is run once during the global speed-threshold pass and again during every session conversion, and `chunk_indices` is re-applied each time those trial frames are revisited. It also performs a separate pass over all behavior files up front to build canonical provenance information.

ii. 
```python
speed_values, speed_thresholds = compute_speed_thresholds(canonical_lookup, sessions, args.frames_per_bin)
...
frames_by_trial = trial_frame_indices(beh, nfr)
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
```

iii. The trajectory confirms this duplication: first a dataset-wide pass was added to compute global quartiles, and later each session was processed again to perform the actual conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does extra work that is not needed by the downstream decoder on the full dataset: it computes duplicate-provenance bookkeeping, builds a representative `sample_data.pkl`, computes summary statistics for printing, stores large provenance metadata, and returns `speed_values` from `compute_speed_thresholds` even though only the thresholds are used later.

ii. 
```python
canonical_lookup, provenance, duplicate_mismatches = build_behavior_lookup(args.root)
```
```python
speed_values, speed_thresholds = compute_speed_thresholds(canonical_lookup, sessions, args.frames_per_bin)
```
```python
sample_data = build_sample_dataset(
    full_data=data,
    nsessions=args.sample_sessions,
    max_trials_per_session=args.sample_trials_per_session,
)
```
```python
"behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
```

iii. The trajectory makes clear that much of this was added for validation, sample-training convenience, and reporting, not because the full converted dataset required it.
