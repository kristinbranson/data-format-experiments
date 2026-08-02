# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treated `data/beh/Imaging_Exp_info.npy` as the authoritative session list, scanned every `Beh_*.npy` file to build a canonical behavior lookup for each recording, and then loaded one neural `.npy` plus one retinotopy `.npz` per session during conversion.

ii. ```python
def load_exp_info(root):
    path = os.path.join(root, "data", "beh", "Imaging_Exp_info.npy")
    return np.load(path, allow_pickle=True).item()

def behavior_files(root):
    beh_dir = os.path.join(root, "data", "beh")
    ...
    files.append((name, np.load(full, allow_pickle=True).item()))
```

```python
spk_path = os.path.join(root, "data", "spk", f"{session['base']}_neural_data.npy")
ret_path = os.path.join(root, "data", "retinotopy", f"{session['mname']}_{session['datexp']}_trans.npz")
spk_obj = np.load(spk_path, allow_pickle=True).item()
ret = np.load(ret_path, allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the AI says it treated `Imaging_Exp_info.npy` as authoritative and canonicalized figure-specific duplicate behavior entries. In trajectory step 83, it explicitly planned to deduplicate to the paper’s 89 recordings and then convert each session from the raw behavior, spiking, and retinotopy files.

## 1-b. How are the data split into subjects?

i. Subjects are defined entirely by `mname`. The converter builds `subjects` from the unique mouse names in session order and stores one `subject_idx` per session.

ii. ```python
for session in sessions:
    if session["mname"] not in subject_to_idx:
        subject_to_idx[session["mname"]] = len(subjects)
        subjects.append(session["mname"])
subject_idx = np.array([subject_to_idx[s["mname"]] for s in sessions], dtype=np.int64)
```

iii. The notes emphasize matching the paper’s 19 mice. The trajectory also shows the AI checking that deduplicating `Imaging_Exp_info.npy` produced 19 unique `mname` values.

## 1-c. How are the data split into sessions?

i. Sessions are defined by the tuple `(mname, datexp, blk)`. Duplicate appearances of the same recording across figure-specific behavior files are merged under the same base session key.

ii. ```python
key = (item["mname"], item["datexp"], item["blk"])
...
"base": f"{item['mname']}_{item['datexp']}_{item['blk']}",
```

```python
def session_base_from_key(key):
    return "_".join(str(key).split("_")[:5])
```

iii. `CONVERSION_NOTES.md` says deduplicating by `(mouse, date, block)` gave 89 unique recordings. The trajectory shows the AI inspecting duplicate behavior entries and deciding they represented the same underlying recording with different figure-specific masking.

## 1-d. How are the data split into trials?

i. Trials are defined with `beh["ntrials"]` and `beh["ft_trInd"]`. For each trial index, the code gathers the frame indices whose frame-level trial id equals that trial.

ii. ```python
def trial_frame_indices(beh, nfr):
    ft_trind = np.asarray(beh["ft_trInd"][:nfr])
    ntrials = int(beh["ntrials"])
    out = []
    for tr in range(ntrials):
        mask = (ft_trind == tr) & is_corr & is_move
        out.append(np.flatnonzero(mask).astype(np.int64))
    return out
```

iii. The notes say “Trial identity comes from `ft_trInd`.” The trajectory shows the AI reading the notebook cell that documents `beh['ntrials']`, `beh['trInd']`, and the frame-level trial metadata.

## 1-e. How are trials filtered based on quality controls?

i. There is almost no explicit trial QC. A trial is kept if, after frame filtering, it still has at least one chunk of retained frames; otherwise it is silently skipped.

ii. ```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    if not chunks:
        continue
```

iii. The notes justify the frame filtering, not a separate trial filter. The trajectory shows the AI checked that zero trials had no retained moving corridor frames, so in practice this skip path was expected not to remove anything on the full dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the deconvolved traces in `*_neural_data.npy`, specifically `spk_obj["spks"]`, with retinotopy `iarea` used to assign or filter neurons by region.

ii. ```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
ret = np.load(ret_path, allow_pickle=True)
iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` states that the raw session files contain deconvolved Suite2p outputs and that retinotopy assignments were used to group neurons into `V1`, `mHV`, `lHV`, and `aHV`.

## 2-b. How is the `neural` data processed?

i. The AI concatenates all imaging planes, keeps only a fixed-size subset of 128 neurons per session, and averages every 3 retained imaging frames into one decoder bin.

ii. ```python
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
spk_sel = spk[selected_neurons].astype(np.float32, copy=False)
...
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
...
neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The notes say this reduction was added “for decoder tractability,” not because it came from the reference code. In trajectory step 67, the AI explicitly says a literal all-neuron conversion was too large and that it would pick a conservative neuron/time reduction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only retinotopically assigned visual-cortex neurons, maps them into four coarse regions, then picks the highest-variance neurons within those regions until it reaches 128 neurons per session.

ii. ```python
def coarse_region_indices(iarea):
    out[iarea == 8] = 0
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
    out[(iarea == 5) | (iarea == 6)] = 2
    out[(iarea == 3) | (iarea == 4)] = 3

def select_neurons(spk, region_idx_full, selected_frames, neurons_per_session):
    ...
    order = np.argsort(var[candidates])[::-1]
```

iii. `CONVERSION_NOTES.md` says the subset was restricted to retinotopically assigned visual-cortex neurons and ranked by variance over retained frames. The trajectory shows that this was introduced as an engineering workaround for runtime and memory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI describes the alignment event as corridor entry / trial start, but in code the neural bins actually begin at the first retained frame that satisfies both corridor and movement filters, then preserve order within trial.

ii. ```python
mask = (ft_trind == tr) & is_corr & is_move
...
chunks = chunk_indices(frame_idx, frames_per_bin)
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. The notes claim temporal alignment to corridor entry / trial start. The trajectory repeatedly describes the retained frames as `ft_CorrSpc && ft_move > 0`, so the practical alignment is to the filtered running frames within each trial rather than an unfiltered trial-start time axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converter estimates the imaging frame interval from `beh["ft"]`, then multiplies by `frames_per_bin=3`; the output metadata reports a bin size of about 944 ms. Yes, temporal rebinning is applied by averaging every 3 retained frames.

ii. ```python
parser.add_argument("--frames-per-bin", type=int, default=3, ...)
...
diffs = np.diff(ft) * SECONDS_PER_DAY * 1000.0
...
"time_bin_size": float(frame_dt_ms * args.frames_per_bin),
"time_binning_rule": f"Average every {args.frames_per_bin} retained imaging frames into one decoder time bin.",
```

iii. The notes explicitly say “Decoder time bins are formed by averaging every 3 retained imaging frames.” The trajectory also records the AI inspecting the notebook statement that the imaging rate was 3.17 Hz before choosing this 3-frame binning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the per-trial sound time `SoundTime` and the frame timestamps `ft`.

ii. ```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
...
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The notes list `time_to_sound_cue_s` among the continuous timing variables computed from original frame timestamps and trial times. The trajectory shows the AI reading the notebook cell that documents `beh['SoundTime']`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each decoder bin, it subtracts each frame time in that chunk from the trial’s sound-cue time, converts days to seconds, and averages across the frames in the chunk.

ii. ```python
trial_input.append(
    np.array(
        [
            np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
            ...
        ],
        dtype=np.float32,
    )
)
```

iii. The notes describe this as the mean seconds to cue within each decoder bin. The trajectory does not show a separate reference implementation for this variable; this appears to be the AI’s own time-domain construction.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the exact same frame chunk used to compute each neural bin, so it is aligned one decoder bin at a time.

ii. ```python
chunk_ft = ft[chunk]
neural_bin = spk_sel[:, chunk].mean(axis=1)
...
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The notes say the continuous timing variables were computed from the original frame timestamps and then placed into the same decoder bins used for neural activity.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session date string `datexp` stored in the session metadata from `Imaging_Exp_info.npy`.

ii. ```python
def parse_date(date_str):
    return dt.datetime.strptime(date_str, "%Y_%m_%d").date()
...
by_subject[session["mname"]].append((idx, parse_date(session["datexp"])))
```

iii. The notes say the variable is the number of calendar days since the subject’s first imaging session. The trajectory shows the AI using session dates rather than any explicit behavioral training-day field.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the code finds the earliest session date and subtracts it from every later session date, yielding a constant per-session offset in calendar days that is repeated across all bins in the session.

ii. ```python
first_date = min(date for _, date in entries)
for idx, date in entries:
    offsets[idx] = float((date - first_date).days)
...
day_value = np.float32(training_days[session_idx])
```

iii. `CONVERSION_NOTES.md` states exactly this rule. The trajectory shows the AI picked this because it needed a simple, reproducible “day of training” variable and did not find a more direct field in the packaged behavior objects.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. The AI did not create an environment-type input at all, so no raw variable is used for this field.

ii. ```python
"input_names": [
    "time_to_sound_cue_s",
    "training_day",
    "time_since_trial_start_s",
    "reward_available",
],
```

iii. The notes only discuss the four requested decoder inputs and never mention an environment-type channel. The trajectory likewise focuses on the user-specified decoder inputs rather than adding extra contextual variables such as `exptype`, `stim`, or `rewType` from `Imaging_Exp_info.npy`.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. The variable is omitted from the converted dataset.

ii. ```python
"input_names": [
    "time_to_sound_cue_s",
    "training_day",
    "time_since_trial_start_s",
    "reward_available",
],
```

iii. The omission is consistent with the AI’s stated scope in the notes: it limited the input channels to the four variables named in the decoder task.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from frame timestamps `ft` and the per-trial corridor-entry time `Trial_start_time`.

ii. ```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
...
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The notes list `time_since_trial_start_s` among the timing variables derived from original timestamps and trial times. The trajectory shows the AI reading the notebook description of `beh['Trial_start_time']`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each decoder bin, it subtracts the trial-start time from every frame time in the chunk, converts from days to seconds, and averages within the chunk.

ii. ```python
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The notes describe this as mean elapsed seconds since corridor entry within each decoder bin.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned on the same per-chunk frame groups used for the neural data, so each neural bin and each time-since-start value come from the same frames.

ii. ```python
chunk_ft = ft[chunk]
neural_bin = spk_sel[:, chunk].mean(axis=1)
...
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The notes say the timing variables were computed inside the same decoder bins used for neural activity.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `isRew`.

ii. ```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
...
float(is_rew[tr]),
```

iii. `CONVERSION_NOTES.md` says `reward_available` is constant within trial and taken from `isRew`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code casts `isRew` to a float and repeats that same value in every decoder bin of the trial.

ii. ```python
trial_input.append(
    np.array(
        [
            ...,
            float(is_rew[tr]),
        ],
        dtype=np.float32,
    )
)
```

iii. The notes justify this as a per-trial discrete context variable. No additional processing is applied.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived directly from the per-trial string label `WallName`.

ii. ```python
wall_name = np.asarray(beh["WallName"])
...
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The notes say `visual_stimulus` comes from `WallName`. The trajectory shows the AI checked the global `WallName` vocabulary and deliberately preserved the full set of categories instead of collapsing them.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code builds a global sorted list of all `WallName` values, maps each trial’s stimulus label to an integer category, and repeats that integer in every decoder bin of the trial.

ii. ```python
def output_stimulus_names(canonical_lookup, sessions):
    names = set()
    ...
    names.update(map(str, np.unique(beh["WallName"])))
    return sorted(names)
...
stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
```

iii. The notes and trajectory both say the converter records the full stimulus vocabulary present across sessions and uses per-trial `WallName` directly.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from the lick-frame array `LickFr`.

ii. ```python
def build_lick_frame_mask(beh, nfr):
    lick_fr = np.asarray(beh["LickFr"], dtype=np.float64)
    lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
```

iii. The notes say `licking` is a binary per-bin output built from `LickFr.astype(int)` mapped to frame bins.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code creates a frame-level boolean lick mask, marks any valid lick frame as `True`, and then labels a decoder bin as licking if any frame in that chunk contains a lick.

ii. ```python
lick_mask = np.zeros(nfr, dtype=bool)
...
lick_mask[lick_idx] = True
...
int(lick_mask[chunk].any()),
```

iii. The notes describe this exactly. The trajectory shows the AI chose the frame-index route because the behavior files already contain `LickFr`.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned at the same chunk level as the neural bins: the bin gets a `1` if any lick occurs in the same retained-frame chunk used to average the neurons.

ii. ```python
neural_bin = spk_sel[:, chunk].mean(axis=1)
...
int(lick_mask[chunk].any()),
```

iii. The notes say the licking output is mapped into the same decoder bins as the neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level corridor position `ft_Pos`.

ii. ```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
...
mean_pos = float(ft_pos[chunk].mean())
```

iii. The trajectory shows the AI inspected `ft_Pos` and the behavior notebook. The notes say the output is position within the 4 m textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. For each chunk it averages `ft_Pos` across the retained frames in that chunk and then discretizes the mean position.

ii. ```python
mean_pos = float(ft_pos[chunk].mean())
...
position_to_bin(mean_pos),
```

iii. The notes describe the output as 1 m corridor position bins. The code uses the chunk mean rather than preserving every original frame.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The code divides by 10, floors the result, and clips it to `0..3`, producing four bins corresponding to roughly `0-10`, `10-20`, `20-30`, and `30-40` position units.

ii. ```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. The notes interpret these as the four 1 m bins `0-1m`, `1-2m`, `2-3m`, and `3-4m`. This relies on the corridor-only frames having `ft_Pos` restricted to about `0..40`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned one decoder chunk at a time using the same filtered frame indices used for the neural averages.

ii. ```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    mean_pos = float(ft_pos[chunk].mean())
```

iii. The notes say the output variables are computed inside the same decoder bins as the neural activity.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-level running-speed trace `ft_RunSpeed`.

ii. ```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
...
mean_speed = float(ft_speed[chunk].mean())
```

iii. The notes explicitly state that the running-speed output is built from `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code averages `ft_RunSpeed` within each decoder chunk, then maps that chunk mean to a quartile-based category.

ii. ```python
mean_speed = float(ft_speed[chunk].mean())
...
speed_to_bin(mean_speed, speed_thresholds),
```

iii. The notes say the thresholds are computed globally over all retained decoder bins and then used to categorize each bin’s mean speed.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The code first gathers the mean speed of every retained decoder bin in every session, computes the 25th, 50th, and 75th percentiles globally, then assigns each chunk with `np.searchsorted(..., side="right")`.

ii. ```python
for indices in trial_frame_indices(beh, nfr):
    for chunk in chunk_indices(indices, frames_per_bin):
        speed_values.append(float(ft_speed[chunk].mean()))
q = np.quantile(speed_values, [0.25, 0.5, 0.75])
...
return int(np.searchsorted(thresholds, value, side="right"))
```

iii. `CONVERSION_NOTES.md` says the quartiles are computed globally from retained bins, not per session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned on the same chunked retained-frame bins used for neural activity.

ii. ```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    mean_speed = float(ft_speed[chunk].mean())
```

iii. The notes say the running-speed categories are computed for the same decoder bins as the neural signal.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several ad hoc safeguards: it ignores unreadable behavior files, tolerates `NaN` equality when comparing duplicate entries, truncates to `min(spk.shape[1], len(beh["ft"]))`, drops non-finite lick frames, and skips trials with no retained chunks.

ii. ```python
try:
    files.append((name, np.load(full, allow_pickle=True).item()))
except Exception:
    continue
...
nfr = min(spk.shape[1], len(beh["ft"]))
...
lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
...
if not chunks:
    continue
```

iii. The notes emphasize duplicate canonicalization and zero mismatch checks. The trajectory also shows the AI verified that duplicate behavior objects matched on the core arrays once `NaN` handling was taken into account.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive parts are scanning and loading all behavior files, loading large spiking arrays, computing variance over selected frames for neuron ranking, computing global speed thresholds over all sessions/trials/chunks, and then converting every session trial-by-trial and chunk-by-chunk.

ii. ```python
canonical_lookup, provenance, duplicate_mismatches = build_behavior_lookup(args.root)
speed_values, speed_thresholds = compute_speed_thresholds(canonical_lookup, sessions, args.frames_per_bin)
...
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
...
for session_idx, session in enumerate(sessions):
    sess_neural, sess_input, sess_output, ...
```

iii. The trajectory repeatedly discusses runtime and memory as the main engineering constraint and shows the AI measuring total decoder bins, region counts, and trial lengths before implementing its reduction strategy.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested loops over trials and chunks in `trial_frame_indices`, `compute_speed_thresholds`, and `convert_session` are obvious vectorization targets. The subject/session bookkeeping loops are also simple but low impact.

ii. ```python
for tr in range(ntrials):
    mask = (ft_trind == tr) & is_corr & is_move
...
for indices in trial_frame_indices(beh, nfr):
    for chunk in chunk_indices(indices, frames_per_bin):
...
for tr, frame_idx in enumerate(frames_by_trial):
    for chunk in chunks:
```

iii. The notes do not discuss optimization, but the trajectory makes clear that runtime pressure drove several implementation choices, so these loops are where further optimization would matter most.

## 12-c. What processing does the code repeat multiple times?

i. It recomputes per-trial retained frame lists in multiple places, reloads and scans all behavior bundles during canonicalization, and repeatedly performs chunk-wise averaging patterns for speed, neural, input, and output features.

ii. ```python
for session in sessions:
    ...
    for indices in trial_frame_indices(beh, nfr):
```

```python
frames_by_trial = trial_frame_indices(beh, nfr)
...
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
```

iii. The same retained-frame logic appears once in `compute_speed_thresholds` and again in `convert_session`. The trajectory also shows repeated inspection of duplicate behavior entries because the source behavior files are organized by figure panels rather than canonical sessions.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computed values are not used downstream: `speed_values` is returned but only the thresholds are stored, `kept_trial_indices` is collected but never used, extensive provenance metadata is saved although the decoder does not consume it, and the script also builds a sample dataset unrelated to the full-data decoder evaluation.

ii. ```python
speed_values, speed_thresholds = compute_speed_thresholds(...)
...
kept_trial_indices = []
...
kept_trial_indices.append(tr)
```

```python
"behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
...
sample_data = build_sample_dataset(...)
```

iii. The notes frame these extras as documentation and validation aids. They are useful for auditing, but they do not affect the decoder inputs or outputs consumed downstream.
