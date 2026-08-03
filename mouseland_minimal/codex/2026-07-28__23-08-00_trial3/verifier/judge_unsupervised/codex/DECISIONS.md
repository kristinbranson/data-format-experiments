# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master session table from `Imaging_Exp_info.npy`, scans every `Beh_*.npy` file under `data/beh`, and then, for each deduplicated session, loads one behavior object, one neural file from `data/spk`, and one retinotopy file from `data/retinotopy`.

ii.
```python
def load_exp_info(root):
    path = os.path.join(root, "data", "beh", "Imaging_Exp_info.npy")
    return np.load(path, allow_pickle=True).item()

def behavior_files(root):
    beh_dir = os.path.join(root, "data", "beh")
    files = []
    for name in sorted(os.listdir(beh_dir)):
        if not name.startswith("Beh_") or not name.endswith(".npy"):
            continue
        full = os.path.join(beh_dir, name)
        try:
            files.append((name, np.load(full, allow_pickle=True).item()))
        except Exception:
            continue
    return files
```

```python
spk_path = os.path.join(root, "data", "spk", f"{session['base']}_neural_data.npy")
ret_path = os.path.join(root, "data", "retinotopy", f"{session['mname']}_{session['datexp']}_trans.npz")

spk_obj = np.load(spk_path, allow_pickle=True).item()
ret = np.load(ret_path, allow_pickle=True)
```

iii. The justification in `CONVERSION_NOTES.md` says the agent treated `Imaging_Exp_info.npy` as the authoritative recording list and regarded the `Beh_*.npy` files as figure-specific views over those recordings. The trajectory shows it first inspected `Imaging_Exp_info.npy`, then the behavior bundles, then the per-session neural and retinotopy files before writing the converter.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique mouse names (`mname`) from the deduplicated session list. The converter builds `subjects` and `subject_idx` directly from that field.

ii.
```python
subjects = []
subject_to_idx = {}
for session in sessions:
    if session["mname"] not in subject_to_idx:
        subject_to_idx[session["mname"]] = len(subjects)
        subjects.append(session["mname"])
subject_idx = np.array([subject_to_idx[s["mname"]] for s in sessions], dtype=np.int64)
```

iii. The notes say the converted dataset preserves the paper’s 19 mice exactly. The trajectory also shows the agent checked that the deduplicated sessions implied 19 unique subjects before proceeding.

## 1-c. How are the data split into sessions?

i. Sessions are deduplicated by `(mname, datexp, blk)`. The agent ignores the figure-group keys in `Imaging_Exp_info.npy` and the `stimtype` suffixes when deciding the canonical session identity.

ii.
```python
def collect_unique_sessions(exp_info):
    sessions = []
    seen = set()
    for entries in exp_info.values():
        for item in entries:
            key = (item["mname"], item["datexp"], item["blk"])
            if key in seen:
                continue
            seen.add(key)
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
    return sessions
```

iii. The notes explicitly say “Deduplicating by `(mouse, date, block)` yields 89 unique recordings.” In the trajectory, the agent inspected duplicate entries and concluded that most were literal reuse of the same recording across figure groupings.

## 1-d. How are the data split into trials?

i. Trials are defined by `beh["ntrials"]` together with the frame-level trial index array `ft_trInd`. For each trial number, the converter collects the frame indices whose `ft_trInd` equals that trial id, after frame filtering.

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

iii. The notes say “Trial identity comes from `ft_trInd`.” In the trajectory, the agent first considered `StartFr`/`EndFr`, then settled on using the already frame-aligned trial fields in the behavior bundles.

## 1-e. How are trials filtered based on quality controls?

i. The converter does not drop trials using an explicit trial-level quality metric. Instead, it filters frames within each trial to those with `ft_CorrSpc == True` and `ft_move > 0`, then drops a trial only if no frames remain after chunking.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    if not chunks:
        continue
    ...
    neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. `CONVERSION_NOTES.md` says this was meant to match the paper’s “running-only” analyses while also excluding gray-space frames. The trajectory shows the agent explicitly decided to keep only `ft_CorrSpc && ft_move > 0` time points.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-session deconvolved Suite2p traces in `spk_obj["spks"]`, concatenated across imaging planes.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```

iii. The notes say the raw neural source is the deconvolved Suite2p output and that raw session files contain very large neuron counts. The trajectory confirms the agent inspected the shapes and value ranges of those arrays before deciding to subset them.

## 2-b. How is the `neural` data processed?

i. The converter concatenates planes, selects a fixed subset of 128 neurons per session, then averages every 3 retained frames within each trial into one decoder bin. Each trial is saved as `neurons × time_bins`.

ii.
```python
selected_neurons = select_neurons(spk, region_idx_full, selected_frames, neurons_per_session)
spk_sel = spk[selected_neurons].astype(np.float32, copy=False)
...
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    trial_neural.append(neural_bin)
...
neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. The notes justify this as a tractability compromise: the full all-neuron recordings were too large for the required pickle and decoder, so the agent imposed a fixed-size reduction and 3-frame temporal averaging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered to retinotopically assigned visual-cortex neurons only, then ranked by variance over retained frames and subsampled to 128 per session with proportional allocation across four coarse areas.

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
var = variance_over_columns(spk, selected_frames)
...
order = np.argsort(var[candidates])[::-1]
picked = candidates[order[:target]]
```

iii. The notes say the goal was to keep a decoder-manageable subset while preserving representation across `V1`, `mHV`, `lHV`, and `aHV`. The trajectory shows the agent recovered the same coarse area mapping from the paper’s `neu_area_ID` helper before inventing the subsampling rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The saved neural trials are aligned to corridor entry / trial start only in the sense that trial membership comes from `ft_trInd`; within each trial, the retained running-corridor frames are kept in temporal order and grouped into bins.

ii.
```python
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
    ...
    for chunk in chunks:
        neural_bin = spk_sel[:, chunk].mean(axis=1)
```

iii. The notes explicitly label the temporal alignment event as “corridor entry / trial start.” The trajectory shows the agent decided the decoder should be time-aligned rather than use the paper code’s interpolated position representation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converter estimates the median imaging frame interval from `beh["ft"]` and multiplies it by `frames_per_bin=3`. In practice the saved time bin is about 0.95 s, produced by averaging 3 imaging frames.

ii.
```python
def compute_frame_dt_ms(canonical_lookup, sessions):
    ...
    diffs = np.diff(ft) * SECONDS_PER_DAY * 1000.0
    ...
    return float(np.median(dts))
```

```python
"time_bin_size": float(frame_dt_ms * args.frames_per_bin),
"time_binning_rule": f"Average every {args.frames_per_bin} retained imaging frames into one decoder time bin.",
```

iii. The notes say “Decoder time bins are formed by averaging every 3 retained imaging frames.” The trajectory also shows the agent measured the imaging interval at roughly 3.17 Hz before choosing that binning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and the per-frame timestamps `ft`.

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
...
np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The notes say `time_to_sound_cue_s` is computed from original frame timestamps and trial times. The trajectory shows the agent chose the time-domain fields rather than the frame-index fields such as `SoundFr`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each decoder bin, the converter computes the mean of `(SoundTime - ft)` across the frames in that bin and converts day units to seconds.

ii.
```python
trial_input.append(
    np.array(
        [
            np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY),
            day_value,
            np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
            float(is_rew[tr]),
        ],
        dtype=np.float32,
    )
)
```

iii. `CONVERSION_NOTES.md` describes this as “mean seconds to cue within each decoder bin.” The rationale was to make the variable continuous and time-varying inside the same bins as the neural data.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the exact same frame chunk that defines each neural bin; one scalar is stored per neural time bin.

ii.
```python
for chunk in chunks:
    chunk_ft = ft[chunk]
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    ...
    np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY)
```

iii. The notes say the continuous timing variables are computed “within each decoder bin,” and the trajectory shows the agent’s design goal was shared binning across neural, inputs, and outputs.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session’s `datexp` string and grouped by subject `mname`.

ii.
```python
def compute_training_days(sessions):
    by_subject = defaultdict(list)
    for idx, session in enumerate(sessions):
        by_subject[session["mname"]].append((idx, parse_date(session["datexp"])))
```

iii. The notes say `training_day` is “calendar days since the subject’s first imaging session.” The trajectory shows this was an invented continuous proxy rather than something directly provided by the paper code.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The converter parses each `datexp`, finds the first date recorded for that mouse, subtracts it from the current session date, and stores that scalar on every time bin in the session.

ii.
```python
first_date = min(date for _, date in entries)
for idx, date in entries:
    offsets[idx] = float((date - first_date).days)
```

```python
day_value = np.float32(training_days[session_idx])
```

iii. The notes explicitly justify it as a calendar-day measure. No stronger justification than practicality appears in the trajectory.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time` and `ft`.

ii.
```python
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
...
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The notes say `time_since_trial_start_s` is one of the continuous timing variables computed from original frame timestamps and trial times.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each neural bin, the code averages `(ft - Trial_start_time)` across the frames in that bin and converts the result to seconds.

ii.
```python
np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The trajectory indicates the agent wanted a time-domain representation explicitly aligned to the trial-start event requested by the decoder instructions.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned one-to-one with the neural bins because both are computed from the same frame chunk.

ii.
```python
for chunk in chunks:
    chunk_ft = ft[chunk]
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    ...
    np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY)
```

iii. The notes describe the time variables as being computed “within each decoder bin,” which is the stated alignment logic.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean array `isRew`.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
...
float(is_rew[tr])
```

iii. The notes say `reward_available` is “constant within trial, taken from `isRew`.”

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code casts `isRew[tr]` to a float and repeats that same value in every bin of the trial.

ii.
```python
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

iii. The notes justify this as a per-trial discrete context variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial stimulus label array `WallName`.

ii.
```python
wall_name = np.asarray(beh["WallName"])
...
stim_idx = stimulus_to_idx[str(wall_name[tr])]
```

iii. The notes say `visual_stimulus` is taken directly from `WallName`. The trajectory shows the agent also inspected `stim_id`, but chose the string labels as the canonical decoder output.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The converter builds a global sorted stimulus vocabulary across sessions, maps each trial’s `WallName` to an integer id, and repeats that category on every time bin of the trial.

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
stim_idx = stimulus_to_idx[str(wall_name[tr])]
...
trial_output.append(
    np.array(
        [
            stim_idx,
            ...
        ],
        dtype=np.int64,
    )
)
```

iii. The notes call this the “full stimulus vocabulary present across sessions.” The trajectory shows this was chosen partly to avoid the masked / cohort-specific `stim_id` conventions in the behavior bundles.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from the frame-index array `LickFr`.

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

iii. The notes say `licking` is a binary per-bin output using `LickFr.astype(int)` mapped to frame bins.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code converts all finite `LickFr` values to integer frame indices, marks those frames in a boolean mask, and labels a decoder bin as licking if any marked frame falls inside that chunk.

ii.
```python
lick_mask = build_lick_frame_mask(beh, nfr)
...
int(lick_mask[chunk].any())
```

iii. `CONVERSION_NOTES.md` states this exact rule. The trajectory does not show a deeper paper-based justification; it is a practical choice to make licking binary in the same bins as the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by intersecting the lick-frame mask with each neural chunk; each neural bin gets one binary lick label.

ii.
```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    ...
    int(lick_mask[chunk].any())
```

iii. The notes say the lick variable is “mapped to frame bins,” meaning the same chunking used for neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the per-frame position array `ft_Pos`.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
...
mean_pos = float(ft_pos[chunk].mean())
```

iii. The notes say the output is a 4-bin version of “position in corridor.” The trajectory shows the agent checked that `Texture_Length` is 40 and corridor-frame `ft_Pos` stays in `[0, 40)`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. For each decoder bin, the code averages `ft_Pos` across the retained frames in that chunk and then discretizes the result.

ii.
```python
mean_pos = float(ft_pos[chunk].mean())
...
position_to_bin(mean_pos)
```

iii. The notes say position is based on the 4 m textured corridor. The trajectory shows the agent verified that `ft_CorrSpc` frames indeed lie in the textured 0–40 range.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The converter divides position by 10, floors it, and clips to `[0, 3]`, giving four 10-unit bins corresponding to 1 m each.

ii.
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```

iii. The notes describe the resulting labels as `0-1m`, `1-2m`, `2-3m`, `3-4m`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned one value per neural chunk, using the same retained frames that were averaged into the neural bin.

ii.
```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    mean_pos = float(ft_pos[chunk].mean())
```

iii. The notes say the outputs are produced “per decoder bin,” which is the alignment rule used everywhere in the converter.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
...
mean_speed = float(ft_speed[chunk].mean())
```

iii. The notes identify running-speed quartiles as being computed from `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code first computes global quartile thresholds from the mean speed of every retained chunk in every session. Then, during conversion, it computes each chunk’s mean speed and maps it into those quartiles.

ii.
```python
def compute_speed_thresholds(canonical_lookup, sessions, frames_per_bin):
    speed_values = []
    ...
            for chunk in chunk_indices(indices, frames_per_bin):
                speed_values.append(float(ft_speed[chunk].mean()))
    speed_values = np.asarray(speed_values, dtype=np.float32)
    q = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return speed_values, q.astype(np.float32)
```

iii. The notes justify this as computing quartiles “globally over all retained decoder bins.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. A chunk’s mean speed is assigned using `np.searchsorted` against the three global quartile thresholds.

ii.
```python
def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))
```

iii. The notes define the output labels as `speed_q1` through `speed_q4`.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is aligned one category per neural chunk, using the same retained frames used to build the neural bin.

ii.
```python
for chunk in chunks:
    neural_bin = spk_sel[:, chunk].mean(axis=1)
    mean_speed = float(ft_speed[chunk].mean())
```

iii. The notes say the quartiles are computed from retained decoder bins and then applied at that same bin level.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles small inconsistencies defensively rather than with explicit curation. It skips unreadable behavior files, truncates session length to `min(spk.shape[1], len(beh["ft"]))`, ignores NaNs in lick frames, bounds-checks lick indices, and skips trials with zero retained chunks. It raises a hard error only for missing canonical behavior entries, sessions with no retained running-corridor frames, or sessions with no selected visual-cortex neurons.

ii.
```python
try:
    files.append((name, np.load(full, allow_pickle=True).item()))
except Exception:
    continue
```

```python
nfr = min(spk.shape[1], len(beh["ft"]))
...
lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
...
if not chunks:
    continue
```

iii. The notes frame this as canonicalization and sanity checking rather than as formal QC. The trajectory shows the agent spent time confirming that duplicate behavior entries matched before relying on that simplification.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are loading multi-gigabyte neural files, concatenating the full plane lists into one huge matrix, computing variance on retained frames for neuron ranking, and making a full-dataset pass to compute speed quartiles before the main conversion pass.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```

```python
var = variance_over_columns(spk, selected_frames)
...
speed_values, speed_thresholds = compute_speed_thresholds(canonical_lookup, sessions, args.frames_per_bin)
```

iii. The trajectory explicitly records the agent discovering that single session files were multiple gigabytes and that object-array `.npy` files could not be memmapped, which drove many later tractability choices.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-building loop is heavily Python-level: `trial_frame_indices` loops over every trial, `compute_speed_thresholds` loops over every trial and chunk again, and `convert_session` loops over every trial and chunk to compute all inputs and outputs. These are natural candidates for vectorization or reuse of precomputed indices.

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
    ...
    for chunk in chunks:
        ...
```

iii. The trajectory shows the agent recognized I/O and memory constraints, but the final implementation still leaves most per-trial work in Python loops.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly reconstructs trial-frame lists from behavior arrays, repeatedly chunks those lists, and scans the behavior corpus more than once for canonicalization, threshold estimation, and conversion. It also computes and stores sample-data selection metadata in a separate pass.

ii.
```python
for session in sessions:
    beh = canonical_lookup[session["base"]][2]
    ...
    for indices in trial_frame_indices(beh, nfr):
        for chunk in chunk_indices(indices, frames_per_bin):
            ...
```

```python
frames_by_trial = trial_frame_indices(beh, nfr)
...
for tr, frame_idx in enumerate(frames_by_trial):
    chunks = chunk_indices(frame_idx, frames_per_bin)
```

iii. The trajectory shows the agent first wrote a simpler version and then kept layering extra passes for validation and sample-dataset construction.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes `speed_values` but only retains the thresholds, builds `kept_trial_indices` but never uses it, and stores extensive provenance / sample-selection metadata that the decoder never consumes. It also fully materializes the full neural matrix before throwing almost all neurons away.

ii.
```python
speed_values, speed_thresholds = compute_speed_thresholds(...)
```

```python
kept_trial_indices = []
...
kept_trial_indices.append(tr)
```

```python
"behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
```

iii. The trajectory shows these additions came from the agent’s desire to document and sanity-check the conversion rather than from the decoder’s data requirements.
