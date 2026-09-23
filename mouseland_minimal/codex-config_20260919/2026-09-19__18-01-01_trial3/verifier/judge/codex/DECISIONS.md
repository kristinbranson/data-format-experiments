# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `beh/Imaging_Exp_info.npy`, builds a unique physical-session inventory keyed by `(mname, datexp, blk)`, loads each behavior file at most once, stores a reduced behavior dict per physical session, and later loads retinotopy and spike data per session. Trials are then generated from `behavior["ntrials"]` after the session-level interpolation step.

ii.
```python
exp_info = np.load(
    data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True
).item()
```
```python
for exp_type, rows in exp_info.items():
    for row in rows:
        pid = physical_id(row)
        if pid not in references:
            references[pid] = []
            sessions.append(pid)
        references[pid].append((exp_type, row))
```
```python
path = data_root / "beh" / f"Beh_{exp_type}.npy"
raw = np.load(path, allow_pickle=True).item()
...
behaviors[pid] = extract_behavior(raw[key])
```
```python
iarea = np.load(retino_path, allow_pickle=True)["iarea"]
raw_neural = np.load(spk_path, allow_pickle=True).item()
planes = raw_neural["spks"]
```

iii. In the trajectory, the agent said it would work from the 89 imaging recordings plus behavior and retinotopy files (step 8), concluded there were 89 unique recordings reused across 142 metadata references (step 16), and justified loading each behavior file once while keeping each physical recording once.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The output `subjects` list is created in first-seen session order, and `subject_idx` is built by looking up each session's mouse in that list.

ii.
```python
def physical_id(row: dict) -> tuple[str, str, str]:
    return row["mname"], row["datexp"], str(row["blk"])
```
```python
subjects: list[str] = []
...
if mouse not in subjects:
    subjects.append(mouse)
subject_idx.append(subjects.index(mouse))
```

iii. In step 16, the agent explicitly summarized the dataset as 89 unique recordings from 19 mice and said it would represent each physical recording once.

## 1-c. How are the data split into sessions?

i. A session is one physical imaging recording identified by `(mname, datexp, blk)`. Duplicate analysis references are merged so each physical recording appears once.

ii.
```python
def physical_id(row: dict) -> tuple[str, str, str]:
    return row["mname"], row["datexp"], str(row["blk"])
```
```python
if pid not in references:
    references[pid] = []
    sessions.append(pid)
references[pid].append((exp_type, row))
```
```python
if len(sessions) != 89:
    raise ValueError(f"Expected 89 unique imaging recordings, found {len(sessions)}")
```

iii. The agent's stated justification was that the metadata contains 142 references but only 89 unique physical recordings, so duplicated paper-analysis references should be represented once (steps 16, 43, 53).

## 1-d. How are the data split into trials?

i. Trials are split by `behavior["ntrials"]`, not by framewise trial windows. For each trial, the neural and covariate data are resampled onto a fixed 40-bin grid representing the first 4 m of the corridor, with 0.1 m samples per trial.

ii.
```python
targets = (
    np.arange(behavior["ntrials"], dtype=np.float64)[:, None] * FULL_TRIAL_BINS
    + np.arange(CORRIDOR_BINS, dtype=np.float64)[None, :]
).ravel()
```
```python
cube = np.empty(
    (behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32
)
```
```python
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```

iii. In step 20, the agent said it had settled on the paper's running-only interpolation onto 0.1 m bins, retaining the first 4 m (40 samples) of each trial and excluding the gray interval.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not perform the reference solution's empty-trial or long-trial filtering. It keeps all `ntrials` trials for each session and only raises hard errors if the frame streams are inconsistent.

ii.
```python
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
...
session_input, session_output = trial_covariates(behavior, day, speed_edges)
```
```python
if len(behavior["ft_move"]) < nframes:
    raise ValueError("Behavior frame stream is shorter than neural activity")
```

iii. The trajectory repeatedly emphasized "all 89 recordings, all trials, and all visual-cortex neurons" with no arbitrary trial subsampling (steps 23, 26), which is the clearest justification the agent gave for not filtering trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived from `spks` in each session's neural file and `iarea` in the corresponding retinotopy file.

ii.
```python
raw_neural = np.load(spk_path, allow_pickle=True).item()
planes = raw_neural["spks"]
```
```python
iarea = np.load(retino_path, allow_pickle=True)["iarea"]
keep, regions = area_indices(iarea)
```

iii. The agent justified this as preserving the released Suite2p deconvolved activity and the paper's visual-area annotations rather than inventing an extra neural preprocessing step (steps 11, 16, 23).

## 2-b. How is the `neural` data processed?

i. The AI concatenates neurons across planes conceptually through `keep`, filters to visual cortex, then linearly interpolates activity over cumulative VR position using only moving frames. The result is a fixed `(n_neurons, 40)` float32 trial matrix for each trial.

ii.
```python
moving = behavior["ft_move"][:nframes] > 0
x = np.asarray(behavior["ft_PosCum"][:nframes][moving], dtype=np.float64)
```
```python
values = y[:, lo] * (1.0 - weight) + y[:, hi] * weight
cube[:, dest_offset : dest_offset + count, :] = values.reshape(
    count, behavior["ntrials"], CORRIDOR_BINS
).transpose(1, 0, 2)
```

iii. In steps 20, 23, and 26, the agent said it intentionally matched the paper helper by using running-only linear interpolation onto 0.1 m bins and numerically checked the vectorized implementation against the reference helper to float32 precision.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by retinotopic area only. The AI keeps neurons whose `iarea` maps to `V1`, `mHV`, `lHV`, or `aHV`, and drops others.

ii.
```python
region = np.full(len(iarea), -1, dtype=np.int8)
region[iarea == 8] = 0
region[np.isin(iarea, [0, 1, 2, 9])] = 1
region[np.isin(iarea, [5, 6])] = 2
region[np.isin(iarea, [3, 4])] = 3
keep = region >= 0
```

iii. The agent said it would preserve all released Suite2p-selected neurons but restrict to visually assigned cortex according to the paper helper, with no extra neuron subsampling (steps 16, 23, 88).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to corridor entry and stores each trial as the first 40 spatial bins of the textured corridor. Every trial therefore starts at corridor entry and has the same length after interpolation.

ii.
```python
targets = (
    np.arange(behavior["ntrials"], dtype=np.float64)[:, None] * FULL_TRIAL_BINS
    + np.arange(CORRIDOR_BINS, dtype=np.float64)[None, :]
).ravel()
```
```python
"temporal_alignment_event": "entry into the 4 m textured corridor (trial start)",
"off_start": 0.0,
"off_end": CORRIDOR_BINS / VR_SPEED_DM_S,
```

iii. Step 20 gives the agent's main justification: corridor-entry alignment plus the first 4 m of the corridor was, in its view, the cleanest way to match the paper processing while satisfying the decoder task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI treats each 0.1 m spatial sample as a time bin. Using a fixed VR speed of 6 decimeters/s, it records a bin size of `1000 / 6 = 166.67 ms`. Temporal/spatial rebinning is applied through linear interpolation from imaging frames to these fixed bins.

ii.
```python
VR_SPEED_DM_S = 6.0
CORRIDOR_BINS = 40
FULL_TRIAL_BINS = 60
```
```python
"time_bin_size": 1000.0 / VR_SPEED_DM_S,
```
```python
"resampling": (
    "Linear interpolation over cumulative VR position using only ft_move>0 "
    "frames, matching code/utils.py spk_pos_interp; first 40 of the "
    "paper's 60 0.1-m bins retained (4-m corridor, gray space excluded)."
),
```

iii. In step 20, the agent explicitly justified the 40 x 0.1 m representation by converting the paper's fixed VR speed into uniform 166.67 ms samples.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this input from `SoundDelPos` together with the synthetic per-trial position grid `0..39` and the fixed VR speed.

ii.
```python
"SoundDelPos": np.asarray(d["SoundDelPos"]),
```
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
...
cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
```

iii. The agent did not separately justify `SoundDelPos`, but this follows directly from its step-20 choice to express all trial covariates on the same 0.1 m running-only grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue position is reduced modulo the 60-bin trial template, then the AI subtracts the current spatial bin from the cue position and divides by the assumed fixed VR speed to obtain seconds to cue at each sample.

ii.
```python
cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
```

iii. The justification is implicit in steps 20 and 91: once the agent committed to a fixed 0.1 m grid, it treated distance-to-cue divided by VR speed as the aligned time-to-cue signal.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by construction to the same 40-bin per-trial grid as the neural data.

ii.
```python
inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
...
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
session_input, session_output = trial_covariates(behavior, day, speed_edges)
```

iii. In step 91, the agent said the sample plot showed cue countdown aligned with the neural trajectories, reflecting its shared-bin alignment strategy.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives training day from metadata rows in `Imaging_Exp_info.npy`, preferring an explicit `days` field and otherwise falling back to `sess#`.

ii.
```python
def day_value(references: list[tuple[str, dict]]) -> float:
    explicit = [row["days"] for _, row in references if "days" in row]
    if explicit:
        return float(explicit[0])
    sessions = [row["sess#"] for _, row in references if "sess#" in row]
    return float(min(sessions)) if sessions else 0.0
```

iii. The agent explicitly justified this in step 64, stating that later training recordings use explicit `days` annotations and the remaining recordings use the release's session annotation.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI takes either the first explicit `days` value or the minimum `sess#` across duplicated references for a physical recording, converts it to float, and broadcasts it across all 40 samples of each trial.

ii.
```python
day = day_value(references[pid])
```
```python
inp[1] = day
```

iii. The trajectory justification is again step 64: use release annotations directly instead of reconstructing session order by date.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not derived from `StartFr` or frame timestamps. The AI derives it from the fixed 40-bin position grid and the assumed constant VR speed.

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
elapsed_s = positions / VR_SPEED_DM_S
```

iii. The agent's step-20 decision to use a 0.1 m corridor-entry-aligned template is the justification for replacing raw frame timing with template elapsed time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes elapsed time as `position_bin / 6 decimeters_per_second`, giving a shared `[0, 1/6, 2/6, ...]` template for every trial.

ii.
```python
elapsed_s = positions / VR_SPEED_DM_S
...
inp[2] = elapsed_s
```

iii. In step 20 the agent justified this by treating 0.1 m bins at fixed VR speed as uniform temporal samples.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by using the exact same 40 bins used for the interpolated neural data.

ii.
```python
inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
...
cube = np.empty(
    (behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32
)
```

iii. The agent's shared-grid justification appears throughout the trajectory, especially steps 20 and 91.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived from `isRew`.

ii.
```python
"isRew": np.asarray(d["isRew"], dtype=bool),
...
inp[3] = float(behavior["isRew"][trial])
```

iii. The trajectory does not contain a separate discussion of this variable; it is simply carried through on the chosen trial grid.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the per-trial boolean reward flag to float and broadcasts it across all 40 samples of the trial.

ii.
```python
inp[3] = float(behavior["isRew"][trial])
```

iii. No separate justification was given beyond the general design choice to make all four decoder inputs available at every sample.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The visual stimulus output is derived from `WallName`.

ii.
```python
"WallName": np.asarray(d["WallName"]),
...
out[0] = visual_category(behavior["WallName"][trial])
```

iii. In step 48, the agent said the exemplar names map cleanly into four requested categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collapses corridor texture exemplars and swaps into four categories: `circle`, `leaf`, `rock`, and a final category it names `brick`, while also accepting raw names beginning with `wood`.

ii.
```python
CATEGORY_NAMES = ["circle", "leaf", "rock", "brick"]
```
```python
if name.startswith("circle"):
    return 0
if name.startswith("leaf"):
    return 1
if name.startswith("rock"):
    return 2
if name.startswith("wood") or name.startswith("brick"):
    return 3
```

iii. The trajectory justification is step 48, where the agent said it had mapped all exemplar types into four requested categories and interpreted the raw `wood` family as a brick-texture family.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The AI derives licking from `LickTrind` and `LickPos`, not from `LickFr`.

ii.
```python
"LickTrind": np.asarray(d["LickTrind"]).astype(np.int64),
"LickPos": np.asarray(d["LickPos"]),
```
```python
lick = np.zeros((behavior["ntrials"], CORRIDOR_BINS), dtype=np.int16)
lick_bin = np.floor(behavior["LickPos"]).astype(np.int64)
```

iii. The trajectory does not explicitly justify this variable choice; it is a consequence of the agent's step-20 commitment to spatially resampled trial data.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick is assigned to a trial with `LickTrind`, floored into a 0.1 m bin with `LickPos`, and the corresponding spatial bin is set to 1. Multiple licks in one bin still produce a single binary 1.

ii.
```python
valid = (
    (behavior["LickTrind"] >= 0)
    & (behavior["LickTrind"] < behavior["ntrials"])
    & (lick_bin >= 0)
    & (lick_bin < CORRIDOR_BINS)
)
lick[behavior["LickTrind"][valid], lick_bin[valid]] = 1
```

iii. The only explicit justification is the general one from steps 20 and 91: all outputs should live on the same 40-sample spatial/temporal template as the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by trial index and 0.1 m bin index on the same 40-bin grid as the interpolated neural data.

ii.
```python
out[1] = lick[trial]
...
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```

iii. Step 91 states that the plotted samples showed aligned signals, which is the trajectory's evidence for this alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI does not read `ft_Pos` for the converted output. Instead, position is derived from the synthetic fixed bin index `0..39` associated with the resampled 4 m template.

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
```

iii. The trajectory justification is step 20: once the agent adopted a fixed 4 m x 0.1 m template, position became implicit in bin index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI assigns each of the 40 fixed bins to one of four 1 m classes using integer division by 10 and reuses the same class vector for every trial.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
...
out[2] = position_class
```

iii. This follows directly from the step-20 representation choice to keep the first 4 m of corridor as fixed bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into four equal-length 1 m categories by grouping bins `0-9`, `10-19`, `20-29`, and `30-39`.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
```

iii. The agent justified this indirectly in step 20 as a natural consequence of keeping 40 bins across a 4 m corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned by using the same 40 trial bins as the interpolated neural data.

ii.
```python
out[2] = position_class
...
cube = np.empty(
    (behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32
)
```

iii. The trajectory repeatedly claims the 40-bin representation is aligned across signals (steps 20, 88, 91).

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `run_pos`, which the AI treats as per-trial values on the same 60-position template and then truncates to the first 40 bins.

ii.
```python
"run_pos": np.asarray(d["run_pos"], dtype=np.float32)[:, :CORRIDOR_BINS],
```
```python
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
```

iii. The trajectory does not name `run_pos` directly, but steps 20 and 88 show the agent wanted running-speed labels already aligned to the 40-bin template and balanced globally across quartiles.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI concatenates `run_pos` over all sessions, computes global 25/50/75% quantile edges, and digitizes each trial's 40 values against those shared thresholds.

ii.
```python
all_speeds = np.concatenate([behaviors[pid]["run_pos"].ravel() for pid in sessions])
speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75]).astype(np.float32)
```
```python
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
```

iii. In step 88, the agent highlighted that the resulting global speed classes were exactly balanced at 25% each, which is its main justification for this processing choice.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded with three global quantile edges computed across the entire dataset, producing four quartile bins labeled `0-25%`, `25-50%`, `50-75%`, and `75-100%`.

ii.
```python
speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75]).astype(np.float32)
```
```python
"output_values": [
    CATEGORY_NAMES,
    ["not licking", "licking"],
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    ["0-25%", "25-50%", "50-75%", "75-100%"],
],
```

iii. The agent explicitly pointed to exact 25% global class balance in step 88.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by storing one quartile label per 0.1 m bin on the same 40-bin per-trial grid used for the neural data.

ii.
```python
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
...
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```

iii. The trajectory justification is the same shared-grid claim from steps 20, 88, and 91.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly fails fast on structural mismatches instead of trying to repair them. It checks for neural-plane consistency, behavior shorter than neural activity, missing behavior records, and retinotopy/neuron-count mismatches. For licking it drops invalid trial/bin indices rather than erroring.

ii.
```python
if len(behavior["ft_move"]) < nframes:
    raise ValueError("Behavior frame stream is shorter than neural activity")
```
```python
if sum(len(p) for p in planes) != len(iarea):
    raise ValueError(f"Neuron/retinotopy mismatch for {pid}")
```
```python
missing = [pid for pid in sessions if pid not in behaviors]
if missing:
    raise KeyError(f"No behavior record found for {missing}")
```
```python
valid = (
    (behavior["LickTrind"] >= 0)
    & (behavior["LickTrind"] < behavior["ntrials"])
    & (lick_bin >= 0)
    & (lick_bin < CORRIDOR_BINS)
)
```

iii. The trajectory emphasizes that the build passed all sessions without data-quality exceptions (step 58), so the agent appears to have viewed the dataset as sufficiently clean to justify hard checks.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading very large spike files, interpolating them into `(trial, neuron, 40)` cubes, and serializing the resulting approximately 274 GB pickle.

ii.
```python
raw_neural = np.load(spk_path, allow_pickle=True).item()
planes = raw_neural["spks"]
```
```python
values = y[:, lo] * (1.0 - weight) + y[:, hi] * weight
cube[:, dest_offset : dest_offset + count, :] = values.reshape(
    count, behavior["ntrials"], CORRIDOR_BINS
).transpose(1, 0, 2)
```
```python
with open(tmp_path, "wb", buffering=16 * 1024 * 1024) as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly highlighted the approximate 274 GB output size and the cost of the interpolation and validation passes (steps 23, 26, 77, 97).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the core interpolation. Remaining loops that could still be reduced are the per-plane/per-chunk neuron loops in `interpolate_session` and the per-trial loop in `trial_covariates`.

ii.
```python
for plane in planes:
    plane_keep = np.flatnonzero(keep[source_offset : source_offset + len(plane)])
    ...
    for start in range(0, len(plane_keep), chunk_neurons):
        ids = plane_keep[start : start + chunk_neurons]
```
```python
for trial in range(behavior["ntrials"]):
    ...
    session_inputs.append(inp)
    session_outputs.append(out)
```

iii. In steps 23 and 26, the agent said it had deliberately implemented a vectorized interpolation checked against the paper helper, implying these remaining loops were acceptable engineering tradeoffs rather than oversights.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats some lightweight administrative work: it scans `Imaging_Exp_info.npy` once to build `references` and again to materialize `behaviors`, and during session assembly it repeatedly performs `subjects.index(mouse)`. The heavy data transforms are otherwise session-local.

ii.
```python
for exp_type, rows in exp_info.items():
    for row in rows:
        ...
```
```python
for exp_type, rows in exp_info.items():
    unresolved = [row for row in rows if physical_id(row) not in behaviors]
    ...
```
```python
if mouse not in subjects:
    subjects.append(mouse)
subject_idx.append(subjects.index(mouse))
```

iii. The trajectory does not call these repeats out explicitly; its focus was on correctness and throughput of the vectorized interpolation (steps 23 and 26).

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There is no large discarded downstream transform, but there is some minor unnecessary work: `extract_behavior` keeps `Corridor_Length` and `Texture_Length` even though the conversion never uses them, and the script performs frequent `gc.collect()` calls that do not change the converted values.

ii.
```python
return {
    ...
    "Corridor_Length": float(d["Corridor_Length"]),
    "Texture_Length": float(d["Texture_Length"]),
}
```
```python
del raw
gc.collect()
```
```python
del raw_neural, planes, cube, iarea, keep, regions
gc.collect()
```

iii. The trajectory frames these as pragmatic implementation details for handling a huge conversion rather than as deliberate analytic choices; no specific scientific justification was given.
