# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded from three subdirectories under `data`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. `beh/Imaging_Exp_info.npy` is loaded first as the master index listing every recording grouped by experiment type. Each behavior file (`Beh_<exp_type>.npy`) is loaded once for its group of sessions. Neural data is loaded per-session from `spk/<mouse>_<date>_<block>_neural_data.npy`. Retinotopy is loaded from `retinotopy/<mouse>_<date>_trans.npz`. The AI additionally extracts only the needed behavior fields into a lean dictionary to reduce memory.

ii.
```python
exp_info = np.load(data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
```
```python
path = data_root / "beh" / f"Beh_{exp_type}.npy"
raw = np.load(path, allow_pickle=True).item()
behaviors[pid] = extract_behavior(raw[key])
```
```python
spk_path = data_root / "spk" / f"{mouse}_{date}_{block}_neural_data.npy"
raw_neural = np.load(spk_path, allow_pickle=True).item()
planes = raw_neural["spks"]
```
```python
retino_path = data_root / "retinotopy" / f"{mouse}_{date}_trans.npz"
iarea = np.load(retino_path, allow_pickle=True)["iarea"]
```

iii. From the trajectory (step 8): "I'm narrowing this to the imaging-session files and tracing the authors' exact running-only selection, binning, trial fields, neuron curation, and cortical-area labels." The AI examined the full data layout and chose to load each behavior file once per experiment type and release it immediately afterward to save memory.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info. The AI builds a list of subjects in the order they are first encountered while iterating sessions, then assigns each session a `subject_idx` into that list.

ii.
```python
if mouse not in subjects:
    subjects.append(mouse)
subject_idx.append(subjects.index(mouse))
```

iii. From trajectory (step 16): "there are 89 unique recordings from 19 mice." The mouse name is directly available in the metadata.

## 1-c. How are the data split into sessions?

i. A session is one physical recording, identified by the tuple (mname, datexp, blk). The AI deduplicates recordings that appear under multiple experiment types, keeping only the first occurrence. This yields 89 unique sessions.

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

iii. From trajectory (step 16): "the 142 metadata references intentionally reuse recordings for different paper analyses. I'll represent each physical recording once."

## 1-d. How are the data split into trials?

i. Trials are taken directly from the behavior data — each session has `ntrials` trials. ALL trials are kept (no filtering). Each trial is represented as a fixed-length array of 40 spatial bins (0.1 m each across the 4 m textured corridor), rather than variable-length temporal frames. The neural data is spatially interpolated onto these bins.

ii.
```python
cube = interpolate_session(planes, keep, behavior, chunk_neurons)
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```
```python
CORRIDOR_BINS = 40
```

iii. From trajectory (step 20): "I'll use the authors' running-only linear interpolation onto 0.1 m bins, retaining the first 4 m (40 samples) of each trial."

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered. All `ntrials` trials from each session are kept. There is no outlier-length filtering or empty-trial removal.

ii.
```python
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```
```python
for trial in range(behavior["ntrials"]):
    # all trials processed
```

iii. From trajectory (step 20): The AI chose to keep all trials because spatial interpolation produces fixed-length (40-bin) representations for every trial regardless of how long the mouse took to traverse the corridor. The problem of variable-length trials caused by animals stopping is avoided by the spatial binning approach.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains a list of one neurons-by-frames array per imaging plane. The visual area of each neuron comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
spk_path = data_root / "spk" / f"{mouse}_{date}_{block}_neural_data.npy"
raw_neural = np.load(spk_path, allow_pickle=True).item()
planes = raw_neural["spks"]
```
```python
iarea = np.load(retino_path, allow_pickle=True)["iarea"]
```

iii. From trajectory (step 11): "3.17 Hz deconvolved Suite2p activity" — the raw deconvolved traces are the source.

## 2-b. How is the `neural` data processed?

i. The neural data undergoes spatial interpolation matching the paper's `spk_pos_interp` method. Only frames where the VR moved (`ft_move > 0`) are used. The cumulative VR position (`ft_PosCum`) of these frames forms the x-axis for linear interpolation. Neural activity is interpolated onto 40 evenly-spaced target positions (0.1 m bins across the 4 m corridor) per trial. The result is stored as float32.

ii.
```python
moving = behavior["ft_move"][:nframes] > 0
x = np.asarray(behavior["ft_PosCum"][:nframes][moving], dtype=np.float64)
targets = (
    np.arange(behavior["ntrials"], dtype=np.float64)[:, None] * FULL_TRIAL_BINS
    + np.arange(CORRIDOR_BINS, dtype=np.float64)[None, :]
).ravel()
lo, hi, weight = interpolation_lookup(x, targets)
```
```python
y = plane[ids][:, moving]
values = y[:, lo] * (1.0 - weight) + y[:, hi] * weight
```

iii. From trajectory (step 20): "I'll use the authors' running-only linear interpolation onto 0.1 m bins, retaining the first 4 m (40 samples) of each trial. Because VR advances at a fixed 0.6 m/s, these are also uniform 166.67 ms samples aligned to corridor entry. This directly matches their released processing."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area. A neuron is kept if its `iarea` code maps to V1 (8), mHV (0,1,2,9), lHV (5,6), or aHV (3,4). Neurons with iarea -1 or 7 (outside visual cortex) are excluded. This matches the paper's `neu_area_ID` function.

ii.
```python
def area_indices(iarea: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    region = np.full(len(iarea), -1, dtype=np.int8)
    region[iarea == 8] = 0
    region[np.isin(iarea, [0, 1, 2, 9])] = 1
    region[np.isin(iarea, [5, 6])] = 2
    region[np.isin(iarea, [3, 4])] = 3
    keep = region >= 0
    return keep, region[keep]
```

iii. From trajectory (step 16): "preserving all released Suite2p-selected neurons—there is no paper-supported extra neuron subsampling."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to corridor entry (trial start). Through spatial interpolation, each trial starts at position 0 (corridor entry) and covers 40 bins of 0.1 m each. All trials have the same fixed length of 40 bins. The alignment is inherent in the spatial binning: bin 0 always corresponds to corridor entry.

ii.
```python
targets = (
    np.arange(behavior["ntrials"], dtype=np.float64)[:, None] * FULL_TRIAL_BINS
    + np.arange(CORRIDOR_BINS, dtype=np.float64)[None, :]
).ravel()
```

iii. From trajectory (step 20): "retaining the first 4 m (40 samples) of each trial... aligned to corridor entry."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is spatially binned rather than temporally binned. Each bin is 0.1 m of corridor, and at the VR speed of 60 cm/s (6 dm/s), this corresponds to 166.67 ms per bin. The AI reports `time_bin_size = 1000.0 / VR_SPEED_DM_S = 166.67 ms`. This is a spatial resampling via linear interpolation, not temporal rebinning.

ii.
```python
VR_SPEED_DM_S = 6.0
```
```python
'time_bin_size': 1000.0 / VR_SPEED_DM_S,  # = 166.67 ms
```

iii. From trajectory (step 20): "Because VR advances at a fixed 0.6 m/s, these are also uniform 166.67 ms samples aligned to corridor entry."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundDelPos`, which gives the sound delivery position in decimeters for each trial.

ii.
```python
cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
```

iii. From trajectory: The AI chose to work in position/spatial domain since the data was spatially interpolated. `SoundDelPos` gives the position at which the sound cue is delivered.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue position in decimeters is converted to a time difference by dividing by VR speed (6 dm/s). For each spatial bin, the input is `(cue_dm - bin_position) / VR_SPEED_DM_S`, giving a positive value before the cue and negative after.

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
```

iii. The AI converted spatial distances to time using the fixed VR speed, consistent with working in the spatial domain.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed over the same 40 spatial bins as the neural data. Both share the same positional grid, so alignment is inherent.

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
```

iii. All variables are on the same 40-bin spatial grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `days` field in `Imaging_Exp_info.npy` entries when available, and from the `sess#` field otherwise.

ii.
```python
def day_value(references: list[tuple[str, dict]]) -> float:
    explicit = [row["days"] for _, row in references if "days" in row]
    if explicit:
        return float(explicit[0])
    sessions = [row["sess#"] for _, row in references if "sess#" in row]
    return float(min(sessions)) if sessions else 0.0
```

iii. From trajectory (step 64): "Explicit `days` annotations are being used for the later training recordings where provided; elsewhere the release's session annotation supplies the per-trial training-day input."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The `days` or `sess#` value is taken directly as a float and broadcast as a constant across all 40 bins of every trial in that session. When `days` is available, it is used directly. When only `sess#` is available, the minimum session number across analysis groups is used.

ii.
```python
day = day_value(references[pid])
```
```python
inp[1] = day
```

iii. The AI used the explicit metadata annotations rather than deriving day counts from date strings.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the spatial bin positions and VR speed. No raw behavioral variable is used directly — it is computed deterministically from the bin index.

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
elapsed_s = positions / VR_SPEED_DM_S
```

iii. Since data is spatially binned at a constant VR speed, position directly maps to elapsed time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each spatial bin's position (in decimeters) is divided by the VR speed (6 dm/s) to get elapsed time in seconds. Bin 0 gives 0 s, bin 39 gives 6.5 s.

ii.
```python
elapsed_s = positions / VR_SPEED_DM_S
inp[2] = elapsed_s
```

iii. The computation is deterministic from the spatial grid; no raw frame timestamps are needed.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. It shares the same 40-bin spatial grid as the neural data, so alignment is inherent.

ii.
```python
inp[2] = elapsed_s  # same 40-bin grid as neural
```

iii. All variables use the same positional grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in the rewarded corridor.

ii.
```python
inp[3] = float(behavior["isRew"][trial])
```

iii. Directly available from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float (0.0 or 1.0) and broadcast as a constant across all 40 bins of the trial.

ii.
```python
inp[3] = float(behavior["isRew"][trial])
```

iii. No processing needed beyond type conversion.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
out[0] = visual_category(behavior["WallName"][trial])
```

iii. From trajectory (step 48): "naive recordings with all exemplar types; their names map cleanly into the four requested categories (circle, leaf, rock, brick/'wood')."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names are collapsed into four categories by prefix matching: `circle*` -> 0, `leaf*` -> 1, `rock*` -> 2, `wood*`/`brick*` -> 3. The AI calls the fourth category "brick" rather than "wood". The category index is broadcast across all 40 bins.

ii.
```python
CATEGORY_NAMES = ["circle", "leaf", "rock", "brick"]

def visual_category(name: str) -> int:
    name = str(name).lower()
    if name.startswith("circle"): return 0
    if name.startswith("leaf"): return 1
    if name.startswith("rock"): return 2
    if name.startswith("wood") or name.startswith("brick"): return 3
    raise ValueError(f"Unrecognized visual stimulus name: {name!r}")
```

iii. The AI noted that the raw data files use "wood" names for what the paper describes as a brick-texture family, hence the renaming.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTrind` (trial index of each lick) and `LickPos` (position of each lick in decimeters).

ii.
```python
"LickTrind": np.asarray(d["LickTrind"]).astype(np.int64),
"LickPos": np.asarray(d["LickPos"]),
```

iii. The AI used position-based lick data since the neural data is spatially binned.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick positions are floored to integer spatial bins. A bin is marked 1 if at least one lick occurred in its 0.1 m range, 0 otherwise. Only licks with valid trial indices and positions within [0, 40) are included.

ii.
```python
lick = np.zeros((behavior["ntrials"], CORRIDOR_BINS), dtype=np.int16)
lick_bin = np.floor(behavior["LickPos"]).astype(np.int64)
valid = (
    (behavior["LickTrind"] >= 0)
    & (behavior["LickTrind"] < behavior["ntrials"])
    & (lick_bin >= 0)
    & (lick_bin < CORRIDOR_BINS)
)
lick[behavior["LickTrind"][valid], lick_bin[valid]] = 1
```

iii. Licking is mapped to spatial bins to match the spatial domain of the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are binned by their spatial position (`LickPos`) into the same 40-bin grid as the neural data. Alignment is inherent in sharing the spatial grid.

ii.
```python
out[1] = lick[trial]  # same 40-bin spatial grid
```

iii. All data streams share the 40-bin spatial grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived deterministically from the spatial bin index — no raw behavioral variable is needed. The position is inherent in the grid structure.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
```

iii. Since data is spatially binned in 0.1 m increments, every 10 bins corresponds to 1 m.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 bins are divided into 4 groups of 10 (bins 0-9 -> class 0 "0-1m", 10-19 -> class 1, etc.) by integer division.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
out[2] = position_class
```

iii. The spatial binning makes position discretization trivial.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1 m bins, each containing 10 spatial samples (0.1 m each). Integer division by 10 gives the category index 0-3.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
```

iii. Matches the instruction for 4 equal-length 1-m-long spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position class is defined on the same 40-bin spatial grid as neural data. Alignment is inherent.

ii.
```python
out[2] = position_class  # same 40 bins as neural
```

iii. All data streams share the spatial grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `run_pos`, the running speed interpolated at each spatial position for each trial (a `ntrials x 60` array, truncated to the first 40 bins).

ii.
```python
"run_pos": np.asarray(d["run_pos"], dtype=np.float32)[:, :CORRIDOR_BINS],
```

iii. The AI used the position-interpolated running speed to match the spatial binning of neural data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 quartile bins. The quartile edges are computed GLOBALLY across all sessions and all spatial bins, using `np.quantile` at [0.25, 0.50, 0.75]. Then `np.digitize` assigns each value to a bin.

ii.
```python
all_speeds = np.concatenate([behaviors[pid]["run_pos"].ravel() for pid in sessions])
speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75]).astype(np.float32)
```
```python
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
```

iii. From trajectory (step 88): "Speed classes exactly balanced at 25% each." The global quartile approach ensures each bin contains exactly 25% of data across the full dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global value-based quartile edges are computed, and `np.digitize` assigns each speed value to one of 4 bins (0-25%, 25-50%, 50-75%, 75-100%). This is value-based thresholding (not rank-based).

ii.
```python
speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75]).astype(np.float32)
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
```

iii. The edges are from value quantiles which guarantee 25% of data in each bin globally.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` is already position-interpolated, providing one speed value per 0.1 m spatial bin. It shares the same 40-bin grid as the neural data.

ii.
```python
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
```

iii. All data streams share the spatial grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI validates that neural planes have consistent frame counts and that behavior frame streams are at least as long as neural activity. Lick events are filtered by valid trial index and position range. The AI also checks that neuron counts match between spike data and retinotopy. A total neuron/retinotopy mismatch raises an error.

ii.
```python
if any(p.shape[1] != nframes for p in planes):
    raise ValueError("Neural planes have inconsistent frame counts")
if len(behavior["ft_move"]) < nframes:
    raise ValueError("Behavior frame stream is shorter than neural activity")
if sum(len(p) for p in planes) != len(iarea):
    raise ValueError(f"Neuron/retinotopy mismatch for {pid}")
```
```python
valid = (
    (behavior["LickTrind"] >= 0)
    & (behavior["LickTrind"] < behavior["ntrials"])
    & (lick_bin >= 0)
    & (lick_bin < CORRIDOR_BINS)
)
```

iii. The AI raises errors for data inconsistencies rather than silently handling them, except for out-of-range licks which are filtered.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the neural spike files (~274 GB total) and performing the spatial interpolation for each session. The interpolation involves reading all frames, filtering moving frames, and computing interpolated values for all neurons.

ii.
```python
raw_neural = np.load(spk_path, allow_pickle=True).item()
planes = raw_neural["spks"]
cube = interpolate_session(planes, keep, behavior, chunk_neurons)
```

iii. From trajectory (step 69): "All 89 sessions have converted successfully" — the conversion took substantial time processing all 274 GB of neural data.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `trial_covariates` iterates over trials to build input and output arrays, but most of the computation is constant across trials (elapsed_s, position_class) or could be vectorized (stimulus, reward broadcasting). The interpolation itself is already vectorized across neurons in chunks.

ii.
```python
for trial in range(behavior["ntrials"]):
    cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
    inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
    # ... per-trial computation
```

iii. The trial loop is not a bottleneck compared to I/O and interpolation.

## 12-c. What processing does the code repeat multiple times?

i. The `elapsed_s` and `position_class` arrays are recomputed identically inside the trial loop for every trial, when they could be computed once outside the loop.

ii.
```python
for trial in range(behavior["ntrials"]):
    # These are constant across trials but computed inside the loop:
    inp[2] = elapsed_s  # same for every trial
    out[2] = position_class  # same for every trial
```

iii. The repeated computation is trivial in cost.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes the full interpolation for all 60 spatial bins (FULL_TRIAL_BINS) in the target positions, but only retains the first 40 (CORRIDOR_BINS). The gray-space bins (40-59) are used in the interpolation lookup but not stored. Additionally, `run_pos` is loaded as a full ntrials x 60 array but immediately truncated to `[:, :CORRIDOR_BINS]`.

ii.
```python
CORRIDOR_BINS = 40
FULL_TRIAL_BINS = 60
targets = (
    np.arange(behavior["ntrials"], dtype=np.float64)[:, None] * FULL_TRIAL_BINS
    + np.arange(CORRIDOR_BINS, dtype=np.float64)[None, :]
).ravel()
```
```python
"run_pos": np.asarray(d["run_pos"], dtype=np.float32)[:, :CORRIDOR_BINS],
```

iii. The truncation is intentional to exclude the gray corridor space that has no visual stimulus.
