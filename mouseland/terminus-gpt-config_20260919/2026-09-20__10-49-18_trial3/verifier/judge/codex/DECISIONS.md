# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI used the spike files in `/app/data/spk` as the authoritative session list, then loaded every `Beh_*.npy` file and grouped behavior dictionaries by a “physical” session id with `_swap1`/`_swap2` removed. It loaded retinotopy per session from the corresponding `*_trans.npz` file. It only used `Imaging_Exp_info.npy` later for metadata, not as the master index for session enumeration.

ii. 
```python
def neural_files():
    return {
        os.path.basename(p).removesuffix("_neural_data.npy"): p
        for p in glob.glob(os.path.join(ROOT, "spk", "*_neural_data.npy"))
    }
```
```python
def load_behavior_views(valid_ids):
    views = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(ROOT, "beh", "Beh_*.npy"))):
        obj = np.load(path, allow_pickle=True).item()
        for key, beh in obj.items():
            sid = physical_id(key)
            if sid in valid_ids:
                views[sid].append((group, key, beh))
```
```python
rp = retinotopy_path(sid)
with np.load(rp) as r:
    region_idx = map_regions(r["iarea"])
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as a way to deduplicate the 142 behavior-analysis entries down to the 89 physical recordings and to avoid treating duplicate behavior views as distinct sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by the mouse-name prefix of each physical session id, i.e. `sid.split("_")[0]`. The `subjects` list is the sorted unique set of those names, and `subject_idx` maps each selected session to that list.

ii. 
```python
subjects = sorted({sid.split("_")[0] for sid in selected})
subject_map = {v: i for i, v in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_map[s.split("_")[0]] for s in selected], dtype=np.int64),
```

iii. The AI’s notes treat `<mouse>_<YYYY>_<MM>_<DD>_<block>` as the physical session id, so the mouse prefix is the natural subject key.

## 1-c. How are the data split into sessions?

i. A session is one physical recording identified by the spike filename stem, e.g. `TX60_2021_04_10_1`. Duplicate behavior views are merged onto that session by stripping `_swap1`/`_swap2` from behavior keys.

ii. 
```python
def physical_id(key):
    return re.sub(r"_swap[12]$", "", key)
```
```python
paths_all = neural_files()
if len(paths_all) != 89:
    raise RuntimeError(f"Expected 89 unique neural sessions, found {len(paths_all)}")
...
selected = sorted(paths_all)
```

iii. The notes explicitly say that the dataset has 89 unique physical session IDs and that duplicate behavior views are “not additional recordings or trials.”

## 1-d. How are the data split into trials?

i. Trials are taken as contiguous frame windows from `ceil(StartFr)` through `ceil(GrayFr)`, clipped to the neural frame count. The AI did not use `ft_trInd` and `ft_CorrSpc` to define trial membership.

ii. 
```python
def trial_bounds(beh, nfr):
    starts = np.ceil(np.asarray(beh["StartFr"], dtype=np.float64)).astype(np.int64)
    ends = np.ceil(np.asarray(beh["GrayFr"], dtype=np.float64)).astype(np.int64)
    starts = np.clip(starts, 0, nfr)
    ends = np.clip(ends, 0, nfr)
```
```python
for i, (a, z) in enumerate(zip(starts, ends)):
    frames = np.arange(a, z, dtype=np.float64)
```

iii. In the trajectory, the AI justified this by saying `ceil(StartFr)` is the first frame whose `ft_trInd` belongs to the new trial, `ceil(GrayFr)` is the exclusive end of the 4 m corridor, and `[ceil(StartFr), ceil(GrayFr))` avoids including a previous-trial frame.

## 1-e. How are trials filtered based on quality controls?

i. The AI did not apply a trial-length or empty-trial filter. It kept every trial whose `ceil(StartFr)` / `ceil(GrayFr)` bounds formed a positive-length window after clipping to `nfr`; otherwise it raised an error instead of selectively dropping the trial.

ii. 
```python
if np.any(ends <= starts):
    bad = np.flatnonzero(ends <= starts)
    raise ValueError(f"Invalid trial windows: {bad[:20].tolist()}")
```
```python
for i, (a, z) in enumerate(zip(starts, ends)):
    ...
    neural_trials.append(nt)
```

iii. The AI’s notes explicitly say “Retain all 38,110 valid windows; do not invent a duration filter,” even though it observed rare extremely long stalled trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the three arrays in `obj["spks"]` from each `*_neural_data.npy` file. Brain-region labels are derived from retinotopy `iarea`.

ii. 
```python
obj = np.load(path, allow_pickle=True).item()
parts = obj["spks"]
```
```python
with np.load(rp) as r:
    region_idx = map_regions(r["iarea"])
```

iii. The notes say the released neural values are already Suite2p deconvolved traces and that retinotopy `iarea` provides the anatomical mapping.

## 2-b. How is the `neural` data processed?

i. The AI concatenated the three `spks` arrays only within each retained trial slice, did no normalization or deconvolution, and stored each trial as float32. Trial lengths remain variable.

ii. 
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
neural_trials.append(nt)
```

iii. The AI justified this as preserving the reference `load_spk` neuron order while avoiding a redundant full-session concatenation copy.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI did not drop neurons outside the four named visual regions. Instead it mapped them to a fifth `"unassigned"` class and retained all neurons.

ii. 
```python
REGIONS = ["V1", "mHV", "lHV", "aHV", "unassigned"]
```
```python
def map_regions(iarea):
    a = np.asarray(iarea)
    out = np.full(a.shape, 4, dtype=np.int16)
    out[a == 8] = 0
    out[np.isin(a, [0, 1, 2, 9])] = 1
    out[np.isin(a, [5, 6])] = 2
    out[np.isin(a, [3, 4])] = 3
    return out
```

iii. The notes justify this as keeping “all released Suite2p traces” and making unmapped retinotopy explicit rather than silently dropping those ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry by starting each neural window at `ceil(StartFr)` and ending at `ceil(GrayFr)`. The arrays therefore begin at the first included post-entry frame and end at corridor exit, with variable duration.

ii. 
```python
starts = np.ceil(np.asarray(beh["StartFr"], dtype=np.float64)).astype(np.int64)
ends = np.ceil(np.asarray(beh["GrayFr"], dtype=np.float64)).astype(np.int64)
```
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```

iii. The trajectory says this was chosen because it aligned to the requested trial-start event and excluded gray-space frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI kept one sample per native imaging frame. It estimated frame duration per session from the median difference in `ft`, converted from MATLAB days to seconds, and did not apply any temporal rebinning.

ii. 
```python
def session_dt_seconds(beh):
    d = np.diff(np.asarray(beh["ft"], dtype=np.float64))
    d = d[np.isfinite(d) & (d > 0)]
    dt = float(np.median(d) * 86400.0)
```
```python
"time_bin_size": float(np.median(dt_values)),
```

iii. The notes state that the imaging-frame grid already synchronizes neural and behavioral data, so no resampling was needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derived it from `SoundFr` and the per-trial frame index range; it did not use absolute `ft` timestamps directly in the final computation.

ii. 
```python
sound = np.asarray(beh["SoundFr"], dtype=np.float64)
...
frames = np.arange(a, z, dtype=np.float64)
inp[0] = (sound[i] - frames) * dt
```

iii. The notes frame this as a frame-index-based temporal alignment problem once `dt` has been inferred from `ft`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computed signed continuous time-to-cue in seconds as `(SoundFr - current_frame) * dt`, so it is positive before the cue and negative after.

ii. 
```python
inp[0] = (sound[i] - frames) * dt
```

iii. The trajectory explicitly says the converter should represent time-to-cue as signed seconds, `cue_time - current_time`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on exactly the same frame indices `a:z` used for that trial’s neural slice, so it has the same number of time bins as the neural data.

ii. 
```python
frames = np.arange(a, z, dtype=np.float64)
...
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
inp[0] = (sound[i] - frames) * dt
```

iii. The AI’s broader justification is that all decoder inputs and outputs must share the neural trial time axis.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in each session id, parsed as `<mouse>_<YYYY>_<MM>_<DD>_<block>`.

ii. 
```python
def parse_session_id(sid):
    m = re.fullmatch(r"(.+?)_(\d{4}_\d{2}_\d{2})_([^_]+)", sid)
```
```python
mouse, date, _ = parse_session_id(sid)
d = datetime.strptime(date, "%Y_%m_%d").date()
```

iii. The notes say `sess#` was heterogeneous or missing, so the AI relied on recording dates instead.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computed calendar days since each mouse’s earliest recording among all 89 sessions and broadcast that scalar across each trial.

ii. 
```python
def subject_day_values(session_ids):
    first = {}
    dates = {}
    for sid in session_ids:
        mouse, date, _ = parse_session_id(sid)
        d = datetime.strptime(date, "%Y_%m_%d").date()
        dates[sid] = d
        first[mouse] = min(first.get(mouse, d), d)
    return {sid: float((dates[sid] - first[sid.split("_")[0]]).days) for sid in session_ids}
```
```python
inp[1] = day_value
```

iii. The AI justified this as a continuous, reproducible substitute for inconsistent or absent session-number metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and the per-trial frame indices, together with the inferred frame duration `dt`.

ii. 
```python
frames = np.arange(a, z, dtype=np.float64)
...
inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. The notes describe this as frame-relative timing anchored to corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computed signed continuous seconds since corridor entry as `(current_frame - StartFr) * dt`. Because the first included frame is `ceil(StartFr)`, values start near zero and are nonnegative in the retained trial window.

ii. 
```python
inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. The trajectory says the converter should use entry-aligned corridor-only windows and continuous timing based on those frames.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same frame indices `a:z` as the neural trial slice, so it is exactly time-aligned bin-by-bin with the neural array.

ii. 
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. The AI’s implementation enforces equal timepoint counts for neural, input, and output trial arrays.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the trial-level `isRew` array.

ii. 
```python
reward = np.asarray(beh["isRew"], dtype=np.int16)
...
inp[3] = reward[i]
```

iii. The AI’s notes describe this as the rewarded-corridor flag, not actual reward delivery time.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond integer casting and per-trial broadcasting across all time bins of that trial.

ii. 
```python
reward = np.asarray(beh["isRew"], dtype=np.int16)
...
inp[3] = reward[i]
```

iii. The AI justified broadcasting because the decoder validator/trainer expects aligned time axes for all variables.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derived visual stimulus labels from `TrialStim` merged across duplicate behavior views, with `WallName` used only as a fallback when all views contained the placeholder `stimulus_of_trial`.

ii. 
```python
concrete = {
    str(b["TrialStim"][i]) for _, _, b in session_views
    if str(b["TrialStim"][i]) != "stimulus_of_trial"
}
...
wall = {str(b["WallName"][i]) for _, _, b in session_views}
```

iii. The notes say this was meant to recover concrete labels from swap-session duplicate views while avoiding masked `TrialStim` placeholders.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI merged duplicate-view semantic labels, fell back to `WallName` when necessary, mapped the result into eight categories (`circle1`, `circle2`, `circle3`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`), and broadcast the category index across the trial.

ii. 
```python
STIM_VALUES = [
    "circle1", "circle2", "circle3", "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
STIM_TO_ID = {v: i for i, v in enumerate(STIM_VALUES)}
```
```python
stimuli = merged_stimuli(session_views)
...
out[0] = STIM_TO_ID[str(stimuli[i])]
```

iii. The AI justified this as preserving the more specific semantic labels available in duplicate behavior views instead of collapsing them to broader texture families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the session-level list of lick frame coordinates.

ii. 
```python
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
lick_idx = np.floor(lick_frames).astype(np.int64)
```

iii. The notes say `LickFr` is already expressed in the imaging-frame coordinate system.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI floored each fractional `LickFr` value to a frame index, clipped licks to the imaged frame range, collapsed duplicate licks within a frame to 1, and produced a binary per-frame raster.

ii. 
```python
lick_session = np.zeros(nfr, dtype=np.int16)
lick_idx = np.floor(lick_frames).astype(np.int64)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
if len(lick_idx):
    lick_session[np.unique(lick_idx)] = 1
```

iii. The trajectory explicitly states that `floor(LickFr)` was chosen as the imaging interval containing the event and that multiple licks in one frame should collapse to a binary presence flag.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. After rasterizing licks on the full session frame grid, the AI slices the same `a:z` frame interval used for the neural trial.

ii. 
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
out[1] = lick_session[a:z]
```

iii. This follows the AI’s general rule that all trial variables share the neural trial frame axis.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level `ft_Pos` signal within each trial window.

ii. 
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
...
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. The notes state that source position runs over a 0–40 visual corridor corresponding to 4 physical metres.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI digitized source position into four bins using thresholds at 10, 20, and 30 source units, corresponding to 1 m bins over the 4 m visual corridor.

ii. 
```python
# Source visual corridor is 0..40 (= four physical metres).
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. The notes explicitly map 10 source units to 1 metre and describe the requested four equal corridor bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It uses `np.digitize(..., [10.0, 20.0, 30.0])` and clips the result into classes 0–3.

ii. 
```python
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. The AI’s notes say these are the `0–1 m`, `1–2 m`, `2–3 m`, and `3–4 m` bins required by the task.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sliced on the same `a:z` frame interval used for the trial’s neural slice, so it is frame-aligned with the neural data.

ii. 
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. The AI treated `ft_Pos` as already defined on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-level `ft_RunSpeed` signal.

ii. 
```python
speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
```

iii. The notes describe running speed as a frame-synchronous behavioral stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computed a single set of global quartile thresholds across all retained corridor frames of the selected sessions, then digitized each trial’s framewise running speeds against those thresholds.

ii. 
```python
def compute_speed_thresholds(selected_ids, views, nfr_by_session):
    ...
    q = np.quantile(values, [0.25, 0.5, 0.75]).astype(np.float64)
    return q, int(values.size), (float(values.min()), float(values.max()))
```
```python
speed_q, n_speed, speed_range = compute_speed_thresholds(selected, all_views, nfr_for_speed)
...
out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. The notes justify this by reading the instruction “each corresponding to 25% of the data” literally as a global data-wide quartile definition.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The categories are the four bins induced by the three global quantile thresholds `speed_q`, applied with `np.digitize` and clipped to 0–3.

ii. 
```python
speed_q, n_speed, speed_range = compute_speed_thresholds(selected, all_views, nfr_for_speed)
...
out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. The notes explicitly record the learned thresholds and explain that tied zeros may prevent exact 25% occupancy after deterministic thresholding.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sliced on the same `a:z` frame interval used for the neural trial and therefore has identical trial length.

ii. 
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. The AI consistently aligned frame-level outputs by trial-window slicing on the neural frame axis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled duplicate behavior views by consistency-checking them and merging only semantic stimulus labels; clipped trial bounds and frame-level behavioral streams to the imaged frame range; clipped licks to valid frame indices; and raised exceptions on malformed trial windows or inconsistent duplicate views rather than attempting imputation.

ii. 
```python
for key in ("StartFr", "GrayFr", "SoundFr", "isRew", "ft_Pos", "ft_RunSpeed"):
    if not np.allclose(np.asarray(b0[key]), np.asarray(b[key]), equal_nan=True):
        raise ValueError(f"Duplicate views differ in {key}")
```
```python
starts = np.clip(starts, 0, nfr)
ends = np.clip(ends, 0, nfr)
```
```python
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
```

iii. The notes say the raw dataset is clean overall, but that behavior streams can be slightly longer than imaging and duplicate behavior “views” must not be mistaken for extra data.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is reading each large neural session file and concatenating/slicing trial-wise neural blocks. In sample mode, the shape scan also rereads the spike files once before conversion.

ii. 
```python
obj = np.load(path, allow_pickle=True).item()
parts = obj["spks"]
...
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```
```python
def inspect_neural_shapes(paths):
    for j, sid in enumerate(paths, 1):
        obj = np.load(paths[sid], allow_pickle=True).item()
```

iii. The notes repeatedly identify the 434 GB spike corpus as the main runtime bottleneck and describe later optimizations as reducing redundant neural copies, not eliminating neural I/O.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loop is the per-trial loop in `convert_session`, which repeatedly constructs `frames`, concatenates the three plane slices, and fills trial arrays. The global speed-threshold collection loop also iterates session-by-session and trial-by-trial to build `chunks`.

ii. 
```python
for i, (a, z) in enumerate(zip(starts, ends)):
    frames = np.arange(a, z, dtype=np.float64)
    ...
    nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```
```python
for sid in selected_ids:
    ...
    chunks.extend(speed[a:z] for a, z in zip(starts, ends))
```

iii. The AI’s notes say it already removed an earlier redundant full-session neural concatenation, so the remaining loops are mostly tied to variable-length per-trial outputs and large-file I/O.

## 12-c. What processing does the code repeat multiple times?

i. It repeats some session-level processing: `choose_behavior` is used during speed-threshold computation and again during per-session conversion; sample mode runs a separate neural shape scan before the actual conversion; and session metadata is assembled independently from the behavior loading used for conversion.

ii. 
```python
all_views = load_behavior_views(paths_all.keys())
...
speed_q, n_speed, speed_range = compute_speed_thresholds(selected, all_views, nfr_for_speed)
...
n, i, o, r, st = convert_session(
    sid, paths[sid], all_views[sid], shapes[sid], speed_q, day[sid]
)
```
```python
if args.sample:
    shapes, source_dtypes = inspect_neural_shapes(paths)
```

iii. The AI justified the sample-mode rescan as diagnostic and the reused behavior structures as a tradeoff to avoid rereading every behavior file per session.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extra metadata not needed by the downstream decoder, such as per-session `source_neural_dtypes`, rich `experiment_views` metadata, and detailed conversion statistics. In sample mode it also performs a full neural shape scan that is only for diagnostics. More broadly, the converter eagerly loads all behavior views up front even though downstream decoding only consumes the final trial-aligned arrays.

ii. 
```python
st["experiment_views"] = meta_views.get(sid, [])
st["source_neural_dtypes"] = source_dtypes[sid]
session_stats.append(st)
```
```python
if args.sample:
    shapes, source_dtypes = inspect_neural_shapes(paths)
```

iii. The notes describe these as validation, reproducibility, or plotting aids rather than decoder-required data transformations.
