# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the same three source directories as the reference: `/app/data/beh` (behavior, one `Beh_<exp_type>.npy` per experiment type plus the master index `Imaging_Exp_info.npy`), `/app/data/spk` (`<mouse>_<date>_<blk>_neural_data.npy`, a dict with `spks` = list of 3 per-plane neuron × frame arrays) and `/app/data/retinotopy` (`<mouse>_<date>_trans.npz`, field `iarea`). `Imaging_Exp_info.npy` is read first and every entry in every experiment-type group is expanded into a `SessionView` (exp_type, behavior key, `(mname, datexp, blk)` triplet, behavior file path). Behavior keys append `_<stimtype>` when the index entry has a `stimtype` field. Because one recording can be listed under several experiment types, the views are grouped by triplet and one "canonical" view per triplet is chosen, giving 89 sessions / 19 mice. Spikes and retinotopy are then loaded once per session inside the conversion loop.

ii.
```python
def load_exp_info() -> dict[str, list[dict[str, Any]]]:
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()

def collect_session_views(exp_info):
    views = defaultdict(list)
    for exp_type, records in exp_info.items():
        beh_path = os.path.join(BEH_DIR, f"Beh_{exp_type}.npy")
        beh_all = np.load(beh_path, allow_pickle=True).item()
        for rec in records:
            triplet = (rec["mname"], rec["datexp"], rec["blk"])
            key = "_".join(triplet)
            has_stimtype = "stimtype" in rec
            if has_stimtype:
                key = f"{key}_{rec['stimtype']}"
            if key not in beh_all:
                raise KeyError(f"Missing behavior key {key} in {beh_path}")
            views[triplet].append(SessionView(...))
    return views
```
```python
def load_spike_planes(triplet):
    mouse, datexp, blk = triplet
    path = os.path.join(SPK_DIR, f"{mouse}_{datexp}_{blk}_neural_data.npy")
    obj = np.load(path, allow_pickle=True).item()
    return [np.asarray(x) for x in obj["spks"]]

def load_iarea(triplet):
    mouse, datexp, _blk = triplet
    path = os.path.join(RET_DIR, f"{mouse}_{datexp}_trans.npz")
    return np.load(path, allow_pickle=True)["iarea"]
```

iii. From CONVERSION_NOTES Step 1/Step 5: the AI identified `load_exp_beh`, `load_spk` and `load_retino` in `/app/code/utils.py` as the reference loaders and states that its loading "matches `load_spk`, `load_exp_beh`, and `load_retino` by using the same raw files and the same retinotopy region grouping". It notes the reference code does not compute dF/F — `spks` are already deconvolved traces, so they are used directly.

## 1-b. How are the data split into subjects?

i. The subject is the mouse name `mname`, which is the first element of the `(mname, datexp, blk)` triplet that identifies each session. `subjects` is the sorted list of unique mouse names (19), and `subject_idx` is the index of each session's mouse into that list. Sessions are ordered by `(mouse, date, block)`.

ii.
```python
for triplet in sorted(views.keys(), key=lambda x: (x[0], parse_date(x[1]), int(x[2]))):
```
```python
subjects = sorted({triplet[0] for triplet, _view, _exp_types in sessions})
subject_lookup = {name: idx for idx, name in enumerate(subjects)}
...
data["subject_idx"].append(subject_lookup[mouse])
data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int16)
```

iii. The mouse identity is given explicitly by `mname` in `Imaging_Exp_info.npy`, so no split has to be inferred. The AI's Step 9 consistency table reports 19 subjects, matching the paper's "89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` recording. The AI observed that `Imaging_Exp_info.npy` contains 142 entries and 99 distinct behavior keys, but only 89 distinct recordings, because the same recording is listed under several experiment types and because `swap1`/`swap2` `stimtype` variants re-key the same recording. It therefore deduplicates to 89 recordings, and for each recording picks a single "canonical" behavior view using a deterministic priority: (1) most finite entries in `stim_id`, (2) most `UniqWalls`, (3) prefer the view without a `stimtype` suffix, then exp_type/key as tie-breakers.

ii.
```python
def behavior_view_priority(beh, has_stimtype):
    stim_id = np.asarray(beh["stim_id"], dtype=float)
    nfinite = int(np.isfinite(stim_id).sum())
    n_walls = int(len(beh["UniqWalls"]))
    return (nfinite, n_walls, 0 if has_stimtype else 1)

def choose_canonical_view(views):
    out = []
    for triplet in sorted(views.keys(), key=lambda x: (x[0], parse_date(x[1]), int(x[2]))):
        candidates = []
        for view in views[triplet]:
            beh = np.load(view.beh_path, allow_pickle=True).item()[view.key]
            candidates.append((behavior_view_priority(beh, view.has_stimtype), view))
        candidates.sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[1].exp_type, x[1].key), reverse=True)
        out.append((triplet, candidates[0][1], sorted(set(exp_types))))
    return out
```

iii. CONVERSION_NOTES Step 4/Step 5 Key Decision 1: "The paper reports 89 recordings in 19 mice, while code/data expose 142 experiment entries and 99 behavior keys because some recordings are reused under multiple experiment labels or swap interpretations. For the decoder dataset, the unique underlying recording is the biologically meaningful session." The canonical-view priority is intended to pick the richest / least-remapped behavior description of each recording.

## 1-d. How are the data split into trials?

i. Trials are the trials declared by the behavior file (`ntrials`), and the frames of trial *k* are the imaged frames labelled with that trial index in `ft_trInd` **and** inside the texture corridor (`ft_CorrSpc`). Behavior frame arrays are first truncated to the number of imaged frames `nfr`. `ft_trInd` contains NaNs (inter-trial frames), which are masked out explicitly before rounding to int. Trials keep their native, variable length; the 2 m grey space is excluded, so each trial is exactly one 0–4 m corridor traversal starting at corridor entry.

ii.
```python
frame_trial_raw = np.asarray(beh["ft_trInd"], dtype=float)[:nfr]
frame_trial = np.full(frame_trial_raw.shape, -1, dtype=int)
valid_frame_trial = np.isfinite(frame_trial_raw)
frame_trial[valid_frame_trial] = np.rint(frame_trial_raw[valid_frame_trial]).astype(int)
...
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
```

iii. Step 5 Key Decision 4: "The paper's main sensory analyses focus on the 0-4 m texture corridor, and the requested 4 spatial bins are explicitly 1 m each. Excluding the 2 m grey period makes the position output well defined and avoids mixing inter-trial activity into trial content." Step 5 Key Decision 5: native imaging frames are kept rather than the paper's position-interpolated tensors because the decoder task requires continuous time-to-cue and time-since-start signals.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, all of them minimal: (1) a trial with fewer than 2 corridor frames is dropped; (2) a trial whose `WallName` does not map to one of `circle/leaf/rock/wood` is dropped; (3) a trial with non-finite `SoundTime` or `Trial_start_time`, or any non-finite value in its neural/input/output/speed arrays, is dropped. A session left with fewer than 2 usable trials raises an error (it does not occur). **No trial-length outlier filter is applied.** In the full run 0 of 38,110 trials were dropped, so all trials of all 89 sessions are kept — including a 5,607-frame (29-minute) trial in `TX88_2022_07_19_1` in which the mouse stalls at 1–2 m. Consequently `T_max = 5607` bins and the timing inputs span [-1763 s, +724 s] (the human reference, which removes trials longer than the 99th percentile, has `T_max = 238` and timing inputs within ±75 s).

ii.
```python
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
    if mask.sum() < 2:
        skipped_trials += 1
        continue
    wall_name = str(beh["WallName"][trial_idx])
    family = family_from_wall_name(wall_name)
    if family not in family_lookup:
        skipped_trials += 1
        continue
    cue_time = float(beh["SoundTime"][trial_idx])
    start_time = float(beh["Trial_start_time"][trial_idx])
    if not np.isfinite(cue_time) or not np.isfinite(start_time):
        skipped_trials += 1
        continue
    ...
    if not (np.all(np.isfinite(neural_trial)) and np.all(np.isfinite(input_trial))
            and np.all(np.isfinite(output_trial)) and np.all(np.isfinite(speed_values))):
        skipped_trials += 1
        continue
if len(neural_trials) < 2:
    raise ValueError(f"Session {session_id} retained fewer than 2 usable trials.")
```

iii. The AI's stated sanity check was "no trials were silently lost during canonical-session conversion (`raw_trials_total == converted_trials_total`, skipped trials total = 0)", i.e. it treated *keeping every trial* as the success criterion. In Step 10 "Check 5: Check for edge cases" it explicitly investigated the longest trial and concluded: "the longest retained trial (`TX88_2022_07_19_1`, trial 391) has 5,607 corridor frames spanning 1,764.99 s with median running speed 0.0 ... Raw behavior reproduces the same prolonged stall, so this is a genuine behavioral edge case rather than an indexing bug." It verified the trial is real but did not consider excluding it.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<mouse>_<date>_<blk>_neural_data.npy` (a list of three per-imaging-plane neuron × frame arrays of deconvolved fluorescence), together with `iarea` from `retinotopy/<mouse>_<date>_trans.npz`, which gives the visual area of each neuron in the concatenated plane order. Behavior variables (`ft_WallID`, `ft_move`, `ft_CorrSpc`, `ft_GraySpc`, `stim_id`, `UniqWalls`) additionally enter the neuron-*selection* step (see 2-c). The code asserts that the total neuron count over the three planes equals `len(iarea)`.

ii.
```python
planes = load_spike_planes(triplet)     # list of (n_neurons_plane, n_frames)
iarea = load_iarea(triplet)
total_neurons_raw = sum(plane.shape[0] for plane in planes)
if total_neurons_raw != iarea.shape[0]:
    raise ValueError(f"Neuron count mismatch for {session_id}: ...")
```

iii. Step 1/Step 4: "Raw neural data are already stored as session arrays in `*_neural_data.npy`; the reference code does not compute `dF/F`. Instead it loads `['spks']` and concatenates them directly" and "All our analyses were based on deconvolved fluorescence traces" (paper). So `spks` is used as-is.

## 2-b. How is the `neural` data processed?

i. No signal processing at all is applied to the traces: no dF/F, no deconvolution, no smoothing, no z-scoring, no rebinning. For each trial, the selected neurons' columns for that trial's frame mask are taken and stored as `float16`. The only transformation of the neural *matrix* is the neuron subsetting described in 2-c: the selected rows are pulled out plane-by-plane (cast to `float32`), and the per-trial slices of each plane block are concatenated along the neuron axis. Trials keep their native length (no padding).

ii.
```python
selected_plane_arrays = [
    np.asarray(planes[plane_idx][local_idx], dtype=np.float32)
    for plane_idx, local_idx, _region_id in selected_blocks
]
del planes
...
neural_trial = np.concatenate(
    [selected_plane[:, mask] for selected_plane in selected_plane_arrays],
    axis=0,
).astype(np.float16, copy=False)
```

iii. Step 4 discrepancy table: "Use the stored `spks` matrices directly as neural activity. No additional fluorescence preprocessing or `dF/F` calculation is warranted." `float16` was chosen as a memory speed-up ("Store neural trial arrays as `float16` and inputs as `float32`").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are stacked.
(1) **Anatomical**, matching the reference: only neurons in the four grouped visual areas are eligible — `iarea == 8` → V1, `{0,1,2,9}` → mHV, `{5,6}` → lHV, `{3,4}` → aHV; everything else is discarded.
(2) **Selectivity-based subsampling, which the reference does not do**: for each session the AI computes the paper's d′ (the same formula as `utils.dprime`, `2(µ1−µ2)/(σ1+σ2)`) between the two "primary" stimuli (`stim_id == 2` vs `stim_id == 0`, else the two smallest finite `stim_id`s) over *running corridor frames* (`ft_WallID == stim & ft_CorrSpc & ft_move>0`); requires the neuron to be "corridor responsive" (mean activity in either stimulus > mean in grey space); prefers `|d′| ≥ 0.3` (falling back, per plane × region, to the sub-threshold corridor-responsive pool if no neuron in that plane/region clears 0.3); and then keeps at most **64 neurons per region**, ranked by `|d′|`. This yields ≤256 neurons/session: **22,163 neurons in total (249/session) out of 4,691,034 raw neurons**, i.e. ~0.5% of the data, versus 4.1 M kept by the reference.

ii.
```python
def compute_dprime(x1, x2):
    u1, u2 = np.nanmean(x1, axis=1), np.nanmean(x2, axis=1)
    s1, s2 = np.nanstd(x1, axis=1), np.nanstd(x2, axis=1)
    denom = s1 + s2
    out = np.zeros_like(u1, dtype=np.float32)
    valid = denom > 0
    out[valid] = (2.0 * (u1[valid] - u2[valid]) / denom[valid]).astype(np.float32)
    return out
```
```python
stim_a_fr = (ft_wall == stim_a) & ft_corr & ft_move
stim_b_fr = (ft_wall == stim_b) & ft_corr & ft_move
gray_fr   = ft_gray & ft_move
...
dp = compute_dprime(plane[:, stim_a_fr], plane[:, stim_b_fr])
corr_neu = (plane[:, stim_a_fr].mean(axis=1) > plane[:, gray_fr].mean(axis=1)) | \
           (plane[:, stim_b_fr].mean(axis=1) > plane[:, gray_fr].mean(axis=1))
for region_name in REGION_NAMES:
    region_pool = masks[region_name] & corr_neu & np.isfinite(dp)
    strong_pool = region_pool & (np.abs(dp) >= 0.3)
    chosen_pool = strong_pool if strong_pool.any() else region_pool
    ...
candidates.sort(key=lambda x: x[0], reverse=True)
chosen = candidates[:N_PER_REGION]        # N_PER_REGION = 64
```

iii. Step 5 Key Decisions 7 and 8: "Exporting all recorded neurons would create an impractically large dense dataset. To stay close to the paper, neuron selection will follow the paper's main stimulus-selectivity logic: compute d′ between the primary trained stimulus pair (`stim_id` 2 versus 0) on corridor running frames, keep corridor-responsive neurons with `|d′| >= 0.3`, and cap the export to up to 64 top-`|d′|` neurons per grouped visual region"; and "Restrict exported neurons to the four grouped visual regions ... the paper's primary anatomical groupings [which] avoid unlabeled/out-of-map units that the reference analyses often exclude." Step 10 Check 3 admits the deviation: "neuron filtering matches the paper's corridor-running selectivity logic (`ft_move > 0`, `ft_CorrSpc`, `d′ >= 0.3`) but caps each grouped region at 64 neurons for tractable decoder training." (Step 1 notes had recorded that the paper uses odd/even trial splits for neuron selection "to avoid circularity", but no such split is used here.)

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial is the contiguous run of frames labelled with that trial index and flagged as inside the texture corridor, so the first column of every trial matrix is the first imaged frame after corridor entry. Trials are variable length (they end where the traversal ends, at the grey space); nothing is padded or truncated to a common window. Metadata records `temporal_alignment_event = "corridor entry (trial start)"`, `off_start = 0.0`, `off_end = None`. All input/output streams are indexed with the *same* frame `mask`, so no cross-stream shift is possible.

ii.
```python
mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
neural_trial = np.concatenate([p[:, mask] for p in selected_plane_arrays], axis=0)
trial_times = ft[mask]
lick_out = lick_vec[mask]; pos_out = position_to_bin(ft_pos[mask]); speed_values = ft_speed[mask]
```
```python
"temporal_alignment_event": "corridor entry (trial start)",
"off_start": 0.0,
"off_end": None,
```

iii. Step 5 Key Decision 4/5 (corridor-only frames aligned to entry, native frames rather than position interpolation). Step 10 Check 3: "temporal alignment matches the reference use of `ft_trInd` plus `[:nfr]` trimming, but keeps native imaging frames instead of 0.1 m position interpolation because the decoder task requires continuous time-to-cue and time-since-start signals."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame; no rebinning, resampling or interpolation is performed. The bin size written to metadata is measured empirically: the median of the per-session medians of `diff(ft)`, converted from MATLAB datenum days to ms — **314.694 ms** (session medians 314.392–315.367 ms), consistent with the ~3.17 Hz imaging rate. Because no rebinning is done, the bin size is identical across trials and sessions to within the scanner's jitter.

ii.
```python
def compute_frame_timing(sessions):
    dts = []
    for triplet, view, _exp_types in sessions:
        beh = load_behavior(view)
        ft = np.asarray(beh["ft"], dtype=float)
        dts.append(float(np.nanmedian(np.diff(ft)) * SECONDS_PER_DAY * 1000.0))
    return float(np.median(dts)), dts
```
```python
"time_bin_size": float(time_bin_ms),
```

iii. Step 5 Key Decision 5: native imaging frames are preserved because "the decoder task requires time-to-cue and time-since-start signals. Native imaging frames preserve real trial timing better than the paper's position-interpolated tensors." The bin size is derived from the data rather than the nominal rate so that the reported value reflects what is actually in the file.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The per-trial `SoundTime` (absolute MATLAB datenum timestamp of the sound cue) and the per-frame timestamps `ft` (truncated to `nfr`). I verified against the raw behavior files that `SoundTime` is exactly `np.interp(SoundFr, arange(nfr), ft)`, i.e. numerically identical to the reference's interpolation of `SoundFr` onto the frame time axis (max difference 0.0 s).

ii.
```python
ft = np.asarray(beh["ft"], dtype=float)[:nfr]
...
trial_times = ft[mask]
cue_time = float(beh["SoundTime"][trial_idx])
```

iii. Step 5 mapping table: "Cue-alignment logic from `SoundFr`/`SoundTime` usage in `spk_2_cue` and methods text". Step 10 Check 3: "input construction uses the same raw timing variables the paper uses for cue/trial alignment (`SoundTime`, `Trial_start_time`, `ft`)."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every frame of the trial: `(SoundTime[trial] − ft[frame]) × 86400`, i.e. seconds remaining until the cue — positive before the cue and negative after it, matching the "time *to* sound cue" wording. Stored as `float32`, time-varying, one value per bin. No clipping, no binarization.

ii.
```python
SECONDS_PER_DAY = 24.0 * 3600.0
...
input_trial = np.vstack([
    (cue_time - trial_times) * SECONDS_PER_DAY,
    ...
]).astype(np.float32)
```

iii. Step 5: "Continuous, time-varying; negative after cue onset." The ×86400 conversion is needed because `ft`/`SoundTime` are MATLAB datenums in days.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from `ft[mask]`, where `mask` is the *same* boolean frame mask used to slice the neural columns of that trial, so the two arrays have identical length and are sample-for-sample aligned by imaging frame.

ii.
```python
mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
trial_times = ft[mask]
neural_trial = np.concatenate([p[:, mask] for p in selected_plane_arrays], axis=0)
```

iii. Every stream in this dataset is on the imaging-frame grid, so using one mask per trial guarantees alignment. The AI's Step 10 sanity check reconstructed trial 10 of `TX108_2023_03_13_1` from the raw files and found `input_allclose = True`, max abs diff 0.0.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The recording date `datexp` and block `blk` from `Imaging_Exp_info.npy`, grouped per mouse. No behavior variable carries a training-day counter, so the chronology of the recordings is used.

ii.
```python
def subject_day_map(sessions):
    per_subject = defaultdict(list)
    for triplet, _view, _exp_types in sessions:
        mouse, datexp, blk = triplet
        per_subject[mouse].append((parse_date(datexp), int(blk), triplet))
    out = {}
    for mouse, entries in per_subject.items():
        entries.sort()
        first_date = entries[0][0]
        for dt, blk, triplet in entries:
            out[triplet] = float((dt - first_date).days + 0.01 * (blk - 1))
    return out
```

iii. Step 5 mapping table: "No exact reference helper; derived from recording chronology in `Imaging_Exp_info.npy`. Best available continuous training-day proxy in imaging data." Key Decision 10: "The imaging data do not expose an exact per-trial 'training day' number, so subject-specific elapsed recording day is the most defensible continuous proxy."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. **Calendar days elapsed** since that mouse's first recording, plus `0.01 × (blk − 1)` to order two blocks recorded on the same day. The value is a per-session scalar broadcast across every bin of every trial of the session. Range over the dataset: 0 to 92 (the human reference instead counts *recorded sessions*, giving 0 to 7).

ii.
```python
out[triplet] = float((dt - first_date).days + 0.01 * (blk - 1))
```
```python
day_value = day_map[triplet]
input_trial = np.vstack([..., np.full(mask.sum(), day_value, dtype=np.float32), ...])
```

iii. As above: the AI treats elapsed calendar days from the first recording of each mouse as the most literal continuous reading of "day of training", and the `0.01 × blk` term as a tie-break that keeps same-day blocks distinct and ordered.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The per-trial `Trial_start_time` (absolute timestamp of corridor entry) and the frame timestamps `ft`. I verified against the raw files that `Trial_start_time` equals `np.interp(StartFr, arange(nfr), ft)` exactly, i.e. it is the same quantity the reference computes by interpolating `StartFr`.

ii.
```python
start_time = float(beh["Trial_start_time"][trial_idx])
trial_times = ft[mask]
```

iii. Step 5 mapping table: "`Trial_start_time`, `ft` → `input[2]` = `time_since_trial_start_s` ... Trial-start timing fields from behavior dictionary. Continuous, time-varying, aligned to corridor entry."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(ft[frame] − Trial_start_time[trial]) × 86400`, in seconds, positive after entry. Because the trial's frames all lie inside the corridor, the value starts at ~0 and increases monotonically. Stored as `float32`, time-varying. Observed range [0.0, 1765.2] s — the upper end coming from the un-filtered stalled trials (see 1-e); the reference's range is [0.0, 74.8] s.

ii.
```python
input_trial = np.vstack([
    (cue_time - trial_times) * SECONDS_PER_DAY,
    np.full(mask.sum(), day_value, dtype=np.float32),
    (trial_times - start_time) * SECONDS_PER_DAY,
    np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32),
]).astype(np.float32)
```

iii. Simple time difference against the corridor-entry timestamp, converted from datenum days to seconds; the sign convention follows the "time *since* start" wording.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Identically to 3-c: it is `ft[mask]` minus a scalar, with `mask` the same frame mask used for the neural columns, so it is one value per neural time bin, aligned by imaging frame.

ii.
```python
trial_times = ft[mask]
(trial_times - start_time) * SECONDS_PER_DAY
```

iii. All streams are on the imaging-frame grid; the AI's Step 10 raw-data reconstruction of a trial matched the saved inputs exactly (`np.allclose`, max abs diff 0.0).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew`, the per-trial flag marking trials run in the rewarded corridor.

ii.
```python
np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32)
```

iii. Step 5 mapping table: "`isRew` → `input[3]` = `reward_available`. Boolean per trial cast to {0,1}, repeated across frames ... For unsupervised sessions this remains 0 because no reward is available."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast: `bool(isRew[trial])` → `{0.0, 1.0}` float, broadcast across all bins of the trial. It is 0 for every trial of naive/unsupervised sessions and varies by trial in the supervised task sessions (the verification log shows per-session ranges of `[0,0]` or `[0,1]`).

ii.
```python
np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32)
```

iii. The flag is already exactly the requested variable ("1 if in rewarded corridor, 0 if not, discrete, per-trial"), so no processing is warranted.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName`, the per-trial name of the corridor wall texture (`circle1..3`, `leaf1..3`, `leaf1_swap1/2`, `rock1/2`, `wood1/2/5`, `wood1_swap1/2`). `stim_id`/`TrialStim` are deliberately *not* used for the label.

ii.
```python
wall_name = str(beh["WallName"][trial_idx])
family = family_from_wall_name(wall_name)
```

iii. Step 5 Key Decision 2: "Duplicate views of the same recording often differ only in `stim_id` remapping, while `WallName` stays fixed. This avoids double-counting and keeps labels tied to actual presented stimuli."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The exemplar name is collapsed to its texture family by prefix matching against `("circle", "leaf", "rock", "wood", "brick")` (falling back to the leading alphabetic run). `brick` is explicitly removed from the label vocabulary, so `output_values[0] = ["circle", "leaf", "rock", "wood"]` and any trial whose family is outside those four would be dropped (none are, in practice). Swap variants (`leaf1_swap1`) collapse to their base family via the prefix rule. The per-trial integer label is broadcast across all bins of the trial and stored as `int16`. Resulting distribution: circle 0.320, leaf 0.484, rock 0.081, wood 0.116 (reference: 0.312 / 0.481 / 0.082 / 0.125).

ii.
```python
def family_from_wall_name(name: str) -> str:
    lowered = name.lower()
    for prefix in ("circle", "leaf", "rock", "wood", "brick"):
        if lowered.startswith(prefix):
            return prefix
    match = re.match(r"([a-zA-Z]+)", lowered)
    if match:
        return match.group(1)
    return lowered

family_values = ["circle", "leaf", "rock", "wood", "brick"]
family_values = [x for x in family_values if x != "brick"]
family_lookup = {name: idx for idx, name in enumerate(family_values)}
...
stim_out = np.full(mask.sum(), family_lookup[family], dtype=np.int16)
```

iii. Step 5 Key Decision 3: "The output requested is stimulus category ('circle, leaf, etc.'). Mapping exemplar/swap names to `circle`, `leaf`, `rock`, and `wood` better matches the task specification and paper framing."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr`, the (fractional) imaging-frame number of each lick in the session.

ii.
```python
lick_fr = np.asarray(beh["LickFr"], dtype=float)
```

iii. Step 5 mapping table: "`LickFr`, `LickTrind` → `output[1]` = `licking`. Convert lick events to binary per-frame vector within each trial after frame trimming; value 1 if one or more licks assigned to frame. ... `spk_2_firstLick`, `spk_2_cue` cast lick frame indices to integers."

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is built once: each lick's frame number is floored to the frame it falls in, licks outside `[0, nfr)` are discarded, and the corresponding frames are set to 1 (a frame with several licks is still 1). The trial's values are the entries of that vector at the trial's frames. Overall distribution: 96.5% no-lick / 3.5% lick (reference 95.9% / 4.1% — the small difference follows from the extra stalled frames the AI retains). Many naive/unsupervised sessions are 100% no-lick.

ii.
```python
def build_lick_frame_vector(beh, nfr):
    lick_vec = np.zeros(nfr, dtype=np.int8)
    lick_fr = np.asarray(beh["LickFr"], dtype=float)
    lick_idx = np.asarray(np.floor(lick_fr), dtype=int)
    valid = (lick_idx >= 0) & (lick_idx < nfr)
    lick_vec[lick_idx[valid]] = 1
    return lick_vec
```
```python
lick_out = lick_vec[mask].astype(np.int16)
```

iii. The decoder spec asks for binary, time-varying licking; the raw data are event frame numbers, so flooring to the containing frame is the natural discretization. The `valid` mask implements the AI's Step 4 finding that behavior can run past the imaging.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the lick vector is already on the neural grid; the trial's values are taken with the same `mask` used for the neural columns, so it has exactly the trial's length and is aligned bin-for-bin.

ii.
```python
lick_vec = build_lick_frame_vector(beh, nfr)
...
lick_out = lick_vec[mask].astype(np.int16)
neural_trial = np.concatenate([p[:, mask] for p in selected_plane_arrays], axis=0)
```

iii. Same frame-grid argument as the other streams; the Step 10 sanity check verified the saved outputs against a raw-data reconstruction (`output_allclose = True`).

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the per-imaging-frame position in the corridor in decimeters (0–40 across the 4 m texture, 40–60 through the 2 m grey space), truncated to `nfr`.

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=float)[:nfr]
```

iii. Step 4 resolved the units: "Data use decimeter units. Therefore 60 bins = 6 m full corridor, 40 bins = 4 m texture region, 20 bins = 2 m grey space."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 (decimeters → meters), floored to an integer, and clipped to `[0, 3]`; stored as `int16` and broadcast to nothing (it is genuinely time-varying, one label per bin). Because trials only contain texture-corridor frames, positions essentially never exceed 40 dm, so the clip only guards the endpoint and any marginally negative position at entry.

ii.
```python
def position_to_bin(pos):
    bins = np.floor(np.asarray(pos, dtype=float) / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
...
pos_out = position_to_bin(ft_pos[mask])
```

iii. Step 5 mapping table: "Discretize position into 4 fixed 1 m bins over the 0-4 m texture corridor: `[0,10), [10,20), [20,30), [30,40+]` in data units. Paper methods define 0-4 m texture region; code frequently uses `:40` bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length 1 m spatial bins with hard edges at 0/10/20/30/40 dm, labelled `["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"]` — exactly as the decoder spec requires ("4 equal-length, 1-m-long spatial bins"). The bins are not equal-occupancy; the observed distribution is [0.285, 0.233, 0.236, 0.246] (reference [0.254, 0.242, 0.247, 0.257]; the excess in bin 0 comes from the retained stall trials).

ii.
```python
"output_values": [..., ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"], ...]
bins = np.floor(np.asarray(pos, dtype=float) / 10.0).astype(np.int16)
return np.clip(bins, 0, 3)
```

iii. The threshold is dictated by the Decoder Task specification, and the decimeter unit of `ft_Pos` makes the division by 10 exact.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one entry per imaging frame, and the trial's positions are taken with the same `mask` as the neural columns, so it is aligned bin-for-bin and has the trial's length.

ii.
```python
pos_out = position_to_bin(ft_pos[mask])
neural_trial = np.concatenate([p[:, mask] for p in selected_plane_arrays], axis=0)
```

iii. Common frame grid for all streams; verified by the Step 10 raw-reconstruction spot check.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the running speed at each imaging frame, truncated to `nfr`.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)[:nfr]
...
speed_values = np.asarray(ft_speed[mask], dtype=np.float32)
```

iii. Step 5 mapping table: "`ft_RunSpeed` within corridor frames → `output[3]` = `running_speed_bin` ... same framewise variable source."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The raw speeds of every retained frame are collected during the session loop and kept aside; after **all** sessions are converted, the pooled vector (1,373,170 frames) is rank-ordered and split into four equal-count groups by `np.array_split` of the sort order, and the resulting bin labels are written back into the per-trial output arrays in the order they were collected (with a checksum that the cursor consumed exactly all frames). The split is therefore **global across the whole dataset**, not per session (the reference computes quartiles within each session). Ties at exactly 0 cm/s (~30% of frames) are broken by the stable `mergesort` order, i.e. by frame index.

ii.
```python
def assign_rank_speed_bins(all_speed):
    order = np.argsort(all_speed, kind="mergesort")
    speed_bins = np.empty(all_speed.size, dtype=np.int16)
    for bin_idx, idx_chunk in enumerate(np.array_split(order, 4)):
        speed_bins[idx_chunk] = bin_idx
    edges = np.quantile(all_speed, [0.25, 0.5, 0.75]).astype(np.float32)
    ...
```
```python
all_speed_values = np.concatenate([t for s in speed_value_trials_per_session for t in s])
all_speed_bins, speed_edges, speed_value_ranges = assign_rank_speed_bins(all_speed_values)
cursor = 0
for session_idx, session_speed in enumerate(speed_value_trials_per_session):
    for trial_idx, trial_speed in enumerate(session_speed):
        n_time = int(trial_speed.size)
        trial_speed_bins = all_speed_bins[cursor: cursor + n_time]
        cursor += n_time
        data["output"][session_idx][trial_idx] = np.vstack(
            [data["output"][session_idx][trial_idx], trial_speed_bins[np.newaxis, :]]).astype(np.int16)
if cursor != int(all_speed_bins.size):
    raise ValueError("Global speed-bin assignment did not consume all frames.")
```

iii. Step 5 Key Decision 9: "This follows the decoder specification exactly and avoids session-specific binning that would confound cross-session decoding. Because many corridor frames have tied zero speed, the final implementation assigns quartiles by global rank after conversion rather than by simple thresholding." Step 10 Issues: threshold-based quantiles gave `[0.302, 0.198, 0.250, 0.250]`; the rank method gives exactly `[0.250, 0.250, 0.250, 0.250]`.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Not by value thresholds but by **rank**: four equal-count bins (`q1`–`q4`), each holding exactly 25% of all retained corridor frames (343,293 / 343,293 / 343,292 / 343,292). Value-based quartile edges are computed and stored in metadata (`speed_bin_edges`, `speed_bin_value_ranges`) for reference but are not used to assign labels. Because the ranking is global, per-session occupancy is strongly unequal — e.g. one session's fractions are [0.677, 0.186, 0.114, 0.023] and another has 0.000 frames in the top bin.

ii.
```python
order = np.argsort(all_speed, kind="mergesort")
for bin_idx, idx_chunk in enumerate(np.array_split(order, 4)):
    speed_bins[idx_chunk] = bin_idx
```
```python
"output_values": [..., ["q1", "q2", "q3", "q4"]],
"speed_binning_method": "global_rank_quartiles_over_all_converted_corridor_frames",
```

iii. Required by the Decoder Task ("Running speed discretized into 4 bins, each corresponding to 25% of the data"); rank-based assignment is the only way to hit exactly 25% given the large mass of tied zero-speed frames.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is sampled per imaging frame; the per-trial speed values are taken with the same `mask` as the neural columns, and the bin labels are written back into each trial at the same positions via a running cursor over the concatenation order, with an assertion that the cursor consumes exactly all frames. So the speed row has the trial's length and is aligned bin-for-bin.

ii.
```python
speed_values = np.asarray(ft_speed[mask], dtype=np.float32)
...
trial_speed_bins = all_speed_bins[cursor: cursor + n_time]
data["output"][session_idx][trial_idx] = np.vstack(
    [data["output"][session_idx][trial_idx], trial_speed_bins[np.newaxis, :]]).astype(np.int16)
```

iii. Same frame grid as all other streams; the cursor checksum plus the Step 10 raw-data spot check (`output_allclose = True`) are the AI's verification that the write-back preserves order.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several small defensive measures, all documented:
- **Behavior longer than imaging**: every frame-indexed behavior array is truncated to `nfr = planes[0].shape[1]`, both by a generic trimming helper and again by explicit `[:nfr]` slices at use.
- **NaNs in `ft_trInd`** (inter-trial frames): masked out with `np.isfinite` before rounding to int, so they never join a trial.
- **Licks after the last imaged frame**: dropped by the `0 <= idx < nfr` validity mask.
- **Non-finite cue/start times or non-finite values anywhere in a trial's tensors**: the trial is skipped and counted in `skipped_ntrials`.
- **Structural consistency**: the code raises if a behavior key named in the index is missing, if the spike neuron count disagrees with `len(iarea)`, if no neuron is selected, or if a session retains fewer than 2 trials (this last one aborts the whole run rather than skipping the session).
In the full run none of the data-level guards fired: 0 trials skipped.

ii.
```python
def trim_behavior_framewise(beh, nfr):
    trimmed = dict(beh)
    for key, value in beh.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] >= nfr:
            if key.startswith("ft_") or key in {"ft", "AftCueFr", "BefCueFr"}:
                trimmed[key] = value[:nfr]
    return trimmed
```
```python
valid_frame_trial = np.isfinite(frame_trial_raw)
...
valid = (lick_idx >= 0) & (lick_idx < nfr)
...
if not np.isfinite(cue_time) or not np.isfinite(start_time):
    skipped_trials += 1
    continue
```

iii. Step 4 discrepancy table: "Spot checks across 10 evenly spaced recordings showed `len(ft)` exceeds spike frame count by small offsets (`+1` to `+3` frames) ... Follow reference behavior exactly: neural frame count is authoritative for framewise alignment, and behavior frame arrays should be trimmed to `nfr` before framewise indexing." Step 5 Key Decision 6 repeats this. The finite-value guards are described as edge-case robustness in Step 10 Check 5.

## 12-a. What are the most time-consuming steps of the code?

i. Measured from `conversion_full_out.txt`: total 1,373.6 s, of which 1,357.5 s (98.8%) is the per-session loop and ~16 s the pre-passes. Within a session the dominant costs are (1) reading the ~4.5 GB spike file (`np.load` of the three planes; 405 GB across the dataset) — unavoidable and also the reference's bottleneck; (2) the AI-specific neuron-selection pass, which boolean-column-indexes every full plane three times (`plane[:, stim_a_fr]`, `plane[:, stim_b_fr]`, `plane[:, gray_fr]`) to compute d′ and the corridor-responsiveness test, materialising multi-GB temporaries for 20k–90k neurons; and (3) the per-trial `np.concatenate` of the selected plane blocks, repeated for every one of the up-to-789 trials in a session. The AI itself noted a single pathological session (`TX61_2021_06_19_1`, 86.5 s) and observed the run take 17–37 s/session under load.

ii.
```python
obj = np.load(path, allow_pickle=True).item()     # ~4.5 GB per session
return [np.asarray(x) for x in obj["spks"]]
```
```python
dp = compute_dprime(plane[:, stim_a_fr], plane[:, stim_b_fr])
corr_neu = (plane[:, stim_a_fr].mean(axis=1) > plane[:, gray_fr].mean(axis=1)) | \
           (plane[:, stim_b_fr].mean(axis=1) > plane[:, gray_fr].mean(axis=1))
```

iii. Step 6: "Raw dense spike files are very large; naive export of all neurons would be impractical in both time and output size." Speed-ups claimed: processing sessions one at a time, plane-wise selection instead of one giant concatenated session matrix, capping neurons per region, `float16` storage, and a single vectorized post-pass for the speed quartiles.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain:
- **The per-trial loop** (`for trial_idx in range(ntrials)`) rescans the full `frame_trial`/`ft_corr` arrays to build `mask` for each of up to 789 trials, instead of grouping all frames by trial index in one pass (e.g. `np.argsort`/`np.split` on `ft_trInd`). This is the same loop the human reference flags.
- **The candidate-collection loop in `select_neurons`** builds a Python tuple `(score, plane_idx, idx)` for every surviving neuron of every plane and region and then sorts that Python list; with tens of thousands of candidates per session this is pure-Python work that `np.argpartition` on the `|d′|` array would do in one vectorized call.
- **Per-trial neural assembly**: `np.concatenate([p[:, mask] for p in selected_plane_arrays])` is executed once per trial; the ≤256 selected rows could be concatenated into one `(n_sel, nfr)` array once per session and then sliced. The AI tried exactly this and reverted it after observing a sample-run regression, so the redundant concatenation remains.
Minor: `mask.sum()` is recomputed four times per trial.

ii.
```python
for trial_idx in range(ntrials):
    mask = valid_frame_trial & (frame_trial == trial_idx) & ft_corr
    ...
    input_trial = np.vstack([
        (cue_time - trial_times) * SECONDS_PER_DAY,
        np.full(mask.sum(), day_value, dtype=np.float32),
        (trial_times - start_time) * SECONDS_PER_DAY,
        np.full(mask.sum(), float(bool(beh["isRew"][trial_idx])), dtype=np.float32)])
```
```python
candidate_by_region[region_name].extend(
    (float(score), plane_idx, int(idx)) for score, idx in zip(scores.tolist(), local_idx.tolist()))
...
candidates.sort(key=lambda x: x[0], reverse=True)
```

iii. The AI does not identify any remaining vectorizable loop in CONVERSION_NOTES; Step 6 only lists the speed-ups it added, and Step 10 records that "Attempted session-level neural-trace pre-concatenation caused a runtime regression during sample conversion. Reverted to the earlier plane-wise slicing implementation because it was stable and preserved correctness."

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are unpickled far more often than necessary — four separate passes over the same data: `collect_session_views` loads each of the 23 `Beh_*.npy` files once, `choose_canonical_view` re-loads the *entire* behavior file once per view (142 loads), `compute_frame_timing` loads one behavior per session (89 loads), and `convert_dataset` loads it again for each session (89 loads); in `--sample` mode `choose_sample_sessions` adds another 89. That is ~340 full unpicklings of a 6.6 GB corpus to read a handful of fields. (Empirically these loads are fast from page cache — ~16 s of the 1,374 s run — so the waste is real but not dominant.) Within a session, the per-trial `np.concatenate` of the selected plane blocks and the repeated `mask.sum()` are the other repeated work. The human reference reads each behavior file exactly once and reports no repeated processing.

ii.
```python
beh_all = np.load(beh_path, allow_pickle=True).item()          # collect_session_views
...
beh = np.load(view.beh_path, allow_pickle=True).item()[view.key]   # choose_canonical_view, per view
...
def load_behavior(view):                                        # compute_frame_timing AND convert_dataset
    return np.load(view.beh_path, allow_pickle=True).item()[view.key]
```

iii. Not identified in CONVERSION_NOTES. The AI's stated I/O strategy is only "Process sessions one at a time to avoid holding multiple raw recordings in memory" and "Use a lightweight pre-pass only for frame-interval estimates" — the pre-pass is in fact a full re-read of every behavior file.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items are computed and then never used, or used only for a scalar:
- **`compute_frame_timing`** loads all 89 behavior files and computes 89 per-session median frame intervals solely to write one number (`time_bin_size`) into metadata; the per-session list `frame_dt_ms_all` is only printed.
- **`assign_rank_speed_bins` returns value-based quartile `edges` and per-bin `value_ranges`** which are stored in metadata but play no role in labelling (labels come from the rank split) and are not consumed downstream.
- **`trim_behavior_framewise`** copies and trims *every* `ft_*` array in the behavior dict, though only `ft`, `ft_Pos`, `ft_RunSpeed`, `ft_CorrSpc`, `ft_trInd`, `ft_WallID`, `ft_move`, `ft_GraySpc` are used — and the code then re-slices `[:nfr]` on top of the already-trimmed arrays.
- **`family_values`** is built with `"brick"` and then immediately filtered to remove it.
- Most significantly, **d′ and corridor-responsiveness are computed for all 4.69 M neurons and then 99.5% of them are discarded**; the selection statistics (`corridor_responsive_fraction`, `abs_dprime_ge_0p3_fraction`) are recorded per session but never used again. This work is intrinsic to the AI's chosen curation, but the reference performs no analogous pass at all.

ii.
```python
time_bin_ms, frame_dt_ms_all = compute_frame_timing(canonical_sessions)   # 89 file loads for one scalar
```
```python
edges = np.quantile(all_speed, [0.25, 0.5, 0.75]).astype(np.float32)   # never used for labelling
data["metadata"]["speed_bin_edges"] = speed_edges.astype(float).tolist()
```
```python
family_values = ["circle", "leaf", "rock", "wood", "brick"]
family_values = [x for x in family_values if x != "brick"]
```

iii. Not discussed in CONVERSION_NOTES; the AI's efficiency notes focus only on the speed-ups it added. The metadata extras (`speed_bin_edges`, `selection_meta`, per-session `skipped_ntrials`) are presented as documentation/provenance rather than as processing inputs.
