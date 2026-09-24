# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats `data/beh/Imaging_Exp_info.npy` as the authoritative index of imaging recordings, and reads three directories: `data/beh/` (behavior, one `Beh_<exp_type>.npy` per figure/experiment group), `data/spk/<session>_neural_data.npy` (deconvolved traces, one file per session, a list of per-plane arrays), and `data/retinotopy/<mouse>_<date>_trans.npz` (`iarea`, the visual area of each neuron). Rather than mapping each index entry to the behavior file it was listed under, the AI eagerly loads *every* `Beh_*.npy` into memory, keys every behavior entry by its session base (`mname_YYYY_MM_DD_blk`, parsed off the first five underscore fields of the behavior key), and keeps one "canonical" entry per session (the first file in alphabetical order). Duplicate entries of the same recording are cross-checked field by field to confirm they are the same recording. Neural and retinotopy files are then read once per session inside `convert_session`.

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

def session_base_from_key(key):
    return "_".join(str(key).split("_")[:5])
```
```python
    spk_path = os.path.join(root, "data", "spk", f"{session['base']}_neural_data.npy")
    ret_path = os.path.join(root, "data", "retinotopy", f"{session['mname']}_{session['datexp']}_trans.npz")

    spk_obj = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
    ret = np.load(ret_path, allow_pickle=True)
    iarea = np.asarray(ret["iarea"], dtype=np.int64)
```

iii. From the trajectory (steps 20–39): the AI first traced the authors' `data_process_script.ipynb` and `utils.py` to learn the field layout, then discovered that "some recordings are reused across figure groupings", so it decided "I should not trust the figure-specific group names as the primary dataset split". It verified that duplicated behavior objects are literally identical ("ordinary duplicate group assignments are literally the same behavior object reused across figures") and that `swap1`/`swap2` pairs differ only in a partially masked `stim_id`, concluding it "can safely deduplicate to 89 unique recordings and keep the full trial set for each recording".

## 1-b. How are the data split into subjects?

i. Subjects are the `mname` field of each index entry. Sessions are grouped by mouse in order of first appearance in the deduplicated session list; `subjects` is that list of 19 unique names and `subject_idx` is each session's index into it. Result: 89 sessions over 19 mice, matching the paper.

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

iii. The mouse name is given directly by the index, so nothing has to be inferred. The AI used the subject count as a sanity check against the paper: "Unique subject count matches the paper: **19**" (CONVERSION_NOTES.md; printed as a "Reference check" at the top of the conversion log).

## 1-c. How are the data split into sessions?

i. A session is the triple `(mname, datexp, blk)` from `Imaging_Exp_info.npy`; the first occurrence wins and later duplicates under other experiment types are skipped. The base string `mname_datexp_blk` names the spike file and keys the behavior lookup. This yields 89 unique recordings. The AI additionally verified, across all behavior files, that duplicate entries for the same base agree on `ntrials`, `WallName`, `ft_trInd`, `ft_Pos`, `ft_CorrSpc`, `ft_move`, `ft_RunSpeed`, `StartFr`, `EndFr`, `SoundFr` and `LickFr` (0 mismatches reported).

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
            sessions.append({... "base": f"{item['mname']}_{item['datexp']}_{item['blk']}" ...})
    return sessions
```
```python
            ref = canonical[base][2]
            mismatch = []
            for key in keys_to_compare:
                if key not in beh or key not in ref:
                    continue
                if not arrays_equal(ref[key], beh[key]):
                    mismatch.append(key)
```

iii. Trajectory step 39: "The `swap1/swap2` case turned out to be another figure-analysis duplication, not separate trial subsets... So I can safely deduplicate to 89 unique recordings." The AI also caught and fixed a bug in which its session base dropped the block number (step 92: "behavior keys include three date components plus the block, so my canonical session base was dropping the block number").

## 1-d. How are the data split into trials?

i. Trials are taken as the data declares them: `range(beh['ntrials'])`, with each neural frame assigned to a trial by `ft_trInd`. Within a trial, the AI keeps only frames that are (a) inside the textured corridor (`ft_CorrSpc`) **and** (b) frames where the VR was advancing (`ft_move > 0`), i.e. the mouse was running. Frames are then grouped in order into 3-frame decoder bins. Trials are variable length (4 to 60 bins; mean 7.5 bins ≈ 22 retained frames); no padding or truncation. This differs from the human reference, which keeps *every* corridor frame of the traversal (mean 32.6 frames/trial). The `ft_move` restriction drops ~27–61% of corridor frames depending on session, so the retained frames of a trial are not temporally contiguous.

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

def chunk_indices(indices, chunk_size):
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
```

iii. The `ft_move > 0` criterion comes straight from the authors' code: the AI grepped `utils.py` and found `VRmove = beh['ft_move'][:nfr]>0` / `fr_valid = VRmove & isCorridor  # only use activity inside the texture area plus mouse is running (VR moving)` at five call sites, and noted the paper "explicitly restrict[s] to running time points" (CONVERSION_NOTES.md). Step 46: "segment trials by `StartFr`→`EndFr` aligned to corridor entry, and restrict each trial's retained frames to `ft_move>0` to match the paper's 'running-only' analyses." It also checked (step 76) that no trial loses all of its frames under this rule: "zero trials 0 of 38110".

## 1-e. How are trials filtered based on quality controls?

i. Essentially none. A trial is dropped only if it has zero retained frames (`if not chunks: continue`) — which the AI verified never happens, so all 38,110 trials are kept. There is no trial-duration outlier filter, no minimum-trial check per session, and no session-level exclusion. The `ft_move > 0` frame filter incidentally shortens "parked mouse" trials (max 60 bins), but the *wall-clock span* of such trials is untouched: the converted `time_since_trial_start` input reaches 1764.8 s (≈29 min) and `time_to_sound_cue` spans [−1763.0, +723.2] s, versus the reference's [0, 74.8] s and [−72.2, +73.4] s, because bins in those trials are separated by minutes of dropped stationary frames. The human reference drops trials longer than the 99th percentile of traversal length (382 of 38,110 trials).

ii.
```python
    for tr, frame_idx in enumerate(frames_by_trial):
        chunks = chunk_indices(frame_idx, frames_per_bin)
        if not chunks:
            continue
```
(no other trial-level filter exists in the script)

iii. The AI's justification is implicit: it verified that every trial retains at least one moving corridor frame ("zero trials 0 of 38110", step 76) and that per-trial retained frame counts looked plausible (step 73: "kept frame count mean/med/min/max 22.6 / 22 / 20 / 31"), and treated the running restriction as the data-quality criterion the paper itself applies. Long stationary traversals are never discussed anywhere in the trajectory or notes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `data/spk/<session>_neural_data.npy`, a list of per-imaging-plane (neurons × frames) deconvolved arrays, concatenated along the neuron axis. Area labels come from `iarea` in `data/retinotopy/<mouse>_<date>_trans.npz`, truncated to the number of concatenated rows. Identical sources to the reference.

ii.
```python
    spk_obj = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
    ret = np.load(ret_path, allow_pickle=True)
    iarea = np.asarray(ret["iarea"], dtype=np.int64)
    ...
    region_idx_full = coarse_region_indices(iarea[: spk.shape[0]])
```

iii. CONVERSION_NOTES.md: "Imaging traces are deconvolved Suite2p outputs"; the AI verified (step 48) that the number of concatenated `spks` rows equals `len(iarea)` for every session, so plane order and retinotopy order correspond.

## 2-b. How is the `neural` data processed?

i. Two transformations are applied on top of the raw deconvolved traces. (1) Neuron subselection: exactly 128 neurons per session (see 2-c). (2) Temporal averaging: the retained (corridor + running) frames of a trial are grouped in order into chunks of 3 and each chunk is averaged, giving one column per decoder bin. Output dtype is float32; trials are stored at their own length (n_neurons=128 × n_bins). No normalization, z-scoring or smoothing.

ii.
```python
        for chunk in chunks:
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            ...
            trial_neural.append(neural_bin)
        neural_trials.append(np.stack(trial_neural, axis=1).astype(np.float32))
```

iii. From CONVERSION_NOTES.md: "A literal all-neuron, frame-level conversion would be too large for both the required pickle output and the provided decoder", so the AI applied "a documented reduction for decoder tractability: coarse time binning after frame selection and a fixed-size visual-cortex neuron subset per session" (step 83). The traces themselves are left as deconvolved values because "Imaging traces are deconvolved Suite2p outputs", i.e. already the quantity the paper analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Area assignment, same as the reference: `iarea` codes are mapped to V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4), and neurons with no coarse-area assignment get −1 and are excluded. (2) An additional, paper-external subsampling step: within each session the AI keeps only **128** neurons, allocated across the four areas in proportion to their available counts (largest-remainder rounding) and, within each area, ranked by *variance* of the deconvolved trace computed over up to 2048 of the retained frames. This drops ~99.7% of the recorded neurons: the reference dataset averages 46,128 neurons per session (min 17,363, max 78,815); the AI's dataset has exactly 128 in every session.

ii.
```python
def coarse_region_indices(iarea):
    out = np.full(iarea.shape[0], -1, dtype=np.int64)
    out[iarea == 8] = 0
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
    out[(iarea == 5) | (iarea == 6)] = 2
    out[(iarea == 3) | (iarea == 4)] = 3
    return out

def select_neurons(spk, region_idx_full, selected_frames, neurons_per_session):
    region_counts = [(region_idx_full == ridx).sum() for ridx in range(len(REGION_NAMES))]
    targets = proportional_region_targets(region_counts, neurons_per_session)
    var = variance_over_columns(spk, selected_frames)
    selected = []
    for ridx, target in enumerate(targets):
        ...
        order = np.argsort(var[candidates])[::-1]
        picked = candidates[order[:target]]
        selected.append(np.sort(picked))
```
```python
    parser.add_argument("--neurons-per-session", type=int, default=128, ...)
```

iii. CONVERSION_NOTES.md: "Raw session files contain very large neuron counts... A literal all-neuron, frame-level conversion would be too large for both the required pickle output and the provided decoder. For decoder tractability, I retained a fixed-size subset of 128 neurons per session: restricted to retinotopically assigned visual cortex neurons, proportionally allocated across V1, mHV, lHV, aHV, ranked within area by variance over retained running/corridor frames." Trajectory step 67: "I hit the real scaling constraint... the intended conversion must include an explicit reduction step." Step 123 documents that the variance ranking was further capped to a fixed subset of frames purely to save wall time. The original per-session neuron counts are preserved in `metadata['session_info']`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry / trial start: a trial's columns are the frames labelled with that trial by `ft_trInd` inside the textured corridor, taken in order from the first such frame, so bin 0 is the first running frame after corridor entry (`time_since_trial_start` minimum over the dataset is 0.2 s). Trials keep their own length (no common window, no padding); `off_start = 0.0`, `off_end = None`. All other streams are indexed with exactly the same frame indices, so neural, input and output are aligned by construction. The one deviation from the reference is that the frames making up a trial are only the running frames, so the bins of a trial are not uniformly spaced in real time.

ii.
```python
        frames_by_trial = trial_frame_indices(beh, nfr)
        ...
        for chunk in chunks:
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
```
```python
            "temporal_alignment_event": "corridor entry / trial start",
            "off_start": 0.0,
            "off_end": None,
```

iii. CONVERSION_NOTES.md: "Temporal alignment event: corridor entry / trial start. Trial identity comes from `ft_trInd`." All streams are frame-indexed in the raw data, so the AI aligns everything by shared frame indices rather than by re-interpolating times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — rebinning by a factor of 3. The imaging frame interval is measured empirically as the median of per-session median `diff(ft)` (314.694 ms, consistent with the 3.17 Hz rate), and the decoder bin is defined as 3 frames, so `metadata['time_bin_size'] = 944.08` ms. The reference applies no rebinning and stores native 315 ms frames. Two caveats: the final chunk of a trial may contain 1 or 2 frames rather than 3, and because non-running frames were removed before chunking, a "944 ms" bin can in fact span many seconds (or minutes) of wall clock, so the recorded bin size is nominal rather than actual.

ii.
```python
def compute_frame_dt_ms(canonical_lookup, sessions):
    dts = []
    for session in sessions:
        ...
        diffs = np.diff(ft) * SECONDS_PER_DAY * 1000.0
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            dts.append(np.median(diffs))
    return float(np.median(dts))
```
```python
            "time_bin_size": float(frame_dt_ms * args.frames_per_bin),
            "time_binning_rule": f"Average every {args.frames_per_bin} retained imaging frames into one decoder time bin.",
```

iii. Rebinning is part of the "documented reduction for decoder tractability" (step 83/85: "3-frame temporal binning so the resulting dataset is small enough to validate with the provided decoder"). The frame interval is measured from the data rather than hard-coded, and reported as a sanity check ("Global median imaging frame step (ms): 314.694").

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `beh['SoundTime']` (the MATLAB datenum timestamp of the cue on each trial) and `beh['ft']` (the datenum timestamp of every imaging frame). The reference instead interpolates the fractional frame number `SoundFr` onto the frame-time axis; I verified numerically that the two are identical to floating-point precision (median difference 0.0 s), and that `SoundTime` is never NaN anywhere in the dataset (0 of 63,177 trials).

ii.
```python
    ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
    sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
```

iii. The AI enumerated the behavior fields early (step 19) and chose the time-stamped variables (`SoundTime`, `Trial_start_time`) over the frame-number variables, noting in CONVERSION_NOTES.md that "Continuous timing variables are computed from the original frame timestamps (`ft`) and trial times".

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each decoder bin, the per-frame differences `SoundTime[trial] − ft[frame]` are converted from days to seconds and averaged over the frames in the bin. Sign convention is time *to* the cue: positive before the cue, negative after — the same convention as the reference. Stored as float32.

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

iii. CONVERSION_NOTES.md: "`time_to_sound_cue_s`: mean seconds to cue within each decoder bin" — the bin mean is the natural summary given the 3-frame averaging of the neural data.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from `ft[chunk]`, exactly the frame indices whose spike columns were averaged into the same neural bin, so the two are aligned bin for bin by construction.

ii.
```python
        for chunk in chunks:
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            ...
            trial_input.append(np.array([np.mean((sound_time[tr] - chunk_ft) * SECONDS_PER_DAY), ...]))
```

iii. Every stream in this dataset is indexed by imaging frame, so using the same frame indices for all streams is the alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `datexp` from the session index entry, parsed as a calendar date, plus the mouse identity (`mname`) used to find that mouse's earliest imaging date.

ii.
```python
def parse_date(date_str):
    return dt.datetime.strptime(date_str, "%Y_%m_%d").date()

def compute_training_days(sessions):
    by_subject = defaultdict(list)
    for idx, session in enumerate(sessions):
        by_subject[session["mname"]].append((idx, parse_date(session["datexp"])))
```

iii. The date string is the only field that orders a mouse's sessions; the AI used it directly rather than inferring a training stage from the experiment-group names (which it had already decided were unreliable, step 28).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Elapsed **calendar days** since the mouse's first imaging session: `(date − first_date).days`, constant within a session and broadcast to every bin of every trial. The range across the dataset is 0–92. The human reference instead counts *recorded sessions* (0, 1, 2, … per mouse), range 0–7; so the two encode the same ordering but on very different scales, and the AI's version has large gaps where a mouse was re-imaged much later.

ii.
```python
    offsets = {}
    for subject, entries in by_subject.items():
        first_date = min(date for _, date in entries)
        for idx, date in entries:
            offsets[idx] = float((date - first_date).days)
    return offsets
```
```python
    day_value = np.float32(training_days[session_idx])
    ...
    trial_input.append(np.array([..., day_value, ...], dtype=np.float32))
```
```python
            "training_day_rule": "Calendar days since the subject's first imaging session.",
```

iii. The rule is stated in metadata and CONVERSION_NOTES.md ("`training_day`: calendar days since the subject's first imaging session"). The trajectory contains no further discussion; the AI treated elapsed days as the literal reading of "day of training".

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `beh['Trial_start_time']` (datenum timestamp of corridor entry for each trial) and `beh['ft']`. I verified this is numerically identical to the reference's `np.interp(StartFr, arange(nfr), frame_time)` (median difference 0.0 s).

ii.
```python
    trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
```

iii. Same reasoning as 3-a: the timestamped trial-boundary fields are used directly instead of converting fractional frame numbers to times.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Per bin, `mean((ft[chunk] − Trial_start_time[trial]) × 86400)`, in seconds, positive after entry — the same sign convention as the reference. The minimum value in the converted dataset is 0.2 s (first running frame after entry). Because non-running frames are dropped but not "compressed out" of the clock, values can reach 1764.8 s on trials where the mouse parked mid-corridor.

ii.
```python
                        np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
```

iii. CONVERSION_NOTES.md: "`time_since_trial_start_s`: mean elapsed seconds since corridor entry within each decoder bin."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from `ft[chunk]`, the same frames averaged into the corresponding neural bin — identical alignment to 3-c.

ii.
```python
        for chunk in chunks:
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            ...
            np.mean((chunk_ft - trial_start[tr]) * SECONDS_PER_DAY),
```

iii. All streams are frame-indexed, so shared frame indices are the alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial flag marking trials run in the rewarded corridor. Same source as the reference.

ii.
```python
    is_rew = np.asarray(beh["isRew"], dtype=bool)
    ...
                        float(is_rew[tr]),
```

iii. The field is a direct per-trial boolean; no derivation is needed.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float (0.0/1.0) and broadcast to every bin of the trial as input dimension 3. The converted dataset's range is [0, 1] overall and 0 throughout the unsupervised/naive sessions, as expected.

ii.
```python
            trial_input.append(
                np.array([..., float(is_rew[tr])], dtype=np.float32)
            )
```

iii. CONVERSION_NOTES.md: "`reward_available`: constant within trial, taken from `isRew`."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']`, the per-trial name of the corridor wall texture. The vocabulary is built globally by scanning every behavior entry, giving 15 sorted names, and each trial's name is mapped to its index. (`TrialStim` was inspected and rejected: it has only 8 values including the placeholder `'stimulus_of_trial'`, i.e. it is masked in the swap sessions.)

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
```

iii. Trajectory step 58: "Some mice used rock/brick variants in addition to circle/leaf, and the converted output labels need to preserve whatever the actual per-trial `WallName` values are without accidentally collapsing categories." Steps 62–63 show the AI comparing the 15 `WallName` values against the 8 `TrialStim` values before choosing `WallName`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. None beyond the global name→index mapping: all **15** raw wall names are kept as distinct categories (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`), stored per bin as a constant across the trial. The human reference collapses these to the four base textures (circle / leaf / rock / wood). Consequences: chance level is 1/15 = 0.067 rather than 0.25, and several classes are rare (wood1_swap1 = 0.8% of bins, circle3 = 1.0%). Class frequencies in the AI's data are heavily uneven (0.8%–26%).

ii.
```python
        "output_values": [
            stim_names,
            ["no_lick", "lick"],
            ...
        ],
```
```python
            trial_output.append(np.array([stim_idx, ...], dtype=np.int64))
```
```python
            "stimulus_rule": "Per-trial visual category taken directly from beh['WallName'] for each trial.",
```

iii. The explicit rationale (step 58) is to avoid "accidentally collapsing categories", i.e. to preserve the labels exactly as the raw data records them and let the decoder deal with the finer partition.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']`, the (fractional) imaging-frame number of every lick in the session, converted to a boolean per-frame mask. Same source as the reference.

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

iii. CONVERSION_NOTES.md: "`licking`: binary per decoder bin, using `LickFr.astype(int)` mapped to frame bins." The frame number is truncated to the frame the lick falls in, exactly as the reference does.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A bin is 1 if any of its (up to 3) frames carries at least one lick, else 0. Non-finite, negative, and out-of-range lick frames are discarded. Because only running frames enter the bins, licks emitted while the mouse was stationary inside the corridor are lost; the resulting lick rate is 6.6% of bins (reference: 4.1% of frames).

ii.
```python
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

iii. `any()` rather than a mean/count is the natural way to keep the variable binary after 3-frame binning, consistent with the stated format requirement that licking be a binary time series.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the mask is already on the neural grid; the bin value is read with the same `chunk` frame indices used for the neural columns, giving one value per neural bin.

ii.
```python
        for chunk in chunks:
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            ...
            int(lick_mask[chunk].any()),
```

iii. Shared frame indexing, as for every other stream.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the corridor position at each imaging frame in decimeters (0–40 across the 4 m texture), truncated to `nfr`. Same source as the reference.

ii.
```python
    ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. It is the per-frame position variable documented in the authors' notebook, already on the neural frame grid.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The positions of the frames in a bin are averaged, and the bin mean is then discretized. No other processing.

ii.
```python
            mean_pos = float(ft_pos[chunk].mean())
            ...
                        position_to_bin(mean_pos),
```

iii. Averaging within the bin is consistent with how the neural data and the continuous inputs are binned.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `floor(mean_pos / 10)` clipped to [0, 3], i.e. four 1-m bins (`0-1m`, `1-2m`, `2-3m`, `3-4m`) — identical to the reference's `np.clip(ft_Pos // 10, 0, 3)`. Realized distribution: 24.6 / 23.9 / 22.7 / 28.9 % (reference: ~25% each).

ii.
```python
def position_to_bin(value):
    return int(np.clip(math.floor(value / 10.0), 0, 3))
```
```python
        "output_values": [..., ["0-1m", "1-2m", "2-3m", "3-4m"], ...],
```

iii. CONVERSION_NOTES.md: "`position_bin`: 4 bins covering the 4 m textured corridor" — the decimeter units of `ft_Pos` make the 1 m bins the task asks for a direct division by 10.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame, and the bin value uses the same `chunk` frame indices as the neural bin, so alignment is exact.

ii.
```python
        for chunk in chunks:
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            mean_pos = float(ft_pos[chunk].mean())
```

iii. Shared frame indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the running speed at each imaging frame. Same source as the reference.

ii.
```python
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
```

iii. The authors' field documentation lists `ft_RunSpeed` as "running speed for each neural frame", already frame-aligned.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is averaged over the frames of a bin, then discretized against **global** quartile thresholds. The thresholds are computed in a separate pre-pass over *every* session: for each session, the same corridor+running frame selection and 3-frame chunking is reproduced, the bin-mean speed is collected, and `np.quantile(..., [0.25, 0.5, 0.75])` is taken over the ~286k bins of the whole dataset, giving [13.63, 25.20, 39.69]. The reference instead ranks speeds *within each session* and splits the ranks into four equal groups.

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

iii. CONVERSION_NOTES.md: "`running_speed_bin`: quartile bin of mean `ft_RunSpeed`, with thresholds computed globally over all retained decoder bins", and the sanity-check line "Running-speed quartiles are computed globally from the retained bins, not per session". The global quartiles are also printed in the conversion log and stored in metadata.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.searchsorted(thresholds, value, side='right')` maps a bin's mean speed to 0–3 (`speed_q1`…`speed_q4`). Because the thresholds are dataset-wide quantiles of exactly the same quantity, the realized marginal distribution is 25.0 / 25.0 / 25.0 / 25.0 %, matching the instruction that each bin hold 25% of the data (the reference achieves the same marginal via per-session ranks). Note that the upstream `ft_move > 0` filter removes almost all exactly-zero-speed samples, so the lowest bin means "slow running" rather than "stationary"; this also removes the tie-at-zero problem that motivated the reference's rank-based split.

ii.
```python
def speed_to_bin(value, thresholds):
    return int(np.searchsorted(thresholds, value, side="right"))
```
```python
            "speed_thresholds": [float(x) for x in speed_thresholds.tolist()],
            "speed_thresholds_definition": "Quartiles of mean ft_RunSpeed over all retained decoder bins in the full dataset.",
```

iii. The AI wanted the quartile definition to be a property of the dataset as a whole rather than of each session, so that the label means the same thing in every session, and computed it from the same binned quantity that is later labelled.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame; the bin value is the mean over the same `chunk` frames used for the neural bin.

ii.
```python
        for chunk in chunks:
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            mean_speed = float(ft_speed[chunk].mean())
```

iii. Shared frame indexing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several small defensive measures: (a) the behavior can run longer than the imaging, so every behavior stream is truncated to `nfr = min(n_spike_frames, len(ft))`; (b) `iarea` is truncated to the number of concatenated spike rows; (c) lick frames that are non-finite, negative, or ≥ `nfr` are discarded; (d) trials left with no retained frames are skipped; (e) sessions missing a behavior entry raise immediately (`Missing behavior entries for sessions: ...`), and sessions with no corridor frames or no assigned visual-cortex neuron raise a `RuntimeError`; (f) behavior files that fail to load are skipped in `behavior_files`. NaNs are not otherwise checked, but I confirmed `SoundTime` has no NaNs anywhere in the dataset, and the converted inputs contain no NaNs. Unlike the reference there is no per-session `try/except` that lets the run continue past a bad session, and no explicit "≥ 2 trials per session" guard (not needed here: the minimum is 84 trials).

ii.
```python
    nfr = min(spk.shape[1], len(beh["ft"]))
    region_idx_full = coarse_region_indices(iarea[: spk.shape[0]])
    frames_by_trial = trial_frame_indices(beh, nfr)
    selected_frames = np.concatenate(frames_by_trial)
    if selected_frames.size == 0:
        raise RuntimeError(f"No running corridor frames found for session {session['base']}")
```
```python
    lick_idx = lick_fr[np.isfinite(lick_fr)].astype(np.int64)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
```

iii. The AI inspected the field shapes and NaN fractions early (step 42: `ft_move` / `ft_RunSpeed` "nanfrac 0.0") and concluded the dataset is clean apart from the behavior/imaging length mismatch, which the authors' own code also handles with `[:nfr]`.

## 12-a. What are the most time-consuming steps of the code?

i. (1) Reading the per-session spike files — they are multi-GB each (e.g. 9.3 GB for `TX109_2023_03_16_1`) and are fully materialized by `np.concatenate(spk_obj['spks'])`; this dominates the ~25 min full run. (2) The per-neuron variance pass used for neuron selection, which touches every neuron of the session; the AI had to cap it at 2048 frames after measuring that it was the bottleneck. (3) The startup pre-passes: loading *all* `Beh_*.npy` files, cross-comparing duplicate entries array by array, and the pure-Python speed-threshold pass that re-derives every trial's frames and takes ~286k per-chunk `mean()` calls. (4) Inside the conversion, the per-bin Python loop that calls `.mean()` separately for the neural, speed and position values of each bin.

ii.
```python
    spk_obj = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_obj["spks"]], axis=0)
```
```python
def variance_over_columns(matrix, cols, block_cols=1024, max_var_frames=2048):
    if cols.size > max_var_frames:
        keep = np.linspace(0, cols.size - 1, max_var_frames, dtype=np.int64)
        cols = cols[keep]
```

iii. Trajectory steps 120–129: the AI checked that the process was CPU-bound rather than stalled, identified neuron ranking as the hotspot ("neuron ranking was using every retained frame in a session and that's unnecessary for stable variance estimates"), capped it, re-timed a 2-session run (~14 s) and only then launched the full pass.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Four places. (1) `trial_frame_indices` scans the whole frame index once per trial (`ft_trind == tr`) instead of grouping all frames by trial in a single pass (e.g. `np.argsort`/`np.split`), and it is executed twice per session (threshold pass + conversion). (2) The per-chunk loop in `compute_speed_thresholds` calls `.mean()` on 1–3 element slices ~286k times in Python; `np.add.reduceat` over chunk boundaries would do it in one call. (3) The per-bin loop in `convert_session` does the same for the neural matrix, speed, position, and the timing inputs — all four could be computed for the whole trial at once with `np.add.reduceat` and vectorized arithmetic. (4) `region_counts = [(region_idx_full == ridx).sum() for ridx in range(4)]` could be a single `np.bincount`.

ii.
```python
    for tr in range(ntrials):
        mask = (ft_trind == tr) & is_corr & is_move
        out.append(np.flatnonzero(mask).astype(np.int64))
```
```python
        for chunk in chunks:
            chunk_ft = ft[chunk]
            neural_bin = spk_sel[:, chunk].mean(axis=1)
            mean_speed = float(ft_speed[chunk].mean())
            mean_pos = float(ft_pos[chunk].mean())
```

iii. Not discussed in the trajectory; the AI's only performance work was capping the variance computation, after which it judged the run "tractable" (step 129) and left the remaining loops alone.

## 12-c. What processing does the code repeat multiple times?

i. (1) `trial_frame_indices` and `chunk_indices` are computed for every session twice — once in `compute_speed_thresholds` and again in `convert_session`. (2) The chunk-mean speed of every bin is computed twice for the same reason. (3) All behavior files are loaded in full and held in memory for the whole run (`canonical_lookup` keeps a reference to every canonical entry, and `provenance` to every duplicate), and duplicate entries are compared array-by-array across 11 fields even though the result is only printed. (4) `compute_frame_dt_ms` makes yet another pass over every session's `ft`.

ii.
```python
    training_days = compute_training_days(sessions)
    frame_dt_ms = compute_frame_dt_ms(canonical_lookup, sessions)
    speed_values, speed_thresholds = compute_speed_thresholds(canonical_lookup, sessions, args.frames_per_bin)
```
```python
        for indices in trial_frame_indices(beh, nfr):
            for chunk in chunk_indices(indices, frames_per_bin):
                speed_values.append(float(ft_speed[chunk].mean()))
```

iii. The repeated pass is a consequence of the AI's choice to define the speed quartiles globally: the thresholds must be known before any session is written out. The AI notes only that these passes use "the lighter metadata... side" (step 54), i.e. behavior rather than spikes, so the duplication does not cost spike I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The duplicate-entry verification (`arrays_equal` over 11 fields for every duplicated behavior entry) is a diagnostic only: its result is printed and otherwise unused. (2) `provenance` — the full list of (file, key) pairs for every session — is computed and then embedded in the output pickle as `metadata['behavior_entry_provenance']`, which no downstream step reads. (3) `compute_speed_thresholds` returns the full 286k-element `speed_values` array, which is never used after the quantiles are taken. (4) `np.random.seed(args.seed)` is set but no randomness is used. (5) A second output file, `sample_data.pkl`, plus the whole `choose_sample_session_indices` / `build_sample_dataset` machinery, is built on every run even though only `converted_data.pkl` is required. (6) The per-neuron variance over all 20k–90k neurons of a session is computed only to discard 99.7% of them — work that exists solely because of the neuron-subsetting decision. (7) `summarize_dataset` recomputes summary statistics that `train_decoder.py --verify-only` also prints.

ii.
```python
            "behavior_entry_provenance": {base: list(entries) for base, entries in provenance.items()},
```
```python
    sample_data = build_sample_dataset(
        full_data=data,
        nsessions=args.sample_sessions,
        max_trials_per_session=args.sample_trials_per_session,
    )
```

iii. Most of this is deliberate self-checking: the AI wanted to prove that deduplicating behavior entries was safe (steps 28–39) and wanted a small representative subset it could verify and train on quickly before committing to the full run (steps 87–89, 163–169: "the first six sessions are all non-rewarded, so the sample subset has no licking examples. I'm fixing the sample subset selection to make it representative"). None of it is described as needed by the final dataset.
