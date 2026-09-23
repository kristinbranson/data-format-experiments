# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates every subject directory and every `.nwb` file, sorts the 152 session paths by numeric mouse/session, and loads each session directly with `h5py`. Sessions are converted in parallel and retained in sorted order.

ii. `for sub in sorted(os.listdir(DATA_ROOT)):` / `if fn.endswith(".nwb"): paths.append(...)`; `with h5py.File(path, "r") as f:`; `ProcessPoolExecutor(max_workers=args.workers)`.

iii. It states that `/app/data` contains 11 subject directories and 152 NWB files, and chose direct HDF5 access for speed while reading the same NWB datasets.

## 1-b. How are the data split into subjects?

i. Subject identity comes from `general/subject/subject_id`; the final unique mouse list is numerically sorted and each session gets an index into it.

ii. `subject = f["general/subject/subject_id"][()].decode()`; `subjects = sorted({r["info"]["subject"] for r in results}, key=lambda s: int(s.replace("m", "")))`.

iii. The notes report 11 mice and validate the subject/session totals against the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Session paths are ordered by subject and session number; each conversion result becomes one entry in the session-level lists.

ii. `ses = int(base.split("_")[1].replace("ses-", ""))`; `"neural": [r["neural"] for r in results]`.

iii. The file naming and NWB session metadata identify sessions, and parallel mapping preserves input order.

## 1-d. How are the data split into trials?

i. Trials are laps indexed from a positive `trial_start` sample through, but excluding, the positive `teleport` sample: `[trial_start, teleport)`.

ii. `trial_start = np.where(b["trial_start"]["data"][:] > 0)[0]`; `teleport = np.where(b["teleport"]["data"][:] > 0)[0]`; `for k, (lo, hi) in enumerate(zip(si, ti)):`.

iii. The notes call this the behaviorally exact lap window and intentionally reject the paper helper's one-frame shift so neural and behavior samples remain aligned.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed when the lick sensor appears stuck: more than 30% of lap frames have raw cumulative lick count above 2. No minimum-length filter is applied.

ii. `lick_error = np.array([(lick[lo:hi] > 2).sum() / (hi - lo) > LICK_ERROR_FRAC ...])`; `if lick_error[k]: continue`.

iii. This reproduces the paper's 81 bad-lick trials. Because lick is a required categorical output and cannot be NaN, the agent drops affected trials rather than only blanking lick values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from suite2p `Fluorescence` (`F`) and `Neuropil` (`Fneu`) arrays, restricted to `iscell` ROIs and pooled across planes; it does not use the stored NWB `Deconvolved` signal.

ii. `dset = f["processing/ophys/Fluorescence"][plane]["data"]`; `f["processing/ophys/Neuropil"][plane]["data"]`; `keep = np.where(iscell[...])[0]`.

iii. The paper computes its analyzed signal from F/Fneu; the stored deconvolution is a different suite2p product.

## 2-b. How is the `neural` data processed?

i. The default full conversion exports smoothed ΔF/F: subtract `0.7*Fneu`, add segment mean neuropil back, estimate a maximin baseline (Gaussian sigma 15, 300-frame minimum then maximum), normalize by absolute baseline, and Gaussian-smooth at sigma 2. OASIS events are optional via `--signal events`, but were not used in `converted_data.pkl`.

ii. `f_ -= NEU_COEF * fneu_`; `flow = gaussian_filter1d(x, BASELINE_SMOOTH_SIGMA, axis=1)`; `flow = minimum_filter1d(flow, BASELINE_FILTER_WIN, axis=-1)`; `d = (x - flow) / np.abs(flow)`; `activity = spks if signal == "events" else dff`.

iii. The agent ports the paper's dF/F pipeline and chose ΔF/F after comparing decoder results, despite noting that paper decoding used deconvolved events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only manually curated `iscell > 0` ROIs are loaded. Cells with Pearson correlation between in-trial ΔF/F and speed greater than 0.5 are then removed as putative interneurons.

ii. `iscell = seg["iscell"][:, 0] > 0`; `r_speed = (dc.astype(np.float64) @ vc) / denom`; `keep_cells = r_speed <= INTERNEURON_SPEED_R`.

iii. Both filters are explicitly attributed to the Methods; observed removal fractions were checked against the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are sliced at the same frame indices as behavior, with time zero at `trial_start`; no interpolation or shift is applied.

ii. `act = np.ascontiguousarray(activity[:, lo:hi], dtype=np.float32)`.

iii. The agent says the exact `[trial_start, teleport)` window preserves samplewise neural/behavior alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native frames are retained with no rebinning: 15.5078125 Hz, or 64.4836 ms per bin, across sessions.

ii. `FRAME_PERIOD = 1.0 / FRAME_RATE`; `"time_bin_size": 1000.0 / FRAME_RATE`.

iii. This is the per-plane rate used by the paper; retaining it avoids information loss.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is generated from the native imaging-frame index and the fixed frame period, with the raw behavior timestamps used to establish/common-crop stream length but not directly subtracted per trial.

ii. `inp[0] = np.arange(T, dtype=np.float32) * FRAME_PERIOD`.

iii. The fixed 15.5078125 Hz rate was validated as common to all recordings.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based frame counter is multiplied by seconds per frame.

ii. `np.arange(T, dtype=np.float32) * FRAME_PERIOD`.

iii. This makes time reset to zero and advance by one native frame in every trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It has exactly `hi-lo` values and is created alongside the same neural slice, so column j denotes neural frame j after trial start.

ii. `T = hi - lo`; `act = activity[:, lo:hi]`; `inp = np.empty((4, T), ...)`.

iii. Shared indices and common cropping are used to guarantee equal lengths.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the NWB behavior `environment` series.

ii. `beh = {k: b[k]["data"][:] for k in [..., "environment", ...]}`.

iii. The series encodes ENV1/ENV2 and was checked to be constant within each lap.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median within a trial is rounded and cast to integer, then broadcast over all trial frames.

ii. `environment = np.array([int(np.round(np.median(beh["environment"][lo:hi]))) ...])`; `inp[1] = environment[k]`.

iii. Median/rounding robustly obtains the constant per-trial binary context.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based lap index implied by the paired trial-start/teleport boundaries, not the stored `trial number` values.

ii. `trial_number = np.arange(ntrials_raw, dtype=np.int64)`.

iii. It preserves original lap numbering even if a later quality filter drops a lap.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Sequential integers are generated before filtering and broadcast across each retained trial.

ii. `inp[2] = trial_number[k]`.

iii. This represents within-session trial number as required.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Current outcomes derive from Reward event timestamps plus whether `reward_zone` was entered; previous outcome is the preceding raw lap's current outcome.

ii. `is_reward = np.array([bool(np.any((S["reward_idx"] >= lo) & ... ) and np.any(rzone[lo:hi] > 0)) ...])`; `prev_outcome[1:] = is_reward[:-1]`.

iii. The conjunction ports `behavior.get_trial_types`; calculating before dropped trials preserves experienced history.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Outcomes are binary and shifted one lap. The first recorded trial is assigned 1, then values are broadcast per trial.

ii. `prev_outcome[0] = 1`; `prev_outcome[1:] = is_reward[:-1]`; `inp[3] = prev_outcome[k]`.

iii. The agent assumes preceding warm-up laps were usually rewarded, making 1 the expected first value; this affects one lap per session.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` and an active A/B/C zone inferred from the NWB `identifier` scene name, including a switch after lap 30.

ii. `scene = parse_scene(f["identifier"][()].decode())`; `zone_labels = reward_zone_labels(S["scene"], ntrials_raw)`; `p = pos[lo:hi]`.

iii. This ports the reference repository's scene grammar and reward-zone schedule; assignments were checked against empirical `reward_zone` flags (maximum reported error 8.5 cm).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is position minus the near edge before a zone, zero inside it, and position minus the far edge after it; the result is then categorized.

ii. `d[before] = pos[before] - zone_start`; `d[after] = pos[after] - zone_end`; `out[0] = bin_distance_to_reward(...)`.

iii. This implements distance to the nearest point of the active reward interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks create seven classes with boundaries -50, -10, 0, 10, and 50 cm, reserving class 3 for exact zero/inside-zone samples.

ii. `out[dist < -50] = 0`; `out[(dist >= -10) & (dist < 0)] = 2`; `out[dist == 0] = 3`; `out[dist > 50] = 6`.

iii. The masks implement the requested inequalities exactly and avoid `digitize` ambiguity at zero.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the same `[lo:hi)` indices; the resulting output has T columns.

ii. `act = activity[:, lo:hi]`; `p = pos[lo:hi]`; `out = np.empty((6, T), ...)`.

iii. No interpolation is necessary because streams are frame-aligned in the NWB after common cropping.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from behavior `position`.

ii. `pos = beh["position"]`; `p = pos[lo:hi]`.

iii. This is the mouse's centimeter position on the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position samples are directly discretized; `np.clip` absorbs values outside nominal track limits into end classes.

ii. `return np.clip(np.digitize(pos, [90.0, 180.0, 270.0, 360.0]), 0, 4)`.

iii. Five equal 90 cm bins implement the task specification.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholds are 90, 180, 270, and 360 cm, yielding integer classes 0-4.

ii. `out[1] = bin_position(p)`.

iii. These are five equal divisions of a 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the identical trial slice as neural data.

ii. `activity[:, lo:hi]` and `pos[lo:hi]`.

iii. Common frame indexing provides direct alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It derives from behavior `lick`, a per-frame cumulative lick count.

ii. `lick = beh["lick"]`.

iii. The Methods describe converting lick counts to a binary vector.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive count becomes 1 and zero becomes 0; corrupted trials are removed first.

ii. `out[3] = (lick[lo:hi] > 0).astype(np.int64)`.

iii. This matches the required no/yes output and the paper's binarization.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks and activity are selected with the same lap indices.

ii. `activity[:, lo:hi]`; `lick[lo:hi]`.

iii. Both streams share native imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB `identifier` scene string and raw lap index (for scheduled switches), not directly from per-frame `reward_zone` values.

ii. `zone_labels = reward_zone_labels(S["scene"], ntrials_raw)`.

iii. The scene naming encodes A/B/C and switch conditions used by the paper's behavior code.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene grammar assigns A/B/C, switch sessions change after trial 30, labels map to 0/1/2, and the class is broadcast across trial frames.

ii. `labels[:change_trial] = ...; labels[change_trial:] = ...`; `zone_idx = np.array([ZONE_TO_IDX[l] ...])`; `out[4] = zone_idx[k]`.

iii. It is a port of `behavior.get_reward_zones` and was empirically sanity-checked.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses the `Reward` event timestamps, trial boundaries, and the per-frame `reward_zone` flag.

ii. `reward_times = ...["Reward"]["timestamps"][:]`; `reward_idx = np.searchsorted(timestamps, reward_times)`; `np.any(rzone[lo:hi] > 0)`.

iii. The agent attributes the combined rule to `behavior.get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward times are mapped to frame indices. A trial is 1 only if a reward index falls in `[lo,hi)` and its reward-zone flag is positive somewhere; otherwise 0. It is broadcast over T.

ii. `bool(np.any((S["reward_idx"] >= lo) & (S["reward_idx"] < hi)) and np.any(rzone[lo:hi] > 0))`; `out[5] = is_reward[k]`.

iii. This is intended to exclude non-task reward artifacts and represent per-lap rewarded/omitted status.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior streams are cropped to their common minimum length; extensive assertions check trial consistency and finite activity; bad lick-sensor trials are removed. No general imputation is done.

ii. `T = min(F.shape[1], len(beh["position"]))`; `assert len(trial_start) == len(teleport)`; `if not np.all(np.isfinite(act)): raise RuntimeError(...)`.

iii. The notes identify occasional extra imaging frames and use conservative truncation; known lick corruption is handled by reproducing the paper's criterion.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large contiguous fluorescence arrays, per-segment maximin filtering, optional OASIS deconvolution, and serializing the 9.6 GB pickle dominate. Per-session timings explicitly separate load and dF/F work.

ii. `t_load = time.time()`; `t_dff = time.time()`; `pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)`.

iii. The notes report full conversion around 45 seconds with process-level session parallelism and explain that full HDF5 reads outperform fancy indexing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial mask construction, reward/outcome/environment/lick-error list comprehensions, trial assembly, reward-zone mapping, and baseline-segment loops could partly be vectorized. The agent instead vectorizes over neurons and parallelizes sessions because variable lap boundaries make trial loops natural.

ii. `for lo, hi in zip(si, ti): in_trial[lo:hi] = True`; `for k, (lo, hi) in enumerate(zip(si, ti)):`; `for lo, hi in segments:`.

iii. The notes characterize the approximately 80-iteration lap loop as acceptable/structurally necessary for per-lap baselines.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly loops over trial boundaries to form baseline segments/masks, calculate outcome/environment/lick QC, and assemble arrays; `compute_dff` traverses segments separately to populate data and then filter them. Optional plotting recomputes distances, bins, and empirical zone-entry positions.

ii. `for lo, hi in segments:` appears in both segment population and dF/F computation; several `[... for lo, hi in zip(si, ti)]` expressions repeat trial slicing.

iii. The agent prioritizes clarity and bounded memory; diagnostic repetitions occur only with `--show-processing`.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `load_session` reads `autoreward`, `scanning`, `trial number`, plane indices, and some metadata not used in decoder arrays. `reward_zone` is used only for reward classification/QC, and optional events/plots are discarded unless requested. Timing and traceability metadata are retained but not decoder features.

ii. `beh = {k: ... for k in [..., "trial number", "autoreward", "scanning"]}`; `plane_of_cell = ...`; `spks = ... if deconvolve else None`.

iii. Extra fields support validation, provenance, optional diagnostics, and alternative-signal generation rather than downstream model inputs.
