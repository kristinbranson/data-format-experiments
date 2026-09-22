# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively selects every `*_behavior+ophys.nwb` file one directory below `/app/data/sub-*`, sorts the paths, and opens each file directly with `h5py`. Each file becomes one output session.

ii. `files = sorted(glob.glob(os.path.join(ROOT, 'sub-*', '*_behavior+ophys.nwb')))` and `with h5py.File(path, 'r') as f:`

iii. The trajectory says this avoids loading raw movies, includes all 152 deposited behavior+ophys sessions, and uses direct NWB/HDF5 inspection for efficiency.

## 1-b. How are the data split into subjects?

i. Subject IDs are taken from the parent directory name, deduplicated, naturally sorted, and mapped to an integer `subject_idx` for every session.

ii. `subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-', '') for p in files}, key=lambda x: int(x[1:]) if x.startswith('m') and x[1:].isdigit() else x)` and `data['subject_idx'].append(submap[subj])`

iii. The trajectory identified the per-subject directory layout and verified the full set of files before conversion.

## 1-c. How are the data split into sessions?

i. Each sorted NWB file is treated as one session and appended once to the session-level lists.

ii. `for k, path in enumerate(files): neu, inp, out, nc, nt = convert_session(path)` followed by `data['neural'].append(neu)`.

iii. The agent interpreted each `sub-*_ses-*_behavior+ophys.nwb` as a deposited session and reported successful conversion of 152 sessions.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples in `trial_start`; candidate ends are every positive sample in `teleport`. `trial_pairs` pairs a start with the first subsequent end only when it precedes the next start, using the start inclusively and teleport exclusively.

ii. `starts = np.flatnonzero(beh('trial_start') > 0)`; `ends = np.flatnonzero(beh('teleport') > 0)`; `pairs = trial_pairs(starts, ends)`; trial slices use `a:b`.

iii. The agent found the stored trial-number stream changes during some ITIs and therefore considered explicit trial-start/teleport streams cleaner and consistent with the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. There is no per-trial duration or quality filter. Malformed/unpaired starts and ends are omitted by `trial_pairs`; a session raises if fewer than two complete pairs remain.

ii. `if b < next_a and b > a: pairs.append((int(a), b))` and `if len(pairs) < 2: raise ValueError(...)`.

iii. The trajectory treats all paired trials, including sessions with legitimate variable trial counts, as usable. It gives no paper-based justification for omitting the reference's short-trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from `processing/ophys/Deconvolved/plane0/data`, with ROI indices mapped through its `rois` table (or the fluorescence ROI table as fallback) and the segmentation `iscell` column.

ii. `dset = f['processing/ophys/Deconvolved/plane0/data']` and `iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] == 1`.

iii. The agent explicitly reasoned that this was native, already frame-aligned Suite2p deconvolved activity and inspected ROI mappings to handle subsetted columns.

## 2-b. How is the `neural` data processed?

i. Apart from ROI selection, float32 conversion, trial slicing, and transpose to neurons-by-time, no signal processing is performed. In particular, dF/F, neuropil correction, smoothing, and deconvolution are not recomputed.

ii. `deconv = dset[:, cell_idx]` and `neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())`.

iii. The module docstring says the deposited Suite2p deconvolved stream is used at its native imaging rate; the trajectory regarded this as the deposited processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs whose mapped PlaneSegmentation row has `iscell == 1` are retained. There is no putative-interneuron/speed-correlation filter.

ii. `cell_idx = np.flatnonzero(iscell[roi_rows])` and `deconv = dset[:, cell_idx]`.

iii. The trajectory emphasizes correct DynamicTableRegion mapping and Suite2p `iscell` filtering, but does not justify omitting the paper's interneuron exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural frames are sliced beginning at the `trial_start` frame, so time zero is trial start; slicing ends before the teleport frame.

ii. `for i, (a, b) in enumerate(pairs): neural_trials.append(... deconv[a:b, :] ... )`.

iii. The agent concluded that behavioral and neural streams in the NWB were already frame-aligned, so shared indices required no interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The declared resolution is `1000 / 15.5078125 = 64.4839 ms`; no rebinning or resampling is applied.

ii. `RATE = 15.5078125`; `DT_MS = 1000.0 / RATE`; metadata stores `'time_bin_size': DT_MS`.

iii. The agent described this as the native imaging rate and verified that behavior was already resampled to imaging frames.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the integer frame offset within each paired trial and the hard-coded imaging rate, rather than the loaded timestamp values (which are used for reward-event membership).

ii. `inp[0] = np.arange(n, dtype=np.float32) / RATE`.

iii. The trajectory found a constant native frame rate and frame alignment, motivating a frame-count clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based sequence of trial frames is divided by 15.5078125 Hz.

ii. `n = b - a` and `inp[0] = np.arange(n, dtype=np.float32) / RATE`.

iii. This makes the first trial frame exactly zero and advances by one native time bin.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is created with exactly `b-a` values, matching the same `a:b` neural slice.

ii. `n = b - a`; neural uses `deconv[a:b, :]`; input uses `np.arange(n)`.

iii. The agent relied on the NWB behavior streams already being sampled on neural frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavioral `environment` data over the current trial.

ii. `environment = beh('environment')` and `ev = environment[a:b]`.

iii. The agent identified this stream as the explicit ENV1/ENV2 indicator.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded, the remaining trial values are reduced to their median and rounded to an integer, or defaulted to 0 if none remain; the result is repeated across the trial.

ii. `ev = ev[ev >= 0]`; `env = int(np.rint(np.median(ev))) if len(ev) else 0`; `inp[1] = env`.

iii. The comment says this avoids ITI `-1` values and supports both constant sessions and ENV1-to-ENV2 switch sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index over paired trials, not the raw `trial number` stream.

ii. `for i, (a, b) in enumerate(pairs): ... inp[2] = i`.

iii. The agent observed that the raw trial-number stream changes during ITIs and preferred the sequential index defined by explicit trial boundaries.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is performed beyond broadcasting the loop index over every frame in the trial.

ii. `inp[2] = i`.

iii. The choice provides a continuous, within-session, zero-based trial number.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, the position timestamps used as the common frame clock, and the previous paired trial's time interval.

ii. `reward_times = f[B + 'Reward/timestamps'][:]`; `ts = f[B + 'position/timestamps'][:]`.

iii. The agent treated occurrence of any reward timestamp inside a trial as that trial's binary outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current-trial reward flags are precomputed by interval membership. A running `prev`, initialized to 0, is broadcast through each trial and then updated to the current outcome.

ii. `rewarded = np.array([np.any((reward_times >= ts[a]) & (reward_times < ts[b])) for a, b in pairs], dtype=np.int8)`; `inp[3] = prev`; `prev = int(rewarded[i])`.

iii. This directly implements omitted=0/rewarded=1 and defines the first trial's previous outcome as 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position`, `reward_zone`, and fixed zone starts A/B/C at 80/200/320 cm with width 50 cm. Zone identity is inferred per trial from positions at positive reward-zone samples.

ii. `p = position[a:b][zone_event[a:b] > 0]`; `zones[i] = int(np.argmin(np.abs(ZONE_STARTS - np.median(p))))`.

iii. The agent reasoned that the event can occur just after entry, so median event position is assigned to the nearest nominal zone start; missing omission/aborted trials are filled from a nearby observed trial.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For the inferred zone, distance is position minus the start before the zone, zero within its inclusive 50-cm span, and position minus the end after it.

ii. `dist = np.where(pos < z0, pos - z0, np.where(pos > z1, pos - z1, 0.0))`.

iii. This implements signed distance to the closest zone edge, with all positions inside the zone represented by zero.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. `np.digitize` uses internal edges -50, -10, 0, the next floating-point number above 0, 10, and 50, yielding categories 0 through 6.

ii. `out[0] = np.digitize(dist, [-50, -10, 0, np.nextafter(0.0, 1.0), 10, 50])`.

iii. The code comment states that default `right=False` and the tiny positive boundary implement the requested exact-zero class and edge behavior.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the exact same `a:b` indices as neural data and produces one distance category per neural frame.

ii. Neural uses `deconv[a:b, :]`, while `pos = position[a:b]` supplies the distance calculation.

iii. The agent relied on deposited frame alignment rather than performing additional temporal matching.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` data.

ii. `position = beh('position')` and `pos = position[a:b]`.

iii. The stream is the animal's location along the 450-cm virtual corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial slice is not transformed before categorical binning.

ii. `pos = position[a:b]`; `out[1] = np.digitize(pos, [90, 180, 270, 360])`.

iii. The requested output is absolute corridor position, so the raw aligned positions are used.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Internal boundaries 90, 180, 270, and 360 cm create five categories.

ii. `out[1] = np.digitize(pos, [90, 180, 270, 360])`.

iii. These are five equal 90-cm divisions of the 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial frame interval `a:b` is used for position and neural data.

ii. `neural_trials.append(... deconv[a:b, :] ...)` and `pos = position[a:b]`.

iii. The agent found behavior already resampled to imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` stream.

ii. `lick = beh('lick')` and `lick[a:b]`.

iii. This is the deposited framewise lick measurement.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Every positive value is mapped to 1 and all other values to 0.

ii. `out[3] = (lick[a:b] > 0).astype(np.int8)`.

iii. The task requires a binary no/yes output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks and neural activity use the same `a:b` trial slice.

ii. `lick[a:b]` corresponds frame-for-frame to `deconv[a:b, :]`.

iii. The agent judged the behavioral streams to be pre-aligned to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from positive samples of the behavioral `reward_zone` stream and the corresponding `position`, interpreted against fixed A/B/C starts.

ii. `p = position[a:b][zone_event[a:b] > 0]` and `np.argmin(np.abs(ZONE_STARTS - np.median(p)))`.

iii. The agent used event-position evidence because the event itself does not directly encode the A/B/C label.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Finite in-track event positions are median-reduced and classified by nearest zone start. Trials lacking evidence receive the label of the nearest trial with a known label. The resulting 0/1/2 label is repeated across the trial.

ii. `p = p[np.isfinite(p) & (p >= 0) & (p <= 450)]`; `zones = nearest_fill(zones)`; `out[4] = zones[i]`.

iii. The comment says nearest filling preserves abrupt experimental switches while handling omission or aborted traversals.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, the position timestamps, and each trial's start/end frame indices.

ii. `reward_times = f[B + 'Reward/timestamps'][:]` and `(reward_times >= ts[a]) & (reward_times < ts[b])`.

iii. A delivery timestamp inside a trial is direct evidence that the trial was rewarded.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, any reward timestamp in the half-open timestamp interval produces 1; otherwise it produces 0. The scalar is broadcast across all trial frames.

ii. `rewarded = np.array([np.any((reward_times >= ts[a]) & (reward_times < ts[b])) ...], dtype=np.int8)` and `out[5] = rewarded[i]`.

iii. This implements the requested per-trial binary label while excluding an event at the teleport boundary.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid trial pairings are skipped; sessions with fewer than two complete trials fail. Missing reward-zone labels are nearest-filled and a session with no known label fails. Nonfinite/out-of-track zone-event positions are ignored. ROI table mismatch fails explicitly, and a fallback ROI mapping is used when the Deconvolved `rois` dataset is absent. Missing environment values default to 0. There is no neural/behavior length reconciliation or short-trial filter.

ii. Examples include `if b < next_a and b > a`, `zones = nearest_fill(zones)`, `if not len(known): raise ValueError(...)`, and `if len(roi_rows) != dset.shape[1]: raise ValueError(...)`.

iii. The trajectory shows the agent discovered and fixed ROI subset mapping, while code comments explain nearest filling for omission/aborted trials. Other defaults are defensive but not tied to the reference pipeline.

## 13-a. What are the most time-consuming steps of the code?

i. Reading each large deconvolved dataset, materializing/copying every trial matrix, holding the full converted data in memory, and serializing the large pickle dominate. Reward checks and trial construction add smaller repeated costs.

ii. `deconv = dset[:, cell_idx]`, `np.asarray(deconv[a:b, :], dtype=np.float32).T.copy()`, and `pickle.dump(data, fh, protocol=4)`.

iii. The trajectory reports conversion progress over 152 large NWBs and then a noticeable final serialization period; it does not explicitly profile timings.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. `nearest_fill` loops over missing labels; `trial_pairs` loops over starts; zone inference and reward detection loop over trials; final trial construction loops again. Reward membership and much of zone classification could be vectorized, although variable trial slices naturally remain convenient as loops.

ii. Representative snippets are `for i in missing:`, `for k, a in enumerate(start):`, and `rewarded = np.array([np.any(...) for a, b in pairs])`.

iii. The agent did not discuss vectorization explicitly; its trajectory prioritized correctness and direct framewise construction.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trial pairs once for zone inference, once for reward outcomes, and once for constructing outputs; it also repeatedly slices the same behavior ranges and repeatedly compares all reward timestamps for every trial.

ii. `for i, (a, b) in enumerate(pairs):` appears in zone inference and output construction, while the reward list comprehension iterates over `pairs` separately.

iii. No explicit justification is recorded; the separation keeps inference, outcome calculation, and output assembly simple.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `speed` only to bin it as a requested output, so that work is not discarded. The main avoidable intermediates are returned `len(pairs)`, per-session `session_info`, repeated copied slices, and loaded timestamps after reward flags are formed; these are either metadata or temporary rather than analysis inputs. Unlike the reference, it performs no preliminary survey or recomputed dF/F that is later discarded.

ii. `return neural_trials, input_trials, output_trials, len(cell_idx), len(pairs)` and `session_info.append({...})`.

iii. The trajectory provides no explicit discussion of discarded work; it deliberately chose a single conversion pass and direct HDF5 access.
