# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 25 fixed-delay and 19 randomized-delay sessions and their selected probes, then loads each `data_structure_<animal>_<date>.mat` with `matio.load_obj`; motion energy is loaded separately with `loadmat_var`. It does not discover files by globbing.

ii.
```python
FIXED_DELAY_SESSIONS = [('JEB6', '2021-04-18', [2]), ...]
RANDOM_DELAY_SESSIONS = [('JEB11', '2022-05-10', [1]), ...]
...
for anm, date, probes, folder, task in sessions:
    rez = process_session(anm, date, probes, folder, task)
```

iii. The trajectory says the lists were transcribed from the authors' `load*_ALMVideo.m` scripts so commented-out and unlisted recordings would not be included. It also explicitly retained randomized-delay sessions because the paper analyzed them with the same go-cue-aligned pipeline.

## 1-b. How are the data split into subjects?

i. The animal string in each hard-coded session tuple defines the subject. Subjects are accumulated in first-seen order, and each retained session receives the corresponding integer `subject_idx`.

ii.
```python
if anm not in subjects:
    subjects.append(anm)
subject_idx.append(subjects.index(anm))
```

iii. The trajectory treats the filename/session-list animal identifier as authoritative and reports 14 mice.

## 1-c. How are the data split into sessions?

i. Each `(animal, date, probes)` tuple and its matching MAT file is one session and becomes one element of `neural`, `input`, and `output`. Sessions with fewer than two retained trials or fewer than ten retained units are skipped.

ii.
```python
path = os.path.join(folder, 'data_structure_%s_%s.mat' % (anm, date))
...
if trials.size < MIN_TRIALS: return None
...
if use.sum() < MIN_UNITS: return None
```

iii. The agent cited the loader scripts for session identity and the Methods statement requiring at least ten units; the decoder specification motivated the two-trial minimum.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the trial table. Boolean Bpod fields are resized to that count, retained trial indices are obtained with `np.flatnonzero(keep)`, and 1-based spike trial labels are matched with `trials + 1`.

ii.
```python
ntrials_all = int(np.asarray(bp['Ntrials']).reshape(-1)[0])
trials = np.flatnonzero(keep)
starts = np.searchsorted(utrial, trials + 1, side='left')
```

iii. The trajectory recognized Bpod fields and cluster trial labels as explicit trial assignments, so boundaries did not need reconstruction.

## 1-e. How are trials filtered based on quality controls?

i. It keeps non-early, non-stimulation trials with a finite go cue and a hit/miss/no outcome. After neural processing it additionally drops any retained trial whose entire smoothed population is zero.

ii.
```python
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
...
has_spikes = rates.sum(axis=(0, 1)) > 0
rates = rates[:, :, has_spikes]
trials = trials[has_spikes]
```

iii. The paper's conditions motivated early/stimulation exclusion. Decoder warnings and inspection of two truncated JEB24 recordings motivated dropping 61 all-zero late trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: each unit's `trial`, `trialtm`, and `quality`, plus `bp.ev.goCue` for alignment and probe location metadata for regions.

ii.
```python
utrial = np.asarray(unit['trial'], dtype=float).reshape(-1)
utm = np.asarray(unit['trialtm'], dtype=float).reshape(-1)
aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
```

iii. The agent identified these as the fields used by the authors' spike-alignment pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed per unit/trial, divided by 0.01 s to yield Hz, and smoothed over time using a port of the authors' causal half-Gaussian `mySmooth` with `N=15` and reflected prepending. Selected probes are concatenated.

ii.
```python
dat[:, k] = bin_spikes(aligned, edges)
dat = my_smooth(dat / DT)
rates = np.stack(rates, axis=1)
```

iii. The agent cited `WorkingWithDataObjs.m`, Extended Data Figure 2, and Figure 3 scripts as using `params.smooth=15`, and deliberately interpreted `mySmooth` as causal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It rejects quality labels `garbage`, misspelled `gabrga`, `noisy`, and `real?`, retains unlabeled and `poor` units, then removes units with mean processed rate at or below 1 Hz. It drops a session if fewer than ten units remain.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
if quality.lower() in BAD_QUALITY: continue
...
use = rates.mean(axis=(0, 2)) > LOW_FR
```

iii. The trajectory inspected `findClusters` and all observed quality strings, changed the code to retain unlabeled clusters, and cited the paper's `>1 Hz` and `>=10 units` criteria.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time is shifted by that trial's go-cue time before binning.

ii.
```python
aligned = utm[starts[k]:stops[k]] - gocue[trials[k]]
dat[:, k] = bin_spikes(aligned, edges)
```

iii. The agent states that this follows the paper's go-cue alignment and correctly distinguishes the auditory cue in DR from water delivery in WC.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses 500 non-overlapping 10 ms bins from -2.5 to +2.5 s. Raw spikes are rebinned into this grid and all video streams are interpolated directly to its bin centers.

ii.
```python
TMIN, TMAX, DT = -2.5, 2.5, 0.01
edges = np.round(np.arange(TMIN, TMAX + DT / 2, DT), 10)
time = edges[:-1] + DT / 2
```

iii. The agent justified 10 ms with `params.dt = 1/100` found in selected author scripts. This conflicts with the human reference's 5 ms setting.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the conversion-defined bin grid rather than a varying raw field; `bp.ev.goCue` establishes zero for the aligned data.

ii.
```python
time = edges[:-1] + DT / 2
time_row = time.astype(np.float32)[None, :]
```

iii. The trajectory describes the sole input as time from the go cue in seconds.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes the 500 bin centers and copies the same one-row vector into every trial.

ii.
```python
inputs.append(time_row.copy())
```

iii. No further transformation was considered necessary because the grid is already relative to the alignment event.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact edges used to histogram aligned spikes, so each input sample corresponds to the same neural bin.

ii.
```python
dat[:, k] = bin_spikes(aligned, edges)
time = edges[:-1] + DT / 2
```

iii. The agent intended one common time grid for neural, input, and behavioral streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses Bpod `R`, `L`, `hit`, `miss`, and `no` trial flags.

ii.
```python
right = (R & hit) | (L & miss)
left = (L & hit) | (R & miss)
```

iii. The agent cites the paper's `getPrevChoice.m`: outcome plus instructed side reconstructs actual lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct trials take the instructed side, misses take the opposite side, and ignore trials are class 2; codes are left 0, right 1, none 2 and are repeated across time.

ii.
```python
lick_dir = np.full(ntrials_all, 2, dtype=np.int8)
lick_dir[left] = 0; lick_dir[right] = 1; lick_dir[no] = 2
out[0, :] = lick_dir[trix]
```

iii. This implements the requested three categories while preserving ignore trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `bp.autowater`.

ii.
```python
aw = np.asarray(bp['autowater'], dtype=float).reshape(-1)
```

iii. The agent identifies autowater as the authors' WC/DR proxy.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It accommodates both 1/2 and 0/1 encodings, maps autowater true to WC (0) and false to DR (1), and repeats the value over time.

ii.
```python
autowater = (aw == 2) if np.nanmax(aw) > 1 else (np.nan_to_num(aw) > 0.5)
context = np.where(autowater, 0, 1).astype(np.int8)
```

iii. The encoding accommodation came from inspection of old and new sessions; the category order follows the prompt.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = as_bool(bp['hit'], ntrials_all)
miss = as_bool(bp['miss'], ntrials_all)
no = as_bool(bp['no'], ntrials_all)
```

iii. The agent used the explicit mutually exclusive Bpod outcome flags.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss maps to incorrect 0, hit to correct 1, and no response to ignore 2; the label is repeated over time.

ii.
```python
outcome = np.full(ntrials_all, 2, dtype=np.int8)
outcome[miss] = 0; outcome[hit] = 1; outcome[no] = 2
```

iii. This directly follows the requested category order.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-camera `tongue` feature's x/y coordinates from `obj.traj`, plus frame times, go cues, and ephys/behavior bitcode fields. It does not use the bottom-camera `top_tongue` or the likelihood channel.

ii.
```python
TONGUE_FEATURE = ('tongue', 0)
xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
```

iii. The final trajectory calls this “side-camera `tongue`” and says it follows the paper, although the human reference combines both tongue views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Side-view x/y are linearly interpolated to 10 ms bin centers, coordinate differences are computed with central/one-sided finite differences, and their Euclidean magnitude is used. It neither divides by elapsed seconds nor applies likelihood filtering, position smoothing, or cross-view normalization/averaging.

ii.
```python
xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
tongue_speed[:, k] = speed(xy[:, 0], xy[:, 1])
...
return np.sqrt(deriv(x) ** 2 + deriv(y) ** 2)
```

iii. The trajectory says velocity streams were interpolated onto bin centers. It did not document a substantive justification for omitting likelihood, real-time differentiation, smoothing, and the second view.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single session-wide 50th percentile of all finite tongue-speed timepoints defines class 0 below and class 1 at/above; NaN is class 2.

ii.
```python
thresh = np.percentile(x[ok], 50)
out[ok] = (x[ok] >= thresh).astype(np.int8)
```

iii. The agent explicitly cites the prompt's per-session median split and separate not-visible class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It estimates one session video-clock offset, computes `frameTimes - vidshift - goCue`, then interpolates x/y onto the neural bin centers.

ii.
```python
return ft - vidshift - align_time, ts
xy = interp_trace(tt, ts[:, 0:2, tongue_ix], time)
```

iii. The agent cites `findVideoOffset.m` and intended the shared grid to align camera and neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses x/y coordinates for both bottom-camera `top_paw` and `bottom_paw`, plus frame times and clock/alignment fields; likelihood is ignored.

ii.
```python
PAW_FEATURES = [('top_paw', 1), ('bottom_paw', 1)]
```

iii. The agent cited the Methods statement that paws were tracked in the bottom view, but did not justify averaging two distinct paws. The reference uses only `top_paw` because it is reliably tracked.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw's x/y is interpolated to bin centers, interior gaps are nearest-filled, finite-difference displacement magnitude is computed, and available paw speeds are averaged. There is no likelihood cut, Gaussian smoothing, or division by time.

ii.
```python
xy = np.stack([fill_interior_nans(xy[:, 0]), fill_interior_nans(xy[:, 1])], axis=1)
sp.append(speed(xy[:, 0], xy[:, 1]))
paw_speed[:, k] = np.nanmean(np.stack(sp, axis=1), axis=1)
```

iii. The final trajectory describes “mean of bottom-camera `top_paw`/`bottom_paw`”; no evidence-backed rationale for this deviation appears.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the same session-wide finite-value median split, with NaN as class 2.

ii.
```python
paw_cls, paw_thresh = discretize(paw_speed)
```

iii. This follows the prompt's thresholding rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-view frame times are clock-corrected and go-cue shifted, then paw coordinates are interpolated to the 10 ms neural centers.

ii.
```python
tt, ts = trial_video_times(views[1][trix], vidshift, gocue[trix])
xy = interp_trace(tt, ts[:, 0:2, ix], time)
```

iii. The agent intended this to reproduce `findVideoOffset` and share the neural grid.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads per-trial motion-energy traces from `motionEnergy_<animal>_<date>.mat`, and uses side-camera frame times plus bitcode/go-cue fields for timing.

ii.
```python
me = loadmat_var(me_path, 'me')
me_data = me.get('data') if isinstance(me, dict) else me
```

iii. The trajectory notes these standalone files provide the already-reduced motion-energy signal.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Nested data is unwrapped at most twice, linearly interpolated to bin centers, interior NaNs are nearest-filled, and no spatial/temporal feature recomputation is done.

ii.
```python
if isinstance(me_data, dict): me_data = me_data.get('data')
me_trace[:, k] = fill_interior_nans(interp_trace(tt[:n], y[:n], time))
```

iii. The agent reasoned motion energy was already one value per frame and should not be recomputed; it did not justify filling missing values, which the reference avoids.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. All finite session timepoints are split at their median; below is 0, at/above is 1, and missing/no-video is 2.

ii.
```python
me_cls, me_thresh = discretize(me_trace)
```

iii. This directly follows the prompt's per-session 50th-percentile requirement.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The code uses corrected side-camera frame times, subtracts the trial go cue, and interpolates the corresponding trace onto neural bin centers.

ii.
```python
tt, _ = trial_video_times(traj[0][trix], vidshift, gocue[trix])
me_trace[:, k] = ... interp_trace(tt[:n], y[:n], time)
```

iii. The agent recognized that motion energy has one sample per side-camera frame and applied the same video clock correction as tracking.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Boolean arrays are resized to `Ntrials`; missing motion/video remains NaN then becomes class 2; missing frame times fall back to a nominal 400 Hz clock and 0.5 s shift; interior paw and motion-energy gaps are nearest-filled; mismatched lengths are truncated; and all-zero neural trials are removed.

ii.
```python
if x.size != n: x = np.resize(x, n)
...
ft = (np.arange(ts.shape[0]) + 1) / 400.0
...
n = min(tt.size, y.size)
```

iii. The trajectory investigated truncated spike recordings and missing/frame-count anomalies. It aimed to retain usable trials, but several fallbacks fabricate or resize data rather than transparently marking it missing as the reference does.

## 11-a. What are the most time-consuming steps of the code?

i. MAT loading and the nested session/unit/trial video and spike processing dominate; the trajectory shows full conversions taking many minutes and repeated long waits.

ii.
```python
obj = load_obj(path)
for unit in units:
    for k in range(trials.size):
```

iii. The agent did not provide a formal profile, but its repeated conversion runs and waits show loading plus per-session processing were costly.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit inner loop over trials for spike histograms could be replaced by a 2-D trial/time histogram. Some output assembly and interpolation loops are inherently complicated by ragged trial data, but the spike loop is clearly vectorizable.

ii.
```python
for k in range(trials.size):
    if stops[k] > starts[k]:
        dat[:, k] = bin_spikes(aligned, edges)
```

iii. The trajectory focused on correctness and did not document why it retained this avoidable loop; the reference vectorizes spike counting.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly calls `trial_video_times` and interpolates the same bottom-camera timeline separately for each paw, and repeats complete conversion/validation runs during development. Within a final run, session loading and video offset calculation occur once per session.

ii.
```python
for ix, _ in paw_ix:
    xy = interp_trace(tt, ts[:, 0:2, ix], time)
```

iii. The agent did not discuss repeated computation explicitly; its design does reuse the one session offset, while repeated feature interpolation remains.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It stores and computes `qualities`, single-unit counts, multiple thresholds, extensive session metadata, and probe detail not used by decoder tensors. The general MAT reader also materializes many unused source fields.

ii.
```python
qualities.append(quality)
...
'n_single_units': int(sum(...)),
'tongue_velocity_threshold': float(tongue_thresh),
```

iii. The trajectory used some of these values for audit statistics and debugging, so they were useful during development, but they are not consumed by downstream decoding.
