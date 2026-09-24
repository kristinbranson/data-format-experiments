# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 two-context ALM sessions in `CONTEXT_SESSION_SPECS`, all from `data/Ephys_Behavior`, loads each complete v7.3 session with `mat73`, and loads its companion motion-energy file with SciPy. It does not load the randomized-delay sessions or the other fixed-delay neural sessions.

ii. `CONTEXT_SESSION_SPECS = [SessionSpec("JEB6", "2021-04-18", 1), ...]`

```python
obj = mat73.loadmat(spec.data_path)["obj"]
me = load_motion_energy(spec)
```

iii. The notes say the raw directory is a superset and choose the 12-session Figure 8/context subset because behavioral context is a requested output and varies there.

## 1-b. How are the data split into subjects?

i. Each `SessionSpec` explicitly stores the animal ID; unique IDs are sorted into `subjects`, and each session is mapped to its subject index.

ii.
```python
subjects = sorted({sess["subject"] for sess in converted_sessions})
"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in converted_sessions])
```

iii. The notes describe normalizing loader metadata to mouse IDs. The resulting subset contains seven subjects.

## 1-c. How are the data split into sessions?

i. Every hard-coded animal/date pair is one session and becomes one element of each top-level session list.

ii.
```python
"neural": [sess["neural"] for sess in converted_sessions],
"input": [sess["input"] for sess in converted_sessions],
"output": [sess["output"] for sess in converted_sessions],
```

iii. The agent identifies the figure-specific session list as the clearest code path for context analysis.

## 1-d. How are the data split into trials?

i. Raw trial arrays are indexed by zero-based Bpod trial index. After filtering, each retained index is used consistently for spikes, labels, video, and motion energy, and each becomes one list element.

ii.
```python
valid_trials = select_valid_trials(obj)
for local_trial_idx, trial_idx in enumerate(kept_trials):
    session_neural.append(neural_trial.astype(np.float32))
```

iii. The notes treat the Bpod fields and trial-indexed streams as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they are hit or miss, not early, not `no`, and not stimulated; trials without a first left/right lick at or after go cue are then removed. Ignore trials and trials with no qualifying lick are excluded.

ii.
```python
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
...
if lick_dir is None:
    continue
```

iii. The notes say this follows the paper's omission of early/ignore trials and keeps outcome well-defined.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the selected probe's `obj.clu`: per-unit `trialtm`, `trial`, and `quality`, plus `bp.ev.goCue` for alignment.

ii.
```python
clu = obj["clu"][spec.probe_index]
trialdat = bin_unit_spikes(clu["trialtm"][unit_idx], clu["trial"][unit_idx], align_times, kept_trials)
```

iii. The agent correctly identifies electrophysiological spike times and the loader-selected ALM probe as the intended neural source.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned, counted in 5-ms bins, divided by 0.005 to form Hz, and causally Gaussian-smoothed with a 15-sample window and reflected prefix. Values are stored as `float32` neuron-by-time arrays.

ii.
```python
rates = aligned_counts / DT
rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
```

iii. The notes state that this reproduces `alignSpikes`, `getSeq`, and reference `mySmooth`, with no baseline normalization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Quality labels exactly equal to `garbage`, `gabrga`, `noisy`, or `real?` are excluded (case-sensitive); surviving units must have mean aligned-window firing rate strictly above 1 Hz. `poor` is not excluded.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. The notes justify these as junk-quality removal and the paper's greater-than-1-Hz criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time has that trial's `bp.ev.goCue` subtracted before bin assignment.

ii.
```python
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
```

iii. The agent notes that go cue is both requested and the reference default; it treats WC `goCue` as the equivalent stored event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 5 ms over −2.5 to +2.5 s (1000 bins). Spikes are binned directly once; no later temporal rebinning is applied.

ii.
```python
DT = 0.005
TIME_AXIS = np.arange(TMIN, TMAX, DT) + DT / 2.0
```

iii. The agent resolves conflicting examples in favor of the paper/default 5-ms grid.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common bin-center grid rather than a trial-varying raw field; `bp.ev.goCue` defines zero for the aligned streams.

ii. `TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0`

iii. The notes say the requested input is the common aligned time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code forms 5-ms bin centers from −2.4975 through +2.4975 s and repeats the same one-row array for every trial.

ii. `time_input = TIME_AXIS[None, :].astype(np.float32)`

iii. This gives the decoder a continuous time-from-event value at every neural bin.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the same bins used to count go-cue-aligned spikes, so corresponding columns align exactly.

ii.
```python
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
time_input = TIME_AXIS[None, :]
```

iii. The common grid was chosen to make all streams share the neural time base.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from per-trial left and right lick event times, `bp.ev.lickL` and `bp.ev.lickR`, and go-cue time.

ii.
```python
lick_l = event_list_to_array(obj["bp"]["ev"]["lickL"][trial_idx])
lick_r = event_list_to_array(obj["bp"]["ev"]["lickR"][trial_idx])
```

iii. The agent argues that actual behavior is preferable to instructed side, particularly on error trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Pre-go licks are discarded, the earliest remaining side is encoded left=0/right=1, ties go right, and no-lick trials are removed. The label is repeated across time.

ii.
```python
lick_l = lick_l[lick_l >= go_time]
lick_r = lick_r[lick_r >= go_time]
return 0 if first_l < first_r else 1
```

iii. The notes call this the actual first post-alignment lick and deliberately omit a no-lick class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It comes directly from `bp.autowater`.

ii. `autowater = ensure_1d_numeric(bp["autowater"])`

iii. The notes identify this as the exact context variable used in the paper/code.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Nonzero autowater is encoded WC=0 and zero is DR=1, then repeated across time.

ii. `0 if autowater[trial_idx] != 0 else 1`

iii. This is a direct relabeling matching the requested category order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The retained-trial outcome uses `bp.hit`; selection also reads `bp.miss` and `bp.no`.

ii.
```python
hit = ensure_1d_numeric(bp["hit"])
valid = (~no) & (hit | miss)
```

iii. The agent restricts conversion to trials where outcome is hit or miss.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hits are correct=1 and retained non-hits are incorrect=0, repeated across time. Ignore=2 is never represented because ignore trials are dropped.

ii. `1 if hit[trial_idx] != 0 else 0`

iii. The notes say excluding ignores makes the outcome well-defined, despite the prompt explicitly listing ignore.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the side-view `tongue` feature's x/y coordinates and frame times in `obj.traj[0]`, plus bitcode timing and go cues. DLC likelihood is not explicitly used.

ii. `tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)`

iii. The plan says only exact needed features are extracted, although it planned side-view tongue rather than combining views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Valid x/y samples are linearly interpolated onto the 5-ms grid; coordinate gradients are computed and combined as Euclidean magnitude. Missing tongue gradients become zero. Position smoothing is skipped for tongue.

ii.
```python
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. The agent describes scalar speed magnitude as the most defensible collapse of x/y velocity.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session median is computed over positions marked visible in kept trials. Values at/above it and visible become 1; everything else, including invisibility, becomes 0. There is no class 2.

ii.
```python
tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. Notes acknowledge that invisible periods initially dominated and state they were assigned to the low bin to keep output binary and avoid validator warnings.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video offset is computed from behavior and ephys bitcodes; frame time minus offset minus trial go cue is interpolated at neural bin centers.

ii. `shifted_time = frame_times - vidshift - align_times[trial_idx]`

iii. The notes follow the reference video-clock correction and universal go-cue alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses both `top_paw` and `bottom_paw` x/y trajectories from bottom view (`obj.traj[1]`), plus timing fields.

ii. `for paw_feat in ("top_paw", "bottom_paw"):`

iii. The planning notes explicitly propose averaging the two bottom-view paw features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw is interpolated, gradients are baseline-corrected and nearest-filled, converted to speed magnitude, then the two paw speeds are averaged wherever available.

ii.
```python
xv = xv - basederiv[0]
yv = yv - basederiv[0]
np.divide(np.nansum(paw_stack, axis=0), paw_counts, out=paw_speed, where=paw_counts > 0)
```

iii. The agent intended scalar magnitude and a mean across the two named paw traces.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The median over kept-session values is the threshold; values below are 0 and values at/above are 1. NaNs compare false and also become 0, with no not-visible class.

ii. `(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64)`

iii. This implements the planned binary session-median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-view frame times are clock-corrected and go-cue-relative, then positions are interpolated to the neural bin centers before differentiation.

ii. `shifted_time = frame_times - vidshift - align_times[trial_idx]`

iii. The common time base is intended to align every output to neural columns.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads per-trial `me.data` from the standalone motion-energy file and uses side-camera frame times plus bitcode/go-cue timing. `moveThresh` is loaded but unused.

ii.
```python
me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
return {"data": np.atleast_1d(me.data), "moveThresh": float(me.moveThresh)}
```

iii. Notes identify the companion motion-energy traces and reference interpolation procedure.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Frame trace and time are truncated to their common length, finite samples are linearly interpolated onto bin centers, and missing ends are nearest-filled.

ii.
```python
n = min(frame_times.size, me_trial.size)
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

iii. The agent says this matches `loadMotionEnergy` alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The kept-session median is used; below is 0 and at/above is 1. Missing values become 0 through comparison, with no no-video class.

ii. `(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64)`

iii. This follows the agent's planned binary session-median split but not the requested third category.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the session video offset and trial go cue, then interpolated at the same 5-ms centers as neural data.

ii. `shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]`

iii. The notes use the reference clock correction and shared time axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing frame times are synthesized at 400 Hz when trajectory length is known; streams are truncated on length mismatch; coordinate gaps are interpolated/nearest-filled; tongue missingness becomes low velocity; paw/motion NaNs also become class 0. Trials with no qualifying lick are dropped.

ii.
```python
frame_times = (np.arange(ts.shape[0]) + 1.0) / 400.0
out[:, col] = fill_nearest_1d(out[:, col])
```

iii. Notes favor interpolation and validator-compatible binary outputs; they report fixing an all-low tongue issue by computing the median only over visible samples.

## 11-a. What are the most time-consuming steps of the code?

i. Full MATLAB-object loading and trial-by-trial kinematic interpolation are identified as dominant costs; measured conversion is about 7.4 s/session.

ii. `obj = mat73.loadmat(spec.data_path)["obj"]`

iii. The notes explicitly call full-object loading simple but not lean and identify Python kinematic loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops in feature alignment, velocity, motion-energy alignment, and final trial assembly, plus the per-unit loop, remain. Spike assignment within a unit is vectorized with `np.add.at`.

ii.
```python
for trial_idx in range(n_trials):
...
np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
```

iii. The notes say ragged kinematic data remain trial-by-trial while vectorized spike binning removes nested time-bin loops.

## 11-c. What processing does the code repeat multiple times?

i. Feature lookup and full per-trial alignment are repeated separately for tongue, top paw, and bottom paw; `compute_vidshift` is called once in `convert_session` and again inside motion-energy alignment; arrays are traversed again for plotting when requested.

ii.
```python
vidshift = compute_vidshift(obj)
...
def align_motion_energy(...):
    vidshift = compute_vidshift(obj)
```

iii. The agent notes limiting extraction to three features, but does not document the duplicated offset calculation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `mat73` materializes the whole session although only selected fields are used; `moveThresh`, `kept_unit_indices`, thresholds/summaries, and plotting intermediates are computed but not included in the final decoder dataset. Motion-energy `moveThresh` is never used.

ii.
```python
kept_unit_indices.append(unit_idx)
return {"data": ..., "moveThresh": float(me.moveThresh)}
```

iii. The notes acknowledge full-object loading as unnecessary overhead and say feature extraction was otherwise narrowed to requested signals.
