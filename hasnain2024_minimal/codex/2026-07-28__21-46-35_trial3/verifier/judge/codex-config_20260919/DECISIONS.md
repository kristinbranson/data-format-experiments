# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 12 Figure 8 ALM context sessions and one probe per session, all from `data/Ephys_Behavior`. It opens each `data_structure_*.mat` with `h5py` and loads the corresponding standalone motion-energy file with `scipy.io.loadmat`.

ii.
```python
DATA_DIR = ROOT / "data" / "Ephys_Behavior"
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]
sessions = [load_session(spec) for spec in SESSION_SPECS]
with h5py.File(spec.data_path, "r") as f:
```

iii. The trajectory says the AI chose the Figure 8 roster because it considered that the paper's curated two-context cohort and preferred it over globbing files with autowater trials. It reported 12 sessions as a sanity target.

## 1-b. How are the data split into subjects?

i. Subject identity is the `animal` field hard-coded in each `SessionSpec`; unique animals are retained in first-appearance order and each session receives an index into that list.

ii.
```python
subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in sessions])
```

iii. The AI preserved the seven packaged subject IDs and explicitly declined to collapse them even though the manuscript said six mice, because it found no evidence for renaming.

## 1-c. How are the data split into sessions?

i. Each hard-coded animal/date pair and probe is treated as one session and becomes one element of `neural`, `input`, and `output`. Only 12 fixed-folder sessions are used.

ii.
```python
@property
def stem(self) -> str:
    return f"{self.animal}_{self.date}"
"neural": [sess["neural"] for sess in sessions],
```

iii. The AI justified the selection as the Figure 8 two-context analysis roster, not the complete authors' loading-script roster.

## 1-d. How are the data split into trials?

i. Trials are the row/index positions in Bpod fields up to `Ntrials`. Included trial indices are used consistently to select spikes and behavioral outputs.

ii.
```python
"Ntrials": int(round(read_h5_scalar(bp_group["Ntrials"]))),
included_trials = np.flatnonzero(included_mask)
for col, tr in enumerate(included_trials):
    spike_mask = session_trial == tr
```

iii. The trajectory identifies the Bpod fields and cluster `trial` values as the direct trial mapping, so it did not reconstruct boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI retains only hit or miss trials that are neither early-lick nor photostimulation trials. Thus ignore/no-response trials are discarded. It checks that at least two trials remain, but does not cut trials occurring after ephys recording ends.

ii.
```python
def trial_selector(bp):
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)
```

iii. The AI said the paper omitted ignore trials and interpreted outcome as binary, despite the task explicitly requesting an `ignore` category. Early and stimulation exclusion were intended to follow the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the selected `obj.clu` probe's per-cluster `trial`, `trialtm`, and `quality`, plus per-trial `bp.ev.goCue` for alignment.

ii.
```python
clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
session_trial, _trialtm, aligned = load_trial_spikes(..., bp[ALIGN_EVENT])
```

iii. The trajectory correctly mapped raw spikes to `trial`/`trialtm` and selected probes from the manuscript loaders.

## 2-b. How is the `neural` data processed?

i. Aligned spikes are histogrammed into 10 ms bins, divided by `DT` to obtain Hz, and passed through a causal half-Gaussian implementation (`my_smooth`, window 15) separately for each trial.

ii.
```python
counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. The AI believed it was reproducing Figure 8's `mySmooth.m` single-trial pipeline rather than storing raw counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It removes clusters labeled `garbage`, `gabrga`, `noisy`, or `real?`, then retains units whose mean across seven condition PSTHs exceeds 1 Hz. It does not exclude `poor`.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
mean_fr = compute_psth_mean_fr(...)
if mean_fr <= LOW_FR:
    continue
```

iii. The AI cited `findClusters(..., {'all'})` and the Figure 8 `>1 Hz` setting, and used the reported 522 units as a check (obtaining 520).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's `trialtm` is reduced by the go-cue time for its own trial before binning.

ii.
```python
aligned = trialtm - align_times[session_trial]
```

iii. The AI identified `goCue` as the required and paper-used alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 10 ms bins and 550 time points spanning -3.0 to 2.5 seconds. Spikes are directly binned and video streams are interpolated to bin centers.

ii.
```python
TMIN, TMAX = -3.0, 2.5
DT = 1 / 100
edges = np.arange(TMIN, TMAX + DT, DT)
```

iii. The AI chose the Figure 8 script's `dt=10 ms` and window, believing these were the applicable reference settings.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed vector of 10 ms bin centers relative to zero (go-cue onset), rather than a raw field beyond the choice of `goCue` as alignment event.

ii.
```python
time = edges[:-1] + DT / 2
session_input = [np.asarray(time[None, :], dtype=np.float32) ...]
```

iii. The AI used the time axis it selected from the Figure 8 parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It creates equally spaced edges from -3.0 to 2.5 seconds and takes adjacent-bin centers. No trial-dependent processing occurs.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. No separate justification was given beyond matching the selected temporal grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `edges` define spike histograms and `time` is their center vector, repeated for every trial.

ii.
```python
counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
np.asarray(time[None, :], dtype=np.float32)
```

iii. The common grid was intended to guarantee exact alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It uses `bp.hit` and instructed side `bp.R`; because only hit/miss trials remain, a hit uses the instructed side and a miss uses its opposite.

ii.
```python
hit = bool(bp["hit"][trial_idx])
is_right_trial = bool(bp["R"][trial_idx])
if hit: return int(is_right_trial)
return int(not is_right_trial)
```

iii. The AI explicitly aimed to label actual response direction, not instructed direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It encodes left as 0 and right as 1 and repeats the scalar over all time bins. There is no `none` class because ignore trials were filtered out.

ii.
```python
np.full(tongue_speed.shape[0], lick_dir, dtype=np.int16)
"output_values": [["left", "right"], ...]
```

iii. The AI reasoned that hits lick the instructed port and misses lick the opposite port, but treated the requested no-lick cases as excluded.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes directly from the per-trial `bp.autowater` flag.

ii.
```python
"autowater": read_h5_vector(bp_group["autowater"])
context = 0 if bp["autowater"][tr] else 1
```

iii. The AI interpreted autowater as WC and its absence as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater=True` becomes WC (0), otherwise DR (1), repeated over time.

ii.
```python
np.full(tongue_speed.shape[0], context, dtype=np.int16)
"output_values": [["left", "right"], ["WC", "DR"], ...]
```

iii. This was treated as a direct relabeling.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived only from `bp.hit` after the selector has restricted trials to hit or miss; hit is correct and every retained non-hit is incorrect.

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
```

iii. The AI read the paper's omission of ignores as grounds to make this binary.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Incorrect is 0 and correct is 1, repeated over time; no ignore code/value is emitted.

ii.
```python
np.full(tongue_speed.shape[0], outcome, dtype=np.int16)
"output_values": [..., ["incorrect", "correct"], ...]
```

iii. The AI prioritized its interpretation of the paper over the task's explicit three-category outcome.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses bottom-camera DLC `top_tongue` and `bottom_tongue` x/y coordinates, their frame times and dropped-frame marker, plus clock-offset fields and go cue.

ii.
```python
feat_indices = {"top_tongue": ..., "bottom_tongue": ...}
top_tongue_xvel, top_tongue_yvel = velocities["top_tongue"]
```

iii. The AI called averaging the two bottom-view tongue landmarks the least arbitrary reduction of tracking geometry. It did not use the side-view tongue as the reference does.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Coordinates are linearly interpolated to the 10 ms grid. Gradients are computed per coordinate; NaN tongue velocities become zero. The two landmark velocity components are averaged, then their Euclidean speed is taken.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
xvel[:, tr] = np.gradient(tsinterp[:, 0])
tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
tongue_speed = np.nan_to_num(np.sqrt(tongue_tip_xvel**2 + tongue_tip_yvel**2), nan=0.0)
```

iii. The AI said this followed aligned DLC processing and preserved a usable trace, although it does not apply likelihood filtering or the reference's smoothing and two-camera normalization.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session threshold is the 50th percentile of strictly positive speed values on included trials. Every bin is coded 0 below it and 1 at/above it; no not-visible class exists.

ii.
```python
positive_tongue = tongue_use[tongue_use > 0]
tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
(tongue_speed[:, tr] >= tongue_thresh).astype(np.int16)
```

iii. After validation showed the all-bin median was zero and produced a constant class, the AI deliberately changed to the positive-only median. It acknowledged this as decoder-specific.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video time is corrected by a session offset and trial go cue, then positions are interpolated directly onto neural bin centers.

ii.
```python
vidshift = mode(bitcode.bitstart) / fs - mode(bp.ev.bitStart)
interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The trajectory says this mirrors `findVideoOffset.m` and the repository's video alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera DLC x/y tracks for both `top_paw` and `bottom_paw`, with frame times, dropped-frame information, video offset, and go cue.

ii.
```python
"top_paw": find_feat_index(...),
"bottom_paw": find_feat_index(...),
```

iii. The AI chose to average both tracked paws, whereas the reference uses only reliably tracked `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It interpolates and nearest-fills paw positions, computes gradients, subtracts a median derivative baseline (with the y component erroneously subtracting `basederiv[0]`), nearest-fills velocities, calculates each paw's speed, and averages the two speeds.

ii.
```python
xvel[:, tr] -= basederiv[0]
yvel[:, tr] -= basederiv[0]
paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
```

iii. The AI described the mean of two paws as its geometry reduction, but did not discuss the apparent y-baseline indexing error.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The threshold is the session median over all included-trial bins. Bins are binary low/high; missing values have already been zero-filled and are not a third class.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_use, 50))
(paw_speed[:, tr] >= paw_thresh).astype(np.int16)
```

iii. The AI viewed this as the literal requested 50th-percentile split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame coordinates are clock-corrected and interpolated to the same 10 ms bin centers as neural data.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The shared offset/grid was intended to match the paper's video alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads each session's standalone `motionEnergy_<animal>_<date>.mat` `me.data`; side-camera frame times provide timestamps. `me.moveThresh` is read only for metadata/sanity output.

ii.
```python
me = mat["me"]
trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
frame_times_by_trial = [read_frame_times(f, side_group, tr) ...]
```

iii. The AI found embedded motion energy inconsistent and followed the paper loader's standalone files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Raw per-frame motion energy is linearly interpolated onto 10 ms centers, and missing/edge samples are nearest-filled before thresholding.

ii.
```python
out[:, tr] = interp_with_nan(..., me_trial, taxis)
out[:, tr] = fill_nearest_1d(out[:, tr])
```

iii. The AI believed interpolation plus nearest-fill reproduced `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A per-session median over included trial bins defines binary low/high classes. There is no `no video` category.

ii.
```python
me_thresh = float(np.nanpercentile(me_use, 50))
(motion_energy[:, tr] >= me_thresh).astype(np.int16)
```

iii. The AI treated the median as the literal decoder specification, after filling missing samples.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are shifted by the session video/behavior clock offset and the trial go cue, then interpolated onto neural bin centers.

ii.
```python
interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
```

iii. The AI explicitly matched the bitcode-offset formula from the authors' code.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Interpolation ignores NaN samples; insufficient data yields all NaNs. Non-tongue tracks and motion energy are nearest-filled, with an entirely missing trace becoming zeros. Tongue velocity NaNs also become zeros. Trials with NaN `NdroppedFrames` are skipped. Missingness is therefore imputed into binary low/high values instead of represented by the required third classes.

ii.
```python
if idx.size == 0:
    return np.zeros_like(x)
if np.isnan(ndropped):
    continue
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
```

iii. The trajectory noticed all-NaN feature trials and chose to tolerate them with zero-fill. Its stated goal was avoiding invalid output and degenerate training labels.

## 11-a. What are the most time-consuming steps of the code?

i. The code does not instrument substeps, but its dominant work is repeated HDF5 dereferencing/reading, per-feature trial interpolation, and per-unit/per-condition spike histograms and convolutions. The trajectory repeatedly waited on full conversion and described file/layout handling as the main practical concern.

ii.
```python
for clu_idx in range(qds.shape[0]):
    mean_fr = compute_psth_mean_fr(...)
    neural_by_trial.append(build_neural_trials(...))
```

iii. No explicit benchmark-based justification was recorded; the AI focused on successful full conversion and validation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Loops over trials in motion interpolation, feature alignment, velocity, output construction, and spike construction, plus output-column smoothing and clusters, are candidates. In particular, spike counting loops over every included trial for every retained cluster, unlike the reference's per-cluster 2-D histogram.

ii.
```python
for tr in range(n_trials): ...
for tr in range(xpos.shape[1]): ...
for col, tr in enumerate(included_trials):
    spike_mask = session_trial == tr
```

iii. The trajectory did not discuss vectorization; it prioritized faithful MATLAB-style operations and robustness to ragged raw arrays.

## 11-c. What processing does the code repeat multiple times?

i. HDF5 reference arrays and per-trial feature metadata are reread frequently. Every cluster recomputes condition membership and histograms, and every feature independently rereads frame times and builds the aligned coordinates. Neural smoothing is performed once for filtering condition PSTHs and again for exported single trials.

ii.
```python
refs = np.asarray(dataset[()], dtype=object).reshape(-1)
for feat_name, feat_idx in feat_indices.items():
    xpos, ypos = aligned_feature_position(...)
```

iii. No justification about repeated processing appears in the trajectory; reuse was implemented only for the session offset and resulting velocity dictionaries.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads unused Bpod `L`, `sample`, and `delay`; computes/stores unused `kept_cluster_indices`, `prefilter_unit_count`, and `cluster_ids`; reads raw motion-energy `moveThresh` only for a sanity record; computes seven condition PSTHs solely to reduce them to one filtering scalar; and writes a sample dataset in addition to the requested full dataset.

ii.
```python
"L": read_h5_vector(bp_group["L"]),
kept_cluster_indices.append(clu_idx)
cluster_ids.append(clu_idx)
raw_motion_energy, raw_move_thresh = load_motion_energy(...)
```

iii. The extra sample dataset and sanity fields were added for reproducible validation; the other unused intermediates were not justified.
