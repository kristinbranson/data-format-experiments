# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded a 12-session two-context subset in `CONTEXT_SESSION_SPECS` and loaded only `data/Ephys_Behavior/data_structure_<session>.mat` with `mat73`, plus `motionEnergy_<session>.mat` with `scipy.io.loadmat`. It did not search both ephys folders or support the reference's mixed MAT-file session loader.

ii. ```python
CONTEXT_SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 1),
    ...
    SessionSpec("JEB19", "2023-04-18", 0),
]

obj = mat73.loadmat(spec.data_path)["obj"]
me = load_motion_energy(spec)
```

iii. Step 5 and Step 6 in `CONVERSION_NOTES.md` say the agent intentionally converted the Figure 8 two-context ALM subset because context varies there, and it describes `mat73` session loading plus `loadmat` motion-energy loading as a simple implementation choice.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the session spec's animal id (`spec.animal`) and then deduplicated/sorted when assembling the dataset.

ii. ```python
result = {
    "session_id": spec.session_id,
    "subject": spec.animal,
    ...
}

subjects = sorted({sess["subject"] for sess in converted_sessions})
```

iii. Step 5 says loader/session metadata are normalized to mouse IDs and carried through to `subjects`/`subject_idx`.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` entry is treated as one session, with one session file and one companion motion-energy file, all from `data/Ephys_Behavior`.

ii. ```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe_index: int

    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"
```

iii. Step 5 says the agent intentionally used the Figure 8 context-analysis session list as the session definition.

## 1-d. How are the data split into trials?

i. Trials are indexed directly from per-trial Bpod arrays. `select_valid_trials` builds a trial-index list, and per-trial labels/neural/video slices are then assembled by iterating over those trial indices.

ii. ```python
def select_valid_trials(obj: dict) -> np.ndarray:
    ...
    return np.flatnonzero(valid)

for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
```

iii. The AI did not give a separate written rationale here beyond using the Bpod trial table and per-trial event arrays directly.

## 1-e. How are trials filtered based on quality controls?

i. The AI kept only hit or miss trials, excluding early trials, ignore/no-response trials, and stimulation trials, and it additionally dropped any trial lacking a detectable first post-alignment lick.

ii. ```python
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
...
lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
if lick_dir is None:
    continue
```

iii. Step 5 says this was done to keep outcome well defined and to match the paper's omission of early/ignore trials in context analyses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the selected probe's cluster spike arrays, specifically `clu['trialtm']` and `clu['trial']`, with alignment times from `bp['ev']['goCue']`; unit quality labels are used for filtering.

ii. ```python
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
...
mean_fr = mean_firing_rate_window(
    clu["trialtm"][unit_idx],
    np.asarray(clu["trial"][unit_idx], dtype=np.int64),
    align_times,
)
```

iii. Step 5 maps `obj.clu{probe}[*].trialtm`, `obj.clu{probe}[*].trial`, and `bp.ev.goCue` to the neural output and cites the reference `alignSpikes`/`getSeq` pipeline.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the AI histogrammed spikes into 5 ms bins from -2.5 s to +2.5 s around go cue, divided counts by `DT` to get firing rates, and applied a causal half-Gaussian-style smoother via `my_smooth`.

ii. ```python
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
rates = aligned_counts / DT
rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
```

iii. Step 6 says the agent intended a 'reference-style' 5 ms firing-rate representation with causal Gaussian smoothing matching `mySmooth`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI excluded clusters whose quality string was exactly one of `garbage`, `gabrga`, `noisy`, or `real?`, then retained only units with mean firing rate greater than 1 Hz across the aligned window.

ii. ```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
...
return quality not in QUALITY_EXCLUDE
...
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. Step 5 and Step 10 say this was chosen to match `findClusters(..., {'all'})` and the Figure 8 `lowFR = 1` setting.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the per-trial `goCue` time from each spike's `trialtm`, placing spikes on a go-cue-centered clock.

ii. ```python
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
```

iii. Step 5 states that all signals were aligned to `bp.ev.goCue`, which the agent treated as the universal event for DR and WC trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a 5 ms time step (`DT = 0.005`) and a -2.5 s to +2.5 s window, with no secondary rebinning.

ii. ```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
```

iii. Step 6 says the AI chose 5 ms bins to stay paper-consistent while using a compact go-cue-centered window.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read as a raw data field. It is a synthetic aligned time axis defined by `TMIN`, `TMAX`, and `DT`, intended to represent time relative to each trial's `goCue`.

ii. ```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
...
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. Step 5 explicitly says the common aligned time axis is repeated per trial as the single decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No additional processing is applied beyond creating the 5 ms bin-center vector and repeating it for each trial.

ii. ```python
time_input = TIME_AXIS[None, :].astype(np.float32)
session_input.append(time_input)
```

iii. The notes treat this as a target-format adaptation rather than a transformation of a raw stream.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time input is exactly the same grid used to bin spikes and interpolate behavioral signals, so its bins share the neural time base by construction.

ii. ```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
...
rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
...
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. Step 5 says the common aligned time axis is the decoder input and the neural binning grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derived lick direction from the first post-go-cue lick-port events in `bp.ev.lickL` and `bp.ev.lickR`, not from instructed side plus trial outcome.

ii. ```python
lick_l = event_list_to_array(obj["bp"]["ev"]["lickL"][trial_idx])
lick_r = event_list_to_array(obj["bp"]["ev"]["lickR"][trial_idx])
```

iii. Step 5 says the agent deliberately used the actual first post-alignment lick side because it interpreted the task as asking for true lick direction rather than instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the AI found the earliest lick after go cue and encoded left as 0 and right as 1. If no such lick was found, it dropped the trial entirely rather than emitting a `none` class.

ii. ```python
lick_l = lick_l[lick_l >= go_time]
lick_r = lick_r[lick_r >= go_time]
...
if math.isinf(first_l) and math.isinf(first_r):
    return None
return 0 if first_l < first_r else 1
```

iii. Step 5 says this was a deliberate semantic choice to represent actual behavioral lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived directly from `bp['autowater']`.

ii. ```python
autowater = ensure_1d_numeric(bp["autowater"])
```

iii. Step 5 says `autowater` is exactly the context variable used in the reference code.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabeled `autowater != 0` as WC (0) and `autowater == 0` as DR (1).

ii. ```python
trial_labels.append(
    (
        lick_dir,
        0 if autowater[trial_idx] != 0 else 1,
        1 if hit[trial_idx] != 0 else 0,
    )
)
```

iii. Step 5 explicitly describes the mapping WC = 0 and DR = 1 from `autowater`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is effectively derived from `bp['hit']` after trial filtering has already restricted trials to hit or miss; `miss` only matters indirectly through the earlier valid-trial filter.

ii. ```python
hit = ensure_1d_numeric(bp["hit"])
...
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
...
1 if hit[trial_idx] != 0 else 0,
```

iii. Step 5 says the agent restricted to hit/miss trials so outcome would stay well-defined and binary.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as correct = 1 for hit trials and incorrect = 0 otherwise; ignore is not represented.

ii. ```python
trial_labels.append(
    (
        lick_dir,
        0 if autowater[trial_idx] != 0 else 1,
        1 if hit[trial_idx] != 0 else 0,
    )
)
```

iii. The notes say ignore trials were excluded to keep outcome well-defined and compatible with the chosen hit/miss-only trial set.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived only from the side-camera `tongue` feature in `obj['traj'][0]`, with frame times from that view and session video offset from the bitcode fields.

ii. ```python
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
tongue_vx, tongue_vy = compute_velocity(tongue_x, tongue_y, "tongue")
```

iii. Step 5 says the planned tongue output came from a side-view tongue speed magnitude; the notes do not claim to combine both tongue views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolated tongue x/y positions directly onto the neural time grid, computed per-bin gradients, took speed magnitude, and used visibility only to define the thresholded mask, not a per-run likelihood filtering/smoothing procedure.

ii. ```python
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
ypos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 1], left=np.nan, right=np.nan)
...
tongue_vx, tongue_vy = compute_velocity(tongue_x, tongue_y, "tongue")
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. Step 6 describes this as 'reference-style position interpolation and velocity calculation'; Step 7 records a later change to compute the median from tongue-visible timepoints only.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computed a session median using only tongue-visible bins from kept trials, then emitted a binary 0/1 trace; invisible bins were forced into 0 instead of receiving a separate category.

ii. ```python
if np.any(tongue_visible_kept):
    tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
...
tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. Step 7 and trajectory step 213 say the median was moved to visible-only points because an all-bins median collapsed the tongue output, while invisible periods were intentionally left in the low bin.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI aligned tongue data by subtracting a session-wide video offset and the trial's `goCue`, then interpolated the resulting positions onto the shared `TIME_AXIS` used by neural data.

ii. ```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

iii. Step 5 says all signals share a universal go-cue-centered time axis, with video offset correction coming from the bitcode fields.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera features, `top_paw` and `bottom_paw`, which are converted to speeds and then averaged where finite.

ii. ```python
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
    paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
```

iii. Step 5 explicitly planned to use the mean speed magnitude across `top_paw` and `bottom_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI aligned paw positions onto the neural time grid, took gradients of the interpolated positions, converted them to speed magnitudes, and averaged the two paw streams where available.

ii. ```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_counts = np.sum(np.isfinite(paw_stack), axis=0)
paw_speed = np.full(paw_counts.shape, np.nan, dtype=np.float64)
np.divide(np.nansum(paw_stack, axis=0), paw_counts, out=paw_speed, where=paw_counts > 0)
```

iii. Step 5 says the agent reduced x/y velocity to scalar speed and averaged the two paw features as a single requested output stream.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI thresholded paw speed at the session 50th percentile and output a binary 0/1 trace with no separate `not visible` category.

ii. ```python
paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
...
(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64),
```

iii. Step 5 says continuous movement variables were discretized with per-session medians as required by the task.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned by subtracting the session video offset and per-trial `goCue`, then interpolating onto the common neural `TIME_AXIS`.

ii. ```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

iii. The AI used the same shared video-offset and go-cue alignment strategy for all video-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the companion `motionEnergy_<session>.mat` file, specifically the `me.data` trace.

ii. ```python
me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
me = me_mat["me"]
return {
    "data": np.atleast_1d(me.data),
    "moveThresh": float(me.moveThresh),
}
```

iii. Step 5 says motion energy is taken from the aligned motion-energy stream in the companion file.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligned motion-energy frame values to go cue, interpolated them onto the neural time grid, nearest-filled missing bins, and then thresholded the continuous trace by session median.

ii. ```python
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

iii. Step 6 says motion energy was interpolated onto the neural time axis; Step 5 frames the later thresholding as the task-required discretization.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI used the session median of the aligned motion-energy trace to produce a binary below/above split, with no dedicated `no video` category.

ii. ```python
me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))
...
(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64),
```

iii. Step 5 says all continuous movement variables were median-split per session.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session video offset and the trial's `goCue`, then interpolating onto the shared `TIME_AXIS`.

ii. ```python
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
```

iii. The notes say motion energy shares the same aligned neural time axis as the other streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handled missing timestamps by synthesizing a nominal 400 Hz frame grid, filled missing non-tongue position bins and motion-energy bins by nearest interpolation, and treated missing tongue visibility as the low binary class rather than a distinct missing-data class.

ii. ```python
if frame_times is None:
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
...
x[~good] = np.interp(idx[~good], idx[good], x[good])
...
tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. Step 10 says the fallback 400 Hz grid was used for empty/NaN frame times, and that invisible tongue periods were intentionally kept in the low bin after the visible-only threshold fix.

## 11-a. What are the most time-consuming steps of the code?

i. From the AI's own notes, the heaviest steps are loading full MATLAB session objects with `mat73` and the trial-by-trial kinematic alignment/interpolation loops.

ii. ```python
obj = mat73.loadmat(spec.data_path)["obj"]
...
for trial_idx in range(n_trials):
    ...
    xpos[:, trial_idx] = np.interp(...)
```

iii. Step 6 explicitly lists full-session `mat73` loading and Python-loop kinematic alignment as the main inefficiencies.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI left trial loops in feature alignment/velocity extraction and unit loops in neural processing, while vectorizing spike accumulation within units using `np.add.at`.

ii. ```python
for trial_idx in range(n_trials):
    ...
for unit_idx, use_unit in enumerate(quality_keep):
    ...
np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
```

iii. Step 6 explicitly notes vectorized spike binning as a speedup and trial-by-trial feature extraction as a remaining inefficiency.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several session-level operations: it computes video alignment separately for motion energy and for other kinematic features, aligns each requested feature in its own full trial loop, and allocates the same time input array once per trial.

ii. ```python
vidshift = compute_vidshift(obj)
tongue_x, tongue_y = align_feature_positions(...)
...
motion_energy = align_motion_energy(obj, me, align_times, TIME_AXIS)
```

iii. The AI did not explicitly discuss repeated processing in its notes; this is reconstructed from the code path itself.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Besides loading full MAT objects, the AI computes plotting artifacts and summaries not used in `converted_data.pkl`, keeps an unused `kept_unit_indices` list, reads `moveThresh` but never uses it, and supports optional plot generation unrelated to downstream decoding.

ii. ```python
kept_unit_indices = []
...
return {
    "data": np.atleast_1d(me.data),
    "moveThresh": float(me.moveThresh),
}
...
if make_plot:
    create_processing_plot(...)
```

iii. Step 6 acknowledges that full-object loading is not lean; the rest of this answer is inferred directly from the code.
