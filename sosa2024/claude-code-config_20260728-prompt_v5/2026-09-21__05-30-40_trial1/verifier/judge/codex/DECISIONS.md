# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every `sub-*` directory under `/app/data`, then gathers every `.nwb` file in each subject directory. Each NWB file is treated as one session and is opened directly with `h5py` rather than `pynwb`.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])

all_nwb_files = []
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
    all_nwb_files.extend(nwb_files)
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The rationale is mostly implicit in the code. In the trajectory the agent emphasized that there are 11 subject directories and 152 NWB files, and in `CONVERSION_NOTES.md` Step 2 it documented the one-file-per-session directory structure.

## 1-b. How are the data split into subjects?

i. Subjects are identified from directory names beginning with `sub-`. Session metadata strips the `sub-` prefix and the final dataset stores sorted unique subject IDs.

ii. ```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
...
session_info = {
    'subject': subject.replace('sub-', ''),
    ...
}
...
unique_subjects = sorted(set(si['subject'] for si in session_infos))
```

iii. Step 2 of `CONVERSION_NOTES.md` says the data are organized by subject directories such as `sub-m3` and `sub-m11..m19`.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The session identifier is parsed from the filename.

ii. ```python
nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
...
ses_id = os.path.basename(nwb_path).split('_')[1]
```

iii. The agent documented in `CONVERSION_NOTES.md` Step 2 that each subject has 12-14 NWB files, one per day/session.

## 1-d. How are the data split into trials?

i. Trials are split using samples where `trial_start > 0` as starts and samples where `teleport > 0` as ends. The code pairs these arrays directly after converting both to 1-indexed coordinates.

ii. ```python
trial_start_inds = np.where(trial_start_signal > 0)[0] + 1
teleport_inds = np.where(teleport_signal > 0)[0] + 1

if len(trial_start_inds) != len(teleport_inds):
    min_len = min(len(trial_start_inds), len(teleport_inds))
    trial_start_inds = trial_start_inds[:min_len]
    teleport_inds = teleport_inds[:min_len]
```

iii. In the trajectory the agent repeatedly described alignment as trial-start to teleport. It did not justify using all positive `teleport` samples instead of teleport onsets; that choice is only visible in the code.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit minimum-length or behavioral-quality trial filter. The code only skips degenerate trials where `e <= s`, and later skips whole sessions with fewer than two surviving trials. Lick-sensor correction is applied to the lick output but does not remove trials.

ii. ```python
for i in range(n_trials):
    s = trial_start_inds[i] - 1
    e = teleport_inds[i] - 1

    if e <= s:
        continue
    ...

if len(neural_trials) < 2:
    print(f"    WARNING: Skipping session with {len(neural_trials)} trials")
    continue
```

iii. This matches `CONVERSION_NOTES.md` Step 3/5, which says “All valid trials included” and does not mention the reference solution’s `<50`-timepoint filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from raw fluorescence and neuropil traces in the NWB `Fluorescence` and `Neuropil` groups, not from the NWB `Deconvolved` signal.

ii. ```python
F_plane = ophys[f'Fluorescence/{plane}/data'][:]
Fneu_plane = ophys[f'Neuropil/{plane}/data'][:]
...
events, dff = compute_dff_and_deconvolve(
    F_concat, Fneu_concat, trial_start_inds, teleport_inds,
    imaging_rate, n_planes
)
```

iii. The trajectory explicitly says the NWB deconvolved signal had no NaNs during ITIs and therefore looked like raw suite2p output, so the agent decided to recompute the paper-style signal from `F` and `Fneu`.

## 2-b. How is the `neural` data processed?

i. The script concatenates planes, restricts samples to trial windows, subtracts `0.7 * Fneu`, adds back the per-trial neuropil mean, estimates a maximin baseline with Gaussian smoothing plus min/max filters, computes dF/F, smooths dF/F again, and deconvolves each trial with OASIS.

ii. ```python
f_[:, s:e] = F[:, s:e]
f_neu_[:, s:e] = Fneu[:, s:e]
...
f_ -= NEU_COEF * f_neu_
...
f_[:, s:e] = f_[:, s:e] + NEU_COEF * np.nanmean(f_neu_[:, s:e], axis=1, keepdims=True)
flow[:, s:e] = nansmooth(f_[:, s:e], 15, axis=1)
flow[:, s:e] = minimum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
flow[:, s:e] = maximum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
spks[:, s:e] = oasis(trial_dff, 2000, TAU, frame_rate / n_planes)
```

iii. Step 1 and Step 6 of `CONVERSION_NOTES.md` describe this as a reproduction of the paper’s preprocessing pipeline. In the trajectory the agent explicitly committed to recomputing dF/F and deconvolution rather than trusting the stored NWB deconvolved traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `iscell > 0` are kept. The script does not implement the reference interneuron filter based on correlation of dF/F with running speed.

ii. ```python
iscell_full = ophys['ImageSegmentation/PlaneSegmentation/iscell'][:]
...
iscell_plane = iscell_full[plane_roi_mask, 0]
...
cell_mask_all.append(iscell_plane > 0)
...
cell_indices = np.where(cell_mask)[0]
events_cells = events[cell_indices, :]
```

iii. Step 5 and Step 10 of `CONVERSION_NOTES.md` explicitly justify skipping the interneuron filter because it affects only about 0.42% of cells and would add complexity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial from its `trial_start` sample up to its `teleport` sample and storing each slice as a trial matrix.

ii. ```python
for i in range(n_trials):
    s = trial_start_inds[i] - 1
    e = teleport_inds[i] - 1
    ...
    trial_neural = events_cells[:, s:e].copy()
    neural_trials.append(trial_neural.astype(np.float32))
```

iii. The agent repeatedly described the target alignment as “trial start to teleport” in the trajectory and set metadata `temporal_alignment_event` to `start of trial (trial_start signal)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at one imaging frame per bin. For multi-plane sessions the code treats the effective per-plane rate as `imaging_rate / n_planes`. No additional temporal rebinning is applied.

ii. ```python
effective_rate = imaging_rate
if n_planes > 1:
    effective_rate = imaging_rate / n_planes
...
'time_bin_size': 1000.0 / 15.5078125,
```

iii. Step 5 of `CONVERSION_NOTES.md` says “Time bin = imaging frame” and Step 3 records the expected bin size as about 64.5 ms.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not derived from a stored NWB time series. Instead the code derives it from trial length in frames and the effective imaging rate.

ii. ```python
n_tp = e - s
...
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. Step 5 of `CONVERSION_NOTES.md` says this variable is formed as `(frame_idx - trial_start_idx) / imaging_rate`, so the agent chose frame-count time rather than behavioral timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code creates a regularly spaced sequence starting at 0 with increments of `1 / effective_rate` seconds.

ii. ```python
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
input_tv = time_from_start.reshape(1, -1)
```

iii. The justification in `CONVERSION_NOTES.md` is that one time bin equals one imaging frame, so elapsed time can be recovered directly from frame index.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: `n_tp` is taken from the same `[s:e]` trial slice used for neural data, and the generated time vector has exactly that length.

ii. ```python
s = trial_start_inds[i] - 1
e = teleport_inds[i] - 1
n_tp = e - s
trial_neural = events_cells[:, s:e].copy()
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. The agent’s trajectory states that all streams are aligned to imaging frames and that trial-start alignment should use the same per-trial slices as the neural data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment` time series.

ii. ```python
environment = bts['environment/data'][:]
...
env_vals = environment[s:e]
```

iii. Step 5 of `CONVERSION_NOTES.md` maps “NWB environment” to `environment_type` with `0=ENV1, 1=ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, negative values are ignored and the median remaining value is used as the per-trial environment label. If a trial has no non-negative values, the previous trial’s value is reused.

ii. ```python
env_per_trial = np.zeros(n_trials, dtype=np.int64)
for i in range(n_trials):
    env_vals = environment[s:e]
    valid_env = env_vals[env_vals >= 0]
    if len(valid_env) > 0:
        env_per_trial[i] = int(np.median(valid_env))
    elif i > 0:
        env_per_trial[i] = env_per_trial[i - 1]
```

iii. The notes justify ignoring the pre-scanning `-1` values and treating the environment as a per-trial binary context variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over extracted trials, not from the stored NWB `trial number` values.

ii. ```python
for i in range(n_trials):
    ...
    trial_number = np.float32(i)
```

iii. Step 5 of `CONVERSION_NOTES.md` explicitly says trial number is a 0-indexed per-session trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index `i` is cast to float and broadcast across all timepoints in the trial.

ii. ```python
trial_number = np.float32(i)
...
np.full((1, n_tp), trial_number, dtype=np.float32)
```

iii. No further justification was given beyond using the within-session extracted trial order.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps` and also from the `autoreward` time series. A trial is marked rewarded if either source indicates reward delivery.

ii. ```python
reward_timestamps = bts['Reward/timestamps'][:]
autoreward = bts['autoreward/data'][:]
...
rewarded = determine_reward_per_trial(
    reward_timestamps, behav_timestamps, trial_start_inds, teleport_inds
)
...
if np.any(autoreward[s:e] > 0):
    rewarded[i] = 1
```

iii. Step 5 says previous-trial outcome should reflect rewarded vs omitted trials, and the code broadens “rewarded” to include autorewarded laps as well.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First the code computes a per-trial `rewarded` vector. Then, for each trial, it sets the input to 0 on the first trial and otherwise copies `rewarded[i-1]` across the whole current trial.

ii. ```python
if i == 0:
    prev_outcome = np.float32(0)
else:
    prev_outcome = np.float32(rewarded[i - 1])
...
np.full((1, n_tp), prev_outcome, dtype=np.float32)
```

iii. This follows the decoder specification in the notes: previous-trial outcome is a binary per-trial contextual input.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the animal’s `position` and from reward-zone identity inferred from the `reward_zone` signal within each trial.

ii. ```python
position = bts['position/data'][:]
rz_signal = bts['reward_zone/data'][:]
...
zone_labels, zone_coords = determine_reward_zone_per_trial(
    position, rz_signal, trial_start_inds, teleport_inds
)
...
dist = compute_distance_to_reward_zone(trial_pos, zs, ze)
```

iii. In the trajectory the agent decided the raw `reward_zone` values were not directly the decoder target, so it would infer the active reward zone from where `reward_zone > 0` occurred relative to position.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint the script computes signed distance to the current trial’s reward-zone interval: negative before the zone, zero inside it, positive after it.

ii. ```python
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    dist = np.zeros_like(position)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    dist[before] = position[before] - zone_start
    dist[inside] = 0.0
    dist[after] = position[after] - zone_end
    return dist
```

iii. Step 5 of `CONVERSION_NOTES.md` describes the same signed-distance definition and explicitly says it should be discretized afterward.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is binned into seven categories using the decoder specification’s thresholds.

ii. ```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The thresholds come directly from the decoder task and are reiterated in Step 5 of the notes.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same `[s:e]` per-trial slices that define the neural matrices.

ii. ```python
trial_neural = events_cells[:, s:e].copy()
trial_pos = position[s:e]
...
dist = compute_distance_to_reward_zone(trial_pos, zs, ze)
```

iii. The alignment is implicit in the shared trial slicing; the agent’s trajectory repeatedly described all variables as aligned to trial-start-based frame windows.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavioral `position` time series.

ii. ```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
```

iii. Step 5 maps “NWB position” directly to the absolute-position decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is taken and then discretized; there is no additional interpolation or transformation.

ii. ```python
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. The notes say absolute position should be discretized into five equal bins across the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The code uses five 90 cm bins: `<90`, `90-180`, `180-270`, `270-360`, and `>=360` cm.

ii. ```python
bins[position < 90] = 0
bins[(position >= 90) & (position < 180)] = 1
bins[(position >= 180) & (position < 270)] = 2
bins[(position >= 270) & (position < 360)] = 3
bins[position >= 360] = 4
```

iii. This matches the track length and the binning plan documented in Step 5.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same per-trial `[s:e]` slice indices as the neural trial matrix.

ii. ```python
trial_neural = events_cells[:, s:e].copy()
trial_pos = position[s:e]
```

iii. The agent treated behavior as frame-aligned to the same trial windows used for neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick` time series.

ii. ```python
lick = bts['lick/data'][:]
...
trial_lick = lick_binary[s:e]
```

iii. Step 5 maps “NWB lick” to a binary lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code first applies a lick-sensor correction: if more than 35% of frames in a trial have `lick > 2`, the whole trial’s lick trace is zeroed. The corrected trace is then binarized as `lick > 0`.

ii. ```python
lick_corrected = lick.copy()
for i in range(n_trials):
    trial_lick = lick_corrected[s:e]
    n_frames = len(trial_lick)
    if n_frames > 0 and np.sum(trial_lick > 2) / n_frames > LICK_ERROR_THR:
        lick_corrected[s:e] = 0

lick_binary = (lick_corrected > 0).astype(np.int64)
```

iii. Step 1 and Step 5 of `CONVERSION_NOTES.md` explicitly cite the reference `correct_lick_sensor_error()` logic and say this correction should be used.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the binarized lick vector with the same trial indices used for neural activity.

ii. ```python
trial_neural = events_cells[:, s:e].copy()
trial_lick = lick_binary[s:e]
```

iii. The alignment is implicit in the shared `[s:e]` trial slicing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the behavioral `reward_zone` signal together with `position`.

ii. ```python
position = bts['position/data'][:]
rz_signal = bts['reward_zone/data'][:]
...
zone_labels, zone_coords = determine_reward_zone_per_trial(
    position, rz_signal, trial_start_inds, teleport_inds
)
```

iii. The trajectory explains that the raw `reward_zone` values were treated as a signal indicating where the active zone was, not as the final categorical location itself.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the script finds positions where `reward_zone > 0`, averages those positions, assigns the nearest of the three fixed zone centers, and falls back to the previous trial’s zone or zone A if there is no positive reward-zone signal.

ii. ```python
rz_active = trial_rz > 0
if np.any(rz_active):
    rz_positions = trial_pos[rz_active]
    rz_center = np.mean(rz_positions)
else:
    if len(zone_labels) > 0:
        zone_labels.append(zone_labels[-1])
        zone_coords.append(zone_coords[-1])
        continue
    else:
        zone_labels.append('A')
        zone_coords.append(REWARD_ZONES['A'])
        continue

dists = {k: abs(rz_center - c) for k, c in ZONE_CENTERS.items()}
zone = min(dists, key=dists.get)
```

iii. In the trajectory the agent justified this by inspecting where `reward_zone` was nonzero relative to position and concluding that zone identity could be recovered from the center of that support.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, with `autoreward` also able to force a rewarded label.

ii. ```python
reward_timestamps = bts['Reward/timestamps'][:]
autoreward = bts['autoreward/data'][:]
...
rewarded = determine_reward_per_trial(
    reward_timestamps, behav_timestamps, trial_start_inds, teleport_inds
)
...
if np.any(autoreward[s:e] > 0):
    rewarded[i] = 1
```

iii. The notes frame reward outcome as whether reward was delivered on the trial; the code operationalizes that using both explicit reward timestamps and the autoreward trace.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each trial is labeled rewarded if any reward timestamp falls between that trial’s start and end timestamps, or if any `autoreward` sample is positive in that trial. The resulting binary value is then broadcast across the trial.

ii. ```python
def determine_reward_per_trial(reward_timestamps, behav_timestamps, trial_starts, teleports):
    rewarded = np.zeros(n_trials, dtype=np.int64)
    for i in range(n_trials):
        t_start = behav_timestamps[s]
        t_end = behav_timestamps[min(e, len(behav_timestamps) - 1)]
        in_trial = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(in_trial):
            rewarded[i] = 1
...
reward_out = int(rewarded[i])
np.full((1, n_tp), reward_out, dtype=np.int64)
```

iii. The notes say reward outcome should be binary per trial. The trajectory indicates the agent wanted a direct rewarded-vs-omitted trial label rather than a frame-level reward-event signal.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases: mismatched counts of trial starts and teleports are truncated to the shorter length; trials with `e <= s` are skipped; missing reward-zone support falls back to the previous zone or zone A; all-negative environment values fall back to the previous trial; and suspected stuck lick trials are zeroed out.

ii. ```python
if len(trial_start_inds) != len(teleport_inds):
    min_len = min(len(trial_start_inds), len(teleport_inds))
    trial_start_inds = trial_start_inds[:min_len]
    teleport_inds = teleport_inds[:min_len]
...
if e <= s:
    continue
...
if np.any(rz_active):
    ...
else:
    if len(zone_labels) > 0:
        zone_labels.append(zone_labels[-1])
    else:
        zone_labels.append('A')
...
if len(valid_env) > 0:
    env_per_trial[i] = int(np.median(valid_env))
elif i > 0:
    env_per_trial[i] = env_per_trial[i - 1]
```

iii. Some of these choices are justified in Step 5 and Step 10 of `CONVERSION_NOTES.md` (lick correction, ignoring `-1` environment values). The truncation and fallback rules are visible in the code but not deeply justified in the notes.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive part is per-session dF/F computation and OASIS deconvolution, plus loading large NWB arrays. The script prints timing for `dF/F + deconv`, and the full run in the notes took about 984 s for 152 sessions.

ii. ```python
t_dff = time.time()
events, dff = compute_dff_and_deconvolve(...)
print(f"    dF/F + deconv: {time.time() - t_dff:.1f}s")
...
F_plane = ophys[f'Fluorescence/{plane}/data'][:]
Fneu_plane = ophys[f'Neuropil/{plane}/data'][:]
```

iii. `CONVERSION_NOTES.md` Step 9 says full conversion took 984 s, and the runtime logging in the code highlights dF/F + deconvolution as the main per-session cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several per-trial loops could be combined or partially vectorized: the dF/F trial loops, reward-zone inference, reward detection, environment summarization, lick correction, and the final trial-building loop. Plane loading is also purely iterative.

ii. ```python
for start, stop in zip(start_inds, stop_inds):
    ...
for i in range(n_trials):
    ...  # reward zones
for i in range(n_trials):
    ...  # reward outcome
for i in range(n_trials):
    ...  # environment
for i in range(n_trials):
    ...  # lick correction
for i in range(n_trials):
    ...  # build neural/input/output trials
```

iii. The AI did not explicitly discuss vectorization in its notes, but the implementation makes the repeated trial-level loops obvious.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly walks the same trial boundaries for different purposes: dF/F extraction, reward-zone inference, reward labeling, environment summarization, lick correction, and final tensor construction all loop over the same trial list separately.

ii. ```python
for start, stop in zip(start_inds, stop_inds):
    ...
for i in range(n_trials):
    ...  # determine zones
for i in range(n_trials):
    ...  # determine rewards
for i in range(n_trials):
    ...  # env
for i in range(n_trials):
    ...  # lick correction
for i in range(n_trials):
    ...  # build outputs
```

iii. This was not called out explicitly in the notes, but it follows directly from the control flow in `process_session()` and `compute_dff_and_deconvolve()`.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and retains full-session dF/F traces even though downstream decoding uses only deconvolved events. It also loads some raw arrays that are not used for the final dataset, such as `trial_num` and `reward_data`, and keeps optional plotting machinery that is irrelevant to the saved pickle.

ii. ```python
reward_data = bts['Reward/data'][:]
trial_num = bts['trial number/data'][:]
...
events, dff = compute_dff_and_deconvolve(...)
...
if show_processing and session_idx < 2:
    plot_processing(..., dff[cell_indices, :], ...)
```

iii. There is no explicit justification in the notes beyond using plots for sanity checks. The retained `dff` output mainly supports optional visualization, not the final decoder dataset.
