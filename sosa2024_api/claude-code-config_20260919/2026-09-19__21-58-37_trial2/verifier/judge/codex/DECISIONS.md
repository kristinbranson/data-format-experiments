# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively identifies every NWB file one directory below `/app/data`, sorts them by subject directory and parsed session number, and loads each session with `pynwb.NWBHDF5IO`. Full mode uses all 152 files; a multiprocessing pool converts sessions.

ii. `files = sorted(glob.glob(os.path.join(args.data_dir, 'sub-*', '*.nwb')))`; `io = NWBHDF5IO(path, 'r', load_namespaces=True); nwb = io.read()`

iii. The agent justified this as complete coverage of the deposited archive and used pynwb as required. It reported 152 sessions from 11 mice.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file, and a first-seen ordered unique subject list plus per-session index is assembled.

ii. `subject = nwb.subject.subject_id`; `if sub not in subjects: subjects.append(sub); subject_idx.append(subjects.index(sub))`

iii. The NWB subject metadata is authoritative and the resulting 11 mice match the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Sessions are sorted by subject and `ses-XX`; `session_id` combines subject and experimental day.

ii. `files.sort(key=lambda p: (p.split('/')[-2], int(p.split('ses-')[1][:2])))`; `session_id=f"{sess['subject']}_day{sess['exp_day']:02d}"`

iii. File/session metadata and the reference repository's per-day organization support this mapping.

## 1-d. How are the data split into trials?

i. Trial starts are every positive `trial_start` sample and ends are every positive `teleport` sample. Each trial is sliced `[start, teleport)`, excluding the teleport frame.

ii. `tstart_inds = np.where(beh['trial_start'] > 0)[0]`; `teleport_inds = np.where(beh['teleport'] > 0)[0]`; `s, e = int(tstart_inds[i]), int(teleport_inds[i])`

iii. The agent cited reference behavior code and argued that the teleport frame has meaningless interpolated position. It asserted paired starts/ends and ordering.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped when more than 30% of frames have lick count greater than 2, or when neural, position, speed, or lick values are non-finite. No minimum-length filter is applied.

ii. `lick_error[i] = np.mean(L > 2) > LICK_ERROR_THRESH`; `if lick_error[i]: continue`; `if (not np.all(np.isfinite(neural)) ...): continue`

iii. The 30% rule reproduces the paper's 81 sensor-error trials. Dropping rather than NaN-marking is justified because lick is a required categorical decoder output and NaNs are invalid.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It is derived from raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu), restricted by `iscell`; it does not use the stored NWB `Deconvolved` series.

ii. `rrs = ophys['Fluorescence'][key]`; `ophys['Neuropil'][key].data[:nframes, :]`; `iscell = np.asarray(seg['iscell'].data)[:, 0] > 0`

iii. The notes correctly distinguish the NWB's suite2p deconvolution from the paper pipeline's events computed from per-trial dF/F.

## 2-b. How is the `neural` data processed?

i. Fneu is subtracted with coefficient 0.7; per-window maximin baselines use Gaussian sigma 15 and 300-sample minimum/maximum filters; dF/F is formed and Gaussian-smoothed with sigma 2. OASIS (`tau=0.7`) is optional, but the delivered full conversion used the default dF/F rather than events.

ii. `f_ -= NEU_COEF * fneu_`; `flow[:, s:e] = ndimage.maximum_filter1d(x, MAXIMIN_WIN, axis=-1)`; `dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])`; `neural_full = (events if neural_signal == 'events' else dff)[keep_cells]`

iii. The agent argued dF/F was closer to raw data and decoded better, while acknowledging that the paper's decoder and human reference use OASIS events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only suite2p-curated `iscell` ROIs are loaded, then cells with Pearson correlation `r(dF/F,speed)>0.5` are excluded as putative interneurons.

ii. `mask = iscell[plane_idx == p]`; `is_int = np.nan_to_num(r_speed, nan=0.0) > INT_R_THRESH`; `keep_cells = ~is_int`

iii. Both filters are taken from the Methods/reference pipeline; reported exclusion fractions were close to the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural and behavior already share imaging-frame indices. Neural is sliced from the `trial_start` frame through the frame before teleport, so time zero is trial start.

ii. `neural = neural_full[:, s:e]`; `t = beh['time'][s:e] - beh['time'][s]`

iii. The agent relied on the archive's prior VR-to-imaging alignment and chose unshifted slices to keep all streams sample-for-sample aligned, differing from a one-sample internal reference dF/F convention.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one bin: 64.484 ms (15.5078125 Hz). No rebinning or resampling is applied.

ii. `'time_bin_size': 1000.0 / 15.5078125`; `fs = sess['rate'] / sess['n_planes']`

iii. All sessions share the effective per-plane rate; preserving frames avoids blurring lick and speed signals.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses timestamps of the raw `position` behavioral series.

ii. `frame_times = np.asarray(beh['position'].timestamps[:], dtype=np.float64)`

iii. Position timestamps are the common imaging-aligned behavioral time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial-start index is subtracted from every timestamp in that trial.

ii. `t = beh['time'][s:e] - beh['time'][s]`

iii. This directly implements time since the requested alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The identical `[s:e]` frame slice is used for timestamps and neural data.

ii. `t = beh['time'][s:e] - beh['time'][s]`; `neural = neural_full[:, s:e]`

iii. NWB behavior was already interpolated onto imaging frames, so no further alignment is needed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from behavioral `environment` (`morph`).

ii. `morph=get('environment')`; `m = np.unique(beh['morph'][s:e])`

iii. The archived coding is 0=ENV1 and 1=ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The first unique within-trial value is rounded to an integer and broadcast over the trial.

ii. `morph[i] = int(np.round(m[0]))`; `np.full(len(pos), float(morph[i]))`

iii. The variable should be constant within a trial; the notes report that this was checked.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It comes from the behavioral `trial number` series.

ii. `trialnum=get('trial number')`; `tn = np.unique(beh['trialnum'][s:e])`

iii. The archive directly contains the 0-indexed within-session trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The first unique within-trial value is converted to integer and broadcast across frames.

ii. `trialnum[i] = int(tn[0])`; `np.full(len(pos), float(trialnum[i]))`

iii. Constancy within trials was sanity-checked.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It uses sparse `Reward` timestamps mapped to frames plus the per-frame `reward_zone` flag.

ii. `reward_frames = np.searchsorted(frame_times, reward_times)`; `isreward[i] = int(np.any(beh['reward'][s:e] > 0) and np.any(beh['rzone'][s:e] > 0))`

iii. This reproduces reference `get_trial_types`, which requires both delivery and zone activation.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current-trial binary outcomes are shifted by one trial; the first trial gets 0, and values are broadcast in time. The previous raw trial is used even if that trial is later filtered.

ii. `prev_outcome = np.concatenate([[0], isreward[:-1]])`; `np.full(len(pos), float(prev_outcome[i]))`

iii. Zero represents no preceding imaged/rewarded trial and avoids discarding first trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It derives from behavioral position and active reward-zone coordinates inferred from the NWB identifier's scene name and trial index.

ii. `scene = nwb.identifier.rstrip('/').split('/')[-1]`; `rz_coords, rz_labels = scene_reward_zones(sess['scene'], n_trials_raw)`; `pos = beh['pos'][s:e]`

iii. Scene identity labels omission trials where the reward-zone flag may be absent and matches reference `get_reward_zones`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is zero inside the active interval, position minus zone start before it, and position minus zone end after it.

ii. `d = np.zeros_like(pos)`; `d[pos < rz_start] = pos[pos < rz_start] - rz_start`; `d[pos > rz_end] = pos[pos > rz_end] - rz_end`

iii. This is distance to the nearest point of the zone, with sign encoding before/after.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks create the seven instructed classes at -50, -10, 0, 10, and 50 cm, with exact zero class 3.

ii. `out[:] = 3`; `out[(d >= -10) & (d < 0)] = 2`; `out[(d > 0) & (d <= 10)] = 4`; `out[d > 50] = 6`

iii. The masks directly encode the requested inequalities and avoid ambiguity around zero.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural use the same frame slice and therefore identical trial lengths.

ii. `pos = beh['pos'][s:e]`; `neural = neural_full[:, s:e]`

iii. The archive's behavioral streams are already on the imaging-frame time base.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavioral `position`.

ii. `pos=get('position')`; `pos = beh['pos'][s:e]`

iii. The raw variable is corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is divided by 90 cm, cast to integer, and clipped to classes 0–4.

ii. `return np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)`

iii. The 450 cm track divided into five equal bins gives 90 cm per bin; clipping absorbs slight out-of-range samples.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Classes correspond to <90, 90–<180, 180–<270, 270–<360, and >=360 cm after clipping.

ii. `TRACK_LENGTH = 450.0`; `np.clip((pos / (TRACK_LENGTH / 5)).astype(np.int64), 0, 4)`

iii. This matches the required five equal track bins for ordinary nonnegative positions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Both are sliced with the same `[s:e]` indices.

ii. `pos = beh['pos'][s:e]`; `neural = neural_full[:, s:e]`

iii. No resampling is required because behavior is imaging-aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavioral `lick` count per imaging frame.

ii. `lick=get('lick')`; `lick = beh['lick'][s:e]`

iii. The reference alignment pipeline stores per-frame lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After sensor-error trial filtering, any positive count becomes 1 and zero remains 0.

ii. `(lick > 0).astype(np.int64)`

iii. This implements the binary no/yes decoder output and matches reference binarization.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays use the same trial frame indices.

ii. `lick = beh['lick'][s:e]`; `neural = neural_full[:, s:e]`

iii. Both streams share imaging-frame timing in the NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from the scene name in `nwb.identifier` and the raw trial index, not from the intermittent `reward_zone` flag.

ii. `scene = nwb.identifier.rstrip('/').split('/')[-1]`; `rz_coords, rz_labels = scene_reward_zones(sess['scene'], n_trials_raw)`

iii. Scene naming provides zone identity even on omission trials and follows the reference function.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene suffixes map A/B/C to coordinates 80–130/200–250/320–370 cm. Switch scenes use the first zone for trials 0–29 and second thereafter; labels map to 0/1/2 and are broadcast.

ii. `labels = np.array([first] * min(CHANGE_TRIAL, n_trials) + [second] * max(0, n_trials - CHANGE_TRIAL))`; `np.full(len(pos), RZ_LABELS.index(rz_labels[i]))`

iii. The switch occurs after 30 trials as stated in the paper and was validated against zone flags.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It derives from sparse `Reward` timestamps and behavioral `reward_zone` flags.

ii. `reward_times = np.asarray(beh['Reward'].timestamps[:])`; `isreward[i] = int(np.any(beh['reward'][s:e] > 0) and np.any(beh['rzone'][s:e] > 0))`

iii. The conjunction is the reference definition of a rewarded trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are assigned by `searchsorted` (then clipped), a trial is positive when it contains both a reward event and zone flag, and the binary result is broadcast over its frames.

ii. `reward_frames = np.clip(np.searchsorted(frame_times, reward_times), 0, nframes - 1)`; `np.full(len(pos), isreward[i], dtype=np.int64)`

iii. This creates the required per-trial omitted/rewarded label; the notes report no reward-without-zone cases.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Imaging arrays are truncated to behavior length (covering the known one-extra-frame case); reward indices are clipped; malformed start/end structure triggers assertions; any converted trial with non-finite neural or behavioral values is dropped. No interpolation or imputation is performed.

ii. `rrs.data[:nframes, :]`; `reward_frames = np.clip(...)`; `assert len(tstart_inds) == len(teleport_inds)`; `if (not np.all(np.isfinite(neural)) ...): continue`

iii. The agent tied truncation to the reference one-frame correction and preferred dropping unusable trials to producing invalid categorical arrays; the full run reported zero NaN drops.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large F/Fneu arrays, per-window baseline filtering/dF/F, optional OASIS, and writing the 2.6 GB pickle dominate. Sessions are processed in parallel.

ii. `rrs.data[:nframes, :]`; `compute_dff_events(...)`; `ProcessPoolExecutor(max_workers=nworkers, mp_context=ctx)`; `pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)`

iii. The notes measured a 48 s parallel conversion plus 9 s pickle write and identified contiguous reads and skipping unused OASIS as speedups.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial loops for reward/morph/trial-number/lick-error summaries and assembly, baseline-window loops, and per-plane reads remain. Variable trial windows make full vectorization awkward. Cell-speed correlations were already vectorized.

ii. `for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):`; `for s, e in windows:`; `r = (Dm @ Sm) / denom`

iii. The agent explicitly replaced a per-cell correlation loop with a matrix product (about 100x faster); it retained natural loops over variable-length trials/windows.

## 13-c. What processing does the code repeat multiple times?

i. It makes multiple passes over trial windows: filling F/Fneu, estimating baselines, smoothing/deconvolving, computing trial summaries, and assembling output arrays. Diagnostic mode again derives plotted distance and concatenated behavior.

ii. `for s, e in windows:` appears in multiple stages of `compute_dff_events`; separate `for i ...` loops summarize and then build trials.

iii. The passes reflect distinct processing phases and limit memory complexity, though some per-trial summaries could be combined with assembly.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. For the delivered default dF/F run, OASIS is correctly skipped. Plane IDs are carried into metadata while all neurons receive CA1 region index; `autoreward`, date, some timing/statistical fields, and raw zone flags are loaded or computed mainly for metadata/QC. Plot mode computes events and extensive diagnostics that are not stored.

ii. `autoreward=get('autoreward')`; `events if neural_signal == 'events' or show_processing`; `info['plane_idx'] = planes.tolist()` while `brain_region_idx.append(np.zeros(...))`

iii. The agent viewed diagnostics and provenance as validation aids; they are not needed by decoder training, but default full conversion avoids the largest optional waste (OASIS and plots).
