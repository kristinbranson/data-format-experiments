# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent lists `sub-*` directories and all `.nwb` files in each, opens every candidate file once with `h5py`, and reads trials, events, units, and tracking arrays directly from HDF5 paths. Files can subsequently be skipped by session/unit criteria.

ii. `subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])` and `with h5py.File(nwb_path, 'r') as f:`.

iii. The trajectory says it inspected the NWB layout, counted sessions/subjects, and chose direct NWB/HDF5 loading for the full conversion.

## 1-b. How are the data split into subjects?

i. Subject folders (`sub-*`) define mice. Only subjects with a retained session are appended to `subject_names`; each retained session receives that list index.

ii. `subject_names.append(sub)` and `sub_idx = subject_names.index(sub)`.

iii. The folder hierarchy was treated as the authoritative subject grouping.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session and produces one nested entry in `neural`, `input`, and `output`; files failing session or unit criteria return `None`.

ii. `for nf in nwb_files: ... result = process_session(nwb_path)`.

iii. The agent inferred one NWB file per recording session and deliberately added paper-derived performance/count criteria.

## 1-d. How are the data split into trials?

i. Trial rows are indexed by `intervals/trials/id`; the same row index addresses trial columns and the corresponding go cue. Retained indices become separate trial arrays.

ii. `n_trials = len(f['intervals/trials/id'])` and `for trial_idx in trial_indices:`.

iii. The agent regarded the NWB trials table and one go cue per row as the natural trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Sessions must exceed 65% hit rate among non-water, non-early, non-stim, responding control trials and contain at least 50 correct trials per side. Within retained sessions, auto-water and free-water trials are removed; early, ignore, and stim trials are kept. `obs_intervals` is not used.

ii. `if performance <= 0.65 or n_correct_left < 50 or n_correct_right < 50: return None` and `trial_mask = (auto_water == 0) & (free_water == 0)`.

iii. The trajectory explicitly balanced paper curation against required decoder classes: it kept early/no-response/stim trials, but believed session performance filtering should still be applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It uses `units/spike_times`, `units/spike_times_index`, go-cue timestamps, and the good-unit indices from `units/classification`.

ii. `spike_times_flat = f['units/spike_times'][:]` and `go_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]`.

iii. The agent identified sorted spike timestamps as the raw neural representation.

## 2-b. How is the `neural` data processed?

i. For every retained trial and unit, absolute spikes are shifted by the go cue, histogrammed into non-overlapping bins, and divided by 0.05 s to obtain Hz. No smoothing or normalization is applied.

ii. `aligned = spks - go_t` and `fr_matrix[u, :] = bin_spike_times(aligned, T_START, T_END, BIN_WIDTH)`.

iii. The trajectory planned firing rates matching the requested 50-ms decoder representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `good` are retained. Sessions with a non-object classifier column or no good units are dropped; `unit_quality`, `is_good_trials`, and `obs_intervals` are not used.

ii. `good_mask = classification == 'good'` and `if len(good_indices) == 0: return None`.

iii. The agent concluded classifier QC matches the white paper and intentionally refused to fall back to the older `unit_quality` label for the one unclassified session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Session-absolute spike times are shifted by that trial's go-cue timestamp, then restricted to -2.5 through +1.5 s by histogram edges.

ii. `aligned = spks - go_t` with `T_START = -2.5` and `T_END = 1.5`.

iii. The agent noted spikes and behavioral events share the NWB clock, so subtraction is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The result has 80 non-overlapping 50-ms bins over four seconds. Raw spike events are temporally binned; there is no additional resampling or smoothing.

ii. `BIN_WIDTH = 0.05` and `n_bins = int(round((t_end - t_start) / bin_width))`.

iii. The bin size and window were taken directly from the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times` and each trial's go cue; the last sample start strictly before that go cue is selected.

ii. `before = sample_start_ts[sample_start_ts < go_times[i]]` and `tone_onset_rel_go[i] = before[-1] - go_times[i]`.

iii. The trajectory recognized extra sample events from early-lick replays and chose the last pre-go sample as the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The selected onset is expressed relative to go, then subtracted from each bin center. If none is found, the code imputes -1.85 s.

ii. `tone_rel = -1.85` (fallback) and `time_from_tone = bin_centers - tone_rel`.

iii. The agent observed tone onset was consistently about 1.85 s before go and used that as a defensive fallback.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use identical go-relative bin centers; each value is elapsed time from tone at the neural bin center.

ii. `bin_centers = get_bin_centers(...)` and `time_from_tone = bin_centers - tone_rel`.

iii. The shared go-cue clock was the stated basis for alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus go-cue timestamps.

ii. `onset_from_trial_start = float(photostim_onset_str[i])` and `go_from_trial_start = go_times[i] - trial_start_times[i]`.

iii. The agent chose actual stored timing rather than inferring stimulation from protocol terminology.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Onset and offset are converted to go-relative seconds. A bin is one when its center is at or after onset and before offset, otherwise zero; `N/A` trials remain all zero.

ii. `photostim_binary = ((bin_centers >= on_t) & (bin_centers < off_t)).astype(float)`.

iii. The agent wanted the requested time-varying on/off signal rather than a trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-start-relative stimulation times are shifted onto the go-relative axis and compared with the same centers used by neural bins.

ii. `photostim_on_rel[i] = onset_from_trial_start - go_from_trial_start`.

iii. The trajectory explicitly worked through trial-start versus go-cue timing.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from `outcome` and `trial_instruction`: hit means instructed side, miss means opposite side, and ignore means no lick.

ii. `if out == 'ignore': choice = 2` and `elif out == 'hit': choice = 0 if instr == 'left' else 1`.

iii. The agent reasoned these two columns fully determine actual response direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. It codes left/right/no lick as 0/1/2 and repeats the scalar across all 80 bins.

ii. `output_data[0, :] = choice`.

iii. Repetition allows scalar and time-varying outputs to share one rectangular array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is read directly from `intervals/trials/outcome`.

ii. `outcome = np.array([x.decode() for x in f['intervals/trials/outcome'][:]])`.

iii. The stored categories already match the requested output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore/miss/hit codes 0/1/2 and are repeated across bins.

ii. `outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]` and `output_data[1, :] = outcome_val`.

iii. The mapping follows the requested category order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii. `early_lick = np.array([x.decode() for x in f['intervals/trials/early_lick'][:]])`.

iii. The trials table provides the required flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and every other value maps to 1; the value is repeated over bins.

ii. `early_val = 0 if early_lick[trial_idx] == 'no early' else 1`.

iii. The agent used the requested binary no/yes coding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses side-camera tongue-tracking data column 1 (y), column 2 (likelihood), and matching timestamps.

ii. `tongue_y = tongue_data[:, 1]` and `tongue_likelihood = tongue_data[:, 2]`.

iii. The agent identified these fields after inspecting the behavioral time series.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood over 0.9 are treated as visible. The code computes session percentiles from all visible raw frames, then chooses one nearest frame for every 50-ms bin center and classifies it; it does not average frames within bins.

ii. `visible_mask = tongue_likelihood > TONGUE_LIKELIHOOD_THRESH` and `frame_idx = np.searchsorted(tongue_ts, abs_t)`.

iii. The trajectory proposed nearest-frame sampling for alignment and selected 0.9 as a high-confidence DLC threshold, while acknowledging the nested loop was slow.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of visible raw-frame y values are thresholds: below 40th is 0, 40th through 60th inclusive is 1, above 60th is 2, and low-likelihood samples are 3.

ii. `pct40 = np.percentile(y_visible, 40)` and `elif y_val <= pct60: tongue_y_disc[b] = 1`.

iii. The per-session 40/60 split and explicit not-visible class came from the instructions; the trajectory chose visible raw frames as the percentile population.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each go-relative neural bin center it finds the first camera timestamp at or after the absolute center, optionally switches to the previous frame only if that frame is closer, and uses that one frame.

ii. `abs_t = go_t + tc` and `frame_idx = np.searchsorted(tongue_ts, abs_t)`.

iii. The agent relied on the shared session clock and nearest-frame matching.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. An unclassified session or one with no good units is dropped; fewer than two retained trials also drops a session. Missing tone onset is imputed as -1.85 s. No visible tongue yields thresholds of zero and category 3 at sampled bins. The code does not handle empty tongue timestamp arrays or trials outside `obs_intervals`.

ii. `else: return None`, `tone_rel = -1.85`, and `pct40 = pct60 = 0.0`.

iii. The trajectory explicitly investigated the single unclassified session and chose exclusion rather than weaker QC; other fallbacks were pragmatic defensive choices.

## 10-a. What are the most time-consuming steps of the code?

i. Neural construction loops over every retained trial and every unit and calls `np.histogram`; tongue construction additionally loops over all trials and 80 bins with timestamp searches. File I/O and the large final pickle are also costly.

ii. `for trial_idx in trial_indices:` followed by `for u, spks in enumerate(unit_spikes):` and `for b, tc in enumerate(bin_centers):`.

iii. During execution the agent observed the tongue nested loop was especially slow and considered optimizing it.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning could vectorize all trials per unit using flattened edges and `searchsorted`. Photostim and tone calculations could vectorize over trials. Nearest camera-frame lookup could vectorize all trial/bin centers, though the reference's within-bin aggregation still needs grouped reduction.

ii. The principal avoidable loop nest is `for trial_idx ... for u ... bin_spike_times(...)`; tongue uses `for trial_idx ... for b ... np.searchsorted(...)`.

iii. The agent intended NumPy-heavy processing but later recognized its tongue implementation remained a performance bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. It recreates identical histogram edges inside `bin_spike_times` for every unit-trial pair, subtracts each go cue from an entire unit's spike train for every trial, and performs individual camera searches for every bin.

ii. `edges = np.linspace(t_start, t_end, n_bins + 1)` occurs inside the repeatedly called function.

iii. No explicit justification was given; these repetitions follow from the straightforward nested-loop implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `photostim_power` indirectly not at all, but does compute unused locals `n_units_total` and `n_trials`; more materially, it aligns/subtracts the full spike train separately for each trial even though only window counts survive. Session-selection statistics are computed solely to discard sessions and are not saved except as descriptive metadata.

ii. `n_units_total = len(classification)` is never subsequently used; `aligned = spks - go_t` creates a full temporary whose out-of-window values are discarded by the histogram.

iii. The trajectory did not identify these as deliberate; they are implementation overhead rather than scientific decisions.
