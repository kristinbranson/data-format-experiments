# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `/app/data/sub-*/*.nwb` file, sorts the paths, opens each file once with `h5py`, and reads subject, trial, event, unit, spike, and camera datasets directly from the NWB HDF5 hierarchy. Exceptions are caught per file, so a failed file is logged and skipped.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
for i, nwb_file in enumerate(nwb_files):
    try:
        result = process_session(nwb_file, show_processing=args.show_processing)
        if result is not None:
            all_sessions.append(result)
    except Exception as e:
        print(f"  ERROR processing {nwb_file}: {e}")
```
```python
f = h5py.File(nwb_path, 'r')
```

iii. The notes say the archive contains 28 subject directories and 174 session NWB files and that NWB exposes units, trials, events, and behavioral time series. Sorting gives deterministic order; direct `h5py` access was chosen instead of `pynwb`.

## 1-b. How are the data split into subjects?

i. Each session's `general/subject/subject_id` is read. At assembly, unique IDs are sorted, and every session receives the corresponding integer `subject_idx`.

ii.
```python
subject_id = f['general']['subject']['subject_id'][()].decode()
all_subject_ids = sorted(set(s['subject_id'] for s in all_sessions))
subject_to_idx = {sid: i for i, sid in enumerate(all_subject_ids)}
subject_idx_list.append(subject_to_idx[sess['subject_id']])
```

iii. The notes identify `general/subject/subject_id` as the canonical source for `subjects/subject_idx`; the full result contained 28 subjects.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Each successful `process_session` result becomes one entry of the session-level `neural`, `input`, and `output` lists; the filename is retained as `session_name` metadata.

ii.
```python
return {'session_name': basename, 'neural': neural_data,
        'input': input_data, 'output': output_data, ...}
...
for sess in all_sessions:
    neural_list.append(sess['neural'])
```

iii. The agent documented the archive layout as one session NWB per dated filename. It skips the single file with no good units, yielding 173 sessions.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials. The same retained row indices select go cues and trial variables, and per-trial matrices are appended to lists. The code asserts one go cue per trials-table row.

ii.
```python
n_trials_total = f['intervals']['trials']['id'].shape[0]
trial_indices = np.where(trial_mask)[0]
go_start_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
assert len(go_start_times) == n_trials_total
go_times = go_start_times[trial_indices]
```

iii. The notes describe the NWB trials table and go events as the natural trial definitions and report sanity checks on dimensions.

## 1-e. How are trials filtered based on quality controls?

i. The agent removes trials marked `auto_water` or `free_water`, keeps early-lick, ignore, and photostimulation trials because they are requested decoder variables, and drops a session if fewer than two trials remain. It does not filter against `units/obs_intervals`, so many behavior-only trials with no recorded spikes remain.

ii.
```python
is_auto_or_free = (auto_water == 1) | (free_water == 1)
trial_mask = ~is_auto_or_free
trial_indices = np.where(trial_mask)[0]
if n_trials < 2:
    return None
```

iii. The notes justify keeping early-lick, ignore, and stimulated trials because otherwise the requested targets/input would be removed, and call auto/free-water trials nongenuine. However, verification printed extensive all-zero-neural warnings; the agent did not identify the missing `obs_intervals` filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from ragged `units/spike_times` and `units/spike_times_index`, restricted by `units/classification`, and are placed relative to `BehavioralEvents/go_start_times`.

ii.
```python
classification = f['units']['classification'][:]
good_indices = np.where(classification == b'good')[0]
spike_times_data = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
neural_data = compute_firing_rates_fast(unit_spike_times, go_times, BIN_EDGES)
```

iii. The notes identify spike times as the neural source, go cue as time zero, and `classification == 'good'` as the published classifier QC.

## 2-b. How is the `neural` data processed?

i. Ragged spike arrays are sliced once per good unit. For every unit and trial, spikes in the four-second window are histogrammed into nonoverlapping bins; counts are divided by 0.05 s to yield Hz. No smoothing, normalization, or baseline subtraction is used.

ii.
```python
for ni in range(n_units):
    st = unit_spike_times_list[ni]
    for ti in range(n_trials):
        counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
        all_rates[ni, ti] = counts
all_rates /= bin_width
```

iii. The agent states that 50 ms nonoverlapping firing-rate bins are mandated by the decoder task, superseding the paper code's sliding-bin settings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose byte-valued `classification` equals `b'good'` are retained; a session with zero such units is skipped. No individual metric thresholds are added.

ii.
```python
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
if n_good == 0:
    return None
```

iii. The notes say this field contains the already-applied region-specific classifier decision from the QC pipeline. The result retained 69,453 good units and skipped the unclassified session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's absolute go-cue timestamp is added to the common relative bin edges from -2.5 to +1.5 s. Absolute spike timestamps are histogrammed against those shifted edges.

ii.
```python
go_t = go_times[ti]
t_lo = go_t + bin_edges[0]
t_hi = go_t + bin_edges[-1]
counts, _ = np.histogram(st[i_lo:i_hi], bins=bin_edges + go_t)
```

iii. The notes state NWB timestamps share the session clock and the requested alignment is go-cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50 ms bins over [-2.5, +1.5) s. Raw spike times are binned directly; there is no subsequent rebinning.

ii.
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

iii. This exactly follows the decoder instructions; the agent explicitly rejected the reference analysis's other sliding-window parameters for this task.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final code it is not derived from a raw per-trial tone event. It uses only common go-relative bin centers and the protocol constant `TONE_ONSET_REL_GO = -1.85`.

ii.
```python
TONE_ONSET_REL_GO = -1.85
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO
```

iii. The notes say an earlier event lookup appeared wrong on 12.5% of trials because early licks replay sample events, so the agent replaced it with fixed task timing. This overlooks that the last sample onset before go identifies the replayed tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The constant -1.85 s is subtracted from every bin center, producing the same vector for every trial; it is cast to `float32` when stacked.

ii.
```python
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO
trial_input = np.stack([time_from_tone.astype(np.float32), photostim_ts], axis=0)
```

iii. The stated rationale is that fixed sample (0.65 s) plus delay (1.2 s) timing is more reliable than event matching. That is only true for trials without epoch replay.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the same 80 go-relative bin centers as neural activity, but use a fixed tone-to-go offset, so the array shape/bin indexing aligns while replayed trials' semantic tone alignment does not.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
time_from_tone = BIN_CENTERS - TONE_ONSET_REL_GO
```

iii. The agent viewed go cue as the shared zero and protocol timing as sufficient for all trials.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trials-table `photostim_onset`, `photostim_duration`, and `start_time`, plus each trial's go-cue timestamp. `N/A` onset denotes no stimulation.

ii.
```python
photostim_onset = f['intervals']['trials']['photostim_onset'][:]
photostim_dur = f['intervals']['trials']['photostim_duration'][:]
onset_abs = float(photostim_onset[i]) + trial_start_times[i]
onset_rel_go = onset_abs - go_times[i]
```

iii. The notes say the tabular timestamps should be trusted over a textual generalization about stimulation timing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Trial-relative onset is converted to absolute time, then to go-relative onset; duration supplies offset. A float vector is set to one where bin centers lie in the half-open interval `[onset, offset)`, otherwise zero.

ii.
```python
offset_rel_go = onset_rel_go + dur
photostim_ts = np.zeros(N_BINS, dtype=np.float32)
photostim_ts[(BIN_CENTERS >= onset_rel) & (BIN_CENTERS < offset_rel)] = 1.0
```

iii. This directly represents the requested on/off state at every time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation bounds are expressed relative to the same go cue as the neural bin edges and sampled at the identical bin centers.

ii.
```python
onset_rel_go = onset_abs - go_times[i]
photostim_ts[(BIN_CENTERS >= onset_rel) & (BIN_CENTERS < offset_rel)] = 1.0
```

iii. The notes identify go cue as the common alignment event and the interval calculation preserves each trial's recorded timing.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trials-table `outcome` and `trial_instruction`: hit means instructed side, miss means opposite side, and ignore means no lick.

ii.
```python
if out == 'hit':
    choice = 0 if inst == 'left' else 1
elif out == 'miss':
    choice = 1 if inst == 'left' else 0
else:
    choice = 2
```

iii. The notes explain that no explicit choice field exists and these two columns fully determine left/right/no-lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded `0=left`, `1=right`, `2=no_lick`, then repeated over all 80 bins in output row 0.

ii.
```python
np.full(N_BINS, choice, dtype=np.int64)
```

iii. The coding follows `output_values`; repetition permits all outputs to share a `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome` (`ignore`, `miss`, or `hit`).

ii.
```python
outcome = np.array([x.decode() for x in f['intervals']['trials']['outcome'][:]])
```

iii. The notes state that the raw categories exactly match the requested target.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to `0=ignore`, `1=miss`, `2=hit` and repeated across 80 bins in output row 1.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[out]
np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. This implements the declared category ordering and common time-shaped output representation.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick` (`no early` or `early`).

ii.
```python
early_lick = np.array([x.decode() for x in f['intervals']['trials']['early_lick'][:]])
```

iii. The agent keeps these trials specifically because early lick is a requested output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and everything else to 1, repeated across all bins in row 2.

ii.
```python
early_val = 0 if trial_early_lick[ti] == 'no early' else 1
np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The notes define the same binary coding and treat it as a per-trial output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y) and 2 (DLC likelihood) of `Camera0_side_TongueTracking`; column 0 is unused.

ii.
```python
tongue_ts = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
tongue_data_raw = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
tongue_y = tongue_data[:, 1]
tongue_lh = tongue_data[:, 2]
```

iii. The notes identify side-camera tongue y and likelihood as the available approximately 300 Hz measurement.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For retained trial windows, raw frame y values with likelihood at least 0.9 are pooled to compute session thresholds. Within each trial/bin, the code averages all y values only if the bin's *mean likelihood* is at least 0.9; otherwise it assigns not visible. Thus low-confidence y frames can enter a visible bin's mean.

ii.
```python
vis_mask = tongue_lh[i_lo:i_hi] >= likelihood_thresh
visible_y_all.append(tongue_y[i_lo:i_hi][vis_mask])
p40 = np.percentile(all_visible, 40)
p60 = np.percentile(all_visible, 60)
...
avg_lh = trial_lh_v[b_mask].mean()
if avg_lh >= likelihood_thresh:
    avg_y = trial_y_v[b_mask].mean()
```

iii. The notes justify a 0.9 visibility threshold and per-session percentiles over all visible raw y observations. They do not justify using mean likelihood or including low-confidence y in bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-level 40th/60th percentiles of pooled visible raw frames define classes: 0 below p40, 1 from p40 through p60, 2 above p60, and 3 when a bin is absent or its mean likelihood is below 0.9.

ii.
```python
if avg_y < p40:
    trial_bins[b] = 0
elif avg_y <= p60:
    trial_bins[b] = 1
else:
    trial_bins[b] = 2
```

iii. The 40/60 split and per-session scope come from the instructions. The agent chose raw-frame rather than 50 ms-bin distributions and a 0.9 confidence cutoff.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera samples are selected from `[go-2.5, go+1.5)`, converted to go-relative time, and assigned to the same 80 relative bin edges used for spikes.

ii.
```python
i_lo = np.searchsorted(tongue_ts, go_t + bin_edges[0])
i_hi = np.searchsorted(tongue_ts, go_t + bin_edges[-1])
rel_ts = trial_ts - go_t
bin_idx = np.searchsorted(bin_edges, rel_ts, side='right') - 1
```

iii. The agent assumes camera, event, and spike timestamps share the NWB session clock, so no interpolation or offset correction is needed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A no-good-unit session is skipped; sessions with fewer than two retained trials are skipped; tongue bins with absent/insufficient-confidence tracking become class 3; and per-session exceptions are printed and skipped. However, behavior trials outside recorded `obs_intervals` are not recognized as missing neural data and become all-zero neural trials.

ii.
```python
if n_good == 0: return None
if n_trials < 2: return None
trial_bins = np.full(n_bins, 3, dtype=np.int64)
...
except Exception as e:
    print(f"  ERROR processing {nwb_file}: {e}")
```

iii. The notes discuss the unclassified session and not-visible tongue state, but the verification output's numerous all-zero trials was not corrected. Broad exception skipping also risks silently losing an otherwise valid session.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant conversion work is neural binning because it runs a Python nested loop over every good unit and retained trial and calls `searchsorted`/`histogram` for each pair. Reading large spike/camera arrays and writing the very large pickle are also costly. Optional plotting adds work only with `--show-processing`.

ii.
```python
for ni in range(n_units):
    for ti in range(n_trials):
        counts, _ = np.histogram(...)
```

iii. The code calls this routine “fast” and says it vectorizes across trials, but the implementation still loops over trials. Full logs show multi-second session times; the notes emphasize optimization and validation rather than a measured stage-level profile.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural binning's inner trial loop can be eliminated by searching all trial edges at once per unit, as in the reference. Photostimulation interval construction/input assembly, output construction, spike-start extraction, and tongue binning also contain Python loops; several can be array operations or `bincount`-based grouping.

ii.
```python
for ni in range(n_units):
    for ti in range(n_trials): ...
for i in range(n_trials): ...       # photostim intervals
for ti in range(n_trials): ...      # inputs/outputs/tongue passes
for b in range(n_bins): ...         # tongue bins
```

iii. The agent claimed the neural routine was “vectorized across trials,” which is inaccurate. It did use `bincount` ideas for tongue data but retained an avoidable 80-bin inner loop.

## 10-c. What processing does the code repeat multiple times?

i. It makes two passes over trials for tongue processing (collect thresholds, then discretize), loops again over trials to build inputs and outputs, repeatedly casts the identical `time_from_tone` vector to `float32`, and repeatedly creates constant per-trial output vectors. `get_photostim_intervals` processes all trials before retained trials are selected.

ii.
```python
for ti in range(n_trials): ...  # first tongue pass
for ti in range(n_trials): ...  # second tongue pass
...
time_from_tone.astype(np.float32)
np.full(N_BINS, choice, dtype=np.int64)
```

iii. The notes do not explicitly acknowledge these repetitions; they frame the implementation as optimized and processing each source once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes `performance` solely for logs/metadata, computes tongue `p40`/`p60` but discards them after `process_session`, assigns unused local variables (`trial_starts`, `go_t`, `bin_width` in tongue processing), processes photostimulation intervals for trials later excluded, and contains extensive plotting code unused in the full run. Broad brain-region remapping is used in output, so it is not discarded.

ii.
```python
performance = n_correct / n_control_responded if n_control_responded > 0 else 0
trial_starts = trial_start_times[trial_indices]
tongue_discretized, p40, p60 = compute_tongue_y_all_trials(...)
```

iii. Performance was retained as a sanity-check metadata field, and optional plots were intended for visual validation. The notes value those checks, although they are not needed by decoder training.
