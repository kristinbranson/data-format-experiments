# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively finds every NWB file below the selected data directory, sorts the paths, opens each with `NWBHDF5IO`, and reads its trials, units, behavioral events, and tongue series. `--sample` limits this to the first two files.

ii. `files = sorted(args.data_dir.rglob('*.nwb'))` and `io = NWBHDF5IO(str(path), 'r', load_namespaces=True); nwb = io.read()`.

iii. The notes identify the release as one NWB file per session and report 174 files across 28 subjects; they treat the local release as authoritative.

## 1-b. How are the data split into subjects?

i. Each session uses `nwb.subject.subject_id`, falling back to the parent directory name. Subjects are accumulated in first-seen file order and sessions receive an integer index.

ii. `sid = getattr(getattr(nwb, 'subject', None), 'subject_id', None)` and `subject_to_idx[subj] = len(subjects)`.

iii. The notes recognize the 28 subjects from `sub-*` directories and prefer NWB metadata when available.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; accepted session lists are appended once per file, and the filename stem is stored as its ID.

ii. `for path in files: sess = process_session(path, ...)` and `'session_id': path.stem`.

iii. The notes state that the DANDI release contains one behavior/ecephys/ogen NWB file per session.

## 1-d. How are the data split into trials?

i. Trial start/stop rows define trials. Go and sample events are assigned by taking the first event inside each trial interval; trials without a finite go cue are skipped.

ii. `m = (event_times >= a) & (event_times <= b); out[i] = event_times[m][0]` and `if not np.isfinite(go): continue`.

iii. The mapping plan calls for NWB trial metadata and explicit event timestamps. It does not justify choosing the first sample event when early licks can replay an epoch.

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filter is presence of an assigned go event. It does not use unit observation intervals or remove `free_water` trials. A whole session is later skipped if it has fewer than two retained trials or no retained units.

ii. `if not np.isfinite(go): continue` and `if len(neural_trials) < 2 or len(reg_kept) == 0: ... continue`.

iii. The notes deliberately propose retaining early-lick and no-response trials because those are decoder targets, but leave the paper/session and missing-spike filters unresolved.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each selected unit's NWB `spike_times`, with go cue timestamps defining absolute bin edges.

ii. `spikes.append(np.asarray(nwb.units['spike_times'][i], dtype=float))` and `rel_edges = align_time + BIN_EDGES`.

iii. The notes map NWB unit spike times to the neural field and cite the explicit `go_start_times` stream.

## 2-b. How is the `neural` data processed?

i. For every trial and selected unit, `np.histogram` counts spikes in the go-aligned bins and divides by 0.05 s to produce Hz. The resulting neuron-by-time matrix is then transposed before saving, producing time-by-neuron matrices.

ii. `arr[i] = np.histogram(st, bins=rel_edges)[0].astype(np.float32) / BIN` and `neural_trials = [x.T.astype(np.float32, copy=False) for x in neural_trials]`.

iii. The notes planned 50-ms firing rates consistent across sessions, but do not justify the final transpose.

## 2-c. How is the `neural` data filtered based on quality controls?

i. A fuzzy column search selects the first quality-like unit column. On these files this resolves to `unit_quality`, and string values containing `good` are retained. The classifier `classification` is not searched explicitly.

ii. `good_col = find_first(cols, ['good', 'quality', 'label', 'unit_quality'])` and `mask = np.array([('good' in v) ... for v in sval])`.

iii. The notes say to use units labeled good by the reference classifier when available, but the implementation leaves the exact flag unresolved and uses a generic heuristic.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The first go event inside each trial is used as time zero, and relative edges from -2.5 to +1.5 s are added to its absolute timestamp.

ii. `go_times = assign_events_to_trials(get_event_times(nwb, 'go_start_times'), ...)` and `rel_edges = align_time + BIN_EDGES`.

iii. Go-cue alignment is explicitly identified in the notes as required and supported by NWB timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The intended resolution is 50 ms and raw spikes are histogrammed directly, with no smoothing. However, the edge expression creates 82 edges and therefore 81 bins, extending one bin beyond +1.5 s.

ii. `BIN = 0.05` and `BIN_EDGES = np.arange(T_START, T_END + BIN + 1e-9, BIN)`.

iii. The notes specify 50-ms bins over -2.5 to +1.5 s, but do not catch the off-by-one edge.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial start/stop boundaries, and the go cue. The first sample event inside a trial is treated as tone onset; absent samples fall back to `go - 1.5`.

ii. `sample_times = assign_events_to_trials(get_event_times(nwb, 'sample_start_times'), ...)` and `sample_times[i] if np.isfinite(sample_times[i]) else go - 1.5`.

iii. The notes identify the sample/tone timestamps but say the exact mapping still required refinement.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center, the absolute center time minus the assigned tone time is returned as a float32 vector.

ii. `time_from_tone = (go_time + BIN_CENTERS) - tone_time`.

iii. The mapping plan describes a continuous per-bin elapsed time since sample onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same `go_time + BIN_CENTERS` grid as the spike bins, so index positions correspond (subject to the shared 81-bin error).

ii. `time_from_tone = (go_time + BIN_CENTERS) - tone_time` and `rel_edges = align_time + BIN_EDGES`.

iii. The notes require a time-varying input aligned to the go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It derives global intervals by pairing `BehavioralEvents/photostim_start_times` and `photostim_stop_times` by position, truncating to the shorter stream.

ii. `s = get_event_times(nwb, 'photostim_start_times')`, `e = ...('photostim_stop_times')`, and `np.c_[s[:n], e[:n]]`.

iii. The notes explicitly map these event streams to a binary input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. It initializes zeros and marks a bin one when its center lies inclusively within any session photostimulation interval.

ii. `phot[(abs_centers >= a) & (abs_centers <= b)] = 1.0`.

iii. The notes describe a binary, time-varying indicator and note that stimulation ends before the go cue.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute photostimulation intervals are compared with the same go-aligned absolute bin centers used by the neural grid.

ii. `abs_centers = go_time + BIN_CENTERS`.

iii. The notes call for aligning event-stream intervals to the go cue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The code looks for a direct `choice`, `lick_direction`, or `response_side` column. None exists in the supplied trials table, so it derives choice from no raw variable and defaults every trial to class 2 (`no_lick`). It does not combine `trial_instruction` with `outcome`.

ii. `choice_col = find_first(cols, ['choice', 'lick_direction', 'response_side'])` and `choice = ... if choice_col is not None else 2`.

iii. The notes explicitly say the exact NWB encoding still needed determination; no justification is supplied for the fallback.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. If a direct column existed, strings containing left/right map to 0/1 and several other strings to 2; in this dataset all trials receive 2. The scalar is repeated across every time bin.

ii. `if 'left' in s: return 0`, `if 'right' in s: return 1`, and `np.full(len(BIN_CENTERS), choice, dtype=np.uint8)`.

iii. The notes require left/right/no-lick categorical labels but never validate the implemented inference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It finds and uses the trials-table `outcome` column.

ii. `outcome_col = find_first(cols, ['outcome', 'trial_outcome', 'result', 'correctness'])`.

iii. The mapping plan names trial metadata/outcome as the source.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped ignore=0, miss=1, and hit=2, then repeated over time.

ii. `if 'ignore' in s: return 0; if 'miss' in s: return 1; if 'hit' in s ...: return 2`.

iii. The notes list the required three categorical values.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It finds the trials-table `early_lick` column through a substring match.

ii. `early_col = find_first(cols, ['early', 'early_lick'])`.

iii. The mapping plan identifies NWB trial metadata as the source.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The mapper returns 1 whenever the string contains `early`; consequently both `early` and `no early` map to yes. The value is repeated across bins.

ii. `if 'true' in s or 'yes' in s or s == '1' or 'early' in s: return 1`.

iii. The notes only state that this should be a no/yes category and do not document or catch this string-matching error.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking`: column 1 as y, column 2 as likelihood/visibility, and the series timestamps.

ii. `y = data[:, 1]` and `vis = np.asarray(data[:, 2] > 0.5)`.

iii. The notes identify that series and say its coordinate/confidence columns needed identification; the code assumes the conventional layout.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. It linearly interpolates y and a numeric visibility flag at each bin center, then masks samples whose interpolated visibility is not above 0.5. It does not average all frames within each 50-ms bin.

ii. `y_interp = np.interp(sample_t, t, y, ...)`, `vis_interp = np.interp(... ) > 0.5`, and `y_interp[~vis_interp] = np.nan`.

iii. The notes planned time-varying tongue tracking and a not-visible class but did not settle or validate the aggregation method.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th/60th percentiles are computed over finite interpolated values from retained trial windows only. Values below, between (inclusive), and above become 0/1/2; nonfinite values become 3.

ii. `q40, q60 = np.percentile(all_y, [40, 60])` and assignments `d[...] = 0/1/2`, with `d` initialized to 3.

iii. The notes correctly require per-session percentiles, but do not justify restricting the percentile population to trial-window center samples rather than session-wide 50-ms means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is sampled at `go_time + BIN_CENTERS`, giving one value at each neural-bin center.

ii. `sample_t = go_time + BIN_CENTERS`.

iii. The notes require tongue output aligned to the go cue but include no completed alignment sanity check.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing go events drop trials; missing sample events use a 1.5-s pre-go fallback; missing photostim streams produce all-zero intervals; unavailable direct labels get fixed defaults; out-of-range/invisible tongue samples become class 3. Sessions with fewer than two trials or no units are skipped. There is no handling for trials lacking spikes despite behavioral records.

ii. Examples include `if not np.isfinite(go): continue`, `else go - 1.5`, `return np.empty((0, 2), dtype=float)`, and `else 2`/`else 0` label defaults.

iii. The notes acknowledge unresolved release discrepancies and planned checks, but most checks remain unchecked and later steps remain marked incomplete.

## 10-a. What are the most time-consuming steps of the code?

i. Re-reading the full tongue time series once per trial and histogramming every unit separately for every trial are the dominant computations; NWB I/O and pickling also contribute.

ii. `tongue_y_trials.append(interpolate_tongue_y(nwb, go))` calls `tongue_series(nwb)` inside the trial loop, while `for i, st in enumerate(spike_times): np.histogram(...)` is also called per trial.

iii. The notes leave runtime estimates and speedups blank, so this assessment follows the actual loop structure rather than a documented benchmark.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Event-to-trial assignment scans all events for every trial; neural binning loops over trials and units; tongue interpolation loops over trials; and region indexing loops over units. Trials can be vectorized with sorted-event searches, spike edge searches can cover all trials per unit, and the tongue series can be loaded/binned once per session.

ii. Representative loops are `for i, (a, b) in enumerate(...)`, `for i, st in enumerate(spike_times)`, and `for i, go in enumerate(go_times)`.

iii. The notes have placeholders for inefficiencies and speedups but contain no completed analysis.

## 10-c. What processing does the code repeat multiple times?

i. It reads the same full tongue `data` and `timestamps` from NWB for every retained trial. It also reconstructs neural absolute edges per trial and scans every session-level photostimulation interval for every trial.

ii. `interpolate_tongue_y` calls `tongue_series(nwb)` each time, and `for a, b in photostim_intervals` is inside `build_inputs`, itself called in the trial loop.

iii. No justification is documented; the development notes retain empty placeholders.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It imports matplotlib and `Counter` without using them, constructs an unused `lower` dictionary in `find_first`, and stores diagnostic column names/quantiles only in metadata. Most importantly, it does substantial repeated tongue I/O whose earlier copies are discarded after each trial.

ii. `import matplotlib.pyplot as plt`, `from collections import Counter`, and `lower = {c.lower(): c for c in cols}`.

iii. The notes do not discuss discarded work and leave the relevant review steps incomplete.
