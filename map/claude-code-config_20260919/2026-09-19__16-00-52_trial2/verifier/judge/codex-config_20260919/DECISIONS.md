# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `sub-*/*.nwb` file, processes files in parallel, and reads each NWB directly with `h5py` into NumPy arrays.

ii. `files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))`; `with h5py.File(path, 'r') as f:`; `with ProcessPoolExecutor(max_workers=nworkers) as ex:`

iii. It treated one NWB as one session and chose direct HDF5 access plus multiprocessing for speed. Notes report 174 files and about 36 seconds on 24 workers.

## 1-b. How are the data split into subjects?

i. It reads `general/subject/subject_id`, makes a sorted unique subject list from accepted sessions, and stores each session's index into that list.

ii. `d['subject'] = f['general/subject/subject_id'][()].decode()`; `subjects = sorted({r['subject'] for r in results})`; `'subject_idx': np.array([sub_index[r['subject']] for r in results], dtype=np.int64)`

iii. The NWB subject field is the canonical identifier; the reported result has 28 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Results are sorted by a subject/timestamp identifier parsed from the filename; rejected sessions are omitted.

ii. `d['session_id'] = session_id_from_path(path)`; `results.sort(key=lambda r: r['session_id'])`

iii. The archive is organized one file per recording session. The agent additionally applied paper-inspired session-performance, trial-count, and tongue-tracking rejection rules.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. The agent maps `units/obs_intervals` starts back to trial rows, then indexes trial columns and the one-per-trial go events with those ephys-covered indices.

ii. `ephys_trials = np.searchsorted(trial_start_all, obs[:, 0])`; `d['trial_start'] = trial_start_all[et]`; `d['go_time'] = ev['go_start_times']['timestamps'][:][et]`

iii. It reasoned that some behavioral trials lack electrophysiology and asserted that observed-interval starts exactly match trial starts.

## 1-e. How are trials filtered based on quality controls?

i. It first retains ephys-covered trials, then removes both auto-water and free-water trials and trials with less than 90% video coverage of the four-second window. These removals can also cause session rejection through session-level criteria.

ii. `keep = (~d['auto_water']) & (~d['free_water'])`; `cov = video_coverage(d, win_start)`; `keep &= cov >= MIN_VIDEO_COVERAGE`

iii. Water trials were considered non-behavioral reports because reward is not choice-contingent. Full-window video was required to label tongue output. Photostim, early-lick, and ignore trials were deliberately retained because they define requested variables.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, with `classification`, anatomy, and go-cue timestamps used for curation and alignment.

ii. `spike_times = u['spike_times'][:]`; `unit_spikes = [spike_times[starts[i]:st_index[i]] for i in idx]`; `go = d['go_time']`

iii. Spike times are the raw neural representation, while go times define the requested windows.

## 2-b. How is the `neural` data processed?

i. Spikes are assigned to nonoverlapping trial windows and 50-ms bins with vectorized indexing/`bincount`; counts are divided by 0.05 to produce float32 Hz. No smoothing or normalization is applied.

ii. `b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)`; `counts = np.bincount(flat, minlength=n_units * n_trials * N_BINS)`; `return (counts / BIN_WIDTH).astype(np.float32)`

iii. This matches the reference analysis's rate=True histogram concept while implementing the task-mandated bins efficiently.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'`, an annotation mapping to one of 14 regions, and at least one spike in all retained extracted windows. Hemisphere is added to the region label.

ii. `keep_unit = good & (regions != '')`; `active = fr.any(axis=(1, 2))`; `fr = fr[active]`

iii. The classifier is the paper's spike-sorting QC. The agent justified anatomical and silence filters as matching region-based analyses and removing uninformative units; it did not apply a 2-Hz cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial starts at go cue minus 2.5 s. Session-absolute spike timestamps are binned from that start through go cue plus 1.5 s.

ii. `win_start = go + T_START`; `b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)`

iii. NWB event and spike timestamps share a clock, so no interpolation or offset correction was considered necessary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. It creates 80 nonoverlapping 50-ms bins over `[-2.5, 1.5)` seconds. Raw spikes are histogrammed directly into that grid; there is no later rebinning.

ii. `BIN_WIDTH = 0.05`; `N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))`; `BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)`

iii. The width and extraction interval are explicit decoder requirements.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, per-trial go times, and bin centers, selecting the last sample start at or before each go cue.

ii. `si = np.searchsorted(d['sample_start'], d['go_time'], side='right') - 1`; `return d['sample_start'][si] - d['go_time']`

iii. Early licks can replay the sample epoch, so the final preceding onset is the instruction sequence actually preceding that go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone onset is expressed relative to go, then subtracted from every go-relative bin center to yield seconds since tone onset.

ii. `tone_rel = tone_onset_rel(d)[trial_idx]`; `time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]`

iii. This directly constructs the requested continuous, time-varying input.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 bin centers used for the neural histograms.

ii. `BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2`; `np.stack([time_from_tone[i], stim[i]])`

iii. Sharing the go-relative centers ensures elementwise temporal correspondence.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It reads `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, associates events with trials using trial starts, and subtracts each trial's go time.

ii. `d['stim_on'] = ev['photostim_start_times']['timestamps'][:]`; `ti = np.searchsorted(d['trial_start'], d['stim_on'], side='right') - 1`; `on[ti] = s_on - d['go_time'][ti]`

iii. Recorded event timestamps were preferred over assuming a fixed stimulation epoch because timing varies among sessions.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The input is 1 when a bin center lies inclusively between the recorded laser on/off times and 0 otherwise.

ii. `stim = ((BIN_CENTERS[None, :] >= o) & (BIN_CENTERS[None, :] <= f_)).astype(np.float32)`

iii. Center sampling yields exactly ten bins for a nominal 0.5-s stimulus and avoids treating a roughly 1-ms stop-time overshoot as another active bin.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Event times are converted to go-relative time and sampled at the neural bin centers.

ii. `off[ti] = s_off - d['go_time'][ti]`; `BIN_CENTERS[None, :] >= o`

iii. The streams share the NWB clock and the same go-cue origin.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial `outcome` and `trial_instruction`.

ii. `hit = d['outcome'] == 'hit'`; `miss = d['outcome'] == 'miss'`; `left = d['instruction'] == 'left'`

iii. The trial table lacks a direct choice field: hits use the instructed side, misses the opposite side, and ignores indicate no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. It maps left/right/no lick to 0/1/2 and broadcasts the per-trial category across all 80 bins.

ii. `ch = np.full(len(d['outcome']), 2, dtype=np.int64)`; `ch[hit & left] = 0`; `ch[miss & left] = 1`; `np.full(N_BINS, choice[i])`

iii. Broadcasting permits all outputs, including time-varying tongue position, to share a `(4,80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii. `d['outcome'] = _decode(trials['outcome'][:])[et]`

iii. The raw categories exactly match the requested ignore/miss/hit output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to 0 ignore, 1 miss, 2 hit and broadcast across 80 bins.

ii. `outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}`; `np.full(N_BINS, outcome[i])`

iii. Fixed integer coding supplies the categorical decoder target.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii. `d['early_lick'] = _decode(trials['early_lick'][:])[et]`

iii. The NWB trial table explicitly records this trial-level flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` becomes 1 and all other expected values (`'no early'`) become 0; the value is broadcast across bins.

ii. `early = (d['early_lick'] == 'early').astype(np.int64)[trial_idx]`; `np.full(N_BINS, early[i])`

iii. This provides the requested no/yes categorical output while keeping early-lick trials.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y) and 2 (DeepLabCut likelihood) of `Camera0_side_TongueTracking/data`.

ii. `tongue = tk['data'][:]`; `d['tongue_t'] = tk['timestamps'][:]`; `d['tongue_y'] = tongue[:, 1]`; `d['tongue_lik'] = tongue[:, 2]`

iii. This is the side-camera tongue marker stream; likelihood determines visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood over 0.5 are assigned to retained trial/bin windows and averaged per bin. Bins without a visible frame remain class 3. Percentiles are computed from visible bin means in retained trial windows.

ii. `visible = d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH`; `ybar[has] = ysum[has] / nvis[has]`; `p_lo, p_hi = np.percentile(ybar[has], [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])`

iii. The likelihood is strongly bimodal, so 0.5 was viewed as insensitive. Notes say percentiles should be per session and use bin means rather than raw frames, but the implementation limits the percentile population to retained extracted trial windows.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. For bins with visible tongue, values below p40 are 0, values strictly above p60 are 2, and the rest (including exact boundaries) are 1; no-visible bins are 3.

ii. `out = np.full(ybar.shape, 3, dtype=np.int64)`; `cls = np.where(ybar < p_lo, 0, np.where(ybar > p_hi, 2, 1))`; `out[has] = cls[has]`

iii. This follows the requested four labels, with thresholds calculated per accepted session from accepted trial-window bin means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Visible camera timestamps are assigned to the same `win_start = go-2.5` windows and 50-ms bin indices as spikes.

ii. `ti = np.searchsorted(win_start, vt, side='right') - 1`; `b = ((vt - win_start[ti]) / BIN_WIDTH).astype(np.int64)`

iii. Camera, events, and spikes use the common session clock; the agent additionally excludes trials without near-complete video coverage.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Trials outside ephys coverage are excluded; inadequate-video trials are excluded; invisible tongue bins get class 3; one implausible tongue-tracking session is rejected; unmappable/silent units are removed. Worker exceptions are caught and the whole source session is skipped.

ii. `keep &= cov >= MIN_VIDEO_COVERAGE`; `if vis_frac > MAX_TONGUE_VISIBLE_FRACTION: return None, info`; `except Exception as e: ... return None, dict(..., rejected=f'error: {e}')`

iii. The agent avoided imputing absent neural/video data and documented archive limitations. It viewed extreme visibility as tracking failure and exception isolation as protection for the full parallel run.

## 10-a. What are the most time-consuming steps of the code?

i. NWB/HDF5 reading and neural spike binning are the timed per-session costs; serialization of the large pickle is another material cost. Sessions are parallelized.

ii. `t_read = time.time() - t0`; `fr = bin_spikes(d['unit_spikes'], win_start)`; `with ProcessPoolExecutor(max_workers=nworkers)`; `pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)`

iii. The notes identify file reads and spike processing as the scalable work and report roughly 36 s conversion using 24 workers.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Core spike and tongue binning are already vectorized with `searchsorted`/`bincount`. Remaining loops include unit spike-list extraction, hemisphere assignment, per-trial packing, and plotting/debug loops; packing could be array-based but the target ultimately requires lists.

ii. `unit_spikes = [spike_times[starts[i]:st_index[i]] for i in idx]`; `for j, i in enumerate(idx):`; `neural = [np.ascontiguousarray(fr[:, i, :]) for i in range(n_tr)]`

iii. The agent explicitly optimized away trial-by-unit histogram loops with a single-session histogram and prioritized clarity for ragged extraction and final packaging.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly derives trial-window assignments separately for spikes and visible tongue frames, repeatedly allocates/broadcasts constant trial outputs during packing, and each worker independently opens and decodes one file. The shared bin grid itself is defined only once.

ii. Both `bin_spikes` and `tongue_bin_position` use `ti = np.searchsorted(win_start, ..., side='right') - 1`; packing calls `np.full(N_BINS, ...)` three times per trial.

iii. The repetition reflects different timestamp streams and the required nested output format; the agent otherwise designed a single pass per session.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads several fields used only for curation/provenance (`task`, lick events, anatomy details, trial stops), computes extensive timing/session diagnostics, and optionally builds debug payloads/plots that are removed before serialization. It also bins all ephys trials before selecting retained trials.

ii. `d['lick_left'] = ev['left_lick_times']['timestamps'][:]`; `fr = bin_spikes(d['unit_spikes'], win_start); fr = fr[:, trial_idx, :]`; `if 'debug' in res: ... del res['debug']`

iii. These were used for validation, plots, rejection logic, or provenance rather than decoder tensors. The notes emphasize sanity checks and traceability, accepting some extra processing.
