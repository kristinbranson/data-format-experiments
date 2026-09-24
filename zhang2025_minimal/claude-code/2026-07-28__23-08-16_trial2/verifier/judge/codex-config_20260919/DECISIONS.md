# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent initializes ONE on `/app/data/one_cache`, reads the Zhang BWM release CSV, takes its unique session IDs, then loads trials/wheel/camera per session with `SessionLoader` and spikes/clusters per listed probe with `SpikeSortingLoader`. It attempts all 459 release sessions, not a dataset-availability search or datalimit CSV restriction.

ii. `bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)`; `eids = bwm_df['eid'].unique()`; `sl = SessionLoader(one=one, eid=eid)`; `ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)`.

iii. The trajectory says it found 393 apparently complete sessions during inspection but ultimately processed the release CSV's 459 unique sessions through ONE. It justified ONE as correctly resolving revision-tagged files and reported 444 successful sessions after missing-data skips.

## 1-b. How are the data split into subjects?

i. Subject names come from the release CSV rows for each `eid`; a first-seen ordered list and mapping assign a subject index to every retained session.

ii. `subject = rows.iloc[0]['subject']`; `if subject not in subject_to_idx: subject_to_idx[subject] = len(subjects_list)`; `subject_idx_list.append(sm['subject_idx'])`.

iii. The agent relied on the BWM release metadata's explicit subject field and noted that the completed data contained 139 subjects, matching the paper.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the release CSV is treated as one session; all probe rows sharing that `eid` are selected and merged.

ii. `eids = bwm_df['eid'].unique()`; `rows = bwm_df[bwm_df['eid'] == eid]`.

iii. The trajectory treats the release table's `eid` as the session identifier and iterates it directly.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies one table row per trial. Surviving rows provide `stimOn_times`; spike and behavioral windows are extracted independently around each onset and appended as trial arrays.

ii. `sl.load_trials()`; `align_times = masked_trials['stimOn_times'].values`; `for trial in range(n_trials): session_neural.append(sm['binned_spikes'][trial])`.

iii. The agent followed the native trials table and reference 2-second trial-window organization.

## 1-e. How are trials filtered based on quality controls?

i. It retains trials with first-movement reaction time in `[0.08, 2.0]`, nonmissing stimulus, choice, feedback, prior, first movement, and feedback type, and nonzero choice. It subsequently requires wheel and camera coverage/finite values. It skips a session if fewer than 10 trials survive either stage.

ii. `mask &= rt >= min_rt`; `mask &= rt <= max_rt`; `for event in nan_exclude: mask &= ~trials[event].isnull()`; `mask &= trials['choice'] != 0`; `combined_mask = wheel_mask & whisker_mask`; `if combined_mask.sum() < 10: continue`.

iii. It explicitly stated these filters matched the reference code and papers. The 10-trial session cutoff was not separately justified; it exceeds the task's minimum of two.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from each probe's `spikes.times` and `spikes.clusters`; merged cluster/channel metadata supplies the cluster index and Beryl region label.

ii. `spikes, clusters, channels = ssl.load_spike_sorting()`; `clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()`.

iii. The agent identified spike times and assignments as the needed neural source and used cluster metadata for anatomical indexing.

## 2-b. How is the `neural` data processed?

i. Probes are merged, cluster IDs are offset, spikes are time-sorted, and counts are accumulated into a `(trial, cluster, 100)` float32 array. Counts are saved directly; they are not divided by 20 ms into Hz.

ii. `spikes_copy['clusters'] = spikes_copy['clusters'] + cluster_max`; `np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)`; `session_neural.append(sm['binned_spikes'][trial].astype(np.float32))`.

iii. The trajectory says this matched the reference binning configuration and optimized the initial slow trial binning while checking identical sample results. It did not discuss the reference solution's conversion from counts to firing rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It deliberately uses every spike-sorted cluster. No `label >= 1` quality filter and no `void` anatomical filter is applied.

ii. `cluster_ids = np.arange(len(clusters))`; the module docstring says: `Neuron quality: use all clusters (no QC filter)`.

iii. It reasoned that Zhang's `prepare_data -> load_spiking_data` default is `qc=None` and claimed good-cluster filtering occurs only at the region-analysis level. This conflicts with the human's choice to reproduce the data paper's stringent 75,708-unit QC and discard `void`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are selected from `stimOn_times - 0.5` (inclusive) to `stimOn_times + 1.5` (exclusive); subtracting the window start via the bin formula aligns them to stimulus onset.

ii. `t_start = align_times[trial_idx] + window[0]`; `i_start = np.searchsorted(sorted_times, t_start, side='left')`; `bin_indices = np.floor((times_in_window - t_start) / binsize)`.

iii. The agent repeatedly states stimulus onset and the `[-0.5, 1.5]` window match the reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms: a 2-second window is rebinned into 100 nonoverlapping spike-count bins. There is no further neural resampling or smoothing.

ii. `BINSIZE = 0.02`; `N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))`.

iii. It cites the papers/reference configuration of 20-ms bins and 100 steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the stimulus-aligned window constants and bin size, with `stimOn_times` defining absolute alignment; it does not use another raw signal.

ii. `align_times = masked_trials['stimOn_times'].values`; `time_since_stim = np.linspace(WINDOW[0] + BINSIZE, WINDOW[1], N_BINS)`.

iii. The agent says the same relative time axis applies to every trial and follows the reference alignment/window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It creates 100 evenly spaced values from -0.48 through 1.50 seconds (inclusive), i.e. right bin edges rather than centers.

ii. `np.linspace(WINDOW[0] + BINSIZE, WINDOW[1], N_BINS).astype(np.float32)`.

iii. It explicitly believed these were bin centers “matching reference code,” although the human uses centers from -0.49 through 1.49.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The 100 values are stacked in the same positions as the 100 neural bins, but represent each bin's right edge, not its center.

ii. `inp = np.stack([time_input, trial_num], axis=0)` while neural is `sm['binned_spikes'][trial]` with the same `N_BINS`.

iii. The agent intended exact binwise alignment and described the vector as the bin centers, but the implemented grid is shifted +10 ms from the true centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It derives block boundaries from consecutive changes in the trials table's `probabilityLeft`.

ii. `pLeft = trials_df['probabilityLeft'].values`; `if i == 0 or pLeft[i] != pLeft[i-1]: block_counter = 1`.

iii. It recognized that the prior is constant within a block and therefore changes identify block starts.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. It loops over the unfiltered trials, resets at each prior change, numbers positions starting at 1, then masks trials and broadcasts the value across all 100 time bins.

ii. `trial_num[i] = block_counter; block_counter += 1`; `return trial_num[mask]`; `trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial])`.

iii. Computing before filtering preserves real within-block position. The agent expressly chose “1-indexed,” unlike the human's zero-based count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It derives choice from `trials.choice`; no-choice zero trials are filtered first.

ii. `choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()`.

iii. The agent correctly identified the raw choice column and the need to exclude no-responses.

## 5-b. What processing is involved in computing `output` *Choice*?

i. It maps raw `+1` to category 1 and every other retained value (`-1`) to 0, then broadcasts it over time. Thus its labels are opposite the required IBL mapping: raw `+1` is left and should be 0; raw `-1` is right and should be 1.

ii. `choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)`; `choice_arr = np.full(N_BINS, sm['choice'][trial])`.

iii. The comment claims “left=-1 -> 0, right=1 -> 1,” revealing the agent misunderstood IBL's sign convention despite intending to follow the decoder spec.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It derives prior from the trials table's `probabilityLeft`.

ii. `pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values`.

iii. The agent followed the explicitly supplied raw variable and task mapping.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are recoded to 0, 1, and 2, respectively, then broadcast across the trial's 100 bins.

ii. `prior[pLeft == 0.2] = 0`; `prior[pLeft == 0.5] = 1`; `prior[pLeft == 0.8] = 2`; `prior_arr = np.full(N_BINS, sm['prior'][trial])`.

iii. It directly follows the decoder specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses the absolute value of `SessionLoader`'s wheel velocity, which loader derives from raw wheel position and timestamps.

ii. `sl.load_wheel()`; `wheel_times = sl.wheel['times'].values`; `wheel_speed = np.abs(sl.wheel['velocity'].values)`.

iii. The agent says absolute wheel velocity matches the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Loader-computed velocity is absolutized, linearly interpolated at 100 target timestamps, and later discretized. Trials lacking two samples, finite values, or edge coverage within 20 ms are rejected.

ii. `f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')`; `binned_beh[trial_idx] = f(x_interp)`; `wheel_binned, wheel_mask = interpolate_behavior_to_bins(...)`.

iii. It intended to match the reference interpolation and added explicit coverage checks to avoid bad boundary data.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two tercile thresholds are computed once over every retained wheel sample in every session; each session is digitized with these global thresholds into 0/1/2.

ii. `wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])`; `wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges)`.

iii. The agent described “global discretization thresholds” and sought three quantile-balanced categories. This differs from the human's per-session terciles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel values are evaluated on a stimulus-relative grid from -0.48 to +1.50 seconds and paired positionally with neural bins. This is the bins' right-edge grid, shifted 10 ms later than neural bin centers.

ii. `x_interp = np.linspace(t_start + binsize, t_end, n_bins)`.

iii. The agent believed this matched the reference bin centers, so the 10-ms offset was unintended.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It tries the left camera's `whiskerMotionEnergy` and frame times first, falling back to the right camera.

ii. `sl.load_motion_energy(views=['left'])`; `me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values`; analogous right-camera fallback.

iii. It followed the reference's side-camera motion energy and reported skipping sessions lacking it.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace is linearly interpolated to 100 trial timestamps, subjected to the same coverage/finite checks as wheel, and then discretized without other filtering or normalization.

ii. `whisker_binned, whisker_mask = interpolate_behavior_to_bins(me_times, me_values, align_times, WINDOW, BINSIZE)`.

iii. The agent states interpolation of the released motion energy matches the reference approach.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It computes dataset-global 33rd/67th percentiles across all retained sessions and digitizes samples into 0/1/2.

ii. `whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])`; `whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges)`.

iii. It deliberately used global quantiles for three bins. The human instead thresholds each session independently.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera values are interpolated on the same -0.48 to +1.50 right-edge grid used for wheel and placed alongside the 100 neural bins, causing the same +10-ms center offset.

ii. `x_interp = np.linspace(t_start + binsize, t_end, n_bins)`.

iii. The agent intended a shared bin-center grid but implemented right edges.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Probe-load failures are warned and remaining probes used; sessions with no probes, no camera motion energy, or fewer than 10 valid trials are skipped. Behavior trials with missing/nonfinite/insufficient/poor-coverage samples are removed. Broad per-session exceptions are logged and skipped.

ii. `except Exception as e: print(f'  Warning: Failed to load probe {pid}: {e}')`; `if whisker_binned is None: continue`; `combined_mask = wheel_mask & whisker_mask`; outer `except Exception ... continue`.

iii. The trajectory reports 444 processed and 15 skipped sessions, mostly for absent motion energy, and regards skipping incomplete data as expected.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large spike files through ONE and spike binning dominated runtime. The initial full run was killed after 46 sessions in about 30 minutes; optimized binning improved throughput, but the agent concluded disk loading remained the main bottleneck.

ii. `spikes, clusters, channels = ssl.load_spike_sorting()`; the trial loop and `np.add.at(...)` perform binning.

iii. The trajectory explicitly measured both phases and attributed remaining ~16 seconds/session largely to loading 1–2 GB of spike data per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial spike selection/binning and per-trial behavior interpolation could be vectorized further; cluster-ID mapping and final per-trial list construction are also loops. The agent did optimize the original spike implementation but retained a loop over trials.

ii. `for trial_idx in range(n_trials): ... np.add.at(...)`; `for trial_idx in range(n_trials): ... f(x_interp)`; `for trial in range(n_trials): session_neural.append(...)`.

iii. The trajectory says the first spike-binning implementation was the bottleneck, so it rewrote it with searchsorted and vectorized accumulation and verified identical sample output.

## 10-c. What processing does the code repeat multiple times?

i. It sorts spike times in `merge_probes` and again in `bin_spikes_in_window`; behavior interpolation repeats nearly identical work separately for wheel and whisker; final arrays are copied/cast trial by trial. The unused helper `discretize_to_bins` duplicates the percentile/digitize logic later written inline.

ii. `sort_idx = np.argsort(merged_spikes_dict['times'], kind='stable')` and later `sort_idx = np.argsort(spike_times)`; two calls to `interpolate_behavior_to_bins(...)`; inline `np.percentile`/`np.digitize` despite `discretize_to_bins`.

iii. The trajectory does not justify these repetitions; its optimization discussion focused only on spike binning and I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several storage lists/sets are initialized but never used; `discretize_to_bins` and imported `ismember` are unused. Cluster/channel metadata is fully merged although only acronym/index information is ultimately retained. Raw behavior is held both in global lists and `session_meta`, increasing memory, and final sanity checks scan every trial/output again.

ii. `all_neural = []`, `all_input = []`, `all_output = []`, `all_subject_names = []`, `all_brain_regions_set = set()`; `from iblutil.numerical import ismember`; `def discretize_to_bins(...)`; `all_wheel_speed_raw.append(wheel_binned)` plus `'wheel_binned': wheel_binned`.

iii. The agent did not identify these as unnecessary. Its trajectory instead accepted substantial memory use and ran exhaustive post-conversion sanity summaries.
