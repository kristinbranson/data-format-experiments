# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 44 sessions, their source folder, and ALM probe(s). It opens each `data_structure_<animal>_<date>.mat` with h5py when possible and otherwise scipy, and separately loads the corresponding motion-energy file.

ii. `SESSION_META = [...]`; `fdata, fmt = load_mat_file(data_fpath)`; `me_fpath = os.path.join(DATA_DIR, dataset_dir, f'motionEnergy_{sess_id}.mat')`

iii. The notes say the list and probe indices came from the paper's loading scripts, and both MATLAB v7.3/HDF5 and v5 layouts occur.

## 1-b. How are the data split into subjects?

i. The animal field in each hard-coded session tuple is treated as the subject. Unique animal names are sorted, and each retained session receives the corresponding index.

ii. `all_animals = sorted(set(r['anm'] for r in all_results))`; `subject_idx = np.array([subjects.index(r['anm']) for r in all_results])`

iii. The notes report 14 animals across the 44 sessions and acknowledge that this is one more than the paper's aggregate count.

## 1-c. How are the data split into sessions?

i. Every `SESSION_META` tuple/file is one session and becomes one outer-list element in `neural`, `input`, and `output`; sessions failing minimum trial or neuron checks are skipped.

ii. `for ... in sessions_to_process: result = process_session(...)`; `neural = [r['neural'] for r in all_results]`

iii. The agent chose all 25 fixed-delay and 19 randomized-delay sessions listed by the authors' loaders.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the number of trials. Behavioral arrays, spike trial labels, trajectory cell entries, and motion-energy entries are indexed with the same zero-based trial index after converting spike labels from one-based indexing.

ii. `ntrials = int(...)`; `trial_num = tr_idx + 1`; `neural_trials.append(trialdat[:, :, tr_idx].T.copy())`

iii. The agent regarded the Bpod trial table and stored spike trial numbers as direct trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Early-lick and photostimulation trials are excluded; ignore trials are retained. Sessions with fewer than two such trials are skipped. Trials after electrophysiology recording ended are not removed and can contain all-zero neural data.

ii. `valid_trials_mask = ~early & ~stim_enable`; `if len(valid_trial_indices) < 2: return None`

iii. The notes justify early/stim removal from reference practice and keeping ignores because the decoder requires an ignore class. They explicitly accept 61 all-zero trial warnings in two sessions as recording-ending-early cases.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu` on the listed ALM probes: each cluster's `quality`, `trialtm`, and `trial`, plus `bp.ev.goCue` for alignment.

ii. `trialtm = f[trialtm_ref][:].flatten()`; `trial = f[trial_ref][:].flatten().astype(int)`; `aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]`

iii. The notes identify the authors' `findClusters`, `alignSpikes`, and `getSeq` functions as the model.

## 2-b. How is the `neural` data processed?

i. Aligned spikes are histogrammed into 10-ms bins, divided by 0.01 s to obtain Hz, and convolved with a 15-sample one-sided Gaussian kernel using a reflected prefix. The result is float32 firing rate.

ii. `DT = 1.0 / 100`; `counts, _ = np.histogram(spk_t, bins=time_edges)`; `fr = counts.astype(np.float32) / DT`; `smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)`

iii. The agent chose the 10-ms tutorial setting and described the kernel as matching `mySmooth`; it noted that another default used 5 ms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Labels exactly equal to `garbage`, `gabrga`, `noisy`, or `real?` are removed case-sensitively. Remaining neurons are retained only when mean rate across all bins and trials exceeds 0.5 Hz; sessions with fewer than ten neurons are skipped.

ii. `EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}`; `keep_mask = mean_fr > LOW_FR`; `LOW_FR = 0.5`

iii. The notes prefer `getDefaultParams.m`'s 0.5-Hz value over the paper's stated 1 Hz and claim case-sensitive matching to MATLAB.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For every spike, the go-cue time of its labeled trial is subtracted from its trial-relative time before histogramming over -2.5 to +2.5 seconds.

ii. `tr = trial[t_idx] - 1`; `aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]`

iii. This was chosen to match the authors' `alignSpikes.m` with `goCue` as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 500 10-ms bins from -2.5 to +2.5 s. Spikes are binned directly at 10 ms; video streams are interpolated to the same bin centers.

ii. `DT = 1.0 / 100`; `time_edges = np.arange(TMIN, TMAX + DT, DT)`

iii. The agent selected 10 ms from `WorkingWithDataObjs.m`, despite documenting a 5-ms default elsewhere in the reference code.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is constructed from the configured window and bin width, conceptually relative to each trial's `bp.ev.goCue`, rather than read as a raw per-trial variable.

ii. `time_centers = time_edges[:-1] + DT / 2`

iii. The notes say this continuous axis implements the requested go-cue-relative input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers are computed once and copied as a `(1, 500)` float32 array for every retained trial.

ii. `inp = time_centers.astype(np.float32).reshape(1, -1)`

iii. No additional processing was considered necessary.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Its values are the centers of the exact histogram edges used for neural spikes, so each input sample corresponds to the same neural bin.

ii. `time_centers = time_edges[:-1] + DT / 2`; `np.histogram(spk_t, bins=time_edges)`

iii. The shared grid was intended to guarantee alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from `bp.hit`, `bp.miss`, `bp.R`, and `bp.L`; `R/L` encode the instructed/correct side rather than the observed choice.

ii. `hit = ...astype(bool)`; `miss = ...astype(bool)`; `R = ...`; `L = ...`

iii. The notes record that an initial direct R/L implementation was corrected after recognizing this distinction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Correct trials use the instructed side, misses use its opposite, and all remaining/ignore trials are class 2 (`none`). The scalar class is broadcast across time.

ii. `lick_dir = np.full(ntrials, 2)`; `lick_dir[hit & R] = 1`; `lick_dir[miss & R] = 0`; `out[0, :] = lick_dir[tr_idx]`

iii. The agent states that this represents actual lick direction and preserves the requested no-lick class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived directly from `bp.autowater`.

ii. `autowater = f['obj/bp/autowater'][:].flatten().astype(bool)`

iii. The notes identify autowater trials as water-cued context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater is encoded WC=0 and all other trials DR=1, then broadcast across time.

ii. `context[~autowater] = 1`; `context[autowater] = 0`; `out[1, :] = context[tr_idx]`

iii. This directly follows the requested category order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome uses `bp.hit` and `bp.miss`; `bp.no` is loaded but not needed because neither hit nor miss implies ignore.

ii. `no_resp = ...`; `outcome = np.full(ntrials, 2)`; `outcome[hit] = 1`; `outcome[miss] = 0`

iii. The agent retained ignore trials specifically because the decoder output requests that class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Miss is incorrect=0, hit is correct=1, and the default is ignore=2; the result is broadcast over all time bins.

ii. `outcome = np.full(ntrials, 2, dtype=int)`; `out[2, :] = outcome[tr_idx]`

iii. The mapping follows the specified output value order.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses the side-camera `obj.traj` feature named `tongue`: x, y, likelihood, and frame times. It does not use the bottom-camera `top_tongue` feature.

ii. `side_cam = f[side_ref]`; `if name == 'tongue': tongue_idx = j`; `tongue_x = ts[tongue_idx, 0, :]`

iii. The notes broadly say velocity comes from DLC x/y with confidence filtering, without justifying omission of the second tongue view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. First differences of x and y are divided by the median frame interval, combined as Euclidean speed, padded by repeating the last value, masked where confidence is below 0.9, and linearly interpolated to decoder time centers. No coordinate smoothing or cross-camera normalization is applied.

ii. `dx = np.diff(x) / dt_frames`; `vel = np.sqrt(dx**2 + dy**2)`; `vel[tongue_conf < 0.9] = np.nan`; `np.interp(time_centers, aligned_ft[valid], vel[valid], left=np.nan, right=np.nan)`

iii. The agent describes confidence <0.9 as not visible and considered interpolation sufficient for placing video on the decoder grid.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The median of all finite tongue speeds from retained trials in the session is the threshold: below=0, at/above=1, NaN=2.

ii. `threshold = np.median(flat_valid)`; `disc[valid_mask & (trial_vel < threshold)] = 0`; `disc[valid_mask & (trial_vel >= threshold)] = 1`

iii. This directly implements the requested per-session 50th-percentile split and not-visible category.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video shift is estimated as median SpikeGLX bit-start/sample-rate minus median behavioral bit-start, with a 0.5-s fallback. Go cue is then subtracted and speeds are interpolated onto neural bin centers.

ii. `vidshift = sglx_bitstart / sglx_fs - bitStart`; `aligned_ft = frame_times - vidshift - goCue[tr_idx]`; `np.interp(time_centers, aligned_ft[valid], vel[valid], ...)`

iii. The notes cite `findVideoOffset.m`, although the reference uses modes and frame binning rather than these fallbacks/interpolation.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses every bottom-camera DLC feature whose name contains `paw`, including both tracked paws, with x, y, likelihood, and bottom-camera frame times.

ii. `paw_indices = [j for j, name in enumerate(bottom_feats) if 'paw' in name.lower()]`

iii. The notes state simply that paw velocity is derived from DLC coordinates; they do not document why both paws are averaged.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Speed is first-difference Euclidean displacement per median frame interval for each paw; confidence <0.9 becomes NaN, paw speeds are averaged with `nanmean`, then interpolated to decoder times. There is no position smoothing.

ii. `vel = compute_velocity_from_xy(px, py, aligned_ft, dt_frames)`; `vel[pc < 0.9] = np.nan`; `avg_vel = np.nanmean(paw_vels, axis=0)`

iii. The agent used the same basic DLC velocity method as for tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session median over finite values in retained trials defines below=0 and at/above=1; NaN is not-visible=2.

ii. `paw_disc = discretize_velocity(paw_vel_all, valid_trial_indices, not_visible_val=2)`

iii. This implements the prompt's per-session percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are corrected with the session video shift, made go-cue-relative, and linearly interpolated to the neural time centers.

ii. `aligned_ft = frame_times - vidshift - goCue[tr_idx]`; `paw_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], avg_vel[valid], ...)`

iii. The agent intended a common grid for all streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is read from the separate `motionEnergy_<session>.mat` variable `me`, handling a struct with `data` or a direct object array, and uses side-camera frame times for timing.

ii. `me_data = sio.loadmat(me_fpath, squeeze_me=False)`; `me_raw = me_data['me']`; `me_trials = me_raw['data'][0, 0]`

iii. The notes say separate files exist across sessions and document fixing a direct-array variant discovered during conversion.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Raw per-frame motion energy is linearly interpolated to decoder bin centers. Any remaining internal/edge NaNs in a partly observed trial are then nearest/linearly filled by sample index before discretization.

ii. `me_all[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], me_trial[valid])`; `col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])`

iii. The notes treat motion energy as already spatially processed and therefore requiring only temporal alignment and median thresholding.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Finite values in retained trials are split at their session median into below=0 and at/above=1; an entirely absent trace remains no-video=2.

ii. `me_disc = discretize_velocity(me_all, valid_trial_indices, not_visible_val=2)`

iii. This follows the specified 50th-percentile rule.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are offset-corrected and made go-cue-relative, then motion energy is interpolated to the neural bin centers. Synthetic 400-Hz frame times and a 0.5-s shift are fallbacks.

ii. `frame_times = np.arange(len(me_trial)) / 400.0`; `aligned_ft = frame_times - vidshift - goCue[tr_idx]`; `np.interp(time_centers, aligned_ft[valid], me_trial[valid])`

iii. The agent cites the common video-offset method and accepted fallbacks to keep sessions usable.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Many parsing errors are silently caught, leaving video arrays NaN and hence class 2. Missing timing can instead trigger assumed 0.5-s video offset or synthetic 400-Hz frames. Partially missing motion energy is filled. Late behavioral trials with no ephys are retained as all-zero neural trials.

ii. `except: vidshift = VIDEO_OFFSET_DEFAULT`; `except: frame_times = np.arange(len(me_trial)) / 400.0`; `except: continue`; `col[nans] = np.interp(...)`

iii. The notes call absent video “not visible/no video” and explicitly accept the zero-neural trials, prioritizing retaining trials and successful decoding.

## 11-a. What are the most time-consuming steps of the code?

i. File loading plus nested spike processing dominate. Full conversion was reported at about 132 seconds, roughly 3 seconds per session.

ii. `for i in range(n_neurons_raw):` followed by `for tr_idx in range(ntrials):`; `fdata, fmt = load_mat_file(data_fpath)`

iii. The trajectory estimated runtime from a two-session sample and observed larger sessions and scipy loads could be slower.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-spike go-cue subtraction, neuron-by-trial spike masking/histograms, per-neuron convolution, per-trial DLC processing, and output construction are explicit loops. Spike alignment and counting in particular could be vectorized.

ii. `for t_idx in range(len(trialtm)):`; `for i in range(n_neurons_raw):`; `for tr_idx in range(ntrials):`

iii. The agent did not document a vectorization analysis; its runtime notes treated the observed full runtime as acceptable.

## 11-c. What processing does the code repeat multiple times?

i. Nearly the entire session pipeline is duplicated for HDF5 and v5 files. Video-offset calculation appears independently in tongue/paw and motion-energy loaders, and the identical time input is allocated for every trial.

ii. `process_session_h5(...)` / `process_session_v5(...)`; separate `vidshift = ...` blocks; `inp = time_centers.astype(np.float32).reshape(1, -1)` inside trial loops.

iii. The notes justify separate format handling but do not discuss the duplicated processing cost.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `bp.no` is loaded but unused; HDF5 constructs `out_per_trial` and `out_time_varying` but discards both; `compute_velocity_from_xy` receives unused `frame_times`; and rates are computed for excluded trials before only valid trials are extracted.

ii. `no_resp = ...`; `out_per_trial = ...`; `out_time_varying = ...`; `def compute_velocity_from_xy(x, y, frame_times, dt_frames):`

iii. These costs were not identified in the notes; the agent focused its review on correctness and decoder accuracy.
