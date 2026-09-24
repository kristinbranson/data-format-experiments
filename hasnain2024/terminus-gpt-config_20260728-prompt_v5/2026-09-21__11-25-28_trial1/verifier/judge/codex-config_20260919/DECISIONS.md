# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `data_structure_*.mat` only in `/app/data/RandomizedDelay_Ephys_Behavior`, sorted the paths, and loaded each as HDF5 with a scipy fallback. Motion energy was loaded separately by matching session name. It therefore loaded the randomized-delay directory rather than the reference's curated 44-session list across both task directories.

ii. `files = sorted(BASE.glob('data_structure_*.mat'))`; `sessions = [load_session(f) for f in files]`; `with h5py.File(path, 'r') ... except OSError: sio.loadmat(...)`.

iii. The notes say that directory contained 22 files, that two lacked `clu`, and that the paper reported the randomized-delay experiment as 19 sessions. The agent chose file discovery and later exclusions, believing the residual mismatch reflected an undocumented paper exclusion.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from the filename, unique IDs are sorted, and each retained session receives an integer index.

ii. `'subject': path.stem.split('_')[2]`; `subjects = sorted({s['subject'] for s in sessions})`; `subject_idx.append(subject_map[sess['subject']])`.

iii. The notes treat the filename/session naming convention as the reliable source of subject identity.

## 1-c. How are the data split into sessions?

i. Each discovered `data_structure_<subject>_<date>.mat` is one session. Sessions without `clu`, or with fewer than 10 selected units, are excluded. Twenty sessions were retained.

ii. `sessions = [s for s in sessions if s['has_clu']]`; `sessions = [s for s in sessions if len(select_units(s)) >= 10]`.

iii. The agent cited the decoder's need for neural data and the paper's at-least-10-units criterion.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the count. The code iterates trial numbers 1 through `Ntrials`, selects spikes by `clu.trial`, and creates one neural/input/output item per row of behavioral arrays.

ii. `for tr in range(1, session['ntrials'] + 1):`; `mask = (u['trial'] == tr)`; `for tr in range(ntr): ... outputs.append(out)`.

iii. The notes identify the behavioral arrays and go-cue array as per-trial fields.

## 1-e. How are trials filtered based on quality controls?

i. They are not filtered. Although `early` is loaded and the notes considered excluding early trials, all `Ntrials` enter the result; photostimulation and trials beyond recorded ephys are also not checked.

ii. `'early': ...`; `build_neural_trials(sess, units)` and `build_outputs(sess, ...)` both use the complete `ntrials`.

iii. The notes say early/ignore trials were omitted in many paper analyses but ignore must remain an output class. They never document a final justification for retaining early or photostimulation trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data use `obj.clu` fields `trial`, `trialtm`, `tm`, and `quality`. `tm` is used for the rate filter; `trial` and `trialtm` generate trial traces. `bp.ev.goCue` is loaded but not used in neural construction.

ii. `mask = (u['trial'] == tr)`; `np.histogram(u['trialtm'][mask], bins=edges)`; `unit_fr_gt1` uses `unit['tm']`.

iii. The agent states that reference `alignSpikes` uses `trialtm - event` and planned go-cue alignment, but its implementation omits that operation.

## 2-b. How is the `neural` data processed?

i. Per unit/trial spike times are histogrammed into 20 ms bins over -2.4 to 2.0, divided by 0.02 to obtain Hz, and convolved with a Gaussian kernel of sigma 2 bins (40 ms), using zero-like `same` convolution boundaries.

ii. `counts, _ = np.histogram(...)`; `arr[i] = gaussian_smooth(counts.astype(np.float32) / dt, smooth_sigma)`.

iii. The notes cite the reference's bin-rate-and-smooth sequence, but do not justify changing the reference's 5 ms bins and roughly 14 ms smoothing to 20 ms and 40 ms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are retained only for normalized labels `{multi,fair,good,great,excellent}` and a whole-recording rate `len(tm)/(max(tm)-min(tm)) > 1 Hz`; sessions then need at least 10 retained units.

ii. `KEEP_QUALITY = {...}`; `keep = [u for u in session['units'] if u['quality'] in KEEP_QUALITY and unit_fr_gt1(u)]`.

iii. The notes explain that quality filtering plus >1 Hz brought 3,139 raw clusters down to 874, close to the paper's 845, and cite the paper's >1 Hz and session-size rules.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is not actually aligned. The metadata says go cue, but raw `trialtm` is binned without subtracting `session['goCue'][tr-1]`.

ii. `np.histogram(u['trialtm'][mask], bins=edges)`; there is no go-cue subtraction in `build_neural_trials`.

iii. The notes explicitly say alignment should be `trialtm - event` and claim go-cue consistency, making this an implementation error rather than a documented alternate decision.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural spikes are rebinned to 20 ms (220 bins from -2.4 to 2.0 s). Video and motion traces are interpolated to the same number of bins, but by normalized sample position rather than time.

ii. `dt=0.02`; `edges = np.arange(tmin, tmax + dt, dt)`; `np.interp(np.linspace(0,1,ntime), ...)`; metadata has `'time_bin_size': 20.0`.

iii. The README documents 20 ms, but the notes only identified the reference bin/smoothing scheme and provide no rationale for departing from its 5 ms resolution.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the hard-coded neural histogram edges, not from a raw variable; the loaded `goCue` values are unused.

ii. `centers = edges[:-1] + dt / 2`; `INPUT_NAMES = ['time_from_go_cue']`.

iii. The agent intended a continuous time signal relative to the required alignment event.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin centers from -2.39 through 1.99 s are cast to float32 and copied for every trial.

ii. `arr = centers.astype(np.float32)[None, :]`; `return [arr.copy() for _ in range(ntrials)]`.

iii. The notes describe this as the decoder's sole input and report an exact bin-center sanity check.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It has the same length and nominal histogram centers as neural data, but it is not semantically aligned because neural `trialtm` was never shifted by go cue.

ii. Both are produced from `centers` returned by `build_neural_trials`, but the spike histogram uses unshifted `trialtm`.

iii. The agent claimed the input and converted traces matched expected centers, without checking the missing event subtraction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived directly from instructed-side flags `bp.L` and `bp.R`, without using hit/miss/no to infer the actual lick.

ii. `lick[session['L']] = 0`; `lick[session['R']] = 1`.

iii. The planning notes proposed `R`, `L`, “and/or lick event side,” and `none` for ignore trials, but the implementation did not complete that logic.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Values default to none (2), then any L/R trial is overwritten as left/right. Consequently ignore trials with an instructed side are incorrectly assigned a lick, and misses are assigned the instructed rather than actual opposite direction.

ii. `lick = np.full(ntr, 2, ...)`; `lick[L] = 0`; `lick[R] = 1`.

iii. The agent wanted categorical left/right/none labels but supplied no justification for equating instruction side with response side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It uses `obj.bp.autowater`.

ii. `'autowater': ...`; `context = np.where(session['autowater'], 0, 1)`.

iii. Reference-code comments in the notes were used to infer that autowater distinguishes WC from DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct binary mapping assigns autowater true to WC=0 and false to DR=1, then repeats the value across time.

ii. `np.where(..., 0, 1)`; `np.full(ntime, context[tr])`.

iii. The notes call for this polarity, though an earlier checklist said it still needed confirmation.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It uses `bp.miss`, `bp.hit`, and `bp.no`.

ii. The loaders read all three fields and `build_outputs` indexes each Boolean array.

iii. The notes identify these mutually exclusive behavioral condition fields and retain ignore because the requested decoder output requires it.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Defaults to ignore=2; miss maps to incorrect=0, hit to correct=1, and no to ignore=2. The per-trial class is repeated across time.

ii. `outcome[session['miss']] = 0`; `outcome[session['hit']] = 1`; `outcome[session['no']] = 2`.

iii. This follows the requested categories; retaining ignores is explicitly justified despite their exclusion from some paper analyses.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. For HDF5 sessions only, it uses the first trajectory view's `ts` x/y/likelihood for the first found feature among `tongue`, `left_tongue`, and `right_tongue`. `frameTimes` is read but discarded; the second tongue camera is not used.

ii. `extract_h5_traj_view(... view0 ...)`; `speed_from_ts(... ['tongue','left_tongue','right_tongue'])`.

iii. The notes cite trajectory features and DLC visibility, and acknowledge that velocity outputs began as placeholders before HDF5 extraction was added.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Frames with likelihood below 0.5 are invalid. Speed is Euclidean first difference in pixel coordinates (not divided by frame time and not smoothed). NaNs are filled with the trial's median speed, then the entire trace and validity mask are independently interpolated by normalized position to 220 samples.

ii. `dx = np.diff(xy[0], prepend=np.nan)`; `spd = np.sqrt(dx**2 + dy**2)`; `np.nan_to_num(... nan=np.nanmedian(...))`; `resample_trace_to_bins(...)`.

iii. The notes broadly planned speed from coordinates and missing-class encoding, but do not justify the 0.5 cutoff, median fill, lack of time derivative, or normalized-position interpolation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session median over all resampled bins whose interpolated validity is at least 0.5 defines classes 0/1; other bins are 2.

ii. `med = np.nanmedian(tongue_vals)`; `tongue_out[tr, valid] = (rr[valid] >= med)`.

iii. The per-session 50th percentile and not-visible class follow the prompt.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Only array length is matched. Frame timestamps, session video-clock offset, trial go cue, and neural bin edges are not used; normalized trace position is interpolated to 220 points.

ii. `x_old = np.linspace(0.0, 1.0, trace.size)` and `x_new = np.linspace(0.0, 1.0, ntime)`; `ft0` is unused.

iii. The agent claims go-cue alignment globally but provides no justification for discarding frame times.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. For HDF5 sessions, it uses the second trajectory view's x/y/likelihood and selects `top_paw`, falling back to `bottom_paw`. MAT sessions remain missing.

ii. `extract_h5_traj_view(... view1 ...)`; `speed_from_ts(... ['top_paw','bottom_paw'])`.

iii. The notes planned paw landmarks where available and missing class otherwise.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. It uses the same likelihood>=0.5, unsmoothed coordinate first difference, median NaN fill, and normalized-position interpolation as tongue velocity.

ii. `speed_from_ts` is shared; `rr = resample_trace_to_bins(np.nan_to_num(spd1, ...), ntime)`.

iii. No detailed justification beyond deriving speed and representing unavailable video is documented.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session median across resampled valid paw bins produces below-median=0 and at/above=1; invalid bins remain not-visible=2.

ii. `med = np.nanmedian(paw_vals)`; `paw_out[tr, valid] = (rr[valid] >= med)`.

iii. This implements the requested per-session 50th-percentile rule.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. It is not time aligned; its length is normalized to neural length while `ft1` is discarded.

ii. `feat1, ts1, ft1 = ...`; subsequent code never uses `ft1` and calls `resample_trace_to_bins`.

iii. No justification for the missing clock/go-cue alignment is given.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It primarily uses `me.data` from `motionEnergy_<session>.mat`; it attempts a scalar per-trial `obj.me` fallback. `moveThresh` is loaded but unused.

ii. `load_motion_energy_sidecar`; `trial_traces = np.asarray(me_data, dtype=object).ravel()`; fallback `session['me_obj']`.

iii. The notes identify sidecar traces as the source and prescribe no-video when unavailable.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each trace is linearly interpolated by normalized sample position to 220 bins. A session median over all resampled values discretizes it. The fallback repeats one per-trial scalar across time.

ii. `rr = resample_trace_to_bins(trc, ntime)`; `med = np.nanmedian(session_vals)`; `motion[i] = (rr >= med)`.

iii. The notes say motion energy was already reduced to a trace and needed resampling/discretization, but do not justify ignoring actual frame times.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Available bins are split by the median of all resampled values in the session; unavailable data remain class 2. The sidecar's `moveThresh` is deliberately/implicitly ignored.

ii. `med = np.nanmedian(...)`; `(rr >= med).astype(np.int64)`; initial `motion = np.full(..., 2)`.

iii. A session median matches the explicit decoder instruction; missing files are documented as no-video.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It is not aligned by camera timestamps, clock offset, or go cue. Traces are merely stretched/compressed to neural array length.

ii. `resample_trace_to_bins` uses only `trace.size` and `ntime` with `[0,1]` coordinates.

iii. The notes claim go-cue alignment and report a self-consistency check against this same resampling, not an external timing check.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Many parsing/extraction errors are silently swallowed. Missing trajectory/video leaves class 2; low-likelihood bins do likewise, though NaN speeds are median-filled before validity masking. Missing neural sessions are dropped. Alternate MAT trajectories are not processed. No bad trials are removed.

ii. Multiple `except Exception: pass`; arrays initialize with `np.full(..., 2)`; `sessions = [s for s in sessions if s['has_clu']]`.

iii. The documented principle is to encode unavailable video as not-visible/no-video rather than drop otherwise usable trials, but silent broad exceptions obscure genuine format errors.

## 11-a. What are the most time-consuming steps of the code?

i. The agent does not profile or identify them conclusively. Likely costs are loading all session/HDF5 cluster arrays, nested trial-by-unit spike histograms, and reopening each HDF5 file for per-trial trajectory dereferencing.

ii. `sessions = [load_session(f) for f in files]`; nested loops in `build_neural_trials`; `with h5py.File(...)` in `build_video_outputs_h5`.

iii. Notes report roughly 1–3 seconds/session in sample mode but leave timing tables incomplete.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested trial-by-unit neural loop could histogram a unit across all trials at once. Output stacking and median collection could also be array-based; variable-length video extraction reasonably remains trialwise.

ii. `for tr ...: for i, u in enumerate(units): mask = ...`; `session_vals.extend(rr.tolist())`; `for tr in range(ntr): ... np.vstack(...)`.

iii. The notes contain placeholders for code inefficiencies/speedups and no substantive justification.

## 11-c. What processing does the code repeat multiple times?

i. Session HDF5 files are first fully opened for behavioral/neural loading and reopened for video. Each unit's full trial vector is compared anew for every trial. Identical input and per-trial categorical time rows are copied repeatedly.

ii. `load_session(f)` then `h5py.File(BASE / f"data_structure_{...}")`; `(u['trial'] == tr)` inside nested loops; `[arr.copy() for _ in range(ntrials)]`.

iii. No rationale is documented; efficiency notes were left unfilled.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `sample`, `reward`, unit `site`, and motion-energy `moveThresh` without using them; `frameTimes` are decoded but discarded; `tm` is retained solely for filtering. It also builds copied time arrays and repeated constant categorical rows.

ii. Loader keys include `'sample'`, `'reward'`, and `'site'`; `me_thresh` and `ft0`/`ft1` are assigned but unused.

iii. These fields aided exploration or were anticipated for alignment/thresholding, but the final notes do not justify their continued loading.
