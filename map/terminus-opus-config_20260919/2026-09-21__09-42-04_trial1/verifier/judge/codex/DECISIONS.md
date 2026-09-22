# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every NWB file under `/app/data/sub-*`, sorts the paths, and converts one file per session with `h5py`. By default it uses a 12-process pool; `--sample` limits this to two files.

ii. `files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))` and `with h5py.File(filepath, 'r') as f:`; parallel conversion uses `with Pool(args.nproc) as pool: ... pool.imap(_worker, ...)`.

iii. It justified NWB/HDF5 as the published container, sorting as deterministic ordering, and multiprocessing/whole-session reads as speedups. It reported 174 files and 28 mice before curation.

## 1-b. How are the data split into subjects?

i. Each session reads `general/subject/subject_id`; unique IDs are sorted and each retained session is mapped to its index.

ii. `subject = _decode(f['general/subject/subject_id'][()])`; `subjects = sorted({s['subject'] for s in sessions})`; `subject_idx = np.array([subject_index[s['subject']] for s in sessions])`.

iii. The notes identify this as the canonical NWB subject field and verify 28 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; retained files become output sessions in sorted file order. The NWB identifier and filename are stored in metadata.

ii. `identifier = _decode(f['identifier'][()])` and `'neural': [s['neural'] for s in sessions]`.

iii. The agent states that the DANDI layout is one file per acquisition session. A session is dropped if it has no retained units or fewer than two usable trials.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials and are paired with go-cue events. If counts differ, each go cue is assigned to the preceding trial start; trials lacking finite go cues are removed.

ii. `start_time = tr['start_time'][:]`; `gi = np.searchsorted(start_time, go_all, side='right') - 1`; `go[gi] = go_all`.

iii. It regarded the NWB trials table as authoritative, noted that normally there is exactly one go cue per trial, and added the fallback for malformed/missing events.

## 1-e. How are trials filtered based on quality controls?

i. It removes auto-water and free-water trials, trials without a finite go cue, trials outside the intersection of observation intervals for retained units, and trials whose binned activity contains no spike across all retained neurons. Early-lick, ignore, and photostimulation trials remain. Sessions with fewer than two trials are dropped.

ii. `keep = (auto_water == 0) & (free_water == 0) & np.isfinite(go) & observed`; later, `nonempty = fr.sum(axis=(0, 2)) > 0`.

iii. It cites the repository's regular-trial mask for removing water-delivery trials, but retains early-lick/ignore/photostim because they are requested decoder variables. Observation and all-zero filtering prevent unrecorded trials from appearing as genuine zero firing.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and `spike_times_index`, go-cue timestamps, `units/classification`, and nonempty `units/anno_name`.

ii. `spike_times = u['spike_times'][:]`; `spike_index = u['spike_times_index'][:]`; `good = (classification == 'good') & (anno != '')`.

iii. The agent says spike times are the neural representation and that classifier-good units with histology reproduce the reference good-unit lists.

## 2-b. How is the `neural` data processed?

i. For each retained unit, spikes are counted in all trial/bin edges using `searchsorted`, differenced within trials, converted to Hz by division by 0.05, and stored as float32. No smoothing or normalization is applied.

ii. `pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)`; `out[k] = np.diff(pos, axis=1)`; `out /= BIN_SIZE`.

iii. It follows the reference convention of spike count divided by window width while adopting the task-required bins. Flattening edges vectorizes over trials.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and a nonempty CCF annotation. Sessions with zero such units are discarded.

ii. `good = (classification == 'good') & (anno != '')`; `unit_idx = np.where(good)[0]`; `if len(unit_idx) == 0: return None`.

iii. The classifier label is described as the Chen/Liu region-specific QC result; the extra annotation condition is justified as matching the method repository's ephys-plus-histology requirement. The notes say all good units are annotated in practice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges are formed by adding the fixed -2.5 to +1.5 s grid to every trial's go-cue timestamp, then spikes are counted against those edges.

ii. `edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()`.

iii. The streams share the NWB session clock, so no interpolation or clock correction is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The result has 80 nonoverlapping 50-ms bins from -2.5 to +1.5 s relative to go cue. Raw spike timestamps are binned directly; the paper's sliding 40-ms/3.4-ms representation is not reused.

ii. `BIN_SIZE = 0.05`; `N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))`.

iii. This deliberate change follows the decoder instructions while retaining the reference rate calculation.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, go-cue times, and trial starts. The selected tone is the last sample onset before go cue.

ii. `j = np.searchsorted(sample_all, go, side='right') - 1`; `tone = ... sample_all[...]`.

iii. Early licking can replay the sample epoch, so the agent chose the final onset that precedes the response go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. It adds each bin center relative to go cue to the go-minus-tone interval. Missing or out-of-trial tone events fall back to the nominal 1.85-s interval.

ii. `tone = np.where(bad_tone, go - 1.85, tone)`; `time_from_tone = (go_keep - tone[trial_idx])[:, None] + BIN_CENTERS[None, :]`.

iii. The notes justify 1.85 s as 0.65-s sample plus 1.2-s delay and describe the fallback as an edge-case safeguard.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the same go-relative 50-ms bin centers used by neural activity.

ii. `BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])` and the formula above.

iii. Shared bin centers make input column `t` correspond to neural column `t`.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, `photostim_power`, trial start, and go-cue time.

ii. `ps_onset = ...`; `ps_dur = ...`; `ps_power = ...`; `has_stim = (ps_power > 0) & np.isfinite(ps_onset) & np.isfinite(ps_dur)`.

iii. The onset is stored relative to trial start, so trial start and go cue are needed to put it on the aligned axis; positive power distinguishes actual stimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. It creates a binary series and labels a bin 1 if any part of the bin overlaps the stimulation interval; otherwise 0.

ii. `overlap = (BIN_EDGES[1:] > s0) & (BIN_EDGES[:-1] < s1)`; `photostim[k, overlap] = 1.0`.

iii. It intended this to represent whether the laser was on during each bin and retained stimulated trials because photostimulation is a requested input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-relative onset is converted to go-relative time, then compared with the identical neural bin edges.

ii. `s0 = start_time[i] + ps_onset[i] - go[i]`; `s1 = s0 + ps_dur[i]`.

iii. This places laser and neural activity on the common go-cue clock.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial `outcome` and `trial_instruction`.

ii. `licked_left = np.where(oc == 'hit', ins == 'left', ins == 'right')`; `choice = np.where(oc == 'ignore', 2, np.where(licked_left, 0, 1))`.

iii. A hit is the instructed side, a miss the opposite side, and ignore means no response. The agent spot-checked this against lick events.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Codes are 0 left, 1 right, and 2 no lick, repeated across all 80 bins.

ii. `np.full(N_BINS, choice[i])` is stacked into output row 0.

iii. Repetition lets per-trial and time-varying outputs share one `(4,80)` target array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii. `outcome = np.array([_decode(x) for x in tr['outcome'][:]])`.

iii. The source already contains the requested three categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Ignore, miss, and hit map to 0, 1, and 2 and are repeated over time.

ii. `outcome_code = np.where(oc == 'ignore', 0, np.where(oc == 'miss', 1, 2))`; `np.full(N_BINS, outcome_code[i])`.

iii. The mapping follows requested output-value order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It uses `intervals/trials/early_lick`.

ii. `early = np.array([_decode(x) for x in tr['early_lick'][:]])`.

iii. The NWB trial table explicitly supplies the flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'no early'` maps to 0 and every other value maps to 1; the value is repeated over all bins.

ii. `early_code = np.array([0 if e == 'no early' else 1 for e in el_tr])`.

iii. It interprets the trial flag as the requested no/yes category.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses the data and timestamps of `Camera0_side_TongueTracking`: data column 1 is y and column 2 is tracking likelihood, plus go times for alignment.

ii. `vdata = tt_ds['data'][:]`; `vts = tt_ds['timestamps'][:]`; `vis = vdata[:, 2] > TONGUE_LIKELIHOOD_THRESH`; `yv = vdata[:, 1]`.

iii. The notes identify this as the side-view tongue marker and use likelihood to distinguish visible from retracted/unreliable tongue measurements.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood above 0.9 are visible. Per session, percentiles are computed from all raw visible-frame y values. Within each trial/bin, the last visible frame wins; bins without one are class 3.

ii. `thr_lo, thr_hi = np.percentile(yv[vis], [40, 60])`; `ybin[idx] = yy`; `cls = np.full((ntrials, N_BINS), 3, dtype=np.int64)`.

iii. The agent claims last-frame selection matches the repository marker-alignment convention and notes likelihood is strongly bimodal, making 0.9 relatively insensitive.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. A last-frame y below the session 40th percentile is 0, above the 60th is 2, and inclusive middle values are 1; no visible frame is 3.

ii. `c = np.where(ybin[good] < thr_lo, 0, np.where(ybin[good] > thr_hi, 2, 1))`.

iii. It follows the requested per-session 40/60 split, but defines those percentiles on visible raw frames.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera samples in each `[go-2.5, go+1.5)` window are assigned to the same 50-ms go-relative bins as spikes; the last visible sample assigned to each bin is used.

ii. `lo = np.searchsorted(ts, t0, side='left')`; `idx = np.floor((tt - t0[i]) / BIN_SIZE).astype(int)`.

iii. NWB camera and spike timestamps share the session clock, and the notes cite reference last-frame alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Byte/string decoding maps malformed nonstrings to empty text; units without annotations and sessions without units are dropped; absent go cues and unrecorded/all-zero trials are dropped; invalid tones use a nominal fallback; invisible tongue bins use category 3; worker exceptions are printed and return `None`.

ii. `return ''` in `_decode`; `tone = np.where(bad_tone, go - 1.85, tone)`; `except Exception as e: ... return None`.

iii. The notes frame exclusion as preferable to fabricating neural data, while legitimate tongue absence gets an explicit category. They enumerate partial recordings, a stopped final trial, NaN annotations, and missing-event safeguards.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies reading full spike/video arrays, per-unit spike binning, tongue processing, multiprocessing overhead, and serializing the roughly 12-GB result; measured per-session stages are recorded in `timings`.

ii. `timings['read_spikes'] = ...`; `timings['bin_spikes'] = ...`; `timings['tongue'] = ...`; `pickle.dump(data, fh, protocol=4)`.

iii. It reports whole-session conversion around 0.3-0.6 s in samples and says output pickling dominates the full run after parallelization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining explicit loops are per unit for spike binning, per trial for tongue binning, per selected trial for photostimulation, and per trial to construct list outputs. Trials/bins in spike binning are already vectorized; photostimulation and output construction could be vectorized, and tongue samples could be grouped globally.

ii. `for k, u in enumerate(unit_idx):`; `for i in range(ntrials):`; `for k, i in enumerate(trial_idx):`.

iii. The notes emphasize that flattening all trial edges reduced the costly nested spike loops and that ragged unit spike trains still require a unit-level operation.

## 10-c. What processing does the code repeat multiple times?

i. It decodes `classification` and `anno_name` twice in each session (before observation filtering and again for final unit selection), iterates retained trials for photostim, tongue, and list construction, and converts closely related structures into both intermediate dense arrays and per-trial lists.

ii. `classification_pre = ...` / `classification = ...`; `anno_pre = ...` / `anno = ...`.

iii. No explicit justification is given for duplicate unit-column reads; the first copy is needed before deriving observation coverage, though the values could be reused.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes electrode y, hemisphere labels, per-session timing/debug fields, original trial indices, and plotting-only inputs; most are not decoder tensors. Hemisphere is retained only in metadata, while `ey`, several session fields, and timings are discarded during final assembly. It also computes coarse brain-region labels because the target schema requires them, even though the supplied decoder may not use them.

ii. `ex, ey, ez = ...`; `hemi = ...`; session fields include `'trial_idx'`, `'timings'`, and `'elapsed'`, but final assembly retains only selected metadata.

iii. These support validation, diagnostics, and documentation. The agent generally argues computed fields are useful for sanity checks, though they are not all consumed downstream.
