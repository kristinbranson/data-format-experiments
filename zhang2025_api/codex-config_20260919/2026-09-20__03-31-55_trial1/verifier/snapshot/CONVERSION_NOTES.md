# Dataset Conversion Notes

## Overview
- **Dataset**: IBL brain-wide map public data (local ONE cache)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `dataarchitecture.pdf`
- `datapaper.pdf`
- `decoder.py`
- `docker-compose.yaml`
- `ibl_docs/`
- `methodpaper.pdf`
- `methods.txt`
- `stage_cache.sh`
- `train_decoder.py`

Environment check: Python 3.13.15; NumPy 2.3.5; PyTorch 2.6.0+cu124. The required notes-file checkpoint (`ls -la /app/CONVERSION_NOTES.md`) passed.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `prepare_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Resolves all probe insertions with `ONE.eid2pid`, loads each with `SpikeSortingLoader`, merges probes, loads trials/masks and continuous behavior. |
| `load_spiking_data` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING/CURATION | Uses `SpikeSortingLoader.load_spike_sorting` and `merge_clusters`; optional `label >= qc` selection. Reference call leaves `qc=None`, so all clusters are retained while `label >= 1` is recorded as metadata. |
| `merge_probes` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Reindexes clusters across probes, concatenates, and stable-sorts spikes by time. |
| `load_trials_and_mask` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Excludes RT outside 0.08–2 s, trial duration >10 s, missing required events, and no-choice (`choice == 0`); retains initial unbiased block. |
| `bin_spiking_data` / `get_spike_data_per_interval` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Makes stimulus-aligned intervals and bins counts with half-open time intervals using `bincount2D`; reference parameters are 20 ms bins and −0.5 to +1.5 s around `stimOn_times`. |
| `load_target_behavior` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | LOADING | Uses `SessionLoader`; wheel speed is absolute smoothed wheel velocity; whisker energy uses left camera with right fallback. |
| `get_behavior_per_interval` / `bin_behaviors` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | PROCESSING | Linearly interpolates continuous behavior at bin-end timestamps from −0.48 through +1.5 s; forms per-trial choice, block, reward, and contrast. |
| `align_spike_behavior` | `code/code_zhang2025/src/utils/ibl_data_utils.py` | CURATION | Removes trials missing behavior and applies the trial mask, then enforces equal trial counts. |
| `standardize_spike_data` | `code/code_zhang2025/src/utils/data_loader_utils.py` | PROCESSING | Decoder-side standardization of binned activity per time step using training data. |
| `create_dataset` | `code/code_zhang2025/src/utils/dataset_utils.py` | PROCESSING | Stores trial spike matrices sparsely with behavior and cluster metadata. |

### Notes
The applicable source is `code_zhang2025`; the other tree is the supporting `ibllib` package. The reference cache script uses the ONE API plus brainbox loaders, merges all probes within an EID, maps regions to the Beryl atlas, and outputs trial × time × neuron count tensors. It sets `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`, and `binsize=0.02`. This is electrophysiology, so delta-F/F is not applicable. The code exposes cluster QC but its published caching path does not filter clusters (`qc=None`); this apparent choice will be checked against the papers and BWM release criteria in Steps 3–4. A potential reference-code bug is that `align_spike_behavior` combines Python lists using boolean `and`, meaning only the final behavior mask is effectively retained; our conversion should preserve intended intersection logic while documenting the correction.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
The local IBL ONE cache is rooted at `/app/data/one_cache`. It contains three complementary release-table directories (`2022_Q4_IBL_et_al_BWM`, `2025_Q3_IBL_et_al_BWM`, and `Brainwidemap`) and lab/subject/date/session ALF trees. All contents were queried through `one.api.One`; arrays/tables were loaded through `ONE.load_dataset`/`ONE.load_object`, never by direct file reads. The task-session population is the intersection of the 2025 behavioral release and Brainwidemap: 459 sessions, 139 subjects, and 699 pykilosort probe collections. Native formats are NumPy (`.npy`) for spike times/cluster IDs, wheel/camera timestamps and motion energy; Parquet (`.pqt`) for trials, cluster metrics and pose/features; JSON for probe descriptions.

Available task variables registered in ALF include `trials.table` (stimulus, choice, feedback, probabilityLeft and event times), separate go-cue/stim-off arrays, wheel timestamps/position, camera timestamps, camera ROI motion energy, pose/features, and licks. Ephys objects include spikes (times, clusters, depths, amplitudes, templates), cluster metrics/channels/depths/UUIDs, channel atlas coordinates and probe descriptions. The most recent staged spike-sort revision is selected by ONE (typically `#2024-05-06#`, with `#2024-03-12#` for some insertions).

Cache integrity note: the ONE index registers `_ibl_trials.table.pqt` for these sessions, but the staged path is unavailable to `ONE.load_object`; the independent `goCueTrigger_times` arrays are available for 458/459 sessions. This is a staging/index discrepancy, not permission to bypass ONE. It must be resolved before conversion (Steps 4–6); required choice and block labels cannot be inferred safely from wheel alone.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 621,733 clusters across 699 insertions; 75,708 have metrics `label >= 1` |
| Neurons / session | mean 1,354.54; median 1,299; range 135–3,140 |
| Subjects | 139 |
| Sessions / subject | 459 sessions total; mean 3.30, median 3, range 1–13 |
| Trials (total) | 295,675 directly counted from `goCueTrigger_times` for 458/459 sessions; one session's trial array is unstaged, so full total is slightly larger |
| Trials / session | available sessions: mean 645.58; median 601.5; range 401–1,525 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 621,733 units; 75,708 well-isolated | Data paper: “621,733 units (including multineuron activity)” and “75,708 well-isolated neurons” |
| Neurons / session | Method paper subset average 676, approximately 300 to >2,000; full release mean 1,354.5 units/session from data | Method paper STAR Methods; data-derived count includes all units and merged probes |
| Subjects | 139 (94 male, 45 female) | Data paper Methods |
| Sessions / subject | 459 sessions / 139 subjects | Data paper: “459 sessions, 699 insertions and 621,733 neurons remained” |
| Trials (total) | Approximately 296k before trial-level analysis exclusions (295,675 over 458 staged sessions plus one missing) | Data-derived; paper reports session mean rather than an exact total |
| Trials / session | Mean 645 recorded; analyses retained sessions with at least 400 trials | Data paper Results |
| Neural data time bin | 20 ms for 2-s reference cache/dynamic decoding; 50 ms in paper for static choice/prior analyses | Methods paper/code; task requires one common grid, favoring the 20-ms reference cache |
| Behavior data time bin | Native whisker signal described as 60 Hz in method paper (camera-specific 60/150 Hz in data paper); reference code linearly resamples to 20 ms | Method paper and data paper video methods |
| Reward rate | Not reported as one release-wide value | No numeric release-wide rate found; must measure after loading trials |
| Zero-contrast performance | 58.7 ± 0.4% correct | Data paper Results |
| Block structure | First 90 trials at 0.5; then 0.2/0.8 blocks, 20–100 trials, empirical mean 51 | Data paper Results/Methods |


### Processing Details
The methods-paper cache representation is a 2-s interval with 100 non-overlapping 20-ms spike-count bins. Its supplied code aligns −0.5 to +1.5 s around stimulus onset, matching the requested common alignment. The published variable-specific analysis instead uses −0.5 to +1.5 s stimulus-aligned choice, −0.6 to −0.1 s stimulus-aligned prior with 50-ms bins, and 0 to +1 s movement-aligned dynamic behaviors with 20-ms bins. Because the requested target requires one common temporal grid and explicitly says stimulus-onset alignment, the code’s 2-s/20-ms cache convention is the directly applicable reference. Continuous wheel speed is absolute smoothed wheel velocity. Whisker motion energy is mean absolute adjacent-frame difference within a whisker-pad ROI anchored between nose and eye; the reference code uses left camera with right-camera fallback and linear resampling.

### Curation Steps

**Neuron curation rules**:
The data paper’s well-isolated-unit rule is amplitude >50 µV, noise cutoff <20, and acceptable refractory-period violations (RIGOR metrics), represented in supplied cluster metrics as `label >= 1`. Paper region analyses further require Allen grey matter, at least five well-isolated neurons per session, and at least two sessions per region. However, the methods paper explicitly bins “all neurons” sorted by Kilosort 2.5, and the supplied caching code calls `load_spiking_data(..., qc=None)`. The conversion choice must be resolved in Step 4 based on which reference is applicable to neural decoding.

**Trial curation rules**:
Exclude trials missing choice, `probabilityLeft`, feedback type/time, stimulus-on time, or first-movement time. Exclude first-movement latency outside 0.08–2.00 s. Supplied code additionally imposes trial length ≤10 s and excludes no-choice (`choice == 0`) trials. Initial unbiased trials are retained. Release-level session criteria include ≥250 trials, ≥90% correct on full-contrast stimuli in both biased blocks, ≥3 incorrect included trials, and hardware QC; the paper’s main analyses retained sessions with ≥400 trials.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice | Paper uses accuracy/AUC; exact main-panel values are graphical, not tabulated. Example multi-session RRR comparisons span approximately 0.51–0.91 depending on session/model; multi-session improves choice accuracy by ~2% relative to single-session RRR in Fig. 2. |
| Prior | Evaluated with Pearson correlation (not categorical accuracy); Fig. 2 examples: 0.74 vs 0.37 and 0.76 vs 0.54 (RRR vs ridge). |
| Wheel speed | Evaluated with R² in the papers; exact aligned main-panel values graphical. Unaligned example R² values extend to ~0.75. |
| Whisker motion energy | Evaluated with R² in the papers; exact aligned main-panel values graphical. Unaligned example R² values extend to ~0.80. |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session population | `bwm_release.csv`/cache code selects BWM release EIDs | ONE releases intersect at 459 task sessions/139 mice/699 probes | 459 sessions/699 probes/139 mice after release criteria; method paper analyzes 433 sessions with required behavior | Start from 459 task EIDs; retain only sessions passing trial filters with both requested continuous outputs and ≥2 valid trials. This should approach the method paper’s 433 usable sessions. |
| Trial table | `SessionLoader.load_trials()` / `ONE.load_object` | Base `_ibl_trials.table.pqt` index entries point to unstaged paths; actual files are revision `#2025-03-03#` | Trial fields and filters require table | Patch only the in-memory ONE dataset `rel_path` to the known staged revision, then load with ONE. Test loads returned all required columns and expected 0.5 initial blocks. No direct data-file reading is used. |
| Spike-sort revisions | Code assumes default pykilosort collection | Staged preferred revisions are usually `#2024-05-06#` (some `#2024-03-12#`) | Kilosort 2.5 units, 621,733 total | Let ONE select the most recent/default staged revision through the base pykilosort collection; verified total exactly 621,733. |
| Neuron QC | Supplied cache calls `qc=None`; stores `label >= 1` only as metadata | 621,733 all clusters; 75,708 `label >= 1` | Method paper says “all neurons”; data paper anatomical analyses use 75,708 well-isolated units | Retain all clusters, matching the decoder paper and exact supplied cache path. Record quality in metadata statistics; do not silently substitute the stricter anatomical-analysis population. |
| Time alignment/binning | Cache code: stimulus onset, −0.5:+1.5 s, 20 ms | Required streams have timestamps sufficient for that interval | Static and dynamic paper analyses use different windows/alignment; user explicitly requires stimulus onset and a common grid | Use the supplied cache convention: 100 bins, −0.5:+1.5 s, 20 ms. This is the only reference-compatible common grid satisfying the task. |
| Continuous behavior sampling | Linear interpolation at bin ends; left-camera whisker energy with right fallback | Camera-specific native sampling; wheel available throughout | Paper describes 60-Hz whisker energy, movement-aligned dynamic targets | Follow cache code’s interpolation onto stimulus-aligned bin-end timestamps. Use left ROI motion energy and right fallback, matching code. |
| Prior definition | Cache code stores trial `probabilityLeft` as `block`; methods paper also studies an inferred continuous belief prior | `probabilityLeft` takes 0.2/0.5/0.8 | Decoder task explicitly maps 0.2→0, 0.5→1, 0.8→2 | Use categorical `probabilityLeft`, not the separately modeled latent belief, because the requested coding is exact and unambiguous. |
| Choice sign | Native IBL choice is −1/0/+1; code removes 0 | Loaded data contain both −1 and +1 | Decoder task requires left=0, right=1 | IBL convention is choice −1 = left and +1 = right; map −1→0, +1→1 after excluding 0/missing. |

All cross-source discrepancies are resolved. The trial-table issue was validated by loading two sessions through ONE after an in-memory index revision correction; returned keys include choice, probabilityLeft, stimulus/feedback/movement times, contrasts, reward and intervals. Hash/size warnings arise because the old indexed row describes the superseded base file; conversion will disable hash checking for this known revision mapping and perform shape/value checks instead.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `spikes.times`, `spikes.clusters` across all session probes | `neural` | Merge/reindex probes; count spikes in 100 half-open 20-ms bins from −0.5 to +1.5 s around each `stimOn_times`; transpose to neuron × time; float32 | `prepare_data`, `merge_probes`, `bin_spiking_data` | Retain all Kilosort clusters, matching reference decoder cache. |
| Relative bin-end times | `input[0]` | Fixed float32 vector −0.48, −0.46, …, 1.50 s | `get_behavior_per_interval` | “time since stimulus onset”, continuous/time-varying. |
| Run length within consecutive `probabilityLeft` block | `input[1]` | Zero-based integer ordinal reset whenever block probability changes; cast float32 and broadcast over 100 time bins | task-specific | The initial 0.5 block is treated as a block. |
| `trials.choice` | `output[0]` | Exclude missing/0; map native −1 (left)→0 and +1 (right)→1; broadcast across time | `load_trials_and_mask`, `bin_behaviors` | Static trial target represented on common time grid. |
| `trials.probabilityLeft` | `output[1]` | Exact map 0.2→0, 0.5→1, 0.8→2; broadcast across time | `bin_behaviors` (`block`) | Requested categorical “prior probability of left”; not inferred belief prior. |
| `SessionLoader.wheel.velocity` | `output[2]` | Absolute velocity; interpolate linearly at bin-end timestamps; session-wise tertile discretization of finite aligned samples to labels 0/1/2 | `load_target_behavior`, `get_behavior_per_interval` | Output values slow/medium/fast. Store thresholds in metadata. |
| Left `whiskerMotionEnergy` (right fallback) | `output[3]` | Linear interpolation at bin ends; session-wise tertile discretization of finite aligned samples to labels 0/1/2 | `load_target_behavior`, `get_behavior_per_interval` | Session-wise thresholds handle camera gain/ROI scale differences; store thresholds. |
| ONE session details `subject` | `subjects`, `subject_idx` | Unique subject strings in first-seen order and per-session integer index | ONE `get_details` | Only retained sessions represented. |
| Cluster atlas locations | `brain_regions`, `brain_region_idx` | Map Allen IDs/acronyms to Beryl acronyms with `BrainRegions`; global region vocabulary plus per-session neuron indices | `list_brain_regions` | `void`/root values retained explicitly rather than dropping neurons. |

### Key Decisions
1. **Common representation**: Every input/output trial is `(dimension, 100)` so static variables are broadcast. This is required because static and dynamic targets coexist and the validator/trainer concatenate all time points.
2. **20-ms count data**: Preserve integer spike counts numerically but store float32 as requested by the validator. Do not convert counts to rates or smooth them; the reference cache and decoder consume counts.
3. **Trial curation**: Apply the paper/code intersection: required non-NaN events, RT 0.08–2.0 s inclusive, feedback−goCue ≤10 s, choice nonzero, valid prior code, complete wheel and whisker coverage of the full interval, and finite interpolated samples. Keep sessions only with ≥2 surviving trials.
4. **Behavior interpolation**: Match reference bin-end timestamps and linear interpolation. Reject intervals whose continuous stream begins/ends more than one bin from requested boundaries. Do not impute NaNs because categorical discretization would conceal missing data.
5. **Three-bin discretization**: Compute 1/3 and 2/3 quantiles separately per session from all valid aligned values. This mirrors the reference’s per-session standardization and avoids camera/lab calibration differences. Use `np.digitize(..., right=False)`; if tied thresholds collapse classes, flag/drop the session rather than rank-splitting identical physical values.
6. **Trial number in block**: Use zero-based ordinal calculated on the original chronological trial table before trial exclusions, so removing an invalid trial does not renumber later valid trials or invent shorter blocks.
7. **Dataset scope**: Start from the 459 release task sessions (or the provided `DATALIMIT_SUBSET.csv` when present), then data-availability/trial filters determine the final count. Sample mode uses the first two successfully converted sessions, not merely the first two attempted.
8. **Memory/performance**: Bin all trials of a session vectorially using spike index ranges and `bincount`, and pickle protocol 5. Float32 counts require ~4× rather than float64; nevertheless full all-unit data are large, so sessions are processed sequentially and released promptly.

### Planned Sanity Checks
- [ ] Neural: independently use `brainbox.population.decode.get_spike_counts_in_bins` for selected trial intervals and require `np.allclose` to converted counts.
- [ ] Input: independently calculate relative bin times and original-table block run length for selected trials; require `np.allclose`.
- [ ] Output: independently load trials/wheel/motion energy with ONE/SessionLoader, interpolate three selected trials, apply saved thresholds/mappings, and require `np.allclose`.
- [ ] Dataset statistics: reproduce 459/139/699/621,733 pre-filter counts; compare usable sessions to the method paper’s 433.
- [ ] Shape/range checks: all trials `(N,100)`, input `(2,100)`, output `(4,100)`; finite; choice {0,1}; prior/wheel/whisker {0,1,2}; every retained session ≥2 trials.
- [ ] Alignment plots: raw behavior samples plus interpolation grid/labels, neural population count, and event at t=0 for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with `--full` (default), `--sample`, and `--show-processing`. It builds a composite local ONE index, corrects only the in-memory trial revision path, loads trials/wheel/motion energy with `SessionLoader`, loads/merges spike sorting with `SpikeSortingLoader`, maps regions through Beryl, applies documented filters, bins counts, interpolates behaviors, discretizes outputs, validates every trial, records failures/thresholds/session provenance, and writes protocol-5 pickle output. `--show-processing` creates one four-stage plot per converted session for at most two sessions. Python compilation and focused block-number/tertile unit checks pass.

Code inefficiencies identified:
The original reference uses a multiprocessing pool per operation and repeatedly constructs large intermediate arrays. SessionLoader wheel smoothing is intrinsically expensive and produces dense 1-kHz traces. Dense target-format neural matrices dominate memory and output size.

Code speedups added:
Sessions are processed sequentially to bound peak memory; spike searches use sorted arrays and `searchsorted`; per-trial counts use a single flattened `bincount`; only retained trials are binned; arrays are float32/int8; probes are stable-sorted once after merging; objects are garbage-collected between sessions. Further optimization will be guided by the measured Step 7 timing.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2,020 session-neurons |
| Neurons / session | 1,239; 781 (mean 1,010) |
| Subjects | 2 |
| Sessions / subject | 1 each |
| Trials (total) | 644 retained of 1,338 raw |
| Trials / session | 198; 446 |
| Time since onset range | [-0.48, 1.50] internally (verifier rounds to [-0.5, 1.5]) |
| Trial number in block range | [0, 94] |
| Choice distribution | [0.500, 0.500] |
| Prior distribution | [0.443, 0.118, 0.439] |
| Wheel class distribution | [0.333, 0.333, 0.333] |
| Whisker class distribution | [0.333, 0.333, 0.333] |

### Processing Plots Review
Inspected both `processing_<eid>.png` files. Population spike counts and rasters are finite and structured; t=0 stimulus marker lies at the expected location; wheel responses rise after stimulus/movement; interpolated wheel and whisker traces align exactly with their categorical step labels; no discontinuity or one-bin temporal shift is visible. The first two failed attempts exposed and fixed (1) an interval-start versus first-bin-end boundary comparison and (2) double indexing of filtered behavior labels. The successful rerun produced both plots and a valid pickle.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Session-parallel full processing (16 workers) | Reduces compute estimate from ~46 min sequential to ~3–8 min plus serialization |
| Float32 neural / int8 outputs | Approx. half the float64 neural footprint and 1/8 categorical footprint |

| Step | Time / Session | Estimated Total Time |
| Behavior + spikes + binning | 5.95 s/session mean | 45.5 min sequential; approximately 3–8 min with 16 workers depending on storage contention |
| Serialization | sample 228 MB within total 12.4 s | full estimated ~52 GB; several additional minutes |

Formal verifier result: data format valid with no errors or warnings. `sample_data.pkl` is 228 MB; `conversion_sample_out.txt` and `verification_sample_out.txt` exist. Manual inspection confirmed `(neurons,100)`, `(2,100)`, `(4,100)`, float32 neural counts, int8 targets, plausible mean 20-ms counts (0.164–0.166/neuron) and nonzero fractions (0.116–0.140).

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| Choice | 0.6833 | 0.6345 |
| Prior probability of left | 0.8024 | 0.7716 |
| Wheel speed (3 bins) | 0.6113 | 0.5780 |
| Whisker motion energy (3 bins) | 0.5998 | 0.5645 |

Training completed on GPU. Loss decreased monotonically from 1.9039 (epoch 1) to 0.7076 (epoch 200); validation loss was 0.7547. Every validation balanced accuracy exceeded uniform chance (0.5 choice; 0.333 for three-class outputs), with modest train–validation gaps (largest ratio 1.08). The required `train_decoder_sample_out.txt` exists.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 98 GiB (pickle protocol 5)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 621,733 clusters; 75,708 good | `qc=None` (all clusters) | 621,733 clusters | 594,965 session-neurons in retained complete sessions | Match conditional on 18 exclusions |
| Mean neurons/session | Not stated | all clusters, merged probes | 1,354 across 459 candidates | 1,349.13 | Match |
| Subjects | 139 | release inclusion | 139 | 135 | 4 occur only in excluded sessions |
| Sessions | 459 release sessions | complete-modality cache subset | 459 intersection | 441 | 18 justified exclusions |
| Trials (total) | about 295,675 raw (458 tables) | behavioral filters | about 295,675 raw | 186,844 retained | Expected filter reduction |
| Trials/session (mean) | 645 raw | filtered | about 646 raw | 423.68 retained | Expected filter reduction |
| Time input | -0.5 to +1.5 s | same | N/A | reported [-0.5, 1.5]; bin ends -0.48 to 1.50 | Match |
| Trial-in-block input | block ordinal | derived | variable | [0, 98] | Match definition |
| Choice distribution | approximately balanced | left=-1/right=+1 | available | [0.492, 0.508] | Plausible |
| Prior distribution | 0.2/0.5/0.8 | same | available | [0.417, 0.141, 0.442] | Plausible |
| Wheel classes | decoder-specific tertiles | N/A | continuous | [0.333, 0.333, 0.333] | Exact by construction |
| Whisker classes | decoder-specific tertiles | N/A | continuous | [0.333, 0.333, 0.333] | Exact by construction |

The corrected full run took 1,235.8 s (20.6 min), including serialization. Validation reports: `Data format is valid, no errors or warnings.` Every trial has 100 time bins and all categorical ranges are valid.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Validator log**: no errors or warnings after correction. The initial run warned about three all-zero trials; these were source trials outside a valid spike-recording interval and are now removed before tertile fitting.
2. **Independent original-data spot check**: `sanity_checks.py` loaded raw trials, wheel, motion energy, spikes, clusters, and channels through ONE/brainbox (without calling conversion functions). For session `004d8fd5-41e7-4f1b-a45b-0d4ad76fe446`, converted trial 5/raw trial 15, independent time/block input, choice/prior output, interpolated/discretized wheel and whisker outputs, and full neuron-by-time spike counts each passed `np.allclose()`.
3. **Neural sanity**: independently reconstructed half-open 20-ms counts from each probe and concatenated them in probe order; exact `np.allclose` pass. No all-zero trial remains anywhere according to the full validator.
4. **Input sanity**: independently reconstructed bin-end grid and zero-based within-block ordinal; exact `np.allclose` pass.
5. **Output sanity**: independently reconstructed raw choice/prior plus continuous behavior interpolation and stored-threshold classes; exact `np.allclose` pass.
6. **Reference loading comparison**: both use `SpikeSortingLoader.load_spike_sorting`, `merge_clusters`, `SessionLoader`, and ONE. Scientific files are never opened directly.
7. **Reference curation comparison**: reference `load_spiking_data(..., qc=None)` retains all Kilosort clusters; conversion does the same. Trial event, choice, 80-ms/2-s RT, 10-s duration, block-value, and behavior-coverage filters match `ibl_data_utils.py`; initial 0.5-prior trials remain because the requested prior output explicitly includes class 0.5.
8. **Reference alignment/binning comparison**: reference caching sets `stimOn_times`, `(-0.5, 1.5)`, and 0.02-s bins. Conversion matches these and reference `get_behavior_per_interval` bin-end interpolation. Spike bins are half-open to prevent boundary double-counting.
9. **Construction comparison**: time and block number are task-specified decoder inputs; choice/prior and tertile wheel/whisker are task-specified outputs. Session-wise tertiles are an explicit decoder-task transform absent from the reference continuous data.
10. **Statistics/edge cases**: 441/459 sessions retained, 135/139 subjects, 594,965 clusters, mean 1,349.13/session (source mean about 1,354), 186,844 trials. Checked first/last half-open spike edges, behavior timestamp coverage, invalid cluster IDs, missing trial columns, absent cameras, flat continuous streams, and sessions with fewer than two valid trials.

### Issues Found and Resolved
- **Iteration 1 — false session exclusions**: sharing a mutable ONE object across 16 threads caused nondeterministic `NoneType` failures, excluding 94 otherwise valid sessions. Reproducing one failed session sequentially succeeded. Fixed by giving every worker a thread-local composite ONE instance; full conversion and every check were rerun. Sessions increased from 344 to 441 and no `TypeError` failures remain.
- **Iteration 1 — zero-neural periods**: three adjacent trials in one session had population-wide zero counts over two seconds. These were treated as invalid recording periods, removed before behavior thresholds were fit, and all checks rerun. Validator warning is gone.
- **Remaining 18 exclusions (not fixable without fabrication)**: 14 sessions have neither left nor right whisker motion energy; 2 lack `probabilityLeft`; 1 has an all-zero continuous stream whose tertiles collapse; 1 has zero trials passing required filters. Because all four outputs are mandatory, imputing these signals or retaining malformed classes would violate the task.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, 7.7726 at epoch 1 to 1.9404 at epoch 200; test loss 0.8806.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Choice | 0.5763 | 0.5664 | Chance 0.5; modest but within paper's graphical cross-session range |
| Prior probability of left | 0.6054 | 0.5963 | Chance 0.333; strong categorical decoding |
| Wheel speed (3 bins) | 0.5859 | 0.5839 | Chance 0.333 |
| Whisker motion energy (3 bins) | 0.5721 | 0.5717 | Chance 0.333 |

The exact required command completed on GPU and produced `train_decoder_full_out.txt`, `sample_trials.png`, and `predictions.png`. It trained on 149,293 trials and evaluated 37,551 held-out trials. All validation accuracies exceed chance and all train/validation ratios are below 1.02.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from papers |
|----------|-------------------|-------------------------|
| Choice | 0.5664 balanced accuracy (chance 0.5; 1.13x) | Paper graphical session/model values span about 0.51--0.91; method and pooling differ |
| Prior class | 0.5963 (chance 0.333; 1.79x) | Paper reports continuous Pearson r, examples 0.74/0.76 for RRR and 0.37/0.54 for ridge; not directly comparable to 3-class accuracy |
| Wheel-speed class | 0.5839 (chance 0.333; 1.75x) | Paper reports continuous R², not tertile accuracy; unaligned examples reach about 0.75 R² |
| Whisker-energy class | 0.5717 (chance 0.333; 1.72x) | Paper reports continuous R², not tertile accuracy; unaligned examples reach about 0.80 R² |

Choice is above chance but below the requested 1.5x-chance diagnostic threshold. It was investigated rather than dismissed: three specific raw trials (converted/raw indices 5/15, 25/91, 50/185) were loaded independently through ONE and their choice classes exactly matched converted values; the full choice distribution is balanced (0.492/0.508); stimulus alignment and spike counts passed exact source `np.allclose`; processing plots show stimulus at zero with no shift; and neuron/trial filters match the reference. The result is also inside the paper's graphical session/model range. The lower value than the two-session sample (0.6345) is consistent with a single shared model spanning 441 heterogeneous sessions and subjects, not a conversion error. No conversion change was justified.

Train/validation ratios are choice 1.017, prior 1.015, wheel 1.003, and whisker 1.001. There is neither a >1.5x gap nor evidence of leakage/overfitting. All three-class outputs exceed 1.5x chance.

### Issues Found and Resolved
- No new conversion defect was found. The low-choice diagnostic was fully audited as described above; altering alignment, labels, or filtering would contradict exact raw-data checks and the stimulus-aligned task specification.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

`README.md` documents content, schema, loading, reproduction commands, exclusions, and final decoder results. Independent investigation code/output were moved to `cache/` and catalogued in `cache/README_CACHE.md`. All required conversion, validation, training, plotting, and documentation artifacts remain in `/app`.
