# Dataset Conversion Notes

## Overview
- **Dataset**: Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping (provided paper/code/data)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Setup verification:
- `/app/CONVERSION_NOTES.md` created before exploration and existence confirmed.
- Python 3 and imports of numpy and torch succeeded.

Directory contents (top level at initialization):
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `joblib.load` in Figure 1 setup | `demos/georep_hpc_figure1.ipynb` | LOADING | Loads one joblib file per animal into `dat[animal]`; downstream code accesses `position`, `trace`, `maps`, `envs`, and geometry fields. |
| `get_rate_maps` | `src/utils.py:313` | PROCESSING | Builds occupancy and event-rate maps from frame-aligned position and trace at 30 fps; defaults to 15 spatial bins/axis and Gaussian spatial smoothing (sigma 1.5 bins). |
| `fit_decoder` | `src/utils.py:1776` | PROCESSING | Gaussian-smooths traces along time, average-pools trace and behavior in non-overlapping 3-frame bins, one-hot encodes 2-D position, and fits flat-prior Gaussian naive Bayes. |
| `test_decoder` | `src/utils.py:1806` | PROCESSING | Applies the same temporal processing and computes predicted position/error. |
| `decode_position_within` | `src/utils.py:1845` | CURATION / PROCESSING | Within-day 5-fold position decoding; computes smoothed speed, keeps frames >5 cm/s, includes cells with summed moving-frame activity >5, and uses precomputed maps. |
| `clean_rate_maps` / `get_masked_maps` | `src/utils.py:463/482` | PROCESSING | Cleans/masks rate maps according to environment geometry for representational analyses. |

### Notes
- Repository README identifies the files as animal-level Python joblib or MATLAB files. Fields are: `SFPs` (registered-cell footprints by day), `blocked` (blocked partition indices in row-major 3x3 geometry; `-1` means no blocked partition), `centroids`, `envs`, `maps` (`sampling`, `smoothed`, `unsmoothed`), `position` (per-day x-y traces), and `trace` (per-day neural traces).
- Figure 1 calls `decode_position_within(dat[animal]['position'].T, dat[animal]['trace'].T, dat[animal]['maps']['smoothed'])`, establishing direct frame alignment and that stored arrays need transposition into time × features for reference functions.
- Neural data are calcium-derived trace/event activity already provided by the authors, not raw fluorescence. No reference function computes dF/F during analysis; therefore conversion should not recompute dF/F.
- Reference position decoding uses 30 Hz source frames, speed filtering (>5 cm/s after Gaussian smoothing with sigma 5 frames), active-cell filtering (moving-frame trace sum >5 per day), Gaussian temporal trace smoothing and 3-frame average pooling for fitting/testing, 15x15 position maps, and 5-fold within-day CV.
- The new task differs explicitly in requiring 1-minute trials and categorical 3x3 position. Those required changes will be planned later while retaining reference loading/alignment and justified curation where applicable.
- The notebooks and `main.py` are analysis orchestration; `src/utils.py` contains core transformations. No raw trace extraction or registration is performed by this repository because registered, processed traces/maps are distributed.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is 15 GB with 142 files. Top level contains seven extensionless compressed joblib animal files, seven larger MATLAB `.mat` duplicates, a derived `behav_dict`, `.fetch_complete`, and `precomputed_results/` analysis outputs. Primary conversion source will be the compact joblib animal files because these are what the reference notebooks load and they contain all required aligned streams.
- Subjects: `QLAK-CA1-08`, `QLAK-CA1-30`, `QLAK-CA1-50`, `QLAK-CA1-51`, `QLAK-CA1-56`, `QLAK-CA1-74`, `QLAK-CA1-75`.
- Each joblib object is `{subject_id: data_dict}`. Fields are `SFPs`, `blocked`, `centroids`, `envs`, `maps`, `position`, `trace`.
- Dense axis conventions: `trace` = (day, registered-cell union, frame), `position` = (day, 2, frame), `centroids` = (registered-cell union, 2, day), `SFPs` = (35, 35, registered-cell union, day), map arrays = (15, 15, [cell,] day). Position and trace are exactly frame-aligned.
- Trace values are float64 in [0,1], with NaNs for cells not recorded on a day. Per-day finite trace presence exactly equals finite centroid presence in every animal. SFP presence does not always match and is not a suitable inclusion indicator.
- Position is finite and ranges 0–75 cm on both axes. Sampled final frames remain active, so arrays are not zero/NaN tail padded.
- `blocked` is a per-day list of row-major 3x3 blocked indices. Geometry mapping: square `(-1)` (none blocked), o `(4)`, l `(1,2,4,5)`, u `(4,5)`, bit donut `(0,4)`, rectangle `(0,3,6)`, + `(0,2,6,8)`, glenn `(0,8)`, i `(3,5)`, t `(3,5,6,8)`.
- Geometry/day counts are 27 open-square and 20 of each other geometry (207 total). Each of the seven animals has all ten geometries.
- There are no native trials. A day is one continuous ~40-minute recording; required 1-minute trials will be constructed later.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 5,413 within-animal registered identities; 69,744 day-level recordings |
| Neurons / session | mean 336.93; range 113–564 |
| Subjects | 7 |
| Sessions / subject | 31 for six subjects; 21 for QLAK-CA1-51 (207 total) |
| Trials (total) | none natively; continuous sessions |
| Trials / session | none natively |
| Frames (total) | 14,903,019 |
| Frames / session | 71,866–72,219 (~39.9–40.1 min at 30 Hz) |

Per-subject source summary:
| Subject | Sessions | Registered union | Mean cells/session | Cell range | Frames/session |
|---------|----------|------------------|--------------------|------------|----------------|
| QLAK-CA1-08 | 31 | 515 | 213.90 | 153–254 | 71,866 |
| QLAK-CA1-30 | 31 | 875 | 381.03 | 336–422 | 71,866 |
| QLAK-CA1-50 | 31 | 942 | 400.87 | 214–564 | 71,866 |
| QLAK-CA1-51 | 21 | 554 | 230.05 | 113–323 | 72,219 |
| QLAK-CA1-56 | 31 | 862 | 380.87 | 251–529 | 72,091 |
| QLAK-CA1-74 | 31 | 713 | 312.26 | 258–405 | 72,060 |
| QLAK-CA1-75 | 31 | 952 | 405.03 | 263–535 | 72,071 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 unique registered neurons; 69,744 session-level rate maps | “5,413 unique neurons across 207 sessions ... forming 69,744 rate maps” |
| Neurons / session | 69,744 / 207 = 336.93 mean | Derived from the quoted totals |
| Subjects | 7 mice | Seven animal IDs in released data; all source totals reproduce paper totals |
| Sessions / subject | Up to 31; sequence repeats up to 3 times | “same sequence was repeated up to three times”; sequence boundaries 1–11, 11–21, 21–31 |
| Sessions (total) | 207 | Quoted directly |
| Trials (total) | N/A in paper; sessions continuous | “All sessions were 40 min” |
| Trials / session | N/A natively | Required 1-minute trials are a downstream-task modification |
| Neural data time bin | 30 Hz source frames (33.33 ms) | Cellular and behavior streams “simultaneously acquired ... at 30 Hz” |
| Behavior data time bin | 30 Hz source frames (33.33 ms) | Same quote; all frames timestamped for post-hoc alignment |
| Arena | 75 × 75 cm divided into 3 × 3 partitions | Quoted in overview/methods |
| Geometries | 10 | “sequence of 10 geometrically distinct environments” |
| Session duration | 40 min, one/day | Quoted directly |
| Reward rate | N/A | Free exploration; no reward outcome described |
| Spatial map bins | 5 × 5 cm (15 × 15) | Paper rate-map methods |
| Spatial map smoothing | isotropic Gaussian SD 5 cm | Paper rate-map methods |

### Processing Details
- Miniscope calcium video and overhead behavioral video were acquired simultaneously at 30 Hz, with every frame timestamped for post-hoc alignment. Released `trace` and `position` are already frame-aligned.
- Raw calcium videos underwent motion correction, cell segmentation, and transient extraction before release. The transient signal was the derivative of median-subtracted calcium, Gaussian-smoothed with SD 5 frames, noise-normalized using the negative derivative half-normal distribution, and binarized at z > 2.5. The binary rising-phase vector was treated as firing rate for all analyses.
- Position was head location tracked with DeepLabCut. No interpolation or separate resampling is described for the released aligned streams.
- Rate maps use 5 cm × 5 cm bins and Gaussian smoothing SD 5 cm, then are divided into nine 25 cm × 25 cm partitions in row-major order.
- Empirical Bayesian position decoding used spatially and temporally binned position/trace data, 5-fold within-session splits, one-hot positions, a flat-prior Gaussian naive Bayes model (`var_smoothing=1e-9`), and Euclidean distance error on withheld data. Reference code specifies 3-frame temporal bins after Gaussian trace smoothing, 15 bins/axis, speed >5 cm/s, and per-day activity sum >5.
- Environmental sequences start and end with square. Sequence 1 spans sessions 1–11, sequence 2 overlaps at square session 11 and spans 11–21, and sequence 3 spans 21–31.

### Curation Steps

**Neuron curation rules**:
- Motion correction was manually inspected for artifacts.
- Spatial footprints were manually verified to remove lens artifacts.
- Cells were registered across sessions using landmarks, spatial footprints, and/or centroids.
- Released per-day cells are those with finite traces/centroids. For reference decoding, code additionally requires summed activity >5 during moving frames.
- Place-cell significance (split-half map correlation >99th percentile of 1,000 circular position shuffles) is an analysis-specific classification and was not required for population position decoding; therefore it should not be imposed globally.

**Trial curation rules**:
- Native data are continuous 40-minute sessions, not trials.
- Reference decoder selects frames where Gaussian-smoothed speed exceeds 5 cm/s. Whether to preserve this for the downstream time-series format must balance reference consistency against continuous fixed-duration trial requirements and will be resolved in Step 5.
- No rewarded/unrewarded or failed trials exist; mice freely explored.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| 2-D animal position (within session) | Paper evaluates Euclidean distance error on withheld folds but does not report a scalar categorical accuracy comparable to the required 9-class decoder. Figure/code establish successful above-null spatial decoding qualitatively/through error maps. |

The paper contains no directly comparable 3×3 categorical accuracy. The provided downstream neural decoder therefore supplies the relevant chance baseline (1/9 = 11.11%) and empirical validation metric.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset size | Animal joblib loading; registered cells selected per day | 7 animals, 207 days, 5,413 registered identities, 69,744 day-level cells | Exactly 5,413 / 207 / 69,744 | Exact agreement; use all seven primary animal joblib files. |
| Session duration | Assumes 30 fps | 71,866–72,219 frames (2395.5–2407.3 s) | 40 min at 30 Hz | Small −4.5 to +7.3 s variation is expected after timestamp alignment. Use complete non-overlapping 60 s windows and drop only the <60 s remainder so all trials have identical size. |
| Neural representation | Functions consume released `trace` directly | Finite trace values are binary 0/1; NaN marks absent cells | Binary rising-phase events thresholded at z>2.5 and treated as firing rate | Exact agreement; no dF/F or deconvolution is needed. Use finite per-day cells and released binary events. |
| Position alignment | Directly passes `position.T` and `trace.T` | Same frame count and day axis; finite positions | Streams simultaneously acquired at 30 Hz and timestamp-aligned | Exact agreement; preserve common frame indices. |
| Spatial map smoothing | `get_rate_maps` default sigma 1.5 bins; decoder consumes distributed maps | Precomputed 15×15 maps supplied | Methods state 5 cm bins and Gaussian SD 5 cm (1 bin) | This minor prose/default discrepancy concerns precomputed rate maps, not direct trace/position conversion. Do not regenerate maps. Required output bins are direct 25 cm partitions. |
| Decoder curation | `decode_position_within` uses speed >5 cm/s and cell activity sum >5 | Precomputed errors exist for all 207 sessions / 1,035 folds | Decoder is within-session 5-fold; paper does not state speed threshold in prose | Speed selection is specific to fitting the reference pointwise decoder. Deleting low-speed samples would break contiguous fixed-duration time series and bias required location occupancy. Preserve all frames in trials; apply the reference active-cell criterion only if needed after quantifying its impact in Step 5. |
| Decoder outcome | Euclidean error from 15×15 bins | Precomputed fold errors span 5.76–32.46 cm | No categorical accuracy reported | Global precomputed reference error is a consistency benchmark; downstream 9-class chance is 11.11%, not directly comparable. |
| Trial definition | No trial segmentation | Continuous days | 40-minute free exploration sessions | Required one-minute trials are an explicit downstream modification; segment from session start into complete contiguous windows. |

### Final Consistent Understanding
- Each animal file contains already curated, registered, binary CA1 calcium-event traces and aligned DeepLabCut x-y position at 30 Hz. A target session corresponds to one animal-day/environment.
- A day-specific neuron is present exactly when its centroid and trace are finite. The 69,744 paper rate maps equal the sum of these day-specific neurons, so additional place-cell filtering would conflict with the reported population dataset.
- Environment input is the static 3×3 blocked-partition mask from `blocked`; `-1` means no blocked cells. Position output is obtained directly by 25 cm partitioning of 0–75 cm coordinates into row-major class `y_bin*3+x_bin`, clipping the exact 75 cm boundary into bin 2.
- Reference precomputed within-session Gaussian-naive-Bayes results contain 1,035 fold errors (207×5), confirming every source session was analyzable. These are continuous-distance errors rather than the requested categorical metric.
- The reference code temporally smooths/pools three source frames (100 ms) for decoding. This is the most directly applicable common neural/behavior time bin and reduces storage while preserving exact alignment.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day, cell, frame]` | `neural[session][trial]` | Select day-present cells passing moving-frame event sum >5; Gaussian smooth each day along time with sigma 3 source frames; non-overlapping mean pool 3 frames; split into 600-bin trials; transpose to neuron × time; float32 | `decode_position_within`, `fit_decoder`, `test_decoder` | 100 ms bins exactly match reference decoder temporal processing. |
| `blocked[day]` | `input[session][trial]` | Build length-9 float32 binary mask in row-major 3×3 order: 1=blocked, 0=accessible; `-1` gives all zeros | README geometry definition | Static per trial as explicitly required; repeated for each trial without a time axis. |
| `position[day, x/y, frame]` | `output[session][trial]` | Mean pool each coordinate over the same 3 frames; x/y bin = floor(coord/25 cm), clipped to 0..2; class = `y_bin*3+x_bin`; shape 1×600; int64 | `fit_decoder` behavior pooling; paper 3×3 partitions | Row-major labels: NW/N/NE, W/center/E, SW/S/SE according to source x west→east and y north→south map convention. Exact coordinate 75 is clipped into final bin. |
| animal filename | `subjects`, `subject_idx` | Sorted seven IDs; one target session per animal-day | Figure 1 loading | 207 sessions total. |
| CA1 recordings | `brain_regions`, `brain_region_idx` | `brain_regions=['CA1']`; zeros for each retained neuron | Paper title/methods | All recordings are hippocampal CA1. |
| day/environment metadata | `metadata['session_info']` | Subject, source day, environment, blocked indices, source/used frame counts, retained neuron count, trial count | README/paper | Enables source traceability and edge-case review. |

### Names and categorical values
- `input_names`: `blocked_NW`, `blocked_N`, `blocked_NE`, `blocked_W`, `blocked_center`, `blocked_E`, `blocked_SW`, `blocked_S`, `blocked_SE`.
- `output_names`: `position_bin`.
- `output_values[0]`: `NW`, `N`, `NE`, `W`, `center`, `E`, `SW`, `S`, `SE`, corresponding exactly to integer classes 0–8.

### Key Decisions
1. **Session and trial definition**: Each animal-day is one target session. Split pooled data from session start into complete contiguous 60-second trials (600 samples at 100 ms). This yields 8,187 trials: 39/day for subjects 08/30/50, 40/day for subjects 51/56/74/75. Exclude only incomplete tails; never pad or wrap.
2. **Temporal processing**: Match reference `fit_decoder` by applying `scipy.ndimage.gaussian_filter1d(..., sigma=3, axis=time)` to neural traces and averaging non-overlapping 3-frame groups. Pool x/y positions over identical groups before discretization. Smooth across the continuous session before splitting to avoid artificial minute-boundary edge effects.
3. **Cell curation**: Start from finite day-specific cells (the paper-curated population), then match `decode_position_within`: compute 30-Hz displacement speed, Gaussian smooth with sigma 5 frames, select speed >5 cm/s, and retain cells with >5 binary events in those moving frames. Planning data show 68,862 retained of 69,744 day-level recordings (98.74%); all sessions remain far above usable neuron counts. Do not apply place-cell-only filtering because reference position decoding uses the population.
4. **Timepoint curation**: Preserve all timepoints, including immobility, because deleting frames would violate contiguous one-minute trial timing and bias the requested location distribution. The speed mask is used only for reference cell-quality curation.
5. **Output discretization**: Use physical fixed boundaries 0,25,50,75 cm from the experimental 3×3 geometry, not per-session min/max. Pool continuous position before categorization, matching reference temporal behavior pooling.
6. **Input representation**: Geometry is a static 9-vector per trial, exactly as requested. A binary blocked mask is more explicit and decoder-compatible than variable-length blocked-index lists or environment-name categories.
7. **Dtypes/storage**: Neural and inputs float32, outputs int64, indices int64. Avoid retaining source arrays after each animal and use vectorized smoothing/pooling. The full pickle is expected to be several GB because it preserves 207 sessions, ~69k session-cell recordings, and 100-ms bins.
8. **Missing/edge data**: Position is fully finite. NaN traces occur only for absent registered cells and are removed before smoothing. Coordinate values are clipped after division solely to handle exact 75-cm endpoints. Every source session yields at least 39 trials.

### Planned Sanity Checks
- [ ] Structural: exactly 7 subjects, 207 sessions, 8,187 trials; matching trial counts across neural/input/output; every trial has 600 neural/output time bins.
- [ ] Neural source spot-check: independently load raw joblib and reproduce selected neurons plus Gaussian smoothing/3-frame pooling; compare selected trial elements with `np.allclose`.
- [ ] Input source spot-check: independently expand raw `blocked` indices and compare converted masks with `np.allclose`; verify mask sums equal blocked-index counts and square is all zero.
- [ ] Output source spot-check: independently average raw position triples, apply fixed 25-cm boundaries, and compare converted labels with `np.allclose`; verify all classes are integers 0–8.
- [ ] Alignment: for plotted samples, overlay pooled x/y trajectory colored by output class and 3×3/blocked boundaries; confirm neural/output trial starts use identical source frames.
- [ ] Statistics: retained session-neuron sum expected 68,862; full class counts before any conversion bug expected approximately `[489566,483825,663743,370439,281301,377493,568320,693665,983848]` when all planned trials are included.
- [ ] Geometry: 27 square sessions and 20 per other geometry; labels in blocked bins should be rare and investigated if substantial.
- [ ] Values: neural finite/nonnegative after smoothing; inputs binary; output class fractions expected 5.73%–20.03%, with no missing classes.
- [ ] Reference consistency: source finite cell sum 69,744 and registered union 5,413 remain reproducible; reference cell filter removes only the documented 882 day-cell recordings.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` was created and passes `python3 -m py_compile`. It supports the required invocation and mutually exclusive `--full` (default) / `--sample`, plus `--show-processing`.

Implementation notes:
- Loads primary animal joblib files in deterministic subject/day order and releases each animal before loading the next.
- Implements reference speed/activity cell curation, Gaussian temporal smoothing, aligned 3-frame pooling, fixed 25-cm position discretization, complete-minute splitting, geometry-mask construction, metadata, and structural assertions.
- Processing plots show (1) pooled x/y plus class over time, (2) processed neural activity, and (3) trajectory over 3×3 boundaries with blocked bins shaded.
- Uses atomic pickle writes (`.tmp` then rename) to avoid leaving a corrupt target if interrupted.
- Prints animal load time, per-session conversion time, aggregate cell/trial/class statistics, output size, and save/total times.

Code inefficiencies identified:
- Source joblib files are compressed and must each be decompressed once.
- The target list-of-trial-arrays format necessarily copies each 60-second trial.
- Class-count accumulation concatenates only small output arrays and is negligible compared with neural processing.

Code speedups added:
- Vectorized Gaussian filtering, reshape-based pooling, and label construction replace per-frame loops.
- Float32 neural storage halves memory/disk relative to source float64.
- One-animal-at-a-time loading bounds source memory while retaining converted target arrays.
- Smoothing is performed once per continuous session before slicing all trials.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 335 retained day-level recordings (338 before reference activity filter) |
| Neurons / session | mean 167.5; range 153–182 |
| Subjects represented | 1 of 7 (`QLAK-CA1-08`) |
| Sessions | 2 |
| Trials (total) | 78 |
| Trials / session | 39 |
| Time bins / trial | 600 at 100 ms (60 s) |
| Neural range | finite, nonnegative smoothed values; float32 |
| Input range | [0,1], static length-9 masks |
| Output distribution | `[0.1035,0.0616,0.1349,0.1023,0.0438,0.1038,0.1037,0.1172,0.2292]` |

### Format Validation
- `/app/sample_data.pkl`, `/app/conversion_sample_out.txt`, and `/app/verification_sample_out.txt` were created.
- `train_decoder.py --verify-only` completed with no errors or warnings and reported “Data verification complete.”
- Independent inspection confirmed required keys, aligned trial-list lengths, `(neurons,600)` neural arrays, `(9,)` static inputs, `(1,600)` int64 outputs, matching brain-region indices, and complete metadata/session_info.
- Both sessions contain all nine output classes; geometry masks correctly represent square (all zero) and `o` (center blocked).

### Processing Plots Review
- Two nonblank PNGs were generated, one per sample session, with aligned position/class traces, processed neural heatmaps, 3×3 trajectories, and blocked geometry overlays.
- Position boundaries and categorical colors coincide; trial duration is exactly 60 s; no temporal offset, clipping artifact, or empty neural panel was observed.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Vectorized session smoothing/pooling and trial slicing | Per-session processing ~0.3 s in sample |
| One compressed animal load, then all days | Avoids repeated ~9–18 s decompression |
| Float32 neural output | Approximately halves output size and serialization cost |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Source loading amortized | ~0.3–0.6 s | ~100 s for seven animal files |
| Processing | ~0.3 s for 153–182 neurons; scales with neuron count | ~2–4 min |
| Serialization | sample 0.02 s; full projected ~6 GiB | likely <2 min |
| Total full conversion | sample total 10.81 s including initial load/plots | conservatively <10 min, below 15-min optimization threshold |

The projected full size is roughly 6 GiB by scaling the 0.030-GiB sample by retained neuron-time, which is feasible with available memory/disk.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- `train_decoder.py` finished successfully.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-----------------------|-------------------------|
| position_bin | 0.4415 | 0.3663 |

- Chance is 1/9 = 0.1111; validation accuracy is 3.30× chance.
- Training loss decreased monotonically from 2.276995 at epoch 1 to 1.718352 at epoch 200; test loss was 1.803432.
- Training/validation accuracy ratio is 1.21, below the 1.5 overfitting concern threshold.
- The strong above-chance result on two sessions supports correct neural-position temporal alignment and output encoding.
- Output log: `/app/train_decoder_sample_out.txt`.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6.127 GiB
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; validation completed successfully with no errors found
- Full conversion runtime: 219.18 s (3.65 min), including 5.33 s serialization

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Unique registered neurons | 5,413 | Animal-level registered unions | 5,413 | Source provenance retained; target uses day-specific cells | Yes |
| Session-level neurons | 69,744 rate maps | finite day trace then activity >5 for decoder | 69,744 present; 68,862 pass moving-event criterion | 68,862 | Yes, exact planned reference decoder curation |
| Mean retained neurons/session | N/A after decoder curation | activity-filtered | 332.67 | 332.67 | Yes |
| Subjects | 7 | seven animal files | 7 | 7 | Yes |
| Sessions | 207 | every animal-day | 207 | 207 | Yes |
| Trials (total) | N/A | N/A | 8,187 complete minutes possible | 8,187 | Yes |
| Trials/session | N/A | N/A | 39 or 40 | 39 or 40 | Yes |
| Time bins/trial | N/A | 3-frame pooling | 600 at 100 ms for 60 s | 600 | Yes |
| Environment sessions | 10 geometries | source `envs`/`blocked` | square 27; every other geometry 20 | same | Yes |
| Input range | blocked geometry | indices -1 or 0..8 | binary expansion [0,1] | [0,1] | Yes |
| Position range/classes | 3×3 experimental partitions | one-hot spatial bins | classes 0..8 | classes 0..8 | Yes |
| Output distribution | not reported | N/A | `[.099663,.098495,.135121,.075412,.057266,.076848,.115696,.141213,.200287]` | exact same | Yes |
| Brain region | CA1 | CA1 data | CA1 | CA1 | Yes |

### Integrity Review
- Independent first/middle/last session spot checks (sessions 0, 103, 206) confirmed metadata, finite values, dtypes, neuron counts, trial counts, and exact shapes.
- Subject session counts are 31 for six animals and 21 for QLAK-CA1-51, matching source sequences.
- All nine output classes are represented with fractions 5.73%–20.03%; no extreme class collapse exists.
- Tail handling is explicit in `session_info`: only incomplete final minute is discarded, with no padding. Source and used frame counts are retained.
- Full verification reported 68,862 CA1 neurons and “Data verification complete.”

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Searched `/app/verification_full_out.txt` case-insensitively for error, warning, invalid, failed, and traceback. No issues were reported; the log ends with `Data verification complete.`
2. **Independent raw-source sanity checks**: `/app/sanity_checks_out.txt` records three independently implemented comparisons that load raw joblib directly and do not import conversion code. Sessions/trials `(0,5)`, `(103,17)`, and `(206,39)` span first/middle/last subjects and both 39- and 40-trial sessions. Neural arrays matched reference cell selection + smoothing + pooling with `np.allclose` and maximum absolute error 0.0. Expanded blocked masks and discretized outputs also matched with exact `np.allclose`.
3. **Neural sanity**: Every trial is finite and nonnegative float32 with fractional values after reference smoothing/pooling (not merely binary). Brain-region index lengths equal retained neuron counts. Retained counts sum to 68,862; no session is empty.
4. **Input sanity**: Every input is a binary static length-9 vector. Geometry-mask frequencies reproduce 27 square and 20 of each of the other nine designs. Mask sums match raw blocked-index counts; `-1` maps to all zeros.
5. **Output sanity**: Every output is int64 `(1,600)` with classes 0–8. Exact raw pooled positions and fixed-boundary labels matched all spot checks. All classes occur globally and none exceeds 20.03%.
6. **Loading comparison**: Reference Figure 1 uses `joblib.load` per animal and accesses `trace`, `position`, `blocked`, `envs`, and `maps`. `animal_files`/`convert` load the same seven primary joblib files and same raw fields; precomputed maps are intentionally not regenerated because direct aligned streams define the requested decoder data.
7. **Neuron/trial filtering comparison**: Reference `decode_position_within` (utils.py:1845) identifies finite day cells and retains activity sum >5 during Gaussian-smoothed speed >5 cm/s. `select_cells` implements the same. Conversion preserves all contiguous frames because deleting immobility would violate fixed-duration one-minute trials; this explicit downstream difference is documented.
8. **Temporal alignment comparison**: Reference passes trace and behavior from identical frames. `process_session` pools identical 3-frame slices and raw-source checks prove exact trial alignment. There is no interpolation, shift, or independent truncation.
9. **Binning comparison**: Reference `fit_decoder`/`test_decoder` (utils.py:1776/1806) apply Gaussian trace smoothing sigma 3 and non-overlapping 3-frame average pooling. Conversion matches this before splitting. Position is averaged in the same bins, then changed from reference 15×15 to required experimental 3×3 categories.
10. **Input construction comparison**: Reference README defines row-major blocked indices; conversion expands them to the task-required static binary geometry vector. Raw checks passed exactly.
11. **Output construction comparison**: Paper defines nine 25×25 cm partitions of the 75×75 cm arena. Conversion uses fixed physical boundaries and row-major `y*3+x`, clipping only exact upper endpoints. This differs from reference decoder's 15×15 one-hot solely because the requested output is 3×3.
12. **Key statistics**: Reproduced every available reference statistic: 7 subjects, 207 sessions, 5,413 registered identities, 69,744 day-cell records before decoder filtering, 10 geometries, 30 Hz source, ~40 min/session, CA1 region, and 0–75 cm coordinates. Converted counts and output fractions match the independent planning pass exactly.
13. **Edge cases**: Exhaustively asserted at least two trials/session; equal neural/input/output trial lists; exactly 600 bins/trial; remainder in `[0,1799]`; source = used + remainder; consistent neurons across trials; correct dtypes/ranges. Sessions with 71,866 frames produce 39 trials; sessions ≥72,000 produce 40; no off-by-one, padding, or wrapping occurs.
14. **Blocked-bin occupancy check**: Quantified labels falling in blocked partitions. A small nonzero fraction is expected from head tracking near/over 25-cm partition boundaries and three-frame averaging; geometry masks and raw labels are independently exact, so this is not a conversion misalignment.

### Issues Found and Resolved
- **Initial generic exploration failed on ragged blocked lists**: Replaced coercion with ragged-aware inspection; no conversion impact.
- **Nominal 40-min frame counts vary**: Resolved by retaining complete 60-s windows and recording every discarded remainder; exhaustive arithmetic checks pass.
- **Paper map smoothing prose vs utility default**: Irrelevant to direct trace/position conversion; distributed maps are not regenerated.
- **Reference speed frame deletion vs fixed trials**: Used speed only for identical cell-quality selection, not timepoint deletion. This preserves required continuous timing while matching applicable curation.
- **No validation warnings**: None require justification or suppression.

### Iteration Result
No conversion mismatch was found. All independent raw comparisons, exhaustive structural checks, reference-code comparisons, and key-statistics checks passed; therefore no reconversion was necessary.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes; 2.307310 at epoch 1 to 1.156622 at epoch 200.
- Test loss: 1.137490.
- Training completed on CUDA with 6,531 training trials and 1,656 validation trials.
- `train_decoder.py finished successfully.`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-----------------------|-------------------------|-------|
| position_bin | 0.6973 | 0.6081 | Chance 0.1111; validation is 5.47× chance |

- Training/validation ratio is 1.15, below the 1.5 concern threshold.
- Loss decreased smoothly throughout all 200 epochs with no instability.
- Full output: `/app/train_decoder_full_out.txt`.
- Sample plots requested by `--plot-samples` were generated and retained.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Accuracy | Chance | Expectation from Paper |
|----------|-------------------|--------|------------------------|
| position_bin (3×3) | train 0.6973; validation 0.6081 | 0.1111 | No 3×3 categorical accuracy reported. Paper/reference evaluates 15×15 position by Euclidean error; released results have mean 13.49 cm, median 12.38 cm. |

#### Check 1: Accuracy vs chance
- Validation balanced accuracy is 5.4729× chance, far above both the 1.0× bug criterion and 1.5× investigation criterion.
- Training balanced accuracy is 6.2757× chance.
- All nine classes are represented and balanced accuracy prevents the most frequent SE class from inflating the metric.

#### Check 2: Accuracy comparison to paper
- Every empirical decoding metric found in the paper was reviewed. The paper describes 5-fold Gaussian-naive-Bayes position decoding and Euclidean withheld-position error, but does not state a categorical accuracy.
- The distributed `within_decoding` results provide 1,035 fold errors: mean 13.49 cm, median 12.38 cm, range 5.76–32.46 cm. This is not mathematically interchangeable with 3×3 accuracy because errors within or across 25-cm boundaries map differently.
- The achieved 60.81% balanced 9-class accuracy is qualitatively consistent with useful spatial information and substantially above chance. No lower directly comparable paper value exists.

#### Check 3: Train vs validation gap
- Train/validation ratio is 1.1467, below the specified 1.5 threshold.
- Absolute gap is 0.0892; validation is 87.2% of training accuracy. This is not concerning given 207 heterogeneous animal-day sessions.
- Test loss (1.1375) is slightly below final training loss (1.1566), providing no sign of severe overfitting.

#### Required debugging checks
1. **Three raw trials**: Sessions/trials `(0,5)`, `(103,17)`, `(206,39)` were independently regenerated from original files. Outputs matched exactly; neural arrays matched with maximum error 0.
2. **Temporal alignment**: Processing plots overlay pooled x/y and labels on a shared time axis, and independent raw checks use identical frame slices for neural and position. Sample and full decoder accuracy further support alignment.
3. **Output variation**: Classes range from 5.73% to 20.03%; no class is near 99%, and all classes occur in both sample sessions and globally.
4. **Neural filtering**: `select_cells` reproduces reference speed smoothing, >5 cm/s moving mask, and >5 event threshold. Raw retained neuron counts match exactly.
5. **Reference processing**: Gaussian neural smoothing sigma 3 and 3-frame mean pooling match `fit_decoder`/`test_decoder`; position pooling uses identical bins; only the explicitly required 3×3 output replaces 15×15 reference bins.

### Issues Found and Resolved
- No low-accuracy, overfitting, class-collapse, alignment, filtering, or processing issue was found.
- Paper and downstream metrics are noncommensurate; this is documented rather than making an invalid numeric equality claim.
- No conversion iteration was required. Re-running all Step 10 checks remained unnecessary because Step 12 found no defect.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading instructions, format, processing summary, key statistics, and decoder result.
- [x] `cache/` created and investigation scripts/extracted paper text moved there.
- [x] `cache/README_CACHE.md` documents every cached investigation file.
- [x] `CONVERSION_NOTES.md` completed with decisions, rationale, source/code/paper comparisons, sanity checks, validation results, and accuracy reviews.
- [x] All required conversion, validation, and decoder logs exist.
- [x] Required sample/full pickle files exist and pass validation.
- [x] Top-level bytecode cache removed; user-facing deliverables, required logs, and diagnostic plots retained.

### Final Deliverables
- Full dataset: `/app/converted_data.pkl` (6.127 GiB)
- Reproducible script: `/app/convert_data.py`
- User documentation: `/app/README.md`
- Detailed audit trail: `/app/CONVERSION_NOTES.md`
- Full validation: no errors or warnings
- Full decoder validation balanced accuracy: 0.6081 (chance 0.1111)

