# Dataset Conversion Notes

## Overview
- **Dataset**: A flexible hippocampal population code for experience relative to reward
- **Date started**: 2026-03-11
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
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
| `create_sess` | `code/src/reward_relative/preprocessing.py` | LOADING | Creates a `sess` object from scanbox/VR/session metadata and calls alignment + behavior loading. |
| `append_session_data` | `code/src/reward_relative/preprocessing.py` | LOADING | Adds aligned VR data, Suite2p outputs, and behavior timeseries/trial matrices to `sess`. |
| `dff` | `code/src/reward_relative/preprocessing.py` | PROCESSING | Computes trial-masked dF/F from fluorescence, with neuropil subtraction and optional deconvolution to `events`. |
| `multi_anim_sess` | `code/src/reward_relative/utilities.py` | PROCESSING | Loads per-session `sess` pickles, computes dF/F and `events`, reward zones, trial subsets, and place-cell outputs across animals. |
| `get_trial_types` | `code/src/reward_relative/behavior.py` | PROCESSING | Derives per-trial reward outcome and environment morph from aligned VR data. |
| `get_reward_zones` | `code/src/reward_relative/behavior.py` | PROCESSING | Assigns reward-zone coordinates and labels (`A/B/C`) from scene metadata, including switch sessions. |
| `define_trial_subsets` | `code/src/reward_relative/behavior.py` | PROCESSING | Splits trials into pre/post switch sets or halves for single-zone sessions. |
| `get_reward_inds` / `get_reward_times` / `get_omission_inds` / `get_omission_trials` | `code/src/reward_relative/rewardAnalysis.py` | PROCESSING | Finds reward-delivery and omission-related frame indices using aligned behavioral signals. |
| `get_timeseries_data` | `code/src/reward_relative/glmUtils.py` | PROCESSING | Extracts continuously sampled neural `events` and aligned behavior variables (`rel_pos`, `pos`, `trials`, `speed`, `rewards`) after masking invalid / low-speed / bad-lick samples. |
| `CircularRegression` / `train_vs_test_blocks` | `code/src/reward_relative/decode.py` | PROCESSING | Decoder used in the paper for circular reward-relative position from deconvolved activity. |

### Notes
- Reference data flow from code:
  1. Raw imaging + VR -> `sess` via `create_sess` / `append_session_data`.
  2. `sess.align_VR_to_2P()` aligns Unity VR to imaging frame times.
  3. Behavior timeseries added directly from aligned VR (`licks`, `rewards`, `speed`).
  4. dF/F is not stored in the raw `sess`; it is computed later by `multi_anim_sess` using `preprocessing.dff`.
  5. Deconvolved calcium `events` are derived from dF/F and used for the decoder notebook.
- The code is for 2-photon calcium imaging, not electrophysiology.
- Delta F over F does need to be computed in the original analysis pipeline.
- Cell curation is not an ephys spike-quality filter. Instead, ROIs are manually curated in the Suite2p GUI, which updates `iscell.npy`; `sess.load_suite2p_data()` loads that curation.
- `multi_anim_sess_README.md` states each sample is one imaging frame at about 15.5 Hz (~64.5 ms/sample), and `trial_matrices` are spatial bins of 10 cm across 0-450 cm.
- Decoder-relevant continuous variables in reference code:
  - neural: `sess.timeseries['events']`
  - position: `sess.vr_data['pos']`
  - reward-relative position: computed from reward-zone start position per trial
  - trial id: repeated sample-wise from trial index
  - speed: `sess.timeseries['speed']`
  - rewards: `sess.timeseries['rewards']`
  - licks: `sess.timeseries['licks']`, with values >1 clipped to 1 and invalid sensors masked
- Important masking/alignment details from `get_timeseries_data`:
  - Only samples from `trial_start_inds` to `teleport_inds` are kept.
  - Trial slices use `start-1:stop-1`, so reference indexing is effectively 1-based for stored trial boundary indices.
  - Samples with NaNs in `events`, low speed (`<2 cm/s` by default), or invalid lick samples are removed for the paper’s GLM/decoder analyses.
  - Reward and omission signals are represented as state variables that switch after reward delivery or reward-zone entry, respectively, within a trial.
- The paper’s Figure 3 decoder notebook decodes circular reward-relative position from deconvolved events, and down-samples observations to match position occupancy across trial sets before training.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Top level of `data/`:
  - `dandiset.yaml`
  - one folder per subject: `sub-m3`, `sub-m4`, `sub-m7`, `sub-m11`, `sub-m12`, `sub-m13`, `sub-m14`, `sub-m15`, `sub-m17`, `sub-m18`, `sub-m19`
- Each subject folder contains one NWB file per session, named like `sub-m3_ses-01_behavior+ophys.nwb`.
- Total files in `data/`: 152 NWB session files.
- Native format: NWB/HDF5, with:
  - `processing/behavior/BehavioralTimeSeries`
  - `processing/ophys/{Fluorescence, Neuropil, Deconvolved, ImageSegmentation, Backgrounds_0}`
  - `acquisition/TwoPhotonSeries`
- Representative session structure (`sub-m3_ses-01`):
  - Behavior time series: `Reward`, `autoreward`, `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, `trial_start`
  - Ophys interfaces:
    - `Fluorescence/plane0/data`: frames x ROIs raw fluorescence
    - `Neuropil/plane0/data`: frames x ROIs neuropil fluorescence
    - `Deconvolved/plane0/data`: frames x ROIs deconvolved activity
    - `ImageSegmentation/PlaneSegmentation`: ROI table with `pixel_mask`, `iscell`, `planeIdx`
- Behavioral streams use explicit timestamps; ophys series use `starting_time` with frame rate attribute.
- Imaging plane metadata indicates brain region `hippocampus, CA1`.
- Important native encodings observed:
  - `environment`: values include `0`, `1` and invalid pre-sync `-1`
  - `trial number`: includes invalid pre-sync `-1`
  - `position`: includes invalid pre-sync `-500`
  - `reward_zone`: integer-coded values `0..8` observed in sample files
  - `iscell`: shape `(n_rois, 2)` in sample files, consistent with Suite2p export `[is_cell_flag, probability]`
- Example session (`sub-m3_ses-01`):
  - 33,913 frames
  - 2,929 total ROIs before `iscell` filtering
  - frame rate 15.5078125 Hz
  - behavior timestamps span 0 to 2186.77 s

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 138,678 curated ROIs (`iscell[:,0] == 1`); 312,110 total ROIs before curation |
| Neurons / session | mean 912.36 curated (min 155, max 2341) |
| Subjects | 11 |
| Sessions / subject | 12 for `sub-m11`, 14 for all others |
| Trials (total) | 12,217 |
| Trials / session | mean 80.38 (min 41, max 100) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Not stated directly for switch cohort; NWB should reflect curated switch-task cells | Paper gives per-session range rather than dataset total |
| Neurons / session | 155-2172 putative pyramidal neurons/session | “This approach yielded 155–2172 putative pyramidal neurons per session” |
| Subjects | 11 switch-task mice in main cohort | “Each mouse encountered a different starting reward zone… (n = 11 mice)” |
| Sessions / subject | 14 planned imaging days; m11 starts on day 3 | “task… for a total of 14 days”; “except m11, for whom imaging started on day 3” |
| Trials (total) | 12,376 imaged trials across 11 switch mice (from lick-QC context) | “81 out of 12,376 trials removed across 11 switch mice” |
| Trials / session | 80-100 targeted; mean 80.5 +/- 7.4 across all imaging days in 14 mice | “We targeted 80–100 trials per session… (mean ± s.d., 80.5 ± 7.4 trials…)” |
| Neural data time bin | ~0.0645 s per frame (~15.5 Hz) | “sampled at ~15.5 Hz”; “0.0645 s imaging frame samples” |
| Behavior data time bin | Same as neural, sampled at imaging frame rate | “All behavioral and neural time series were sampled at ~15.5 Hz” |
| Reward rate | ~85% rewarded, ~15% omitted | “Reward was randomly omitted on approximately 15% of trials” |
| Track length | 450 cm | “450 cm linear track” |
| Reward zone size / locations | 50 cm; A=80-130, B=200-250, C=320-370 cm | “zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm” |
| Switch timing | After 30 trials on switch days | “Each switch occurred after 30 trials” |
| Spatial binning | 45 bins of 10 cm | “binned the 450 cm linear track into 45 bins of 10 cm each” |
| Interneuron exclusion | 0.42 +/- 0.85% of cells excluded | “excluding 0.42 ± 0.85% of cells” |
| Lick-error trial exclusion | ~0.65% of imaged trials; 81/12,376 | “~0.65% of all imaged trials, n = 81 out of 12,376” |
| RR-cell fraction | 16.3 +/- 5.3% of place cells over switch days | “16.3 ± 5.3% of place cells averaged over all switch days” |
| Decoder result | RR cells decode RR position above shuffle after switch; z-scored decode >2 from about -104.5 to +152.7 cm | “only the RR population decoded… better than shuffle… range of z-scored decode >2: -104.5 ± 20.1 cm to +152.7 ± 22.9 cm” |


### Processing Details
- Task structure:
  - 450 cm linear VR track with hidden 50 cm reward zone.
  - Reward zones A/B/C are fixed track spans: A `80-130`, B `200-250`, C `320-370` cm.
  - On switch days, reward zone changes after 30 trials.
  - ENV2 introduced on day 8; two mice (`m17`, `m18`) started in ENV2.
- Temporal alignment and sampling:
  - Neural and behavioral data are analyzed at the imaging frame rate (~15.5 Hz, ~64.5 ms).
  - Reference code aligns Unity VR data to imaging frames before any later analyses.
  - For many analyses, only in-trial samples are used; teleport period treatment depends on session/imaging settings.
- Neural processing:
  - Fluorescence is motion-corrected and segmented with Suite2P.
  - dF/F is computed independently within each trial using a maximin baseline with a 20 s window.
  - dF/F is smoothed with a 2-sample Gaussian.
  - Deconvolved `events` are extracted with OASIS and are used for decoder and place-cell analyses.
- Behavioral processing:
  - Activity at running speeds `<2 cm/s` is excluded from spatial neural analyses and RR decoding.
  - Licks are binary after correction; corrupted lick trials are set to NaN and excluded from licking analyses.
  - Reward outcome is trial-specific with random omissions on ~15% of trials.
- Decoder-specific details:
  - Target in the paper is circular reward-relative position (`-pi` to `pi`), aligned to reward-zone start.
  - Tenfold CV, 90% train / 10% test.
  - Occupancy downsampling matches each RR position bin (`2*pi/45`, about 10 cm).
  - Shuffles independently circularly shift each neuron’s session timeseries.

### Curation Steps

**Neuron curation rules**:
- Manual Suite2P curation removes ROIs that:
  - contain multiple somata or dendrites
  - lack visually obvious transients
  - appear overexpressed
  - show high continuous fluorescence fluctuation typical of putative interneurons
- Additional putative interneurons are excluded if corr(dF/F, running speed) > 0.5.
- Multi-plane mice pool planes for most analyses.

**Trial curation rules**:
- Random reward omissions are part of the task and are not discarded by default.
- Lick-analysis trials are removed when >30% of imaging-frame samples in the trial have cumulative lick count >2.
- Many analyses exclude samples with running speed <2 cm/s rather than dropping entire trials.
- Some analyses use:
  - pre-switch 30 trials
  - final 30 post-switch trials
  - first vs second half of trials on non-switch days

### Decoders Trained
| Decoded variable | Accuracy |
| Reward-relative position (paper decoder) | No single scalar value stated in text; qualitatively above shuffle for RR cells after switch, with z-scored decode >2 spanning about -104.5 to +152.7 cm around reward |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session container | Code operates on a `sess` object produced earlier, then adds `dff` and `events` | NWB files already store aligned behavior plus `Fluorescence`, `Neuropil`, and `Deconvolved` | Paper describes the same pipeline: align VR to imaging, compute dF/F, deconvolve | Treat NWB as an exported `sess`-equivalent. Use NWB aligned behavior streams and map `Deconvolved/plane0/data` to reference `events` unless a later check shows mismatch. |
| Reward-zone labeling | Code infers reward-zone coordinates from scene names like `Env2_LocationA_to_B` | NWB `reward_zone` samples are not A/B/C labels; values increase within-zone and are only meaningful as `>0` occupancy | Paper defines zone A/B/C by track coordinates and describes scene-based switches | Parse scene/location from NWB `identifier`; use `reward_zone > 0` only as occupancy / entry signal, and derive A/B/C labels from scene metadata for per-trial zone identity. |
| Trial timing | Code uses `trial_start_inds` / `teleport_inds` and slices `start-1:stop-1` | NWB stores frame-wise `trial_start`, `teleport`, `trial number`, timestamps | Paper defines trials as laps ending in teleport period | Reconstruct trial start/end frame indices from NWB behavior streams so trial segmentation matches the reference semantics. |
| Environment identity | Code uses `morph` from VR data, with `Env1 -> 0`, `Env2 -> 1` | NWB `environment` contains valid values `0/1` plus pre-sync invalid `-1` | Paper describes two environments ENV1 / ENV2 | Use valid `environment` values directly from NWB after excluding invalid pre-sync samples, and cross-check against scene identifier. |
| ROI curation count | Code/paper report 155-2172 putative pyramidal neurons/session after curation | NWB `iscell[:,0]==1` gives 155-2341 curated ROIs/session; max session is multi-plane `m18 ses03` | Paper states 155-2172/session and excludes additional putative interneurons by speed correlation | Likely requires additional reference-style interneuron exclusion and possibly careful handling of multi-plane pooled ROIs. This becomes a planned consistency check for conversion. |
| dF/F availability | Code computes dF/F from `F`/`Fneu` with per-trial maximin baseline | NWB exports `Fluorescence`, `Neuropil`, and `Deconvolved`, but not stored `dff` | Paper explicitly defines dF/F computation and deconvolution | For decoder inputs we mainly need neural activity. Candidate paths are: (1) use NWB `Deconvolved` if it matches reference `events`; (2) recompute dF/F / deconvolution from `Fluorescence` and `Neuropil` if necessary. Need later sanity check against raw NWB signals and paper stats. |
| Total session count | Task schedule implies 14 days per mouse | NWB has 152 sessions across 11 mice | Paper notes m11 started imaging on day 3 | This matches: 11*14 = 154 planned switch sessions, minus 2 missing early m11 sessions = 152. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/Deconvolved/plane0/data` | `neural` | Select curated ROIs (`iscell[:,0] == 1`), transpose to `(neurons, time)` per trial, slice `trial_start:teleport` | `glmUtils.get_timeseries_data`, `utilities.multi_anim_sess` | Candidate equivalent of reference `sess.timeseries['events']`; later sanity-check against raw NWB. |
| `processing/behavior/BehavioralTimeSeries/position` timestamps | `input[0]` = `time_from_trial_start_s` | `timestamps - timestamps[trial_start]` within each trial | Reference alignment is imaging-frame synchronized behavior | Continuous, time-varying, aligned to trial start as required by decoder task. |
| `processing/behavior/BehavioralTimeSeries/environment` | `input[1]` = `environment` | Per-trial constant repeated across frames; valid values only (`0/1`) | `behavior.get_trial_types` (`morph`) | Cross-check against scene identifier; encode `ENV1=0`, `ENV2=1`. |
| Reconstructed within-session trial index | `input[2]` = `trial_number` | 0-based per-trial index repeated across frames | `glmUtils.get_timeseries_data` (`trials`) | Use reconstructed trial order from `trial_start` events, not raw `trial number` during teleport. |
| Previous trial reward outcome | `input[3]` = `previous_trial_outcome` | For trial `t`, use reward outcome of trial `t-1`; first trial defaults to `0`; repeat across frames | Consistent with task spec; related to `behavior.get_trial_types` | Binary, `0=omitted/unrewarded`, `1=rewarded`. |
| `position` + trial reward-zone coordinates | `output[0]` = `distance_to_reward_zone_bin` | Signed distance to nearest point in active reward zone: negative before zone, zero inside, positive after zone; discretize into 7 bins from task spec | Paper reward-zone definitions + scene parsing | Use per-trial active zone A/B/C inferred from scene / switch status. |
| `position` | `output[1]` = `absolute_position_bin` | Bin position on `0-450 cm` track into 5 equal bins | Paper track length 450 cm | Use only in-trial frames; exclude teleport. |
| `speed` | `output[2]` = `speed_bin` | Discretize cm/s into `<2`, `2-10`, `10-20`, `20-40`, `>40` | `glmUtils.get_timeseries_data` speed stream; paper threshold at 2 cm/s | Do not drop low-speed frames because they are a decoder target class here. |
| `lick` | `output[3]` = `lick` | Clip cumulative lick counts to binary `0/1`; bad-lick trials dropped by QC rule | `glmUtils.get_timeseries_data`, paper lick QC | Target is time-varying binary lick. |
| Scene-derived active reward zone | `output[4]` = `reward_zone_location` | Map `A/B/C -> 0/1/2`; repeat across frames in the trial | `behavior.get_reward_zones` | Derived from NWB `identifier` scene string and switch-after-30 rule. |
| Trial reward delivery from `BehavioralTimeSeries/Reward` | `output[5]` = `reward_outcome` | `1` if any reward event timestamp falls within trial, else `0`; repeat across frames | `behavior.get_trial_types`, `rewardAnalysis.get_reward_inds` | Uses actual delivered reward, matching decoder task. |

### Key Decisions
1. **Neural signal = deconvolved activity**: The reference decoder and place-cell pipeline use deconvolved calcium activity (`events`). NWB already exports `Deconvolved`, so this is the closest native match.
2. **Trial-aligned frames exclude teleport period**: Reference code keeps samples from trial start until teleport onset. I will reconstruct trials from `trial_start` impulses to the next `teleport` impulse, excluding teleport frames.
3. **Use native imaging-frame resolution**: The reference data are sampled at ~15.5 Hz and behavior is already synchronized to this grid. No rebinning in time unless a later validation forces it.
4. **Per-trial constants will be repeated across time**: Although some decoder variables are per-trial, repeating them across each trial’s time axis gives a uniform `(n_variables, n_timepoints)` representation and matches the decoder’s time-varying format.
5. **Scene identifier drives reward-zone identity**: NWB `reward_zone` is not the A/B/C label. Active reward-zone identity will be parsed from the session identifier (for example `Env1_LocationA_to_B`, `Env1_B_to_Env2_C`) and the 30-trial switch rule from the paper/code.
6. **Bad lick trials will be removed**: Because lick is a required decoder output and NaNs are undesirable, trials meeting the paper’s lick-sensor failure rule will be excluded rather than carried with missing labels.
7. **Initial neuron filter = `iscell[:,0] == 1`**: This matches Suite2p manual curation. Additional interneuron-style exclusions remain a consistency check because of the modest count mismatch with the paper.
8. **No speed-based sample exclusion in final conversion**: The paper excludes `<2 cm/s` for spatial analyses, but here speed itself is an output class. I will preserve these frames and encode low speed as class `0`.
9. **Absolute position uses track-only interval**: Position bins will cover the 450 cm corridor only; teleport frames are excluded, so no teleport-specific bins are needed.

### Planned Sanity Checks
- [ ] Check 1: For a chosen session/trial, verify `neural[session][trial]` equals the raw NWB `Deconvolved` samples at the same curated ROI indices and frame range with `np.allclose`.
- [ ] Check 2: For chosen trials, verify reconstructed trial start/end frames match raw `trial_start` / `teleport` impulse locations and that converted time starts at 0 s.
- [ ] Check 3: For chosen trials, verify raw `position` reproduces `absolute_position_bin` and `distance_to_reward_zone_bin` exactly under the planned binning rule.
- [ ] Check 4: Verify per-trial reward outcome from converted data matches presence/absence of raw `Reward` events within the raw frame/timestamp interval.
- [ ] Check 5: Verify scene parsing and 30-trial switch logic reproduce per-trial reward-zone labels consistent with paper/code examples.

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `convert_data.py` with CLI:
  - `python -u convert_data.py <outpicklefile>`
  - `--full`
  - `--sample`
  - `--show-processing`
- Current implementation choices:
  - Reads NWB directly with `h5py`
  - Uses NWB `Deconvolved/plane0/data` as neural activity
  - Filters ROIs with `iscell[:,0] == 1`
  - Reconstructs trials from `trial_start` impulse to `teleport` impulse
  - Drops trials failing lick-sensor QC
  - Derives environment and reward-zone identity from the NWB `identifier` scene string and the 30-trial switch rule
  - Repeats per-trial inputs/outputs across frames so all trials are stored as `(d, T)`
- Added processing plots for up to 2 sessions to visualize:
  - raw position with reward-zone overlay
  - speed
  - lick
  - sample neural activity
  - discretized outputs
  - trial-duration distribution
- Added per-session timing / summary prints to support runtime estimation in Step 7.

Code inefficiencies identified:
- Full-session deconvolved activity is currently loaded into memory per session before trial slicing.
- No parallel file processing yet.

Code speedups added:
- Uses direct HDF5 access with `h5py` instead of slower NWB object loading.
- Reads only curated ROI columns from the deconvolved matrix.
- Stores neural trials as `float16` to reduce output size; training code later casts to `float32`.
- Processes sessions sequentially to keep peak memory bounded by one session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 323 |
| Neurons / session | 155, 168 |
| Subjects | 1 (`m11`) |
| Sessions / subject | 2 |
| Trials (total) | 160 |
| Trials / session | 80, 80 |
| `time_from_trial_start_s` range | [0.0, 30.6] |
| `environment` range | [0.0, 0.0] |
| `trial_number` range | [0.0, 79.0] |
| `previous_trial_outcome` range | [0.0, 1.0] |
| `distance_to_reward_zone_bin` distribution | [0.0883, 0.0921, 0.0577, 0.2850, 0.0217, 0.0714, 0.3839] |
| `absolute_position_bin` distribution | [0.3116, 0.2065, 0.2029, 0.1457, 0.1333] |
| `speed_bin` distribution | [0.1208, 0.0623, 0.0656, 0.2332, 0.5181] |
| `lick` distribution | [0.8198, 0.1802] |
| `reward_zone_location` distribution | [0.7919, 0.2081, 0.0000] |
| `reward_outcome` distribution | [0.1609, 0.8391] |

### Processing Plots Review
- Reviewed `processing_sub-m11_ses-03.png`.
- Trial-aligned position ramps monotonically from near 0 cm to near track end.
- Reward-zone overlay falls in the expected corridor segment.
- Distance/position/speed discretizations change at plausible times with no visible temporal offset from the raw traces.
- Lick output appears sparse and synchronized to the raw lick trace.
- No obvious teleport contamination inside trial segments.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Direct `h5py` reads instead of NWB object loading | Large reduction in per-session I/O and parsing overhead |
| Read only curated ROI columns | Avoids loading non-cell ROIs in single-plane sessions |
| `float16` neural storage | Roughly halves pickle size and write bandwidth |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion on two small sessions | ~1.0-1.1 s/session | Not representative; underestimates full run |
| Large multi-plane benchmark (`m18 ses-03`) | ~2.49 s/session | Conservative upper bound ~6.3 min for 152 sessions |
| Full conversion estimate | ~1.5-2.5 s/session | ~4-7 min total |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `distance_to_reward_zone_bin` | 0.4065 | 0.3179 |
| `absolute_position_bin` | 0.5390 | 0.4601 |
| `speed_bin` | 0.3693 | 0.3308 |
| `lick` | 0.7240 | 0.6716 |
| `reward_zone_location` | 0.9615 | 0.9695 |
| `reward_outcome` | 0.6444 | 0.6938 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 4.5G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Not stated directly | `iscell` + later curation/filtering | 138,678 curated ROIs by `iscell[:,0]==1` | 138,678 | Yes vs data |
| Mean neurons/session | 155-2172 range reported | curated ROIs from Suite2P plus exclusions | mean 912.36, max 2341 | mean 912.36, max 2341 | Yes vs data; paper max differs |
| Subjects | 11 switch mice | 11 included animals in switch task analyses | 11 | 11 | Yes |
| Sessions | 152 implied by 11*14 minus 2 missing m11 days | session list in NWB export | 152 | 152 | Yes |
| Trials (total) | 12,376 imaged trials quoted in lick-QC context | trial starts / teleports per session | 12,217 raw reconstructed trials | 12,147 after dropping 70 bad-lick trials | Converted matches planned QC; paper vs data differ |
| Trials/session (mean) | 80.5 +/- 7.4 across all imaging days in 14 mice | one session per day, targeted 80-100 | mean 80.38 raw | mean 79.91 after QC | Close |
| `time_from_trial_start_s` range | Variable trial durations | aligned continuous samples | [0.0, 216.5] raw trial spans | [0.0, 216.5] | Yes |
| `environment` range | `ENV1/ENV2` | `morph` 0/1 | [0, 1] | [0, 1] | Yes |
| `trial_number` range | ~80-100 trials/session | sample-wise trial index in code | [0, 99] | [0, 99] | Yes |
| `previous_trial_outcome` range | omitted/rewarded | derived from per-trial reward outcome | [0, 1] | [0, 1] | Yes |
| `distance_to_reward_zone_bin` distribution | N/A | N/A | derived from raw position + scene | [0.2511, 0.1019, 0.0733, 0.2382, 0.0207, 0.0719, 0.2428] | Internal check only |
| `absolute_position_bin` distribution | N/A | N/A | derived from raw position | [0.2120, 0.1768, 0.2310, 0.2265, 0.1537] | Internal check only |
| `speed_bin` distribution | N/A | N/A | derived from raw speed | [0.1174, 0.0875, 0.1340, 0.3189, 0.3422] | Internal check only |
| `lick` distribution | sparse licking expected | code clips licks to binary | raw lick stream available | [0.7767, 0.2233] | Plausible |
| `reward_zone_location` distribution | A/B/C balanced across cohort expected | scene-based reward zones | scene strings span A/B/C | [0.3316, 0.3362, 0.3322] | Yes |
| `reward_outcome` distribution | ~15% omissions | `isreward` per trial | raw reward events imply ~84.7% reward before QC | [0.1581, 0.8419] | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reports no errors and no warnings.
2. **Sanity checks against raw NWB**:
   - Single-plane spot check (`session 0`, `trial 10`): raw deconvolved activity vs converted neural matched with `np.allclose(..., atol=1e-3)`.
   - Multi-plane spot check (`session 84`, `trial 5`, `m18 ses-03`): raw plane0+plane1 reconstruction in segmentation-table order vs converted neural matched with `np.allclose(..., atol=1e-3)`.
   - Input checks for the same trials: reconstructed time, environment, trial number, and previous reward outcome matched converted inputs exactly.
   - Output checks for the same trials: raw position/speed/lick/reward-derived outputs matched converted outputs exactly.
   - Switch edge case check (`sub-m11_ses-08_behavior+ophys.nwb`): trial 29 resolves to `(Env1, B)` and trial 30 resolves to `(Env2, C)`, confirming cross-environment switch parsing.
3. **Reference code comparison**:
   - `(a) data loading`: reference loads a `sess` object; converter loads the NWB export directly. This is a packaging difference, not a semantic one.
   - `(b) neuron/trial filtering`: converter applies `iscell` and lick-trial QC; reference also mentions a speed-correlation interneuron exclusion. A deconvolved-speed proxy check on the largest session found 0 neurons above `r > 0.5`, so this is not the source of the main count mismatch.
   - `(c) temporal alignment`: converter uses `trial_start -> teleport`, matching the reference in-trial slicing logic.
   - `(d) binning`: converter keeps the native ~15.5 Hz time base and applies task-required output discretization; paper/reference spatial binning is 10 cm, used here only where needed for target outputs.
   - `(e) input construction`: time, env, trial number, and previous outcome are built from raw aligned behavior and repeated across frames.
   - `(f) output construction`: outputs are derived directly from raw behavior plus scene-defined reward zones and checked against raw NWB values.
4. **Key statistics comparison**:
   - Raw NWB reconstructed reward rate is `0.8466`, omission rate `0.1534`, matching the paper’s approximately 15% omission design.
   - Raw reconstructed trials from `trial_start`/`teleport` total `12,216`; converted data keep `12,147` after dropping `69` lick-error trials (`0.565%`), close to the paper’s reported `81/12,376 = 0.654%`.
   - Raw curated ROI count range is `155` to `2341`; converted data preserve the same range.
5. **Edge-case checks**:
   - `sub-m11_ses-03` has `81` nonnegative trial numbers but only `80` `trial_start` and `teleport` events. Converter follows `trial_start`/`teleport` and therefore excludes one partial/non-aligned trial, which is consistent with trial-start alignment.
   - Multi-plane sessions (`m17`, `m18`) required explicit reconstruction from `plane0` and `plane1`; this was fixed before the full conversion.

### Issues Found and Resolved
- **Multi-plane ROI loading bug**: Initial converter assumed a single `plane0` response matrix and failed on `m17/m18`. Fixed by reconstructing a full session matrix from `plane0` and `plane1` using `planeIdx`, then applying `iscell` curation in segmentation-table order. Re-ran sample and full conversion after the fix.
- **Paper-vs-release trial count mismatch**: Paper quote (`12,376` trials) does not match raw NWB reconstructed trials (`12,216`). This discrepancy is present in the released data itself and is not introduced by conversion.
- **Paper-vs-release max neuron count mismatch**: Paper reports up to `2172` neurons/session, while raw NWB has one multi-plane session with `2341` curated ROIs. This discrepancy is present in the released data itself; converted data preserve the release values.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `distance_to_reward_zone_bin` | 0.4742 | 0.4272 | Well above chance (0.1429) |
| `absolute_position_bin` | 0.5374 | 0.5108 | Well above chance (0.2000) |
| `speed_bin` | 0.4346 | 0.4141 | Well above chance (0.2000) |
| `lick` | 0.5917 | 0.5847 | Above chance (0.5000) but smallest margin |
| `reward_zone_location` | 0.8433 | 0.8129 | Strongly above chance (0.3333) |
| `reward_outcome` | 0.5759 | 0.5281 | Slightly above chance (0.5000) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Paper |
| `distance_to_reward_zone_bin` | 0.4272 validation balanced accuracy | Paper does not report this exact target; should be clearly above chance if alignment is correct |
| `absolute_position_bin` | 0.5108 | Paper does not report this exact target |
| `speed_bin` | 0.4141 | Paper does not report this exact target |
| `lick` | 0.5847 | Paper does not report this exact target |
| `reward_zone_location` | 0.8129 | Paper does not report this exact target; high accuracy expected because zone identity is stable within trials |
| `reward_outcome` | 0.5281 | Paper does not report this exact target |
| Paper RR-position decoder (closest conceptual comparison) | Not directly comparable to these outputs | Paper reports RR decoding above shuffle after the switch only for RR cells, with z-scored decode >2 spanning about -104.5 cm to +152.7 cm relative to reward |

[Analysis of any low accuracies]
- All outputs are above uniform-chance balanced accuracy.
- `lick` (`0.5847` vs chance `0.5000`) and `reward_outcome` (`0.5281` vs chance `0.5000`) are below the heuristic `1.5x chance` threshold, so they were investigated further.
- Investigation results for `lick` and `reward_outcome`:
  - Raw-label checks on 3 trials (`m11 ses-03` trials 0 and 10, `m18 ses-03` trial 5) matched converted outputs exactly.
  - `lick` class balance is `77.67% no / 22.33% yes`; `reward_outcome` is `15.81% no / 84.19% yes`, so neither output is degenerate.
  - Train/validation ratios are small (`lick 1.012`, `reward_outcome 1.091`), arguing against strong overfitting or leakage.
  - Temporal alignment had already been checked visually in processing plots and quantitatively in Step 10 sanity checks.
- Interpretation:
  - `reward_outcome` is encoded as a per-trial label repeated across time, per the decoder task spec. This is inherently less decodable from pre-reward neural activity than the paper’s time-varying `rewarded` state variable, so only modest above-chance performance is expected.
  - `lick` is a sparse binary event and therefore also has a relatively small margin above chance despite being correctly aligned.
- No evidence from these checks suggests a conversion bug that would justify changing the labels away from the user-specified task definition.

### Issues Found and Resolved
- **Borderline accuracy for `lick` and `reward_outcome`**: Investigated via raw-trial label checks, class-balance analysis, and train-vs-validation gap analysis. No labeling or alignment bug found; retained current representation because it matches the decoder task specification.
- **Paper accuracy comparison limits**: The paper reports decoder performance for circular reward-relative position and shuffled controls, not for the six task outputs required here. Documented the closest conceptual comparison instead of claiming a direct numeric match.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
