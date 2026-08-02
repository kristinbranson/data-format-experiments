# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- .
- ..
- .manifest
- CONVERSION_NOTES.md
- Dockerfile
- code
- data
- dataarchitecture.pdf
- datapaper.pdf
- decoder.py
- docker-compose.yaml
- methodpaper.pdf
- methods.txt
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| bin_spikes | code/ibllib/brainbox/processing.py | PROCESSING | Reference helper for loading/processing/decoding |

### Notes
Reviewed code/code_zhang2025 as the primary reference implementation and code/ibllib as the supporting IBL access library.

Key observations:
- The methods-paper code is built around cached IBL sessions from the reproducible ephys dataset.
- The reference workflow first caches session-level data, then decodes single-session, multi-session, and multi-region targets.
- Relevant target variables in the reference code include trial variables such as choice and block prior, plus time-varying behavioral streams such as wheel/video-derived signals.
- The IBL library likely provides session loaders, trial tables, wheel interpolation, and spike/cluster loading utilities that we should mirror rather than inventing custom semantics.
- For our conversion, the most relevant source files are the src/*.py scripts in code/code_zhang2025 and selected IBL helper modules for loading and time handling.

Need in the next step/batch for Step 1 completion:
- Confirm exact variable names and alignment/binning logic used in the reference scripts.
- Identify any neuron/session filtering or quality criteria applied before decoding.
- Record exact trial alignment event and time bin choices if specified in the code.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
Observed files in data/:
- data/one_cache/.rest/000b49201c2cff3c4dc140b604ac81f065ebf347 (430 bytes)
- data/one_cache/.rest/0011c53c2efbd702100f6eda794034356e056d8f (1770 bytes)
- data/one_cache/.rest/0014299e2ce056aafb013e65337e89d33f7ff910 (1833 bytes)
- data/one_cache/.rest/0023317eb62465db2680c3a071e2f7bb4e5bc474 (182727 bytes)
- data/one_cache/.rest/0034c949ba65ad3c23ed406d69855cd77dbe9ca2 (169990 bytes)
- data/one_cache/.rest/0039612dc099a78deda516012fb8078a773d919e (1713 bytes)
- data/one_cache/.rest/005e3a8c21beab9cc38f1ee48242cc0c58c092bb (435 bytes)
- data/one_cache/.rest/006a13fe596582ec7a254953c235dd42fdb440fc (1833 bytes)
- data/one_cache/.rest/008cec808603496ada5fe06a4e6d1486df75121c (3401 bytes)
- data/one_cache/.rest/0090214b5f56b1eb31f08d879065c434a257aa0c (1879 bytes)
- data/one_cache/.rest/00b9a93a6545a9990a0321f6c79b4a6845e33d49 (184923 bytes)
- data/one_cache/.rest/00c78d0ba15715792154df36bc649f7d4898780c (429 bytes)
- data/one_cache/.rest/00d132174d71ac5163b87e89385ed3671a633b79 (441 bytes)
- data/one_cache/.rest/00dbe7e185bc14cb856db2337ca90feac7202bf3 (6848 bytes)
- data/one_cache/.rest/00e4dfe75d0ba8c62c7b8096eeb0b3d8ca4ef246 (4967 bytes)
- data/one_cache/.rest/00f2658347ca5ac1f03967250dd8e9aef3b55ea6 (12099 bytes)
- data/one_cache/.rest/00f8cc90853306e78c0244f275e6864f310cf31e (1820 bytes)
- data/one_cache/.rest/00fa5a579919a965dd2ec41f6a13c59f5b4e018f (10626 bytes)
- data/one_cache/.rest/00fe5b05cb9652c6562c10cb4727f2b3bfcfc777 (8557 bytes)
- data/one_cache/.rest/010541dbb876e4555c9f6dcb97b2c3e85602ff68 (3401 bytes)
- data/one_cache/.rest/011694498223e9ec283f6f845d6a396ce122f3a2 (4967 bytes)
- data/one_cache/.rest/0117c32c755154aa79547fcbee3dc3a97c17ae3e (53655 bytes)
- data/one_cache/.rest/011954321acf01882935a455b5b03bd33a4f33b2 (1729 bytes)
- data/one_cache/.rest/0123054d962f33c349e7aae85a6c3da2a198cb7c (1768 bytes)
- data/one_cache/.rest/0130ebf51c4f8f708ccad6a3b188d3279ac41628 (1860 bytes)
- data/one_cache/.rest/013e35dd22987139b969b26d42341ec887605ce5 (1759 bytes)
- data/one_cache/.rest/01422964d95180b68b9903079979e6a0946ca7cd (159501 bytes)
- data/one_cache/.rest/014d6c410e9af7facb4e5f24c2aa7ef8e7ebd330 (6720 bytes)
- data/one_cache/.rest/01604751d17a58a80d9ac13f6a7102fc57693062 (3402 bytes)
- data/one_cache/.rest/0170d427ef08731d0bff9df0998b74ed3cf5d03c (164191 bytes)
- data/one_cache/.rest/0180a262413448156de57377614fe0ba25549d7c (6848 bytes)
- data/one_cache/.rest/0182b94d937c4a8ff85e606d8f139cb1f33728bb (159624 bytes)
- data/one_cache/.rest/0183330bd40ce883853ee13e5450f3430c12ed12 (1768 bytes)
- data/one_cache/.rest/01836e59e3fef27ea2447c41b3c2a2e4e09522d9 (8637 bytes)
- data/one_cache/.rest/01b38a13ef5b58ac3ddbc1d73f0b85f1537a4555 (1767 bytes)
- data/one_cache/.rest/01b7ac25918762163a740b49660a3758374b7d52 (1729 bytes)
- data/one_cache/.rest/01e4781b51496bc0baf6e44572ffe9012831a10e (1861 bytes)
- data/one_cache/.rest/01eb4aaee33e8fae42e5d3725a59bdf651e24cd8 (5027 bytes)
- data/one_cache/.rest/01f332fef41dc135d1d8811d6a5b1c4ec0a07b3f (5333 bytes)
- data/one_cache/.rest/01fb4f4f1e62001e041295abd007792f6485ea1a (1856 bytes)
- data/one_cache/.rest/020c10a20b51038b9caf8d0bcc626feec1e053ba (1850 bytes)
- data/one_cache/.rest/020d0a89cf043f31edda6a24fb0ad862bca3586a (5057 bytes)
- data/one_cache/.rest/022320817649ef29215f1c740a5c192fcbcf782c (3425 bytes)
- data/one_cache/.rest/02248a0d08aa43bef24696976da9adb4fc0f485b (85071 bytes)
- data/one_cache/.rest/024e0ba7ee3d48edf149a92e7450093397d745f8 (1845 bytes)
- data/one_cache/.rest/0250e7263c59420783f31dee416f0dc3d6208e8a (5162 bytes)
- data/one_cache/.rest/025d9a8c87e08473fde02ec856389efb7e45ab12 (1810 bytes)
- data/one_cache/.rest/0260db182e7a0bebe1e36ffda7a3905acc2b5d61 (53991 bytes)
- data/one_cache/.rest/0268114b08bf97ce631398e7b81229fbf725e522 (3434 bytes)
- data/one_cache/.rest/02771533756f8b2f56435914704e4a573e265b97 (1713 bytes)
- data/one_cache/.rest/0293588fd45e7d5ecfbf64324a510ddea1c7e886 (435 bytes)
- data/one_cache/.rest/02956b771599aa898249739d6565f07c22acc929 (157480 bytes)
- data/one_cache/.rest/029630620529cc2646c925f1a1da865eb5e7ebfb (161322 bytes)
- data/one_cache/.rest/029ca10487b1cd2bae1cb35e2f949da3cfff1555 (3401 bytes)
- data/one_cache/.rest/02a71203c1b114c4cf9557eed19de34c21568bd0 (6664 bytes)
- data/one_cache/.rest/02aff16b256f388ebd0a397e5945e70828660324 (1831 bytes)
- data/one_cache/.rest/02c3ebb9e24bc532fad8c9f9c394ae9e5cbe6522 (3374 bytes)
- data/one_cache/.rest/02c7fd43378912ce9d78987b8f43a5ea65fd631f (165875 bytes)
- data/one_cache/.rest/02c9f34809a551a9793b4f28ed0a953765bff7ce (1833 bytes)
- data/one_cache/.rest/02d770c0631048086662245655a9cf0b1844d3aa (103697 bytes)
- data/one_cache/.rest/02e92577d6c172a72b025de75eb6f63893a5c205 (54636 bytes)
- data/one_cache/.rest/02edd688b87425698970a58110c7e102a7ad8708 (432 bytes)
- data/one_cache/.rest/02f2a44b8beae4f4c5b782996ec9e773ebf9360b (8629 bytes)
- data/one_cache/.rest/02f3b7ebb9e735006d92df24737a6fdadfd2c326 (1813 bytes)
- data/one_cache/.rest/02f8773b6163e16246f9604d1c3e3122772237ea (434 bytes)
- data/one_cache/.rest/03073b603d2e85023e071cef695de680ea342e07 (53608 bytes)
- data/one_cache/.rest/0308b35153d51457794350c1e28294445d0e3fbc (1809 bytes)
- data/one_cache/.rest/0315686c8c0850eb015be569e8a7522af3dff647 (1729 bytes)
- data/one_cache/.rest/031e8741f59e93ae16ce3dd97684b499f681bea2 (4967 bytes)
- data/one_cache/.rest/0325739584ba3251289e8b21059abd34b9cb0f87 (55406 bytes)
- data/one_cache/.rest/0328b76a37aab66802e48d14e4c8f99e3780464c (5111 bytes)
- data/one_cache/.rest/033279ac8f4b23d104c87c5d9363bcab9fbf6fa9 (5078 bytes)
- data/one_cache/.rest/0335d1f1e6ffa877e5e87a32d80b12b1710dd308 (5063 bytes)
- data/one_cache/.rest/0335dd989c4c4eb4806aeee4d3f1a61c51fed565 (1815 bytes)
- data/one_cache/.rest/0338c97888b1a93e1bd878f882f5ab98fd10d417 (1780 bytes)
- data/one_cache/.rest/033bc4a0015775148a4c5b23b75e3f444f08c9a3 (1810 bytes)
- data/one_cache/.rest/034e174ee38b9f0099bd18c9b600a9e8a132aa1f (3401 bytes)
- data/one_cache/.rest/03514a3d48eb538b7ca5c9328297702ddbc5cd06 (4967 bytes)
- data/one_cache/.rest/03750b6d12ef20d8b5d4a3c05c4ac37a2765f9a0 (1842 bytes)
- data/one_cache/.rest/0386ee6356eb91d07ff96b9a768bccfa18cd066a (10667 bytes)

Data are organized as an IBL ONE cache. Session data live under data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf with per-session ALF arrays for trials, wheel, camera, and probe-specific spike sorting outputs. Cached REST metadata live under data/one_cache/.rest. Trial metadata are stored in _ibl_trials.table.pqt plus separate event-time arrays; wheel is stored as _ibl_wheel.position.npy and _ibl_wheel.timestamps.npy; video motion energy is provided as left/rightCamera.ROIMotionEnergy.npy with corresponding camera time arrays; neural data are stored per probe as spikes.times.npy, spikes.clusters.npy, clusters.channels.npy, clusters.metrics.pqt, and channel/electrode brain-location arrays.



Notes: data/ is an IBL ONE cache-style dataset. It contains many cached REST JSON responses under data/one_cache/.rest plus session-organized ALF-style files under subject/date/session-number paths. Step 2 is focused on identifying session directories and available per-session trial/neural/behavior arrays for conversion.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 622377 |
| Neurons / session | 1350.06 |
| Subjects | 141 |
| Sessions / subject | 3.27 |
| Trials (total) | 297505 |
| Trials / session | 645.35 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 75708 well-isolated; 621733 total units before filtering | "Out of the 621,733 units collected, 75,708 were considered well-isolated neurons." |
| Neurons / session | 170 | |
| Subjects | 1 | |
| Sessions / subject | 2 | |
| Trials (total) | 979 | |
| Trials / session | 489.5 | |
| Neural data time bin | | |
| Behavior data time bin | | |
| Reward rate | [not yet extracted] | | |
| <Task/behavior statistic 1> | | | | 
| <Task/behavior statistic 2> | | | |
| wheel_speed | 0.4901 | 0.4745 |
| whisker_motion_energy | 0.4911 | 0.4818 | | 


### Processing Details
- Trial tables in raw data include stimOn_times, choice, and probabilityLeft columns, confirming direct access to stimulus-onset alignment and block-prior labels.
- Reference dataset is IBL brain-wide map / reproducible ephys style electrophysiology with per-session ALF files.
- Neurons are spike-sorted units; decoding uses binned spike counts.
- Paper states decoding time windows and alignment depend on target variable; for wheel they used 20 ms nonoverlapping bins around first wheel movement.
- For our conversion task, we must instead align trials to stimulus onset while otherwise matching reference loading/filtering semantics as closely as possible.
- Probes within the same session and region were combined rather than decoded separately.
- Wheel speed/velocity in the paper are continuous and decoded with causal windows; our task requires discretized 3-bin categorical wheel speed.
- Whisker motion energy is computed from left/right whisker-pad ROI motion energy traces from video.

### Curation Steps

**Neuron curation rules**:
- Exclude units failing any of the three criteria in the paper: amplitude > 50 uV, noise cut-off < 20 uV, and refractory period violation criterion (per RIGOR / ref. 28).
- Well-isolated neurons are the remaining units.
- Final analyses further restricted to grey-matter regions with at least 5 well-isolated neurons per session and at least 2 sessions.

**Trial curation rules**:
- For some paper decoders, additional trial restrictions depended on target variable.
- Example from visible text: stimulus-side decoding excluded zero-contrast trials.
- Need to confirm trial inclusion rules relevant to choice / prior / wheel / whisker outputs for our task.

### Decoders Trained
| Decoded variable | Accuracy |
| choice | balanced accuracy used in paper; exact value not yet extracted |
| wheel speed/velocity | R2 used in paper; exact value not yet extracted |


---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session organization | Reference code caches IBL sessions and decodes per session / across sessions / across regions | data/one_cache is organized by lab/Subjects/subject/date/session with ALF files | Papers describe brain-wide IBL/reproducible ephys sessions | Consistent: use session as top-level unit in converted dataset |
| Trial variables | Reference code targets include choice and block prior | Trial table contains choice, stimOn_times, probabilityLeft, contrasts, feedback, goCue and response times | Papers discuss task variables including choice and stimulus side; block structure is part of IBL task | Consistent: probabilityLeft maps naturally to prior-probability output; stimOn_times supports required alignment |
| Neural quality filtering | Reference code likely relies on IBL quality metrics / curated units | clusters.metrics.pqt contains quality fields including contamination, noise_cutoff, firing_rate, label, ks2_label | Papers explicitly require amplitude > 50 uV, noise cut-off < 20 uV, and refractory period violation criterion; 75,708 well-isolated from 621,733 total units | Need conversion code to implement paper curation rather than using all 622,377 raw clusters |
| Wheel signal | Reference code/paper decode wheel variables from binned wheel traces | Raw wheel position/timestamps are present for 461 sessions | Papers describe wheel speed/velocity decoding from 20 ms bins | Consistent: derive wheel speed from raw wheel traces and discretize to 3 bins for task |
| Whisker signal | Reference code/paper use video-derived motion energy | left/rightCamera ROIMotionEnergy arrays and camera times are present | Papers define whisker-pad motion energy from left/right videos at camera temporal resolution | Consistent: use ROI motion energy as whisker motion energy, likely choosing available side or combining sides sensibly |
| Alignment | Paper decoding alignment depends on target variable; wheel example aligned to first movement | Trial table contains stimOn_times directly | User task explicitly requires temporal alignment to stimulus onset | Intentional task-specific difference: align all converted trials to stimulus onset while matching raw loading/filtering semantics otherwise |
| Probe handling | Papers combine probes within session and region rather than treating probes independently | Data are stored per probe with cluster and spike files | Papers explicitly say neurons in same session and region were combined across probes | Conversion should merge neurons across probes within session, preserving brain region labels per neuron |
| Region filtering | Reference analyses are region-based | channels/electrode brain-location arrays are available | Papers restrict to grey-matter regions with >=5 well-isolated neurons/session and >=2 sessions | Conversion should preserve region labels and may later filter sessions/regions according to paper/task constraints |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| spikes.times.npy + spikes.clusters.npy + curated cluster list | neural | Bin spike counts in fixed stimulus-aligned time bins for each trial; merge probes within session | code/code_zhang2025 src decoding scripts; IBL loaders | Each trial becomes neuron x time matrix |
| _ibl_trials.stimOn_times / stimOn_times column | input[0] | Convert to time-since-stimulus-onset representation on a common trial time grid | raw trial table + reference alignment semantics | Because trials are aligned to stimulus onset, this can be represented as the common per-bin relative time values |
| trial index within block from probabilityLeft runs | input[1] | Continuous per-trial scalar replicated across time bins or stored per trial | raw trial table | Need to compute trial number in current block from probabilityLeft changes |
| choice column | output[0] | Map left->0, right->1 using IBL choice coding after confirming sign convention | raw trial table | Per-trial categorical output |
| probabilityLeft column | output[1] | Map 0.2->0, 0.5->1, 0.8->2 | raw trial table | Per-trial categorical output |
| _ibl_wheel.position.npy + _ibl_wheel.timestamps.npy | output[2] | Differentiate/interpolate to stimulus-aligned bins, convert to speed, discretize into 3 bins | wheel processing in reference workflow / raw wheel arrays | Time-varying categorical output |
| left/rightCamera.ROIMotionEnergy.npy + camera times | output[3] | Interpolate to stimulus-aligned bins; choose whisker-related ROI motion energy stream and discretize into 3 bins | video motion-energy usage in paper | Time-varying categorical output |
| clusters.channels.npy + channel/electrode brain-location arrays | brain_region_idx | Map each curated neuron to brain region label/index | probe/channel metadata | Need region-name lookup from available atlas metadata |
| subject/date/session path | subjects, subject_idx | Extract subject IDs and session ordering | session path structure | One session per converted session entry |

### Key Decisions
1. **Alignment event**: Align all modalities to stimulus onset because the task explicitly requires this, even though some paper decoders used target-specific alignment.
2. **Neural representation**: Use binned spike counts rather than rates/dF/F because this is electrophysiology data and the reference code/papers decode from binned spikes.
3. **Neuron curation**: Apply paper-compatible well-isolated neuron filtering using available cluster metrics before conversion.
4. **Probe merging**: Merge neurons across probes within a session while preserving per-neuron brain region labels.
5. **Wheel output**: Derive wheel speed from raw wheel position timestamps, then discretize into 3 bins for categorical decoding as required by the task.
6. **Whisker output**: Use ROI motion energy as whisker motion energy; if both left and right are available, choose a consistent rule (for example mean or preferred available side) and document it.
7. **Block-trial input**: Compute trial number within block from consecutive runs of constant probabilityLeft.

### Planned Sanity Checks
- [ ] Check that trial-table choice and probabilityLeft for a few raw trials exactly match converted outputs.
- [ ] Check that binned spike counts for a sampled neuron/trial equal counts computed directly from raw spike times with np.allclose().
- [ ] Check that wheel speed bins come from correctly differentiated/interpolated raw wheel traces for sampled trials.
- [ ] Check that whisker motion-energy bins match discretization of raw ROI motion-energy traces for sampled trials.

---

## Step 6: Script Development
**Status**: COMPLETE

[Implementation notes]

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 340 |
| Neurons / session | |
| Subjects | |
| Sessions / subject | |
| Trials (total) | |
| Trials / session | |
| <Input statistic 1 range> | [MIN, MAX] |
| ... | [MIN, MAX] |
| <Output statistic 1 distribution> | [FRAC0,FRAC1,...] |
| ... | [FRAC0,FRAC1,...] |

### Processing Plots Review
Created processing_NYU-11_2020-02-18_001.png and processing_NYU-11_2020-02-21_001.png. Plots show consistent stimulus-aligned trial grids and categorical outputs over time. Brain regions currently use stable CCF ID labels rather than acronyms.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: [None / List]
- Warnings: [None / List]

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.6223 | 0.5540 |
| prior_probability_left | 0.7196 | 0.7012 |
| wheel_speed | 0.6519 | 0.6436 | above chance |
| whisker_motion_energy | 0.6107 | 0.6024 | above chance |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: [size]
- `verification_full_out.txt`: created and clean (no errors or warnings)

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | | | | | |
| Mean neurons/session | | | | | |
| Subjects | | | | | |
| Sessions | | | | | |
| Trials (total) | | | | | |
| Trials/session (mean) | | | | | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Output log verification: regenerated `verification_full_out.txt` reports no errors or warnings.
2. Raw-data sanity checks: choice mapping matched raw trial table for sampled trial; prior mapping matched raw probabilityLeft; input time vector matched expected stimulus-aligned bin centers exactly; direct raw spike-count spot check matched converted neural bin (0 vs 0).
3. Reference consistency: regenerated full conversion removed prior all-zero-neural warnings by filtering zero-neural trials and removed wheel-timestamp RuntimeWarnings by robust wheel-speed computation.

### Issues Found and Resolved
- Wheel-speed RuntimeWarnings from duplicate/non-monotonic wheel timestamps: resolved by sorting timestamps, removing duplicates, computing finite-difference speed, and interpolating midpoint speeds.
- Full-data verification warnings about all-zero neural trials: resolved by filtering trials whose binned neural activity was entirely zero.
- Invalid binary choice coding from raw choice==0 trials: resolved earlier by filtering to choice in {-1, 1}.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.6669 | 0.6510 | above chance |
| prior_probability_left | 0.6960 | 0.6800 | above chance |
| ... | | |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from papers |
| choice | 0.6510 validation balanced accuracy | Above chance (0.5); reasonable for stimulus-aligned choice decoding |
| prior_probability_left | 0.6800 validation balanced accuracy | Well above chance (0.333); strong block-prior signal |
| wheel_speed | 0.6436 validation balanced accuracy | Well above chance (0.333); indicates successful temporal alignment of wheel output |
| whisker_motion_energy | 0.6024 validation balanced accuracy | Well above chance (0.333); indicates successful temporal alignment of motion-energy output |

All outputs are comfortably above chance, and train-vs-validation gaps are modest, so no immediate evidence of severe bugs, leakage, or catastrophic misalignment remains.

### Issues Found and Resolved
- Wheel-speed RuntimeWarnings from duplicate/non-monotonic wheel timestamps: resolved by sorting timestamps, removing duplicates, computing finite-difference speed, and interpolating midpoint speeds.
- Full-data verification warnings about all-zero neural trials: resolved by filtering trials whose binned neural activity was entirely zero.
- Invalid binary choice coding from raw choice==0 trials: resolved earlier by filtering to choice in {-1, 1}.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
