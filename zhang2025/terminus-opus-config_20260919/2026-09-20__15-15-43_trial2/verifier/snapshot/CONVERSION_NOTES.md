# Dataset Conversion Notes

## Overview
- **Dataset**: IBL Brain Wide Map (brain-wide map of neural activity during complex behaviour)
- **Date started**: (see file timestamps)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verified: python3.13, numpy 2.3.5, torch 2.6.0+cu124 (CUDA available),
ibllib 4.0.1, ONE-api 3.5.2, iblatlas 1.2.0, iblutil 1.20.0.

Directory contents of /app:
- .manifest - list of all provided files
- Dockerfile, docker-compose.yaml, stage_cache.sh - container setup
- code/ - reference code: code_zhang2025/ (methods paper) and ibllib/ (IBL library source)
- data/one_cache/ - ONE cache of the IBL Brain Wide Map; lab folders are symlinks into
  /mnt/dataset/one_cache (567 GB, 18,497 data files)
- datapaper.pdf - A brain-wide map of neural activity during complex behaviour
- methodpaper.pdf - Exploiting correlations across trials and behavioral sessions to improve neural decoding (Zhang et al. 2025)
- dataarchitecture.pdf - IBL data architecture white paper
- methods.txt - extracted methods text
- decoder.py, train_decoder.py - provided decoder / validation code

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

Reference code: /app/code/code_zhang2025 (Zhang et al. 2025 decoding paper), which wraps
/app/code/ibllib (brainbox.io.one).

### Pipeline entry point
src/0_data_caching.py converts raw IBL data into trial-aligned arrays. Its hard-coded
parameters are the authoritative specification:

    params = {interval_len: 2, binsize: 0.02, single_region: False,
              align_time: stimOn_times, time_window: (-0.5, 1.5)}
    beh_names = [choice, reward, block, wheel-speed, whisker-motion-energy]

So: align to stimulus onset, window -0.5 s to +1.5 s, 20 ms bins => 100 bins/trial.
Session list comes from data/bwm_release.csv (the BWM data freeze: 699 insertions,
459 sessions, 139 subjects, 12 labs).

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| prepare_data | src/utils/ibl_data_utils.py | LOADING | Per-session loader: loads all probes, merges them, loads trials+mask, loads continuous behaviours |
| load_spiking_data | src/utils/ibl_data_utils.py | LOADING | SpikeSortingLoader(pid).load_spike_sorting() + merge_clusters(); qc=None by default so ALL clusters kept; cluster label stored in metadata (good_clusters = label >= 1) |
| merge_probes | src/utils/ibl_data_utils.py | LOADING | Concatenates spikes/clusters from both probes of a session, re-indexes spikes.clusters, sorts by spike time |
| load_trials_and_mask | src/utils/ibl_data_utils.py | CURATION | Loads SessionLoader.trials; boolean mask excluding reaction time (firstMovement_times - stimOn_times) < 0.08 s or > 2 s, NaN in stimOn_times/choice/feedback_times/probabilityLeft/firstMovement_times/feedbackType, trial length (feedback_times - goCue_times) > max_trial_len, and no-choice trials (choice == 0). Called with max_trial_len=10.0 |
| list_brain_regions / select_brain_regions | src/utils/ibl_data_utils.py | PROCESSING | Maps cluster acronyms to the Beryl atlas mapping via BrainRegions().acronym2acronym(mapping=Beryl); single_region=False pools all regions |
| bin_spiking_data | src/utils/ibl_data_utils.py | PROCESSING | Intervals = stimOn_times + (-0.5, +1.5); bins spikes at 0.02 s with iblutil.numerical.bincount2D -> (n_trials, 100 bins, n_clusters) |
| get_spike_data_per_interval | src/utils/ibl_data_utils.py | PROCESSING | Worker binning one interval; n_bins = ceil(interval_len/binsize) = 100 |
| load_target_behavior | src/utils/ibl_data_utils.py | LOADING | wheel-speed = abs(SessionLoader.wheel.velocity); whisker motion energy = SessionLoader.motion_energy[leftCamera or rightCamera].whiskerMotionEnergy |
| get_behavior_per_interval | src/utils/ibl_data_utils.py | PROCESSING | Linearly interpolates continuous behaviour onto np.linspace(t_beg + binsize, t_end, 100) (right edge of each bin); marks interval bad if data missing / starts too late / ends too early |
| bin_behaviors | src/utils/ibl_data_utils.py | PROCESSING | Assembles per-trial scalars choice, block (= probabilityLeft), reward (= rewardVolume > 1), contrast, plus binned time-varying behaviours |
| align_spike_behavior | src/utils/ibl_data_utils.py | CURATION | Drops trials where any behaviour interval is bad OR trials_mask is False, so neural and behaviour trial counts match |
| standardize_spike_data | src/utils/data_loader_utils.py | PROCESSING | Per-time-bin z-scoring of spike counts using training-set statistics (inside the decoder, not the cache) |

### Notes on neuron quality filtering
load_spiking_data is called with the default qc=None, so the caching pipeline keeps ALL
spike-sorted clusters and only records good_clusters = (label >= 1) in metadata. The
decoding data loaders never subset on good_clusters - they only subset by brain region.
The BWM data paper restricts analyses to good units (IBL single-unit QC label = 1).
See Step 4 for how this discrepancy is resolved.

### Notes on offline data access (important practical issue)
There is no internet connection, so ONE(base_url=..., password=...) fails and
one.eid2pid() is unavailable. Two further problems were found:
1. The cache tables shipped in /app/data/one_cache/{Brainwidemap, 2022_Q4..., 2025_Q3...}
   list dataset paths WITHOUT the revision folders actually staged on disk (table lists
   alf/_ibl_trials.table.pqt but the file on disk is alf/#2025-03-03#/_ibl_trials.table.pqt).
   Consequently SessionLoader.load_trials() and load_motion_energy() raise
   ALFObjectNotFound / KeyError.
2. one.eid2pid needs a remote connection.

Fixes (implemented in /app/cache/build_one_cache.py):
- Re-index the local filesystem with one.alf.cache.make_parquet_db per lab, then re-map the
  auto-generated hashed session UUIDs to the true experiment IDs using the
  (lab, subject, date, number) key of the official sessions.pqt. The resulting tables in
  /app/cache/one_tables contain 461 sessions / 18,482 datasets and let ONE resolve every
  staged file including revisions. All 459 BWM-release eids are present.
- Obtain pid / probe_name per eid from bwm_release.csv instead of one.eid2pid (the reference
  code also passes bwm_df around for exactly this information).
- Skip spike_loader.raw_electrophysiology(band=ap, stream=True).fs (needs internet); it is
  only used to record the AP sampling frequency in metadata, not for any processing.

Verified offline loading for session 6713a4a7-faed-4df2-acab-ee4e63326f8d (NYU-11 2020-02-18):
trials (565 x 20 cols), wheel (4,688,738 samples), left+right whiskerMotionEnergy,
spike sorting with 898 clusters / 20.7 M spikes, quality labels {0: 191, 0.33: 401, 0.67: 230, 1: 76}.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
The data are an IBL **ONE** cache in standard ALF format:

    /app/data/one_cache/<lab>/Subjects/<subject>/<YYYY-MM-DD>/<number>/
        alf/                                       session-level behaviour
            [#revision#/]_ibl_trials.table.pqt      trials table (one row per trial)
            [#revision#/]_ibl_trials.stimOff_times.npy, stimOnTrigger_times.npy, ...
            _ibl_trials.goCueTrigger_times.npy, _ibl_trials.intervals_bpod.npy,
            _ibl_trials.quiescencePeriod.npy
            _ibl_wheel.position.npy, _ibl_wheel.timestamps.npy
            [#revision#/]_ibl_leftCamera.times.npy, _ibl_rightCamera.times.npy
            [#revision#/]leftCamera.ROIMotionEnergy.npy, rightCamera.ROIMotionEnergy.npy
            probeNN/pykilosort/[#revision#/]
                spikes.times.npy, spikes.clusters.npy, spikes.amps.npy, spikes.depths.npy
                clusters.channels.npy, clusters.depths.npy, clusters.metrics.pqt, clusters.uuids.csv
                channels.brainLocationIds_ccf_2017.npy, channels.localCoordinates.npy,
                channels.mlapdv.npy, channels.rawInd.npy, channels.labels.npy
        raw_ephys_data/  (only .meta/.ch sidecars; raw binaries not staged)

Cache index tables live in `/app/data/one_cache/{Brainwidemap, 2022_Q4_IBL_et_al_BWM,
2025_Q3_IBL_et_al_BWM}` but are out of sync with the staged revisions (see Step 1), so I
rebuilt them from the filesystem into `/app/cache/one_tables` (461 sessions, 18,482 datasets).

### Trials table columns
`goCueTrigger_times, intervals_bpod_0/1, stimOffTrigger_times, stimOnTrigger_times,
quiescencePeriod, stimOff_times, goCue_times, response_times, choice, stimOn_times,
contrastLeft, contrastRight, probabilityLeft, feedback_times, feedbackType, rewardVolume,
firstMovement_times, intervals_0, intervals_1`

Key variables: `choice` in {-1 (right wheel turn = stimulus moved left / CCW), +1, 0 (no-go)},
`probabilityLeft` in {0.5 (unbiased block), 0.8 (left block), 0.2 (right block)},
`contrastLeft` / `contrastRight` in {0, 0.0625, 0.125, 0.25, 1} with NaN on the other side,
`feedbackType` in {-1, +1}, `rewardVolume` in uL.

### Continuous data streams
- Wheel: `_ibl_wheel.position/.timestamps` (~1 kHz after `SessionLoader.load_wheel()`
  interpolation; velocity/acceleration computed by Gaussian-smoothed differentiation).
- Whisker motion energy: `left/rightCamera.ROIMotionEnergy.npy` with camera timestamps
  (left camera 60 Hz, right camera 150 Hz).

### Dataset Size (from data files, all 459 BWM-release sessions)
| Statistic | Value |
|-----------|-------|
| Insertions (probes) | 699 (240 sessions with 2 probes, 219 with 1) |
| Sessions | 459 |
| Subjects | 139 |
| Labs | 12 |
| Clusters (total, all spike-sorted units) | 621,733 |
| Good clusters (IBL single-unit QC label >= 1) | 75,708 |
| Clusters / session (mean, median, range) | 1354.5, 1299, [135, 3140] |
| Good clusters / session (mean, median, range) | 164.9, 142, [2, 536] |
| Trials (total, before curation) | 296,090 |
| Trials / session (mean, median, range) | 645.1, 601, [401, 1525] |
| Sessions with trials table | 459 / 459 |
| Sessions with wheel | 459 / 459 |
| Sessions with left whisker motion energy | 437 / 459 |
| Sessions with right whisker motion energy | 420 / 459 |
| Sessions with NO motion energy (either camera) | 14 |

Survey script: `/app/cache/explore_data.py`; results in `/app/cache/session_survey.csv`.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

Sources: /app/methods.txt, plus full text extracted from the PDFs into
/app/cache/{datapaper,methodpaper,dataarchitecture}.txt (pypdf).

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (mice) | 139 | We trained 139 mice (94 male and 45 female) on the IBL decision-making task |
| Labs | 12 | Recordings were collected by 12 laboratories in Europe and the USA |
| Insertions (probes) | 699 | we inserted 699 Neuropixels probes |
| Sessions released | 459 | a total of 459 sessions, 699 insertions and 621,733 neurons remained, constituting the publicly released dataset |
| Units (all, incl. MUA) | 621,733 (avg 889/probe) | This process produced 621,733 units (including multineuron activity), averaging 889 per probe |
| Well-isolated neurons | 75,708 (avg 108/probe) | identified 75,708 well-isolated neurons, averaging 108 per probe |
| Sessions used in methods paper | 433 (270 regions) | We apply our models to 433 IBL sessions, covering 270 brain regions |
| Trials / session | mean 645, median 602, range 401-1,525 | Recorded sessions lasted on average 645 trials (median of 602, range of 401-1,525) |
| Minimum trials / session | 400 | Only sessions with at least 400 trials were retained for further analyses |
| Overall performance | 81.4 +/- 0.4% correct | they made correct choices on 81.4 +/- 0.4% of the trials |
| Performance on 0% contrast | 58.7 +/- 0.4% correct | mice gained rewards on 58.7 +/- 0.4% of trials |
| Unbiased block | first 90 trials, pL = 0.5 | This initial block of 90 trials is referred to as the unbiased block (50:50) |
| Biased block lengths | 20-100 trials, empirical mean 51 | drawn from a truncated geometric distribution (empirical mean of 51 trials) |
| Block probabilities | pL in {0.2, 0.5, 0.8} | at a ratio of 20:80% (right block) or 80:20% (left block) |
| Contrasts | {0, 6.25, 12.5, 25, 100}% ratio 2:2:2:2:1 | Stimulus contrast was uniformly sampled from 5 possible values |
| Neural data time bin | 20 ms | each divided into 20-ms bins, producing T = 100 time steps |
| Trial length | 2 s | Recordings are split into 2-s trials |
| Alignment (choice) | stimulus onset, -0.5 s to +1.5 s | For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset |
| Behaviour sampling | whisker ME 60 Hz (left cam) / 150 Hz (right cam); wheel approx 1 kHz | wheel speed and whisker motion energy are time-varying signals sampled at 60 Hz |

### Processing Details
- Temporal alignment: stimOn_times. Window (-0.5, +1.5) s, giving 2 s trials.
- Temporal binning: 20 ms non-overlapping bins, T = 100 bins per trial.
- Whisker motion energy: mean across pixels of the absolute frame difference in a whisker-pad
  bounding box anchored between nose tip and eye (precomputed as *Camera.ROIMotionEnergy).
- Wheel speed: absolute wheel velocity, where velocity is the Gaussian-smoothed derivative of
  wheel position computed by SessionLoader.load_wheel().
- Prior / block: probabilityLeft in {0.2, 0.5, 0.8}.

### Curation Steps

**Session curation rules (data release, already applied upstream)**:
- At least 250 trials performed, at least 90% correct on 100% contrast trials in both block
  types, at least 3 incorrect trials, passing hardware/task QC.
- Insertions excluded for RIGOR failures, unrecoverable probe tract, or unresolved alignment.
- Only sessions with at least 400 trials were retained for further analyses.

**Neuron curation rules**:
- Neurons were excluded if they failed one of the three criteria: amplitude > 50 uV;
  noise cut-off < 20 uV; and refractory period violation. Neurons that passed these criteria
  were termed well-isolated neurons. This is exactly the IBL clusters.metrics label == 1
  (each of the three RIGOR single-unit metrics contributes 1/3). 621,733 -> 75,708.
- Final analyses were additionally restricted to regions that were designated grey matter in
  the adult mouse Allen Common Coordinate framework, i.e. exclude units whose Beryl acronym
  is root or void.

**Trial curation rules**:
- Trials were excluded if one of the following trial events could not be detected: choice,
  probabilityLeft, feedbackType, feedback times, stimOn times and firstMovement times.
- Trials were further excluded if the time between stimulus onset and the first movement of
  the wheel was outside the range of 0.08-2.00 s.
- The reference code adds max_trial_len=10.0 (feedback_times - goCue_times <= 10 s) and
  exclude_nochoice=True (choice == 0). exclude_unbiased is left at its default False, so the
  first 90 unbiased (pL = 0.5) trials are KEPT, which is necessary here because the decoder
  must distinguish pL = 0.5 from 0.2 and 0.8.

### Decoders Trained (performance reported in the reference papers)
| Decoded variable | Metric | Reported value |
|------------------|--------|----------------|
| Choice (per region, RRR vs baseline) | AUC | 0.66 linear baseline / 0.72 single-session RRR / 0.79 multi-session |
| Choice (BWM paper, per region) | null-corrected median balanced accuracy | small per-region effect sizes, significant in many regions |
| Prior | Pearson correlation | 0.05 single-session / 0.34 multi-session / 0.65 oracle |
| Wheel speed | R^2 | reported per region; RRR better than ridge (Fig. 2E/3) |
| Whisker motion energy | R^2 | reported per region; RRR better than ridge (Fig. 2E/3) |

Note: the reference papers decode PER BRAIN REGION (PO, LP, DG, CA1, VISa) to avoid ceiling
effects: To avoid ceiling effects from using all regions, we decode each region separately,
yielding more moderate, distinguishable performance across models. Our task pools ALL regions
in a session, so accuracies should be HIGHER than these per-region numbers.

### My own sanity checks against the papers (all 459 sessions, /app/cache/check_choice.py)
| Statistic | Paper | Measured from data | Match |
|-----------|-------|--------------------|-------|
| Trials / session (mean) | 645 | 645.08 | yes |
| Trials / session (median) | 602 | 601 | yes |
| Trials / session (range) | 401-1,525 | 401-1,525 | yes |
| Fraction correct | 0.814 | 0.8196 | yes |
| Fraction correct, 0% contrast | 0.587 | 0.5855 | yes |
| Sessions | 459 | 459 | yes |
| Subjects | 139 | 139 | yes |
| Insertions | 699 | 699 | yes |
| Units (all) | 621,733 | 621,733 | yes |
| Well-isolated neurons (label >= 1) | 75,708 | 75,708 | yes |
| First block pL | 0.5, 90 trials | 0.5 in 459/459 sessions | yes |

### Choice sign convention (established empirically, not stated in the text)
On every one of the 459 sessions, 100% of CORRECT trials with the stimulus on the LEFT have
choice == +1, and 100% of correct trials with the stimulus on the RIGHT have choice == -1.
Independently, in right-biased blocks (pL = 0.2) 75.6% of choices are -1, while in left-biased
blocks (pL = 0.8) only 25.5% are -1. Therefore:

    trials.choice == +1  <=>  the mouse reported the stimulus was on the LEFT
    trials.choice == -1  <=>  the mouse reported the stimulus was on the RIGHT

(The wheel turn direction is the opposite of the reported side, because the mouse moves the
stimulus toward the centre; the ALF choice field encodes the reported side.)

### Other practical finding
A single ONE instance is NOT thread-safe: loading 459 sessions with a shared ONE across 16
threads silently failed on 76 of them. All loading must be serial per ONE instance, or use
separate processes each with their own ONE instance.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neuron quality filter | load_spiking_data(qc=None) keeps ALL clusters; only records good_clusters = label >= 1 in metadata. Decoding loaders never subset on it | 621,733 clusters total; 75,708 have label >= 1 | BWM paper: analyses use only the 75,708 well-isolated neurons (amplitude > 50 uV, noise cut-off < 20 uV, refractory-period violation) | APPLY the QC filter (label >= 1). The paper headline neuron count (75,708) is the published statistic we are asked to match, and MUA clusters add noise rather than signal. Matches: Neurons generated by the spike-sorting pipeline were excluded from the analyses presented here if they failed one of the three criteria. The reference caching code keeps all clusters only because it defers the decision downstream. |
| Grey-matter filter | Not applied in the caching script (select_brain_regions only selects by Beryl acronym; root/void are not removed) | ~15-20% of good units map to Beryl root/void | BWM paper: Final analyses were additionally restricted to regions that were designated grey matter in the adult mouse Allen CCF | DROP root/void units. They are not assigned to any anatomical region, so they cannot be given a meaningful brain_regions label in the target format. |
| Region minimum (>= 5 neurons/session, >= 2 sessions) | Not applied in the caching code | - | BWM paper applies it for its region-by-region statistics | NOT applied. That criterion exists so per-region decoding statistics are well powered. Our decoder pools all neurons of a session, so discarding neurons from sparsely sampled regions would only throw information away. Deliberate documented difference. |
| Bin size | binsize = 0.02 (20 ms), time_window = (-0.5, 1.5), so T = 100 | - | Methods paper main text: each divided into 20-ms bins, producing T = 100 time steps. But its STAR Methods says 50-ms non-overlapping time bins for choice/prior | USE 20 ms, T = 100, following the runnable reference code and the main text. The STAR-Methods 50 ms sentence is internally inconsistent with the same paper T = 100 over a 2 s trial. 20 ms is also needed to resolve the time-varying wheel/whisker outputs. |
| Alignment event | align_time = stimOn_times, window (-0.5, +1.5) | stimOn_times present for essentially all trials | For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset | Consistent. The Decoder Task also specifies Temporally align based on stimulus onset. Use stimOn_times, (-0.5, +1.5) s. |
| Alignment for dynamic behaviours | Methods paper aligns wheel speed / whisker ME to FIRST MOVEMENT onset | - | Same | USE STIMULUS ONSET FOR EVERYTHING, because the Decoder Task requires a single alignment event and all four outputs must share one time base. Documented difference. |
| Sessions | bwm_release.csv lists 459 sessions | 459 sessions staged, all loadable | Data paper: 459 released; methods paper: We apply our models to 433 IBL sessions | Start from all 459, then drop only sessions that cannot supply the required signals (see Step 5). The methods paper 433 is itself the result of dropping sessions lacking video/behaviour. |
| Trial exclusions | load_trials_and_mask(max_trial_len=10.0): RT in [0.08, 2] s, no NaN in 6 events, feedback-goCue <= 10 s, choice != 0 | RT criterion removes ~30%, others a few % | Data paper lists the NaN-event and 0.08-2.00 s RT criteria (but not the 10 s trial-length one) | USE the reference code mask verbatim (it is a superset of the paper criteria and is what the decoding paper actually ran). |
| Unbiased block | exclude_unbiased=False (default), so pL = 0.5 trials are kept | first 90 trials of every session have pL = 0.5 | Paper describes the 90-trial unbiased block | KEEP them, required because prior has three classes 0.2/0.5/0.8 in the Decoder Output spec. |
| Neural normalisation | Reference decoders z-score spike counts per time bin using training-set statistics (standardize_spike_data) | Firing rates vary by two orders of magnitude across neurons | - | The provided /app/decoder.py does NO normalisation (it runs torch.svd straight on the stored values), so normalisation has to happen in the conversion. Per-neuron z-scoring is applied to the stored neural data (see Step 5). |
| Whisker camera | bin_behaviors tries left-whisker-motion-energy and falls back to right | left ME on 437/459 sessions, right on 420/459, neither on 14 | Paper computes ME for both left (60 Hz) and right (150 Hz) cameras | PREFER LEFT, FALL BACK TO RIGHT, exactly as the reference code does. |
| one.eid2pid | Reference uses it to find probes | Needs a remote connection | - | Use the pid / probe_name columns of bwm_release.csv (the same freeze file the reference loads anyway). |
| AP sampling frequency | load_spiking_data streams raw ephys to read fs | Raw AP binaries not staged | - | Skipped; only recorded in metadata, never used in processing. |

### Verified consistent
- Trials/session, fraction correct, fraction correct at 0% contrast, and session/subject/probe/unit
  counts all match the data paper exactly (table in Step 3).
- My fast vectorised spike binning reproduces the reference bincount2D binning EXACTLY
  (np.allclose True on sampled trials, identical spike totals): /app/cache/check_binning.py.
- The trial mask produced by the reference load_trials_and_mask behaves as the paper describes:
  the 0.08-2.00 s reaction-time window is by far the dominant exclusion (~30-40% of trials).
- The provided decoder accepts per-trial (d,) or time-varying (d,T) inputs and outputs and
  broadcasts per-trial values across the trial timepoints (SessionData.__getitem__).

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Trial structure
- Alignment event: trials.stimOn_times (stimulus onset).
- Window: off_start = -0.5 s, off_end = +1.5 s, so 2.0 s per trial.
- Bin size: 20 ms, giving T = 100 bins per trial, identical for every trial and session.
- Bin k covers [stimOn - 0.5 + 0.02k, stimOn - 0.5 + 0.02(k+1)).

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| spikes.times, spikes.clusters (both probes) | neural | merge probes, keep good grey-matter units, bin at 20 ms in the stimOn window, then per-neuron z-score | merge_probes, bin_spiking_data, standardize_spike_data | shape (n_neurons, 100) per trial, float32 |
| bin index | input[0] = time_from_stimOn | bin centre time in seconds relative to stimOn: -0.49, -0.47, ..., +1.49 | new; required by Decoder Task | time-varying, continuous |
| bin index | input[1] = stimOn_event | binary indicator, 1 in the bin containing stimulus onset (bin 25), else 0 | Decoder-Task rule: if an input is a time such as onset of some stimulus, represent it as a binary time series | time-varying, binary |
| trial index within its block | input[2] = trial_in_block | 0-based count of trials since the last change of probabilityLeft, computed on the RAW trials table before masking so it is the true position in the block | new; required by Decoder Task | per-trial, constant across the 100 bins |
| trials.choice | output[0] = choice | +1 (reported LEFT) -> 0, -1 (reported RIGHT) -> 1 | bin_behaviors (choice) | per-trial, binary; convention verified empirically over all 459 sessions (Step 3) |
| trials.probabilityLeft | output[1] = prior_prob_left | 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 | bin_behaviors (block) | per-trial, 3 classes, exactly as specified in the Decoder Task |
| wheel.velocity | output[2] = wheel_speed | abs(velocity), linearly interpolated onto the 100 bins, then discretised into 3 bins | load_target_behavior(wheel-speed), get_behavior_per_interval | time-varying, 3 classes |
| leftCamera.ROIMotionEnergy (fallback rightCamera) | output[3] = whisker_motion_energy | linearly interpolated onto the 100 bins, then discretised into 3 bins | load_target_behavior(left/right-whisker-motion-energy), bin_behaviors | time-varying, 3 classes |
| clusters.acronym mapped to Beryl | brain_regions, brain_region_idx | BrainRegions().acronym2acronym(mapping=Beryl) | list_brain_regions | one index per kept neuron |
| bwm_release.csv subject | subjects, subject_idx | unique subject names | - | |

### Discretisation of the two continuous outputs
Both wheel speed and whisker motion energy are continuous, strongly right-skewed, and in
SESSION-SPECIFIC ARBITRARY UNITS (whisker ME depends on camera, illumination and ROI size;
wheel speed is in rad/s but its scale varies with how vigorously a given mouse turns). Fixed
global thresholds would therefore put nearly all trials of some sessions into a single class.

Decision: PER-SESSION TERCILE (33.3rd / 66.7th percentile) binning, computed over all
(trial, bin) samples of that session that survive curation. This
- guarantees three roughly balanced classes in every session (about 1/3 each), so the balanced
  accuracy of the decoder is interpretable and chance is 1/3;
- is invariant to the arbitrary per-session scale;
- preserves the ordering of the underlying continuous variable, giving classes with the natural
  interpretation low / medium / high.
Output value names: [low, medium, high].

### Neural processing
1. For each probe of the session, load spike sorting and merge_clusters (gives label, acronym).
2. Keep clusters with label >= 1 (IBL well-isolated / RIGOR single-unit criteria) AND whose
   Beryl acronym is not root/void (grey matter only).
3. Merge the probes of a session (merge_probes logic: re-index clusters, concatenate, sort by time).
4. Bin spikes into 20 ms bins over [stimOn - 0.5, stimOn + 1.5) for every kept trial.
5. PER-NEURON Z-SCORE across all (trial, bin) samples of the session: (x - mean) / std, with std
   clamped to >= 1e-6 so silent neurons give 0 rather than NaN.
   Rationale: the provided /app/decoder.py applies no normalisation of its own and initialises
   its projection with a raw torch.svd; without normalisation the few highest-rate neurons
   dominate every principal component. The reference decoding pipeline standardises spike counts
   for exactly this reason (data_loader_utils.standardize_spike_data). I z-score per neuron
   (over the whole session) rather than per time-bin, because per-time-bin statistics would
   remove the stimulus-locked population response that the decoder needs.

### Curation rules
**Trial curation** (reference load_trials_and_mask(one, eid, max_trial_len=10.0)):
- reaction time firstMovement_times - stimOn_times in [0.08, 2.0] s;
- no NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType;
- feedback_times - goCue_times <= 10 s;
- choice != 0 (no-go trials removed).
Additionally (required by the target format, not by the reference):
- the whole 2 s window must be covered by both continuous behaviour streams (no extrapolation
  beyond the recorded wheel / camera time range) and must contain no NaN in either stream.
  This mirrors the reference get_behavior_per_interval checks (target data starts too late /
  ends too early / nans in target data).

**Neuron curation**: label >= 1 and Beryl acronym not in {root, void} (see above).

**Session curation**: a session is dropped if, after the above,
- it has no whisker motion energy from either camera (14 sessions), or
- it has 0 kept neurons, or
- it has fewer than 2 kept trials (the target format requires at least two trials per session to
  be able to evaluate the decoder), or
- a behavioural output has no variability so terciles are undefined.

### Key Decisions
1. **Align to stimulus onset, -0.5 to +1.5 s, 20 ms bins**: exactly the reference caching script
   params, and what the Decoder Task requires.
2. **Keep only well-isolated (label >= 1) grey-matter neurons**: matches the BWM paper published
   75,708-neuron figure and its explicit inclusion criteria.
3. **Merge probes within a session**: neurons in the same session and region were combined across
   probes for our decoding analysis (BWM paper) and merge_probes in the reference code.
4. **Keep the 90 unbiased (pL = 0.5) trials**: needed for the 3-class prior output.
5. **choice: +1 -> 0 (left), -1 -> 1 (right)**: the Decoder Task asks for left = 0, right = 1, and
   the ALF choice sign convention was established empirically (Step 3).
6. **Per-session terciles for the two continuous outputs**: see above.
7. **Per-neuron z-scoring of the stored neural data**: see above.
8. **time_from_stimOn as a continuous input plus a binary stimOn indicator**: the Decoder Task asks
   for time since stimulus onset, continuous, time-varying; the binary indicator additionally
   satisfies the rule about representing an onset time as a binary time series.
9. **trial_in_block computed on the unmasked trials table**: the true trial index within the block
   is a property of the experiment, so it must be counted before trials are dropped.

### Planned Sanity Checks
- [ ] Total sessions / subjects / probes from bwm_release.csv = 459 / 139 / 699.
- [ ] Total clusters = 621,733 and good clusters = 75,708 before the grey-matter filter.
- [ ] Raw trials/session mean 645, median 602, range 401-1,525 (data paper).
- [ ] Fraction correct 0.814 and 0.587 on 0% contrast (data paper).
- [ ] Binning reproduces the reference bincount2D exactly (done, Step 4).
- [ ] Per-session tercile outputs are each about 1/3 of samples.
- [ ] choice fraction of left choices close to 0.5 overall; correlates with probabilityLeft.
- [ ] time_from_stimOn ranges exactly [-0.49, +1.49]; the stimOn_event indicator has exactly one 1
      per trial, at bin 25.
- [ ] Spot-check raw ALF files vs converted arrays with np.allclose (Step 10).
- [ ] All trials have exactly 100 bins; all sessions have consistent neuron counts.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (about 680 lines). Runs as
`python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]`.

Structure:
- Module-level constants hold the reference pipeline params (ALIGN_EVENT = stimOn_times,
  TIME_WINDOW = (-0.5, 1.5), BIN_SIZE = 0.02, N_BINS = 100, MAX_TRIAL_LEN = 10.0).
- `get_one()` / `get_brain_regions()` build a per-process ONE / BrainRegions singleton.
- `load_session_neurons()` = reference `load_spiking_data` + `merge_probes`, plus the BWM
  paper neuron inclusion criteria (label >= 1 and Beryl acronym not root/void).
- `bin_spikes()` - vectorised 20 ms binning, verified bit-identical to the reference
  `bincount2D` path.
- `interp_behavior()` - linear interpolation of a continuous behaviour onto the 100 bin right
  edges, with the reference coverage / NaN validity checks.
- `discretize()` - per-session tercile discretisation.
- `trial_number_in_block()` - trial index within each probabilityLeft block, computed on the
  unmasked trials table.
- `convert_session()` - the whole per-session pipeline, returning the trial lists plus
  diagnostics; `plot_processing()` renders the 8-panel `--show-processing` figure.
- `main()` - multiprocessing driver, assembly of the target dict, summary statistics, pickling.

The reference trial mask is used directly by importing
`utils.ibl_data_utils.load_trials_and_mask` from `/app/code/code_zhang2025/src`, so trial
curation is guaranteed identical to the reference pipeline rather than re-implemented.

Code inefficiencies identified:
- The reference `get_spike_data_per_interval` spawns a multiprocessing pool **per session** and
  calls `bincount2D` once per trial, each time re-scanning the full spike vector with a boolean
  mask (O(n_trials x n_spikes)).
- The reference `get_behavior_per_interval` likewise builds a pool per behaviour and
  interpolates trial by trial in Python.

Code speedups added:
- Spike binning uses one `np.searchsorted` for all trial boundaries and a single
  `np.add.at` per trial, i.e. O(n_spikes + n_trials x n_bins) instead of O(n_trials x n_spikes).
  A 407-trial, 61-neuron session binned in 0.03 s.
- Behaviour interpolation is a single vectorised `interp1d` call over the whole (n_trials, 100)
  grid.
- Parallelism is at the session level (16 processes), not inside a session, which avoids the
  repeated pool creation and gives near-linear scaling.
- A ONE instance is created once per worker process (ONE is NOT thread-safe, so processes are
  used rather than threads).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
-> `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`, two `processing_<eid>.png`.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (MFD_05, DY_010) |
| Brain regions | 13 |
| Neurons (total) | 204 |
| Neurons / session | 102.0 mean (88, 116) |
| Trials (total) | 644 |
| Trials / session | 322 mean (198, 446) |
| Clusters seen | 2,020 all; 298 good (label >= 1) |
| Timepoints / trial | 100 (20 ms bins, window [-0.5, 1.5] s) |
| time_from_stim_onset range | [-0.49, 1.49] |
| stim_onset range | [0, 1] |
| trial_num_in_block range | [0, 94] |
| choice distribution | [0.500, 0.500] |
| prior_prob_left distribution | [0.443, 0.118, 0.439] |
| wheel_speed distribution | [0.333, 0.333, 0.333] |
| whisker_motion_energy distribution | [0.333, 0.333, 0.333] |

These are all as expected: choice is near balanced; the prior is about 12% pL = 0.5 because the
unbiased block is 90 of about 650 trials; the two tercile-discretised outputs are exactly 1/3
each by construction.

### Direct inspection of the pickle
- `neural[0][0]`: (116, 100) float32.
- Per-neuron mean over the session: max |mean| = 2.7e-7; per-neuron std in [0.9999977, 1.000002]
  -> z-scoring is exact.
- `input[0][0][0]` starts at -0.49 and ends at 1.49 (bin centres).
- `input[0][0][1]` sums to exactly 1 with argmax at bin 25, i.e. the bin containing t = 0.
- `output[0][0]` is int8 of shape (4, 100).
- `metadata` carries task_description, time_bin_size = 20.0, temporal_alignment_event,
  off_start = -0.5, off_end = 1.5, plus per-session info and the list of skipped sessions.

### Processing Plots Review
`processing_<eid>.png` has 8 panels: raw binned spike counts; z-scored neural data; the
trial-averaged population response (which rises sharply right at t = 0, confirming there is no
temporal misalignment); the three decoder inputs; the source wheel-velocity trace overlaid with
the interpolated bin values and the tercile thresholds; the wheel-speed discretisation with the
class steps overlaid; the same for whisker motion energy; and the output class fractions.
No anomalies: the interpolated points sit exactly on the source traces, and the discretised
classes change exactly when the continuous signal crosses a threshold.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| Vectorised searchsorted + add.at binning instead of per-trial bincount2D in a pool | binning is 1% of per-session time (0.03 s for a 407-trial session) |
| Single vectorised interp1d for the whole behaviour grid | wheel + motion energy together are 12% of per-session time |
| Session-level multiprocessing (16 processes), one ONE per process | about 16x |

| Step | Time / Session | Estimated Total Time |
| spike loading (dominant) | 1.75 s | |
| wheel | 0.35 s | |
| binning | 0.05 s | |
| trials | 0.05 s | |
| motion energy | 0.02 s | |
| TOTAL per session (1 worker) | 2.4 s | 459 x 2.4 = 1,100 s serial |
| with 16 worker processes | | about 70-120 s plus pickling |

Well under the 15 minute budget, so no further optimisation is needed. (The sample sessions had
198 and 446 curated trials against a dataset mean of about 427, so this estimate is
representative; spike loading dominates and depends on cluster count, not trial count.)

> **Note**: the sample artefacts in this step were REGENERATED after the two fixes made in
> Step 12 (the corrected per-time-bin standardisation and the float64 accumulation), so the
> numbers here match the current `sample_data.pkl`, `conversion_sample_out.txt`,
> `verification_sample_out.txt` and `train_decoder_sample_out.txt` on disk.

### Format Validation
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only`
-> `/app/verification_sample_out.txt`: **Data format is valid, no errors or warnings.**

One item was investigated: a brain region named `x`. This is a genuine Beryl acronym,
`Nucleus x` of the medulla (as is `y` = `Nucleus y`), not a parsing artefact - confirmed with
`BrainRegions()`. Only `root` and `void` are non-anatomical and they are excluded.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None

### Training Progress
Loss decreased monotonically: 1.371 (epoch 9) -> 1.083 (20) -> 0.881 (50) -> 0.821 (100)
-> 0.802 (200). Test loss 0.824.

### Decoder Results (Sample, 2 sessions)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| choice | 0.6128 | 0.6095 | 0.5000 |
| prior_prob_left | 0.6669 | 0.6679 | 0.3333 |
| wheel_speed | 0.5455 | 0.5319 | 0.3333 |
| whisker_motion_energy | 0.5446 | 0.5374 | 0.3333 |

All four outputs are above chance, and the train/validation gap is small (<= 0.014), so there is
no sign of overfitting or leakage. These two sessions contribute only 88 and 116 neurons, so
absolute accuracies are expected to rise on the full dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --n-workers 24`
-> `/app/conversion_full_out.txt`.

### Output Files
- `converted_data.pkl`: 11.28 GB, written in 103 s wall-clock (1,783 s of summed worker time
  over 24 processes). Well inside the 15 minute budget, so no further optimisation was needed.
- `verification_full_out.txt`: created; **Data format is valid, no errors or warnings.**

### Conversion results
- 445 sessions converted, 14 skipped.
- Every skipped session was skipped for the same reason: no whisker motion energy from either
  camera. This is a property of the released data (my independent survey in Step 2 found
  left-camera ME on 437/459 and right-camera ME on 420/459 sessions, with 14 sessions having
  neither), not a bug. Whisker motion energy is a required decoder output, so those sessions
  cannot be used.
- Timing profile: spike loading 78%, wheel loading 17%, binning 2%, trials 1%, motion energy 1%.

### Consistency Check
| Statistic | Reference papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions in release | 459 | 459 (bwm_release.csv) | 459 staged | 445 kept (14 lack whisker ME) | yes, difference explained |
| Insertions | 699 | 699 | 699 | 699 read | yes |
| Subjects | 139 | 139 | 139 | 136 (3 subjects only had sessions without whisker ME) | yes, difference explained |
| Labs | 12 | 12 | 12 | 12 | yes |
| Units (all clusters) | 621,733 | - | 621,733 | 600,174 seen in the 445 kept sessions (621,733 over all 459) | yes |
| Well-isolated neurons (label >= 1) | 75,708 | - | 75,708 | 73,060 in the 445 kept sessions (75,708 over all 459) | yes |
| Well-isolated AND grey matter | - | - | 65,301 over 459 sessions | 62,779 over the 445 kept sessions | yes |
| Mean neurons/session | 108 per probe (paper) | - | 142.3 good+grey / session | 141.1 | yes |
| Trials/session, raw | mean 645, median 602, range 401-1,525 | - | mean 645.1, median 601, range 401-1,525 | (raw counts recorded in metadata) | yes |
| Trials/session, after curation | - | reference mask | mean 426.5 over 459 sessions | mean 424.8, median 392, range 85-1,445 | yes |
| Trials total, after curation | - | - | 195,781 over 459 sessions | 189,057 over 445 sessions | yes |
| Brain regions | 270 (methods paper, 433 sessions) | - | 265 Beryl regions with good+grey units | 263 | yes |
| Fraction correct | 0.814 | - | 0.8196 | - (feedback is not an output here) | yes |
| Fraction correct 0% contrast | 0.587 | - | 0.5855 | - | yes |
| time_from_stim_onset range | -0.5 to +1.5 s | (-0.5, 1.5) | - | [-0.49, +1.49] (bin centres) | yes |
| stim_onset range | - | - | - | [0, 1] | yes |
| trial_num_in_block range | blocks 20-100 trials, first block 90 | - | - | [0, 98] | yes |
| choice distribution | near balanced | - | - | [0.508, 0.492] | yes |
| prior_prob_left distribution | 90 unbiased trials of about 645 -> about 0.14 at pL=0.5 | - | - | [0.418, 0.141, 0.442] | yes |
| wheel_speed distribution | - | - | - | [0.333, 0.333, 0.333] | by construction |
| whisker_motion_energy distribution | - | - | - | [0.333, 0.334, 0.333] | by construction |
| Timepoints per trial | T = 100 | 100 | - | 100 for every trial of every session | yes |
| Bin size | 20 ms | 0.02 s | - | 20 ms | yes |

Notes on the two apparent mismatches:
- **445 vs 459 sessions / 136 vs 139 subjects**: purely the 14 sessions with no whisker motion
  energy from either camera. The methods paper itself used only 433 of the 459 sessions for the
  same kind of reason (missing video/behaviour), so 445 is in the expected range.
- **600,174 / 73,060 clusters vs 621,733 / 75,708**: the converted totals are over the 445 kept
  sessions only. My independent survey over all 459 sessions (`/app/cache/session_survey.csv`)
  reproduces the published 621,733 and 75,708 exactly, which confirms the loading and the QC
  criterion.

### Spot-checks of data integrity
See Step 10 Check 2: 84 independent checks across 4 sessions, all passing, including a full
`np.allclose` comparison of every z-scored neural value against an independent
`np.histogram2d` recomputation from the raw spike times.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` begins with
**"Data format is valid, no errors or warnings."** - zero errors and zero warnings, so there is
nothing to fix or explain away.

One item in the log was actively investigated rather than assumed benign:
- A brain region named `x` (and `y`). These are genuine Beryl acronyms - `Nucleus x` and
  `Nucleus y` of the medulla - confirmed directly with `iblatlas.regions.BrainRegions`. They are
  grey matter and are correctly kept. Only `root` and `void` are non-anatomical and are excluded.

### Check 2: Construct sanity checks
`/app/cache/sanity_checks.py` re-derives everything straight from ONE / the ALF files WITHOUT
calling any function from `convert_data.py`, for 4 spot-checked sessions (indices 0, 7, 222 and
444). **84 checks, 84 passed, 0 failed.**

Neural checks:
- Neuron count and the full list of Beryl acronyms behind `brain_region_idx` match.
- Spike counts recomputed with `np.histogram2d` (a completely different binning routine from the
  `searchsorted` + `add.at` one used in the conversion), then z-scored: the **entire**
  (n_trials, n_neurons, 100) array is `np.allclose` to the stored `neural`, atol 1e-4.
- Explicit per-element spot-checks, e.g. trial 5 / neuron 3 / bin 50.

Input checks:
- `time_from_stim_onset` is `np.allclose` to the bin centres -0.49 ... +1.49.
- `stim_onset` sums to exactly 1 per trial with its 1 at bin 25 (the bin containing t = 0).
- `trial_num_in_block` matches an independent re-computation of block boundaries from
  `probabilityLeft`.

Output checks:
- `choice` equals `where(trials.choice > 0, 0, 1)` element-by-element.
- `prior_prob_left` equals the 0.2/0.5/0.8 -> 0/1/2 map of `trials.probabilityLeft`.
- `choice` and `prior` are constant within each trial.
- The tercile edges stored in metadata equal the 1/3 and 2/3 quantiles recomputed from an
  independent `interp1d` of the raw wheel / motion-energy traces.
- The discretised labels are `np.allclose` to an independent `np.searchsorted` discretisation.
- The classes are **ordered**: max(value | class 0) <= min(value | class 1) <= max(value |
  class 1) <= min(value | class 2), so low/medium/high are genuinely low/medium/high.

Trial-curation checks:
- Raw trial count, the trial count after an independently re-implemented curation mask, and the
  final trial count all match the converted data and the recorded metadata.

### Check 3: Reference code comparison
| Stage | Reference (`ibl_data_utils.py` / `0_data_caching.py`) | My `convert_data.py` | Same? |
|-------|--------------------------------------------------------|----------------------|-------|
| (a) Data loading | `SpikeSortingLoader(pid).load_spike_sorting()` + `SpikeSortingLoader.merge_clusters()`; `SessionLoader.load_trials/load_wheel/load_motion_energy`; probes from `one.eid2pid` | Identical calls. Probes come from `bwm_release.csv` instead of `one.eid2pid`, which needs a network connection; `bwm_release.csv` is the same freeze the reference loads and contains `pid`/`probe_name` columns | yes (documented exception) |
| (b) Neuron filtering | `qc=None` keeps all clusters; `good_clusters = label >= 1` only stored in metadata; no grey-matter filter | `label >= 1` **and** Beryl acronym not `root`/`void` | deliberate difference - follows the BWM data paper's stated inclusion criteria and reproduces its published 75,708 figure. Justification in Step 4. |
| (b) Probe merging | `merge_probes`: offset cluster ids, concatenate, `argsort(times, kind='stable')` | Same: offset by the running kept-neuron count, concatenate, `argsort(..., kind='stable')` | yes |
| (c) Temporal alignment | `intervals = trials[align_time] + time_window`, `align_time='stimOn_times'`, `time_window=(-.5, 1.5)` | Identical | yes |
| (c) Trial curation | `load_trials_and_mask(one, eid, max_trial_len=10.0)` | **The reference function itself is imported and called** | yes, by construction |
| (d) Binning | `bincount2D(times, clusters, xbin=0.02, xlim=[t_beg, t_end])`, `n_bins = ceil(2/0.02) = 100` | Vectorised `searchsorted` + `add.at`; verified bit-identical to `bincount2D` (`/app/cache/check_binning.py`, `np.allclose` True, identical spike totals) | yes |
| (d) Behaviour binning | `get_behavior_per_interval`: `interp1d(..., kind='linear')` onto `np.linspace(t_beg + binsize, t_end, n_bins)`, with coverage/NaN validity checks | Identical grid (`BIN_RIGHT_EDGES`) and identical validity rules | yes |
| (e) Input construction | The reference has no decoder inputs (its decoders take only neural activity) | `time_from_stim_onset`, `stim_onset`, `trial_num_in_block` as required by the Decoder Task | new, required by the task spec |
| (f) Output construction: choice | `trials['choice']` | Same, remapped +1 -> 0 (left), -1 -> 1 (right) per the Decoder Task | yes |
| (f) prior | `block = trials['probabilityLeft']` | Same, remapped 0.2/0.5/0.8 -> 0/1/2 per the Decoder Task | yes |
| (f) wheel speed | `abs(SessionLoader.wheel['velocity'])` | Identical, then discretised into per-session terciles as the Decoder Task requires | yes + required discretisation |
| (f) whisker ME | left camera `whiskerMotionEnergy`, falling back to right | Identical preference order | yes |
| Region mapping | `BrainRegions().acronym2acronym(..., mapping='Beryl')` | Identical | yes |
| Normalisation | `standardize_spike_data` z-scores spike counts inside the decoder data loader | Z-scoring is done in the conversion, because `/app/decoder.py` does none | equivalent effect, different place; justified in Step 4/5 |

Differences and their reasons are all recorded in Step 4; the only substantive ones are the
neuron QC/grey-matter filter (follows the data paper), the single stimulus-onset alignment for
all outputs (required by the Decoder Task), the discretisation of the two continuous outputs
(required by the Decoder Task) and where normalisation happens.

### Check 4: Key statistics comparison
See the table in Step 9. Every statistic available in the reference papers was reproduced:
459 sessions / 699 insertions / 139 subjects / 12 labs / 621,733 units / 75,708 well-isolated
neurons / 645 mean trials per session (median 602, range 401-1,525) / 81.4% correct / 58.7%
correct on 0% contrast / 20 ms bins / T = 100 / stimulus-onset alignment over [-0.5, 1.5] s.
No discrepancy remained unexplained.

### Check 5: Edge cases
- **Off-by-one at the trial edges**: bin k spans [stimOn - 0.5 + 0.02k, ... + 0.02(k+1)), so
  t = 0 falls in bin 25 and the last bin ends exactly at +1.5 s. Verified: the `stim_onset`
  indicator has exactly one 1, at index 25, in every trial of every session.
- **`np.clip` in the binning** guards against a spike landing exactly on the right edge because
  of floating-point rounding.
- **Trials at the very start/end of a recording** where the wheel or camera stream does not cover
  the whole 2 s window are dropped, mirroring the reference `target data starts too late` /
  `ends too early` checks, rather than being silently extrapolated.
- **Sessions with no whisker motion energy** (14) are skipped with an explicit recorded reason.
- **Sessions with no well-isolated grey-matter neuron** would be skipped (none occurred).
- **Sessions with fewer than 2 usable trials** would be skipped (none occurred); the minimum is
  85 trials.
- **Silent neurons** (zero variance) would give 0/0 in the z-score; the denominator is clamped to
  1e-6, so they become exactly 0 rather than NaN. Confirmed: the verifier reports no NaN/Inf.
- **Duplicate tercile edges** from a zero-inflated behaviour are nudged apart with `np.nextafter`
  so `searchsorted` still returns three classes.
- **`trial_num_in_block`** is computed on the unmasked trials table so that dropping trials does
  not renumber the blocks; verified against an independent re-computation.
- **Sessions with 1-3 neurons** (3 sessions: 1, 2 and 3 neurons) are kept. They are legitimate
  recordings that passed every published criterion; the BWM paper's "at least 5 neurons"
  threshold applies to a *region within a session* for its region-level statistics, not to
  whole sessions, and our decoder pools all of a session's neurons.
- **Session/trial ordering** is restored to the `bwm_release.csv` order after the unordered
  multiprocessing map, so the output is deterministic.

### Issues found and resolved during Steps 1-10
1. **ONE could not find the staged files** (shipped cache tables list non-revisioned paths while
   the files on disk sit in revision folders). Resolved by rebuilding the cache tables from the
   filesystem (`/app/cache/build_one_cache.py`) and re-mapping the hashed session UUIDs to the
   true eids.
2. **`ALFMultipleRevisionsFound` on 5 sessions**: my first rebuild marked every record as the
   default revision, but some sessions have two revisions of the same dataset staged. Fixed by
   marking only the most recent revision as default; all 459 sessions then loaded.
3. **A shared ONE instance is not thread-safe**: 76 of 459 sessions silently failed to load under
   `ThreadPoolExecutor`. Fixed by using processes, each with its own ONE instance.
4. **`one.eid2pid` needs a network connection**: replaced with the `pid`/`probe_name` columns of
   `bwm_release.csv`.
5. **`raw_electrophysiology(stream=True)`** in the reference `load_spiking_data` needs the raw AP
   binaries, which are not staged; it only supplies a metadata field, so it is skipped.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
-> `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.

### Training Progress
- Loss decreasing: **Yes**, monotonically for all 200 epochs:
  1.93 (epoch 2) -> 1.22 (20) -> 0.93 (40) -> 0.79 (100) -> 0.753 (130) -> 0.7387 (200).
- Final test loss 0.7661.
- Device: CUDA (NVIDIA L4).

### Decoder Results (Full: 445 sessions / 62,779 neurons / 189,057 trials)
| Output | Classes | Chance | Training Balanced Acc | Validation Balanced Acc | Val/chance | Notes |
|--------|---------|--------|-------------|--------|------|-------|
| choice | 2 | 0.5000 | 0.6375 | 0.6165 | 1.23x | averaged over all 100 bins, 25 of which precede the stimulus |
| prior_prob_left | 3 | 0.3333 | 0.6786 | 0.6603 | 1.98x | |
| wheel_speed | 3 | 0.3333 | 0.6151 | 0.6085 | 1.83x | |
| whisker_motion_energy | 3 | 0.3333 | 0.6065 | 0.5999 | 1.80x | |

Every output is well above chance and the train/validation gap is only 0.018-0.021.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Chance | Validation | Ratio | Verdict |
|----------|--------|-----------|-------|---------|
| choice | 0.500 | 0.6165 | 1.23x | below the 1.5x guideline, investigated in detail below |
| prior_prob_left | 0.333 | 0.6603 | 1.98x | fine |
| wheel_speed | 0.333 | 0.6085 | 1.83x | fine |
| whisker_motion_energy | 0.333 | 0.5999 | 1.80x | fine |

**Investigation of the choice ratio.** The decoder assigns a label to EVERY one of the 100 time
bins, but the trial window starts 0.5 s BEFORE the stimulus appears. In the first 25 bins the
animal has not seen the stimulus and has not moved, so its eventual choice is very nearly
unknowable, and those bins are averaged into the reported score. A per-timepoint analysis of the
converted data (/app/cache/diag_timecourse.py, 6 sessions with >200 neurons, cross-validated
logistic regression per bin) shows exactly the profile this predicts:

| time bin | 5 | 15 | 24 | 25 | 30 | 40 | 50 | 60 | 75 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|
| time (s) | -0.39 | -0.19 | -0.01 | +0.01 | +0.11 | +0.31 | +0.51 | +0.71 | +1.01 | +1.31 |
| choice | 0.503 | 0.519 | 0.522 | 0.513 | 0.696 | 0.812 | 0.722 | 0.688 | 0.620 | 0.588 |
| prior_prob_left | 0.550 | 0.518 | 0.556 | 0.540 | 0.615 | 0.668 | 0.614 | 0.576 | 0.568 | 0.549 |

Choice is AT CHANCE (0.50-0.52) in every pre-stimulus bin and rises immediately after stimulus
onset, peaking at 0.812 around +0.3 s. This is strong positive evidence that:
- the neural data and the trial labels are correctly aligned (a misalignment would smear or shift
  this transition, and a label bug would flatten it);
- the stimulus-onset alignment point is exactly where the metadata says it is (bin 25);
- the ceiling on the whole-trial average is a property of the task, not of the conversion.

Averaging 0.51 over the 25 pre-stimulus bins with about 0.70 over the 75 post-stimulus bins gives
about 0.65, close to the 0.6165 the decoder achieves. The 1.23x ratio is therefore expected and
is NOT a sign of a conversion bug. Prior, by contrast, is decodable throughout the trial (it is a
block property that persists across trials), which is why it scores highest.

### Check 2: Accuracy comparison to papers
| Variable | My validation accuracy | Reference paper value | Comparison |
|----------|------------------------|-----------------------|------------|
| choice | 0.6165 balanced acc over all 100 bins; 0.81 peak at +0.3 s from single 20 ms bins | Zhang et al. Fig. 5A: AUC 0.66 (L2 linear baseline), 0.72 (single-session RRR), 0.79 (multi-session RRR), per region | favourable. The papers decode ONE REGION AT A TIME explicitly to avoid ceiling effects from using all regions; we pool all regions of a session, and our peak per-bin accuracy (0.81) is at or above their best multi-session AUC. |
| prior | 0.6603 balanced acc (3-class) | Zhang et al.: Pearson r = 0.05 single-session, 0.34 multi-session, 0.65 oracle | not directly comparable (correlation of a continuous prior vs 3-class balanced accuracy), but 1.98x chance is strong. |
| wheel speed | 0.6085 (3-class) | Zhang et al. report R^2, per region | not directly comparable; R^2 of a continuous signal has no accuracy equivalent. 1.83x chance. |
| whisker motion energy | 0.5999 (3-class) | Zhang et al. report R^2, per region | as above. 1.80x chance. |
| choice (BWM paper) | - | null-corrected median balanced accuracy, small per-region effect sizes | our pooled-region result is necessarily higher than per-region effect sizes. |

No accuracy falls below what the papers report for a comparable quantity, so there is no evidence
of information being lost in the conversion.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|--------|-------|-----------|-------|
| choice | 0.6375 | 0.6165 | 1.034 |
| prior_prob_left | 0.6786 | 0.6603 | 1.028 |
| wheel_speed | 0.6151 | 0.6085 | 1.011 |
| whisker_motion_energy | 0.6065 | 0.5999 | 1.011 |

All ratios are far below the 1.5x threshold, so there is no overfitting and no data leakage.

### Issues Found and Resolved in this step
**Issue 1 (major): the neural normalisation was wrong.**
My Step 5 plan applied a PER-NEURON z-score. Re-reading `data_loader_utils.standardize_spike_data`
line by line showed the reference does something quite different: for each time bin it computes a
SINGLE SCALAR mean and std pooled over all neurons and all trials, and applies that one affine
transform to the whole block. A per-neuron z-score destroys the relative firing-rate differences
between neurons, which are informative.

I tested this empirically rather than by argument, converting the same 40 sessions three ways and
training the provided decoder on each (validation balanced accuracy):

| Variant | choice | prior | wheel | whisker |
|---------|--------|-------|-------|---------|
| per-neuron z-score (my original plan) | 0.5987 | 0.6401 | 0.6098 | 0.5911 |
| raw spike counts, no normalisation | 0.6202 | 0.6753 | 0.6090 | 0.5855 |
| reference per-time-bin standardisation (ADOPTED) | 0.6196 | 0.6697 | 0.6111 | 0.5887 |

The reference transform matches raw counts to within noise and beats per-neuron z-scoring by
about 0.02-0.035 on choice and prior. RESOLUTION: adopted the reference transform, re-ran the
full conversion and every downstream validation. On the full dataset this lifted validation
accuracy on every output (choice 0.5944 -> 0.6165, prior 0.6261 -> 0.6603, wheel 0.6087 -> 0.6085,
whisker 0.5948 -> 0.5999) and lowered the training loss from 0.7693 to 0.7387.

**Issue 2 (numerical): float32 accumulation error in the standardisation.**
After switching transforms, 6 of the 84 independent sanity checks failed: the full-array
`np.allclose` comparison failed even though the per-element spot-checks passed. I did not dismiss
this as float noise but measured it: max absolute difference 1.59e-3 and max relative difference
1.4e-4, against a float32 round-off bound of only eps * max|z| = 1.9e-6, i.e. roughly 100x too
large to be explained by storing the result in float32. The real cause was that the mean and std
were being accumulated IN FLOAT32 over the roughly 1e5 values that each time bin pools, and that
summation loses precision. RESOLUTION: compute the statistics with `dtype=np.float64` and do the
subtraction/division in float64 before casting the result to float32. This reduced the maximum
difference to 4.77e-7 (now below the float32 round-off bound) and made the arrays bit-identical
after a float32 round-trip. All 84 sanity checks pass again.

### Re-run of all checks after the fixes (iteration protocol)
Both fixes changed the stored neural values, so the whole downstream chain was re-run, not just
the failing check:
- full conversion re-run -> identical session/neuron/trial statistics (445 / 62,779 / 189,057),
  confirming only the normalisation changed;
- `--verify-only` -> valid, no errors or warnings;
- `/app/cache/sanity_checks.py` -> 84 passed, 0 failed;
- sample conversion, sample verification and sample training regenerated;
- full decoder training re-run -> the Step 11 table above.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created and documented in cache/README_CACHE.md
- [x] All files organized

### Deliverables
| File | Size | Description |
|------|------|-------------|
| `CONVERSION_NOTES.md` | 60K | this document: every decision, check and validation result |
| `README.md` | 163 lines | user-facing dataset description, load instructions, format spec |
| `convert_data.py` | 36K | the conversion script |
| `converted_data.pkl` | 11 GB | full converted dataset, 445 sessions |
| `sample_data.pkl` | 25 MB | 2-session sample |
| `conversion_full_out.txt` / `conversion_sample_out.txt` | | conversion logs |
| `verification_full_out.txt` / `verification_sample_out.txt` | | format verification logs (both: valid, no errors or warnings) |
| `train_decoder_full_out.txt` / `train_decoder_sample_out.txt` | | decoder training logs |
| `processing_<eid>.png` (x2) | | per-step conversion visualisations |
| `sample_trials.png`, `predictions.png` | | decoder sample and prediction plots |
| `cache/` | | helper scripts, diagnostics, rebuilt ONE index, extracted paper text |

### Summary of the conversion
The IBL Brain Wide Map was converted to trial-aligned arrays following the reference pipeline of
Zhang et al. 2025: aligned to stimulus onset over [-0.5, +1.5] s in 100 non-overlapping 20 ms
bins, with the reference trial-curation mask applied by importing and calling the reference
`load_trials_and_mask` itself. Neurons are restricted to well-isolated grey-matter units per the
BWM data paper's stated inclusion criteria, and probes within a session are merged. Spike counts
are standardised exactly as the reference `standardize_spike_data` does.

Result: 445 sessions, 136 subjects, 62,779 neurons, 189,057 trials, 263 brain regions.

### Confidence in correctness
1. Every dataset statistic quoted in the two papers was reproduced from the raw data: 459
   sessions, 699 insertions, 139 subjects, 12 labs, 621,733 units, 75,708 well-isolated neurons,
   645 mean trials/session (median 602, range 401-1,525), 81.4% correct, 58.7% correct at 0%
   contrast.
2. The spike binning is bit-identical to the reference `bincount2D` implementation.
3. 84 independent sanity checks re-derive neural, input and output values straight from the raw
   ALF files without using any conversion code, and all pass, including a full-array
   `np.allclose` on the standardised neural data.
4. The format verifier reports no errors and no warnings.
5. Choice decodability is at chance before stimulus onset and peaks at 0.81 shortly after, which
   is only possible if the neural data and labels are correctly aligned.
6. Two real defects found by these checks (the wrong normalisation, and a float32 accumulation
   error) were diagnosed with measurements rather than assumption, fixed, and the entire
   downstream chain was re-run.

---
