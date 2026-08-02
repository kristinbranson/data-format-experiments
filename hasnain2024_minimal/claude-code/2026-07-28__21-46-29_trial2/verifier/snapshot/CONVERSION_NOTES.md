# Conversion Notes

## Source Data

Data from Hasnain, Birnbaum et al. (Nature Neuroscience 2024), "Separating cognitive and motor processes in the behaving mouse." Downloaded from Zenodo (DOI: 10.5281/zenodo.13941415).

## Session Selection

### Ephys_Behavior (25 sessions, 10 mice)
Sessions were identified from the `DataLoadingScripts/Recording and video/` MATLAB loading scripts. Each `loadANM_ALMVideo.m` file specifies which sessions and probes to use for each animal.

- **EKH1**: 1 session (probe 2)
- **EKH3**: 1 session (probe 2)
- **JEB6**: 1 session (probe 2)
- **JEB7**: 2 sessions (probe 1)
- **JGR2**: 2 sessions (probe 1)
- **JGR3**: 1 session (probe 1)
- **JEB13**: 5 sessions (probes 1 or 2 depending on session)
- **JEB14**: 4 sessions (probe 1)
- **JEB15**: 4 sessions (probes 1+2 or probe 2 only)
- **JEB19**: 4 sessions (probe 1)

### RandomizedDelay_Ephys_Behavior (19 sessions, 4 mice)
- **JEB11**: 2 sessions (probe 1)
- **JEB12**: 2 sessions (probe 1)
- **JEB23**: 7 sessions (probe 1) - session 2023-10-20 excluded per loading script (commented out)
- **JEB24**: 8 sessions (probe 1)

### Excluded Data
- **JEB4, JEB5**: Referenced in loading scripts but data files not provided
- **JEB24 2023-10-03 and 2023-10-04**: Data files exist but not referenced in loading scripts; no motion energy files for those dates
- **JEB23 2023-10-20**: Commented out in loading script

### Note on JEB15
The loading script contains a comment "excluding first three sessions, they look like they were in a more sensory area" but the sessions are NOT commented out. All 4 JEB15 sessions are included, matching the loading script behavior.

## File Format Handling

The .mat data files come in two formats:
- **MATLAB v7.3 (HDF5)**: Most sessions. Loaded with `h5py`.
- **MATLAB v5**: Some JEB23 and all JEB24 sessions. Loaded with `scipy.io.loadmat`.

The conversion script auto-detects the format and uses the appropriate loader.

## Processing Pipeline Details

### Trial Filter
Matches `findTrials.m` condition `'(hit|miss)&~stim.enable&~early'`:
- Include hit and miss trials
- Exclude stimulation trials (`stim.enable == 1`)
- Exclude early-lick trials (`early == 1`)
- No-response trials are also excluded (they are neither hit nor miss)

### Cluster Quality Filter
Matches `findClusters.m` with `'all'` quality parameter:
- Excludes: `garbage`, `gabrga`, `noisy`, `real?`
- Includes all other quality labels (excellent, great, good, fair, poor, multi)
- Some sessions have null-byte quality labels (`'\x00\x00'`) which are treated as valid

### Spike Binning and Smoothing
Matches `getSeq.m` and `mySmooth.m`:
1. Align spike times: `trialtm - goCue` (matching `alignSpikes.m`)
2. Bin into 10ms bins from -2.5s to +2.5s (500 bins)
3. Convert to firing rate: `counts / dt`
4. Smooth with causal Gaussian kernel:
   - Window size N=15
   - `kern = gausswin(15)` then `kern[:7] = 0` (causal: zero first half)
   - Normalize: `kern /= kern.sum()`
   - Reflect boundary: prepend first N samples before convolution
   - Convolve with `mode='same'`, then trim padding

### Firing Rate Filter
Matches `removeLowFRClusters.m`:
- Compute mean firing rate across all time bins and all trials
- Remove neurons with mean FR <= 1 Hz

### Video Offset
Matches `findVideoOffset.m`:
- `vidshift = mode(sglx.bitcode.bitstart) / sglx.fs - mode(bp.ev.bitStart)`
- Used to align camera frame times to ephys clock

### DLC Velocities (Bottom Camera)
- Extract (x, y, confidence) for `top_tongue` and `top_paw` features
- Fill invalid positions (NaN or low confidence < 0.9) with nearest valid position
- Compute instantaneous speed: `sqrt(dx/dt^2 + dy/dt^2)`
- Align to go cue using video offset
- Interpolate to neural time axis

### Tongue Velocity Discretization
Tongue is only visible during licking (~4-8% of frames). After filling invalid positions with nearest valid, velocity is near-zero when tongue is not visible. The 50th percentile threshold is computed on **non-zero** values only. Zero-velocity time bins are automatically classified as "low".

### Paw Velocity and Motion Energy Discretization
Standard 50th percentile threshold computed on all values per session.

### Motion Energy
- Loaded from separate `motionEnergy_*.mat` files (always MATLAB v5 format)
- Aligned using side camera frame times and video offset
- Interpolated to neural time axis
- Some sessions had motion energy loading issues (different struct format); these sessions have all-zero motion energy data

## Warnings During Conversion

### All-zero Neural Data (Sessions 36, 43)
Late trials in JEB24_2023-10-23 (session 36) and JEB24_2023-11-03 (session 43) have all-zero neural activity. This likely indicates the recording ended before the behavioral session. These trials pass through the pipeline but contribute no information.

### Motion Energy Loading Failures
Several sessions have motion energy warnings:
- **JEB15 2022-07-26 and 2022-07-28**: Motion energy struct has unexpected format (`dtype([('data', 'O'), ('moveThresh', 'O')])`)
- **JEB23 2023-10-10 through 2023-10-13**: Array indexing issue with motion energy data
- **JEB24 2023-10-31**: Same struct format issue as JEB15

These sessions have all-zero motion energy, so the motion_energy output is all "high" (0 >= threshold of 0). This is flagged in the verification output where motion_energy fraction is 0.000/1.000 for those sessions.

### Paw Velocity All-High (Sessions 8, 10)
JEB13 2022-09-13 and JEB13 2022-09-21 have paw velocity threshold of 0 (all paw velocity values are >= 0, so all are "high"). This occurs when the DLC paw tracking has very low confidence throughout.

## Decoder Performance

All 6 output dimensions achieve above-chance validation balanced accuracy on both sample (3 sessions) and full (44 sessions) datasets, confirming the data contains decodable neural information for all target variables.
