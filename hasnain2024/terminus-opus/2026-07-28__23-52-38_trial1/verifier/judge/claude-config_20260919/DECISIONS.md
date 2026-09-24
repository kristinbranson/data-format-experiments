# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by **globbing the two ephys data folders** (`data/Ephys_Behavior`, `data/RandomizedDelay_Ephys_Behavior`) for files matching `data_structure_<ANM>_<YYYY-MM-DD>.mat`, skipping animals whose name starts with `MAH` (behavior-only). For each discovered session it looks up the probe number(s) in a `probe_map` built by **regex-parsing the authors' `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` scripts** (`meta(end).anm / .date / .probe`, with `meta(end+1)` also matched, and commented-out `%` lines skipped); sessions absent from the scripts fall back to probe `[1]`. A sidecar `motionEnergy_<ANM>_<DATE>.mat` is recorded if present. Each `data_structure` file is then read once by `load_session_data`, which sniffs the file header and dispatches to an **h5py reader for MATLAB v7.3** files or a **scipy.io reader for MATLAB v5** files, normalising both into one dict with `bp`, `clu`, `sglx`, `traj`. This yielded 47 candidate sessions, of which 43 survive (2 excluded by the behavioral inclusion criterion, 2 have no `clu` field). Motion energy is loaded separately with `scipy.io.loadmat`. The whole per-session call is wrapped in `try/except` so a failing session is skipped with a printed traceback rather than aborting the run.

ii.
```python
def get_available_sessions():
    script_dir = 'code/DataLoadingScripts/Recording and video'
    probe_map = {}  # (animal, date) -> probes
    for f in sorted(os.listdir(script_dir)):
        ...
            m = re.search(r"meta\(end(?:\+1)?\)\.probe\s*=\s*(.+?);", line)
    for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
            ...
            if animal.startswith('MAH'):
                continue
            probes = probe_map.get((animal, date), [1])  # default to probe 1
```

```python
def load_session_data(filepath):
    """Load a session's data from a .mat file (auto-detect format)."""
    fmt = detect_mat_format(filepath)
    if fmt == 'hdf5':
        return load_session_data_hdf5(filepath)
    elif fmt == 'v5':
        return load_session_data_v5(filepath)
```

iii. CONVERSION_NOTES Step 1 records `loadSessionData.m` → `processData.m` as the authors' pipeline, and Step 6 says "Implemented dual-format loader (HDF5 and MATLAB v5)… Auto-detect file format". The trajectory shows the dual-format loader was added reactively after ~15 RandomizedDelay sessions failed with "file signature not found" (steps 77–90), and the probe-parsing regex was fixed after EKH1 was loaded with the wrong probe (steps 61–66). The AI justified globbing over a hard-coded list implicitly, by treating every non-`MAH` data file as usable and then relying on `UseInclusionCritera` to prune; Step 9 notes the resulting extra session ("1 extra (JEB23_2023-10-20 not in scripts)") but it was left in.

## 1-b. How are the data split into subjects?

i. The subject is the animal id parsed from the filename (`data_structure_<ANM>_<DATE>.mat` → `ANM`). It is stored per session as `result['animal']`; at assembly `subjects` is the sorted set of unique animals and `subject_idx` is each session's index into that list. Result: 14 subjects over 43 sessions.

ii.
```python
m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
animal = m.group(1)
date = m.group(2)
...
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The AI never states a rationale explicitly; the filename/loading-script convention (`meta(end).anm`) is the authors' own way of identifying animals, and the AI's probe map is keyed on exactly `(animal, date)` pairs parsed the same way.

## 1-c. How are the data split into sessions?

i. One `data_structure_*.mat` file = one session = one element of `neural` / `input` / `output`. Both task folders are pooled into a single flat session list, so fixed-delay and randomized-delay sessions are treated uniformly. Sessions are then filtered by two criteria applied at conversion time: the authors' behavioral criterion, `check_session_inclusion` (**>40 right-hit DR trials AND >40 left-hit DR trials**, where DR-hit means `R & hit & ~stim.enable & ~autowater & ~early`), and a minimum of **10 units** after neuron curation. Sessions with no `clu` field error out and are dropped. Final count: 43 sessions (23 fixed-delay + 20 randomized-delay) vs. the paper's 25 + 19.

ii.
```python
def check_session_inclusion(session_data, probes):
    """Matching UseInclusionCritera.m: >40 R hit DR trials AND >40 L hit DR trials."""
    r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    return np.sum(r_hit_dr) > MIN_HIT_TRIALS and np.sum(l_hit_dr) > MIN_HIT_TRIALS, n_r, n_l
```
```python
MIN_UNITS = 10  # minimum units per session
...
if n_neurons < MIN_UNITS:
    print(f'    EXCLUDED: only {n_neurons} units (need >={MIN_UNITS})')
    return None
```

iii. CONVERSION_NOTES Step 1 lists `UseInclusionCritera.m` as "Removes sessions with <=40 right hit or <=40 left hit DR trials", and Step 3 records "Sessions included only if they had at least 10 units" from the methods. The AI documented the resulting mismatch in Step 9 ("DR sessions 25 → 23, Close - 2 excluded by criteria"; "RD sessions 19 → 20, 1 extra (JEB23_2023-10-20 not in scripts)") and in trajectory step 110 explicitly identified that it should probably exclude JEB23_2023-10-20, but did not change the code.

## 1-d. How are the data split into trials?

i. A trial is one index `0..Ntrials-1` into the flat per-trial vectors of `obj.bp` (`L`, `R`, `hit`, `miss`, `no`, `autowater`, `early`, `stim.enable`, `ev.goCue`, …). Spikes carry their own 1-based trial label (`clu.trial`) and a within-trial time (`clu.trialtm`), so spikes are assigned to trial `j` by `trial_nums == j+1`. Camera trajectories and motion energy are stored as one entry per trial and are indexed by the same `trix`. No trial boundaries are reconstructed.

ii.
```python
bp['Ntrials'] = int(f['obj/bp/Ntrials'][0, 0])
...
for j in range(Ntrials):
    trial_num = j + 1  # MATLAB 1-indexed
    spk_mask = trial_nums == trial_num
    ...
    aligned_times = trialtm[spk_mask] - goCue[j]
```
```python
for trix in range(Ntrials):
    trial = cam_data['trials'][trix]
```

iii. Not separately justified; the AI's Step 2 exploration established that `obj.bp` fields are `Ntrials`-long vectors and that `clu` entries carry `trial`/`trialtm`, and the reference `getSeq.m` uses exactly `ismember(obj.clu{prb}(clu).trial, j)`.

## 1-e. How are trials filtered based on quality controls?

i. Three per-trial masks are ANDed: **not early-lick** (`~bp.early`), **not photostim** (`~bp.stim_enable`), and **not an ignore trial** (`~bp.no`). Everything else is kept. A session with fewer than 2 surviving trials is dropped. There is **no filter for trials that run past the end of the ephys recording**: the verification log reports 30 trials in 2 sessions whose neural data is entirely zero, and these were knowingly kept. Across the dataset 11,981 trials survive.

ii.
```python
# ---- Select valid trials ----
# Include all trials that are not early and not stim
bp = session['bp']
valid_trials = ~bp['early'] & ~bp['stim_enable']
# Also exclude 'no' (ignore) trials
valid_trials = valid_trials & ~bp['no']

trial_indices = np.where(valid_trials)[0]

if len(trial_indices) < 2:
    print(f'    EXCLUDED: only {len(trial_indices)} valid trials')
    return None
```

iii. CONVERSION_NOTES Step 3 gives the trial curation rule as "Exclude early lick and stim trials" — the removal of ignore trials is not listed there and appears only as an inline code comment. Trajectory step 47 records the AI's reading of the reference decoding scripts as "Outcome: hit=1, miss=0, no/ignore=NaN", which is the apparent motivation for dropping ignore trials. For the all-zero neural trials, Step 10 Check 1 says: "These are likely trials at the end of the recording session where the recording had ended. The decoder should handle these gracefully."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — per cluster the fields `trial` (1-based trial of each spike), `trialtm` (spike time relative to trial start) and `quality` (manual curation label). `obj.bp.ev.goCue` supplies the alignment time and `obj.bp.Ntrials` the trial count. The unused field `clu.tm` (session-clock spike times) is also read into memory. Probes are selected per session from the parsed probe map and concatenated.

ii.
```python
unit['tm'] = np.array(f[clu_group['tm'][u, 0]]).flatten()
unit['trial'] = np.array(f[clu_group['trial'][u, 0]]).flatten().astype(int)
unit['trialtm'] = np.array(f[clu_group['trialtm'][u, 0]]).flatten()
unit['quality'] = read_h5_string(f, clu_group['quality'][u, 0])
```
```python
units = session['clu'][probe_idx]
goCue = session['bp']['ev']['goCue']
...
firing_rates = np.concatenate(all_firing_rates, axis=0)  # (n_neurons, n_time, n_trials)
```

iii. Step 1 of CONVERSION_NOTES documents `alignSpikes.m` ("trialtm_aligned = trialtm - goCue(trial)") and `getSeq.m` ("Bins spikes (histc with edges), smooths (mySmooth), creates trialdat (time x units x trials)"), which is exactly the set of fields the AI uses.

## 2-b. How is the `neural` data processed?

i. Per unit and per trial, go-cue-aligned spike times are histogrammed into the 5 ms bin grid, divided by the bin width to give **spikes/s (Hz)**, then smoothed along time with a **causal Gaussian kernel of window 15 bins** built to imitate `mySmooth.m`: a Gaussian window is generated, its first `N//2` taps are zeroed to make it causal, it is renormalised to sum to 1, and it is convolved with `mode='same'`. No baseline subtraction, normalisation or z-scoring. Units from multiple probes are concatenated. The kernel's standard deviation is set to `np.std(np.arange(1, N+1))` ≈ 4.32 bins (≈21.6 ms), taken from MATLAB's `kernsd = std(1:N)` line — but that line is only a *returned diagnostic*; the actual `gausswin(15)` kernel has σ = (N−1)/(2·2.5) = 2.8 bins (14 ms). The implemented filter is therefore ~1.5× wider than the reference's. No boundary condition is applied (`mySmooth` was called with `params.bctype`).

ii.
```python
def causal_gaussian_kernel(N):
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(1, N+1))))
    kern[:N//2] = 0          # Make causal: zero the first half
    kern = kern / kern.sum()
    return kern

def smooth_signal(x, N):
    """Apply causal gaussian smoothing matching mySmooth.m."""
    kern = causal_gaussian_kernel(N)
    return np.convolve(x, kern, mode='same')
```
```python
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float64) / DT
fr_smooth = smooth_signal(fr, SMOOTH_N)
firing_rates[i, :, j] = fr_smooth
```

iii. Trajectory step 50: "Now I understand the smoothing: Uses a Gaussian window of size N (15 by default); Makes it CAUSAL by zeroing the first half of the kernel; Normalizes the kernel to sum to 1; Convolves with 'same' mode". CONVERSION_NOTES Step 1 lists `mySmooth` as "Causal gaussian smoothing: gausswin(N), zero first half, normalize, conv same" and Step 10 Check 3 asserts "Binning: matches getSeq.m (histc/histogram with edges, smooth with causal gaussian)".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit-level filters plus one session-level filter. (1) **Quality**: any cluster whose lower-cased `quality` string contains the substring `garbage` is dropped; everything else (Fair, Good, Poor, Multi, blank) is kept. (2) **Firing rate**: after binning and smoothing, each unit's mean over all time bins and all trials must exceed **`LOW_FR = 0.5` Hz**. (3) **Session**: at least `MIN_UNITS = 10` surviving units, else the session is dropped. The mean rate is computed over *all* `Ntrials`, including the early/stim/ignore trials that are later discarded, and over the full ±2.5 s window. Result: 2,443 units, 17–142 per session.

ii.
```python
LOW_FR = 0.5   # minimum firing rate threshold (Hz)
MIN_UNITS = 10  # minimum units per session
...
for u_idx, unit in enumerate(units):
    quality = unit['quality'].lower().strip()
    if 'garbage' not in quality:
        valid_units.append(u_idx)
...
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)  # mean over time and trials
fr_mask = mean_frs > LOW_FR
```

iii. CONVERSION_NOTES Step 1 documents `deleteGarbageClu` ("Removes clusters with quality='garbage'") and `removeLowFRClusters` ("Removes units with mean FR < lowFR across all conditions"), Step 3 records "FR threshold | 0.5 Hz (code)", and Step 4's discrepancy table resolves the paper-vs-code conflict explicitly: "FR threshold | Code says lowFR=0.5 | Paper says 1 Hz for some analyses | **Used 0.5 Hz as in code**". Step 1 also notes "Quality filtering: 'all' (keeps Fair, Good, Poor, Multi; removes garbage)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By a single subtraction. `clu.trialtm` is already on the behaviour clock relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `aligned_times = trialtm[spikes of trial j] - goCue[j]` puts every spike in seconds from go-cue onset for that trial. These aligned times are then histogrammed directly into the fixed −2.5…2.5 s edge grid, so spikes outside the window simply fall outside the histogram. No interpolation or per-session offset is needed for the neural stream (the video streams get a separate clock correction, see 7-d).

ii.
```python
for j in range(Ntrials):
    trial_num = j + 1  # MATLAB 1-indexed
    spk_mask = trial_nums == trial_num
    if not np.any(spk_mask):
        continue
    # Align to goCue
    aligned_times = trialtm[spk_mask] - goCue[j]
    # Bin spikes
    counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. CONVERSION_NOTES Step 1: "`alignSpikes` … Aligns spike times to goCue: trialtm_aligned = trialtm - goCue(trial)"; Step 10 Check 3: "Temporal alignment: matches alignSpikes.m (trialtm - goCue)". `ALIGN_EVENT = 'goCue'` is set at the top of the script to mirror `params.alignEvent`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **5 ms** (`DT = 1/200`), over a window of **−2.5 to +2.5 s** from the go cue, giving **1001 bins** per trial (the axis is `arange(-2.5, 2.5+DT/2, DT)`, i.e. MATLAB's `tmin:dt:tmax`, and the histogram edges append one extra edge at the end, so bin *k* covers `[t_k, t_k+5 ms)`). Every trial in every session has exactly 1001 bins. Spikes are binned directly at 5 ms — there is **no rebinning or resampling** of the neural data. The camera streams (~400 Hz) are brought onto this same axis by linear interpolation rather than by averaging frames into bins (see 7-d/9-d). `metadata['time_bin_size'] = 5.0` ms.

ii.
```python
DT = 1.0 / 200.0  # 5ms time bins
TMIN = -2.5
TMAX = 2.5

def make_time_axis():
    """Create time axis matching MATLAB code: tmin:dt:tmax."""
    time_axis = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = time_axis[time_axis <= TMAX + 1e-10]
    return time_axis
...
edges = np.append(time_axis, time_axis[-1] + DT)
```

iii. The script header says "Parameters (matching reference code getDefaultParams.m)"; CONVERSION_NOTES Step 1 records "getDefaultParams … dt=1/200 (5ms), tmin=-2.5, tmax=2.5" and Step 3's table lists "Neural time bin | 5 ms (dt=1/200) | code". Step 10 Check 2 states "Input data: time axis is [-2.5, 2.5] with 1001 bins at 5ms".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw variable — it is the conversion's own time axis, defined by the go-cue alignment (`bp.ev.goCue` sets t = 0) and the fixed window/bin parameters `TMIN`, `TMAX`, `DT`. `input_names = ['time_from_go_cue']`, dimension 1.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
time_axis = make_time_axis()
...
'input_names': ['time_from_go_cue'],
```

iii. Step 5's mapping table: "time from goCue | input[0] | Time axis -2.5 to 2.5 | Same for all trials". The window is taken from `getDefaultParams.m` (`params.tmin`/`params.tmax`).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond building the axis. The same 1001-value `float32` vector is reshaped to `(1, 1001)` and appended once per trial, so the input array is identical for every trial and every session.

ii.
```python
# Input: time from goCue (same for all trials)
inp = time_axis.astype(np.float32).reshape(1, -1)  # (1, n_timepoints)
input_trials.append(inp)
```

iii. No justification is given beyond the Decoder Task specification ("Time from go cue onset in seconds (continuous, time-varying)").

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. The same `time_axis` object is used to build the spike histogram edges (`edges = append(time_axis, time_axis[-1]+DT)`) and to fill the input, so input element *k* and neural column *k* refer to the same interval. The reported value is the **left edge** of bin *k* rather than its centre, so the nominal time lags the bin's mean time by 2.5 ms; this offset is identical for all trials, sessions and streams. The verification log confirms `time_from_go_cue: [-2.5, 2.5]` in every session.

ii.
```python
edges = np.append(time_axis, time_axis[-1] + DT)
n_time = len(time_axis)
...
counts, _ = np.histogram(aligned_times, bins=edges)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. Not separately justified. The AI's Step 10 Check 2 sanity check was "Input data: time axis is [-2.5, 2.5] with 1001 bins at 5ms".

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. A **single** per-trial field, `obj.bp.R` — the *instructed* lick direction of the trial (`R=1` → right, otherwise left). `bp.hit` / `bp.miss`, which would say whether the animal actually licked the instructed port, are **not** used for this output, and `bp.no` (ignore trials) is used only to delete those trials from the dataset entirely. There are therefore only two classes: `['left', 'right']`.

ii.
```python
# Lick direction: R=1, L=0
lick_dir = 1 if bp['R'][trial_idx] else 0
...
out[0, :] = lick_dir  # constant across time
...
'output_values': [
    ['left', 'right'],           # lick_direction: 0=left, 1=right
```

iii. Step 5's mapping table states "obj.bp.R | output[0]: lick_direction | R=1, L=0 | Per trial". The AI's reading of the reference decoding code (trajectory step 47) was "Choice decoding: right=1, left=-1 (or 0)", i.e. it adopted the paper's *choice/instruction* label rather than the realised lick. No rationale is given for omitting the "none" class; it follows mechanically from having deleted ignore trials in 1-e.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct relabelling of one boolean, broadcast constant across all 1001 time bins of the trial so the output array is uniformly `(6, 1001)`. Because a miss trial is one where the animal licked the port *opposite* to the instruction, the label is wrong on every miss trial (13.6% of the converted dataset by the AI's own `outcome` distribution). The resulting distribution is `{left 0.499, right 0.501}`.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
...
out = np.zeros((6, len(time_axis)), dtype=np.int64)
out[0, :] = lick_dir  # constant across time
```

iii. Step 5 rationale as above. CONVERSION_NOTES nowhere discusses the hit/miss correction or the missing "none" class; Step 12's only comment on this output is that 0.660 validation balanced accuracy is "reasonable given the different architecture".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`. Trials where water is delivered automatically without a cue are the water-cued (WC) context; all others are delayed-response (DR).

ii.
```python
bp['autowater'] = np.array(f['obj/bp/autowater']).flatten().astype(bool)
...
# Behavioral context: WC=0, DR=1
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. Trajectory step 24: "obj/bp/autowater: WC vs DR indicator (autowater=1 means WC)"; Step 5's mapping table: "obj.bp.autowater | output[1]: behavioral_context | WC=0, DR=1". This is also how the reference `findTrials.m` conditions are written (`~autowater` = DR).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling of the boolean, broadcast constant across the 1001 bins. Codes follow the prompt's ordering, `output_values = ['WC','DR']`. Overall distribution `{WC 0.074, DR 0.926}`; 15 of 43 sessions are pure DR (range `[1.0, 1.0]`), which is expected since only some sessions ran the two-context paradigm.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
...
out[1, :] = context   # constant across time
...
['WC', 'DR'],                # behavioral_context: 0=WC, 1=DR
```

iii. Step 7 of CONVERSION_NOTES comments on the sample distribution: "behavioral_context is imbalanced (23.7% WC, 76.3% DR) - expected since DR blocks are typically longer" (trajectory step 71).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. A **single** per-trial flag, `obj.bp.hit`. `bp.miss` is loaded but not used for this output, and `bp.no` is used only to remove ignore trials from the dataset. Among the kept trials (neither early, stim, nor ignore) `hit` and `miss` are complementary, so the output is effectively hit-vs-miss with two classes `['incorrect', 'correct']` — there is no `ignore` class.

ii.
```python
bp['hit'] = np.array(f['obj/bp/hit']).flatten().astype(bool)
bp['miss'] = np.array(f['obj/bp/miss']).flatten().astype(bool)
bp['no'] = np.array(f['obj/bp/no']).flatten().astype(bool)
...
# Outcome: correct=1, incorrect=0
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. Step 5 mapping: "obj.bp.hit | output[2]: outcome | incorrect=0, correct=1 | Per trial". The choice traces back to trajectory step 47, where the AI read the reference analysis convention as "Outcome: hit=1, miss=0, no/ignore=NaN" — i.e. the paper treats ignore trials as undefined and excludes them, and the AI carried that over to the decoder dataset.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct relabelling of the boolean, broadcast constant across the 1001 bins. Resulting distribution `{incorrect 0.136, correct 0.864}`.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
...
out[2, :] = outcome   # constant across time
...
['incorrect', 'correct'],    # outcome: 0=incorrect, 1=correct
```

iii. "outcome is imbalanced (7.3% incorrect, 92.7% correct) - expected for well-trained mice" (trajectory step 71). Step 12 notes the 0.661 accuracy: "Outcome information may be more distributed or timing-dependent."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`: **camera 0 (side view), feature `tongue`**, channels x and y of `traj(trix).ts`. Supporting fields are `traj(trix).frameTimes`, `traj(trix).NdroppedFrames`, `bp.ev.goCue`, and — for the clock correction — `obj.sglx.fs` and `obj.sglx.bitcode.bitstart` together with `bp.ev.bitStart`. The bottom camera's `top_tongue` / `bottom_tongue` features are catalogued during exploration but **not used**. The likelihood channel (`ts[:, 2, :]`) is not read; visibility is inferred from the NaNs the authors already wrote into x/y.

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
...
feat_idx = None
for i, name in enumerate(feat_names):
    if name.lower() == feat_name.lower():
        feat_idx = i
        break
...
x = ts[feat_idx, 0, :]  # x coordinate
y = ts[feat_idx, 1, :]  # y coordinate
```

iii. Trajectory step 54: "Camera 0 (side view): tongue, left_tongue, right_tongue, jaw, trident, nose, lickport; Camera 1 (front view): top_tongue, … top_paw, bottom_paw …  For the decoder task: Tongue velocity: use tongue feature from camera 0 (x,y velocity magnitude)". No reason is given for preferring the side view over combining both views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per trial: (1) trials whose `NdroppedFrames` is NaN are **skipped entirely** (left as NaN), mirroring `findPosition.m`; (2) frame times are shifted onto the go-cue clock (see 7-d); (3) **x and y are linearly interpolated onto the 1001-bin axis with `scipy.interpolate.interp1d`, after first dropping the NaN samples** — so gaps in which the tongue was untracked are bridged by interpolation rather than left as NaN (MATLAB's `interp1` would have propagated the NaNs); (4) no positional smoothing is applied (the reference also skips smoothing for tongue features); (5) velocity is `np.gradient(xpos)` / `np.gradient(ypos)` **without passing the time step**, so the units are pixels per 5 ms bin, not pixels per second — a constant rescaling that is harmless for a within-session percentile split; (6) **remaining NaN velocities are set to 0** ("tongue not visible"), following `findVelocity.m`; (7) the speed is `sqrt(xv² + yv²)`. No cross-camera normalisation is applied (only one view is used).

ii.
```python
valid = ~np.isnan(x) & ~np.isnan(aligned_times)
if np.sum(valid) < 2:
    continue
f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
              bounds_error=False, fill_value=np.nan)
f_y = interp1d(aligned_times[valid], y[valid], kind='linear',
              bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```
```python
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
if not is_tongue:
    ...
else:
    # Set tongue velocity to 0 if not visible
    xv[np.isnan(xv)] = 0
    yv[np.isnan(yv)] = 0
...
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. Docstring: "Extract velocity for a DLC feature, matching findPosition + findVelocity." Trajectory step 37: "For tongue: NaN velocity set to 0 (tongue not visible); For non-tongue: NaN velocity filled with nearest value", and step 36–38 record `findPosition.m`'s `interp1(traj.frameTimes - vidshift - goCue, ts, taxis)` and its rule not to smooth tongue features.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A **per-session** threshold: `np.nanpercentile(tongue_vel[:, valid_trials], 50)`, pooled over all bins and all kept trials. Bins with speed `>= threshold` get class 1, all others class 0; NaN bins are also forced to class **0**. Only **two** classes are emitted (`['low','high']`) — the specified third class, `2 = not visible`, is **not implemented**.

Because step 7-b writes 0 (not NaN) into every bin in which the tongue is invisible, and the tongue is invisible in the large majority of bins, the 50th percentile is **0.00 in every one of the 43 sessions** (visible in the conversion log: `Thresholds: tongue=0.00, …` for every session). Since speed is non-negative, `tv >= 0` is true everywhere, so the output collapses to a near-constant: the verification log reports `tongue_velocity: {low 0.057, high 0.943}` overall, with 35 sessions at exactly `[1.0, 1.0]` (100% class 1) and 5 sessions at exactly 0.500/0.500. The decoder's 0.941 "balanced accuracy" on this output is an artefact of this degeneracy.

ii.
```python
tongue_valid = tongue_vel[:, trial_indices]
tongue_thresh = np.nanpercentile(tongue_valid[~np.isnan(tongue_valid)], 50) if np.any(~np.isnan(tongue_valid)) else 0
...
tv = tongue_vel[:, trial_idx].copy()
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
# Handle NaN: set to 0 (below threshold)
tv_disc[np.isnan(tv)] = 0
```

iii. The AI diagnosed the problem in detail (trajectory steps 56–60): "The threshold is 0.00, which means the 50th percentile is 0 … Since velocity magnitude is always >= 0, the condition `>= 0` is always true"; "Only 81/1792 frames have visible tongue data for trial 0 … The 50th percentile is 0.0000 because most values are 0". It considered an epsilon, a strict `>`, and percentiles over non-zero values only, then abandoned the fix: "The task specification says to discretize with 50th percentile threshold." CONVERSION_NOTES Step 7 records the symptom as expected — "tongue_velocity is very imbalanced (0.6% low, 99.4% high) - this is expected since tongue is rarely visible" — and Step 12 repeats it without correction.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A per-session video-clock correction is computed once by `compute_video_offset`, reproducing `findVideoOffset.m`: `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`. Frame times are then mapped to go-cue time as `frameTimes − vidshift − goCue[trix]`, and the tracked positions are linearly interpolated **at** the 1001 neural bin times, so the two streams share one axis by construction. If `frameTimes` is missing or all-NaN the fallback is a synthetic `(1..n_frames)/400` axis. If `sglx` is absent, `vidshift` falls back to 0.

ii.
```python
def compute_video_offset(session):
    """Compute video-neural offset matching findVideoOffset.m."""
    ...
    vidFileOffset = float(stats.mode(bitcode_bitstart_vals, keepdims=False).mode) / session['sglx']['fs']
    vidshift = vidFileOffset - bitStart_mode
    return vidshift
```
```python
if trial['frameTimes'] is not None and not np.all(np.isnan(trial['frameTimes'])):
    frame_times = trial['frameTimes']
else:
    n_frames = ts.shape[2]
    frame_times = np.arange(1, n_frames + 1) / VIDEO_FS
# Align time to goCue
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
```

iii. CONVERSION_NOTES Step 1: "findVideoOffset … Computes video-neural offset: mode(bitcode.bitstart)/fs - mode(bitStart)"; "findPosition … Interpolates DLC positions to neural time axis aligned to goCue". Step 6: "Video offset computation matching findVideoOffset.m". The synthetic-frameTimes fallback mirrors `findPosition.m`'s `traj(trix).frameTimes = (1:size(traj(trix).ts,1))./400`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` tracking, **camera 1 (bottom/front view), feature `top_paw`**, channels x and y. `bottom_paw` is catalogued but not used. Same auxiliary fields as the tongue (`frameTimes`, `NdroppedFrames`, `goCue`, `vidshift`).

ii.
```python
# ---- Extract paw velocity (top_paw from camera 1) ----
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. Trajectory step 54: "Paw at camera 1 indices 4-5 … Paw velocity: use top_paw from camera 1 (x,y velocity magnitude)". No stated reason for preferring `top_paw` over `bottom_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same `extract_feature_velocity` path as the tongue, but taking the **non-tongue** branches: (1) trials with NaN `NdroppedFrames` are skipped; (2) x and y are interpolated onto the 1001-bin axis; (3) **residual NaNs in the interpolated position are nearest-filled** (`_fill_nearest`), matching `findPosition.m`'s `fillmissing(...,'nearest')`; (4) velocity is `np.gradient` of position (again in units per bin, not per second); (5) a **baseline derivative is subtracted** — `np.nanmedian(np.diff(xpos))` from `xv` and `np.nanmedian(np.diff(ypos))` from `yv` (the MATLAB original subtracts `basederiv(1)`, the *x* baseline, from both components; the AI subtracts each axis's own baseline); (6) NaN velocities are nearest-filled; (7) the speed is `sqrt(xv² + yv²)`. The reference code applies `mySmooth(ts, 1, 'reflect')` to non-tongue positions, which with N=1 is a no-op, and the AI correctly treats it as such.

ii.
```python
# Fill missing for non-tongue
if not is_tongue:
    mask_nan = np.isnan(xpos[:, trix])
    if np.any(mask_nan) and not np.all(mask_nan):
        xpos[:, trix] = _fill_nearest(xpos[:, trix])
        ypos[:, trix] = _fill_nearest(ypos[:, trix])
...
if not is_tongue:
    # Subtract baseline derivative
    base_x = np.nanmedian(np.diff(xpos[:, trix]))
    base_y = np.nanmedian(np.diff(ypos[:, trix]))
    xv = xv - base_x
    yv = yv - base_y
    # Fill missing
    xv = _fill_nearest(xv)
    yv = _fill_nearest(yv)
```

iii. Trajectory step 37: "Uses gradient() on position (which in MATLAB computes central differences); For non-tongue features, subtracts baseline derivative; … For non-tongue: NaN velocity filled with nearest value". Inline comment in the script: "In the reference code, non-tongue features are smoothed with mySmooth(ts, 1, 'reflect'); N=1 means no smoothing". Step 6 of CONVERSION_NOTES: "Velocity computation matching findVelocity.m".

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile of the paw speed pooled over all bins and kept trials; `>= threshold` → 1, otherwise 0; NaN → 0. Again only **two** classes (`['low','high']`); the specified `2 = not visible` class is **not implemented** — and because the pipeline nearest-fills untracked positions and velocities, essentially no bin remains NaN anyway. The resulting split is a clean median split: verification reports `paw_velocity: {low 0.506, high 0.494}` and per-session fractions all near 0.500.

ii.
```python
paw_valid = paw_vel[:, trial_indices]
paw_thresh = np.nanpercentile(paw_valid[~np.isnan(paw_valid)], 50) if np.any(~np.isnan(paw_valid)) else 0
...
pv = paw_vel[:, trial_idx].copy()
pv_disc = np.zeros(len(time_axis), dtype=np.int64)
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. Step 5's mapping table: "paw velocity | output[4] | Discretize 50th pct | Time-varying". Trajectory step 71: "paw_velocity is reasonable (54% low, 46% high)". The AI never discusses the missing "not visible" class for any of the three movement outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue: the session's `vidshift` from `findVideoOffset` is subtracted from `frameTimes`, then the trial's `goCue`, and the positions are linearly interpolated at the 1001 neural bin times. The paw uses camera 1's own `frameTimes`, so a camera-specific frame count is handled correctly.

ii.
```python
cam_data = session['traj'][cam_idx]      # cam_idx == 1 for the paw
...
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

iii. Same as 7-d — one shared `extract_feature_velocity` implementation "matching findPosition + findVelocity", with the session-level `vidshift` computed once in `process_session` and passed in.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone sidecar file `motionEnergy_<ANM>_<DATE>.mat` next to the data structure, read with `scipy.io.loadmat`. `load_motion_energy` handles **three** on-disk layouts: (a) `me` is directly an (n_trials, 1) cell array; (b) `me` is a struct `{data, moveThresh}` whose `data` is the per-trial cell array; (c) `me.data` is itself a struct, so the real data is `me.data.data` (double-nested). One trace per trial, one value per camera frame. Frame times come from camera 0 (`session['traj'][0]`). The in-object copy `obj.me` is not used. `moveThresh` is parsed but never used for discretisation.

ii.
```python
if me_raw.dtype.names and 'data' in me_raw.dtype.names:
    me_data_field = me_raw['data'][0, 0]
    # MATLAB code: if isstruct(me.data); me.data = me.data.data; end
    if hasattr(me_data_field, 'dtype') and me_data_field.dtype.names and 'data' in me_data_field.dtype.names:
        me_data_arr = me_data_field['data'][0, 0]
        ...
    else:
        me_data_arr = me_data_field
else:
    # Direct format: me is the trial data array directly
    me_data_arr = me_raw
    me_thresh = 0
```

iii. Trajectory steps 122–124: "The ME data structure varies between sessions … The loadMotionEnergy.m code handles this: `if isstruct(me.data); me.data = me.data.data; end`", followed by the discovery of the bare-cell-array variant in JEB23_2023-10-10. This was a *fix*: before it, seven sessions had `ME threshold = 0.0000` and a constant motion-energy output; after it, all 43 sessions have a genuine median split (verification: per-session fractions all ≈0.500).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling — motion energy is already one scalar per frame. Per trial the trace is linearly interpolated onto the 1001-bin axis (`interp1d`, NaN outside the frame range), and residual NaNs are then **nearest-filled** per trial. Traces and frame-time vectors of unequal length are truncated to the shorter of the two; trials with fewer than 2 usable samples are left as NaN.

ii.
```python
n = min(len(trial_me), len(aligned_times))
if n < 2:
    continue
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
               bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_me(time_axis)
...
# Fill NaN with nearest
for trix in range(Ntrials):
    if not np.all(np.isnan(me_aligned[:, trix])):
        me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. Docstring "Align motion energy to goCue, matching loadMotionEnergy.m". Trajectory step 33: "1. Load from motionEnergy_*.mat file; 2. Find video offset using findVideoOffset; 3. Align to goCue and interpolate to same time axis as neural data (obj.time); 4. Fill NaN values with nearest" — the reference's `me.data = fillmissing(me.data,'nearest')` is reproduced exactly.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile over all bins and kept trials; `>= threshold` → 1, otherwise 0; NaN → 0. Only **two** classes (`['low','high']`); the specified `2 = no video` class is **not implemented**, so a session without usable motion energy would silently come out as 100% class 0 rather than 100% "no video". After the 9-a loader fix every session does have data, and the distribution is a clean median split: `motion_energy: {low 0.501, high 0.499}`.

ii.
```python
me_valid = me_aligned[:, trial_indices]
me_thresh = np.nanpercentile(me_valid[~np.isnan(me_valid)], 50) if np.any(~np.isnan(me_valid)) else 0
...
me = me_aligned[:, trial_idx].copy()
me_disc = np.zeros(len(time_axis), dtype=np.int64)
me_disc[me >= me_thresh] = 1
me_disc[np.isnan(me)] = 0
```

iii. Step 5 mapping: "motion energy | output[5] | Discretize 50th pct | Time-varying". CONVERSION_NOTES Step 9 flags the pre-fix symptom ("Some sessions have ME threshold = 0 (missing motion energy data)"), and trajectory step 130 records the resolution: "the motion energy fix worked perfectly - all sessions now have ME fractions around 0.500, with no all-zero or all-one sessions". Note the reference code's own `me.moveThresh` (the authors' movement threshold) was deliberately not used, because the Decoder Task specifies a 50th-percentile split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same clock correction and same grid as the tracking: `frameTimes(camera 0) − vidshift − goCue[trix]`, then linear interpolation at the 1001 neural bin times. Motion energy has one value per side-camera frame, so camera 0's frame times are the right ones. If `frameTimes` is missing or all-NaN the fallback builds `(1..n_frames)/400` and subtracts a **fixed 0.5 s** instead of `vidshift` — exactly the fallback branch of the reference `loadMotionEnergy.m`.

ii.
```python
cam_data = session['traj'][0]  # camera 0
trial_traj = cam_data['trials'][trix] if trix < len(cam_data['trials']) else None
if trial_traj is not None and trial_traj['frameTimes'] is not None and not np.all(np.isnan(trial_traj['frameTimes'])):
    frame_times = trial_traj['frameTimes']
    aligned_times = frame_times - vidshift - goCue[trix]
else:
    n_frames = trial_me.shape[0]
    ...
    frame_times = np.arange(1, n_frames + 1) / VIDEO_FS
    aligned_times = frame_times - 0.5 - goCue[trix]
```

iii. The reference `loadMotionEnergy.m` reads `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)` with `catch` → `frameTimes = (1:size(obj.traj{1}(trix).ts,1))./400; interp1(frameTimes-0.5-alignTimes(trix), …)`. The AI's code reproduces both branches, including the literal `-0.5`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several distinct cases, handled as follows.
- **Two MATLAB file formats**: header sniffing dispatches to an h5py or a scipy.io reader; `ts` is transposed in the v5 reader so both paths produce `(features, 3, frames)`.
- **Probe slots with no cluster struct** (JEB6 probe 1 is a bare `uint64` dataset): detected and replaced with an empty unit list rather than crashing.
- **Sessions with no `clu` field at all** (JEB24_2023-10-03/04): the exception propagates to the per-session `try/except` in `main`, the session is skipped with a traceback, and the run continues.
- **Three motion-energy file layouts**: unwrapped as described in 9-a.
- **Trials with NaN `NdroppedFrames`**: skipped for kinematics (left NaN → class 0), mirroring `findPosition.m`.
- **Missing / all-NaN `frameTimes`**: synthetic 400 Hz frame times.
- **Missing `reward` / `bitStart` fields**: filled with NaN; a missing `sglx` gives `vidshift = 0`.
- **NaN positions/velocities**: nearest-filled for the paw, zeroed for the tongue, nearest-filled for motion energy; any NaN surviving into discretisation becomes class 0.
- **Trials past the end of the ephys recording**: **not handled** — 30 trials in 2 sessions enter the dataset with all-zero firing rates for every neuron, flagged as warnings by the verifier and knowingly left in.

ii.
```python
# Skip probes with invalid clu data (not a group or missing 'tm')
if not isinstance(clu_group, h5py.Group) or 'tm' not in clu_group:
    clu_list.append([])  # empty unit list for this probe
    continue
```
```python
if trial['ts'] is None:
    continue
if np.isnan(trial['NdroppedFrames']):
    continue
```
```python
try:
    result = process_session(sess_info, time_axis, show_processing=args.show_processing)
    ...
except Exception as e:
    print(f'  -> ERROR: {e}')
    traceback.print_exc()
```

iii. CONVERSION_NOTES Step 10 Check 5: "Handled probes with invalid clu data (JEB6 probe 1); Handled MATLAB v5 vs v7.3 format differences; Handled sessions without clu data (JEB24_2023-10-03/04)". Step 10 Check 1 on the zero-neural trials: "These are likely trials at the end of the recording session where the recording had ended. The decoder should handle these gracefully." The NaN-handling rules for kinematics are justified as copies of `findPosition.m`/`findVelocity.m`. Note also that `warnings.filterwarnings('ignore')` is set globally at import and several `except:` clauses are bare, so some failures are silent.

## 11-a. What are the most time-consuming steps of the code?

i. The script prints a timing breakdown for every session. Two steps dominate roughly equally: **reading the `.mat` file** (2.6–6 s/session) and **spike binning + smoothing** (2.3–6 s/session, up to ~12 s for the 142-unit two-probe sessions). Kinematics are much cheaper (tongue ≈0.1 s, paw 0.3–0.8 s, motion energy ≈0.1 s). Full conversion of 47 candidate sessions took **≈5.1 minutes**, comfortably under the 15-minute budget, so no optimisation was pursued.

ii.
```python
t0 = time.time()
session = load_session_data(sess_info['filepath'])
t_load = time.time() - t0
print(f'    Loaded in {t_load:.1f}s')
...
t_spk = time.time() - t0
print(f'    Spike processing: {t_spk:.1f}s, {n_neurons} total units')
...
print(f'    Tongue velocity: {t_tongue:.1f}s')
print(f'    Paw velocity: {t_paw:.1f}s')
print(f'    Motion energy: {t_me:.1f}s')
```

iii. CONVERSION_NOTES Step 7: "~8s per session, ~5 min for full 47 sessions". Trajectory step 72: "Each session takes ~5-11 seconds. With 47 sessions, that's roughly 47 * 8 = 376 seconds ≈ 6 minutes. Well within the 15-minute limit." The estimate proved accurate (5.1 min actual), so the AI never revisited the bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three places.
1. **`process_spikes`**: a nested `for unit … for trial` loop that rebuilds a full boolean mask `trial_nums == trial_num` over the unit's entire spike vector on *every* trial, i.e. O(n_units × n_trials × n_spikes) work, then calls `np.histogram` and `np.convolve` once per (unit, trial). All of this collapses to one 2-D histogram over (trial, time) per unit plus one `gaussian_filter1d`/`fftconvolve` over the whole matrix.
2. **`_fill_nearest`**: for each NaN index it computes `np.abs(valid - i)` over all valid indices and takes the argmin — O(n²) per 1001-sample trace, called for x, y, xv, yv of the paw and once for motion energy on every trial. A `np.searchsorted` or forward/backward-fill would be O(n).
3. **Per-trial `interp1d` construction** in `extract_feature_velocity` and `align_motion_energy`, and the per-trial `np.gradient` loop — these could be batched where frame counts match, and the smoothing convolution could be applied to the whole `(n_time, n_trials)` matrix at once.

ii.
```python
for i, u_idx in enumerate(valid_units):
    ...
    for j in range(Ntrials):
        trial_num = j + 1  # MATLAB 1-indexed
        spk_mask = trial_nums == trial_num
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr = counts.astype(np.float64) / DT
        fr_smooth = smooth_signal(fr, SMOOTH_N)
```
```python
for i in range(len(arr)):
    if mask[i]:
        dists = np.abs(valid - i)
        arr[i] = arr[valid[np.argmin(dists)]]
```

iii. The instructions asked for vectorised loops, and Step 6's "Code inefficiencies identified / Code speedups added" slots in CONVERSION_NOTES were left unfilled. The AI's stated justification for stopping was budget-based only ("Well within the 15-minute limit"). The nested spike loop is in fact a direct transliteration of the reference `getSeq.m` MATLAB loop, which is the likely reason it was written that way.

## 11-c. What processing does the code repeat multiple times?

i. Little is recomputed structurally — each file is read once, `vidshift` is computed once per session and passed into all three video streams — but there are several redundancies.
- **Work on discarded trials**: firing rates, tongue velocity, paw velocity and motion energy are all computed for **all `Ntrials`**, and only afterwards is the ~10–25% of trials that are early/stim/ignore dropped.
- **`compute_video_offset` computes `bitStart_mode` twice**: once with `np.median` and immediately again with `scipy.stats.mode`, the first result being discarded.
- **Redundant per-trial dereferencing** in the v5 loader: `t = cam_raw[0, trix]` is re-fetched in three separate `try` blocks for `ts`, `frameTimes` and `NdroppedFrames`.
- **Discretisation is recomputed in the plotting path** (`plot_processing` re-derives `tv_disc`/`pv_disc`/`me_disc` from the continuous arrays instead of reusing the stored outputs — and, notably, it plots a *different* rule than the one that is saved, since it omits the NaN→0 step).
- `np.gradient` is run over positions that were already nearest-filled, then the result is nearest-filled again.

ii.
```python
bitStart_mode = float(np.median(session['bp']['ev']['bitStart'][~np.isnan(...)]))
# Use scipy.stats.mode equivalent
from scipy import stats
bitStart_vals = session['bp']['ev']['bitStart'][~np.isnan(...)]
if len(bitStart_vals) > 0:
    bitStart_mode = float(stats.mode(bitStart_vals, keepdims=False).mode)
```
```python
firing_rates = np.zeros((len(valid_units), n_time, Ntrials), dtype=np.float32)   # all trials
...
trial_indices = np.where(valid_trials)[0]                                        # subset only here
```

iii. Not discussed in CONVERSION_NOTES. Computing everything over `Ntrials` and subsetting late does have a defensible side: the low-firing-rate criterion and the per-session percentile thresholds are session-level statistics, and computing kinematics for all trials keeps trial indices aligned with the raw arrays — though the AI in fact computes the percentiles on the *filtered* subset (`tongue_vel[:, trial_indices]`), so the extra trials are pure waste.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly in the loading stage.
- **`clu.tm`** — the full session-clock spike-time vector of every cluster — is read for every unit and never used (only `trial` and `trialtm` are needed).
- **`bp.ev.sample`, `bp.ev.delay`, `bp.ev.reward`** are read and never used; `bp.L` and `bp.no` are read but used only inside the session-inclusion test / trial mask.
- **All DLC features of both cameras** are materialised per trial (`traj['ts']`, 7 features × 3 channels on the side view, 10 on the bottom view) although only `tongue` and `top_paw` are ever indexed; likewise `NdroppedFrames` is kept per trial but used only as a validity flag.
- **`me.moveThresh`** is parsed (including from the double-nested layout) and never used, since the Decoder Task mandates a 50th-percentile split instead.
- **Firing rates and all three kinematic streams are computed for trials that are then discarded** (see 11-c) — roughly 3,200 of 15,000 trials.
- The interpolated *positions* `xpos`/`ypos` are only ever used to produce velocity; they are not saved.

ii.
```python
unit['tm'] = np.array(f[clu_group['tm'][u, 0]]).flatten()      # never used downstream
...
bp['ev']['sample'] = np.array(f['obj/bp/ev/sample']).flatten()  # never used
bp['ev']['delay'] = np.array(f['obj/bp/ev/delay']).flatten()    # never used
```
```python
ts = np.array(f[ts_ref])  # shape: (n_feats, 3, n_timepoints) — all features kept
trial_data['ts'] = ts
```
```python
me_thresh = float(me_raw['moveThresh'][0, 0].flatten()[0])      # parsed, never used
return {'data': me_trials, 'moveThresh': me_thresh}
```

iii. Not discussed in CONVERSION_NOTES. Reading whole structs is the natural consequence of the "load the session into a uniform dict, then process" design, and the loading cost is a fixed ~3–6 s/session either way; the AI's timing notes treat loading as irreducible. The computation of discarded trials is the more substantive waste and is not acknowledged anywhere.
