# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 12 sessions (`SESSION_SPECS`), each an `(animal, date, probes)` triple transcribed from the author loader scripts that `Scripts/Figure 8/Figure8a_thru_c.m` calls (`loadJEB6/JEB7/EKH1/EKH3/JGR2/JGR3/JEB19_ALMVideo.m`), in the loader's own order (JEB19's dates are listed newest-first because its loader is). `DATA_ROOT` is fixed to `/app/data/Ephys_Behavior`; the `RandomizedDelay_Ephys_Behavior` folder and the other 13 fixed-delay sessions present there (JEB13 ×5, JEB14 ×4, JEB15 ×4) are never opened. Each session is one MATLAB v7.3 file read with `h5py` using targeted field/reference reads rather than materialising the whole `obj` tree; the paired `motionEnergy_<anm>_<date>.mat` (v5) is read with `scipy.io.loadmat`. The result is 12 sessions, 7 subjects, 3,116 trials and 521 units.

ii.
```python
DATA_ROOT = Path("/app/data/Ephys_Behavior")
...
# Exact Figure 8 loader order. JEB19's loader lists dates in reverse order.
SESSION_SPECS = (
    SessionSpec("JEB6", "2021-04-18", (2,)),
    SessionSpec("JEB7", "2021-04-29", (1,)),
    ...
    SessionSpec("JEB19", "2023-04-18", (1,)),
)
```
```python
def convert_session(spec: SessionSpec) -> tuple[dict, dict]:
    with h5py.File(spec.data_path, "r") as handle:
        bp = load_behavior(handle)
        ...
```
```python
def direct_field(handle: h5py.File, path: str, dtype=None) -> np.ndarray:
    array = np.asarray(handle[path]).ravel(order="F")
    return array.astype(dtype, copy=False) if dtype is not None else array
```

iii. From CONVERSION_NOTES Step 4/Step 5: "Figure 8 context scripts load JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19: 12 sessions … Exactly those 12 have substantial WC blocks; DR-only JEB13/14/15 and randomized-delay files do not represent the requested balanced context task", and the paper's "In 12 sessions from six mice, animals performed the two-context task. In total, 522 units (214 well-isolated single units) were recorded". Key Decision 1: "Cohort is the 12-session two-context set: It exactly matches Figure 8's executable session list and the paper's session count, and supplies both WC/DR classes." The randomized-delay folder is excluded because "It has only 99/7,583 WC-labeled trials, so it is principally a randomized-delay DR dataset". `mat73`-style whole-file loading was tried and abandoned as "slow/memory-heavy", motivating the direct HDF5 field reads.

## 1-b. How are the data split into subjects?

i. The subject is the animal prefix of the session name, taken from the hard-coded `SessionSpec.animal` field (equivalently the part of the filename before the underscore). `subjects` is the list of unique animals in first-appearance (loader) order and `subject_idx` indexes each session into it. Seven distinct IDs result — JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19 — which the AI deliberately keeps even though the paper reports six mice for this cohort.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probes: tuple[int, ...]

    @property
    def session_id(self) -> str:
        return f"{self.animal}_{self.date}"
```
```python
subjects = list(dict.fromkeys(spec.animal for spec in specs))
subject_idx = np.array([subjects.index(spec.animal) for spec in specs], dtype=np.int64)
```

iii. Step 4 discrepancy table: "Context loader names contain 7 unique IDs … Paper says six mice → Preserve the seven explicit source IDs in `subjects`; collapsing distinct IDs would fabricate identity. Treat the one-mouse difference as a paper/release-version reporting discrepancy." The animal identity is taken from the filename/loader metadata rather than `obj.meta`.

## 1-c. How are the data split into sessions?

i. One session = one `SessionSpec` = one `data_structure_<anm>_<date>.mat` file plus its `motionEnergy_<anm>_<date>.mat` partner, converted by one call to `convert_session` and becoming one element of `neural`, `input`, `output`, `brain_region_idx` and `metadata['session_info']`. The probe(s) named by the author loader are used, so a two-probe session would be concatenated (none of the 12 selected sessions uses two probes). Session-level guards reject a session with fewer than two retained trials or fewer than 10 retained units (the paper's session-inclusion rule).

ii.
```python
    @property
    def data_path(self) -> Path:
        return DATA_ROOT / f"data_structure_{self.session_id}.mat"

    @property
    def motion_path(self) -> Path:
        return DATA_ROOT / f"motionEnergy_{self.session_id}.mat"
```
```python
        if len(raw_trials) < 2:
            raise ValueError(f"{spec.session_id}: fewer than two retained trials")
        ...
        if len(clusters) < 10:
            raise ValueError(f"{spec.session_id}: only {len(clusters)} retained units")
```

iii. Same as 1-a: the author `load<ANM>_ALMVideo.m` scripts define which sessions and probes enter the analysis, and Figure 8 (the two-context figure) enumerates exactly these 12. The ≥10-unit guard comes from Methods: "Recording sessions were included for analysis only if they had at least 10 units."

## 1-d. How are the data split into trials?

i. A trial is one entry of the per-trial `obj.bp` arrays. `load_behavior` reads `R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `ev.goCue` and `ev.bitStart`, and asserts that every one of them has exactly `obj.bp.Ntrials` entries, that `hit+miss+no == 1` on every trial, that `R+L == 1`, and that every go cue is finite. Trial identity is carried as 0-based indices into those arrays (`raw_trials`), and spikes/frames/motion-energy traces are attached to trials through their own stored trial index (`clu.trial`, one `traj` entry per trial, one `me.data` cell per trial).

ii.
```python
def load_behavior(handle: h5py.File) -> dict[str, np.ndarray]:
    fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
    bp = {field: direct_field(handle, f"obj/bp/{field}", bool) for field in fields}
    bp["stim"] = direct_field(handle, "obj/bp/stim/enable", bool)
    bp["goCue"] = direct_field(handle, "obj/bp/ev/goCue", np.float64)
    bp["bitStart"] = direct_field(handle, "obj/bp/ev/bitStart", np.float64)
    ntrials = int(np.asarray(handle["obj/bp/Ntrials"]).squeeze())
    if any(len(value) != ntrials for value in bp.values()):
        raise ValueError("Behavior fields do not all match Ntrials")
    outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
    if not np.all(outcome_sum == 1):
        raise ValueError("hit/miss/no are not mutually exhaustive")
```
```python
    if len(motion_raw) != len(bp["hit"]):
        raise ValueError("Motion-energy trial count differs from behavior")
```

iii. Step 2: "`obj.bp` has length-`Ntrials` float64/logical arrays … Every static-context session has a finite go cue for every trial. Outcomes are mutually represented by hit/miss/no in the inspected totals." The Bpod table already defines trials, so no trial boundaries are inferred; the assertions were added so that any file violating those assumptions fails loudly rather than being silently mis-split.

## 1-e. How are trials filtered based on quality controls?

i. Exactly two exclusions, applied once up front: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`). Everything else is kept, explicitly including `no`/ignore trials, which the paper drops but which the decoder spec requires (the `ignore` outcome class and the `none` lick-direction class). Across the 12 sessions this keeps 3,116 of 3,626 trials. No trial is dropped for video, motion-energy or recording-length reasons; the AI verified that all selected sessions have spikes on every retained trial (I independently confirmed the minimum per-trial mean rate is 2.1 Hz, i.e. no empty trials, so the expert's extra "trial past end of recording" cut is not needed for this cohort).

ii.
```python
        keep_trials = ~bp["early"] & ~bp["stim"]
        raw_trials = np.flatnonzero(keep_trials)
        if len(raw_trials) < 2:
            raise ValueError(f"{spec.session_id}: fewer than two retained trials")
```

iii. Step 4: "Reference condition code also removes stimulation-enabled trials … Exclude stimulation and early trials. Retain hit, miss, and no/ignore because the requested output explicitly requires all three outcomes and `none` lick direction." Methods support the early-lick rule ("Trials in which the animal contacted the lickport before the reward ('early lick') were omitted from analyses") and the Figure 8 `params.condition` strings all carry `~stim.enable`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the author-specified probe only: per cluster, `quality` (manual curation label), `trial` (1-based trial index of each spike) and `trialtm` (spike time relative to that trial's start). The alignment variable `obj.bp.ev.goCue` is the second ingredient. `clu.tm`, `clu.spkWavs` and `clu.site` are not read.

ii.
```python
        group = handle[probe_refs[probe - 1]]
        quality_refs = matlab_cell_refs(group["quality"])
        trial_refs = matlab_cell_refs(group["trial"])
        trialtm_refs = matlab_cell_refs(group["trialtm"])
        for cluster_index, quality_ref in enumerate(quality_refs):
            quality = deref_string(handle, quality_ref).strip().lower()
            if quality in QUALITY_EXCLUDE:
                continue
            trials = deref_vector(handle, trial_refs[cluster_index], np.int64) - 1
            trial_times = deref_vector(handle, trialtm_refs[cluster_index], np.float64)
```

iii. Step 1/Step 2: "This is extracellular electrophysiology, not calcium imaging; delta-F/F is not applicable. Neural source data are sorted spike times in `obj.clu{probe}(cluster)`", and `alignSpikes.m` "Computes per-spike aligned time as raw within-trial spike time minus the per-trial alignment event (go cue here)", i.e. `trialtm` is the field the reference itself aligns.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into the 5 ms grid with `np.add.at`, divided by the bin width to give spikes/s, and smoothed along time with the reference's *causal* Gaussian: MATLAB `gausswin(15)` (σ = (15−1)/(2·2.5) = 2.8 samples), first seven coefficients zeroed, renormalised, convolved in `same` mode after prepending the first 15 samples of the trace and then trimming that prefix — implemented as an FIR `lfilter` with the surviving eight taps. No normalisation, baseline subtraction or z-scoring; stored values are firing rates in Hz as `float32`.

ii.
```python
def causal_gaussian_kernel(n: int = 15) -> np.ndarray:
    """Match MATLAB gausswin(n, 2.5), then the released causalization."""
    kernel = gaussian(n, std=(n - 1) / (2 * 2.5), sym=True).astype(np.float32)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    return kernel

def reference_smooth(values: np.ndarray, n: int = 15) -> np.ndarray:
    """Match mySmooth(..., 15, 'reflect') on the last (time) dimension."""
    values = np.asarray(values, dtype=np.float32)
    padded = np.concatenate((values[..., :n], values), axis=-1)
    causal_coefficients = KERNEL[n // 2 :]
    filtered = lfilter(causal_coefficients, [1.0], padded, axis=-1)
    return filtered[..., n:].astype(np.float32, copy=False)
```
```python
        bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
        in_window = (bins >= 0) & (bins < N_TIME)
        np.add.at(counts[unit], (mapped_trials[in_window], bins[in_window]), 1)
    return reference_smooth(counts / DT)
```

iii. Step 4: "`mySmooth` uses MATLAB `gausswin(15)`, zeros its first seven coefficients, normalizes, and convolves causally … Reproduce the 15-bin 5-ms causal Gaussian exactly (effective history about 35 ms), with the reference leading boundary handling." Key Decision 5 spells out the kernel derivation. Step 6 records the verification: "The causal smoother was numerically compared to explicit `np.convolve` and matched to float32 precision (maximum absolute difference `9.54e-7`)." Raw firing rates rather than z-scores are stored because "the supplied decoder handles its own normalization/training" (Step 3).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both copied from the reference. (1) Cluster quality: the label is stripped and lower-cased and dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the four labels `findClusters(...,'all')` excludes; every other label (including `poor`, `fair` and `multi`) is kept. (2) Low firing rate: for each surviving cluster the AI rebuilds the Figure 8 condition PSTHs (the same seven `params.condition` strings, `tmin=-3`, `tmax=2.5`, `dt=1/100`, causal-smoothed), takes the grand mean across conditions and time as in `removeLowFRClusters`, and keeps units with mean rate strictly `> 1` Hz (Figure 8's `params.lowFR = 1`). This yields 521 units (mean 43.4/session, range 27–67), of which 214 carry single-unit labels.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0
```
```python
def low_fr_filter(clusters, bp):
    """Recreate the Figure 8 context-workflow low-FR calculation."""
    conditions = (
        np.ones(ntrials, dtype=bool),
        hit & ~stim & ~aw,
        hit & ~stim & aw,
        miss & ~stim & ~aw,
        miss & ~stim & aw,
        hit & ~stim & ~aw & ~early,
        hit & ~stim & aw & ~early,
    )
    edges = np.arange(-3.0, 2.5 + 0.005, 0.01, dtype=np.float64)
    for unit, cluster in enumerate(clusters):
        aligned = cluster["trial_times"] - bp["goCue"][spike_trials]
        for condition in conditions:
            counts = np.histogram(aligned[use], bins=edges)[0].astype(np.float32)
            rate = counts / (float(np.sum(condition)) * 0.01)
            condition_psths.append(reference_smooth(rate))
        mean_rates[unit] = np.nanmean(np.stack(condition_psths))
    keep = mean_rates > LOW_FR_HZ
```

iii. Step 4: "Most figure scripts use `lowFR=1`; only generic defaults use 0.5 … Use strict `>1 Hz`; figure scripts and paper override the generic default", backed by Methods: "All units with firing rates exceeding 1 Hz were included in all other analyses." Step 9 records the cross-check: "Selected 12 sessions have 528 quality-eligible units … A direct recreation of the code-style PSTH criterion retains 521 units, including exactly 214 single units" against the paper's 522/214, "consistent to one released unit".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction. `clu.trialtm` is already on the behaviour clock relative to its own trial's start and `bp.ev.goCue` is on the same clock, so each spike's aligned time is `trialtm − goCue[trial]`; spikes falling outside [−2.5, 2.5) s are simply dropped. No interpolation, no per-session offset (the camera streams need one, the spikes do not).

ii.
```python
        spike_trials = spike_trials[use]
        mapped_trials = mapped_trials[use]
        aligned = cluster["trial_times"][use] - bp["goCue"][spike_trials]
        bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
        in_window = (bins >= 0) & (bins < N_TIME)
```

iii. Step 1: `alignSpikes` "Computes per-spike aligned time as raw within-trial spike time minus the per-trial alignment event (go cue here)", with `params.alignEvent = 'goCue'` in both `getDefaultParams.m` and the Figure 8 script. Step 3 notes that the same `goCue` event field also holds the water-drop time on WC trials, so alignment is uniform across contexts. Step 10 edge-case check: "all retained spike alignments … There are zero exact boundary spikes, removing last-edge ambiguity."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins, 1000 of them spanning [−2.5, +2.5) s from the go cue, identical for every trial and session and shared by the neural, input and all three video-derived output streams. The stored time vector is the bin centres, −2.4975 … +2.4975 s. There is no rebinning of an already-binned stream: spikes are counted directly into this grid, and video/motion-energy samples are interpolated onto the same grid. A coarser 10 ms, −3 to 2.5 s grid is used *only* inside the low-FR unit filter, to reproduce the Figure 8 PSTH parameters.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.005
TIME = (np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2).astype(np.float32)
N_TIME = len(TIME)
```
```python
            "time_bin_size": 5.0,
            "off_start": TMIN,
            "off_end": TMAX,
```

iii. Step 4: "Some illustrative scripts use 10 ms; generic defaults and several figures use 5 ms … Single-trial analysis explicitly uses 5 ms → Use 5-ms bins", and "Context figure scripts sometimes start at −3 s; generic loader/default and primary single-trial workflows use −2.5 to +2.5 s → Use bin centers over [−2.5,+2.5) (1000 bins), matching `getDefaultParams/getSeq`". Step 1 confirms the centre convention: "`getSeq` time centers are `tmin + dt/2` through `tmax − dt/2`; therefore 1000 time points at 5 ms."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing is read from the files for this variable: it is the analysis grid itself, defined by the alignment window and bin size, i.e. implicitly by `bp.ev.goCue` which defines time zero. The same 1000-element `TIME` vector is emitted for every trial of every session.

ii.
```python
TIME = (np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2).astype(np.float32)
...
        input_trials = [TIME[None, :].copy() for _ in raw_trials]
```
```python
        "input_names": ["time_from_go_cue_s"],
```

iii. Step 5 mapping table: "Bin centers `−2.5+dt/2 : dt : 2.5−dt/2` → `input[0]` … Reference code function `getSeq` (`obj.time`); Name: `time_from_go_cue_s`; explicitly continuous per task, not an onset indicator." The Decoder Task specifies "Time from go cue onset in seconds (continuous, time-varying)", so the continuous axis rather than a binary onset indicator is used.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The vector is generated analytically as `arange(-2.5, 2.5, 0.005) + 0.0025`, cast to `float32`, and copied per trial into shape `(1, 1000)`. The converter re-checks at the end that every stored input equals that vector.

ii.
```python
            if not np.allclose(x[0], TIME):
                raise AssertionError(f"Session {session} trial {trial}: time input mismatch")
```

iii. Step 5 planned sanity check: "Exact input check with `np.allclose`: every trial input equals the analytically generated 1000 bin centers and contains zero between the two central bins as expected for bin centers." Verified in Step 10 check 2.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spike times are expressed relative to the same trial's go cue and floor-binned into `[TMIN + k·DT, TMIN + (k+1)·DT)`, and the input value at index k is the centre of that same interval, so bin k denotes the same interval in both streams. All video-derived outputs are interpolated onto the same `TIME`, so all four streams share one axis.

ii.
```python
        aligned = cluster["trial_times"][use] - bp["goCue"][spike_trials]
        bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
```
```python
        x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
```

iii. Step 10: the independent audit "independently bins raw spikes … and matches all 1000 converted neural samples with `np.allclose` (maximum absolute error 0). The analytical input centers match with `np.allclose`."

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial `obj.bp` fields: the instructed/reward side `R` (with `L` checked as its complement) and the outcome flags `hit`, `miss`, `no`. The direction actually licked is not recorded, so it is inferred from the instructed side together with whether the animal was correct.

ii.
```python
    fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
    bp = {field: direct_field(handle, f"obj/bp/{field}", bool) for field in fields}
    ...
    side_sum = bp["R"].astype(int) + bp["L"].astype(int)
    if not np.all(side_sum == 1):
        raise ValueError("R/L are not mutually exhaustive")
```

iii. Step 4: "`R/L`, hit/miss/no jointly determine actual response … Required 'lick direction' is actual response: hit uses instructed side, miss uses opposite side, no/ignore maps to none." Key Decision 11: "Actual lick rather than instructed side … This implements the named output and avoids silently decoding stimulus side."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A per-trial relabelling with three classes (`left` 0, `right` 1, `none` 2): an ignore trial (`no`) is `none`; a hit is the instructed side; a miss is the opposite of the instructed side. The value is then broadcast across all 1000 time bins so it can share one `(6, T)` int8 output array with the time-varying outputs.

ii.
```python
    for kept, raw in enumerate(raw_trials):
        if bp["no"][raw]:
            lick = 2
        elif bp["hit"][raw]:
            lick = 1 if bp["R"][raw] else 0
        elif bp["miss"][raw]:
            lick = 0 if bp["R"][raw] else 1
        else:
            raise ValueError("Unrecognized trial outcome")
```
```python
        out = np.empty((6, N_TIME), dtype=np.int8)
        out[:3] = static[trial, :, None]
```

iii. Key Decision 6: "Decoder code permits per-trial 1-D labels but mixed static/time-varying outputs cannot share one array. Broadcasting lick/context/outcome across T preserves their per-trial semantics." Output values are declared as `["left", "right", "none"]`, matching the Decoder Task's "(left, right, none, per-trial)". Step 10 verified the mapping "for every one of 3,116 trials" against the raw `bp` arrays.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `obj.bp.autowater`: true on water-cued (WC) trials where water is delivered at a random port with all auditory cues omitted, false on delayed-response (DR) trials.

ii.
```python
    fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
    bp = {field: direct_field(handle, f"obj/bp/{field}", bool) for field in fields}
```

iii. Step 1: `getBlockNum_AltContextTask` "Treats `bp.autowater` as the WC/DR context indicator and finds context-switch blocks"; Step 4: "`autowater=1` is WC; 0 is DR". The AI also used the per-session autowater fraction as the empirical criterion for identifying the two-context cohort (Step 2: "1,449/8,260 trials have `autowater=1`").

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling, `WC = 0` if `autowater` else `DR = 1`, broadcast across the 1000 bins. Two classes only, matching the spec. Resulting distribution is 31.5% WC / 68.5% DR.

ii.
```python
        context = 0 if bp["autowater"][raw] else 1
```
```python
            ["WC", "DR"],
```

iii. Step 4: "Encode `[WC, DR]` as `[0,1]`" — the ordering is taken from the Decoder Task line "Behavioral context (WC, DR, per-trial)".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The three mutually exclusive per-trial flags `obj.bp.hit`, `obj.bp.miss` and `obj.bp.no`. All three are read (rather than inferring `no` as the complement), and their mutual exclusivity/exhaustiveness is asserted at load time.

ii.
```python
    outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
    if not np.all(outcome_sum == 1):
        raise ValueError("hit/miss/no are not mutually exhaustive")
```

iii. Step 1: `getOutcome` "Encodes hit as 1, miss as 0, and ignore (`bp.no`) as NaN in the reference analysis" — the AI keeps the hit/miss codes and promotes the reference's NaN to an explicit third class because the Decoder Task requires it.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-class relabelling broadcast across time: `incorrect` 0 on miss trials, `correct` 1 on hits, `ignore` 2 on `no` trials. Ignore trials are retained in the dataset rather than dropped, which is a deliberate departure from the paper's analyses.

ii.
```python
        outcome = 0 if bp["miss"][raw] else (1 if bp["hit"][raw] else 2)
```
```python
            ["incorrect", "correct", "ignore"],
```

iii. Step 3/Step 4: "The requested output explicitly includes `ignore`, so ignore trials must be retained despite their omission from paper analyses; this is a necessary task-specific exception. They also provide the required `none` lick-direction class." Class order follows the Decoder Task's "(incorrect, correct, ignore)".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — specifically the feature named `tongue`, taking its x and y columns from `ts` plus that camera's `frameTimes`. The bottom camera's `top_tongue` (and the other tongue landmarks in either view) are not used. `obj.sglx.bitcode.bitstart`, `obj.sglx.fs` and `bp.ev.bitStart` are also needed to put frames on the behaviour clock, and `bp.ev.goCue` to align.

ii.
```python
        tongue[kept_trial], tongue_visible[kept_trial], tongue_x, tongue_y = feature_speed(
            handle, side, int(raw_trial), "tongue", aligned_side_time, True
        )
```
```python
def trajectory_for_trial(handle, group, trial):
    ts_ref = matlab_cell_refs(group["ts"])[trial]
    ft_ref = matlab_cell_refs(group["frameTimes"])[trial]
    stored = deref_array(handle, ts_ref)
    # HDF5 dimension order is reversed relative to MATLAB: feature, coord, frame.
    trajectory = np.transpose(stored, (2, 1, 0)).astype(np.float64, copy=False)
    frame_times = deref_vector(handle, ft_ref, np.float64)
```

iii. Step 5 mapping table: "Side-view `traj.ts[:,:,tongue]` … Main side-camera `tongue` landmark is the paper's tongue-tip representation." This is the first entry of `params.traj_features{1}` in both `getDefaultParams.m` and the Figure 8 script. Step 7: "Tongue visibility is only about 9.4%, but this is expected: the source tracker stores tongue coordinates mainly during protrusions." No justification is given for preferring one view over combining the two.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, mirroring `findPosition.m`/`findVelocity.m`. (1) Frame times are put on the go-cue clock (7-d) and the tongue's x and y are linearly interpolated onto the 1000-bin `TIME` axis with `np.interp`, returning NaN outside the frame range and propagating NaN through untracked frames (DLC already stores NaN where likelihood is low), so no explicit likelihood cut is needed. (2) No smoothing and no gap filling are applied — the reference explicitly skips both for tongue features. (3) Speed is `hypot(gradient(x), gradient(y))` on the uniform 5 ms grid, i.e. pixels per bin; no baseline-derivative subtraction (the reference applies that only to non-tongue features). (4) Any bin whose x or y is NaN, or whose gradient is NaN because it abuts a gap, is marked not visible.

ii.
```python
    x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
    y = interpolate_with_nans(aligned_time, trajectory[:, 1, feature_index], TIME)
    visible = np.isfinite(x) & np.isfinite(y)
    if is_tongue:
        x_for_velocity, y_for_velocity = x, y
    else:
        x_for_velocity, y_for_velocity = fill_nearest(x), fill_nearest(y)
    ...
        x_velocity = np.gradient(x_for_velocity)
        y_velocity = np.gradient(y_for_velocity)
        ...
        speed = np.hypot(x_velocity, y_velocity)
    speed[~visible] = np.nan
    # A derivative adjacent to a visibility gap may itself be undefined even
    # when the position at the center sample is finite.
    visible = visible & np.isfinite(speed)
```

iii. Step 1: "Align DLC x/y trajectories to the neural time axis via interpolation; derive x/y frame-gradient velocities; retain tongue invisibility masks and fill other-feature gaps", and "Kinematic velocity in reference code is the per-frame gradient in pixels/sample, not explicitly rescaled to pixels/s." Key Decision 7: "Velocity means speed magnitude. Median-thresholding a signed component would classify direction rather than velocity magnitude." Key Decision 8: "Visibility precedes filling … tongue remains unfilled as in the paper."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session: the median of the tongue speed pooled over all retained trials and all *visible* bins (class-2 bins are excluded from the percentile). Bins at or above it get class 1, below it class 0, and untracked bins class 2 (`not_visible`). The converter raises if the threshold is not finite. Resulting distribution across the 12 sessions is 2.8% / 2.8% / 94.5%.

ii.
```python
    tongue_threshold = float(np.nanmedian(tongue[tongue_visible]))
    ...
    if not np.all(np.isfinite(thresholds)):
        raise ValueError(f"Could not compute finite session thresholds: {thresholds}")
```
```python
            valid = visibility[output_index - 3, trial]
            classes = np.full(N_TIME, 2, dtype=np.int8)
            classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
```

iii. Key Decision 9: "Session percentile is computed over retained trials and visible/valid aligned timepoints: This defines each converted session internally, yields ~50/50 classes 0/1 among visible samples, and excludes class-2 samples from the percentile." This implements the Decoder Task's "discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible". Step 10 verified "All class-0/1 median splits are within 0.006 of 50/50 per session."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a per-session offset is computed from the bitcode pulse recorded by both systems — `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` — exactly as `findVideoOffset.m` does. Each trial's frame times become `frameTimes − offset − goCue[trial]`, and the x/y traces are then interpolated onto the shared 1000-bin neural axis, so tongue bin k and neural bin k are the same interval.

ii.
```python
def video_offset_seconds(handle: h5py.File, bp: dict[str, np.ndarray]) -> float:
    neural_bit_start = direct_field(handle, "obj/sglx/bitcode/bitstart", np.float64)
    fs = float(np.asarray(handle["obj/sglx/fs"]).squeeze())
    return mode_value(neural_bit_start) / fs - mode_value(bp["bitStart"])
```
```python
        aligned_side_time = side_frames - offset - bp["goCue"][raw_trial]
```

iii. Step 4: "Computed offset is 0.49004 s in older sessions and 0.9900159 s for JEB19; fixed 0.5 s would misalign JEB19 → Use per-session bitcode-derived offset, never a universal 0.5-s shift." Key Decision 10 repeats this. Step 7: "Go cue is at zero in neural, position, velocity, and motion panels; aligned traces have no discontinuity at zero", and Step 10 check 3 independently reconstructed the full 1000-bin tongue trace from raw HDF5 reads and matched with `np.allclose`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera (`obj.traj{2}`), feature `top_paw` only, using that camera's own `frameTimes`. `bottom_paw` is not used. Same bitcode/go-cue variables as the tongue.

ii.
```python
        paw[kept_trial], paw_visible[kept_trial], paw_x, paw_y = feature_speed(
            handle, bottom, int(raw_trial), "top_paw", aligned_bottom_time, False
        )
```
```python
        _, bottom_frames = trajectory_for_trial(handle, bottom, int(raw_trial))
        aligned_bottom_time = bottom_frames - offset - bp["goCue"][raw_trial]
```

iii. Step 5 mapping table: "`top_paw` matches paper figure code's paw feature." The bottom camera is the view in which the paws are tracked in `params.traj_features{2}`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same interpolation-onto-`TIME` as the tongue, but following the reference's non-tongue branch: the visibility mask is recorded first, then x and y are nearest-filled across gaps (`fill_nearest`), the gradient is taken, and a per-trial "baseline derivative" — the median of the frame-to-frame differences — is subtracted from both components, **including the reference's use of the x baseline for the y component**. Speed is the magnitude of the two corrected derivatives in pixels per bin, and is then re-masked to NaN wherever the paw was not actually visible, so the nearest-filled samples never reach the output.

ii.
```python
        if not is_tongue:
            stacked = np.column_stack((x_for_velocity, y_for_velocity))
            differences = np.diff(stacked, axis=0)
            baseline_derivative = np.array([
                np.median(column[np.isfinite(column)]) if np.any(np.isfinite(column)) else 0.0
                for column in differences.T
            ])
            # Match the released function, including its use of x baseline for y.
            x_velocity = x_velocity - baseline_derivative[0]
            y_velocity = y_velocity - baseline_derivative[0]
        speed = np.hypot(x_velocity, y_velocity)
    speed[~visible] = np.nan
```

iii. Step 4: "Reference nearest-fills non-tongue positions and retains tongue missingness separately → Save visibility masks before any filling. Use class 2 at missing tongue/paw timepoints as required; fill only to calculate visible neighboring velocities consistently." The in-code comment records the deliberate decision to reproduce `findVelocity.m`'s `xvel - basederiv(1); yvel - basederiv(1)` rather than "fix" it.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: the session median over retained trials and visible bins, `>=` → 1, `<` → 0, untracked bins → 2. No cross-camera normalisation is needed because only one view is used. Resulting distribution is 43.5% / 43.5% / 12.9%.

ii.
```python
    paw_threshold = float(np.nanmedian(paw[paw_visible]))
    ...
            classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
```
```python
            ["below_50th_percentile", "at_or_above_50th_percentile", "not_visible"],
```

iii. Same as 7-c — Key Decision 9 and the Decoder Task's per-session 50th-percentile rule. Step 7: "Paw visibility is 95.4–96.0%" in the two sample sessions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same per-session bitcode offset, but using the **bottom** camera's own `frameTimes` (the AI computes an aligned time vector per camera rather than reusing the side camera's), then the same interpolation onto the shared 1000-bin axis.

ii.
```python
        side_trajectory, side_frames = trajectory_for_trial(handle, side, int(raw_trial))
        del side_trajectory  # Loaded again feature-wise; retained here for timestamps.
        aligned_side_time = side_frames - offset - bp["goCue"][raw_trial]
        _, bottom_frames = trajectory_for_trial(handle, bottom, int(raw_trial))
        aligned_bottom_time = bottom_frames - offset - bp["goCue"][raw_trial]
```

iii. Step 5: "Same alignment" as the tongue; the two cameras are read separately because frame counts can differ between views. Step 10 check 3 audited the paw trace against raw HDF5 reads with `np.allclose`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each data structure, field `me.data` (unwrapped one further level if it is itself a struct), which holds one variable-length trace per trial with one value per camera frame. `obj.me` is not used. The side camera's `frameTimes` provide the time base.

ii.
```python
def load_motion_energy(path: Path) -> np.ndarray:
    motion = loadmat(path, simplify_cells=True)["me"]
    data = motion["data"]
    if isinstance(data, dict):
        data = data["data"]
    return np.atleast_1d(data)
```

iii. Step 2: "Paired motion-energy files contain `me.data`, a length-`Ntrials` object array of variable-length float64 traces, and an author threshold `moveThresh`. All 8,260 static-context trials have nonempty finite traces." The nested unwrap mirrors `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end`. Step 3: "Motion energy is the per-frame 99th percentile of pixelwise absolute differences between five-frame future and past medians", i.e. the spatial reduction is already done upstream.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond alignment and resampling. The trace is linearly interpolated from the aligned side-camera frame times onto the 1000-bin axis, then residual NaNs (bins outside the frame coverage) are filled with the nearest valid value, exactly as `loadMotionEnergy.m` does with `fillmissing(...,'nearest')`. The authors' stored `moveThresh` is deliberately ignored in favour of the specified median split.

ii.
```python
            interpolated = interpolate_with_nans(motion_times, trial_motion, TIME)
            motion[kept_trial] = fill_nearest(interpolated)
```
```python
def fill_nearest(values: np.ndarray) -> np.ndarray:
    ...
    nearest = np.where(indices - left <= right - indices, left, right)
    out = values.copy()
    out[~finite] = values[nearest[~finite]]
    return out
```

iii. Step 4: "Paper/code manual bimodal threshold … Decoder specification overrides this: use the median of aligned valid motion energy per session." Step 1 records that `loadMotionEnergy` "aligns/interpolates to neural time, nearest-fills edge NaNs, and thresholds movement", which is what the AI reproduces.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session median over all valid aligned samples; `>=` → 1, `<` → 0. Because the nearest-fill leaves no NaN, the third class (`no_video`) is reserved in `output_values` but never emitted — the observed distribution is 50.0% / 50.0% / 0%.

ii.
```python
    valid_motion = np.isfinite(motion)
    motion_threshold = float(np.nanmedian(motion[valid_motion]))
```
```python
            ["below_50th_percentile", "at_or_above_50th_percentile", "no_video"],
```

iii. Step 12: "Motion at 1.499× empirical chance: No data bug. All motion outputs derive exactly from raw aligned traces, classes are 49.97/50.03% … Inventing absent-video samples or altering thresholds to game this metric would violate the source and task." The AI's reading is that class 2 means the video stream is absent, and every selected session has video.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same per-session bitcode offset and same grid as the tracking, using the side camera's frame times (motion energy has one value per side-camera frame). If the trace length does not match the frame count, or the frame times are absent/all-NaN, the converter falls back to `loadMotionEnergy.m`'s catch branch: a nominal 400 Hz clock with the code's explicit 0.5 s shift.

ii.
```python
            if len(trial_motion) == len(side_frames) and np.sum(np.isfinite(side_frames)) >= 2:
                motion_times = aligned_side_time
            else:
                # Same catch-path as loadMotionEnergy.m when frameTimes are
                # absent/all-NaN: nominal 400 Hz and its explicit 0.5-s shift.
                motion_times = np.arange(1, len(trial_motion) + 1, dtype=np.float64) / 400
                motion_times = motion_times - 0.5 - bp["goCue"][raw_trial]
```

iii. Step 10 issue log: "JEB19 2023-04-19 retained trial 208/raw trial 222 has a finite 3,087-sample raw motion trace but all-NaN camera timestamps. The first converter version assigned 1,000 class-2 bins. Reference `loadMotionEnergy.m` instead catches this case and uses a nominal 400-Hz clock with an explicit 0.5-s shift. Implemented that fallback, regenerated the full pickle, reran full verification, and reran every audit."

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Four categories. (a) **Structural inconsistencies fail loudly**: mismatched `Ntrials`, non-exclusive hit/miss/no, non-exclusive R/L, non-finite go cues, mismatched spike trial/time vector lengths, mismatched DLC frame/frameTimes lengths, mismatched motion-energy trial counts, an unexpected `ts` rank, a missing camera view, a requested probe that does not exist, a session with <2 trials or <10 units, and non-finite session thresholds all raise. (b) **Absent tracking of a feature** in a trial returns an all-NaN trace and therefore 1000 `not visible` bins. (c) **Untracked frames** (DLC likelihood low ⇒ x/y already NaN) propagate NaN through interpolation into the `not visible` class, and derivatives adjacent to a gap are additionally invalidated. (d) **All-NaN camera timestamps** for motion energy fall back to the reference's nominal 400 Hz clock; bins outside the video's coverage are nearest-filled for motion energy and for paw *position* (but the paw's visibility mask is taken before filling, so filled paw samples are re-masked out). Nothing is interpolated across a tongue gap, and no trial is dropped for a video problem.

ii.
```python
    if feature not in names:
        nan = np.full(N_TIME, np.nan, dtype=np.float64)
        return nan, np.zeros(N_TIME, dtype=bool), nan, nan
```
```python
def interpolate_with_nans(times, values, target):
    finite_time = np.isfinite(times)
    if np.sum(finite_time) < 2:
        return np.full(len(target), np.nan, dtype=np.float64)
    ...
    return np.interp(target, x, y, left=np.nan, right=np.nan)
```
```python
            if not (np.all(np.isfinite(n)) and np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                raise AssertionError(f"Session {session} trial {trial}: non-finite values")
```

iii. Key Decision 8: "Visibility precedes filling: Use finite raw/interpolated x/y to assign class 2, then fill paw positions only for computing speed. This is required to avoid erasing `not visible`." Step 10's issue log documents the one real data defect found (JEB19 2023-04-19 trial 222's all-NaN frame times) and its reference-derived repair. Step 5: "No NaN/Inf is permitted in target arrays", enforced by `validate_converted`.

## 11-a. What are the most time-consuming steps of the code?

i. The AI states that reading nested MATLAB object references dominates, and that the DLC per-trial reads are the cost centre; full conversion is 42.8 s for 12 sessions (~3.6 s/session), so no further optimisation was pursued. Profiling one session confirms this: `align_kinematics_and_motion` is 4.20 s of the 4.48 s total (94%), of which `trajectory_for_trial` (HDF5 `ts`/`frameTimes` reads) is 2.95 s and `feature_names_for_trial` (decoding the `featNames` cell array) is 0.90 s; `build_neural` is only 0.14 s and `low_fr_filter` is negligible.

ii.
```python
def trajectory_for_trial(handle: h5py.File, group: h5py.Group, trial: int):
    ts_ref = matlab_cell_refs(group["ts"])[trial]
    ft_ref = matlab_cell_refs(group["frameTimes"])[trial]
    stored = deref_array(handle, ts_ref)
```
```python
    print(
        f"[{spec.session_id}] {raw_trial_count}->{len(raw_trials)} trials, "
        f"{len(quality_clusters)}->{len(clusters)} units, {elapsed:.2f} s",
        flush=True,
    )
```

iii. Step 6: "Nested MATLAB references require per-cluster/per-trial reads; naive full `mat73` loading was stopped because it materialized the complete nested file and was slow/memory-heavy." Step 7: "Conversion compute 3.87 s/session → 46.4 s for 12 sessions … well below the 15-minute optimization threshold."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's position is that the remaining trial loops are irreducible because DLC traces are ragged: "DLC arrays are variable-length and therefore require a trial loop, but all transformations within a trace are vectorized." It did vectorise the spike binning (`np.add.at` over all of a unit's spikes at once) and the smoothing (a single `lfilter` over the whole `(units, trials, time)` tensor). Loops that remain and *are* vectorisable, but that the AI does not identify: the per-trial loops in `static_trial_outputs` and `discretize_outputs` (pure array indexing over rectangular `(n_trials, 1000)` data), the per-unit loop in `build_neural`, and the unit × condition double loop in `low_fr_filter` (all seven condition PSTHs of a unit could be computed with one `histogram2d`, as the expert's spike counting does).

ii. Vectorised:
```python
        np.add.at(counts[unit], (mapped_trials[in_window], bins[in_window]), 1)
    return reference_smooth(counts / DT)
```
Not vectorised:
```python
    for trial in range(len(static)):
        out = np.empty((6, N_TIME), dtype=np.int8)
        out[:3] = static[trial, :, None]
        for output_index, values in enumerate(continuous, start=3):
            ...
```

iii. Step 6, as quoted above. Because total runtime is 43 s the AI judged further vectorisation unnecessary, which is defensible, but the claim that the remaining loops *require* trial iteration is only true of the DLC loop.

## 11-c. What processing does the code repeat multiple times?

i. The AI's notes claim redundancy was eliminated ("Field-level HDF5 reads … avoid unnecessary file I/O"; the video offset is computed once per session; the smoothing kernel is built once at module level; the neural tensor is allocated once per session). In fact the code does repeat work per trial: `trajectory_for_trial` is called **four times per trial** (1,208 calls for 302 trials) — once for the side camera purely to obtain `frameTimes` (its `ts` array is read and immediately `del`eted), once for the bottom camera for the same reason, and once again inside each `feature_speed` call — so every trial's full `ts` array is read twice per camera. In addition, `feature_names_for_trial` re-reads and re-decodes the camera's `featNames` cell array on every trial although it is constant within a session (0.9 s/session). The motion-energy file is read once per session, and per-cluster spike vectors are read once.

ii. The redundant reads:
```python
        side_trajectory, side_frames = trajectory_for_trial(handle, side, int(raw_trial))
        del side_trajectory  # Loaded again feature-wise; retained here for timestamps.
        ...
        _, bottom_frames = trajectory_for_trial(handle, bottom, int(raw_trial))
```
```python
def feature_speed(handle, group, trial, feature, aligned_time, is_tongue):
    trajectory, frame_times = trajectory_for_trial(handle, group, trial)
    names = feature_names_for_trial(handle, group, trial)
```
Computed once, correctly:
```python
        offset = video_offset_seconds(handle, bp)
```

iii. Step 6 claims "Code speedups added: Field-level HDF5 reads, vectorized `np.add.at` spike binning, one session-sized float32 neural tensor, and `scipy.signal.lfilter` for the short causal kernel", and the `del side_trajectory` comment shows the duplicate read was noticed but accepted rather than removed. No justification for the repetition is given beyond overall runtime being acceptable.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, none of which the notes identify as waste. (a) Each `trajectory_for_trial` call decodes the full `(frames, 3, 10)` array for **all** tracked features (tongue, jaw, nose, lickport, nostrils, both paws, four tongue landmarks) when exactly one feature is used, and does so four times per trial — the dominant cost of the whole conversion. (b) The side-camera trajectory is read and immediately discarded when only `frameTimes` is wanted. (c) The likelihood column of `ts` is transposed along with x/y but never used. (d) `low_fr_filter` builds and causally smooths seven condition PSTHs per cluster only to take their grand mean, and the smoothing barely changes that mean; the full `mean_rates` vector for the rejected clusters is kept only for a diagnostic histogram. (e) `fill_nearest` is applied to the paw x/y positions, but the samples it invents are re-masked to NaN immediately afterwards, so only the gradients immediately adjacent to gaps survive. (f) The `diagnostics` dictionary (sample neural matrices, kinematic samples, static labels) is assembled for all 12 sessions but only the first two are ever plotted.

ii.
```python
    stored = deref_array(handle, ts_ref)
    trajectory = np.transpose(stored, (2, 1, 0)).astype(np.float64, copy=False)
```
```python
        mean_rates[unit] = np.nanmean(np.stack(condition_psths))
    keep = mean_rates > LOW_FR_HZ
```
```python
        diagnostics = {
            "session_id": spec.session_id,
            "mean_rates_all_quality_units": mean_rates,
            "neural_sample": neural[: min(12, len(clusters)), 0].copy(),
            ...
        }
```

iii. The notes justify the low-FR PSTH reconstruction as fidelity to `removeLowFRClusters` ("Rate filtering reproduces figure code, not generic defaults"), and the diagnostics as material for the `--show-processing` plots and the unit-curation histogram. The redundant full-trajectory decoding is not discussed; the only efficiency claim is that the total runtime (43 s) is far under the 15-minute budget, so no further trimming was attempted.
