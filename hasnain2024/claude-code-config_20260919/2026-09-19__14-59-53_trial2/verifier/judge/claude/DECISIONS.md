# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. One session = one MATLAB `data_structure_<anm>_<date>.mat` file, living in one of two
folders (`/app/data/Ephys_Behavior` for the 25 fixed-delay sessions,
`/app/data/RandomizedDelay_Ephys_Behavior` for the 19 randomized-delay sessions), with its
motion energy in a sibling `motionEnergy_<anm>_<date>.mat`. The AI does **not** glob the
folders: it hard-codes a 44-entry table `SESSIONS = (anm, date, ALM probe number(s), task)`
transcribed from the authors' own `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`
scripts, so the three `data_structure_*.mat` files present on disk but absent from the load
scripts are excluded. Each file is read once by `load_obj` using `pymatreader.read_mat`,
which transparently handles both MAT v7 and v7.3. The two behaviour-only optogenetics folders
(`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) are not touched
because they contain no spikes. Sessions are processed independently and in parallel with a
`ProcessPoolExecutor` (16 workers), then concatenated.

ii.
```python
FIXED_DELAY = [
    ("JEB6", "2021-04-18", [2]),
    ...
    ("JEB15", "2022-07-29", [2]),
]
RANDOMIZED_DELAY = (
    [("JEB11", "2022-05-10", [1]), ...]
    + [("JEB23", d, [1]) for d in (...)]
    + [("JEB24", d, [1]) for d in (...)]
)
SESSIONS = ([(a, d, p, "fixed") for a, d, p in FIXED_DELAY]
            + [(a, d, p, "randomized") for a, d, p in RANDOMIZED_DELAY])

DATA_DIR = {"fixed": "/app/data/Ephys_Behavior",
            "randomized": "/app/data/RandomizedDelay_Ephys_Behavior"}


def data_path(anm, date, task):
    return os.path.join(DATA_DIR[task], f"data_structure_{anm}_{date}.mat")


def load_obj(anm, date, task):
    """loadObjs.m -- read one session's data object (handles MAT v7 and v7.3)."""
    from pymatreader import read_mat
    return read_mat(data_path(anm, date, task))["obj"]
```
```python
with ProcessPoolExecutor(max_workers=args.workers) as ex:
    for r in ex.map(process_session, sessions):
        results.append(r)
```

iii. From CONVERSION_NOTES.md Step 5 / Step 10 Check 3: the `load<ANM>_ALMVideo.m` scripts are
"the definitive record of which sessions and which probe entered their analysis"; using them
"reproduces the paper's n = 19 randomized-delay sessions" (the folder holds 22 files) and the
paper's n = 25 fixed-delay sessions. The behaviour-only optogenetic directories "have no spikes
and cannot be used by a neural decoder". Both fixed- and randomized-delay sessions are kept in
one dataset because "the harness trains a per-session projection, so mixing them is harmless
and keeps all available ALM recordings with video".

## 1-b. How are the data split into subjects?

i. The subject is the animal id carried in the session table (`anm`, identical to the part of
the filename before the underscore). `subjects` is the sorted set of unique animals over the
kept sessions, and `subject_idx` is each session's index into that list. This gives 14 mice
across the 44 sessions (10 fixed-delay, 4 randomized-delay). The animal id in the table is used
rather than a field inside the file because `obj.meta/ex.anm` is not present in every session.

ii.
```python
subjects = sorted({r["info"]["anm"] for _, r in kept})
data = {
    ...
    "subjects": subjects,
    "subject_idx": np.array([subjects.index(r["info"]["anm"]) for _, r in kept],
                            dtype=np.int64),
```

iii. Step 4 of CONVERSION_NOTES.md records that the load scripts give 10 distinct animals over
the 25 fixed-delay sessions while the paper says "nine mice"; the AI investigated
(`NullPotent/SessionMeta.csv` lists only 24 of the 25 sessions and 9 mice), concluded the paper
under-counts by one, kept the load-script list because the *session* count matches exactly, and
documented the discrepancy rather than "fixing" it.

## 1-c. How are the data split into sessions?

i. One entry of `SESSIONS` = one session = one file = one element of `neural`, `input`,
`output`, `subject_idx` and `brain_region_idx`. The task folder is carried in the table entry
(`"fixed"` / `"randomized"`), so `data_path`/`me_path` resolve the right directory and the two
task variants are treated uniformly. Each session is converted by `process_session`, which
returns lists of per-trial arrays plus an `info` dict; the per-session results are then filtered
(≥10 units, ≥2 trials) and assembled in the fixed `SESSIONS` order. All 44 survive.

ii.
```python
def process_session(args):
    anm, date, probes, task = args
    obj = load_obj(anm, date, task)
    ...
    return dict(neural=neural, input=inputs, output=outputs, info=info)
```
```python
kept, dropped = [], []
for s, r in zip(sessions, results):
    i = r["info"]
    if i["n_units"] < MIN_UNITS:
        dropped.append((i["session"], f"only {i['n_units']} units (< {MIN_UNITS})"))
    elif i["ntrials_kept"] < 2:
        dropped.append((i["session"], f"only {i['ntrials_kept']} trials"))
    else:
        kept.append((s, r))
```

iii. Session-level curation is justified by the Methods: "Recording sessions were included for
analysis only if they had at least 10 units". The ≥2-trial rule is the harness requirement. In
practice neither rule drops anything (minimum observed is 17 units and 193 trials), so the
session set is exactly the reference load-script list.

## 1-d. How are the data split into trials?

i. A trial is one row of the Bpod table `obj.bp`: every per-trial field is read and truncated to
`bp.Ntrials` (`vec(...)[:ntrials_all]`), because a few `bp` fields are stored longer than the
trial count. One go cue per trial comes from `bp.ev.goCue`. Spikes carry their own 1-based trial
number in `clu.trial` and camera frames are already stored per trial as cell arrays
(`traj{view}.ts{trial}`, `.frameTimes{trial}`, and one motion-energy trace per trial), so no
trial boundary has to be reconstructed. `as_cell_list` re-wraps single-element MATLAB cell
arrays that `pymatreader` unwraps.

ii.
```python
n = int(vec(bp["Ntrials"])[0])
early = vec(bp["early"])[:n] > 0
stim = vec(bp["stim"]["enable"])[:n] > 0
```
```python
gocue = vec(bp["ev"][ALIGN_EVENT])[:ntrials_all]
R = vec(bp["R"])[:ntrials_all] > 0
...
tr = np.ravel(np.asarray(trialno[i], dtype=np.float64)).astype(np.int64) - 1
ok = (tr >= 0) & (tr < ntrials) & np.isfinite(tm)
```
```python
def as_cell_list(x, n):
    """Normalise a MATLAB cell-array field to a python list with n entries."""
    if isinstance(x, list):
        return x
    return [x] * n if n == 1 else [x] + [None] * (n - 1)
```

iii. Step 4 / Step 10 Check 5: the AI verified that `hit + miss + no == Ntrials` in all 44
sessions, that `R` xor `L` holds on every trial, that there are no NaNs in
`R/L/hit/miss/no/early/autowater/goCue`, and that `clu.trial` is 1-based (`max == Ntrials`), so
the Bpod table defines trials directly and no inference is needed.

## 1-e. How are trials filtered based on quality controls?

i. Three filters, all applied before anything is computed:
(1) early-lick trials (`bp.early`) are dropped — the Methods say early-lick trials "were omitted
from analyses"; (2) photoinactivation trials (`bp.stim.enable`) are dropped — the reference's
`params.condition` strings all begin `~stim.enable&~early`; (3) trials on which **no** cluster on
the ALM probe(s) fired a single spike are dropped, because in two JEB24 sessions the SpikeGLX
recording stops before the behavioural session does and those trials would enter the dataset as
all-zero neural matrices. Detection for (3) deliberately uses *every* cluster on the probe,
including the ones the quality filter rejects. Hit/miss/ignore and DR/WC trials are all kept
because they are decoder classes. Result: 13,762 of 14,972 trials kept (976 early, 187 stim,
64 no-ephys).

ii.
```python
def select_trials(bp, obj=None, probes=()):
    n = int(vec(bp["Ntrials"])[0])
    early = vec(bp["early"])[:n] > 0
    stim = vec(bp["stim"]["enable"])[:n] > 0
    keep = ~(early | stim)
    if obj is not None:
        keep &= trials_with_ephys(obj, probes, n)
    return np.flatnonzero(keep), n
```
```python
def trials_with_ephys(obj, probes, ntrials):
    has = np.zeros(ntrials, dtype=bool)
    for p in probes:
        c = clu[p - 1]
        trialno = c["trial"] if isinstance(c["trial"], list) else [c["trial"]]
        for t in trialno:
            a = np.ravel(np.asarray(t, dtype=np.float64))
            a = a[np.isfinite(a)].astype(np.int64) - 1
            a = a[(a >= 0) & (a < ntrials)]
            has[a] = True
    return has
```

iii. CONVERSION_NOTES.md Step 9 / Step 10 Check 1: the first full run produced 61
"all neural data is zero" warnings concentrated in JEB24 2023-10-23 and 2023-11-03; the AI
traced this to the ephys recording ending early (last trial with any sorted spike = 314/343 and
312/346) and added `trials_with_ephys()`. `obj.trials.bp.haveEphys` was deliberately not used
"because its length disagrees with `bp.Ntrials` in the one session where it is informative —
319 entries for 318 trials". Step 10 Check 3 lists this as a deliberate difference from the
reference: "An all-zero neural matrix is not data and the harness flags it; dropping is required
for a decoder."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}`, the spike-sorted clusters of the designated ALM probe(s) only (probe index
from the load-script table; two-probe sessions concatenate both). Per cluster the fields used are
`trialtm` (spike time relative to that trial's start, on the behaviour clock), `trial` (1-based
trial number of each spike) and `quality` (the manual curation label, used only for filtering).
`obj.bp.ev.goCue` is the second input, since it defines the alignment.

ii.
```python
def bin_and_smooth(obj, cluster_ids, align_times, ntrials):
    """getSeq.m -- (ntrials, nclusters, NT) smoothed firing rates in spikes/s."""
    ...
    for ci, (p, i) in enumerate(cluster_ids):
        c = clu[p]
        trialtm = c["trialtm"] if isinstance(c["trialtm"], list) else [c["trialtm"]]
        trialno = c["trial"] if isinstance(c["trial"], list) else [c["trial"]]
        tm = np.ravel(np.asarray(trialtm[i], dtype=np.float64))
        tr = np.ravel(np.asarray(trialno[i], dtype=np.float64)).astype(np.int64) - 1
```

iii. Step 10 Check 3 (a)/(d): "session list and ALM probe from `load<ANM>_ALMVideo.m`";
alignment and binning are `alignSpikes.m` + `getSeq.m`, which build `obj.trialdat` from exactly
these fields. Step 4 records that the AI checked `obj.ex.probe.loc` and found it internally
inconsistent for JEB15 (probe 2 labelled `L M1TJ`, and a `DUMMY` probe designated for
2022-07-29), so it follows the load scripts for probe choice, as the paper does.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, counted into 10 ms bins over [-2.5, 2.5] s
(`histc` semantics, last bin dropped), divided by the bin width to give spikes/s, and smoothed
along time with the reference's **causal** Gaussian kernel: `gausswin(15)` (alpha = 2.5) with the
first `floor(15/2) = 7` taps zeroed and renormalised, applied with `'reflect'` boundary handling.
This reproduces `obj.trialdat` exactly. No normalisation, baseline subtraction, or z-scoring is
applied; values are stored as `float32` firing rates in Hz. Units from both probes of a two-probe
session are concatenated into one population. Smoothing is applied once per session to a
`(515, nclusters·ntrials)` matrix as an 8-tap causal FIR (the first 7 taps are zero), so no FFT
and no round-off.

ii.
```python
def causal_gaussian(n=SMOOTH_N, alpha=2.5):
    """mySmooth.m kernel: gausswin(n) with the first floor(n/2) taps zeroed, normalised."""
    k = np.arange(n)
    w = np.exp(-0.5 * (alpha * (k - (n - 1) / 2.0) / ((n - 1) / 2.0)) ** 2)
    w[: n // 2] = 0.0
    return w / w.sum()


def my_smooth(x, kern=KERNEL, bctype=BCTYPE):
    n = kern.size
    half = (n - 1) // 2
    if bctype == "reflect":
        xp = np.concatenate([x[:n], x], axis=0)
        trim = n
    ...
    for k in range(half, n):
        w = kern[k]
        j = k - half                      # delay in samples (>= 0)
        if j == 0:
            out += w * xp
        else:
            out[j:] += w * xp[:-j]
    return out[trim:]
```
```python
    counts[ci] = np.bincount(flat, minlength=ntrials * NT).reshape(ntrials, NT)
rate = counts / DT                               # spikes/s
m = rate.reshape(-1, NT).T                       # (NT, nclu*ntrials)
m = my_smooth(m)
return m.T.reshape(len(cluster_ids), ntrials, NT)
```

iii. Step 5 Key Decision 4: "Neural representation: smoothed firing rate in spikes/s
(`obj.trialdat`), i.e. the same quantity the paper feeds to its own choice/context decoders,
including the causal 150 ms Gaussian kernel with `'reflect'` boundary handling." Step 6 records a
unit test (`/app/cache/test_primitives.py`) showing `my_smooth` reproduces a literal
transcription of `mySmooth.m` (`np.convolve(concat(x[:15], x), kern, 'same')[15:]`) to 2.2e-16,
that an impulse at bin 100 produces no output before bin 100 (causality) with its peak exactly at
bin 100, and that the kernel equals `[0]*7 + [0.25098, 0.23547, 0.19447, 0.14137, 0.09046,
0.05096, 0.02527, 0.01103]`. Step 8 additionally reports a control experiment showing that
rescaling the firing rates does not change decoder accuracy.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First `findClusters(quality, {'all'})`: the manual curation label is trimmed and
compared **case-sensitively** against the drop list `{'garbage', 'gabrga', 'noisy', 'real?'}` —
everything else is kept, including multi-units, `Poor`-labelled units, and the 7 clusters whose
quality field is a non-string (treated as `''`, i.e. kept). Second, after binning, any unit whose
mean smoothed rate over the analysis window and the kept trials is ≤ 1 Hz is dropped
(`removeLowFRClusters.m` with `params.lowFR = 1`). Across the dataset this leaves **2,457** units
of 2,514 quality-passing clusters, 17–141 per session (mean 55.8).

ii.
```python
QUALITY_REJECT = ("garbage", "gabrga", "noisy", "real?")  # findClusters(..., 'all')

def select_clusters(obj, probes):
    keep = []
    for p in probes:
        c = clu[p - 1]
        if not isinstance(c, dict) or "quality" not in c:
            continue
        quals = c["quality"] if isinstance(c["quality"], list) else [c["quality"]]
        for i, q in enumerate(quals):
            q = str(q).strip() if isinstance(q, (str, np.str_)) else ""
            if q not in QUALITY_REJECT:
                keep.append((p - 1, i))
    return keep
```
```python
mean_fr = rates.reshape(n_quality, -1).mean(axis=1)           # removeLowFRClusters.m
keep_unit = mean_fr > LOW_FR
rates = rates[keep_unit]
```

iii. Step 10 Check 3 (b): "identical string set and case sensitivity" to `findClusters.m`; Step
10 Check 5 records the deliberate consequences — "JEB6 has one `Noisy` (capital N) → kept,
reproducing MATLAB `ismember(...,'noisy')`" and "clusters with a non-string quality (`[]`), 7
across the dataset, treated as `''`, which is not in the reject list, i.e. kept — exactly what
`findClusters.m` does". The 1 Hz cut is the paper's ("All units with firing rates exceeding 1 Hz
were included in all other analyses"). Step 4 notes that `getDefaultParams.m` uses `lowFR = 0.5`
but every shipped figure script uses `lowFR = 1`, and the figure-script value is used.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A single subtraction. `clu.trialtm` is already on the behaviour clock and already relative to
its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm − goCue[trial]` puts
every spike in seconds from the go cue with no interpolation or offset. This is applied per spike
using the spike's own trial index. `params.advance_movement = 0`, so there is no neural/video lag.
The camera streams need a separate clock correction (7-d); the spikes do not.

ii.
```python
aligned = tm - align_times[tr]              # alignSpikes.m
b = bin_index(aligned)
ok = b >= 0
flat = tr[ok] * NT + b[ok]
```
```python
gocue = vec(bp["ev"][ALIGN_EVENT])[:ntrials_all]   # ALIGN_EVENT = "goCue"
```

iii. Step 10 Check 3 (c): "`alignSpikes.m`: `trialtm − bp.ev.goCue(trial)` … identical formulas".
Step 5 Key Decision 2: "Alignment: go cue (`obj.bp.ev.goCue`), as required by the task and as used
by every reference analysis (`params.alignEvent = 'goCue'`). In WC trials this field holds the
water-drop time, which is the matched event for that context." Step 4 cross-checks the event
times against the Methods (`ev.sample ≈ 0.3 s`, `ev.delay ≈ 1.6 s`, `ev.goCue ≈ 2.5 s` for the
fixed-delay task, i.e. 1.3 s sample tone + 0.9 s delay). Step 12 verifies the alignment
empirically: the population PSTH peaks at t = 0 and shows the sample-tone response at exactly
−2.2 s, and time-resolved decoding of lick direction is at chance before the sample tone and peaks
just after the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins** (`DT = 1/100`) over a **[-2.5, +2.5] s** window around the go cue → **500 bins
per trial**, identical for every trial and session. The 501 edges are `-2.5:0.01:2.5` and the
stored time axis is the bin centres `edges[:-1] + dt/2` (−2.495 … +2.495 s), matching `obj.time`.
Spikes are counted directly into this grid (`histc` semantics, last bin dropped), so there is no
rebinning of the neural data. The video streams are *resampled* onto the same grid by linear
interpolation of position / motion energy (not by averaging frames), which is what
`findPosition.m` does; the 400 Hz video is therefore decimated to 100 Hz. `bin_index` reproduces
`histc` edge semantics exactly.

ii.
```python
TMIN, TMAX, DT = -2.5, 2.5, 1.0 / 100.0   # params.tmin / params.tmax / params.dt
EDGES = np.round(np.arange(TMIN, TMAX + DT / 2.0, DT), 10)   # 501 bin edges
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)            # 500 bin centres
NT = TIME.size


def bin_index(t):
    """histc(t, EDGES) bin index; -1 (or >= NT) when outside the analysis window."""
    idx = np.searchsorted(EDGES, t, side="right") - 1
    idx[(idx < 0) | (idx >= NT)] = -1
    return idx
```

iii. Step 4 Discrepancies: "`getDefaultParams.m`: `dt=1/200`, `lowFR=0.5` vs every figure script:
`dt=1/100`, `lowFR=1` … Use the figure-script values (`dt=1/100`, `lowFR=1`);
`getDefaultParams` is unused by the shipped analyses", and "`WorkingWithDataObjs.m` comment says
'use a 5 ms bin width' but the code sets `params.dt = 1/100` → comment is stale; use 10 ms".
Step 10 Check 5 unit-tests the edge convention: `t = -2.5 → bin 0`, `t = 0 → bin 250` (centre
+0.005 s, i.e. the go cue falls at the *start* of bin 250), `t = 2.4999 → bin 499`, `t = 2.5`
dropped (the `N(1:end-1)` step).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw variable at all — it is the analysis time base itself, i.e. the
centres of the 500 bins of the [-2.5, +2.5] s window around the go cue, which is `obj.time` in the
reference pipeline. It is the single decoder input, identical for every trial and every session,
and is stored once and shared (by reference) across all trials to save memory.

ii.
```python
INPUT_NAMES = ["time_from_go_cue"]
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)            # 500 bin centres
```
```python
input_trial = np.ascontiguousarray(TIME[None, :], dtype=np.float32)   # shared, read-only
for k, it in enumerate(trials):
    ...
    inputs.append(input_trial)
```

iii. Step 5 Variable Mapping: "`obj.time` (= bin centres) → `input[0]` `time_from_go_cue`,
continuous ramp −2.495 … +2.495 s, `float32` (1, 500) … only decoder input, per the task spec".
Step 10 Check 3 (e): "`obj.time` is the reference time base → `input[0] = obj.time` broadcast over
the trial … (the input set is dictated by the task spec)".

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The array is the bin-centre vector built once at module level from `TMIN`, `TMAX` and
`DT` and cast to `float32`. No per-trial computation, no interpolation, no rescaling.

ii.
```python
EDGES = np.round(np.arange(TMIN, TMAX + DT / 2.0, DT), 10)   # 501 bin edges
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)            # 500 bin centres
```

iii. N/A — the variable is defined by the decoder-task specification ("Time from go cue onset in
seconds (continuous, time-varying)"), and the window/binning choice is justified under 2-e.
The verification log confirms the range is identical ([-2.5, 2.5] to printed precision;
[-2.495, 2.495] exactly) for all 44 sessions.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spike times are expressed relative to their own trial's go cue
and counted into `EDGES`; the input is the centre of those same bins, so bin *k* of the input and
bin *k* of the neural matrix are the same interval by construction. The video outputs are
interpolated onto the same `TIME` axis (plus `params.advance_movement = 0`), so all four streams
share one time axis.

ii.
```python
aligned = tm - align_times[tr]              # alignSpikes.m
b = bin_index(aligned)                       # bins defined by EDGES
```
```python
TIME = (EDGES[:-1] + DT / 2.0).astype(np.float64)
taxis = TIME + ADVANCE_MOVEMENT              # ADVANCE_MOVEMENT = 0.0
```

iii. Step 5 Key Decision 13: "No neural/video lag (`params.advance_movement = 0`), per all
reference scripts." Step 10 Check 5 unit-tests the bin-centre convention
(`obj.time = edges + dt/2` then truncated; bin 250 centre = +0.005 s).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial Bpod flags: the instructed side `bp.R` and `bp.L`, and the outcome flags
`bp.hit`, `bp.miss` (with `bp.no` used to force the "none" class). The lick direction itself is
not recorded, so it is inferred from the instructed side combined with whether the animal was
correct.

ii.
```python
R = vec(bp["R"])[:ntrials_all] > 0
L = vec(bp["L"])[:ntrials_all] > 0
hit = vec(bp["hit"])[:ntrials_all] > 0
miss = vec(bp["miss"])[:ntrials_all] > 0
no = vec(bp["no"])[:ntrials_all] > 0
```

iii. Step 5 Variable Mapping and Step 10 Check 3 (f) both cite the reference's
`getPrevChoice.m`, whose formula is `choice = (R&hit)|(L&miss)` with NaN on `no` — the AI uses
"the same formula". Step 10 Check 5 records the verification that `R` xor `L` holds on every trial
and `hit + miss + no == Ntrials` in all 44 sessions, so the mapping is exhaustive and unambiguous.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A pure relabelling, broadcast over all 500 timepoints of the trial: the animal licked the
instructed port on a hit and the other port on a miss, and did not lick on an ignore trial.
Codes: left = 0, right = 1, none = 2. The array is initialised to 2 so that any trial that is
neither a hit nor a miss falls through to "none"; `lick[no] = 2` is then applied explicitly.
Stored as `int8`, constant across the trial's time axis.

ii.
```python
lick = np.full(ntrials_all, 2, dtype=np.int8)                     # 2 = none
lick[(L & hit) | (R & miss)] = 0                                  # left
lick[(R & hit) | (L & miss)] = 1                                  # right
lick[no] = 2                                                      # getPrevChoice: NaN on ignore
```
```python
out = np.empty((6, NT), dtype=np.int8)
out[0] = lick[it]
```

iii. Step 10 Check 3 (f): "identical definitions [to `getPrevChoice.m`], with the reference's
`NaN` on ignore trials replaced by the explicit third class the task spec requires
(`none` / `ignore`)". Result over the full dataset: left 0.423 / right 0.446 / none 0.131 — close
to balanced left/right, as expected for a task with equal numbers of left and right trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks trials on which water was delivered
without any auditory cue — i.e. the water-cued (WC) context. Everything else is the
delayed-response (DR) context.

ii.
```python
autowater = vec(bp["autowater"])[:ntrials_all] > 0
```

iii. Step 4 Discrepancies: the reference's `params.condition` strings use `autowater` to separate
WC from DR, and the authors' tutorial says "this field can be used as a proxy for obtaining
water-cued blocks and delayed-response blocks". The AI cross-checked against the Methods'
description ("A behavioral session began with approximately 100 DR trials and was then followed
by alternating blocks of WC and DR trials") and confirmed that `autowater` "forms contiguous
blocks starting ≈trial 100 in exactly 12 sessions" — matching the paper's 12 two-context sessions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling with the codes required by the task spec — autowater → WC (0), otherwise
DR (1) — broadcast over the trial's 500 timepoints as `int8`. The 32 sessions without WC blocks
are kept with a constant DR label rather than dropped.

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)               # 0 = WC, 1 = DR
...
out[1] = context[it]
```

iii. Step 5 Key Decision 1: "`context` is degenerate (all DR) in the 32 sessions without WC blocks
but is still a correct label there, and the harness computes **balanced** accuracy pooled over
sessions, so keeping them adds data without corrupting the context label. The 12 two-context
sessions supply the WC class." Step 10 Check 4 cross-checks the fraction: WC is 31.5% of kept
trials in the 12 two-context sessions, and "the Methods describe ≈100 DR trials followed by
alternating 10–25 trial blocks, which for a ~300-trial session predicts ≈33%".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial Bpod flags: `bp.hit`, `bp.miss` and `bp.no`. Unlike the lick direction, `bp.no`
is read explicitly rather than inferred, although the AI verified the three are mutually exclusive
and sum to `Ntrials` in every session.

ii.
```python
hit = vec(bp["hit"])[:ntrials_all] > 0
miss = vec(bp["miss"])[:ntrials_all] > 0
no = vec(bp["no"])[:ntrials_all] > 0
```

iii. Step 5 Variable Mapping cites `getOutcome.m` (`outcome = hit`, NaN on `no`). Step 4 records
the cross-check `hit + miss + no == Ntrials` in all 44 sessions, and Step 10 Check 5 lists
"`bp` field NaNs / ambiguity: none".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabelling into the three classes the task spec asks for, broadcast over the 500 timepoints
as `int8`: the array is initialised to 0 (incorrect), hits are set to 1 (correct) and `no` trials
to 2 (ignore); miss trials therefore keep 0. Ignore trials are kept as a class rather than
dropped, unlike the paper, which omits them from its analyses.

ii.
```python
outcome = np.full(ntrials_all, 0, dtype=np.int8)                  # 0 = incorrect
outcome[hit] = 1                                                  # 1 = correct
outcome[no] = 2                                                   # 2 = ignore
...
out[2] = outcome[it]
```

iii. Step 10 Check 3, deliberate difference 2: "`ignore` trials keep a label instead of `NaN`.
`getOutcome.m`/`getPrevChoice.m` set NaN so that MATLAB analyses skip them. The task spec requires
`none` and `ignore` as explicit classes, so they are coded 2." Step 10 Check 4 sanity-checks the
resulting rates against the Methods' training criterion of >70% accuracy: "DR hit rate excluding
ignore trials **0.857**, WC **0.915**". Full-dataset distribution: incorrect 0.120 / correct 0.749
/ ignore 0.131.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — using the feature named
`tongue`, which is `params.traj_features{1}{1}` in the reference. Per trial the fields used are
`ts` (shape `(n_frames, 3, n_features)`: x, y, likelihood) and `frameTimes`. `obj.bp.ev.goCue` and
the bitcode fields `obj.sglx.bitcode.bitstart` / `obj.sglx.fs` / `obj.bp.ev.bitStart` are also
needed to put the frames on the go-cue clock. The bottom camera's tongue markers
(`top_tongue` etc.) are not used.

ii.
```python
TONGUE_VIEW, TONGUE_FEAT = 0, "tongue"            # side camera
```
```python
i_tongue = feature_index(obj, TONGUE_VIEW, TONGUE_FEAT)
...
ft_side = frame_times(obj, TONGUE_VIEW, it)
if ft_side is not None:
    ts = np.asarray(ts_side[it], dtype=np.float64)
    if ts.ndim == 3 and ts.shape[0] == ft_side.size:
        x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)
        tongue_v[k] = feature_speed(x, y, subtract_baseline=False)
```
```python
def feature_index(obj, view, name):
    """findDLCFeatIndex.m -- index of a DLC feature in obj.traj{view}.featNames."""
```

iii. Step 5 Key Decision 11: "Tongue: side-camera `tongue` feature (`params.traj_features{1}{1}`),
the canonical tongue-tip marker." Step 10 Check 5 records that `featNames` was verified identical
across all trials of all 44 sessions (7 side / 10 bottom features), and that the resulting
"not visible" fraction (0.909) matches the raw DLC NaN rate measured directly in the files
(0.86–0.97 per session).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, mirroring `findPosition.m` + `findVelocity.m`.
(1) The frames' x and y are linearly interpolated from the corrected frame times onto the 500-bin
time axis, giving NaN outside the video's temporal coverage. Frames whose DLC likelihood is low
are already stored as NaN by the authors (the AI verified `isnan(x) == (likelihood <= 0.9)`
exactly), so the interpolation propagates "not visible" automatically; no explicit likelihood cut
is needed. No smoothing is applied to the tongue, exactly as `findPosition.m`
(`if ~contains(feat,'tongue') ... mySmooth(ts, 1, 'reflect')`).
(2) The speed is `‖(gradient(x), gradient(y))‖` in pixels per 10 ms bin, computed separately on
**each contiguous visible run** so that a visibility gap never contaminates the derivative at its
edges; an isolated single visible sample is given speed 0.
(3) No baseline-drift subtraction for the tongue (`findVelocity.m`: "NOT FOR TONGUE").
(4) Bins where the tongue is not tracked, or where the video does not cover the bin, stay NaN and
become class 2.

ii.
```python
def interp_feature(ts, featix, ft, align_time, taxis):
    """findPosition.m -- interpolate one DLC feature's (x, y) onto the aligned time base."""
    src_t = ft - align_time
    x = interp_trace(src_t, ts[:, 0, featix], taxis)
    y = interp_trace(src_t, ts[:, 1, featix], taxis)
    return x, y


def feature_speed(x, y, subtract_baseline):
    """findVelocity.m -- |d(pos)/dbin| at every timepoint where the feature is visible."""
    visible = np.isfinite(x) & np.isfinite(y)
    vx = np.full(x.shape, np.nan); vy = np.full(y.shape, np.nan)
    idx = np.flatnonzero(visible)
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], brk + 1)); stops = np.concatenate((brk, [idx.size - 1]))
    for s, e in zip(starts, stops):
        seg = idx[s:e + 1]
        if seg.size == 1:
            vx[seg] = 0.0; vy[seg] = 0.0
        else:
            vx[seg] = np.gradient(x[seg]); vy[seg] = np.gradient(y[seg])
    if subtract_baseline:
        ...
    return np.hypot(vx, vy)
```

iii. Step 5 Key Decision 12: "Velocity units are pixels per 10 ms bin (the reference's
`gradient()` on the interpolated position, no unit conversion). Only the within-session rank
matters after the median split." Step 10 Check 3, deliberate difference 3: "`findVelocity.m` runs
`gradient()` over the whole trial and then replaces NaN with 0 (tongue) or the nearest value
(other features), which is appropriate for assembling a dense regressor matrix but injects
spurious zero velocities at the edge of every visibility gap. Because the task spec makes
'not visible' its own class, the velocity only has to be defined where the feature *is* visible,
and a per-segment derivative is the correct estimate there."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the threshold is the **median (50th percentile) of the valid (finite) speeds over
all kept trials and all timepoints of that session**. Bins at or above the threshold get class 1,
bins below get class 0, and bins where the speed is NaN (tongue untracked, or the video does not
cover the bin) get class 2. Invalid bins are excluded from the percentile. The threshold is stored
in the session metadata.

ii.
```python
def discretise(v):
    valid = np.isfinite(v)
    out = np.full(v.shape, 2, dtype=np.int8)
    if valid.any():
        thr = float(np.median(v[valid]))
        out[valid] = (v[valid] >= thr).astype(np.int8)
    else:
        thr = np.nan
    return out, thr

tongue_c, thr_tongue = discretise(tongue_v)
```

iii. Step 5 Key Decision 8: "per session, the median (50th percentile) over all *valid*
(visible / video-covered) timepoints of all kept trials in that session. Invalid timepoints are
class 2 and are excluded from the percentile, otherwise the 'not visible' bins (≈90% of the tongue
trace) would drag the threshold to 0." Step 10 Check 5 verifies the split is exact: class-0 and
class-1 counts per session differ by a median of 1 over all 132 session × output combinations.
Resulting distribution: below 0.046 / above 0.046 / not_visible 0.909.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera runs on its own clock, which starts before the behaviour clock. The offset is
computed once per session from the bitcode pulse both streams record — exactly
`findVideoOffset.m`: `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`. Each frame's
time relative to the go cue is then `frameTimes − vidshift − goCue[trial]`, and the x/y traces are
linearly interpolated onto the same 500-bin, 10 ms `TIME` grid used for the spikes, with
`params.advance_movement = 0` (no lag). `np.interp` returns NaN outside the frame-time range, so
bins the video does not cover become class 2.

ii.
```python
def video_shift(obj):
    """findVideoOffset.m -- seconds to subtract from traj frameTimes to get trial time."""
    bit_start = mode_of(vec(obj["bp"]["ev"]["bitStart"]))
    fs = float(vec(obj["sglx"]["fs"])[0])
    vid_file_offset = mode_of(vec(obj["sglx"]["bitcode"]["bitstart"])) / fs
    return vid_file_offset - bit_start
```
```python
vidshift = video_shift(obj)
taxis = TIME + ADVANCE_MOVEMENT
...
align = gocue[it] + vidshift
x, y = interp_feature(ts, i_tongue, ft_side, align, taxis)   # src_t = ft - align
```
```python
def interp_trace(src_t, src_y, taxis):
    """interp1(src_t, src_y, taxis) -- linear, NaN outside the source range."""
    return np.interp(taxis, src_t, src_y, left=np.nan, right=np.nan)
```

iii. Step 4: "`vidshift ≈ 0.49 s` puts the first video frame at 0.025 s in trial time, i.e.
essentially at `bitStart` (0.04 s) — the tutorial's 'subtract 0.5 second from frametimes'".
Step 10 Check 3 (c): "identical formulas, same `findVideoOffset`". The offset is computed once per
session rather than per trial. Step 7/12 record the visual verification in `processing_*.png`
panel 4 (raw 400 Hz DLC tongue y overlaid on the interpolated 100 Hz trace and the class vector)
and the time-resolved decoding curves.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{2}` — the **bottom camera only**, the only view in which
the paws are tracked — using **both** paw features, `top_paw` and `bottom_paw`. The same `ts` /
`frameTimes` fields and the same session video offset are used as for the tongue.

ii.
```python
PAW_VIEW, PAW_FEATS = 1, ("top_paw", "bottom_paw")  # bottom camera (paws only tracked there)
```
```python
i_paws = [feature_index(obj, PAW_VIEW, f) for f in PAW_FEATS]
...
ft_bot = frame_times(obj, PAW_VIEW, it)
if ft_bot is not None:
    ts = np.asarray(ts_bot[it], dtype=np.float64)
    if ts.ndim == 3 and ts.shape[0] == ft_bot.size:
        sp = []
        for ip in i_paws:
            x, y = interp_feature(ts, ip, ft_bot, align, taxis)
            sp.append(feature_speed(x, y, subtract_baseline=True))
```

iii. Step 5 Key Decision 10: "Paw: mean speed of the visible subset of {`top_paw`, `bottom_paw`}
(bottom camera only, as in the Methods). Using a single paw would make 'not visible' reach 88–96%
in some sessions because DLC loses one paw or the other; the union is 0–17%." Both features are in
the reference's `params.traj_features{2}`. Step 10 Check 3, deliberate difference 4, repeats the
coverage measurements.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue for each paw — interpolate x and y onto the 10 ms grid, then
`‖(gradient(x), gradient(y))‖` per contiguous visible run — **plus** the reference's
baseline-drift subtraction for non-tongue features: `basederiv = median(diff(pos), 'omitnan')` is
subtracted from the velocity. The AI reproduces `findVelocity.m` literally here, including its
apparent typo of subtracting `basederiv(1)` (the x-axis median) from *both* axes. The two paws'
speeds are then combined with `np.nanmean` over the visible subset, so a bin is NaN (class 2) only
when **neither** paw is tracked. No cross-view normalisation is applied (there is only one camera).

ii.
```python
    if subtract_baseline:
        # findVelocity.m: basederiv = median(diff(pos),'omitnan'); xvel/yvel -= basederiv(1)
        d = np.diff(np.stack([x, y], axis=1), axis=0)
        base = np.nanmedian(d, axis=0)
        if np.isfinite(base[0]):
            vx = vx - base[0]
            vy = vy - base[0]
    return np.hypot(vx, vy)
```
```python
sp = np.stack(sp, axis=0)
paw_v[k] = np.nanmean(sp, axis=0)   # mean over the visible paw(s)
```

iii. Step 4 Discrepancies: "`findVelocity` baseline subtraction … subtracts `basederiv(1)` from
**both** `xvel` and `yvel` → Apparent typo in the reference; effect is negligible (both medians
≈0 px/bin). Reproduced **as written** to stay faithful." Step 10 Check 3, deliberate difference 4:
"DLC loses `top_paw` on up to 88% of frames in some sessions and `bottom_paw` on up to 96% in
others, so a single marker would make 'not visible' the dominant class for reasons that have
nothing to do with the animal. The union is unavailable on only 0.6–24% of bins."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Exactly as for the tongue: the same `discretise` helper, applied to the combined paw speed
matrix of the session. The per-session 50th percentile of the finite values splits class 0
(below) from class 1 (at or above); NaN bins — where neither paw is tracked or the video does not
cover the bin — become class 2. Full-dataset distribution: below 0.470 / above 0.470 /
not_visible 0.059.

ii.
```python
paw_c, thr_paw = discretise(paw_v)
```
```python
OUTPUT_VALUES = [
    ...
    ["below_median", "above_median", "not_visible"],   # paw_velocity
    ...
]
```

iii. Same as 7-c: Step 5 Key Decision 8 and the Decoder Task specification
("0: < 50th percentile, 1: >= 50th percentile, 2: not visible", per-session threshold).
Step 10 Check 5 records that the class-0/class-1 balance is exact to ±1 bin per session.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but timed by the **bottom** camera's own `frameTimes`: the session
video offset is subtracted, then the trial's go cue, and the positions are interpolated onto the
same 500-bin, 10 ms grid as the spikes. Using each view's own frame times matters because the two
cameras can record different numbers of frames; `frame_times` is called separately for each view
and the code guards that `ts.shape[0] == frameTimes.size` before using a trial.

ii.
```python
ft_bot = frame_times(obj, PAW_VIEW, it)
...
align = gocue[it] + vidshift
x, y = interp_feature(ts, ip, ft_bot, align, taxis)
```
```python
def frame_times(obj, view, itrial):
    """frameTimes for one trial, or None when the video timing is unusable."""
    ft = obj["traj"][view]["frameTimes"]
    ft = ft[itrial] if isinstance(ft, list) else ft
    if ft is None:
        return None
    a = np.ravel(np.asarray(ft, dtype=np.float64))
    if a.size < 2 or not np.all(np.isfinite(a)):
        return None
    return a
```

iii. Same offset and grid as every other stream (Step 10 Check 3 (c)), so the paw needs no
separate treatment. Step 10 Check 5 notes the video coverage limits explicitly: "video starts
≈0.03 s into the trial and ends 5–9 s in, so short-delay trials have no video before the window
start and some long-delay randomized trials none after +2.5 s → those bins are class 2, which is
exactly what 'no video' means" — visible in the per-session not-visible fractions
(0.006–0.024 for fixed-delay, 0.07–0.08 for randomized-delay sessions).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file per session, `motionEnergy_<anm>_<date>.mat`, holding one trace per trial with
one value per side-camera frame. `obj.me` is used only as a fallback when the standalone file is
missing. Three different on-disk layouts occur (a bare cell array, `{data, moveThresh}`, and
`{data: {data, ...}}`), all normalised by the loader. The side camera's `frameTimes` provide the
time base.

ii.
```python
def load_motion_energy(anm, date, task, obj, ntrials):
    """loadMotionEnergy.m -- list of per-trial 400 Hz motion-energy traces (or None)."""
    me = None
    p = me_path(anm, date, task)
    if os.path.exists(p):
        me = read_mat(p)["me"]
    elif "me" in obj:
        me = obj["me"]
    if me is None:
        return [None] * ntrials
    if isinstance(me, dict):                      # struct with .data (+ .moveThresh)
        me = me["data"]
        if isinstance(me, dict):                  # me.data.data (older objects)
            me = me["data"]
    me = as_trial_list(me, ntrials)
    ...
```

iii. Step 4 Discrepancies: "`loadMotionEnergy` expects `me.data` (struct); JEB23/JEB24
`motionEnergy_*.mat` store a bare cell array; those sessions also carry `obj.me` → Loader
normalises both layouts to a list of per-trial traces." Step 10 Check 3 (a) notes this is a
superset of the reference loader ("falling back to `obj.me` for the JEB23/JEB24 layout the
reference loader does not handle").

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling. The value is already a single scalar per frame — the paper's Methods
compute it per pixel as the absolute difference between the median over the next five frames and
the median over the previous five frames, then reduce each frame to the 99th percentile across
pixels — so nothing is smoothed, differentiated, or recombined. The trace is linearly interpolated
onto the 500-bin 10 ms axis, and bins outside the video's coverage become NaN → class 2. A trial
is used only if its trace length equals the side camera's frame count (verified to hold for every
trial of every session), and traces of length ≤ 1 are treated as missing.

ii.
```python
tr = me_traces[it]
if tr is not None and ft_side is not None and tr.size == ft_side.size:
    me_v[k] = interp_trace(ft_side - align, tr, taxis)
```
```python
a = np.ravel(np.asarray(me[i], dtype=np.float64))
out.append(a if a.size > 1 else None)
```

iii. Step 10 Check 5: "motion-energy trace length vs `frameTimes`: equal for every trial of every
session (checked all 44); a mismatch would fall back to class 2." Step 5 Variable Mapping lists the
transform as "interp to `obj.time` using `frameTimes − vidshift − goCue`; class 2 where the video
does not cover the bin; else 0/1 by the **session** median of covered bins", citing
`loadMotionEnergy.m`. The spatial reduction is already done upstream, so re-deriving anything
would only discard information.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretise` helper and the same rule: the per-session 50th percentile of the finite
interpolated values over all kept trials splits class 0 (below) from class 1 (at or above); bins
the video does not cover become class 2, here named `no_video`. The AI did **not** use the
authors' own per-session manual movement threshold (`me.moveThresh`), because the task spec
prescribes a median split. Full-dataset distribution: below 0.480 / above 0.481 /
no_video 0.039.

ii.
```python
me_c, thr_me = discretise(me_v)
```
```python
OUTPUT_VALUES = [
    ...
    ["below_median", "above_median", "no_video"],     # motion_energy
]
```

iii. Step 5 Key Decision 8 (the shared thresholding rule) and the Decoder Task specification.
Step 10 Check 5 addresses the one wrinkle: "ties at the median — `motion_energy` in 5 sessions has
up to 1.9% more class-1 than class-0 bins because the raw motion-energy values are quantised and
many bins equal the median exactly → correct behaviour of the specified `>=` rule".

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the tracking. Motion energy has exactly one value per frame of the
**side** camera, so `ft_side` (corrected by the session video offset and the trial's go cue) is
its time base, and `np.interp` places it on the shared 500-bin 10 ms axis, returning NaN outside
the video's coverage.

ii.
```python
align = gocue[it] + vidshift
ft_side = frame_times(obj, TONGUE_VIEW, it)      # TONGUE_VIEW == 0 == side camera
...
if tr is not None and ft_side is not None and tr.size == ft_side.size:
    me_v[k] = interp_trace(ft_side - align, tr, taxis)
```

iii. Step 10 Check 3 (c): the same `findVideoOffset` and the same `frameTimes − vidshift − goCue`
formula as the tracked features, with `params.advance_movement = 0`. The
`tr.size == ft_side.size` guard is itself the evidence that motion energy follows the side
camera's frames. Step 12 checks the result behaviourally: mean "above median" motion energy rises
sharply at t = 0 and differs by context (`processing_*.png` panel 8), and the harness decodes
motion energy at 0.714 balanced accuracy, the highest of the three kinematic outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Nothing is imputed; gaps are kept and marked. The cases handled:
- **Trials with unusable `frameTimes`** (all-NaN, or fewer than 2 frames, or any non-finite value):
  `frame_times` returns `None` and the trial's tongue/paw/motion-energy rows stay NaN → 500 bins of
  class 2. Exactly 3 such trials exist dataset-wide; 2 of them are early-lick/stim trials dropped
  anyway. The trial is kept, because its neural and behavioural data are unaffected.
- **Untracked frames**: DLC's low-likelihood frames are already NaN in the file (verified:
  `isnan(x) == (likelihood <= 0.9)` exactly), so they propagate through `np.interp` to NaN →
  class 2.
- **Video not covering the window**: `np.interp(..., left=np.nan, right=np.nan)` gives NaN outside
  the recorded frame range → class 2.
- **Frame-count / trace-length mismatches**: guarded by `ts.ndim == 3 and ts.shape[0] == ft.size`
  and `tr.size == ft_side.size`; a mismatch falls back to class 2. Each view is timed by its own
  `frameTimes`.
- **Trials with no spikes at all** (ephys recording ended early): dropped entirely (see 1-e).
- **Clusters with a non-string `quality`** (7 dataset-wide): treated as `''` and kept, reproducing
  `findClusters.m`.
- **Empty / `DUMMY` probes** (JEB6 probe 1, JEB15 2022-07-29 probe 1): `select_clusters` skips
  non-dict or `quality`-less entries; neither is in the reference probe list anyway.
- **Single-element MATLAB cell arrays** that `pymatreader` unwraps: re-wrapped by `as_cell_list`.
- **Motion energy missing or degenerate** (`None`, size ≤ 1): treated as missing → class 2.
- **Out-of-range or non-finite spike times/trial numbers**: masked out before binning.

ii.
```python
    if a.size < 2 or not np.all(np.isfinite(a)):
        return None
```
```python
ok = (tr >= 0) & (tr < ntrials) & np.isfinite(tm)
tm, tr = tm[ok], tr[ok]
if tm.size == 0:
    continue
```
```python
def discretise(v):
    valid = np.isfinite(v)
    out = np.full(v.shape, 2, dtype=np.int8)     # NaN -> "not visible" / "no video"
```

iii. Step 10 Check 5 tabulates every edge case with its finding and handling, and Step 5 Key
Decision 9 explains the principle: "'Not visible' / 'no video' covers both DLC drop-out and bins
outside the video's temporal coverage … Three trials in the whole dataset have unusable
`frameTimes` (all NaN); their kinematic/ME outputs are entirely class 2." The AI verified the one
surviving such trial (JEB19 2023-04-19 → converted trial 208) is entirely class 2 for all three
video outputs. Nothing is interpolated or nearest-filled, in deliberate contrast to
`findVelocity.m`/`findPosition.m`, because the task spec provides an explicit class for missing
data (Step 10 Check 3, deliberate difference 3).

## 11-a. What are the most time-consuming steps of the code?

i. Reading the `.mat` files. The per-session timing printed by the script shows load times of
1.1–7.0 s against 0.1–0.8 s for all neural processing and 0.2–0.4 s for all video processing —
i.e. loading is 80–90% of each session's runtime. `pymatreader` materialises the whole `obj` tree,
including `clu.spkWavs` (~1.4 GB for the largest 290 MB session), which the conversion never uses.
The whole 44-session conversion takes **14.5 s wall clock** with 16 worker processes (≈0.3 s per
session wall, ≈4 s CPU), plus 2.4 s to write the 1.61 GB pickle.

ii.
```python
    t0 = time.time()
    obj = load_obj(anm, date, task)
    t_load = time.time() - t0
    ...
    timing=dict(load=t_load, neural=t_neural, video=t_video, total=time.time() - t0),
```
```python
print(f"  [{len(results):2d}/{len(sessions)}] {i['session']:>18s} "
      f"... t={i['timing']['total']:.1f}s "
      f"(load {i['timing']['load']:.1f}, neural {i['timing']['neural']:.1f}, "
      f"video {i['timing']['video']:.1f})", flush=True)
```

iii. Step 6: "`pymatreader` reads the whole `obj`, including `clu.spkWavs`, which dominates the
runtime (6–8 s and ~1.4 GB for the largest 290 MB session)." Step 7 records the run-time estimate
and the speed-ups; Step 9 notes the full run took 17 s against a conservative 90 s estimate. The
data must be read once regardless, so the AI addressed it by parallelising across sessions rather
than by changing the reader.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorised the two loops that would have dominated: spike binning is a single
`np.bincount` over a flattened `(trial, bin)` index per cluster instead of a per-trial ×
per-neuron loop (~10⁴ iterations/session), and smoothing is applied once per session to a
`(515, nclusters·ntrials)` matrix instead of per column. Sessions are run in parallel processes.
The loops that remain, and could in principle be vectorised further:
- `for ci, (p, i) in enumerate(cluster_ids)` in `bin_and_smooth` — could be one `bincount` over a
  flattened `(cluster, trial, bin)` index across all clusters at once.
- `for k, it in enumerate(trials)` in `process_session`, which interpolates each trial's video —
  genuinely ragged (each trial has a different frame count), so this is hard to vectorise; frames
  could be concatenated across trials with an offset trick, as the reference solution does for
  spikes.
- `for s, e in zip(starts, stops)` in `feature_speed` (per contiguous visible run) — could be done
  with a segmented-gradient formulation.
- `for k in range(half, n)` in `my_smooth` — 8 taps, already the fastest exact form.
- `for k, it in enumerate(trials)` in the assembly block, which slices `rates` per trial; could
  be a single list comprehension over a transposed array.
None of these matter, because loading dominates the runtime.

ii.
```python
flat = tr[ok] * NT + b[ok]
counts[ci] = np.bincount(flat, minlength=ntrials * NT).reshape(ntrials, NT)
```
```python
m = rate.reshape(-1, NT).T                       # (NT, nclu*ntrials)
m = my_smooth(m)
```
```python
with ProcessPoolExecutor(max_workers=args.workers) as ex:
    for r in ex.map(process_session, sessions):
```

iii. Step 6 "Code speedups added": "Spike binning is a single `np.bincount` per cluster over a
flattened `(trial, bin)` index instead of a per-trial loop"; "Smoothing is applied once per session
to a `(515, nclusters·ntrials)` matrix, as an 8-tap causal FIR (the kernel's first 7 taps are
zero), so no FFT and no round-off"; "Sessions are processed in parallel with `ProcessPoolExecutor`
(12–16 workers; each worker peaks at ~1.5 GB)". Step 7 confirms the resulting estimate and Step 9
the 14.5 s measured runtime.

## 11-c. What processing does the code repeat multiple times?

i. The AI eliminated the repeats that cost anything — each `.mat` file is read once per
`process_session` call, the video offset is computed once per session rather than per trial, the
DLC feature indices are resolved once per session, and the bin grid and smoothing kernel are built
once at module level and shared by every trial, session and stream. What is still recomputed:
- `trials_with_ephys(obj, probes, ntrials_all)` runs **twice** per session — once inside
  `select_trials` and again to fill the `n_trials_no_ephys` metadata field.
- `vec(bp["early"])` and `vec(bp["stim"]["enable"])` are re-read for the same metadata dict.
- In `--show-processing` mode, `show_processing` calls `load_obj` again, re-reading the entire
  session file, and re-runs `select_trials`, `select_clusters`, `video_shift`,
  `load_motion_energy`, `feature_index` and `interp_feature` — a complete second pass over data the
  worker already had. This affects only the two diagnostic sessions, not the full conversion.
All of these are cheap relative to loading, and none of them affects the output.

ii. Computed once per session and reused:
```python
vidshift = video_shift(obj)
taxis = TIME + ADVANCE_MOVEMENT
me_traces = load_motion_energy(anm, date, task, obj, ntrials_all)
i_tongue = feature_index(obj, TONGUE_VIEW, TONGUE_FEAT)
i_paws = [feature_index(obj, PAW_VIEW, f) for f in PAW_FEATS]
```
Recomputed for metadata:
```python
info = dict(
    ...
    n_trials_early=int(np.sum(vec(bp["early"])[:ntrials_all] > 0)),
    n_trials_stim=int(np.sum(vec(bp["stim"]["enable"])[:ntrials_all] > 0)),
    n_trials_no_ephys=int(np.sum(~trials_with_ephys(obj, probes, ntrials_all))),
```
Second full read in the diagnostic path:
```python
def show_processing(args, result, outdir="/app"):
    anm, date, probes, task = args
    obj = load_obj(anm, date, task)
    bp = obj["bp"]
    trials, ntrials_all = select_trials(bp, obj, probes)
```

iii. Step 6 documents the speed-ups that removed the expensive repeats (single `bincount`,
one batched smoothing call, parallel sessions). The remaining repeats are bookkeeping for the
`session_info` metadata (which Step 10 Check 2 then uses as provenance for the independent sanity
checks) and the once-per-run diagnostic plots, and the AI did not flag them because the measured
runtime (14.5 s) was already far below the 15-minute budget in the instructions.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four kinds, in decreasing cost:
- **Loading**: `pymatreader` walks the whole `obj` tree, materialising `clu.spkWavs`, `clu.tm`,
  `sglx`'s per-trial index arrays and the ~13 DLC features other than `tongue`, `top_paw` and
  `bottom_paw` — none of which reach the output. The AI identified this as the dominant cost.
- **Binning and smoothing of trials that are then dropped**: `bin_and_smooth` runs over all
  `ntrials_all` raw trials and the result is subset to the kept trials afterwards
  (`rates[:, trials, :]`), so ~8% of the spike binning and smoothing work is discarded.
- **Binning and smoothing of clusters that are then dropped**: the >1 Hz filter is applied after
  smoothing, so the 57 sub-1 Hz units dataset-wide are fully processed and then thrown away.
- **Metadata-only computations**: the discretisation thresholds returned by `discretise`, the
  `n_trials_early` / `n_trials_stim` / `n_trials_no_ephys` counts, `n_autowater_kept`,
  `trial_index` / `cluster_index`, and per-trial `lick` / `context` / `outcome` values for raw
  trials that are never emitted. These are kept deliberately as provenance for the sanity checks.
Everything else computed after loading ends up in the output.

ii.
```python
rates = bin_and_smooth(obj, cluster_ids, gocue, ntrials_all)      # (nclu, ntr, NT)
rates = rates[:, trials, :]                                       # then subset
n_quality = len(cluster_ids)
if n_quality:
    mean_fr = rates.reshape(n_quality, -1).mean(axis=1)           # removeLowFRClusters.m
    keep_unit = mean_fr > LOW_FR
    rates = rates[keep_unit]
```
```python
    trial_index=trials.astype(np.int32),   # 0-based index into the raw session's trials
    cluster_index=np.array([(p + 1, i) for p, i in cluster_ids], dtype=np.int32
                           )[keep_unit] if n_quality else np.zeros((0, 2), np.int32),
```

iii. Step 6: "`pymatreader` reads the whole `obj`, including `clu.spkWavs`, which dominates the
runtime". The order of the >1 Hz filter is not an oversight but a requirement — the reference's
`removeLowFRClusters.m` defines the mean rate on the *binned* data (`mean(mean(obj.psth,3))`), so
the rates must exist before the filter can be applied. Step 10 Check 2 explains why the
metadata-only arrays exist: "Provenance for the comparison comes from the `trial_index` and
`cluster_index` arrays now stored in `metadata['session_info']`", which let the independent sanity
checks address raw trials and clusters without importing the conversion functions.
