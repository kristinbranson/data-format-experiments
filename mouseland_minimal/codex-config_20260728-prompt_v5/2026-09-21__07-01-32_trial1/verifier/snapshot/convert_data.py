import os
import pickle
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path("/app")
DATA_DIR = ROOT / "data"
SPK_DIR = DATA_DIR / "spk"
BEH_DIR = DATA_DIR / "beh"
RETINO_DIR = DATA_DIR / "retinotopy"
OUTPUT_PKL = ROOT / "converted_data.pkl"
NEURAL_DIR = ROOT / "converted_data_neural"

TRIAL_DURATION_MAX_S = 40.0
MAX_FRAME_GAP_S = 1.0
MIN_FRAMES_PER_TRIAL = 5
INPUT_NAMES = [
    "time_to_sound_cue_s",
    "day_of_training",
    "time_since_trial_start_s",
    "reward_available",
]
OUTPUT_NAMES = [
    "stimulus_category",
    "licking",
    "position_bin",
    "running_speed_bin",
]
STIMULUS_CATEGORIES = [
    "circle1",
    "circle2",
    "circle3",
    "leaf1",
    "leaf2",
    "leaf3",
    "leaf1_swap1",
    "leaf1_swap2",
]
STIMULUS_TO_INDEX = {name: idx for idx, name in enumerate(STIMULUS_CATEGORIES)}
WALL_TO_CANONICAL = {
    "circle1": "circle1",
    "rock1": "circle1",
    "circle2": "circle2",
    "rock2": "circle2",
    "circle3": "circle3",
    "leaf1": "leaf1",
    "wood1": "leaf1",
    "leaf2": "leaf2",
    "wood2": "leaf2",
    "leaf3": "leaf3",
    "wood5": "leaf3",
    "leaf1_swap1": "leaf1_swap1",
    "wood1_swap1": "leaf1_swap1",
    "leaf1_swap2": "leaf1_swap2",
    "wood1_swap2": "leaf1_swap2",
}
REGION_NAMES = ["V1", "mHV", "aHV", "lHV", "unknown"]
REGION_TO_INDEX = {name: idx for idx, name in enumerate(REGION_NAMES)}
_MEMMAP_CACHE = {}


def _ensure_importable_module_name():
    if __name__ == "__main__":
        sys.modules.setdefault("convert_data", sys.modules[__name__])


_ensure_importable_module_name()


def raw_session_key_from_behavior_key(key: str) -> str:
    if key.endswith("_swap1") or key.endswith("_swap2"):
        return key.rsplit("_", 1)[0]
    return key


def canonical_stimulus_name(wall_name: str) -> str:
    if wall_name not in WALL_TO_CANONICAL:
        raise KeyError(f"Unknown wall name: {wall_name}")
    return WALL_TO_CANONICAL[wall_name]


def canonical_stimulus_index(wall_name: str) -> int:
    return STIMULUS_TO_INDEX[canonical_stimulus_name(wall_name)]


def parse_session_key(raw_key: str):
    parts = raw_key.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected raw session key: {raw_key}")
    subject = parts[0]
    datestr = "_".join(parts[1:4])
    blk = parts[4]
    return subject, datestr, blk


def session_date(raw_key: str) -> datetime:
    _, datestr, _ = parse_session_key(raw_key)
    return datetime.strptime(datestr, "%Y_%m_%d")


def region_index_from_iarea(iarea: np.ndarray) -> np.ndarray:
    iarea = np.asarray(iarea)
    out = np.full(iarea.shape, REGION_TO_INDEX["unknown"], dtype=np.int16)
    out[iarea == 8] = REGION_TO_INDEX["V1"]
    out[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = REGION_TO_INDEX["mHV"]
    out[(iarea == 3) | (iarea == 4)] = REGION_TO_INDEX["aHV"]
    out[(iarea == 5) | (iarea == 6)] = REGION_TO_INDEX["lHV"]
    return out


def memmap_cache_key(path: str, base_shape, dtype_str: str):
    return (path, tuple(base_shape), dtype_str)


def get_cached_memmap(path: str, base_shape, dtype_str: str):
    key = memmap_cache_key(path, base_shape, dtype_str)
    arr = _MEMMAP_CACHE.get(key)
    if arr is None:
        arr = np.memmap(path, dtype=np.dtype(dtype_str), mode="r", shape=tuple(base_shape))
        _MEMMAP_CACHE[key] = arr
    return arr


def rebuild_mapped_trial(path: str, base_shape, dtype_str: str, row_start: int, row_stop: int):
    base = get_cached_memmap(path, base_shape, dtype_str)
    arr = base[row_start:row_stop, :].T.view(MappedTrialArray)
    arr._mapped_meta = (path, tuple(base_shape), dtype_str, int(row_start), int(row_stop))
    return arr


class MappedTrialArray(np.ndarray):
    def __new__(cls, path: str, base_shape, dtype_str: str, row_start: int, row_stop: int):
        return rebuild_mapped_trial(path, base_shape, dtype_str, row_start, row_stop)

    def __array_finalize__(self, obj):
        self._mapped_meta = getattr(obj, "_mapped_meta", None)

    def __reduce__(self):
        return (rebuild_mapped_trial, self._mapped_meta)


MappedTrialArray.__module__ = "convert_data"
rebuild_mapped_trial.__module__ = "convert_data"


@dataclass
class TrialPlan:
    trial_index: int
    frame_idx: np.ndarray
    duration_s: float
    max_gap_s: float
    row_start: int
    row_stop: int


@dataclass
class SessionPlan:
    raw_key: str
    subject: str
    date: datetime
    block: str
    behavior_file: str
    behavior_key: str
    experiment_types: list[str]
    nframes_behavior: int
    trial_plans: list[TrialPlan]
    median_dt_s: float
    region_idx: np.ndarray
    day_of_training: float | None = None

    @property
    def total_frames_kept(self) -> int:
        return sum(tp.row_stop - tp.row_start for tp in self.trial_plans)


def compute_trial_plans_from_behavior(beh, nfr: int):
    ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
    ft_tr = np.asarray(beh["ft_trInd"][:nfr])
    corr = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    speed = np.clip(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32), 0.0, None)
    gray_time = np.asarray(beh["Gray_space_time"], dtype=np.float64)
    trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)

    dt = np.diff(ft) * 24.0 * 3600.0
    median_dt_s = float(np.median(dt)) if dt.size else np.nan

    row_cursor = 0
    trial_plans = []
    speed_values = []
    ntrials = int(beh["ntrials"])
    for tr in range(ntrials):
        idx = np.flatnonzero((ft_tr == tr) & corr)
        if idx.size < MIN_FRAMES_PER_TRIAL:
            continue
        duration_s = float((gray_time[tr] - trial_start[tr]) * 24.0 * 3600.0)
        max_gap_s = 0.0
        if idx.size > 1:
            max_gap_s = float(np.max(np.diff(ft[idx]) * 24.0 * 3600.0))
        if duration_s > TRIAL_DURATION_MAX_S or max_gap_s > MAX_FRAME_GAP_S:
            continue

        nframes_trial = int(idx.size)
        trial_plans.append(
            TrialPlan(
                trial_index=tr,
                frame_idx=idx.astype(np.int32),
                duration_s=duration_s,
                max_gap_s=max_gap_s,
                row_start=row_cursor,
                row_stop=row_cursor + nframes_trial,
            )
        )
        row_cursor += nframes_trial
        speed_values.append(speed[idx])

    return trial_plans, median_dt_s, speed_values


def build_behavior_index():
    candidates = defaultdict(list)
    behavior_files = sorted(
        fp for fp in BEH_DIR.glob("Beh_*.npy")
        if fp.name not in {"Imaging_Exp_info.npy", "example_bef_and_aft_learning_behavior.npy"}
        and fp.is_file()
    )
    for fp in behavior_files:
        beh = np.load(fp, allow_pickle=True).item()
        for key, dat in beh.items():
            raw_key = raw_session_key_from_behavior_key(key)
            candidates[raw_key].append(
                {
                    "file": fp.name,
                    "key": key,
                    "uniq_walls": len(np.unique(dat["WallName"])),
                    "n_finite_stim_id": int(np.isfinite(np.asarray(dat["stim_id"], dtype=float)).sum()),
                    "placeholder_count": int(
                        np.sum(np.asarray(dat["TrialStim"]) == "stimulus_of_trial")
                    ),
                    "has_swap_suffix": int(key.endswith("_swap1") or key.endswith("_swap2")),
                }
            )
    return candidates


def select_behavior_candidate(raw_key: str, candidates_for_session):
    if not candidates_for_session:
        raise KeyError(f"No behavior data found for session {raw_key}")

    def score(item):
        return (
            item["uniq_walls"],
            item["n_finite_stim_id"],
            -item["placeholder_count"],
            -item["has_swap_suffix"],
            item["file"],
            item["key"],
        )

    best = max(candidates_for_session, key=score)
    return best["file"], best["key"]


def build_experiment_type_index():
    info = np.load(BEH_DIR / "Imaging_Exp_info.npy", allow_pickle=True).item()
    out = defaultdict(set)
    for exp_type, rows in info.items():
        for row in rows:
            raw_key = f"{row['mname']}_{row['datexp']}_{row['blk']}"
            out[raw_key].add(exp_type)
    return {k: sorted(v) for k, v in out.items()}


def build_session_plans():
    behavior_index = build_behavior_index()
    exp_index = build_experiment_type_index()
    session_plans = []
    speed_values = []
    time_bin_values = []

    spk_files = sorted(SPK_DIR.glob("*_neural_data.npy"))
    for spk_file in spk_files:
        raw_key = spk_file.name.replace("_neural_data.npy", "")
        subject, _, block = parse_session_key(raw_key)
        date = session_date(raw_key)
        beh_fname, beh_key = select_behavior_candidate(raw_key, behavior_index[raw_key])
        beh = np.load(BEH_DIR / beh_fname, allow_pickle=True).item()[beh_key]

        nfr = len(beh["ft"]) - 1
        trial_plans, median_dt_s, speed_values_curr = compute_trial_plans_from_behavior(beh, nfr)
        if np.isfinite(median_dt_s):
            time_bin_values.append(median_dt_s)
        speed_values.extend(speed_values_curr)

        if len(trial_plans) < 2:
            raise RuntimeError(
                f"Session {raw_key} has fewer than 2 valid trials after filtering "
                f"({len(trial_plans)} kept)"
            )

        retino = np.load(RETINO_DIR / f"{raw_key.rsplit('_', 1)[0]}_trans.npz", allow_pickle=True)
        region_idx = region_index_from_iarea(retino["iarea"])
        session_plans.append(
            SessionPlan(
                raw_key=raw_key,
                subject=subject,
                date=date,
                block=block,
                behavior_file=beh_fname,
                behavior_key=beh_key,
                experiment_types=exp_index.get(raw_key, []),
                nframes_behavior=nfr,
                trial_plans=trial_plans,
                median_dt_s=median_dt_s,
                region_idx=region_idx,
            )
        )

    subjects = sorted({plan.subject for plan in session_plans})
    first_date = {
        subject: min(plan.date for plan in session_plans if plan.subject == subject)
        for subject in subjects
    }
    for plan in session_plans:
        plan.day_of_training = float((plan.date - first_date[plan.subject]).days)

    all_speed = np.concatenate(speed_values).astype(np.float32)
    speed_quantiles = np.quantile(all_speed, [0.25, 0.50, 0.75]).astype(np.float32)
    median_time_bin_ms = 1000.0 * float(np.median(np.asarray(time_bin_values)))
    return session_plans, subjects, speed_quantiles, median_time_bin_ms


def load_behavior(plan: SessionPlan):
    beh = np.load(BEH_DIR / plan.behavior_file, allow_pickle=True).item()
    return beh[plan.behavior_key]


def stimulus_series_for_trial(beh, tr: int, T: int):
    stim_idx = canonical_stimulus_index(str(beh["WallName"][tr]))
    return np.full((T,), stim_idx, dtype=np.int16)


def lick_series_for_trial(beh, tr: int, frame_idx: np.ndarray, nfr: int):
    lick = np.zeros((frame_idx.size,), dtype=np.int16)
    lick_frames = np.asarray(np.rint(beh["LickFr"]).astype(np.int64))
    lick_trials = np.asarray(np.rint(beh["LickTrind"]).astype(np.int64))
    sel = lick_trials == int(tr)
    if not np.any(sel):
        return lick
    lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
    frame_to_rel = np.full((nfr,), -1, dtype=np.int32)
    frame_to_rel[frame_idx] = np.arange(frame_idx.size, dtype=np.int32)
    rel = frame_to_rel[lick_frames]
    rel = rel[rel >= 0]
    if rel.size:
        lick[np.unique(rel)] = 1
    return lick


def position_series_for_trial(pos_trial: np.ndarray):
    pos_clipped = np.clip(pos_trial, 0.0, 39.999)
    return np.floor(pos_clipped / 10.0).astype(np.int16)


def speed_series_for_trial(speed_trial: np.ndarray, speed_quantiles: np.ndarray):
    return np.digitize(speed_trial, speed_quantiles, right=True).astype(np.int16)


def write_session_neural_memmap(plan: SessionPlan, session_idx: int):
    beh = load_behavior(plan)
    spk_obj = np.load(SPK_DIR / f"{plan.raw_key}_neural_data.npy", allow_pickle=True).item()
    spk_chunks = spk_obj["spks"]
    if not spk_chunks:
        raise RuntimeError(f"No spike chunks found for {plan.raw_key}")

    spk_nfr = int(spk_chunks[0].shape[1])
    if spk_nfr != plan.nframes_behavior:
        if abs(spk_nfr - plan.nframes_behavior) > 2:
            raise RuntimeError(
                f"Frame mismatch for {plan.raw_key}: behavior has {plan.nframes_behavior}, "
                f"spikes have {spk_nfr}"
            )
        trial_plans, median_dt_s, _ = compute_trial_plans_from_behavior(beh, spk_nfr)
        if len(trial_plans) < 2:
            raise RuntimeError(
                f"Session {plan.raw_key} has fewer than 2 valid trials after spike-frame trimming"
            )
        plan.nframes_behavior = spk_nfr
        plan.trial_plans = trial_plans
        plan.median_dt_s = median_dt_s

    total_neurons = int(sum(chunk.shape[0] for chunk in spk_chunks))
    if total_neurons != len(plan.region_idx):
        raise RuntimeError(
            f"Region count mismatch for {plan.raw_key}: retinotopy has {len(plan.region_idx)}, "
            f"spikes have {total_neurons}"
        )

    keep_idx = np.concatenate([tp.frame_idx for tp in plan.trial_plans]).astype(np.int64)
    out_path = NEURAL_DIR / f"session_{session_idx:03d}_{plan.raw_key}.dat"
    base_shape = (int(keep_idx.size), total_neurons)
    base = np.memmap(out_path, dtype=np.float16, mode="w+", shape=base_shape)

    col_start = 0
    time_block = 4096
    for chunk in spk_chunks:
        n_chunk = int(chunk.shape[0])
        for c0 in range(0, keep_idx.size, time_block):
            c1 = min(c0 + time_block, keep_idx.size)
            block = chunk[:, keep_idx[c0:c1]].T
            base[c0:c1, col_start:col_start + n_chunk] = block
        col_start += n_chunk

    base.flush()
    del base
    return str(out_path), base_shape


def assemble_dataset():
    if NEURAL_DIR.exists():
        shutil.rmtree(NEURAL_DIR)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)

    session_plans, subjects, speed_quantiles, median_time_bin_ms = build_session_plans()
    subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_to_index[plan.subject] for plan in session_plans], dtype=np.int16
        ),
        "brain_regions": REGION_NAMES,
        "brain_region_idx": [],
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "output_values": [
            STIMULUS_CATEGORIES,
            ["no_lick", "lick"],
            ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"],
            ["q1", "q2", "q3", "q4"],
        ],
        "metadata": {
            "task_description": (
                "Two-photon calcium activity from mice traversing 4 m virtual corridors "
                "with naturalistic textures, aligned to corridor entry."
            ),
            "time_bin_size": float(median_time_bin_ms),
            "temporal_alignment_event": "corridor entry (trial start)",
            "off_start": 0.0,
            "off_end": None,
            "trial_filter": {
                "max_corridor_duration_s": TRIAL_DURATION_MAX_S,
                "max_frame_gap_s": MAX_FRAME_GAP_S,
                "min_frames_per_trial": MIN_FRAMES_PER_TRIAL,
            },
            "stimulus_categories": STIMULUS_CATEGORIES,
            "speed_bin_quantiles_cm_per_s": speed_quantiles.astype(float).tolist(),
            "session_info": [],
        },
    }

    for session_idx, plan in enumerate(session_plans):
        print(f"[{session_idx + 1}/{len(session_plans)}] Converting {plan.raw_key}", flush=True)
        beh = load_behavior(plan)
        memmap_path, base_shape = write_session_neural_memmap(plan, session_idx)

        nfr = plan.nframes_behavior
        ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
        pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
        speed = np.clip(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32), 0.0, None)
        sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
        trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
        is_rew = np.asarray(beh["isRew"], dtype=bool)

        neural_trials = []
        input_trials = []
        output_trials = []
        for tp in plan.trial_plans:
            tr = tp.trial_index
            idx = tp.frame_idx
            T = idx.size
            neural_trials.append(
                MappedTrialArray(memmap_path, base_shape, np.dtype(np.float16).str, tp.row_start, tp.row_stop)
            )

            t_frame = ft[idx]
            input_trial = np.vstack(
                [
                    ((sound_time[tr] - t_frame) * 24.0 * 3600.0).astype(np.float32),
                    np.full((T,), plan.day_of_training, dtype=np.float32),
                    ((t_frame - trial_start[tr]) * 24.0 * 3600.0).astype(np.float32),
                    np.full((T,), float(is_rew[tr]), dtype=np.float32),
                ]
            )

            lick = lick_series_for_trial(beh, tr, idx, nfr)
            pos_bins = position_series_for_trial(pos[idx])
            speed_bins = speed_series_for_trial(speed[idx], speed_quantiles)
            output_trial = np.vstack(
                [
                    stimulus_series_for_trial(beh, tr, T),
                    lick,
                    pos_bins,
                    speed_bins,
                ]
            ).astype(np.int16)

            input_trials.append(input_trial)
            output_trials.append(output_trial)

        data["neural"].append(neural_trials)
        data["input"].append(input_trials)
        data["output"].append(output_trials)
        data["brain_region_idx"].append(plan.region_idx.astype(np.int16))
        data["metadata"]["session_info"].append(
            {
                "session_key": plan.raw_key,
                "behavior_file": plan.behavior_file,
                "behavior_key": plan.behavior_key,
                "experiment_types": plan.experiment_types,
                "n_trials_kept": len(plan.trial_plans),
                "n_frames_kept": int(plan.total_frames_kept),
                "median_dt_s": float(plan.median_dt_s),
                "day_of_training": float(plan.day_of_training),
            }
        )

    return data


def main():
    data = assemble_dataset()
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dataset to {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
