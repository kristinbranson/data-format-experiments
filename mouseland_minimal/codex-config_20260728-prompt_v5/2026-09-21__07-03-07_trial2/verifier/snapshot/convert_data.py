import gc
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime

import numpy as np


DATA_ROOT = "/app/data"
BEH_ROOT = os.path.join(DATA_ROOT, "beh")
SPK_ROOT = os.path.join(DATA_ROOT, "spk")
RETINO_ROOT = os.path.join(DATA_ROOT, "retinotopy")
OUT_PATH = "/app/converted_data.pkl"

SECONDS_PER_DAY = 24 * 3600
MS_PER_DAY = 24 * 3600 * 1000

# The paper groups visual areas into these four broad retinotopic regions.
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV"]
AREA_CODES = {
    "V1": np.array([8], dtype=int),
    "mHV": np.array([0, 1, 2, 9], dtype=int),
    "lHV": np.array([5, 6], dtype=int),
    "aHV": np.array([3, 4], dtype=int),
}

# To keep the exported dataset trainable, cap the number of retained neurons per
# broad visual area after applying the paper-consistent running/corridor filter.
MAX_NEURONS_PER_AREA = 128


def parse_base_session_key(key: str) -> str:
    parts = key.split("_")
    if parts[-1].startswith("swap"):
        return "_".join(parts[:-1])
    return key


def parse_subject_and_date(base_key: str) -> tuple[str, datetime]:
    parts = base_key.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    return subject, date


def wall_to_category(name: str) -> str:
    match = re.match(r"([A-Za-z]+)", str(name))
    if match is None:
        raise ValueError(f"Could not parse stimulus family from wall name {name!r}")
    category = match.group(1).lower()
    if category not in {"circle", "leaf", "rock", "wood"}:
        raise ValueError(f"Unexpected stimulus category {category!r} from wall name {name!r}")
    return category


def merge_behavior_records() -> dict[str, dict]:
    merged = {}
    beh_files = sorted(
        fname for fname in os.listdir(BEH_ROOT) if fname.startswith("Beh_") and fname.endswith(".npy")
    )
    for fname in beh_files:
        print(f"Loading behavior file {fname}", flush=True)
        beh = np.load(os.path.join(BEH_ROOT, fname), allow_pickle=True).item()
        for key, record in beh.items():
            base_key = parse_base_session_key(key)
            if base_key not in merged:
                merged[base_key] = {k: v for k, v in record.items()}
                continue

            # The duplicated session entries differ only in stimulus annotation
            # fields used for distinct figure analyses. Merge the stimulus IDs so
            # each physical recording is represented once with the richest mapping.
            stim_old = np.asarray(merged[base_key]["stim_id"], dtype=float)
            stim_new = np.asarray(record["stim_id"], dtype=float)
            if stim_old.shape == stim_new.shape:
                fill = np.isnan(stim_old) & ~np.isnan(stim_new)
                stim_old[fill] = stim_new[fill]
                merged[base_key]["stim_id"] = stim_old
    return merged


def collect_subject_info(session_keys: list[str]) -> tuple[list[str], dict[str, int], dict[str, float]]:
    per_subject_dates = defaultdict(list)
    for key in session_keys:
        subject, date = parse_subject_and_date(key)
        per_subject_dates[subject].append(date)

    subjects = sorted(per_subject_dates)
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    first_dates = {subject: min(dates) for subject, dates in per_subject_dates.items()}

    day_since_first = {}
    for key in session_keys:
        subject, date = parse_subject_and_date(key)
        day_since_first[key] = float((date - first_dates[subject]).days)

    return subjects, subject_to_idx, day_since_first


def compute_time_bin_size_ms(behaviors: dict[str, dict]) -> float:
    dts = []
    for beh in behaviors.values():
        ft = np.asarray(beh["ft"])
        if ft.size > 1:
            dts.append(np.median(np.diff(ft)) * MS_PER_DAY)
    return float(np.median(dts))


def compute_speed_edges(behaviors: dict[str, dict]) -> np.ndarray:
    speeds = []
    for beh in behaviors.values():
        mask = np.asarray(beh["ft_CorrSpc"], dtype=bool) & (np.asarray(beh["ft_move"]) > 0)
        if np.any(mask):
            speeds.append(np.asarray(beh["ft_RunSpeed"])[mask])
    all_speeds = np.concatenate(speeds)
    return np.quantile(all_speeds, [0.25, 0.5, 0.75]).astype(np.float32)


def describe_speed_bins(edges: np.ndarray) -> list[str]:
    return [
        f"<= {edges[0]:.2f} cm/s",
        f"{edges[0]:.2f}-{edges[1]:.2f} cm/s",
        f"{edges[1]:.2f}-{edges[2]:.2f} cm/s",
        f"> {edges[2]:.2f} cm/s",
    ]


def select_neurons_for_session(spk_planes: list[np.ndarray], iarea: np.ndarray, beh: dict) -> tuple[np.ndarray, np.ndarray]:
    area_candidates = {region: {"plane": [], "row": [], "score": []} for region in BRAIN_REGIONS}
    offset = 0

    for plane_idx, plane in enumerate(spk_planes):
        nfr = plane.shape[1]
        corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
        gray_mask = np.asarray(beh["ft_GraySpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
        plane_areas = iarea[offset : offset + plane.shape[0]]

        if not np.any(corr_mask) or not np.any(gray_mask):
            offset += plane.shape[0]
            continue

        corr_mean = plane[:, corr_mask].mean(axis=1)
        gray_mean = plane[:, gray_mask].mean(axis=1)
        corr_var = plane[:, corr_mask].var(axis=1)
        responsive = corr_mean > gray_mean

        for region in BRAIN_REGIONS:
            region_mask = np.isin(plane_areas, AREA_CODES[region])
            keep = responsive & region_mask
            if not np.any(keep):
                continue
            rows = np.flatnonzero(keep).astype(np.int32)
            area_candidates[region]["plane"].append(np.full(rows.shape, plane_idx, dtype=np.int16))
            area_candidates[region]["row"].append(rows)
            area_candidates[region]["score"].append(corr_var[rows].astype(np.float32))

        offset += plane.shape[0]

    selected_rows_by_plane = defaultdict(list)
    selected_region_idx_by_plane = defaultdict(list)
    region_to_idx = {region: idx for idx, region in enumerate(BRAIN_REGIONS)}

    for region in BRAIN_REGIONS:
        candidate = area_candidates[region]
        if not candidate["row"]:
            continue
        plane_arr = np.concatenate(candidate["plane"])
        row_arr = np.concatenate(candidate["row"])
        score_arr = np.concatenate(candidate["score"])

        # Highest variance first, then plane/row for deterministic tiebreaks.
        order = np.lexsort((row_arr, plane_arr, -score_arr))
        top = order[:MAX_NEURONS_PER_AREA]
        chosen = sorted(zip(plane_arr[top], row_arr[top]), key=lambda x: (x[0], x[1]))
        for plane_idx, row_idx in chosen:
            selected_rows_by_plane[int(plane_idx)].append(int(row_idx))
            selected_region_idx_by_plane[int(plane_idx)].append(region_to_idx[region])

    selected_chunks = []
    selected_regions = []
    for plane_idx in range(len(spk_planes)):
        rows = selected_rows_by_plane.get(plane_idx, [])
        if not rows:
            continue
        selected_chunks.append(spk_planes[plane_idx][rows])
        selected_regions.extend(selected_region_idx_by_plane[plane_idx])

    if not selected_chunks:
        raise RuntimeError("No neurons survived the selection criteria for this session.")

    selected_spk = np.concatenate(selected_chunks, axis=0)
    region_idx = np.asarray(selected_regions, dtype=np.int16)
    return selected_spk, region_idx


def build_trial_data(
    selected_spk: np.ndarray,
    beh: dict,
    day_value: float,
    category_to_idx: dict[str, int],
    speed_edges: np.ndarray,
    dt_seconds: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    nfr = selected_spk.shape[1]
    trial_idx = np.asarray(beh["ft_trInd"][:nfr])
    move_corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
    ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
    frame_numbers = np.arange(nfr, dtype=np.float32)

    lick_binary = np.zeros(nfr, dtype=np.int8)
    lick_fr = np.asarray(beh["LickFr"], dtype=int)
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
    if lick_fr.size:
        lick_binary[np.unique(lick_fr)] = 1

    neural_trials = []
    input_trials = []
    output_trials = []

    for tr in range(int(beh["ntrials"])):
        mask = (trial_idx == tr) & move_corr_mask
        if not np.any(mask):
            continue

        selected_frames = frame_numbers[mask]
        time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
        time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
        reward_available = np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
        day_arr = np.full(mask.sum(), day_value, dtype=np.float32)

        wall_name = str(beh["WallName"][tr])
        stim_category = category_to_idx[wall_to_category(wall_name)]
        visual_category = np.full(mask.sum(), stim_category, dtype=np.int16)

        lick = lick_binary[mask].astype(np.int16)
        pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
        speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
        speed_bin = np.clip(speed_bin, 0, 3)

        neural_trials.append(selected_spk[:, mask].astype(np.float16))
        input_trials.append(
            np.vstack([time_to_sound, day_arr, time_since_start, reward_available]).astype(np.float32)
        )
        output_trials.append(np.vstack([visual_category, lick, pos_bin, speed_bin]).astype(np.int16))

    return neural_trials, input_trials, output_trials


def main():
    print("Merging behavior records...", flush=True)
    behaviors = merge_behavior_records()
    session_keys = sorted(behaviors)
    print("Computing subject, timing, and speed metadata...", flush=True)
    subjects, subject_to_idx, day_since_first = collect_subject_info(session_keys)
    speed_edges = compute_speed_edges(behaviors)
    time_bin_size = compute_time_bin_size_ms(behaviors)

    all_categories = sorted(
        {
            wall_to_category(wall_name)
            for beh in behaviors.values()
            for wall_name in np.asarray(beh["WallName"]).tolist()
        }
    )
    category_to_idx = {category: idx for idx, category in enumerate(all_categories)}

    data = {
        "neural": [],
        "input": [],
        "output": [],
        "subjects": subjects,
        "subject_idx": [],
        "brain_regions": BRAIN_REGIONS,
        "brain_region_idx": [],
        "input_names": [
            "time_to_sound_cue_s",
            "day_of_training",
            "time_since_trial_start_s",
            "reward_available",
        ],
        "output_names": [
            "visual_stimulus_category",
            "licking",
            "position_bin",
            "running_speed_bin",
        ],
        "output_values": [
            all_categories,
            ["no_lick", "lick"],
            ["0-1m", "1-2m", "2-3m", "3-4m"],
            describe_speed_bins(speed_edges),
        ],
        "metadata": {
            "task_description": (
                "Head-fixed mice ran through 4 m visual virtual-reality corridors. "
                "Neural inputs are Suite2p deconvolved calcium traces restricted to running "
                "frames inside the corridor. Decoder outputs are stimulus family, licking, "
                "1 m position bins, and running-speed quartiles."
            ),
            "time_bin_size": time_bin_size,
            "temporal_alignment_event": "corridor entry (trial start)",
            "off_start": 0.0,
            "off_end": None,
            "source_paper": "Unsupervised pretraining in biological neural networks",
            "neural_processing": (
                "Used deconvolved Suite2p traces. Kept only frames where the animal was "
                "running and located within the 4 m corridor, matching the paper's running-only analyses."
            ),
            "neuron_selection": (
                f"Excluded retinotopy-unassigned neurons. Within V1, mHV, lHV, and aHV, kept "
                f"running-corridor-responsive neurons with mean activity greater than running gray-space "
                f"activity, ranked by corridor-frame variance, capped at {MAX_NEURONS_PER_AREA} per area per session."
            ),
            "day_of_training_definition": (
                "Calendar days since the first imaging session for each subject, used because "
                "a consistent explicit training-day count was not available for every recording."
            ),
            "running_speed_bin_edges_cm_s": speed_edges.tolist(),
            "session_keys": session_keys,
        },
    }

    print(f"Merged {len(session_keys)} unique recordings from behavior files.", flush=True)
    print(f"Stimulus categories: {all_categories}", flush=True)
    print(f"Running-speed quartile edges (cm/s): {speed_edges.tolist()}", flush=True)
    print(f"Nominal time bin size (ms): {time_bin_size:.3f}", flush=True)

    for session_idx, base_key in enumerate(session_keys):
        subject, _ = parse_subject_and_date(base_key)
        print(f"[{session_idx + 1:02d}/{len(session_keys)}] Processing {base_key}", flush=True)

        beh = behaviors[base_key]
        spk_path = os.path.join(SPK_ROOT, f"{base_key}_neural_data.npy")
        retino_path = os.path.join(RETINO_ROOT, f"{subject}_{base_key.split('_', 1)[1][:10]}_trans.npz")

        spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
        iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
        nneu_total = sum(arr.shape[0] for arr in spk_planes)
        if nneu_total != iarea.shape[0]:
            raise RuntimeError(
                f"Neuron count mismatch for {base_key}: spikes={nneu_total}, retinotopy={iarea.shape[0]}"
            )

        selected_spk, region_idx = select_neurons_for_session(spk_planes, iarea, beh)
        dt_seconds = float(np.median(np.diff(np.asarray(beh["ft"], dtype=np.float64))) * SECONDS_PER_DAY)
        neural_trials, input_trials, output_trials = build_trial_data(
            selected_spk=selected_spk,
            beh=beh,
            day_value=day_since_first[base_key],
            category_to_idx=category_to_idx,
            speed_edges=speed_edges,
            dt_seconds=dt_seconds,
        )

        if len(neural_trials) < 2:
            raise RuntimeError(f"Session {base_key} has fewer than two valid trials after filtering.")

        data["neural"].append(neural_trials)
        data["input"].append(input_trials)
        data["output"].append(output_trials)
        data["subject_idx"].append(subject_to_idx[subject])
        data["brain_region_idx"].append(region_idx)

        print(
            f"  kept {region_idx.size} neurons, {len(neural_trials)} trials, "
            f"median T={np.median([trial.shape[1] for trial in neural_trials]):.1f}",
            flush=True,
        )

        del spk_planes, iarea, selected_spk, neural_trials, input_trials, output_trials
        gc.collect()

    data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int16)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved converted dataset to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
