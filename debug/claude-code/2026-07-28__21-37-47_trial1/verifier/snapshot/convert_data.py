"""Generate tiny random data in the standardized pickle format for debug/testing."""

import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generate random debug data')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--datadir', type=str, default='./data',
                        help='Data directory (unused, for CLI compatibility)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', default=True)
    mode.add_argument('--sample', action='store_true')
    args = parser.parse_args()

    np.random.seed(42)

    nsessions = 4
    ntrials_per_session = 8
    T = 100
    nneurons = 50
    dinput = 4
    doutput = 6
    ncategories = [7, 5, 5, 2, 3, 2]

    input_names = ['time_from_trial_start', 'environment', 'trial_number', 'previous_reward_outcome']
    output_names = ['distance_to_reward_zone', 'position', 'speed', 'lick', 'reward_location', 'reward_outcome']

    all_neural, all_input, all_output = [], [], []
    all_region_idx, subject_idx = [], []

    for s in range(nsessions):
        neural_trials = [np.random.randn(nneurons, T).astype(np.float32) for _ in range(ntrials_per_session)]
        input_trials = [np.random.rand(dinput, T).astype(np.float32) for _ in range(ntrials_per_session)]
        output_trials = []
        for _ in range(ntrials_per_session):
            out = np.zeros((doutput, T), dtype=np.int64)
            for d in range(doutput):
                out[d] = np.random.randint(0, ncategories[d], size=T)
            output_trials.append(out)
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        all_region_idx.append(np.zeros(nneurons, dtype=np.int32))
        subject_idx.append(s % 2)

    output_values = [[f"cat_{i}" for i in range(nc)] for nc in ncategories]

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': ['subj_0', 'subj_1'],
        'subject_idx': np.array(subject_idx, dtype=np.int32),
        'brain_regions': ['CA1'],
        'brain_region_idx': all_region_idx,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {'time_bin_size': 64.48},
    }

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
