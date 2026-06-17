import argparse
import os
import sys
from modules.DataLoad import pyg_moldataset
from tqdm import tqdm
import subprocess
import pickle
from multiprocessing import Pool
# import ansi
import pandas
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def process_smile(args):
    idx, smile, method, timeout = args
    try:
        result = subprocess.run([
            sys.executable, os.path.join(BASE_DIR, 'modules', 'getSubstructure.py'), '--smile',
            smile, '--method', method
        ], check=True, timeout=timeout, capture_output=True, text=True)
        clean_stdout =  result.stdout.replace("\n\x1b[0m", "")
        return (idx, smile, clean_stdout.strip())
    except subprocess.TimeoutExpired:
        return (idx, smile, f'{smile}\t{str(set([smile]))}')

def get_result_dir():
    work_dir = os.path.abspath(os.path.dirname(__file__))
    result_dir = os.path.join(work_dir, '../dataset/sub_preprocess')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessing for Dataset')
    parser.add_argument(
        '--dataset', default='toxic_all', type=str,
        help='the dataset to preprocess'
    )
    parser.add_argument(
        '--method', choices=['brics', 'recap'], default='brics',
        help='the method to decompose molecules'
    )
    parser.add_argument(
        '--timeout', default=300, type=int,
        help='maximal time to process a single molecule'
    )
    parser.add_argument(
        '--num_workers', default=10, type=int,
        help='number of worker processes to use'
    )

    args = parser.parse_args()
    print(args)

    result_dir = get_result_dir()
    data_name = args.dataset.replace('-', '_')
    if not os.path.exists(os.path.join(result_dir, data_name)):
        os.mkdir(os.path.join(result_dir, data_name))

    dataset_dir = os.path.join(BASE_DIR, 'dataset', data_name)
    original_data = pandas.read_csv(
        os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
        compression='gzip'
    )
    smiles = original_data.smiles
    file_name = 'substructures.pkl' if args.method == 'brics' \
        else 'substructures_recap.pkl'
    file_name = os.path.join(result_dir, data_name, file_name)

    with Pool(args.num_workers) as pool:
        tasks = [(idx, smile, args.method, args.timeout) for idx, smile in enumerate(smiles)]
        results = list(tqdm(pool.imap(process_smile, tasks), total=len(tasks)))

    escapes = []
    substruct_list = []

    with open(file_name, 'w') as f:
        for idx, smile, result in results:
            if 'TimeoutExpired' in result:
                escapes.append(idx)
            f.write(result + '\n')

    if len(escapes) > 0:
        print('[INFO] the following molecules are processed unsuccessfully:')
        [print(smiles[x]) for x in escapes]

    with open(file_name) as f:
        for line in f:
            if len(line) <= 1:
                continue
            line = line.strip().split('\t')
            assert len(line) == 2, f'Invalid Line {line}'
            assert type(eval(line[1])) == set, f'Invalid value1 {line[1]}'
            if len(eval(line[1])) == 0:
                print(
                    f'[INFO] empty substruct find for {line[0]},'
                    'consider itself as a substructure'
                )
                substruct_list.append(set(line[0]))
            else:
                substruct_list.append(eval(line[1]))

    with open(file_name, 'wb') as f:
        pickle.dump(substruct_list, f)
