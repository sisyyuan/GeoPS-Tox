# import pandas as pd
# import shutil, os
# import os.path as osp
# import torch
# import numpy as np
# from torch_geometric.data import InMemoryDataset
# from ogb.utils.url import decide_download, download_url, extract_zip
# from ogb.io.read_graph_pyg import read_graph_pyg
# from ogb.utils import smiles2graph
# import torch
# from torch_geometric.data import InMemoryDataset, Data
# from rdkit import Chem
# from ogb.utils import smiles2graph
# from tqdm import tqdm
# from rdkit.Chem.Scaffolds import MurckoScaffold
# from collections import defaultdict
# from sklearn.model_selection import train_test_split
# import random

# class CustomMoleculeDataset(InMemoryDataset):
#     def __init__(self, name, seed,root='dataset', transform=None, pre_transform=None, meta_dict=None):
#         self.name = name
#         self.dir_name = '_'.join(name.split('-'))
#         self.original_root = root
#         self.root = osp.join(root, self.dir_name)
#         self.meta_info = None
#         self.seed=seed
#         if 'ZINC' in self.name:
#             print(f"The dataset name {name} contains 'ZINK'. Assigning specific meta_dict.")
#             meta_dict = {
#                 'data type': 'mol',
#                 'num tasks': 1,
#                 'download_name': '',
#                 'eval metric': 'rocauc',
#                 'version': 1,
#                 'add_inverse_edge': 'True',
#                 'has_edge_attr': 'True',
#                 'binary': 'False',
#                 'url': '',
#                 'additional node files': 'None',
#                 'additional edge files': 'None',
#                 'split': 'scaffold',
#                 'task type': 'binary classification',
#                 'has_node_attr': 'True',
#                 'num classes': 2
#             }
        
#         if meta_dict is None:
#             print(osp.join(os.path.dirname(__file__)))
#             master = pd.read_csv('./dataset/master.csv', index_col=0, keep_default_na=False)
#             if self.name not in master:
#                 error_mssg = f'Invalid dataset name {self.name}.\n'
#                 error_mssg += 'Available datasets are as follows:\n'
#                 error_mssg += '\n'.join(master.keys())
#                 raise ValueError(error_mssg)
#             self.meta_info = master[self.name].to_dict()
#         else:
#             self.meta_info = meta_dict
        
#         self.num_tasks = int(self.meta_info['num tasks'])
#         self.eval_metric = self.meta_info['eval metric']
#         self.task_type = self.meta_info['task type']
#         self.__num_classes__ = int(self.meta_info['num classes'])
#         self.binary = self.meta_info['binary'] == 'True'
#         self.split_type = self.meta_info['split']
        
#         super(CustomMoleculeDataset, self).__init__(self.root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
    
#     @property
#     def raw_file_names(self):
#         return [f'{self.name}.csv']

#     @property
#     def processed_file_names(self):
#         return 'data.pt'

#     def download(self):
#         pass  # Implement download logic if needed

#     def process(self):
#         data_list = []

#         df = pd.read_csv(osp.join(self.raw_dir, f'{self.name}.csv'))
#         for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
#             smiles = row['smiles']
#             if 'ZINC' in self.name:
#                 y = None  # 无监督学习任务没有标签
#             else:
#                 y = row['label']
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 continue

#             graph = smiles2graph(smiles)
#             x = torch.tensor(graph['node_feat'], dtype=torch.long)
#             edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
#             edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
#             data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
#             data.smiles = smiles
#             data_list.append(data)

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

#     def get_idx_split(self, split_type=None):
#         if split_type is None:
#             split_type = self.meta_info['split']
        
#         path = osp.join(self.root, 'split', split_type)

#         # short-cut if split_dict.pt exists
#         if os.path.isfile(os.path.join(path, 'split_dict.pt')):
#             return torch.load(os.path.join(path, 'split_dict.pt'))

#         if all(os.path.isfile(osp.join(path, f'{split}.csv.gz')) for split in ['train', 'valid', 'test']):
#             train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
#             valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
#             test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]
#         else:
#             # 自动生成分割
#             print("No pre-split files found. Generating new splits...")
#             train_idx, valid_idx, test_idx = self._generate_splits()

#             # 保存分割
#             os.makedirs(path, exist_ok=True)
#             pd.DataFrame(train_idx).to_csv(osp.join(path, 'train.csv.gz'), index=False, header=False, compression='gzip')
#             pd.DataFrame(valid_idx).to_csv(osp.join(path, 'valid.csv.gz'), index=False, header=False, compression='gzip')
#             pd.DataFrame(test_idx).to_csv(osp.join(path, 'test.csv.gz'), index=False, header=False, compression='gzip')

#         return {'train': torch.tensor(train_idx, dtype=torch.long), 
#                 'valid': torch.tensor(valid_idx, dtype=torch.long), 
#                 'test': torch.tensor(test_idx, dtype=torch.long)}

#     def _generate_splits(self, scaffold_split=True, valid_size=0.1, test_size=0.1, balanced=True):
#         if scaffold_split:
#             print("Using scaffold split...")
#             print(self.seed)
#             all_smiles = [data.smiles for data in self]
#             scaffolds = defaultdict(list)

#             for i, smiles in enumerate(all_smiles):
#                 mol = Chem.MolFromSmiles(smiles)
#                 scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
#                 scaffolds[scaffold].append(i)

#             scaffold_sets = list(scaffolds.values())

#             if balanced:
#                 big_scaffolds, small_scaffolds = [], []
#                 for scaffold_set in scaffold_sets:
#                     if len(scaffold_set) > valid_size * len(all_smiles) / 2 or len(scaffold_set) > test_size * len(all_smiles) / 2:
#                         big_scaffolds.append(scaffold_set)
#                     else:
#                         small_scaffolds.append(scaffold_set)
#                 random.seed(self.seed)
#                 random.shuffle(big_scaffolds)
#                 random.shuffle(small_scaffolds)
#                 scaffold_sets = big_scaffolds + small_scaffolds
#             else:
#                 random.shuffle(scaffold_sets)
#             train_idx, valid_idx, test_idx = [], [], []
#             for scaffold_set in scaffold_sets:
#                 if len(train_idx) < (1 - valid_size - test_size) * len(all_smiles):
#                     train_idx += scaffold_set
#                 elif len(valid_idx) < valid_size * len(all_smiles):
#                     valid_idx += scaffold_set
#                 else:
#                     test_idx += scaffold_set

#             return train_idx, valid_idx, test_idx
#         else:
#             print("Using random split...")
#             indices = list(range(len(self)))
#             train_idx, temp_idx = train_test_split(indices, test_size=valid_size + test_size, random_state=42)
#             valid_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (valid_size + test_size), random_state=42)

#             return train_idx, valid_idx, test_idx

#     @property
#     def num_classes(self):
#         return self.__num_classes__
#################################
import pickle
import pandas as pd
import shutil, os
import pandas

import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg
from ogb.utils import smiles2graph
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from ogb.utils import smiles2graph
from tqdm import tqdm
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import train_test_split
import random
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import AllChem
from substructurepos import get_match
from sklearn.utils import shuffle
from ogb.utils import smiles2graph


# 引入新的 scaffold_split 和 generate_scaffolds 方法
def _generate_scaffold(smiles, include_chirality=False, is_standard=False):
    if is_standard:
        scaffold = MurckoScaffoldSmiles(smiles=smiles, includeChirality=True)
    else:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffolds(dataset, log_every_n=1000, sort=True, is_standard=False):
    scaffolds = {}
    data_len = len(dataset)

    for ind, smiles in enumerate(dataset.smiles):
        scaffold = _generate_scaffold(smiles, is_standard=is_standard)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    if sort:
        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), 
                key=lambda x: (len(x[1]), x[1][0]), 
                reverse=True
            )
        ]
    else:
        scaffold_sets = [value for key, value in scaffolds.items()]

    return scaffold_sets

def scaffold_split(dataset, r_val, r_test, log_every_n=1000, is_standard=False):
    r_train = 1.0 - r_val - r_test
    scaffold_sets = generate_scaffolds(dataset, log_every_n, is_standard=is_standard)

    train_cutoff = r_train * len(dataset)
    valid_cutoff = (r_train + r_val) * len(dataset)
    train_inds = []
    valid_inds = []
    test_inds = []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

# 修改后的 CustomMoleculeDataset
class CustomMoleculeDataset(InMemoryDataset):
    def __init__(self, name,root='dataset', transform=None, pre_transform=None, meta_dict=None,fragments=None):
        self.fragment=fragments



        self.name = name
        self.dir_name = '_'.join(name.split('-'))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.meta_info = None
        if 'ZINC' in self.name:
            print(f"The dataset name {name} contains 'ZINK'. Assigning specific meta_dict.")
            meta_dict = {
                'data type': 'mol',
                'num tasks': 1,
                'download_name': '',
                'eval metric': 'rocauc',
                'version': 1,
                'add_inverse_edge': 'True',
                'has_edge_attr': 'True',
                'binary': 'False',
                'url': '',
                'additional node files': 'None',
                'additional edge files': 'None',
                'split': 'scaffold',
                'task type': 'binary classification',
                'has_node_attr': 'True',
                'num classes': 2
            }
        
        if meta_dict is None:
            print(osp.join(os.path.dirname(__file__)))

            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "../dataset/master.csv")
            master = pd.read_csv(file_path, index_col=0, keep_default_na=False)
            # if self.name not in master:
            #     # error_mssg = f'Invalid dataset name {self.name}.\n'
            #     # error_mssg += 'Available datasets are as follows:\n'
            #     # error_mssg += '\n'.join(master.keys())
            #     # raise ValueError(error_mssg)
            #     self.name='BACE'
            self.meta_info = master[self.name].to_dict()
        else:
            self.meta_info = meta_dict
        
        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'
        self.split_type = self.meta_info['split']
        
        super(CustomMoleculeDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    
    @property
    def raw_file_names(self):
        return [f'{self.name}.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  # Implement download logic if needed

    def process(self):
        data_list = []
        failed_indices = []
        failed_indices = []
        toxic0 = pd.read_csv(osp.join(self.raw_dir, f'{"Cardiotoxicity_Cardiotoxicity-5"}.csv'))
        toxic1 = pd.read_csv(osp.join(self.raw_dir, f'{"Hepatotoxicity_Hepatotoxicity"}.csv'))
        toxic2 = pd.read_csv(osp.join(self.raw_dir, f'{"Irritation and Corrosion_Eye Irritation"}.csv'))
        toxic3 = pd.read_csv(osp.join(self.raw_dir, f'{"Respiratory Toxicity_Respiratory Toxicity"}.csv'))

        toxic_all = []
        toxic_all.append(toxic0)
        toxic_all.append(toxic1)
        toxic_all.append(toxic2)
        toxic_all.append(toxic3)
        dataset_dir = self.root
        original_data = pandas.read_csv(
            os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
            compression='gzip'
        )
        df = original_data
        data_list = []
        i=-1

        # df = pd.read_csv(osp.join(self.raw_dir, f'{"toxic"}.csv'))
        # df = pd.read_csv(osp.join(self.raw_dir, f'{self.name}.csv'))
        fragments = []
        for tag, toxic in enumerate(toxic_all):
            for _,(_, row) in tqdm(enumerate(toxic.iterrows()), total=len(toxic), desc="Processing molecules"):
                i = i + 1

                smiles = row['smiles']
                if 'ZINC' in self.name:
                    y = None  # 无监督学习任务没有标签
                else:
                    y = row['label']
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                graph = smiles2graph(smiles)
                x = torch.tensor(graph['node_feat'], dtype=torch.long)
                edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
                edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
                # mol = Chem.MolFromSmiles(mol)
                mol = Chem.AddHs(mol)
                # 3. 生成 3D 构象（嵌入3D坐标）
                status = AllChem.EmbedMolecule(mol, randomSeed=42)
                mol = Chem.RemoveHs(mol)
                if status != 0:
                    print(i)
                    failed_indices.append(i)

                    # raise ValueError("Embedding 3D coordinates failed for molecule")
                    continue
                # if len(self.fragment[i])==1:
                #     continue
                fragment=[]
                for frag in self.fragment[i]:
                    # print(f"\n子结构: {frag}")
                    matches = get_match(mol, frag)

                    if matches:
                    # print("MCS 匹配的原子索引:", matches)
                        # 提取坐标
                        conf = mol.GetConformer()

                        coords = [conf.GetAtomPosition(idx) for idx in matches[0]]
                        atom =[mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in matches[0]]
                        frag_coords = {frag: [(p.x, p.y, p.z) for p in coords]}
                        fragment.append(frag_coords)
                    elif len(self.fragment[i])==1:
                        conf = mol.GetConformer()
                        all_coords=[]
                        for d in range(mol.GetNumAtoms()):
                            pos = conf.GetAtomPosition(d)
                            all_coords.append((pos.x, pos.y, pos.z))
                            frag_coords={frag: all_coords}

                        # 然后创建 {frag: 所有坐标} 的字典结构
                        fragment.append(frag_coords)
                    elif matches:
                        # print("MCS 匹配的原子索引:", matches)
                        # 提取坐标
                        conf = mol.GetConformer()


                        coords = [conf.GetAtomPosition(idx) for idx in matches[0]]
                        frag_coords={frag: [(p.x, p.y, p.z) for p in coords]}
                        fragment.append(frag_coords)


                    else:
                        print("未匹配: {}".format(smiles))
                        continue
                    sub_graph=smiles2graph(frag)
                    if(sub_graph['num_nodes']!=len(list(frag_coords.values())[0])):
                        print("长度不统一: {}".format(smiles))
                fragments.append(fragment)

                # mol = Chem.RemoveHs(mol)
                conf = mol.GetConformer()
                pos = conf.GetPositions()
                pos = torch.tensor(pos, dtype=torch.float)
                posc = pos - pos.mean(dim=0)

                atomic_number = []

                for atom in mol.GetAtoms():
                    atomic_number.append(atom.GetAtomicNum())

                z = torch.tensor(atomic_number, dtype=torch.long)

                data = Data(z=z,pos=pos,posc=posc,x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,tag=torch.tensor(tag, dtype=torch.float))
                data.smiles = smiles
                data_list.append(data)

        df_clean = df.drop(failed_indices).reset_index(drop=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        with open(osp.join(self.processed_dir, 'data.pkl'), "wb") as f:
            pickle.dump(fragments, f)
        # Save the cleaned DataFrame to a new CSV file
        df_clean.to_csv(osp.join(self.raw_dir, 'toxic_new.csv'), index=False)

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']
        
        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        if all(os.path.isfile(osp.join(path, f'{split}.csv.gz')) for split in ['train', 'valid', 'test']):
            train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
            valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
            test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]
        else:
            # 自动生成分割
            print("No pre-split files found. Generating new splits...")
            train_idx, valid_idx, test_idx = scaffold_split(self, r_val=0.1, r_test=0.1)

            # 保存分割
            os.makedirs(path, exist_ok=True)
            pd.DataFrame(train_idx).to_csv(osp.join(path, 'train.csv.gz'), index=False, header=False, compression='gzip')
            pd.DataFrame(valid_idx).to_csv(osp.join(path, 'valid.csv.gz'), index=False, header=False, compression='gzip')
            pd.DataFrame(test_idx).to_csv(osp.join(path, 'test.csv.gz'), index=False, header=False, compression='gzip')

        return {'train': torch.tensor(train_idx, dtype=torch.long), 
                'valid': torch.tensor(valid_idx, dtype=torch.long), 
                'test': torch.tensor(test_idx, dtype=torch.long)}
    
    def get_random_split(self, split_ratio=0.9):
        """
        Randomly splits the dataset into training and test sets with a given ratio.
        
        :param split_ratio: Proportion of data to be used for training. Default is 0.9.
        :return: A dictionary containing 'train' and 'test' splits.
        """
        # Load the dataset
        df = pd.read_csv(osp.join(self.raw_dir, 'toxic_new.csv'))
        
        # If it's a supervised task, split by labels
        # if 'ZINC' not in self.name:
        #     pdata = df[df['label'] == 1]  # Positive samples
        #     ndata = df[df['label'] == 0]  # Negative samples
        #
        #     # Randomly select 90% of the data for training
        #     ptrain_idx = pdata.sample(frac=split_ratio, random_state=42).index
        #     ntrain_idx = ndata.sample(frac=split_ratio, random_state=42).index
        #
        #     # Create train and test sets
        #     train_idx = list(ptrain_idx) + list(ntrain_idx)
        #     test_idx = df.index.difference(train_idx)
        # else:
        #     # If it's an unsupervised task, just randomly split the entire dataset
        #     train_idx = df.sample(frac=split_ratio, random_state=42).index
        #     test_idx = df.index.difference(train_idx)
        data_size=len(df)
        ids = shuffle(range(data_size))
        train_size = int(data_size * 4 / 5)  # 定义 train_size，取前80%的数据作为训练集
        train_idx = torch.tensor(ids[:train_size])  # 训练集索引
        test_idx = torch.tensor(ids[train_size:])  # 测试集索引
        # Return the split indices as tensors
        return {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long)
        }

    @property
    def num_classes(self):
        return self.__num_classes__
