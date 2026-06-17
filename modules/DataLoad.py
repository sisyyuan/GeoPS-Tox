import pandas
import os
import pickle
from modules.Datasets import CustomMoleculeDataset
import random
from substructurepos import substruct_pos
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import pandas as pd

def pyg_moldataset(d_name):
    work_dir = os.path.abspath(os.path.dirname(__file__))
    from torch_geometric.loader import DataLoader
    data_dir = os.path.join(work_dir, '../dataset')
    print(data_dir)
    dataset = CustomMoleculeDataset(d_name,root = data_dir)   
    # dataset = PygGraphPropPredDataset(d_name, root = data_dir)
    data_name = d_name.replace('-', '_')
    dataset_dir = os.path.join(work_dir, '../dataset', data_name)
    original_data = pandas.read_csv(
        os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
        compression = 'gzip'
    )
    smiles = original_data.smiles
    return smiles, dataset


def pyg_molsubdataset(d_name,seed,preprocess_method = 'brics'):

    work_dir = os.path.abspath(os.path.dirname(__file__))
    # print(work_dir)
    data_dir = os.path.join(work_dir, '../dataset')
    # df = pd.read_csv(osp.join(self.raw_dir, f'{d_name}.csv'))
    data_name = d_name.replace('-', '_')
    dataset_dir = os.path.join(work_dir, '../dataset', data_name)
    original_data = pandas.read_csv(
        os.path.join(dataset_dir, 'mapping', 'mol.csv.gz'),
        compression = 'gzip'
    )
    smiles = original_data.smiles

   # 假设 work_dir 和 data_name 已经定义
    pre_name = os.path.join(work_dir, '../dataset/sub_preprocess/', data_name)
    # 检查 data_name 是否包含 "A_",
    if "A_" in data_name:
        pre_name = os.path.join(work_dir, '../dataset/sub_preprocess/', 'ZINC250')

    pre_file = os.path.join(pre_name, 'substructures.pkl') \
        if preprocess_method == 'brics' else \
        os.path.join(pre_name, 'substructures_recap.pkl')

    if not os.path.exists(pre_file):
        raise IOError('please run preprocess script for dataset')

    with open(pre_file, 'rb') as f:
        substructures = pickle.load(f)
        dataset = CustomMoleculeDataset(d_name, root=data_dir,fragments=substructures)
    file_path = os.path.join(dataset.processed_dir, 'data.pkl')
    if not os.path.exists(file_path):
        raise IOError(
            f"Missing processed fragment-coordinate file: {file_path}. "
            "Delete stale processed files and rerun training or preprocessing to regenerate it."
        )
    with open(file_path, "rb") as f:
        loaded_list = pickle.load(f)
        # for i,smile in enumerate(dataset.data.smiles):
    #     mol = Chem.MolFromSmiles(mol)
    #     mol = Chem.AddHs(mol)
    #     # 3. 生成 3D 构象（嵌入3D坐标）
    #     status = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    #     if status != 0:
    #         # raise ValueError("Embedding 3D coordinates failed for molecule")
    #         print(i)
    #         continue
    #     mol = Chem.RemoveHs(mol)
    #     conf = mol.GetConformer()
    #     pos = conf.GetPositions()
    #     pos = torch.tensor(pos, dtype=torch.float)
    #     posc = pos - pos.mean(dim=0)


        # mol = Chem.AddHs(Chem.MolFromSmiles(smile))
        # AllChem.EmbedMolecule(mol, randomSeed=42)
        # mol = Chem.RemoveHs(mol)  # 仅保留重原子

        # 2. BRICS 拆分子
        # fragments = list(BRICS.BRICSDecompose(mol))
        # print("BRICS 子结构:", fragments)

        # 3. 对每个子结构运行 MCS 匹配
        # for frag in fragments:
        #     print(f"\n子结构: {frag}")
        #     matches = get_match(mol, frag)
        #     if matches:
        #         print("MCS 匹配的原子索引:", matches)
        #         # 提取坐标
        #         conf = mol.GetConformer()
        #         for match in matches:
        #             coords = [conf.GetAtomPosition(idx) for idx in match]
        #             print("匹配原子坐标:", [(p.x, p.y, p.z) for p in coords])
        #     else:
        #         print("未找到匹配")



        print(1)

    return smiles, loaded_list, dataset
