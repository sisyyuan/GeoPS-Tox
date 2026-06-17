from rdkit import Chem
from rdkit.Chem import BRICS, Recap
import numpy as np
from ogb.utils import smiles2graph
from torch_geometric.data import Data
import torch
from functools import reduce
import warnings
from rdkit.Chem import AllChem
from collections import defaultdict
# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="not removing hydrogen atom without neighbors")
def get_substructure(mol = None, smile = None, decomp = 'brics'):
    assert mol is not None or smile is not None, \
        'need at least one info of mol'
    assert decomp in ['brics', 'recap'], 'invalid decomposition method'
    if mol is None:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smile}")
    if decomp == 'brics':
        try:
            substructures = BRICS.BRICSDecompose(mol)
        except Exception as e:
            raise ValueError(f"Error decomposing molecule using BRICS: {e}")
    else:
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        substructures = set(leaves.keys())
    return substructures

###key 将一个批次中的不同唯一子结构转换为全局图
def graph_from_substructure(smile,subs, return_mask=False, return_type='numpy',return_sub_idx=True):

    all_keys = reduce(
        lambda x, y: x.union(y.keys()),  # 合并键集合
        (d for sublist in subs for d in sublist),  # 展开所有字典
        set()  # 初始空集合
    )
    sub_struct_list = list(all_keys)### reduce(function, iterable, initializer)
    sub_to_idx = {x: idx for idx, x in enumerate(sub_struct_list)}
    mask = np.zeros([len(subs), len(sub_struct_list)], dtype=bool)## (512, 666)
    ### key smiles2graph
    sub_graph = [smiles2graph(x) for x in sub_struct_list]
    # for tag, toxic in enumerate(toxic_all):
    #     for i, mol in enumerate(tqdm(toxic['Canonical SMILES'])):
    #         mol = Chem.MolFromSmiles(mol)
    #         mol = Chem.AddHs(mol)
    #         # 3. 生成 3D 构象（嵌入3D坐标）
    #         status = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    #         if status != 0:
    #             # raise ValueError("Embedding 3D coordinates failed for molecule")
    #             print(i)
    #             continue
    #         mol = Chem.RemoveHs(mol)
    #         conf = mol.GetConformer()
    #         pos = conf.GetPositions()
    #         pos = torch.tensor(pos, dtype=torch.float)
    #         posc = pos - pos.mean(dim=0)
    for idx, (sub)in enumerate(subs):
        mask[idx][list(sub_to_idx[list(t.keys())[0]] for t in sub)] = True ### mask->(512, 666)在对应位置标记
    # for idx, sub in enumerate(sub_struct_list):
    #     mol = Chem.MolFromSmiles(smile[idx])
    #     mol = Chem.AddHs(mol)
    #     status = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    #     if status != 0:
    # #             # raise ValueError("Embedding 3D coordinates failed for molecule")
    #         print(idx)
    #         continue
    #     mol_noH = Chem.RemoveHs(mol)
    #     conf = mol.GetConformer()
    #     pos = conf.GetPositions()
    #     pos = torch.tensor(pos, dtype=torch.float)
    #     posc = pos - pos.mean(dim=0)
    #     #         # 3. 生成 3D 构象（嵌入3D坐标）
    #     #         status = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    #
    #     frag_mol = Chem.MolFromSmiles(sub)
    #     matches = mol_noH.GetSubstructMatches(frag_mol)
    #     if matches:
    #             print(f"\n子结构 {sub} 的原子索引: {matches[0]}")
    #             for atom_idx in matches[0]:
    #                 pos = conf.GetAtomPosition(atom_idx)
    #                 print(f"原子 {atom_idx}: (x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f})")
    smiles_coords = defaultdict(list)

    # 遍历所有分子和子结构
    for mol_subs in subs:
        for sub_dict in mol_subs:
            for smiles, coords in sub_dict.items():
                smiles_coords[smiles].append(coords)

    # 对每个SMILES的坐标进行平均
    averaged_coords = {}
    for smiles, coord_list in smiles_coords.items():
        # 假设坐标是numpy数组或可以求平均的格式
        avg_coord = np.mean(coord_list, axis=0)
        avg_coord=avg_coord - avg_coord.mean(axis=0)
        averaged_coords[smiles] = avg_coord


    edge_idxes, edge_feats, node_feats, lstnode, batch,poses,pos,z ,posc= [], [], [], 0, [],[],[],[],[]
    for smiles in sub_struct_list:
        if smiles in averaged_coords:
            # pos.append({smiles: averaged_coords[smiles]})
            poses.append(averaged_coords.get(smiles, None))
    for idx, (graph,pot,smiles) in enumerate(zip(sub_graph,poses,sub_struct_list)):
        edge_idxes.append(graph['edge_index'] + lstnode) ### 将子图边索引调整为全局边索引
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        pos.append(pot)
        posc1 = pot - pot.mean(axis=0)
        posc.append(posc1)
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)
        result = np.where(graph['node_feat'][:,0] == 118, 0, graph['node_feat'][:,0] + 1)
        # if result.size() !=pos.shape[0]:

        z.append(result)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),  # 注意：edge_index 通常是 long 类型！
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0),
        'pos': np.concatenate(pos, axis=0).astype(np.float32),
        'posc': np.concatenate(posc, axis=0).astype(np.float32),
        'z': np.concatenate(z, axis=0)
    }

    assert return_type in ['numpy', 'torch', 'pyg'], 'Invaild return type'
    if return_type in ['torch', 'pyg']:
        for k, v in result.items():
            result[k] = torch.from_numpy(v)

    result['num_nodes'] = lstnode
    if return_type == 'pyg':
        result = Data(**result)
        result.pos=result.pos.float()
        result.posc=result.posc.float()


        # result.
    if return_sub_idx:
        return result, mask, sub_to_idx
    elif return_mask:
        return result, mask
    else:
        return result