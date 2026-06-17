from rdkit import Chem
from rdkit.Chem import BRICS, rdFMCS
from rdkit.Chem import AllChem
import re


# def clean_brics_smiles(smiles):
#     """
#     处理 BRICS 子结构的 SMILES 字符串：
#     1. 将 [n*] 替换为 *
#     2. 对于没有连接点标记的末端原子添加 ;D1 限制
#     """
#     # 首先替换所有的 [n*] 为 *
#     cleaned = re.sub(r'\[\d+\*\]', '*', smiles)
#     return cleaned
#
#     # 使用RDKit解析SMILES以获得更精确的原子信息
#     mol = Chem.MolFromSmiles(cleaned)
#     if mol is None:
#
#         return cleaned  # 如果无法解析，返回原始处理结果
#
#     # 获取所有原子
#     atoms = [atom for atom in mol.GetAtoms()]
#
#     # 找出所有连接点(*)的原子索引
#     link_points = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 0]
#
#     # 找出可能是末端的原子（非连接点且度=1）
#     terminal_atoms = []
#     for atom in atoms:
#         if atom.GetIdx() not in link_points and atom.GetDegree() == 1:
#             # 检查它是否通过单键连接
#             bond = atom.GetBonds()[0]
#             if bond.GetBondType() == Chem.BondType.SINGLE:
#                 terminal_atoms.append(atom.GetIdx())
#     # 如果有末端原子需要处理
#     if terminal_atoms:
#         cleaned1 = ''
#
#         # 将SMILES转换为原子列表
#         tokens = []
#         for atom in mol.GetAtoms():
#             symbol = atom.GetSymbol()
#             if atom.GetIdx() in terminal_atoms:
#                 tokens.append(f'[{symbol};D1]')
#             else:
#                 tokens.append(symbol)
#
#         # 重新生成SMILES
#         new_mol = Chem.RWMol(mol)
#
#         for i, atom in enumerate(new_mol.GetAtoms()):
#             atom.SetProp('_displayLabel', tokens[i])
#             cleaned=cleaned1=cleaned1+tokens[i]
#             cleaned2 = Chem.MolToSmiles(new_mol)
#             if 'c1ccccc1' in cleaned2:
#                 return cleaned2
#
#
#     return cleaned


import re

from ogb.utils import smiles2graph

def clean_brics_smiles(smiles):
    """
    处理 BRICS 子结构的 SMILES 字符串：
    1. 将 [n*] 形式的连接点替换为 *
    2. 确保 * 前后如果是字母（非数字、非键符号等），则插入 ~
    3. 正确处理 * 前的化学键（=、#、:、/、\等）
    4. 如果 * 出现在括号 () 内，则转换为 ~*
    """
    # 1. 替换所有的 [n*] 为 *
    cleaned = re.sub(r'\[\d+\*\]', '*', smiles)

    # 2. 处理 * 后接字母的情况（*=C → *=~C，*C → *~C）
    cleaned = re.sub(r'\*([a-zA-Z])', r'*~\1', cleaned)

    # 3. 处理 * 前接字母的情况（C=* → C=*，C* → C~*）
    # 匹配字母 + 可选键符号（=、#、:、/、\等） + *
    cleaned = re.sub(r'([a-zA-Z])([=#:/\\]?)\*', r'\1\2~*', cleaned)

    # 4. 处理括号内的 *（(* → (~*）
    cleaned = re.sub(r'\((\s*)\*', r'(\1~*', cleaned)

    # 5. 避免重复插入 ~（如 C~*~C → C~*~C，而不是 C~~*~~C）
    cleaned = re.sub(r'~\~+', '~', cleaned)

    return cleaned
def get_match(mol, frag_smiles):

    # 1. 预处理子结构（替换通配符）
    frag_smiles_cleaned = clean_brics_smiles(frag_smiles)
    # frag_mol = Chem.MolFromSmiles(frag_smiles_cleaned)
    # if not frag_mol:
    #     print(f"警告: 子结构 {frag_smiles_cleaned} 无效")
    #     return None

    # 2. 转换为 SMARTS（支持通配符 *）
    # frag_smarts = Chem.MolToSmarts(frag_mol)
    #
    # # 3. 运行 MCS 匹配
    # mcs_result = rdFMCS.FindMCS(
    #     [mol, Chem.MolFromSmarts(frag_smarts)],
    #     timeout=10,  # 超时时间（秒）
    #     completeRingsOnly=True  # 确保环完整性
    # )
    #
    # if not mcs_result.canceled:
    #     # 提取匹配的原子索引
    #     mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    substructure = Chem.MolFromSmarts(frag_smiles_cleaned)
    matches = mol.GetSubstructMatches(substructure)
    # print(f"找到 {len(matches)} 个匹配:")
    return matches
    # else:
    #     print("MCS 匹配超时或失败")
    #     return None
def substruct_pos(smile,fragments):
    # smiles = "CCOC(=O)C1CCCN1C"  # 示例分子
    mol = Chem.AddHs(Chem.MolFromSmiles(smile))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)  # 仅保留重原子

    # 2. BRICS 拆分子
    # fragments = list(BRICS.BRICSDecompose(mol))
    # print("BRICS 子结构:", fragments)

    # 3. 对每个子结构运行 MCS 匹配
    for frag in fragments:
        print(f"\n子结构: {frag}")
        matches = get_match(mol, frag)
        if matches:
            print("MCS 匹配的原子索引:", matches)
            # 提取坐标
            conf = mol.GetConformer()
            for match in matches:
                coords = [conf.GetAtomPosition(idx) for idx in match]
                print("匹配原子坐标:", [(p.x, p.y, p.z) for p in coords])
        else:
            print("未找到匹配")

# 示例调用
if __name__ == "__main__":
    # 1. 准备分子（生成 3D 结构）
    smiles = "Nc1nnc(-c2cccc(Cl)c2Cl)c(N)n1"  # 示例分子
    print(smiles)
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)  # 仅保留重原子

    # 2. BRICS 拆分子
    fragments = list(BRICS.BRICSDecompose(mol))
    print("BRICS 子结构:", fragments)

    # 3. 对每个子结构运行 MCS 匹配
    for frag in fragments:
        # frag="[5*]N1CCC([7*])CC1"
        print("MCS 匹配的原子索引:", frag)

        # frag="[7*]C1C(=O)Nc2ccc(F)cc21"
        # print(f"\n子结构: {frag}")
        matches = get_match(mol, frag)
        # matches = get_match(mol, frag)
        if len(fragments) == 1:
            conf = mol.GetConformer()
            all_coords = []
            for d in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(d)
                all_coords.append((pos.x, pos.y, pos.z))

            # 然后创建 {frag: 所有坐标} 的字典结构
            # fragment.append({frag: all_coords})
            print(all_coords)
        if matches:
            print("MCS 匹配的原子索引:", matches)
            # 提取坐标
            conf = mol.GetConformer()
            # for match in matches:
            coords = [conf.GetAtomPosition(idx) for idx in matches[0]]
            print("匹配原子坐标:", [(p.x, p.y, p.z) for p in coords])
            frag_coords={(p.x, p.y, p.z) for p in coords}
        else:
            print("未找到匹配")
        sub_graph = smiles2graph(frag)
        if (sub_graph['num_nodes'] != len(list(frag_coords))):
            print("长度不统一: {}".format(smiles))
