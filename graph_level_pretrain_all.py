import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import json
import time
import os
from modules.GNNs import GNNGraph
from modules.DataLoad import pyg_molsubdataset
from torch.optim import Adam
from modules.model import Framework
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import argparse
from modules.utils import get_device
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogitsLoss
import torch
from modules.model import LEFTNet
import yaml
from easydict import EasyDict
from datetime import datetime
import torch.optim as optim
from rdkit import RDLogger
import logging

# 获取当前时间并格式化为时间戳
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 创建日志目录
os.makedirs("logs", exist_ok=True)

# 动态日志文件名
log_file_name = f"logs/training_{current_time}.log"
# 关闭 RDKit 的警告信息
RDLogger.DisableLog('rdApp.warning')
# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler(log_file_name),  # 输出到带时间戳的文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 示例日志记录
logger.info("Logging system initialized.")

def get_file_name(args):
    file_name = [f'bs_{args.batch_size}']
    file_name.append(f'lr_{args.lr}')
    file_name.append(f'dr_{args.drop_ratio}')
    file_name.append(f'ep_bb_{args.epoch_backbone}')
    file_name.append(f'dataset_{args.dataset}')
    current_time = time.time()
    file_name.append(f'{current_time}')
    return '-'.join(file_name) + '.json', current_time

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name

def get_basename(file_name):
    ans = os.path.basename(file_name)
    if '.' in ans:
        ans = ans.split('.')
        ans = '.'.join(ans[:-1])
    return ans


def get_work_dir(args):
    file_dir = [f'{get_basename(args.encoder)}']
    file_dir.append(f'{get_basename(args.subencoder)}')
    current_time = time.time()
    file_dir.append(f'{current_time}')
    return os.path.join(args.dataset, '-'.join(file_dir))


def build_model_from_config(config):
    model_type = config['type']
    if model_type == 'gin':
        model = GNNGraph(gnn_type = 'gin', **config['params'])
    elif model_type == 'egnn':
        model  = LEFTNet(pos_require_grad=False, cutoff=5.0, num_layers=4,
                hidden_channels=128, num_radial=32, y_mean=0, y_std=1)
    else:
        raise ValueError(f'Invalid model type called {model_type}')
    return model


def init_args():
    parser = argparse.ArgumentParser('Parser for Experiments on OGB')
    parser.add_argument(
        '--config', type=str
        ,default='./configs/molnet.yml'
        )
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    config.parameters.work_dir = get_work_dir(config.parameters)
    config.parameters.exp_name, config.parameters.time = get_file_name(config.parameters)
    return config


def eval_one_epoch(loader, model, device, verbose=False):
    model.eval()
    auc_results = {}
    acc_results = {}
    aupr_results = {}
    preds, targets,tags = [], [],[]
    iterx = tqdm(loader) if verbose else loader
    for subs, graphs in iterx:
        subs = [eval(x) for x in subs]
        graphs = graphs.to(device)
        with torch.no_grad():
            out= model(subs, graphs,type="egnn")
        preds.append(out.detach().cpu().numpy())  # 积累预测结果
        targets.append(graphs.y.cpu().numpy())
        tags.append(graphs.tag.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    tags = np.concatenate(tags, axis=0)
    for tag in np.unique(tags):
        # 筛选对应分类的样本
        mask = (tags == tag)
        preds_tag = preds[mask]
        targets_tag = targets[mask]

        # 计算 AUC
        # if len(np.unique(targets_tag)) == 2:  # 确保标签中有两个类别
        auc = roc_auc_score(targets_tag, preds_tag)
        predicted_classes = [1 if p >= 0.5 else 0 for p in preds_tag]  # 根据阈值0.5转为类别
        acc = accuracy_score(targets_tag, predicted_classes)

        # AUPR
        aupr = average_precision_score(targets_tag, preds_tag)

        auc_results[tag] = auc
        acc_results[tag] = acc
        aupr_results[tag] = aupr

    return roc_auc_score(targets, preds), auc_results, acc_results, aupr_results
    # auc = roc_auc_score(y_gt, y_pred)
    # print(graphs,y_gt)
    # return auc


if __name__ == '__main__':
    args = init_args()
    print(args)
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(os.path.join('log', args.parameters.work_dir)):
        os.makedirs(os.path.join('log', args.parameters.work_dir))
    seed_everything(args.parameters.seed)

    with open(args.parameters.encoder) as f:
        encoder_config = json.load(f)
    with open(args.parameters.subencoder) as f:
        subencoder_config = json.load(f)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  
    os.makedirs(args.train.result_dir, exist_ok=True)
    loss_path = args.train.result_dir + f'{args.parameters.dataset}_{current_time}_loss'
    best_path = args.train.result_dir + f'{args.parameters.dataset}_{current_time}_best'
    PElabeled_smiles,PElabeled_subs, PElabeled_dataset=pyg_molsubdataset(
        args.parameters.PElabeldataset,args.parameters.seed,args.parameters.decomp
    )
    device = get_device(args.parameters.device)

    encoder = build_model_from_config(encoder_config)
    subencoder = build_model_from_config(subencoder_config)
    wrjModel = Framework(
        encoder = encoder, subencoder = subencoder,
        base_dim = encoder_config['result_dim'],
        sub_dim = subencoder_config['result_dim'],
        num_class = PElabeled_dataset.num_tasks, drop_ratio = args.parameters.drop_ratio
    ).to(device)

    ################################################
    PElabeled_data_idx=PElabeled_dataset.get_random_split()
    PE_train_idx = PElabeled_data_idx['train']
    PE_test_idx = PElabeled_data_idx['test']
    PE_train_dataset = PElabeled_dataset[PE_train_idx]
    PE_test_dataset = PElabeled_dataset[PE_test_idx]
    PE_train_subs=[str(PElabeled_subs[x.item()]) for x in PE_train_idx]
    PE_test_subs=[str(PElabeled_subs[x.item()]) for x in PE_test_idx]
    ########### pe-label##########
    pe_train_loader = DataLoader(
        list(zip(PE_train_subs, PE_train_dataset)),
        batch_size = args.parameters.batch_size, shuffle = True
    )
    pe_test_loader = DataLoader(
        list(zip(PE_test_subs, PE_test_dataset)),
        batch_size = args.parameters.batch_size, shuffle = False
    )
    optimizer = Adam(wrjModel.parameters(), lr = args.parameters.lr,weight_decay=0)
    # 设置 ReduceLROnPlateau 学习率调度器
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',        # 'min' 表示你要最小化的度量（比如验证损失）
        patience=10,       # 忍耐次数，在减少学习率之前允许多少个epoch没有改进
        factor=0.6,        # 学习率减少的比例，新学习率 = 旧学习率 * factor
        min_lr=1e-6        # 学习率的下限
    )
    train_curv, valid_curv, test_curv = [], [], []
    best_model, best_reference_model = None, None
    best_valid, best_sum, best_ep,best_test = 0, 0, 0, 0
    # 初始化列表以存储每个 epoch 的训练损失
    train_losses = []

    for ep in range(args.parameters.epoch_backbone):


        print(f'[INFO] Graph level training on {ep} epoch')
        wrjModel = wrjModel.train()
        epoch_loss = 0  # 初始化每个 epoch 的损失
        for subs, graphs in tqdm(pe_train_loader):
            subs = [eval(x) for x in subs]
            graphs = graphs.to(device)
            preds= wrjModel(subs, graphs,type="egnn")
            loss = BCEWithLogitsLoss(preds.squeeze(dim=1), graphs.y.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # epoch_loss += loss.item()  # 累加每个 batch 的损失

        # 计算并记录平均训练损失
        avg_train_loss = epoch_loss / len(pe_train_loader)
        train_losses.append(avg_train_loss)

        print('[INFO] Evaluating the models')
        # train_perf = eval_one_epoch(
        #     pe_train_loader, evaluator, wrjModel,
        #     device, verbose = True
        # )
        # valid_perf = eval_one_epoch(
        #     valid_loader, evaluator, wrjModel,
        #     device, verbose = True
        # )
    
        test_mae,auc,acc,aupr = eval_one_epoch(
            pe_test_loader, wrjModel,
            device, verbose = True
        )
        checkpoint_dir = 'checkpoints_egnngin'
        os.makedirs(checkpoint_dir, exist_ok=True)
        if(ep%50==0):
            checkpoint_path = os.path.join(checkpoint_dir, f'framework_model_epoch_{ep}.pth')
            torch.save({
                'model_state_dict': wrjModel.state_dict(),
                'encoder_config': encoder_config,
                'subencoder_config': subencoder_config,
                'base_dim': encoder_config['result_dim'],
                'sub_dim': subencoder_config['result_dim'],
                'num_class': PElabeled_dataset.num_tasks,
                'drop_ratio': args.parameters.drop_ratio,
                'model_architecture': str(type(wrjModel))
            }, checkpoint_path)

        # save_path = os.path.join('log', args.parameters.work_dir)
        # if test_perf[PElabeled_dataset.eval_metric] > best_test:
        #     best_valid = test_perf[PElabeled_dataset.eval_metric]
        #     best_model = deepcopy(wrjModel.state_dict())
        #     torch.save(best_model, f'{save_path}/pretrain_predtest_KANO_50.pt')
        # valid_curv.append(test_perf[PElabeled_dataset.eval_metric])
        # test_curv.append(test_perf[PElabeled_dataset.eval_metric])
        print({
            'test_mae': test_mae,
            'cardiotoxicity': {'auc': auc[0], 'acc': acc[0], 'aupr': aupr[0]},
            'hepatotoxicity': {'auc': auc[1], 'acc': acc[1], 'aupr': aupr[1]},
            'irritation': {'auc': auc[2], 'acc': acc[2], 'aupr': aupr[2]},
            'respiratory': {'auc': auc[3], 'acc': acc[3], 'aupr': aupr[3]},
        })
        logger.info({
            'test_mae': test_mae,
            'cardiotoxicity': {'auc': auc[0], 'acc': acc[0], 'aupr': aupr[0]},
            'hepatotoxicity': {'auc': auc[1], 'acc': acc[1], 'aupr': aupr[1]},
            'irritation': {'auc': auc[2], 'acc': acc[2], 'aupr': aupr[2]},
            'respiratory': {'auc': auc[3], 'acc': acc[3], 'aupr': aupr[3]},
        })

    # print(save_path)
    # torch.save(best_model, f'{save_path}/pretrain_predtest_KANO_50.pt')
    # pretrain_model = torch.load(f'{save_path}/pretrain_predtest_KANO_50.pt', map_location = device)
    # wrjModel.load_state_dict(pretrain_model)
    #### before result
    # test_perf = eval_one_epoch(
    #         test_loader, evaluator, wrjModel,
    #         device, verbose = True
    #     )
    # valid_perf = eval_one_epoch(
    #         valid_loader, evaluator, wrjModel,
    #         device, verbose = True
    #     )
    # test_perf = eval_one_epoch(
    #         pe_test_loader, evaluator, wrjModel,
    #         device, verbose = True
    #     )
    # print('Best',{'Test': test_perf})
