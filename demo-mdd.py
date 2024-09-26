import torch
import numpy as np
from WaFerDataReader import shuffleIt
from trainer import Trainer
from model.UNet import UNet
from data_loader import WaFerCNNDataset
from wafer_utils import read_ROISignal_by_txt
from WaFerUtils import getPearson_list


if __name__ == "__main__":
     # 打乱顺序
    shuffle_switch = True
    """---------------------------读取数据---------------------------""" 
    mat_dir = r"E:/WorkSpace/workspace_3/dataset/ROISignals_FunImgARCWF/"
    mdd_txt_path = r"E:/WorkSpace/workspace_3/dataset/mdd.txt"
    hc_txt_path = r"E:/WorkSpace/workspace_3/dataset/hc.txt"
    # MDD
    mdd_rois, mdd_ids = read_ROISignal_by_txt(mat_dir, mdd_txt_path, profix='ROISignals', begin_idx=228, end_idx=428)
    # HC
    hc_rois, hc_ids = read_ROISignal_by_txt(mat_dir, hc_txt_path, profix='ROISignals', begin_idx=228, end_idx=428)

    # 生成 Pearson 标签  [按照你的需求，修改生成代码]
    mdd_labels = getPearson_list(data=mdd_rois)
    mdd_labels = [np.expand_dims(mdd_labels[i], axis=0) for i in range(len(mdd_labels))]
    hc_labels = getPearson_list(data=hc_rois)
    hc_labels = [np.expand_dims(hc_labels[i], axis=0) for i in range(len(hc_labels))]

    """---------------------------数据合并、打包---------------------------""" 
    # 合并
    rois = [*mdd_rois, *hc_rois]
    labels = [*mdd_labels, *hc_labels]
    ids =  [*mdd_ids, *hc_ids]

    # 打包
    data_package = {
        'rois': rois,
        'labels': labels,
        'ids': ids
    }
    keys = ['rois', 'labels', 'ids']
    ids = data_package['ids']
    shuffle_switch = True
    """---------------------------打乱数据集 (打乱后不可二次训练)---------------------------"""
    if shuffle_switch:
        print(" [Warning] - 数据集顺序被打乱, 本次结果不可二次训练")
        data_package, sub_id_list = shuffleIt(data_package=data_package)
    """---------------------------数据信息提取---------------------------"""
    data = data_package[keys[0]]
    labels = data_package[keys[1]]
    """---------------------------划分数据集---------------------------"""
    # 数据集划分比例
    split_rate = "7:2:1"
    r1, r2, r3 = split_rate.split(':')

    # 单位长度
    per = len(labels) // (int(r1) + int(r2) + int(r3)) 
    # 开始、结束索引
    train_begin = 0
    train_end = train_begin + per * int(r1)
    val_begin = train_end
    val_end = val_begin + per * int(r2)
    test_begin = val_end

    # 对标签 划分训练集、验证集、测试集
    train_labels = labels[train_begin: train_end]
    val_labels = labels[val_begin: val_end]
    test_labels = labels[test_begin:]

    # 对数据 划分训练集、验证集、测试集
    train_data = data[train_begin: train_end]
    val_data = data[val_begin: val_end]
    test_data = data[test_begin:]
    """---------------------------封装数据集---------------------------"""
    # 封装 训练集、验证集、测试集
    train_dataset = WaFerCNNDataset(data=train_data, labels=train_labels, is_train=True)
    val_dataset = WaFerCNNDataset(data=val_data, labels=val_labels, is_train=False)
    test_dataset = WaFerCNNDataset(data=test_data, labels=test_labels, is_train=False)
    """---------------------------实例化模型---------------------------"""
    # 替换成你的模型
    model = UNet(1, 1)
    """---------------------------实例化训练器---------------------------"""
    # 创建训练器
    trainer = Trainer(
        classes_num=0,  # 回归任务将这个改成0
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        loss_func=torch.nn.MSELoss()
    )

    # 训练
    trainer.do_train(epoch_num=500)

