import torch
from trainer import Trainer
from model.UNet import UNet
from data_loader import WaFerCNNDataset



if __name__ == "__main__":
    # 样例数据
    data = torch.zeros(size=(100, 200, 100), dtype=torch.float)  # 数据，样本数为 100
    labels = torch.zeros(size=(100,), dtype=torch.long)  # 标签，样本数为 100

    """---------------------------划分训练集和测试集---------------------------"""
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

    # 对标签 划分训练集、测试集
    train_labels = labels[train_begin: train_end]
    val_labels = labels[val_begin: val_end]
    test_labels = labels[test_begin:]

    # 对数据 划分训练集、测试集
    train_data = data[train_begin: train_end]
    val_data = data[val_begin: val_end]
    test_data = data[test_begin:]
    """---------------------------封装数据集---------------------------"""
    # 封装训练集和验证集
    train_dataset = WaFerCNNDataset(data=train_data, labels=train_labels, is_train=True)
    val_dataset = WaFerCNNDataset(data=val_data, labels=val_labels, is_train=False)
    test_dataset = WaFerCNNDataset(data=test_data, labels=test_labels, is_train=False)
    
   
    # 实例化模型
    model = UNet(1, 1)

  
    # 创建训练器
    trainer = Trainer(
        classes_num=2, 
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    # 训练
    trainer.do_train(epoch_num=500)

