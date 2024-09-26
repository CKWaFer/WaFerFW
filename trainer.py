import torch
import numpy as np
import random
import numpy as np
import torch
import os
import math
import platform
import tqdm
import time
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from wafer_utils import create_roc_data, draw_ROC
from data_loader import WaFerCNNDataset
from logger import Logger
import warnings
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, 
        classes_num:int,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        init_args: str=None,  # 用于 K折交叉验证 重构模型
        is_k_fold=False,  # K折交叉验证 开关
        k_data_package=[None, None, None],  # K折交叉验证数据包
        ckpt_dir=None,  # 检查点目录
        ckpt_path=None,  # 检查点路径
        train_batch_size=32,
        val_batch_size=32,
        test_batch_size=32,
        data_loader_shuffle=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        optimizer_name='adam',
        lr=1e-3,  # Adam=1e-3, SGD=1e-2
        weight_decay=3e-5,  # Adam=3e-5, SGD=1e-4
        momentum=0.9,
        loss_func=torch.nn.CrossEntropyLoss(),
        logs_dir='./runs/',
        min_distance=0.018,  # 指标间最小差距
        early_stop_epoch=50,  # 触发Early Stop的 Epoch轮数
        best_from='val',  # 从 val / test 中得到最好指标
        positive_sample_list=None,  # 将第 idx 类作为正样本，其余作为负样本，计算最好指标，None为计算平均值的最好指标
        classes_list=None,  # 类别映射列表
        pre_process_func=None,  # 自定义前处理函数
        post_process_func=None,  # 自定义后处理函数
    ):
        # 模型信息
        self.model_info_prefix = ""
        # 类别数
        self.classes_num = classes_num
        # 用于 K折交叉验证 重构模型
        self.init_args = init_args
        # 损失函数
        self.loss_func = loss_func
        # 数据加载器
        self.train_data_loader = self.set_data_loader(dataset=train_dataset, batch_size=train_batch_size, shuffle=data_loader_shuffle)
        self.val_data_loader = self.set_data_loader(dataset=val_dataset, batch_size=val_batch_size, shuffle=data_loader_shuffle)
        self.test_data_loader = self.set_data_loader(dataset=test_dataset, batch_size=test_batch_size, shuffle=data_loader_shuffle)

        # K折交叉验证
        self.is_k_fold = is_k_fold
        if self.is_k_fold:
            try:
                # 读取 K折交叉验证数据包
                self.k, self.k_data, self.k_labels = k_data_package
            except BaseException:
                print("\033[36m- K折交叉验证数据包需满足[k: int, data: torch.tensor, labels: torch.tensor]\033[0m")
                print("\033[31m- K折交叉验证数据包加载失败, 正在停止训练...\033[0m")
                exit()
        self.cur_k = 0  # 当前为 第k折
        

        # ckpt路径
        self.ckpt_dir = ckpt_dir
        # ckpt路径
        self.ckpt_path = ckpt_path
        # 运算平台
        self.device = device
        # 模型
        self.model = model.to(self.device)
        # 获取 运算平台名称
        self.device_name = str([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())][0]).split(',')[0].split(
            '=')[1].replace('\'', '') if self.device.find('cuda') >= 0 else 'CPU ({}架构)'.format(platform.machine())
        # 获取 模型名
        self.model_name = self.model.__class__.__name__

        # 从 val / test 中得到最好指标
        self.best_from = best_from
        # 将第 idx 类作为正样本，其余作为负样本，计算最好指标，None为计算平均值的最好指标
        self.positive_sample_list = positive_sample_list

        # 指标间最小差距
        self.min_distance = min_distance
        # 当前epoch
        self.cur_epoch = 0
        # 初始化 最好ACC
        self.best_acc = 0.
        # 初始化 最好REC
        self.best_rec = 0.
        # 初始化 最好PREC
        self.best_prec = 0.
        # 初始化 最好SPE
        self.best_spe = 0.
        # 初始化 最好SEN
        self.best_sen = 0.
        # 初始化 最好f1_score
        self.best_f1_score = 0.
        # 最好的 epoch
        self.best_epoch = 0
        # 最低的 loss
        self.best_loss = 0. + 1e16

        # Early Stop
        self.early_stop_epoch = early_stop_epoch
        

        # 日志路径
        self.logs_dir = logs_dir
        # 当前训练编号(日期)
        self.create_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())  # 模型实例化的日期
        # 日志记录器
        self.logger = Logger(log_dir=logs_dir, log_name=self.create_time)

        # 载入权重
        self.load_weight()

        # 载入类别列表
        self.classes_list = classes_list

        # 载入 自定义前处理函数
        self.pre_process_func = pre_process_func

        # 载入 自定义后处理函数
        self.post_process_func = post_process_func

        # 优化器
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.optimizer = self.set_optimizer()


    # 配置 optimizer
    def set_optimizer(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise BaseException("未知的优化器")
            return None

    # 配置 DataLoader
    def set_data_loader(self, dataset: torch.utils.data.dataset, batch_size: int, shuffle: bool):
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    # 初始化 模型与训练器
    def reset_trainer(self):
        # 当前epoch
        self.cur_epoch = 0
        # 初始化 最好ACC
        self.best_acc = 0.
        # 初始化 最好REC
        self.best_rec = 0.
        # 初始化 最好PREC
        self.best_prec = 0.
        # 初始化 最好SPE
        self.best_spe = 0.
        # 初始化 最好SEN
        self.best_sen = 0.
        # 初始化 最好f1_score
        self.best_f1_score = 0.
        # 最好的 epoch
        self.best_epoch = 0
        # 最低的 loss
        self.best_loss = 0. + 1e16


        # 更新 optimizer
        self.optimizer = self.set_optimizer()

    # Early Stop
    def check_early_stop(self):
        # 一段时间内模型无更好结果
        if self.cur_epoch > self.best_epoch + self.early_stop_epoch:
            # 触发早停机制
            info = f"- 模型经{self.early_stop_epoch}个Epoch仍未得到更好结果, 触发Early Stop并停止训练..."
            print(f"\033[31m{info}\033[0m")
            # 写入日志
            self.logger.write_log(info, mode='Early Stop')

            return True
        else:
            return False
    # 载入权重
    def load_weight(self):
        """ 尝试加载预训练模型参数 """
        # 模型参数文件保存路径
        self.ckpt_dir = "./weights/" if self.ckpt_dir is None else self.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

        # 尝试获取ckpt_path
        if self.ckpt_path is None:
            fns = os.listdir(self.ckpt_dir)
            for fn in fns:
                # 判断是否是 .pt
                if fn[-3:] != ".pt":
                    continue
                # 尝试获取 best
                able_fn = None
                if f"best_{self.model_name}" in fn:
                    able_fn = fn
                # 判断是否成功获取
                if able_fn is not None:
                    self.ckpt_path = os.path.join(self.ckpt_dir, able_fn)
        
        # 载入模型
        if self.ckpt_path is not None:
            try:
                # 载入模型参数
                self.model.load_state_dict(torch.load(self.ckpt_path), strict=False)
            except FileNotFoundError:
                print("[WaFer提示]：未发现预训练模型权重")
        else:
            # 重置 模型可学习参数
            for m in self.model.modules():
                try:
                    # 尝试初始化参数
                    m.reset_parameters()
                except BaseException:
                    pass

    # 前处理
    def pre_process(self, data_block):
        # 默认前处理函数
        def _func(data_block):
            # 如有需要可重写本函数
            data = data_block[0].float()

            # 判断是分类问题还是回归问题
            if self.classes_num > 0:
                labels = data_block[1].long()
            else:
                labels = data_block[1].float()
                
            return [data.to(self.device)], labels.to(self.device)
        
        # 返回前处理结果
        return _func(data_block) if self.pre_process_func is None else self.pre_process_func(data_block)
    # 后处理
    def post_process(self, predict):
        # 默认后处理函数
        def _func(predict):
            # 如有需要可重写本函数
            return predict
        
        # 返回后处理结果
        return _func(predict) if self.post_process_func is None else self.post_process_func(predict)
    

    # TODO 修复BUG: 数据尺寸不同时，数据加载器报错的问题
    def load_data(self, data_loader: DataLoader):
        # 数据加载列表
        data_load_list = []
        # 尝试捕捉Exceptionn
        try:
            for data_block in data_loader:
                data_load_list.append(data_block)
        except RuntimeError:
            # 捕捉到Exceptionn，手动划分 batch_size
            data_load_list = []
            cur_idx = 0
            tmp_list = []
            for data_block in data_loader.dataset:
                # 将输入块存入临时列表
                tmp_list.append(data_block)
                # 更新下标指向
                cur_idx += 1
                # 判断: 是否放满batch_size
                if cur_idx % data_loader.batch_size == 0:
                    data_load_list.append(tmp_list)
                    tmp_list = []
            # 判断: 最后一个列表有没有被放入
            if len(tmp_list) != 0:
                data_load_list.append(tmp_list)
                tmp_list = []
        
        return data_load_list
    
    # 训练过程
    def _train(self, cur_epoch, epoch_num, *args):
        epoch_loss = 0
        # 将模型设置成 训练模式
        self.model.train()

        # 按Batch取出数据块
        for batch_idx, data_block in enumerate(self.load_data(self.train_data_loader)):
            batch_idx += 1

            # 提取 数据与标签
            data, labels = self.pre_process(data_block=data_block)
            # 前向传播
            predict = self.model.forward(*data)
            # 后处理
            predict = self.post_process(predict)

            # batch损失
            batch_loss = 0
            
            # 将优化器设置为0梯度 - 清除上一个batch计算的梯度
            self.optimizer.zero_grad()

            # 利用损失函数对 预测结果和标签 计算损失loss - 预测结果与真实情况的差距
            loss = self.loss_func(predict.to(self.device), labels.to(self.device))

            # 反向传播 - 根据模型梯度图(模型拟合出的函数的导函数)求解最小值
            loss.backward()

            # 优化器根据反向传播结果更新模型参数
            self.optimizer.step() 

            # 信息统计 #
            batch_loss = loss.data  # 当前batch的loss
            epoch_loss += batch_loss  # 当前epoch已经训练完成batch的loss总和
            avg_loss = epoch_loss / batch_idx  # 当前epoch的平均loss

            # 显示信息 #
            train_info = "- epoch: {}/{}{:5.5}batch: {}/{}{:5.5}batch_size: {}{:5.5}batch_loss: {:9.9}{:5.5}epoch_loss: {:9.9}".format(
                cur_epoch,
                epoch_num,
                " ",
                batch_idx,
                math.ceil(len(self.train_data_loader.dataset) / self.train_data_loader.batch_size),
                " ",
                self.train_data_loader.batch_size,
                " ",
                batch_loss,
                " ",
                avg_loss
            )
            
            print(f"\r{train_info}", end="")
            # 记录日志
            self.logger.write_log(log=train_info, mode='TRAIN', info=f"K-FOLD-{self.cur_k}" if self.cur_epoch != 0 else "/")
        # 本 Epoch 训练结束换行
        print("\n")

    # 验证过程
    def _val(self, mode='val'):
        classes_tf_list = [[0, 0] for _ in range(self.classes_num)]  # 存放各类别的正确数和错误数，初始化为[0, 0]  [正确数, 错误数]

        tp = 0  # 存放各类别TP，初始化为0
        tn = 0  # 存放各类别TN，初始化为0
        fp = 0  # 存放各类别FP，初始化为0
        fn = 0  # 存放各类别FN，初始化为0

        acc = 0  # 存放各类别ACC，初始化为0
        rec = 0  # 存放各类别REC，初始化为0
        prec = 0  # 存放各类别PREC，初始化为0
        sen = 0  # 存放各类别SEN，初始化为0
        spe = 0  # 存放各类别SPE，初始化为0
        f1_score = 0  # 存放各类别F1-Score，初始化为0

        

        # ROC
        predict_list = list()
        label_list = list()
        loss_list = list()
        
        # 将模型设置成 验证模式
        self.model.eval()

        # 选择 data_loader
        data_loader = self.test_data_loader if mode == 'test' else self.val_data_loader

        # 按Batch取出数据块
        for data_block in tqdm.tqdm(self.load_data(data_loader)):
            # 提取 数据与标签
            data, labels = self.pre_process(data_block=data_block)
            # 进行前向传播，得到模型预测的结果 - 以无梯度状态
            with torch.no_grad():
                predict = self.model.forward(*data)
            # 后处理
            predict = self.post_process(predict)
            if self.classes_num > 0:
                # 记录
                predict_list.extend(predict.tolist())
                label_list.extend(labels.tolist())

                loss_func_name = str(self.loss_func.named_modules)[: -1].split(' ')[-1]
                # 取出数值最大的下标作为分类结果
                if loss_func_name == "CrossEntropyLoss()":
                    _, predict = predict.max(1)  # 最大的1个
                elif loss_func_name == "BCELoss()":
                    predict = predict.view(-1, 1).repeat(1, 2)
                    predict[:, 0] =  1 - predict[:, 0]
                    _, predict = predict.max(1)  # 最大的1个
                else:
                    ...

                # 信息统计 #
                for i in range(labels.size(0)):
                    # classes_num_list[int(labels[i])] += 1  # 根据标签统计当前batch各种类的数量
                    # 统计当前batch各种类预测正确的数量
                    if torch.eq(predict[i], labels[i]):  # 预测正确
                        # 统计正确数
                        classes_tf_list[int(labels[i])][0] += 1
                        # 判断: TP or TN
                        if int(labels[i]) in self.positive_sample_list:
                            # TP
                            tp += 1
                        else:
                            # TN
                            tn += 1
                    else:  # 预测错误
                        # 统计错误数
                        classes_tf_list[int(labels[i])][1] += 1
                        # 判断: FP or FN
                        if int(labels[i]) in self.positive_sample_list:
                            # FN
                            fn += 1
                        else:
                            # FP
                            fp += 1
            else:
                # 回归问题
                # 提取 数据与标签
                data, labels = self.pre_process(data_block)
                # 进行前向传播，得到模型预测的结果 - 以无梯度状态
                with torch.no_grad():
                    predict = self.model.forward(*data)
                # 后处理
                predict = self.post_process(predict)
                with torch.no_grad():
                    # 计算loss
                    loss = self.loss_func(predict, labels)
                    # 记录
                    loss_list.append(loss)

        
        if self.classes_num > 0:
            # 计算 性能指标
            # 正确率 ACC
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            # 召回率 REC
            rec = tp / (tp + fn + 1e-8)
            # 精度 PREC
            prec = tp / (tp + fp + 1e-8)
            # 特异度 SPE
            spe = tn / (tn + fp + 1e-8)
            # # 灵敏度 SEN
            sen = tp / (tp + fn + 1e-8)
            # F1 Score
            f1_score = 2 * prec * rec / (prec + rec + 1e-8)

            # 指标输出语句
            val_info = "- 正确率(ACC): {:7.7}%{:5.5}召回率(REC): {:7.7}%{:5.5}精度(PREC): {:7.7}%{:5.5}特异度(SPE): {:7.7}%{:5.5}敏感度(SEN): {:7.7}%{:5.5}F1-Score: {:7.7}%\n".format(
                np.round(acc * 100, 4), 
                " ",
                np.round(rec * 100, 4), 
                " ",
                np.round(prec * 100, 4), 
                " ",
                np.round(spe * 100, 4), 
                " ",
                np.round(sen * 100, 4), 
                " ",
                np.round(f1_score * 100, 4)
            )
            # 逐个
            if mode == 'test':
                for i in range(self.classes_num):                                
                    val_info += "  - 『类别{}{:>5}』:正确率(ACC)({}/{}): {:7.7}%{:5.5}正确数: {}{:5.5}错误数: {}{:5.5}样本类型: {}{:5.5}\n".format(
                        str(i),
                        "" if self.classes_list is None else self.classes_list[i],
                        classes_tf_list[i][0],
                        classes_tf_list[i][0] + classes_tf_list[i][1],
                        np.round(classes_tf_list[i][0] / (classes_tf_list[i][0] + classes_tf_list[i][1]) * 100, 4),
                        " ",
                        classes_tf_list[i][0], 
                        " ",
                        classes_tf_list[i][1], 
                        " ",
                        "正样本" if i in self.positive_sample_list else "负样本", 
                        " ",
                    )
            else:
                pass
        else:
            # 回归问题
            avg_loss = 0
            for l in loss_list:
                avg_loss += (l / len(loss_list))
            val_info = "  - 验证结果 loss: {:7.7}".format(avg_loss)

        # 打印 验证信息
        if mode != 'pre_val':
            print(val_info)
        # 记录日志
        self.logger.write_log(
            log="{}{}".format('\n' if self.classes_num > 0 and mode == 'test' else '', val_info), 
            mode=mode, 
            info=f"K-FOLD-{self.cur_k}" if self.cur_epoch != 0 else "/"
        )

        # 计算综合评分 统计最好结果
        if self.classes_num > 0:
            cur_best_acc = acc
            cur_best_rec = rec
            cur_best_prec = prec
            cur_best_spe = spe
            cur_best_sen = sen
            cur_best_f1_score = f1_score

            # 载入 指标, 计算分数
            score = np.array([cur_best_acc, cur_best_spe, cur_best_sen])  # 论文评判指标
            # 计算指标间差距
            score_distance = np.max(score) - np.min(score)
            # 判断是否为最好结果
            if self.best_acc <= cur_best_acc and score_distance <= self.min_distance:
                # 2分类 记录 ROC
                if self.classes_num == 2:
                    # 生成 ROC 数据
                    fpr_list, tpr_list = create_roc_data(predict_list=predict_list, label_list=label_list)

                    # 保存 指标 和 图
                    if mode == self.best_from:  # 作为参考指标
                        # 记录最好结果
                        self.best_acc = cur_best_acc
                        self.best_rec = cur_best_rec
                        self.best_prec = cur_best_prec
                        self.best_spe = cur_best_spe
                        self.best_sen = cur_best_sen
                        self.best_f1_score = cur_best_f1_score
                        self.best_epoch = self.cur_epoch

                        # 保存 ROC
                        is_show = False
                        is_save = True
                        save_path = os.path.join(self.logger.log_dir, '{}best_roc_acc{:.4}_epoch{}.png'.format(self.model_info_prefix, self.best_acc, self.best_epoch))
                    elif mode == 'pre_val':
                        return False
                    elif mode in ['val', 'test']:  # 不作为best参考指标
                        is_show = False
                        is_save = False
                        save_path = None
                        return False
                    else:  # 单独预测看结果
                        is_show = True
                        is_save = False
                        save_path = None
                    # 画 ROC图
                    draw_ROC(fpr_list=fpr_list, tpr_list=tpr_list,
                        is_show=is_show, is_save=is_save, save_path=save_path)
                return True
            else:
                return False
        else:
            # 回归问题
            self.model_info_prefix = "min_loss-".format(avg_loss)
            if mode == self.best_from:  # 作为参考指标
                # 记录最好结果
                if torch.isnan(avg_loss):
                    return False
                if self.best_loss > avg_loss:
                    self.best_loss = avg_loss
                    return True
                else:
                    return False

    # 删除上一次best
    def del_old_best(self):    
        for fn in os.listdir(self.ckpt_dir):
            # 判断是否是 .pt
            if fn[-3:] != ".pt":
                continue
            # 尝试删除 best
            if f"{self.model_info_prefix}best_{self.model_name}" in fn:
                os.remove(os.path.join(self.ckpt_dir, fn))
        for fn in os.listdir(self.logger.log_dir):
            # 判断是否是 .pt
            if fn[-3:] != ".pt":
                continue
            # 尝试删除 best
            if f"{self.model_info_prefix}best_{self.model_name}" in fn:
                os.remove(os.path.join(self.logger.log_dir, fn))

    # 删除上一次last
    def del_old_last(self):    
        for fn in os.listdir(self.ckpt_dir):
            # 判断是否是 .pt
            if fn[-3:] != ".pt":
                continue
            # 尝试删除 best
            if f"{self.model_info_prefix}last_{self.model_name}" in fn:
                os.remove(os.path.join(self.ckpt_dir, fn))
        for fn in os.listdir(self.logger.log_dir):
            # 判断是否是 .pt
            if fn[-3:] != ".pt":
                continue
            # 尝试删除 best
            if f"{self.model_info_prefix}last_{self.model_name}" in fn:
                os.remove(os.path.join(self.logger.log_dir, fn))


    # 训练流程
    def do_train(self, epoch_num=300, *args):
        # 训练前验证
        print("\r\033[32m- 训练前验证 -\033[0m")
        self._val(mode='pre_val')

        # best 评判模型
        best_from = 'test' if self.is_k_fold else 'val'

        # 模型设置训练状态
        for e in range(epoch_num):
            # 更新 当前Epoch
            self.cur_epoch = e + 1

            # 检查 Early Stop
            if self.check_early_stop():
                break

            # 训练
            print(f"\r\033[34m# 训练(TRAIN) # \033[0m  with \033[31m{self.device_name}\033[0m")
            self._train(cur_epoch=self.cur_epoch, epoch_num=epoch_num)
            
            # 验证
            print("\r\033[32m# 验证(VAL) #\033[0m")
            val_best = self._val(mode="val")

            # 测试
            print("\r\033[36m# 测试(TEST) #\033[0m")
            test_best = self._val(mode="test")

            # 选取 best 评判结果
            if best_from == 'val':
                is_best = val_best
            elif best_from == 'test':
                is_best = test_best
            else:
                is_best = False

            # 保存参数
            if is_best:
                best_ckpt_name = "{}best_{}_acc{:.4}_epoch{}.pt".format(self.model_info_prefix, self.model_name, self.best_acc, self.cur_epoch)
                best_ckpt_path = os.path.join(self.ckpt_dir, best_ckpt_name)
                # 删除上一次best
                self.del_old_best()
                # 保存 best
                torch.save(self.model.state_dict(), best_ckpt_path)
                # 保存 best 到 runs
                torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, best_ckpt_name))

            # last 路径
            last_ckpt_name = "{}last_{}.pt".format(self.model_info_prefix, self.model_name)
            last_ckpt_path = os.path.join(self.ckpt_dir, last_ckpt_name)
            # 删除上一次last
            self.del_old_last()
            # 保存 last
            torch.save(self.model.state_dict(), last_ckpt_path)
            # 保存 last 到 runs
            torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, last_ckpt_name))

            # 清理内存
            import gc
            gc.collect()
        # 总结 本次训练的 best
        result_info = "- 训练结束, 得到的最优结果: best_epoch: {},    best_acc: {:.5}    best_spe: {:.5},    best_sen: {:.5},    best_rec: {:.5},    best_prec: {:.5}\n".format(
            self.best_epoch, 
            self.best_acc,
            self.best_spe,
            self.best_sen,
            self.best_rec,
            self.best_prec
        )
        # 打印信息
        print(result_info)
        # 记录日志
        self.logger.write_log(log=result_info, mode='RESULT')
        

    # K折交叉验证
    def do_k_fold_train(self, k_data_package, init_args, epoch_num=300, shuffle=True, reload_test=True, init_func: callable=None):  # reload_test: 是否覆盖掉 test数据集
        # 设置 best
        self.best_from = 'test'
        # 加载 k-flod 参数
        self.init_args = init_args
        # K折交叉验证
        self.is_k_fold = True
        if self.is_k_fold:
            try:
                # 读取 K折交叉验证数据包
                self.k, self.k_data, self.k_labels = k_data_package
            except BaseException:
                print("\033[36m- K折交叉验证数据包需满足[k: int, data: torch.tensor, labels: torch.tensor]\033[0m")
                print("\033[31m- K折交叉验证数据包加载失败, 正在停止训练...\033[0m")
                exit()

        # 分割数据集
        k_data_list = [[] for _ in range(self.k)]
        k_label_list = [[] for _ in range(self.k)]
        per_len = math.ceil(len(self.k_data) / self.k)
        for i in range(len(self.k_data)):
            # 分割点下标
            idx = i // per_len
            
            k_data_list[idx].append(self.k_data[i])
            k_label_list[idx].append(self.k_labels[i])

        # k-fold 结果存放
        k_acc = list()  # 正确率
        k_spe = list()  # 特异度
        k_sen = list()  # 敏感度
        k_rec = list()  # 召回率
        k_prec = list()  # 精度

        # 开启 K折交叉验证
        for i in range(self.k):
            # 更新 k折信息
            self.cur_k = i + 1
            # 划分 训练集和测试集
            tmp_kdl = k_data_list[:]
            tmp_kll = k_label_list[:]

            val_data = tmp_kdl.pop(i)
            train_data = [i for item in tmp_kdl for i in item]
            val_labels= tmp_kll.pop(i)
            train_labels = [i for item in tmp_kll for i in item]

            # 封装数据集
            train_dataset = WaFerCNNDataset(data=train_data, labels=train_labels, is_train=True)
            val_dataset = WaFerCNNDataset(data=val_data, labels=val_labels, is_train=False)

            # 更新 数据加载器
            self.train_data_loader = self.set_data_loader(dataset=train_dataset, batch_size=self.train_data_loader.batch_size, shuffle=shuffle)
            self.val_data_loader = self.set_data_loader(dataset=val_dataset, batch_size=self.val_data_loader.batch_size, shuffle=shuffle)
            # 是否覆盖掉 test数据集
            if reload_test:
                self.test_data_loader = self.val_data_loader


            # 初始化 模型
            # 重置 模型可学习参数
            for m in self.model.modules():
                try:
                    # 尝试初始化参数
                    m.reset_parameters()
                except BaseException:
                    pass
            self.model.__init__(*eval(self.init_args))
            self.model = self.model.to(self.device)

            # 执行 自定义函数
            if init_func is not None:
                init_func(self.model)  # 必须传入模型网络

            # 初始化 参数与训练器
            self.reset_trainer()  

            # 设置 模型信息前缀
            self.model_info_prefix = f"K-FOLD-{self.cur_k}-"
            # 记录日志
            info = f"=======================================第 {self.cur_k} 折 交叉验证, 共 {self.k} 折======================================="
            print(f"\033[37m{info}\033[0m")
            self.logger.write_log(log=info, mode=self.model_info_prefix[:-1], info=f"K-FOLD-{self.cur_k}" if self.cur_epoch != 0 else "/")
            # 进入训练
            self.do_train(epoch_num=epoch_num)

            # 统计 第i折的指标
            k_acc.append(self.best_acc)  # ACC
            k_spe.append(self.best_spe)  # SPE
            k_sen.append(self.best_sen)  # SEN
            k_rec.append(self.best_rec)  # REC
            k_prec.append(self.best_prec)  # PREC
        # 交叉验证完成
        info = f"======================================={self.cur_k} 折 交叉验证完成======================================="
        print(f"\033[37m{info}\033[0m")
        # 计算结果
        acc = np.array(k_acc).mean()
        spe = np.array(k_spe).mean()
        sen = np.array(k_sen).mean()
        rec = np.array(k_rec).mean()
        prec = np.array(k_prec).mean()
        # 总结
        # 每折最好结果
        result_info = "  {:4}{:5}{:>7}{:5}{:>7}{:5}{:>7}{:5}{:>7}{:5}{:>7}{:5}\n".format(
            "k","",
            "ACC","",
            "SPE","",
            "SEN","",
            "REC","",
            "PREC","",
        )
        for i in range(self.k):
            result_info += "  {:4}{:5}{:>7.5}{:5}{:>7.5}{:5}{:>7.5}{:5}{:>7.5}{:5}{:>7.5}{:5}\n".format(
                str(i + 1),"",
                k_acc[i],"",
                k_spe[i],"",
                k_sen[i],"",
                k_rec[i],"",
                k_prec[i],"",
            )
        result_info += "  {:4}{:5}{:>7.5}{:5}{:>7.5}{:5}{:>7.5}{:5}{:>7.5}{:5}{:>7.5}{:5}\n".format(
            "AVG", "",
            acc, "",
            spe, "",
            sen, "",
            rec, "",
            prec, "",
        )
    # 打印信息
        print(result_info)
        # 记录日志
        self.logger.write_log(log=info, mode='K-FOLD FINISH')
        self.logger.write_log(log=result_info, mode='K-FOLD RESULT')


            
