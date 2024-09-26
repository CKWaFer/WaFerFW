import torch
import numpy as np
import os
import math
import platform
import gc
import time
import datetime
from torch.utils import data as data_utils
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class WaFerNet(torch.nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(WaFerNet, self).__init__()
        self.learn_param = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float32),
                                              requires_grad=True)  # 默认可学习参数
        self.class_num = None  # 分类数(N分类问题)  在 self.init()里初始化
        self.model_type = None
        self.model_root = None  # 模型参数加载文件根目录  在 self.init()里初始化
        self.model_path = None  # 模型参数加载文件路径  在 self.init()里初始化
        self.model_file_type = None  # 模型参数文件类型 在 self.init()里初始化
        self.create_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())  # 模型实例化的日期
        self.done_epoch = 0  # 已完成的epoch    在 self.init()里读取 .pkl文件名获取
        self.train_data_loader = None  # 训练集加载器  在 self.init()里初始化
        self.val_data_loader = None  # 验证集加载器  在 self.init()里初始化
        self.test_data_loader = None  # 测试集加载器  在 self.init()里初始化
        self.device = device  # 运行方式 cuda 或 cpu  在 self.init()里初始化
        self.device_name = None  # 显卡名称 在self.init()里获取
        self.loss_func = None  # 损失函数  在 self.init()里初始化
        self.optimizer = None  # 优化器  在 self.init()里初始化

        self.is_val = None  # 是否验证  在 self.init()里初始化
        self.optimizer_dict = dict()  # 自定义优化器字典，如果此变量不为空,则框架原始optimizer将无法正常更新参数
        self.loss_list = list()  # 自定义损失列表，如果此变量不为空,则框架原始loss将无法正常反向传播
        self.with_label = False  # 训练过程中，前向传播是否在data中封装labels  在 self.init()里初始化
        self.model_info = ""  # 用户对模型的描述，用于模型参数的区分  在 self.init()里初始化
        self.p = 0.005  # 用于评判最好模型的指标，默认最好指标与最差指标相差不超过0.005  在 self.init()里初始化

        self.args = None  # 其他的变量(由元组形式保存)  在 self.init()里初始化
        self.kwargs = None  # 其他的变量(由字典形式保存)  在 self.init()里初始化

    def forward(self, x):
        forward_error = "[WaFer警告] 您需要重写forward(x)函数来定义前向传播过程，但目前并没有被重写！"
        raise WaFerError(forward_error)

    def do_train(self, epoch):
        # 模型参数文件保存路径
        save_root = "./weights/" if self.model_root is None else self.model_root
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        # 设定 最好指标
        best_score = 0  # 最好的综合指标
        if self.is_val:
            res = self._do_val(mode='best')  # 最好的AUC(ROC曲线)
            best_ac = res[0]  # 最好的正确率
            best_score = best_ac  # 最好的综合指标
            if self.class_num == 2:
                best_ac = res[0]  # 最好的正确率
                best_f1score = res[2]  # F1-Score
                best_auc = float(auc([0., res[1][2][0], 1.], [0., res[1][2][1], 1.]))  # 最好的AUC(ROC曲线)
                # 计算综合评分
                score = np.array([best_ac, best_f1score, best_auc])
                max_distance = np.max(score) - np.min(score)
                if max_distance <= self.p:
                    # best_score = np.max(score)  # 最好的综合指标
                    best_score = best_ac  # 最好的综合指标
        train_start_time = time.time()
        for e in range(epoch):
            epoch_start_time = time.time()
            if self.train_data_loader is None:  # 检查 训练集加载器是否加载
                raise WaFerError("训练数据集还没导入哦")
            loss_avg = None  # 损失loss
            count = 0  # 样本计数器
            for i, data_block in enumerate(self.train_data_loader):
                begin_time = time.time()

                data = data_block[0]
                labels = data_block[1]

                model_input = list()
                # 训练过程中，前向传播是否在data中封装labels
                if self.with_label:
                    model_input.append(data)
                    model_input.append(labels)
                else:
                    model_input.append(data)
                if len(model_input) == 1:
                    model_input = model_input[0]

                # 前向传播
                predict = self.do_predict(data=model_input, mode='train')
                # 计算loss
                loss = self.do_loss(predict=predict, labels=labels)

                # 将loss封装成列表
                loss_list = list()
                if type(loss) is not type(list()):
                    loss_list.append(loss)
                else:
                    for l in loss:
                        loss_list.append(l)

                # 释放内存
                del data
                del labels
                del loss
                gc.collect()

                # 记录平均loss
                if loss_avg is None:
                    loss_avg = list()
                    for idx in range(len(loss_list)):
                        loss_avg.append(loss_list[idx])
                else:
                    for idx in range(len(loss_list)):
                        loss_avg[idx] += loss_list[idx]
                count += 1  # 累加batch完成数

                # 打印当前Batch的loss
                info = self.show_loss(e=e, i=i, loss=loss_list)
                # 记录当前信息
                self.log_result("train.log", info + "\n")

                # 记录当前时间
                end_time = time.time()
                if (1 + i) < math.ceil(len(self.train_data_loader.dataset) / self.train_data_loader.batch_size):
                    use_time = math.ceil(end_time - begin_time)
                    not_done_batch = math.ceil(len(self.train_data_loader.dataset) / self.train_data_loader.batch_size) - (1 + i)
                    not_done_time = use_time * not_done_batch
                    use_time = datetime.timedelta(seconds=not_done_time)
                    print(" | 剩余时间: {}".format(use_time), end='')

                # 反向传播计算梯度
                self.do_backward(clean_graph_after_end=True)

            # 1个epoch的训练过程结束 #
            self.done_epoch += 1
            # 自定义运行程序
            self.do_afterEpoch()

            # 计算平均loss
            for n in range(len(loss_avg)):
                loss_avg[n] /= count  # 计算本次epoch的 平均loss
            # 打印平均loss
            loss_avg_str = ""
            for l in loss_avg:
                loss_avg_str += " {:.4} ".format(l)
            print(" | \033[32m平均loss:[{}]\033[0m".format(loss_avg_str), end='')
            # 记录当前信息
            self.log_result("train.log", "- 训练集平均loss: [{}] -\n".format(loss_avg_str))

            # 记录当前epoch用时
            now_time = time.time()
            epoch_use_time = datetime.timedelta(seconds=(math.ceil(now_time - epoch_start_time)))
            all_use_time = datetime.timedelta(seconds=(math.ceil(now_time - train_start_time)))
            print(" | epoch用时: {}  总用时: {}".format(epoch_use_time, all_use_time))

            # 进行准确率验证
            if self.is_val is True:
                # 验证并保存模型参数
                res = self._do_val(mode='val')  # 获取正确率
                cur_ac = res[0]  # 最好的正确率
                info = res[1]  # 信息

                if self.class_num == 2:
                    cur_score = 0
                    cur_f1score = res[2]  # F1-Score
                    cur_auc = float(auc([0., info[2][0], 1.], [0., info[2][1], 1.]))  # 最好的AUC(ROC曲线)
                    # 计算综合评分
                    score = np.array([cur_ac, cur_f1score, cur_auc])
                    max_distance = np.max(score) - np.min(score)
                    if max_distance <= self.p:
                        # cur_score = np.max(score)  # 当前的综合指标
                        cur_score = cur_ac  # 当前的综合指标

                # 记录当前的综合指标
                if self.class_num != 2:
                    cur_score = cur_ac

                if self.class_num == 2:
                    print(
                        " | \033[34m正确率: {:.4}%\033[0m, \033[35mF1-Score: {:.4}%\033[0m, \033[36mROC: {:.4}%\033[0m".format(
                            cur_ac * 100, cur_f1score * 100, cur_auc * 100), end='')
                    print(info[0])
                else:
                    print(
                        " | \033[34m正确率: {:.4}%\033[0m".format(
                            cur_ac * 100), end='')
                    print(info)

                # 记录当前信息
                self.log_result("train.log", "\n[验证集结果]\n{}".format(info[1]))
                self.log_result("train.log", "----------------------------------------------------------\n")

            # 保存 best训练模型参数
            if self.is_val:
                if cur_score > best_score:  # 本次结果为最好结果
                    best_score = cur_score  # 更新最好结果
                    # 保存 best训练模型参数
                    best_model_info = "{}/best_{}{}{}".format(save_root,
                                                              self.__class__.__name__,
                                                              self.model_info,
                                                              self.model_file_type)
                    # 保存到 ./model
                    torch.save(self.state_dict(), best_model_info)
                    # 保存到 ./result
                    cur_result_info = './result/' + self.create_time + "/best_{}{}{}".format(self.__class__.__name__, self.model_info, self.model_file_type)
                    torch.save(self.state_dict(), cur_result_info)
                    if self.class_num == 2:
                        # 保存 ROC曲线图
                        self.drwa_ROC(info[2][0], info[2][1])

            # 保存 last训练模型参数
            last_model_info = "{}/last_{}{}{}".format(save_root,
                                                      self.__class__.__name__,
                                                      self.model_info,
                                                      self.model_file_type)
            torch.save(self.state_dict(), last_model_info)
            # 保存到 ./result
            cur_result_info = './result/' + self.create_time + "/last_{}{}{}".format(self.__class__.__name__, self.model_info, self.model_file_type)
            torch.save(self.state_dict(), cur_result_info)

    # 验证过程
    def _do_val(self, mode: str):
        data_loader = self.test_data_loader if mode == 'test' else self.val_data_loader
        res = list()
        if self.class_num > 0:  # 是分类问题
            p = 0  # 正样本数
            n = 0  # 负样本数
            total_right = np.zeros(self.class_num, dtype=int)  # 总样本正确数
            total_wrong = np.zeros(self.class_num, dtype=int)  # 总样本错误数
            tp = 0  # 正样本正确数
            tn = 0  # 负样本正确数
            fp = 0  # 正样本错误数
            fn = 0  # 负样本错误数
            # 检查测试集加载器是否加载
            if data_loader is None:  # 检查测试集加载器是否加载
                raise WaFerError("数据集还没导入哦")
            count = 0  # 样本计数器
            for i, data_block in enumerate(data_loader):
                begin_time = time.time()

                data = data_block[0]
                labels = data_block[1]

                # 前向传播，得到预测结果
                with torch.no_grad():  # 在无梯度的情况下进行前向传播
                    predict = self.do_predict(data=data, mode=mode)

                predict = predict.to('cpu')
                count += 1  # 记录完成Batch数

                # 在Val中，且在计算正确率等指标之前，对标签和预测结果处理函数
                predict, labels = self.do_beforeResult(predict=predict, labels=labels)

                assert predict.shape == labels.shape  # 预测结果和标签尺寸必须完全相同

                # 将预测结果放到CPU, 并将其和labels数据类型保持一致
                predict = predict.type_as(labels)

                # 统计结果
                for k in range(labels.shape[0]):
                    if (predict[k].eq(labels[k])).all():  # 预测正确
                        total_right[labels[k]] += 1  # 对应类+1
                        if self.class_num == 2:
                            if predict[k].all():  # 正样本
                                tp += 1
                                p += 1
                            else:  # 负样本
                                tn += 1
                                n += 1
                    else:  # 预测错误
                        total_wrong[labels[k]] += 1  # 对应类+1
                        if self.class_num == 2:
                            if predict[k].all():  # 正样本
                                fp += 1
                                n += 1
                            else:  # 负样本
                                fn += 1
                                p += 1

                end_time = time.time()
                # 显示验证状态
                if mode == 'best':
                    print('\r', " - \033[36m训练前验证：ite:{}/{}, batch_size:{}\033[0m -".format(
                        1 + i,
                        math.ceil(len(data_loader.dataset) / data_loader.batch_size),
                        data_loader.batch_size
                    ), end='', flush=True)
                elif mode == 'val':
                    print('\r', "\t- 验证：ite:{}/{}, batch_size:{}".format(
                        1 + i,
                        math.ceil(len(data_loader.dataset) / data_loader.batch_size),
                        data_loader.batch_size
                    ), end='', flush=True)
                elif mode == 'test':
                    print('\r', "\t\033[31m- 测试结果 -\033[0m\t\t\t测试：ite:{}/{}, batch_size:{}".format(
                        1 + i,
                        math.ceil(len(data_loader.dataset) / data_loader.batch_size),
                        data_loader.batch_size
                    ), end='', flush=True)
                if (1 + i) < math.ceil(len(data_loader.dataset) / data_loader.batch_size):
                    use_time = math.ceil(end_time - begin_time)
                    not_done_batch = math.ceil(len(data_loader.dataset) / data_loader.batch_size) - (1 + i)
                    not_done_time = use_time * not_done_batch
                    use_time = datetime.timedelta(seconds=not_done_time)
                    print(" | 剩余时间: {}".format(use_time), end='')
                # 释放内存
                del data
                del labels
                gc.collect()
            # 验证完成
            no_zero_add = 1e-31  # 避免除数为0
            if self.class_num == 2:
                # 正确率
                accuracy = (tp + tn) / (p + n + no_zero_add)
                # 错误率
                error_rate = (fp + fn) / (p + n + no_zero_add)
                # 灵敏度
                sensitive = tp / (p + no_zero_add)
                # 特效度
                specificity = tn / (n + no_zero_add)
                # 精度
                precision = tp / (tp + fp + no_zero_add)
                # 召回率
                recall = tp / (tp + fn + no_zero_add)
                # F1 Score
                f1_score = 2 * precision * recall / (precision + recall + no_zero_add)
                # 结果信息
                info = str("  正确率(ACC): {:.4}%, F1 Score: {:.4}%\n" +
                           "  灵敏度(SEN): {:.4}%, 特效度(SPE): {:.4}%\n" +
                           "  精度(PREC): {:.4}%, 召回率(REC): {:.4}%\n" +
                           "  正样本正确数：{}/{}, 负样本正确数：{}/{}\n").format(accuracy * 100,
                                                                              f1_score * 100,
                                                                              sensitive * 100,
                                                                              specificity * 100,
                                                                              precision * 100,
                                                                              recall * 100,
                                                                              tp, p,
                                                                              tn, n)
                if mode == 'val' or mode == 'best':
                    info = [" | 正样本正确数: {}/{}  负样本正确数: {}/{}".format(tp, p, tn, n),
                            info,
                            (1 - specificity, sensitive)]
                elif mode == 'test':
                    info = [info, (1 - specificity, sensitive)]
                # 保存结果
                res.append(accuracy)
                res.append(info)
                res.append(f1_score)
            else:
                # 正确率
                accuracy = total_right.sum() / (total_right.sum() + total_wrong.sum() + no_zero_add)
                # 错误率
                error_rate = total_wrong.sum() / (total_right.sum() + total_wrong.sum() + no_zero_add)
                # 结果信息
                tr = "      "
                for i in range(self.class_num):
                    tr += "类型{}正确数：{}/{}".format(i + 1, total_right[i], total_right[i] + total_wrong[i])
                    if (i + 1) % 4 == 0:
                        tr += "\n      "
                    else:
                        tr += "    "
                info = str("\n{}").format(tr)

                # 保存结果
                res.append(accuracy)
                res.append(info)
        else:
            # 回归问题
            # MAE
            # RMAE
            # MAPE
            pass
        return res

    # 测试过程
    def do_test(self):
        print(" |-----------------------------------------------------------------|")
        res = self._do_val(mode='test')
        ac, info = res[0], res[1]
        self.show_test_info(info=info[0])
        if self.class_num == 2:
            self.drwa_ROC(info[1][0], info[1][1], mode="test")

    def show_loss(self, e, i, loss):  # 打印训练过程的loss
        loss_str = ""
        for l in loss:
            loss_str += " {:.4} ".format(l)
        info = "epoch:{}, ite:{}/{}, batch_size:{}, \033[33mloss:[{}]\033[0m".format(
            1 + e,
            1 + i,
            math.ceil(len(self.train_data_loader.dataset) / self.train_data_loader.batch_size),
            self.train_data_loader.batch_size,
            loss_str)
        print('\r ', info, end='', flush=True)
        return info

    def show_test_info(self, info: str):  # 打印测试结果
        print("\n |-----------------------------------------------------------------|")
        print(info)

    # 运行结果记录
    def log_result(self, fn, text, result_root="./result/"):
        # 标准化路径
        result_root = result_root.replace("\\", "/")
        result_root = result_root if result_root[-1] == '/' else result_root + "/"
        # 检查路径是否存在，不存在则新建
        if not os.path.isdir(result_root):
            os.mkdir(result_root)
        # 检查当前训练的文件夹，不存在则新建
        cur_train_dir = result_root + self.create_time + "/"
        if not os.path.isdir(cur_train_dir):
            os.mkdir(cur_train_dir)
        # 检查当前要记录文件，不存在则新建
        cur_log_file = cur_train_dir + fn
        if not os.path.isfile(cur_log_file):
            f = open(cur_log_file, "w", encoding="utf-8")
        else:
            f = open(cur_log_file, "a", encoding="utf-8")

        # 内容过滤
        text = text.replace("\033[33m", "")
        text = text.replace("\033[0m", "")
        f.write(text)
        f.close()

    # 画出 AUC/ROC曲线
    def drwa_ROC(self, fpr, tpr, save_dir="./result/", mode="val"):
        # 标准化路径
        save_dir = save_dir.replace("\\", "/")
        save_dir = save_dir if save_dir[-1] == '/' else save_dir + "/"
        # 检查路径是否存在，不存在则新建
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # 检查当前训练的文件夹，不存在则新建
        cur_train_dir = save_dir + self.create_time + "/"
        if not os.path.isdir(cur_train_dir):
            os.mkdir(cur_train_dir)
        # 检查当前要记录文件，不存在则新建
        cur_ROC_file = cur_train_dir + "best_ROC.png"

        # 计算 ROC指标并绘图
        # fpr, tpr, threshold = roc_curve(labels, pred)
        fpr = [0., fpr, 1.]
        tpr = [0., tpr, 1.]

        roc_auc = auc(fpr, tpr)  # 准确率代表所有正确的占所有数据的比值

        lw = 2
        plt.subplot(1, 1, 1)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve')  # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Epoch: {}  AUC={:.3}'.format(self.done_epoch,roc_auc), y=1.)
        plt.legend(loc="lower right")
        if mode == "val":
            plt.savefig(cur_ROC_file)
        else:
            plt.show()
        plt.close()  # 关闭窗口

    # 前向传播，此部分在需要时可被重写
    def do_predict(self, data, mode: str):
        if mode == 'train':
            self.train()  # 训练 BN dropout等
        else:
            self.eval()  # 打开 BN dropout等
        # 将数据放到device上去
        data = data.to(self.device)

        # 前向传播，得到预测结果
        predict = self.forward(data)

        if mode != 'train':
            if len(predict.shape) >= 2:
                # 取 预测结果的 最大值
                _, predict = predict.max(1)
            else:
                for i in range(predict.shape[0]):
                    predict[i] = 0. if predict[i] < 0.5 else 1.
            return predict
        else:
            return predict

    # 计算loss，此部分在需要时可被重写
    def do_loss(self, predict, labels):
        # 将数据放到device上去
        labels = labels.to(self.device)

        # 得到loss
        loss = self.loss_func(predict, labels)
        loss = self.add_loss_package(opt_name='opt', loss=loss)

        return loss

    # 向优化器字典中添加优化器
    def add_optimizer(self, opt_name, opt: torch.optim.Optimizer, model=None):
        # 判断 优化器是否存在
        if opt_name in self.optimizer_dict.keys():
            raise WaFerError("优化器已存在，请检查优化器名称是否正确。")
        # 创建优化器字典内容
        opt_dict = dict()
        opt_dict['opt'] = opt
        opt_dict['model'] = model
        # 添加到优化器字典
        self.optimizer_dict[opt_name] = opt_dict

    # 向loss列表中添加loss和对应的优化器
    def add_loss_package(self, opt_name, loss):
        # 判断 优化器是否存在
        if not opt_name in self.optimizer_dict.keys():
            raise WaFerError("优化器不存在，请检查优化器名称是否正确。")
        loss_dict = dict()
        loss_dict['opt_name'] = opt_name
        loss_dict['loss'] = loss

        # 添加到loss列表
        self.loss_list.append(loss_dict)

        # 返回 loss数据
        return loss

    # 优化器设置0梯度
    def do_optimizer_zero_grad(self, param=None):
        if len(self.optimizer_dict) == 0:
            self.optimizer.zero_grad()
        else:
            if param is None:
                self.optimizer_dict['opt']['opt'].zero_grad()
                # raise WaFerError("需要指定优化器，请在param中指定优化器名称。优化器名称是自定义的，请输入前进行核实。")
            else:
                self.optimizer_dict[param]['opt'].zero_grad()

    # 反向传播计算梯度
    def do_backward(self, losses=None, is_zero_grad=True, clean_graph_after_end=False):
        if len(self.loss_list) == 0:
            if losses is not None:
                for loss in losses:
                    if is_zero_grad is True:
                        self.do_optimizer_zero_grad()
                    loss.backward()  # 反向传播计算梯度
                    self.do_step()
        else:
            list_lenght = len(self.loss_list)
            count = 0
            for loss_package in self.loss_list:
                count += 1
                if not (('opt_name' in loss_package.keys()) and ('loss' in loss_package.keys())):
                    raise WaFerError("loss列表中的数据有误，请进行核实。")
                opt_name = loss_package['opt_name']
                l = loss_package['loss']
                # 检查是否需要反向传播
                if l.requires_grad is False:
                    continue
                if is_zero_grad is True:
                    self.do_optimizer_zero_grad(param=opt_name)
                # 判断是否在最后一次反向传播清理计算图
                if count == list_lenght and clean_graph_after_end:
                    l.backward()  # 清理计算图
                else:
                    l.backward(retain_graph=True)  # 保留计算图
                self.do_step(param=opt_name)
            # 清空loss列表
            self.loss_list.clear()

    # 更新参数(进行优化)
    def do_step(self, param=None):
        if len(self.optimizer_dict) == 0:
            self.optimizer.step()  # 更新参数(进行优化)
        else:
            if param is None:
                self.optimizer_dict['opt']['opt'].step()
                # raise WaFerError("需要指定优化器，请在(param=)中指定优化器名称。优化器名称是自定义的，请输入前进行核实。")
            else:
                self.optimizer_dict[param]['opt'].step()

    # 在epoch结束时，运行本函数
    def do_afterEpoch(self, param=None):
        pass

    # 在Val中，且在计算正确率等指标之前，对标签和预测结果处理函数
    def do_beforeResult(self, predict, labels, param=None):
        return predict, labels

    # 初始化函数
    def init(self,
             class_num: int,  # 分类数(N分类问题) [必填项]  若不分类(回归问题)，则填 0
             model_type='cnn',  # 模型类型 [必填项]  仅支持 'cnn'，'gcn'
             model_root='./weights/',  # 模型参数文件(.pth) 的保存根目录
             model_path=None,  # 具体 模型参数文件(.pth) 优先级更高
             model_file_type=".pth",  # 模型文件类型
             train_dataset=None,  # 训练数据集
             val_dataset=None,  # 验证数据集
             test_dataset=None,  # 测试数据集
             # ------------------------GCN特有参数------------------------ #
             train_nodes=None,  # 训练结点
             train_edges_index=None,  # 训练边
             train_edges_attr=None,  # 训练边
             val_nodes=None,  # 验证结点
             val_edges_index=None,  # 验证边
             val_edges_attr=None,  # 验证边
             test_nodes=None,  # 测试结点
             test_edges_index=None,  # 测试边
             test_edges_attr=None,  # 测试边
             # ---------------------------------------------------------- #
             # ------------------------CNN特有参数------------------------ #
             train_data=None,  # 训练数据
             val_data=None,  # 验证数据
             test_data=None,  # 测试数据
             # ---------------------------------------------------------- #
             train_labels=None,  # 训练标签
             val_labels=None,  # 验证标签
             test_labels=None,  # 测试标签
             train_batch_size=32,  # batch尺寸
             val_batch_size=32,
             test_batch_size=32,
             shuffle=True,  # 是否打乱
             loss_func=torch.nn.CrossEntropyLoss(),  # 损失函数
             learning_rate=2e-4,  # 优化器学习率
             weight_decay=5e-4,  # 权值衰减
             is_val=True,
             with_label=False,  # 前向传播是否在data中封装labels
             p=0.005,  # 用于评判最好模型的指标，默认最好指标与最差指标相差不超过0.005
             model_info='',  # 用户对模型的描述，用于模型参数的区分  在 self.init()里初始化
             init_weight=False,  # 权重参数是否初始化
             device='cuda' if torch.cuda.is_available() else 'cpu',
             *args,  # 其他的变量(由元组形式保存)
             **kwargs  # 其他的变量(由字典形式保存)
             ):
        # 初始化 分类数
        self.class_num = class_num
        # 初始化 模型类型
        self.model_type = model_type
        # 初始化 模型参数文件类型
        self.model_file_type = model_file_type
        # 初始化 模型描述信息
        self.model_info = model_info

        # 寻找最好的模型参数文件
        if model_path is None:  # 无具体 模型参数文件
            if model_root is not None:
                model_root = std_root(model_root)  # 标准化文件夹路径
                # 判断模型文件夹是否有效
                if os.path.isdir(model_root):
                    best_fn = "best_" + self.__class__.__name__ + self.model_info + self.model_file_type
                    model_path = model_root + best_fn  # 选取最新的 .pth
                else:
                    model_root = None

        # 如果未找到模型文件，则将模型文件路径设置成None
        model_path = '' if model_path is None else model_path
        if not os.path.exists(model_path):
            model_path = None

        # 载入模型
        if model_path is not None:
            try:
                # 载入模型参数
                # self.load_state_dict(torch.load(model_path), strict=True)
                self.load_state_dict(torch.load(model_path), strict=False)
                # 初始化 模型文件夹路径
                self.model_root = model_root
                # 初始化 模型文件路径
                self.model_path = model_path
            except FileNotFoundError:
                print("[WaFer提示]：预训练参数未加载，因为没找到对应的{}文件".format(self.model_file_type))
        else:
            if init_weight:
                # 第一次训练 初始化模型参数
                for m in self.modules():
                    if isinstance(m, (torch.nn.Conv2d,
                                      torch.nn.ConvTranspose2d,
                                      torch.nn.BatchNorm2d,
                                      torch.nn.Linear)):
                        # 高斯分布初始化
                        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                        # # 0值初始化
                        # torch.nn.init.zeros_(m.weight.data)
                    continue

        # 载入数据集
        if self.model_type == 'cnn':
            # 检查 是否加入了 成品 训练数据集 验证数据集 测试数据集
            if (train_dataset is None) or (val_dataset is None) or (test_dataset is None):
                # 将 数据及标签 封装成WaFerDataset
                # 检查 数据和标签 是否为 numpy.array，若是，则转为 tensor
                # 训练数据
                if type(train_data) is list:
                    train_data = np.array(train_data)
                if type(train_labels) is list:
                    train_labels = np.array(train_labels)

                if type(train_data) is np.ndarray:
                    # train_data = train_data.astype(np.float32)
                    train_data = torch.from_numpy(train_data)
                    # if is_nii is True:
                    #     train_data = torch.unsqueeze(train_data, dim=1)
                if type(train_labels) is np.ndarray:
                    # train_labels = train_labels.astype(np.int64)
                    train_labels = torch.from_numpy(train_labels)

                # 验证数据
                if type(val_data) is list:
                    val_data = np.array(val_data)
                if type(val_labels) is list:
                    val_labels = np.array(val_labels)

                if type(val_data) is np.ndarray:
                    # val_data = val_data.astype(np.float32)
                    val_data = torch.from_numpy(val_data)
                    # if is_nii is True:
                    #     val_data = torch.unsqueeze(val_data, dim=1)
                if type(val_labels) is np.ndarray:
                    # val_labels = val_labels.astype(np.int64)
                    val_labels = torch.from_numpy(val_labels)

                # 测试数据
                if type(test_data) is list:
                    test_data = np.array(test_data)
                if type(test_labels) is list:
                    test_labels = np.array(test_labels)

                if type(test_data) is np.ndarray:
                    # test_data = test_data.astype(np.float32)
                    test_data = torch.from_numpy(test_data)
                    # if is_nii is True:
                    #     test_data = torch.unsqueeze(test_data, dim=1)
                if type(test_labels) is np.ndarray:
                    # test_labels = test_labels.astype(np.int64)
                    test_labels = torch.from_numpy(test_labels)

                # 将数据封装成 WaferDataset(继承于Dataset)
                train_dataset = WaFerCNNDataset(data=train_data, labels=train_labels, is_train=True)
                val_dataset = WaFerCNNDataset(data=val_data, labels=val_labels, is_train=False)
                test_dataset = WaFerCNNDataset(data=test_data, labels=test_labels, is_train=False)

            self.train_data_loader = data_utils.DataLoader(dataset=train_dataset,
                                                           batch_size=train_batch_size,
                                                           shuffle=shuffle)
            self.val_data_loader = data_utils.DataLoader(dataset=val_dataset,
                                                         batch_size=val_batch_size,
                                                         shuffle=shuffle)
            self.test_data_loader = data_utils.DataLoader(dataset=test_dataset,
                                                          batch_size=test_batch_size,
                                                          shuffle=shuffle)
        else:
            raise WaFerError("模型种类目前还不支持哦")

        # 初始化 优化器
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        # 将 默认优化器添加到 优化器字典
        self.add_optimizer(opt_name='opt', opt=self.optimizer)
        # 初始化 损失函数
        self.loss_func = loss_func
        # 获取 运算平台名称
        self.device_name = str([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())][0]).split(',')[0].split(
            '=')[1].replace('\'', '') if self.device.find('cuda') >= 0 else 'CPU (基于 {} 架构)'.format(platform.machine())
        # 打印框架信息
        self.show_WaFerFW()
        # 设置是否验证(用于GAN等无监督训练等)
        self.is_val = is_val
        # 训练过程中，前向传播是否在data中封装labels
        self.with_label = with_label
        # 用于评判最好模型的指标，默认最好指标与最差指标相差不超过0.005
        self.p = p
        # 其他的变量(由元组形式保存)  在 self.init()里初始化
        self.args = args
        # 其他的变量(由字典形式保存)  在 self.init()里初始化
        self.kwargs = kwargs
        # 将模型放入 指定平台上
        self.device = device
        self.to(self.device)

    # 输出前向传播结果
    def show_forward(self, image):
        if type(image) is np.ndarray:
            image = torch.from_numpy(image).float()

        image = torch.unsqueeze(image, dim=0)
        image = image.to(self.device)
        with torch.no_grad():
            return self.forward(image)

    # 打印网络结构
    def show_net(self):
        print(self)

    # 打印 WaFerFW信息
    def show_WaFerFW(self):
        # 打印信息
        print("\033[36m||======================================================WaFerFW 训练框架======================================================||\033[0m")
        print("               作者：\033[34m{}\033[0m             模型名称：{}{}             运算平台：{}\n".format('WaFer', self.__class__.__name__, self.model_info, self.device_name))


# 标准化文件夹路径格式
def std_root(root: str):
    root = root.replace('\\', '/')
    root = root if root[-1] == '/' else root + '/'

    return root


class WaFerError(BaseException):
    def with_traceback(self, tb):
        super(WaFerError, self).with_traceback(tb)


# GCN特有函数 - 获取图数据
def get_g_data(x=None,  # 结点 需要包括特征  例：10个结点，每个有5个特征，则x.shape=[10, 5]
               edge_index=None,  # 边  shape=[2, 边数量]   C为边的数量
               edge_attr=None,  # 边的属性  shape=[边数量，边特征数]
               y=None,  # 标签 shape=[有标签的结点数量]
               pos=None  # 结点的坐标 shape=[结点数，(x, y, z)]
     ):
    # 实际上，Data对象不仅仅限制于这些属性，我们可以通过data.face来扩展Data，以张量保存三维网格中三角形的连接性。
    data = pyg_Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    pos=pos)

    return data


# CNN特有 - 数据集
class WaFerCNNDataset(data_utils.Dataset):
    def __init__(self, data, labels, is_train):
        self.data = data
        self.labels = labels
        self.is_train = is_train

        self.idx = list()
        for item in data:
            self.idx.append(item)
        pass

    def __getitem__(self, item):
        input_data = self.idx[item]
        target = self.labels[item]

        return input_data, target

    def __len__(self):
        return len(self.idx)
