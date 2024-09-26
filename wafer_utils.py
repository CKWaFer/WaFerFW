import scipy.io as sio
import os
import tqdm
import numpy as np
import WaFerUtils as wfu
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# 根据被试列表 读取 mat
def read_mats_by_ids(mat_dir: str, bs_ids: list, profix='ROISignals_'):
    mat_list = list()
    none_list = list()
    for bs_id in tqdm.tqdm(bs_ids):
        mat_fn = f"{profix}_{bs_id}.mat"
        mat_path = os.path.join(mat_dir, mat_fn)
        if not os.path.isfile(mat_path):
            none_list.append(bs_id)
            continue
        mat = sio.loadmat(mat_path)[profix]  # 读取文件
        mat_list.append(mat)
    print(f" - 数据缺失: {none_list}")
    return mat_list

def read_mats_by_txt(mat_dir: str, bs_txt_path: str, profix: str):
    # 打开文件
    with open(bs_txt_path, 'r') as file:
        # 读取所有行并存储到列表中
        lines = file.readlines()
    # 移除列表中每行末尾的换行符
    bs_ids = [line.strip() for line in lines]

    mat_list = list()
    none_list = list()
    min_tp = 1e5
    for bs_id in tqdm.tqdm(bs_ids):
        mat_fn = f"{profix}_{bs_id}.mat"
        mat_path = os.path.join(mat_dir, mat_fn)
        if not os.path.isfile(mat_path):
            none_list.append(bs_id)
            continue
        mat = sio.loadmat(mat_path)[profix]  # 读取文件
        if mat.shape[1] != 1833:
            none_list.append(bs_id)
            continue
        if min_tp > mat.shape[0]:
            min_tp = mat.shape[0]
        mat_list.append(mat)
    print(f" - 数据缺失/排除: {none_list}")
    for i in range(len(mat_list)):
        mat_list[i] = mat_list[i][:min_tp, 228: 428]
    return mat_list

def read_fc_by_txt(mat_dir: str, bs_txt_path: str, profix: str):
    # 打开文件
    with open(bs_txt_path, 'r') as file:
        # 读取所有行并存储到列表中
        lines = file.readlines()
    # 移除列表中每行末尾的换行符
    bs_ids = [line.strip() for line in lines]

    fc_list = list()
    none_list = list()
    id_list = list()
    for bs_id in tqdm.tqdm(bs_ids):
        mat_fn = f"{profix}_{bs_id}.mat"
        mat_path = os.path.join(mat_dir, mat_fn)
        if not os.path.isfile(mat_path):
            none_list.append(bs_id)
            continue
        mat = sio.loadmat(mat_path)[profix]  # 读取文件
        mat = np.transpose(mat, (1, 0))
        mat = mat[228: 428]
        fc = wfu.getPearson_ndarray(data=mat, show=False)
        if fc.shape != (200, 200):
            # raise BaseException(f"{mat_fn}: {fc.shape} - {mat.shape}")
            none_list.append(bs_id)
            continue
        fc_list.append(fc)
        id_list.append(bs_id)
    print(f" - 数据缺失: {none_list}")
    return fc_list, id_list

def read_1d_by_txt(roi_dir: str, bs_txt_path: str, min_time_point: int=1, suffix: str='rois_cc200'):
    # 打开文件
    lines = None
    with open(bs_txt_path, 'r') as file:
        # 读取所有行并存储到列表中
        lines = file.readlines()
    # 移除表头
    lines = lines[1:]

    none_list = list()
    except_list = list()

    rois_list = list()
    label_list = list()
    sub_id_list = list()
    site_list = list()
    age_list = list()
    sex_list = list()

    for line in tqdm.tqdm(lines):
        sub_id, site_id, label, age, sex = line.split("\t")
        
        label = int(label)
        age = float(age)
        sex = int(sex)

        label -= 1
        sex -= 1

        fn = f"{site_id}_{sub_id}_{suffix}.1D"
        fp = os.path.join(roi_dir, fn)

        if not os.path.isfile(fp):
            none_list.append(f"{site_id}_{sub_id}")
            continue
        roi_signal = np.loadtxt(fp)[1:].transpose((1, 0))  # 将数据shape变成(脑区, 时间点)
        if roi_signal.shape[1] < min_time_point:
            except_list.append(f"{site_id}_{sub_id}")
            continue
        rois_list.append(roi_signal)
        label_list.append(label)
        sub_id_list.append(sub_id)
        site_list.append(site_id)
        age_list.append(age)
        sex_list.append(sex)
    print(f" - 数据缺失({len(none_list)}): \n{none_list}")
    print(f" - 数据排除: {except_list}")
    return {
        'rois': rois_list, 
        'labels': label_list,
        'ids': sub_id_list,
        'sites': site_list,
        'ages': age_list,
        'sexes': sex_list
    }

# 生成 ROC 数据
def create_roc_data(predict_list: list, label_list: list):
    # 按阈值生成 新的预测结果 
    tpr_list = list()
    fpr_list = list()
    for i in range(len(predict_list)):
        # 当前正样本阈值
        p = 1 - i * (1 / len(predict_list))

        # 取出 正样本概率，并做最大最小归一
        d = np.array(predict_list)[:, 1]
        d -= d.reshape(-1).min()
        d /= np.abs(d.reshape(-1).max())
        norm_predict_list = d.tolist()

        y_list = [1 if predict >= p else 0 for predict in norm_predict_list]

        # 统计 混淆矩阵
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(label_list)):
            l = label_list[i]
            y = y_list[i]
            if l == 0:
                if y == l:
                    tn += 1
                else:
                    fp += 1
            else:
                if y == l:
                    tp += 1
                else:
                    fn += 1
        # 记录 tpr fpr
        tpr_list.append(tp / (tp + fn + 1e-8))
        fpr_list.append(fp / (fp + tn + 1e-8))
    return fpr_list, tpr_list

# 画出 AUC/ROC曲线
def draw_ROC(fpr_list, tpr_list, is_show=True, is_save=False, save_path=''):
    roc_auc = auc(fpr_list, tpr_list)  # 准确率代表所有正确的占所有数据的比值

    plt.subplot(1, 1, 1)
    plt.plot(fpr_list, tpr_list, color='royalblue', lw=1, label='ROC curve')  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.title('AUC={:.3}'.format(roc_auc), y=1.)
    plt.legend(loc="lower right")

    # 修改文件名
    save_path = "{}_auc{:.4}.png".format(save_path[:-4], roc_auc)

    # 展示图片
    if is_show:
        plt.show()
    # 保存图片
    if is_save:
        plt.savefig(save_path)
    plt.close()  # 关闭窗口


# 多线程 
thread_info = None
thread_work_dict = dict()
idx_list = None
def muti_thread(data_list, func, workers_num=3, show=True):
    global thread_info
    global thread_work_dict
    global idx_list

    thread_info = [0 for _ in range(workers_num)]
    idx_list = [i for i in range(len(data_list))]


    # 线程处理函数
    def _thread_func(func, data_list, idx_list, thread_id):
        global thread_info

        for idx in idx_list:
            thread_info[thread_id] += 1
            d = data_list[idx]

            # 自定义处理函数
            res = func(d)
            if show:
                print(f"\r- 正在执行多线程任务: {thread_info} -> ({np.array(thread_info).sum()}/{len(data_list)})", end="")
            thread_work_dict[idx] = res



    # 多线程 计算
    per_len = math.ceil(len(data_list) / workers_num)
    # 分割数据
    cur_idx = 0
    tq = [list() for _ in range(workers_num)]
    for idx in idx_list:
        tq[cur_idx // per_len].append(idx)
        cur_idx += 1
    # 开启多线程
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(len(tq)):
            executor.submit(lambda a: _thread_func(*a), (func, data_list, tq[i], i))
        executor.shutdown()
            
    # 合并结果
    result_list = list()
    for i in range(len(thread_work_dict.keys())):
         result_list.append(thread_work_dict[i])
    
    if show:
        print("")
    return result_list


def draw(m1, m2):
        # 显示结果
        import matplotlib.pyplot as plt
        import torch

        if type(m1) == torch.Tensor:
            m1 = m1.detach().cpu().numpy()
        if type(m2) == torch.Tensor:
            m2 = m2.detach().cpu().numpy()

        # 使用matplotlib显示矩阵
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # 显示第一个矩阵
        cax1 = ax1.matshow(m1, cmap='RdBu_r', vmin=-1, vmax=1)
        fig.colorbar(cax1, ax=ax1)
        ax1.set_title('Pearson')

        # 显示第二个矩阵
        cax2 = ax2.matshow(m2, cmap='RdBu_r', vmin=-1, vmax=1)
        fig.colorbar(cax2, ax=ax2)
        ax2.set_title('Predict')

        plt.show()


def show_info():
    sites = np.unique(data_package1[keys[2]])
    site_dict = dict()
    for s in sites:
        site_dict[s] = [0, 0, 0, 0]
    for i in range(len(ids)):
        site = data_package1[keys[2]][i]
        label = data_package1[keys[1]][i]
        sex = data_package1[keys[5]][i]
        if label == 0:
            if sex == 0:
                site_dict[site][0] += 1
            else:
                site_dict[site][1] += 1
        else:
            if sex == 0:
                site_dict[site][2] += 1
            else:
                site_dict[site][3] += 1
    for k in site_dict.keys():
        res = "{}\t{}\t{}\t{}\t{}".format(k, site_dict[k][0], site_dict[k][1], site_dict[k][2], site_dict[k][3])
        print(res)
    exit()


def draw_2():
    # TODO
    predict1 = tfm_slice_features.transpose(0, 1).detach().cpu()
    predict2 = predict.transpose(0, 1).detach().cpu()

    new_features1 = torch.zeros(size=(data.shape[0], int(self.ori_features_lenght)))
    new_features2 = torch.zeros(size=(data.shape[0], int(self.ori_features_lenght)))
    for i in range(predict1.shape[0]):
        # 第一部分开始指针
        point_1 = 0
        # 第二部分开始指针
        point_2 = self.stride
        # 第二部分结束指针
        point_end = point_2 + (self.feature_lenght - self.stride) if point_2 + (self.feature_lenght - self.stride) < self.ori_features_lenght else self.ori_features_lenght
        for j in range(predict1.shape[1]):
            # 取出当前片段
            slice1 = predict1[i, j]
            slice2 = predict2[i, j]
            # 按照步长将序列分为两个部分
            part1_1 = slice1[: self.stride]
            part1_2 = slice1[self.stride:]
            part2_1 = slice2[: self.stride]
            part2_2 = slice2[self.stride:]
            # 判断是否刚开始
            if j != 0:
                part1_1 *= 0.5
                part2_1 *= 0.5
            # 判断是否结束
            if point_end + self.stride <= self.ori_features_lenght:
                part1_2 *= 0.5
                part2_2 *= 0.5
            # 粘入数据
            new_features1[i, point_1: point_2] += part1_1
            new_features2[i, point_1: point_2] += part2_1
            if point_end < self.ori_features_lenght:
                new_features1[i, point_2: point_end] += part1_2
                new_features2[i, point_2: point_end] += part2_2
            else:
                new_features1[i, point_2:] += part1_2
                new_features2[i, point_2:] += part2_2

            # 更新指针位置
            # 第一部分开始指针
            point_1 = point_2
            # 第二部分开始指针
            point_2 = point_end
            # 第二部分结束指针
            point_end = point_2 + (self.feature_lenght - self.stride) if point_2 + (self.feature_lenght - self.stride) < self.ori_features_lenght else self.ori_features_lenght
            # 结束跳出循环
            if point_end >= self.ori_features_lenght:
                break

    # 把特征还原成相关矩阵
    from WaFerFWv2.WaFerUtils import get_idx_feature2matrix
    xy_list = get_idx_feature2matrix([i for i in range(int(self.ori_features_lenght))], naoqu_num=self.feature_lenght)
    # 空相关矩阵
    ori_pcoor = torch.zeros((data.shape[0], self.feature_lenght, self.feature_lenght))
    out_pcoor1 = torch.zeros((data.shape[0], self.feature_lenght, self.feature_lenght))
    out_pcoor2 = torch.zeros((data.shape[0], self.feature_lenght, self.feature_lenght))
    # 还原
    for i in range(len(xy_list)):
        (x, y) = xy_list[i]
        ori_pcoor[:, x, y] = data[:, i]
        out_pcoor1[:, x, y] = new_features1[:, i]
        out_pcoor2[:, x, y] = new_features2[:, i]

        ori_pcoor[:, y, x] = data[:, i]
        out_pcoor1[:, y, x] = new_features1[:, i]
        out_pcoor2[:, y, x] = new_features2[:, i]

    # tensor转ndarray
    ori_pcoor = ori_pcoor.detach().cpu().numpy()
    out_pcoor1 = out_pcoor1.detach().cpu().numpy()
    out_pcoor2 = out_pcoor2.detach().cpu().numpy()

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    for i in range(ori_pcoor.shape[0]):
        matrix1 = ori_pcoor[i]
        matrix2 = out_pcoor1[i] - matrix1
        matrix3 = out_pcoor2[i] - matrix1

        # 创建蓝色橙色的colormap
        cmap = LinearSegmentedColormap.from_list('bright', ['blue', 'yellow'])

        # 标准化数据
        normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        matrix1_norm = normalize(matrix1)
        matrix2_norm = normalize(matrix2)
        matrix3_norm = normalize(matrix3)
        
        # 绘制矩阵图
        fig, axes = plt.subplots(1, 3)
        
        axes[0].imshow(matrix1_norm, cmap=cmap)
        axes[1].imshow(matrix2_norm, cmap=cmap)
        axes[2].imshow(matrix3_norm, cmap=cmap)
        
        # 设置坐标轴不可见
        for ax in axes:
            ax.set_axis_off()
        

        
        plt.show()
    exit()
