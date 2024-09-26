import tqdm
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import SimpleITK as sitk
import WaFerUtils as wfu
import  wafer_utils as wafer_utils
from torch.utils import data as data_utils


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
    print(f" - 数据缺失: {none_list}")
    return fc_list

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


# 加载 ABIDE_PREC 的 .1D和nii.gz
def read_abide_prec_by_txt(data_dir: str, txt_path: str, suffix: str='func_preproc', f_type='.1D', brain_mask_path="None"):
    # 打开文件
    lines = None
    with open(txt_path, 'r') as file:
        # 读取所有行并存储到列表中
        lines = file.readlines()
    # 移除表头
    lines = lines[1:]

    # 不存在文件的列表，排除列表
    none_list = list()
    except_list = list()
    data_package = list()
    # 文件字典
    fns_dict = {}
    for fn in os.listdir(data_dir):
        sub_id = fn.split("_")[-3]
        fns_dict[sub_id] = fn
    for line in lines:
        sub_id, site, label, age, sex = line.split("\t")
        
        label = int(label)
        age = float(age)
        sex = int(sex)

        # 校正 标签和性别 到 0和1
        label -= 1
        sex -= 1
        
        if sub_id in fns_dict.keys():
            fn = fns_dict[sub_id]
            fp = os.path.join(data_dir, fn)
            data_package.append([fp, label, sub_id, site, age, sex])
        else:
            none_list.append(f"{site}_{sub_id}")
            fp = None

    # 对 单个数据 的处理逻辑函数
    def _get_data(data_package, f_type=f_type, brain_mask_path=brain_mask_path):
        if f_type == '.1D':
            data_package[0] = np.loadtxt(data_package[0])[1:].transpose((1, 0))  # 将数据shape变成(脑区, 时间点)
        elif f_type == '.nii.gz':
            # 读取 .nii.gz
            img = sitk.ReadImage(data_package[0])
            # 获取数据
            data = sitk.GetArrayFromImage(img)  # 获取数据，获取到的数据格式是np.ndarray

            """ .nii.gz必须 完成 ROISignal提取, 否则将过大导致内存溢出 """
            # TODO 按照脑区模板 提取 ROI
            mask = sitk.GetArrayFromImage(sitk.ReadImage(brain_mask_path))
            # 获取脑区类别
            brain_blocks = np.unique(mask).tolist()
            # ROI列表
            rois_list = list()
            # 获取 ROI数据
            for bb in brain_blocks:
                idx = np.where(mask == bb)
                rois = list()
                other_mean = 0
                for i in range(len(data)):
                    roi = data[i]
                    roi = roi[idx]

                    roi = np.mean(roi)  # 均值
                    # roi = np.median(roi)  # 中位数
                    if bb == 0:
                        other_mean = roi
                        continue

                    roi -= other_mean
                    rois.append(roi)
                if bb == 0:
                    continue

                rois_list.append(rois)
            # ROI Signal           
            data_package[0] = np.array(rois_list)
        else:
            raise BaseException(f"暂不支持 {f_type} 格式文件的读取")

        return data_package

    # 多线程加载
    res = wafer_utils.muti_thread(data_list=data_package, func=_get_data, workers_num=8)

    # 整理数据
    data_list = list()
    label_list = list()
    sub_id_list = list()
    site_list = list()
    age_list = list()
    sex_list = list()
    for r in res:
        data_list.append(r[0])
        label_list.append(r[1])
        sub_id_list.append(r[2])
        site_list.append(r[3])
        age_list .append(r[4])
        sex_list.append(r[5])
    print(f" - 数据缺失({len(none_list)}): \n{none_list}")
    print(f" - 数据排除({len(except_list)}): \n{except_list}")
    return {
        'data': data_list, 
        'labels': label_list,
        'ids': sub_id_list,
        'sites': site_list,
        'ages': age_list,
        'sexes': sex_list
    }

