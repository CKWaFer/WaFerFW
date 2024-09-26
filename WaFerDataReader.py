import math
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import WaFerUtilsGCN as wfugcn
import WaFerUtils as wfdu
from data_loader import WaFerCNNDataset
import networkx as nx
# import SimpleITK as sitk
# import open3d as o3d
import warnings
warnings.filterwarnings('ignore')


def get_data_from_mat(file_path: str, key_word: str):
    """
    通过具体的文件路径来读取.mat文件，并返回
    :param file_path: .mat的文件路径
    :param key_word: .mat文件的标题(在MATLAB里能看到)
    :return: 返回的数据
    """
    data = sio.loadmat(file_path)  # 读取文件
    if key_word in data.keys():  # 检查文件标题是否匹配，若不匹配，则不返回
        return data[key_word]  # 返回对应的数据
    else:
        return None  # 不返回


def data_reader(json_root: str,
                key_words: list,
                min_len=None,
                file_type='file0',
                new_size=None,
                time_point_dim=-1
                ):
    json_key_words = key_words
    json_key_words.append(file_type)
    bss_info = wfdu.get_bs_info_from_json(json_root=json_root, key_words=json_key_words)
    data_package = dict()
    for bs_info in bss_info:
        path_roi_s = bs_info[file_type]

        if path_roi_s[-4:] == '.mat':
            roi_s = wfugcn.get_data_from_mat(path_roi_s, key_word='ROISignals')
            if time_point_dim != -1:
                roi_s = np.array(roi_s).transpose((-1, -2))  # 将数据shape变成(脑区, 时间点)
            try:
                if min_len is not None:
                    if roi_s.shape[-1] < min_len:
                        raise BaseException
            except BaseException:
                print("{}被排除. 原因：长度不足{}".format(path_roi_s, min_len))
                roi_s = None
        elif path_roi_s[-3:] == '.1D':
            try:
                roi_s = np.loadtxt(path_roi_s)[1:].transpose((1, 0))  # 将数据shape变成(脑区, 时间点)
                if min_len is not None:
                    if roi_s.shape[-1] < min_len:
                        raise BaseException
            except BaseException:
                print("{}被排除. 原因：长度不足{}".format(path_roi_s, min_len))
                roi_s = None
        elif path_roi_s[-7:] == '.nii.gz':
            try:
                roi_s = read_nii(path=path_roi_s, new_size=new_size)  # 将数据shape变成(脑区, 时间点)
                if min_len is not None and len(roi_s.shape) == 4:
                    if roi_s.shape[-1] < min_len:
                        raise BaseException
            except BaseException:
                print("{}被排除. 原因：长度不足{}".format(path_roi_s, min_len))
                roi_s = None

        if roi_s is not None:
            if not "data" in data_package.keys():
                data_package["data"] = list()
            data_package["data"].append(roi_s.astype(np.float32))

            for key in bs_info.keys():
                if key is file_type:
                    continue
                if not key in data_package.keys():
                    data_package[key] = list()
                data_package[key].append(bs_info[key])

    return data_package


# 找出被试中最短时间点的长度，并将所有被试时间点统一成这个长度
def same_time_point(roiss: list, min_len=None):
    if min_len is None:
        l = dict()
        for i in range(len(roiss)):
            cur_len = roiss[i].shape[-1]
            if not cur_len in l.keys():
                l[cur_len] = 0
            l[cur_len] += 1
            if min_len is None:
                min_len = roiss[i].shape[-1]
                continue
            if min_len > roiss[i].shape[-1]:
                min_len = roiss[i].shape[-1]
    print(f" - 时间点截取长度：{min_len}")
    res_roiss = list()
    for rois in roiss:
        rois = rois[:, : min_len]  # 统一所有的被试的时间点长度
        res_roiss.append(rois)
    return res_roiss, min_len


# 单个被试的 ROI信号按滑动窗口切分，并生成图数据
def roi_to_graph(rois: np.ndarray,
                 slice_num: int,  # 被试的时间点一共能切多少片
                 window_width: int,  # 滑动窗口的宽度
                 window_stride: int,  # 滑动窗口的移动步长
                 pearson_threshold: float  # 皮尔逊阈值
                 ):
    data_list = list()
    for i in range(slice_num):
        # 获得 当前的ROI切片
        cur_rois = rois[:, i * window_stride: i * window_stride + window_width]
        # 计算 皮尔逊相关矩阵(邻接矩阵)
        cur_roic = wfdu.get_pearson_from_roi(rois=cur_rois)
        # 设置 皮尔逊阈值(消除小于阈值的图连接)
        cur_roic[cur_roic < pearson_threshold] = 0
        # 通过 邻接矩阵 构图
        g = nx.from_numpy_matrix(A=cur_roic)
        # 通过图 获取边信息
        edges_index = np.array(g.edges).T  # 边
        edges_attr = np.array([g.edges[tuple_info]['weight'] for tuple_info in g.edges])  # 边属性，此处为皮尔逊系数
        # 获取 结点属性
        assert cur_rois.shape[0] == len(g.nodes)  # 断言：结点数和脑区数相同
        x = cur_rois
        # 获取 结点标签(此处的标签无实际意义,仅为构图所需)
        y = np.zeros(x.shape[0])
        # 将 结点、边、边属性和结点标签打包成数据包(列表)
        data = [x, edges_index, edges_attr, y]
        # 将数据包添加到数据列表
        data_list.append(data)
    return data_list


# 通过Json文件 获取 ROI数据，并生成数据集(Json基于 WaFerDpabiUtils.create_train_val_by_mat_and_csv_json()生成)
def get_roi_dataset(json_root: str,
                    json_key_words: list,
                    is_train: bool,
                    min_len=None
                    ):
    # 通过json文件读取被试的ROI数据和标签
    roiss, ys, sites, id = data_reader(json_root=json_root,
                                       key_words=json_key_words,
                                       min_len=min_len)
    # 找出被试中最短时间点的长度，并将所有被试时间点统一成这个长度
    roiss, min_len = same_time_point(roiss=roiss, min_len=min_len)
    # 将数据封装成 WaFerCNNDataset(继承于Dataset)
    dataset = WaFerCNNDataset(data=roiss, labels=ys, is_train=is_train)
    return dataset


# 通过Json文件 获取 ROI数据，并生成打包成npy
def get_roi_npy(json_root: str,
                json_key_words: list,
                is_filter_len=True,
                min_len=None,
                file_type="file0",
                new_size=None,
                time_point_dim=-1
                ):
    # 通过json文件读取被试的ROI数据和标签
    data_package = data_reader(json_root=json_root,
                               key_words=json_key_words,
                               min_len=min_len,
                               file_type=file_type,
                               new_size=new_size,
                               time_point_dim=time_point_dim
                               )
    # 找出被试中最短时间点的长度，并将所有被试时间点统一成这个长度
    if is_filter_len:
        data_package["data"], min_len = same_time_point(roiss=data_package["data"],
                                                        min_len=min_len)
    return data_package


# 将ROI数据集转换成图数据集
def rois_to_graphs(roiss: list,
                   window_width=20,  # 滑动窗口的宽度
                   window_stride=2,  # 滑动窗口的移动步长
                   pearson_threshold=0.6454,  # 皮尔逊阈值
                   min_len=None
                   ):
    # 找出被试中最短时间点的长度，并将所有被试时间点统一成这个长度
    roiss, min_len = same_time_point(roiss=roiss, min_len=min_len)
    # 计算 被试的时间点一共能切多少片
    slice_num = math.floor((min_len - window_width) / window_stride)  # 用math.floor 舍弃最后一个不完整的切片
    # 对时间点进行滑动窗口采样，逐被试进行基于皮尔逊相关构图
    bss_g_dataset = list()  # 所有被试滑动窗口前片后的图数据集
    for rois in roiss:
        # 对rois切片，生成图数据
        data_list = roi_to_graph(rois=rois,
                                 slice_num=slice_num,
                                 window_width=window_width,
                                 window_stride=window_stride,
                                 pearson_threshold=pearson_threshold)
        bss_g_dataset.append(data_list)
    return bss_g_dataset


# 通过相关矩阵获取图数据(边和边权重)
def get_edges_by_A(A, p=float('-inf')):
    if type(A) is np.ndarray:  # Numpy数组
        # 获取边 (12431889, 3)  (边的数量, [被试编号, 起点, 终点])
        edges_index = np.argwhere(A > float('-inf'))
        # 获取边权重(皮尔逊相关) (12431889, 1)  (边的数量, [权重])
        edges_weight = np.reshape(A, (-1, 1))
        # 拼接数据 (12431889, 4)  (边的数量, [被试编号, 起点, 终点, 边权重])
        edges = np.concatenate((edges_index, edges_weight), axis=-1)
        # 计算被试数
        bs_num = np.unique(edges[:, 0]).shape[0]
        # 按被试分割数据
        edges = np.reshape(edges, (bs_num, -1, 4))
        # 去除被试标记位
        edges = edges[:, :, 1:]
        # # 卡相关性阈值(去掉不满足相关性条件的边)
        # edges = edges[np.where(edges[:, -1] > p)[0]]
    else:  # Tensor张量
        # 获取边 (12431889, 3)  (边的数量, [被试编号, 起点, 终点])
        edges_index = torch.argwhere(A > float('-inf'))
        # 获取边权重(皮尔逊相关) (12431889, 1)  (边的数量, [权重])
        edges_weight = torch.reshape(A, (-1, 1))
        # 拼接数据 (12431889, 4)  (边的数量, [被试编号, 起点, 终点, 边权重])
        edges = torch.concatenate((edges_index, edges_weight),dim=1)
        # 计算被试数
        bs_num = torch.unique(edges[:, 0]).shape[0]
        # 按被试分割数据
        edges = torch.reshape(edges, (bs_num, -1, 4))
        # 去除被试标记位
        edges = edges[:, :, 1:]
        # # 卡相关性阈值(去掉不满足相关性条件的边)
        # edges = edges[torch.where(edges[:, :, -1] > p)[0]]

    # 读取方法：def edges_to_index_weight(edges)
    return edges  # 边信息 (12431889, 4)  (边的数量, [被试编号, 起点, 终点, 边权重])


# 边信息 转 边和边权重
def edges_to_index_weight(edges):
    if type(edges) is np.ndarray:  # Numpy数组
        # 取出边和边权重
        edges_index = edges[:, :-1].astype(np.int64)
        edges_weight = edges[:, -1:].astype(np.float32)
        # 调整尺寸
        edges_index = np.transpose(edges_index, 2, 1)
        edges_weight = np.squeeze(edges_weight, axis=-1)
    else:  # Tensor张量
        # 取出边和边权重
        edges_index = edges[:, :, :-1].long()
        edges_weight = edges[:, :, -1:].float()
        # 调整尺寸
        edges_index = torch.transpose(edges_index, 2, 1)
        edges_weight = torch.squeeze(edges_weight, dim=-1)
    return edges_index, edges_weight


def mixup_data(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_nodes, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):

    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()

    y_0 = 1 - y_1

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


# 将数据和标签进行打乱
def shuffleIt(data_package, sub_id_list=None):
    keys = list(data_package.keys())
    if sub_id_list is None:
        order = np.random.permutation(np.array(range(0, len(data_package[keys[0]]), 1)))
    else:
        order = []
        for sub_id in sub_id_list:
            if sub_id in data_package['ids']:
                idx = data_package['ids'].index(sub_id_list)
                order.append(idx)
            elif f"{int(sub_id)}" in data_package['ids']:
                idx = data_package['ids'].index(f"{int(sub_id)}")
                order.append(idx)
            elif f"00{sub_id}" in data_package['ids']:
                idx = data_package['ids'].index(f"00{sub_id}")
                order.append(idx)
            else:
                raise BaseException("sub_id列表不匹配")
            
    shuffle_data_package = dict()
    for i in order:
        for key in data_package.keys():
            if not key in shuffle_data_package.keys():
                shuffle_data_package[key] = list()
            shuffle_data_package[key].append(data_package[key][i])

    sub_id_list = [data_package['ids'][i] for i in order]

    return shuffle_data_package, sub_id_list


# # 读取 .nii
# def read_nii(path: str, new_size=None):
#     img = sitk.ReadImage(path)
#     if new_size is not None:
#         img = wfdu.resize_image_itk(itkimage=img, newSize=new_size)
#     img = sitk.GetArrayFromImage(img)

#     return img

# 读取.arff文件
def read_arff(file_path, is2numpy=False):
    with open(file_path, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    if is2numpy is True:
        df = df.to_numpy()
    return df


# 读取从ABIDE下载的预处理数据
def Reader4ABIDE(data_root,
                 csv_path,
                 name="abide",
                 info="default",
                 save_root='../dataset/',
                 csv_id_keyword='SUB_ID',
                 csv_keywords=['DX_GROUP', 'SITE_ID', 'AGE_AT_SCAN'],
                 json_key_words=['id', 'DX_GROUP', 'SITE_ID', 'AGE_AT_SCAN', 'file0'],
                 min_len=None,
                 time_point_dim=-1,
                 is_filter_len=False):
    save_root = f'{save_root}{name}/{info}/'
    # 以 csv为准，从Dpabi结果文件夹中找到每一个被试，并把其相关信息封装到json文件中
    # 打乱数据集，并按照要求划分训练集和测试集(按照比例划分)
    wfdu.create_train_val_by_data_and_csv_json(data_root=data_root,
                                               csv_path=csv_path,
                                               save_root=save_root,
                                               csv_id_keyword=csv_id_keyword,
                                               csv_keywords=csv_keywords,
                                               train_rate=1)
    # 训练集
    data_package = get_roi_npy(json_root=f'{save_root}train',
                               json_key_words=json_key_words,
                               min_len=min_len,
                               file_type='file0',
                               time_point_dim=time_point_dim,
                               is_filter_len=is_filter_len)
    return data_package


# def fun1(naoqu_list, template_path: str):
#     # 选择TopK
#     naoqu_list = naoqu_list[0:30]
#     # 读取脑区模板
#     points_value = read_nii(template_path)
#     # points_value = wfdr.read_nii(temp_path, new_size=(610, 730, 610))
#     # 生成点云坐标
#     nq = np.unique(points_value)
#     # 生成颜色
#     # color = np.random.rand(len(nq), 3)
#     default_color = np.array([0.74509, 0.74509, 0.74509])
#     acti_color = np.array([1, 0, 0])

#     point_cloud_list = list()
#     for i in range(len(nq)):
#         print(f"\r - 正在渲染3D点云，本步骤可能非常耗时，请耐心等待...({i}/{len(nq) - 1})", end='')
#         if i == 0:
#             continue
#         # 取出当前脑区的坐标
#         points = np.argwhere(points_value == i)
#         # 实例化3D点云对象
#         point_cloud = o3d.geometry.PointCloud()
#         # 构建3D点云
#         point_cloud.points = o3d.utility.Vector3dVector(points)
#         # 设置颜色
#         if i in naoqu_list:
#             point_cloud.paint_uniform_color(list(acti_color))  # 给所有点云一个统一的颜色
#         else:
#             point_cloud.paint_uniform_color(list(default_color))  # 给所有点云一个统一的颜色
#         # 将当前点云添加到列表
#         point_cloud_list.append(point_cloud)
#     print(" - 渲染完成！")
#     o3d.visualization.draw_geometries(point_cloud_list)
#     # o3d.visualization.draw_geometries(point_cloud_list)
#     # o3d.io.write_point_cloud("example_point_cloud.pcd", point_cloud_list)


# # 数据做脑区模板映射成ROI
# def brain2roi_signal(data, tp_path, is_skip_0=True):
#     tp = read_nii(tp_path)
#     roi_ids = np.unique(tp)
#     if len(data.shape) == 3:
#         data = np.expand_dims(data, axis=0)
#     rois = None
#     for roi_id in roi_ids:
#         # 是否跳过 0号
#         if is_skip_0:
#             if roi_id == 0:
#                 continue
#         x, y, z = np.where(tp == roi_id)
#         roi = np.mean(data[:, x, y, z], axis=-1)
#         roi = np.expand_dims(roi, axis=1)
#         if rois is None:
#             rois = roi
#         else:
#             rois = np.concatenate([rois, roi], axis=-1)

#     return rois
