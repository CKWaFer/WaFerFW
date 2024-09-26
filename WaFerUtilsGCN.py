import torch
import nibabel as nib
import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def get_data_from_nii(file_path: str):
    if os.path.isfile(file_path):
        return nib.load(file_path).get_fdata()
    else:
        return None


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


def get_data_info_from_dpabi(dpabi_data_path: str,
                             root='./dataset/dpabi_data/',
                             folder_prefix_name=None,
                             key_words=None,
                             file_type='.mat'):
    """
    从DPABI的工作目录中，找出需要的数据文件，将不同被试的数据收集整理起来，
    并将对应的文件路径存储在本项目的./dataset/dpabi_data中
    :param dpabi_data_path: dpabi的工作目录(数据预处理的根目录)
    :param root: ./dataset/dpabi_data/
    :param folder_prefix_name:(只需要写前缀就行，会自动识别) 文件夹名，例如：'ROISignals'
    :param key_words: 需要的处理结果的关键字  例如：'ROISignals', 'ROICorrelation'
    :param file_type: 需要的文件类型 例如：.mat .nii
    """

    if key_words is None:  # 如果没有指定关键字，则默认使用以下的关键字
        key_words = ['ROISignals', 'ROICorrelation']
    if folder_prefix_name is None:  # 如果没有指定文件夹前缀，那么默认去找下述前缀的文件夹
        folder_prefix_name = 'ROISignals'

    # 从形参中获取DPABI数据预处理的根目录，如果末尾没有'/'，我们将补上'/'
    dpabi_data_path = dpabi_data_path if dpabi_data_path[-1] == '/' else dpabi_data_path + '/'
    results_path = dpabi_data_path + 'Results/'  # 将路径锁定到 Results文件夹中
    # 用模糊匹配的方式找出与我传入的前缀相同的文件夹(这是一个数组，会匹配所有符合要求的文件夹名)
    # glob产生的路径可能是使用'\'作为路径分隔符，要注意！
    results_paths = glob.glob(os.path.join(results_path, (folder_prefix_name + '*')))  # 目标文件夹

    # # # 我已经找到了预处理的目标文件所在的文件夹，我需要在Python项目中建立文件的位置映射 # # #

    if not os.path.exists(root):  # 查看存放数据路径的根文件夹，如果没有，我就新建立一个
        os.makedirs(root)  # 新建文件夹

    # 上面模糊匹配可能会存在很多个符合条件的文件夹，在这里，我将对找到的文件夹一一查看。
    # 找出里面的数据并分门别类地存放好
    for res_path in results_paths:
        # 通过文件夹路径，取出文件夹名的前缀
        # 策略：将文件路径以'/'作为分隔符进行分割，取出最后一个即文件夹名的前缀
        res_path = res_path.replace('\\', '/')  # Windows中使用'\'作为分隔符，会导致脚本错误，因此需要将'\'替换为'/'
        predo_folder = res_path.split("/")[-1]

        # 上面已经拿到了文件夹的前缀，这里我要在Python项目里建立对应的文件夹，来存放对应的文件路径
        predo_dpabi_data_path = root + predo_folder + '/'
        if not os.path.exists(predo_dpabi_data_path):  # 如果此文件夹不存在，那么我就新建一个
            os.makedirs(predo_dpabi_data_path)  # 新建文件夹

        # 因为可能会需要很多种文件，即关键词不止一个，所以这里我需要对传入的关键词进行一一检查，
        # 从而找出对应的文件路径
        for key_word in key_words:
            # 因为同一关键词对应的文件可能会出现不同的预训练结果，并且不同被试的命名也不同，
            # 所以这里我使用了模糊匹配的策略，去找到符合指定关键字的指定类别的文件
            # glob产生的路径可能是使用'\'作为路径分隔符，要注意！
            files_path = glob.glob(os.path.join(res_path, (key_word + '*' + file_type)))

            # 将文件进行排序
            # 原因：由于我们要保证同一被试所对应的不同数据的读取顺序要一一对应的，但在实际读取中，、
            # 会出现顺序上的差异，从而导致文件没办法对应上去，因此，我在这里做了排序
            # 策略：通过文件路径的分割，尝试找出 被试名称 部分，如果里面存在数字，那我们使用数字大小进行排序
            #     否则只能依赖于文件读取的顺序进行排序。
            files_path.sort(key=lambda x: int("".join(list(filter(str.isdigit, x.split(file_type)[0].split("_")[-1])))), reverse=False)  # 升序排列

            # 通过上面收集到的文件路径，一一遍历去对其进行处理
            for file_path in files_path:
                # 通过文件路径分割得到文件名
                # 策略：使用'/'将文件进行分割，我们认为最后一个部分就是文件名
                file_path = file_path.replace('\\', '/')  # Windows中使用'\'作为分隔符，会导致脚本错误，因此需要将'\'替换为'/'
                file_name = file_path.split("/")[-1]

                # 通过对文件名的分割得到 被试名
                # 策略：先使用'.'对文件名分割，我认为倒数第二部分即为去除掉后缀名的部分
                # 再使用'关键词+_'对文件名进行分割，我认为倒数第一个部分是 被试名
                per_name = file_name.split(".")[-2].split(key_word + "_")[-1]

                # 尝试用所匹配到的文件路径去读取文件，并验证文件中是否存在指定的关键字
                # 如果有，将返回具体数据；如果没有，则返回None
                d = None
                if file_type == '.mat':
                    d = get_data_from_mat(file_path=file_path, key_word=key_word)
                elif file_type == '.nii':
                    d = get_data_from_nii(file_path=file_path)
                else:
                    d = None

                if d is not None:  # 不等于None，即读取到了，即证明文件路径有效且与关键字相匹配
                    # 已经收集到了文件信息，我们要在Python项目中建立对应的文件夹进行存储
                    # 我们通过 上级目录+被试名的格式，拼接被试文件夹的路径。
                    bs_dpabi_data_path = predo_dpabi_data_path + per_name + '/'

                    # 通过上述路径尝试寻找文件夹，若没找到，我们就新建这个文件夹
                    if not os.path.exists(bs_dpabi_data_path):
                        os.makedirs(bs_dpabi_data_path)  # 新建文件夹

                    # 以'被试文件夹/关键词.txt'的命名策略拼接文件路径
                    bs_file_path = bs_dpabi_data_path + key_word + ".txt"
                    with open(bs_file_path, 'w') as f:  # 以写入的方式打开路径
                        f.write(file_path)  # 将找到的数据文件路径写入本文件中
                    f.close()  # 关闭文件
    pass


def get_data_path(dpabi_data_path: str,
                  predo_folder: str,
                  root='./dataset/dpabi_data/',
                  key_words=None,
                  file_type=None):
    """
    在Python项目中./dataset/dpabi_data中寻找get_data_info_from_dpabi()收集整理的数据路径，
    并将里面对应的路径读取，并以返回值的方式进行返回
    :param dpabi_data_path: dpabi的工作目录(数据预处理的根目录)
    :param predo_folder:预处理的文件夹名，要和读取的预处理文件夹名匹配（可以去DPABI预处理工作目录里看看具体的名字是什么）
    :param root: ./dataset/dpabi_data/
    :param key_words: 需要的处理结果的关键字  例如：'ROISignals', 'ROICorrelation'
    :param file_type: 需要的文件类型 例如：.mat .nii
    """

    if key_words is None:  # 如果没有指定关键字，则默认使用以下的关键字
        key_words = ['ROISignals', 'ROICorrelation']

    # 调用此函数，去收集被试数据，并存放到Python项目里的root路径下
    get_data_info_from_dpabi(dpabi_data_path=dpabi_data_path,
                             root=root,
                             folder_prefix_name=predo_folder.split('_')[0],  # 取出文件前缀
                             key_words=key_words,
                             file_type=file_type)

    # 从形参中获取root路径，如果末尾没有'/'，我们将补上'/'
    root = root if root[-1] == '/' else root + '/'
    # 将root和预处理文件夹名拼接，得到内层文件夹路径
    folder_path = root + predo_folder + '/'

    # 通过路径遍历，来找出此层所有的文件夹名（此处应该会存放 被试文件夹）
    # 注意：此处未对文件做过滤，可能会引发“把文件误判成文件夹”的错误，以后有时间记得做！
    bs_folders = os.listdir(folder_path)

    # 对文件名进行排序（理论上可以不做，因为文件已经分门别类的放好了）
    bs_folders.sort(key=lambda x: int("".join(list(filter(str.isdigit, x.split("_")[-1])))),
                    reverse=False)  # 升序排列

    # 这是我们要return的列表，里面会存放我们读取到的文件路径
    dpabi_data_path = []

    # 对上面找到的被试文件夹进行一一遍历
    for bs in bs_folders:
        # 通过 上层目录 + 被试文件夹名，拼接被试文件夹的路径
        bs_path = folder_path + bs + '/'
        # 通过文件遍历的方式，获取到所有的用于存储数据文件路径的文件名
        files_name = os.listdir(bs_path)
        per_data = []  # 用于记录对应的每个被试的文件路径
        for file_name in files_name:
            with open(bs_path + file_name) as f:  # 打开文件，读取其文件路径
                # 此列表起解释作用
                # [0]: 说明了文件是什么数据
                # [1]: 文件路径
                list_info = list()

                list_info.append(file_name.split('.txt')[0])  # 存入文件说明
                list_info.append(f.read())  # 存入文件路径
            f.close()  # 关闭文件
            # 将上面的'文件说明+文件路径'列表添加到每个被试的列表中
            per_data.append(list_info)
        # 将每个被试的文件路径添加到总列表中
        dpabi_data_path.append(per_data)
    return dpabi_data_path  # 返回总列表


def save_data_from_dpabi_to_npy(dpabi_data_path: str,
                                predo_folder: str,
                                root='./dataset/',
                                key_words=None,
                                p=-1,
                                topk_rate=0.3,
                                is_self_con=True,
                                file_type='.mat'):
    """
       读取DPABI预处理的数据，将其转换成 结点、边、边权重、标签，
       并以.npy的形式保存在 ./dataset/dpabi_data/graph中
       其中，会以0, 1, 2, ...去做文件夹的命名，不去关注被试名、预处理过程和预处理数据种类
       :param dpabi_data_path: dpabi的工作目录(数据预处理的根目录)
       :param predo_folder:预处理的文件夹名，要和读取的预处理文件夹名匹配（可以去DPABI预处理工作目录里看看具体的名字是什么）
       :param root: ./dataset/dpabi_data/
       :param key_words: 需要的处理结果的关键字  例如：'ROISignals', 'ROICorrelation'
       :param p: 皮尔逊相关的阈值，如果为 -1<p<=1,则卡阈值，否则不卡阈值
       :param topk_rate: 根据相关性，选取k(比例)的打上1的标签，其余为0
       :param is_self_con: 是否考虑自连接
       :param file_type: 需要的文件类型 例如：.mat .nii
       """

    print("> [WaFer提示]正在寻找DPABI数据...")
    data_path = get_data_path(dpabi_data_path=dpabi_data_path,
                              predo_folder=predo_folder,
                              root=root + 'dpabi_data/',
                              key_words=key_words,
                              file_type=file_type)
    print("> -完成！")
    if file_type == '.mat':
        print("> [WaFer提示]正在将数据转换成图...")
        for i in range(len(data_path)):
            print("> > [WaFer提示]正在将第{}份数据转换成图...".format(i + 1))
            roic = get_data_from_mat(data_path[i][0][1], key_word=data_path[i][0][0])
            rois = get_data_from_mat(data_path[i][1][1], key_word=data_path[i][1][0])
            roic = torch.from_numpy(np.array(roic))
            rois = torch.from_numpy(np.array(rois))
            x = rois.T

            if (p > -1) and (p <= 1):
                roic[torch.where(roic < p)] = 0

            if not is_self_con:
                diag = torch.diag(roic)
                a_diag = torch.diag_embed(diag)  # 仅对角线为1
                roic = roic - a_diag

            G = nx.from_numpy_matrix(roic.numpy())

            edges_index = np.array(G.edges).T
            edges_attr = np.array([e[1][2]['weight'] for e in enumerate(G.edges(data=True))]).T

            # 生成y
            degree = np.array(G.degree())[:, 1]
            # 边权重在 前 topk_rate * x.size(0)的数量的，我认为是1，否则是0
            tk = torch.topk(torch.from_numpy(degree), int(topk_rate * x.size(0)))[1].numpy()
            y = np.zeros(x.size(0)).astype(np.int64)
            y[tk] = 1

            save_root = root + 'graph/' + predo_folder + '/' + str(i) + '/'
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            np.save(save_root + '/' + 'x' + '.npy', x)
            np.save(save_root + '/' + 'edges_index' + '.npy', edges_index)
            np.save(save_root + '/' + 'edges_attr' + '.npy', edges_attr)
            np.save(save_root + '/' + 'y' + '.npy', y)
            print("> > -完成！")
        print("> -完成！")
    pass


# save_data_from_dpabi_to_npy(dpabi_data_path='D:/WorkSpace/Dataset/DemoData/demo/',
#                             predo_folder='ROISignals_FunImgARCSDFB',
#                             key_words=['ROISignals', 'ROICorrelation'],
#                             file_type='.mat')

# save_data_from_dpabi_to_npy(dpabi_data_path='D:/WorkSpace/Dataset/DemoData/demo/',
#                             predo_folder='ALFF_FunImgARCSD',
#                             key_words=['ALFFMap', 'mALFFMap', 'zALFFMap'],
#                             file_type='.nii')


def get_g_dataset(graph_root: str):
    # 从形参中获取graph_root路径，如果末尾没有'/'，我们将补上'/'
    graph_root = graph_root if graph_root[-1] == '/' else graph_root + '/'
    data = list()
    if not os.path.isdir(graph_root):
        return None

    bs_folders_name = os.listdir(graph_root)
    gs_path = [graph_root + bs_name + '/' for bs_name in bs_folders_name]
    if gs_path is []:
        return None
    min_tp = 0
    for i in range(len(gs_path)):
        d = list()
        g_path = gs_path[i]

        """
        # x: 结点 需要包括特征  例：10个结点，每个有5个特征，则x.shape=[10, 5]
        # edges_index: 边  shape=[2, 边数量]   C为边的数量
        # edges_attr: 边的属性  shape=[边数量，边特征数]
        # y: 标签 shape=[有标签的结点数量]
        """
        x, edges_index, edges_attr, y = get_graph_data_from_npy(graph_root=g_path)

        # 转np.array
        x = np.array(x).squeeze()
        edges_index = np.array(edges_index).squeeze()
        edges_attr = np.array(edges_attr).squeeze()
        y = np.array(y).squeeze()

        # 转Tensor
        x = torch.from_numpy(x).float()
        edges_index = torch.from_numpy(edges_index).long()
        edges_attr = torch.from_numpy(edges_attr).float()
        y = torch.from_numpy(y).long()

        # 找出最小的时间点数量
        if (x.size(1) < min_tp) or (min_tp == 0):
            min_tp = x.size(1)

        d.append(x)
        d.append(edges_index)
        d.append(edges_attr)
        d.append(y)
        data.append(d)

    pyg_dataset = get_dataset_from_data(data=data, min_tp=min_tp)

    return pyg_dataset


def get_dataset_from_data(data: list,
                          min_tp=None  # 最短的时间点，用来校正数据的时间点长度
                          ):
    """
    data的结构需要满足：
        d.append(x)
        d.append(edges_index)
        d.append(edges_attr)
        d.append(y)

        data.append(d)
    """
    pyg_d = list()
    for d in data:
        if min_tp is not None:
            # 由于时间点不同，此处需要校正图像时间点
            diff = d[0].size(1) - min_tp
            d[0] = d[0][:, diff:]

        x = torch.from_numpy(np.array(d[0])).float()
        ei = torch.from_numpy(np.array(d[1])).long()
        ea = torch.from_numpy(np.array(d[2])).float()
        y = torch.from_numpy(np.array(d[3])).long()

        pd = pyg_Data(x=x,
                      edge_index=ei,
                      edge_attr=ea,
                      y=y)

        pyg_d.append(pd)

    dataset = WaFerGDataset(data=pyg_d, is_train=True)

    return dataset


def save_graph_A(save_root: str, a: np.array, is_train: bool):
    dir_path = save_root + '/graph-a/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    t = None
    if is_train is True:
        t = '-train'
    else:
        t = '-test'
    np.save(dir_path + 'a' + t + '.npy', a)


def load_graph_A(path: str):
    a = np.load(path)
    return np.array(a)


def get_graph_data_from_npy(graph_root: str):
    x = np.load(graph_root + 'x.npy')  # 结点 需要包括特征  例：10个结点，每个有5个特征，则x.shape=[10, 5]
    edges_index = np.load(graph_root + 'edges_index.npy')  # 边  shape=[2, 边数量]   C为边的数量
    edges_attr = np.load(graph_root + 'edges_attr.npy')  # 边的属性  shape=[边数量，边特征数]
    y = np.load(graph_root + 'y.npy')  # 标签 shape=[有标签的结点数量]

    return x, edges_index, edges_attr, y


def show_graph(graph_root=None, g=None):
    if graph_root is not None:
        x, ei, ea, y = get_graph_data_from_npy(graph_root=graph_root)
        g = nx.Graph()

        for i in range(x.shape[0]):
            g.add_node(i, weight=x[i, :])
        g.add_edges_from(ei.T)
        # g.add_weighted_edges_from(ea)
    nx.draw(g, with_labels=True)
    plt.show()


# 通过相关矩阵获取图数据(边和边权重)
def get_edges_by_A(A, p: tuple=(float('-inf'), float('inf'))):
    is_batch = False  # 默认不是批次输入
    # 判断是否为批次输入
    if len(A.shape) > 2:
        is_batch = True
    if type(A) is np.ndarray:  # Numpy数组
        # 判断单个还是批次
        if not is_batch:
            A = np.expand_dims(A, axis=0)
        # 获取边 (12431889, 3)  (边的数量, [被试编号, 起点, 终点])
        # edges_index = np.argwhere((float('-inf') < A < float('inf')))
        edges_index = np.argwhere(A != np.nan)
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
        # 卡相关性阈值(去掉不满足相关性条件的边)
        ori_shape = edges.shape  # 记录原始尺寸
        edges = edges.reshape(-1, 3)  # 拉成2维，便于卡阈值
        # 找出不符合的值
        bad_idx_min = edges[:, -1] < p[0]  # 小于下限阈值
        bad_idx_max = edges[:, -1] > p[1]  # 大于上限阈值
        edges[bad_idx_min, -1] = 0.  # 将其置成0
        edges[bad_idx_max, -1] = 0.  # 将其置成0
        edges = edges.reshape(ori_shape)  # 还原尺寸
    else:  # Tensor张量
        # 判断单个还是批次
        if not is_batch:
            A = torch.unsqueeze(A, dim=0)
        # 获取边 (12431889, 3)  (边的总数量, [本条边从属的图编号, 起点, 终点])
        edges_index = torch.from_numpy(np.argwhere(A.detach().cpu().numpy() != np.nan)).to(A.device)

        # 获取边权重(皮尔逊相关) (12431889, 1)  (边的数量, [权重])
        edges_weight = torch.reshape(A, (-1, 1))
        # 拼接数据 (12431889, 4)  (边的数量, [本条边从属的图编号, 起点, 终点, 边权重])
        edges = torch.cat((edges_index, edges_weight),dim=1)
        # 计算 图的数量
        bs_num = torch.unique(edges[:, 0]).shape[0]
        # 按从属关系分割数据
        edges = torch.reshape(edges, (bs_num, -1, 4))
        # 去除被试标记位
        edges = edges[:, :, 1:]
        # 卡相关性阈值(去掉不满足相关性条件的边)
        ori_shape = edges.shape  # 记录原始尺寸
        edges = edges.reshape(-1, 3)  # 拉成2维，便于卡阈值
        # 找出不符合的值
        bad_idx_min = edges[:, -1] < p[0]  # 小于下限阈值
        bad_idx_max = edges[:, -1] > p[1]  # 大于上限阈值
        edges[bad_idx_min, -1] = 0.  # 将其置成0
        edges[bad_idx_max, -1] = 0.  # 将其置成0
        edges = edges.reshape(ori_shape)  # 还原尺寸

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
    # 判断单个还是批次
    if edges_index.shape[0] == 1 and edges_weight.shape[0] == 1:
        edges_index = edges_index[0]
        edges_weight = edges_weight[0]
    return edges_index, edges_weight
