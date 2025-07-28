from utils import *

import os
import numpy as np
from matplotlib import pyplot as plt
import math
import utils
import cityhash
import struct
from sklearn.manifold import TSNE
import umap.umap_ as umap
import time
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from matplotlib.patches import Patch


def generate_labels_with_ground_truth(data, ground_truth_data):
    n_base = data.shape[0]
    n_query = ground_truth_data.shape[0]

    labels = -1 * np.ones(n_base, dtype=np.int32)
    for qid in range(n_query):
        for idx in ground_truth_data[qid]:
            if labels[idx] == -1:
                labels[idx] = qid

    return labels


def print_hnsw_outdegree_distribute():
    row_points = [
        # streamV13
        # np.array([
        #   [19, 2], [21, 2], [25, 4], [24, 1], [22, 5], [23, 7], [28, 27], [26, 8], [27, 13], [31, 71], [29, 38], [30, 52], [47, 154865], [36, 395204], [38, 304035], [32, 1176022], [34, 575078], [48, 148352], [41, 227830], [42, 208252], [40, 252883], [37, 343741], [58, 150249], [35, 465449], [56, 139838], [33, 758925], [60, 167800], [59, 156692], [54, 135756], [63, 268272], [53, 135985], [17, 1], [64, 1598447], [62, 209639], [39, 274226], [61, 183742], [45, 171019], [52, 136495], [57, 144531], [49, 143439], [55, 137837], [43, 193830], [46, 162043], [51, 137187], [50, 140609], [44, 181497],
        # ]),
        # np.array([
        #   [1, 11511], [2, 19658], [3, 31414], [6, 77653], [5, 60859], [4, 45306], [10, 152665], [7, 95503], [12, 185649], [13, 200986], [16, 231673], [17, 238802], [25, 239575], [19, 247038], [8, 114968], [11, 170606], [55, 93703], [42, 156829], [32, 239234], [56, 90508], [9, 133925], [28, 227264], [30, 217322], [52, 105087], [29, 222419], [15, 223802], [62, 74036], [61, 75577], [14, 213024], [50, 113368], [37, 193615], [41, 163741], [64, 112943], [31, 211875], [59, 80745], [36, 201450], [35, 209946], [58, 84151], [21, 248531], [44, 144319], [63, 72226], [46, 133107], [23, 245802], [39, 177539], [34, 221170], [33, 230173], [27, 231994], [47, 128387], [43, 150830], [51, 109571], [45, 138749], [54, 97543], [53, 100907], [20, 247755], [22, 247464], [57, 86700], [24, 243465], [40, 170622], [60, 77989], [18, 244151], [49, 118144], [26, 235295], [48, 123875], [38, 185262]
        # ])

        # sift1m
        np.array([
            [23, 1], [28, 5], [24, 1], [29, 3], [30, 8], [26, 3], [25, 2], [27, 8], [31, 21], [32, 37200], [33, 47339], [35, 43109], [34, 46312], [37, 36353], [39, 30821], [43, 23800], [38, 33549], [44, 22888], [36, 39513], [55, 18258], [54, 18223], [60, 20483], [
                59, 19924], [63, 32600], [40, 28942], [58, 19304], [53, 18211], [64, 170227], [41, 27215], [61, 22054], [62, 24875], [57, 18750], [45, 21725], [56, 18354], [48, 19421], [52, 18391], [50, 18451], [49, 19081], [47, 19800], [42, 25699], [51, 18194], [46, 20882]
        ]),
        np.array([
            [2, 6], [1, 1], [4, 93], [3, 25], [5, 213], [6, 474], [8, 1689], [9, 2579], [7, 935], [10, 3921], [11, 5327], [12, 6952], [23, 25252], [21, 23158], [53, 12356], [13, 8932], [63, 7769], [40, 23004], [44, 19092], [36, 27076], [59, 9274], [51, 13630], [39, 24151], [56, 10845], [64, 7242], [17, 16743], [18, 18520], [47, 16801], [24, 26236], [50, 14272], [27, 27426], [49, 15226], [38, 24859], [
                42, 20669], [31, 27418], [55, 11202], [32, 29050], [43, 19935], [58, 9733], [60, 8755], [37, 25853], [61, 8421], [14, 10831], [20, 21814], [33, 29355], [34, 29057], [52, 13045], [16, 14650], [30, 27524], [45, 18157], [48, 16119], [46, 17563], [19, 20329], [25, 26811], [54, 11910], [22, 24238], [35, 28191], [29, 27630], [28, 27583], [26, 27294], [41, 21793], [57, 10279], [15, 12647], [62, 8065]
        ])
    ]

    # 初始绘图
    plot_groups(row_points, "../output/scatter_plot.png")

    pass


def plot_hist():
    src_path = "/home/chencheng12/project/ann_data/script/streamAnnRecallV13_1000w_hist_100.fvecs"
    output_path = "../output/streamAnnRecallV13_1000w_hist_100.png"

    datas = read_fvecs(src_path, True)

    num_plots = len(datas)
    cols = 2
    rows = (num_plots + 1) // cols  # 自动计算行数

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3))  # 每行高度设为 3

    # 确保 axes 是二维数组
    axes = np.array(axes).reshape(-1, cols)

    # 绘制每个直方图
    for idx, data in enumerate(datas):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.hist(data, bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'Data Group {idx+1}')
    else:
        # 多余的 subplot 关闭（如果不是偶数）
        for i in range(num_plots, rows * cols):
            r, c = divmod(i, cols)
            fig.delaxes(axes[r][c])

    # 自动布局 & 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_data_with_tsne():
    root_path = "/home/chencheng12/project/ann_data/data"
    data_name = "sift1m"

    data_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    data = read_fvecs(data_path, True)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(data)  # data: shape (N, D)

    plt.scatter(embedding[:, 0], embedding[:, 1])

    plt.title("t-SNE Visualization")
    plt.savefig(f'../output/{data_name}_tsne.png', dpi=300)

    return


def plot_data_with_umap():
    root_path = "/home/web_server/cc/project/ANN-Data/data"
    data_name = "ReferAnnRecallV7_10w"
    query_n = 10

    data_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    ground_truth_path = os.path.join(
        root_path, data_name, data_name + "_groundtruth.ivecs")

    data = read_fvecs(data_path, True)
    ground_truth_data = read_ivecs(ground_truth_path, True)[:query_n]
    labels = generate_labels_with_ground_truth(data, ground_truth_data)

    # 映射颜色
    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels != -1]
    n_clusters = len(cluster_labels)

    cmap = cm.get_cmap('tab20', n_clusters)
    colors = cmap(np.arange(n_clusters))
    point_colors = np.full((len(labels), 4), fill_value=(
        0.6, 0.6, 0.6, 0.1))  # RGBA 灰色半透明

    for idx, label in enumerate(cluster_labels):
        point_colors[labels == label] = colors[idx]

    t_start = time.time()
    reducer = umap.UMAP(n_neighbors=64, min_dist=0.1,
                        metric='cosine', random_state=42)
    data_2d = reducer.fit_transform(data)
    t_end = time.time()
    print(f"cost: {t_end - t_start:.2f} s")

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], s=1,
                          c=point_colors)
    plt.title(f'UMAP Visualization of {data_name}')
    plt.grid(True)

    # 显示图例
    legend_elements = [
        Patch(facecolor=(0.6, 0.6, 0.6, 0.4), edgecolor='none', label='-1 (Unlabeled)')]

    for idx, label in enumerate(cluster_labels):
        color = colors[idx]
        legend_elements.append(
            Patch(facecolor=color, edgecolor='none', label=f'Label {label}'))

    plt.legend(handles=legend_elements, title='Labels',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'../output/{data_name}_umap.png', dpi=300)

    return


if __name__ == '__main__':
    # print_xy()
    # sqr_func()
    # print_xy()
    # get_shard_num()
    # plot_hist()
    # plot_data_with_tsne()
    plot_data_with_umap()

    pass
