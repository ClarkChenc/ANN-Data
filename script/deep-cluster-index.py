import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
import pathlib

from utils import *


def kmeans(data, n_cluster, max_iter=300, random_state=42):
    km = KMeans(
        n_clusters=n_cluster,
        max_iter=max_iter,
        n_init='auto',
        random_state=random_state,
    )

    km.fit_predict(data)
    return km.cluster_centers_, km.labels_


class DeepClusterIndex:
    def __init__(self, data, n_level, n_cluster, index_path):
        self.data = data
        self.n_cluster = n_cluster
        self.n_level = n_level
        self.index_path = index_path

        # layer1
        # c1 emb
        # c2 emb
        # ...
        # layer2
        # c1 emb
        # c2 emb
        # ...
        self.codebooks = None
        code_books_path = os.path.join(index_path, 'codebooks.npy')
        if os.path.exists(code_books_path):
            print(f"Codebooks already exist at {code_books_path}. Loading...")
            self.codebooks = np.load(code_books_path)
        else:
            print(f"Building codebooks at {code_books_path}...")
            pathlib.Path(index_path).mkdir(parents=True, exist_ok=True)
            self.train_codebooks()
            np.save(code_books_path, self.codebooks)

        # key, [id1, id2, ...]
        self.index = None
        index_data_path = os.path.join(index_path, 'index.npy')
        if os.path.exists(index_data_path):
            print(f"Index already exists at {index_data_path}. Loading...")
            self.index = np.load(index_data_path, allow_pickle=True)
        else:
            print(f"Building index at {index_data_path}...")
            pathlib.Path(index_data_path).mkdir(parents=True, exist_ok=True)
            self.build_index()
            np.save(index_data_path, self.index)

    def train_codebooks(self):
        self.codebooks = []
        residual = self.data.copy()

        for i in range(self.n_level):
            print(f"Training level {i + 1} / {self.n_level} codebook...")
            centers, labels = kmeans(residual, self.n_cluster)  # 全局
            self.codebooks.append(centers)
            residual = residual - centers[labels]               # 更新残差
        return

    def build_index(self):
        self.index = {}

        code = ""
        for i in range(self.data.shape[0]):
            item = self.data[i].copy()
            for level in range(self.n_level):
                codebook = self.codebooks[level]
                # 从 codebook中找到最近的中心，使用 ip 距离
                dis = np.dot(codebook, item)
                idx = np.argmax(dis)
                code += str(idx) + "_"
            code = code[:-1]  # 去掉最后一个下划线
            if code not in self.index:
                self.index[code] = []
            self.index[code].append(i)
        return

    def search(self, query_data):
        query_code = ""
        for level in range(self.n_level):
            codebook = self.codebooks[level]
            dis = np.dot(codebook, query_data)
            idx = np.argmax(dis)
            query_code += str(idx) + "_"
        query_code = query_code[:-1]  # 去掉最后一个下划线
        if query_code in self.index:
            return self.index[query_code]
        else:
            return []


def compute_recall(index, query_data, ground_truth):
    recall_count = 0
    query_count = query_data.shape[0]
    recall_score = 0.0

    for i in range(query_count):
        ret_data = index.search(query_data[i])
        if len(ret_data) == 0:
            print("No results found.")
            continue
        # 计算召回率
        gt = ground_truth[i]
        if gt in ret_data:
            recall_count += 1
    recall_score = recall_count / query_count if query_count > 0 else 0.0

    return recall_score


def run():
    root_path = "../data/"
    data_name = "ReferAnnRecallV7_100w"
    n_layer = 3
    n_cluster = 500

    data_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")
    ground_truth_path = os.path.join(
        root_path, data_name, data_name + "_groundtruth.ivecs")
    index_path = os.path.join(root_path, "codebooks",
                              data_name, f"dc_index_{n_layer}_{n_cluster}")

    base_data = read_fvecs(data_path)
    query_data = read_fvecs(query_path)
    ground_truth = read_ivecs(ground_truth_path)

    print(f"begin to build index...{index_path}")
    index = DeepClusterIndex(base_data, n_layer, n_cluster, index_path)

    print(f"begin search...{query_path}")
    recall_score = compute_recall(index, query_data, ground_truth)

    print(f"Recall score: {recall_score}")

    return


def test():
    ls = [1, 2, 3, 4, 5, 1]

    filter_ls = ls == 1
    print(filter_ls)
    return


if __name__ == '__main__':
    run()
