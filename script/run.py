from datasets import load_dataset
import numpy as np
import os
import time
import pathlib
import struct
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gc
import pandas as pd
import ast
import pprint
import json

def do_gc():
  gc.collect()
  pass  

def show_vec(data):
  output = ''
  for i in range(len(data)):
    output += str(data[i]) + ', '
  print(output + "\n")

def read_fvecs(file_name: str, show_shape : bool = False) -> np.ndarray:
    print("begin to read fvecs: ", file_name)        
    with open(file_name, 'rb') as f:
        data = np.fromfile(f, dtype = np.float32)
        
    dim = struct.unpack('i', data[0])[0]            
    data = data.reshape(-1, dim + 1)
    
    if show_shape:
        print(f"data.shape: {data.shape}, dim: {dim}")

    # 仅使用 emb 部分
    return data[:, 1:]

def write_fvecs(filename, data):
    data = np.asarray(data, dtype=np.float32)
    dim = data.shape[1]
    with open(filename, 'wb') as f:
        for row in data:
            f.write(np.array([dim], dtype=np.int32).tobytes())
            f.write(row.tobytes())

def read_ivecs(file_name: str) -> np.ndarray:
    print("begin to read ivecs: ", file_name)        
    with open(file_name, 'rb') as f:
        data = np.fromfile(f, dtype = np.int32)
        
    dim = struct.unpack('i', data[0])[0]            
    data = data.reshape(-1, dim + 1)

    # 仅使用 emb 部分
    return data[:, 1:]
    
def write_ivecs(file_name: str, data: np.ndarray) -> None:
    print("begin to write ivecs")
    data = np.asarray(data, dtype = np.int32)
    dim = data.shape[1]
    with open(file_name, 'wb') as f:
        for row in data:
            f.write(np.array([dim], dtype = np.int32).tobytes())
            f.write(row.tobytes())
            
def read_vecs_at(file_path: str, index: int) -> None:
    print(f"Reading {file_path}")
    file = Path(file_path)
    ext = file.suffix
    
    value_type = np.float32
    if ext == ".fvecs":
        value_type = np.float32
    elif ext == ".ivecs":
        value_type = np.int32
    
    with open(file_path , "rb") as f:
        dim_expytes = f.read(4)
        if not dim_expytes:
            raise ValueError("文件为空")
        d = struct.unpack('i', dim_expytes)[0]
        
        vec_size = 4 * (d + 1)
        offset = index * vec_size
        f.seek(offset)
        raw = f.read(vec_size)
        
        d_check = struct.unpack('i', raw[:4])[0]
        if d != d_check:
            print(f"维度不匹配: {d} != {d_check}")
            raise ValueError("维度不匹配")
        vec = np.frombuffer(raw[4:], dtype=value_type)
        print(f"dim: {d} \nvec: {vec}")
        
        # get l2 distance
        l2_squared = np.sum(vec ** 2)
        print(f"l2_squared: {l2_squared}")
        
    return vec
        
def pca_dim_expasenalyze(file_name, data_num, target_var =0.9):
    # PCA 降维
    data = read_fvecs(file_name)
    
    enable_random_sample = True
    if enable_random_sample:
        np.random.seed(42)
        indices = np.random.permutation(data_num)
        data = data[indices]
    else:
        data = data[:data_num]
    print("data.shape:", data.shape)
    
    # 标准化
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # PCA 降维
    pca = PCA()
    pca.fit(data)
    
    explained_variance_ratio = pca.explained_variance_ratio_  # 每个主成分的方差比例
    cumulative_variance = np.cumsum(explained_variance_ratio)  # 累计解释方差
    # print("explained_variance_ratio:", explained_variance_ratio)
    # print("cumulative_variance:", cumulative_variance)
    for i in range(len(cumulative_variance)):
      print(f"{i} accumulate var: {cumulative_variance[i]}")
    
    target_k = np.argmax(cumulative_variance >= target_var) + 1  # 找到95%累计方差对应的k值
    print(f"target_var: {target_var}, target_k: {target_k}")
    
def huggingface_dataset_download() -> None:
    save_name = "cohere_zh"
    #data_url = f"bigcode/stack-exchange-embeddings-20230914"
    #data_url = f"Cohere/wikipedia-22-12-zh-embeddings"
    #data_url = f"Cohere/wikipedia-22-12-zh-embeddings"
    data_url = f"nielsr/datacomp-small-with-embeddings-and-cluster-labels"
    

    save_path = '../data/' + save_name + "/" + save_name + ".fvecs"
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    # 字段映射，根据需要修改
    # 内部字段：外部字段
    field_map = {
        "id": "id", 
        "emb": "clip_l14_embedding"
    }
    
    dataset = load_dataset(data_url, split="train", streaming=True)
    
    dim = 0
    if dataset:
        # 计算嵌入维度
        for doc in dataset:
            dim = len(doc[field_map["emb"]])
            break
    print(f"Embedding dimension: {dim}")
    
    with open(save_path, "wb") as f:
        doc_cnt = 0
        for doc in dataset:
            # print(doc[field_map["id"]], doc[field_map["emb"]])
            f.write(np.array([dim], dtype = np.int32).tobytes())
            f.write(np.array(doc[field_map["emb"]], dtype = np.float32).tobytes())
            
            doc_cnt += 1
            if doc_cnt % 100000 == 0:
                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"{cur_time}: Processed {doc_cnt} documents")
        print(f"download done")

def transfer_csv_to_fvecs() -> None:
  src_path = "/mnt/test/cc/project/ANN-Data/data/innerstreamv7_1000w/innerstreamv7_1000w.csv"
  dst_path = "/home/web_server/cc/project/ANN-Data/data/innerstreamv7_1000w/innerstreamv7_1000w.fvecs"
  emb_field = 'emb'

  df = pd.read_csv(src_path)
  df[emb_field] = df[emb_field].apply(ast.literal_eval)

  data = np.array(df[emb_field].tolist())
  write_fvecs(dst_path, data)

  print(f"transfer {src_path} done, shape: {data.shape}")

  pass

def get_hnsw_v2_linkdata():
  src_path = "/home/web_server/cc/project/ANN-Flash/statistics/codebooks/sift1m/index_hnsw-v2_INT16_512_32.txt"
  data_dim  = 128
  subvec_num = 4
  target_ids = {221339}

  print(f"show hnsw_v2 linkdata: {src_path}")
  id_map = {}
  with open(src_path, 'rb') as f:
    # offsetlevel 0
    c = f.read(8)

    c = f.read(8)
    max_element = struct.unpack('q', c)[0]

    c = f.read(8)
    cur_element_count = struct.unpack('q', c)[0]
    print(f"cur_element_count: {cur_element_count}")

    c = f.read(8)
    size_per_data = struct.unpack('q', c)[0]

    c = f.read(8)
    linkdata_offset = struct.unpack('q', c)[0]

    c = f.read(8)
    label_offset = struct.unpack('q', c)[0]

    c = f.read(8)
    data_offset = struct.unpack('q', c)[0]

    c = f.read(4)
    max_level = struct.unpack('i', c)[0]

    c = f.read(4)
    enter_point = struct.unpack('i', c)[0]

    c = f.read(8)
    max_M = struct.unpack('q', c)[0]

    c = f.read(8)
    max_M0 = struct.unpack('q', c)[0]

    c = f.read(8)
    M = struct.unpack('q', c)[0]

    c = f.read(8)
    mult = struct.unpack('d', c)[0]

    c = f.read(8)
    ef_c = struct.unpack('q', c)[0]

    tmp_target_link_data = {}
    for i in range(cur_element_count):
      if i % 100000 == 0:
        print(f"processing doc {i} ...")

      c = f.read(4)
      neighbor_size = struct.unpack('i', c)[0]

      c = f.read(max_M0 * 4)
      neighbor_ids = struct.unpack(f'{max_M0}i', c)
      neighbor_ids = [x for x in neighbor_ids if x != 0]

      n_link_data = {}
      for j in range(len(neighbor_ids)):
        nid = neighbor_ids[j]
        n_link_data[nid]  = {}

        c = f.read(4)
        dis = struct.unpack('f', c)[0]
        n_link_data[nid]['dis'] = dis

        c = f.read(subvec_num * 4)
        sub_dis = np.frombuffer(c, dtype = np.float32)
        n_link_data[nid]['sub_dis'] = sub_dis

      left_link_data_size = (max_M0 - len(neighbor_ids)) * (subvec_num + 1) * 4
      c = f.read(left_link_data_size)

      c = f.read(data_dim * 4)

      c = f.read(8)
      label = struct.unpack('q', c)[0]
      id_map[i] = label

      if label in target_ids:
        tmp_target_link_data[label] = n_link_data

    target_link_data = {}
    for k, val in tmp_target_link_data.items():
      target_link_data[k] = {}

      n_link_data = {}
      for n_k, n_val in val.items():
        n_link_data[id_map[n_k]] = n_val

      target_link_data[k] = n_link_data

    pprint.pprint(target_link_data)

  return

  

def analyze_hnsw_neighbors():
  src_path = "/mnt/test/cc/project/ANN-Data/data/statistics/codebooks/streamAnnRecallV13_1000w/index_hnsw_INT16_512_32.txt"
  data_dim = 128

  print(f"analyze hnsw neighbors: {src_path}")

  #target_ids = {828963, 3049115, 3357286, 9904061, 7420272, 2155121}
  target_ids = {1604028}

  internal_in_degree_nodes = {}
  in_degree_nodes = {}
  out_degree_nodes = {}

  id_map = {}

  with open(src_path, 'rb') as f:
    # offsetlevel 0
    c = f.read(8)

    c = f.read(8)
    max_element = struct.unpack('q', c)[0]

    c = f.read(8)
    cur_element_count = struct.unpack('q', c)[0]
    print(f"cur_element_count: {cur_element_count}")

    c = f.read(8)
    size_per_data = struct.unpack('q', c)[0]

    c = f.read(8)
    label_offset = struct.unpack('q', c)[0]

    c = f.read(8)
    data_offset = struct.unpack('q', c)[0]

    c = f.read(4)
    max_level = struct.unpack('i', c)[0]

    c = f.read(4)
    enter_point = struct.unpack('i', c)[0]

    c = f.read(8)
    max_M = struct.unpack('q', c)[0]

    c = f.read(8)
    max_M0 = struct.unpack('q', c)[0]

    c = f.read(8)
    M = struct.unpack('q', c)[0]

    c = f.read(8)
    mult = struct.unpack('d', c)[0]

    c = f.read(8)
    ef_c = struct.unpack('q', c)[0]

    for i in range(cur_element_count):
      if i % 100000 == 0:
        print(f"processing doc {i} ...")

      c = f.read(4)
      neighbor_size = struct.unpack('i', c)[0]

      c = f.read(max_M0 * 4)
      neighbor_ids = struct.unpack(f'{max_M0}i', c)
      neighbor_ids = [x for x in neighbor_ids if x != 0]

      c = f.read(data_dim * 4)

      c = f.read(8)
      label = struct.unpack('q', c)[0]
      id_map[i] = label

      if label not in out_degree_nodes:
        out_degree_nodes[label] = []
      out_degree_nodes[label].extend(neighbor_ids)

      for nid in neighbor_ids:
        if nid not in internal_in_degree_nodes:
          internal_in_degree_nodes[nid] = []
        internal_in_degree_nodes[nid].append(label)

  ret_in_degree_nodes = {}
  ret_out_degree_nodes = {}

  # 变换 out_degree_nodes
  for label, neighbor_ids in out_degree_nodes.items():
    neighbor_labels = list(map(lambda x: id_map[x], neighbor_ids))

    if label in target_ids:
      ret_out_degree_nodes[label] = neighbor_labels

  # 变换 in_degree_nodes
  for nid, neighbor_labels in internal_in_degree_nodes.items():
    label = id_map[nid]

    if label in target_ids:
      ret_in_degree_nodes[label] = neighbor_labels

  print("in_degree_nodes:")
  for key, val in ret_in_degree_nodes.items():
    print(f"{key}: \n{val}")

  print("out degree nodes:")
  for key, val in ret_out_degree_nodes.items():
    print(f"{key}: \n{val}")
  
  return

def analyze_flash_neighbors():
  src_path = "/mnt/test/cc/project/ANN-Data/data/statistics/codebooks/streamAnnRecallV13_1000w/index_flash_INT16_512_32_32_256_64_0_0_1.txt"
  data_dim = 128

  target_ids = {828963, 3049115, 3357286, 9904061, 7420272, 2155121}
  in_degree_nodes = {}
  out_degree_nodes = {}


  id_map = {}

  with open(src_path, 'rb') as f:
    c = f.read(8)
    max_elements = struct.unpack('q', c)[0]
    print(f'max_elements: {max_elements}')

    c = f.read(8)
    cur_element_count = struct.unpack('q', c)[0]
    print(f"cur_element_cout: {cur_element_count}")
   
    c = f.read(8)
    m = struct.unpack('q', c)[0]
    
    c = f.read(8)
    max_M = struct.unpack('q', c)[0]

    c = f.read(8)
    max_M0 = struct.unpack('q', c)[0]

    c = f.read(8)
    ef_c = struct.unpack('q', c)[0]

    c = f.read(8)
    mult = struct.unpack('d', c)[0]

    c = f.read(4)
    max_level = struct.unpack('i', c)[0]

    c = f.read(4)
    enterpoint_node = struct.unpack('i', c)[0]

    c = f.read(8)
    size_data_per_element = struct.unpack('q', c)[0]
    print(f'size_data_per_elements: {size_data_per_element}')

    c = f.read(8)
    size_links_level0 = struct.unpack('q', c)[0]

    c = f.read(8)
    size_links_per_element = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_level0 = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_link_list0 = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_link_list = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_linklist_data0 = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_linklist_data = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_data = struct.unpack('q', c)[0]

    c = f.read(8)
    label_offset = struct.unpack('q', c)[0]

    for i in range(cur_element_count):
      if i % 1000000 == 0:
        print(f"processing doc {i}")

      c = f.read(4)
      neighbor_size = struct.unpack('i', c)[0]

      c = f.read(max_M0 * 4)
      neighbor_ids = struct.unpack(f'{max_M0}i', c)
      neighbor_ids = [x for x in neighbor_ids if x != 0]

      c = f.read(data_dim * 4)

      c = f.read(8)
      label = struct.unpack('q', c)[0]
      id_map[i] = label
      
      if label in target_ids:
        if label not in out_degree_nodes:
          out_degree_nodes[label] = []
        out_degree_nodes[label].extend(list(map(lambda x: id_map[x], neibor_ids)))

      for nid in neighbor_ids:
        nid = id_map[nid]
        if nid in target_ids:
          if nid not in in_degree_nodes:
            in_degree_nodes[nid] = []
          in_degree_nodes[nid].append(label)

  print("in_degree_nodes:")
  for key, val in in_degree_nodes.items():
    print(f"{key}: \n{val}")

  print("out degree nodes:")
  for key, val in out_degree_nodes.items():
    print(f"{key}: \n{val}")
  
 
def transfer_flash_to_ivecs():
  src_path = "/home/chencheng12/project/ann_data/data/codebooks/sift1m/index_flash_400_32_INT16_64_512_PCA_128.txt"
  dst_path = '/home/chencheng12/project/ann_data/data/codebooks/sift1m/index_flash_400_32_INT16_64_512_PCA_128.flash.ivecs'
  subvec_size = 64

  data_map = {}

  with open(src_path, 'rb') as f:
    c = f.read(8)
    max_elements = struct.unpack('q', c)[0]
    print(f'max_elements: {max_elements}')

    c = f.read(8)
    cur_element_count = struct.unpack('q', c)[0]
    print(f"cur_element_cout: {cur_element_count}")
   
    c = f.read(8)
    m = struct.unpack('q', c)[0]
    
    c = f.read(8)
    max_M = struct.unpack('q', c)[0]

    c = f.read(8)
    max_M0 = struct.unpack('q', c)[0]

    c = f.read(8)
    ef_c = struct.unpack('q', c)[0]

    c = f.read(8)
    mult = struct.unpack('d', c)[0]

    c = f.read(4)
    max_level = struct.unpack('i', c)[0]

    c = f.read(4)
    enterpoint_node = struct.unpack('i', c)[0]

    c = f.read(8)
    size_data_per_element = struct.unpack('q', c)[0]

    c = f.read(8)
    size_links_level0 = struct.unpack('q', c)[0]

    c = f.read(8)
    size_links_per_element = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_level0 = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_link_list0 = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_link_list = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_linklist_data0 = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_linklist_data = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_data = struct.unpack('q', c)[0]

    c = f.read(8)
    label_offset = struct.unpack('q', c)[0]
    print(f"{label_offset}")

    for i in range(cur_element_count):
      if i % 1000000 == 0:
        print(f"processing doc {i}")

      c = f.read(4)
      neighbor_size = struct.unpack('i', c)[0]

      c = f.read(max_M0 * 4)
      neighbor_ids = struct.unpack(f'{max_M0}i', c)
      neighbor_ids = [x for x in neighbor_ids if x != 0]

      align_size = offset_data - 4 - max_M0 * 4 
      c = f.read(align_size)

      c = f.read(subvec_size * 2)
      data =np.frombuffer(c, dtype = np.uint16)

      c = f.read(8)
      label = struct.unpack('q', c)[0]
      
      # print(f"label: {label}")
      data_map[label] = data
      

      align_size = size_data_per_element - label_offset - 8 
      c = f.read(align_size)

  datas = []
  for i in range(cur_element_count):
    data = data_map[i]
    if len(data) == 0:
      print(f"miss data: {i}")
      break
    datas.append(data)

  write_ivecs(dst_path, datas)

  return

def transfer_hnsw_to_fvecs():
  src_path ="/mnt/test/cc/project/ANN-Data/data/streamAnnRecallV13/flanker.index"
  dst_path ="/home/web_server/cc/project/ANN-Data/data/streamAnnRecallV13/streamAnnRecallV13.fvecs"
  data_dim = 128

  embs = []
  with open(src_path, 'rb') as f:
    c = f.read(8)
    offset_level0 = struct.unpack('q', c)[0]

    c = f.read(8)
    cur_element_count = struct.unpack('q', c)[0]
    print(f"cur_element_cout: {cur_element_count}")

    c = f.read(8)
    size_data_per_element = struct.unpack('q', c)[0]

    c = f.read(8)
    label_offset = struct.unpack('q', c)[0]

    c = f.read(8)
    offset_data = struct.unpack('q', c)[0]

    c = f.read(4)
    max_level = struct.unpack('i', c)[0]

    c = f.read(4)
    enterpoint_node = struct.unpack('i', c)[0]

    c = f.read(8)
    max_M = struct.unpack('q', c)[0]

    c = f.read(8)
    max_M0 = struct.unpack('q', c)[0]

    c = f.read(8)
    m = struct.unpack('q', c)[0]

    c = f.read(8)
    mult = struct.unpack('d', c)[0]

    c = f.read(8)
    ef_c = struct.unpack('q', c)[0]

    for i in range(cur_element_count):
      if i % 1000000 == 0:
        print(f"processing doc {i}")

      c = f.read(4)
      neighbor_size = struct.unpack('i', c)[0]

      c = f.read(max_M0 * 4)
      neighbor_ids = struct.unpack(f'{max_M0}i', c)

      c = f.read(data_dim * 2)
      emb = np.frombuffer(c, dtype = np.float16)
      embs.append(emb)

      c = f.read(8)
      label = struct.unpack('q', c)[0]

  data = np.array(embs, dtype=np.float32)
  print(f'data.shape: {data.shape}')

  write_fvecs(dst_path, data)

       
def transfer_npy_to_fvecs() -> None:
    src_path = "/home/web_server/cc/project/ANN-Data/data/dup128d_1000w/dup128d_1000w.npy"
    dst_path = "/home/web_server/cc/project/ANN-Data/data/dup128d_1000w/dup128d_1000w.fvecs"
    
    with open(src_path, "rb") as f:
        data = np.load(f)
        print(f"read from {src_path}, data.shape: {data.shape}")
        
        write_fvecs(dst_path, data)
    print(f"transfer done, write to {dst_path}")
           
          
def split_dataset(seed = 42) -> None:
    # 将数据分割成 base / query 两个部分
    root_path = "../data/"
    data_name = "streamAnnRecallV13_1000w"
    n_query = 20000 

    data_path = os.path.join(root_path, data_name, data_name + ".fvecs")
    base_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")   
    
    data = read_fvecs(data_path) 
    print("data.shape:", data.shape)

    np.random.seed(seed)
    np.random.shuffle(data)
    
    query_index = data[:n_query, :]
    print("query_index.shape:", query_index.shape)
    
    base_index = data[n_query:, :]
    print("base_index.shape:", base_index.shape)
    
    write_fvecs(base_path, base_index)
    write_fvecs(query_path, query_index)
    
    print(f"split_data complete: base_index: {base_index.shape}, query_index: {query_index.shape}")

def compute_flash_distance():
    codebooks_path = "/home/web_server/cc/project/ANN-Data/data/statistics/codebooks/streamAnnRecallV13_1000w/codebooks_flash_INT16_512_32_32_256_64_0_0_1.txt"
    data_path = ""
    data_index = ""

    data_dim = 128
    subvector_num = 32
    cluster_num = 256

    subvector_len = data_dim // subvector_num

    codebook = []
    qmin = 0
    qmax = 0

    with open(codebooks_path, 'rb') as f:
      c = f.read(4)
      qmin = struct.unpack('f', c)[0]

      c = f.read(4)
      qmax = struct.unpack('f', c)[0]

      # pre_length
      for i in range(subvector_num):
        c = f.read(8)

      # subvector length
      for i in range(subvector_num):
        c = f.read(8)

      for i in range(cluster_num):
        c = f.read(data_dim * 4)
        emb = np.frombuffer(c, dtype=np.float32)
        codebook  = np.concatenate((codebook, emb))

    print(f"codebook: {codebook}")
    
    
def compute_distance():
    # 计算距离
    root_path = "../data/"
    #data_name = "streamAnnRecallV13_1000w"
    data_name = "sift1m"
    
    #query_index = [3933039, 1792875, 3357286]

    #query_index = [221339]
    query_index = [33]
    base_index = [5452]
    
    base_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    #query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")

    base_data = read_fvecs(base_path)
    query_data = read_fvecs(query_path)
    
    query = query_data[query_index]
    base = base_data[base_index]


    sub_vec_num = 2
    sub_vec_dim = query.shape[1] // sub_vec_num
    for i in range(sub_vec_num):
      sub_query = query[:, i * sub_vec_dim : (i + 1) * sub_vec_dim]
      sub_base = base[:, i * sub_vec_dim : (i + 1) * sub_vec_dim]

      l2_dis  = np.sum(sub_query ** 2, axis=1, keepdims=True) + np.sum(sub_base ** 2, axis=1) - 2 * np.dot(sub_query, sub_base.T)
      print(f"{i} : {l2_dis}", end = "\t")

    
    # 计算距离
    
    l2_squared  = np.sum(query ** 2, axis=1, keepdims=True) + np.sum(base ** 2, axis=1) - 2 * np.dot(query, base.T)
    print("total l2_score: ", l2_squared)

    
def compute_batch_id(id, base, query, topk, base_sqr = None):
    # 计算距离

    # score = np.dot(query, base.T)

    print(f"begin to compute {id}")
    if base_sqr is not None:
      score = np.sum(query ** 2, axis=1, keepdims=True) + base_sqr - 2 * np.dot(query, base.T)
    else:
      score = np.sum(query ** 2, axis=1, keepdims=True) + np.sum(base ** 2, axis=1) - 2 * np.dot(query, base.T)
    
    # 获取 topk 的索引
    topk_indices = np.argsort(score, axis=1)[:, :topk]
    print(f"compute_batch_id {id} done, topk_indices shape: {topk_indices.shape}")
    return topk_indices

def compute_groundtruth_expatch_with_parallel(base_data, query_data, topk, batch_size = 100, n_jobs = 2):
    n_query = query_data.shape[0]
    batches = [(i, query_data[i : i + batch_size]) for i in range(0, n_query, batch_size)]
    print("batches len:", len(batches))
    
    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs, backend = 'loky') (
            delayed(compute_batch_id)(i, base_data, q_batch, topk) for i, q_batch in batches
        )
    else :
        results = []
        base_sqr = np.sum(base_data ** 2, axis = 1)
        for i in range(len(batches)):
            q_batch = batches[i][1]
            topk_indices = compute_batch_id(i, base_data, q_batch, topk, base_sqr)
            results.append(topk_indices)

    return np.vstack(results)

def compute_groundtruth_safe(base, query, topk=100, query_batch_size=100, base_batch_size=10000):
    nq = query.shape[0]
    nb = base.shape[0]
    dim = base.shape[1]
    
    final_topk_indices = np.zeros((nq, topk), dtype=np.int32)
    final_topk_dists = np.full((nq, topk), np.inf, dtype=np.float32)

    for q_start in range(0, nq, query_batch_size):
        q_end = min(q_start + query_batch_size, nq)

        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{cur_time} cur query: {q_start}, query_end: {nq}")
        q_batch = query[q_start:q_end]  # (Bq, dim)

        # 当前 batch 的 topk 初始化
        batch_dists = np.full((q_end - q_start, topk), np.inf, dtype=np.float32)
        batch_indices = np.full((q_end - q_start, topk), -1, dtype=np.int32)

        for b_start in range(0, nb, base_batch_size):
            b_end = min(b_start + base_batch_size, nb)
            b_batch = base[b_start:b_end]  # (Bb, dim)

            # 距离计算 (L2 squared)
            dists = (
                np.sum(q_batch ** 2, axis=1, keepdims=True) +
                np.sum(b_batch ** 2, axis=1) -
                2 * np.dot(q_batch, b_batch.T)
            )  # shape: (Bq, Bb)

            # 合并现有结果
            combined_dists = np.concatenate([batch_dists, dists], axis=1)
            combined_indices = np.concatenate([
                batch_indices,
                np.arange(b_start, b_end).reshape(1, -1).repeat(q_end - q_start, axis=0)
            ], axis=1)

            # 选出 topk（不需要完全排序）
            topk_idx = np.argpartition(combined_dists, topk, axis=1)[:, :topk]
            row_idx = np.arange(q_end - q_start)[:, None]

            batch_dists = np.take_along_axis(combined_dists, topk_idx, axis=1)
            batch_indices = np.take_along_axis(combined_indices, topk_idx, axis=1)

            # 再次排序（按距离升序，稳定结果）
            order = np.argsort(batch_dists, axis=1)
            batch_dists = np.take_along_axis(batch_dists, order, axis=1)
            batch_indices = np.take_along_axis(batch_indices, order, axis=1)

        # final_topk_dists[q_start:q_end] = batch_dists
        final_topk_indices[q_start:q_end] = batch_indices

    return final_topk_indices 

def analyze_query_2_data_dis():
    root_path = '../data'

    data_name = "streamAnnRecallV13_1000w"
    query_num = 100
    
    use_pca = True
    pca_dim = 128
    
    base_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")
    dst_path = os.path.join(root_path, data_name, data_name + f"_hist_{query_num}.fvecs")
    
    base = read_fvecs(base_path)
    query = read_fvecs(query_path)[:query_num, :]


    l2_dis  = np.sum(query ** 2, axis=1, keepdims=True) + np.sum(base ** 2, axis=1) - 2 * np.dot(query, base.T)

    write_fvecs(dst_path, l2_dis)

    print("analyze done")
    
def generate_groundtruth_with_direction() -> None:
    # 生成 groundtruth 文件
    root_path = "../data/"
    #data_name = "streamAnnRecallV13_1000w"
    data_name = "sift1m"
    # 计算 query 的 topk groundtruth
    topk = 100
    subvec_num = 1
    
    base_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")
    groundtruth_path = os.path.join(root_path, data_name, data_name + "_groundtruth.ivecs")
    
    base = read_fvecs(base_path)
    query = read_fvecs(query_path)
    subvec_dim = query.shape[1] // subvec_num
    
    dis_list = []
    # for i in range(len(query)):
    for i in range(1):
        if i % 100000 == 0:
            print(f"processing query {i} ...")
        q = query[i]
        
        candidate = set()
        for j in range(subvec_num):
            sub_q = q[j * subvec_dim : (j + 1) * subvec_dim]
            sub_base = base[:, j * subvec_dim : (j + 1) * subvec_dim]
            
            topk_index = compute_groundtruth_safe(sub_base, sub_q.reshape(1, -1), topk, query_batch_size=1, base_batch_size=100000)
            for k in range(len(topk_index[0])):
                candidate.add(int(topk_index[0][k]))
        print(candidate)
        
        index = list(candidate)
        selected = base[index]  # shape: (n, dim)
        # 计算每一行与 query 的 L2 距离
        dists = np.linalg.norm(selected - q, axis=1)

        # 取距离最小的 k 个的索引（在 index 中的相对位置）
        topk_pos = np.argpartition(dists, k)[:k]
        # 可选：按真实距离排序
        topk_pos = topk_pos[np.argsort(dists[topk_pos])]
          
        print(f"{topk_pos}")
    print("dis_list done")
    
    # write_ivecs(groundtruth_path, groundtruth)
    return
 
      
def generate_groundtruth() -> None:
    # 生成 groundtruth 文件
    root_path = "../data/"
    #data_name = "streamAnnRecallV13_1000w"
    data_name = "sift1m"
    # 计算 query 的 topk groundtruth
    topk = 100
    
    use_pca = True
    pca_dim = 128
    sample_ratio = 0.1
    
    base_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")
    groundtruth_path = os.path.join(root_path, data_name, data_name + "_groundtruth.ivecs")
    
    base_data = read_fvecs(base_path)
    query_data = read_fvecs(query_path)
    if use_pca:
        sample_num = int(base_data.shape[0] * sample_ratio)
        indices = np.random.permutation(sample_num)
        sample_data = base_data[indices]
        pca = PCA(n_components = pca_dim)
        pca.fit(sample_data)
        
        base_data = pca.transform(base_data)
        query_data = pca.transform(query_data)
        groundtruth_path = os.path.join(root_path, data_name, data_name + f"_groundtruth_pca_{pca_dim}.ivecs")
        print("pca done")
    
    #groundtruth = compute_groundtruth_batch_with_parallel(base_data, query_data, topk, batch_size = 500, n_jobs = 1)
    groundtruth = compute_groundtruth_safe(base_data, query_data, topk, query_batch_size=1000, base_batch_size = 1000000)
    write_ivecs(groundtruth_path, groundtruth)
    
    print("groundtruth done")
    
def compare_recall() -> None:
    base_groundtruth_path = "/home/chencheng12/project/ann_data/data/sift1m/sift1m_groundtruth.ivecs"
    compare_groundtruth_path = "/home/chencheng12/project/ann_data/data/sift1m/sift1m_groundtruth_pca_128.ivecs"
    
    groundtruth_a = read_ivecs(base_groundtruth_path) 
    groundtruth_b = read_ivecs(compare_groundtruth_path)
    dim_exp = groundtruth_a.shape[1]
    
    query_num = groundtruth_a.shape[0]
  
    compare_dim = 50
    groundtruth_a = groundtruth_a[:, :compare_dim]
    groundtruth_b = groundtruth_b[:, :compare_dim]
    
    # 计算 recall
    recalls = []
    
    total_dim_num = query_num * compare_dim
    correct_dim_num = 0
    for i in range(query_num):
        a = set(groundtruth_a[i])
        b = set(groundtruth_b[i])
        
        # 计算 recall
        correct_dim_num += len(a.intersection(b))
        # recall = len(a.intersection(b)) / len(a)
        # recalls.append(recall)
        
    # recall_score =  np.mean(recalls)
    recall_score = correct_dim_num / total_dim_num
    print(f"recall score: {recall_score}")
    
def compute_kmeans(file_path: str, n_subvector: int, n_class: int) -> None:
    # 计算 PQ
    data = read_fvecs(file_path)
    n_data, dim = data.shape
    
    sub_dim = dim // n_subvector
    print(f"n_data: {n_data}, dim: {dim}, sub_dim: {sub_dim}")
    
    # 计算 kmeans
        
    print("pq done")
    
def read_pq_codebook(file_path: str, dim: int, n_subvector: int, n_class: int, enable_pca: bool = False, pca_principal_dim: int = 0) -> None:
    print(f"load codebook from {file_path}")
    q_min = 0.0
    q_max = 0.0
    
    pre_lengths = []
    subvector_lengths = []
    
    with open(file_path, 'rb') as f:
        pca_data_mean = None
        pca_principal = None
        if enable_pca:
          pca_data_mean = np.zeros((dim), dtype = np.float32)
          for i in range(dim):
              pca_data_mean[i] = struct.unpack('f', f.read(4))[0]
          print(f"pca_data_mean shape: {pca_data_mean.shape}")
          
          pca_principal = np.zeros((dim, pca_principal_dim), dtype = np.float32)
          for i in range(dim):
              for j in range(pca_principal_dim):
                  pca_principal[i][j] = struct.unpack('f', f.read(4))[0]
          print(f"pca_principal shape: {pca_principal.shape}")
          
        q_min = struct.unpack('f', f.read(4))[0]
        q_max = struct.unpack('f', f.read(4))[0]
        print(f"q_min: {q_min}, q_max: {q_max}")
    
        for i in range(n_subvector):
            pre_length = struct.unpack('q', f.read(8))[0]
            pre_lengths.append(pre_length)
            
        for i in range(n_subvector):
            subvector_length = struct.unpack('q', f.read(8))[0]
            subvector_lengths.append(subvector_length)
            
        codebook = np.zeros((n_class, dim), dtype = np.float32)
        for i in range(n_class):
            for j in range(dim):
                codebook[i][j] = struct.unpack('f', f.read(4))[0]
        print(f"codebook shape: {codebook.shape}")
        # print(f"codebook: {codebook}")
        
    return q_min, q_max, codebook, pca_data_mean, pca_principal, pre_lengths



def encode_pq(data: np.ndarray, codebook: np.ndarray, pre_lengths: np.ndarray, quantize_type: np.dtype, q_min: float, q_max: float, is_query: bool = True, pca_mean: np.ndarray = None, pca_principal: np.ndarray = None) -> (np.ndarray, np.ndarray):    
    # PQ 编码
    n_data, dim = data.shape
    n_class = codebook.shape[0]
    n_subvector = len(pre_lengths)
        
    subvector_len = []
    for i in range(n_subvector):
      if i == n_subvector - 1:
          subvector_len.append(dim - pre_lengths[i])
      else:
        subvector_len.append(pre_lengths[i + 1] - pre_lengths[i])

    # 数据, 每行 subvector 最后一位存放最佳 index
    pq_code = np.zeros((n_data, n_subvector, n_class + 1), dtype = quantize_type)
    # 数据到 center 的距离 
    dis_to_center = np.zeros((n_data), dtype = np.float32)
    for k in range(n_data):
        if k % 100 == 0:
          print("processing data: ", k)
        tmp_distance_table = np.zeros((n_subvector, n_class), dtype = float)
        
        if pca_mean is not None and pca_principal is not None:
            # PCA 处理
            norm_data = data[k] - pca_mean
            data[k] = np.dot(norm_data, pca_principal)
        
        data_min = np.inf
        data_max = 0
        for i in range(n_subvector):
            subvector = data[k][pre_lengths[i] : pre_lengths[i] + subvector_len[i]]
            subvector_codebook = codebook[:, pre_lengths[i] : pre_lengths[i] + subvector_len[i]]
            max_dis = 0
            
            best_class = 0
            min_distance = np.inf
            for j in range(n_class):
                # 计算距离 l2
                dis = np.linalg.norm(subvector - subvector_codebook[j]) ** 2               
                
                if dis < data_min:
                    data_min = dis
                if dis > max_dis:
                    max_dis = dis
                
                percent = (dis - q_min) / q_max
                if percent > 1:
                    percent = 1
                
                tmp_distance_table[i][j] = dis
                
                if dis < min_distance:
                    min_distance = dis 
                    best_class = j
                    
            # 存储 subvector 的的最佳 center
            pq_code[k][i][n_class] = best_class
            data_max += max_dis
            dis_to_center[k] += min_distance

        data_max -= data_min    
            
        # 根据 tmp_distance_table 填写量化距离
        quantized_min = q_min        
        quantized_max = q_max
        if is_query:
            quantized_min = data_min
            quantized_max = data_max
            
        # print(f"quantized_min: {quantized_min}, quantized_max: {quantized_max}")
        
        for i in range(n_subvector):
            for j in range(n_class):
                percent = (tmp_distance_table[i][j] - quantized_min) / quantized_max
                if percent > 1:
                    percent = 1

                if quantize_type == np.float32:
                    dis = tmp_distance_table[i][j]
                else:
                    dis = np.iinfo(quantize_type).max *  percent
                
                # 存储 subvector 到各个 center 的量化距离
                pq_code[k][i][j] = dis
              
    return pq_code, dis_to_center
    
def pq_dis(query_data, base_data):
    _, n_subvector, dim = query_data.shape
    
    if n_subvector != base_data.shape[1] or dim != base_data.shape[2]:
        print(f"query_data.shape: {query_data.shape}, base_data.shape: {base_data.shape}")
        raise ValueError("维度不匹配")
    
    n_query = query_data.shape[0]
    n_base = base_data.shape[0]
    
    ret = np.zeros((n_query, n_base), dtype = np.float32)
    for n in range(n_query):
        for m in range(n_base):
            res = 0
            for i in range(n_subvector):
                best_index = round(base_data[m][i][-1])
                res += query_data[n][i][best_index]
            ret[n][m] = res
        
    return ret  

def get_subvector_dis(query, data, subvec_size):
  data_dim = query.shape[1]
  subvec_len = data_dim // subvec_size

  query_num = query.shape[0]
  data_num = data.shape[0]

  ret = np.zeros((query_num, data_num, subvec_size), dtype = np.float32)

  for i in range(query_num):
    for j in range(data_num):
      debug_dis = np.sum((query[i] - data[j]) ** 2)
      for k in range(subvec_size):
        sub_query = query[i][k * subvec_len : (k + 1) * subvec_len]
        sub_data = data[j][k * subvec_len : (k + 1) * subvec_len]

        l2_dis = np.sum((sub_query - sub_data) ** 2)
        ret[i][j][k] = l2_dis

  return ret

def get_data_by_flash_encode():
    src_path = "/home/web_server/cc/project/ANN-Data/data/streamAnnRecallV13_1000w/streamAnnRecallV13_1000w.flash.ivecs"
    target_encode = [113, 253, 162, 73, 30, 10, 44, 57, 144, 2, 217, 175, 105, 56, 65, 50, 179, 234, 243, 62, 240, 228, 81, 183, 5, 204, 158, 0, 248, 114, 51, 213]
    

    cand = []
    datas = read_ivecs(src_path)

    for i in range(len(datas)):
      if i % 2000000:
        print(f"process {i} ....")
      data = datas[i]
      if np.array_equal(data, target_encode):
        cand.append(i)

    print(f"cand: {cand}")

    return
  
def cal_inversion_degree():
  
  #  pca 128 256 -> 0.776
  # data = [
  #   1404.,1297.,1504.,1579.,1556.,1686.,1695.,1513.,1602.,1658.,1794.,1559.,1723.,1735.,1746.,1784.,1627.,1747.,1718.,1790.,1744.,1862.,1872.,1834.,1920.,1814.,1832.,1870.,1874.,1893.,1831.,1874.,1832.,1922.,1746.,1878.,1978.,1909.,1760.,1706.,1879.,1857.,1882.,1822.,1949.,1863.,1877.,1924.,1856.,1987.,1975.,1912.,2010.,1957.,1919.,1877.,1976.,1982.,2077.,1788.,1943.,1951.,2112.,1835.,1953.,1966.,1993.,1997.,1985.,1972.,2068.,2006.,1839.,1967.,1817.,2098.,1785.,1869.,1773.,2022.,1919.,1869.,1877.,1937.,2013.,2200.,2015.,1998.,2008.,1861.,1964.,2027.,1918.,1991.,1894.,1976.,2005.,2099.,1891.,1886.
  # ] 
  # pca 128 512  -> 0.783
#   data = [
# 1407.,1315.,1522.,1593.,1553.,1711.,1712.,1544.,1637.,1665.,1809.,1563.,1738.,1750.,1768.,1803.,1650.,1771.,1742.,1788.,1771.,1876.,1871.,1843.,1940.,1829.,1853.,1880.,1902.,1892.,1851.,1890.,1851.,1932.,1771.,1875.,1980.,1910.,1777.,1731.,1889.,1848.,1884.,1838.,1973.,1879.,1884.,1943.,1862.,1996.,2006.,1920.,2034.,1963.,1953.,1870.,1981.,2001.,2090.,1814.,1959.,1951.,2123.,1869.,1965.,1965.,1996.,2025.,2004.,1974.,2074.,1998.,1847.,1981.,1843.,2101.,1794.,1884.,1799.,2016.,1959.,1902.,1906.,1954.,2009.,2221.,2007.,2017.,2040.,1887.,1966.,2033.,1932.,2014.,1906.,1989.,2014.,2120.,1917.,1914.
#   ]
  
  # pca 64_512 -> 0.845
  # data= [1525.,1507.,1622.,1763.,1830.,1743.,1844.,1860.,1877.,1940.,2022.,2030.,1933.,1972.,1974.,1946.,1982.,1948.,2055.,2081.,2027.,2060.,2137.,2106.,2068.,2116.,2113.,2146.,2080.,2030.,2131.,2092.,2109.,2126.,2146.,2153.,2164.,2178.,2145.,2191.,2147.,2084.,2162.,2111.,2189.,2109.,2176.,2157.,2199.,2181.,2197.,2201.,2188.,2178.,2191.,2155.,2200.,2210.,2243.,2099.,2167.,2189.,2245.,2179.,2230.,2216.,2322.,2212.,2275.,2259.,2193.,2243.,2129.,2203.,2155.,2305.,2126.,2171.,2133.,2328.,2234.,2165.,2265.,2200.,2202.,2278.,2229.,2227.,2184.,2215.,2285.,2253.,2120.,2220.,2157.,2212.,2278.,2315.,2266.,2287.]
  
  # pca 64_256   -> 0.843
  # data = [
  #     1449.,1482.,1640.,1726.,1794.,1754.,1840.,1876.,1894.,1946.,1968.,1920.,1960.,1991.,1947.,2028.,2037.,1942.,2044.,1994.,2033.,2001.,2085.,2058.,2109.,2107.,2058.,2141.,2081.,2012.,2161.,2088.,2134.,2069.,2052.,2080.,2099.,2182.,2104.,2153.,2167.,2173.,2168.,2158.,2152.,2148.,2135.,2138.,2176.,2159.,2201.,2163.,2188.,2187.,2165.,2126.,2214.,2231.,2262.,2221.,2189.,2216.,2230.,2162.,2201.,2234.,2218.,2193.,2196.,2188.,2183.,2284.,2128.,2171.,2222.,2260.,2068.,2223.,2074.,2225.,2201.,2110.,2250.,2189.,2279.,2266.,2217.,2216.,2183.,2182.,2290.,2280.,2182.,2237.,2204.,2211.,2189.,2219.,2290.,2185.
  
  # pca 64_256_float -> 0.845
  data = [
    1412.,1479.,1576.,1702.,1757.,1712.,1833.,1828.,1786.,1843.,1985.,1895.,1962.,1914.,1939.,1912.,1938.,1964.,1964.,1929.,2003.,1944.,2028.,2027.,2096.,2046.,1974.,2065.,2022.,2022.,2049.,1961.,2064.,2073.,2048.,2053.,2090.,2086.,2041.,2117.,2121.,2096.,2082.,2087.,2118.,2140.,2089.,2172.,2168.,2134.,2112.,2071.,2131.,2087.,2168.,2083.,2103.,2165.,2171.,2087.,2156.,2142.,2134.,2094.,2138.,2201.,2186.,2178.,2200.,2160.,2182.,2272.,2015.,2168.,2137.,2194.,2113.,2171.,2008.,2223.,2139.,2085.,2144.,2180.,2173.,2210.,2165.,2139.,2161.,2143.,2209.,2223.,2216.,2288.,2141.,2178.,2122.,2147.,2169.,2160.
  ] 
  
  # ]
  # pca 32 256 -> 0.673
  #   data = [
  # 2957.,2848.,2927.,2955.,2838.,3313.,3248.,3007.,3217.,3714.,3422.,3023.,3388.,3561.,3675.,3401.,3209.,3154.,3550.,3317.,3524.,4058.,3378.,3758.,3497.,3835.,3696.,3599.,3357.,3490.,3281.,3283.,3373.,3798.,3728.,3898.,3984.,3535.,3821.,3546.,3324.,3612.,3525.,3776.,3888.,3751.,4047.,3952.,3544.,4045.,3707.,3807.,4027.,4056.,4007.,3344.,3676.,3577.,3893.,3917.,3365.,3364.,3779.,3519.,4378.,4158.,3493.,3388.,3791.,4024.,4030.,3605.,3823.,3510.,4404.,3667.,3466.,3401.,3540.,4180.,4070.,3187.,3970.,3720.,3586.,4288.,3932.,3957.,3811.,3392.,3438.,4108.,4256.,3731.,3414.,3631.,3636.,4324.,3842.,3375.
  #   ]
  # nopca 32 256 -> 0.649
  # data = [
  # 13122.,13229.,13026.,13237.,13153.,13464.,12396.,13680.,12518.,13407.,12252.,12422.,13275.,12871.,13491.,12797.,12529.,14110.,12523.,13406.,13447.,13120.,12780.,13579.,13468.,12585.,13424.,12872.,13964.,13696.,13469.,12970.,12911.,12717.,13012.,13895.,13463.,12734.,13492.,14303.,12582.,13037.,13270.,13757.,13045.,13132.,12776.,12850.,14241.,13767.,12996.,13420.,13639.,13733.,13566.,13153.,13973.,14954.,14327.,12995.,13640.,13586.,13972.,13067.,13122.,13026.,13840.,13057.,13717.,13963.,14038.,13914.,14209.,13639.,13140.,13829.,14496.,14148.,13205.,13845.,13201.,13938.,14212.,13075.,13912.,13673.,13158.,12908.,13993.,14240.,13295.,13916.,14021.,12891.,13079.,13207.,13933.,14083.,13246.,13836.
  # ] 
  
  size =len(data)
  def merge_sort(arr):
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, inv_left = merge_sort(arr[:mid])
        right, inv_right = merge_sort(arr[mid:])
        merged, inv_split = merge(left, right)
        return merged, inv_left + inv_right + inv_split

  def merge(left, right):
        merged = []
        i = j = inv_count = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv_count += len(left) - i  # 这里发生逆序
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv_count

  _, count = merge_sort(data)
  
  score = 1 - (count / (size * (size - 1) / 2.0))
  print(f"Total inversion count: {score}")
  return
    
def analyze_pq_space():
    base_data_path = "/home/chencheng12/project/ann_data/data/codebooks/sift1m/index_flash_400_32_INT16_64_512_PCA_128.flash.ivecs"    
    
    dim = 128
    n_subvector = 64
    n_class = 256
    quantize_type = np.uint16
    #quantize_type = np.float32
    
    def get_encode_hash(encode: np.ndarray) -> str:
      # print(f"encode: {encode}")
      str_encode = ""
      for i in range(encode.shape[0]):
        str_encode += str(encode[i]) + "-"
      return str_encode
    
    base_data = read_ivecs(base_data_path)
    
    hash_dic = {}
    count = 0
    for i in range(len(base_data)):
        if i % 100000 == 0:
            print(f"processing base data {i} ...")
        pq_base_data = base_data[i]
        # print(f"pq_base_data: {pq_base_data}")
        hashcode = get_encode_hash(pq_base_data)
        
        if hashcode not in hash_dic:
          hash_dic[hashcode] = 1
        else :
          hash_dic[hashcode] += 1
          
    stat_dic = {}
    for k, val in hash_dic.items():
      if val not in stat_dic:
        stat_dic[val] = 1
      else:
        stat_dic[val] += 1

    print(f"stat_dic: {stat_dic}")
    return

def compute_pq_dis():
    codebook_path = "/home/chencheng12/project/ann_data/data/codebooks/sift1m/codebooks_flash_400_32_INT16_64_512_PCA_128.txt"
    base_data_path = "/home/chencheng12/project/ann_data/data/sift1m/sift1m_base.fvecs"
    query_data_path = "/home/chencheng12/project/ann_data/data/sift1m/sift1m_query.fvecs"
    
    enable_pca = False
    if "_PCA_" in codebook_path:
        enable_pca = True
        print("enable pca")
    
    dim = 128
    n_subvector = 64
    n_class = 256
    quantize_type = np.uint16
    #quantize_type = np.float32
    

    pca_principal_dim = dim
    
    query_index = [0]
    #base_index = [828963, 3049115, 3357286, 9904061, 7420272, 2155121]
    base_index = [828963]
    
    base_data = read_fvecs(base_data_path)[base_index]
    query_data = read_fvecs(query_data_path)[query_index]
    
    q_min, q_max, codebook, pca_mean, pca_principal, pre_lengths = read_pq_codebook(codebook_path, dim, n_subvector, n_class, enable_pca, pca_principal_dim)
    
    pq_query_data, _ = encode_pq(query_data, codebook, pre_lengths, quantize_type, q_min, q_max, True, pca_mean, pca_principal)
    pq_base_data, _ = encode_pq(base_data, codebook, pre_lengths, quantize_type, q_min, q_max, False, pca_mean, pca_principal)

    
    with np.printoptions(threshold=np.inf):
      #print(f"codebook: {codebook}")
      print(f"pq_base_data: {pq_base_data}" )
      #print(f"pq_base_data.shape: {pq_base_data.shape}")
      #print(f"pq_base_data, quant_data: {pq_base_data[:, :, 256:]}")
      pass
    
    dis = pq_dis(pq_query_data, pq_base_data)
    # # dis = pq_dis(pq_base_data, pq_query_data)
    print(f"pq_dis: {dis}")

    l2_squared  = np.sum(query_data ** 2, axis=1, keepdims=True) + np.sum(base_data ** 2, axis=1) - 2 * np.dot(query_data, base_data.T)
    print(f"l2 dis: {l2_squared}")

    # l2_subvec_dis = get_subvector_dis(query_data, base_data, n_subvector)
    # print(f"l2 subvec dis: {l2_subvec_dis}")    
    
def compute_ip_dis():
    emb_a = np.array([0.459445, -0.784541, 0.332678, -0.227722, -0.607348, 0.393955, 0.205528, 0.197445, -0.904817, 0.153068, 0.246663, 0.274318, 0.0881884, 0.145625, -0.0334524, 0.382487, -0.286576, -1.18207, 0.377633, 0.165978, 0.806362, 0.835557, 0.564039, -0.0281437, -1.02787, 0.212187, -0.00910927, -0.195821, -0.136047, 0.963257, 0.476727, 0.250169, 1.08216, 0.23707, 0.249443, 0.562702, -0.2029, -0.545165, -0.89555, 0.426188, -0.125037, -0.186914, -0.752456, 0.103072, 0.116132, -0.20982, -0.423909, -0.809318, 0.511859, 0.29592, 0.544764, 0.116327, -0.661453, -0.107545, 0.18781, -1.22592, -0.379517, -0.468464, -0.0258397, 0.763438, 0.0275091, -0.193696, -0.164714, -0.729794, 0.412649, -0.67853, -0.150152, -0.0906334, -0.134262, 0.337062, -0.712863, 0.416704, -0.375496, -0.11359, 0.253077, -0.163563, 0.170667, 0.412346, 0.378557, -0.054914, -0.437873, 0.0959022, 0.312791, 0.440058, -0.15666, -0.147657, -0.0227138, 0.0661865, -0.561744, -0.459809, -0.784124, 0.493712, -1.09031, 0.181439, 0.0537629, -0.225572, -0.826041, -0.719915, 0.723571, -0.884072, 1.59089, -0.00823073, 0.0534151, 0.225651, 0.345826, 0.0948889, 0.0158797, -0.268326, -0.927716, 0.0952692, -0.11808, 0.107657, 0.684564, 0.174419, 0.28774, 0.754071, 0.0170375, -0.587092, 0.0851364, 0.320395, 0.11343, -0.264117, 0.153833, -0.838757, 0.135967, -0.253026, 0.842531, -0.227813], dtype = np.float32)
    
    emb_b = np.array([0.06262028962373734,  0.0032479427754878998,  0.07070232927799225,  -0.4133709967136383,  0.15274563431739807,  -0.2605890929698944,  -0.18639801442623138,  -0.15205122530460358,  -0.08651528507471085,  -0.12189537286758423,  0.02056165039539337,  -0.2585018575191498,  0.27294838428497314,  -0.09318022429943085,  -0.10741257667541504,  0.14421972632408142,  0.27561232447624207,  -0.34954574704170227,  -0.002767842262983322,  -0.02240435779094696,  0.29512688517570496,  0.18264268338680267,  0.08279849588871002,  0.33869075775146484,  0.2985718250274658,  0.32152262330055237,  -0.13614316284656525,  -0.48245933651924133,  -0.28031712770462036,  0.09477049857378006,  -0.03983171284198761,  0.08124972879886627,  0.2018161416053772,  -0.2490965873003006,  0.2808612585067749,  0.0436740405857563,  -0.33843499422073364,  0.12215898185968399,  0.09207557141780853,  -0.05849360674619675,  -0.06314876675605774,  -0.501409113407135,  0.038755521178245544,  -0.06292086839675903,  -0.17180682718753815,  -0.02670738659799099,  -0.02932419627904892,  -0.1926269829273224,  -0.11813541501760483,  -0.04949063062667847,  -1.2815918922424316,  -0.33288460969924927,  0.1416797637939453,  -0.4257826507091522,  -0.22680766880512238,  2.2778279781341553,  -0.3192138671875,  -0.16381020843982697,  -0.1211342066526413,  0.08921413123607635,  -0.10389865934848785,  0.373623788356781,  0.08463914692401886,  -0.048546165227890015,  0.09790521115064621,  -0.023721396923065186,  0.11025974154472351,  -0.17566238343715668,  0.2665412724018097,  -0.0063791424036026,  -0.20229417085647583,  0.0952170193195343,  0.6589530110359192,  0.26556113362312317,  -0.08710236102342606,  0.5102168917655945,  0.03145092353224754,  -0.399854838848114,  -0.2913246154785156,  -0.12951600551605225,  0.35668158531188965,  -0.10591374337673187,  0.2745687961578369,  0.06547752022743225,  -0.048597365617752075,  0.037555500864982605,  -0.05181920900940895,  -0.04345338046550751,  -0.3087169826030731,  0.222487211227417,  -0.332529217004776,  -0.14327572286128998,  0.8142602443695068,  -0.25746792554855347,  0.06167442351579666,  -0.1352652609348297,  -0.20300276577472687,  0.12226176261901855,  -0.08869995176792145,  -0.19193275272846222,  -1.9685537815093994,  0.242129847407341,  -0.0009398758411407471,  0.39218002557754517,  0.09819511324167252,  0.07326765358448029,  0.026059985160827637,  -0.5741429328918457,  -0.29881635308265686,  -0.2716735601425171,  -0.004341386258602142,  -0.09244855493307114,  -0.18070167303085327,  0.09474547952413559,  0.33317118883132935,  -0.09429614245891571,  -0.02017897367477417,  0.13837072253227234,  -0.01796853542327881,  -0.14637267589569092,  -0.0678606927394867,  -0.04852774366736412,  0.17577555775642395,  0.21043406426906586,  0.11118210852146149,  -0.03502504900097847,  -0.38223835825920105,  -0.5547353029251099], dtype = np.float32)
    
    
    ip_dis = np.dot(emb_a, emb_b)
    ip_dis = 1 - ip_dis
    print(f"ip_dis: {ip_dis}")
    pass

def random_emb(dim: int):
    np.random.seed(42)
    n_data = 1
    data = np.random.rand(n_data, dim).astype(np.float32)
    print(f"random_emb: {data.shape}")
    
    data = data[0]
    emb_str = ""
    for i in range(dim):
        emb_str += f"{data[i]:.6f}, "
    
    print(f"random_emb: \n{emb_str}")
    pass
    
               
if __name__ == "__main__":
  try:

    #huggingface_dataset_download()
    #transfer_npy_to_fvecs()
    #transfer_csv_to_fvecs()
    #transfer_hnsw_to_fvecs()
    
    #split_dataset()
    # generate_groundtruth()
    # generate_groundtruth_with_direction()
    
    #read_fvecs("/mnt/test/cc/project/ANN-Data/data/bigcode/bigcode_base.fvecs", True)
    
    #read_vecs_at("/home/chencheng12/project/ann_data/data/sift_single/sift_single_query.fvecs", 0)
    #read_vecs_at("/home/web_server/cc/project/ANN-Data/data/streamAnnRecallV13_1000w/streamAnnRecallV13_1000w.flash.ivecs", 3357286)
    #read_vecs_at("/home/web_server/cc/project/ANN-Data/data/streamAnnRecallV13_1000w/streamAnnRecallV13_1000w.fvecs", 3357286)
    # read_vecs_at("/home/chencheng12/project/ann_data/data/sift1m/sift1m_groundtruth.ivecs", 0)

    #analyze_query_2_data_dis()

    # compute_distance()
    # compute_pq_dis()
    analyze_pq_space()
    # cal_inversion_degree()
    #analyze_flash_neighbors()
    #analyze_hnsw_neighbors()
    # transfer_flash_to_ivecs()
    #get_hnsw_v2_linkdata()
    
    #pca_dim_analyze("/mnt/test/cc/project/ANN-Data/data//sift1m_base.fvecs", 200000, 0.90)
    # compare_recall()
    #get_data_by_flash_encode()
    
    # read_pq_codebook("/home/chencheng12/project/ann_data/data/codebooks/sift/codebooks_flash_INT8_512_32_16_256_64_0_1_0.txt", 128, 16, 256)
    # compute_ip_dis()
    # random_emb(128)
    pass
  finally:
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)
    pass
