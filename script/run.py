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

def read_fvecs(file_name: str) -> np.ndarray:
    print("begin to read fvecs: ", file_name)        
    with open(file_name, 'rb') as f:
        data = np.fromfile(f, dtype = np.float32)
        
    dim = struct.unpack('i', data[0])[0]            
    data = data.reshape(-1, dim + 1)

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
    print("cc debug: dim:", dim)
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
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise ValueError("文件为空")
        d = struct.unpack('i', dim_bytes)[0]
        
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
        # l2_squared = np.sum(vec ** 2)
        # print(f"l2_squared: {l2_squared}")
        
    return vec
        
def pca_dim_analyze(file_name, data_num, target_var =0.9):
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
    
    target_k = np.argmax(cumulative_variance >= target_var) + 1  # 找到95%累计方差对应的k值
    print(f"target_var: {target_var}, target_k: {target_k}")
    
def huggingface_dataset_download() -> None:
    save_name = "cohere"
    data_url = f"Cohere/wikipedia-22-12-zh-embeddings"
    save_path = '../data/' + save_name + ".fvecs"

    # 字段映射，根据需要修改
    field_map = {
        "id": "id", 
        "emb": "emb"
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
          
def split_dataset(seed = 42) -> None:
    # 将数据分割成 base / query 两个部分
    root_path = "../data/"
    data_name = "sift_single"

    data_path = os.path.join(root_path, data_name, data_name + ".fvecs")
    base_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")   
    
    data = read_fvecs(data_path) 
    print("data.shape:", data.shape)
    n_query = 1
    
    np.random.seed(seed)
    indices = np.random.permutation(data.shape[0])
    
    query_index = indices[:n_query]
    base_index = indices[n_query:]
    
    query_index = data[query_index]
    base_index = data[base_index]
    print("base_index.shape:", base_index.shape)
    print("query_index.shape:", query_index.shape)
    
    write_fvecs(base_path, base_index)
    write_fvecs(query_path, query_index)
    
    print(f"split_data complete: base_index: {base_index.shape}, query_index: {query_index.shape}")
    
def compute_distance():
    # 计算距离
    root_path = "../data/"
    data_name = "sift_single"
    
    query_index = [0]
    # base_index = [1147585, 1404049, 638789, 1035362]
    # base_index = [932085, 934876, 561813, 708177]
    base_index = [2]
    
    base_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")
    
    base_data = read_fvecs(base_path)
    query_data = read_fvecs(query_path)
    
    query = query_data[query_index]
    base = base_data[base_index]
    
    # 计算距离
    ip_score = np.dot(query, base.T)
    print("ip_score: ", ip_score)
    
    l2_squared  = np.sum(query ** 2, axis=1, keepdims=True) + np.sum(base ** 2, axis=1) - 2 * np.dot(query, base.T)
    print("l2_score: ", l2_squared)
    
def compute_batch_id(id, base, query, topk):
    # 计算距离

    # score = np.dot(query, base.T)
    score = np.sum(query ** 2, axis=1, keepdims=True) + np.sum(base ** 2, axis=1) - 2 * np.dot(query, base.T)
    
    # 获取 topk 的索引
    topk_indices = np.argsort(score, axis=1)[:, :topk]
    print(f"compute_batch_id {id} done, topk_indices shape: {topk_indices.shape}")
    return topk_indices

def compute_groundtruth_batch_with_parallel(base_data, query_data, topk, batch_size = 100, n_jobs = 4):
    n_query = query_data.shape[0]
    batches = [(i, query_data[i : i + batch_size]) for i in range(0, n_query, batch_size)]
    print("batches len:", len(batches))
    
    results = Parallel(n_jobs=n_jobs, backend = 'loky') (
        delayed(compute_batch_id)(i, base_data, q_batch, topk) for i, q_batch in batches
    )

    return np.vstack(results)
      
def generate_groundtruth() -> None:
    # 生成 groundtruth 文件
    root_path = "../data/"
    data_name = "sift_single"
    # 计算 query 的 topk groundtruth
    topk = 1
    
    use_pca = False
    pca_dim = 128
    
    base_path = os.path.join(root_path, data_name, data_name + "_base.fvecs")
    query_path = os.path.join(root_path, data_name, data_name + "_query.fvecs")
    groundtruth_path = os.path.join(root_path, data_name, data_name + "_groundtruth.ivecs")
    
    base_data = read_fvecs(base_path)
    query_data = read_fvecs(query_path)
    if use_pca:
        pca = PCA(n_components = pca_dim)
        base_data = pca.fit_transform(base_data)
        query_data = pca.transform(query_data)
        groundtruth_path = os.path.join(root_path, data_name, data_name + "_groundtruth_pca.ivecs")
        print("pca done")
    
    groundtruth = compute_groundtruth_batch_with_parallel(base_data, query_data, topk, batch_size = 100, n_jobs = 5)
    write_ivecs(groundtruth_path, groundtruth)
    
    print("groundtruth done")
    
def compare_recall() -> None:
    base_groundtruth_path = "/home/chencheng12/project/ann_data/data/sift/sift_groundtruth.ivecs.org"
    compare_groundtruth_path = "/home/chencheng12/project/ann_data/data/sift/sift_groundtruth.ivecs"
    
    groundtruth_a = read_ivecs(base_groundtruth_path)
    dim_a = groundtruth_a.shape[1]
    
    groundtruth_b = read_ivecs(compare_groundtruth_path)
    dim_b = groundtruth_b.shape[1]
    
    query_num = groundtruth_a.shape[0]
    
    compare_dim = min(dim_a, dim_b)
    groundtruth_a = groundtruth_a[:, :compare_dim]
    groundtruth_b = groundtruth_b[:, :compare_dim]
    
    # 计算 recall
    recalls = []
    for i in range(query_num):
        a = set(groundtruth_a[i])
        b = set(groundtruth_b[i])
        
        # 计算 recall
        recall = len(a.intersection(b)) / len(a)
        recalls.append(recall)
        
    recall_score =  np.mean(recalls)
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
    q_min = 0.0
    q_max = 0.0
    
    pre_lengths = []
    subvector_lengths = []
    
    with open(file_path, 'rb') as f:
        q_min = struct.unpack('f', f.read(4))[0]
        q_max = struct.unpack('f', f.read(4))[0]
        print(f"q_min: {q_min}, q_max: {q_max}")
        
        for i in range(n_subvector):
            pre_length = struct.unpack('q', f.read(8))[0]
            pre_lengths.append(pre_length)
        for i in range(n_subvector):
            subvector_length = struct.unpack('q', f.read(8))[0]
            subvector_lengths.append(subvector_length)
        # print(f"pre_lengths: {pre_lengths}")
        # print(f"subvector_lengths: {subvector_lengths}")
        
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
            
    
        codebook = np.zeros((n_class, dim), dtype = np.float32)
        for i in range(n_class):
            for j in range(dim):
                codebook[i][j] = struct.unpack('f', f.read(4))[0]
        print(f"codebook shape: {codebook.shape}")
        # print(f"codebook: {codebook}")
        
    return q_min, q_max, codebook, pca_data_mean, pca_principal

def encode_pq(data: np.ndarray, codebook: np.ndarray, n_subvector: int, quantize_type: np.dtype, q_min: float, q_max: float, is_query: bool = True, pca_mean: np.ndarray, pca_principal: np.ndarray) -> np.ndarray:    
    # PQ 编码
    n_data, dim = data.shape
    n_class = codebook.shape[0]
    n_subvector_dim = dim // n_subvector
    
    pq_code = np.zeros((n_data, n_subvector, n_class + 1), dtype = quantize_type)
    for k in range(n_data):
        # 计算 PCA
        if pca_mean not None:
            data[k] = data[k] - pca_mean
            data[k] = np.dot(data[k], pca_principal)
        
        tmp_distance_table = np.zeros((n_subvector, n_class), dtype = float)
        
        data_min = np.inf
        data_max = 0
        for i in range(n_subvector):
            subvector = data[k][i * n_subvector_dim : (i + 1) * n_subvector_dim]
            subvector_codebook = codebook[:, i * n_subvector_dim : (i + 1) * n_subvector_dim]
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
        data_max -= data_min    
            
        # 根据 tmp_distance_table 填写量化距离
        quantized_min = q_min        
        quantized_max = q_max
        if is_query:
            quantized_min = data_min
            quantized_max = data_max
            
        print(f"quantized_min: {quantized_min}, quantized_max: {quantized_max}")
        
        for i in range(n_subvector):
            for j in range(n_class):
                percent = (tmp_distance_table[i][j] - quantized_min) / quantized_max
                if percent > 1:
                    percent = 1
                
                dis = np.iinfo(quantize_type).max *  percent
                
                # 存储 subvector 到各个 center 的量化距离
                pq_code[k][i][j] = dis

              
    return pq_code
    
def pq_dis(query_data, base_data):
    _, n_subvector, dim = query_data.shape
    
    if n_subvector != base_data.shape[1] or dim != base_data.shape[2]:
        print(f"query_data.shape: {query_data.shape}, base_data.shape: {base_data.shape}")
        raise ValueError("维度不匹配")
    
    n_query = query_data.shape[0]
    n_base = base_data.shape[0]
    
    ret = np.zeros((n_query, n_base), dtype = np.float32)
    for m in range(n_base):
        for n in range(n_query):
            res = 0
            for i in range(n_subvector):
                best_index = base_data[m][i][-1]
                res += query_data[n][i][best_index]
            ret[n][m] = res
        
    return ret  
    
def compute_pq_dis():
    codebook_path = "/home/chencheng12/project/ann_data/data/codebooks/sift/codebooks_flash_INT16_512_32_32_256_128_0_1_0.txt"
    base_data_path = "/home/chencheng12/project/ann_data/data/sift/sift_base.fvecs"
    query_data_path = "/home/chencheng12/project/ann_data/data/sift/sift_query.fvecs"
    
    dim = 128
    n_subvector = 32
    n_class = 256
    quantize_type = np.uint16
    
    enable_pca = False
    pca_principal_dim = 0
    
    # base_index = [998047, 993263, 939382, 794616, 749838]
    # base_index = [932085, 934876, 561813, 708177, 706771, 695756, 435345, 701258, 455537, 872728, 36538, 562594, 908244, 600499, 893601, 619660, 562167, 746931, 565419, 236647, 568573,565814, 36267, 2176, 931632, 454263, 3752, 910119, 722642, 843384, 886630, 68299, 779712, 871066, 721706, 49874, 886222, 480497, 619829, 701919, 882, 87578,224263, 4009, 871568, 478814, 225116, 904911, 391655, 541845, 565484, 2837, 102903, 159953, 171663, 957845, 791852, 368702, 453447, 915482, 930567, 544275, 180955, 59844, 882946, 899809, 882961, 988166, 860056, 221339, 556209, 544202, 394507, 486457, 529986, 732473, 104122, 923811, 564914, 36139, 710644, 806773, 465294, 237161, 871048, 569837, 374617, 463781, 956733, 919197, 678385, 158759, 240996, 931948, 16429, 91348, 63349, 398306, 931721, 989762]
    
    base_index = [756694, 544275, 16429]
    query_index = [0]
    
    base_data = read_fvecs(base_data_path)[base_index]
    query_data = read_fvecs(query_data_path)[query_index]
    
    q_min, q_max, codebook, pca_mean, pca_principal = read_pq_codebook(codebook_path, dim, n_subvector, n_class, enable_pca, pca_principal_dim)
    pq_base_data = encode_pq(base_data, codebook, n_subvector, quantize_type, q_min, q_max, False)
    pq_query_data = encode_pq(query_data, codebook, n_subvector, quantize_type, q_min, q_max, True)
    
    with np.printoptions(threshold=np.inf):
        # print(f"pq_base_data: {pq_base_data}" )
        # print(f"pq_base_data.shape: {pq_base_data.shape}")
        # print(f"pq_base_data, quant_data: {pq_base_data[:, :, 256:]}")
        pass
    
    dis = pq_dis(pq_query_data, pq_base_data)
    # # dis = pq_dis(pq_base_data, pq_query_data)
    print(f"pq_dis: {dis}")
    
    
    
def compute_ip_dis():
    emb_a = np.array([0.459445, -0.784541, 0.332678, -0.227722, -0.607348, 0.393955, 0.205528, 0.197445, -0.904817, 0.153068, 0.246663, 0.274318, 0.0881884, 0.145625, -0.0334524, 0.382487, -0.286576, -1.18207, 0.377633, 0.165978, 0.806362, 0.835557, 0.564039, -0.0281437, -1.02787, 0.212187, -0.00910927, -0.195821, -0.136047, 0.963257, 0.476727, 0.250169, 1.08216, 0.23707, 0.249443, 0.562702, -0.2029, -0.545165, -0.89555, 0.426188, -0.125037, -0.186914, -0.752456, 0.103072, 0.116132, -0.20982, -0.423909, -0.809318, 0.511859, 0.29592, 0.544764, 0.116327, -0.661453, -0.107545, 0.18781, -1.22592, -0.379517, -0.468464, -0.0258397, 0.763438, 0.0275091, -0.193696, -0.164714, -0.729794, 0.412649, -0.67853, -0.150152, -0.0906334, -0.134262, 0.337062, -0.712863, 0.416704, -0.375496, -0.11359, 0.253077, -0.163563, 0.170667, 0.412346, 0.378557, -0.054914, -0.437873, 0.0959022, 0.312791, 0.440058, -0.15666, -0.147657, -0.0227138, 0.0661865, -0.561744, -0.459809, -0.784124, 0.493712, -1.09031, 0.181439, 0.0537629, -0.225572, -0.826041, -0.719915, 0.723571, -0.884072, 1.59089, -0.00823073, 0.0534151, 0.225651, 0.345826, 0.0948889, 0.0158797, -0.268326, -0.927716, 0.0952692, -0.11808, 0.107657, 0.684564, 0.174419, 0.28774, 0.754071, 0.0170375, -0.587092, 0.0851364, 0.320395, 0.11343, -0.264117, 0.153833, -0.838757, 0.135967, -0.253026, 0.842531, -0.227813], dtype = np.float32)
    
    emb_b = np.array([0.06262028962373734,  0.0032479427754878998,  0.07070232927799225,  -0.4133709967136383,  0.15274563431739807,  -0.2605890929698944,  -0.18639801442623138,  -0.15205122530460358,  -0.08651528507471085,  -0.12189537286758423,  0.02056165039539337,  -0.2585018575191498,  0.27294838428497314,  -0.09318022429943085,  -0.10741257667541504,  0.14421972632408142,  0.27561232447624207,  -0.34954574704170227,  -0.002767842262983322,  -0.02240435779094696,  0.29512688517570496,  0.18264268338680267,  0.08279849588871002,  0.33869075775146484,  0.2985718250274658,  0.32152262330055237,  -0.13614316284656525,  -0.48245933651924133,  -0.28031712770462036,  0.09477049857378006,  -0.03983171284198761,  0.08124972879886627,  0.2018161416053772,  -0.2490965873003006,  0.2808612585067749,  0.0436740405857563,  -0.33843499422073364,  0.12215898185968399,  0.09207557141780853,  -0.05849360674619675,  -0.06314876675605774,  -0.501409113407135,  0.038755521178245544,  -0.06292086839675903,  -0.17180682718753815,  -0.02670738659799099,  -0.02932419627904892,  -0.1926269829273224,  -0.11813541501760483,  -0.04949063062667847,  -1.2815918922424316,  -0.33288460969924927,  0.1416797637939453,  -0.4257826507091522,  -0.22680766880512238,  2.2778279781341553,  -0.3192138671875,  -0.16381020843982697,  -0.1211342066526413,  0.08921413123607635,  -0.10389865934848785,  0.373623788356781,  0.08463914692401886,  -0.048546165227890015,  0.09790521115064621,  -0.023721396923065186,  0.11025974154472351,  -0.17566238343715668,  0.2665412724018097,  -0.0063791424036026,  -0.20229417085647583,  0.0952170193195343,  0.6589530110359192,  0.26556113362312317,  -0.08710236102342606,  0.5102168917655945,  0.03145092353224754,  -0.399854838848114,  -0.2913246154785156,  -0.12951600551605225,  0.35668158531188965,  -0.10591374337673187,  0.2745687961578369,  0.06547752022743225,  -0.048597365617752075,  0.037555500864982605,  -0.05181920900940895,  -0.04345338046550751,  -0.3087169826030731,  0.222487211227417,  -0.332529217004776,  -0.14327572286128998,  0.8142602443695068,  -0.25746792554855347,  0.06167442351579666,  -0.1352652609348297,  -0.20300276577472687,  0.12226176261901855,  -0.08869995176792145,  -0.19193275272846222,  -1.9685537815093994,  0.242129847407341,  -0.0009398758411407471,  0.39218002557754517,  0.09819511324167252,  0.07326765358448029,  0.026059985160827637,  -0.5741429328918457,  -0.29881635308265686,  -0.2716735601425171,  -0.004341386258602142,  -0.09244855493307114,  -0.18070167303085327,  0.09474547952413559,  0.33317118883132935,  -0.09429614245891571,  -0.02017897367477417,  0.13837072253227234,  -0.01796853542327881,  -0.14637267589569092,  -0.0678606927394867,  -0.04852774366736412,  0.17577555775642395,  0.21043406426906586,  0.11118210852146149,  -0.03502504900097847,  -0.38223835825920105,  -0.5547353029251099], dtype = np.float32)
    
    
    ip_dis = np.dot(emb_a, emb_b)
    ip_dis = 1 - ip_dis
    print(f"ip_dis: {ip_dis}")
    pass
    
               
if __name__ == "__main__":
    # huggingface_dataset_download()
    # split_dataset()
    # generate_groundtruth()
    # read_vecs_at("/home/chencheng12/project/ann_data/data/cohere_zh/cohere_zh_groundtruth.ivecs", 0)
    # read_vecs_at("/home/chencheng12/project/ann_data/data/cohere_zh/cohere_zh_query.fvecs", 0)
    
    # read_vecs_at("/home/chencheng12/project/ann_data/data/cohere_zh/cohere_zh_groundtruth_ip.ivecs", 0)
    # read_vecs_at("/home/chencheng12/project/ann_data/data/sift/sift_groundtruth.ivecs", 0)
    # read_vecs_at("/home/chencheng12/project/ann_data/data/sift/sift_query.fvecs", 0)
    
    # read_vecs_at("/home/chencheng12/project/ann_data/data/sift_single/sift_single_query.fvecs", 0)
    # read_vecs_at("/home/chencheng12/project/ann_data/data/sift_single/sift_single_base.fvecs", 2)

    # compute_distance()
    
    # pca_dim_analyze("/home/chencheng12/project/ann_data/data/cohere_zh/cohere_zh_base.fvecs", 10000, 0.90)
    # compare_recall()
    
    # read_pq_codebook("/home/chencheng12/project/ann_data/data/codebooks/sift/codebooks_flash_INT8_512_32_16_256_64_0_1_0.txt", 128, 16, 256)
    compute_pq_dis()
    
    # compute_ip_dis()
    pass