import numpy as np
import struct


def read_fvecs(file_name: str, show_shape: bool = False) -> np.ndarray:
    print("begin to read fvecs: ", file_name)
    with open(file_name, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    dim = struct.unpack('i', data[0])[0]
    data = data.reshape(-1, dim + 1)

    if show_shape:
        print(f"data.shape: {data.shape}, dim: {dim}")

    # 仅使用 emb 部分
    return data[:, 1:]


def read_ivecs(file_name: str, show_shape: bool = False) -> np.ndarray:
    print("begin to read ivecs: ", file_name)
    with open(file_name, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)

    dim = struct.unpack('i', data[0])[0]
    data = data.reshape(-1, dim + 1)

    if show_shape:
        print(f"data.shape: {data.shape}, dim: {dim}")

    # 仅使用 emb 部分
    return data[:, 1:]


def generate_normal_random(size, min, max, mean=0.5, std_dev=1):

    samples = []
    while len(samples) < size:
        new_samples = np.random.normal(
            loc=mean, scale=std_dev, size=(size * 2, ))
        valid_samples = new_samples[(
            new_samples >= min) & (new_samples <= max)]
        samples.extend(valid_samples.tolist())

    return np.array(samples[:size])


def plot_xy_data(group_data, output_path):
    # 合并所有数据
    points = np.vstack(group_data)

    # 生成标签
    labels = np.concatenate([
        np.full(len(group), i) for i, group in enumerate(group_data)
    ])

    # 绘图
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(points[idx, 0], points[idx, 1], label=f'Group {label+1}')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig(output_path)  # 可选：dpi 控制清晰度
    plt.close()  # 关闭图形，避免在脚本中反复显示
