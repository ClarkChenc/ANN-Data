import numpy as np
import struct
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd


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


def read_umap_data(data_path):
    data = np.load(data_path)
    return data


def write_umap_data(data, output_path):
    np.save(output_path, data)
    return


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


def plot_3d_data(data_3d, extra_info, output_path):
    labels = extra_info["labels"]
    dis = extra_info["dis"]

    df = pd.DataFrame(data_3d, columns=["x", "y", "z"])
    df["label"] = labels
    df["label_str"] = df["label"].astype(str)
    df.loc[df["label"] == -1, "label_str"] = "Unlabeled"
    df["index"] = np.arange(len(labels))  # 可选：数据点索引
    df["dis"] = dis

    # 创建一个 figure
    fig = go.Figure()

    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.T10  # 颜色列表可选其他

    for i, label in enumerate(unique_labels):
        subset = df[df["label"] == label]
        size = 1 if label == -1 else 2
        opacity = 0.4 if label == -1 else 1.0

        color = colors[i % len(colors)]
        if label == -1:
            color = "rgba(150,150,150,0.4)"

        # 设置 hover 信息
        hover_texts = [
            f"index: {idx}<br>dis: {dis:.5f}" for idx, dis in zip(subset["index"], subset['dis'])
        ]

        fig.add_trace(
            go.Scatter3d(
                x=subset["x"],
                y=subset["y"],
                z=subset["z"],
                mode="markers",
                name=f"Label {label}",
                text=hover_texts,
                hoverinfo="text",
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity
                )
            )
        )

    fig.update_layout(
        title="UMAP 3D Visualization",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend_title="Labels",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    # 💾 保存为 HTML
    fig.write_html(output_path)
    print(f"Saved interactive 3D plot to: {output_path}")

    return
