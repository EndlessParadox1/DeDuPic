import argparse
import os
import sys

import clip
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device, jit=True)

def get_views(img: Image.Image) -> list:
    return [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        # img.transpose(Image.FLIP_TOP_BOTTOM),
        # img.rotate(90),
        # img.rotate(180),
        # img.rotate(270),
        # img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90),
        # img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270),
    ]

def fetch_features(paths: list[str]) -> np.ndarray:
    features = []

    for i, path in enumerate(tqdm(paths, desc="提取特征", unit="张图片")):
        try:
            img = Image.open(path).convert("RGB")
            imgs = [preprocess(view) for view in get_views(img)]
            with torch.no_grad():
                tensor = torch.stack(imgs).to(device)  # [2, 3, 224, 224]
                feat = model.encode_image(tensor).cpu().numpy()  # [2, 768]
                features.append(feat)
        except Exception as e:
            print(f"读取失败：{path}", e)

    return np.array(features)

def extract_features_from_folder(folder_path):
    paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)
                   if fname.lower().endswith((".jpg", ".jpeg", ".png"))]
    features = fetch_features(paths)
    return paths, features

def cluster_similar_images(paths, features, threshold):
    n = len(paths)
    G = nx.Graph()

    for idx, path in enumerate(paths):
        G.add_node(idx, path=path)

    for i in range(n):
        for j in range(i + 1, n):
            sim_matrix = cosine_similarity(features[i], features[j])
            if np.any(sim_matrix >= threshold):
                G.add_edge(i, j)

    clusters = []
    for component in nx.connected_components(G):
        if len(component) > 1:
            group = [paths[idx] for idx in component]
            clusters.append(group)
    return clusters


def confirm_and_delete_clusters(clusters, max_per_row=4):
    for k, group in enumerate(clusters):
        n = len(group)
        if n > 20:
            continue
        print(f"第{k}组：共 {n} 张，点击你想要**删除**的图片, 按Enter/Space继续，按Esc退出")

        ncols = min(n, max_per_row)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
        axes = axes.flatten()

        selected_for_deletion = set()

        def onclick(event):
            for i, ax in enumerate(axes[:n]):
                if ax == event.inaxes:
                    if i in selected_for_deletion:
                        selected_for_deletion.remove(i)
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                    else:
                        selected_for_deletion.add(i)
                        for spine in ax.spines.values():
                            spine.set_edgecolor("red")
                            spine.set_linewidth(3)
                            spine.set_visible(True)
                    fig.canvas.draw_idle()
                    break

        action = None
        def onkey(event):
            nonlocal action
            if event.key == " " or event.key == "enter":
                plt.close()
                action = 0
            elif event.key == "escape":
                plt.close()
                action = 1

        for i, path in enumerate(group):
            img = Image.open(path)
            ax = axes[i]
            ax.imshow(img)
            size_kb = os.path.getsize(path) // 1024
            ax.set_title(f"{size_kb} KB")
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", onkey)
        plt.show()

        if action == 1:
            print("用户退出")
            sys.exit(1)

        if not selected_for_deletion:
            print("未选择任何图片，跳过此组")
            continue

        for i in selected_for_deletion:
            path = group[i]
            try:
                os.remove(path)
                print(f"已删除：{path}")
            except Exception as e:
                print(f"删除失败：{path}", e)

def main(folder_path, threshold):
    if not os.path.exists(folder_path):
        print(f"文件夹不存在：{folder_path}")
        sys.exit(1)

    paths, feats = extract_features_from_folder(folder_path)
    print(f"共提取 {len(paths)} 张图片的特征")

    print("正在查找相似图片组...")
    clusters = cluster_similar_images(paths, feats, threshold)

    if clusters:
        print(f"共找到 {len(clusters)} 组相似图片")
        confirm_and_delete_clusters(clusters)
    else:
        print("没有发现明显相似的图片")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像去重")
    parser.add_argument("--folder", type=str, required=True, help="图片所在的文件夹路径")
    parser.add_argument("--threshold", type=float, default=0.98, help="相似度阈值（默认 0.98）")
    args = parser.parse_args()

    main(args.folder, args.threshold)
