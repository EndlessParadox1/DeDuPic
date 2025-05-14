import os
import clip
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

# 提取单张图片的特征
def get_clip_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    with torch.no_grad():
        return model.encode_image(image).cpu().numpy().flatten()  # [512]


# 提取整个文件夹的图像特征并保存
def extract_features_from_folder(folder_path):
    features = []
    image_paths = []
    for fname in tqdm(os.listdir(folder_path), desc="提取特征", unit="张图片"):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, fname)
            try:
                vec = get_clip_embedding(full_path)
                features.append(vec)
                image_paths.append(full_path)
            except Exception as e:
                print(f"读取失败：{full_path}", e)

    return image_paths, np.array(features)

# 基于相似度构建图，并返回相似图像组
def cluster_similar_images(image_paths, features, threshold):
    sim_matrix = cosine_similarity(features)
    G = nx.Graph()

    for idx, path in enumerate(image_paths):
        G.add_node(idx, path=path)

    n = len(image_paths)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                G.add_edge(i, j)

    clusters = []
    for component in nx.connected_components(G):
        if len(component) > 1:
            group = [image_paths[i] for i in component]
            clusters.append(group)
    return clusters

# 显示图片并提示删除选择，显示大小信息
def confirm_and_delete_clusters(clusters):
    for group in clusters:
        n = len(group)
        if n > 5:
            continue
        print(f"发现相似图片组：共 {n} 张")
        fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))

        for i, path in enumerate(group):
            try:
                img = Image.open(path)
                axs[i].imshow(img)
                size_kb = os.path.getsize(path) // 1024
                axs[i].set_title(f"{i}\n{size_kb} KB")
                axs[i].axis("off")
            except Exception as e:
                axs[i].set_title(f"加载失败：{path}, {e}")
                axs[i].axis("off")

        plt.tight_layout()
        plt.show()

        try:
            keep_idx = int(input(f"请输入要保留的图片编号（0 ~ {len(group)-1}），其余将被删除，回车跳过："))
        except:
            print("跳过此组")
            continue

        for i, path in enumerate(group):
            if i != keep_idx:
                try:
                    os.remove(path)
                    print(f"已删除：{path}")
                except Exception as e:
                    print(f"删除失败：{path}", e)

# 主函数
def main(folder_path, similarity_threshold):
    paths, feats = extract_features_from_folder(folder_path)
    print(f"共提取 {len(paths)} 张图片的特征")

    print("正在查找相似图片组...")
    clusters = cluster_similar_images(paths, feats, threshold=similarity_threshold)
    print(f"共找到 {len(clusters)} 组相似图片")

    if clusters:
        confirm_and_delete_clusters(clusters)
    else:
        print("没有发现明显相似的图片")


if __name__ == "__main__":
    folder = "./images"
    main(folder, similarity_threshold=0.98)
