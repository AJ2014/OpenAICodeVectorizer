import os
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from chromadb.config import Settings
import chromadb
from .utils import load_config

# Version: 1.2
# Updated: 2025-05-12 17:10:00

def visualize_chroma_vectors(chroma_db_path=None, collection_name=None, max_points=2000):
    config = load_config()
    chroma_db_path = chroma_db_path or os.path.abspath(config['chroma_db_path'])
    collection_name = collection_name or config['collection_name']

    print(f"加载ChromaDB: {chroma_db_path}, 集合: {collection_name}")
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_collection(collection_name)

    # 获取所有向量和元数据
    all_count = collection.count()
    print(f"集合内向量总数: {all_count}")
    if all_count == 0:
        print("没有可视化的数据。"); return

    # 分批获取所有数据（避免一次性拉取过多）
    batch_size = 500
    all_embeddings = []
    all_metadatas = []
    all_ids = []
    for offset in range(0, all_count, batch_size):
        batch = collection.get(
            include=["embeddings", "metadatas"],
            offset=offset,
            limit=min(batch_size, max_points-len(all_embeddings))
        )
        all_embeddings.extend(batch["embeddings"])
        all_metadatas.extend(batch["metadatas"])
        all_ids.extend(batch["ids"])
        if len(all_embeddings) >= max_points:
            break
    print(f"实际可视化点数: {len(all_embeddings)}")

    # PCA降维到2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(all_embeddings)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
    # 标注部分点的文件名
    for i, meta in enumerate(all_metadatas):
        if i % max(1, len(all_metadatas)//30) == 0:  # 最多显示30个标签
            label = meta.get("source") or all_ids[i]
            plt.annotate(label, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
    plt.title(f"Chroma向量可视化 (PCA降维, 共{len(all_embeddings)}个点)", fontproperties=font)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.show() 