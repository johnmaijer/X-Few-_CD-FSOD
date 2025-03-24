import torch
import os


def load_cache(dataset_name: str, out_dir: str, device="cpu"):
    """
    加载缓存的特征和标签
    参数:
        dataset_name: 数据集名称（需与保存时的名称一致）
        out_dir: 文件保存目录（需与保存时的目录一致）
        device: 加载到CPU/GPU
    """
    keys_path = os.path.join(out_dir, f"{dataset_name}_keys.pt")
    values_path = os.path.join(out_dir, f"{dataset_name}_values.pt")

    # 加载文件
    cache_keys = torch.load(keys_path, map_location=device)
    cache_values = torch.load(values_path, map_location=device)

    return cache_keys, cache_values


# 使用示例
if __name__ == "__main__":
    # 参数需要与保存时一致
    dataset = "dataset3_5shot"
    save_dir = "./cache_models"

    # 加载到CPU
    keys, values = load_cache(dataset_name=dataset, out_dir=save_dir)
    print("keys:", keys)
    print("values:" , values)

    # 加载到GPU（如果可用）
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # keys, values = load_cache(..., device=device)

    # 验证形状
    print(f"Keys shape: {keys.shape}")  # 应该为 [num_samples, feature_dim]
    print(f"Values shape: {values.shape}")  # 应该为 [num_samples, num_classes]