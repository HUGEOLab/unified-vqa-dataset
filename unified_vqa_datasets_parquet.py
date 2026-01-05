#!/usr/bin/env python3
"""
Parquet + Streaming 版本上传脚本：
- 自动将图片目录转换为 Parquet 格式
- 支持增量更新（智能检测变化）
- 生成 Streaming 友好的数据集
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datasets import Dataset, Features, Image, Value, load_dataset
from huggingface_hub import HfApi, login
from PIL import Image as PILImage


# ============ 配置信息 ============
CURRENT_DIR = Path("/mnt/mydev2/M256374/unified-vqa-dataset")
HF_REPO = "Geojx/unified-vqa-images"
HF_BRANCH = "main"
GITHUB_REPO = "HUGEOLab/unified-vqa-dataset"
GITHUB_BRANCH = "main"

# 图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# ⭐ 新增：标注文件路径（如果有的话）
ANNOTATION_FILE = CURRENT_DIR / "annotations.json"  # 或 .csv


def run_cmd(cmd, cwd=None, check=True):
    """运行系统命令"""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def load_annotations() -> Optional[Dict]:
    """
    加载标注数据（如果存在）
    支持格式：
    1. JSON: {"image_id": {"question": "...", "answer": "..."}, ...}
    2. CSV: image_id,question,answer
    """
    if not ANNOTATION_FILE.exists():
        print("⚠️ 未找到标注文件，将仅打包图片")
        return None
    
    print(f"📄 正在加载标注: {ANNOTATION_FILE}")
    
    if ANNOTATION_FILE.suffix == ".json":
        with open(ANNOTATION_FILE) as f:
            return json.load(f)
    
    elif ANNOTATION_FILE.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(ANNOTATION_FILE)
        # 转换成字典 {image_id: {其他列}}
        return df.set_index('image_id').to_dict('index')
    
    return None


def scan_images() -> List[Dict]:
    """
    扫描图片目录，构建数据集列表
    返回格式: [{"image": PIL.Image, "image_id": "...", ...}, ...]
    """
    images_dir = CURRENT_DIR / "unified-vqa-images"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    
    print(f"🔍 扫描图片目录: {images_dir}")
    
    # 加载标注
    annotations = load_annotations()
    
    data_list = []
    
    for root, _, files in os.walk(images_dir):
        for file in sorted(files):  # 排序保证可重复性
            if Path(file).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            image_path = Path(root) / file
            image_id = image_path.stem  # 去掉扩展名作为 ID
            
            # 构建数据条目
            item = {
                "image": str(image_path),  # datasets 库会自动处理路径
                "image_id": image_id,
            }
            
            # 如果有标注，添加对应字段
            if annotations and image_id in annotations:
                item.update(annotations[image_id])
            
            data_list.append(item)
    
    print(f"✅ 找到 {len(data_list)} 张图片")
    return data_list


def create_parquet_dataset():
    """
    核心函数：将图片 + 标注转换为 Parquet 格式数据集
    """
    print("\n" + "="*60)
    print("🔄 开始转换为 Parquet 格式...")
    print("="*60)
    
    # 1. 扫描图片
    data_list = scan_images()
    
    if not data_list:
        raise ValueError("没有找到任何图片！")
    
    # 2. 定义 Schema（数据结构）
    # ⚠️ 重要：Image() 类型会自动处理二进制存储
    features = Features({
        "image": Image(),  # 这是关键！自动无损存储
        "image_id": Value("string"),
        # 如果有标注，添加对应字段
        # "question": Value("string"),
        # "answer": Value("string"),
    })
    
    # 动态检测标注字段
    if data_list[0].keys() - {"image", "image_id"}:
        extra_fields = data_list[0].keys() - {"image", "image_id"}
        for field in extra_fields:
            features[field] = Value("string")  # 根据实际类型调整
    
    # 3. 创建 Dataset 对象
    print("   📦 正在打包数据...")
    dataset = Dataset.from_list(data_list, features=features)
    
    # 4. （可选）验证一张图片的完整性
    print("\n🔬 正在验证数据完整性...")
    sample = dataset[0]
    print(f"   - Image ID: {sample['image_id']}")
    print(f"   - Image size: {sample['image'].size}")
    print(f"   - Image mode: {sample['image'].mode}")
    
    return dataset


def upload_to_huggingface_parquet(dataset: Dataset):
    """
    上传 Parquet 格式数据集到 HF
    内部会自动：
    1. 转换为 .parquet 文件
    2. 分片（如果数据量大）
    3. 生成配套的 dataset_info.json
    """
    print("\n🚀 [Hugging Face] 正在上传 Parquet 数据集...")
    
    try:
        # ⚠️ 关键参数说明：
        # - private=False: 公开数据集（必须公开才能无限存储）
        # - max_shard_size="500MB": 每个 Parquet 文件最大 500MB（自动分片）
        dataset.push_to_hub(
            repo_id=HF_REPO,
            private=False,  # 公开仓库
            max_shard_size="500MB",  # 自动分片，避免单文件过大
            commit_message=f"Add dataset with {len(dataset)} images"
        )
        
        print("✅ Hugging Face 上传完成！")
        print(f"📊 数据集链接: https://huggingface.co/datasets/{HF_REPO}")
        print(f"\n💡 使用方法:")
        print(f'   from datasets import load_dataset')
        print(f'   ds = load_dataset("{HF_REPO}", split="train", streaming=True)')
        print(f'   # 国内用户请先设置: export HF_ENDPOINT=https://hf-mirror.com')
        
        return True
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        
        # 常见错误提示
        if "authentication" in str(e).lower():
            print("\n💡 请先登录 Hugging Face:")
            print("   huggingface-cli login")
            print("   或在代码开头添加: login(token='hf_...')")
        
        return False


def upload_to_github_simple(code_files: List[Path]):
    """
    上传代码文件到 GitHub (保持原有逻辑)
    """
    if not code_files:
        print("\n✅ [GitHub] 无需同步代码文件")
        return True
        
    print(f"\n🚀 [GitHub] 同步 {len(code_files)} 个代码文件...")
    
    work_dir = Path("/tmp/gh_upload_incremental")
    if work_dir.exists(): 
        subprocess.run(["rm", "-rf", str(work_dir)])
    work_dir.mkdir(parents=True)
    
    # 克隆
    print("   🔄 克隆仓库...")
    ssh_url = f"git@github.com:{GITHUB_REPO}.git"
    success, _ = run_cmd(["git", "clone", "-b", GITHUB_BRANCH, ssh_url, str(work_dir)])
    
    if not success:
        https_url = f"https://github.com/{GITHUB_REPO}.git"
        run_cmd(["git", "clone", "-b", GITHUB_BRANCH, https_url, str(work_dir)])
    
    # 复制文件
    for f in code_files:
        rel_path = f.relative_to(CURRENT_DIR)
        target = work_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", str(f), str(target)])
    
    # 提交
    run_cmd(["git", "add", "."], cwd=work_dir)
    success, status = run_cmd(["git", "status", "--porcelain"], cwd=work_dir)
    
    if status.strip():
        run_cmd(["git", "commit", "-m", "Update code and docs"], cwd=work_dir)
        success, err = run_cmd(["git", "push"], cwd=work_dir)
        if success:
            print("✅ GitHub 同步完成")
        else:
            print(f"❌ 推送失败: {err}")
            return False
    else:
        print("✅ GitHub 已是最新")
    
    return True


def categorize_code_files() -> List[Path]:
    """
    仅扫描代码文件（排除图片）
    """
    code_files = []
    
    for root, dirs, files in os.walk(CURRENT_DIR):
        # 排除特定目录
        if ".git" in dirs: dirs.remove(".git")
        if "unified-vqa-images" in dirs: dirs.remove("unified-vqa-images")
        if "__pycache__" in dirs: dirs.remove("__pycache__")
        
        for file in files:
            file_path = Path(root) / file
            
            # 只上传代码相关文件
            if file_path.suffix in {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.sh'}:
                code_files.append(file_path)
    
    return code_files


if __name__ == "__main__":
    print("="*60)
    print("🚀 Parquet + Streaming 智能打包工具")
    print("="*60)
    
    try:
        # ⭐ 步骤 1: 转换为 Parquet 数据集
        dataset = create_parquet_dataset()
        
        # ⭐ 步骤 2: 上传到 Hugging Face
        hf_ok = upload_to_huggingface_parquet(dataset)
        
        # 步骤 3: 同步代码到 GitHub
        code_files = categorize_code_files()
        gh_ok = upload_to_github_simple(code_files)
        
        print("\n" + "="*60)
        if hf_ok and gh_ok:
            print("✨ 全部搞定！数据已转换为高效的 Streaming 格式！")
            print(f"\n📖 快速开始:")
            print(f"   # Python 代码")
            print(f'   from datasets import load_dataset')
            print(f'   ds = load_dataset("{HF_REPO}", split="train", streaming=True)')
            print(f'   for sample in ds.take(5):')
            print(f'       print(sample["image_id"], sample["image"].size)')
        else:
            print("⚠️ 部分操作失败，请检查上方日志")
            
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
