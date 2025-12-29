#!/usr/bin/env python3
"""
统一上传脚本 (智能增量版)：
- 自动检测 Hugging Face 已存在文件，跳过重复上传
- 增加重试机制，防止网络波动中断
- 修复 GitHub 上传逻辑
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Set

# ============ 配置信息 ============
CURRENT_DIR = Path("/mnt/mydev2/M256374/unified-vqa-dataset")
HF_REPO = "Geojx/unified-vqa-images"
HF_BRANCH = "main"
GITHUB_REPO = "HUGEOLab/unified-vqa-dataset"
GITHUB_BRANCH = "main"

# 图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

def run_cmd(cmd, cwd=None, check=True):
    """运行系统命令"""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def get_remote_hf_files(repo_id: str) -> Set[str]:
    """获取 HF 仓库中已存在的所有文件列表"""
    print(f"🔍 正在检查 {repo_id} 已有的文件 (用于断点续传)...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # 获取仓库文件列表
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"✅ 远程仓库已有 {len(files)} 个文件")
        return set(files)
    except Exception as e:
        print(f"⚠️ 无法获取远程列表 (可能是新仓库): {e}")
        return set()

def categorize_files() -> Tuple[List[Path], List[Path]]:
    """分类文件"""
    image_files = []
    other_files = []
    
    images_dir = CURRENT_DIR / "unified-vqa-images"
    
    print(f"\n📂 扫描本地目录: {CURRENT_DIR}")
    
    # 扫描图片
    if images_dir.exists():
        for root, _, files in os.walk(images_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(file_path)
    
    # 扫描非图片 (排除 .git 和图片目录)
    for root, dirs, files in os.walk(CURRENT_DIR):
        if ".git" in dirs: dirs.remove(".git")
        if "unified-vqa-images" in dirs: dirs.remove("unified-vqa-images")
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                other_files.append(file_path)

    return image_files, other_files

def upload_to_huggingface_incremental(image_files: List[Path]):
    """增量上传图片"""
    if not image_files:
        return True
    
    # 1. 获取远程文件列表
    remote_files = get_remote_hf_files(HF_REPO)
    
    # 2. 过滤已存在的文件
    files_to_upload = []
    skipped_count = 0
    
    print("⚖️  正在对比差异...")
    for f in image_files:
        # 计算相对路径 (例如 unified-vqa-images/train/001.jpg)
        rel_path = f.relative_to(CURRENT_DIR).as_posix() # 强制使用 / 分隔符
        
        if rel_path in remote_files:
            skipped_count += 1
        else:
            files_to_upload.append(f)
            
    print(f"📊 差异对比完成:")
    print(f"   - 总文件数: {len(image_files)}")
    print(f"   - ✅ 已存在 (跳过): {skipped_count}")
    print(f"   - 🚀 待上传: {len(files_to_upload)}")
    
    if not files_to_upload:
        print("🎉 所有图片都已上传，无需操作！")
        return True

    # 3. 开始上传剩余文件
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    
    # 降低 Batch size 提高成功率
    BATCH_SIZE = 500
    total_batches = (len(files_to_upload) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n🚀 开始上传剩余的 {len(files_to_upload)} 个文件...")
    
    for i in range(0, len(files_to_upload), BATCH_SIZE):
        batch = files_to_upload[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        operations = []
        for f in batch:
            rel_path = f.relative_to(CURRENT_DIR).as_posix()
            operations.append(CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=str(f)))
        
        # 重试循环
        for attempt in range(3):
            try:
                print(f"   📤 上传批次 {batch_num}/{total_batches} ({len(batch)} files)...", end="", flush=True)
                api.create_commit(
                    repo_id=HF_REPO,
                    operations=operations,
                    commit_message=f"Upload batch {batch_num} (incremental)",
                    repo_type="dataset"
                )
                print(" ✅ 成功")
                break
            except Exception as e:
                print(f" ❌ 失败: {str(e)[:50]}... 重试 {attempt+1}/3")
                time.sleep(5)
        else:
            print("❌ 此批次重试多次均失败，脚本终止")
            return False
            
    return True

def upload_to_github_simple(other_files: List[Path]):
    """GitHub 上传 (保持简单有效)"""
    if not other_files:
        return True
        
    print(f"\n🚀 [GitHub] 同步 {len(other_files)} 个非图片文件...")
    
    # 使用临时目录操作 git，避免污染当前目录
    work_dir = Path("/tmp/gh_upload_incremental")
    if work_dir.exists(): subprocess.run(["rm", "-rf", str(work_dir)])
    work_dir.mkdir(parents=True)
    
    # 克隆
    print("   🔄 克隆仓库...")
    repo_url = f"https://github.com/{GITHUB_REPO}.git"
    # 如果配置了 SSH key，优先用 SSH
    ssh_url = f"git@github.com:{GITHUB_REPO}.git"
    
    success, _ = run_cmd(["git", "clone", "-b", GITHUB_BRANCH, ssh_url, str(work_dir)])
    if not success:
        print("   ⚠️ SSH 克隆失败，尝试 HTTPS...")
        run_cmd(["git", "clone", "-b", GITHUB_BRANCH, repo_url, str(work_dir)])
    
    # 复制文件
    for f in other_files:
        rel_path = f.relative_to(CURRENT_DIR)
        target = work_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", str(f), str(target)])
        
    # 提交
    run_cmd(["git", "add", "."], cwd=work_dir)
    success, status = run_cmd(["git", "status", "--porcelain"], cwd=work_dir)
    
    if status.strip():
        print("   💾 提交更改...")
        run_cmd(["git", "commit", "-m", "Update dataset files"], cwd=work_dir)
        print("   ⬆️  推送中...")
        success, err = run_cmd(["git", "push"], cwd=work_dir)
        if success:
            print("✅ GitHub 同步完成")
        else:
            print(f"❌ GitHub 推送失败: {err}")
            return False
    else:
        print("✅ GitHub 已是最新")
        
    return True

if __name__ == "__main__":
    print("="*60)
    print("📦 智能断点续传工具")
    print("="*60)
    
    # 1. 检查 & 分类
    imgs, others = categorize_files()
    
    # 2. 增量上传 HF
    hf_ok = upload_to_huggingface_incremental(imgs)
    
    # 3. 同步 GitHub
    gh_ok = upload_to_github_simple(others)
    
    print("\n" + "="*60)
    if hf_ok and gh_ok:
        print("✨ 全部搞定！早点休息吧！")
    else:
        print("⚠️ 还有点小问题，请检查上方日志。")
