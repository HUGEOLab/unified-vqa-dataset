# #!/usr/bin/env python3
# """
# 统一上传脚本 (智能增量版)：
# - 自动检测 Hugging Face 已存在文件，跳过重复上传
# - 增加重试机制，防止网络波动中断
# - 修复 GitHub 上传逻辑
# """

# import os
# import subprocess
# import time
# from pathlib import Path
# from typing import List, Tuple, Set
# from huggingface_hub import HfApi


# # ============ 配置信息 ============
# CURRENT_DIR = Path("/mnt/mydev2/M256374/unified-vqa-dataset")
# HF_REPO = "Geojx/unified-vqa-images"
# HF_BRANCH = "main"
# GITHUB_REPO = "HUGEOLab/unified-vqa-dataset"
# GITHUB_BRANCH = "main"

# # 图片扩展名
# IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


# def run_cmd(cmd, cwd=None, check=True):
#     """运行系统命令"""
#     try:
#         result = subprocess.run(
#             cmd, cwd=cwd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         return True, result.stdout
#     except subprocess.CalledProcessError as e:
#         return False, e.stderr


# def get_remote_hf_files(repo_id: str) -> Set[str]:
#     """获取 HF 仓库中已存在的所有文件列表"""
#     print(f"🔍 正在检查 {repo_id} 已有的文件 (用于断点续传)...")
#     try:
#         from huggingface_hub import HfApi
#         api = HfApi()
#         # 获取仓库文件列表
#         files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
#         print(f"✅ 远程仓库已有 {len(files)} 个文件")
#         return set(files)
#     except Exception as e:
#         print(f"⚠️ 无法获取远程列表 (可能是新仓库): {e}")
#         return set()


# def categorize_files() -> Tuple[List[Path], List[Path]]:
#     """分类文件"""
#     image_files = []
#     other_files = []
    
#     images_dir = CURRENT_DIR / "unified-vqa-images"
    
#     print(f"\n📂 扫描本地目录: {CURRENT_DIR}")
    
#     # 扫描图片
#     if images_dir.exists():
#         for root, _, files in os.walk(images_dir):
#             for file in files:
#                 file_path = Path(root) / file
#                 if file_path.suffix.lower() in IMAGE_EXTENSIONS:
#                     image_files.append(file_path)
    
#     # 扫描非图片 (排除 .git 和图片目录)
#     for root, dirs, files in os.walk(CURRENT_DIR):
#         if ".git" in dirs: dirs.remove(".git")
#         if "unified-vqa-images" in dirs: dirs.remove("unified-vqa-images")
        
#         for file in files:
#             file_path = Path(root) / file
#             if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
#                 other_files.append(file_path)

#     return image_files, other_files


# def upload_to_huggingface_incremental(image_files: List[Path]):
#     """
#     使用 upload_large_folder 自动分批上传，解决 >25k 文件限制问题
#     """
#     print(f"\n🚀 [Hugging Face] 正在同步图片目录 (智能分批模式)...")
    
#     # 确保 huggingface_hub 是最新版
#     # 终端运行: pip install -U huggingface_hub
    
#     api = HfApi()
    
#     try:
#         print("   ⏳ 正在计算文件哈希并准备分批提交 (这可能需要几分钟)...")
#         # upload_large_folder 专治 413 Payload Too Large
#         api.upload_large_folder(
#             folder_path=str(CURRENT_DIR / "unified-vqa-images"),
#             repo_id=HF_REPO,
#             repo_type="dataset",
#             # 依然保留过滤器，只上传图片
#             allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"],
#             # 注意：upload_large_folder 不支持自定义 commit_message，因为它会产生多个 commit
#             # 也不需要手动 loop，它内部会自动并发处理
#         )
#         print("✅ Hugging Face 同步完成")
#         return True
#     except Exception as e:
#         print(f"❌ 上传失败: {e}")
#         # 如果是因为版本太旧不支持，提示更新
#         if "has no attribute 'upload_large_folder'" in str(e):
#             print("💡 请更新库: pip install -U huggingface_hub")
#         return False


# def upload_to_github_simple(other_files: List[Path]):
#     """GitHub 上传 (保持简单有效)"""
#     if not other_files:
#         return True
        
#     print(f"\n🚀 [GitHub] 同步 {len(other_files)} 个非图片文件...")
    
#     # 使用临时目录操作 git，避免污染当前目录
#     work_dir = Path("/tmp/gh_upload_incremental")
#     if work_dir.exists(): subprocess.run(["rm", "-rf", str(work_dir)])
#     work_dir.mkdir(parents=True)
    
#     # 克隆
#     print("   🔄 克隆仓库...")
#     repo_url = f"https://github.com/{GITHUB_REPO}.git"
#     # 如果配置了 SSH key，优先用 SSH
#     ssh_url = f"git@github.com:{GITHUB_REPO}.git"
    
#     success, _ = run_cmd(["git", "clone", "-b", GITHUB_BRANCH, ssh_url, str(work_dir)])
#     if not success:
#         print("   ⚠️ SSH 克隆失败，尝试 HTTPS...")
#         run_cmd(["git", "clone", "-b", GITHUB_BRANCH, repo_url, str(work_dir)])
    
#     # 复制文件
#     for f in other_files:
#         rel_path = f.relative_to(CURRENT_DIR)
#         target = work_dir / rel_path
#         target.parent.mkdir(parents=True, exist_ok=True)
#         subprocess.run(["cp", str(f), str(target)])
        
#     # 提交
#     run_cmd(["git", "add", "."], cwd=work_dir)
#     success, status = run_cmd(["git", "status", "--porcelain"], cwd=work_dir)
    
#     if status.strip():
#         print("   💾 提交更改...")
#         run_cmd(["git", "commit", "-m", "Update dataset files"], cwd=work_dir)
#         print("   ⬆️  推送中...")
#         success, err = run_cmd(["git", "push"], cwd=work_dir)
#         if success:
#             print("✅ GitHub 同步完成")
#         else:
#             print(f"❌ GitHub 推送失败: {err}")
#             return False
#     else:
#         print("✅ GitHub 已是最新")
        
#     return True


# if __name__ == "__main__":
#     print("="*60)
#     print("📦 智能断点续传工具")
#     print("="*60)
    
#     # 1. 检查 & 分类
#     imgs, others = categorize_files()
    
#     # 2. 增量上传 HF
#     hf_ok = upload_to_huggingface_incremental(imgs)
    
#     # 3. 同步 GitHub
#     gh_ok = upload_to_github_simple(others)
    
#     print("\n" + "="*60)
#     if hf_ok and gh_ok:
#         print("✨ 全部搞定！早点休息吧！")
#     else:
#         print("⚠️ 还有点小问题，请检查上方日志。")
