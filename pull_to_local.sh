#!/bin/bash
# 从上游仓库拉取更新到 reinforce_pro_max 分支
# 用法: ./sync_upstream.sh

set -e  # 遇到错误立即退出

echo "=== 从上游同步 reinforce_pro_max 分支 ==="

# 1. 检查当前分支必须是 reinforce_pro_max
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "reinforce_pro_max" ]; then
    echo "错误: 当前分支是 '$CURRENT_BRANCH'，必须在 'reinforce_pro_max' 分支上运行此脚本"
    echo "请先执行: git checkout reinforce_pro_max"
    exit 1
fi
echo "当前分支: $CURRENT_BRANCH ✓"

# 2. 检查是否有未提交的更改
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[1/4] 检测到本地修改，暂存中..."
    git stash push -m "WIP: auto-stash before sync upstream $(date +%Y%m%d_%H%M%S)"
    STASHED=1
else
    echo "[1/4] 无本地修改需要暂存"
    STASHED=0
fi

# 3. 从上游拉取更新
echo "[2/4] 从 upstream 拉取更新..."
git fetch upstream

# 4. Rebase 到 upstream/main
echo "[3/4] 更新 reinforce_pro_max 分支..."
NEW_COMMITS=$(git log --oneline HEAD..upstream/main | wc -l)
if [ "$NEW_COMMITS" -eq 0 ]; then
    echo "  已是最新，无需更新"
else
    echo "  上游新提交 ($NEW_COMMITS 个):"
    git log --oneline HEAD..upstream/main | head -10
    git rebase upstream/main
    echo "  分支已更新"
fi

# 5. 恢复暂存的更改
if [ "$STASHED" -eq 1 ]; then
    echo "[4/4] 恢复暂存的本地修改..."
    if ! git stash pop; then
        echo "警告: stash pop 有冲突，请手动解决"
        exit 1
    fi
else
    echo "[4/4] 无需恢复暂存"
fi

echo ""
echo "=== 同步完成 ==="
echo "如需推送到 origin，请运行: ./push_to_origin.sh"
