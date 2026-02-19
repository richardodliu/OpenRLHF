#!/bin/bash
# 推送 reinforce_pro_max 分支到我的仓库
# 用法: ./push_to_origin.sh

set -e  # 遇到错误立即退出

# 排除列表：这些文件不会被 git add
EXCLUDE_FILES=(
    "AGENTS.md"
    "CLAUDE.md"
    "PREFIX_CUMULATIVE_IS.md"
    "PREFIX_CUMULATIVE_IS_verification.md"
    "REINFORCE_PRO_MAX.md"
    "REINFORCE_PRO_MAX.md"
    "REINFORCE_PRO_MAX_issues.md"
    "push_to_origin.sh"
    "sync_upstream.sh"
    "test_n_design.py"
    "test_reinforce_pro_max.py"
)

echo "=== 推送 reinforce_pro_max 到 origin ==="

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
    echo ""
    echo "检测到未提交的更改:"
    git status --short
    echo ""
    read -p "是否先提交这些更改? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "请输入 commit message: " COMMIT_MSG
        # 添加所有文件，但排除列表中的文件
        git add -A
        for file in "${EXCLUDE_FILES[@]}"; do
            git reset HEAD -- "$file" 2>/dev/null || true
        done
        git commit -m "$COMMIT_MSG"
    else
        echo "跳过提交，继续推送..."
    fi
fi

# 3. 显示将要推送的提交
echo ""
echo "将要推送的提交:"
COMMITS_TO_PUSH=$(git log --oneline origin/reinforce_pro_max..HEAD 2>/dev/null | wc -l || echo "0")
if [ "$COMMITS_TO_PUSH" -eq 0 ]; then
    echo "  无新提交需要推送"
else
    git log --oneline origin/reinforce_pro_max..HEAD 2>/dev/null || git log --oneline -5
    echo "  共 $COMMITS_TO_PUSH 个提交"
fi

# 4. 确认推送
echo ""
read -p "确认推送到 origin/reinforce_pro_max? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 检查是否需要 force push (rebase 后通常需要)
    LOCAL_HEAD=$(git rev-parse HEAD)
    REMOTE_HEAD=$(git rev-parse origin/reinforce_pro_max 2>/dev/null || echo "none")

    if [ "$REMOTE_HEAD" = "none" ]; then
        echo "推送新分支..."
        git push -u origin reinforce_pro_max
    elif git merge-base --is-ancestor "$REMOTE_HEAD" "$LOCAL_HEAD" 2>/dev/null; then
        echo "正常推送..."
        git push origin reinforce_pro_max
    else
        echo "检测到 rebase，需要 force push..."
        read -p "确认 force push? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push --force-with-lease origin reinforce_pro_max
        else
            echo "取消推送"
            exit 1
        fi
    fi
    echo ""
    echo "=== 推送完成 ==="
else
    echo "取消推送"
fi
