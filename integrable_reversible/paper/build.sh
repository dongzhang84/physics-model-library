#!/usr/bin/env bash
# 一键编译 latex_cn/main.tex。用法：在本目录下执行  ./build.sh
# 依赖：XeLaTeX（不是 pdflatex）+ BibTeX。

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v xelatex >/dev/null 2>&1; then
    echo "错误：未找到 xelatex。请安装 MacTeX / TeX Live 后重试。" >&2
    exit 1
fi

# 若图还在 docs/，编译前自动拷入本目录
if [ ! -f eight_theorems_toymodel.png ] && [ -f ../docs/eight_theorems_toymodel.png ]; then
    cp ../docs/eight_theorems_toymodel.png .
fi

echo "[1/4] xelatex 第一遍（生成 .aux）……"
xelatex -interaction=nonstopmode main.tex >/dev/null
echo "[2/4] bibtex（解析参考文献）……"
bibtex main >/dev/null || true
echo "[3/4] xelatex 第二遍（写入引用）……"
xelatex -interaction=nonstopmode main.tex >/dev/null
echo "[4/4] xelatex 第三遍（解析交叉引用）……"
xelatex -interaction=nonstopmode main.tex >/dev/null

echo "完成。输出：$(pwd)/main.pdf"
