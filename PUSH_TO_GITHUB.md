# Push To GitHub

本地仓库已经整理并提交完成：

```bash
cd /Users/dengcanze/Documents/CSST/Codex/github/CSST-cluster-validation
git status
```

当前 remote 已设置为：

```bash
git remote -v
```

目标地址：

```text
https://github.com/chance-deng/CSST-cluster-validation.git
```

## 推荐上传方式

1. 在 GitHub 网页创建空仓库：

   `https://github.com/new`

   仓库名建议：

   `CSST-cluster-validation`

2. 在本机终端运行：

```bash
cd /Users/dengcanze/Documents/CSST/Codex/github/CSST-cluster-validation
git push -u origin main
```

如果本机没有 GitHub HTTPS token，可以改用 SSH：

```bash
git remote set-url origin git@github.com:chance-deng/CSST-cluster-validation.git
git push -u origin main
```

## 当前 Codex 环境状态

Codex 环境已完成：

- repo 结构整理；
- Markdown 报告图片相对路径修复；
- git 初始化；
- 首次提交；
- remote 设置。

但当前环境没有可用 GitHub 登录：

- `gh` 命令不存在；
- SSH 返回 `Permission denied (publickey)`；
- HTTPS push 返回 `could not read Username for 'https://github.com'`。

因此最后一步需要在已经登录 GitHub 的终端中执行。

