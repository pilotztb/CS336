# 创建github仓库并实现在服务器上连接（通过vscode）

## **完整指南：通过 VS Code 将远程服务器项目连接到 GitHub**

**目标场景**: 你正在使用本地的 VS Code，通过 "Remote - SSH" 扩展连接到一台远程Linux服务器。你希望服务器上的一个项目（例如 `cs336`）能被 Git 管理，并推送到 GitHub 上的一个新仓库。

### **阶段一：基础环境准备 (在服务器上)**

在开始之前，确保你的服务器满足以下条件。

1. **安装 Git**:
   如果你的服务器还没有安装 Git，请先安装。打开服务器的终端，执行：

   ```bash
   # 对于 Ubuntu/Debian 系统
   sudo apt update
   sudo apt install git
   
   # 对于 CentOS/RHEL 系统
   sudo yum install git
   ```

2. **首次配置 Git**:
   告诉 Git 你是谁。这个名字和邮箱会出现在你的每一次提交记录里。

   ```bash
   git config --global user.name "你的GitHub用户名"
   git config --global user.email "你的GitHub注册邮箱"
   ```

### **阶段二：配置服务器与 GitHub 的 SSH 免密连接**

这是解决 `Permission denied (publickey)` 错误的关键，**只需为每台服务器配置一次**。

1. **生成 SSH 密钥对**:
   在服务器的终端里执行（将引号中的邮箱换成你自己的）：

   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

   接下来会提问，**连续按三次 `Enter` 回车键**使用默认设置即可。

2. **将公钥添加到 GitHub**:
   a. **复制公钥内容**：在服务器终端执行以下命令，它会显示出你的公钥。

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

   b. **完整复制**屏幕上显示的所有内容（以 `ssh-ed25519` 开头，以你的邮箱结尾）。

   c. **粘贴到 GitHub**：

     * 登录 GitHub 官网。
     * 点击右上角头像 -\> **Settings**。
     * 左侧菜单选择 -\> **SSH and GPG keys**。
     * 点击绿色按钮 -\> **New SSH key**。
     * **Title**: 任意起一个名字，比如 `My CS336 Server`。
     * **Key**: 粘贴你刚刚复制的公钥内容。
     * 点击 **Add SSH key**。

3. **测试连接**:
   回到服务器终端，执行：

   ```bash
   ssh -T git@github.com
   ```

   如果看到包含你用户名的欢迎语 (`Hi pilotztb! You've successfully authenticated...`)，则表示 SSH 连接配置成功！

### **阶段三：在 GitHub 上创建远程仓库**

1.  登录 GitHub，点击右上角的 `+` 号，选择 **New repository**。
2.  **Repository name**: 填写你的仓库名（例如 `CS336`）。
3.  **关键一步**: **不要勾选** "Add a README file", "Add .gitignore", "Choose a license"。保持仓库为空，这样可以避免第一次推送时出现冲突。
4.  点击 **Create repository**。
5.  创建成功后，页面会显示仓库的地址。找到并**复制 SSH 地址**，它看起来像这样：`git@github.com:pilotztb/CS336.git`。

### **阶段四：在 VS Code 中操作，完成首次代码推送**

现在，所有准备工作都已完成。我们回到 VS Code（已通过SSH连接到服务器）进行最后的操作。

1. **打开项目文件夹和终端**:
   在 VS Code 里，打开你在服务器上的项目文件夹（例如 `/home/ztb/cs336`）。然后通过菜单 `Terminal` -\> `New Terminal` 打开一个终端，它的路径应该就在你的项目文件夹下。

2. **初始化本地仓库**:

   ```bash
   git init
   ```

3. **添加所有文件到暂存区**:

   ```bash
   git add .
   ```

4. **提交文件**:

   ```bash
   git commit -m "Initial commit"
   ```

5. **（重要）将主分支重命名为 `main`**:
   为了解决 `master`/`main` 名称不匹配的问题，并与当前社区标准保持一致，执行：

   ```bash
   git branch -m master main
   ```

6. **关联远程仓库**:
   将你在**阶段三**复制的 SSH 地址关联到本地仓库（`origin` 是远程仓库的默认别名）。

   ```bash
   git remote add origin git@github.com:pilotztb/CS336.git
   ```

7. **推送代码到 GitHub**:
   这是最后一步，将本地的 `main` 分支推送到远程 `origin` 仓库。

   ```bash
   git push -u origin main
   ```

     * `-u` 参数会建立本地 `main` 分支和远程 `main` 分支的联系，以后你再推送时，只需简单地使用 `git push` 即可。

**至此，你的服务器项目已经通过 VS Code 成功连接并推送到了 GitHub 仓库！** 你可以刷新 GitHub 页面，看到你的代码已经上传上去了。

# 通过vscode实现想github仓库push和pull

## 命令行

### **第1步：创建或修改文件**

首先，你需要在你的项目文件夹中进行实际的工作，比如创建新文件或修改已有文件。

```bash
# 示例：创建一个名为 new_file.md 的新文件，并写入内容
echo "这是一些新的笔记内容。" > new_file.md
```

### **第2步：查看状态 (好习惯)**

在执行任何 Git 操作前，先用 `git status` 检查一下仓库的当前状态。

```bash
git status
```

  * 这个命令会告诉你哪些文件是新创建的（untracked files）、哪些文件被修改了。

### **第3步：暂存更改**

决定好要将哪些更改记录下来后，使用 `git add` 命令将它们添加到“暂存区”。

```bash
# 暂存所有更改（最常用）
git add .

# 或者，只暂存某一个文件
# git add new_file.md
```

### **第4步：提交更改**

将暂存区里的所有内容创建一个版本快照（commit），并附上清晰的说明信息。

```bash
# 使用 -m 参数来直接附带提交信息
git commit -m "Add new notes in new_file.md"
```

  * 提交信息（双引号内的内容）应当清晰地描述你这次的改动。

### **第5步：推送到 GitHub**

最后，将你本地仓库中新建的提交推送到 GitHub 远程仓库。

```bash
git push origin main
```

  * `origin` 是你远程仓库的默认名称。
  * `main` 是你要推送到的分支名称。

-----

### **完整流程示例**

从创建文件到推送到 GitHub 的一次完整操作：

```bash
# 1. 在项目中创建一个新文件
echo "def new_feature():\n    return True" > feature.py

# 2. 检查状态，会看到 feature.py 是“未跟踪”
git status

# 3. 暂存所有更改
git add .

# 4. 提交更改
git commit -m "Feat: Add new feature toggle function"

# 5. 推送到 GitHub
git push origin main
```

