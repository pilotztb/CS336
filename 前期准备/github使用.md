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

# 关于git设计的理论讲解

## **1. `add`的暂存和`commit`的提交是不是都是本地的仓库？**

不。`git add` 操作的目标是**暂存区 (Staging Area)**，而 `git commit` 操作的目标是**本地仓库 (Local Repository)**。它们是本地工作流程中两个不同的、连续的阶段。

在你的本地机器上，Git 管理着三个核心的数据区域：

1.  **工作区 (Working Directory)**
    * 这是你电脑文件系统中实际看到和编辑的项目文件和目录。

2.  **暂存区 (Staging Area / Index)**
    * 这是一个位于 `.git` 目录下的索引文件。它记录了你**下一次准备提交**的文件快照信息。
    * 执行 `git add <文件名>` 命令时，Git 会获取该文件在工作区的当前内容，计算其哈希值，并将这个快照信息记录到暂存区，标记为等待提交。

3.  **本地仓库 (Local Repository)**
    * 这是位于项目根目录下的 `.git` 文件夹。它是一个包含了项目完整历史记录的数据库，里面存储着所有的提交对象 (commits)、树对象 (trees) 和数据对象 (blobs)。
    * 执行 `git commit` 命令时，Git 会使用暂存区中的快照信息，创建一个新的提交对象，并将其永久地保存在本地仓库的版本历史中。

**总结**：`add` 和 `commit` 都是完全在本地机器上执行的操作，但它们作用于两个不同的逻辑区域：`add` 是从工作区到暂存区，`commit` 是从暂存区到本地仓库。

---

## **2. 也就是说有两个仓库一个在本地一个在云？**

是的，这个理解是正确的。Git 是一个分布式版本控制系统，其设计就是基于多个仓库实例。

* **本地仓库 (Local Repository)**
  * 这是一个在你本机上功能完备的仓库。它拥有项目的全部历史记录。所有的版本创建 (`commit`)、分支管理 (`branch`)、历史查看 (`log`) 等核心操作都在本地仓库中完成，无需网络连接。

* **远程仓库 (Remote Repository)**
  * 这是托管在另一台服务器上的仓库实例，例如在 GitHub 上。它的主要目的是作为数据备份和一个中心点，供一个或多个开发者同步各自的本地仓库，从而实现协作。

---

## **3. 确定`push`就是用本地覆盖云？**

不，`git push` 的默认行为不是“覆盖”，而是一个有安全检查的**同步**操作。

* **`push` 的工作原理**：它将你本地仓库中存在、而远程仓库中不存在的提交对象，传输到远程仓库。然后，它会尝试移动远程仓库中的分支指针，使其指向与你本地分支相同的最新提交。

* **安全检查机制 (Fast-forward)**：
  默认情况下，`push` 操作只在“快进模式 (Fast-forward)”下才能成功。这意味着，只有当远程分支的历史记录是你本地分支历史记录的一个子集时（即远程分支没有在你上次同步后产生任何新的提交），`push` 才会成功。你的提交会平滑地“追加”到远程历史的末尾。

* **推送被拒绝 (Non-fast-forward)**：
  如果在你准备推送时，已经有其他新的提交被推送到远程分支，那么远程分支的历史和你本地分支的历史就产生了分叉。此时，`push` 会被拒绝，并提示 `non-fast-forward` 错误。这是 Git 的核心安全特性，它**防止你无意中丢弃或覆盖**他人的提交。Git 会强制你先执行 `git pull` 将远程的新提交拉取到本地，与你的工作合并后，才能再次推送。

* **强制推送 (`--force`)**：
  Git 确实提供了一个可以实现“覆盖”的命令：`git push --force`。这个命令会跳过安全检查，强制让远程分支的指针指向你本地分支的当前提交。这是一个**破坏性操作**，因为它可能会丢弃远程仓库上存在但你本地没有的提交，**在团队协作中应极其谨慎地使用**。

**总结**：`git push` 默认是一个安全的同步操作，旨在增加历史记录，而不是覆盖。