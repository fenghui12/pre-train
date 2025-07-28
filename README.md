# LoRA 微调与管理助手

这是一个用于 LoRA (Low-Rank Adaptation) 微调大型语言模型并进行管理的工具。它提供了一个直观的图形用户界面 (GUI)，使用户能够轻松地训练自定义 LoRA 模型，并将其合并、转换为 GGUF 格式，最终导入到 Ollama 中进行本地推理。

## 主要功能

- **LoRA 模型训练:**
  - 支持从 Hugging Face Hub 加载基座模型进行 LoRA 微调。
  - 支持在现有 LoRA 适配器上继续训练。
  - 训练过程实时进度和日志显示。
- **模型管理:**
  - 将微调后的 LoRA 适配器与基座模型合并。
  - 将合并后的模型转换为 GGUF 格式，兼容 `llama.cpp` 和 Ollama。
  - 自动导入 GGUF 模型到 Ollama，方便本地部署和推理。
- **推理界面:**
  - 提供一个独立的 GUI 界面，用于加载微调后的模型进行交互式推理。
  - 支持多轮对话和上下文记忆。

## 项目结构

- `train_core.py`: 核心训练逻辑，负责 LoRA 微调的实际执行。
- `train_ui.py`: 训练功能的图形用户界面。
- `inference_core.py`: 核心推理逻辑，负责加载模型和生成响应。
- `inference_ui.py`: 推理功能的图形用户界面。
- `merge_and_import.py`: 模型合并、GGUF 转换和 Ollama 导入的逻辑。
- `requirements.txt`: 项目 Python 依赖项。
- `shangganwenxue.jsonl`, `zhexuejia.jsonl`: 示例数据集。
- `lora_*_finetuned/`: 训练输出目录，包含 LoRA 适配器和训练日志。

## 安装与使用

### 1. 环境准备

确保您的系统安装了 Python 3.9+ 和 `pip`。

建议使用 `conda` 或 `venv` 创建虚拟环境：

```bash
conda create -n lora_env python=3.10
conda activate lora_env
```

### 2. 安装依赖

首先，根据您的 CUDA 版本安装 PyTorch。访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合您系统的命令。例如，对于 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

然后，安装项目所需的其他依赖：

```bash
pip install -r requirements.txt
```

**注意:** `bitsandbytes` 在 Windows 上可能需要额外的编译步骤。如果遇到问题，请参考其官方文档。

### 3. 克隆 `llama.cpp` (用于模型合并与导入)

`merge_and_import.py` 依赖于 `llama.cpp` 仓库中的转换脚本。请将其克隆到项目目录的父目录或您的用户主目录中：

```bash
# 克隆到项目父目录
git clone https://github.com/ggerganov/llama.cpp.git ../llama.cpp

# 或者克隆到用户主目录
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
```

### 4. 运行训练 GUI

```bash
python train_ui.py
```

在训练界面中：

- **新 LoRA 训练:** 选择或输入 Hugging Face 模型 ID，选择训练数据 (JSONL 格式)，指定输出目录，然后点击“开始新 LoRA 训练”。
- **继续训练现有 LoRA:** 选择一个本地已训练的 LoRA 模型目录，选择新的训练数据，指定输出目录，然后点击“继续训练”。

### 5. 运行推理 GUI

```bash
python inference_ui.py
```

在推理界面中：

- 点击“选择目录...”选择一个包含 `final_lora_adapter` 子目录的训练输出目录。
- 点击“加载模型”加载模型。
- 输入指令和输入文本，点击“发送”进行推理。
- 可以选择启用上下文模式以保留对话记忆。

### 6. 模型合并与导入到 Ollama

在 `train_ui.py` 界面中：

- 在“模型管理”部分，选择一个本地已训练的 LoRA 模型目录。
- 点击“合并 LoRA 并导入到 Ollama”。
- 输入您希望在 Ollama 中使用的模型名称。

**注意:** 确保您已安装 Ollama 并正在运行。

## 示例数据格式

训练数据应为 JSONL 格式，每行一个 JSON 对象，包含 `instruction`, `input`, `output` 字段。例如：

```json
{"instruction": "把这句话变成押韵的问句。", "input": "猫咪在睡觉。", "output": "猫咪在睡觉，是否睡得香甜又美妙？"}
{"instruction": "请总结以下文章。", "input": "[文章内容]", "output": "[总结内容]"}
```

## 许可证

本项目采用 MIT 许可证。
