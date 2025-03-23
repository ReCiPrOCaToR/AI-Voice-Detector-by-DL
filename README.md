# AI配音检测器 (AI Voice Detector)

完全由cursor开发，自娱自乐产品，基于深度学习的AI配音检测系统，可用于检测视频中是否使用了AI生成的配音。

## 项目背景

随着AI语音合成技术的快速发展，越来越多的内容创作者开始使用AI生成的配音来制作视频。虽然这提高了创作效率，但同时也带来了信息真实性的问题。本项目旨在开发一个能够准确检测视频中是否使用了AI生成配音的工具，帮助观众辨别内容的真实性。

## 主要功能

- **音频/视频文件检测**：分析本地文件中是否包含AI生成的配音
- **实时监听检测**：实时检测系统播放的音频中是否包含AI配音
- **浏览器扩展**：在B站等视频平台上直接检测视频中的AI配音
- **批量处理**：支持批量检测多个音频/视频文件
- **可视化结果**：提供检测结果的可视化展示

## 技术实现

- **深度学习框架**：PyTorch
- **模型架构**：CNN+LSTM、Wav2Vec 2.0、AST (Audio Spectrogram Transformer)
- **音频处理**：Librosa, Spleeter, WebRTC VAD
- **特征提取**：MFCC, Mel频谱图, 音高特征
- **GUI界面**：PyQt5

## 安装指南

### 环境要求

- Python 3.8+
- CUDA支持（用于GPU加速，推荐但非必须）
- FFmpeg（用于音频/视频处理）

### 安装步骤

1. 克隆项目仓库：

```bash
git clone https://github.com/yourusername/AI-Voice-Detector-by-DL.git
cd AI-Voice-Detector-by-DL
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 安装FFmpeg（如果尚未安装）：

Windows:
```
下载FFmpeg并添加到系统PATH
```

Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

macOS:
```bash
brew install ffmpeg
```

## 使用方法

### 桌面应用

1. 启动应用：

```bash
python -m src.detection.gui_app
```

2. 进入应用后可以：
   - 在"实时检测"选项卡开始实时监听
   - 在"文件检测"选项卡选择要分析的文件
   - 在"设置"选项卡调整检测参数

### 命令行使用

检测单个文件：

```bash
python -m src.detection.cli_detector --input path/to/video.mp4 --output results/
```

批量检测：

```bash
python -m src.detection.cli_detector --input path/to/folder/ --output results/ --batch
```

### 浏览器扩展

1. 在Chrome扩展管理页面启用开发者模式
2. 加载解压缩的扩展程序，选择 `src/detection/browser_extension` 目录
3. 在B站等视频网站上使用扩展检测视频中的AI配音

## 模型训练

如需自行训练模型，请按以下步骤操作：

1. 准备数据集：

```bash
python -m src.data_collection.collect_data --config config/data_collection.yaml
```

2. 特征提取：

```bash
python -m src.audio_processing.feature_extraction --input data/raw/ --output data/features/
```

3. 训练模型：

```bash
python -m src.training.trainer --config config/training.yaml
```

## 贡献指南

欢迎贡献代码、报告问题或提出新功能建议！请遵循以下步骤：

1. Fork本项目仓库
2. 创建特性分支：`git checkout -b feature/your-feature-name`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature-name`
5. 提交Pull Request

## 许可证

本项目采用MIT许可证，详情请参阅LICENSE文件。

## 致谢

- 感谢所有AI配音技术的研究者和开发者
- 感谢开源社区提供的工具和库
- 感谢所有为本项目贡献的人 
