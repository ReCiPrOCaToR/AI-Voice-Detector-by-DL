#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI配音检测器桌面应用
提供图形界面进行实时检测
"""

import os
import sys
import time
import json
import logging
import threading
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from queue import Queue

import numpy as np
import torch
import pyaudio
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QProgressBar,
    QSlider, QTabWidget, QTextEdit, QGroupBox, QFormLayout,
    QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QUrl
from PyQt5.QtGui import QIcon, QPixmap, QFont, QDesktopServices

from .detector import AIVoiceDetector
from ..audio_processing.voice_separator import detect_voice_activity

logger = logging.getLogger(__name__)

# 音频缓冲队列，用于实时检测
audio_buffer = Queue(maxsize=10)

class AudioRecorder(QThread):
    """音频录制线程"""
    finished = pyqtSignal()
    update_status = pyqtSignal(str)
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.is_recording = False
        self.audio_buffer = []
    
    def run(self):
        """运行录音线程"""
        self.is_recording = True
        self.update_status.emit("开始录音")
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # 清空缓冲区
            while not audio_buffer.empty():
                audio_buffer.get()
            
            # 开始录音
            while self.is_recording:
                data = stream.read(self.chunk_size)
                self.audio_buffer.append(data)
                
                # 累积大约1秒的音频进行检测
                if len(self.audio_buffer) >= self.sample_rate // self.chunk_size:
                    # 将音频数据放入队列
                    audio_data = b''.join(self.audio_buffer)
                    if not audio_buffer.full():
                        audio_buffer.put(audio_data)
                    self.audio_buffer = []
            
            # 关闭流
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            self.update_status.emit("录音停止")
            
        except Exception as e:
            logger.error(f"录音失败: {e}")
            self.update_status.emit(f"录音失败: {e}")
        
        self.finished.emit()
    
    def stop(self):
        """停止录音"""
        self.is_recording = False

class DetectionWorker(QThread):
    """检测工作线程"""
    result_ready = pyqtSignal(dict)
    update_status = pyqtSignal(str)
    
    def __init__(self, detector: AIVoiceDetector, sample_rate=16000, chunk_size=1024, channels=1):
        super().__init__()
        self.detector = detector
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.is_running = False
    
    def run(self):
        """运行检测线程"""
        self.is_running = True
        self.update_status.emit("开始检测")
        
        try:
            while self.is_running:
                if not audio_buffer.empty():
                    # 从队列获取音频数据
                    audio_data = audio_buffer.get()
                    
                    # 将字节数据转换为numpy数组
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # 检测
                    result = self.detector.detect_segment(audio_np, self.sample_rate)
                    
                    # 发送结果
                    self.result_ready.emit(result)
                else:
                    # 短暂休眠，避免CPU占用过高
                    time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"检测失败: {e}")
            self.update_status.emit(f"检测失败: {e}")
    
    def stop(self):
        """停止检测"""
        self.is_running = False

class WaveformCanvas(FigureCanvas):
    """音频波形显示画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # 初始化波形图
        self.time_data = np.linspace(0, 1, 1000)
        self.amplitude_data = np.zeros(1000)
        self.line, = self.ax.plot(self.time_data, self.amplitude_data)
        
        # 设置图表
        self.ax.set_ylim(-1, 1)
        self.ax.set_title('音频波形')
        self.ax.set_xlabel('时间 (秒)')
        self.ax.set_ylabel('振幅')
        self.ax.grid(True)
        
        # 指示AI语音的区域
        self.ai_regions = []
        
        self.fig.tight_layout()
    
    def update_waveform(self, audio_data, is_ai=False):
        """更新波形图"""
        # 清除之前的AI区域标记
        for region in self.ai_regions:
            region.remove()
        self.ai_regions = []
        
        # 更新数据
        n = len(audio_data)
        if n > 0:
            self.time_data = np.linspace(0, n / 16000, n)
            self.amplitude_data = audio_data
            self.line.set_xdata(self.time_data)
            self.line.set_ydata(self.amplitude_data)
            self.ax.set_xlim(0, n / 16000)
        
        # 如果检测为AI语音，添加红色背景
        if is_ai:
            region = self.ax.axvspan(0, n / 16000, color='red', alpha=0.3)
            self.ai_regions.append(region)
        
        # 更新画布
        self.fig.canvas.draw()

class AIVoiceDetectorGUI(QMainWindow):
    """AI配音检测器图形界面"""
    def __init__(self, model_path: str):
        super().__init__()
        
        # 加载模型
        self.detector = AIVoiceDetector(model_path)
        
        # 初始化UI
        self.init_ui()
        
        # 工作线程
        self.audio_recorder = None
        self.detection_worker = None
        
        # 实时检测结果历史
        self.detection_history = []
        
        # 设置
        self.settings = {
            'threshold': 0.5,
            'sample_rate': 16000,
            'chunk_size': 1024,
            'channels': 1
        }
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('AI配音检测器')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 选项卡
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # 创建选项卡内容
        realtime_tab = self.create_realtime_tab()
        file_tab = self.create_file_tab()
        settings_tab = self.create_settings_tab()
        about_tab = self.create_about_tab()
        
        # 添加选项卡
        tabs.addTab(realtime_tab, "实时检测")
        tabs.addTab(file_tab, "文件检测")
        tabs.addTab(settings_tab, "设置")
        tabs.addTab(about_tab, "关于")
        
        # 状态栏
        self.statusBar().showMessage('就绪')
        
        # 菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        
        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
    
    def create_realtime_tab(self):
        """创建实时检测选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 波形显示
        self.waveform_canvas = WaveformCanvas(tab, width=7, height=3)
        layout.addWidget(self.waveform_canvas)
        
        # 检测结果显示
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_label = QLabel("未开始检测")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        result_layout.addWidget(self.result_label)
        
        self.probability_bar = QProgressBar()
        self.probability_bar.setRange(0, 100)
        self.probability_bar.setValue(0)
        result_layout.addWidget(self.probability_bar)
        
        layout.addWidget(result_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("开始检测")
        self.start_button.clicked.connect(self.start_realtime_detection)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止检测")
        self.stop_button.clicked.connect(self.stop_realtime_detection)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        layout.addLayout(control_layout)
        
        # 检测历史
        history_group = QGroupBox("检测历史")
        history_layout = QVBoxLayout(history_group)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        history_layout.addWidget(self.history_text)
        
        layout.addWidget(history_group)
        
        return tab
    
    def create_file_tab(self):
        """创建文件检测选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 文件选择
        file_group = QGroupBox("选择文件")
        file_layout = QHBoxLayout(file_group)
        
        self.file_path_label = QLabel("未选择文件")
        file_layout.addWidget(self.file_path_label)
        
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_button)
        
        layout.addWidget(file_group)
        
        # 检测按钮
        detect_button = QPushButton("检测文件")
        detect_button.clicked.connect(self.detect_file)
        layout.addWidget(detect_button)
        
        # 检测进度
        self.file_progress = QProgressBar()
        self.file_progress.setRange(0, 100)
        self.file_progress.setValue(0)
        layout.addWidget(self.file_progress)
        
        # 结果显示
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)
        
        self.file_result_text = QTextEdit()
        self.file_result_text.setReadOnly(True)
        result_layout.addWidget(self.file_result_text)
        
        layout.addWidget(result_group)
        
        return tab
    
    def create_settings_tab(self):
        """创建设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 检测设置
        detection_group = QGroupBox("检测设置")
        detection_layout = QFormLayout(detection_group)
        
        # 阈值设置
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.settings['threshold'] * 100))
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        self.threshold_label = QLabel(f"AI判断阈值: {self.settings['threshold']}")
        detection_layout.addRow(self.threshold_label, self.threshold_slider)
        
        layout.addWidget(detection_group)
        
        # 音频设置
        audio_group = QGroupBox("音频设置")
        audio_layout = QFormLayout(audio_group)
        
        # 采样率
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["8000", "16000", "22050", "44100", "48000"])
        self.sample_rate_combo.setCurrentText(str(self.settings['sample_rate']))
        self.sample_rate_combo.currentTextChanged.connect(self.update_sample_rate)
        audio_layout.addRow("采样率:", self.sample_rate_combo)
        
        # 块大小
        self.chunk_size_combo = QComboBox()
        self.chunk_size_combo.addItems(["512", "1024", "2048", "4096"])
        self.chunk_size_combo.setCurrentText(str(self.settings['chunk_size']))
        self.chunk_size_combo.currentTextChanged.connect(self.update_chunk_size)
        audio_layout.addRow("块大小:", self.chunk_size_combo)
        
        # 通道数
        self.channels_combo = QComboBox()
        self.channels_combo.addItems(["1", "2"])
        self.channels_combo.setCurrentText(str(self.settings['channels']))
        self.channels_combo.currentTextChanged.connect(self.update_channels)
        audio_layout.addRow("通道数:", self.channels_combo)
        
        layout.addWidget(audio_group)
        
        # 保存按钮
        save_button = QPushButton("保存设置")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        return tab
    
    def create_about_tab(self):
        """创建关于选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 应用信息
        info_label = QLabel(
            "<h1>AI配音检测器</h1>"
            "<p>使用深度学习模型检测视频或音频中的AI配音</p>"
            "<p>版本: 1.0</p>"
            "<p>© 2023 AI语音检测团队</p>"
        )
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setTextFormat(Qt.RichText)
        layout.addWidget(info_label)
        
        # 功能说明
        features_group = QGroupBox("功能说明")
        features_layout = QVBoxLayout(features_group)
        
        features_text = QTextEdit()
        features_text.setReadOnly(True)
        features_text.setHtml(
            "<h3>主要功能</h3>"
            "<ul>"
            "<li>实时检测系统声音中的AI配音</li>"
            "<li>分析音频/视频文件中的AI配音</li>"
            "<li>可视化检测结果</li>"
            "<li>查看音频频谱特征</li>"
            "</ul>"
            "<h3>使用说明</h3>"
            "<p>1. <b>实时检测</b>: 点击「开始检测」按钮，程序将开始捕获系统声音并实时分析。</p>"
            "<p>2. <b>文件检测</b>: 在「文件检测」选项卡中选择要分析的音频或视频文件，然后点击「检测文件」。</p>"
            "<p>3. <b>设置</b>: 在「设置」选项卡中调整检测阈值和音频参数。</p>"
        )
        features_layout.addWidget(features_text)
        
        layout.addWidget(features_group)
        
        # 关于模型
        model_group = QGroupBox("模型信息")
        model_layout = QVBoxLayout(model_group)
        
        model_label = QLabel(f"当前使用模型: {os.path.basename(self.detector.model_path)}")
        model_layout.addWidget(model_label)
        
        layout.addWidget(model_group)
        
        return tab
    
    def update_threshold(self):
        """更新阈值设置"""
        value = self.threshold_slider.value() / 100.0
        self.settings['threshold'] = value
        self.threshold_label.setText(f"AI判断阈值: {value:.2f}")
    
    def update_sample_rate(self, value):
        """更新采样率设置"""
        self.settings['sample_rate'] = int(value)
    
    def update_chunk_size(self, value):
        """更新块大小设置"""
        self.settings['chunk_size'] = int(value)
    
    def update_channels(self, value):
        """更新通道数设置"""
        self.settings['channels'] = int(value)
    
    def save_settings(self):
        """保存设置"""
        try:
            # 这里可以将设置保存到文件
            settings_file = os.path.join(os.path.dirname(__file__), "settings.json")
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            
            self.statusBar().showMessage('设置已保存', 3000)
        except Exception as e:
            logger.error(f"保存设置失败: {e}")
            self.statusBar().showMessage(f'保存设置失败: {e}', 3000)
    
    def browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频或视频文件",
            "",
            "媒体文件 (*.wav *.mp3 *.flac *.mp4 *.avi *.mkv *.webm);;所有文件 (*)"
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            self.statusBar().showMessage(f'已选择文件: {os.path.basename(file_path)}', 3000)
    
    def detect_file(self):
        """检测文件"""
        file_path = self.file_path_label.text()
        
        if file_path == "未选择文件":
            QMessageBox.warning(self, "警告", "请先选择要检测的文件")
            return
        
        try:
            # 显示进度
            self.file_progress.setValue(10)
            self.statusBar().showMessage('开始检测文件...', 3000)
            
            # 创建临时输出目录
            temp_dir = tempfile.mkdtemp()
            output_json = os.path.join(temp_dir, "result.json")
            
            # 检测文件
            from .detector import detect_ai_voice
            result = detect_ai_voice(
                self.detector.model_path,
                file_path,
                temp_dir,
                threshold=self.settings['threshold'],
                visualize=True
            )
            
            # 更新进度
            self.file_progress.setValue(90)
            
            # 显示结果
            if 'error' in result:
                self.file_result_text.setPlainText(f"检测失败: {result['error']}")
            else:
                # 格式化结果
                formatted_result = json.dumps(result, indent=2, ensure_ascii=False)
                self.file_result_text.setPlainText(formatted_result)
                
                # 如果有可视化图像，显示
                output_img = os.path.join(temp_dir, "result.png")
                if os.path.exists(output_img):
                    # 这里可以添加显示图像的代码
                    pass
            
            # 完成
            self.file_progress.setValue(100)
            self.statusBar().showMessage(
                f"检测完成，结果: {'AI配音' if result.get('is_ai', False) else '人声配音'}, "
                f"概率: {result.get('ai_probability', 0) * 100:.1f}%",
                5000
            )
            
        except Exception as e:
            logger.error(f"检测文件失败: {e}")
            self.file_progress.setValue(0)
            self.file_result_text.setPlainText(f"检测失败: {e}")
            self.statusBar().showMessage(f'检测失败: {e}', 5000)
    
    def start_realtime_detection(self):
        """开始实时检测"""
        try:
            # 更新UI状态
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.result_label.setText("开始检测...")
            self.history_text.clear()
            self.detection_history = []
            
            # 创建录音线程
            self.audio_recorder = AudioRecorder(
                sample_rate=self.settings['sample_rate'],
                chunk_size=self.settings['chunk_size'],
                channels=self.settings['channels']
            )
            self.audio_recorder.update_status.connect(self.update_status)
            
            # 创建检测线程
            self.detection_worker = DetectionWorker(
                self.detector,
                sample_rate=self.settings['sample_rate'],
                chunk_size=self.settings['chunk_size'],
                channels=self.settings['channels']
            )
            self.detection_worker.result_ready.connect(self.handle_detection_result)
            self.detection_worker.update_status.connect(self.update_status)
            
            # 启动线程
            self.audio_recorder.start()
            self.detection_worker.start()
            
            self.statusBar().showMessage('实时检测已启动', 3000)
            
        except Exception as e:
            logger.error(f"启动实时检测失败: {e}")
            self.statusBar().showMessage(f'启动失败: {e}', 5000)
            self.stop_realtime_detection()
    
    def stop_realtime_detection(self):
        """停止实时检测"""
        # 停止线程
        if self.audio_recorder and self.audio_recorder.isRunning():
            self.audio_recorder.stop()
            self.audio_recorder.wait()
        
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.stop()
            self.detection_worker.wait()
        
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.result_label.setText("检测已停止")
        
        self.statusBar().showMessage('实时检测已停止', 3000)
    
    @pyqtSlot(dict)
    def handle_detection_result(self, result):
        """处理检测结果"""
        try:
            # 添加时间戳
            result['timestamp'] = time.time()
            
            # 保存到历史
            self.detection_history.append(result)
            
            # 更新UI
            if result['is_ai']:
                self.result_label.setText(f"检测结果: AI配音 ({result['ai_probability'] * 100:.1f}%)")
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.result_label.setText(f"检测结果: 人声配音 ({result['human_probability'] * 100:.1f}%)")
                self.result_label.setStyleSheet("color: green; font-weight: bold;")
            
            # 更新概率条
            self.probability_bar.setValue(int(result['ai_probability'] * 100))
            
            # 当AI概率高于阈值时，进度条显示为红色，否则显示为绿色
            if result['ai_probability'] >= self.settings['threshold']:
                self.probability_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            else:
                self.probability_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            
            # 更新历史文本
            timestamp = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
            history_text = f"[{timestamp}] {'AI配音' if result['is_ai'] else '人声配音'} - " \
                          f"AI概率: {result['ai_probability'] * 100:.1f}%, " \
                          f"人声概率: {result['human_probability'] * 100:.1f}%"
            
            self.history_text.append(history_text)
            
            # 更新波形图
            # 这里需要从audio_buffer中获取最新的音频数据
            if not audio_buffer.empty():
                audio_data = audio_buffer.get()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                self.waveform_canvas.update_waveform(audio_np, result['is_ai'])
                # 放回队列
                audio_buffer.put(audio_data)
            
        except Exception as e:
            logger.error(f"处理检测结果失败: {e}")
    
    @pyqtSlot(str)
    def update_status(self, status):
        """更新状态栏消息"""
        self.statusBar().showMessage(status, 3000)
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        # 停止所有线程
        self.stop_realtime_detection()
        event.accept()

def run(model_path: str):
    """运行桌面应用"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = AIVoiceDetectorGUI(model_path)
    window.show()
    
    # 执行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    # 简单测试
    model_path = "../../models/best_model.pth"
    run(model_path) 