#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI语音检测模块
实现AI配音检测功能
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model.model_factory import create_model
from src.audio_processing.voice_separator import detect_voice_activity, extract_voice_segments
from src.audio_processing.feature_extractor import extract_mel_spectrogram, extract_mfcc

logger = logging.getLogger(__name__)

class AIVoiceDetector:
    """
    AI语音检测器
    """
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径
            device: 计算设备
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model, self.feature_type = self._load_model(model_path)
        self.model.eval()
        
        logger.info(f"初始化AI语音检测器，使用模型: {model_path}, 特征类型: {self.feature_type}")
    
    def _load_model(self, model_path: str) -> Tuple[nn.Module, str]:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            模型和特征类型
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 获取特征类型
            feature_type = checkpoint.get('feature_type', 'mel_spectrogram')
            
            # 创建模型
            model_type = 'cnn_lstm'  # 默认模型类型
            if 'model_type' in checkpoint:
                model_type = checkpoint['model_type']
            
            model = create_model(model_type)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            logger.info(f"成功加载模型: {model_path}")
            return model, feature_type
        
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        提取特征
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            特征
        """
        if self.feature_type == 'mel_spectrogram':
            features = extract_mel_spectrogram(audio, sr=sr)
        elif self.feature_type == 'mfcc':
            features = extract_mfcc(audio, sr=sr)
        else:
            # 默认使用梅尔频谱图
            features = extract_mel_spectrogram(audio, sr=sr)
        
        # 添加通道维度
        features = np.expand_dims(features, axis=0)
        
        return features
    
    def detect_segment(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        检测单个音频片段
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            检测结果
        """
        # 确保音频长度足够
        if len(audio) < sr:  # 小于1秒的音频
            padding = np.zeros(sr - len(audio))
            audio = np.concatenate([audio, padding])
        
        # 提取特征
        features = self._extract_features(audio, sr)
        
        # 转换为张量
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 获取结果
        is_ai = bool(np.argmax(probs))
        ai_prob = float(probs[1])
        human_prob = float(probs[0])
        
        result = {
            "is_ai": is_ai,
            "ai_probability": ai_prob,
            "human_probability": human_prob
        }
        
        return result
    
    def detect_audio(
        self, 
        audio_path: str, 
        output_path: Optional[str] = None,
        threshold: float = 0.5,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        检测整个音频文件
        
        Args:
            audio_path: 音频文件路径
            output_path: 输出结果路径
            threshold: AI判断阈值
            visualize: 是否可视化结果
            
        Returns:
            检测结果
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # 检测人声段
        voice_segments = detect_voice_activity(audio_path)
        
        if not voice_segments:
            logger.warning(f"未检测到人声段: {audio_path}")
            result = {
                "file_path": audio_path,
                "is_ai": False,
                "ai_probability": 0.0,
                "human_probability": 1.0,
                "segments": []
            }
            return result
        
        # 逐段检测
        segment_results = []
        
        for i, (start, end) in enumerate(voice_segments):
            # 提取片段
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            segment_audio = audio[start_idx:end_idx]
            
            # 跳过过短片段
            if len(segment_audio) < sr * 0.5:  # 小于0.5秒
                continue
            
            # 检测
            detection = self.detect_segment(segment_audio, sr)
            
            # 保存结果
            segment_result = {
                "segment_id": i,
                "start_time": start,
                "end_time": end,
                "duration": end - start,
                "is_ai": detection["is_ai"],
                "ai_probability": detection["ai_probability"],
                "human_probability": detection["human_probability"]
            }
            
            segment_results.append(segment_result)
        
        # 计算整体结果
        if segment_results:
            # 加权计算，较长的片段权重更高
            total_duration = sum(seg["duration"] for seg in segment_results)
            weighted_ai_prob = sum(seg["ai_probability"] * seg["duration"] for seg in segment_results) / total_duration
            
            is_ai = weighted_ai_prob >= threshold
        else:
            weighted_ai_prob = 0.0
            is_ai = False
        
        # 整合结果
        result = {
            "file_path": audio_path,
            "is_ai": is_ai,
            "ai_probability": weighted_ai_prob,
            "human_probability": 1 - weighted_ai_prob,
            "segments": segment_results
        }
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 可视化
        if visualize and output_path:
            self._visualize_result(audio, sr, result, output_path.replace(".json", ".png"))
        
        return result
    
    def _visualize_result(
        self, 
        audio: np.ndarray, 
        sr: int, 
        result: Dict[str, Any], 
        output_path: str
    ) -> None:
        """
        可视化检测结果
        
        Args:
            audio: 音频数据
            sr: 采样率
            result: 检测结果
            output_path: 输出图像路径
        """
        # 计算波形图
        plt.figure(figsize=(15, 10))
        
        # 波形图
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(audio, sr=sr)
        plt.title("Audio Waveform")
        
        # 标记人声段
        for segment in result["segments"]:
            start = segment["start_time"]
            end = segment["end_time"]
            color = "red" if segment["is_ai"] else "green"
            plt.axvspan(start, end, alpha=0.3, color=color)
        
        # 梅尔频谱图
        plt.subplot(3, 1, 2)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        
        # AI概率图
        plt.subplot(3, 1, 3)
        segment_starts = [segment["start_time"] for segment in result["segments"]]
        segment_ends = [segment["end_time"] for segment in result["segments"]]
        segment_probs = [segment["ai_probability"] for segment in result["segments"]]
        
        # 绘制段级别AI概率
        for i in range(len(segment_starts)):
            plt.plot([segment_starts[i], segment_ends[i]], [segment_probs[i], segment_probs[i]], 'r-', linewidth=2)
        
        # 添加阈值线
        plt.axhline(y=0.5, color='b', linestyle='--')
        
        plt.xlim(0, len(audio) / sr)
        plt.ylim(0, 1)
        plt.xlabel("Time (s)")
        plt.ylabel("AI Probability")
        plt.title(f"AI Voice Detection (Overall: {result['ai_probability']:.2f})")
        
        # 添加标题
        plt.suptitle(f"AI Voice Detection Result: {'AI' if result['is_ai'] else 'Human'} ({result['ai_probability']:.2f})")
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def detect_ai_voice(
    model_path: str,
    input_path: str,
    output_dir: str,
    threshold: float = 0.5,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    检测AI配音
    
    Args:
        model_path: 模型路径
        input_path: 输入音频或视频文件路径
        output_dir: 输出目录
        threshold: AI判断阈值
        visualize: 是否可视化结果
        
    Returns:
        检测结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化检测器
    detector = AIVoiceDetector(model_path)
    
    # 获取输入文件名
    input_filename = os.path.basename(input_path)
    name, ext = os.path.splitext(input_filename)
    
    # 判断是否为音频文件
    audio_exts = ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg']
    is_audio = ext.lower() in audio_exts
    
    if is_audio:
        # 如果是音频文件，直接检测
        audio_path = input_path
    else:
        # 如果是视频文件，提取音频
        audio_path = os.path.join(output_dir, f"{name}.wav")
        try:
            import subprocess
            cmd = f'ffmpeg -i "{input_path}" -q:a 0 -map a "{audio_path}" -y'
            subprocess.call(cmd, shell=True)
            logger.info(f"从视频中提取音频: {input_path} -> {audio_path}")
        except Exception as e:
            logger.error(f"提取音频失败: {e}")
            return {"error": str(e)}
    
    # 设置输出路径
    output_json = os.path.join(output_dir, f"{name}_result.json")
    
    # 检测
    try:
        result = detector.detect_audio(
            audio_path,
            output_path=output_json,
            threshold=threshold,
            visualize=visualize
        )
        
        logger.info(f"检测完成: {input_path}, 结果: {'AI' if result['is_ai'] else '人声'}, 概率: {result['ai_probability']:.2f}")
        return result
    
    except Exception as e:
        logger.error(f"检测失败: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # 简单测试
    model_path = "../../models/best_model.pth"
    input_path = "../../data/samples/test.wav"
    output_dir = "../../results"
    
    detect_ai_voice(model_path, input_path, output_dir) 