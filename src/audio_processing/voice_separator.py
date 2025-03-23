#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频处理模块 - 人声分离器
用于从视频/音频中分离人声和背景音
"""

import os
import json
import logging
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from spleeter.separator import Separator
import webrtcvad

logger = logging.getLogger(__name__)

# VAD配置
VAD_FRAME_DURATION_MS = 30  # 每帧持续时间(毫秒)
VAD_AGGRESSIVENESS = 3  # VAD敏感度(0-3)，越高越严格

def init_voice_separator() -> Separator:
    """
    初始化Spleeter人声分离器
    
    Returns:
        Separator对象
    """
    try:
        # 使用预训练的2stem模型(vocals + accompaniment)
        separator = Separator("spleeter:2stems")
        logger.info("成功初始化Spleeter分离器")
        return separator
    except Exception as e:
        logger.error(f"初始化Spleeter分离器失败: {e}")
        raise

def separate_audio_file(
    separator: Separator,
    audio_path: str,
    output_dir: str
) -> Tuple[str, str]:
    """
    分离单个音频文件的人声和背景音
    
    Args:
        separator: Spleeter分离器
        audio_path: 音频文件路径
        output_dir: 输出目录
        
    Returns:
        人声文件路径和背景音文件路径的元组
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不带扩展名）
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 设置输出路径
    vocals_path = os.path.join(output_dir, f"{filename}_vocals.wav")
    background_path = os.path.join(output_dir, f"{filename}_background.wav")
    
    try:
        # 执行分离
        separator.separate_to_file(
            audio_path,
            output_dir,
            filename_format="{filename}_{instrument}.{codec}"
        )
        
        logger.info(f"成功分离音频: {audio_path}")
        return vocals_path, background_path
    except Exception as e:
        logger.error(f"分离音频失败 {audio_path}: {e}")
        return "", ""

def detect_voice_activity(
    audio_path: str,
    sample_rate: int = 16000,
    frame_duration_ms: int = VAD_FRAME_DURATION_MS,
    aggressiveness: int = VAD_AGGRESSIVENESS
) -> List[Tuple[float, float]]:
    """
    检测音频中的人声活动段
    
    Args:
        audio_path: 音频文件路径
        sample_rate: 采样率
        frame_duration_ms: 帧持续时间(毫秒)
        aggressiveness: VAD敏感度
        
    Returns:
        人声活动段列表，每个元素为(开始时间, 结束时间)的元组
    """
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # 初始化VAD
    vad = webrtcvad.Vad(aggressiveness)
    
    # 计算每帧样本数
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    # 将音频分成固定大小的帧
    frames = []
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        # 转换为16位PCM
        frame = (frame * 32768).astype(np.int16).tobytes()
        frames.append(frame)
    
    # 检测每帧是否为人声
    is_speech = []
    for frame in frames:
        try:
            is_speech.append(vad.is_speech(frame, sample_rate))
        except:
            is_speech.append(False)
    
    # 找出连续的人声段
    segments = []
    in_speech = False
    start = 0
    
    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            # 开始新语音段
            in_speech = True
            start = i
        elif not speech and in_speech:
            # 结束语音段
            in_speech = False
            end = i
            # 转换为时间(秒)
            start_time = start * frame_duration_ms / 1000
            end_time = end * frame_duration_ms / 1000
            segments.append((start_time, end_time))
    
    # 处理最后一段
    if in_speech:
        end = len(is_speech)
        start_time = start * frame_duration_ms / 1000
        end_time = end * frame_duration_ms / 1000
        segments.append((start_time, end_time))
    
    # 合并接近的段
    merged_segments = []
    if segments:
        current_start, current_end = segments[0]
        for start, end in segments[1:]:
            # 如果与上一段间隔小于0.5秒，则合并
            if start - current_end < 0.5:
                current_end = end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_segments.append((current_start, current_end))
    
    return merged_segments

def extract_voice_segments(
    audio_path: str,
    segments: List[Tuple[float, float]],
    output_path: str,
    min_duration: float = 1.0
) -> str:
    """
    根据人声活动段提取音频片段
    
    Args:
        audio_path: 音频文件路径
        segments: 人声活动段列表
        output_path: 输出文件路径
        min_duration: 最小段持续时间(秒)
        
    Returns:
        输出文件路径
    """
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # 过滤太短的段
    valid_segments = [seg for seg in segments if seg[1] - seg[0] >= min_duration]
    
    # 如果没有有效段，返回空
    if not valid_segments:
        logger.warning(f"未检测到有效人声段: {audio_path}")
        return ""
    
    # 提取并连接所有人声段
    extracted_audio = np.array([])
    for start, end in valid_segments:
        # 转换时间为采样点索引
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(audio), end_idx)
        # 提取片段
        segment_audio = audio[start_idx:end_idx]
        # 拼接
        extracted_audio = np.concatenate([extracted_audio, segment_audio])
    
    # 保存提取的人声
    sf.write(output_path, extracted_audio, sr)
    
    # 保存段信息
    segments_info = {
        "original_file": audio_path,
        "segments": valid_segments,
        "total_duration": sum(end - start for start, end in valid_segments)
    }
    
    # 保存段信息到同名JSON文件
    segments_json_path = os.path.splitext(output_path)[0] + "_segments.json"
    with open(segments_json_path, "w", encoding="utf-8") as f:
        json.dump(segments_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"成功提取 {len(valid_segments)} 个人声段，总时长: {segments_info['total_duration']:.2f}秒")
    return output_path

def separate_voices(input_dir: str, output_dir: str) -> None:
    """
    批量处理音频文件，分离人声和检测人声段
    
    Args:
        input_dir: 输入目录，包含原始音频
        output_dir: 输出目录
    """
    # 初始化分离器
    separator = init_voice_separator()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有音频文件
    categories = ["ai", "human", "unknown"]
    
    for category in categories:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        
        # 如果类别目录不存在，跳过
        if not os.path.exists(category_input_dir):
            logger.warning(f"类别目录不存在: {category_input_dir}")
            continue
        
        # 创建输出类别目录
        os.makedirs(category_output_dir, exist_ok=True)
        
        # 查找音频文件
        audio_files = []
        for ext in ['.wav', '.mp3', '.aac', '.m4a', '.flac']:
            audio_files.extend(glob.glob(os.path.join(category_input_dir, f"*{ext}")))
        
        if not audio_files:
            logger.warning(f"未找到音频文件: {category_input_dir}")
            continue
        
        logger.info(f"开始处理 {category} 类别，共 {len(audio_files)} 个文件")
        
        # 处理每个文件
        for audio_file in tqdm(audio_files, desc=f"处理 {category} 类别"):
            # 创建每个文件的输出目录
            filename = os.path.splitext(os.path.basename(audio_file))[0]
            file_output_dir = os.path.join(category_output_dir, filename)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # 分离人声和背景音
            vocals_path, _ = separate_audio_file(separator, audio_file, file_output_dir)
            
            if not vocals_path or not os.path.exists(vocals_path):
                logger.warning(f"分离人声失败: {audio_file}")
                continue
            
            # 检测人声活动段
            segments = detect_voice_activity(vocals_path)
            
            if not segments:
                logger.warning(f"未检测到人声段: {vocals_path}")
                continue
            
            # 提取人声段
            segments_output_path = os.path.join(file_output_dir, f"{filename}_vocals_segments.wav")
            extract_voice_segments(vocals_path, segments, segments_output_path)
    
    logger.info(f"所有音频处理完成，输出目录: {output_dir}")

if __name__ == "__main__":
    # 简单测试
    input_dir = "../../data/raw"
    output_dir = "../../data/processed"
    separate_voices(input_dir, output_dir) 