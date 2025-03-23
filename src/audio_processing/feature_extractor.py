#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频处理模块 - 特征提取器
提取音频的特征用于模型训练
"""

import os
import json
import logging
import glob
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
import librosa
import torchaudio
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 特征提取配置
SAMPLE_RATE = 16000  # 采样率
N_FFT = 1024  # FFT窗口大小
HOP_LENGTH = 512  # 帧移
N_MELS = 128  # Mel频带数量
SEGMENT_DURATION = 3  # 音频切片长度(秒)
OVERLAP = 1.5  # 重叠长度(秒)

def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    加载音频文件
    
    Args:
        file_path: 音频文件路径
        sr: 目标采样率
        
    Returns:
        音频数据
    """
    try:
        audio, sr_orig = librosa.load(file_path, sr=sr, mono=True)
        return audio
    except Exception as e:
        logger.error(f"加载音频失败 {file_path}: {e}")
        return np.array([])

def extract_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 40,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    提取MFCC特征
    
    Args:
        audio: 音频数据
        sr: 采样率
        n_mfcc: MFCC系数数量
        n_fft: FFT窗口大小
        hop_length: 帧移
        
    Returns:
        MFCC特征
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # 标准化特征
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
    return mfccs

def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS
) -> np.ndarray:
    """
    提取梅尔频谱图特征
    
    Args:
        audio: 音频数据
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: Mel滤波器数量
        
    Returns:
        梅尔频谱图特征
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    # 转换为分贝单位
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # 标准化
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
    return mel_spec_db

def extract_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    提取基频特征
    
    Args:
        audio: 音频数据
        sr: 采样率
        hop_length: 帧移
        
    Returns:
        基频特征
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length
    )
    # 将NaN替换为0
    f0 = np.nan_to_num(f0)
    return f0

def extract_spectral_contrast(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_bands: int = 6
) -> np.ndarray:
    """
    提取谱对比度特征
    
    Args:
        audio: 音频数据
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_bands: 频带数量
        
    Returns:
        谱对比度特征
    """
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands
    )
    return contrast

def segment_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    segment_duration: float = SEGMENT_DURATION,
    overlap: float = OVERLAP,
    min_segment_duration: float = 1.0
) -> List[np.ndarray]:
    """
    将长音频分割成小片段
    
    Args:
        audio: 音频数据
        sr: 采样率
        segment_duration: 片段长度(秒)
        overlap: 重叠长度(秒)
        min_segment_duration: 最小片段长度(秒)
        
    Returns:
        音频片段列表
    """
    # 音频总长(秒)
    total_duration = len(audio) / sr
    
    # 如果音频长度小于最小片段长度，直接返回原音频
    if total_duration < min_segment_duration:
        return []
    
    # 如果音频长度小于片段长度，直接返回原音频
    if total_duration <= segment_duration:
        return [audio]
    
    # 计算采样点
    segment_samples = int(segment_duration * sr)
    hop_samples = int((segment_duration - overlap) * sr)
    
    # 分割音频
    segments = []
    for start in range(0, len(audio) - segment_samples + 1, hop_samples):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
    
    return segments

def extract_features_from_file(file_path: str, label: str) -> List[Dict]:
    """
    从单个音频文件提取特征
    
    Args:
        file_path: 音频文件路径
        label: 标签 (ai 或 human)
        
    Returns:
        特征字典列表
    """
    # 加载音频
    audio = load_audio(file_path)
    if len(audio) == 0:
        logger.warning(f"音频加载失败或为空: {file_path}")
        return []
    
    # 分割音频
    segments = segment_audio(audio)
    if not segments:
        logger.warning(f"音频过短，无法分割: {file_path}")
        return []
    
    # 为每个片段提取特征
    features_list = []
    for i, segment in enumerate(segments):
        # 提取不同特征
        mfcc = extract_mfcc(segment)
        mel_spec = extract_mel_spectrogram(segment)
        pitch = extract_pitch(segment)
        spectral_contrast = extract_spectral_contrast(segment)
        
        # 匹配特征长度（以最短的为准）
        min_length = min(mfcc.shape[1], mel_spec.shape[1], len(pitch), spectral_contrast.shape[1])
        mfcc = mfcc[:, :min_length]
        mel_spec = mel_spec[:, :min_length]
        pitch = pitch[:min_length]
        spectral_contrast = spectral_contrast[:, :min_length]
        
        # 创建特征字典
        feature_dict = {
            "file_path": file_path,
            "segment_id": i,
            "mfcc": mfcc,
            "mel_spectrogram": mel_spec,
            "pitch": pitch,
            "spectral_contrast": spectral_contrast,
            "label": label
        }
        
        features_list.append(feature_dict)
    
    return features_list

def extract_features(processed_dir: str) -> None:
    """
    批量提取特征并保存
    
    Args:
        processed_dir: 预处理后的数据目录
    """
    # 确保输出目录存在
    features_dir = os.path.join(processed_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    # 查找所有音频文件
    categories = ["ai", "human"]
    all_features = []
    
    for category in categories:
        category_dir = os.path.join(processed_dir, category)
        
        # 如果类别目录不存在，跳过
        if not os.path.exists(category_dir):
            logger.warning(f"类别目录不存在: {category_dir}")
            continue
        
        # 遍历每个处理后的视频目录
        video_dirs = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        
        for video_dir in tqdm(video_dirs, desc=f"处理 {category} 类别"):
            video_path = os.path.join(category_dir, video_dir)
            
            # 查找分离出的人声段
            segment_files = glob.glob(os.path.join(video_path, "*_vocals_segments.wav"))
            
            if not segment_files:
                logger.warning(f"未找到人声段文件: {video_path}")
                continue
            
            # 处理每个人声段文件
            for segment_file in segment_files:
                # 提取特征
                features_list = extract_features_from_file(segment_file, category)
                all_features.extend(features_list)
    
    # 保存所有特征
    if all_features:
        logger.info(f"共提取 {len(all_features)} 个特征样本")
        features_path = os.path.join(features_dir, "all_features.pkl")
        with open(features_path, "wb") as f:
            pickle.dump(all_features, f)
        
        # 创建特征数据集信息
        dataset_info = {
            "total_samples": len(all_features),
            "ai_samples": sum(1 for f in all_features if f["label"] == "ai"),
            "human_samples": sum(1 for f in all_features if f["label"] == "human"),
            "feature_types": ["mfcc", "mel_spectrogram", "pitch", "spectral_contrast"]
        }
        
        # 保存数据集信息
        info_path = os.path.join(features_dir, "dataset_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"特征提取完成，保存在 {features_path}")
    else:
        logger.warning("未提取到有效特征")

def create_training_validation_split(
    features_dir: str,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> None:
    """
    创建训练集和验证集划分
    
    Args:
        features_dir: 特征目录
        train_ratio: 训练集比例
        random_seed: 随机种子
    """
    features_path = os.path.join(features_dir, "all_features.pkl")
    
    if not os.path.exists(features_path):
        logger.error(f"特征文件不存在: {features_path}")
        return
    
    # 加载特征
    with open(features_path, "rb") as f:
        all_features = pickle.load(f)
    
    # 按标签分组
    ai_features = [f for f in all_features if f["label"] == "ai"]
    human_features = [f for f in all_features if f["label"] == "human"]
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 打乱数据
    np.random.shuffle(ai_features)
    np.random.shuffle(human_features)
    
    # 划分训练集和验证集
    ai_train_size = int(len(ai_features) * train_ratio)
    human_train_size = int(len(human_features) * train_ratio)
    
    ai_train = ai_features[:ai_train_size]
    ai_val = ai_features[ai_train_size:]
    human_train = human_features[:human_train_size]
    human_val = human_features[human_train_size:]
    
    # 合并训练集和验证集
    train_features = ai_train + human_train
    val_features = ai_val + human_val
    
    # 再次打乱
    np.random.shuffle(train_features)
    np.random.shuffle(val_features)
    
    # 保存训练集和验证集
    train_path = os.path.join(features_dir, "train_features.pkl")
    val_path = os.path.join(features_dir, "val_features.pkl")
    
    with open(train_path, "wb") as f:
        pickle.dump(train_features, f)
    
    with open(val_path, "wb") as f:
        pickle.dump(val_features, f)
    
    # 保存划分信息
    split_info = {
        "total_samples": len(all_features),
        "train_samples": len(train_features),
        "validation_samples": len(val_features),
        "train_ai_samples": len(ai_train),
        "train_human_samples": len(human_train),
        "val_ai_samples": len(ai_val),
        "val_human_samples": len(human_val),
        "train_ratio": train_ratio,
        "random_seed": random_seed
    }
    
    split_info_path = os.path.join(features_dir, "split_info.json")
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据集划分完成: 训练集 {len(train_features)} 样本, 验证集 {len(val_features)} 样本")

if __name__ == "__main__":
    # 简单测试
    processed_dir = "../../data/processed"
    extract_features(processed_dir)
    features_dir = os.path.join(processed_dir, "features")
    create_training_validation_split(features_dir) 