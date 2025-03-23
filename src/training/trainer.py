#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练模块
实现模型训练、验证和评估
"""

import os
import json
import time
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AudioFeatureDataset(Dataset):
    """
    音频特征数据集
    """
    def __init__(
        self,
        features_list: List[Dict],
        feature_type: str = 'mel_spectrogram',
        transform: Optional[Any] = None
    ):
        """
        初始化数据集
        
        Args:
            features_list: 特征字典列表
            feature_type: 使用的特征类型 ('mel_spectrogram', 'mfcc', 'combined')
            transform: 数据增强/变换函数
        """
        self.features_list = features_list
        self.feature_type = feature_type
        self.transform = transform
        
        # 标签映射: ai -> 1, human -> 0
        self.label_map = {"ai": 1, "human": 0}
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        feature_dict = self.features_list[idx]
        
        # 获取指定类型的特征
        if self.feature_type == 'mel_spectrogram':
            feature = feature_dict['mel_spectrogram'].astype(np.float32)
        elif self.feature_type == 'mfcc':
            feature = feature_dict['mfcc'].astype(np.float32)
        elif self.feature_type == 'combined':
            # 组合多种特征，这里简单拼接MFCC和谱对比度
            mfcc = feature_dict['mfcc'].astype(np.float32)
            contrast = feature_dict['spectral_contrast'].astype(np.float32)
            
            # 确保两个特征的时间维度相同
            min_time_dim = min(mfcc.shape[1], contrast.shape[1])
            feature = np.vstack([mfcc[:, :min_time_dim], contrast[:, :min_time_dim]])
        else:
            raise ValueError(f"不支持的特征类型: {self.feature_type}")
        
        # 添加通道维度
        feature = np.expand_dims(feature, axis=0)
        
        # 数据变换/增强
        if self.transform:
            feature = self.transform(feature)
        
        # 获取标签
        label = self.label_map[feature_dict['label']]
        
        return feature, label

def create_dataloaders(
    features_dir: str,
    batch_size: int = 32,
    feature_type: str = 'mel_spectrogram',
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        features_dir: 特征目录
        batch_size: 批次大小
        feature_type: 特征类型
        num_workers: 数据加载线程数
        
    Returns:
        训练数据加载器和验证数据加载器
    """
    # 加载训练和验证特征
    train_path = os.path.join(features_dir, "train_features.pkl")
    val_path = os.path.join(features_dir, "val_features.pkl")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error(f"训练或验证集文件不存在: {train_path}, {val_path}")
        raise FileNotFoundError(f"训练或验证集文件不存在")
    
    with open(train_path, "rb") as f:
        train_features = pickle.load(f)
    
    with open(val_path, "rb") as f:
        val_features = pickle.load(f)
    
    logger.info(f"加载 {len(train_features)} 训练样本和 {len(val_features)} 验证样本")
    
    # 创建数据集
    train_dataset = AudioFeatureDataset(train_features, feature_type=feature_type)
    val_dataset = AudioFeatureDataset(val_features, feature_type=feature_type)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        
    Returns:
        训练指标字典
    """
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    for batch_idx, (features, labels) in enumerate(tqdm(dataloader, desc="训练")):
        features = features.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累加损失
        total_loss += loss.item()
        
        # 收集预测结果和标签
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(preds)
        targets.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary')
    recall = recall_score(targets, predictions, average='binary')
    f1 = f1_score(targets, predictions, average='binary')
    
    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return metrics

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        验证指标字典
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="验证"):
            features = features.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 累加损失
            total_loss += loss.item()
            
            # 收集预测结果和标签
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            targets.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary')
    recall = recall_score(targets, predictions, average='binary')
    f1 = f1_score(targets, predictions, average='binary')
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets, predictions)
    
    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }
    
    return metrics

def plot_training_history(history: Dict[str, List], save_path: str) -> None:
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        save_path: 图表保存路径
    """
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(
    model: nn.Module,
    data_dir: str,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 0.001,
    save_dir: str = 'models',
    feature_type: str = 'mel_spectrogram'
) -> nn.Module:
    """
    训练模型
    
    Args:
        model: 模型
        data_dir: 数据目录
        batch_size: 批次大小
        epochs: 训练轮数
        lr: 学习率
        save_dir: 模型保存目录
        feature_type: 特征类型
        
    Returns:
        训练好的模型
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 创建数据加载器
    features_dir = os.path.join(data_dir, "features")
    train_loader, val_loader = create_dataloaders(
        features_dir,
        batch_size=batch_size,
        feature_type=feature_type
    )
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"训练指标: {train_metrics}")
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"验证指标: {val_metrics}")
        
        # 更新学习率调度器
        scheduler.step(val_metrics['loss'])
        
        # 保存历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_path = os.path.join(save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'feature_type': feature_type
            }, best_model_path)
            logger.info(f"保存最佳模型: {best_model_path}, F1: {best_val_f1:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, f"final_model.pth")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'feature_type': feature_type
    }, final_model_path)
    logger.info(f"保存最终模型: {final_model_path}")
    
    # 绘制训练历史
    history_path = os.path.join(save_dir, "training_history.png")
    plot_training_history(history, history_path)
    
    # 保存训练历史
    history_json_path = os.path.join(save_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    return model

if __name__ == "__main__":
    # 简单测试
    from src.model.model_factory import create_model
    
    model = create_model('cnn_lstm')
    data_dir = "../../data/processed"
    save_dir = "../../models"
    
    train_model(model, data_dir, batch_size=32, epochs=2, lr=0.001, save_dir=save_dir) 