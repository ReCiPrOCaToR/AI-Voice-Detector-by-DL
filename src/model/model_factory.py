#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型工厂模块
负责创建不同类型的模型用于AI语音检测
"""

import os
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.models as audio_models

logger = logging.getLogger(__name__)

class CNNLSTM(nn.Module):
    """
    CNN+LSTM混合模型
    使用CNN提取特征，LSTM捕获时序信息
    """
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 128,  # MEL频谱图高度
        input_width: int = 256,   # MEL频谱图宽度
        conv_channels: int = 32,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super(CNNLSTM, self).__init__()
        
        # CNN部分
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(conv_channels, conv_channels*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(conv_channels*2, conv_channels*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channels*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算CNN输出后的尺寸
        cnn_output_height = input_height // 8  # 三次池化，每次减半
        cnn_output_width = input_width // 8
        
        # LSTM部分
        self.lstm_input_size = cnn_output_height * conv_channels * 4
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 分类器
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)  # 双向LSTM，所以是2倍hidden_size
        
    def forward(self, x):
        # 输入x形状: [batch_size, 1, height, width]
        
        # CNN特征提取
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 重塑为LSTM输入形状: [batch_size, width//8, height//8 * channels]
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
        x = x.reshape(batch_size, width, channels * height)  # [batch_size, sequence_length, features]
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出，或者使用注意力机制
        # 这里简单取最后一个时间步
        final_out = lstm_out[:, -1, :]
        
        # 分类
        x = self.dropout(final_out)
        x = self.fc(x)
        
        return x

class Wav2VecClassifier(nn.Module):
    """
    基于预训练Wav2Vec 2.0模型的分类器
    """
    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        fine_tune: bool = False,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(Wav2VecClassifier, self).__init__()
        
        # 加载预训练Wav2Vec 2.0模型
        if pretrained_path and os.path.exists(pretrained_path):
            model_name = pretrained_path
        else:
            # 使用torchaudio里的预训练模型
            model_name = "wav2vec2_base"
        
        try:
            self.wav2vec = audio_models.wav2vec2_model.wav2vec2_base()
            logger.info(f"成功加载Wav2Vec 2.0模型: {model_name}")
        except Exception as e:
            logger.error(f"加载Wav2Vec 2.0模型失败: {e}")
            raise
        
        # 是否微调预训练模型
        for param in self.wav2vec.parameters():
            param.requires_grad = fine_tune
        
        # 获取Wav2Vec 2.0输出特征维度
        self.feature_dim = 768  # Wav2Vec 2.0 base模型的输出维度
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # 输入x形状: [batch_size, sequence_length]
        # Wav2Vec 2.0 要求输入是原始波形
        
        # 提取特征
        with torch.no_grad() if not self.wav2vec.training else torch.enable_grad():
            features, _ = self.wav2vec.extract_features(x)
        
        # 取最后一层特征
        x = features[-1]
        
        # 聚合序列维度
        x = torch.mean(x, dim=1)  # 简单平均池化，也可以用注意力或其他聚合方法
        
        # 分类
        x = self.classifier(x)
        
        return x

class AudioSpectrogramTransformer(nn.Module):
    """
    基于音频频谱图的Transformer模型(AST)
    """
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 128,
        input_width: int = 256,
        patch_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 256,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super(AudioSpectrogramTransformer, self).__init__()
        
        assert input_height % patch_size == 0, "输入高度必须能被patch_size整除"
        assert input_width % patch_size == 0, "输入宽度必须能被patch_size整除"
        
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # 计算patch数量和单个patch的维度
        num_patches = (input_height // patch_size) * (input_width // patch_size)
        patch_dim = input_channels * patch_size * patch_size
        
        # patch切分和线性投影
        self.patch_embedding = nn.Conv2d(
            input_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # 输入x形状: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embedding(x)  # [batch_size, hidden_dim, height/patch_size, width/patch_size]
        
        # 重塑以适应Transformer
        x = x.flatten(2)  # [batch_size, hidden_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, hidden_dim]
        
        # 添加分类token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embedding[:, :(x.size(1))]
        x = self.dropout(x)
        
        # Transformer编码
        x = x.transpose(0, 1)  # [sequence, batch, features]，适应PyTorch Transformer接口
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # [batch, sequence, features]
        
        # 使用CLS token进行分类
        x = x[:, 0]
        
        # 分类
        x = self.mlp_head(x)
        
        return x

def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    创建指定类型的模型
    
    Args:
        model_type: 模型类型('cnn_lstm', 'wav2vec', 'ast')
        **kwargs: 其他模型参数
        
    Returns:
        模型实例
    """
    if model_type == 'cnn_lstm':
        model = CNNLSTM(**kwargs)
    elif model_type == 'wav2vec':
        model = Wav2VecClassifier(**kwargs)
    elif model_type == 'ast':
        model = AudioSpectrogramTransformer(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    logger.info(f"创建模型: {model_type}")
    return model

if __name__ == "__main__":
    # 简单测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试CNN+LSTM模型
    cnn_lstm = create_model('cnn_lstm')
    x = torch.randn(4, 1, 128, 256)  # [batch_size, channels, height, width]
    output = cnn_lstm(x)
    print(f"CNN+LSTM输出形状: {output.shape}")
    
    # 测试AST模型
    ast = create_model('ast')
    output = ast(x)
    print(f"AST输出形状: {output.shape}")
    
    # 测试Wav2Vec模型
    wav2vec = create_model('wav2vec')
    x_wav = torch.randn(4, 16000)  # [batch_size, audio_samples]
    output = wav2vec(x_wav)
    print(f"Wav2Vec输出形状: {output.shape}") 