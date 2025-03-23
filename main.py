#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI语音检测器主程序入口点
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# 导入项目模块
from src.data_collection import bilibili_crawler
from src.audio_processing import voice_separator, feature_extractor
from src.model import model_factory
from src.training import trainer
from src.detection import detector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "logs.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AI语音检测器')
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 数据采集子命令
    collect_parser = subparsers.add_parser('collect', help='收集训练数据')
    collect_parser.add_argument('--keywords', type=str, nargs='+', default=['yunxi配音', 'AI配音'],
                               help='B站搜索关键词')
    collect_parser.add_argument('--num_videos', type=int, default=100,
                               help='要下载的视频数量')
    collect_parser.add_argument('--output_dir', type=str, default='data/raw',
                               help='原始数据保存目录')
    
    # 预处理子命令
    preprocess_parser = subparsers.add_parser('preprocess', help='预处理音频数据')
    preprocess_parser.add_argument('--input_dir', type=str, default='data/raw',
                                 help='原始数据目录')
    preprocess_parser.add_argument('--output_dir', type=str, default='data/processed',
                                 help='处理后数据保存目录')
    
    # 训练子命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_dir', type=str, default='data/processed',
                            help='处理后的数据目录')
    train_parser.add_argument('--model_type', type=str, default='cnn_lstm',
                            choices=['cnn_lstm', 'wav2vec', 'ast'],
                            help='模型类型')
    train_parser.add_argument('--batch_size', type=int, default=32,
                            help='批次大小')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='训练轮数')
    train_parser.add_argument('--lr', type=float, default=0.001,
                            help='学习率')
    train_parser.add_argument('--save_dir', type=str, default='models',
                            help='模型保存目录')
    
    # 检测子命令
    detect_parser = subparsers.add_parser('detect', help='检测AI配音')
    detect_parser.add_argument('--model_path', type=str, required=True,
                             help='模型路径')
    detect_parser.add_argument('--input', type=str, required=True,
                             help='输入音频/视频文件或URL')
    detect_parser.add_argument('--output', type=str, default='results',
                             help='结果保存目录')
    
    # 启动窗口应用
    app_parser = subparsers.add_parser('app', help='启动GUI应用')
    app_parser.add_argument('--model_path', type=str, required=True,
                          help='模型路径')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if args.command == 'collect':
        logger.info(f"开始从B站采集数据，关键词: {args.keywords}, 数量: {args.num_videos}")
        bilibili_crawler.collect_videos(args.keywords, args.num_videos, args.output_dir)
    
    elif args.command == 'preprocess':
        logger.info(f"开始预处理音频数据: {args.input_dir} -> {args.output_dir}")
        # 调用音频处理模块进行预处理
        voice_separator.separate_voices(args.input_dir, args.output_dir)
        feature_extractor.extract_features(args.output_dir)
    
    elif args.command == 'train':
        logger.info(f"开始训练模型: {args.model_type}")
        # 创建模型
        model = model_factory.create_model(args.model_type)
        # 训练模型
        trainer.train_model(
            model, 
            args.data_dir, 
            args.batch_size, 
            args.epochs, 
            args.lr, 
            args.save_dir
        )
    
    elif args.command == 'detect':
        logger.info(f"检测AI配音: {args.input}")
        detector.detect_ai_voice(args.model_path, args.input, args.output)
    
    elif args.command == 'app':
        logger.info("启动GUI应用")
        # 引入GUI模块并启动应用
        from src.detection import gui_app
        gui_app.run(args.model_path)
    
    else:
        logger.error("请指定有效的子命令. 使用 --help 查看帮助信息.")

if __name__ == "__main__":
    # 创建日志目录
    if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
        os.makedirs(os.path.join(ROOT_DIR, "logs"))
    main() 