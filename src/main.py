#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI配音检测器主入口
集成了数据采集、预处理、特征提取、模型训练和检测功能
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ai_voice_detector.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AI配音检测器")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="操作命令")
    
    # collect命令 - 数据采集
    collect_parser = subparsers.add_parser("collect", help="采集数据")
    collect_subparsers = collect_parser.add_subparsers(dest="collect_type", help="采集类型")
    
    # 从链接采集
    links_parser = collect_subparsers.add_parser("links", help="从B站视频链接采集")
    links_group = links_parser.add_mutually_exclusive_group(required=True)
    links_group.add_argument("--links", "-l", type=str, nargs="+", help="B站视频链接列表，多个链接用空格分隔")
    links_group.add_argument("--file", "-f", type=str, help="包含B站视频链接的文本文件，每行一个链接")
    links_parser.add_argument("--output_dir", "-o", type=str, default="data/raw", help="输出目录，默认为data/raw")
    links_parser.add_argument("--category", "-c", type=str, choices=["ai", "human", "auto"], default="auto", help="视频分类，默认为自动判断")
    links_parser.add_argument("--max_duration", "-d", type=int, default=300, help="最大视频时长（秒），默认300秒")
    links_parser.add_argument("--sleep", "-s", type=float, default=1.0, help="请求间隔（秒），防止被封IP，默认1秒")
    links_parser.add_argument("--cookie", type=str, default=None, help="B站cookie文件路径，提供更稳定的访问")
    links_parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    
    # 关键词搜索采集
    keywords_parser = collect_subparsers.add_parser("keywords", help="通过关键词搜索采集")
    keywords_parser.add_argument("--keywords", "-k", type=str, nargs="+", required=True, help="搜索关键词，多个关键词用空格分隔")
    keywords_parser.add_argument("--num_videos", "-n", type=int, default=50, help="每个关键词采集的视频数量，默认50")
    keywords_parser.add_argument("--output_dir", "-o", type=str, default="data/raw", help="输出目录，默认为data/raw")
    keywords_parser.add_argument("--max_duration", "-d", type=int, default=300, help="最大视频时长（秒），默认300秒")
    keywords_parser.add_argument("--sleep", "-s", type=float, default=1.0, help="请求间隔（秒），防止被封IP，默认1秒")
    keywords_parser.add_argument("--cookie", type=str, default=None, help="B站cookie文件路径，提供更稳定的访问")
    keywords_parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    
    # preprocess命令 - 数据预处理
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理数据")
    preprocess_parser.add_argument("--input_dir", "-i", type=str, default="data/raw", help="原始数据目录，默认为data/raw")
    preprocess_parser.add_argument("--output_dir", "-o", type=str, default="data/processed", help="处理后数据目录，默认为data/processed")
    preprocess_parser.add_argument("--vad_mode", type=int, choices=[0, 1, 2, 3], default=3, help="VAD模式，0-3，值越大越严格，默认3")
    preprocess_parser.add_argument("--vad_window", type=int, default=30, help="VAD窗口大小（毫秒），默认30ms")
    preprocess_parser.add_argument("--min_speech_duration", type=float, default=0.5, help="最小语音片段时长（秒），默认0.5s")
    preprocess_parser.add_argument("--max_silence_duration", type=float, default=0.5, help="最大静音时长（秒），默认0.5s")
    preprocess_parser.add_argument("--num_workers", "-w", type=int, default=4, help="并行工作线程数，默认4")
    preprocess_parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    
    # extract命令 - 特征提取
    extract_parser = subparsers.add_parser("extract", help="提取特征")
    extract_parser.add_argument("--processed_dir", "-p", type=str, default="data/processed", help="处理后数据目录，默认为data/processed")
    extract_parser.add_argument("--output_dir", "-o", type=str, default="data/features", help="特征输出目录，默认为data/features")
    extract_parser.add_argument("--feature_type", "-t", type=str, choices=["mfcc", "mel", "all"], default="mel", help="特征类型，默认为mel频谱图")
    extract_parser.add_argument("--segment_duration", "-d", type=float, default=3.0, help="音频分段时长（秒），默认3.0s")
    extract_parser.add_argument("--overlap", type=float, default=0.5, help="分段重叠比例，默认0.5")
    extract_parser.add_argument("--sample_rate", "-sr", type=int, default=16000, help="采样率，默认16000Hz")
    extract_parser.add_argument("--n_mfcc", type=int, default=40, help="MFCC系数数量，默认40")
    extract_parser.add_argument("--n_mels", type=int, default=128, help="Mel滤波器组数量，默认128")
    extract_parser.add_argument("--num_workers", "-w", type=int, default=4, help="并行工作线程数，默认4")
    extract_parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    
    # train命令 - 模型训练
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data_dir", "-d", type=str, default="data/features", help="特征数据目录，默认为data/features")
    train_parser.add_argument("--model_type", "-m", type=str, choices=["cnn_lstm", "wav2vec", "ast"], default="cnn_lstm", help="模型类型，默认为cnn_lstm")
    train_parser.add_argument("--batch_size", "-b", type=int, default=32, help="批次大小，默认32")
    train_parser.add_argument("--epochs", "-e", type=int, default=50, help="训练轮数，默认50")
    train_parser.add_argument("--lr", type=float, default=0.001, help="学习率，默认0.001")
    train_parser.add_argument("--save_dir", "-s", type=str, default="models", help="模型保存目录，默认为models")
    train_parser.add_argument("--val_split", type=float, default=0.2, help="验证集比例，默认0.2")
    train_parser.add_argument("--patience", "-p", type=int, default=10, help="早停耐心值，默认10")
    train_parser.add_argument("--gpu", "-g", type=int, default=0, help="使用的GPU编号，默认0，-1表示使用CPU")
    train_parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    
    # app命令 - 启动GUI应用
    app_parser = subparsers.add_parser("app", help="启动GUI应用")
    app_parser.add_argument("--model_path", "-m", type=str, default=None, help="模型路径，默认使用models/best_model.pth")
    
    # detect命令 - 检测文件
    detect_parser = subparsers.add_parser("detect", help="检测文件")
    detect_parser.add_argument("--input", "-i", type=str, required=True, help="输入文件或目录路径")
    detect_parser.add_argument("--output", "-o", type=str, default="./results", help="输出结果目录，默认为./results")
    detect_parser.add_argument("--model", "-m", type=str, default=None, help="模型路径，默认使用内置最佳模型")
    detect_parser.add_argument("--threshold", "-t", type=float, default=0.5, help="AI检测阈值，范围0-1，默认0.5")
    detect_parser.add_argument("--batch", "-b", action="store_true", help="批量处理模式，当输入为目录时使用")
    detect_parser.add_argument("--recursive", "-r", action="store_true", help="递归处理子目录，与--batch一起使用")
    detect_parser.add_argument("--visualize", "-v", action="store_true", help="生成可视化结果")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    if args.command is None:
        logger.error("请指定操作命令。使用 -h 查看帮助。")
        sys.exit(1)
    
    # 采集数据
    if args.command == "collect":
        if args.collect_type == "links":
            from data_collection.collect_from_links import collect_data_from_links
            
            # 获取视频链接
            if args.links:
                links = args.links
            else:  # args.file
                try:
                    with open(args.file, "r", encoding="utf-8") as f:
                        links = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    logger.error(f"读取链接文件时出错: {e}")
                    sys.exit(1)
            
            if not links:
                logger.error("未提供视频链接")
                sys.exit(1)
            
            logger.info(f"从 {len(links)} 个链接开始采集数据")
            
            # 采集数据
            collect_data_from_links(
                links=links,
                output_dir=args.output_dir,
                category=args.category,
                max_duration=args.max_duration,
                workers=4,  # 固定为4，避免并发请求过多
                sleep_interval=args.sleep,
                cookie_file=args.cookie,
                verbose=args.verbose
            )
        
        elif args.collect_type == "keywords":
            # 这里可以添加关键词搜索采集功能
            logger.error("关键词搜索采集功能暂未实现")
            sys.exit(1)
        
        else:
            logger.error(f"未知的采集类型: {args.collect_type}")
            sys.exit(1)
    
    # 预处理数据
    elif args.command == "preprocess":
        from audio_processing.voice_separator import separate_voices
        
        # 检查输入目录是否存在
        if not os.path.exists(args.input_dir):
            logger.error(f"输入目录不存在: {args.input_dir}")
            sys.exit(1)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 设置日志级别
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # 预处理数据
        logger.info(f"开始预处理数据: {args.input_dir} -> {args.output_dir}")
        
        separate_voices(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            vad_mode=args.vad_mode,
            vad_window=args.vad_window,
            min_speech_duration=args.min_speech_duration,
            max_silence_duration=args.max_silence_duration,
            num_workers=args.num_workers
        )
        
        logger.info("数据预处理完成")
    
    # 提取特征
    elif args.command == "extract":
        from audio_processing.feature_extractor import extract_features_from_directory
        
        # 检查输入目录
        if not os.path.exists(args.processed_dir):
            logger.error(f"处理后数据目录不存在: {args.processed_dir}")
            sys.exit(1)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 设置日志级别
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # 提取特征
        logger.info(f"开始提取特征: {args.processed_dir} -> {args.output_dir}")
        
        extract_features_from_directory(
            input_dir=args.processed_dir,
            output_dir=args.output_dir,
            feature_type=args.feature_type,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            n_mels=args.n_mels,
            num_workers=args.num_workers
        )
        
        logger.info("特征提取完成")
    
    # 训练模型
    elif args.command == "train":
        from training.trainer import train_model
        from model.model_factory import create_model
        
        # 检查数据目录
        if not os.path.exists(args.data_dir):
            logger.error(f"特征数据目录不存在: {args.data_dir}")
            sys.exit(1)
        
        # 创建模型保存目录
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 设置日志级别
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # 设置设备
        import torch
        device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
        
        # 创建模型
        model = create_model(args.model_type)
        
        # 训练模型
        logger.info(f"开始训练模型，类型: {args.model_type}, 设备: {device}")
        
        train_model(
            model=model,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            val_split=args.val_split,
            patience=args.patience,
            device=device
        )
        
        logger.info("模型训练完成")
    
    # 启动GUI应用
    elif args.command == "app":
        from detection.gui_app import run
        
        # 确定模型路径
        if args.model_path is None:
            # 使用默认模型路径
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
            model_path = os.path.join(models_dir, "best_model.pth")
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"默认模型文件不存在: {model_path}")
                logger.error("请使用 --model_path 参数指定模型路径")
                sys.exit(1)
        else:
            model_path = args.model_path
            if not os.path.exists(model_path):
                logger.error(f"指定的模型文件不存在: {model_path}")
                sys.exit(1)
        
        # 启动GUI应用
        logger.info(f"启动GUI应用，使用模型: {model_path}")
        run(model_path)
    
    # 检测文件
    elif args.command == "detect":
        from detection.cli_detector import main as detect_main
        
        # 直接使用cli_detector模块中的main函数
        # 参数已通过命令行传递，不需要再次提供
        detect_main()
    
    else:
        logger.error(f"未知的命令: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 