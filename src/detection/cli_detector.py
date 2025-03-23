#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI配音检测器命令行工具
支持单个文件或批量检测
"""

import os
import sys
import json
import logging
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
import pandas as pd
import numpy as np

from .detector import detect_ai_voice

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("detector.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AI配音检测命令行工具")
    
    parser.add_argument("--input", "-i", type=str, required=True,
                      help="输入文件或目录路径")
    
    parser.add_argument("--output", "-o", type=str, default="./results",
                      help="输出结果目录，默认为./results")
    
    parser.add_argument("--model", "-m", type=str, default=None,
                      help="模型路径，默认使用内置最佳模型")
    
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                      help="AI检测阈值，范围0-1，默认0.5")
    
    parser.add_argument("--batch", "-b", action="store_true",
                      help="批量处理模式，当输入为目录时使用")
    
    parser.add_argument("--recursive", "-r", action="store_true",
                      help="递归处理子目录，与--batch一起使用")
    
    parser.add_argument("--workers", "-w", type=int, default=4,
                      help="并行工作线程数，默认4")
    
    parser.add_argument("--visualize", "-v", action="store_true",
                      help="生成可视化结果")
    
    parser.add_argument("--format", "-f", type=str, choices=["json", "csv", "both"], default="json",
                      help="输出格式，支持json或csv或both，默认json")
    
    parser.add_argument("--verbose", action="store_true",
                      help="显示详细输出")
    
    return parser.parse_args()

def detect_single_file(
    file_path: str, 
    model_path: str, 
    output_dir: str, 
    threshold: float = 0.5,
    visualize: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    检测单个文件中的AI配音
    
    Args:
        file_path: 音频或视频文件路径
        model_path: 模型路径
        output_dir: 输出目录
        threshold: AI检测阈值
        visualize: 是否生成可视化结果
        verbose: 是否显示详细日志
    
    Returns:
        检测结果字典
    """
    try:
        # 设置日志级别
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用detector模块的检测函数
        result = detect_ai_voice(
            model_path=model_path,
            audio_path=file_path,
            output_dir=output_dir,
            threshold=threshold,
            visualize=visualize
        )
        
        # 添加文件信息
        result["file_path"] = file_path
        result["file_name"] = os.path.basename(file_path)
        
        # 保存结果到JSON文件
        output_file = os.path.join(
            output_dir, 
            f"{os.path.splitext(os.path.basename(file_path))[0]}_result.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"文件 {file_path} 检测完成，结果: {'AI配音' if result['is_ai'] else '人声配音'}, 概率: {result['ai_probability'] * 100:.1f}%")
        
        return result
    
    except Exception as e:
        logger.error(f"检测文件 {file_path} 时出错: {e}")
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "error": str(e),
            "is_ai": False,
            "ai_probability": 0.0,
            "human_probability": 0.0,
            "success": False
        }

def batch_process(
    input_dir: str,
    model_path: str,
    output_dir: str,
    threshold: float = 0.5,
    recursive: bool = False,
    workers: int = 4,
    visualize: bool = False,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    批量处理目录中的所有音频/视频文件
    
    Args:
        input_dir: 输入目录
        model_path: 模型路径
        output_dir: 输出目录
        threshold: AI检测阈值
        recursive: 是否递归处理子目录
        workers: 并行工作线程数
        visualize: 是否生成可视化结果
        verbose: 是否显示详细日志
    
    Returns:
        所有文件的检测结果列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有音频和视频文件
    patterns = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.mp4", "*.avi", "*.mkv", "*.webm", "*.mov"]
    files = []
    
    if recursive:
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    else:
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    # 去重和排序
    files = sorted(list(set(files)))
    
    if not files:
        logger.warning(f"在目录 {input_dir} 中未找到支持的音频/视频文件")
        return []
    
    logger.info(f"找到 {len(files)} 个文件待处理")
    
    # 创建进度条
    progress_bar = tqdm.tqdm(total=len(files), desc="处理进度")
    
    # 存储所有结果
    all_results = []
    
    # 并行处理文件
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(
                detect_single_file, 
                file_path, 
                model_path, 
                output_dir, 
                threshold, 
                visualize, 
                verbose
            ): file_path for file_path in files
        }
        
        # 处理结果
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
                all_results.append({
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "error": str(e),
                    "is_ai": False,
                    "ai_probability": 0.0,
                    "human_probability": 0.0,
                    "success": False
                })
            
            # 更新进度条
            progress_bar.update(1)
    
    # 关闭进度条
    progress_bar.close()
    
    # 保存汇总结果
    if all_results:
        # JSON格式
        with open(os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # CSV格式
        try:
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False, encoding="utf-8")
        except Exception as e:
            logger.error(f"保存CSV结果时出错: {e}")
    
    # 打印统计信息
    ai_count = sum(1 for r in all_results if r.get("is_ai", False) and not r.get("error", False))
    human_count = sum(1 for r in all_results if not r.get("is_ai", False) and not r.get("error", False))
    error_count = sum(1 for r in all_results if r.get("error", False))
    
    logger.info(f"批量处理完成: 共 {len(all_results)} 个文件")
    logger.info(f"检测结果: AI配音 {ai_count} 个, 人声配音 {human_count} 个, 处理失败 {error_count} 个")
    
    return all_results

def save_summary_report(
    results: List[Dict[str, Any]], 
    output_dir: str, 
    format_type: str = "both"
) -> None:
    """
    保存汇总报告
    
    Args:
        results: 检测结果列表
        output_dir: 输出目录
        format_type: 输出格式，json、csv或both
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算统计信息
    total = len(results)
    ai_count = sum(1 for r in results if r.get("is_ai", False) and not r.get("error", False))
    human_count = sum(1 for r in results if not r.get("is_ai", False) and not r.get("error", False))
    error_count = sum(1 for r in results if r.get("error", False))
    
    # 成功率
    success_rate = (total - error_count) / total if total > 0 else 0
    
    # AI占比
    ai_ratio = ai_count / (ai_count + human_count) if (ai_count + human_count) > 0 else 0
    
    # 平均AI概率
    ai_probs = [r.get("ai_probability", 0) for r in results if not r.get("error", False)]
    avg_ai_prob = sum(ai_probs) / len(ai_probs) if ai_probs else 0
    
    # 创建汇总数据
    summary = {
        "total_files": total,
        "ai_voice_count": ai_count,
        "human_voice_count": human_count,
        "error_count": error_count,
        "success_rate": success_rate,
        "ai_ratio": ai_ratio,
        "average_ai_probability": avg_ai_prob,
        "detection_timestamp": pd.Timestamp.now().isoformat()
    }
    
    # 保存汇总报告
    if format_type in ["json", "both"]:
        with open(os.path.join(output_dir, "summary_report.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    if format_type in ["csv", "both"]:
        # 创建DataFrame并保存为CSV
        df = pd.DataFrame([summary])
        df.to_csv(os.path.join(output_dir, "summary_report.csv"), index=False, encoding="utf-8")
    
    logger.info(f"汇总报告已保存到 {output_dir}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 确定模型路径
    if args.model is None:
        # 使用默认模型路径
        module_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.abspath(os.path.join(module_dir, "..", "..", "models"))
        model_path = os.path.join(models_dir, "best_model.pth")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"默认模型文件不存在: {model_path}")
            logger.error("请使用 --model 参数指定模型路径")
            sys.exit(1)
    else:
        model_path = args.model
        if not os.path.exists(model_path):
            logger.error(f"指定的模型文件不存在: {model_path}")
            sys.exit(1)
    
    logger.info(f"使用模型: {model_path}")
    
    # 处理输入路径
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        logger.error(f"输入路径不存在: {input_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据输入类型确定处理模式
    if os.path.isfile(input_path):
        # 单文件模式
        logger.info(f"开始检测单个文件: {input_path}")
        result = detect_single_file(
            input_path, 
            model_path, 
            output_dir, 
            args.threshold, 
            args.visualize, 
            args.verbose
        )
        
        # 如果指定CSV格式，也保存为CSV
        if args.format in ["csv", "both"]:
            try:
                df = pd.DataFrame([result])
                csv_file = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(os.path.basename(input_path))[0]}_result.csv"
                )
                df.to_csv(csv_file, index=False, encoding="utf-8")
                logger.info(f"CSV结果已保存到: {csv_file}")
            except Exception as e:
                logger.error(f"保存CSV结果时出错: {e}")
        
    elif os.path.isdir(input_path):
        if args.batch:
            # 批量处理模式
            logger.info(f"开始批量处理目录: {input_path}")
            results = batch_process(
                input_path, 
                model_path, 
                output_dir, 
                args.threshold, 
                args.recursive, 
                args.workers, 
                args.visualize, 
                args.verbose
            )
            
            # 保存汇总报告
            save_summary_report(results, output_dir, args.format)
        else:
            logger.error("输入路径是目录，但未指定批量处理模式。请使用 --batch 参数开启批量处理。")
            sys.exit(1)
    else:
        logger.error(f"无法识别的输入路径类型: {input_path}")
        sys.exit(1)
    
    logger.info("检测任务完成")

if __name__ == "__main__":
    main() 