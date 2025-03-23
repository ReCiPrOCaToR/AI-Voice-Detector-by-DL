#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从提供的B站视频链接列表采集数据
支持从文件或命令行参数读取链接
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import concurrent.futures

import requests
import numpy as np
from tqdm import tqdm
from bilibili_api import video, Credential
import yt_dlp
import ffmpeg

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_collection.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从B站视频链接采集数据")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--links", "-l", type=str, nargs="+",
                      help="B站视频链接列表，多个链接用空格分隔")
    group.add_argument("--file", "-f", type=str,
                      help="包含B站视频链接的文本文件，每行一个链接")
    
    parser.add_argument("--output_dir", "-o", type=str, default="data/raw",
                      help="输出目录，默认为data/raw")
    
    parser.add_argument("--category", "-c", type=str, choices=["ai", "human", "auto"], default="auto",
                      help="视频分类，默认为自动判断")
    
    parser.add_argument("--max_duration", "-d", type=int, default=300,
                      help="最大视频时长（秒），默认300秒")
    
    parser.add_argument("--workers", "-w", type=int, default=4,
                      help="并行下载线程数，默认4")
    
    parser.add_argument("--sleep", "-s", type=float, default=1.0,
                      help="请求间隔（秒），防止被封IP，默认1秒")
    
    parser.add_argument("--cookie", type=str, default=None,
                      help="B站cookie文件路径，提供更稳定的访问")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="显示详细日志")
    
    return parser.parse_args()

def extract_bvid(url: str) -> str:
    """
    从B站URL中提取BVID
    
    Args:
        url: B站视频URL
        
    Returns:
        视频的BVID
    """
    # 处理常见的B站URL格式
    if "bilibili.com/video/" in url:
        parts = url.split("bilibili.com/video/")
        if len(parts) > 1:
            bv_part = parts[1].split("/")[0].split("?")[0]
            return bv_part
    
    # 如果已经是BVID格式
    if url.startswith("BV"):
        return url
    
    raise ValueError(f"无法从URL中提取BVID: {url}")

def get_video_info(bvid: str, credential: Optional[Credential] = None) -> Dict[str, Any]:
    """
    获取B站视频信息
    
    Args:
        bvid: 视频的BVID
        credential: B站凭证，可选
        
    Returns:
        视频信息字典
    """
    v = video.Video(bvid=bvid, credential=credential)
    info = v.get_info()
    return info

def download_audio(
    bvid: str, 
    output_path: str, 
    max_duration: int = 300,
    verbose: bool = False
) -> bool:
    """
    下载B站视频的音频
    
    Args:
        bvid: 视频的BVID
        output_path: 输出文件路径
        max_duration: 最大视频时长（秒）
        verbose: 是否显示详细日志
    
    Returns:
        下载是否成功
    """
    try:
        url = f"https://www.bilibili.com/video/{bvid}"
        
        # 配置yt-dlp选项
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'quiet': not verbose,
            'no_warnings': not verbose,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        
        # 下载音频
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # 检查下载的文件是否存在
        wav_path = f"{output_path}.wav"
        if not os.path.exists(wav_path):
            logger.error(f"下载失败，文件不存在: {wav_path}")
            return False
        
        # 检查音频时长
        probe = ffmpeg.probe(wav_path)
        duration = float(probe['format']['duration'])
        
        if duration > max_duration:
            logger.warning(f"视频时长 {duration:.1f}s 超过限制 {max_duration}s，将进行裁剪")
            
            # 创建临时文件
            temp_path = f"{output_path}_temp.wav"
            
            # 裁剪音频
            (
                ffmpeg
                .input(wav_path)
                .output(temp_path, t=max_duration)
                .run(quiet=not verbose, overwrite_output=True)
            )
            
            # 替换原文件
            os.replace(temp_path, wav_path)
            
            logger.info(f"已裁剪音频到 {max_duration}s")
        
        return True
    
    except Exception as e:
        logger.error(f"下载视频 {bvid} 音频时出错: {e}")
        return False

def process_video_link(
    url: str,
    output_dir: str,
    category: str = "auto",
    max_duration: int = 300,
    credential: Optional[Credential] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    处理单个视频链接
    
    Args:
        url: 视频URL
        output_dir: 输出目录
        category: 视频分类（ai, human, auto）
        max_duration: 最大视频时长（秒）
        credential: B站凭证，可选
        verbose: 是否显示详细日志
    
    Returns:
        处理结果
    """
    try:
        # 提取BVID
        bvid = extract_bvid(url)
        
        # 获取视频信息
        video_info = get_video_info(bvid, credential)
        
        # 确定分类
        if category == "auto":
            # 自动判断是AI还是人声
            # 这里可以添加更复杂的判断逻辑
            # 暂时随机分配，实际使用时应该根据提供的链接确定
            category = "ai"  # 假设用户提供的都是AI配音视频
        
        # 创建分类目录
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # 构建输出文件路径
        output_filename = f"{bvid}_{int(time.time())}"
        output_path = os.path.join(category_dir, output_filename)
        
        # 下载音频
        success = download_audio(bvid, output_path, max_duration, verbose)
        
        if not success:
            return {
                "url": url,
                "bvid": bvid,
                "success": False,
                "error": "下载失败"
            }
        
        # 保存视频元数据
        metadata = {
            "url": url,
            "bvid": bvid,
            "title": video_info["title"],
            "author": video_info["owner"]["name"],
            "category": category,
            "duration": video_info["duration"],
            "description": video_info["desc"],
            "collection_time": time.time(),
            "tags": video_info.get("tags", []),
        }
        
        # 写入元数据JSON文件
        with open(f"{output_path}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已下载并处理视频: {bvid}, 分类: {category}")
        
        return {
            "url": url,
            "bvid": bvid,
            "title": video_info["title"],
            "category": category,
            "success": True,
            "output_path": f"{output_path}.wav"
        }
    
    except Exception as e:
        logger.error(f"处理视频链接 {url} 时出错: {e}")
        return {
            "url": url,
            "success": False,
            "error": str(e)
        }

def collect_data_from_links(
    links: List[str],
    output_dir: str,
    category: str = "auto",
    max_duration: int = 300,
    workers: int = 4,
    sleep_interval: float = 1.0,
    cookie_file: Optional[str] = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    从视频链接列表收集数据
    
    Args:
        links: 视频链接列表
        output_dir: 输出目录
        category: 视频分类（ai, human, auto）
        max_duration: 最大视频时长（秒）
        workers: 并行工作线程数
        sleep_interval: 请求间隔（秒）
        cookie_file: B站cookie文件路径，可选
        verbose: 是否显示详细日志
    
    Returns:
        处理结果列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载B站凭证（如果有）
    credential = None
    if cookie_file and os.path.exists(cookie_file):
        try:
            with open(cookie_file, "r", encoding="utf-8") as f:
                cookie_data = json.load(f)
                
            credential = Credential(
                sessdata=cookie_data.get("SESSDATA", ""),
                bili_jct=cookie_data.get("bili_jct", ""),
                buvid3=cookie_data.get("buvid3", "")
            )
            logger.info("已加载B站凭证")
        except Exception as e:
            logger.warning(f"加载B站凭证失败: {e}")
    
    # 保存视频链接列表
    links_file = os.path.join(output_dir, "collected_links.txt")
    with open(links_file, "w", encoding="utf-8") as f:
        for link in links:
            f.write(f"{link}\n")
    
    # 处理结果
    results = []
    
    # 进度条
    progress_bar = tqdm(total=len(links), desc="下载进度")
    
    # 顺序处理，避免同时请求过多导致被封
    for url in links:
        try:
            result = process_video_link(
                url, 
                output_dir, 
                category, 
                max_duration, 
                credential, 
                verbose
            )
            results.append(result)
            
            # 随机延时，防止请求过于频繁
            time.sleep(sleep_interval + random.uniform(0, 1))
            
        except Exception as e:
            logger.error(f"处理链接 {url} 时出错: {e}")
            results.append({
                "url": url,
                "success": False,
                "error": str(e)
            })
        
        # 更新进度条
        progress_bar.update(1)
    
    # 关闭进度条
    progress_bar.close()
    
    # 保存结果摘要
    summary = {
        "total": len(links),
        "success": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", False)),
        "collection_time": time.time()
    }
    
    with open(os.path.join(output_dir, "collection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据采集完成: 共 {len(links)} 个链接，成功 {summary['success']} 个，失败 {summary['failed']} 个")
    
    return results

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
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
        workers=args.workers,
        sleep_interval=args.sleep,
        cookie_file=args.cookie,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 