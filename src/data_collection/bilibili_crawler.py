#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B站视频爬虫模块
用于收集AI配音和真人配音的训练数据
"""

import os
import json
import time
import logging
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Union, Tuple

import requests
from bs4 import BeautifulSoup
import yt_dlp
from tqdm import tqdm

logger = logging.getLogger(__name__)

# B站API地址
BILIBILI_SEARCH_API = "https://api.bilibili.com/x/web-interface/search/all/v2"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
]

def get_random_user_agent() -> str:
    """获取随机User-Agent"""
    return random.choice(USER_AGENTS)

def search_bilibili_videos(keyword: str, page: int = 1, page_size: int = 20) -> Dict:
    """
    搜索B站视频
    
    Args:
        keyword: 搜索关键词
        page: 页码
        page_size: 每页结果数量
        
    Returns:
        搜索结果字典
    """
    headers = {
        "User-Agent": get_random_user_agent(),
        "Referer": "https://www.bilibili.com/",
    }
    
    params = {
        "keyword": keyword,
        "page": page,
        "page_size": page_size,
        "search_type": "video",  # 只搜索视频
    }
    
    try:
        response = requests.get(BILIBILI_SEARCH_API, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"搜索B站视频失败: {e}")
        return {"code": -1, "data": None, "message": str(e)}

def extract_video_info(search_results: Dict) -> List[Dict]:
    """
    从搜索结果中提取视频信息
    
    Args:
        search_results: 搜索结果字典
        
    Returns:
        视频信息列表
    """
    video_list = []
    
    try:
        if search_results.get("code") != 0 or not search_results.get("data"):
            logger.warning(f"搜索结果无效: {search_results.get('message', '未知错误')}")
            return video_list
        
        # 获取视频结果
        for item in search_results["data"]["result"]:
            if item["result_type"] == "video":
                for video in item["data"]:
                    video_info = {
                        "bvid": video.get("bvid", ""),
                        "title": video.get("title", "").strip(),
                        "author": video.get("author", ""),
                        "duration": video.get("duration", ""),
                        "url": f"https://www.bilibili.com/video/{video.get('bvid', '')}",
                    }
                    video_list.append(video_info)
    except Exception as e:
        logger.error(f"提取视频信息失败: {e}")
    
    return video_list

def download_video(video_info: Dict, output_dir: str, label: str) -> bool:
    """
    下载B站视频并提取音频
    
    Args:
        video_info: 视频信息字典
        output_dir: 输出目录
        label: 标签 (ai 或 human)
        
    Returns:
        下载是否成功
    """
    # 创建输出目录
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # 清理文件名，避免特殊字符导致问题
    safe_title = "".join([c if c.isalnum() or c in " _-" else "_" for c in video_info["title"]])
    safe_title = safe_title[:50]  # 限制长度
    
    # 输出文件名
    output_filename = f"{safe_title}_{video_info['bvid']}"
    output_path = os.path.join(label_dir, output_filename)
    
    # 保存视频元数据
    with open(f"{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(video_info, f, ensure_ascii=False, indent=2)
    
    # 设置yt-dlp选项
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': f"{output_path}.%(ext)s",
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_info["url"]])
        logger.info(f"成功下载视频: {video_info['title']} [{label}]")
        return True
    except Exception as e:
        logger.error(f"下载视频失败 {video_info['bvid']}: {e}")
        return False

def classify_video(video_info: Dict) -> str:
    """
    根据视频信息初步判断是AI配音还是人声配音
    这里使用简单的关键词匹配，实际项目中可能需要更复杂的方法
    
    Args:
        video_info: 视频信息字典
        
    Returns:
        'ai' 或 'human' 或 'unknown'
    """
    title = video_info["title"].lower()
    # 判断是否包含AI配音关键词
    ai_keywords = ["yunxi", "云溪", "云析", "ai配音", "人工智能配音", "机器配音", 
                  "智能配音", "文字转语音", "text to speech", "tts"]
    
    for keyword in ai_keywords:
        if keyword.lower() in title:
            return "ai"
    
    # 判断是否明确标注为人声配音
    human_keywords = ["真人配音", "专业配音", "配音演员"]
    for keyword in human_keywords:
        if keyword.lower() in title:
            return "human"
    
    # 无法确定的情况下，标记为未知
    return "unknown"

def collect_videos(keywords: List[str], num_videos: int, output_dir: str) -> None:
    """
    收集视频作为训练数据
    
    Args:
        keywords: 搜索关键词列表
        num_videos: 要收集的视频数量
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集的视频列表
    ai_videos = []
    human_videos = []
    unknown_videos = []
    
    # 每个关键词收集的视频数量
    videos_per_keyword = max(num_videos // len(keywords), 10)
    
    # 遍历关键词
    for keyword in keywords:
        logger.info(f"搜索关键词: {keyword}")
        page = 1
        collected = 0
        
        # 分页搜索，直到收集足够数量的视频
        while collected < videos_per_keyword and page <= 50:  # 最多搜索50页
            search_results = search_bilibili_videos(keyword, page=page)
            videos = extract_video_info(search_results)
            
            if not videos:
                logger.warning(f"没有找到更多视频，关键词: {keyword}, 页码: {page}")
                break
            
            # 对每个视频进行分类
            for video in videos:
                label = classify_video(video)
                if label == "ai":
                    ai_videos.append((video, label))
                elif label == "human":
                    human_videos.append((video, label))
                else:
                    unknown_videos.append((video, label))
                
                collected += 1
                if collected >= videos_per_keyword:
                    break
            
            # 翻页
            page += 1
            time.sleep(1)  # 避免频繁请求
    
    # 确保AI和人声样本数量平衡
    min_count = min(len(ai_videos), len(human_videos))
    selected_ai = ai_videos[:min_count] if len(ai_videos) > min_count else ai_videos
    selected_human = human_videos[:min_count] if len(human_videos) > min_count else human_videos
    
    # 如果样本不足，可以从未知类别中补充
    if min_count < num_videos // 2:
        needed = num_videos // 2 - min_count
        selected_unknown = unknown_videos[:needed]
        logger.warning(f"AI或人声样本不足，从未知类别中补充 {len(selected_unknown)} 个样本")
    else:
        selected_unknown = []
    
    # 准备下载队列
    download_queue = selected_ai + selected_human + selected_unknown
    logger.info(f"准备下载 {len(download_queue)} 个视频 (AI: {len(selected_ai)}, 人声: {len(selected_human)}, 未知: {len(selected_unknown)})")
    
    # 使用线程池并行下载
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for video_info, label in download_queue:
            futures.append(
                executor.submit(download_video, video_info, output_dir, label)
            )
        
        # 显示进度
        for _ in tqdm(futures, total=len(futures), desc="下载视频"):
            pass
    
    logger.info(f"视频采集完成，保存在 {output_dir}")
    
    # 生成数据集信息文件
    dataset_info = {
        "total_videos": len(download_queue),
        "ai_videos": len(selected_ai),
        "human_videos": len(selected_human),
        "unknown_videos": len(selected_unknown),
        "keywords": keywords,
        "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 简单测试
    output_dir = "../../data/raw"
    keywords = ["yunxi配音", "AI配音", "真人配音"]
    collect_videos(keywords, 20, output_dir) 