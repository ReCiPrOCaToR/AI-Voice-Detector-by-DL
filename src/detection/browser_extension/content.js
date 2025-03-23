/**
 * AI配音检测器 - 内容脚本
 * 负责在网页中捕获音频并发送到后端进行AI配音检测
 */

let isDetecting = false;
let detectInterval = null;
let currentVideoElement = null;
let audioContext = null;
let mediaStreamSource = null;
let analyser = null;
let audioBuffer = [];
let detectionResults = [];
let overlayElement = null;
let apiEndpoint = 'http://localhost:5000/api/detect';  // 默认API端点

// 初始化
function initialize() {
  console.log('AI配音检测器初始化中...');
  
  // 从存储获取设置
  chrome.storage.sync.get(['apiEndpoint', 'detectInterval', 'isEnabled'], (result) => {
    if (result.apiEndpoint) {
      apiEndpoint = result.apiEndpoint;
    }
    if (result.detectInterval) {
      detectInterval = result.detectInterval;
    } else {
      detectInterval = 5000; // 默认5秒
    }
    
    // 检查是否已启用
    if (result.isEnabled !== false) {
      // 创建UI覆盖层
      createOverlay();
      
      // 尝试找到视频元素并开始检测
      findVideoAndStartDetection();
    }
  });
  
  // 监听来自扩展的消息
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'toggleDetection') {
      if (message.enabled) {
        findVideoAndStartDetection();
      } else {
        stopDetection();
      }
      sendResponse({success: true});
    } else if (message.action === 'getStatus') {
      sendResponse({
        isDetecting: isDetecting,
        hasVideo: !!currentVideoElement,
        results: detectionResults
      });
    } else if (message.action === 'updateSettings') {
      if (message.apiEndpoint) {
        apiEndpoint = message.apiEndpoint;
      }
      if (message.detectInterval) {
        detectInterval = message.detectInterval;
      }
      sendResponse({success: true});
    }
    return true;  // 保持消息通道开放
  });
}

// 创建UI覆盖层
function createOverlay() {
  // 如果已存在，则不重复创建
  if (overlayElement) return;
  
  // 创建覆盖层
  overlayElement = document.createElement('div');
  overlayElement.id = 'ai-voice-detector-overlay';
  overlayElement.innerHTML = `
    <div class="ai-voice-detector-container">
      <div class="ai-voice-detector-header">
        <span class="ai-voice-detector-title">AI配音检测</span>
        <span class="ai-voice-detector-close">×</span>
      </div>
      <div class="ai-voice-detector-body">
        <div class="ai-voice-detector-status">未开始检测</div>
        <div class="ai-voice-detector-result"></div>
        <div class="ai-voice-detector-progress">
          <div class="ai-voice-detector-progress-bar"></div>
        </div>
      </div>
    </div>
  `;
  
  // 添加样式
  const style = document.createElement('style');
  style.textContent = `
    #ai-voice-detector-overlay {
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 9999;
      width: 300px;
      background: rgba(0, 0, 0, 0.8);
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
      color: white;
      font-family: Arial, sans-serif;
      transition: opacity 0.3s ease;
      opacity: 0.8;
    }
    #ai-voice-detector-overlay:hover {
      opacity: 1;
    }
    .ai-voice-detector-container {
      padding: 10px;
    }
    .ai-voice-detector-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .ai-voice-detector-title {
      font-weight: bold;
    }
    .ai-voice-detector-close {
      cursor: pointer;
      font-size: 20px;
    }
    .ai-voice-detector-status {
      margin-bottom: 5px;
      font-size: 14px;
    }
    .ai-voice-detector-result {
      margin-bottom: 10px;
      font-size: 16px;
      font-weight: bold;
    }
    .ai-voice-detector-progress {
      height: 5px;
      background: #444;
      border-radius: 3px;
      overflow: hidden;
    }
    .ai-voice-detector-progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #3498db, #2ecc71);
      transition: width 0.3s ease;
    }
    .ai-voice-detector-result.ai {
      color: #ff6b6b;
    }
    .ai-voice-detector-result.human {
      color: #2ecc71;
    }
  `;
  
  document.head.appendChild(style);
  document.body.appendChild(overlayElement);
  
  // 添加关闭按钮事件
  const closeButton = overlayElement.querySelector('.ai-voice-detector-close');
  if (closeButton) {
    closeButton.addEventListener('click', () => {
      stopDetection();
      overlayElement.style.display = 'none';
      
      // 保存设置
      chrome.storage.sync.set({isEnabled: false});
    });
  }
}

// 更新UI
function updateUI(status, result = null) {
  if (!overlayElement) return;
  
  const statusElem = overlayElement.querySelector('.ai-voice-detector-status');
  const resultElem = overlayElement.querySelector('.ai-voice-detector-result');
  const progressBar = overlayElement.querySelector('.ai-voice-detector-progress-bar');
  
  if (statusElem) {
    statusElem.textContent = status;
  }
  
  if (result && resultElem) {
    if (result.is_ai) {
      resultElem.textContent = `检测结果: AI配音 (${(result.ai_probability * 100).toFixed(1)}%)`;
      resultElem.className = 'ai-voice-detector-result ai';
    } else {
      resultElem.textContent = `检测结果: 人声配音 (${(result.human_probability * 100).toFixed(1)}%)`;
      resultElem.className = 'ai-voice-detector-result human';
    }
  }
  
  if (progressBar) {
    if (status === '检测中...') {
      // 进度动画
      let progress = 0;
      const interval = setInterval(() => {
        progress = (progress + 1) % 100;
        progressBar.style.width = `${progress}%`;
      }, 50);
      
      progressBar.dataset.interval = interval;
    } else {
      // 清除动画
      if (progressBar.dataset.interval) {
        clearInterval(parseInt(progressBar.dataset.interval));
      }
      progressBar.style.width = '100%';
    }
  }
}

// 查找视频元素并开始检测
function findVideoAndStartDetection() {
  // 尝试找到视频元素
  const videoElements = document.querySelectorAll('video');
  
  if (videoElements.length > 0) {
    // 选择第一个视频元素
    currentVideoElement = videoElements[0];
    
    // 开始检测
    startDetection();
  } else {
    console.log('未找到视频元素');
    updateUI('未找到视频元素');
    
    // 继续尝试查找视频元素
    setTimeout(findVideoAndStartDetection, 2000);
  }
}

// 开始检测
function startDetection() {
  if (isDetecting || !currentVideoElement) return;
  
  console.log('开始AI配音检测');
  isDetecting = true;
  
  // 显示覆盖层
  if (overlayElement) {
    overlayElement.style.display = 'block';
  }
  
  updateUI('检测中...');
  
  // 初始化音频上下文
  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // 创建媒体元素源
    mediaStreamSource = audioContext.createMediaElementSource(currentVideoElement);
    
    // 创建分析器
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    
    // 连接节点
    mediaStreamSource.connect(analyser);
    analyser.connect(audioContext.destination);
    
    // 定期采集音频并发送进行检测
    detectInterval = setInterval(captureAudioAndDetect, detectInterval);
    
    // 添加视频事件监听
    currentVideoElement.addEventListener('pause', onVideoPause);
    currentVideoElement.addEventListener('play', onVideoPlay);
    
  } catch (error) {
    console.error('初始化音频上下文失败:', error);
    updateUI('无法访问视频音频');
    isDetecting = false;
  }
}

// 停止检测
function stopDetection() {
  if (!isDetecting) return;
  
  console.log('停止AI配音检测');
  isDetecting = false;
  
  // 清除检测定时器
  if (detectInterval) {
    clearInterval(detectInterval);
    detectInterval = null;
  }
  
  // 断开音频连接
  if (mediaStreamSource && analyser && audioContext) {
    try {
      mediaStreamSource.disconnect(analyser);
      analyser.disconnect(audioContext.destination);
    } catch (error) {
      console.error('断开音频连接失败:', error);
    }
  }
  
  // 移除视频事件监听
  if (currentVideoElement) {
    currentVideoElement.removeEventListener('pause', onVideoPause);
    currentVideoElement.removeEventListener('play', onVideoPlay);
  }
  
  // 关闭音频上下文
  if (audioContext && audioContext.state !== 'closed') {
    audioContext.close().catch(error => {
      console.error('关闭音频上下文失败:', error);
    });
  }
  
  audioContext = null;
  mediaStreamSource = null;
  analyser = null;
  
  updateUI('检测已停止');
}

// 视频暂停事件处理
function onVideoPause() {
  if (isDetecting) {
    updateUI('视频已暂停');
  }
}

// 视频播放事件处理
function onVideoPlay() {
  if (isDetecting) {
    updateUI('检测中...');
  }
}

// 采集音频并发送检测
function captureAudioAndDetect() {
  if (!isDetecting || !currentVideoElement || currentVideoElement.paused) return;
  
  try {
    // 采集音频数据
    const bufferLength = analyser.frequencyBinCount;
    const audioData = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(audioData);
    
    // 转换为16位PCM
    const pcmData = new Int16Array(bufferLength);
    for (let i = 0; i < bufferLength; i++) {
      pcmData[i] = (audioData[i] - 128) * 256;
    }
    
    // 将数据转换为Base64进行传输
    const blob = new Blob([pcmData], { type: 'audio/pcm' });
    const reader = new FileReader();
    
    reader.onloadend = function() {
      const base64data = reader.result.split(',')[1];
      
      // 发送到后端API
      sendAudioForDetection(base64data);
    };
    
    reader.readAsDataURL(blob);
    
  } catch (error) {
    console.error('采集音频失败:', error);
    updateUI('音频采集失败');
  }
}

// 发送音频到后端进行检测
function sendAudioForDetection(audioData) {
  if (!audioData) return;
  
  updateUI('正在分析...');
  
  fetch(apiEndpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      audio_data: audioData,
      format: 'pcm',
      current_time: currentVideoElement.currentTime
    })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP错误! 状态: ${response.status}`);
    }
    return response.json();
  })
  .then(result => {
    console.log('检测结果:', result);
    
    // 保存结果
    detectionResults.push({
      timestamp: currentVideoElement.currentTime,
      result: result
    });
    
    // 更新UI
    updateUI('检测完成', result);
    
    // 通知后台脚本
    chrome.runtime.sendMessage({
      action: 'detectionResult',
      result: result,
      timestamp: currentVideoElement.currentTime,
      url: window.location.href
    });
  })
  .catch(error => {
    console.error('检测请求失败:', error);
    updateUI('检测请求失败');
  });
}

// 启动脚本
initialize(); 