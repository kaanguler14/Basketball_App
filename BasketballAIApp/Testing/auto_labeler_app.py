"""
🏀 Otomatik Etiketleme Uygulaması
- Dataset klasörü seç
- Prompt ayarlama
- Grounding DINO ile otomatik etiketle
- Ayrı sayfada editör
"""

from flask import Flask, render_template_string, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import torch
from PIL import Image
import os
from pathlib import Path
import base64
import threading
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
config = {
    "images_dir": "",
    "labels_dir": "",
    "confidence": 0.3,
    "classes": [
        {"name": "basketball", "prompt": "orange basketball ball", "color": "#FFA500"},
        {"name": "rim", "prompt": "basketball hoop ring", "color": "#00FF00"},
        {"name": "player", "prompt": "person playing basketball", "color": "#0080FF"}
    ]
}

model = None
processor = None
device = None
is_labeling = False

# ==================== MAIN PAGE ====================
MAIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>🏀 Otomatik Etiketleme</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 30px; }
        
        .header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 2px solid #e94560;
            margin-bottom: 40px;
        }
        .header h1 { font-size: 42px; color: #e94560; margin-bottom: 10px; }
        .header p { color: #888; font-size: 16px; }
        
        .card {
            background: #16213e;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 25px;
            border: 2px solid #0f3460;
        }
        .card-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px solid #0f3460;
        }
        .card-icon {
            width: 50px;
            height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        .card-title { font-size: 20px; font-weight: bold; }
        
        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #888;
            font-size: 14px;
            font-weight: 500;
        }
        .form-group input, .form-group textarea {
            width: 100%;
            padding: 14px 18px;
            border: 2px solid #0f3460;
            border-radius: 10px;
            background: #1a1a2e;
            color: #fff;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        .form-group input:focus, .form-group textarea:focus {
            outline: none;
            border-color: #e94560;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .class-config {
            display: flex;
            gap: 15px;
            align-items: center;
            padding: 15px;
            background: #0f3460;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .class-color {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        .class-inputs { flex: 1; display: flex; gap: 15px; }
        .class-inputs input {
            flex: 1;
            padding: 12px;
            border: 2px solid #1a1a2e;
            border-radius: 8px;
            background: #1a1a2e;
            color: #fff;
            font-size: 13px;
        }
        .class-inputs input:focus { border-color: #e94560; outline: none; }
        
        .btn {
            padding: 14px 30px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 15px;
            font-weight: bold;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }
        .btn-primary { background: #e94560; color: #fff; }
        .btn-primary:hover { background: #d63050; transform: translateY(-2px); }
        .btn-success { background: #00b894; color: #fff; }
        .btn-success:hover { background: #00a383; transform: translateY(-2px); }
        .btn-secondary { background: #0f3460; color: #fff; }
        .btn-secondary:hover { background: #1a4a7a; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-lg { padding: 18px 40px; font-size: 17px; }
        
        .progress-section {
            display: none;
            margin-top: 30px;
        }
        .progress-section.visible { display: block; }
        .progress-bar {
            height: 40px;
            background: #0f3460;
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #e94560, #ff6b6b);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
        }
        .progress-text { color: #888; font-size: 14px; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-top: 25px;
        }
        .stat-card {
            background: #0f3460;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
        }
        .stat-value { font-size: 36px; font-weight: bold; color: #e94560; }
        .stat-label { font-size: 13px; color: #888; margin-top: 8px; }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            margin-top: 30px;
            justify-content: center;
        }
        
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 10px;
            font-weight: bold;
            animation: slideIn 0.3s;
            z-index: 1000;
        }
        .toast.success { background: #00b894; }
        .toast.error { background: #e94560; }
        .toast.info { background: #0984e3; }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .loading-overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
        }
        .loading-overlay.hidden { display: none; }
        .spinner {
            width: 70px; height: 70px;
            border: 5px solid #0f3460;
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-content { text-align: center; }
        .loading-content p { font-size: 16px; color: #888; }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .slider-container input[type="range"] {
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: #0f3460;
            border-radius: 4px;
        }
        .slider-container input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #e94560;
            border-radius: 50%;
            cursor: pointer;
        }
        .slider-value {
            background: #0f3460;
            padding: 8px 15px;
            border-radius: 8px;
            font-weight: bold;
            min-width: 60px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="loading-overlay hidden" id="loadingOverlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <p id="loadingText">Model yükleniyor...</p>
        </div>
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🏀 Otomatik Etiketleme</h1>
            <p>Grounding DINO ile akıllı nesne tespiti ve etiketleme</p>
        </div>
        
        <!-- Folder Settings -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: #e94560;">📁</div>
                <div class="card-title">Klasör Ayarları</div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Resim Klasörü</label>
                    <input type="text" id="imagesDir" placeholder="D:\\path\\to\\images">
                </div>
                <div class="form-group">
                    <label>Etiket Klasörü (otomatik oluşturulur)</label>
                    <input type="text" id="labelsDir" placeholder="D:\\path\\to\\labels">
                </div>
            </div>
            <div class="form-group">
                <label>Confidence Eşiği</label>
                <div class="slider-container">
                    <input type="range" id="confidence" min="0.1" max="0.9" step="0.05" value="0.3" 
                           oninput="document.getElementById('confValue').textContent = this.value">
                    <span class="slider-value" id="confValue">0.3</span>
                </div>
            </div>
        </div>
        
        <!-- Class & Prompt Settings -->
        <div class="card">
            <div class="card-header">
                <div class="card-icon" style="background: #00b894;">🎯</div>
                <div class="card-title">Sınıf ve Prompt Ayarları</div>
            </div>
            <div id="classConfigs">
                <div class="class-config">
                    <input type="color" class="class-color" value="#FFA500" id="color0">
                    <div class="class-inputs">
                        <input type="text" id="name0" value="basketball" placeholder="Sınıf adı">
                        <input type="text" id="prompt0" value="orange basketball ball" placeholder="Tespit promptu">
                    </div>
                </div>
                <div class="class-config">
                    <input type="color" class="class-color" value="#00FF00" id="color1">
                    <div class="class-inputs">
                        <input type="text" id="name1" value="rim" placeholder="Sınıf adı">
                        <input type="text" id="prompt1" value="basketball hoop ring" placeholder="Tespit promptu">
                    </div>
                </div>
                <div class="class-config">
                    <input type="color" class="class-color" value="#0080FF" id="color2">
                    <div class="class-inputs">
                        <input type="text" id="name2" value="player" placeholder="Sınıf adı">
                        <input type="text" id="prompt2" value="person playing basketball" placeholder="Tespit promptu">
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Action Buttons -->
        <div class="action-buttons">
            <button class="btn btn-primary btn-lg" id="startBtn" onclick="startLabeling()">
                🚀 Etiketlemeyi Başlat
            </button>
            <button class="btn btn-success btn-lg" onclick="openEditor()">
                ✏️ Editörü Aç
            </button>
            <button class="btn btn-secondary btn-lg" onclick="window.location.href='/editor'">
                🔗 Editöre Git
            </button>
        </div>
        
        <!-- Progress Section -->
        <div class="progress-section" id="progressSection">
            <div class="card">
                <div class="card-header">
                    <div class="card-icon" style="background: #0984e3;">⏳</div>
                    <div class="card-title">İlerleme</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: 0%">0%</div>
                </div>
                <p class="progress-text" id="progressText">Hazırlanıyor...</p>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="statTotal">0</div>
                        <div class="stat-label">Toplam Resim</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat0">0</div>
                        <div class="stat-label" id="statLabel0">🏀 Basketball</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat1">0</div>
                        <div class="stat-label" id="statLabel1">⭕ Rim</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="stat2">0</div>
                        <div class="stat-label" id="statLabel2">🧑 Player</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        socket.on('progress', (data) => {
            hideLoading();  // İlk progress geldiğinde loading'i kapat
            document.getElementById('progressFill').style.width = data.percent + '%';
            document.getElementById('progressFill').textContent = data.percent + '%';
            document.getElementById('progressText').textContent = data.message;
        });
        
        socket.on('stats', (data) => {
            document.getElementById('statTotal').textContent = data.total;
            document.getElementById('stat0').textContent = data.class0 || 0;
            document.getElementById('stat1').textContent = data.class1 || 0;
            document.getElementById('stat2').textContent = data.class2 || 0;
        });
        
        socket.on('complete', () => {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('startBtn').innerHTML = '🔄 Tekrar Etiketle';
            document.getElementById('editorBtn').style.display = 'inline-flex';
            hideLoading();
            showToast('Etiketleme tamamlandı!', 'success');
        });
        
        socket.on('error', (data) => {
            showToast(data.message, 'error');
            document.getElementById('startBtn').disabled = false;
            hideLoading();
        });
        
        async function startLabeling() {
            const imagesDir = document.getElementById('imagesDir').value.trim();
            let labelsDir = document.getElementById('labelsDir').value.trim();
            const confidence = parseFloat(document.getElementById('confidence').value);
            
            if (!imagesDir) {
                showToast('Resim klasörü giriniz!', 'error');
                return;
            }
            
            if (!labelsDir) {
                labelsDir = imagesDir + '_labels';
                document.getElementById('labelsDir').value = labelsDir;
            }
            
            const classes = [];
            for (let i = 0; i < 3; i++) {
                classes.push({
                    name: document.getElementById('name' + i).value,
                    prompt: document.getElementById('prompt' + i).value,
                    color: document.getElementById('color' + i).value
                });
                document.getElementById('statLabel' + i).textContent = document.getElementById('name' + i).value;
            }
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('progressSection').classList.add('visible');
            showLoading('Ayarlar kaydediliyor...');
            
            const response = await fetch('/api/start_labeling', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    images_dir: imagesDir,
                    labels_dir: labelsDir,
                    confidence: confidence,
                    classes: classes
                })
            });
            
            const data = await response.json();
            
            if (!data.success) {
                showToast(data.error, 'error');
                document.getElementById('startBtn').disabled = false;
                hideLoading();
            }
        }
        
        function openEditor() {
            window.open('/editor', '_blank');
        }
        
        function showToast(msg, type) {
            const t = document.createElement('div');
            t.className = 'toast ' + type;
            t.textContent = msg;
            document.body.appendChild(t);
            setTimeout(() => t.remove(), 3000);
        }
        
        function showLoading(text) {
            document.getElementById('loadingText').textContent = text;
            document.getElementById('loadingOverlay').classList.remove('hidden');
        }
        
        function hideLoading() {
            document.getElementById('loadingOverlay').classList.add('hidden');
        }
    </script>
</body>
</html>
"""

# ==================== EDITOR PAGE ====================
EDITOR_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>✏️ Etiket Editörü</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #0f0f1a;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }
        
        .layout {
            display: flex;
            height: 100vh;
        }
        
        .sidebar {
            width: 280px;
            background: #16213e;
            display: flex;
            flex-direction: column;
            border-right: 2px solid #0f3460;
        }
        .sidebar-header {
            padding: 20px;
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            font-size: 18px;
            font-weight: bold;
        }
        .search-box {
            padding: 15px;
            background: #0f3460;
        }
        .search-box input {
            width: 100%;
            padding: 12px 15px;
            border: none;
            border-radius: 8px;
            background: #1a1a2e;
            color: #fff;
            font-size: 14px;
        }
        .filter-tabs {
            display: flex;
            padding: 10px 15px;
            gap: 5px;
        }
        .filter-tab {
            flex: 1;
            padding: 8px;
            border: none;
            border-radius: 5px;
            background: #0f3460;
            color: #888;
            cursor: pointer;
            font-size: 12px;
        }
        .filter-tab.active { background: #e94560; color: #fff; }
        .file-list {
            flex: 1;
            overflow-y: auto;
        }
        .file-item {
            padding: 12px 20px;
            cursor: pointer;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            transition: background 0.2s;
        }
        .file-item:hover { background: #0f3460; }
        .file-item.active { background: #e94560; }
        .file-badge {
            background: #0f3460;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
        }
        .file-item.active .file-badge { background: #fff; color: #e94560; }
        
        .main-area {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .toolbar {
            padding: 12px 20px;
            background: #16213e;
            display: flex;
            gap: 15px;
            align-items: center;
            border-bottom: 2px solid #0f3460;
            flex-wrap: wrap;
        }
        .tool-group {
            display: flex;
            gap: 8px;
            padding-right: 15px;
            border-right: 1px solid #0f3460;
        }
        .tool-group:last-child { border-right: none; }
        .tool-btn {
            padding: 10px 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            background: #0f3460;
            color: #aaa;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .tool-btn:hover { background: #1a4a7a; color: #fff; }
        .tool-btn.active { background: #e94560; color: #fff; }
        .tool-btn.save { background: #00b894; color: #fff; }
        .tool-btn.save:hover { background: #00a383; }
        .class-dot {
            width: 14px;
            height: 14px;
            border-radius: 4px;
        }
        
        .canvas-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #0a0a12;
            position: relative;
        }
        #canvas { cursor: crosshair; }
        
        .status-bar {
            padding: 10px 20px;
            background: #16213e;
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #0f3460;
        }
        .status-bar span { display: flex; align-items: center; gap: 10px; }
        .kbd {
            background: #0f3460;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
        }
        
        .toast {
            position: fixed;
            bottom: 80px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 10px;
            font-weight: bold;
            animation: slideUp 0.3s;
            z-index: 1000;
        }
        .toast.success { background: #00b894; }
        .toast.error { background: #e94560; }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .empty-state {
            text-align: center;
            color: #444;
            padding: 40px;
        }
        .empty-state h2 { margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="layout">
        <div class="sidebar">
            <div class="sidebar-header">✏️ Etiket Editörü</div>
            <div class="search-box">
                <input type="text" placeholder="🔍 Dosya ara..." oninput="filterFiles(this.value)">
            </div>
            <div class="filter-tabs">
                <button class="filter-tab active" onclick="setFilter('all')">Tümü</button>
                <button class="filter-tab" onclick="setFilter('labeled')">Etiketli</button>
                <button class="filter-tab" onclick="setFilter('empty')">Boş</button>
            </div>
            <div class="file-list" id="fileList">
                <div class="empty-state">
                    <h2>📂</h2>
                    <p>Resim bulunamadı</p>
                </div>
            </div>
        </div>
        
        <div class="main-area">
            <div class="toolbar">
                <div class="tool-group">
                    <button class="tool-btn active" id="drawMode" onclick="setMode('draw')">✏️ Çiz</button>
                    <button class="tool-btn" id="selectMode" onclick="setMode('select')">🔲 Seç</button>
                </div>
                <div class="tool-group" id="classButtons"></div>
                <div class="tool-group">
                    <button class="tool-btn" onclick="deleteSelected()">🗑️ Etiketi Sil</button>
                    <button class="tool-btn" onclick="deleteAll()">🗑️ Tüm Etiketleri Sil</button>
                </div>
                <div class="tool-group">
                    <button class="tool-btn" onclick="deleteImage()" style="background: #c0392b;">🗑️ Resmi Sil</button>
                </div>
                <div class="tool-group">
                    <span id="saveStatus" style="color: #00b894; font-size: 12px;">✓ Kaydedildi</span>
                </div>
                <div class="tool-group">
                    <button class="tool-btn" onclick="navigate(-1)">⬅️</button>
                    <button class="tool-btn" onclick="navigate(1)">➡️</button>
                </div>
            </div>
            
            <div class="canvas-container" id="canvasContainer">
                <canvas id="canvas"></canvas>
            </div>
            
            <div class="status-bar">
                <span id="imageInfo">Bir resim seçin</span>
                <span id="labelCount">Etiket: 0</span>
                <span>
                    <span class="kbd">1-3</span> Sınıf
                    <span class="kbd">D</span> Çiz
                    <span class="kbd">V</span> Seç
                    <span class="kbd">Del</span> Sil
                    <span class="kbd">Ctrl+S</span> Kaydet
                    <span class="kbd">← →</span> Gezin
                </span>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let config = {};
        let images = [];
        let filteredImages = [];
        let currentIndex = -1;
        let currentImage = null;
        let labels = [];
        let selectedLabel = -1;
        let mode = 'draw';
        let currentClass = 0;
        let currentFilter = 'all';
        
        let isDrawing = false, isDragging = false, isResizing = false;
        let resizeHandle = null;
        let startX, startY, lastX, lastY, dragOffsetX, dragOffsetY;
        let autoSaveTimeout = null;
        let hasUnsavedChanges = false;
        
        async function init() {
            const response = await fetch('/api/config');
            config = await response.json();
            
            // Setup class buttons
            const container = document.getElementById('classButtons');
            config.classes.forEach((cls, idx) => {
                const btn = document.createElement('button');
                btn.className = 'tool-btn' + (idx === 0 ? ' active' : '');
                btn.id = 'class' + idx;
                btn.onclick = () => setClass(idx);
                btn.innerHTML = '<span class="class-dot" style="background: ' + cls.color + ';"></span>' + cls.name;
                container.appendChild(btn);
            });
            
            await loadImages();
        }
        
        async function loadImages() {
            const response = await fetch('/api/images');
            const data = await response.json();
            images = data.images;
            applyFilter();
            
            if (images.length > 0) selectImage(0);
        }
        
        function filterFiles(query) {
            query = query.toLowerCase();
            filteredImages = images.filter(img => img.name.toLowerCase().includes(query));
            if (currentFilter === 'labeled') filteredImages = filteredImages.filter(img => img.label_count > 0);
            else if (currentFilter === 'empty') filteredImages = filteredImages.filter(img => img.label_count === 0);
            renderFileList();
        }
        
        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-tab').forEach((btn, idx) => {
                btn.classList.toggle('active', 
                    (filter === 'all' && idx === 0) || 
                    (filter === 'labeled' && idx === 1) || 
                    (filter === 'empty' && idx === 2)
                );
            });
            applyFilter();
        }
        
        function applyFilter() {
            if (currentFilter === 'all') filteredImages = [...images];
            else if (currentFilter === 'labeled') filteredImages = images.filter(img => img.label_count > 0);
            else filteredImages = images.filter(img => img.label_count === 0);
            renderFileList();
        }
        
        function renderFileList() {
            const container = document.getElementById('fileList');
            if (filteredImages.length === 0) {
                container.innerHTML = '<div class="empty-state"><h2>📂</h2><p>Resim bulunamadı</p></div>';
                return;
            }
            container.innerHTML = filteredImages.map((img, idx) => {
                const realIdx = images.indexOf(img);
                return '<div class="file-item ' + (realIdx === currentIndex ? 'active' : '') + '" onclick="selectImage(' + realIdx + ')">' +
                    '<span>' + img.name + '</span>' +
                    '<span class="file-badge">' + img.label_count + '</span></div>';
            }).join('');
        }
        
        async function selectImage(index) {
            currentIndex = index;
            const img = images[index];
            
            const response = await fetch('/api/image/' + img.name);
            const data = await response.json();
            
            labels = data.labels.map(l => ({
                classId: parseInt(l.class_id),
                x: l.x_center, y: l.y_center,
                w: l.width, h: l.height
            }));
            
            selectedLabel = -1;
            currentImage = new Image();
            currentImage.onload = () => { resizeCanvas(); render(); };
            currentImage.src = 'data:image/jpeg;base64,' + data.image;
            
            document.getElementById('imageInfo').textContent = img.name + ' (' + (index + 1) + '/' + images.length + ')';
            updateLabelCount();
            renderFileList();
        }
        
        function resizeCanvas() {
            const container = document.getElementById('canvasContainer');
            const maxW = container.clientWidth - 40;
            const maxH = container.clientHeight - 40;
            const ratio = currentImage.width / currentImage.height;
            
            if (ratio > maxW / maxH) {
                canvas.width = maxW;
                canvas.height = maxW / ratio;
            } else {
                canvas.height = maxH;
                canvas.width = maxH * ratio;
            }
        }
        
        function render() {
            if (!currentImage) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            
            labels.forEach((label, idx) => {
                const color = config.classes[label.classId]?.color || '#fff';
                const name = config.classes[label.classId]?.name || 'unknown';
                
                const x = (label.x - label.w/2) * canvas.width;
                const y = (label.y - label.h/2) * canvas.height;
                const w = label.w * canvas.width;
                const h = label.h * canvas.height;
                
                ctx.strokeStyle = color;
                ctx.lineWidth = idx === selectedLabel ? 4 : 2;
                ctx.strokeRect(x, y, w, h);
                
                if (idx === selectedLabel) {
                    ctx.fillStyle = color + '33';
                    ctx.fillRect(x, y, w, h);
                    ctx.fillStyle = '#fff';
                    [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([hx,hy]) => {
                        ctx.fillRect(hx-5, hy-5, 10, 10);
                    });
                }
                
                ctx.fillStyle = color;
                ctx.fillRect(x, y-24, ctx.measureText(name).width + 16, 22);
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 13px sans-serif';
                ctx.fillText(name, x + 8, y - 7);
            });
            
            if (isDrawing) {
                const color = config.classes[currentClass]?.color || '#fff';
                const r = getDrawingRect();
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 6]);
                ctx.strokeRect(r.x, r.y, r.w, r.h);
                ctx.setLineDash([]);
            }
        }
        
        function getDrawingRect() {
            return {
                x: Math.min(startX, lastX), y: Math.min(startY, lastY),
                w: Math.abs(lastX - startX), h: Math.abs(lastY - startY)
            };
        }
        
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            startX = lastX = e.clientX - rect.left;
            startY = lastY = e.clientY - rect.top;
            
            if (mode === 'select') {
                if (selectedLabel >= 0) {
                    const label = labels[selectedLabel];
                    const lx = (label.x - label.w/2) * canvas.width;
                    const ly = (label.y - label.h/2) * canvas.height;
                    const lw = label.w * canvas.width;
                    const lh = label.h * canvas.height;
                    
                    const handles = [{n:'tl',x:lx,y:ly},{n:'tr',x:lx+lw,y:ly},{n:'bl',x:lx,y:ly+lh},{n:'br',x:lx+lw,y:ly+lh}];
                    for (let h of handles) {
                        if (Math.abs(startX - h.x) < 12 && Math.abs(startY - h.y) < 12) {
                            isResizing = true; resizeHandle = h.n; return;
                        }
                    }
                }
                
                for (let i = labels.length - 1; i >= 0; i--) {
                    const label = labels[i];
                    const lx = (label.x - label.w/2) * canvas.width;
                    const ly = (label.y - label.h/2) * canvas.height;
                    const lw = label.w * canvas.width;
                    const lh = label.h * canvas.height;
                    
                    if (startX >= lx && startX <= lx + lw && startY >= ly && startY <= ly + lh) {
                        selectedLabel = i;
                        isDragging = true;
                        dragOffsetX = startX - (label.x * canvas.width);
                        dragOffsetY = startY - (label.y * canvas.height);
                        render(); return;
                    }
                }
                selectedLabel = -1; render();
            } else {
                isDrawing = true;
            }
        });
        
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            lastX = e.clientX - rect.left;
            lastY = e.clientY - rect.top;
            
            if (isDrawing) render();
            else if (isDragging && selectedLabel >= 0) {
                labels[selectedLabel].x = (lastX - dragOffsetX) / canvas.width;
                labels[selectedLabel].y = (lastY - dragOffsetY) / canvas.height;
                render();
            } else if (isResizing && selectedLabel >= 0) {
                const label = labels[selectedLabel];
                let lx = (label.x - label.w/2) * canvas.width;
                let ly = (label.y - label.h/2) * canvas.height;
                let lx2 = lx + label.w * canvas.width;
                let ly2 = ly + label.h * canvas.height;
                
                if (resizeHandle.includes('l')) lx = lastX;
                if (resizeHandle.includes('r')) lx2 = lastX;
                if (resizeHandle.includes('t')) ly = lastY;
                if (resizeHandle.includes('b')) ly2 = lastY;
                
                label.w = Math.abs(lx2 - lx) / canvas.width;
                label.h = Math.abs(ly2 - ly) / canvas.height;
                label.x = (Math.min(lx, lx2) + Math.abs(lx2-lx)/2) / canvas.width;
                label.y = (Math.min(ly, ly2) + Math.abs(ly2-ly)/2) / canvas.height;
                render();
            }
        });
        
        canvas.addEventListener('mouseup', () => {
            let changed = false;
            if (isDrawing) {
                const r = getDrawingRect();
                if (r.w > 10 && r.h > 10) {
                    labels.push({
                        classId: currentClass,
                        x: (r.x + r.w/2) / canvas.width,
                        y: (r.y + r.h/2) / canvas.height,
                        w: r.w / canvas.width,
                        h: r.h / canvas.height
                    });
                    selectedLabel = labels.length - 1;
                    updateLabelCount();
                    changed = true;
                }
            }
            if (isDragging || isResizing) changed = true;
            
            isDrawing = isDragging = isResizing = false;
            resizeHandle = null;
            render();
            
            if (changed) scheduleAutoSave();
        });
        
        canvas.addEventListener('contextmenu', e => e.preventDefault());
        
        function setMode(m) {
            mode = m;
            document.getElementById('selectMode').classList.toggle('active', m === 'select');
            document.getElementById('drawMode').classList.toggle('active', m === 'draw');
            canvas.style.cursor = m === 'draw' ? 'crosshair' : 'default';
        }
        
        function setClass(c) {
            currentClass = c;
            config.classes.forEach((_, i) => {
                document.getElementById('class' + i).classList.toggle('active', i === c);
            });
            if (selectedLabel >= 0) {
                labels[selectedLabel].classId = c;
                render();
                scheduleAutoSave();
            }
        }
        
        function deleteSelected() {
            if (selectedLabel >= 0) {
                labels.splice(selectedLabel, 1);
                selectedLabel = -1;
                updateLabelCount();
                render();
                scheduleAutoSave();
            }
        }
        
        function deleteAll() {
            if (labels.length > 0 && confirm('Tüm etiketleri silmek istediğinizden emin misiniz?')) {
                labels = []; selectedLabel = -1;
                updateLabelCount(); render();
                scheduleAutoSave();
            }
        }
        
        function scheduleAutoSave() {
            hasUnsavedChanges = true;
            document.getElementById('saveStatus').textContent = '⏳ Kaydediliyor...';
            document.getElementById('saveStatus').style.color = '#f39c12';
            
            if (autoSaveTimeout) clearTimeout(autoSaveTimeout);
            autoSaveTimeout = setTimeout(() => saveLabels(true), 1000);  // 1 saniye sonra kaydet
        }
        
        async function saveLabels(isAutoSave = false) {
            if (currentIndex < 0) return;
            const img = images[currentIndex];
            
            const response = await fetch('/api/save/' + img.name, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({labels: labels.map(l => ({
                    class_id: l.classId.toString(),
                    x_center: l.x, y_center: l.y,
                    width: l.w, height: l.h
                }))})
            });
            
            if (response.ok) {
                img.label_count = labels.length;
                hasUnsavedChanges = false;
                document.getElementById('saveStatus').textContent = '✓ Kaydedildi';
                document.getElementById('saveStatus').style.color = '#00b894';
                renderFileList();
                if (!isAutoSave) showToast('Kaydedildi!', 'success');
            }
        }
        
        async function deleteImage() {
            if (currentIndex < 0) return;
            const img = images[currentIndex];
            
            if (!confirm('Bu resmi ve etiketlerini silmek istediğinizden emin misiniz?\\n\\n' + img.name)) {
                return;
            }
            
            const response = await fetch('/api/delete_image/' + img.name, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                const deletedIndex = currentIndex;
                images.splice(currentIndex, 1);
                applyFilter();
                
                if (images.length === 0) {
                    currentIndex = -1;
                    currentImage = null;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    document.getElementById('imageInfo').textContent = 'Resim kalmadı';
                } else {
                    const newIndex = Math.min(deletedIndex, images.length - 1);
                    selectImage(newIndex);
                }
                
                showToast('Resim silindi!', 'success');
            } else {
                showToast('Silme hatası!', 'error');
            }
        }
        
        function navigate(dir) {
            if (filteredImages.length === 0) return;
            const currentInFiltered = filteredImages.indexOf(images[currentIndex]);
            let newFilteredIdx = currentInFiltered + dir;
            if (newFilteredIdx < 0) newFilteredIdx = filteredImages.length - 1;
            if (newFilteredIdx >= filteredImages.length) newFilteredIdx = 0;
            selectImage(images.indexOf(filteredImages[newFilteredIdx]));
        }
        
        function updateLabelCount() {
            document.getElementById('labelCount').textContent = 'Etiket: ' + labels.length;
        }
        
        function showToast(msg, type) {
            const t = document.createElement('div');
            t.className = 'toast ' + type;
            t.textContent = msg;
            document.body.appendChild(t);
            setTimeout(() => t.remove(), 2000);
        }
        
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'ArrowRight') navigate(1);
            if (e.key === 'Delete' || e.key === 'Backspace') deleteSelected();
            if (e.key === 's' && e.ctrlKey) { e.preventDefault(); saveLabels(); }
            if (e.key === 'd') setMode('draw');
            if (e.key === 'v') setMode('select');
            for (let i = 0; i < config.classes.length; i++) {
                if (e.key === (i + 1).toString()) setClass(i);
            }
        });
        
        window.addEventListener('resize', () => {
            if (currentImage) { resizeCanvas(); render(); }
        });
        
        init();
    </script>
</body>
</html>
"""

def load_model():
    global model, processor, device
    
    if model is None:
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model loaded on {device}")

def detect_objects(image_path):
    global model, processor, device, config
    
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    # Build prompt from config
    prompt = ". ".join([c["prompt"] for c in config["classes"]]) + "."
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, target_sizes=[(h, w)]
    )[0]
    
    mask = results["scores"] > config["confidence"]
    
    detections = []
    boxes = results["boxes"][mask].cpu().numpy()
    scores = results["scores"][mask].cpu().numpy()
    labels_list = [l for l, m in zip(results["labels"], mask) if m]
    
    for box, score, label in zip(boxes, scores, labels_list):
        label_lower = label.lower().strip()
        
        # Önce spesifik kelimelere bak (rim/hoop öncelikli)
        class_id = None
        
        # Rim kontrolü (hoop, ring, rim, net içeriyorsa)
        if any(word in label_lower for word in ["hoop", "ring", "rim", "net", "goal"]):
            class_id = 1  # rim
        # Player kontrolü (person, player, man, woman içeriyorsa)
        elif any(word in label_lower for word in ["person", "player", "man", "woman", "people"]):
            class_id = 2  # player
        # Basketball kontrolü (ball içeriyorsa ve hoop değilse)
        elif "ball" in label_lower or (label_lower == "basketball"):
            class_id = 0  # basketball
        else:
            # Config'deki promptlarla eşleştir
            for idx, cls in enumerate(config["classes"]):
                prompt_words = cls["prompt"].lower().split()
                # En az 2 kelime eşleşmeli veya sınıf adı tam eşleşmeli
                matches = sum(1 for word in prompt_words if word in label_lower)
                if matches >= 2 or cls["name"].lower() in label_lower:
                    class_id = idx
                    break
        
        if class_id is None:
            continue
        
        x1, y1, x2, y2 = box
        detections.append({
            "class_id": class_id,
            "x_center": ((x1 + x2) / 2) / w,
            "y_center": ((y1 + y2) / 2) / h,
            "width": (x2 - x1) / w,
            "height": (y2 - y1) / h
        })
    
    return detections

def labeling_thread():
    global config, is_labeling
    
    is_labeling = True
    
    try:
        socketio.emit('progress', {'percent': 0, 'message': 'Model yükleniyor...'})
        load_model()
        
        os.makedirs(config["labels_dir"], exist_ok=True)
        
        image_files = []
        seen = set()
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in Path(config["images_dir"]).glob(ext):
                if img_path.name.lower() not in seen:
                    seen.add(img_path.name.lower())
                    image_files.append(img_path)
        
        total = len(image_files)
        stats = {'total': 0, 'class0': 0, 'class1': 0, 'class2': 0}
        
        for i, img_path in enumerate(image_files):
            try:
                detections = detect_objects(img_path)
                
                label_path = Path(config["labels_dir"]) / (img_path.stem + '.txt')
                with open(label_path, 'w') as f:
                    for det in detections:
                        f.write(f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f}\n")
                        stats[f'class{det["class_id"]}'] = stats.get(f'class{det["class_id"]}', 0) + 1
                
                stats['total'] += 1
                percent = int((i + 1) / total * 100)
                socketio.emit('progress', {'percent': percent, 'message': f'{img_path.name} ({i+1}/{total})'})
                socketio.emit('stats', stats)
                
            except Exception as e:
                print(f"Error: {img_path}: {e}")
        
        socketio.emit('complete', {})
        
    except Exception as e:
        socketio.emit('error', {'message': str(e)})
    
    finally:
        is_labeling = False

@app.route('/')
def index():
    return render_template_string(MAIN_PAGE)

@app.route('/editor')
def editor():
    return render_template_string(EDITOR_PAGE)

@app.route('/api/config')
def get_config():
    return jsonify(config)

@app.route('/api/start_labeling', methods=['POST'])
def start_labeling():
    global config, is_labeling
    
    if is_labeling:
        return jsonify({'success': False, 'error': 'Etiketleme zaten çalışıyor!'})
    
    data = request.json
    
    if not os.path.exists(data['images_dir']):
        return jsonify({'success': False, 'error': 'Resim klasörü bulunamadı!'})
    
    config["images_dir"] = data['images_dir']
    config["labels_dir"] = data['labels_dir']
    config["confidence"] = data['confidence']
    config["classes"] = data['classes']
    
    thread = threading.Thread(target=labeling_thread)
    thread.start()
    
    return jsonify({'success': True})

@app.route('/api/images')
def get_images():
    images = []
    
    if not config.get("images_dir") or not os.path.exists(config["images_dir"]):
        return jsonify({'images': []})
    
    seen = set()
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in Path(config["images_dir"]).glob(ext):
            if img_path.name.lower() in seen:
                continue
            seen.add(img_path.name.lower())
            
            label_path = Path(config["labels_dir"]) / (img_path.stem + '.txt')
            label_count = 0
            if label_path.exists():
                with open(label_path) as f:
                    label_count = len([l for l in f if l.strip()])
            images.append({'name': img_path.name, 'label_count': label_count})
    
    images.sort(key=lambda x: x['name'])
    return jsonify({'images': images})

@app.route('/api/image/<filename>')
def get_image(filename):
    img_path = Path(config["images_dir"]) / filename
    label_path = Path(config["labels_dir"]) / (Path(filename).stem + '.txt')
    
    with open(img_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    labels = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append({
                        'class_id': parts[0],
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
    
    return jsonify({'image': img_data, 'labels': labels})

@app.route('/api/save/<filename>', methods=['POST'])
def save_labels(filename):
    data = request.json
    labels = data.get('labels', [])
    
    os.makedirs(config["labels_dir"], exist_ok=True)
    label_path = Path(config["labels_dir"]) / (Path(filename).stem + '.txt')
    
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(f"{label['class_id']} {label['x_center']:.6f} {label['y_center']:.6f} {label['width']:.6f} {label['height']:.6f}\n")
    
    return jsonify({'success': True})

@app.route('/api/delete_image/<filename>', methods=['DELETE'])
def delete_image(filename):
    try:
        # Resmi sil
        img_path = Path(config["images_dir"]) / filename
        if img_path.exists():
            os.remove(img_path)
        
        # Etiketi sil
        label_path = Path(config["labels_dir"]) / (Path(filename).stem + '.txt')
        if label_path.exists():
            os.remove(label_path)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🏀 Otomatik Etiketleme Uygulaması")
    print("=" * 60)
    print("\n🌐 Ana sayfa: http://localhost:5000")
    print("✏️ Editör: http://localhost:5000/editor")
    print("\n🛑 Durdurmak için: Ctrl+C")
    print("=" * 60)
    
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
