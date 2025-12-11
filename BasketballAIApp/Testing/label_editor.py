"""
Web tabanlı YOLO Etiket Editörü
- Klasör seçimi
- Bounding box çizme
- Bounding box taşıma/boyutlandırma
- Etiket silme
"""

from flask import Flask, render_template_string, jsonify, request, send_file
import cv2
import os
import sys
from pathlib import Path
import base64
import argparse

app = Flask(__name__)

# Varsayılan klasörler (komut satırından değiştirilebilir)
IMAGES_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\hepsi"
LABELS_DIR = r"D:\repos\Basketball_App\BasketballAIApp\Datasets\frames_output\labels"

CLASS_NAMES = ["basketball", "rim", "player"]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Etiket Editörü</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #0f0f1a; 
            color: #eee;
            min-height: 100vh;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 12px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e94560;
            height: 60px;
        }
        .header h1 { color: #e94560; font-size: 22px; }
        .header-controls { display: flex; gap: 15px; align-items: center; }
        .folder-display {
            background: #0f3460;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 12px;
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .container { 
            display: flex; 
            height: calc(100vh - 60px);
        }
        .sidebar {
            width: 260px;
            background: #16213e;
            display: flex;
            flex-direction: column;
            border-right: 2px solid #0f3460;
        }
        .search-box {
            padding: 10px;
            background: #0f3460;
        }
        .search-box input {
            width: 100%;
            padding: 8px 12px;
            border: none;
            border-radius: 5px;
            background: #1a1a2e;
            color: #eee;
            font-size: 13px;
        }
        .file-list {
            flex: 1;
            overflow-y: auto;
        }
        .file-item {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
        }
        .file-item:hover { background: #0f3460; }
        .file-item.active { background: #e94560; }
        .file-item .badge {
            background: #0f3460;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
        }
        .file-item.active .badge { background: #fff; color: #e94560; }
        
        .main-area {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .toolbar {
            background: #16213e;
            padding: 10px 20px;
            display: flex;
            gap: 10px;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        .tool-group {
            display: flex;
            gap: 5px;
            padding: 0 15px;
            border-right: 1px solid #0f3460;
        }
        .tool-group:last-child { border-right: none; }
        .tool-btn {
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 13px;
            background: #0f3460;
            color: #aaa;
            transition: all 0.2s;
        }
        .tool-btn:hover { background: #1a4a7a; color: #fff; }
        .tool-btn.active { background: #e94560; color: #fff; }
        .class-btn {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .class-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
        
        .canvas-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #0a0a15;
            position: relative;
            overflow: hidden;
        }
        #canvas {
            cursor: crosshair;
            max-width: 100%;
            max-height: 100%;
        }
        
        .info-bar {
            background: #16213e;
            padding: 8px 20px;
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #888;
        }
        
        .message {
            position: fixed;
            top: 70px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 13px;
            animation: slideIn 0.3s;
            z-index: 1000;
        }
        .message.success { background: #00b894; color: #fff; }
        .message.error { background: #e94560; color: #fff; }
        .message.info { background: #0984e3; color: #fff; }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .help-text {
            color: #666;
            font-size: 11px;
        }
        
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
        }
        .modal-content {
            background: #16213e;
            padding: 30px;
            border-radius: 10px;
            min-width: 500px;
        }
        .modal h2 { margin-bottom: 20px; color: #e94560; }
        .modal input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #0f3460;
            border-radius: 5px;
            background: #1a1a2e;
            color: #fff;
            font-size: 14px;
            margin-bottom: 15px;
        }
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        .modal-btn {
            padding: 10px 25px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .modal-btn.primary { background: #e94560; color: #fff; }
        .modal-btn.secondary { background: #0f3460; color: #fff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏀 YOLO Etiket Editörü</h1>
        <div class="header-controls">
            <div class="folder-display" id="folderDisplay">{{ images_dir }}</div>
            <button class="tool-btn" onclick="showFolderModal()">📁 Klasör Değiştir</button>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="search-box">
                <input type="text" placeholder="🔍 Ara..." oninput="filterFiles(this.value)">
            </div>
            <div class="file-list" id="fileList"></div>
        </div>
        
        <div class="main-area">
            <div class="toolbar">
                <div class="tool-group">
                    <button class="tool-btn" id="selectMode" onclick="setMode('select')">🔲 Seç</button>
                    <button class="tool-btn active" id="drawMode" onclick="setMode('draw')">✏️ Çiz</button>
                </div>
                <div class="tool-group">
                    <button class="tool-btn class-btn active" id="class0" onclick="setClass(0)">
                        <span class="class-color" style="background: #FFA500;"></span> Basketball
                    </button>
                    <button class="tool-btn class-btn" id="class1" onclick="setClass(1)">
                        <span class="class-color" style="background: #00FF00;"></span> Rim
                    </button>
                    <button class="tool-btn class-btn" id="class2" onclick="setClass(2)">
                        <span class="class-color" style="background: #0080FF;"></span> Player
                    </button>
                </div>
                <div class="tool-group">
                    <button class="tool-btn" onclick="deleteSelected()">🗑️ Seçileni Sil</button>
                    <button class="tool-btn" onclick="deleteAll()">🗑️ Tümünü Sil</button>
                </div>
                <div class="tool-group">
                    <button class="tool-btn" onclick="saveLabels()" style="background: #00b894;">💾 Kaydet</button>
                </div>
                <div class="tool-group">
                    <button class="tool-btn" onclick="navigate(-1)">← Önceki</button>
                    <button class="tool-btn" onclick="navigate(1)">Sonraki →</button>
                </div>
            </div>
            
            <div class="canvas-container">
                <canvas id="canvas"></canvas>
            </div>
            
            <div class="info-bar">
                <span id="imageInfo">Resim seçin</span>
                <span id="labelCount">Etiket: 0</span>
                <span class="help-text">Çizmek için sürükle | Taşımak için kutuyu sürükle | Boyutlandırmak için köşeleri sürükle | Del: Sil</span>
            </div>
        </div>
    </div>
    
    <div class="modal" id="folderModal" style="display: none;">
        <div class="modal-content">
            <h2>📁 Klasör Ayarları</h2>
            <label style="color: #888; font-size: 13px;">Resim Klasörü:</label>
            <input type="text" id="imagesDirInput" value="{{ images_dir }}">
            <label style="color: #888; font-size: 13px;">Etiket Klasörü:</label>
            <input type="text" id="labelsDirInput" value="{{ labels_dir }}">
            <div class="modal-buttons">
                <button class="modal-btn secondary" onclick="closeFolderModal()">İptal</button>
                <button class="modal-btn primary" onclick="changeFolders()">Uygula</button>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let images = [];
        let filteredImages = [];
        let currentIndex = -1;
        let currentImage = null;
        let labels = [];
        let selectedLabel = -1;
        
        let mode = 'draw';  // 'draw' or 'select'
        let currentClass = 0;
        let isDrawing = false;
        let isDragging = false;
        let isResizing = false;
        let resizeHandle = null;
        let startX, startY;
        let dragOffsetX, dragOffsetY;
        
        const colors = ['#FFA500', '#00FF00', '#0080FF'];
        const classNames = ['basketball', 'rim', 'player'];
        
        let scale = 1;
        let offsetX = 0, offsetY = 0;
        
        // Load images list
        async function loadImages() {
            const response = await fetch('/api/images');
            const data = await response.json();
            images = data.images;
            filteredImages = images;
            renderFileList();
            showMessage(`${images.length} resim yüklendi`, 'info');
        }
        
        function filterFiles(query) {
            query = query.toLowerCase();
            filteredImages = images.filter(img => img.name.toLowerCase().includes(query));
            renderFileList();
        }
        
        function renderFileList() {
            const container = document.getElementById('fileList');
            container.innerHTML = filteredImages.map((img, idx) => `
                <div class="file-item ${filteredImages[idx] === images[currentIndex] ? 'active' : ''}" 
                     onclick="selectImage(${images.indexOf(img)})">
                    <span>${img.name}</span>
                    <span class="badge">${img.label_count}</span>
                </div>
            `).join('');
        }
        
        async function selectImage(index) {
            currentIndex = index;
            const img = images[index];
            
            const response = await fetch(`/api/image/${img.name}`);
            const data = await response.json();
            
            labels = data.labels.map(l => ({
                classId: parseInt(l.class_id),
                x: l.x_center,
                y: l.y_center,
                w: l.width,
                h: l.height
            }));
            
            selectedLabel = -1;
            
            currentImage = new Image();
            currentImage.onload = () => {
                resizeCanvas();
                render();
            };
            currentImage.src = 'data:image/jpeg;base64,' + data.image;
            
            document.getElementById('imageInfo').textContent = 
                `${img.name} (${currentIndex + 1}/${images.length})`;
            updateLabelCount();
            renderFileList();
        }
        
        function resizeCanvas() {
            const container = canvas.parentElement;
            const maxW = container.clientWidth - 40;
            const maxH = container.clientHeight - 40;
            
            const imgRatio = currentImage.width / currentImage.height;
            const containerRatio = maxW / maxH;
            
            if (imgRatio > containerRatio) {
                canvas.width = maxW;
                canvas.height = maxW / imgRatio;
            } else {
                canvas.height = maxH;
                canvas.width = maxH * imgRatio;
            }
            
            scale = canvas.width / currentImage.width;
        }
        
        function render() {
            if (!currentImage) return;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
            
            // Draw labels
            labels.forEach((label, idx) => {
                const x = (label.x - label.w/2) * canvas.width;
                const y = (label.y - label.h/2) * canvas.height;
                const w = label.w * canvas.width;
                const h = label.h * canvas.height;
                
                ctx.strokeStyle = colors[label.classId];
                ctx.lineWidth = idx === selectedLabel ? 4 : 2;
                ctx.strokeRect(x, y, w, h);
                
                // Fill for selected
                if (idx === selectedLabel) {
                    ctx.fillStyle = colors[label.classId] + '33';
                    ctx.fillRect(x, y, w, h);
                    
                    // Resize handles
                    ctx.fillStyle = '#fff';
                    const handleSize = 8;
                    ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize);
                    ctx.fillRect(x + w - handleSize/2, y - handleSize/2, handleSize, handleSize);
                    ctx.fillRect(x - handleSize/2, y + h - handleSize/2, handleSize, handleSize);
                    ctx.fillRect(x + w - handleSize/2, y + h - handleSize/2, handleSize, handleSize);
                }
                
                // Label
                ctx.fillStyle = colors[label.classId];
                ctx.fillRect(x, y - 20, 80, 18);
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 12px sans-serif';
                ctx.fillText(classNames[label.classId], x + 5, y - 6);
            });
            
            // Draw current drawing
            if (isDrawing) {
                const rect = getDrawingRect();
                ctx.strokeStyle = colors[currentClass];
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
                ctx.setLineDash([]);
            }
        }
        
        function getDrawingRect() {
            const x = Math.min(startX, lastX);
            const y = Math.min(startY, lastY);
            const w = Math.abs(lastX - startX);
            const h = Math.abs(lastY - startY);
            return {x, y, w, h};
        }
        
        let lastX, lastY;
        
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            startX = x;
            startY = y;
            lastX = x;
            lastY = y;
            
            if (mode === 'select' || e.button === 2) {
                // Check if clicking on resize handle
                if (selectedLabel >= 0) {
                    const label = labels[selectedLabel];
                    const lx = (label.x - label.w/2) * canvas.width;
                    const ly = (label.y - label.h/2) * canvas.height;
                    const lw = label.w * canvas.width;
                    const lh = label.h * canvas.height;
                    
                    const handleSize = 12;
                    const handles = [
                        {name: 'tl', x: lx, y: ly},
                        {name: 'tr', x: lx + lw, y: ly},
                        {name: 'bl', x: lx, y: ly + lh},
                        {name: 'br', x: lx + lw, y: ly + lh}
                    ];
                    
                    for (let h of handles) {
                        if (Math.abs(x - h.x) < handleSize && Math.abs(y - h.y) < handleSize) {
                            isResizing = true;
                            resizeHandle = h.name;
                            return;
                        }
                    }
                }
                
                // Check if clicking on a label
                for (let i = labels.length - 1; i >= 0; i--) {
                    const label = labels[i];
                    const lx = (label.x - label.w/2) * canvas.width;
                    const ly = (label.y - label.h/2) * canvas.height;
                    const lw = label.w * canvas.width;
                    const lh = label.h * canvas.height;
                    
                    if (x >= lx && x <= lx + lw && y >= ly && y <= ly + lh) {
                        selectedLabel = i;
                        isDragging = true;
                        dragOffsetX = x - (label.x * canvas.width);
                        dragOffsetY = y - (label.y * canvas.height);
                        render();
                        return;
                    }
                }
                
                selectedLabel = -1;
                render();
            } else {
                isDrawing = true;
            }
        });
        
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            lastX = x;
            lastY = y;
            
            if (isDrawing) {
                render();
            } else if (isDragging && selectedLabel >= 0) {
                labels[selectedLabel].x = (x - dragOffsetX) / canvas.width;
                labels[selectedLabel].y = (y - dragOffsetY) / canvas.height;
                render();
            } else if (isResizing && selectedLabel >= 0) {
                const label = labels[selectedLabel];
                const lx = (label.x - label.w/2) * canvas.width;
                const ly = (label.y - label.h/2) * canvas.height;
                const lw = label.w * canvas.width;
                const lh = label.h * canvas.height;
                
                let newX1 = lx, newY1 = ly, newX2 = lx + lw, newY2 = ly + lh;
                
                if (resizeHandle.includes('l')) newX1 = x;
                if (resizeHandle.includes('r')) newX2 = x;
                if (resizeHandle.includes('t')) newY1 = y;
                if (resizeHandle.includes('b')) newY2 = y;
                
                const newW = Math.abs(newX2 - newX1);
                const newH = Math.abs(newY2 - newY1);
                const newCX = (Math.min(newX1, newX2) + newW/2) / canvas.width;
                const newCY = (Math.min(newY1, newY2) + newH/2) / canvas.height;
                
                label.x = newCX;
                label.y = newCY;
                label.w = newW / canvas.width;
                label.h = newH / canvas.height;
                
                render();
            }
        });
        
        canvas.addEventListener('mouseup', (e) => {
            if (isDrawing) {
                const rect = getDrawingRect();
                if (rect.w > 10 && rect.h > 10) {
                    const label = {
                        classId: currentClass,
                        x: (rect.x + rect.w/2) / canvas.width,
                        y: (rect.y + rect.h/2) / canvas.height,
                        w: rect.w / canvas.width,
                        h: rect.h / canvas.height
                    };
                    labels.push(label);
                    selectedLabel = labels.length - 1;
                    updateLabelCount();
                }
            }
            
            isDrawing = false;
            isDragging = false;
            isResizing = false;
            resizeHandle = null;
            render();
        });
        
        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        function setMode(m) {
            mode = m;
            document.getElementById('selectMode').classList.toggle('active', m === 'select');
            document.getElementById('drawMode').classList.toggle('active', m === 'draw');
            canvas.style.cursor = m === 'draw' ? 'crosshair' : 'default';
        }
        
        function setClass(c) {
            currentClass = c;
            for (let i = 0; i < 3; i++) {
                document.getElementById('class' + i).classList.toggle('active', i === c);
            }
            
            // Update selected label class
            if (selectedLabel >= 0) {
                labels[selectedLabel].classId = c;
                render();
            }
        }
        
        function deleteSelected() {
            if (selectedLabel >= 0) {
                labels.splice(selectedLabel, 1);
                selectedLabel = -1;
                updateLabelCount();
                render();
                showMessage('Etiket silindi', 'success');
            }
        }
        
        function deleteAll() {
            if (labels.length === 0) return;
            if (confirm('Tüm etiketleri silmek istediğinizden emin misiniz?')) {
                labels = [];
                selectedLabel = -1;
                updateLabelCount();
                render();
                showMessage('Tüm etiketler silindi', 'success');
            }
        }
        
        async function saveLabels() {
            if (currentIndex < 0) return;
            
            const img = images[currentIndex];
            const saveLabels = labels.map(l => ({
                class_id: l.classId.toString(),
                x_center: l.x,
                y_center: l.y,
                width: l.w,
                height: l.h
            }));
            
            const response = await fetch(`/api/save/${img.name}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({labels: saveLabels})
            });
            
            if (response.ok) {
                img.label_count = labels.length;
                renderFileList();
                showMessage('Kaydedildi!', 'success');
            } else {
                showMessage('Kaydetme hatası!', 'error');
            }
        }
        
        function navigate(dir) {
            let newIndex = currentIndex + dir;
            if (newIndex < 0) newIndex = images.length - 1;
            if (newIndex >= images.length) newIndex = 0;
            selectImage(newIndex);
        }
        
        function updateLabelCount() {
            document.getElementById('labelCount').textContent = `Etiket: ${labels.length}`;
        }
        
        function showMessage(text, type) {
            const msg = document.createElement('div');
            msg.className = 'message ' + type;
            msg.textContent = text;
            document.body.appendChild(msg);
            setTimeout(() => msg.remove(), 2000);
        }
        
        function showFolderModal() {
            document.getElementById('folderModal').style.display = 'flex';
        }
        
        function closeFolderModal() {
            document.getElementById('folderModal').style.display = 'none';
        }
        
        async function changeFolders() {
            const imagesDir = document.getElementById('imagesDirInput').value;
            const labelsDir = document.getElementById('labelsDirInput').value;
            
            const response = await fetch('/api/set_folders', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({images_dir: imagesDir, labels_dir: labelsDir})
            });
            
            if (response.ok) {
                document.getElementById('folderDisplay').textContent = imagesDir;
                closeFolderModal();
                currentIndex = -1;
                currentImage = null;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                loadImages();
            } else {
                showMessage('Klasör bulunamadı!', 'error');
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            
            if (e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'ArrowRight') navigate(1);
            if (e.key === 'Delete' || e.key === 'Backspace') deleteSelected();
            if (e.key === 's' && e.ctrlKey) { e.preventDefault(); saveLabels(); }
            if (e.key === '1') setClass(0);
            if (e.key === '2') setClass(1);
            if (e.key === '3') setClass(2);
            if (e.key === 'd') setMode('draw');
            if (e.key === 'v') setMode('select');
        });
        
        window.addEventListener('resize', () => {
            if (currentImage) {
                resizeCanvas();
                render();
            }
        });
        
        loadImages();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, images_dir=IMAGES_DIR, labels_dir=LABELS_DIR)

@app.route('/api/images')
def get_images():
    images = []
    
    if not os.path.exists(IMAGES_DIR):
        return jsonify({'images': []})
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        for img_path in Path(IMAGES_DIR).glob(ext):
            label_path = Path(LABELS_DIR) / (img_path.stem + '.txt')
            label_count = 0
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_count = len([l for l in f.readlines() if l.strip()])
            
            images.append({
                'name': img_path.name,
                'label_count': label_count
            })
    
    images.sort(key=lambda x: x['name'])
    return jsonify({'images': images})

@app.route('/api/image/<filename>')
def get_image(filename):
    img_path = Path(IMAGES_DIR) / filename
    label_path = Path(LABELS_DIR) / (Path(filename).stem + '.txt')
    
    if not img_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    
    with open(img_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
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
    
    return jsonify({
        'image': img_data,
        'labels': labels
    })

@app.route('/api/save/<filename>', methods=['POST'])
def save_labels(filename):
    global LABELS_DIR
    
    data = request.json
    labels = data.get('labels', [])
    
    os.makedirs(LABELS_DIR, exist_ok=True)
    label_path = Path(LABELS_DIR) / (Path(filename).stem + '.txt')
    
    with open(label_path, 'w') as f:
        for label in labels:
            line = f"{label['class_id']} {label['x_center']:.6f} {label['y_center']:.6f} {label['width']:.6f} {label['height']:.6f}\n"
            f.write(line)
    
    return jsonify({'success': True})

@app.route('/api/set_folders', methods=['POST'])
def set_folders():
    global IMAGES_DIR, LABELS_DIR
    
    data = request.json
    new_images_dir = data.get('images_dir', '')
    new_labels_dir = data.get('labels_dir', '')
    
    if not os.path.exists(new_images_dir):
        return jsonify({'error': 'Images folder not found'}), 400
    
    IMAGES_DIR = new_images_dir
    LABELS_DIR = new_labels_dir
    
    os.makedirs(LABELS_DIR, exist_ok=True)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Label Editor')
    parser.add_argument('--images', '-i', help='Images directory')
    parser.add_argument('--labels', '-l', help='Labels directory')
    parser.add_argument('--port', '-p', type=int, default=5000, help='Port number')
    
    args = parser.parse_args()
    
    if args.images:
        IMAGES_DIR = args.images
    if args.labels:
        LABELS_DIR = args.labels
    
    print("=" * 60)
    print("🏀 YOLO Etiket Editörü")
    print("=" * 60)
    print(f"📁 Resimler: {IMAGES_DIR}")
    print(f"📁 Etiketler: {LABELS_DIR}")
    print(f"\n🌐 Tarayıcıda açın: http://localhost:{args.port}")
    print("\n⌨️  Klavye Kısayolları:")
    print("   ← →      : Önceki/Sonraki resim")
    print("   1 2 3    : Sınıf seç (basketball/rim/player)")
    print("   D        : Çizim modu")
    print("   V        : Seçim modu")
    print("   Del      : Seçili etiketi sil")
    print("   Ctrl+S   : Kaydet")
    print("\n🖱️  Mouse:")
    print("   Sürükle      : Yeni kutu çiz")
    print("   Kutuya tıkla : Seç")
    print("   Köşeleri sürükle : Boyutlandır")
    print("\n🛑 Durdurmak için: Ctrl+C")
    print("=" * 60)
    
    app.run(debug=True, port=args.port, threaded=True)
