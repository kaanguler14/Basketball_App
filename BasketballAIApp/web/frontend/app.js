/**
 * Basketball Shot Analyzer - Frontend
 * ====================================
 */

// Elements - Upload
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const selectedFileName = document.getElementById('selected-file-name');
const selectedFileSize = document.getElementById('selected-file-size');
const removeFileBtn = document.getElementById('remove-file');
const numPlayersSelect = document.getElementById('num-players');
const btnStart = document.getElementById('btn-start');

// Elements - Sections
const uploadSection = document.getElementById('upload-section');
const calibrationSection = document.getElementById('calibration-section');
const processingSection = document.getElementById('processing-section');
const resultsSection = document.getElementById('results-section');

// Elements - Calibration
const calibrationCanvas = document.getElementById('calibration-canvas');
const previewImage = document.getElementById('preview-image');
const pointCountSpan = document.getElementById('point-count');
const btnUndo = document.getElementById('btn-undo');
const btnClear = document.getElementById('btn-clear');
const btnStartProcess = document.getElementById('btn-start-process');

// Elements - Processing
const statusMessage = document.getElementById('status-message');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');

// Elements - Results
const statAccuracy = document.getElementById('stat-accuracy');
const statMade = document.getElementById('stat-made');
const statTotal = document.getElementById('stat-total');
const statPoints = document.getElementById('stat-points');
const stat2pt = document.getElementById('stat-2pt');
const stat3pt = document.getElementById('stat-3pt');
const resultVideo = document.getElementById('result-video');
const downloadLink = document.getElementById('download-link');
const btnNewAnalysis = document.getElementById('btn-new-analysis');

// State
let selectedFile = null;
let currentJobId = null;
let pollInterval = null;
let threePointLine = [];
let canvasCtx = null;
let imageWidth = 0;
let imageHeight = 0;

// API Base URL
const API_BASE = '';

// ============ File Selection ============

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFileSelect(file) {
    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv'];
    
    const hasValidExtension = validExtensions.some(ext => 
        file.name.toLowerCase().endsWith(ext)
    );
    
    if (!hasValidExtension) {
        alert('Lütfen geçerli bir video dosyası seçin (.mp4, .avi, .mov, .mkv)');
        return;
    }
    
    selectedFile = file;
    
    dropZone.style.display = 'none';
    fileInfo.style.display = 'flex';
    
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatFileSize(file.size);
    
    btnStart.disabled = false;
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    
    dropZone.style.display = 'block';
    fileInfo.style.display = 'none';
    
    btnStart.disabled = true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ============ Upload & Calibration ============

btnStart.addEventListener('click', uploadAndShowCalibration);

async function uploadAndShowCalibration() {
    if (!selectedFile) return;
    
    btnStart.disabled = true;
    btnStart.innerHTML = '<span>⏳ Yükleniyor...</span>';
    
    try {
        // Upload file
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('num_players', numPlayersSelect.value);
        
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Video yüklenemedi');
        }
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        // Load preview image for calibration
        await loadPreviewImage();
        
        // Show calibration section
        uploadSection.style.display = 'none';
        calibrationSection.style.display = 'block';
        
    } catch (error) {
        console.error('Upload error:', error);
        alert(`Hata: ${error.message}`);
        btnStart.disabled = false;
        btnStart.innerHTML = '<span>🎯 Devam Et</span>';
    }
}

async function loadPreviewImage() {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    return new Promise((resolve, reject) => {
        img.onload = () => {
            // Set canvas size
            calibrationCanvas.width = img.width;
            calibrationCanvas.height = img.height;
            imageWidth = img.width;
            imageHeight = img.height;
            
            // Get context and draw image
            canvasCtx = calibrationCanvas.getContext('2d');
            canvasCtx.drawImage(img, 0, 0);
            
            // Reset points
            threePointLine = [];
            updatePointCount();
            
            resolve();
        };
        
        img.onerror = () => reject(new Error('Preview yüklenemedi'));
        img.src = `${API_BASE}/api/preview/${currentJobId}?t=${Date.now()}`;
    });
}

// ============ Canvas Drawing ============

calibrationCanvas.addEventListener('click', (e) => {
    const rect = calibrationCanvas.getBoundingClientRect();
    const scaleX = calibrationCanvas.width / rect.width;
    const scaleY = calibrationCanvas.height / rect.height;
    
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    
    threePointLine.push([x, y]);
    updatePointCount();
    redrawCanvas();
});

btnUndo.addEventListener('click', () => {
    if (threePointLine.length > 0) {
        threePointLine.pop();
        updatePointCount();
        redrawCanvas();
    }
});

btnClear.addEventListener('click', () => {
    threePointLine = [];
    updatePointCount();
    redrawCanvas();
});

function updatePointCount() {
    pointCountSpan.textContent = `Seçilen nokta: ${threePointLine.length}`;
    btnStartProcess.disabled = threePointLine.length < 3;
}

function redrawCanvas() {
    // Reload preview image and draw points
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
        canvasCtx.clearRect(0, 0, calibrationCanvas.width, calibrationCanvas.height);
        canvasCtx.drawImage(img, 0, 0);
        
        if (threePointLine.length === 0) return;
        
        // Draw filled polygon
        if (threePointLine.length >= 3) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(threePointLine[0][0], threePointLine[0][1]);
            for (let i = 1; i < threePointLine.length; i++) {
                canvasCtx.lineTo(threePointLine[i][0], threePointLine[i][1]);
            }
            canvasCtx.closePath();
            canvasCtx.fillStyle = 'rgba(255, 107, 53, 0.2)';
            canvasCtx.fill();
        }
        
        // Draw lines
        canvasCtx.beginPath();
        canvasCtx.moveTo(threePointLine[0][0], threePointLine[0][1]);
        for (let i = 1; i < threePointLine.length; i++) {
            canvasCtx.lineTo(threePointLine[i][0], threePointLine[i][1]);
        }
        if (threePointLine.length >= 3) {
            canvasCtx.closePath();
        }
        canvasCtx.strokeStyle = '#FF6B35';
        canvasCtx.lineWidth = 3;
        canvasCtx.stroke();
        
        // Draw points
        threePointLine.forEach((point, index) => {
            canvasCtx.beginPath();
            canvasCtx.arc(point[0], point[1], 8, 0, Math.PI * 2);
            canvasCtx.fillStyle = '#FF6B35';
            canvasCtx.fill();
            canvasCtx.strokeStyle = '#fff';
            canvasCtx.lineWidth = 2;
            canvasCtx.stroke();
            
            // Draw number
            canvasCtx.fillStyle = '#fff';
            canvasCtx.font = 'bold 12px Arial';
            canvasCtx.textAlign = 'center';
            canvasCtx.textBaseline = 'middle';
            canvasCtx.fillText(index + 1, point[0], point[1]);
        });
    };
    
    img.src = `${API_BASE}/api/preview/${currentJobId}?t=${Date.now()}`;
}

// ============ Start Processing ============

btnStartProcess.addEventListener('click', startProcessing);

async function startProcessing() {
    if (threePointLine.length < 3) {
        alert('En az 3 nokta seçmelisiniz');
        return;
    }
    
    // Show processing section
    calibrationSection.style.display = 'none';
    processingSection.style.display = 'block';
    
    updateProgress(5, 'İşlem başlıyor...');
    
    try {
        // Send 3PT line and start processing
        const formData = new FormData();
        formData.append('three_point_line', JSON.stringify(threePointLine));
        
        const response = await fetch(`${API_BASE}/api/start-process/${currentJobId}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('İşlem başlatılamadı');
        }
        
        // Start polling for status
        pollStatus();
        
    } catch (error) {
        console.error('Start processing error:', error);
        showError(error.message);
    }
}

function pollStatus() {
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/status/${currentJobId}`);
            
            if (!response.ok) {
                throw new Error('Durum alınamadı');
            }
            
            const data = await response.json();
            
            updateProgress(data.progress, data.message);
            
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                showResults(data.result);
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                showError(data.message);
            }
            
        } catch (error) {
            console.error('Poll error:', error);
            clearInterval(pollInterval);
            showError('Bağlantı hatası');
        }
    }, 1000);
}

function updateProgress(percent, message) {
    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
    statusMessage.textContent = message;
}

// ============ Results ============

function showResults(result) {
    processingSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    const stats = result.stats;
    
    // Update stats
    statAccuracy.textContent = `${stats.accuracy || 0}%`;
    statMade.textContent = stats.made_shots || 0;
    statTotal.textContent = stats.total_shots || 0;
    
    // 2PT/3PT breakdown
    const twoPt = stats.two_pointers || 0;
    const threePt = stats.three_pointers || 0;
    stat2pt.textContent = twoPt;
    stat3pt.textContent = threePt;
    
    // Total points
    const totalPoints = (twoPt * 2) + (threePt * 3);
    statPoints.textContent = totalPoints;
    
    // Set video source
    const videoUrl = `${API_BASE}${result.video_url}`;
    resultVideo.src = videoUrl;
    downloadLink.href = videoUrl;
    downloadLink.download = `basketball_analysis_${currentJobId}.mp4`;
    
    // Animate stats
    animateStats();
}

function animateStats() {
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        card.style.animation = 'none';
        card.offsetHeight; // Trigger reflow
        card.style.animation = `fadeInUp 0.5s ease-out ${index * 0.1}s both`;
    });
}

function showError(message) {
    processingSection.style.display = 'none';
    calibrationSection.style.display = 'none';
    uploadSection.style.display = 'block';
    
    alert(`Hata: ${message}`);
    resetState();
}

function resetState() {
    currentJobId = null;
    selectedFile = null;
    threePointLine = [];
    
    clearFile();
    btnStart.innerHTML = '<span>🎯 Devam Et</span>';
}

// ============ New Analysis ============

btnNewAnalysis.addEventListener('click', async () => {
    // Cleanup previous job
    if (currentJobId) {
        try {
            await fetch(`${API_BASE}/api/cleanup/${currentJobId}`, {
                method: 'DELETE'
            });
        } catch (e) {
            console.log('Cleanup failed:', e);
        }
    }
    
    // Reset state
    resetState();
    
    // Show upload section
    resultsSection.style.display = 'none';
    uploadSection.style.display = 'block';
});
