/**
 * BSL Sign Recorder - Main Application
 *
 * Captures hand landmarks via camera using TensorFlow.js hand-pose-detection
 * and exports recordings as JSON for BSL sign data collection.
 */

// =============================================================================
// State
// =============================================================================

const state = {
    detector: null,
    videoStream: null,
    isDetecting: false,
    isRecording: false,
    currentRecording: null,
    recordedSigns: new Set(),
    selectedSignId: null,
    mirrorVideo: true,
    recordingDuration: 3000,
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    video: document.getElementById('video'),
    canvas: document.getElementById('canvas'),
    status: document.getElementById('status'),
    statusText: document.querySelector('.status-text'),
    countdown: document.getElementById('countdown'),
    recordingIndicator: document.getElementById('recording-indicator'),
    cameraSelect: document.getElementById('camera-select'),
    mirrorCheckbox: document.getElementById('mirror'),
    categorySelect: document.getElementById('category-select'),
    signSelect: document.getElementById('sign-select'),
    signInfo: document.getElementById('sign-info'),
    durationInput: document.getElementById('duration'),
    recordBtn: document.getElementById('record-btn'),
    stopBtn: document.getElementById('stop-btn'),
    reviewPanel: document.getElementById('review-panel'),
    frameCount: document.getElementById('frame-count'),
    recDuration: document.getElementById('rec-duration'),
    handsDetected: document.getElementById('hands-detected'),
    saveBtn: document.getElementById('save-btn'),
    discardBtn: document.getElementById('discard-btn'),
    recordedCount: document.getElementById('recorded-count'),
    totalCount: document.getElementById('total-count'),
    progressFill: document.getElementById('progress-fill'),
    signChecklist: document.getElementById('sign-checklist'),
};

// =============================================================================
// Camera Module
// =============================================================================

async function initCamera() {
    try {
        // Get available cameras
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');

        // Populate camera dropdown
        elements.cameraSelect.innerHTML = cameras.map((cam, i) =>
            `<option value="${cam.deviceId}">${cam.label || `Camera ${i + 1}`}</option>`
        ).join('');

        // Start with first camera
        if (cameras.length > 0) {
            await startCamera(cameras[0].deviceId);
        }
    } catch (err) {
        console.error('Camera init failed:', err);
        setStatus('error', 'Camera access denied');
    }
}

async function startCamera(deviceId) {
    // Stop existing stream
    if (state.videoStream) {
        state.videoStream.getTracks().forEach(t => t.stop());
    }

    try {
        const constraints = {
            video: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user',
            }
        };

        state.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.video.srcObject = state.videoStream;

        await new Promise(resolve => {
            elements.video.onloadedmetadata = resolve;
        });

        // Set canvas size
        elements.canvas.width = elements.video.videoWidth;
        elements.canvas.height = elements.video.videoHeight;

        updateMirror();

    } catch (err) {
        console.error('Camera start failed:', err);
        setStatus('error', 'Failed to start camera');
    }
}

function updateMirror() {
    state.mirrorVideo = elements.mirrorCheckbox.checked;
    elements.video.classList.toggle('mirrored', state.mirrorVideo);
    elements.canvas.classList.toggle('mirrored', state.mirrorVideo);
}

// =============================================================================
// Detection Module
// =============================================================================

async function initDetector() {
    setStatus('loading', 'Loading hand detection model...');

    try {
        const model = handPoseDetection.SupportedModels.MediaPipeHands;
        const detectorConfig = {
            runtime: 'mediapipe',
            solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240',
            modelType: 'full',
            maxHands: 2,
        };

        state.detector = await handPoseDetection.createDetector(model, detectorConfig);

        setStatus('ready', 'Ready to record');
        elements.recordBtn.disabled = false;

        // Start detection loop
        startDetectionLoop();

    } catch (err) {
        console.error('Detector init failed:', err);
        setStatus('error', 'Failed to load model');
    }
}

function startDetectionLoop() {
    state.isDetecting = true;
    detectFrame();
}

async function detectFrame() {
    if (!state.isDetecting || !state.detector) return;

    try {
        const hands = await state.detector.estimateHands(elements.video, {
            flipHorizontal: false,
        });

        drawLandmarks(hands);

        // If recording, store frame
        if (state.isRecording && state.currentRecording) {
            const timestamp = Date.now() - state.currentRecording.startTime;
            state.currentRecording.frames.push({
                timestamp_ms: timestamp,
                hands: hands.map(hand => ({
                    handedness: hand.handedness,
                    confidence: hand.score,
                    landmarks: hand.keypoints3D
                        ? hand.keypoints3D.map(p => [p.x, p.y, p.z])
                        : hand.keypoints.map(p => [p.x / elements.video.videoWidth, p.y / elements.video.videoHeight, 0]),
                })),
            });
        }

    } catch (err) {
        console.error('Detection error:', err);
    }

    // Continue loop at ~30fps
    requestAnimationFrame(detectFrame);
}

function drawLandmarks(hands) {
    const ctx = elements.canvas.getContext('2d');
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

    // Finger colors
    const fingerColors = {
        thumb: '#ef4444',
        index_finger: '#f97316',
        middle_finger: '#eab308',
        ring_finger: '#22c55e',
        pinky: '#3b82f6',
        wrist: '#a855f7',
    };

    // Connections
    const connections = [
        [0,1],[1,2],[2,3],[3,4],           // thumb
        [0,5],[5,6],[6,7],[7,8],           // index
        [0,9],[9,10],[10,11],[11,12],      // middle
        [0,13],[13,14],[14,15],[15,16],    // ring
        [0,17],[17,18],[18,19],[19,20],    // pinky
        [5,9],[9,13],[13,17],              // palm
    ];

    for (const hand of hands) {
        const kp = hand.keypoints;

        // Draw connections
        ctx.lineWidth = 3;
        for (const [i, j] of connections) {
            const p1 = kp[i];
            const p2 = kp[j];

            // Get color based on finger
            let color = '#a855f7';
            if (i >= 1 && i <= 4) color = fingerColors.thumb;
            else if (i >= 5 && i <= 8) color = fingerColors.index_finger;
            else if (i >= 9 && i <= 12) color = fingerColors.middle_finger;
            else if (i >= 13 && i <= 16) color = fingerColors.ring_finger;
            else if (i >= 17 && i <= 20) color = fingerColors.pinky;

            ctx.strokeStyle = color;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }

        // Draw keypoints
        for (let i = 0; i < kp.length; i++) {
            const p = kp[i];

            let color = fingerColors.wrist;
            if (i >= 1 && i <= 4) color = fingerColors.thumb;
            else if (i >= 5 && i <= 8) color = fingerColors.index_finger;
            else if (i >= 9 && i <= 12) color = fingerColors.middle_finger;
            else if (i >= 13 && i <= 16) color = fingerColors.ring_finger;
            else if (i >= 17 && i <= 20) color = fingerColors.pinky;

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
            ctx.fill();

            // White center
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

// =============================================================================
// Recording Module
// =============================================================================

async function startRecording() {
    const signId = elements.signSelect.value;
    if (!signId) {
        alert('Please select a sign first');
        return;
    }

    state.selectedSignId = signId;
    state.recordingDuration = parseInt(elements.durationInput.value) * 1000;

    // Countdown
    elements.recordBtn.classList.add('hidden');
    elements.countdown.classList.remove('hidden');

    for (let i = 3; i > 0; i--) {
        elements.countdown.textContent = i;
        await sleep(1000);
    }

    elements.countdown.classList.add('hidden');

    // Start recording
    state.isRecording = true;
    state.currentRecording = {
        sign_id: signId,
        startTime: Date.now(),
        frames: [],
    };

    elements.recordingIndicator.classList.remove('hidden');
    elements.stopBtn.classList.remove('hidden');
    setStatus('recording', 'Recording...');

    // Auto-stop after duration
    setTimeout(() => {
        if (state.isRecording) {
            stopRecording();
        }
    }, state.recordingDuration);
}

function stopRecording() {
    if (!state.isRecording) return;

    state.isRecording = false;
    elements.recordingIndicator.classList.add('hidden');
    elements.stopBtn.classList.add('hidden');
    setStatus('ready', 'Ready');

    // Calculate stats
    const recording = state.currentRecording;
    const duration = Date.now() - recording.startTime;
    const frameCount = recording.frames.length;
    const framesWithHands = recording.frames.filter(f => f.hands.length > 0).length;
    const handPercent = frameCount > 0 ? Math.round((framesWithHands / frameCount) * 100) : 0;

    // Show review panel
    elements.frameCount.textContent = frameCount;
    elements.recDuration.textContent = duration;
    elements.handsDetected.textContent = handPercent;
    elements.reviewPanel.classList.remove('hidden');
}

function saveRecording() {
    const recording = state.currentRecording;
    if (!recording) return;

    const duration = recording.frames.length > 0
        ? recording.frames[recording.frames.length - 1].timestamp_ms
        : 0;

    const exportData = {
        sign_id: recording.sign_id,
        recorded_at: new Date().toISOString(),
        duration_ms: duration,
        frame_count: recording.frames.length,
        frames: recording.frames,
    };

    // Generate filename
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `${recording.sign_id}_${timestamp}.json`;

    // Download
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);

    // Mark as recorded
    state.recordedSigns.add(recording.sign_id);
    saveProgress();
    updateProgressUI();

    // Reset
    discardRecording();
}

function discardRecording() {
    state.currentRecording = null;
    elements.reviewPanel.classList.add('hidden');
    elements.recordBtn.classList.remove('hidden');
}

// =============================================================================
// Progress Tracking
// =============================================================================

function loadProgress() {
    const saved = localStorage.getItem('bsl_recorded_signs');
    if (saved) {
        state.recordedSigns = new Set(JSON.parse(saved));
    }
}

function saveProgress() {
    localStorage.setItem('bsl_recorded_signs', JSON.stringify([...state.recordedSigns]));
}

function updateProgressUI() {
    const total = SIGN_DEFINITIONS.length;
    const recorded = state.recordedSigns.size;
    const percent = total > 0 ? (recorded / total) * 100 : 0;

    elements.recordedCount.textContent = recorded;
    elements.totalCount.textContent = total;
    elements.progressFill.style.width = `${percent}%`;

    // Update checklist
    const category = elements.categorySelect.value;
    const signs = getSignsByCategory(category || null);

    elements.signChecklist.innerHTML = signs.map(sign => {
        const isRecorded = state.recordedSigns.has(sign.id);
        const isSelected = sign.id === elements.signSelect.value;
        return `<div class="sign-chip ${isRecorded ? 'recorded' : ''} ${isSelected ? 'selected' : ''}"
                     data-sign-id="${sign.id}"
                     title="${sign.name}: ${sign.description}">
            ${sign.name}
        </div>`;
    }).join('');
}

// =============================================================================
// UI Setup
// =============================================================================

function setupUI() {
    // Populate categories
    elements.categorySelect.innerHTML =
        '<option value="">All categories</option>' +
        CATEGORIES.map(cat =>
            `<option value="${cat.id}">${cat.icon} ${cat.name}</option>`
        ).join('');

    // Populate signs
    updateSignSelect();

    // Set total count
    elements.totalCount.textContent = SIGN_DEFINITIONS.length;

    // Load saved progress
    loadProgress();
    updateProgressUI();
}

function updateSignSelect() {
    const category = elements.categorySelect.value;
    const signs = getSignsByCategory(category || null);

    elements.signSelect.innerHTML =
        '<option value="">Select a sign...</option>' +
        signs.map(sign => {
            const recorded = state.recordedSigns.has(sign.id) ? ' ✓' : '';
            return `<option value="${sign.id}">${sign.name}${recorded}</option>`;
        }).join('');

    updateProgressUI();
}

function updateSignInfo() {
    const signId = elements.signSelect.value;
    const sign = getSignById(signId);

    if (sign) {
        elements.signInfo.querySelector('.sign-description').textContent = sign.description;
        elements.signInfo.querySelector('.sign-hands').textContent =
            sign.twoHanded ? '✋✋ Two-handed sign' : '✋ One-handed sign';
        elements.signInfo.style.display = 'block';
    } else {
        elements.signInfo.style.display = 'none';
    }

    updateProgressUI();
}

function setStatus(type, text) {
    elements.status.className = `status status-${type}`;
    elements.statusText.textContent = text;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// Event Listeners
// =============================================================================

function setupEventListeners() {
    // Camera
    elements.cameraSelect.addEventListener('change', e => startCamera(e.target.value));
    elements.mirrorCheckbox.addEventListener('change', updateMirror);

    // Sign selection
    elements.categorySelect.addEventListener('change', updateSignSelect);
    elements.signSelect.addEventListener('change', updateSignInfo);

    // Recording
    elements.recordBtn.addEventListener('click', startRecording);
    elements.stopBtn.addEventListener('click', stopRecording);
    elements.saveBtn.addEventListener('click', saveRecording);
    elements.discardBtn.addEventListener('click', discardRecording);

    // Checklist click
    elements.signChecklist.addEventListener('click', e => {
        const chip = e.target.closest('.sign-chip');
        if (chip) {
            elements.signSelect.value = chip.dataset.signId;
            updateSignInfo();
        }
    });

    // Duration validation
    elements.durationInput.addEventListener('change', e => {
        let val = parseInt(e.target.value);
        if (isNaN(val) || val < 1) val = 1;
        if (val > 10) val = 10;
        e.target.value = val;
    });
}

// =============================================================================
// Initialization
// =============================================================================

async function init() {
    setupUI();
    setupEventListeners();

    await initCamera();
    await initDetector();
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
