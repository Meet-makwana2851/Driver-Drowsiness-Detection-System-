// Configuration
const UPDATE_INTERVAL = 500; // ms
const THRESHOLDS = {
    EAR: 0.21,
    MAR: 0.55,
    DROWSY_COUNTER: 3
};

let lastAlertTime = 0;
let alertCooldown = 5000; // ms

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    startMonitoring();
});

// Start monitoring detection state
function startMonitoring() {
    setInterval(updateDetectionState, UPDATE_INTERVAL);
}

// Update detection state from API
async function updateDetectionState() {
    try {
        const response = await fetch('/api/detection-state');
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error('Error fetching detection state:', error);
    }
}

// Update UI with latest detection data
function updateUI(data) {
    // Update Status
    updateStatus(data);
    
    // Update Metrics
    updateMetrics(data);
    
    // Update Timestamp
    updateTimestamp(data.timestamp);
    
    // Handle Alerts
    if (data.alert_triggered) {
        triggerAlert(data);
    }
}

// Update Status Display
function updateStatus(data) {
    const statusCard = document.getElementById('statusCard');
    const statusText = document.querySelector('.status-text');
    const statusDisplay = document.getElementById('statusDisplay');
    const statusBadge = document.getElementById('statusBadge');
    const statusBadgeText = document.getElementById('statusText');
    
    statusText.textContent = data.status;
    statusDisplay.textContent = data.status;
    
    // Update card styling
    if (data.is_drowsy) {
        statusCard.classList.add('drowsy');
        statusBadge.style.borderColor = '#ef4444';
        statusBadgeText.textContent = '🔴 DROWSY';
        statusBadgeText.style.color = '#ef4444';
    } else {
        statusCard.classList.remove('drowsy');
        statusBadge.style.borderColor = var(--primary-color);
        statusBadgeText.textContent = '🟢 ALERT';
        statusBadgeText.style.color = '#10b981';
    }
}

// Update Metrics Display
function updateMetrics(data) {
    // EAR
    updateMetric('ear', data.ear, THRESHOLDS.EAR);
    
    // MAR
    updateMetric('mar', data.mar, THRESHOLDS.MAR);
    
    // ML Probability
    document.getElementById('mlValue').textContent = data.ml_probability.toFixed(1) + '%';
    const mlBar = document.getElementById('mlBar');
    mlBar.style.width = Math.min(data.ml_probability, 100) + '%';
    
    // Drowsy Counter
    updateCounter(data.drowsy_counter);
}

// Helper function to update individual metrics
function updateMetric(type, value, threshold) {
    const valueElement = document.getElementById(type + 'Value');
    const barElement = document.getElementById(type + 'Bar');
    const statusElement = document.getElementById(type + 'Status');
    
    valueElement.textContent = value.toFixed(3);
    
    // Calculate percentage for bar (0-0.5 for EAR, 0-1.0 for MAR)
    const maxValue = type === 'ear' ? 0.5 : 1.0;
    const percentage = (value / maxValue) * 100;
    barElement.style.width = Math.min(percentage, 100) + '%';
    
    // Update status indicator
    let isThresholdExceeded = false;
    if (type === 'ear' && value < threshold) {
        isThresholdExceeded = true;
        statusElement.textContent = '⚠️ CLOSED';
        statusElement.className = 'danger';
        barElement.classList.add('danger');
    } else if (type === 'mar' && value > threshold) {
        isThresholdExceeded = true;
        statusElement.textContent = '⚠️ YAWNING';
        statusElement.className = 'danger';
        barElement.classList.add('danger');
    } else {
        statusElement.textContent = '✓ NORMAL';
        statusElement.className = '';
        barElement.classList.remove('danger', 'warning');
    }
}

// Update drowsy counter
function updateCounter(counter) {
    const counterValue = document.getElementById('counterValue');
    const counterBar = document.getElementById('counterBar');
    const counterDisplay = document.querySelector('.counter-display');
    
    counterValue.textContent = counter;
    const percentage = (counter / THRESHOLDS.DROWSY_COUNTER) * 100;
    counterBar.style.width = Math.min(percentage, 100) + '%';
    
    if (counter >= THRESHOLDS.DROWSY_COUNTER) {
        counterBar.classList.add('danger');
        counterDisplay.classList.add('alert');
    } else {
        counterBar.classList.remove('danger');
        counterDisplay.classList.remove('alert');
    }
}

// Update timestamp
function updateTimestamp(timestamp) {
    const date = new Date(timestamp);
    const timeString = date.toLocaleTimeString();
    document.getElementById('timestamp').textContent = 'Last updated: ' + timeString;
}

// Trigger alert
function triggerAlert(data) {
    const now = Date.now();
    
    // Check cooldown period
    if (now - lastAlertTime < alertCooldown) {
        return;
    }
    
    lastAlertTime = now;
    
    const alertSection = document.getElementById('alertSection');
    const alertMessage = document.getElementById('alertMessage');
    const alertTime = document.getElementById('alertTime');
    
    // Set alert message based on reason
    let message = 'Your eyes appear closed!';
    if (data.drowsy_reason.includes('YAWNING')) {
        message = 'You are yawning! Stay alert!';
    } else if (data.drowsy_reason.includes('ML')) {
        message = 'Drowsiness detected by AI model!';
    } else if (data.drowsy_reason.includes('EYES')) {
        message = 'Your eyes are closed!';
    }
    
    alertMessage.textContent = message;
    alertTime.textContent = new Date().toLocaleTimeString();
    
    // Show alert
    alertSection.style.display = 'flex';
    
    // Play sound
    playAlertSound();
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        if (alertSection.style.display === 'flex') {
            alertSection.style.display = 'none';
        }
    }, 3000);
}

// Acknowledge alert
function acknowledgeAlert() {
    document.getElementById('alertSection').style.display = 'none';
}

// Play alert sound
function playAlertSound() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // Beep pattern
    oscillator.frequency.value = 1000;
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.1);
    
    oscillator.start(audioContext.currentTime + 0.15);
    oscillator.stop(audioContext.currentTime + 0.25);
    
    oscillator.start(audioContext.currentTime + 0.3);
    oscillator.stop(audioContext.currentTime + 0.4);
}

// FPS calculation (simple counter)
let frameCount = 0;
let lastFpsTime = Date.now();

setInterval(() => {
    const now = Date.now();
    const elapsed = (now - lastFpsTime) / 1000;
    const fps = Math.round(frameCount / elapsed);
    document.getElementById('fpsDisplay').textContent = 'FPS: ' + fps;
    
    frameCount = 0;
    lastFpsTime = now;
}, 1000);

// Increment frame count on video load/update
document.getElementById('videoFeed').addEventListener('load', () => {
    frameCount++;
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        acknowledgeAlert();
    }
});

// Handle visibility change
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden');
    } else {
        console.log('Page visible');
    }
});
