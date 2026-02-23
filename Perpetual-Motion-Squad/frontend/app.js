/* ================================================================
   TerrainAI — Frontend Application Logic
   Premium Martian Editorial Interface

   Visual systems: loading screen, starfield, dust particles,
   shooting stars, cursor glow, card tilt, scroll-reveal.
   ALL features: upload preview, cinematic pipeline, before/after
   comparison slider, safety gauge, multi-model compare, uncertainty
   heatmap toggle, counter animations, inference history, GradCAM
   description, and all original API/ML logic preserved.
   ================================================================ */

const API = '';  // Same origin

// ============================================================================
// STATE
// ============================================================================
const state = {
    currentSection: 'analyze',
    currentFile: null,
    currentResult: null,
    gradcamData: null,
    distChart: null,
    safetyChart: null,
    qualityChart: null,
    inferenceHistory: [],
    uncertaintyVisible: false,
    // XAI state
    routePoints: [],
    whatIfResult: null,
    plannerStart: null,
    plannerEnd: null,
    plannerRoutes: null,
    radarChart: null,
    robustnessData: null,
    disagreementData: null,
};

// ============================================================================
// INIT
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUploadZones();
    initResultTabs();
    initComparisonSlider();
    initUncertaintyToggle();
    initMultiModelCompare();
    checkHealth();
    loadModels();
    
    // Fine-tuning features
    loadUserModelIfExists();
    initFineTuning();

    // Visual systems
    initLoadingScreen();
    generateStarfield();
    generateDustParticles();
    initShootingStars();
    initCursorGlow();
    initScrollReveal();
    initCardTilt();
    initCounterAnimations();

    // XAI features
    initXAITabs();
    initRouteShap();
    initWhatIf();
    initRobustnessTest();
    initRoutePlanner();
    initDisagreement();
    initMissionReport();
});

// ============================================================================
// LOADING SCREEN
// ============================================================================
function initLoadingScreen() {
    const screen = document.getElementById('loadingScreen');
    if (!screen) return;
    setTimeout(() => {
        screen.classList.add('dismissed');
        setTimeout(() => screen.remove(), 1000);
    }, 2200);
}

// ============================================================================
// COUNTER ANIMATIONS — Hero stats count up from 0
// ============================================================================
function initCounterAnimations() {
    document.querySelectorAll('.count-up').forEach(el => {
        const target = parseInt(el.dataset.target) || 0;
        animateCounter(el, 0, target, 600);
    });
}

function animateCounter(el, start, end, duration) {
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(start + (end - start) * eased);
        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            el.textContent = end;
            el.classList.add('count-done');
        }
    }
    requestAnimationFrame(update);
}

// ============================================================================
// STARFIELD
// ============================================================================
function generateStarfield() {
    const container = document.getElementById('starfield');
    if (!container) return;
    const motionOk = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const count = motionOk ? 150 : 60;
    const frag = document.createDocumentFragment();
    for (let i = 0; i < count; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        const size = Math.random() * 2 + 0.5;
        star.style.cssText = `left:${Math.random() * 100}%;top:${Math.random() * 100}%;width:${size}px;height:${size}px;opacity:${Math.random() * 0.5 + 0.1};animation-duration:${Math.random() * 5 + 3}s;animation-delay:${Math.random() * 5}s;`;
        frag.appendChild(star);
    }
    container.appendChild(frag);
}

// ============================================================================
// DUST PARTICLES
// ============================================================================
function generateDustParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 35; i++) {
        const dust = document.createElement('div');
        dust.className = 'dust';
        const size = Math.random() * 3 + 1;
        dust.style.cssText = `left:${Math.random() * 100}%;bottom:${-(Math.random() * 20)}%;width:${size}px;height:${size}px;animation-duration:${Math.random() * 20 + 15}s;animation-delay:${Math.random() * 15}s;`;
        frag.appendChild(dust);
    }
    container.appendChild(frag);
}

// ============================================================================
// SHOOTING STARS
// ============================================================================
function initShootingStars() {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    function spawnStar() {
        const star = document.createElement('div');
        star.className = 'shooting-star';
        star.style.top = `${Math.random() * 40}%`;
        star.style.left = `${Math.random() * 60}%`;
        star.style.transform = `rotate(${-25 - Math.random() * 20}deg)`;
        document.body.appendChild(star);
        setTimeout(() => star.remove(), 1400);
    }
    function schedule() {
        const delay = 4000 + Math.random() * 8000;
        setTimeout(() => { spawnStar(); schedule(); }, delay);
    }
    setTimeout(schedule, 3000);
}

// ============================================================================
// CURSOR GLOW
// ============================================================================
function initCursorGlow() {
    const glow = document.getElementById('cursorGlow');
    if (!glow) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { glow.style.display = 'none'; return; }
    document.addEventListener('mousemove', (e) => {
        glow.style.left = e.clientX + 'px';
        glow.style.top = e.clientY + 'px';
    }, { passive: true });
}

// ============================================================================
// SCROLL REVEAL
// ============================================================================
function initScrollReveal() {
    const els = document.querySelectorAll('.reveal, .reveal-stagger');
    if (!els.length) return;
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) { entry.target.classList.add('revealed'); observer.unobserve(entry.target); }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });
    els.forEach(el => observer.observe(el));
}

// ============================================================================
// CARD 3D TILT
// ============================================================================
function initCardTilt() {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    document.querySelectorAll('.tilt-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;
            const rotateX = ((y - rect.height / 2) / (rect.height / 2)) * -4;
            const rotateY = ((x - rect.width / 2) / (rect.width / 2)) * 4;
            card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(800px) rotateX(0) rotateY(0) translateY(0)';
        });
    });
}

// ============================================================================
// ML PIPELINE VISUALIZATION
// ============================================================================
function showMLPipeline() {
    const pipeline = document.getElementById('mlPipeline');
    if (!pipeline) return;
    pipeline.classList.remove('hidden');
    pipeline.querySelectorAll('.ml-step').forEach(step => {
        step.classList.remove('is-active', 'is-complete', 'is-error');
    });
}

function activateMLStep(stepIndex) {
    const pipeline = document.getElementById('mlPipeline');
    if (!pipeline) return;
    pipeline.querySelectorAll('.ml-step').forEach(step => {
        const idx = parseInt(step.dataset.step);
        if (idx < stepIndex) { step.classList.remove('is-active'); step.classList.add('is-complete'); }
        else if (idx === stepIndex) { step.classList.remove('is-complete', 'is-error'); step.classList.add('is-active'); }
    });
    moveRoverToStep(stepIndex);
}

function moveRoverToStep(stepIndex) {
    const rover = document.getElementById('pipelineRover');
    const pipeline = document.getElementById('mlPipeline');
    if (!rover || !pipeline) return;
    const steps = pipeline.querySelectorAll('.ml-step');
    if (!steps.length) return;
    rover.classList.add('visible', 'moving');
    const targetStep = steps[Math.min(stepIndex - 1, steps.length - 1)];
    const track = pipeline.querySelector('.ml-pipeline-track');
    if (!track || !targetStep) return;
    const trackRect = track.getBoundingClientRect();
    const stepRect = targetStep.getBoundingClientRect();
    const nodeEl = targetStep.querySelector('.ml-step-node');
    const nodeRect = nodeEl ? nodeEl.getBoundingClientRect() : stepRect;
    const centerX = nodeRect.left + nodeRect.width / 2 - trackRect.left;
    rover.style.left = `${centerX}px`;
    setTimeout(() => rover.classList.remove('moving'), 750);
}

function completeMLPipeline() {
    const pipeline = document.getElementById('mlPipeline');
    if (!pipeline) return;
    pipeline.querySelectorAll('.ml-step').forEach(step => {
        step.classList.remove('is-active', 'is-error'); step.classList.add('is-complete');
    });
    document.body.classList.remove('is-processing');
    const rover = document.getElementById('pipelineRover');
    if (rover) setTimeout(() => { rover.classList.remove('visible', 'moving'); }, 1200);
}

function failMLPipeline() {
    const pipeline = document.getElementById('mlPipeline');
    if (!pipeline) return;
    pipeline.querySelectorAll('.ml-step').forEach(step => {
        if (step.classList.contains('is-active')) { step.classList.remove('is-active'); step.classList.add('is-error'); }
    });
    document.body.classList.remove('is-processing');
}

// ============================================================================
// NAVIGATION
// ============================================================================
function initNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => switchSection(btn.dataset.section));
    });
}

function switchSection(name) {
    state.currentSection = name;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-section="${name}"]`).classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById(`section-${name}`).classList.add('active');
}

// ============================================================================
// HEALTH CHECK
// ============================================================================
async function checkHealth() {
    try {
        const res = await fetch(`${API}/api/health`);
        const data = await res.json();
        document.getElementById('statusDot').classList.add('connected');
        document.getElementById('statusText').textContent = `${data.gpu || 'CPU'} · ${data.active_model || 'Loading...'}`;
        document.getElementById('statDevice').textContent = data.gpu ? 'GPU' : 'CPU';
        const modelNames = {
            mit_b3: 'FPN + MiT-B3',
            mit_b1: 'DeepLabV3+ + EfficientNet-B4',
            mit_b0: 'Linknet + MobileNetV2',
        };
        document.getElementById('statModel').textContent = modelNames[data.active_model] || data.active_model || '—';
    } catch (e) {
        document.getElementById('statusText').textContent = 'Server offline';
    }
}

// ============================================================================
// UPLOAD ZONES — With Image Preview Before Submission
// ============================================================================
function initUploadZones() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('imageInput');
    const defaultUI = document.getElementById('uploadDefault');
    const previewUI = document.getElementById('uploadPreview');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnChange = document.getElementById('btnChange');

    zone.addEventListener('click', (e) => {
        if (e.target.closest('.btn-analyze') || e.target.closest('.btn-change')) return;
        if (!previewUI.classList.contains('hidden')) return;
        input.click();
    });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) showImagePreview(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files[0]) showImagePreview(input.files[0]); });

    btnAnalyze.addEventListener('click', (e) => {
        e.stopPropagation();
        if (state.currentFile) handleImageUpload(state.currentFile);
    });
    btnChange.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploadZone();
        input.click();
    });

    // Video upload
    const vzone = document.getElementById('videoUploadZone');
    const vinput = document.getElementById('videoInput');
    vzone.addEventListener('click', () => vinput.click());
    vzone.addEventListener('dragover', e => { e.preventDefault(); vzone.classList.add('dragover'); });
    vzone.addEventListener('dragleave', () => vzone.classList.remove('dragover'));
    vzone.addEventListener('drop', e => {
        e.preventDefault(); vzone.classList.remove('dragover');
        if (e.dataTransfer.files[0]) handleVideoUpload(e.dataTransfer.files[0]);
    });
    vinput.addEventListener('change', () => { if (vinput.files[0]) handleVideoUpload(vinput.files[0]); });
}

function showImagePreview(file) {
    state.currentFile = file;
    const defaultUI = document.getElementById('uploadDefault');
    const previewUI = document.getElementById('uploadPreview');
    const previewImg = document.getElementById('previewImage');
    const previewName = document.getElementById('previewName');
    const previewDims = document.getElementById('previewDims');

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewName.textContent = file.name;
        const img = new Image();
        img.onload = () => { previewDims.textContent = `${img.width} × ${img.height}`; };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    defaultUI.classList.add('hidden');
    previewUI.classList.remove('hidden');
}

function resetUploadZone() {
    document.getElementById('uploadDefault').classList.remove('hidden');
    document.getElementById('uploadPreview').classList.add('hidden');
    state.currentFile = null;
}

// ============================================================================
// IMAGE SEGMENTATION
// ============================================================================
async function handleImageUpload(file) {
    state.currentFile = file;
    state.gradcamData = null;

    document.body.classList.add('is-processing');
    showMLPipeline();
    activateMLStep(1);

    const pipelineSection = document.getElementById('pipelineSection');
    const resultsSection = document.getElementById('resultsSection');
    pipelineSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const track = document.getElementById('pipelineTrack');
    track.innerHTML = `<div class="film-step is-active" style="flex:0 0 100%;opacity:1;transform:none;padding:40px;text-align:center;">
        <div class="scan-line"></div>
        <div class="spinner" style="margin:0 auto 12px;"></div>
        <div class="film-step-name">Analyzing image & applying DIP pipeline...</div>
    </div>`;

    setTimeout(() => activateMLStep(2), 500);

    const form = new FormData();
    form.append('file', file);

    setTimeout(() => activateMLStep(3), 1000);

    try {
        const res = await fetch(`${API}/api/segment`, { method: 'POST', body: form });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();
        state.currentResult = data;

        activateMLStep(4);
        renderCinematicPipeline(data.preprocessing_steps);
        renderResults(data);

        setTimeout(() => completeMLPipeline(), 400);
        setTimeout(() => initCardTilt(), 500);

        fetchRecommendation(file);
    } catch (err) {
        failMLPipeline();
        track.innerHTML = `<div class="film-step is-active" style="flex:0 0 100%;opacity:1;transform:none;border-color:var(--danger);padding:30px;text-align:center;">
            <div class="film-step-name" style="color:var(--danger)">Error: ${err.message}</div>
        </div>`;
    }
}

// ============================================================================
// CINEMATIC PIPELINE STRIP — Film Reel with Sequential Reveal
// ============================================================================
function renderCinematicPipeline(steps) {
    const track = document.getElementById('pipelineTrack');
    track.innerHTML = '';

    steps.forEach((step, idx) => {
        if (idx > 0) {
            const connector = document.createElement('div');
            connector.className = 'film-connector';
            connector.innerHTML = '→';
            track.appendChild(connector);
        }

        const el = document.createElement('div');
        el.className = 'film-step';
        el.style.animationDelay = `${idx * 0.15}s`;
        el.innerHTML = `
            <div class="scan-line"></div>
            <img class="film-step-img revealing" src="data:image/jpeg;base64,${step.image}" alt="${step.name}" style="animation-delay:${idx * 0.2}s;">
            <div class="film-step-info">
                <div class="film-step-num">Step ${idx + 1}</div>
                <div class="film-step-name">${step.name}</div>
                <div class="film-step-desc">${step.description}</div>
            </div>
        `;
        track.appendChild(el);

        // Sequential active glow
        setTimeout(() => {
            el.classList.add('is-active');
            setTimeout(() => el.classList.remove('is-active'), 1200);
        }, idx * 400);
    });

    // Final card: segmented result
    setTimeout(() => {
        const connector = document.createElement('div');
        connector.className = 'film-connector';
        connector.innerHTML = '→';
        track.appendChild(connector);

        const final = document.createElement('div');
        final.className = 'film-step';
        final.style.animationDelay = `${steps.length * 0.15}s`;
        final.innerHTML = `
            <div class="scan-line"></div>
            ${state.currentResult ? `<img class="film-step-img revealing" src="data:image/jpeg;base64,${state.currentResult.mask}" alt="Result" style="animation-delay:${steps.length * 0.2}s;">` : ''}
            <div class="film-step-info">
                <div class="film-step-num">Final</div>
                <div class="film-step-name">Segmented!</div>
                <div class="film-step-desc">Semantic mask generated by MiT-B3 transformer</div>
            </div>
        `;
        track.appendChild(final);
        setTimeout(() => final.classList.add('is-active'), 200);
    }, steps.length * 200);
}

// ============================================================================
// RESULTS RENDERING
// ============================================================================
function renderResults(data) {
    const section = document.getElementById('resultsSection');
    section.classList.remove('hidden');

    // Comparison slider images
    const compOriginal = document.getElementById('compareOriginal');
    const compAfter = document.getElementById('compareAfter');
    if (compOriginal) compOriginal.src = `data:image/jpeg;base64,${data.original}`;
    if (compAfter) compAfter.src = `data:image/jpeg;base64,${data.overlay}`;

    // Standard view images
    const resultOrig = document.getElementById('resultOriginal');
    const resultOut = document.getElementById('resultOutput');
    if (resultOrig) resultOrig.src = `data:image/jpeg;base64,${data.original}`;
    if (resultOut) resultOut.src = `data:image/jpeg;base64,${data.overlay}`;

    // Inference badge + history
    const badge = document.getElementById('inferenceBadge');
    const conf = data.confidence_score_pct ?? 0;
    const acc = data.accuracy_estimate_pct ?? 0;
    badge.textContent = `${data.inference_time_ms}ms · Conf ${conf}% · Acc(est) ${acc}%`;
    updateInferenceHistory(data.inference_time_ms, data.model_used);

    // Safety gauge
    if (data.safety_percentages) {
        renderSafetyGauge(data.safety_percentages);
        renderSafetyChart(data.safety_percentages);
    }

    // Quality metrics chart + meta
    renderQualityChart(data);
    updateQualityMeta(data);

    // Class legend
    const legend = document.getElementById('classLegend');
    legend.innerHTML = data.class_distribution.map(c =>
        `<div class="legend-item">
            <div class="legend-color" style="background:rgb(${c.color.join(',')})"></div>
            <span>${c.percentage}%</span>
        </div>`
    ).join('');

    // Chart
    renderDistributionChart(data.class_distribution);

    // Show compare view by default
    switchResultView('compare');

    // Show XAI features container
    const xaiFeatures = document.getElementById('xaiFeatures');
    if (xaiFeatures) {
        xaiFeatures.classList.remove('hidden');
        // Prime the route SHAP and planner base images with the safety map
        const routeBase = document.getElementById('routeShapBaseImg');
        if (routeBase) routeBase.src = `data:image/jpeg;base64,${data.safety}`;
        const plannerBase = document.getElementById('plannerBaseImg');
        if (plannerBase) plannerBase.src = `data:image/jpeg;base64,${data.safety}`;
        // Reset XAI state
        state.routePoints = [];
        state.plannerStart = null;
        state.plannerEnd = null;
        state.plannerRoutes = null;
        state.disagreementData = null;
        state.robustnessData = null;
        // Auto-analyze failure modes
        if (data.confidence_grid) analyzeAndRenderFailureModes(data.confidence_grid);
        // Reset whatif canvas
        setTimeout(() => primeWhatIfCanvas(), 300);
    }

    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v && v.trim()) ? v.trim() : fallback;
}

function renderSafetyGauge(pcts) {
    setTimeout(() => {
        document.getElementById('safeBarFill').style.width = `${pcts.safe}%`;
        document.getElementById('safePct').textContent = `${pcts.safe}%`;
        document.getElementById('cautionBarFill').style.width = `${pcts.caution}%`;
        document.getElementById('cautionPct').textContent = `${pcts.caution}%`;
        document.getElementById('obstacleBarFill').style.width = `${pcts.obstacle}%`;
        document.getElementById('obstaclePct').textContent = `${pcts.obstacle}%`;
    }, 200);
}

function updateInferenceHistory(timeMs, modelUsed) {
    state.inferenceHistory.push({ time: timeMs, model: modelUsed });
    if (state.inferenceHistory.length > 5) state.inferenceHistory.shift();

    const historyEl = document.getElementById('inferenceHistory');
    if (state.inferenceHistory.length >= 2) {
        const prev = state.inferenceHistory[state.inferenceHistory.length - 2];
        const ratio = prev.time / timeMs;
        if (ratio > 1.1) {
            historyEl.innerHTML = `<span class="inference-faster">↑ ${ratio.toFixed(1)}× faster vs ${prev.model.replace('mit_', 'MiT-')}</span>`;
        } else if (ratio < 0.9) {
            historyEl.innerHTML = `<span class="inference-slower">↓ ${(1 / ratio).toFixed(1)}× slower vs ${prev.model.replace('mit_', 'MiT-')}</span>`;
        } else {
            historyEl.textContent = '';
        }
    }
}

function renderDistributionChart(distribution) {
    const ctx = document.getElementById('distChart').getContext('2d');
    if (state.distChart) state.distChart.destroy();
    state.distChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: distribution.map(c => `${c.percentage}%`),
            datasets: [{
                data: distribution.map(c => c.percentage),
                backgroundColor: distribution.map(c => `rgb(${c.color.join(',')})`),
                borderColor: 'rgba(8,8,8,0.9)', borderWidth: 2
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { animateRotate: true, duration: 1200 },
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: 'rgba(212,165,116,0.55)', font: { size: 11, family: 'Inter' }, padding: 8 }
                }
            }
        }
    });
}

function renderSafetyChart(pcts) {
    const canvas = document.getElementById('safetyChart');
    if (!canvas || typeof Chart === 'undefined') return;
    const ctx = canvas.getContext('2d');
    if (state.safetyChart) state.safetyChart.destroy();

    const safe = Number(pcts.safe ?? 0);
    const caution = Number(pcts.caution ?? 0);
    const obstacle = Number(pcts.obstacle ?? 0);
    const neutral = Math.max(0, 100 - safe - caution - obstacle);

    state.safetyChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Safe', 'Caution', 'Obstacle', 'Neutral'],
            datasets: [{
                data: [safe, caution, obstacle, neutral],
                backgroundColor: [
                    cssVar('--safe', '#4ade80'),
                    cssVar('--caution', '#fbbf24'),
                    cssVar('--danger', '#ef4444'),
                    'rgba(212,165,116,0.18)',
                ],
                borderColor: 'rgba(8,8,8,0.9)',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '62%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: 'rgba(212,165,116,0.55)',
                        font: { size: 11, family: 'Inter' },
                        padding: 8,
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${ctx.parsed}%`
                    }
                }
            }
        }
    });
}

function renderQualityChart(data) {
    const canvas = document.getElementById('qualityChart');
    if (!canvas || typeof Chart === 'undefined') return;
    const ctx = canvas.getContext('2d');
    if (state.qualityChart) state.qualityChart.destroy();

    const conf = Number(data.confidence_score_pct ?? 0);
    const high = Number(data.high_confidence_pixels_pct ?? 0);
    const acc = Number(data.accuracy_estimate_pct ?? 0);

    const accent = cssVar('--mars-400', '#e8622d');

    state.qualityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Confidence', 'High-Conf Pixels', 'Acc (est)'],
            datasets: [{
                label: 'Percent',
                data: [conf, high, acc],
                backgroundColor: [
                    'rgba(232,98,45,0.35)',
                    'rgba(212,165,116,0.28)',
                    'rgba(74,222,128,0.22)'
                ],
                borderColor: [accent, 'rgba(212,165,116,0.55)', cssVar('--safe', '#4ade80')],
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { color: 'rgba(212,165,116,0.55)', font: { size: 11, family: 'Inter' } },
                    grid: { color: 'rgba(200,130,80,0.06)' }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: 'rgba(212,165,116,0.55)', callback: (v) => `${v}%` },
                    grid: { color: 'rgba(200,130,80,0.06)' }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.y}%`
                    }
                }
            }
        }
    });
}

function updateQualityMeta(data) {
    const modelEl = document.getElementById('qmModel');
    const inputEl = document.getElementById('qmInput');
    const timeEl = document.getElementById('qmTime');
    if (modelEl) modelEl.textContent = (data.model_used || '—').toUpperCase();
    if (inputEl) inputEl.textContent = data.input_size || '—';
    if (timeEl) timeEl.textContent = (data.inference_time_ms != null) ? `${data.inference_time_ms} ms` : '—';
}

// ============================================================================
// BEFORE/AFTER COMPARISON SLIDER
// ============================================================================
function initComparisonSlider() {
    const slider = document.getElementById('comparisonSlider');
    const handle = document.getElementById('compareHandle');
    const afterWrap = document.getElementById('compareAfterWrap');
    if (!slider || !handle || !afterWrap) return;

    let isDragging = false;

    function updateSlider(x) {
        const rect = slider.getBoundingClientRect();
        let pos = (x - rect.left) / rect.width;
        pos = Math.max(0.05, Math.min(0.95, pos));
        afterWrap.style.width = `${(1 - pos) * 100}%`;
        handle.style.left = `${pos * 100}%`;
    }

    slider.addEventListener('mousedown', (e) => { isDragging = true; updateSlider(e.clientX); });
    document.addEventListener('mousemove', (e) => { if (isDragging) updateSlider(e.clientX); });
    document.addEventListener('mouseup', () => { isDragging = false; });

    slider.addEventListener('touchstart', (e) => { isDragging = true; updateSlider(e.touches[0].clientX); }, { passive: true });
    document.addEventListener('touchmove', (e) => { if (isDragging) updateSlider(e.touches[0].clientX); }, { passive: true });
    document.addEventListener('touchend', () => { isDragging = false; });
}

// ============================================================================
// RESULT TABS
// ============================================================================
function initResultTabs() {
    document.querySelectorAll('.result-tab[data-view]').forEach(tab => {
        tab.addEventListener('click', () => switchResultView(tab.dataset.view));
    });
}

function switchResultView(view) {
    document.querySelectorAll('.result-tab[data-view]').forEach(t => t.classList.remove('active'));
    const activeTab = document.querySelector(`[data-view="${view}"]`);
    if (activeTab) activeTab.classList.add('active');

    const compWrap = document.getElementById('comparisonWrap');
    const origCard = document.getElementById('resultOriginalCard');
    const outCard = document.getElementById('resultOutputCard');
    const label = document.getElementById('resultViewLabel');
    const img = document.getElementById('resultOutput');
    const data = state.currentResult;
    if (!data) return;

    const loadingEl = document.getElementById('gradcamLoading');
    const gradcamCaption = document.getElementById('gradcamCaption');

    if (view === 'compare') {
        compWrap.style.display = 'block';
        origCard.classList.add('hidden');
        outCard.classList.add('hidden');
        gradcamCaption.classList.add('hidden');
        return;
    }

    compWrap.style.display = 'none';
    origCard.classList.remove('hidden');
    outCard.classList.remove('hidden');

    switch (view) {
        case 'overlay':
            label.textContent = 'Segmented Overlay';
            img.src = `data:image/jpeg;base64,${data.overlay}`;
            img.classList.add('sweep-reveal');
            setTimeout(() => img.classList.remove('sweep-reveal'), 900);
            loadingEl.classList.add('hidden');
            gradcamCaption.classList.add('hidden');
            break;
        case 'mask':
            label.textContent = 'Segmentation Mask';
            img.src = `data:image/jpeg;base64,${data.mask}`;
            loadingEl.classList.add('hidden');
            gradcamCaption.classList.add('hidden');
            break;
        case 'safety':
            label.textContent = 'Traversability Heatmap';
            img.src = `data:image/jpeg;base64,${data.safety}`;
            loadingEl.classList.add('hidden');
            gradcamCaption.classList.add('hidden');
            break;
        case 'gradcam':
            label.textContent = 'GradCAM Explainability';
            if (state.gradcamData) {
                img.src = `data:image/jpeg;base64,${state.gradcamData.gradcam}`;
                loadingEl.classList.add('hidden');
                if (state.gradcamData.description) {
                    gradcamCaption.textContent = state.gradcamData.description;
                    gradcamCaption.classList.remove('hidden');
                }
            } else {
                loadingEl.classList.remove('hidden');
                gradcamCaption.classList.add('hidden');
                fetchGradCAM();
            }
            break;
        case 'confidence':
            label.textContent = 'Prediction Confidence';
            if (data.confidence) img.src = `data:image/jpeg;base64,${data.confidence}`;
            loadingEl.classList.add('hidden');
            gradcamCaption.classList.add('hidden');
            break;
    }
}

async function fetchGradCAM() {
    if (!state.currentFile) return;
    const form = new FormData();
    form.append('file', state.currentFile);
    try {
        const res = await fetch(`${API}/api/explain`, { method: 'POST', body: form });
        const data = await res.json();
        state.gradcamData = data;
        document.getElementById('resultOutput').src = `data:image/jpeg;base64,${data.gradcam}`;
        document.getElementById('gradcamLoading').classList.add('hidden');
        document.getElementById('resultViewLabel').textContent = `GradCAM — Focus: ${data.target_class_name}`;
        if (data.description) {
            const caption = document.getElementById('gradcamCaption');
            caption.textContent = data.description;
            caption.classList.remove('hidden');
        }
    } catch (err) {
        document.getElementById('gradcamLoading').innerHTML = `<span style="color:var(--danger)">GradCAM failed: ${err.message}</span>`;
    }
}

// ============================================================================
// UNCERTAINTY HEATMAP TOGGLE
// ============================================================================
function initUncertaintyToggle() {
    const btn = document.getElementById('uncertaintyToggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
        state.uncertaintyVisible = !state.uncertaintyVisible;
        btn.classList.toggle('active', state.uncertaintyVisible);
        btn.textContent = state.uncertaintyVisible ? 'Hide Uncertainty Map' : 'Show Uncertainty Map';
        const canvas = document.getElementById('confidenceCanvas');
        if (!canvas) return;
        if (state.uncertaintyVisible && state.currentResult && state.currentResult.confidence_grid) {
            renderUncertaintyOverlay(canvas, state.currentResult.confidence_grid);
            canvas.classList.remove('hidden');
        } else {
            canvas.classList.add('hidden');
        }
    });
}

function renderUncertaintyOverlay(canvas, grid) {
    canvas.width = grid.grid_w;
    canvas.height = grid.grid_h;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(grid.grid_w, grid.grid_h);
    for (let i = 0; i < grid.cells.length; i++) {
        const cell = grid.cells[i];
        const conf = cell.p / 100;
        const uncertainty = 1 - conf;
        const r = Math.round(255 * uncertainty);
        const g = Math.round(80 * (1 - uncertainty));
        const b = Math.round(20 * (1 - uncertainty));
        const a = Math.round(180 * uncertainty);
        imgData.data[i * 4] = r;
        imgData.data[i * 4 + 1] = g;
        imgData.data[i * 4 + 2] = b;
        imgData.data[i * 4 + 3] = a;
    }
    ctx.putImageData(imgData, 0, 0);
}

// ============================================================================
// MODEL RECOMMENDATION
// ============================================================================
async function fetchRecommendation(file) {
    const form = new FormData();
    form.append('file', file);
    try {
        const res = await fetch(`${API}/api/recommend-model`, { method: 'POST', body: form });
        const data = await res.json();
        renderRecommendation(data);
    } catch (err) {
        document.getElementById('recommendationPanel').innerHTML = `<p style="color:var(--text-3)">Could not analyze image</p>`;
    }
}

function renderRecommendation(data) {
    const panel = document.getElementById('recommendationPanel');
    const maxScore = Math.max(...Object.values(data.scores), 1);
    const modelNames = {
        mit_b3: 'FPN + MiT-B3',
        mit_b1: 'DeepLabV3+ + EfficientNet-B4',
        mit_b0: 'Linknet + MobileNetV2',
    };
    const scoreBars = Object.entries(data.scores).map(([key, score]) => {
        const name = modelNames[key] || key;
        const pct = Math.round((score / maxScore) * 100);
        const isWinner = key === data.recommended;
        return `<div class="model-score-bar">
            <span class="model-score-label" style="${isWinner ? 'color:var(--mars-400);font-weight:600;' : ''}">${name}</span>
            <div class="model-score-track"><div class="model-score-fill accuracy" style="width:${pct}%"></div></div>
            <span style="font-size:11px;color:var(--text-3);font-family:'JetBrains Mono',monospace;">${score}pts</span>
        </div>`;
    }).join('');

    const narrative = generateModelNarrative(data);

    panel.innerHTML = `
        <div class="rec-header">
            <span class="rec-model-badge">${data.recommended_name}</span>
            <span class="rec-confidence">Confidence: ${Math.round(data.confidence * 100)}%</span>
        </div>
        <ul class="rec-reasons">${data.reasons.map(r => `<li>${r}</li>`).join('')}</ul>
        <div style="margin-top:14px;margin-bottom:10px;font-size:12px;font-weight:500;letter-spacing:1px;text-transform:uppercase;color:var(--text-2);">Scene Analysis</div>
        <div class="rec-radar-wrap"><canvas id="radarChart"></canvas></div>
        <div class="rec-narrative">${narrative}</div>
        <div style="margin-top:14px;margin-bottom:10px;font-size:12px;font-weight:500;letter-spacing:1px;text-transform:uppercase;color:var(--text-2);">Model Scores</div>
        <div class="model-score-bars">${scoreBars}</div>
        <div class="rec-analysis">
            <div class="rec-stat">Color var: <strong>${data.analysis.color_variance}</strong></div>
            <div class="rec-stat">Saturation: <strong>${data.analysis.saturation}</strong></div>
            <div class="rec-stat">Edge density: <strong>${data.analysis.edge_density}</strong></div>
            <div class="rec-stat">Contrast: <strong>${data.analysis.contrast}</strong></div>
            <div class="rec-stat">Resolution: <strong>${data.analysis.resolution}</strong></div>
            <div class="rec-stat">Megapixels: <strong>${data.analysis.megapixels}</strong></div>
            <div class="rec-stat">Brightness var: <strong>${data.analysis.brightness_variance}</strong></div>
        </div>
    `;

    // Render radar chart
    setTimeout(() => renderRadarChart(data), 50);
}

function generateModelNarrative(data) {
    const modelNames = {
        mit_b3: 'FPN + MiT-B3',
        mit_b1: 'DeepLabV3+ + EfficientNet-B4',
        mit_b0: 'Linknet + MobileNetV2',
    };
    const model = modelNames[data.recommended] || data.recommended_name;
    const conf = Math.round(data.confidence * 100);
    const topReason = data.reasons[0] || 'Scene complexity analyzed.';
    const edgeLvl = data.analysis.edge_density > 0.15 ? 'high' : data.analysis.edge_density > 0.08 ? 'moderate' : 'low';
    const colorLvl = data.analysis.color_variance > 40 ? 'high' : data.analysis.color_variance > 20 ? 'moderate' : 'low';
    return `<strong>${model}</strong> recommended with <strong>${conf}% confidence</strong>. ${topReason}. The scene exhibits ${edgeLvl} boundary complexity and ${colorLvl} color variance, making this model's capacity the optimal balance for safe UGV terrain navigation.`;
}

function renderRadarChart(data) {
    const ctx = document.getElementById('radarChart');
    if (!ctx) return;
    if (state.radarChart) { state.radarChart.destroy(); state.radarChart = null; }
    const a = data.analysis;
    const vals = [
        Math.min(parseFloat(a.color_variance) / 60 * 100, 100),
        Math.min(parseFloat(a.edge_density) / 0.25 * 100, 100),
        Math.min(parseFloat(a.contrast) / 80 * 100, 100),
        Math.min(parseFloat(a.megapixels) / 4 * 100, 100),
        Math.min(parseFloat(a.saturation) / 200 * 100, 100),
    ];
    state.radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Color Variance', 'Edge Complexity', 'Contrast', 'Resolution', 'Saturation'],
            datasets: [{
                label: 'Scene Metrics',
                data: vals,
                backgroundColor: 'rgba(232,98,45,0.08)',
                borderColor: 'rgba(232,98,45,0.5)',
                borderWidth: 1.5,
                pointBackgroundColor: 'rgba(232,98,45,0.7)',
                pointRadius: 3,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0, max: 100,
                    grid: { color: 'rgba(180,140,100,0.06)' },
                    angleLines: { color: 'rgba(180,140,100,0.08)' },
                    ticks: { display: false },
                    pointLabels: { color: 'rgba(212,165,116,0.5)', font: { size: 10, family: 'Inter' } },
                }
            },
            plugins: { legend: { display: false } },
            animation: { duration: 1000 },
        },
    });
}

// ============================================================================
// MULTI-MODEL COMPARISON
// ============================================================================
function initMultiModelCompare() {
    const btn = document.getElementById('btnCompareModels');
    if (!btn) return;
    btn.addEventListener('click', () => {
        if (!state.currentFile) { alert('Upload an image first.'); return; }
        runModelComparison(state.currentFile);
    });
}

async function runModelComparison(file) {
    const btn = document.getElementById('btnCompareModels');
    const grid = document.getElementById('compareModelsGrid');
    btn.classList.add('loading');
    btn.textContent = 'Comparing...';
    grid.classList.remove('hidden');
    grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:30px;"><div class="spinner" style="margin:0 auto 12px;"></div><div style="font-size:13px;color:var(--text-3);">Running inference on all 3 models...</div></div>`;

    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch(`${API}/api/compare-models`, { method: 'POST', body: form });
        const data = await res.json();
        renderModelComparison(data.results);
    } catch (err) {
        grid.innerHTML = `<div style="grid-column:1/-1;color:var(--danger);padding:20px;">Comparison failed: ${err.message}</div>`;
    }
    btn.classList.remove('loading');
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2" ry="2"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/></svg> Compare All Models`;
}

function renderModelComparison(results) {
    const grid = document.getElementById('compareModelsGrid');
    const names = {
        'mit_b0': 'Linknet + MobileNetV2',
        'mit_b1': 'DeepLabV3+ + EfficientNet-B4',
        'mit_b3': 'FPN + MiT-B3'
    };
    grid.innerHTML = Object.entries(results).map(([key, r]) => {
        if (r.error) return `<div class="compare-model-card"><div class="compare-model-info"><div class="compare-model-name">${names[key]}</div><div style="color:var(--danger);font-size:12px;">Error: ${r.error}</div></div></div>`;
        return `<div class="compare-model-card">
            <img src="data:image/jpeg;base64,${r.overlay}" alt="${key}">
            <div class="compare-model-info">
                <div class="compare-model-name">${names[key]}</div>
                <div class="compare-model-time">${r.inference_time_ms}ms inference</div>
                <div style="font-size:11px;color:var(--text-3);margin-top:6px;">
                    Safe: ${r.safety_percentages.safe}% · Caution: ${r.safety_percentages.caution}% · Obstacle: ${r.safety_percentages.obstacle}%
                </div>
            </div>
        </div>`;
    }).join('');
}

// ============================================================================
// VIDEO PROCESSING
// ============================================================================
async function handleVideoUpload(file) {
    const pipeline = document.getElementById('videoPipeline');
    const results = document.getElementById('videoResults');
    pipeline.classList.remove('hidden');
    results.classList.add('hidden');

    const fill = document.getElementById('videoProgressFill');
    const text = document.getElementById('videoProgressText');

    fill.style.width = '0%';
    fill.style.background = 'var(--grad-warm)';
    text.textContent = 'Uploading video...';
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 3, 90);
        fill.style.width = `${progress}%`;
        if (progress < 20) text.textContent = 'Extracting frames at 200ms intervals...';
        else if (progress < 40) text.textContent = 'Running DIP preprocessing on each frame...';
        else if (progress < 60) text.textContent = 'Segmenting frames with adaptive model switching...';
        else if (progress < 75) text.textContent = 'Generating GradCAM heatmaps...';
        else text.textContent = 'Stitching output videos...';
    }, 800);

    const form = new FormData();
    form.append('file', file);
    form.append('interval_ms', '200');

    try {
        const res = await fetch(`${API}/api/segment-video`, { method: 'POST', body: form });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        clearInterval(progressInterval);
        fill.style.width = '100%';
        text.textContent = 'Complete!';

        state.videoData = data;
        results.classList.remove('hidden');
        document.getElementById('segmentedVideo').src = data.videos.sidebyside;

        const stats = document.getElementById('videoStats');
        stats.innerHTML = `
            <div class="video-stat">${data.frames_processed} frames</div>
            <div class="video-stat">${data.interval_ms}ms interval</div>
            <div class="video-stat">${data.output_fps} FPS output</div>
            <div class="video-stat">${data.avg_fps} FPS throughput</div>
            <div class="video-stat">${data.total_inference_ms}ms total</div>
            <div class="video-stat">${data.avg_inference_per_frame_ms}ms/frame</div>
            <div class="video-stat">Confidence: ${data.confidence_score_pct ?? 0}%</div>
            <div class="video-stat">Accuracy(est): ${data.accuracy_estimate_pct ?? 0}%</div>
            <div class="video-stat">${data.video_info.width}x${data.video_info.height}</div>
            <div class="video-stat">${data.video_info.duration_s}s duration</div>
        `;

        // ── Velocity-Adaptive Dashboard ──────────────────
        renderVelocityDashboard(data);

        const previews = document.getElementById('videoPreviews');
        previews.innerHTML = Object.entries(data.previews).map(([key, b64]) => `
            <div class="result-card">
                <div class="result-label">${key.charAt(0).toUpperCase() + key.slice(1)}</div>
                <img src="data:image/jpeg;base64,${b64}" class="result-img" alt="${key}">
            </div>
        `).join('');

        document.querySelectorAll('#videoTabs .result-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const videoType = tab.dataset.video;
                document.querySelectorAll('#videoTabs .result-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const player = document.getElementById('segmentedVideo');
                player.src = state.videoData.videos[videoType];
                player.load();
            });
        });

        results.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        clearInterval(progressInterval);
        fill.style.width = '100%';
        fill.style.background = 'var(--danger)';
        text.textContent = `Error: ${err.message}`;
    }
}


// ============================================================================
// VELOCITY-ADAPTIVE DASHBOARD
// ============================================================================

function renderVelocityDashboard(data) {
    const dashboard = document.getElementById('velocityDashboard');
    if (!data.velocity || !dashboard) return;
    dashboard.style.display = '';

    const v = data.velocity;
    const perFrame = v.per_frame || [];

    // ── Summary cards ──
    const modelNames = {
        mit_b3: 'FPN + MiT-B3',
        mit_b1: 'DeepLabV3+ + EfficientNet-B4',
        mit_b0: 'Linknet + MobileNetV2'
    };
    const tierClass = { accurate: 'accurate', balanced: 'balanced', fast: 'fast' };

    let usageHtml = '';
    for (const [mk, count] of Object.entries(v.model_usage)) {
        const pct = ((count / data.frames_processed) * 100).toFixed(0);
        const tier = perFrame.find(f => f.model_key === mk)?.model_tier || 'balanced';
        const displayName = perFrame.find(f => f.model_key === mk)?.model_name || modelNames[mk] || mk;
        usageHtml += `<span class="vel-model-tag ${tierClass[tier] || ''}">${displayName}: ${count} (${pct}%)</span> `;
    }

    document.getElementById('velocitySummary').innerHTML = `
        <div class="vel-stat"><span class="vel-label">Avg Velocity</span><span class="vel-value">${v.avg_velocity} px/s</span></div>
        <div class="vel-stat"><span class="vel-label">Max Velocity</span><span class="vel-value">${v.max_velocity} px/s</span></div>
        <div class="vel-stat"><span class="vel-label">Min Velocity</span><span class="vel-value">${v.min_velocity} px/s</span></div>
        <div class="vel-stat"><span class="vel-label">Confidence</span><span class="vel-value">${data.confidence_score_pct ?? 0}%</span></div>
        <div class="vel-stat"><span class="vel-label">Accuracy (est.)</span><span class="vel-value">${data.accuracy_estimate_pct ?? 0}%</span></div>
        <div class="vel-stat" style="flex:1 1 auto;"><span class="vel-label">Model Usage</span><span class="vel-value">${usageHtml}</span></div>
    `;

    // ── Canvas bar chart ──
    renderVelocityChart(perFrame, v.thresholds);

    // ── Per-frame table ──
    let rows = perFrame.map(f => {
        const tier = f.model_tier || 'balanced';
        return `<tr>
            <td>${f.frame_index}</td>
            <td>${f.velocity}</td>
            <td>${f.displacement_px}</td>
            <td>${f.tracked_points}</td>
            <td><span class="vel-model-tag ${tierClass[tier]}">${f.model_name || modelNames[f.model_key] || f.model_key}</span></td>
            <td>${f.confidence_score_pct ?? 0}%</td>
            <td>${f.accuracy_estimate_pct ?? 0}%</td>
            <td>${f.inference_ms} ms</td>
        </tr>`;
    }).join('');

    document.getElementById('velocityTable').innerHTML = `
        <table>
            <thead><tr><th>#</th><th>Velocity (px/s)</th><th>Displacement</th><th>Points</th><th>Model</th><th>Confidence</th><th>Accuracy (est.)</th><th>Inference</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
}


function renderVelocityChart(perFrame, thresholds) {
    const canvas = document.getElementById('velocityChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 160 * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = '160px';
    ctx.scale(dpr, dpr);

    const W = rect.width, H = 160;
    const pad = { top: 16, right: 12, bottom: 24, left: 42 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const velocities = perFrame.map(f => f.velocity);
    const maxV = Math.max(...velocities, thresholds.high * 1.2, 50);

    // Background
    ctx.fillStyle = 'transparent';
    ctx.fillRect(0, 0, W, H);

    // Threshold zones
    const yLow = pad.top + plotH * (1 - thresholds.low / maxV);
    const yHigh = pad.top + plotH * (1 - thresholds.high / maxV);

    ctx.fillStyle = 'rgba(0,200,100,0.06)';
    ctx.fillRect(pad.left, yLow, plotW, pad.top + plotH - yLow);
    ctx.fillStyle = 'rgba(255,180,0,0.06)';
    ctx.fillRect(pad.left, yHigh, plotW, yLow - yHigh);
    ctx.fillStyle = 'rgba(255,60,60,0.06)';
    ctx.fillRect(pad.left, pad.top, plotW, yHigh - pad.top);

    // Threshold lines
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(0,200,100,0.5)';
    ctx.beginPath(); ctx.moveTo(pad.left, yLow); ctx.lineTo(W - pad.right, yLow); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,60,60,0.5)';
    ctx.beginPath(); ctx.moveTo(pad.left, yHigh); ctx.lineTo(W - pad.right, yHigh); ctx.stroke();
    ctx.setLineDash([]);

    // Labels for thresholds
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillStyle = 'rgba(0,200,100,0.7)';
    ctx.fillText(`${thresholds.low} — Accurate`, pad.left + 4, yLow - 3);
    ctx.fillStyle = 'rgba(255,60,60,0.7)';
    ctx.fillText(`${thresholds.high} — Fast`, pad.left + 4, yHigh - 3);

    // Bars
    const n = velocities.length;
    if (n === 0) return;
    const barW = Math.max(2, Math.min(14, (plotW / n) - 1));
    const gap = (plotW - barW * n) / (n + 1);

    const colors = { accurate: '#00c864', balanced: '#ffb400', fast: '#ff5050' };

    for (let i = 0; i < n; i++) {
        const v = velocities[i];
        const tier = perFrame[i].model_tier;
        const barH = (v / maxV) * plotH;
        const x = pad.left + gap + i * (barW + gap);
        const y = pad.top + plotH - barH;

        ctx.fillStyle = colors[tier] || '#888';
        ctx.fillRect(x, y, barW, barH);
    }

    // Y-axis ticks
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    for (let tick = 0; tick <= maxV; tick += Math.ceil(maxV / 5)) {
        const y = pad.top + plotH * (1 - tick / maxV);
        ctx.fillText(tick.toFixed(0), pad.left - 4, y + 3);
    }

    // X-axis label
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.fillText('Frame →', W / 2, H - 4);
}

// ============================================================================
// MODELS PAGE
// ============================================================================
async function loadModels() {
    try {
        const res = await fetch(`${API}/api/models`);
        const models = await res.json();
        renderModels(models);
    } catch (e) {
        renderModels(null);
    }
}

function renderModels(models) {
    const grid = document.getElementById('modelsGrid');
    const configs = models || {
        'mit_b3': { name: 'FPN + MiT-B3', official_name: 'FPN + MiT-B3', params: '~47M', speed: 'Medium', accuracy: 'Highest', use_case: 'Complex monochromatic scenes', description: 'Best accuracy. Deep feature extraction.', active: true, loaded: false },
        'mit_b1': { name: 'DeepLabV3+ + EfficientNet-B4', official_name: 'DeepLabV3+ + EfficientNet-B4', params: '~25M', speed: 'Medium-Fast', accuracy: 'High', use_case: 'General purpose', description: 'EfficientNet-based checkpoint-compatible model.', active: false, loaded: false },
        'mit_b0': { name: 'Linknet + MobileNetV2', official_name: 'Linknet + MobileNetV2', params: '~4M', speed: 'Very Fast', accuracy: 'Moderate', use_case: 'High-motion real-time video', description: 'Ultra-fast non-FPN model for latency-critical inference.', active: false, loaded: false },
    };

    const speedMap = { 'Very Fast': 95, 'Fast': 70, 'Medium-Fast': 58, 'Medium': 45 };
    const accMap = { 'Highest': 95, 'High': 75, 'Good': 55, 'Moderate': 45 };

    grid.innerHTML = Object.entries(configs).map(([key, m]) => `
        <div class="model-card ${m.active ? 'active-model' : ''}" id="model-${key}">
            <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:8px;">
                <h3>${m.name}</h3>
                <span class="model-loaded-badge ${m.loaded ? 'in-memory' : 'not-loaded'}">${m.loaded ? 'In Memory' : 'Load on Use'}</span>
            </div>
            ${m.official_name ? `<div class="model-spec" style="margin:-4px 0 8px 0;"><strong>Official:</strong> ${m.official_name}</div>` : ''}
            <p>${m.description}</p>
            <div class="model-specs">
                <div class="model-spec"><strong>Params:</strong> ${m.params}</div>
                <div class="model-spec"><strong>Speed:</strong> ${m.speed}</div>
                <div class="model-spec"><strong>Accuracy:</strong> ${m.accuracy}</div>
                <div class="model-spec"><strong>Best for:</strong> ${m.use_case}</div>
            </div>
            <div class="model-score-bars">
                <div class="model-score-bar">
                    <span class="model-score-label">Accuracy</span>
                    <div class="model-score-track"><div class="model-score-fill accuracy" style="width:${accMap[m.accuracy] || 50}%"></div></div>
                </div>
                <div class="model-score-bar">
                    <span class="model-score-label">Speed</span>
                    <div class="model-score-track"><div class="model-score-fill speed" style="width:${speedMap[m.speed] || 50}%"></div></div>
                </div>
            </div>
            <button class="activate-btn" onclick="activateModel('${key}')">${m.active ? '✓ Active' : 'Activate'}</button>
        </div>
    `).join('');
}

async function activateModel(key) {
    const form = new FormData();
    form.append('model_key', key);
    try {
        await fetch(`${API}/api/set-model`, { method: 'POST', body: form });
        loadModels();
        checkHealth();
    } catch (e) {
        alert('Failed to switch model. Is the server running?');
    }
}

// ============================================================================
// XAI TABS
// ============================================================================
function initXAITabs() {
    document.querySelectorAll('.xai-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.xai-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const panelId = tab.dataset.xai + 'Panel';
            document.querySelectorAll('#xaiPanels .xai-panel').forEach(p => p.classList.add('hidden'));
            const panel = document.getElementById(panelId);
            if (panel) panel.classList.remove('hidden');
        });
    });
}

// ============================================================================
// FEATURE 1: ROUTE-LEVEL SHAP
// ============================================================================
function initRouteShap() {
    const canvas = document.getElementById('routeShapCanvas');
    const undoBtn = document.getElementById('routeShapUndo');
    const clearBtn = document.getElementById('routeShapClear');
    const analyzeBtn = document.getElementById('routeShapAnalyze');
    if (!canvas) return;

    canvas.addEventListener('click', (e) => {
        if (!state.currentResult) return;
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        state.routePoints.push({ x, y });
        drawRouteOnCanvas();
    });

    canvas.addEventListener('touchstart', (e) => {
        if (!state.currentResult) return;
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        const x = (touch.clientX - rect.left) / rect.width;
        const y = (touch.clientY - rect.top) / rect.height;
        state.routePoints.push({ x, y });
        drawRouteOnCanvas();
    }, { passive: false });

    if (undoBtn) undoBtn.addEventListener('click', () => {
        state.routePoints.pop();
        drawRouteOnCanvas();
    });
    if (clearBtn) clearBtn.addEventListener('click', () => {
        state.routePoints = [];
        drawRouteOnCanvas();
        document.getElementById('routeShapResults').innerHTML = '<div class="xai-placeholder">Click points on the image to draw a route, then click Analyze.</div>';
    });
    if (analyzeBtn) analyzeBtn.addEventListener('click', () => {
        if (state.routePoints.length < 2) { return; }
        if (!state.currentResult || !state.currentResult.confidence_grid) return;
        const shapData = computeRouteSHAP(state.routePoints, state.currentResult.confidence_grid);
        renderRouteShapResults(shapData);
    });

    // Sync canvas size when image loads
    const baseImg = document.getElementById('routeShapBaseImg');
    if (baseImg) {
        baseImg.addEventListener('load', () => {
            canvas.width = baseImg.offsetWidth;
            canvas.height = baseImg.offsetHeight;
        });
    }
}

function drawRouteOnCanvas() {
    const canvas = document.getElementById('routeShapCanvas');
    const baseImg = document.getElementById('routeShapBaseImg');
    if (!canvas || !baseImg) return;
    canvas.width = baseImg.offsetWidth;
    canvas.height = baseImg.offsetHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const pts = state.routePoints;
    if (!pts.length) return;

    // Draw polyline
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(232,98,45,0.85)';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 3]);
    ctx.lineCap = 'round';
    ctx.moveTo(pts[0].x * canvas.width, pts[0].y * canvas.height);
    for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x * canvas.width, pts[i].y * canvas.height);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw point markers
    pts.forEach((p, i) => {
        const px = p.x * canvas.width, py = p.y * canvas.height;
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? 'rgba(74,222,128,0.9)' : i === pts.length - 1 ? 'rgba(239,68,68,0.9)' : 'rgba(232,98,45,0.8)';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}

function sampleGridAlongLine(grid, x0, y0, x1, y1) {
    // Bresenham-style line walk on 64×64 grid
    const gw = grid.grid_w, gh = grid.grid_h;
    const cells = [];
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let t = 0; t <= steps; t++) {
        const gx = Math.min(Math.floor(x0 + (x1 - x0) * t / steps), gw - 1);
        const gy = Math.min(Math.floor(y0 + (y1 - y0) * t / steps), gh - 1);
        cells.push(grid.cells[gy * gw + gx]);
    }
    return cells;
}

function computeRouteSHAP(points, grid) {
    const safetyWeights = { safe: 0, caution: 0.5, obstacle: 1.0, neutral: 0.1 };
    const safetyMap = {
        'Landscape': 'safe', 'Sky': 'neutral', 'Trees': 'obstacle',
        'Lush Bushes': 'caution', 'Dry Grass': 'safe', 'Dry Bushes': 'caution',
        'Rocks': 'obstacle', 'Ground Clutter': 'caution', 'Flowers': 'safe', 'Logs': 'obstacle',
    };
    const classCounts = {};
    const classConf = {};
    let totalCells = 0;
    let totalRisk = 0;

    for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i], p1 = points[i + 1];
        const gx0 = p0.x * grid.grid_w, gy0 = p0.y * grid.grid_h;
        const gx1 = p1.x * grid.grid_w, gy1 = p1.y * grid.grid_h;
        const cells = sampleGridAlongLine(grid, gx0, gy0, gx1, gy1);
        cells.forEach(cell => {
            if (!cell) return;
            const name = cell.n;
            const conf = cell.p / 100;
            const safety = safetyMap[name] || 'neutral';
            const risk = safetyWeights[safety] * conf;
            classCounts[name] = (classCounts[name] || 0) + 1;
            classConf[name] = (classConf[name] || 0) + conf;
            totalRisk += risk;
            totalCells++;
        });
    }

    if (!totalCells) return null;
    const normalizedRisk = totalRisk / totalCells;
    const classBreakdown = Object.entries(classCounts).map(([name, count]) => {
        const safety = safetyMap[name] || 'neutral';
        const avgConf = classConf[name] / count;
        const contribution = (count / totalCells) * safetyWeights[safety] * avgConf;
        return { name, count, pct: count / totalCells * 100, safety, avgConf, contribution };
    }).sort((a, b) => b.contribution - a.contribution);

    return { normalizedRisk, classBreakdown, totalCells, totalSegments: points.length - 1 };
}

function renderRouteShapResults(data) {
    if (!data) return;
    const resultsEl = document.getElementById('routeShapResults');
    const risk = data.normalizedRisk;
    const safetyPct = Math.max(0, Math.min(100, Math.round((1 - risk) * 100)));
    const riskClass = risk < 0.2 ? 'safe' : risk < 0.5 ? 'caution' : 'danger';
    const safetyLabel = risk < 0.2 ? 'Safe' : risk < 0.5 ? 'Moderate Risk' : 'High Risk';

    const maxContrib = Math.max(...data.classBreakdown.map(c => c.contribution), 0.001);
    const bars = data.classBreakdown.map(c => {
        const pct = Math.round(c.contribution / maxContrib * 100);
        const fillClass = { safe: 'safe-shap', caution: 'caution-shap', obstacle: 'obstacle-shap', neutral: 'neutral-shap' }[c.safety];
        return `<div class="shap-bar-item">
            <div class="shap-bar-label">
                <span>${c.name}</span>
                <span>${c.pct.toFixed(1)}% of route · ${(c.contribution * 100).toFixed(1)} risk pts</span>
            </div>
            <div class="shap-bar-track"><div class="shap-bar-fill ${fillClass}" style="width:${pct}%"></div></div>
        </div>`;
    }).join('');

    resultsEl.innerHTML = `
        <div class="route-score-header">Route Safety Score</div>
        <div class="route-total-risk ${riskClass}">${safetyPct}% — ${safetyLabel}</div>
        <div style="font-size:11px;color:var(--text-3);margin-bottom:16px;font-family:'JetBrains Mono',monospace;">${data.totalSegments} segments · ${data.totalCells} cells sampled</div>
        <div style="font-size:12px;font-weight:500;letter-spacing:1px;text-transform:uppercase;color:var(--text-2);margin-bottom:10px;">Class Contributions</div>
        ${bars}
    `;
}

// ============================================================================
// FEATURE 2: WHAT-IF SCENARIO EDITOR
// ============================================================================
function initWhatIf() {
    const sliders = ['whatifBrightness', 'whatifContrast', 'whatifNoise', 'whatifBlur'];
    sliders.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('input', () => {
            updateWhatIfLabels();
            applyWhatIfTransforms();
        });
    });
    const reanalyze = document.getElementById('whatifReanalyze');
    if (reanalyze) reanalyze.addEventListener('click', whatIfReanalyze);
    const reset = document.getElementById('whatifReset');
    if (reset) reset.addEventListener('click', () => {
        ['whatifBrightness', 'whatifContrast', 'whatifNoise', 'whatifBlur'].forEach(id => {
            const el = document.getElementById(id);
            if (el) { el.value = id === 'whatifContrast' ? 100 : 0; }
        });
        updateWhatIfLabels();
        primeWhatIfCanvas();
        document.getElementById('whatifDelta').classList.add('hidden');
        document.getElementById('whatifResultWrap').innerHTML = '<div class="xai-placeholder">Adjust sliders and click Re-Analyze</div>';
    });
}

function updateWhatIfLabels() {
    const bEl = document.getElementById('whatifBrightness');
    const cEl = document.getElementById('whatifContrast');
    const nEl = document.getElementById('whatifNoise');
    const blEl = document.getElementById('whatifBlur');
    if (bEl) document.getElementById('whatifBrightnessVal').textContent = bEl.value > 0 ? `+${bEl.value}` : bEl.value;
    if (cEl) document.getElementById('whatifContrastVal').textContent = (parseFloat(cEl.value) / 100).toFixed(1) + '×';
    if (nEl) document.getElementById('whatifNoiseVal').textContent = nEl.value;
    if (blEl) document.getElementById('whatifBlurVal').textContent = blEl.value + 'px';
}

function primeWhatIfCanvas() {
    if (!state.currentResult) return;
    const canvas = document.getElementById('whatifCanvas');
    if (!canvas) return;
    applyWhatIfTransforms();
}

function applyWhatIfTransforms() {
    if (!state.currentResult) return;
    const canvas = document.getElementById('whatifCanvas');
    if (!canvas) return;
    const brightnessVal = parseFloat(document.getElementById('whatifBrightness')?.value || 0);
    const contrastVal = parseFloat(document.getElementById('whatifContrast')?.value || 100);
    const noiseVal = parseFloat(document.getElementById('whatifNoise')?.value || 0);
    const blurVal = parseFloat(document.getElementById('whatifBlur')?.value || 0);

    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        const brightnessNorm = 1 + brightnessVal / 100;
        const contrastNorm = contrastVal / 100;
        ctx.filter = `blur(${blurVal}px) brightness(${brightnessNorm}) contrast(${contrastNorm})`;
        ctx.drawImage(img, 0, 0);
        ctx.filter = 'none';

        if (noiseVal > 0) {
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const d = imageData.data;
            for (let i = 0; i < d.length; i += 4) {
                const noise = (Math.random() - 0.5) * noiseVal * 2.55;
                d[i] = Math.max(0, Math.min(255, d[i] + noise));
                d[i + 1] = Math.max(0, Math.min(255, d[i + 1] + noise));
                d[i + 2] = Math.max(0, Math.min(255, d[i + 2] + noise));
            }
            ctx.putImageData(imageData, 0, 0);
        }
    };
    img.src = `data:image/jpeg;base64,${state.currentResult.original}`;
}

async function whatIfReanalyze() {
    if (!state.currentResult) return;
    const canvas = document.getElementById('whatifCanvas');
    if (!canvas) return;
    const resultWrap = document.getElementById('whatifResultWrap');
    resultWrap.innerHTML = `<div style="text-align:center;padding:40px;"><div class="spinner" style="margin:0 auto 12px;"></div><div style="font-size:12px;color:var(--text-3);">Re-analyzing...</div></div>`;

    canvas.toBlob(async (blob) => {
        const form = new FormData();
        form.append('file', blob, 'whatif.png');
        try {
            const res = await fetch(`${API}/api/segment`, { method: 'POST', body: form });
            const data = await res.json();
            state.whatIfResult = data;
            resultWrap.innerHTML = `<img src="data:image/jpeg;base64,${data.overlay}" style="width:100%;display:block;" alt="Re-analyzed">`;
            renderWhatIfDelta(state.currentResult.safety_percentages, data.safety_percentages);
        } catch (err) {
            resultWrap.innerHTML = `<div class="xai-placeholder" style="color:var(--danger);">Error: ${err.message}</div>`;
        }
    }, 'image/png');
}

function renderWhatIfDelta(original, modified) {
    const deltaEl = document.getElementById('whatifDelta');
    if (!deltaEl) return;
    const keys = ['safe', 'caution', 'obstacle'];
    const labels = { safe: 'Safe', caution: 'Caution', obstacle: 'Obstacle' };
    deltaEl.innerHTML = keys.map(k => {
        const delta = modified[k] - original[k];
        const sign = delta > 0 ? '+' : '';
        const cls = k === 'safe' ? (delta > 0 ? 'positive' : 'negative') : (delta < 0 ? 'positive' : 'negative');
        return `<div class="whatif-delta-item">
            <div class="whatif-delta-label">${labels[k]}</div>
            <div class="whatif-delta-val ${cls}">${sign}${delta.toFixed(1)}%</div>
        </div>`;
    }).join('');
    deltaEl.classList.remove('hidden');
}

// ============================================================================
// FEATURE 3: FAILURE MODE EXPLORER
// ============================================================================
function analyzeAndRenderFailureModes(grid) {
    const modes = analyzeFailureModes(grid);
    renderFailureModes(modes);
}

function analyzeFailureModes(grid) {
    const gw = grid.grid_w, gh = grid.grid_h;
    const cells = grid.cells;
    const total = gw * gh;
    const modes = [];

    // Pattern 1: Low-confidence cells (< 50%)
    const lowConfCount = cells.filter(c => c.p < 50).length;
    const lowConfPct = (lowConfCount / total * 100).toFixed(1);
    modes.push({
        title: 'Low-Confidence Regions',
        desc: 'Areas where the model prediction probability is below 50%, indicating ambiguous terrain that could be misclassified.',
        pct: parseFloat(lowConfPct),
        severity: parseFloat(lowConfPct) > 30 ? 'high' : parseFloat(lowConfPct) > 15 ? 'medium' : 'low',
        icon: '⚠',
    });

    // Pattern 2: Boundary confusion
    let boundaryCount = 0;
    for (let y = 0; y < gh; y++) {
        for (let x = 0; x < gw; x++) {
            const cell = cells[y * gw + x];
            if (x < gw - 1) {
                const right = cells[y * gw + x + 1];
                if (cell.c !== right.c && cell.p < 70 && right.p < 70) boundaryCount++;
            }
            if (y < gh - 1) {
                const below = cells[(y + 1) * gw + x];
                if (cell.c !== below.c && cell.p < 70 && below.p < 70) boundaryCount++;
            }
        }
    }
    const boundaryPct = (boundaryCount / (gw * gh) * 100).toFixed(1);
    modes.push({
        title: 'Boundary Confusion',
        desc: 'Adjacent cells with different predicted classes both showing low confidence — the model is unsure about class transitions.',
        pct: parseFloat(boundaryPct),
        severity: parseFloat(boundaryPct) > 20 ? 'high' : parseFloat(boundaryPct) > 8 ? 'medium' : 'low',
        icon: '⊘',
    });

    // Pattern 3: Obstacle near safe zones
    let obstacleNearSafe = 0;
    const SAFETY = { Landscape: 'safe', Sky: 'neutral', Trees: 'obstacle', 'Lush Bushes': 'caution', 'Dry Grass': 'safe', 'Dry Bushes': 'caution', Rocks: 'obstacle', 'Ground Clutter': 'caution', Flowers: 'safe', Logs: 'obstacle' };
    for (let y = 0; y < gh; y++) {
        for (let x = 0; x < gw; x++) {
            const cell = cells[y * gw + x];
            if (SAFETY[cell.n] === 'obstacle') {
                const neighbors = [];
                if (x > 0) neighbors.push(cells[y * gw + x - 1]);
                if (x < gw - 1) neighbors.push(cells[y * gw + x + 1]);
                if (y > 0) neighbors.push(cells[(y - 1) * gw + x]);
                if (y < gh - 1) neighbors.push(cells[(y + 1) * gw + x]);
                if (neighbors.some(n => SAFETY[n.n] === 'safe')) obstacleNearSafe++;
            }
        }
    }
    const obstaclePct = (obstacleNearSafe / total * 100).toFixed(1);
    modes.push({
        title: 'Obstacle-Safe Adjacency',
        desc: 'Obstacle-class regions directly adjacent to safe terrain — potential hazard zones where rapid terrain type changes occur.',
        pct: parseFloat(obstaclePct),
        severity: parseFloat(obstaclePct) > 10 ? 'high' : parseFloat(obstaclePct) > 4 ? 'medium' : 'low',
        icon: '◈',
    });

    // Pattern 4: High variance micro-regions
    let microVariance = 0;
    for (let y = 0; y < gh - 2; y++) {
        for (let x = 0; x < gw - 2; x++) {
            const patch = [
                cells[y * gw + x], cells[y * gw + x + 1],
                cells[(y + 1) * gw + x], cells[(y + 1) * gw + x + 1],
            ];
            const classes = new Set(patch.map(c => c.c));
            if (classes.size >= 3) microVariance++;
        }
    }
    const microPct = (microVariance / total * 100).toFixed(1);
    modes.push({
        title: 'High-Variance Micro-Regions',
        desc: '2×2 patches where 3 or more distinct terrain classes are predicted — fragmented segmentation indicating potential noise sensitivity.',
        pct: parseFloat(microPct),
        severity: parseFloat(microPct) > 15 ? 'high' : parseFloat(microPct) > 6 ? 'medium' : 'low',
        icon: '⋮',
    });

    return modes;
}

function renderFailureModes(modes) {
    const summaryEl = document.getElementById('failureSummary');
    const cardsEl = document.getElementById('failureCards');
    if (!summaryEl || !cardsEl) return;

    const highCount = modes.filter(m => m.severity === 'high').length;
    const medCount = modes.filter(m => m.severity === 'medium').length;

    summaryEl.innerHTML = `
        <span class="failure-stat-badge ${highCount > 0 ? 'critical' : ''}">${highCount} Critical</span>
        <span class="failure-stat-badge ${medCount > 0 ? 'warn' : ''}">${medCount} Warnings</span>
        <span class="failure-stat-badge">${modes.length - highCount - medCount} OK</span>
    `;

    cardsEl.innerHTML = modes.map(m => `
        <div class="failure-card severity-${m.severity}">
            <div class="failure-card-title">${m.icon} ${m.title}</div>
            <div class="failure-card-desc">${m.desc}</div>
            <div class="failure-card-metric ${m.severity}">${m.pct}% of terrain</div>
        </div>
    `).join('');
}

// ============================================================================
// FEATURE 4: ROBUSTNESS SHAKE TEST
// ============================================================================
function initRobustnessTest() {
    const btn = document.getElementById('btnShakeTest');
    if (!btn) return;
    btn.addEventListener('click', () => {
        if (!state.currentFile) { return; }
        runRobustnessTest();
    });
}

async function runRobustnessTest() {
    const btn = document.getElementById('btnShakeTest');
    const container = document.getElementById('shakeContainer');
    btn.disabled = true;
    btn.textContent = 'Running…';
    container.classList.add('hidden');

    const form = new FormData();
    form.append('file', state.currentFile);
    try {
        const res = await fetch(`${API}/api/robustness-test`, { method: 'POST', body: form });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();
        state.robustnessData = data;
        renderRobustnessGrid(data);
        container.classList.remove('hidden');
    } catch (err) {
        container.innerHTML = `<div class="xai-placeholder" style="color:var(--danger);">Error: ${err.message}</div>`;
        container.classList.remove('hidden');
    }
    btn.disabled = false;
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg> Re-run Shake Test`;
}

function renderRobustnessGrid(data) {
    const gridEl = document.getElementById('shakeGrid');
    if (!gridEl) return;
    const types = ['noise', 'blur', 'brightness'];
    const sevs = ['low', 'medium', 'high'];
    const typeLabels = { noise: 'Noise', blur: 'Blur', brightness: 'Brightness' };
    const sevIcons = { low: 'Lo', medium: 'Med', high: 'Hi' };

    let html = `<div class="shake-header">
        <div class="shake-header-cell"></div>
        ${sevs.map(s => `<div class="shake-header-cell">${sevIcons[s]}</div>`).join('')}
    </div>`;

    types.forEach(t => {
        html += `<div class="shake-row"><div class="shake-row-label">${typeLabels[t]}</div>`;
        sevs.forEach(s => {
            const r = data.results[t][s];
            const delta = r.safety_delta ? r.safety_delta.safe : 0;
            const sign = delta >= 0 ? '+' : '';
            const badgeClass = delta > 1 ? 'positive' : delta < -1 ? 'negative' : 'neutral';
            html += `<div class="shake-cell">
                <img src="data:image/jpeg;base64,${r.overlay}" alt="${t} ${s}" loading="lazy">
                <div class="shake-delta-badge ${badgeClass}">${sign}${delta.toFixed(1)}%</div>
                <div class="shake-cell-label">${r.inference_time_ms}ms</div>
            </div>`;
        });
        html += '</div>';
    });

    gridEl.innerHTML = html;
}

// ============================================================================
// FEATURE 5: ROUTE PLANNER + RISK BUDGET
// ============================================================================

// Minimal binary min-heap for A*
class MinHeap {
    constructor() { this.heap = []; }
    push(item, priority) { this.heap.push({ item, priority }); this._bubbleUp(this.heap.length - 1); }
    pop() {
        if (this.heap.length === 1) return this.heap.pop().item;
        const top = this.heap[0].item;
        this.heap[0] = this.heap.pop();
        this._sinkDown(0);
        return top;
    }
    isEmpty() { return this.heap.length === 0; }
    _bubbleUp(i) {
        while (i > 0) {
            const parent = Math.floor((i - 1) / 2);
            if (this.heap[parent].priority <= this.heap[i].priority) break;
            [this.heap[parent], this.heap[i]] = [this.heap[i], this.heap[parent]];
            i = parent;
        }
    }
    _sinkDown(i) {
        const n = this.heap.length;
        while (true) {
            let smallest = i;
            const l = 2 * i + 1, r = 2 * i + 2;
            if (l < n && this.heap[l].priority < this.heap[smallest].priority) smallest = l;
            if (r < n && this.heap[r].priority < this.heap[smallest].priority) smallest = r;
            if (smallest === i) break;
            [this.heap[smallest], this.heap[i]] = [this.heap[i], this.heap[smallest]];
            i = smallest;
        }
    }
}

function initRoutePlanner() {
    const canvas = document.getElementById('plannerCanvas');
    const baseImg = document.getElementById('plannerBaseImg');
    if (!canvas || !baseImg) return;

    baseImg.addEventListener('load', () => {
        canvas.width = baseImg.offsetWidth;
        canvas.height = baseImg.offsetHeight;
    });

    canvas.addEventListener('click', (e) => {
        if (!state.currentResult) return;
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;

        if (!state.plannerStart) {
            state.plannerStart = { x, y };
            document.getElementById('plannerHint').textContent = 'Click to set end point';
        } else if (!state.plannerEnd) {
            state.plannerEnd = { x, y };
            document.getElementById('plannerHint').textContent = 'Computing routes…';
            computeAndRenderRoutes();
        } else {
            // Reset and start over
            state.plannerStart = { x, y };
            state.plannerEnd = null;
            document.getElementById('plannerHint').textContent = 'Click to set end point';
            document.getElementById('plannerResults').innerHTML = '<div class="xai-placeholder">Set start and end points on the map to compute routes.</div>';
            drawPlannerCanvas();
        }
        drawPlannerCanvas();
    });
}

function drawPlannerCanvas() {
    const canvas = document.getElementById('plannerCanvas');
    const baseImg = document.getElementById('plannerBaseImg');
    if (!canvas || !baseImg) return;
    canvas.width = baseImg.offsetWidth;
    canvas.height = baseImg.offsetHeight;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw routes if computed
    if (state.plannerRoutes) {
        const routeColors = ['rgba(74,222,128,0.85)', 'rgba(232,98,45,0.85)', 'rgba(251,191,36,0.85)'];
        const grid = state.currentResult.confidence_grid;
        state.plannerRoutes.forEach((route, idx) => {
            if (!route) return;
            ctx.beginPath();
            ctx.strokeStyle = routeColors[idx];
            ctx.lineWidth = 2.5;
            ctx.setLineDash(idx === 0 ? [] : [5, 3]);
            route.forEach((nodeIdx, i) => {
                const gx = nodeIdx % grid.grid_w;
                const gy = Math.floor(nodeIdx / grid.grid_w);
                const px = (gx + 0.5) / grid.grid_w * canvas.width;
                const py = (gy + 0.5) / grid.grid_h * canvas.height;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        });
    }

    // Draw start/end markers
    if (state.plannerStart) {
        const sx = state.plannerStart.x * canvas.width;
        const sy = state.plannerStart.y * canvas.height;
        ctx.beginPath(); ctx.arc(sx, sy, 7, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(74,222,128,0.9)'; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.font = 'bold 8px Inter';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('S', sx, sy);
    }
    if (state.plannerEnd) {
        const ex = state.plannerEnd.x * canvas.width;
        const ey = state.plannerEnd.y * canvas.height;
        ctx.beginPath(); ctx.arc(ex, ey, 7, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239,68,68,0.9)'; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.font = 'bold 8px Inter';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('E', ex, ey);
    }
}

function aStarPathfind(gridCells, gw, gh, start, end, costFn) {
    const startIdx = start.y * gw + start.x;
    const endIdx = end.y * gw + end.x;
    const gScore = new Float32Array(gw * gh).fill(Infinity);
    const fScore = new Float32Array(gw * gh).fill(Infinity);
    const cameFrom = new Int32Array(gw * gh).fill(-1);
    const openSet = new MinHeap();

    // ── Perspective-aware distance scaling ──────────────────────────────
    // In a forward-facing camera, pixels near the top of the image are
    // physically farther away → 1 pixel spans more real-world distance.
    // We model this as a linear depth scale:
    //   row 0       (farthest)  → scale = PERSPECTIVE_MAX_SCALE
    //   row gh - 1  (nearest)   → scale = 1.0
    // This makes the planner prefer near-ground manoeuvres over
    // far-field detours that look short in pixel space but are large IRL.
    const PERSPECTIVE_MAX_SCALE = 4.0;
    const perspScale = (row) => {
        const t = 1 - row / Math.max(gh - 1, 1);   // 1 at top, 0 at bottom
        return 1.0 + t * (PERSPECTIVE_MAX_SCALE - 1.0);
    };

    // Admissible heuristic: use the minimum possible scale (1.0) so we
    // never overestimate the remaining cost.
    const heuristic = (idx) => {
        const x = idx % gw, y = Math.floor(idx / gw);
        return Math.sqrt((x - end.x) ** 2 + (y - end.y) ** 2); // * 1.0
    };

    gScore[startIdx] = 0;
    fScore[startIdx] = heuristic(startIdx);
    openSet.push(startIdx, fScore[startIdx]);
    const dirs = [[-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1], [-1, -1, 1.414], [-1, 1, 1.414], [1, -1, 1.414], [1, 1, 1.414]];

    let iterations = 0;
    while (!openSet.isEmpty() && iterations++ < 10000) {
        const current = openSet.pop();
        if (current === endIdx) {
            const path = [];
            let node = current;
            while (node !== -1) { path.unshift(node); node = cameFrom[node]; }
            return path;
        }
        const cx = current % gw, cy = Math.floor(current / gw);
        for (const [dx, dy, moveCost] of dirs) {
            const nx = cx + dx, ny = cy + dy;
            if (nx < 0 || nx >= gw || ny < 0 || ny >= gh) continue;
            const nIdx = ny * gw + nx;
            // Average perspective scale of the two neighbouring rows
            const pScale = (perspScale(cy) + perspScale(ny)) * 0.5;
            const tentative = gScore[current] + moveCost * pScale * costFn(gridCells[nIdx]);
            if (tentative < gScore[nIdx]) {
                cameFrom[nIdx] = current;
                gScore[nIdx] = tentative;
                fScore[nIdx] = tentative + heuristic(nIdx);
                openSet.push(nIdx, fScore[nIdx]);
            }
        }
    }
    return null;
}

function computeAndRenderRoutes() {
    if (!state.currentResult || !state.plannerStart || !state.plannerEnd) return;
    const grid = state.currentResult.confidence_grid;
    const gw = grid.grid_w, gh = grid.grid_h;

    const toGrid = (p) => ({
        x: Math.min(Math.floor(p.x * gw), gw - 1),
        y: Math.min(Math.floor(p.y * gh), gh - 1),
    });

    const start = toGrid(state.plannerStart);
    const end = toGrid(state.plannerEnd);

    const SAFETY = { Landscape: 'safe', Sky: 'neutral', Trees: 'obstacle', 'Lush Bushes': 'caution', 'Dry Grass': 'safe', 'Dry Bushes': 'caution', Rocks: 'obstacle', 'Ground Clutter': 'caution', Flowers: 'safe', Logs: 'obstacle' };

    const costFns = [
        cell => { const s = SAFETY[cell.n] || 'neutral'; return { safe: 1, caution: 5, obstacle: 80, neutral: 2 }[s]; },   // Safest
        cell => 1,   // Shortest
        cell => { const s = SAFETY[cell.n] || 'neutral'; return { safe: 1, caution: 3, obstacle: 25, neutral: 1.5 }[s]; }, // Balanced
    ];

    state.plannerRoutes = costFns.map(fn => aStarPathfind(grid.cells, gw, gh, start, end, fn));
    drawPlannerCanvas();

    const routeNames = ['Safest', 'Shortest', 'Balanced'];
    const routeColors = ['route-safe', 'route-short', 'route-balanced'];

    const resultsEl = document.getElementById('plannerResults');
    if (!state.plannerRoutes.some(r => r)) {
        resultsEl.innerHTML = '<div class="xai-placeholder">No viable path found between the selected points.</div>';
        document.getElementById('plannerHint').textContent = 'Click to reset';
        return;
    }

    resultsEl.innerHTML = state.plannerRoutes.map((route, idx) => {
        if (!route) return `<div class="route-option-card ${routeColors[idx]}"><div class="route-option-title">${routeNames[idx]}</div><div class="route-option-meta" style="color:var(--danger);">No path found</div></div>`;
        const budget = computeRiskBudget(route, grid);
        const totalDist = route.length;
        return `<div class="route-option-card ${routeColors[idx]}">
            <div class="route-option-title">${['🟢', '🔴', '🟡'][idx]} ${routeNames[idx]} Route</div>
            <div class="route-option-meta">${totalDist} cells · Safe: ${budget.safe.toFixed(0)}% · Risk: ${(budget.caution + budget.obstacle).toFixed(0)}%</div>
            <div class="route-risk-budget">
                <div class="route-budget-seg safe-seg" style="width:${budget.safe}%"></div>
                <div class="route-budget-seg caution-seg" style="width:${budget.caution}%"></div>
                <div class="route-budget-seg obstacle-seg" style="width:${budget.obstacle}%"></div>
                <div class="route-budget-seg neutral-seg" style="width:${budget.neutral}%"></div>
            </div>
            <div class="route-budget-legend">
                <span>🟢 ${budget.safe.toFixed(0)}% safe</span>
                <span>🟡 ${budget.caution.toFixed(0)}% caution</span>
                <span>🔴 ${budget.obstacle.toFixed(0)}% obstacle</span>
            </div>
        </div>`;
    }).join('');

    document.getElementById('plannerHint').textContent = 'Click to plan a new route';
}

function computeRiskBudget(route, grid) {
    const SAFETY = { Landscape: 'safe', Sky: 'neutral', Trees: 'obstacle', 'Lush Bushes': 'caution', 'Dry Grass': 'safe', 'Dry Bushes': 'caution', Rocks: 'obstacle', 'Ground Clutter': 'caution', Flowers: 'safe', Logs: 'obstacle' };
    const counts = { safe: 0, caution: 0, obstacle: 0, neutral: 0 };
    route.forEach(idx => {
        const cell = grid.cells[idx];
        const s = cell ? (SAFETY[cell.n] || 'neutral') : 'neutral';
        counts[s]++;
    });
    const total = route.length || 1;
    return { safe: counts.safe / total * 100, caution: counts.caution / total * 100, obstacle: counts.obstacle / total * 100, neutral: counts.neutral / total * 100 };
}

// ============================================================================
// FEATURE 6: DISAGREEMENT HEATMAP
// ============================================================================
function initDisagreement() {
    const btn = document.getElementById('btnDisagreement');
    if (!btn) return;
    btn.addEventListener('click', () => {
        if (!state.currentFile) return;
        runDisagreement();
    });
}

async function runDisagreement() {
    const btn = document.getElementById('btnDisagreement');
    const results = document.getElementById('disagreementResults');
    btn.disabled = true;
    btn.textContent = 'Computing…';
    results.classList.add('hidden');

    const form = new FormData();
    form.append('file', state.currentFile);
    try {
        const res = await fetch(`${API}/api/model-disagreement`, { method: 'POST', body: form });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();
        state.disagreementData = data;
        renderDisagreement(data);
        results.classList.remove('hidden');
    } catch (err) {
        results.innerHTML = `<div class="xai-placeholder" style="color:var(--danger);">Error: ${err.message}</div>`;
        results.classList.remove('hidden');
    }
    btn.disabled = false;
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/></svg> Recompute`;
}

function renderDisagreement(data) {
    const statsEl = document.getElementById('disagreementStats');
    const gridEl = document.getElementById('disagreementGrid');
    const s = data.statistics;

    statsEl.innerHTML = `
        <div class="disagree-stat"><div class="disagree-stat-val agree">${s.full_agreement}%</div><div class="disagree-stat-label">Full Agreement</div></div>
        <div class="disagree-stat"><div class="disagree-stat-val partial">${s.partial_disagreement}%</div><div class="disagree-stat-label">Partial Disagreement</div></div>
        <div class="disagree-stat"><div class="disagree-stat-val full">${s.full_disagreement}%</div><div class="disagree-stat-label">Full Disagreement</div></div>
    `;

    const modelNames = {
        mit_b0: 'Linknet + MobileNetV2 Prediction',
        mit_b1: 'DeepLabV3+ + EfficientNet-B4 Prediction',
        mit_b3: 'FPN + MiT-B3 Prediction'
    };
    const imgCards = Object.entries(data.per_model_predictions).map(([key, b64]) =>
        `<div class="disagree-card"><img src="data:image/jpeg;base64,${b64}" alt="${key}" loading="lazy"><div class="disagree-card-label">${modelNames[key] || key}</div></div>`
    ).join('');

    gridEl.innerHTML = `
        <div class="disagree-card" style="grid-column:1/-1;">
            <img src="data:image/jpeg;base64,${data.overlay}" alt="Disagreement Overlay" loading="lazy" style="aspect-ratio:16/7;">
            <div class="disagree-card-label">Disagreement Heatmap (Hot = High Disagreement)</div>
        </div>
        ${imgCards}
    `;
}

// ============================================================================
// FEATURE 8: MISSION BRIEFING REPORT
// ============================================================================
function initMissionReport() {
    const generateBtn = document.getElementById('btnGenerateReport');
    const downloadBtn = document.getElementById('btnDownloadReport');
    if (!generateBtn) return;
    generateBtn.addEventListener('click', generateMissionReport);
    if (downloadBtn) downloadBtn.addEventListener('click', downloadReport);
}

function generateMissionReport() {
    if (!state.currentResult) return;
    const preview = document.getElementById('reportPreview');
    const canvas = document.getElementById('reportCanvas');
    if (!canvas) return;

    const W = 1200, H = 900;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    const data = state.currentResult;

    // Background
    ctx.fillStyle = '#080808';
    ctx.fillRect(0, 0, W, H);

    // Subtle grid
    ctx.strokeStyle = 'rgba(212,165,116,0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 60) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 60) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    // Top bar
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, 'rgba(232,98,45,0.15)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, 80);
    ctx.fillStyle = 'rgba(232,98,45,0.8)';
    ctx.fillRect(0, 78, W, 2);

    // Title
    ctx.font = '300 32px "Cormorant Garamond", Georgia, serif';
    ctx.fillStyle = '#f0e6dc';
    ctx.textAlign = 'left';
    ctx.fillText('TERRAIN AI  ·  MISSION BRIEFING', 36, 48);

    // Metadata line
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(212,165,116,0.45)';
    const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
    ctx.fillText(`${ts} UTC  ·  Model: ${(data.model_used || 'mit_b3').toUpperCase()}  ·  Inference: ${data.inference_time_ms}ms  ·  Input: ${data.input_size || '—'}`, 36, 66);

    // Load and draw images
    const loadImg = (b64) => new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = `data:image/jpeg;base64,${b64}`;
    });

    Promise.all([
        loadImg(data.original),
        loadImg(data.overlay),
        loadImg(data.safety),
    ]).then(([origImg, overlayImg, safetyImg]) => {
        const imgY = 100, imgH = 360, imgW = 370, gap = 14;

        const drawImgCard = (img, x, y, w, h, label) => {
            ctx.fillStyle = 'rgba(140,80,50,0.05)';
            ctx.strokeStyle = 'rgba(200,130,80,0.1)';
            ctx.lineWidth = 1;
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
            if (img) ctx.drawImage(img, x, y, w, h);
            // label
            ctx.fillStyle = 'rgba(8,8,8,0.75)';
            ctx.fillRect(x + 8, y + 8, label.length * 7.5 + 20, 22);
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.fillStyle = 'rgba(212,165,116,0.7)';
            ctx.textAlign = 'left';
            ctx.fillText(label.toUpperCase(), x + 18, y + 23);
        };

        drawImgCard(origImg, 36, imgY, imgW, imgH, 'Original');
        drawImgCard(overlayImg, 36 + imgW + gap, imgY, imgW, imgH, 'Segmented Overlay');
        drawImgCard(safetyImg, 36 + (imgW + gap) * 2, imgY, imgW, imgH, 'Safety Map');

        // Safety gauge
        const gaugeY = imgY + imgH + 30;
        ctx.font = '300 20px "Cormorant Garamond", Georgia, serif';
        ctx.fillStyle = 'rgba(212,165,116,0.6)';
        ctx.textAlign = 'left';
        ctx.fillText('TERRAIN SAFETY ASSESSMENT', 36, gaugeY);

        const sp = data.safety_percentages || {};
        const bars = [
            { label: 'Safe', pct: sp.safe || 0, color: '#4ade80' },
            { label: 'Caution', pct: sp.caution || 0, color: '#fbbf24' },
            { label: 'Obstacle', pct: sp.obstacle || 0, color: '#ef4444' },
        ];
        bars.forEach((b, i) => {
            const bY = gaugeY + 20 + i * 36;
            ctx.font = '11px Inter';
            ctx.fillStyle = b.color;
            ctx.textAlign = 'left';
            ctx.fillText(b.label, 36, bY + 14);
            ctx.fillStyle = 'rgba(180,140,100,0.06)';
            ctx.fillRect(120, bY, 700, 20);
            ctx.fillStyle = b.color;
            ctx.fillRect(120, bY, 700 * (b.pct / 100), 20);
            ctx.font = '11px "JetBrains Mono", monospace';
            ctx.fillStyle = b.color;
            ctx.textAlign = 'right';
            ctx.fillText(`${b.pct}%`, 850, bY + 14);
        });

        // Class distribution
        const distY = gaugeY + 140;
        ctx.font = '300 20px "Cormorant Garamond", Georgia, serif';
        ctx.fillStyle = 'rgba(212,165,116,0.6)';
        ctx.textAlign = 'left';
        ctx.fillText('CLASS DISTRIBUTION', 36, distY);
        const dist = data.class_distribution || [];
        dist.slice(0, 6).forEach((c, i) => {
            const cY = distY + 20 + i * 28;
            const color = c.color ? `rgb(${c.color.join(',')})` : '#888';
            ctx.fillStyle = color;
            ctx.fillRect(36, cY, 12, 12);
            ctx.font = '11px Inter';
            ctx.fillStyle = 'rgba(240,230,220,0.7)';
            ctx.textAlign = 'left';
            ctx.fillText(c.name, 56, cY + 10);
            ctx.fillStyle = 'rgba(180,140,100,0.08)';
            ctx.fillRect(220, cY, 300, 12);
            ctx.fillStyle = color;
            ctx.fillRect(220, cY, 300 * (c.percentage / 100), 12);
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.fillStyle = 'rgba(212,165,116,0.4)';
            ctx.textAlign = 'right';
            ctx.fillText(`${c.percentage}%`, 540, cY + 10);
        });

        // Footer
        ctx.fillStyle = 'rgba(232,98,45,0.4)';
        ctx.fillRect(0, H - 34, W, 1);
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(180,140,100,0.25)';
        ctx.textAlign = 'left';
        ctx.fillText('TerrainAI — Semantic Segmentation for Autonomous UGV Navigation  ·  PyTorch + FPN + MiT Encoder', 36, H - 12);
        ctx.textAlign = 'right';
        ctx.fillText('CONFIDENTIAL MISSION DATA', W - 36, H - 12);

        preview.classList.remove('hidden');
    });
}

function downloadReport() {
    const canvas = document.getElementById('reportCanvas');
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `terrainai-briefing-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

// ============================================================================
// FINE-TUNING
// ============================================================================
async function loadUserModelIfExists() {
    const savedUserId = localStorage.getItem('finetuned_model_id');
    if (!savedUserId) return;
    
    try {
        const formData = new FormData();
        formData.append('user_id', savedUserId);
        
        const response = await fetch(`${API}/api/load-user-model`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('Loaded custom model:', result.model_key);
            // Refresh health status to show custom model
            await checkHealth();
        } else {
            // Model file might have been deleted, clear localStorage
            localStorage.removeItem('finetuned_model_id');
        }
    } catch (error) {
        console.error('Failed to load user model:', error);
        localStorage.removeItem('finetuned_model_id');
    }
}

async function resetToBaseModel() {
    try {
        const response = await fetch(`${API}/api/reset-to-base-model`, {
            method: 'POST'
        });
        
        if (response.ok) {
            localStorage.removeItem('finetuned_model_id');
            await checkHealth();
            showNotification('Reset to base model', 'success');
        }
    } catch (error) {
        showNotification('Failed to reset model', 'error');
    }
}

function initFineTuning() {
    const imagesInput = document.getElementById('finetuneImagesInput');
    const masksInput = document.getElementById('finetuneMasksInput');
    const btnStartFinetune = document.getElementById('btnStartFinetune');
    const imagesCount = document.getElementById('imagesCount');
    const masksCount = document.getElementById('masksCount');
    
    let selectedImages = [];
    let selectedMasks = [];
    
    if (!imagesInput || !masksInput || !btnStartFinetune) return;
    
    // Handle images selection
    imagesInput.addEventListener('change', (e) => {
        selectedImages = Array.from(e.target.files);
        if (selectedImages.length > 10) {
            showNotification('Maximum 10 images allowed', 'error');
            selectedImages = selectedImages.slice(0, 10);
        }
        imagesCount.textContent = `${selectedImages.length} file(s) selected`;
        updateFinetuneButton();
    });
    
    // Handle masks selection
    masksInput.addEventListener('change', (e) => {
        selectedMasks = Array.from(e.target.files);
        if (selectedMasks.length > 10) {
            showNotification('Maximum 10 masks allowed', 'error');
            selectedMasks = selectedMasks.slice(0, 10);
        }
        masksCount.textContent = `${selectedMasks.length} file(s) selected`;
        updateFinetuneButton();
    });
    
    // Enable button only when both images and masks are selected
    function updateFinetuneButton() {
        if (selectedImages.length > 0 && selectedMasks.length > 0) {
            if (selectedImages.length === selectedMasks.length) {
                btnStartFinetune.disabled = false;
            } else {
                btnStartFinetune.disabled = true;
                if (selectedImages.length > 0 && selectedMasks.length > 0) {
                    showNotification('Number of images and masks must match', 'warning');
                }
            }
        } else {
            btnStartFinetune.disabled = true;
        }
    }
    
    // Start fine-tuning
    btnStartFinetune.addEventListener('click', async () => {
        if (selectedImages.length !== selectedMasks.length) {
            showNotification('Number of images and masks must match', 'error');
            return;
        }
        
        const progressDiv = document.getElementById('finetuneProgress');
        const resultsDiv = document.getElementById('finetuneResults');
        
        // Show progress
        progressDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');
        btnStartFinetune.disabled = true;
        
        // Animate progress steps
        animateProgressSteps();
        
        try {
            const formData = new FormData();
            
            // Append all images
            selectedImages.forEach(img => {
                formData.append('images', img);
            });
            
            // Append all masks
            selectedMasks.forEach(mask => {
                formData.append('masks', mask);
            });
            
            const response = await fetch(`${API}/api/finetune`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Fine-tuning failed');
            }
            
            const result = await response.json();
            
            // Complete all steps
            completeAllSteps();
            
            // Wait a bit before showing results
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Hide progress, show results
            progressDiv.classList.add('hidden');
            resultsDiv.classList.remove('hidden');
            
            // Update results display
            document.getElementById('modelId').textContent = result.user_id;
            document.getElementById('numImages').textContent = result.num_images;
            document.getElementById('resultsMessage').textContent = result.message;
            
            // Display metrics if available
            if (result.metrics) {
                document.getElementById('initialLoss').textContent = result.metrics.initial_loss.toFixed(4);
                document.getElementById('finalLoss').textContent = result.metrics.final_loss.toFixed(4);
                document.getElementById('improvement').textContent = `${result.metrics.improvement}%`;
            }
            
            // Display comparison if available
            const comparisonSection = document.getElementById('comparisonSection');
            if (result.comparison) {
                document.getElementById('baseGradcam').src = `data:image/png;base64,${result.comparison.base_gradcam}`;
                document.getElementById('finetunedGradcam').src = `data:image/png;base64,${result.comparison.finetuned_gradcam}`;
                comparisonSection.classList.remove('hidden');
            } else {
                comparisonSection.classList.add('hidden');
            }
            
            // Save user model ID to localStorage for persistence
            localStorage.setItem('finetuned_model_id', result.user_id);
            
            // Automatically load and activate the new model
            const loadFormData = new FormData();
            loadFormData.append('user_id', result.user_id);
            
            await fetch(`${API}/api/load-user-model`, {
                method: 'POST',
                body: loadFormData
            });
            
            // Refresh health status to show new custom model
            await checkHealth();
            
            showNotification('Fine-tuning completed! Custom model now active.', 'success');
            
        } catch (error) {
            progressDiv.classList.add('hidden');
            resetProgressSteps();
            showNotification(`Error: ${error.message}`, 'error');
            btnStartFinetune.disabled = false;
        }
    });
}

function animateProgressSteps() {
    // Activate steps sequentially with realistic timing
    const steps = [
        { id: 'step1', delay: 100 },   // Loading model
        { id: 'step2', delay: 800 },   // Freezing layers
        { id: 'step3', delay: 1500 },  // Preparing dataset
        { id: 'step4', delay: 2500 },  // Training (longest)
        { id: 'step5', delay: 8000 },  // Saving
    ];
    
    steps.forEach(({ id, delay }) => {
        setTimeout(() => {
            const step = document.getElementById(id);
            if (step) {
                step.classList.add('active');
                // Show spinner
                const spinner = step.querySelector('.step-spinner');
                if (spinner) spinner.style.display = 'block';
            }
        }, delay);
    });
}

function completeAllSteps() {
    // Mark all steps as complete with checkmarks
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        if (step) {
            step.classList.add('completed');
            step.classList.remove('active');
            const spinner = step.querySelector('.step-spinner');
            const check = step.querySelector('.step-check');
            if (spinner) spinner.style.display = 'none';
            if (check) check.classList.remove('hidden');
        }
    }
}

function resetProgressSteps() {
    // Reset all steps to initial state
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        if (step) {
            step.classList.remove('active', 'completed');
            const spinner = step.querySelector('.step-spinner');
            const check = step.querySelector('.step-check');
            if (spinner) spinner.style.display = 'none';
            if (check) check.classList.add('hidden');
        }
    }
}

function showNotification(message, type = 'info') {
    // Simple notification - you can enhance this with a proper notification system
    const statusText = document.getElementById('statusText');
    if (statusText) {
        const originalText = statusText.textContent;
        statusText.textContent = message;
        statusText.style.color = type === 'error' ? '#ef4444' : type === 'success' ? '#4ade80' : '#f59e0b';
        setTimeout(() => {
            statusText.textContent = originalText;
            statusText.style.color = '';
        }, 5000);
    }
}

// ============================================================================
// CONFIDENCE HOVER TOOLTIP
// ============================================================================
(function initConfidenceHover() {
    document.addEventListener('DOMContentLoaded', () => {
        const outputImg = document.getElementById('resultOutput');
        const tooltip = document.getElementById('confTooltip');
        const confValue = document.getElementById('confValue');
        if (!outputImg || !tooltip || !confValue) return;

        outputImg.addEventListener('mousemove', (e) => {
            const grid = state.currentResult && state.currentResult.confidence_grid;
            if (!grid) { tooltip.classList.add('hidden'); return; }
            const rect = outputImg.getBoundingClientRect();
            const x = e.clientX - rect.left, y = e.clientY - rect.top;
            const gx = Math.min(Math.floor((x / rect.width) * grid.grid_w), grid.grid_w - 1);
            const gy = Math.min(Math.floor((y / rect.height) * grid.grid_h), grid.grid_h - 1);
            const cell = grid.cells[gy * grid.grid_w + gx];
            if (!cell) return;

            confValue.textContent = `${cell.p}%`;
            confValue.className = 'conf-value';
            if (cell.p >= 80) confValue.classList.add('conf-high');
            else if (cell.p >= 50) confValue.classList.add('conf-mid');
            else confValue.classList.add('conf-low');

            tooltip.style.left = `${x}px`;
            tooltip.style.top = `${y}px`;
            tooltip.classList.remove('hidden');
        });

        outputImg.addEventListener('mouseleave', () => { tooltip.classList.add('hidden'); });
    });
})();