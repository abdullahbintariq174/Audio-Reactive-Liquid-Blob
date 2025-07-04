import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// --- SETTINGS ---
const SETTINGS = {
    camera: {
        dollySpeed: 0.1,
        dollyDistance: 3,
    },
    blob: {
        noiseSpeed: 0.05,
        baseDisplacement: 0.8,
        colorSpeed: 0.1,
    },
    audio: {
        bassThreshold: 150, // Hz
        midThreshold: 2000, // Hz
        beatDetection: {
            smoothing: 0.8,
            thresholdMultiplier: 1.5,
            decay: 0.97,
        },
        bassScale: { min: 0.9, max: 1.3 },
        midDisplacement: 1.5,
        highColorSpeed: 1.5,
    },
    postProcessing: {
        bloomStrength: 0.5,
        bloomThreshold: 0.1,
        bloomRadius: 0.8,
    }
};

// --- GLSL SHADERS ---
const vertexShader = `
    uniform float u_time;
    uniform float u_amplitude;
    uniform float u_frequency;
    uniform float u_beat_flash;
    uniform float u_audio_displacement;

    varying vec3 v_normal;
    varying vec3 v_position;
    varying float v_noise;

    // 3D Simplex Noise
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

    float snoise(vec3 v) {
        const vec2 C = vec2(1.0/6.0, 1.0/3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
        vec3 i  = floor(v + dot(v, C.yyy));
        vec3 x0 = v - i + dot(i, C.xxx);
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min(g.xyz, l.zxy);
        vec3 i2 = max(g.xyz, l.zxy);
        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy;
        vec3 x3 = x0 - D.yyy;
        i = mod289(i);
        vec4 p = permute(permute(permute(
            i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));
        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;
        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_);
        vec4 x = x_ * ns.x + ns.yyyy;
        vec4 y = y_ * ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
        vec4 b0 = vec4(x.xy, y.xy);
        vec4 b1 = vec4(x.zw, y.zw);
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
        vec3 p0 = vec3(a0.xy,h.x);
        vec3 p1 = vec3(a0.zw,h.y);
        vec3 p2 = vec3(a1.xy,h.z);
        vec3 p3 = vec3(a1.zw,h.w);
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }

    void main() {
        v_normal = normal;
        v_position = position;

        float noise = snoise(position * u_frequency + u_time);
        v_noise = noise;

        float displacement = u_amplitude * noise * (1.0 + u_audio_displacement);
        vec3 new_position = position + normal * displacement;

        gl_Position = projectionMatrix * modelViewMatrix * vec4(new_position, 1.0);
    }
`;

const fragmentShader = `
    uniform float u_time;
    uniform float u_color_speed;
    uniform float u_beat_flash;
    
    varying vec3 v_normal;
    varying vec3 v_position;
    varying float v_noise;

    // Color palette function
    vec3 palette(float t) {
        vec3 a = vec3(0.5, 0.5, 0.5);
        vec3 b = vec3(0.5, 0.5, 0.5);
        vec3 c = vec3(1.0, 1.0, 1.0);
        vec3 d = vec3(0.3, 0.4, 0.5);
        return a + b * cos(6.28318 * (c * t + d));
    }

    void main() {
        float time_factor = u_time * u_color_speed;
        vec3 base_color = palette(time_factor + v_noise * 0.1);

        vec3 view_dir = normalize(cameraPosition - v_position);
        float fresnel = 1.0 - dot(v_normal, view_dir);
        fresnel = pow(fresnel, 2.0);
        vec3 fresnel_color = vec3(0.8, 0.8, 1.0) * fresnel;

        vec3 final_color = base_color + fresnel_color;
        final_color *= (1.0 + u_beat_flash);

        gl_FragColor = vec4(final_color, 0.75);
    }
`;

// --- MAIN APP CLASS ---
class App {
    constructor() {
        this.init();
        this.setupEventListeners();
    }

    init() {
        // Core Three.js Setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.z = 12;
        this.renderer = new THREE.WebGLRenderer({
            canvas: document.querySelector('#three-canvas'),
            antialias: true,
            alpha: true,
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.clock = new THREE.Clock();

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.enablePan = false;
        this.controls.enableZoom = false;
        
        // Geometry & Material
        const geometry = new THREE.IcosahedronGeometry(4, 64);
        this.material = new THREE.ShaderMaterial({
            uniforms: {
                u_time: { value: 0.0 },
                u_amplitude: { value: SETTINGS.blob.baseDisplacement },
                u_frequency: { value: 0.3 },
                u_color_speed: { value: SETTINGS.blob.colorSpeed },
                u_beat_flash: { value: 0.0 },
                u_audio_displacement: { value: 0.0 },
            },
            vertexShader,
            fragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });

        this.blob = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.blob);

        // Post-Processing (Bloom)
        const renderScene = new RenderPass(this.scene, this.camera);
        this.bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.5, 0.4, 0.85
        );
        this.bloomPass.threshold = SETTINGS.postProcessing.bloomThreshold;
        this.bloomPass.strength = SETTINGS.postProcessing.bloomStrength;
        this.bloomPass.radius = SETTINGS.postProcessing.bloomRadius;

        this.composer = new EffectComposer(this.renderer);
        this.composer.addPass(renderScene);
        this.composer.addPass(this.bloomPass);
        
        // Audio Setup
        this.audio = {
            ready: false,
            context: null,
            source: null,
            analyser: null,
            dataArray: null,
            frequencyData: null,
            bufferLength: 0,
            bassAverage: 0,
            midAverage: 0,
            highAverage: 0,
            overallLevel: 0,
            beatDetection: {
                bassHistory: [],
                beatThreshold: 0,
                lastBeat: 0,
                beatFlash: 0,
            },
        };

        // Performance monitoring
        this.performance = {
            lastTime: 0,
            frameCount: 0,
            fps: 0,
        };

        this.animate();
    }

    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start button (microphone)
        document.getElementById('start-button').addEventListener('click', () => this.startMicrophone());

        // File button
        document.getElementById('file-button').addEventListener('click', () => this.openFileDialog());

        // Bloom controls
        document.getElementById('bloom-toggle').addEventListener('change', (e) => {
            this.bloomPass.enabled = e.target.checked;
        });

        document.getElementById('bloom-slider').addEventListener('input', (e) => {
            this.bloomPass.strength = parseFloat(e.target.value);
        });

        // Drag and drop
        document.body.addEventListener('dragover', (e) => {
            e.preventDefault();
            document.body.classList.add('drag-over');
        });

        document.body.addEventListener('dragleave', (e) => {
            if (e.target === document.body) {
                document.body.classList.remove('drag-over');
            }
        });

        document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            document.body.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
    }

    async startMicrophone() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.setupAudioContext(stream);
            this.showAudioUI();
            this.displayMessage('Microphone activated!', 'success');
        } catch (error) {
            this.displayMessage('Microphone access denied: ' + error.message, 'error');
        }
    }

    openFileDialog() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.mp3,.wav,.ogg';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleFileUpload(file);
            }
        };
        input.click();
    }

    async handleFileUpload(file) {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.loop = true;
            source.start();
            
            this.setupAudioContext(null, audioContext, source);
            this.showAudioUI();
            this.displayMessage(`Playing: ${file.name}`, 'success');
        } catch (error) {
            this.displayMessage('Error loading audio file: ' + error.message, 'error');
        }
    }

    setupAudioContext(stream, audioContext = null, source = null) {
        this.audio.context = audioContext || new (window.AudioContext || window.webkitAudioContext)();
        
        if (stream) {
            this.audio.source = this.audio.context.createMediaStreamSource(stream);
        } else if (source) {
            this.audio.source = source;
        }

        this.audio.analyser = this.audio.context.createAnalyser();
        this.audio.analyser.fftSize = 2048;
        this.audio.analyser.smoothingTimeConstant = 0.8;
        
        this.audio.source.connect(this.audio.analyser);
        
        this.audio.bufferLength = this.audio.analyser.frequencyBinCount;
        this.audio.dataArray = new Uint8Array(this.audio.bufferLength);
        this.audio.frequencyData = new Uint8Array(this.audio.bufferLength);
        
        this.audio.ready = true;
    }

    showAudioUI() {
        document.getElementById('controls').style.display = 'none';
        document.getElementById('stats-overlay').style.display = 'flex';
    }

    displayMessage(message, type = 'info') {
        const errorElement = document.getElementById('error-message');
        errorElement.textContent = message;
        errorElement.style.color = type === 'error' ? '#ff8080' : '#80ff80';
        
        setTimeout(() => {
            if (errorElement.textContent === message) {
                errorElement.textContent = '';
            }
        }, 3000);
    }

    processAudio() {
        if (!this.audio.ready) return;

        this.audio.analyser.getByteFrequencyData(this.audio.frequencyData);

        // Calculate frequency band averages
        const bassEnd = Math.floor((SETTINGS.audio.bassThreshold / (this.audio.context.sampleRate / 2)) * this.audio.bufferLength);
        const midEnd = Math.floor((SETTINGS.audio.midThreshold / (this.audio.context.sampleRate / 2)) * this.audio.bufferLength);

        let bassSum = 0, midSum = 0, highSum = 0;
        let bassCount = 0, midCount = 0, highCount = 0;

        for (let i = 0; i < this.audio.bufferLength; i++) {
            const value = this.audio.frequencyData[i];
            
            if (i < bassEnd) {
                bassSum += value;
                bassCount++;
            } else if (i < midEnd) {
                midSum += value;
                midCount++;
            } else {
                highSum += value;
                highCount++;
            }
        }

        this.audio.bassAverage = bassCount > 0 ? bassSum / bassCount : 0;
        this.audio.midAverage = midCount > 0 ? midSum / midCount : 0;
        this.audio.highAverage = highCount > 0 ? highSum / highCount : 0;
        this.audio.overallLevel = (this.audio.bassAverage + this.audio.midAverage + this.audio.highAverage) / 3;

        // Beat detection
        const beatDetection = this.audio.beatDetection;
        beatDetection.bassHistory.push(this.audio.bassAverage);
        if (beatDetection.bassHistory.length > 10) {
            beatDetection.bassHistory.shift();
        }

        const avgBass = beatDetection.bassHistory.reduce((a, b) => a + b, 0) / beatDetection.bassHistory.length;
        beatDetection.beatThreshold = avgBass * SETTINGS.audio.beatDetection.thresholdMultiplier;

        const now = Date.now();
        if (this.audio.bassAverage > beatDetection.beatThreshold && now - beatDetection.lastBeat > 100) {
            beatDetection.lastBeat = now;
            beatDetection.beatFlash = 1.0;
        }

        beatDetection.beatFlash *= SETTINGS.audio.beatDetection.decay;
    }

    updateVisuals() {
        const time = this.clock.getElapsedTime();
        
        // Update shader uniforms
        this.material.uniforms.u_time.value = time * SETTINGS.blob.noiseSpeed;
        
        if (this.audio.ready) {
            // Audio-reactive parameters
            const bassNormalized = this.audio.bassAverage / 255;
            const midNormalized = this.audio.midAverage / 255;
            const highNormalized = this.audio.highAverage / 255;
            
            // Scale blob based on bass
            const scale = THREE.MathUtils.lerp(
                SETTINGS.audio.bassScale.min,
                SETTINGS.audio.bassScale.max,
                bassNormalized
            );
            this.blob.scale.setScalar(scale);
            
            // Displacement based on mids
            this.material.uniforms.u_audio_displacement.value = midNormalized * SETTINGS.audio.midDisplacement;
            
            // Color speed based on highs
            this.material.uniforms.u_color_speed.value = SETTINGS.blob.colorSpeed * (1 + highNormalized * SETTINGS.audio.highColorSpeed);
            
            // Beat flash
            this.material.uniforms.u_beat_flash.value = this.audio.beatDetection.beatFlash;
            
            // Camera dolly
            const dollyAmount = Math.sin(time * 0.5) * SETTINGS.camera.dollyDistance * bassNormalized;
            this.camera.position.z = 12 + dollyAmount;
        }
        
        // Update UI
        this.updateUI();
    }

    updateUI() {
        // FPS counter
        const now = performance.now();
        this.performance.frameCount++;
        
        if (now - this.performance.lastTime >= 1000) {
            this.performance.fps = Math.round((this.performance.frameCount * 1000) / (now - this.performance.lastTime));
            this.performance.frameCount = 0;
            this.performance.lastTime = now;
            
            const fpsElement = document.getElementById('fps-counter');
            if (fpsElement) {
                fpsElement.textContent = this.performance.fps;
            }
        }
        
        // Audio level bar
        if (this.audio.ready) {
            const levelBar = document.getElementById('level-bar');
            if (levelBar) {
                const level = (this.audio.overallLevel / 255) * 100;
                levelBar.style.width = `${level}%`;
            }
        }
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.composer.setSize(window.innerWidth, window.innerHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        this.processAudio();
        this.updateVisuals();
        this.controls.update();
        
        // Use bloom composer if enabled, otherwise regular renderer
        if (this.bloomPass.enabled) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }
}

// Initialize the app when the page loads
new App();
