import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Mic, FileAudio, Pause, Play } from 'lucide-react';

// Settings for the audio-reactive blob
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
    bassThreshold: 150,
    midThreshold: 2000,
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

// Vertex shader for the liquid blob
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

// Fragment shader for the liquid blob
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

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const sceneRef = useRef<any>(null);
  const audioRef = useRef<any>(null);
  const htmlAudioRef = useRef<HTMLAudioElement | null>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentFile, setCurrentFile] = useState<string>('');
  const [message, setMessage] = useState('');
  const [fps, setFps] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [isMicrophone, setIsMicrophone] = useState(false);

  const displayMessage = (msg: string, isError = false) => {
    setMessage(msg);
    setTimeout(() => setMessage(''), 3000);
  };

  // Initialize Three.js scene
  useEffect(() => {
    if (!canvasRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 12;
    
    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: true,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.enableZoom = false;
    
    // Create the blob geometry and material
    const geometry = new THREE.IcosahedronGeometry(4, 64);
    const material = new THREE.ShaderMaterial({
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
    
    const blob = new THREE.Mesh(geometry, material);
    scene.add(blob);
    
    // Post-processing setup
    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      1.5, 0.4, 0.85
    );
    bloomPass.threshold = SETTINGS.postProcessing.bloomThreshold;
    bloomPass.strength = SETTINGS.postProcessing.bloomStrength;
    bloomPass.radius = SETTINGS.postProcessing.bloomRadius;
    
    const composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);
    
    const clock = new THREE.Clock();
    
    // Store references
    sceneRef.current = {
      scene,
      camera,
      renderer,
      controls,
      composer,
      bloomPass,
      material,
      blob,
      clock,
    };

    // Initialize audio storage
    audioRef.current = {
      ready: false,
      context: null,
      source: null,
      analyser: null,
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

    // Animation loop
    let frameCount = 0;
    let lastTime = 0;
    
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      
      const time = clock.getElapsedTime();
      
      // Update shader uniforms
      material.uniforms.u_time.value = time * SETTINGS.blob.noiseSpeed;
      
      // Audio processing
      if (audioRef.current.ready && audioRef.current.analyser) {
        processAudio();
        
        const bassNormalized = audioRef.current.bassAverage / 255;
        const midNormalized = audioRef.current.midAverage / 255;
        const highNormalized = audioRef.current.highAverage / 255;
        
        // Scale blob based on bass
        const scale = THREE.MathUtils.lerp(
          SETTINGS.audio.bassScale.min,
          SETTINGS.audio.bassScale.max,
          bassNormalized
        );
        blob.scale.setScalar(scale);
        
        // Displacement based on mids
        material.uniforms.u_audio_displacement.value = midNormalized * SETTINGS.audio.midDisplacement;
        
        // Color speed based on highs
        material.uniforms.u_color_speed.value = SETTINGS.blob.colorSpeed * (1 + highNormalized * SETTINGS.audio.highColorSpeed);
        
        // Beat flash
        material.uniforms.u_beat_flash.value = audioRef.current.beatDetection.beatFlash;
        
        // Camera dolly
        const dollyAmount = Math.sin(time * 0.5) * SETTINGS.camera.dollyDistance * bassNormalized;
        camera.position.z = 12 + dollyAmount;
        
        // Update UI
        setAudioLevel(audioRef.current.overallLevel);
      }
      
      controls.update();
      composer.render();
      
      // FPS calculation
      frameCount++;
      const now = performance.now();
      if (now - lastTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - lastTime)));
        frameCount = 0;
        lastTime = now;
      }
    };
    
    animate();

    // Handle window resize
    const handleResize = () => {
      if (!sceneRef.current) return;
      
      const { camera, renderer, composer } = sceneRef.current;
      
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      composer.setSize(window.innerWidth, window.innerHeight);
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (sceneRef.current) {
        sceneRef.current.renderer.dispose();
      }
    };
  }, []);

  // Process audio data
  const processAudio = () => {
    if (!audioRef.current.ready || !audioRef.current.analyser || !audioRef.current.frequencyData) return;

    audioRef.current.analyser.getByteFrequencyData(audioRef.current.frequencyData);

    // Calculate frequency band averages
    const bassEnd = Math.floor((SETTINGS.audio.bassThreshold / (audioRef.current.context.sampleRate / 2)) * audioRef.current.bufferLength);
    const midEnd = Math.floor((SETTINGS.audio.midThreshold / (audioRef.current.context.sampleRate / 2)) * audioRef.current.bufferLength);

    let bassSum = 0, midSum = 0, highSum = 0;
    let bassCount = 0, midCount = 0, highCount = 0;

    for (let i = 0; i < audioRef.current.bufferLength; i++) {
      const value = audioRef.current.frequencyData[i];
      
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

    audioRef.current.bassAverage = bassCount > 0 ? bassSum / bassCount : 0;
    audioRef.current.midAverage = midCount > 0 ? midSum / midCount : 0;
    audioRef.current.highAverage = highCount > 0 ? highSum / highCount : 0;
    audioRef.current.overallLevel = (audioRef.current.bassAverage + audioRef.current.midAverage + audioRef.current.highAverage) / 3;

    // Beat detection
    const beatDetection = audioRef.current.beatDetection;
    beatDetection.bassHistory.push(audioRef.current.bassAverage);
    if (beatDetection.bassHistory.length > 10) {
      beatDetection.bassHistory.shift();
    }

    const avgBass = beatDetection.bassHistory.reduce((a: number, b: number) => a + b, 0) / beatDetection.bassHistory.length;
    beatDetection.beatThreshold = avgBass * SETTINGS.audio.beatDetection.thresholdMultiplier;

    const now = Date.now();
    if (audioRef.current.bassAverage > beatDetection.beatThreshold && now - beatDetection.lastBeat > 100) {
      beatDetection.lastBeat = now;
      beatDetection.beatFlash = 1.0;
    }

    beatDetection.beatFlash *= SETTINGS.audio.beatDetection.decay;
  };

  // Start microphone input
  const startMicrophone = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setupAudioContext(stream);
      setIsPlaying(true);
      setIsMicrophone(true);
      setCurrentFile('Microphone');
      displayMessage('Microphone activated!');
    } catch (error) {
      displayMessage('Microphone access denied: ' + (error as Error).message, true);
    }
  };

  // Handle file upload with HTML5 Audio for playback
  const handleFileUpload = async (file: File) => {
    try {
      // Create HTML5 Audio element for playback
      const audio = new Audio();
      const url = URL.createObjectURL(file);
      audio.src = url;
      audio.loop = true;
      audio.volume = 0.7;
      
      // Wait for audio to be ready
      await new Promise((resolve, reject) => {
        audio.addEventListener('canplaythrough', resolve);
        audio.addEventListener('error', reject);
        audio.load();
      });
      
      // Start playing the audio
      await audio.play();
      
      // Set up Web Audio API for analysis
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaElementSource(audio);
      
      // Connect to destination for playback
      source.connect(audioContext.destination);
      
      // Set up audio analysis
      setupAudioContext(null, audioContext, source);
      
      htmlAudioRef.current = audio;
      setIsPlaying(true);
      setIsMicrophone(false);
      setIsPaused(false);
      setCurrentFile(file.name);
      displayMessage(`Playing: ${file.name}`);
    } catch (error) {
      displayMessage('Error loading audio file: ' + (error as Error).message, true);
    }
  };

  // Set up audio context for analysis
  const setupAudioContext = (stream: MediaStream | null, audioContext?: AudioContext, source?: MediaElementAudioSourceNode) => {
    const context = audioContext || new (window.AudioContext || (window as any).webkitAudioContext)();
    
    let audioSource: MediaStreamAudioSourceNode | MediaElementAudioSourceNode;
    if (stream) {
      audioSource = context.createMediaStreamSource(stream);
    } else if (source) {
      audioSource = source;
    } else {
      return;
    }

    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;
    
    // Connect audio source to analyser for visualization
    audioSource.connect(analyser);
    
    const bufferLength = analyser.frequencyBinCount;
    const frequencyData = new Uint8Array(bufferLength);
    
    audioRef.current.ready = true;
    audioRef.current.context = context;
    audioRef.current.source = audioSource;
    audioRef.current.analyser = analyser;
    audioRef.current.frequencyData = frequencyData;
    audioRef.current.bufferLength = bufferLength;
  };

  // Toggle playback for file audio
  const togglePlayback = () => {
    if (htmlAudioRef.current && !isMicrophone) {
      if (isPaused) {
        htmlAudioRef.current.play();
        setIsPaused(false);
      } else {
        htmlAudioRef.current.pause();
        setIsPaused(true);
      }
    }
  };

  // Stop audio and reset
  const stopAudio = () => {
    if (htmlAudioRef.current) {
      htmlAudioRef.current.pause();
      htmlAudioRef.current.currentTime = 0;
      URL.revokeObjectURL(htmlAudioRef.current.src);
      htmlAudioRef.current = null;
    }
    
    audioRef.current.ready = false;
    setIsPlaying(false);
    setIsPaused(false);
    setIsMicrophone(false);
    setCurrentFile('');
    setAudioLevel(0);
  };

  // File input handler
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Stop current audio if playing
      if (isPlaying) {
        stopAudio();
      }
      handleFileUpload(file);
    }
  };

  return (
    <div className="w-full h-screen relative bg-black overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full" />
      
      {/* UI Controls */}
      <div className="absolute top-4 left-4 z-10">
        <Card className="w-80 bg-black/70 border-white/20 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white text-lg">Audio-Reactive Liquid Blob</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {!isPlaying ? (
              <div className="space-y-3">
                <Button 
                  onClick={startMicrophone}
                  className="w-full bg-blue-600 hover:bg-blue-700"
                >
                  <Mic className="w-4 h-4 mr-2" />
                  Use Microphone
                </Button>
                
                <div className="relative">
                  <Button 
                    onClick={() => document.getElementById('file-input')?.click()}
                    className="w-full bg-purple-600 hover:bg-purple-700"
                  >
                    <FileAudio className="w-4 h-4 mr-2" />
                    Load Audio File
                  </Button>
                  <input
                    id="file-input"
                    type="file"
                    accept=".mp3,.wav,.ogg,.m4a"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </div>
                
                <p className="text-gray-400 text-sm text-center">
                  Upload music to see the blob dance to your beats!
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="text-white">
                  <p className="text-sm font-medium mb-2">Now Playing:</p>
                  <p className="text-xs text-gray-300 truncate">{currentFile}</p>
                </div>
                
                <div className="space-y-2">
                  <p className="text-white text-sm">Audio Level</p>
                  <div className="w-full h-2 bg-gray-700 rounded overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-cyan-500 to-magenta-500 transition-all duration-75"
                      style={{ width: `${(audioLevel / 255) * 100}%` }}
                    />
                  </div>
                </div>
                
                <div className="flex gap-2">
                  {!isMicrophone && (
                    <Button
                      onClick={togglePlayback}
                      className="flex-1 bg-green-600 hover:bg-green-700"
                    >
                      {isPaused ? <Play className="w-4 h-4 mr-2" /> : <Pause className="w-4 h-4 mr-2" />}
                      {isPaused ? 'Play' : 'Pause'}
                    </Button>
                  )}
                  
                  <Button
                    onClick={stopAudio}
                    className="flex-1 bg-red-600 hover:bg-red-700"
                  >
                    Stop
                  </Button>
                </div>
                
                <div className="text-white text-sm">
                  FPS: {fps}
                </div>
              </div>
            )}
            
            {message && (
              <div className="text-sm p-2 rounded text-green-400 bg-green-900/20">
                {message}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}