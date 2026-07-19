"use client";

import {
  Activity,
  AlertTriangle,
  CircleStop,
  Download,
  Gauge,
  HardDrive,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Sparkles,
  Square,
  Tag,
  Usb,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8765";
const CHANNEL_COLORS = [
  "#67e8c5",
  "#70b7ff",
  "#a78bfa",
  "#f0abfc",
  "#fbbf72",
  "#fb7185",
  "#a3e635",
  "#22d3ee",
];

type QualityState = "waiting" | "present" | "flat" | "saturated" | "unstable";

type Quality = {
  state: QualityState;
  rmsMv: number;
  peakToPeakMv: number;
  clippingPercent: number;
  dcOffsetMv: number;
};

type DashboardStatus = {
  device: {
    available: boolean;
    name: string;
    vendorId: string;
    productId: string;
    transport: string;
    channels: number;
    reportBytes: number;
  };
  acquisition: {
    running: boolean;
    sessionId: string | null;
    startedAt: string | null;
    durationSeconds: number;
    nominalFs: number;
    measuredFs: number;
    reports: number;
    samples: number;
    delayedReportPeriodsEstimate: number;
    lastReportAgeSeconds: number | null;
    error: string | null;
    units: "mV_ADC_filtered";
  };
  signal: {
    displayUnit: string;
    conversionVoltsPerCount: number;
    rawEncoding: string;
    pipeline: string;
    bandpassHz: [number, number];
    bandpassOrder: number;
    notchHz: number;
    notchQ: number;
    metricWindowSeconds: number;
    pgaGainIndex: number;
    pgaGain: number;
    electrodeUvCalibrated: boolean;
  };
  recording: {
    active: boolean;
    filename: string | null;
    rows: number;
    durationSeconds: number;
  };
  quality: Quality[];
};

type SampleRow = { sequence: number; elapsed: number; values: number[] };
type Recording = { filename: string; bytes: number; modifiedAt: string; downloadUrl: string };

const EMPTY_QUALITY: Quality = {
  state: "waiting",
  rmsMv: 0,
  peakToPeakMv: 0,
  clippingPercent: 0,
  dcOffsetMv: 0,
};

function formatDuration(seconds: number) {
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  return `${String(minutes).padStart(2, "0")}:${String(safe % 60).padStart(2, "0")}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function qualityLabel(state: QualityState) {
  return {
    waiting: "대기",
    present: "신호 있음",
    flat: "평탄",
    saturated: "포화",
    unstable: "불안정",
  }[state];
}

function apiError(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

async function apiRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `API 오류 (${response.status})`);
  return payload as T;
}

function WaveformCanvas({
  rows,
  visible,
  selected,
  windowSeconds,
  fixedScale,
  renderDelayMs,
  paused,
}: {
  rows: SampleRow[];
  visible: boolean[];
  selected: number;
  windowSeconds: number;
  fixedScale: number;
  renderDelayMs: number;
  paused: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rowsRef = useRef(rows);
  const settingsRef = useRef({ visible, selected, windowSeconds, fixedScale, renderDelayMs, paused });
  const playheadRef = useRef<number | null>(null);
  const previousFrameRef = useRef<number | null>(null);

  useEffect(() => {
    rowsRef.current = rows;
  }, [rows]);

  useEffect(() => {
    settingsRef.current = { visible, selected, windowSeconds, fixedScale, renderDelayMs, paused };
  }, [visible, selected, windowSeconds, fixedScale, renderDelayMs, paused]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let animationFrame = 0;
    let canvasWidth = 0;
    let canvasHeight = 0;

    const draw = (frameTime: number) => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, bounds.width);
      const height = Math.max(1, bounds.height);
      const nextCanvasWidth = Math.floor(width * ratio);
      const nextCanvasHeight = Math.floor(height * ratio);
      if (canvasWidth !== nextCanvasWidth || canvasHeight !== nextCanvasHeight) {
        canvasWidth = nextCanvasWidth;
        canvasHeight = nextCanvasHeight;
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        animationFrame = requestAnimationFrame(draw);
        return;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#07100f";
      ctx.fillRect(0, 0, width, height);

      const currentRows = rowsRef.current;
      const settings = settingsRef.current;
      const left = 88;
      const right = 16;
      const top = 16;
      const bottom = 28;
      const plotWidth = width - left - right;
      const plotHeight = height - top - bottom;
      const shown = settings.visible.flatMap((on, index) => (on ? [index] : []));

      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(133, 168, 158, .1)";
      for (let i = 0; i <= 10; i += 1) {
        const x = left + (plotWidth * i) / 10;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, top + plotHeight);
        ctx.stroke();
      }

      if (!shown.length) {
        ctx.fillStyle = "#78928a";
        ctx.font = "13px var(--font-geist-sans)";
        ctx.textAlign = "center";
        ctx.fillText("표시할 채널을 선택하세요", width / 2, height / 2);
        animationFrame = requestAnimationFrame(draw);
        return;
      }

      const laneHeight = plotHeight / shown.length;
      shown.forEach((channel, lane) => {
        const mid = top + laneHeight * (lane + 0.5);
        const amplitude = laneHeight * 0.4;
        ctx.strokeStyle = channel === settings.selected ? "rgba(103, 232, 197, .22)" : "rgba(133, 168, 158, .08)";
        ctx.beginPath();
        ctx.moveTo(left, mid);
        ctx.lineTo(width - right, mid);
        ctx.stroke();
        ctx.fillStyle = CHANNEL_COLORS[channel];
        ctx.font = `${channel === settings.selected ? 650 : 500} 11px var(--font-geist-mono)`;
        ctx.textAlign = "left";
        ctx.fillText(`CH ${channel + 1}`, 8, mid + 4);
        ctx.fillStyle = "#607870";
        ctx.font = "8px var(--font-geist-mono)";
        ctx.textAlign = "right";
        ctx.fillText(`+${settings.fixedScale.toFixed(2)}`, left - 5, mid - amplitude + 3);
        ctx.fillText("0", left - 5, mid + 3);
        ctx.fillText(`−${settings.fixedScale.toFixed(2)}`, left - 5, mid + amplitude + 3);
      });
      ctx.fillStyle = "#78928a";
      ctx.font = "8px var(--font-geist-mono)";
      ctx.textAlign = "right";
      ctx.fillText("mV ADC", left - 5, 9);

      if (!currentRows.length) {
        playheadRef.current = null;
        previousFrameRef.current = frameTime;
        ctx.fillStyle = "#78928a";
        ctx.font = "13px var(--font-geist-sans)";
        ctx.textAlign = "center";
        ctx.fillText("측정을 시작하면 필터링된 EEG 파형이 표시됩니다", left + plotWidth / 2, height / 2);
        animationFrame = requestAnimationFrame(draw);
        return;
      }

      const firstAvailable = currentRows[0].elapsed;
      const latestAvailable = currentRows[currentRows.length - 1].elapsed;
      const renderDelay = settings.renderDelayMs / 1000;
      const bufferedDuration = latestAvailable - firstAvailable;
      if (bufferedDuration < renderDelay) {
        playheadRef.current = null;
        previousFrameRef.current = frameTime;
        ctx.fillStyle = "#78928a";
        ctx.font = "13px var(--font-geist-sans)";
        ctx.textAlign = "center";
        ctx.fillText(`부드러운 표시 버퍼 준비 중 · ${Math.round((bufferedDuration / renderDelay) * 100)}%`, left + plotWidth / 2, height / 2);
        animationFrame = requestAnimationFrame(draw);
        return;
      }

      const frameDelta = previousFrameRef.current === null
        ? 0
        : Math.min(0.1, Math.max(0, (frameTime - previousFrameRef.current) / 1000));
      previousFrameRef.current = frameTime;
      const initialPlayhead = latestAvailable - renderDelay;
      const playheadIsOutsideBuffer = playheadRef.current === null
        || playheadRef.current > latestAvailable
        || latestAvailable - playheadRef.current > Math.max(2, settings.windowSeconds);
      if (playheadIsOutsideBuffer) {
        playheadRef.current = initialPlayhead;
      } else if (!settings.paused) {
        playheadRef.current = Math.min(latestAvailable - 0.025, playheadRef.current + frameDelta);
      }

      const newest = playheadRef.current ?? initialPlayhead;
      playheadRef.current = newest;
      const start = newest - settings.windowSeconds;
      const points = currentRows.filter((row) => row.elapsed >= start && row.elapsed <= newest);
      shown.forEach((channel, lane) => {
        if (points.length < 2) return;
        const values = points.map((row) => row.values[channel]);
        const scale = settings.fixedScale;
        const mid = top + laneHeight * (lane + 0.5);
        const amplitude = laneHeight * 0.39;
        ctx.strokeStyle = CHANNEL_COLORS[channel];
        ctx.lineWidth = channel === settings.selected ? 1.45 : 1;
        ctx.globalAlpha = channel === settings.selected ? 1 : 0.78;
        ctx.beginPath();
        points.forEach((row, index) => {
          const x = left + ((row.elapsed - start) / settings.windowSeconds) * plotWidth;
          const normalized = Math.max(-1, Math.min(1, row.values[channel] / scale));
          const y = mid - normalized * amplitude;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
        if (values.some((value) => Math.abs(value) > scale)) {
          ctx.fillStyle = "#fb7185";
          ctx.font = "8px var(--font-geist-mono)";
          ctx.textAlign = "right";
          ctx.fillText("OVER", width - right - 3, mid - amplitude + 8);
        }
      });

      ctx.fillStyle = "#6f8981";
      ctx.font = "10px var(--font-geist-mono)";
      ctx.textAlign = "center";
      for (let i = 0; i <= 5; i += 1) {
        const seconds = settings.windowSeconds - (settings.windowSeconds * i) / 5;
        ctx.fillText(i === 5 ? "현재" : `-${seconds.toFixed(1)}s`, left + (plotWidth * i) / 5, height - 9);
      }
      animationFrame = requestAnimationFrame(draw);
    };
    animationFrame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animationFrame);
  }, []);

  return <canvas ref={canvasRef} className="waveform-canvas" role="img" aria-label="고정 축 8채널 EEG 실시간 필터 파형" />;
}

function calculateSpectrum(rows: SampleRow[], channel: number, fs: number) {
  const size = 256;
  if (rows.length < size || fs <= 0) return { bins: [] as { hz: number; db: number }[], bands: [] as { name: string; value: number }[] };
  const values = rows.slice(-size).map((row) => row.values[channel]);
  const mean = values.reduce((sum, value) => sum + value, 0) / size;
  const bins: { hz: number; db: number }[] = [];
  const bandDefs = [
    ["Delta", 0.5, 4],
    ["Theta", 4, 8],
    ["Alpha", 8, 13],
    ["Beta", 13, 30],
    ["Gamma", 30, 45],
  ] as const;
  const sums = bandDefs.map(() => 0);
  const windowPower = Array.from({ length: size }, (_, n) => {
    const weight = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (size - 1));
    return weight * weight;
  }).reduce((sum, value) => sum + value, 0);
  const df = fs / size;
  for (let k = 1; k < size / 2; k += 1) {
    const hz = (k * fs) / size;
    if (hz > 45) break;
    let real = 0;
    let imaginary = 0;
    for (let n = 0; n < size; n += 1) {
      const windowed = (values[n] - mean) * (0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (size - 1)));
      const angle = (2 * Math.PI * k * n) / size;
      real += windowed * Math.cos(angle);
      imaginary -= windowed * Math.sin(angle);
    }
    const psd = (2 * (real * real + imaginary * imaginary)) / (fs * windowPower);
    bins.push({ hz, db: 10 * Math.log10(Math.max(psd, 1e-12)) });
    bandDefs.forEach(([, low, high], index) => {
      if (hz >= low && hz < high) sums[index] += psd * df;
    });
  }
  const total = sums.reduce((sum, value) => sum + value, 0) || 1;
  return {
    bins,
    bands: bandDefs.map(([name], index) => ({ name, value: (sums[index] / total) * 100 })),
  };
}

function SpectrumCanvas({ bins }: { bins: { hz: number; db: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(bounds.width * ratio);
      canvas.height = Math.floor(bounds.height * ratio);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, bounds.width, bounds.height);
      const left = 42;
      const bottom = 24;
      const top = 12;
      const width = bounds.width - left - 8;
      const height = bounds.height - top - bottom;
      ctx.strokeStyle = "rgba(133, 168, 158, .12)";
      ctx.lineWidth = 1;
      [0, 10, 20, 30, 40].forEach((hz) => {
        const x = left + (hz / 45) * width;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, top + height);
        ctx.stroke();
        ctx.fillStyle = "#6f8981";
        ctx.font = "9px var(--font-geist-mono)";
        ctx.textAlign = "center";
        ctx.fillText(String(hz), x, bounds.height - 7);
      });
      const dbMin = -80;
      const dbMax = 40;
      [-80, -40, 0, 40].forEach((db) => {
        const y = top + ((dbMax - db) / (dbMax - dbMin)) * height;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(left + width, y);
        ctx.stroke();
        ctx.fillStyle = "#6f8981";
        ctx.textAlign = "right";
        ctx.fillText(String(db), left - 4, y + 3);
      });
      if (!bins.length) {
        ctx.fillStyle = "#78928a";
        ctx.font = "12px var(--font-geist-sans)";
        ctx.textAlign = "center";
        ctx.fillText("스펙트럼 계산 대기", left + width / 2, top + height / 2);
        return;
      }
      ctx.beginPath();
      bins.forEach((bin, index) => {
        const x = left + (bin.hz / 45) * width;
        const normalized = (Math.max(dbMin, Math.min(dbMax, bin.db)) - dbMin) / (dbMax - dbMin);
        const y = top + height - normalized * height;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = "#67e8c5";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      const gradient = ctx.createLinearGradient(0, top, 0, top + height);
      gradient.addColorStop(0, "rgba(103, 232, 197, .18)");
      gradient.addColorStop(1, "rgba(103, 232, 197, 0)");
      ctx.lineTo(left + width, top + height);
      ctx.lineTo(left, top + height);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();
      ctx.fillStyle = "#6f8981";
      ctx.textAlign = "right";
      ctx.fillText("Hz", bounds.width - 8, bounds.height - 7);
      ctx.save();
      ctx.translate(9, top + height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText("dB(mV²/Hz)", 0, 0);
      ctx.restore();
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [bins]);
  return <canvas ref={canvasRef} className="spectrum-canvas" role="img" aria-label="선택 채널의 고정 dB 축 파워 스펙트럼 밀도" />;
}

export default function Home() {
  const [status, setStatus] = useState<DashboardStatus | null>(null);
  const [rows, setRows] = useState<SampleRow[]>([]);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [selectedChannel, setSelectedChannel] = useState(0);
  const [visible, setVisible] = useState(() => Array(8).fill(true));
  const [windowSeconds, setWindowSeconds] = useState(5);
  const [fixedScale, setFixedScale] = useState(100);
  const [gainIndex, setGainIndex] = useState(6);
  const [renderDelayMs, setRenderDelayMs] = useState(450);
  const [displayPaused, setDisplayPaused] = useState(false);
  const [recordLabel, setRecordLabel] = useState("");
  const [customMarker, setCustomMarker] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [markers, setMarkers] = useState<{ label: string; time: string }[]>([]);
  const sequenceRef = useRef(0);
  const sessionRef = useRef<string | null>(null);
  const inFlightRef = useRef(false);

  const refreshStatus = useCallback(async () => {
    try {
      const next = await apiRequest<DashboardStatus>("/api/status");
      setStatus(next);
      setApiOnline(true);
      if (sessionRef.current !== next.acquisition.sessionId) {
        sessionRef.current = next.acquisition.sessionId;
        sequenceRef.current = 0;
        setRows([]);
        setMarkers([]);
      }
    } catch {
      setApiOnline(false);
    }
  }, []);

  const refreshRecordings = useCallback(async () => {
    try {
      const payload = await apiRequest<{ recordings: Recording[] }>("/api/recordings");
      setRecordings(payload.recordings);
    } catch {
      // 상태 표시가 API 오프라인을 별도로 알린다.
    }
  }, []);

  useEffect(() => {
    const initialTimer = window.setTimeout(() => {
      void refreshStatus();
      void refreshRecordings();
    }, 0);
    const statusTimer = window.setInterval(refreshStatus, 1000);
    const recordingsTimer = window.setInterval(refreshRecordings, 5000);
    return () => {
      window.clearTimeout(initialTimer);
      window.clearInterval(statusTimer);
      window.clearInterval(recordingsTimer);
    };
  }, [refreshRecordings, refreshStatus]);

  useEffect(() => {
    if (!status?.acquisition.running) return;
    const pull = async () => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      try {
        const payload = await apiRequest<{
          sessionId: string;
          reset: boolean;
          latestSequence: number;
          rows: SampleRow[];
        }>(`/api/data?after=${sequenceRef.current}&limit=2048`);
        if (payload.reset) setRows([]);
        if (payload.rows.length) {
          sequenceRef.current = payload.rows[payload.rows.length - 1].sequence;
          if (!displayPaused) {
            setRows((previous) => [...previous, ...payload.rows].slice(-7000));
          }
        }
      } catch (error) {
        setMessage(apiError(error));
      } finally {
        inFlightRef.current = false;
      }
    };
    pull();
    const timer = window.setInterval(pull, 80);
    return () => window.clearInterval(timer);
  }, [displayPaused, status?.acquisition.running]);

  useEffect(() => {
    if (!message) return;
    const timer = window.setTimeout(() => setMessage(null), 6000);
    return () => window.clearTimeout(timer);
  }, [message]);

  const perform = async (path: string, body: Record<string, unknown> = {}) => {
    setBusy(true);
    try {
      await apiRequest(path, { method: "POST", body: JSON.stringify(body) });
      await refreshStatus();
    } catch (error) {
      setMessage(apiError(error));
    } finally {
      setBusy(false);
    }
  };

  const toggleChannel = (index: number) => {
    setVisible((previous) => previous.map((value, channel) => (channel === index ? !value : value)));
  };

  const toggleDisplayPause = () => {
    if (displayPaused) setRows([]);
    setDisplayPaused(!displayPaused);
  };

  const addMarker = async (label: string) => {
    const normalized = label.trim();
    if (!normalized) return;
    setBusy(true);
    try {
      await apiRequest("/api/marker", { method: "POST", body: JSON.stringify({ label: normalized }) });
      setMarkers((previous) => [{ label: normalized, time: new Date().toLocaleTimeString("ko-KR") }, ...previous].slice(0, 8));
      setCustomMarker("");
    } catch (error) {
      setMessage(apiError(error));
    } finally {
      setBusy(false);
    }
  };

  const fs = status?.acquisition.measuredFs || status?.acquisition.nominalFs || 256;
  const spectrum = useMemo(() => calculateSpectrum(rows, selectedChannel, fs), [rows, selectedChannel, fs]);
  const quality = status?.quality[selectedChannel] ?? EMPTY_QUALITY;
  const isRunning = Boolean(status?.acquisition.running);
  const isRecording = Boolean(status?.recording.active);
  const deviceReady = Boolean(status?.device.available);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><Activity size={18} strokeWidth={2.2} /></div>
          <div>
            <p className="eyebrow">brainToArm / acquisition</p>
            <h1>PolyG-I Live Monitor</h1>
          </div>
        </div>
        <div className="header-actions">
          <div className={`connection-pill ${apiOnline && deviceReady ? "connected" : ""}`}>
            <span className="status-dot" />
            {!apiOnline ? "로컬 API 오프라인" : deviceReady ? "PolyG-I 연결됨" : "장치 미감지"}
          </div>
          {!isRunning ? (
            <button className="primary-button" disabled={busy || !apiOnline || !deviceReady} onClick={() => perform("/api/acquisition/start", { gainIndex })}>
              <Play size={16} fill="currentColor" /> 측정 시작
            </button>
          ) : (
            <button className="danger-button" disabled={busy} onClick={() => perform("/api/acquisition/stop")}>
              <CircleStop size={16} /> 측정 정지
            </button>
          )}
        </div>
      </header>

      {message && <div className="toast" role="alert"><AlertTriangle size={16} />{message}</div>}
      {!apiOnline && (
        <div className="offline-banner">
          <AlertTriangle size={18} />
          <div><strong>수집 서비스에 연결할 수 없습니다.</strong><span><code>python3 laptop/eeg_dashboard.py</code>로 로컬 서비스를 시작하세요.</span></div>
          <button className="icon-button" aria-label="다시 연결" onClick={refreshStatus}><RefreshCw size={16} /></button>
        </div>
      )}

      <section className="metric-grid" aria-label="측정 요약">
        <article className="metric-card">
          <div className="metric-icon"><Radio size={17} /></div>
          <div><span>수집 상태</span><strong className={isRunning ? "accent-text" : ""}>{isRunning ? "LIVE" : "STOPPED"}</strong></div>
          <small>{isRunning ? formatDuration(status?.acquisition.durationSeconds ?? 0) : "측정 대기"}</small>
        </article>
        <article className="metric-card">
          <div className="metric-icon"><Gauge size={17} /></div>
          <div><span>실측 샘플링</span><strong>{status?.acquisition.measuredFs ? `${status.acquisition.measuredFs.toFixed(1)} Hz` : "—"}</strong></div>
          <small>설정 {status?.acquisition.nominalFs ?? 256} Hz · D1WD10</small>
        </article>
        <article className="metric-card">
          <div className="metric-icon"><HardDrive size={17} /></div>
          <div><span>수신 샘플</span><strong>{(status?.acquisition.samples ?? 0).toLocaleString()}</strong></div>
          <small>긴 수신 간격 +{status?.acquisition.delayedReportPeriodsEstimate ?? 0} periods · 누락 단정 안 함</small>
        </article>
        <article className={`metric-card ${isRecording ? "recording" : ""}`}>
          <div className="metric-icon"><Square size={15} fill={isRecording ? "currentColor" : "none"} /></div>
          <div><span>CSV 기록</span><strong>{isRecording ? "RECORDING" : "OFF"}</strong></div>
          <small>{isRecording ? `${status?.recording.rows.toLocaleString()} 행 · ${formatDuration(status?.recording.durationSeconds ?? 0)}` : "세션 기록 안 함"}</small>
        </article>
      </section>

      <section className="dashboard-grid">
        <article className="panel waveform-panel">
          <div className="panel-header waveform-header">
            <div>
              <p className="panel-kicker"><span className={isRunning ? "live-dot" : "idle-dot"} />EEG 0.5–45 HZ · NOTCH 60 HZ</p>
              <h2>실시간 EEG 파형 · 공통 고정 축</h2>
            </div>
            <div className="toolbar">
              <label>렌더링
                <select value={renderDelayMs} onChange={(event) => { setRenderDelayMs(Number(event.target.value)); setRows([]); }}>
                  <option value={450}>부드럽게 · 0.45초</option><option value={80}>저지연 · 0.08초</option>
                </select>
              </label>
              <label>시간창
                <select value={windowSeconds} onChange={(event) => setWindowSeconds(Number(event.target.value))}>
                  <option value={2}>2초</option><option value={5}>5초</option><option value={10}>10초</option>
                </select>
              </label>
              <label>공통 Y축
                <select value={fixedScale} onChange={(event) => setFixedScale(Number(event.target.value))}>
                  <option value={0.1}>±0.10 mV</option><option value={0.25}>±0.25 mV</option><option value={0.5}>±0.50 mV</option><option value={1}>±1.00 mV</option><option value={2.5}>±2.50 mV</option><option value={5}>±5.00 mV</option><option value={10}>±10.00 mV</option><option value={25}>±25.00 mV</option><option value={50}>±50.00 mV</option><option value={100}>±100.00 mV</option><option value={250}>±250.00 mV</option><option value={500}>±500.00 mV</option><option value={1000}>±1000.00 mV</option><option value={2500}>±2500.00 mV</option><option value={5000}>±5000.00 mV</option>
                </select>
              </label>
              <label>EEG PGA
                <select value={gainIndex} disabled={isRunning} onChange={(event) => setGainIndex(Number(event.target.value))}>
                  <option value={4}>index 4 · ×1.00</option><option value={5}>index 5 · ×1.36</option><option value={6}>index 6 · ×1.70</option><option value={7}>index 7 · ×2.55</option><option value={8}>index 8 · ×3.40</option><option value={9}>index 9 · ×4.25</option><option value={10}>index 10 · ×5.67</option><option value={11}>index 11 · ×6.80</option><option value={12}>index 12 · ×8.50</option><option value={13}>index 13 · ×10.20</option><option value={14}>index 14 · ×11.90</option><option value={15}>index 15 · ×17.00</option>
                </select>
              </label>
              <button className={`secondary-button ${displayPaused ? "active" : ""}`} onClick={toggleDisplayPause} disabled={!isRunning}>
                {displayPaused ? <Play size={15} /> : <Pause size={15} />}{displayPaused ? "보기 재개" : "보기 일시정지"}
              </button>
            </div>
          </div>
          <WaveformCanvas rows={rows} visible={visible} selected={selectedChannel} windowSeconds={windowSeconds} fixedScale={fixedScale} renderDelayMs={renderDelayMs} paused={displayPaused} />
          <div className="channel-controls" aria-label="채널 표시 설정">
            {visible.map((on, index) => (
              <div className={`channel-control ${selectedChannel === index ? "selected" : ""}`} key={index}>
                <button className="channel-select" onClick={() => setSelectedChannel(index)} aria-label={`채널 ${index + 1} 분석`}>
                  <span className="channel-color" style={{ background: CHANNEL_COLORS[index] }} />CH {index + 1}
                </button>
                <label className="visibility-toggle"><input type="checkbox" checked={on} onChange={() => toggleChannel(index)} /><span>{on ? "표시" : "숨김"}</span></label>
              </div>
            ))}
          </div>
          <p className="data-note"><Sparkles size={13} />모든 채널은 같은 고정 Y축과 0 mV 기준선을 씁니다. 값은 D1WD10 전압계수로 환산한 ADC 입력 mV이며, 실시간 4차 0.5–45 Hz band-pass + 60 Hz notch 결과입니다.</p>
        </article>

        <aside className="analysis-column">
          <article className="panel spectrum-panel">
            <div className="panel-header compact">
              <div><p className="panel-kicker">PSD · CH {selectedChannel + 1}</p><h2>파워 스펙트럼 밀도</h2></div>
              <span className={`quality-badge ${quality.state}`}>{qualityLabel(quality.state)}</span>
            </div>
            <SpectrumCanvas bins={spectrum.bins} />
            <div className="band-list">
              {spectrum.bands.length ? spectrum.bands.map((band) => (
                <div className="band-row" key={band.name}><span>{band.name}</span><div><i style={{ width: `${Math.max(2, band.value)}%` }} /></div><strong>{band.value.toFixed(1)}%</strong></div>
              )) : ["Delta", "Theta", "Alpha", "Beta", "Gamma"].map((band) => (
                <div className="band-row muted" key={band}><span>{band}</span><div><i /></div><strong>—</strong></div>
              ))}
            </div>
            <p className="fine-print">256점 Hann 창 · one-sided PSD(mV²/Hz) · 고정 −80~40 dB축. 밴드는 0.5–45 Hz 총 파워 대비 비율이며 진단 지표가 아닙니다.</p>
          </article>

          <article className="panel quality-panel">
            <div className="panel-header compact"><div><p className="panel-kicker">SIGNAL CHECK</p><h2>채널 상태</h2></div></div>
            <div className="quality-list">
              {(status?.quality ?? Array(8).fill(EMPTY_QUALITY)).map((item, index) => (
                <button className={`quality-row ${selectedChannel === index ? "selected" : ""}`} key={index} onClick={() => setSelectedChannel(index)}>
                  <span className="channel-color" style={{ background: CHANNEL_COLORS[index] }} />
                  <strong>CH {index + 1}</strong>
                  <small><span>RMS {item.rmsMv.toFixed(3)} · p-p {item.peakToPeakMv.toFixed(3)} mV</span><span>clip {item.clippingPercent.toFixed(3)}% · raw DC {item.dcOffsetMv.toFixed(3)} mV</span></small>
                  <span className={`quality-state ${item.state}`}>{qualityLabel(item.state)}</span>
                </button>
              ))}
            </div>
            <p className="fine-print warning"><AlertTriangle size={13} />최근 2.0초 필터 출력의 RMS/p-p와 ADC rail 포화율을 계산합니다. 전극 임피던스 측정값은 아닙니다.</p>
          </article>
        </aside>
      </section>

      <section className="lower-grid">
        <article className="panel session-panel">
          <div className="panel-header compact"><div><p className="panel-kicker">SESSION LOG</p><h2>기록과 이벤트</h2></div>{isRecording && <span className="recording-indicator"><i />기록 중</span>}</div>
          <div className="recording-controls">
            <input value={recordLabel} onChange={(event) => setRecordLabel(event.target.value)} placeholder="기록 이름 (선택)" maxLength={60} disabled={isRecording} />
            {!isRecording ? (
              <button className="secondary-button" disabled={!isRunning || busy} onClick={() => perform("/api/recording/start", { label: recordLabel })}><Square size={14} fill="currentColor" />CSV 기록 시작</button>
            ) : (
              <button className="danger-button small" disabled={busy} onClick={async () => { await perform("/api/recording/stop"); await refreshRecordings(); }}><CircleStop size={15} />기록 종료</button>
            )}
          </div>
          <div className="marker-section">
            <div className="marker-input"><Tag size={15} /><input value={customMarker} onChange={(event) => setCustomMarker(event.target.value)} placeholder="이벤트 표식" maxLength={80} onKeyDown={(event) => { if (event.key === "Enter") addMarker(customMarker); }} /><button disabled={!isRecording || busy || !customMarker.trim()} onClick={() => addMarker(customMarker)}>추가</button></div>
            <div className="marker-presets">{["눈 깜빡임", "눈 감음", "휴식 시작", "과제 시작"].map((label) => <button key={label} disabled={!isRecording || busy} onClick={() => addMarker(label)}>+ {label}</button>)}</div>
          </div>
          <div className="event-feed">
            {markers.length ? markers.map((marker, index) => <div key={`${marker.time}-${index}`}><span>{marker.time}</span><strong>{marker.label}</strong></div>) : <p>기록 중 이벤트를 추가하면 CSV의 다음 샘플 행에 표식이 저장됩니다.</p>}
          </div>
        </article>

        <article className="panel recordings-panel">
          <div className="panel-header compact"><div><p className="panel-kicker">LOCAL ARCHIVE</p><h2>최근 기록</h2></div><button className="icon-button" aria-label="기록 새로고침" onClick={refreshRecordings}><RefreshCw size={15} /></button></div>
          <div className="recording-list">
            {recordings.length ? recordings.slice(0, 6).map((recording) => (
              <a key={recording.filename} href={`${API_BASE}${recording.downloadUrl}`}>
                <div><strong>{recording.filename}</strong><span>{new Date(recording.modifiedAt).toLocaleString("ko-KR")} · {formatBytes(recording.bytes)}</span></div><Download size={16} />
              </a>
            )) : <div className="empty-recordings"><HardDrive size={20} /><span>저장된 CSV 기록이 없습니다.</span></div>}
          </div>
        </article>

        <article className="panel device-panel">
          <div className="panel-header compact"><div><p className="panel-kicker">DEVICE & PROTOCOL</p><h2>장치 정보</h2></div><Usb size={19} /></div>
          <dl className="device-details">
            <div><dt>장치</dt><dd>{status?.device.name ?? "PolyG-I"}</dd></div>
            <div><dt>전송</dt><dd>{status?.device.transport ?? "USB HID"} · {status?.device.reportBytes ?? 1024} B/report</dd></div>
            <div><dt>식별자</dt><dd>{status?.device.vendorId ?? "0x0F1F"} : {status?.device.productId ?? "0x0010"}</dd></div>
            <div><dt>프로토콜</dt><dd>D1WD10 · 16 physical / 8 EEG</dd></div>
            <div><dt>전압 환산</dt><dd>−1.25 / 32768 V/count</dd></div>
            <div><dt>EEG PGA</dt><dd>index {status?.signal?.pgaGainIndex ?? gainIndex} · ×{status?.signal?.pgaGain?.toFixed(2) ?? "1.70"}</dd></div>
            <div><dt>처리</dt><dd>BP 0.5–45 Hz (4th) · N60 Q30</dd></div>
          </dl>
          <p className="fine-print">ADC 입력 전압까지는 TeleScan D1WD10 DLL과 공식 문서에 맞춰 환산합니다. 전극 입력 µV는 장치의 고정 전치증폭 이득이 공개·교정되지 않아 표시하지 않습니다.</p>
        </article>
      </section>

      <footer><span>localhost 전용 · 데이터는 이 Mac 밖으로 전송되지 않습니다.</span><span>PolyG-I / D1WD10 / 256 Hz / 8 EEG / ADC mV</span></footer>
    </main>
  );
}
