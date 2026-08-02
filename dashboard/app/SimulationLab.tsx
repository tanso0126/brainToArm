"use client";

import {
  Activity,
  Box,
  Brain,
  Check,
  CircleAlert,
  Crosshair,
  Cuboid,
  Eye,
  Gauge,
  Grip,
  Layers3,
  Pause,
  Play,
  Plus,
  Radar,
  RotateCcw,
  Save,
  ScanSearch,
  Sparkles,
  Trash2,
  Undo2,
  Video,
  WifiOff,
  X,
  Zap,
} from "lucide-react";
import type { CSSProperties, ReactNode } from "react";
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

const API_BASE = "http://127.0.0.1:8765";

type Shape = "box" | "cylinder" | "sphere";
type SignalSource = "manual" | "mock" | "polyg";

type ErrpStatus = {
  baselineReady: boolean;
  baselineSource: "fresh" | "saved" | null;
  baselineCreatedAt: string | null;
  backend: "baseline" | "model";
  threshold: number;
  windowSeconds: number;
  baselineSeconds: number;
  channels: number[];
  bandHz: [number, number];
  baselineStdMv: number | null;
  calibrationWindow: {
    seconds: number;
    samples: number;
    requiredSamples: number;
    quality: {
      state: "waiting" | "present" | "flat" | "saturated" | "unstable";
      rmsMv: number;
      peakToPeakMv: number;
      rawPeakToPeakMv: number;
      clippingPercent: number;
      dcOffsetMv: number;
    };
    channelQualities: Record<string, {
      state: "waiting" | "present" | "flat" | "saturated" | "unstable";
      rmsMv: number;
      peakToPeakMv: number;
      rawPeakToPeakMv: number;
      clippingPercent: number;
      dcOffsetMv: number;
    }>;
    requiredChannels: number[];
    blockingChannels: string[];
    ready: boolean;
  };
  lastDecision: {
    isError: boolean;
    rawDetected: boolean;
    probability: number;
    threshold: number;
    applied: boolean;
    override: boolean;
    samples: number;
    marker: string;
    channels: number[];
    bandHz: [number, number];
    robotWeight: number;
    humanWeight: number;
    errpApplyStride: number;
    relativeTar: number | null;
    negativeDeflectionMv: number | null;
    baselineStdMv: number | null;
    zScore: number | null;
  } | null;
  asynchronous: AsyncErrpStatus;
};

type AsyncErrpStatus = {
  enabled: boolean;
  mode: "trained-model" | "baseline-heuristic";
  trained: boolean;
  windowMs: number;
  stepMs: number;
  logicalEvaluationsPerSecond: number;
  requiredConsecutive: number;
  refractoryMs: number;
  threshold: number;
  bufferedSamples: number;
  requiredSamples: number;
  evaluations: number;
  probability: number | null;
  aboveThreshold: boolean;
  consecutive: number;
  detectionSequence: number;
  detectedAt: string | null;
  negativeDeflectionMv: number | null;
  baselineStdMv: number | null;
  zScore: number | null;
};

type LoadStatus = {
  baselineReady: boolean;
  thetaChannels: number[];
  alphaChannels: number[];
  thetaBandHz: [number, number];
  alphaBandHz: [number, number];
  windowSeconds: number;
  updateSeconds: number;
  restTar: number | null;
  tar: number | null;
  relativeTar: number | null;
  smoothedRelativeTar: number | null;
  thetaPowers: number[];
  alphaPowers: number[];
  valid: boolean;
  reason: string;
  robotWeight: number;
  humanWeight: number;
  errpThreshold: number;
  errpApplyStride: number;
  strongErrpOverrideThreshold: number;
};

type SavedBaseline = {
  available: boolean;
  compatible: boolean;
  autoLoadEnabled: boolean;
  reason: string;
  createdAt: string | null;
  path: string;
  gainIndex?: number;
  samplingHz?: number;
};

function calibrationWindowLabel(window: ErrpStatus["calibrationWindow"] | undefined) {
  if (!window) return "EEG 데이터 대기";
  if (window.samples < window.requiredSamples) {
    return `데이터 수집 중 ${window.samples}/${window.requiredSamples}`;
  }
  if (window.blockingChannels.length) {
    return `데이터 충분 · 신호 문제 CH ${window.blockingChannels.join(", ")}`;
  }
  return "CH1·2·3·4·8 정상 · 보정 가능";
}

type SimObject = {
  id: string;
  label: string;
  shape: Shape;
  color: string;
  sizeMm: number;
  xMm: number;
  yMm: number;
  zMm: number;
  originXmm: number;
  originYmm: number;
  status: "table" | "held" | "basket";
};

type SimEvent = {
  id: number;
  kind: "info" | "move" | "error" | "success";
  text: string;
  at: string;
};

type SimulationStatus = {
  engine: string;
  physics: boolean;
  cameraOnlySelection: boolean;
  running: boolean;
  phase: string;
  cycle: number;
  activeId: string | null;
  lastDeliveredId: string | null;
  postDeliveryReviewSeconds: number | null;
  postDeliveryReviewMode: "until-stopped";
  rejectedIds: string[];
  objects: SimObject[];
  basket: { xMm: number; yMm: number };
  servoDeg: number[];
  toolMm: number[];
  workspace: {
    radiusMm: [number, number];
    yawDeg: [number, number];
    baseNeutralDeg: number;
    baseMode: string;
  };
  detector: {
    source: string;
    visibleIds: string[];
    targetCenter: [number, number] | null;
    markerRow: number | null;
    pixelCount: number;
  };
  events: SimEvent[];
};

type ObjectDraft = {
  label: string;
  shape: Shape;
  color: string;
  sizeMm: number;
  radiusMm: number;
  yawDeg: number;
};

const PHASE_COPY: Record<string, { title: string; detail: string }> = {
  idle: { title: "준비됨", detail: "물체와 목표 트레이를 배치한 뒤 자동 실행하세요." },
  scanning: { title: "3D 탐색", detail: "1번 축은 90°에 고정하고 2·3·4번과 손목 RGB 카메라로 전후 작업영역을 훑습니다." },
  target: { title: "후보 제시 · ErrP", detail: "카메라로 찾은 후보에 대한 오류 반응을 확인합니다." },
  reaching: { title: "접근 중", detail: "고정된 단일 시상면과 실제 관절 제한 안에서 목표 깊이로 이동합니다." },
  grasping: { title: "물리 파지", detail: "MuJoCo 접촉으로 닫고 실제 물체 상승을 검증합니다." },
  transporting: { title: "운반 중", detail: "물체를 든 상태로 목표 트레이에 이동합니다." },
  evaluating: { title: "배송 확인 · ErrP", detail: "정지 버튼을 누를 때까지 CH8 거부 반응을 계속 판정합니다." },
  returning: { title: "원위치 복귀", detail: "트레이에서 회수해 저장된 원점으로 되돌립니다." },
  completed: { title: "배송 완료", detail: "늦은 거부가 오면 다시 회수할 수 있습니다." },
  paused: { title: "일시정지", detail: "현재 MuJoCo 상태를 유지합니다." },
  safety_hold: { title: "안전 정지", detail: "몸체 또는 바닥 보호 경계가 경로를 차단했습니다." },
  error: { title: "엔진 오류", detail: "실행 기록의 원인을 확인하세요." },
};

const JOINT_LABELS = ["1 Base · FIXED", "2 Shoulder", "3 Elbow", "4 Wrist pitch", "5 Gripper", "6 Wrist roll · FIXED"];

function polar(x: number, y: number) {
  return {
    radiusMm: Math.hypot(x, y),
    yawDeg: Math.atan2(y, x) * 180 / Math.PI,
  };
}

function cartesian(radiusMm: number, yawDeg: number) {
  const radians = yawDeg * Math.PI / 180;
  return {
    xMm: radiusMm * Math.cos(radians),
    yMm: radiusMm * Math.sin(radians),
  };
}

async function simulationRequest<T = SimulationStatus>(
  path: string,
  body?: Record<string, unknown>,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "MuJoCo 요청 실패");
  return payload;
}

function PlacementMap({
  status,
  selectedId,
  onSelect,
  onPlace,
  disabled,
}: {
  status: SimulationStatus;
  selectedId: string | "basket" | null;
  onSelect: (id: string | "basket") => void;
  onPlace: (id: string | "basket", radiusMm: number, yawDeg: number) => void;
  disabled: boolean;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);
  const radiusRange = status.workspace.radiusMm;
  const pointFor = useCallback((xMm: number, yMm: number) => {
    const { radiusMm } = polar(xMm, yMm);
    const radiusT = (radiusMm - radiusRange[0]) / (radiusRange[1] - radiusRange[0]);
    return {
      x: 34 + radiusT * 232,
      y: 61,
    };
  }, [radiusRange]);

  const placeAt = useCallback((clientX: number) => {
    if (!selectedId || disabled || !ref.current) return;
    const bounds = ref.current.getBoundingClientRect();
    const x = Math.max(34, Math.min(266, (clientX - bounds.left) / bounds.width * 300));
    const radiusMm = radiusRange[0] + (x - 34) / 232 * (radiusRange[1] - radiusRange[0]);
    onPlace(selectedId, radiusMm, 0);
  }, [disabled, onPlace, radiusRange, selectedId]);

  return (
    <svg
      ref={ref}
      className="sim-placement-map"
      viewBox="0 0 300 126"
      role="img"
      aria-label="3D 장면의 물체와 트레이 위치를 정하는 위에서 본 배치 도구"
      onPointerDown={(event) => {
        if (!selectedId || disabled) return;
        dragging.current = true;
        event.currentTarget.setPointerCapture(event.pointerId);
        placeAt(event.clientX);
      }}
      onPointerMove={(event) => {
        if (dragging.current) placeAt(event.clientX);
      }}
      onPointerUp={(event) => {
        dragging.current = false;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
      }}
    >
      <title>MuJoCo 3D 작업영역 배치 도구</title>
      <path d="M30 48 H270 V74 H30 Z" className="workspace-band outer" />
      <path d="M34 55 H266 V67 H34 Z" className="workspace-band inner" />
      <path d="M18 61 H282" className="workspace-center" />
      <circle cx="15" cy="61" r="8" className="workspace-base" />
      <text x="150" y="105" textAnchor="middle">servo1 90° 고정 · 전후 {radiusRange[0].toFixed(0)}–{radiusRange[1].toFixed(0)} mm 단일 시상면</text>
      {status.objects.map((item) => {
        const point = pointFor(item.xMm, item.yMm);
        return (
          <g
            key={item.id}
            className={selectedId === item.id ? "selected" : ""}
            onPointerDown={(event) => {
              event.stopPropagation();
              onSelect(item.id);
            }}
          >
            <circle cx={point.x} cy={point.y} r="7" fill={item.color} />
            <text x={point.x} y={point.y - 10} textAnchor="middle">{item.label}</text>
          </g>
        );
      })}
      {(() => {
        const point = pointFor(status.basket.xMm, status.basket.yMm);
        return (
          <g
            className={selectedId === "basket" ? "basket selected" : "basket"}
            onPointerDown={(event) => {
              event.stopPropagation();
              onSelect("basket");
            }}
          >
            <rect x={point.x - 9} y={point.y - 6} width="18" height="12" rx="3" />
            <text x={point.x} y={point.y - 10} textAnchor="middle">목표</text>
          </g>
        );
      })()}
    </svg>
  );
}

function SimulationLab({
  eegRunning,
  apiOnline,
  errpStatus,
  loadStatus,
  savedBaseline,
  eegPanel,
}: {
  eegRunning: boolean;
  apiOnline: boolean;
  errpStatus: ErrpStatus | null;
  loadStatus: LoadStatus | null;
  savedBaseline: SavedBaseline | null;
  eegPanel: ReactNode;
}) {
  const [status, setStatus] = useState<SimulationStatus | null>(null);
  const [engineError, setEngineError] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectedId, setSelectedId] = useState<string | "basket" | null>(null);
  const [draft, setDraft] = useState<ObjectDraft | null>(null);
  const [signalSource, setSignalSource] = useState<SignalSource>("polyg");
  const [localErrpReady, setLocalErrpReady] = useState(false);
  const [errpBusy, setErrpBusy] = useState(false);
  const [errpError, setErrpError] = useState("");
  const targetErrpGeneration = useRef(0);
  const handledAsyncDetectionRef = useRef(0);
  const [asyncErrp, setAsyncErrp] = useState<AsyncErrpStatus | null>(null);
  const overviewImageRef = useRef<HTMLImageElement>(null);
  const wristImageRef = useRef<HTMLImageElement>(null);
  const errpReady = Boolean(
    (errpStatus?.baselineReady && loadStatus?.baselineReady)
    || localErrpReady);
  const calibrationWindow = errpStatus?.calibrationWindow;
  const shownAsyncErrp = asyncErrp ?? errpStatus?.asynchronous ?? null;

  const refreshStatus = useCallback(async () => {
    try {
      const next = await simulationRequest<SimulationStatus>("/api/simulation/status");
      setStatus((current) => (
        current && JSON.stringify(current) === JSON.stringify(next)
          ? current
          : next
      ));
      setEngineError("");
      return next;
    } catch (error) {
      setEngineError(error instanceof Error ? error.message : String(error));
      return null;
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(refreshStatus, 0);
    const timer = window.setInterval(refreshStatus, 400);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refreshStatus]);

  useEffect(() => {
    const updateFrames = () => {
      const stamp = Date.now();
      if (overviewImageRef.current) {
        overviewImageRef.current.src = `${API_BASE}/api/simulation/frame?camera=overview&width=960&height=540&t=${stamp}`;
      }
      if (wristImageRef.current) {
        wristImageRef.current.src = `${API_BASE}/api/simulation/frame?camera=wrist&width=640&height=360&t=${stamp}`;
      }
    };
    updateFrames();
    const timer = window.setInterval(() => {
      if (!document.hidden) updateFrames();
    }, status?.running ? 160 : 800);
    return () => window.clearInterval(timer);
  }, [status?.running]);

  const selected = useMemo(
    () => status?.objects.find((item) => item.id === selectedId) ?? null,
    [selectedId, status?.objects],
  );
  const active = useMemo(
    () => status?.objects.find((item) => item.id === status.activeId) ?? null,
    [status],
  );

  const selectSceneItem = useCallback((id: string | "basket") => {
    setSelectedId(id);
    if (id === "basket") {
      setDraft(null);
      return;
    }
    const item = status?.objects.find((value) => value.id === id);
    if (!item) return;
    const position = polar(item.originXmm, item.originYmm);
    setDraft({
      label: item.label,
      shape: item.shape,
      color: item.color,
      sizeMm: item.sizeMm,
      radiusMm: position.radiusMm,
      yawDeg: position.yawDeg,
    });
  }, [status?.objects]);

  const perform = useCallback(async (
    path: string,
    body?: Record<string, unknown>,
  ) => {
    setBusy(true);
    try {
      const next = await simulationRequest<SimulationStatus>(path, body ?? {});
      setStatus(next);
      setEngineError("");
      return next;
    } catch (error) {
      setEngineError(error instanceof Error ? error.message : String(error));
      return null;
    } finally {
      setBusy(false);
    }
  }, []);

  const reject = useCallback(() => {
    void perform("/api/simulation/reject");
  }, [perform]);

  const calibrateErrp = useCallback(async () => {
    if (!apiOnline || !eegRunning) return;
    setErrpBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/errp/calibrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seconds: 8 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "ErrP 보정 실패");
      setLocalErrpReady(true);
      setErrpError("");
      setEngineError("");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setErrpError(message);
      setEngineError(message);
    } finally {
      setErrpBusy(false);
    }
  }, [apiOnline, eegRunning]);

  const loadSavedBaseline = useCallback(async () => {
    if (!apiOnline || !eegRunning || !savedBaseline?.compatible) return;
    setErrpBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/baseline/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "저장 기준 불러오기 실패");
      setLocalErrpReady(true);
      setErrpError("");
      setEngineError("");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setErrpError(message);
      setEngineError(message);
    } finally {
      setErrpBusy(false);
    }
  }, [apiOnline, eegRunning, savedBaseline]);

  const decisionPhase = status?.phase ?? "";
  const decisionCycle = status?.cycle ?? 0;
  const decisionObjectId = decisionPhase === "evaluating"
    ? status?.lastDeliveredId
    : status?.activeId;

  useEffect(() => {
    if (signalSource !== "polyg" || !errpReady || !eegRunning) return;
    if (decisionPhase !== "target" || !decisionObjectId) return;

    let cancelled = false;
    let startTimer = 0;
    const generation = ++targetErrpGeneration.current;

    const checkTarget = async () => {
      setErrpBusy(true);
      try {
        const response = await fetch(`${API_BASE}/api/errp/check`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ marker: "SIM_TARGET_PRESENTED" }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "ErrP 판정 실패");
        if (!cancelled && payload.isError) reject();
      } catch (error) {
        if (!cancelled) {
          setEngineError(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (targetErrpGeneration.current === generation) setErrpBusy(false);
      }
    };

    // Deferring the request prevents React development-mode effect probing
    // from duplicating this one event-locked target-presentation epoch.
    startTimer = window.setTimeout(() => void checkTarget(), 80);
    return () => {
      cancelled = true;
      window.clearTimeout(startTimer);
      if (targetErrpGeneration.current === generation) {
        targetErrpGeneration.current += 1;
        setErrpBusy(false);
      }
    };
  }, [
    decisionCycle,
    decisionObjectId,
    decisionPhase,
    eegRunning,
    errpReady,
    reject,
    signalSource,
  ]);

  useEffect(() => {
    if (signalSource !== "polyg" || !errpReady || !eegRunning) return;
    if (decisionPhase !== "evaluating" || !decisionObjectId) return;

    let cancelled = false;
    let inFlight = false;
    let armed = false;
    let timer = 0;

    const pull = async () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      try {
        const response = await fetch(`${API_BASE}/api/errp/async`, {
          cache: "no-store",
        });
        const payload = await response.json() as AsyncErrpStatus;
        if (!response.ok) throw new Error("비동기 ErrP 상태 조회 실패");
        if (cancelled) return;
        setAsyncErrp(payload);
        if (!armed) {
          // Arm from a fresh server sequence at phase entry so an older
          // transport-phase false positive cannot reject the delivered item.
          handledAsyncDetectionRef.current = payload.detectionSequence;
          armed = true;
          return;
        }
        if (payload.detectionSequence > handledAsyncDetectionRef.current) {
          handledAsyncDetectionRef.current = payload.detectionSequence;
          reject();
        }
      } catch (error) {
        if (!cancelled) {
          setEngineError(error instanceof Error ? error.message : String(error));
        }
      } finally {
        inFlight = false;
      }
    };

    timer = window.setInterval(() => void pull(), 50);
    void pull();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    decisionCycle,
    decisionObjectId,
    decisionPhase,
    eegRunning,
    errpReady,
    reject,
    signalSource,
  ]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.code === "Space") {
        event.preventDefault();
        void perform(status?.running ? "/api/simulation/stop" : "/api/simulation/start");
      }
      if (event.key.toLowerCase() === "x") reject();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [perform, reject, status?.running]);

  const saveDraft = useCallback(async () => {
    if (!selected || !draft) return;
    const xy = cartesian(draft.radiusMm, draft.yawDeg);
    await perform("/api/simulation/objects/update", {
      id: selected.id,
      label: draft.label,
      shape: draft.shape,
      color: draft.color,
      sizeMm: draft.sizeMm,
      ...xy,
    });
  }, [draft, perform, selected]);

  const place = useCallback((
    id: string | "basket",
    radiusMm: number,
    yawDeg: number,
  ) => {
    const xy = cartesian(radiusMm, yawDeg);
    if (id === "basket") {
      void perform("/api/simulation/basket/update", xy);
      return;
    }
    const item = status?.objects.find((value) => value.id === id);
    if (!item) return;
    void perform("/api/simulation/objects/update", {
      id,
      label: item.label,
      shape: item.shape,
      color: item.color,
      sizeMm: item.sizeMm,
      ...xy,
    });
    if (selectedId === id) {
      setDraft((current) => current ? { ...current, radiusMm, yawDeg } : current);
    }
  }, [perform, selectedId, status?.objects]);

  const phase = PHASE_COPY[status?.phase ?? "idle"] ?? {
    title: status?.phase ?? "엔진 연결 중",
    detail: "MuJoCo 상태를 읽고 있습니다.",
  };
  const rejected = new Set(status?.rejectedIds ?? []);
  const frameBase = `${API_BASE}/api/simulation/frame`;

  if (!status) {
    return (
      <section className="sim-lab sim-engine-loading">
        <div className="sim-engine-empty">
          <WifiOff size={30} />
          <h2>MuJoCo 3D 엔진에 연결하는 중입니다.</h2>
          <p>{engineError || "로컬 API가 3D 모델과 STL 자산을 준비하고 있습니다."}</p>
          <button className="sim-primary" onClick={() => void refreshStatus()}>다시 연결</button>
        </div>
      </section>
    );
  }

  return (
    <section className="sim-lab sim-3d-lab">
      <div className="sim-hero">
        <div>
          <p className="sim-eyebrow"><Sparkles size={13} /> MUJOCO SHARED AUTONOMY STUDIO</p>
          <h2>실제 로봇 형상과 접촉 물리로 검증합니다.</h2>
          <p>원본 STL, 실제 서보 범위, 손목 RGB 카메라, free-body 접촉을 사용합니다. 2D 그림은 장면 배치 도구일 뿐 제어 결과가 아닙니다.</p>
        </div>
        <div className="sim-hero-actions">
          <span className={`sim-state-pill phase-${status.phase}`}><i />{phase.title}</span>
          {!status.running ? (
            <button className="sim-primary" disabled={busy || !status.objects.some((item) => item.status === "table")} onClick={() => void perform("/api/simulation/start")}>
              <Play size={17} fill="currentColor" />자동 실행
            </button>
          ) : (
            <button className="sim-secondary" disabled={busy} onClick={() => void perform("/api/simulation/stop")}>
              <Pause size={17} />일시정지
            </button>
          )}
          <button className="sim-icon-button" disabled={busy} onClick={() => void perform("/api/simulation/reset")} aria-label="3D 시뮬레이션 초기화">
            <RotateCcw size={17} />
          </button>
        </div>
      </div>

      {engineError && <div className="sim-engine-error"><CircleAlert size={16} />{engineError}</div>}

      <div className="sim-status-strip">
        <div><span>현재 단계</span><strong>{phase.title}</strong><small>{phase.detail}</small></div>
        <div><span>선택 후보</span><strong>{active?.label ?? "—"}</strong><small>{active ? `wrist RGB · ${status.detector.pixelCount} px` : "카메라 탐색 대기"}</small></div>
        <div><span>물리 엔진</span><strong>{status.engine} contact</strong><small>물체 상승·손끝 추종으로 파지 검증</small></div>
        <div><span>실험 사이클</span><strong>#{status.cycle}</strong><small>실물 동일 · servo1 90° 고정</small></div>
      </div>

      <div className="sim-layout sim-3d-layout">
        <div className="sim-main-column">
          <article className="sim-card sim-viewport-card">
            <div className="sim-card-head">
              <div><p>PHYSICS OVERVIEW</p><h3>MuJoCo 3D 작업 장면</h3></div>
              <span className="sim-live"><i />{status.running ? "LIVE PHYSICS" : "PAUSED"}</span>
            </div>
            <div className="sim-frame-wrap overview">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                ref={overviewImageRef}
                src={`${frameBase}?camera=overview&width=960&height=540`}
                alt="원본 로봇팔 STL과 물체 접촉을 렌더링한 MuJoCo 3D 장면"
              />
              <div className="sim-frame-hud">
                <span><Cuboid size={13} />free body {status.objects.length}</span>
                <span><Crosshair size={13} />tool {status.toolMm.map((value) => value.toFixed(0)).join(", ")} mm</span>
              </div>
            </div>
          </article>

          <div className="sim-lower-row sim-3d-lower">
            <article className="sim-card sim-camera-card">
              <div className="sim-card-head">
                <div><p>DEPLOYMENT OBSERVATION</p><h3>로봇 손목 RGB 카메라</h3></div>
                <span className="sim-live"><i />RGB ONLY</span>
              </div>
              <div className="sim-frame-wrap wrist">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  ref={wristImageRef}
                  src={`${frameBase}?camera=wrist&width=640&height=360`}
                  alt="실물과 동일한 위치에서 렌더링한 집게 손목 RGB 카메라"
                />
                <div className="sim-frame-hud">
                  <span><Eye size={13} />보이는 후보 {status.detector.visibleIds.length}</span>
                  <span><Radar size={13} />{status.detector.source}</span>
                </div>
              </div>
            </article>

            <article className="sim-card sim-joint-card">
              <div className="sim-card-head"><div><p>COMMAND TELEMETRY</p><h3>실제 서보 명령</h3></div><Gauge size={18} /></div>
              <div className="sim-joint-list">
                {JOINT_LABELS.map((label, index) => (
                  <div key={label}>
                    <span>{label}</span>
                    <strong>{status.servoDeg[index].toFixed(1)}°</strong>
                    <i style={{ "--joint-value": `${status.servoDeg[index] / 180 * 100}%` } as CSSProperties} />
                  </div>
                ))}
              </div>
            </article>
          </div>

          {eegPanel}

          <article className="sim-card sim-placement-card">
            <div className="sim-card-head">
              <div><p>SCENE AUTHORING · NOT THE SIMULATOR</p><h3>물체와 목표 위치 배치</h3></div>
              <div className="sim-object-tools">
                <button onClick={() => void perform("/api/simulation/objects/add", { shape: "sphere" })} disabled={status.running || busy}><Plus size={13} />공</button>
                <button onClick={() => void perform("/api/simulation/objects/add", { shape: "box" })} disabled={status.running || busy}><Plus size={13} />블록</button>
                <button onClick={() => void perform("/api/simulation/objects/add", { shape: "cylinder" })} disabled={status.running || busy}><Plus size={13} />원통</button>
              </div>
            </div>
            <PlacementMap
              status={status}
              selectedId={selectedId}
              onSelect={selectSceneItem}
              onPlace={place}
              disabled={status.running || busy}
            />
            <p className="sim-note">점 또는 파란 목표 트레이를 선택한 뒤 좌우로 드래그하세요. 1번 모터가 고정되어 모든 배치는 실제와 같은 단일 전후 작업선으로 강제됩니다.</p>
          </article>
        </div>

        <aside className="sim-side-column">
          <article className="sim-card sim-signal-card">
            <div className="sim-card-head"><div><p>EXTERNAL CORRECTION</p><h3>거부 신호</h3></div><Brain size={20} /></div>
            <div className="sim-signal-tabs">
              <button className={signalSource === "manual" ? "active" : ""} onClick={() => setSignalSource("manual")}>수동</button>
              <button className={signalSource === "mock" ? "active" : ""} onClick={() => setSignalSource("mock")}>가상 ErrP</button>
              <button className={signalSource === "polyg" ? "active" : ""} onClick={() => setSignalSource("polyg")}>PolyG-I</button>
            </div>
            <button className="sim-reject-button" onClick={reject} disabled={busy || (!status.activeId && !status.lastDeliveredId)}>
              <CircleAlert size={21} />
              <span>{signalSource === "mock" ? "가상 ErrP 보내기" : "아니야 · 다른 물체"}</span>
              <kbd>X</kbd>
            </button>
            <p className="sim-helper">탐색·접근·파지·운반 중은 물론 트레이에 놓은 뒤에도 동작합니다. 늦은 거부는 실제로 다시 집어 원점 복귀 후 다음 물체로 이어집니다.</p>
            {signalSource === "polyg" && (
              <div className={`sim-eeg-box ${errpReady ? "ready" : ""}`}>
                <div><span><Activity size={15} />PolyG-I</span><strong>{apiOnline && eegRunning ? "측정 중" : "준비 안 됨"}</strong></div>
                <div><span><Zap size={15} />판정 입력</span><strong>CH8 단독 · 1–10 Hz</strong></div>
                <div><span><Zap size={15} />연속 TAR</span><strong>θ CH1–4 / α CH8</strong></div>
                <div><span><Zap size={15} />최근 8초 품질</span><strong>{calibrationWindowLabel(calibrationWindow)}</strong></div>
                <div><span><Zap size={15} />휴식 기준</span><strong>{errpReady ? "ErrP + TAR 완료" : "통합 보정 필요"}</strong></div>
                <div><span><Radar size={15} />현재 ErrP 감시</span><strong>{status.phase === "evaluating" ? "비동기 슬라이딩 · 정지할 때까지" : errpBusy ? "행동 직후 event-locked 판정" : "다음 행동 대기"}</strong></div>
                {loadStatus?.baselineReady && <>
                  <div><span><Brain size={15} />TAR / 휴식 대비</span><strong>{loadStatus.tar?.toFixed(3) ?? "—"} / {loadStatus.smoothedRelativeTar != null ? `${loadStatus.smoothedRelativeTar >= 0 ? "+" : ""}${(loadStatus.smoothedRelativeTar * 100).toFixed(1)}%` : "—"}</strong></div>
                  <div><span><Brain size={15} />자율성 가중치</span><strong>로봇 {(loadStatus.robotWeight * 100).toFixed(0)} · 인간 {(loadStatus.humanWeight * 100).toFixed(0)}</strong></div>
                  <div><span><Zap size={15} />ErrP 행동 반영</span><strong>비동기 1창 ≥{(loadStatus.errpThreshold * 100).toFixed(0)}% 즉시 거부</strong></div>
                </>}
                {shownAsyncErrp && <>
                  <div><span><Activity size={15} />실시간 P(error)</span><strong>{shownAsyncErrp.probability == null ? `${shownAsyncErrp.bufferedSamples}/${shownAsyncErrp.requiredSamples} 준비` : `${(shownAsyncErrp.probability * 100).toFixed(1)}% · 연속 ${shownAsyncErrp.consecutive}/${shownAsyncErrp.requiredConsecutive}`}</strong></div>
                  <div><span><Gauge size={15} />슬라이딩 판정기</span><strong>{shownAsyncErrp.windowMs.toFixed(0)}ms 창 · {shownAsyncErrp.stepMs.toFixed(1)}ms 이동</strong></div>
                  <div><span><Brain size={15} />판정기 상태</span><strong>{shownAsyncErrp.trained ? "피험자 학습 모델" : "휴식 기반 휴리스틱 · 미학습"}</strong></div>
                </>}
                <button onClick={calibrateErrp} disabled={!apiOnline || !eegRunning || errpBusy || !calibrationWindow?.ready}>{errpBusy ? "처리 중…" : calibrationWindow?.ready ? "최근 8초로 통합 보정·저장" : "신호 확인 후 보정"}</button>
                {savedBaseline?.available && <button onClick={loadSavedBaseline} disabled={!apiOnline || !eegRunning || errpBusy || !savedBaseline.compatible || errpReady}>저장 안정 기준 불러오기</button>}
                {savedBaseline?.available && <small>{savedBaseline.compatible ? `${savedBaseline.createdAt ? new Date(savedBaseline.createdAt).toLocaleString("ko-KR") : "이전"} · 측정 시작 시 자동 적용 · PGA index ${savedBaseline.gainIndex ?? "—"} · 같은 피험자/전극 배치 전용` : savedBaseline.reason}</small>}
                {!calibrationWindow?.ready && <small>샘플 수는 안정도 점수가 아닙니다. 데이터가 충분한데 차단되면 전극·REF/GND·포화 상태를 확인하세요.</small>}
                {errpError && <p className="sim-errp-error"><CircleAlert size={13} />{errpError}</p>}
                {errpStatus?.lastDecision ? (
                  <div className={`sim-errp-decision ${errpStatus.lastDecision.isError ? "detected" : ""}`}>
                    <span>최근 P(error)</span>
                    <strong>{(errpStatus.lastDecision.probability * 100).toFixed(1)}% / 기준 {(errpStatus.lastDecision.threshold * 100).toFixed(0)}%</strong>
                    <span>행동 반영</span>
                    <strong>{errpStatus.lastDecision.override ? "강한 ErrP 즉시 반영" : errpStatus.lastDecision.applied ? "이번 판정 반영" : `관측만 · 매 ${errpStatus.lastDecision.errpApplyStride}번째`}</strong>
                    <span>CH8 z-score</span>
                    <strong>{errpStatus.lastDecision.zScore?.toFixed(2) ?? "모델 판정"}</strong>
                  </div>
                ) : null}
                <small>행동 직후 ErrP는 event-locked로, 배송 후 감시는 CH8의 겹치는 슬라이딩 창으로 처리합니다. 비동기는 한 창이 50%를 넘는 즉시 거부합니다. 미학습 휴리스틱은 진단용이며, 분노 자체를 측정하는 감정 분류기가 아닙니다.</small>
              </div>
            )}
          </article>

          <article className="sim-card sim-object-card">
            <div className="sim-card-head"><div><p>3D OBJECT INSPECTOR</p><h3>장면 물체</h3></div><Box size={18} /></div>
            <div className="sim-scene-list">
              {status.objects.map((item) => (
                <button
                  key={item.id}
                  className={`${selectedId === item.id ? "active" : ""} ${rejected.has(item.id) ? "rejected" : ""}`}
                  onClick={() => selectSceneItem(item.id)}
                >
                  <i style={{ background: item.color }} />
                  <span><strong>{item.label}</strong><small>{item.status} · z {item.zMm.toFixed(1)} mm</small></span>
                  {rejected.has(item.id) ? <X size={14} /> : item.status === "basket" ? <Check size={14} /> : null}
                </button>
              ))}
              <button className={selectedId === "basket" ? "active basket" : "basket"} onClick={() => selectSceneItem("basket")}>
                <i /><span><strong>목표 트레이</strong><small>{status.basket.xMm.toFixed(0)}, {status.basket.yMm.toFixed(0)} mm</small></span>
              </button>
            </div>
            {selected && draft ? (
              <div className="sim-object-editor sim-3d-editor">
                <label>이름<input value={draft.label} disabled={status.running} onChange={(event) => setDraft({ ...draft, label: event.target.value })} /></label>
                <label>모양<select value={draft.shape} disabled={status.running} onChange={(event) => setDraft({ ...draft, shape: event.target.value as Shape })}><option value="box">블록</option><option value="cylinder">원통</option><option value="sphere">공</option></select></label>
                <label>색상<input type="color" value={draft.color} disabled={status.running} onChange={(event) => setDraft({ ...draft, color: event.target.value })} /></label>
                <label>크기 <b>{draft.sizeMm.toFixed(1)} mm</b><input type="range" min="4.5" max="8" step=".5" value={draft.sizeMm} disabled={status.running} onChange={(event) => setDraft({ ...draft, sizeMm: Number(event.target.value) })} /></label>
                <label>반경 <b>{draft.radiusMm.toFixed(1)} mm</b><input type="range" min={status.workspace.radiusMm[0]} max={status.workspace.radiusMm[1]} step=".5" value={draft.radiusMm} disabled={status.running} onChange={(event) => setDraft({ ...draft, radiusMm: Number(event.target.value) })} /></label>
                <label>1번 축 <b>90° 고정</b><input type="range" min="90" max="90" value="90" disabled aria-label="1번 베이스 축 고정값" readOnly /></label>
                <div>
                  <button className="sim-save-button" onClick={saveDraft} disabled={status.running || busy}><Save size={13} />3D 장면 반영</button>
                  <button className="sim-delete-button" onClick={() => void perform("/api/simulation/objects/delete", { id: selected.id })} disabled={status.running || busy}><Trash2 size={13} />삭제</button>
                </div>
              </div>
            ) : (
              <div className="sim-empty compact"><Grip size={20} /><span>{selectedId === "basket" ? "배치 도구에서 목표 위치를 드래그하세요." : "물체를 선택하면 외형과 위치를 조절할 수 있습니다."}</span></div>
            )}
          </article>

          <article className="sim-card sim-stack-card">
            <div className="sim-card-head"><div><p>REJECTION MEMORY</p><h3>거부된 물체</h3></div><Layers3 size={18} /></div>
            <div className="sim-rejected-list">
              {status.rejectedIds.length ? status.rejectedIds.map((id, index) => {
                const item = status.objects.find((value) => value.id === id);
                return item ? (
                  <div key={id}><span style={{ background: item.color }}>{index + 1}</span><div><strong>{item.label}</strong><small>원위치 복귀 · 현재 사이클 재선택 금지</small></div><X size={15} /></div>
                ) : null;
              }) : <div className="sim-empty compact"><Check size={20} /><span>아직 거부된 물체가 없습니다.</span></div>}
            </div>
          </article>

          <article className="sim-card sim-event-card">
            <div className="sim-card-head"><div><p>PHYSICS ACTION TRACE</p><h3>실행 기록</h3></div><ScanSearch size={18} /></div>
            <div className="sim-event-list">
              {status.events.map((event) => (
                <div key={event.id} className={event.kind}><i /><span>{event.at}</span><p>{event.text}</p></div>
              ))}
            </div>
          </article>
        </aside>
      </div>

      <div className="sim-bottom-callout">
        <div><Video size={19} /><span><strong>실제 3D 렌더</strong> 원본 STL과 MuJoCo 접촉 결과를 그대로 표시합니다.</span></div>
        <div><Eye size={19} /><span><strong>손목 RGB 선택</strong> 물체 ID나 depth가 아니라 렌더된 카메라 픽셀에 보여야 후보가 됩니다.</span></div>
        <div><Undo2 size={19} /><span><strong>가역적 ErrP 행동</strong> 배송 후 거부도 물리 회수·원점 복귀·다음 후보로 이어집니다.</span></div>
      </div>
    </section>
  );
}

export default memo(SimulationLab);
