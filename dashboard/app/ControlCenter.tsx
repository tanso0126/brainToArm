"use client";

import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  Brain,
  Camera,
  Check,
  ChevronRight,
  CircleStop,
  Cpu,
  Crosshair,
  Eye,
  Gauge,
  Hand,
  Play,
  RefreshCw,
  Save,
  Settings2,
  SlidersHorizontal,
  Usb,
  Video,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8765";

type ControlStatus = {
  platform: { system: string; release: string; python: string; windows: boolean };
  settings: {
    camera: string;
    armPort: string;
    armMode: "wrist-3dof";
    candidateIndex: number;
    candidateReviewSeconds: number;
    maxTaskSeconds: number;
    errpEnabled: boolean;
    tarEnabled: boolean;
    autoRejectSimulation: boolean;
    autoRejectPhysical: boolean;
  };
  camera: { running: boolean; index: number | null; frameAgeSeconds: number | null; previewUrl: string };
  arm: {
    connected: boolean;
    port: string | null;
    mode: string;
    activeServos: number[];
    fixedServos: number[];
    pose: number[] | null;
    distanceMm: number | null;
    error?: string;
  };
  task: {
    running: boolean;
    phase: string;
    activeCandidate: number | null;
    rejectedCandidates: number[];
    result: Record<string, unknown> | null;
  };
  detections: { index: number; center: number[]; bbox: number[]; area: number; confidence: number }[];
  events: { id: number; kind: string; text: string; at: string }[];
  lastError: string | null;
};

type EegSummary = {
  apiOnline: boolean;
  deviceReady: boolean;
  running: boolean;
  baselineReady: boolean;
  probability: number | null;
  threshold: number;
  tar: number | null;
  relativeTar: number | null;
  robotWeight: number;
  humanWeight: number;
};

const PHASES: Record<string, [string, string]> = {
  idle: ["대기", "장치를 연결하고 물체 후보를 확인하세요."],
  "candidate-review": ["후보 검토", "선택한 물체를 보며 ErrP 거부 신호를 기다립니다."],
  approaching: ["실시간 접근", "2·3·4번 모터가 카메라와 초음파를 따라 연속 보정합니다."],
  completed: ["파지·HOME 완료", "물체를 잡은 상태로 HOME 복귀했습니다."],
  stopped: ["사용자 중지", "남은 이동 명령을 취소했습니다."],
  "emergency-stop": ["긴급정지", "외부 서보 전원을 분리하면 물리 전원도 차단됩니다."],
  failed: ["작업 중단", "실행 기록에서 원인을 확인하세요."],
};

async function request<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `요청 실패 (${response.status})`);
  return payload as T;
}

export default function ControlCenter({
  eeg,
  startEeg,
  stopEeg,
  openEeg,
  openSimulation,
}: {
  eeg: EegSummary;
  startEeg: () => Promise<void> | void;
  stopEeg: () => Promise<void> | void;
  openEeg: () => void;
  openSimulation: () => void;
}) {
  const [status, setStatus] = useState<ControlStatus | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState(0);
  const [settings, setSettings] = useState<ControlStatus["settings"] | null>(null);
  const [jog, setJog] = useState({ shoulder: 90, elbow: 90, wrist: 170, gripper: 90 });
  const [homeConfirmed, setHomeConfirmed] = useState(false);
  const [diagnostic, setDiagnostic] = useState<Record<string, unknown> | null>(null);
  const frameRef = useRef<HTMLImageElement>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await request<ControlStatus>("/api/control/status");
      setStatus(next);
      setSettings((current) => current ?? next.settings);
      if (next.arm.pose && !next.task.running) {
        setJog({
          shoulder: next.arm.pose[1],
          elbow: next.arm.pose[2],
          wrist: next.arm.pose[3],
          gripper: next.arm.pose[4],
        });
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(refresh, 900);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!status?.camera.running) return;
    const timer = window.setInterval(() => {
      if (frameRef.current) {
        frameRef.current.src = `${API_BASE}/api/control/camera/frame?t=${Date.now()}`;
      }
    }, 120);
    return () => window.clearInterval(timer);
  }, [status?.camera.running]);

  useEffect(() => {
    if (!error) return;
    const timer = window.setTimeout(() => setError(null), 9000);
    return () => window.clearTimeout(timer);
  }, [error]);

  const act = useCallback(async (
    name: string,
    path: string,
    body: Record<string, unknown> = {},
  ) => {
    setBusy(name);
    setError(null);
    try {
      await request(path, body);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  }, [refresh]);

  const saveSettings = async () => {
    if (!settings) return;
    await act("settings", "/api/control/settings", settings);
  };

  const phase = PHASES[status?.task.phase ?? "idle"] ?? [status?.task.phase ?? "연결 중", "현재 단계를 확인하고 있습니다."];
  const setupCount = [eeg.running, status?.camera.running, status?.arm.connected].filter(Boolean).length;
  const allConnected = setupCount === 3;
  const selectedRejected = status?.task.rejectedCandidates.includes(selectedCandidate) ?? false;
  const canStart = Boolean(allConnected && status?.detections.length && !status?.task.running);
  const armPose = status?.arm.pose;
  const progress = useMemo(() => [
    { label: "PolyG-I", detail: eeg.running ? "8채널 측정 중" : eeg.deviceReady ? "연결됨 · 측정 대기" : "장치 미감지", ready: eeg.running, icon: Brain },
    { label: "손목 카메라", detail: status?.camera.running ? `카메라 ${status.camera.index}번 · LIVE` : "연결 대기", ready: Boolean(status?.camera.running), icon: Camera },
    { label: "Arduino Uno", detail: status?.arm.connected ? `${status.arm.port} · 2·3·4축+집게` : "연결 대기", ready: Boolean(status?.arm.connected), icon: Usb },
  ], [eeg.deviceReady, eeg.running, status?.arm.connected, status?.arm.port, status?.camera.index, status?.camera.running]);

  return (
    <section className="control-center">
      {error && <div className="control-toast"><AlertTriangle size={17} /><div><strong>작업을 진행하지 못했습니다.</strong><span>{error}</span></div><button onClick={() => setError(null)} aria-label="오류 닫기"><X size={15} /></button></div>}

      <div className="control-hero">
        <div>
          <p className="sim-eyebrow"><Cpu size={13} /> WINDOWS ALL-IN-ONE CONTROL CENTER</p>
          <h2>연결부터 뇌파 제어, 자동 파지까지 한 화면에서 진행합니다.</h2>
          <p>터미널 명령은 필요 없습니다. 아래 세 장치를 차례로 연결한 뒤 후보를 확인하고 자동 작업을 시작하세요.</p>
        </div>
        <div className="control-hero-actions">
          <span className={`control-ready-pill ${allConnected ? "ready" : ""}`}><i />{setupCount}/3 연결</span>
          <button className="control-ghost" onClick={openSimulation}><Eye size={16} />3D 시뮬레이션</button>
          <button className="control-ghost" onClick={openEeg}><Activity size={16} />EEG 전체 화면</button>
        </div>
      </div>

      <div className="control-setup-grid">
        {progress.map((item, index) => {
          const Icon = item.icon;
          return (
            <article key={item.label} className={item.ready ? "ready" : ""}>
              <span>{index + 1}</span><Icon size={21} />
              <div><strong>{item.label}</strong><small>{item.detail}</small></div>
              {item.ready ? <Check size={18} /> : <ChevronRight size={18} />}
              {index === 0 && (
                <button disabled={busy !== null || !eeg.apiOnline || !eeg.deviceReady} onClick={() => eeg.running ? stopEeg() : startEeg()}>{eeg.running ? "측정 정지" : "측정 시작"}</button>
              )}
              {index === 1 && (
                <button disabled={busy !== null} onClick={() => void act("camera", status?.camera.running ? "/api/control/camera/stop" : "/api/control/camera/start", { camera: settings?.camera ?? "auto" })}>{status?.camera.running ? "연결 해제" : "카메라 연결"}</button>
              )}
              {index === 2 && (
                <button disabled={busy !== null || !status?.camera.running} onClick={() => void act("arm", status?.arm.connected ? "/api/control/arm/disconnect" : "/api/control/arm/connect", { port: settings?.armPort ?? "auto" })}>{status?.arm.connected ? "연결 해제" : "Uno 연결"}</button>
              )}
            </article>
          );
        })}
      </div>

      <div className="control-main-grid">
        <div className="control-main-column">
          <article className="control-card camera-console">
            <div className="control-card-head">
              <div><p>WRIST RGB · FASTSAM MULTI-OBJECT</p><h3>실시간 손목 카메라와 파지 후보</h3></div>
              <div>
                <span className={`sim-live ${status?.camera.running ? "" : "stopped"}`}><i />{status?.camera.running ? "LIVE" : "OFFLINE"}</span>
                <button disabled={!status?.arm.connected || busy !== null || status?.task.running} onClick={() => void act("distance", "/api/control/arm/distance")}><Gauge size={14} />거리 측정</button>
                <button disabled={!status?.camera.running || busy !== null || status?.task.running} onClick={() => void act("detect", "/api/control/objects/detect")}><RefreshCw size={14} />다중 물체 찾기</button>
              </div>
            </div>
            <div className="control-camera-frame">
              {status?.camera.running ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img ref={frameRef} src={`${API_BASE}/api/control/camera/frame`} alt="실시간 손목 웹캠과 FastSAM 파지 후보 번호" />
              ) : (
                <div><Video size={36} /><strong>카메라 연결 대기</strong><span>손목 웹캠을 연결한 뒤 위의 카메라 연결 버튼을 누르세요.</span></div>
              )}
              <div className="camera-hud"><span><Crosshair size={13} />후보 {status?.detections.length ?? 0}개</span><span><Gauge size={13} />초음파 {status?.arm.distanceMm ? `${status.arm.distanceMm.toFixed(0)} mm` : "—"}</span></div>
            </div>
            <div className="candidate-strip">
              {(status?.detections.length ?? 0) > 0 ? status?.detections.map((item) => (
                <button key={item.index} className={`${selectedCandidate === item.index ? "selected" : ""} ${status?.task.rejectedCandidates.includes(item.index) ? "rejected" : ""}`} disabled={status?.task.running} onClick={() => setSelectedCandidate(item.index)}>
                  <span>#{item.index + 1}</span><div><strong>파지 후보 {item.index + 1}</strong><small>신뢰도 {(item.confidence * 100).toFixed(0)}% · 중심 {item.center.map(Math.round).join(", ")}</small></div>{selectedCandidate === item.index ? <Check size={16} /> : null}
                </button>
              )) : <div className="candidate-empty"><Crosshair size={18} /><span>카메라를 연결한 뒤 <b>다중 물체 찾기</b>를 누르세요.</span></div>}
            </div>
          </article>

          <article className="control-card task-console">
            <div className="task-state">
              <span className={`phase phase-${status?.task.phase ?? "idle"}`}><i />{phase[0]}</span>
              <div><strong>{phase[0]}</strong><small>{phase[1]}</small></div>
              {status?.task.activeCandidate != null && <b>현재 후보 #{status.task.activeCandidate + 1}</b>}
            </div>
            <div className="task-actions">
              <button className="task-start" disabled={!canStart || busy !== null || selectedRejected} onClick={() => void act("task", "/api/control/task/start", { candidateIndex: selectedCandidate, robotWeight: eeg.robotWeight })}><Play size={18} fill="currentColor" />선택 후보 자동 파지</button>
              <button className="task-reject" disabled={!status?.task.running || busy !== null} onClick={() => void act("reject", "/api/control/task/reject")}><Brain size={18} />아니야 · 다음 물체</button>
              <button className="task-stop" disabled={!status?.task.running || busy !== null} onClick={() => void act("stop", "/api/control/task/stop")}><CircleStop size={18} />작업 중지</button>
              <button className="emergency-stop" disabled={!status?.arm.connected} onClick={() => void act("emergency", "/api/control/arm/stop")}><AlertOctagon size={19} />긴급정지</button>
            </div>
            <p className="task-note">자동 접근은 실시간 영상 추적·2/3/4축 보정·초음파 정지·즉시 파지·HOME 복귀 순서입니다. 긴급 상황에서는 GUI 버튼과 함께 외부 서보 전원도 분리하세요.</p>
          </article>
        </div>

        <aside className="control-side-column">
          <article className="control-card eeg-bridge-card">
            <div className="control-card-head"><div><p>POLYG-I SHARED AUTONOMY</p><h3>뇌파 행동 반영</h3></div><Brain size={20} /></div>
            <div className="eeg-bridge-metrics">
              <div><span>ErrP 입력</span><strong>CH8 · 1–10 Hz</strong></div>
              <div><span>현재 P(error)</span><strong>{eeg.probability == null ? "판정 대기" : `${(eeg.probability * 100).toFixed(1)}%`}</strong></div>
              <div><span>즉시 거부 기준</span><strong>≥{(eeg.threshold * 100).toFixed(0)}%</strong></div>
              <div><span>TAR</span><strong>{eeg.tar == null ? "휴식 보정 필요" : `${eeg.tar.toFixed(3)} · Δ ${((eeg.relativeTar ?? 0) * 100).toFixed(1)}%`}</strong></div>
              <div><span>자율성</span><strong>로봇 {(eeg.robotWeight * 100).toFixed(0)} · 인간 {(eeg.humanWeight * 100).toFixed(0)}</strong></div>
            </div>
            <div className={`eeg-route-state ${eeg.running && eeg.baselineReady ? "ready" : ""}`}><Zap size={17} /><div><strong>{eeg.running && eeg.baselineReady ? "ErrP 자동 연결 준비됨" : "EEG 측정·보정 필요"}</strong><span>확정 ErrP는 후보 검토 중 다음 물체로 전환되며, 이동 중이면 현재 이동을 정지합니다.</span></div></div>
            <button onClick={openEeg}><Activity size={15} />파형·보정 화면 열기</button>
          </article>

          <article className="control-card manual-card">
            <div className="control-card-head"><div><p>MANUAL RECOVERY</p><h3>수동 관절 조작</h3></div><SlidersHorizontal size={19} /></div>
            <p>자동 작업이 멈춘 뒤에만 사용됩니다. 1번과 6번은 고정입니다.</p>
            {([[
              "shoulder", "2번 어깨", 0, 150,
            ], ["elbow", "3번 팔꿈치", 0, 180], ["wrist", "4번 손목", 130, 180], ["gripper", "집게", 90, 180]] as const).map(([key, label, min, max]) => (
              <label key={key}><span>{label}<b>{jog[key]}°</b></span><input type="range" min={min} max={max} value={jog[key]} disabled={!status?.arm.connected || status?.task.running} onChange={(event) => setJog({ ...jog, [key]: Number(event.target.value) })} /></label>
            ))}
            <div className="manual-actions"><button disabled={!status?.arm.connected || status?.task.running || busy !== null} onClick={() => void act("jog", "/api/control/arm/jog", jog)}><Hand size={15} />선택 각도로 이동</button><button disabled={!status?.arm.connected || status?.task.running || !homeConfirmed || busy !== null} onClick={() => void act("home", "/api/control/arm/home", { gripper: jog.gripper })}>HOME</button></div>
            <label className="home-confirm"><input type="checkbox" checked={homeConfirmed} onChange={(event) => setHomeConfirmed(event.target.checked)} /><span>주변에 손·케이블이 없고 HOME 경로가 비어 있음을 확인했습니다.</span></label>
            {armPose && <small>현재 명령 자세: {armPose.map((value, index) => `${index + 1}:${value}°`).join(" · ")}</small>}
          </article>

          <article className="control-card settings-card">
            <div className="control-card-head"><div><p>DEVICE & BEHAVIOR</p><h3>연결·판정 설정</h3></div><Settings2 size={19} /></div>
            {settings && <>
              <label><span>카메라 번호</span><input value={settings.camera} disabled={status?.camera.running} onChange={(event) => setSettings({ ...settings, camera: event.target.value })} placeholder="auto 또는 0" /></label>
              <label><span>Arduino 포트</span><input value={settings.armPort} disabled={status?.arm.connected} onChange={(event) => setSettings({ ...settings, armPort: event.target.value.toUpperCase() })} placeholder="AUTO 또는 COM5" /></label>
              <label><span>후보 검토 시간</span><input type="number" min="0.5" max="15" step="0.5" value={settings.candidateReviewSeconds} onChange={(event) => setSettings({ ...settings, candidateReviewSeconds: Number(event.target.value) })} /></label>
              <label><span>자동 작업 제한(초)</span><input type="number" min="10" max="300" value={settings.maxTaskSeconds} onChange={(event) => setSettings({ ...settings, maxTaskSeconds: Number(event.target.value) })} /></label>
              <label className="toggle"><span>ErrP 자동 반영</span><input type="checkbox" checked={settings.errpEnabled} onChange={(event) => setSettings({ ...settings, errpEnabled: event.target.checked, autoRejectPhysical: event.target.checked, autoRejectSimulation: event.target.checked })} /></label>
              <label className="toggle"><span>TAR 자율성 배분</span><input type="checkbox" checked={settings.tarEnabled} onChange={(event) => setSettings({ ...settings, tarEnabled: event.target.checked })} /></label>
              <button disabled={busy !== null} onClick={() => void saveSettings()}><Save size={15} />설정 저장</button>
            </>}
          </article>

          <article className="control-card event-card">
            <div className="control-card-head"><div><p>KOREAN ACTION LOG</p><h3>실행 기록</h3></div><button onClick={async () => { try { setDiagnostic(await request("/api/control/diagnose")); } catch (cause) { setError(String(cause)); } }}><RefreshCw size={14} />진단</button></div>
            {diagnostic && <div className="diagnostic-result"><Check size={15} /><span>움직임 없는 진단 완료 · FastSAM {diagnostic.fastsam ? "정상" : "없음"} · 카메라 {diagnostic.cameraFrame ? "LIVE" : "대기"}</span></div>}
            <div className="control-event-list">{status?.events.slice(0, 14).map((event) => <div key={event.id} className={event.kind}><i /><span>{event.at}</span><p>{event.text}</p></div>)}</div>
          </article>
        </aside>
      </div>
    </section>
  );
}
