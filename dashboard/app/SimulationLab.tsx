"use client";

import {
  Activity,
  Box,
  Brain,
  Check,
  CircleAlert,
  Crosshair,
  Eye,
  Grip,
  Layers3,
  Pause,
  Play,
  Plus,
  RotateCcw,
  ScanSearch,
  Send,
  Sparkles,
  Trash2,
  Undo2,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8765";
const HOME = { x: 19, y: 50, angle: 0 };
const BASKET_DEFAULT = { x: 78, y: 50 };
const OBJECT_COLORS = ["#ff5d00", "#376dfa", "#078641", "#a839fd", "#f7097d", "#fca50e"];

type Shape = "circle" | "box" | "capsule";
type ObjectStatus = "table" | "held" | "basket";
type SignalSource = "manual" | "mock" | "polyg";
type Phase =
  | "idle"
  | "scanning"
  | "target"
  | "reaching"
  | "grasping"
  | "transporting"
  | "evaluating"
  | "returning"
  | "completed"
  | "paused";

type SimObject = {
  id: string;
  label: string;
  color: string;
  shape: Shape;
  x: number;
  y: number;
  originX: number;
  originY: number;
  size: number;
  status: ObjectStatus;
};

type ArmPose = { x: number; y: number; angle: number };
type EventItem = { id: number; kind: "info" | "move" | "error" | "success"; text: string; at: string };
type CameraDetection = { id: string; confidence: number; area: number };

const INITIAL_OBJECTS: SimObject[] = [
  { id: "object-1", label: "주황 원통", color: "#ff5d00", shape: "capsule", x: 47, y: 27, originX: 47, originY: 27, size: 7, status: "table" },
  { id: "object-2", label: "파란 블록", color: "#376dfa", shape: "box", x: 55, y: 46, originX: 55, originY: 46, size: 7, status: "table" },
  { id: "object-3", label: "초록 공", color: "#078641", shape: "circle", x: 44, y: 69, originX: 44, originY: 69, size: 7, status: "table" },
  { id: "object-4", label: "보라 캡슐", color: "#a839fd", shape: "capsule", x: 65, y: 72, originX: 65, originY: 72, size: 6, status: "table" },
];

const PHASE_COPY: Record<Phase, { title: string; detail: string }> = {
  idle: { title: "준비됨", detail: "물체를 배치하고 자동 실행을 시작하세요." },
  scanning: { title: "손목 카메라 탐색", detail: "카메라 시야 안의 물체 후보를 순서대로 확인합니다." },
  target: { title: "후보 제시 · ErrP 창", detail: "선택한 물체를 강조하고 사용자의 오류 반응을 기다립니다." },
  reaching: { title: "접근 중", detail: "카메라 중심을 유지하며 물체까지 이동합니다." },
  grasping: { title: "파지 중", detail: "집게를 정렬하고 물체를 들어 올립니다." },
  transporting: { title: "바구니로 운반", detail: "선택한 물체를 목표 위치로 옮깁니다." },
  evaluating: { title: "배송 확인 · ErrP 창", detail: "바구니에 놓인 결과에 대한 오류 반응을 다시 확인합니다." },
  returning: { title: "원위치 복귀", detail: "거부된 물체를 처음 놓였던 위치로 되돌립니다." },
  completed: { title: "한 사이클 완료", detail: "거부 신호가 없어 선택한 물체를 배송했습니다." },
  paused: { title: "일시정지", detail: "현재 장면을 유지하고 있습니다." },
};

function clamp(value: number, low: number, high: number) {
  return Math.max(low, Math.min(high, value));
}

function nowLabel() {
  return new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function distance(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function objectName(shape: Shape) {
  return shape === "circle" ? "공" : shape === "box" ? "블록" : "캡슐";
}

function drawObject(
  ctx: CanvasRenderingContext2D,
  object: SimObject,
  x: number,
  y: number,
  radius: number,
  active: boolean,
  rejected: boolean,
) {
  ctx.save();
  ctx.translate(x, y);
  ctx.shadowColor = "rgba(0,0,0,.18)";
  ctx.shadowBlur = active ? 18 : 8;
  ctx.shadowOffsetY = 4;
  ctx.fillStyle = object.color;
  ctx.strokeStyle = active ? "#ff5d00" : rejected ? "#e30f32" : "rgba(12,12,12,.22)";
  ctx.lineWidth = active ? 4 : rejected ? 3 : 1;
  ctx.beginPath();
  if (object.shape === "circle") {
    ctx.arc(0, 0, radius, 0, Math.PI * 2);
  } else if (object.shape === "box") {
    ctx.roundRect(-radius, -radius * 0.82, radius * 2, radius * 1.64, Math.max(3, radius * 0.24));
  } else {
    ctx.roundRect(-radius * 1.25, -radius * 0.62, radius * 2.5, radius * 1.24, radius);
  }
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.stroke();
  if (rejected) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-radius * 0.45, -radius * 0.45);
    ctx.lineTo(radius * 0.45, radius * 0.45);
    ctx.moveTo(radius * 0.45, -radius * 0.45);
    ctx.lineTo(-radius * 0.45, radius * 0.45);
    ctx.stroke();
  }
  ctx.restore();
}

function WorldCanvas({
  objects,
  arm,
  basket,
  activeId,
  rejected,
  running,
  onMoveObject,
  onMoveBasket,
  onSelect,
}: {
  objects: SimObject[];
  arm: ArmPose;
  basket: { x: number; y: number };
  activeId: string | null;
  rejected: Set<string>;
  running: boolean;
  onMoveObject: (id: string, x: number, y: number) => void;
  onMoveBasket: (x: number, y: number) => void;
  onSelect: (id: string | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ kind: "object" | "basket"; id?: string } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(bounds.width * ratio));
      canvas.height = Math.max(1, Math.floor(bounds.height * ratio));
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const width = bounds.width;
      const height = bounds.height;
      const sx = width / 100;
      const sy = height / 100;

      ctx.fillStyle = "#f7f7f5";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(12,12,12,.055)";
      ctx.lineWidth = 1;
      for (let x = 0; x <= 100; x += 5) {
        ctx.beginPath();
        ctx.moveTo(x * sx, 0);
        ctx.lineTo(x * sx, height);
        ctx.stroke();
      }
      for (let y = 0; y <= 100; y += 5) {
        ctx.beginPath();
        ctx.moveTo(0, y * sy);
        ctx.lineTo(width, y * sy);
        ctx.stroke();
      }

      const gripX = arm.x * sx;
      const gripY = arm.y * sy;
      const fovLength = 28 * Math.min(sx, sy);
      const fovHalf = Math.PI * 0.27;
      ctx.fillStyle = "rgba(255,93,0,.055)";
      ctx.strokeStyle = "rgba(255,93,0,.22)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(gripX, gripY);
      ctx.lineTo(gripX + Math.cos(arm.angle - fovHalf) * fovLength, gripY + Math.sin(arm.angle - fovHalf) * fovLength);
      ctx.arc(gripX, gripY, fovLength, arm.angle - fovHalf, arm.angle + fovHalf);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      const basketX = basket.x * sx;
      const basketY = basket.y * sy;
      const basketW = 15 * sx;
      const basketH = 22 * sy;
      ctx.fillStyle = "rgba(55,109,250,.07)";
      ctx.strokeStyle = "#376dfa";
      ctx.lineWidth = 2;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.roundRect(basketX - basketW / 2, basketY - basketH / 2, basketW, basketH, 12);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#376dfa";
      ctx.font = "700 11px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("목표 바구니", basketX, basketY - basketH / 2 - 9);

      const base = { x: 12 * sx, y: 50 * sy };
      const target = { x: gripX, y: gripY };
      const l1 = 38 * sx;
      const l2 = 38 * sx;
      const dx = target.x - base.x;
      const dy = target.y - base.y;
      const d = clamp(Math.hypot(dx, dy), 1, l1 + l2 - 1);
      const a = Math.atan2(dy, dx);
      const cosShoulder = clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1);
      const shoulder = a - Math.acos(cosShoulder);
      const elbow = { x: base.x + Math.cos(shoulder) * l1, y: base.y + Math.sin(shoulder) * l1 };
      ctx.lineCap = "round";
      ctx.strokeStyle = "#d2d2d0";
      ctx.lineWidth = 22;
      ctx.beginPath();
      ctx.moveTo(base.x, base.y);
      ctx.lineTo(elbow.x, elbow.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 13;
      ctx.stroke();
      [base, elbow].forEach((joint) => {
        ctx.fillStyle = "#202020";
        ctx.beginPath();
        ctx.arc(joint.x, joint.y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#ff5d00";
        ctx.beginPath();
        ctx.arc(joint.x, joint.y, 3, 0, Math.PI * 2);
        ctx.fill();
      });

      const visibleObjects = objects.filter((object) => object.status !== "held");
      visibleObjects.forEach((object) => {
        const inBasket = object.status === "basket";
        const ox = (inBasket ? basket.x + ((object.id.charCodeAt(object.id.length - 1) % 3) - 1) * 2.2 : object.x) * sx;
        const oy = (inBasket ? basket.y + ((object.id.charCodeAt(object.id.length - 1) % 2) ? 2 : -2) : object.y) * sy;
        drawObject(ctx, object, ox, oy, object.size * Math.min(sx, sy) * 0.48, object.id === activeId, rejected.has(object.id));
      });
      const held = objects.find((object) => object.status === "held");
      if (held) drawObject(ctx, held, gripX, gripY, held.size * Math.min(sx, sy) * 0.48, true, rejected.has(held.id));

      ctx.save();
      ctx.translate(gripX, gripY);
      ctx.rotate(arm.angle);
      ctx.fillStyle = "#202020";
      ctx.fillRect(-8, -6, 22, 12);
      ctx.fillStyle = "#376dfa";
      ctx.fillRect(8, -18, 23, 6);
      ctx.fillStyle = "#e30f32";
      ctx.fillRect(8, 12, 23, 6);
      ctx.fillStyle = "#ff5d00";
      ctx.beginPath();
      ctx.arc(-4, 0, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      ctx.fillStyle = "#949494";
      ctx.font = "500 10px system-ui";
      ctx.textAlign = "left";
      ctx.fillText(running ? "자동 제어 중 · 배치 잠김" : "물체와 바구니를 드래그해 배치하세요", 14, height - 14);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [objects, arm, basket, activeId, rejected, running]);

  const point = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return { x: ((event.clientX - bounds.left) / bounds.width) * 100, y: ((event.clientY - bounds.top) / bounds.height) * 100 };
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (running) return;
    const p = point(event);
    const hitObject = [...objects].reverse().find((object) => object.status === "table" && distance(object, p) <= object.size * 0.9);
    if (hitObject) {
      dragRef.current = { kind: "object", id: hitObject.id };
      onSelect(hitObject.id);
    } else if (distance(basket, p) <= 12) {
      dragRef.current = { kind: "basket" };
      onSelect(null);
    } else {
      onSelect(null);
    }
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current || running) return;
    const p = point(event);
    if (dragRef.current.kind === "basket") onMoveBasket(clamp(p.x, 66, 82), clamp(p.y, 18, 82));
    else if (dragRef.current.id) onMoveObject(dragRef.current.id, clamp(p.x, 31, 69), clamp(p.y, 12, 88));
  };
  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  return (
    <canvas
      ref={canvasRef}
      className="sim-world-canvas"
      aria-label="물체를 직접 배치할 수 있는 로봇 작업대 평면 시뮬레이션"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    />
  );
}

function CameraCanvas({
  objects,
  arm,
  activeId,
  rejected,
  onDetections,
}: {
  objects: SimObject[];
  arm: ArmPose;
  activeId: string | null;
  rejected: Set<string>;
  onDetections: (detections: CameraDetection[]) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const visible = useMemo(() => {
    const dir = { x: Math.cos(arm.angle), y: Math.sin(arm.angle) };
    return objects
      .filter((object) => object.status !== "basket")
      .map((object) => {
        if (object.status === "held") return { object, depth: 3, lateral: 0, confidence: 0.99 };
        const dx = object.x - arm.x;
        const dy = object.y - arm.y;
        const depth = dx * dir.x + dy * dir.y;
        const lateral = -dx * dir.y + dy * dir.x;
        const cone = Math.max(7, depth * 0.75);
        const confidence = clamp(1.05 - depth / 70 - Math.abs(lateral) / Math.max(20, cone * 2), 0.35, 0.98);
        return { object, depth, lateral, confidence };
      })
      .filter((item) => item.depth > 0 && item.depth < 58 && Math.abs(item.lateral) < Math.max(7, item.depth * 0.78))
      .sort((a, b) => b.depth - a.depth);
  }, [objects, arm]);

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
      const width = bounds.width;
      const height = bounds.height;
      ctx.fillStyle = "#dedbd4";
      ctx.fillRect(0, 0, width, height);
      const horizon = height * 0.16;
      const gradient = ctx.createLinearGradient(0, horizon, 0, height);
      gradient.addColorStop(0, "#c9c5bb");
      gradient.addColorStop(1, "#f5f2eb");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, horizon, width, height - horizon);
      ctx.strokeStyle = "rgba(70,60,45,.10)";
      for (let row = 0; row < 8; row += 1) {
        const y = horizon + ((row + 1) / 8) ** 1.7 * (height - horizon);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      visible.forEach(({ object, depth, lateral }) => {
        const near = 1 - depth / 65;
        const size = clamp(18 + near * 46, 18, 64);
        const x = width / 2 + (lateral / Math.max(depth, 6)) * width * 0.75;
        const y = horizon + (1 - depth / 62) * height * 0.73;
        drawObject(ctx, object, x, y, size * 0.48, object.id === activeId, rejected.has(object.id));
      });

      // Camera-only perception: re-read the rendered RGB pixels and recover a
      // bounding box for each learned object color. The controller receives
      // only this detection list; it does not select from world coordinates.
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const detections: Array<CameraDetection & { box: [number, number, number, number] }> = [];
      const sampleStep = Math.max(1, Math.round(ratio));
      const scanBottom = Math.floor((height - 42) * ratio);
      objects.filter((object) => object.status !== "basket").forEach((object) => {
        const hex = object.color.replace("#", "");
        const targetRgb = [
          Number.parseInt(hex.slice(0, 2), 16),
          Number.parseInt(hex.slice(2, 4), 16),
          Number.parseInt(hex.slice(4, 6), 16),
        ];
        let minX = canvas.width;
        let minY = canvas.height;
        let maxX = -1;
        let maxY = -1;
        let count = 0;
        for (let y = 0; y < scanBottom; y += sampleStep) {
          for (let x = 0; x < canvas.width; x += sampleStep) {
            const index = (y * canvas.width + x) * 4;
            const delta = Math.abs(pixels.data[index] - targetRgb[0])
              + Math.abs(pixels.data[index + 1] - targetRgb[1])
              + Math.abs(pixels.data[index + 2] - targetRgb[2]);
            if (delta > 52) continue;
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
            count += 1;
          }
        }
        if (count < 12 || maxX <= minX || maxY <= minY) return;
        const area = ((maxX - minX) * (maxY - minY)) / (ratio * ratio);
        detections.push({
          id: object.id,
          area,
          confidence: clamp(0.58 + Math.log10(Math.max(10, count)) * 0.11, 0.62, 0.99),
          box: [minX / ratio, minY / ratio, maxX / ratio, maxY / ratio],
        });
      });
      detections.forEach((detection) => {
        const object = objects.find((item) => item.id === detection.id);
        if (!object) return;
        const [x1, y1, x2, y2] = detection.box;
        const active = detection.id === activeId;
        ctx.strokeStyle = active ? "#ff5d00" : "rgba(32,32,32,.55)";
        ctx.lineWidth = active ? 2.5 : 1.25;
        ctx.strokeRect(x1 - 4, y1 - 4, x2 - x1 + 8, y2 - y1 + 8);
        if (active) {
          const labelWidth = Math.max(118, x2 - x1 + 8);
          ctx.fillStyle = "#ff5d00";
          ctx.fillRect(x1 - 4, Math.max(2, y1 - 25), labelWidth, 21);
          ctx.fillStyle = "#fff";
          ctx.font = "700 10px ui-monospace";
          ctx.textAlign = "left";
          ctx.fillText(`${object.label}  ${(detection.confidence * 100).toFixed(0)}%`, x1 + 2, Math.max(16, y1 - 11));
        }
      });
      onDetections(detections.map(({ id, confidence, area }) => ({ id, confidence, area })));
      ctx.fillStyle = "#376dfa";
      ctx.fillRect(width * 0.27, height - 27, width * 0.16, 9);
      ctx.fillStyle = "#e30f32";
      ctx.fillRect(width * 0.57, height - 27, width * 0.16, 9);
      ctx.strokeStyle = "rgba(255,255,255,.8)";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(width / 2, height * 0.34);
      ctx.lineTo(width / 2, height - 16);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(12,12,12,.72)";
      ctx.fillRect(10, 10, 118, 25);
      ctx.fillStyle = "#fff";
      ctx.font = "700 10px ui-monospace";
      ctx.textAlign = "left";
      ctx.fillText(`WRIST CAM · ${detections.length} DET`, 18, 26);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [visible, activeId, rejected, objects, onDetections]);

  return (
    <div className="sim-camera-wrap">
      <canvas ref={canvasRef} className="sim-camera-canvas" aria-label="로봇 손목 카메라 시뮬레이션과 물체 감지 결과" />
      <div className="sim-camera-legend"><span><i className="blue" />왼쪽 집게</span><span><i className="red" />오른쪽 집게</span><span><Crosshair size={12} />시각 중심선</span></div>
    </div>
  );
}

export default function SimulationLab({ eegRunning, apiOnline }: { eegRunning: boolean; apiOnline: boolean }) {
  const [objects, setObjects] = useState<SimObject[]>(INITIAL_OBJECTS);
  const [basket, setBasket] = useState(BASKET_DEFAULT);
  const [arm, setArm] = useState<ArmPose>(HOME);
  const [phase, setPhase] = useState<Phase>("idle");
  const [running, setRunning] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [rejectedIds, setRejectedIds] = useState<string[]>([]);
  const [signalSource, setSignalSource] = useState<SignalSource>("manual");
  const [events, setEvents] = useState<EventItem[]>([
    // SSR and the browser can have different clocks/time zones. Keep the
    // hydration snapshot deterministic; only user/runtime events use nowLabel().
    { id: 1, kind: "info", text: "시뮬레이션 작업실이 준비되었습니다.", at: "시작 전" },
  ]);
  const [cycle, setCycle] = useState(1);
  const [lastDeliveredId, setLastDeliveredId] = useState<string | null>(null);
  const [errpReady, setErrpReady] = useState(false);
  const [errpBusy, setErrpBusy] = useState(false);

  const objectsRef = useRef(objects);
  const armRef = useRef(arm);
  const basketRef = useRef(basket);
  const rejectedRef = useRef(new Set<string>());
  const runningRef = useRef(false);
  const rejectRef = useRef(false);
  const generationRef = useRef(0);
  const eventIdRef = useRef(2);
  const detectionsRef = useRef<CameraDetection[]>([]);

  const syncObjects = useCallback((updater: SimObject[] | ((previous: SimObject[]) => SimObject[])) => {
    setObjects((previous) => {
      const next = typeof updater === "function" ? updater(previous) : updater;
      objectsRef.current = next;
      return next;
    });
  }, []);
  const syncArm = useCallback((next: ArmPose) => {
    armRef.current = next;
    setArm(next);
  }, []);
  const addEvent = useCallback((text: string, kind: EventItem["kind"] = "info") => {
    setEvents((previous) => [{ id: eventIdRef.current++, kind, text, at: nowLabel() }, ...previous].slice(0, 18));
  }, []);
  const syncRejected = useCallback((next: Set<string>) => {
    rejectedRef.current = next;
    setRejectedIds([...next]);
  }, []);

  useEffect(() => {
    basketRef.current = basket;
  }, [basket]);
  useEffect(() => () => {
    generationRef.current += 1;
    runningRef.current = false;
  }, []);

  const pause = useCallback((ms: number, interruptible = true) => new Promise<boolean>((resolve) => {
    const token = generationRef.current;
    const started = performance.now();
    const tick = () => {
      if (token !== generationRef.current || !runningRef.current) return resolve(false);
      if (interruptible && rejectRef.current) return resolve(false);
      if (performance.now() - started >= ms) return resolve(true);
      window.setTimeout(tick, 40);
    };
    tick();
  }), []);

  const moveArm = useCallback((target: ArmPose, duration = 1200, interruptible = true) => new Promise<boolean>((resolve) => {
    const token = generationRef.current;
    const start = armRef.current;
    const started = performance.now();
    const tick = (time: number) => {
      if (token !== generationRef.current || !runningRef.current) return resolve(false);
      if (interruptible && rejectRef.current) return resolve(false);
      const amount = clamp((time - started) / duration, 0, 1);
      const smooth = amount * amount * (3 - 2 * amount);
      syncArm({
        x: start.x + (target.x - start.x) * smooth,
        y: start.y + (target.y - start.y) * smooth,
        angle: start.angle + (target.angle - start.angle) * smooth,
      });
      if (amount >= 1) resolve(true);
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }), [syncArm]);

  const pointAt = useCallback((from: { x: number; y: number }, to: { x: number; y: number }) => (
    Math.atan2(to.y - from.y, to.x - from.x)
  ), []);

  const evaluateErrp = useCallback(async (marker: string) => {
    if (signalSource !== "polyg") return false;
    if (!apiOnline || !eegRunning || !errpReady) {
      addEvent("PolyG-I 판정 조건이 준비되지 않아 자동 거부를 적용하지 않았습니다.", "error");
      return false;
    }
    setErrpBusy(true);
    try {
      const response = await fetch(`${API_BASE}/api/errp/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ marker }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "ErrP 판정 실패");
      addEvent(`ErrP ${marker}: ${(payload.probability * 100).toFixed(1)}% · ${payload.isError ? "거부" : "통과"}`, payload.isError ? "error" : "success");
      return Boolean(payload.isError);
    } catch (error) {
      addEvent(error instanceof Error ? error.message : String(error), "error");
      return false;
    } finally {
      setErrpBusy(false);
    }
  }, [signalSource, apiOnline, eegRunning, errpReady, addEvent]);

  const returnRejected = useCallback(async (target: SimObject) => {
    setPhase("returning");
    addEvent(`${target.label}: 거부 스택에 추가 · 원위치 복귀`, "error");
    const nextRejected = new Set(rejectedRef.current);
    nextRejected.add(target.id);
    syncRejected(nextRejected);
    let latest = objectsRef.current.find((item) => item.id === target.id) ?? target;
    if (latest.status === "basket") {
      const basketNow = basketRef.current;
      await moveArm({ x: basketNow.x, y: basketNow.y, angle: pointAt(armRef.current, basketNow) }, 850, false);
      syncObjects((current) => current.map((item) => item.id === target.id ? { ...item, status: "held" } : item));
      await pause(280, false);
      latest = { ...latest, status: "held" };
    }
    if (latest.status === "held") {
      const origin = { x: target.originX, y: target.originY };
      await moveArm({ ...origin, angle: pointAt(armRef.current, origin) }, 1450, false);
      syncObjects((current) => current.map((item) => item.id === target.id ? { ...item, x: item.originX, y: item.originY, status: "table" } : item));
      await pause(260, false);
      addEvent(`${target.label}: 원래 자리 (${target.originX.toFixed(0)}, ${target.originY.toFixed(0)})에 내려놓음`, "move");
      await moveArm(HOME, 820, false);
    } else {
      await moveArm(HOME, 900, false);
    }
    rejectRef.current = false;
    setLastDeliveredId((current) => current === target.id ? null : current);
    setActiveId(null);
  }, [addEvent, moveArm, pause, pointAt, syncObjects, syncRejected]);

  const runAutomation = useCallback(async () => {
    while (runningRef.current) {
      let available = objectsRef.current.filter((item) => item.status === "table" && !rejectedRef.current.has(item.id));
      if (!available.length) {
        const tableObjects = objectsRef.current.filter((item) => item.status === "table");
        if (!tableObjects.length) {
          setPhase("completed");
          setRunning(false);
          runningRef.current = false;
          addEvent("테이블에 처리할 물체가 없습니다.", "success");
          return;
        }
        syncRejected(new Set());
        setCycle((value) => value + 1);
        addEvent("모든 물체가 거부되어 거부 스택을 초기화했습니다.", "info");
        await pause(650, false);
        available = tableObjects;
      }

      setPhase("scanning");
      setActiveId(null);
      const seen = new Map<string, CameraDetection>();
      for (const heading of [-0.85, 0, 0.85]) {
        await moveArm({ ...armRef.current, angle: heading }, 360, false);
        await pause(180, false);
        detectionsRef.current.forEach((detection) => {
          if (!rejectedRef.current.has(detection.id)) {
            const previous = seen.get(detection.id);
            if (!previous || detection.confidence > previous.confidence) seen.set(detection.id, detection);
          }
        });
      }
      const candidates = available.filter((item) => seen.has(item.id));
      if (!candidates.length) {
        setPhase("paused");
        setRunning(false);
        runningRef.current = false;
        addEvent("손목 카메라 스캔에서 선택 가능한 물체를 찾지 못했습니다.", "error");
        return;
      }
      candidates.sort((a, b) => {
        const detectionDelta = (seen.get(b.id)?.confidence ?? 0) - (seen.get(a.id)?.confidence ?? 0);
        return Math.abs(detectionDelta) > 0.02 ? detectionDelta : distance(armRef.current, a) - distance(armRef.current, b);
      });
      const target = candidates[0];
      setActiveId(target.id);
      setSelectedId(target.id);
      addEvent(`RGB 픽셀 감지: ${target.label} 후보 발견 · ${((seen.get(target.id)?.confidence ?? 0) * 100).toFixed(0)}%`, "info");
      const targetAngle = pointAt(armRef.current, target);
      await moveArm({ ...armRef.current, angle: targetAngle }, 520, true);
      if (rejectRef.current) {
        await returnRejected(target);
        continue;
      }

      setPhase("target");
      addEvent(`후보 제시: ${target.label} · 선택 오류 반응 확인`, "move");
      const errorAtSelection = await evaluateErrp("SIM_TARGET_PRESENTED");
      if (errorAtSelection) rejectRef.current = true;
      else if (signalSource !== "polyg") await pause(900, true);
      if (rejectRef.current) {
        await returnRejected(target);
        continue;
      }

      setPhase("reaching");
      addEvent(`${target.label}에 카메라 중심을 유지하며 접근`, "move");
      const reached = await moveArm({ x: target.x, y: target.y, angle: targetAngle }, 1500, true);
      if (!reached && rejectRef.current) {
        await returnRejected(target);
        continue;
      }
      setPhase("grasping");
      await pause(320, true);
      if (rejectRef.current) {
        await returnRejected(target);
        continue;
      }
      syncObjects((current) => current.map((item) => item.id === target.id ? { ...item, status: "held" } : item));
      addEvent(`${target.label} 파지 확인 · 집게 사이에 유지`, "success");
      await pause(300, true);
      if (rejectRef.current) {
        await returnRejected({ ...target, status: "held" });
        continue;
      }

      setPhase("transporting");
      const basketNow = basketRef.current;
      addEvent(`${target.label}을 목표 바구니로 운반`, "move");
      const transported = await moveArm({ x: basketNow.x, y: basketNow.y, angle: pointAt(armRef.current, basketNow) }, 1750, true);
      if (!transported && rejectRef.current) {
        await returnRejected({ ...target, status: "held" });
        continue;
      }
      syncObjects((current) => current.map((item) => item.id === target.id ? { ...item, status: "basket" } : item));
      setLastDeliveredId(target.id);
      setPhase("evaluating");
      addEvent(`${target.label} 바구니 도착 · 배송 결과 ErrP 확인`, "success");

      const errorAfterDrop = await evaluateErrp("SIM_BASKET_DROP");
      if (errorAfterDrop) rejectRef.current = true;
      else if (signalSource !== "polyg") await pause(2600, true);
      if (rejectRef.current) {
        await returnRejected({ ...target, status: "basket" });
        continue;
      }
      setPhase("completed");
      setRunning(false);
      runningRef.current = false;
      setActiveId(target.id);
      addEvent(`${target.label}: 거부 신호 없음 · 배송 확정`, "success");
      return;
    }
  }, [addEvent, evaluateErrp, moveArm, pause, pointAt, returnRejected, signalSource, syncObjects, syncRejected]);

  const start = useCallback(() => {
    if (runningRef.current) return;
    generationRef.current += 1;
    runningRef.current = true;
    rejectRef.current = false;
    setRunning(true);
    setPhase("scanning");
    addEvent(`자동 사이클 ${cycle} 시작`, "info");
    void runAutomation();
  }, [addEvent, cycle, runAutomation]);

  const stop = useCallback(() => {
    generationRef.current += 1;
    runningRef.current = false;
    rejectRef.current = false;
    setRunning(false);
    setPhase("paused");
    addEvent("시뮬레이션을 일시정지했습니다.", "info");
  }, [addEvent]);

  const reject = useCallback(() => {
    if (runningRef.current && activeId) {
      rejectRef.current = true;
      addEvent("외부 거부 신호 수신", "error");
      return;
    }
    if (lastDeliveredId) {
      const target = objectsRef.current.find((item) => item.id === lastDeliveredId);
      if (!target) return;
      generationRef.current += 1;
      runningRef.current = true;
      rejectRef.current = true;
      setRunning(true);
      setActiveId(target.id);
      addEvent("배송 완료 후 거부 신호 수신 · 바구니 회수 시작", "error");
      void (async () => {
        await returnRejected({ ...target, status: "basket" });
        if (runningRef.current) void runAutomation();
      })();
    }
  }, [activeId, addEvent, lastDeliveredId, returnRejected, runAutomation]);

  const reset = useCallback(() => {
    generationRef.current += 1;
    runningRef.current = false;
    rejectRef.current = false;
    setRunning(false);
    setObjects(INITIAL_OBJECTS);
    objectsRef.current = INITIAL_OBJECTS;
    setBasket(BASKET_DEFAULT);
    basketRef.current = BASKET_DEFAULT;
    syncArm(HOME);
    syncRejected(new Set());
    setActiveId(null);
    setSelectedId(null);
    setLastDeliveredId(null);
    setCycle(1);
    setPhase("idle");
    addEvent("작업대와 거부 스택을 초기 상태로 되돌렸습니다.", "info");
  }, [addEvent, syncArm, syncRejected]);

  const addObject = useCallback((shape: Shape) => {
    if (running || objects.length >= 9) return;
    const index = objects.length;
    const id = `object-${Date.now()}`;
    const x = 38 + (index % 4) * 8;
    const y = 20 + ((index * 19) % 62);
    const object: SimObject = {
      id,
      label: `${OBJECT_COLORS[index % OBJECT_COLORS.length] === "#ff5d00" ? "주황" : "새"} ${objectName(shape)}`,
      color: OBJECT_COLORS[index % OBJECT_COLORS.length],
      shape,
      x,
      y,
      originX: x,
      originY: y,
      size: 7,
      status: "table",
    };
    syncObjects((current) => [...current, object]);
    setSelectedId(id);
    addEvent(`${object.label}을 작업대에 추가했습니다.`, "info");
  }, [running, objects.length, syncObjects, addEvent]);

  const deleteSelected = useCallback(() => {
    if (running || !selectedId) return;
    const target = objectsRef.current.find((item) => item.id === selectedId);
    syncObjects((current) => current.filter((item) => item.id !== selectedId));
    const next = new Set(rejectedRef.current);
    next.delete(selectedId);
    syncRejected(next);
    setSelectedId(null);
    if (target) addEvent(`${target.label}을 작업대에서 제거했습니다.`, "info");
  }, [running, selectedId, syncObjects, syncRejected, addEvent]);

  const calibrateErrp = useCallback(async () => {
    if (!apiOnline || !eegRunning) return;
    setErrpBusy(true);
    addEvent("최근 8초 휴식 EEG로 ErrP 기준을 계산합니다.", "info");
    try {
      const response = await fetch(`${API_BASE}/api/errp/calibrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seconds: 8 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "ErrP 보정 실패");
      setErrpReady(true);
      addEvent(`ErrP 휴식 보정 완료 · ${payload.samples} samples`, "success");
    } catch (error) {
      addEvent(error instanceof Error ? error.message : String(error), "error");
    } finally {
      setErrpBusy(false);
    }
  }, [apiOnline, eegRunning, addEvent]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.code === "Space") {
        event.preventDefault();
        if (runningRef.current) stop();
        else start();
      }
      if (event.key.toLowerCase() === "x") reject();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [start, stop, reject]);

  const active = objects.find((item) => item.id === activeId);
  const selected = objects.find((item) => item.id === selectedId);
  const rejected = useMemo(() => new Set(rejectedIds), [rejectedIds]);
  const phaseCopy = PHASE_COPY[phase];
  const handleDetections = useCallback((detections: CameraDetection[]) => {
    detectionsRef.current = detections;
  }, []);

  return (
    <section className="sim-lab">
      <div className="sim-hero">
        <div>
          <p className="sim-eyebrow"><Sparkles size={13} /> SHARED AUTONOMY SANDBOX</p>
          <h2>물체를 놓고, 로봇의 선택을 지켜보세요.</h2>
          <p>실물 없이도 카메라 탐색부터 파지·배송·ErrP 거부·원위치 복귀까지 한 사이클을 반복 검증합니다.</p>
        </div>
        <div className="sim-hero-actions">
          <span className={`sim-state-pill phase-${phase}`}><i />{phaseCopy.title}</span>
          {!running ? (
            <button className="sim-primary" onClick={start} disabled={!objects.some((item) => item.status === "table")}><Play size={17} fill="currentColor" />자동 실행</button>
          ) : (
            <button className="sim-secondary" onClick={stop}><Pause size={17} />일시정지</button>
          )}
          <button className="sim-icon-button" onClick={reset} aria-label="시뮬레이션 초기화"><RotateCcw size={17} /></button>
        </div>
      </div>

      <div className="sim-status-strip">
        <div><span>현재 단계</span><strong>{phaseCopy.title}</strong><small>{phaseCopy.detail}</small></div>
        <div><span>선택 후보</span><strong>{active?.label ?? "—"}</strong><small>{active ? `원위치 ${active.originX.toFixed(0)}, ${active.originY.toFixed(0)}` : "카메라 탐색 대기"}</small></div>
        <div><span>거부 스택</span><strong>{rejected.size} / {objects.filter((item) => item.status !== "basket").length || objects.length}</strong><small>모두 거부되면 자동 초기화</small></div>
        <div><span>실험 사이클</span><strong>#{cycle}</strong><small>{signalSource === "polyg" ? "PolyG-I ErrP" : signalSource === "mock" ? "가상 ErrP" : "수동 외부 신호"}</small></div>
      </div>

      <div className="sim-layout">
        <div className="sim-main-column">
          <article className="sim-card sim-workspace-card">
            <div className="sim-card-head">
              <div><p>INTERACTIVE TABLE</p><h3>로봇 작업대</h3></div>
              <div className="sim-object-tools">
                <span>물체 추가</span>
                <button onClick={() => addObject("circle")} disabled={running || objects.length >= 9}><Plus size={13} />공</button>
                <button onClick={() => addObject("box")} disabled={running || objects.length >= 9}><Plus size={13} />블록</button>
                <button onClick={() => addObject("capsule")} disabled={running || objects.length >= 9}><Plus size={13} />캡슐</button>
                <button className="danger" onClick={deleteSelected} disabled={running || !selected}><Trash2 size={13} />선택 삭제</button>
              </div>
            </div>
            <WorldCanvas
              objects={objects}
              arm={arm}
              basket={basket}
              activeId={activeId}
              rejected={rejected}
              running={running}
              onMoveObject={(id, x, y) => syncObjects((current) => current.map((item) => item.id === id ? { ...item, x, y, originX: x, originY: y } : item))}
              onMoveBasket={(x, y) => setBasket({ x, y })}
              onSelect={setSelectedId}
            />
          </article>

          <div className="sim-lower-row">
            <article className="sim-card sim-camera-card">
              <div className="sim-card-head">
                <div><p>EYE IN HAND</p><h3>손목 카메라 · 감지 결과</h3></div>
                <span className="sim-live"><i />LIVE SIM</span>
              </div>
              <CameraCanvas
                objects={objects}
                arm={arm}
                activeId={activeId}
                rejected={rejected}
                onDetections={handleDetections}
              />
            </article>

            <article className="sim-card sim-stack-card">
              <div className="sim-card-head"><div><p>REJECTION MEMORY</p><h3>거부된 물체</h3></div><Layers3 size={18} /></div>
              <div className="sim-rejected-list">
                {rejectedIds.length ? rejectedIds.map((id, index) => {
                  const object = objects.find((item) => item.id === id);
                  return object ? (
                    <div key={id}><span style={{ background: object.color }}>{index + 1}</span><div><strong>{object.label}</strong><small>원위치 복귀 완료 · 재선택 금지</small></div><X size={15} /></div>
                  ) : null;
                }) : <div className="sim-empty"><Check size={20} /><span>아직 거부된 물체가 없습니다.</span></div>}
              </div>
              <p className="sim-note">스택의 모든 물체가 거부되면 목록을 비우고 첫 후보부터 새 사이클을 시작합니다.</p>
            </article>
          </div>
        </div>

        <aside className="sim-side-column">
          <article className="sim-card sim-signal-card">
            <div className="sim-card-head"><div><p>EXTERNAL CORRECTION</p><h3>거부 신호</h3></div><Brain size={20} /></div>
            <div className="sim-signal-tabs">
              <button className={signalSource === "manual" ? "active" : ""} onClick={() => setSignalSource("manual")} disabled={running}>수동</button>
              <button className={signalSource === "mock" ? "active" : ""} onClick={() => setSignalSource("mock")} disabled={running}>가상 ErrP</button>
              <button className={signalSource === "polyg" ? "active" : ""} onClick={() => setSignalSource("polyg")} disabled={running}>PolyG-I</button>
            </div>
            <button className="sim-reject-button" onClick={reject} disabled={!activeId && !lastDeliveredId}>
              <CircleAlert size={21} />
              <span>{signalSource === "manual" ? "아니야 · 다른 물체" : signalSource === "mock" ? "가상 ErrP 보내기" : "수동 거부 보조"}</span>
              <kbd>X</kbd>
            </button>
            <p className="sim-helper">이동 중에도, 집은 뒤에도, 바구니에 놓은 뒤에도 누를 수 있습니다. 로봇은 물체를 원래 좌표로 되돌린 뒤 다음 후보를 선택합니다.</p>

            {signalSource === "polyg" && (
              <div className={`sim-eeg-box ${errpReady ? "ready" : ""}`}>
                <div><span><Activity size={15} />PolyG-I 연결</span><strong>{apiOnline && eegRunning ? "측정 중" : "준비 안 됨"}</strong></div>
                <div><span><Zap size={15} />휴식 기준</span><strong>{errpReady ? "보정 완료" : "보정 필요"}</strong></div>
                <button onClick={calibrateErrp} disabled={!apiOnline || !eegRunning || errpBusy}>{errpBusy ? "계산 중…" : "최근 8초로 ErrP 보정"}</button>
                <small>후보 제시와 바구니 도착 순간마다 0.2초 이전 + 0.8초 이후의 실제 EEG epoch를 판정합니다.</small>
              </div>
            )}
          </article>

          <article className="sim-card sim-object-card">
            <div className="sim-card-head"><div><p>OBJECT INSPECTOR</p><h3>선택한 물체</h3></div><Box size={18} /></div>
            {selected ? (
              <div className="sim-object-editor">
                <div className="sim-object-preview" style={{ background: selected.color }} />
                <label>이름<input value={selected.label} disabled={running} onChange={(event) => syncObjects((current) => current.map((item) => item.id === selected.id ? { ...item, label: event.target.value } : item))} /></label>
                <label>색상<input type="color" value={selected.color} disabled={running} onChange={(event) => syncObjects((current) => current.map((item) => item.id === selected.id ? { ...item, color: event.target.value } : item))} /></label>
                <label>크기<input type="range" min="4" max="11" value={selected.size} disabled={running} onChange={(event) => syncObjects((current) => current.map((item) => item.id === selected.id ? { ...item, size: Number(event.target.value) } : item))} /></label>
                <small>좌표 {selected.x.toFixed(1)}, {selected.y.toFixed(1)} · {selected.status === "table" ? "테이블" : selected.status === "held" ? "집게에 파지" : "바구니"}</small>
              </div>
            ) : <div className="sim-empty tall"><Grip size={22} /><span>작업대에서 물체를 선택하세요.</span></div>}
          </article>

          <article className="sim-card sim-event-card">
            <div className="sim-card-head"><div><p>ACTION TRACE</p><h3>실행 기록</h3></div><ScanSearch size={18} /></div>
            <div className="sim-event-list">
              {events.map((event) => (
                <div key={event.id} className={event.kind}><i /> <span>{event.at}</span><p>{event.text}</p></div>
              ))}
            </div>
          </article>
        </aside>
      </div>

      <div className="sim-bottom-callout">
        <div><Eye size={19} /><span><strong>카메라 기반 선택</strong> 작업대의 정답 ID를 직접 읽지 않고 손목 시야에 들어온 후보를 순차 평가합니다.</span></div>
        <div><Undo2 size={19} /><span><strong>가역적 행동</strong> 늦은 거부도 바구니 회수 → 원위치 복귀 → 다음 후보로 이어집니다.</span></div>
        <div><Send size={19} /><span><strong>내일 연결</strong> PolyG-I 측정·휴식 보정 후 신호 소스만 바꾸면 같은 상태 머신을 사용합니다.</span></div>
      </div>
    </section>
  );
}
