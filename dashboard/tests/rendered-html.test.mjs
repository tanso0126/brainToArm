import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the shared-autonomy simulation studio", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>brainToArm · Shared Autonomy Studio<\/title>/i);
  assert.match(html, /Simulation Studio/);
  assert.match(html, /시뮬레이션 작업실/);
  assert.match(html, /MuJoCo 3D 엔진에 연결하는 중입니다/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("starter preview is removed and localhost API is explicit", async () => {
  const [page, simulationLab, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/SimulationLab.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  assert.match(page, /http:\/\/127\.0\.0\.1:8765/);
  assert.match(page, /mV_ADC_filtered/);
  assert.match(page, /공통 고정 축/);
  assert.match(page, /채널별 자동 축/);
  assert.match(page, /98백분위 절대 진폭/);
  assert.match(page, /function independentAutoScale/);
  assert.match(page, /const quantum = magnitude \/ 10/);
  assert.match(page, /CH1–4 θ \/ CH8 α/);
  assert.match(page, /저장 안정 기준 불러오기/);
  assert.match(page, /측정 시작 시 자동 적용/);
  assert.match(page, /보통 ErrP 기준은 50% 고정/);
  assert.match(simulationLab, /정지 버튼을 누를 때까지/);
  assert.match(simulationLab, /비동기 슬라이딩 · 정지할 때까지/);
  assert.match(simulationLab, /\/api\/errp\/async/);
  assert.match(simulationLab, /useState<SignalSource>\("polyg"\)/);
  assert.match(simulationLab, /window\.setInterval\(\(\) => void pull\(\), 50\)/);
  assert.match(simulationLab, /비동기 1창/);
  assert.match(simulationLab, /한 창이 50%를 넘는 즉시/);
  assert.match(page, /샘플 수는 안정도 점수가 아니며/);
  assert.match(page, /\/api\/baseline\/load/);
  assert.match(page, /index 0 · ×0\.10/);
  assert.match(page, /index 2 · ×0\.40/);
  assert.match(page, /필터 p-p는 진단값일 뿐 ADC 포화 판정에 사용하지 않습니다/);
  assert.match(page, /강한 ≥/);
  assert.match(page, /즉시/);
  assert.match(page, /requestAnimationFrame/);
  assert.match(page, /부드럽게 · 0\.45초/);
  assert.match(page, /SimulationLab/);
  assert.match(packageJson, /braintoarm-eeg-dashboard/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)));
});
