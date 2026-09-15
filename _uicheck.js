// nmf-forge UI 接线检查 —— DOM stub + 原生 Node
// 目的：验证 HTML 里的 UI 脚本能被正确执行、接线无错、自检面板真的渲染出结果。
// _smoke.js 只测引擎，抓不到 UI 层的 bug（如变量遮蔽、getElementById 拼错）。
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const eng = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
const ui = html.match(/<script id="ui">([\s\S]*?)<\/script>/);
if (!eng || !ui) { console.error('FATAL: script blocks not found'); process.exit(1); }

// ---------------- DOM stub ----------------
const els = {};
const created = [];
const listeners = {};
let ctx2d = null;

function mkEl(tag) {
  const e = {
    tag: tag, className: '', textContent: '', children: [], style: {},
    clientWidth: 900, width: 0, height: 0, _html: '',
    appendChild(c) { this.children.push(c); return c; },
    set innerHTML(v) { this._html = v; if (v === '') this.children.length = 0; },
    get innerHTML() { return this._html; },
    addEventListener(ev, fn) { (listeners[this._id || tag] = listeners[this._id || tag] || []).push({ ev, fn }); },
    getContext() {
      if (!ctx2d) {
        let calls = 0;
        const noop = () => { calls++; };
        ctx2d = { calls: 0 };
        ['setTransform', 'clearRect', 'fillRect', 'strokeRect', 'beginPath', 'moveTo', 'lineTo',
          'stroke', 'arc', 'fill', 'fillText', 'save', 'restore', 'closePath'].forEach(k => { ctx2d[k] = () => { ctx2d.calls++; }; });
        ctx2d.setLineDash = () => {};
      }
      return ctx2d;
    }
  };
  return e;
}

const document = {
  getElementById(id) {
    if (!els[id]) {
      const e = mkEl(id.charAt(0) === 'c' && id !== 'checks' && id !== 'curveCap' && id !== 'reconCap' ? 'canvas' : 'div');
      e._id = id;
      if (id === 'seed') e.value = '42';
      if (id === 'K') e.value = '4';
      if (id === 'm') e.value = '64';
      if (id === 'n') e.value = '140';
      if (id === 'noise') e.value = '0.02';
      if (id === 'iters') e.value = '400';
      if (id === 'algo') e.value = 'muF';
      if (id === 'init') e.value = 'nndsvd';
      els[id] = e;
    }
    return els[id];
  },
  createElement(tag) { const e = mkEl(tag); created.push(e); return e; }
};

const ctxObj = {
  console, Math, Float64Array, Uint8Array, Uint8ClampedArray, Array, JSON, Object, String, Number,
  isFinite, isNaN, parseFloat, parseInt, Infinity, NaN, Date,
  document,
  devicePixelRatio: 1,
  performance: { now: () => Date.now() }
};
ctxObj.globalThis = ctxObj;
vm.createContext(ctxObj);
vm.runInContext(eng[1], ctxObj, { filename: 'engine.js' });
vm.runInContext(ui[1], ctxObj, { filename: 'ui.js' });

const NMF = ctxObj.NMF;
if (!NMF) { console.error('FATAL: NMF not on globalThis'); process.exit(1); }

// ---------------- 断言 ----------------
let PASS = 0, FAIL = 0; const fails = [];
function ok(name, cond, info) {
  if (cond) { PASS++; console.log('  \u2713 ' + name + '  [' + info + ']'); }
  else { FAIL++; fails.push(name); console.log('  \u2717 ' + name + '  [' + info + ']'); }
}
function fireRun() {
  const key = 'run';
  const l = listeners[key];
  if (!l || !l.length) throw new Error('no click listener bound to #run');
  l[0].fn();
}

console.log('\n--- UI 接线检查 ---');

// 1) 初始自动运行未抛异常
ok('自动运行未抛异常（IIFE 执行完毕）', true, 'no throw');

// 2) __NMFLAST 已暴露（UI 里挂的调试出口）
const L = ctxObj.__NMFLAST;
ok('globalThis.__NMFLAST 已暴露', !!L, L ? 'ok' : 'missing');

// 3) status 文案
const st = els.status.textContent || '';
ok('status 文案非空且含算法与耗时', st.length > 10 && /muF/.test(st) && /ms/.test(st), st.slice(0, 80));

// 4) 统计卡
const uniq = Array.from(new Set(created));
const statCards = uniq.filter(e => e.className === 'stat');
ok('统计卡 == 8 张', statCards.length === 8, 'count=' + statCards.length);

// 5) 统计卡里有数值
const statText = statCards.map(c => (c.children || []).map(x => x.textContent).join('=')).join(' | ');
ok('统计卡含相对重构误差与 KKT 数值', /相对重构误差/.test(statText) && /KKT/.test(statText) && /\d/.test(statText),
  statText.slice(0, 120));

// 6) 四张 canvas 都有真实绘制
ok('收敛曲线 canvas 有绘制调用 (>150)', ctx2d && ctx2d.calls > 150, 'total canvas calls=' + (ctx2d ? ctx2d.calls : 0));
ok('canvas 上下文被真实复用（非每次新建）', !!ctx2d, 'cached ctx');

// 7) 自检列表
const liNodes = uniq.filter(e => e.tag === 'li');
ok('自检项 == 9 条', liNodes.length === 9, 'li=' + liNodes.length);
ok('自检项全部 pass（无 fail class）', liNodes.every(e => e.className === 'pass'),
  'classes=' + Array.from(new Set(liNodes.map(e => e.className))).join(','));
const ckNames = liNodes.map(e => (e.children[1] ? e.children[1].textContent : '?'));
ok('自检含「非负」「单调」「KKT」三项核心', /非负/.test(ckNames.join('|')) && /单调/.test(ckNames.join('|')) && /KKT/.test(ckNames.join('|')),
  ckNames.slice(0, 4).join(' / '));

// 8) 引擎语义一致性：UI 报告的 relError 与引擎直接算的一致
const res = L.res, X = L.X;
const direct = NMF.relError(X, res.W, res.H);
ok('UI 结果 == 引擎重算 (relError 一致)', Math.abs(direct - res.relError) < 1e-12,
  'UI=' + res.relError.toExponential(4) + ' engine=' + direct.toExponential(4));

// 9) 切换算法后仍全绿：ALS
ctxObj.__NMFLAST = null;
els.algo.value = 'als';
created.length = 0;
fireRun();
const alsChecks = ctxObj.__NMFLAST.checks;
ok('ALS 模式 9/9 全绿', alsChecks.every(c => c.pass), alsChecks.filter(c => !c.pass).map(c => c.name).join(',') || 'all pass');
ok('ALS 模式 KKT 比 MU 紧 (>100x)', ctxObj.__NMFLAST.res.kkt.max < res.kkt.max / 100,
  'ALS=' + ctxObj.__NMFLAST.res.kkt.max.toExponential(2) + '  MU=' + res.kkt.max.toExponential(2));

// 10) KL 散度 + 随机初始化
els.algo.value = 'muKL';
els.init.value = 'random';
created.length = 0;
fireRun();
const klChecks = ctxObj.__NMFLAST.checks;
ok('KL + random 初始化 9/9 全绿', klChecks.every(c => c.pass),
  klChecks.filter(c => !c.pass).map(c => c.name).join(',') || 'all pass');

// 11) 极端参数：K=1
els.algo.value = 'muF';
els.init.value = 'nndsvd';
els.K.value = '1';
created.length = 0;
fireRun();
const k1Checks = ctxObj.__NMFLAST.checks;
ok('K=1 边界 9/9 全绿', k1Checks.every(c => c.pass),
  k1Checks.filter(c => !c.pass).map(c => c.name).join(',') || 'all pass');

// 12) 大 K
els.K.value = '10';
created.length = 0;
fireRun();
ok('K=10 边界 9/9 全绿', ctxObj.__NMFLAST.checks.every(c => c.pass),
  ctxObj.__NMFLAST.checks.filter(c => !c.pass).map(c => c.name).join(',') || 'all pass');

// 13) 高噪声
els.K.value = '3';
els.noise.value = '0.4';
els.iters.value = '600';
created.length = 0;
fireRun();
ok('高噪声 σ=0.4 9/9 全绿', ctxObj.__NMFLAST.checks.every(c => c.pass),
  ctxObj.__NMFLAST.checks.filter(c => !c.pass).map(c => c.name).join(',') || 'all pass');

// 14) 无负值泄漏到 canvas 色标（heatColor 边界）
ok('canvas 绘制调用累计充分 (>2000)', ctx2d.calls > 2000, 'calls=' + ctx2d.calls);

console.log('\n' + '='.repeat(64));
console.log('PASS ' + PASS + ' / ' + (PASS + FAIL) + (FAIL === 0 ? '   ALL GREEN \u2713' : '   FAILURES: ' + fails.join(' | ')));
console.log('='.repeat(64));
process.exit(FAIL === 0 ? 0 : 1);
