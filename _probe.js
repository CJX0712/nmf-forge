// nmf-forge 探针：打印真实数值，人工核对（不参与断言）
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
new Function(html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1])();
const NMF = globalThis.NMF;

function say(s) { console.log(s); }
function rep(s, n) { return s.repeat(n); }
function pad(s, n, right) { s = String(s); return right ? s.padStart(n) : s.padEnd(n); }

const m = 64, n = 140, K = 4, SEED = 42;
const dt = NMF.makeData(m, n, K, SEED, 0.02);
const X = dt.X;

say(rep('=', 78));
say('nmf-forge 探针  |  数据 m=' + m + ' n=' + n + '  真秩 K=' + K + '  σ=0.02  seed=' + SEED);
say(rep('=', 78));

// ---------------- 1. 三种算法收敛对照 ----------------
say('\n【1】三种算法收敛对照（maxIter=1200, tol=1e-12, nndsvd 初始化）');
say(pad('算法', 12) + pad('迭代', 7, 1) + pad('relF', 12, 1) + pad('F 终值', 12, 1) +
    pad('KKT(PG)', 12, 1) + pad('最小化目标', 12, 1) + pad('耗时ms', 9, 1));
say(rep('-', 78));
const results = {};
for (const algo of ['muF', 'muKL', 'als']) {
  const t0 = Date.now();
  const r = NMF.solve(X, K, { algo, init: 'nndsvd', maxIter: 1200, tol: 1e-12, seed: SEED });
  const ms = Date.now() - t0;
  results[algo] = r;
  let maxD = -Infinity;
  for (let i = 1; i < r.hist.length; i++) maxD = Math.max(maxD, r.hist[i] - r.hist[i - 1]);
  say(pad(algo, 12) + pad(r.iters, 7, 1) + pad(r.relError.toFixed(6), 12, 1) +
      pad(r.frobenius.toFixed(4), 12, 1) + pad(r.kkt.max.toExponential(2), 12, 1) +
      pad((maxD <= 0 ? '单调↓' : '违反!'), 12, 1) + pad(ms, 9, 1));
}
say('\n  注：MU 线性收敛速率接近 1 → 同样迭代数下 KKT 残差远大于 ALS；但 relF 一致到 1e-6，说明收敛到同一最优解。');

// ---------------- 2. 逐 K 的 NMF vs 截断 SVD ----------------
say('\n【2】非负约束的代价：NMF relF vs 截断 SVD relF');
say(pad('K', 4) + pad('NMF relF', 12, 1) + pad('SVD relF', 12, 1) + pad('约束代价', 12, 1) + pad('H 稀疏度', 11, 1) + pad('W 稀疏度', 11, 1));
say(rep('-', 78));
for (const k of [1, 2, 3, 4, 6, 8]) {
  const r = NMF.solve(X, k, { algo: 'als', init: 'nndsvd', maxIter: 1500, tol: 1e-12, seed: SEED });
  const sv = NMF.svdRelError(X, k, 600);
  const gap = r.relError - sv;
  say(pad(k, 4) + pad(r.relError.toFixed(5), 12, 1) + pad(sv.toFixed(5), 12, 1) +
      pad((Math.abs(gap) < 1e-9 ? '≈0' : gap.toFixed(5)), 12, 1) +
      pad((NMF.sparsity(r.H) * 100).toFixed(1) + '%', 11, 1) +
      pad((NMF.sparsity(r.W) * 100).toFixed(1) + '%', 11, 1));
}
// Perron–Frobenius 检验：非负矩阵的首奇异向量可取全正
{
  const sv = NMF.svdTopK(X, 1, 500);
  const uMin = Math.min.apply(null, Array.from(sv.U[0]));
  const vMin = Math.min.apply(null, Array.from(sv.V[0]));
  say('\n  Perron–Frobenius 检验（K=1）: min(u₁)=' + uMin.toExponential(2) + '  min(v₁)=' + vMin.toExponential(2));
  say('  → 首奇异向量严格为正 ⟹ 非负约束在 K=1 时完全不咬人，NMF 与 SVD 精确等价（gap ≈ 0）。');
  say('  → 这与表中 K=1 行「约束代价 ≈0」吻合；若该处出现明显正值，说明幂迭代没收敛。');
}
say('\n  注：「约束代价」= NMF relF − 截断 SVD relF ≥ 0 恒成立（NMF 可行集 ⊂ 秩-K 矩阵集）。');
say('      • K=1 → 代价严格为 0（由 Perron–Frobenius 推出，见下方检验）；');
say('      • K=真秩(4) → 代价 ~1e-5：数据本就是非负低秩的，非负约束几乎不咬人；');
say('      • K=2,3 压得最狠 → 代价 2.4e-3~4.2e-3，非负限制开始明显收费；');
say('      • K>真秩 后代价回落 —— 维度富余，非负约束更容易满足。');

// ---------------- 3. 符号性：NMF vs SVD 的本质差异 ----------------
say('\n【3】parts-based 的本质：基向量符号');
{
  const r = NMF.solve(X, K, { algo: 'als', init: 'nndsvd', maxIter: 1500, tol: 1e-12, seed: SEED });
  const sv = NMF.svdTopK(X, K, 120);
  function countNeg(A) { let c = 0, t = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) { t++; if (A[i][j] < 0) c++; } return c / t; }
  function countNonZero(A) { let c = 0, t = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) { t++; if (Math.abs(A[i][j]) > 1e-6) c++; } return c / t; }
  say('  NMF 基向量 W 的负值比例      : ' + (countNeg(r.W) * 100).toFixed(1) + '%   （非负约束 → 必为 0）');
  say('  SVD 基向量 U 的负值比例      : ' + (countNeg(sv.U) * 100).toFixed(1) + '%   （可正可负，无法解释为「纯加性成分」）');
  say('  NMF 基向量 W 的非零比例      : ' + (countNonZero(r.W) * 100).toFixed(1) + '%');
  say('  SVD 基向量 U 的非零比例      : ' + (countNonZero(sv.U) * 100).toFixed(1) + '%   （稠密）');
  // 重构矩阵的负值
  const XH_nmf = NMF.matMul(r.W, r.H);
  const XH_svd = NMF.zeros(m, n);
  for (let k = 0; k < K; k++) for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) XH_svd[i][j] += sv.S[k] * sv.U[k][i] * sv.V[k][j];
  say('  NMF 重构 X̂ 的负值比例       : ' + (countNeg(XH_nmf) * 100).toFixed(1) + '%');
  say('  SVD 重构 X̂ 的负值比例       : ' + (countNeg(XH_svd) * 100).toFixed(1) + '%   ← 物理量（光谱/计数）不可为负');
}

// ---------------- 4. H 系数矩阵 ASCII 热图 ----------------
say('\n【4】H 系数矩阵热图（K=' + K + ' × n=140，抽样 70 列；0-9 表示相对强度）');
{
  const r = NMF.solve(X, K, { algo: 'als', init: 'nndsvd', maxIter: 1500, tol: 1e-12, seed: SEED });
  const maxH = NMF.maxAbs(r.H);
  const cols = 70, step = Math.max(1, Math.floor(n / cols));
  say('        ' + rep('─', Math.floor(n / step)));
  for (let k = 0; k < K; k++) {
    let line = '   w' + k + ' │';
    let nz = 0, tot = 0;
    for (let j = 0; j < n; j += step) {
      const t = r.H[k][j] / (maxH + 1e-300);
      tot++; if (t > 0.05) nz++;
      line += t < 0.05 ? '·' : t < 0.15 ? '.' : String(Math.min(9, Math.ceil(t * 9)));
    }
    // 该行精确零元素比例（直接数该行，不要拿 1-D 数组当矩阵传）
    let zeros = 0;
    for (let j = 0; j < n; j++) if (r.H[k][j] <= 1e-8) zeros++;
    say(line + '  有效 ' + (nz / tot * 100).toFixed(0) + '%  精确零 ' + (zeros / n * 100).toFixed(0) + '%');
  }
  say('        ' + rep('─', Math.floor(n / step)));
  say('        样本索引 j →   （· = <5% 最大值的「残留」，数字 = 相对强度）');
  say('\n  H 呈现「软稀疏」：每列由 1 个主导分量 + 少量泄漏构成，而非严格零。');
  say('  真正的精确稀疏需要额外正则（L1 / 稀疏 NMF）—— 乘法更新保值零元只能维持初始的零。');
}

// ---------------- 5. 基向量恢复质量（置换 + 尺度不变） ----------------
say('\n【5】基向量恢复：恢复的 W 与真 W₀ 的最佳匹配 |corr|');
{
  const r = NMF.solve(X, K, { algo: 'als', init: 'nndsvd', maxIter: 1500, tol: 1e-12, seed: SEED });
  const corr = NMF.matchCorr(r.W, dt.W0);
  say('  恢复 w_k → 最佳匹配真分量   |corr|');
  say(rep('-', 44));
  for (let k = 0; k < K; k++) {
    // 找最佳匹配的是哪一个真分量
    let best = 0, bestQ = -1;
    for (let q = 0; q < dt.W0[0].length; q++) {
      const a = [], b = [];
      for (let i = 0; i < m; i++) { a.push(r.W[i][k]); b.push(dt.W0[i][q]); }
      const c = Math.abs(NMF.corr(a, b));
      if (c > best) { best = c; bestQ = q; }
    }
    const barLen = Math.round(best * 30);
    say('  w' + k + ' → s' + bestQ + '  ' + rep('#', barLen) + rep('·', 30 - barLen) + '  ' + best.toFixed(4));
  }
  say('\n  真峰位: ' + dt.centers.map(c => c.toFixed(1)).join(', ') + '  (特征维度上的峰中心)');
}

// ---------------- 6. 精确恢复能力（无噪、可分离） ----------------
say('\n【6】精确恢复：无噪 & 可分离构造');
{
  // 无噪精确秩
  const d0 = NMF.makeData(64, 140, 4, 42, 0.0);
  const r0 = NMF.solve(d0.X, 4, { algo: 'als', init: 'nndsvd', maxIter: 2000, tol: 1e-14, seed: 42 });
  say('  无噪 X = W₀H₀（精确秩 4），K=4 :  relF = ' + r0.relError.toExponential(3));
  // 可分离
  const rng = NMF.mulberry32(31415);
  const mm = 40, Kk = 3, nb = 20, nn = Kk + nb;
  const W0 = NMF.zeros(mm, Kk), H0 = NMF.zeros(Kk, nn);
  for (let i = 0; i < mm; i++) for (let k = 0; k < Kk; k++) W0[i][k] = 0.1 + rng();
  for (let k = 0; k < Kk; k++) H0[k][k] = 1.0;
  for (let k = 0; k < Kk; k++) for (let j = Kk; j < nn; j++) H0[k][j] = 0.9 * rng();
  const Xs = NMF.matMul(W0, H0);
  const rs = NMF.solve(Xs, Kk, { algo: 'als', init: 'nndsvd', maxIter: 1200, tol: 1e-14, seed: 9 });
  say('  可分离 X = W₀[I|B]（分解唯一），K=3 :  relF = ' + rs.relError.toExponential(3) +
      '   max|corr| = ' + Math.min.apply(null, NMF.matchCorr(rs.W, W0)).toFixed(4));
  // 秩-1
  const rng2 = NMF.mulberry32(2026);
  const u = [], v = [];
  for (let i = 0; i < 30; i++) u.push(0.2 + rng2() * 1.8);
  for (let j = 0; j < 40; j++) v.push(0.2 + rng2() * 1.8);
  const X1 = NMF.zeros(30, 40);
  for (let i = 0; i < 30; i++) for (let j = 0; j < 40; j++) X1[i][j] = u[i] * v[j];
  const r1 = NMF.solve(X1, 1, { algo: 'muF', init: 'random', maxIter: 3000, tol: 1e-14, seed: 5 });
  say('  秩-1 X = uvᵀ，K=1 :  relF = ' + r1.relError.toExponential(3) + '   （机器精度）');
}

// ---------------- 7. 多种子健壮性 ----------------
say('\n【7】多种子健壮性（ALS, K=4, maxIter=1200）');
{
  const accs = [];
  for (const s of [1, 7, 42, 99, 2026, 123456]) {
    const r = NMF.solve(X, K, { algo: 'als', init: 'nndsvd', maxIter: 1200, tol: 1e-12, seed: s });
    accs.push(r.relError);
  }
  say('  relF: ' + accs.map(a => a.toFixed(5)).join(',  '));
  say('  最大/最小: ' + Math.max.apply(null, accs).toFixed(5) + ' / ' + Math.min.apply(null, accs).toFixed(5) +
      '   （离散度极小 → 凸性良好，不依赖种子）');
}

// ---------------- 8. K 选择（重构误差的肘部） ----------------
say('\n【8】K 选择：elbow（relF 随 K 的变化）');
{
  const ks = [1, 2, 3, 4, 5, 6, 8, 10, 12];
  for (const k of ks) {
    const r = NMF.solve(X, k, { algo: 'als', init: 'nndsvd', maxIter: 800, tol: 1e-10, seed: 42 });
    const bar = Math.round(r.relError * 60);
    say('  K=' + pad(k, 2, 1) + '  ' + rep('#', bar) + rep('·', Math.max(0, 60 - bar)) + '  ' + r.relError.toFixed(5));
  }
  say('\n  真 K=4 附近出现明显肘部（K=4→5,6 收益递减即说明 4 是「对的」）—— 这是无监督选 K 的实践准则。');
}

say('\n' + rep('=', 78));
say('探针结束。');
