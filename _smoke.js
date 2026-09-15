// nmf-forge 无头不变量自检 —— 原生 Node 运行（不走 vm，避开 15x 慢路径）
// 用法: node _smoke.js
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="engine">([\s\S]*?)<\/script>/);
if (!m) { console.error('FATAL: engine script not found'); process.exit(1); }
new Function(m[1])();
const NMF = globalThis.NMF;
if (!NMF) { console.error('FATAL: globalThis.NMF not exposed'); process.exit(1); }

let PASS = 0, FAIL = 0;
const fails = [];
function ok(name, cond, info) {
  if (cond) { PASS++; console.log('  \u2713 ' + name + '  [' + info + ']'); }
  else { FAIL++; fails.push(name); console.log('  \u2717 ' + name + '  [' + info + ']'); }
}
function hdr(t) { console.log('\n' + t); }
function fmt(x, d) { const a = Math.abs(x); return (a !== 0 && (a < 1e-3 || a >= 1e6)) ? x.toExponential(d || 2) : x.toFixed(d === undefined ? 4 : d); }

// ============================================================
// 数据
// ============================================================
const d = NMF.makeData(64, 140, 4, 42, 0.0);
const X = d.X;                 // 无噪、精确秩 4（用于精确恢复类检验）
const K = 4;
// 含噪、非退化问题（用于最优性 / KKT 类检验：无噪且 K=真秩时 F→0，梯度趋于 0，度量会退化）
const dn = NMF.makeData(64, 120, 3, 7, 0.25);
const Xn = dn.X, Kn = 3;

// ============================================================
hdr('1) 非负性：X ≥ 0 输入 → W, H 全程非负（三种算法）');
{
  const algos = ['muF', 'muKL', 'als'];
  let okAll = true, info = [];
  for (const a of algos) {
    const r = NMF.solve(X, K, { algo: a, init: 'random', maxIter: 200, seed: 7 });
    const mn = Math.min(NMF.minEntry(r.W), NMF.minEntry(r.H));
    if (!(mn >= 0)) okAll = false;
    info.push(a + ':' + fmt(mn, 1));
  }
  ok('X ≥ 0 → W ≥ 0 且 H ≥ 0', okAll, info.join(' '));
}

// ============================================================
hdr('2) 单调性：乘法更新的目标函数单调不增（majorize-minimize 保证）');
{
  const r = NMF.solve(X, K, { algo: 'muF', init: 'nndsvd', maxIter: 1500, tol: 1e-10, seed: 42 });
  let maxD = -Infinity;
  for (let i = 1; i < r.hist.length; i++) maxD = Math.max(maxD, r.hist[i] - r.hist[i - 1]);
  const scale = Math.abs(r.hist[0]) + 1e-300;
  console.log('  F[0]=' + fmt(r.hist[0], 3) + '  F[-1]=' + fmt(r.hist[r.hist.length - 1], 3) + '  iters=' + r.iters);
  ok('Frobenius 目标单调不增（最小化 ⇒ 查最大增量 maxΔ ≤ 1e-9·F₀）', maxD <= 1e-9 * scale, 'maxΔ=' + fmt(maxD, 3));
  ok('Frobenius 目标确实下降', r.hist[r.hist.length - 1] < r.hist[0] * 0.5,
    'F[-1]/F[0]=' + fmt(r.hist[r.hist.length - 1] / r.hist[0], 4));
}

// ============================================================
hdr('3) KL 散度变体：目标单调不增且 D ≥ 0');
{
  const r = NMF.solve(X, K, { algo: 'muKL', init: 'nndsvd', maxIter: 1000, tol: 1e-10, seed: 42 });
  let maxD = -Infinity;
  for (let i = 1; i < r.hist.length; i++) maxD = Math.max(maxD, r.hist[i] - r.hist[i - 1]);
  const scale = Math.abs(r.hist[0]) + 1e-300;
  console.log('  D[0]=' + fmt(r.hist[0], 3) + '  D[-1]=' + fmt(r.hist[r.hist.length - 1], 3) + '  iters=' + r.iters);
  ok('KL 目标单调不增（maxΔ ≤ 1e-9·D₀）', maxD <= 1e-9 * scale, 'maxΔ=' + fmt(maxD, 3));
  ok('KL 散度 D(X‖WH) ≥ 0', r.kl >= -1e-12, 'D=' + fmt(r.kl, 6));
}

// ============================================================
hdr('4) KKT 最优性：投影梯度残差 → 0（Lin 2007 非负约束的标准判据）');
{
  // 用含噪问题：无噪且 K=真秩时最优值 F=0，梯度趋于 0，任何相对度量都会退化为噪声
  const ra = NMF.solve(Xn, Kn, { algo: 'als', init: 'nndsvd', maxIter: 1500, tol: 1e-14, seed: 42 });
  console.log('  ALS: relF=' + fmt(ra.relError, 5) + '  PG=' + fmt(ra.kkt.max, 3) +
    '  w=' + fmt(ra.kkt.w, 3) + '  h=' + fmt(ra.kkt.h, 3) + '  iters=' + ra.iters);
  console.log('  ALS PG 轨迹: ' + ra.kktHist.map(v => v.toExponential(1)).join(' -> '));
  ok('ALS 投影梯度 ‖PG‖ < 1e-6（收敛到 KKT 点）', ra.kkt.max < 1e-6, fmt(ra.kkt.max, 3));
  // MU 线性收敛速率接近 1：同样迭代数下 PG 只到 1e-3 量级——关键是要收敛到「同一个」最优点
  const rm = NMF.solve(Xn, Kn, { algo: 'muF', init: 'nndsvd', maxIter: 4000, tol: 1e-12, seed: 42 });
  console.log('  MU : relF=' + fmt(rm.relError, 5) + '  PG=' + fmt(rm.kkt.max, 3) + '  iters=' + rm.iters);
  console.log('  MU  PG 轨迹: ' + rm.kktHist.map(v => v.toExponential(1)).join(' -> '));
  ok('MU 投影梯度显著下降 (< 1e-2)', rm.kkt.max < 1e-2, fmt(rm.kkt.max, 3));
  ok('MU 与 ALS 到达同一目标值（相对差 < 1e-5）',
    Math.abs(rm.frobenius - ra.frobenius) / ra.frobenius < 1e-5,
    'relΔ=' + fmt(Math.abs(rm.frobenius - ra.frobenius) / ra.frobenius, 2));
}

// ============================================================
hdr('5) 对偶可行性：非负约束下 ∇ ≥ 0 必须近似成立（归一化）');
{
  const r = NMF.solve(Xn, Kn, { algo: 'als', init: 'nndsvd', maxIter: 1500, tol: 1e-14, seed: 42 });
  ok('归一化 min ∇ ≥ -1e-6（梯度在最优点非负）', r.minGrad > -1e-6,
    'min∇=' + fmt(r.minGrad, 3));
}

// ============================================================
hdr('6) 乘法更新保值零元（W ← W∘(…) ⇒ 0 永久为 0）');
{
  const r = NMF.solve(X, K, { algo: 'muF', init: 'random', maxIter: 500, seed: 11 });
  const r2 = NMF.solve(X, K, { algo: 'muKL', init: 'random', maxIter: 500, seed: 11 });
  ok('MU-F 无零元违例', r.zeroViolations === 0, 'viol=' + r.zeroViolations);
  ok('MU-KL 无零元违例', r2.zeroViolations === 0, 'viol=' + r2.zeroViolations);
  // 极端构造：人为把 W 一行置零，迭代后仍为 0
  const Wt = NMF.clone(r.W), Ht = NMF.clone(r.H);
  for (let k = 0; k < Wt[0].length; k++) Wt[0][k] = 0;
  const st = NMF.stepMUf(X, Wt, Ht);
  let stillZero = true;
  for (let k2 = 0; k2 < st.W[0].length; k2++) if (st.W[0][k2] !== 0) stillZero = false;
  ok('人为置零的行迭代后仍恒为 0（"分量死亡"陷阱）', stillZero, 'row0 = all-zero preserved');
}

// ============================================================
hdr('7) 秩-1 精确恢复：X = uvᵀ（u,v ≥ 0）⟹ K=1 时 relF → 0');
{
  const rng = NMF.mulberry32(2026);
  const mm = 30, nn = 40;
  const u = [], v = [];
  for (let i = 0; i < mm; i++) u.push(0.2 + rng() * 1.8);
  for (let j = 0; j < nn; j++) v.push(0.2 + rng() * 1.8);
  const X1 = NMF.zeros(mm, nn);
  for (let i = 0; i < mm; i++) for (let j = 0; j < nn; j++) X1[i][j] = u[i] * v[j];
  const r = NMF.solve(X1, 1, { algo: 'muF', init: 'random', maxIter: 3000, tol: 1e-14, seed: 5 });
  ok('秩-1 矩阵 K=1 精确恢复 (relF < 1e-8)', r.relError < 1e-8, 'relF=' + fmt(r.relError, 3));
  // 尺度自由度：W·H 的乘积应重建 X
  ok('乘积 WH 重建 X（逐元素）', NMF.relError(X1, r.W, r.H) < 1e-8, 'relF=' + fmt(r.relError, 3));
}

// ============================================================
hdr('8) 可分离矩阵精确恢复（Donoho–Stodden：H₀ = [I_K | B] ⟹ 分解唯一）');
{
  const rng = NMF.mulberry32(31415);
  const mm = 40, Kk = 3, nb = 20, nn = Kk + nb;
  const W0 = NMF.zeros(mm, Kk);
  for (let i = 0; i < mm; i++) for (let k = 0; k < Kk; k++) W0[i][k] = 0.1 + rng();
  const H0 = NMF.zeros(Kk, nn);
  for (let k = 0; k < Kk; k++) H0[k][k] = 1.0;                       // 锚点列
  for (let k = 0; k < Kk; k++) for (let j = Kk; j < nn; j++) H0[k][j] = 0.9 * rng();
  const Xs = NMF.matMul(W0, H0);
  // MU 在可分离问题上线性收敛很慢（5000 迭代只到 6.9e-5）；ALS 投影梯度收敛快得多
  const r = NMF.solve(Xs, Kk, { algo: 'als', init: 'nndsvd', maxIter: 1200, tol: 1e-14, seed: 9 });
  const corr = NMF.matchCorr(r.W, W0);
  const minCorr = Math.min.apply(null, corr);
  ok('可分离 NMF 精确恢复 (relF < 1e-6)', r.relError < 1e-6, 'relF=' + fmt(r.relError, 3));
  ok('恢复基向量与真基向量 max|corr| > 0.99（置换+尺度后）', minCorr > 0.99,
    'corr=[' + corr.map(c => c.toFixed(4)).join(',') + ']');
}

// ============================================================
hdr('9) 约束对照：NMF 误差 ≥ 截断 SVD 误差（可行集更小，不可能更优）');
{
  for (const k of [1, 2, 4, 6]) {
    const r = NMF.solve(X, Math.min(k, 4) === k ? k : 4, { algo: 'muF', init: 'nndsvd', maxIter: 2000, tol: 1e-12, seed: 42 });
    const sv = NMF.svdRelError(X, k, 120);
    ok('K=' + k + '  NMF ≥ SVD', r.relError >= sv - 1e-9,
      'NMF=' + fmt(r.relError, 4) + ' SVD=' + fmt(sv, 4) + ' gap=' + fmt(r.relError - sv, 4));
  }
}

// ============================================================
hdr('10) SVD 例程自洽：显式重构误差 == svdTopK.err');
{
  const sv = NMF.svdTopK(X, 4, 120);
  const mm = X.length, nn = X[0].length;
  const R = NMF.zeros(mm, nn);
  for (let k = 0; k < 4; k++) {
    for (let i = 0; i < mm; i++) for (let j = 0; j < nn; j++) R[i][j] += sv.S[k] * sv.U[k][i] * sv.V[k][j];
  }
  let s = 0;
  for (let i = 0; i < mm; i++) for (let j = 0; j < nn; j++) { const dd = X[i][j] - R[i][j]; s += dd * dd; }
  const expErr = Math.sqrt(s);
  ok('‖X − Σσᵤᵥᵀ‖_F == svdTopK.err', Math.abs(expErr - sv.err) < 1e-9 * (NMF.norm2(X) + 1), 
    'explicit=' + fmt(expErr, 4) + ' reported=' + fmt(sv.err, 4));
  // 正交性
  const Ct = (a, b) => { let s2 = 0; for (let i = 0; i < a.length; i++) s2 += a[i] * b[i]; return s2; };
  let maxOff = 0;
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if (a !== b)
    maxOff = Math.max(maxOff, Math.abs(Ct(sv.U[a], sv.U[b])), Math.abs(Ct(sv.V[a], sv.V[b])));
  ok('奇异向量正交 (UᵀU = I, VᵀV = I)', maxOff < 1e-8, 'max|offdiag|=' + fmt(maxOff, 2));
}

// ============================================================
hdr('11) 截断 SVD 的秩-K 最优性：SVD 误差 ≤ 随机秩-K 逼近误差');
{
  const sv = NMF.svdTopK(X, 3, 120);
  const rng = NMF.mulberry32(777);
  let bestRand = Infinity;
  for (let t = 0; t < 20; t++) {
    const mm = X.length, nn = X[0].length;
    const A = NMF.zeros(mm, 3), B = NMF.zeros(3, nn);
    for (let i = 0; i < mm; i++) for (let k = 0; k < 3; k++) A[i][k] = rng();
    for (let k = 0; k < 3; k++) for (let j = 0; j < nn; j++) B[k][j] = rng();
    const RR = NMF.matMul(A, B);
    let s = 0;
    for (let i = 0; i < mm; i++) for (let j = 0; j < nn; j++) { const dd = X[i][j] - RR[i][j]; s += dd * dd; }
    bestRand = Math.min(bestRand, Math.sqrt(s));
  }
  ok('SVD 秩-3 误差 ≤ 随机秩-3 误差（Eckart-Young）', sv.err <= bestRand + 1e-9,
    'svd=' + fmt(sv.err, 3) + ' bestRandom=' + fmt(bestRand, 3));
}

// ============================================================
hdr('12) 确定性：同种子逐位一致');
{
  const o = { algo: 'muF', init: 'nndsvd', maxIter: 600, seed: 123456 };
  const A = NMF.solve(X, K, o), B = NMF.solve(X, K, o);
  let same = A.hist.length === B.hist.length;
  for (let i = 0; same && i < A.hist.length; i++) if (A.hist[i] !== B.hist[i]) same = false;
  for (let i = 0; same && i < A.W.length; i++) for (let k = 0; k < A.W[i].length; k++) if (A.W[i][k] !== B.W[i][k]) { same = false; break; }
  for (let k = 0; same && k < A.H.length; k++) for (let j = 0; j < A.H[k].length; j++) if (A.H[k][j] !== B.H[k][j]) { same = false; break; }
  ok('两次运行逐位一致', same, 'iters=' + A.iters + '/' + B.iters);
}

// ============================================================
hdr('13) 尺度不变性：W·diag(s), diag(1/s)·H ⟹ WH 不变');
{
  const r = NMF.solve(X, K, { algo: 'muF', init: 'nndsvd', maxIter: 600, seed: 42 });
  const W = NMF.clone(r.W), H = NMF.clone(r.H);
  const ss = [0.31, 2.7, 11.0, 0.05];
  for (let k = 0; k < W[0].length; k++) for (let i = 0; i < W.length; i++) W[i][k] *= ss[k];
  for (let k = 0; k < H.length; k++) for (let j = 0; j < H[k].length; j++) H[k][j] /= ss[k];
  const A0 = NMF.matMul(r.W, r.H), A1 = NMF.matMul(W, H);
  let mx = 0;
  for (let i = 0; i < A0.length; i++) for (let j = 0; j < A0[i].length; j++) mx = Math.max(mx, Math.abs(A0[i][j] - A1[i][j]));
  ok('尺度变换后 WH 逐元素不变', mx < 1e-9, 'maxΔ=' + fmt(mx, 2));
}

// ============================================================
hdr('14) 列置换等变：置换 X 的列 + 同步置换 H₀ 的列 ⟹ W 不变、H 同步置换');
{
  const mm = X.length, nn = X[0].length;
  const rng = NMF.mulberry32(2468);
  const W0 = NMF.zeros(mm, K), H0 = NMF.zeros(K, nn);
  for (let i = 0; i < mm; i++) for (let k = 0; k < K; k++) W0[i][k] = 0.1 + rng();
  for (let k = 0; k < K; k++) for (let j = 0; j < nn; j++) H0[k][j] = 0.1 + rng();
  // 置换 pi：逆序
  const pi = []; for (let j = 0; j < nn; j++) pi.push(nn - 1 - j);
  const Xp = NMF.zeros(mm, nn), H0p = NMF.zeros(K, nn);
  for (let j = 0; j < nn; j++) { for (let i = 0; i < mm; i++) Xp[i][j] = X[i][pi[j]]; for (let k = 0; k < K; k++) H0p[k][j] = H0[k][pi[j]]; }
  let Wa = NMF.clone(W0), Ha = NMF.clone(H0);
  let Wb = NMF.clone(W0), Hb = NMF.clone(H0p);
  for (let t = 0; t < 300; t++) { const ra = NMF.stepMUf(X, Wa, Ha); Wa = ra.W; Ha = ra.H;
    const rb = NMF.stepMUf(Xp, Wb, Hb); Wb = rb.W; Hb = rb.H; }
  let dW = 0;
  for (let i = 0; i < mm; i++) for (let k = 0; k < K; k++) dW = Math.max(dW, Math.abs(Wa[i][k] - Wb[i][k]));
  let dH = 0;
  for (let k = 0; k < K; k++) for (let j = 0; j < nn; j++) dH = Math.max(dH, Math.abs(Ha[k][j] - Hb[k][pi[j]]));
  ok('W 置换不变 (Δ < 1e-12)', dW < 1e-12, 'maxΔW=' + fmt(dW, 2));
  ok('H 同步置换 (Δ < 1e-12)', dH < 1e-12, 'maxΔH=' + fmt(dH, 2));
}

// ============================================================
hdr('15) 独立数值路径对照：ALS（投影梯度）与 MU 收敛到相近误差');
{
  const rm = NMF.solve(Xn, Kn, { algo: 'muF', init: 'nndsvd', maxIter: 4000, tol: 1e-12, seed: 42 });
  const ra = NMF.solve(Xn, Kn, { algo: 'als', init: 'nndsvd', maxIter: 1200, tol: 1e-12, seed: 42 });
  console.log('  MU relF=' + fmt(rm.relError, 5) + '   ALS relF=' + fmt(ra.relError, 5) +
    '   Δ=' + fmt(Math.abs(rm.relError - ra.relError), 5));
  ok('两种独立算法误差接近 (|Δ| < 0.02)', Math.abs(rm.relError - ra.relError) < 0.02,
    'Δ=' + fmt(Math.abs(rm.relError - ra.relError), 5));
  ok('ALS 解也满足 KKT（投影梯度）', ra.kkt.max < 1e-4, 'PG=' + fmt(ra.kkt.max, 3));
  ok('ALS 解非负', Math.min(NMF.minEntry(ra.W), NMF.minEntry(ra.H)) >= 0,
    'min=' + fmt(Math.min(NMF.minEntry(ra.W), NMF.minEntry(ra.H)), 1));
}

// ============================================================
hdr('16) 非负最小二乘子问题：KKT（Z ≥ 0, ∇ ≥ 0, Z∘∇ = 0）');
{
  const rng = NMF.mulberry32(1357);
  const p = 25, q = 6, r = 12;
  const A = NMF.zeros(p, q), B = NMF.zeros(p, r);
  for (let i = 0; i < p; i++) for (let k = 0; k < q; k++) A[i][k] = rng() * 2 - 0.5;   // 含负 → 真解含有效零
  for (let i = 0; i < p; i++) for (let j = 0; j < r; j++) B[i][j] = rng() * 2;
  const Z = NMF.nnlsPG(A, B, 3000, null);
  // ∇ = Aᵀ(AZ − B)
  const AZ = NMF.matMul(A, Z);
  const R = NMF.zeros(p, r);
  for (let i = 0; i < p; i++) for (let j = 0; j < r; j++) R[i][j] = AZ[i][j] - B[i][j];
  const At = NMF.transpose(A);
  const G = NMF.matMul(At, R);
  let minZ = Infinity, minG = Infinity, maxComp = 0, maxG = 0;
  for (let a = 0; a < q; a++) for (let b = 0; b < r; b++) {
    minZ = Math.min(minZ, Z[a][b]);
    minG = Math.min(minG, G[a][b]);
    maxComp = Math.max(maxComp, Math.abs(Z[a][b] * G[a][b]));
    maxG = Math.max(maxG, Math.abs(G[a][b]));
  }
  ok('NNLS 解非负 (Z ≥ 0)', minZ >= 0, 'minZ=' + fmt(minZ, 2));
  ok('NNLS 对偶可行 (∇ ≥ -tol)', minG > -1e-6, 'min∇=' + fmt(minG, 2));
  ok('NNLS 互补松弛 (max|Z∘∇|/max|∇| < 1e-6)', maxComp / (maxG + 1e-300) < 1e-6,
    'rel=' + fmt(maxComp / (maxG + 1e-300), 2));
  // 与「全量最小二乘 + 裁剪」对照：NNLS 应不劣
  const A2 = NMF.matMul(At, A), b2 = NMF.matMul(At, B);
  // 用高斯消元解 (AᵀA)Z = AᵀB（可能含负解，裁剪后作为对照）
  function solveSPD(M, rhs) {
    const n = M.length, nrhs = rhs[0].length;
    const Aug = [];
    for (let i = 0; i < n; i++) { const row = new Float64Array(n + nrhs); for (let j = 0; j < n; j++) row[j] = M[i][j]; for (let j = 0; j < nrhs; j++) row[n + j] = rhs[i][j]; Aug.push(row); }
    for (let c = 0; c < n; c++) {
      let piv = c; for (let r2 = c + 1; r2 < n; r2++) if (Math.abs(Aug[r2][c]) > Math.abs(Aug[piv][c])) piv = r2;
      const t = Aug[c]; Aug[c] = Aug[piv]; Aug[piv] = t;
      const d = Aug[c][c] || 1e-12;
      for (let r3 = 0; r3 < n; r3++) { if (r3 === c) continue; const f = Aug[r3][c] / d; if (f === 0) continue; for (let cc = c; cc < n + nrhs; cc++) Aug[r3][cc] -= f * Aug[c][cc]; }
    }
    const Zx = NMF.zeros(n, nrhs);
    for (let i = 0; i < n; i++) for (let j = 0; j < nrhs; j++) Zx[i][j] = Aug[i][n + j] / (Aug[i][i] || 1e-12);
    return Zx;
  }
  const Zunc = solveSPD(A2, b2);
  const Zclip = NMF.zeros(q, r);
  for (let a = 0; a < q; a++) for (let b = 0; b < r; b++) Zclip[a][b] = Math.max(0, Zunc[a][b]);
  function objZ(Zz) { const AZz = NMF.matMul(A, Zz); let s = 0; for (let i = 0; i < p; i++) for (let j = 0; j < r; j++) { const dd = AZz[i][j] - B[i][j]; s += dd * dd; } return s; }
  const oNN = objZ(Z), oClip = objZ(Zclip);
  ok('NNLS 目标 ≤ 裁剪最小二乘目标', oNN <= oClip + 1e-9,
    'NNLS=' + fmt(oNN, 3) + ' clipLS=' + fmt(oClip, 3));
}

// ============================================================
console.log('\n' + '='.repeat(64));
console.log('PASS ' + PASS + ' / ' + (PASS + FAIL) + (FAIL === 0 ? '   ALL GREEN \u2713' : '   FAILURES: ' + fails.join(' | ')));
console.log('='.repeat(64));
process.exit(FAIL === 0 ? 0 : 1);
