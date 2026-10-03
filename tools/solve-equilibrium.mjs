import { writeFileSync } from 'node:fs';

// Special weakly dominates every other legal action whenever affordable.
// Both special -> subtract five from both; only one special -> immediate win.
// The unbounded game therefore reduces exactly to the 25 states below.
const N = 5, EPS = 1e-11;
const cost = [0, 0, 1, 5];
const combos = (n, k) => {
  const out = [];
  function visit(a, start) {
    if (a.length === k) return out.push(a);
    for (let i = start; i < n; i++) visit([...a, i], i + 1);
  }
  visit([], 0);
  return out;
};
function linear(a, b) {
  const n = b.length, m = a.map((r, i) => [...r, b[i]]);
  for (let j = 0; j < n; j++) {
    let pivot = j;
    for (let i = j + 1; i < n; i++) if (Math.abs(m[i][j]) > Math.abs(m[pivot][j])) pivot = i;
    if (Math.abs(m[pivot][j]) < 1e-13) return null;
    [m[j], m[pivot]] = [m[pivot], m[j]];
    const scale = m[j][j];
    for (let k = j; k <= n; k++) m[j][k] /= scale;
    for (let i = 0; i < n; i++) if (i !== j) {
      const t = m[i][j];
      for (let k = j; k <= n; k++) m[i][k] -= t * m[j][k];
    }
  }
  return m.map(r => r[n]);
}
function minimax(a) {
  const rows = a.length, cols = a[0].length;
  let best = null;
  for (let k = 1; k <= Math.min(rows, cols); k++) {
    for (const r of combos(rows, k)) for (const c of combos(cols, k)) {
      const eq = c.map(j => [...r.map(i => a[i][j]), -1]);
      eq.push([...r.map(() => 1), 0]);
      const result = linear(eq, [...c.map(() => 0), 1]);
      if (!result || result.slice(0, k).some(x => x < -EPS)) continue;
      const p = Array(rows).fill(0);
      r.forEach((i, t) => { p[i] = Math.max(0, result[t]); });
      const sum = p.reduce((s, x) => s + x, 0);
      for (let i = 0; i < rows; i++) p[i] /= sum;
      const guarantee = Math.min(...Array.from({length: cols}, (_, j) => p.reduce((s, x, i) => s + x * a[i][j], 0)));
      if (guarantee < result[k] - EPS) continue;
      if (!best || guarantee > best.value + EPS) best = {value: guarantee, p};
    }
  }
  if (!best) throw Error('No minimax vertex');
  return best;
}
function valueAt(v, a, b) {
  const tierA = Math.floor(a / 5), tierB = Math.floor(b / 5);
  if (tierA !== tierB) return Math.sign(tierA - tierB);
  return v[(a % 5) * N + b % 5];
}
function matrix(v, a, b) {
  const own = a ? [0, 1, 2] : [0, 1], opponent = b ? [0, 1, 2] : [0, 1];
  return own.map(x => opponent.map(y => {
    if (x === 2 && y === 0) return 1;
    if (y === 2 && x === 0) return -1;
    return valueAt(v, a + (x === 0 ? 1 : -cost[x]), b + (y === 0 ? 1 : -cost[y]));
  }));
}
let v = Array(N * N).fill(0), residual = Infinity, rounds = 0;
for (; rounds < 20000; rounds++) {
  const next = v.map((_, s) => minimax(matrix(v, Math.floor(s / N), s % N)).value);
  residual = Math.max(...next.map((x, s) => Math.abs(x - v[s])));
  v = next;
  if (residual < 1e-13) break;
}
if (residual > 1e-10) throw Error('Value iteration did not converge');
const strategies = v.map((value, s) => {
  if (value < -1 + 1e-9) return [1, 0, 0];
  const p = minimax(matrix(v, Math.floor(s / N), s % N)).p;
  return [0, 1, 2].map(i => Number((p[i] ?? 0).toFixed(10)));
});
// Verify both players' saddle inequalities, symmetry and the rounded table.
let exploitability = 0, symmetry = 0;
for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
  const m = matrix(v, a, b), p = strategies[a * N + b], q = strategies[b * N + a];
  const low = Math.min(...m[0].map((_, j) => m.reduce((sum, row, i) => sum + p[i] * row[j], 0)));
  const high = Math.max(...m.map(row => row.reduce((sum, x, j) => sum + q[j] * x, 0)));
  exploitability = Math.max(exploitability, high - low);
  symmetry = Math.max(symmetry, Math.abs(v[a * N + b] + v[b * N + a]));
  if (Math.abs(p.reduce((s, x) => s + x, 0) - 1) > 1e-9) throw Error('Invalid probabilities');
  if (a === 0 && p[2]) throw Error('Unaffordable attack');
}
if (exploitability > 1e-8 || symmetry > 1e-8) throw Error('Saddle verification failed');
const report = {rounds, residual, exploitability, symmetry, values: v, strategies};
writeFileSync(new URL('./equilibrium.json', import.meta.url), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
