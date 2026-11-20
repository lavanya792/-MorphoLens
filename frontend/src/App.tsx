import { Fragment, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip,
  BarChart, Bar, PieChart, Pie, Cell, CartesianGrid,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ReferenceLine
} from "recharts";

/* ───────────────────────── API (unchanged) ───────────────────────── */
const API = "https://lavanya213-morpholens-backend.hf.space";

type Page = "overview" | "analysis" | "anomalies";
type Tab = "morpho" | "dist" | "corr" | "spatial" | "table";
type Analysis = { analysis_id: number; image_name: string; width: number; height: number; total_nuclei: number; status: string; created_at: string };
type Nucleus = {
  nucleus_id: number; area: number; perimeter: number; circularity: number; eccentricity: number;
  mean_intensity: number; glcm_contrast: number; pc1: number | null; pc2: number | null;
  umap1: number | null; umap2: number | null; kmeans_cluster: number | null; dbscan_cluster: number | null;
  anomaly_status: string | null; anomaly_score: number | null;
  [key: string]: any;
};

/* ───────────────────────── Design tokens for charts ───────────────────────── */
const NORMAL = "#8FA4CC";
const ANOM = "#FF4D6D";
const INK2 = "#A9A5C0";
const LINE = "#2C2840";
const GREY = "#7C7899";
// Colour-blind-safe categorical set
const palette = ["#5B9BE6", "#3FBFA8", "#EBAE4B", "#B18AE0", "#7CC4EE", "#A5B85A"];
const ax = { tick: { fontSize: 11, fill: INK2 }, stroke: LINE, tickLine: false as const };
const tipStyle = { background: "#2A2640", border: "1px solid #3D3860", borderRadius: 8, color: "#fff", fontSize: 12, padding: "8px 10px" };

const FEATURES = [
  { key: "area", label: "Area", short: "Area" },
  { key: "perimeter", label: "Perimeter", short: "Perim." },
  { key: "circularity", label: "Circularity", short: "Circ." },
  { key: "eccentricity", label: "Eccentricity", short: "Ecc." },
  { key: "aspect_ratio", label: "Aspect ratio", short: "Aspect" },
  { key: "solidity", label: "Solidity", short: "Solidity" },
  { key: "extent", label: "Extent", short: "Extent" },
  { key: "major_axis_length", label: "Major axis", short: "Major" },
  { key: "minor_axis_length", label: "Minor axis", short: "Minor" },
  { key: "mean_intensity", label: "Mean intensity", short: "Mean I" },
  { key: "std_intensity", label: "Intensity SD", short: "SD I" },
  { key: "min_intensity", label: "Min intensity", short: "Min I" },
  { key: "max_intensity", label: "Max intensity", short: "Max I" },
  { key: "glcm_contrast", label: "GLCM contrast", short: "Contrast" },
  { key: "glcm_correlation", label: "GLCM correlation", short: "Texture r" },
  { key: "glcm_energy", label: "GLCM energy", short: "Energy" },
  { key: "glcm_homogeneity", label: "GLCM homogeneity", short: "Homog." },
  { key: "glcm_dissimilarity", label: "GLCM dissimilarity", short: "Dissim." }
];
const RADAR_KEYS = ["area", "circularity", "eccentricity", "solidity", "mean_intensity", "std_intensity", "glcm_contrast", "glcm_homogeneity"];

/* ───────────────────────── Numeric helpers ───────────────────────── */
const isNum = (v: any): v is number => typeof v === "number" && Number.isFinite(v);
const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
const sd = (a: number[]) => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
};
const median = (a: number[]) => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const fmt = (v: any, d = 3) => (!isNum(v) ? "—" : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(d));
function pearson(xs: any[], ys: any[]) {
  const x: number[] = [], y: number[] = [];
  xs.forEach((v, i) => { if (isNum(v) && isNum(ys[i])) { x.push(v); y.push(ys[i]); } });
  if (x.length < 3) return 0;
  const mx = mean(x), my = mean(y);
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < x.length; i++) { num += (x[i] - mx) * (y[i] - my); dx += (x[i] - mx) ** 2; dy += (y[i] - my) ** 2; }
  return dx && dy ? num / Math.sqrt(dx * dy) : 0;
}
function makeBins(values: number[], n = 22) {
  let lo = Infinity, hi = -Infinity;
  values.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; });
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (hi === lo) hi = lo + 1;
  const step = (hi - lo) / n;
  return {
    n, step,
    index: (v: number) => Math.max(0, Math.min(n - 1, Math.floor((v - lo) / step))),
    center: (i: number) => lo + step * (i + 0.5)
  };
}
const digitsFor = (step: number) => (step >= 10 ? 0 : step >= 1 ? 1 : step >= 0.1 ? 2 : step >= 0.01 ? 3 : 4);
function histRows(all: Nucleus[], key: string) {
  const vals = all.map(n => n[key]).filter(isNum) as number[];
  const b = makeBins(vals);
  const rows = Array.from({ length: b.n }, (_, i) => ({ x: b.center(i), normal: 0, anomaly: 0 }));
  all.forEach(n => {
    const v = n[key];
    if (!isNum(v)) return;
    const r = rows[b.index(v)];
    if (n.anomaly_status === "Anomaly") r.anomaly++; else r.normal++;
  });
  return { rows, b, d: digitsFor(b.step), mean: mean(vals), median: median(vals) };
}
const shortName = (s: string) => {
  if (!s || s.length <= 24) return s || "—";
  const dot = s.lastIndexOf(".");
  const ext = dot > 0 && s.length - dot <= 6 ? s.slice(dot) : "";
  const base = ext ? s.slice(0, dot) : s;
  return `${base.slice(0, 8)}…${base.slice(-6)}${ext}`;
};
const clusterName = (c: number | null | undefined) => (c == null ? "Not run" : `Cluster ${c}`);

/* ───────────────────────── App ───────────────────────── */
export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [tab, setTab] = useState<Tab>("morpho");
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [details, setDetails] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [stage, setStage] = useState(0);

  // view state
  const [proj, setProj] = useState<"pca" | "umap" | "feat">("pca");
  const [colorBy, setColorBy] = useState<"cluster" | "dbscan" | "status">("cluster");
  const [fx, setFx] = useState("area");
  const [fy, setFy] = useState("circularity");
  const [rings, setRings] = useState(true);
  const [histFeat, setHistFeat] = useState("area");
  const [pick, setPick] = useState<number | null>(null);
  const [swapXY, setSwapXY] = useState(false);
  const [showNormal, setShowNormal] = useState(true);
  const [showImage, setShowImage] = useState(true);

  async function refresh(preferred?: number) {
    const [a] = await Promise.all([axios.get(`${API}/api/dashboard/analyses`)]);
    const list: Analysis[] = a.data;
    setAnalyses(list);
    const id = preferred ?? selected ?? list[0]?.analysis_id ?? null;
    if (id !== null) await loadAnalysis(id, false);
    else { setSelected(null); setDetails(null); setStats(null); }
  }
  async function loadAnalysis(id: number, changePage = true) {
    const [d, s] = await Promise.all([
      axios.get(`${API}/api/dashboard/analysis/${id}`),
      axios.get(`${API}/api/dashboard/stats`, { params: { analysis_id: id } })
    ]);
    setSelected(id); setDetails(d.data); setStats(s.data); setPick(null);
    if (changePage) setPage("analysis");
  }
  useEffect(() => { refresh().catch(e => setMessage(errorText(e))); }, []);
  function errorText(e: any) { return e.response?.data?.detail || e.message || "Request failed"; }
  async function upload() {
    if (!file) return;
    setBusy(true); setStage(1); setMessage("Uploading image and running segmentation…");
    try {
      const form = new FormData(); form.append("file", file);
      const r = await axios.post(`${API}/api/upload/analyze`, form, { headers: { "Content-Type": "multipart/form-data" } });
      const id: number = r.data.analysis_id;
      setStage(2); setMessage(`Segmentation complete: ${r.data.total_nuclei} nuclei. Running analytics…`);
      await axios.post(`${API}/api/analytics/run/${id}`);
      await refresh(id); setPage("analysis");
      setStage(3); setMessage(`Analysis #${id} completed.`);
    } catch (e: any) { setStage(0); setMessage(errorText(e)); }
    finally { setBusy(false); }
  }
  async function removeAnalysis() {
    if (!selected || !window.confirm(`Permanently delete analysis #${selected}, its measurements and stored images?`)) return;
    try {
      await axios.delete(`${API}/api/dashboard/analysis/${selected}`);
      setMessage(`Analysis #${selected} deleted.`); setPage("overview"); await refresh();
    } catch (e: any) { setMessage(errorText(e)); }
  }

  /* ─── derived data ─── */
  const nuclei: Nucleus[] = useMemo(() => details?.nuclei || [], [details]);
  const metrics: any[] = details?.metrics || [];
  const anomalies = useMemo(() => nuclei.filter(n => n.anomaly_status === "Anomaly"), [nuclei]);
  const normal = useMemo(() => nuclei.filter(n => n.anomaly_status === "Normal"), [nuclei]);
  const current = analyses.find(a => a.analysis_id === selected);
  const rate = nuclei.length ? (anomalies.length / nuclei.length) * 100 : 0;

  const feats = useMemo(() => FEATURES.filter(f => {
    const v = nuclei.map(n => n[f.key]).filter(isNum) as number[];
    return v.length > 1 && sd(v) > 0;
  }), [nuclei]);
  const fstat = useMemo(() => {
    const o: Record<string, { m: number; s: number }> = {};
    feats.forEach(f => { const v = nuclei.map(n => n[f.key]).filter(isNum) as number[]; o[f.key] = { m: mean(v), s: sd(v) }; });
    return o;
  }, [feats, nuclei]);
  const z = (n: Nucleus, k: string) => (isNum(n[k]) && fstat[k] ? (n[k] - fstat[k].m) / fstat[k].s : 0);
  const validKey = (k: string) => (feats.some(f => f.key === k) ? k : feats[0]?.key || k);
  const labelOf = (k: string) => FEATURES.find(f => f.key === k)?.label || k;

  const boundary = useMemo(() => {
    const a = anomalies.map(n => n.anomaly_score).filter(isNum) as number[];
    const b = normal.map(n => n.anomaly_score).filter(isNum) as number[];
    if (!a.length || !b.length) return NaN;
    return (Math.max(...a) + Math.min(...b)) / 2;
  }, [anomalies, normal]);
  const ranked = useMemo(() => [...anomalies].sort((a, b) => (a.anomaly_score ?? 0) - (b.anomaly_score ?? 0)), [anomalies]);
  const focus = ranked.find(n => n.nucleus_id === pick) || ranked[0] || null;

  /* projection scatter */
  const axisX = proj === "pca" ? "PC1" : proj === "umap" ? "UMAP 1" : labelOf(validKey(fx));
  const axisY = proj === "pca" ? "PC2" : proj === "umap" ? "UMAP 2" : labelOf(validKey(fy));
  const points = useMemo(() => nuclei.map(n => {
    const x = proj === "pca" ? n.pc1 : proj === "umap" ? n.umap1 : n[validKey(fx)];
    const y = proj === "pca" ? n.pc2 : proj === "umap" ? n.umap2 : n[validKey(fy)];
    return { x, y, id: n.nucleus_id, status: n.anomaly_status, cluster: n.kmeans_cluster, db: n.dbscan_cluster, score: n.anomaly_score, xl: axisX, yl: axisY };
  }).filter(p => isNum(p.x) && isNum(p.y)), [nuclei, proj, fx, fy, feats]);
  const groups = useMemo(() => {
    const map = new Map<string, { name: string; color: string; pts: any[] }>();
    points.forEach(p => {
      let key: string, name: string, color: string;
      if (colorBy === "cluster") { key = String(p.cluster ?? "na"); name = clusterName(p.cluster); color = p.cluster == null ? GREY : palette[Math.abs(p.cluster) % palette.length]; }
      else if (colorBy === "dbscan") { key = String(p.db ?? "na"); name = p.db === -1 ? "Noise" : p.db == null ? "Not run" : `Group ${p.db}`; color = p.db == null || p.db === -1 ? GREY : palette[Math.abs(p.db) % palette.length]; }
      else { key = p.status || "na"; name = p.status === "Anomaly" ? "Potential anomaly" : p.status || "Not run"; color = p.status === "Anomaly" ? ANOM : NORMAL; }
      if (!map.has(key)) map.set(key, { name, color, pts: [] });
      map.get(key)!.pts.push(p);
    });
    return [...map.values()].sort((a, b) => a.name.localeCompare(b.name));
  }, [points, colorBy]);
  const ringPts = useMemo(() => points.filter(p => p.status === "Anomaly"), [points]);

  /* cluster composition */
  const composition = useMemo(() => {
    const m: Record<string, { cluster: string; normal: number; anomaly: number }> = {};
    nuclei.forEach(n => {
      const k = String(n.kmeans_cluster ?? "na");
      m[k] = m[k] || { cluster: n.kmeans_cluster == null ? "Not run" : `C${n.kmeans_cluster}`, normal: 0, anomaly: 0 };
      if (n.anomaly_status === "Anomaly") m[k].anomaly++; else m[k].normal++;
    });
    return Object.entries(m).sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true })).map(([, v]) => v);
  }, [nuclei]);

  /* radar: mean z-score per cluster across core features */
  const kIds = useMemo(() => [...new Set(nuclei.map(n => n.kmeans_cluster).filter(c => c != null))].sort((a: any, b: any) => a - b) as number[], [nuclei]);
  const radarFeats = feats.filter(f => RADAR_KEYS.includes(f.key));
  const radarData = useMemo(() => radarFeats.map(f => {
    const clamp = (v: number) => Math.max(-2, Math.min(2, v));
    const row: any = { feature: f.short };
    kIds.forEach(k => { row[`c${k}`] = clamp(mean(nuclei.filter(n => n.kmeans_cluster === k).map(n => z(n, f.key)))); });
    row.anomaly = anomalies.length ? clamp(mean(anomalies.map(n => z(n, f.key)))) : 0;
    return row;
  }), [radarFeats.length, kIds, nuclei, anomalies, fstat]);

  /* DBSCAN + donut */
  const dbData = useMemo(() => {
    const c: Record<string, number> = {};
    nuclei.forEach(n => { const k = String(n.dbscan_cluster ?? "na"); c[k] = (c[k] || 0) + 1; });
    return Object.entries(c).sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([k, count]) => ({ group: k === "-1" ? "Noise" : k === "na" ? "Not run" : `Group ${k}`, count, noise: k === "-1", idx: Number(k) }));
  }, [nuclei]);
  const donut = [{ name: "Normal", value: normal.length }, { name: "Potential anomaly", value: anomalies.length }];

  /* anomaly drivers + focus z-scores */
  const drivers = useMemo(() => feats.map(f => ({
    name: f.label,
    flagged: +mean(anomalies.map(n => Math.abs(z(n, f.key)))).toFixed(2),
    normal: +mean(normal.map(n => Math.abs(z(n, f.key)))).toFixed(2)
  })).sort((a, b) => b.flagged - a.flagged).slice(0, 8), [feats, anomalies, normal, fstat]);
  const focusZ = useMemo(() => !focus ? [] : feats.map(f => ({ name: f.label, z: +z(focus, f.key).toFixed(2) }))
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z)).slice(0, 8), [focus, feats, fstat]);

  /* correlation */
  const pairs = useMemo(() => {
    const out: { a: string; b: string; r: number }[] = [];
    for (let i = 0; i < feats.length; i++) for (let j = i + 1; j < feats.length; j++)
      out.push({ a: feats[i].label, b: feats[j].label, r: pearson(nuclei.map(n => n[feats[i].key]), nuclei.map(n => n[feats[j].key])) });
    return out.sort((p, q) => Math.abs(q.r) - Math.abs(p.r)).slice(0, 6);
  }, [feats, nuclei]);

  async function exportCsv(rows: Nucleus[] = nuclei) {
    if (!rows.length) return;
    const cols = Object.keys(rows[0] as any);
    const csv = [cols.join(","), ...rows.map((r: any) => cols.map(c => JSON.stringify(r[c] ?? "")).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = `morpholens_${page}_${selected}.csv`; a.click(); URL.revokeObjectURL(url);
  }

  const titles: Record<Page, [string, string]> = {
    overview: ["Workspace", "Upload specimens, review runs and see how the selected image scores."],
    analysis: ["Image analysis", "Segmentation output, morphology space, distributions and spatial layout."],
    anomalies: ["Anomaly review", "Nuclei whose morphology departs from the rest of this image."]
  };
  const mapProps = { nuclei, width: current?.width, height: current?.height, src: `/api/dashboard/analysis/${selected}/image`, swap: swapXY, showNormal, showImage, selectedId: focus?.nucleus_id ?? null };

  return <div className="shell">
    <header className="top">
      <div className="brand">
        <svg viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="17" fill="none" stroke="currentColor" strokeWidth="2" /><ellipse cx="15" cy="17" rx="6" ry="5" fill="currentColor" opacity=".9" /><ellipse cx="25.5" cy="24" rx="5" ry="4.2" fill="#D6567F" /><circle cx="26" cy="12.5" r="2" fill="currentColor" opacity=".5" /></svg>
        <div><b>MorphoLens</b><small>Nuclear morphology analytics</small></div>
      </div>
      <nav aria-label="Primary">
        <button className={page === "overview" ? "tabbtn on" : "tabbtn"} onClick={() => setPage("overview")}>Workspace</button>
        <button className={page === "analysis" ? "tabbtn on" : "tabbtn"} onClick={() => setPage("analysis")} disabled={!details}>Image analysis</button>
        <button className={page === "anomalies" ? "tabbtn on" : "tabbtn"} onClick={() => setPage("anomalies")} disabled={!details}>Anomaly review{details && <i className="count">{anomalies.length}</i>}</button>
      </nav>
      <div className="topRight">
        {analyses.length > 0 && <label className="runPick"><span>Run</span>
          <select value={selected ?? ""} onChange={e => loadAnalysis(Number(e.target.value), false).catch(err => setMessage(errorText(err)))}>
            {analyses.map(a => <option key={a.analysis_id} value={a.analysis_id}>#{a.analysis_id} · {shortName(a.image_name)}</option>)}
          </select></label>}
        <span className="modelTag">U-Net++ · ResNet34</span>
      </div>
    </header>

    <main className="page">
      <div className="pagehead"><h1>{titles[page][0]}</h1><p>{titles[page][1]}</p></div>

      {/* ═════════ OVERVIEW ═════════ */}
      {page === "overview" && <>
        <section className="card uploader">
          <div className="uploadText">
            <h2>New analysis</h2>
            <p>Upload a microscopy image (PNG, JPG or TIFF). Nuclei are segmented, measured, clustered and screened for statistical outliers.</p>
          </div>
          <div className="uploadControls">
            <label className={file ? "drop has" : "drop"}>
              <input type="file" accept=".png,.jpg,.jpeg,.tif,.tiff" onChange={e => setFile(e.target.files?.[0] || null)} />
              <span>{file ? file.name : "Choose an image"}</span>
            </label>
            <button className="btn primary" disabled={!file || busy} onClick={upload}>{busy ? "Processing…" : "Upload and analyze"}</button>
          </div>
          {stage > 0 && <ol className="stepper" aria-label="Run progress">
            {["Segmentation", "Features and analytics", "Ready"].map((l, i) => <li key={l} className={stage > i + 1 || stage === 3 ? "done" : stage === i + 1 ? "now" : ""}><i>{stage > i + 1 || stage === 3 ? "✓" : i + 1}</i>{l}</li>)}
          </ol>}
          {busy && <div className="progress" role="progressbar" aria-label="Processing"><i /></div>}
          {message && <div className="notice" aria-live="polite">{message}</div>}
        </section>

        {details ? <>
          <div className="sectionLine"><h2>Selected run · #{selected}</h2><span title={current?.image_name}>{shortName(current?.image_name || "")} · values are scoped to this image only</span></div>
          <section className="kpis">
            <Kpi label="Nuclei" value={stats?.total_nuclei} />
            <Kpi label="Normal" value={stats?.normal_nuclei} />
            <Kpi label="Potential anomalies" value={stats?.anomalies} tone="alert" />
            <Kpi label="Anomaly rate" value={`${stats?.anomaly_percentage ?? 0}%`} tone="alert" gauge={stats?.anomaly_percentage ?? 0} />
            <Kpi label="Median area" value={fmt(median(nuclei.map(n => n.area).filter(isNum) as number[]), 1)} unit="px²" />
          </section>
        </> : <p className="empty">No analyses yet. Upload an image to begin.</p>}

        {details && nuclei.length > 0 && <div className="grid split fieldRow">
          <ChartCard title="Specimen field" sub="Nucleus centroids on the original image. The ringed nucleus has the lowest score."
            legend={[{ label: "Normal", color: NORMAL }, { label: "Potential anomaly", color: ANOM, ring: true }]}>
            <SpatialMap {...mapProps} onPick={(id: number) => { setPick(id); setPage("anomalies"); }} />
          </ChartCard>
          <ChartCard title="Outlier score distribution" sub="Lower Isolation Forest scores are more anomalous. The dashed line is the decision boundary." legend={[{ label: "Normal", color: NORMAL }, { label: "Potential anomaly", color: ANOM }]}>
            <Hist nuclei={nuclei} field="anomaly_score" xLabel="Isolation Forest score" markers={isNum(boundary) ? [{ value: boundary, label: "boundary", color: "#ECEAF5" }] : []} height={300} />
          </ChartCard>
        </div>}

        <div className="grid split">
          <section className="card">
            <div className="cardhead"><div><h2>Run history</h2><p>Select a run to load it across every page.</p></div></div>
            <div className="tableWrap short"><table><thead><tr><th>Run</th><th>Image</th><th>Nuclei</th><th>Status</th><th>Created</th><th /></tr></thead>
              <tbody>{analyses.map(a => <tr key={a.analysis_id} className={selected === a.analysis_id ? "sel" : ""} onClick={() => loadAnalysis(a.analysis_id, false)}>
                <td>#{a.analysis_id}</td><td className="wrap" title={a.image_name}>{shortName(a.image_name)}</td><td>{a.total_nuclei}</td><td><span className="pill">{a.status}</span></td>
                <td>{a.created_at ? String(a.created_at).slice(0, 16).replace("T", " ") : "—"}</td>
                <td><button className="btn ghost sm" onClick={e => { e.stopPropagation(); loadAnalysis(a.analysis_id, true); }}>Open</button></td></tr>)}</tbody></table></div>
          </section>
          <section className="card">
            <div className="cardhead"><div><h2>Analysis pipeline</h2><p>Each run passes through these stages in order.</p></div></div>
            <ol className="pipeline">{["Image upload", "U-Net++ segmentation", "Watershed instance separation", "Morphology and texture features", "PCA, UMAP and clustering", "Isolation Forest outlier screen"].map(x => <li key={x}>{x}</li>)}</ol>
            <p className="muted">Anomaly labels are algorithmic outlier flags. They are not disease classifications.</p>
          </section>
        </div>
      </>}

      {/* ═════════ ANALYSIS ═════════ */}
      {page === "analysis" && <>{!details ? <p className="empty">Select an analysis first.</p> : <>
        <section className="runbar">
          <div><h2>Analysis #{selected}</h2><p className="filename" title={current?.image_name}>{shortName(current?.image_name || "")} · {nuclei.length} nuclei · {current?.width} × {current?.height} px</p></div>
          <div className="actions"><button className="btn ghost" onClick={() => exportCsv()}>Export feature CSV</button><button className="btn danger" onClick={removeAnalysis}>Delete analysis</button></div>
        </section>

        <section className="scope">
          <ImagePanel title="Original" src={`/api/dashboard/analysis/${selected}/image`} />
          <ImagePanel title="Binary mask" src={`/api/dashboard/analysis/${selected}/mask`} />
          <ImagePanel title="Instance overlay" src={`/api/dashboard/analysis/${selected}/overlay`} />
        </section>

        <section className="kpis">
          <Kpi label="Nuclei detected" value={nuclei.length} />
          <Kpi label="Normal" value={normal.length} />
          <Kpi label="Potential anomalies" value={anomalies.length} tone="alert" />
          <Kpi label="Anomaly rate" value={`${rate.toFixed(2)}%`} tone="alert" gauge={rate} />
          <Kpi label="Feature dimensions" value={feats.length} sub="after variance filter" />
        </section>

        <div className="seg wide" role="tablist" aria-label="Analysis views">
          {([["morpho", "Morphology space"], ["dist", "Distributions"], ["corr", "Correlation"], ["spatial", "Spatial map"], ["table", "Measurements"]] as [Tab, string][]).map(([k, l]) =>
            <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>)}
        </div>

        {tab === "morpho" && <>
          <ChartCard title="Morphology projection" sub={`${nuclei.length} nuclei embedded by ${proj === "pca" ? "principal components" : proj === "umap" ? "UMAP" : "two raw features"}. Rings mark potential anomalies.`}
            actions={<div className="controls">
              <Field label="Axes"><Seg value={proj} onChange={setProj} options={[["pca", "PCA"], ["umap", "UMAP"], ["feat", "Features"]]} /></Field>
              {proj === "feat" && <><Field label="X"><select value={validKey(fx)} onChange={e => setFx(e.target.value)}>{feats.map(f => <option key={f.key} value={f.key}>{f.label}</option>)}</select></Field>
                <Field label="Y"><select value={validKey(fy)} onChange={e => setFy(e.target.value)}>{feats.map(f => <option key={f.key} value={f.key}>{f.label}</option>)}</select></Field></>}
              <Field label="Color by"><Seg value={colorBy} onChange={setColorBy} options={[["cluster", "K-Means"], ["dbscan", "DBSCAN"], ["status", "Status"]]} /></Field>
              <label className="check"><input type="checkbox" checked={rings} onChange={e => setRings(e.target.checked)} />Highlight anomalies</label>
            </div>}
            legend={[...groups.map(g => ({ label: g.name, color: g.color })), ...(rings ? [{ label: "Potential anomaly", color: ANOM, ring: true }] : [])]}>
            {points.length === 0 ? <p className="empty">This projection has no coordinates for the selected run. Run analytics again to compute it.</p> :
              <div className="chart tall"><ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 12, right: 18, left: 4, bottom: 22 }}>
                  <CartesianGrid stroke={LINE} strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" name={axisX} {...ax} domain={["auto", "auto"]} tickFormatter={(v: any) => fmt(v, 1)} label={{ value: axisX, position: "insideBottom", offset: -12, fontSize: 12, fill: INK2 }} />
                  <YAxis type="number" dataKey="y" name={axisY} {...ax} domain={["auto", "auto"]} tickFormatter={(v: any) => fmt(v, 1)} width={46} label={{ value: axisY, angle: -90, position: "insideLeft", fontSize: 12, fill: INK2 }} />
                  <ZAxis range={[40, 40]} />
                  <Tooltip content={<PointTip />} cursor={{ strokeDasharray: "3 3", stroke: GREY }} />
                  {groups.map(g => <Scatter key={g.name} name={g.name} data={g.pts} shape={dot(g.color)} isAnimationActive={false} />)}
                  {rings && <Scatter name="Potential anomaly" data={ringPts} shape={ring} isAnimationActive={false} />}
                </ScatterChart></ResponsiveContainer></div>}
          </ChartCard>

          <div className="grid g2">
            <ChartCard title="Cluster composition" sub="Normal and flagged nuclei within each K-Means cluster." legend={[{ label: "Normal", color: NORMAL }, { label: "Potential anomaly", color: ANOM }]}>
              <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={composition} margin={{ top: 10, right: 10, left: 0, bottom: 4 }}>
                <CartesianGrid stroke={LINE} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="cluster" {...ax} /><YAxis allowDecimals={false} {...ax} width={34} />
                <Tooltip contentStyle={tipStyle} cursor={{ fill: "rgba(156,144,240,.12)" }} />
                <Bar dataKey="normal" name="Normal" stackId="a" fill={NORMAL} /><Bar dataKey="anomaly" name="Potential anomaly" stackId="a" fill={ANOM} radius={[4, 4, 0, 0]} />
              </BarChart></ResponsiveContainer></div>
            </ChartCard>
            <ChartCard title="Cluster morphology profile" sub="Mean z-score of core features per cluster (clipped to ±2). The dashed line is the flagged group." legend={[...kIds.map(k => ({ label: `Cluster ${k}`, color: palette[Math.abs(k) % palette.length] })), ...(anomalies.length ? [{ label: "Flagged", color: ANOM }] : [])]}>
              {radarData.length < 3 ? <p className="empty">Not enough feature dimensions for a profile.</p> :
                <div className="chart"><ResponsiveContainer width="100%" height="100%"><RadarChart data={radarData} outerRadius="72%">
                  <PolarGrid stroke={LINE} /><PolarAngleAxis dataKey="feature" tick={{ fontSize: 11, fill: INK2 }} />
                  <PolarRadiusAxis domain={[-2, 2]} tick={{ fontSize: 9, fill: GREY }} angle={90} tickCount={5} stroke={LINE} />
                  {kIds.map(k => <Radar key={k} dataKey={`c${k}`} name={`Cluster ${k}`} stroke={palette[Math.abs(k) % palette.length]} fill={palette[Math.abs(k) % palette.length]} fillOpacity={0.14} strokeWidth={2} isAnimationActive={false} />)}
                  {anomalies.length > 0 && <Radar dataKey="anomaly" name="Flagged" stroke={ANOM} fill="none" strokeWidth={2} strokeDasharray="5 4" isAnimationActive={false} />}
                  <Tooltip contentStyle={tipStyle} formatter={(v: any) => Number(v).toFixed(2)} />
                </RadarChart></ResponsiveContainer></div>}
            </ChartCard>
          </div>
        </>}

        {tab === "dist" && <>
          <div className="grid g2">
            <ChartCard title="Feature distribution" sub="Stacked histogram. Dashed lines show the mean and median."
              actions={<Field label="Feature"><select value={validKey(histFeat)} onChange={e => setHistFeat(e.target.value)}>{feats.map(f => <option key={f.key} value={f.key}>{f.label}</option>)}</select></Field>}
              legend={[{ label: "Normal", color: NORMAL }, { label: "Potential anomaly", color: ANOM }]}>
              <Hist nuclei={nuclei} field={validKey(histFeat)} xLabel={labelOf(validKey(histFeat))}
                markers={[{ value: mean(nuclei.map(n => n[validKey(histFeat)]).filter(isNum) as number[]), label: "mean", color: "#ECEAF5" }, { value: median(nuclei.map(n => n[validKey(histFeat)]).filter(isNum) as number[]), label: "median", color: "#5B9BE6" }]} />
            </ChartCard>
            <ChartCard title="Isolation Forest score" sub="Dashed line is the boundary between normal and flagged nuclei." legend={[{ label: "Normal", color: NORMAL }, { label: "Potential anomaly", color: ANOM }]}>
              <Hist nuclei={nuclei} field="anomaly_score" xLabel="Score" markers={isNum(boundary) ? [{ value: boundary, label: "boundary", color: "#ECEAF5" }] : []} />
            </ChartCard>
          </div>
          <div className="grid g2">
            <ChartCard title="Normal vs potential anomalies" sub="Share of nuclei flagged as outliers.">
              <div className="chart donutWrap"><ResponsiveContainer width="100%" height="100%"><PieChart>
                <Pie data={donut} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="82%" paddingAngle={2} stroke="none" isAnimationActive={false}>
                  {donut.map((_, i) => <Cell key={i} fill={i === 0 ? NORMAL : ANOM} />)}</Pie>
                <Tooltip contentStyle={tipStyle} /></PieChart></ResponsiveContainer>
                <div className="donutLabel"><b>{rate.toFixed(1)}%</b><span>flagged</span></div></div>
            </ChartCard>
            <ChartCard title="DBSCAN groups" sub="Density-based groups. Noise points are nuclei outside any dense region.">
              <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={dbData} layout="vertical" margin={{ top: 6, right: 18, left: 8, bottom: 4 }}>
                <CartesianGrid stroke={LINE} strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" allowDecimals={false} {...ax} /><YAxis type="category" dataKey="group" {...ax} width={64} />
                <Tooltip contentStyle={tipStyle} cursor={{ fill: "rgba(156,144,240,.12)" }} />
                <Bar dataKey="count" name="Nuclei" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {dbData.map((d, i) => <Cell key={i} fill={d.noise || d.group === "Not run" ? GREY : palette[Math.abs(d.idx) % palette.length]} />)}</Bar>
              </BarChart></ResponsiveContainer></div>
            </ChartCard>
          </div>
        </>}

        {tab === "dist" && <FeatureStats feats={feats} normal={normal} anomalies={anomalies} nuclei={nuclei} />}

        {tab === "corr" && <div className="grid split">
          <ChartCard title="Feature correlation" sub="Pearson r across all nuclei in this image. Blue is positive, amber is negative.">
            <Heatmap feats={feats} rows={nuclei} />
          </ChartCard>
          <div className="stack">
            <section className="card"><div className="cardhead"><div><h2>Strongest relationships</h2><p>Highly correlated features carry overlapping information.</p></div></div>
              <ul className="pairs">{pairs.map(p => <li key={p.a + p.b}><span>{p.a} <em>×</em> {p.b}</span><b className={p.r >= 0 ? "pos" : "neg"}>{p.r.toFixed(2)}</b></li>)}</ul></section>
            {metrics.length > 0 && <section className="card"><div className="cardhead"><div><h2>Recorded model metrics</h2><p>Stored with this analysis by the backend.</p></div></div>
              <ul className="pairs">{metrics.map((m: any, i: number) => <li key={i}><span>{m.metric_name}{m.model_name ? <em> · {m.model_name}</em> : null}</span><b>{typeof m.metric_value === "number" ? m.metric_value.toFixed(4) : String(m.metric_value)}</b></li>)}</ul></section>}
          </div>
        </div>}

        {tab === "spatial" && <ChartCard title="Spatial distribution of nuclei" sub="Each marker is a nucleus centroid placed on the original image. Rings mark potential anomalies."
          actions={<MapControls {...{ swapXY, setSwapXY, showNormal, setShowNormal, showImage, setShowImage }} />}
          legend={[{ label: "Normal", color: NORMAL }, { label: "Potential anomaly", color: ANOM, ring: true }]}>
          <SpatialMap {...mapProps} onPick={(id: number) => { setPick(id); setPage("anomalies"); }} />
        </ChartCard>}

        {tab === "table" && <section className="card">
          <div className="cardhead"><div><h2>Nucleus-level measurements</h2><p>Click a column heading to sort. Export the CSV for all 18 features.</p></div>
            <button className="btn ghost" onClick={() => setPage("anomalies")}>Review anomalies</button></div>
          <NucleusTable rows={nuclei} />
        </section>}
      </>}</>}

      {/* ═════════ ANOMALIES ═════════ */}
      {page === "anomalies" && <>{!details ? <p className="empty">Select an analysis first.</p> : <>
        <section className="alertBand">
          <div className="bigStat"><b>{anomalies.length}</b><span>of {nuclei.length} nuclei flagged</span></div>
          <div className="bandCells">
            <div><small>Anomaly rate</small><strong>{rate.toFixed(2)}%</strong></div>
            <div><small>Score boundary</small><strong>{fmt(boundary, 4)}</strong></div>
            <div><small>Lowest score</small><strong>{fmt(ranked[0]?.anomaly_score, 4)}</strong></div>
            <div><small>Lead feature</small><strong>{drivers[0]?.name ?? "—"}</strong></div>
          </div>
          <button className="btn ghost" onClick={() => exportCsv(anomalies)} disabled={!anomalies.length}>Export anomalies</button>
        </section>

        {anomalies.length === 0 ? <section className="card"><p className="empty">No potential anomalies were flagged for this image.</p></section> : <>
          <div className="grid review">
            <ChartCard title="Where they sit" sub="Select a ring to inspect that nucleus."
              actions={<MapControls {...{ swapXY, setSwapXY, showNormal, setShowNormal, showImage, setShowImage }} />}>
              <SpatialMap {...mapProps} onPick={(id: number) => setPick(id)} />
            </ChartCard>
            <section className="card focusCard">
              {focus && <>
                <div className="cardhead"><div><h2>Nucleus {focus.nucleus_id}</h2><p>Rank {ranked.indexOf(focus) + 1} of {ranked.length} · score {fmt(focus.anomaly_score, 4)}</p></div><span className="pill anomaly">Anomaly</span></div>
                <dl className="facts">
                  <div><dt>Area</dt><dd>{fmt(focus.area, 1)}</dd></div><div><dt>Circularity</dt><dd>{fmt(focus.circularity)}</dd></div>
                  <div><dt>Eccentricity</dt><dd>{fmt(focus.eccentricity)}</dd></div><div><dt>Mean intensity</dt><dd>{fmt(focus.mean_intensity, 1)}</dd></div>
                  <div><dt>K-Means</dt><dd>{focus.kmeans_cluster ?? "—"}</dd></div><div><dt>DBSCAN</dt><dd>{focus.dbscan_cluster === -1 ? "Noise" : focus.dbscan_cluster ?? "—"}</dd></div>
                </dl>
                <h3>Deviation from this image (z-score)</h3>
                <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={focusZ} layout="vertical" margin={{ top: 4, right: 18, left: 4, bottom: 4 }}>
                  <CartesianGrid stroke={LINE} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" {...ax} domain={["auto", "auto"]} /><YAxis type="category" dataKey="name" {...ax} width={112} />
                  <ReferenceLine x={0} stroke={INK2} />
                  <Tooltip contentStyle={tipStyle} cursor={{ fill: "rgba(156,144,240,.12)" }} formatter={(v: any) => [Number(v).toFixed(2), "z-score"]} />
                  <Bar dataKey="z" isAnimationActive={false} radius={[3, 3, 3, 3]}>{focusZ.map((d, i) => <Cell key={i} fill={d.z >= 0 ? ANOM : "#5B9BE6"} />)}</Bar>
                </BarChart></ResponsiveContainer></div>
              </>}
            </section>
          </div>

          <div className="grid g2">
            <ChartCard title="What drives the flags" sub="Mean absolute z-score by feature. Larger gaps show features that separate flagged nuclei." legend={[{ label: "Flagged", color: ANOM }, { label: "Normal", color: NORMAL }]}>
              <div className="chart tall"><ResponsiveContainer width="100%" height="100%"><BarChart data={drivers} layout="vertical" margin={{ top: 4, right: 18, left: 4, bottom: 4 }}>
                <CartesianGrid stroke={LINE} strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" {...ax} /><YAxis type="category" dataKey="name" {...ax} width={112} />
                <Tooltip contentStyle={tipStyle} cursor={{ fill: "rgba(156,144,240,.12)" }} />
                <Bar dataKey="flagged" name="Flagged" fill={ANOM} radius={[0, 3, 3, 0]} isAnimationActive={false} />
                <Bar dataKey="normal" name="Normal" fill={NORMAL} radius={[0, 3, 3, 0]} isAnimationActive={false} />
              </BarChart></ResponsiveContainer></div>
            </ChartCard>
            <ChartCard title="Flagged nuclei in morphology space" sub="Same PCA embedding as the analysis page, with the selected nucleus enlarged." legend={[{ label: "Normal", color: NORMAL }, { label: "Potential anomaly", color: ANOM }, { label: "Selected", color: "#ECEAF5" }]}>
              <div className="chart tall"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 10, right: 16, left: 4, bottom: 22 }}>
                <CartesianGrid stroke={LINE} strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" {...ax} domain={["auto", "auto"]} tickFormatter={(v: any) => fmt(v, 1)} label={{ value: "PC1", position: "insideBottom", offset: -12, fontSize: 12, fill: INK2 }} />
                <YAxis type="number" dataKey="y" {...ax} domain={["auto", "auto"]} tickFormatter={(v: any) => fmt(v, 1)} width={46} label={{ value: "PC2", angle: -90, position: "insideLeft", fontSize: 12, fill: INK2 }} />
                <ZAxis range={[40, 40]} />
                <Tooltip content={<PointTip />} cursor={{ strokeDasharray: "3 3", stroke: GREY }} />
                <Scatter data={nuclei.filter(n => n.anomaly_status !== "Anomaly" && isNum(n.pc1) && isNum(n.pc2)).map(n => ({ x: n.pc1, y: n.pc2, id: n.nucleus_id, status: n.anomaly_status, cluster: n.kmeans_cluster, db: n.dbscan_cluster, score: n.anomaly_score, xl: "PC1", yl: "PC2" }))} shape={dot(NORMAL, 0.45)} isAnimationActive={false} />
                <Scatter data={ranked.filter(n => isNum(n.pc1) && isNum(n.pc2)).map(n => ({ x: n.pc1, y: n.pc2, id: n.nucleus_id, status: n.anomaly_status, cluster: n.kmeans_cluster, db: n.dbscan_cluster, score: n.anomaly_score, xl: "PC1", yl: "PC2" }))} shape={dot(ANOM, 0.9, 5.2)} isAnimationActive={false} />
                {focus && isNum(focus.pc1) && isNum(focus.pc2) && <Scatter data={[{ x: focus.pc1, y: focus.pc2, id: focus.nucleus_id, status: focus.anomaly_status, cluster: focus.kmeans_cluster, db: focus.dbscan_cluster, score: focus.anomaly_score, xl: "PC1", yl: "PC2" }]} shape={ring} isAnimationActive={false} />}
              </ScatterChart></ResponsiveContainer></div>
            </ChartCard>
          </div>

          <section className="card">
            <div className="cardhead"><div><h2>Flagged nuclei, ranked</h2><p>Most anomalous first. Select a row to inspect it above.</p></div></div>
            <NucleusTable rows={ranked} onPick={(id: number) => { setPick(id); window.scrollTo({ top: 0, behavior: "smooth" }); }} selectedId={focus?.nucleus_id} />
          </section>
        </>}

        <section className="card interpret"><h2>How to read this result</h2>
          <p>Isolation Forest flags nuclei whose feature values are unusual relative to the rest of this image. It does not decide whether a nucleus is diseased or clinically abnormal. Compare the flagged positions with the original image and check the segmentation outline before drawing conclusions.</p></section>
      </>}</>}

      <footer>MorphoLens · Exploratory microscopy analytics · Research use only, not for clinical diagnosis</footer>
    </main>
  </div>;
}

/* ───────────────────────── Chart shapes & tooltip ───────────────────────── */
const dot = (color: string, op = 0.72, r = 4.2) => (p: any) =>
  <circle cx={p.cx} cy={p.cy} r={r} fill={color} fillOpacity={op} stroke="#0F0D18" strokeWidth={0.7} />;
const ring = (p: any) => <g><circle cx={p.cx} cy={p.cy} r={9} fill="none" stroke="#0F0D18" strokeWidth={4} /><circle cx={p.cx} cy={p.cy} r={9} fill="none" stroke={ANOM} strokeWidth={2} /></g>;

function PointTip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return <div className="tip">
    <b>Nucleus {p.id}</b>
    <span>{p.xl}: {fmt(p.x)}</span><span>{p.yl}: {fmt(p.y)}</span>
    <span>K-Means {p.cluster ?? "—"} · DBSCAN {p.db === -1 ? "Noise" : p.db ?? "—"}</span>
    <span className={p.status === "Anomaly" ? "tipFlag" : ""}>{p.status ?? "Not run"}{isNum(p.score) ? ` · score ${p.score.toFixed(4)}` : ""}</span>
  </div>;
}

/* ───────────────────────── Components ───────────────────────── */
function Kpi({ label, value, sub, unit, tone, gauge }: { label: string; value: any; sub?: string; unit?: string; tone?: string; gauge?: number }) {
  const g = isNum(Number(gauge)) ? Math.max(0, Math.min(100, Number(gauge))) : null;
  return <div className={tone === "alert" ? "kpi alert" : "kpi"}>
    <small>{label}</small>
    <strong>{value ?? "—"}{unit && <u>{unit}</u>}</strong>
    {sub && <em>{sub}</em>}
    {g !== null && <svg className="gauge" viewBox="0 0 44 44" aria-hidden="true"><circle cx="22" cy="22" r="17" fill="none" stroke="var(--line)" strokeWidth="4" /><circle cx="22" cy="22" r="17" fill="none" stroke="var(--crimson)" strokeWidth="4" strokeLinecap="round" strokeDasharray={`${(Math.max(g, 1.5) / 100) * 106.8} 106.8`} transform="rotate(-90 22 22)" /></svg>}
  </div>;
}
function Field({ label, children }: any) { return <label className="field"><span>{label}</span>{children}</label>; }
function Seg({ value, onChange, options }: any) {
  return <div className="seg">{options.map(([k, l]: [string, string]) => <button key={k} className={value === k ? "on" : ""} aria-pressed={value === k} onClick={() => onChange(k)}>{l}</button>)}</div>;
}
function ChartCard({ title, sub, legend, actions, children }: any) {
  return <section className="card chartCard">
    <div className="cardhead"><div><h2>{title}</h2>{sub && <p>{sub}</p>}</div></div>
    {actions && <div className="cardtools">{actions}</div>}
    {children}
    {legend && legend.length > 0 && <ul className="legend">{legend.map((l: any) => <li key={l.label}><i className={l.ring ? "sw ring" : "sw"} style={l.ring ? { borderColor: l.color } : { background: l.color }} />{l.label}</li>)}</ul>}
  </section>;
}
function MapControls({ swapXY, setSwapXY, showNormal, setShowNormal, showImage, setShowImage }: any) {
  return <div className="controls">
    <label className="check"><input type="checkbox" checked={showImage} onChange={e => setShowImage(e.target.checked)} />Show image</label>
    <label className="check"><input type="checkbox" checked={showNormal} onChange={e => setShowNormal(e.target.checked)} />Show normal nuclei</label>
    <label className="check" title="Use this if markers look mirrored across the diagonal"><input type="checkbox" checked={swapXY} onChange={e => setSwapXY(e.target.checked)} />Swap X / Y</label>
  </div>;
}

function Hist({ nuclei, field, markers = [], height = 260, xLabel }: any) {
  const h = useMemo(() => histRows(nuclei, field), [nuclei, field]);
  return <div className="chart" style={{ height }}><ResponsiveContainer width="100%" height="100%">
    <BarChart data={h.rows} margin={{ top: 18, right: 14, left: 0, bottom: 20 }} barCategoryGap={1}>
      <CartesianGrid stroke={LINE} strokeDasharray="3 3" vertical={false} />
      <XAxis dataKey="x" {...ax} interval="preserveStartEnd" minTickGap={30} tickFormatter={(v: any) => Number(v).toFixed(h.d)} label={{ value: xLabel, position: "insideBottom", offset: -12, fontSize: 12, fill: INK2 }} />
      <YAxis allowDecimals={false} {...ax} width={34} />
      <Tooltip contentStyle={tipStyle} cursor={{ fill: "rgba(156,144,240,.12)" }} labelFormatter={(v: any) => `${xLabel}: ${Number(v).toFixed(h.d + 1)}`} />
      <Bar dataKey="normal" name="Normal" stackId="a" fill={NORMAL} isAnimationActive={false} />
      <Bar dataKey="anomaly" name="Potential anomaly" stackId="a" fill={ANOM} isAnimationActive={false} />
      {markers.filter((m: any) => isNum(m.value)).map((m: any) =>
        <ReferenceLine key={m.label} x={h.b.center(h.b.index(m.value))} stroke={m.color} strokeDasharray="5 3" strokeWidth={1.5}
          label={{ value: m.label, position: "top", fontSize: 10, fill: m.color }} />)}
    </BarChart></ResponsiveContainer></div>;
}

function Heatmap({ feats, rows }: any) {
  const m: number[][] = useMemo(() => feats.map((a: any) => feats.map((b: any) =>
    pearson(rows.map((n: any) => n[a.key]), rows.map((n: any) => n[b.key])))), [feats, rows]);
  const n = feats.length;
  if (n < 2) return <p className="empty">Not enough variable features to correlate.</p>;
  const bg = (r: number) => r >= 0 ? `rgba(91,155,230,${Math.min(1, Math.abs(r))})` : `rgba(235,174,75,${Math.min(1, Math.abs(r))})`;
  return <div>
    <div className="heatScroll"><div className="heat" style={{ gridTemplateColumns: `78px repeat(${n}, minmax(28px, 1fr))` }}>
      <span />
      {feats.map((f: any) => <span key={f.key} className="hcol" title={f.label}>{f.short}</span>)}
      {feats.map((a: any, i: number) => <Fragment key={a.key}>
        <span className="hrow" title={a.label}>{a.short}</span>
        {feats.map((b: any, j: number) => {
          const r = m[i][j];
          return <span key={b.key} className="hcell" style={{ background: bg(r), color: Math.abs(r) > 0.6 ? "#0F0D18" : "#ECEAF5" }}
            title={`${a.label} × ${b.label}: r = ${r.toFixed(2)}`}>{i !== j && Math.abs(r) >= 0.5 ? r.toFixed(1) : ""}</span>;
        })}
      </Fragment>)}
    </div></div>
    <div className="scale"><span>−1</span><i /><span>+1</span></div>
  </div>;
}

function SpatialMap({ nuclei, width, height, src, swap, showNormal, showImage, selectedId, onPick }: any) {
  const pts = nuclei.map((n: any) => ({ n, x: swap ? n.centroid_y : n.centroid_x, y: swap ? n.centroid_x : n.centroid_y }))
    .filter((p: any) => isNum(p.x) && isNum(p.y));
  if (!pts.length) return <p className="empty">Centroid coordinates are not available for this run.</p>;
  const w = width || Math.max(...pts.map((p: any) => p.x)) * 1.05;
  const h = height || Math.max(...pts.map((p: any) => p.y)) * 1.05;
  const s = Math.max(w, h) / 420;
  return <div className="spatial"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Nucleus centroid map">
    {showImage && width && <image href={`${API}${src}`} width={w} height={h} preserveAspectRatio="none" />}
    {showNormal && pts.filter((p: any) => p.n.anomaly_status !== "Anomaly").map((p: any) =>
      <circle key={p.n.nucleus_id} cx={p.x} cy={p.y} r={s * 1.5} fill="#9FB4DA" fillOpacity={0.7} stroke="#07050D" strokeWidth={s * 0.3}><title>{`Nucleus ${p.n.nucleus_id}`}</title></circle>)}
    {pts.filter((p: any) => p.n.anomaly_status === "Anomaly").map((p: any) => {
      const sel = p.n.nucleus_id === selectedId;
      return <g key={p.n.nucleus_id} className="mk" onClick={() => onPick && onPick(p.n.nucleus_id)}>
        <circle cx={p.x} cy={p.y} r={s * (sel ? 6.5 : 4)} fill="none" stroke="#0F0D18" strokeWidth={s * (sel ? 2.8 : 2.2)} />
        <circle cx={p.x} cy={p.y} r={s * (sel ? 6.5 : 4)} fill={sel ? "rgba(255,77,109,.22)" : "none"} stroke={ANOM} strokeWidth={s * (sel ? 1.5 : 1.1)} />
        <circle cx={p.x} cy={p.y} r={s * 1.1} fill={ANOM} />
        <title>{`Nucleus ${p.n.nucleus_id} · score ${fmt(p.n.anomaly_score, 4)}`}</title>
      </g>;
    })}
  </svg></div>;
}

function ImagePanel({ title, src }: { title: string; src: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  return <figure className="imagePanel">
    <div className="vp">{failed ? <span className="vpMissing">{title} unavailable</span> :
      <img src={`${API}${src}`} alt={title} onError={() => setFailed(true)} />}</div>
    <figcaption>{title}</figcaption>
  </figure>;
}

function NucleusTable({ rows, onPick, selectedId }: any) {
  const [sort, setSort] = useState<{ k: string; dir: 1 | -1 } | null>(null);
  const [limit, setLimit] = useState(25);
  const cols: [string, string][] = [["nucleus_id", "Nucleus"], ["area", "Area"], ["circularity", "Circularity"], ["eccentricity", "Eccentricity"], ["solidity", "Solidity"], ["mean_intensity", "Mean int."], ["kmeans_cluster", "K-Means"], ["dbscan_cluster", "DBSCAN"], ["anomaly_status", "Isolation Forest"], ["anomaly_score", "Score"]];
  const sorted = useMemo(() => {
    if (!sort) return rows;
    return [...rows].sort((a: any, b: any) => {
      const x = a[sort.k], y = b[sort.k];
      if (x == null) return 1; if (y == null) return -1;
      return (x > y ? 1 : x < y ? -1 : 0) * sort.dir;
    });
  }, [rows, sort]);
  const num = (v: any, d: number) => (v == null || v === "" ? "—" : Number(v).toFixed(d));
  return <>
    <div className="tableWrap"><table><thead><tr>{cols.map(([k, l]) =>
      <th key={k} aria-sort={sort?.k === k ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
        <button className="th" onClick={() => setSort(sort?.k === k ? { k, dir: (sort.dir * -1) as 1 | -1 } : { k, dir: 1 })}>{l}<i>{sort?.k === k ? (sort.dir === 1 ? "▲" : "▼") : ""}</i></button></th>)}</tr></thead>
      <tbody>{sorted.slice(0, limit).map((n: any) => <tr key={n.nucleus_id} className={(n.nucleus_id === selectedId ? "sel " : "") + (onPick ? "click" : "")} onClick={() => onPick && onPick(n.nucleus_id)}>
        <td>{n.nucleus_id}</td><td>{num(n.area, 1)}</td><td>{num(n.circularity, 3)}</td><td>{num(n.eccentricity, 3)}</td><td>{num(n.solidity, 3)}</td><td>{num(n.mean_intensity, 1)}</td>
        <td>{n.kmeans_cluster ?? "—"}</td><td>{n.dbscan_cluster === -1 ? "Noise" : n.dbscan_cluster ?? "—"}</td>
        <td><span className={n.anomaly_status === "Anomaly" ? "pill anomaly" : "pill"}>{n.anomaly_status ?? "Not run"}</span></td>
        <td>{n.anomaly_score == null ? "—" : Number(n.anomaly_score).toFixed(4)}</td></tr>)}</tbody></table></div>
    {sorted.length > limit && <div className="more"><span>Showing {limit} of {sorted.length}</span><button className="btn ghost sm" onClick={() => setLimit(limit + 50)}>Show 50 more</button></div>}
  </>;
}

function FeatureStats({ feats, normal, anomalies, nuclei }: any) {
  const rows = useMemo(() => feats.map((f: any) => {
    const all = nuclei.map((n: any) => n[f.key]).filter(isNum) as number[];
    const nv = normal.map((n: any) => n[f.key]).filter(isNum) as number[];
    const av = anomalies.map((n: any) => n[f.key]).filter(isNum) as number[];
    const sp = Math.sqrt(((nv.length - 1) * sd(nv) ** 2 + (av.length - 1) * sd(av) ** 2) / Math.max(1, nv.length + av.length - 2));
    const d = av.length > 1 && nv.length > 1 && sp > 0 ? (mean(av) - mean(nv)) / sp : NaN;
    return { f, m: mean(all), s: sd(all), med: median(all), nm: mean(nv), am: av.length ? mean(av) : NaN, d };
  }).sort((x: any, y: any) => (isNum(y.d) ? Math.abs(y.d) : -1) - (isNum(x.d) ? Math.abs(x.d) : -1)), [feats, normal, anomalies, nuclei]);
  return <section className="card chartCard">
    <div className="cardhead"><div><h2>Feature statistics</h2><p>Effect size (Cohen's d) compares flagged against normal nuclei. Bars to the right mean flagged nuclei have higher values.</p></div></div>
    <div className="tableWrap"><table className="fstats"><thead><tr><th>Feature</th><th>Mean ± SD</th><th>Median</th><th>Normal mean</th><th>Flagged mean</th><th>Effect size (d)</th></tr></thead>
      <tbody>{rows.map((r: any) => <tr key={r.f.key}>
        <td>{r.f.label}</td><td>{fmt(r.m, 2)} ± {fmt(r.s, 2)}</td><td>{fmt(r.med, 2)}</td><td>{fmt(r.nm, 2)}</td><td>{fmt(r.am, 2)}</td>
        <td><div className="dbar"><i className="mid" />{isNum(r.d) && <b className={r.d >= 0 ? "pos" : "neg"} style={{ width: `${Math.min(50, Math.abs(r.d) * 12)}%`, [r.d >= 0 ? "left" : "right"]: "50%" } as any} />}<span>{isNum(r.d) ? r.d.toFixed(2) : "—"}</span></div></td>
      </tr>)}</tbody></table></div>
  </section>;
}