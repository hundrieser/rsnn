import React, { useMemo, useRef, useState, useEffect, useCallback } from "react";
import { jsPDF } from "jspdf";
import "svg2pdf.js";

type FileSystemHandlePermissionMode = "read" | "readwrite";

type FileSystemWritableFileStream = {
  write(data: BlobPart | BufferSource | Blob | string): Promise<void>;
  close(): Promise<void>;
};

type FileSystemFileHandle = {
  name: string;
  createWritable(options?: { keepExistingData?: boolean }): Promise<FileSystemWritableFileStream>;
};

type FileSystemDirectoryHandle = {
  name: string;
  getFileHandle(name: string, options?: { create?: boolean }): Promise<FileSystemFileHandle>;
};

declare global {
  interface Window {
    showDirectoryPicker?: (options?: { mode?: FileSystemHandlePermissionMode }) => Promise<FileSystemDirectoryHandle>;
  }
}

/**
 * Recurrent Spiking Neural Network — Canvas GUI
 * --------------------------------------------------
 * Features
 *  - Add/move/delete neurons on an SVG canvas
 *  - Create directed connections (no self-connections); edit weights
 *  - Mark neurons as Input / Hidden / Output
 *  - Assign input spike trains to Input neurons
 *  - Event-driven recurrent spiking simulation with cascades per input time
 *  - Exponential decay with memory coefficient h (supports 0 and ∞)
 *  - Raster plot visualization of spike trains for all neurons
 *
 * Math
 *  For an input/event time t, for each neuron j, let
 *    base_j = P_j * exp(-(t - lastUpdate_j)/h)    (with h=0 => 0 if t>last, h=∞ => 1)
 *    meta_j = max(0, base_j)
 *  Then propagate spikes in waves:
 *    - Start frontier = all input neurons whose exogenous train contains t (they spike).
 *    - While frontier not empty:
 *        For edges i->j from spiking i, meta_j = max(0, meta_j + w_{ij}).
 *        New spikes = { j | meta_j > 1 and not yet spiked at t }.
 *        For each new spike j: record spike at t; meta_j = 0; add j to next frontier.
 *    - After waves, set P_j(t) = 0 if spiked else meta_j, and lastUpdate_j = t.
 *
 * Notes
 *  - Strict threshold: spikes occur iff meta > 1.
 *  - Input neurons: their exogenous spikes seed cascades; they can also spike from network input when not exogenously spiking.
 *  - Edges i->i are disallowed at the UI level.
 */

// ------------------------ Types ------------------------

type Role = "input" | "hidden" | "output";

interface Neuron {
  id: string;
  label: string;
  role: Role;
  x: number;
  y: number;
  labelVisible?: boolean;
  labelOffsetX?: number;
  labelOffsetY?: number;
}

interface Edge {
  id: string;
  sourceId: string;
  targetId: string;
  weight: number; // can be negative
  points?: Array<{ x: number; y: number }>;
}

type GroupCompressionState = {
  center: { x: number; y: number };
  nodes: Record<
    string,
    {
      x: number;
      y: number;
      labelOffsetX: number;
      labelOffsetY: number;
      labelVisible: boolean;
    }
  >;
  edgePoints: Record<string, Point[]>;
  labelVisibleBefore: boolean;
};

interface Group {
  id: string;
  nodeIds: string[];
  hue: number; // for pastel coloring
  label: string;
  labelVisible: boolean;
  compression?: GroupCompressionState;
}

interface Module {
  id: string;
  name: string;
  neurons: Neuron[];
  edges: Edge[];
  inputNeuronIds: string[];
  outputNeuronIds: string[];
  canvas: {
    width: number;
    height: number;
  };
}

interface PotentialSeries {
  times: number[];
  values: Record<string, number[]>; // neuronId -> potentials
}

type CascadeFrame = {
  time: number;
  wave: number;
  sources: string[];
  targets: string[];
  edges: string[];
};

interface SimResult {
  spikeTrains: Record<string, number[]>; // neuronId -> times
  finalP: Record<string, number>;
  potentialSeries: PotentialSeries;
  cascadeFrames: CascadeFrame[];
}

type AnimationFrame = {
  time: number;
  wave: number;
  sourceIds: string[];
  targetIds: string[];
  neuronIds: string[];
  edgeIds: string[];
};

interface RSNNDesignerProps {
  isDarkMode?: boolean;
  onToggleTheme?: () => void;
}

// ------------------------ Utilities ------------------------

function uid(prefix = "id"): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
}

function uniqueLabel(base: string, taken: Set<string>): string {
  const safeBase = base.trim() || "N";
  let attempt = safeBase;
  let counter = 2;
  while (taken.has(attempt)) {
    attempt = `${safeBase}_${counter++}`;
  }
  taken.add(attempt);
  return attempt;
}

function parseTimes(text: string): number[] {
  // Accept comma/space/newline separated numbers
  return text
    .split(/[\,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Number(s))
    .filter((n) => Number.isFinite(n) && n > 0)
    .sort((a, b) => a - b);
}

function uniqueSorted(nums: number[]): number[] {
  const out: number[] = [];
  let last: number | undefined;
  for (const n of nums) {
    if (last === undefined || Math.abs(n - last) > 1e-12) out.push(n);
    last = n;
  }
  return out;
}

function prepareCanvasSvgClone(svgElement: SVGSVGElement, width: number, height: number): SVGSVGElement {
  const clone = svgElement.cloneNode(true) as SVGSVGElement;
  clone.querySelectorAll<SVGRectElement>('rect[fill^="url("]').forEach((rect) => {
    const fill = rect.getAttribute("fill");
    if (fill && fill.includes("-grid")) {
      rect.setAttribute("fill", "#ffffff");
    }
  });
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.removeAttribute("style");
  clone.setAttribute("width", `${width}`);
  clone.setAttribute("height", `${height}`);
  clone.style.width = `${width}px`;
  clone.style.height = `${height}px`;
  return clone;
}

function waitForNextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      resolve();
    });
  });
}

function expDecay(delta: number, h: number): number {
  if (delta <= 0) return 1; // same-time use
  if (!Number.isFinite(h)) return 1; // h = ∞
  if (h === 0) return 0; // h = 0 and delta>0
  return Math.exp(-delta / h);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function snapToGrid(value: number, gridSize = GRID_SIZE): number {
  if (!Number.isFinite(value)) return 0;
  return Math.round(value / gridSize) * gridSize;
}

const NODE_RADIUS = 24;
const EDGE_NODE_PADDING = 6;
const EDGE_TARGET_TRIM = 1;
const EDGE_END_CLEARANCE = 0.35;
const EDGE_STROKE_WIDTH = 2.0;
const GRID_SIZE = 20;
const DEFAULT_LABEL_OFFSET_Y = 0;
const MIN_EVENT_DELAY_MS = 200;
const TIME_SCALE_MS = 600; // milliseconds per simulated second
const DEFAULT_MACRO_WIDTH = 480;
const DEFAULT_MACRO_HEIGHT = 280;
const DEFAULT_EDGE_WEIGHT = 2;
const DEFAULT_CANVAS_WIDTH = 800;
const DEFAULT_CANVAS_HEIGHT = 300;
const MIN_CANVAS_DIMENSION = 200;
const CANVAS_FRAME_PADDING = 20;

const ROLE_FILL: Record<Role, string> = {
  input: "#ffedb4",
  hidden: "#cafde0",
  output: "#b9cefb",
};

const EDGE_GAP = NODE_RADIUS + EDGE_NODE_PADDING;
const COMPRESSED_GROUP_FILL = "#5ae797ff";

type Point = { x: number; y: number };

type EdgeHandleSelection = { edgeId: string; index: number };

function offsetPoint(origin: Point, target: Point, distance: number): Point {
  const dx = target.x - origin.x;
  const dy = target.y - origin.y;
  const len = Math.hypot(dx, dy);
  if (!Number.isFinite(len) || len < 1e-6) {
    return { x: origin.x, y: origin.y };
  }
  const limit = Math.max(0, len - 1e-3);
  const bounded = Math.max(-limit, Math.min(limit, distance));
  const scale = bounded / len;
  return { x: origin.x + dx * scale, y: origin.y + dy * scale };
}

function sanitizeWaypoints(points?: Array<Point>): Array<Point> {
  if (!Array.isArray(points)) return [];
  return points
    .map((pt) => {
      if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return null;
      return { x: pt.x, y: pt.y };
    })
    .filter((pt): pt is Point => pt !== null);
}

function buildEdgePoints(edge: Edge, source: Neuron, target: Neuron): Point[] {
  const waypoints = sanitizeWaypoints(edge.points);
  const firstTarget = waypoints[0] ?? { x: target.x, y: target.y };
  const start = offsetPoint({ x: source.x, y: source.y }, firstTarget, EDGE_GAP);
  const prevForEnd = waypoints.length ? waypoints[waypoints.length - 1] : start;
  const endDistance = Math.max(NODE_RADIUS + EDGE_END_CLEARANCE, EDGE_GAP - EDGE_TARGET_TRIM);
  const end = offsetPoint({ x: target.x, y: target.y }, prevForEnd, endDistance);
  return [start, ...waypoints, end];
}

function totalPathLength(points: Point[]): number {
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
  }
  return total;
}

function pointAlongPath(points: Point[], t: number): Point {
  if (!points.length) return { x: 0, y: 0 };
  if (points.length === 1) return points[0];
  const total = totalPathLength(points);
  if (total <= 1e-6) return points[0];
  const targetDistance = Math.max(0, Math.min(1, t)) * total;
  let traveled = 0;
  for (let i = 1; i < points.length; i++) {
    const start = points[i - 1];
    const end = points[i];
    const segment = Math.hypot(end.x - start.x, end.y - start.y);
    if (traveled + segment >= targetDistance) {
      const remain = targetDistance - traveled;
      const ratio = segment === 0 ? 0 : remain / segment;
      return {
        x: start.x + (end.x - start.x) * ratio,
        y: start.y + (end.y - start.y) * ratio,
      };
    }
    traveled += segment;
  }
  return points[points.length - 1];
}

function pathMidpoint(points: Point[]): Point {
  return pointAlongPath(points, 0.5);
}

function distancePointToSegment(point: Point, a: Point, b: Point): number {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const l2 = abx * abx + aby * aby;
  if (l2 <= 1e-9) {
    return Math.hypot(point.x - a.x, point.y - a.y);
  }
  let t = ((point.x - a.x) * abx + (point.y - a.y) * aby) / l2;
  t = Math.max(0, Math.min(1, t));
  const projX = a.x + abx * t;
  const projY = a.y + aby * t;
  return Math.hypot(point.x - projX, point.y - projY);
}

function projectPointToSegment(point: Point, a: Point, b: Point): Point {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const l2 = abx * abx + aby * aby;
  if (l2 <= 1e-9) return { x: a.x, y: a.y };
  let t = ((point.x - a.x) * abx + (point.y - a.y) * aby) / l2;
  t = Math.max(0, Math.min(1, t));
  return { x: a.x + abx * t, y: a.y + aby * t };
}

function closestPointOnPath(point: Point, pathPoints: Point[]): { point: Point; segment: number } | null {
  if (pathPoints.length < 2) return null;
  let bestDistance = Number.POSITIVE_INFINITY;
  let bestPoint = pathPoints[0];
  let bestSegment = 0;
  for (let i = 0; i < pathPoints.length - 1; i++) {
    const candidate = projectPointToSegment(point, pathPoints[i], pathPoints[i + 1]);
    const dist = Math.hypot(point.x - candidate.x, point.y - candidate.y);
    if (dist < bestDistance) {
      bestDistance = dist;
      bestPoint = candidate;
      bestSegment = i;
    }
  }
  return { point: bestPoint, segment: bestSegment };
}

function waypointInsertIndex(pathPoints: Point[], waypointCount: number, clickPoint: Point): number {
  if (pathPoints.length < 2) return waypointCount;
  let bestSegment = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < pathPoints.length - 1; i++) {
    const start = pathPoints[i];
    const end = pathPoints[i + 1];
    const dist = distancePointToSegment(clickPoint, start, end);
    if (dist < bestDistance) {
      bestDistance = dist;
      bestSegment = i;
    }
  }
  return Math.min(bestSegment, waypointCount);
}

// ------------------------ Simulation ------------------------

function simulate(
  neurons: Neuron[],
  edges: Edge[],
  h: number, // use Number.POSITIVE_INFINITY to represent ∞
  T: number,
  inputSpikeTrains: Record<string, number[]>
): SimResult {
  const ids = neurons.map((n) => n.id);
  const indexOf: Record<string, number> = Object.fromEntries(ids.map((id, i) => [id, i]));

  // adjacency: from -> list of (toIndex, weight)
  const outs: Array<Array<{ j: number; w: number; edgeId: string }>> = neurons.map(() => []);
  for (const e of edges) {
    const i = indexOf[e.sourceId];
    const j = indexOf[e.targetId];
    if (i === undefined || j === undefined) continue;
    outs[i].push({ j, w: e.weight, edgeId: e.id });
  }

  // Gather all input times (union) within (0, T]
  const allTimes: number[] = [];
  for (const n of neurons) {
    const arr = inputSpikeTrains[n.id] || [];
    for (const t of arr) if (t > 0 && t <= T) allTimes.push(t);
  }
  if (T > 0) allTimes.push(T);
  allTimes.sort((a, b) => a - b);
  const times = uniqueSorted(allTimes);

  const k = neurons.length;
  const P = new Float64Array(k).fill(0);
  const lastUpdate = new Float64Array(k).fill(0);
  const spikeTrains: Record<string, number[]> = Object.fromEntries(ids.map((id) => [id, []]));

  const sampleStep = T > 0 ? Math.max(0.05, T / 200) : 0.1;
  const potentialSamples: Array<{ time: number; values: number[] }> = [
    { time: 0, values: Array.from({ length: k }, () => 0) },
  ];
  let prevTime = 0;
  const lastPost = new Float64Array(k).fill(0);
  const cascadeFrames: CascadeFrame[] = [];

  function sampleInterval(start: number, end: number) {
    const dt = end - start;
    if (dt <= 0) return;
    const steps = Math.max(1, Math.ceil(dt / sampleStep));
    for (let s = 1; s <= steps; s++) {
      const tSample = start + (dt * s) / steps;
      const delta = tSample - start;
      const values = new Array<number>(k);
      for (let j = 0; j < k; j++) {
        const base = lastPost[j];
        const decayed = base > 0 ? base * expDecay(delta, h) : 0;
        values[j] = decayed > 1e-12 ? decayed : 0;
      }
      potentialSamples.push({ time: tSample, values });
    }
  }

  for (const t of times) {
    sampleInterval(prevTime, t);

    // base potentials at time t (decay from lastUpdate)
    const meta = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      const base = P[j] * expDecay(t - lastUpdate[j], h);
      meta[j] = base > 0 ? base : 0; // positive part
    }

    const spiked = new Array<boolean>(k).fill(false);

    // initial frontier: input neurons that spike at t
    const frontier: number[] = [];
    for (let i = 0; i < k; i++) {
      const id = ids[i];
      const train = inputSpikeTrains[id] || [];
      // presence of t within numerical tolerance
      if (binaryIncludes(train, t)) {
        spiked[i] = true;
        spikeTrains[id].push(t);
        frontier.push(i);
        meta[i] = 0; // reset upon spike
      }
    }

    const timeFrames: CascadeFrame[] = [];
    let waveIndex = 0;

    if (frontier.length) {
      const initialTargets = frontier.map((idx) => ids[idx]);
      timeFrames.push({
        time: t,
        wave: waveIndex++,
        sources: [],
        targets: initialTargets,
        edges: [],
      });
    }

    // waves of propagation
    while (frontier.length) {
      const currentFrontier = frontier.slice();
      const next: number[] = [];
      const contributions = new Float64Array(k);

      // accumulate raw contributions from the current frontier
      for (const i of currentFrontier) {
        const out = outs[i];
        for (let p = 0; p < out.length; p++) {
          const { j, w } = out[p];
          contributions[j] += w;
        }
      }

      // apply contributions in a synchronized fashion so opposing weights cancel
      for (let j = 0; j < k; j++) {
        if (spiked[j]) continue; // already fired at time t
        const delta = contributions[j];
        if (delta === 0) continue;
        const updated = meta[j] + delta;
        meta[j] = updated > 0 ? updated : 0;
      }

      // find new spikes strictly above threshold
      for (let j = 0; j < k; j++) {
        if (!spiked[j] && meta[j] > 1) {
          spiked[j] = true;
          meta[j] = 0; // reset on spike
          spikeTrains[ids[j]].push(t);
          next.push(j);
        }
      }

      if (next.length) {
        const nextIds = next.map((idx) => ids[idx]);
        const nextSet = new Set(next);
        const edgeIdSet = new Set<string>();
        for (const idx of currentFrontier) {
          const out = outs[idx];
          for (let p = 0; p < out.length; p++) {
            const { j, edgeId } = out[p];
            if (nextSet.has(j)) {
              edgeIdSet.add(edgeId);
            }
          }
        }
        timeFrames.push({
          time: t,
          wave: waveIndex++,
          sources: currentFrontier.map((idx) => ids[idx]),
          targets: nextIds,
          edges: Array.from(edgeIdSet),
        });
      }

      frontier.length = 0;
      Array.prototype.push.apply(frontier, next);
    }

    if (timeFrames.length) {
      cascadeFrames.push(...timeFrames);
    }

    // finalize potentials and timestamps at time t
    for (let j = 0; j < k; j++) {
      P[j] = meta[j];
      lastUpdate[j] = t;
    }

    potentialSamples.push({ time: t, values: Array.from(P) });
    lastPost.set(P);
    prevTime = t;
  }

  // Only keep spikes up to T
  for (const id of ids) {
    spikeTrains[id] = (spikeTrains[id] || []).filter((s) => s <= T);
  }

  const finalP: Record<string, number> = Object.fromEntries(ids.map((id, j) => [id, P[j]]));
  const potentialSeriesValues: Record<string, number[]> = {};
  for (let j = 0; j < k; j++) {
    potentialSeriesValues[ids[j]] = potentialSamples.map((sample) => sample.values[j] ?? 0);
  }
  const potentialSeries: PotentialSeries = {
    times: potentialSamples.map((sample) => sample.time),
    values: potentialSeriesValues,
  };
  return { spikeTrains, finalP, potentialSeries, cascadeFrames };
}

function binaryIncludes(arr: number[], val: number): boolean {
  // Assume arr is sorted. Check equality within small tolerance.
  let lo = 0,
    hi = arr.length - 1;
  const eps = 1e-9;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const x = arr[mid];
    if (Math.abs(x - val) <= eps) return true;
    if (x < val) lo = mid + 1;
    else hi = mid - 1;
  }
  return false;
}

// ------------------------ Main Component ------------------------

export default function RSNNDesigner({ isDarkMode = false, onToggleTheme }: RSNNDesignerProps) {
  const [neurons, setNeurons] = useState<Neuron[]>(() => [
    { id: uid("n"), label: "N1", role: "input", x: 120, y: 140, labelVisible: false, labelOffsetX: 0, labelOffsetY: DEFAULT_LABEL_OFFSET_Y },
    { id: uid("n"), label: "N2", role: "hidden", x: 360, y: 200, labelVisible: false, labelOffsetX: 0, labelOffsetY: DEFAULT_LABEL_OFFSET_Y },
    { id: uid("n"), label: "N3", role: "output", x: 640, y: 160, labelVisible: false, labelOffsetX: 0, labelOffsetY: DEFAULT_LABEL_OFFSET_Y },
  ]);

  const [edges, setEdges] = useState<Edge[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [selectedEdgeHandle, setSelectedEdgeHandle] = useState<EdgeHandleSelection | null>(null);
  const [mode, setMode] = useState<"select" | "connect">("select");
  const [connectSrc, setConnectSrc] = useState<string | null>(null);
  const [connectPoints, setConnectPoints] = useState<Array<{ x: number; y: number }>>([]);
  const [activeSelectionContext, setActiveSelectionContext] = useState<"main" | "module">("main");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const canvasExportRef = useRef<HTMLDivElement | null>(null);
  const rasterExportRef = useRef<HTMLDivElement | null>(null);
  const potentialExportRef = useRef<HTMLDivElement | null>(null);

  const [T, setT] = useState<number>(10);
  const [hKind, setHKind] = useState<"finite" | "zero" | "infty">("finite");
  const [hVal, setHVal] = useState<number>(3);
  const [canvasWidth, setCanvasWidth] = useState<number>(DEFAULT_CANVAS_WIDTH);
  const [canvasHeight, setCanvasHeight] = useState<number>(DEFAULT_CANVAS_HEIGHT);
  const [canvasWidthInput, setCanvasWidthInput] = useState<string>(() => String(DEFAULT_CANVAS_WIDTH));
  const [canvasHeightInput, setCanvasHeightInput] = useState<string>(() => String(DEFAULT_CANVAS_HEIGHT));
  const [defaultEdgeWeight, setDefaultEdgeWeight] = useState<number>(DEFAULT_EDGE_WEIGHT);
  const [defaultEdgeWeightInput, setDefaultEdgeWeightInput] = useState<string>(String(DEFAULT_EDGE_WEIGHT));

  const [inputText, setInputText] = useState<Record<string, string>>(() => {
    const obj: Record<string, string> = {};
    // Give default spikes to the initial input neuron only
    obj["__placeholder__"] = ""; // will be reconciled below
    return obj;
  });

  const [sim, setSim] = useState<SimResult | null>(null);

  const [modules, setModules] = useState<Module[]>([]);
  const [activeModuleId, setActiveModuleId] = useState<string | null>(null);
  const [moduleNeurons, setModuleNeurons] = useState<Neuron[]>([]);
  const [moduleEdges, setModuleEdges] = useState<Edge[]>([]);
  const [moduleSelectedNodeIds, setModuleSelectedNodeIds] = useState<string[]>([]);
  const [moduleSelectedEdgeId, setModuleSelectedEdgeId] = useState<string | null>(null);
  const [moduleSelectedEdgeHandle, setModuleSelectedEdgeHandle] = useState<EdgeHandleSelection | null>(null);
  const [moduleMode, setModuleMode] = useState<"select" | "connect">("select");
  const [moduleConnectSrc, setModuleConnectSrc] = useState<string | null>(null);
  const [moduleConnectPoints, setModuleConnectPoints] = useState<Array<{ x: number; y: number }>>([]);
  const [moduleCanvasWidth, setModuleCanvasWidth] = useState<number>(DEFAULT_MACRO_WIDTH);
  const [moduleCanvasHeight, setModuleCanvasHeight] = useState<number>(DEFAULT_MACRO_HEIGHT);
  const [moduleInputIds, setModuleInputIds] = useState<string[]>([]);
  const [moduleOutputIds, setModuleOutputIds] = useState<string[]>([]);

  const [isAnimating, setIsAnimating] = useState<boolean>(false);
  const [highlightedNodeIds, setHighlightedNodeIds] = useState<string[]>([]);
  const [highlightedEdgeIds, setHighlightedEdgeIds] = useState<string[]>([]);
  const [recentNodeIds, setRecentNodeIds] = useState<string[]>([]);
  const [recentEdgeIds, setRecentEdgeIds] = useState<string[]>([]);
  const [animationFrames, setAnimationFrames] = useState<AnimationFrame[]>([]);
  const [animationIndex, setAnimationIndex] = useState<number>(-1);
  const [animationClock, setAnimationClock] = useState<number>(0);
  const animationFramesRef = useRef<AnimationFrame[]>([]);
  const animationIndexRef = useRef<number>(-1);
  const isAnimatingRef = useRef<boolean>(false);
  const playbackTimeoutRef = useRef<number | null>(null);
  const [hideStandardWeights, setHideStandardWeights] = useState<boolean>(false);
  const [touchMultiSelect, setTouchMultiSelect] = useState<boolean>(false);
  const [isCanvasFullScreen, setIsCanvasFullScreen] = useState<boolean>(false);
  const [showCleanDialog, setShowCleanDialog] = useState<boolean>(false);

  const copyBufferRef = useRef<{
    neurons: Neuron[];
    edges: Edge[];
    inputTexts: Record<string, string>;
    groups: Array<{
      nodeIds: string[];
      hue: number;
      label: string;
      labelVisible: boolean;
      compression: GroupCompressionState | null;
    }>;
  } | null>(null);
  const moduleCopyBufferRef = useRef<{
    neurons: Neuron[];
    edges: Edge[];
    inputIds: string[];
    outputIds: string[];
  } | null>(null);

  const canvasWidthString = useMemo(() => String(canvasWidth), [canvasWidth]);
  const canvasHeightString = useMemo(() => String(canvasHeight), [canvasHeight]);

  useEffect(() => {
    setCanvasWidthInput((prev) => (prev === canvasWidthString ? prev : canvasWidthString));
  }, [canvasWidthString]);

  useEffect(() => {
    setCanvasHeightInput((prev) => (prev === canvasHeightString ? prev : canvasHeightString));
  }, [canvasHeightString]);

  const applyCanvasSize = useCallback(() => {
    const rawWidth = Number(canvasWidthInput);
    const rawHeight = Number(canvasHeightInput);

    if (Number.isFinite(rawWidth)) {
      const nextWidth = Math.max(MIN_CANVAS_DIMENSION, rawWidth);
      if (nextWidth !== canvasWidth) {
        setCanvasWidth(nextWidth);
      }
      const nextWidthString = String(nextWidth);
      if (canvasWidthInput !== nextWidthString) {
        setCanvasWidthInput(nextWidthString);
      }
    } else if (canvasWidthInput !== canvasWidthString) {
      setCanvasWidthInput(canvasWidthString);
    }

    if (Number.isFinite(rawHeight)) {
      const nextHeight = Math.max(MIN_CANVAS_DIMENSION, rawHeight);
      if (nextHeight !== canvasHeight) {
        setCanvasHeight(nextHeight);
      }
      const nextHeightString = String(nextHeight);
      if (canvasHeightInput !== nextHeightString) {
        setCanvasHeightInput(nextHeightString);
      }
    } else if (canvasHeightInput !== canvasHeightString) {
      setCanvasHeightInput(canvasHeightString);
    }
  }, [
    canvasWidthInput,
    canvasHeightInput,
    canvasWidth,
    canvasHeight,
    canvasWidthString,
    canvasHeightString,
  ]);

  const handleCanvasSizeKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === "Enter") {
        event.preventDefault();
        applyCanvasSize();
      }
    },
    [applyCanvasSize]
  );

  const isCanvasSizeDirty = canvasWidthInput !== canvasWidthString || canvasHeightInput !== canvasHeightString;

  // Ensure inputText has entries for input neurons (and only them)
  useEffect(() => {
    setInputText((prev) => {
      const copy: Record<string, string> = {};
      for (const n of neurons) {
        if (n.role === "input") {
          copy[n.id] = prev[n.id] ?? "1 4 7";
        }
      }
      return copy;
    });
  }, [neurons]);

  useEffect(() => {
    function handleEscape(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      let consumed = false;
      if (showCleanDialog) {
        setShowCleanDialog(false);
        consumed = true;
      }
      if (!consumed && mode === "connect" && connectSrc) {
        cancelConnect();
        consumed = true;
      }
      if (!consumed && moduleMode === "connect" && moduleConnectSrc) {
        cancelModuleConnect();
        consumed = true;
      }
      if (consumed) {
        e.preventDefault();
        e.stopPropagation();
      }
    }
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [mode, connectSrc, moduleMode, moduleConnectSrc, cancelConnect, cancelModuleConnect, showCleanDialog]);

  useEffect(() => {
    setSelectedEdgeHandle((prev) => {
      if (!selectedEdgeId) return null;
      if (!prev || prev.edgeId !== selectedEdgeId) return null;
      const edge = edges.find((e) => e.id === selectedEdgeId);
      if (!edge) return null;
      const waypoints = sanitizeWaypoints(edge.points);
      if (!waypoints.length) return null;
      const nextIndex = Math.max(0, Math.min(waypoints.length - 1, prev.index));
      if (nextIndex !== prev.index) {
        return { edgeId: selectedEdgeId, index: nextIndex };
      }
      return prev;
    });
  }, [selectedEdgeId, edges]);

  useEffect(() => {
    setModuleSelectedEdgeHandle((prev) => {
      if (!moduleSelectedEdgeId) return null;
      if (!prev || prev.edgeId !== moduleSelectedEdgeId) return null;
      const edge = moduleEdges.find((e) => e.id === moduleSelectedEdgeId);
      if (!edge) return null;
      const waypoints = sanitizeWaypoints(edge.points);
      if (!waypoints.length) return null;
      const nextIndex = Math.max(0, Math.min(waypoints.length - 1, prev.index));
      if (nextIndex !== prev.index) {
        return { edgeId: moduleSelectedEdgeId, index: nextIndex };
      }
      return prev;
    });
  }, [moduleSelectedEdgeId, moduleEdges]);

  const h = useMemo(
    () => (hKind === "infty" ? Number.POSITIVE_INFINITY : hKind === "zero" ? 0 : Math.max(0, hVal)),
    [hKind, hVal]
  );

  const frameTotal = animationFrames.length;
  const frameIndexDisplay = animationIndex < 0 ? 0 : animationIndex + 1;
  const hasAnimation = frameTotal > 0;
  const canStepBackward = hasAnimation && animationIndex >= 0;
  const canStepForward = hasAnimation && animationIndex < frameTotal - 1;
  const canResume = !isAnimating && hasAnimation && animationIndex < frameTotal - 1;
  const freezeDisabled = isAnimating ? false : !canResume;
  const canJumpToEnd = hasAnimation && animationIndex !== frameTotal - 1;
  const animateButtonLabel = isAnimating ? "Freeze" : canResume ? "Resume" : "Animate";
  const animateButtonAction = isAnimating ? pauseAnimation : canResume ? resumeAnimation : animateSpikes;
  const animateButtonTitle = isAnimating
    ? "Pause spike propagation animation"
    : canResume
    ? "Resume spike propagation animation"
    : "Play spike propagation animation";
  const currentFrame = animationIndex >= 0 ? animationFrames[animationIndex] : null;
  const currentWaveDisplay = currentFrame ? currentFrame.wave + 1 : 0;
  const formattedClock = Number.isFinite(animationClock) ? animationClock.toFixed(2) : "0.00";

  const inputSpikeTrains = useMemo(() => {
    const obj: Record<string, number[]> = {};
    for (const [id, text] of Object.entries(inputText)) {
      obj[id] = parseTimes(text);
    }
    return obj;
  }, [inputText]);

  // Derived maps
  const byId = useMemo(() => Object.fromEntries(neurons.map((n) => [n.id, n])), [neurons]);
  const hiddenNeurons = useMemo(() => neurons.filter((n) => n.role === "hidden"), [neurons]);

  const activeModule = useMemo(() => modules.find((module) => module.id === activeModuleId) ?? null, [modules, activeModuleId]);
  const moduleById = useMemo(() => Object.fromEntries(moduleNeurons.map((n) => [n.id, n])), [moduleNeurons]);

  const selectionGroups = useMemo(() => {
    if (!selectedNodeIds.length) return [] as Group[];
    const sel = new Set(selectedNodeIds);
    return groups.filter((g) => g.nodeIds.some((id) => sel.has(id)));
  }, [groups, selectedNodeIds]);

  const canShowGroupLabel = selectionGroups.some((g) => !g.labelVisible);
  const canHideGroupLabel = selectionGroups.some((g) => g.labelVisible);
  const canToggleGroupLabel = selectionGroups.length > 0 && (canShowGroupLabel || canHideGroupLabel);
  const primaryGroup = selectionGroups.length === 1 ? selectionGroups[0] : null;
  const [groupLabelDraft, setGroupLabelDraft] = useState<string>("");
  const isPrimaryGroupCompressed = Boolean(primaryGroup?.compression);

  type CanvasSnapshot = {
    neurons: Neuron[];
    edges: Edge[];
    groups: Group[];
    panels: Array<{ id: string; neuronId: string; x: number; y: number }>;
  };
  const undoStackRef = useRef<CanvasSnapshot[]>([]);
  const isRestoringRef = useRef<boolean>(false);
  const [canUndo, setCanUndo] = useState<boolean>(false);
  const nodeDragSnapshotRef = useRef(false);
  const labelDragSnapshotRef = useRef(false);
  const edgeDragSnapshotRef = useRef(false);

  const [dynamicsPanels, setDynamicsPanels] = useState<
    Array<{
      id: string;
      neuronId: string;
      x: number;
      y: number;
    }>
  >([]);
  const [showDynamicsHeaders, setShowDynamicsHeaders] = useState<boolean>(true);
  const [showOutputPotentials, setShowOutputPotentials] = useState<boolean>(false);
  const [includeColors, setIncludeColors] = useState<boolean>(true);
  const [includeAnimationCounterInPdf, setIncludeAnimationCounterInPdf] = useState<boolean>(false);
  const [isExportingAnimationPdfs, setIsExportingAnimationPdfs] = useState<boolean>(false);

  const openDynamicsPanelForNeuron = useCallback(
    (neuronId: string) => {
      const neuron = byId[neuronId];
      if (!neuron) return;
      setDynamicsPanels((prev) => {
        if (prev.some((panel) => panel.neuronId === neuronId)) return prev;
        return [
          ...prev,
          {
            id: uid("dyn"),
            neuronId,
            x: neuron.x + 32,
            y: neuron.y - 32,
          },
        ];
      });
    },
    [byId]
  );

  function cloneSnapshot(): CanvasSnapshot {
    const clonedNeurons = neurons.map((n) => ({ ...n }));
    const clonedEdges = edges.map((e) => ({
      ...e,
      points: e.points ? e.points.map((pt) => ({ ...pt })) : undefined,
    }));
    const clonedGroups = groups.map((g) => ({
      ...g,
      nodeIds: [...g.nodeIds],
      compression: g.compression
        ? {
            center: { ...g.compression.center },
            nodes: Object.fromEntries(
              Object.entries(g.compression.nodes).map(([id, data]) => [id, { ...data }])
            ),
            edgePoints: Object.fromEntries(
              Object.entries(g.compression.edgePoints ?? {}).map(([edgeId, points]) => [
                edgeId,
                points.map((pt) => ({ ...pt })),
              ])
            ),
            labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
          }
        : undefined,
    }));
    const clonedPanels = dynamicsPanels.map((p) => ({ ...p }));
    return { neurons: clonedNeurons, edges: clonedEdges, groups: clonedGroups, panels: clonedPanels };
  }

  function recordSnapshot() {
    if (isRestoringRef.current) return;
    const snapshot = cloneSnapshot();
    undoStackRef.current.push(snapshot);
    if (undoStackRef.current.length > 100) undoStackRef.current.shift();
    setCanUndo(true);
  }

  function restoreSnapshot(snapshot: CanvasSnapshot) {
    isRestoringRef.current = true;
    setNeurons(snapshot.neurons.map((n) => ({ ...n })));
    setEdges(
      snapshot.edges.map((e) => ({
        ...e,
        points: e.points ? e.points.map((pt) => ({ ...pt })) : undefined,
      }))
    );
    setGroups(
      snapshot.groups.map((g) => ({
        ...g,
        nodeIds: [...g.nodeIds],
        compression: g.compression
          ? {
              center: { ...g.compression.center },
              nodes: Object.fromEntries(
                Object.entries(g.compression.nodes).map(([id, data]) => [id, { ...data }])
              ),
              edgePoints: Object.fromEntries(
                Object.entries(g.compression.edgePoints ?? {}).map(([edgeId, points]) => [
                  edgeId,
                  points.map((pt) => ({ ...pt })),
                ])
              ),
              labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
            }
          : undefined,
      }))
    );
    setDynamicsPanels(snapshot.panels.map((p) => ({ ...p })));
    setSelectedNodeIds([]);
    setSelectedEdgeId(null);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
    setConnectSrc(null);
    setConnectPoints([]);
    setSim(null);
    isRestoringRef.current = false;
    nodeDragSnapshotRef.current = false;
    labelDragSnapshotRef.current = false;
    edgeDragSnapshotRef.current = false;
  }

  function undoCanvas(): boolean {
    const stack = undoStackRef.current;
    if (!stack.length) return false;
    const snapshot = stack.pop()!;
    restoreSnapshot(snapshot);
    setCanUndo(stack.length > 0);
    return true;
  }

  function beginNodeDrag() {
    nodeDragSnapshotRef.current = false;
    labelDragSnapshotRef.current = false;
    edgeDragSnapshotRef.current = false;
  }

  function ensureNodeDragSnapshot() {
    if (isRestoringRef.current) return;
    if (!nodeDragSnapshotRef.current) {
      recordSnapshot();
      nodeDragSnapshotRef.current = true;
    }
  }

  function beginLabelDrag() {
    nodeDragSnapshotRef.current = false;
    labelDragSnapshotRef.current = false;
    edgeDragSnapshotRef.current = false;
  }

  function ensureLabelDragSnapshot() {
    if (isRestoringRef.current) return;
    if (!labelDragSnapshotRef.current) {
      recordSnapshot();
      labelDragSnapshotRef.current = true;
    }
  }

  function beginEdgeDrag() {
    nodeDragSnapshotRef.current = false;
    labelDragSnapshotRef.current = false;
    edgeDragSnapshotRef.current = false;
  }

  function ensureEdgeDragSnapshot() {
    if (isRestoringRef.current) return;
    if (!edgeDragSnapshotRef.current) {
      recordSnapshot();
      edgeDragSnapshotRef.current = true;
    }
  }

  function endDrag() {
    nodeDragSnapshotRef.current = false;
    labelDragSnapshotRef.current = false;
    edgeDragSnapshotRef.current = false;
  }

  useEffect(() => {
    setGroups((prev) => {
      if (!prev.length) return prev;
      let mutated = false;
      const taken = new Set<string>();
      const next: Group[] = [];
      for (const g of prev) {
        const nodeIds = Array.isArray(g.nodeIds) ? g.nodeIds : [];
        if (nodeIds.length < 2) {
          mutated = true;
          continue;
        }
        let label = typeof g.label === "string" ? g.label.trim() : "";
        if (!label) {
          mutated = true;
          label = `Group ${taken.size + 1}`;
        }
        if (taken.has(label)) {
          mutated = true;
          const base = label;
          let counter = 2;
          let attempt = `${base} ${counter}`;
          while (taken.has(attempt)) {
            counter += 1;
            attempt = `${base} ${counter}`;
          }
          label = attempt;
        }
        taken.add(label);
        const labelVisible = g.labelVisible === true;
        next.push({
          ...g,
          nodeIds,
          label,
          labelVisible,
          compression: g.compression
            ? {
                center: { ...g.compression.center },
                nodes: Object.fromEntries(
                  Object.entries(g.compression.nodes).map(([id, data]) => [id, { ...data }])
                ),
                edgePoints: Object.fromEntries(
                  Object.entries(g.compression.edgePoints ?? {}).map(([edgeId, points]) => [
                    edgeId,
                    points.map((pt) => ({ ...pt })),
                  ])
                ),
                labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
              }
            : undefined,
        });
      }
      return mutated ? next : prev;
    });
  }, []);

  useEffect(() => {
    if (primaryGroup) setGroupLabelDraft(primaryGroup.label ?? "");
    else setGroupLabelDraft("");
  }, [primaryGroup?.id, primaryGroup?.label]);

  function openModuleEditor(module: Module | null) {
    if (!module) {
      setActiveModuleId(null);
      setModuleNeurons([]);
      setModuleEdges([]);
      setModuleSelectedNodeIds([]);
      setModuleSelectedEdgeId(null);
      setModuleMode("select");
      setModuleConnectSrc(null);
      setModuleConnectPoints([]);
      setModuleCanvasWidth(DEFAULT_MACRO_WIDTH);
      setModuleCanvasHeight(DEFAULT_MACRO_HEIGHT);
      setModuleInputIds([]);
      setModuleOutputIds([]);
      return;
    }
    setActiveModuleId(module.id);
    setModuleNeurons(module.neurons.map((n) => ({ ...n })));
    setModuleEdges(module.edges.map((e) => ({ ...e })));
    setModuleSelectedNodeIds([]);
    setModuleSelectedEdgeId(null);
    setModuleMode("select");
    setModuleConnectSrc(null);
    setModuleConnectPoints([]);
    setModuleCanvasWidth(module.canvas?.width ?? DEFAULT_MACRO_WIDTH);
    setModuleCanvasHeight(module.canvas?.height ?? DEFAULT_MACRO_HEIGHT);
    setModuleInputIds([...(module.inputNeuronIds || [])]);
    setModuleOutputIds([...(module.outputNeuronIds || [])]);
  }

  // Auto-save module edits back into modules list while editing
  useEffect(() => {
    if (!activeModuleId) return;
    // Keep input/output ids valid to current module node set
    const nodeIdSet = new Set(moduleNeurons.map((n) => n.id));
    const filteredInputs = moduleInputIds.filter((id) => nodeIdSet.has(id));
    const filteredOutputs = moduleOutputIds.filter((id) => nodeIdSet.has(id));
    setModules((prev) =>
      prev.map((m) =>
        m.id === activeModuleId
          ? {
              ...m,
              neurons: moduleNeurons,
              edges: moduleEdges,
              inputNeuronIds: filteredInputs,
              outputNeuronIds: filteredOutputs,
              canvas: { width: moduleCanvasWidth, height: moduleCanvasHeight },
            }
          : m
      )
    );
  }, [
    activeModuleId,
    moduleNeurons,
    moduleEdges,
    moduleInputIds,
    moduleOutputIds,
    moduleCanvasWidth,
    moduleCanvasHeight,
  ]);

  function createModule() {
    const id = uid("module");
    const module: Module = {
      id,
      name: `Module ${modules.length + 1}`,
      neurons: [],
      edges: [],
      inputNeuronIds: [],
      outputNeuronIds: [],
      canvas: { width: DEFAULT_MACRO_WIDTH, height: DEFAULT_MACRO_HEIGHT },
    };
    setModules((prev) => [...prev, module]);
    openModuleEditor(module);
  }

  function deleteModuleById(id: string) {
    setModules((prev) => prev.filter((module) => module.id !== id));
    if (activeModuleId === id) {
      openModuleEditor(null);
    }
  }

  function renameModule(id: string, name: string) {
    const safeName = name.trim() || "Module";
    setModules((prev) => prev.map((module) => (module.id === id ? { ...module, name: safeName } : module)));
  }

  function instantiateModule(module: Module) {
    if (!module.neurons.length) {
      alert("This module does not contain any neurons yet.");
      return;
    }
    if (!isRestoringRef.current) recordSnapshot();
    stopAnimation();
    const takenLabels = new Set(neurons.map((n) => n.label));
    const minX = Math.min(...module.neurons.map((n) => n.x));
    const maxX = Math.max(...module.neurons.map((n) => n.x));
    const minY = Math.min(...module.neurons.map((n) => n.y));
    const maxY = Math.max(...module.neurons.map((n) => n.y));
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const targetCenterX = canvasWidth / 2;
    const targetCenterY = canvasHeight / 2;
    const offsetX = Number.isFinite(centerX) ? targetCenterX - centerX : 0;
    const offsetY = Number.isFinite(centerY) ? targetCenterY - centerY : 0;
    const boardMinX = NODE_RADIUS;
    const boardMinY = NODE_RADIUS;
    const boardMaxX = Math.max(NODE_RADIUS, canvasWidth - NODE_RADIUS);
    const boardMaxY = Math.max(NODE_RADIUS, canvasHeight - NODE_RADIUS);

    const idMap = new Map<string, string>();
    const newNeurons = module.neurons.map((node) => {
      const newId = uid("n");
      idMap.set(node.id, newId);
      const baseLabel = module.name ? `${module.name}-${node.label}` : node.label;
      const label = uniqueLabel(baseLabel, takenLabels);
      const rawX = clamp(node.x + offsetX, boardMinX, boardMaxX);
      const rawY = clamp(node.y + offsetY, boardMinY, boardMaxY);
      const x = clamp(snapToGrid(rawX), boardMinX, boardMaxX);
      const y = clamp(snapToGrid(rawY), boardMinY, boardMaxY);
      const labelOffsetX = Number.isFinite(node.labelOffsetX) ? (node.labelOffsetX as number) : 0;
      const labelOffsetY = Number.isFinite(node.labelOffsetY) ? (node.labelOffsetY as number) : DEFAULT_LABEL_OFFSET_Y;
      return {
        id: newId,
        label,
        role: "hidden" as Role,
        x,
        y,
        labelVisible: false,
        labelOffsetX,
        labelOffsetY,
      };
    });

    const newEdges = module.edges
      .map((edge) => {
        const sourceId = idMap.get(edge.sourceId);
        const targetId = idMap.get(edge.targetId);
        if (!sourceId || !targetId) return null;
        const transformedPoints = Array.isArray(edge.points)
          ? edge.points
              .map((pt) => {
                if (!pt || typeof pt !== "object") return null;
                const px = clamp(snapToGrid(clamp(pt.x + offsetX, boardMinX, boardMaxX)), boardMinX, boardMaxX);
                const py = clamp(snapToGrid(clamp(pt.y + offsetY, boardMinY, boardMaxY)), boardMinY, boardMaxY);
                return { x: px, y: py };
              })
              .filter((p): p is { x: number; y: number } => p !== null)
          : undefined;
        const points = transformedPoints && transformedPoints.length ? transformedPoints : undefined;
        return { id: uid("e"), sourceId, targetId, weight: edge.weight, points } as Edge;
      })
      .filter((edge): edge is Edge => !!edge);

    setNeurons((prev) => [...prev, ...newNeurons]);
    setEdges((prev) => [...prev, ...newEdges]);

    const newNodeIds = newNeurons.map((n) => n.id);
    if (newNodeIds.length) {
      setSelectedNodeIds(newNodeIds);
      setSelectedEdgeId(null);
      setSelectedEdgeHandle(null);
      setActiveSelectionContext("main");

      setGroups((prev) => {
        const nextHue = computeGroupHue(prev);
        const baseLabel = (module.name ?? "").trim();
        const label = nextGroupLabel(prev, baseLabel || "Module Group");
        const group: Group = { id: uid("grp"), nodeIds: newNodeIds, hue: nextHue, label, labelVisible: false };
        return [...prev, group];
      });
    }
  }

  function saveGroupAsModule(group: Group) {
    const nodeSet = new Set(group.nodeIds);
    const groupNeurons = neurons
      .filter((n) => nodeSet.has(n.id))
      .map((n) => {
        const stored = group.compression?.nodes?.[n.id];
        return {
          ...n,
          x: stored ? stored.x : n.x,
          y: stored ? stored.y : n.y,
          labelVisible: stored ? stored.labelVisible : n.labelVisible === true,
          labelOffsetX: stored
            ? stored.labelOffsetX
            : Number.isFinite(n.labelOffsetX)
            ? (n.labelOffsetX as number)
            : 0,
          labelOffsetY: stored
            ? stored.labelOffsetY
            : Number.isFinite(n.labelOffsetY)
            ? (n.labelOffsetY as number)
            : DEFAULT_LABEL_OFFSET_Y,
        } as Neuron;
      });
    if (!groupNeurons.length) return;

    const groupEdges = edges.filter((e) => nodeSet.has(e.sourceId) && nodeSet.has(e.targetId));
    const minX = Math.min(...groupNeurons.map((n) => n.x));
    const maxX = Math.max(...groupNeurons.map((n) => n.x));
    const minY = Math.min(...groupNeurons.map((n) => n.y));
    const maxY = Math.max(...groupNeurons.map((n) => n.y));
    const spanX = maxX - minX || 0;
    const spanY = maxY - minY || 0;
    const padding = 60;
    const canvasWidth = Math.max(DEFAULT_MACRO_WIDTH, spanX + padding * 2);
    const canvasHeight = Math.max(DEFAULT_MACRO_HEIGHT, spanY + padding * 2);

    setModules((prev) => {
      const takenNames = new Set(prev.map((m) => m.name));
      const baseName = (group.label ?? "").trim() || "Group Module";
      let name = baseName;
      let counter = 2;
      while (takenNames.has(name)) {
        name = `${baseName} ${counter++}`;
      }

      const nodeIdMap = new Map<string, string>();
      const moduleNeurons = groupNeurons.map((n) => {
        const newId = uid("mn");
        nodeIdMap.set(n.id, newId);
        return {
          id: newId,
          label: n.label,
          role: n.role,
          x: n.x - minX + padding,
          y: n.y - minY + padding,
          labelVisible: n.labelVisible === true,
          labelOffsetX: n.labelOffsetX,
          labelOffsetY: n.labelOffsetY,
        } as Neuron;
      });

      const moduleEdges = groupEdges
        .map((edge) => {
          const sourceId = nodeIdMap.get(edge.sourceId);
          const targetId = nodeIdMap.get(edge.targetId);
          if (!sourceId || !targetId) return null;
          const waypoints = sanitizeWaypoints(edge.points).map((pt) => ({
            x: pt.x - minX + padding,
            y: pt.y - minY + padding,
          }));
          const points = waypoints.length ? waypoints : undefined;
          return { id: uid("me"), sourceId, targetId, weight: edge.weight, points } as Edge;
        })
        .filter((edge): edge is Edge => edge !== null);

      const inputNeuronIds = moduleNeurons
        .filter((n) => n.role === "input")
        .map((n) => n.id);
      const outputNeuronIds = moduleNeurons
        .filter((n) => n.role === "output")
        .map((n) => n.id);

      const module: Module = {
        id: uid("module"),
        name,
        neurons: moduleNeurons,
        edges: moduleEdges,
        inputNeuronIds,
        outputNeuronIds,
        canvas: { width: canvasWidth, height: canvasHeight },
      };

      return [...prev, module];
    });
  }

  function addModuleNeuron(role: Role = "hidden") {
    if (!activeModuleId) return;
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, moduleCanvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, moduleCanvasHeight - NODE_RADIUS);
    setModuleNeurons((ns) => {
      const id = uid("n");
      const label = `M${ns.length + 1}`;
      const proposedX = 100 + (ns.length % 6) * 60;
      const proposedY = 80 + ((ns.length * 70) % Math.max(140, moduleCanvasHeight - 120));
      return [
        ...ns,
        {
          id,
          label,
          role,
          x: clamp(snapToGrid(clamp(proposedX, minX, maxX)), minX, maxX),
          y: clamp(snapToGrid(clamp(proposedY, minY, maxY)), minY, maxY),
          labelVisible: false,
          labelOffsetX: 0,
          labelOffsetY: DEFAULT_LABEL_OFFSET_Y,
        },
      ];
    });
  }

  function deleteModuleSelection(): boolean {
    if (!moduleSelectedEdgeId && moduleSelectedNodeIds.length === 0) return false;
    setModuleEdges((es) =>
      es.filter((e) => {
        if (moduleSelectedEdgeId && e.id === moduleSelectedEdgeId) return false;
        if (moduleSelectedNodeIds.includes(e.sourceId) || moduleSelectedNodeIds.includes(e.targetId)) return false;
        return true;
      })
    );
    if (moduleSelectedNodeIds.length) {
      setModuleNeurons((ns) => ns.filter((n) => !moduleSelectedNodeIds.includes(n.id)));
      setModuleInputIds((ids) => ids.filter((id) => !moduleSelectedNodeIds.includes(id)));
      setModuleOutputIds((ids) => ids.filter((id) => !moduleSelectedNodeIds.includes(id)));
    }
    setModuleSelectedEdgeId(null);
    setModuleSelectedNodeIds([]);
    setModuleSelectedEdgeHandle(null);
    setActiveSelectionContext("module");
    return true;
  }

  function beginModuleConnect(id: string) {
    if (moduleMode !== "connect") return;
    setModuleConnectSrc(id);
    setModuleConnectPoints([]);
  }

  function completeModuleConnect(id: string) {
    if (moduleMode !== "connect" || !moduleConnectSrc) return;
    if (moduleConnectSrc === id) {
      setModuleConnectSrc(null);
      setModuleConnectPoints([]);
      return;
    }
    const src = moduleConnectSrc;
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, moduleCanvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, moduleCanvasHeight - NODE_RADIUS);
    let createdEdgeId: string | null = null;
    setModuleEdges((es) => {
      if (es.some((e) => e.sourceId === src && e.targetId === id)) return es;
      const waypointList = moduleConnectPoints.map((pt) => ({
        x: clamp(snapToGrid(pt.x), minX, maxX),
        y: clamp(snapToGrid(pt.y), minY, maxY),
      }));
      const points = waypointList.length ? waypointList : undefined;
      const edge: Edge = { id: uid("e"), sourceId: src, targetId: id, weight: defaultEdgeWeight, points };
      createdEdgeId = edge.id;
      return [...es, edge];
    });
    setModuleConnectSrc(null);
    setModuleConnectPoints([]);
    if (createdEdgeId) {
      setModuleSelectedNodeIds([]);
      setModuleSelectedEdgeId(createdEdgeId);
      setModuleSelectedEdgeHandle(null);
      setActiveSelectionContext("module");
    }
  }

  function addModuleWaypoint(pt: { x: number; y: number }) {
    if (moduleMode !== "connect" || !moduleConnectSrc) return;
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, moduleCanvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, moduleCanvasHeight - NODE_RADIUS);
    const x = clamp(snapToGrid(pt.x), minX, maxX);
    const y = clamp(snapToGrid(pt.y), minY, maxY);
    setModuleConnectPoints((prev) => [...prev, { x, y }]);
  }

  function cancelModuleConnect() {
    setModuleConnectSrc(null);
    setModuleConnectPoints([]);
  }

  function moveModuleNodes(updates: Array<{ id: string; x: number; y: number }>) {
    if (!updates.length) return;
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, moduleCanvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, moduleCanvasHeight - NODE_RADIUS);
    const map = new Map(
      updates.map((u) => [
        u.id,
        {
          id: u.id,
          x: clamp(snapToGrid(clamp(u.x, minX, maxX)), minX, maxX),
          y: clamp(snapToGrid(clamp(u.y, minY, maxY)), minY, maxY),
        },
      ])
    );
    setModuleNeurons((ns) => ns.map((n) => (map.has(n.id) ? { ...n, ...map.get(n.id)! } : n)));
  }

  function moveModuleLabelOffsets(updates: Array<{ id: string; labelOffsetX: number; labelOffsetY: number }>) {
    if (!updates.length) return;
    const map = new Map(updates.map((u) => [u.id, u]));
    setModuleNeurons((ns) => ns.map((n) => (map.has(n.id) ? { ...n, ...map.get(n.id)! } : n)));
  }

  function setModuleLabelVisibility(id: string, visible: boolean) {
    setModuleNeurons((ns) => ns.map((n) => (n.id === id ? { ...n, labelVisible: visible } : n)));
  }

  function setModuleLabelOffset(id: string, offsetX: number, offsetY: number) {
    setModuleNeurons((ns) =>
      ns.map((n) => (n.id === id ? { ...n, labelOffsetX: offsetX, labelOffsetY: offsetY } : n))
    );
  }

  function resetModuleLabelOffset(id: string) {
    setModuleNeurons((ns) =>
      ns.map((n) =>
        n.id === id ? { ...n, labelOffsetX: 0, labelOffsetY: DEFAULT_LABEL_OFFSET_Y } : n
      )
    );
  }

  function updateModuleEdgeWaypoint(edgeId: string, index: number, point: Point) {
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, moduleCanvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, moduleCanvasHeight - NODE_RADIUS);
    const x = clamp(snapToGrid(point.x), minX, maxX);
    const y = clamp(snapToGrid(point.y), minY, maxY);
    setModuleEdges((es) =>
      es.map((edge) => {
        if (edge.id !== edgeId) return edge;
        const waypoints = sanitizeWaypoints(edge.points);
        if (index < 0 || index >= waypoints.length) return edge;
        const next = [...waypoints];
        next[index] = { x, y };
        return { ...edge, points: next };
      })
    );
    setActiveSelectionContext("module");
  }

  function insertModuleEdgeWaypoint(edgeId: string, point: Point, index: number) {
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, moduleCanvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, moduleCanvasHeight - NODE_RADIUS);
    const x = clamp(snapToGrid(point.x), minX, maxX);
    const y = clamp(snapToGrid(point.y), minY, maxY);
    let insertedIndex: number | null = null;
    setModuleEdges((es) =>
      es.map((edge) => {
        if (edge.id !== edgeId) return edge;
        const waypoints = sanitizeWaypoints(edge.points);
        const next = [...waypoints];
        const safeIndex = Math.max(0, Math.min(index, next.length));
        next.splice(safeIndex, 0, { x, y });
        insertedIndex = safeIndex;
        return { ...edge, points: next };
      })
    );
    if (insertedIndex !== null) {
      setModuleSelectedEdgeId(edgeId);
      setModuleSelectedEdgeHandle({ edgeId, index: insertedIndex });
      setActiveSelectionContext("module");
    }
  }

  function removeModuleEdgeWaypoint(edgeId: string, index: number): boolean {
    const edge = moduleEdges.find((e) => e.id === edgeId);
    if (!edge) return false;
    const waypoints = sanitizeWaypoints(edge.points);
    if (index < 0 || index >= waypoints.length) return false;
    setModuleEdges((es) =>
      es.map((e) => {
        if (e.id !== edgeId) return e;
        const next = [...waypoints];
        next.splice(index, 1);
        return { ...e, points: next.length ? next : undefined };
      })
    );
    const nextLength = waypoints.length - 1;
    if (nextLength > 0) {
      const nextIndex = Math.min(index, nextLength - 1);
      setModuleSelectedEdgeHandle({ edgeId, index: nextIndex });
    } else {
      setModuleSelectedEdgeHandle(null);
    }
    setActiveSelectionContext("module");
    return true;
  }

  function removeModuleSelectedEdgeWaypoint(): boolean {
    if (!moduleSelectedEdgeHandle || !moduleSelectedEdgeId) return false;
    if (moduleSelectedEdgeHandle.edgeId !== moduleSelectedEdgeId) return false;
    return removeModuleEdgeWaypoint(moduleSelectedEdgeHandle.edgeId, moduleSelectedEdgeHandle.index);
  }

  function copyModuleSelection(): boolean {
    if (!moduleSelectedNodeIds.length) return false;
    const selectedSet = new Set(moduleSelectedNodeIds);
    const nodes = moduleNeurons
      .filter((n) => selectedSet.has(n.id))
      .map((n) => ({
        ...n,
        labelOffsetX: Number.isFinite(n.labelOffsetX) ? (n.labelOffsetX as number) : 0,
        labelOffsetY: Number.isFinite(n.labelOffsetY) ? (n.labelOffsetY as number) : DEFAULT_LABEL_OFFSET_Y,
      }));
    if (!nodes.length) return false;
    const copiedEdges = moduleEdges
      .filter((e) => selectedSet.has(e.sourceId) && selectedSet.has(e.targetId))
      .map((e) => ({ ...e, points: sanitizeWaypoints(e.points) }));
    const inputs = moduleInputIds.filter((id) => selectedSet.has(id));
    const outputs = moduleOutputIds.filter((id) => selectedSet.has(id));
    moduleCopyBufferRef.current = {
      neurons: nodes,
      edges: copiedEdges,
      inputIds: inputs,
      outputIds: outputs,
    };
    return true;
  }

  function pasteModuleSelection(): boolean {
    const buffer = moduleCopyBufferRef.current;
    if (!buffer || !buffer.neurons.length) return false;
    const offsetX = GRID_SIZE * 3;
    const offsetY = GRID_SIZE * 3;
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, moduleCanvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, moduleCanvasHeight - NODE_RADIUS);
    const takenLabels = new Set(moduleNeurons.map((n) => n.label));
    const idMap = new Map<string, string>();
    const newNeurons = buffer.neurons.map((node) => {
      const newId = uid("n");
      idMap.set(node.id, newId);
      const label = uniqueLabel(node.label, takenLabels);
      const x = clamp(snapToGrid(node.x + offsetX), minX, maxX);
      const y = clamp(snapToGrid(node.y + offsetY), minY, maxY);
      const labelVisible = node.labelVisible === true;
      const labelOffsetX = Number.isFinite(node.labelOffsetX) ? (node.labelOffsetX as number) : 0;
      const labelOffsetY = Number.isFinite(node.labelOffsetY)
        ? (node.labelOffsetY as number)
        : DEFAULT_LABEL_OFFSET_Y;
      return {
        id: newId,
        label,
        role: node.role,
        x,
        y,
        labelVisible,
        labelOffsetX,
        labelOffsetY,
      } as Neuron;
    });
    if (!newNeurons.length) return false;
    const edgeIdMap = new Map<string, string>();
    const newEdges = buffer.edges
      .map((edge) => {
        const sourceId = idMap.get(edge.sourceId);
        const targetId = idMap.get(edge.targetId);
        if (!sourceId || !targetId) return null;
        const transformed = sanitizeWaypoints(edge.points).map((pt) => ({
          x: clamp(snapToGrid(pt.x + offsetX), minX, maxX),
          y: clamp(snapToGrid(pt.y + offsetY), minY, maxY),
        }));
        const points = transformed.length ? transformed : undefined;
        const newEdgeId = uid("e");
        edgeIdMap.set(edge.id, newEdgeId);
        return { id: newEdgeId, sourceId, targetId, weight: edge.weight, points } as Edge;
      })
      .filter((edge): edge is Edge => !!edge);
    setModuleNeurons((prev) => [...prev, ...newNeurons]);
    if (newEdges.length) {
      setModuleEdges((prev) => [...prev, ...newEdges]);
    }
    const newIds = newNeurons.map((n) => n.id);
    setModuleSelectedNodeIds(newIds);
    setModuleSelectedEdgeId(null);
    setModuleSelectedEdgeHandle(null);
    setActiveSelectionContext("module");
    const newInputIds = buffer.inputIds
      .map((id) => idMap.get(id))
      .filter((id): id is string => !!id);
    if (newInputIds.length) {
      setModuleInputIds((prev) => Array.from(new Set([...prev, ...newInputIds])));
    }
    const newOutputIds = buffer.outputIds
      .map((id) => idMap.get(id))
      .filter((id): id is string => !!id);
    if (newOutputIds.length) {
      setModuleOutputIds((prev) => Array.from(new Set([...prev, ...newOutputIds])));
    }
    return true;
  }

  function handleModuleNodePointerDown(
    id: string,
    modifiers: { append: boolean; toggle: boolean }
  ): string[] {
    let nextSelection: string[] = [];
    setModuleSelectedNodeIds((prev) => {
      let next: string[];
      if (modifiers.toggle) {
        if (prev.includes(id)) next = prev.filter((nid) => nid !== id);
        else next = [...prev, id];
      } else if (modifiers.append) {
        if (prev.includes(id)) next = prev.filter((nid) => nid !== id);
        else next = [...prev, id];
      } else {
        next = [id];
      }
      nextSelection = next;
      return next;
    });
    setModuleSelectedEdgeId(null);
    setModuleSelectedEdgeHandle(null);
    setActiveSelectionContext("module");
    return nextSelection;
  }

  function handleModuleEdgePointerDown(id: string) {
    setModuleSelectedNodeIds([]);
    setModuleSelectedEdgeId(id);
    setModuleSelectedEdgeHandle(null);
    setActiveSelectionContext("module");
  }

  function clearModuleSelection() {
    setModuleSelectedNodeIds([]);
    setModuleSelectedEdgeId(null);
    setModuleSelectedEdgeHandle(null);
    setActiveSelectionContext("module");
  }

  function setModuleRole(id: string, role: Role) {
    setModuleNeurons((ns) => ns.map((n) => (n.id === id ? { ...n, role } : n)));
  }

  function setModuleLabel(id: string, label: string) {
    setModuleNeurons((ns) => ns.map((n) => (n.id === id ? { ...n, label } : n)));
  }

  function setModuleWeight(edgeId: string, w: number) {
    setModuleEdges((es) => es.map((e) => (e.id === edgeId ? { ...e, weight: w } : e)));
  }

  function removeModuleEdge(edgeId: string) {
    setModuleEdges((es) => es.filter((e) => e.id !== edgeId));
    setModuleSelectedEdgeId((prev) => (prev === edgeId ? null : prev));
    setModuleSelectedEdgeHandle((prev) => (prev && prev.edgeId === edgeId ? null : prev));
  }

  function setModuleInput(id: string, flag: boolean) {
    setModuleInputIds((prev) => {
      if (flag) {
        if (prev.includes(id)) return prev;
        return [...prev, id];
      }
      return prev.filter((nid) => nid !== id);
    });
  }

  function setModuleOutput(id: string, flag: boolean) {
    setModuleOutputIds((prev) => {
      if (flag) {
        if (prev.includes(id)) return prev;
        return [...prev, id];
      }
      return prev.filter((nid) => nid !== id);
    });
  }

  function clearPlaybackTimer() {
    if (playbackTimeoutRef.current !== null) {
      window.clearTimeout(playbackTimeoutRef.current);
      playbackTimeoutRef.current = null;
    }
  }

  function setAnimationPlaying(playing: boolean) {
    isAnimatingRef.current = playing;
    setIsAnimating(playing);
  }

  function updateAnimationFramesState(frames: AnimationFrame[]) {
    animationFramesRef.current = frames;
    setAnimationFrames(frames);
  }

  function updateAnimationIndexState(index: number) {
    animationIndexRef.current = index;
    setAnimationIndex(index);
  }

  function clearHighlights() {
    setHighlightedNodeIds([]);
    setHighlightedEdgeIds([]);
    setRecentNodeIds([]);
    setRecentEdgeIds([]);
  }

  function displayFrame(index: number) {
    const frames = animationFramesRef.current;
    const frame = frames[index];
    if (!frame) return;
    updateAnimationIndexState(index);
    setAnimationClock(frame.time);
    const EPS = 1e-9;
    let start = index;
    while (start > 0 && Math.abs(frames[start - 1].time - frame.time) <= EPS) {
      start -= 1;
    }
    const persistentNodes = new Set<string>();
    const persistentEdges = new Set<string>();
    for (let i = start; i < index; i++) {
      const prior = frames[i];
      prior.sourceIds.forEach((id) => persistentNodes.add(id));
      prior.targetIds.forEach((id) => persistentNodes.add(id));
      prior.edgeIds.forEach((id) => persistentEdges.add(id));
    }
    frame.sourceIds.forEach((id) => persistentNodes.add(id));
    const newNodes = frame.targetIds.filter((id) => !persistentNodes.has(id));
    const newEdges = frame.edgeIds.filter((id) => !persistentEdges.has(id));
    frame.targetIds.forEach((id) => persistentNodes.add(id));
    frame.edgeIds.forEach((id) => persistentEdges.add(id));
    setHighlightedNodeIds(Array.from(persistentNodes));
    setHighlightedEdgeIds(Array.from(persistentEdges));
    setRecentNodeIds(newNodes);
    setRecentEdgeIds(newEdges);
  }

  function displayNoFrame() {
    updateAnimationIndexState(-1);
    setAnimationClock(0);
    clearHighlights();
  }

  function pauseAnimation() {
    clearPlaybackTimer();
    setAnimationPlaying(false);
  }

  function resumeAnimation() {
    if (isAnimatingRef.current) return;
    const frames = animationFramesRef.current;
    if (!frames.length) return;
    const currentIndex = animationIndexRef.current;
    if (currentIndex >= frames.length - 1) return;
    setAnimationPlaying(true);
    scheduleNextFrame(currentIndex);
  }

  function scheduleNextFrame(fromIndex: number) {
    if (!isAnimatingRef.current) return;
    const frames = animationFramesRef.current;
    const nextIndex = fromIndex + 1;
    if (nextIndex >= frames.length) {
      setAnimationPlaying(false);
      return;
    }
    const prevTime = fromIndex >= 0 ? frames[fromIndex].time : frames[nextIndex].time;
    const nextTime = frames[nextIndex].time;
    const delta = fromIndex >= 0 ? Math.max(0, nextTime - prevTime) : 0;
    const delay = fromIndex < 0 ? MIN_EVENT_DELAY_MS : Math.max(MIN_EVENT_DELAY_MS, delta * TIME_SCALE_MS);
    playbackTimeoutRef.current = window.setTimeout(() => {
      playbackTimeoutRef.current = null;
      if (!isAnimatingRef.current) return;
      displayFrame(nextIndex);
      if (isAnimatingRef.current) {
        scheduleNextFrame(nextIndex);
      }
    }, delay);
  }

  function stopAnimation() {
    pauseAnimation();
    displayNoFrame();
    updateAnimationFramesState([]);
  }

  function stepForward() {
    const frames = animationFramesRef.current;
    if (!frames.length) return;
    pauseAnimation();
    const nextIndex = Math.min(frames.length - 1, animationIndexRef.current + 1);
    if (nextIndex < 0) {
      displayNoFrame();
      return;
    }
    displayFrame(nextIndex);
  }

  function stepBackward() {
    const frames = animationFramesRef.current;
    if (!frames.length) {
      displayNoFrame();
      return;
    }
    pauseAnimation();
    const prevIndex = animationIndexRef.current - 1;
    if (prevIndex < 0) {
      displayNoFrame();
      return;
    }
    displayFrame(prevIndex);
  }

  function jumpToEnd() {
    const frames = animationFramesRef.current;
    if (!frames.length) return;
    pauseAnimation();
    displayFrame(frames.length - 1);
  }

  function animateSpikes() {
    const result = simulate(neurons, edges, h, T, inputSpikeTrains);
    setSim(result);

    stopAnimation();
    const cascade = result.cascadeFrames ?? [];
    if (!cascade.length) {
      return;
    }
    const frames: AnimationFrame[] = cascade
      .map((step) => {
        const neuronIds = Array.from(new Set([...step.sources, ...step.targets]));
        if (!neuronIds.length && !step.edges.length) {
          return null;
        }
        return {
          time: step.time,
          wave: step.wave,
          sourceIds: step.sources,
          targetIds: step.targets,
          neuronIds,
          edgeIds: step.edges,
        };
      })
      .filter((frame): frame is AnimationFrame => frame !== null);
    if (!frames.length) {
      return;
    }
    updateAnimationFramesState(frames);
    displayNoFrame();
    setAnimationPlaying(true);
    scheduleNextFrame(-1);
  }

  useEffect(() => {
    return () => {
      clearPlaybackTimer();
    };
  }, []);

  function addNeuron(role: Role = "hidden") {
    if (isRestoringRef.current) return;
    recordSnapshot();
    setNeurons((ns) => {
      const id = uid("n");
      const label = `N${ns.length + 1}`;
      const minX = NODE_RADIUS;
      const minY = NODE_RADIUS;
      const maxX = Math.max(NODE_RADIUS, canvasWidth - NODE_RADIUS);
      const maxY = Math.max(NODE_RADIUS, canvasHeight - NODE_RADIUS);
      const rawX = 120 + ns.length * 40;
      const rawY = 100 + ((ns.length * 60) % 240);
      const x = clamp(snapToGrid(rawX), minX, maxX);
      const y = clamp(snapToGrid(rawY), minY, maxY);
      return [
        ...ns,
        {
          id,
          label,
          role,
          x,
          y,
          labelVisible: false,
          labelOffsetX: 0,
          labelOffsetY: DEFAULT_LABEL_OFFSET_Y,
        },
      ];
    });
  }

  function deleteSelected(): boolean {
    if (isRestoringRef.current) return false;
    if (!selectedEdgeId && selectedNodeIds.length === 0) return false;
    recordSnapshot();
    setEdges((es) =>
      es.filter((e) => {
        if (selectedEdgeId && e.id === selectedEdgeId) return false;
        if (selectedNodeIds.includes(e.sourceId) || selectedNodeIds.includes(e.targetId)) return false;
        return true;
      })
    );
    if (selectedNodeIds.length) {
      const toRemove = new Set(selectedNodeIds);
      setDynamicsPanels((prev) => prev.filter((panel) => !toRemove.has(panel.neuronId)));
      setNeurons((ns) => ns.filter((n) => !toRemove.has(n.id)));
      // prune groups: remove deleted nodes and drop groups with <2 members
      setGroups((gs) =>
        gs
          .map((g) => {
            const remainingIds = g.nodeIds.filter((id) => !toRemove.has(id));
            const compression = g.compression
              ? {
                  center: { ...g.compression.center },
                  nodes: Object.fromEntries(
                    Object.entries(g.compression.nodes)
                      .filter(([id]) => !toRemove.has(id))
                      .map(([id, data]) => [id, { ...data }])
                  ),
                  edgePoints: Object.fromEntries(
                    Object.entries(g.compression.edgePoints ?? {}).map(([edgeId, points]) => [
                      edgeId,
                      points.map((pt) => ({ ...pt })),
                    ])
                  ),
                  labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
                }
              : undefined;
            return {
              ...g,
              nodeIds: remainingIds,
              labelVisible: g.labelVisible === true,
              compression: compression && Object.keys(compression.nodes).length >= 2 ? compression : undefined,
            };
          })
          .filter((g) => g.nodeIds.length >= 2)
      );
    }
    setSelectedEdgeId(null);
    setSelectedNodeIds([]);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
    return true;
  }

  function collapseSelectedNeuron(): boolean {
    if (isRestoringRef.current) return false;
    if (selectedEdgeId) return false;
    if (selectedNodeIds.length !== 1) return false;
    const neuronId = selectedNodeIds[0];
    const incomingEdges = edges.filter((edge) => edge.targetId === neuronId);
    const outgoingEdges = edges.filter((edge) => edge.sourceId === neuronId);
    if (incomingEdges.length !== 1 || outgoingEdges.length !== 1) return false;
    const incomingEdge = incomingEdges[0];
    const outgoingEdge = outgoingEdges[0];
    const sourceId = incomingEdge.sourceId;
    const targetId = outgoingEdge.targetId;
    if (!sourceId || !targetId) return false;
    if (sourceId === targetId) return false;
    recordSnapshot();
    // Average the weights of incoming/outgoing edges for the new combined connection
    const combinedWeight = (incomingEdge.weight + outgoingEdge.weight) / 2;
    const toRemove = new Set([neuronId]);
    let createdEdgeId: string | null = null;
    setEdges((prevEdges) => {
      const nextEdges = prevEdges.filter(
        (edge) =>
          edge.id !== incomingEdge.id &&
          edge.id !== outgoingEdge.id &&
          edge.sourceId !== neuronId &&
          edge.targetId !== neuronId
      );
      const existingIndex = nextEdges.findIndex((edge) => edge.sourceId === sourceId && edge.targetId === targetId);
      if (existingIndex >= 0) {
        const existing = nextEdges[existingIndex];
        createdEdgeId = existing.id;
        nextEdges[existingIndex] = { ...existing, weight: combinedWeight, points: undefined };
      } else {
        createdEdgeId = uid("e");
        nextEdges.push({
          id: createdEdgeId,
          sourceId,
          targetId,
          weight: combinedWeight,
        });
      }
      return nextEdges;
    });
    setDynamicsPanels((prev) => prev.filter((panel) => !toRemove.has(panel.neuronId)));
    setNeurons((ns) => ns.filter((n) => !toRemove.has(n.id)));
    setGroups((gs) =>
      gs
        .map((g) => {
          const remainingIds = g.nodeIds.filter((id) => !toRemove.has(id));
          const compression = g.compression
            ? {
                center: { ...g.compression.center },
                nodes: Object.fromEntries(
                  Object.entries(g.compression.nodes)
                    .filter(([id]) => !toRemove.has(id))
                    .map(([id, data]) => [id, { ...data }])
                ),
                edgePoints: Object.fromEntries(
                  Object.entries(g.compression.edgePoints ?? {}).map(([edgeId, points]) => [
                    edgeId,
                    points.map((pt) => ({ ...pt })),
                  ])
                ),
                labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
              }
            : undefined;
          return {
            ...g,
            nodeIds: remainingIds,
            labelVisible: g.labelVisible === true,
            compression: compression && Object.keys(compression.nodes).length >= 2 ? compression : undefined,
          };
        })
        .filter((g) => g.nodeIds.length >= 2)
    );
    setSelectedEdgeId(createdEdgeId);
    setSelectedNodeIds([]);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
    return true;
  }

  function beginConnect(id: string) {
    if (mode !== "connect") return;
    setConnectSrc(id);
    setConnectPoints([]);
  }

  function completeConnect(id: string) {
    if (mode !== "connect" || !connectSrc) return;
    if (connectSrc === id) {
      setConnectSrc(null);
      setConnectPoints([]);
      return;
    }
    if (isRestoringRef.current) return;
    recordSnapshot();
    const src = connectSrc;
    const boardMinX = NODE_RADIUS;
    const boardMinY = NODE_RADIUS;
    const boardMaxX = Math.max(NODE_RADIUS, canvasWidth - NODE_RADIUS);
    const boardMaxY = Math.max(NODE_RADIUS, canvasHeight - NODE_RADIUS);
    let createdEdgeId: string | null = null;
    setEdges((es) => {
      if (es.some((e) => e.sourceId === src && e.targetId === id)) return es;
      const waypointList = connectPoints.map((pt) => ({
        x: clamp(snapToGrid(pt.x), boardMinX, boardMaxX),
        y: clamp(snapToGrid(pt.y), boardMinY, boardMaxY),
      }));
      const points = waypointList.length ? waypointList : undefined;
      const edge: Edge = { id: uid("e"), sourceId: src, targetId: id, weight: defaultEdgeWeight, points };
      createdEdgeId = edge.id;
      return [...es, edge];
    });
    setConnectSrc(null);
    setConnectPoints([]);
    if (createdEdgeId) {
      setSelectedNodeIds([]);
      setSelectedEdgeId(createdEdgeId);
      setSelectedEdgeHandle(null);
      setActiveSelectionContext("main");
    }
  }

  function addConnectWaypoint(pt: { x: number; y: number }) {
    if (mode !== "connect" || !connectSrc) return;
    if (isRestoringRef.current) return;
    const boardMinX = NODE_RADIUS;
    const boardMinY = NODE_RADIUS;
    const boardMaxX = Math.max(NODE_RADIUS, canvasWidth - NODE_RADIUS);
    const boardMaxY = Math.max(NODE_RADIUS, canvasHeight - NODE_RADIUS);
    const x = clamp(snapToGrid(pt.x), boardMinX, boardMaxX);
    const y = clamp(snapToGrid(pt.y), boardMinY, boardMaxY);
    setConnectPoints((prev) => [...prev, { x, y }]);
  }

  function cancelConnect() {
    setConnectSrc(null);
    setConnectPoints([]);
  }

  function exitConnectMode() {
    setMode("select");
    cancelConnect();
  }

  function runSimulation() {
    stopAnimation();
    const result = simulate(neurons, edges, h, T, inputSpikeTrains);
    setSim(result);
  }

  function clearSimulation() {
    stopAnimation();
    setSim(null);
  }

  function setRole(id: string, role: Role) {
    if (isRestoringRef.current) return;
    recordSnapshot();
    setNeurons((ns) => ns.map((n) => (n.id === id ? { ...n, role } : n)));
  }

  function setLabel(id: string, label: string) {
    if (isRestoringRef.current) return;
    recordSnapshot();
    setNeurons((ns) => ns.map((n) => (n.id === id ? { ...n, label } : n)));
  }

  function setWeight(edgeId: string, w: number) {
    if (isRestoringRef.current) return;
    recordSnapshot();
    setEdges((es) => es.map((e) => (e.id === edgeId ? { ...e, weight: w } : e)));
  }

  function removeEdge(edgeId: string) {
    if (isRestoringRef.current) return;
    recordSnapshot();
    setEdges((es) => es.filter((e) => e.id !== edgeId));
    setSelectedEdgeId((prev) => (prev === edgeId ? null : prev));
    setSelectedEdgeHandle((prev) => (prev && prev.edgeId === edgeId ? null : prev));
  }

  // ------------------------ Grouping (Main Canvas) ------------------------
  function computeGroupHue(existing: Group[]): number {
    // Cycle through pleasant pastel hues
    const palette = [25, 85, 155, 205, 265, 315];
    const used = existing.map((g) => g.hue);
    const unused = palette.find((h) => !used.includes(h));
    if (unused !== undefined) return unused;
    return palette[existing.length % palette.length];
  }

  function nextGroupLabel(existing: Group[], preferred?: string | null): string {
    const basePref = preferred ? preferred.trim() : "";
    if (basePref) return basePref;
    return `Group ${existing.length + 1}`;
  }

  function groupSelectionMain(): boolean {
    const sel = Array.from(new Set(selectedNodeIds));
    if (sel.length < 2) return false;
    if (isRestoringRef.current) return false;
    recordSnapshot();
    // merge any overlapping groups with selection
    const selSet = new Set(sel);
    const overlapping = groups.filter((g) => g.nodeIds.some((id) => selSet.has(id)));
    const mergedIds = new Set<string>(sel);
    for (const g of overlapping) for (const id of g.nodeIds) mergedIds.add(id);
    const hue = overlapping.length ? overlapping[0].hue : computeGroupHue(groups);
    const baseLabel = overlapping.find((g) => g.label)?.label;
    const label = baseLabel ? baseLabel : nextGroupLabel(groups);
    const labelVisible = overlapping.some((g) => g.labelVisible);
    const newGroup: Group = {
      id: uid("grp"),
      nodeIds: Array.from(mergedIds),
      hue,
      label,
      labelVisible,
    };
    setGroups((prev) => {
      const remaining = prev.filter((g) => !overlapping.includes(g));
      return [...remaining, newGroup];
    });
    const mergedSet = new Set(newGroup.nodeIds);
    setNeurons((ns) =>
      ns.map((n) => (mergedSet.has(n.id) ? { ...n, role: "hidden" as Role } : n))
    );
    // ensure selection equals the group contents
    setSelectedNodeIds(newGroup.nodeIds);
    setSelectedEdgeId(null);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
    return true;
  }

  function hasSelectionGroup(): boolean {
    if (!selectedNodeIds.length || !groups.length) return false;
    const sel = new Set(selectedNodeIds);
    return groups.some((g) => g.nodeIds.some((id) => sel.has(id)));
  }

  function setSelectionGroupLabelVisibility(visible: boolean): boolean {
    if (!selectedNodeIds.length) return false;
    if (isRestoringRef.current) return false;
    const sel = new Set(selectedNodeIds);
    let changed = false;
    setGroups((prev) => {
      let mutated = false;
      const next = prev.map((g) => {
        if (g.nodeIds.some((id) => sel.has(id))) {
          if (g.labelVisible !== visible) {
            mutated = true;
            return { ...g, labelVisible: visible };
          }
        }
        return g;
      });
      if (!mutated) return prev;
      if (isRestoringRef.current) return prev;
      recordSnapshot();
      changed = true;
      return next;
    });
    return changed;
  }

  function renameGroup(groupId: string, nextLabel: string) {
    if (isRestoringRef.current) return;
    setGroups((prev) => {
      const exists = prev.find((g) => g.id === groupId);
      if (!exists) return prev;
      const current = (exists.label ?? "").trim();
      const desired = nextLabel.trim();
      const normalized = desired || nextGroupLabel(prev.filter((g) => g.id !== groupId));
      if (normalized === exists.label) return prev;
      recordSnapshot();
      return prev.map((g) => (g.id === groupId ? { ...g, label: normalized } : g));
    });
  }

  function compressGroup(groupId: string): boolean {
    if (isRestoringRef.current) return false;
    const group = groups.find((g) => g.id === groupId);
    if (!group || group.compression) return false;
    const nodes = neurons.filter((n) => group.nodeIds.includes(n.id));
    if (!nodes.length) return false;
    recordSnapshot();
    const centerX = nodes.reduce((sum, n) => sum + n.x, 0) / nodes.length;
    const centerY = nodes.reduce((sum, n) => sum + n.y, 0) / nodes.length;
    const stored: GroupCompressionState["nodes"] = Object.fromEntries(
      nodes.map((n) => [
        n.id,
        {
          x: n.x,
          y: n.y,
          labelOffsetX: Number.isFinite(n.labelOffsetX) ? (n.labelOffsetX as number) : 0,
          labelOffsetY: Number.isFinite(n.labelOffsetY) ? (n.labelOffsetY as number) : DEFAULT_LABEL_OFFSET_Y,
          labelVisible: n.labelVisible === true,
        },
      ])
    );
    const groupNodeSet = new Set(Object.keys(stored));
    const edgePointStore: Record<string, Point[]> = {};
    for (const edge of edges) {
      if (!edge || !edge.points || !edge.points.length) continue;
      if (!groupNodeSet.has(edge.sourceId) && !groupNodeSet.has(edge.targetId)) continue;
      const sanitized = sanitizeWaypoints(edge.points);
      if (!sanitized.length) continue;
      edgePointStore[edge.id] = sanitized.map((pt) => ({ ...pt }));
    }
    const affectedEdgeIds = new Set(Object.keys(edgePointStore));
    if (affectedEdgeIds.size) {
      setEdges((prev) =>
        prev.map((edge) =>
          affectedEdgeIds.has(edge.id) ? { ...edge, points: [{ x: centerX, y: centerY }] } : edge
        )
      );
      if (selectedEdgeHandle && affectedEdgeIds.has(selectedEdgeHandle.edgeId)) {
        setSelectedEdgeHandle(null);
      }
    }
    setNeurons((ns) =>
      ns.map((n) =>
        stored[n.id]
          ? {
              ...n,
              x: centerX,
              y: centerY,
              labelVisible: false,
              labelOffsetX: 0,
              labelOffsetY: 0,
            }
          : n
      )
    );
    setGroups((prev) =>
      prev.map((g) =>
        g.id === groupId
          ? {
              ...g,
              compression: {
                center: { x: centerX, y: centerY },
                nodes: stored,
                edgePoints: edgePointStore,
                labelVisibleBefore: g.labelVisible === true,
              },
              labelVisible: true,
            }
          : g
      )
    );
    return true;
  }

  function expandGroup(groupId: string): boolean {
    if (isRestoringRef.current) return false;
    const group = groups.find((g) => g.id === groupId);
    if (!group || !group.compression) return false;
    recordSnapshot();
    const saved = group.compression.nodes;
    const savedEdgePoints = group.compression.edgePoints ?? {};
    const edgeRestoreEntries = Object.entries(savedEdgePoints);
    if (edgeRestoreEntries.length) {
      const restoreMap = new Map(
        edgeRestoreEntries.map(([edgeId, points]) => [edgeId, points.map((pt) => ({ ...pt }))])
      );
      setEdges((prev) =>
        prev.map((edge) => {
          const points = restoreMap.get(edge.id);
          if (!points) return edge;
          return { ...edge, points };
        })
      );
      if (selectedEdgeHandle && restoreMap.has(selectedEdgeHandle.edgeId)) {
        setSelectedEdgeHandle(null);
      }
    }
    const labelVisibleBefore = group.compression.labelVisibleBefore ?? (group.labelVisible === true);
    setNeurons((ns) =>
      ns.map((n) =>
        saved[n.id]
          ? {
              ...n,
              x: saved[n.id].x,
              y: saved[n.id].y,
              labelOffsetX: saved[n.id].labelOffsetX,
              labelOffsetY: saved[n.id].labelOffsetY,
              labelVisible: saved[n.id].labelVisible,
            }
          : n
      )
    );
    setGroups((prev) =>
      prev.map((g) =>
        g.id === groupId
          ? {
              ...g,
              compression: undefined,
              labelVisible: labelVisibleBefore,
            }
          : g
      )
    );
    return true;
  }

  function ungroupSelectionMain(): boolean {
    if (!selectedNodeIds.length) return false;
    const sel = new Set(selectedNodeIds);
    const overlapping = groups.filter((g) => g.nodeIds.some((id) => sel.has(id)));
    if (!overlapping.length) return false;
    if (isRestoringRef.current) return false;
    recordSnapshot();
    setGroups((prev) => prev.filter((g) => !overlapping.includes(g)));
    // keep current selection as-is (nodes remain selected individually)
    setSelectedEdgeId(null);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
    return true;
  }

  // Prune groups if neurons list changes (e.g., import or external edits)
  useEffect(() => {
    const validIds = new Set(neurons.map((n) => n.id));
    setGroups((prev) =>
      prev
        .map((g) => {
          const remainingIds = g.nodeIds.filter((id) => validIds.has(id));
          const compression = g.compression
            ? {
                center: { ...g.compression.center },
                nodes: Object.fromEntries(
                  Object.entries(g.compression.nodes)
                    .filter(([id]) => validIds.has(id))
                    .map(([id, data]) => [id, { ...data }])
                ),
                edgePoints: Object.fromEntries(
                  Object.entries(g.compression.edgePoints ?? {}).map(([edgeId, points]) => [
                    edgeId,
                    points.map((pt) => ({ ...pt })),
                  ])
                ),
                labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
              }
            : undefined;
          return {
            ...g,
            nodeIds: remainingIds,
            labelVisible: g.labelVisible === true,
            compression: compression && Object.keys(compression.nodes).length >= 2 ? compression : undefined,
          };
        })
        .filter((g) => g.nodeIds.length >= 2)
    );
    setDynamicsPanels((prev) => prev.filter((panel) => validIds.has(panel.neuronId)));
  }, [neurons]);

  function clearSelection() {
    setSelectedNodeIds([]);
    setSelectedEdgeId(null);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
  }

  function performCanvasClean() {
    if (isRestoringRef.current) return;
    const hasContent =
      neurons.length > 0 ||
      edges.length > 0 ||
      groups.length > 0 ||
      dynamicsPanels.length > 0 ||
      sim !== null ||
      selectedNodeIds.length > 0 ||
      selectedEdgeId !== null ||
      connectSrc !== null ||
      connectPoints.length > 0 ||
      mode !== "select" ||
      highlightedNodeIds.length > 0 ||
      highlightedEdgeIds.length > 0 ||
      recentNodeIds.length > 0 ||
      recentEdgeIds.length > 0 ||
      animationFrames.length > 0 ||
      animationIndex >= 0 ||
      isAnimating ||
      groupLabelDraft.trim().length > 0;
    if (hasContent) {
      recordSnapshot();
    }
    clearSimulation();
    cancelConnect();
    setMode("select");
    setNeurons([]);
    setEdges([]);
    setGroups([]);
    setDynamicsPanels([]);
    clearSelection();
    setGroupLabelDraft("");
    copyBufferRef.current = null;
    moduleCopyBufferRef.current = null;
  }

  function handleCleanDialog(action: "abort" | "clean" | "save") {
    if (action === "abort") {
      setShowCleanDialog(false);
      return;
    }
    if (action === "save") {
      exportState();
      performCanvasClean();
    } else if (action === "clean") {
      performCanvasClean();
    }
    setShowCleanDialog(false);
  }

  function handleNodePointerDown(
    id: string,
    modifiers: { append: boolean; toggle: boolean }
  ): string[] {
    // If the node is in a group, operate on the entire group's nodes
    const group = groups.find((g) => g.nodeIds.includes(id));
    const groupIds = group ? group.nodeIds : [id];
    let nextSelection: string[] = [];
    setSelectedNodeIds((prev) => {
      let next: string[];
      if (modifiers.toggle) {
        // toggle all in the group at once
        const anySelected = groupIds.some((gid) => prev.includes(gid));
        if (anySelected) next = prev.filter((nid) => !groupIds.includes(nid));
        else next = [...prev, ...groupIds.filter((gid) => !prev.includes(gid))];
      } else if (modifiers.append) {
        const anySelected = groupIds.some((gid) => prev.includes(gid));
        if (anySelected) next = prev.filter((nid) => !groupIds.includes(nid));
        else next = [...prev, ...groupIds.filter((gid) => !prev.includes(gid))];
      } else {
        next = [...groupIds];
      }
      nextSelection = next;
      return next;
    });
    setSelectedEdgeId(null);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
    return nextSelection;
  }

  function handleGroupPointerDown(
    groupId: string,
    modifiers: { append: boolean; toggle: boolean }
  ): string[] {
    if (isRestoringRef.current) return [];
    const group = groups.find((g) => g.id === groupId);
    if (!group) return [];
    const groupIds = Array.from(new Set(group.nodeIds));
    if (!groupIds.length) return [];
    let nextSelection: string[] = [];
    setSelectedNodeIds((prev) => {
      let next: string[];
      if (modifiers.toggle) {
        const anySelected = groupIds.some((id) => prev.includes(id));
        if (anySelected) next = prev.filter((id) => !groupIds.includes(id));
        else next = [...prev, ...groupIds.filter((id) => !prev.includes(id))];
      } else if (modifiers.append) {
        const anySelected = groupIds.some((id) => prev.includes(id));
        if (anySelected) next = prev.filter((id) => !groupIds.includes(id));
        else next = [...prev, ...groupIds.filter((id) => !prev.includes(id))];
      } else {
        next = [...groupIds];
      }
      const unique = Array.from(new Set(next));
      nextSelection = unique;
      return unique;
    });
    setSelectedEdgeId(null);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
    return nextSelection;
  }

  function handleMarqueeSelect(ids: string[], modifiers: { append: boolean; toggle: boolean }) {
    if (!ids.length) return;
    setSelectedNodeIds((prev) => {
      const existing = new Set(prev);
      const filteredIds = ids.filter((id) => byId[id]);
      if (!filteredIds.length) return prev;
      let next: string[];
      if (modifiers.toggle) {
        const set = new Set(existing);
        const anySelected = filteredIds.some((id) => set.has(id));
        if (anySelected) filteredIds.forEach((id) => set.delete(id));
        else filteredIds.forEach((id) => set.add(id));
        next = Array.from(set);
      } else if (modifiers.append) {
        const set = new Set(existing);
        filteredIds.forEach((id) => set.add(id));
        next = Array.from(set);
      } else {
        next = filteredIds;
      }
      return next;
    });
    setSelectedEdgeId(null);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
  }

  function moveSelectedNodesBy(dx: number, dy: number): boolean {
    if (isRestoringRef.current) return false;
    if (!selectedNodeIds.length) return false;
    const updates = selectedNodeIds
      .map((id) => {
        const node = byId[id];
        if (!node) return null;
        return { id, x: node.x + dx, y: node.y + dy };
      })
      .filter((u): u is { id: string; x: number; y: number } => u !== null);
    if (!updates.length) return false;
    beginNodeDrag();
    moveNodes(updates);
    endDrag();
    return true;
  }

  function moveModuleSelectedNodesBy(dx: number, dy: number): boolean {
    if (!moduleSelectedNodeIds.length) return false;
    const updates = moduleSelectedNodeIds
      .map((id) => {
        const node = moduleById[id];
        if (!node) return null;
        return { id, x: node.x + dx, y: node.y + dy };
      })
      .filter((u): u is { id: string; x: number; y: number } => u !== null);
    if (!updates.length) return false;
    moveModuleNodes(updates);
    return true;
  }

  function openDynamicsPanels() {
    if (!sim) {
      alert("Run the simulation before showing neuron dynamics.");
      return;
    }
    for (const id of selectedNodeIds) {
      openDynamicsPanelForNeuron(id);
    }
  }

  function moveDynamicsPanel(panelId: string, x: number, y: number) {
    setDynamicsPanels((prev) => prev.map((p) => (p.id === panelId ? { ...p, x, y } : p)));
  }

  function closeDynamicsPanel(panelId: string) {
    setDynamicsPanels((prev) => prev.filter((p) => p.id !== panelId));
  }

  function closeAllDynamicsPanels() {
    if (!dynamicsPanels.length) return;
    setDynamicsPanels([]);
  }

  function handleEdgePointerDown(id: string) {
    setSelectedNodeIds([]);
    setSelectedEdgeId(id);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
  }

  function moveNodes(updates: Array<{ id: string; x: number; y: number }>) {
    if (!updates.length) return;
    if (isRestoringRef.current) return;
    ensureNodeDragSnapshot();
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, canvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, canvasHeight - NODE_RADIUS);
    const prevPositions = new Map<string, { x: number; y: number }>();
    const currentById = new Map(neurons.map((n) => [n.id, n]));
    const updateById = new Map(updates.map((u) => [u.id, u]));
    const groupMoveNodeIds = new Set<string>();
    const TRANSLATION_EPSILON = 1e-6;

    for (const group of groups) {
      if (!group.nodeIds.length) continue;
      let baseDx: number | null = null;
      let baseDy: number | null = null;
      let allMembersMoved = true;
      for (const nodeId of group.nodeIds) {
        const prev = currentById.get(nodeId);
        const target = updateById.get(nodeId);
        if (!prev || !target) {
          allMembersMoved = false;
          break;
        }
        const dx = target.x - prev.x;
        const dy = target.y - prev.y;
        if (baseDx === null || baseDy === null) {
          baseDx = dx;
          baseDy = dy;
          continue;
        }
        if (Math.abs(dx - baseDx) > TRANSLATION_EPSILON || Math.abs(dy - baseDy) > TRANSLATION_EPSILON) {
          allMembersMoved = false;
          break;
        }
      }
      if (allMembersMoved && baseDx !== null && baseDy !== null) {
        for (const nodeId of group.nodeIds) {
          groupMoveNodeIds.add(nodeId);
        }
      }
    }

    const normalizedUpdates = updates.map((u) => {
      const prev = currentById.get(u.id);
      if (prev) prevPositions.set(u.id, { x: prev.x, y: prev.y });
      const allowOutside = groupMoveNodeIds.has(u.id);
      const rawX = allowOutside ? u.x : clamp(u.x, minX, maxX);
      const rawY = allowOutside ? u.y : clamp(u.y, minY, maxY);
      const snappedX = snapToGrid(rawX);
      const snappedY = snapToGrid(rawY);
      return {
        id: u.id,
        x: allowOutside ? snappedX : clamp(snappedX, minX, maxX),
        y: allowOutside ? snappedY : clamp(snappedY, minY, maxY),
      };
    });
    const map = new Map(normalizedUpdates.map((u) => [u.id, u]));
    setNeurons((ns) => ns.map((n) => (map.has(n.id) ? { ...n, ...map.get(n.id)! } : n)));

    const deltaById = new Map<string, { dx: number; dy: number }>();
    for (const { id } of normalizedUpdates) {
      const prev = prevPositions.get(id);
      const next = map.get(id);
      if (!prev || !next) continue;
      const dx = next.x - prev.x;
      const dy = next.y - prev.y;
      if (dx === 0 && dy === 0) continue;
      deltaById.set(id, { dx, dy });
    }

    const movedIds = new Set(normalizedUpdates.map((u) => u.id));
    const boardMinX = minX;
    const boardMinY = minY;
    const boardMaxX = maxX;
    const boardMaxY = maxY;
    if (deltaById.size) {
      const edgeTranslations = new Map<string, { dx: number; dy: number }>();
      const EPSILON = 1e-6;
      for (const edge of edges) {
        if (!edge || !edge.sourceId || !edge.targetId) continue;
        const waypoints = sanitizeWaypoints(edge.points);
        if (!waypoints.length) continue;
        const sourceDelta = deltaById.get(edge.sourceId);
        const targetDelta = deltaById.get(edge.targetId);
        if (!sourceDelta || !targetDelta) continue;
        if (Math.abs(sourceDelta.dx - targetDelta.dx) > EPSILON || Math.abs(sourceDelta.dy - targetDelta.dy) > EPSILON) continue;
        edgeTranslations.set(edge.id, sourceDelta);
      }
      if (edgeTranslations.size) {
        setEdges((es) =>
          es.map((edge) => {
            const translation = edgeTranslations.get(edge.id);
            if (!translation) return edge;
            const waypoints = sanitizeWaypoints(edge.points);
            if (!waypoints.length) return edge;
            const { dx, dy } = translation;
            if (dx === 0 && dy === 0) return edge;
            const allowOutside = groupMoveNodeIds.has(edge.sourceId) && groupMoveNodeIds.has(edge.targetId);
            const translated = waypoints.map((pt) => {
              const x = pt.x + dx;
              const y = pt.y + dy;
              if (allowOutside) {
                return { x, y };
              }
              return {
                x: clamp(x, boardMinX, boardMaxX),
                y: clamp(y, boardMinY, boardMaxY),
              };
            });
            return { ...edge, points: translated };
          })
        );
      }
    }

    const groupTranslations: Array<{ groupId: string; nodeIds: string[]; dx: number; dy: number }> = [];
    for (const g of groups) {
      if (!g.nodeIds.length) continue;
      const allMoved = g.nodeIds.every((id) => movedIds.has(id));
      if (!allMoved) continue;
      const refId = g.nodeIds.find((id) => prevPositions.has(id) && map.has(id));
      if (!refId) continue;
      const prev = prevPositions.get(refId)!;
      const next = map.get(refId)!;
      const dx = next.x - prev.x;
      const dy = next.y - prev.y;
      if (dx === 0 && dy === 0) continue;
      groupTranslations.push({ groupId: g.id, nodeIds: [...g.nodeIds], dx, dy });
    }

    if (groupTranslations.length) {
      setGroups((prev) =>
        prev.map((g) => {
          const trans = groupTranslations.find((gt) => gt.groupId === g.id);
          if (!trans || !g.compression) return g;
          const { dx, dy } = trans;
          const updatedNodes: GroupCompressionState["nodes"] = Object.fromEntries(
            Object.entries(g.compression.nodes).map(([nodeId, data]) => [nodeId, { ...data, x: data.x + dx, y: data.y + dy }])
          );
          const updatedEdgePoints = Object.fromEntries(
            Object.entries(g.compression.edgePoints ?? {}).map(([edgeId, points]) => [
              edgeId,
              points.map((pt) => ({
                x: clamp(pt.x + dx, boardMinX, boardMaxX),
                y: clamp(pt.y + dy, boardMinY, boardMaxY),
              })),
            ])
          );
          return {
            ...g,
            compression: {
              center: { x: g.compression.center.x + dx, y: g.compression.center.y + dy },
              nodes: updatedNodes,
              edgePoints: updatedEdgePoints,
              labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
            },
          };
        })
      );
      const collapsedEdgeCenters = new Map<string, { x: number; y: number }>();
      for (const trans of groupTranslations) {
        const group = groups.find((g) => g.id === trans.groupId);
        if (!group?.compression) continue;
        const newCenterX = group.compression.center.x + trans.dx;
        const newCenterY = group.compression.center.y + trans.dy;
        for (const edgeId of Object.keys(group.compression.edgePoints ?? {})) {
          collapsedEdgeCenters.set(edgeId, { x: newCenterX, y: newCenterY });
        }
      }
      if (collapsedEdgeCenters.size) {
        setEdges((prev) =>
          prev.map((edge) => {
            const center = collapsedEdgeCenters.get(edge.id);
            if (!center) return edge;
            return { ...edge, points: [{ x: center.x, y: center.y }] };
          })
        );
      }
    }
  }

  function moveLabelOffsets(updates: Array<{ id: string; labelOffsetX: number; labelOffsetY: number }>) {
    if (!updates.length) return;
    if (isRestoringRef.current) return;
    ensureLabelDragSnapshot();
    const map = new Map(updates.map((u) => [u.id, u]));
    setNeurons((ns) => ns.map((n) => (map.has(n.id) ? { ...n, ...map.get(n.id)! } : n)));
  }

  function setLabelVisibility(id: string, visible: boolean) {
    if (isRestoringRef.current) return;
    recordSnapshot();
    setNeurons((ns) => ns.map((n) => (n.id === id ? { ...n, labelVisible: visible } : n)));
  }

  function setLabelOffset(id: string, offsetX: number, offsetY: number) {
    if (isRestoringRef.current) return;
    recordSnapshot();
    setNeurons((ns) =>
      ns.map((n) => (n.id === id ? { ...n, labelOffsetX: offsetX, labelOffsetY: offsetY } : n))
    );
  }

  function resetLabelOffset(id: string) {
    if (isRestoringRef.current) return;
    recordSnapshot();
    setNeurons((ns) =>
      ns.map((n) =>
        n.id === id ? { ...n, labelOffsetX: 0, labelOffsetY: DEFAULT_LABEL_OFFSET_Y } : n
      )
    );
  }

  function updateEdgeWaypoint(edgeId: string, index: number, point: Point) {
    if (isRestoringRef.current) return;
    ensureEdgeDragSnapshot();
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, canvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, canvasHeight - NODE_RADIUS);
    const x = clamp(snapToGrid(point.x), minX, maxX);
    const y = clamp(snapToGrid(point.y), minY, maxY);
    setEdges((es) =>
      es.map((edge) => {
        if (edge.id !== edgeId) return edge;
        const waypoints = sanitizeWaypoints(edge.points);
        if (index < 0 || index >= waypoints.length) return edge;
        const next = [...waypoints];
        next[index] = { x, y };
        return { ...edge, points: next };
      })
    );
    setActiveSelectionContext("main");
  }

  function insertEdgeWaypoint(edgeId: string, point: Point, index: number) {
    if (isRestoringRef.current) return;
    recordSnapshot();
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, canvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, canvasHeight - NODE_RADIUS);
    const x = clamp(snapToGrid(point.x), minX, maxX);
    const y = clamp(snapToGrid(point.y), minY, maxY);
    let insertedIndex: number | null = null;
    setEdges((es) =>
      es.map((edge) => {
        if (edge.id !== edgeId) return edge;
        const waypoints = sanitizeWaypoints(edge.points);
        const next = [...waypoints];
        const safeIndex = Math.max(0, Math.min(index, next.length));
        next.splice(safeIndex, 0, { x, y });
        insertedIndex = safeIndex;
        return { ...edge, points: next };
      })
    );
    if (insertedIndex !== null) {
      setSelectedEdgeId(edgeId);
      setSelectedEdgeHandle({ edgeId, index: insertedIndex });
      setActiveSelectionContext("main");
    }
  }

  function removeEdgeWaypoint(edgeId: string, index: number): boolean {
    if (isRestoringRef.current) return false;
    recordSnapshot();
    const edge = edges.find((e) => e.id === edgeId);
    if (!edge) return false;
    const waypoints = sanitizeWaypoints(edge.points);
    if (index < 0 || index >= waypoints.length) return false;
    setEdges((es) =>
      es.map((e) => {
        if (e.id !== edgeId) return e;
        const next = [...waypoints];
        next.splice(index, 1);
        return { ...e, points: next.length ? next : undefined };
      })
    );
    const nextLength = waypoints.length - 1;
    if (nextLength > 0) {
      const nextIndex = Math.min(index, nextLength - 1);
      setSelectedEdgeHandle({ edgeId, index: nextIndex });
    } else {
      setSelectedEdgeHandle(null);
    }
    setActiveSelectionContext("main");
    return true;
  }

  function removeSelectedEdgeWaypoint(): boolean {
    if (!selectedEdgeHandle || !selectedEdgeId) return false;
    if (selectedEdgeHandle.edgeId !== selectedEdgeId) return false;
    return removeEdgeWaypoint(selectedEdgeHandle.edgeId, selectedEdgeHandle.index);
  }

  function copySelection(): boolean {
    if (!selectedNodeIds.length) return false;
    const selectedSet = new Set(selectedNodeIds);
    const nodes = neurons
      .filter((n) => selectedSet.has(n.id))
      .map((n) => ({
        ...n,
        labelOffsetX: Number.isFinite(n.labelOffsetX) ? (n.labelOffsetX as number) : 0,
        labelOffsetY: Number.isFinite(n.labelOffsetY) ? (n.labelOffsetY as number) : DEFAULT_LABEL_OFFSET_Y,
      }));
    if (!nodes.length) return false;
    const copiedEdges = edges
      .filter((e) => selectedSet.has(e.sourceId) && selectedSet.has(e.targetId))
      .map((e) => ({ ...e, points: sanitizeWaypoints(e.points) }));
    const selectedEdgeIds = new Set(copiedEdges.map((e) => e.id));
    const copiedGroups = groups
      .filter((g) => g.nodeIds.every((id) => selectedSet.has(id)))
      .map((g) => ({
        nodeIds: [...g.nodeIds],
        hue: g.hue,
        label: typeof g.label === "string" ? g.label : "",
        labelVisible: g.labelVisible === true,
        compression: g.compression
          ? {
              center: { ...g.compression.center },
              nodes: Object.fromEntries(
                Object.entries(g.compression.nodes)
                  .filter(([id]) => selectedSet.has(id))
                  .map(([id, data]) => [id, { ...data }])
              ),
              edgePoints: Object.fromEntries(
                Object.entries(g.compression.edgePoints ?? {})
                  .filter(([edgeId]) => selectedEdgeIds.has(edgeId))
                  .map(([edgeId, points]) => [edgeId, points.map((pt) => ({ ...pt }))])
              ),
              labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
            }
          : null,
      }));
    const inputs: Record<string, string> = {};
    for (const node of nodes) {
      if (node.role === "input") {
        inputs[node.id] = inputText[node.id] ?? "";
      }
    }
    copyBufferRef.current = { neurons: nodes, edges: copiedEdges, inputTexts: inputs, groups: copiedGroups };
    return true;
  }

  function pasteSelection(): boolean {
    const buffer = copyBufferRef.current;
    if (!buffer || !buffer.neurons.length) return false;
    if (isRestoringRef.current) return false;
    recordSnapshot();
    const offsetX = GRID_SIZE * 3;
    const offsetY = GRID_SIZE * 3;
    const minX = NODE_RADIUS;
    const minY = NODE_RADIUS;
    const maxX = Math.max(NODE_RADIUS, canvasWidth - NODE_RADIUS);
    const maxY = Math.max(NODE_RADIUS, canvasHeight - NODE_RADIUS);
    const takenLabels = new Set(neurons.map((n) => n.label));
    const idMap = new Map<string, string>();
    const newNeurons = buffer.neurons.map((node) => {
      const newId = uid("n");
      idMap.set(node.id, newId);
      const label = uniqueLabel(node.label, takenLabels);
      const x = clamp(snapToGrid(node.x + offsetX), minX, maxX);
      const y = clamp(snapToGrid(node.y + offsetY), minY, maxY);
      const labelVisible = node.labelVisible === true;
      const labelOffsetX = Number.isFinite(node.labelOffsetX) ? (node.labelOffsetX as number) : 0;
      const labelOffsetY = Number.isFinite(node.labelOffsetY)
        ? (node.labelOffsetY as number)
        : DEFAULT_LABEL_OFFSET_Y;
      return {
        id: newId,
        label,
        role: node.role,
        x,
        y,
        labelVisible,
        labelOffsetX,
        labelOffsetY,
      } as Neuron;
    });
    if (!newNeurons.length) return false;
    const edgeIdMap = new Map<string, string>();
    const newEdges = buffer.edges
      .map((edge) => {
        const sourceId = idMap.get(edge.sourceId);
        const targetId = idMap.get(edge.targetId);
        if (!sourceId || !targetId) return null;
        const transformed = sanitizeWaypoints(edge.points).map((pt) => ({
          x: clamp(snapToGrid(pt.x + offsetX), minX, maxX),
          y: clamp(snapToGrid(pt.y + offsetY), minY, maxY),
        }));
        const points = transformed.length ? transformed : undefined;
        const newEdgeId = uid("e");
        edgeIdMap.set(edge.id, newEdgeId);
        return { id: newEdgeId, sourceId, targetId, weight: edge.weight, points } as Edge;
      })
      .filter((edge): edge is Edge => !!edge);
    setNeurons((prev) => [...prev, ...newNeurons]);
    if (newEdges.length) {
      setEdges((prev) => [...prev, ...newEdges]);
    }
    const newIds = newNeurons.map((n) => n.id);
    setSelectedNodeIds(newIds);
    setSelectedEdgeId(null);
    setSelectedEdgeHandle(null);
    setActiveSelectionContext("main");
    setInputText((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const [oldId, text] of Object.entries(buffer.inputTexts)) {
        const mapped = idMap.get(oldId);
        if (mapped) {
          next[mapped] = text;
          changed = true;
        }
      }
      return changed ? next : prev;
    });

    const groupsToCopy = Array.isArray(buffer.groups) ? buffer.groups : [];
    if (groupsToCopy.length) {
      const groupPrototypes = groupsToCopy
        .map((g) => {
          const mapped = g.nodeIds
            .map((id) => idMap.get(id))
            .filter((id): id is string => typeof id === "string");
          if (mapped.length < 2) return null;
          return {
            nodeIds: mapped,
            hue: Number.isFinite(g.hue) ? g.hue : null,
            label: typeof g.label === "string" ? g.label : "",
            labelVisible: g.labelVisible === true,
            compression: g.compression
              ? {
                  center: { ...g.compression.center },
                  nodes: Object.fromEntries(
                    Object.entries(g.compression.nodes)
                      .map(([id, data]) => {
                        const mappedId = idMap.get(id);
                        if (!mappedId) return null;
                        return [mappedId, { ...data, x: data.x + offsetX, y: data.y + offsetY }];
                      })
                      .filter((entry): entry is [string, GroupCompressionState["nodes"][string]] => Boolean(entry))
                  ),
                  edgePoints: Object.fromEntries(
                    Object.entries(g.compression.edgePoints ?? {})
                      .map(([edgeId, points]) => {
                        const mappedEdgeId = edgeIdMap.get(edgeId);
                        if (!mappedEdgeId) return null;
                        const transformedPoints = points
                          .map<Point | null>((pt) => {
                            const px = clamp(snapToGrid(pt.x + offsetX), minX, maxX);
                            const py = clamp(snapToGrid(pt.y + offsetY), minY, maxY);
                            if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
                            return { x: px, y: py };
                          })
                          .filter((pt): pt is Point => pt !== null);
                        if (!transformedPoints.length) return null;
                        return [mappedEdgeId, transformedPoints];
                      })
                      .filter((entry): entry is [string, Point[]] => Boolean(entry))
                  ),
                  labelVisibleBefore: g.compression.labelVisibleBefore ?? (g.labelVisible === true),
                }
              : null,
          };
        })
        .filter(
          (g): g is {
            nodeIds: string[];
            hue: number | null;
            label: string;
            labelVisible: boolean;
            compression: {
              center: { x: number; y: number };
              nodes: Record<string, { x: number; y: number; labelOffsetX: number; labelOffsetY: number; labelVisible: boolean }>;
              edgePoints: Record<string, Point[]>;
              labelVisibleBefore: boolean;
            } | null;
          } => Boolean(g)
        );
      if (groupPrototypes.length) {
        setGroups((prev) => {
          const next = [...prev];
          for (const proto of groupPrototypes) {
            const hue = proto.hue ?? computeGroupHue(next);
            const label = nextGroupLabel(next, proto.label);
            next.push({
              id: uid("grp"),
              nodeIds: proto.nodeIds,
              hue,
              label,
              labelVisible: proto.labelVisible,
              compression:
                proto.compression && Object.keys(proto.compression.nodes).length >= 2
                  ? {
                      center: { x: proto.compression.center.x + offsetX, y: proto.compression.center.y + offsetY },
                      nodes: Object.fromEntries(
                        Object.entries(proto.compression.nodes).map(([id, data]) => [id, { ...data }])
                      ),
                      edgePoints: Object.fromEntries(
                        Object.entries(proto.compression.edgePoints ?? {}).map(([edgeId, points]) => [
                          edgeId,
                          points.map((pt) => ({ ...pt })),
                        ])
                      ),
                      labelVisibleBefore: proto.compression.labelVisibleBefore,
                    }
                  : undefined,
            });
          }
          return next;
        });
      }
    }
    return true;
  }

  useEffect(() => {
    function isEditable(target: EventTarget | null): boolean {
      const el = target as HTMLElement | null;
      if (!el) return false;
      if (el.isContentEditable) return true;
      return Boolean(el.closest("input, textarea, select, [contenteditable='true'], [contenteditable='']"));
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (isEditable(event.target)) return;
      const key = event.key.toLowerCase();
      if ((event.metaKey || event.ctrlKey) && !event.shiftKey && key === "z") {
        let handled = false;
        if (activeSelectionContext !== "module") {
          handled = undoCanvas();
        }
        if (handled) event.preventDefault();
      } else if (key === "arrowup" || key === "arrowdown" || key === "arrowleft" || key === "arrowright") {
        let handled = false;
        const step = event.shiftKey ? GRID_SIZE * 3 : GRID_SIZE;
        const dx = key === "arrowleft" ? -step : key === "arrowright" ? step : 0;
        const dy = key === "arrowup" ? -step : key === "arrowdown" ? step : 0;
        if (dx !== 0 || dy !== 0) {
          if (activeSelectionContext === "module") {
            handled = moveModuleSelectedNodesBy(dx, dy);
          } else {
            handled = moveSelectedNodesBy(dx, dy);
          }
        }
        if (handled) event.preventDefault();
      } else if ((event.metaKey || event.ctrlKey) && key === "c") {
        const handled =
          activeSelectionContext === "module" ? copyModuleSelection() : copySelection();
        if (handled) event.preventDefault();
      } else if ((event.metaKey || event.ctrlKey) && key === "v") {
        const handled =
          activeSelectionContext === "module" ? pasteModuleSelection() : pasteSelection();
        if (handled) event.preventDefault();
      } else if ((event.metaKey || event.ctrlKey) && key === "g" && !event.shiftKey) {
        let handled = false;
        if (activeSelectionContext !== "module") {
          handled = groupSelectionMain();
        }
        if (handled) event.preventDefault();
      } else if ((event.metaKey || event.ctrlKey) && event.shiftKey && key === "g") {
        let handled = false;
        if (activeSelectionContext !== "module") {
          handled = ungroupSelectionMain();
        }
        if (handled) event.preventDefault();
      } else if ((key === "delete" || key === "backspace") && !event.metaKey && !event.ctrlKey) {
        let handled = false;
        if (activeSelectionContext === "module") {
          if (removeModuleSelectedEdgeWaypoint()) handled = true;
          else if (moduleSelectedEdgeId || moduleSelectedNodeIds.length) {
            deleteModuleSelection();
            handled = true;
          }
        } else {
          if (event.shiftKey && collapseSelectedNeuron()) {
            handled = true;
          } else if (removeSelectedEdgeWaypoint()) {
            handled = true;
          } else if (selectedEdgeId || selectedNodeIds.length) {
            deleteSelected();
            handled = true;
          }
        }
        if (handled) event.preventDefault();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    activeSelectionContext,
    copyModuleSelection,
    copySelection,
    deleteModuleSelection,
    deleteSelected,
    collapseSelectedNeuron,
    moduleSelectedEdgeId,
    moduleSelectedNodeIds,
    pasteModuleSelection,
    pasteSelection,
    removeModuleSelectedEdgeWaypoint,
    removeSelectedEdgeWaypoint,
    selectedEdgeId,
    selectedNodeIds,
    groups,
    ungroupSelectionMain,
    undoCanvas,
    moveSelectedNodesBy,
    moveModuleSelectedNodesBy,
  ]);

  function exportState() {
    const groupPayload = groups.map((group) => {
      const compression = group.compression
        ? {
            center: { x: group.compression.center.x, y: group.compression.center.y },
            nodes: Object.fromEntries(
              Object.entries(group.compression.nodes).map(([nodeId, data]) => [
                nodeId,
                {
                  x: data.x,
                  y: data.y,
                  labelOffsetX: data.labelOffsetX,
                  labelOffsetY: data.labelOffsetY,
                  labelVisible: data.labelVisible,
                },
              ])
            ),
            edgePoints: Object.fromEntries(
              Object.entries(group.compression.edgePoints ?? {}).map(([edgeId, points]) => [
                edgeId,
                points.map((pt) => ({ x: pt.x, y: pt.y })),
              ])
            ),
            labelVisibleBefore: group.compression.labelVisibleBefore === true,
          }
        : undefined;

      return {
        id: group.id,
        nodeIds: [...group.nodeIds],
        hue: group.hue,
        label: group.label,
        labelVisible: group.labelVisible === true,
        compression,
      };
    });

    const payload = {
      version: 2,
      neurons,
      edges,
      groups: groupPayload,
      settings: {
        T,
        hKind,
        hVal,
      },
      canvas: {
        width: canvasWidth,
        height: canvasHeight,
      },
      inputs: inputText,
      dynamicsPanels: dynamicsPanels.map((panel) => ({
        id: panel.id,
        neuronId: panel.neuronId,
        x: panel.x,
        y: panel.y,
      })),
      dynamicsSettings: {
        showDynamicsHeaders,
        showOutputPotentials,
        includeColors,
        includeAnimationCounterInPdf,
      },
      modules: modules.map((module) => ({
        id: module.id,
        name: module.name,
        neurons: module.neurons,
        edges: module.edges,
        inputNeuronIds: module.inputNeuronIds,
        outputNeuronIds: module.outputNeuronIds,
        canvas: { ...module.canvas },
      })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = `rsnn-designer-${timestamp}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function triggerImport() {
    fileInputRef.current?.click();
  }

  async function importStateFile(file: File) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);

      let sanitizedNeurons: Neuron[] | null = null;
      let sanitizedGroups: Group[] = [];
      if (Array.isArray(data.neurons)) {
        const sanitizedList: Neuron[] = data.neurons
          .map((n: any) => {
            if (!n || typeof n !== "object") return null;
            const role: Role = n.role === "input" || n.role === "output" ? n.role : "hidden";
            const id = typeof n.id === "string" ? n.id : uid("n");
            const label = typeof n.label === "string" ? n.label : id.slice(-4);
            const x = Number(n.x);
            const y = Number(n.y);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            const labelVisible = n.labelVisible === true;
            const offsetX = Number(n.labelOffsetX);
            const offsetY = Number(n.labelOffsetY);
            return {
              id,
              label,
              role,
              x,
              y,
              labelVisible,
              labelOffsetX: Number.isFinite(offsetX) ? offsetX : 0,
              labelOffsetY: Number.isFinite(offsetY) ? offsetY : DEFAULT_LABEL_OFFSET_Y,
            } as Neuron;
          })
          .filter((n: Neuron | null): n is Neuron => n !== null);
        sanitizedNeurons = sanitizedList;
        setNeurons(sanitizedList);
      }

      if (Array.isArray(data.edges)) {
        const baseNeurons = sanitizedNeurons ?? neurons;
        const nodeIds = new Set(baseNeurons.map((n) => n.id));
        const sanitized: Edge[] = data.edges
          .map((e: any) => {
            if (!e || typeof e !== "object") return null;
            const sourceId = typeof e.sourceId === "string" ? e.sourceId : null;
            const targetId = typeof e.targetId === "string" ? e.targetId : null;
            if (!sourceId || !targetId) return null;
            if (!nodeIds.has(sourceId) || !nodeIds.has(targetId)) return null;
            const id = typeof e.id === "string" ? e.id : uid("e");
            const weight = Number(e.weight);
            if (!Number.isFinite(weight)) return null;
            const points = Array.isArray(e.points)
              ? e.points
                  .map((pt: any) => {
                    const px = Number(pt?.x);
                    const py = Number(pt?.y);
                    if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
                    return { x: px, y: py };
                  })
                  .filter((pt: { x: number; y: number } | null): pt is { x: number; y: number } => pt !== null)
              : undefined;
            return { id, sourceId, targetId, weight, points: points && points.length ? points : undefined } as Edge;
          })
          .filter((e: Edge | null): e is Edge => e !== null);
        setEdges(sanitized);
      }

      if (Array.isArray(data.groups)) {
        const baseNeurons = sanitizedNeurons ?? neurons;
        const neuronIds = new Set(baseNeurons.map((n) => n.id));
        sanitizedGroups = data.groups
          .map((g: any) => {
            if (!g || typeof g !== "object") return null;
            const nodeIdsRaw = Array.isArray(g.nodeIds) ? g.nodeIds.filter((id: any) => typeof id === "string" && neuronIds.has(id)) : [];
            if (nodeIdsRaw.length < 2) return null;
            const id = typeof g.id === "string" ? g.id : uid("grp");
            const hue = Number(g.hue);
            const label = typeof g.label === "string" ? g.label : "Group";
            const labelVisible = g.labelVisible === true;

            let compression: GroupCompressionState | undefined;
            const rawCompression = g.compression;
            if (rawCompression && typeof rawCompression === "object") {
              const centerX = Number(rawCompression.center?.x);
              const centerY = Number(rawCompression.center?.y);
              if (Number.isFinite(centerX) && Number.isFinite(centerY) && rawCompression.nodes && typeof rawCompression.nodes === "object") {
                const compressionNodes: GroupCompressionState["nodes"] = {};
                const compressionEdgePoints: Record<string, Point[]> = {};
                for (const [nodeId, nodeData] of Object.entries(rawCompression.nodes as Record<string, any>)) {
                  if (!neuronIds.has(nodeId)) continue;
                  if (!nodeData || typeof nodeData !== "object") continue;
                  const px = Number((nodeData as any).x);
                  const py = Number((nodeData as any).y);
                  const offsetX = Number((nodeData as any).labelOffsetX);
                  const offsetY = Number((nodeData as any).labelOffsetY);
                  if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(offsetX) || !Number.isFinite(offsetY)) {
                    continue;
                  }
                  compressionNodes[nodeId] = {
                    x: px,
                    y: py,
                    labelOffsetX: offsetX,
                    labelOffsetY: offsetY,
                    labelVisible: (nodeData as any).labelVisible === true,
                  };
                }
                if (Object.keys(compressionNodes).length >= 2) {
                  if (rawCompression.edgePoints && typeof rawCompression.edgePoints === "object") {
                    for (const [edgeId, value] of Object.entries(rawCompression.edgePoints as Record<string, any>)) {
                      if (typeof edgeId !== "string") continue;
                      if (!Array.isArray(value)) continue;
                      const sanitizedPoints = value
                        .map((pt: any) => {
                          const px = Number(pt?.x);
                          const py = Number(pt?.y);
                          if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
                          return { x: px, y: py };
                        })
                        .filter((pt: Point | null): pt is Point => pt !== null);
                      if (sanitizedPoints.length) {
                        compressionEdgePoints[edgeId] = sanitizedPoints;
                      }
                    }
                  }
                  compression = {
                    center: { x: centerX, y: centerY },
                    nodes: compressionNodes,
                    edgePoints: compressionEdgePoints,
                    labelVisibleBefore:
                      rawCompression.labelVisibleBefore === undefined
                        ? labelVisible
                        : rawCompression.labelVisibleBefore === true,
                  };
                }
              }
            }

            return {
              id,
              nodeIds: nodeIdsRaw,
              hue: Number.isFinite(hue) ? hue : 0,
              label,
              labelVisible,
              compression,
            } as Group;
          })
          .filter((g: Group | null): g is Group => g !== null);
      }

      const baseNeuronsForPanels = sanitizedNeurons ?? neurons;
      const neuronIdSetForPanels = new Set(baseNeuronsForPanels.map((n) => n.id));
      const sanitizedPanels = Array.isArray(data.dynamicsPanels)
        ? data.dynamicsPanels
            .map((panel: any) => {
              if (!panel || typeof panel !== "object") return null;
              const neuronId = typeof panel.neuronId === "string" ? panel.neuronId : null;
              if (!neuronId || !neuronIdSetForPanels.has(neuronId)) return null;
              const x = Number(panel.x);
              const y = Number(panel.y);
              if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
              const id = typeof panel.id === "string" ? panel.id : uid("dyn");
              return { id, neuronId, x, y };
            })
            .filter((panel: { id: string; neuronId: string; x: number; y: number } | null): panel is { id: string; neuronId: string; x: number; y: number } => panel !== null)
        : [];

      // Restore groups and dynamics panels from import
      setGroups(sanitizedGroups);
      setDynamicsPanels(sanitizedPanels);

      if (data && typeof data.dynamicsSettings === "object") {
        const dynamicsSettings = data.dynamicsSettings as any;
        const showHeadersValue = dynamicsSettings.showDynamicsHeaders;
        if (typeof showHeadersValue === "boolean") {
          setShowDynamicsHeaders(showHeadersValue);
        } else {
          setShowDynamicsHeaders(true);
        }
        const showOutputValue = dynamicsSettings.showOutputPotentials;
        if (typeof showOutputValue === "boolean") {
          setShowOutputPotentials(showOutputValue);
        } else {
          const legacyShowIoValue = dynamicsSettings.showIoPotentials;
          setShowOutputPotentials(legacyShowIoValue === true);
        }
        const includeColorsValue = dynamicsSettings.includeColors;
        if (typeof includeColorsValue === "boolean") {
          setIncludeColors(includeColorsValue);
        } else {
          setIncludeColors(true);
        }
        const includeCounterValue = dynamicsSettings.includeAnimationCounterInPdf;
        if (typeof includeCounterValue === "boolean") {
          setIncludeAnimationCounterInPdf(includeCounterValue);
        } else {
          setIncludeAnimationCounterInPdf(false);
        }
      } else {
        setShowDynamicsHeaders(true);
        setShowOutputPotentials(false);
        setIncludeColors(true);
        setIncludeAnimationCounterInPdf(false);
      }

      if (data && typeof data.inputs === "object") {
        const entries = Object.entries(data.inputs as Record<string, unknown>).map(([k, v]) => [k, typeof v === "string" ? v : ""]);
        setInputText(Object.fromEntries(entries));
      }

      if (data && typeof data.settings === "object") {
        if (typeof data.settings.T === "number") setT(data.settings.T);
        if (data.settings.hKind === "finite" || data.settings.hKind === "zero" || data.settings.hKind === "infty") {
          setHKind(data.settings.hKind);
        }
        if (typeof data.settings.hVal === "number") setHVal(data.settings.hVal);
      }

      if (data && typeof data.canvas === "object") {
        if (typeof data.canvas.width === "number") setCanvasWidth(Math.max(200, data.canvas.width));
        if (typeof data.canvas.height === "number") setCanvasHeight(Math.max(200, data.canvas.height));
      }

      let importedModules: Module[] = [];
      if (Array.isArray(data.modules)) {
        importedModules = data.modules
          .map((m: any) => {
            if (!m || typeof m !== "object") return null;
            const id = typeof m.id === "string" ? m.id : uid("module");
            const name = typeof m.name === "string" ? m.name : "Module";
            const rawNeurons = Array.isArray(m.neurons) ? m.neurons : [];
            const neuronList: Neuron[] = rawNeurons
              .map((n: any) => {
                if (!n || typeof n !== "object") return null;
                const role: Role = n.role === "input" || n.role === "output" ? n.role : "hidden";
                const nid = typeof n.id === "string" ? n.id : uid("n");
                const label = typeof n.label === "string" ? n.label : nid.slice(-4);
                const x = Number(n.x);
                const y = Number(n.y);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
                const labelVisible = n.labelVisible === true;
                const offsetX = Number(n.labelOffsetX);
                const offsetY = Number(n.labelOffsetY);
                return {
                  id: nid,
                  label,
                  role,
                  x,
                  y,
                  labelVisible,
                  labelOffsetX: Number.isFinite(offsetX) ? offsetX : 0,
                  labelOffsetY: Number.isFinite(offsetY) ? offsetY : DEFAULT_LABEL_OFFSET_Y,
                } as Neuron;
              })
              .filter((n: Neuron | null): n is Neuron => n !== null);
            const neuronIds = new Set(neuronList.map((n) => n.id));
            const rawEdges = Array.isArray(m.edges) ? m.edges : [];
            const edgeList: Edge[] = rawEdges
              .map((e: any) => {
                if (!e || typeof e !== "object") return null;
                const sourceId = typeof e.sourceId === "string" ? e.sourceId : null;
                const targetId = typeof e.targetId === "string" ? e.targetId : null;
                if (!sourceId || !targetId) return null;
                if (!neuronIds.has(sourceId) || !neuronIds.has(targetId)) return null;
                const edgeId = typeof e.id === "string" ? e.id : uid("e");
                const weight = Number(e.weight);
                if (!Number.isFinite(weight)) return null;
                const points = Array.isArray(e.points)
                  ? e.points
                      .map((pt: any) => {
                        const px = Number(pt?.x);
                        const py = Number(pt?.y);
                        if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
                        return { x: px, y: py };
                      })
                      .filter((pt: { x: number; y: number } | null): pt is { x: number; y: number } => pt !== null)
                  : undefined;
                return { id: edgeId, sourceId, targetId, weight, points: points && points.length ? points : undefined } as Edge;
              })
              .filter((e: Edge | null): e is Edge => e !== null);
            const inputIds = Array.isArray(m.inputNeuronIds)
              ? m.inputNeuronIds.filter((nid: any) => typeof nid === "string" && neuronIds.has(nid))
              : [];
            const outputIds = Array.isArray(m.outputNeuronIds)
              ? m.outputNeuronIds.filter((nid: any) => typeof nid === "string" && neuronIds.has(nid))
              : [];
            const canvasWidthModule =
              m.canvas && typeof m.canvas.width === "number" ? Math.max(200, m.canvas.width) : DEFAULT_MACRO_WIDTH;
            const canvasHeightModule =
              m.canvas && typeof m.canvas.height === "number" ? Math.max(200, m.canvas.height) : DEFAULT_MACRO_HEIGHT;
            return {
              id,
              name,
              neurons: neuronList,
              edges: edgeList,
              inputNeuronIds: inputIds,
              outputNeuronIds: outputIds,
              canvas: { width: canvasWidthModule, height: canvasHeightModule },
            } as Module;
          })
          .filter((module: Module | null): module is Module => module !== null);
      }

      if (importedModules.length) {
        setModules((prevModules) => {
          if (!importedModules.length) return prevModules;
          const existingIds = new Set(prevModules.map((module) => module.id));
          const existingNames = new Set(prevModules.map((module) => module.name));
          const appended = importedModules.map((module) => {
            let nextId = module.id;
            while (existingIds.has(nextId)) {
              nextId = uid("module");
            }
            existingIds.add(nextId);

            const baseName = (module.name || "").trim() || "Module";
            const nextName = uniqueLabel(baseName, existingNames);
            return {
              ...module,
              id: nextId,
              name: nextName,
            };
          });
          if (!appended.length) return prevModules;
          return [...prevModules, ...appended];
        });
      }
      openModuleEditor(null);

      setSelectedNodeIds([]);
      setSelectedEdgeId(null);
      setSim(null);
      setMode("select");
      setConnectSrc(null);
    } catch (err) {
      console.error("Failed to import RSNN state", err);
      alert("Failed to import file. Please ensure it is a valid RSNN Designer export.");
    }
  }

  function onImportInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      importStateFile(file).finally(() => {
        e.target.value = "";
      });
    }
  }

  async function exportAnimationPdfs() {
    if (isExportingAnimationPdfs) return;
    const frames = animationFramesRef.current;
    if (!frames.length) {
      alert("Run an animation before exporting frame PDFs.");
      return;
    }
    if (typeof window.showDirectoryPicker !== "function") {
      alert("Your browser does not support selecting folders for export.");
      return;
    }
    const container = canvasExportRef.current;
    if (!container) {
      alert("Canvas view unavailable. Please try again.");
      return;
    }
    if (!(container.querySelector("svg") instanceof SVGSVGElement)) {
      alert("Canvas SVG could not be located.");
      return;
    }
    setIsExportingAnimationPdfs(true);
    const exportWidth = Math.max(200, canvasWidth);
    const exportHeight = Math.max(200, canvasHeight);
    const wasAnimating = isAnimatingRef.current;
    const previousIndex = animationIndexRef.current;
    pauseAnimation();

    const formatTimeToken = (time: number) => {
      return time.toFixed(3).replace(/\.?0+$/, "").replace(/\./g, "p") || "0";
    };

    try {
      const directoryHandle = await window.showDirectoryPicker({ mode: "readwrite" });
      if (!directoryHandle) {
        return;
      }
      let exportedCount = 0;
      for (let i = 0; i < frames.length; i++) {
        const frame = frames[i];
        displayFrame(i);
        await waitForNextFrame();

        const currentSvg = container.querySelector("svg");
        if (!(currentSvg instanceof SVGSVGElement)) {
          continue;
        }
        const clone = prepareCanvasSvgClone(currentSvg, exportWidth, exportHeight);

        const pdf = new jsPDF({
          orientation: exportWidth >= exportHeight ? "landscape" : "portrait",
          unit: "px",
          format: [exportWidth, exportHeight],
        });
        pdf.setFillColor(255, 255, 255);
        pdf.rect(0, 0, exportWidth, exportHeight, "F");
        await pdf.svg(clone, {
          x: 0,
          y: 0,
          width: exportWidth,
          height: exportHeight,
        });

        if (includeAnimationCounterInPdf) {
          const overlayParts = [
            `t = ${frame.time.toFixed(3)}s`,
            `Step ${i + 1}/${frames.length}`,
            `Wave ${frame.wave + 1}`,
          ];
          const overlayText = overlayParts.join(" • ");
          const fontSize = 14;
          pdf.setFontSize(fontSize);
          const paddingX = 18;
          const paddingY = 10;
          const textWidth = pdf.getTextWidth(overlayText);
          const overlayHeight = fontSize + paddingY * 2;
          const overlayWidth = textWidth + paddingX * 2;
          const overlayX = Math.max(16, exportWidth - overlayWidth - 16);
          const overlayY = exportHeight - overlayHeight - 16;
          pdf.setFillColor(0, 0, 0);
          pdf.roundedRect(overlayX, overlayY, overlayWidth, overlayHeight, 12, 12, "F");
          pdf.setTextColor(255, 255, 255);
          pdf.text(overlayText, overlayX + paddingX, overlayY + overlayHeight / 2, {
            baseline: "middle",
          });
        }

        const stepToken = String(i + 1).padStart(3, "0");
        const timeToken = formatTimeToken(frame.time);
        const waveToken = String(frame.wave + 1).padStart(2, "0");
        const fileName = `steps_${stepToken}_time_${timeToken}_wave_${waveToken}.pdf`;

        const fileHandle = await directoryHandle.getFileHandle(fileName, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(pdf.output("arraybuffer"));
        await writable.close();
        exportedCount += 1;
      }
      if (exportedCount > 0) {
        alert(`Exported ${exportedCount} PDF${exportedCount === 1 ? "" : "s"} to "${directoryHandle.name}".`);
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        console.error(error);
        alert("Failed to export animation PDFs. Please try again.");
      }
    } finally {
      if (previousIndex >= 0) {
        displayFrame(previousIndex);
      } else {
        displayNoFrame();
      }
      if (wasAnimating) {
        resumeAnimation();
      }
      setIsExportingAnimationPdfs(false);
    }
  }

  function exportCanvasPdf() {
    const container = canvasExportRef.current;
    if (!container) {
      alert("Canvas view unavailable. Please try again.");
      return;
    }
    const svgElement = container.querySelector("svg");
    if (!(svgElement instanceof SVGElement)) {
      alert("Canvas SVG could not be located.");
      return;
    }

    const clone = svgElement.cloneNode(true) as SVGSVGElement;
    const exportWidth = Math.max(200, canvasWidth);
    const exportHeight = Math.max(200, canvasHeight);

    clone.querySelectorAll<SVGRectElement>('rect[fill^="url("]').forEach((rect) => {
      const fill = rect.getAttribute("fill");
      if (fill && fill.includes("-grid")) {
        rect.setAttribute("fill", "#ffffff");
      }
    });

    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.removeAttribute("style");
    clone.setAttribute("width", `${exportWidth}`);
    clone.setAttribute("height", `${exportHeight}`);
    clone.style.width = `${exportWidth}px`;
    clone.style.height = `${exportHeight}px`;

    const serializedSvg = clone.outerHTML;
    const showCounterOverlay = includeAnimationCounterInPdf && hasAnimation;
    const overlayTextParts: string[] = [];
    if (showCounterOverlay) {
      overlayTextParts.push(`t = ${formattedClock}s`);
      overlayTextParts.push(`Step ${frameIndexDisplay}/${frameTotal}`);
      if (currentFrame) {
        overlayTextParts.push(`Wave ${currentWaveDisplay}`);
      }
    }
    const overlayHtml = showCounterOverlay ? `<div class="overlay">${overlayTextParts.join(" • ")}</div>` : "";

    const html = `<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>RSNN Canvas Export</title>
    <style>
      :root { color-scheme: only light; }
      * { box-sizing: border-box; }
      html, body { margin: 0; padding: 0; }
      body { background: #ffffff; width: ${exportWidth}px; height: ${exportHeight}px; display: flex; align-items: center; justify-content: center; font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      .export-shell { position: relative; width: ${exportWidth}px; height: ${exportHeight}px; }
      .export-shell svg { width: 100%; height: 100%; display: block; }
      .export-shell .overlay { position: absolute; right: 16px; bottom: 16px; background: rgba(0, 0, 0, 0.75); color: #ffffff; font-size: 12px; padding: 6px 12px; border-radius: 9999px; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25); font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      svg, svg * { font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; }
      @page { size: ${exportWidth}px ${exportHeight}px; margin: 0; }
    </style>
  </head>
  <body>
    <div class="export-shell">
      ${serializedSvg}
      ${overlayHtml}
    </div>
    <script>
      window.addEventListener('load', function () {
        setTimeout(function () {
          window.focus();
          window.print();
        }, 100);
        window.addEventListener('afterprint', function () {
          window.close();
        }, { once: true });
      });
    </script>
  </body>
</html>`;

    const popupWidth = Math.max(640, exportWidth + 80);
    const popupHeight = Math.max(480, exportHeight + 120);
    const exportWindow = window.open("", "_blank", `width=${popupWidth},height=${popupHeight},resizable=yes,scrollbars=yes`);
    if (!exportWindow) {
      alert("Unable to open export window. Please allow pop-ups for this site.");
      return;
    }
    exportWindow.document.open();
    exportWindow.document.write(html);
    exportWindow.document.close();
    exportWindow.focus();
  }

  function exportPdf() {
    const canvasHtml = canvasExportRef.current?.innerHTML?.trim() || "<p>Canvas view unavailable.</p>";
    const rasterHtml = rasterExportRef.current?.innerHTML?.trim() || "<p>Raster view unavailable.</p>";
    const potentialHtml = potentialExportRef.current?.innerHTML?.trim() || "<p>Potential view unavailable.</p>";

    const timestamp = new Date().toLocaleString();
    const roleCounts = neurons.reduce(
      (acc, n) => {
        acc[n.role] += 1;
        return acc;
      },
      { input: 0, hidden: 0, output: 0 } as Record<Role, number>
    );
    const hLabel = hKind === "infty" ? "∞" : hKind === "zero" ? "0" : h.toFixed(3);

    const html = `<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>RSNN Designer Report</title>
    <style>
      :root { color-scheme: only light; }
      body { font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }
      h1 { font-size: 24px; margin: 0 0 8px 0; }
      h2 { font-size: 18px; margin: 32px 0 12px; }
      p { margin: 4px 0; }
      .meta { font-size: 14px; margin-bottom: 16px; }
      .meta p { margin: 2px 0; }
      .panel { margin-bottom: 28px; page-break-inside: avoid; }
      .panel > div { border: 1px solid #d1d5db; border-radius: 12px; padding: 16px; background: #ffffff; }
      svg { width: 100%; height: auto; }
      .footer { margin-top: 40px; font-size: 12px; color: #6b7280; }
    </style>
  </head>
  <body>
    <header>
      <h1>RSNN Designer Diagnostics</h1>
      <div class="meta">
        <p><strong>Generated:</strong> ${timestamp}</p>
        <p><strong>Horizon T:</strong> ${T}</p>
        <p><strong>Memory h:</strong> ${hLabel}</p>
        <p><strong>Canvas:</strong> ${canvasWidth} × ${canvasHeight}</p>
        <p><strong>Neurons:</strong> ${neurons.length} (input ${roleCounts.input}, hidden ${roleCounts.hidden}, output ${roleCounts.output})</p>
        <p><strong>Edges:</strong> ${edges.length}</p>
      </div>
    </header>

    <section class="panel">
      <h2>Network Canvas</h2>
      <div class="canvas-wrapper">${canvasHtml}</div>
    </section>

    <section class="panel">
      <h2>Spike Raster</h2>
      <div class="raster-wrapper">${rasterHtml}</div>
    </section>

    <section class="panel">
      <h2>Potential Traces</h2>
      <div class="potential-wrapper">${potentialHtml}</div>
    </section>

    <div class="footer">Use your browser's print dialog to save this report as PDF.</div>
  </body>
</html>`;

    const reportWindow = window.open("", "_blank", "width=1000,height=800");
    if (!reportWindow) {
      alert("Unable to open report window. Please allow pop-ups for this site.");
      return;
    }
    reportWindow.document.open();
    reportWindow.document.write(html);
    reportWindow.document.close();
    reportWindow.focus();
    setTimeout(() => {
      try {
        reportWindow.print();
      } catch (err) {
        console.error("Print failed", err);
      }
    }, 300);
  }

  const controlsPanel = (
    <div className="space-y-4">
      <div className="p-3 rounded-2xl border shadow-sm bg-white space-y-2">
        <div className="text-lg font-semibold">Simulation</div>
        <div className="grid grid-cols-3 items-center gap-2">
          <label className="col-span-1">Horizon T</label>
          <input
            className="col-span-2 px-2 py-1 border rounded-lg"
            type="number"
            min={0.1}
            step={0.1}
            value={T}
            onChange={(e) => setT(Number(e.target.value))}
          />

          <label className="col-span-1">Memory h</label>
          <div className="col-span-2 flex items-center gap-2">
            <select className="px-2 py-1 border rounded-lg" value={hKind} onChange={(e) => setHKind(e.target.value as any)}>
              <option value="finite">finite</option>
              <option value="zero">0</option>
              <option value="infty">∞</option>
            </select>
            {hKind === "finite" && (
              <input
                className="px-2 py-1 border rounded-lg w-24"
                type="number"
                min={0}
                step={0.1}
                value={hVal}
                onChange={(e) => setHVal(Number(e.target.value))}
              />
            )}
          </div>

          <div className="col-span-3 flex gap-2">
            <button className="px-3 py-1 rounded-full border bg-black text-white" onClick={runSimulation}>
              Run
            </button>
            <button className="px-3 py-1 rounded-full border" onClick={animateButtonAction} title={animateButtonTitle}>
              {animateButtonLabel}
            </button>
            {hasAnimation && (
              <button className="px-3 py-1 rounded-full border" onClick={stopAnimation}>
                {isAnimating ? "Stop" : "Hide"}
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="p-3 rounded-2xl border shadow-sm bg-white space-y-2">
        <div className="text-lg font-semibold">Canvas Layout</div>
        <div className="grid grid-cols-3 items-center gap-2">
          <label className="col-span-1">Width</label>
          <input
            className="col-span-2 px-2 py-1 border rounded-lg"
            type="number"
            min={MIN_CANVAS_DIMENSION}
            step={50}
            value={canvasWidthInput}
            onChange={(e) => setCanvasWidthInput(e.target.value)}
            onKeyDown={handleCanvasSizeKeyDown}
          />

          <label className="col-span-1">Height</label>
          <input
            className="col-span-2 px-2 py-1 border rounded-lg"
            type="number"
            min={MIN_CANVAS_DIMENSION}
            step={50}
            value={canvasHeightInput}
            onChange={(e) => setCanvasHeightInput(e.target.value)}
            onKeyDown={handleCanvasSizeKeyDown}
          />
          <button type="button" className="col-span-3 px-3 py-1 rounded-full border" onClick={applyCanvasSize} disabled={!isCanvasSizeDirty}>
            Apply
          </button>
        </div>
      </div>

      <div className="p-3 rounded-2xl border shadow-sm bg-white space-y-2">
        <div className="text-lg font-semibold">File I/O</div>
        <div className="flex flex-wrap gap-2">
          <button className="px-3 py-1 rounded-full border" onClick={exportCanvasPdf}>
            Export Canvas as PDF
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={exportAnimationPdfs}
            disabled={!hasAnimation || isExportingAnimationPdfs}
            title={
              hasAnimation
                ? isExportingAnimationPdfs
                  ? "Generating PDFs for each animation step..."
                  : "Select a folder to export a PDF for each animation frame."
                : "Run a spike animation to enable per-frame PDF export."
            }
          >
            {isExportingAnimationPdfs ? "Exporting..." : "Export Animation PDFs"}
          </button>
          <button className="px-3 py-1 rounded-full border" onClick={exportState}>
            Export JSON
          </button>
          <button className="px-3 py-1 rounded-full border" onClick={triggerImport}>
            Import JSON
          </button>
          <input ref={fileInputRef} type="file" accept="application/json" className="hidden" onChange={onImportInputChange} />
        </div>
        <label
          className={`mt-2 flex items-center gap-2 text-xs text-gray-600 ${hasAnimation ? "" : "opacity-60"}`}
          title={
            hasAnimation
              ? "Toggle whether the animation counter overlay is included in PDF canvas exports."
              : "Run a simulation with spike animation to enable the counter overlay in PDF exports."
          }
        >
          <input
            type="checkbox"
            className="accent-black"
            checked={includeAnimationCounterInPdf}
            disabled={!hasAnimation}
            onChange={(e) => setIncludeAnimationCounterInPdf(e.target.checked)}
          />
          Include animation counter overlay in PDF
        </label>
      </div>

      <div className="p-3 rounded-2xl border shadow-sm bg-white space-y-2">
        <div className="text-lg font-semibold">Input Spike Trains</div>
        {neurons.filter((n) => n.role === "input").length === 0 && <div className="text-gray-500">Add an Input neuron to edit trains.</div>}
        <div className="space-y-3 max-h-72 overflow-auto pr-1">
          {neurons
            .filter((n) => n.role === "input")
            .map((n) => (
              <div key={n.id} className="border rounded-xl p-2">
                <div className="flex items-center gap-2 mb-1">
                  <div className="font-medium">{n.label}</div>
                  <div className="text-xs text-gray-500">({n.id.slice(-4)})</div>
                </div>
                <label className="text-xs text-gray-600">times (s): space/comma separated</label>
                <textarea
                  className="w-full mt-1 p-2 border rounded-lg font-mono text-xs"
                  rows={2}
                  placeholder="e.g., 0.5 1 1.5 3 4.2"
                  value={inputText[n.id] || ""}
                  onChange={(e) => setInputText((prev) => ({ ...prev, [n.id]: e.target.value }))}
                />
                <div className="flex items-center gap-2 mt-1">
                  <button
                    className="px-2 py-1 border rounded-full"
                    onClick={() => setInputText((prev) => ({ ...prev, [n.id]: jitterPoissonText(T, 3) }))}
                    title="Fill with a Poisson(λ=3 Hz) sample over [0,T]"
                  >
                    Poisson λ=3
                  </button>
                  <button
                    className="px-2 py-1 border rounded-full"
                    onClick={() => setInputText((prev) => ({ ...prev, [n.id]: integerSpikeText(T) }))}
                    title="Fill with spikes at integer seconds up to T"
                  >
                    Integer times
                  </button>
                  <button className="px-2 py-1 border rounded-full" onClick={() => setInputText((prev) => ({ ...prev, [n.id]: "" }))}>
                    Clear
                  </button>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );

  const modulesSection = (
    <>
      <div className="border-t border-gray-200 my-6" />
      <div className="space-y-4">
        <div className="p-3 rounded-2xl border shadow-sm bg-white space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-lg font-semibold">Modules</div>
            <button className="px-3 py-1 rounded-full border" onClick={createModule}>
              + New Module
            </button>
          </div>
          {modules.length === 0 ? (
            <div className="text-gray-500 text-sm">Create a module to reuse neuron motifs on the main canvas.</div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2">
              {modules.map((module) => {
                const isActive = module.id === activeModuleId;
                return (
                  <div key={module.id} className={`border rounded-xl p-3 flex flex-col gap-3 ${isActive ? "border-black" : "border-gray-200"}`}>
                    <div className="flex items-start gap-3">
                      <div className="flex-1 min-w-0 space-y-2">
                        <input className="w-full px-2 py-1 border rounded-lg" value={module.name} onChange={(e) => renameModule(module.id, e.target.value)} />
                        <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-gray-500">
                          <span>{module.neurons.length} neurons</span>
                          <span>{module.edges.length} edges</span>
                          <span>{module.inputNeuronIds.length} inputs</span>
                          <span>{module.outputNeuronIds.length} outputs</span>
                        </div>
                      </div>
                      {isActive && <span className="shrink-0 rounded-full bg-black px-2 py-0.5 text-xs font-medium text-white">Editing</span>}
                    </div>
                    <div className="flex flex-wrap gap-2 border-t pt-2">
                      <button
                        className={`px-3 py-1 rounded-full border ${isActive ? "bg-black text-white" : ""}`}
                        onClick={() => openModuleEditor(module)}
                      >
                        {isActive ? "Open Editor" : "Edit"}
                      </button>
                      <button className="px-3 py-1 rounded-full border" onClick={() => instantiateModule(module)}>
                        Place on Board
                      </button>
                      <button className="px-3 py-1 rounded-full border" onClick={() => deleteModuleById(module.id)}>
                        Delete
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {activeModuleId && (
          <div className="p-3 rounded-2xl border shadow-sm bg-white space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-lg font-semibold">Module Editor{activeModule ? ` — ${activeModule.name}` : ""}</div>
              <div className="flex gap-2">
                <button
                  className={`px-3 py-1 rounded-full border ${moduleMode === "select" ? "bg-black text-white" : ""}`}
                  onClick={() => {
                    setModuleMode("select");
                    cancelModuleConnect();
                  }}
                >
                  Select
                </button>
                <button
                  className={`px-3 py-1 rounded-full border ${moduleMode === "connect" ? "bg-black text-white" : ""}`}
                  onClick={() => {
                    setModuleMode("connect");
                    setModuleConnectSrc(null);
                    setModuleConnectPoints([]);
                  }}
                >
                  Connect
                </button>
                <button className="px-3 py-1 rounded-full border" onClick={() => openModuleEditor(null)}>
                  Close
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <button className="px-3 py-1 rounded-full border" onClick={() => addModuleNeuron("hidden")}>
                + Hidden
              </button>
              <button className="px-3 py-1 rounded-full border" onClick={() => addModuleNeuron("input")}>
                + Input
              </button>
              <button className="px-3 py-1 rounded-full border" onClick={() => addModuleNeuron("output")}>
                + Output
              </button>
              <button
                className="px-3 py-1 rounded-full border"
                onClick={() => deleteModuleSelection()}
                disabled={!moduleSelectedEdgeId && moduleSelectedNodeIds.length === 0}
              >
                Delete Selected
              </button>
            </div>

            <div className="grid grid-cols-12 gap-4">
              <div className="col-span-7 lg:col-span-8 space-y-2">
                <CanvasView
                  neurons={moduleNeurons}
                  edges={moduleEdges}
                  selectedNodeIds={moduleSelectedNodeIds}
                  selectedEdgeId={moduleSelectedEdgeId}
                  canvasWidth={moduleCanvasWidth}
                  canvasHeight={moduleCanvasHeight}
                  mode={moduleMode}
                  connectSrc={moduleConnectSrc}
                  connectPoints={moduleConnectPoints}
                  groups={[]}
                  dynamicsPanels={[]}
                  dynamicsData={{ sim: null, T: T }}
                  includeColors={includeColors}
                  stickyMultiSelect={touchMultiSelect}
                  onSelectGroup={(group, _modifiers) => []}
                  onNodePointerDown={handleModuleNodePointerDown}
                  onSelectEdge={handleModuleEdgePointerDown}
                  onClearSelection={clearModuleSelection}
                  onMoveNodes={moveModuleNodes}
                  onMoveLabels={moveModuleLabelOffsets}
                  hideStandardWeights={hideStandardWeights}
                  onBeginDrag={(_ids) => {}}
                  onBeginLabelDrag={(_id) => {}}
                  onBeginEdgeDrag={(_edgeId, _index) => {}}
                  onEndDrag={() => {}}
                  onMarqueeSelect={() => {}}
                  onCompressGroup={() => {}}
                  onExpandGroup={() => {}}
                  onMovePanel={() => {}}
                  onClosePanel={() => {}}
                  onBeginConnect={beginModuleConnect}
                  onCompleteConnect={completeModuleConnect}
                  onAddWaypoint={addModuleWaypoint}
                  onCancelConnect={cancelModuleConnect}
                  activeEdgeHandle={moduleSelectedEdgeHandle}
                  onSelectEdgeHandle={(edgeId, index) => {
                    if (edgeId === null || index === null) {
                      setModuleSelectedEdgeHandle(null);
                    } else {
                      setModuleSelectedEdgeHandle({ edgeId, index });
                      setActiveSelectionContext("module");
                    }
                  }}
                  onMoveEdgePoint={updateModuleEdgeWaypoint}
                  onInsertEdgePoint={insertModuleEdgeWaypoint}
                  onEditEdge={setModuleSelectedEdgeId}
                  highlightedNodeIds={[]}
                  highlightedEdgeIds={[]}
                />
                <EdgeEditor edge={moduleEdges.find((e) => e.id === moduleSelectedEdgeId) || null} onChange={setModuleWeight} onRemove={removeModuleEdge} />
              </div>
              <div className="col-span-5 lg:col-span-4 space-y-3">
                <div className="p-3 rounded-xl border bg-white space-y-2">
                  <div className="text-lg font-semibold">Module Canvas</div>
                  <div className="grid grid-cols-3 items-center gap-2">
                    <label className="col-span-1">Width</label>
                    <input
                      className="col-span-2 px-2 py-1 border rounded-lg"
                      type="number"
                      min={200}
                      step={20}
                      value={moduleCanvasWidth}
                      onChange={(e) => setModuleCanvasWidth(Math.max(200, Number(e.target.value)))}
                    />
                    <label className="col-span-1">Height</label>
                    <input
                      className="col-span-2 px-2 py-1 border rounded-lg"
                      type="number"
                      min={200}
                      step={20}
                      value={moduleCanvasHeight}
                      onChange={(e) => setModuleCanvasHeight(Math.max(200, Number(e.target.value)))}
                    />
                  </div>
                </div>

                <ModuleNeuronInspector
                  neurons={moduleNeurons}
                  selectedIds={moduleSelectedNodeIds}
                  inputIds={moduleInputIds}
                  outputIds={moduleOutputIds}
                  onLabel={setModuleLabel}
                  onRole={setModuleRole}
                  onInput={setModuleInput}
                  onOutput={setModuleOutput}
                  onLabelVisible={setModuleLabelVisibility}
                  onLabelOffset={setModuleLabelOffset}
                  onResetLabelOffset={resetModuleLabelOffset}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );

  const spikeRasterSection = (
    <div className="p-3 rounded-2xl border shadow-sm bg-white">
      <div className="flex items-center justify-between mb-2">
        <div className="text-lg font-semibold">Spike Raster</div>
        <div className="text-xs text-gray-500">
          Strict threshold {">"} 1; h = {hKind === "infty" ? "∞" : hKind === "zero" ? 0 : h.toFixed(3)}
        </div>
      </div>
      {!sim && <div className="text-gray-500 text-sm mt-2">Run the simulation to see spikes.</div>}
      <div ref={rasterExportRef}>
        <RasterPlot neurons={neurons} spikeTrains={sim?.spikeTrains || {}} T={T} selectedNeuronIds={selectedNodeIds} />
      </div>
    </div>
  );

  return (
    <div
      className="w-full text-sm"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          cancelConnect();
          cancelModuleConnect();
        }
      }}
    >
      <div className="flex flex-col gap-3 page-shell max-w-6xl mx-auto">
        <header className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-bold min-w-0">Spiking Neural Network Simulator      </h1>
          {onToggleTheme ? (
            <div className="theme-toggle-wrapper">
              <button className="theme-toggle-button" onClick={onToggleTheme} type="button">
                {isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
              </button>
            </div>
          ) : null}
          <span className="ml-auto" />
          <span className="ml-auto" />
          <button
            className={`px-3 py-1 rounded-full border transition ${
              mode === "select" ? "bg-black text-white border-black" : "bg-white text-gray-700 border-gray-300"
            }`}
            onClick={() => {
              exitConnectMode();
            }}
            title="Select / move nodes"
          >
            Select
          </button>
          <button
            className={`px-3 py-1 rounded-full border transition ${
              mode === "connect" ? "bg-black text-white border-black" : "bg-white text-gray-700 border-gray-300"
            }`}
            onClick={() => {
              setMode("connect");
              setConnectSrc(null);
              setConnectPoints([]);
            }}
            title="Connect nodes"
          >
            Connect
          </button>
          <button
            className={`px-3 py-1 rounded-full border transition ${
              isCanvasFullScreen ? "bg-black text-white border-black" : "bg-white text-gray-700 border-gray-300"
            }`}
            onClick={() => setIsCanvasFullScreen((prev) => !prev)}
            title={
              isCanvasFullScreen
                ? "Restore split layout with controls beside the canvas"
                : "Expand canvas and move controls below"
            }
          >
            {isCanvasFullScreen ? "Full Screen" : "Full Screen"}
          </button>
          <button
            className={`px-3 py-1 rounded-full border transition ${
              hideStandardWeights ? "bg-black text-white border-black" : "bg-white text-gray-700 border-gray-300"
            }`}
            onClick={() => setHideStandardWeights((prev) => !prev)}
            title="Toggle visibility of weights equal to 1, 2, or -2"
          >
            {hideStandardWeights ? "Show Std Weights" : "Hide Std Weights"}
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => undoCanvas()}
            disabled={!canUndo}
            title="Undo (Ctrl/Cmd+Z)"
          >
            Undo
          </button>
          <label className="ml-2 flex items-center gap-2 text-xs text-gray-600">
            Default weight for connections: 
            <input
              className="w-20 px-2 py-0.6 border rounded-lg text-sm"
              type="number"
              step={0.1}
              value={defaultEdgeWeightInput}
              onChange={(e) => {
                const next = e.target.value;
                setDefaultEdgeWeightInput(next);
                const parsed = Number(next);
                if (Number.isFinite(parsed)) {
                  setDefaultEdgeWeight(parsed);
                }
              }}
              onBlur={() => {
                const parsed = Number(defaultEdgeWeightInput);
                if (Number.isFinite(parsed)) {
                  const normalized = parsed;
                  setDefaultEdgeWeight(normalized);
                  setDefaultEdgeWeightInput(String(normalized));
                } else {
                  setDefaultEdgeWeightInput(String(defaultEdgeWeight));
                }
              }}
            />
          </label>
          <br>
          </br><button
              className="px-3 py-1 rounded-full border"
              onClick={() => setShowCleanDialog(true)}
              title="Remove all neurons, edges, and groups from the canvas"
            >
              Clean canvas
            </button>
          <div className="flex flex-wrap items-center gap-2">
            <div className="h-6 w-px bg-gray-300 mx-1" />
            <button className="px-3 py-1 rounded-full border" onClick={() => addNeuron("input")}>+ Input</button>
            
            <button className="px-3 py-1 rounded-full border" onClick={() => addNeuron("hidden")}>+ Hidden</button>
            <button className="px-3 py-1 rounded-full border" onClick={() => addNeuron("output")}>+ Output</button>
          </div>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => deleteSelected()}
            disabled={!selectedEdgeId && selectedNodeIds.length === 0}
            title="Delete selected"
          >
            Delete
          </button>
          <button
            className={`px-3 py-1 rounded-full border ${touchMultiSelect ? "bg-blue-50 border-blue-300 text-blue-700" : ""}`}
            onClick={() => setTouchMultiSelect((prev) => !prev)}
            title="Enable to keep adding to the selection when tapping (helpful on touch screens)"
          >
            {touchMultiSelect ? "Multi-Select: On" : "Multi-Select: Off"}
          </button>
          <br></br>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => groupSelectionMain()}
            disabled={selectedNodeIds.length < 2}
            title="Group selected (Ctrl/Cmd+G)"
          >
            Group
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => ungroupSelectionMain()}
            disabled={!hasSelectionGroup()}
            title="Ungroup selected (Shift+Ctrl/Cmd+G)"
          >
            Ungroup
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => setSelectionGroupLabelVisibility(canShowGroupLabel)}
            disabled={!canToggleGroupLabel}
            title={
              canShowGroupLabel
                ? "Show group label below selection"
                : "Hide group label for selection"
            }
          >
            {canShowGroupLabel ? "Show Group Label" : "Hide Group Label"}
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => {
              if (primaryGroup) saveGroupAsModule(primaryGroup);
            }}
            disabled={!primaryGroup}
            title="Save selected group as module"
          >
            Save as Module
          </button>
          <label
            className="ml-2 flex items-center gap-2 text-xs text-gray-600"
            style={{ opacity: primaryGroup ? 1 : 0.5 }}
          >
            Group label: 
            <input
              className="w-32 px-2 py-0.5 border rounded-lg text-sm"
              type="text"
              disabled={!primaryGroup}
              value={primaryGroup ? groupLabelDraft : ""}
              onChange={(e) => setGroupLabelDraft(e.target.value)}
              onBlur={() => {
                if (primaryGroup) renameGroup(primaryGroup.id, groupLabelDraft);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  if (primaryGroup) {
                    renameGroup(primaryGroup.id, groupLabelDraft);
                    (e.target as HTMLInputElement).blur();
                  }
                }
              }}
            />
          </label>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => {
              if (primaryGroup && !isPrimaryGroupCompressed) {
                compressGroup(primaryGroup.id);
              }
            }}
            disabled={!primaryGroup || isPrimaryGroupCompressed}
            title="Compress group into single node"
          >
            Compress
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => {
              if (primaryGroup && isPrimaryGroupCompressed) {
                expandGroup(primaryGroup.id);
              }
            }}
            disabled={!primaryGroup || !isPrimaryGroupCompressed}
            title="Expand compressed group"
          >
            Expand
          </button>

          {/* <button
            className="px-3 py-1 rounded-full border"
            onClick={openDynamicsPanels}
            disabled={!selectedNodeIds.length || !sim}
            title="Show spiking dynamics for selected neuron(s)"
          >
            Show Dynamics
          </button> */}
          <br></br>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={closeAllDynamicsPanels}
            disabled={dynamicsPanels.length === 0}
            title="Hide all dynamics panels"
          >
            Hide all Spiking & Potential Plots
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => setShowDynamicsHeaders((prev) => !prev)}
            title="Toggle dynamics headers"
          >
            {showDynamicsHeaders ? "Hide Headers in Potential Plots" : "Show Headers in Potential Plots"}
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => setShowOutputPotentials((prev) => !prev)}
            title="Toggle potential traces for output neurons in canvas panels"
          >
            {showOutputPotentials ? "Hide Output Potentials" : "Show Output Potentials"}
          </button>
          <button
            className="px-3 py-1 rounded-full border"
            onClick={() => setIncludeColors((prev) => !prev)}
            title="Toggle colorful styling for neurons, edges, and group compressions"
          >
            {includeColors ? "Take out color" : "Include Color"}
          </button>
        
        </header>

        {showCleanDialog && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
            <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-xl space-y-4">
              <div className="text-base font-semibold text-gray-900 text-center">
                Are you sure about what you want to do?
              </div>
              <div className="flex flex-col gap-2">
                <button
                  className="px-3 py-2 rounded-full border border-gray-300 bg-white text-gray-800 hover:bg-gray-100 transition"
                  onClick={() => handleCleanDialog("abort")}
                >
                  Abort
                </button>
                <button
                  className="px-3 py-2 rounded-full border border-red-300 bg-red-50 text-red-700 hover:bg-red-100 transition"
                  onClick={() => handleCleanDialog("clean")}
                >
                  Clean anyway without saving
                </button>
                <button
                  className="px-3 py-2 rounded-full border border-blue-300 bg-blue-50 text-blue-700 hover:bg-blue-100 transition"
                  onClick={() => handleCleanDialog("save")}
                >
                  Save and clean
                </button>
              </div>
            </div>
          </div>
        )}

        <div
          className={`flex gap-4 ${isCanvasFullScreen ? "flex-col" : "flex-col xl:flex-row xl:items-start xl:justify-center"}`}
        >
          {/* Canvas */}
          <div className={`w-full ${isCanvasFullScreen ? "" : "xl:max-w-[960px] mx-auto xl:mx-0"}`}>
            <div ref={canvasExportRef} className={`relative w-full ${isCanvasFullScreen ? "min-h-[80vh]" : ""}`}>
              <CanvasView
                neurons={neurons}
                edges={edges}
                selectedNodeIds={selectedNodeIds}
                selectedEdgeId={selectedEdgeId}
                canvasWidth={canvasWidth}
                canvasHeight={canvasHeight}
                mode={mode}
                connectSrc={connectSrc}
                connectPoints={connectPoints}
                groups={groups}
                dynamicsPanels={dynamicsPanels}
                dynamicsData={{ sim, T }}
                showDynamicsHeaders={showDynamicsHeaders}
                showOutputPotentials={showOutputPotentials}
                includeColors={includeColors}
                hideStandardWeights={hideStandardWeights}
                stickyMultiSelect={touchMultiSelect}
                onSelectGroup={(group, modifiers) => handleGroupPointerDown(group.id, modifiers)}
                onNodePointerDown={handleNodePointerDown}
                onNodeDoubleClick={openDynamicsPanelForNeuron}
                onSelectEdge={handleEdgePointerDown}
                onClearSelection={clearSelection}
                onMoveNodes={moveNodes}
                onMoveLabels={moveLabelOffsets}
                onBeginDrag={(_ids) => beginNodeDrag()}
                onBeginLabelDrag={(_id) => beginLabelDrag()}
                onBeginEdgeDrag={() => beginEdgeDrag()}
                onEndDrag={endDrag}
                onMarqueeSelect={handleMarqueeSelect}
                onCompressGroup={compressGroup}
                onExpandGroup={expandGroup}
                onMovePanel={moveDynamicsPanel}
                onClosePanel={closeDynamicsPanel}
                onBeginConnect={beginConnect}
                onCompleteConnect={completeConnect}
                onAddWaypoint={addConnectWaypoint}
                onCancelConnect={cancelConnect}
                onExitConnectMode={exitConnectMode}
                activeEdgeHandle={selectedEdgeHandle}
                onSelectEdgeHandle={(edgeId, index) => {
                  if (edgeId === null || index === null) {
                    setSelectedEdgeHandle(null);
                  } else {
                    setSelectedEdgeHandle({ edgeId, index });
                    setActiveSelectionContext("main");
                  }
                }}
                onMoveEdgePoint={updateEdgeWaypoint}
                onInsertEdgePoint={insertEdgeWaypoint}
                onEditEdge={setSelectedEdgeId}
                highlightedNodeIds={highlightedNodeIds}
                highlightedEdgeIds={highlightedEdgeIds}
                recentNodeIds={recentNodeIds}
                recentEdgeIds={recentEdgeIds}
                fullWidth={isCanvasFullScreen}
              />
            {hasAnimation && (
              <div className="pointer-events-none absolute inset-0 flex flex-col justify-end">
                <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2 sm:gap-3 px-3 pb-3">
                  <div className="pointer-events-auto flex flex-wrap gap-2">
                    <button className="px-3 py-1 rounded-full border bg-white/80 text-sm" onClick={stepBackward} disabled={!canStepBackward}>
                      Prev Step
                    </button>
                    <button
                      className="px-3 py-1 rounded-full border bg-white/80 text-sm"
                      onClick={isAnimating ? pauseAnimation : resumeAnimation}
                      disabled={freezeDisabled}
                    >
                      {isAnimating ? "Freeze" : "Resume"}
                    </button>
                    <button className="px-3 py-1 rounded-full border bg-white/80 text-sm" onClick={stepForward} disabled={!canStepForward}>
                      Next Step
                    </button>
                    <button
                      className="px-3 py-1 rounded-full border bg-white/80 text-sm"
                      onClick={jumpToEnd}
                      disabled={!canJumpToEnd}
                    >
                      End
                    </button>
                  </div>
                  <div className="pointer-events-auto bg-black/70 text-white text-xs font-mono px-3 py-1 rounded-lg shadow">
                    t = {formattedClock}s • Step {frameIndexDisplay}/{frameTotal}
                    {currentFrame ? (
                      <>
                        {" "}
                        • Wave {currentWaveDisplay}
                      </>
                    ) : null}
                  </div>
                </div>
              </div>
            )}
          </div>
          {selectedNodeIds.length === 1 && (
            <div className="mt-4"><br></br>
              <NeuronInspector
                neurons={neurons}
                selectedIds={selectedNodeIds}
                onLabel={setLabel}
                onRole={setRole}
                onLabelVisible={setLabelVisibility}
                onLabelOffset={setLabelOffset}
                onResetLabelOffset={resetLabelOffset}
              />
            </div>
          )}
          <EdgeEditor edge={edges.find((e) => e.id === selectedEdgeId) || null} onChange={setWeight} onRemove={removeEdge} />
        </div>

        {!isCanvasFullScreen && (
          <div className="w-full xl:w-[360px] mx-auto xl:mx-0">{controlsPanel}</div>
        )}
        </div>

        {isCanvasFullScreen ? (
          <>
            <div className="w-full max-w-3xl mx-auto mt-4">{controlsPanel}</div>
            <div className="mt-4">{modulesSection}</div>
            <div className="border-t border-gray-200 my-4" />
            <div className="mt-4">{spikeRasterSection}</div>
          </>
        ) : (
          <>
            {modulesSection}
            <div className="border-t border-gray-200 my-4" />
            <div className="mt-4">{spikeRasterSection}</div>
          </>
        )}

        {/* <div className="p-3 rounded-2xl border shadow-sm bg-white">
        <div className="flex items-center justify-between mb-2">
          <div className="font-medium">Potential Traces</div>
          <div className="text-xs text-gray-500">decay shown for h = {hKind === "infty" ? "∞" : hKind === "zero" ? 0 : h.toFixed(3)}</div>
        </div>
        <div ref={potentialExportRef}>
          <PotentialPlot neurons={hiddenNeurons} series={sim?.potentialSeries || null} T={T} />
        </div>
        </div> */}

        
      </div>
    </div>
  );
}

// ------------------------ Canvas View ------------------------

function CanvasView({
  neurons,
  edges,
  selectedNodeIds,
  selectedEdgeId,
  canvasWidth,
  canvasHeight,
  mode,
  connectSrc,
  connectPoints = [],
  groups = [],
  dynamicsPanels = [],
  dynamicsData = { sim: null, T: 0 },
  showDynamicsHeaders = true,
  showOutputPotentials = false,
  includeColors = true,
  hideStandardWeights = false,
  stickyMultiSelect = false,
  onSelectGroup,
  onNodePointerDown,
  onNodeDoubleClick,
  onSelectEdge,
  onClearSelection,
  onMoveNodes,
  onMoveLabels,
  onBeginConnect,
  onCompleteConnect,
  onAddWaypoint,
  onCancelConnect,
  onExitConnectMode = () => {},
  activeEdgeHandle = null,
  onSelectEdgeHandle,
  onMoveEdgePoint,
  onInsertEdgePoint,
  onEditEdge,
  highlightedNodeIds = [],
  highlightedEdgeIds = [],
  recentNodeIds = [],
  recentEdgeIds = [],
  onBeginDrag,
  onBeginLabelDrag,
  onBeginEdgeDrag,
  onEndDrag,
  onMarqueeSelect,
  onCompressGroup = () => {},
  onExpandGroup = () => {},
  onMovePanel = () => {},
  onClosePanel = () => {},
  fullWidth = false,
}: {
  neurons: Neuron[];
  edges: Edge[];
  selectedNodeIds: string[];
  selectedEdgeId: string | null;
  canvasWidth: number;
  canvasHeight: number;
  mode: "select" | "connect";
  connectSrc: string | null;
  connectPoints?: Array<{ x: number; y: number }>;
  groups?: Group[];
  dynamicsPanels?: Array<{ id: string; neuronId: string; x: number; y: number }>;
  dynamicsData: { sim: SimResult | null; T: number };
  showDynamicsHeaders?: boolean;
  showOutputPotentials?: boolean;
  includeColors?: boolean;
  hideStandardWeights?: boolean;
  stickyMultiSelect?: boolean;
  onSelectGroup: (group: Group, modifiers: { append: boolean; toggle: boolean }) => string[];
  onNodePointerDown: (id: string, modifiers: { append: boolean; toggle: boolean }) => string[];
  onNodeDoubleClick?: (id: string) => void;
  onSelectEdge: (edgeId: string) => void;
  onClearSelection: () => void;
  onMoveNodes: (updates: Array<{ id: string; x: number; y: number }>) => void;
  onMoveLabels: (updates: Array<{ id: string; labelOffsetX: number; labelOffsetY: number }>) => void;
  onBeginConnect: (id: string) => void;
  onCompleteConnect: (id: string) => void;
  onAddWaypoint: (point: { x: number; y: number }) => void;
  onCancelConnect: () => void;
  onExitConnectMode?: () => void;
  activeEdgeHandle?: EdgeHandleSelection | null;
  onSelectEdgeHandle: (edgeId: string | null, index: number | null) => void;
  onMoveEdgePoint: (edgeId: string, index: number, point: { x: number; y: number }) => void;
  onInsertEdgePoint: (edgeId: string, point: { x: number; y: number }, index: number) => void;
  onEditEdge: (edgeId: string | null) => void;
  highlightedNodeIds?: string[];
  highlightedEdgeIds?: string[];
  recentNodeIds?: string[];
  recentEdgeIds?: string[];
  onBeginDrag: (ids: string[]) => void;
  onBeginLabelDrag: (id: string) => void;
  onBeginEdgeDrag: (edgeId: string, index: number) => void;
  onEndDrag: () => void;
  onMarqueeSelect: (ids: string[], modifiers: { append: boolean; toggle: boolean }) => void;
  onCompressGroup?: (groupId: string) => void;
  onExpandGroup?: (groupId: string) => void;
  onMovePanel: (panelId: string, x: number, y: number) => void;
  onClosePanel: (panelId: string) => void;
  fullWidth?: boolean;
}) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [drag, setDrag] = useState<
    | {
        ids: string[];
        origin: { x: number; y: number };
        startPositions: Record<string, { x: number; y: number }>;
      }
    | null
  >(null);
  const [labelDrag, setLabelDrag] = useState<
    | {
        id: string;
        origin: { x: number; y: number };
        startOffset: { x: number; y: number };
      }
    | null
  >(null);
  const [edgePointDrag, setEdgePointDrag] = useState<{ edgeId: string; index: number } | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [marquee, setMarquee] = useState<
    | {
        origin: { x: number; y: number };
        current: { x: number; y: number };
        append: boolean;
        toggle: boolean;
      }
    | null
  >(null);

  function coord(e: React.PointerEvent | React.MouseEvent) {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    const p = pt.matrixTransform(ctm.inverse());
    return { x: p.x, y: p.y };
  }

  function endPointerSequence() {
    if (marquee) {
      finalizeMarqueeSelection();
      return;
    }
    if (panelDrag) {
      setPanelDrag(null);
      return;
    }
    setDrag(null);
    setLabelDrag(null);
    setEdgePointDrag(null);
    onEndDrag();
  }

  function pointerModifiers(event: React.PointerEvent | React.MouseEvent) {
    return {
      append: event.shiftKey || stickyMultiSelect,
      toggle: event.metaKey || event.ctrlKey,
    };
  }

  const R = NODE_RADIUS; // node radius
  const viewWidth = Math.max(200, canvasWidth);
  const viewHeight = Math.max(200, canvasHeight);
  const idPrefix = useMemo(() => uid("canvas"), []);
  const highlightedNodeSet = useMemo(() => new Set(highlightedNodeIds), [highlightedNodeIds]);
  const highlightedEdgeSet = useMemo(() => new Set(highlightedEdgeIds), [highlightedEdgeIds]);
  const recentNodeSet = useMemo(() => new Set(recentNodeIds), [recentNodeIds]);
  const recentEdgeSet = useMemo(() => new Set(recentEdgeIds), [recentEdgeIds]);
  const gap = EDGE_GAP;
  const { sim: dynamicsSim, T: dynamicsT } = dynamicsData;

  const [panelDrag, setPanelDrag] = useState<
    | {
        panelId: string;
        origin: { x: number; y: number };
        start: { x: number; y: number };
      }
    | null
  >(null);

  function finalizeMarqueeSelection() {
    setMarquee((prev) => {
      if (!prev) return null;
      const { origin, current, append, toggle } = prev;
      const xMin = Math.min(origin.x, current.x);
      const xMax = Math.max(origin.x, current.x);
      const yMin = Math.min(origin.y, current.y);
      const yMax = Math.max(origin.y, current.y);
      const ids = neurons
        .filter((n) => n.x >= xMin && n.x <= xMax && n.y >= yMin && n.y <= yMax)
        .map((n) => n.id);
      if (ids.length) {
        onMarqueeSelect(ids, { append, toggle });
      }
      return null;
    });
  }

  const containerStyle: React.CSSProperties = fullWidth
    ? {
        boxSizing: "border-box",
        padding: CANVAS_FRAME_PADDING,
        width: "100vw",
        marginLeft: "calc(50% - 50vw)",
        marginRight: "calc(50% - 50vw)",
      }
    : {
        boxSizing: "border-box",
        padding: CANVAS_FRAME_PADDING,
        maxWidth: viewWidth + CANVAS_FRAME_PADDING * 2,
      };
  const containerClass = fullWidth
    ? "rounded-2xl border shadow-sm bg-white select-none"
    : "rounded-2xl border shadow-sm bg-white select-none mx-auto w-full";

  return (
    <div className={containerClass} style={containerStyle}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${viewWidth} ${viewHeight}`}
        style={{
          width: "100%",
          height: "auto",
          touchAction: "none",
          display: "block",
        }}
        onPointerMove={(e) => {
          if (drag || labelDrag || edgePointDrag || marquee || panelDrag) {
            e.preventDefault();
          }
          const { x, y } = coord(e);
          setMousePos({ x, y });
          if (marquee) {
            setMarquee((prev) => (prev ? { ...prev, current: { x, y } } : prev));
            return;
          }
          if (panelDrag) {
            onMovePanel(panelDrag.panelId, panelDrag.start.x + (x - panelDrag.origin.x), panelDrag.start.y + (y - panelDrag.origin.y));
            return;
          }
          if (drag) {
            const dx = x - drag.origin.x;
            const dy = y - drag.origin.y;
            const updates = drag.ids.map((id) => {
              const start = drag.startPositions[id];
              return { id, x: start.x + dx, y: start.y + dy };
            });
            onMoveNodes(updates);
          }
          if (labelDrag) {
            const dx = x - labelDrag.origin.x;
            const dy = y - labelDrag.origin.y;
            onMoveLabels([
              {
                id: labelDrag.id,
                labelOffsetX: labelDrag.startOffset.x + dx,
                labelOffsetY: labelDrag.startOffset.y + dy,
              },
            ]);
          }
          if (edgePointDrag) {
            onMoveEdgePoint(edgePointDrag.edgeId, edgePointDrag.index, { x, y });
          }
        }}
        onPointerUp={endPointerSequence}
        onPointerLeave={endPointerSequence}
        onPointerCancel={endPointerSequence}
        onContextMenu={(e) => {
          if (mode === "connect" && connectSrc) {
            e.preventDefault();
            onCancelConnect();
          }
        }}
      >
        {/* defs */}
        <defs>
          <pattern id={`${idPrefix}-grid`} width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#838383ff" strokeWidth="1" strokeOpacity="0.5" />
          </pattern>
          <marker
            id={`${idPrefix}-arrow`}
            viewBox="0 0 12 12"
            refX="10"
            refY="6"
            markerWidth="12"
            markerHeight="12"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M 0 2 L 10 6 L 0 10 z" fill="context-stroke" stroke="none" />
          </marker>
        </defs>

        {/* background (click to clear selection or add waypoints) */}
        <rect
          x={0}
          y={0}
          width={viewWidth}
          height={viewHeight}
          fill="white"
          onPointerDown={(e) => {
            e.preventDefault();
            if (mode === "select" && e.shiftKey) {
              const { x, y } = coord(e);
              setMarquee({
                origin: { x, y },
                current: { x, y },
                append: e.shiftKey,
                toggle: e.metaKey || e.ctrlKey,
              });
              e.stopPropagation();
              return;
            }
            onSelectEdgeHandle(null, null);
            setEdgePointDrag(null);
            if (mode === "connect") {
              e.stopPropagation();
              onExitConnectMode();
              return;
            }
            onClearSelection();
          }}
        />

        {marquee && (() => {
          const { origin, current } = marquee;
          const x1 = Math.min(origin.x, current.x);
          const y1 = Math.min(origin.y, current.y);
          const width = Math.abs(origin.x - current.x);
          const height = Math.abs(origin.y - current.y);
          if (width < 2 && height < 2) return null;
          return (
            <rect
              x={x1}
              y={y1}
              width={width}
              height={height}
              fill="rgba(59, 130, 246, 0.15)"
              stroke="#3b82f6"
              strokeDasharray="6 4"
              pointerEvents="none"
            />
          );
        })()}

        {/* group rectangles behind edges and nodes */}
        {groups.map((g) => {
          const nodes = g.nodeIds
            .map((id) => neurons.find((n) => n.id === id))
            .filter((n): n is Neuron => Boolean(n));
          if (!nodes.length) return null;
          const isCompressed = Boolean(g.compression);
          if (isCompressed) return null;
          const minX = Math.min(...nodes.map((n) => n.x));
          const maxX = Math.max(...nodes.map((n) => n.x));
          const minY = Math.min(...nodes.map((n) => n.y));
          const maxY = Math.max(...nodes.map((n) => n.y));
          const centerX = isCompressed ? g.compression!.center.x : (minX + maxX) / 2;
          const centerY = isCompressed ? g.compression!.center.y : (minY + maxY) / 2;
          const pad = R + 10;
          const compressedPadding = 6;
          const x = isCompressed ? centerX - (R + compressedPadding / 2) : minX - pad;
          const y = isCompressed ? centerY - (R + compressedPadding / 2) : minY - pad;
          const w = isCompressed ? R * 2 + compressedPadding : (maxX - minX) + pad * 2;
          const h = isCompressed ? R * 2 + compressedPadding : (maxY - minY) + pad * 2;
          const fill = includeColors ? "rgba(243, 244, 246, 0.6)" : "#ffffff";
          const stroke = "#111827";
          const label = g.label?.trim() ?? "";
          const labelY = isCompressed ? centerY + 4 + compressedPadding / 2 : Math.min(y + h + 18, viewHeight - 8);
          return (
            <g
              key={g.id}
              onPointerDown={(e) => {
                e.preventDefault();
                if (mode === "connect") {
                  e.stopPropagation();
                  onExitConnectMode();
                  return;
                }
                if (mode !== "select") return;
                e.stopPropagation();
                const selection = onSelectGroup(g, pointerModifiers(e));
                const idsToDrag = selection.length ? selection : [];
                if (!idsToDrag.length) return;
                onBeginDrag(idsToDrag);
                const point = coord(e);
                setEdgePointDrag(null);
                setLabelDrag(null);
                const startPositions: Record<string, { x: number; y: number }> = {};
                for (const id of idsToDrag) {
                  const target = neurons.find((node) => node.id === id);
                  if (target) startPositions[id] = { x: target.x, y: target.y };
                }
                if (!Object.keys(startPositions).length) return;
                setDrag({ ids: idsToDrag, origin: { x: point.x, y: point.y }, startPositions });
              }}
              onDoubleClick={(e) => {
                e.stopPropagation();
                if (mode === "connect") {
                  onExitConnectMode();
                  onCompressGroup(g.id);
                  return;
                }
                if (mode !== "select") return;
                onCompressGroup(g.id);
              }}
              style={{ cursor: mode === "select" ? "pointer" : "default" }}
            >
              <rect
                x={x}
                y={y}
                width={w}
                height={h}
                rx={isCompressed ? 8 : 12}
                ry={isCompressed ? 8 : 12}
                fill={fill}
                stroke={stroke}
                strokeWidth={2}
                strokeDasharray="6 4"
                opacity={mode === "select" ? 1 : 0.8}
              />
              {g.labelVisible && label && (
                <text
                  x={isCompressed ? centerX : x + w / 2}
                  y={labelY}
                  textAnchor="middle"
                  fontSize={12}
                  fill="#1f2937"
                  fontWeight={600}
                >
                  {label}
                </text>
              )}
            </g>
          );
        })}
        <rect
          x={0}
          y={0}
          width={viewWidth}
          height={viewHeight}
          fill={`url(#${idPrefix}-grid)`}
          stroke="#838383"
          strokeOpacity="0.5"
          strokeWidth={1}
          pointerEvents="none"
        />

        {/* edges */}
        {edges.map((e) => {
          const source = neurons.find((n) => n.id === e.sourceId);
          const target = neurons.find((n) => n.id === e.targetId);
          if (!source || !target) return null;
          const waypoints = sanitizeWaypoints(e.points);
          const pathPoints = buildEdgePoints(e, source, target);
          if (pathPoints.length < 2) return null;
          const pathD = pathPoints
            .map((pt, idx) => `${idx === 0 ? "M" : "L"} ${pt.x} ${pt.y}`)
            .join(" ");
          const isSel = selectedEdgeId === e.id;
          const baseColor = includeColors ? weightToColor(e.weight) : "#000000";
          const isHighlight = highlightedEdgeSet.has(e.id);
          const isRecent = recentEdgeSet.has(e.id);
          const strokeColor = isRecent ? "#f97316" : isHighlight ? "#facc15" : baseColor;
          const strokeWidth = isRecent ? EDGE_STROKE_WIDTH + 1.4 : isHighlight ? EDGE_STROKE_WIDTH + 1.0 : isSel ? EDGE_STROKE_WIDTH + 0.6 : EDGE_STROKE_WIDTH;
          const strokeDasharray = includeColors ? undefined : weightToDashPattern(e.weight);
          const arrowTip = pathPoints[pathPoints.length - 1];
          const arrowHitRadius = Math.max(strokeWidth * 1.4, 7);
          const labelPoint = pathMidpoint(pathPoints);
          const showWeightLabel = !hideStandardWeights || !isStandardWeight(e.weight);
          return (
            <g
              key={e.id}
              onPointerDown={(ev) => {
                ev.preventDefault();
                onSelectEdge(e.id);
                onSelectEdgeHandle(e.id, null);
                setEdgePointDrag(null);
                ev.stopPropagation();
              }}
              onDoubleClick={(ev) => {
                ev.stopPropagation();
                ev.preventDefault();
                if (mode !== "select") {
                  onSelectEdge(e.id);
                  onEditEdge(e.id);
                  return;
                }
                const rawPoint = coord(ev);
                const closest = closestPointOnPath(rawPoint, pathPoints);
                const pointForInsert = closest ? closest.point : rawPoint;
                const segmentIndex = closest
                  ? closest.segment
                  : waypointInsertIndex(pathPoints, waypoints.length, pointForInsert);
                const insertIndex = Math.min(segmentIndex, waypoints.length);
                onSelectEdge(e.id);
                onInsertEdgePoint(e.id, pointForInsert, insertIndex);
                onSelectEdgeHandle(e.id, insertIndex);
              }}
            >
              <path
                d={pathD}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                strokeDasharray={strokeDasharray}
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
                pointerEvents="stroke"
                markerEnd={`url(#${idPrefix}-arrow)`}
              />
              <circle
                cx={arrowTip.x}
                cy={arrowTip.y}
                r={arrowHitRadius}
                fill="transparent"
                stroke="none"
                pointerEvents="all"
              />
              {/* weight label */}
              {showWeightLabel && (
                <text x={labelPoint.x} y={labelPoint.y - 6} textAnchor="middle" fontSize={12} fill="#1f2937">
                  {e.weight}
                </text>
              )}
              {isSel &&
                waypoints.map((pt, idx) => {
                  const isHandleActive =
                    activeEdgeHandle && activeEdgeHandle.edgeId === e.id && activeEdgeHandle.index === idx;
                  return (
                    <circle
                      key={`${e.id}-handle-${idx}`}
                      cx={pt.x}
                      cy={pt.y}
                      r={6}
                      fill={isHandleActive ? "#f97316" : "#ffffff"}
                      stroke={isHandleActive ? "#c2410c" : "#111827"}
                      strokeWidth={isHandleActive ? 2 : 1.5}
                      className="cursor-move"
                      onPointerDown={(ev) => {
                        ev.preventDefault();
                        if (mode !== "select") return;
                        ev.stopPropagation();
                        onSelectEdge(e.id);
                        onSelectEdgeHandle(e.id, idx);
                        onBeginEdgeDrag(e.id, idx);
                        setEdgePointDrag({ edgeId: e.id, index: idx });
                      }}
                    />
                  );
                })}
            </g>
          );
        })}

        {/* connecting preview line (ghost) */}
        {mode === "connect" && connectSrc && (() => {
          const src = neurons.find((n) => n.id === connectSrc);
          if (!src) return null;
          const previewPoints = sanitizeWaypoints(connectPoints);
          if (mousePos) {
            previewPoints.push({ x: mousePos.x, y: mousePos.y });
          }
          const firstTarget = previewPoints[0] ?? (mousePos ? { x: mousePos.x, y: mousePos.y } : { x: src.x + gap, y: src.y });
          const start = offsetPoint({ x: src.x, y: src.y }, firstTarget, gap);
          const pathSeq = [start, ...previewPoints];
          if (pathSeq.length < 2) return null;
          const d = pathSeq.map((pt, idx) => `${idx === 0 ? "M" : "L"} ${pt.x} ${pt.y}`).join(" ");
          return (
            <path
              d={d}
              stroke="#4b5563"
              strokeWidth={EDGE_STROKE_WIDTH}
              strokeDasharray="6 4"
              fill="none"
              pointerEvents="none"
            />
          );
        })()}

        {/* nodes */}
        {neurons.map((n) => {
          const isSel = selectedNodeIds.includes(n.id);
          const isHighlight = highlightedNodeSet.has(n.id);
          const isRecent = recentNodeSet.has(n.id);
          const fill = includeColors ? ROLE_FILL[n.role] ?? ROLE_FILL.hidden : "#ffffff";
          const stroke = isRecent ? "#f97316" : isHighlight ? "#facc15" : isSel ? "#0f172a" : "#374151";
          const labelText = (n.label || "").trim() || n.id.slice(-4);
          const labelVisible = n.labelVisible === true && labelText.length > 0;
          const labelOffsetX = Number.isFinite(n.labelOffsetX) ? (n.labelOffsetX as number) : 0;
          const labelOffsetY = Number.isFinite(n.labelOffsetY)
            ? (n.labelOffsetY as number)
            : DEFAULT_LABEL_OFFSET_Y;
          const scale = isRecent || isHighlight ? 1.08 : isSel ? 1.04 : 1;
          return (
            <g
              key={n.id}
              transform={`translate(${n.x},${n.y})`}
              onPointerDown={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setEdgePointDrag(null);
                if (mode === "select") {
                  const point = coord(e);
                  const selection = onNodePointerDown(n.id, pointerModifiers(e));
                  const idsToDrag = selection.length ? selection : [n.id];
                  if (!idsToDrag.length) {
                    setDrag(null);
                    return;
                  }
                  onBeginDrag(idsToDrag);
                  const startPositions: Record<string, { x: number; y: number }> = {};
                  for (const id of idsToDrag) {
                    const target = neurons.find((node) => node.id === id);
                    if (target) startPositions[id] = { x: target.x, y: target.y };
                  }
                  setDrag({ ids: idsToDrag, origin: { x: point.x, y: point.y }, startPositions });
                } else if (mode === "connect") {
                  if (!connectSrc) onBeginConnect(n.id);
                  else onCompleteConnect(n.id);
                }
              }}
              onDoubleClick={(e) => {
                if (mode !== "select") return;
                e.stopPropagation();
                onNodeDoubleClick?.(n.id);
              }}
            >
              <g transform={`scale(${scale})`} style={{ transition: "transform 160ms ease-out" }}>
                {isHighlight && !isRecent && (
                  <circle r={R + 8} fill="rgba(250, 204, 21, 0.18)" stroke="#facc15" strokeWidth={2} />
                )}
                {isRecent && (
                  <circle r={R + 8} fill="rgba(249, 115, 22, 0.22)" stroke="#f97316" strokeWidth={2.4} />
                )}
                <circle r={R} fill={fill} stroke={stroke} strokeWidth={isSel ? 3 : 2} />
              </g>
              {labelVisible && (
                <g
                  transform={`translate(${labelOffsetX},${labelOffsetY})`}
                  className="cursor-move"
                  onPointerDown={(e) => {
                    e.preventDefault();
                    if (mode !== "select") return;
                    e.stopPropagation();
                    onNodePointerDown(n.id, pointerModifiers(e));
                    onBeginLabelDrag(n.id);
                    const { x, y } = coord(e);
                    setDrag(null);
                    setEdgePointDrag(null);
                    setLabelDrag({
                      id: n.id,
                      origin: { x, y },
                      startOffset: { x: labelOffsetX, y: labelOffsetY },
                    });
                  }}
                >
                  <text
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={12}
                    fill="#0f172a"
                    fontWeight={500}
                    pointerEvents="none"
                  >
                    {labelText}
                  </text>
                </g>
              )}
              {mode === "connect" && connectSrc === n.id && (
                <circle r={R + 4} fill="none" stroke="#111827" strokeDasharray="4 3" />
              )}
            </g>
          );
        })}

        {/* compressed groups overlay */}
        {groups.map((g) => {
          if (!g.compression) return null;
          const nodes = g.nodeIds
            .map((id) => neurons.find((n) => n.id === id))
            .filter((n): n is Neuron => Boolean(n));
          if (!nodes.length) return null;
          const centerX = g.compression.center.x;
          const centerY = g.compression.center.y;
          const compressedPadding = 6;
          const w = R * 2 + compressedPadding;
          const h = R * 2 + compressedPadding;
          const x = centerX - w / 2;
          const y = centerY - h / 2;
          const fill = includeColors ? COMPRESSED_GROUP_FILL : "#ffffff";
          const stroke = "#111827";
          const label = g.label?.trim() ?? "";
          const labelY = centerY + 4 + compressedPadding / 2;
          const groupSelected = g.nodeIds.some((id) => selectedNodeIds.includes(id));
          const strokeWidth = groupSelected ? 4 : 2;
          return (
            <g
              key={`${g.id}-compressed`}
              onPointerDown={(e) => {
                e.preventDefault();
                if (mode === "connect") {
                  e.stopPropagation();
                  onExitConnectMode();
                  return;
                }
                if (mode !== "select") return;
                e.stopPropagation();
                const selection = onSelectGroup(g, pointerModifiers(e));
                const idsToDrag = selection.length ? selection : [];
                if (!idsToDrag.length) return;
                onBeginDrag(idsToDrag);
                const point = coord(e);
                setEdgePointDrag(null);
                setLabelDrag(null);
                const startPositions: Record<string, { x: number; y: number }> = {};
                for (const id of idsToDrag) {
                  const target = neurons.find((node) => node.id === id);
                  if (target) startPositions[id] = { x: target.x, y: target.y };
                }
                if (!Object.keys(startPositions).length) return;
                setDrag({ ids: idsToDrag, origin: { x: point.x, y: point.y }, startPositions });
              }}
              onDoubleClick={(e) => {
                e.stopPropagation();
                if (mode === "connect") {
                  onExitConnectMode();
                  onExpandGroup(g.id);
                  return;
                }
                if (mode !== "select") return;
                onExpandGroup(g.id);
              }}
              style={{ cursor: mode === "select" ? "pointer" : "default" }}
            >
              <rect
                x={x}
                y={y}
                width={w}
                height={h}
                rx={8}
                ry={8}
                fill={fill}
                stroke={stroke}
                strokeWidth={strokeWidth}
                strokeDasharray="6 4"
              />
              {g.labelVisible && label && (
                <text
                  x={centerX}
                  y={labelY}
                  textAnchor="middle"
                  fontSize={12}
                  fill="#1f2937"
                  fontWeight={600}
                >
                  {label}
                </text>
              )}
            </g>
          );
        })}

        {dynamicsPanels.map((panel) => {
          const neuron = neurons.find((n) => n.id === panel.neuronId);
          if (!neuron) return null;
          const spikes = dynamicsSim?.spikeTrains?.[panel.neuronId] ?? [];
          const series = dynamicsSim?.potentialSeries;
          const potentials = series?.values?.[panel.neuronId] ?? [];
          const times = series?.times ?? [];
          const panelWidth = 180;
          const padding = 12;
          const headerHeight = 28;
          const extraHeight = showDynamicsHeaders ? 0 : headerHeight;
          const spikeHeightBase = 30;
          const potentialHeightBase = 60;
          const isOutputNeuron = neuron.role === "output";
          const showPotentialPlot = neuron.role === "hidden" || (showOutputPotentials && isOutputNeuron);
          const potentialExtraShare = Math.round(extraHeight * 0.35);
          const spikeHeight = spikeHeightBase + (showPotentialPlot ? potentialExtraShare : extraHeight);
          const potentialHeight = showPotentialPlot ? potentialHeightBase + extraHeight - potentialExtraShare : 0;
          const labelWidth = 26;
          const contentHeight = padding + spikeHeight + padding + (showPotentialPlot ? potentialHeight + padding : 0);
          const panelHeight = contentHeight + (showDynamicsHeaders ? headerHeight : 0);
          const contentOffsetY = showDynamicsHeaders ? headerHeight : 0;
          const spikeTop = padding;
          const potentialTop = spikeTop + spikeHeight + (showPotentialPlot ? padding : 0);
          const timeMax = dynamicsT > 0 ? dynamicsT : Math.max(...(times.length ? [times[times.length - 1]] : [1]));
          const safeMax = timeMax > 0 ? timeMax : 1;
          const plotLeft = padding + labelWidth;
          const plotRight = panelWidth - padding;
          const plotWidth = Math.max(0, plotRight - plotLeft);
          const axisLabelX = plotLeft - 10;
          const spikeColor = includeColors ? "#2563eb" : "#111827";
          const potentialColor = includeColors ? "#f97316" : "#111827";
          const spikeLines = spikes.map((t) => {
            const x = plotWidth > 0 ? (t / safeMax) * plotWidth + plotLeft : plotLeft;
            return { x };
          });
          const potentialPath = (() => {
            if (!showPotentialPlot || !times.length || !potentials.length) return "";
            const maxVal = potentials.reduce((acc, v) => Math.max(acc, v), 1);
            const safeVal = maxVal > 0 ? maxVal : 1;
            return times
              .map((t, idx) => {
                const val = potentials[idx] ?? 0;
                const x = plotWidth > 0 ? (t / safeMax) * plotWidth + plotLeft : plotLeft;
                const y = potentialTop + potentialHeight - (val / safeVal) * potentialHeight;
                return `${idx === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
              })
              .join(" ");
          })();
          return (
            <g
              key={panel.id}
              transform={`translate(${panel.x},${panel.y})`}
              onPointerDown={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const { x, y } = coord(e);
                setPanelDrag({ panelId: panel.id, origin: { x, y }, start: { x: panel.x, y: panel.y } });
              }}
            >
              <rect x={0} y={0} width={panelWidth} height={panelHeight} fill="#f8fafc" stroke="#1f2937" strokeWidth={1.5} />
              {showDynamicsHeaders && (
                <>
                  <rect x={0} y={0} width={panelWidth} height={headerHeight} fill="#e2e8f0" opacity={0.9} />
                  <text
                    x={padding}
                    y={headerHeight / 2}
                    fontSize={12}
                    fill="#0f172a"
                    fontWeight={600}
                    dominantBaseline="middle"
                    textAnchor="start"
                  >
                    {neuron.label}
                  </text>
                  <text
                    x={panelWidth - padding}
                    y={headerHeight / 2}
                    fontSize={14}
                    fill="#1f2937"
                    fontWeight={600}
                    dominantBaseline="middle"
                    textAnchor="end"
                    style={{ cursor: "pointer" }}
                    onPointerDown={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onClosePanel(panel.id);
                    }}
                  >
                    ×
                  </text>
                </>
              )}
              <g transform={`translate(0, ${contentOffsetY})`}>
                {dynamicsSim ? (
                  <>
                    <g>
                      <text
                        x={axisLabelX}
                        y={spikeTop + spikeHeight / 2}
                        fontSize={10}
                        fill="#111827"
                        dominantBaseline="middle"
                        textAnchor="middle"
                        transform={`rotate(-90 ${axisLabelX} ${spikeTop + spikeHeight / 2})`}
                      >
                        Spikes
                      </text>
                      <line
                        x1={plotLeft}
                        y1={spikeTop}
                        x2={plotRight}
                        y2={spikeTop}
                        stroke="#d1d5db"
                      />
                      {spikeLines.map(({ x }, idx) => (
                        <line
                          key={idx}
                          x1={x}
                          x2={x}
                          y1={spikeTop + 2}
                          y2={spikeTop + spikeHeight - 8}
                          stroke={spikeColor}
                          strokeWidth={2}
                        />
                      ))}
                    </g>
                    {showPotentialPlot && (
                      <g>
                        <text
                          x={axisLabelX}
                          y={potentialTop + potentialHeight / 2}
                          fontSize={10}
                          fill="#111827"
                          dominantBaseline="middle"
                          textAnchor="middle"
                          transform={`rotate(-90 ${axisLabelX} ${potentialTop + potentialHeight / 2})`}
                        >
                          Potential
                        </text>
                        <rect
                          x={plotLeft}
                          y={potentialTop}
                          width={plotWidth}
                          height={potentialHeight}
                          fill="#ffffff"
                          stroke="#d1d5db"
                        />
                        {potentialPath && (
                          <path d={potentialPath} fill="none" stroke={potentialColor} strokeWidth={1.6} />
                        )}
                      </g>
                    )}
                  </>
                ) : (
                  <text x={padding} y={spikeTop + 20} fontSize={10} fill="#111827">
                    Run the simulation to view dynamics.
                  </text>
                )}
              </g>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ------------------------ Edge Editor ------------------------

function EdgeEditor({ edge, onChange, onRemove }: { edge: Edge | null; onChange: (id: string, w: number) => void; onRemove: (id: string) => void }) {
  if (!edge) return null;
  return (
    <div className="mt-2 p-3 rounded-2xl border shadow-sm bg-white flex items-end gap-2 w-full">
      <div className="grow">
        <div className="text-lg font-semibold">Edge Connection</div>
        <div className="text-xs text-gray-500 mb-2">
          {edge.sourceId.slice(-4)} → {edge.targetId.slice(-4)}
        </div>
        <label className="text-xs text-gray-600">weight</label>
        <input
          type="number"
          step={1}
          className="block w-full px-2 py-1 border rounded-lg"
          value={edge.weight}
          onChange={(e) => onChange(edge.id, Number(e.target.value))}
        />
      </div>
      <button className="px-3 py-1 rounded-full border" onClick={() => onRemove(edge.id)}>
        Remove
      </button>
    </div>
  );
}

// ------------------------ Neuron Inspector ------------------------

function NeuronInspector({
  neurons,
  selectedIds,
  onLabel,
  onRole,
  onLabelVisible,
  onLabelOffset,
  onResetLabelOffset,
}: {
  neurons: Neuron[];
  selectedIds: string[];
  onLabel: (id: string, label: string) => void;
  onRole: (id: string, role: Role) => void;
  onLabelVisible: (id: string, visible: boolean) => void;
  onLabelOffset: (id: string, offsetX: number, offsetY: number) => void;
  onResetLabelOffset: (id: string) => void;
}) {
  if (selectedIds.length !== 1) return null;

  const neuron = neurons.find((n) => n.id === selectedIds[0]) || null;
  if (!neuron)
    return null;
  const baseClass = "p-3 rounded-2xl border shadow-sm bg-white min-h-[260px]";
  const labelVisible = neuron.labelVisible === true;
  const labelOffsetX = Number.isFinite(neuron.labelOffsetX) ? (neuron.labelOffsetX as number) : 0;
  const labelOffsetY = Number.isFinite(neuron.labelOffsetY)
    ? (neuron.labelOffsetY as number)
    : DEFAULT_LABEL_OFFSET_Y;
  return (
    <div className={`${baseClass} space-y-2`}>
      <div className="text-lg font-semibold">Neuron Inspector</div>
      <div className="grid grid-cols-3 items-center gap-2">
        <label className="col-span-1">Label</label>
        <input className="col-span-2 px-2 py-1 border rounded-lg" value={neuron.label} onChange={(e) => onLabel(neuron.id, e.target.value)} />

        <label className="col-span-1">Role</label>
        <select className="col-span-2 px-2 py-1 border rounded-lg" value={neuron.role} onChange={(e) => onRole(neuron.id, e.target.value as Role)}>
          <option value="input">Input</option>
          <option value="hidden">Hidden</option>
          <option value="output">Output</option>
        </select>

        <label className="col-span-1">Show Label</label>
        <input
          className="col-span-2"
          type="checkbox"
          checked={labelVisible}
          onChange={(e) => onLabelVisible(neuron.id, e.target.checked)}
        />

        <label className="col-span-1">Offset X</label>
        <input
          className="col-span-2 px-2 py-1 border rounded-lg"
          type="number"
          value={Math.round(labelOffsetX)}
          onChange={(e) => onLabelOffset(neuron.id, Number(e.target.value), labelOffsetY)}
        />

        <label className="col-span-1">Offset Y</label>
        <input
          className="col-span-2 px-2 py-1 border rounded-lg"
          type="number"
          value={Math.round(labelOffsetY)}
          onChange={(e) => onLabelOffset(neuron.id, labelOffsetX, Number(e.target.value))}
        />

        <div className="col-span-3 flex justify-end">
          <button className="px-3 py-1 rounded-full border" onClick={() => onResetLabelOffset(neuron.id)}>
            Reset Label Offset
          </button>
        </div>
      </div>
    </div>
  );
}

function ModuleNeuronInspector({
  neurons,
  selectedIds,
  inputIds,
  outputIds,
  onLabel,
  onRole,
  onInput,
  onOutput,
  onLabelVisible,
  onLabelOffset,
  onResetLabelOffset,
}: {
  neurons: Neuron[];
  selectedIds: string[];
  inputIds: string[];
  outputIds: string[];
  onLabel: (id: string, label: string) => void;
  onRole: (id: string, role: Role) => void;
  onInput: (id: string, flag: boolean) => void;
  onOutput: (id: string, flag: boolean) => void;
  onLabelVisible: (id: string, visible: boolean) => void;
  onLabelOffset: (id: string, offsetX: number, offsetY: number) => void;
  onResetLabelOffset: (id: string) => void;
}) {
  if (selectedIds.length === 0)
    return (
      <div className="p-3 rounded-xl border bg-white">
        <div className="font-medium">Module Neuron</div>
        <div className="text-gray-500 text-sm">Select a module neuron to edit its metadata.</div>
      </div>
    );

  if (selectedIds.length > 1)
    return (
      <div className="p-3 rounded-xl border bg-white">
        <div className="font-medium">Module Neuron</div>
        <div className="text-gray-500 text-sm">Multiple neurons selected ({selectedIds.length}).</div>
        <div className="text-gray-500 text-sm">Use a single selection to edit labels and I/O flags.</div>
      </div>
    );

  const neuron = neurons.find((n) => n.id === selectedIds[0]) || null;
  if (!neuron)
    return (
      <div className="p-3 rounded-xl border bg-white">
        <div className="font-medium">Module Neuron</div>
        <div className="text-gray-500 text-sm">Neuron not found.</div>
      </div>
    );

  const isInput = inputIds.includes(neuron.id);
  const isOutput = outputIds.includes(neuron.id);
  const labelVisible = neuron.labelVisible === true;
  const labelOffsetX = Number.isFinite(neuron.labelOffsetX) ? (neuron.labelOffsetX as number) : 0;
  const labelOffsetY = Number.isFinite(neuron.labelOffsetY)
    ? (neuron.labelOffsetY as number)
    : DEFAULT_LABEL_OFFSET_Y;

  return (
    <div className="p-3 rounded-xl border bg-white space-y-2">
      <div className="font-medium">Module Neuron</div>
      <div className="grid grid-cols-3 items-center gap-2">
        <label className="col-span-1">Label</label>
        <input
          className="col-span-2 px-2 py-1 border rounded-lg"
          value={neuron.label}
          onChange={(e) => onLabel(neuron.id, e.target.value)}
        />

        <label className="col-span-1">Role</label>
        <select
          className="col-span-2 px-2 py-1 border rounded-lg"
          value={neuron.role}
          onChange={(e) => onRole(neuron.id, e.target.value as Role)}
        >
          <option value="input">Input</option>
          <option value="hidden">Hidden</option>
          <option value="output">Output</option>
        </select>

        <label className="col-span-1">Show Label</label>
        <input
          className="col-span-2"
          type="checkbox"
          checked={labelVisible}
          onChange={(e) => onLabelVisible(neuron.id, e.target.checked)}
        />

        <label className="col-span-1">Offset X</label>
        <input
          className="col-span-2 px-2 py-1 border rounded-lg"
          type="number"
          value={Math.round(labelOffsetX)}
          onChange={(e) => onLabelOffset(neuron.id, Number(e.target.value), labelOffsetY)}
        />

        <label className="col-span-1">Offset Y</label>
        <input
          className="col-span-2 px-2 py-1 border rounded-lg"
          type="number"
          value={Math.round(labelOffsetY)}
          onChange={(e) => onLabelOffset(neuron.id, labelOffsetX, Number(e.target.value))}
        />

        <label className="col-span-1">Module Input</label>
        <input
          className="col-span-2"
          type="checkbox"
          checked={isInput}
          onChange={(e) => onInput(neuron.id, e.target.checked)}
        />

        <label className="col-span-1">Module Output</label>
        <input
          className="col-span-2"
          type="checkbox"
          checked={isOutput}
          onChange={(e) => onOutput(neuron.id, e.target.checked)}
        />

        <div className="col-span-3 flex justify-end">
          <button className="px-3 py-1 rounded-full border" onClick={() => onResetLabelOffset(neuron.id)}>
            Reset Label Offset
          </button>
        </div>
      </div>
    </div>
  );
}

// ------------------------ Raster Plot ------------------------

function RasterPlot({
  neurons,
  spikeTrains,
  T,
  selectedNeuronIds,
}: {
  neurons: Neuron[];
  spikeTrains: Record<string, number[]>;
  T: number;
  selectedNeuronIds: string[];
}) {
  const selectedSet = useMemo(() => new Set(selectedNeuronIds), [selectedNeuronIds]);
  const displayNeurons = selectedNeuronIds.length ? neurons.filter((n) => selectedSet.has(n.id)) : neurons;

  const height = Math.max(120, 20 * displayNeurons.length + 40);
  const width = 800;
  const padLeft = 60,
    padRight = 20,
    padTop = 20,
    padBottom = 30;

  // x scale
  function x(t: number) {
    return padLeft + ((width - padLeft - padRight) * t) / Math.max(1e-9, T);
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height }} className="rounded-xl border bg-white">
      {/* Axes */}
      <line x1={padLeft} y1={padTop} x2={padLeft} y2={height - padBottom} stroke="#e5e7eb" />
      <line x1={padLeft} y1={height - padBottom} x2={width - padRight} y2={height - padBottom} stroke="#e5e7eb" />

      {/* X ticks */}
      {Array.from({ length: 6 }).map((_, i) => {
        const t = (T * i) / 5;
        const xx = x(t);
        return (
          <g key={i}>
            <line x1={xx} y1={height - padBottom} x2={xx} y2={height - padBottom + 4} stroke="#9ca3af" />
            <text x={xx} y={height - padBottom + 16} textAnchor="middle" fontSize={10} fill="#6b7280">
              {t.toFixed(1)}
            </text>
          </g>
        );
      })}

      {/* Rows */}
      {displayNeurons.map((n, row) => {
        const y = padTop + 20 + row * 20;
        const color = n.role === "input" ? "#60a5fa" : n.role === "output" ? "#f59e0b" : "#a78bfa";
        const spikes = (spikeTrains[n.id] || []).filter((s) => s <= T);
        return (
          <g key={n.id}>
            <text x={10} y={y + 4} fontSize={11} fill="#374151">
              {n.label}
            </text>
            <line x1={padLeft} y1={y} x2={width - padRight} y2={y} stroke="#181c24ff" />
            {spikes.map((s, i) => (
              <line key={i} x1={x(s)} x2={x(s)} y1={y - 6} y2={y + 6} stroke={color} strokeWidth={2} />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

// ------------------------ Potential Plot ------------------------

function PotentialPlot({ neurons: hiddenNeurons, series, T }: { neurons: Neuron[]; series: PotentialSeries | null; T: number }) {
  if (!series || series.times.length === 0) {
    return <div className="text-gray-500 text-sm">Run the simulation to plot potentials.</div>;
  }

  if (!hiddenNeurons.length) {
    return <div className="text-gray-500 text-sm">No hidden neurons available for potential traces.</div>;
  }

  const width = 800;
  const height = 280;
  const padLeft = 60;
  const padRight = 20;
  const padTop = 10;
  const padBottom = 40;

  const maxTime = Math.max(series.times[series.times.length - 1] ?? 0, T);
  const spanTime = Math.max(1e-9, maxTime);

  let maxVal = 0;
  for (const neuron of hiddenNeurons) {
    const arr = series.values[neuron.id];
    if (!arr) continue;
    for (const v of arr) if (v > maxVal) maxVal = v;
  }
  if (!Number.isFinite(maxVal) || maxVal <= 0) maxVal = 1;
  else maxVal *= 1.05; // add a little headroom
  const spanVal = Math.max(1e-9, maxVal);

  function xCoord(t: number) {
    return padLeft + ((width - padLeft - padRight) * t) / spanTime;
  }

  function yCoord(v: number) {
    const clamped = v < 0 ? 0 : v > spanVal ? spanVal : v;
    const usableHeight = height - padTop - padBottom;
    return padTop + usableHeight * (1 - clamped / spanVal);
  }

  const tickCount = 6;

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height }} className="rounded-xl border bg-white">
        {/* Axes */}
        <line x1={padLeft} y1={padTop} x2={padLeft} y2={height - padBottom} stroke="#e5e7eb" />
        <line x1={padLeft} y1={height - padBottom} x2={width - padRight} y2={height - padBottom} stroke="#e5e7eb" />

        {/* X ticks */}
        {Array.from({ length: tickCount }).map((_, i) => {
          const t = (spanTime * i) / (tickCount - 1);
          const xx = xCoord(t);
          return (
            <g key={`xtick-${i}`}>
              <line x1={xx} y1={height - padBottom} x2={xx} y2={height - padBottom + 4} stroke="#9ca3af" />
              <text x={xx} y={height - padBottom + 16} textAnchor="middle" fontSize={10} fill="#6b7280">
                {t.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Y ticks */}
        {Array.from({ length: tickCount }).map((_, i) => {
          const v = (spanVal * i) / (tickCount - 1);
          const yy = yCoord(v);
          return (
            <g key={`ytick-${i}`}>
              <line x1={padLeft - 4} y1={yy} x2={padLeft} y2={yy} stroke="#9ca3af" />
              <text x={padLeft - 8} y={yy + 4} textAnchor="end" fontSize={10} fill="#6b7280">
                {v.toFixed(2)}
              </text>
              <line x1={padLeft} y1={yy} x2={width - padRight} y2={yy} stroke="#f3f4f6" />
            </g>
          );
        })}

        {/* Potential traces */}
        {hiddenNeurons.map((n) => {
          const values = series.values[n.id];
          if (!values || values.length !== series.times.length) return null;
          const color = "#7c3aed";
          const path = series.times
            .map((t, idx) => {
              const v = values[idx] ?? 0;
              const cmd = idx === 0 ? "M" : "L";
              return `${cmd} ${xCoord(t)} ${yCoord(v)}`;
            })
            .join(" ");
          return <path key={n.id} d={path} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />;
        })}
      </svg>

      <div className="flex flex-wrap gap-3 text-xs">
        {hiddenNeurons.map((n) => {
          const color = "#7c3aed";
          return (
            <div key={n.id} className="flex items-center gap-2">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-gray-700">{n.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ------------------------ Helpers ------------------------

function jitterPoissonText(T: number, lambdaHz: number): string {
  // Simple Poisson process via exponential inter-arrivals
  const out: number[] = [];
  let t = 0;
  while (t < T) {
    const u = Math.random();
    const dt = -Math.log(1 - u) / Math.max(1e-9, lambdaHz);
    t += dt;
    if (t < T) out.push(t);
  }
  return out.map((v) => Number(v.toFixed(3))).join(" ");
}

function integerSpikeText(T: number): string {
  const limit = Math.floor(T);
  if (limit < 1) return "";
  return Array.from({ length: limit }, (_, i) => String(i + 1)).join(" ");
}

function rgbToHex(r: number, g: number, b: number): string {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  const toHex = (v: number) => clamp(v).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function lerpColor(from: [number, number, number], to: [number, number, number], t: number): string {
  const tt = Math.min(1, Math.max(0, t));
  const [fr, fg, fb] = from;
  const [tr, tg, tb] = to;
  const r = fr + (tr - fr) * tt;
  const g = fg + (tg - fg) * tt;
  const b = fb + (tb - fb) * tt;
  return rgbToHex(r, g, b);
}

function isStandardWeight(weight: number): boolean {
  const EPS = 1e-6;
  return Math.abs(weight - 1) <= EPS || Math.abs(weight - 2) <= EPS || Math.abs(weight + 2) <= EPS;
}

function weightToDashPattern(weight: number): string | undefined {
  const EPS = 1e-6;
  if (weight < -EPS) return "8 4";
  if (Math.abs(weight - 1) <= EPS) return "1 8";
  if (weight > 1 + EPS) return undefined;
  return undefined;
}

function weightToColor(weight: number): string {
  const EPS = 1e-9;
  if (weight > 1 + EPS) return "#000000";
  if (Math.abs(weight - 1) <= EPS) return "#2563eb";
  if (weight >= 0) {
    return lerpColor([255, 255, 255], [37, 99, 235], weight);
  }
  if (weight <= -1 - EPS || Math.abs(weight + 1) <= EPS) return "#dc2626";
  return lerpColor([255, 255, 255], [220, 38, 38], -weight);
}
