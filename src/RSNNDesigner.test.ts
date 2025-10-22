import { describe, expect, it } from "vitest";
import {
  EDGE_NODE_PADDING,
  EDGE_TARGET_TRIM,
  EDGE_END_CLEARANCE,
  GRID_SIZE,
  NODE_RADIUS,
  buildEdgePoints,
  binaryIncludes,
  distancePointToSegment,
  offsetPoint,
  parseTimes,
  pathMidpoint,
  pointAlongPath,
  sanitizeWaypoints,
  simulate,
  snapToGrid,
  totalPathLength,
  uniqueLabel,
  uniqueSorted,
  waypointInsertIndex,
  projectPointToSegment,
  closestPointOnPath,
  expDecay,
  type Edge,
  type Neuron,
  type Point,
} from "./RSNNDesigner";

const makeNeuron = (id: string, role: Neuron["role"], x: number, y: number): Neuron => ({
  id,
  label: id,
  role,
  x,
  y,
});

const makeEdge = (id: string, sourceId: string, targetId: string, weight: number, points?: Point[]): Edge => ({
  id,
  sourceId,
  targetId,
  weight,
  points,
});

describe("utility helpers", () => {
  it("creates unique labels from a base string", () => {
    const taken = new Set<string>(["N"]);
    expect(uniqueLabel(" N ", taken)).toBe("N_2");
    expect(uniqueLabel("custom", taken)).toBe("custom");
    const fresh = new Set<string>();
    expect(uniqueLabel("   ", fresh)).toBe("N");
    expect(taken.has("custom")).toBe(true);
  });

  it("parses spike time text, keeping positive sorted numbers", () => {
    const result = parseTimes("0, 1, 2 2 3\n4,5,notanumber,-1,3");
    expect(result).toEqual([1, 2, 2, 3, 3, 4, 5]);
  });

  it("deduplicates nearly identical sorted numbers", () => {
    const nums = [0, 1, 1 + 1e-13, 1.5, 1.5 + 2e-12];
    expect(uniqueSorted(nums)).toEqual([0, 1, 1.5, 1.500000000002]);
  });

  it("computes exponential decay with edge cases", () => {
    expect(expDecay(-1, 5)).toBe(1);
    expect(expDecay(2, Number.POSITIVE_INFINITY)).toBe(1);
    expect(expDecay(2, 0)).toBe(0);
    expect(expDecay(2, 4)).toBeCloseTo(Math.exp(-0.5));
  });

  it("snaps values to the grid size and handles invalid input", () => {
    expect(snapToGrid(33)).toBe(Math.round(33 / GRID_SIZE) * GRID_SIZE);
    expect(snapToGrid(-7)).toBeCloseTo(0);
    expect(snapToGrid(Number.NaN)).toBe(0);
  });

  it("sanitizes waypoints by removing invalid entries", () => {
    const points = sanitizeWaypoints([
      { x: 1, y: 2 },
      // @ts-expect-error testing invalid point
      { x: Number.NaN, y: 5 },
      { x: 3, y: 4 },
    ]);
    expect(points).toEqual([
      { x: 1, y: 2 },
      { x: 3, y: 4 },
    ]);
  });

  it("builds edge points with start, waypoints, and trimmed end", () => {
    const source = makeNeuron("A", "input", 0, 0);
    const target = makeNeuron("B", "output", 100, 0);
    const edge = makeEdge("e1", source.id, target.id, 2, [{ x: 60, y: 20 }]);
    const points = buildEdgePoints(edge, source, target);

    const expectedStart = offsetPoint(
      { x: source.x, y: source.y },
      { x: 60, y: 20 },
      NODE_RADIUS + EDGE_NODE_PADDING
    );
    expect(points).toHaveLength(3);
    expect(points[0].x).toBeCloseTo(expectedStart.x);
    expect(points[0].y).toBeCloseTo(expectedStart.y);
    expect(points[1]).toEqual({ x: 60, y: 20 });

    const expectedEndOffset =
      Math.max(NODE_RADIUS + EDGE_END_CLEARANCE, NODE_RADIUS + EDGE_NODE_PADDING - EDGE_TARGET_TRIM);
    const expectedEnd = offsetPoint({ x: target.x, y: target.y }, { x: 60, y: 20 }, expectedEndOffset);
    expect(points[2].x).toBeCloseTo(expectedEnd.x);
    expect(points[2].y).toBeCloseTo(expectedEnd.y);
  });

  it("computes path lengths, positions, and midpoints", () => {
    const path: Point[] = [
      { x: 0, y: 0 },
      { x: 3, y: 4 },
    ];
    expect(totalPathLength(path)).toBe(5);
    expect(pointAlongPath(path, 0)).toEqual({ x: 0, y: 0 });
    expect(pointAlongPath(path, 1)).toEqual({ x: 3, y: 4 });
    expect(pointAlongPath(path, 0.5)).toEqual({ x: 1.5, y: 2 });
    expect(pathMidpoint(path)).toEqual({ x: 1.5, y: 2 });
  });

  it("computes distances and projections on segments", () => {
    const a = { x: 0, y: 0 };
    const b = { x: 6, y: 0 };
    const p = { x: 3, y: 4 };
    expect(distancePointToSegment(p, a, b)).toBe(4);
    expect(projectPointToSegment(p, a, b)).toEqual({ x: 3, y: 0 });
  });

  it("finds the closest point on a polyline path", () => {
    const path: Point[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 20, y: 10 },
    ];
    const click = { x: 11, y: 4 };
    const result = closestPointOnPath(click, path);
    expect(result).not.toBeNull();
    expect(result?.segment).toBe(1);
    expect(result?.point.x).toBeCloseTo(10);
    expect(result?.point.y).toBeCloseTo(4);
  });

  it("selects waypoint insert indices based on segment proximity", () => {
    const path: Point[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 20, y: 10 },
    ];
    const waypointCount = 2;
    const nearFirstSegment = waypointInsertIndex(path, waypointCount, { x: 5, y: 2 });
    const nearSecondSegment = waypointInsertIndex(path, waypointCount, { x: 9, y: 6 });
    const nearThirdSegment = waypointInsertIndex(path, waypointCount, { x: 15, y: 9 });

    expect(nearFirstSegment).toBe(0);
    expect(nearSecondSegment).toBe(1);
    expect(nearThirdSegment).toBe(2);
  });
});

describe("simulation", () => {
  it("propagates spikes through a feed-forward edge when above threshold", () => {
    const neurons = [
      makeNeuron("A", "input", 0, 0),
      makeNeuron("B", "output", 100, 0),
    ];
    const edges = [makeEdge("e1", "A", "B", 1.5)];
    const result = simulate(neurons, edges, 5, 2, { A: [1] });

    expect(result.spikeTrains["A"]).toEqual([1]);
    expect(result.spikeTrains["B"]).toEqual([1]);
    expect(result.finalP["B"]).toBeCloseTo(0);
    expect(result.cascadeFrames).toHaveLength(2);
    expect(result.cascadeFrames[0]).toMatchObject({
      time: 1,
      wave: 0,
      sources: [],
      targets: ["A"],
      edges: [],
    });
    expect(result.cascadeFrames[1]).toMatchObject({
      time: 1,
      wave: 1,
      sources: ["A"],
      targets: ["B"],
      edges: ["e1"],
    });
  });

  it("retains subthreshold potential without triggering spikes", () => {
    const neurons = [
      makeNeuron("A", "input", 0, 0),
      makeNeuron("B", "output", 100, 0),
    ];
    const edges = [makeEdge("e1", "A", "B", 0.8)];
    const result = simulate(neurons, edges, Number.POSITIVE_INFINITY, 2, { A: [1] });

    expect(result.spikeTrains["B"]).toEqual([]);
    expect(result.finalP["B"]).toBeCloseTo(0.8);
    expect(result.spikeTrains["A"]).toEqual([1]);
  });
});

describe("binary search helpers", () => {
  it("checks membership with tolerance", () => {
    const arr = [1, 2, 3];
    expect(binaryIncludes(arr, 2)).toBe(true);
    expect(binaryIncludes(arr, 2 + 5e-10)).toBe(true);
    expect(binaryIncludes(arr, 2 + 2e-8)).toBe(false);
  });
});
