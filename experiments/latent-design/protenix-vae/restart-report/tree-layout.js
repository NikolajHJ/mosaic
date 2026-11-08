"use strict";

// Step-resolved lineage for saved observations. World y decreases as the
// accumulated update depth grows, leaving each lineage root at the bottom.
globalThis.ProtenixTree = (() => {
  const number = value => value !== null && value !== "" && Number.isInteger(Number(value))
    ? Number(value) : null;
  const cycleNumber = value => {
    const match = /^cycle-(\d+)$/i.exec(String(value ?? ""));
    return match ? Number(match[1]) : number(value);
  };
  const parts = branch => String(branch ?? "").split(/[\\/]/).filter(Boolean);
  const branchName = branch => parts(branch).at(-1) || "";
  const pathCycle = branch => parts(branch).slice(0, -1)
    .map(part => /^cycle-\d+$/i.test(part) ? cycleNumber(part) : part === "paired" ? 0 : null)
    .find(value => value !== null) ?? null;
  const branchKey = (branch, cycle) => `${cycle}/${branchName(branch)}`;
  const identity = row => row[1] && row[2] ? `smiles:${row[2]}` : !row[1] && row[3] ? `raw:${row[3]}` : null;

  function stateGroups(lane) {
    const anchors = new Set((lane.branch.repeat_anchor_steps || []).map(number));
    const groups = [];
    for (const step of lane.steps) {
      const row = lane.rows.get(step), key = identity(row), prev = groups.at(-1);
      if (key && prev?.key === key && (anchors.has(step) || step === prev.steps.at(-1) + 1)) {
        prev.steps.push(step);
      } else {
        groups.push({row, key, steps: [step]});
      }
    }
    return groups;
  }

  function lowerBound(items, target, key) {
    let lo = 0, hi = items.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (key(items[mid]) < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function build(job, {stepY = 4, laneGap = 16, rootGap = 24} = {}) {
    if (![stepY, laneGap, rootGap].every(value => Number.isFinite(value) && value > 0)) {
      throw new RangeError("Tree spacing must be finite and positive");
    }
    const lanes = [], exact = new Map(), byName = new Map();
    for (const [ordinal, branch] of (job?.branches || []).entries()) {
      const cycle = cycleNumber(branch.cycle) ?? pathCycle(branch.name) ?? 0;
      const name = branchName(branch.name);
      if (!name) continue;
      const rows = new Map();
      for (const row of branch.states || []) {
        const step = Array.isArray(row) ? number(row[0]) : null;
        if (step !== null && step >= 0 && !rows.has(step)) rows.set(step, row);
      }
      if (!rows.size) continue;
      const steps = [...rows.keys()].sort((a, b) => a - b);
      const key = branchKey(name, cycle);
      const lane = {id: `${key}#${ordinal}`, key, name, cycle, branch, steps, rows,
        attachment: branch.source ? "unresolved" : "root", source: null, nodes: [], edges: [],
        byStep: new Map(), state: 0, ordinal};
      lanes.push(lane);
      if (!exact.has(key)) exact.set(key, []);
      if (!byName.has(name)) byName.set(name, []);
      exact.get(key).push(lane);
      byName.get(name).push(lane);
    }

    function lookup(branch, cycle) {
      const name = branchName(branch);
      const explicit = cycleNumber(cycle), embedded = pathCycle(branch);
      if (explicit !== null && embedded !== null && explicit !== embedded) return null;
      const knownCycle = explicit ?? embedded;
      const options = knownCycle === null ? byName.get(name) : exact.get(branchKey(name, knownCycle));
      return options?.length === 1 ? options[0] : null;
    }

    for (const lane of lanes) {
      const hint = lane.branch.source;
      if (!hint) continue;
      const parent = lookup(hint.branch, hint.cycle);
      const step = number(hint.step);
      if (!parent || parent === lane || !parent.rows.has(step) || !lane.rows.has(0)) continue;
      lane.source = {lane: parent, step};
    }

    function setDepth(lane) {
      if (lane.state === 2) return;
      lane.state = 1;
      if (lane.source?.lane.state === 1) lane.source = null;
      if (lane.source) {
        setDepth(lane.source.lane);
        lane.baseDepth = lane.source.lane.baseDepth + lane.source.step + 1;
        lane.attachment = "exact";
      } else {
        lane.baseDepth = 0;
      }
      lane.startDepth = lane.baseDepth + lane.steps[0];
      lane.endDepth = lane.baseDepth + lane.steps.at(-1);
      lane.state = 2;
    }
    lanes.forEach(setDepth);
    lanes.sort((a, b) => a.startDepth - b.startDepth || a.cycle - b.cycle || a.ordinal - b.ordinal);

    // Pack branch intervals into nearby lanes. A child can continue straight
    // up its parent's lane once the parent's observed tail has ended.
    const laneEnds = new Map(), fan = new Map();
    const rootCount = lanes.filter(lane => !lane.source).length;
    const rootStride = rootCount <= 3 ? Math.max(1, Math.round(rootGap / laneGap)) : 1;
    let nextRootLane = 0;
    for (const lane of lanes) {
      const free = index => !laneEnds.has(index) || laneEnds.get(index) < lane.startDepth;
      if (!lane.source) {
        while (!free(nextRootLane)) nextRootLane += rootStride;
        lane.column = nextRootLane;
        nextRootLane += rootStride;
      } else {
        const parent = lane.source.lane;
        const sibling = fan.get(parent.id) || 0;
        fan.set(parent.id, sibling + 1);
        const side = sibling % 2 === 0 ? 1 : -1;
        let column = parent.column;
        for (let distance = 1; !free(column); distance++) {
          const first = parent.column + side * distance;
          const second = parent.column - side * distance;
          if (free(first)) column = first;
          else if (free(second)) column = second;
        }
        lane.column = column;
      }
      laneEnds.set(lane.column, lane.endDepth);
    }

    const minColumn = lanes.reduce((min, lane) => Math.min(min, lane.column), Infinity);
    const maxColumn = lanes.reduce((max, lane) => Math.max(max, lane.column), -Infinity);
    const center = lanes.length ? (minColumn + maxColumn) / 2 : 0;
    const nodes = [], edges = [], roots = [], restartEdges = [];
    for (const lane of lanes) {
      lane.x = (lane.column - center) * laneGap;
      lane.minY = -lane.endDepth * stepY;
      lane.maxY = -lane.startDepth * stepY;
      let previous = lane.source?.lane.byStep.get(lane.source.step) || null;
      for (const [offset, group] of stateGroups(lane).entries()) {
        const {row, steps} = group, step = steps[0];
        const depth = lane.baseDepth + step;
        const node = {index: nodes.length, id: `${lane.id}:${step}`, branch: lane.name,
          laneId: lane.id, branchKey: lane.key, cycle: lane.cycle, arm: lane.branch.arm, step, depth,
          valid: row[1], smiles: row[2] ?? null, rawKey: row[3] ?? null, relaxedLoss: row[4] ?? null,
          x: lane.x, y: -depth * stepY, parentIndex: previous?.index ?? null, incoming: null};
        if (steps.length > 1) node.endStep = steps.at(-1);
        nodes.push(node);
        lane.nodes.push(node);
        for (const alias of steps) lane.byStep.set(alias, node);
        if (previous) {
          const kind = offset === 0 ? "restart" : "step";
          const edge = {source: previous, target: node, kind,
            sourceStep: kind === "restart" ? lane.source.step : previous.step,
            missingUpdates: kind === "step" ? step - (previous.endStep ?? previous.step) - 1 : 0};
          node.incoming = edge;
          edges.push(edge);
          if (kind === "restart") restartEdges.push(edge);
          else lane.edges.push(edge);
        } else {
          roots.push(node);
        }
        previous = node;
      }
      delete lane.rows;
    }

    function nodeAt(branch, cycle, step) {
      return lookup(branch, cycle)?.byStep.get(number(step)) || null;
    }

    function ancestry(nodeOrIndex) {
      let node = typeof nodeOrIndex === "number" ? nodes[nodeOrIndex] : nodeOrIndex;
      const path = [], links = [];
      while (node) {
        path.push(node);
        if (node.incoming) links.push(node.incoming);
        node = node.parentIndex === null ? null : nodes[node.parentIndex];
      }
      return {nodes: path.reverse(), edges: links.reverse()};
    }

    const laneById = new Map(lanes.map(lane => [lane.id, lane]));

    function timeline(route) {
      const points = [];
      let distinctOnly = false;
      for (let start = 0; start < route.nodes.length;) {
        const first = route.nodes[start];
        const lane = laneById.get(first.laneId);
        let next = start + 1;
        while (next < route.nodes.length && route.nodes[next].laneId === first.laneId) next++;
        const last = route.nodes[next - 1];
        const fork = route.edges[next - 1];
        const full = Array.isArray(lane.branch.loss_series) && lane.branch.loss_series.length > 0;
        distinctOnly ||= !full;
        const saved = full ? lane.branch.loss_series : lane.branch.states.map(row => [row[0], row[4]]);
        const lastSavedStep = saved.at(-1)?.[0] ?? last.step;
        const cutoff = fork?.kind === "restart" ? fork.sourceStep
          : last === lane.nodes.at(-1) ? Math.max(last.step, lastSavedStep) : last.step;
        let state = start;
        let previousStep = null;
        for (const row of saved) {
          const [step, relaxed, chemistry, weight, valence] = row;
          if (step < first.step || step > cutoff) continue;
          while (state + 1 < next && route.nodes[state + 1].step <= step) state++;
          points.push({node: route.nodes[state], step, depth: lane.baseDepth + step,
            relaxed, chemistry, weight, valence, restart: previousStep === null && start > 0,
            gap: previousStep !== null && step !== previousStep + 1});
          previousStep = step;
        }
        start = next;
      }
      return {points, distinctOnly};
    }

    const byX = [...lanes].sort((a, b) => a.x - b.x || a.minY - b.minY);
    const bounds = {
      minX: byX[0]?.x ?? 0,
      maxX: byX.at(-1)?.x ?? 0,
      minY: lanes.reduce((min, lane) => Math.min(min, lane.minY), 0),
      maxY: 0
    };
    bounds.width = bounds.maxX - bounds.minX;
    bounds.height = bounds.maxY - bounds.minY;

    function visible(x0, y0, x1, y1) {
      const left = Math.min(x0, x1), right = Math.max(x0, x1);
      const top = Math.min(y0, y1), bottom = Math.max(y0, y1);
      const inView = {nodes: [], edges: [], lanes: []};
      const first = lowerBound(byX, left, lane => lane.x);
      for (let i = first; i < byX.length && byX[i].x <= right; i++) {
        const lane = byX[i];
        if (lane.minY > bottom || lane.maxY < top) continue;
        inView.lanes.push(lane);
        const start = lowerBound(lane.nodes, -bottom, node => -node.y);
        let end = start;
        while (end < lane.nodes.length && lane.nodes[end].y >= top) {
          inView.nodes.push(lane.nodes[end++]);
        }
        // Include both line segments crossing the viewport boundary.
        for (let j = Math.max(0, start - 1); j < Math.min(lane.edges.length, end); j++) {
          inView.edges.push(lane.edges[j]);
        }
      }
      for (const edge of restartEdges) {
        const {source, target} = edge;
        if (Math.min(source.x, target.x) <= right && Math.max(source.x, target.x) >= left
            && target.y <= bottom && source.y >= top) inView.edges.push(edge);
      }
      return inView;
    }

    function nearest(x, y, radius) {
      if (!(Number.isFinite(radius) && radius >= 0)) return null;
      let closest = null, best = radius * radius;
      for (const node of visible(x - radius, y - radius, x + radius, y + radius).nodes) {
        const distance = (node.x - x) ** 2 + (node.y - y) ** 2;
        if (distance < best || (distance === best && (!closest || node.index < closest.index))) {
          closest = node;
          best = distance;
        }
      }
      return closest;
    }

    return {nodes, edges, roots, lanes, bounds, nodeAt, ancestry, timeline, visible, nearest};
  }

  return {build, branchKey};
})();
