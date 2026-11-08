"use strict";

const ProtenixPathChart = (() => {
  const names = {relaxed: "Relaxed contact loss", chemistry: "Repair loss", valence: "Valence excess"};

  function mount(host, selectMetric, status, selectPoint) {
    let points = [], index = 0, position = null;
    const observer = new ResizeObserver(draw);
    observer.observe(host);

    function draw() {
      const key = selectMetric.value;
      const width = Math.max(280, Math.round(host.clientWidth));
      const height = 250;
      const left = 52, right = width - 16, top = 16, bottom = height - 38;
      const values = points.map(point => point[key]).filter(Number.isFinite);
      if (!values.length) {
        host.innerHTML = empty("This path has no saved " + names[key].toLowerCase() + ".");
        position = null;
        return;
      }
      const minimum = Math.min(...values), maximum = Math.max(...values);
      const pad = Math.max((maximum - minimum) * 0.08, Math.abs(minimum) * 0.005, 0.0001);
      const lo = minimum - pad, hi = maximum + pad;
      const start = points[0].depth, span = Math.max(1, points.at(-1).depth - start);
      const x = depth => left + (depth - start) / span * (right - left);
      const y = value => bottom - (value - lo) / (hi - lo) * (bottom - top);
      position = {x, y, width, left, right, start, span};
      const ticks = [];
      for (let tick = 0; tick <= 4; tick++) {
        const value = lo + (hi - lo) * tick / 4;
        const yy = y(value);
        ticks.push('<line class="chart-grid" x1="' + left + '" y1="' + yy + '" x2="' + right +
          '" y2="' + yy + '"/><text x="' + (left - 7) + '" y="' + (yy + 4) +
          '" text-anchor="end">' + fmt(value, Math.abs(hi - lo) < 0.02 ? 4 : 2) + '</text>');
      }
      const tickCount = width < 470 ? 3 : 6;
      const lastDepth = points.at(-1).depth;
      for (let tick = 0; tick <= tickCount; tick++) {
        if (start === lastDepth && tick) break;
        const depth = Math.round(start + (lastDepth - start) * tick / tickCount);
        const xx = x(depth);
        ticks.push('<text x="' + xx + '" y="' + (bottom + 17) +
          '" text-anchor="middle">' + count(depth) + '</text>');
      }
      let coordinates = "", hasPrevious = false;
      const forks = [], segments = [];
      for (const point of points) {
        const xx = x(point.depth);
        if (point.restart) forks.push('<line class="chart-restart" x1="' + xx + '" x2="' + xx +
          '" y1="' + top + '" y2="' + bottom + '"><title>Restart at path step ' +
          count(point.depth) + '</title></line>');
        if (point.restart || point.gap || !Number.isFinite(point[key])) {
          if (coordinates) segments.push(coordinates);
          coordinates = "";
          hasPrevious = false;
        }
        if (!Number.isFinite(point[key])) continue;
        coordinates += (hasPrevious ? " L" : "M") + xx.toFixed(2) + " " + y(point[key]).toFixed(2);
        hasPrevious = true;
      }
      if (coordinates) segments.push(coordinates);
      const line = segments.map(segment => '<path class="chart-loss-line" d="' +
        (segment.includes(" L") ? segment : segment + " l0.05 0") + '"/>').join("");
      host.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height + '" role="img" aria-label="' +
        names[key] + ' along the selected path. Click to select an update.">' +
        ticks.join("") + forks.join("") + line +
        '<text class="chart-axis-label" x="' + (left + right) / 2 + '" y="' + (height - 4) +
        '" text-anchor="middle">Updates along the selected path</text>' +
        '<line class="chart-cursor" data-chart-cursor y1="' + top + '" y2="' + bottom + '"/>' +
        '<circle class="chart-point" data-chart-point r="4.4"/></svg>';
      show(index);
    }

    function show(next) {
      index = next;
      const point = points[index];
      if (!point || !position) {
        status.textContent = "";
        return;
      }
      const value = point[selectMetric.value];
      const xx = position.x(point.depth);
      const cursor = host.querySelector("[data-chart-cursor]");
      const dot = host.querySelector("[data-chart-point]");
      cursor?.setAttribute("x1", xx);
      cursor?.setAttribute("x2", xx);
      if (dot) {
        dot.style.display = Number.isFinite(value) ? "" : "none";
        dot.setAttribute("cx", xx);
        if (Number.isFinite(value)) dot.setAttribute("cy", position.y(value));
      }
      status.textContent = names[selectMetric.value] + " · " +
        (Number.isFinite(value) ? fmt(value, 5) : "No saved value");
    }

    host.addEventListener("click", event => {
      const svg = host.querySelector("svg");
      if (!position || !svg || !points.length) return;
      const rect = svg.getBoundingClientRect();
      const chartX = (event.clientX - rect.left) * position.width / rect.width;
      const depth = position.start + (chartX - position.left) / (position.right - position.left) * position.span;
      let lo = 0, hi = points.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (points[mid].depth < depth) lo = mid + 1;
        else hi = mid;
      }
      if (lo && Math.abs(points[lo - 1].depth - depth) < Math.abs(points[lo].depth - depth)) lo--;
      selectPoint(lo);
    });
    selectMetric.addEventListener("change", draw);

    function set(next) {
      points = next;
      for (const option of selectMetric.options) {
        option.disabled = !points.some(point => Number.isFinite(point[option.value]));
      }
      if (selectMetric.selectedOptions[0]?.disabled) selectMetric.value = "relaxed";
      index = 0;
      draw();
    }

    return {set, show, destroy: () => observer.disconnect()};
  }

  return {mount};
})();
