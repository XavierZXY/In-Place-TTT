const API_URL = "/api/ruler-results";
const AUTO_REFRESH_MS = 30000;
const COLORS = ["#166a63", "#ad5a2a", "#315f9f", "#8a5a16", "#5b6f2a", "#8f3f58", "#3b7086", "#6b5c9a"];

const state = {
  payload: null,
  view: "base",
  metric: "__overall__",
  fixedLength: null,
  selectedModels: new Set(),
  search: "",
  autoRefreshTimer: null,
};

const els = {
  dataSubtitle: document.getElementById("dataSubtitle"),
  statusBanner: document.getElementById("statusBanner"),
  resultCount: document.getElementById("resultCount"),
  lengthCount: document.getElementById("lengthCount"),
  taskCount: document.getElementById("taskCount"),
  lastScan: document.getElementById("lastScan"),
  refreshButton: document.getElementById("refreshButton"),
  autoRefreshToggle: document.getElementById("autoRefreshToggle"),
  viewPreset: document.getElementById("viewPreset"),
  metricSelect: document.getElementById("metricSelect"),
  lengthSelect: document.getElementById("lengthSelect"),
  modelSearch: document.getElementById("modelSearch"),
  selectVisibleButton: document.getElementById("selectVisibleButton"),
  clearModelsButton: document.getElementById("clearModelsButton"),
  modelList: document.getElementById("modelList"),
  lengthCompareHint: document.getElementById("lengthCompareHint"),
  bestAtLength: document.getElementById("bestAtLength"),
  barChart: document.getElementById("barChart"),
  rankingBody: document.getElementById("rankingBody"),
  lineChart: document.getElementById("lineChart"),
  taskMatrixHint: document.getElementById("taskMatrixHint"),
  taskHeatmap: document.getElementById("taskHeatmap"),
  lengthHeatmap: document.getElementById("lengthHeatmap"),
};

function formatPercent(value) {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-";
  return new Intl.NumberFormat("zh-CN").format(value);
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "-";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function formatDateTime(iso) {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString("zh-CN", { hour12: false });
}

function metricLabel(metric) {
  return metric === "__overall__" ? "overall" : metric;
}

function setStatus(message, tone = "error") {
  if (!message) {
    els.statusBanner.hidden = true;
    els.statusBanner.textContent = "";
    return;
  }
  els.statusBanner.hidden = false;
  els.statusBanner.textContent = message;
  els.statusBanner.dataset.tone = tone;
}

function getRecords() {
  return state.payload?.records ?? [];
}

function isRecordInView(record) {
  if (state.view === "all") return true;
  if (state.view === "base") return record.method === "base" && !record.isSmoke && !record.hasChatTemplate;
  if (state.view === "chat") return record.hasChatTemplate && !record.isSmoke;
  if (state.view === "smoke") return record.isSmoke;
  return true;
}

function getViewRecords() {
  return getRecords().filter(isRecordInView);
}

function getVisibleRecords() {
  const query = state.search.trim().toLowerCase();
  return getViewRecords().filter((record) => !query || record.name.toLowerCase().includes(query));
}

function getSelectedRecords() {
  return getViewRecords().filter((record) => state.selectedModels.has(record.id));
}

function getLengths(records = getViewRecords()) {
  return [...new Set(records.flatMap((record) => record.lengths))].sort((a, b) => a - b);
}

function getTasks(records = getViewRecords()) {
  return [...new Set(records.flatMap((record) => record.tasks))].sort();
}

function getMetricScore(record, length, metric = state.metric) {
  const summary = record.summaries[String(length)];
  if (!summary) return null;
  if (metric === "__overall__") return summary.overall ?? null;
  return summary.tasks?.[metric] ?? null;
}

function scoreValue(record, length, metric = state.metric) {
  const score = getMetricScore(record, length, metric);
  return Number.isFinite(score?.score) ? score.score : null;
}

function scoreN(record, length, metric = state.metric) {
  const score = getMetricScore(record, length, metric);
  return Number.isFinite(score?.n) ? score.n : null;
}

function selectedLength() {
  if (state.fixedLength !== null) return state.fixedLength;
  const lengths = getLengths();
  return lengths[0] ?? null;
}

function preserveOrSelectDefaults(previousIds = new Set()) {
  const records = getViewRecords();
  const validIds = new Set(records.map((record) => record.id));
  const kept = [...previousIds].filter((id) => validIds.has(id));
  state.selectedModels = new Set(kept.length > 0 ? kept : records.map((record) => record.id));

  const tasks = getTasks(records);
  if (state.metric !== "__overall__" && !tasks.includes(state.metric)) {
    state.metric = "__overall__";
  }

  const lengths = getLengths(records);
  if (!lengths.includes(state.fixedLength)) {
    state.fixedLength = lengths[0] ?? null;
  }
}

function updateSummary() {
  const records = getViewRecords();
  const lengths = getLengths(records);
  const tasks = getTasks(records);
  els.resultCount.textContent = formatNumber(records.length);
  els.lengthCount.textContent = lengths.length ? lengths.map(formatNumber).join(", ") : "0";
  els.taskCount.textContent = formatNumber(tasks.length);
  els.lastScan.textContent = formatDateTime(state.payload?.generatedAt);
  els.dataSubtitle.textContent = state.payload?.resultsRoot
    ? `扫描目录：${state.payload.resultsRoot}`
    : "正在加载 summary_all_lengths.json";
}

function renderMetricSelect() {
  const tasks = getTasks();
  const metrics = ["__overall__", ...tasks];
  els.metricSelect.innerHTML = metrics
    .map((metric) => `<option value="${escapeAttr(metric)}">${escapeHtml(metricLabel(metric))}</option>`)
    .join("");
  els.metricSelect.value = state.metric;
}

function renderLengthSelect() {
  const lengths = getLengths();
  els.lengthSelect.innerHTML = lengths
    .map((length) => `<option value="${length}">${formatNumber(length)}</option>`)
    .join("");
  if (state.fixedLength !== null) {
    els.lengthSelect.value = String(state.fixedLength);
  }
}

function renderModelList() {
  const records = getVisibleRecords();
  if (!records.length) {
    els.modelList.innerHTML = `<div class="empty-state">当前筛选没有匹配的结果目录</div>`;
    return;
  }

  els.modelList.innerHTML = records
    .map((record) => {
      const checked = state.selectedModels.has(record.id) ? "checked" : "";
      const methodClass = record.isSmoke ? "smoke" : record.hasChatTemplate ? "chat" : "base";
      const lengths = record.lengths.map(formatNumber).join(", ");
      const template = record.hasChatTemplate ? "chat template" : "no template";
      return `
        <label class="model-option">
          <input type="checkbox" value="${escapeAttr(record.id)}" ${checked} />
          <span>
            <span class="model-name" title="${escapeAttr(record.name)}">${escapeHtml(record.name)}</span>
            <span class="model-meta">
              <span class="tag ${methodClass}">${escapeHtml(record.method)}</span>
              <span class="tag">${escapeHtml(template)}</span>
              <span class="tag">${escapeHtml(lengths)}</span>
            </span>
          </span>
        </label>
      `;
    })
    .join("");
}

function renderBarChart() {
  const length = selectedLength();
  const rows = getSelectedRecords()
    .map((record) => ({
      record,
      value: scoreValue(record, length),
      n: scoreN(record, length),
      elapsed: record.summaries[String(length)]?.elapsed_s,
    }))
    .filter((row) => row.value !== null)
    .sort((a, b) => b.value - a.value);

  els.lengthCompareHint.textContent =
    length === null
      ? "没有可用长度。"
      : `固定长度 ${formatNumber(length)}，指标 ${metricLabel(state.metric)}。`;

  if (!rows.length) {
    els.bestAtLength.textContent = "-";
    els.barChart.innerHTML = `<div class="empty-state">所选模型在该长度下没有 ${metricLabel(state.metric)} 数据</div>`;
    els.rankingBody.innerHTML = "";
    return;
  }

  els.bestAtLength.textContent = `最佳：${rows[0].record.name} ${formatPercent(rows[0].value)}`;
  els.barChart.innerHTML = rows
    .map((row) => {
      const width = Math.max(2, Math.min(100, row.value * 100));
      return `
        <div class="bar-row">
          <div class="bar-label" title="${escapeAttr(row.record.name)}">${escapeHtml(row.record.name)}</div>
          <div class="bar-track"><div class="bar-fill" style="width: ${width}%"></div></div>
          <div class="bar-value">${formatPercent(row.value)}</div>
        </div>
      `;
    })
    .join("");

  els.rankingBody.innerHTML = rows
    .map(
      (row, index) => `
        <tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(row.record.name)}</td>
          <td class="number-cell">${formatPercent(row.value)}</td>
          <td class="number-cell">${row.n === null ? "-" : formatNumber(row.n)}</td>
          <td class="number-cell">${formatDuration(row.elapsed)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderLineChart() {
  const records = getSelectedRecords();
  const lengths = getLengths(records);
  if (!records.length || !lengths.length) {
    els.lineChart.innerHTML = `<div class="empty-state">请选择至少一个模型</div>`;
    return;
  }

  const series = records
    .map((record, index) => ({
      record,
      color: COLORS[index % COLORS.length],
      points: lengths
        .map((length) => ({ length, value: scoreValue(record, length), n: scoreN(record, length) }))
        .filter((point) => point.value !== null),
    }))
    .filter((item) => item.points.length);

  if (!series.length) {
    els.lineChart.innerHTML = `<div class="empty-state">所选模型没有 ${metricLabel(state.metric)} 趋势数据</div>`;
    return;
  }

  const width = Math.max(700, lengths.length * 132 + 130);
  const height = 340;
  const pad = { top: 24, right: 28, bottom: 58, left: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xFor = (length) => {
    const index = lengths.indexOf(length);
    return pad.left + (lengths.length === 1 ? plotW / 2 : (index / (lengths.length - 1)) * plotW);
  };
  const yFor = (value) => pad.top + (1 - value) * plotH;

  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((tick) => {
      const y = yFor(tick);
      return `
        <line class="grid-line" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}"></line>
        <text class="chart-label" x="${pad.left - 10}" y="${y + 4}" text-anchor="end">${Math.round(tick * 100)}</text>
      `;
    })
    .join("");

  const xLabels = lengths
    .map((length) => {
      const x = xFor(length);
      return `
        <line class="grid-line" x1="${x}" y1="${pad.top}" x2="${x}" y2="${height - pad.bottom}"></line>
        <text class="chart-label" x="${x}" y="${height - 24}" text-anchor="middle">${formatNumber(length)}</text>
      `;
    })
    .join("");

  const paths = series
    .map((item) => {
      const points = item.points.map((point) => `${xFor(point.length)},${yFor(point.value)}`).join(" ");
      const circles = item.points
        .map(
          (point) => `
            <circle class="chart-point-dot" cx="${xFor(point.length)}" cy="${yFor(point.value)}" r="4.5" fill="${item.color}"></circle>
            <circle
              class="chart-point-hit"
              cx="${xFor(point.length)}"
              cy="${yFor(point.value)}"
              r="13"
              data-model="${escapeAttr(item.record.name)}"
              data-length="${formatNumber(point.length)}"
              data-score="${point.value.toFixed(6)}"
              data-percent="${formatPercent(point.value)}"
              data-n="${point.n === null ? "-" : formatNumber(point.n)}"
              data-metric="${escapeAttr(metricLabel(state.metric))}"
            ></circle>
          `,
        )
        .join("");
      return `
        <polyline fill="none" stroke="${item.color}" stroke-width="2.4" points="${points}"></polyline>
        ${circles}
      `;
    })
    .join("");

  const legend = series
    .map((item) => {
      const lastPoint = item.points[item.points.length - 1];
      return `
        <div class="trend-legend-item">
          <span class="trend-legend-dot" style="background: ${item.color}"></span>
          <span class="trend-legend-text" title="${escapeAttr(item.record.name)}">${escapeHtml(item.record.name)}</span>
          <span class="trend-legend-score">${formatPercent(lastPoint.value)}</span>
        </div>
      `;
    })
    .join("");

  els.lineChart.innerHTML = `
    <div class="trend-layout">
      <div class="trend-chart-scroll">
        <svg class="chart-svg" style="min-width: ${width}px; height: ${height}px" viewBox="0 0 ${width} ${height}" role="img" aria-label="模型长度趋势">
          ${grid}
          ${xLabels}
          <text class="chart-label axis-title" x="${pad.left}" y="${height - 14}">context length</text>
          <text class="chart-label axis-title" x="16" y="${pad.top}" transform="rotate(-90 16 ${pad.top})">score</text>
          <line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>
          <line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"></line>
          ${paths}
        </svg>
      </div>
      <aside class="trend-legend" aria-label="趋势图图例">
        <div class="trend-legend-title">图例 / 最新长度分数</div>
        ${legend}
      </aside>
      <div class="chart-tooltip" hidden></div>
    </div>
  `;
  bindTrendTooltip();
}

function bindTrendTooltip() {
  const layout = els.lineChart.querySelector(".trend-layout");
  const tooltip = els.lineChart.querySelector(".chart-tooltip");
  if (!layout || !tooltip) return;

  layout.addEventListener("pointermove", (event) => {
    const point = event.target.closest?.(".chart-point-hit");
    if (!point) {
      tooltip.hidden = true;
      return;
    }

    const box = layout.getBoundingClientRect();
    const nextLeft = Math.min(event.clientX - box.left + 14, box.width - 250);
    const nextTop = Math.max(8, event.clientY - box.top - 72);

    tooltip.innerHTML = `
      <strong>${escapeHtml(point.dataset.model)}</strong>
      <span>长度：${escapeHtml(point.dataset.length)}</span>
      <span>指标：${escapeHtml(point.dataset.metric)}</span>
      <span>分数：${escapeHtml(point.dataset.score)} (${escapeHtml(point.dataset.percent)})</span>
      <span>样本数：${escapeHtml(point.dataset.n)}</span>
    `;
    tooltip.style.left = `${Math.max(8, nextLeft)}px`;
    tooltip.style.top = `${nextTop}px`;
    tooltip.hidden = false;
  });

  layout.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
  });
}

function heatColor(value) {
  if (!Number.isFinite(value)) return "#f2f5f8";
  const clamped = Math.max(0, Math.min(1, value));
  const hue = clamped >= 0.72 ? 170 : clamped >= 0.45 ? 42 : 8;
  const sat = clamped >= 0.72 ? 42 : 55;
  const light = 92 - clamped * 31;
  return `hsl(${hue} ${sat}% ${light}%)`;
}

function renderTaskHeatmap() {
  const records = getSelectedRecords();
  const length = selectedLength();
  const tasks = getTasks(records).filter((task) => records.some((record) => scoreValue(record, length, task) !== null));
  els.taskMatrixHint.textContent =
    length === null ? "没有可用长度。" : `固定长度 ${formatNumber(length)}，展示各任务分数。`;

  if (!records.length || !tasks.length) {
    els.taskHeatmap.innerHTML = `<div class="empty-state">当前长度没有逐任务数据</div>`;
    return;
  }

  const header = tasks.map((task) => `<th>${escapeHtml(task)}</th>`).join("");
  const body = records
    .map((record) => {
      const cells = tasks
        .map((task) => {
          const value = scoreValue(record, length, task);
          return `<td class="heat-cell" style="background: ${heatColor(value)}">${formatPercent(value)}</td>`;
        })
        .join("");
      return `<tr><td title="${escapeAttr(record.name)}">${escapeHtml(record.name)}</td>${cells}</tr>`;
    })
    .join("");

  els.taskHeatmap.innerHTML = `
    <table class="heatmap-table">
      <thead><tr><th>模型</th>${header}</tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderLengthHeatmap() {
  const records = getSelectedRecords();
  const lengths = getLengths(records);
  if (!records.length || !lengths.length) {
    els.lengthHeatmap.innerHTML = `<div class="empty-state">请选择至少一个模型</div>`;
    return;
  }

  const header = lengths.map((length) => `<th>${formatNumber(length)}</th>`).join("");
  const body = records
    .map((record) => {
      const cells = lengths
        .map((length) => {
          const value = scoreValue(record, length);
          return `<td class="heat-cell" style="background: ${heatColor(value)}">${formatPercent(value)}</td>`;
        })
        .join("");
      return `<tr><td title="${escapeAttr(record.name)}">${escapeHtml(record.name)}</td>${cells}</tr>`;
    })
    .join("");

  els.lengthHeatmap.innerHTML = `
    <table class="heatmap-table">
      <thead><tr><th>模型</th>${header}</tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderAll() {
  updateSummary();
  renderMetricSelect();
  renderLengthSelect();
  renderModelList();
  renderBarChart();
  renderLineChart();
  renderTaskHeatmap();
  renderLengthHeatmap();
}

async function loadData({ keepSelection = true } = {}) {
  const previousIds = keepSelection ? new Set(state.selectedModels) : new Set();
  els.refreshButton.disabled = true;
  els.refreshButton.textContent = "加载中";
  try {
    const response = await fetch(`${API_URL}?t=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.payload = await response.json();
    preserveOrSelectDefaults(previousIds);
    const skipped = state.payload.skipped ?? [];
    if (skipped.length) {
      setStatus(`有 ${skipped.length} 个结果目录未能读取，详见 API 返回的 skipped 字段。`, "warn");
    } else {
      setStatus("");
    }
    renderAll();
  } catch (error) {
    setStatus(`无法加载 /api/ruler-results：${error.message}。请通过 python server.py 启动本页面。`);
  } finally {
    els.refreshButton.disabled = false;
    els.refreshButton.textContent = "刷新数据";
  }
}

function setAutoRefresh(enabled) {
  if (state.autoRefreshTimer) {
    clearInterval(state.autoRefreshTimer);
    state.autoRefreshTimer = null;
  }
  if (enabled) {
    state.autoRefreshTimer = setInterval(() => loadData({ keepSelection: true }), AUTO_REFRESH_MS);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function truncate(value, maxLength) {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}...` : value;
}

els.refreshButton.addEventListener("click", () => loadData({ keepSelection: true }));

els.autoRefreshToggle.addEventListener("change", (event) => {
  setAutoRefresh(event.target.checked);
});

els.viewPreset.addEventListener("change", (event) => {
  state.view = event.target.value;
  preserveOrSelectDefaults(new Set());
  renderAll();
});

els.metricSelect.addEventListener("change", (event) => {
  state.metric = event.target.value;
  renderAll();
});

els.lengthSelect.addEventListener("change", (event) => {
  state.fixedLength = Number(event.target.value);
  renderAll();
});

els.modelSearch.addEventListener("input", (event) => {
  state.search = event.target.value;
  renderModelList();
});

els.modelList.addEventListener("change", (event) => {
  if (event.target instanceof HTMLInputElement && event.target.type === "checkbox") {
    if (event.target.checked) {
      state.selectedModels.add(event.target.value);
    } else {
      state.selectedModels.delete(event.target.value);
    }
    renderBarChart();
    renderLineChart();
    renderTaskHeatmap();
    renderLengthHeatmap();
  }
});

els.selectVisibleButton.addEventListener("click", () => {
  state.selectedModels = new Set(getVisibleRecords().map((record) => record.id));
  renderAll();
});

els.clearModelsButton.addEventListener("click", () => {
  state.selectedModels = new Set();
  renderAll();
});

loadData({ keepSelection: false });
