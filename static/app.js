// Resize Plotly charts on window resize.
window.addEventListener("resize", function () {
  document.querySelectorAll(".js-plotly-plot").forEach(function (el) {
    Plotly.Plots.resize(el);
  });
});

// Render a Plotly figure with a left-to-right reveal animation.
window.drawPlotlyChart = function (divId, fig) {
  var el = document.getElementById(divId);
  if (!el) return;
  var cfg = { displayModeBar: false, responsive: true };

  var renderPromise = el.data
    ? Plotly.react(divId, fig.data, fig.layout, cfg)
    : Plotly.newPlot(divId, fig.data, fig.layout, cfg);

  renderPromise.then(function () {
    var existing = document.getElementById(divId + "-reveal");
    if (existing) existing.remove();

    var overlay = document.createElement("div");
    overlay.id = divId + "-reveal";
    overlay.style.cssText = "position:absolute;top:0;right:0;bottom:0;left:0;"
      + "background:#0f172a;pointer-events:none;z-index:5;"
      + "transition:clip-path 700ms cubic-bezier(0.65,0,0.35,1);"
      + "clip-path:inset(0 0 0 0);";
    el.style.position = "relative";
    el.appendChild(overlay);
    overlay.getBoundingClientRect();
    requestAnimationFrame(function () {
      overlay.style.clipPath = "inset(0 0 0 100%)";
    });
    setTimeout(function () { overlay.remove(); }, 750);
  });
};

// Sync visible control value into the hidden #controls-form field.
function syncVisibleToHidden(name, value) {
  var hidden = document.querySelector("#controls-form [name='" + name + "']");
  if (hidden) hidden.value = value;
}

// Called by dashboard_data.html after every render to keep the hidden form in sync.
window.syncAppliedState = function (model, granularity) {
  syncVisibleToHidden("model", model);
  syncVisibleToHidden("granularity", granularity);
};

// Fire the hidden apply button (triggers the hx-get for dashboard-data).
function applyControls() {
  var btn = document.getElementById("apply-btn");
  if (btn) btn.click();
}

// Auto-submit when model or granularity changes.
document.body.addEventListener("change", function (evt) {
  var el = evt.target;
  if (!el.closest || !el.closest("#dashboard-data")) return;

  var name = el.getAttribute("name");
  if (!name) return;

  syncVisibleToHidden(name, el.value);

  if (name === "model" || name === "granularity") {
    applyControls();
  }
});

// ---- Loading states ----
// Control change (model/granularity): fixed overlay sized to the visible
// portion of #dashboard-data — vertically clipped to the viewport, not the
// full scrolled document height.
// Refresh data: lightweight shimmer on chart cards + disabled button state.

function _ensureLoadingStyles() {
  if (document.getElementById("fc-loading-style")) return;
  var s = document.createElement("style");
  s.id = "fc-loading-style";
  s.textContent = [
    "@keyframes fc-bounce{0%,80%,100%{transform:translateY(0);opacity:.35}40%{transform:translateY(-10px);opacity:1}}",
    "@keyframes fc-shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}",
    ".fc-dot{width:10px;height:10px;border-radius:50%;background:#6366f1;animation:fc-bounce 1.2s ease-in-out infinite;}",
    ".fc-dot:nth-child(2){animation-delay:.2s}",
    ".fc-dot:nth-child(3){animation-delay:.4s}",
    ".fc-chart-shimmer{position:relative;overflow:hidden;}",
    ".fc-chart-shimmer::after{content:'';position:absolute;inset:0;pointer-events:none;",
    "background:linear-gradient(90deg,transparent 0%,rgba(99,102,241,.07) 50%,transparent 100%);",
    "animation:fc-shimmer 1.4s ease-in-out infinite;}",
  ].join("");
  document.head.appendChild(s);
}

function showDashboardLoading() {
  if (document.getElementById("dashboard-loading-overlay")) return;
  _ensureLoadingStyles();
  var container = document.getElementById("dashboard-data");
  if (!container) return;

  var rect = container.getBoundingClientRect();
  var top    = Math.max(rect.top, 0);
  var height = Math.min(rect.bottom, window.innerHeight) - top;
  if (height <= 0) height = window.innerHeight;

  var overlay = document.createElement("div");
  overlay.id = "dashboard-loading-overlay";
  overlay.style.cssText = "position:fixed;z-index:50;pointer-events:none;"
    + "top:" + top + "px;left:" + rect.left + "px;"
    + "width:" + rect.width + "px;height:" + height + "px;"
    + "display:flex;align-items:center;justify-content:center;"
    + "background:rgba(2,6,23,0.78);backdrop-filter:blur(3px);border-radius:1rem;";
  overlay.innerHTML =
    '<div style="display:flex;flex-direction:column;align-items:center;gap:20px;">'
    + '<div style="display:flex;gap:10px;">'
    + '<div class="fc-dot"></div><div class="fc-dot"></div><div class="fc-dot"></div>'
    + '</div>'
    + '<span style="font-size:12px;font-weight:500;color:#64748b;letter-spacing:.05em;text-transform:uppercase;">Loading</span>'
    + '</div>';
  document.body.appendChild(overlay);
}

function hideDashboardLoading() {
  var el = document.getElementById("dashboard-loading-overlay");
  if (el) el.remove();
}

function showRefreshLoading() {
  _ensureLoadingStyles();
  ["hero-chart-wrap", "weekly-chart"].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.classList.add("fc-chart-shimmer");
  });
  var btn = document.getElementById("refresh-data-btn");
  if (btn) { btn.disabled = true; btn.style.opacity = "0.5"; }
}

function hideRefreshLoading() {
  ["hero-chart-wrap", "weekly-chart"].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.classList.remove("fc-chart-shimmer");
  });
  var btn = document.getElementById("refresh-data-btn");
  if (btn) { btn.disabled = false; btn.style.opacity = ""; }
}

document.body.addEventListener("htmx:beforeRequest", function (evt) {
  if (evt.detail.elt && evt.detail.elt.id === "refresh-data-btn") {
    showRefreshLoading();
  } else if (evt.detail.target && evt.detail.target.id === "dashboard-data") {
    showDashboardLoading();
  }
});

document.body.addEventListener("htmx:afterSwap", function (evt) {
  if (evt.detail.target && evt.detail.target.id === "dashboard-data") {
    hideDashboardLoading();
  }
});

document.body.addEventListener("htmx:afterSettle", function (evt) {
  if (evt.detail.elt && evt.detail.elt.id === "refresh-data-btn") {
    hideRefreshLoading();
  }
});

document.body.addEventListener("htmx:responseError", function (evt) {
  hideDashboardLoading();
  hideRefreshLoading();
});
