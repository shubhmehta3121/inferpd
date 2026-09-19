/**
 * Pitch-page charts: requested vs served QPS, then time-to-first-token.
 */
(function () {
  var qpsCanvas = document.getElementById("pitch-chart-qps");
  var ttftCanvas = document.getElementById("pitch-chart-ttft");
  if (!qpsCanvas || !ttftCanvas || typeof Chart === "undefined") return;

  function themeColor(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function fmtSec(v) {
    if (v == null) return "—";
    return v.toFixed(v >= 10 ? 1 : 2) + " seconds";
  }

  function logTick(v) {
    var decade = Math.pow(10, Math.floor(Math.log10(v)));
    var mantissa = v / decade;
    if (Math.abs(mantissa - 1) > 0.01 && Math.abs(mantissa - 3) > 0.01) return "";
    if (v < 1) return v.toFixed(1) + " sec";
    return String(Math.round(v)) + " sec";
  }

  function baseOptions() {
    var axisColor = themeColor("--text-muted", "#94a3b8");
    var gridColor = themeColor("--grid-line", "rgba(255,255,255,0.06)");
    return {
      axisColor: axisColor,
      gridColor: gridColor,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: axisColor,
            boxWidth: 14,
            padding: 14,
            font: { size: 11 },
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Requested load (queries per second)",
            color: axisColor,
          },
          ticks: { color: axisColor },
          grid: { color: gridColor },
        },
      },
    };
  }

  fetch("data/results.json")
    .then(function (res) {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    })
    .then(function (data) {
      var stacks = data.stacks || [];
      var stack = stacks.find(function (s) {
        return s.highlight && s.points && s.points.length;
      });
      if (!stack) {
        stack = stacks.find(function (s) {
          return s.points && s.points.length;
        });
      }
      if (!stack || !stack.points.length) throw new Error("No chart data");

      var prodP50 = data.targets.latency.ttft_p50_s;
      var prodP95 = data.targets.latency.ttft_p95_s;
      var labels = stack.points.map(function (p) {
        return String(p.qps_target);
      });
      var requested = stack.points.map(function (p) {
        return p.qps_target;
      });
      var served = stack.points.map(function (p) {
        return p.qps_achieved;
      });
      var p50 = stack.points.map(function (p) {
        return p.ttft_p50;
      });
      var p95 = stack.points.map(function (p) {
        return p.ttft_p95;
      });

      var base = baseOptions();
      var axisColor = base.axisColor;
      var gridColor = base.gridColor;

      new Chart(qpsCanvas, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Requested",
              data: requested,
              backgroundColor: "rgba(212, 168, 83, 0.55)",
            },
            {
              label: "Actually served",
              data: served,
              backgroundColor: "rgba(96, 165, 250, 0.65)",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: base.plugins.legend,
            tooltip: {
              callbacks: {
                title: function (items) {
                  return "Requested " + items[0].label + " queries/sec";
                },
                label: function (ctx) {
                  var n = ctx.parsed.y == null ? "—" : Number(ctx.parsed.y).toFixed(2);
                  return ctx.dataset.label + ": " + n + " queries/sec";
                },
              },
            },
          },
          scales: {
            x: base.scales.x,
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: "Queries per second",
                color: axisColor,
              },
              ticks: { color: axisColor },
              grid: { color: gridColor },
            },
          },
        },
      });

      new Chart(ttftCanvas, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Median wait (P50)",
              data: p50,
              borderColor: "rgba(74, 222, 128, 0.95)",
              backgroundColor: "rgba(74, 222, 128, 0.12)",
              fill: false,
              tension: 0.2,
            },
            {
              label: "Slow tail (P95)",
              data: p95,
              borderColor: "rgba(248, 113, 113, 0.95)",
              backgroundColor: "rgba(248, 113, 113, 0.08)",
              fill: false,
              tension: 0.2,
            },
            {
              label: "Production median (~2s)",
              data: labels.map(function () {
                return prodP50;
              }),
              borderColor: "rgba(212, 168, 83, 0.9)",
              borderDash: [6, 4],
              pointRadius: 0,
              fill: false,
            },
            {
              label: "Production tail (~3.5s)",
              data: labels.map(function () {
                return prodP95;
              }),
              borderColor: "rgba(148, 163, 184, 0.75)",
              borderDash: [6, 4],
              pointRadius: 0,
              fill: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: base.plugins.legend,
            tooltip: {
              callbacks: {
                title: function (items) {
                  return "At " + items[0].label + " requested queries/sec";
                },
                label: function (ctx) {
                  return ctx.dataset.label + ": " + fmtSec(ctx.parsed.y);
                },
              },
            },
          },
          scales: {
            x: base.scales.x,
            y: {
              type: "logarithmic",
              min: 0.1,
              title: {
                display: true,
                text: "Time to first token (seconds)",
                color: axisColor,
              },
              ticks: { color: axisColor, callback: logTick, maxTicksLimit: 12 },
              grid: { color: gridColor },
            },
          },
        },
      });
    })
    .catch(function () {
      var wrap = document.getElementById("pitch-chart-wrap");
      if (wrap) {
        wrap.innerHTML =
          '<p class="text-muted">Charts unavailable offline. See full benchmarks in the repo.</p>';
      }
    });
})();
