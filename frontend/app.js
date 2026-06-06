/**
 * Scaling Monosemanticity — Interactive Dashboard
 * Main application logic
 */

(function () {
  "use strict";

  // ==========================================
  // Utility Functions
  // ==========================================

  function $(sel, ctx = document) {
    return ctx.querySelector(sel);
  }
  function $$(sel, ctx = document) {
    return [...ctx.querySelectorAll(sel)];
  }

  function formatNumber(n, decimals = 2) {
    if (n === null || n === undefined) return "—";
    if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(1) + "T";
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return Number(n).toFixed(decimals);
  }

  function formatFlops(f) {
    if (f >= 1e15) return (f / 1e15).toFixed(2) + " PF";
    if (f >= 1e12) return (f / 1e12).toFixed(2) + " TF";
    if (f >= 1e9) return (f / 1e9).toFixed(2) + " GF";
    return f.toFixed(0) + " F";
  }

  // ==========================================
  // 1. Navigation
  // ==========================================

  function initNavigation() {
    const nav = $(".nav");
    const links = $$(".nav__link");
    const hamburger = $(".nav__hamburger");
    const linkContainer = $(".nav__links");
    const sections = $$("section[id]");

    // Scroll shadow
    window.addEventListener("scroll", () => {
      nav.classList.toggle("scrolled", window.scrollY > 40);
    });

    // Active link tracking
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.id;
            links.forEach((l) => l.classList.remove("active"));
            const active = $(`.nav__link[data-section="${id}"]`);
            if (active) active.classList.add("active");
          }
        });
      },
      { rootMargin: "-30% 0px -60% 0px" }
    );
    sections.forEach((s) => observer.observe(s));

    // Smooth scroll on click
    links.forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const target = link.dataset.section;
        const el = document.getElementById(target);
        if (el) {
          el.scrollIntoView({ behavior: "smooth" });
          linkContainer.classList.remove("open");
        }
      });
    });

    // Hamburger toggle
    if (hamburger) {
      hamburger.addEventListener("click", () => {
        linkContainer.classList.toggle("open");
      });
    }
  }

  // ==========================================
  // 2. Scroll Reveal Animations
  // ==========================================

  function initScrollReveal() {
    const reveals = $$(".reveal");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -50px 0px" }
    );
    reveals.forEach((el) => observer.observe(el));
  }

  // ==========================================
  // 3. Hero Particles
  // ==========================================

  function initParticles() {
    const container = $(".hero__particles");
    if (!container) return;
    const colors = ["#00d4ff", "#7c3aed", "#f472b6", "#34d399"];
    for (let i = 0; i < 30; i++) {
      const p = document.createElement("div");
      p.className = "hero__particle";
      p.style.left = Math.random() * 100 + "%";
      p.style.animationDelay = Math.random() * 8 + "s";
      p.style.animationDuration = 6 + Math.random() * 6 + "s";
      p.style.background = colors[Math.floor(Math.random() * colors.length)];
      p.style.width = p.style.height = 2 + Math.random() * 3 + "px";
      container.appendChild(p);
    }
  }

  // ==========================================
  // 4. Feature Explorer
  // ==========================================

  function initFeatureExplorer() {
    const featureInput = $("#feature-index-input");
    const loadBtn = $("#load-feature-btn");
    const container = $("#feature-detail-container");

    if (!featureInput || !loadBtn || !container) return;

    function renderFeature(data) {
      if (!data) {
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-state__icon">🔍</div>
            <div class="empty-state__text">Feature not found. Try indices: 42, 137, or 256</div>
          </div>`;
        return;
      }

      const statsHTML = `
        <div class="card card--flat">
          <h4 style="margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
            <span class="tag tag--primary">Feature #${data.feature_idx}</span>
            Statistics
          </h4>
          <div class="feature-stats">
            <div class="stat-card">
              <div class="stat-card__value stat-card__value--primary">${data.max_activation.toFixed(2)}</div>
              <div class="stat-card__label">Max Activation</div>
            </div>
            <div class="stat-card">
              <div class="stat-card__value stat-card__value--secondary">${data.mean_activation.toFixed(4)}</div>
              <div class="stat-card__label">Mean Activation</div>
            </div>
            <div class="stat-card">
              <div class="stat-card__value stat-card__value--accent">${(data.fraction_active * 100).toFixed(2)}%</div>
              <div class="stat-card__label">Fraction Active</div>
            </div>
            <div class="stat-card">
              <div class="stat-card__value stat-card__value--success">${data.mean_activation_when_active.toFixed(3)}</div>
              <div class="stat-card__label">Mean When Active</div>
            </div>
          </div>
        </div>`;

      const examplesHTML = `
        <div class="card card--flat">
          <h4 style="margin-bottom: 1rem;">Top Activating Examples</h4>
          <div class="token-list">
            ${data.top_examples
              .map(
                (ex) => `
              <div class="token-example">
                ${ex.context_text}
                <span class="token-example__activation">⚡ ${ex.activation.toFixed(2)}</span>
              </div>`
              )
              .join("")}
          </div>
        </div>`;

      container.innerHTML = `<div class="feature-detail">${statsHTML}${examplesHTML}</div>`;
    }

    function loadFeature() {
      const idx = parseInt(featureInput.value, 10);
      if (isNaN(idx)) return;

      container.innerHTML = `<div class="loading-overlay"><div class="spinner"></div>Loading feature ${idx}…</div>`;

      // Try API first, then fall back to demo data
      fetch(`/api/features/${idx}`)
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((data) => renderFeature(data))
        .catch(() => {
          const demo = window.DEMO_DATA?.features?.[idx];
          renderFeature(demo || null);
        });
    }

    loadBtn.addEventListener("click", loadFeature);
    featureInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") loadFeature();
    });

    // Load default feature
    featureInput.value = 42;
    loadFeature();
  }

  // ==========================================
  // 5. Scaling Laws Charts
  // ==========================================

  function initScalingCharts() {
    if (typeof Chart === "undefined") return;

    const data = window.DEMO_DATA?.scalingLaws;
    if (!data) return;

    const sweep = data.sweep;
    const chartDefaults = {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1500, easing: "easeOutQuart" },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: "#94a3b8",
            font: { family: "'Inter', sans-serif", size: 11 },
            boxWidth: 8,
            boxHeight: 8,
            usePointStyle: true,
          },
        },
        tooltip: {
          backgroundColor: "rgba(10, 14, 39, 0.95)",
          titleColor: "#e2e8f0",
          bodyColor: "#94a3b8",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
          titleFont: { family: "'Inter', sans-serif", weight: "bold" },
          bodyFont: { family: "'Inter', sans-serif" },
        },
      },
      scales: {
        x: {
          type: "logarithmic",
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: {
            color: "#64748b",
            font: { family: "'Inter', sans-serif", size: 10 },
            callback: function (v) { return formatFlops(v); },
          },
          title: {
            display: true,
            color: "#64748b",
            font: { family: "'Inter', sans-serif", size: 11 },
          },
        },
        y: {
          type: "logarithmic",
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: {
            color: "#64748b",
            font: { family: "'Inter', sans-serif", size: 10 },
          },
          title: {
            display: true,
            color: "#64748b",
            font: { family: "'Inter', sans-serif", size: 11 },
          },
        },
      },
    };

    // Group by feature count for colored series
    const featureCounts = [...new Set(sweep.map((r) => r.n_features))];
    const seriesColors = ["#00d4ff", "#7c3aed", "#f472b6"];

    // --- Loss vs Compute ---
    const lossCtx = document.getElementById("chart-loss")?.getContext("2d");
    if (lossCtx) {
      new Chart(lossCtx, {
        type: "scatter",
        data: {
          datasets: featureCounts.map((fc, i) => ({
            label: `${formatNumber(fc, 0)} features`,
            data: sweep
              .filter((r) => r.n_features === fc)
              .map((r) => ({ x: r.flops, y: r.loss })),
            backgroundColor: seriesColors[i],
            borderColor: seriesColors[i],
            pointRadius: 6,
            pointHoverRadius: 9,
            showLine: true,
            borderWidth: 2,
            tension: 0.3,
          })),
        },
        options: {
          ...chartDefaults,
          scales: {
            ...chartDefaults.scales,
            x: { ...chartDefaults.scales.x, title: { ...chartDefaults.scales.x.title, text: "Compute (FLOPs)" } },
            y: { ...chartDefaults.scales.y, title: { ...chartDefaults.scales.y.title, text: "Loss" } },
          },
        },
      });
    }

    // --- Variance Explained vs Compute ---
    const varCtx = document.getElementById("chart-variance")?.getContext("2d");
    if (varCtx) {
      new Chart(varCtx, {
        type: "scatter",
        data: {
          datasets: featureCounts.map((fc, i) => ({
            label: `${formatNumber(fc, 0)} features`,
            data: sweep
              .filter((r) => r.n_features === fc)
              .map((r) => ({ x: r.flops, y: r.variance_explained })),
            backgroundColor: seriesColors[i],
            borderColor: seriesColors[i],
            pointRadius: 6,
            pointHoverRadius: 9,
            showLine: true,
            borderWidth: 2,
            tension: 0.3,
          })),
        },
        options: {
          ...chartDefaults,
          scales: {
            ...chartDefaults.scales,
            x: { ...chartDefaults.scales.x, title: { ...chartDefaults.scales.x.title, text: "Compute (FLOPs)" } },
            y: {
              ...chartDefaults.scales.y,
              type: "linear",
              title: { ...chartDefaults.scales.y.title, text: "Variance Explained" },
              min: 0.5,
              max: 1.0,
            },
          },
        },
      });
    }

    // --- L0 vs Compute ---
    const l0Ctx = document.getElementById("chart-l0")?.getContext("2d");
    if (l0Ctx) {
      new Chart(l0Ctx, {
        type: "scatter",
        data: {
          datasets: featureCounts.map((fc, i) => ({
            label: `${formatNumber(fc, 0)} features`,
            data: sweep
              .filter((r) => r.n_features === fc)
              .map((r) => ({ x: r.flops, y: r.l0 })),
            backgroundColor: seriesColors[i],
            borderColor: seriesColors[i],
            pointRadius: 6,
            pointHoverRadius: 9,
            showLine: true,
            borderWidth: 2,
            tension: 0.3,
          })),
        },
        options: {
          ...chartDefaults,
          scales: {
            ...chartDefaults.scales,
            x: { ...chartDefaults.scales.x, title: { ...chartDefaults.scales.x.title, text: "Compute (FLOPs)" } },
            y: {
              ...chartDefaults.scales.y,
              type: "linear",
              title: { ...chartDefaults.scales.y.title, text: "L0 (Active Features)" },
            },
          },
        },
      });
    }
  }

  // ==========================================
  // 6. Steering Playground
  // ==========================================

  function initSteering() {
    const slider = $("#steering-coeff-slider");
    const coeffDisplay = $("#steering-coeff-value");
    const featureSelect = $("#steering-feature-select");
    const promptInput = $("#steering-prompt-input");
    const runBtn = $("#steering-run-btn");
    const baselineOutput = $("#steering-baseline-output");
    const steeredOutput = $("#steering-steered-output");

    if (!slider || !runBtn) return;

    slider.addEventListener("input", () => {
      coeffDisplay.textContent = slider.value;
    });

    function runSteering() {
      const coeff = parseInt(slider.value, 10);
      const prompt = promptInput?.value || "";

      baselineOutput.textContent = "Generating…";
      steeredOutput.textContent = "Generating…";

      // Try API first, fall back to demo
      fetch("/api/steering", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          feature_idx: parseInt(featureSelect?.value || "42", 10),
          coefficient: coeff,
        }),
      })
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((data) => {
          baselineOutput.textContent = data.baseline || "—";
          steeredOutput.innerHTML = data.steered || "—";
        })
        .catch(() => {
          // Use demo data
          const demos = window.DEMO_DATA?.steering;
          const key = Object.keys(demos || {})[0];
          const demo = demos?.[key];
          if (demo) {
            baselineOutput.textContent = demo.baseline;
            // Find closest coefficient
            const coeffs = Object.keys(demo.results).map(Number);
            const closest = coeffs.reduce((prev, curr) =>
              Math.abs(curr - coeff) < Math.abs(prev - coeff) ? curr : prev
            );
            const steered = demo.results[String(closest)] || demo.baseline;
            // Highlight differences
            steeredOutput.textContent = steered;
          }
        });
    }

    runBtn.addEventListener("click", runSteering);

    // Populate with demo on load
    const demos = window.DEMO_DATA?.steering;
    if (demos) {
      const key = Object.keys(demos)[0];
      const demo = demos[key];
      if (promptInput) promptInput.value = demo.prompt;
      baselineOutput.textContent = demo.baseline;
      const steered = demo.results["30"] || Object.values(demo.results)[0];
      steeredOutput.textContent = steered;
    }
  }

  // ==========================================
  // 7. Training Monitor
  // ==========================================

  function initTrainingMonitor() {
    if (typeof Chart === "undefined") return;

    const data = window.DEMO_DATA?.trainingHistory;
    if (!data) return;

    const history = data.history;
    const summary = data.summary;

    // --- Training Curves Chart ---
    const ctx = document.getElementById("chart-training")?.getContext("2d");
    if (ctx) {
      new Chart(ctx, {
        type: "line",
        data: {
          labels: history.map((h) => h.step),
          datasets: [
            {
              label: "Total Loss",
              data: history.map((h) => h.loss),
              borderColor: "#00d4ff",
              backgroundColor: "rgba(0, 212, 255, 0.1)",
              fill: true,
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 4,
              tension: 0.4,
            },
            {
              label: "MSE",
              data: history.map((h) => h.mse),
              borderColor: "#7c3aed",
              backgroundColor: "rgba(124, 58, 237, 0.05)",
              fill: true,
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 4,
              tension: 0.4,
            },
            {
              label: "Var. Explained",
              data: history.map((h) => h.variance_explained),
              borderColor: "#34d399",
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 4,
              tension: 0.4,
              yAxisID: "y1",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 2000, easing: "easeOutQuart" },
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: {
              labels: {
                color: "#94a3b8",
                font: { family: "'Inter', sans-serif", size: 11 },
                boxWidth: 10,
                boxHeight: 3,
                usePointStyle: false,
              },
            },
            tooltip: {
              backgroundColor: "rgba(10, 14, 39, 0.95)",
              titleColor: "#e2e8f0",
              bodyColor: "#94a3b8",
              borderColor: "rgba(255,255,255,0.1)",
              borderWidth: 1,
              cornerRadius: 8,
              padding: 12,
            },
          },
          scales: {
            x: {
              grid: { color: "rgba(255,255,255,0.04)" },
              ticks: { color: "#64748b", font: { size: 10 }, maxTicksLimit: 10 },
              title: { display: true, text: "Training Step", color: "#64748b", font: { size: 11 } },
            },
            y: {
              grid: { color: "rgba(255,255,255,0.04)" },
              ticks: { color: "#64748b", font: { size: 10 } },
              title: { display: true, text: "Loss", color: "#64748b", font: { size: 11 } },
              position: "left",
            },
            y1: {
              grid: { drawOnChartArea: false },
              ticks: { color: "#34d399", font: { size: 10 } },
              title: { display: true, text: "Var. Explained", color: "#34d399", font: { size: 11 } },
              position: "right",
              min: 0,
              max: 1,
            },
          },
        },
      });
    }

    // --- Dead Features Gauge ---
    const gaugeEl = $(".gauge__fill--primary");
    if (gaugeEl && summary) {
      const circumference = 2 * Math.PI * 70; // radius=70
      const pct = summary.dead_feature_fraction;
      const offset = circumference - pct * circumference;
      gaugeEl.style.strokeDasharray = circumference;
      // Start fully hidden, animate to target
      gaugeEl.style.strokeDashoffset = circumference;
      setTimeout(() => {
        gaugeEl.style.strokeDashoffset = offset;
      }, 500);

      const valueEl = $(".gauge__value");
      if (valueEl) valueEl.textContent = (pct * 100).toFixed(1) + "%";
    }

    // --- Summary Metrics ---
    const metricEls = {
      "metric-final-loss": summary?.final_loss,
      "metric-final-var": summary?.final_variance_explained,
      "metric-final-l0": summary?.final_l0,
      "metric-dead-count": summary?.n_dead_features,
    };
    Object.entries(metricEls).forEach(([id, value]) => {
      const el = document.getElementById(id);
      if (el && value !== undefined) {
        el.textContent = typeof value === "number" && value < 10
          ? value.toFixed(3)
          : formatNumber(value, 1);
      }
    });
  }

  // ==========================================
  // 8. Architecture Node Hover Effect
  // ==========================================

  function initArchAnimation() {
    const nodes = $$(".arch-node");
    nodes.forEach((node, i) => {
      node.style.animationDelay = i * 120 + "ms";
    });
  }

  // ==========================================
  // 9. Initialize Everything
  // ==========================================

  function init() {
    initNavigation();
    initScrollReveal();
    initParticles();
    initArchAnimation();
    initFeatureExplorer();
    initScalingCharts();
    initSteering();
    initTrainingMonitor();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
