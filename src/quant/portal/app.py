import os
import sys
from flask import Flask,Response,redirect,render_template_string,url_for
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
if project_root not in sys.path:
    sys.path.insert(0,project_root)

from quant.factor_system.dashboard.app import app as factor_dashboard_app
from quant.strategy_system.dashboard.app import app as strategy_dashboard_app

portal_app = Flask(__name__)

_PORTAL_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quant Portal</title>
  <style>
    :root {
      --bg: #f3efe5;
      --panel: rgba(255,255,255,0.78);
      --ink: #172121;
      --muted: #556565;
      --line: rgba(23,33,33,0.12);
      --accent: #0b6e4f;
      --accent-2: #1d4ed8;
      --shadow: 0 18px 45px rgba(23,33,33,0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(11,110,79,0.20), transparent 34%),
        radial-gradient(circle at bottom right, rgba(29,78,216,0.18), transparent 32%),
        linear-gradient(135deg, #f8f4eb 0%, var(--bg) 48%, #ebe6db 100%);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .shell {
      width: min(1100px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 56px 0 64px;
    }
    .hero {
      padding: 8px 0 28px;
    }
    .eyebrow {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.62);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    h1 {
      margin: 18px 0 12px;
      font-size: clamp(36px, 6vw, 62px);
      line-height: 0.94;
      letter-spacing: -0.04em;
    }
    .subtitle {
      max-width: 760px;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.75;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 22px;
      margin-top: 30px;
    }
    .card {
      position: relative;
      overflow: hidden;
      padding: 26px 24px 22px;
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.52);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .card::after {
      content: "";
      position: absolute;
      inset: auto -30px -34px auto;
      width: 120px;
      height: 120px;
      border-radius: 999px;
      opacity: 0.12;
    }
    .card.factor::after { background: var(--accent); }
    .card.strategy::after { background: var(--accent-2); }
    .card h2 {
      margin: 0;
      font-size: 28px;
      letter-spacing: -0.03em;
    }
    .card p {
      margin: 12px 0 0;
      color: var(--muted);
      line-height: 1.7;
      min-height: 84px;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }
    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 138px;
      padding: 11px 16px;
      border-radius: 12px;
      text-decoration: none;
      font-weight: 600;
      transition: transform 0.16s ease, box-shadow 0.16s ease, opacity 0.16s ease;
    }
    .button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 25px rgba(23,33,33,0.10);
    }
    .button.primary.factor {
      background: var(--accent);
      color: #fff;
    }
    .button.primary.strategy {
      background: var(--accent-2);
      color: #fff;
    }
    .button.secondary {
      background: rgba(255,255,255,0.64);
      border: 1px solid var(--line);
      color: var(--ink);
    }
    .footer {
      margin-top: 28px;
      color: var(--muted);
      font-size: 14px;
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <span class="eyebrow">Quant Portal</span>
      <h1>One door for factor and strategy dashboards.</h1>
      <div class="subtitle">
        This portal keeps the display layer independent from update flows. Open the factor dashboard, weekly factor report,
        and strategy dashboard from one stable entry instead of binding page access to runtime app startup.
      </div>
    </section>

    <section class="grid">
      <article class="card factor">
        <h2>Factor System</h2>
        <p>
          Browse factor lists, detail pages, section-period views, and the weekly report from the existing factor dashboard.
        </p>
        <div class="actions">
          <a class="button primary factor" href="/factors/">Open Factors</a>
          <a class="button secondary" href="/factors/weekly-report">Weekly Report</a>
        </div>
      </article>

      <article class="card strategy">
        <h2>Strategy System</h2>
        <p>
          Review strategy net value, compare strategies, and benchmark against major indices from the strategy dashboard.
        </p>
        <div class="actions">
          <a class="button primary strategy" href="/strategies/">Open Strategies</a>
        </div>
      </article>
    </section>

    <div class="footer">
      Health check: <a href="{{ url_for('healthz') }}">/healthz</a>
    </div>
  </main>
</body>
</html>
"""


@portal_app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@portal_app.route("/")
def index():
    return render_template_string(_PORTAL_TEMPLATE)


@portal_app.route("/healthz")
def healthz():
    return {
        "status": "ok",
        "apps": {
            "factors": "/factors/",
            "strategies": "/strategies/",
        },
    }


app = DispatcherMiddleware(
    portal_app,
    {
        "/factors": factor_dashboard_app,
        "/strategies": strategy_dashboard_app,
    },
)


if __name__ == "__main__":
    port = int(os.environ.get("PORTAL_PORT", "5003"))
    run_simple("0.0.0.0", port, app, use_reloader=False, use_debugger=False, threaded=True)
