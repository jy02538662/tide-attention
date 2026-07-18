"""Generate lightweight promo assets for Tide Attention.

No video dependencies are required. The script writes:
  - assets/promo/tide_attention_promo.html  (animated 16:9 promo page)
  - assets/promo/tide_attention_poster.svg  (static poster)
  - assets/promo/storyboard.md             (copy for posts / video narration)

To make an actual MP4, open the HTML file in a browser and screen-record it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "promo"

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Tide Attention Promo</title>
<style>
  :root {
    --bg: #080b12;
    --panel: rgba(255,255,255,0.08);
    --fg: #f8fbff;
    --muted: #9fb0c7;
    --cyan: #54d6ff;
    --blue: #6b7cff;
    --pink: #ff5cc8;
    --green: #7dffb2;
    --yellow: #ffe27a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    display: grid;
    place-items: center;
    background: radial-gradient(circle at 25% 20%, #16264c, transparent 34%),
                radial-gradient(circle at 75% 65%, #3d1141, transparent 34%),
                var(--bg);
    color: var(--fg);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
  }
  .stage {
    width: min(1200px, 96vw);
    aspect-ratio: 16 / 9;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255,255,255,.08), rgba(255,255,255,.02));
    box-shadow: 0 40px 120px rgba(0,0,0,.45);
  }
  .grid {
    position: absolute;
    inset: 0;
    background-image: linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(circle, black, transparent 75%);
  }
  .title {
    position: absolute;
    left: 64px;
    top: 52px;
  }
  h1 {
    margin: 0;
    font-size: 68px;
    letter-spacing: -0.05em;
    line-height: .9;
  }
  .subtitle {
    margin-top: 16px;
    color: var(--muted);
    font-size: 24px;
  }
  .tag {
    display: inline-flex;
    gap: 10px;
    align-items: center;
    margin-top: 18px;
    padding: 10px 14px;
    border-radius: 999px;
    background: rgba(84,214,255,.12);
    border: 1px solid rgba(84,214,255,.28);
    color: var(--cyan);
    font-weight: 700;
  }
  .cards {
    position: absolute;
    left: 64px;
    right: 64px;
    bottom: 56px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
  }
  .card {
    min-height: 245px;
    border-radius: 24px;
    padding: 28px;
    background: var(--panel);
    border: 1px solid rgba(255,255,255,.12);
    backdrop-filter: blur(20px);
    position: relative;
  }
  .card h2 { margin: 0 0 14px; font-size: 32px; }
  .card p { color: var(--muted); font-size: 20px; line-height: 1.4; margin: 0; }
  .metric {
    position: absolute;
    left: 28px;
    bottom: 28px;
    font-size: 38px;
    font-weight: 800;
  }
  .good { color: var(--green); }
  .hard { color: var(--pink); }
  .arrow {
    position: absolute;
    left: calc(50% - 35px);
    bottom: 167px;
    width: 70px;
    height: 70px;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--cyan), var(--pink));
    display: grid;
    place-items: center;
    font-size: 36px;
    font-weight: 900;
    animation: pulse 1.8s infinite;
  }
  .flow {
    position: absolute;
    right: 64px;
    top: 64px;
    width: 430px;
    height: 210px;
    border-radius: 22px;
    background: rgba(0,0,0,.22);
    border: 1px solid rgba(255,255,255,.10);
    padding: 22px;
  }
  .line { display: flex; align-items: center; gap: 12px; margin: 13px 0; color: var(--muted); font-size: 19px; }
  .pill { padding: 7px 11px; border-radius: 999px; background: rgba(255,255,255,.09); color: var(--fg); font-weight: 700; }
  .cyan { color: var(--cyan); }
  .pink { color: var(--pink); }
  .footer {
    position: absolute;
    right: 64px;
    bottom: 24px;
    color: rgba(255,255,255,.55);
    font-size: 14px;
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); box-shadow: 0 0 0 rgba(84,214,255,0); }
    50% { transform: scale(1.08); box-shadow: 0 0 36px rgba(84,214,255,.45); }
  }
  .card.clear { animation: rise 5.8s ease-in-out infinite; }
  .card.conflict { animation: rise 5.8s ease-in-out infinite .7s; }
  @keyframes rise { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
</style>
</head>
<body>
  <main class="stage">
    <div class="grid"></div>
    <section class="title">
      <h1>Tide<br/>Attention</h1>
      <div class="subtitle">Defect-gated long-context memory control</div>
      <div class="tag">CPU demo · no GPU · non-commercial release</div>
    </section>
    <section class="flow">
      <div class="line"><span class="pill">Input</span> + <span class="pill">Memory</span></div>
      <div class="line">detect <span class="cyan">coherence</span> / <span class="pink">contradiction</span></div>
      <div class="line"><span class="cyan">yang_sparse</span> when clear</div>
      <div class="line"><span class="pink">yin_full + deep retrieve</span> on conflict</div>
    </section>
    <section class="cards">
      <div class="card clear">
        <h2>Clear memory</h2>
        <p>One coherent answer. Stay sparse, retrieve shallow, keep cost low.</p>
        <div class="metric good">0.191× full FLOPs</div>
      </div>
      <div class="card conflict">
        <h2>Conflicting memory</h2>
        <p>Multiple contradictory answers. Trigger full/deep path and recover the valid fact.</p>
        <div class="metric hard">deep retrieve</div>
      </div>
    </section>
    <div class="arrow">→</div>
    <div class="footer">Controlled CPU demo, not a GPT/Claude/Kimi comparison.</div>
  </main>
</body>
</html>
'''

SVG = r'''<svg width="1200" height="675" viewBox="0 0 1200 675" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop stop-color="#081020"/>
      <stop offset="1" stop-color="#220927"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" x2="1">
      <stop stop-color="#54d6ff"/>
      <stop offset="1" stop-color="#ff5cc8"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="675" rx="32" fill="url(#bg)"/>
  <circle cx="270" cy="140" r="230" fill="#1b4fff" opacity="0.22"/>
  <circle cx="910" cy="430" r="260" fill="#ff33b8" opacity="0.18"/>
  <text x="70" y="115" fill="#f8fbff" font-family="Arial, sans-serif" font-size="72" font-weight="800">Tide Attention</text>
  <text x="74" y="165" fill="#9fb0c7" font-family="Arial, sans-serif" font-size="27">Clear memory stays cheap. Conflicting memory makes the model think harder.</text>
  <rect x="70" y="218" width="500" height="260" rx="28" fill="#ffffff" opacity="0.08" stroke="#ffffff" stroke-opacity="0.14"/>
  <text x="105" y="278" fill="#7dffb2" font-family="Arial, sans-serif" font-size="38" font-weight="800">Clear memory</text>
  <text x="105" y="330" fill="#d7e2f2" font-family="Arial, sans-serif" font-size="25">yang_sparse · shallow retrieve</text>
  <text x="105" y="415" fill="#7dffb2" font-family="Arial, sans-serif" font-size="54" font-weight="900">0.191× full FLOPs</text>
  <rect x="630" y="218" width="500" height="260" rx="28" fill="#ffffff" opacity="0.08" stroke="#ffffff" stroke-opacity="0.14"/>
  <text x="665" y="278" fill="#ff5cc8" font-family="Arial, sans-serif" font-size="38" font-weight="800">Conflict memory</text>
  <text x="665" y="330" fill="#d7e2f2" font-family="Arial, sans-serif" font-size="25">yin_full · deep retrieve</text>
  <text x="665" y="415" fill="#ff5cc8" font-family="Arial, sans-serif" font-size="54" font-weight="900">recover valid fact</text>
  <rect x="70" y="535" width="560" height="54" rx="27" fill="url(#accent)" opacity="0.25"/>
  <text x="96" y="570" fill="#f8fbff" font-family="Arial, sans-serif" font-size="24" font-weight="700">CPU-friendly v0.1-preview · non-commercial release</text>
  <text x="805" y="570" fill="#9fb0c7" font-family="Arial, sans-serif" font-size="20">Controlled demo, not commercial-model SOTA claim.</text>
</svg>
'''

STORYBOARD = """# Tide Attention promo storyboard

## 8-second post / short video

1. Hook: "Clear memory stays cheap. Conflicting memory makes the model think harder."
2. Clear case: `yang_sparse`, shallow retrieve, `0.191x full FLOPs`.
3. Conflict case: `yin_full`, deep retrieve, recovers `PHOENIX` despite ORION/DRAGON/WOLVES distractors.
4. Boundary: CPU-friendly controlled demo, not a GPT/Claude/Kimi comparison.

## Suggested post copy

I released Tide Attention v0.1-preview: a CPU-friendly, non-commercial controller for long-context memory conflicts.

- clear memory -> sparse / cheap
- conflicting memory -> full + deep retrieve
- no GPU required for the default demo

Repo tagline: Defect-gated sparse/full attention for long-context memory conflicts.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tide_attention_promo.html").write_text(HTML, encoding="utf-8")
    (OUT / "tide_attention_poster.svg").write_text(SVG, encoding="utf-8")
    (OUT / "storyboard.md").write_text(STORYBOARD, encoding="utf-8")
    print(f"Wrote {OUT / 'tide_attention_promo.html'}")
    print(f"Wrote {OUT / 'tide_attention_poster.svg'}")
    print(f"Wrote {OUT / 'storyboard.md'}")
    print("Open the HTML in a browser and screen-record it to make an MP4/GIF.")


if __name__ == "__main__":
    main()
