<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Обучение v2</title>
<link rel="stylesheet" href="/static/train.css?v=1">
</head><body>
<header>
  <div class="brand"><span class="live" id="live"></span><b>Обучение модели v2</b></div>
  <select id="run" title="запуск"></select>
  <span class="pill" id="phase">—</span>
  <span class="pill" id="conn">подключаюсь…</span>
  <span class="grow"></span>
  <label class="ctl">сглаживание <input type="range" id="smooth" min="0" max="97" value="85"><output id="smooth_v">0.85</output></label>
  <label class="ctl">окно
    <select id="range"><option value="0">весь путь</option><option value="1500">1500 шагов</option><option value="600">600 шагов</option><option value="200">200 шагов</option></select></label>
</header>
<main>
  <section class="hero card">
    <div class="hero-top">
      <div class="big"><span id="step_live">0</span><small> / <span id="steps">0</span> шагов</small></div>
      <div class="meta">
        <div><em>эпоха</em><b id="epoch">—</b></div>
        <div><em>с на шаг</em><b id="sps">—</b></div>
        <div><em>осталось</em><b id="eta">—</b></div>
        <div><em>конец</em><b id="etaclock">—</b></div>
        <div><em>идёт уже</em><b id="elapsed">—</b></div>
      </div>
    </div>
    <div class="bar"><i id="prog"></i></div>
    <div class="note" id="note"></div>
  </section>

  <section class="kpis" id="kpis"></section>

  <section class="card">
    <h2>Качество на экзамене <small>после каждой эпохи, 52 кадра 18.09 и трекинг на отложенном окне · нажмите на подпись, чтобы скрыть линию</small></h2>
    <div class="legend" id="leg_q"></div>
    <div class="chart tall" id="ch_quality"></div>
    <div class="refs">Для сравнения: v1 — найдено 79.3 % · точность 88 % · маска 0.81 · мелкие 71 %. YOLO-ученик — 88.4 % · 82.5 % · 0.87 · 77.9 %.</div>
    <div class="tablewrap"><table id="ep"><thead><tr><th>эпоха</th><th>шаг</th><th>время</th><th>найдено</th><th>точность</th><th>мелкие</th><th>маска IoU</th><th>без дублей: найдено</th><th>без дублей: точн.</th><th>IDF1</th><th>смены</th><th>найдено (трек)</th></tr></thead><tbody></tbody></table></div>
  </section>

  <section class="card">
    <h2>Железо в реальном времени <small>раз в секунду, последние 10 минут</small></h2>
    <div class="grid g4" id="hw"></div>
  </section>

  <section class="card">
    <h2>Скорость обучения</h2>
    <div class="grid g3" id="speed"></div>
  </section>

  <section class="card">
    <h2>Потери по шагам <small>бледная линия — как есть, яркая — сглаженная; чем ниже, тем лучше</small></h2>
    <div id="losses"></div>
  </section>
</main>
<div class="tip" id="tip"></div>
<script src="/static/train.js?v=1"></script>
</body></html>
