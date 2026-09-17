# Graph Report - cctv-multicam  (2026-09-18)

## Corpus Check
- 66 files · ~215,420 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 1 file(s) not represented in the graph (top: (none) 1)

## Summary
- 419 nodes · 838 edges · 23 communities (20 shown, 3 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 56 edges (avg confidence: 0.87)
- Token cost: 110,000 input · 10,742 output

## Community Hubs (Navigation)
- Пороги и оценка по эталону
- Калибровка камеры по плитке
- Найденное и отклонённое
- Планы и инфраструктура проекта
- Ночной сбор визитов
- Дисторсия и совмещение полов
- Разметка кадров учителем
- Разметчик личностей
- Трекинг в пикселях
- Доступ к машине и фоновые задачи
- Сбор треклетов для ReID
- Разметка пар разрывов
- Публикация разметчиков наружу
- Очередь ночных задач
- Передача GPU между задачами
- Сравнение ReID-моделей
- Супервизор ночей
- Обучение ученика-сегментатора
- Экспорт датасета сегментации
- Валидационные кадры из сырых записей
- Геометрия человека на полу
- Запуск откреплённых задач
- Очередь детекции клипов

## God Nodes (most connected - your core abstractions)
1. `Camera` - 23 edges
2. `load()` - 20 edges
3. `build()` - 18 edges
4. `Stream` - 14 edges
5. `mask()` - 14 edges
6. `Порядок запуска конвейера` - 14 edges
7. `main()` - 12 edges
8. `Калибровка камер по самому залу` - 12 edges
9. `calibrate()` - 11 edges
10. `main()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `yolo26n-seg (ученик-сегментатор)` --semantically_similar_to--> `yolo26x-seg @1536`  [INFERRED] [semantically similar]
  AGENTS.md → README.md
- `Галерея ТЦ за стеклом` --conceptually_related_to--> `cam1 (камера у входа)`  [INFERRED]
  AGENTS.md → README.md
- `Отклонено: маски внутри дескриптора внешности` --references--> `OSNet-AIN x1.0 (osnet_ain_x1_0_msmt17.pt)`  [INFERRED]
  AGENTS.md → README.md
- `Дистилляция: большая модель размечает, маленькие учатся` --conceptually_related_to--> `OSNet x0.25`  [AMBIGUOUS]
  AGENTS.md → README.md
- `AUC различимости людей по внешности` --rationale_for--> `Правило: пороги брать из данных`  [INFERRED]
  README.md → AGENTS.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Стадии конвейера: от сырой записи до визита** — readme_syraya_zapis, readme_segmentaciya, readme_vneshnost, readme_sinhronizaciya, readme_geometriya, readme_treking, readme_skleyka, readme_vizit [EXTRACTED 1.00]
- **Самокалибровка по самим записям (без работ на объекте)** — multicam_plumb, multicam_calib_joint_k, multicam_validate_people, multicam_reg_appearance, multicam_fit_floor_map, agents_shvy_plitki, agents_distorsiya_k1, docs_pipeline_calib_json, agents_bez_rabot_na_obekte [EXTRACTED 1.00]
- **Цикл ручного эталона и измерения качества** — multicam_export_pieces, multicam_label_pieces, multicam_gt_from_manual, readme_ground_truth, multicam_calibrate_thresholds, multicam_calibrate_tracker, multicam_score_config, agents_klip_c103700, agents_merit_a_ne_verit [EXTRACTED 1.00]

## Communities (23 total, 3 thin omitted)

### Community 0 - "Пороги и оценка по эталону"
Cohesion: 0.05
Nodes (73): Правило: пороги брать из данных, data/cam_sync.json, Порядок запуска конвейера, main(), pieces_of(), Appearance thresholds derived from ground truth, not guessed. For every pair of…, Tracker-level appearance gates from ground truth. Inside one camera the tracker…, frames_needed() (+65 more)

### Community 1 - "Калибровка камеры по плитке"
Cohesion: 0.11
Nodes (36): backproject(), calibrate(), draw(), floor_res(), init(), Camera calibration from the shop's own floor: rectangular tiles + verticals.…, Soft prior: CCTV cameras are mounted close to level (cam2 solved to -0.1 deg on…, roll_res() (+28 more)

### Community 2 - "Найденное и отклонённое"
Cohesion: 0.08
Nodes (39): Дробление треков, Отклонено: уточнение личностей итерациями (EM), Галерея ТЦ за стеклом, Клип c103700 (10:37, настроечный), Клип c171400 (17:14, самый людный час), Клип c183400 (18:34, вечерний свет), Отклонено: маски внутри дескриптора внешности, Правило: мерить, а не верить (+31 more)

### Community 3 - "Планы и инфраструктура проекта"
Cohesion: 0.08
Nodes (32): 3D-визуализация (следующий этап), Дистилляция: большая модель размечает, маленькие учатся, Расписание видеокарты (свободна 21:00–10:00), Метрическая калибровка (рост зависит от места), Репозиторий экспериментов multicam, Подозрение на наклон cam1, Отклонено: решение позы cam1 по людям (PnP, совместная оптимизация), multicam/data/raw_clips/<клип>/ (+24 more)

### Community 4 - "Ночной сбор визитов"
Cohesion: 0.10
Nodes (22): CamStream, CamTracker, log(), main(), past(), process_window(), Nightly collector: turn one trading day of raw footage from both cameras into…, People count per keyframe of one segment: [[pts_s, count, [[cx, cy], ...]],… (+14 more)

### Community 5 - "Дисторсия и совмещение полов"
Cohesion: 0.11
Nodes (21): Ограничение заказчика: никаких работ на объекте, Дисторсия k1 ≈ −0.24, Правило: проверять калибровку видом сверху и ростом людей, Швы плитки и отвесные линии как опора калибровки, Точность совмещения полов (медиана 0.19 м, p90 0.44 м, хвосты до 1.4 м), calib.json, Калибровка с нуля (если камеру сдвинули), plumb_chains() (+13 more)

### Community 6 - "Разметка кадров учителем"
Cohesion: 0.15
Nodes (14): box_centre(), fill_track_gaps(), log(), main(), outside_roi(), past_deadline(), polygon_for(), Distillation harvest: label footage with the big model at full strength. This… (+6 more)

### Community 7 - "Разметчик личностей"
Cohesion: 0.21
Nodes (15): before_request, clips(), crop(), gate(), index(), label(), labels_path(), merge() (+7 more)

### Community 8 - "Трекинг в пикселях"
Cohesion: 0.19
Nodes (11): app_matrix(), iou_matrix(), load_gates(), Per-camera tracking in image space (ByteTrack-style, with clothing colour).…, Cut a track where the clothing colour changes abruptly and stays changed. Two…, Appearance distance between tracks and detections. With ReID embeddings (unit…, split_on_appearance_change(), Carry detection-level ground truth from one detector's boxes to another's (same… (+3 more)

### Community 9 - "Доступ к машине и фоновые задачи"
Cohesion: 0.25
Nodes (10): Доступ с Мака через тоннель cloudflared, Папка jobs/ (фоновые задачи), Правило: никогда не убивать python.exe по имени, Правило: долгие задачи запускать откреплённо, jobs/bg.py, api(), execute(), get_kernel() (+2 more)

### Community 10 - "Сбор треклетов для ReID"
Cohesion: 0.31
Nodes (7): crop_cadence(), log(), main(), past_deadline(), Collect dense tracklets: the raw material for replacing OSNet. What a ReID…, How often to keep a crop, as a function of how long the track has lived. Dense…, watch()

### Community 11 - "Разметка пар разрывов"
Cohesion: 0.29
Nodes (8): api_label(), api_pairs(), img(), index(), labels(), get, post, One-keystroke labeller for identity-break candidates. Each candidate is "track…

### Community 12 - "Публикация разметчиков наружу"
Cohesion: 0.31
Nodes (7): build_apps(), log(), main(), protect(), Serve both labelling tools behind a shared key, then publish them via…, serve(), tunnel()

### Community 13 - "Очередь ночных задач"
Cohesion: 0.46
Nodes (7): hours_until(), log(), main(), Tonight's GPU queue: nano student -> small student -> raw 25 fps tracklets.…, running(), start(), wait_for_exit()

### Community 14 - "Передача GPU между задачами"
Cohesion: 0.47
Nodes (5): log(), main(), pids_running(), Hand the GPU from the teacher harvest to the tracklet harvest at 08:00. The…, PIDs of python.exe whose command line contains `needle`.

### Community 15 - "Сравнение ReID-моделей"
Cohesion: 0.53
Nodes (5): auc(), cos_dist(), lab_dist(), main(), How well does an appearance descriptor tell people apart, measured on ground…

### Community 16 - "Супервизор ночей"
Cohesion: 0.60
Nodes (4): log(), main(), next_run(), Run the nightly collector once per night, for as many nights as the raw…

### Community 17 - "Обучение ученика-сегментатора"
Cohesion: 0.60
Nodes (4): log(), main(), Train the -seg student on teacher pseudo-labels, inside a fixed GPU window.…, watchdog()

### Community 18 - "Экспорт датасета сегментации"
Cohesion: 0.67
Nodes (3): main(), norm_poly(), Teacher suggestions -> Ultralytics YOLO-seg dataset, split by day. Pseudo-…

### Community 19 - "Валидационные кадры из сырых записей"
Cohesion: 0.67
Nodes (3): log(), main(), Harvest the hand-review validation set from the raw 25 fps recordings. Held out…

## Ambiguous Edges - Review These
- `OSNet x0.25` → `Дистилляция: большая модель размечает, маленькие учатся`  [AMBIGUOUS]
  AGENTS.md · relation: conceptually_related_to
- `Расписание видеокарты (свободна 21:00–10:00)` → `Дистилляция: большая модель размечает, маленькие учатся`  [AMBIGUOUS]
  AGENTS.md · relation: conceptually_related_to

## Knowledge Gaps
- **1 isolated node(s):** `ByteTrack-подобный трекер с внешностью`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 118 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `OSNet x0.25` and `Дистилляция: большая модель размечает, маленькие учатся`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Расписание видеокарты (свободна 21:00–10:00)` and `Дистилляция: большая модель размечает, маленькие учатся`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Порядок запуска конвейера` connect `Пороги и оценка по эталону` to `Найденное и отклонённое`, `Планы и инфраструктура проекта`, `Разметчик личностей`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `mask()` connect `Калибровка камеры по плитке` to `Пороги и оценка по эталону`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `Camera` connect `Пороги и оценка по эталону` to `Калибровка камеры по плитке`, `Геометрия человека на полу`, `Дисторсия и совмещение полов`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **What connects `ByteTrack-подобный трекер с внешностью` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Пороги и оценка по эталону` be split into smaller, more focused modules?**
  _Cohesion score 0.050560512468542665 - nodes in this community are weakly interconnected._