# Порядок запуска

Всё тяжёлое выполняется на машине с видеокартой; код разворачивается туда как есть.

```bash
# 1. сырая запись обеих камер (10:00-21:00, сегменты по 15 минут)
RA_RAW_CAMERA=cam1 python jobs/raw_recorder.py
RA_RAW_CAMERA=cam2 python jobs/raw_recorder.py

# 2. сегментация окна записи (клип = кусок дня, обе камеры синхронно)
python multicam/detect_raw.py c103700 20260917 10:37:00 570

# 3. дескрипторы внешности
python multicam/embed_clip.py c103700 osnet_ain_x1_0_msmt17.pt

# 4. сдвиг времени между камерами -> data/cam_sync.json
python multicam/sync_estimate.py c103700

# 5. люди, визиты, оценка (если есть эталон)
python multicam/run_clip.py c103700

# 6. разметка личностей вручную (браузер, :5070)
python multicam/export_pieces.py c103700
python multicam/label_pieces.py
python multicam/gt_from_manual.py c103700

# 7. пороги из эталона и сводная оценка
python multicam/calibrate_tracker.py c103700
python multicam/calibrate_thresholds.py c103700
python multicam/score_config.py c103700

# 8. видео для проверки глазами
python multicam/render_raw.py c103700 430 500 exit.mp4
```

## Калибровка с нуля (если камеру сдвинули)

```bash
python multicam/plumb.py                    # дисторсия по прямым краям
python multicam/calib_joint_k.py calib.json # фокус, пропорция плитки, позы камер
python multicam/validate_people.py calib.json
python multicam/reg_appearance.py           # совмещение полов двух камер по людям
```

Проверять калибровку следует видом сверху и ростом людей, а не наложением линий на кадр:
ошибка наклона даёт правдоподобные линии и неверные метры.
