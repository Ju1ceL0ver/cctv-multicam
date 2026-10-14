# zone_lab — side and door events by the owner's zones (13–14.10.2026)

Experiments behind the new live door (`../door_live_seg.py`). Data are not in git; on the machine they live in
`multicam/zone_lab/` (copies) and `multicam/data/door_v2/` (`zone_feats.json.gz`, `zone_pix*.json.gz`,
`zone_baseline.json`, `dsc_cache/` — doorseg_c predictions on all 43 /doorside stretches).

| file | what |
|---|---|
| `rules.py` | deterministic zone rules + track consistency -> events (owner's 3 rules, legs occlusion, door window, inheritance) |
| `lab2.py` | doorseg_c cached predictions -> tracker variants -> zone rows; head->feet kNN |
| `lab3.py` | buffered side (majority window) — rejected: worse on events |
| `lab4.py` | **side by a small boosting model on zone geometry** (the live engine) |
| `hyp.py`, `stack.py`, `trees.py`, `knn.py` | feature hypotheses: memory, neighbours, distances, pixel kNN, logic combinations (L1) |
| `exp_queue.py` + `queue.json` | experiment queue on the machine -> `results.jsonl` (train F1 chooses, test F1 reports) |
| `rescore.py`, `score_*.py`, `errs_test.py` | scoring against the owner's corrected truth (/zonecheck answers, gray zone excluded) |
