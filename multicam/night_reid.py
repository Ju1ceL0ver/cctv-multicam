"""Tonight's ReID work, alone on the video card from 21:00: measure the stock model, teach one
copy on 17.09 only and measure it on 18.09 (a day it never saw -- the honest number), then
teach the copy meant for use on both days. Everything lands in data/logs/reid_night.json.

It waits for the card: the live counter owns it until 21:00. It gives up at 09:15 so the
counter gets it back at 10:00 whatever happens.

usage: night_reid.py        (started with bg.spawn; RA_NOW=1 skips the wait, for a test)"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
REPORT = ROOT / 'data' / 'logs' / 'reid_night.json'
WEIGHTS = ROOT / 'data' / 'weights'
BASE = WEIGHTS / 'osnet_ain_x1_0_msmt17.pt'
START, LAST_START, GIVE_UP = '21:02', '06:00', '09:15'   # begin, latest moment to begin, stop


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def card_free():
    """True when nothing is computing on the card. Memory alone misleads: on 23.09 the desktop
    and a Jupyter kernel that had once loaded a model sat on 3.8 GB with the card idle, and the
    job waited past 22:00. The live counter at work shows as load, so load decides."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout.split(',')
        used, load = float(out[0]), float(out[1])
        return load < 30 and used < 8000
    except Exception:
        return False


def late(now=None):
    now = (now or datetime.now()).strftime('%H:%M')
    return GIVE_UP <= now < START


def next_start(now=None):
    """The coming 21:02. Started at 08:40 the card looked free -- the counter only takes it at
    10:00 -- and the first version began training at once, into the shop's hours."""
    from datetime import timedelta
    now = now or datetime.now()
    h, m = map(int, START.split(':'))
    at = now.replace(hour=h, minute=m, second=0, microsecond=0)
    hm = now.strftime('%H:%M')
    if hm >= START or hm < LAST_START:
        return now              # the night is on already: 22:26 must not mean tomorrow evening
    return at if at > now else at + timedelta(days=1)


def save(report):
    report['updated'] = datetime.now().isoformat(timespec='seconds')
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    import train_reid
    import reid_eval_shop
    report = {'started': datetime.now().isoformat(timespec='seconds'), 'steps': []}
    if os.environ.get('RA_NOW') != '1':
        at = next_start()
        log('waiting for %s' % at.isoformat(timespec='minutes'))
        while datetime.now() < at or not card_free():
            if datetime.now() >= at and late():
                report['stopped'] = 'видеокарта так и не освободилась до утра'
                save(report)
                return
            time.sleep(60)
    log('card is free, starting')
    # two lengths on 17.09 alone show whether more training still helps on a day it never saw
    plan = [('stock', None, None, None, 'не обучалась'),
            ('shop_d17_3k', ['20260917'], WEIGHTS / 'osnet_ain_x1_0_shop_d17_3k.pt', 3000, '18.09 не видела: честная проверка'),
            ('shop_d17_12k', ['20260917'], WEIGHTS / 'osnet_ain_x1_0_shop_d17_12k.pt', 12000, '18.09 не видела: честная проверка'),
            ('shop_d1718', ['20260917', '20260918'], WEIGHTS / 'osnet_ain_x1_0_shop_d1718.pt', 12000,
             'видела 18.09: цифры на 18.09 завышены, это модель для работы')]
    for name, days, out, iters, note in plan:
        if late():
            report['stopped'] = 'утро: видеокарта нужна счётчику'
            break
        step = {'name': name, 'days': days, 'iters': iters, 'note': note}
        try:
            if days:
                t = time.time()
                train_reid.train(str(out), days, iters=iters, log=log)
                step['train_s'] = round(time.time() - t)
            step.update(reid_eval_shop.evaluate(out or BASE))
        except Exception as exc:
            import traceback
            step['error'] = traceback.format_exc()[-2000:]
        log('%s: %s' % (name, json.dumps({k: step.get(k) for k in ('door', 'held_out', 'error')}, ensure_ascii=False)))
        report['steps'].append(step)
        save(report)
    report['finished'] = datetime.now().isoformat(timespec='seconds')
    save(report)


if __name__ == '__main__':
    main()
