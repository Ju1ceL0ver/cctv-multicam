import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))
import sam31_jobs as J  # noqa: E402


def test_windows_finish_one_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(J, 'JOBS', tmp_path)
    (tmp_path / 'jobs').mkdir()

    def job(tag, cam, start, status, made, claimed_at=0):
        jid = '%s_%s_%05d' % (tag, cam, start)
        J.write(tmp_path / 'jobs' / (jid + '.json'), {'id': jid, 'tag': tag, 'cam': cam, 'start': start, 'stop': start + 160,
                                                     'status': status, 'made': made, 'claimed_at': claimed_at})
        return jid
    job('B_1', 'cam1', 0, 'open', made=10)                      # a newer window, alphabetically first
    job('A_2', 'cam1', 0, 'done', made=5)
    late = job('A_2', 'cam2', 0, 'claimed', made=5, claimed_at=900)
    fresh = job('A_2', 'cam2', 152, 'claimed', made=5, claimed_at=990)
    now = 1000.0
    assert J.pick(now)[1]['tag'] == 'B_1'                        # nothing late yet: the next window
    now = 901.0 + J.SPARE_S
    assert J.pick(now)[1]['id'] == late                          # the silent worker's job again, before B_1
    assert fresh != late
