import datetime
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location('night_probe', Path(__file__).parents[1] / 'multicam/door_seg_night_probe.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class NightGuards(unittest.TestCase):
    def test_boundaries_and_foreign_jobs(self):
        at = lambda h, m: datetime.datetime(2026, 10, 10, h, m)
        self.assertFalse(P.night(at(20, 59)))
        self.assertTrue(P.night(at(21, 0)))
        self.assertTrue(P.night(at(9, 34)))
        self.assertFalse(P.night(at(9, 35)))
        jobs = [dict(pid=os.getpid()+10, name='python.exe', cmdline=['python.exe', 'door_live.py']),
                dict(pid=os.getpid()+11, name='Photoshop.exe', cmdline=[]),
                dict(pid=os.getpid()+12, name='python.exe', cmdline=['python.exe', 'door_seg_colab_restore.py'])]
        self.assertEqual(len(P.blockers(at(21, 1), jobs)), 2)
        self.assertEqual(P.blockers(at(12, 0), jobs), ['daytime'])
        self.assertEqual(P.blockers(at(21, 1), [dict(pid=os.getpid(), name='python.exe', cmdline=['door_seg_night_probe.py'])]), [])
        wrappers = [dict(pid=os.getpid()+21, name='python.exe', cmdline=['door_seg_night_probe.py']),
                    dict(pid=os.getpid()+22, name='python.exe', cmdline=['door_seg_night_probe.py'])]
        self.assertEqual(P.blockers(at(21, 1), wrappers, parent_pids=[os.getpid()+21]),
                         ['GPU job PID %s' % (os.getpid()+22)])

    def test_train_cohort_rejects_role_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'data/door_side').mkdir(parents=True)
            f = root / 'data/door_side/20260918.json'
            state = {tag: dict(done=True, role='train') for tag in P.TAGS}
            for tag in P.TAGS:
                src = root / 'data/sam31_door' / tag / 'cam1'; src.mkdir(parents=True)
                (src / 'video.mp4').touch()
            f.write_text(json.dumps(state))
            self.assertEqual(P.cohort(root), list(P.TAGS))
            state[P.TAGS[1]]['role'] = 'test'; f.write_text(json.dumps(state))
            with self.assertRaises(ValueError): P.cohort(root)


if __name__ == '__main__':
    unittest.main()
