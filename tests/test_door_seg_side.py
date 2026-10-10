import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'multicam'))
import door_seg_side as S


def test_mask_margin_accounts_for_line_orientation():
    assert S.quantization_margin({'p1':[0,0],'p2':[10,0]})==4
    assert S.quantization_margin({'p1':[0,0],'p2':[0,10]})==4
    assert S.quantization_margin({'p1':[0,0],'p2':[10,10]})==pytest.approx(4*2**.5)
    with pytest.raises(ValueError,match='zero length'):
        S.quantization_margin({'p1':[0,0],'p2':[0,0]})


def test_student_margin_does_not_mutate_shared_settings(monkeypatch):
    original={'h_in':0,'h_out':8,'table':{'full':[100]}}
    def init(self):
        self.side=original
        self.line={'p1':[0,0],'p2':[10,0]}
    monkeypatch.setattr(S.BaseLive,'__init__',init)
    live=S.Live()
    assert live.side['h_in']==4
    assert original['h_in']==0
    assert live.side['h_out']==8
    assert live.side['table'] is not original['table']
