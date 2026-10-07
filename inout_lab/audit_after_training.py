"""One-shot dependent check for the already-running local training process.
No restart, refit or recurring automation. Starts inference only once report exists.
"""
import os,time,json
from pathlib import Path
from audit_dual_view import audit
root=Path(__file__).parent
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--day',default='20260919');a=parser.parse_args();report=root/('diagnostics/dualview_'+a.day+'_report.json');start=time.monotonic()
 while not report.exists():
  try:os.kill(a.pid,0)
  except ProcessLookupError:raise RuntimeError('Training process ended without its result; inspect its original session. No restart attempted.')
  if time.monotonic()-start>3600:raise RuntimeError('Dependent audit timeout; inspect original training process. No restart attempted.')
  time.sleep(2)
 result=json.loads(report.read_text());assert result['day']==a.day and result['epochs']==60
 audit(a.day)
