"""Task Scheduler entry point; captures startup failures before keeper imports."""
import os
import runpy
import sys
import traceback
from pathlib import Path
from datetime import datetime
root=Path(__file__).resolve().parent
logs=root/'data/logs';logs.mkdir(parents=True,exist_ok=True)
os.chdir(root);sys.path.insert(0,str(root))
# Scheduled SYSTEM tasks do not inherit an activated Conda environment.
prefix=Path(sys.executable).parent
os.environ['PATH']=os.pathsep.join(str(p) for p in
    (prefix,prefix/'Library/bin',prefix/'Scripts'))+os.pathsep+os.environ.get('PATH','')
with (logs/'keeper_startup.log').open('a',encoding='utf-8',buffering=1) as log:
    sys.stdout=log;sys.stderr=log
    print(datetime.now().isoformat(),'launcher',os.getpid())
    try:runpy.run_path(str(root/'keeper.py'),run_name='__main__')
    except Exception:
        traceback.print_exc();raise
