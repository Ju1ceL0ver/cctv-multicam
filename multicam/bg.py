"""Start a long job that outlives the Jupyter kernel we talk to the box through.

The tunnel drops a call after 100 s and a kernel restart takes its children with it,
so every pass longer than a minute goes through here: Start-Process hands the job to
the system, and we do not wait for powershell to answer. Output lands in
data/logs/NAME.log."""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, 'data', 'logs')


def _q(s):
    return "'" + str(s).replace("'", "''") + "'"


def spawn(name, args, env=None, cwd=ROOT, wait=6):
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, name + '.log')
    err = log + '.err'
    for f in (log, err):
        open(f, 'w').close()
    pre = ''.join('$env:%s=%s; ' % (k, _q(v)) for k, v in (env or {}).items())
    cmd = (pre + 'Start-Process -FilePath %s -ArgumentList %s -WorkingDirectory %s '
           '-RedirectStandardOutput %s -RedirectStandardError %s -WindowStyle Hidden'
           % (_q(sys.executable), ','.join(_q(a) for a in args), _q(cwd), _q(log), _q(err)))
    subprocess.Popen(['powershell', '-NoProfile', '-Command', cmd], cwd=cwd, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(wait)
    return log


def running(pattern):
    """Command lines of live python processes containing pattern, so nothing starts twice."""
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "Get-CimInstance Win32_Process -Filter \"name like '%python%'\" | "
                          "%{ '{0}|{1}' -f $_.ProcessId, ($_.CommandLine -replace '\\s+',' ') }"],
                         capture_output=True, text=True).stdout
    return [l.strip() for l in out.split('\n') if pattern in l]


if __name__ == '__main__':
    print('started ->', spawn(sys.argv[1], sys.argv[2:]))
