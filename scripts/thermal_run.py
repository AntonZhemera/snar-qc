#!/usr/bin/env python3
"""Run (or attach to) a command under a CPU/GPU thermal cap.

Pauses (SIGSTOP) the target process group when the CPU (AMD k10temp) or the NVIDIA GPU
temperature reaches a high-water mark, and resumes (SIGCONT) once *both* fall back below the
low-water mark (hysteresis avoids flapping). Effectively duty-cycles a sustained load so the
package temperature cannot run away on a machine with no working thermal daemon.

Two modes:
  * launch  -- ``thermal_run.py [opts] -- <command...>``  : start the command and guard it.
  * attach  -- ``thermal_run.py [opts] --attach-pid PID`` : guard an already-running process
               group (e.g. to change the caps live without losing its progress).

POSIX-only at this time: it relies on POSIX process-group signals (SIGSTOP/SIGCONT) and reads
``/sys/class/hwmon``, so it is Linux-only and has NOT been tested on Windows. Run it only on a
POSIX host; elsewhere run the pipeline without it (or port the throttle first).
"""
from __future__ import annotations
import argparse, glob, os, shutil, signal, subprocess, sys, time


def find_k10temp():
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(os.path.join(d, "name")) as fh:
                if fh.read().strip() == "k10temp":
                    cand = os.path.join(d, "temp1_input")
                    return cand if os.path.exists(cand) else None
        except OSError:
            continue
    return None


def read_cpu(path):
    if not path:
        return None
    try:
        with open(path) as fh:
            return int(fh.read().strip()) / 1000.0
    except OSError:
        return None


def read_gpu(exe):
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vals = [int(x) for x in out.stdout.split() if x.strip().lstrip("-").isdigit()]
        return float(max(vals)) if vals else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-high", type=float, default=87.0)
    ap.add_argument("--cpu-low", type=float, default=74.0)
    ap.add_argument("--gpu-high", type=float, default=80.0)
    ap.add_argument("--gpu-low", type=float, default=70.0)
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--heartbeat", type=float, default=30.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--attach-pid", type=int, default=None,
                    help="guard an existing process group instead of launching a command")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if bool(cmd) == bool(args.attach_pid):
        ap.error("give exactly one of: -- <command>  OR  --attach-pid PID")

    cpu_path = find_k10temp()
    gpu_exe = shutil.which("nvidia-smi")
    logf = open(args.log, "a") if args.log else sys.stderr

    def log(msg):
        logf.write(f"[thermal {time.strftime('%H:%M:%S')}] {msg}\n"); logf.flush()

    log(f"cpu_sensor={cpu_path or 'NONE'} gpu={'nvidia-smi' if gpu_exe else 'NONE'}")
    log(f"caps: CPU resume<={args.cpu_low} pause>={args.cpu_high}  "
        f"GPU resume<={args.gpu_low} pause>={args.gpu_high}")
    if cpu_path is None and gpu_exe is None:
        log("WARNING: no temperature source -- running WITHOUT thermal protection")

    if args.attach_pid:
        own_child = False
        proc = None
        try:
            pgid = os.getpgid(args.attach_pid)
        except ProcessLookupError:
            log(f"attach: pid {args.attach_pid} not found"); return 1
        log(f"attach: guarding pgid {pgid} (pid {args.attach_pid})")

        def alive():
            try:
                os.kill(args.attach_pid, 0)
                return True
            except ProcessLookupError:
                return False
    else:
        own_child = True
        log(f"launch: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, start_new_session=True)
        pgid = os.getpgid(proc.pid)

        def alive():
            return proc.poll() is None

    paused = False
    peak_cpu = peak_gpu = 0.0
    last_beat = 0.0
    try:
        while alive():
            cpu, gpu = read_cpu(cpu_path), read_gpu(gpu_exe)
            if cpu is not None:
                peak_cpu = max(peak_cpu, cpu)
            if gpu is not None:
                peak_gpu = max(peak_gpu, gpu)
            hot = (cpu is not None and cpu >= args.cpu_high) or \
                  (gpu is not None and gpu >= args.gpu_high)
            cool = (cpu is None or cpu <= args.cpu_low) and \
                   (gpu is None or gpu <= args.gpu_low)
            if not paused and hot:
                os.killpg(pgid, signal.SIGSTOP); paused = True
                log(f"PAUSE  CPU={cpu} GPU={gpu}")
            elif paused and cool:
                os.killpg(pgid, signal.SIGCONT); paused = False
                log(f"RESUME CPU={cpu} GPU={gpu}")
            now = time.monotonic()
            if now - last_beat >= args.heartbeat:
                log(f"{'PAUSED' if paused else 'run'} CPU={cpu} GPU={gpu} "
                    f"(peak CPU={peak_cpu:.0f} GPU={peak_gpu:.0f})")
                last_beat = now
            time.sleep(args.poll)
        rc = proc.returncode if own_child else 0
    except KeyboardInterrupt:
        if paused:
            os.killpg(pgid, signal.SIGCONT)
        if own_child:
            os.killpg(pgid, signal.SIGTERM)
            log("interrupted -> SIGTERM to child group")
            return 130
        log("interrupted -> detaching (job left running)")
        return 0
    finally:
        if paused:
            try:
                os.killpg(pgid, signal.SIGCONT)  # never leave the target stopped
            except ProcessLookupError:
                pass
    log(f"done rc={rc}  peak CPU={peak_cpu:.0f} GPU={peak_gpu:.0f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
