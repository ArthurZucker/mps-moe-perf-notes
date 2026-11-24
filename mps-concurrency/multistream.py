"""Independent kernels on real torch.mps.Stream queues with explicit events.

torch 2.15's Python Event.record/wait raise for non-default streams, but the C++ event ops
already use the current stream. This harness bypasses the stale Python check, forks N
streams from a default-stream event, joins them back with events, and verifies exact output.

Run: python multistream.py [steps] [size]
"""
import statistics, sys, time
import torch

torch.zeros(1, device="mps")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 20


def record(ev):
    torch._C._mps_recordEvent(ev._Event__eventId)


def wait(ev):
    torch._C._mps_waitForEvent(ev._Event__eventId)


def event():
    return torch.mps.Event()


def run_serial(a, b):
    with torch.mps.stream(torch.mps.default_stream()):
        for _ in range(STEPS):
            a = torch.sin(a) + 0.5
            b = torch.cos(b) - 0.25
    return a, b


def run_streams(a, b, nstreams):
    streams = [torch.mps.Stream() for _ in range(nstreams)]
    default = torch.mps.default_stream()
    fork = event()
    joins = [event() for _ in streams]
    with torch.mps.stream(default):
        record(fork)
    for i, s in enumerate(streams):
        with torch.mps.stream(s):
            wait(fork)
            x = a if i % 2 == 0 else b
            for _ in range(STEPS):
                x = torch.sin(x) + 0.5 if i % 2 == 0 else torch.cos(x) - 0.25
            if i % 2 == 0:
                a = x
            else:
                b = x
            record(joins[i])
    with torch.mps.stream(default):
        for ev in joins:
            wait(ev)
    return a, b


def bench(fn, *args):
    fn(*args)
    torch.mps.synchronize()
    ts = []
    for _ in range(3):
        t = time.perf_counter(); fn(*args); torch.mps.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


a0 = torch.randn(SIZE, device="mps")
b0 = torch.randn(SIZE, device="mps")
a, b = run_serial(a0.clone(), b0.clone())
ref_a, ref_b = a.clone(), b.clone()
print(f"torch {torch.__version__}, size {SIZE}, {STEPS} dependent steps per branch")
print(f"serial 2 branches: {bench(run_serial, a0.clone(), b0.clone())*1e3:8.2f} ms")
for n in (2, 4, 8):
    # n branches are all one of the two chains; split each side into n/2 streams
    s = [torch.mps.Stream() for _ in range(n)]
    default = torch.mps.default_stream()
    fork = event(); joins = [event() for _ in s]
    def run():
        global a, b
        a, b = a0.clone(), b0.clone()
        with torch.mps.stream(default):
            record(fork)
        for i, st in enumerate(s):
            with torch.mps.stream(st):
                wait(fork)
                x = a0 if i % 2 == 0 else b0
                for _ in range(STEPS):
                    x = torch.sin(x) + 0.5 if i % 2 == 0 else torch.cos(x) - 0.25
                if i % 2 == 0: a = x
                else: b = x
                record(joins[i])
        with torch.mps.stream(default):
            for ev in joins: wait(ev)
    t = bench(run)
    ok = torch.equal(a, ref_a) and torch.equal(b, ref_b)
    print(f"{n:2d} streams: {t*1e3:8.2f} ms  speedup {bench(run_serial, a0.clone(), b0.clone())/t:4.2f}x  exact={ok}")
