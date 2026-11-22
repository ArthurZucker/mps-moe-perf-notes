"""GPU timeline profile of the decode loop (Metal System Trace), so every experiment is judged on what the GPU did,
not only on tok/s. The child loads + compiles exactly like hc_bench (same config files and HC_* levers), warms up, then
waits; the driver attaches `xctrace record` for the measured window, the child decodes `steps` tokens inside
torch.mps.profiler signposts, then the trace's GPU tables are exported and summarized.

    PYTHONPATH=src:benchmark_v2/inkling2_metal:benchmark_v2/inkling2_metal/hillclimb \
        python benchmark_v2/inkling2_metal/hillclimb/hc_profile.py configs/<name>.json [steps]
    python .../hc_profile.py --diff profiles/<a>.json profiles/<b>.json

Writes profiles/<name>.trace (open in Instruments) and profiles/<name>.json (kernel table, busy / gap stats, encoders).
"""
import collections, json, os, re, statistics, subprocess, sys, time, xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "profiles")


def child(cfg_path, steps, flag):
    import torch
    import hc_bench
    from hc_bench import on_ac
    if not on_ac():
        sys.exit("refusing to run on battery")
    model, tokenizer, ids, gen = hc_bench.load(json.load(open(cfg_path)))
    with torch.inference_mode():
        for _ in range(2):
            model.generate(ids, max_new_tokens=16, min_new_tokens=16, **gen)
        torch.mps.synchronize()
        print(f"READY {os.getpid()}", flush=True)
        while not os.path.exists(flag):
            time.sleep(0.05)
        time.sleep(1.0)  # let xctrace start recording
        t = time.perf_counter()  # no torch.mps.profiler signposts: they add host time to every op
        model.generate(ids, max_new_tokens=steps, min_new_tokens=steps, **gen)
        torch.mps.synchronize()
        print(f"DONE {steps / (time.perf_counter() - t):.1f} tok/s (under trace)", flush=True)


def export_tables(trace):
    toc = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--toc"], capture_output=True, text=True).stdout
    schemas = sorted(set(re.findall(r'schema="([^"]+)"', toc)))
    tables = {}
    for schema in schemas:
        if not re.search(r"gpu|metal|encoder|command-buffer|signpost", schema):
            continue
        xml = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath",
                              f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'], capture_output=True, text=True).stdout
        if xml.strip():
            tables[schema] = xml
    return schemas, tables


def rows(xml):
    """(start_ns, duration_ns, label, extra) per row; xctrace dedups repeated values with id / ref attributes."""
    root, ids, out = ET.fromstring(xml), {}, []
    for node in root.iter("row"):
        vals = {}
        for c in node:
            if "ref" in c.attrib:
                c = ids.get(c.attrib["ref"], c)
            elif "id" in c.attrib:
                ids[c.attrib["id"]] = c
            vals.setdefault(c.tag, c.attrib.get("fmt") or (c.text or ""))
            if c.tag in ("start-time", "duration", "event-time"):
                vals[c.tag + "_ns"] = int(c.text or 0) if (c.text or "").isdigit() else None
        out.append(vals)
    return out


def summarize(name, trace, wall_tok_s):
    schemas, tables = export_tables(trace)
    summary = {"name": name, "trace": trace, "schemas": schemas, "tok_s_under_trace": wall_tok_s, "tables": {}}
    for schema, xml in tables.items():
        rs = [r for r in rows(xml) if r.get("start-time_ns") is not None and r.get("duration_ns")]
        if not rs:
            continue
        rs.sort(key=lambda r: r["start-time_ns"])
        span = rs[-1]["start-time_ns"] + rs[-1]["duration_ns"] - rs[0]["start-time_ns"]
        busy, end = 0, rs[0]["start-time_ns"]
        gaps = []
        for r in rs:  # union of intervals = GPU busy time on this channel
            s0, e0 = r["start-time_ns"], r["start-time_ns"] + r["duration_ns"]
            if s0 > end:
                gaps.append(s0 - end)
            busy += max(0, e0 - max(s0, end))
            end = max(end, e0)
        label_keys = [k for k in rs[0] if k not in ("start-time", "duration", "start-time_ns", "duration_ns")]
        by = collections.defaultdict(list)
        for r in rs:
            key = " | ".join(str(r.get(k, ""))[:60] for k in label_keys[:3])
            by[key].append(r["duration_ns"])
        top = sorted(by.items(), key=lambda kv: -sum(kv[1]))[:40]
        summary["tables"][schema] = {
            "rows": len(rs), "span_ms": span / 1e6, "busy_ms": busy / 1e6, "busy_pct": 100 * busy / max(span, 1),
            "gaps": len(gaps), "gap_median_us": statistics.median(gaps) / 1e3 if gaps else 0,
            "gap_total_ms": sum(gaps) / 1e6, "label_keys": label_keys,
            "top": [{"label": k, "count": len(v), "total_ms": sum(v) / 1e6, "mean_us": sum(v) / len(v) / 1e3} for k, v in top],
        }
    return summary


def report(s):
    print(f"== {s['name']}  ({s['tok_s_under_trace']}); schemas: {', '.join(s['schemas'])}")
    for schema, t in s["tables"].items():
        print(f"  [{schema}] {t['rows']} intervals over {t['span_ms']:.1f} ms | busy {t['busy_ms']:.1f} ms ({t['busy_pct']:.0f}%) | "
              f"{t['gaps']} gaps, median {t['gap_median_us']:.1f} us, total {t['gap_total_ms']:.1f} ms | keys {t['label_keys'][:4]}")
        for row in t["top"][:12]:
            print(f"      {row['total_ms']:8.2f} ms  {row['count']:6d} x {row['mean_us']:7.1f} us  {row['label']}")


def diff(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    for schema in sorted(set(a["tables"]) & set(b["tables"])):
        ta, tb = a["tables"][schema], b["tables"][schema]
        print(f"[{schema}] busy {ta['busy_ms']:.1f} -> {tb['busy_ms']:.1f} ms | gaps {ta['gap_total_ms']:.1f} -> {tb['gap_total_ms']:.1f} ms | "
              f"intervals {ta['rows']} -> {tb['rows']}")
        la, lb = {r["label"]: r for r in ta["top"]}, {r["label"]: r for r in tb["top"]}
        for label in sorted(set(la) | set(lb), key=lambda k: -max(la.get(k, {}).get("total_ms", 0), lb.get(k, {}).get("total_ms", 0)))[:20]:
            ra, rb = la.get(label, {"total_ms": 0, "count": 0}), lb.get(label, {"total_ms": 0, "count": 0})
            print(f"    {ra['total_ms']:8.2f} -> {rb['total_ms']:8.2f} ms  ({ra['count']} -> {rb['count']})  {label}")


def driver(cfg_path, steps):
    name = json.load(open(cfg_path))["name"]
    os.makedirs(OUT, exist_ok=True)
    trace, flag = os.path.join(OUT, f"{name}.trace"), os.path.join(OUT, f".{name}.go")
    for p in (flag,):
        if os.path.exists(p): os.remove(p)
    subprocess.run(["rm", "-rf", trace])
    proc = subprocess.Popen([sys.executable, __file__, "--child", cfg_path, str(steps), flag], stdout=subprocess.PIPE, text=True)
    pid = None
    for line in proc.stdout:
        print("  child:", line.rstrip(), flush=True)
        if line.startswith("READY"):
            pid = int(line.split()[1]); break
    if pid is None:
        sys.exit("child failed before READY")
    rec = subprocess.Popen(["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--attach", str(pid),
                            "--output", trace, "--time-limit", f"{max(8, steps // 8)}s"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    open(flag, "w").close()
    tok_s = ""
    for line in proc.stdout:
        print("  child:", line.rstrip(), flush=True)
        if line.startswith("DONE"):
            tok_s = line.strip()[5:]
    proc.wait(); rec.wait()
    print(f"trace: {trace} | {tok_s}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "hc_trace.py"), trace, str(steps)])


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        driver(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64)
