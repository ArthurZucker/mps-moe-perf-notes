"""Analyze a Metal System Trace of one decode run: the python process's Compute-channel GPU intervals (kicks), GPU busy
vs idle in the measured window, gaps between kicks, command buffers / encoders. Columns are mapped by position from the
table schema (several columns share an XML tag), and xctrace's id / ref value dedup is resolved.

    python hc_trace.py profiles/<name>.trace [tokens]
"""
import collections, re, statistics, subprocess, sys, xml.etree.ElementTree as ET


def table(trace, schema):
    xml = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath",
                          f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'], capture_output=True, text=True).stdout
    root = ET.fromstring(xml)
    cols = [c.findtext("mnemonic") for c in root.iter("col")]
    ids, out = {}, []
    for row in root.iter("row"):
        rec = {}
        for el in row.iter():  # ids can sit on nested elements; refs may point at any of them
            if "id" in el.attrib:
                ids[el.attrib["id"]] = el
        for name, cell in zip(cols, list(row)):
            if "ref" in cell.attrib:
                cell = ids.get(cell.attrib["ref"], cell)
            rec[name] = (cell.text, cell.attrib.get("fmt", ""), cell)
        out.append(rec)
    return out


def main(trace, tokens):
    rows = table(trace, "metal-gpu-intervals")
    mine = []
    for r in rows:
        proc = r.get("process", (None, "", None))[1]
        chan = r.get("channel-name", (None, "", None))[1]
        if "python" in proc and "Compute" in chan and r["duration"][0]:
            start, dur = int(r["start"][0]), int(r["duration"][0])
            mine.append((start, dur, r.get("cmdbuffer-id", (None, "", None))[1], r.get("encoder-id", (None, "", None))[1],
                         int(r.get("event-depth", ("0",))[0] or 0), r.get("event-label", (None, "", None))[1]))
    chans = collections.Counter((r.get("process", (None, "?"))[1][:20], r.get("channel-name", (None, "?"))[1]) for r in rows)
    print("intervals by (process, channel):", dict(chans.most_common(8)))
    if not mine:
        return print("no python compute intervals")
    depths = collections.Counter(m[4] for m in mine)
    print("python compute intervals by nesting depth:", dict(depths))
    top = sorted((m for m in mine if m[4] == min(depths)), key=lambda m: m[0])
    t0, t1 = top[0][0], top[-1][0] + top[-1][1]
    busy, end, gaps = 0, t0, []
    for s, d, *_ in top:
        if s > end: gaps.append(s - end)
        busy += max(0, s + d - max(s, end)); end = max(end, s + d)
    span = t1 - t0
    cbs, encs = len({m[2] for m in top}), len({m[3] for m in mine})
    print(f"window {span / 1e6:.1f} ms for {tokens} tokens ({span / 1e6 / tokens:.2f} ms/token) | GPU busy {busy / 1e6:.1f} ms "
          f"({100 * busy / span:.1f}%) | idle {sum(gaps) / 1e6:.1f} ms in {len(gaps)} gaps (median {statistics.median(gaps) / 1e3 if gaps else 0:.1f} us, "
          f"p90 {sorted(gaps)[int(len(gaps) * .9)] / 1e3 if gaps else 0:.1f} us)")
    print(f"top-level kicks {len(top)} ({len(top) / tokens:.1f}/token) | command buffers {cbs} ({cbs / tokens:.1f}/token) | encoders {encs} ({encs / tokens:.1f}/token)")
    durs = sorted(d for _, d, *_ in top)
    print(f"kick duration median {durs[len(durs) // 2] / 1e3:.1f} us, p90 {durs[int(len(durs) * .9)] / 1e3:.1f} us, max {durs[-1] / 1e3:.1f} us")
    print("sample labels:", [m[5][:70] for m in top[:3]])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64)
