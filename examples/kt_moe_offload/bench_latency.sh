#!/usr/bin/env bash
# Measure TTFT, TPOT and end-to-end latency against an already-running server.
# PORT=8100 CONCURRENCY=1 REQUESTS=10 PROMPT_TOKENS=4096 GEN_TOKENS=128 \
#   PROMPT_FILE=/path/to/text.txt bash bench_latency.sh
# Counts generated token IDs, including empty text and speculative token batches.
# TPOT measures client delivery: (last ID batch - first ID batch)/(tokens - 1).
# OUTPUT_DIR must be new; it contains requests, timestamped SSE and results.
set -euo pipefail
export OUTPUT_DIR=${OUTPUT_DIR:-$PWD/throughput/$(date -u +%Y%m%dT%H%M%S%NZ)}
python3 - <<'PY'
"""TPOT = (last ID batch - first ID batch)/(output tokens - 1); client delivery timing."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib import request

def require(ok, message):
    if not ok:
        raise ValueError(message)

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def open_http(url, payload=None, timeout=3600):
    body = None if payload is None else json.dumps(payload).encode()
    return request.build_opener(request.ProxyHandler({})).open(
        request.Request(url, body, {"Content-Type": "application/json"}), timeout=timeout)

def read_sse(lines):
    data = []
    for raw, received in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line and data:
            yield "\n".join(data), received
            data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    require(not data, "EOF inside an unfinished SSE frame")

def measure(events, args, started):
    ids, text, batches = [], [], []
    done = finish = usage = response_id = first_text = None
    for data, received in events:
        require(done is None, "SSE data after [DONE]")
        if data == "[DONE]":
            done = received
            continue
        payload = json.loads(data)
        require(isinstance(payload, dict) and "error" not in payload, f"server SSE error: {payload}")
        response_id = response_id or payload.get("id")
        require(response_id and payload.get("id") == response_id and payload.get("model") == args.model,
                "response id/model mismatch")
        choices = payload.get("choices")
        require(isinstance(choices, list), "missing choices array")
        if not choices:
            require(finish is not None and usage is None, "early/duplicate final usage")
            usage = payload.get("usage")
            require(isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")), "invalid final usage")
            continue
        require(finish is None and usage is None and payload.get("usage") is None and len(choices) == 1,
                "unexpected choice/usage after finish")
        choice = choices[0]
        chunk = choice.get("token_ids")
        require(type(choice.get("index")) is int and choice["index"] == 0 and isinstance(chunk, list)
                and all(type(t) is int and t >= 0 for t in chunk), "missing/invalid output token IDs")
        require(isinstance(choice.get("text"), str), "invalid text")
        if chunk:
            ids.extend(chunk)
            batches.append((received, len(chunk)))
        text.append(choice["text"])
        if choice["text"] and first_text is None:
            first_text = received
        finish = choice.get("finish_reason")
    require(done is not None, "missing [DONE]")
    require(finish == "length", f"expected finish_reason=length; got {finish!r}")
    require(usage is not None, "missing final usage")
    n = len(ids)
    require(n == args.gen_tokens == usage["completion_tokens"], "output IDs/usage/requested count mismatch")
    require(usage["prompt_tokens"] == args.prompt_tokens and usage["total_tokens"] == args.prompt_tokens + n,
            "prompt/total usage mismatch")
    first, last = batches[0][0], batches[-1][0]
    require(started <= first <= last <= done, "invalid timestamp ordering")
    return {"status": "pass", "output_tokens": n, "usage": usage, "finish_reason": finish,
            "token_ids": ids, "text": "".join(text), "response_id": response_id,
            "first_batch_tokens": batches[0][1], "token_batch_count": len(batches),
            "ttft_ids_s": first - started, "ttft_text_s": None if first_text is None else first_text - started,
            "e2e_done_s": done - started, "tpot_delivery_ms": (last - first) / (n - 1) * 1000 if n > 1 else None}

def run_request(index, args, prompt):
    stem = args.output_dir / f"request-{index:04d}"
    payload = {"model": args.model, "prompt": prompt, "max_tokens": args.gen_tokens, "temperature": 0.0,
               "stream": True, "ignore_eos": True, "n": 1, "echo": False, "return_token_ids": True,
               "stream_options": {"include_usage": True, "continuous_usage_stats": False}}
    write_json(stem.with_suffix(".request.json"), payload)
    events = []
    def capture(response):
        for data, received in read_sse((raw, time.perf_counter()) for raw in response):
            events.append({"received_perf": received, "elapsed_s": received - started, "data": data})
            yield data, received
    started = time.perf_counter()
    try:
        with open_http(args.base + "/v1/completions", payload) as response:
            require(response.headers.get_content_type() == "text/event-stream", "expected SSE")
            result = measure(capture(response), args, started)
    except Exception as exc:
        result = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    stem.with_suffix(".events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    result.update(request_index=index, start_perf=started)
    write_json(stem.with_suffix(".result.json"), result)
    return result

def build_prompt(sample, target, token_counter):
    source = " ".join(sample.split())
    require(source, "empty prose")
    per_char = token_counter(source[:4000]) / len(source[:4000])
    require(per_char > 0, "zero prompt tokens in calibration")
    chars, tried = max(1, round(target / per_char)), set()
    for _ in range(10):
        if chars in tried:
            break
        tried.add(chars)
        text = (source * (chars // len(source) + 1))[:chars]
        got = token_counter(text)
        if got == target:
            return text, source
        step = round((target - got) / per_char)
        chars = max(1, chars + (step if step else (1 if got < target else -1)))
    raise ValueError(f"prompt target {target} not reached")

SAMPLE = """
The scheduler keeps a queue of requests and decides, on every step, which of
them make progress. A request that has not yet produced its first token is in
the prefill phase: its whole prompt must be read before anything can be emitted,
and the cost of that grows with the length of the prompt. Once the first token
is out, the request moves into decoding, where each further token costs roughly
the same as the one before it, and the pressure moves from arithmetic to memory.
Serving systems are judged on both, and the two rarely improve together.
Memory is the harder half. Keys and values computed during prefill are kept so
that decoding does not recompute them, and that cache grows with every token in
flight. Capacity is finite, so a scheduler that admits too many requests will
find itself evicting work it has already paid for, recomputing it later, and
delivering worse latency than if it had admitted fewer. Admission control, block
size, and the eviction policy all trade throughput against tail latency, and the
right balance depends on the shape of the traffic rather than on the model.
Mixture-of-experts models complicate this further. Each token is routed to a
small subset of the available experts, so the weights actually touched depend on
the content of the request rather than its length alone. Two prompts of equal
size can differ substantially in cost when one concentrates on a handful of
experts and the other spreads across many. Systems that keep only part of the
expert set in fast memory are especially sensitive: a request whose tokens route
to resident experts runs at one speed, and a request that reaches for the rest
runs at another, without anything in the request itself announcing which it is.
Measurement has to respect all of this. A benchmark that sends the same token
thousands of times measures a routing pattern no real workload produces; one
that samples the vocabulary uniformly measures a different unreal pattern. What
the operator wants to know is the behaviour under traffic resembling their own,
which is why published serving numbers are usually quoted against a corpus of
real conversations rather than against synthetic filler, and why a number taken
under one corpus should not be carried to another without saying so.
Latency figures also depend on how they are defined. Time to first token covers
queueing, prompt processing, and the first forward pass; time per output token
covers everything after. Reporting a single average hides the distribution, and
the distribution is what users experience: a service whose median is excellent
and whose tail is poor will be described as slow. Percentiles, the number of
requests behind them, and the conditions under which they were gathered belong
next to any figure that is meant to be acted upon.
"""

def main():
    out = Path(os.environ["OUTPUT_DIR"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(output_dir=out, model=os.getenv("MODEL_NAME", "dsv41"))
    for name, default in (("concurrency", 1), ("requests", 10), ("prompt_tokens", 4096), ("gen_tokens", 128)):
        setattr(args, name, int(os.getenv(name.upper(), default)))
        require(getattr(args, name) > 0, f"{name} must be positive")
    args.base = f"http://{os.getenv('HOST', '127.0.0.1')}:{int(os.getenv('PORT', '8100'))}"
    write_json(out / "run.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    sample = Path(os.environ["PROMPT_FILE"]).read_text(encoding="utf-8") if os.getenv("PROMPT_FILE") else SAMPLE
    with open_http(args.base + "/v1/models", timeout=600) as response:
        require(any(m["id"] == args.model for m in json.load(response)["data"]), "requested model is not served")
    def token_counter(text):
        with open_http(args.base + "/v1/completions", {"model": args.model, "prompt": text, "max_tokens": 1,
                                                     "temperature": 0.0}, timeout=600) as response:
            return json.load(response)["usage"]["prompt_tokens"]
    prompt, prose = build_prompt(sample, args.prompt_tokens, token_counter)
    (out / "prompt.txt").write_text(prompt)
    (out / "source_prose.txt").write_text(prose)
    print(f"prompt: {args.prompt_tokens} tokens of English prose ({len(set(prose.split()))} distinct source words)", flush=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda i: run_request(i, args, prompt), range(1, args.requests + 1)))
    summary = {"status": "pass" if all(r["status"] == "pass" for r in results) else "fail",
               "wall_s": time.perf_counter() - started, "requests": args.requests, "results": results}
    if summary["status"] == "pass":
        for metric in ("ttft_ids_s", "e2e_done_s", "tpot_delivery_ms"):
            values = sorted(r[metric] for r in results if r[metric] is not None)
            summary[metric] = None if not values else dict(mean=statistics.fmean(values), min=values[0],
                p50=values[len(values) // 2], max=values[-1], sample_sd=statistics.stdev(values) if len(values) > 1 else None)
        summary["aggregate_e2e_output_tokens_per_s"] = sum(r["output_tokens"] for r in results) / summary["wall_s"]
    write_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"evidence: {out}")
    return 0 if summary["status"] == "pass" else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY
