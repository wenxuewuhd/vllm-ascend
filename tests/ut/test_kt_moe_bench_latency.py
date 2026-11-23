# SPDX-License-Identifier: Apache-2.0
"""The HTTP benchmark must count token IDs rather than visible text chunks."""

import copy
import json
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "examples/kt_moe_offload/bench_latency.sh"


def load_benchmark():
    source = SCRIPT.read_text().split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY\n", 1)[0]
    module = ModuleType("benchmark")
    exec(compile(source, str(SCRIPT), "exec"), module.__dict__)
    return module


def events():
    def choice(ids, text="", finish=None):
        return {
            "id": "cmpl-test",
            "model": "dsv41",
            "choices": [{"index": 0, "text": text, "token_ids": ids, "finish_reason": finish}],
        }

    return [
        (choice([10, 11, 12]), 0.2),
        (choice([13, 14], "ab"), 0.4),
        (choice([15, 16, 17], "cd", "length"), 0.6),
        (
            {
                "id": "cmpl-test",
                "model": "dsv41",
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
            },
            0.7,
        ),
        ("[DONE]", 0.8),
    ]


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.bench = load_benchmark()
        self.args = SimpleNamespace(model="dsv41", prompt_tokens=4, gen_tokens=8)

    def measure(self, stream):
        wire = ((data if isinstance(data, str) else json.dumps(data), received) for data, received in stream)
        return self.bench.measure(wire, self.args, 0)

    def test_speculative_batches_and_empty_text(self):
        result = self.measure(events())
        self.assertEqual(result["token_ids"], list(range(10, 18)))
        self.assertEqual(result["token_batch_count"], 3)
        self.assertEqual(result["first_batch_tokens"], 3)
        self.assertEqual(result["ttft_ids_s"], 0.2)
        self.assertEqual(result["ttft_text_s"], 0.4)
        self.assertEqual(result["e2e_done_s"], 0.8)
        self.assertEqual(result["tpot_delivery_ms"], (0.6 - 0.2) / 7 * 1000)

    def test_rejects_incomplete_or_inconsistent_stream(self):
        variants = [events()[:-1], events()[:3] + events()[4:]]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            stream = events()
            stream[-2][0]["usage"][key] += 1
            variants.append(stream)
        for key, value in (("token_ids", None), ("token_ids", [True]), ("finish_reason", "stop")):
            stream = events()
            stream[2][0]["choices"][0][key] = value
            variants.append(stream)
        for key in ("id", "model"):
            stream = events()
            stream[1][0][key] = "other"
            variants.append(stream)
        variants += [events() + [copy.deepcopy(events()[0])], events()[:-1] + events()[-2:]]
        for stream in variants:
            with self.subTest(stream=stream), self.assertRaises(ValueError):
                self.measure(stream)

    def test_sse_uses_frame_boundaries(self):
        lines = [(b": heartbeat\r\n", 0.1), (b'data: {"a":\n', 0.2), (b"data: 1}\n", 0.3), (b"\n", 0.4)]
        self.assertEqual(list(self.bench.read_sse(lines)), [('{"a":\n1}', 0.4)])
        with self.assertRaises(ValueError):
            list(self.bench.read_sse(lines[:-1]))

    def test_calibrates_exact_prompt_length(self):
        prompt, source = self.bench.build_prompt("中文 text\n mixed words", 4096, len)
        self.assertEqual(len(prompt), 4096)
        self.assertEqual(source, "中文 text mixed words")


if __name__ == "__main__":
    unittest.main()
