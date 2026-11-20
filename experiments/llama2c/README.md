# Offcut Relay local-runtime experiment

The llama2.c reranker matched **1 of 3** expected evidence passages. The existing retrieval ranking matched **3 of 3** on the same cases. The deployed ranking remains the existing retrieval method. The out-of-scope query returned no evidence. See `evaluation.json` for candidate scores and per-case outcomes.

The source is the public Offcut Relay local inventory bundle. This experiment adds a C continuation scorer and a Python evaluation runner. It uses CPU inference and needs no model server or hosted API.

## Reproduce

Place these upstream files in `experiments/llama2c`.

| File | Upstream source | SHA-256 |
| --- | --- | --- |
| `run.c` | [llama2.c](https://github.com/karpathy/llama2.c/blob/2fbf7059aab6f7e44047da1ff5c0ba53057a248e/run.c) | `9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd` |
| `stories260K.bin` | [TinyStories checkpoint](https://huggingface.co/karpathy/tinyllamas/blob/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/stories260K.bin) | `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696` |
| `tok512.bin` | [Matching tokenizer](https://huggingface.co/karpathy/tinyllamas/blob/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/tok512.bin) | `037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312` |

Run these commands from the extracted root on Linux with a C compiler, Python 3.10 or newer and Node.js 18 or newer.

```sh
cc -O2 experiments/llama2c/score_continuation.c -lm -o experiments/llama2c/score_continuation
python3 -m workflow.llama2c_eval experiments/llama2c/score_continuation \
  experiments/llama2c/stories260K.bin experiments/llama2c/tok512.bin
```

Small floating-point differences are possible across builds. These three fixed fixtures are a smoke test for the reranking approach. They do not measure its accuracy on operator questions.
