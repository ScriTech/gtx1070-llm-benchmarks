# Local LLMs on two GTX 1070s: what actually works

We tested 14 models on a pair of **NVIDIA GTX 1070 8 GB** cards (Pascal, 16 GB of VRAM in total), mostly with **llama.cpp**. All 14 were speed-tested; 10 of them (17 configurations in all) were scored on the same 72-point quality suite, and the best ones were pushed to 128K to 256K tokens of context. This repository holds the results, the settings that worked, and the problems we hit.

Tested 27 to 30 October 2026. Everything here comes from one machine, so treat the numbers as a ranking on this hardware, not as absolute quality scores.

## Test hardware

| | |
|---|---|
| Machine | Dell Precision Tower 7810 workstation |
| CPUs | 2&times; Intel Xeon E5-2650 v3 (10 cores / 20 threads each, 40 threads total) |
| RAM | 128 GB DDR4 (8&times; 16 GB, running at 2133 MT/s) |
| GPUs | 2&times; NVIDIA GeForce GTX 1070, 8 GB each (Pascal, compute capability 6.1) |
| Host OS | Proxmox VE 9.2 (kernel 7.0 Proxmox build) |
| Test environment | Unprivileged Debian 13 LXC container: 8 vCPUs, 32 GB RAM, both GPUs passed through as device nodes |
| GPU software | NVIDIA driver 580.173.02, CUDA 12.8 |
| Storage | SATA SSDs |

Models were loaded from SSD and tested one at a time.

## Our suggestions

| If you want... | Use this | Quality | Decode |
|---|---|---|---|
| The best all-round model | **Gemma 4 26B-A4B QAT**, tensor split, q8 KV | 98.6% | 46.9 tok/s |
| The fastest at high quality | **Gemma 4 12B QAT** with an MTP draft model, layer split | 97.2% | 71.4 tok/s |
| Top accuracy, and you can wait | **Qwen3.6 35B-A3B**, layer split, `-ncmoe 24` | 100% | 13.2 tok/s |
| Top accuracy at a usable speed | **Ornith 1.5 35B-A3B APEX** on the [PolyStrata](https://github.com/VecSzn/PolyStrata) engine | 100% | 30.0 tok/s |
| A strong dense 27B | **Qwen3.5 27B** IQ4_XS, tensor split, q4 KV | 97.2% | 18.1 tok/s |
| A small, quick model | **Ornith 1.5 9B** with MTP, or **Gemma 4 E4B** with MTP on a single card | 95.8% | 52 to 55 tok/s |

Avoid **Qwen3.5 9B**: it failed the quality suite (55.6% plain, 37.5% with MTP) because its reasoning gets stuck in a loop on the code test until it runs out of tokens.

### Commands that worked

All four were run with `-c 65536 -np 1 -fa on -ngl 99 --jinja` plus:

```
# Gemma 4 26B-A4B QAT (gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf)
-sm tensor -ctk q8_0 -ctv q8_0 -ub 256

# Gemma 4 12B QAT + MTP (gemma-4-12b-it-qat-q4_0.gguf, gemma-4-12B-it-qat-MTP-Q4_0.gguf)
-sm layer -ctk f16 -ctv f16 -ub 512 -md gemma-4-12B-it-qat-MTP-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 8

# Qwen3.6 35B-A3B (Qwen3.6-35B-A3B-UD-IQ4_XS.gguf)
-sm layer -ncmoe 24 -ctk q8_0 -ctv q8_0

# Qwen3.5 27B (Qwen3.5-27B-IQ4_XS.gguf)
-sm tensor -ctk q4_0 -ctv q4_0 -ub 256
```

`-sm tensor` (tensor split across both cards) needs a recent llama.cpp. We used upstream commit `d05706b4f`.

The Ornith 35B APEX result used PolyStrata, not llama.cpp. It needed `--vram-reserve-mib 1200` to stay stable on these cards. See the PolyStrata repository for how to start it.

## Where to get the models and engines

Weights are GGUF files from Hugging Face. We ran llama.cpp directly and did not test through Ollama, so quantizations and speeds there may differ. Ollama has library pages for the model families ([gemma4](https://ollama.com/library/gemma4), [qwen3.5](https://ollama.com/library/qwen3.5), [qwen3.6](https://ollama.com/library/qwen3.6)), but its tags are not the same files we tested.

| Model | File we used | Source |
|---|---|---|
| Gemma 4 26B-A4B QAT | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) |
| Gemma 4 12B QAT | `gemma-4-12b-it-qat-q4_0.gguf` | [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) |
| Gemma 4 12B MTP draft | `gemma-4-12B-it-qat-MTP-Q4_0.gguf` | [magnitudedev/gemma-4-12B-it-qat-MTP-GGUF](https://huggingface.co/magnitudedev/gemma-4-12B-it-qat-MTP-GGUF) |
| Gemma 4 E4B QAT | `gemma-4-E4B_q4_0-it.gguf` | [google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf) |
| Gemma 4 E4B MTP draft | `gemma-4-E4B-it-qat-MTP-Q4_0.gguf` | [magnitudedev/gemma-4-E4B-it-qat-MTP-GGUF](https://huggingface.co/magnitudedev/gemma-4-E4B-it-qat-MTP-GGUF) |
| Qwen3.6 35B-A3B | `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| Qwen3.8 27B | `Qwen3.8-27B-UD-IQ4_XS.gguf` | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| Qwen3.5 27B | `Qwen3.5-27B-IQ4_XS.gguf` | [unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) |
| Qwen3.5 9B (with MTP) | `Qwen3.5-9B-UD-Q4_K_XL.gguf` | [unsloth/Qwen3.5-9B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF) |
| Ornith 1.5 35B-A3B | `Ornith-1.5-35B-Q4_K_M.gguf` | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) |
| Ornith 1.5 9B | `Ornith-1.5-9B-Q4_K_M.gguf` | [ornith-ai/Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) |
| Ornith 1.5 35B-A3B APEX | `Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf` | [mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) |

Engines: [llama.cpp](https://github.com/ggml-org/llama.cpp) (upstream, commit `d05706b4f`) and [PolyStrata](https://github.com/VecSzn/PolyStrata) (MIT licensed, built on [Strata](https://github.com/Niko1221/Strata)). The PolyStrata Gemma 26B run also used the MTP drafter that PolyStrata ships with. We also benchmarked the [PXA](https://github.com/poisonxa16/pxa) llama.cpp fork; see Credits.

## Quality results

The suite is 72 points. Retrieval (3) means finding facts buried in a long document, the agent tests (6 and 4) are tool-use scenarios, triage (12) is classifying support tickets, extraction (5) is pulling fields out of messy text, and code (42) is writing two Python functions and passing 14 unit tests, three attempts each. Everything ran at 65K context with thinking on and no token cap. Pass is 90% or more, marginal is 70 to 89%, fail is below 70%.

| Model | Config | Score | Percent | Verdict | Decode tok/s |
|---|---|---|---|---|---|
| Gemma 4 26B-A4B QAT | tensor, q8 KV + ngram-map-k | 72/72 | 100.0% | Pass | 46.9 |
| Ornith 1.5 35B-A3B APEX | PolyStrata, reserve 1200 MiB | 72/72 | 100.0% | Pass | 30.0 |
| Qwen3.6 35B-A3B | layer, -ncmoe 24, q8 KV | 72/72 | 100.0% | Pass | 13.2 |
| Gemma 4 26B-A4B QAT | tensor, q8 KV, ub256 | 71/72 | 98.6% | Pass | 46.9 |
| Ornith 1.5 35B-A3B | layer, -ncmoe 30, f16 KV, MTP | 71/72 | 98.6% | Pass | 18.7 |
| Ornith 1.5 35B-A3B | layer, -ncmoe 30, f16 KV | 71/72 | 98.6% | Pass | 16.9 |
| Qwen3.8 27B | tensor, q8 KV, ub256 | 71/72 | 98.6% | Pass | 15.7 |
| Gemma 4 12B QAT | layer, f16 KV, MTP draft 8 | 70/72 | 97.2% | Pass | 71.4 |
| Gemma 4 12B QAT | layer, f16 KV | 70/72 | 97.2% | Pass | 23.0 |
| Qwen3.5 27B | tensor, q4 KV, ub256 | 70/72 | 97.2% | Pass | 18.1 |
| Ornith 1.5 9B | tensor, f16 KV + draft-mtp | 69/72 | 95.8% | Pass | 54.8 |
| Gemma 4 E4B QAT | 1 GPU, f16 KV, MTP, ub1024 | 69/72 | 95.8% | Pass | 52.8 |
| Ornith 1.5 9B | tensor, f16 KV, ub512 | 69/72 | 95.8% | Pass | 45.1 |
| Gemma 4 26B-A4B QAT | PolyStrata, MTP | 69/72 | 95.8% | Pass | 24.4 |
| Gemma 4 E4B QAT | tensor, f16 KV | 68/72 | 94.4% | Pass | 48.9 |
| Qwen3.5 9B | tensor, f16 KV | 40/72 | 55.6% | Fail | 42.9 |
| Qwen3.5 9B | tensor, f16 KV, MTP, ub256 | 27/72 | 37.5% | Fail | 55.0 |

Rows with "PolyStrata" in the config column ran on the [PolyStrata](https://github.com/VecSzn/PolyStrata) engine, an open-source inference engine with its own runtime for some models. Every other row ran on upstream llama.cpp.

## Long context

Each run loads the server at that context size, fills about 90% of it, hides three codes at 10%, 50% and 90% depth and asks for them back (thinking off). Every run that accepted the prompt found all three codes. Rows below are the f16-KV runs; the CSV also has q8_0.

| Model | Config | Context | Prefill tok/s | Decode tok/s | Peak VRAM GB |
|---|---|---|---|---|---|
| Gemma 4 12B QAT | layer, MTP draft 8 | 128K | 108 | 13.6 | 6.2 |
| Gemma 4 12B QAT | layer, MTP draft 8 | 192K | 78 | 7.0 | 7.2 |
| Gemma 4 12B QAT | layer, MTP draft 8 | 256K | 60 | 5.7 | 7.5 |
| Gemma 4 12B QAT | layer | 128K | 138 | 12.2 | 5.5 |
| Gemma 4 12B QAT | layer | 192K | 101 | 10.1 | 6.3 |
| Qwen3.5 9B | tensor | 128K | 342 | 28.8 | 4.9 |
| Qwen3.5 9B | tensor | 192K | 263 | 25.3 | 6.0 |
| Qwen3.5 9B | tensor, MTP, ub256 | 128K | 299 | 46.5 | 5.3 |
| Qwen3.5 9B | tensor, MTP, ub256 | 192K | 227 | 39.6 | 6.5 |
| Gemma 4 E4B QAT | 1 GPU, MTP, ub1024 | 128K | 170 | 35.9 | 5.7 |
| Gemma 4 E4B QAT | tensor | 128K | 291 | 31.8 | 3.1 |

Gemma 4 E4B was trained for 128K, so our 192K runs were rejected by the server.

## KV cache types

Both of the models below were tested at 65K context. K and V are the key and value caches.

| Model | Setup | K / V | Result | Codes found | Decode tok/s | Peak VRAM GB |
|---|---|---|---|---|---|---|
| Qwen3.5 27B | tensor, ub256 | q4_0 / q4_0 | Fits | 3/3 | 18.1 | 7.7 |
| Qwen3.5 27B | tensor, ub256 | iq4_nl / iq4_nl | No fit |  |  |  |
| Qwen3.5 27B | tensor, ub256 | q5_0 / q4_0 | Fits | 3/3 | 17.9 | 7.8 |
| Qwen3.5 27B | tensor, ub256 | q8_0 / q4_0 | No fit |  |  |  |
| Qwen3.5 27B | tensor, ub256 | q5_0 / q5_0 | Fits | 3/3 | 17.8 | 7.8 |
| Qwen3.6 35B-A3B | layer, -ncmoe 24 | q8_0 / q8_0 | Fits | 3/3 | 13.2 | 7.6 |
| Qwen3.6 35B-A3B | layer, -ncmoe 24 | f16 / f16 | Fits | 3/3 | 13.2 | 7.8 |
| Qwen3.6 35B-A3B | layer, -ncmoe 24 | q5_0 / q5_0 | Fits | 3/3 | 13.0 | 7.5 |
| Qwen3.6 35B-A3B | layer, -ncmoe 24 | q8_0 / q4_0 | Fits | 3/3 | 13.1 | 7.5 |
| Qwen3.6 35B-A3B | layer, -ncmoe 24 | q4_0 / q4_0 | Fits | 3/3 | 13.1 | 7.5 |
| Qwen3.6 35B-A3B | layer, -ncmoe 22 | q4_0 / q4_0 | No fit |  |  |  |
| Qwen3.6 35B-A3B | layer, -ncmoe 20 | q4_0 / q4_0 | No fit |  |  |  |

## What we learned

- **Tensor split is usually fastest**, but it needs a recent llama.cpp and leaves little memory headroom on 8 GB cards.
- **q8 KV is slower than f16 for decode on Pascal** (10 to 25% in our sweep) with the same recall, so use it only when you need the memory.
- **MTP draft models raised decode speed on the small Gemma models, Ornith 9B and Qwen3.5 9B**, but at 192K context the 12B was slower with MTP than without.
- **ngram speculation only helps repetitive text.** It gave no speed gain on our general-purpose tests.
- **Qwen3.5 27B only fits at 65K with q5 or q4 KV** on two 8 GB cards. q8 KV runs out of memory even with a small micro-batch.
- **Qwen3.6 35B-A3B is insensitive to KV type** (about 13 tok/s whether f16, q8 or q4), because few of its layers use attention. Offloading fewer expert layers (`-ncmoe` below 24) ran out of memory.
- **Some models loop.** Qwen3.5 9B repeated the same line hundreds of times in its reasoning and never answered the code test.

## Caveats

- One machine, two identical GTX 1070s, and a suite we wrote ourselves. A 1-point difference on a 72-point score is noise.
- Speeds are short single-request measurements, not throughput under load.
- Thinking was on and uncapped, which favours models that reason carefully and punishes ones that loop.
- Where we tested only part of a configuration it says so in the table.

## Credits

None of this would exist without other people's open-source work. Thank you to:

| Project | Author | What it gave us |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Georgi Gerganov and the ggml-org contributors | The inference engine behind almost every result here (MIT). |
| [PolyStrata](https://github.com/VecSzn/PolyStrata) | VecSzn and the PolyStrata contributors | The engine for the Ornith 35B APEX rows and the PolyStrata Gemma 26B row (MIT). |
| [Strata](https://github.com/Niko1221/Strata) | Niko1221 and the Strata contributors | The runtime PolyStrata is built on (MIT). |
| [PXA](https://github.com/poisonxa16/pxa) | poisonxa16 | A llama.cpp fork with extra CUDA kernels that we built and benchmarked on these cards (MIT). On the 27B model we compared, stock llama.cpp with `-sm tensor` was faster (15.7 against 12.6 tok/s), so PXA is not behind any headline result, but its tests shaped how we compared engines. |
| [Unsloth](https://huggingface.co/unsloth) | Unsloth AI | Quantized GGUF builds of the Gemma and Qwen models. |
| [magnitudedev](https://huggingface.co/magnitudedev) | magnitudedev | The MTP draft models for Gemma 4 12B and E4B. |
| [mudler](https://huggingface.co/mudler) | mudler | The APEX quantization of Ornith 1.5 35B-A3B. |
| [ornith-ai](https://huggingface.co/ornith-ai) | ornith-ai | The Ornith 1.5 models and GGUFs. |
| [Google](https://huggingface.co/google) | Google | The Gemma 4 models and their QAT builds. |
| [Qwen](https://huggingface.co/Qwen) | The Qwen team | The Qwen3.5, Qwen3.6 and Qwen3.8 models. |

The models and engines keep their own licenses. If we have missed anyone who should be credited, please open an issue and we will add you.

## Files

- `data/quality.csv`, `data/speed.csv`, `data/long_context_sweep.csv`, `data/kv_cache.csv`: the raw tables behind this page.
- `report.html`: the same results as a single page you can open offline (charts, sortable tables).
- `benchmark-data.xlsx`: the data as a workbook.
- `LICENSE`: the results, text and charts here are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit this repository if you reuse them. The models and engines linked above keep their own licenses.
