# Benchmarks

Results comparing this Rust implementation against the equivalent pure-Python
implementation in
[Neural_Networks_from_Scratch](https://github.com/AkshatWM/Neural_Networks_from_Scratch/tree/main/my_tokenizer/gpt_bpe_tokenizer).

> **Status: work in progress.** These figures come from a single machine and a
> single 10 MB subset. More measurements across dataset and vocabulary sizes are
> planned. Treat the numbers below as indicative rather than final.

## Setup

| | |
| --- | --- |
| Dataset | `TinyStoriesV2-GPT4-train`, 10 MB subset (10,958,972 characters) |
| Vocabulary size | 1024 (256 base byte tokens + 767 merges + `<\|endoftext\|>`) |
| Build | release (`cargo build --release` / `maturin develop --release`) |
| Dataset variant | Pre-stripped — contains zero `<\|endoftext\|>` tokens |
| CPU | 12th Gen Intel Core i5-12450H |
| RAM | 16 GB (15.7 GB usable) |
| Rust | rustc / cargo 1.98.0 |
| Python | 3.12.0 |

> **Note on the dataset.** The `-tokenizer-` variants used here have had their
> `<|endoftext|>` document separators removed. The raw
> `TinyStoriesV2-GPT4-train.txt` contains roughly 1,270 per megabyte. These
> benchmarks therefore measure plain-text tokenization only and never exercise
> the special-token code path.

## Rust vs. Python

Same dataset, same vocabulary size. Python baseline: train 1776.784 s, encode
16.506 s, decode 0.341 s. Rust (standalone release harness): train 803.27 ms,
encode 519.56 ms, decode 66.78 ms.

| Stage | Python | Rust | Speedup | Latency reduction |
| --- | --- | --- | --- | --- |
| Train | 1776.784 s | 803.27 ms | **~2212x** | **99.95%** |
| Encode | 16.506 s | 519.56 ms | **~31.8x** | **96.85%** |
| Decode | 0.341 s | 66.78 ms | **~5.1x** | **80.42%** |
| **Total** | 1793.631 s | 1.390 s | **~1291x** | **99.92%** |

Training dominates the total on both sides, so the end-to-end reduction tracks
the training figure closely.

## Absolute Rust timings

10 MB subset, vocabulary size 1024:

| Stage | Standalone harness | Extension run 1 | Extension run 2 |
| --- | --- | --- | --- |
| Train | 803.27 ms | 842 ms | 806 ms |
| Encode | 519.56 ms | 579 ms | 675 ms |
| Decode | 66.78 ms | 99 ms | 332 ms |

The standalone harness is the binary used to develop and check the algorithm;
the extension runs exercise the same code through the Python bindings. Train
and encode are stable across runs. Decode is short enough that one-off
allocation and interpreter overhead show up as run-to-run spread.

## Where the time goes

The earlier Rust trainer spent ~87–89 s on this dataset. It recomputed pair
statistics over every pretokenized chunk and rebuilt the corpus after every
merge, so training cost scaled with `num_merges x corpus_length`.

The current trainer keeps the same algorithm and output but cuts that cost:

- Pretokenized chunks are deduplicated into unique words with occurrence
  counts, so each merge pass scans unique words instead of every chunk.
- Pair counts are weighted by those occurrence counts during the same pass.
- Words are merged in place rather than reallocated per merge.
- Counts are `u64`, and ties are broken deterministically (highest count, then
  lowest pair), so a trained model no longer depends on hash-map iteration
  order.

That is an ~111x improvement over the previous Rust trainer (89.27 s -> 803.27 ms)
on this dataset. Training is now about 1.5x the cost of encoding rather than the
roughly 60x gap measured before.

Encoding finds the lowest-rank adjacent pair in a single scan and caches the
token ids of repeated chunks within a call, which is why it stays in the
hundreds of milliseconds on a 10 MB input.

## TODO

- [ ] Benchmark with `<|endoftext|>` present, using the raw train file or a
      sample of it. The current dataset exercises the special-token path zero
      times, so its correctness and cost are both unmeasured.
- [ ] Run the 50 MB and 200 MB subsets and report compression ratio against
      training time, as originally planned.
- [ ] Measure whether training time still scales linearly with corpus size now
      that the merge passes are over unique words rather than all chunks.
- [ ] Re-run the Python baseline on the same machine as the Rust figures so the
      speedup is a same-hardware comparison.
