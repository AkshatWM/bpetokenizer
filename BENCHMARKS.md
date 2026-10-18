# Benchmarks

Preliminary results comparing this Rust implementation against the equivalent
pure-Python implementation in
[Neural_Networks_from_Scratch](https://github.com/AkshatWM/Neural_Networks_from_Scratch/tree/main/my_tokenizer/gpt_bpe_tokenizer).

> **Status: work in progress.** These figures come from an early run. More
> measurements across dataset sizes and vocabulary sizes are planned, along
> with further algorithmic optimizations to the Rust trainer. Treat the
> numbers below as indicative rather than final.

## Setup

| | |
| --- | --- |
| Dataset | `TinyStoriesV2-GPT4-train`, 10 MB subset |
| Vocabulary size | 1024 (256 base byte tokens + 767 merges + `<\|endoftext\|>`) |
| Build | release (`--release` / `maturin develop --release`) |
| Dataset variant | Pre-stripped — contains zero `<\|endoftext\|>` tokens |

> **Note on the dataset.** The `-tokenizer-` variants used here have had their
> `<|endoftext|>` document separators removed. The raw
> `TinyStoriesV2-GPT4-train.txt` contains roughly 1,270 per megabyte. These
> benchmarks therefore measure plain-text tokenization only and never exercise
> the special-token code path.

## Rust vs. Python

Same dataset, same vocabulary size:

| Metric | Result |
| --- | --- |
| Training speedup | **~11.7x** faster than the Python implementation |
| Latency reduction | **91.4%** — under one tenth of the original time |

## Absolute Rust timings

10 MB subset, vocabulary size 1024:

| Stage | Run 1 | Run 2 |
| --- | --- | --- |
| Train | 89.27 s | 86.79 s |
| Encode | 1.51 s | 1.44 s |
| Decode | 71.02 ms | 68.46 ms |

Run 1 and Run 2 bracket a change that moved `<|endoftext|>` handling out of a
separate preprocessing pass and into the encoder itself. Because the benchmark
dataset contains no `<|endoftext|>` tokens, that change cannot affect these
numbers — the ~3% spread between the two runs is run-to-run noise rather than a
real gain.

## Where the time goes

Training is roughly 60x slower than encoding at this vocabulary size. That is
inherent to BPE training: it is `O(num_merges x corpus_length)`, repeatedly
recomputing pair statistics and rewriting the corpus after every merge. It is
also the reason a Rust rewrite is worth doing — the whole cost sits in tight
integer loops over large byte arrays, which is close to the ideal case for Rust
over pure Python.

## TODO

- [ ] Record absolute Python timings. Currently only the relative speedup (11.7x)
      and latency reduction (91.4%) are captured, so the Python baseline
      cannot be reconstructed from this document alone.
- [ ] Benchmark with `<|endoftext|>` present, using the raw train file or a
      sample of it. The current dataset exercises the special-token path zero
      times, so its correctness and cost are both unmeasured.
- [ ] Run the 50 MB and 200 MB subsets and report compression ratio against
      training time, as originally planned.
- [ ] Re-run after the next optimization pass and track the improvement.
- [ ] Record the benchmark machine's CPU and RAM for reproducibility.
