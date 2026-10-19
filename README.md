# bpetokenizer

A fast byte-pair encoding (BPE) tokenizer written in Rust, exposed to Python as a
native extension module via [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs).

It implements the same core algorithm as
[minbpe](https://github.com/karpathy/minbpe), but the training and encoding
loops run in Rust, which makes training on multi-hundred-megabyte corpora
practical.

It is the Rust counterpart to the pure-Python BPE tokenizer in
[Neural_Networks_from_Scratch](https://github.com/AkshatWM/Neural_Networks_from_Scratch/tree/main/my_tokenizer/gpt_bpe_tokenizer),
built as a faster drop-in replacement for it. See [BENCHMARKS.md](BENCHMARKS.md)
for the measured speedup.

## Features

- **GPT-4 style pre-tokenization** using the `cl100k_base` regex pattern.
- **Training** from raw text with a target vocabulary size.
- **Encode / decode** with lossy (`replace`) or `strict` UTF-8 handling.
- **Save / load** the learned merges to a small text file.
- **`<|endoftext|>` support**: reserved as the final token id and recognised as
  a special token during encoding.

## Requirements

- Python 3.8 or newer
- A Rust toolchain (install from [rustup.rs](https://rustup.rs))
- [maturin](https://www.maturin.rs): `pip install maturin`

## Building

From the project root:

```bash
maturin develop --release
```

This compiles the extension and installs it into the active virtual environment.
You can then import it as `bpe_tokenizer`:

```python
from bpe_tokenizer import RegexTokenizer
```

## Usage

### Training

```python
from bpe_tokenizer import RegexTokenizer

tokenizer = RegexTokenizer()

with open("tinystories.txt", encoding="utf-8") as f:
    text = f.read()

# vocab_size includes the 256 base byte tokens and the <|endoftext|> token
tokenizer.train(text, vocab_size=2048, verbose=True)

tokenizer.save("tinystories")  # writes tinystories.model
```

`verbose=True` prints one line per merge, which is useful when training on a
small sample.

### Encoding and decoding

```python
ids = tokenizer.encode("hello world!")
print(ids)

print(tokenizer.decode(ids))
# -> "hello world!"
```

`encode` splits on `<|endoftext|>`, so the special token round-trips as a single
id instead of being tokenized as literal text.

`decode` accepts an `errors` argument:

```python
tokenizer.decode(ids, errors="replace")  # invalid UTF-8 becomes U+FFFD (default)
tokenizer.decode(ids, errors="strict")   # invalid UTF-8 raises ValueError
```

### Saving and loading

```python
tokenizer.save("mymodel")        # creates mymodel.model
tokenizer.load("mymodel.model")  # restores merges, vocab, and <|endoftext|>
```

The model file is a plain text format (`rustbpe v1` header followed by one
`a b merged_id` merge per line), so it is easy to diff and inspect.

It records merges only. On load, the `<|endoftext|>` id is recovered as one
past the highest merge id and the token is re-attached, so a freshly loaded
tokenizer encodes and decodes the special token correctly.

Because the id is inferred this way, `save` only accepts models whose merges
fill every slot. If training stopped early (for example on a tiny corpus),
there is no way to record the original `<|endoftext|>` id in this format, and
`save` raises an `OSError` instead of writing a model that would load wrong.

## Performance

Against the pure-Python implementation this replaces, on a 10 MB TinyStories
subset with a 1024-token vocabulary:

| Stage | Python | Rust | Speedup |
| --- | --- | --- | --- |
| Train | 1776.78 s | 0.80 s | ~2212x |
| Encode | 16.51 s | 0.52 s | ~31.8x |
| Decode | 0.341 s | 0.067 s | ~5.1x |

Training dominates the total on both sides, so the end-to-end latency reduction
is 99.92%. The trainer deduplicates pretokenized chunks and weights pair counts
by their frequency, so each merge pass scans unique words instead of every
chunk, and words are merged in place. Encoding finds the lowest-rank adjacent
pair in one scan and caches repeated chunks within a call.

These numbers are preliminary and come from a single machine. Full methodology,
absolute timings, hardware, and what is still being measured are in
[BENCHMARKS.md](BENCHMARKS.md).

## API

| Method | Description |
| --- | --- |
| `RegexTokenizer()` | Create a tokenizer with the 256 base byte tokens and the GPT-4 pre-tokenization pattern. |
| `train(target: str, vocab_size: int, verbose: bool = False)` | Train merges on `target` up to `vocab_size` tokens. |
| `encode(text: str) -> list[int]` | Encode text, handling `<|endoftext|>` as a special token. |
| `decode(ids: list[int], errors: str = "replace") -> str` | Decode ids to text, with `"replace"` or `"strict"` error handling. |
| `save(file_prefix: str)` | Write merges to `{file_prefix}.model`. |
| `load(model_file: str)` | Load merges from a `.model` file, restoring the vocabulary and `<\|endoftext\|>`. |

## License

MIT — see [LICENSE](LICENSE).
