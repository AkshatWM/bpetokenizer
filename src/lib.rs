use fancy_regex::Regex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    fs,
    io::{self, ErrorKind},
};

const GPT4_REGEX_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
const EOT_STR: &str = "<|endoftext|>";

type Pair = (u32, u32);

#[pyclass(name = "RegexTokenizer")]
pub struct RegexTokenizer {
    pattern: Regex,
    merges: FxHashMap<Pair, u32>,
    vocab: FxHashMap<u32, Vec<u8>>,
    eot_id: Option<u32>,
}

impl Default for RegexTokenizer {
    fn default() -> Self {
        Self {
            pattern: Regex::new(GPT4_REGEX_PATTERN).expect("GPT-4 pattern should be there"),
            merges: FxHashMap::default(),
            vocab: (0u32..256).map(|x| (x, vec![x as u8])).collect(),
            eot_id: None,
        }
    }
}

#[pymethods]
impl RegexTokenizer {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    #[pyo3(signature = (target, vocab_size, verbose=false))]
    pub fn train(&mut self, target: String, vocab_size: u32, verbose: bool) -> PyResult<()> {
        if vocab_size <= 256 {
            return Err(PyValueError::new_err(format!(
                "vocab_size must be greater than 256, got {vocab_size}"
            )));
        }

        // The highest id is reserved for <|endoftext|>. Deriving it here keeps it
        // in step with vocab_size, rather than relying on a separate constant
        // that has to be updated by hand.
        let eot_id = vocab_size - 1;

        let text_chunks: Vec<&str> = self
            .pretokenize(target.as_str())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut index: FxHashMap<&str, usize> = FxHashMap::default();
        let mut words: Vec<Vec<u32>> = Vec::new();
        let mut freqs: Vec<u64> = Vec::new();
        for text in text_chunks {
            match index.get(text) {
                Some(&i) => freqs[i] += 1,
                None => {
                    index.insert(text, words.len());
                    words.push(text.bytes().map(|b| b as u32).collect());
                    freqs.push(1);
                }
            }
        }
        let num_merges: u32 = vocab_size - 256 - 1;
        let mut merges: FxHashMap<Pair, u32> = FxHashMap::default();
        let mut vocab: FxHashMap<u32, Vec<u8>> = (0u32..256).map(|x| (x, vec![x as u8])).collect();
        // stats -> {Pair : int}
        // merges -> {Pair : TokenId}
        for i in 0..num_merges {
            let mut stats: FxHashMap<Pair, u64> = FxHashMap::default();
            for (word, &freq) in words.iter().zip(&freqs) {
                get_stats(word, freq, &mut stats);
            }
            if stats.len() == 0 {
                break;
            }

            // Highest count, then the lexicographically smallest pair. Relying
            // on hash-map iteration order for ties would make trained models
            // depend on the map implementation.
            let best_pair = *stats
                .iter()
                .max_by(|(pair_a, count_a), (pair_b, count_b)| {
                    count_a.cmp(count_b).then_with(|| pair_b.cmp(pair_a))
                })
                .map(|(pair, _)| pair)
                .unwrap();
            let idx: u32 = 256 + i;

            for word in words.iter_mut() {
                merge_in_place(word, best_pair, idx);
            }

            merges.insert(best_pair, idx);

            let merged = [
                vocab[&best_pair.0].as_slice(),
                vocab[&best_pair.1].as_slice(),
            ]
            .concat();
            vocab.insert(idx, merged);
            if verbose {
                println!(
                    "merge {:?}/{:?}: {:?} -> {:?} ({:?}) had {:?} occurrences",
                    i + 1,
                    num_merges,
                    best_pair,
                    idx,
                    vocab.get(&idx),
                    stats.get(&best_pair)
                );
            }
        }
        vocab.insert(eot_id, EOT_STR.as_bytes().to_vec());
        self.merges = merges;
        self.vocab = vocab;
        self.eot_id = Some(eot_id);
        Ok(())
    }

    #[pyo3(signature = (ids, errors = "replace"))]
    fn decode(&self, ids: Vec<u32>, errors: &str) -> PyResult<String> {
        match errors {
            "replace" => self
                .decode_lossy(&ids)
                .map_err(|e| PyValueError::new_err(format!("{:?}", e))),
            "strict" => self
                .decode_strict(&ids)
                .map_err(|e| PyValueError::new_err(format!("{:?}", e))),
            other => Err(PyValueError::new_err(format!(
                "unsupported errors mode: {other:?} (use 'replace' or 'strict')"
            ))),
        }
    }

    pub fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let mut ids = Vec::new();
        let mut cache: FxHashMap<&str, Vec<u32>> = FxHashMap::default();

        for (i, part) in text.split(EOT_STR).enumerate() {
            if i > 0 {
                let eot_id = self.eot_id.ok_or_else(|| {
                    PyValueError::new_err("cannot encode '<|endoftext|>': eot_id is not set")
                })?;
                ids.push(eot_id);
            }
            for chunk in self
                .pretokenize(part)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
            {
                let toks = cache
                    .entry(chunk)
                    .or_insert_with(|| self.encode_chunk(chunk));
                ids.extend_from_slice(toks);
            }
        }
        Ok(ids)
    }

    pub fn save(&self, file_prefix: &str) -> io::Result<()> {
        // The file stores merges only, and load() recovers the EOT id as one
        // past the highest merge id. Only save models that fill every merge
        // slot; otherwise the EOT id would shift on reload.
        let eot_id = self.eot_id.ok_or_else(|| {
            io::Error::new(ErrorKind::InvalidInput, "no eot id: call train() first")
        })?;
        if self.merges.len() as u32 != eot_id - 256 {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "cannot save: training stopped before filling the vocabulary, \
                 so the eot id is not recoverable from the merge file",
            ));
        }

        let mut out = String::from("rustbpe v1\n");
        let mut merges: Vec<_> = self.merges.iter().collect();
        merges.sort_by_key(|&(_, &idx)| idx);
        for (&(a, b), &idx) in merges {
            out.push_str(&format!("{a} {b} {idx}\n"));
        }
        fs::write(format!("{file_prefix}.model"), out)
    }

    pub fn load(&mut self, model_file: &str) -> io::Result<()> {
        let text = fs::read_to_string(model_file)?;
        let mut lines = text.lines();
        if lines.next() != Some("rustbpe v1") {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "unrecognized header",
            ));
        }

        // Parse into a temporary map so a bad file cannot leave the tokenizer
        // half-loaded.
        let mut merges: FxHashMap<Pair, u32> = FxHashMap::default();
        for line in lines {
            let nums: Vec<u32> = line
                .split_whitespace()
                .map(|s| s.parse())
                .collect::<Result<_, _>>()
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
            let &[a, b, idx] = nums.as_slice() else {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "merge lines must be '<a> <b> <idx>'",
                ));
            };
            if a >= idx || b >= idx {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "merge inputs must have lower ids than the merged token",
                ));
            }
            if merges.insert((a, b), idx).is_some() {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "duplicate merge pair",
                ));
            }
        }

        // The model file stores merges only. Require them to occupy a
        // contiguous range 256.., then the EOT id is exactly one past the
        // highest merge id (or 256 when there are no merges).
        let mut ids: Vec<u32> = merges.values().copied().collect();
        ids.sort_unstable();
        for (expected, &idx) in (256u32..).zip(&ids) {
            if idx != expected {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "merge ids must form a contiguous range starting at 256",
                ));
            }
        }
        let eot_id = 256 + ids.len() as u32;

        self.merges = merges;
        self.rebuild_vocab();
        self.eot_id = Some(eot_id);
        self.vocab.insert(eot_id, EOT_STR.as_bytes().to_vec());
        Ok(())
    }
}

#[derive(Debug)]
pub enum DecodeError {
    UnknownId(u32),
    InvalidUtf8(String),
}

impl RegexTokenizer {
    fn pretokenize<'a>(&self, target: &'a str) -> Result<Vec<&'a str>, fancy_regex::Error> {
        self.pattern
            .find_iter(target)
            .map(|m| m.map(|m| m.as_str()))
            .collect()
    }

    fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, DecodeError> {
        let mut tokens = Vec::new();
        for &id in ids {
            let bytes = self.vocab.get(&id).ok_or(DecodeError::UnknownId(id))?;
            tokens.extend_from_slice(bytes);
        }
        Ok(tokens)
    }

    fn decode_lossy(&self, ids: &[u32]) -> Result<String, DecodeError> {
        self.decode_bytes(ids)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    }

    fn decode_strict(&self, ids: &[u32]) -> Result<String, DecodeError> {
        let bytes = self.decode_bytes(ids)?;
        String::from_utf8(bytes).map_err(|e| DecodeError::InvalidUtf8(e.to_string()))
    }

    fn encode_chunk(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();

        while ids.len() >= 2 {
            // find the adjacent pair with the lowest merge rank
            let mut best: Option<(Pair, u32)> = None;
            for w in ids.windows(2) {
                let pair = (w[0], w[1]);
                if let Some(&rank) = self.merges.get(&pair) {
                    if best.map_or(true, |(_, r)| rank < r) {
                        best = Some((pair, rank));
                    }
                }
            }
            let Some((pair, idx)) = best else { break };
            merge_in_place(&mut ids, pair, idx);
        }
        ids
    }

    fn rebuild_vocab(&mut self) {
        self.vocab = (0..256u32).map(|i| (i, vec![i as u8])).collect();
        let mut merges: Vec<_> = self.merges.iter().collect();
        merges.sort_by_key(|&(_, &idx)| idx);
        for (&(a, b), &idx) in merges {
            let mut token = self.vocab[&a].clone();
            token.extend_from_slice(&self.vocab[&b]);
            self.vocab.insert(idx, token);
        }
    }
}

fn get_stats(word: &[u32], freq: u64, stats: &mut FxHashMap<Pair, u64>) {
    for w in word.windows(2) {
        *stats.entry((w[0], w[1])).or_insert(0) += freq;
    }
}

fn merge_in_place(ids: &mut Vec<u32>, pair: Pair, new_id: u32) {
    let (mut r, mut w) = (0, 0);
    while r < ids.len() {
        if r + 1 < ids.len() && ids[r] == pair.0 && ids[r + 1] == pair.1 {
            ids[w] = new_id;
            r += 2;
        } else {
            ids[w] = ids[r];
            r += 1;
        }
        w += 1;
    }
    ids.truncate(w);
}

#[pymodule]
fn bpe_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RegexTokenizer>()?;
    Ok(())
}
