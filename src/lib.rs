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
        self.eot_id = Some(vocab_size - 1);
        let text_chunks: Vec<&str> = self
            .pretokenize(target.as_str())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut ids: Vec<Vec<u32>> = Vec::new();
        for text in text_chunks {
            let tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
            ids.push(tokens);
        }
        let num_merges: u32 = vocab_size - 256 - 1;
        let mut merges: FxHashMap<Pair, u32> = FxHashMap::default();
        let mut vocab: FxHashMap<u32, Vec<u8>> = (0u32..256).map(|x| (x, vec![x as u8])).collect();
        // stats -> {Pair : int}
        // merges -> {Pair : TokenId}
        for i in 0..num_merges {
            let mut stats: FxHashMap<Pair, u32> = FxHashMap::default();
            for chunk_ids in &ids {
                get_stats(&chunk_ids, &mut stats);
            }
            if stats.len() == 0 {
                break;
            }

            let pair = stats
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(pair, _)| *pair);
            let idx: u32 = 256 + i;

            let best_pair = match pair {
                Some(p) => p,
                None => break,
            };

            ids = ids
                .into_iter()
                .map(|chunk_ids| merge(&chunk_ids, best_pair, idx))
                .collect();

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
                    pair,
                    idx,
                    vocab.get(&idx),
                    stats.get(&(pair.unwrap()))
                );
            }
        }
        vocab.insert(self.eot_id.unwrap(), EOT_STR.as_bytes().to_vec());
        self.merges = merges;
        self.vocab = vocab;
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
                ids.extend(self.encode_chunk(chunk));
            }
        }
        Ok(ids)
    }

    pub fn save(&self, file_prefix: &str) -> io::Result<()> {
        let mut out = String::from("rustbpe v1\n");
        for (&(a, b), &idx) in &self.merges {
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
        self.merges.clear();
        for line in lines {
            let nums: Vec<u32> = line
                .split_whitespace()
                .map(|s| s.parse())
                .collect::<Result<_, _>>()
                .map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
            self.merges.insert((nums[0], nums[1]), nums[2]);
        }

        // The model file stores merges only, so the reserved <|endoftext|> id and
        // its bytes are not in it. Recover the id as one past the highest merge
        // and re-attach the token, otherwise encode/decode of <|endoftext|>
        // breaks on a freshly loaded tokenizer.
        self.eot_id = self.merges.values().max().map(|&idx| idx + 1);
        self.rebuild_vocab();
        if let Some(eot_id) = self.eot_id {
            self.vocab.insert(eot_id, EOT_STR.as_bytes().to_vec());
        }
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
        let mut counts: FxHashMap<Pair, u32> = FxHashMap::default();
        while ids.len() >= 2 {
            counts.clear();
            get_stats(&ids, &mut counts);
            let pair = *counts
                .keys()
                .min_by_key(|p| self.merges.get(p).copied().unwrap_or(u32::MAX))
                .unwrap();

            let Some(&idx) = self.merges.get(&pair) else {
                break;
            };
            ids = merge(&ids, pair, idx);
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

fn get_stats(ids: &[u32], counts: &mut FxHashMap<Pair, u32>) {
    for window in ids.windows(2) {
        let pair: Pair = (window[0], window[1]);
        *counts.entry(pair).or_insert(0) += 1;
    }
}

fn merge(ids: &[u32], pair: Pair, idx: u32) -> Vec<u32> {
    let mut newids = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == pair.0 && i + 1 < ids.len() && ids[i + 1] == pair.1 {
            newids.push(idx);
            i += 2;
        } else {
            newids.push(ids[i]);
            i += 1
        }
    }
    newids
}

#[pymodule]
fn bpe_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RegexTokenizer>()?;
    Ok(())
}
