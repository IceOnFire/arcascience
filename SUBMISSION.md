# Submission notes

**Matteo Antony Mistretta**: AI Engineer Technical Test, October 2026

## Before you read

I completed this test with the help of an AI coding assistant. The instructions didn't mention AI tools either way, so I'm stating it upfront. I used it the way I described in our first interview: as a multiplier, with my judgement last. I directed the work, questioned every result, and asked where every number came from; several corrections in the notebooks came out of that process. I made sure I understand and can defend each step.

Data science is outside my usual field, so this took longer than the suggested 3–4 hours: most of a day, about 8 hours. The extra time went into understanding and verifying the work, not into producing more of it. The commit history shows the actual sequence.

## Where to find what

The notebooks are meant to be read in this order. Each one runs on its own and reads only files in this repository: **no API key is needed to re-run them**.

| Notebook | What it covers | Headline result |
|---|---|---|
| [`part1a.ipynb`](part1a.ipynb) | Data exploration | 19% of rows repeat an anchor, concentrated in the small categories; `is_coa_tool` leaks the label; some labels are noisy |
| [`part1b.ipynb`](part1b.ipynb) | Optimizing the embedder | **914 s → 11.4 s (80x)**; each optimization measured separately; why batched output is not bit-identical, and an `--exact` mode that is |
| [`part1c.ipynb`](part1c.ipynb) | Embedding visualization | Strong local structure, weak global clustering: a 10-nearest-neighbours vote already reaches 70% |
| [`part2.ipynb`](part2.ipynb) | Classifier | **73.9% accuracy / 0.716 macro-F1 on unseen endpoint texts** (5 splits); a careless split would have reported 80.7%; 80% of errors involve General Case |
| [`bonus_b.ipynb`](bonus_b.ipynb) | Bonus B: triplet validation | 99.8% of triplets satisfied (MiniLM: 85.6%); the 7 violations are corrupted rows; triplet quality does not predict classification quality |
| [`appendix_classic_baseline.ipynb`](appendix_classic_baseline.ipynb) | Extra: a classic baseline | TF-IDF reaches 68.8%; TF-IDF + embeddings in a plain logistic regression matches the neural network |

**About the appendix.** It's not a second bonus. I was curious how the techniques that were standard when I graduated in 2006 would do against modern embeddings: no embeddings, just counting words. The honest answer is "surprisingly well, but not as well", and the combination of the two turned out to be the simplest model that matches the neural network. I think that result is worth knowing before putting a model in production.

## The main decision you should know about

Identical anchor texts appear in several rows (Part 1A). I split the data **by text**, so the same text never appears in both training and test. This gives lower numbers than a plain random split (73.9% vs 80.7%, Part 2, Section 10), but they measure what matters: performance on endpoints the model has never seen.

## From notebook to production

Not implemented, but this is how I would take the classifier further, given the findings above:

- **Serving:** a FastAPI endpoint: text → embedding API → classifier → category and confidence. Load the model once at startup and batch requests, as in `optimized_embedder.py`.
- **Versioning:** pin the embedding model's version together with the classifier's weights. The classifier depends on that exact embedding space, and Part 1B shows that even batching changes the embeddings slightly.
- **Human (or LLM) in the loop:** predictions above 0.9 confidence, 61% of the test set, are 89% correct (Part 2, Section 9). Route the rest to review.
- **Monitoring:** track the confidence distribution and the share of each predicted category over time. A drop in confidence or a shift in category shares is the first sign of drift. Periodically have experts label a sample, to measure accuracy on live data.
- **Simplest first:** before deploying the neural network, I would compare it with the logistic regression on TF-IDF + embeddings (appendix): same accuracy, simpler to run and explain.

## Running it

- **Python 3.14** (the `.python-version` file says 3.12; I used 3.14, the version available on my machine). Install with `pip install -r requirements.txt`.
- **First run downloads** two small public models from Hugging Face (a tokenizer in Part 1A, MiniLM in Bonus B), pinned to fixed versions.
- **Re-generating the measurements** (embeddings, timings) needs `GPU_MODEL_SERVER_API_KEY`; the exact commands are at the top of `part1b.ipynb`.

## Repository layout

```
optimized_embedder.py   Part 1B script (default, --include-triplets, --exact)
benchmarks/             ablation and API profiling scripts behind Part 1B
logs/                   timing logs and benchmark results read by Part 1B
embeddings/             anchor/positive/negative embeddings, plus naive/ and exact/ reference outputs
models/                 the trained Part 2 classifier
figures/                every chart, also saved as PNG
```
