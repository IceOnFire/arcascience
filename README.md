# AI/NLP Engineer Technical Assessment
**Duration:** 3-4 hours (3 hours required + optional 30-60 min bonus)
**Focus:** Biomedical NLP, Deep Learning, Embeddings

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# You may need to install additional packages for your analysis:
# pip install torch scikit-learn matplotlib seaborn umap-learn
```

### 2. Configure API Access

You should have received an API key via email. Set it as an environment variable:

```bash
export GPU_MODEL_SERVER_API_KEY="your-api-key-here"
```

### 3. Verify Setup

```bash
# Test that you can import basic packages
python -c "import pandas, numpy; print('Setup successful!')"

# Test API access (optional)
python naive_embedder.py --csv data/endpoints_data.csv --dimension 768 --output test.npy
```

### 4. Launch Jupyter

```bash
jupyter lab
# Or: jupyter notebook
```

## Getting Started

1. Explore the dataset in `data/endpoints_data.csv`
2. Review the provided `naive_embedder.py`
3. Start with Part 1A: Data Exploration
4. Work through the required parts sequentially

---

## Overview

You will work with a fine-tuned biomedical embedding model designed to generate semantic representations of clinical trial endpoints. Your task is to build and evaluate a classification model using these embeddings, analyze performance trade-offs, and demonstrate your understanding of modern NLP techniques.

You'll complete this assessment in a **Jupyter notebook** where you'll document your analysis, code, and findings. The notebook serves as both your workspace and your submission document, combining code execution with written explanations.

**Important - Notebook Structure:** As you work through each part, use markdown cells to explain your reasoning, interpret results, and document decisions **as you go**. Don't just write code and save all explanations for the end. After each analysis or visualization, add a markdown cell explaining what you found and why it matters. This narrative style demonstrates your thought process and makes your work much easier to evaluate.

---

## Background: The Embedding Model

### Purpose
The embedding model transforms clinical endpoint text (e.g., "Blood pressure measurement", "Adverse event frequency") into dense vector representations that capture semantic similarity. The model was trained specifically on biomedical/clinical data to understand relationships between different types of clinical outcomes.

### Key Capabilities
- **Input:** Clinical endpoint text strings
- **Output:** 768-dimensional dense embeddings with Matryoshka capability
- **[Matryoshka Embeddings](https://arxiv.org/abs/2205.13147):** The model supports truncation to smaller dimensions [768, 512, 256, 128, 64] while maintaining strong performance
- **Base Model:** Fine-tuned from PubMedBERT ([microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext))

### Endpoint Categories
The model is trained to understand 9 types of clinical endpoints:

1. **General Case** - Standard clinical measurements and outcomes
2. **COA Tool** - Clinical Outcome Assessment instruments (e.g., pain scales, quality of life questionnaires)
3. **Pharmacokinetic** - Drug absorption, distribution, metabolism measures (e.g., Cmax, AUC)
4. **Adverse Event** - Safety and toxicity outcomes
5. **Complications** - Disease complications and clinical events
6. **Immunogenicity** - Immune response measurements
7. **Mortality** - Death and survival outcomes
8. **Cost** - Economic and resource utilization measures
9. **Demographic** - Population characteristics

### Data Characteristics
- Training data includes both real clinical trial endpoints and synthetically generated examples
- Special augmentation was applied to COA Tool endpoints to improve performance on clinical assessment instruments
- Model trained using triplet loss with anchor-positive-negative samples to ensure semantically similar endpoints have closer embeddings

---

## Test Structure

This assessment has two sections: **Required Parts** (must complete) and **Bonus Options** (choose one if time permits).

### REQUIRED PARTS (~2.75 - 3 hours)

Complete all four required parts before proceeding to bonus options.

---

### Part 1A: Initial Exploration (25 minutes)
**Goal:** Understand the dataset characteristics and patterns relevant to classification

The dataset contains triplet structure (anchor, positive, negative) from the embedding model's training process, but for this classification task, **focus your exploratory analysis on the anchor texts and their categories** (stored in the `case` column), these are what you'll be classifying in later parts.

Conduct comprehensive exploratory data analysis examining the dataset from multiple angles. Investigate the distribution of the 9 endpoint categories and assess class balance, understanding imbalances will inform your modeling approach in Part 2. Analyze text characteristics including length patterns (characters, words, tokens if you choose), vocabulary differences across categories, and how clinical terminology manifests in the endpoints. Explore the metadata flags (is_coa_tool, is_synthetic, is_root_pair_similar) to understand data composition and potential biases. Check for data quality issues like missing values or duplicates, and identify patterns that might affect downstream classification performance.

**Deliverable:**
At the end of this section write a concise summary paragraph highlighting 5-7 key insights from your exploration. Discuss patterns you observed, potential challenges for classification, class imbalance considerations, and notable characteristics that stood out. Connect your observations to implications for later modeling work.

---

### Part 1B: Code Optimization Challenge (30-35 minutes)
**Goal:** Optimize the provided embedding generation script for significant speedup

Begin by reading the provided `naive_embedder.py` to understand the baseline implementation and identify optimization opportunities.

**Background:**
We've provided `naive_embedder.py`, a working but inefficient script that generates embeddings for all endpoints. While functional, it has multiple performance bottlenecks that make it impractical for production use. Your optimized version will be used for all subsequent parts, so this section is critical to complete well.

By default, generate embeddings for anchor texts only, saved as `embeddings/anchor.npy`. This is sufficient for Parts 1C and 2. For the optional Triplet Validation bonus (Option B), you can add an `--include-triplets` flag to generate embeddings for all three components (hint: consider using argparse for command-line argument handling).

**Your Task:**
Run the baseline script and measure its runtime to establish a reference point. Analyze the code to identify what's making it slow, think about where time is being spent in the naive implementation. The biggest performance gains often come from reducing network overhead and improving data processing patterns. Consider how APIs typically perform best when handling requests, and whether you're doing any unnecessary work. Error handling and resilience are also important for production systems that need to handle transient failures gracefully.

Create `optimized_embedder.py` in your submission root as a separate file from the naive version. Your optimized script must produce identical embeddings to verify correctness. Verify manually by comparing output shapes and spot-checking values. Aim for at least 5-10x speedup through practical improvements that you can clearly explain. (Gold standard solutions typically achieve 70x+ speedup.)

Document timing comparison in two places: first, your `optimized_embedder.py` should print timing statistics when executed; second, create a notebook cell showing the before/after measurements with calculated speedup. In your notebook, include a dedicated section explaining the bottlenecks you identified, optimizations you implemented, and why each change improved performance. Connect speedup results to specific optimizations.

Tip: You can start working on the optimization while waiting of the results of `naive_embedder.py` to complete, but make sure to run it fully to get accurate timing.

**Deliverable:**
- `optimized_embedder.py` with timing output
- Notebook cell documenting timing comparison (naive baseline vs optimized)
- Notebook section explaining bottlenecks, optimizations, and speedup achieved

**Evaluation Criteria:**
Speedup achieved (5x = good, 10x+ = excellent), code quality and readability, correct identification of bottlenecks, and practical well-justified optimizations.

---

### Part 1C: Embedding Visualization (20 minutes)
**Goal:** Assess embedding space quality through multiple complementary visualizations

Use different visualization strategies to understand how well the embeddings capture category distinctions. Common techniques include dimensionality reduction (t-SNE, UMAP, PCA) to project the high-dimensional space into 2D for visual inspection, per-category analyses showing how tightly each of the 9 endpoint types clusters, or quantitative comparisons like distances between category centroids versus spread within categories. Choose approaches that help you build intuition about embedding quality and predict classification challenges.

Beyond visual inspection, consider measuring embedding quality quantitatively. How far apart are category centers in the original 768-dimensional space? Do embeddings within the same category cluster tightly compared to distances between categories? These metrics can validate what you observe visually and quantify separation quality.

**Deliverable:**
Create at least one substantive visualization of the embedding space. This could be a single comprehensive figure with multiple subplots (e.g., t-SNE projection + per-category analysis), or multiple complementary visualizations (e.g., UMAP + quantitative metrics). Focus on depth of insight, your visualization(s) should combine different perspectives such as visual projections, quantitative methods, or per-category analyses. Document your key observations about category separation, overlap patterns, outliers, and what these patterns suggest for downstream classification difficulty.

---

### Part 2: Classification Model (90-100 minutes)
**Goal:** Build and train a neural network classifier on frozen embeddings.
Input: 768-dim embeddings
Output: 9 endpoint categories.

**Note:** Use the embeddings generated from your optimized script in Part 1B. (`embeddings/anchor.npy`)

**Overview:**
Build a neural network that takes the pre-computed 768-dimensional embeddings as input and predicts one of the 9 endpoint categories (from the `case` column). Since the embeddings are already semantically meaningful, a relatively simple feedforward architecture should work well.

**Data Preparation:**
Split your data into train, validation, and test sets. Stratified splitting is particularly important here given that the dataset has some class imbalance. The validation set will help you monitor overfitting during training and tune hyperparameters without touching your test set.

**Model Implementation:**
Design a neural network classifier using PyTorch. As a starting point, consider a feedforward architecture with 2-3 fully connected layers, the embeddings are already semantically rich, so elaborate architectures aren't necessary. You're welcome to experiment with different designs, but focus on proper regularization and training practices rather than architectural complexity. Think about what techniques help prevent overfitting when working with limited training data, regularization methods like dropout or weight decay can be valuable. Consider also whether the class imbalance you observed in Part 1A might affect training dynamics.

**Training:**
Implement a proper training loop with validation monitoring. Choose an appropriate loss function and optimizer for multi-class classification. Training time is typically **2-4 minutes on CPU** (or 20-60 seconds on GPU if available) for 30-40 epochs with early stopping, if it's taking significantly longer, consider simplifying your architecture or reducing the number of epochs. Most of your time will be spent on implementation and analysis, not waiting for training.

**Evaluation:**
Evaluate your model on the held-out test set using multiple metrics. Accuracy alone can be misleading with imbalanced classes, so consider metrics that account for per-class performance. Create visualizations of your training process: loss and accuracy curves over epochs for train and validation sets to diagnose overfitting patterns, and a confusion matrix showing test set predictions to understand which categories are confused. Write a **brief interpretation (2-3 paragraphs)** identifying which categories perform best/worst and connecting your results back to what you observed in the embedding visualization (Part 1C).

**Note on Error Analysis:** Basic performance analysis is required here, identifying which categories are challenging and interpreting your confusion matrix. If you choose Bonus Option A later, you'll perform a significantly deeper investigation with manual example inspection, systematic error categorization, and detailed improvement proposals.

**Deliverable:**
Complete notebook section with:
- Clean, documented code for model training
- Training curves (loss and accuracy over epochs for train/val sets)
- Test set performance: accuracy, F1 scores, confusion matrix visualization, and other relevant metrics of your choice
- Plots/visualizations that help interpret model performance
- Written interpretation discussing model performance, potential improvements, and connection to embedding quality

**Expected Performance:**
Well-implemented models with proper training practices typically achieve test accuracy in the 75-85% range on this dataset. Performance above 85% suggests excellent work, while results below 70% indicate implementation issues worth investigating. Focus on demonstrating proper ML practices rather than chasing perfect accuracy.

---

## BONUS OPTIONS (~30-50 minutes)

**Instructions:** If you complete all required parts (1A, 1B, 1C, 2) with time remaining, choose **ONE** bonus option to demonstrate advanced skills. Quality over coverage, it's better to do one thoroughly than attempt multiple incompletely.

**Evaluation:** Each bonus option is worth +15 points and weighted equally. Choose the one that best showcases your strengths or interests.

---

### Bonus Option A: Deep Error Analysis (40-50 minutes)
**Skills demonstrated:** Critical thinking, domain understanding, systematic debugging

**Background:**
No model is perfect. Understanding failure modes is crucial for improving performance and knowing when to trust model predictions. Systematic error analysis reveals patterns that simple accuracy metrics miss.

**Important:** This bonus requires analysis **substantially beyond** the basic error interpretation in Part 2. You must manually inspect misclassified examples, categorize error types systematically, and propose concrete, actionable improvements. Simply expanding your Part 2 confusion matrix analysis is insufficient for full credit.

**Your Task:**
Perform a deep investigation into where and why your classifier fails. Go beyond the basic confusion matrix analysis from Part 2 by: (1) manually inspecting 10-15 misclassified examples and categorizing the error types (ambiguous cases, semantically similar categories, potential labeling errors, embedding quality issues), (2) identifying systematic patterns in which category pairs get confused and analyzing the root causes (data scarcity, semantic overlap, embedding limitations), and (3) connecting your findings back to the embedding visualization patterns from Part 1C.

Your analysis should reveal insights not visible in Part 2's metrics. Think critically about root causes and propose specific, feasible improvements rather than generic suggestions.

**Deliverable:**
- Per-category performance breakdown with insights about best/worst performers
- Confusion matrix analysis identifying systematic error patterns
- Manual error analysis with 10-15 examples and categorization
- 2-3 concrete improvement proposals with justification (e.g., data augmentation strategies, architecture changes, alternative training approaches)

---

### Bonus Option B: Triplet Validation Analysis (30-40 minutes)
**Skills demonstrated:** Understanding embedding training, evaluation methodology, comparative analysis

**Background:**
The embedding model was trained using triplet loss with (anchor, positive, negative) structure. Well-trained embeddings should satisfy: $cosine\_similarity(anchor, positive) > cosine\_similarity(anchor, negative)$. Measuring how often this constraint holds provides direct validation of embedding quality.

**Your Task:**
Validate the triplet structure in your dataset. You can use your fine-tuned embeddings (run `optimized_embedder.py --include-triplets` to generate embeddings for all three components). In addition, you can compare against a general-purpose model as a baseline. The sentence-transformers library with the `all-MiniLM-L6-v2` model provides quick comparison embeddings for this purpose.

Calculate what percentage of triplets satisfy the constraint overall and per-category. Are there categories where the constraint frequently fails? What does this tell you about embedding quality for those categories? If you compare fine-tuned vs general-purpose embeddings, what improvement does domain-specific training provide?

**Deliverable:**
- Triplet satisfaction rate (overall and per-category)
- Visualization showing similarity distributions for positive vs negative pairs
- If comparing models: improvement analysis from fine-tuning
- Brief interpretation: what do violations tell us about embedding quality? How does this correlate with classification performance from Part 2?

---

## API Information

**Endpoint:** `https://model-gpu-test.arcascience.ai/text-embedding`

**Authentication:**
The API requires authentication. All requests must include an `X-API-Key` header with your API key.

**Setting up your API key:**
1. You should have received an API key via email
2. Set it as an environment variable before running the code:
   ```bash
   export GPU_MODEL_SERVER_API_KEY="your-api-key-here"
   ```
3. The provided `naive_embedder.py` script reads this environment variable automatically

**Troubleshooting:**
- If you didn't receive an API key or the API is unreachable, notify us immediately

**Server Configuration:**
- Multi-GPU setup with load balancing
- Designed to handle concurrent requests efficiently
- Recommended batch size: 32-256 texts per request for optimal throughput

### Request Format

```python
POST https://model-gpu-test.arcascience.ai/text-embedding
Headers: X-API-Key: <your-api-key>
Content-Type: application/json

{
  "input": ["endpoint1", "endpoint2"],
  "normalize": true,
  "model": "arca-endpoint-embedder",
  "truncation_dimension": 768  # Optional: 768, 512, 256, 128, 64
}
```

### Response Format

```python
{
  "result": [[...], [...]],  # Embeddings for each input text
  "metadata": {
    "model_name": "arca-endpoint-embedder",
    "model_version": "...",
    "total_time": 0.0112,
    ...
  }
}
```

**API Usage:**
See `naive_embedder.py` for complete working examples with authentication.

---

## Submission Requirements

### 1. Jupyter Notebook(s)
Your notebook(s) are your primary deliverable and should contain all required parts (1A, 1C, 2) plus your chosen bonus option (if completed). You may organize your work in a single notebook or split it across multiple notebooks (e.g., `part1.ipynb` and `part2.ipynb`). Write as a narrative document with clear section headers, well-documented code, and markdown explanations throughout.

Ensure reproducibility by setting random seeds and documenting your environment. All analysis and interpretations should be in the notebook(s) - no separate report is needed.

### 2. Optimized Embedder Script
- `optimized_embedder.py` - Your improved version from Part 1B
- By default, generates embeddings for anchor texts only
- Saves to `embeddings/anchor.npy`
- Optional: Use `--include-triplets` flag for bonus triplet validation work (Bonus Option B)
- Should be standalone and runnable
- Document timing comparison in your notebook with before/after measurements
- Your script should also print timing statistics when executed

### 3. File Organization
Your submission directory structure:
```
technical_test/                 # Root directory (provided to you)
├── README.md                   # This file
├── part1.ipynb                 # Part 1 work (or use a single notebook)
├── part2.ipynb                 # Part 2 + bonus work (or combine with part1)
├── optimized_embedder.py       # Your optimized script (Part 1B)
├── naive_embedder.py           # Provided baseline (don't modify)
├── embeddings/
│   └── anchor.npy              # Required (generated by optimized_embedder.py)
│   ├── positive.npy            # Optional (for Bonus Option B)
│   └── negative.npy            # Optional (for Bonus Option B)
├── models/                     # Saved model weights (optional)
├── figures/                    # Additional visualizations (optional)
├── data/
│   └── endpoints_data.csv      # Provided dataset
└── requirements.txt            # Dependencies
```

**Note:** Notebook naming is flexible - use `part1.ipynb` + `part2.ipynb`, or a single `notebook.ipynb`, or your own naming scheme. The key is clear organization with all required parts present.

---

## Evaluation Rubric (100 points)

### Required Parts (85 points)

| Component | Points | Key Criteria |
|-----------|--------|-------------|
| **Part 1A: Data Exploration** | 10 | Dataset understanding, insightful observations, quality of summary |
| **Part 1B: Code Optimization** | 25 | Speedup achieved (10), Code quality (8), Documentation (4), Explanation (3) |
| **Part 1C: Visualization** | 10 | Quality of embedding space visualization, depth of insights |
| **Part 2: Classifier** | 40 | Implementation correctness (12), Training procedure (12), Evaluation & analysis (16) |

### Bonus Options (Choose ONE, +15 points)

| Option | Points | Key Criteria |
|--------|--------|-------------|
| **Option A: Error Analysis** | +15 | Depth of investigation, quality of examples, feasibility of proposals |
| **Option B: Triplet Validation** | +15 | Methodology, comparative analysis, interpretation quality |

**Total: 85 points (100 with bonus)**

**Note on Grading:** Partial credit is awarded for incomplete sections. For example, completing Part 1 and Part 2 without a bonus option still earns 85% of total points. We value quality of work over exhaustive completion.

---

## Tips for Success

**Time Management:**
Budget your time across the required parts first before considering bonus work. Part 1 (A+B+C combined) should take roughly 75-80 minutes total. Part 1B is especially critical since your optimized embeddings are needed for everything that follows. Allocate the bulk of your time to Part 2 (90-100 minutes), as it's the most substantial component. If you complete required parts in under 3 hours, choose one bonus that plays to your strengths.

**Code Optimization (Part 1B):**
Start by measuring the baseline to establish a reference point. The biggest performance gains usually come from the simplest changes, look for obvious inefficiencies before attempting complex optimizations. Make sure your optimized version produces identical embeddings to verify correctness. Practical improvements that you can clearly explain are more valuable than clever tricks.

**Model Building (Part 2):**
Focus on demonstrating solid ML practices rather than achieving perfect accuracy. Proper train/val/test splits with stratification, validation monitoring to prevent overfitting, and thoughtful evaluation are more important than elaborate architectures. Training should take 2-5 minutes on CPU (or under a minute on GPU), if it's taking significantly longer, consider simplifying your architecture or reducing epochs. Connect your results back to what you observed in the embedding visualization, does your model struggle with categories that showed overlap in 2D projections?

**Analysis and Communication:**
Don't just report numbers, interpret them and explain what they mean. Use markdown cells **after each analysis step** to document your observations, not just at the end of each part. For example, after creating a visualization, immediately add a markdown cell explaining what the plot reveals and why it matters. This narrative approach shows your thinking process and is far more valuable than dumping code and writing one long explanation at the end. Connect your findings to domain knowledge about clinical endpoints when possible, and be honest about limitations and uncertainties.

---

## Evaluation Criteria Summary

We evaluate candidates across multiple dimensions:

**Technical Execution:** Correct implementation of optimizations and neural networks, proper training procedures with validation, appropriate evaluation metrics, and reproducible code with documentation.

**Problem-Solving:** Ability to identify performance bottlenecks and implement practical improvements, design effective model architectures for the task, and conduct systematic analysis of results.

**Communication:** Clear code documentation, well-structured notebook narrative, effective visualizations, and insightful interpretations that go beyond just reporting numbers.

**Depth:** Quality of insights from data exploration, understanding of model behavior and failure modes (especially in bonus work), and ability to connect findings across different parts of the assessment.

The bonus options allow you to showcase different strengths, choose the one where you can demonstrate the most depth and insight.

---

## Questions?

If you encounter any blocking issues during the test, please reach out to us. Otherwise, document your assumptions and proceed. We value seeing your decision-making process and problem-solving approach.

Good luck! 🚀
