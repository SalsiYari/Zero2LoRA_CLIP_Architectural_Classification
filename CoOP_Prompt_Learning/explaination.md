# Technical and Comparative Analysis: CoOp vs. Weighted Prompt Ensemble with CLIP

This document analyzes the differences between the two Prompt Learning methodologies applied to the CLIP 
---

## 1. CoOp (Context Optimization)
*Reference in the repository:* `models/coop.py` (`PromptLearner`) and `engine/trainer_coop.py`

### Logic and Concept
The **CoOp (Context Optimization)** technique shifts the prompting paradigm from the discrete domain (natural language words) to the **continuous domain** (embedding vector space). 

Instead of defining a fixed phrase like *"a photo of a [CLASS] architecture"*, CoOp introduces $M$ learnable tokens (called **soft tokens** or *continuous context*). Each token is a continuous vector in $\mathbb{R}^D$, where $D$ is the textual embedding dimension of CLIP (e.g., $D=512$ for ViT-B/32).

### Mechanical and Mathematical Functioning
1. **Initialization**: A tensor of learnable parameters `self.ctx` is instantiated, drawn from a normal distribution with a predefined standard deviation (default: `0.02`).
2. **Prompt Construction**: The `PromptLearner` module physically replaces the embedding vectors associated with the dummy tokens ("X"), placing them immediately after the beginning-of-sequence (BOS) token and before the tokens associated with the real class name:
   $$\text{Prompt} = [\text{BOS}], v_1, v_2, \dots, v_M, [\text{CLASS}], [\text{EOT}]$$
3. **Point of Intervention**: The optimization intervenes at the **input** of the Text Encoder. Since the `self.ctx` parameters change at each iteration, the Text Transformer **cannot pre-compute the features**. During each forward pass, the entire Text Encoder must process the embedding sequence to generate the new textual features conditioned by the optimized context.

---

## 2. Weighted Prompt Ensemble
*Reference in the repository:* `models/prompt_ensemble.py` (`PromptEnsemble`) and `engine/trainer_ensemble.py`

### Logic and Concept
Unlike CoOp, this variant works entirely in the **discrete domain** at the input level, leveraging a fixed collection of $T$ templates written by human experts (e.g., the 15 templates defined in `configs/default.py`). The learning does not occur by modifying the input words or embeddings, but rather by optimizing the **relevance (weight) of each template** for the specific task.

### Mechanical and Mathematical Functioning
1. **Frozen Gradient Extraction**: At the beginning of the training, the `_build_text_features()` function computes the Text Encoder output embeddings for each combination of class and template under the `@torch.no_grad()` context. This generates a static tensor of normalized textual features with dimensions $[T, C, D]$, where $T$ is the number of templates and $C$ is the number of classes.
2. **Linear Aggregation via Softmax**: During training, the network optimizes a vector of static logits (`template_logits`). The effective weights of the ensemble are normalized using a Softmax function to ensure their sum equals 1:
   $$w_i = \frac{e^{\theta_i}}{\sum_{j=1}^T e^{\theta_j}}$$
   The final textual vectors are computed as a weighted linear combination:
   $$\mathbf{f}_c = \text{Normalize}\left( \sum_{i=1}^T w_i \cdot \mathbf{f}_{i, c} \right)$$
3. **Point of Intervention**: The optimization occurs exclusively at the **output** of the Text Encoder. The textual encoder is executed only once at initialization and then frozen.

---

## 3. Comparative Analysis of Resources and Behavior

### Computational Cost and Memory
* **`main_coop.py`**: It is computationally intensive. Since the input to the Text Encoder continuously varies, submitting a batch requires computing the gradient through the entire architecture of the textual Transformer, in addition to the Vision Encoder for the images. It requires higher VRAM allocation (mitigated in the code by using Mixed Precision `torch.autocast`).
* **`main_ensemble.py`**: It is extremely lightweight. Because the textual features are cached (pre-computed), the training loop only needs to compute the image features and solve a linear algebra operation (matrix multiplication and convex combination). It has very few parameters to optimize (equal to the number of templates $T$, or $C \times T$ in *per-class* mode).

### Flexibility: Shared vs. Per-Class
Both modules implement the dichotomy between global optimization (*shared*) and local optimization (*per-class*), faithfully mapped via the `--mode` arguments in the code:

| Methodology | `shared` Configuration | `per_class` Configuration |
| :--- | :--- | :--- |
| **CoOp** | A single continuous vector context $[M, D]$ is learned and shared to discriminate all classes in the dataset. | $C$ independent contexts $[C, M, D]$ are allocated. Each class develops its own soft-token geometry. |
| **Prompt Ensemble** | A global vector of weights $[T]$ is learned. A template (e.g., *"the {} architectural style"*) assumes the same importance across all classes. | A weight matrix $[C, T]$ is allocated. The system learns that certain templates work better for specific classes and worse for others. |

---

## 4. Flow Structure in the Modular Laboratory

The code refactoring cleanly isolates these two logics, allowing for atomic maintenance:
1. The configuration files (`configs/default.py`) centrally expose the discrete templates and data paths.
2. The `main_coop.py` and `main_ensemble.py` files act exclusively as high-level orchestrators, importing the models from `models/` and the calculation loops from `engine/`.
3. Mathematical reproducibility is guaranteed by the shared `utils/core_utils.py` module, which freezes CuDNN determinism and pseudo-random number generators based on a seed.
