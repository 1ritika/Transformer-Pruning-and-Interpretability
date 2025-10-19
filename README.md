# 🤖 Transformer Pruning and Interpretability — T5 Summarization

This project explores **evaluation and interpretability of Transformer models** using the **T5 architecture** for text summarization.  
It involves computing evaluation metrics from scratch and performing **attention-head pruning** across encoder, decoder, and cross-attention layers to study redundancy and interpretability.

---

## 🚀 Features
- Implemented **ROUGE** and **BLEU** metric computation from scratch to evaluate summarization quality.  
- Conducted **attention-head pruning** at varying sparsity levels (0–100%) for:  
  - Encoder self-attention  
  - Decoder self-attention  
  - Cross-attention  
- Quantitatively analyzed performance degradation to identify redundant heads and interpret model structure.

---

## 📈 Results
| Layer Type | BLEU @ 0% | BLEU @ 20% | BLEU @ 40% | BLEU @ 60% | BLEU @ 80% |
|-------------|-----------|------------|------------|------------|------------|
| **Encoder** | 0.384 | 0.375 | 0.364 | 0.309 | 0.112 |
| **Decoder** | 0.384 | 0.383 | 0.376 | 0.373 | 0.247 |
| **Cross-Attn** | 0.384 | **0.364** | 0.359 | 0.194 | 0.047 |

> BLEU decreased marginally (0.38 → 0.36 at 20% sparsity in cross-attention), confirming redundancy in several attention heads.

---

## 🧠 Key Insights
- Moderate pruning (≤20%) in cross-attention caused **minimal performance loss**, suggesting structural redundancy.  
- Attention pruning acts as an **interpretability probe**, revealing which heads are critical for content alignment.  
- Implementing metrics manually reinforced understanding of **text evaluation pipelines** and Transformer outputs.

---

## 🛠️ Requirements
```bash
pip install torch transformers datasets nltk numpy matplotlib
