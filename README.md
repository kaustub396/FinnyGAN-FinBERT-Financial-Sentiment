# FinnyGAN: GAN-Augmented FinBERT for Financial Sentiment Analysis

A research-oriented NLP system that improves financial sentiment classification by augmenting FinBERT with GAN-generated synthetic embeddings.

Domain: FinTech / Quant NLP  
Techniques: Transformers, GANs, Representation Learning  
Dataset: FIQA (Kaggle)

Use-Cases:

- Algorithmic trading sentiment signals
- Earnings call sentiment analysis
- News-driven risk assessment
- Portfolio exposure analysis
- Market regime detection

- ## 🧠 System Architecture

![FinnyGAN Architecture](Finny/finny_GAN.pdf)


This system can be integrated into trading pipelines as a sentiment alpha feature.
Instead of generating text, FinnyGAN:

Extracts [CLS] embeddings from FinBERT

Trains a GAN in embedding space

Generates synthetic financial embeddings

Fine-tunes FinBERT using real + synthetic embeddings

This avoids grammatical issues of text GANs while focusing on semantic diversity.

Base model: FinBERT (yiyanghkust/finbert-tone)

GAN: MLP Generator + Discriminator (embedding space)

Synthetic samples: 1000

Training strategy:

Encoder freezing (warm-up)

Joint optimization of real + synthetic losses

Evaluation metric: Accuracy, Precision, Recall, F1 (weighted)

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| FinBERT (baseline) | 0.7606 | 0.7608 | 0.7606 | 0.7605 |
| RoBERTa | 0.5897 | 0.3520 | 0.5897 | 0.4408 |
| FinRoBERTa | 0.7427 | 0.7235 | 0.7527 | 0.7516 |
| **FinnyGAN (stable)** | **0.7776** | **0.7872** | **0.7784** | **0.7800** |


⚠️ Limitations (Important)

Synthetic embeddings are weakly labeled (random sentiment assignment)

GAN is unconditional (not class-aware)

Performance variance is expected due to GAN instability

These are known research limitations, not implementation bugs.

🚀 Future Work

Conditional GAN for label-aware embedding synthesis

Pseudo-labeling of synthetic samples

Diffusion-based embedding generation

Joint sentiment + market-impact modeling
