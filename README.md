# Invoice Information Extraction: Comparative Analysis of Transformer Models

## Overview
A research-driven benchmarking study comparing two state-of-the-art transformer architectures — LayoutLMv3 and Donut — for structured field extraction from scanned invoice documents. The study evaluates both models under two controlled experimental conditions: template-aware (familiar layouts) and template-agnostic (unseen layouts), addressing a core real-world deployment challenge in enterprise document AI.

> **Note:** This was a 3-person group project.

## Research Question
How well do layout-sensitive vs. OCR-free transformer models adapt to invoice formats they have never seen before — and which architecture is more suitable for real-world enterprise deployment where vendor invoice formats vary widely?

## Models Compared

### LayoutLMv3 (Template-Aware)
- Multimodal transformer processing three input types simultaneously: raw document image, OCR-extracted text tokens (via Tesseract), and 2D bounding boxes
- Uses BERT-style token embeddings fused with CNN visual embeddings and spatial position embeddings
- Fine-tuned using HuggingFace Trainer API, AdamW optimizer (lr=5e-5), cross-entropy loss, early stopping on validation F1
- Excels on familiar, structured layouts — strong spatial priors

### Donut (Template-Agnostic) — My Implementation
- OCR-free encoder-decoder model (ViT encoder + autoregressive decoder) that processes raw document images directly into structured JSON output
- Eliminates OCR preprocessing entirely — no bounding boxes, no text tokens — making it robust to scan quality, font variation, and layout shifts
- Fine-tuned using HuggingFace Seq2SeqTrainer on `naver-clova-ai/donut-base`
- Preprocessing: images resized to standard resolution, JSON annotations flattened and tokenized with structured task prompts
- Training: batch size=1, lr=5e-5, 8-10 minutes on Google Colab

## Experimental Design

| Strategy | Description | Training | Testing |
|---|---|---|---|
| Strategy 1 (Template-Aware) | Models trained and tested on familiar layouts | 7,500 images | 1,250 images |
| Strategy 2 (Template-Agnostic) | Models tested on completely unseen layouts | 500 pairs | 200 pairs |

**Extracted Fields:** InvoiceNumber, InvoiceDate, DueDate, VendorName, TotalAmount

## Evaluation Metrics
- **Token-Level Precision, Recall, F1** — correctness of token-level field classification
- **Exact Match Accuracy (EMA)** — full structured JSON prediction correctness
- **Generalization Delta** — F1 drop from Strategy 1 to Strategy 2, measuring layout sensitivity

## Results

| Epoch | Training Loss | Validation Loss | Accuracy | F1 |
|---|---|---|---|---|
| 1 | 0.0000 | 0.000019 | 1.0000 | 1.0000 |
| 2 | 1.9609 | 1.9487 | 0.2313 | 0.0869 |
| 3 | 1.9657 | 2.1818 | 0.2313 | 0.0869 |

**Key Finding:** LayoutLMv3 achieved perfect F1 under template-aware conditions but showed significant sensitivity to layout shifts. Donut demonstrated stronger generalization to unseen invoice formats, making it better suited for real-world deployment where vendor layouts vary unpredictably.

**Trade-off Summary:**
- Use **LayoutLMv3** when invoice templates are standardized and controlled (e.g. internal enterprise systems)
- Use **Donut** when invoice formats are diverse and unpredictable (e.g. multi-vendor procurement pipelines)

## Limitations
- Perfect F1 in epoch 1 may reflect data overlap or reduced test complexity rather than true generalization — results should be interpreted with caution
- Strategy 2 was only partially executed due to compute constraints (500 training pairs vs. planned 3,000)
- Dataset of 10,000 invoices was a scaled-down version of the originally planned 30,000
- Planned LangChain + Ollama2 chatbot frontend was not completed within the project timeframe

## Tools and Technologies
- **Language:** Python
- **Frameworks:** HuggingFace Transformers, HuggingFace Datasets, Seq2SeqTrainer, Trainer API
- **Models:** LayoutLMv3, Donut (naver-clova-ai/donut-base)
- **OCR:** Tesseract (for LayoutLMv3 preprocessing)
- **Environment:** Google Colab

## Repository Structure
```
invoice-extraction-transformer-comparison/
├── invoice_extraction_report.pdf           # Full project report
└── README.md
```

## Author
**Vaishnavi Tiwari**
M.S. Data Science, DePaul University  
[LinkedIn](https://www.linkedin.com/in/vaishnavi-tiwari-1a18971ab) · [GitHub](https://github.com/tiwari-vaish)
