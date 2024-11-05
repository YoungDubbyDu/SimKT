# SimKT: Similarity-enhanced Question Embedding method for Knowledge Tracing
This repository contains the code for **SimKT** (Similarity-enhanced Question Embedding method for Knowledge Tracing), as described in the paper *"Question Embedding on Weighted Heterogeneous Information Network for Knowledge Tracing"* (accepted by ACM TKDD).

## Overview

SimKT is a knowledge tracing model that leverages similarity-enhanced question embeddings to improve prediction accuracy for student performance on exercises. The primary components and folders in this repository are described below.

## Repository Structure

- **`pre_emb`**: Contains code and pre-trained embeddings generated from the random walk process. These embeddings represent the core idea of the model and are essential for understanding how question embeddings are constructed.
- **`data`**: Stores the processed ASSIST09 dataset. Other datasets used in the paper can be downloaded separately (refer to the paper's link for additional dataset sources).
- **`SimKT_code`**: Contains the knowledge tracing code for predicting student performance, generating the final results presented in the paper.

## How to Run

To reproduce the results for the ASSIST09 dataset, execute the following command:

```bash
python main_09.py
```

## Requirements

> **Note**: `requirements.txt` should list all dependencies. If this file is not available, please include relevant libraries (such as PyTorch, DGL, or any other specific libraries used in your code).

## Citation

If you find this work helpful or use it in your research, please consider citing our paper:

> *Jianwen Sun, Shangheng Du, Jianpeng Zhou, Xin Yuan, Xiaoxuan Shen, and Ruxia Liang. 2024. Question Embedding on Weighted Heterogeneous Information Network for Knowledge Tracing. ACM Trans. Knowl. Discov. Data Just Accepted (November 2024). https://doi.org/10.1145/3703158*
