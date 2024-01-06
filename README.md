# Prediction of Genetic Mutations in Pediatric Brain Tumors Pipeline


## Overview

This repository hosts the codebase for a deep learning pipeline, specifically designed to predict genetic mutation drivers in Pediatric Low-Grade Gliomas (LGGs) using whole slide images (WSIs). Addressing the challenge of pediatric brain tumors, this project emphasizes a computational approach as a viable alternative to genetic sequencing, especially crucial for rapid, cost-effective diagnostics in various healthcare settings. Please note that this repository only includes the codebase due to privacy and confidentiality concerns.


## Pipeline Components

* patch_extraction.py: This script is central to preprocessing, transforming gigapixel WSIs into manageable patches. It's configurable for different image sizes and magnification levels, adapting to diverse datasets (`config.py`).

* classification_patches.py: Implements a convolutional neural network for initial patch classification. It provides dual functionality - categorizing patches based on slide-level labels and extracting rich feature vectors for each patch, setting the stage for deeper analysis.

* visualize_predictions.py: A tool to visually represent model predictions. It generates intuitive heatmaps, offering a clear visual interpretation of patch-level predictions.

* aggregation_patch_features.py: Includes several approaches for aggregating patch-level data. This script aggregates diverse patch-level feature vectors into a singular, comprehensive slide-level representation used for downstream analysis.

* aggregation_multimodal_features.py: This script extends the pipeline's capability by integrating slide-level features with clinical data, creating a multifaceted representation of each case.

* classification_wsi.py: The culmination of the pipeline. It employs the aggregated feature vectors for final classification of WSIs, encapsulating the entire predictive power of the model.


## Research Context & Impact

This project leverages deep learning to improve diagnostic accuracy in pediatric oncology, focusing on pediatric Low-Grade Gliomas (LGGs). The pipeline adeptly identifies genetic mutations using digital pathology images, focusing on the three most common genetic drivers in pediatric LGGs. This approach aligns with existing research and emphasizes accessibility, aiming to benefit regions where advanced genetic sequencing is less accessible. It
promises significant contributions to precision medicine, representing a potential tool for accurate and timely diagnosis in pediatric brain tumors, especially in underserved areas.