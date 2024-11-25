# Similarity Bias in Fault Diagnosis Benchmarks

This repository contains the benchmarks and implementations for the study **"The similarity bias problem: what it is and how it impacts vibration-based intelligent fault diagnosis"**. It provides code, methods, and results for evaluating the impact of similarity bias in machine learning models applied to vibration data for intelligent fault diagnosis.

## Abstract (Paper Overview)

The study investigates the **similarity bias** problem in vibration-based machine learning research. It evaluates performance differences between conventional and deep learning models under different cross-validation strategies to address the bias issue. Key contributions include:

- Raising awareness about similarity bias.
- Designing experiments on public datasets to mitigate its effects.
- Providing benchmark methods and results for future research.
- Recommending best practices for vibration-based datasets in machine learning.

## Benchmarks and Methods

This repository contains the implementation and evaluation of three classification methods:

- **Nearest Neighbor**
- **Random Forest**
- **XResNet18-1D (Deep Convolutional Neural Network)**

### Datasets
The classifiers were trained and tested on the following datasets:
- **CWRU** (12 kHz sample rate)
- **CWRU** (48 kHz sample rate)
- **MFPT** (97.656 kHz sample rate)
- **PU**
- **IMS**
- **UOC**

### Cross-Validation Strategies
Each experiment employed three cross-validation strategies to address similarity bias:

1. **`bias_usual`**: Random split, the default approach dividing the dataset randomly into folds.
2. **`unbiased`**: Predefined folds based on dataset characteristics (e.g., grouping by operational conditions like load levels).
3. **`bias_mirrored`**: Random split while replicating the class distribution of the `unbiased` division.

### Multi-Round Cross-Validation
For datasets supporting multiple rounds of cross-validation, approximately 30 observations were generated for robust statistical hypothesis testing. Documentation for these divisions is available in `docs/multi_rounds/`.

## Repository Structure
- `vibnet/`: Code for training and testing models across datasets and cross-validation strategies.
- `docs/`: Documentation on data splits, methodology, and multi-round cross-validation.
- `cfgs/`: Configuration files for experiments.
- `tests/`: Tests used to validate modifications at the benchmark

## Experiment Designs

More details about the experimental designs and methodologies can be found in the original paper:
"The similarity bias problem: what it is and how it impacts vibration-based intelligent fault diagnosis."

## Authors
This repository is maintained by the authors of the paper. For inquiries or collaboration, please contact the primary author(s) listed in the paper.
