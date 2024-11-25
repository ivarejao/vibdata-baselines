### `docs/multi_round/CWRU_48.md`

# Multi-Round Cross-Validation: CWRU Dataset (48 kHz Sample Rate)

This document explains the multi-round cross-validation strategy applied to the CWRU dataset with a sample rate of 48 kHz. The experiment leverages the dataset's distinct load conditions to create diverse fold combinations for robust hypothesis testing.

---

## Methodology

The CWRU dataset comprises samples with three labels:
- **Outer Race Fault** (O)
- **Inner Race Fault** (I)
- **Roller Fault** (R)

Each label is recorded under **four different load conditions**:
- **0 horsepower (hp)**
- **1 horsepower (hp)**
- **2 horsepower (hp)**
- **3 horsepower (hp)**

For multi-round cross-validation, the dataset is divided into **four folds**, with each fold containing samples corresponding to a specific load condition. In subsequent rounds, samples from one load condition are swapped between folds, generating new combinations. 

This strategy ensures that the classifier evaluates its performance across different operational conditions, reducing potential similarity bias.

---

## Experimental Setup

- **Total rounds:** 8
- **Folds per round:** 4
- **Combination logic:** Each round redistributes label samples recorded under a single load condition across folds.
- **Example notation:** 
  - `O 0` represents **Outer Race Fault** samples recorded at **0 hp**.
  - `R 1` represents **Roller Fault** samples recorded at **1 hp**.
  - `I 3` represents **Inner Race Fault** samples recorded at **3 hp**.

---

## Round Configurations

### **Round 0**
- Fold 0: `O 0`, `R 0`, `I 0`
- Fold 1: `O 1`, `R 1`, `I 1`
- Fold 2: `O 2`, `R 2`, `I 2`
- Fold 3: `O 3`, `R 3`, `I 3`

### **Round 1**
- Fold 0: `O 3`, `R 2`, `I 3`
- Fold 1: `O 0`, `R 3`, `I 0`
- Fold 2: `O 1`, `R 0`, `I 1`
- Fold 3: `O 2`, `R 1`, `I 2`

### **Round 2**
- Fold 0: `O 2`, `R 3`, `I 1`
- Fold 1: `O 3`, `R 0`, `I 2`
- Fold 2: `O 0`, `R 1`, `I 3`
- Fold 3: `O 1`, `R 2`, `I 0`

### **Round 3**
- Fold 0: `O 1`, `R 3`, `I 3`
- Fold 1: `O 2`, `R 0`, `I 0`
- Fold 2: `O 3`, `R 1`, `I 1`
- Fold 3: `O 0`, `R 2`, `I 2`

### **Round 4**
- Fold 0: `O 2`, `R 1`, `I 1`
- Fold 1: `O 3`, `R 2`, `I 2`
- Fold 2: `O 0`, `R 3`, `I 3`
- Fold 3: `O 1`, `R 0`, `I 0`

### **Round 5**
- Fold 0: `O 1`, `R 2`, `I 1`
- Fold 1: `O 2`, `R 3`, `I 2`
- Fold 2: `O 3`, `R 0`, `I 3`
- Fold 3: `O 0`, `R 1`, `I 0`

### **Round 6**
- Fold 0: `O 2`, `R 0`, `I 1`
- Fold 1: `O 3`, `R 1`, `I 2`
- Fold 2: `O 0`, `R 2`, `I 3`
- Fold 3: `O 1`, `R 3`, `I 0`

### **Round 7**
- Fold 0: `O 3`, `R 0`, `I 1`
- Fold 1: `O 0`, `R 1`, `I 2`
- Fold 2: `O 1`, `R 2`, `I 3`
- Fold 3: `O 2`, `R 3`, `I 0`

---

## Summary

This multi-round cross-validation approach ensures:
1. Diverse combinations of samples across folds.
2. Reduced impact of similarity bias.
3. Statistically robust evaluation with over 30 observations.

The distribution scheme is documented to enable reproducibility and comparison across future studies.