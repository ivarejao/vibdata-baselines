# Multi-Round Cross-Validation: CWRU Dataset (48 kHz Sample Rate)

This document explains the updated multi-round cross-validation strategy applied to the CWRU dataset with a sample rate of 48 kHz. The experiment leverages the dataset's distinct load conditions and fault categories to create diverse fold combinations for robust hypothesis testing.

---

## Methodology

The CWRU dataset comprises samples across three labels:
- **Inner Race Fault (IR)**
- **Outer Race Fault (OR)**
- **Ball Fault (B)**

Each label is recorded under **four different load conditions**:
- **0 horsepower (hp)**
- **1 horsepower (hp)**
- **2 horsepower (hp)**
- **3 horsepower (hp)**

For multi-round cross-validation:
1. **Four folds** are created per round, with each fold containing samples corresponding to a specific load condition.
2. **Rounds reassign samples**, creating diverse combinations of fault conditions across folds.

This strategy evaluates model performance across varying operational and fault conditions while reducing similarity bias.

---

## Experimental Setup

- **Total rounds:** 8
- **Folds per round:** 4
- **Combination logic:** Each round redistributes samples of fault conditions across folds to create new training/testing splits.
- **Example notation:**
  - `IR 0` represents **Inner Race Fault** samples recorded at **0 hp**.
  - `B 1` represents **Ball Fault** samples recorded at **1 hp**.
  - `OR 2` represents **Outer Race Fault** samples recorded at **2 hp**.

---

## Round Configurations

### **Round 0**
- Fold 0: `IR 0`, `B 0`, `OR 0`
- Fold 1: `IR 1`, `B 1`, `OR 1`
- Fold 2: `IR 2`, `B 2`, `OR 2`
- Fold 3: `IR 3`, `B 3`, `OR 3`

### **Round 1**
- Fold 0: `IR 0`, `B 0`, `OR 1`
- Fold 1: `IR 1`, `B 2`, `OR 0`
- Fold 2: `IR 2`, `B 3`, `OR 3`
- Fold 3: `IR 3`, `B 1`, `OR 2`

### **Round 2**
- Fold 0: `IR 0`, `B 1`, `OR 3`
- Fold 1: `IR 1`, `B 0`, `OR 0`
- Fold 2: `IR 2`, `B 2`, `OR 1`
- Fold 3: `IR 3`, `B 3`, `OR 2`

### **Round 3**
- Fold 0: `IR 0`, `B 1`, `OR 2`
- Fold 1: `IR 1`, `B 3`, `OR 3`
- Fold 2: `IR 2`, `B 0`, `OR 1`
- Fold 3: `IR 3`, `B 2`, `OR 0`

### **Round 4**
- Fold 0: `IR 0`, `B 2`, `OR 3`
- Fold 1: `IR 1`, `B 0`, `OR 1`
- Fold 2: `IR 2`, `B 1`, `OR 2`
- Fold 3: `IR 3`, `B 3`, `OR 0`

### **Round 5**
- Fold 0: `IR 0`, `B 0`, `OR 3`
- Fold 1: `IR 1`, `B 1`, `OR 2`
- Fold 2: `IR 2`, `B 2`, `OR 0`
- Fold 3: `IR 3`, `B 3`, `OR 1`

### **Round 6**
- Fold 0: `IR 0`, `B 3`, `OR 1`
- Fold 1: `IR 1`, `B 1`, `OR 0`
- Fold 2: `IR 2`, `B 2`, `OR 3`
- Fold 3: `IR 3`, `B 0`, `OR 2`

### **Round 7**
- Fold 0: `IR 0`, `B 3`, `OR 2`
- Fold 1: `IR 1`, `B 2`, `OR 1`
- Fold 2: `IR 2`, `B 0`, `OR 3`
- Fold 3: `IR 3`, `B 1`, `OR 0`

---

## Summary

This updated multi-round cross-validation setup ensures:
1. Diverse combinations of fault and load conditions across folds.
2. Reduction in similarity bias.
3. Over 30 observations across all rounds for robust statistical analysis.
