# Multi-Round Cross-Validation: MFPT Dataset (48,828 Hz Sample Rate)

This document outlines the multi-round cross-validation strategy for the MFPT dataset recorded at 48,828 Hz. The experiment utilizes fault categories and varying load conditions to create diverse folds for robust evaluation.

---

## Methodology

The MFPT dataset contains two fault types:
- **Outer Race Fault (OR)**
- **Inner Race Fault (IR)**

Each fault type was recorded under **seven distinct load conditions**:
- **Outer Race Fault (OR):** 25 lbs, 50 lbs, 100 lbs, 150 lbs, 200 lbs, 250 lbs, 300 lbs
- **Inner Race Fault (IR):** 0 lbs, 50 lbs, 100 lbs, 150 lbs, 200 lbs, 250 lbs, 300 lbs

### Key Details:
1. **Folds per round:** 7 (one per load condition).
2. **Total rounds:** 5.
3. **Combination strategy:** Fault conditions are reassigned across folds in each round to introduce variation.
4. **Notation:**
   - `OR 25`: Outer Race Fault recorded at 25 lbs load.
   - `IR 150`: Inner Race Fault recorded at 150 lbs load.

---

## Round Configurations

### **Round 0**
- Fold 0: `OR 25`, `IR 0`
- Fold 1: `OR 50`, `IR 50`
- Fold 2: `OR 100`, `IR 100`
- Fold 3: `OR 150`, `IR 150`
- Fold 4: `OR 200`, `IR 200`
- Fold 5: `OR 250`, `IR 250`
- Fold 6: `OR 300`, `IR 300`

### **Round 1**
- Fold 0: `OR 25`, `IR 150`
- Fold 1: `OR 50`, `IR 0`
- Fold 2: `OR 100`, `IR 200`
- Fold 3: `OR 150`, `IR 50`
- Fold 4: `OR 200`, `IR 250`
- Fold 5: `OR 250`, `IR 300`
- Fold 6: `OR 300`, `IR 100`

### **Round 2**
- Fold 0: `OR 25`, `IR 250`
- Fold 1: `OR 50`, `IR 300`
- Fold 2: `OR 100`, `IR 50`
- Fold 3: `OR 150`, `IR 100`
- Fold 4: `OR 200`, `IR 150`
- Fold 5: `OR 250`, `IR 0`
- Fold 6: `OR 300`, `IR 200`

### **Round 3**
- Fold 0: `OR 25`, `IR 50`
- Fold 1: `OR 50`, `IR 200`
- Fold 2: `OR 100`, `IR 150`
- Fold 3: `OR 150`, `IR 250`
- Fold 4: `OR 200`, `IR 300`
- Fold 5: `OR 250`, `IR 100`
- Fold 6: `OR 300`, `IR 0`

### **Round 4**
- Fold 0: `OR 25`, `IR 200`
- Fold 1: `OR 50`, `IR 250`
- Fold 2: `OR 100`, `IR 300`
- Fold 3: `OR 150`, `IR 0`
- Fold 4: `OR 200`, `IR 100`
- Fold 5: `OR 250`, `IR 150`
- Fold 6: `OR 300`, `IR 50`

---

## Summary

The described cross-validation setup ensures:
1. Diverse fault and load condition combinations across folds in all rounds.
2. Comprehensive testing of models against varying operational conditions.
3. Increased generalization and reliability in performance evaluation.