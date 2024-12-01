### `docs/multi_round/PU.md`

# Multi-Round Cross-Validation: PU Dataset

This document describes the multi-round cross-validation strategy for the PU dataset. The dataset includes fault categories recorded under **four different working conditions** to simulate a variety of real-world scenarios.

---

## Methodology

The PU dataset consists of four labels:
- **Normal (N)**
- **Outer Ring Fault (ORI)**
- **Inner Ring Fault (IRI)**
- **Outer and Inner Ring Fault (OIRI)**

### Working Conditions:
Each label was recorded under the following conditions:
1. **C1:** Speed = 1500 RPM, Torque = 0.7 Nm, Radial Force = 1000 N
2. **C2:** Speed = 900 RPM, Torque = 0.7 Nm, Radial Force = 1000 N
3. **C3:** Speed = 1500 RPM, Torque = 0.1 Nm, Radial Force = 1000 N
4. **C4:** Speed = 1500 RPM, Torque = 0.7 Nm, Radial Force = 400 N

### Key Details:
1. **Folds per round:** 4 (one per working condition).
2. **Total rounds:** 8.
3. **Combination strategy:** Labels are reassigned across folds in each round to ensure variation.
4. **Notation:** Each label is combined with its working condition, e.g., `ORI C2` for Outer Ring Fault in Condition C2.

---

## Round Configurations

### **Round 0**
- Fold 0: `N C2`, `ORI C2`, `OIRI C2`, `IRI C2`
- Fold 1: `N C3`, `ORI C3`, `OIRI C3`, `IRI C3`
- Fold 2: `N C1`, `ORI C1`, `OIRI C1`, `IRI C1`
- Fold 3: `N C4`, `ORI C4`, `OIRI C4`, `IRI C4`

### **Round 1**
- Fold 0: `N C2`, `ORI C2`, `OIRI C3`, `IRI C3`
- Fold 1: `N C3`, `ORI C4`, `OIRI C2`, `IRI C1`
- Fold 2: `N C1`, `ORI C3`, `OIRI C1`, `IRI C4`
- Fold 3: `N C4`, `ORI C1`, `OIRI C4`, `IRI C2`

### **Round 2**
- Fold 0: `N C2`, `ORI C4`, `OIRI C2`, `IRI C3`
- Fold 1: `N C3`, `ORI C3`, `OIRI C1`, `IRI C4`
- Fold 2: `N C1`, `ORI C2`, `OIRI C4`, `IRI C1`
- Fold 3: `N C4`, `ORI C1`, `OIRI C3`, `IRI C2`

### **Round 3**
- Fold 0: `N C2`, `ORI C1`, `OIRI C1`, `IRI C3`
- Fold 1: `N C3`, `ORI C4`, `OIRI C4`, `IRI C2`
- Fold 2: `N C1`, `ORI C2`, `OIRI C2`, `IRI C1`
- Fold 3: `N C4`, `ORI C3`, `OIRI C3`, `IRI C4`

### **Round 4**
- Fold 0: `N C2`, `ORI C3`, `OIRI C1`, `IRI C4`
- Fold 1: `N C3`, `ORI C4`, `OIRI C2`, `IRI C3`
- Fold 2: `N C1`, `ORI C2`, `OIRI C4`, `IRI C2`
- Fold 3: `N C4`, `ORI C1`, `OIRI C3`, `IRI C1`

### **Round 5**
- Fold 0: `N C2`, `ORI C4`, `OIRI C3`, `IRI C1`
- Fold 1: `N C3`, `ORI C1`, `OIRI C4`, `IRI C3`
- Fold 2: `N C1`, `ORI C3`, `OIRI C2`, `IRI C4`
- Fold 3: `N C4`, `ORI C2`, `OIRI C1`, `IRI C2`

### **Round 6**
- Fold 0: `N C2`, `ORI C2`, `OIRI C3`, `IRI C1`
- Fold 1: `N C3`, `ORI C3`, `OIRI C2`, `IRI C2`
- Fold 2: `N C1`, `ORI C4`, `OIRI C1`, `IRI C4`
- Fold 3: `N C4`, `ORI C1`, `OIRI C4`, `IRI C3`

### **Round 7**
- Fold 0: `N C2`, `ORI C1`, `OIRI C2`, `IRI C3`
- Fold 1: `N C3`, `ORI C4`, `OIRI C3`, `IRI C2`
- Fold 2: `N C1`, `ORI C3`, `OIRI C4`, `IRI C1`
- Fold 3: `N C4`, `ORI C2`, `OIRI C1`, `IRI C4`

---

## Summary

This setup provides a comprehensive evaluation framework that:
1. Tests the model's adaptability to varying working conditions.
2. Ensures diverse label combinations across all rounds.
3. Enhances the generalization and robustness of fault diagnosis systems.
