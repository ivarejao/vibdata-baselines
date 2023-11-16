from typing import Dict, List
import numpy as np

class ReportDict(dict):
    def __init__(self, fields: List[str]):
        super(ReportDict, self).__init__({f: [] for f in fields})

    def update(self, reportset: Dict, **kwargs):
        for key in reportset:
            self[key].extend(reportset[key])
        for key, values in kwargs.items():
            self[key].extend(values)

def array_info(arr):# Display the array

    print("\nShape:", arr.shape, ", Data Type:", arr.dtype)
    
    print("Min:", np.min(arr), ", Max:", np.max(arr))
    
    print("Mean:", np.mean(arr), ", Median:", np.median(arr), ", Std Dev:", np.std(arr), end=" ")

    # Get unique elements and their counts
    print("\nUnique Elements and Counts:")
    unique_elements, counts = np.unique(arr, return_counts=True)
    print("Unique Elements:", unique_elements)
    print("Counts:", counts)