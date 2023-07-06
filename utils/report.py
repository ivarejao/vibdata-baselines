from typing import Dict, List


class ReportDict(dict):
    def __init__(self, fields: List[str]):
        super(ReportDict, self).__init__({f: [] for f in fields})

    def update(self, reportset: Dict, **kwargs):
        for key in reportset:
            self[key].extend(reportset[key])
        for key, values in kwargs.items():
            self[key].extend(values)
