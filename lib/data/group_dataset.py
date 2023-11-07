import os

import numpy as np
import numpy.typing as npt
from vibdata.deep.DeepDataset import DeepDataset
from vibdata.deep.signal.core import SignalSample

from lib.config import Config


class GroupDataset:
    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        self.dataset = dataset
        self.config = config
        self.groups_dir = self.config["dataset"]["groups_dir"]
        self.groups_file = os.path.join(self.groups_dir, "groups_" + self.config["dataset"]["name"] + ".npy")

    def groups(self) -> npt.NDArray[np.int_]:
        """
        Get the groups from all samples of the dataset. It tries to load from memory at `groups_dir` but if it
        doesnt exists it will compute the groups and save it in `groups_file`.

        Returns:
            npt.NDArray[np.int_]: groups of all dataset
        """
        if os.path.exists(self.groups_file):
            return np.load(self.groups_file)
        else:
            groups = np.array(list(map(self._assigne_group, self.dataset)))
            np.save(self.groups_file, groups)
            return groups

    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        """
        Get a signal sample and based on the dataset criterion, assigne a group
        to the given sample

        Args:
            sample (SignalSample): sample to be assigned

        Returns:
            int: group id
        """
        pass


class GroupCWRU(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        return sample["metainfo"]["load"]


class GroupEAS(GroupDataset):
    normal_groups = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        unbalance_factor = sample["metainfo"]["unbalance_factor"]

        if unbalance_factor == 45:
            return 1
        elif unbalance_factor == 60:
            return 2
        elif unbalance_factor == 75:
            return 3
        elif unbalance_factor == 152:
            return 4
        elif unbalance_factor == 0:
            group = min(GroupEAS.normal_groups, key=GroupEAS.normal_groups.get)
            GroupEAS.normal_groups[group] += 1
            return group
        else:
            raise Exception("Unexpected sample")


class GroupIMS(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        pass


class GroupMAFAULDA(GroupDataset):
    normal_groups = {
        1: 0,
        2: 0,
        3: 0,
    }

    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        super().__init__(dataset, config)

        keys = dataset.get_labels_name()
        values = dataset.get_labels()
        pass

    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        if sample["metainfo"]["label"] == 13:  # Normal
            group = min(GroupMAFAULDA.normal_groups, key=GroupMAFAULDA.normal_groups.get)
            GroupMAFAULDA.normal_groups[group] += 1
            return group
        else:
            test_measure = sample['metainfo']['test_measure']
            pass


class GroupMFPT(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        pass


class GroupPU(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        rotation_speed = sample["metainfo"]["file_name"][:3]
        load_torque = sample['metainfo']['load_nm']
        radial_force = sample['metainfo']['radial_force_n']
        if rotation_speed == "N15" and load_torque == 0.7 and radial_force == 1000:
            return 1
        elif rotation_speed == "N09" and load_torque == 0.7 and radial_force == 1000:
            return 2
        elif rotation_speed == "N15" and load_torque == 0.1 and radial_force == 1000:
            return 3
        elif rotation_speed == "N15" and load_torque == 0.7 and radial_force == 400:
            return 4
        else:
            raise Exception("Unexpected operating condition")


class GroupUOC(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        severity = sample['metainfo']['severity']
        if severity != "-":
            return int(severity)
        else:
            pass


class GroupXJTU(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        file_name = sample["metainfo"]["file_name"]
        if "Bearing1" in file_name:
            return 1
        elif "Bearing2" in file_name:
            return 2
        elif "Bearing3" in file_name:
            return 3
        else:
            raise Exception(f"The file {file_name} does not belong to any group")
