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
            os.makedirs(self.groups_dir, exist_ok=True)  # Ensure that the directory exists
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

    labels = {}

    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        super().__init__(dataset, config)

        keys = dataset.get_labels()
        values = dataset.get_labels_name()

        GroupMAFAULDA.labels = dict(zip(keys, values))

    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        label = sample["metainfo"]["label"]
        label_str = GroupMAFAULDA.labels[label]

        if label_str == "Normal":
            group = min(GroupMAFAULDA.normal_groups, key=GroupMAFAULDA.normal_groups.get)
            GroupMAFAULDA.normal_groups[group] += 1
            return group

        else:
            test_measure = sample['metainfo']['test_measure']

            if label_str == "Horizontal Misalignment":
                if test_measure == "0.5mm":
                    return 1
                elif test_measure == "1.0mm":
                    return 2
                elif test_measure == "1.5mm":
                    return 3
                elif test_measure == "2.0mm":
                    return 4

            elif label_str == "Vertical Misalignment":
                if test_measure == "0.51mm" or test_measure == "0.63mm":
                    return 1
                elif test_measure == "1.27mm":
                    return 2
                elif test_measure == "1.40mm":
                    return 3
                elif test_measure == "1.78mm" or test_measure == "1.90mm":
                    return 4

            elif label_str == "Imbalance":
                if test_measure == "6g" or test_measure == "10g":
                    return 1
                elif test_measure == "15g" or test_measure == "20g":
                    return 2
                elif test_measure == "25g" or test_measure == "30g":
                    return 3
                elif test_measure == "35g":
                    return 4

            elif 17 <= label <= 22:  # Bearing
                if test_measure == "0g":
                    return 1
                elif test_measure == "6g":
                    return 2
                elif test_measure == "20g":
                    return 3
                elif test_measure == "35g":
                    return 4

        raise Exception("Unexpected sample")


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
    healthy_groups = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    missing_tooth_groups = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    root_crack_groups = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    spalling_groups = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    labels = {}

    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        super().__init__(dataset, config)

        keys = dataset.get_labels()
        values = dataset.get_labels_name()

        GroupUOC.labels = dict(zip(keys, values))

    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        severity = sample['metainfo']['severity']
        if severity != "-":
            return int(severity)
        else:
            label = sample["metainfo"]["label"]
            label_str = GroupUOC.labels[label]

            if label_str == "Healthy":
                group_dict = GroupUOC.healthy_groups
            elif label_str == "Missing Tooth":
                group_dict = GroupUOC.missing_tooth_groups
            elif label_str == "Root Crack":
                group_dict = GroupUOC.root_crack_groups
            elif label_str == "Spalling":
                group_dict = GroupUOC.spalling_groups
            else:
                raise Exception("Unexpected sample")

            group = min(group_dict, key=group_dict.get)
            group_dict[group] += 1
            return group


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
