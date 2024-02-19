import os

import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from vibdata.deep.DeepDataset import DeepDataset
from vibdata.deep.signal.core import SignalSample

from vibnet.config import Config


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
            mapped_samples = map(
                self._assigne_group,
                tqdm(self.dataset, total=len(self.dataset), unit="sample", desc="Grouping dataset: "),
            )
            groups = np.array(list(mapped_samples))
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
    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        super().__init__(dataset, config)

        self.normals_bins = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }

    def _assigne_group(self, sample: SignalSample) -> int:
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
            group = min(self.normals_bins, key=self.normals_bins.get)
            self.normals_bins[group] += 1
            return group
        else:
            raise Exception("Unexpected sample")


class GroupIMS(GroupDataset):
    NUM_FOLDS = 3

    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        super().__init__(dataset, config)

        keys = dataset.get_labels()
        values = dataset.get_labels_name()

        self.labels_name = dict(zip(keys, values))
        # Compute how much of each class uniformed distributed should be assinged to each fold

        name_to_label = dict(zip(values, keys))

        metainfo = dataset.get_metainfo()
        defects_frequency = metainfo[metainfo.label != name_to_label["Normal"]].label.value_counts()
        # Create a dict with the amount of samples per fold
        self.defects_bins = {
            label: {"samples_per_fold": np.ceil(total / GroupIMS.NUM_FOLDS), "current_amount": 0}
            for label, total in defects_frequency.items()
        }

    def _get_group_divided(self, label: int):
        current_amount = self.defects_bins[label]["current_amount"]
        samples_per_fold = self.defects_bins[label]["samples_per_fold"]

        group = (current_amount // samples_per_fold) + 1
        self.defects_bins[label]["current_amount"] += 1
        return group

    def _assigne_group(self, sample: SignalSample) -> int:
        bearing = sample["metainfo"]["bearing"]
        test = sample["metainfo"]["test"]
        label = sample["metainfo"]["label"]
        label_str = self.labels_name[label]

        if test == 1 and bearing == 3:
            return 1 if label_str == "Normal" else self._get_group_divided(label)
        elif test == 1 and bearing == 4:
            return 2 if label_str == "Normal" else self._get_group_divided(label)
        elif test == 2 and bearing == 1:
            return 3 if label_str == "Normal" else self._get_group_divided(label)
        else:
            raise Exception(
                "Unexpected sample. The sample received is one of the conditions left out.\n"
                "The sample is of test: " + str(test) + " and bearing: " + str(bearing)
            )


class GroupMAFAULDA(GroupDataset):
    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        super().__init__(dataset, config)

        keys = dataset.get_labels()
        values = dataset.get_labels_name()

        self.labels = dict(zip(keys, values))
        self.normal_groups = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }

    def _assigne_group(self, sample: SignalSample) -> int:
        label = sample["metainfo"]["label"]
        label_str = self.labels[label]

        if label_str == "Normal":
            group = min(self.normal_groups, key=self.normal_groups.get)
            self.normal_groups[group] += 1
            return group
        else:
            test_measure = sample["metainfo"]["test_measure"]

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
            else:
                raise Exception("Unexpected sample")


class GroupMFPT(GroupDataset):
    NUM_FOLDS = 3

    FAKE_OUTER_RACE_270_LABEL = 100

    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        super().__init__(dataset, config)

        keys = dataset.get_labels()
        values = dataset.get_labels_name()

        self.labels_name = dict(zip(keys, values))
        name_to_label = dict(zip(values, keys))

        metainfo = dataset.get_metainfo()
        outer_race_270_mask = (metainfo.label == name_to_label["Outer Race"]) & (metainfo.load == 270)
        outer_race_270_frequency = metainfo[outer_race_270_mask].label.value_counts()
        # Trick so that can differentiate from outer race label
        outer_race_270_frequency.index = [GroupMFPT.FAKE_OUTER_RACE_270_LABEL]
        others_labels_frequency = metainfo[~outer_race_270_mask].label.value_counts()

        labels_frequency = pd.concat([outer_race_270_frequency, others_labels_frequency])
        # Create a dict with the amount of samples per fold
        self.labels_bins = {
            label: {"samples_per_fold": total // GroupMFPT.NUM_FOLDS, "current_amount": 0}
            for label, total in labels_frequency.items()
        }

    def _get_group_divided(self, label):
        current_amount = self.labels_bins[label]["current_amount"]
        samples_per_fold = self.labels_bins[label]["samples_per_fold"]

        group = (current_amount // samples_per_fold) + 1
        self.labels_bins[label]["current_amount"] += 1
        return group

    def _assigne_group(self, sample: SignalSample) -> int:
        label = sample["metainfo"]["label"]
        label_str = self.labels_name[label]
        load = sample["metainfo"]["load"]

        if label_str == "Outer Race" and load == 270:
            label = self.FAKE_OUTER_RACE_270_LABEL

        return self._get_group_divided(label)


class GroupPU(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        rotation_speed = sample["metainfo"]["file_name"][:3]
        load_torque = sample["metainfo"]["load_nm"]
        radial_force = sample["metainfo"]["radial_force_n"]
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

    NUM_FOLDS = 5

    def __init__(self, dataset: DeepDataset, config: Config) -> None:
        super().__init__(dataset, config)

        keys = dataset.get_labels()
        values = dataset.get_labels_name()

        self.labels_name = dict(zip(keys, values))

        self.labels_bins = {label: {fold: 0 for fold in range(1, GroupUOC.NUM_FOLDS + 1)} for label in keys}

    def _assigne_group(self, sample: SignalSample) -> int:
        severity = sample["metainfo"]["severity"]
        if severity != "-":
            return int(severity)
        else:
            label = sample["metainfo"]["label"]
            group = min(self.labels_bins[label], key=self.labels_bins[label].get)
            self.labels_bins[label][group] += 1
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
