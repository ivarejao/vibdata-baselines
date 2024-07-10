import itertools

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_random_state
from sklearn.model_selection._split import GroupsConsumerMixin, check_array, _RepeatedSplits


class RepeteadNewLogo(_RepeatedSplits):
    def __init__(self, n_repeats: int = 1, random_state=None, y=None, groups=None):
        self.n_splits = int(len(np.unique(groups)) / len(
            np.unique(y)
        ))  # n_splits = total_groups / total_labels = num_conditions
        self.combinations = self._compute_combinations(y, groups)
        super().__init__(
            NewLogo,
            n_repeats=n_repeats,
            random_state=random_state,
            combinations=self.combinations,
            update_combinations=True,
        )

    def _compute_combinations(self, y, groups):
        labels = np.unique(y)
        # Create a dictionary to map labels to their unique groups
        initial_states = {label: np.unique(groups[y == label]).tolist() for label in labels}

        def shift_groups(groups, shift):
            return groups[-shift:] + groups[:-shift]

        def backtrack(currrent_combination, label_idx):
            # print(currrent_combination)
            # print(initial_states)
            # print(currrent_combination.keys() == initial_states.keys())
            # print("---")
            if currrent_combination.keys() == initial_states.keys():
                yield currrent_combination
                return 

            for shift in range(self.n_splits):  
                label = labels[label_idx]
                shifted_groups = shift_groups(initial_states[label], shift)
                currrent_combination[label] = shifted_groups
                yield from backtrack(currrent_combination.copy(), label_idx + 1)
                currrent_combination.pop(label)

        yield from backtrack({}, 0)


class NewLogo(GroupsConsumerMixin, BaseCrossValidator):

    # TODO: Remove this class variable if possible
    round_combination = None

    def __init__(self, shuffle: bool = False, random_state=None, combinations=None, update_combinations=False):
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                (
                    "Setting a random_state has no effect since shuffle is "
                    "False. You should leave "
                    "random_state to its default (None), or set shuffle=True."
                ),
            )

        if combinations is None:
            raise ValueError("combinations should not be None.")

        self.shuffle = shuffle
        self.random_state = random_state
        if update_combinations:
            NewLogo.round_combination = next(combinations)
        self.current_combination = NewLogo.round_combination 
        # Ensure that the group distribution is consistent
        lengths = [len(values) for values in self.current_combination.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All values in groups_distribution should have the same length.")
        self.n_splits = lengths[0]

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side-effects during iteration
        groups = check_array(groups, input_name="groups", copy=True, ensure_2d=False, dtype=None)
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups
            )
        # self.groups_distribution : Dict[Int, List[Int]]
        for i in range(self.n_splits):
            fold_groups = []
            for label, values in self.current_combination.items():
                fold_groups.append(values[i])
            test_mask = np.isin(groups, fold_groups)
            yield test_mask

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class LeaveBalancedGroupsOut(GroupsConsumerMixin, BaseCrossValidator):
    def __init__(self, shuffle: bool = False, random_state=None, used_combinations: set = None):
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                (
                    "Setting a random_state has no effect since shuffle is "
                    "False. You should leave "
                    "random_state to its default (None), or set shuffle=True."
                ),
            )

        self.shuffle = shuffle
        self.random_state = random_state
        self.used_combinations = used_combinations if used_combinations is not None else set()

    def _compute_combinations(self, y, groups):
        labels = np.unique(y)
        # Create a dictionary to map labels to their unique groups
        label_to_groups = {label: np.unique(groups[y == label]) for label in labels}

        # Create combinations that ensure each fold has one unique group per label
        group_combinations = list(itertools.product(*label_to_groups.values()))
        if self.shuffle:
            check_random_state(self.random_state).shuffle(group_combinations)
        return group_combinations

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side-effects during iteration
        groups = check_array(groups, input_name="groups", copy=True, ensure_2d=False, dtype=None)
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups
            )

        group_combinations = self._compute_combinations(y, groups)
        # TODO: Improve loop and checking
        folds_created = 0
        for combination in group_combinations:
            if combination not in self.used_combinations:
                self.used_combinations.add(combination)
                folds_created += 1
                test_mask = np.isin(groups, combination)
                yield test_mask
            if folds_created == self.n_splits:
                break
        print(folds_created)
        if folds_created != self.n_splits:
            raise Exception("There are not enough combinations for the number of splits")

    def split(self, X, y=None, groups=None):
        self.n_splits = len(np.unique(groups)) / len(
            np.unique(y)
        )  # n_splits = total_groups / total_labels = num_conditions
        return super().split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


from vibdata.deep.signal.core import SignalSample

from vibnet.data.group_dataset import GroupDataset


class GroupRepeatedCWRU48k(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        return sample_metainfo["label"].astype(str) + " " + sample_metainfo["load"].astype(int).astype(str)
