# TODO: Make dict keys consistent with the rest of the code


class ResultManager(object):
    """
    This class manages the different outputs of a model to avoid redundant computations.
    """
    output_str = "output"
    predictions_str = "predictions"
    prediction_probs_str = "prediction_probs"
    labels_str = "labels"
    hp_scores_str = "hp_scores"
    ouput_order = (
        labels_str,
        prediction_probs_str,
        predictions_str,
        hp_scores_str
    )
    f1_string = "f1_thresholds"
    positivity_ratio_str = "positivity_ratio"

    def __init__(self, result_dict, use_cache=True):
        """
        Initialize a new result manager.
        :param result_dict: dict with the results. If no results are available this dict is empty.
        """
        self.result_dict = result_dict
        self.use_cache = use_cache
        if self.output_str in result_dict:
            self.output = result_dict[self.output_str]
        else:
            self.output = dict()
        result_dict[self.output_str] = self.output

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def check_output_dict(self, split):
        """
        Check if all the outputs for a specific split are present.
        :param split: a string of the split name.
        :return: True if all the outputs for that split are present.
        """
        if not self.use_cache:
            return False
        if self.output is None:
            return False
        if split not in self.output:
            return False
        split_output = self.output[split]
        return (
            self.prediction_probs_str in split_output and
            self.predictions_str in split_output and
            self.labels_str in split_output and
            self.hp_scores_str in split_output
        )

    def get_outputs(self, split):
        """
        Get the four outputs for the split.
        :param split: a string of the split name.
        :return: a list with the four outputs in the order defined in the class constants.
        """
        split_output = self.output[split]
        return [split_output[o] for o in self.ouput_order]

    def update_outputs(self, split, **kwargs):
        """
        Save outputs for a specific split. Outputs are passed as keyword arguments with the split name as the keyword.
        :param split: a string of the split name.
        """
        if split not in self.output:
            self.output[split] = dict()
        split_output = self.output[split]
        for key, value in kwargs.items():
            if key not in self.ouput_order:
                raise ValueError("Invalid output name: " + key)
            split_output[key] = value

    def print_stored(self):
        """
        Print the saved outputs.
        """
        for split in self.output:
            print("Split: {}".format(split))
            print(list(self.output[split]))

