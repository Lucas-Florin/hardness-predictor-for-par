
# TODO: Documentation
# TODO: Make dict keys consistent with the rest of the code


class ResultManager(object):
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

    def __init__(self, result_dict):
        self.result_dict = result_dict
        if self.output_str in result_dict:
            self.output = result_dict[self.output_str]
        else:
            self.output = dict()
        result_dict[self.output_str] = self.output

    def check_output_dict(self, split):
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
        split_output = self.output[split]
        return [split_output[o] for o in self.ouput_order]

    def update_outputs(self, split, **kwargs):
        if split not in self.output:
            self.output[split] = dict()
        split_output = self.output[split]
        for key, value in kwargs.items():
            if key not in self.ouput_order:
                raise ValueError("Invalid output name: " + key)
            split_output[key] = value

    def print_stored(self):
        for split in self.output:
            print("Split: {}".format(split))
            print(list(self.output[split]))

