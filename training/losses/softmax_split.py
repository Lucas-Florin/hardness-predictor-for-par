import torch
import torch.nn as nn
import numpy as np


class SoftmaxSplit(nn.Module):

    def __init__(self, attribute_groupings):
        super(SoftmaxSplit, self).__init__()
        self.attribute_groupings = np.array(attribute_groupings, dtype=np.int)

        self.num_groups = self.attribute_groupings.max() + 1
        self.children_dict = dict()  # Stores all the children of this layer.

        # Initialize the corresponding operation for each group.
        for group in range(self.num_groups):
            idxs = self.attribute_groupings == group
            if idxs.sum() == 1:
                # Groups of one (groupless attributes) get Sigmoid.
                self.children_dict[group] = nn.Sigmoid()
            else:
                # Groups of two or more get the Softmax over all the attributes in that group.
                self.children_dict[group] = nn.Softmax(dim=1)

    def forward(self, inputs):
        outputs = list()
        for group in range(self.num_groups):

            idxs = np.argwhere(self.attribute_groupings == group)
            low_idx = idxs[0].item()  # The index of the first occurrence of teh group.
            high_idx = idxs[-1].item()  # The index of the last occurrence of teh group.
            group_input = inputs[:, low_idx : high_idx + 1]  # The part of the input corresponding to each group.
            outputs.append(self.children_dict[group](group_input))  # Process group through corresponding operation.

        return torch.cat(outputs, dim=1)


