# -*- coding: utf-8 -*-
"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    soft_ret = [];
    for i in range(len(x)):
        soft_ret.append(np.exp(x)[i] / sum(np.exp(x)))
    return np.array(soft_ret)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt

x = np.arange(-2.0, 6.0, 0.1)
# it creates a range with scores not only one array
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
