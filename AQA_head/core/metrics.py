
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.stats import spearmanr as sp

def spearmanCorr(y_true, y_pred):
    return sp(y_true,y_pred)
