from experiments.backpropogation.run import run as bp_run
from experiments.knn.k_variation import k_variation
from experiments.knn.run import run as knn_run
from experiments.logistic_regresssion.run import run as lr_run

knn_run()
k_variation()
lr_run()
bp_run()