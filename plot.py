import matplotlib
import matplotlib.pyplot as plt

import os
import pandas as pd

import params

# Create a DataFrame that will be used to create a plot and find the best model. 
log_path = os.path.join(params.TRAIN_DIR, "log.csv")
LogDF = pd.read_csv(log_path, index_col=0)

# Find the best step. 
best_test_step = LogDF['Test Error'].idxmin()
best_test_score = LogDF['Test Error'].min()
best_train_score = LogDF['Train Error'][best_test_step]

# Create a learning plot. 
pd.DataFrame(LogDF, columns=['Train Error', 'Test Error']).plot()
plt.annotate(
    "{0:.4f}".format(best_test_score), 
    xy=(best_test_step, best_test_score), 
    xytext=(best_test_step, best_test_score+0.04), 
    arrowprops=dict(facecolor='orange'))
plt.annotate(
    "{0:.4f}".format(best_train_score), 
    xy=(best_test_step, best_train_score), 
    xytext=(best_test_step, best_train_score+0.04), 
    arrowprops=dict(facecolor='blue'))
plt.suptitle("Best Step: {}".format(best_test_step))
plot_path = os.path.join(params.TRAIN_DIR, "plot.png")
plt.savefig(plot_path)
