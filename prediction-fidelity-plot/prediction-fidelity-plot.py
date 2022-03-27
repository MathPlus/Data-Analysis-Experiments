
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

lr = linear_model.LinearRegression()
X, measured = datasets.load_diabetes(return_X_y=True)

predicted = cross_val_predict(lr, X, measured, cv=10)

( r , r_pval ) = pearsonr( measured , predicted )

fig, ax = plt.subplots()
ax.scatter(measured, predicted)
ax.plot([measured.min(), measured.max()], [measured.min(), measured.max()], "k--", lw=2)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.title( 'Measured vs. Predicted; R = {r:.1f}'.format(r = r) )
plt.show()

"""

The above code was downloaded from an open source, then modified and augmented.
https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html

Below is code for the prediction fidelity plot.

"""


n = len(measured)

error_meas_rel_pred = np.abs( 1.0 - measured / predicted )
# max(error_meas_rel_pred) == 1.9713433761257009
err_max = 2.0
err_min = 0.0
err_step = 0.01

err_range = np.arange( err_min , err_max , err_step )

fidelity_ratio = [ np.sum( error_meas_rel_pred <= err ) / n for err in err_range ]

fig, ax = plt.subplots()
ax.plot( err_range , fidelity_ratio , 'k-' , lw=2 )
ax.set_xlim([0.0,0.81])
ax.set_xlabel("Tolerance: relative deviation of measurement from prediction")
ax.set_ylabel("Prediction fidelity proportion")
plt.title( 'Prediction fidelity plot' )
plt.grid( linestyle = '--' , linewidth = 0.5 )
plt.show()