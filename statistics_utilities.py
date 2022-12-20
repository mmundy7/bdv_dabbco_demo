# Necessary imports
import pandas as pd, string, numpy as np, time, pickle, gc as gc, warnings
import seaborn as sns, matplotlib.pyplot as plt

# Normality Test Imports
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from scipy.stats import shapiro, normaltest, anderson

# Other settings
gc.enable()
warnings.filterwarnings("ignore")

# Function to plot qq plots for a list of data
def qq_plot(data):
    qqplot(np.asarray(data), line='s')
    pyplot.show()

# Function to plot a histogram of data
def plot_hist(data, name, bin_num):
    # Create a temp numpy array
    temp = np.asarray(data)
    # Plot the distribution
    n, bins, patches = plt.hist(temp, bins=bin_num)
    # Set the title
    plt.suptitle("{} Error Distribution".format(str(name)))
    # Show the figure
    plt.show()       

# Shapiro Wilk Test
def shapiro_wilk(data, alpha):
    # Run the shapiro test
    stat, p = shapiro(np.asarray(data))
    p=float(p)
    
    print("Shapiro Wilks Test | P Value: {} | Statistic: {}".format(p, stat))
    print(p, alpha)
    # If p is greater than the alpha value passed
    if p > alpha:
        return True
    else:
        return False

# Dagastino K^2 Test
def dagastino(data, alpha):
    # Run the shapiro test
    stat, p = normaltest(np.asarray(data))
    p=float(p)

    print("Dagastino Test | P Value: {} | Statistic: {}".format(p, stat))

    if p > alpha:
        return True
    else:
        return False

# Anderson-Darling Test
def anderson_darling(data):
    normal = 0
    non_normal = 0
    result = anderson(np.asarray(data), dist='norm')
    
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        print("Anderson Darling Test | Critical Value: {} | Statistic: {}".format(result.critical_values[i], result.statistic))
        if result.statistic < result.critical_values[i]:
            normal+=1
        else:
            non_normal+=1
    
    if normal >= non_normal:
        return True
    else:
        return False
                    