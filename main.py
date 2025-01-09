import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

sns.set_style("darkgrid")  # set the style

# Import dataset
data = pd.read_csv("Breast_cancer_data.csv")  # read the csv file
# print(data.head(10))  # show the first 10 rows

# Basic EDA
# print(data["diagnosis"].hist())  # show the histmap

corr = data.iloc[:, :-1].corr(method="pearson")  # pearson method
cmap = sns.diverging_palette(250, 354, 80, 60, center="dark",
                             as_cmap=True)  # style
sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True,
            linewidths=.2)  # show the heatmap

# pick the relatively independent variables
data = data[["mean_radius", "mean_texture", "mean_smoothness", "diagnosis"]]
data.head(10)

# normal distribution or not
fig, axes = plt.subplot(1, 3, figsize=(18, 6), sharey=True)
sns.histplot(data, ax=axes[0], x="mean_radius", kde=True, color='r')
sns.histplot(data, ax=axes[1], x="mean_smoothness", kde=True, color='b')
sns.histplot(data, ax=axes[2], x="mean_texture", kde=True)


# calculate P(Y=y) for all possible y
def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y] == i]) / len(df))
    return prior


# Approach 1: Calculate P(X=x|Y=y) using Gaussian dist.
def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y] == label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val-mean)**2 / (2 * std**2)))
    return p_x_given_y


# Calculate P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) for all y and find the maximum
def naive_bayes_gaussian(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i],
                                                               x[i], Y,
                                                               labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)


# test Gaussian model

train, test = train_test_split(data, test_size=.2, random_state=41)
X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes_gaussian(train, X=X_test, Y="diagnosis")

print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))

