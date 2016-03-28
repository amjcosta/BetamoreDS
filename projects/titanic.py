# 1. Read titanic.csv into a data frame

import pandas as pd
file = "/Users/amy/Betamore/BetamoreDS/data/titanic.csv"
data = pd.read_csv(file, index_col='PassengerId')


# 2. Define Pclass and Parch as features, Survived as response

features = ['Pclass', 'Parch']
X = data[features]
y = data.Survived

# 3. Split data into training and test sets.

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 4. Fit a logistic regression and examine coefficients.

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

zip(features, logreg.coef_[0])
# returns => [('Pclass', -0.83366726386774226), ('Parch', 0.24449218857505925)]
# Interpretation:
# -- a 1 unit increase in class decreases odds of survival by 0.83 units
#    (i.e. the poorer classes are more likely to die)
# -- a 1 unit increase in parents and children increases odds of survival
#    by 0.24 units
#    (i.e. families are more likely to survive)

# 5. Create a confusion matrix and document sensitivity and specificity.

from sklearn import metrics
preds = logreg.predict(X_test)
print metrics.confusion_matrix(y_test, preds)
# returns => [[461  88]
# 			  [190 152]]
# interpretation: not very accurate. 190 false positives versus 461 true positives;
# 88 false negatives versus 152 true negatives.
# TODO: actually fit the model next time :)

