from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB  # chose for boolean values, it could be gaussian instead of it
from sklearn.metrics import confusion_matrix
import pandas as pd

main_data = pd.read_csv('bmi_info.csv')

x = main_data.iloc[:, 1:3].values  # independent variable
y = main_data.iloc[:, 3:].values  # dependent variable

# Split data as train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Model Building

# Chose the Bernoulli distribution algorithm to predict the binary(could be also say boolean) values like female or male
b_nb = BernoulliNB()

b_nb.fit(X_train, y_train.ravel())

y_pred = b_nb.predict(X_test)  # making prediction as always

"""
Naive Bayes  ->
    
    Bernoulli Naive Bayes   :   The naive Bayes training and classification algorithms for data 
                                that is distributed according to multivariate Bernoulli distributions. 
                                there may be multiple features but each one is assumed to be a binary-valued 
                                (Bernoulli, boolean) variable. 
    
    Gaussian Naive Bayes    :   If your data is increasing continuously you can implement
                                the Gaussian Naive Bayes algorithm for classification.
                            
    Multinomial Naive Bayes :   The naive Bayes algorithm for multinomially distributed data,
                                and is one of the two classic naive Bayes variants used in text classification
                                (where the data are typically represented as word vector counts, although 
                                tf-idf vectors are also known to work well in practice).
   
for more information Complement and Out-of-core Naive Bayes methods visit the url below
https://scikit-learn.org/stable/modules/naive_bayes.html          
"""

# Outputting on the console to examine our prediction

for i in range(8):
    print('predicted value -> ', y_pred[i], 'original value -> ', y_test[i])

# Evaluating with Confusion Matrix

# Definitions -> a, b, c, d

# a -> the count of that we predict female and the result is female
# b -> the count of that we predict female and the result is not female
# c -> the count of that we predict male and the result is female
# d -> the count of that we predict male and the result is male

# [True Positive('a' ), False Negative('b']
# [False Positive('c'), True Negative('d')]

cm = confusion_matrix(y_pred, y_test)
print(cm)
