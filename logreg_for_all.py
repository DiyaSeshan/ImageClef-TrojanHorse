# import numpy as np
# import pandas as pd
# from sklearn import metrics
# from sklearn import linear_model
# from sklearn import svm
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, f1_score
# df = pd.read_csv(r'filepathorientation\orientation-all-data.tsv',sep='\t', encoding='ISO-8859-1')
# #print(df)
# df['text_en'] = df['text_en'].str.lower()
# X = np.array(df['text_en'])
# Y = np.array(df['label'])
# df.dropna(how="any")
# logreg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')
# v = CountVectorizer()
# # Fit the vectorizer with the training data and transform both training and testing data
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# x_train_transformed = v.fit_transform(x_train)
# x_test_transformed = v.transform(x_test)

# # Fit the logistic regression model with the transformed training data
# logreg.fit(x_train_transformed, y_train)

# # Make predictions on the transformed training data to check training accuracy
# yhat_train = logreg.predict(x_train_transformed)

# print("Training Predictions:")
# #print(yhat_train)

# print("Training Accuracy:")
# print(metrics.accuracy_score(yhat_train, y_train))

# print("Training Confusion Matrix:")
# print(metrics.confusion_matrix(yhat_train, y_train))

# # Make predictions on the transformed test data to check generalization
# yhat_test = logreg.predict(x_test_transformed)

# print("Test Predictions:")
# #print(yhat_test)

# print("Test Accuracy:")
# print(metrics.accuracy_score(yhat_test, y_test))

# print("Test Confusion Matrix:")
# print(metrics.confusion_matrix(yhat_test, y_test))

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
l2=["at", "ba", "be", "bg", "cz", "dk",
                        "es-ct", "es-ga","es-pv", "es", "fi", "fr",
                        "gb", "gr", "hr", "hu", "it", "lv",
                        "nl", "pl", "pt", "rs", "si",
                        "tr", "ua"]
l4=[ "fi", "fr", "gb", "gr", "hr", "hu", "it", "lv",
                        "nl", "pl", "pt", "rs", "si",
                        "tr", "ua"]
#df=pd.read_csv()
l3=[]
for i in l2:
    print(i)
    df=pd.read_csv(r'filepath\power-'+i+'-train.tsv',sep='\t')
    df['text_en'] = df['text_en'].str.lower()
    print(len(df))
    df=df.drop(df[df['text_en'].isnull()].index)
    logreg = LogisticRegression(C=0.3,max_iter=300)#solver='liblinear',
    v = CountVectorizer()
    x_train = np.array(df['text_en'])
    y_train = np.array(df['label'])
    X_train = v.fit_transform(np.array(x_train))
    #y_train = v.fit_transform(y_train)
    df.dropna(how="any",inplace=True)
    logreg.fit(X_train, y_train)

    yhat_train = logreg.predict(X_train)

    print(yhat_train)
    print("Training Accuracy:\n"+ str(metrics.accuracy_score(yhat_train, y_train)))
    print("Training Confusion Matrix: \n"+ str(metrics.confusion_matrix(yhat_train, y_train)))
    train_precision = precision_score(y_train, yhat_train, average='macro')
    train_recall = recall_score(y_train, yhat_train, average='macro')
    train_f1 = f1_score(y_train, yhat_train, average='macro')

    print("Training Precision (Macro):", train_precision)
    print("Training Recall (Macro):", train_recall)
    print("Training F1-Score (Macro):", train_f1)
    df_test=pd.read_csv(r'filepath\power-'+i+'-test.tsv',sep='\t')
    df_test=df_test.drop(df_test[df_test['text_en'].isnull()].index)
    df_test['text_en'] = df_test['text_en'].str.lower()
    x_test = np.array(df_test['text_en'])
    x_test = v.transform(x_test)
    yhat_test = logreg.predict(x_test)
    print("Test data predictions:")
    print(yhat_test[15])
    df_test['logreg']=yhat_test
    df_test.to_csv(r'filepath\power-'+i+'-pred_v10.tsv',sep='\t')
    print("done")

avg=0
coun=0
for i in l2:
    print(i)
    coun+=1
    df=pd.read_csv(r'filepath\power-'+i+'-train.tsv',sep='\t')
    avg+=len(df)
    print(f"Length of df{coun}: ",len(df))
print("Overall average:",avg/coun)
# l2=["at", "ba", "be", "bg", "cz", "dk",
#                         "es-ct", "es-ga","es-pv", "es", "fi", "fr",
#                         "gb", "gr", "hr", "hu", "it", "lv",
#                         "nl", "pl", "pt", "rs", "si",
#                         "tr", "ua"]
# l4=[ "fi", "fr", "gb", "gr", "hr", "hu", "it", "lv",
#                         "nl", "pl", "pt", "rs", "si",
#                         "tr", "ua"]
# l3=[]
# for i in l2:
#     print(i)
#     df=pd.read_csv(r'path\power-'+i+'-train.tsv',sep='\t')
#     df['text_en'] = df['text_en'].str.lower()
#     df=df.drop(df[df['text_en'].isnull()].index)
#     df.dropna(how="any",inplace=True)
#     logreg = LogisticRegression()
#     v = CountVectorizer()
#     X=np.array(df['text_en'])
#     Y=np.array(df['label'])
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
#     # x_train = np.array(df['text_en'])
#     # y_train = np.array(df['label'])
#     X_train = v.fit_transform(np.array(x_train))
#     X_test = v.fit_transform(np.array(x_test))
#     #y_train = v.fit_transform(y_train)
#     df.dropna(how="any",inplace=True)
#     logreg.fit(X_train, y_train)
#     #x_test=pd.read_csv(r'')
#     yhat_train = logreg.predict(X_test)

#     print(yhat_train)
#     print("Training Accuracy:\n"+ str(metrics.accuracy_score(yhat_train, y_test)))
#     print("Training Confusion Matrix: \n"+ str(metrics.confusion_matrix(yhat_train, y_test)))
#     train_precision = precision_score(y_test, yhat_train, average='macro')
#     train_recall = recall_score(y_train, yhat_test, average='macro')
#     train_f1 = f1_score(y_test, yhat_train, average='macro')

#     print("Training Precision (Macro):", train_precision)
#     print("Training Recall (Macro):", train_recall)
#     print("Training F1-Score (Macro):", train_f1)
#     df_test=pd.read_csv(rpathpower-'+i+'-test.tsv',sep='\t')
#     df_test=df_test.drop(df_test[df_test['text_en'].isnull()].index)
#     df_test['text_en'] = df_test['text_en'].str.lower()
#     x_test = np.array(df_test['text_en'])
#     x_test = v.transform(x_test)
#     yhat_test = logreg.predict(x_test)
#     print("Test data predictions:")
#     print(yhat_test[15])
#     df_test['logreg']=yhat_test
#     df_test.to_csv(r'path\power-'+i+'-pred_v10.tsv',sep='\t')
#     print("done")


#"pathr\power-all-train.tsv"

df=pd.read_csv(r"filepath\power-all-train.tsv",sep='\t')
df['text_en'] = df['text_en'].str.lower()
print(len(df))
df=df.drop(df[df['text_en'].isnull()].index)
logreg = LogisticRegression(C=0.3,max_iter=300)#solver='liblinear',
v = CountVectorizer()
x_train = np.array(df['text_en'])
y_train = np.array(df['label'])
X_train = v.fit_transform(np.array(x_train))
y_train = v.fit_transform(y_train)
df.dropna(how="any",inplace=True)
logreg.fit(X_train, y_train)
l3=[]
trainavg=0
for i in l2:
    df2=pd.read_csv(r'filepath\power-'+i+'-train.tsv',sep='\t')
    df2=df2.drop(df2[df2['text_en'].isnull()].index)
    x_train2 = np.array(df2['text_en'])
    X_train2 = v.fit_transform(np.array(x_train2))
    print(i)
    yhat_train = logreg.predict(X_train2)
    print(yhat_train)
    print("Training Accuracy:\n"+ str(metrics.accuracy_score(yhat_train, y_train)))
    print("Training Confusion Matrix: \n"+ str(metrics.confusion_matrix(yhat_train, y_train)))
    train_precision = precision_score(y_train, yhat_train, average='macro')
    train_recall = recall_score(y_train, yhat_train, average='macro')
    train_f1 = f1_score(y_train, yhat_train, average='macro')

    print("Training Precision (Macro):", train_precision)
    print("Training Recall (Macro):", train_recall)
    print("Training F1-Score (Macro):", train_f1)
    trainavg+=train_f1
    # df_test=pd.read_csv(r'filepath\power-'+i+'-test.tsv',sep='\t')
    # df_test=df_test.drop(df_test[df_test['text_en'].isnull()].index)
    # df_test['text_en'] = df_test['text_en'].str.lower()
    # x_test = np.array(df_test['text_en'])
    # x_test = v.transform(x_test)
    # yhat_test = logreg.predict(x_test)
    # print("Test data predictions:")
    # print(yhat_test[15])
    # df_test['logreg']=yhat_test
    # df_test.to_csv(r'filepath\power-'+i+'-pred_v10.tsv',sep='\t')
    # print("done")
