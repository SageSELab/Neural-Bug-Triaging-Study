import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

project_name = "mozilla_core_fold"
n_components = 3000

for f in range(1,11):
    df_train = pd.read_csv(f"../Data/{project_name}{f}_train.csv")
    df_test = pd.read_csv(f"../Data/{project_name}{f}_test.csv")

    tranin_dev = df_train['Component'].values
    test_dev = df_test['Component'].values

    not_present_in_training = [i for i in test_dev if i not in tranin_dev]
    not_present_in_test = [i for i in tranin_dev if i not in test_dev]

    df_test = df_test[~df_test['Component'].isin(not_present_in_training)]
    df_train = df_train[~df_train['Component'].isin(not_present_in_test)]

    unique_assign_to = dict([(v, k) for k,v  in zip(range(0, len(df_train["Component"].unique())), df_train["Component"].unique())])

    df_train['label'] = df_train['Component'].map(lambda x: unique_assign_to[x])
    df_test['label'] = df_test['Component'].map(lambda x: unique_assign_to[x])

    vectorizer = TfidfVectorizer(tokenizer= lambda x: x.split(' ') , ngram_range=(1, 2), dtype=np.float32)

    train_tfidf = vectorizer.fit_transform(df_train['text'].values)
    test_tfidf = vectorizer.transform(df_test['text'].values)

    while True:
        svd = TruncatedSVD(n_components=n_components)
        train_svd = svd.fit_transform(train_tfidf)
        if sum(svd.explained_variance_ratio_) > 0.95:
            break
        n_components += 3000
    
    svd = TruncatedSVD(n_components=n_components)
    train_svd = svd.fit_transform(train_tfidf)
    
    print(f"Fold {f} var: {sum(svd.explained_variance_ratio_)}")
    
    test_svd = svd.transform(test_tfidf)

    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(train_svd, df_train['label'].values)

    pred = svm_classifier.predict_proba(test_svd)
    np.save(f"pred_mozilla_core_svm_fold{f}.npy", pred)