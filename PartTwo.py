import pandas as pd # sort and organise data

from pathlib import Path # to access files in other directories
from sklearn.feature_extraction.text import TfidfVectorizer # for 2b)
from sklearn.model_selection import train_test_split # for 2b)
from sklearn.ensemble import RandomForestClassifier # for 2c)
from sklearn.svm import LinearSVC # for 2c)
from sklearn.metrics import f1_score, classification_report # for 2c)

def read_csv(csv_path=Path.cwd() / "p2-texts" / "hansard40000.csv"): # 2a) Read the handsard40000.csv dataset in the texts directory into a dataframe. Sub-set and rename the dataframe as follows:
    # check path works
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist!")
        return None

    df = pd.read_csv(csv_path)

    # 2a) i. rename the 'Labour (Co-op)' value in party column to 'Labour', and then:
    df['party'] = df ['party'].replace('Labour (Co-op)', 'Labour')

    # 2a) ii. remove any rows where the value of the 'party' column is not one of the four most common party names, and remove the 'Speaker' value.
    df = df[df['party'] != 'Speaker']
    most_common_parties = df['party'].value_counts().head(4).index.tolist()
    df = df[df['party'].isin(most_common_parties)]

    # 2a) iii. remove any rows where the value in the 'speech_class' column is not 'Speech'
    df = df[df['speech_class'] == 'Speech']

    # 2a) iv. remove any rows where the text in the 'speech' column in less than 1000 characters long.
    df = df[df['speech'].str.len() >= 1000]

    print(df.shape)
    return df # show original df

def vectorise_speeches(): # 2b) Vectorise the speeches using TfidfVectorizer from scikit-learn. Use the default parameters, except for omitting English stopwords and setting max_features to 3000. Split the data into a train and test set, using sampling witrh a random seed of 26.
    df = read_csv()

    if df is None or df.empty:
        print("Error: Data not found")
        return None

    # get features and labels using approach learned in scikit-learn in lab 4
    X = df['speech']
    y = df['party']

    # TfidfVectorizer
    vectoriser = TfidfVectorizer(stop_words = 'english', # omitting English stopwords
                                 max_features = 3000) # max_features set to 3000

    # vectorise speeches
    X_vectorised = vectoriser.fit_transform(X)

    # split data into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_vectorised,
                                                        y,
                                                        test_size = 0.25,
                                                        stratify = y,
                                                        random_state= 26) # random seed 26

    return X_train, X_test, y_train, y_test, vectoriser

def classifier_train(): # 2c) Train RandomForest (with n_estimators = 300) and SVM with linear kernel classifiers on the training set, and print the scikit-learn macro-average f1 score and classification report for each classifier on the test set. The label that you are trying to predict is the 'party' value.
    result = vectorise_speeches()

    if result is None:
        return None

    X_train, X_test, y_train, y_test, vectoriser = result

    classifiers = [(RandomForestClassifier(n_estimators = 300), "Random Forest"), # random forest with n_estimators at 300
                   (LinearSVC(), "SVM with Linear Kernel")]

    # train the classifiers
    for clf, name in classifiers:
        clf.fit(X_train, y_train)

        # predictions for party
        y_pred = clf.predict(X_test)

        # calc and print the scikit-learn macro-average f1 score and classification reports
        macro_f1 = f1_score(y_test, y_pred, average = 'macro')
        print(f"Marco-average f1 score: {macro_f1}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Testing for 2a) - PASSED
    # print("Testing 2a")
    # print("-" * 30)
    # df = read_csv()

    # Testing for 2b) - PASSED
    # print("Testing 2b")
    # print("-" * 30)
    # result = vectorise_speeches()
    # if result is not None:
    #     X_train, X_test, y_train, y_test, vectoriser = result
    #     print("Passed")
    #     print(f"Training set shape: {X_train.shape}")
    #     print(f"Test set shape: {X_test.shape}")
    #     print(f"Training labels shape: {y_train.shape}")
    #     print(f"Test labels shape: {y_test.shape}")
    #     print(f"First 10 names: {vectoriser.get_feature_names_out()[:10]}")
    #     print(f"Unique parties in train: {y_train.unique()}")
    # else:
    #     print("Error: vectoriser failed")

    # Testing for 2c) - PENDING
    print("Testing 2c")
    print("-" * 30)

    classifier_train()

    print("\n" + "-" * 30)
    print("Passed")
