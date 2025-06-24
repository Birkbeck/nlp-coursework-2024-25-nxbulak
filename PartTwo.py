import pandas as pd # sort and organise data
import re # for regular expressions

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

def vectorise_speeches(ngram_range = (1,1)): # 2b) Vectorise the speeches using TfidfVectorizer from scikit-learn. Use the default parameters, except for omitting English stopwords and setting max_features to 3000. Split the data into a train and test set, using sampling witrh a random seed of 26.
    df = read_csv()

    if df is None or df.empty:
        print("Error: Data not found")
        return None

    # get features and labels using approach learned in scikit-learn in lab 4
    X = df['speech']
    y = df['party']

    # TfidfVectorizer
    vectoriser = TfidfVectorizer(stop_words = 'english', # omitting English stopwords
                                 max_features = 3000, # max_features set to 3000
                                 ngram_range = ngram_range) # 2d) adjust the parameters of the Tfidfvectorizer so that unigrams, bi-grams and tri-grams will be considered features

    # vectorise speeches
    X_vectorised = vectoriser.fit_transform(X)

    # split data into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_vectorised,
                                                        y,
                                                        test_size = 0.25,
                                                        stratify = y,
                                                        random_state= 26) # random seed 26

    return X_train, X_test, y_train, y_test, vectoriser

def classifier_train(ngram_range = (1,1)): # 2c) Train RandomForest (with n_estimators = 300) and SVM with linear kernel classifiers on the training set, and print the scikit-learn macro-average f1 score and classification report for each classifier on the test set. The label that you are trying to predict is the 'party' value.
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

def political_phrases(): # for better context when implementing funct for 2e)
    df = read_csv()

    if df is None or df.empty:
        print("Error: Data not found")
        return []

    # TF-IDF to find prominent phrases
    phrase_vectoriser = TfidfVectorizer(ngram_range = (2,3), # two or three word phrases
                                        max_features = 30, # max 30 phrases
                                        stop_words = 'english',
                                        min_df = 5) # min 5 times for consistency

    # fit all speeches
    phrase_vectoriser.fit(df['speech'])

    # get prominent phrases
    prominent_phrases = phrase_vectoriser.get_feature_names_out()

    return prominent_phrases.tolist()

def custom_tokenizer(text): # 2e) Implement a new custom tokenizer and pass it to the tokenizer argument of TfidfVectorizer.

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

    # Testing for 2c) - PASSED
    # print("Testing 2c")
    # print("-" * 30)
    # classifier_train(ngram_range = (1,1))
    # print("\n" + "-" * 30)
    # print("Passed")

    # Testing for 2d) - PASSED
    # print("Testing 2d")
    # print("-" * 30)
    # classifier_train(ngram_range = (1,3))

    # Testing political_phrases - PASSED
    # print("Testing political_phrases")
    # print("-" * 30)
    #
    # phrases = political_phrases()
    #
    # if phrases:
    #     print(f"{len(phrases)} political phrases:")
    #     print()
    #     for i, phrase in enumerate(phrases, 1):
    #         print(f"{i:2d}. {phrase}")
    #
    #     print()
    #     print("Phrases for tokenizer:")
    #     print(phrases[:10])
    # else:
    #     print("No political phrases extracted")
    #
    # print("\n" + "-" * 30)
    # print("Passed")