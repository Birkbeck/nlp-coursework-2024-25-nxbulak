#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk                # import Natural Language Toolkit
import spacy               # import spaCy
import pandas as pd        # sort and organise data
import glob                # to locate specific file type
import os                  # to look up operating system info
import math                # mathematical functions (PMI)

from nltk import word_tokenize, sent_tokenize   # to split text into words and sentences
from pathlib import Path                        # to access files in other directories
from collections import Counter                 # to count frequencies of words

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def read_novels(path=Path.cwd() / "p1-texts" / "novels"): # 1a) i. create a pandas dataframe with the following columns: text, title, author, year
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""

    # get file path & use glob to find .txt files in said path
    txt_files = glob.glob(str(path / "*.txt"))

    # check path works ok
    print(f"Looking for novels in: {path}")
    print(f"Found {len(txt_files)} .txt files")

    data = []

    for txt_file in txt_files:
        filename = os.path.splitext(os.path.basename(txt_file))[0]
        parts = filename.split('-')

        # check print
        print(f"Processing: {filename}")
        print(f"Parts: {parts}")

        if len(parts) >= 3: # minimum text should contain title-author-year
            year_str = parts[-1] # taking last part (year)
            author = parts[-2] # taking second to last part (author)
            title_parts = parts[:-2] # title of novel
            title = '-'.join(title_parts).replace('_', ' ')

            try:
                year = int(year_str) # change year to int
                print(f"Year: {year}")

                with open(txt_file, 'r', encoding='utf-8') as file: # read file content
                    content = file.read()

                data.append({'text': content, 'title': title, 'author': author, 'year': year})
                print(f"added: {title} by {author} ({year})") # check all added

            except ValueError:
                print(f"Error: Year conversion failed for '{year_str}' to int for {filename}")
            except Exception as e:
                print(f"Error: Reading file name {filename} failed for {e}")

        else:
            print(f"Error: filename not parsed: {filename}")

    df = pd.DataFrame(data) # create data frame
    if not df.empty: # 1a) ii. sort the dataframe by the year column before returning it, resetting or ignoring the dataframe index
        df = df.sort_values('year').reset_index(drop=True)

    print(f"\nTotal novels loaded: {len(df)}")
    return df

# if __name__ == "__main__": # testing for path - PASSED
#     from pathlib import Path
#
#     # check path
#     novels_path = Path.cwd() / "p1-texts" / "novels"
#     print(f"Current working directory: {Path.cwd()}")
#     print(f"Looking for novels in: {novels_path}")
#     print(f"Path found: {novels_path.exists()}")
#     print(f"Path is in directory: {novels_path.is_dir()}")
#
#     # locate files
#     txt_files = list(novels_path.glob("*.txt"))
#     print(f"Found {len(txt_files)} .txt files")
#
#     # show first files found
#     for file in txt_files[:5]:
#         print(f" - {file.name}")
#
#     print("\n" + "-" * 30)
#     print("Test on read_novels funct:")
#     try:
#         df = read_novels()
#         print(f"Loaded {len(df)} novels")
#         print(df[['title', 'author', 'year']].head())
#     except Exception as e:
#         print(f"Error in read_novels funct: {e}")

# if __name__ == "__main__": # testing for question 1a) - PASSED
#     df = read_novels()
#     print(df.columns.tolist())
#     print(df[['title', 'author', 'year']].head())

def count_syl(word,
              d):  # 1c) This function should return a dictionary mapping the title of each novel to the Flesch-Kincaid reading grade level score of the text (this func: calc syllables for words)
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()  # convert to lowercase for consistency

    if word in d:  # try CMU dict first
        return len([p for p in d[word][0] if p[-1].isdigit()])  # count stress-marked phemones

    # count vowel groups when word not in dict
    vowels = 'aeiouy'
    count = 0
    prev_vowel = False  # check if prev character was a vowel

    for char in word:
        if char in vowels:  # only count as new syllable if prev character was not a vowel
            if not prev_vowel:
                count += 1
            prev_vowel = True
        else:
            prev_vowel = False  # consonants

    return max(1, count)  # minimum 1 syllable


def fk_level(text,
             d):  # 1c) This function should return a dictionary mapping the title of each novel to the Flesch-Kincaid reading grade level score of the text (this func: calc FK for single text)
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """

    sentences = sent_tokenize(text)  # split text into sentences
    words = [w for w in word_tokenize(text.lower()) if
             w.isalpha()]  # lower case for consistency and remove punct and numbers

    if len(sentences) == 0 or len(words) == 0:  # avoid / by zero
        return 0

    total_syllables = sum(count_syl(word, d) for word in words)  # count syllables across the words

    # calc averages for Flesch-Kincaid
    avg_sentence_length = len(words) / len(sentences)  # words per sentence
    avg_syllables = total_syllables / len(words)  # syllables per word

    return 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59  # Flesch-Kincaid Grade Level Formula - Confirmed by PN 19/06/2025


def get_fks(
        df):  # 1c) This function should return a dictionary mapping the title of each novel to the Flesch-Kincaid reading grade level score of the text (this func: helper to apply FK df)
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results

# if __name__ == '__main__':  # testing for question 1c) - PASSED
#
#     print("count_syl, fk_level and get_fks test commence")
#     print("-" * 30)
#
#     # download NLTK data & load CMU dict
#     nltk.download('cmudict', quiet=True)
#     nltk.download('punkt', quiet=True)
#     cmudict = nltk.corpus.cmudict.dict()
#
#     # test 1: count_syl
#     print("\n1. Testing count_syl:")
#     print("-" * 30)
#     test_words = ["hello", "beans", "lullaby"]
#     for word in test_words:
#         syllables = count_syl(word, cmudict)
#         print(f"'{word}' syllables: {syllables}")
#
#     # test 2: fk_level
#     print("\n2. Testing fk_level:")
#     print("-" * 30)
#     test_texts = ["The owl looked at the moon. It sighed!",
#                   "Wherever the river may flow, it will always lead back to the sea."]
#     for text in test_texts:
#         fk_score = fk_level(text, cmudict)
#         print(f"'{text}' fk_score: {fk_score:.4f}")
#
#     # test 3: get_fks
#     print("\n3. Testing get_fks:")
#     print("-" * 30)
#     try:
#         df = read_novels()
#         fk_results = get_fks(df)
#         print(f"Calc FK for {len(fk_results)} novels")
#
#         # first 3 results
#         count = 0
#         for title, fk in fk_results.items():
#             print(f"'{title}' fk: {fk}")
#             count += 1
#             if count >= 3:
#                 break
#         print(f"\n All fk funct passed")
#
#     except Exception as e:
#         print(f"Error: {e}")
#
#     print("-" * 30)

def nltk_ttr(text): # 1b) This function should return a dictionary mapping the title of each novel to its type-token ratio
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""

    tokens = word_tokenize(text.lower()) # tokenize using NLTK library only

    clean_tokens = [token for token in tokens if token.isalpha()] # remove punctuation and ignore case for counting types

    if len(clean_tokens) == 0: # precaution to avoid errors
        return 0

    # 1b) This function should return a dictionary mapping the title of each novel to its type-token ratio
    types = set(clean_tokens)
    ttr = len(types)/len(clean_tokens)
    return ttr


def get_ttrs(df): # 1b) This function should return a dictionary mapping the title of each novel to its type-token ratio
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results

# if __name__ == "__main__": # testing for question 1b) - PASSED
#     print("nltk_ttr & get_ttrs testing commence")
#
#     # test 1: nltk_ttr function - string test: PASSED
#     print("\n1. Testing nltk_ttr:")
#     print("-" * 30)
#
#     test_text1 = "The owl looked at the moon. It then sighed!"
#     result1 = nltk_ttr(test_text1)
#     print(f"Text: '{test_text1}'")
#     print(f"TTR: {result1}")
#     print(f"Expected: ~0.889 (8 unique / 9 total, no punctuation included)")
#
#     # test 2: get_ttrs function on novels
#     print("\n2. Testing get_ttrs:")
#     print("-" * 30)
#
#     try:
#         df = read_novels()
#         ttr_results = get_ttrs(df)
#         print(f"Calculated ttr results for {len(ttr_results)} novels")
#         print(f"Sample results:")
#
#         # show first  results
#         count = 0
#         for title, ttr in ttr_results.items():
#             print(f"{title}: {ttr:.4f}")
#             count += 1
#             if count >= 3:
#                 break
#
#     except Exception as e:
#         print(f"Error: {e}")
#
# print("\nTesting finished")

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"): # 1e) parse: The goal of this function is to process the texts with spaCy's tokenizer and parser, and store the processed texts.
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes
    the resulting  DataFrame to a pickle file"""

    # 1e) Load spaCy and max_length
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000

    # 1e) i. Use spaCy nlp method to add a new column to the dataframe that contains parsed and tokenized Doc objects for each text.
    print("Parsing and tokenizing texts with spaCy")
    parsed_docs = []
    for i, row in df.iterrows():
        print(f"Parsing {row['title']}")
        doc = nlp(row["text"])
        parsed_docs.append(doc)
    df["parsed"] = parsed_docs

    # 1e) ii. Serialise the resulting dataframe (i.e., write it out to disk) using the pickle format.
    store_path.mkdir(parents = True, exist_ok = True)
    output_path = store_path / out_name
    df.to_pickle(output_path)
    print(f"Dataframe serialised to: {output_path}")

    # 1e) iii. Return the dataframe.
    return df


def load_parsed_df(store_path=Path.cwd() / "pickles", pickle_name="parsed.pickle"): # 1e) iv. Load the dataframe from the pickle file and use it for the remainder of this coursework part. Note: one or more of the texts may exceed the default maximum length or parse the text in sections.
    pickle_path = store_path / pickle_name
    if pickle_path.exists():
        df = pd.read_pickle(pickle_path)
        print(f"Parsed dataframe loaded from {pickle_path}")
        print(f"Dataframe contains {len(df)} novels with parsed column")
        return df
    else:
        print(f"Error: Pickle file not found in {pickle_path}")
        return None

# if __name__ == "__main__": # testing for question 1e) - PASSED
#     print("parse test commence")
#     print("-" * 30)
#
#     # create df for test
#     test_data = {'text': ["The owl looked at the moon.", "Operation time!" ], 'title': ["Divergent", "Animal Farm"], 'author': ["Pukinskaite", "Sergeant"], 'year': [1854, 1954]}
#     test_df = pd.DataFrame(test_data)
#
#     try:
#         import tempfile
#         from pathlib import Path
#
#         # test parse funct
#         test_dir = Path(tempfile.mkdtemp())
#         result_df = parse(test_df, store_path=test_dir, out_name = "test.pickle")
#
#         # results
#         print(f"Parse complete: {len(result_df)} rows, {len(result_df.columns)} columns")
#         print(f"First doc has {len(result_df['parsed'].iloc[0])} tokens")
#         print(f"Pickle file exists: {(test_dir / 'test.pickle').exists()}")
#
#         # test load
#         loaded_df = load_parsed_df(store_path=test_dir, pickle_name="test.pickle")
#         print(f"Loaded complete: {len(loaded_df)} rows")
#
#     except Exception as e:
#         print(f"Error: {e}")
#
#     print("parse and load_parsed_df test passed")

def syntactic_objects(doc): # 1f) i. The title of each novel and a list of the ten most common syntactic objects overall in the text.
    """Extracts the most common syntactic objects overall in the text.""" # adjusted as per Moodle announcement.
    syn_objects = []

    for token in doc: # go through all tokens in parsed doc
        if token.dep_ == 'dobj': # capture direct objects
            syn_objects.append(token.lemma_.lower())

    syn_objects_count = Counter(syn_objects)
    return syn_objects_count.most_common(10)

def subjects_by_verb_count(doc, verb): # 1f) ii. The title of each novel and a list of the ten most common syntactic subjects of the veb 'to hear' (in any tense) in the text, ordered by their frequency.
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []

    for token in doc: # go through all tokens in parsed doc
        if token.lemma_.lower() == verb.lower(): # look for subjects related to verb
            for child in token.children: # tenses checked
                if child.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']:
                    subjects.append(child.lemma_.lower())

    subjects_count = Counter(subjects)
    return subjects_count.most_common(10)


def subjects_by_verb_pmi(doc, target_verb): # 1f) iii. The title of each novel and a list of the ten most common syntactic subjects of the verb 'to hear' (in any tense) in the text, ordered by their Pointwise Mutual Information.
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    verb_subject_pairs = []
    all_verbs = []
    all_subjects = []

    for token in doc: # collect all verbs
        if token.pos_ == 'VERB':
            all_verbs.append(token.lemma_.lower())
        if token.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']: # collect all subjects
            all_subjects.append(token.lemma_.lower())
            if token.head.lemma_.lower() == target_verb.lower(): # check subj related to target verb
                verb_subject_pairs.append((target_verb.lower(), token.lemma_.lower()))

    pair_counts = Counter(verb_subject_pairs)
    verb_counts = Counter(all_verbs)
    subject_counts = Counter(all_subjects)

    # totals for PMI
    total_pairs = len(verb_subject_pairs)
    total_verbs = len(all_verbs)
    total_subjects = len(all_subjects)

    # PMI for each subject with target verb
    pmi_scores = []
    target_verb_lower = target_verb.lower()

    for (verb, subject), pair_count in pair_counts.items():
        if verb == target_verb_lower and pair_count >= 1: # min
            # P(verb, subject)
            p_verb_subject = pair_count / total_pairs if total_pairs > 0 else 0

            # P(verb) and P(subject)
            p_verb = verb_counts[target_verb_lower] / total_verbs if total_verbs > 0 else 0
            p_subject = subject_counts[subject] / total_subjects if total_subjects > 0 else 0

            # PMI = log(P(verb, subject) / (P(verb * P(subject)))
            if p_verb > 0 and p_subject > 0 and p_verb_subject > 0:
                pmi = math.log(p_verb_subject / (p_verb * p_subject))
                pmi_scores.append((subject, pmi, pair_count))

    pmi_scores.sort(key = lambda x: x[1], reverse = True) # sort descending and return top 10

    return [(subject, count) for subject, pmi, count in pmi_scores[:10]]

if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" / "parsed.pickle")
    #print(syntactic_objects(df)) # leads to errors as it's calling a df - not required in question specified - changed to below instead

    for i, row in df.iterrows():
        print(row["title"])
        print(syntactic_objects(row["parsed"]))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

    print("Testing syntactic obj funct. with simple sentence")
    print("-" * 30)
    test_doc = nlp("She read the book. He saw her. They heard nothing.")
    for token in test_doc:
        if token.dep_ == 'dobj':
            print(f"Direct object: {token.text} (head: {token.head.text})")

