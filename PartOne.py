#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk                # import Natural Language Toolkit
import spacy               # import spaCy
from pathlib import Path   # to access files in other directories
import pandas as pd        # sort and organise data
import glob                # to locate specific file type
import os                  # to look up operating system info

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass

# 1a) i. create a pandas dataframe with the following columns: text, title, author, year
# 1a) ii. sort the dataframe by the year column before returning it, resetting or ignoring the dataframe index
def read_novels(path=Path.cwd() / "texts" / "novels"):

    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""

    # get file path & use glob to find .txt files in said path
    dirpath = r'C:\Users\karin\OneDrive - Birkbeck, University of London\5. Natural Language Processing\Coursework\p1-novels'
    txt_files = glob.glob(os.path.join(dirpath, '*.txt'))

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

    pass

# test for read_novels
if __name__ == "__main__":
    print("reading novels...")
    df = read_novels()
    print("done")

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    pass


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

