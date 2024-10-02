import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk


def lower_text(text: str):
    return text.lower()


def remove_punctuation(text: str):
    text_nopunct = re.sub('[^A-Za-z0-9\s]', '', text)
    return text_nopunct


def remove_multiple_spaces(text: str):
    text_no_doublespace = re.sub('\s+', ' ', text)
    return text_no_doublespace


def tokenize_text(text: str) -> list[str]:
    return word_tokenize(text)


def stem_words(tokenized_text: list[str]) -> list[str]:
    stemmer = WordNetLemmatizer()
    output = [stemmer.lemmatize(text) for text in tokenized_text]
    return output


def preprocessing_stage(text):
    _lowered = lower_text(text)
    _without_punct = remove_punctuation(_lowered)
    _single_spaced = remove_multiple_spaces(_without_punct)
    _tokenized = tokenize_text(_single_spaced)
    _stemmed = stem_words(_tokenized)
    _stemmed = ' '.join(_stemmed)
    
    return _stemmed