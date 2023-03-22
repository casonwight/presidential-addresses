"""
This script is intended to take a Speeches object
and create a SpeechFeatures object that contains
features for each speech.

The SpeechFeatures object will have the following
features:

- tfidf scores
- average word2vec embedding
- bert embedding
- instructor embedding
- sentiment analysis
- other predetermined features (number of words, number of sentences, etc.)

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

class SpeechFeatures:
    def __init__(self, speeches):
        self.speeches = speeches
        self.models = self.load_models(self.speeches.speeches_long)

    def load_models(self, speech_text: pd.Series):
        all_models = {
            'tfidf': TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 4)).fit(speech_text),
            'word2vec': spacy.load('en_core_web_sm'),
            'sentence2vec': SentenceTransformer('all-MiniLM-L6-v2'),
            'instructor': INSTRUCTOR('hkunlp/instructor-base')
        }
        return all_models

    def get_tfidf_features(self, speech_text: pd.Series):
        tfidf_features = self.models['tfidf'].transform(speech_text).toarray()
        tfidf_features = pd.DataFrame(tfidf_features, self.models['tfidf'].get_feature_names_out())
        return tfidf_features

    def get_word2vec_features(self, speech_text: pd.Series):
        word2vec_features = np.array([self.models['word2vec'](speech).vector for speech in speech_text])
        num_cols = word2vec_features.shape[1]
        word2vec_features = pd.DataFrame(word2vec_features, columns=[f"word2vec_{i}" for i in range(num_cols)])
        return word2vec_features
    
    def get_sentence2vec_features(self, speech_text: pd.Series):
        sentence2vec_features = self.models['sentence2vec'].encode(speech_text)
        num_cols = sentence2vec_features.shape[1]
        sentence2vec_features = pd.DataFrame(sentence2vec_features, columns=[f"sentence2vec_{i}" for i in range(num_cols)])
        return sentence2vec_features
    
    def get_instructor_features(self, speech_text: pd.Series):
        instruction = "Represent the presidential speech:"
        instructor_features = self.models['instructor'].encode([[instruction, x] for x in speech_text])
        num_cols = instructor_features.shape[1]
        instructor_features = pd.DataFrame(instructor_features, columns=[f"instructor_{i}" for i in range(num_cols)])
        return instructor_features

    def get_features(self, long=True, feature_type="tfidf", **kwargs):
        speech_text = self.speeches.speeches_long['text'] if long else self.speeches.speeches['transcript']

        if feature_type == "tfidf":
            features = self.get_tfidf_features(speech_text)
        elif feature_type == "word2vec":
            features = self.get_word2vec_features(speech_text)
        elif feature_type == "sentence2vec":
            features = self.get_sentence2vec_features(speech_text)
        elif feature_type == "instructor":
            features = self.get_instructor_features(speech_text)
        else:
            raise ValueError("feature_type must be one of tfidf, word2vec, sentence2vec, or instructor")

        return features
    

if __name__ == "__main__":
    from speeches import Speeches

    speeches = Speeches()
    print(speeches.speeches)
    print(speeches.speeches_long)
    
    speech_features = SpeechFeatures(speeches)
    print(speech_features.get_features(long=True, feature_type="tfidf"))