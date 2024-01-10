import numpy as np
import pandas as pd
import contractions

data = pd.read_csv(r"D:/Data Scientist/Unsupervised Learning/Text Mining & NLP/GitHub/Abstractive text summerization/news_summary.csv",encoding = "latin-1")

data.columns

data = data['text']

data = data.iloc[0:10]

text = ' '.join(data)

text = contractions.fix(text)

from autocorrect import Speller

def correct_spelling(word, spell_checker):
    return spell_checker(word)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
from nltk.stem import PorterStemmer, WordNetLemmatizer

stopwords = set(stopwords.words('english'))

def text_preprocessing(text):
    # Convert the text to lower case
    text = text.lower()
    
    # tokenize the text
    words = word_tokenize(text)
    
    # remove stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords]
    
    # remove punctuations
    filtered_words = [''.join(char for char in word if char not in punctuation)for word in filtered_words]
    
    # Initialize the spell checker
    spell = Speller(lang = 'en')
    corrected_words = [correct_spelling(word, spell) for word in filtered_words]
    
    # Apply stemming
   # stemmer = PorterStemmer()
   # stemmed_words = [stemmer.stem(word) for word in corrected_words]
    
    # Apply Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatize_words = [lemmatizer.lemmatize(word) for word in corrected_words]
    
    #join into string
    filtered_text = ' '.join(lemmatize_words)
    
    return filtered_text

preprocessed_text = text_preprocessing(text)

tokenized_words  = word_tokenize(preprocessed_text)

from nltk.probability import FreqDist

word_frequencies = FreqDist(tokenized_words )

max_frequency = max(word_frequencies.values())

# Normalize the frequency

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency
    
# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Function to calculate the sentence score
def calculate_sentence_score(sentence, word_frequencies):
    words = word_tokenize(sentence)
    score = sum(word_frequencies[word] for word in words if word in word_frequencies)
    return score

# Calculate the score for each sentence
sentence_scores = [calculate_sentence_score(sentence, word_frequencies) for sentence in sentences]

# Print the sentence scores
for i, score in enumerate(sentence_scores):
    print(f"Sentence {i + 1} Score: {score}")    

# Select the top N sentences for the summary
top_sentences = sorted(((sentence, score) for sentence, score in zip(sentences, sentence_scores)), key=lambda x: x[1], reverse=True)

# Choose the top N sentences (e.g., top 3)
top_n = 3
summary_sentences = top_sentences[:top_n]

# Extract only the sentences from the summary
summary = [sentence for sentence, score in summary_sentences]

after = ' '.join(summary)
print(after)

print('original text: ', len(text))
print('after summerization: ', len(after))



