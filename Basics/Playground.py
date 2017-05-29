import nltk

text = "Onur had a little cat. His paws were white as snow"


#TOKENIZATION


# functions to break text into sentences or words
from nltk.tokenize import word_tokenize, sent_tokenize


# break text into sentences and store it in a list
sents = sent_tokenize(text)


# break text into words
words = [word_tokenize(sent) for sent in sents]


# REMOVING STOPWORDS

from nltk.corpus import stopwords
from string import punctuation


#creating set of stopwords in english and a list of punctutations
customStopwords = set(stopwords.words('english')+ list(punctuation))

wordsWOStopWords = [word for word in word_tokenize(text) if word not in customStopwords]


# IDENTIFYING BIGRAMS

from nltk.collocations import *
bigram_measures = nltk.collacations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWOStopWords)

# Distinct bigrams and their frequencies sorted by their frequencies
sorted(finder.ngram_fd.items())

# STEMMING AND PARTS OF SPEECH TAGGING

text2 = "Onur closed on closing night when he was in the mood to close."

from nltk.stem.lancaster import LancasterStemmer



# Returns all stemmed words
st = LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]


# Assign related part of speech
nltk.pos_tag(word_tokenize(text2))

# WORD SENSE DISAMBIGUATION

from nltk.corpus import wordnet as wn
for ss in wn.synsets('bass'):
    print(ss, ss.definition())

from nltk.wsd import lesk

sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass"), 'bass')
sense2 = lesk(word_tokenize("This sea bass was really har dto catch"), 'bass')









