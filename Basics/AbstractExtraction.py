### APPROACH
### Find the ost important words
### Compute a significance score for sentences based on words they contain
### Pick the most significant sentences
###
### CHOOSING IMPORTANT WORDS
### Authors tend to repeat the words that are important to the theme of the text
### compute the word frequency
###
### SENTENCE SIGNIFICANCE
### Sentences which encapsulate more of the important words are more significant
### Sentence significance score = sum(Word Importance)

import urllib
from bs4 import BeautifulSoup
import nltk
import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# construct a frequency distribution of words
from nltk.probability import FreqDist

from heapq import nlargest

from collections import defaultdict


nltk.download('punkt')
nltk.download('stopwords')


articleURL = "https://www.washingtonpost.com/world/national-security/jared-kushner-now-a-focus-in-russia-investigation/2017/05/25/f078db74-40c7-11e7-8c25-44d09ff5a4a8_story.html?utm_term=.d30239306acb"
def getTextWAPO(url):

    page = urllib.urlopen(url).read().decode('utf8')

    soup = BeautifulSoup(page, "lxml")

    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))

    return text.encode('ascii', errors='replace').replace("?", " ")


text = getTextWAPO(articleURL)


def summarize(text, n):


    sents = sent_tokenize(text)
    assert n <= len(sents)

    word_sent = word_tokenize(text.lower())

    _stopwords = set(stopwords.words('english')+ list(punctuation))

    word_sent = [word for word in word_sent if word not in _stopwords]

    freq = FreqDist(sorted(word_sent))

    ranking = defaultdict(int)

    for i, sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]


    #print(ranking)

    sent_idx = nlargest(n, ranking, key=ranking.get)

    for j in sorted(sent_idx):
        print(sents[j])



summarize(text, 3)





