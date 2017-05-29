import urllib2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict
import nltk
from sklearn.neighbors import KNeighborsClassifier

def getPosts(url, links):
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response)
    for a in soup.findAll('a'):
        try:
            url = a['href']
            title = a['title']
            if title == 'Older Posts':
                print title, url
                links.append(url)
                getPosts(url, links)
        except:
            title = ""
    return


blogURL = "http://doxydonkey.blogspor.in"
links = []
getPosts(blogURL, links)


def getText(testURL):
    request = urllib2.Request(testURL)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response)
    mydivs = soup.findAll("div", {"class": "post-body"})

    posts = []
    for div in mydivs:
        posts += map(lambda p:p.text.encode('ascii', errors= 'replace').replace("?", " "), div.findAll("li"))

    return posts

posts =[]
for link in links:
    posts += getText(link)




vectorizer = TfidfVectorizer(max_df=4, min_df=2, stop_words='english')

x = vectorizer.fit_transform(posts)

km = KMeans(n_clusters = 3, init = 'k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(x)

np.unique(km.labels_, return_counts= True)

text ={}
for i,cluster in enumerate(km.labels_):
    oneDocument = posts[i]
    if cluster not in text.keys:
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument


_stopwords = set(stopwords.words('english') + list(punctuation) + ["million", "billion", "year", "millions", "billions", "y/y", "'s", "''"])


keywords = {}
counts = {}

for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, key = freq.get)
    counts[cluster]=freq

unique_keys = {}

for cluster in range(3):
    other_clusters = list(set(range(3))- set([cluster]))
    keys_other_clusters = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique= set(keywords[cluster])- keys_other_clusters
    unique_keys[cluster]=nlargest(10, unique, key=counts[cluster].get)



article = "After a story published late last week about Tesla in the Guardian, investors might zero in on one comment in particular that CEO Elon Musk made: He said the company's 'market cap is higher than we have any right to deserve' and that building an auto-maker from scratch is 'the worst way to earn money, honestly. (A company spokesman declined to comment further. In doing so, Musk displayed what some analysts call his trademark unscripted style, saying that he slept in a sleeping bag on the floor of the factory, knowing people were having a hard time, working long hours, and on hard jobs. And in response to accusations the company was putting production before people, Musk said this is not some situation where, for example, we are just greedy capitalists who decided to skimp on safety in order to have more profits and dividends. It’s just a question of how much money we lose. And how do we survive? How do we not die and have everyone lose their jobs? Musk was speaking in response to a story that spoke with 15 current and former factory workers from its Fremont, Calif., plant that employs 10,000 workers, who described a culture of long hours, working through pain and even workers collapsing on the factory floor. The publication reported that its official measure of injuries and illnesses was higher than the industry average between 2013 and 2016, but the company responded that the numbers had improved -- its safety record is now reportedly 32 percent below average -- and has added a third shift to cut down on hours, as well as a team of ergonomics experts. As part of his response, Musk said that we're doing this because we believe in a sustainable energy future, trying to accelerate the advent of clean transport and clean energy production, not because we think this is a way to get rich. And he repeated a line similar to one he's said before. I do believe this market cap is higher than we have any right to deserve amid the company's 43 percent stock surge since the beginning of the year, which in April brought its $51 billion market capitalization above that of General Motors or Ford, which named a new CEO Thursday to take on the Silicon Valley upstart. [Ford shuffles executive ranks in its race to take on Silicon Valley] Analysts said it's unusual for CEOs to talk about their stock price that way, typically promoting its potential to go higher. In my experience, in over 10 years covering stocks and clean energy, I’ve never heard another CEO say that, said Ben Kallo, an analyst at Robert W. Baird who has an outperform rating on Tesla's stock. It's not a first for Musk, either: Twice in the second half of 2013, following a big run-up in the stock, he said the value was higher than the company deserved; a year later, he made a similar quip, saying our stock price is kind of high right now to be totally honest. Kallo said it may be a sign Musk is trying to temper expectations, as well as be straightforward with the market and let people know they have a tough road ahead of them. The company has said it intends to expand its production goals five-fold, from the 80,000 it produced last year to 500,000 in 2018. Usually a CEO is trying to do the opposite, especially when you have capital needs. Laura Rittenhouse, who runs an eponymous firm that evaluates the candor of CEOs and corporations, says it’s a smart CEO who realizes that if the stock gets overheated and people are disappointed, there’s a whole set of other problems to deal with. It's not unique, however. Back in 2013, for instance, Netflix CEO Reed Hastings sounded a warning about his company's stock, saying in a letter to shareholders that some of the euphoria today feels like 2003. [Tesla's crazy climb to America's most valuable car company] Toni Sacconaghi, an analyst with Bernstein Research, said Musk's remarks about his company's valuation probably weren't a deliberate signal to investors. I chalk it up more to Elon's fairly unscripted nature. If you listen to Tesla earnings conference calls, Elon just starts with questions, he said, rather than  a prepared statement. He says whatever’s on his mind."


classifier = KNeighborsClassifier()
classifier.fit(x, km.labels_)

test = vectorizer.transform([article.decode('utf8').encode('ascii', errors='ignore')])

classifier.predict(test)









