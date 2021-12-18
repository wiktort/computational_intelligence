import operator
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# b - tockenize
with open('articles/Angkor-Asias-ancient-Hydraulic-City.txt') as bbc:
    text = bbc.readlines()
    text = "".join(text)
    text = text.lower()
tokens = nltk.word_tokenize(text)
print("number of tokens: {}".format(len(tokens)))

# c - remove stop words
stop_words = set(stopwords.words("english"))
filtered_words = []
for w in tokens:
    if w not in stop_words:
        filtered_words.append(w)
print("Filtered words (without stop words): {}".format(len(filtered_words)))

# d - add extra words to remove
stop_words_manual = set(["'s", "n't" "'re", "ii"])
filtered_words_2 = []
for w in filtered_words:
    if w not in stop_words_manual and not all(not char.isalnum() for char in w):
        filtered_words_2.append(w)
print("Filtered words (without manual stop words and symbol based words): {}".format(len(filtered_words_2)))

# e - lematization
lem = WordNetLemmatizer()
lematized_words = []
for w in filtered_words_2:
    lematized_words.append(lem.lemmatize(w))

# f - print barChart
dictionary = dict(Counter(lematized_words))
sortedWords = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
y = []
x = []
for i in range(10):
    y.append(sortedWords[i][1])
    x.append(sortedWords[i][0])
plt.figure(1)
plt.bar(x, y)
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)


# g
wordcloud = WordCloud(background_color="white").generate(" ".join(lematized_words))
plt.figure(2)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.show()