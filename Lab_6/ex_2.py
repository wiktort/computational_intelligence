import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def prepare_words(file_name: str):
    # tockenize
    with open('articles/{}'.format(file_name)) as article:
        text = article.readlines()
        text = "".join(text)
        text = text.lower()
    tokens = nltk.word_tokenize(text)

    # remove stop words and apply lematization
    stop_words = set(stopwords.words("english"))
    stop_words.update(["'s", "n't" "'re", "ii", ',', '.', '-', '–', '_', "''", '``'])
    lem = WordNetLemmatizer()

    ready_words = []
    for w in tokens:
        if w not in stop_words:
            ready_words.append(lem.lemmatize(w))

    return ready_words

article_file_names = [
    'Angkor-Asias-ancient-Hydraulic-City.txt',
    'Can-taiwan-become-asias-next-great-hiking-destination.txt',
    'The-innovative-technology-that-powerd-the-inca.txt'
]

prepared_words = []
for file_name in article_file_names:
    prepared_words.append(prepare_words(file_name))

# a - Document-Term Matrix (DTM)
prepared_articles = []
for words_set in prepared_words:
    prepared_articles.append(" ".join(words_set))

coun_vect = CountVectorizer()
count_matrix = coun_vect.fit_transform(prepared_articles)
count_array = count_matrix.toarray()
dtm = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())

print("Document-Term Matrix:\n{}\n".format(dtm))

# b - TF (Term Frequency) and TF-IDF (Term Frequency - Inverse Data Frequency)
all_words = dtm.columns
number_of_words = len(all_words)
number_of_articles = len(prepared_articles)

# tf
tf_array = []
for row in count_array:
    tf_row = []
    for word_count in row:
        tf_row.append(round(word_count/number_of_words, 6))
    tf_array.append(tf_row)
tf = pd.DataFrame(data=tf_array, columns=all_words)
print("Term Frequency Matrix:\n{}\n".format(tf))

# idf
idf_array = []
for word in all_words:
    number_of_articles_with_word = 0
    for word_count in dtm[word]:
        number_of_articles_with_word += (word_count / (word_count if word_count > 0 else 1))
    idf = math.log(number_of_articles / number_of_articles_with_word)
    idf_array.append([word, idf])
idf = pd.DataFrame(data=idf_array, columns=['Word', 'IDF'])
print("Inverse Data Frequency Matrix:\n{}\n".format(idf))

# tfidf
tfidf_array = []
for row in tf_array:
    tfidf_row = []
    for index in range(0, number_of_words):
        tfidf_value = row[index] * idf_array[index][1]
        tfidf_row.append(tfidf_value)
    tfidf_array.append(tfidf_row)
tfidf = pd.DataFrame(data=tfidf_array, columns=all_words)
print("Term Frequency - Inverse Data Frequency Matrix:\n{}\n".format(tfidf))


# c - cosine similarity
def square_reduce(array: []):
    acc = 0
    for value in array:
        acc += math.pow(value, 2)
    return acc


def count_cosine_similarity(tf_1: [], tf_2: []):
    tf_1_sum = math.sqrt(square_reduce(tf_1))
    tf_2_sum = math.sqrt(square_reduce(tf_2))
    product_sum = 0
    for tf_index in range(0, len(tf_1)):
        product_sum += (tf_1[tf_index] * tf_2[tf_index])
    return product_sum / (tf_1_sum * tf_2_sum)


cosine_sim_array = []
for index in range(0, number_of_articles):
    article_name = article_file_names[index].removesuffix(".txt")
    for index_2 in range(0, number_of_articles):
        if index == index_2:
            break
        article_2_name = article_file_names[index_2].removesuffix(".txt")
        similarity_name = " & ".join(sorted([article_name, article_2_name]))
        similarity_value = count_cosine_similarity(tf_array[index], tf_array[index_2])
        cosine_sim_array.append([similarity_name, similarity_value])

cosine_similarity = pd.DataFrame(data=cosine_sim_array, columns=["Document pairs", "Cosine similarity"])
print("Cosine Similarity Matrix:\n{}\n".format(cosine_similarity))



