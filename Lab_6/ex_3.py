from nltk.sentiment.vader import SentimentIntensityAnalyzer

with open('reviews/bad_review.txt') as file:
    bad_review = file.readlines()[0]

with open('reviews/good_review.txt') as file:
    good_review = file.readlines()[0]


sid = SentimentIntensityAnalyzer()

# bad review
ss_bad = sid.polarity_scores(bad_review)
print('\nBad review scores:')
for k in sorted(ss_bad):
    print('{0}: {1}, \n'.format(k, ss_bad[k]), end='')

# good review
ss_good = sid.polarity_scores(good_review)
print("\nGood review scores:")
for k in sorted(ss_good):
    print('{0}: {1}, \n'.format(k, ss_good[k]), end='')


print('\nShort sentence test:')
ss_1 = sid.polarity_scores("VADER is smart, handsome, and funny.")
print("\nPositive short sentence:")
print("VADER is smart, handsome, and funny.")
for k in sorted(ss_1):
    print('{0}: {1}, \n'.format(k, ss_1[k]), end='')


ss_2 = sid.polarity_scores("Today SUX!")
print("\nNegative short sentence:")
print("Today SUX!")
for k in sorted(ss_2):
    print('{0}: {1}, \n'.format(k, ss_2[k]), end='')