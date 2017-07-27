import codecs, re

from editdistance import eval as distance
from random import shuffle
from difflib import SequenceMatcher
from sklearn.externals import joblib

f = codecs.open('corpus2.csv', encoding = 'utf-8-sig')
lines = f.read().split('\r\n')
f.close()

border = int(lines.pop(0))
marked = lines[:border]
not_marked = lines[border:]

pattern = re.compile('[0-9]+[а-я]?')
mask = lambda words: ['_NUMBER_' if pattern.fullmatch(word) else word for word in words]
tokenize = lambda x: mask(re.findall('[А-Яа-яЁё0-9A-Za-z_]+', x))
only_numbers = lambda query: re.findall('[0-9]+', ' '.join(query))

print("Checking the corpus for consistency")

if min([line.count(',') >= 3 or bool(print(line)) for line in marked]) == True:
	print("Corpus test 1 succeeded")
else:
	print("Corpus test 1 failed (!)")

if min([line.count(',') >= 1 or bool(print(line)) for line in not_marked]) == True:
	print("Corpus test 2 succeeded")
else:
	print("Corpus test 2 failed (!)")

marked_queries = [query.rsplit(',', 3) for query in marked]
tokenized_queries = [tokenize(query) for raw_query, query, suggestion, label in marked_queries]
for (counter, (raw_query, query, suggestion, label)) in enumerate(marked_queries):
	if SequenceMatcher(a = query, b = suggestion).find_longest_match(0, len(query) -1, 0, len(suggestion) -1).size < 3 and len(query) > 6 and suggestion != '<?????>' or (len(query) > 4 and len(suggestion) < 3) or suggestion in ['', 'окок', 'jr']:
		print('Corpus test 3 failed (!) at query "%s", suggestion %s, number %d' % (query, suggestion, counter))

print("Corpus test 3 finished")

for (counter, (raw_query, query, suggestion, label)) in enumerate(marked_queries):
	if int(query != suggestion) != int(label):
		print('Corpus test 4 failed (!) at query "%s", suggestion %s, label %s, number %d' % (query, suggestion, label, counter))

print("Corpus test 4 finished")

dic = {}
for raw_query, query, suggestion, label in marked_queries:
	if query in dic:
		if suggestion not in dic[query]:
			dic[query].append(suggestion)
	else:
		dic[query] = [suggestion]

for key in dic:
	if len(dic[key]) > 1:
		print("(!) Conflicting suggestions for %s:" % key, *dic[key])

print("Corpus test 5 finished")

for counter, (raw_query, query, suggestion, label) in enumerate(marked_queries):
	if only_numbers(query) != only_numbers(suggestion) and suggestion != '<?????>':
		print("Corpus test 6 failed (!) at query %s, suggestion %s, number %d" % (query, suggestion, counter))

print("Corpus test 6 has been completed.")

shuffle(tokenized_queries)
from morpho_model import queries2morpho
morpho = queries2morpho(tokenized_queries[:100])
if len(morpho) == 100:
	print("Morpho test 1 succeeded")
else:
	print("Morpho test 1 failed (!): feature extractor returned %d examples instead of %d" % (len(morpho), 100))

for query, morpho_query in zip(tokenized_queries[:100], morpho):
	if len(query) != len(morpho_query):
		print("Lengthes of %s and %s are different" % (' '.join(query), ' '.join(morpho2query)))

print("Morpho test 2 has been completed.")

if queries2morpho([['дыбенко', '_NUMBER_']]) == [['Npmsny', '_NUMBER_']]:
	print("Morpho test 3 succeeded.")
else:
	print("Morpho test 3 failed (!)")

from prep_model0 import corpus2prep
prep = corpus2prep([' '.join(query) for query in tokenized_queries[:100]])
if len(prep) == 100:
	print("Prep0 test 1 succeeded")
else:
	print("Prep0 test 1 failed (!): feature extractor returned %d examples instead of %d" % (len(prep), 100))

for query, prep_query in zip(tokenized_queries[:100], prep):
	if len(query) != len(prep_query):
		print("Lengthes of %s and %s are different" % (' '.join(query), ' '.join(prep2query)))

print("Prep0 test has been completed.")

if queries2morpho([['дыбенко', '_NUMBER_']]) == [['Npmsny', '_NUMBER_']]:
	print("Prep0 test 3 succeeded.")
else:
	print("Prep0 test 3 failed (!)")

from gensim.models import Word2Vec
my_w2v1 = Word2Vec.load('word2vecs/my_w2v1')
if len(my_w2v1.score(tokenized_queries[:100])) == 100:
	print("Word2vec test 1 succeeded.")
else:
	print("Word2vec test 1 failed (!).")

from prep_model1 import corpus2prep
if len(corpus2prep(["гостиницы на ул ленина в екатеринбурге"])) == 1 and len(corpus2prep(["гостиницы на ул ленина в екатеринбурге"])[0]) == 3:
	print("Prep1 test succeeded.")
else:
	print("Prep1 test failed (!)")

from metrics import count
sentence = "Улица Довлатова праспект шахтёроф сройка".split()
suggestion = "Улица Далматова проспект шахтёроф тройка".split()
correct = "Улица Довлатова проспект шахтёров стройка".split()

if count(sentence, suggestion, correct) == (1, 1, 2):
	print("Metrics test 1 succeeded.")
else:
	print("Metrics test 1 failed (!). It returned %s instead of %s" % (str(count(sentence, suggestion, correct)), "(1, 1, 2)"))

print("Checking data")
from sklearn.externals import joblib
X = joblib.load('X.pkl')
y = joblib.load('y.pkl')
if X.shape[0] == y.shape[0] and len(X.shape) == 2 and len(y.shape) == 1:
	print("Data test 1 succeeded")
else:
	print("Data test 1 failed. Shapes: ", X.shape, y.shape)

from reranking import tokenize, suggest
from random import choice

f = codecs.open("corpus2.csv", mode = "rU", encoding = "utf-8-sig")
corpus_lines = f.read().split('\r\n')
border = int(corpus_lines.pop(0))
marked_queries = [query.rsplit(',', 3) for query in corpus_lines[:border]]
misspelled = [(tokenize(query), tokenize(suggestion)) for (raw_query, query, suggestion, label) in marked_queries if label == '1' and suggestion != '<?????>' and suggestion != '']

random_misspelled_query = choice([a for a, b in misspelled if '_NUMBER_' in a])
if random_misspelled_query in suggest(random_misspelled_query):
	print("Suggestion test #1 finished successfully at query", random_misspelled_query)
else:
	print("Suggestion test #1 failed (!) at query", query)

random_misspelled_query = choice([a for a, b in misspelled if '_NUMBER_' not in a])
if random_misspelled_query in suggest(random_misspelled_query):
	print("Suggestion test #2 finished successfully at query", random_misspelled_query)
else:
	print("Suggestion test #2 failed (!) at query", query)

(random_misspelled_query, sug) = choice([(a, b) for a, b in misspelled if len(a) != len(b)])
sugs = suggest(random_misspelled_query)

if len(sug) in [len(s) for s in sugs]:
	print("Suggestion test #3 succeded at query ", random_misspelled_query, "with sugs", *sugs)
else:
	print("Suggestion test #3 failed (!) at query ", random_misspelled_query, "with sugs", *sugs)

from reranking import corpus2features, suggest
Scaler = joblib.load('Scaler.pkl')

random_misspelled_query = choice([a for a, b in misspelled])
sugs = suggest(random_misspelled_query)
try:
	features = corpus2features([random_misspelled_query] * len(sugs), sugs)
except Exception as error:
	print("Feature extractor test #1 failed at query %s (!) with the following error message: %s" % (random_misspelled_query, error))
else:
	print("Feature extractor test #1 has been successfully completed at query %s. " % query)
	logreg = joblib.load('logreg.pkl')
	if logreg.coef_.shape[1] == features.shape[1]:
		print("Feature extractor test #2 has been completed successfully at query %s: feature shape %s is consistent with logreg coefficients shape %s" % (random_misspelled_query, str(features.shape), str(logreg.coef_.shape)))
		features = Scaler.transform(features)
		if features.mean() > 1:
			print("Scaler test failed because mean value %f > 1" % features.mean())
		elif features.mean() < -1:
			print("Scaler test failed because mean value %f < -1" % features.mean())
		elif features.std() > 2:
			print("Scaler test failed because standard deviation value %f > 2" % features.std())
		elif features.std() < 0.5:
			print("Scaler test failed because standard deviation value %f < 0.5" % features.std())
		else:
			print("Scaler test has been successfully completed with mean value %f and std value %f" % (features.mean(), features.std()))
	else:
		print("Feature extractor test #2 has failed at query %s: feature shape %s is consistent with logreg coefficients shape %s" % (random_misspelled_query, str(features.shape), str(logreg.coef_.shape)))
