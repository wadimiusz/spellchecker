import codecs, sys, pickle

from treetagger import TreeTagger
from pynlpl.lm.lm import ARPALanguageModel
from itertools import groupby

tt_ru = TreeTagger(language = 'russian')

f = codecs.open('/home/v.fomin/queries.csv')
lines = f.read().split('\n')
f.close()

if __name__ == '__main__':
	if 'save' in sys.argv:
		tagged_lines = tt_ru.tag(' _NEWLINE_ '.join(lines))
		pickle.dump(tagged_lines, open('tagged_lines.pkl', 'wb'))
	else:
		tagged_lines = pickle.load(open('tagged_lines.pkl', 'rb'))


	groups = [[]]
	for original, tag, lemma in tagged_lines:
			if original == '_NUMBER_':
				groups[-1].append(original)
			elif tag[0] == 'S':
				groups.append([original])
			elif original == '_NEWLINE_':
				groups.append([])
			else:
				groups[-1].append(tag)

	lines = '\n'.join([' '.join(group) for group in groups])

	f = codecs.open('/home/v.fomin/queries_prep1.csv', 'w')
	f.write(lines)
	f.close()

def corpus2prep(lines):
	groups = [[[]]]
	tagged_lines = tt_ru.tag(' _NEWLINE_ '.join(lines))
	for original, tag, lemma in tagged_lines:
			if original == '_NUMBER_':
				groups[-1][-1].append(original)
			elif tag[0] == 'S':
				groups[-1].append([original])
			elif original == '_NEWLINE_':
				groups.append([[]])
			else:
				groups[-1][-1].append(tag)
	return groups
	

LM = ARPALanguageModel("/home/v.fomin/prep_model/prep_model1")
score = LM.score
