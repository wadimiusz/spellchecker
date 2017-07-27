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

	result = []
	for original, tag, lemma in tagged_lines:
			if original in ['_NUMBER_', '_NEWLINE_'] or tag[0] == 'S':
				result.append(original)
			else:
				result.append(tag)

	result  = [list(group) for k, group in groupby(result, lambda x: x == '_NEWLINE_') if not k]

	lines = '\n'.join([' '.join(line) for line in result])

	f = codecs.open('/home/v.fomin/queries_prep0.csv', 'w')
	f.write(lines)
	f.close()

def corpus2prep(lines):
	tagged_lines = tt_ru.tag(' _NEWLINE_ '.join(lines))
	result = []
	for original, tag, lemma in tagged_lines:
			if original in ['_NUMBER_', '_NEWLINE_'] or tag[0] == 'S':
				result.append(original)
			else:
				result.append(tag)
	
	result  = [list(group) for k, group in groupby(result, lambda x: x == '_NEWLINE_') if not k]
	return result
	

LM = ARPALanguageModel("/home/v.fomin/prep_model/prep_model0")
score = LM.score
