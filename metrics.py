from Levenshtein import distance
from re import findall
import codecs

split_apart = lambda l, i: (l[:i], l[i:])
tokenize = lambda x: findall('[А-Яа-яЁё0-9]+', x)

none_alike = lambda l1, l2: min([el1 != el2 for el1 in l1 for el2 in l2] + [True])

def best_option(words1, words2, words3):
	result = [(split_apart(words1, i+1), split_apart(words2, j+1), split_apart(words3, m+1)) for i in range(len(words1)) for j in range(len(words2)) for m in range(len(words3))]
	#print(result)
	#result = [(x, y, z) for (x, y, z) in result if (none_alike(x[0],y[0]) or none_alike(y[0], z[0])) and len(x) != 0 and len(y) != 0 and len(z) != 0]
	
	#print(result)
	result = sorted(result, key = lambda x: distance(''.join(x[0][0]), ''.join(x[1][0])) + distance(''.join(x[1][0]), ''.join(x[2][0])))
	return result[0]

def alignment(words1, words2, words3):
	words1 = words1[:]
	words2 = words2[:]
	words3 = words3[:]
	result = []
	i = 0
	while len(words1) > 0 and len(words2) > 0 and len(words3) > 0:
		i += 1
		if words1[0] == words2[0] == words3[0]:
			result.append(([words1[0]], [words2[0]], [words3[0]]))
			words1 = words1[1:]
			words2 = words2[1:]
			words3 = words3[1:]
		else:
			bestie = best_option(words1, words2, words3)
			result.append((bestie[0][0], bestie[1][0], bestie[2][0]))
			words1 = bestie[0][1]
			words2 = bestie[1][1]
			words3 = bestie[2][1]
	
	if max(len(words1), len(words2), len(words3)) > 0:
		result.append([words1[:], words2[:], words3[:]])
		
	#assert len(words1) == len(words2)
	#assert len(words2) == len(words3)
	
	return result
	
def count(sentence, correct, suggestion):
	align = alignment(sentence, correct, suggestion)
	TP = len([print("TP: ", (a, b, c)) for (a, b, c) in align if a != b and b == c])
	FP = len([print("FP: ", (a, b, c)) for (a, b, c) in align if a == b and b != c])
	FN = len([print("FN: ", (a, b, c)) for (a, b, c) in align if a != b and b != c])

	return (TP, FP, FN)

if __name__ == '__main__':
	f = codecs.open('result.txt', mode = 'r', encoding = 'utf-8-sig')
	lines = [line.split(',') for line in f.read().split('\n')]
	lines = [line for line in lines if len(line) == 4]
	f.close()
	
	#print(lines[:10])
	print(len(lines))
	
	true_positives = 0
	false_negatives = 0
	false_positives = 0
	
	for (sentence, correct, suggestion1, suggestion2) in lines:
		(TP, FP, FN) = count(tokenize(sentence), tokenize(correct), tokenize(suggestion1))
		true_positives += TP
		false_negatives += FN
		false_positives += FP
	
	print("TP: ", true_positives)
	print("FN: ", false_negatives)
	print("FP: ",  false_positives)
	precision = true_positives / (true_positives + false_positives)
	recall = true_positives / (true_positives + false_negatives)
	print("Precision: %f" % precision)
	print("Recall: %f" % recall)
	print("F1: %f" % (2 * precision * recall / (precision + recall)))
	
	true_positives = 0
	false_negatives = 0
	false_positives = 0
	
	for (sentence, correct, suggestion1, suggestion2) in lines:
		(TP, FP, FN) = count(tokenize(sentence), tokenize(correct), tokenize(suggestion2))
		true_positives += TP
		false_negatives += FN
		false_positives += FP
	
	print("TP: ", true_positives)
	print("FN: ", false_negatives)
	print("FP: ", false_positives)
	precision = true_positives / (true_positives + false_positives)
	recall = true_positives / (true_positives + false_negatives)
	print("Precision: %f" % precision)
	print("Recall: %f" % recall)
	print("F1: %f" % (2 * precision * recall / (precision + recall)))
