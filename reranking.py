import sys

from sklearn.externals import joblib

a = ['Длина исправления в словах',
'Простой Левенштейн между оригиналом и исправлением',
'Нграмная модель',
'Число неизвестных словарю слов в исправлении',
'Число словарных слов, которые исправление сделало несловарными',
'Число несловарных слов, которые исправление сделало словарными',
'Разница в частотности слов (новое минус старое)',
'Число исправлений в неизвестных словах',
'Число исправлений в известных словарю словах',
'Число исправлений, где вставлена/пропущена/изменена всего одна буква',
'Число вставок пробелов',
'Число изъятий пробелов',
'Сколько неизвестных слов можно получить, слепив какие-нибудь знакомые словарю слова',
'Морфологический счёт',
'Взвешенный Левенштейн между оригиналом и исправлением',
'Взвешенный Левенштейн с дорогущим удалением',
'Взвешенный Левенштейн с дорогущей вставкой',
'Простой Левенштейн между фонетическими кодами',
'Семантическая модель',
'Предложная модель раз',
'Предложная модель два']

a = [('%d. ' + line) % (counter + 1) for counter, line in enumerate(a)]
if 'train' in sys.argv:
	from sklearn.externals import joblib
	from sklearn.linear_model import LogisticRegression
	from sklearn.preprocessing import StandardScaler

	X_train = joblib.load('X_train.pkl')
	y_train = joblib.load('y_train.pkl')
	Scaler = StandardScaler()

	print("X_train.shape: ", X_train.shape)
	print("y_train.shape", y_train.shape)
	
	X_train = Scaler.fit_transform(X_train)

	logreg = LogisticRegression().fit(X_train, y_train)
else:
	logreg = joblib.load('logreg.pkl')

b = list(logreg.coef_[0])

print("FEATURE IMPORTANCE ORDERED BY NUMBER OF FEATURE:")

[print(a, b) for a, b in zip(a, b)]

print("\nFEATURE IMPORTANCE ORDERED BY ITS VALUE: ")

[print(a, b) for a, b in sorted(zip(a, b), key = lambda x: -abs(x[1]))]
