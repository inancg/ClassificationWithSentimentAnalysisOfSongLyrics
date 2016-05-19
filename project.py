from sklearn import svm
from sklearn import preprocessing
import numpy as np
import random as r
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

raw_smiths = [[-1, 11, 18, 13], [-1, 12, 21, 12], [-1, 11, 17, 12], [-1, 10, 18, 7], [-1, 8, 10, 5], [0, 2, 1, 7], [0, 2, 9, 11], [0, 12, 11, 19], [0, 8, 5, 13], [1, 7, 4, 6], [-1, 7, 16, 1], [0, 0, 8, 13], [-1, 1, 13, 10], [-1, 5, 17, 9], [1, 22, 19, 10], [-1, 11, 15, 10], [1, 20, 16, 9], [-1, 2, 23, 3], [-1, 13, 16, 6], [-1, 11, 23, 6], [-1, 17, 29, 5], [-1, 9, 20, 12], [0, 4, 7, 13], [1, 12, 7, 11], [-1, 8, 26, 8], [-1, 1, 10, 5], [0, 7, 4, 11], [0, 7, 8, 12], [-1, 13, 20, 15], [-1, 1, 9, 4], [-1, 6, 12, 11], [0, 6, 19, 35], [0, 3, 6, 25], [-1, 6, 8, 1], [0, 9, 13, 17], [-1, 15, 21, 18], [0, 17, 20, 22], [0, 8, 7, 27], [0, 4, 9, 11], [0, 10, 9, 13], [0, 0, 8, 31], [0, 6, 4, 7], [-1, 2, 21, 12], [-1, 8, 26, 11], [0, 13, 14, 19], [0, 10, 15, 20], [-1, 16, 24, 14], [0, 20, 8, 27], [-1, 6, 13, 5], [0, 2, 10, 12], [-1, 17, 19, 16], [0, 1, 17, 21], [0, 12, 10, 21], [-1, 9, 25, 18], [-1, 7, 14, 9], [-1, 5, 13, 10], [0, 5, 8, 21], [-1, 8, 20, 7], [-1, 0, 34, 7], [0, 3, 5, 25], [-1, 1, 12, 0], [-1, 9, 17, 3], [0, 1, 9, 15], [-1, 5, 8, 7], [-1, 7, 30, 7], [-1, 6, 24, 12]]
raw_moz = [[0, 8, 7, 12], [-1, 6, 17, 11], [0, 4, 10, 11], [-1, 7, 16, 14], [-1, 6, 14, 10], [-1, 8, 16, 11], [1, 16, 14, 13], [-1, 6, 13, 7],[0, 2, 4, 14], [0, 1, 10, 17], [0, 8, 9, 13], [0, 2, 6, 22], [0, 11, 9, 12], [0, 7, 17, 27], [0, 4, 12, 16], [-1, 11, 12, 5], [1, 7, 6, 5],[-1, 17, 30, 25], [0, 6, 11, 14], [-1, 3, 10, 7], [0, 7, 10, 20], [0, 12, 18, 24], [0, 0, 14, 17], [0, 7, 12, 14], [-1, 13, 16, 7], [-1, 13, 14, 9], [-1, 9, 17, 11], [1, 21, 6, 9],[0, 6, 2, 15], [-1, 3, 17, 13], [0, 6, 7, 15], [0, 6, 10, 16], [-1, 8, 15, 12], [-1, 5, 17, 9], [-1, 4, 15, 6],[1, 17, 4, 14], [0, 4, 9, 20], [-1, 0, 18, 9], [0, 4, 10, 12], [-1, 5, 10, 6], [0, 6, 7, 17], [-1, 5, 21, 4], [-1, 5, 9, 2], [0, 6, 9, 10],[0, 7, 9, 20], [-1, 12, 17, 9], [0, 7, 6, 17], [0, 8, 21, 25], [0, 7, 12, 17], [0, 15, 14, 27], [0, 5, 11, 14], [0, 8, 19, 24],[0, 7, 6, 13], [0, 5, 7, 13], [1, 9, 9, 2], [0, 4, 2, 11], [0, 6, 6, 25], [0, 2, 8, 11], [1, 17, 16, 10], [0, 3, 4, 29], [0, 5, 11, 19], [1, 14, 8, 14], [1, 14, 13, 7], [-1, 4, 16, 4],[1, 7, 6, 5], [-1, 7, 8, 4], [1, 7, 2, 7], [-1, 0, 7, 3], [-1, 4, 12, 0], [-1, 4, 6, 5], [0, 4, 4, 8], [1, 8, 5, 2],[0, 5, 4, 11], [0, 4, 7, 11], [0, 2, 2, 9], [-1, 4, 12, 6], [0, 1, 7, 14], [0, 6, 6, 11], [0, 6, 6, 7], [1, 14, 4, 3], [0, 1, 6, 13], [-1, 4, 21, 0], [0, 3, 10, 16],[-1, 4, 10, 10], [-1, 8, 12, 4], [0, 0, 4, 14], [0, 6, 9, 24], [-1, 8, 12, 8], [-1, 2, 14, 5],[1, 13, 6, 11], [0, 2, 3, 11], [-1, 3, 30, 7], [-1, 4, 22, 1], [0, 9, 17, 28],[1, 13, 6, 11], [1, 14, 10, 14], [0, 8, 8, 19], [0, 2, 3, 11], [0, 8, 2, 29], [-1, 3, 25, 14], [-1, 9, 28, 10], [-1, 3, 30, 7], [-1, 2, 25, 15], [0, 0, 7, 12], [-1, 4, 22, 1], [1, 10, 9, 1], [1, 21, 17, 13], [0, 9, 17, 28],[1, 15, 11, 11], [1, 15, 10, 6], [-1, 7, 16, 3], [1, 8, 4, 4], [-1, 2, 26, 6], [1, 13, 5, 7], [1, 12, 6, 12], [-1, 3, 17, 5], [0, 4, 17, 23], [0, 4, 11, 14], [0, 8, 13, 17], [0, 6, 7, 23], [-1, 9, 10, 2], [-1, 9, 24, 4]]
#Indexes
GOODS		= 0
BADS		= 1
NEUTS		= 2
#Results
POS			= 1
NEUTRAL	= 0
NEG 		= -1
#Artists
SMITHS 	= 0
MOZ 		= 1
#Methods
SCALED 			= 0
NORMALIZED 	= 1
#Limits
MIN 	=	0
MAX 	= 1

def format_songs(los):
	result = []
	for song in los:
		result.append(song[1:])
	return result

def format_songs_grid(los):
	result = []
	for song in los[0:66]:
		result.append(song[1:3])
	return result

def create_samples(indexes, arr):
	samples = []
	for i in sorted(indexes,reverse=True):
		samples.append(arr.pop(i))
	return samples

def test_func(randoms, artist, times, clf):
	correct = 0
	wrong = 0
	for i in range(0,times):
		tmp_correct = 0
		tmp_wrong = 0
		mp = clf.predict(randoms)
		for p in mp:
			if p==artist 	: correct += 1
			else					:	wrong		+= 1
		correct += tmp_correct
		wrong += tmp_wrong
	return (correct,wrong, (correct/float(correct+wrong)))

def ready_for_testing(raw_smiths,raw_moz,sample_count):
	smiths	= format_songs(raw_smiths)
	moz			= format_songs(raw_moz[:len(smiths)])

	smiths_randoms_indexes	= r.sample(range(0,len(smiths)), sample_count)
	moz_randoms_indexes			= r.sample(range(0,len(moz)), sample_count)

	#This also alters arrays smiths and moz
	smiths_randoms 	= create_samples(smiths_randoms_indexes, smiths)
	moz_randoms 		= create_samples(moz_randoms_indexes, moz)


	songs = smiths + moz
	length_smiths	= len(smiths)
	length_moz		= len(moz)
	artists = [0 for i in range(0,length_smiths)] + [1 for i in range(length_smiths,length_smiths+length_moz)]
	return (smiths_randoms,moz_randoms,songs,artists)

def prepare_to_log(raw_smiths,raw_moz,sample_count):
	test_data = ready_for_testing(raw_smiths,raw_moz,sample_count)
	smiths_randoms = test_data[0]
	moz_randoms = test_data[1]
	songs = test_data[2]
	artists = test_data[3]
	clf = svm.SVC()
	#clf = svm.LinearSVC()
	clf.fit(songs,artists)
	moz_acc = test_func(smiths_randoms,SMITHS,1,clf)[2]
	smiths_acc = test_func(moz_randoms,MOZ,1,clf)[2]
	print(smiths_acc,moz_acc)
	return (smiths_acc, moz_acc)

def log_results(raw_smiths, raw_moz, times, sample_count, preprocessed=False, pp_method="None"):
	smiths_acc = 0
	moz_acc = 0
	for i in range(0,times):
		if not preprocessed:
			results = prepare_to_log(raw_smiths,raw_moz,sample_count)
			moz_acc += results[1]
			smiths_acc += results[0]
		else:
			tmp_smiths = list(raw_smiths)
			tmp_moz = list(raw_moz)
			if pp_method == SCALED:
				smiths_data = preprocessing.scale(tmp_smiths)
				moz_data = preprocessing.scale(tmp_moz)
			elif pp_method == NORMALIZED:
				smiths_data = preprocessing.normalize(tmp_smiths,norm="l2")
				moz_data = preprocessing.normalize(tmp_moz,norm="l2")
			else :
				print ("Error")
				return
			results = prepare_to_log(smiths_data,moz_data,sample_count)
			moz_acc += results[1]
			smiths_acc += results[0]
	
	smiths_acc /= times
	moz_acc /= times
	f = open('Results','a')
	f.write("\nMorrissey\n")
	f.write(str(moz_acc))
	f.write("\nSmiths\n")
	f.write(str(smiths_acc))
	f.close()

#Manual Tests for SVC and Linear SVC
'''
log_results(raw_smiths,raw_moz,10,5)
log_results(raw_smiths,raw_moz,10,5, preprocessed=True, pp_method=SCALED)
log_results(raw_smiths,raw_moz,10,5, preprocessed=True, pp_method=NORMALIZED)
log_results(raw_smiths,raw_moz,500,10)
log_results(raw_smiths,raw_moz,500,10, preprocessed=True, pp_method=SCALED)
log_results(raw_smiths,raw_moz,500,10, preprocessed=True, pp_method=NORMALIZED)
log_results(raw_smiths,raw_moz,500,15)
log_results(raw_smiths,raw_moz,500,15, preprocessed=True, pp_method=SCALED)
log_results(raw_smiths,raw_moz,500,15, preprocessed=True, pp_method=NORMALIZED)
'''


#Functions from scikit-learn website (http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html) are used in the following function.
# Author : scikit-learn developers, 2010 - 2014) 
# Licence : BSD
def plotting(method="SVC", preprocessed=False):
	#Plotting Codes
	g_smiths = format_songs_grid(raw_smiths)
	g_moz = format_songs_grid(raw_moz)
	length_g_smiths = len(g_smiths)
	length_g_moz = len(g_moz)
	if preprocessed:
		songs = np.concatenate((preprocessing.scale(g_smiths),preprocessing.scale(g_moz)))
		x_limits = [-4,5]
		y_limits = [-4,6]
		method = "Preprocessed(Scaled) SVC"
	else :
		songs = g_smiths + g_moz
		x_limits = [-3,25]
		y_limits = [-2,35]
	artists = [0 for i in range(0,length_g_smiths)] + [1 for i in range(length_g_smiths,length_g_smiths+length_g_moz)]
	# Create a mesh to plot in
	h = 0.01
	songs_limits, artists_limits = np.meshgrid(np.arange(x_limits[MIN], x_limits[MAX], h),
	                     np.arange(y_limits[MIN], y_limits[MAX], h))
	#clf = svm.SVC().fit(songs, artists)
	clf = svm.LinearSVC().fit(songs, artists)
	points = clf.predict(np.c_[songs_limits.ravel(), artists_limits.ravel()])
	# Put the result into a color plot
	points = points.reshape(songs_limits.shape)
	plt.contourf(songs_limits, artists_limits, points, alpha=0.7, cmap=plt.cm.Paired)
	# Plot also the training points
	first_songs = [song[0] for song in songs]
	second_songs = [song[1] for song in songs]
	plt.scatter(first_songs, second_songs, c=artists, cmap=plt.cm.Paired)
	plt.xlim(songs_limits.min(), songs_limits.max())
	plt.ylim(artists_limits.min(), artists_limits.max())
	plt.title(method)
	#plt.title("Linear SVC")
	plt.xlabel('Positive Lines')
	plt.ylabel('Negative Lines')
	plt.show()

#plotting()
#plotting(preprocessed=True)

# Following linear regression code is taken from (http://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html)
# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# Licence: BSD
def linear_regression(raw_songs):
	songs = format_songs_grid(raw_songs)
	x = np.array([song[0] for song in songs])
	y = np.array([song[1] for song in songs])

	lr = LinearRegression()
	lr.fit(x[:, np.newaxis], y)

	fig = plt.figure()
	plt.plot(x, y, 'r.', markersize=15)

	plt.plot(x, lr.predict(x[:, np.newaxis]), 'c',markersize= 10)
	plt.legend(('Data', 'Linear Regression'), loc='lower right')
	#plt.title('Linear Regression Morrissey')
	plt.title('Linear Regression The Smiths')
	plt.xlabel('Positive Lines')
	plt.ylabel('Negative Lines')
	plt.show()

#linear_regression(raw_smiths)