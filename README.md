#Classification and Analyze of Lyrics of Morrissey and The Smiths
Inanc Gurkan, 2016

#Formatting Lyrics
Lyrics used that can't be uploaded here because of the licencing issues. Yet it is possible to tell the format. Lyrics was in format as follows and without "-" signs(that breaks a functionality of a library):

*Lyrics lyrics lyrics lyrics*

*Lyrics lyrics lyrics lyrics*

*Lyrics lyrics lyrics lyrics*

##Libraries and functions used

####In data gathering stage;
* json
* requests
* glob
* http://text-processing.com/api/sentiment/ API for analysing lyrics line by line.

####In the actual implementation;
* sklearn.svm
* sklearn.preprocessing
* numpy
* matplotlib.pyplot
* sklearn.linear_model.LinearRegression

##Methods
####Data Gathering Stage
Lyrics are gathered and formatted as follows
`[result,positive,negative,neutral]`
####Actual Implementation
######Formatting Songs
The songs are formatted as `[positive,negative,neutral]` for better usability
######Sampling
n samples were treated (where n is [5,10,15]) as test set, and the rest as the training set.
######Logging
A logging mechanism created to automate the testing process
######Preprocessing
Two preprocessing methods used; *scale* and *normalize*.

Both were tested throughly, results provided in the following Results section.
######SVM Methods
Two different SVM methods used; SVC and LinearSVC

Both were tested throughly, results provided in the following Results section.

The following commands are used to test these

`
log_results(raw_smiths,raw_moz,10,5)
log_results(raw_smiths,raw_moz,10,5, preprocessed=True, pp_method=SCALED)
log_results(raw_smiths,raw_moz,10,5, preprocessed=True, pp_method=NORMALIZED)
log_results(raw_smiths,raw_moz,500,10)
log_results(raw_smiths,raw_moz,500,10, preprocessed=True, pp_method=SCALED)
log_results(raw_smiths,raw_moz,500,10, preprocessed=True, pp_method=NORMALIZED)
log_results(raw_smiths,raw_moz,500,15)
log_results(raw_smiths,raw_moz,500,15, preprocessed=True, pp_method=SCALED)
log_results(raw_smiths,raw_moz,500,15, preprocessed=True, pp_method=NORMALIZED)
`
######Formatting Songs for Grid 