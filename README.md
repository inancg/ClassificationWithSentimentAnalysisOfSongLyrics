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
#####Formatting Songs
The songs are formatted as `[positive,negative,neutral]` for better usability
#####Sampling
n samples were treated (where n is [5,10,15]) as test set, and the rest as the training set.
#####Logging
A logging mechanism created to automate the testing process
#####Preprocessing
Two preprocessing methods used; *scale* and *normalize*.

Both were tested throughly, results provided in the following Results section.
#####SVM Methods
Two different SVM methods used; SVC and LinearSVC

Both were tested throughly, results provided in the following Results section.
#####Formatting Songs for Grid 
To plot the graphs in 2D, the data is reduced to 2D in this function

##Test Results
**Here are the results for tests
SVC Results
Correct, False, Accuracy

-Sample Count = 5
-Test Count = 500
-Length of Arrays = 66, 66**

Morrissey
0.7312000000000036
Smiths
0.3935999999999998

Morrissey
0.4531999999999998
Smiths
0.5744000000000007

Morrissey
0.42919999999999975
Smiths
0.7208000000000031

Morrissey
0.54
Smiths
0.6599999999999999


-Sample Count = 10
-Test Count = 500
-Length of Arrays = 66, 66

Morrissey
0.730799999999999
Smiths
0.38360000000000044

Morrissey
0.46339999999999987
Smiths
0.5747999999999994

Morrissey
0.4238000000000003
Smiths
0.6948000000000006

Morrissey
0.46960000000000013
Smiths
0.6484


-Sample Count = 15
-Test Count = 500
-Length of Arrays = 66, 66

Morrissey
0.736266666666669
Smiths
0.38213333333333505

Morrissey
0.49120000000000025
Smiths
0.5789333333333332

Morrissey
0.43120000000000097
Smiths
0.7065333333333356

Morrissey
0.45666666666666744
Smiths
0.6610666666666674

--------------------------------

Linear SVC Results
Correct, False, Accuracy

-Sample Count = 5
-Test Count = 500
-Length of Arrays = 66, 66

Morrissey
0.66
Smiths
0.33999999999999997

Morrissey
0.56
Smiths
0.32

Morrissey
0.6799999999999999
Smiths
0.7000000000000001

Morrissey
0.4400000000000001
Smiths
0.6599999999999999


-Sample Count = 10
-Test Count = 500
-Length of Arrays = 66, 66

Morrissey
0.49719999999999975
Smiths
0.5576000000000003

Morrissey
0.5338000000000003
Smiths
0.46980000000000055

Morrissey
0.5547999999999998
Smiths
0.5717999999999998

Morrissey
0.4536000000000002
Smiths
0.6563999999999997


-Sample Count = 15
-Test Count = 500
-Length of Arrays = 66, 66
Morrissey
0.4607999999999995
Smiths
0.5922666666666664

Morrissey
0.5239999999999997
Smiths
0.4646666666666672

Morrissey
0.5617333333333331
Smiths
0.5770666666666658

Morrissey
0.4568000000000006
Smiths
0.6488000000000013