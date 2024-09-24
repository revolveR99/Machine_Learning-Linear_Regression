#Assignment

V.1 Data Analysis
We will see some basic steps of data exploration. Of course, these
are not the only techniques available or the only one step to follow.
Each data set and problem has to be approached in an unique way. You
will surely find other ways to analyze your data in the future.
First of all, take a look at the available data. look in what format it is presented, if
there are various types of data, the different ranges, and so on. It is important to make
an idea of your raw material before starting. The more you work on data - the more you
develop an intuition about how you will be able to use it.
In this part, Professor McGonagall asks you to produce a program called describe.[extension].
This program will take a dataset as a parameter. All it has to do is to display information
for all numerical features like in the example:
$> describe.[extension] dataset_train.csv
Feature 1 Feature 2 Feature 3 Feature 4
Count 149.000000 149.000000 149.000000 149.000000
Mean 5.848322 3.051007 3.774497 1.205369
Std 5.906338 3.081445 4.162021 1.424286
Min 4.300000 2.000000 1.000000 0.100000
25% 5.100000 2.800000 1.600000 0.300000
50% 5.800000 3.000000 4.400000 1.300000
75% 6.400000 3.300000 5.100000 1.800000
Max 7.900000 4.400000 6.900000 2.50000

#NOTE :It is forbidden to use any function that makes the job done for you
like: count, mean, std, min, max, percentile, etc... no matter the
language that you use. Of course, it is also forbidden to use the
describe library or any function that looks similar(more or less) to
it from another library.

V.2 Data Visualization
Data visualization is a powerful tool for a data scientist. It allows you to make insights
and develop an intuition of what your data looks like. Visualizing your data also allows
you to detect defects or anomalies.
In this section, you are asked to create a set of scripts, each using a particular visualization method to answer a question. There is not necessarily a single answer to the
question.
V.2.1 Histogram
Make a script called histogram.[extension] which displays a histogram answering the
next question :
Which Hogwarts course has a homogeneous score distribution between all four houses?
V.2.2 Scatter plot
Make a script called scatter_plot.[extension] which displays a scatter plot answering
the next question :
What are the two features that are similar ?
V.2.3 Pair plot
Make a script called pair_plot.[extension] which displays a pair plot or scatter plot
matrix (according to the library that you are using).
From this visualization, what features are you going to use for your logistic regression?

V.3 Logistic Regression
You arrive at the last part: code your Magic Hat. To do this, you have to perform a
multi-classifier using a logistic regression one-vs-all.
You will have to make two programs :
• First one will train your models, it’s called logreg_train.[extension]. It takes
as a parameter dataset_train.csv. . For the mandatory part, you must use the
technique of gradient descent to minimize the error. The program generates a file
containing the weights that will be used for the prediction.
• A second has to be named logreg_predict.[extension]. It takes as a parameter
dataset_test.csv and a file containing the weights trained by previous program.
In order to evaluate the performance of your classifier this second program will have
to generate a prediction file houses.csv formatted exactly as follows:
$> cat houses.csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Hufflepuff
4,Slytherin
5,Ravenclaw
6,Hufflepuff
[...]

# FT_LINEAR_REGRESSION
The aim of this project is to introduce you to the basic concept behind machine learning. For this project, you will have to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm. 


## Project description

First of the projects in the machine learning area at 42 network. Objective is to create a linear regression function in the language of choise **(python3)**, train the model on the given dataset, save generated indexes and use them to predict car price depending on it's mileage. 

`train.py` functions with `csv` files with `,` as separator
## Plot Data after Training
![Screenshots](/pic/LR-Graph.png)
## Live Plotting of Training
![Screenshots](/pic/LR-Live.gif)
## Live Progress of Training from terminal
![Screenshots](/pic/FT_LINEAR_REGRESSION_TRAINING.gif)
## Scatter predicted price
![Screenshots](/pic/PredictGraph.png)

## [Linear Regression - Wiki](https://en.wikipedia.org/wiki/Linear_regression)

## Usage

Clone and change directory to project, then
	
	python3 train.py [flags]
	python3 predict.py [your_desired_mileage [path/to/thetas.txt] [flags2]]

train.py : If no file is passed as a parameter, function takes 'data.csv' as default file
### Flags

	-po 	- plot standardized dataset and solution
	-pn 	- plot original dataset and solution
	-hs 	- plot history of COST over iterations
	-l      - set learning rate (affects speed of learning), must be followed by a number
    -it 	- set number of iterations (affects accuracy of result), must be followed by a number, by default its uncapped
    -in     - --input : takes the name of datasets file 
    -o      - --output: the output file name of the THETAS
    -lv     - --live : take snapshots on every loop during the iteration to form a GIF showing the progress of the training live
      
### Flags2

	-sc 	- scatter the predicted price with datasets values, must be followed with path o datasets file
