# lt2222-v22-a3
Written by Maria Irena Szawerna.
These are the scripts and results for Assignment 3 in the Introduction to Machine Learning course V22 at the University of Gothenburg.
Run the scripts with -h to find out what arguments they use. 
sample.py creates a requested number of samples and splits them into training and testing data that is saved in the files of your choice.
train.py uses the training samples from sample.py to train and save a classifier model of your choice: MultinomialNB or SVC from sklearn.
test.py uses the trained and saved model from train.py and the testing samples from sample.py to evaluate the models, you can select the type of averaging.
perplexity.py uses the trained model and a text file to calculate the perplexity of that model given the text.
