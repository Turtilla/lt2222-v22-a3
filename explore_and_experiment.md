# EXPLORE AND EXPERIMENT
Written by Maria Irena Szawerna 23.03.2022 for LT2222 V22  

## The technicalities
This is the report for Assignment 3 in the Introduction to Machine Learning course. Its aim is to report my experimentation and findings with the scripts that I constructed to try to predict the next consonant 
based on four characters (including letters, numbers, punctuation, whitespace, etc.) in a row. The scripts make use of two classification models, a support vector machine with a linear kernel and a multinomial
Naive Bayes classifier, both from scikit learn. The data on which the experimentation is done comes from a file that was shared with us containing UN speeches.  
## Step 1
As a preparation for this discussion I unzipped the file and split it into two parts: UN-english-sample.txt and UN-english-pp.txt. The first one was used to construct all of the samples, while the second one
served as a text to test the system perplexity on (so it was important that none of the models were trained on it, hence it being split). I used sample.py to prepare training and testing files for a number of
different sample sizes, with the train/test split at 80/20. I prepared samples of the following sizes:  
+ 100 samples
+ 1000 samples
+ 10000 samples
+ 25000 samples
+ 50000 samples  
Any larger samples sizes made the SVM model's learning time longer than the recommended maximum of 1h. Each of these sets of samples were split into two files, a training one and a testing one.
## Step 2
I trained both models - SVM/SVC and MultinomialNB - using train.py on every collection of samples obtained in the previous step. Naive Bayes models seemed to take equally long to train, while SVC took longer the 
bigger the sample size was. 
## Step 3
Using test.py I obtained evaluation measures for each of the models using the respective test sample sets, both with micro- and macro-averaging. The results can be seen in in the tables below:
### 100 samples
| Measure | sklearn's MultinomialNB | sklearn's SVC with a linear kernel |
| ----------- | ----------------------- | ---------------------------------- |
| accuracy | 0.15 | 0.45 |
| micro precision | 0.15 | 0.45 |
| macro precision | ~0.10 | ~0.19 |
| micro recall | 0.15 | 0.45 |
| macro recall | ~0.07 | ~0.24 |
| micro f1 score | 0.15 | 0.45 |
| macro f1 score | ~0.08 | ~0.20 |
### 1000 samples
| Measure | sklearn's MultinomialNB | sklearn's SVC with a linear kernel |
| ----------- | ----------------------- | ---------------------------------- |
| accuracy | 0.05 | 0.18 |
| micro precision | 0.05 | 0.18 |
| macro precision | ~0.02 | ~0.12 |
| micro recall | 0.05 | 0.18 |
| macro recall | ~0.07 | ~0.12 |
| micro f1 score | 0.05 | 0.18 |
| macro f1 score | ~0.02 | ~0.11 |
### 10000 samples
| Measure | sklearn's MultinomialNB | sklearn's SVC with a linear kernel |
| ----------- | ----------------------- | ---------------------------------- |
| accuracy | ~0.11 | ~0.30 |
| micro precision | ~0.11 | ~0.30 |
| macro precision | ~0.04 | ~0.18 |
| micro recall | ~0.11 | ~0.30 |
| macro recall | ~0.07 | ~0.15 |
| micro f1 score | ~0.11 | ~0.30 |
| macro f1 score | ~0.05 | ~0.15 |
### 25000 samples
| Measure | sklearn's MultinomialNB | sklearn's SVC with a linear kernel |
| ----------- | ----------------------- | ---------------------------------- |
| accuracy | ~0.10 | ~0.35 |
| micro precision | ~0.10 | ~0.35 |
| macro precision | ~0.05 | ~0.41 |
| micro recall | ~0.10 | ~0.35 |
| macro recall | ~0.09 | ~0.24 |
| micro f1 score | ~0.10 | ~0.35 |
| macro f1 score | ~0.04 | ~0.27 |
### 50000 samples
| Measure | sklearn's MultinomialNB | sklearn's SVC with a linear kernel |
| ----------- | ----------------------- | ---------------------------------- |
| accuracy | ~0.10 | ~0.28 |
| micro precision | ~0.10 | ~0.28 |
| macro precision | ~0.04 | ~0.23 |
| micro recall | ~0.10 | ~0.28 |
| macro recall | ~0.07 | ~0.17 |
| micro f1 score | ~0.10 | ~0.28 |
| macro f1 score | ~0.04 | ~0.18 |
## Step 4
Using perplexity.py I tested each model on the same UN-english-pp.txt file (the first 1000 lines of the original file) to get comparable perplexity scores. It is important to note that this part was more tricky
than expected, as sklearn's models output probabilities in ln, not log2, so they needed to be recalculated. In addition, the originally used CategoricalNB was throwing errors when encountering UNK tokens, which
is why I received permission to use MultinomialNB which did not do that. The results can be found in the table below:
| Sample size | sklearn's MultinomialNB | sklearn's SVC with a linear kernel |
| ----------- | ----------------------- | ---------------------------------- |
| 100 samples | ~609.8239 | ~18.1359 |
| 1000 samples | ~238.0000 | ~17.5837 |
| 10000 samples | ~114.6978 | ~13.3063 |
| 25000 samples | ~160.6719 | ~12.8929 |
| 50000 samples | ~68.3851 | ~14.2848 |
## Discussion
From both the evaluation measures and perplexity scores the conclusion can be drawn that an SVC model is better suited for this task than a Multinomial Naive Bayes classifier. However, the deviations from the
trend of "the bigger the training sample size, the better the model" show that it is difficult to predict the next consonant based on the preceding characters, and some of the scores (like the high evaluation for
the 100 sample size for SVC or the relatively lower one for 50000 samples for the same type of model) are just lucky or unlucky flukes depending on what was there in the samples (which were randomly selected).
Perplexity shows that small sample sizes do not make for good NB models, but the bigger the sample size, the lower the perplexity, without exceptions. The same cannot be said for SVC, where the perplexity for the
biggest sample size model was higher than the preceding one. All in all though, every SVC model had lower perplexity than even the best NB model.  
Looking once more at the evaluation measures, we can conclude rather safely that this kind of a problem is not easy to model or predict, and 4 characters in a row are not a good predictor for what consonant comes
next. Even when trained on big sample sizes the models show low evaluation measures, at best oscillating around 30%; the exception is the lucky 45% on a very small training and testing sample, but this is likely
just due to lucky data distribution. Both models seem to learn better with bigger sample sizes, with SVC being overall better at this particular task. 
Hopefully this experimentation and discussion is sufficient for this assignment. The sample files (test_XYZ.pickle and train_XYZ.pickle) and model files (model_XYZ_SVC/NB.pickle) will be uploaded to the repo 
together with the scripts and this discussion.



