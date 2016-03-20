

Mahout-Naive Bayesian Classification 
============================================
This is a demonstration of detecting Spam emails from non-spam using Naive Bayesian Classification algorithm on Mahout.

In order to classify a text document using NaiveBayesClassifier the user first needs to copy the model, labelindex file, dictionary and document frequency from HDFS to local filesystem.

As an example
```
 hadoop fs -get labelindex labelindex
 hadoop fs -get model model
 hadoop fs -get <sparse vector>/dictionary.file-0  dictionary.file-0
 hadoop fs -getmerge <sparse vector>/df-count df-count
```

*model* and *labelindex* are outputs of **mahout trainnb** and *<sparse vector>* should be replaced with the output directory of **mahout seq2sparse** command.

Finally in order to run the Naive Bayesian classification utility use the following syntax.

```
java -jar NBClassifier.jar model labelindex dictionary.file-0 df-count <input text file path>
```
The output artifact of maven build is renamed to NBClassifier.jar

  

