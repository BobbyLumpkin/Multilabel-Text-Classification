# Multilabel-Text-Classification

An exploration of multilabel methods applied to text categorization. We train binary relevance, novel algorithmic adaptations, and deep learning models to the popular "Reuters-21578" dataset (often used as a benchmark in text categorization). We compare models trained using two different feature-generating approaches: (1) tf-idf vectors and (2) sequences of integers fed into learned word-embeddings.

In the bag-of-words approach (tf-idf vectors), We'll also explore various dimension reduction techniques. We look at principal components analysis for deriving low dimensional linear representations of our data, and ANN autoencoders for deriving low dimensional (potentially) non-linear representations of our data.

---

## <u>Important Folders \& Files for Use</u>

**<u>Preprocessing and Dimension Reduction:</u>**
In order to train classification models, we need to generate algorithm-interpretable features from the text in each document. There are many common methods to these ends, and we will explore two throughout this project. The first, TF-IDF, is known as a "bag-of-words" approach. We will use these features to train the majority of our models. The second retains the ordering-structure in documents, converting sequences of words into sequences of integers. Before generating either type of feature, we clean the text, eliminating "stop words", urls, etc.. Furthermore, tf-idf vectors tend to be large and sparse, so we implement linear and nonlinear dimension reduction techniques to generate the features that will be used for training.

* `Text_Preprocessing.ipynb`: 

**<u>Binary Relevance Models:</u>**
For every considered base classifier (kNN, Gradient Boosted Trees, and SVMs), the corresponding binary relevance models are trained on both the separable PC scores and the autoencoder encodings generated in `Preprocessing and Dimension Reduction/tfidf_Dimension_Reduction.ipynb`. Additionally, we utilize both constant and learned threshold functions.

* `kNN Based/kNN_Based_Models.ipynb`: Loads preprocessed, dimension-reduced data and trains binary relevance models, using kNN classifiers as the base models. 

* `Gradient Boosted Trees Based/GBT_Based_Models.ipynb`: Loads preprocessed, dimension-reduced data and trains binary relevance models, using gradient boosted trees as the base models.

* `SVM Based/SVM_Based_Models.ipynb`: Loads preprocessed, dimension-reduced data and trains binary relevance models, using support vector machines as the base models.