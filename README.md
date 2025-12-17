# Summary 📖
This repository contains the notes and notebooks from IBM’s **Machine Learning with Python** course, which is part of the *AI Engineering Professional Certificate*.

## Module 1 - Machine Learning in Action
- **Definition of ML:** Machine learning (ML) is a **subset of artificial intelligence (AI)**. It involves using data and algorithms to enable computers to imitate how humans learn and make decisions, gradually improving their accuracy.
- **Applications of ML:** ML has many applications in the modern world:
    - **Healthcare:** Doctors use ML to prescribe the correct medicine to patients.
    - **Banking:** Bankers use ML to decide whether to approve or reject a loan application.
    - **E-commerce:** Businesses use ML to generate customer recommendations.
    - **Classification Applications:** This method is used to build applications for email filtering, speech-to-text, handwriting recognition, biometric identification, and document classification.
    - **Support Vector Machines (SVM):** Applications include speech recognition, anomaly detection, and noise filtering.
    - **Clustering Applications:** Clustering can be applied to identify music genres, segment user groups, or analyze market segments.
    
## Module 2 - Linear and Logistic Regression & Polynomial and Non-Linear Regression
- **Learning Methods:** ML models learn using **supervised, unsupervised, semi-supervised, and reinforcement learning** methods.
- **Technique Selection:** Selecting an ML technique depends on several factors, including the problem being solved, the type of data available, the resources, and the desired outcome.
- **ML Tools and Pipelines:** ML tools provide functionalities for machine learning pipelines. These pipelines include modules necessary for **data preprocessing** and **building, evaluating, optimizing, and implementing ML models**.
- ML tools use algorithms to simplify complex tasks, such as handling big data, conducting statistical analyses, and making predictions.

## Module 3 - Building Supervised Learning Models
Supervised learning involves methods where the model learns from labeled data.

### Regression Models

- Regression is a type of supervised learning model.
- It models a relationship between an explanatory feature and a **continuous target variable**.
- **Simple Regression:** This occurs when a single independent variable estimates a dependent variable. It can be linear or nonlinear.
- **Multiple Regression:** This process is used when more than one independent variable is present.
- **Multiple Linear Regression:** This is an extension of the simple linear regression model, utilizing two or more independent variables to estimate a dependent variable.
- **Regression Trees:** These are built by considering the features of a data set one by one. A regression tree is analogous to a decision tree but **predicts continuous values** rather than discrete classes. They are created by recursively splitting the data set into subsets to maximize information gained from the split and minimize the randomness of the classes assigned to the split nodes.

### Classification Models

- Classification is a supervised ML method that uses fully trained models to predict labels on new data. The labels form a **categorical variable with discrete values**.
- The **distinguishing feature** between classification and regression is the characteristic of the target, or labeled data.
- **Logistical Regression:** Training involves looking for the best parameters that map the input features to the target outcomes, with the objective of predicting classes with minimal error.
- **K-Nearest Neighbors (KNN):** This supervised ML algorithm uses a group of labeled data points to learn how to label other data points. KNN is used for both classification and regression.
- **Support Vector Machines (SVM):** This supervised learning technique builds classification and regression models. It maps each data instance as a point in multidimensional space, where input features are values for a specific coordinate.
- **Decision Tree:** This is an algorithm used for classifying data points. In a decision tree, each internal node corresponds to a test. Each branch corresponds to the result of the test, and each terminal (leaf) node assigns its data to a class.

## Module 4 - Building Unsupervised Learning Models
Unsupervised techniques aim to discover hidden patterns and structures in data.

- **Complementary Techniques:** Clustering, dimension reduction, and feature engineering are complementary techniques in ML and data science, working together to **improve model performance, quality, and interpretability**.
- **Clustering:** This technique automatically groups data points into clusters based on similarities.
- **Dimension Reduction (Dimensionality Reduction):** This simplifies the visualization of high-dimensional clustering, aiding feature engineering, and improving model quality.
    - It reduces the number of features required for a data model.
    - Algorithms reduce the number of data set features without sacrificing critical information.
    - High-dimensional data is often very difficult to analyze and visualize, so these algorithms simplify the data set for ML models.

## Module 5 - Evaluating and Validating Machine Learning Models
- **Supervised Learning Evaluation:** This process establishes how well an ML model can predict the outcome for unseen data. It is essential for understanding model effectiveness and involves comparing model predictions to ground-truth labels.
    - **Classification Metrics:** Common metrics include **accuracy, confusion matrix, precision, and recall**.
    - **Regression Evaluation:** This involves determining how accurately the model can predict **continuous numerical values**, such as exam grades. Regression models are not foolproof and often make prediction errors.
- **Unsupervised Evaluation:** Methods assess the quality of the discovered patterns and how effectively the model groups similar data points.
- **Model Validation:** This method is used to **optimize the ML model** without risking its ability to predict well on unseen data.
    - Validation helps prevent overfitting when selecting the best model configuration by tuning hyperparameters.
    - **Process:** Validation means tuning the model on the training data and only testing it on unseen test data once the model is deemed well trained. No snooping is involved.
    - **Data Snooping/Leakage:** Checking performance on the test data before optimization is complete is called data snooping, which is a form of data leakage.

## Notebooks

The folder Contains practical labs with topics such as:

- DBScan vs HDBscan
- Decision Trees / SVM
- Evaluating Models
- K-Means Customer Segmentation
- KNN Classification
- Logistic Regression
- ML Pipelines & GridSearchCV
- Multiple Linear Rgression
- Multi-class Classification
- PCA
- Random Forest & XGBoost
- Regression Trees
- Regularization in Linear Regression
- Simple Linear Regression
- tSNE & UMAP