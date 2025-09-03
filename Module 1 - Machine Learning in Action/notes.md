# An Overview of Machine Learning
Artificial intelligence, or AI, makes computers appear intelligent by simulating the cognitive abilities of humans. AI is a general field with a broad scope. It comprises computer vision, natural language processing, generative AI, machine learning, and deep learning.

![ai-meaning](images/videoframe_38540.png)

Machine learning, or ML, is the subset of AI that uses algorithms and requires feature engineering by practitioners. Deep learning distinguishes itself from traditional machine learning by using many-layered neural networks that automatically extract features from highly complex, unstructured big data.

![ai-ml-and-dl](images/videoframe_56096.png)

Machine learning teaches computers to learn from data, identify patterns, and make decisions without receiving any explicit instructions from a human being. ML algorithms use computational methods to learn information directly from data without depending on any fixed algorithm.

![how-ml-works](images/videoframe_72927.png)

- ML models learn using supervised, unsupervised, semi-supervised, and reinforcement learning.
 
 Machine learning models differ in how they learn. For example, supervised learning models train on labeled data to learn how to make inferences on new data, enabling them to predict otherwise unknown labels. Unsupervised learning works without labels by finding patterns in data. Semi-supervised learning trains on a relatively small subset of data that is already labeled, and iteratively retrains itself by adding new labels that it generates with reasonably high confidence. Reinforcement learning simulates an artificially intelligent agent interacting with its environment and learns how to make decisions based on feedback from its environment.

![ml-paradigms](images/videoframe_104335.png)

Selecting a machine learning technique depends on several factors, such as the problem you're trying to solve, the type of data you have, the available resources, and the desired outcome.

### ML Tecniques
- **Classification**: A classification technique is used to predict the class or category of a case, such as whether a cell is benign or malignant or whether a customer will churn.

- **Regression**:  The regression-slash-estimation technique is used to predict continuous values, such as the price of a house based on its characteristics or the CO2 emissions from a car's engine.

- **Clustering**: Clustering groups of similar cases, for example, can find similar patients or can be used for customer segmentation in the banking field. 

- **Association**: The association technique is used to find items or events that often co-occur, such as grocery items usually bought together by a particular customer or market segment. 

- **Anomaly detection**: Anomaly detection is used to discover abnormal and unusual cases. For example, it's used for credit card fraud detection.

- **Sequence mining**: Sequence mining is used to predict the next event. For instance, the clickstream analytics in websites. 

- **Dimension reduction**: Is used to reduce data size, particularly the number of features needed.

- **Recommendation systems**: This associates people's preferences with others who have similar tastes and recommends new items to them, such as books or movies.

![ml-techniques](images/videoframe_157699.png)
![ml-techniques](images/videoframe_186006.png)

To train a machine learning model on supervised data, it's possible select between the classification or regression technique. Classification categorizes input data into one of several predefined categories or classes, and then makes predictions about the class membership of new observations. Regression is different from classification. It does not make predictions about the class membership of new input data. Regression predicts continuous numerical values from input data.

![classification-regression](images/videoframe_198714.png)

Clustering is one of many unsupervised machine learning techniques for data modeling.  It's used to group data points or objects that are somehow similar. For example, in this chart, we use a clustering algorithm to represent data points in green, red, and blue color. The uncategorized data points represented in black color are considered noise.

![clustering](images/videoframe_216500.png)

## Applications of ML
ML applied to predict diseases, analyze consumer behavior, recognize images, and more.

![appplications](images/videoframe_350776.png)

## Machine Learning Model Lifecycle (MLML)
Problem Definition -> Data Collection -> Data Preparation -> Model Development and Evaluation -> Model Deployment

![ml-lifecycle](images/videoframe_62478.png)

## A day in the life of a MLE
- Each of the steps of the MLML is important to the success of the solution.
- After deployment, continuous monitoring and improvement is required to ensure the quality of the ML solution.

## Data Scientist vs. AI Engineer
![differences-between-a-ds-and-a-aie](images/videoframe_591990.png)

## Tools for ML

### What's data?

Data is a collection of raw facts, figures, or information used to draw insights, inform decisions, and fuel advanced technologies. Data is central to every machine learning algorithm and the source of all the information the algorithm uses to discover patterns and make predictions.

![data](images/videoframe_52637.png)

### Machine learning tools
Machine learning tools provide functionalities for machine learning pipelines, which include modules for data preprocessing and building, evaluating, optimizing, and implementing machine learning models. These tools use algorithms to simplify complex tasks, such as handling big data, conducting statistical analyses, and making predictions.

For example, Pandas library, used in data manipulation and analysis, and Scikit-learn library, used in supervised and unsupervised learning algorithms for linear regression

![ml-tools](images/videoframe_76600.png)

Machine learning tools serve several purposes. You can use them to store and retrieve data, work with plots, graphs, and dashboards, explore, visually inspect, and clean data, and prepare data for developing machine learning models.

![ml-tools-uses](images/videoframe_155807.png)
![ml-tools-categories](images/videoframe_164013.png)

Let's check them out!

![data-processing-and-analytics-tools](images/videoframe_194206.png)
![data-processing-and-analytics-tools](images/videoframe_224512.png)
![data-visualization-tools](images/videoframe_267132.png)
![ml-ecosystem-tools](images/videoframe_309035.png)
![deep-learning-tools](images/videoframe_348055.png)
![computer-vision-tools](images/videoframe_394391.png)
![nlp-tools](images/videoframe_433595.png)
![generative-ai-tools](images/videoframe_476204.png)

## Scikit-learn Machine Learning Ecosystem

The machine learning, or ML ecosystem, refers to the interconnected tools, frameworks, libraries, platforms, and processes that support developing, deploying, and managing machine learning models.

![ml-ecosystem](images/videoframe_67969.png)

Python offers a wide variety of tools and libraries for machine learning. Several open-sourced Python libraries comprise one of the most widely used ecosystems for machine learning: 

**NumPy** provides foundational machine learning support with its efficient numerical computations on large multidimensional data arrays.

**Pandas**, built on NumPy and Matplotlib, is used for data analysis, visualization, cleaning, and preparing data for machine learning. Pandas uses versatile arrays called data frames to handle data.

**SciPy**, built on NumPy, is used for scientific computing and has modules for optimization, integration, linear regression, and more.

**Matplotlib** is built on NumPy and has an extensive, highly customizable set of visualization tools. 

**Scikit-learn**, built on NumPy, SciPy, and Matplotlib, is used for building classical machine learning models.

![python-tools](images/videoframe_117893.png)

### Scikit-learn

![scikit-learn](images/videoframe_138311.png)
![scikit-learn](images/videoframe_151278.png)

Most of the tasks that need to be done in a machine learning pipeline are already implemented in scikit-learn, including data preprocessing tasks like data cleaning, scaling, feature selection and feature extraction, train or test splitting, model setup and fitting, hyperparameter tuning with cross-validation, prediction, evaluation, and exporting the model to be used in production.

# Module 1 Summary and Highlights

- Artificial intelligence (AI) simulates human cognition, while machine learning (ML) uses algorithms and requires feature engineering to learn from data.

- Machine learning includes different types of models: supervised learning, which uses labeled data to make predictions; unsupervised learning, which finds patterns in unlabeled data; and semi-supervised learning, which trains on a small subset of labeled data.

- Key factors for choosing a machine learning technique include the type of problem to be solved, the available data, available resources, and the desired outcome.

- Machine learning techniques include anomaly detection for identifying unusual cases like fraud, classification for categorizing new data, regression for predicting continuous values, and clustering for grouping similar data points without labels.

- Machine learning tools support pipelines with modules for data preprocessing, model building, evaluation, optimization, and deployment.

- R is commonly used in machine learning for statistical analysis and data exploration, while Python offers a vast array of libraries for different machine learning tasks. Other programming languages used in ML include Julia, Scala, Java, and JavaScript, each suited to specific applications like high-performance computing and web-based ML models.

- Data visualization tools such as Matplotlib and Seaborn create customizable plots, ggplot2 enables building graphics in layers, and Tableau provides interactive data dashboards.

- Python libraries commonly used in machine learning include NumPy for numerical computations, Pandas for data analysis and preparation, SciPy for scientific computing, and Scikit-learn for building traditional machine learning models.

- Deep learning frameworks such as TensorFlow, Keras, Theano, and PyTorch support the design, training, and testing of neural networks used in areas like computer vision and natural language processing.

- Computer vision tools enable applications like object detection, image classification, and facial recognition, while natural language processing (NLP) tools like NLTK, TextBlob, and Stanza facilitate text processing, sentiment analysis, and language parsing.

- Generative AI tools use artificial intelligence to create new content, including text, images, music, and other media, based on input data or prompts.

- Scikit-learn provides a range of functions, including classification, regression, clustering, data preprocessing, model evaluation, and exporting models for production use.

- The machine learning ecosystem includes a network of tools, frameworks, libraries, platforms, and processes that collectively support the development and management of machine learning models.