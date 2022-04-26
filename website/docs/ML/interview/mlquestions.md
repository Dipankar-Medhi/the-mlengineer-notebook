---
sidebar_label: Machine Learning Questions
title: Machine Learning Questions
sidebar_position: 4
---

:::note Source
https://www.kaggle.com/getting-started/183949
::: 

Q1.	What is Machine Learning?
Machine learning is the study of computer algorithms that improve automatically through experience. It is seen as a subset of artificial intelligence. Machine Learning explores the study and construction of algorithms that can learn from and make predictions on data. You select a model to train and then manually perform feature extraction. Used to devise complex models and algorithms that lend themselves to a prediction which in commercial use is known as predictive analytics.

Q2.	What is Supervised Learning?

Supervised learning is the machine learning task of inferring a function from labeled training data. The training data consist of a set of training examples.

Algorithms: Support Vector Machines, Regression, Naive Bayes, Decision Trees, K-nearest Neighbor Algorithm and Neural Networks

E.g. If you built a fruit classifier, the labels will be “this is an orange, this is an apple and this is a banana”, based on showing the classifier examples of apples, oranges and bananas.

Q3.	What is Unsupervised learning?

Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labelled responses.

Algorithms: Clustering, Anomaly Detection, Neural Networks and Latent Variable Models

E.g. In the same example, a fruit clustering will categorize as “fruits with soft skin and lots of dimples”, “fruits with shiny hard skin” and “elongated yellow fruits”.

Q4.	What are the various algorithms?

There are various algorithms. Here is a list.
 

 



Q5.	What is ‘Naive’ in a Naive Bayes?
https://en.wikipedia.org/wiki/Naive_Bayes_classifier
 
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. Bayes’ theorem states the following relationship, given class variable y and dependent feature vector x1through xn:

Using the naive conditional independence assumption that each xi is independent: for all i, this relationship is simplified to:


Since P(x1, … , xn) is constant given the input, we can use the following classification rule:


and we can use Maximum A Posteriori (MAP) estimation to estimate P(y) and P(y|xi); the former is then the relative frequency of class y in the training set.


The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(y|xi): can be Bernoulli, Binomial, Gaussian, and so on.

Q6.	What is PCA? When do you use it?
https://en.wikipedia.org/wiki/Principal_component_analysis https://blog.umetrics.com/what-is-principal-component-analysis-pca-and-how-it-is-used https://blog.umetrics.com/why-preprocesing-data-creates-better-data-analytics-models

Principal component analysis (PCA) is a statistical method used in Machine Learning. It consists in projecting data in a higher  dimensional  space  into   a   lower   dimensional  space   by   maximizing the variance of each dimension.

The process works as following. We define a matrix A with n rows (the single observations of a dataset – in a tabular format, each single row) and p columns, our features. For this matrix we construct a variable space with as many dimensions as there are features. Each feature represents one coordinate axis. For




 
each feature, the length has been standardized according to a scaling criterion, normally by scaling to unit variance. It is determinant to scale the features to a common scale, otherwise the features with a greater magnitude will weigh more in determining the principal components. Once plotted all the observations and computed the mean of each variable, that mean will be represented by a point in the center of our plot (the center of gravity). Then, we subtract each observation with the mean, shifting the coordinate system with the center in the origin. The best fitting line resulting is the line that best accounts for the shape of the point swarm. It represents the maximum variance direction in the data. Each observation may be projected onto this line in order to get a coordinate value along the PC-line. This value is known as a score. The next best-fitting line can be similarly chosen from directions perpendicular to the first. Repeating this process yields an orthogonal basis in which different individual dimensions of the data are uncorrelated. These basis vectors are called principal components.


PCA is mostly used as a tool in exploratory data analysis and for making predictive models. It is often used to visualize genetic distance and relatedness between populations.

Q7.	Explain SVM algorithm in detail.
https://en.wikipedia.org/wiki/Support_vector_machine

Classifying data is a common task in machine learning. Suppose some given data points each belong to one of two classes, and the goal is to decide which class a new data point will be in. In the case of support- vector machines, a data point is viewed as a p-dimensional vector (a list of p numbers), and we want to know whether we can separate such points with a (p − 1)-dimensional hyperplane. This is called a linear classifier. There are many hyperplanes that might classify the data. One reasonable choice as the best hyperplane is the one that represents the largest separation, or margin, between the two classes. So, we choose the hyperplane so that the distance from it to the nearest data point on each side is maximized. If such a hyperplane exists, it is known as the maximum-margin hyperplane and the linear classifier it defines is known as a maximum-margin classifier; or equivalently, the perceptron of optimal stability. The best hyper plane that divides the data is H3.
We have n data (x1, y1), … , (xn, yn) and p different features xi = (x1, … , xp) and yi  is either 1 or -1.
i	i
The equation of the hyperplane H3 is as the set of points x satisfying:
 

w ∙ x − b = 0

where  w  is  the  (not  necessarily  normalized) normal  vector to  the  hyperplane. The   parameter
 
b
 
‖w‖
 
determines the offset of the hyperplane from the origin along the normal vector w.
 

So, for each i, either xi is in the hyperplane of 1 or -1. Basically, xi satisfies:
w ∙ xi − b ≥ 1	or	w ∙ xi − b ≤ −1

•	SVMs are helpful in text and hypertext categorization, as their application can significantly reduce the need for labeled training instances in both the standard inductive and transductive settings. Some methods for shallow semantic parsing are based on support vector machines.
•	Classification of images can also be performed using SVMs. Experimental results show that SVMs achieve significantly higher search accuracy than traditional query refinement schemes after just three to four rounds of relevance feedback.
•	Classification of satellite data like SAR data using supervised SVM.
•	Hand-written characters can be recognized using SVM.

Q8.	What are the support vectors in SVM?
In the diagram, we see that the sketched lines mark the distance from the classifier (the hyper plane) to the closest data points called the support vectors (darkened data points). The distance between the two thin lines is called the margin.

To extend SVM to cases in which the data are not linearly separable, we introduce the hinge loss function,

max (0, 1 − yi(w ∙ xi − b))
 
This function is zero if x lies on the correct side of the margin. For data on the wrong side of the margin, the function's value is proportional to the distance from the margin.

Q9.	What are the different kernels in SVM?
There are four types of kernels in SVM.
1.	LinearKernel
2.	Polynomial kernel
3.	Radial basis kernel
4.	Sigmoid kernel

Q10.	What are the most known ensemble algorithms?
https://towardsdatascience.com/the-ultimate-guide-to-adaboost-random-forests-and-xgboost-7f9327061c4f

The most popular trees are: AdaBoost, Random Forest, and eXtreme Gradient Boosting (XGBoost).

AdaBoost is best used in a dataset with low noise, when computational complexity or timeliness of results is not a main concern and when there are not enough resources for broader hyperparameter tuning due to lack of time and knowledge of the user.

Random forests should not be used when dealing with time series data or any other data where look- ahead bias should be avoided, and the order and continuity of the samples need to be ensured. This algorithm can handle noise relatively well, but more knowledge from the user is required to adequately tune the algorithm compared to AdaBoost.

The main advantages of XGBoost is its lightning speed compared to other algorithms, such as AdaBoost, and its regularization parameter that successfully reduces variance. But even aside from the regularization parameter, this algorithm leverages a learning rate (shrinkage) and subsamples from the features like random forests, which increases its ability to generalize even further. However, XGBoost is more difficult to understand, visualize and to tune compared to AdaBoost and random forests. There is a multitude of hyperparameters that can be tuned to increase performance.

Q11.	Explain Decision Tree algorithm in detail.
https://en.wikipedia.org/wiki/Decision_tree_learning https://www.kdnuggets.com/2019/02/decision-trees-introduction.html/2 https://medium.com/@naeemsunesara/giniscore-entropy-and-information-gain-in-decision-trees- cbc08589852d

A decision tree is a supervised machine learning algorithm mainly used for Regression and Classification. It breaks down a data set into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision tree can handle both categorical and numerical data. The term Classification and Regression Tree (CART) analysis is an umbrella term used to refer to both of the above procedures.

Some techniques, often called ensemble methods, construct more than one decision tree:
 
•	Boosted trees Incrementally building an ensemble by training each new instance to emphasize the training instances previously mis-modeled. A typical example is AdaBoost. These can be used for regression-type and classification-type problems.
•	Bootstrap aggregated (or bagged) decision trees, an early ensemble method, builds multiple decision trees by repeatedly resampling training data with replacement, and voting the trees for a consensus prediction.
o A random forest classifier is a specific type of bootstrap aggregating.
•	Rotation forest – in which every decision tree is trained by first applying principal component analysis (PCA) on a random subset of the input features.

A special case of a decision tree is a decision list, which is a one-sided decision tree, so that every internal node has exactly 1 leaf node and exactly 1 internal node as a child (except for the bottommost node, whose only child is a single leaf node). While less expressive, decision lists are arguably easier to understand than general decision trees due to their added sparsity, permit non-greedy learning methods and monotonic constraints to be imposed.

Notable decision tree algorithms include:

•	ID3 (Iterative Dichotomiser 3)
•	C4.5 (successor of ID3)
•	CART (Classification and Regression Tree)
•	Chi-square automatic interaction detection (CHAID). Performs multi-level splits when computing classification trees.
•	MARS: extends decision trees to handle numerical data better.
•	Conditional Inference Trees. Statistics-based approach that uses non-parametric tests as splitting criteria, corrected for multiple testing to avoid overfitting. This approach results in unbiased predictor selection and does not require pruning.

Q12.	What are Entropy and Information gain in Decision tree algorithm?

https://www.saedsayad.com/decision_tree.htm
https://medium.com/@naeemsunesara/giniscore-entropy-and-information-gain-in-decision-trees- cbc08589852d

There are a lot of algorithms which are employed to build a decision tree, ID3 (Iterative Dichotomiser 3), C4.5, C5.0, CART (Classification and Regression Trees) to name a few but at their core all of them tell us what questions to ask and when.

The below table has color and diameter of a fruit and the label tells the name of the fruit. How do we build a decision tree to classify the fruits?
 
 

Here is how we will build the tree. We will start with a node which will ask a true or false question to split the data into two. The two resulting nodes will each ask a true or false question again to split the data further and so on.
There are 2 main things to consider with the above approach:
•	Which is the best question to ask at each node
•	When do we stop splitting the data further?

Let’s start building the tree with the first or the topmost node. There is a list of possible questions which can be asked. The first node can ask the following questions:
•	Is the color green?
•	Is the color yellow?
•	Is the color red?
•	Is the diameter ≥ 3?
•	Is the diameter ≥ 1?

Of these possible set of questions, which one is the best to ask so that our data is split into two sets after the first node? Remember we are trying to split or classify our data into separate classes. Our question should be such that our data is partitioned into as unmixed or pure classes as possible. An impure set or class here refers to one which has many different types of objects for example if we ask the question for the above data, “Is the color green?” our data will be split into two sets one of which will be pure the other will have a mixed set of labels. If we assign a label to a mixed set, we have higher chances of being incorrect. But how do we measure this impurity?


Gini Impurity and Information Gain - CART

CART (Classification and Regression Trees) → uses Gini Index (Classification) as metric.




 
The Gini Impurity (GI) metric measures the homogeneity of a set of items. The lowest possible value of GI is 0.0. The maximum value of GI depends on the particular problem being investigated but gets close to 1.0.
Suppose for example you have 12 items — apples, grapes, lemons. If there are 0 apples, 0 grapes, 12 lemons, then you have minimal impurity (this is good for decision trees) and GI = 0.0. But if you have 4 apples, 4 grapes, 4 lemons, you have maximum impurity and it turns out that GI = 0.667.
I’ll show example calculations. Maximum GI: Apples, Grapes, Lemons


When the number of items is evenly distributed, as in the example above, you have maximum GI but the exact value depends on how many items there are. A bit less than maximum GI:

In the example above, the items are not quite evenly distributed, and the GI is slightly less (which is better when used for decision trees). Minimum GI:

 
   	   	 	       	 	     


The Gini index is not at all the same as a different metric called the Gini coefficient. The Gini impurity metric can be used when creating a decision tree but there are alternatives, including Entropy Information gain. The advantage of GI is its simplicity.

Information Gain
Information gain   is another   metric which tells   us how much a question unmixes the   labels at   a node. “Mathematically it is just a difference between impurity values before splitting the data at a node and the weighted average of the impurity after the split”. For instance, if we go back to our data of apples, lemons and grapes and ask the question “Is the color Green?”

The information gain by asking this question is 0.144. Similarly, we can ask another question from the set of possible questions split the data and compute information gain. This is also called (Recursive Binary Splitting).

 
The question where we have the highest information gain “Is diameter ≥ 3?” is the best question to ask. Note that the information gain is same for the question “Is the color red?” we just picked the first one at random.
Repeating the same method at the child node we can complete the tree. Note that no further questions can be asked which would increase the information gain.


Also note that the rightmost leaf which says 50% Apple & 50% lemon means that this class cannot be divided further, and this branch can tell an apple or a lemon with 50% probability. For the grape and apple branches we stop asking further questions since the Gini Impurity is 0 for those.

Entropy and Information Gain – ID3

ID3 (Iterative Dichotomiser 3) → uses Entropy function and Information gain as metrics.
 
 

If the sample is completely homogeneous the entropy is zero and if the sample is an equally divided it has entropy of one.

To build a decision tree, we need to calculate two types of entropy using frequency tables as follows:

a)	Entropy using the frequency table of one attribute:

 

 

Information Gain
The information gain is based on the decrease in entropy after a dataset is split on an attribute. Constructing a decision tree is all about finding attribute that returns the highest information gain (i.e., the most homogeneous branches).

Step 1: Calculate entropy of the target.


Step 2: The dataset is then split on the different attributes. The entropy for each branch is calculated. Then it is added proportionally, to get total entropy for the split. The resulting entropy is subtracted from the entropy before the split. The result is the Information Gain or decrease in entropy.

 

 

Step 3: Choose attribute with the largest information gain as the decision node, divide the dataset by its branches and repeat the same process on every branch.



 
 

Step 4b: A branch with entropy more than 0 needs further splitting.


Step 5: The ID3 algorithm is run recursively on the non-leaf branches, until all data is classified.

Q13.	What is pruning in Decision Tree?

Pruning is a technique in machine learning and search algorithms that reduces the size of decision trees by removing sections of the tree that provide little power to classify instances. So, when we remove sub- nodes of a decision node, this process is called pruning or opposite process of splitting.

Q14.	What is logistic regression? State an example when you have used logistic regression recently.
Logistic Regression often referred to as the logit model is a technique to predict the binary outcome from a linear combination of predictor variables. Since we are interested in a probability outcome, a line does not fit the model. Logistic Regression is a classification algorithm that works by trying to learn a function
 
that  approximates  P( |X).  It  makes  the  central  assumption  that  P(X| )  can  be  approximated  as  a sigmoid function applied to a linear combination of input features.







For example, if you want to predict whether a particular political leader will win the election or not. In this case, the outcome of prediction is binary i.e. 0 or 1 (Win/Lose). The predictor variables here would be the amount of money spent for election campaigning of a particular candidate, the amount of time spent in campaigning, etc.

Q15.	What is Linear Regression?
Linear regression is a statistical technique where the score of a variable Y is predicted from the score of a second variable X. X is referred to as the predictor variable and Y as the criterion variable.

 

 

Q16.	What Are the Drawbacks of the Linear Model?
Some drawbacks of the linear model are:
•	The assumption of linearity of the model
•	It can’t be used for count outcomes or binary outcomes.
•	There are overfitting or underfitting problems that it can’t solve.

Q17.	What	is	the	difference	between	Regression	and	classification	ML techniques?
Both Regression and classification machine learning techniques come under Supervised machine learning algorithms. In Supervised machine learning algorithm, we have to train the model using labelled data set, while training we have to explicitly provide the correct labels and algorithm tries to learn the pattern from input to output. If our labels are discrete values then it will a classification problem, but if our labels are continuous values then it will be a regression problem.

Q18.	What are Recommender Systems?
https://en.wikipedia.org/wiki/Recommender_system
 
Recommender Systems are a subclass of information filtering systems that are meant to predict the preferences or ratings that a user would give to a product. Recommender systems are widely used in movies, news, research articles, products, social tags, music, etc.
Examples include movie recommenders in IMDB, Netflix & BookMyShow, product recommenders in e- commerce sites like Amazon, eBay & Flipkart, YouTube video recommendations and game recommendations in Xbox.

Q19.	What is Collaborative filtering? And a content based?
The process of filtering used by most of the recommender systems to find patterns or information by collaborating viewpoints, various data sources and multiple agents. Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users. It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user. It looks at the items they like (usually based on rating) and combines them to create a ranked list of suggestions. Similar users are those with similar rating and on the based on that they get recommendations. In content based, we look only at the item level, recommending on similar items sold.


An example of collaborative filtering can be to predict the rating of a particular user based on his/her ratings for other movies and others’ ratings for all movies. This concept is widely used in recommending movies in IMDB, Netflix & BookMyShow, product recommenders in e-commerce sites like Amazon, eBay & Flipkart, YouTube video recommendations and game recommendations in Xbox.

Q20.	How can outlier values be treated?

Outlier values can be identified by using univariate or any other graphical analysis method. If the number of outlier values is few then they can be assessed individually but for a large number of outliers, the values can be substituted with either the 99th or the 1st percentile values.
All extreme values are not outlier values. The most common ways to treat outlier values:
1.	Change it with a mean or median
2.	Standardize the feature, changing the distribution but smoothing the outliers
3.	Log transform the feature (with many outliers)
4.	Drop the value



 
5.	First/third quartile value if more than 2σ

Q21.	What are the various steps involved in an analytics project?
The following are the various steps involved in an analytics project:
1.	Understand the Business problem
2.	Explore the data and become familiar with it
3.	Prepare the data for modeling by detecting outliers, treating missing values, transforming variables, etc.
4.	After data preparation, start running the model, analyze the result and tweak the approach. This is an iterative step until the best possible outcome is achieved.
5.	Validate the model using a new data set.
6.	Start implementing the model and track the result to analyze the performance of the model over the period of time.

Q22.	During analysis, how do you treat missing values?
The extent of the missing values is identified after identifying the variables with missing values. If any patterns are identified the analyst has to concentrate on them as it could lead to interesting and meaningful business insights.
If there are no patterns identified, then the missing values can be substituted with mean or median values (imputation) or they can simply be ignored. Assigning a default value which can be mean, minimum or maximum value. Getting into the data is important.
If it is a categorical variable, the default value is assigned. The missing value is assigned a default value. If you have a distribution of data coming, for normal distribution give the mean value.
If 80% of the values for a variable are missing, then you can answer that you would be dropping the variable instead of treating the missing values.

Q23.	How will you define the number of clusters in a clustering algorithm?
https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/

Though the Clustering Algorithm is not specified, this question is mostly in reference to K-Means clustering where “K” defines the number of clusters. The objective of clustering is to group similar entities in a way that the entities within a group are similar to each other, but the groups are different from each other.
 
For example, the following image shows three different groups.


Within Sum of squares is generally used to explain the homogeneity within a cluster. If you plot WSS (as the sum of the squared distance between each member of the cluster and its centroid) for a range of number of clusters, you will get the plot shown below.
•	The Graph is generally known as Elbow Curve.
•	Red circled a point in above graph i.e. Number of Cluster = 3 is the point after which you don’t see any decrement in WSS.
•	This point is known as the bending point and taken as K in K – Means.

 
This is the widely used approach but few data scientists also use Hierarchical clustering first to create dendrograms and identify the distinct groups from there.
The algorithm starts by finding the two points that are closest to each other on the basis of Euclidean distance. If we look back at Graph1, we can see that points 2 and 3 are closest to each other while points 7 and 8 are closes to each other. Therefore a cluster will be formed between these two points first. In Graph2, you can see that the dendograms have been created joining points 2 with 3, and 8 with 7. The vertical height of the dendogram shows the Euclidean distances between points. From Graph2, it can be seen that Euclidean distance between points 8 and 7 is greater than the distance between point 2 and 3. The next step is to join the cluster formed by joining two points to the next nearest cluster or point which in turn results in another cluster. If you look at Graph1, point 4 is closest to cluster of point 2 and 3, therefore in Graph2 dendrogram is generated by joining point 4 with dendrogram of point 2 and 3. This process continues until all the points are joined together to form one big cluster.
Once one big cluster is formed, the longest vertical distance without any horizontal line passing through it is selected and a horizontal line is drawn through it. The number of vertical lines this newly created horizontal line passes is equal to number of clusters. Take a look at the following plot:

 
We can see that the largest vertical distance without any horizontal line passing through it is represented by blue line. So we draw a new horizontal red line that passes through the blue line. Since it crosses the blue line at two points, therefore the number of clusters will be 2. Basically the horizontal line is a threshold, which defines the minimum distance required to be a separate cluster. If we draw a line further down, the threshold required to be a new cluster will be decreased and more clusters will be formed as see in the image below:


In the above plot, the horizontal line passes through four vertical lines resulting in four clusters: cluster of points 6,7,8 and 10, cluster of points 3,2,4 and points 9 and 5 will be treated as single point clusters.

Q24.	What is Ensemble Learning?

In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. Ensembles are a divide-and-conquer approach used to improve performance. The main principle behind ensemble methods is that a group of “weak learners” can come together to form a “strong

 
learner”. Each classifier, individually, is a “weak learner,” while all the classifiers taken together are a “strong learner”.


Q25.	Describe in brief any type of Ensemble Learning.
https://medium.com/@ruhi3929/bagging-and-boosting-method-c036236376eb

Ensemble learning has many types but two more popular ensemble learning techniques are mentioned below.

Bagging

Bagging tries to implement similar learners on small sample populations and then takes a mean of all the predictions. In generalized bagging, you can use different learners on different population. As you expect this helps us to reduce the variance error.

Pros 
➢	Bagging method helps when we face variance or overfitting in the model. It provides an environment to deal with variance by using N learners of same size on same algorithm.
➢	During the  sampling  of  train  data,  there  are  many  observations  which  overlaps.  So,  the
combination of these learners helps in overcoming the high variance.
➢	Bagging uses Bootstrap sampling method (Bootstrapping is any test or metric that uses random sampling with replacement and falls under the broader class of resampling methods.)
Cons 
➢	Bagging is not helpful in case of bias or underfitting in the data.
➢	Bagging ignores the value with the highest and the lowest result which may have a wide difference and provides an average result.

Boosting

Boosting is an iterative technique which adjusts the weight of an observation based on the last classification. If an observation was classified incorrectly, it tries to increase the weight of this observation and vice versa. Boosting in general decreases the bias error and builds strong predictive models. However, they may over fit on the training data.

Pros 
➢	Boosting technique takes care of the weightage of the higher accuracy sample and lower accuracy sample and then gives the combined results.
➢	Net error is evaluated in each learning steps. It works good with interactions.
➢	Boosting technique helps when we are dealing with bias or underfitting in the data set.
➢	Multiple boosting techniques are available. For example: AdaBoost, LPBoost, XGBoost, GradientBoost, BrownBoost
Cons 
➢	Boosting technique often ignores overfitting or variance issues in the data set.




 
➢	It increases the complexity of the classification.
➢	Time and computation can be a bit expensive.


There are multiple areas where Bagging and Boosting technique is used to boost the accuracy.
•	Banking: Loan defaulter prediction, fraud transaction
•	Credit risks
•	Kaggle competitions
•	Fraud detection
•	Recommender system for Netflix
•	Malware
•	Wildlife conservations and so on.

Q26.	What is a Random Forest? How does it work?
Random forest is a versatile machine learning method capable of performing:
•	regression
•	classification
•	dimensionality reduction
•	treat missing values
•	outlier values
 
It is a type of ensemble learning method, where a group of weak models combine to form a powerful model. The random forest starts with a standard machine learning technique called a “decision tree” which, in ensemble terms, corresponds to our weak learner. In a decision tree, an input is entered at the top and as it traverses down the tree the data gets bucketed into smaller and smaller sets.

In Random Forest, we grow multiple trees as opposed to a single tree. To classify a new object based on attributes, each tree gives a classification. The forest chooses the classification having the most votes (Over all the trees in the forest) and in case of regression, it takes the average of outputs by different trees.

Q27.	How Do You Work Towards a Random Forest?
https://blog.citizennet.com/blog/2012/11/10/random-forests-ensembles-and-performance-metrics

The underlying principle of this technique is that several weak learners combined to provide a keen learner. Here is how such a system is trained for some number of trees T:

1.	Sample N cases at random with replacement to create a subset of the data. The subset should be about 66% of the total set.
2.	At each node:
a.	For some number m (see below), m predictor variables are selected at random from all the predictor variables.
b.	The predictor variable that provides the best split, according to some objective function, is used to do a binary split on that node.
c.	At the next node, choose another m variables at random from all predictor variables and do the same.

Depending upon the value of m, there are three slightly different systems:
 

•	Random splitter selection: m = 1
•	Breiman’s bagger: m = total number of predictor variables (p)
•	Random forest: m ≪ number of predictor variables.
o Brieman suggests three possible values for m: 1 √p, √p,  2√p



When a new input is entered into the system, it is run down all of the trees. The result may either be an average or weighted average of all of the terminal nodes that are reached, or, in the case of categorical variables, a voting majority.

Note that:
➢	With a large number of predictors (p ≫ 0), the eligible predictor set (m) will be quite different from node to node.
➢	The greater the inter-tree correlation, the greater the random forest error rate, so one pressure
on the model is to have the trees as uncorrelated as possible.
➢	As m goes down, both inter-tree correlation and the strength of individual trees go down. So some optimal value of m must be discovered.
➢	Strengths: Random forest runtimes are quite fast, and they are able to deal with unbalanced and
missing data.
➢	Weaknesses: Random Forest used for regression cannot predict beyond the range in the training data, and that they may over-fit data sets that are particularly noisy. Of course, the best test of any algorithm is how well it works upon your own data set.

Q28.        What cross-validation technique would you use on a time series data set?
 
Instead of using k-fold cross-validation, you should be aware of the fact that a time series is not randomly distributed data — It is inherently ordered by chronological order.
In case of time series data, you should use techniques like forward=chaining — Where you will be model on past data then look at forward-facing data.

fold 1: training[1], test[2] fold 2: training[1 2], test[3]
fold 3: training[1 2 3], test[4]
fold 4: training[1 2 3 4], test[5]

Q29.	What is a Box-Cox Transformation?
The dependent variable for a regression analysis might not satisfy one or more assumptions of an ordinary least squares regression. The residuals could either curve as the prediction increases or follow the skewed distribution. In such scenarios, it is necessary to transform the response variable so that the data meets the required assumptions. A Box-Cox transformation is a statistical technique to transform non-normal dependent variables into a normal shape. If the given data is not normal then most of the statistical techniques assume normality. Applying a Box-Cox transformation means that you can run a broader number of tests.
A Box-Cox transformation is a way to transform non-normal dependent variables into a normal shape. Normality is an important assumption for many statistical techniques, if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests. The Box-Cox transformation is named after statisticians George Box and Sir David Roxbee Cox who collaborated on a 1964 paper and developed the technique.

Q30.	How Regularly Must an Algorithm be Updated?
You will want to update an algorithm when:
•	You want the model to evolve as data streams through infrastructure
•	The underlying data source is changing
•	There is a case of non-stationarity (mean, variance change over the time)
•	The algorithm underperforms/results lack accuracy

Q31.    If you are having 4GB RAM in your machine and you want to train your model on 10GB data set. How would you go about this problem? Have you ever faced this kind of problem in your machine learning/data science experience so far?
First of all, you have to ask which ML model you want to train.
For Neural networks: Batch size with Numpy array will work. Steps:
1.	Load the whole data in the Numpy array. Numpy array has a property to create a mapping of the complete data set, it doesn’t load complete data set in memory.
2.	You can pass an index to Numpy array to get required data.
3.	Use this data to pass to the Neural network.
4.	Have a small batch size.
For SVM: Partial fit will work. Steps:
1.	Divide one big data set in small size data sets.




 
2.	Use a partial fit method of SVM, it requires a subset of the complete data set.
3.	Repeat step 2 for other subsets.
However, you could actually face such an issue in reality. So, you could check out the best laptop for Machine Learning to prevent that. Having said that, let’s move on to some questions on deep learning.

### More questions

:::note Source
https://github.com/andrewekhalel/MLQuestions
:::

:::tip Resources
https://leetcode.com/discuss/career/807563/preparing-for-machine-learning-engineer-role-vs-software-engineer-role-in-india
:::

#### [](https://github.com/andrewekhalel/MLQuestions#1-whats-the-trade-off-between-bias-and-variance-src)1) What's the trade-off between bias and variance? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.  [[src]](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)

#### [](https://github.com/andrewekhalel/MLQuestions#2-what-is-gradient-descent-src)2) What is gradient descent? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

[[Answer]](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)

Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).

Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.

#### [](https://github.com/andrewekhalel/MLQuestions#3-explain-over--and-under-fitting-and-how-to-combat-them-src)3) Explain over- and under-fitting and how to combat them? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

[[Answer]](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)

#### [](https://github.com/andrewekhalel/MLQuestions#4-how-do-you-combat-the-curse-of-dimensionality-src)4) How do you combat the curse of dimensionality? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

-   Manual Feature Selection
-   Principal Component Analysis (PCA)
-   Multidimensional Scaling
-   Locally linear embedding  
    [[src]](https://towardsdatascience.com/why-and-how-to-get-rid-of-the-curse-of-dimensionality-right-with-breast-cancer-dataset-7d528fb5f6c0)

#### [](https://github.com/andrewekhalel/MLQuestions#5-what-is-regularization-why-do-we-use-it-and-give-some-examples-of-common-methods-src)5) What is regularization, why do we use it, and give some examples of common methods? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

A technique that discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. Examples

-   Ridge (L2 norm)
-   Lasso (L1 norm)  
    The obvious  _disadvantage_  of  **ridge**  regression, is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the  _final model will include all predictors_. However, in the case of the  **lasso**, the L1 penalty has the effect of forcing some of the coefficient estimates to be  _exactly equal_  to zero when the tuning parameter λ is sufficiently large. Therefore, the lasso method also performs variable selection and is said to yield sparse models.  [[src]](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

#### [](https://github.com/andrewekhalel/MLQuestions#6-explain-principal-component-analysis-pca-src)6) Explain Principal Component Analysis (PCA)? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

[[Answer]](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)

#### [](https://github.com/andrewekhalel/MLQuestions#7-why-is-relu-better-and-more-often-used-than-sigmoid-in-neural-networks-src)7) Why is ReLU better and more often used than Sigmoid in Neural Networks? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

Imagine a network with random initialized weights ( or normalised ) and almost 50% of the network yields 0 activation because of the characteristic of ReLu ( output 0 for negative values of x ). This means a fewer neurons are firing ( sparse activation ) and the network is lighter.  [[src]](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

#### [](https://github.com/andrewekhalel/MLQuestions#8-given-stride-s-and-kernel-sizes--for-each-layer-of-a-1-dimensional-cnn-create-a-function-to-compute-the-receptive-field-of-a-particular-node-in-the-network-this-is-just-finding-how-many-input-nodes-actually-connect-through-to-a-neuron-in-a-cnn-src)8) Given stride S and kernel sizes for each layer of a (1-dimensional) CNN, create a function to compute the  [receptive field](https://www.quora.com/What-is-a-receptive-field-in-a-convolutional-neural-network)  of a particular node in the network. This is just finding how many input nodes actually connect through to a neuron in a CNN. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

The receptive field are defined portion of space within an inputs that will be used during an operation to generate an output.

Considering a CNN filter of size k, the receptive field of a peculiar layer is only the number of input used by the filter, in this case k, multiplied by the dimension of the input that is not being reduced by the convolutionnal filter a. This results in a receptive field of k*a.

More visually, in the case of an image of size 32x32x3, with a CNN with a filter size of 5x5, the corresponding recpetive field will be the the filter size, 5 multiplied by the depth of the input volume (the RGB colors) which is the color dimensio. This thus gives us a recpetive field of dimension 5x5x3.

#### [](https://github.com/andrewekhalel/MLQuestions#9-implement-connected-components-on-an-imagematrix-src)9) Implement  [connected components](http://aishack.in/tutorials/labelling-connected-components-example/)  on an image/matrix. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### [](https://github.com/andrewekhalel/MLQuestions#10-implement-a-sparse-matrix-class-in-c-src)10) Implement a sparse matrix class in C++. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://www.geeksforgeeks.org/sparse-matrix-representation/)

#### [](https://github.com/andrewekhalel/MLQuestions#11-create-a-function-to-compute-an-integral-image-and-create-another-function-to-get-area-sums-from-the-integral-imagesrc)11) Create a function to compute an  [integral image](https://en.wikipedia.org/wiki/Summed-area_table), and create another function to get area sums from the integral image.[[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://www.geeksforgeeks.org/submatrix-sum-queries/)

#### [](https://github.com/andrewekhalel/MLQuestions#12-how-would-you-remove-outliers-when-trying-to-estimate-a-flat-plane-from-noisy-samples-src)12) How would you remove outliers when trying to estimate a flat plane from noisy samples? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates.  [[src]](https://en.wikipedia.org/wiki/Random_sample_consensus)

#### [](https://github.com/andrewekhalel/MLQuestions#13-how-does-cbir-work-src)13) How does  [CBIR](https://www.robots.ox.ac.uk/~vgg/publications/2013/arandjelovic13/arandjelovic13.pdf)  work? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://en.wikipedia.org/wiki/Content-based_image_retrieval)  Content-based image retrieval is the concept of using images to gather metadata on their content. Compared to the current image retrieval approach based on the keywords associated to the images, this technique generates its metadata from computer vision techniques to extract the relevant informations that will be used during the querying step. Many approach are possible from feature detection to retrieve keywords to the usage of CNN to extract dense features that will be associated to a known distribution of keywords.

With this last approach, we care less about what is shown on the image but more about the similarity between the metadata generated by a known image and a list of known label and or tags projected into this metadata space.

#### [](https://github.com/andrewekhalel/MLQuestions#14-how-does-image-registration-work-sparse-vs-dense-optical-flow-and-so-on-src)14) How does image registration work? Sparse vs. dense  [optical flow](http://www.ncorr.com/download/publications/bakerunify.pdf)  and so on. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### [](https://github.com/andrewekhalel/MLQuestions#15-describe-how-convolution-works-what-about-if-your-inputs-are-grayscale-vs-rgb-imagery-what-determines-the-shape-of-the-next-layer-src)15) Describe how convolution works. What about if your inputs are grayscale vs RGB imagery? What determines the shape of the next layer? [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://dev.to/sandeepbalachandran/machine-learning-convolution-with-color-images-2p41)

#### [](https://github.com/andrewekhalel/MLQuestions#16-talk-me-through-how-you-would-create-a-3d-model-of-an-object-from-imagery-and-depth-sensor-measurements-taken-at-all-angles-around-the-object-src)16) Talk me through how you would create a 3D model of an object from imagery and depth sensor measurements taken at all angles around the object. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### [](https://github.com/andrewekhalel/MLQuestions#17-implement-sqrtconst-double--x-without-using-any-special-functions-just-fundamental-arithmetic-src)17) Implement SQRT(const double & x) without using any special functions, just fundamental arithmetic. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

The taylor series can be used for this step by providing an approximation of sqrt(x):

[[Answer]](https://math.stackexchange.com/questions/732540/taylor-series-of-sqrt1x-using-sigma-notation)

#### [](https://github.com/andrewekhalel/MLQuestions#18-reverse-a-bitstring-src)18) Reverse a bitstring. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

If you are using python3 :

```
data = b'\xAD\xDE\xDE\xC0'
my_data = bytearray(data)
my_data.reverse()

```

#### [](https://github.com/andrewekhalel/MLQuestions#19-implement-non-maximal-suppression-as-efficiently-as-you-can-src)19) Implement non maximal suppression as efficiently as you can. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

#### [](https://github.com/andrewekhalel/MLQuestions#20-reverse-a-linked-list-in-place-src)20) Reverse a linked list in place. [[src](https://www.reddit.com/r/computervision/comments/7gku4z/technical_interview_questions_in_cv/)]

[[Answer]](https://www.geeksforgeeks.org/reverse-a-linked-list/)

#### [](https://github.com/andrewekhalel/MLQuestions#21-what-is-data-normalization-and-why-do-we-need-it-src)21) What is data normalization and why do we need it? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

Data normalization is very important preprocessing step, used to rescale values to fit in a specific range to assure better convergence during backpropagation. In general, it boils down to subtracting the mean of each data point and dividing by its standard deviation. If we don't do this then some of the features (those with high magnitude) will be weighted more in the cost function (if a higher-magnitude feature changes by 1%, then that change is pretty big, but for smaller features it's quite insignificant). The data normalization makes all features weighted equally.

#### [](https://github.com/andrewekhalel/MLQuestions#22-why-do-we-use-convolutions-for-images-rather-than-just-fc-layers-src)22) Why do we use convolutions for images rather than just FC layers? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

Firstly, convolutions preserve, encode, and actually use the spatial information from the image. If we used only FC layers we would have no relative spatial information. Secondly, Convolutional Neural Networks (CNNs) have a partially built-in translation in-variance, since each convolution kernel acts as it's own filter/feature detector.

#### [](https://github.com/andrewekhalel/MLQuestions#23-what-makes-cnns-translation-invariant-src)23) What makes CNNs translation invariant? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

As explained above, each convolution kernel acts as it's own filter/feature detector. So let's say you're doing object detection, it doesn't matter where in the image the object is since we're going to apply the convolution in a sliding window fashion across the entire image anyways.

#### [](https://github.com/andrewekhalel/MLQuestions#24-why-do-we-have-max-pooling-in-classification-cnns-src)24) Why do we have max-pooling in classification CNNs? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

for a role in Computer Vision. Max-pooling in a CNN allows you to reduce computation since your feature maps are smaller after the pooling. You don't lose too much semantic information since you're taking the maximum activation. There's also a theory that max-pooling contributes a bit to giving CNNs more translation in-variance. Check out this great video from Andrew Ng on the  [benefits of max-pooling](https://www.coursera.org/learn/convolutional-neural-networks/lecture/hELHk/pooling-layers).

#### [](https://github.com/andrewekhalel/MLQuestions#25-why-do-segmentation-cnns-typically-have-an-encoder-decoder-style--structure-src)25) Why do segmentation CNNs typically have an encoder-decoder style / structure? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

The encoder CNN can basically be thought of as a feature extraction network, while the decoder uses that information to predict the image segments by "decoding" the features and upscaling to the original image size.

#### [](https://github.com/andrewekhalel/MLQuestions#26-what-is-the-significance-of-residual-networks-src)26) What is the significance of Residual Networks? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

The main thing that residual connections did was allow for direct feature access from previous layers. This makes information propagation throughout the network much easier. One very interesting paper about this shows how using local skip connections gives the network a type of ensemble multi-path structure, giving features multiple paths to propagate throughout the network.

#### [](https://github.com/andrewekhalel/MLQuestions#27-what-is-batch-normalization-and-why-does-it-work-src)27) What is batch normalization and why does it work? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. The idea is then to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. This is done for each individual mini-batch at each layer i.e compute the mean and variance of that mini-batch alone, then normalize. This is analogous to how the inputs to networks are standardized. How does this help? We know that normalizing the inputs to a network helps it learn. But a network is just a series of layers, where the output of one layer becomes the input to the next. That means we can think of any layer in a neural network as the first layer of a smaller subsequent network. Thought of as a series of neural networks feeding into each other, we normalize the output of one layer before applying the activation function, and then feed it into the following layer (sub-network).

#### [](https://github.com/andrewekhalel/MLQuestions#28-why-would-you-use-many-small-convolutional-kernels-such-as-3x3-rather-than-a-few-large-ones-src)28) Why would you use many small convolutional kernels such as 3x3 rather than a few large ones? [[src](http://houseofbots.com/news-detail/2849-4-data-science-and-machine-learning-interview-questions)]

This is very well explained in the  [VGGNet paper](https://arxiv.org/pdf/1409.1556.pdf). There are 2 reasons: First, you can use several smaller kernels rather than few large ones to get the same receptive field and capture more spatial context, but with the smaller kernels you are using less parameters and computations. Secondly, because with smaller kernels you will be using more filters, you'll be able to use more activation functions and thus have a more discriminative mapping function being learned by your CNN.

#### [](https://github.com/andrewekhalel/MLQuestions#29-why-do-we-need-a-validation-set-and-test-set-what-is-the-difference-between-them-src)29) Why do we need a validation set and test set? What is the difference between them? [[src](https://www.toptal.com/machine-learning/interview-questions)]

When training a model, we divide the available data into three separate sets:

-   The training dataset is used for fitting the model’s parameters. However, the accuracy that we achieve on the training set is not reliable for predicting if the model will be accurate on new samples.
-   The validation dataset is used to measure how well the model does on examples that weren’t part of the training dataset. The metrics computed on the validation data can be used to tune the hyperparameters of the model. However, every time we evaluate the validation data and we make decisions based on those scores, we are leaking information from the validation data into our model. The more evaluations, the more information is leaked. So we can end up overfitting to the validation data, and once again the validation score won’t be reliable for predicting the behaviour of the model in the real world.
-   The test dataset is used to measure how well the model does on previously unseen examples. It should only be used once we have tuned the parameters using the validation set.

So if we omit the test set and only use a validation set, the validation score won’t be a good estimate of the generalization of the model.

#### [](https://github.com/andrewekhalel/MLQuestions#30-what-is-stratified-cross-validation-and-when-should-we-use-it-src)30) What is stratified cross-validation and when should we use it? [[src](https://www.toptal.com/machine-learning/interview-questions)]

Cross-validation is a technique for dividing data between training and validation sets. On typical cross-validation this split is done randomly. But in stratified cross-validation, the split preserves the ratio of the categories on both the training and validation datasets.

For example, if we have a dataset with 10% of category A and 90% of category B, and we use stratified cross-validation, we will have the same proportions in training and validation. In contrast, if we use simple cross-validation, in the worst case we may find that there are no samples of category A in the validation set.

Stratified cross-validation may be applied in the following scenarios:

-   On a dataset with multiple categories. The smaller the dataset and the more imbalanced the categories, the more important it will be to use stratified cross-validation.
-   On a dataset with data of different distributions. For example, in a dataset for autonomous driving, we may have images taken during the day and at night. If we do not ensure that both types are present in training and validation, we will have generalization problems.

#### [](https://github.com/andrewekhalel/MLQuestions#31-why-do-ensembles-typically-have-higher-scores-than-individual-models-src)31) Why do ensembles typically have higher scores than individual models? [[src](https://www.toptal.com/machine-learning/interview-questions)]

An ensemble is the combination of multiple models to create a single prediction. The key idea for making better predictions is that the models should make different errors. That way the errors of one model will be compensated by the right guesses of the other models and thus the score of the ensemble will be higher.

We need diverse models for creating an ensemble. Diversity can be achieved by:

-   Using different ML algorithms. For example, you can combine logistic regression, k-nearest neighbors, and decision trees.
-   Using different subsets of the data for training. This is called bagging.
-   Giving a different weight to each of the samples of the training set. If this is done iteratively, weighting the samples according to the errors of the ensemble, it’s called boosting. Many winning solutions to data science competitions are ensembles. However, in real-life machine learning projects, engineers need to find a balance between execution time and accuracy.

#### [](https://github.com/andrewekhalel/MLQuestions#32-what-is-an-imbalanced-dataset-can-you-list-some-ways-to-deal-with-it-src)32) What is an imbalanced dataset? Can you list some ways to deal with it? [[src](https://www.toptal.com/machine-learning/interview-questions)]

An imbalanced dataset is one that has different proportions of target categories. For example, a dataset with medical images where we have to detect some illness will typically have many more negative samples than positive samples—say, 98% of images are without the illness and 2% of images are with the illness.

There are different options to deal with imbalanced datasets:

-   Oversampling or undersampling. Instead of sampling with a uniform distribution from the training dataset, we can use other distributions so the model sees a more balanced dataset.
-   Data augmentation. We can add data in the less frequent categories by modifying existing data in a controlled way. In the example dataset, we could flip the images with illnesses, or add noise to copies of the images in such a way that the illness remains visible.
-   Using appropriate metrics. In the example dataset, if we had a model that always made negative predictions, it would achieve a precision of 98%. There are other metrics such as precision, recall, and F-score that describe the accuracy of the model better when using an imbalanced dataset.

#### [](https://github.com/andrewekhalel/MLQuestions#33-can-you-explain-the-differences-between-supervised-unsupervised-and-reinforcement-learning-src)33) Can you explain the differences between supervised, unsupervised, and reinforcement learning? [[src](https://www.toptal.com/machine-learning/interview-questions)]

In supervised learning, we train a model to learn the relationship between input data and output data. We need to have labeled data to be able to do supervised learning.

With unsupervised learning, we only have unlabeled data. The model learns a representation of the data. Unsupervised learning is frequently used to initialize the parameters of the model when we have a lot of unlabeled data and a small fraction of labeled data. We first train an unsupervised model and, after that, we use the weights of the model to train a supervised model.

In reinforcement learning, the model has some input data and a reward depending on the output of the model. The model learns a policy that maximizes the reward. Reinforcement learning has been applied successfully to strategic games such as Go and even classic Atari video games.

#### [](https://github.com/andrewekhalel/MLQuestions#34-what-is-data-augmentation-can-you-give-some-examples-src)34) What is data augmentation? Can you give some examples? [[src](https://www.toptal.com/machine-learning/interview-questions)]

Data augmentation is a technique for synthesizing new data by modifying existing data in such a way that the target is not changed, or it is changed in a known way.

Computer vision is one of fields where data augmentation is very useful. There are many modifications that we can do to images:

-   Resize
-   Horizontal or vertical flip
-   Rotate
-   Add noise
-   Deform
-   Modify colors Each problem needs a customized data augmentation pipeline. For example, on OCR, doing flips will change the text and won’t be beneficial; however, resizes and small rotations may help.

#### [](https://github.com/andrewekhalel/MLQuestions#35-what-is-turing-test-src)35) What is Turing test? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

The Turing test is a method to test the machine’s ability to match the human level intelligence. A machine is used to challenge the human intelligence that when it passes the test, it is considered as intelligent. Yet a machine could be viewed as intelligent without sufficiently knowing about people to mimic a human.

#### [](https://github.com/andrewekhalel/MLQuestions#36-what-is-precision)36) What is Precision?

Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances  
Precision = true positive / (true positive + false positive)  
[[src]](https://en.wikipedia.org/wiki/Precision_and_recall)

#### [](https://github.com/andrewekhalel/MLQuestions#37-what-is-recall)37) What is Recall?

Recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. Recall = true positive / (true positive + false negative)  
[[src]](https://en.wikipedia.org/wiki/Precision_and_recall)

#### [](https://github.com/andrewekhalel/MLQuestions#38-define-f1-score-src)38) Define F1-score. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

It is the weighted average of precision and recall. It considers both false positive and false negative into account. It is used to measure the model’s performance.  
F1-Score = 2 * (precision * recall) / (precision + recall)

#### [](https://github.com/andrewekhalel/MLQuestions#39-what-is-cost-function-src)39) What is cost function? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

Cost function is a scalar functions which Quantifies the error factor of the Neural Network. Lower the cost function better the Neural network. Eg: MNIST Data set to classify the image, input image is digit 2 and the Neural network wrongly predicts it to be 3

#### [](https://github.com/andrewekhalel/MLQuestions#40-list-different-activation-neurons-or-functions-src)40) List different activation neurons or functions. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

-   Linear Neuron
-   Binary Threshold Neuron
-   Stochastic Binary Neuron
-   Sigmoid Neuron
-   Tanh function
-   Rectified Linear Unit (ReLU)

#### [](https://github.com/andrewekhalel/MLQuestions#41-define-learning-rate)41) Define Learning Rate.

Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. [[src](https://en.wikipedia.org/wiki/Learning_rate)]

#### [](https://github.com/andrewekhalel/MLQuestions#42-what-is-momentum-wrt-nn-optimization)42) What is Momentum (w.r.t NN optimization)?

Momentum lets the optimization algorithm remembers its last step, and adds some proportion of it to the current step. This way, even if the algorithm is stuck in a flat region, or a small local minimum, it can get out and continue towards the true minimum.  [[src]](https://www.quora.com/What-is-the-difference-between-momentum-and-learning-rate)

#### [](https://github.com/andrewekhalel/MLQuestions#43-what-is-the-difference-between-batch-gradient-descent-and-stochastic-gradient-descent)43) What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?

Batch gradient descent computes the gradient using the whole dataset. This is great for convex, or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution, either local or global. Additionally, batch gradient descent, given an annealed learning rate, will eventually find the minimum located in it's basin of attraction.

Stochastic gradient descent (SGD) computes the gradient using a single sample. SGD works well (Not well, I suppose, but better than batch gradient descent) for error manifolds that have lots of local maxima/minima. In this case, the somewhat noisier gradient calculated using the reduced number of samples tends to jerk the model out of local minima into a region that hopefully is more optimal.  [[src]](https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent)

#### [](https://github.com/andrewekhalel/MLQuestions#44-epoch-vs-batch-vs-iteration)44) Epoch vs. Batch vs. Iteration.

-   **Epoch**: one forward pass and one backward pass of  **all**  the training examples
-   **Batch**: examples processed together in one pass (forward and backward)
-   **Iteration**: number of training examples / Batch size

#### [](https://github.com/andrewekhalel/MLQuestions#45-what-is-vanishing-gradient-src)45) What is vanishing gradient? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

As we add more and more hidden layers, back propagation becomes less and less useful in passing information to the lower layers. In effect, as information is passed back, the gradients begin to vanish and become small relative to the weights of the networks.

#### [](https://github.com/andrewekhalel/MLQuestions#46-what-are-dropouts-src)46) What are dropouts? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

Dropout is a simple way to prevent a neural network from overfitting. It is the dropping out of some of the units in a neural network. It is similar to the natural reproduction process, where the nature produces offsprings by combining distinct genes (dropping out others) rather than strengthening the co-adapting of them.

#### [](https://github.com/andrewekhalel/MLQuestions#47-define-lstm-src)47) Define LSTM. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

Long Short Term Memory – are explicitly designed to address the long term dependency problem, by maintaining a state what to remember and what to forget.

#### [](https://github.com/andrewekhalel/MLQuestions#48-list-the-key-components-of-lstm-src)48) List the key components of LSTM. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

-   Gates (forget, Memory, update & Read)
-   tanh(x) (values between -1 to 1)
-   Sigmoid(x) (values between 0 to 1)

#### [](https://github.com/andrewekhalel/MLQuestions#49-list-the-variants-of-rnn-src)49) List the variants of RNN. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

-   LSTM: Long Short Term Memory
-   GRU: Gated Recurrent Unit
-   End to End Network
-   Memory Network

#### [](https://github.com/andrewekhalel/MLQuestions#50-what-is-autoencoder-name-few-applications-src)50) What is Autoencoder, name few applications. [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

Auto encoder is basically used to learn a compressed form of given data. Few applications include

-   Data denoising
-   Dimensionality reduction
-   Image reconstruction
-   Image colorization

#### [](https://github.com/andrewekhalel/MLQuestions#51-what-are-the-components-of-gan-src)51) What are the components of GAN? [[src](https://intellipaat.com/interview-question/artificial-intelligence-interview-questions/)]

-   Generator
-   Discriminator

#### [](https://github.com/andrewekhalel/MLQuestions#52-whats-the-difference-between-boosting-and-bagging)52) What's the difference between boosting and bagging?

Boosting and bagging are similar, in that they are both ensembling techniques, where a number of weak learners (classifiers/regressors that are barely better than guessing) combine (through averaging or max vote) to create a strong learner that can make accurate predictions. Bagging means that you take bootstrap samples (with replacement) of your data set and each sample trains a (potentially) weak learner. Boosting, on the other hand, uses all data to train each learner, but instances that were misclassified by the previous learners are given more weight so that subsequent learners give more focus to them during training.  [[src]](https://www.quora.com/Whats-the-difference-between-boosting-and-bagging)

#### [](https://github.com/andrewekhalel/MLQuestions#53-explain-how-a-roc-curve-works-src)53) Explain how a ROC curve works.  [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)

The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).

#### [](https://github.com/andrewekhalel/MLQuestions#54-whats-the-difference-between-type-i-and-type-ii-error-src)54) What’s the difference between Type I and Type II error?  [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)

Type I error is a false positive, while Type II error is a false negative. Briefly stated, Type I error means claiming something has happened when it hasn’t, while Type II error means that you claim nothing is happening when in fact something is. A clever way to think about this is to think of Type I error as telling a man he is pregnant, while Type II error means you tell a pregnant woman she isn’t carrying a baby.

#### [](https://github.com/andrewekhalel/MLQuestions#55-whats-the-difference-between-a-generative-and-discriminative-model-src)55) What’s the difference between a generative and discriminative model?  [[src]](https://www.springboard.com/blog/machine-learning-interview-questions/)

A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.

#### [](https://github.com/andrewekhalel/MLQuestions#56-instance-based-versus-model-based-learning)56) Instance-Based Versus Model-Based Learning.

-   **Instance-based Learning**: The system learns the examples by heart, then generalizes to new cases using a similarity measure.
    
-   **Model-based Learning**: Another way to generalize from a set of examples is to build a model of these examples, then use that model to make predictions. This is called model-based learning.  [[src]](https://medium.com/@sanidhyaagrawal08/what-is-instance-based-and-model-based-learning-s1e10-8e68364ae084)
    

#### [](https://github.com/andrewekhalel/MLQuestions#57-when-to-use-a-label-encoding-vs-one-hot-encoding)57) When to use a Label Encoding vs. One Hot Encoding?

This question generally depends on your dataset and the model which you wish to apply. But still, a few points to note before choosing the right encoding technique for your model:

We apply One-Hot Encoding when:

-   The categorical feature is not ordinal (like the countries above)
    
-   The number of categorical features is less so one-hot encoding can be effectively applied We apply Label Encoding when:
    
-   The categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school)
    
-   The number of categories is quite large as one-hot encoding can lead to high memory consumption
    

[[src]](https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/)

#### [](https://github.com/andrewekhalel/MLQuestions#58-what-is-the-difference-between-lda-and-pca-for-dimensionality-reduction)58) What is the difference between LDA and PCA for dimensionality reduction?

Both LDA and PCA are linear transformation techniques: LDA is a supervised whereas PCA is unsupervised – PCA ignores class labels.

We can picture PCA as a technique that finds the directions of maximal variance. In contrast to PCA, LDA attempts to find a feature subspace that maximizes class separability.

[[src]](https://sebastianraschka.com/faq/docs/lda-vs-pca.html)

#### [](https://github.com/andrewekhalel/MLQuestions#59-what-is-t-sne)59) What is t-SNE?

t-Distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space.

[[src]](https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1)

#### [](https://github.com/andrewekhalel/MLQuestions#60-what-is-the-difference-between-t-sne-and-pca-for-dimensionality-reduction)60) What is the difference between t-SNE and PCA for dimensionality reduction?

The first thing to note is that PCA was developed in 1933 while t-SNE was developed in 2008. A lot has changed in the world of data science since 1933 mainly in the realm of compute and size of data. Second, PCA is a linear dimension reduction technique that seeks to maximize variance and preserves large pairwise distances. In other words, things that are different end up far apart. This can lead to poor visualization especially when dealing with non-linear manifold structures. Think of a manifold structure as any geometric shape like: cylinder, ball, curve, etc.

t-SNE differs from PCA by preserving only small pairwise distances or local similarities whereas PCA is concerned with preserving large pairwise distances to maximize variance.

[[src]](https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1)

#### [](https://github.com/andrewekhalel/MLQuestions#61-what-is-umap)61) What is UMAP?

UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data.

[[src]](https://arxiv.org/abs/1802.03426#:~:text=UMAP%20)

#### [](https://github.com/andrewekhalel/MLQuestions#62-what-is-the-difference-between-t-sne-and-umap-for-dimensionality-reduction)62) What is the difference between t-SNE and UMAP for dimensionality reduction?

The biggest difference between the the output of UMAP when compared with t-SNE is this balance between local and global structure - UMAP is often better at preserving global structure in the final projection. This means that the inter-cluster relations are potentially more meaningful than in t-SNE. However, it's important to note that, because UMAP and t-SNE both necessarily warp the high-dimensional shape of the data when projecting to lower dimensions, any given axis or distance in lower dimensions still isn’t directly interpretable in the way of techniques such as PCA.

[[src]](https://pair-code.github.io/understanding-umap/)

#### [](https://github.com/andrewekhalel/MLQuestions#63-how-random-number-generator-works-eg-rand-function-in-python-works)63) How Random Number Generator Works, e.g. rand() function in python works?

It generates a pseudo random number based on the seed and there are some famous algorithm, please see below link for further information on this.  [[src]](https://en.wikipedia.org/wiki/Linear_congruential_generator)