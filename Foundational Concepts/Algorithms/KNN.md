[[ML Algorithms]] 

#### KNN
- supervised learning algorithm both for classification and regression.
- principle is to find the predefined number of training samples closest to the new point, and predict the label from these training samples
When a new point comes, the algorithm will follow these steps:
1. Calculate the Euclidean distance between the new point and all training data
2. Pick the top-K closest training data
3. For 
	1. For regression problem, take the average of the labels as the result; 
	2. for classification problem, take the most common label of these labels as the result.
	   Conditional probability of $Y$ belonging to class $j$ i.e $Y_j$ i.e. $\hat{P}(Y_j \mid X = x_o) = \frac{1}{K}\sum_{i \in Neighbourhood} I(y_i = j)$
	   Then KNN applies Bayes Rule to classify the query point to a class

```python
#numpy implementation of Knn
def KNN(training_data, query, k, func):
    """
    training_data: all training data points
    query: new query point
    k: user-defined constant, number of closest training data
    func: functions used to get the the target label
    """
    # Step one: calculate the Euclidean distance between the new point and all training data
    neighbors= []
    for index, data in enumerate(training_data):
        # distance between the target data and the current example from the data.
        distance = euclidean_distance(data[:-1], query)
        neighbors.append((distance, index))

    # Step two: pick the top-K closest training data
    sorted_neighbors = sorted(neighbors)
    k_nearest = sorted_neighbors[:k]
    k_nearest_labels = [training_data[i][1] for distance, i in k_nearest]

    # Step three: For regression problem, take the average of the labels as the result;
    #             for classification problem, take the most common label of these labels as the result.
    return k_nearest, func(k_nearest_labels)
```
