# Discrete Bayesian Classifier

`DiscreteBayesianClassifier` is a classification model that works by first partitioning the feature space into multiple small profiles using various discretization methods. It then calculates the class conditional risk on each profile to assign labels.

## Key Features

- **Discretization Methods**:
  - **KMeans**: Based on the K-means clustering algorithm.
  - **FCM** (Fuzzy C-Means): Based on fuzzy C-means clustering。
  - **DT** (Decision Tree): Based on decision tree discretization. (to be implemented)

### Example of KMeans Discretization

Below is an example of how to use the KMeans discretization method:

```python
import numpy as np
from dbc import DiscreteBayesianClassifier
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Create classifier instance
clf = DiscreteBayesianClassifier(discretization_method="kmeans", discretization_params={"n_clusters": 3})

# Fit model
clf.fit(X, y)

# Predict
y_pred = clf.predict(X, loss_function=np.identity(len(np.unique(y))))
print(y_pred)
```

## Reference

- [1] C. Gilet, “Classifieur Minimax Discret pour l’aide  au Diagnostic Médical dans la  Médecine Personnalisée,” Université Côte d’Azur, 2021.


## Contribution

Contributions to this project are welcome. Please submit feature requests and bug reports. If you would like to contribute code, please submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.