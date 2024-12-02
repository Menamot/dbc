# Discrete Bayesian Classifier

`DiscreteBayesianClassifier` is a classification model that works by first partitioning the feature space into multiple small profiles using various discretization methods. It then calculates the class conditional risk on each profile to assign labels.

## Key Features

- **Discretization Methods**:
  - **KMeans**: Based on the K-means clustering algorithm.
  - **FCM** (Fuzzy C-Means): Based on fuzzy C-means clustering。
  - **DT** (Decision Tree): Based on decision tree discretization. (to be implemented)

### How to install

To install the `dbc-menamot` package, run the following command in your terminal:

```sh
pip install dbc-menamot
```

Make sure you have activated the correct Python environment to avoid potential dependency conflicts.

### Example of KMeans Discretization

Below is an example of how to use the KMeans discretization method:

```python
from dbc import KmeansDiscreteBayesianClassifier
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Create classifier instance
clf = KmeansDiscreteBayesianClassifier(n_clusters=10)

# Fit model
clf.fit(X, y)

# Predict
y_pred = clf.predict(X)
print(y_pred)
```

## Reference

- [1] C. Gilet, “Classifieur Minimax Discret pour l’aide  au Diagnostic Médical dans la  Médecine Personnalisée,” Université Côte d’Azur, 2021.
- [2] C. Gilet, S. Barbosa, and L. Fillatre, “Discrete Box-Constrained Minimax Classifier for Uncertain and Imbalanced Class Proportions,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 2923–2937, Jun. 2022, doi: 10.1109/TPAMI.2020.3046439.
- [3] Chen, Wenlong, et al. "Robust Discrete Bayesian Classifier Under Covariate and Label Noise." International Conference on Scalable Uncertainty Management. Cham: Springer Nature Switzerland, 2024.



## Contribution

Contributions to this project are welcome. Please submit feature requests and bug reports. If you would like to contribute code, please submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.