# Introduction
This is a Python package that implement two kinds classes: `DiscreteBayesianClassifier` and 
`DiscreteMinimaxClassifier`.

`DiscreteBayesianClassifier` is a classification model that works by first partitioning the feature space into multiple
small profiles using various discretization methods. It then implements Bayesian decision rule to get the result.

`DiscreteMinimaxClassifier` is a classifier proposed by Cyprien Gilet and it base on `DiscreteBayesianClassifier`. By 
calculating a best prior probability that minimize the maximum class conditional risk, `DiscreteMinimaxClassifier` can
balance the risk of different classes, and also make it perform well in the face of imbalanced datasets. 

## Discretization Methods Implemented
- **K-Means** : `KmeansDiscreteBayesianClassifier` and `KmeansDiscreteMinimaxClassifier`
- **Decision Tree** : `DecisionTreeDiscreteBayesianClassifier` and `KmeansDiscreteMinimaxClassifier`
- **Fuzzy C-Means** : `CmeansDiscreteBayesianClassifier`

## How to install

To install the `dbc` package, run the following command in your terminal:

```sh
pip install dbc-menamot
```

Make sure you have activated the correct Python environment to avoid potential dependency conflicts.

## Example

A notebook file is provided to give a example in examples folder. And below is an simple example of how to use 
`KmeansDiscreteBayesianClassifier`:

```python
from dbc.main import KmeansDiscreteBayesianClassifier
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

# Reference

- [1] C. Gilet, “Classifieur Minimax Discret pour l’aide au Diagnostic Médical dans la Médecine Personnalisée,”
  Université Côte d’Azur, 2021.
- [2] C. Gilet, S. Barbosa, and L. Fillatre, “Discrete Box-Constrained Minimax Classifier for Uncertain and Imbalanced
  Class Proportions,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 2923–2937, Jun. 2022, doi:
  10.1109/TPAMI.2020.3046439.
- [3] Chen, Wenlong, et al. "Robust Discrete Bayesian Classifier Under Covariate and Label Noise." International
  Conference on Scalable Uncertainty Management. Cham: Springer Nature Switzerland, 2024.

# Contribution

Contributions to this project are welcome. Please submit feature requests and bug reports. If you would like to
contribute code, please submit a pull request.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.
