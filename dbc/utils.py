import numpy as np
from itertools import combinations

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def compute_p_hat(profile_labels: np.ndarray, y: np.ndarray, n_classes: int, n_clusters: int):
    """
    Compute the probability estimates for each class and cluster. The function calculates the relative frequency of
    each cluster label within each class and returns these probabilities as a matrix.

    :param profile_labels: Array containing cluster labels for each sample.
    :type profile_labels: np.ndarray
    :param y: Array containing class labels for each sample.
    :type y: np.ndarray
    :param n_classes: Number of distinct classes.
    :type n_classes: int
    :param n_clusters: Number of distinct clusters.
    :type n_clusters: int
    :return: A matrix where each row represents a class and each column represents a cluster. Each entry contains the
             probability estimate of the cluster for the respective class.
    :rtype: np.ndarray
    """
    p_hat = np.zeros((n_classes, n_clusters))

    for k in range(n_classes):
        indices_of_class_k = np.where(y == k)[0]
        nk = len(indices_of_class_k)
        p_hat[k] = np.bincount(profile_labels[indices_of_class_k], minlength=n_clusters) / nk
        # Count number of occurrences of each value in array of non-negative ints.
    return p_hat


def compute_prior(y: np.ndarray, n_classes: int):
    """
    Compute the prior probabilities for each class.

    This function calculates the prior probabilities for each class k in the
    range [0, n_classes-1]. It returns a numpy array where each element
    represents the proportion of occurrences of the class in the input array y.

    :param y: A numpy array of class labels.
    :type y: np.ndarray
    :param n_classes: Total number of unique classes.
    :type n_classes: int
    :return: A numpy array of prior probabilities.
    :rtype: np.ndarray
    """
    pi = np.zeros(n_classes)
    total_count = len(y)

    for k in range(n_classes):
        pi[k] = np.sum(y == k) / total_count
    return pi


def predict_profile_label(prior, p_hat, loss_function):
    """
    Predict the profile label based on prior probabilities, the predicted
    probabilities, and a loss function.

    This function calculates the class risk using the provided prior
    probabilities, predicted probabilities, and the loss function, and then
    returns the label with the minimum risk.

    :param prior: Array of prior probabilities for each class.
    :type prior: np.ndarray
    :param p_hat: Matrix of predicted probabilities for each class and instance.
    :type p_hat: np.ndarray
    :param loss_function: Matrix of loss values for each class combination.
    :type loss_function: np.ndarray
    :return: Array of predicted labels for each instance.
    :rtype: np.ndarray
    """
    class_risk = (prior.reshape(-1, 1) * loss_function).T @ p_hat
    l_predict = np.argmin(class_risk, axis=0)
    return l_predict


def compute_conditional_risk(y_true: np.ndarray, y_pred: np.ndarray, loss_function: np.ndarray = None):
    """
    Computes the conditional risk and the normalized confusion matrix for given true labels,
    predicted labels, and a loss function.

    This function uses label encoding to transform string labels into integer codes. It then
    calculates the confusion matrix and normalizes it. Finally, it computes the conditional
    risk by multiplying the loss function with the normalized confusion matrix and summing the
    resulting products.

    :param y_true:
        The true labels of the data as a NumPy array.
    :param y_pred:
        The predicted labels as a NumPy array.
    :param loss_function:
        A loss function represented as a NumPy array.
    :return:
        A tuple containing the conditional risk and the normalized confusion matrix.
    """
    if loss_function is None:
        n_classes = len(set(y_true))
        loss_function = np.ones((n_classes, n_classes)) - np.eye(n_classes)

    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)


    confusion_matrix_normalized = confusion_matrix(y_true_encoded, y_pred_encoded, normalize='true')


    conditional_risk = np.sum(np.multiply(loss_function, confusion_matrix_normalized), axis=1)

    return conditional_risk, confusion_matrix_normalized

def compute_p_hat_soft(degree, y_soft, n_classes):
    n_clusters = degree.shape[0]
    p_hat = np.zeros((n_classes, n_clusters))

    for k in range(n_classes):
        # 使用每个样本属于类别k的概率
        class_probs = y_soft[:, k]  # 获取第k类的概率

        for t in range(n_clusters):
            # 计算加权和
            weighted_sum = np.sum(degree[t, :] * class_probs)
            # 计算概率和
            prob_sum = np.sum(class_probs)

            if prob_sum > 0:
                p_hat[k, t] = weighted_sum / prob_sum

    return p_hat

def compute_p_hat_with_soft_labels(degree, y):
    """
    degree: shape (n_clusters, n_samples)
    y: shape (n_samples, n_classes), each row is a soft label or one-hot
    returns: p_hat of shape (n_classes, n_clusters)
    """
    y_T = y.T  # shape (n_classes, n_samples)
    sum_y = np.sum(y_T, axis=1, keepdims=True)  # shape (n_classes, 1)
    # Avoid division by zero
    sum_y[sum_y == 0] = 1
    p_hat = (y_T @ degree.T) / sum_y  # shape (n_classes, n_clusters)
    return p_hat

def compute_p_hat_with_degree(degree, y, n_classes):
    n_clusters = degree.shape[0]
    p_hat = np.zeros((n_classes, n_clusters))
    for k in range(n_classes):
        indices_of_class_k = np.where(y == k)[0]
        mk = indices_of_class_k.size
        for t in range(n_clusters):
            if mk > 0:
                p_hat[k, t] = np.sum(degree[t, indices_of_class_k]) / mk
    return p_hat

# def compute_p_hat_with_degree(degree, y, n_classes):
#     n_clusters = degree.shape[0]
#     n_samples = degree.shape[1]
#     p_hat = np.zeros((n_classes, n_clusters))
#
#     # degree_sum_over_t: 每个样本 i 的 degree 总和 (shape: n_samples,)
#     degree_sum_over_t = np.sum(degree, axis=0)
#
#     for k in range(n_classes):
#         indices_of_class_k = np.where(y == k)[0]
#         mk = indices_of_class_k.size
#         for t in range(n_clusters):
#             if mk > 0:
#                 # 每个 i 的 degree 权重 * degree 本身
#                 weighted_degree = (degree[t, indices_of_class_k] / degree_sum_over_t[indices_of_class_k]) * degree[t, indices_of_class_k]
#                 p_hat[k, t] = np.sum(weighted_degree) / mk
#     return p_hat

def compute_prob(membership_degree,membership_degree_pred, p_hat, prior,option=1):
    # P(Z)=\sum_i P(Z|X_i) * P(X_i)
    # pz = np.sum(membership_degree.T, axis=0)/membership_degree.shape[1]

    # P(Z)=\sum_k P(Z|Y) * P(Y)
    pz = np.sum(prior.reshape(-1,1) * p_hat, axis=0)
    diag_matrix = np.diag(1.0 / pz)
    return prior * (membership_degree_pred.T @ diag_matrix @ p_hat.T)

# def compute_derivative_part(membership_degree,membership_degree_pred, p_hat, loss_function):
#     pz = np.sum(membership_degree.T, axis=0) / membership_degree.shape[1]
#     diag_matrix = np.diag(1.0 / pz)
#     return  membership_degree_pred.T @ diag_matrix @ p_hat.T @ loss_function
#
# def compute_risk(membership_degree,membership_degree_pred, p_hat, prior, loss_function):
#     pz = np.sum(membership_degree.T, axis=0) / membership_degree.shape[1]
#     diag_matrix = np.diag(1.0 / pz)
#     return  prior * (membership_degree_pred.T @ diag_matrix @ p_hat.T) @ loss_function

# def compute_derivative(membership_degree,membership_degree_pred, p_hat, prior, loss_function):
#     a = compute_risk(membership_degree,membership_degree_pred, p_hat, prior, loss_function)  # (N, C)
#     b = compute_derivative_part(membership_degree,membership_degree_pred, p_hat, loss_function)  # (N, C)
#
#     avg_risk = np.mean(a, axis=1, keepdims=True)  # (N, 1)
#     avg_derivate = np.mean(b, axis=1, keepdims=True)  # (N, 1)
#
#     gradient_per_sample = 2 * (a - avg_risk) * (b - avg_derivate)  # (N, C)
#
#     # 现在，我们需要将其映射回 K 维空间
#     final_gradient = membership_degree @ gradient_per_sample  # (K, C) @ (N, C) = (K,)
#
#     return final_gradient

def compute_b_risk(membership_degree, p_hat, prior, loss_function):
    return membership_degree.T @ ((prior.reshape(-1, 1) * loss_function).T @ p_hat).T
def compute_posterior(membership_degree, p_hat, prior, loss_function):
    """
    Parameters
    ----------
    membership_degree : Array

    p_hat : Array of floats
        Probability estimate of observing the features profile.
    prior : Array of floats
        Real class proportions.
    loss_function : Array
        Loss function.

    Returns
    -------
    Yhat : Vector
        Predicted labels.
    """

    class_risk = membership_degree.T @ ((prior.reshape(-1, 1) * loss_function).T @ p_hat).T + 1e-10
    a = np.sum(class_risk, axis=1)[:, np.newaxis] - class_risk
    prob = np.divide(a, np.sum(a, axis=1)[:, np.newaxis])
    return prob


def discretize_features(samples, decision_tree_model):
    '''
    Parameters
    ----------
    samples : DataFrame
    Features to be discretized.
    decision_tree_model : Decision Tree Classifier Model
    Model used for discretization.

    Returns
    -------
    discretized_features : Vector
        Discretized features.
    '''
    applied_samples = DecisionTreeClassifier.apply(decision_tree_model, samples, check_input=True)

    def create_value_index_map(applied_samples):
        unique_values, inverse_indices = np.unique(applied_samples, return_inverse=True)
        value_to_index_map = {value: idx for idx, value in enumerate(unique_values)}
        return np.array([value_to_index_map[value] for value in applied_samples])

    discretized_features = create_value_index_map(applied_samples)

    return discretized_features


def compute_global_risk(conditional_risk, prior):
    """
    Parameters
    ----------
    conditional_risk : ndarray of shape (K,)
        Conditional risk
    prior : ndarray of shape (K,)
        Proportion of classes

    Returns
    -------
    global_risk : float
        Global risk.
    """

    global_risk = np.sum(conditional_risk * prior)

    return global_risk


def num2cell(a):
    if type(a) is np.ndarray:
        return [num2cell(x) for x in a]
    else:
        return a



def proj_onto_polyhedral_set(pi, Box, K):
    '''
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex.
    Box : Array
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    piStar : Array of floats
            Priors projected onto the box-constrained simplex.

    '''

    # Verification of constraints
    for i in range(K):
        for j in range(2):
            if Box[i, j] < 0:
                Box[i, j] = 0
            if Box[i, j] > 1:
                Box[i, j] = 1

    # Generate matrix G:
    U = np.concatenate((np.eye(K), -np.eye(K), np.ones((1, K)), -np.ones((1, K))))
    eta = Box[:, 1].tolist() + (-Box[:, 0]).tolist() + [1] + [-1]

    n = U.shape[0]

    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = np.vdot(U[i, :], U[j, :])

    # Generate subsets of {1,...,n}:
    M = (2 ** n) - 1
    I = num2cell(np.zeros((1, M)))

    i = 0
    for l in range(n):
        T = list(combinations(list(range(n)), l + 1))
        for p in range(i, i + len(T)):
            I[0][p] = T[p - i]
        i = i + len(T)

    # Algorithm

    for m in range(M):
        Im = I[0][m]

        Gmm = np.zeros((len(Im), len(Im)))
        ligne = 0
        for i in Im:
            colonne = 0
            for j in Im:
                Gmm[ligne, colonne] = G[i, j]
                colonne += 1
            ligne += 1

        if np.linalg.det(Gmm) != 0:

            nu = np.zeros((2 * K + 2, 1))
            w = np.zeros((len(Im), 1))
            for i in range(len(Im)):
                w[i] = np.vdot(pi, U[Im[i], :]) - eta[Im[i]]

            S = np.linalg.solve(Gmm, w)

            for e in range(len(S)):
                nu[Im[e]] = S[e]

            if np.any(nu < -10 ** (-10)) == False:
                A = G.dot(nu)
                z = np.zeros((1, 2 * K + 2))
                for j in range(2 * K + 2):
                    z[0][j] = np.vdot(pi, U[j, :]) - eta[j] - A[j]

                if np.all(z <= 10 ** (-10)) == True:
                    pi_new = pi
                    for i in range(2 * K + 2):
                        pi_new = pi_new - nu[i] * U[i, :]

    piStar = pi_new

    # Remove noisy small calculus errors:
    piStar = piStar / piStar.sum()

    return piStar



def proj_simplex_Condat(K, pi):
    """
    This function is inspired from the article: L.Condat, "Fast projection onto the simplex and the
    ball", Mathematical Programming, vol.158, no.1, pp. 575-585, 2016.
    Parameters
    ----------
    K : int
        Number of classes.
    pi : Array of floats
        Vector to project onto the simplex.

    Returns
    -------
    piProj : List of floats
        Priors projected onto the simplex.

    """

    linK = np.linspace(1, K, K)
    piProj = np.maximum(pi - np.max(((np.cumsum(np.sort(pi)[::-1]) - 1) / (linK[:]))), 0)
    piProj = piProj / np.sum(piProj)
    return piProj


def proj_onto_U(pi, Box, K):
    '''
    Parameters
    ----------
    pi : Array of floats
        Vector to project onto the box-constrained simplex..
    Box : Matrix
        {'none', matrix} : Box-constraint on the priors.
    K : int
        Number of classes.

    Returns
    -------
    pi_new : Array of floats
            Priors projected onto the box-constrained simplex.

    '''

    check_U = 0
    if pi.sum() == 1:
        for k in range(K):
            if (pi[0][k] >= Box[k, 0]) & (pi[0][k] <= Box[k, 1]):
                check_U = check_U + 1

    if check_U == K:
        pi_new = pi

    if check_U < K:
        pi_new = proj_onto_polyhedral_set(pi, Box, K)

    return pi_new



def compute_piStar(pHat, y_train, K, L, N, Box):
    """
    Parameters
    ----------
    pHat : Array of floats
        Probability estimate of observing the features profile in each class.
    y_train : Dataframe
        Real labels of the training set.
    K : int
        Number of classes.
    L : Array
        Loss Function.
    N : int
        Number of iterations in the projected subgradient algorithm.
    Box : Array
        {'none', matrix} : Box-constraints on the priors.

    Returns
    -------
    piStar : Array of floats
        Least favorable priors.
    rStar : float
        Global risks.
    RStar : Array of float
        Conditional risks.
    V_iter : Array
        Values of the V function at each iteration.
    stockpi : Array
        Values of pi at each iteration.

    """
    # IF BOX-CONSTRAINT == NONE (PROJECTION ONTO THE SIMPLEX)
    if Box is None:
        pi = compute_prior(y_train, K).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0, :] = mu_k
            stockpi[:, n - 1] = pi[0, :]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_simplex_Condat(K, w)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0, :] = mu_k
        stockpi[:, n - 1] = pi[0, :]

        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

    # IF BOX-CONSTRAINT
    if Box is not None:
        pi = compute_prior(y_train, K).reshape(1, -1)
        rStar = 0
        piStar = pi
        RStar = 0

        V_iter = []
        stockpi = np.zeros((K, N))

        for n in range(1, N + 1):
            # Compute subgradient R at point pi (see equation (21) in the paper)
            lambd = np.dot(L, pi.T * pHat)
            R = np.zeros((1, K))

            mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
            R[0, :] = mu_k
            stockpi[:, n - 1] = pi[0, :]

            r = compute_global_risk(R, pi)
            V_iter.append(r)
            if r > rStar:
                rStar = r
                piStar = pi
                RStar = R
                # Update pi for iteration n+1
            gamma = 1 / n
            eta = np.maximum(float(1), np.linalg.norm(R))
            w = pi + (gamma / eta) * R
            pi = proj_onto_U(w, Box, K)

        # Check if pi_N == piStar
        lambd = np.dot(L, pi.T * pHat)
        R = np.zeros((1, K))

        mu_k = np.sum(L[:, np.argmin(lambd, axis=0)] * pHat, axis=1)
        R[0, :] = mu_k
        stockpi[:, n - 1] = pi[0, :]

        r = compute_global_risk(R, pi)
        if r > rStar:
            rStar = r
            piStar = pi
            RStar = R

    return piStar, rStar, RStar, V_iter, stockpi

def compute_SPDBC_class_conditional_risk(X, y, class_index, loss_function, p_hat, membership_degree, pi):
    nb_class_k = np.unique(y, return_counts=True)[1]
    # K = loss_function.shape[0]
    # T = p_hat.shape[1]
    r = 0

    # for i in range(X.shape[0]):
    #     lambd = np.zeros(K)
    #     for l in range(0, K):
    #         for t in range(0, T):
    #             for k in range(0, K):
    #                 lambd[l] += loss_function[k, l] * pi[k] * p_hat[k, t] * membership_degree.T[i, t]
    #     for l in range(0, K):
    #         if lambd[l] == np.min(lambd) and y[i] == class_index:
    #             r += loss_function[class_index, l] / nb_class_k[class_index]

    # 预先计算一次固定部分 (K, K, T) 张量
    M = loss_function[:, :, None] * pi[:, None, None] * p_hat[:, None, :]  # (K, K, T)

    for i in range(X.shape[0]):
        # 提取样本i的membership度 (T,)
        membership = membership_degree.T[i]  # (T,)

        # 计算 weighted sum，先做广播相乘，再在特征T维度求和
        # temp = M * membership[None, None, :]  # (K, K, T)
        # lambd = temp.sum(axis=2).sum(axis=0)  # (K,)
        lambd = np.einsum('ijk,k->j', M, membership)

        # 找到lambd最小值对应的类别l_min
        l_min = np.argmin(lambd)

        # 只对真实标签是class_index的样本进行累加
        if y[i] == class_index:
            r += loss_function[class_index, l_min] / nb_class_k[class_index]

    return r


def compute_SPDBC_pi_star(X, y, loss_function, p_hat, membership_degree, pi,
                          alpha=1, beta=0.9, n_iter=300, eps=1e-3, return_history=False):
    """
    优化SPDBC的先验概率pi，使用带动量的Projected Gradient Descent方法。

    Parameters
    ----------
    X : ndarray
        特征数据
    y : ndarray
        标签数据
    loss_function : ndarray
        K x D的损失矩阵
    p_hat : ndarray
        条件概率估计值
    membership_degree : ndarray
        每个样本对每个profile的隶属度
    pi : ndarray
        初始先验概率
    alpha : float
        学习率调节因子
    beta : float
        动量因子（通常为0.9）
    n_iter : int
        最大迭代次数
    eps : float
        收敛阈值（使用 ∑|G|）
    return_history : bool
        是否返回risk收敛历史（∑|G|）

    Returns
    -------
    pi : ndarray
        优化后的先验概率
    risk_history : list (可选)
        每次迭代的梯度绝对值总和（用于收敛观察）
    """
    K = loss_function.shape[0]
    risk_history = []
    v = np.zeros(K)  # 初始化动量项

    for n in range(1, n_iter + 1):
        # 计算当前 class-conditional 风险和 global 风险
        class_conditional_risk = np.array([
            compute_SPDBC_class_conditional_risk(X, y, k, loss_function, p_hat, membership_degree, pi)
            for k in range(K)
        ])
        global_risk = np.dot(pi, class_conditional_risk)
        G = class_conditional_risk - global_risk

        grad = np.sum(G**2)
        risk_history.append(grad)

        # 判断收敛
        if grad < eps:
            break

        # 更新动量项
        v = beta * v + (1 - beta) * G

        # 步长和归一化因子
        gamma = alpha / n
        eta = max(1.0, np.sum(v**2))  # 用动量向量v代替G

        # 使用带动量的方向更新 pi
        w = pi + (gamma / eta) * v
        pi_new = proj_simplex_Condat(K, w)

        # 更新 pi
        pi = pi_new

    if return_history:
        return pi, risk_history
    else:
        return pi
