import numpy as np
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def compute_pz_given_y(membership, y):
    n_classes = len(set(y))
    n_clusters = membership.shape[0]
    pz_given_y = np.zeros((n_classes, n_clusters))
    for k in range(n_classes):
        indices_of_class_k = np.where(y == k)[0]
        nk = indices_of_class_k.size
        for t in range(n_clusters):
            pz_given_y[k, t] = np.sum(membership[t, indices_of_class_k]) / nk
    return pz_given_y


def compute_pz(p_hat, prior_train):
    # P(Z)=\sum_k P(Z|Y) * P(Y)
    pz = np.sum(prior_train.reshape(-1, 1) * p_hat, axis=0)
    if np.min(pz) < 1e-12:
        # 如果警告出现，代表有一个profile完全处于分布外，请检查
        warnings.warn(
            "Some components of P(Z) are extremely small (< 1e-12). "
            "This may cause numerical instability in later computations.",
            RuntimeWarning,
        )
    return pz


def compute_prior(y: np.ndarray):
    n_classes = len(set(y))
    prior = np.zeros(n_classes)
    n_sample = len(y)

    for k in range(n_classes):
        prior[k] = np.sum(y == k) / n_sample
    return prior


def compute_class_conditional_risk_with_labels(
    y_true: np.ndarray, y_pred: np.ndarray, loss_function: np.ndarray = None
):
    if loss_function is None:
        n_classes = len(set(y_true))
        loss_function = np.ones((n_classes, n_classes)) - np.eye(n_classes)

    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    confusion_matrix_normalized = confusion_matrix(
        y_true_encoded, y_pred_encoded, normalize="true"
    )

    conditional_risk = np.sum(
        np.multiply(loss_function, confusion_matrix_normalized), axis=1
    )

    return conditional_risk, confusion_matrix_normalized


def compute_posterior(
    membership_pred,
    p_hat,
    prior_train,
    prior_pred,
):
    pz = compute_pz(p_hat, prior_train)
    diag_matrix = np.diag(1.0 / pz)
    prob = (prior_pred.reshape(-1, 1) * (p_hat @ (diag_matrix @ membership_pred))).T
    return prob


# def compute_sample_conditional_risk(membership_degree, p_hat, prior, loss_function):
#     # 牢记这里并没有考虑Pz作为归一化项，添加后结果会不同
#     return membership_degree.T @ ((prior.reshape(-1, 1) * loss_function).T @ p_hat).T
#
#
# def compute_final_prob(membership, p_hat, prior, loss_function):
#     sample_conditional_risk = compute_sample_conditional_risk(
#         membership, p_hat, prior, loss_function
#     )
#     a = np.sum(sample_conditional_risk, axis=1)[:, np.newaxis] - sample_conditional_risk
#     prob = np.divide(a, np.sum(a, axis=1)[:, np.newaxis])
#     return prob


def compute_class_conditional_risk(
    y, loss_function, p_hat, membership, prior_train, prior_pred
):
    """
    Compute class-conditional risk for SPDBC (vectorized).

    Parameters
    ----------
    y : array of shape (n_samples,)
        True class labels.
    loss_function : array of shape (n_classes, n_classes)
        Loss matrix L[y_true, y_pred].
    p_hat : array of shape (n_classes, n_clusters)
        P(Z|Y).
    membership : array of shape (n_clusters, n_samples)
        Memberships P(Z|X_i).
    prior_train : array of shape (n_classes,)
        Training class priors P(Y) used to estimate P(Z).
    prior_pred : array of shape (n_classes,)
        Prediction priors used for risk weighting.

    Returns
    -------
    class_conditional_risk : array of shape (n_classes,)
        Average risk per class.
    """
    # ----- shapes -----
    # n_classes := n_classes, T := n_clusters, S := n_samples
    # loss_function: (n_classes, n_classes)
    # p_hat:         (n_classes, n_clusters)
    # Pz_given_x:    (n_clusters, n_samples)
    # prior_train:   (n_classes,)
    # prior_pred:    (n_classes,)
    n_classes = loss_function.shape[0]

    # Class counts for averaging per class
    n_class_k = np.bincount(y, minlength=n_classes)

    # P(Z) using provided helper
    pz = compute_pz(p_hat, prior_train)  # (T,)

    # Build M = L[:, :, None] * prior_pred[:, None, None] * (p_hat[:, None, :] / pz)
    # Shape: (n_classes, n_classes, n_clusters)
    M = (
        loss_function[:, :, None]
        * prior_pred[:, None, None]
        * (p_hat[:, None, :] / pz[None, None, :])
    )

    # We need, for each sample with membership v (T,),  lambd = sum_{i,k} M[i, :, k] * v[k]
    # This equals (sum over i of M)[ :, :] @ v
    # Pre-sum over true-class axis i -> M_sum has shape (C_pred, T)
    M_sum = M.sum(axis=0)  # (n_classes, T)

    # For all samples in one shot:
    # Pz_given_x is (T, S).  Lambda_all = (M_sum @ Pz_given_x)^\top -> (S, n_classes)
    Lambda_all = (M_sum @ membership).T  # (S, n_classes)

    # For each sample, choose predicted class that minimizes lambda
    l_min_idx = np.argmin(Lambda_all, axis=1)  # (S,)

    # Per-sample incurred loss is L[y_true, y_pred*] where y_pred* = argmin_j lambda_j
    sample_losses = loss_function[y, l_min_idx]  # (S,)

    # Average per true class: sum of losses within class / number of samples in class
    # Use bincount with weights to aggregate per-class sums
    sum_losses_per_class = np.bincount(y, weights=sample_losses, minlength=n_classes)

    # Avoid divide-by-zero for classes absent in y
    with np.errstate(invalid="ignore", divide="ignore"):
        class_conditional_risk = np.where(
            n_class_k > 0,
            sum_losses_per_class / n_class_k,
            0.0,
        )

    return class_conditional_risk


def compute_class_conditional_risk_old_version(
    y, loss_function, p_hat, Pz_given_x, prior_train, prior_pred
):
    """
    Compute class-conditional risk for SPDBC.

    Parameters
    ----------
    y : array of shape (n_samples,)
        True class labels.
    loss_function : array of shape (n_classes, n_classes)
        Loss matrix L[y_true, y_pred].
    p_hat : array of shape (n_classes, n_clusters)
        P(Z|Y).
    Pz_given_x : array of shape (n_clusters, n_samples)
        Memberships P(Z|X_i).
    prior_train : array of shape (n_classes,)
        Training class priors P(Y) used to estimate P(Z).
    prior_pred : array of shape (n_classes,)
        Prediction priors used for risk weighting.

    Returns
    -------
    class_conditional_risk : array of shape (n_classes,)
        Average risk per class.
    """
    n_classes = loss_function.shape[0]
    n_class_k = np.bincount(y)
    class_conditional_risk = np.zeros(n_classes)
    # P(Z)
    pz = compute_pz(p_hat, prior_train)
    # 加上 1/P(Z)
    M = (
        loss_function[:, :, None]
        * prior_pred[:, None, None]
        * (p_hat[:, None, :] / (pz[None, None, :] + 1e-12))
    )
    for class_index in range(n_classes):
        indices = np.where(y == class_index)[0]
        for i in indices:
            sample_membership = Pz_given_x.T[i]  # (T,)
            lambd = np.einsum("ijk,k->j", M, sample_membership)
            l_min = np.argmin(lambd)
            class_conditional_risk[class_index] += (
                loss_function[class_index, l_min] / n_class_k[class_index]
            )
    return class_conditional_risk


def proj_simplex_Condat(n_classes, prob):
    """
    This function is inspired from the article: L.Condat, "Fast projection onto the simplex and the
    ball", Mathematical Programming, vol.158, no.1, pp. 575-585, 2016.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    prob : Array of floats
        Vector to project onto the simplex.

    Returns
    -------
    piProj : List of floats
        Priors projected onto the simplex.

    """

    linK = np.linspace(1, n_classes, n_classes)
    piProj = np.maximum(
        prob - np.max(((np.cumsum(np.sort(prob)[::-1]) - 1) / (linK[:]))), 0
    )
    piProj = piProj / np.sum(piProj)
    return piProj


def compute_prior_best(
    y,
    loss_function,
    p_hat,
    membership_degree,
    prior_train,
    alpha=1,
    beta=0.5,
    n_iter=300,
    eps=1e-2,
    return_history=False,
):

    prior_best = prior_train.copy()
    n_classes = loss_function.shape[0]
    if n_classes == 2:
        return compute_prior_best_binary(
            y=y,
            loss_function=loss_function,
            p_hat=p_hat,
            membership=membership_degree,
            prior_train=prior_train,
            eps=eps,
            return_history=return_history,
        )
    history = []
    v = np.zeros(n_classes)  # 初始化动量项

    for n in range(n_iter):
        # 计算当前 class-conditional 风险和 global 风险
        class_conditional_risk = compute_class_conditional_risk(
            y=y,
            loss_function=loss_function,
            p_hat=p_hat,
            membership=membership_degree,
            prior_train=prior_train,
            prior_pred=prior_best,
        )

        # global_risk = np.dot(pi, class_conditional_risk)
        global_risk = np.mean(class_conditional_risk)

        G = class_conditional_risk - global_risk

        max_gap = np.max(class_conditional_risk) - np.min(class_conditional_risk)
        history.append(max_gap)
        # 判断收敛
        if max_gap < eps:
            break

        # 更新动量项
        v = beta * v + (1 - beta) * G

        # 步长和归一化因子
        gamma = alpha / (n + 1)
        eta = max(1.0, np.sum(v**2))  # 用动量向量v代替G

        # 使用带动量的方向更新 pi
        w = prior_best + (gamma / eta) * v
        prior_best = proj_simplex_Condat(n_classes, w)

    if return_history:
        return prior_best, history
    else:
        return prior_best


def compute_prior_best_binary(
    y,
    loss_function,
    p_hat,
    membership,
    prior_train,
    n_iter=100,
    eps=1e-3,
    return_history=False,
    prior_update_eps=1e-5,
):
    def compute_risk_gap(prior_pred):
        class_conditional_risk = compute_class_conditional_risk(
            y=y,
            loss_function=loss_function,
            p_hat=p_hat,
            membership=membership,
            prior_train=prior_train,
            prior_pred=prior_pred,
        )
        return class_conditional_risk[0] - class_conditional_risk[1]

    # 两端点
    prior_lower = np.array([0.0, 1.0])  # pi0=0
    prior_upper = np.array([1.0, 0.0])  # pi0=1
    gap_lower = compute_risk_gap(prior_lower)
    gap_upper = compute_risk_gap(prior_upper)

    history = []

    if abs(gap_lower) <= abs(gap_upper):
        prior_best, gap_best = prior_lower, gap_lower
    else:
        prior_best, gap_best = prior_upper, gap_upper

    # 常规二分
    for it in range(n_iter):
        pi_mid = 0.5 * (prior_lower + prior_upper)
        gap_mid = compute_risk_gap(pi_mid)

        if return_history:
            history.append(gap_mid)

        # 更新最优近似
        if abs(gap_mid) < abs(gap_best):
            prior_best, gap_best = pi_mid.copy(), gap_mid

        # 命中根
        if abs(gap_mid) <= eps:
            if return_history:
                return pi_mid, history
            return pi_mid

        # 更新区间
        if gap_lower * gap_mid > 0:
            prior_lower, gap_lower = pi_mid, gap_mid
        else:
            prior_upper, gap_upper = pi_mid, gap_mid

        # 区间收敛
        if (prior_upper[0] - prior_lower[0]) <= prior_update_eps:
            break

    if return_history:
        return prior_best, history
    else:
        return prior_best
