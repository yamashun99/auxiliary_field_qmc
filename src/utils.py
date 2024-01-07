import numpy as np


def qr_dr_decomposition(A):
    # QR分解
    Q, R = np.linalg.qr(A)

    # D行列（Rの対角成分から構成）
    D = np.diag(np.diag(R))

    # R'行列（RをDで割る）
    R_prime = np.diag(1 / np.diag(D)) @ R

    return Q, D, R_prime


def r2k(OiOj, kx, ky):
    OOk = 0
    for i in range(OiOj.shape[0]):
        for j in range(OiOj.shape[1]):
            OOk += OiOj[i, j] * np.exp(1.0j * (kx * i + ky * j))
    return OOk
