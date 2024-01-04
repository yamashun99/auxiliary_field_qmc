import numpy as np
import scipy.linalg


class AFQMC:
    def __init__(self, N, L, beta, t, U, mu, s, dimension):
        self.N = N
        self.L = L
        self.t = t
        self.U = U
        self.delta_tau = beta / L
        self.mu = mu
        self.s = s
        self.a = np.arccosh(np.exp(self.U * self.delta_tau / 2)) / 2
        self.size = N**dimension
        if dimension == 1:
            self.exp_A = self.make_exp_A()
        elif dimension == 2:
            self.exp_A = self.make_exp_A_2d()
        # グリーン関数の初期化（アップスピンとダウンスピンの両方）
        self.G_up = [None for _ in range(self.L)]
        self.G_dn = [None for _ in range(self.L)]
        self.G_up[self.L - 1] = self.G(L - 1, 1)
        self.G_dn[self.L - 1] = self.G(L - 1, -1)

    def make_X(self, gamma_up, gamma_dn, l, i):
        X_up = 1 + gamma_up * (1 - self.G_up[l][i, i])
        X_dn = 1 + gamma_dn * (1 - self.G_dn[l][i, i])
        return X_up, X_dn

    def time_update_green_function(self, l):
        AB_up = self.exp_A @ self.make_exp_B(1, l)
        AB_dn = self.exp_A @ self.make_exp_B(-1, l)
        self.G_up[l - 1] = AB_up @ self.G_up[l] @ np.linalg.inv(AB_up)
        self.G_dn[l - 1] = AB_dn @ self.G_dn[l] @ np.linalg.inv(AB_dn)

    def stabilize(self, l):
        self.G_up[l] = self.G(l, 1)
        self.G_dn[l] = self.G(l, -1)

    def i_sweep(self, l):
        for i in range(self.size):
            gamma_up = self.gamma(self.s[l][i], 1)
            gamma_dn = self.gamma(self.s[l][i], -1)
            X_up, X_dn = self.make_X(gamma_up, gamma_dn, l, i)
            # フリップ確率を計算する
            # p_flip = X_up * X_dn
            p_flip = X_up * X_dn / (1 + X_up * X_dn)
            # 0と1の間の乱数を生成し、フリップを試みる
            if np.random.rand() < p_flip:
                # 補助場をフリップ
                self.s[l, i] = -self.s[l, i]
                # グリーン関数を更新
                self.update_green_function(l, gamma_up, gamma_dn, X_up, X_dn, i)
                # self.stabilize(l)

    def update_green_function(self, l, gamma_up, gamma_dn, X_up, X_dn, i):
        G_up_i = np.zeros((self.size, self.size))
        G_up_i[:, i] = self.G_up[l][:, i]
        G_up_i[i, i] -= 1
        F_inv_up = np.eye(self.size) + gamma_up / X_up * G_up_i

        G_dn_i = np.zeros((self.size, self.size))
        G_dn_i[:, i] = self.G_dn[l][:, i]
        G_dn_i[i, i] -= 1
        F_inv_dn = np.eye(self.size) + gamma_dn / X_dn * G_dn_i

        self.G_up[l] = F_inv_up @ self.G_up[l]
        self.G_dn[l] = F_inv_dn @ self.G_dn[l]

    def matrix_A(self):
        matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            if i - 1 >= 0:
                matrix[i, i - 1] = 1
            if i + 1 < self.N:
                matrix[i, i + 1] = 1
        matrix[0, self.N - 1] = 1
        matrix[self.N - 1, 0] = 1
        return self.t * self.delta_tau * matrix

    def matrix_A_2d(self):
        matrix = np.zeros((self.size, self.size))
        # 各サイト間のホッピングを設定
        for i in range(self.size):
            x, y = i % self.N, i // self.N  # x, y座標への変換
            # 右隣のサイトへのホッピング
            if x < self.N - 1:
                matrix[i, i + 1] = 1
            else:
                matrix[i, i + 1 - self.N] = 1  # 周期的境界条件
            # 左隣のサイトへのホッピング
            if x > 0:
                matrix[i, i - 1] = 1
            else:
                matrix[i, i - 1 + self.N] = 1  # 周期的境界条件
            # 上のサイトへのホッピング
            if y > 0:
                matrix[i, i - self.N] = 1
            else:
                matrix[i, i - self.N + self.size] = 1  # 周期的境界条件
            # 下のサイトへのホッピング
            if y < self.N - 1:
                matrix[i, i + self.N] = 1
            else:
                matrix[i, i + self.N - self.size] = 1  # 周期的境界条件
        return self.t * self.delta_tau * matrix

    def h(self, spin_sign, l):
        return (
            2 * self.a * spin_sign * self.s[l] - (self.U / 2 - self.mu) * self.delta_tau
        )

    def make_exp_B(self, spin_sign, l):
        return np.diag(np.exp(self.h(spin_sign, l)))

    def make_exp_A(self):
        return scipy.linalg.expm(self.matrix_A())

    def make_exp_A_2d(self):
        return scipy.linalg.expm(self.matrix_A_2d())

    def G(self, l, spin_sign):
        I = np.eye(self.size)
        product = I
        for l_prime in range(l + 1, self.L):
            product = product @ self.exp_A @ self.make_exp_B(spin_sign, l_prime)
        for l_prime in range(l + 1):
            product = product @ self.exp_A @ self.make_exp_B(spin_sign, l_prime)
        return np.linalg.inv(I + product)

    def gamma(self, s_li, spin_sign):
        return np.exp(-4 * self.a * spin_sign * s_li) - 1

    def kronecker_delta(self, i, j):
        return 1 if i == j else 0

    def sigma_sigma_bar(self, l, i, j, sigma):
        if sigma == "up":
            G_sigma = self.G_up
            G_sigma_bar = self.G_dn
        elif sigma == "dn":
            G_sigma = self.G_dn
            G_sigma_bar = self.G_up
        return (1 - G_sigma[l][i, i]) * (1 - G_sigma_bar[l][j, j])

    def sigma_sigma(self, l, i, j, sigma):
        if sigma == "up":
            G_sigma = self.G_up
        elif sigma == "dn":
            G_sigma = self.G_dn
        sum = 0
        sum += (1 - G_sigma[l][i, i]) * (1 - G_sigma[l][j, j])
        sum += (self.kronecker_delta(i, j) - G_sigma[l][j, i]) * G_sigma[l][i, j]
        return sum

    def Szz(self, delta_x):
        sum = 0
        for l in range(self.L):
            for i in range(self.N):
                j = (i + delta_x) % self.N
                sum += (
                    self.sigma_sigma(l, i, j, "up")
                    + self.sigma_sigma(l, i, j, "dn")
                    - self.sigma_sigma_bar(l, i, j, "up")
                    - self.sigma_sigma_bar(l, i, j, "dn")
                )
        return sum / self.L / self.N

    def make_Szz_pipi(self):
        Szz_pipi = 0
        for ix in range(self.N):
            Szz_pipi += (-1) ** ix * self.Szz(ix)
        return Szz_pipi / self.N
