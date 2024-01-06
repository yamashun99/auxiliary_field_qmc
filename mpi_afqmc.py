import numpy as np
import os
from mpi4py import MPI
from afqmc import *


# MPIのコミュニケータ、ランク、サイズを取得
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_Szz_beta(p, n_stab, M):
    Szz_beta = []
    for beta in range(6):
        Szz = np.zeros((p["N"], p["N"]))
        p["beta"] = beta
        afqmc = AFQMC(**p)
        for m in range(M):
            print(beta, m)
            for l in reversed(range(p["L"])):
                if l % n_stab == 0:
                    afqmc.stabilize(l)
                afqmc.i_sweep(l)
                afqmc.time_update_green_function(l)
            if m > 3:
                Szz += np.array(
                    [
                        [afqmc.make_Szz_2d(i, j) for i in range(p["N"])]
                        for j in range(p["N"])
                    ]
                )
        Szz /= M - 4
        Szz_beta.append(Szz)
    return np.array(Szz_beta)


# 使用例
U = 4
mu = U / 2
L = 20
N = 10
dimension = 2
size = N**dimension
# s = -np.ones((L, size))
# sはL×Nの行列で、要素は1か-1
s = np.random.choice([-1, 1], size=(L, size))
p = {
    "N": N,
    "L": L,
    "beta": 3,
    "t": 1.0,
    "U": U,
    "mu": mu,
    "s": s,
    "dimension": dimension,
    "rank": rank,
}

n_stab = 1
M = 100


local_result = get_Szz_beta(p, n_stab, M)

# プロセス0で結果を集約
if rank == 0:
    total_result = np.zeros_like(local_result)
else:
    total_result = None

# MPI集約操作（例：MPI SUM）
comm.Reduce(local_result, total_result, op=MPI.SUM, root=0)

# プロセス0で結果を出力
if rank == 0:
    # total_resultをサイズで割って平均を求めるなど、最終的な結果を計算
    final_result = total_result / size
    print("Final result:", final_result)
    os.makedirs("data", exist_ok=True)
    np.save("data/Szz.npy", final_result)

filename = f"data/Szz_{rank}.npy"
np.save(filename, local_result)
