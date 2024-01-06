from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def compute(rank):
    # ここで何らかの計算を行う
    # 例：ランクの2乗を返す
    i = 0
    for _ in range(100_0):
        # a = np.random.rand(100, 100)
        a = np.array([[i + j for i in range(100)] for j in range(100)])
        # a = np.ones((100, 100))
        # 固有値を計算
        w = np.linalg.eig(a)
    return np.array(rank**2, dtype=np.float64)


# 各プロセスで関数を実行
local_result = compute(rank)

# プロセス0で結果を集約
if rank == 0:
    total_result = np.zeros(1, dtype=np.float64)
else:
    total_result = None

# 全てのプロセスから結果を集約する
# 例：合計値を計算
comm.Reduce(local_result, total_result, op=MPI.SUM, root=0)

# プロセス0で集約された結果を出力
if rank == 0:
    print(f"Total sum of squares: {total_result[0]}")
