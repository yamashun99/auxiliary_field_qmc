import sys

sys.path.append("..")
import numpy as np
import os
from mpi4py import MPI
from src.afqmc import *
import h5py


# MPIのコミュニケータ、ランク、サイズを取得
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_SzSz(p, n_stab, n_mc, n_thermal):
    SzSz = np.zeros((p["N"], p["N"]))
    afqmc = AFQMC(**p)
    for i_mc in range(n_mc):
        for l in reversed(range(p["L"])):
            if l % n_stab == 0:
                afqmc.stabilize(l)
            afqmc.i_sweep(l)
            afqmc.time_update_green_function(l)
        if i_mc > n_thermal:
            SzSz += np.array(
                [
                    [afqmc.make_SzSz_2d(i, j) for i in range(p["N"])]
                    for j in range(p["N"])
                ]
            )
    SzSz /= n_mc - n_thermal - 1
    return SzSz


p = {}
p["N"] = 10
p["L"] = 20
p["beta"] = None
p["t"] = 1.0
p["U"] = 4
p["mu"] = p["U"] / 2
p["dimension"] = 2
p["random_seed"] = 123 * rank + 567

for beta in range(6):
    if rank == 0:
        print(f"beta={beta}")
    p["beta"] = beta
    SzSz = get_SzSz(p, n_stab=1, n_mc=100, n_thermal=3)

    # プロセス0で結果を集約
    if rank == 0:
        total_result = np.zeros_like(SzSz)
    else:
        total_result = None

    # MPI集約操作（例：MPI SUM）
    comm.Reduce(SzSz, total_result, op=MPI.SUM, root=0)

    # プロセス0で結果を出力
    if rank == 0:
        final_result = total_result / size
        print("Final result:", final_result)
        os.makedirs("data", exist_ok=True)
        with h5py.File("./data/results.h5", "a") as f:
            # グループが存在するか確認し、存在しなければ作成
            group_name = f"beta={beta}"
            if group_name not in f:
                group_beta = f.create_group(group_name)
            else:
                group_beta = f[group_name]
            if "SzSz" in group_beta:
                del group_beta["SzSz"]
            group_beta.create_dataset("SzSz", data=final_result)

    os.makedirs("data", exist_ok=True)
    with h5py.File(f"./data/results_rank{rank}.h5", "a") as f:
        # グループが存在するか確認し、存在しなければ作成
        group_name = f"beta={beta}"
        if group_name not in f:
            group_beta = f.create_group(group_name)
        else:
            group_beta = f[group_name]
        if "SzSz" in group_beta:
            del group_beta["SzSz"]
        group_beta.create_dataset("SzSz", data=SzSz)
