import os
import shutil

from api.base.paths import path_base, path_dataset
from api.base.paths import path_solutions
from api.base.paths import path_figs, path_figs_init, path_figs_train, path_figs_test, path_figs_once, path_figs_failed, path_figs_all

print(f"警告！警告！即将删除“{path_base}”！！！")
print(f"请进行确认！！！")
flag = input("请进行确认（True/确认删除，False/不删除）：")

if flag:
    print("正在删除...")
    shutil.rmtree(path_base)

    os.mkdir(path_base)

    os.mkdir(path_dataset)
    os.mkdir(path_solutions)

    os.mkdir(path_figs)
    os.mkdir(path_figs_all)
    os.mkdir(path_figs_init)
    os.mkdir(path_figs_train)
    os.mkdir(path_figs_test)
    os.mkdir(path_figs_once)
    os.mkdir(path_figs_failed)
else:
    print("已停止删除程序...")
