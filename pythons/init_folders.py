import os

from modules.base_path import path_base, path_dataset
from modules.base_path import path_pts, path_solutions
from modules.base_path import path_figs, path_figs_init, path_figs_train, path_figs_test, path_figs_once, path_figs_failed, path_figs_all

os.mkdir(path_base)

os.mkdir(path_dataset)
os.mkdir(path_pts)
os.mkdir(path_solutions)

os.mkdir(path_figs)
os.mkdir(path_figs_all)
os.mkdir(path_figs_init)
os.mkdir(path_figs_train)
os.mkdir(path_figs_test)
os.mkdir(path_figs_once)
os.mkdir(path_figs_failed)

