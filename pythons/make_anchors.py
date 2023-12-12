import math
import os
import numpy as np

from modules.base_path import path_solutions, path_anchors
from modules.base_paras import num_step

if __name__ == '__main__':
    solutions = os.listdir(path_solutions)
    solutions.sort(key=lambda x: int(x.split(".")[0]))

    anchors = np.zeros([len(solutions), (2 * num_step + 1) * 3])
    for index, solution in enumerate(solutions):
        sol = np.loadtxt(os.path.join(path_solutions, solution))
        for step in range(num_step):
            sol_step = sol[sol[:, 0] == (step+1), :]
            if (step + 1) == num_step:
                delta_theta = abs(sol_step[:, 4] - math.pi/2 + 0.1)
                c = np.argmin(delta_theta)
            else:
                c = sol_step.shape[0] - 1
            a = 0
            delta_ = abs(sol_step[:, 2] - (sol_step[0, 2] + sol_step[-1, 2]) / 2)
            b = np.argmin(delta_)
            anchors[index, (step*6):(step*6+3*3)] = sol_step[[a, b, c], 2:5].reshape(1, -1)

        schedule = "▋" * math.floor((index + 1) / len(solutions) * 10)
        print(f"\rDatasets: {schedule}, {(index + 1)}/{len(solutions)}", end='')
    np.savetxt(path_anchors, anchors)


