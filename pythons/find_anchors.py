import math
import os
import numpy as np
import time
import random
import psutil

from modules.base_path import path_log, path_result, path_anchors_failed
from modules.base_paras import num_anchor_state, num_samples
from modules.util_sample_tries import SampleTries

if __name__ == '__main__':
    # 程序运行中不显示报错
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    x_list = np.linspace(start=-8.5, stop=-1.5, num=29)
    y_list = np.linspace(start=7.0, stop=12.0, num=21)
    theta_list = np.linspace(start=-math.pi / 3, stop=math.pi / 3, num=13)
    anchors_failed_list = np.zeros([1, num_anchor_state])

    log = open(path_log, 'w')
    log.write(f"START AT {time.asctime(time.localtime())}")
    log.close()

    schedule_num = 0
    schedule_all = x_list.shape[0] * y_list.shape[0] * theta_list.shape[0]
    for index_x, x in enumerate(x_list):
        for index_y, y in enumerate(y_list):
            for index_theta, theta in enumerate(theta_list):
                schedule_num += 1

                # 设置初始化数据
                init_data = []
                for sample in range(num_samples):
                    init_data.append([x + random.uniform(-1, 1) * 0.01, y + random.uniform(-1, 1) * 0.01,
                                      theta + random.uniform(-1, 1) * 0.01])

                # 创教多线程
                sample_tries = SampleTries(init_data)
                sample_tries.main()

                schedule = schedule_num / schedule_all
                process = "=" * math.floor(schedule * 50)
                process_all = " " * math.floor((1 - schedule) * 50)
                time_s = time.time()

                # 等待各线程完成
                while True:
                    sample_tries.get_state()
                    result = open(path_result, 'w')
                    result.write(f"\nNow: {time.asctime(time.localtime())}")
                    result.write(f"\nProcess:[{process}{process_all}] {schedule * 100:.2f}%")
                    result.write(f"\nInit_Data: [{x:.2f}, {y:.2f}, {theta:.2f}], Samples: {num_samples}")
                    result.write(f"\nDuration: {(time.time() - time_s):.2f}s, CPU: {psutil.cpu_percent()}%")
                    result.write(f"\n\t{sample_tries.Title}")
                    result.write(f"\n\tState:[{sample_tries.State}]")
                    result.write(f"\n-------------------------------------------------")
                    result.close()

                    print(open(path_result, 'r').read())
                    for i in range(len(open(path_result, 'r').readlines())):
                        print("\033[F\033[K", end='')
                    time.sleep(0.01)

                    if sample_tries.judge_finish() and sample_tries.judge_finish_thread():
                        sample_tries.get_state()
                        result = open(path_result, 'w')
                        result.write(f"\nNow: {time.asctime(time.localtime())}")
                        result.write(f"\nProcess:[{process}{process_all}] {schedule * 100:.2f}%")
                        result.write(f"\nInit_Data: [{x:.4f}, {y:.4f}, {theta:.4f}], Samples: {num_samples}")
                        result.write(f"\n\t{sample_tries.Title} {(time.time() - time_s):.2f}s")
                        result.write(f"\n\tState:[{sample_tries.State}]")
                        result.write(f"\n-------------------------------------------------")
                        result.close()
                        print(open(path_result, 'r').read())
                        break

                # 提取各线程结果
                for try_optim in sample_tries.tries_optim:
                    if try_optim.flag_success:
                        pass
                    else:
                        if try_optim.flag_init_data:
                            anchors_failed_list = np.append(anchors_failed_list, np.asarray([try_optim.init_data]), axis=0)
                # 对规划失败且初始化成功的线程进行绘图
                sample_tries.get_plot_failed()

                # 记录结果
                sample_tries.get_log()
                log = open(path_log, 'a+')
                log.write(f"\n-------------------------------------------------")
                log.write(f"\n{time.asctime(time.localtime())}")
                log.write(f"\nInit_Data: [{x:.4f}, {y:.4f}, {theta:.4f}], Samples: {num_samples}")
                log.write(f"\nResult:")
                log.write(f"\n\tSuccessOrNot: [{sample_tries.SuccessOrNot}]")
                log.write(f"\n\tInitOrNot: [{sample_tries.InitOrNot}]")
                log.write(f"\n\tTime: [{sample_tries.Time}]s")
                log.write(f"\n\tTimeoutOrNot: [{sample_tries.TimeInOrNot}]")
                log.write(f"\n\tResultOrNot: [{sample_tries.ResultOrNot}]")
                log.write(f"\n\tRunningOrNot: [{sample_tries.RunningOrNot}]")
                log.close()

                np.savetxt(path_anchors_failed, anchors_failed_list)

    anchors_failed_list = anchors_failed_list[1:, :]
    np.savetxt(path_anchors_failed, anchors_failed_list)
