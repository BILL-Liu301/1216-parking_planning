from .base_paras import num_samples
from .util_try_optim import TryOptim
from .util_show_init import show_init


class SampleTries:
    def __init__(self, init_data):
        self.init_data = init_data
        self.tries_optim = []

        self.Title = ""
        self.State = ""
        self.FinishOrNot = " "
        self.SuccessOrNot = " "
        self.InitOrNot = " "
        self.Time = " "
        self.TimeInOrNot = " "
        self.ResultOrNot = " "
        self.RunningOrNot = " "

    def set_tries(self):
        for sample in range(num_samples):
            self.tries_optim.append(TryOptim(self.init_data[sample]))

    def set_run(self):
        for try_optim in self.tries_optim:
            try_optim.start()

    def judge_finish(self):
        flag = True
        for try_optim in self.tries_optim:
            flag = flag and try_optim.flag_finish
        return flag

    def judge_finish_thread(self):
        flag = True
        for try_optim in self.tries_optim:
            flag = flag and (not try_optim.is_alive())
        return flag

    def get_state(self):
        self.State = "\t"
        self.Title = "\t"
        for index_try, try_optim in enumerate(self.tries_optim):
            self.Title += f" Try_{index_try} \t"
            if not try_optim.flag_init_data:
                self.State += "INIFail\t"  # 数据初始化失败
            else:
                if try_optim.flag_finish:
                    if try_optim.flag_timeout:
                        self.State += "TimeOut\t"   # 求解超时
                    elif not try_optim.flag_result:
                        self.State += "ResFail\t"   # 结果的SNOPT不为1
                    elif not try_optim.flag_running:
                        self.State += "MATFail\t"   # 求解过程报错
                    elif try_optim.flag_success:
                        self.State += "Success\t"  # 求解成功
                    else:
                        self.State += "Failure\t"  # 求解失败
                else:
                    self.State += "Running\t"  # 正在求解

    def get_log(self):
        self.FinishOrNot = " "
        for try_optim in self.tries_optim:
            self.FinishOrNot += "√ " if try_optim.flag_finish else "× "
            self.SuccessOrNot += "√ " if try_optim.flag_success else "× "
            self.InitOrNot += "√ " if try_optim.flag_init_data else "× "
            self.Time += f"{try_optim.planning_time} "
            self.TimeInOrNot += "√ " if (not try_optim.flag_timeout) else "× "
            self.ResultOrNot += "√ " if try_optim.flag_result else "× "
            self.RunningOrNot += "√ " if try_optim.flag_running else "× "

    def get_plot_failed(self):
        # 对规划失败且初始化成功的线程进行绘图
        for index_try, try_optim in enumerate(self.tries_optim):
            if try_optim.flag_init_data and (not try_optim.flag_success):
                show_init(self.init_data[index_try])

    def main(self):
        self.set_tries()
        self.set_run()
