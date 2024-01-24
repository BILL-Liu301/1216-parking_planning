import os
from threading import Thread
import numpy as np

from api.base.paras import num_planning_time_ref, paras_base
from .exception import InitDataMeaningLess, MatlabTimeOut, MatlabResultFail
from .call_optim import CallOptim
from .wait_matlab import WaitMatlab
from api.base.paths import path_solutions


class TryOptim(Thread):
    def __init__(self, init_data):
        Thread.__init__(self)
        self.init_data = init_data
        self.flag_finish = False
        self.flag_init_data = True
        self.flag_timeout = False
        self.flag_result = True
        self.flag_running = True
        self.flag_success = False
        self.planning_time_ref = num_planning_time_ref
        self.planning_time = 0
        self.wm = WaitMatlab(self.planning_time_ref)
        self.co = CallOptim(init_data=init_data)
        self.result = None
        self.anchors = None

    def run(self):
        try:
            ymid_r = self.init_data[1]
            ymid_f = ymid_r + np.sin(self.init_data[2]) * paras_base["Car_L"]
            ymid_f = ymid_f + np.sin(self.init_data[2]) * (paras_base["Car_Length"] - paras_base["Car_L"]) / 2

            yf_r = ymid_f + np.sin(self.init_data[2] - np.pi / 2) * paras_base["Car_Width"] / 2
            yf_l = ymid_f + np.sin(self.init_data[2] + np.pi / 2) * paras_base["Car_Width"] / 2
            if yf_r < paras_base["Parking_Y"] or yf_l > (paras_base["Freespace_Y"] + paras_base["Parking_Y"]):
                raise InitDataMeaningLess

            self.co.start()
            self.wm.start()
            while True:
                if self.co.done():
                    self.wm.stop()
                    break
                else:
                    if self.wm.timeout:
                        self.co.cancel()
                        raise MatlabTimeOut(self.planning_time_ref)
            self.result = self.co.result()
            SNOPT_info = self.result['SNOPT_info']
            if SNOPT_info != 1.0:
                raise MatlabResultFail(SNOPT_info)
        except InitDataMeaningLess as e:
            self.flag_init_data = False
        except MatlabTimeOut as e:
            self.flag_timeout = True
        except MatlabResultFail as e:
            self.flag_result = False
        except Exception as e:
            self.flag_running = False
        else:
            self.flag_success = True
            self.anchors = self.result['parking_anchors']
            solution = np.asarray(self.result['solution'])
            np.savetxt(os.path.join(path_solutions, f"{self.init_data[0]:.4f}_{self.init_data[1]:.4f}_{self.init_data[2]:.4f}.txt"), solution)

        finally:
            self.co.quit()
            self.planning_time = self.wm.planning_time
            self.flag_finish = True
