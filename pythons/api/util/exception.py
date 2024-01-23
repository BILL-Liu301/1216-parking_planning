class InitDataMeaningLess(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return f"初始化数据无效。"


class MatlabTimeOut(Exception):
    def __init__(self, planning_time_ref):
        self.planning_time_ref = planning_time_ref

    def __str__(self):
        return f"MATLAB伪谱法路径规划已经超过{self.planning_time_ref}s。"


class MatlabResultFail(Exception):
    def __init__(self, SNOPT_info):
        self.SNOPT_info = SNOPT_info

    def __str__(self):
        return f"当前求解结果为{self.SNOPT_info}。"


class MatlabPlanningMeaningless(Exception):
    def __init__(self, others_pkl_filename, l, index_list):
        self.others_pkl_filename = others_pkl_filename
        self.l = l
        self.index_list = index_list

    def __str__(self):
        return f"对于{self.others_pkl_filename}，在{self.l}下，对场景{self.index_list}的规划没有意义。"
