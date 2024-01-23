# 1216-parking_planning
The trajectory planning for parking. Include "Making Dataset", "Training", "Testing" and "Analyse"

# Parking_Trajectory_Planner
使用pytorch_lighting对模型进行重构  
模型内总共包含了四个相同的组，每个组对应倒车入库轨迹的四个阶段  
每个组包含了以下几个部分：  
【encode_last_anchor】：对上一时刻的点进行编码  
【encode_last_map_linear_q，encode_last_map_linear_k，encode_last_map_linear_v，encode_last_map_attention，encode_last_map_linear，encode_last_map_norm】：对上一时刻的地图进行编码  
【main_lstm，main_norm】：核心lstm  
【decode_mean，decode_var】：对均值与方差进行解码


## macro
![macro.png](best_version%2Fmacro.png)
## micro
![micro.png](best_version%2Fmicro.png)
