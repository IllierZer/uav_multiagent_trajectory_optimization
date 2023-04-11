import numpy as np
# defines the parameters involved in formulating the optimization problem

p_max_val = 50 # (m)
p_min_val = -50 # (m)
v_max_val = 2 # m/s
v_min_val = -2 # m/s
a_max_val = 0.5 # m/s^2
a_min_val = -0.5 # m/s^2
j_max_val = 2. # m/s^3
j_min_val = -2. # m/s^3


p_max = p_max_val * np.ones(3)
p_min = p_min_val * np.ones(3)
v_max = v_max_val * np.ones(3)
v_min = v_min_val * np.ones(3)
a_max = a_max_val * np.ones(3)
a_min = a_min_val * np.ones(3)
j_max = j_max_val * np.ones(3)
j_min = j_min_val * np.ones(3)
R = 0.5 # distance of avoidance (m)
g = -9.81 # m/s^2
