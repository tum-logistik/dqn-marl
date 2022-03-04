import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# create polynomial convex function
def convex_poly_func(x_vals, theo_max = 100, theo_opt_x = 50, senstivity = 0.01):
    func_val = theo_max
    for i in range(len(x_vals)):
        func_val = func_val - senstivity * np.power((x_vals[i] - theo_opt_x), 2)
    return func_val

def convex_q_gen(s, a):
    q_val = convex_poly_func(a)
    noise = np.clip(np.random.normal(5,2,1)[0], 0, 100)
    return q_val + noise

