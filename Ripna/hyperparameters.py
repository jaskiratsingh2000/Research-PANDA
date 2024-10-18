import numpy as np

noise = 0
n = 20
radius = 50
delt = 0.05

epsilon = np.pi/64
k=50

min_radius=15 # changed 

spread = np.pi/16

sensor_range = 10
"""
priority = False, Gaussian, Uniform
"""
priority_type="Gaussian"
pert=True

maxv = 7

"""
priority to import to the other files
"""
seed = 5


"""
b represents the spread of the repulsion field
"""
b=0.05

"""
strength of vector force fields
"""
attractive_gain=20
repulsive_gain=12


"""
distance within which if you get it is a collision
"""
collision_distance=1


"""
clipping power represents how strong clipping is
"""
clipping_power=1


#Looking into tiime scales as RIPNA is for fixed wings and PANDa for both, ours is more heterogenous (includes fixed and quad)