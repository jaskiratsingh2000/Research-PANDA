import numpy as np

noise = 0
n = 20
radius = 50
delt = 0.05 #changed from 0.05

spread = np.pi/16

sensor_range = 10
"""
priority = False, Gaussian, Uniform
"""
priority_type="Gaussian"
pert=True

maxv = 7 #changed from 7

"""
priority to import to the other files
"""
seed = 5


"""
b represents the spread of the repulsion field
"""
b=0.05 #changed from 0.05

"""
strength of vector force fields
"""
attractive_gain=20 # changed from 14
repulsive_gain=12


"""
distance within which if you get it is a collision
"""
collision_distance=1


"""
clipping power represents how strong clipping is
"""
clipping_power=1