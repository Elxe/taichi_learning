# mls-mpm: translate from c++ version of mls-mpm 88 line 

import taichi as ti 
import os
import math
import numpy as np
import matplotlib.pyplot as plt

real = ti.f32
ti.init(arch=ti.gpu)

dim = 2
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3

# Snow material properties
particle_mass = 1
vol = 1            # Particle volume
hardening = 10     # Snow hardening factor
E = 1e4            # Young's modulus
nu = 0.2           # Poisson ratio
plastic = True

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

actuator_id = ti.var(ti.i32)
particle_type = ti.var(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

# Initial lame parameters
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# taichi layout
@ti.layout

