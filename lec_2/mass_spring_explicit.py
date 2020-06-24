import taichi as ti

ti.init(debug=True)

max_num_particles = 256

dt = 1e-3

num_particles = ti.var(ti.i32, shape=())
spring_stiffness = ti.var(ti.f32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())

particle_mass = 1
bottom = 0.05
top = 0.95

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

A = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
b = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))

connection_radius = 0.15

@ti.kernel
def substep(cur_x: ti.f32, cur_y: ti.f32, mag: ti.f32):
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None]) # damping
        i2cur = [cur_x, cur_y] - x[i]
        total_force = i2cur.normalized() * mag * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        v[i] += dt * total_force / particle_mass
        
    # Collide with boundary
    for i in range(n):
        if x[i].y < bottom:
            x[i].y = bottom
            v[i].y = -v[i].y
        if x[i].y > top:
            x[i].y = top
            v[i].y = -v[i].y
        if x[i].x < bottom:
            x[i].x = bottom
            v[i].x = -v[i].x
        if x[i].x > top:
            x[i].x = top
            v[i].x = -v[i].x

    # Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt

        
@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32): # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1
    
    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1
    
    
gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)

spring_stiffness[None] = 10000
damping[None] = 20

new_particle(0.3, 0.3)
new_particle(0.3, 0.4)
new_particle(0.4, 0.4)

mouse_x, mouse_y = 0, 0
magnification = 1

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        
        if not paused[None]:
            pass
        else:
            if e.key == ti.GUI.LMB:
                new_particle(e.pos[0], e.pos[1])
            elif e.key == 'c':
                num_particles[None] = 0
                rest_length.fill(0)
            elif e.key == 's':
                if gui.is_pressed('Shift'):
                    spring_stiffness[None] /= 1.1
                else:
                    spring_stiffness[None] *= 1.1
            elif e.key == 'd':
                if gui.is_pressed('Shift'):
                    damping[None] /= 1.1
                else:
                    damping[None] *= 1.1

    if not paused[None]:
        mouse_x, mouse_y = gui.get_cursor_pos()
        lmb_pressed = gui.is_pressed(ti.GUI.LMB)
        rmb_pressed = gui.is_pressed(ti.GUI.RMB)
        sign = 1
        if (lmb_pressed or rmb_pressed) and magnification >= -20 and magnification <= 20:
            magnification *= 1.1
        elif not (lmb_pressed or rmb_pressed):
            magnification = 1

        if rmb_pressed:
            sign = -1

        for step in range(10):
            substep(mouse_x, mouse_y, sign * magnification)

    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)
    
    gui.rect([top, top], [bottom, bottom], radius=1, color=0x0)
    
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.show()

