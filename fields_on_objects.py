import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)
dim=3
dt = 2e-4
pi=ti.math.pi
rad,mass=1e-4,1e-3 #M,K,S  
x_border=[0,1]
y_border=[0,1]
z_border=[0,1]
energy_loss=0.9
G=[0,-9.8,0]
K=1
num_pts=1000
scale=1
N=10000





n_particles=10000
rad,mass,Q=1e-2,1e-3,1


particles=ti.Vector.field(3,ti.f32,shape=(n_particles),needs_grad=True)
velocity=ti.Vector.field(3,ti.f32,shape=(n_particles))
acceleration=ti.Vector.field(3,ti.f32,shape=(n_particles))
indices=ti.field(ti.f32,shape=(n_particles))
sphere=ti.Vector.field(3,ti.f32,shape=(num_pts))


@ti.kernel
def init():
    for i in ti.ndrange(n_particles):
                particles[i]=ti.Vector([ti.randn(ti.f32) ,ti.randn(ti.f32)  ,ti.randn(ti.f32) ])  
                #velocity[i]=ti.Vector([ti.randn(ti.f32),ti.randn(ti.f32),ti.randn(ti.f32)])  
                velocity[i]=ti.Vector([0,0,0])
                #acceleration[i]=ti.Vector(G)


#def MakeSphere():
    for i in ti.ndrange(num_pts):
        indices[i]=i+0.5
        phi = ti.acos(1 - 2*indices[i]/num_pts)
        theta = pi * (1 + 5**0.5) * indices[i]
        sphere[i]=ti.Vector([ti.cos(theta) * ti.sin(phi), ti.sin(theta) * ti.sin(phi), ti.cos(phi)])


@ti.kernel
def compute_forces():
    for i in ti.ndrange(n_particles):
        for j in ti.ndrange(n_particles):
                r_vector= particles[j] - particles[i]
                r_size=ti.sqrt(r_vector[0] ** 2 + r_vector[1] ** 2 + r_vector[2] ** 2)
                r_hat = r_vector * r_size
                acceleration[i]+= ( (K * Q) * r_size**2) * r_hat #multipy instead of dividing because -1<r<1




@ti.kernel
def advance():
    for i in ti.ndrange(n_particles):
        #velocity[i]+= acceleration[i] * dt
        particles[i] += velocity [i] *dt + acceleration[i] * dt **2




@ti.kernel
def borders():
    #"""
    for i in ti.ndrange(n_particles):
        for j in ti.static(range(3)):

            if particles[i][j]<=0:
                acceleration[i][j]=-acceleration[i][j] 
                particles[i][j] = 0.0001

            

            if particles[i][j]>=1:
                acceleration[i][j]=-acceleration[i][j] 
                particles[i][j] =0.999
    """
    for i in ti.ndrange(num_pts):
        for j in ti.ndrange(n_particles):
                r_vector=  particles[j] - sphere[i] 
                r_size=ti.sqrt(r_vector[0] ** 2 + r_vector[1] ** 2 + r_vector[2] ** 2)
                r_hat = r_vector * r_size
                if r_size <=1e-3:
                    acceleration[j]=-acceleration[j] 
    
                    #particles[j]=  sphere[i] 
                #if r_size <=3:
                    #acceleration[j]=-acceleration[j]
                    #particles[j]=sphere[i]
    """
#@ti.kernel
#def rotate():




def substep():

    compute_forces()
    advance()
    borders()





res = (500, 500)
window = ti.ui.Window("test", res, vsync=True,show_window=True)

canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(1.2, 1.2, 1.2 ) 
camera.lookat(0.5,0.5,0.5)
camera.fov(90)



def render():
    #camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0.2, 0.2, 0.2))


    scene.particles(particles,rad,color=(0.9,0.2,0.7))

    #scene.particles(sphere,1e-3,color=(0.3,0.1,0.5))
    scene.point_light(pos=(9, 9, 900), color=(1, 1, 1))
    scene.point_light(pos=(0, 1, 0), color=(1, 1, 1))
    #scene.point_light(pos=(0, 0, 1), color=(1, 1, 1))
    #scene.point_light(pos=(5, 5, 5), color=(1, 1, 1))
    #scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    #scene.mesh(particles,None,None)

    canvas.scene(scene)


positive=1
negative=1
def main():
    result_dir='3d_particles444'
    video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=20, automatic_build=False,)
    frame_id = 0
    init()
    x,y,z = 2,2,2
    #while window.running:
    for frame in range(110):

        render()
        substep()
        video_manager.write_frame(window.get_image_buffer())
        window.show()
        #print(f'Frame {frame} is recorded', end='')


        if  y>-2:
            x-=1e-2
            y-=15e-3
            z-=15e-3
            camera.position(x, y, z) 
    print()
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True,mp4= False)
    #print(f'MP4 video is saved to {video_manager.get_output_filename("field_describe.mp4")}')
    print(f'GIF video is saved to {video_manager.get_output_filename("field_describe.gif")}')


if __name__ == '__main__':
    main()