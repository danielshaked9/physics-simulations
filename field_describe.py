import taichi as ti
import numpy as np
res=512 #1000 --> x 1000 --> y area of screen --> 10000 pixesls
dens=20 #pixels between points on the field
N=int((res/dens))
tt=np.linspace(0,1,N)
npfield=np.empty((N,N,2))
for i in range(N):
    for j in range(N):
        npfield[i,j][0]=tt[j]
        npfield[i,j][1]=tt[i]
npfieldi=npfield.reshape(N*N,2)

ti.init(arch=ti.gpu)
Q=-1.6e-19  #charge - normalized for visualization 1e-19 * 1et
K=9e9
img_par=ti.Vector.field(2,ti.f64,shape=(N,N))
img_par.from_numpy(npfield)
r_vector=ti.Vector.field(2,ti.f64,shape=(N,N))
r_hat=ti.Vector.field(2,ti.f64,shape=(N,N))
e_field=ti.Vector.field(2,ti.f64,shape=(N,N))
mousex=ti.field(ti.f64,())
mousey=ti.field(ti.f64,())
gui=ti.GUI('Field',res=res)



@ti.kernel
def compute_field():
    for i in ti.ndrange(N):
        for j in ti.ndrange(N):
            r_vector[i,j]=ti.Vector([img_par[i,j][0] - mousex[None] , img_par[i,j][1] - mousey[None]])
            r_hat[i,j] = r_vector[i,j] / ( ti.pow(ti.sqrt(r_vector[i,j][0] ** 2 + r_vector[i,j][1] ** 2),2) )
            e_field[i,j]= (K * Q) * r_hat[i,j] 





def substep():
    compute_field()





while gui.running:

    mousex[None], mousey[None] = gui.get_cursor_pos()
    #mousey[None], mousex[None] = gui.get_cursor_pos()
    mouse=np.array(([mousex[None],mousey[None]]))

    substep()
    #gui.circles(npfieldi,8,color=0xFF0000)
    if Q>0:
        gui.circle(mouse,0xFF0000,8)
        for i in range(N):
            for j in range(N):
                gui.arrow(npfield[i,j],e_field[i,j] * 1e6 ,1,0xFF0000)
    else:
        gui.circle(mouse,0x0000FF,8)

        for i in range(N):
            for j in range(N):
                gui.arrow(npfield[i,j],e_field[i,j] * 1e6 ,1,0xFF00FF)
    
    #gui.arrow_field(-e_field.to_numpy() * 1e6)

    gui.show()
print (Q)
