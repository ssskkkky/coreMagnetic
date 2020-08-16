from netgen.csg import *
from ngsolve import *


def MakeGeometry():
    geometry = CSGeometry()
    box = OrthoBrick(Pnt(-1,-1,-1),Pnt(2,1,2)).bc("outer")

    core = OrthoBrick(Pnt(0.45,-0.05,0),Pnt(0.55,0.05,1)).maxh(0.2).mat("core")
    geometry.Add ((box-core).mat("air"))
    geometry.Add (core)
    return geometry

ngmesh = MakeGeometry().GenerateMesh(maxh=0.5)
mesh = Mesh(ngmesh)

fes = H1(mesh, order=4,dirichlet=[1])

u = fes.TrialFunction()
v = fes.TestFunction()

a = BilinearForm(fes, symmetric=True)
a += grad(u)*grad(v)*dx

f = LinearForm(fes)
f += CoefficientFunction((1)) * v * dx("core")

a.Assemble()
f.Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

Draw (gfu, mesh, "B-field")
