import autodiff_multivector as ad
import versor_pybind11 as vsr

jac, statements = ad.diff_stan(vsr.Vec(1,2,3), vsr.Biv(1,1,1))
print(jac)
print(statements)
print
jac, statements = ad.diff_ceres(vsr.Vec(1,2,3), vsr.Biv(1,1,1))
print(jac)
print(statements)
