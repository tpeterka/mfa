import mfa
import diy

domain = diy.DoubleContinuousBounds([-4., -4.], [4., 4.])

print("Before Pybind:")
print("dimension for min bound =", domain.min.dimension())
print("dimension for max bound =", domain.max.dimension())
print("After Pybind:")
mfa.get_bound(domain)