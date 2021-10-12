"""
@ref https://www.johndcook.com/blog/2019/12/22/van-der-pol/    
"""
def van_der_pol(t, z, mu):
    x, y = z
    return [y, mu * (1 - x**2) * y - x]

def van_der_pol_with_eps(t, z, eps, mu = 1.0):
    x, y = z
    return [y, eps * (mu * (1 - x**2) * y - x)]