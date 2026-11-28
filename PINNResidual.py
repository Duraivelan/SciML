#PINN residual with autograd
x.requires_grad_(True)
u = model(x)
du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
residual = d2u - f(x) # e.g. u'' = f(x)
loss = residual.pow(2).mean() + 10 * ((model(x_bc) - u_bc) ** 2).mean()
