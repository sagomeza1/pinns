
import torch
import torch.nn as nn

from typing import Optional, Any, Tuple, List

mse = nn.MSELoss()

def compute_derivatives(model, t, x, y):
    """Calcula las derivadas necesarias usando autograd."""
    # Habilitar gradientes para las entradas
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True
    
    outputs = model(t, x, y)
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p = outputs[:, 2:3]

    # Gradientes de primer orden
    grads_u = torch.autograd.grad(u, [t, x, y], grad_outputs=torch.ones_like(u), create_graph=True)
    u_t, u_x, u_y = grads_u[0], grads_u[1], grads_u[2]

    grads_v = torch.autograd.grad(v, [t, x, y], grad_outputs=torch.ones_like(v), create_graph=True)
    v_t, v_x, v_y = grads_v[0], grads_v[1], grads_v[2]

    grads_p = torch.autograd.grad(p, [x, y], grad_outputs=torch.ones_like(p), create_graph=True)
    p_x, p_y = grads_p[0], grads_p[1]

    return u, v, p, u_t, u_x, u_y, v_t, v_x, v_y, p_x, p_y

def loss_navier_stokes(model, t, x, y):
    """Calcula el residuo de las ecuaciones de Navier-Stokes."""
    u, v, p, u_t, u_x, u_y, v_t, v_x, v_y, p_x, p_y = compute_derivatives(model, t, x, y)

    # Ecuaciones residuales (asumiendo adimensionalización previa)
    # Continuidad
    e1 = u_x + v_y
    # Momentum X
    e2 = u_t + (u * u_x + v * u_y) + p_x
    # Momentum Y
    e3 = v_t + (u * v_x + v * v_y) + p_y

    mse = nn.MSELoss()
    zeros = torch.zeros_like(e1)
    
    return mse(e1, zeros) + mse(e2, zeros) + mse(e3, zeros)

def loss_u(model, t, x, y, target) -> float:
    outputs = model(t, x, y)
    u = outputs[:, 0:1]
    variance = torch.var(target) if torch.var(target) > 0 else 1.0
    return mse(u, target)

def loss_v(model, t, x, y, target) -> float:
    outputs = model(t, x, y)
    v = outputs[:, 1:2]
    variance = torch.var(target) if torch.var(target) > 0 else 1.0
    return mse(v, target)

def loss_p(model, t, x, y, target) -> float:
    outputs = model(t, x, y)
    p = outputs[:, 2:3]
    variance = torch.var(target) if torch.var(target) > 0 else 1.0
    return mse(p, target)

def loss_data_variable(pred, target):
    """MSE normalizado por varianza (std^2) del target."""
    mse = nn.MSELoss()
    variance = torch.var(target) if torch.var(target) > 0 else 1.0
    return mse(pred, target) / variance

