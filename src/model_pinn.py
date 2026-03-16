import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class GammaBiasLayer(nn.Module):
    """
    Capa densa personalizada con normalización de pesos, 
    factor gamma y sesgo explícito.
    Equivalente a la implementación de TF provista.
    """
    def __init__(self, in_features, out_features):
        super(GammaBiasLayer, self).__init__()
        # Weight Normalization similar a tfa.layers.WeightNormalization
        self.linear = weight_norm(nn.Linear(in_features, out_features, bias=False))
        
        # Inicialización uniforme [-1, 1] para los pesos
        # nn.init.uniform_(self.linear.weight, -1, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        
        # Parámetros entrenables gamma y bias
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return self.gamma * self.linear(x) + self.bias
    
class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_neurons=600):
        super(PINN, self).__init__()
        
        # Capa de entrada
        self.input_layer = GammaBiasLayer(input_dim, hidden_neurons)
        
        n_layers = 2 * (input_dim + output_dim)
        d = 2
        
        # Capas ocultas con activación Tanh (la capa de entrada es la primera, para un total de 8)
        self.tanh_layers = nn.ModuleList([
            GammaBiasLayer(hidden_neurons, hidden_neurons) for _ in range(n_layers - d)
        ])
        
        # Capas ocultas con activación Lineal
        self.linear_layers = nn.ModuleList([
            GammaBiasLayer(hidden_neurons, hidden_neurons) for _ in range(n_layers - d, n_layers)
        ])
        
        # Capa de salida
        self.output_layer = GammaBiasLayer(hidden_neurons, output_dim)
        
        self.activation_tanh = nn.Tanh()

    def forward(self, t, x, y):
        # Concatenar entradas
        inputs = torch.cat([t, x, y], dim=1)
        
        h = self.activation_tanh(self.input_layer(inputs))
        for layer in self.tanh_layers:
            h = self.activation_tanh(layer(h))
        
        for layer in self.linear_layers:
            h = layer(h)
        
        output = self.output_layer(h)
        return output # Retorna [u, v, p]