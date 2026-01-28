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
        nn.init.uniform_(self.linear.weight, -1, 1)
        
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
        
        # Capas ocultas (8 capas según lógica del script original: 2 * (input + output))
        # El script original calcula layers dinámicamente.
        # layers = [3] + (2*(3+3))*[neurons] + [3] => [3, 600, ..., 600, 3] -> 12 hidden layers
        n_layers = 2 * (input_dim + output_dim) 
        self.hidden_layers = nn.ModuleList([
            GammaBiasLayer(hidden_neurons, hidden_neurons) for _ in range(n_layers - 1)
        ])
        
        # Capa de salida
        self.output_layer = GammaBiasLayer(hidden_neurons, output_dim)
        
        self.activation = nn.Tanh()

    def forward(self, t, x, y):
        # Concatenar entradas
        inputs = torch.cat([t, x, y], dim=1)
        
        h = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        
        output = self.output_layer(h)
        return output # Retorna [u, v, p]