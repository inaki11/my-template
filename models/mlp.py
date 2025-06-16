import torch.nn as nn

def build_model(input_size, output_size, config):
    # Extract model parameters from the config (assumes they are under the "model" key)
    # first assert that all required keys are present
    required_keys = ["num_layers", "hidden_size", "activation", "dropout"]
    print(f"Config dict inside MLP: {config}")
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required model configuration key: {key}")
        
    num_layers = config.num_layers
    hidden_size = config.hidden_size
    activation_name = config.activation
    dropout_rate = config.dropout
    # Choose an activation function based on the config setting
    if activation_name == "relu":
        activation_fn = nn.ReLU
    elif activation_name == "tanh":
        activation_fn = nn.Tanh
    elif activation_name == "sigmoid":
        activation_fn = nn.Sigmoid
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")

    layers = []

    # Input layer to first hidden layer
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(activation_fn())
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))

    # Add additional hidden layers if specified
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

    # Final layer to output
    layers.append(nn.Linear(hidden_size, output_size))

    model = nn.Sequential(*layers)
    return model
