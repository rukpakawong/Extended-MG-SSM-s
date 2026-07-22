import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the LSTM model

        Arg:
            input_size (int): the number of features in the input data.
            hidden_size: the number of features in the hidden state.
            num_layers: the number of LSTM layers.
            output_size: the size of the output from the final linear layer.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The core LSTM layer
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True # ensures inputs are read as (batch_size, sequence_length, features)
                            )

        # A fully connected layer to map the final hidden state to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Arg: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Initialize the hidden state (h0) and cell state (c0) with zeros
        # Shapes: (num_layeers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input and initial states through the LSTM
        # out shape: (batch_size, sequence_length, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We generally only care about the hidden state from the final time step
        # out shape: (batch_size, hidden_size)
        final_time_step_out = out[:, -1, :]

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the BiLSTM model

        Args:
            input_size (int): the number of features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of LSTM layers.
            output_size (int): the size of the output from the final linear layer.
        """
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The core BiLSTM layer
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, # ensures inputs are read as (batch_size, sequence_length, features)
                            bidirectional=True # makes the LSTM bidirectional
                            )

        # A fully connected layer to map the final hidden state to the desired output size.
        # Note: Because it is bidirectional, the hidden state size is doubled (hidden_size * 2).
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Initialize the hidden state (h0) and cell state (c0) with zeros.
        # Shapes: (num_layers * 2, batch_size, hidden_size) due to bidirectionality.
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Pass the input and initial states through the LSTM
        # out shape: (batch_size, sequence_length, hidden_size * 2)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We generally only care about the hidden state from the final time step
        # out shape: (batch_size, hidden_size * 2)
        final_time_step_out = out[:, -1, :]

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the GRU model

        Args:
            input_size (int): the number of features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of GRU layers.
            output_size (int): the size of the output from the final linear layer.
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The core GRU layer
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True # ensures inputs are read as (batch_size, sequence_length, features)
                          )

        # A fully connected layer to map the final hidden state to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Initialize the hidden state (h0) with zeros.
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input and initial state through the GRU
        # Note: Unlike LSTM, GRU only takes and returns a hidden state (no cell state)
        # out shape: (batch_size, sequence_length, hidden_size)
        out, hn = self.gru(x, h0)

        # We generally only care about the hidden state from the final time step
        # out shape: (batch_size, hidden_size)
        final_time_step_out = out[:, -1, :]

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions
    
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the BiGRU model

        Args:
            input_size (int): the number of features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of GRU layers.
            output_size (int): the size of the output from the final linear layer.
        """
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The core BiGRU layer
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True, # ensures inputs are read as (batch_size, sequence_length, features)
                          bidirectional=True # makes the GRU bidirectional
                          )

        # A fully connected layer to map the final hidden state to the desired output size.
        # Note: Because it is bidirectional, the hidden state size is doubled (hidden_size * 2).
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Initialize the hidden state (h0) with zeros.
        # Shape: (num_layers * 2, batch_size, hidden_size) due to bidirectionality.
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Pass the input and initial state through the GRU
        # Note: GRU only takes and returns a hidden state (no cell state)
        # out shape: (batch_size, sequence_length, hidden_size * 2)
        out, hn = self.gru(x, h0)

        # We generally only care about the hidden state from the final time step
        # out shape: (batch_size, hidden_size * 2)
        final_time_step_out = out[:, -1, :]

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions

class DiscreteSSMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the Discrete State Space Model (SSM)

        Args:
            input_size (int): the number of features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of SSM layers stacked together.
            output_size (int): the size of the output from the final linear layer.
        """
        super(DiscreteSSMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # PyTorch does not have a built-in nn.SSM layer.
        # We manually construct the A, B, C, and D matrices for each layer.
        self.ssm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # The first layer takes the raw input_size, subsequent layers take the hidden_size
            layer_input_size = input_size if i == 0 else hidden_size
            
            # A: State transition matrix (hidden state -> hidden state)
            # B: Input matrix (input -> hidden state)
            # C: Output matrix (hidden state -> layer output)
            # D: Feedthrough matrix (input -> layer output)
            
            # We use nn.Linear(bias=False) to act as pure matrix multiplications
            matrices = nn.ModuleDict({
                'A': nn.Linear(hidden_size, hidden_size, bias=False),
                'B': nn.Linear(layer_input_size, hidden_size, bias=False),
                'C': nn.Linear(hidden_size, hidden_size, bias=False),
                'D': nn.Linear(layer_input_size, hidden_size, bias=False)
            })
            self.ssm_layers.append(matrices)

        # A fully connected layer to map the final output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        batch_size, seq_len, _ = x.size()

        # Initialize the hidden state (h0) for each layer with zeros.
        # We use a standard Python list to hold the state for each layer.
        # (This prevents PyTorch autograd in-place modification errors during backprop).
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        # We must process the sequence step-by-step
        for t in range(seq_len):
            # Extract the current timestep: shape (batch_size, input_size)
            x_t = x[:, t, :] 
            
            for layer_idx, layer in enumerate(self.ssm_layers):
                
                # 1. State Equation: h_t = A * h_{t-1} + B * x_t
                h[layer_idx] = layer['A'](h[layer_idx]) + layer['B'](x_t)
                
                # 2. Output Equation: y_t = C * h_t + D * x_t
                y_t = layer['C'](h[layer_idx]) + layer['D'](x_t)
                
                # The output (y_t) becomes the input (x_t) for the next layer in the stack
                x_t = y_t

        # We generally only care about the output from the final time step of the final layer
        # x_t currently holds the final time step output because the loops have finished
        final_time_step_out = x_t

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions