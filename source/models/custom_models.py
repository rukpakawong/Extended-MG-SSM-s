import torch
import torch.nn as nn

class MGSSMCell(nn.Module):
    def __init__(self, input_size, hidden_size, gate_size):
        """
        Initialize a single Multiplicative Gated State Space Model (MGSSM) Cell.
        """
        super(MGSSMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_size = gate_size

        # Linear input gate parameters
        self.W_A = nn.Linear(hidden_size, hidden_size)
        self.W_B = nn.Linear(input_size, hidden_size)
        self.W_bh = nn.Parameter(torch.zeros(hidden_size))

        # Multiplicative input gate parameters
        self.W_E = nn.Linear(gate_size, gate_size)
        self.W_bg = nn.Parameter(torch.zeros(gate_size))

    def forward(self, x, hidden_state):
        """
        Defines the forward pass for a single time step in the cell.
        """
        h_prev, g_prev = hidden_state

        # Hidden state computation
        h_t = self.W_A(h_prev) + self.W_B(x) + self.W_bh
        
        # Gated state computation
        g_t = self.W_E(g_prev) * torch.repeat_interleave(x, self.gate_size, dim=1)

        return h_t, g_t

class MGSSMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, gate_size=0):
        """
        Initialize the customized MGSSM Model.

        Args:
            input_size (int): the number of expected features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of recurrent layers stacked together.
            output_size (int): the size of the output from the final linear layer.
            gate_size (int): the number of features in the multiplicative gate cell.
        """
        super(MGSSMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Default gate_size to half of hidden_size if not specified
        if gate_size == 0:
            gate_size = int(hidden_size / 2)
        self.gate_size = gate_size

        # The core recurrent blocks
        self.SSM_blocks = nn.ModuleList([
            MGSSMCell(input_size if i == 0 else hidden_size, hidden_size, gate_size) 
            for i in range(num_layers)
        ])

        # Output mapping layers
        self.W_C = nn.Linear(hidden_size, output_size)
        self.W_D = nn.Linear(input_size, output_size)
        self.W_J = nn.Linear(gate_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize the hidden state (h0) and gated state (g0) for each layer
        # Shapes: h0 is (batch_size, hidden_size), g0 is (batch_size, gate_size)
        hidden_state = [
            (torch.zeros(batch_size, self.hidden_size).to(x.device),
             torch.ones(batch_size, self.gate_size).to(x.device)) 
            for _ in range(self.num_layers)
        ]

        # Process the sequence step-by-step
        for t in range(seq_len):
            # Extract the current timestep: shape (batch_size, input_size)
            input_t = x[:, t, :]
            
            for i in range(self.num_layers):
                h_prev, g_prev = hidden_state[i]

                # Pass through the MGSSM Cell
                h_t, g_t = self.SSM_blocks[i](input_t, (h_prev, g_prev)) 
                
                # Update the hidden state for the next time step
                hidden_state[i] = (h_t, g_t) 
                
                # The hidden output becomes the input for the next layer in the stack
                input_t = h_t 

        # We generally only care about the hidden and gate states from the final time step of the last layer
        h_final, g_final = hidden_state[-1]

        # Pass through the linear output layers to get final predictions
        # x[:, -1, :] grabs the very last raw input timestep (cleaner Python syntax for seq_len - 1)
        out = self.W_C(h_final) + self.W_D(x[:, -1, :]) + self.W_J(g_final)
        
        return out

class MGSSMsCell(nn.Module):
    def __init__(self, input_size, hidden_size, gate_size):
        """
        Initialize a single Multiplicative Gated State Space Model (MGSSMs) Cell.
        """
        super(MGSSMsCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_size = gate_size

        # Linear input gate parameters
        self.W_A = nn.Linear(hidden_size, hidden_size)
        self.W_B = nn.Linear(input_size, hidden_size)
        self.W_bh = nn.Parameter(torch.zeros(hidden_size))

        # Multiplicative input gate parameters
        self.W_E = nn.Linear(gate_size, gate_size)
        self.W_F = nn.Linear(input_size, gate_size)
        self.W_bg = nn.Parameter(torch.zeros(gate_size))

    def forward(self, x, hidden_state):
        """
        Defines the forward pass for a single time step in the cell.
        """
        h_prev, g_prev = hidden_state

        # Hidden state computation
        h_t = self.W_A(h_prev) + self.W_B(x) + self.W_bh
        
        # Gated state computation
        g_t = self.W_E(g_prev) * torch.repeat_interleave(x, self.gate_size, dim=1) + self.W_F(x) + self.W_bg

        return h_t, g_t

class MGSSMsModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, gate_size=0):
        """
        Initialize the customized MGSSMs Model.

        Args:
            input_size (int): the number of expected features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of recurrent layers stacked together.
            output_size (int): the size of the output from the final linear layer.
            gate_size (int): the number of features in the multiplicative gate cell.
        """
        super(MGSSMsModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Default gate_size to half of hidden_size if not specified
        if gate_size == 0:
            gate_size = int(hidden_size / 2)
        self.gate_size = gate_size

        # The core recurrent blocks
        self.SSM_blocks = nn.ModuleList([
            MGSSMsCell(input_size if i == 0 else hidden_size, hidden_size, gate_size) 
            for i in range(num_layers)
        ])

        # Output mapping layers
        self.W_C = nn.Linear(hidden_size, output_size)
        self.W_D = nn.Linear(input_size, output_size)
        self.W_J = nn.Linear(gate_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize the hidden state (h0) and gated state (g0) for each layer
        # Shapes: h0 is (batch_size, hidden_size), g0 is (batch_size, gate_size)
        hidden_state = [
            (torch.zeros(batch_size, self.hidden_size).to(x.device),
             torch.ones(batch_size, self.gate_size).to(x.device)) 
            for _ in range(self.num_layers)
        ]

        # Process the sequence step-by-step
        for t in range(seq_len):
            # Extract the current timestep: shape (batch_size, input_size)
            input_t = x[:, t, :]
            
            for i in range(self.num_layers):
                h_prev, g_prev = hidden_state[i]

                # Pass through the MGSSMs Cell
                h_t, g_t = self.SSM_blocks[i](input_t, (h_prev, g_prev)) 
                
                # Update the hidden state for the next time step
                hidden_state[i] = (h_t, g_t) 
                
                # The hidden output becomes the input for the next layer in the stack
                input_t = h_t 

        # We generally only care about the hidden and gate states from the final time step of the last layer
        h_final, g_final = hidden_state[-1]

        # Pass through the linear output layers to get final predictions
        # x[:, -1, :] grabs the very last raw input timestep (cleaner Python syntax for seq_len - 1)
        out = self.W_C(h_final) + self.W_D(x[:, -1, :]) + self.W_J(g_final)
        
        return out