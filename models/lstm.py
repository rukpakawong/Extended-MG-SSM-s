import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the LSTM architecture

        Arg:
            input_size: the number of features in the input data.
            hidden_size: the number of features in the hidden state.
            num_layers: the number of LSTM layers.
            output_size: the dimensionality of the final output.
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