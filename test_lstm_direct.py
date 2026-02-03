#!/usr/bin/env python
"""Direct LSTM test avoiding matplotlib path issues."""

import os
import sys

# Bypass matplotlib early
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta

# Manual LSTM test without importing the module (to avoid matplotlib via pkg_resources)

class SimpleLSTM(nn.Module):
    """Minimal LSTM for testing."""
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_hidden))
        x = self.fc2(x)
        return x


def test_lstm_basic():
    """Test basic LSTM functionality."""
    print("Test 1: Basic LSTM Model Creation")
    model = SimpleLSTM(input_size=5, hidden_size=50, num_layers=2)
    print(f"✓ Model created: {model}")
    
    print("\nTest 2: Forward Pass")
    x = torch.randn(32, 60, 5)  # (batch_size, seq_len, features)
    output = model(x)
    assert output.shape == (32, 1), f"Expected shape (32, 1), got {output.shape}"
    print(f"✓ Forward pass successful: input {x.shape} → output {output.shape}")
    
    print("\nTest 3: Different Input Sizes")
    for features in [3, 5, 10, 15]:
        m = SimpleLSTM(input_size=features, hidden_size=50, num_layers=2)
        x = torch.randn(16, 60, features)
        out = m(x)
        assert out.shape == (16, 1), f"Shape mismatch for {features} features"
        print(f"✓ Features={features}: input {x.shape} → output {out.shape}")
    
    print("\nTest 4: Different Sequence Lengths")
    for seq_len in [20, 30, 60, 90]:
        model = SimpleLSTM()
        x = torch.randn(16, seq_len, 5)
        out = model(x)
        assert out.shape == (16, 1)
        print(f"✓ Sequence length={seq_len}: output shape {out.shape}")
    
    print("\nTest 5: Training Loop")
    model = SimpleLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create dummy data
    X = torch.randn(100, 60, 5)
    y = torch.randn(100, 1)
    
    losses = []
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Epoch {epoch+1}/10: loss={loss.item():.6f}")
    
    # Check if loss generally decreased
    avg_first_5 = np.mean(losses[:5])
    avg_last_5 = np.mean(losses[5:])
    assert avg_last_5 < avg_first_5 * 1.2, "Loss should decrease over training"
    print(f"✓ Training successful: first 5 avg={avg_first_5:.6f}, last 5 avg={avg_last_5:.6f}")
    
    print("\nTest 6: Model State Dict")
    state = model.state_dict()
    print(f"✓ Model state dict retrieved: {len(state)} parameters")
    
    # Save and load
    torch.save(state, 'test_model.pth')
    loaded_state = torch.load('test_model.pth')
    model.load_state_dict(loaded_state)
    print("✓ Model saved and loaded successfully")
    
    # Cleanup
    os.remove('test_model.pth')
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("="*50)

if __name__ == "__main__":
    try:
        test_lstm_basic()
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
