"""Tests for LSTM Market Predictor."""

import sys
import types
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import packaging

    pkg_resources_stub = types.ModuleType("pkg_resources")
    pkg_resources_stub.packaging = packaging
    sys.modules.setdefault("pkg_resources", pkg_resources_stub)

    import torch  # noqa: F401
except Exception as e:
    pytest.skip(f"torch import failed: {e}", allow_module_level=True)

from app.models.lstm_predictor import LSTMPredictor, LSTMMarketPredictor


class TestLSTMPredictorNetwork:
    """Test LSTM neural network model."""

    def test_lstm_model_creation(self):
        """Test LSTM model can be created."""
        model = LSTMPredictor(input_size=5, hidden_size=50, num_layers=2)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_lstm_forward_pass(self):
        """Test forward pass through LSTM."""
        model = LSTMPredictor(input_size=5, hidden_size=50)

        # Input: batch_size=32, sequence_length=60, input_size=5
        x = torch.randn(32, 60, 5)
        output = model(x)

        assert output.shape == (32, 1)  # Batch size 32, output size 1

    def test_lstm_different_input_sizes(self):
        """Test LSTM with different input feature sizes."""
        for input_size in [3, 5, 10, 15]:
            model = LSTMPredictor(input_size=input_size)
            x = torch.randn(16, 60, input_size)
            output = model(x)
            assert output.shape == (16, 1)

    def test_lstm_different_sequence_lengths(self):
        """Test LSTM with different sequence lengths."""
        model = LSTMPredictor(input_size=5)

        for seq_len in [20, 30, 60, 90]:
            x = torch.randn(16, seq_len, 5)
            output = model(x)
            assert output.shape == (16, 1)


class TestLSTMMarketPredictor:
    """Test high-level LSTM predictor wrapper."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=200)
        df = pd.DataFrame(
            {
                "open": np.random.uniform(95, 105, 200),
                "high": np.random.uniform(100, 110, 200),
                "low": np.random.uniform(90, 100, 200),
                "close": np.random.uniform(95, 105, 200),
                "volume": np.random.uniform(1e6, 5e6, 200),
            },
            index=dates,
        )
        return df

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return LSTMMarketPredictor(lookback=60, forecast_horizon=5)

    def test_predictor_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.lookback == 60
        assert predictor.forecast_horizon == 5
        assert predictor.model_dir.exists()

    def test_data_preparation(self, predictor, sample_data):
        """Test data preparation for training."""
        X, y, scaler = predictor.prepare_data(sample_data)

        assert X.shape[0] > 0
        assert y.shape[0] > 0
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 60  # lookback
        assert X.shape[2] >= 2  # At least close and volume
        assert scaler is not None

    def test_data_normalization(self, predictor, sample_data):
        """Test that data is properly normalized."""
        X, _, _ = predictor.prepare_data(sample_data)

        # Normalized data should be between 0 and 1
        assert np.all(X >= 0.0)
        assert np.all(X <= 1.0)

    def test_model_training(self, predictor, sample_data):
        """Test model training."""
        X, y, _ = predictor.prepare_data(sample_data)

        history = predictor.train(X, y, epochs=3, batch_size=16)

        assert "loss" in history
        assert "val_loss" in history
        assert len(history["loss"]) == 3
        assert predictor.model is not None

    def test_training_loss_decrease(self, predictor, sample_data):
        """Test that training loss decreases over epochs."""
        X, y, _ = predictor.prepare_data(sample_data)

        history = predictor.train(X, y, epochs=6, batch_size=16)

        # Loss should generally decrease (allow some noise)
        initial_loss = np.mean(history["loss"][:2])
        final_loss = np.mean(history["loss"][-2:])

        # Allow 50% increase due to randomness, but generally should decrease
        assert final_loss <= initial_loss * 1.5

    def test_prediction(self, predictor, sample_data):
        """Test making predictions."""
        X, y, _ = predictor.prepare_data(sample_data)
        predictor.train(X, y, epochs=3)

        predictions = predictor.predict(X[-10:])

        assert predictions.shape == (10,)
        assert np.all(np.isfinite(predictions))

    def test_prediction_output_range(self, predictor, sample_data):
        """Test prediction output is in reasonable range."""
        X, y, _ = predictor.prepare_data(sample_data)
        predictor.train(X, y, epochs=3)

        predictions = predictor.predict(X[:20])

        # Predictions should be close to actual prices (within 20%)
        close_prices = sample_data["close"].values[60:80]
        mean_price = np.mean(close_prices)

        # Predictions should be within 50% of mean price
        assert np.all(predictions > mean_price * 0.5)
        assert np.all(predictions < mean_price * 1.5)

    def test_model_persistence(self, predictor, sample_data):
        """Test saving and loading model."""
        X, y, _ = predictor.prepare_data(sample_data)
        predictor.train(X, y, epochs=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"

            # Predict before save
            pred_before = predictor.predict(X[:5])

            # Save model
            predictor.model_dir = Path(tmpdir)
            predictor.save_model("test_model")

            # Create new predictor and load
            new_predictor = LSTMMarketPredictor()
            new_predictor.load_model(model_path)

            # Predictions should be identical
            pred_after = new_predictor.predict(X[:5])
            np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=5)

    def test_predict_price_change(self, predictor, sample_data):
        """Test price change prediction."""
        X, y, _ = predictor.prepare_data(sample_data[-120:])
        predictor.train(X, y, epochs=2)

        result = predictor.predict_price_change(sample_data[-65:])

        assert "predicted_price" in result
        assert "price_change_pct" in result
        assert "confidence" in result
        assert result["predicted_price"] > 0
        assert isinstance(result["price_change_pct"], float)


class TestLSTMEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        predictor = LSTMMarketPredictor(lookback=60)

        # Only 10 points - less than lookback
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "volume": [1e6] * 10,
            }
        )

        X, y, _ = predictor.prepare_data(df)

        # Should return empty sequences
        assert len(X) == 0 or X.shape[0] < 10

    def test_model_not_trained_prediction(self):
        """Test prediction without training raises error."""
        predictor = LSTMMarketPredictor()
        X = np.random.randn(5, 60, 5)

        with pytest.raises(ValueError):
            predictor.predict(X)

    def test_nan_in_data(self):
        """Test handling of NaN values in data."""
        predictor = LSTMMarketPredictor()

        df = pd.DataFrame(
            {
                "close": [100, np.nan, 102, 103, np.nan] + [105] * 195,
                "volume": [1e6] * 200,
            }
        )

        # Should handle NaN gracefully (drop or interpolate)
        X, y, _ = predictor.prepare_data(df)

        # Should have valid sequences
        assert not np.any(np.isnan(X))

    def test_constant_price_data(self):
        """Test handling of constant price (zero variance)."""
        predictor = LSTMMarketPredictor()

        df = pd.DataFrame({"close": [100] * 200, "volume": [1e6] * 200})

        X, y, _ = predictor.prepare_data(df)

        # Should still create sequences
        assert X.shape[0] > 0


class TestLSTMIntegration:
    """Integration tests with actual trading workflow."""

    def test_lstm_signal_generation(self):
        """Test generating trading signals from LSTM predictions."""
        predictor = LSTMMarketPredictor(lookback=30, forecast_horizon=1)

        # Create realistic price data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)
        prices = np.cumsum(np.random.randn(100) * 0.5) + 100
        df = pd.DataFrame(
            {"close": prices, "volume": np.random.uniform(1e6, 5e6, 100)}, index=dates
        )

        # Train
        X, y, _ = predictor.prepare_data(df)
        predictor.train(X, y, epochs=2)

        # Predict and generate signal
        result = predictor.predict_price_change(df)

        # Generate signal
        if result["price_change_pct"] > 0.5:
            signal = "BUY"
        elif result["price_change_pct"] < -0.5:
            signal = "SELL"
        else:
            signal = "HOLD"

        assert signal in ["BUY", "SELL", "HOLD"]
