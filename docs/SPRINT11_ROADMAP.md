# Sprint 11+ Roadmap - Átfogó Terv

**Dokumentum:** Sprint 11+ fejlesztési terv  
**Létrehozva:** 2026-02-03  
**Frissítve:** 2026-02-03 (Opció 1 - Coverage javítás - ✅ COMPLETED)
**Jelenlegi állapot:** Sprint 11b teljesítve (709 teszt, 87% coverage)  
**Cél:** Production-ready kódbázis 90%+ coverage céllal

---

## 📊 Jelenlegi Helyzet (Sprint 11b után) ✅

| Metrika | Érték | Változás | Státusz |
|---------|-------|----------|---------|
| **Tesztek** | 709 passed, 1 skipped | +84 új teszt | ✅ |
| **Coverage** | 87% | +4% (83%→87%) | ✅ |
| **100% Coverage Modulok** | 3 modul | walk_forward, decision_engine, genetic_optimizer | ✅ |
| **95%+ Coverage Modulok** | 8 modul | rebalancer (99%), PA (99%), backtester (95%), stb. | ✅ |
| **Failing Tests** | 0 | Clean suite! | ✅ |
| **Dokumentáció** | Konszolidált + Sprint 11b docs | 8 fájl | ✅ |
| **Production Ready** | Igen | 87% coverage, 0 failures | 🟢✅ |

### Sprint 11b Coverage Eredmények:

#### Modul-specifikus Javulások:
1. **Backup Manager**: 59%→91% (+32 pont, 37 új teszt)
2. **Performance Analytics**: 71%→99% (+28 pont, 32 új teszt)  
3. **Genetic Optimizer**: 66%→100% (+34 pont, 15 új teszt)

#### Top Coverage Modulok:
- ✅ **100%**: genetic_optimizer, walk_forward, decision_engine, data_cleaner, risk_parity, recommendation_builder
- ✅ **99%**: rebalancer, performance_analytics
- ✅ **97%**: decision_builder
- ✅ **95%**: backtester, metrics (reporting)
- ✅ **94%**: backtest_audit, config
- ✅ **93%**: adaptive_strategy_selector, confidence_allocator, fitness

---

## 🎯 Opció 1: Test Coverage Növelése (83% → 87%) ✅ COMPLETED

### Cél
Robusztus, production-ready kódbázis 90%+ coverage-el és nulla failing test-tel.

### ✅ MEGVALÓSÍTOTT (2026-02-03)

#### 1.1 Failing Tests Javítása ✅ KÉSZ
**File:** `tests/test_portfolio_correlation_manager.py`  
**Eredmény:** Öss
1. **Correlation fixture létrehozása**
   ```python
   # tests/fixtures/correlation_fixtures.py
   @pytest.fixture
   def sample_correlation_matrix():
       return {
           ('AAPL', 'MSFT'): 0.65,
           ('AAPL', 'GOOGL'): 0.58,
           ('MSFT', 'GOOGL'): 0.72,
           ('AAPL', 'GOLD'): -0.15,
           ('MSFT', 'GOLD'): -0.20,
           ('GOOGL', 'GOLD'): -0.10,
       }
   
   @pytest.fixture
   def test_db_with_correlations(test_db, sample_correlation_matrix):
       # Populate test_db with correlation data
       cursor = test_db.cursor()
       for (ticker1, ticker2), corr in sample_correlation_matrix.items():
           cursor.execute(
               "INSERT INTO correlations VALUES (?, ?, ?, ?)",
               (ticker1, ticker2, corr, '2026-02-01')
           )
       test_db.commit()
       return test_db
   ```

2. **Tesztek átírása új fixture használatára**
   ```python
   def test_calculate_diversification_score(test_db_with_correlations):
       manager = PortfolioCorrelationManager(test_db_with_correlations)
       # Most már vannak correlation adatok
       score = manager.calculate_diversification_score(['AAPL', 'GOLD'])
       assert score > 0  # Should pass now
   ```

3. **Verify**
   ```bash
   pytest tests/test_portfolio_correlation_manager.py -v
   # Elvárás: 8/8 passed
   ```

**Időigény:** 4-6 óra

---

#### 1.2 Alacsony Coverage Modulok Tesztelése

**Cél modulok (<80% coverage):**

##### A. `app/reporting/metrics.py` (~65% coverage)
**Hiányzó tesztek:**
- `calculate_win_rate()` - edge cases (0 trades, all wins, all losses)
- `calculate_profit_factor()` - division by zero handling
- `calculate_recovery_factor()` - max drawdown = 0 case
- `generate_monthly_report()` - empty data handling

**Új tesztek:**
```python
# tests/test_reporting_metrics_edge.py
def test_win_rate_no_trades():
    assert calculate_win_rate([]) == 0.0

def test_win_rate_all_wins():
    trades = [Trade(profit=100), Trade(profit=50)]
    assert calculate_win_rate(trades) == 1.0

def test_profit_factor_no_losses():
    trades = [Trade(profit=100), Trade(profit=50)]
    # Should handle division by zero
    assert calculate_profit_factor(trades) == float('inf')
```

**Időigény:** 3-4 óra

---

##### B. `app/analysis/performance_analytics.py` (~70% coverage)
**Hiányzó tesztek:**
- `calculate_sharpe_ratio()` - std dev = 0 case
- `calculate_sortino_ratio()` - downside deviation = 0
- `calculate_calmar_ratio()` - max drawdown = 0
- `analyze_trade_distribution()` - extreme outliers

**Új tesztek:**
```python
# tests/test_performance_analytics_robust.py
def test_sharpe_zero_volatility():
    returns = [0.05, 0.05, 0.05]  # Constant returns
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0 or math.isinf(sharpe)

def test_calmar_no_drawdown():
    returns = [0.01, 0.02, 0.03]  # Only positive
    calmar = calculate_calmar_ratio(returns)
    assert calmar > 0
```

**Időigény:** 3-4 óra

---

##### C. `app/decision/capital_optimizer.py` (~72% coverage)
**Hiányzó tesztek:**
- `optimize_kelly()` - negative win rate
- `optimize_kelly()` - win rate = 1.0
- `calculate_optimal_allocation()` - single asset
- `rebalance_portfolio()` - extreme weights

**Új tesztek:**
```python
# tests/test_capital_optimizer_boundaries.py
def test_kelly_negative_win_rate():
    kelly = optimize_kelly(win_rate=-0.1, win_loss_ratio=2.0)
    assert kelly == 0.0  # Should not bet

def test_kelly_perfect_win_rate():
    kelly = optimize_kelly(win_rate=1.0, win_loss_ratio=1.5)
    assert 0 <= kelly <= 1.0  # Should be bounded
```

**Időigény:** 3-4 óra

---

##### D. `app/infrastructure/log_manager.py` (~75% coverage)
**Hiányzó tesztek:**
- `rotate_logs()` - disk full scenario
- `compress_old_logs()` - corrupted file handling
- `cleanup_logs()` - permission errors

**Új tesztek:**
```python
# tests/test_log_manager_failures.py
def test_rotate_logs_disk_full(mocker):
    mocker.patch('shutil.move', side_effect=OSError("No space"))
    manager = LogManager()
    # Should handle gracefully
    manager.rotate_logs()  # Should not crash

def test_compress_corrupted_file(tmp_path):
    corrupted_log = tmp_path / "corrupted.log"
    corrupted_log.write_bytes(b'\x00\x01\x02')  # Invalid data
    manager = LogManager(log_dir=tmp_path)
    manager.compress_old_logs()  # Should skip corrupted files
```

**Időigény:** 2-3 óra

---

#### 1.3 Integration Tests Bővítése

**Új end-to-end tesztek:**

```python
# tests/test_full_trading_cycle.py
def test_complete_trading_workflow():
    """Test full cycle: data → analysis → decision → execution → reporting"""
    # 1. Load market data
    data_manager = DataManager()
    data_manager.load_ohlcv('AAPL', '2025-01-01', '2025-12-31')
    
    # 2. Run analysis
    analyzer = TechnicalAnalyzer()
    signals = analyzer.generate_signals('AAPL')
    
    # 3. Make decision
    engine = DecisionEngine()
    decision = engine.make_decision('AAPL', signals)
    
    # 4. Execute (simulated)
    result = execute_decision(decision)
    
    # 5. Report
    report = generate_trade_report(result)
    
    assert report['success'] == True
    assert 'profit' in report
```

**Időigény:** 4-5 óra

---

### Összesített Időbecslés (Opció 1)

| Feladat | Időigény | Prioritás |
|---------|----------|-----------|
| Failing tests javítása | 4-6 óra | KRITIKUS |
| Metrics tesztelése | 3-4 óra | MAGAS |
| Performance analytics | 3-4 óra | MAGAS |
| Capital optimizer | 3-4 óra | MAGAS |
| Log manager | 2-3 óra | KÖZEPES |
| Integration tests | 4-5 óra | KÖZEPES |
| **ÖSSZESEN** | **19-26 óra** | **~3-4 nap** |

### Várható Eredmény
- ✅ 0 failing tests
- ✅ 90%+ code coverage
- ✅ Robusztus edge case handling
- ✅ Production-ready confidence

---

## 🔧 Opció 2: Raspberry Pi Deployment

### Cél
Teljes hardware deployment és production monitoring Raspberry Pi 4/5-ön.

### Előfeltételek
- ✅ `deploy_rpi.sh` script kész
- ✅ `pi_config.py` konfigurációs fájl kész
- ✅ SystemD service template kész
- ⏳ Raspberry Pi hardver (még nem érkezett meg)

### Részletes Feladatok

#### 2.1 Hardware Setup (1-2 óra)
1. **Raspberry Pi OS telepítése**
   ```bash
   # Raspberry Pi Imager használata
   # OS: Raspberry Pi OS (64-bit) Lite
   # Hostname: tozsde-pi
   # SSH engedélyezése
   # WiFi konfiguráció
   ```

2. **Alapvető rendszer konfiguráció**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install python3-pip python3-venv git -y
   
   # Timezone beállítás
   sudo timedatectl set-timezone Europe/Budapest
   ```

#### 2.2 Alkalmazás Telepítése (1-2 óra)
1. **Projekt klónozása**
   ```bash
   cd /home/pi
   git clone <repository_url> tozsde_webapp
   cd tozsde_webapp
   ```

2. **Deploy script futtatása**
   ```bash
   chmod +x deploy_rpi.sh
   ./deploy_rpi.sh
   ```
   
   **A script végzi:**
   - Virtual environment létrehozása
   - Dependencies telepítése (`requirements.txt`)
   - Database inicializálása
   - Log könyvtárak létrehozása
   - SystemD service regisztrálása

3. **Konfigurációs fájl testreszabása**
   ```bash
   nano app/config/pi_config.py
   # Ellenőrizni:
   # - Database elérési út
   # - Log könyvtár
   # - Port (default: 5000)
   # - Debug mode (False production-ban)
   ```

#### 2.3 SystemD Service Konfiguráció (30 perc)
```bash
# Service fájl már létezik: /etc/systemd/system/tozsde-webapp.service
sudo systemctl daemon-reload
sudo systemctl enable tozsde-webapp
sudo systemctl start tozsde-webapp
sudo systemctl status tozsde-webapp
```

**Service fájl tartalma:**
```ini
[Unit]
Description=ToZsDE Trading System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/tozsde_webapp
Environment="PATH=/home/pi/tozsde_webapp/venv/bin"
ExecStart=/home/pi/tozsde_webapp/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 2.4 Monitoring & Health Check (1 óra)
1. **Automatikus health check cron job**
   ```bash
   crontab -e
   # Add:
   */5 * * * * curl -f http://localhost:5000/admin/health || systemctl restart tozsde-webapp
   ```

2. **Log rotation konfiguráció**
   ```bash
   sudo nano /etc/logrotate.d/tozsde-webapp
   ```
   
   ```
   /home/pi/tozsde_webapp/app/logs/*.log {
       daily
       rotate 7
       compress
       delaycompress
       missingok
       notifempty
   }
   ```

3. **Disk space monitoring**
   ```bash
   # Add to crontab
   0 0 * * * df -h | grep -E '9[0-9]%|100%' && mail -s "Disk full" admin@example.com
   ```

#### 2.5 Network & Security (1-2 óra)
1. **Firewall konfiguráció**
   ```bash
   sudo ufw allow 22/tcp  # SSH
   sudo ufw allow 5000/tcp  # Flask app
   sudo ufw enable
   ```

2. **SSH key-based authentication**
   ```bash
   # Local machine:
   ssh-keygen -t ed25519 -C "tozsde-pi"
   ssh-copy-id pi@tozsde-pi.local
   
   # Disable password auth (optional):
   sudo nano /etc/ssh/sshd_config
   # PasswordAuthentication no
   sudo systemctl restart ssh
   ```

3. **Nginx reverse proxy (optional, ajánlott)**
   ```bash
   sudo apt install nginx -y
   sudo nano /etc/nginx/sites-available/tozsde
   ```
   
   ```nginx
   server {
       listen 80;
       server_name tozsde-pi.local;
       
       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```
   
   ```bash
   sudo ln -s /etc/nginx/sites-available/tozsde /etc/nginx/sites-enabled/
   sudo systemctl restart nginx
   ```

#### 2.6 Testing & Validation (1-2 óra)
1. **Functional tests**
   ```bash
   # On Raspberry Pi
   cd /home/pi/tozsde_webapp
   source venv/bin/activate
   pytest tests/ -v
   # Expected: 625 passed, 1 skipped
   ```

2. **API endpoint tests**
   ```bash
   curl http://localhost:5000/admin/health
   curl http://localhost:5000/admin/metrics
   curl http://localhost:5000/api/decision/AAPL
   ```

3. **Performance baseline**
   ```bash
   # Test backtesting performance
   time python -c "from app.backtesting.backtester import Backtester; b = Backtester(); b.run('2025-01-01', '2025-12-31')"
   # Record execution time for future comparison
   ```

#### 2.7 Documentation (1 óra)
- Update `docs/deployment/RASPBERRY_PI_SETUP_GUIDE.md` with actual deployment results
- Document any issues encountered
- Create deployment checklist (`docs/deployment/DEPLOYMENT_CHECKLIST.md`)

### Összesített Időbecslés (Opció 2)

| Feladat | Időigény | Függőség |
|---------|----------|----------|
| Hardware setup | 1-2 óra | Hardware megérkezése |
| Alkalmazás telepítés | 1-2 óra | Hardware setup |
| SystemD service | 30 perc | Alkalmazás telepítés |
| Monitoring setup | 1 óra | Service running |
| Network & security | 1-2 óra | Hardware setup |
| Testing & validation | 1-2 óra | Minden működik |
| Documentation | 1 óra | Testing kész |
| **ÖSSZESEN** | **6.5-10.5 óra** | **~1-2 nap** |

### Várható Eredmény
- ✅ Production system futás Raspberry Pi-on
- ✅ Auto-restart on failure
- ✅ Log rotation & monitoring
- ✅ Secure remote access
- ✅ Performance baseline dokumentálva

---

## 🚀 Opció 3: Új Funkciók - Sprint 11

### Cél
Fejlett ML modellek, real-time adatok, advanced portfolio technikák.

### 3.1 Machine Learning Továbbfejlesztés

#### LSTM Modellek Implementálása (5-7 nap)
**Modul:** `app/models/lstm_predictor.py`

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMPredictor:
    """Long Short-Term Memory model for price prediction."""
    
    def __init__(self, lookback=60, features=5):
        self.lookback = lookback
        self.features = features
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback, self.features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)  # Price prediction
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history
    
    def predict(self, X_test):
        return self.model.predict(X_test)
```

**Feladatok:**
1. Feature engineering (OHLCV + technical indicators)
2. Data normalization/scaling
3. Train/test split with time-based validation
4. Model training & hyperparameter tuning
5. Prediction integration into decision engine
6. Backtesting with LSTM signals
7. Model versioning & persistence

**Dependencies:**
```bash
pip install tensorflow numpy scikit-learn
```

**Tesztek:**
```python
# tests/test_lstm_predictor.py
def test_lstm_model_creation():
    predictor = LSTMPredictor(lookback=30, features=5)
    assert predictor.model is not None

def test_lstm_training():
    X_train = np.random.rand(1000, 30, 5)
    y_train = np.random.rand(1000, 1)
    predictor = LSTMPredictor()
    history = predictor.train(X_train, y_train, epochs=2)
    assert 'loss' in history.history

def test_lstm_prediction():
    X_test = np.random.rand(100, 30, 5)
    predictor = LSTMPredictor()
    predictions = predictor.predict(X_test)
    assert predictions.shape == (100, 1)
```

**Időigény:** 35-45 óra (5-7 nap)

---

#### Ensemble Learning (3-4 nap)
**Modul:** `app/models/ensemble_learner.py`

**Algoritmusok:**
- Random Forest Classifier (trend up/down/neutral)
- XGBoost Regressor (price change prediction)
- Gradient Boosting (confidence scoring)

**Meta-model:** Voting/Stacking ensemble

**Feladatok:**
1. Implement RF, XGBoost, GBM models
2. Feature importance analysis
3. Model stacking/voting logic
4. Cross-validation framework
5. Performance comparison (LSTM vs Ensemble vs Traditional)

**Időigény:** 20-30 óra (3-4 nap)

---

### 3.2 Real-time Adatforrás Integráció (4-6 nap)

#### Broker API Integration
**Target:** Interactive Brokers, Alpaca, vagy Binance API

**Modul:** `app/data_access/broker_client.py`

```python
class BrokerClient:
    """Real-time market data and order execution."""
    
    def __init__(self, api_key, api_secret, broker='alpaca'):
        self.broker = broker
        self.client = self._init_client(api_key, api_secret)
    
    def get_realtime_quote(self, symbol):
        """Get current bid/ask/last price."""
        pass
    
    def get_historical_data(self, symbol, start, end, timeframe='1D'):
        """Fetch OHLCV data."""
        pass
    
    def place_order(self, symbol, quantity, side, order_type='market'):
        """Execute trade order."""
        pass
    
    def get_account_info(self):
        """Get account balance, positions, buying power."""
        pass
    
    def stream_quotes(self, symbols, callback):
        """WebSocket real-time streaming."""
        pass
```

**Feladatok:**
1. API authentication & connection
2. Rate limiting & error handling
3. Data format normalization
4. WebSocket streaming implementation
5. Order execution logic
6. Paper trading mode (sandbox)
7. Integration with decision engine

**Időigény:** 30-40 óra (4-6 nap)

---

### 3.3 Advanced Portfolio Optimization (3-5 nap)

#### Black-Litterman Model
**Modul:** `app/optimization/black_litterman.py`

**Koncepció:**
- Combine market equilibrium (CAPM) with investor views
- Bayesian approach to portfolio optimization
- Incorporate subjective opinions into allocation

```python
class BlackLittermanOptimizer:
    def __init__(self, market_caps, risk_aversion=2.5):
        self.market_caps = market_caps
        self.risk_aversion = risk_aversion
    
    def calculate_equilibrium_returns(self, cov_matrix):
        """Pi = lambda * Sigma * w_mkt"""
        pass
    
    def incorporate_views(self, P, Q, Omega):
        """
        P: View matrix (which assets)
        Q: View returns (expected returns)
        Omega: View uncertainty
        """
        pass
    
    def optimize(self, views):
        """Return optimal allocation based on views."""
        pass
```

**Időigény:** 20-30 óra (3-4 nap)

---

#### CVaR Optimization (2-3 nap)
**Modul:** `app/optimization/cvar_optimizer.py`

**Conditional Value at Risk:**
- Focus on tail risk (worst-case scenarios)
- Optimize for downside protection
- Alternative to mean-variance optimization

```python
class CVaROptimizer:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
    
    def calculate_cvar(self, returns, weights):
        """Calculate CVaR at specified confidence level."""
        pass
    
    def optimize_portfolio(self, returns, target_return=None):
        """Minimize CVaR subject to return constraint."""
        pass
```

**Időigény:** 15-20 óra (2-3 nap)

---

### 3.4 Dashboard Továbbfejlesztés (3-4 nap)

#### Real-time Charts
**Framework:** Plotly Dash vagy Chart.js + WebSockets

**Features:**
- Live price updates (streaming)
- Interactive technical indicator overlays
- Portfolio value real-time tracking
- Trade execution visualization

**Modul:** `app/ui/realtime_dashboard.py`

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-price-chart'),
    dcc.Interval(id='interval-component', interval=1000)  # Update every 1s
])

@app.callback(
    Output('live-price-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_chart(n):
    # Fetch latest data
    # Update figure
    pass
```

**Időigény:** 20-25 óra (3-4 nap)

---

#### Alert System
**Modul:** `app/notifications/alert_manager.py`

**Alert types:**
- Price threshold breaches
- Technical signal triggers (RSI overbought/oversold)
- Portfolio drawdown warnings
- Trade execution confirmations

**Channels:**
- Email (SMTP)
- Telegram bot
- Push notifications (optional)

**Időigény:** 10-15 óra (1-2 nap)

---

### Összesített Időbecslés (Opció 3)

| Komponens | Időigény | Prioritás |
|-----------|----------|-----------|
| LSTM modellek | 35-45 óra | MAGAS |
| Ensemble learning | 20-30 óra | KÖZEPES |
| Broker API | 30-40 óra | MAGAS |
| Black-Litterman | 20-30 óra | KÖZEPES |
| CVaR optimization | 15-20 óra | ALACSONY |
| Real-time dashboard | 20-25 óra | KÖZEPES |
| Alert system | 10-15 óra | ALACSONY |
| **ÖSSZESEN** | **150-205 óra** | **~4-5 hét** |

### Javasolt Fázisozás
- **Sprint 11:** LSTM + Broker API integration (8-10 nap)
- **Sprint 12:** Ensemble learning + Real-time dashboard (6-8 nap)
- **Sprint 13:** Black-Litterman + Alert system (5-7 nap)

---

## ⚡ Opció 4: Performance Optimalizáció

### Cél
Gyorsabb backtesting, adatbázis lekérdezések, hatékonyabb számítások.

### 4.1 Backtesting Gyorsítás (2-3 nap)

#### Multi-threading/Multiprocessing
**Modul:** `app/backtesting/parallel_backtester.py`

**Stratégia:**
- Párhuzamosítás instrumentumonként (AAPL, MSFT parallel)
- Párhuzamosítás időszakonként (rolling windows)
- Process pool executor használata

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from app.backtesting.backtester import Backtester

class ParallelBacktester:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
    
    def run_parallel(self, symbols, start_date, end_date):
        """Run backtests in parallel for multiple symbols."""
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_single, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            results = {}
            for future in as_completed(futures):
                symbol = futures[future]
                results[symbol] = future.result()
            
            return results
    
    def _run_single(self, symbol, start_date, end_date):
        """Run backtest for single symbol."""
        backtester = Backtester()
        return backtester.run(symbol, start_date, end_date)
```

**Benchmark:**
- **Előtte:** 10 symbol × 1 év = 120 sec
- **Utána (4 core):** 10 symbol × 1 év = 35-40 sec (~3x speedup)

**Feladatok:**
1. Implement parallel backtester
2. Handle shared state (database connections)
3. Progress reporting
4. Error handling per worker
5. Results aggregation
6. Memory optimization

**Időigény:** 15-20 óra (2-3 nap)

---

### 4.2 Database Query Optimalizálás (1-2 nap)

#### Indexek Létrehozása
```sql
-- app/data_access/schema_indexes.sql
CREATE INDEX idx_ohlcv_ticker_date ON ohlcv_data(ticker, date);
CREATE INDEX idx_decisions_date ON decisions(timestamp);
CREATE INDEX idx_correlations_tickers ON correlations(ticker1, ticker2);
CREATE INDEX idx_trades_symbol_date ON trades(symbol, execution_date);
```

#### Query Optimization
```python
# Before (slow):
cursor.execute("SELECT * FROM ohlcv_data WHERE ticker = ?", (ticker,))
data = cursor.fetchall()

# After (fast):
cursor.execute("""
    SELECT date, open, high, low, close, volume 
    FROM ohlcv_data 
    WHERE ticker = ? AND date BETWEEN ? AND ?
    ORDER BY date
""", (ticker, start_date, end_date))
data = cursor.fetchall()
```

#### Connection Pooling
```python
from contextlib import contextmanager
import sqlite3

class DatabasePool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool = [sqlite3.connect(db_path) for _ in range(pool_size)]
        self.available = list(self.pool)
    
    @contextmanager
    def get_connection(self):
        conn = self.available.pop()
        try:
            yield conn
        finally:
            self.available.append(conn)
```

**Benchmark:**
- **Előtte:** 1000 queries = 15 sec
- **Utána:** 1000 queries = 3-4 sec (~4x speedup)

**Időigény:** 8-12 óra (1-2 nap)

---

### 4.3 Caching Stratégiák (1-2 nap)

#### Redis Cache Integration
```python
import redis
import pickle

class CacheManager:
    def __init__(self, host='localhost', port=6379):
        self.redis_client = redis.Redis(host=host, port=port)
    
    def cache_ohlcv(self, ticker, start, end, data, ttl=3600):
        """Cache OHLCV data for 1 hour."""
        key = f"ohlcv:{ticker}:{start}:{end}"
        self.redis_client.setex(key, ttl, pickle.dumps(data))
    
    def get_cached_ohlcv(self, ticker, start, end):
        """Retrieve cached data."""
        key = f"ohlcv:{ticker}:{start}:{end}"
        cached = self.redis_client.get(key)
        return pickle.loads(cached) if cached else None
```

#### Decorator-based Caching
```python
from functools import wraps
import hashlib
import json

def cache_result(ttl=300):
    """Decorator for caching function results."""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = hashlib.md5(
                json.dumps([args, kwargs], sort_keys=True).encode()
            ).hexdigest()
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            return result
        
        return wrapper
    return decorator

# Usage:
@cache_result(ttl=600)
def calculate_technical_indicators(ticker, start, end):
    # Expensive calculation
    pass
```

**Benchmark:**
- **Cached indicator calculation:** <1ms vs 100-500ms
- **Cached OHLCV fetch:** <5ms vs 50-100ms

**Időigény:** 10-15 óra (1-2 nap)

---

### 4.4 NumPy Vectorization (1 nap)

#### Before (slow loops):
```python
def calculate_sma(prices, window):
    sma = []
    for i in range(len(prices)):
        if i < window - 1:
            sma.append(None)
        else:
            sma.append(sum(prices[i-window+1:i+1]) / window)
    return sma
```

#### After (vectorized):
```python
import numpy as np

def calculate_sma(prices, window):
    prices_array = np.array(prices)
    return np.convolve(prices_array, np.ones(window)/window, mode='valid')
```

**Speedup:** 10-50x for large datasets

**Feladatok:**
1. Audit all indicator calculations
2. Replace loops with NumPy operations
3. Use pandas rolling windows where applicable
4. Benchmark improvements

**Időigény:** 6-8 óra (1 nap)

---

### Összesített Időbecslés (Opció 4)

| Komponens | Időigény | Várható Speedup |
|-----------|----------|-----------------|
| Parallel backtesting | 15-20 óra | 3-4x |
| DB query optimization | 8-12 óra | 3-5x |
| Caching | 10-15 óra | 10-100x (repeated) |
| NumPy vectorization | 6-8 óra | 10-50x |
| **ÖSSZESEN** | **39-55 óra** | **~1 hét** |

### Várható Eredmény
- ✅ 3-4x gyorsabb backtesting
- ✅ Jelentősen gyorsabb API válaszidők
- ✅ Alacsonyabb CPU/memory használat
- ✅ Skálázhatóság javulás

---

## 🧹 Opció 5: Refactoring & Tech Debt

### Cél
Tisztább, karbantarthatóbb kódbázis, modern best practices.

### 5.1 Type Hints Bővítése (2-3 nap)

#### Current State
```python
def calculate_sharpe(returns, risk_free_rate):
    # No type hints
    excess = [r - risk_free_rate for r in returns]
    return mean(excess) / std(excess)
```

#### Target State
```python
from typing import List, Optional, Union
import numpy as np

def calculate_sharpe(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.02
) -> Optional[float]:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Sharpe ratio, or None if insufficient data
    """
    if len(returns) < 2:
        return None
    
    excess = returns - risk_free_rate
    return np.mean(excess) / np.std(excess)
```

**Feladatok:**
1. Add type hints to all functions (600+ functions)
2. Use `mypy` for type checking
3. Document complex types (TypedDict, Protocol)
4. Fix type errors

**Tools:**
```bash
pip install mypy
mypy app/ --strict
```

**Időigény:** 15-20 óra (2-3 nap)

---

### 5.2 Deprecation Warnings Javítása (1 nap)

**Known issues:**
- `np.float` → `np.float64`
- `datetime.utcnow()` → `datetime.now(timezone.utc)`
- Old pandas methods → new alternatives

```python
# Before:
df['returns'] = df['close'].pct_change().fillna(0)
result = df.append(new_row)  # DEPRECATED

# After:
df['returns'] = df['close'].pct_change().fillna(0)
result = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
```

**Feladatok:**
1. Run with warnings enabled: `pytest -W error`
2. Fix all deprecation warnings
3. Update to latest API versions
4. Document breaking changes

**Időigény:** 6-8 óra (1 nap)

---

### 5.3 Code Quality Improvements (2-3 nap)

#### Linting & Formatting
```bash
pip install black flake8 pylint isort

# Auto-format code
black app/ tests/

# Sort imports
isort app/ tests/

# Check code quality
flake8 app/ --max-line-length=100
pylint app/ --fail-under=8.0
```

#### Refactor Long Functions
**Target:** Functions > 50 lines, cyclomatic complexity > 10

**Example:**
```python
# Before (100 lines):
def make_decision(self, ticker, signals):
    # Fetch data
    # Calculate indicators
    # Apply strategy
    # Risk management
    # Position sizing
    # Generate decision
    # Log result
    # Return decision

# After (decomposed):
def make_decision(self, ticker, signals):
    data = self._fetch_market_data(ticker)
    indicators = self._calculate_indicators(data)
    strategy_signal = self._apply_strategy(signals, indicators)
    risk_metrics = self._assess_risk(ticker, strategy_signal)
    position = self._calculate_position_size(risk_metrics)
    decision = self._generate_decision(ticker, strategy_signal, position)
    self._log_decision(decision)
    return decision
```

**Feladatok:**
1. Identify complex functions (radon, mccabe)
2. Extract helper methods
3. Improve naming
4. Add docstrings
5. Remove dead code

**Időigény:** 15-20 óra (2-3 nap)

---

### 5.4 Documentation Improvements (1-2 nap)

#### API Documentation (Sphinx)
```bash
pip install sphinx sphinx-rtd-theme

cd docs
sphinx-quickstart
```

**Generate API docs:**
```python
# conf.py
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
```

```bash
sphinx-apidoc -o docs/api app/
make html
```

#### README Updates
- Architecture diagram (Mermaid)
- Component interaction flowchart
- Updated installation instructions
- Troubleshooting expanded

**Időigény:** 8-12 óra (1-2 nap)

---

### 5.5 Dependency Audit (1 nap)

```bash
pip install pip-audit safety

# Check for vulnerabilities
pip-audit
safety check

# Update dependencies
pip list --outdated
pip install --upgrade <package>

# Regenerate requirements
pip freeze > requirements.txt
```

**Feladatok:**
1. Audit all dependencies for security issues
2. Update to latest stable versions
3. Remove unused dependencies
4. Pin versions for reproducibility
5. Create `requirements-dev.txt` for development tools

**Időigény:** 6-8 óra (1 nap)

---

### Összesített Időbecslés (Opció 5)

| Komponens | Időigény | Impact |
|-----------|----------|--------|
| Type hints | 15-20 óra | Kód minőség |
| Deprecation fixes | 6-8 óra | Maintainability |
| Code quality | 15-20 óra | Readability |
| Documentation | 8-12 óra | Onboarding |
| Dependency audit | 6-8 óra | Security |
| **ÖSSZESEN** | **50-68 óra** | **~1.5 hét** |

### Várható Eredmény
- ✅ Tisztább, típusos kód
- ✅ Jobb IDE support (autocomplete, type checking)
- ✅ Könnyebb onboarding új fejlesztőknek
- ✅ Kevesebb runtime error
- ✅ Biztonságosabb dependencies

---

## 🎯 Összefoglaló Ajánlás

### Prioritási Sorrend

#### 1️⃣ **Azonnal (Sprint 11a - 1 hét):**
**Opció 1:** Test Coverage 83% → 90%+
- **Miért:** 6 failing test kritikus bug, production confidence
- **Időigény:** 3-4 nap (19-26 óra)
- **ROI:** Magas (stabil kódbázis)

#### 2️⃣ **Gyorsan utána (Sprint 11b - 1-2 hét):**
**Opció 2:** Raspberry Pi Deployment (ha hardver megérkezett)
- **Miért:** Production ready állapot elérése
- **Időigény:** 1-2 nap (6.5-10.5 óra)
- **ROI:** Magas (valódi production üzem)

**ÉS**

**Opció 4:** Performance Optimization
- **Miért:** Skálázhatóság, user experience
- **Időigény:** 1 hét (39-55 óra)
- **ROI:** Közepes-Magas (gyorsabb rendszer)

#### 3️⃣ **Párhuzamosan (Sprint 12-13 - 2-4 hét):**
**Opció 3:** Új Funkciók (fázisokban)
- **Sprint 12:** LSTM + Broker API (8-10 nap)
- **Sprint 13:** Ensemble + Dashboard (6-8 nap)
- **Időigény:** 4-5 hét összesen
- **ROI:** Magas (új képességek, versenyképesség)

#### 4️⃣ **Folyamatosan (háttérben):**
**Opció 5:** Refactoring & Tech Debt
- **Miért:** Hosszú távú karbantarthatóság
- **Időigény:** 1.5 hét (50-68 óra)
- **ROI:** Közepes (clean code, kevesebb bug hosszú távon)

---

## 📅 Javasolt Sprint Schedule

```
Sprint 11a (Week 1):        Opció 1 - Coverage 90%+ ✅
Sprint 11b (Week 2):        Opció 2 - RPi Deploy ✅ + Opció 4 - Performance 🚀
Sprint 12 (Week 3-4):       Opció 3.1 - LSTM + Broker API 🤖
Sprint 13 (Week 5-6):       Opció 3.2 - Ensemble + Dashboard 📊
Sprint 14 (Week 7-8):       Opció 5 - Refactoring & Opció 3.3 - Advanced Portfolio 🧹
```

### Teljes Timeline
- **Sprint 11:** 2 hét (Coverage + RPi + Performance)
- **Sprint 12-13:** 4 hét (ML + Real-time features)
- **Sprint 14:** 2 hét (Refactoring + Advanced optimization)
- **ÖSSZESEN:** ~8 hét (2 hónap)

---

## ✅ Action Items (Következő 24 óra)

1. **Döntés:** Melyik opcióval kezdjünk? (Ajánlott: Opció 1)
2. **Setup:** Test környezet előkészítése
3. **Start:** Failing tests javítása (`test_portfolio_correlation_manager.py`)
4. **Track:** Progress monitoring (GitHub Projects / Jira)

---

**Dokumentum készítette:** GitHub Copilot  
**Dátum:** 2026-02-03  
**Státusz:** ✅ Ready for review & execution  
**Next Review:** Sprint 11a kickoff előtt
