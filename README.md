# RebalanceAI — DQN-Based Stock Trading

An experimental reinforcement-learning project that models single-stock trading as a sequence of buy, sell, and hold decisions. The notebook uses a custom OpenAI Gym environment and a Deep Q-Network (DQN) agent, then compares the agent's portfolio value with a buy-and-hold baseline.

> **Scope note:** Despite the notebook filename, `ACA_Final_Project_Portfolio_Balancing.ipynb`, the current implementation focuses on trading one stock at a time (AAPL in the examples). It is not yet a multi-asset portfolio-allocation or portfolio-balancing system.



## Project overview

The goal is to explore how a reinforcement-learning agent can make sequential trading decisions using historical market data and technical indicators.

The notebook:

- Downloads OHLCV market data using `yfinance`.
- Calculates technical features such as SMA, EMA, RSI, MACD and its signal line, a rolling Sharpe-style statistic, and—in later experimental variants—short-horizon returns and average volume.
- Defines a custom Gym trading environment with cash balance, shares held, portfolio net worth, and three actions: **hold**, **buy one share**, or **sell one share**.
- Trains a DQN agent using an online policy network, a target network, experience replay, epsilon-greedy action selection, and temporal-difference targets.
- Evaluates the learned greedy policy on historical data and plots portfolio value against a buy-and-hold baseline.
- Visualizes action counts over a test rollout to inspect whether the agent is mostly holding, buying, or selling.

## Methods and implementation

### Market data and features

Historical stock prices are fetched from Yahoo Finance. The main examples use Apple (`AAPL`), with a 2020 training period and later historical windows for evaluation. The notebook contains successive feature-engineering versions, so the exact indicator set and evaluation dates depend on the section being run.

### Trading environment

The custom environment tracks cash, share holdings, current time step, and marked-to-market portfolio value. It exposes a discrete action space:

| Action | Meaning |
| --- | --- |
| `0` | Hold current position |
| `1` | Buy one share, if sufficient cash is available |
| `2` | Sell one share, if a share is held |

The base reward is the change in portfolio net worth. Later variants experiment with small bonuses for favorably timed trades and penalties associated with trading. These are reward-shaping experiments rather than a complete model of real-world execution costs.

### DQN agent

The notebook implements a feed-forward Q-network with two 128-unit ReLU hidden layers and one output per action. Training uses experience replay, a separate target network updated periodically, an Adam optimizer, and an epsilon-greedy exploration schedule.

### Evaluation

The notebook plots the agent's portfolio value alongside a buy-and-hold benchmark and can display the counts and timing of hold, buy, and sell actions. It does not establish that the strategy is profitable or robust; results depend on the data window, random initialization, reward design, and training choices.

## Repository contents

```text
.
├── README.md
└── ACA_Final_Project_Portfolio_Balancing.ipynb
```

The notebook includes data retrieval, feature engineering, the custom environment, DQN training, and evaluation plots. Market data is fetched when the notebook is run, so a separate data file is not required for the examples.

## How to run

### Google Colab (recommended)

1. Open the notebook using the **Open in Colab** badge above.
2. Run the setup cell to install dependencies.
3. Run the cells in order so the data, environment, networks, and trained policy are initialized before evaluation.
4. Inspect the training reward logs, portfolio-value comparisons, and action-count plots.
5. To experiment, change the ticker, date windows, indicators, initial balance, training episodes, exploration schedule, or reward parameters, then rerun the relevant cells.


## Experiments to try

- Compare different train/test date windows and tickers.
- Vary the feature set and observe its effect on the learned action sequence.
- Tune episode count, discount factor, epsilon decay, and target-network update frequency.
- Compare reward formulations with and without trade bonuses or penalties.
- Add transaction costs, slippage, position limits, and risk-adjusted evaluation metrics.
- Use a leakage-safe preprocessing pipeline and repeat experiments across multiple out-of-sample windows.

## Current limitations

- The examples trade a single stock and use fixed one-share actions; they do not allocate capital across a diversified portfolio.
- The notebook contains multiple experimental versions of the environment and feature set. Run the selected version consistently and verify that its training and evaluation cells use the same observation layout and reward definition.
- The current backtest is a learning prototype, not a production trading system. It does not fully model fees, slippage, liquidity, market impact, taxes, or operational constraints.
- Indicator values are back-filled in the current feature preparation. This can introduce look-ahead leakage for early rows, so preprocessing should be revised before drawing rigorous conclusions from backtests.
- Portfolio-value plots are provided, but no claim of benchmark outperformance is made. Evaluate with repeated out-of-sample tests, drawdown, volatility, and risk-adjusted metrics before making strategy conclusions.
