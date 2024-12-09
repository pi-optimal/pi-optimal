<p align="center">
    <img src="https://raw.githubusercontent.com/pi-optimal/pi_optimal/main/media/logo.png" alt="pi_optimal Logo" width="250"/>
</p>

<p align="center">
    <a href="https://github.com/pi-optimal/pi_optimal/releases">
        <img src="https://img.shields.io/github/v/release/pi-optimal/pi_optimal?color=blue" alt="Latest Release"/>
    </a>
    <a href="https://github.com/pi-optimal/pi_optimal/actions/workflows/ci.yml">
        <img src="https://github.com/pi-optimal/pi_optimal/actions/workflows/ci.yml/badge.svg" alt="Build Status"/>
    </a>
    <a href="https://github.com/pi-optimal/pi_optimal/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/pi-optimal/pi_optimal?color=green"/>
    </a>
</p>

<p align="center">
    <strong>
        <a href="https://pi-optimal.io">Website</a>
        •
        <a href="https://pi-optimal.readthedocs.io/en/stable/">Docs</a>
        •
        <a href="https://join.slack.com/t/pioptimal/shared_invite/xyz">Community Slack</a>
    </strong>
</p>

---

# 🤖 What is `pi_optimal`?

`pi_optimal` is an open-source Python library that helps you **model, optimize, and control complex systems through Reinforcement Learning (RL)**. Whether your system involves advertising delivery, energy consumption, inventory management, or any scenario where sequential decision-making is paramount, `pi_optimal` provides a flexible and modular interface to train, evaluate, and deploy RL-based policies.

Built for data scientists, RL practitioners, and developers, `pi_optimal`:

- Offers a **time-series aware RL pipeline**, handling lookback windows and forecasting future states.
- Supports **various action spaces** (continuous, discrete, or multi-dimensional), enabling complex control strategies.
- Integrates easily with **custom reward functions**, empowering you to tailor the agent’s objectives to your business goals.
- Facilitates **multi-step planning**, allowing you to look ahead and optimize future outcomes, not just the immediate next step.

If you find `pi_optimal` useful, consider joining our [community Slack](https://join.slack.com/t/pioptimal/shared_invite/xyz) and give us a ⭐ on GitHub!

---

# 🎯 Why use `pi_optimal`?

In dynamic and complex systems, even experienced operators can struggle to find the best decisions at every step. `pi_optimal` helps you:

- **Automate Decision-Making:** Reduce human overhead by letting RL agents handle routine optimization tasks.
- **Optimize Performance Over Time:** Forecast system states and choose actions that yield smooth, cost-effective, or profit-maximizing trajectories.
- **Incorporate Uncertainty:** Account for uncertainty in future outcomes with built-in approaches to handle uncertain environments.
- **Seamlessly Integrate with Your Workflow:** `pi_optimal` fits easily with your existing code, data pipelines, and infrastructure.

---

# 🌐 Use Cases

- **Advertising Delivery Optimization:** Smooth out ad impressions over time, ensuring efficient, controlled delivery that meets pacing and budget constraints.
- **Energy Management:** Balance supply and demand, optimize resource allocation, and reduce operational costs.
- **Inventory and Supply Chain:** Manage stock levels, forecast demand, and plan orders for just-in-time deliveries.
- **Dynamic Pricing and Bidding:** Adjust bids, prices, and frequency caps in real-time to maximize revenue or reduce costs.

---

# 🚀 Getting Started

## Installation

`pi_optimal` currently relies on [Poetry](https://python-poetry.org/) for installation. Make sure you have Poetry installed, then clone the repository and install:

```bash
git clone https://github.com/pi-optimal/pi_optimal.git
cd pi_optimal
poetry install
```

## Example Usage

Below is a simplified excerpt demonstrating how `pi_optimal` can be applied to optimize ad delivery. For a more detailed walkthrough, refer to the [notebooks](./examples).

```python
import pi_optimal as po
import pandas as pd

# Load historical dataset
df_historical = pd.read_csv('data/historical_adset_control.csv', parse_dates=['created_at'])

# Define state, action, and reward columns
state_cols = [
    'hour_of_day', 'day_of_week', 'adset_impressions_diff', 
    'adset_settings_total_budget', 'adset_settings_remaining_hours', 
    'adset_targetings_total_population'
]

action_cols = [
    'adset_settings_maximum_cpm', 'adset_targetings_frequency_capping_requests', 
    'adset_settings_bidding_strategy', 'adset_settings_pacing_type'
]

reward_col = 'reward'
timestamp_col = 'created_at'
unit_col = 'unit_index'

# Create a TimeseriesDataset
LOOKBACK_TIMESTEPS = 8
historical_dataset = po.datasets.timeseries_dataset.TimeseriesDataset(
    df=df_historical,
    state_columns=state_cols,
    action_columns=action_cols,
    reward_column=reward_col,
    timestep_column=timestamp_col,
    unit_index=unit_col,
    lookback_timesteps=LOOKBACK_TIMESTEPS
)

# Initialize and train an RL agent (MPC-based continuous agent as example)
from pi_optimal.agents.agent import Agent

agent = Agent(
    dataset=historical_dataset,
    type="mpc-continuous",
    config={"uncertainty_weight": 0.5}
)

agent.train()

# Load current data and predict optimal actions
df_current = pd.read_csv('data/current_adset_control.csv', parse_dates=['created_at'])
df_current['reward'] = df_current.apply(your_reward_function, axis=1)  # define a custom reward function

current_dataset = po.datasets.timeseries_dataset.TimeseriesDataset(
    df=df_current,
    dataset_config=historical_dataset.dataset_config,
    lookback_timesteps=LOOKBACK_TIMESTEPS,
    train_processors=False
)

best_actions = agent.predict(current_dataset, inverse_transform=True, n_iter=15)
print("Recommended actions:\n", best_actions)
```

---

# ✨ Features

1. **Time-Series Aware RL**:  
   Directly handle sequences, lookback windows, and rolling state representations.

2. **Flexible Action Spaces**:  
   Support for continuous and discrete actions, or complex multidimensional action vectors.

3. **Custom Reward Functions**:  
   Easily define domain-specific rewards to reflect real-world KPIs.

4. **Multi-Step Planning**:  
   Implement look-ahead strategies that consider future impacts of current actions.

5. **Data Processing and Visualization**:  
   Built-in tools for dataset preparation, trajectory visualization, and iterative evaluation.

---

# 📖 Documentation

- **Tutorials & Examples**: Walk through real-world examples to understand how to best apply `pi_optimal`.
- **API Reference**: Detailed documentation for all classes, methods, and functions.
- **Best Practices**: Learn recommended strategies for defining rewards, choosing architectures, and tuning hyperparameters.

[Read the Docs »](https://pi-optimal.readthedocs.io/en/stable/)

---

# 🤝 Contributing and Community

We welcome contributions from the community! If you have feature requests, bug reports, or want to contribute code:

- Open an issue on [GitHub Issues](https://github.com/pi-optimal/pi_optimal/issues).
- Submit a pull request with your proposed changes.
- Join our [Slack community](https://join.slack.com/t/pioptimal/shared_invite/xyz) to ask questions, share ideas, or get help.

A big thanks to all contributors who make `pi_optimal` better every day!

---

# 🙋 Get Help

If you have questions or need assistance, the fastest way to get answers is via our [community Slack channel](https://join.slack.com/t/pioptimal/shared_invite/xyz). Drop by and say hello!

---

# 🌱 Roadmap

Check out our [roadmap](https://github.com/pi-optimal/pi_optimal/projects) to see what we’re working on next. Have suggestions or would like to see a new feature prioritized? Let us know in our Slack or open an issue.

---

# 📜 License

`pi_optimal` is distributed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

