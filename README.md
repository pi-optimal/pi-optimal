<p align="center">
    <img src="media/logo.png" alt="pi_optimal Logo" width="250"/>
</p>

<p align="center">
    <a href="https://github.com/pi-optimal/pi-optimal/releases">
        <img src="https://img.shields.io/github/v/release/pi-optimal/pi-optimal?color=blue" alt="Latest Release"/>
    </a>
    <a href="https://github.com/pi-optimal/pi-optimal/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/pi-optimal/pi-optimal"/>
    </a>
</p>

<p align="center">
    <strong>
        <a href="https://pi-optimal.com">Website</a>
        •
        <a href="https://pi-optimal.readthedocs.io/en/stable/">Docs</a>
        •
        <a href="https://join.slack.com/t/pioptimal/shared_invite/zt-2w4z32qtt-Q7EdDvmSi9vWFCPb22_qVA">Community Slack</a>
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

If you find `pi_optimal` useful, consider joining our [community Slack](https://join.slack.com/t/pioptimal/shared_invite/zt-2w4z32qtt-Q7EdDvmSi9vWFCPb22_qVA) and give us a ⭐ on GitHub!

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

`pi_optimal` uses [Poetry](https://python-poetry.org/) for dependency management and installation. Follow these steps to get started:

1. **Ensure you are not within a virtual environment** (e.g., deactivate it with `conda deactivate` if using Conda).  
2. **Install Poetry** (if you don’t already have it):

    ```bash
    pipx install poetry
    ```

3. **Clone the repository** and navigate to its directory:

    ```bash
    git clone https://github.com/pi-optimal/pi-optimal.git
    cd pi-optimal
    ```

4. **Install dependencies** using Poetry:

    ```bash
    poetry install
    ```

Once you've completed the installation, you can open any notebook from the [notebooks](./notebooks) directory. To use the installed environment, select the newly created virtual environment in your Jupyter kernel selection. It should appear with a name similar to `pi-optimal-xyz-py3.10`.


## Example Usage

Below is a simplified excerpt demonstrating how `pi_optimal` can be applied to optimize ad delivery. For a more detailed walkthrough, refer to the [notebooks](./notebooks).

```python
import pandas as pd
import pi_optimal as po

# Load historical room climate control data
df_room_history = pd.read_csv('room_climate_history.csv')

# Prepare dataset: define states (e.g., room conditions), actions (e.g., heater settings), and reward (e.g., comfort level)
climate_dataset = po.datasets.TimeseriesDataset(
    df_room_history,
    state_columns=['temperature', 'humidity'],
    action_columns=['heater_power'],
    reward_column='comfort_score',
    timestep_column='timestamp',
    unit_index='room_id',
    lookback_timesteps=8
)

# Train a reinforcement learning agent for climate control
climate_agent = po.Agent(dataset=climate_dataset, type="mpc-continuous", config={"uncertainty_weight": 0.5})
climate_agent.train()

# Load current room data to predict next actions
df_current_conditions = pd.read_csv('current_room_conditions.csv')
current_dataset = po.datasets.TimeseriesDataset(df_current_conditions, dataset_config=climate_dataset.dataset_config, lookback_timesteps=8, train_processors=False)

# Predict optimal heater settings for improved comfort
optimal_actions = climate_agent.predict(current_dataset)
print(optimal_actions)
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

- Open an issue on [GitHub Issues](https://github.com/pi-optimal/pi-optimal/issues).
- Submit a pull request with your proposed changes.
- Join our [Slack community](https://join.slack.com/t/pioptimal/shared_invite/zt-2w4z32qtt-Q7EdDvmSi9vWFCPb22_qVA) to ask questions, share ideas, or get help.

A big thanks to all contributors who make `pi_optimal` better every day!

---

# 🙋 Get Help

If you have questions or need assistance, the fastest way to get answers is via our [community Slack channel](https://join.slack.com/t/pioptimal/shared_invite/zt-2w4z32qtt-Q7EdDvmSi9vWFCPb22_qVA). Drop by and say hello!

---

# 🌱 Roadmap

Check out our [roadmap](https://github.com/pi-optimal/pi-optimal/projects) to see what we’re working on next. Have suggestions or would like to see a new feature prioritized? Let us know in our Slack or open an issue.

---

# 📜 License

`pi_optimal` is distributed under the GNU Affero General Public License (AGPL). See [LICENSE](LICENSE) for details.

