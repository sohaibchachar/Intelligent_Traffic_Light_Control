# Intelligent Traffic Light Control using Reinforcement Learning

## Overview

This project investigates the application of reinforcement learning (RL) to dynamically control traffic signals with the goal of minimizing vehicle wait times and overall traffic congestion. The research replicates and extends the work done in the **IntelliLight** model introduced by Wei et al. (2018), which uses a Deep Q-Network (DQN) to enable intelligent traffic light control.

## Problem Statement

Urban traffic congestion is a significant problem that results in:
- Increased travel times
- Wasted fuel
- Environmental consequences

Current traffic light systems are typically static and operate via pre-defined schedules, resulting in them not being able to adapt to changing traffic conditions. This project addresses this limitation by implementing an adaptive RL-based traffic control system.

## Key Features

- **Deep Q-Network (DQN)** implementation for traffic signal control
- **Memory Palace** enhancement for learning from rare traffic events
- **Phase Gate** enhancement for phase-specific behavior learning
- Support for both **synthetic** and **real-world** traffic data
- Integration with **SUMO** and **CityFlow** traffic simulators
- Dynamic adaptation to changing traffic patterns

## Project Structure

```
Project/
├── README.md
└── Project Milestone 5_ Final Project Report (3).pdf
```

## Technologies Used

### For Synthetic Data Experiments
- **Operating System**: Ubuntu 24.04
- **Python**: 3.6 (Conda environment)
- **Libraries**:
  - SUMO with TraCI module
  - Keras 2.2.0
  - TensorFlow 1.9.0

### For Real-World Data Experiments
- **Operating System**: Ubuntu 24.04
- **Python**: 3.9 (Conda environment)
- **Libraries**:
  - SUMO with TraCI module
  - CityFlow
  - PyTorch 1.11.0
  - CUDA 11.3
  - gym 0.21.0
  - numpy 1.21.5

## MDP Formulation

The traffic signal control problem is formalized as a Markov Decision Process (MDP) with the following components:

### State Space (S)
```
s = (L₁, ..., Lₙ, V₁, ..., Vₙ, W₁, ..., Wₙ, M, Pc, Pn)
```
Where:
- **Lᵢ**: Queue length in lane i
- **Vᵢ**: Number of vehicles in lane i
- **Wᵢ**: Cumulative waiting time in lane i
- **M**: Grid/image representation of vehicle positions
- **Pc, Pn**: Current and next signal phase

### Action Space (A)
```
a ∈ {0, 1}
```
- **0**: Keep current phase
- **1**: Change to next phase

### Reward Function (R)
```
R(s,a) = w₁∑Lᵢ + w₂∑Dᵢ + w₃∑Wᵢ + w₄C + w₅N + w₆T
```
Where:
- **Dᵢ**: Lane delay
- **C**: Penalty for phase switch
- **N**: Number of vehicles passing intersection
- **T**: Travel time
- **w₁-w₆**: Configurable weights

### Discount Factor (γ)
Set to **0.8** to balance immediate and future rewards.

## Model Architecture

The IntelliLight system uses a **Deep Q-Network (DQN)** that:
- Receives grid-based representation of traffic intersection as input
- Processes spatial features through **Convolutional Neural Network (CNN)** layers
- Uses **fully connected layers** to output Q-value estimates
- Implements **experience replay** for training stability
- Utilizes **target network** to stabilize Q-value updates

### Enhancements

1. **Memory Palace**: Separate memory buffer for uncommon/rare traffic events
2. **Phase Gate**: Enables phase-specific behavior learning for different signal phases

## Installation

### Prerequisites
- Ubuntu 24.04 (or compatible Linux distribution)
- Conda package manager
- NVIDIA GPU (recommended for faster training)
- CUDA 11.3 (for real-world experiments)

### Setup for Synthetic Data Experiments

1. Create a Conda environment with Python 3.6:
```bash
conda create -n intellilight python=3.6
conda activate intellilight
```

2. Install required libraries:
```bash
pip install tensorflow==1.9.0 keras==2.2.0
# Install SUMO with TraCI module
```

3. Clone the IntelliLight repository:
```bash
git clone https://github.com/wingsweihua/IntelliLight
cd IntelliLight
```

### Setup for Real-World Data Experiments

1. Create a Conda environment with Python 3.9:
```bash
conda create -n libsignal python=3.9
conda activate libsignal
```

2. Create DaRL directory and clone LibSignal:
```bash
mkdir DaRL
cd DaRL
git clone https://github.com/DaRL-LibSignal/LibSignal
cd LibSignal
```

3. Follow the full installation steps from the [LibSignal GitHub repository](https://github.com/DaRL-LibSignal/LibSignal), including:
   - CityFlow installation
   - SUMO installation
   - Installing requirements.txt
   - Or use the provided Docker image

## Usage

### Running Synthetic Data Experiments

1. Navigate to the `conf/one_run` folder and open `deeplight_agent.conf`
2. Set the `DDQN` parameter to `true`
3. Open `runexp.py` and edit the `list_traffic_files` list
4. Keep only the experiment you want to run and comment out others
5. Optionally modify `traffic_light_dqn.main` call to use `sumo_gui` for visualization
6. Run the experiment:
```bash
python runexp.py
```

**Output Location**: Results are saved in `records/one_run/`
- `log_rewards.txt` → queue length, wait time, delay
- `memories.txt` → reward values
- Model weights saved in `model/one_run/`

### Running Real-World Data Experiments

1. Download the Jinan dataset from: https://github.com/wingsweihua/colight/tree/master/data/Jinan
2. Place the folder into `data/raw_data` directory
3. Navigate to `configs/sim` directory
4. Create a custom CityFlow configuration file (`.cfg`) following the [LibSignal tutorial](https://darl-libsignal.github.io/LibSignalDoc/content/tutorial/Customize%20Dataset.html)
5. In `run.py`, change the network parameter to your config file
6. Run the experiment:
```bash
python run.py
```

**Output Location**: 
- Terminal output for real-time metrics
- Results saved in `data/output_data/tsc/cityflow_dqn/logger/date_BRF.log`
- Model weights in `data/output_data/tsc/cityflow_dqn/`

## Experiments

The project replicates four main experiments from the IntelliLight paper:

### 1. Directional Demand Shift
Tests the agent's ability to handle shifting directional traffic needs (West-East to South-North).

**Results**:
- Paper (Base): Reward = -3.07 | Queue Length = 10.65 | Delay = 2.63
- Paper (BASE+MP+PG): Reward = 0.39 | Queue Length = 0.005 | Delay = 1.59
- **Our Model**: Reward = 28.3 | Queue Length = 0.00 | Delay = 2.40

### 2. Equal Low Traffic
Assesses performance under balanced, low-traffic conditions.

### 3. Imbalanced Traffic
Evaluates handling of uneven traffic distribution across lanes.

### 4. Dynamic Traffic Patterns
Tests adaptation to dynamically changing traffic patterns over time.

## Results

The IntelliLight model demonstrates superior performance compared to:
- **Fixed Time controllers**: Static signal changes
- **Self-Organizing Traffic Lights**: Based on local vehicle counts

The DQN model with Memory Palace and Phase Gate enhancements performed best among all DQN variants.

## Hardware Requirements

- **CPU**: Intel® Core™ i7 (or equivalent)
- **GPU**: NVIDIA GeForce GTX 1060 (or equivalent)
- **RAM**: 8 GB minimum (64 GB available in testing)
- **Storage**: 10 GB minimum (2 TB available in testing)

## References

### Primary Paper
- **IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control**
  - Authors: Hua Wei, Guanjie Zheng, Huaxiu Yao, Zhenhui Li
  - Published at: 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (2018)
  - Link: https://dl.acm.org/doi/10.1145/3219819.3220096

### Resources
- **Original IntelliLight Code**: https://github.com/wingsweihua/IntelliLight
- **Extended Framework (LibSignal)**: https://darl-libsignal.github.io/
- **SUMO (Simulation of Urban MObility)**: https://www.eclipse.org/sumo/
- **TraCI (Traffic Control Interface)**: https://sumo.dlr.de/docs/TraCI.html

## Team Members

- **Tara Walenczyk** (33%): Theoretical and modeling portion, MDP formulation, DQN architecture, enhancements
- **Sohaib Chachar** (33%): Replicated and analyzed IntelliLight experiments, performance metrics, visualizations
- **Andrew Aquino** (33%): Environmental setup, experimental extensions, discount factor analysis, real-world data implementation

## License

This project is based on the IntelliLight research paper and uses code from the original repository. Please refer to the original repository licenses for usage terms.

## Acknowledgments

This research is based on the paper "IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control" by Wei et al. (2018). We acknowledge the original authors for their foundational work in applying reinforcement learning to traffic signal control.

## Future Work

Potential extensions and improvements:
- Testing with additional real-world traffic datasets
- Exploring other RL algorithms (PPO, A3C, etc.)
- Multi-intersection coordination
- Integration with real-time traffic monitoring systems
- Energy efficiency considerations

## Contributing

This is an academic research project. For questions or contributions, please refer to the original IntelliLight repository or contact the project team members.

---

**Note**: This project was completed as part of a Reinforcement Learning course at NJIT (Spring 2025).


