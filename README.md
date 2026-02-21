# TRPO Hopper-v4 from Scratch

A clean, PyTorch-based implementation of Trust Region Policy Optimization (TRPO) designed to solve the Gymnasium `Hopper-v4` environment. 

This project builds the TRPO math completely from scratch—including the Surrogate Objective, Fisher-Vector products, Conjugate Gradient solver, and backtracking line search—without relying on heavy RL libraries like Stable Baselines.

## Features
* **Custom TRPO Implementation:** Pure PyTorch implementation of the math behind Trust Region Policy Optimization.
* **Conjugate Gradient Solver:** Efficiently approximates the inverse Hessian-vector product to maintain the Trust Region without massive memory overhead.
* **Generalized Advantage Estimation (GAE):** Stabilizes the policy gradient variance.
* **Gymnasium v1.0 Ready:** Fully compatible with modern Gymnasium wrappers (`NormalizeObservation`, `TransformObservation`).
* **Entropy Exploration Bonus:** A critical addition to the objective function that prevents the agent from getting stuck in the "standing still" local optimum.
* **Studio Video Recorder:** A custom video rendering loop that correctly transfers observation normalization statistics so the recorded video matches the true training behavior.

## The "Standing" Local Optimum
One of the core challenges addressed in this repository is the Hopper's tendency to find a safe local optimum. 

Because falling terminates the episode and stops the reward, standard TRPO agents often learn to "stand still" or make tiny, safe hops to stay alive for the full 8 seconds. This repo solves the "Genius Coward" problem by introducing an **Entropy Regularization** coefficient, effectively forcing the agent to take physical risks (like leaning forward) until it discovers the high-speed running gait.

## Installation

Ensure you have Python 3.8+ and install the required dependencies. You will need Mujoco installed on your system.

```bash
pip install torch numpy matplotlib
pip install gymnasium[mujoco]
