# Wood Packing Optimization Model

## Project Overview

This project develops a mathematical model to estimate the solid wood volume that can be packed into a 1 × 1 × 1 m container.

The model considers cylindrical wood pieces aligned along the x-axis and compares the resulting conversion factors with current industry rule-of-thumb values.

## Objectives

- Develop a mathematical optimization model for wood packing.
- Maximize the total solid wood volume inside a 1 m³ container.
- Compare model-based conversion factors with current conversion factors.
- Visualize packing arrangements and results.

## Mathematical Model

Maximize:

Z = Σ π rᵢ² lᵢ sᵢ

Subject to:

- Wood pieces must remain inside the container.
- Cylinders must not overlap.
- Each selected cylinder must satisfy the packing constraints.
- Binary selection variables determine whether a piece is included.

## Dataset

The project uses `wood_dataset.csv`, containing wood-piece dimensions for two scenarios:

- S1 logs: 1 m logs (stere)
- S3 pieces: 30–40 cm pieces (lcm)

## Methodology

A greedy packing heuristic with randomized restarts is used to obtain feasible packing solutions.

The solutions are checked against the model constraints using `check_feasible()`.

**Note:** The heuristic provides feasible solutions and lower bounds on the optimal objective value; it does not guarantee a global optimum.

## Conversion Factor Comparison

The model estimates the following conversion factors:

- stere → scm
- lcm → scm
- lcm → stere

These estimates are compared with current rule-of-thumb values.

## Requirements

- Python 3
- NumPy
- Matplotlib

Install the required libraries:

```bash
pip install numpy matplotlib
```

## How to Run

1. Download or clone this repository.
2. Keep `wood_model_vs_current.py` and `wood_dataset.csv` in the same folder.
3. Install the required libraries.
4. Run:

```bash
python wood_model_vs_current.py
```

The program prints the model-versus-current comparison table and generates:

`wood_model_vs_current.png`

## Project Files

| File | Description |
|---|---|
| `wood_model_vs_current.py` | Optimization model and packing algorithm |
| `wood_dataset.csv` | Input dataset |
| `wood_model_vs_current.png` | Generated visualization |

## Technologies Used

Python, NumPy, Matplotlib

---

**Project:** Wood Packing Optimization Model  
**Purpose:** Model-based estimation and comparison of wood conversion factors.# wood-packing-model
