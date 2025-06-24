# Battery EIS Simulation and 2RC + Warburg Model Fitting using PyBaMM

This project performs **Galvanostatic Electrochemical Impedance Spectroscopy (GEIS)** on a lithium-ion battery modeled with the **Single Particle Model with electrolyte (SPMe)** from [PyBaMM](https://github.com/pybamm-team/PyBaMM). The script simulates the voltage response to a sinusoidal current input across a range of frequencies and extracts the battery’s impedance spectrum using **FFT**. It then fits a **2RC + Warburg equivalent circuit model** to the computed impedance data.

Additionally, the script simulates **cell aging** with SEI growth and lithium plating mechanisms over multiple charge/discharge cycles, performs EIS on the aged cell, and compares the fresh vs aged cell impedance.

---

## What This Script Does

### Simulation

* Uses the **SPMe model** with lumped thermal effects, solvent-diffusion limited SEI growth, and reversible lithium plating submodels.
* Loads parameters from the **OKane2022** parameter set.
* Applies a **sinusoidal current excitation** (50 mA amplitude) at logarithmically spaced frequencies from $10^{-4}$ Hz to $10^{3}$ Hz.
* Runs time-domain simulations for multiple sinusoidal periods per frequency.
* Extracts **voltage and current signals** in steady state.
* Uses **Fast Fourier Transform (FFT)** to compute the complex impedance at the excitation frequency.

### Equivalent Circuit Model (ECM) Fitting

* Defines a **2RC + Warburg model** impedance function:

  $$
  Z(\omega) = R_0 + \frac{R_1}{1 + j \omega R_1 C_1} + \frac{R_2}{1 + j \omega R_2 C_2} + \sigma \frac{1 - j}{\sqrt{\omega}}
  $$
* Fits simulated impedance data to the ECM using `scipy.optimize.curve_fit` with combined real and imaginary parts.
* Prints optimized circuit parameters.

### Aging Simulation

* Runs a **gentle aging cycling protocol** (charge/discharge/rest cycles) with slowed SEI growth kinetics.
* Updates initial conditions for the aged cell model.
* Repeats EIS simulation on the aged cell.

### Visualization

* Plots **Nyquist plots** comparing fresh and aged cell impedance.
* Shows fit quality of ECM on fresh cell data.

---

## Requirements

* Python 3.8+
* [PyBaMM](https://github.com/pybamm-team/PyBaMM) (tested with latest stable release)
* numpy
* scipy
* matplotlib

Install dependencies with:

```bash
pip install pybamm numpy scipy matplotlib
```

---

## Usage

Run the script directly:

```bash
python battery_eis_simulation.py
```

The script will:

1. Run EIS simulations on a fresh cell.
2. Fit the 2RC + Warburg ECM to the fresh impedance data.
3. Run an aging simulation with SEI growth and plating.
4. Run EIS simulations on the aged cell.
5. Plot Nyquist plots comparing fresh and aged cells.

---
