# Battery EIS Simulation and 2RC Model Fitting using PyBaMM

This project performs **Galvanostatic Electrochemical Impedance Spectroscopy (GEIS)** on a lithium-ion battery cell modeled using the **Single Particle Model (SPM)** from [PyBaMM](https://github.com/pybamm-team/PyBaMM). The script simulates the voltage response to a sinusoidal current input across a range of frequencies and extracts the battery’s impedance spectrum using **FFT**. It then fits a **2RC equivalent circuit model** to the computed impedance.

---

## What This Script Does

### Simulation
- Uses the **SPM (Single Particle Model)** with parameters from the `OKane2022` dataset.
- Applies a **sinusoidal current input** at different frequencies (`1e-4` Hz to `1e3` Hz).
- Extracts the **voltage response** over time for each frequency.
- Uses **Fast Fourier Transform (FFT)** to compute the complex impedance at the fundamental frequency.

### Fitting
- Defines a **2RC equivalent circuit model**:
  \[
  Z(\omega) = R_0 + \frac{R_1}{1 + j \omega R_1 C_1} + \frac{R_2}{1 + j \omega R_2 C_2}
  \]
- Fits the simulated impedance data to this model using `scipy.optimize.curve_fit`.

### Plotting
- Displays a **Nyquist plot** (`Re(Z)` vs `-Im(Z)`) for both:
  - Simulated impedance
  - Fitted 2RC model

