import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as timer
from scipy.fft import fft
from scipy.optimize import curve_fit

# Set up
model = pybamm.lithium_ion.SPM(options={"surface form": "differential"}, name="SPM")
V = model.variables["Terminal voltage [V]"]
I = model.variables["Current [A]"]
parameter_values = pybamm.ParameterValues("OKane2022")
frequencies = np.logspace(-4, 3, 30)

# Time domain
I = 50 * 1e-3 # 50 mA
number_of_periods = 20
samples_per_period = 16


def current_function(t):
    return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t) # Sinusoidal Current Function as input - GEIS

parameter_values["Current function [A]"] = current_function

start_time = timer.time()

sim = pybamm.Simulation(
    model, parameter_values=parameter_values, solver=pybamm.ScipySolver()
)

impedances_time = []
for frequency in frequencies:
    # Solve
    period = 1 / frequency
    dt = period / samples_per_period
    t_eval = np.array(range(0, 1 + samples_per_period * number_of_periods)) * dt
    sol = sim.solve(t_eval, inputs={"Frequency [Hz]": frequency})
    # Extract final two periods of the solution
    time = sol["Time [s]"].entries[-3 * samples_per_period - 1 :]
    current = sol["Current [A]"].entries[-3 * samples_per_period - 1 :]
    voltage = sol["Voltage [V]"].entries[-3 * samples_per_period - 1 :]
    # FFT
    current_fft = fft(current)
    voltage_fft = fft(voltage)
    # Get index of first harmonic
    idx = np.argmax(np.abs(current_fft))
    impedance = -voltage_fft[idx] / current_fft[idx]
    impedances_time.append(impedance)

end_time = timer.time()
time_elapsed = end_time - start_time
print("Time domain method: ", time_elapsed, "s")


fig, ax = plt.subplots()
data = np.array(impedances_time)
ax.plot(data.real, -data.imag, "o")
ax_max = max(data.real.max(), -data.imag.max()) * 1.1
plt.axis([0, ax_max, 0, ax_max])
plt.gca().set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$Z_\mathrm{Re}$ [Ohm]")
ax.set_ylabel(r"$-Z_\mathrm{Im}$ [Ohm]")
plt.show()

# Define 2RC ECM model function
def impedance_2rc(f, R0, R1, C1, R2, C2):
    omega = 2 * np.pi * f
    Z1 = R1 / (1 + 1j * omega * R1 * C1)
    Z2 = R2 / (1 + 1j * omega * R2 * C2)
    return R0 + Z1 + Z2

# We need to fit real and imaginary parts simultaneously.
# Create a helper function that returns concatenated real and imag parts.
def fit_func(f, R0, R1, C1, R2, C2):
    Z = impedance_2rc(f, R0, R1, C1, R2, C2)
    return np.concatenate([Z.real, Z.imag])

# Prepare data for fitting
f_data = frequencies 
Z_data = np.array(impedances_time)

# Concatenate real and imag parts of measured impedance as target data
Z_meas = np.concatenate([Z_data.real, Z_data.imag])

# Initial guess for parameters: R0, R1, C1, R2, C2
initial_guess = [0.01, 0.01, 1e-3, 0.01, 1e-3]

# Curve fit
params_opt, params_cov = curve_fit(
    fit_func,
    f_data,
    Z_meas,
    p0=initial_guess,
    bounds=(0, np.inf),  
    maxfev=10000
)

R0, R1, C1, R2, C2 = params_opt
print(f"Fitted parameters:\n R0={R0:.4f} Ohm, R1={R1:.4f} Ohm, C1={C1:.6f} F, R2={R2:.4f} Ohm, C2={C2:.6f} F")

# Plot results
Z_fit = impedance_2rc(f_data, *params_opt)

plt.figure(figsize=(8, 6))
plt.plot(Z_data.real, -Z_data.imag, 'o', label='Measured')
plt.plot(Z_fit.real, -Z_fit.imag, '-', label='2RC fit')
plt.xlabel('Real(Z) [Ohm]')
plt.ylabel('-Imag(Z) [Ohm]')
plt.title('Nyquist plot with 2RC fit')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
