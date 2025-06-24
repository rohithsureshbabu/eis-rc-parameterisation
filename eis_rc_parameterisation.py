# %%

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import time as timer
from scipy.fft import fft
from scipy.optimize import curve_fit

pybamm.set_logging_level("INFO")

# Fresh Cell Setup
model = pybamm.lithium_ion.SPMe(
    {   
        "thermal": "lumped", 
        "surface form": "differential",  
        "SEI": "solvent-diffusion limited",
        "lithium plating": "reversible",
    }
)
parameter_values = pybamm.ParameterValues("OKane2022")

# EIS settings
frequencies = np.logspace(-4, 3, 20)  
I = 50e-3  
number_of_periods = 10  
samples_per_period = 16

# Current function for EIS
def current_function(t):
    return I * pybamm.sin(2 * np.pi * pybamm.InputParameter("Frequency [Hz]") * t)

parameter_values["Current function [A]"] = current_function

solver=pybamm.CasadiSolver()

# Fresh cell EIS
print("Running EIS on fresh cell...")
impedances_time = []
sim = pybamm.Simulation(
    model, 
    parameter_values=parameter_values, 
    solver=solver
)

for frequency in frequencies:
    try:
        period = 1 / frequency
        dt = period / samples_per_period
        t_eval = np.arange(0, number_of_periods * period + dt, dt)
        
        sol = sim.solve(t_eval, inputs={"Frequency [Hz]": frequency})
        
        # Steady-state extraction
        steady_state_idx = -2 * samples_per_period - 1  # Last 2 periods
        current = sol["Current [A]"].entries[steady_state_idx:]
        voltage = sol["Voltage [V]"].entries[steady_state_idx:]
        
        # FFT analysis
        current_fft = fft(current)
        voltage_fft = fft(voltage)
        idx = np.argmax(np.abs(current_fft[1:])) + 1  # Skip DC component
        impedance = -voltage_fft[idx] / current_fft[idx]
        impedances_time.append(impedance)
        
        print(f"Completed {frequency:.2e} Hz")
    except Exception as e:
        print(f"Failed at {frequency:.2e} Hz: {str(e)}")
        impedances_time.append(np.nan + 1j*np.nan)



# 2RC and Warburg model
def impedance_2rc_warburg(f, R0, R1, C1, R2, C2, sigma):
    omega = 2 * np.pi * f
    j_omega = 1j * omega
    Z1 = R1 / (1 + j_omega * R1 * C1)
    Z2 = R2 / (1 + j_omega * R2 * C2)
    ZW = sigma * (1 - 1j) / np.sqrt(omega)  # Warburg impedance
    return R0 + Z1 + Z2 + ZW

# Concatenate real and imag parts for joint fitting
def fit_func(f, R0, R1, C1, R2, C2, sigma):
    Z = impedance_2rc_warburg(f, R0, R1, C1, R2, C2, sigma)
    return np.concatenate([Z.real, Z.imag])

initial_guess = [0.01, 0.04, 1e-2, 0.02, 1e-3, 0.01]  
valid_freqs = frequencies[2:]
valid_Z = np.array(impedances_time[2:])
Z_meas = np.concatenate([np.real(valid_Z), np.imag(valid_Z)])

# Fitting
params_opt, _ = curve_fit(
    fit_func,
    valid_freqs,
    Z_meas,
    p0=initial_guess,
    bounds=(0, np.inf),
    maxfev=10000
)


R0, R1, C1, R2, C2, sigma = params_opt

print("Fitted ECM parameters (Fresh Cell with Warburg):")
print(f"R0 = {R0:.4f} Ohm")
print(f"R1 = {R1:.4f} Ohm, C1 = {C1:.6f} F")
print(f"R2 = {R2:.4f} Ohm, C2 = {C2:.6f} F")
print(f"Sigma (Warburg) = {sigma:.6f} Ohm·s^0.5")

# Fit and Real ECM plot
Z_fit = impedance_2rc_warburg(frequencies, *params_opt)

plt.figure(figsize=(8, 6))
plt.plot(np.real(impedances_time), -np.imag(impedances_time), "o", label="Measured (Fresh)")
plt.plot(np.real(Z_fit), -np.imag(Z_fit), "-", label="2RC + Warburg Fit")
plt.xlabel("Real(Z) [Ohm]")
plt.ylabel("-Imag(Z) [Ohm]")
plt.title("Nyquist Plot with 2RC + Warburg ECM Fit")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

# %%

# Aged Cell
model_aged = pybamm.lithium_ion.SPMe(
    {   
        "thermal": "lumped",  
        "surface form": "differential", 
        "SEI": "solvent-diffusion limited",
        "lithium plating": "reversible",
    }
)


experiment_aged = pybamm.Experiment([
    "Discharge at 0.5 C until 3.0 V",  
    "Rest for 1 hour",  
    "Charge at 2 C until 4.2 V",  
    "Hold at 4.2 V until C/20",  
    "Rest for 1 hour"
]*100) 

param_aged = parameter_values.copy()
param_aged.update({
    "SEI kinetic rate constant [m.s-1]": 1e-12,
    "Initial inner SEI thickness [m]": 2e-9,
})

print("\nRunning aging simulation...")
sim = pybamm.Simulation(
        model_aged,
        parameter_values=param_aged,
        experiment=experiment_aged,
        solver=pybamm.CasadiSolver(mode='safe without grid'),
    )

solution_aged = sim.solve()
print("Aging simulation complete. Last voltage:", solution_aged["Voltage [V]"](solution_aged.t[-1]))


# Aged cell EIS
print("\nRunning EIS on aged cell...")
param_aged["Current function [A]"] = current_function

new_model = model_aged.set_initial_conditions_from(solution_aged, inplace=False)

sim_aged2 = pybamm.Simulation(
    new_model,
    parameter_values=param_aged,
    solver=solver
)

impedances_aged = []
for frequency in frequencies:
    try:
        period = 1 / frequency
        dt = period / samples_per_period
        t_eval = np.arange(0, number_of_periods * period + dt, dt)

        sol2 = sim_aged2.solve(
            t_eval,
            inputs={"Frequency [Hz]": frequency}
        )

        steady_state_idx = -2 * samples_per_period - 1
        current2 = sol2["Current [A]"].entries[steady_state_idx:]
        voltage2 = sol2["Voltage [V]"].entries[steady_state_idx:]

        current_fft2 = fft(current2)
        voltage_fft2 = fft(voltage2)
        idx = np.argmax(np.abs(current_fft2[1:])) + 1
        impedance2 = -voltage_fft2[idx] / current_fft2[idx]
        impedances_aged.append(impedance2)

        print(f"Completed {frequency:.2e} Hz")
    except Exception as e:
        print(f"Failed at {frequency:.2e} Hz: {str(e)}")
        impedances_aged.append(np.nan + 1j*np.nan)

# Fresh and Aged cell EIS
if 'solution_aged' in locals():
    valid_idx = ~np.isnan(np.real(impedances_time)) & ~np.isnan(np.real(impedances_aged))
    freq_valid = frequencies[valid_idx]
    Z_fresh = np.array(impedances_time)[valid_idx]
    Z_aged = np.array(impedances_aged)[valid_idx]

    if len(freq_valid) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(np.real(Z_fresh), -np.imag(Z_fresh), "o-", label="Fresh Cell")
        plt.plot(np.real(Z_aged), -np.imag(Z_aged), "s-", label="Aged Cell (5 cycles)")
        plt.xlabel(r"$Z_\mathrm{Re}$ [Ohm]")
        plt.ylabel(r"$-Z_\mathrm{Im}$ [Ohm]")
        plt.title("Nyquist Plot: Fresh vs Aged Cell")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
    else:
        print("No valid impedance points to plot.")
else:
    print("Aged cell simulation not run; skipping Nyquist plot.")

