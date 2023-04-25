# --------------------------------------------------------------------------------------------
#
# visco.py is a code for calculating viscosity from molecular dynamics (MD) simulations.
# Copyright (C) 2021 Omid Shayestehpour
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
# 
# Calculation of viscosity using the Einstein or Green-Kubo expressions.
# Viscosity is computed from the integral of the elements of the pressure tensor
# (or their autocorrelation function) collected from MD simulations.
#
# Notice: the pressure tensor file should have space-separated columns 
# of the following order and units of [atm/bar/Pa]:
# Pxx, Pyy, Pzz, Pxy, Pxz, Pyz
#
# --------------------------------------------------------------------------------------------

import os
import time
import pylab
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit
from multiprocessing import Process
from tqdm import trange

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# convert timelapse from sec to H:M:S
def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

# Run given functions in parallel
def runInParallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

# Show progress-bar
def rng(r, pb):
    if pb == 1:
        rng = trange(r, ncols=100, desc='Progress')
    else:
        rng = range(r)
    return rng

# Define ACF
def acf(x, pb):
    N = x.shape[0]
    size = int(N//2)

    autocorrelation = np.zeros(size, dtype=float)
    for shift in rng(size, pb):
            autocorrelation[shift] = np.mean( (x[:N-shift]) * (x[shift:]) )

    return autocorrelation

# Define ACF using FFT
def acf_fft(data):
    N = data.shape[0]
    lag = N//2

    # Nearest size with power of 2 (for efficiency) to zero-pad the input data
    size = 2 ** np.ceil(np.log2(2*N - 1)).astype('int')

    # Compute the FFT
    FFT = np.fft.fft(data, size)

    # Get the power spectrum
    PWR = FFT.conjugate() * FFT

    # Calculate the autocorrelation from inverse FFT of the power spectrum
    CORR = np.fft.ifft(PWR)[:N].real

    autocorrelation = CORR / np.arange(N, 0, -1)

    return autocorrelation[:lag]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get name of the data file
def getfile():
    global data_file
    data_file = input("\nEnter the path to the pressure tensor data file\nNote: the pressure tensor file should have space-separated columns\n(of the order Pxx, Pyy, Pzz, Pxy, Pxz, Pyz) and units of (atm, bar or Pa): ").strip()
    try:
        with open(data_file, "r") as file:
            file.readline()
    except IOError as error:
        print('\n%s' % error)
        return getfile()

# Get the unit of pressure
def getunit():
    global p_unit
    p_unit = input('\nIn which units are the provided pressure data: Pa[0], Atm[1] or Bar[2]: ').strip()
    if p_unit.isnumeric():
        p_unit = int(p_unit)
        if p_unit != 0 and p_unit != 1 and p_unit != 2:
            print("Your input is not valid!!")
            return getunit()
    else:
        print("Your input is not valid!!")
        return getunit()

# Get the number of steps to be read from data file
def getnumsteps():
    global num_steps
    num_steps = input('\nNumber of steps to read from pressure tensor file: ').strip()
    if num_steps.isnumeric():
        num_steps = int(num_steps)
    else:
        print("Your input is not valid!!")
        return getnumsteps()

# From which time step to start processing the data
def getstartstep():
    global start_step
    start_step = input('\nFrom which time step start processing the data: [0] ').strip()
    if start_step.isnumeric():
        start_step = int(start_step)
    else:
        print("Your input is not valid!!")
        return getstartstep()

# Get the timestep of the saved pressure data
def gettimestep():
    global timestep
    timestep = input('\nEnter the physical timestep between two successive pressure data in your file [ps]: ').strip()
    try:
        timestep = float(timestep)
        if timestep < 0:
            print("timestep must be a positive number!")
            return gettimestep()
    except ValueError:
        print("Your input is not valid!!")
        return gettimestep()

# Get the temperature of the MD simulation
def gettemp():
    global temperature
    temperature = input('\nEnter the temperature of your MD simulation in [K]: ').strip()
    try:
        temperature = float(temperature)
        if temperature < 0:
            print("temperature must be a positive number!")
            return gettemp()
    except ValueError:
        print("Your input is not valid!!")
        return gettemp()

# Get the average volume of the simulation box
def getvolume():
    global volume_avg
    volume_avg = input('\nEnter the average volume of your simulation box in [A^3]: ').strip()
    try:
        volume_avg = float(volume_avg) * 10**(-30)
        if volume_avg < 0:
            print("volume must be a positive number!")
            return getvolume()
    except ValueError:
        print("Your input is not valid!!")
        return getvolume()

# Choose between Green-Kubo and Einstein expression
def getmethod():
    global GKorEn
    GKorEn = input('\nUse Green-Kubo[0] or Einstein[1] expression for viscosity calculation: ').strip()
    if GKorEn.isnumeric():
        GKorEn = int(GKorEn)
        if GKorEn != 0 and GKorEn != 1:
            print("Your input is not valid!!")
            return getmethod()
    else:
        print("Your input is not valid!!")
        return getmethod()

# For Green-Kubo expression of viscosity 
# choose between different off-diagonal components of the pressure tensor
def gettype():
    global GKtype
    GKtype = input('\nUse only 3 off-diagonal components of the P tensor (i.e. Pxy,Pxz,Pyz) [0],\ninclude the diagonal elements using e.g. (Pxx-Pyy)/2 [1]: ').strip()
    if GKtype.isnumeric():
        GKtype = int(GKtype)
        if GKtype != 0 and GKtype != 1:
            print("Your input is not valid!!")
            return gettype()
    else:
        print("Your input is not valid!!")
        return gettype()

# Plot the auto correlation functions?
def acfplot():
    global acf_plot
    acf_plot = input('\nPlot the auto correlation functions? Yes[1]/No[0]: ').strip()
    if acf_plot.isnumeric():
        acf_plot = int(acf_plot)
        if acf_plot != 0 and acf_plot != 1:
            print("Your input is not valid!!")
            return acfplot()
    else:
        print("Your input is not valid!!")
        return acfplot()

# Plot the time evolution of the viscosity estimate?
def viscosityplot():
    global viscosity_plot
    viscosity_plot = input('\nPlot the time evolution of the viscosity estimate? Yes[1]/No[0]: ').strip()
    if viscosity_plot.isnumeric():
        viscosity_plot = int(viscosity_plot)
        if viscosity_plot != 0 and viscosity_plot != 1:
            print("Your input is not valid!!")
            return viscosityplot()
    else:
        print("Your input is not valid!!")
        return viscosityplot()

# Save the time evolution of the viscosity in every n steps
def getsavestep():
    global save_step
    save_step = input('\nSave the time evolution of the viscosity in every n steps: [100] ').strip()
    if save_step.isnumeric():
        save_step = int(save_step)
    else:
        print("Your input is not valid!!")
        return getsavestep()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Collect some inputs from user
getfile()
getunit()
getnumsteps()
getstartstep()
gettimestep()
gettemp()
getvolume()
getmethod()

if GKorEn == 0:
    gettype()
    acfplot()

viscosityplot()
getsavestep()

# Show progress bars, 0 = disable 
pbar = 1

# Conversion ratio from atm(1)/bar(2) to Pa
if p_unit == 0:
    conv_ratio = 1
elif p_unit == 1:
    conv_ratio = 101325
elif p_unit == 2:
    conv_ratio = 100000

# Calculate the kBT value
kBT = Boltzmann * temperature

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set the start time
t_start = time.perf_counter()

# Initiate the pressure tensor component lists
Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = [], [], [], [], [], []

# Read the pressure tensor elements from data file
print('\nReading the pressure tensor data file')
with open(data_file, "r") as file:
    for l in rng(num_steps, pbar):
        line = file.readline()
        step = list(map(float, line.split()))
        Pxx.append(step[0]*conv_ratio)
        Pyy.append(step[1]*conv_ratio)
        Pzz.append(step[2]*conv_ratio)
        Pxy.append(step[3]*conv_ratio)
        Pxz.append(step[4]*conv_ratio)
        Pyz.append(step[5]*conv_ratio)

# Generate the time array
total_steps = num_steps - start_step
end_step = total_steps * timestep
Time = np.linspace(0, end_step, num=total_steps, endpoint=False)

# Convert created lists to numpy arrays starting from the given step
Pxx = np.array(Pxx[start_step:])
Pyy = np.array(Pyy[start_step:])
Pzz = np.array(Pzz[start_step:])
Pxy = np.array(Pxy[start_step:])
Pxz = np.array(Pxz[start_step:])
Pyz = np.array(Pyz[start_step:])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Viscosity from Einstein relation
if GKorEn == 1:

    Pxxyy = (Pxx - Pyy) / 2
    Pyyzz = (Pyy - Pzz) / 2

    # Calculate the viscosity from Einstein relation 
    # by integrating the components of the P tensor
    timestep = (Time[1] - Time[0]) * 10**(-12)

    Pxy_int = integrate.cumtrapz(y=Pxy, dx=timestep, initial=0)
    Pxz_int = integrate.cumtrapz(y=Pxz, dx=timestep, initial=0)
    Pyz_int = integrate.cumtrapz(y=Pyz, dx=timestep, initial=0)
    Pxxyy_int = integrate.cumtrapz(y=Pxxyy, dx=timestep, initial=0)
    Pyyzz_int = integrate.cumtrapz(y=Pyyzz, dx=timestep, initial=0)

    en_int = (Pxy_int**2 + Pxz_int**2 + Pyz_int**2 + Pxxyy_int**2 + Pyyzz_int**2) / 5

    viscosity = en_int[1:] * ( volume_avg / (2 * kBT * Time[1:] * 10**(-12)) )
    print('-'*40)
    print(f"\nViscosity: {round((viscosity[-1] * 1000), 2)} [mPa.s]  (Do not trust this value. You should fit a function to the running integral and take its limit)")

    # Plot the evolution of the viscosity in time
    if viscosity_plot == 1:
        pylab.figure()
        pylab.plot(Time[:viscosity.shape[0]], viscosity[:]*1000, label='Viscosity')
        pylab.xlabel('Time (ps)')
        pylab.ylabel('Viscosity (cP)')
        pylab.legend()
        pylab.show()

    # Save the evolution of the viscosity in time as a csv file
    Time = Time[::save_step]
    viscosity = viscosity[::save_step]
    df = pd.DataFrame({"time(ps)" : Time[:viscosity.shape[0]], "viscosity(Pa.s)" : viscosity[:]})
    df.to_csv("viscosity_Einstein.csv", index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Viscosity from Green-Kubo relation
elif GKorEn == 0:
    # Calculate the shear components of the pressure tensor
    if GKtype == 1:
        Pxxyy = (Pxx - Pyy) / 2
        Pyyzz = (Pyy - Pzz) / 2
        Pxxzz = (Pxx - Pzz) / 2

    # Calculate the ACFs
    def P1():
        Pxy_acf = acf_fft(Pxy)
        with open("Pxy_acf", "wb") as file:
            np.save(file, Pxy_acf)

    def P2():
        Pxz_acf = acf_fft(Pxz)
        with open("Pxz_acf", "wb") as file:
            np.save(file, Pxz_acf)

    def P3():
        Pyz_acf = acf_fft(Pyz)
        with open("Pyz_acf", "wb") as file:
            np.save(file, Pyz_acf)

    if GKtype == 1:

        def P4():
            Pxxyy_acf = acf_fft(Pxxyy)
            with open("Pxxyy_acf", "wb") as file:
                np.save(file, Pxxyy_acf)

        def P5():
            Pyyzz_acf = acf_fft(Pyyzz)
            with open("Pyyzz_acf", "wb") as file:
                np.save(file, Pyyzz_acf)

        def P6():
            Pxxzz_acf = acf_fft(Pxxzz)
            with open("Pxxzz_acf", "wb") as file:
                np.save(file, Pxxzz_acf)

    if GKtype == 0:
        # Run ACF calculations in parallel
        runInParallel(P1, P2, P3)

        # Read back the saved numpy arrays from file to an array
        with open("Pxy_acf", "rb") as file:
            Pxy_acf = np.load(file)

        with open("Pxz_acf", "rb") as file:
            Pxz_acf = np.load(file)

        with open("Pyz_acf", "rb") as file:
            Pyz_acf = np.load(file)

    elif GKtype == 1:
        # Run ACF calculations in parallel
        runInParallel(P1, P2, P3, P4, P5, P6)

        # Read back the saved numpy arrays from file to an array
        with open("Pxy_acf", "rb") as file:
            Pxy_acf = np.load(file)

        with open("Pxz_acf", "rb") as file:
            Pxz_acf = np.load(file)

        with open("Pyz_acf", "rb") as file:
            Pyz_acf = np.load(file)

        with open("Pxxyy_acf", "rb") as file:
            Pxxyy_acf = np.load(file)

        with open("Pyyzz_acf", "rb") as file:
            Pyyzz_acf = np.load(file)

        with open("Pxxzz_acf", "rb") as file:
            Pxxzz_acf = np.load(file)

    # Calculate the average of ACFs
    if GKtype == 0:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3

    elif GKtype == 1:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf + Pxxyy_acf + Pyyzz_acf + Pxxzz_acf) / 6

    # Plot the normalized average ACF
    if acf_plot == 1:
        norm_avg_acf = avg_acf / avg_acf[0]
        pylab.figure()
        pylab.plot(Time[:avg_acf.shape[0]], norm_avg_acf[:], label='Average')
        pylab.xlabel('Time (ps)')
        pylab.ylabel('ACF')
        pylab.legend()
        pylab.show()

    # Save the normalized average ACF as csv file
    norm_avg_acf = avg_acf / avg_acf[0]
    df = pd.DataFrame({"time (ps)" : Time[:avg_acf.shape[0]], "ACF" : norm_avg_acf[:]})
    df.to_csv("avg_acf.csv", index=False)

    # Integrate the average ACF to get the viscosity using Green-Kubo relation
    timestep = (Time[1] - Time[0]) * 10**(-12)
    gk_int = integrate.cumtrapz(y=avg_acf, dx=timestep, initial=0)
    viscosity = gk_int * (volume_avg / kBT)
    print('-'*40)
    print(f"\nViscosity: {round((viscosity[-1] * 1000), 2)} [mPa.s]  (Do not trust this value. You should fit a function to the running integral and take its limit)")

    # Plot the time evolution of the viscosity estimate
    if viscosity_plot == 1:
        pylab.figure()
        pylab.plot(Time[:viscosity.shape[0]], viscosity[:]*1000, label='Viscosity')
        pylab.xlabel('Time (ps)')
        pylab.ylabel('Viscosity (cP)')
        pylab.legend()
        pylab.show()

    # Save running integral of the viscosity as csv file
    Time = Time[::save_step]
    viscosity = viscosity[::save_step]
    df = pd.DataFrame({"time(ps)" : Time[:viscosity.shape[0]], "viscosity(Pa.s)" : viscosity[:]})
    df.to_csv("viscosity_GK.csv", index=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove extra files
    os.remove('Pxy_acf')
    os.remove('Pxz_acf')
    os.remove('Pyz_acf')

    if GKtype == 1:
        os.remove('Pxxyy_acf')
        os.remove('Pxxzz_acf')
        os.remove('Pyyzz_acf')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set the end time and print the execution time
t_end = time.perf_counter()
wall_time = convert_time(t_end - t_start)
print(40*'-')
print("wall-time: " + str(wall_time) + "\n")
