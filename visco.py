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

# Define ACF function
def acf(x, pb):
    N = Time.shape[0]
    size = int(N/2)

    autocorrelation = np.zeros(size, dtype=float)
    for shift in rng(size, pb):
            autocorrelation[shift] = np.mean( (x[:N-shift]) * (x[shift:]) )

    return autocorrelation

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get name of the data file
def getfile():
    global data_file
    data_file = input("\nEnter the path to the pressure tensor data file\nNote: the pressure tensor file should have space-separated columns (of the order Pxx, Pyy, Pzz, Pxy, Pxz, Pyz) and units of (atm, bar or Pa): ").strip()
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

# Get number of steps to be used for ACF calculation
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
    start_step = input('\nFrom which time step to start processing the data: [0] ').strip()
    if start_step.isnumeric():
        start_step = int(start_step)
    else:
        print("Your input is not valid!!")
        return getstartstep()

# Get the timestep of the simulation (assuming pressure data is collected in each timestep)
def gettimestep():
    global timestep
    timestep = input('\nEnter the physical timestep between the successive pressure data in your file [ps]: ').strip()
    try:
        timestep = float(timestep)
        if timestep < 0:
            print("timestep must be a positive number!")
            return gettimestep()
    except ValueError:
        print("Your input is not valid!!")

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
    GKtype = input('\nUse only 3 off-diagonal components of the P tensor (Pxy,Pxz,Pyz) [0],\ninclude the diagonal elements using (Pxx-Pyy)/2 [1],\ninclude the diagonal elements using 4/3*(Pxx-((Pxx+Pyy+Pzz)/3)) [2]: ').strip()
    if GKtype.isnumeric():
        GKtype = int(GKtype)
        if GKtype != 0 and GKtype != 1 and GKtype != 2:
            print("Your input is not valid!!")
            return gettype()
    else:
        print("Your input is not valid!!")
        return gettype()

# Reduce down the data by the given factor
def getfactor():
    global factor
    factor = input('\nUse pressure values from every (n) timesteps for viscosity calculation: [10] ').strip()
    if factor.isnumeric():
        factor = int(factor)
    else:
        print("Your input is not valid!!")
        return getfactor()

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

# Save plots as csv files?
def saveplot():
    global save_plot
    save_plot = input('\nSave plots as csv files? Yes[1]/No[0]: ').strip()
    if save_plot.isnumeric():
        save_plot = int(save_plot)
        if save_plot != 0 and save_plot != 1:
            print("Your input is not valid!!")
            return saveplot()
    else:
        print("Your input is not valid!!")
        return saveplot()

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
getfactor()
saveplot()
if save_plot == 1:
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
# Time = []
Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = [], [], [], [], [], []

# Read the pressure tensor elements from data file
print('\nReading the pressure tensor data file')
with open(data_file, "r") as file:
    for l in rng(num_steps, pbar):
        line = file.readline()
        step = list(map(float, line.split()))
        # Time.append(step[0])
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

# Convert created lists to numpy arrays starting from given step
print('\nPreparing pressure tensor arrays')
# Time = np.array(Time[:num_steps-start_step])
Pxx = np.array(Pxx[start_step:])
Pyy = np.array(Pyy[start_step:])
Pzz = np.array(Pzz[start_step:])
Pxy = np.array(Pxy[start_step:])
Pxz = np.array(Pxz[start_step:])
Pyz = np.array(Pyz[start_step:])

# Reduce down the data by the factor given
# 1 uses all of the gathered data, 10 uses one 10th
# This can be used to speed up getting the autocorrelation function
if factor > 1:
    print('\nReducing down the data by the given factor: {}'.format(factor))
    Time = Time[::factor]
    Pxx = Pxx[::factor]
    Pyy = Pyy[::factor]
    Pzz = Pzz[::factor]
    Pxy = Pxy[::factor]
    Pxz = Pxz[::factor]
    Pyz = Pyz[::factor]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Viscosity from Einstein relation
if GKorEn == 1:

    print('\nCalculating the viscosity from Einstein relation')

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
    print('-'*30 + '\n' + 'Viscosity(Einstein): ' + str(viscosity[-1] * 1000) + ' (mPa.s)')

    # Plot the evolution of the viscosity in time
    if viscosity_plot == 1:
        pylab.figure()
        pylab.plot(Time[:], viscosity[:]*1000, label='Viscosity')
        pylab.xlabel('Time (ps)')
        pylab.ylabel('Viscosity (cP)')
        pylab.legend()
        pylab.show()

    # Save the evolution of the viscosity in time as a csv file
    if save_plot == 1:
        Time = Time[::save_step]
        viscosity = viscosity[::save_step]
        df = pd.DataFrame({"time(ps)" : Time[:], "viscosity(Pa.s)" : viscosity[:]})
        df.to_csv("viscosity_Einstein.csv", index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Viscosity from Green-Kubo relation
elif GKorEn == 0:
    # Calculate the shear components of the pressure tensor
    if GKtype == 1:
        Pxxyy = (Pxx - Pyy) / 2
        Pyyzz = (Pyy - Pzz) / 2
        Pxxzz = (Pxx - Pzz) / 2

    elif GKtype == 2:
        Pxy *= 2
        Pxz *= 2
        Pyz *= 2
        Pxxyz = (4/3) * (Pxx - ((Pxx + Pyy + Pzz) / 3))
        Pyxyz = (4/3) * (Pyy - ((Pxx + Pyy + Pzz) / 3))
        Pzxyz = (4/3) * (Pzz - ((Pxx + Pyy + Pzz) / 3))

    # Calculate the ACFs
    # Note that we use max 1/2 the length of the data as correlation time
    # because beyond that there are not enough available time windows to be accurate
    print('\nCalculating ACFs of the pressure tensor')

    def P1():
        Pxy_acf = acf(Pxy, pbar)
        with open("Pxy_acf", "wb") as file:
            np.save(file, Pxy_acf)

    def P2():
        Pxz_acf = acf(Pxz, 0)
        with open("Pxz_acf", "wb") as file:
            np.save(file, Pxz_acf)

    def P3():
        Pyz_acf = acf(Pyz, 0)
        with open("Pyz_acf", "wb") as file:
            np.save(file, Pyz_acf)

    if GKtype == 1:

        def P4():
            Pxxyy_acf = acf(Pxxyy, 0)
            with open("Pxxyy_acf", "wb") as file:
                np.save(file, Pxxyy_acf)

        def P5():
            Pyyzz_acf = acf(Pyyzz, 0)
            with open("Pyyzz_acf", "wb") as file:
                np.save(file, Pyyzz_acf)

        def P6():
            Pxxzz_acf = acf(Pxxzz, 0)
            with open("Pxxzz_acf", "wb") as file:
                np.save(file, Pxxzz_acf)

    elif GKtype == 2:

        def P4():
            Pxxyz_acf = acf(Pxxyz, 0)
            with open("Pxxyz_acf", "wb") as file:
                np.save(file, Pxxyz_acf)

        def P5():
            Pyxyz_acf = acf(Pyxyz, 0)
            with open("Pyxyz_acf", "wb") as file:
                np.save(file, Pyxyz_acf)

        def P6():
            Pzxyz_acf = acf(Pzxyz, 0)
            with open("Pzxyz_acf", "wb") as file:
                np.save(file, Pzxyz_acf)

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

    elif GKtype == 2:
        # Run ACF calculations in parallel
        runInParallel(P1, P2, P3, P4, P5, P6)

        # Read back the saved numpy arrays from file to an array
        with open("Pxy_acf", "rb") as file:
            Pxy_acf = np.load(file)

        with open("Pxz_acf", "rb") as file:
            Pxz_acf = np.load(file)

        with open("Pyz_acf", "rb") as file:
            Pyz_acf = np.load(file)

        with open("Pxxyz_acf", "rb") as file:
            Pxxyz_acf = np.load(file)

        with open("Pyxyz_acf", "rb") as file:
            Pyxyz_acf = np.load(file)

        with open("Pzxyz_acf", "rb") as file:
            Pzxyz_acf = np.load(file)

    # Calculate the average of ACFs
    if GKtype == 0:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3

    elif GKtype == 1:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf + Pxxyy_acf + Pyyzz_acf + Pxxzz_acf) / 6

    elif GKtype == 2:
        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf + Pxxyz_acf + Pyxyz_acf + Pzxyz_acf) / 10

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
    if save_plot == 1:
        norm_avg_acf = avg_acf / avg_acf[0]
        df = pd.DataFrame({"time (ps)" : Time[:avg_acf.shape[0]], "ACF" : norm_avg_acf[:]})
        df.to_csv("avg_acf.csv", index=False)

    # Integrate the average ACF to get the viscosity using Green-Kubo relation
    timestep = (Time[1] - Time[0]) * 10**(-12)
    gk_int = integrate.cumtrapz(y=avg_acf, dx=timestep, initial=0)
    viscosity = gk_int * (volume_avg / kBT)
    print('-'*40 + '\n' + 'Viscosity(Green-Kubo): ' + str(viscosity[-1] * 1000) + ' (mPa.s)')

    # Plot the time evolution of the viscosity estimate
    if viscosity_plot == 1:
        pylab.figure()
        pylab.plot(Time[:viscosity.shape[0]], viscosity[:]*1000, label='Viscosity')
        pylab.xlabel('Time (ps)')
        pylab.ylabel('Viscosity (cP)')
        pylab.legend()
        pylab.show()

    # Save running integral of the viscosity as csv file
    if save_plot == 1:
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

    elif GKtype == 2:
        os.remove('Pxxyz_acf')
        os.remove('Pyxyz_acf')
        os.remove('Pzxyz_acf')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set the end time and print the execution time
t_end = time.perf_counter()
wall_time = convert_time(t_end - t_start)
print(40*'-')
print("wall-time: " + str(wall_time))
