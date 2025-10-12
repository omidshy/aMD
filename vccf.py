"""
----------------------------------------------------------------------------------------
vccf.py is a code for calculating particles' velocity cross-correlation functions (VACF)
from molecular dynamics simulations.

Open-source free software under GNU GPL v3
Copyright (C) 2024-2025 Omid Shayestehpour

Please cite: J. Chem. Phys. 161, 134505 (2024) (DOI 10.1063/5.0232631).
----------------------------------------------------------------------------------------
"""

import os
import argparse
import numpy as np
import pandas as pd
import freud
from itertools import islice
from tqdm import trange
from utils import plot_results
from vacf import acf

# --------------------------------------------------------
# Parse command-line arguments
def parse_arguments():

    parser = argparse.ArgumentParser(
        prog="vccf.py",
        description="Calculating particles' velocity cross-correlation functions."
    )

    parser.add_argument(
        'data_1',
        # The 'datafile' attribute in the returned 'args' will be an open file object
        type=argparse.FileType('r'),
        help='Name of the data file for the first particle type. '
             'Note: the data file should have space-separated columns of '
             'position vector components (x, y, z) and velocity vector components (vx, vy, vz), '
             'for each line (particle): x  y  z  vx  vy  vz'
             'The order of particles in each frame (i.e. step) should be identical. '
             'The total number of lines should be equal to the (number of steps * number of particles).'
    )
    parser.add_argument(
        'data_2',
        type=argparse.FileType('r'),
        help='Name of the data file for the second particle type.'
    )
    parser.add_argument(
        '-r', '--rmax',
        type=float,
        required=True,
        help='The radial cutoff for identifying the neighboring particles '
             '(in the same unit as the coordinates data).'
    )
    parser.add_argument(
        '-b', '--boxsize',
        type=float,
        nargs='+',
        required=True,
        help='Simulation box dimensions: lx ly lz as 3 space-separated numbers '
             '(in the same unit as the coordinates data).'
    )
    parser.add_argument(
        '-s', '--steps',
        type=int,
        required=True,
        help='Number of steps to read from the data files.'
    )
    parser.add_argument(
        '-n1', '--particles_1',
        type=int,
        required=True,
        help='Number of particles of the first type in each frame of the data files.'
    )
    parser.add_argument(
        '-n2', '--particles_2',
        type=int,
        required=True,
        help='Number of particles of the second type in each frame of the data files.'
    )
    parser.add_argument(
        '-t', '--timestep',
        type=float,
        required=True,
        help='Physical timestep between successive data points in [ps].'
    )
    parser.add_argument(
        '--same',
        action='store_true',
        help='Add this option if the first and second particle types are the same, '
             'i.e., cross-correlation between distinct same-type particles.'
    )

    args = parser.parse_args()

    return args


# VCCF using a neighbors list
def ccf(Avel, Bvel, Apos, Bpos, rmax, box, same=False):
    """
    Computes the velocity cross-correlation function (VCCF) for a set of particles,
    which is a measure of how the velocity of a particle :math:`i` at a given time
    is correlated with the velocity of particles :math:`j` at a later time.

    Parameters
    ----------
    Avel : np.ndarray
        A 3D NumPy array of particle velocities of shape (N, 3, M) where N is
        the number of particles of type :math:`i`, 3 represents the x, y, z dimensions,
        and M is the number of time steps.
    Bvel : np.ndarray
        A 3D NumPy array of particle velocities of shape (N, 3, M) where N is
        the number of particles of type :math:`j`, 3 represents the x, y, z dimensions,
        and M is the number of time steps.
    Apos : np.ndarray
        A 3D NumPy array of particle positions of shape (N, 3, M) where N is
        the number of particles of type :math:`i`, 3 represents the x, y, z dimensions,
        and M is the number of time steps.
    Bpos : np.ndarray
        A 3D NumPy array of particle positions of shape (N, 3, M) where N is
        the number of particles of type :math:`j`, 3 represents the x, y, z dimensions,
        and M is the number of time steps.
    rmax : float
        The cutoff radius around an :math:`i` particle within which the :math:`j` particles
        considered as neighbors.
    box : list of floats
        Length of the 3 cell vectors of the simulation box.
    same : bool
        Are particles :math:`i` and :math:`j` the same type?

    Returns
    -------
    crosscorrelation : np.ndarray
        A 1D NumPy array representing the average VCCF of all :math:`i` particles and
        a neighboring :math:`j` particle.
    crosscorrelation_avg : np.ndarray
        A 1D NumPy array representing the average VCCF of all :math:`i` particles averaged
        over all neighboring :math:`j` particle.
    num_nb : float
        Average number of :math:`j` particles around an :math:`i` particle.
    """
    lag = Avel.shape[2] // 10
    steps = Avel.shape[2] - lag

    numAs = Avel.shape[0]
    numBs = Bvel.shape[0]

    box = freud.box.Box.cube(box[0])

    num_nb = np.zeros(numAs, dtype=int)
    sum_steps, sum_steps_avg  = np.zeros(lag, dtype=np.float32), np.zeros(lag, dtype=np.float32)

    for step in trange(steps, ncols=100, desc='Progress'):

        aq = freud.locality.AABBQuery(box, Bpos[:, :, step])
        '''
        Return pairs as tuples of the form (query_point_index, point_index, distance).
        For instance, a pair (1, 3, 2.0) would indicate that points[3] (B[3]) is one of the
        neighbors for query_points[1] (A[1]), and that they are separated by a distance of 2.0
        '''
        neighbors = [ pair for pair in aq.query(Apos[:, :, step], dict(r_max=rmax, exclude_ii=True if same else False)) ]
        neighbors = [ [pair[1] for pair in neighbors if pair[0] == index] for index in range(numAs) ]

        As, As_avg = np.zeros((numAs, lag), dtype=np.float32), np.zeros((numAs, lag), dtype=np.float32)

        for i in range(numAs):

            sumBs = np.zeros(lag, dtype=np.float32)

            if neighbors[i]:

                for j in neighbors[i]:

                    X = (Avel[i, 0, step]) * (Bvel[j, 0, step:step+lag])
                    Y = (Avel[i, 1, step]) * (Bvel[j, 1, step:step+lag])
                    Z = (Avel[i, 2, step]) * (Bvel[j, 2, step:step+lag])

                    # Sum over all B neighbors
                    sumBs += X + Y + Z

                As[i] = sumBs
                As_avg[i] = sumBs / len(neighbors[i])

            num_nb[i] += len(neighbors[i])

        # Average over all A particles
        sum_steps += np.mean(As, axis=0)
        sum_steps_avg += np.mean(As_avg, axis=0)

    # Average over all time origins (steps)
    crosscorrelation = sum_steps / steps
    crosscorrelation_avg = sum_steps_avg / steps

    # Calculate a normalization factor using the mean square velocities of the A and B particles
    normFactor = np.sqrt(
    np.mean(np.mean(np.sum(np.square(Avel), axis=1), axis=1)) * 
    np.mean(np.mean(np.sum(np.square(Bvel), axis=1), axis=1))
    )

    # return (crosscorrelation / normFactor), (crosscorrelation_avg / normFactor), (num_nb / steps)
    return crosscorrelation/normFactor, crosscorrelation_avg/normFactor, num_nb/steps

# -----------------------------------------------------
def main():
    # Parse the command-line arguments
    args = parse_arguments()

    # Print the parsed command-line arguments
    print(f"Using data files: {args.data_1.name}, {args.data_2.name}")
    print(f"Number of particles of type A: {args.particles_1}, type B: {args.particles_2}")
    print(f"Radial cutoff: {args.rmax}")
    print(f"Simulation box dimensions: {args.boxsize}")
    print(f"Number of steps to read: {args.steps}")
    print(f"Timestep: {args.timestep} ps")

    # Generate a time array
    end_step = args.steps * args.timestep
    time_array = np.linspace(0, end_step, num=args.steps, dtype=float, endpoint=False)

    # Read particle velocities and positions from data files
    print('\nReading the data files')
    pos_1 = np.zeros((args.particles_1, 3, args.steps), dtype=float)
    vel_1 = np.zeros((args.particles_1, 3, args.steps), dtype=float)
    with args.data_1 as file:
        for step in trange(args.steps, ncols=100, desc='Progress'):
            frame = [x.strip() for x in islice(file, args.particles_1)]
            for i, line in enumerate(frame):
                particle = list(map(float, line.split()))
                pos_1[i, :, step] = particle[:3]
                vel_1[i, :, step] = particle[3:]

    pos_2 = np.zeros((args.particles_2, 3, args.steps), dtype=float)
    vel_2 = np.zeros((args.particles_2, 3, args.steps), dtype=float)
    with args.data_2 as file:
        for step in trange(args.steps, ncols=100, desc='Progress'):
            frame = [x.strip() for x in islice(file, args.particles_2)]
            for i, line in enumerate(frame):
                particle = list(map(float, line.split()))
                pos_2[i, :, step] = particle[:3]
                vel_2[i, :, step] = particle[3:]

    # Compute correlation functions
    print('\nCalculating velocity correlation functions')
    vacf_1 = acf(vel_1, 0.1) # using 10% of the total length as max lag time (increase for shorter trajectories)
    vccf_1, avg_vccf_1, num_nb_1 = ccf(
        vel_1,
        vel_2,
        pos_1,
        pos_2,
        args.rmax,
        args.boxsize,
        args.same
        )

    if not args.same:
        vacf_2 = acf(vel_2, 0.1)
        vccf_2, avg_vccf_2, num_nb_2 = ccf(
            vel_2,
            vel_1,
            pos_2,
            pos_1,
            args.rmax,
            args.boxsize,
            args.same
            )

    # Create a directory to save the results
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the velocity cross(auto)-correlation functions
    norm_acf = vacf_1 / vacf_1[0]
    df = pd.DataFrame({"time [ps]" : time_array[:norm_acf.shape[0]], "VACF" : norm_acf[:]})
    file = os.path.join("results", "vacf_A.csv")
    df.to_csv(file, index=False)

    df = pd.DataFrame({"time [ps]" : time_array[:vccf_1.shape[0]], "VCCF" : vccf_1[:]})
    file = os.path.join("results", "vccf_AB.csv")
    df.to_csv(file, index=False)

    df = pd.DataFrame({"time [ps]" : time_array[:avg_vccf_1.shape[0]], "VCCF" : avg_vccf_1[:]})
    file = os.path.join("results", "avg_vccf_AB.csv")
    df.to_csv(file, index=False)

    if not args.same:
        norm_acf = vacf_2 / vacf_2[0]
        df = pd.DataFrame({"time [ps]" : time_array[:norm_acf.shape[0]], "VACF" : norm_acf[:]})
        file = os.path.join("results", "vacf_B.csv")
        df.to_csv(file, index=False)

        df = pd.DataFrame({"time [ps]" : time_array[:vccf_2.shape[0]], "VCCF" : vccf_2[:]})
        file = os.path.join("results", "vccf_BA.csv")
        df.to_csv(file, index=False)

        df = pd.DataFrame({"time [ps]" : time_array[:avg_vccf_2.shape[0]], "VCCF" : avg_vccf_2[:]})
        file = os.path.join("results", "avg_vccf_BA.csv")
        df.to_csv(file, index=False)

    # Write the coordination numbers (i.e. average number of neighbors) to a file
    file = os.path.join("results", "neighbour_counts.txt")
    with open(file, 'w') as out:
        out.write('The average number of j particles in the 1st shell of i particles (i-j):' + '\n\n')
        out.write(f"  {'A-B':>12} = {np.round(np.mean(num_nb_1), 3):>8}")
        out.write(f"  Particles type B around type A\n")
        if not args.same:
            out.write(f"  {'B-A':>12} = {np.round(np.mean(num_nb_2), 3):>8}")
            out.write(f"  Particles type A around type B\n")

    # Plot correlation functions
    if args.same:
        norm_acf_1 = vacf_1 / vacf_1[0]
        data = [
            {
            'x': time_array[:norm_acf_1.shape[0]],
            'y': norm_acf_1[:],
            'label':'VACF A', 'linestyle':'dashed', 'color':'gray'
            },
            {
            'x': time_array[:vccf_1.shape[0]],
            'y': vccf_1[:],
            'label':'VCCF A-B', 'linestyle':'solid', 'color':'purple'
            }
        ]
        labels = {'x': 'Time [ps]','y': '⟨v(0).v(t)⟩'}

        plot_results(data, labels)

    else:
        norm_acf_1 = vacf_1 / vacf_1[0]
        norm_acf_2 = vacf_2 / vacf_2[0]
        data = [
            {
            'x': time_array[:norm_acf_1.shape[0]],
            'y': norm_acf_1[:],
            'label':'VACF A', 'linestyle':'dashed', 'color':'gray'
            },
            {
            'x': time_array[:vccf_1.shape[0]],
            'y': vccf_1[:],
            'label':'VCCF A-B', 'linestyle':'solid', 'color':'purple'
            },

            {
            'x': time_array[:norm_acf_2.shape[0]],
            'y': norm_acf_2[:],
            'label':'VACF B', 'linestyle':'dashed', 'color':'orange'
            },
            {
            'x': time_array[:vccf_2.shape[0]],
            'y': vccf_2[:],
            'label':'VCCF B-A', 'linestyle':'solid', 'color':'green'
            }
        ]
        labels = {'x': 'Time [ps]','y': '⟨v(0).v(t)⟩'}

        plot_results(data, labels)

if __name__ == "__main__":
    main()