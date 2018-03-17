# -*- coding: utf-8 -*-
##############################################################################
# LMA: A Python Library for generating FT-spectra from
#         Molecular Dynamics Trajectories.
#
# Copyright 2017-2018
# Ruhr-University-Bochum Institute of biophysics and the Author
#
# Author: Matthias Massarczyk
#
# LMA is free software:
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
##############################################################################


import os, sys, shutil
import argparse
#import numpy as np

from lma_io import LMALoader
from lma_io import prep_dir
from lma_processing import LMAProcessor
from lma_out import OutputWriter


def run():
    #===================#
    #| Read user input |#
    #===================#
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='trajectory data file (None)')
    parser.add_argument('--top', type=str, default=None, help='additional topology file (None)(like pdb)')
    parser.add_argument('-w','--weighting', type=str, default=None, help='additional weighting file (None)')
    parser.add_argument('--temp', type=str, default='temp_lma', help='lma temp directory (temp_lma)')
    parser.add_argument('--out', type=str, default='lma_out', help='lma output directory (lma_out)')
    parser.add_argument('--dt', type=float, default=0.5, help='time step (0.5 fs)')
    parser.add_argument('--win', type=str, default='tukey', help='window function (tuckey/blackmanharris/None) (tukey)')
    parser.add_argument('--fscale', type=float, default=1.0, help='frequency scale factor (1.0)')
    parser.add_argument('--save_txt', action='store_true', help='store data as txt (True)')
    parser.add_argument('--save_npy', action='store_false', help='store data as npy (False)')
    parser.add_argument('--spec', action='store_false', help='store full ir-spectrum data (True)')
    parser.add_argument('--spec_plot', action='store_false', help='store full ir-spectrum plot (False)')
    parser.add_argument('--spec_plot_3d', action='store_false', help='store full ir-spectrum 3dplot (True)')
    parser.add_argument('--atom_spec', action='store_false', help='calculate and store atomic ir-spectrum data (True)')
    parser.add_argument('--atom_spec_plot', action='store_true', help='store atomic ir-spectrum plot (False)')
    parser.add_argument('--atom_spec_plot_3d', action='store_true', help='store atomic ir-spectrum 3dplot (True)')
    parser.add_argument('--mode_spec', action='store_false', help='calculate and store ir-mode data (True)')
    parser.add_argument('--mode_spec_plot', action='store_true', help='store ir-mode plot (False)')
    parser.add_argument('--mode_spec_plot_3d', action='store_true', help='store ir-mode 3dplot (True)')
    parser.add_argument('--bpf', type=str, default='off', help='apply band-pass filter')
    parser.add_argument('--bpf_range', type=float, nargs="+", default=[0.0, 4000.0], help='defines the range(s) of the band-pass filter (0, 4000)')
    parser.add_argument('--bpf_intra_fit', action='store_false', help='intra fitting within band-pass filter')
    parser.add_argument('--gaussian_filter', type=int, default=0, help='value to apply gaussian filter on spectral data (0)')
    parser.add_argument('--force_recalc', action='store_true', help='do not use saved data in temp (False)')
    parser.add_argument('--time_range', type=int, nargs="+", default=[0.0, 0.0], help='defines the range(s) within the trajectory to use for analysis (0.0, 0.0)')
    parser.add_argument('--atom_groups', type=str, nargs="+", default=["-1"], help='defines group(s) within atom analysis (-1)')
    parser.add_argument('--mode_groups', type=str, nargs="+", default=["-1"], help='defines group(s) within mode analysis (-1)')
    parser.add_argument('--time_resolved', type=str, default='off', help='apply time resolving')
    parser.add_argument('--time_window_length', type=str, default='1', help='defines time window length in slicing mode (1)')
    parser.add_argument('--n_time_windows', type=int, default=1, help='use n time windows in slice mode (1)')
    parser.add_argument('--time_windows_3D', action='store_false', help='plot time resolved 3d spectrum (True)')
    parser.add_argument('--plot_jpg', action='store_true', help='plot image as jpg (False)')
    parser.add_argument('--bpf_length', type=int, default=10000, help='max length of bpf trajectory (10000)')
    parser.add_argument('--bpf_part', type=float, default=0.30, help='bpf trajectory strip out (0.25)')
    parser.add_argument('--pos_data_scale', type=str, default='nm', help='position data is in nm or [A]ngstr. (nm)')
    parser.add_argument('--guess_bonds_off', action='store_false', help='do not try to guess bonds (True)')
    parser.add_argument('--switch_bond', type=int, nargs="+", default=None, help='apply trajectory switch if a bond is present')
    parser.add_argument('--switch_dist', type=int, nargs="+", default=None, help='apply trajectory switch if a distance is fulfilled')
    parser.add_argument('--switch_dist_arg', type=float, nargs="+", default=None, help='apply trajectory switch distance')
    parser.add_argument('--save_switch', action='store_false', help='save switch trajectory')
    parser.add_argument('--coupling_off', action='store_false', help='exclude coupling between vibrating objects')

    #=======================#
    #| organize user input |#
    #=======================#

    print('\nsys.argv:\n', sys.argv)
    sargs = []
    for arg in sys.argv:
        sarg = arg.replace('\\', '/')
        sargs.append(sarg)
    args = parser.parse_args(sargs[1:])
    folder = os.getcwd().replace('\\', '/')+'/'
    #args.out = folder+args.out+'/'
    #args.temp = folder+args.temp+'/'
    args.out = os.path.abspath(args.out)+'/'
    args.temp = os.path.abspath(args.temp)+'/'
    args.input = os.path.abspath(args.input)

    if args.time_resolved.lower() == 'on':
        args.time_resolved = True
    elif args.time_resolved.lower() == 'off':
        args.time_resolved = False
    else:
        raise ValueError('--time_resolved must be either "on" or "off"')

    if args.bpf.lower() == 'on' or args.bpf.lower() == '':
        args.bpf = True
    elif args.bpf.lower() == 'off':
        args.bpf = False
    else:
        raise ValueError('--bpf must be either "on" or "off"')

    try:
        args.time_window_length = int(args.time_window_length)
    except ValueError:
        args.time_window_length = float(args.time_window_length)
    if args.time_window_length >= 1.0: args.time_window_length = int(round(args.time_window_length))
    #print('folder:\n', folder)
    print('args:\n', args)
    prep_dir(args.temp)
    prep_dir(args.out)
    print()
    #print(args.temp)
    #print(args.out)

    #========================#
    #| perform calculations |#
    #========================#
    print('reading data...')
    loader = LMALoader(args)
    print('processing data...')
    proc = LMAProcessor(loader, args)
    if args.n_time_windows == 1 and not args.time_resolved:
        if not args.bpf:
            proc.spectrum()
        writer = OutputWriter(proc, args)
        print('writing data...')
        writer.write()
        if not args.bpf:
            print('plotting data...')
            writer.plot()
    else:
        if not args.bpf:
            proc.spectrum_tw()
        writer = OutputWriter(proc, args)
        print('writing data...')
        writer.write_tw()
        if not args.bpf:
            print('plotting data...')
            writer.plot_tw()
            if args.time_windows_3D:
                writer.plot_3d()

def plz_cite():

    print(
    """
    #======================================================================#
    #| If this code has benefited your research,                          |#
    #| please support us by citing:                                       |#
    #|                                                                    |#
    #| Massarczyk, M. et al. (2017) "Local Mode Analysis:                 |#
    #| Decoding IR Spectra by Visualizing Molecular Details.",            |#
    #| The Journal of Physical Chemistry B, acs.jpcb.6b09343.             |#
    #| https://doi.org/10.1021/acs.jpcb.6b09343                           |#
    #======================================================================#
    this package uses the following packages, please don't forget to cite:
    - numpy          https://doi.org/10.1109%2FMCSE.2011.37
    - matplotlib     https://doi.org/10.1109%2FMCSE.2007.55
    - mdtraj         https://doi.org/10.1016%2Fj.bpj.2015.08.015
    """)


if __name__ == "__main__":
    run()
    plz_cite()
