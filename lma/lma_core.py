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

import warnings

import numpy as np
import scipy.ndimage.filters as scifilter
from scipy.signal import blackmanharris, tukey
import mdtraj
#import matplotlib.pyplot as plt

def dev_spectrum(ARR, start=0, end=0, WIN=True):

    if ( end == 0 or end == 1 ): end = in_arr.shape[0]
    ZM = zero_mean(in_arr=ARR[start:end])
    NE = round_to_next_exponent(val=ARR[start:end].shape[0], base=2)
    FREQ = freq(dt=0.25, exp=NE, last_freq=4000, scale=1.0)
    FREQ2 = np.power(FREQ,2)
    if WIN:
        ZM = apply_window(in_arr=ZM)
    PZ = pad_zeros(in_arr=ZM)
    FT = ftrans(in_arr=PZ, freq_len=FREQ.shape[0])

    return (FREQ, FT*np.conjugate(FT)*FREQ2)

def gaussian_filter(Vec, sigma=None, verbose=True, filter_scale=500.0):
    if sigma is None:
        sigma = int(round(Vec.shape[0]/filter_scale,0))
        if verbose:
            print('setting gaussian sigma to', sigma)
    return scifilter.gaussian_filter1d(np.squeeze(Vec), sigma=sigma)

def round_to_next_exponent(val, base=2):
    import math
    return base**int(np.ceil(math.log(val, base)))


def frequency_list(dtime, full_size, last_freq=4000):
    dtime *= 1e-15
    freqs = (np.arange(1, full_size+1)/(full_size/(1.0/dtime))
                        )[:int(np.round(full_size/2.0))]/29979245800.0
    return np.insert(freqs[:(np.abs(freqs-last_freq)).argmin()], 0, 0.0)


def running_mean(in_arr, period=500):
    ret = np.cumsum(in_arr, axis=0)
    out = np.empty_like(in_arr)
    ret[period:,:,:] = ret[period:,:,:] - ret[:-period,:,:]
    if period % 2 == 0:
        out[int(period/2):-int(period/2)+1,:,:] = ret[period- 1:,:,:] / period
    else:
        out[int(period/2):-int(period/2)+1,:,:] = ret[period- 2:,:,:] / period
    out[:int(period/2),:,:] = in_arr[:int(period/2),:,:].mean(0)
    out[-int(period/2)+1:,:,:] = in_arr[-int(period/2)+1:,:,:].mean(0)
    return out


def compute_product(in_arr_a, in_arr_b=None):
    """Compute the product
    of all elements (atoms) in each timestep with their multiplyer in
    charges (partial charges).
    """
    #try:
    if in_arr_b is not None:
        return in_arr_a * in_arr_b[:,:,np.newaxis]
    else:
        return in_arr_a
    



def zero_mean_old(in_arr, start=0, end=0):
    """Remove the mean value of each element over time
    """
    if ( end == 0 or end == 1 ): end = in_arr.shape[0]
    if in_arr.ndim == 1:
        return (in_arr[start:end]-\
                    in_arr[start:end].mean()[np.newaxis,:])[:,np.newaxis,np.newaxis]
    elif in_arr.ndim == 2:
        return (in_arr[start:end,:]-\
                    in_arr[start:end,:].mean(0)[np.newaxis,:])[:,np.newaxis,:]
    elif in_arr.ndim == 3:
        return in_arr[start:end,:,:]-\
                        in_arr[start:end,:,:].mean(0)[np.newaxis,:,:]


def zero_mean(in_arr, start=0, end=0):
    """Remove the mean value of each element over time
    """
    if ( end == 0 or end == 1 ): end = in_arr.shape[0]
    if in_arr.ndim == 1:
        return in_arr-in_arr.mean()
    
    elif in_arr.ndim == 2:
        return (in_arr[start:end,:]-
                    in_arr[start:end,:].mean(0)[np.newaxis,:])[:,:,np.newaxis]
    
    elif in_arr.ndim == 3:
        return in_arr[start:end,:,:]-\
                        in_arr[start:end,:,:].mean(0)[np.newaxis,:,:]


def apply_window(in_arr, winType='blackmanharris', sym=True, alpha=0.5):
    """Apply a Blackman-Harris window onto the zero mean product
    """

    if str(winType).lower() == 'blackmanharris' or winType == '':
        if in_arr.ndim == 1:
            return in_arr * blackmanharris(in_arr.shape[0], sym=sym)
        if in_arr.ndim == 2:
            return in_arr * blackmanharris(in_arr.shape[0], sym=sym)[:,np.newaxis]
        if in_arr.ndim == 3:
            return in_arr * blackmanharris(in_arr.shape[0], sym=sym)[:,np.newaxis,np.newaxis]
    elif str(winType).lower() == 'tukey':
        if in_arr.ndim == 1:
            return in_arr * tukey(in_arr.shape[0], alpha, sym=sym)
        if in_arr.ndim == 2:
            return in_arr * tukey(in_arr.shape[0], alpha, sym=sym)[:,np.newaxis]
        if in_arr.ndim == 3:
            return in_arr * tukey(in_arr.shape[0], alpha, sym=sym)[:,np.newaxis,np.newaxis]
    elif str(winType).lower() == 'none':
        return in_arr
    else:
        #warnings.warn('there is no blackmanharris window function (not imported from scipy.signal)')
        return in_arr


def pad_zeros(in_arr):
    """Padd n zeros to start and end of each element of in_arr
    """
    if in_arr.ndim == 1:
        pval = int((round_to_next_exponent(
                in_arr.shape[0])-in_arr.shape[0])/2)
        fval = pval + 1 if in_arr.shape[0] & 1 else pval
        z_arr = np.zeros((in_arr.shape[0]+pval+fval))
        z_arr[fval:-pval] = in_arr
    if in_arr.ndim == 2:
        pval = int((round_to_next_exponent(
                in_arr.shape[0])-in_arr.shape[0])/2)
        fval = pval + 1 if in_arr.shape[0] & 1 else pval
        #z_arr = np.zeros((in_arr.shape[0]+pval+fval,in_arr.shape[1], in_arr.shape[2]))
        z_arr = np.zeros((in_arr.shape[0]+pval+fval,in_arr.shape[1]))
        z_arr[fval:-pval,:] = in_arr
    if in_arr.ndim == 3:
        pval = int((round_to_next_exponent(
                in_arr.shape[0])-in_arr.shape[0])/2)
        fval = pval + 1 if in_arr.shape[0] & 1 else pval
        #z_arr = np.zeros((in_arr.shape[0]+pval+fval,in_arr.shape[1]))
        z_arr = np.zeros((in_arr.shape[0]+pval+fval,in_arr.shape[1], in_arr.shape[2]))
        #print('z_arr.shape: ', z_arr.shape)
        #z_arr[fval:-pval,:,:] = in_arr
        z_arr[fval:-pval,:,:] = in_arr
    return z_arr

def freq(dt, exp, last_freq=4000, scale=1.0):
    """Create an frequency array which fitts the length of the  zero_mean 
    product 
    """
    last_freq /= scale
    return frequency_list(dt, exp, last_freq)* scale

def ftrans(in_arr, freq_len):
    """Creates an array with the fourier transformed data of the zero-padded
    product
    """
    if in_arr.ndim == 1:
        return np.fft.fft(in_arr, axis=0)[:freq_len]/in_arr.shape[0]
        
    if in_arr.ndim == 2:
        return np.fft.fft(in_arr, axis=0)[:freq_len,:]/in_arr.shape[0]

    if in_arr.ndim == 3:
        return np.fft.fft(in_arr, axis=0)[:freq_len,:,:]/in_arr.shape[0]


def project_vecs(in_arr, arg=''):
    """Filters non-aktive spektral features
    """
#    SUM = in_arr.sum(1)
#    ABSSUM = np.abs(SUM)
#    normSUM = np.einsum('ij,ij->i', ABSSUM, ABSSUM)
#    temp_arr = np.einsum('ij,ik->ijk',
#                    np.abs(np.einsum('ijk,ik->ij',
#                    np.conjugate(in_arr), SUM)/ normSUM[:,np.newaxis]), SUM)
#    return temp_arr* (np.linalg.norm(SUM, axis=1)/ np.linalg.norm(
#                         temp_arr.sum(1), axis=1))[:,np.newaxis,np.newaxis]

    com_in_arr = np.empty((*in_arr.shape[:2], 6))
    for ii in range(3):
        com_in_arr[:,:,ii*2]   = np.real(in_arr[:,:,ii])
        com_in_arr[:,:,ii*2+1] = np.imag(in_arr[:,:,ii])

    CSUM = com_in_arr.sum(1)

    temp_arr = np.einsum('ij,ik->ijk',
        np.abs(np.einsum('ijk,ik->ij',
            np.conjugate(com_in_arr), CSUM)/
                np.einsum('ik,ik->i', CSUM, CSUM)[:,np.newaxis]), CSUM)
    return temp_arr* (np.linalg.norm(CSUM, axis=1)/ np.linalg.norm(
                temp_arr.sum(1), axis=1))[:,np.newaxis,np.newaxis]

def r_project_vecs(in_arr):
    """Filters non-aktive spektral features
    """
    com_in_arr = np.empty((*in_arr.shape[:2], 6))
    for ii in range(3):
        com_in_arr[:,:,ii*2]   = np.real(in_arr[:,:,ii])
        com_in_arr[:,:,ii*2+1] = np.imag(in_arr[:,:,ii])
    CSUM = com_in_arr.sum(1)
    temp_arr = np.einsum('ij,ik->ijk',
        (np.einsum('ijk,ik->ij',
            np.conjugate(com_in_arr), CSUM)/
                np.einsum('ik,ik->i', CSUM, CSUM)[:,np.newaxis]), CSUM)
    temp_arr = com_in_arr - temp_arr
    return temp_arr[:,:,np.array([0,2,4])] + 1j* temp_arr[:,:,np.array([1,3,5])]


def power_matrix(in_arr, onlydiag=False):
    """Generates the spectral matrix of all elements and relations
    """
    if onlydiag:
        #return np.rollaxis(np.real(np.einsum('ijk,ijk->ij', np.conjugate(in_arr), in_arr)), -1)
        #return np.rollaxis(np.real(np.einsum('ijk,ijk->ji', np.conjugate(in_arr), in_arr)), -1)
        return np.real(np.einsum('ijk,ijk->ji', np.conjugate(in_arr), in_arr))[np.newaxis,:,:]
    else:
        return np.real(np.einsum('wik,wjk->ijw', np.conjugate(in_arr), in_arr))

def power_factor(in_arr, freq_list, factor_type='omega2'):
    """Apply factor array for scaling intensitys
    """
    if factor_type == 'omega2':
        return in_arr* freq_list[np.newaxis,np.newaxis,:]**2
        #return in_arr* freq_list[np.newaxis,:]**2
    elif str(factor_type) == '1':
        return np.copy(in_arr)
    elif str(factor_type) == '-1':
        return in_arr/ freq_list[np.newaxis,np.newaxis,:]
        #return in_arr/ freq_list[np.newaxis,:]
        return in_arr/ freq_list[np.newaxis,np.newaxis,:]**2
        #return in_arr/ freq_list[np.newaxis,:]**2
        return in_arr* np.sqrt(freq_list[np.newaxis,np.newaxis,:])
        #return in_arr* np.sqrt(freq_list[np.newaxis,:])


def distribution_factor_matrix(in_arr, arg=''):
    """Creates a matrix with distribution factors for the relation between all atoms
    in each frequency
    """
    Spec_Diag = in_arr.diagonal(axis1=0, axis2=1)
    if arg == 'sqrt': Spec_Diag = np.sqrt(Spec_Diag)
    if arg == 'power': Spec_Diag = np.power(Spec_Diag,2)
    ret_dist_fac_matrix = np.empty((in_arr.shape[0], in_arr.shape[0],
                                                     in_arr.shape[2]))
    with np.errstate(divide='ignore',invalid='ignore'):
        for elem_ii in range(in_arr.shape[0]):
            for elem_jj in range(in_arr.shape[1]):
                ret_dist_fac_matrix[elem_ii,elem_jj,:] = Spec_Diag[:,elem_ii]/(
                                                         Spec_Diag[:,elem_ii] +
                                                         Spec_Diag[:,elem_jj])
    return ret_dist_fac_matrix

def sum_up_atoms(in_arr_a, in_arr_b):
    """Returns an array with spectra belonging to atoms
    """
    return np.einsum('ijw,ijw->wi', in_arr_a, in_arr_b)* 2

def calc_full_spectrum(self, hint=None):
    """calculate all steps for atom spectra
    """
    self.compute_product()
    #print('XXX')
    #print(self._raw_product)
    self._raw_product = self._raw_product.sum(1)#[]
    self.zero_mean()
    self.freq()
    self.freq2()
    self.pad_zeros()
    self.ftrans()
    self.power_matrix()
    self.power_factor()
    self.distribution_factor_matrix()
    self.sum_up_atoms()
    self._full_spectrum = np.copy(self._atom_specs)
    #print()
    #print(self._full_spectrum.shape)
    #print()
    if self.gfilter:
        self.gaussfilter()
        self._filtered_full_spectrum = np.copy(self._filtered_atom_specs)


def spectrum(arr, args='', win='tukey', power=None, proj=True, diag=False,
             dt=0.5, scale=1.00, dfm=None, wav=None, last_freq=4000.0, switch_arr=None, rproj=False):
    """calculate all steps for atom spectra
    """
#    plt.plot(arr[:,:,0])
#    plt.plot(arr[:,:,1])
#    plt.plot(arr[:,:,2])
#    plt.xlim(0,50)
#    plt.savefig('C:/WorkBench/Promotionsordner/LMA_Folder/lma_out/test.png')
    #print('arr:', arr.shape)
    #print('arr:', arr.shape, np.isnan(arr)[np.isnan(arr) == True].size)

    arr_zero = zero_mean(arr)
    if switch_arr is not None:
        arr_zero *= switch_arr[:,np.newaxis,np.newaxis]
    if wav is None:
        fin_len = round_to_next_exponent(arr.shape[0])
        #wav = frequency_list(dt, fin_len)*scale
        wav = freq(dt, fin_len, last_freq=last_freq, scale=scale)

    #print('wav:', wav.shape)
    arr_pad = pad_zeros(apply_window(arr_zero, winType=win))
    #print('arr_pad:', arr_pad.shape)
    #tarr = np.fft.fft(arr_pad, axis=0)[:wav.shape[0],:,:]/arr_pad.shape[0]
    #print('tarr:', tarr.shape)
    tarr = ftrans(arr_pad, wav.shape[0])
    #print('tarr:', tarr.shape, np.isnan(tarr)[np.isnan(tarr) == True].size)

    if proj:
        ptarr = project_vecs(tarr)
    elif rproj:
        diag = True
        ptarr = r_project_vecs(tarr)
    else:
        ptarr = tarr
    #print('ptarr:', ptarr.shape)

    if diag:
        pm = power_matrix(ptarr, onlydiag=True)
    else:
        pm = power_matrix(ptarr)
    #print('pm:', pm.shape, np.isnan(pm)[np.isnan(pm) == True].size)

    factor_type = 'omega2'
    if power is not None: factor_type = power
    pf = power_factor(pm, wav, factor_type)
    #print('pf:', pf.shape, np.isnan(pf)[np.isnan(pf) == True].size)

    if pf.shape[0] != pf.shape[1] and pf.ndim == 3:
        pf = np.squeeze(pf)
    #print('pf:', pf.shape)
    if diag:
        if pf.shape[0] == 1 and pf.ndim == 2:
            return wav, np.rollaxis(pf, -1)
        elif pf.shape[0] == 1 and pf.shape[1] == 1 and pf.ndim == 3:
            return wav, np.rollaxis(np.squeeze(pf), -1)[:,np.newaxis]
        return wav, np.rollaxis(np.squeeze(pf), -1)

    if not proj:
        return wav, pf

    if dfm is None:
        dfm = distribution_factor_matrix(pf)
    #print('dfm:', dfm.shape, np.isnan(dfm)[np.isnan(dfm) == True].size)

    return wav, sum_up_atoms(pf, dfm)


def pos2vel2acc(pos):
    VelArr = np.zeros_like(pos)
    AccArr = np.zeros_like(pos)
    VelArr[1:,:,:] = np.diff(pos, axis=0)
    VelArr[0,:,:] = VelArr[1,:,:]
    AccArr[1:,:,:] = np.diff(VelArr, axis=0)
    AccArr[0:2,:,:] = AccArr[2,:,:]
    VelArr *= 5e1
    AccArr *= 1e3
    return VelArr, AccArr

def mode_spectrum(TRAJ, Pos, Vel, dt=0.5, scale=1.0, power=-1, dist_switch=True, switch_arr=None, guess=True):

    BONDLenTraj, NORMLenTraj, INPLANELenTraj, Bonds = mode_spectrum_traj(TRAJ, Pos, Vel, dist_switch, guess=guess)
    #print('Pos:', Pos.shape, Pos[Pos == 0.0].size)
    #print('BONDLenTraj:', BONDLenTraj.shape, np.isnan(BONDLenTraj)[np.isnan(BONDLenTraj) == True].size)
    #print('NORMLenTraj:', NORMLenTraj.shape, np.isnan(NORMLenTraj)[np.isnan(NORMLenTraj) == True].size)
    #print('INPLANELenTraj:', INPLANELenTraj.shape, np.isnan(INPLANELenTraj)[np.isnan(INPLANELenTraj) == True].size)

    wav, spec_b = spectrum(BONDLenTraj, proj=False, diag=True, power=power, dt=dt, scale=scale, switch_arr=switch_arr)
    #print('spec_b:', spec_b.shape, np.isnan(spec_b)[np.isnan(spec_b) == True].size)
    wav, spec_n = spectrum(NORMLenTraj, proj=False, diag=True, power=power, dt=dt, scale=scale, switch_arr=switch_arr)
    #print('spec_n:', spec_n.shape, np.isnan(spec_n)[np.isnan(spec_n) == True].size)
    wav, spec_p = spectrum(INPLANELenTraj, proj=False, diag=True, power=power, dt=dt, scale=scale, switch_arr=switch_arr)
    #print('spec_p:', spec_p.shape, np.isnan(spec_p)[np.isnan(spec_p) == True].size)
    
    return wav, spec_b, spec_p, spec_n, Bonds


def mode_spectrum_traj(TRAJ, Pos, Vel, dist_switch=True, guess=True):
    #try:
    #    from lma_processing import guess_bonds
    #except ModuleNotFoundError:
    #    from mma.mdtrajfunc import guess_bonds
    if guess:
        guess_bonds(TRAJ)
    Bonds = list(TRAJ.top.bonds)

    BNList = [[] for ii in range(TRAJ.n_atoms)]
    for atm_a, atm_b in Bonds:
        if atm_b.index not in BNList[atm_a.index]:
            BNList[atm_a.index].append(atm_b.index)
        if atm_a.index not in BNList[atm_b.index]:
            BNList[atm_b.index].append(atm_a.index)
#    PlanePointNeighbours = []
#    for atm_a, atm_b in Bonds:
#        PlanePointNeighbours.append(
#            sorted(list(
#                    set(BNList[atm_a.index]+BNList[atm_b.index])-
#                    set([atm_a.index, atm_b.index]))))

    RegressPointsIdx = []
    for atm_a, atm_b in Bonds:
        RegressPointsIdx.append(
            sorted(list(set(BNList[atm_a.index]+BNList[atm_b.index]))))

    MassList = np.array([atm.element.mass for atm in TRAJ.top.atoms])

#    PlanePoints = np.empty((Pos.shape[0], len(Bonds), 3))
#    PlanePoints.fill(np.nan)
#
#    for bnd_idx, bnd in enumerate(Bonds):
#        nmsum = MassList[PlanePointNeighbours[bnd_idx]].sum()
#        ncsum = (Pos[:,PlanePointNeighbours[bnd_idx],:]*
#                 MassList[np.newaxis,PlanePointNeighbours[bnd_idx],np.newaxis]).sum(1)
#        PlanePoints[:,bnd_idx,:] = ncsum/nmsum


    PLANEVecs = np.empty((Pos.shape[0], len(Bonds), 3))
    PLANEVecs.fill(np.nan)
    PLANEVecs[:,:,0].fill(1.0)
    for bnd_idx, bnd in enumerate(Bonds):
        #print('1:', bnd_idx, bnd)
        #print('2:', RegressPointsIdx[bnd_idx])
        masses = np.round(MassList[RegressPointsIdx[bnd_idx]],0).astype(int)
        #print('3:', masses)
        #print('4:', Pos[0,RegressPointsIdx[bnd_idx],:])
        #maspos = (Pos[:,RegressPointsIdx[bnd_idx],:]*
        #          MassList[np.newaxis,RegressPointsIdx[bnd_idx],np.newaxis])
        #print('\n\t',maspos[0,:,:])
        pos = Pos[:,RegressPointsIdx[bnd_idx],:]
        #print('\n\t',pos[0,:,:])
        Xm = pos[:,:,0]
        Ym = pos[:,:,1]
        Zm = pos[:,:,2]
        XMList = np.empty((pos.shape[0],masses.sum()))
        YMList = np.empty((pos.shape[0],masses.sum()))
        ZMList = np.empty((pos.shape[0],masses.sum()))
        ctr = 0
        for mIdx, mas in enumerate(masses):
            for li in range(mas):
                XMList[:,ctr] = Xm[:,mIdx]
                YMList[:,ctr] = Ym[:,mIdx]
                ZMList[:,ctr] = Zm[:,mIdx]
                ctr += 1
        
        Xm = XMList
        Ym = YMList
        Zm = ZMList
        Xm -= Xm.mean(1)[:,np.newaxis]
        #print('\nXm\t',Xm[0,:])
        #print('Xm\t',Xm.shape)
        Xm2 = Xm*Xm
        #print('Xm2\t',Xm2[0,:])
        #print('Xm2\t',Xm2.shape)
        Ym -= Ym.mean(1)[:,np.newaxis]
        #print('Ym\t',Ym[0,:])
        Zm -= Zm.mean(1)[:,np.newaxis]
        #print('Zm\t',Zm[0,:])
        
        #XmSum = Xm.sum(1)
        #print('XmSum\t',XmSum[0])
        Xm2Sum = Xm2.sum(1)
        #print('Xm2Sum\t',Xm2Sum[0])
        XYmSum = (Xm*Ym).sum(1)
        #print('XYmSum\t',XYmSum[0])
        XZmSum = (Xm*Zm).sum(1)
        #print('XZmSum\t',XZmSum[0])

        PLANEVecs[:,bnd_idx,1] = XYmSum/Xm2Sum
        PLANEVecs[:,bnd_idx,2] = XZmSum/Xm2Sum
        #print(np.linalg.norm(PLANEVecs[:,bnd_idx,:], axis=1)[0])
        PLANEVecs[:,bnd_idx,:] /= (np.linalg.norm(PLANEVecs[:,bnd_idx,:], axis=1))[:,np.newaxis]
        
        #print('\nPLANEVecs\t',PLANEVecs[0,bnd_idx,:])



    BONDVecs = (Pos[:,np.array([bnd[0].index for bnd in Bonds]),:] -
                Pos[:,np.array([bnd[1].index for bnd in Bonds]),:])
    
    if dist_switch:
        BONDVecTraj = np.linalg.norm(BONDVecs, axis=2)
        BOND_dist_switch_Arr = np.empty_like(BONDVecTraj)
        bond_max_dist_list = guess_bond_max_dist(TRAJ)
        for idx_bnd in range(BONDVecs.shape[1]):
            BOND_dist_switch_Arr[:,idx_bnd] = np.array(list(map(bond_switch_func, BONDVecTraj[:,idx_bnd], np.ones_like(BONDVecTraj[:,idx_bnd])*bond_max_dist_list[idx_bnd])))
    
    BONDVecs[BONDVecs == 0.0] += 1.0e-12
    BONDVecs = BONDVecs / np.linalg.norm(BONDVecs, axis=2, keepdims=True)
    #PLANEVecs = (Pos[:,np.array([bnd[0].index for bnd in Bonds]),:] - PlanePoints)
    PLANEVecs[PLANEVecs == 0.0] += 1.0e-12
    CrossVecs = np.cross(BONDVecs, PLANEVecs)
    #print('CrossVecs:', CrossVecs.shape, np.isnan(CrossVecs)[np.isnan(CrossVecs) == True].size)
    NORMVecs = CrossVecs / np.linalg.norm(CrossVecs, axis=2, keepdims=True)
    #print('NORMVecs:', NORMVecs.shape, np.isnan(NORMVecs)[np.isnan(NORMVecs) == True].size)
    INPLANEVecs = np.cross(BONDVecs, NORMVecs)
    INPLANEVecs = INPLANEVecs / np.linalg.norm(INPLANEVecs, axis=2, keepdims=True)
    #print('INPLANEVecs:', INPLANEVecs.shape, np.isnan(INPLANEVecs)[np.isnan(INPLANEVecs) == True].size)

    BONDVecsTDot_A = np.empty((BONDVecs.shape[0], len(Bonds)))
    BONDVecsTDot_B = np.empty((BONDVecs.shape[0], len(Bonds)))
    for idx, (atm_a, atm_b) in enumerate(Bonds):
        BONDVecsTDot_A[:,idx] = np.einsum('ik,ik->i',
            BONDVecs[:,idx,:], Vel[:,atm_a.index,:])
        BONDVecsTDot_B[:,idx] = np.einsum('ik,ik->i',
            BONDVecs[:,idx,:], Vel[:,atm_b.index,:])
    BONDTraj = (BONDVecsTDot_A[:,:,np.newaxis] * BONDVecs -
                BONDVecsTDot_B[:,:,np.newaxis] * BONDVecs)
    BONDLenTraj = np.linalg.norm(BONDTraj, axis=2)
    BONDLenTraj[BONDVecsTDot_A < BONDVecsTDot_B] *= -1
    if dist_switch:
        BONDLenTraj *= BOND_dist_switch_Arr

    NORMVecsTDot_A = np.empty((NORMVecs.shape[0], len(Bonds)))
    NORMVecsTDot_B = np.empty((NORMVecs.shape[0], len(Bonds)))
    for idx, (atm_a, atm_b) in enumerate(Bonds):
        NORMVecsTDot_A[:,idx] = np.einsum('ik,ik->i',
            NORMVecs[:,idx,:], Vel[:,atm_a.index,:])
        NORMVecsTDot_B[:,idx] = np.einsum('ik,ik->i',
            NORMVecs[:,idx,:], Vel[:,atm_b.index,:])
    NORMTraj = (NORMVecsTDot_A[:,:,np.newaxis] * NORMVecs -
                NORMVecsTDot_B[:,:,np.newaxis] * NORMVecs)
    NORMLenTraj = np.linalg.norm(NORMTraj, axis=2)
    NORMLenTraj[NORMVecsTDot_A < NORMVecsTDot_B] *= -1
    if dist_switch:
        NORMLenTraj *= BOND_dist_switch_Arr

    INPLANEVecsTDot_A = np.empty((INPLANEVecs.shape[0], len(Bonds)))
    INPLANEVecsTDot_B = np.empty((INPLANEVecs.shape[0], len(Bonds)))
    for idx, (atm_a, atm_b) in enumerate(Bonds):
        INPLANEVecsTDot_A[:,idx] = np.einsum('ik,ik->i',
            INPLANEVecs[:,idx,:], Vel[:,atm_a.index,:])
        INPLANEVecsTDot_B[:,idx] = np.einsum('ik,ik->i',
            INPLANEVecs[:,idx,:], Vel[:,atm_b.index,:])
    INPLANETraj = (INPLANEVecsTDot_A[:,:,np.newaxis] * INPLANEVecs -
                   INPLANEVecsTDot_B[:,:,np.newaxis] * INPLANEVecs)
    INPLANELenTraj = np.linalg.norm(INPLANETraj, axis=2)
    INPLANELenTraj[INPLANEVecsTDot_A < INPLANEVecsTDot_B] *= -1
    if dist_switch:
        INPLANELenTraj *= BOND_dist_switch_Arr

    return BONDLenTraj, NORMLenTraj, INPLANELenTraj, Bonds


def mode_spec(BONDLenTraj, NORMLenTraj, INPLANELenTraj, dt=0.5, scale=1.0, power=-1, switch_arr=None):
    
    wav, spec_b = spectrum(BONDLenTraj, proj=False, diag=True, power=power, dt=dt, scale=scale, switch_arr=switch_arr)
    wav, spec_n = spectrum(NORMLenTraj, proj=False, diag=True, power=power, dt=dt, scale=scale, switch_arr=switch_arr)
    wav, spec_p = spectrum(INPLANELenTraj, proj=False, diag=True, power=power, dt=dt, scale=scale, switch_arr=switch_arr)
    
    return wav, spec_b, spec_p, spec_n


def fuse_mode_spectrum(allspec, spec_b, spec_p, spec_n, Bonds, gauss_sigma):
    #print('allspec:', allspec.shape)
    #print('spec_b:', spec_b.shape)
    #print('spec_p:', spec_p.shape)
    #print('spec_n:', spec_n.shape)
    #print('Bonds:', Bonds)
    gauss_sigma = 30
    for ii in range(spec_b.shape[1]):
        spec_b[:,ii] = gaussian_filter(spec_b[:,ii], gauss_sigma)
        spec_p[:,ii] = gaussian_filter(spec_p[:,ii], gauss_sigma)
        spec_n[:,ii] = gaussian_filter(spec_n[:,ii], gauss_sigma)

    new_spec_b = np.zeros_like(spec_b)
    new_spec_p = np.zeros_like(spec_p)
    new_spec_n = np.zeros_like(spec_n)
    sum_arr = np.zeros_like(allspec)

    for idx, (bnd_a, bnd_b) in enumerate(Bonds):
        #print(idx, bnd_a, bnd_b)
        sum_arr[:,bnd_a.index] += spec_b[:,idx]
        sum_arr[:,bnd_a.index] += spec_p[:,idx]
        sum_arr[:,bnd_a.index] += spec_n[:,idx]
        sum_arr[:,bnd_b.index] += spec_b[:,idx]
        sum_arr[:,bnd_b.index] += spec_p[:,idx]
        sum_arr[:,bnd_b.index] += spec_n[:,idx]
    
    for idx, (bnd_a, bnd_b) in enumerate(Bonds):
        #print(idx, bnd_a, bnd_b)
        new_spec_b[:,idx] += (spec_b[:,idx]/sum_arr[:,bnd_a.index])*allspec[:,bnd_a.index]
        new_spec_b[:,idx] += (spec_b[:,idx]/sum_arr[:,bnd_b.index])*allspec[:,bnd_b.index]
        new_spec_p[:,idx] += (spec_p[:,idx]/sum_arr[:,bnd_a.index])*allspec[:,bnd_a.index]
        new_spec_p[:,idx] += (spec_p[:,idx]/sum_arr[:,bnd_b.index])*allspec[:,bnd_b.index]
        new_spec_n[:,idx] += (spec_n[:,idx]/sum_arr[:,bnd_a.index])*allspec[:,bnd_a.index]
        new_spec_n[:,idx] += (spec_n[:,idx]/sum_arr[:,bnd_b.index])*allspec[:,bnd_b.index]

    return new_spec_b, new_spec_p, new_spec_n


def bond_switch_traj(TRAJ, guess=True):

    if guess:
        guess_bonds(TRAJ)
    Bonds = list(TRAJ.top.bonds)
    Pos = TRAJ.xyz
    BONDVecs = (Pos[:,np.array([bnd[0].index for bnd in Bonds]),:] -
                Pos[:,np.array([bnd[1].index for bnd in Bonds]),:])
    
    BONDVecTraj = np.linalg.norm(BONDVecs, axis=2)
    BOND_dist_switch_Arr = np.empty_like(BONDVecTraj)
    bond_max_dist_list = guess_bond_max_dist(TRAJ)
    for idx_bnd in range(BONDVecs.shape[1]):
        BOND_dist_switch_Arr[:,idx_bnd] = np.array(list(map(bond_switch_func, BONDVecTraj[:,idx_bnd], np.ones_like(BONDVecTraj[:,idx_bnd])*bond_max_dist_list[idx_bnd])))
    
    return BOND_dist_switch_Arr




def get_band_pass_filter_traj(TRAJ, dt, freq_a=0, freq_b=0, multip=100, smooth=1000, scale=1.0, start=0, end=0, last_freq=4000.0):
    """generate trajectory with band pass filtered motion (vibration)
    """
    dtime = dt
    positions = TRAJ.xyz*10.0
    freq_scale_factor = scale
#    print()
#    print("get_band_pass_filter_traj:")
#    print("freq_a: ", freq_a)
#    print("freq_b: ", freq_b)
#    print("dtime: ", dtime)
#    print("round_to_next_exponent: ",
#                    round_to_next_exponent(positions.shape[0]))
#    print("freq_scale_factor: ", freq_scale_factor)
#    print()

    bpf_wav = frequency_list(dtime,
                round_to_next_exponent(positions.shape[0]),
                last_freq)* freq_scale_factor
    #bpf_wav = frequency_list(dtime,
    #            positions.shape[0],
    #            last_freq)* freq_scale_factor
    #print('\t\tav-less...')
    bpf_zm = zero_mean(positions, start, end)
    
    bpf_arr_pad = pad_zeros(apply_window(bpf_zm, winType='tukey'))
    #print('\t\ttransformation...')
    bpf_fft = calc_ftrans_bpf(bpf_arr_pad)
    #bpf_fft = calc_ftrans_bpf(bpf_zm)
    #print('\t\tcutting...')
    bpf_cutfft = calc_cut_fft_arr_win(bpf_fft,
                    bpf_wav, freq_a, freq_b, freq_scale_factor)

    #print('\t\tback transformation...')
    bpf_rfft = calc_rftrans_bpf(bpf_cutfft)

    #print('\t\trenorming...')
    pdiff = int(round((bpf_arr_pad.shape[0] - positions.shape[0])/2.0))
    pos_idx = np.array(list(range(positions.shape[0])))+pdiff
    bpf_norm_fft = calc_bpf_renorm_fft(bpf_rfft, multip)[pos_idx,:,:]

    #print('\t\tsmoothing...')
    bpf_pos = running_mean(positions, smooth)

    #MDTraj_bpf = mdtraj.Trajectory( (bpf_pos / 10.0)+
    #                    (bpf_norm_fft / 10.0), TRAJ.top)
    
    #print('\t\tcombining...')
    MDTraj_bpf = mdtraj.Trajectory((bpf_pos/10.0)+
            apply_window(bpf_norm_fft/10.0, winType='tukey'), TRAJ.top)

#    MDTraj_bpf = mdtraj.Trajectory((bpf_norm_fft/10.0), TRAJ.top)
    return MDTraj_bpf


def calc_ftrans_bpf(in_arr):
    """Creates an array with the fourier transformed data
    """
    return np.fft.fft(in_arr, axis=0)

def calc_ftrans_bpf_pyfftw(in_arr, fft_arr, fft_func):
    """Creates an array with the fourier transformed data of the zero-padded
    product
    """
    fft_arr[:,:,:] = in_arr
    return fft_func()

#def calc_ftrans_bpf_pyfftw(in_arr):
#    """Creates an array with the fourier transformed data of the zero-padded
#    product
#    """
#    fft_arr = pyfftw.empty_aligned(in_arr.shape, dtype='complex128')
#    fft_o = pyfftw.builders.fft(fft_arr, axis=0, threads=4)
#    fft_arr[:,:,:] = in_arr
#    return fft_o()

def calc_rftrans_bpf(in_arr):
    """Creates an array with the fourier transformed data of the zero-padded
    product
    """
    return np.real(np.fft.ifft(in_arr, axis=0))

def calc_rftrans_bpf_pyfftw(in_arr, fft_arr, fft_func):
    """Creates an array with the fourier transformed data of the zero-padded
    product
    """
    fft_arr[:,:,:] = in_arr
    return fft_func()

#def calc_rftrans_bpf_pyfftw(in_arr):
#    """Creates an array with the fourier transformed data of the zero-padded
#    product
#    """
#    fft_arr = pyfftw.empty_aligned(in_arr.shape, dtype='complex128')
#    fft_o = pyfftw.builders.ifft(fft_arr, axis=0, threads=4)
#    fft_arr[:,:,:] = in_arr
#    return fft_o()


def calc_cut_fft_arr(in_arr, freq_list, freq_a, freq_b):
    s_freq_a = freq_a#* freq_scale_factor
    s_freq_b = freq_b#* freq_scale_factor
    freq_a_ind = np.argmax(freq_list>s_freq_a)
    freq_b_ind = np.argmax(freq_list>s_freq_b)
    out_arr = np.zeros_like(in_arr)

    out_arr[freq_a_ind:freq_b_ind,:,:] = in_arr[freq_a_ind:freq_b_ind,:,:]
    out_arr[out_arr.shape[0]-freq_b_ind+1:out_arr.shape[0]-freq_a_ind+1,:,:] = \
            in_arr[out_arr.shape[0]-freq_b_ind+1:out_arr.shape[0]-freq_a_ind+1,:,:]

    return out_arr

def calc_cut_fft_arr_win(in_arr, freq_list, freq_a, freq_b, freq_scale_factor):
    s_freq_a = freq_a* freq_scale_factor
    s_freq_b = freq_b* freq_scale_factor
    freq_a_ind = np.argmax(freq_list>s_freq_a)
    freq_b_ind = np.argmax(freq_list>s_freq_b)
    out_arr = np.zeros_like(in_arr)

    #print('freq_a_ind:', freq_a_ind, '->', freq_list[freq_a_ind])
    #print('freq_b_ind:', freq_b_ind, '->', freq_list[freq_b_ind])

    #print('ZERO_MEAN!!!')
    out_arr[freq_a_ind:freq_b_ind,:,:] = apply_window(
                                in_arr[freq_a_ind:freq_b_ind,:,:],
                                winType='tukey')
    
    out_arr[out_arr.shape[0]-freq_b_ind+1:out_arr.shape[0]-freq_a_ind+1,:,:] = \
        apply_window(
            in_arr[out_arr.shape[0]-freq_b_ind+1:out_arr.shape[0]-freq_a_ind+1,:,:],
                winType='tukey')

    return out_arr


def calc_gaussfilter_bpf(in_arr, sigma=75):
    out_arr = np.empty_like(in_arr)
    for elem in range(in_arr.shape[1]):
        out_arr[:,elem,0] = gaussian_filter(in_arr[:,elem,0], sigma)
        out_arr[:,elem,1] = gaussian_filter(in_arr[:,elem,1], sigma)
        out_arr[:,elem,2] = gaussian_filter(in_arr[:,elem,2], sigma)

    return out_arr

def get_r_win(in_arr):
    laenge = in_arr.shape[0]
    dev = 0.2
    dev_len = int(laenge* dev)
    win_arr = np.ones(laenge)
    flan_a = np.linspace(0,1,dev_len+1)
    flan_b = np.linspace(1,0,dev_len+1)
    win_arr[:flan_a.shape[0]] = flan_a
    win_arr[-flan_b.shape[0]:] = flan_b
    return win_arr

def calc_bpf_renorm_fft(in_arr, multip=10, setpoint=0.002):
    #from scipy.signal import blackmanharris
    #win = (((1-blackmanharris(
    #                in_arr.shape[0])+0.5))/1.5)[:,np.newaxis,np.newaxis]
    win_mean = np.abs(in_arr).mean()
    win_max = np.abs(in_arr).max()
    #print('XXX:', in_arr.shape)
    Vib_maxs = np.abs(in_arr).max(0)
    #print('Vib_maxs shape:', Vib_maxs.shape)
    #print('Vib_maxs:\n', Vib_maxs)
    #print('Vib_max:', win_max)
    Vib_means = np.abs(in_arr).mean(0)
    #print('Vib_means shape:', Vib_means.shape)
    #print('Vib_means:\n', Vib_means)
    #print('Vib_mean:', win_mean)
    #print(0.005/win_mean)
    #mult = round(setpoint/win_max)
    #mult = round(round(setpoint/win_mean)/4.0)
    #mult = round(setpoint/win_mean, 3)
    mult = np.sqrt(round(setpoint/win_mean, 3)*2)

#    print('setpoint: ', setpoint)
#    print('win_max: ', win_max)
#    print('round(setpoint/win_max): ', round(setpoint/win_max))
#    print('win_mean: ', win_mean)
#    print('round(setpoint/win_mean): ', round(setpoint/win_mean))
#    print('mult: ', mult)
    #mult = 100#00.0
    #mult = multip
    print('\t\tmultiplier: ', mult)
    return in_arr* mult
    #return win* in_arr* mult
    #return in_arr* multip


def get_windowSlices(stop, returnNstates=1, start=0, windowwidth=0):
    Len = stop - start
    offset = start

    if ( returnNstates in [0, 1] ): return ([offset,Len],)
    if windowwidth == 0 or windowwidth == 1: windowwidth = 0.65
    if windowwidth < 1: windowwidth = int(round(windowwidth*Len))
    AllARRs = []
    stepWidh = int(round((Len-windowwidth)/(returnNstates-1)) )
    
    #print( "offset: ", offset)
    #print( "windowwidth: ", windowwidth)
    #print( "stepWidh: ", stepWidh)

    for retN in range(returnNstates):
        tmpARR = []
        start = retN * stepWidh
        stop = start+windowwidth
        if stop > Len:
            stop = Len
            start = stop-windowwidth
        tmpARR = [offset+start, offset+stop]
        AllARRs.append(tmpARR)

    return AllARRs



def bond_cutoff(eA, eB, cutoff=0.035):
    elems = [eA, eB]
    if "H" in elems and "S" in elems: return 0.025
    elif eA == "H" and eB == "H": return -0.06
    elif eA == "D" and eB == "D": return -0.06
    elif eA == "C" and eB == "C": return 0.041
    elif "C" in elems and "O" in elems: return 0.035
    elif "Mg" in elems: return -0.050
    elif "H" in elems and ("O" in elems): return -0.016
    elif "H" in elems and ("C" in elems): return -0.012
    elif "H" in elems and ("N" in elems): return -0.019
    elif "D" in elems and ("O" in elems): return -0.016
    elif "D" in elems and ("C" in elems): return -0.012
    elif "D" in elems and ("N" in elems): return -0.019
    elif eA == "S" and eB == "S": return 0.000
    return cutoff

def is_bond(vdwA, vdwB, Dist, cutoff):
    if (((vdwA + vdwB)/2.0)+cutoff) >= Dist: return True
    else: return False

def bond_max_dist(vdwA, vdwB, cutoff):
    return ((vdwA + vdwB)/2.0)+cutoff

def guess_bonds(TRAJ):
    tblist = []
    NList = mdtraj.compute_neighborlist(TRAJ, cutoff=0.25, frame=0, periodic=False)
    #print(NList)
    for idx, atm in enumerate(TRAJ.top.atoms):
        #print(idx, atm)
        for neig in NList[idx]:
            n_atm = TRAJ.top.atom(neig)
            dist = np.linalg.norm(TRAJ.xyz[0,neig,:]-TRAJ.xyz[0,idx,:])
            cutoff = bond_cutoff(atm.element.symbol, n_atm.element.symbol)
            is_b = is_bond(atm.element.radius, n_atm.element.radius, dist, cutoff)
            #print('\t', atm.element.symbol, n_atm.element.symbol,
            #           atm.element.radius, n_atm.element.radius,
            #           ((atm.element.radius + n_atm.element.radius)/2.0)+cutoff,
            #           dist, cutoff, is_b)
            if is_b:
                if sorted([idx, neig]) not in tblist:
                    tblist.append(sorted([idx, neig]))
    exist_bonds = [[bnd_a.index, bnd_b.index] for (bnd_a, bnd_b) in TRAJ.top.bonds]
    for bnd in sorted(tblist):
        #print(bnd)
        if bnd not in exist_bonds:
            TRAJ.top.add_bond(TRAJ.top.atom(bnd[0]), TRAJ.top.atom(bnd[1]))

def guess_bond_max_dist(TRAJ):
    tblist = []
    BONDS = list(TRAJ.top.bonds)
    for idx, (atm_a, atm_b) in enumerate(BONDS):
        cutoff_add = bond_cutoff(atm_a.element.symbol, atm_b.element.symbol)
        bmax_dist = bond_max_dist(atm_a.element.radius, atm_b.element.radius, cutoff_add)
        #print(idx, atm_a, atm_b, bmax_dist)
        tblist.append(bmax_dist)
    return tblist

def bond_switch_func(is_dist, max_dist, toll=0.125):
    z_ = is_dist
    s_ = max_dist*(toll**2)
    mu = max_dist+((max_dist*toll)/2.0)
    return 1.0/(1.0+np.power(np.e,((z_-mu)/(s_))))


def switch_by_bond(TRAJ, switch_bond, switch_traj=None, guess=True):
    #print('switch_bond:', switch_bond)
    #print('switch_traj:\n', switch_traj)

    if guess:
        guess_bonds(TRAJ)
    Bonds = list(TRAJ.top.bonds)
    #print('Bonds:', Bonds)
    bond_max_dist_list = guess_bond_max_dist(TRAJ)
    #print('bond_max_dist_list:', bond_max_dist_list)

    if switch_traj is None:
        switch_traj = np.ones((TRAJ.n_frames))
    for bnd in switch_bond:
        new_idx = abs(bnd)-1
        atm_A = Bonds[new_idx][0]
        atm_B = Bonds[new_idx][1]
        print('switching:', atm_A, atm_A.index, atm_B, atm_B.index, bond_max_dist_list[new_idx])
        disttraj = mdtraj.compute_distances(TRAJ, [[atm_A.index, atm_B.index]])[:,0]
        if bnd > 0:
            null_ids = np.where(disttraj>bond_max_dist_list[new_idx])
        else:
            null_ids = np.where(disttraj<bond_max_dist_list[new_idx])
        switch_traj[null_ids] = 0.0

    return switch_traj

def switch_by_dist(TRAJ, switch_dist, switch_traj=None): pass
