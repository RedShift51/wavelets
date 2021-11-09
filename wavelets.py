import os
import numpy as np
import copy
import matplotlib.pyplot as plt

def wgh(k0, n_l, n_h):
    array = np.arange(n_l, n_h + 1)
    i0 = array[array != k0].astype(np.float64)
    return np.prod(i0 + 0.5) / np.prod(i0 - k0)

def D_range_corr(Ind, n, s, N_range):
    N_range_corr = Ind + s * (2 * N_range + 1)
    N_range_corr = N_range_corr + max(0, N_range_corr[0]) - \
                    N_range_corr[0]
    N_range_corr = N_range_corr + min(n-1, N_range_corr[-1]) - \
                    N_range_corr[-1]
    N_range_corr = ((N_range_corr[N_range_corr>=0]-Ind)/s-1)/2
    return N_range_corr

def C_range_corr(Ind, n, s, N_range):
    N_range_corr = Ind + s * (2 * N_range + 1)
    N_range_corr = N_range_corr + max(s, N_range_corr[0]) - \
                    N_range_corr[0]
    N_range_corr = N_range_corr + min(n-s-1, N_range_corr[-1]) - \
                    N_range_corr[-1]
    N_range_corr = ((N_range_corr[N_range_corr>=0]-Ind)/s-1)/2
    return N_range_corr

def forward_wt(y, jlvl, order_p, order_u, cv):
    #(y, jlvl, order_p, order_u, cv):
    y0 = copy.deepcopy(y)
    n_range_p = np.arange(-int((order_p+1)/2), \
                order_p-int((order_p+1)/2)+1).astype(np.float64)
    n_range_u = np.arange(-int((order_u+1)/2), \
                order_u-int((order_u+1)/2)+1).astype(np.float64)
    ende = len(y)

    for j in range(jlvl-1, 0, -1):
        s = 2**(jlvl-j-1)
        D_ind = np.arange(s, ende, 2*s)
        C_ind = np.arange(0, ende, 2*s)
        
        # predict
        for i in D_ind:
            range_corr = D_range_corr(i, ende, s, n_range_p).astype(np.float64)
            order_corr = len(range_corr) - 1
            for k in range(order_corr+1):
                y[i] = y[i] - y[int(i+(2*range_corr[k]+1)*s)]*\
                    wgh(range_corr[k], range_corr[0], range_corr[-1])
        
        # update
        for i in C_ind:
            range_corr = C_range_corr(i, ende, s, n_range_u).astype(np.float64)
            order_corr = len(range_corr) - 1
            for k in range(order_corr+1):
                y[i] = y[i] + 0.5*y[int(i+(2*range_corr[k]+1)*s)]*\
                    wgh(range_corr[k], range_corr[0], range_corr[-1])
        
    non_zero = np.sum(np.abs(y) >= cv * np.max(np.abs(y0)))
    y[np.abs(y) < cv * np.max(np.abs(y0))] = 0.

    return y, non_zero

def inverse_wt(y, jlvl, order_p, order_u):
    n_range_p = np.arange(-int((order_p+1)/2), \
                order_p-int((order_p+1)/2)+1).astype(np.float64)
    n_range_u = np.arange(-int((order_u+1)/2), \
                order_u-int((order_u+1)/2)+1).astype(np.float64)
    ende = len(y)
    
    for j in range(1, jlvl):
        s = 2**(jlvl-j-1)
        
        D_ind = np.arange(s, ende, 2*s)
        C_ind = np.arange(0, ende, 2*s)
        
        # update
        for i in C_ind:
            range_corr = C_range_corr(i, ende, s, n_range_u).astype(np.float64)
            order_corr = len(range_corr) - 1
            for k in range(order_corr+1):
                y[i] = y[i] - 0.5*y[int(i+(2*range_corr[k]+1)*s)]*\
                    wgh(range_corr[k], range_corr[0], range_corr[-1])

        # predict
        for i in D_ind:
            range_corr = D_range_corr(i, ende, s, n_range_p).astype(np.float64)
            order_corr = len(range_corr) - 1
            for k in range(order_corr+1):
                y[i] = y[i] + y[int(i+(2*range_corr[k]+1)*s)]*\
                    wgh(range_corr[k], range_corr[0], range_corr[-1])

                
    return y


func = lambda x: (np.sin(10.*x) + np.exp(- 0.25 * ((x-5.)**2) )).astype(np.float64) 

cvals = (10**np.arange(-1., -7., -1.)).astype(np.float64) #10**np.arange(2, )
err_list = []
nz_list = []
param_j = 10

for c0,cval in enumerate(cvals):
    y_ = func(np.arange(0., 10.001, 10./ 4096.)).astype(np.float64)
    #y_ = np.arange(0., 10.001, 10./ 4096.).astype(np.float64)
    ans_, nz = forward_wt(copy.deepcopy(y_), param_j, 3, 3, cval)
    y_rest = inverse_wt(copy.deepcopy(ans_), param_j, 3, 3)
    
    nz_list.append(copy.deepcopy(nz))
    err_list.append(copy.deepcopy(np.max(np.abs(y_-y_rest))))
    print(c0, len(cvals), err_list[-1])




plt.figure(figsize=(12,25))

nz_list = np.array(nz_list).astype(float)
err_list = np.array(err_list).astype(float)

plt.subplot(211)
plt.plot(cvals, err_list)
plt.scatter(cvals, err_list)

plt.grid()
plt.yscale("log")
plt.xscale("log")

plt.subplot(212)
plt.plot(nz_list, err_list)
plt.scatter(nz_list, err_list)

plt.plot(nz_list, cvals)
plt.scatter(nz_list, cvals)

plt.plot(nz_list, 1000000*nz_list**(-4) )
plt.scatter(nz_list, 1000000*nz_list**(-4))

plt.legend(["err(nz)", "cvals(nz)", "nz**(-4) (nz)"])

plt.grid()

plt.yscale("log")
plt.xscale("log")

plt.show()
