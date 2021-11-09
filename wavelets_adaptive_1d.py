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

def forward_wt(signif, y, jlvl, order_p, order_u):
    #(y, jlvl, order_p, order_u, cv):
    y0 = copy.deepcopy(y)
    n_range_p = np.arange(-int((order_p+1)/2), \
                order_p-int((order_p+1)/2)+1).astype(np.float64)
    n_range_u = np.arange(-int((order_u+1)/2), \
                order_u-int((order_u+1)/2)+1).astype(np.float64)
    ende = len(y)
    y[signif == False] = 0.
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
        
    #non_zero = np.sum(np.abs(y) >= cv * np.max(np.abs(y0)))
    #y[np.abs(y) < cv * np.max(np.abs(y0))] = 0.

    return y#, non_zero

def inverse_wt(signif, y, jlvl, order_p, order_u):
    n_range_p = np.arange(-int((order_p+1)/2), \
                order_p-int((order_p+1)/2)+1).astype(np.float64)
    n_range_u = np.arange(-int((order_u+1)/2), \
                order_u-int((order_u+1)/2)+1).astype(np.float64)
    ende = len(y)
    y[signif == False] = 0.
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

def get_significance(y, cv, jlvl):
    signif_ = np.zeros((len(y),), dtype=bool)
    for j in range(jlvl-1, 0, -1):
        s = 2**(jlvl-j-1)
        
        D_ind = np.arange(s, len(y), 2*s)
        C_ind = np.arange(0, len(y), 2*s)
        signif_[D_ind] = (np.abs(y[D_ind]) >= cv) | (signif_[D_ind])
        
        if j == 1:
            signif_[C_ind] = True
    return signif_

def get_adjust(y, L_sign, jlvl_in, j_max):
    ende = len(y)
    D_present = np.zeros((jlvl_in,), dtype=bool)
    D_present[0] = True
    for j in range(0, jlvl_in-1):
        s = 2**(jlvl_in-j-1)
        D_ind = np.arange(s, ende, 2*s)
        D_present[j+1] = True if np.sum(L_sign[D_ind]) > 0 else False
    
    jlvl_out = max(jlvl_in-1, min(\
                max([k0 for k0,k in enumerate(D_present)]), j_max))
    
    y_out, L_out = [], []
    if jlvl_out > jlvl_in:
        s = 2.**(jlvl_out-jlvl_in)
        nprd = ende % 2
        ende = int(s * (ende - nprd) + nprd)
        y_out = np.zeros((ende,))
        L_out = np.zeros((ende,), dtype=bool)
        C_ind = np.arange(0, ende, s).astype(int)
        
        y_out[C_ind] = y
        L_out[C_ind] = L_sign
    elif jlvl_out == jlvl_in:
        y_out = y
        L_out = L_in
    elif jlvl_out < jlvl_in:
        s = 2.**(jlvl_in-jlvl_out)
        C_ind = np.arange(0, ende, s).astype(int)
        
        nprd = ende % 2
        ende = int((ende - nprd) / s + nprd)
        y_out = np.zeros((ende,))
        L_out = np.zeros((ende,), dtype=bool)
        
        y_out = y[C_ind]
        L_out = L_sign[C_ind]

    return y_out, L_out, jlvl_out


def get_adjacent(L_in, jlvl):
    L_adj = copy.deepcopy(L_in)
    ende = len(L_in)
    for j in range(jlvl-1, -1, -1):
        s = 2**(jlvl-j)
        
        S_ind = []
        if j == 0:
            S_ind = np.arange(0, ende, s).astype(int)
        else:
            S_ind = np.arange(s, ende, 2*s).astype(int)
            
        S_ind = S_ind[L_in[S_ind]==1]
        
        if len(S_ind) > 0:
            aux_arr = (np.array([-1, 1])*s/2**(min(j+1, \
                                    jlvl)-j)).astype(int)
            for i in aux_arr:
                L_adj[S_ind[(S_ind + i >= 0) & \
                        (S_ind+i<ende)]+i] = True    
    return L_adj


def reconstruction_check(jlvl, order_p, L_in):
    ende = len(L_in)
    L_out = copy.deepcopy(L_in)
    n_range = np.arange(-int((order_p+1)/2), \
                        order_p-int((order_p+1)/2)+1)
    
    for j in np.arange(jlvl-1, 0, -1):
        s = 2.**(jlvl-j-1)
        
        D_ind = np.arange(s, ende, 2*s).astype(int)
        D_ind = D_ind[np.where(L_out[D_ind]==True)]
        
        for i in D_ind:
            n_range_corr = D_range_corr(i, ende, s, n_range)
            order_corr = len(n_range_corr) - 1
            for k in np.arange(order_corr + 1):
                L_out[int(i + (2*n_range_corr[k]+1)*s)] = \
                    L_out[int(i + (2*n_range_corr[k]+1)*s)] | L_out[int(i)]
                

    return L_out

def adaptive_transform_1d(y, jlvl, order_p, order_u, cv, jmx):
    signif = np.ones((len(y),), dtype=bool)
    y_forw = forward_wt(signif, y, jlvl, order_p, order_u)
    signif = get_significance(y_forw, cv, jlvl)
    y_new, signif, jlvl_new = get_adjust(y, signif, jlvl, jmx)
    signif = get_adjacent(signif, jlvl_new)
    L = reconstruction_check(jlvl_new, order_p, signif)
    y_out = inverse_wt(L, y_new, jlvl_new, order_p, order_u)
    
    return y_out
