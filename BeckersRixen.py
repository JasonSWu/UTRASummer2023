import numpy as np

def BeckersRixen(M0, test, nmin=1, nmax=None, val_perc=10, tol=1e-4):
    nnow = 1
    sm = M0.shape
    m, n = sm
    test_m, test_n = test.shape
    assert m == test_m
    
    if nmax is None:  # Default number of modes to use
        nmax = round(np.min(sm) / 2)
        
    nmax = min(np.min(sm) - 1, nmax)
    
    # Count the zeros by row, want to match ratio of zeros in test set to ensure it works well
    zcount = np.sum(test == 0, axis=1) / test_n
    
    # Add on the error estimate rows
    val_n = int(np.ceil(n * val_perc / 100))   # How many to train on?
    Mp = M0.copy()
    Md = Mp.copy()
    ndx = np.random.randint(n, size=val_n)   # Make a random guess of columns to include
    
    for ii in range(val_n):
        give_up = 100
        while np.prod(M0[:, ndx[ii]]) == 0 and give_up > 0:
            ndx[ii] = np.random.randint(n)  # Try again until you find a column without gaps (is this necessary?)
            give_up -= 1
        Md[:, ndx[ii]] *= (np.random.rand(m) > zcount)  # Zeroing some

    Md = np.concatenate((Md, test), axis = 1) # Append test set so it's also affected by the algorithm

    zdx = np.where(Md == 0)
    
    initerror = np.sum((Mp[:, ndx] - Md[:, ndx]) ** 2)  # The initial error, now let's reduce it!
    itererror = initerror
    nerror = np.zeros(nmax)
    
    # Here's the loop for the number of modes used
    # Don't reset the places at zdx to 0, BeckersRixen paper says not to
    while nnow <= nmax:
        itererror = initerror
        errorup = 2 * initerror
        olderrorup = 3 * initerror
        Mdnew = Md.copy()
        
        while np.abs(errorup / olderrorup - 1) > tol:
            Md[zdx] = Mdnew[zdx]
            U, S, V = np.linalg.svd(Md, full_matrices=False)
            
            Mdnew = np.dot(U[:,:nnow + 1], np.dot(np.diag(S[:nnow + 1]), V[:nnow + 1]))  # Form the new guess
            
            itererror = np.sum((Mp[:, ndx] - Mdnew[:, ndx]) ** 2)  # Find new error
            olderror = np.sum((Mp[:, ndx] - Md[:, ndx]) ** 2)  # Find error update
            
            olderrorup = errorup
            errorup = np.sum((Mp[:, ndx] - Md[:, ndx]) ** 2) - itererror  # Find error update
            
        nerror[nnow-1] = itererror  # Check to ensure error is decreasing with increasing n
        if nnow > 1 and nerror[nnow-1] > nerror[nnow-2]:
            break
        
        nnow += 1  # Take more nodes
    
    Ma = Md
    
    return Ma