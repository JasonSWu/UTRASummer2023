import numpy as np

def BeckersRixen(M0, nmin=None, nmax=None, perc=None, tol=None):
    nnow = 1
    sm = M0.shape
    
    if nmin is None:  # Default number of modes to use
        nmin = 1
    if nmax is None:  # Default number of modes to use
        nmax = round(np.min(sm) / 2)
    if perc is None:  # Default percentage of matrix to use as error est.
        perc = 10
    if tol is None:  # Default error tolerance
        tol = 1e-4
        
    nmax = min(np.min(sm) - 1, nmax)
    
    # Count the zeros by row
    zcount = np.sum(M0 == 0, axis=1) / sm[1]
    
    # Add on the error estimate rows
    added = int(np.ceil(sm[1] * perc / 100))   # How many to add?
    Mp = np.zeros((sm[0], sm[1] + added))  # A bigger matrix with the added rows
    Mp[:, :sm[1]] = M0
    Md = Mp.copy()
    ndx = np.random.randint(sm[1], size=added)   # Make a random guess of columns to include
    
    for ii in range(added):
        while np.prod(M0[:, ndx[ii]]) == 0:
            ndx[ii] = np.random.randint(sm[1])  # Try again until you find a column without gaps
        Mp[:, sm[1] + ii] = M0[:, ndx[ii]]  # Duplicate the original nonzero data
        Md[:, sm[1] + ii] = Mp[:, sm[1] + ii] * (np.random.rand(sm[0]) > zcount)  # Zeroing some

    zdx = np.where(Md == 0)
    adx = np.arange(sm[1], sm[1] + added)
    
    initerror = np.sum((Mp[:, adx] - Md[:, adx]) ** 2)  # The initial error, now let's reduce it!
    itererror = initerror
    nerror = np.zeros(nmax)
    
    # Here's the loop for the number of modes used
    while nnow <= nmax:
        itererror = initerror
        errorup = 2 * initerror
        olderrorup = 3 * initerror
        Md[zdx] = 0
        Mdnew = Md.copy()
        
        while np.abs(errorup / olderrorup - 1) > tol:
            Md[zdx] = Mdnew[zdx]
            U, S, V = np.linalg.svd(Md, full_matrices=False)
            
            for jj in range(nnow + 1, len(S)):
                S[jj] = 0   # Truncate the modes
                U[:, jj] = 0   # Truncate the modes
                V[jj] = 0   # Truncate the modes
            
            Mdnew = U @ np.diag(S) @ V  # Form the new guess
            
            itererror = np.sum((Mp[:, adx] - Mdnew[:, adx]) ** 2)  # Find new error
            olderror = np.sum((Mp[:, adx] - Md[:, adx]) ** 2)  # Find error update
            
            olderrorup = errorup
            errorup = np.sum((Mp[:, adx] - Md[:, adx]) ** 2) - itererror  # Find error update
            
        nerror[nnow-1] = itererror  # Check to ensure error is decreasing with increasing n
        if nnow > 1 and nerror[nnow-1] > nerror[nnow-2]:
            break
        
        nnow += 1  # Take more nodes
    
    Ma = Md[:, :sm[1]]
    
    return Ma, U, S, V