### ORIGINAL: https://github.com/DJArmstrong/TransitSOM

import numpy as np
def CreateSOM(SOMarray,niter=500,learningrate=0.1,learningradius=None,somshape=(20,20),outfile=None):
    """
        Trains a SOM, using an array of pre-prepared lightcurves. Can save the SOM to text file.
        Saved SOM can be reloaded using LoadSOM() function.
    
            Args:
                SOMarray: Array of normalised inputs (e.g. binned transits), of shape [n_inputs,n_bins]. n_inputs > 1
                niter: number of training iterations, default 500. Must be positive integer.
                learningrate: alpha parameter, default 0.1. Must be positive.
                learningradius: sigma parameter, default the largest input SOM dimension. Must be positive.
                somshape: shape of SOM to train, default (20,20). Currently must be 2 dimensional tuple (int, int). Need not be square.
                outfile: File path to save SOM Kohonen Layer to. If None will not save.
    
            Returns:
                The trained SOM object
    """   
    try:
        import sys
        sys.path.insert(1, './otherBasel')
        import somtools
        import selfsom
    except:
        print('Accompanying libraries not in PYTHONPATH or current directory')
        return 0
        
    try:
        assert niter >= 1, 'niter must be >= 1.'
        assert type(niter) is int, 'niter must be integer.'
        assert learningrate > 0, 'learningrate must be positive.'
        if learningradius:
            assert learningradius > 0, 'learningradius must be positive.'
        assert len(somshape)==2, 'SOM must have 2 dimensions.'
        assert type(somshape[0]) is int and type(somshape[1]) is int, 'somshape must contain integers.'
        assert len(SOMarray.shape)==2, 'Input array must be 2D of shape [ninputs, nbins].'
        assert SOMarray.shape[0]>1, 'ninputs must be greater than 1.'
    except AssertionError as error:
        print(error)
        print('Inputs do not meet requirements. See help')
        return 0
        
    nbins = SOMarray.shape[1]
    
    #default learning radius
    if not learningradius:
        learningradius = np.max(somshape)
    
    #define som initialisation function
    def Init(sample):
        return np.random.uniform(0,2,size=(somshape[0],somshape[1],nbins))
    
    #initialise som
    som = selfsom.SimpleSOMMapper(somshape,niter,initialization_func=Init,learning_rate=learningrate,iradius=learningradius)

    #train som
    som.train(SOMarray)

    #save
    if outfile:
        somtools.KohonenSave(som.K,outfile)
    
    #return trained som
    return som
