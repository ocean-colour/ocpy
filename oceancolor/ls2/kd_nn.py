import numpy as np
import warnings

from oceancolor.ls2 import io as ls2_io

from IPython import embed

def Kd_NN_MODIS(Rrs, sza, lambda_):#, Kd_NN_LUT_MODIS):
    """
    Implements the neural network (NN) algorithm to calculate the diffuse 
    attenuation coefficient of downwelling planar irradiance (Kd) at a
    preselected output light wavelength (lambda) using input remote-sensing
    reflectance (Rrs) at MODIS wavelengths and solar zenith angle (sza).

    Args:
        Rrs: (1, 5) float array. Values of spectral remote-sensing reflectance
            (sr^-1) at MODIS light wavelengths: 443, 488, 531, 547, 667 [nm].
        sza: (1, 1) float. Solar zenith angle [deg] associated with input Rrs
            values.
        lambda_: (1, 1) float. Output light wavelength [nm] at which the desired
            value of Kd is estimated for a given input.
        Kd_NN_LUT_MODIS: (1, 1) dict. Structure containing three required
            look-up tables (LUTs). Can be loaded via a similar function (e.g., 
            load_Kd_NN_LUT_MODIS).

            - weights_1: LUT with weights and biases from NN for clear waters
            (where Rrs(488)/Rrs(547) >= 0.85).
            - weights_2: LUT with weights and biases from NN for turbid waters
            (where Rrs(488)/Rrs(547) < 0.85).
            - train_switch: LUT with means and standard deviations of 40,000 
            inputs and outputs used to train the NN.

    Returns:
        Kd: (1, 1) float. The estimated value of the average diffuse attenuation
            coefficient of downwelling planar irradiance [m^-1] between the sea
            surface and first attenuation depth at the output light wavelength
            (lambda) for input spectral Rrs and sza.
    %Version 1.1 (v1.1)
    %
    %Version History: 
    %2018-04-04: Original implementation in C written by David Dessailly
    %2020-03-23: Original Matlab version, D. Jorge 
    %2022-09-01: Revised Matlab version, M. Kehrli
    %2022-11-03: Final Revised MATLAB version (v1.0), M. Kehrli, R. A. Reynolds
    %and D. Stramski
    %2023-10-10: Corrected weights and biases in KdNN LUT for clear waters
    %(v1.1)
    """
    # Load up the tables
    weights_1, weights_2, train_switch = ls2_io.load_Kd_tables()

    # Check function arguments and existence of LUTs
    Rrs = np.asarray(Rrs).reshape(1, -1)
    sza = np.asarray(sza).reshape(1, -1)
    lambda_ = np.asarray(lambda_).reshape(1, -1)

    # Refractive index of seawater
    nw = 1.34

    #calculation of muw [dim], the cosine of the angle of refraction of the
    #solar beam just beneath the sea surface
    #muw = np.cos(np.deg2rad(np.arcsin(np.sin(np.deg2rad(sza)) / nw)))
    muw = np.cos(np.arcsin(np.sin(np.deg2rad(sza))/nw))

    # Combine inputs
    #inputs = np.concatenate((Rrs, [lambda_], [muw]))
    inputs = np.concatenate([Rrs, lambda_, muw], axis=1)

    # Access data from LUT (assuming similar structure as MATLAB)
    #train_switch = Kd_NN_LUT_MODIS["train_switch"]
    #mu = train_switch["MEAN"][1:-1]  # exclude Rrs(667) for clear waters
    means = train_switch.MEAN.values
    stds = train_switch.STD.values
    #std = train_switch["STD"][1:-1]

    # Water type determination using blue-green band ratio
    ratio = inputs[0, 1] / inputs[0, 3]

    # Build NN for clear waters
    if ratio >= 0.85:
        #read in NN weights and biases for clear waters
        weights = weights_1
        #number of input neurons in the NN
        #number of neurons on the first hidden layer in the NN
        #number of neurons on the second hidden layer in the NN
        #number of neurons on the output layer in the NN
        ne, nc1, nc2, ns = 6, 8, 6, 1

        # Extract weights and biases from LUT
        b1, b2, bout = weights["b1"], weights["b2"], weights["bout"]
        #w1, w2, wout = weights["w1"].reshape(nc1, ne), weights["w2"].reshape(nc2, nc1), weights["wout"].reshape(ns, nc2)
        w1, w2, wout = weights["w1"], weights["w2"], weights["wout"]

        # Nans
        b1 = b1[np.isfinite(b1)].values
        b2 = b2[np.isfinite(b2)].values
        bout = bout[np.isfinite(bout)].values
        w1 = w1[np.isfinite(w1)].values
        w2 = w2[np.isfinite(w2)].values
        wout = wout[np.isfinite(wout)].values
        # Reshape
        w1 = w1.reshape(nc1, ne)  # These could be backwards
        w2 = w2.reshape(nc2, nc1)
        wout = wout.reshape(ns, nc2)


        # Check for negative Rrs input
        if np.any(inputs[0, :4] < 0):
            warnings.warn("Negative Rrs input detected. Kd set to NaN.")
            return np.nan

        #mean and stadard deviation of input and output parameters from
        #training dataset; remove Rrs(667) data for clear waters
        mu = np.array(means[1:5].tolist()+means[6:].tolist())
        std = np.array(stds[1:5].tolist()+stds[6:].tolist())
        #mu = mu([2:5,7:9]);
        #std = std([2:5,7:9]);


        #set NN input for clear waters
        #x = inputs([1:4,6:7]);
        keep = np.ones_like(inputs, dtype=bool)
        keep[:,4] = False
        #x = inputs[0:4]+inputs[5:7]
        x = inputs[keep].reshape(1,-1)

        # Normalize inputs
        x_N = np.ones_like(x)
        
        for j in range(6):
            x_N[:, j] = (2/3) * ((x[:, j] - mu[j]) / std[j])

        embed(header='Kd_NN_MODIS 123')

    # Build NN for turbid waters
    elif ratio < 0.85:
        #read in NN weights and biases for turbid waters
        weights = Kd_NN_LUT_MODIS["weights_2"]
        #number of input neurons in the NN
        #number of neurons on the first hidden layer in the NN
        #number of neurons on the second hidden layer in the NN
        #number of neurons on the output layer in the NN
        ne, nc1, nc2, ns = 7, 9, 6, 1

        # Extract weights and biases from LUT
        b1, b2, bout = weights["b1"], weights["b2"], weights["bout"]
        w1, w2, wout = weights["w1"].reshape(nc1, ne), weights["w2"].reshape(nc2, nc1), weights["wout"].reshape(ns, nc2)

        # Check for negative Rrs input
        if np.any(inputs[0] < 0):
            warnings.warn("Negative Rrs input detected. Kd set to NaN.")
            return np.nan

        # Normalize inputs
        x_N = np.empty_like(inputs)
        for j in range(7):
            x_N[:, j] = (2/3) * ((inputs[:, j] - mu[j]) / std[j])

    # Kd inversion 
    embed(header='Kd_NN_MODIS 150')
    Kd = MLP_Kd(x_N, w1, b1, w2, b2, wout, bout, 
                np.array(mu[-1]).reshape(1,-1), 
                np.array(std[-1]).reshape(1,-1)) 

    return Kd


def MLP_Kd(x, w1, b1, w2, b2, wout, bout, muKd, stdKd):
    """
    This function computes the output of the neural network based on the 
    provided inputs, weights, biases, and normalization constants.

    Args:
        x: (rx, ni) float array. The inputs to the NN, where rx is the number 
            of samples and ni is the number of input neurons (6 for clear 
            waters, 7 for turbid waters).
        w1: (nc1, ni) float array. Connection weights of the first hidden layer.
        b1: (nc1, 1) float array. Neuron bias of the first hidden layer.
        w2: (nc2, nc1) float array. Connection weights of the second hidden layer.
        b2: (nc2, 1) float array. Neuron bias of the second hidden layer.
        wout: (1, nc2) float array. Connection weights of the output layer.
        bout: (1, 1) float array. Neuron bias of the output layer.
        muKd: (1, 1) float. The mean output of Kd values from NN training.
        stdKd: (1, 1) float. The standard deviation output of Kd values from 
            NN training.

    Returns:
        Kd: (rx, 1) float array. The estimated Kd value for each sample obtained 
            from the NN.
    """
    # Get number of samples
    rx, _ = x.shape

    # Forward propagation through the NN with tanh activation
    a = 1.715905 * np.tanh(0.6666667 * (np.dot(x, w1.T) + np.ones((1, rx)) * b1))
    b = 1.715905 * np.tanh((2.0 / 3.0) * (np.dot(a, w2.T) + np.ones((1, rx)) * b2))
    y = np.dot(b, wout.T) + bout * np.ones((rx, 1))

    # Denormalize the output (assuming log-transformed training data)
    Kd = 10.0 ** (1.5 * y * stdKd + muKd)

    embed(header='Kd_NN_MODIS 190')
    return Kd
