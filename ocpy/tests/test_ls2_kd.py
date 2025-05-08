import numpy as np

from oceancolor.ls2 import kd_nn

from IPython import embed


def test_kd_nn():
    # Rrs
    Rrs = [[0.00663061627778120,0.00569886961307466,
        0.00261783091270704,0.00206376550494829,0.000155664525215032],
        [0.00278698717627185,0.00555055435696759,0.00965557516418457,
        0.0114125090066875,0.00402413455723548]]

    sza = [30,60,51,30,60,0]
    ilambda = [430,531,645,570,520]#,430;700;400;488;555;620;500;650;610;...


    # Clear
    i = 0
    Kd0 = kd_nn.Kd_NN_MODIS(Rrs[i],sza[i],ilambda[i])
    assert np.isclose(Kd0, 0.04600481236, atol=0.0001)

    # Turbid
    i = 1
    Kd1 = kd_nn.Kd_NN_MODIS(Rrs[i],sza[i],ilambda[i])
    assert np.isclose(Kd1, 0.5623341777, atol=0.0001)

test_kd_nn()