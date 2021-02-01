# Fast signal quality monitoring for coherent communications enabled by CNN-based EVM estimation

---
Published on Journal of Optical Communications and Networking (JOCN) Special Issue on Machine Learning Applied to QoT Estimation in Optical Networks

Links to the manuscript:
- Chalmers (open access authors' version): https://research.chalmers.se/en/publication/521965
- IEEEXplore: https://ieeexplore.ieee.org/document/9326316
- OSA: https://www.osapublishing.org/jocn/abstract.cfm?uri=jocn-13-4-B12

The dataset is available under DOI [10.21227/1684-a275](https://dx.doi.org/10.21227/1684-a275).

---
**Authors:** Yuchuan Fan, [Aleksejs Udalcovs](https://orcid.org/0000-0003-3754-0265), [Xiaodan Pang](https://orcid.org/0000-0003-4906-1704), [Carlos Natalino](https://orcid.org/0000-0001-7501-5547), [Marija Furdek](https://orcid.org/0000-0001-5600-3700), Sergei Popov, [Oskars Ozolins](https://orcid.org/0000-0001-9839-7488)

---

**Abstract:** We propose a fast and accurate signal quality monitoring scheme that uses convolutional neural networks (CNN) for error vector magnitude (EVM) estimation in coherent optical communications. We build a regression model to extract EVM information from complex signal constellation diagrams using a small number of received symbols. For the additive white Gaussian noise (AWGN) impaired channel, the proposed EVM estimation scheme shows a normalized mean absolute estimation error of 3.7% for quadrature phase shift keying (QPSK), 2.2% for 16-ary quadrature amplitude modulation (16QAM), and 1.1% for 64QAM signals, requiring only 100 symbols per constellation cluster in each observation period. Therefore, it can be used as a low-complexity alternative to conventional bit-error-rate (BER) estimation, enabling solutions for intelligent optical performance monitoring.

### Test the implementation of this work in Google Colab:

Link to the Google Colab file: https://colab.research.google.com/drive/1B7KkZAsFtUaBqsdPGQhYwxCLeYPVAdgr?usp=sharing

### Citing this work

Paper:
```
@ARTICLE{FanEtAl:JOCN:EVM:2021,
  author={Y. {Fan} and A. {Udalcovs} and X. {Pang} and C. {Natalino} and M. {Furdek} and S. {Popov} and O. {Ozolins}},
  journal={IEEE/OSA Journal of Optical Communications and Networking}, 
  title={Fast signal quality monitoring for coherent communications enabled by CNN-based EVM estimation}, 
  year={2021},
  volume={13},
  number={4},
  pages={B12-B20},
  doi={10.1364/JOCN.409704}
}
```

Dataset:
```
@data{FanEtAl:Dataset:EVM:2021,
    doi = {10.21227/1684-a275},
    url = {https://dx.doi.org/10.21227/1684-a275},
    author={Y. {Fan} and A. {Udalcovs} and X. {Pang} and C. {Natalino} and M. {Furdek} and S. {Popov} and O. {Ozolins}},
    publisher = {IEEE Dataport},
    title = {2020_JOCN_Constellation_Dataset},
    year = {2020}
}
```

### Running this code

To be able to run the [Python file](JOCN_EVM_estimation.py) in this repository, you should have a Python environment with version 3.7 (it might work with newer versions) and the following packages:
- TensorFlow 2.x (tested with TF 2.4.1)
- Matplotlib
- imageio
- Scikit-Learn

You should also download the dataset mentioned above and unzip it within this project folder.

### Contact

This repository is a fork from [Yuchuan Fan's repository](https://github.com/JhoneFan/2020_JOCN_EVM_Estimation_using_CNN) 
and is maintained by Carlos Natalino [[Twitter](https://twitter.com/NatalinoCarlos)], who can be contacted through carlos.natalino@chalmers.se.
