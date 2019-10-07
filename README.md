# gsm

A Python code of the Gaussian Streaming Model. For `C++` version, see the GSRSD part of https://github.com/wll745881210/CLPT_GSRSD

- Example
```python
import numpy as np
from gsm import gsm

gsrsd = gsm.gsm()
gsrsd.read_clpt(fn_xi='clpt/xi.txt', fn_v='clpt/v12.txt', fn_s='clpt/s12.txt')
ss = np.arange(1, 160, 5)
gsrsd.set_s_mu_sample(ss)
gsrsd.set_y_sample()

gsrsd.set_pars(nu=2.043, f_v=0.814, sFOG=2.0)
xi0, xi2 = gsrsd.c_xi()
```
