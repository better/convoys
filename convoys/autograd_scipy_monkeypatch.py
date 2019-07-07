# Monkey patching scipy to avoid issue with autograd
# Newest scipy moved logsumexp from scipy.misc to scipy.special
# This is fixed in latest autograd, but it's not on PyPI yet

import scipy.misc
if not hasattr(scipy.misc, 'logsumexp'):
    import scipy.special
    scipy.misc.logsumexp = scipy.special.logsumexp
