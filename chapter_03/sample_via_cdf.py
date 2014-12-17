from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.stats import uniform

def sample_via_cdf(x, p, nsamp):
    # get normalized cumulative distribution
    cdf = cumtrapz(p, x, initial=0)
    cdf = cdf/cdf.max()
    # get interpolator
    interp = interp1d(cdf, x)
    # get uniform samples over cdf
    cdf_samp = uniform.rvs(size=nsamp)
    return interp(cdf_samp)
