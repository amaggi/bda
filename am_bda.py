from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


def get_pdf_quantiles(pdf, x, q_list):
    # need the cumulative probability density function
    # to get the quantiles
    cdf = cumtrapz(pdf, x=x, initial=0)
    quant = interp1d(cdf, x)
    x_out = quant(q_list)
    
    # also get the probability at the quantile poitns
    pr = interp1d(x, pdf)
    pr_out = pr(x_out)

    return x_out, pr_out

