from scipy.stats import skewnorm

def sample_asymmetric_distribution(Dl, err_neg, err_pos, num_samples):
    """Example function for the simulation of a covariance matrix from asymmetric errors."""
    mean = Dl
    std_dev = (err_pos + err_neg) / 2  #std deviation
    skew_param = (err_pos - err_neg) / (err_pos + err_neg) * 10  #Skewness factor

    samples = skewnorm.rvs(a=skew_param, loc=mean, scale=std_dev, size=num_samples)
    return samples

