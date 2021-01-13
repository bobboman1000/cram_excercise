################################################
##                                            ##
##   Voluntary Problem Set                    ##
##   Getting to Know Data                     ##
##                                            ##
################################################

# Note: Please round all numeric answers to two decimals


########### Task A ###########

# The pandas data frame r_d has x rows and y columns
a_n_rows = 5383
a_n_cols = 43

# monthly log return of market portfolio at Dec 2019
a_1N_r_m = 0.01

# quarterly log return of market portfolio at Q4 2019
a_1N_r_q = 0.04

# annual log return of market portfolio at 2019
a_1N_r_a = 0.23

# Do annualized mean returns, annualized return volatilities
# differ depending on whether you compute these using daily, monthly, 
# quarterly or annual log returns? ['yes'/'no']
a_r_differs = 'no'
a_vol_differs = 'no'

# Estimating annualized mean returns using daily data (instead of annual data) changes
# precision by factor ...?
a_precision_factor = 1.02


########### Task B ###########

# Cross-sectional average of return skewness for daily/monthly/quarterly/annual freq
b_skew_d = 0.03
b_skew_m = -0.63
b_skew_q = -0.85
b_skew_a = -0.8

# The lower the return frequency, the ['higher'/'lower'] the skeweness
b_q1 = 'lower'

# All else equal, looking at average skewness, the lower the return frequency, 
# the ['more'/'less'] Gaussian are equity returns
b_q2 = 'less'


########### Task C ###########

# Cross-sectional average of return kurtosis for daily/monthly/quarterly/annual freq
c_kurt_d = 21.23
c_kurt_m = 3.82
c_kurt_q = 2.65
c_kurt_a = 0.99

# The lower the return frequency, the ['higher'/'lower'] the kurtosis
c_q1 = 'lower'

# All else equal, looking at average kurtosis, the lower the return frequency, 
# the ['more'/'less'] Gaussian are equity returns
c_q2 = 'more'


########### Task D ###########

# Variance explained by PC1 of daily/monthly/quarterly/annual returns
d_pc1_explained_var_d = 0.46
d_pc1_explained_var_m = 0.44
d_pc1_explained_var_q = 0.51
d_pc1_explained_var_a = 0.53

# Is there any noticeable difference in the variance explained by the respective first principal 
# component? ['yes'/'no']
d_q1 = 'no'


########### Task E ###########

# For the market portfolio, can the density be modelled as Gaussian at daily/monthly/quarterly/annual
# frequency, running a Jarque-Bera test with a type I error of 10%? ['yes'/'no']
e_1N_density_is_gaussian_d = 'no'
e_1N_density_is_gaussian_m = 'no'
e_1N_density_is_gaussian_q = 'no'
e_1N_density_is_gaussian_a = 'no'
