##########################################################
##                                                      ##
## Voluntary Problem Set                                ##
## Linear Factor Models                                 ##
## Fama MacBeth                                         ##
##                                                      ##
##########################################################

# Note: Please round all numeric answers to two decimals


########### Task A.1 ###########

# Write down the betas for company XLNX
beta_XLNX_MKT = 1.17
beta_XLNX_SMB = 0.06
beta_XLNX_HML = -0.8
beta_XLNX_RMW = -1.1
beta_XLNX_CMA = -0.18
beta_XLNX_Rev = -0.08
beta_XLNX_Mom = -0.28


########### Task A.2 ###########

# For which 'investment style' is XLNX well suited:

# strong business cycle sensitivity: ['yes'/'no']
a2_q1 = 'yes'

# growth firm: ['yes'/'no']
a2_q2 = 'yes'

# lots of operating profit: ['yes'/'no']
a2_q3 = 'no'

# agressive investments: ['yes'/'no']
a2_q4 = 'no'

# short-term reversal return strategy: ['yes'/'no']
a2_q5 = 'no'

# positive momentum: ['yes'/'no']
a2_q6 = 'no'


########### Task A.3 ###########

# How much of the variance in the XLNX's stock return is explained by the 7 factors?
variance_explained_FF7_XLNX = 0.45


########### Task A.4 ###########

# What fraction of risk in XLNX is diversifiable, assuming the FF7 model holds?
variance_diversifiable_FF7_XLNX = 0.55


########### Task A.5 ###########

# Which firm is best explained by the FF7 factors (give ticker as answer)? 
# How large is its respective adj-R2? 
best_fit_TSreg = 'JPM'
best_R2_TSreg = 0.71
beta_A5_MKT = 1.23
beta_A5_SMB = -0.35
beta_A5_HML = 1.30
beta_A5_RMW = -0.55
beta_A5_CMA = -0.55
beta_A5_Rev = -0.02
beta_A5_Mom = -0.20

# Characterize its 'investment style' in terms of its beta coefficients and in terms of a qualitative assessment:
# 'The firm whose return is best explained by FF is characterized as follows: a ['large'/'small']
a5_q1 = 'large'
# ...firm with a ['cyclical'/'countercyclical']
a5_q2 = 'cyclical'
# ...business model that based on balance sheet fundamentals can be classified as rather ['expensive'/'cheap']
a5_q3 = 'cheap'
# ...with ['lots of'/'low']
a5_q4 = 'low'
# ...operating profits and ['aggressive'/'passive']
a5_q5 = 'aggressive'
# ...investment strategies and ['positive'/'negative']
a5_q6 = 'negative'
# ...return momentum'


########### Task A.6 ###########

# The Fama-MacBeth market price of risks (annualized in percent) and respective t-stats are
lambda_MKT = 0
lambda_SMB = 0
lambda_HML = 0
lambda_RMW = 0
lambda_CMA = 0
lambda_Rev = 0
lambda_Mom = 0
t_lambda_MKT = 0
t_lambda_SMB = 0
t_lambda_HML = 0
t_lambda_RMW = 0
t_lambda_CMA = 0
t_lambda_Rev = 0
t_lambda_Mom = 0



########### Task A.7 ###########

# What is the main reason for these disappointing results?

# (1) realized lambdas are very noisy, a direct result of the low signal to noise ratio that 
# daily returns have for expected returns ['yes'/'no']
a7_q1 = 'yes'

# (2) the sample is too short ['yes'/'no']
a7_q2 = 'no'

# (3) there are not enough assets in the cross-sectional regression ['yes'/'no']
a7_q3 = 'no'

# (4) an error in the code ['yes'/'no']
a7_q4 = 'no'

# (5) realized betas are very noisy ['yes'/'no']
a7_q5 = 'no'
