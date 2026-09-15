# MeanVariancePortfolio — User Guide

## 1. Overview

MeanVariancePortfolio is a portfolio optimisation class which uses historical asset returns to find an optimal portfolio using a range of methods. It takes a DataFrame of historical returns and uses those observations to estimate:

* Expected return for each asset
* Covariance between assets
* Portfolio expected return
* Portfolio variance
* Portfolio volatility
* Sharpe ratio
* Historical Value at Risk (VaR)
* Historical Conditional Value at Risk (CVaR)

It can then optimise the portfolio according to different objectives, including:

1. Minimum variance for a given target return
2. Maximum expected return subject to a maximum variance
3. Maximum Sharpe ratio
4. VaR and CVaR analysis of the resulting portfolio

The class also supports:

* A risk-free asset
* Long-only portfolios
* Short selling
* Leverage constraints in some optimisation routines

## 2. Pre Portfolio Creation

## 2.1. Required libraries

The class relies on a number of other packages to perform optimisations and general calculations.
```python
import numpy as np
import numpy.typing as npt
import pandas as pd
from numbers import Real
from warnings import warn
from scipy.optimize import (
    minimize,
    LinearConstraint,
    NonlinearConstraint
)
```

### 2.2. Preparing the historical returns

The class expects a pandas.DataFrame, where each column shows the returns of a different asset (percentage change from the previous periods value) and each row is an observation period. For example:
```python
returns = pd.DataFrame({
    "Asset A": [0.01, -0.02, 0.015, 0.005, -0.01],
    "Asset B": [0.02, -0.01, 0.005, 0.01, -0.005],
    "Asset C": [-0.01, 0.015, 0.02, -0.005, 0.01]
})
```
## 3. The Portfolio

### 3.1. Creating a portfolio

The simplest portfolio is created as follows:

portfolio = MeanVariancePortfolio(returns)

By default this creates a portfolio with:
* No risk-free asset
* Short selling allowed

The class estimates expected returns and the covariance matrix directly from the historical observations. The expected return and covariances are calculated using the pandas.DataFrame.mean and pandas.DataFrame.cov functions respectively. Hence, the expected returns are the historical mean returns and covariance matric is the historical sample covariance matrix. When first defined the portfolio weights are set to equal between assets i.e. each asset has a weight of `1/n` where `n` is the number of assets in the portfolio, including risk-free investment if provided. 

### 3.2. Inspecting the portfolio

Several properties are available directly but should not be altered once the portfolio is initial defined. 

* Number of assets: portfolio.numAssets
* Expected asset returns: portfolio.expectedAssetReturn
* Historical returns: portfolio.historicReturns
* Covariance matrix: portfolio.assetCovariance

Statistics which describe the portfolio can also be retrieved directly. Until an optimisation scheme has been enacted the portfolio is assumed to be uniform across assets, with no short positions. After optimisation these statistics are updated. 

* Portfolio weights: portfolio.portfolioWeights
* portfolio.expectedPortfolioReturn
* portfolio.portfolioVariance
* portfolio.portfolioVolatility

### 3.3. Printing the portfolio

The easiest way to see the current portfolio is to use the `MeanVariancePortfolio.printPortfolio()` method. Which prints table containing information for each asset and the portfolio as a whole. For example:

|               |  Asset A  |   Asset B |   Asset C |
|---------------|-----------|-----------|-----------|
|Expected return | 0.008000 |  0.012000 |  0.006000 |
|Volatility |      0.050000 |  0.070000 |  0.040000 |
|Portfolio share |  0.333333|  0.333333 |  0.333333 |

|---|
|Overall expected return: | 0.0087 |---|---|
|Overall volatility: |     0.0382 |---|---|
|Overall variance: |      0.001462 |---|---|
|---|

### 3.4. Portfolio weights

The weights can be read using `MeanVariancePortfolio.portfolioWeights` member and can be changed mannualy, which will also update the `MeanVariancePortfolio.expectedReturn`, `MeanVariancePortfolio.portfolioVariance`, and `MeanVariancePortfolio.portfolioVolatility`. If the weights are set manually, they should be passed as an array-like object with a single dimension of length equal to the numnber of assets within the portfolio, an error will be raised if this is not the case. In addition, if the weights do not sum to one, a requirement of the portfolio weights, then they are normalised to achieve this while maintaining their relative positions between assets. However, it is preferable to find new weights through one of the optimisation methods. 

### 3.5. Allowing or preventing short selling

Short positions are indicated negative weights within the portfolio. By default, the portfolio allows short positions however this can be changed when creating the object. To allow for a long-only portfolio use the `allowShorts` argument in the object definition.
```python
portfolio = MeanVariancePortfolio(
    returns,
    allowShorts=False
)
```
This means each weight is constrained to be between 0.0 and 1.0, and sum to 1.0. Note that is short selling is allowed, weights are not constrained to be less than 1.0, but it is still required that the weights sum to 1.0.

## 4. Portfolio Optimisation

### 4.1. Minimum-variance portfolio

One of the main purposes of the class is to find the portfolio with the lowest variance for a specified expected return. This is done using the `MeanVariancePortfolio.minimiseVariance()` member function. For example:
```python
result = portfolio.minimiseVariance(
    target_return=0.01
)
```
This finds the lowest-variance portfolio with an expected return equal to or greater than 1%. The resulting weights are automatically updated and stored in the `MeanVariancePortfolio.portfolioWeights` class member.

#### 4.1.1. Exact target return

By default, `minimiseVariance()` finds the minimum-variance portfolio such that the expected return is greater than or equal to the target. If, instead, an exact expected return is desired, this can be enforces using the `equal_to` function argument. Which imposes the resulting portfolio must have an expected return equal to `target_return`. When short selling is allowed and no risk-free asset is present, the class finds an analytical solution to this problem rather than using numerical optimisation.

#### 4.1.1. Efficient frontier

The efficient frontier can be found by repeated calls of the `minimiseVariance` function for a range of target returns. This produces the familiar efficient-frontier relationship between expected return and risk. For example:

```python
target_returns = np.linspace(
    portfolio.expectedAssetReturn.min(),
    portfolio.expectedAssetReturn.max(),
    50
)
frontier_returns = []
frontier_volatility = []
for target in target_returns:
    portfolio.minimiseVariance(
        target_return=target,
        equal_to=True
    )
    frontier_returns.append(
        portfolio.expectedPortfolioReturn
    )
    frontier_volatility.append(
        portfolio.portfolioVolatility
    )

import matplotlib.pyplot as plt
plt.plot(
    frontier_volatility,
    frontier_returns
)
plt.xlabel("Portfolio volatility")
plt.ylabel("Expected return")
plt.title("Mean-Variance Efficient Frontier")
plt.show()
```

### 4.2. Maximum expected return

The `maximiseExpectedReturn()` finds an "optimal" portfolio by maximising the portfolios expected return, under the constraint that the portfolio variance cannot exceed a specified value, by the `maximum_variance` argument. Again, the portfolio statistics are updated automatically.
```python
portfolio.maximiseExpectedReturn(
    maximum_variance=0.002
)
```
The function also allows the inclusion of a risk adjustment parameter. A larger `risk_adjustment` value places greater emphasis on reducing variance relative to increasing expected return, by default this value is set to 0.0 and there is no risk aversion.
```python
portfolio.maximiseExpectedReturn(
    maximum_variance=0.002,
    risk_adjustment=10
)
```
### 4.3. Maximum Sharpe-ratio portfolio

The `maximiseSharpeReturn()` method attempts to find the portfolio with the highest Sharpe ratio. The function takes two arguments, one which allows the portfolio to be calculate for a risk-free investment rate which is different from the one defined during object creation, `risk_free_rate`, if no value is given the internal risk-free rate is used. The second argument, `risk_free_proportion`, defines the proportion of the porfolio which is invested in the risk-free asset, by default this is 0.0. We discuss the risk-free investment further in the next section.

## 5. Including a Risk-Free Asset

A risk-free asset can be included when constructing the portfolio, by using the `risk_free_rate` argument. The arguments default value is `None` excludes the asset. A non-zero risk-free investment can be included as such
```python
portfolio = MeanVariancePortfolio(
    returns,
    risk_free_rate=0.002
)
```
This adds a risk-free asset with expected return of 0.2% and Te expected-return vector then becomes

[risk-free return, Asset A return, Asset B return, Asset C return].

The risk-free asset has zero variance and zero covariance with the risky assets. Therefore, if there are three risky assets, the portfolio has four assets. The first portfolio weight represents the risk-free allocation and, therefore, remaining weights correspond to the risky assets. When a risk-free asset is active, the class also provides `MeanVariancePortfolio.riskFreeProportion` and `MeanVariancePortfolio.riskProportion` members which directly retrieve the proportion of the portfolio invested in the risk-less and risky assets respectively. If no risk-free investment is included within the portfolio, these members will be 0.0 and 1.0 respectively.

### 5.1. Borrowing at the risk-free rate

The mean-variance optimisation member functions, `MeanVariancePortfolio.maximiseReturn()` and `MeanVariancePortfolio.minimiseVariance()`, provide options to be able to borrow at the risk-free rate or not by the function argument `allow_borrowing`, which takes an apropriate boolean value depending on the need. Its default value is `True`.

### 5.2. Choosing a risk-free investment

The optimal Sharpe portfolio is not unique when risk-free investment is available. For a risk-free investment $w_0$ any portfolio that can be written as $(1-w_0)\boldsymbol{x}_S$, where $\boldsymbol{x}_S$ is the optimal Sharpe portfolio for risky assets, is optimal. Therfor, choosing a risk-free investment amount is still required after finding the optimal Sharpe portfolio. This can be done by considering the volatility of the resulting portfolio. The function `MeanVariancePortfolio.setRiskFreeInvestment()` allows the user to find calculate a risk-free investment amount for a given amount of volatility, `target_volatility`. The function allows for borrowing at the risk-free rate as well as setting a borrow limit, through the `allow_borrowing` and `borrowing_limit` arguments. 


## 6. Target Excess Return

When a risk-free asset is active, minimiseVariance() can target an excess return rather than an absolute return. For example:
```python
portfolio.minimiseVariance(
    target_return=0.005,
    target_excess_return=True
)
```
If `risk-free rate = 0.002` then the target is an excess return of 0.5%, corresponding to an expected return of `0.002 + 0.005 = 0.007` or 0.7%. This is useful when thinking about portfolio returns relative to the risk-free investment.

## 7. VaR and CVaR

### 7.1. Value at Risk (VaR)

The class also provides historical Value at Risk through the `MeanVariancePortfolio.valueAtRisk()` function. The default confidence level is 95%, although this can be specified (using a decimal value). The method calculates the historical distribution of portfolio returns and takes the 5th percentile. Conceptually this calculates the 5-th percentile of the historic returns i.e. 5% of the historic returns are below its value.

The VaR can also be calculated per asset, rather than as an aggregate, using the `by_asset` argument of the function. The function the returns two objects: the portfolio VaR and the per asset VaR. The per asset VaR is returned as an array. Importantly, the portfolio VaR is calculated from the historical portfolio returns, so it incorporates the historical co-movement between assets.

### 7.2. Conditional Value at Risk (CVaR)

CVaR measures the expected loss in the worst observations beyond the VaR threshold i.e. conditional on the return being less than the 5-th percentile. The function `MeanVariancePortfolio.conditionalValueAtRisk()` calculates this value for the historic data provided to the portfolio.

## 8. Complete Example

The following demonstrates a typical workflow.

```python
import pandas as pd
import numpy as np
# Load historical returns
returns = pd.read_csv("historical_returns.csv")
# Create portfolio
portfolio = MeanVariancePortfolio(
    returns,
    include_risk_free=True,
    _riskFreeRate=0.002,
    _allowShorts=False
)
# Inspect the initial portfolio
portfolio.printPortfolio()
# Find minimum-variance portfolio
# targeting a 0.8% expected return
portfolio.minimiseVariance(
    target_return=0.008,
    equal_to=True
)
# Display the optimised portfolio
portfolio.printPortfolio()
# Calculate Sharpe ratio
sharpe = portfolio.calculateSharpeRatio()
print(f"Sharpe ratio: {sharpe:.4f}")
# Calculate historical VaR
var = portfolio.valueAtRisk(
    qnt=0.95
)
# Calculate historical CVaR
cvar = portfolio.conditionalValueAtRisk(
    qnt=0.95
)
print(f"VaR:  {var:.4%}")
print(f"CVaR: {cvar:.4%}")
```

## 9. Interface Summary

| Method/property |	Purpose |
| --- | --- |
| numAssets	| Number of assets |
| historicReturns | Historical return DataFrame |
| expectedAssetReturn | Estimated expected return of each asset |
| assetCovariance |	Estimated covariance matrix |
| portfolioWeights | Current portfolio weights |
| expectedPortfolioReturn | Expected portfolio return |
| portfolioVariance | Portfolio variance |
| portfolioVolatility | Portfolio volatility |
| riskFreeRate | Risk-free rate |
| riskFreeActive | Whether risk-free asset is included |
| riskProportion | Proportion invested in risky assets |
| riskFreeProportion | Proportion invested in risk-free asset |
| calculateSharpeRatio() | Calculate portfolio Sharpe ratio |
| minimiseVariance() | Find minimum-variance portfolio for a target return |
| maximiseExpectedReturn() | Maximise return subject to variance constraint |
| maximiseSharpeReturn() | Maximise Sharpe ratio |
| valueAtRisk() | Calculate historical VaR |
| conditionalValueAtRisk() | Calculate historical CVaR |
| printPortfolio() | Display portfolio statistics |
