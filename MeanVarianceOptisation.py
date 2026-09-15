import numpy as np
import numpy.typing as npt
import pandas as pd
from warnings import warn
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from numbers import Real

class MeanVariancePortfolio:
    """Mean-variance portfolio built from historical asset returns."""
    def __init__(self, historic_returns: pd.DataFrame, risk_free_rate: float|None=None, 
                 include_risk_free: bool=False, allow_shorts: bool=True):
        if not isinstance(historic_returns, pd.DataFrame):
            raise TypeError("`historic_returns` must be a pandas DataFrame.")
        if historic_returns.empty:
            raise ValueError("`historic_returns` must contain at least one row of returns.")
        
        try:
            returns = historic_returns.apply(pd.to_numeric, errors="raise").astype(float)
        except Exception as exc:
            raise TypeError("`historic_returns` must contain only numeric return columns.") from exc
        
        returns = returns.dropna(how="all")
        if returns.empty:
            raise ValueError("`historic_returns` has no usable return observations.")
        if returns.shape[1] == 0:
            raise ValueError("`historic_returns` must contain at least one asset column.")
        
        self._riskFreeActive = (include_risk_free)
        if risk_free_rate != 0.0 and (not self.riskFreeActive):
            warn("`risk_free_rate` is non-zero but `include_risk_free` is False. A risk-free investment is not included.")
            risk_free_rate = 0.0

        returns_expected = returns.mean().to_numpy(dtype=float)
        returns_covariance = returns.cov()
        if returns_covariance.isnull().to_numpy().any():
            raise ValueError("Unable to calculate covariance; check for missing or insufficient return data.")

        returns_covariance = returns_covariance.to_numpy(dtype=float)
        if returns_covariance.shape != (returns.shape[1], returns.shape[1]):
            raise ValueError("Calculated covariance matrix shape does not match the number of assets.")
        eigs = np.linalg.eigvalsh(returns_covariance)
        if eigs.min() < -1e-8 * eigs.max():
            raise ValueError("Calculated covariance matrix has negative eigenvalues (not PSD).")

        self._numAssets = returns.shape[1] + 1 * (self.riskFreeActive)
        self._historicReturns = returns
        self._assetCovariance = returns_covariance
        self._riskFreeRate = float(risk_free_rate) if (risk_free_rate is not None) else 0.0
        self._allowShorts = bool(allow_shorts)

        self._expectedAssetReturn = returns_expected
        if self.riskFreeActive:
            self._expectedAssetReturn = np.insert(returns_expected, 0, self.riskFreeRate)

        self.riskFreeProportion = 1.0 / self.numAssets 
        self.riskProportion = 1.0 - self.riskFreeProportion
        self.portfolioWeights = np.full(self.numAssets, 1.0 / self.numAssets)

        self._updateExpectedPortfolioReturns()

    # --- GETTERS --- #
    @property 
    def numAssets(self) -> int:
        return self._numAssets
    @property
    def expectedAssetReturn(self) -> npt.NDArray:
        return self._expectedAssetReturn
    @property
    def historicReturns(self) -> pd.DataFrame:
        return self._historicReturns
    @property
    def riskFreeRate(self) -> float:
        return self._riskFreeRate
    @property
    def riskFreeActive(self) -> bool:
        return self._riskFreeActive
    @property
    def allowShorts(self) -> bool:
        return self._allowShorts
    @property
    def assetCovariance(self) -> np.ndarray:
        return self._assetCovariance

    # --- non-constant memebers
    # members that aren't const
    def _weightArrayChecker(self, array: npt.ArrayLike) -> npt.NDArray:
        arr = np.asarray(array, dtype=float)
        if (arr.ndim != 1) or (len(arr) != self.numAssets):
            raise TypeError("Incorrect dimenions")
        return arr
    
    @property 
    def expectedPortfolioReturn(self) -> float:
        return self._expectedPortfolioReturn
    @expectedPortfolioReturn.setter
    def expectedPortfolioReturn(self, value: float) -> None:
        if not isinstance(value, Real):
            raise TypeError("Excpected portfolio return must be a real number.")
        self._expectedPortfolioReturn = float(value)

    @property
    def portfolioWeights(self) -> np.ndarray:
        return self._portfolioWeights
    @portfolioWeights.setter
    def portfolioWeights(self, new_weights: npt.ArrayLike):
        try:
            new_weights = self._weightArrayChecker(new_weights)
        except Exception as exc:
            raise TypeError("Portfolio weights are the incorrect dimension.") from exc
        
        self._portfolioWeights = new_weights

    @property 
    def portfolioVariance(self) -> float:
        return self._portfolioVariance
    @portfolioVariance.setter
    def portfolioVariance(self, var: float) -> None:
        if not isinstance(var, Real) or (var<=0.0):
            raise TypeError("Portfolio variance must be a positive real number.")
        
        self._portfolioVariance = var

    @property 
    def portfolioVolatility(self) -> float:
        return self._portfolioVolatility
    
    @portfolioVolatility.setter
    def portfolioVolatility(self, vol: float) -> None:
        if not isinstance(vol, Real) or (vol<=0.0):
            raise TypeError("Portfolio volatility must be a positive real number.")
        
        self._portfolioVolatility = vol

    @property
    def riskProportion(self) -> float:
        return self._riskProportion
    @riskProportion.setter
    def riskProportion(self, prop: float) -> None:
        if (not isinstance(prop, Real)) or (prop>1.0 or prop<0.0):
            raise TypeError("Risk proportion must be a real number between 0.0 and 1.0.")

        self._riskProportion = prop

    @property
    def riskFreeProportion(self) -> float|None:
        if not self.riskFreeActive:
            warn("Risk-free investment is not allowed.")
        return self._riskFreeProportion
    @riskFreeProportion.setter
    def riskFreeProportion(self, prop: float|None) -> None:
        if (not isinstance(prop, Real)) or (prop>1.0 or prop<0.0):
            raise TypeError("Risk-free proportion must be a real number between 0.0 and 1.0.")

        self._riskFreeProportion = prop

    #   
    # -- INTERNAL FUNCTIONS ----------------------------------------------------------------------------
    #

    def _normalizeWeights(self, tol: float=1e-10) -> None:
        weights = np.asarray(self.portfolioWeights, dtype=float)
        weights_sum = weights.sum()
        if abs(weights_sum - 1.0) > tol and weights_sum != 0.0:
            weights = weights / weights_sum
        if not self.allowShorts:
            weights = np.clip(weights, 0.0, 1.0)
            weights_sum = weights.sum()
            if weights_sum <= tol:
                raise ValueError("Weights collapsed to zero after clipping; check constraints.")
            weights = weights / weights_sum
        self.portfolioWeights = weights

    def _checkLeverageLimitArguments(self, leverage_limit: float|tuple[float, float]|None=None):
            # --- leverage_limit checks
            # Check the passed object is of the correct form i.e. Real, (Real, Real), or None
            if not (isinstance(leverage_limit, Real) or isinstance(leverage_limit, tuple) or (leverage_limit is None)):
                raise TypeError("Leverage limit must be None or, alternatively, a real number or a tuple of two real numbers, representing symmetric and asymmetric bounds respectively.")
            # Check if short positions are allowed.
            if (leverage_limit is not None) and (not self.allowShorts):
                warn("Leverage bound was given but short positions are not allowed within this portfolio. Leverage is ignored.")
                return (None, None), False
            
            # If leverage limit is a float, check that it is positive
            if isinstance(leverage_limit, Real):
                leverage_value = float(leverage_limit)
                if leverage_value <= 0.0:
                    raise ValueError("`leverage_limit` must be a positive real number or a length-two tuple of real numbers.")
            # If leverage_limit is a tuple, check that its values have relatively appropriate values
            if isinstance(leverage_limit, tuple):
                if len(leverage_limit)!=2:
                    raise TypeError("`leverage_limit` must be a positive real number or a length-two tuple of real numbers.")
                if not (isinstance(leverage_limit[0], Real) and isinstance(leverage_limit[1], Real)):
                    raise TypeError("`leverage_limit` must be a positive real number or a length-two tuple of real numbers.")
                if leverage_limit[0]>leverage_limit[1]:
                    raise TypeError("Lower leverage limit is larger than upper leverage limit.")
            # Define a tuple of limits
            leverage_limit_tuple = (-1*leverage_limit, leverage_limit) if isinstance(leverage_limit, Real) else leverage_limit

            return leverage_limit_tuple, True
    
    def _getPortfolioBounds(self, leverage_limit: float|tuple[float, float]|None=None, 
                            allow_borrowing: bool=True, borrowing_limit: float|None=None):
        
        if (borrowing_limit is not None) and  (not isinstance(borrowing_limit, Real)):
            raise TypeError("Borrowing limit must be a positive real number or `None`.")
        if (borrowing_limit is not None) and (borrowing_limit<=0.0):
            raise TypeError("Borrowing limit must be a positive real number or `None`.")
        if (not allow_borrowing) and (borrowing_limit is not None):
            warn("Borrowing is not allowed, `allow_borroing=False`, but borrowing limit has been set. Borrowing limit is ignored.")
        
        n_risky = self.numAssets - (1 if self.riskFreeActive else 0)
        leverage_limit_tuple, leverage_limit_active = self._checkLeverageLimitArguments(leverage_limit)
        
        if not self.allowShorts:
            risky_bounds = [(0., 1.) for _ in range(n_risky)]
        else:
            risky_bounds = ([leverage_limit_tuple for _ in range(n_risky)] 
                           if leverage_limit_active 
                           else [(None, None) for _ in range(n_risky)])

        risk_free_bound = None
        if self.riskFreeActive:
            risk_free_bound = (None, 1.0)
            if not allow_borrowing:
                risk_free_bound = (0., 1.)
            elif borrowing_limit is not None:
                risk_free_bound = (-borrowing_limit, 1.)

        bounds = ([risk_free_bound] + risky_bounds 
                  if self.riskFreeActive 
                  else risky_bounds)

        return bounds
    
    # --- OPTIMISATION FUNCTIONS --- #
    def _varianceFunction(self, x, risk_free_active: bool|None=None) -> float:
        if risk_free_active is None:
            risk_free_active = self.riskFreeActive

        x = np.asarray(x, dtype=float)
        weights = x[1:] if risk_free_active else x
        return float(weights @ self.assetCovariance @ weights)

    def _varianceGradient(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if self.riskFreeActive:
            grad = np.zeros_like(x)
            grad[1:] = 2.0 * self.assetCovariance @ x[1:]
            return grad

        return 2. * self.assetCovariance @ x

    def _volatilityFunction(self, x, risk_free_active: bool|None=None) -> float:
        return np.sqrt(self._varianceFunction(x, risk_free_active))

    def _expectedReturn(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.expectedAssetReturn)

    def _expectedExcessReturns(self, x) -> float:
        if not self.riskFreeActive:
            warn("risk-free investment is not a part of this portfolio. Excess return is returned as expected return.")
            return self._expectedReturn(x)
        return float(x[1:] @ (self.expectedAssetReturn[1:] - self.riskFreeRate))

    def _negRiskAdjustedReturn(self, x, risk_adjustment: float, risk_free_active: bool|None=None) -> float:
        return float(risk_adjustment * self._varianceFunction(x, risk_free_active) - self._expectedReturn(x))

    def _negSharpeRatio(self, x, risk_free_rate: float) -> float:
        """ 
        x: portfolio wieghts, (self.numAssets-1) i.e. has length equal to the number of risky assets
        risk_free_rate: float
        """
        if not self.riskFreeActive:
            raise TypeError("Risk free investment must be available to calculate the Sharpe portfolio.")
        if not isinstance(risk_free_rate, Real) or not (risk_free_rate is None):
            raise TypeError("Risk free rate must be a real number, if being used.")
        
        portfolio_return = x @ self.expectedAssetReturn[1:] 
        volatility = self._volatilityFunction(x, False)
        if volatility <= 1e-16:
            return np.inf

        return float(risk_free_rate - portfolio_return) / volatility

    def _calcVaR(self, returns: np.ndarray, qnt: float=0.95) -> float:
        return -float( np.quantile(returns, q=(1.-qnt)) )

    def _calcCVaR(self, returns: np.ndarray, qnt: float=0.95) -> float:
        var_threshold = np.quantile(returns, q=(1.-qnt))
        tail_returns = returns[returns <= var_threshold]
        return -float( np.mean(tail_returns) )

    # --- UPDATE INTERNALS --- #
    def _updateExpectedPortfolioReturns(self) -> None:

        self.expectedPortfolioReturn = float( self.expectedAssetReturn @ self.portfolioWeights )
        self.portfolioVariance = self._varianceFunction(self.portfolioWeights)
        self.portfolioVolatility = np.sqrt(self.portfolioVariance)

        if self.riskFreeActive:
            self.riskFreeProportion = self.portfolioWeights[0]
            self.riskProportion = 1.0 - self.riskFreeProportion

    # -------------------------- # 
    # --- EXTERNAL FUNCTIONS --- #
    # -------------------------- #
    # --- CALCULATORS --- # 
    def calculateSharpeRatio(self, risk_free_rate: float|None=None) -> float:
        if (risk_free_rate is not None) and (not isinstance(risk_free_rate, Real)):
            raise TypeError("Risk free rate must be a real number or None.")
        if risk_free_rate is None:
            risk_free_rate = self.riskFreeRate

        risky_returns = self.expectedAssetReturn[1:] if self.riskFreeActive else self.expectedAssetReturn
        risky_weights = self.portfolioWeights[1:] if self.riskFreeActive else self.portfolioWeights

        if self.portfolioVolatility <= 0.0:
            excess = self.riskFreeRate - risky_weights @ risky_returns
            return -np.inf if excess <= 0 else np.inf
        
        expected_risky_return = risky_weights @ risky_returns
        return float((expected_risky_return - risk_free_rate) / self.portfolioVolatility)

    def printPortfolio(self) -> None:
        names = self.historicReturns.columns.to_numpy(dtype=str)
        returns = self.expectedAssetReturn
        volatilities = np.diag(self.assetCovariance)
        weights = self.portfolioWeights

        if self._riskFreeActive:
            names = np.insert(names, 0, "risk-free")
            volatilities = np.insert(volatilities, 0, 0.0)

        label_width = max(
            len("Expected return"),
            len("Volatility"),
            len("Portfolio share"),
            max(len(name) for name in names),
        )
        col_width = 14

        header = f"{'':<{label_width}}"
        for name in names:
            header += f" {name:>{col_width}}"

        row_returns = f"{'Expected return':<{label_width}}"
        row_vols = f"{'Volatility':<{label_width}}"
        row_weights = f"{'Portfolio share':<{label_width}}"

        for r, v, w in zip(returns, volatilities, weights):
            row_returns += f" {r:>{col_width}.6f}"
            row_vols += f" {v:>{col_width}.6f}"
            row_weights += f" {w:>{col_width}.6f}"

        print(header)
        print(row_returns)
        print(row_vols)
        print(row_weights)
        print("-" * (label_width + (col_width + 1) * self.numAssets))
        print(f"Overall expected return: {self.expectedPortfolioReturn:.4f}")
        print(f"Overall volatility:      {self.portfolioVolatility:.4f}")
        print(f"Overall variance:      {self.portfolioVariance:.6f}")

    # --- OPTIMISE PORTFOLIO --- # 
    def minimiseVariance(self, target_return: float, equal_return: bool=False, target_excess_return: bool=False,
                         risk_free_rate: float|None=None, allow_borrowing: bool=True, borrowing_limit: float|None=None, 
                         leverage_limit: float|tuple[float, float]|None=None, scale_variance: None|float=1.e5, **kwargs):
        """ 
        Finds the portfolio which minimises portfolio variance. 

        target_return (float): The target for the expected portfolio return. 
        equal_return (bool): If True the portfolio is found with an expected return equal to `target_return`. 
            Default False.
        target_excess_return (bool): If True the expected excess return is used to rather than expected return 
            in the optimisation. Default is False.
        risk_free_rate (float): The risk-free rate that is available to borrow. If no value is passed but risk
            free investments are available within the portfolio, the portfolios risk-free rate is used. 
        allow_borrowing (bool): A indicator to determine whether or you are allowed to bororw at the risk-
            free rate. The default value is True.
        borrowing_limit (float): If borrowing at the risk-free rate is allowed, this imposes an upper borrowing limit.
            The default value is None, imposes no limit. If a limit is given this should be a positive real number. 
        leverage_limit (float)|(tuple): The maximum/minimum amount of leverage allowed for each asset, as a 
            percentage of portfolio value, where positive/negative values indicate long/short positions 
            respectively. Therefore a limit of the form `(-a,b)`, for positive real numbers a and b, would enforce
            a maximum short position of `a` and maximum long position of `b` for each asset. If short positions are 
            not allowed within the portfolio the argument is ignored. If a scalar real number is passed a 
            symmetric limit is used, of the form `(-a, a)`, if a length-two tuple is passed, of the form `(a,b)`, then 
            the first element is taken to be the lower bound, and the second the upper. Note if the two-element tuple
            is passed is not checked that the lower bound is negative.
        scale_variance (float): A real number to scale the variance during the optimisation process. Generally large
            numbers are best for optimisation. Default 1.0e5.
        """
        # --- check variance_scaler is a non-negative real number 
        if not ((isinstance(scale_variance, Real) and scale_variance>0.0) or (scale_variance is None)):
            raise TypeError("Variance scaler must be a positive real number.")
        # Set equal to 1.0 if no scaler is not being used
        variance_scaler = 1.0 if (scale_variance is None) else float(scale_variance)

        # --- target_excess_return
        if target_excess_return:
            # If targetting exces returns set the risk-less rate to the `risk_free_rate` if passed, if not set to the
            # portfolios risk-free rate. If the portfolio has no risk-less rate, use the expected returns not excess. 
            if (risk_free_rate is None):
                if (self.riskFreeActive):
                    risk_free_rate = self.riskFreeRate
                else:
                    risk_free_rate = 0.0
                    warn("Excess return is targeted by the optimisation but no risk-free interest is present in the portfolio and no rate was provided to the optimisation function. The optimisation will continue with the expected return, not excess return.")
            returns = self.expectedAssetReturn.copy() - risk_free_rate
            if self.riskFreeActive and (risk_free_rate != self.riskFreeRate):
                returns[0] = 0.0
        else:
            # If not targeting the excess returns use expected returns.
            returns = self.expectedAssetReturn

        if equal_return and self.allowShorts and (not self.riskFreeActive) and (leverage_limit is None) and (not target_excess_return) and (borrowing_limit is None):
            print("Portfolio.minimiseVariance: Solving linear system of equations.")

            matrix_A = np.zeros((self.numAssets + 2, self.numAssets + 2))
            matrix_A[:self.numAssets, :self.numAssets] = 2.0 * self.assetCovariance
            matrix_A[self.numAssets + 1, :self.numAssets] = 1.0
            matrix_A[:self.numAssets, self.numAssets + 1] = -1.0
            matrix_A[:self.numAssets, self.numAssets] = -returns
            matrix_A[self.numAssets, :self.numAssets] = returns
            vector_b = np.zeros(self.numAssets + 2, dtype=float)
            vector_b[self.numAssets] = target_return
            vector_b[self.numAssets + 1] = 1.0
            try:
                vector_x = np.linalg.solve(matrix_A, vector_b)
            except np.linalg.LinAlgError:
                warn("Analytic solution failed. Optimised solution implemented."
                     "Numerical optimisation will be used.")
                
                return self.minimiseVariance(target_return=target_return, equal_return=False, target_excess_return=target_excess_return,
                                             risk_free_rate=risk_free_rate, allow_borrowing=allow_borrowing, leverage_limit=leverage_limit, 
                                             variance_scaler=variance_scaler, **kwargs)
            weights = np.asarray(vector_x[:self.numAssets], dtype=float)
            self.portfolioWeights = weights
            return weights
        
        else:
            return_constraint = (LinearConstraint(returns, target_return, target_return) 
                                 if equal_return 
                                 else LinearConstraint(returns, target_return, np.inf))
            portfolio_sum = LinearConstraint(np.ones(self.numAssets), 1.0, 1.0)

            portfolio_bounds = self._getPortfolioBounds(leverage_limit, allow_borrowing, borrowing_limit)

            def objectiveFunction(x, scale: None|float=1.0):
                if scale is None:
                    scale = 1.0
                return scale * self._varianceFunction(x)

            def gradientFunction(x, scale: None|float=1.0):
                if scale is None:
                    scale = 1.0
                return scale * self._varianceGradient(x)
            
            result = minimize(objectiveFunction, 
                              x0=self.portfolioWeights, 
                              jac = gradientFunction,
                              args=(scale_variance,),
                              constraints=[return_constraint, portfolio_sum], 
                              bounds=portfolio_bounds,
                              method="SLSQP",
                              **kwargs)
            
            if result.success:
                self._portfolioWeights = np.asarray(result.x, dtype=float)
            else:
                warn("minimization failed. Portfolio not updated.")
            return result

    def maximiseExpectedReturn(self, maximum_variance: float, risk_adjustment: float|None=None,
                               allow_borrowing: bool=True, borrowing_limit: float|None=None, 
                               leverage_limit: float|tuple[float, float]|None=None, **kwargs):
        """ 
        Finds the portfolio which maximise expected returns. 

        maximum_variance (float): the maximum variance allowed. 
        risk_adjustment (float): the risk adjustment parameter.
        allow_borrowing (bool): A indicator to determine whether or you are allowed to bororw at the risk-
            free rate. The default value is True.
        borrowing_limit (float): If borrowing at the risk-free rate is allowed, this imposes an upper borrowing limit.
            The default value is None, imposes no limit. If a limit is given this should be a positive real number. 
        leverage_limit (float)|(tuple): The maximum/minimum amount of leverage allowed for each asset, as a 
            percentage of portfolio value, where positive/negative values indicate long/short positions 
            respectively. Therefore a limit of the form `(-a,b)`, for positive real numbers a and b, would enforce
            a maximum short position of `a` and maximum long position of `b` for each asset. If short positions are 
            not allowed within the portfolio the argument is ignored. If a scalar real number is passed a 
            symmetric limit is used, of the form `(-a, a)`, if a length-two tuple is passed, of the form `(a,b)`, then 
            the first element is taken to be the lower bound, and the second the upper. Note if the two-element tuple
            is passed is not checked that the lower bound is negative.
        **kwargs: additional parameters to be passed to scipy.minimize
        """
        if risk_adjustment is None:
            risk_adjustment = 0.0
        
        variance_maximum = NonlinearConstraint(self._varianceFunction, -np.inf, maximum_variance)
        portfolio_sum = LinearConstraint(np.ones(self.numAssets), 1.0, 1.0)

        portfolio_bounds = self._getPortfolioBounds(leverage_limit, allow_borrowing, borrowing_limit)

        result = minimize(self._negRiskAdjustedReturn, 
                          x0=self.portfolioWeights, 
                          args=(risk_adjustment,), 
                          constraints=[variance_maximum, portfolio_sum], 
                          bounds=portfolio_bounds, 
                          method="SLSQP", 
                          **kwargs)
        
        if result.success:
            self.portfolioWeights = np.asarray(result.x, dtype=float)
        else:
            warn("minimization failed. Portfolio remains unchanged.")
        return result

    def maximiseSharpeReturn(self, risk_free_rate: float|None=None, **kwargs) -> None:
        """ 
        Finds the portfolio which maximises the Sharpe ratio. The function does not assign any weight to the 
        risk-free investment. This can be calculated for a desired amount of volatility using the 
        `setRiskyInvestment` member function.

        risk_free_rate (float): The risk-free return on investment. If no value is passed the portfolio value
            is used.
        **kwargs: Aadditional parameters to be passed to scipy.minimize
        """
        if (not self.riskFreeActive) and (risk_free_rate is None):
            raise RuntimeError("Risk free investment must be available to calculate the Sharpe portfolio.")
        if (risk_free_rate is not None) and (not isinstance(risk_free_rate, Real)):
            raise ValueError("Risk free rate must be a real number of None.")
        
        if (risk_free_rate is None):
            print("Risk free rate is not given. Portfolio risk-free rate is used.")
            risk_free_rate = self.riskFreeRate

        n_risky_assets = self.numAssets - 1
        portfolio_sum = LinearConstraint(np.ones(n_risky_assets), 1.0, 1.0)
        portfolio_bounds = ([(0.0, 1.0)] * n_risky_assets 
                            if not self.allowShorts 
                            else [(None, None)] * n_risky_assets)

        result = minimize(self._negSharpeRatio, 
                          x0=np.full(n_risky_assets, 1 / n_risky_assets), 
                          args=(risk_free_rate, ),
                          constraints=[portfolio_sum], 
                          bounds=portfolio_bounds, 
                          method="SLSQP", 
                          **kwargs)

        if result.success:
            self._portfolioWeights = np.insert(np.asarray(result.x, dtype=float), 
                                                0, 
                                                0.0)
        else:
            warn("Minimisation failed. Portfolio remains unchanged.")
        return result

    def setRiskFreeInvestment(self, target_volatility: float, allow_borrowing: bool = True, 
                              borrowing_limit: float | None = None):
        """
        Combine the current risky portfolio with the risk-free asset to achieve a target 
        portfolio volatility. Primary use to for after using the Sharpe ratio maximisation 
        method, which does not set a risk-free investment. The function updates the all members
        of the class to be printed using the `printPortfolio` function.

        target_volatiliy (float): The desired amount of volatility in the portfolio.
        allow_borrowing (bool): A indicator to determine whether or you are allowed to bororw at the risk-
            free rate. The default value is True.
        borrowing_limit (float): If borrowing at the risk-free rate is allowed, this imposes an upper borrowing limit.
            The default value is None, imposes no limit. If a limit is given this should be a positive real number. 
        **kwargs: Aadditional parameters to be passed to scipy.minimize
        """

        if not isinstance(target_volatility, Real):
            raise TypeError("`target_volatility` must be a real number.")
        if target_volatility <= 0:
            raise ValueError("`target_volatility` must be positive.")
        if not self.riskFreeActive:
            raise RuntimeError("A risk-free asset must be active.")

        risky_proportion = target_volatility / self.portfolioVolatility

        risk_free_proportion = 1.0 - risky_proportion

        if not allow_borrowing and risk_free_proportion < 0.:
            raise ValueError("Target volatility requires borrowing, but borrowing is not allowed.")

        if (borrowing_limit is not None) and (risk_free_proportion < -borrowing_limit):
            raise ValueError("Target volatility requires borrowing beyond the specified borrowing limit.")

        # Current portfolio contains the risky portfolio
        # and possibly a risk-free position.
        risky_weights = (self.portfolioWeights[1:] if self.riskFreeActive else self.portfolioWeights)
        new_weights = np.insert(
            risky_proportion * risky_weights,
            0,
            risk_free_proportion
        )

        self._portfolioWeights = new_weights
        self._updateExpectedPortfolioReturns()

        return new_weights
    
    ## -- Value at Risk --
    def valueAtRisk(self, qnt: float=0.95, by_asset: bool=False):
        """ 
        Calculates and returns the value-at-risk for the portfolio.

        qnt (float): 
        """
        weights = self.portfolioWeights[1:] if self._riskFreeActive else self.portfolioWeights
        portfolio_returns = np.sum( self._historicReturns * weights, axis=1 )
        portfolio_var = self._calcVaR(portfolio_returns, qnt)
        print(f"Portfolio VaR ({qnt:.2%}): {portfolio_var:.6f}")

        if (not by_asset):
            return portfolio_var
        asset_var = np.array([self._calcVaR(self._historicReturns[asset_name].to_numpy(dtype=float), qnt) for asset_name in self._historicReturns] )
        print(f"Asset VaR ({qnt:.2%}):")
        print(pd.DataFrame([asset_var], columns=self._historicReturns.columns, index=["VaR"]).to_string())

        return portfolio_var, asset_var
    
    def conditionalValueAtRisk(self, qnt: float=0.95, by_asset: bool=False):
        weights = self.portfolioWeights[1:] if self._riskFreeActive else self.portfolioWeights
        portfolio_returns = np.sum( self.historicReturns * weights, axis=1 )
        portfolio_cvar = self._calcCVaR(portfolio_returns, qnt)
        print(f"Portfolio CVaR ({qnt:.2%}): {portfolio_cvar:.6f}")
        
        if (not by_asset):
            return portfolio_cvar
        
        asset_cvar = np.array( [self._calcCVaR(self._historicReturns[asset_name].to_numpy(dtype=float), qnt) for asset_name in self.historicReturns] )
        print(f"Asset CVaR ({qnt:.2%}):")
        print(pd.DataFrame([asset_cvar], columns=self.historicReturns.columns, index=["CVaR"]).to_string())
        
        return portfolio_cvar, asset_cvar