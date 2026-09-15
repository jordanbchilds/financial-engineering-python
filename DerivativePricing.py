import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve
from numbers import Real
from warnings import warn

# Black-Scholes model
class BlackScholes:
    def __init__(self, spot_price: float, strike_price: float, annual_risk_free_rate: float, 
               dividend_rate: float, annual_volatility: float, time_to_maturity: float, 
               is_call: bool=True):
        if annual_volatility<=0.0:
            raise TypeError("`annual_volatility` must be greater than zero.")
        if time_to_maturity<=0.0:
            raise TypeError("`time_to_maturity` must be greater than zero.")
        
        self._spotPrice = spot_price
        self._strikePrice = strike_price
        self._annualRiskFreeRate = annual_risk_free_rate
        self._dividendRate = dividend_rate
        self._annualVolatility = annual_volatility
        self._timeToMaturity = time_to_maturity
        self._isCall = is_call

        self._d1, self._d2 = self.get_d1d2()
        self._callPrice = self.spotPrice * np.exp(-self.dividendRate * self.timeToMaturity) * norm.cdf(self._d1) - self.strikePrice * np.exp(-self.annualRiskFreeRate * self.timeToMaturity) * norm.cdf(self._d2)
        self._putPrice = self._callPrice + self.strikePrice*np.exp(-self.annualRiskFreeRate*self.timeToMaturity) - self.spotPrice * np.exp(-self.dividendRate * self.timeToMaturity)
        self._fairPrice = self._callPrice if self.isCall else self._putPrice

    # --- GETTERS --- #
    @property
    def spotPrice(self) -> float:
        return self._spotPrice
    @property
    def strikePrice(self) -> float:
        return self._strikePrice
    @property
    def annualRiskFreeRate(self) -> float:
        return self._annualRiskFreeRate
    @property
    def annualVolatility(self) -> float:
        return self._annualVolatility
    @property
    def dividendRate(self) -> float:
        return self._dividendRate
    @property
    def timeToMaturity(self) -> float:
        return self._timeToMaturity
    @property
    def fairPrice(self) -> float:
        return self._fairPrice
    @property
    def isCall(self) -> bool:
        return self._isCall
    
    ## --- INTERNAL FUNCTIONS --- #
    def _inputConverter(self, value: float|npt.ArrayLike|None, self_value: float) -> npt.NDArray:
        if value is None:
            value = self_value
        value_arr = np.asarray(value, dtype=float)

        return value_arr

    def _check_array_shapes(self, initial_value: float|npt.ArrayLike|None=None, strike_price: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None, volatility: float|npt.ArrayLike|None=None)-> None:
        arrays = [np.asarray(x) for x in (initial_value, strike_price, time_to_maturity, volatility) if x is not None and not np.isscalar(x)]
        if arrays:
            shape = arrays[0].shape
            if not all(x.shape == shape for x in arrays):
                raise ValueError("Array-like arguments must have the same shape.")
    
    def _parameterChecker(self, initial_value: float|npt.ArrayLike|None=None, strike_price: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None, volatility: float|npt.ArrayLike|None=None):
        self._check_array_shapes(initial_value, strike_price, time_to_maturity, volatility)

        initial = self._inputConverter(initial_value, self._spotPrice)
        strike = self._inputConverter(strike_price, self._strikePrice)
        time = self._inputConverter(time_to_maturity, self._timeToMaturity)
        vol = self._inputConverter(volatility, self._annualVolatility)

        if np.any(vol<=0.0):
            raise ValueError("Volatility must be greater than zero.")
        if np.any(time<=0.0):
            raise ValueError("Time to maturity must be greater than zero.")

        return initial, strike, time, vol

    # --- Calculators --- #
    def calculateOptionPrice(self, 
                             initial_value: float|npt.ArrayLike|None=None, 
                             strike_price: float|npt.ArrayLike|None=None, 
                             time_to_maturity: float|npt.ArrayLike|None=None, 
                             volatility: float|npt.ArrayLike|None=None, 
                             number_of_options: int=1) -> float|npt.NDArray:
        
        if all(arg is None for arg in (initial_value, strike_price, time_to_maturity, volatility)) and self._fairPrice is not None:
            return float(number_of_options) * self.fairPrice
        
        initial, strike, time, vol = self._parameterChecker(initial_value, strike_price, time_to_maturity, volatility)
        d1, d2 = self.get_d1d2(initial, strike, time, vol)
        fair_price = (initial * np.exp(-self.dividendRate * time) * norm.cdf(d1) 
                      - strike * np.exp(-self._annualRiskFreeRate * time) * norm.cdf(d2))
        if not self._isCall:
            fair_price += strike*np.exp(-self._annualRiskFreeRate*time) - initial * np.exp(-self.dividendRate * time)

        return float(number_of_options) * fair_price
    
    def _calculate_d1(self, initial_value: float|npt.ArrayLike|None=None, strike_price: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None, volatility: float|npt.ArrayLike|None=None) -> float|npt.NDArray:
        initial, strike, time, vol = self._parameterChecker(initial_value=initial_value, strike_price=strike_price, time_to_maturity=time_to_maturity, volatility=volatility)

        d1 = (np.log(initial / strike) 
              + (self.annualRiskFreeRate 
                 - self.dividendRate 
                 + 0.5*vol**2) * time) 
        d1 /= (vol * np.sqrt(time))
        return d1 + 0.

    def _calculate_d2(self, d1: float|npt.ArrayLike|None=None, initial_value: float|npt.ArrayLike|None=None, strike_price: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None, volatility: float|npt.ArrayLike|None=None) -> float | npt.NDArray:
        initial, strike, time, vol = self._parameterChecker(initial_value=initial_value, strike_price=strike_price, time_to_maturity=time_to_maturity, volatility=volatility)
        if d1 is None:
            d1 = self._calculate_d1(initial, strike, time, vol)
        d1 = np.asarray(d1, dtype=float)

        return d1 - vol * np.sqrt(time)

    def get_d1d2(self, initial_value: float|npt.ArrayLike|None=None, strike_price: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None, volatility: float|npt.ArrayLike|None=None) -> tuple:
        d1 = self._calculate_d1(initial_value=initial_value, strike_price=strike_price, time_to_maturity=time_to_maturity, volatility=volatility)
        d2 = self._calculate_d2(d1, time_to_maturity=time_to_maturity, volatility=volatility)
        return d1, d2

    # --- THE GREEKS --- #
    def getDelta(self, initial_value: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None) -> float | npt.NDArray[np.float64]:
        initial, _, time, _ = self._parameterChecker(initial_value=initial_value, time_to_maturity=time_to_maturity)

        exp_dt = np.exp(-self.dividendRate * time)
        delta_call =  exp_dt * norm.cdf(self._calculate_d1(initial_value=initial, time_to_maturity=time))

        if self.isCall:
            return delta_call
        return delta_call - exp_dt

    def getGamma(self, initial_value: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None) -> float| npt.NDArray[np.float64]:
        initial, _, time, _ = self._parameterChecker(initial_value=initial_value, time_to_maturity=time_to_maturity)
        
        return (np.exp(-self.dividendRate * time) 
                * norm.pdf(self._calculate_d1(initial_value=initial, time_to_maturity=time)) 
                / (self.annualVolatility * initial * np.sqrt(time)) )
    
    def getVega(self, initial_value: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None) -> float | npt.NDArray[np.float64]:
        initial, _, time, _ = self._parameterChecker(initial_value=initial_value, time_to_maturity=time_to_maturity)

        return (np.exp(-self.dividendRate * time) 
                * initial 
                * np.sqrt(time) 
                * norm.pdf(self._calculate_d1(initial_value=initial, time_to_maturity=time)))

    def getTheta(self, initial_value: float|npt.ArrayLike|None=None, time_to_maturity: float|npt.ArrayLike|None=None) -> float | npt.NDArray[np.float64]:
        initial, strike, time, vol = self._parameterChecker(initial_value=initial_value, time_to_maturity=time_to_maturity)
                
        d1, d2 = self.get_d1d2(initial_value=initial, time_to_maturity=time)
        exp_dt = np.exp(-self.dividendRate * time)
        call_theta = (self.dividendRate * initial * exp_dt * norm.cdf(d1) 
                      - 0.5 * initial * vol * exp_dt * norm.pdf(d1) / np.sqrt(time)  
                      - self.annualRiskFreeRate * strike * np.exp(-self.annualRiskFreeRate * time) * norm.cdf(d2))
        if self.isCall:
            return call_theta

        return (call_theta 
                + self.annualRiskFreeRate * strike * np.exp(-self.annualRiskFreeRate * time) 
                - self.dividendRate * initial * np.exp(-self.dividendRate*time))

    # --- Implied volatility --- #
    def calculateImpliedVolatility(self, market_price: float, strike_price: float|None=None, time_to_maturity: float|None=None, output_all: bool=False, **kwargs):
        _, strike, time, vol = self._parameterChecker(strike_price=strike_price, time_to_maturity=time_to_maturity)
        
        vol0 = np.sqrt(2.0 * np.pi / time) * market_price / self.spotPrice

        def root_function(vol, K: float, T: float):
            d1, d2 = self.get_d1d2(volatility=vol, strike_price=K, time_to_maturity=T)
            price_difference = (self.calculateOptionPrice(strike_price=K, time_to_maturity=T, volatility=vol) - market_price)

            return price_difference

        res = fsolve(root_function, 
                     x0=vol0, 
                     args=(strike, time),
                     full_output=output_all, 
                     **kwargs)
        return res

    # --- Stock distribution and payoff --- # 
    def _expectedPayoff(self, payoff_function, density_df: pd.DataFrame, *args): 
        stock_values = density_df["stock_value"].to_numpy()
        density = density_df["density"].to_numpy()
        payoff = np.array([payoff_function(S) for S in stock_values])

        expected_payoff = np.trapezoid(payoff * density, stock_values)
        fair_price = expected_payoff * np.exp(-self._annualRiskFreeRate * self._timeToMaturity)
        
        return fair_price
    
    def estimateStockDistributionFromData(self, strike_price: npt.ArrayLike, market_price: npt.ArrayLike) -> pd.DataFrame:
                strike = np.asarray(strike_price, dtype=float)
                price = np.asarray(market_price, dtype=float)
                delta_array = np.diff(strike)
                delta = delta_array[0]
                if np.any(abs(delta_array-delta)>1.e-10):
                    raise TypeError("strike prices are not equi-distant.")
                
    
                density = (price[2:] - 2.0 * price[1:-1] + price[:-2]) / delta**2
                density *= np.exp(self._annualRiskFreeRate * self._timeToMaturity)
        
                return pd.DataFrame({"stock_value":strike[1:-1], "density": density})

    def payoffFunctionFairPriceFromData(self, payoff_function, strike_price: npt.ArrayLike, market_price: npt.ArrayLike, *args):
        density_df = self.estimateStockDistributionFromData(strike_price=strike_price, market_price=market_price)

        return self._expectedPayoff(payoff_function, density_df, *args)

    # --- from internal values --- # 

    def estimateStockDistribution(self, lower_bound: float, upper_bound: float, n: int|None=None, delta: float|None=None) -> pd.DataFrame:
        if n is None and delta is None:
            n = 100
        elif n is None and isinstance(delta, float):
            n = int( (upper_bound - lower_bound) / delta) + 1
        assert(isinstance(n, int))

        stock_values, delta = np.linspace(lower_bound, upper_bound, num=n, retstep=True)

        density = (self.calculateOptionPrice(strike_price=stock_values[2:]) 
                   - 2.0 * self.calculateOptionPrice(strike_price=stock_values[1:-1])
                   + self.calculateOptionPrice(strike_price=stock_values[:-2])) / delta**2
        
        density *= np.exp(self.annualRiskFreeRate * self.timeToMaturity)

        return pd.DataFrame({"stock_value":stock_values[1:-1], "density": density})

    def payoffFunctionFairPrice(self, payoff_function, lower_bound: float,  upper_bound: float, n: int|None=None, delta: float|None=None, *args):
        density_df = self.estimateStockDistribution(lower_bound, upper_bound, n, delta)
    
        return self._expectedPayoff(payoff_function, density_df, *args)

class BlackScholesPortfolio:
    def __init__(self, stock_paths: pd.DataFrame, period: float|None = None, annual_risk_free_rate: float=0.0):
        if period is None:
            warn("`period` is not provided. Taken as the difference in first and second index of the stock paths dataframe.")
            self.period = stock_paths.index[1] - stock_paths.index[0]
        else:
            self.period = period

        self._numPeriods = stock_paths.shape[0]
        self._periodsPerYear = 1 / self.period
        self._annualRiskFreeRate = annual_risk_free_rate
        self._stockPaths = stock_paths
        self._logReturnPaths = pd.DataFrame(
                                    self.stockPaths.apply(np.log).diff(),
                                    index=stock_paths.index,
                                    columns=stock_paths.columns
                                )

        self._periodToAnnualConversion = 1. / np.sqrt(self.period)
        self._periodRawVolatility = np.std(self.stockPaths, ddof=1, axis=0)
        self._annualRawVolatility = self.periodRawVolatility * self._periodToAnnualConversion
        self._periodLogReturnVolatility = np.std(self.logReturnPaths, ddof=1, axis=0)
        self._annualLogReturnVolatility = self.periodLogReturnVolatility * self._periodToAnnualConversion

    # --- GETTERS --- #
    @property
    def numPeriods(self) -> int:
        return self._numPeriods

    @property
    def periodsPerYear(self) -> float:
        return self._periodsPerYear

    @property
    def annualRiskFreeRate(self) -> float:
        return self._annualRiskFreeRate

    @property
    def stockPaths(self) -> pd.DataFrame:
        return self._stockPaths

    @property
    def logReturnPaths(self) -> pd.DataFrame:
        return self._logReturnPaths

    @property
    def periodRawVolatility(self):
        return self._periodRawVolatility

    @property
    def annualRawVolatility(self):
        return self._annualRawVolatility

    @property
    def periodLogReturnVolatility(self):
        return self._periodLogReturnVolatility

    @property
    def annualLogReturnVolatility(self):
        return self._annualLogReturnVolatility
    
    #
    # -- INTERNAL FUNCTIONS -------------------------------------
    #
    
    # -- BLACK-SCHOLES FUNCTIONS --

    def __createBlackScholesModels(self, strike_price: float, annual_volatility: float|None=None, time_to_maturity: float|None=None, dividend_rate: float=0.0, is_call: bool=True) -> None:
        if isinstance(time_to_maturity, Real) and time_to_maturity<=0.0:
            raise  TypeError("`time_to_maturity` must be positive.")
        if isinstance(annual_volatility, Real) and annual_volatility<=0.0:
            raise TypeError("`annual_volatility` must be positivie.")
        ttm = float(self.stockPaths.index[-1] if time_to_maturity is None else time_to_maturity)

        self.BlackScholesModels = {}
        for asset_name in self.stockPaths.columns:
            vol = (self.annualLogReturnVolatility[asset_name] 
                   if annual_volatility is None 
                   else annual_volatility)
            spot_price = self.stockPaths[asset_name].iloc[0]
            if pd.isna(spot_price):
                raise ValueError(f"Initial stock price for {asset_name!r} is missing.")
            spot_price = float(spot_price)

            self.BlackScholesModels[asset_name] = BlackScholes(
                spot_price = spot_price, 
                strike_price=strike_price, 
                annual_risk_free_rate=self.annualRiskFreeRate, 
                dividend_rate=dividend_rate, 
                annual_volatility=vol, 
                time_to_maturity=ttm,
                is_call=is_call
                )

    #
    # -- EXTERNAL FUNCTIONS -------------------------------------
    #

    # --- GETTERS --- # 
    def getVolatility(self, annual: bool=True, log_returns: bool=False, print_volatility: bool=False) -> np.ndarray:
        if annual:
            print_name = "Annual Volatility"
            vol = (self.annualLogReturnVolatility 
                   if log_returns 
                   else self.annualRawVolatility)
        else:
            print_name = "Period Volatility"
            vol = (self.periodLogReturnVolatility 
                   if log_returns 
                   else self.periodRawVolatility)
        volatility = np.asarray(vol, dtype=float)

        if print_volatility:
            print(
            pd.DataFrame(
                volatility,
                index=self.stockPaths.columns,
                columns=[print_name],
            ).to_string(float_format="{:.6f}".format)
        )

        return volatility

    # --- HEDGING --- #
    def deltaHedge(self, strike_price: float, annual_volatility: float|None=None, time_to_maturity: float | None=None, dividend_rate: float=0.0, is_call: bool=True, number_of_options: int=1):
        if (time_to_maturity is not None) and (time_to_maturity not in self.stockPaths.index):
            raise TypeError("maturity time is not found within stock paths data frame.")
        
        output = {}
        self.__createBlackScholesModels(strike_price, annual_volatility, time_to_maturity, dividend_rate, is_call=is_call)
        self.profit_loss = {}

        ttm = float(self.stockPaths.index[-1] if time_to_maturity is None else time_to_maturity)

        for asset_name in self.logReturnPaths.columns:
            time_index = self.stockPaths.index < ttm
            stock_path = self.stockPaths.loc[time_index, asset_name].to_numpy(dtype=float)
            time_path = self.stockPaths.index[time_index].to_numpy(dtype=float)
            time_until_maturity = (ttm - time_path)

            delta_vector = self.BlackScholesModels[asset_name].getDelta(inistial_value=stock_path, time_to_maturity=time_until_maturity)
            stocks_held = delta_vector
            cash_position = np.empty_like(delta_vector)
            cash_position[0] = self.BlackScholesModels[asset_name].fairPrice - stocks_held[0] * stock_path[0]
            
            for time_index in range(cash_position.shape[0]-1):
                cash_position[time_index+1] = float(cash_position[time_index] * np.exp(self.annualRiskFreeRate * self.period)
                                                    - stock_path[time_index+1]*(delta_vector[time_index+1] - delta_vector[time_index]))

            if not (ttm in self.stockPaths.index):
                print('maturity time is not found within stock paths data frame. Closest valu ')
            stock_final_value = np.asarray(self.stockPaths[asset_name].values[self.stockPaths.index==ttm], dtype=float)
            final_value = stock_final_value[0] if len(stock_final_value)==1 else stock_final_value

            option_payoff = max(final_value-strike_price, 0) if is_call else max(strike_price - final_value, 0)
            self.profit_loss[asset_name] = number_of_options * (delta_vector[-1]*final_value + cash_position[-1]*np.exp(self.annualRiskFreeRate * self.period) - option_payoff)

        print("{:<15} {:<10}".format('Stock','PnL'))
        for asset_name in self.profit_loss.keys():
            print("{:<15} {:<10.4f}".format(asset_name, self.profit_loss[asset_name]))
 