# required packages
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import warnings
from numbers import Real


## Generic Lattice
class Lattice:
    def __init__(self, num_period: int):
        if not isinstance(num_period, (int, np.integer)):
            raise TypeError("`num_period` must be an integer.")
        if num_period<=0:
            raise TypeError("number of periods must be a positive integer.")
        self._numPeriods = num_period
        self._latticeSize = self._numPeriods + 1
        self._lattice = np.zeros((self._latticeSize, self._latticeSize), dtype=float)

    def printLattice(self) -> None:
        for time_index in range(self._latticeSize):
                values = self._lattice[:(time_index+1), time_index]
                print(f"t = {time_index} | " + " | ".join(f"{value:>10.4f}" for value in reversed(values)))

    #
    # ---- INTERNAL FUNCTIONS ---------------------------------------------------------
    #
    def _checkIndex(self, period_index: int, outcome_index: int) -> None:
        if period_index<0 or period_index>self._latticeSize:
            raise IndexError("`period_index` is not within the bounds of the lattice.")
        if outcome_index>(period_index+1) or outcome_index<0:
            raise IndexError("`outcome_index` is not within the bounds of the lattice.")

    #
    # ---- GETTERS -----------------------------------------------------------------
    #    
    @property
    def numPeriods(self) -> int:
         return self._numPeriods

    @property
    def lattice(self) -> np.ndarray:
         return self._lattice
    
    @property
    def latticeSize(self) -> int:
        return self._latticeSize

    def getValue(self, period_index: int, outcome_index: int) -> float:
        self._checkIndex(period_index, outcome_index)
        return self._lattice[period_index - outcome_index, period_index]

    def getValues(self, period_index: int) -> npt.NDArray:
        self._checkIndex(period_index, 0)
        return self._lattice[:(period_index + 1), period_index]
    # 
    #  ---- SETTERS -----------------------------------------------------------------
    #   
    def setValue(self, value: float, period_index: int, outcome_index: int) -> None:
        self._checkIndex(period_index, outcome_index)
        self._lattice[period_index - outcome_index, period_index] = value

    def setAllValues(self, value: float) -> None:
        for period_index in range(self._numPeriods+1):
            for outcome_index in range(period_index+1):
                self.setValue(value, period_index, outcome_index)

class BinomialLattice(Lattice):
    """ 
    Defines a binomial lattice. Starting with an initial value at period=0, each daughter node
    has value defined by the parent node multiplied by an up move or down move factor.

    """
    def __init__(self, initial_value: float, up_move: float, down_move: float, num_periods: int):
        super().__init__(num_periods)
        self._initialValue = initial_value
        self._upMove = up_move
        self._downMove = down_move

        for period_index in range(self.latticeSize):
            for outcome_index in range(period_index+1):
                self.lattice[outcome_index, period_index] = (self.initialValue 
                                                             * (self.upMove ** (period_index - outcome_index)) 
                                                             * (self.downMove ** (outcome_index)))

    @property
    def upMove(self) -> float:
        return self._upMove
    @property
    def downMove(self) -> float:
        return self._downMove
    @property
    def initialValue(self) -> float:
        return self._initialValue

class AssetLattice(Lattice):
    def __int__(self, asset_lattice: Lattice, dividend_yield: float|npt.ArrayLike|None=None, 
                dividend_periods: npt.ArrayLike|None=None):
        super().__init__(asset_lattice.numPeriods)
        self._lattice = asset_lattice

        self._dividendActive = (dividend_yield is None)
        # Check dividend rates and payment index
        if self.dividendActive:
            dividend_yield = np.asarray(dividend_yield, dtype=float)
            dividend_periods = (np.arange(self.numPeriods)
                                      if (dividend_periods is None) 
                                      else np.asarray(dividend_periods, dtype=int))
            
        if isinstance(dividend_periods, np.ndarray) and dividend_periods.size == 1:
            dividend_periods = np.repeat(dividend_periods[0], self.numPeriods)

        if isinstance(dividend_periods, np.ndarray) and isinstance(dividend_yield, np.ndarray):
            num_dividends = len(dividend_periods)
            if len(dividend_yield)==1:
                divdend_rate = np.repeat([dividend_yield[0]], num_dividends)
            if len(dividend_yield) != len(dividend_periods):
                raise ValueError("If providing dividend rates and payment index, they must be the same size.")

        self._dividendYields = dividend_yield
        self._dividendPeriods = dividend_periods

    def _getDividendYield(self, period_index: int):
        if (self.dividendYields is None):
            raise RuntimeError("Yields not given.")
        if (self.dividendPeriods is None):
            raise RuntimeError("Yields not given.")
        
        self._checkIndex(period_index, 0)
        if isinstance(self.dividendYields, np.ndarray) and isinstance(self.dividendPeriods, np.ndarray):
            if np.isin(period_index, self.dividendPeriods):
                return self.dividendYields[self.dividendPeriods == period_index]
        return 0.0
        
    @property
    def dividendActive(self) -> bool:
        return self._dividendActive
    @property
    def dividendYields(self):
        return self._dividendYields
    @property
    def dividendPeriods(self):
        return self._dividendPeriods

    def dividendPayoff(self, period_index: int, outcome_index: int) -> float:
        return self.getValue(period_index, outcome_index)
        
## Interest Rates Lattices
class RatesLattice(Lattice):
    """ 
    Defines a generic lattice of interest rates.
    The class is not defined to be used directly but for interest rate lattices to inherit
    the getDiscount function and Lattice attributes.

    periodDuration (float): The length of a single period in years. The default is 1.0.
    """
    def __init__(self, num_periods: int, period_duration: float=1.0):
        super().__init__(num_periods)

        if period_duration<=0.0:
            raise ValueError("Period duration must be positive.")
        self._periodDuration = period_duration

    def getDiscount(self, period_index: int, outcome_index: int) -> float:
         return 1. / (1. + self.getValue(period_index, outcome_index))

    @property
    def periodDuration(self) -> float:
        return self._periodDuration

class ShortTermRates(RatesLattice):
    def __init__(self, initial_rate: float, up_move: float, down_move: float, num_periods: int, period_duration: float=1.0):
        if (initial_rate > 1.0):
            warnings.warn("Detected `initial_rate` greater than 1.0. The interests rate should be given in decimal form. The rate remains unchanged, redefine the lattice if this is incorrect.")
        if (initial_rate<0.0):
            warnings.warn("A negative initial rate has been given.")

        super().__init__(num_periods, period_duration)
        self._initialValue = initial_rate
        self._upMove = up_move
        self._downMove = down_move

        for period_index in range(self.latticeSize):
            for outcome_index in range(period_index+1):
                self.lattice[outcome_index, period_index] = (self._initialValue 
                                                             * (self._upMove ** (period_index - outcome_index)) 
                                                             * (self._downMove ** (outcome_index)))

    @property
    def upMove(self) -> float:
        return self._upMove
    @property
    def downMove(self) -> float:
        return self._downMove

## Payoff Lattice
class PayoffLattice(Lattice, ABC):
    def __init__(self, strike_price: float|None=None, asset_lattice: Lattice|None=None, 
                 rates_lattice: ShortTermRates|float|None=None, maturity_index: int=1):
        super().__init__(maturity_index)
        if (rates_lattice is not None):
            if isinstance(rates_lattice, Real):
                self._ratesLattice = ShortTermRates(initial_rate=float(rates_lattice), up_move=1.0, down_move=1.0, num_periods=maturity_index)
            elif isinstance(rates_lattice, ShortTermRates):
                self._ratesLattice = rates_lattice
            else:
                raise ValueError("`rates` should be a scalar positive number or a `ShortTermRates` object.")
            
        else:
            self._ratesLattice = None
        self._assetLattice = asset_lattice
        self._strikePrice = strike_price

        if maturity_index<1:
            raise ValueError("The option maturity should be positive")
        self._maturityIndex = maturity_index
        if (self.assetLattice is not None) and (self.maturityIndex>self.assetLattice.numPeriods):
            raise ValueError("The option maturity must be equal to or less than the number of periods in the asset lattices.")
        if (self.ratesLattice is not None) and (self.maturityIndex>self.ratesLattice.numPeriods):
            raise ValueError("The option maturity must be equal to or less than the number of periods in the rates lattices.")
                    
    @property
    def assetLattice(self) -> Lattice|None:
        return self._assetLattice
    @property
    @abstractmethod
    def ratesLattice(self) -> ShortTermRates|None:
        return self._ratesLattice
    @property
    def maturityIndex(self) -> int:
        return self._maturityIndex
    @property
    def fairPrice(self) -> float:
        return self._fairPrice
    @property
    def strikePrice(self) -> float|None:
        return self._strikePrice

    def getFairPrice(self, number_of_options: int) -> float:
        if (number_of_options<0):
            raise ValueError("Number of options held must be positive.")
        return float(number_of_options) * self.fairPrice

    @abstractmethod
    def _backwardValue(self, period_index, outcome_index) -> float:
        pass

    def _priceOption(self, payoff_function, **kwargs):
        raise NotImplementedError(
            "_priceOption is only supported by option lattices."
        )

    def _backwardInduction(self):
        for period_index in range(self.numPeriods, -1, -1):
            for outcome_index in range(period_index + 1):
                payoff = self._backwardValue(period_index, outcome_index)
                self.setValue(payoff, period_index, outcome_index)

        self._fairPrice = self.getValue(0,0)

## Elementary price lattice
class ElementaryPiceLattice(Lattice):
    def __init__(self, interest_rate_lattice: RatesLattice):
        super().__init__(interest_rate_lattice.numPeriods+1)

        self.setValue(1.0, 0, 0)

        for period_index in range(1, self.latticeSize):
            for outcome_index in range(period_index+1):
                if outcome_index == 0:
                    # P_{k+1, 0} = 0.5 * P_{k, 0} / (1.0 + r_{k, 0})
                    val_k0 = (0.5 * self.getValue(period_index-1, 0) 
                              * interest_rate_lattice.getDiscount(period_index-1, 0))
                    self.setValue(val_k0, period_index, 0)
                if outcome_index == period_index:
                    # P_{k+1, k+1} = 0.5 * P_{k, k} / (1.0 + r_{k, k})
                    val_kk = (0.5 * self.getValue(period_index-1 , period_index-1) 
                              * interest_rate_lattice.getDiscount(period_index-1, period_index-1) )
                    self.setValue(val_kk, period_index, period_index)
                if (outcome_index > 0) and (outcome_index<period_index):
                    # P_{k+1, s} = 0.5 * P_{k, s-1} / (1.0 + r_{k, s-1}) + 0.5 * P_{k,s} / (1.0 + r_{k, s})
                    val = (self.getValue(period_index-1, outcome_index-1) 
                           * interest_rate_lattice.getDiscount(period_index-1, outcome_index-1) )
                    val += (self.getValue(period_index-1, outcome_index) 
                            * interest_rate_lattice.getDiscount(period_index-1, outcome_index))
                    val *= 0.5
                    self.setValue(val, period_index, outcome_index)

    def zeroCouponBondPrice(self, notational_value: float, maturity_time: int) -> float:
        self._checkIndex(maturity_time, 0)
        return notational_value * np.sum( self.getValues(maturity_time) )
    
    def getSpotRate(self, period_index: int) -> float:
        self._checkIndex(period_index, 0)
        if period_index == 0:
            raise IndexError("`period_index` of zero given, `getSpotRate` was skipped.")

        p_sum = np.sum( self.getValues(period_index) )
        return p_sum**(-1.0 / period_index) - 1.0
## Hazard lattice
class HazardLattice(Lattice):
    def __init__(self, num_periods: int, hazard_function, *args):
        
        super().__init__(num_periods)
        for period_index in range(self.latticeSize):
            for outcome_index in range(period_index+1):
                val = hazard_function(period_index, outcome_index, *args)
                if val>1.0 or val<0.0:
                    raise TypeError("hazards must be between 0.0 and 1.0.")
                self.setValue(val, period_index, outcome_index)

## Bonds
class BondLattice(PayoffLattice):
    def __init__(self, move_up_prob: float, rates_lattice: ShortTermRates, maturity_index: int,
                 coupon_rate: float = 0.0, coupon_periods: npt.ArrayLike | None = None,
                 hazard_lattice: HazardLattice | None = None, recovery_rate: float = 0.0):
        
        if not isinstance(rates_lattice, ShortTermRates):
            raise TypeError("Interest rates lattice must be of type `ShortTermRates`.")
        # self, strike_price=0.0, asset_lattice: Lattice, rates: ShortTermRates|float, maturity_index: int
        super().__init__(strike_price=0.0, rates_lattice=rates_lattice, maturity_index=maturity_index)

        if not isinstance(hazard_lattice, HazardLattice):
            raise TypeError("`hazard_lattice` must be a `HazardLattice`.")
        if hazard_lattice is None:
            self._hazardLattice = HazardLattice(self.numPeriods, lambda period_index, outcome_index: 0.0)
            if recovery_rate!=0.0:
                warnings.warn("No hazard lattice is provided but recovery rate is non-zero, this will be ignored.")
            self._recoveryRate = 0.0
        else:
            if hazard_lattice.numPeriods < self.numPeriods:
                raise ValueError("hazard_lattice` must have at least as many periods as the bond itself, `n_periods`.")
            if (recovery_rate > 1.0) or (recovery_rate < 0.0):
                raise ValueError("Recovery rate must be between 0.0 and 1.0.")
            self._hazardLattice = hazard_lattice
            self._recoveryRate = recovery_rate

        if (move_up_prob < 0.0) or (move_up_prob > 1.0):
            raise ValueError("`move_up_prob` is a probability and must be between 0.0 and 1.0.")
        if coupon_rate < 0.0:
            raise ValueError("Coupon rate must be non-negative.")

        self._moveUpProb = move_up_prob
        self._moveDownProb = 1.0 - self._moveUpProb
        self._couponRate = coupon_rate
        self._couponPeriods = (np.arange(1, self.numPeriods + 1)
                               if coupon_periods is None 
                               else np.asarray(coupon_periods))
        if self.ratesLattice is None:
            raise ValueError("Rates lattice must be provided.")
        
        self._backwardInduction()
        
    # --- Internal functions --- # 
    def _coupon(self, period_index: int) -> float:
        self._checkIndex(period_index, 0)
        return self.couponRate if period_index in self.couponPeriods else 0.0

    def _backwardValue(self, period_index: int, outcome_index: int) -> float:
        self._checkIndex(period_index, outcome_index)

        if period_index == self.numPeriods:
            return 1.0 + self._coupon(self.numPeriods)
        else:
            if self.ratesLattice is None:
                        raise ValueError("Rates lattice must be provided.")
            node_value = (self.moveDownProb
                        * (1.0 - self.hazardLattice.getValue(period_index, outcome_index))
                        * self.getValue(period_index + 1, outcome_index))
            node_value += (self.moveUpProb
                        * (1.0 - self.hazardLattice.getValue(period_index, outcome_index))
                        * self.getValue(period_index + 1, outcome_index + 1))
            node_value += (self.hazardLattice.getValue(period_index, outcome_index)
                        * self._recoveryRate)
            node_value *= self.ratesLattice.getDiscount(period_index, outcome_index)
            node_value += self._coupon(period_index)

            return node_value

    # --- GETTERS --- #
    @property
    def moveUpProb(self) -> float:
        return self._moveUpProb
    @property
    def moveDownProb(self) -> float:
        return self._moveDownProb
    @property
    def couponRate(self) -> float:
        return self._couponRate
    @property
    def hazardLattice(self) -> HazardLattice:
        return self._hazardLattice
    @property
    def couponPeriods(self) -> npt.NDArray:
        return self._couponPeriods
    @property
    def ratesLattice(self) -> ShortTermRates:
        rates = self._ratesLattice
        if rates is None:
            raise RuntimeError("Rates lattice must be provided.")
        return rates
    
    # --- MODEL SPECIFICS --- #
    def setFaceValue(self, face_value: float) -> None:
        self._lattice *= face_value

class ZeroCouponBondLattice(BondLattice):
    def __init__(self, move_up_prob: float, rates_lattice: ShortTermRates, num_periods: int,
                 hazard_lattice: HazardLattice | None = None, recovery_rate: float = 0.0):
        super().__init__(move_up_prob, rates_lattice, num_periods,
                         coupon_rate=0.0, coupon_periods=[],
                         hazard_lattice=hazard_lattice, recovery_rate=recovery_rate)

## Forward and Future Contract
class FutureContract(PayoffLattice):
    def __init__(self, bond_lattice: BondLattice, maturity_index: int):
        super().__init__(maturity_index=maturity_index)
        self._bondLattice = bond_lattice
        self._moveUpProb = self._bondLattice.moveUpProb
        self._moveDownProb = self._bondLattice.moveDownProb

        if self.bondLattice.numPeriods < self.numPeriods:
            raise TypeError("The bond lattice must have at least as many periods as the contract.")

        self._backwardInduction()

    def _backwardValue(self, period_index: int, outcome_index) -> float:
        self._checkIndex(period_index, outcome_index)
        if period_index == self.maturityIndex:
            return self.bondLattice.getValue(period_index, outcome_index)
        else:
            payoff = (0.5 * (self.getValue(period_index+1, outcome_index) 
                      + self.getValue(period_index+1, outcome_index+1)))
            return payoff
        
    # --- GETTERS --- #
    @property
    def bondLattice(self) -> Lattice:
        return self._bondLattice
    
class ForwardContract(PayoffLattice):
    def __init__(self, bond_lattice: BondLattice, rates_lattice: ShortTermRates, move_up_prob: float, maturity_index: int):
        super().__init__(rates_lattice=rates_lattice, maturity_index=maturity_index)
        self._bondLattice = bond_lattice
        if move_up_prob<0.0 or move_up_prob>1.0:
            raise ValueError("The probability of moving 'upwards' must be a valid probability i.e. between 0.0 amd 1.0.")
        self._moveUpProb = move_up_prob
        self._moveDownProb = 1. - self._moveDownProb

        if self.bondLattice.numPeriods < self.maturityIndex:
            raise ValueError("The bond lattice must have at least as many periods as the contract.")
        if (self.ratesLattice is None):
            raise RuntimeError("Rates lattice not provided.") 
        
        self._zero_coupon_bond = ZeroCouponBondLattice(self._moveUpProb, self.ratesLattice, self.numPeriods)

        self._backwardInduction()
        self._fairPrice = self.getValue(0,0)

    def _backwardValue(self, period_index: int, outcome_index: int) -> float:
        self._checkIndex(period_index, outcome_index)
        if period_index == self.maturityIndex:
            return self.bondLattice.getValue(self.maturityIndex, outcome_index)
        else:
            if (self.ratesLattice is None):
                raise RuntimeError("Rates lattice not provided.")
            payoff = (0.5 * (self.getValue(period_index+1, outcome_index) 
                            + self.getValue(period_index+1, outcome_index+1)))
            payoff *= self.ratesLattice.getDiscount(period_index, outcome_index)
            if period_index==0 and outcome_index==0:
                payoff /= self._zero_coupon_bond.getFairPrice(1)
            return payoff
        
    # --- GETTERS --- #
    @property
    def bondLattice(self) -> Lattice:
        return self._bondLattice
    
# European Options
class EuropeanOptionLattice(PayoffLattice):
    def __init__(self, strike_price: float, asset_lattice: BinomialLattice, rates_lattice: ShortTermRates | float,
        maturity_index: int, call: bool = True, dividend_rate: float | npt.ArrayLike = 0.0, dividend_periods: int | npt.ArrayLike | None = None,
        separate_cashflows: bool = True):
        """        
        strike_price (float): The strike pice of the option.
        asset_lattice (BinomialLattive): The lattice describing the value of the underlying asset.
        rates_lattice (ShortTermRates): Lattice of interest rates of the period.
        dividend_rate (float|npt.ArrayLike): Optional the dividend paid by the underlying asset.
        dividend_payment_index (npt.ArrayLike): Option the periods at which the dividends are paid. 
            Defaults to all periods if not given and dividend_rate is.
        is_call (bool): A indication of where the option is a put or call. Defualts to Ture.
        maturity_index (int): A number of periods until maturity for the option. Defaults to the
            number of periods in the asset lattice.
        """
        super().__init__(maturity_index)

        self._strikePrice = strike_price
        self._assetLattice = asset_lattice
        self._separateCashflows = separate_cashflows

        # --------------------------------------------------
        # Interest rates
        # --------------------------------------------------
        self._constantRate = isinstance(rates_lattice, Real)
        if self._constantRate:
            self.ratesLattice = rates_lattice
        else:
            if not isinstance(rates_lattice, ShortTermRates):
                raise TypeError("`rates_lattice` must be a float or `ShortTermRates`.")
            self.ratesLattice = rates_lattice

        # --------------------------------------------------
        # Option type
        # --------------------------------------------------
        self._callTypeMultiplier = 1.0 if call else -1.0

        # --------------------------------------------------
        # Dividend yield
        # --------------------------------------------------
        dividend_rate = np.asarray(dividend_rate, dtype=float)

        if dividend_rate.ndim == 0:
            self.dividendRate = np.repeat(dividend_rate.item(), self.numPeriods)
        elif dividend_rate.ndim == 1:
            if len(dividend_rate) != self.numPeriods:
                raise ValueError("`dividend_rate` must have length `numPeriods`.")
            self.dividendRate = dividend_rate
        else:
            raise ValueError("`dividend_rate` must be a scalar or one-dimensional array.")

        if np.any(self.dividendRate < 0):
            raise ValueError("`dividend_rate` must be non-negative.")

        # --------------------------------------------------
        # Dividend periods
        # --------------------------------------------------

        if dividend_periods is None:
            self.dividendPeriods = np.array([], dtype=int)
        elif isinstance(dividend_periods, float):
            self.dividendPeriods = np.array([dividend_periods], dtype=int)
        else:
            self.dividendPeriods = np.asarray(dividend_periods,dtype=int)

        if self.dividendPeriods.ndim != 1:
            raise ValueError("`dividend_periods` must be an integer or one-dimensional array.")

        if np.any((self.dividendPeriods <= 0) | (self.dividendPeriods > self.numPeriods)):
            raise ValueError("`dividend_periods` must contain periods between 1 and `numPeriods`.")

        if len(np.unique(self.dividendPeriods)) != len(self.dividendPeriods):
            raise ValueError("`dividend_periods` cannot contain duplicates.")

        # --------------------------------------------------
        # Backward induction
        # --------------------------------------------------
        self._backwardInduction()

    def _dividendYield(self, period_index: int) -> float:
        """Return the dividend yield paid at the end of a period."""
        if period_index + 1 in self.dividendPeriods:
            return self.dividendRate[period_index]
        return 0.0

    def _backwardValue(self, period_index: int, outcome_index: int) -> float:
        stock_price = self.assetLattice.getValue(period_index, outcome_index)
        # Terminal payoff
        if period_index == self.numPeriods:
            return self._callTypeMultiplier * max(stock_price - self.strikePrice,0.0)
        # Short rate
        if self._constantRate:
            rate = self.ratesLattice
        else:
            rate = self.ratesLattice.getValue(period_index, outcome_index)

        dividend_yield = self._dividendYield(period_index)
        up = self.assetLattice.upMove
        down = self.assetLattice.downMove

        # if self.separateCashflows, dividend is paid separately from the option. 
        # The stock's ex-dividend total return is therefore: `price return + dividend yield`
        # giving the following RN probability. Otherwise the dividend is embedded in the 
        # stock-price process. The asset transition is adjusted by the dividend yield.
        # These values are obtained from the underlying lattice after applying the dividend adjustment.

        neutral_prob = ((1.0 + rate - dividend_yield - down) / (up - down)
                        if self.separateCashflows 
                        else (1.0 + rate - down) / (up - down))
        node_value = (neutral_prob * self.getValue(period_index + 1, outcome_index + 1) 
                      + (1.0 - neutral_prob) * self.getValue(period_index + 1, outcome_index)) / (1.0 + rate)

        return node_value
    @property
    def strikePrice(self) -> float:
        return self._strikePrice
    @property
    def assetLattice(self):
        return self._assetLattice
    @property
    def separateCashflows(self) -> bool:
        return self._separateCashflows
       
# Caplets and Floorlets
class CapletLattice(PayoffLattice):
    def __init__(self, strike_price: float, rates_lattice: ShortTermRates, maturity_index: int, risk_neutral_up_prob: float|None=None):
        super().__init__(strike_price=strike_price, asset_lattice=rates_lattice, rates_lattice=rates_lattice, maturity_index=(maturity_index -1))
        
        if (risk_neutral_up_prob is not None) and not isinstance(risk_neutral_up_prob, Real):
            raise TypeError("The risk-neutral probability must be a positive real number.")
        if isinstance(risk_neutral_up_prob, Real) and (risk_neutral_up_prob>1.0 or risk_neutral_up_prob<0.0):
            raise ValueError("The risk-neutral porbability must be between 0.0 and 1.0.")

        self._riskNeutralUpProb = 0.5 if (risk_neutral_up_prob is None) else risk_neutral_up_prob
        self._riskNeutralDownProb = 1. - self._riskNeutralUpProb
        # self._priceOption(self.payoffFunction, self._riskNeutralUpProb)

        self._backwardInduction()

    def _backwardValue(self, period_index: int, outcome_index: int) -> float:
         self._checkIndex(period_index, outcome_index)
         if period_index==self.maturityIndex:
             if self.strikePrice is None:
                 raise RuntimeError("Strike price not given.")
             if self.ratesLattice is None:
                 raise RuntimeError("Rates lattice not given.")
             payoff = max(self.ratesLattice.getValue(self.numPeriods, outcome_index) - self.strikePrice, 0)
             return payoff
         else:
            if self.ratesLattice is None:
                raise RuntimeError("Rates lattice not given.")
            
            payoff = ( self._riskNeutralUpProb*self.getValue(period_index+1, outcome_index+1) 
                      + self._riskNeutralUpProb * self.getValue(period_index+1, outcome_index))
            payoff *= self.ratesLattice.getDiscount(period_index, outcome_index)
            self.setValue(payoff, period_index, outcome_index) 
         return payoff

class FloorletLattice(EuropeanOptionLattice):
    def __init__(self, strike_price: float, rates_lattice: ShortTermRates, maturity_index: int, risk_neutral_up_prob: float|None=None):
        super().__init__(strike_price=strike_price, asset_lattice=rates_lattice, 
                         rates_lattice=rates_lattice, maturity_index=(maturity_index -1))

        self._strikePrice = strike_price
        self._riskNeutralUpProb = 0.5
        self._backwardInduction()

    def _backwardValue(self, period_index: int, outcome_index: int) -> float:
         payoff = max(self.strikePrice - self.ratesLattice.getValue(self.numPeriods, outcome_index), 0)
         payoff *= self.ratesLattice.getDiscount(self.numPeriods, outcome_index)
         return payoff
    
# Caps and Floors
class CapOption:
    def __init__(self, strike_price: float, rates_lattice: ShortTermRates, 
                 maturities: npt.ArrayLike, index_maturities: bool=True):
        
        if not isinstance(rates_lattice, ShortTermRates):
             raise TypeError("Interest rates must be of type `ShortTermRates`.")

        self._ratesLattice = rates_lattice
        self._strikePrice = strike_price
        self._maturities = np.asarray(maturities, dtype=int)
        self._periodDuration = self.rates.periodDuration
        self._accrualPeriod = self.maturities - np.insert(self.maturities[:-1], 0, 0.)
        if self._indexMaturities:
            self._accrualPeriod *= self._periodDuration
        self._indexMaturities = index_maturities

        if (len(self.maturities)==0):
            raise ValueError("Must have at least one maturity time.")
        if np.any(self.maturities<=0):
            raise ValueError("Maturity times must be postive.")

        max_maturity_time = np.max(self._maturities)
        self._numberOfCaplets = len(self.maturities)

        if max_maturity_time>self.rates.numPeriods:
            raise ValueError("The highest maturity time must be less than the number of periods in the rates lattice.")
        if len(np.unique(self.maturities)) != len(self.maturities):
            raise ValueError("Maturity times are not unique.")
        
        caplet_payoffs = []
        for period_index in range(self.numberOfCaplets):
            cap = CapletLattice(strike_price=self.strikePrice, rates_lattice=self._ratesLattice, maturity_index=self.maturities[period_index])
            caplet_payoffs.append( self.accrualPeriod[period_index] * cap.fairPrice )

        self._fairPrice = np.sum(caplet_payoffs)

    @property
    def maturities(self) -> npt.NDArray:
        return self._maturities
    @property
    def strikePrice(self) -> float:
        return self._strikePrice
    @property
    def rates(self) -> ShortTermRates:
        return self._ratesLattice
    @property
    def fairPrice(self) -> float:
        return self._fairPrice
    @property
    def accrualPeriod(self) -> npt.NDArray:
        return self._accrualPeriod
    @property
    def numberOfCaplets(self) -> int:
        return self._numberOfCaplets
    @property
    def ratesLattice(self) -> ShortTermRates:
        return self._ratesLattice
    
    def value(self, notional_value: float) -> float:
        return float(notional_value) * self.fairPrice

class FloorOption:
    def __init__(self, strike_price: float, rates: ShortTermRates, 
                 maturities: npt.ArrayLike, index_maturities: bool=True):
        
        if not isinstance(rates, ShortTermRates):
             raise TypeError("Interest rates must be of type `ShortTermRates`.")

        self._rates = rates
        self._strikePrice = strike_price
        self._maturities = np.asarray(maturities, dtype=int)
        self._periodDuration = self.rates.periodDuration
        self._accrualPeriod = self.maturities - np.insert(self.maturities[:-1], 0, 0.)
        if self._indexMaturities:
            self._accrualPeriod *= self._periodDuration
        self._indexMaturities = index_maturities

        if (len(self.maturities)==0):
            raise ValueError("Must have at least one maturity time.")
        if np.any(self.maturities<=0):
            raise ValueError("Maturity times must be postive.")

        max_maturity_time = np.max(self._maturities)
        self._numberOfCaplets = len(self.maturities)

        if max_maturity_time>rates.numPeriods:
            raise ValueError("The highest maturity time must be less than the number of periods in the rates lattice.")
        if len(np.unique(self.maturities)) != len(self.maturities):
            raise ValueError("Maturity times are not unique.")
        
        floorlet_payoffs = []
        for period_index in range(self.numberOfCaplets):
            floor = FloorletLattice(strike_price=self.strikePrice, rates_lattice=rates, maturity_index=self.maturities[period_index])
            floorlet_payoffs.append( self.accrualPeriod[period_index] * floor.fairPrice )

        self._fairPrice = np.sum(floorlet_payoffs)

    @property
    def maturities(self) -> npt.NDArray:
        return self._maturities
    @property
    def strikePrice(self) -> float:
        return self._strikePrice
    @property
    def rates(self) -> ShortTermRates:
        return self._rates
    @property
    def fairPrice(self) -> float:
        return self._fairPrice
    @property
    def accrualPeriod(self) -> npt.NDArray:
        return self._accrualPeriod
    @property
    def numberOfCaplets(self) -> int:
        return self._numberOfCaplets

    def value(self, notional_value: float) -> float:
        return float(notional_value) * self.fairPrice

## Swap contract
class SwapContract(Lattice):
    def __init__(self, strike_rate: float, maturity_period: int, rates_lattice: ShortTermRates, is_call: bool, first_payment_period: int):
        super().__init__(maturity_period-1)
        self._optionFactor = 1.*(is_call) - 1.*(1 - is_call)
        self._ratesLattice = rates_lattice
        self._strikeRate = strike_rate
        self._firstPaymentPeriod = first_payment_period

        for outcome_index in range(self.latticeSize):
            value = (self._optionFactor 
                     * (rates_lattice.getValue(self.numPeriods, outcome_index) 
                        - self._strikeRate))
            value *= rates_lattice.getDiscount(self.numPeriods, outcome_index)
            self.setValue(value, self.numPeriods, outcome_index)

        for period_index in range(self.numPeriods-1, -1, -1):
            for outcome_index in range(period_index + 1):
                value = (0.5 * (self.getValue(period_index+1, outcome_index) 
                                + self.getValue(period_index+1, outcome_index+1)))
                if period_index >= (self._firstPaymentPeriod-1):
                    value += (self._optionFactor 
                              * (self._ratesLattice.getValue(period_index, outcome_index) 
                                 - self._strikeRate))
                value *= rates_lattice.getDiscount(period_index, outcome_index)
                self.setValue(value, period_index, outcome_index)
        

    # --- GETTERS --- #
    @property
    def ratesLattice(self) -> RatesLattice:
        return self._ratesLattice

    @property
    def strike(self) -> float:
        return self._strikeRate

    @property
    def optionFactor(self) -> float:
        return self._optionFactor

    @property
    def firstPaymentPeriod(self) -> float:
        return self._firstPaymentPeriod

    def getFairPrice(self, notational_value: float) -> float:
        return notational_value * self.getValue(0,0)