# required packages
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import warnings
from numbers import Real

class Option(ABC):
    def __init__(self, strike_price: float, time_to_maturity: float, is_call: bool):
        self._strikePrice = strike_price
        self._timeToMaturity = time_to_maturity
        self._isCall = is_call
        self._payoffFactor = 1*self._isCall - 1*(not self._isCall)

    # --- GETTERS --- # 
    @property
    def strikePrice(self) -> float:
        return self._strikePrice

    @property
    def maturity(self) -> float:
        return self._timeToMaturity

    @property
    def isCall(self) -> bool:
        return self._isCall

    # --- abstract functions --- #
    @abstractmethod
    def payoffFunction(self, asset_price: float) -> float:
        pass

    @abstractmethod
    def isAmerican(self) -> bool:
        return False

class EuropeanOption(Option):
    def __init__(self, strike_price: float, time_to_maturity: float, is_call: bool):
        super().__init__(strike_price, time_to_maturity, is_call)
        self._isAmerican = False

    def payoffFunction(self, asset_price: float) -> float:
        return self._payoffFactor * np.maximum(asset_price - self.strikePrice, 0.0)

    def isAmerican(self) -> bool:
        return self._isAmerican
    
class AmericanOption(Option):
    def __init__(self, strike: float, maturity: float, is_call: bool):
        super().__init__(strike, maturity, is_call)
        self._isAmerican = True

    def payoffFunction(self, asset_price: float) -> float:
        return self._payoffFactor * np.maximum(asset_price - self.strikePrice, 0.0)
    
    def isAmerican(self) -> bool:
        return self._isAmerican

class Lattice:
    def __init__(self, num_period: int):
        if num_period<=0:
            raise ValueError("number of periods must be a positive integer.")
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
                self.lattice[outcome_index, period_index] = (self._initialValue 
                                                             * (self._upMove ** (period_index - outcome_index)) 
                                                             * (self._downMove ** (outcome_index)))

class RatesLattice(Lattice):
    """ 
    Defines a generic lattice of interest rates.
    The class is not defined to be used directly but for interest rate lattices to inherit
    the getDiscount function and Lattice attributes.
    """
    def __init__(self, num_periods: int):
          super().__init__(num_periods)

    def getDiscount(self, period_index: int, outcome_index: int) -> float:
         return 1. / (1. + self.getValue(period_index, outcome_index))

class ShortTermRates(RatesLattice):
    def __init__(self, initial_rate: float, up_move: float, down_move: float, num_periods: int):
        if (initial_rate > 1.0):
            warnings.warn("Detected `initial_rate` greater than 1.0. The interests rate should be given in decimal form. The rate remains unchanged, redefine the lattice if this is incorrect.")
        if (initial_rate<0.0):
            warnings.warn("A negative initial rate has been given.")

        super().__init__(num_periods)
        self._initialValue = initial_rate
        self._upMove = up_move
        self._downMove = down_move

        for period_index in range(self.latticeSize):
            for outcome_index in range(period_index+1):
                self.lattice[outcome_index, period_index] = (self._initialValue 
                                                             * (self._upMove ** (period_index - outcome_index)) 
                                                             * (self._downMove ** (outcome_index)))

class ShortTermRatesBDT(RatesLattice):
    def __init__(self, drift: npt.ArrayLike, volatility: npt.ArrayLike, num_periods: int):
        drift = np.asarray(drift, dtype=float)
        volatility = np.asarray(volatility, dtype=float)

        if drift.size != volatility.size:
            raise TypeError("`drift` and `volatility must have the same size.")
        if drift.ndim != 1:
             raise TypeError("'drift and volatilty must be 1-dimensional.")
        
        super().__init__(drift.size - 1)

        self.volatility = volatility
        self.drift = drift

        for period_index in range(self.latticeSize):
            for outcome_index in range(period_index+1):
                rate = (self.drift[period_index] 
                        * np.exp(self.volatility[period_index] * outcome_index) )
                self.setValue(rate, period_index, outcome_index)

class OptionPricer(Lattice):
    def __init__(self, asset_lattice: Lattice, interest_rate: float|ShortTermRates, num_periods: int, coupon_factor: float=0.0):
        if num_periods > asset_lattice.numPeriods:
            raise ValueError("The maturity time for the option, `n_periods`, must be larger than the asset lattice.")
        
        super().__init__(num_periods)
        if not isinstance(interest_rate, (Real, ShortTermRates)):
            raise ValueError("`interest_rate` must be of type float or `ShortTermRates`.")
        
        self._couponFactor = coupon_factor
        self._shortTermRates = isinstance(interest_rate, ShortTermRates)
        self._assetLattice = asset_lattice

        if isinstance(interest_rate, ShortTermRates):
            if self.numPeriods > asset_lattice.numPeriods:
                raise ValueError("The maturity time for the option, `n_periods`, must be larger than the short-term rates lattice.")

            self._interestRateLattice = interest_rate
            self._riskNeutralUpProb = 0.5

            self._riskNeutralDownProb = 0.5
        elif isinstance(interest_rate, Real):
            asset_up_move = self._assetLattice.getValue(1, 1) / self._assetLattice.getValue(0, 0)
            asset_down_move = self._assetLattice.getValue(1, 0) / self._assetLattice.getValue(0, 0)
            
            self._interestRateLattice = ShortTermRates(interest_rate, 1.0, 1.0, self.numPeriods)
            self._riskNeutralUpProb = (interest_rate - asset_down_move - self._couponFactor) / (asset_up_move - asset_down_move)
            self._riskNeutralDownProb = 1.0 - self._riskNeutralUpProb
        else:
            raise TypeError
   # --- GETTER --- #
    @property
    def couponFactor(self) -> float:
        return self._couponFactor

    @property
    def ratesLattice(self) -> RatesLattice:
        return self._interestRateLattice

    @property
    def assetLattice(self) -> Lattice:
        return self._assetLattice

    def fairPrice(self, option) -> float:
        assert isinstance(self._interestRateLattice, ShortTermRates)

        # calculate the payoff for the final time period of the asset latice
        for outcome_index in range(self.latticeSize):
            final_payoff = option.payoffFunction(self._assetLattice.getValue(self.numPeriods, outcome_index))
            self.setValue(final_payoff, self.numPeriods, outcome_index)

        # work backwards, calculating the expected payoff at each time point
        for period_index in range(self.numPeriods-1, -1, -1):
            for outcome_index in range(period_index+1):

                payoff = ((self._riskNeutralDownProb * self.getValue(period_index+1, outcome_index) 
                          + self._riskNeutralUpProb * self.getValue(period_index+1, outcome_index+1)) 
                          * (self._interestRateLattice.getDiscount(period_index, outcome_index)))
                if option.isAmerican:
                    current_payoff = option.payoff_function(self._assetLattice.getValue(period_index, outcome_index))
                    if payoff < current_payoff:
                        payoff = current_payoff
                self.setValue(payoff, period_index, outcome_index)

        return self.getValue(0,0)

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

class HazardLattice(Lattice):
    def __init__(self, num_periods: int, hazard_function, *args):
        
        super().__init__(num_periods)
        for period_index in range(self.latticeSize):
            for outcome_index in range(period_index+1):
                val = hazard_function(period_index, outcome_index, *args)
                if val>1.0 or val<0.0:
                    raise ValueError("hazards must be between 0.0 and 1.0.")
                self.setValue(val, period_index, outcome_index)

class BondLattice(Lattice):
    def __init__(self, move_up_prob: float, interest_rates: ShortTermRates, num_periods: int,
                 coupon_rate: float = 0.0, coupon_periods: set[int] | None = None,
                 hazard_lattice: HazardLattice | None = None, recovery_rate: float = 0.0):
        super().__init__(num_periods)

        if hazard_lattice is None:
            self._hazardLattice = HazardLattice(self.numPeriods, lambda x: 0.0)
            self._recoveryRate = 0.0
        else:
            if hazard_lattice.numPeriods < self.numPeriods:
                raise ValueError("hazard_lattice` must have at least as many periods as the bond itself, `n_periods`.")
            if (recovery_rate > 1.0) or (recovery_rate < 0.0):
                raise TypeError
            self._hazardLattice = hazard_lattice
            self._recoveryRate = recovery_rate

        if (move_up_prob < 0.0) or (move_up_prob > 1.0):
            raise ValueError("`move_up_prob` must be between 0.0 and 1.0 to be a valid probability.")
        if coupon_rate < 0.0:
            raise ValueError("`coupon_rate` must be non-negative.")

        self._moveUpProb = move_up_prob
        self._moveDownProb = 1.0 - self._moveUpProb
        self._interestRatesLattice = interest_rates
        self._couponRate = coupon_rate
        self._couponPeriods = (set(range(1, num_periods + 1))
                               if coupon_periods is None else set(coupon_periods))

        for outcome_index in range(self.latticeSize):
            self.setValue(1.0 + self._coupon(self.numPeriods), self.numPeriods, outcome_index)

        for period_index in range(self.numPeriods - 1, -1, -1):
            for outcome_index in range(period_index + 1):
                payoff = (self._moveDownProb
                          * (1.0 - self._hazardLattice.getValue(period_index, outcome_index))
                          * self.getValue(period_index + 1, outcome_index))
                payoff += (self._moveUpProb
                           * (1.0 - self._hazardLattice.getValue(period_index, outcome_index))
                           * self.getValue(period_index + 1, outcome_index + 1))
                payoff += (self._hazardLattice.getValue(period_index, outcome_index)
                           * self._recoveryRate)
                payoff *= self._interestRatesLattice.getDiscount(period_index, outcome_index)
                payoff += self._coupon(period_index)

                self.setValue(payoff, period_index, outcome_index)

    def _coupon(self, period_index: int) -> float:
        self._checkIndex(period_index, 0)
        return self._couponRate if period_index in self._couponPeriods else 0.0

    # --- GETTERS --- #
    @property
    def moveUpProb(self) -> float:
        return self._moveUpProb

    @property
    def moveDownProb(self) -> float:
        return self._moveDownProb

    @property
    def ratesLattice(self) -> RatesLattice:
        return self._interestRatesLattice

    @property
    def couponRate(self) -> float:
        return self._couponRate

    # --- MODEL SPECIFICS --- #
    def setFaceValue(self, face_value: float) -> None:
        self._lattice *= face_value

    def getFairPrice(self, face_value: float = 1.0) -> float:
        return face_value * self.getValue(0, 0)

class ZeroCouponBondLattice(BondLattice):
    def __init__(self, move_up_prob: float, interest_rates: ShortTermRates, num_periods: int,
                 hazard_lattice: HazardLattice | None = None, recovery_rate: float = 0.0):
        super().__init__(move_up_prob, interest_rates, num_periods,
                          coupon_rate=0.0, coupon_periods=set(),
                          hazard_lattice=hazard_lattice, recovery_rate=recovery_rate)

class FutureContract(Lattice):
    def __init__(self, bond_lattice: BondLattice, num_periods: int):
        super().__init__(num_periods)
        self._bondLattice = bond_lattice
        self._moveUpProb = self._bondLattice.moveUpProb
        self._moveDownProb = self._bondLattice.moveDownProb

        if self.bondLattice.numPeriods < self.numPeriods:
            raise ValueError("The bond lattice must have at least as many periods as the contract.")

        for outcome_index in range(self.latticeSize):
            self.setValue(self.bondLattice.getValue(self.numPeriods, outcome_index), self.numPeriods, outcome_index)

        for period_index in range(self.numPeriods-1, -1, -1):
            for outcome_index in range(period_index+1):
                payoff = (0.5 * (self.getValue(period_index+1, outcome_index) 
                                 + self.getValue(period_index+1, outcome_index+1)))
                self.setValue(payoff, period_index, outcome_index)

        self._fairPrice = self.getValue(0,0)

    # --- GETTERS --- #
    @property
    def bondLattice(self) -> Lattice:
        return self._bondLattice

    @property
    def fairPrice(self) -> float:
        return self._fairPrice


    def getFairPrice(self) -> float:
        return self._fairPrice
    
class ForwardContract(Lattice):
    def __init__(self, bond_lattice: BondLattice, interest_lattice: ShortTermRates, move_up_prob: float, num_periods: int):
        super().__init__(num_periods)
        self._bondLattice = bond_lattice
        self._ratesLattice = interest_lattice
        if move_up_prob<0.0 or move_up_prob>1.0:
            raise ValueError("The probability of moving 'upwards' must be a valid probability i.e. between 0.0 amd 1.0.")
        self._moveUpProb = move_up_prob
        self._moveDownProb = 1. - self._moveDownProb

        if self.bondLattice.numPeriods < self.numPeriods:
            raise TypeError("The bond lattice must have at least as many periods as the contract.")

        for outcome_index in range(self.latticeSize):
            self.setValue(self._bondLattice.getValue(self.numPeriods, outcome_index), 
                          self.numPeriods, 
                          outcome_index)

        for period_index in range(self.numPeriods-1, -1, -1):
            for outcome_index in range(period_index+1):
                payoff = (0.5 * (self.getValue(period_index+1, outcome_index) 
                                + self.getValue(period_index+1, outcome_index+1)) 
                         * self._ratesLattice.getDiscount(period_index, outcome_index))
                self.setValue(payoff, period_index, outcome_index)

        zero_coupon_bond = ZeroCouponBondLattice(self._moveUpProb, self._ratesLattice, self.numPeriods)
        self._fairPrice = self.getValue(0,0) / zero_coupon_bond.getValue(0,0)

    # --- GETTERS --- #
    @property
    def bondLattice(self) -> Lattice:
        return self._bondLattice

    @property
    def ratesLattice(self) -> ShortTermRates:
        return self._ratesLattice

    @property
    def fairPrice(self) -> float:
        return self._fairPrice

    # getFairPrice is left in as it is the usual convention in other classes 
    # with notational amount arguments
    def getFairPrice(self) -> float:
        return self._fairPrice

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