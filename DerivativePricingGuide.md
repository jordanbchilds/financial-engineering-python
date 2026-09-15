# Black-Scholes Option Pricing

The `BlackScholes` class provides a collection of tools for pricing European options under the **Black-Scholes-Merton model**, calculating option Greeks, estimating implied volatility, and recovering the risk-neutral stock-price distribution from option prices.

The class is designed to work with both **scalar values and NumPy arrays**, making it suitable for analysing individual options as well as option surfaces.

## 1. The Black-Scholes Model

The class assumes the underlying asset follows a geometric Brownian motion under the risk-neutral measure:
$$ dS_t = (r-q)S_t\,dt + \sigma S_t\,dW_t $$
where:
- \($S_t$\) is the underlying asset price at time $t$
- \($r$\) is the continuously compounded risk-free rate
- \($q$\) is the continuously compounded dividend yield
- \($\sigma$\) is the annualised volatility
- \($W_t$\) denotes Brownian motion

For an option with strike price, $K$, and time to maturity, $T$, the Black-Scholes quantities are

$$ d_1 = \frac{ \ln(S_0/K) + (r-q+\frac{1}{2}\sigma^2)T }{ \sigma\sqrt{T} }, $$
$$ d_2=d_1-\sigma\sqrt{T}. $$

The European call price is

$$ C = S_0 e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2), $$

where $\Phi(\cdot)$ is the cumulation probability function of tge standard normal distribution. The put price is obtained by rearranging the put-call parity, giving
$$ P = C + Ke^{-rT} - Se^{-qT}. $$


## 2. Creating an Option

A `BlackScholes` object is created by specifying the market and contract parameters:

```python
from BlackScholes import BlackScholes

option = BlackScholes(
    spot_price=100.0,
    strike_price=105.0,
    annual_risk_free_rate=0.05,
    dividend_rate=0.02,
    annual_volatility=0.20,
    time_to_maturity=1.0,
    is_call=True
)
```

The parameters are:

| Parameter | Description |
|---|---|
| `spot_price` | Current price of the underlying asset, $S_0$ |
| `strike_price` | Option strike price, $K$ |
| `annual_risk_free_rate` | Continuously compounded annual risk-free rate, $r$ |
| `dividend_rate` | Continuously compounded annual dividend yield, $q$ |
| `annual_volatility` | Annualised volatility, $\sigma$ |
| `time_to_maturity` | Time remaining until expiry in years, $T$ |
| `is_call` | `True` for a call, `False` for a put |

Both `annual_volatility` and `time_to_maturity` must be strictly greater than zero.

## 3. Option Price

The fair value of the option is available through the `fairPrice` member, `BlackScholes.fairPrice`, which returns the fair price, as calculated via the Black-Scholes formula, of the option with the parameters provided at object creation. Altrernatively, the fair price can be calculated for varying parameters using the `BlackScholes.calculateOptionPrice()` function, which returns `BlackScholes.fairPrice` if no arguments are passed or returns a float or NumPy array if new values are provided. For example
```python
strikes = np.linspace(start=80, stop=120, num=100)

prices = option.calculateOptionPrice(
    strike_price=strikes
)
```
Calculates the option price for the ranges of strike prices defined in `strikes` and internal values of the remaining parameters. The methods also accept NumPy arrays for `initial_value`, `strike_price`, `time_to_maturity`, and `volatility`. If passing arrays these should be of the same shape, if not an error will be raised, for example
```python
strikes = np.linspace(start=80, stop=120, num=100)
times = np.linspace(start=0.5, stop1.0, num=100)
prices = option.calculateOptionPrice(
    strike_price=strikes,
    time_to_maturity=times
)
```
is fine but 
```python
strikes = np.linspace(start=80, stop=120, num=100)
times = np.linspace(start=0.5, stop1.0, num=120)
prices = option.calculateOptionPrice(
    strike_price=strikes,
    time_to_maturity=times
)
```
will throw an error. 

## 4. The Greeks

The class calculates the principal first- and second-order Greeks. Similarly to the `calulateOptionPrice()` function the Greek functions can also take a range of values in the form of floats or NumPy arrays, subject to the same shape conditions.

### 4.1. Delta

Delta measures the sensitivity of the option price to a change in the underlying price:
$$ \Delta = \frac{\partial V}{\partial S}. $$

```python
option.getDelta()
```
For a call:
$$ \Delta_C=e^{-qT}N(d_1) $$
and for a put:
$$ \Delta_P=e^{-qT}(N(d_1)-1). $$


### 4.2. Gamma

Gamma measures the sensitivity of delta to the underlying price:
$$ \Gamma = \frac{\partial^2 V}{\partial S^2}. $$

```python
option.getGamma()
```

This is the same for both calls and puts:
$$ \Gamma = \frac{ e^{-qT}\phi(d_1) }{ S\sigma\sqrt{T}},$$

where \(\phi(\cdot)\) is the probability density function of a standard normal distribution.

### 4.3. Vega

Vega measures the sensitivity of the option price to volatility:
$$ \text{Vega} = \frac{\partial V}{\partial \sigma}. $$

```python
option.getVega()
```
The implementation returns vega for a **unit change in volatility**. Therefore, if volatility is quoted in percentage points, the sensitivity to a 1 percentage-point change is approximately:

```python
option.getVega() * 0.01
```

### 4.4. Theta

Theta measures the sensitivity of the option value to the passage of time:
$$ \Theta = \frac{\partial V}{\partial T}. $$

```python
option.getTheta()
```

The implementation uses the derivative with respect to time remaining to maturity. Consequently, its sign convention should be considered carefully when interpreting it as the daily passage-of-time decay commonly quoted by option traders. The additional arguments can be used to calculate the value of the Theta (and the other Greeks) for varying parameters, e.g. as time to maturity decreases. 

```python
times = np.linspace(start=1.0, end=0.0, num=252)
option.getTheta(
    time_to_maturity = times
)
```

## 5. Implied Volatility

The implied volatility is the volatility of the model where the fair price is equal to the market price and can be found using the `BlackScholes.calculateImpliedVolatility()` method.

```python
implied_vol = option.calculateImpliedVolatility(
    market_price=7.50
)
```

The method solves $ V_{\mathrm{BS}}(\sigma)-V_{\mathrm{market}}=0 $ numerically using `scipy.optimize.fsolve`. Additional parameters can be passed to `fsolve` through the `**kwargs` convention. Strike and maturity can also be specified as floats of arrays, 
this is useful for constructing an implied-volatility smile or surface from market option prices.

```python
strikes = np.linspace(start=80, stop=120, num=100)
implied_vol = option.calculateImpliedVolatility(
    market_price=7.50,
    strike_price=strikes,
    time_to_maturity=0.75
)
```

**Note:** `fsolve` is an unconstrained root finder. The resulting volatility should therefore be checked for validity, particularly when market prices are close to their arbitrage bounds or when the initial guess is poor.

---

## 6. Estimating the Risk-Neutral Stock Distribution

One of the more advanced features of the class is its ability to estimate the **risk-neutral probability density** of the underlying stock price from option prices. The method is based on the Breeden-Litzenberger relationship:

$$ f_Q(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}, $$

where $f_Q(K)$ is the risk-neutral probability density. The class approximates the second derivative numerically using finite differences.

```python
density = option.estimateStockDistribution(
    lower_bound=50.0,
    upper_bound=150.0,
    n=500
)
```
This returns a Pandas DataFrame containing:
- `stock_value` — possible terminal stock prices
- `density` — estimated risk-neutral probability density

The result can be visualised with:

```python
density.plot(
    x="stock_value",
    y="density"
)
```

Alternatively, the spacing between stock-price points can be specified using `delta`:

```python
density = option.estimateStockDistribution(
    lower_bound=50.0,
    upper_bound=150.0,
    delta=0.25
)
```

## 7. Recovering a Distribution from Market Data

The same approach can be applied directly to observed market option prices.

```python
density = option.estimateStockDistributionFromData(
    strike_price=strikes,
    market_price=market_prices
)
```

The strike prices must be **equally spaced**, since the implementation uses a central finite-difference approximation:

$$\frac{\partial^2 C}{\partial K^2} \approx \frac{ C(K+\Delta K)-2C(K)+C(K-\Delta K) }{ (\Delta K)^2 }.$$

This provides an estimate of the market-implied risk-neutral distribution.

## 8. Pricing Arbitrary Payoffs

Once a risk-neutral distribution has been estimated, the class can use it to value arbitrary payoff functions.

For a payoff $g(S_T)$,
$$ V_0 = e^{-rT} E_Q[g(S_T)] = = e^{-rT} \int g(S_T)f_Q(S_T)\,dS_T$$

A payoff can therefore be supplied as a Python function. For example, a digital call paying £1 if the stock finishes above £110:

```python
def digital_payoff(S):
    return 1.0 if S > 110.0 else 0.0
```

The payoff can then be priced using the model-implied distribution, or obtained from market option prices

```python
price = option.payoffFunctionFairPrice(
    payoff_function=digital_payoff,
    lower_bound=1.0,
    upper_bound=200.0,
    n=5000
)

price = option.payoffFunctionFairPriceFromData(
    payoff_function=digital_payoff,
    strike_price=strikes,
    market_price=market_prices
)
```

Providing a general framework for valuing non-standard payoffs from an estimated risk-neutral distribution.

## 9. Vectorised Calculations

A key feature of the class is that many calculations accept NumPy arrays. For example, an entire strike-price curve can be generated efficiently:

```python
strikes = np.linspace(70, 130, 121)

prices = option.calculateOptionPrice(
    strike_price=strikes
)
```

Similarly, delta can be calculated over a range of underlying prices:

```python
spot_prices = np.linspace(70, 130, 121)

deltas = option.getDelta(
    initial_value=spot_prices
)
```

This makes the class useful for generating:

- option price curves
- Greek profiles
- volatility smiles
- sensitivity analyses
- risk-neutral distributions
- numerical payoff valuations

without repeatedly constructing new option objects.


## 10. Dependencies

The class relies on:

```python
import numpy as np
import numpy.typing as npt
import pandas as pd

from scipy.stats import norm
from scipy.optimize import fsolve
```

The main external dependencies are therefore **NumPy**, **Pandas**, and **SciPy**.

## 11. Summary

The `BlackScholes` class provides three main layers of functionality:

| Functionality | Methods |
|---|---|
| **Black-Scholes pricing** | `calculateOptionPrice()`, `get_d1d2()` |
| **Risk sensitivities** | `getDelta()`, `getGamma()`, `getVega()`, `getTheta()` |
| **Volatility calibration** | `calculateImpliedVolatility()` |
| **Risk-neutral distribution** | `estimateStockDistribution()`, `estimateStockDistributionFromData()` |
| **Arbitrary payoff valuation** | `payoffFunctionFairPrice()`, `payoffFunctionFairPriceFromData()` |

The class therefore progresses from the standard Black-Scholes framework to **market-implied distribution estimation and model-independent payoff valuation**, making it useful as a building block for more advanced financial-engineering applications.
