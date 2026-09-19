# Financial Engineering in Python

A selection of classes developed by me for option pricing under the binomial model and Black-Scholes models. As well as mean-variance portfolio optimisation. The classes were developed and written while undertaken the [Financial Engineering and Risk Management Specialization](https://www.coursera.org/specializations/financialengineering) course provided by Columbia university. The earlier modules of the course focus on using excel in the lectures and for the coursework, this is where I have focused my attention on the python scripts. The final module moves to python where the labs provided sufficient code to complete the assignments. The classes have been separated into seperate python scripts each with a guide of how they may be used. The classes and documentation may be added to at a later, to provide additional functionality or clarity within their guides.  

## Binomial Option Pricing Model

### Pricing a defaultable bond

Let's consider a simple example of pricing bond with a two year maturity, using the binomial model we can construct the lattice using the `BondLattice` class. However, we must construct a model of the short-term interest rates. This can also be described using class within the module. 

Suppose the short term interest rate has an initial rate of 4% and and increases can increase or decrease by 25% each year. For two-year, two-period model the short term interest rate lattice can be constructed as follows.

```python
from BinomialModels import ShortTermRates, BondLattice

rates = ShortTermRates(
    initial_rate = 0.04,
    up_move=1.25
    down_move=0.75,
    num_periods=2
)
```

To construct the bond lattice, and calculate its fair price, we using the `BondLattice` constructor. Here we assume that the default probability is constant and so use the `default_probability` argument. If instead the default probability is assumed to be dynamic, this too can be modelled with a binomial lattice. A recovery rate is also specified, allowing for the retrieval some of the notional value in the event of a defualt.

```python
bond_lattice = BondLattice(
    move_up_prob=0.5, 
    rates_lattice=rates_lattice,
    maturity_index=2, 
    recovery_rate=0.4,
    default_probability=0.1
)

print(f"The fair price of the bond is {bond_lattice.fairPrice:.2f}\n")

print("The pricing lattice is:")
bond_lattice.printLattice()
```

The price calculations are done at construction, and so it is left only to retrieve the desired information. The `fairPrice` method can be used to get the value at the fair price or inspection of the lattice itself. 

### Pricing a Cap

A cap is series of caplets, effectively European call options on the short-term interest rate where the rate is treated as the underlying asset. A caplet is defined ove a single period, and a cap is a portfolio comprising of a series of caplets covering the time until maturity. That is, for each short-term interest rate in the binomial model we calculate the "payoff" at the end of that period discounted by the current interest rate, $r_t$:
$$ \frac{max(r_t - K, 0)}{1 + r_t}. $$

Within the framework a cap option can priced using the `CapOption` class, using a `ShortTermRates` lattice to describe the short-term interest rate dynamics. 

```python
from BinomialModels import Cap

rates_lattice = ShortTermRates(
    initial_rate=0.025, 
    up_move=1.1,
    down_move=0.99,
    num_periods=10
)

cap_option = CapOption(
    strike_price = 0.03
    rates_lattice = rates_lattice
)

print(f"The fair price of the cap option is {cap_option.fairPrice:.4f}")
```

The cap is therefore a series of "payoffs" this is not easily visualised by a single binomial tree. A single caplet can be easily visualised by a tree, similarly to the bond in the previous example.


## Black-Scholes Model

The `BlackScholes` class always us to store and calculate information regarding the Black-Scholes model of an option price. Consider the following option European call option for a stock:

```python
from DerivativePricing import BlackScholes

option = BlackScholes(
    spot_price = 178,
    strike_price = 170,
    annual_risk_free_rate = 0.02,
    annual_volatility=0.20,
    time_to_maturity=1.0,
    is_call=True
)
```

The `is_call` parameter defaults to `True` but it was explicitly stated here for clarity. For this option we can calculate the fair price and the Greeks and the implied volatility for the market price.

```python
option.calculateImpliedVolatiltiy(market_price=100)
```
### Calculating implied volatility

The functions here utilise the `least_squares` function from the `scipy.optimize` package as it allows constraints to be placed in the optimisation. This is import to restrict the volatility to being positive. Importantly, the function also allows the implied volatility to be calculated for range of parameters, not just the ones for this particular option. this allows the volatility surface to be found with relative ease. An example dataset, `apple_stock.xlsx`, is provided within the repo which contains the bid and ask prices for call and put options on Apple stock between 2018 and 2020. Using the mid-value as the market price we can calculate the implied volatility for a range of maturities and strike prices. 

```python
import pandas as pd
apple_stock = pd.read_excel("data_apple.xlsx")
apple_stock["Mid"] = (apple_stock["Ask"] - apple_stock["Bid"]) / 2
market_data = apple_stock[apple_stock.Option_type=="Call"]
```
To find the implied volatilty we first define a `BlackScholes` object

```python
market_data["implied_volatility"] = option.calculateImpliedVolatility(
    market_price=market_data["Mid"].values,
    strike_price=market_data["Strike"].values,
    time_to_maturity=market_data["Maturity_days"].values
)

market_data.head()
```

## Portfolio Optimisation Example

As well as investigating deriviative pricing, the python classes here look at portfolio optimisation. The `MeanVariancePortfolio` class allows the user to find the optimal portfolio under a number of constraints, using a dataset of historic stock returns. Here, we will give an example of using the Sharpe ratio. Firstly, generate some sythetic stock returns data.

```python
from MeanVarianceOptimisation import MeanVariancePortfolio

# Synthetic daily returns
np.random.seed(101)

n_days = 1000
market_returns = np.random.normal(0.0004, 0.010, n_days)

returns = pd.DataFrame({
    "Asset A": market_returns + np.random.normal(0.0001, 0.006, n_days),
    "Asset B": np.random.normal(0.0006, 0.015, n_days),
    "Asset C": np.random.normal(0.0002, 0.008, n_days),
    "Asset D": np.random.normal(0.0008, 0.020, n_days),
})
```

The portfolio object can be defined and the risk-free interest rate can be set, if risk-free investing is allowed within the portfolio. If the risk-free weight is provided the class assumes investing in the risk-free asset is allowed, however, this does not have to be the case and can be changed using the `include_risk_free` argument in the consructor. Whether or not short selling is allowed can also be put in to the model. As is common, short selling is noted by negative weights associated with a stock. 

```python
# Daily risk-free rate
risk_free_rate = 0.0001

# Construct portfolio
portfolio = MeanVariancePortfolio(
    historic_returns=returns,
    risk_free_rate=risk_free_rate,
    include_risk_free=True,
    allow_shorts=True
)
```

Finding the portfolio which maximises the Sharpe ratio can be done using the `maximiseSharpeRatio` method, which allows the user to input whether borrowing against the risk-free weight is allows within the model. 

```python
# Find the maximum-Sharpe portfolio
result = portfolio.maximiseSharpeReturn(
    allow_borrowing=True
)
# Display the resulting portfolio
portfolio.printPortfolio()
print(f"Sharpe ratio: {portfolio.calculateSharpeRatio():.4f}")
```

The method does not assign any weight to the risk-free investment, only to the risky stocks. If such an investment is desired, namely to reduce risk and portfolio volatility, then this can be choosen via the `setTargetVolatility` method.

```python
portfolio.setTargetVolatility(target_volatility=0.004, allow_borrowing=True)
portfolio.printPortfolio()
```
