# Financial Engineering in Python

A selection of classes developed by me for option pricing under the binomial model and Black-Scholes models. As well as mean-variance portfolio optimisation. The classes were developed and written while undertaken the [Financial Engineering and Risk Management Specialization](https://www.coursera.org/specializations/financialengineering) course provided by Columbia university. The earlier modules of the course focus on using excel in the lectures and for the coursework, this is where I have focused my attention on the python scripts. The final module moves to python where the labs provided sufficient code to complete the assignments. The classes have been separated into seperate python scripts each with a guide of how they may be used. The classes and documentation may be added to at a later, to provide additional functionality or clarity within their guides.  

# Binomial Lattice Example

## Pricing a defaultable bond

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

## Pricing a Cap

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


# Derivative Pricing Example

# Portfolio Optimisation Example

