# Guide to the Binomial Pricing Models

This module implements several financial models using binomial lattices. The classes are designed to be combined: an underlying asset or interest-rate lattice is first constructed, and derivative or fixed-income instruments are then priced from those lattices.

The main classes are:

| Class | Purpose |
|---|---|
| `Lattice` | Generic binomial lattice |
| `BinomialLattice` | Binomial lattice for an underlying asset |
| `ShortTermRates` | Binomial lattice of short-term interest rates |
| `EuropeanOption` | European call or put |
| `AmericanOption` | American call or put |
| `OptionPricer` | Prices options using an asset and interest-rate lattice |
| `ElementaryPiceLattice` | State-price lattice |
| `HazardLattice` | Default-probability lattice |
| `BondLattice` | Coupon-paying bond |
| `ZeroCouponBondLattice` | Zero-coupon bond |
| `FutureContract` | Futures contract on a bond |
| `ForwardContract` | Forward contract on a bond |
| `SwapContract` | Interest-rate swap |

---

## 1. Basic Lattice Structure

All of the models inherit from the `Lattice` class. A lattice with `n` periods which contains `n + 1` possible states at maturity. The individual nodes are identified a `period_index`, the time period starting from zero, and an `outcome_index`, the number of downward movements. For example, a three-period lattice may look like:

```text
period 0:       100

period 1:       110          90

period 2:       121          99          81

period 3:       133.10       108.90       89.10       72.90
```

The value at a particular node can be extracted using the `Lattive.getValue()` function, which takes the period index and outcome index as its arguments. All values for a particular period can be retrieved using the `Lattice.getValues()` function, which takes a single argument, `period_index`, and returns a NumPy array.

```python
lattice.getValue(period_index, outcome_index)
lattice.getValues(period_index)
```

The whole lattice can pe printed using the `Lattice.printLattice()` member function. These functions can used with any lattice-type object used here.

## 2. Constructing an Underlying Asset Lattice

The `BinomialLattice` class creates a multiplicative binomial tree, with specified root node and size. The values at each node are determined by its parent node and up/down factors. If the up-factor is $k_u$ and the value at the (i,j)-th node is $a$, then the value at the $(i+1,j+1)$-th node will be $a k_u$. Similarly for a down-factor $k_d$, the value in the $(i+1,j)$-th node is $a k_d$. To create a binomial lattice, use the `initial_value` argument to set the value of the root node, the `up_move`/`down_move` arguments to set the up- and down-factors, and the `num_periods` argument to determine the size of the lattice.

```python
asset_lattice = BinomialLattice(
    initial_value=100,
    up_move=1.10,
    down_move=0.90,
    num_periods=3
)

asset_lattice.printLattice()
```

## 3. Interest-Rate Lattices

An interest rates lattice can be constructed using the same arguments as the binomial lattice. Importantly, the interest rate is given as decimal, rather than a percentage, hence a value of 0.05 in the lattice is equivalent to a 5% interest rate. The class also calculates the discount factor, used in binomial modelling, the discount rate at the $(i,j)$-th node can be found using the `ShortTermRates.getDiscount()` function, whose arguments are the period and outcome index, similarly to the `Lattice.getValue()` function.

```python
rates = ShortTermRates(
    initial_rate=0.05,
    up_move=1.001,
    down_move=0.009,
    num_periods=3
)
```

A fixed interest rates lattive can be created by setting `up_move=down_move=1.0`, which may be useful in the construction of swaps or bonds later on.

## 4. Options

Two simple option types are provided here: European and American. These differ in one key way, that an American option can be executed at any time until the expiration period, while a European option can only be executed at maturity. They are both defined by a strike price, the agreed upon price to buy/sell the option, and a maturity time.  

### 4.1. European option

A European option can be created using the `EuropeanOption` class, and has three argumentes in its construcotr: `strike_price`, `time_to_maturity` and `is_call`. The final argument, `is_call`, determines wether the option is a call or put i.e. whether the owner agrees to buy or sell the option, respectively.

```python
call = EuropeanOption(
    strike_price = 100,
    time_to_maturity = 1.0,
    is_call = True
)
```

#### 4.1.1. Pricing a European option

`OptionPricer` combines an asset lattice, an interest-rate specification and an option. For a constant risk-free rate, 0.05, a three-period binomial model can be created as follows. 

```python
pricer = OptionPricer(
    asset_lattice=stock,
    interest_rate=0.05,
    num_periods=3
)
```

The option price, under the model, can be found using the `OptionPrice.fairPrice()` function which takes an option type, `EuropeanOption` or `AmericanOption`, as an argument. The same could be done with binomial model for a non-constant interest rate, by first creating a `ShortTermRates` object and passing it to the `OptionPricer` constructor via the `interest_rate` argument. A full example:

```python
stock = BinomialLattice(
    initial_value=100,
    up_move=1.10,
    down_move=0.90,
    num_periods=3
)

call = EuropeanOption(
    strike_price=100,
    time_to_maturity=3,
    is_call=True
)

pricer = OptionPricer(
    asset_lattice=stock,
    interest_rate=0.05,
    num_periods=3
)


price = pricer.fairPrice(call)

print(price)
```

Outputs the fair price of the option, found as the root value in `OptionPricer` lattice, which can also be inspected through the `OptionPricer.printLattice()` member function. The terminal nodes contain the option payoffs. The earlier nodes contain the discounted risk-neutral expected values obtained through backward induction at each period.

## 4.2. American options

American options are created using `AmericanOption` class, which acts in a similar way to the `EuropeanOption` class. The payoff is calculated differently to account for the ability to execute at before maturation.

```python
put = AmericanOption(
    strike=100,
    maturity=3,
    is_call=False
)
```

## 5. Bonds

### 5.1. Pricing a zero-coupon bond

A zero-coupon bond pays no divdends, only the payoff at maturity. A bond lattice can be constructed by backwards induction by starting at the payoff at the final period. The class `ZeroCouponBond` creates such a lattice using an interest rates lattice and notional value. 

```python
rates = ShortTermRates(
    initial_value=0.05, 
    up_move=1.001, 
    down_move=0.999,
    maturity=10
)
zero_coupon_bond = ZeroCouponBondLattice(
    notional_value=100
    rates=rates
)
zero_coupon_bond.getFairPrice()
```

This calculates $P(0,T) = N\sum_{s=0}^{T} \pi_{T,s}$, where $N$ is the notional value and $\pi_{T,s}$ are the state prices at maturity.

### 5.2. Coupon-paying bonds

`BondLattice` prices a coupon-paying bond. For example:

```python
bond = BondLattice(
    move_up_prob=0.5,
    interest_rates=rates,
    num_periods=3,
    coupon_rate=0.05
)
```

By default, coupons are paid at every period. A bond with a face value of 100 can be priced using:

```python
bond.getFairPrice(face_value=100)
```

The underlying lattice is expressed relative to a face value of 1, so:

```python
bond.getValue(0, 0)
```

is the price per unit of face value, while:

```python
bond.getFairPrice(100)
```

gives the price for a £100 notional.

You can also scale the entire lattice:

```python
bond.setFaceValue(100)
```

### 5.3. Specifying coupon dates

Coupon payments can be restricted to particular periods using `coupon_periods`. For example, to pay coupons at periods 2 and 4:

```python
bond = BondLattice(
    move_up_prob=0.5,
    interest_rates=rates,
    num_periods=4,
    coupon_rate=0.05,
    coupon_periods={2, 4}
)
```

The set should contain the periods at which coupons are paid.

---

### 5.4. Defaultable bonds

Credit risk can be incorporated using a `HazardLattice`. A hazard lattice contains the probability of default associated with each node.F or example, a constant 2% hazard rate can be constructed with:

```python
hazard_lattice = HazardLattice(
    num_periods=3,
    hazard_function = lambda period, outcome: 0.02
)
```

The hazard function receives period and outcome index, and any hazard-specific parameters, and so the default probability can depend on both time and the state of the interest-rate lattice. A recovery rate is available as well, which returns a percentage of the notional value if a default occurs.

For example the function:

```python
def hazard(period, outcome, a = 0.01, b=0.005):
    return a + b * outcome
```

can be used to create a state-dependent hazard rate. A defaultable bond is then constructed using:

```python
bond = BondLattice(
    move_up_prob=0.5,
    interest_rates=rates,
    num_periods=3,
    coupon_rate=0.05,
    hazard_lattice=hazard_lattice,
    recovery_rate=0.40
)
```

At each node the bond value incorporates:

1. survival to the next period
2. the value of the surviving bond
3. recovery in the event of default
4. discounting
5. coupon payments

Finally, the recovery rate is expressed as a fraction of face value, so `0.40` represents 40% recovery.

## 6. Futures Contracts

`FutureContract` can be used to construct a futures contract on a bond lattice. For example:

```python
future = FutureContract(
    bond_lattice=bond,
    num_periods=2
)
```

The futures contract takes the bond value at the contract's maturity and works backwards using the model's 50/50 up/down probabilities. The futures price is available as:

```python
future.fairPrice 
# or
future.getFairPrice()
```

The underlying bond lattice can also be accessed:

```python
future.bondLattice
```

## 7. Forward Contracts

`ForwardContract` is intended to price a forward contract on a bond using an interest-rate lattice. Unlike a futures contract, the forward price is calculated by discounting the expected future bond value and discounting by the price of a zero-coupon bond. The required inputs are:

```python
forward = ForwardContract(
    bond_lattice=bond,
    interest_lattice=rates,
    move_up_prob=0.5,
    num_periods=2
)
```

The fair forward price is then:

```python
forward.fairPrice
# or
forward.getFairPrice()
```

## 8. Interest-rate swaps

`SwapContract` represents a fixed-for-floating interest-rate swap. For example:

```python
swap = SwapContract(
    strike_rate=0.05,
    maturity_period=4,
    rates_lattice=rates,
    is_call=True,
    first_payment_period=1
)
```

The `strike_rate` is the fixed rate of the swap and the `is_call` argument determines the direction of the payoff. The swap value for a given notional is obtained using the `getFairPrice()` function, which takes the notional value as an argument.

```python
swap.getFairPrice(notational_value=1_000_000)
```

## 9. Important Conventions

### 9.1. Interest rates

Rates should be supplied in decimal form:

```python
0.05     # 5%
0.025    # 2.5%
```

rather than:

```python
5        # incorrect for 5%
2.5      # incorrect for 2.5%
```

### 9.2. Movement factors

The `up_move` and `down_move` arguments are multiplicative factors:

```python
up_move=1.10
down_move=0.90
```

means that an upward movement increases the asset price by 10% and a downward movement decreases it by 10%.

### 9.3. Periods

`num_periods` determines the number of binomial steps. It is not automatically interpreted as years.

For example, if one year is divided into 12 monthly periods:

```python
num_periods=12
```

but the code itself does not currently use the actual time interval when constructing the asset lattice. The movement factors and rates therefore need to be calibrated consistently with the chosen period length.
