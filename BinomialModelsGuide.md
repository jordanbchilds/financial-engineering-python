# Guide to the Binomial Pricing Models

This module implements several financial models using binomial lattices. The classes are designed to be combined: an underlying asset or interest-rate lattice is first constructed, and derivative or fixed-income instruments are then priced from those lattices.

The main classes are:

| Class | Purpose |
|---|---|
| `Lattice` | Generic binomial lattice |
| `BinomialLattice` | Binomial lattice for an underlying asset |
| `ShortTermRates` | Binomial lattice of short-term interest rates |
| `ElementaryPiceLattice` | State-price lattice |
| `HazardLattice` | Default-probability lattice |
| `BondLattice` | Coupon-paying bond |
| `ZeroCouponBondLattice` | Zero-coupon bond |
| `FutureLattice` | Futures contract on a bond |
| `ForwardLattice` | Forward contract on a bond |
| `SwapLattice` | Interest-rate swap |
| `SwaptionLattice` | Interest-rate swap option |
| `CapletLattice` | Caplet option |
| `FloorletLattice` | Floorlet option |
| `CapOption` | Cap, a series of caplet options |
| `FloorOption` | Floor, a series of floorlet options |
---

## 1. Basic Lattice Structure

All of the models inherit from `Lattice`. A lattice with `n` periods contains `n + 1` possible states at maturity. The individual nodes are identified by:

- `period_index`: the time period, starting from zero
- `outcome_index`: the number of downward movements

For example, in a three-period lattice may look like:

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

The whole lattice can pe printed using the `Lattice.printLattice()` member function.

## 2. Constructing an Underlying Asset Lattice

The `BinomialLattice` class creates a multiplicative binomial tree, with specified root node and size. The values at each node are determined by its parent node and up or down factors. If the up-factor is $k_u$ and the value at the (i,j)-th node is $a$, then the value at the $(i+1,j+1)$-th node is $a k_u$, similarly for the down-factor of $k_d$ the value in the $(i+1,j)$-th node is $a k_d$. To create a binomial lattice, use the `initial_value` argument to set the value of the root node, the `up_move`/`down_move` arguments to set the up- and down-factors, and the `num_periods` argument to determine the size of the lattice.

```python
stock = BinomialLattice(
    initial_value=100,
    up_move=1.10,
    down_move=0.90,
    num_periods=3
)
```

## 3. Interest-Rate Lattices

An interest rates lattice can be constructed using the same arguments as the binomial lattice. The interest rate is given as decimal, rather than a percentage, hence a value of 0.05 in the lattive is equivalent to 5% interest rate. The class also calculates the discount factor, used in binomial modelling, the discount rate at the $(i,j)$-th node can be found using the `ShortTermRates.getDiscount()` function, whose arguments are the period and outcome index, similarly to the `Lattice.getValue()` function.

```python
rates = ShortTermRates(
    initial_rate=0.05,
    up_move=1.001,
    down_move=0.009,
    num_periods=3
)
```

A fixed interest rates lattive can be created by setting `up_move=down_move=1.0`, which may be useful in the contraction of swaps or bonds later on.

## 4. Bonds

### 5.1. State prices

`ElementaryPiceLattice` constructs a lattice of elementary state prices. The state price at a node represents the present value of receiving one unit of currency in that particular state. It is created from a short-term interest-rate lattice:

```python
state_prices = ElementaryPiceLattice(rates)
```

which uses $R(0,T) = \left( \sum_s \pi_{T,s} \right)^{-1/T}-1$.


### 5.2. Pricing a zero-coupon bond

A zero-coupon bond pays no dividends, only the payoff at maturity. A bond lattice can be constructed by backwards induction by starting at the paypoff at the final period. The class `ZeroCouponBond` creates such a lattice using an interest rates lattice and notional value. 

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

### 5.4. Coupon-paying bonds

`BondLattice` prices a coupon-paying bond.

For example:

```python
bond = BondLattice(
    move_up_prob=0.5,
    interest_rates=rates,
    num_periods=3,
    coupon_rate=0.05
)
```

By default, coupons are paid at every period.

A bond with a face value of 100 can be priced using:

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

### 5.4.1. Specifying coupon dates

Coupon payments can be restricted to particular periods using `coupon_periods`.

For example, to pay coupons at periods 2 and 4:

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

### 5.5. Defaultable bonds

Credit risk can be incorporated using `HazardLattice`. A hazard lattice contains the probability of default associated with each node.For example, a constant 2% hazard rate can be constructed with:

```python
hazard_lattice = HazardLattice(
    num_periods=3,
    hazard_function = lambda period, outcome: 0.02
)
```

The hazard function receives:

```python
(period, outcome)
```

so the default probability can depend on both time and the state of the interest-rate lattice.

For example:

```python
def hazard(period, outcome):
    return 0.01 + 0.005 * outcome
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

1. survival to the next period,
2. the value of the surviving bond,
3. recovery in the event of default,
4. discounting,
5. coupon payments.

Finally, the recovery rate is expressed as a fraction of face value, so `0.40` represents 40% recovery.

## 5. Futures Contracts

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
future.getFairPrice()
```

The underlying bond lattice can also be accessed:

```python
future.bondLattice
```

## 6. Forward Contracts

`ForwardContract` is intended to price a forward contract on a bond using an interest-rate lattice. Unlike a futures contract, the forward price is calculated by discounting the expected future bond value and normalising by the price of a zero-coupon bond. The required inputs are:

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
forward.getFairPrice()
```

## 7. Interest-rate swaps

`SwapLattice` represents a fixed-for-floating interest-rate swap. For example:

```python
swap_lattice = SwapLattice(
    fixed_rate=0.05,
    maturity_period=4,
    rates_lattice=rates,
    is_call=True,
    first_payment_period=1
)
```

The `strike_rate` is the fixed rate of the swap and the `is_call` argument determines the direction of the payoff. The swap value for a given notional is obtained using the `getFairPrice()` function, which takes the notional value as an argument.

```python
swap_lattice.getFairPrice(notational_value=1_000_000)
```

### 7.1. Interest-rate swpations

An option can be placed on the interest-rate swap previously defined, the fair value of this can be calculated using the `SwaptionLattice`. For this the a `SwapLattice` must be defined to act as the asset in an option pricing, from this object the short-term interest rates are taken and so the `SwaptionLattice` class does not take an interest rates lattice as an arguemnt. The other important things to define for a swaption are the strike price, maturity time and whether the option is a call or a put. Such a class can be initiated as follows.

```python
swaption = SwaptionLattice(
    strike_price = 0.00, 
    swap_lattice=swap_lattice, 
    maturity_index=3,
    is_call=True
)
```
The default is that the swpation is call option, however the argument was passed to the constructor here for clarity. Similarly to lattices' before it, the fair price can be found through the `fairPrice` method and the pricing lattice can be printed via the `printLattice` method.

## 8. Caps and Floors

### 8.1. Caplets

A Caplet is a type of option similar to a European call option but the short-term interest-rate is treated as the unerlying asset. It can be priced by the same backward induction methods used throughout the pricing models here. The `Caplet` class is defined to do this, which is defined by the `fixed rate`, akin to the strike price, short-term interest rates lattice and maturity. 

```python
caplet_lattice = CapletLattice(
    fixed_rate = 0.03, 
    rates_lattice = rates_lattice,
    maturity_index = 3)
```

### 8.2. Caps

A cap is a portfolio of captlets, defined over a period time to hedge against changes inteerst rates. Their value can be calculated by treating them as independent caplets and summing the fair prices. The `CapOption` class is available to do this, which takes a strike price and interest rate lattice as its arguments. By default the class assumes that a caplet is placed over each period, however this can be altered by using the `maturities` argument which should take an array-like object of maturity periods for the caplets in the portfolio.

```python
my_cap = CapOption(strike_price=0.03, 
                   rates_lattice=rates_lattice)

print(f"The fair prices of the indivudal caplets: {my_cap.fairPriceSeries}")
print(f"The fair price of the cap: {my_cap.fairPrice:.4f}")

```

### 8.3. Floorlets and Floors

A floorlet is "put" version of the a caplet and, similarly, a floor option is a "put" version of the cap, in all other respects they are similar and be priced using the `FloorletLattice` and `FloorOption` classes respectively.


```python
my_floor = FloorOption(strike_price=0.03, 
                   rates_lattice=rates_lattice)
print(f"The fair prices of the indivudal flootlets: {my_floor.fairPriceSeries}")
print(f"The fair price of the floot: {my_floor.fairPrice:.4f}")
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