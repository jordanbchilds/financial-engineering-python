# Guide to the Binomial Pricing Models

This module implements financial models using binomial lattices. The classes are designed to be combined: an underlying asset or interest-rate lattice is constructed first, and derivative or fixed-income instruments are then priced from those lattices. The current class structure is:

| Class | Purpose |
|---|---|
| `Lattice` | Generic binomial lattice |
| `BinomialLattice` | Multiplicative binomial lattice for an underlying asset |
| `AssetLattice` | Base class intended for dividend-aware asset lattices |
| `RatesLattice` | Generic interest-rate lattice |
| `ShortTermRates` | Binomial lattice of short-term interest rates |
| `PayoffLattice` | Base class for instruments priced by backward induction |
| `Option` | Base class for European and American options |
| `EuropeanOption` | European call or put payoff |
| `AmericanOption` | American call or put payoff |
| `OptionPricer` | Prices `EuropeanOption` and `AmericanOption` objects |
| `EuropeanOptionLattice` | European option priced directly on an asset/rate lattice |
| `ElementaryPiceLattice` | Elementary state-price lattice |
| `HazardLattice` | Default-probability lattice |
| `BondLattice` | Coupon-paying bond |
| `ZeroCouponBondLattice` | Zero-coupon bond |
| `FutureContract` | Futures contract on a bond |
| `ForwardContract` | Forward contract on a bond |
| `CapletLattice` | Interest-rate caplet |
| `FloorletLattice` | Interest-rate floorlet |
| `CapOption` | Portfolio of caplets forming an interest-rate cap |
| `FloorOption` | Portfolio of floorlets forming an interest-rate floor |
| `SwapContract` | Fixed-for-floating interest-rate swap |


## 1. Basic Lattice Structure

All lattice classes inherit from `Lattice`. A lattice with `n` periods contains `n + 1` possible states at maturity. Nodes are identified by their `period_index`, the time period starting from zero, and thier `outcome_index`, the number of downward movements. For example, a three-period lattice can be represented as:

```text
period 0:       100

period 1:       110          90

period 2:       121          99          81

period 3:       133.10       108.90       89.10       72.90
```

The value at a particular node can be retrieved with the `Lattice.getValue(period_index, outcome_index)` member. All values at a particular period, `period_index`, can be retrieved with `Lattice.getValues(period_index)`. The complete lattice can be displayed using:

```python
Lattice.printLattice()
```

The lattice istelf can also be accessed directly through the `lattice` property, `Lattice.lattice`, although this is not recommended.


## 2. Underlying Asset Lattices

### 2.1. `BinomialLattice`

`BinomialLattice` creates a multiplicative binomial tree. The value at a node is determined by the initial value and the number of up and down movements. If the up factor is $u$ and the down factor is $d$, $ S_{i,j} = S_0 u^{i-j}d^j, $ where $i$ is the period and $j$ is the number of down movements. The object constructor works as follows,

```python
stock = BinomialLattice(
    initial_value=100,
    up_move=1.10,
    down_move=0.90,
    num_periods=3
)
```

The main properties of the lattice can be retrieved with the following members:

```python
stock.initialValue
stock.upMove
stock.downMove
stock.numPeriods
```

## 3. Interest-Rate Lattices

### 3.1. `RatesLattice`

`RatesLattice` is the base class for interest-rate lattices. In addition to the normal lattice functionality, it stores the length of each period in years, RatesLattice.periodDuration and calculateds the one-period discount factor at a node is obtained with the member `RatesLattice.getDiscount(period_index, outcome_index)`, which corresponds to

$$ D_{i,j}=\frac{1}{1+r_{i,j}}. $$

### 3.2. `ShortTermRates`

`ShortTermRates` creates a multiplicative lattice of short-term interest rates and is constructed in the same manor as binomial lattice. For example

```python
rates = ShortTermRates(
    initial_rate=0.05,
    up_move=1.001,
    down_move=0.999,
    num_periods=3
)
```

Rates are supplied in decimal form, such that an initial value of 0.05 is equivalent to 5% interest rate. The rate at each node is constructed from the initial rate and the up/down factors. A constant-rate lattice can be created by setting the up and down moves to equal 1.0.

## 4. Payoff Lattices

`PayoffLattice` is the base class for instruments whose value is calculated using backward induction. It can contain:

- an underlying asset lattice,
- a short-term rate lattice,
- a strike price,
- a maturity index.

The main properties are:

```python
PayoffLattice.assetLattice
PayoffLattice.ratesLattice
PayoffLattice.strikePrice
PayoffLattice.maturityIndex
PayoffLattice.fairPrice
```

The fair price is the value at the root node, `PayoffLattice.fairPrice`, or for a specified number of units/notional, `PayoffLattice.getFairPrice(number_of_options)` can be used.

## 5. `EuropeanOptionLattice`

`EuropeanOptionLattice` directly prices a European option using an underlying `BinomialLattice` with either a constant interest rate, or a `ShortTermRates` Lattice. The option is priced assuming the underlying asset follows a binomial model, which is passed to the objects constructor. For example:

```python
option = EuropeanOptionLattice(
    strike_price=100,
    asset_lattice=stock,
    rates_lattice=0.05,
    maturity_index=3,
    call=True
)
```

The fair price can accessed following the same interface as the `PayoffLattice` object i.e. using `EuropeanOptionLattice.fairPrice` and `EuropeanOptionLattice.fairPrice()`. If using a dynamic short-term interest rate can be described by a `ShortTermRate` object and passed to the `rates_lattice` argument of the constructor.

```python
option = EuropeanOptionLattice(
    strike_price=100,
    asset_lattice=stock,
    rates_lattice=rates,
    maturity_index=3,
    call=True
)
```

### 7.1. Dividend yields

`EuropeanOptionLattice` accepts a proportional dividend yield through `dividend_rate`. A constant dividend yield can be supplied as a scalar or a different yield can be given for each period as an array-like object.

```python
option = EuropeanOptionLattice(
    strike_price=100,
    asset_lattice=stock,
    rates_lattice=0.05,
    maturity_index=3,
    call=True,
    dividend_rate=0.02
)
```

The dividend payment periods are specified using `dividend_periods`:

```python
option = EuropeanOptionLattice(
    strike_price=100,
    asset_lattice=stock,
    rates_lattice=0.05,
    maturity_index=3,
    call=True,
    dividend_rate=0.02,
    dividend_periods=[1, 3]
)
```

The current implementation interprets a dividend period as the period at which the dividend is paid. Dividend yields are associated with the corresponding transition in the pricing recursion.

### 7.2. Separate cashflows

The `separate_cashflows` argument determines how the dividend treatment is handled by the option lattice, the default is True.The setting is available by the `EuropeanOptionLattice.separateCashflows` member. When `True`, the dividend yield is treated separately from the underlying price return when calculating the risk-neutral probability. When `False`, the pricing recursion uses the underlying up/down movements without the separate dividend adjustment. Finally the option can be made a put/call using the `is_call` argument, the default is True. For example:

```python
option = EuropeanOptionLattice(
    strike_price=100,
    asset_lattice=stock,
    rates_lattice=rates,
    maturity_index=3,
    is_call=True,
    dividend_rate=0.02,
    dividend_periods=[1, 3],
    separate_cashflows=True
)
```

## 8. State Prices

`ElementaryPiceLattice` constructs an elementary state-price lattice from a the short-term rates, `ElementaryPiceLattice(rates_lattice)`. The state price at a node represents the present value of one unit of currency received in that state. The class can calculate zero-coupon bond prices:

```python
state_prices.zeroCouponBondPrice(
    notational_value=100,
    maturity_time=3
)
```

and implied spot rates:

```python
state_prices.getSpotRate(3)
```

The zero-coupon bond price is calculated from the sum of the state prices at maturity, $ P(0,T)=N\sum_s \pi_{T,s}$, where $N$ is the notional value and $\pi_{T,s}$ are the state prices.

## 9. Hazard Lattices

`HazardLattice` represents the probability of default at each node. A constant hazard rate can be constructed using a function:

```python
hazard_lattice = HazardLattice(
    num_periods=3,
    hazard_function=lambda period, outcome: 0.02
)
```

The hazard function requires two parameters: `period_index` and `income_index`, so the probability can depend on both time and the state of the Lattice. For example:

```python
def hazard(period, outcome):
    return 0.01 + 0.005 * outcome

hazard_lattice = HazardLattice(
    num_periods=3,
    hazard_function=hazard
)
```

## 10. Bonds

### 10.1. `BondLattice`

`BondLattice` prices a coupon-paying bond using a short-term interest-rate Lattice. A basic bond can be constructed with:

```python
bond = BondLattice(
    move_up_prob=0.5,
    rates_lattice=rates,
    maturity_index=3,
    coupon_rate=0.05
)
```

By default, coupons are paid at every period. The bond value is normalised to one unit of face value. Therefore `BondLattice.fairPrice` is the value per unit of face value. The lattice can be scaled to a particular face value using:

```python
bond.setFaceValue(100)
```

### 10.2. Coupon payment dates

Coupon payments can be restricted to specified periods using the `coupon_periods` argument, on which the coupon is paid.

```python
bond = BondLattice(
    move_up_prob=0.5,
    rates_lattice=rates,
    maturity_index=4,
    coupon_rate=0.05,
    coupon_periods=[2, 4]
)
```

### 10.3. Defaultable bonds

A `HazardLattice` can be supplied to incorporate default risk:

```python
hazard = HazardLattice(
    num_periods=3,
    hazard_function=lambda period, outcome: 0.02
)

bond = BondLattice(
    move_up_prob=0.5,
    rates_lattice=rates,
    maturity_index=3,
    coupon_rate=0.05,
    hazard_lattice=hazard,
    recovery_rate=0.40
)
```

The recovery rate is expressed as a fraction of face value, and so `recovery_rate=0.40` tepresents 40% recovery. The backward induction incorporates survival, default recovery, discounting and coupon payments.

### 10.4. `ZeroCouponBondLattice`

`ZeroCouponBondLattice` is a specialised `BondLattice` with no coupon payments, and can also incorporate a hazard lattice and recovery rate.

```python
zero_coupon_bond = ZeroCouponBondLattice(
    move_up_prob=0.5,
    rates_lattice=rates,
    num_periods=3
)
```

## 11. Futures Contracts

`FutureContract` represents a futures contract on a bond. The contract takes the bond value at the futures maturity and works backwards through the lattice using the bond's movement probabilities. Its fair price is accessable through the `FutureContract.fairPrice` member.

```python
future = FutureContract(
    bond_lattice=bond,
    maturity_index=2
)
```

## 12. Forward Contracts

`ForwardContract` prices a forward contract on a bond using an interest-rate Lattice. The forward price is calculated using the future value of the underlying bond, discounting through the interest-rate lattice and normalising by the corresponding zero-coupon bond price. As before, the fair price can be accessed usual the `fairPrice` member.

```python
forward = ForwardContract(
    bond_lattice=bond,
    rates_lattice=rates,
    move_up_prob=0.5,
    maturity_index=2
)
```

## 13. Caplets and Floorlets

A cap or floor is constructed as a collection of individual caplets or floorlets. The module provides separate lattice classes for these individual instruments.

### 13.1. `CapletLattice`

`CapletLattice` represents a single caplet on the short-term interest rate. The caplet payoff at its relevant maturity is:
$$ \max(r_T-K,0), $$
where $r_T$ is the short-term interest rate and $K$ is the strike rate. One can be constructed by specifying a strike price, maturity index and short-term rates lattice. Additionally, a risk-neutral up probability can optionally be supplied.

```python
caplet = CapletLattice(
    strike_price=0.05,
    rates_lattice=rates,
    maturity_index=3,
     risk_neutral_up_prob=0.5
)
```

### 13.2. `FloorletLattice`

`FloorletLattice` represents a single floorlet, with payoff $ \max(K-r_T,0)$, and can be constructed following the same constructor arguments as the `CapletLattice`.

```python
floorlet = FloorletLattice(
    strike_price=0.05,
    rates_lattice=rates,
    maturity_index=3
)
```

## 14. Caps and Floors

`CapOption` and `FloorOption` aggregate multiple caplets or floorlets. A cap is constructed by specifying:

- a strike rate,
- a short-term rate lattice,
- the maturity/payment periods.

For example:

```python
cap = CapOption(
    strike_price=0.05,
    rates_lattice=rates,
    maturities=[1, 2, 3, 4]
)
```

The cap consists of one caplet for each supplied maturity. The number of caplets is available through:

```python
cap.numberOfCaplets
```

The maturity dates are available through:

```python
cap.maturities
```

The accrual period associated with each caplet is available through:

```python
cap.accrualPeriod
```

The fair value of the cap is:

```python
cap.fairPrice
```

and the value for a specified notional is:

```python
cap.value(1_000_000)
```

### 14.1. Specifying maturity/payment periods

For example:

```python
cap = CapOption(
    strike_price=0.05,
    rates_lattice=rates,
    maturities=[2, 4, 6, 8]
)
```

creates caplets associated with periods 2, 4, 6 and 8.

The `index_maturities` argument controls whether the supplied maturities are interpreted as lattice period indices:

```python
cap = CapOption(
    strike_price=0.05,
    rates_lattice=rates,
    maturities=[2, 4, 6, 8],
    index_maturities=True
)
```

When `index_maturities=True`, the accrual periods are converted into years using `rates.periodDuration`.

### 14.2. Floor options

A floor is constructed similarly:

```python
floor = FloorOption(
    strike_price=0.05,
    rates=rates,
    maturities=[1, 2, 3, 4]
)
```

The floor is the sum of the corresponding floorlets.

The main properties are:

```python
floor.maturities
floor.strikePrice
floor.accrualPeriod
floor.numberOfCaplets
floor.fairPrice
```

The value for a specified notional is:

```python
floor.value(1_000_000)
```

## 15. Interest-Rate Swaps

`SwapContract` represents a fixed-for-floating interest-rate swap.

```python
swap = SwapContract(
    strike_rate=0.05,
    maturity_period=4,
    rates_lattice=rates,
    is_call=True,
    first_payment_period=1
)
```

The `strike_rate` is the fixed rate. The `is_call` argument determines the direction of the swap payoff with a True or False value. The first floating/fixed payment period is controlled by:

```python
first_payment_period
```

The swap's fair value for a given notional is:

```python
swap.getFairPrice(1_000_000)
```

## 16. Typical Workflows

### 16.1. European equity option

```python
stock = BinomialLattice(
    initial_value=100,
    up_move=1.10,
    down_move=0.90,
    num_periods=3
)

option = EuropeanOptionLattice(
    strike_price=100,
    asset_lattice=stock,
    rates_lattice=0.05,
    maturity_index=3,
    call=True
)

print(option.fairPrice)
```

### 16.2. European option with dynamic rates

```python
stock = BinomialLattice(
    initial_value=100,
    up_move=1.10,
    down_move=0.90,
    num_periods=3
)

rates = ShortTermRates(
    initial_rate=0.05,
    up_move=1.001,
    down_move=0.999,
    num_periods=3
)

option = EuropeanOptionLattice(
    strike_price=100,
    asset_lattice=stock,
    rates_lattice=rates,
    maturity_index=3,
    call=True
)

print(option.fairPrice)
```

### 16.3. European option with dividends

```python
option = EuropeanOptionLattice(
    strike_price=100,
    asset_lattice=stock,
    rates_lattice=rates,
    maturity_index=3,
    call=True,
    dividend_rate=0.02,
    dividend_periods=[1, 3],
    separate_cashflows=True
)
```

### 16.4. Interest-rate cap

```python
cap = CapOption(
    strike_price=0.05,
    rates_lattice=rates,
    maturities=[1, 2, 3]
)

print(cap.fairPrice)
print(cap.value(1_000_000))
```

### 16.5. Interest-rate floor

```python
floor = FloorOption(
    strike_price=0.05,
    rates=rates,
    maturities=[1, 2, 3]
)

print(floor.fairPrice)
print(floor.value(1_000_000))
```


## 17. Important Conventions

### 17.1. Interest rates

Interest rates are supplied in decimal form:

```python
0.05     # 5%
0.025    # 2.5%
```

not:

```python
5        # incorrect for 5%
2.5      # incorrect for 2.5%
```

### 17.2. Movement factors

The `up_move` and `down_move` arguments are multiplicative factors:

```python
up_move=1.10
down_move=0.90
```

means an upward movement multiplies the value by 1.10 and a downward movement multiplies it by 0.90.

### 17.3. Periods and time

`num_periods` determines the number of binomial steps. It is not automatically interpreted as a number of years.

For example, twelve monthly periods can be represented using:

```python
num_periods=12
```

For interest-rate lattices, the length of each period can be specified with:

```python
period_duration=1/12
```

The movement factors and interest rates should therefore be calibrated consistently with the chosen period length.

### 17.4. Maturity indices

Lattice maturities are represented using integer period indices. For example:

```text
period 0 → initial time
period 1 → first period
period 2 → second period
...
```

For caps and floors, `maturities` specifies the periods associated with the individual caplets/floorlets.

### 17.5. Notional values

Many lattice prices are normalised to one unit of notional. The `value()` or `getFairPrice()` methods can then be used to scale the result to the desired notional:

```python
value = cap.value(1_000_000)
```

## 18. Class Structure Summary

The main inheritance and composition relationships are:

```text
Lattice
├── BinomialLattice
├── AssetLattice
├── RatesLattice
│   └── ShortTermRates
├── ElementaryPiceLattice
├── HazardLattice
├── PayoffLattice
│   ├── BondLattice
│   │   └── ZeroCouponBondLattice
│   ├── FutureContract
│   ├── ForwardContract
│   ├── EuropeanOptionLattice
│   ├── CapletLattice
│   └── FloorletLattice
└── SwapContract

CapOption
FloorOption
```

The most common workflow is therefore:

```text
Underlying/rate lattice
        │
        ├── EuropeanOptionLattice
        ├── BondLattice
        │      └── FutureContract / ForwardContract
        ├── CapletLattice / FloorletLattice
        │      └── CapOption / FloorOption
        └── SwapContract
```

This structure separates the construction of the underlying stochastic lattice from the valuation logic of the financial instrument built on top of it.
