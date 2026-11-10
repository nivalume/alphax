//! Margin columns: what a position adds to the group's value, and settling it.
//!
//! A margin column holds no asset the way a cash column does. Its `debt` is
//! the position's cost basis, `|position|` times its average entry price, and
//! what the position is worth to the account is only its profit against that
//! basis. Every bar, once the valuation price is known, the loops settle the
//! profit into the group's cash and move the basis to the valuation price, as
//! an exchange marks a futures position to market. So the group's free cash
//! counts every margin position's profit or loss, which is what the margin a
//! new order may post comes out of.
//!
//! Settling is not a cash flow: it moves money between the position and the
//! cash without changing what the account is worth, so it leaves no record.
//! The analysis layer values a margin column from its fills as it values any
//! other, `cash + position × price` with the fills charged at full price, and
//! gets the same equity.

/// The signed cost basis of a margin position: what is subtracted from
/// `position × price` to leave its profit. Zero when flat.
#[inline]
pub(crate) fn owed(position: f64, debt: f64) -> f64 {
    if position > 0.0 {
        debt
    } else if position < 0.0 {
        -debt
    } else {
        0.0
    }
}

/// The profit a margin position has made since it was last settled, valued at
/// `val_price`: what it adds to the group's value. Zero when flat.
#[inline]
pub(crate) fn unsettled(position: f64, debt: f64, val_price: f64) -> f64 {
    if position == 0.0 {
        0.0
    } else {
        position * val_price - owed(position, debt)
    }
}

/// Settles a margin position's profit since it was last settled into `cash`
/// and `free_cash`, and moves its cost basis to `val_price`.
///
/// Skipped when flat, and when the valuation price is `NaN`: the profit is
/// then unknown, and is settled at the next price that is not.
#[inline]
pub(crate) fn settle(
    position: f64,
    debt: &mut f64,
    val_price: f64,
    cash: &mut f64,
    free_cash: &mut f64,
) {
    if position == 0.0 || val_price.is_nan() {
        return;
    }
    let pnl = unsettled(position, *debt, val_price);
    *cash += pnl;
    *free_cash += pnl;
    *debt = position.abs() * val_price;
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "every value here is exact in binary")]
mod tests {
    use super::*;

    #[test]
    fn settling_moves_the_profit_into_cash_and_the_basis_to_the_price() {
        // Long 2 bought at 100, now 110: +20.
        let (mut debt, mut cash, mut free) = (200.0, 50.0, 30.0);
        assert_eq!(unsettled(2.0, debt, 110.0), 20.0);
        settle(2.0, &mut debt, 110.0, &mut cash, &mut free);
        assert_eq!((debt, cash, free), (220.0, 70.0, 50.0));
        assert_eq!(unsettled(2.0, debt, 110.0), 0.0);

        // Short 2 sold at 100, now 110: -20.
        let (mut debt, mut cash, mut free) = (200.0, 50.0, 30.0);
        assert_eq!(unsettled(-2.0, debt, 110.0), -20.0);
        settle(-2.0, &mut debt, 110.0, &mut cash, &mut free);
        assert_eq!((debt, cash, free), (220.0, 30.0, 10.0));
    }

    #[test]
    fn nothing_settles_when_flat_or_unpriced() {
        let (mut debt, mut cash, mut free) = (200.0, 50.0, 30.0);
        settle(2.0, &mut debt, f64::NAN, &mut cash, &mut free);
        settle(0.0, &mut debt, 110.0, &mut cash, &mut free);
        assert_eq!((debt, cash, free), (200.0, 50.0, 30.0));
        assert_eq!(unsettled(0.0, 0.0, f64::NAN), 0.0);
    }
}
