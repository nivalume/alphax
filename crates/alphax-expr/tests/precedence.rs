//! Operator precedence follows mathematics, not a programming language:
//!
//! | binds | operators | notes |
//! |---|---|---|
//! | tightest | `^` (`**`) | right to left: `2^3^2 = 2^9` |
//! | | unary `-` `+` | below the power: `-2^2 = -4`; `2^-1 = 0.5` |
//! | | `*` `/` | left to right |
//! | | `+` `-` | left to right |
//! | | `<` `<=` `>` `>=` `==` `!=` | chained as in mathematics: `a < b < c` is `a < b` and `b < c` |
//! | | `&` | conjunction before disjunction |
//! | loosest | `\|` | |

use alphax_expr::{Expression, FactorInputs, FactorSet};

fn tree(source: &str) -> Expression {
    let set = FactorSet::compile([("f", source)]).unwrap();
    set.expressions().next().unwrap().1.clone()
}

fn value(source: &str, x: &[f64]) -> Vec<f64> {
    let inputs = FactorInputs::new(x.len(), 1)
        .unwrap()
        .with_field("x", x)
        .unwrap();
    let set = FactorSet::compile([("f", source)]).unwrap();
    set.evaluate(&inputs)
        .unwrap()
        .get("f")
        .unwrap()
        .as_slice()
        .to_vec()
}

#[test]
fn powers_bind_tightest_and_from_the_right() {
    assert_eq!(value("-2^2 + 0*$x", &[0.0]), [-4.0]);
    assert_eq!(value("2^3^2 + 0*$x", &[0.0]), [512.0]);
    assert_eq!(value("2^-1 + 0*$x", &[0.0]), [0.5]);
    assert_eq!(tree("$x^2"), tree("$x**2"));
    assert_eq!(value("12 / 3 * 2 - 1 - 1 + 0*$x", &[0.0]), [6.0]);
}

#[test]
fn comparisons_chain_as_in_mathematics() {
    assert_eq!(tree("1 < $x < 3"), tree("(1 < $x) & ($x < 3)"));
    assert_eq!(
        tree("1 < $x + 1 <= 3"),
        tree("(1 < $x + 1) & ($x + 1 <= 3)")
    );
    assert_eq!(tree("$x == 1 == 1"), tree("($x == 1) & (1 == 1)"));
    // Python's `a < b > c` reading, too: each neighbouring pair.
    assert_eq!(tree("0 < $x > 2"), tree("(0 < $x) & ($x > 2)"));
    assert_eq!(
        value("1 < $x < 3", &[0.0, 1.0, 2.0, 3.0]),
        [0.0, 0.0, 1.0, 0.0]
    );
    // Brackets still group: a comparison of a comparison.
    assert_eq!(value("(1 < $x) < 3", &[0.0, 2.0]), [1.0, 1.0]);
}

#[test]
fn relations_bind_tighter_than_conjunction_and_conjunction_than_disjunction() {
    assert_eq!(tree("$x > 1 & $x < 3"), tree("($x > 1) & ($x < 3)"));
    assert_eq!(
        tree("$x > 3 | $x > 1 & $x < 2"),
        tree("($x > 3) | (($x > 1) & ($x < 2))")
    );
    assert_eq!(
        tree("$x < 0 & 1 < $x < 2"),
        tree("($x < 0) & ((1 < $x) & ($x < 2))")
    );
    assert_eq!(value("$x > 1 & $x < 3", &[1.0, 2.0, 3.0]), [0.0, 1.0, 0.0]);
}

#[test]
fn literal_arguments_may_be_arithmetic() {
    assert_eq!(
        value("Mean($x, 2*2) - Mean($x, 4)", &[1.0, 2.0, 3.0, 4.0, 5.0]),
        [0.0; 5]
    );
    let future = FactorSet::compile([("f", "Ref($x, 1-2)")]).unwrap();
    assert!(future.uses_future());
    assert!(FactorSet::compile([("f", "Mean($x, $x)")]).is_err());
}

#[test]
fn true_false_and_unicode_fields_read_as_in_qlib() {
    assert_eq!(value("If($x > 1, True, False)", &[0.0, 2.0]), [0.0, 1.0]);
    let close = [1.0, 2.0];
    let inputs = FactorInputs::new(2, 1)
        .unwrap()
        .with_field("收盘价", &close)
        .unwrap()
        .with_field("成交量（手）", &close)
        .unwrap();
    let set = FactorSet::compile([("f", "$收盘价 * $成交量（手）")]).unwrap();
    assert_eq!(
        set.evaluate(&inputs).unwrap().get("f").unwrap().as_slice(),
        [1.0, 4.0]
    );
}
