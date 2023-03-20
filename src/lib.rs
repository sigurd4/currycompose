#![feature(tuple_trait)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]

//! https://en.wikipedia.org/wiki/Function_composition
//! 
//! A crate providing a trait for performing currying (and non-currying) function-composition in rust.
//! 
//! Non-currying composition:
//! h(x) = g ∘ f = g(f(x))
//! 
//! Currying composition:
//! h(..., x) = g ∘ f = g(f(x), ...)
//! 
//! When currying, arguments of the function being curried with (f) is moved to the end of the argument-list
//! 
//! Both operands must implement FnOnce. If both implement FnMut or Fn, the resulting composition will also implement these traits.
//! 
//! g must also have one or more argument, where the first argument type equals the return type of f.
//! 
//! ```rust
//! #![feature(generic_const_exprs)]
//! 
//! use currycompose::*;
//! 
//! // g ∘ f
//! // where
//! // g :: f32 -> f32
//! // f :: u8 -> f32
//! // g ∘ f :: u8 -> f32
//! let g = |x: f32| x*x;
//! let f = |x: u8| x as f32;
//! 
//! let gf = g.compose(f);
//! 
//! let x = 1;
//! 
//! assert_eq!(gf(x), g(f(x)));
//! 
//! // g ∘ f
//! // where
//! // g :: f32 -> f32 -> f32
//! // f :: u8 -> f32
//! // g ∘ f :: f32 -> u8 -> f32
//! let g = |x: f32, y: f32| x + y;
//! let f = gf;
//! 
//! let gf = g.compose(f);
//! 
//! let x = 1;
//! let y = 1.0;
//! 
//! // note here the argument x has been shifted to the end of the args in gf
//! assert_eq!(gf(y, x), g(f(x), y));
//! 
//! // g ∘ f ∘ f
//! // where
//! // g :: f32 -> f32 -> f32
//! // f :: u8 -> f32
//! // g ∘ f ∘ f :: u8 -> u8 -> f32
//! let gff = gf.compose(f);
//! 
//! let x = 1;
//! let y = 1;
//! 
//! assert_eq!(gff(x, y), g(f(x), f(y)));
//! ```

use std::marker::{Tuple, PhantomData};

use tupleops::{ConcatTuples, TupleConcat, concat_tuples, TupleLength, TupleUnprepend, Head, Tail};

use tuple_split::{TupleSplit, SplitInto};

/// https://en.wikipedia.org/wiki/Function_composition
/// 
/// Trait for composing two functions (currying and non-curring composition)
/// 
/// Non-currying composition:
/// h(x) = g ∘ f = g(f(x))
/// 
/// Currying composition:
/// h(..., x) = g ∘ f = g(f(x), ...)
/// 
/// When currying, arguments of the function being curried with (f) is moved to the end of the argument-list
/// 
/// Both operands must implement FnOnce. If both implement FnMut or Fn, the resulting composition will also implement these traits.
/// 
/// g must also have one or more argument, where the first argument type equals the return type of f.
/// 
/// ```rust
/// #![feature(generic_const_exprs)]
/// 
/// use currycompose::*;
/// 
/// // g ∘ f
/// // where
/// // g :: f32 -> f32
/// // f :: u8 -> f32
/// // g ∘ f :: u8 -> f32
/// let g = |x: f32| x*x;
/// let f = |x: u8| x as f32;
/// 
/// let gf = g.compose(f);
/// 
/// let x = 1;
/// 
/// assert_eq!(gf(x), g(f(x)));
/// 
/// // g ∘ f
/// // where
/// // g :: f32 -> f32 -> f32
/// // f :: u8 -> f32
/// // g ∘ f :: f32 -> u8 -> f32
/// let g = |x: f32, y: f32| x + y;
/// let f = gf;
/// 
/// let gf = g.compose(f);
/// 
/// let x = 1;
/// let y = 1.0;
/// 
/// // note here the argument x has been shifted to the end of the args in gf
/// assert_eq!(gf(y, x), g(f(x), y));
/// 
/// // g ∘ f ∘ f
/// // where
/// // g :: f32 -> f32 -> f32
/// // f :: u8 -> f32
/// // g ∘ f ∘ f :: u8 -> u8 -> f32
/// let gff = gf.compose(f);
/// 
/// let x = 1;
/// let y = 1;
/// 
/// assert_eq!(gff(x, y), g(f(x), f(y)));
/// ```
#[const_trait]
pub trait Compose<F, XG, XF>: Sized
{
    /// Composing two functions
    /// 
    /// h(x) = g ∘ f = g(f(x))
    fn compose(self, with: F) -> Composition<Self, F, XG, XF>;
}

impl<G, F, XG, XF> const Compose<F, XG, XF> for G
where
    XG: Tuple + TupleUnprepend<XG>,
    XF: Tuple,
    Self: FnOnce<XG>,
    F: FnOnce<XF, Output = Head<XG>>,
    (Tail<XG>, XF): TupleConcat<Tail<XG>, XF>,
    ConcatTuples<Tail<XG>, XF>: Tuple,
    Composition<Self, F, XG, XF>: FnOnce<ConcatTuples<Tail<XG>, XF>>
{
    fn compose(self, with: F) -> Composition<Self, F, XG, XF>
    {
        Composition {
            g: self,
            f: with,
            phantom: PhantomData
        }
    }
}

/// A struct representing a function composed with another.
/// 
/// When calling the composition as a function, the leftover arguments of the composition function come first (if curried), then the arguments of the function being composed with.
/// 
/// ```rust
/// #![feature(generic_const_exprs)]
/// 
/// use currycompose::*;
/// 
/// // g ∘ f
/// // where
/// // g :: f32 -> f32
/// // f :: u8 -> f32
/// // g ∘ f :: u8 -> f32
/// let g = |x: f32| x*x;
/// let f = |x: u8| x as f32;
/// 
/// let gf = g.compose(f);
/// 
/// let x = 1;
/// 
/// assert_eq!(gf(x), g(f(x)));
/// 
/// // g ∘ f
/// // where
/// // g :: f32 -> f32 -> f32
/// // f :: u8 -> f32
/// // g ∘ f :: f32 -> u8 -> f32
/// let g = |x: f32, y: f32| x + y;
/// let f = gf;
/// 
/// let gf = g.compose(f);
/// 
/// let x = 1;
/// let y = 1.0;
/// 
/// // note here the argument x has been shifted to the end of the args in gf
/// assert_eq!(gf(y, x), g(f(x), y));
/// 
/// // g ∘ f ∘ f
/// // where
/// // g :: f32 -> f32 -> f32
/// // f :: u8 -> f32
/// // g ∘ f ∘ f :: u8 -> u8 -> f32
/// let gff = gf.compose(f);
/// 
/// let x = 1;
/// let y = 1;
/// 
/// assert_eq!(gff(x, y), g(f(x), f(y)));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Composition<G, F, XG, XF>
{
    g: G,
    f: F,
    phantom: PhantomData<(XG, XF)>,
}

impl<G, F, XG, XF> FnOnce<ConcatTuples<Tail<XG>, XF>> for Composition<G, F, XG, XF>
where
    XG: Tuple + TupleUnprepend<XG>,
    XF: Tuple,
    G: FnOnce<XG>,
    F: FnOnce<XF, Output = Head<XG>>,
    (Tail<XG>, XF): TupleConcat<Tail<XG>, XF>,
    ConcatTuples<Tail<XG>, XF>: Tuple + SplitInto<Tail<XG>, XF>,
    [(); <Tail<XG> as TupleLength>::LENGTH]:,
    ((F::Output,), Tail<XG>): TupleConcat<(F::Output,), Tail<XG>, Type = XG>
{
    type Output = <G as FnOnce<XG>>::Output;

    extern "rust-call" fn call_once(self, args: ConcatTuples<Tail<XG>, XF>) -> Self::Output
    {
        let (left, right): (Tail<XG>, XF) = args.split_tuple();
        self.g.call_once(concat_tuples((self.f.call_once(right),), left))
    }
}

impl<G, F, XG, XF> FnMut<ConcatTuples<Tail<XG>, XF>> for Composition<G, F, XG, XF>
where
    XG: Tuple + TupleUnprepend<XG>,
    XF: Tuple,
    G: FnMut<XG>,
    F: FnMut<XF, Output = Head<XG>>,
    (Tail<XG>, XF): TupleConcat<Tail<XG>, XF>,
    ConcatTuples<Tail<XG>, XF>: Tuple + SplitInto<Tail<XG>, XF>,
    [(); <Tail<XG> as TupleLength>::LENGTH]:,
    ((F::Output,), Tail<XG>): TupleConcat<(F::Output,), Tail<XG>, Type = XG>
{
    extern "rust-call" fn call_mut(&mut self, args: ConcatTuples<Tail<XG>, XF>) -> Self::Output
    {
        let (left, right) = args.split_tuple();
        self.g.call_mut(concat_tuples((self.f.call_mut(right),), left))
    }
}

impl<G, F, XG, XF> Fn<ConcatTuples<Tail<XG>, XF>> for Composition<G, F, XG, XF>
where
    XG: Tuple + TupleUnprepend<XG>,
    XF: Tuple,
    G: Fn<XG>,
    F: Fn<XF, Output = Head<XG>>,
    (Tail<XG>, XF): TupleConcat<Tail<XG>, XF>,
    ConcatTuples<Tail<XG>, XF>: Tuple + SplitInto<Tail<XG>, XF>,
    [(); <Tail<XG> as TupleLength>::LENGTH]:,
    ((F::Output,), Tail<XG>): TupleConcat<(F::Output,), Tail<XG>, Type = XG>
{
    extern "rust-call" fn call(&self, args: ConcatTuples<Tail<XG>, XF>) -> Self::Output
    {
        let (left, right) = args.split_tuple();
        self.g.call(concat_tuples((self.f.call(right),), left))
    }
}