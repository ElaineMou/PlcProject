//! Command Line Argument Parsing (CLAP) module.

// use std::collections::HashMap ;
// use std::fmt ;

use common::Seed ;
use frame::Painter ;

fn default() -> (Seed, Painter,usize,usize) {
  (Seed::mk(), Painter::default(),4,4)
}
fn val_of_next<
  Args: ::std::iter::Iterator<Item = String>,
  T: ::std::str::FromStr
>(args: & mut Args, type_name: & str) -> Result<T, String>
where T::Err: ::std::fmt::Display {
  match args.next() {
    Some(arg) => match T::from_str(& arg) {
      Ok(val) => Ok(val),
      Err(e) => Err(
        format!("could not parse \"{}\" as {}:\n{}", arg, type_name, e)
      ),
    },
    None => Err( format!("expected {}, got nothing", type_name) ),
  }
}

/// Parses command line arguments, returns a seed and a painter.
pub fn parse() -> Result<
  (Seed, Painter, usize, usize), (String, Painter)
> {
  let mut args = ::std::env::args() ;
  // Discard first argument (program name).
  match args.next() {
    Some(_) => (),
    None => unreachable!(),
  } ;

  // Default seed and painter.
  let (mut seed, mut painter, mut depth, mut expect) = default() ;

  // CLAP loop.
  'clap: loop {
    match args.next() {
      Some(ref arg) if arg == "--color" => {
        match val_of_next(& mut args, "bool") {
          Ok(color) => painter = if color {
            Painter::default()
          } else {
            Painter::none()
          },
          // Returning if error.
          Err(s) => return Err( (s, painter) ),
        }
      },
      Some(ref arg) if arg == "--seed" => {
        match args.next() {
          Some(seed_str) => seed = Seed::of_str(& seed_str),
          None => return Err(
            (format!("expected seed, got nothing"), painter)
          ),
        }
      },

      Some(ref arg) if arg == "--depth" => {
        match args.next() {
          Some(depth_str) => depth = depth_str.parse::<usize>().unwrap(),
          None => return Err(
            (format!("expected depth, got nothing"), painter)
          ),
        }
      },

      Some(ref arg) if arg == "--expect" => {
        match args.next() {
          Some(expect_str) => expect = expect_str.parse::<usize>().unwrap(),
          None => return Err(
            (format!("expected expect, got nothing"), painter)
          ),
        }
      },
      Some(arg) => println!("arg: {}", arg),
      None => break 'clap,

    }
  } ;

  Ok( (seed, painter,depth,expect) )
}
