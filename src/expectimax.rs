pub use grid::{ Cell, Grid } ;
pub use frame::{ Frame } ;
pub use common::{ Evolution } ;

extern crate rand;
use rand::{Rng};

use std::iter::*;

#[derive(Copy, Clone)]
pub enum Move {
  Up,
  Down,
  Left,
  Right,
  Spawn((usize,usize,i32,f32)),
  Start
}

#[derive(Clone)]
pub struct State {
  move_used : Move, //Move made to get to this board
  depth : usize, //The current depth
  grid : Grid
}

/// State of a board, with how deep into an expectimax tree we are, what move was used to reach this point, and the grid layout.
impl State {
  pub fn new(move_used : Move, depth : usize, grid : Grid) -> State {
    State {move_used: move_used, depth:depth, grid:grid}
  }
  
  pub fn move_used(& self) -> Move { self.move_used }
}

#[derive(Clone)]
pub struct ExpectiMax {
  max_depth : usize,
  max_width : usize
}

/// Object that alternately calls expecti and max layers to a max depth to return the optimal move.
impl ExpectiMax {
  pub fn new(max_depth : usize, max_width : usize) -> ExpectiMax {
    ExpectiMax { max_depth : max_depth , max_width : max_width}
  }

  /// Evaluates itself if at max depth, otherwise returns max score of possible successors.
  pub fn max_layer(&self, s : &State) -> (State, f32) {
    let moves_vec = get_moves(&s.grid);
    // Return current board's score if at deepest layer allowed, or nothing to do
    if s.depth == self.max_depth || moves_vec.len() == 0 {
      let score = evaluate_moves(&s.grid);
      (s.clone(), score)
    } else {
      let moves_iter = moves_vec.iter();
      // for all possible moves, make a copy of the grid doing that move
      let states_vec : Vec<State> = FromIterator::from_iter(
        moves_iter.map(|&move_ref| {
          State::new(move_ref, s.depth + 1, match move_ref {
              Move::Up => {let mut new_grid = s.grid.clone(); new_grid.up(); new_grid},
              Move::Left => {let mut new_grid = s.grid.clone(); new_grid.left(); new_grid},
              Move::Right => {let mut new_grid = s.grid.clone(); new_grid.right(); new_grid},
              Move::Down => {let mut new_grid = s.grid.clone(); new_grid.down(); new_grid},
              _ => {s.grid.clone()}
          })
        }));

      // For each possible next state, call the expecti layer on it
      let results = states_vec.iter().map(|next_state| {
        self.expecti_layer(next_state)
      });

      // Find the next state with the best score
      let mut max_score : f32 = -1000000000000.;
      let mut max_idx : usize = 0;
      for (idx, score) in results.enumerate() {
        if score > max_score {
          max_score = score;
          max_idx = idx;
        }
      }
      // Return a copy of the best state with the move used to reach it, and its score
      (states_vec.get(max_idx).unwrap().clone(), max_score)
    }
  }

  /// Expecti Layer of tree, considers possible random spawns to predict value
  pub fn expecti_layer(&self, s : &State) -> f32 {
    let moves_vec = get_moves(&s.grid);
    // If already at max depth or no possible moves, return grid score
    if s.depth == self.max_depth || moves_vec.len() == 0 {
      let score = evaluate_moves(&s.grid);
      score
    } else {
      // Count how many empty cells we have
      let num_empty = evaluate_empty_cells(&s.grid);

      // For each empty cell, push its coordinates and chance of a 2-spawn and 4-spawn there
      let mut spawn_points:Vec<Option<(usize,usize,i32,f32)>> = Vec::new();
      for x in s.grid.get_free() {
          match x{
            (a,b) => {
                spawn_points.push(Some((a,b, 2, 0.8 / (num_empty as f32))));
                spawn_points.push(Some((a,b, 4, 0.2 / (num_empty as f32))));
            }
        }
      }

      let num_spawns = num_empty * 2;
      let num_samples = self.max_width;

      // For sample, shuffle all possibilities and take first section of vector
      let slice:&mut[Option<(usize,usize,i32,f32)>] = &mut *spawn_points;
      rand::thread_rng().shuffle(slice);
      let mut shuffled:Vec<Option<(usize,usize,i32,f32)>> = slice.iter().cloned().collect();
      // Sample minimum of number of spawns vs. number of samples
      let sampled_spawns = if num_spawns > num_samples {
        shuffled.truncate(num_samples);
        shuffled
      } else {
        shuffled.truncate(num_spawns);
        shuffled
      };

      // Generate next versions of board for sampled moves
      let states_vec : Vec<State> = FromIterator::from_iter(
        sampled_spawns.iter().map(|&spawn| {
          State::new(Move::Spawn(spawn.unwrap()), s.depth + 1, s.grid.add_tile_to_clone(spawn.unwrap()))
        }));

      // Find scores of next chosen states
      let results = states_vec.iter().map(|next_state| {
        self.max_layer(next_state)
      });

      // Calculate average score for this layer
      let mut cum_score = 0.;
      let mut cum_prob = 0.;
      for (id, (_, score)) in results.enumerate() {
        let state = states_vec.get(id);
        // Extract probability of a spawn occurring to generate this state
        let prob = match state.unwrap().move_used {
            Move::Spawn(x) => { let (_,_,_, proba) = x; proba },
            _ => {0.}
        };
        // Multiply that board's score by this probability
        cum_score += score * prob;
        cum_prob += prob;
      }
      cum_score / cum_prob
    }
  }
}

/// Return a vector of all moves the grid can make without causing nothing to move.
fn get_moves(grid_to_clone: & Grid) -> Vec<Move>{
    let mut vec = Vec::new();
    let mut grid = grid_to_clone.clone();
    if grid.up() != Evolution::Nothing {
        vec.push(Move::Up);
    }
    grid = grid_to_clone.clone();
    if grid.left() != Evolution::Nothing {
        vec.push(Move::Left);
    }
    grid = grid_to_clone.clone();
    if grid.right() != Evolution::Nothing {
        vec.push(Move::Right);
    }
    grid = grid_to_clone.clone();
    if grid.down() != Evolution::Nothing {
        vec.push(Move::Down);
    }
    vec
}

// Calculate sum of scores from heuristics used to evaluate the board
fn evaluate_moves(grid: & Grid) -> f32{
  2.*evaluate_empty_cells(grid) as f32 + evaluate_smoothness(grid) + evaluate_log_of_sum_squares(grid) as f32 + evaluate_near_death(grid) as f32 + 2.*evaluate_large_tile_location(grid) as f32
}

///Returns how many empty cells exist on the board
fn evaluate_empty_cells(grid: & Grid) -> usize{
  let value: usize = grid.get_free().len();
  value
}

///Returns a score based on locations of higher tiles on the board
fn evaluate_large_tile_location(grid: & Grid) -> i8{
  let mut val:i8 = 0;
  let best = get_best_tile_val(grid);
  let sec_best = get_sec_best_tile_val(grid, best);
  let third_best = get_sec_best_tile_val(grid, sec_best);
  let fourth_best = get_sec_best_tile_val(grid, third_best);
  for i in 0..4 {
    for j in 0..4 {
      match grid.grid()[i][j]{
        Some(ref cell) => {
            // Penalize if best tile is in center 4
            if best == cell.val() && (i==1 || i==2) && (j==1 || j==2){
                val = val - 6;
            } // Or else not in corner
            else if best == cell.val() && (i==1 || i==2 || j==1 || j==2){
                val = val - 5;
            } // Same for second-best possible
            if sec_best == cell.val() && (i==1 || i==2) && (j==1 || j==2){
                val = val - 5;
            } // And for third-best possible
            if third_best == cell.val() && (i==1 || i==2) && (j==1 || j==2){
                val = val - 4;
            } // And for fourth-best
            if fourth_best == cell.val() && (i==1 || i==2) && (j==1 || j==2) {
                val = val - 3;
            } 
            else if best == cell.val() && (i==1 || i==2 || j==1 || j==2){
                val = val - 2;
            }
		}, 
	    _ => {},
      }
    }
  }
  val
}

/// Return a severe value if grid would be too full to move
fn evaluate_near_death(grid: & Grid) -> i8{
    let num_empty:usize = evaluate_empty_cells(grid);
    if num_empty == 0 {
        -100
    } else if num_empty == 1 {
        -15
    } else if num_empty == 2 {
        -8
    } else {
        0

    }
}

/// Reward adjacent tiles with similar value
fn evaluate_smoothness(grid: & Grid) -> f32{
    let mut smoothness = 0.0;    
    let mut left;
    let mut right;
    let mut top;
    let mut bottom;

    for i in 0..3 {
        for j in 0..4 {
          left = 0;
          right = 0;
          match grid.grid()[i][j]{
            Some(ref cell) => {
                left = cell.pow();
		    }, 
	        _ => {},
          }
          match grid.grid()[i+1][j]{
            Some(ref cell) => {
                right = cell.pow();
		    }, 
	        _ => {},
          }
          let factor;
          if left > right { 
              factor = left - right; 
          } else {
              factor = right - left;
          }
          smoothness -= factor as f32;
        }
    }

    for i in 0..4 {
        for j in 0..3 {
          top = 0;
          bottom = 0;
          match grid.grid()[i][j]{
            Some(ref cell) => {
                top = cell.pow();
		    }, 
	        _ => {},
          }
          match grid.grid()[i][j+1]{
            Some(ref cell) => {
                bottom = cell.pow();
		    }, 
	        _ => {},
          }
          let factor;
          if top > bottom { 
              factor = top - bottom; 
          } else {
              factor = bottom - top;
          }
          smoothness -= factor as f32;
        }
    }

    -1. * smoothness*smoothness / 200.0
}

/// Calculate the log (base 2) of summed squares of tile values; rewards bigger tile scores rather than more
fn evaluate_log_of_sum_squares(grid: & Grid) -> f32{
    let mut sum_of_squares = 0;
    for i in 0..4 {
        for j in 0..4 {
          match grid.grid()[i][j]{
            Some(ref cell) => {
                sum_of_squares += cell.val()*cell.val();
		    }, 
	        _ => {},
          }
        }
    }

    (sum_of_squares as f32).log2()
}

/// Finds highest value of any tile on the board
fn get_best_tile_val(grid: & Grid) -> usize{
    let mut max = 0;
    for i in 0..4 {
        for j in 0..4 {
          match grid.grid()[i][j]{
            Some(ref cell) => {
                if cell.val()>max {
                    max = cell.val();
                }
		    }, 
	        _ => {},
          }
        }
    }
    max
}

/// Finds second highest value of any tile on the board
fn get_sec_best_tile_val(grid: & Grid, max: usize) -> usize{
    let mut second_max = 0;
    for i in 0..4 {
        for j in 0..4 {
          match grid.grid()[i][j]{
            Some(ref cell) => {
                if cell.val()>second_max && cell.val()<max {
                    second_max = cell.val();
                }
		    }, 
	        _ => {},
          }
        }
    }
    second_max
}
