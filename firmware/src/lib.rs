#![no_std]

pub mod credentials;
pub mod decision;
pub mod diagnostic;
pub mod frontend;
pub mod fusion;
pub mod levels;
pub mod model;
pub mod notification;
pub mod rules;

#[cfg(any(target_os = "none", test))]
mod memory;

#[cfg(test)]
extern crate std;
#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "../build/bundle.rs"]
mod bundle;

pub mod config {
    include!(concat!(env!("OUT_DIR"), "/model_config.rs"));
}
