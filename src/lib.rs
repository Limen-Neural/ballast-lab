// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic reward-modulated plasticity loops for spiking neural networks.
//!
//! This crate provides a small training loop around [`neuromod::SpikingNetwork`]:
//! single-step reward modulation via [`SpikenautTrainer::train_step`] and batch
//! sessions via [`SpikenautTrainer::run_session`].
//!
//! # Features
//!
//! - **default** — core loop only (`neuromod` + serde/tracing/thiserror/rand).
//! - **`integration`** — optional deps on `limbic-critic` and `axon-encoder`.
//!   The trainer API still takes precomputed stimuli and scalar rewards; enable
//!   this when you want those sister crates resolved in the same build.
//!
//! # Quick example
//!
//! ```rust,no_run
//! use neuromod::SpikingNetwork;
//! use plasticity_lab::{SpikenautTrainer, TrainingConfig, TrainingExample};
//!
//! let mut trainer = SpikenautTrainer::new(TrainingConfig::default());
//! let mut network = SpikingNetwork::with_dimensions(32, 8, 64);
//! let batch = vec![TrainingExample {
//!     stimuli: vec![0.25; 64],
//!     reward: 0.2,
//! }];
//! let summary = trainer.run_session(&mut network, &batch).unwrap();
//! assert_eq!(summary.steps_processed, 1);
//! ```
//!
//! See the crate README for ecosystem map, ownership boundaries, and patterns.

pub mod config;
pub mod trainer;

pub use config::TrainingConfig;
pub use trainer::{SpikenautTrainer, TrainerError, TrainingExample, TrainingSummary};
