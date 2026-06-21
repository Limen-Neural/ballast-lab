# plasticity-lab

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)

Generic reward-modulated plasticity loops for spiking neural networks.

## Overview

`plasticity-lab` provides a small, reusable training loop around `neuromod::SpikingNetwork`.
It is intentionally domain-agnostic:

- Input encoding belongs to `axon-encoder`
- Reward shaping belongs to `limbic-critic`
- This crate runs the loop and tracks training summaries

## Scope and Ownership Boundaries

This crate provides generic reward-modulated plasticity loops for spiking neural networks. It is intentionally domain-agnostic.

### Owns
- SNN training loop abstractions
- Plasticity rule implementations (STDP, R-STDP, etc.)
- Integration with `limbic-critic` for reward shaping
- Training progress tracking and metrics
- Checkpointing and model serialization

### Does Not Own
- Project-specific trainer names (`SpikenautTrainer`)
- Domain-specific training logic (mining, trading, etc.)
- Differentiable or online distillation and teacher-student knowledge transfer

### Boundary with SynapticDistill.jl (Linear LIM-25)
- `plasticity-lab` (Rust): reward-modulated STDP / Hebbian plasticity rules and online low-level weight delta computation.
- `SynapticDistill.jl` (Julia): differentiable or online distillation and teacher-student knowledge transfer.
- `SynapticDistill.jl` must not become the home for STDP logic; `plasticity-lab` must not absorb distillation logic.
- A corresponding note should be aligned in `SynapticDistill.jl`.

### Allowed Dependencies
- `neuromod` (for neuromodulator integration)
- `limbic-critic` (for reward shaping)
- `axon-encoder` (for input encoding)
- Math and statistics libraries
- Serialization libraries

### Forbidden Dependencies
- Domain-specific training logic
- Project-specific naming conventions

(See issues #2, #3, #6 for full planning context and migration notes.)

## Usage

```rust
use neuromod::SpikingNetwork;
use plasticity_lab::{SpikenautTrainer, TrainingConfig, TrainingExample};

let mut trainer = SpikenautTrainer::new(TrainingConfig::default());
let mut network = SpikingNetwork::with_dimensions(32, 8, 64);

let batch = vec![
    TrainingExample {
        stimuli: vec![0.25; 64],
        reward: 0.2,
    },
    TrainingExample {
        stimuli: vec![0.4; 64],
        reward: -0.1,
    },
];

let summary = trainer.run_session(&mut network, &batch).unwrap();
println!("processed={}, avg_reward={}", summary.steps_processed, summary.avg_reward);
```

## Notes

- `train_step` accepts dynamic `&[f32]` stimuli and returns `Result<Vec<usize>, StepError>`.
- `run_session` returns a `TrainingSummary` with spike counts and parameter drift metrics.
- `axon-encoder` and `limbic-critic` are optional integration dependencies under the `integration` feature.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
