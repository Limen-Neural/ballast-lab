# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Expanded README user guides: getting started, ecosystem map, feature choice, common patterns, architecture brief, cross-language notes (#14)
- Crate- and item-level rustdoc for public API (`TrainingConfig`, `SpikenautTrainer`, `TrainingSummary`, etc.)
- CI step: `cargo doc --no-deps --all-features` with broken-doc-link warnings denied

### Changed

- License switched from GPL-3.0 to dual MIT/Apache-2.0 (chore for better adoption and to align with Limen-Neural org standard; see #9 and master neuromod#19)
  - Added `LICENSE-MIT` and `LICENSE-APACHE-2.0`
  - Updated `Cargo.toml` with `license = "MIT OR Apache-2.0"`
  - Updated `README.md` license section and added badge
  - Added SPDX-License-Identifier headers to source files
  - Removed old GPL LICENSE

## [0.1.0] - 2026-04 (initial)

### Added

- Generic reward-modulated plasticity loops for SNNs around `neuromod::SpikingNetwork`
- `SpikenautTrainer`, `TrainingConfig`, `TrainingExample`, `TrainingSummary`
- Integration feature for `limbic-critic` and `axon-encoder`
