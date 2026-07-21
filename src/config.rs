// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// Configuration knobs for [`crate::SpikenautTrainer`].
///
/// Values are serializable (serde) so they can be stored with checkpoints or
/// experiment configs. Defaults match a light reward-modulated loop; adjust as
/// homeostasis and plasticity rules expand.
///
/// # Fields
///
/// - `learning_rate` — step size for plasticity updates.
/// - `target_spikes_per_step` — target average spike rate for homeostasis.
/// - `homeostasis_strength` — how strongly activity is pulled toward the target.
/// - `batch_size` — intended batch size for higher-level training orchestration.
/// - `use_reward_modulation` — when `true`, rewards influence neuromodulators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Step size for plasticity-related updates.
    pub learning_rate: f32,
    /// Target average spikes per step (homeostasis setpoint).
    pub target_spikes_per_step: f32,
    /// Strength of homeostatic pull toward [`Self::target_spikes_per_step`].
    pub homeostasis_strength: f32,
    /// Intended batch size for session / orchestration layers.
    pub batch_size: usize,
    /// Whether reward signals modulate neuromodulators during training.
    pub use_reward_modulation: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            target_spikes_per_step: 0.1,
            homeostasis_strength: 0.001,
            batch_size: 1,
            use_reward_modulation: true,
        }
    }
}
