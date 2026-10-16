#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "autograd.hpp"
#include "base.hpp"
#include "mat.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "prng.hpp"

namespace serialize {

inline constexpr u32 kCurrentVersion = 1;
inline constexpr char kMagic[4] = {'P', 'Y', 'R', 'L'};
inline constexpr char kFooter[4] = {'E', 'N', 'D', 'P'};

struct CheckpointInfo {
    u32                  version     = 0;
    u64                  step        = 0;
    u64                  epoch       = 0;
    u64                  rng_state   = 0;
    u64                  rng_inc     = 0;
    u32                  num_params  = 0;
    optim::OptimizerType opt_type    = optim::OptimizerType::Unknown;
};

// Inspects a checkpoint file header without allocating or populating weights.
[[nodiscard]] bool InspectCheckpoint(const std::filesystem::path& path,
                                    CheckpointInfo&              info,
                                    std::string*                 error_msg = nullptr);

// Saves a full training checkpoint: metadata, PRNG state, model parameters, and optimizer moments.
[[nodiscard]] bool SaveCheckpoint(const std::filesystem::path& path,
                                  u64                          step,
                                  u64                          epoch,
                                  const Prng&                  rng,
                                  std::span<Var* const>        params,
                                  const optim::Optimizer*      opt = nullptr,
                                  std::string*                 error_msg = nullptr);

// Loads a full training checkpoint: restores metadata, PRNG state, parameters, and optimizer moments.
[[nodiscard]] bool LoadCheckpoint(const std::filesystem::path& path,
                                  u64&                         step,
                                  u64&                         epoch,
                                  Prng&                        rng,
                                  std::span<Var* const>        params,
                                  optim::Optimizer*            opt = nullptr,
                                  std::string*                 error_msg = nullptr);

// Module-level serialization (saves/loads only the module's parameters).
[[nodiscard]] bool SaveModule(const std::filesystem::path& path,
                              const nn::Module&            module,
                              std::string*                 error_msg = nullptr);

[[nodiscard]] bool LoadModule(const std::filesystem::path& path,
                              nn::Module&                  module,
                              std::string*                 error_msg = nullptr);

// Single Mat serialization.
[[nodiscard]] bool SaveMat(const std::filesystem::path& path,
                           const Mat&                   mat,
                           std::string*                 error_msg = nullptr);

[[nodiscard]] bool LoadMat(const std::filesystem::path& path,
                           Mat&                         mat,
                           std::string*                 error_msg = nullptr);

} // namespace serialize
