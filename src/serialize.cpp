#include "serialize.hpp"

#include <cstring>
#include <fstream>

namespace serialize {

namespace {

inline void SetError(std::string* error_msg, const std::string& msg)
{
    if (error_msg != nullptr) {
        *error_msg = msg;
    }
}

template <typename T>
bool WriteVal(std::ostream& os, const T& val)
{
    os.write(reinterpret_cast<const char*>(&val), sizeof(T));
    return os.good();
}

template <typename T>
bool ReadVal(std::istream& is, T& val)
{
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
    return is.good();
}

bool WriteBytes(std::ostream& os, const void* ptr, std::size_t bytes)
{
    os.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(bytes));
    return os.good();
}

bool ReadBytes(std::istream& is, void* ptr, std::size_t bytes)
{
    is.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(bytes));
    return is.good();
}

} // namespace

bool InspectCheckpoint(const std::filesystem::path& path, CheckpointInfo& info, std::string* error_msg)
{
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        SetError(error_msg, "Failed to open checkpoint file for reading: " + path.string());
        return false;
    }

    char magic[4] = {};
    if (!ReadBytes(is, magic, 4) || std::memcmp(magic, kMagic, 4) != 0) {
        SetError(error_msg, "Invalid file format: magic bytes mismatch in " + path.string());
        return false;
    }

    if (!ReadVal(is, info.version)) {
        SetError(error_msg, "Corrupted header: failed to read version");
        return false;
    }
    if (info.version > kCurrentVersion) {
        SetError(error_msg, "Unsupported checkpoint version: " + std::to_string(info.version));
        return false;
    }

    if (!ReadVal(is, info.step) ||
        !ReadVal(is, info.epoch) ||
        !ReadVal(is, info.rng_state) ||
        !ReadVal(is, info.rng_inc) ||
        !ReadVal(is, info.num_params)) {
        SetError(error_msg, "Corrupted header: failed to read metadata");
        return false;
    }

    // Skip params to inspect optimizer info
    for (u32 i = 0; i < info.num_params; ++i) {
        u32 rows = 0;
        u32 cols = 0;
        if (!ReadVal(is, rows) || !ReadVal(is, cols)) {
            SetError(error_msg, "Corrupted parameter descriptor in checkpoint header");
            return false;
        }
        const u64 num_floats = static_cast<u64>(rows) * cols;
        is.seekg(static_cast<std::streamoff>(num_floats * sizeof(f32)), std::ios::cur);
    }

    u32 has_opt = 0;
    if (ReadVal(is, has_opt) && has_opt != 0) {
        u32 opt_type_val = 0;
        if (ReadVal(is, opt_type_val)) {
            info.opt_type = static_cast<optim::OptimizerType>(opt_type_val);
        }
    } else {
        info.opt_type = optim::OptimizerType::Unknown;
    }

    return true;
}

bool SaveCheckpoint(const std::filesystem::path& path,
                    u64                          step,
                    u64                          epoch,
                    const Prng&                  rng,
                    std::span<Var* const>        params,
                    const optim::Optimizer*      opt,
                    std::string*                 error_msg)
{
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        SetError(error_msg, "Failed to open checkpoint file for writing: " + path.string());
        return false;
    }

    // 1. Magic & Version
    if (!WriteBytes(os, kMagic, 4) || !WriteVal(os, kCurrentVersion)) {
        SetError(error_msg, "Failed writing header");
        return false;
    }

    // 2. Training metadata & RNG state
    u64 rng_state = 0;
    u64 rng_inc   = 0;
    rng.GetState(rng_state, rng_inc);

    if (!WriteVal(os, step) ||
        !WriteVal(os, epoch) ||
        !WriteVal(os, rng_state) ||
        !WriteVal(os, rng_inc)) {
        SetError(error_msg, "Failed writing metadata");
        return false;
    }

    // 3. Parameters
    const u32 num_params = static_cast<u32>(params.size());
    if (!WriteVal(os, num_params)) {
        SetError(error_msg, "Failed writing parameter count");
        return false;
    }

    for (Var* p : params) {
        if (p == nullptr || p->val == nullptr) {
            const u32 zero = 0;
            WriteVal(os, zero);
            WriteVal(os, zero);
            continue;
        }
        const u32 rows = p->val->rows;
        const u32 cols = p->val->cols;
        if (!WriteVal(os, rows) || !WriteVal(os, cols)) {
            SetError(error_msg, "Failed writing parameter dimensions");
            return false;
        }
        const u64 size = static_cast<u64>(rows) * cols;
        if (!WriteBytes(os, p->val->data, size * sizeof(f32))) {
            SetError(error_msg, "Failed writing parameter data");
            return false;
        }
    }

    // 4. Optimizer State
    const u32 has_opt = (opt != nullptr) ? 1 : 0;
    if (!WriteVal(os, has_opt)) {
        SetError(error_msg, "Failed writing optimizer flag");
        return false;
    }

    if (has_opt != 0 && opt != nullptr) {
        const u32 opt_type = static_cast<u32>(opt->Type());
        const f32 lr       = opt->GetLr();
        if (!WriteVal(os, opt_type) || !WriteVal(os, lr)) {
            SetError(error_msg, "Failed writing optimizer header");
            return false;
        }

        if (opt->Type() == optim::OptimizerType::Adam) {
            const auto* adam = static_cast<const optim::Adam*>(opt);
            const u64 opt_step = adam->StepCount();
            if (!WriteVal(os, opt_step)) { return false; }
            const auto& m = adam->M();
            const auto& v = adam->V();
            const u32 num_bufs = static_cast<u32>(m.size());
            if (!WriteVal(os, num_bufs)) { return false; }
            for (u32 i = 0; i < num_bufs; ++i) {
                const u64 sz = m[i].size();
                if (!WriteVal(os, sz)) { return false; }
                if (sz > 0) {
                    if (!WriteBytes(os, m[i].data(), sz * sizeof(f32)) ||
                        !WriteBytes(os, v[i].data(), sz * sizeof(f32))) {
                        SetError(error_msg, "Failed writing Adam moments");
                        return false;
                    }
                }
            }
        } else if (opt->Type() == optim::OptimizerType::AdamW) {
            const auto* adamw = static_cast<const optim::AdamW*>(opt);
            const u64 opt_step = adamw->StepCount();
            if (!WriteVal(os, opt_step)) { return false; }
            const auto& m = adamw->M();
            const auto& v = adamw->V();
            const u32 num_bufs = static_cast<u32>(m.size());
            if (!WriteVal(os, num_bufs)) { return false; }
            for (u32 i = 0; i < num_bufs; ++i) {
                const u64 sz = m[i].size();
                if (!WriteVal(os, sz)) { return false; }
                if (sz > 0) {
                    if (!WriteBytes(os, m[i].data(), sz * sizeof(f32)) ||
                        !WriteBytes(os, v[i].data(), sz * sizeof(f32))) {
                        SetError(error_msg, "Failed writing AdamW moments");
                        return false;
                    }
                }
            }
        } else if (opt->Type() == optim::OptimizerType::SGD) {
            const auto* sgd = static_cast<const optim::SGD*>(opt);
            const auto& vel = sgd->Velocity();
            const u32 num_bufs = static_cast<u32>(vel.size());
            if (!WriteVal(os, num_bufs)) { return false; }
            for (u32 i = 0; i < num_bufs; ++i) {
                const u64 sz = vel[i].size();
                if (!WriteVal(os, sz)) { return false; }
                if (sz > 0) {
                    if (!WriteBytes(os, vel[i].data(), sz * sizeof(f32))) {
                        SetError(error_msg, "Failed writing SGD velocity");
                        return false;
                    }
                }
            }
        } else if (opt->Type() == optim::OptimizerType::RMSProp) {
            const auto* rms = static_cast<const optim::RMSProp*>(opt);
            const auto& sq = rms->SquareAvg();
            const auto& mom = rms->MomBuffer();
            const auto& grad_avg = rms->GradAvg();
            const u32 num_bufs = static_cast<u32>(sq.size());
            if (!WriteVal(os, num_bufs)) { return false; }
            for (u32 i = 0; i < num_bufs; ++i) {
                const u64 sz = sq[i].size();
                if (!WriteVal(os, sz)) { return false; }
                if (sz > 0) {
                    if (!WriteBytes(os, sq[i].data(), sz * sizeof(f32))) { return false; }
                    const u32 has_mom = (!mom.empty() && mom[i].size() == sz) ? 1 : 0;
                    if (!WriteVal(os, has_mom)) { return false; }
                    if (has_mom != 0 && !WriteBytes(os, mom[i].data(), sz * sizeof(f32))) { return false; }
                    const u32 has_ga = (!grad_avg.empty() && grad_avg[i].size() == sz) ? 1 : 0;
                    if (!WriteVal(os, has_ga)) { return false; }
                    if (has_ga != 0 && !WriteBytes(os, grad_avg[i].data(), sz * sizeof(f32))) { return false; }
                }
            }
        }
    }

    // 5. Footer integrity marker
    if (!WriteBytes(os, kFooter, 4)) {
        SetError(error_msg, "Failed writing footer");
        return false;
    }

    return true;
}

bool LoadCheckpoint(const std::filesystem::path& path,
                    u64&                         step,
                    u64&                         epoch,
                    Prng&                        rng,
                    std::span<Var* const>        params,
                    optim::Optimizer*            opt,
                    std::string*                 error_msg)
{
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        SetError(error_msg, "Failed to open checkpoint file for reading: " + path.string());
        return false;
    }

    // 1. Magic & Version
    char magic[4] = {};
    if (!ReadBytes(is, magic, 4) || std::memcmp(magic, kMagic, 4) != 0) {
        SetError(error_msg, "Magic header mismatch");
        return false;
    }

    u32 version = 0;
    if (!ReadVal(is, version)) {
        SetError(error_msg, "Failed reading version");
        return false;
    }
    if (version > kCurrentVersion) {
        SetError(error_msg, "Unsupported checkpoint version: " + std::to_string(version));
        return false;
    }

    // 2. Metadata & RNG
    u64 rng_state = 0;
    u64 rng_inc   = 0;
    if (!ReadVal(is, step) ||
        !ReadVal(is, epoch) ||
        !ReadVal(is, rng_state) ||
        !ReadVal(is, rng_inc)) {
        SetError(error_msg, "Failed reading checkpoint metadata");
        return false;
    }
    rng.SetState(rng_state, rng_inc);

    // 3. Parameters
    u32 num_params = 0;
    if (!ReadVal(is, num_params)) {
        SetError(error_msg, "Failed reading parameter count");
        return false;
    }
    if (num_params != params.size()) {
        SetError(error_msg, "Parameter count mismatch: checkpoint has " +
                 std::to_string(num_params) + ", target model has " +
                 std::to_string(params.size()));
        return false;
    }

    for (std::size_t i = 0; i < params.size(); ++i) {
        Var* p = params[i];
        u32 rows = 0;
        u32 cols = 0;
        if (!ReadVal(is, rows) || !ReadVal(is, cols)) {
            SetError(error_msg, "Failed reading parameter dimensions");
            return false;
        }

        if (p == nullptr || p->val == nullptr) {
            if (rows != 0 || cols != 0) {
                SetError(error_msg, "Null parameter expectation mismatch at index " + std::to_string(i));
                return false;
            }
            continue;
        }

        if (p->val->rows != rows || p->val->cols != cols) {
            SetError(error_msg, "Shape mismatch for parameter " + std::to_string(i) +
                     ": expected [" + std::to_string(p->val->rows) + ", " +
                     std::to_string(p->val->cols) + "], found [" +
                     std::to_string(rows) + ", " + std::to_string(cols) + "]");
            return false;
        }

        const u64 size = static_cast<u64>(rows) * cols;
        if (!ReadBytes(is, p->val->data, size * sizeof(f32))) {
            SetError(error_msg, "Failed reading parameter data at index " + std::to_string(i));
            return false;
        }
    }

    // 4. Optimizer State
    u32 has_opt = 0;
    if (!ReadVal(is, has_opt)) {
        SetError(error_msg, "Failed reading optimizer flag");
        return false;
    }

    if (has_opt != 0) {
        u32 opt_type = 0;
        f32 lr       = 0.0f;
        if (!ReadVal(is, opt_type) || !ReadVal(is, lr)) {
            SetError(error_msg, "Failed reading optimizer header");
            return false;
        }

        if (opt != nullptr) {
            if (static_cast<u32>(opt->Type()) != opt_type) {
                SetError(error_msg, "Optimizer type mismatch: checkpoint has " +
                         std::to_string(opt_type) + ", runtime has " +
                         std::to_string(static_cast<u32>(opt->Type())));
                return false;
            }
            opt->SetLr(lr);

            if (opt->Type() == optim::OptimizerType::Adam) {
                auto* adam = static_cast<optim::Adam*>(opt);
                u64 opt_step = 0;
                u32 num_bufs = 0;
                if (!ReadVal(is, opt_step) || !ReadVal(is, num_bufs)) { return false; }
                adam->SetStepCount(opt_step);
                auto& m = adam->M();
                auto& v = adam->V();
                if (num_bufs != m.size()) {
                    SetError(error_msg, "Adam moment buffer count mismatch");
                    return false;
                }
                for (u32 i = 0; i < num_bufs; ++i) {
                    u64 sz = 0;
                    if (!ReadVal(is, sz) || sz != m[i].size()) { return false; }
                    if (sz > 0) {
                        if (!ReadBytes(is, m[i].data(), sz * sizeof(f32)) ||
                            !ReadBytes(is, v[i].data(), sz * sizeof(f32))) {
                            return false;
                        }
                    }
                }
            } else if (opt->Type() == optim::OptimizerType::AdamW) {
                auto* adamw = static_cast<optim::AdamW*>(opt);
                u64 opt_step = 0;
                u32 num_bufs = 0;
                if (!ReadVal(is, opt_step) || !ReadVal(is, num_bufs)) { return false; }
                adamw->SetStepCount(opt_step);
                auto& m = adamw->M();
                auto& v = adamw->V();
                if (num_bufs != m.size()) {
                    SetError(error_msg, "AdamW moment buffer count mismatch");
                    return false;
                }
                for (u32 i = 0; i < num_bufs; ++i) {
                    u64 sz = 0;
                    if (!ReadVal(is, sz) || sz != m[i].size()) { return false; }
                    if (sz > 0) {
                        if (!ReadBytes(is, m[i].data(), sz * sizeof(f32)) ||
                            !ReadBytes(is, v[i].data(), sz * sizeof(f32))) {
                            return false;
                        }
                    }
                }
            } else if (opt->Type() == optim::OptimizerType::SGD) {
                auto* sgd = static_cast<optim::SGD*>(opt);
                u32 num_bufs = 0;
                if (!ReadVal(is, num_bufs)) { return false; }
                auto& vel = sgd->Velocity();
                if (num_bufs != vel.size()) { return false; }
                for (u32 i = 0; i < num_bufs; ++i) {
                    u64 sz = 0;
                    if (!ReadVal(is, sz) || sz != vel[i].size()) { return false; }
                    if (sz > 0 && !ReadBytes(is, vel[i].data(), sz * sizeof(f32))) {
                        return false;
                    }
                }
            } else if (opt->Type() == optim::OptimizerType::RMSProp) {
                auto* rms = static_cast<optim::RMSProp*>(opt);
                u32 num_bufs = 0;
                if (!ReadVal(is, num_bufs)) { return false; }
                auto& sq = rms->SquareAvg();
                auto& mom = rms->MomBuffer();
                auto& grad_avg = rms->GradAvg();
                if (num_bufs != sq.size()) { return false; }
                for (u32 i = 0; i < num_bufs; ++i) {
                    u64 sz = 0;
                    if (!ReadVal(is, sz) || sz != sq[i].size()) { return false; }
                    if (sz > 0) {
                        if (!ReadBytes(is, sq[i].data(), sz * sizeof(f32))) { return false; }
                        u32 has_mom = 0;
                        if (!ReadVal(is, has_mom)) { return false; }
                        if (has_mom != 0) {
                            if (mom.empty() || mom[i].size() != sz ||
                                !ReadBytes(is, mom[i].data(), sz * sizeof(f32))) {
                                return false;
                            }
                        }
                        u32 has_ga = 0;
                        if (!ReadVal(is, has_ga)) { return false; }
                        if (has_ga != 0) {
                            if (grad_avg.empty() || grad_avg[i].size() != sz ||
                                !ReadBytes(is, grad_avg[i].data(), sz * sizeof(f32))) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }

    // 5. Footer check
    char footer[4] = {};
    if (!ReadBytes(is, footer, 4) || std::memcmp(footer, kFooter, 4) != 0) {
        SetError(error_msg, "Checkpoint integrity compromised: footer mismatch");
        return false;
    }

    return true;
}

bool SaveModule(const std::filesystem::path& path, const nn::Module& module, std::string* error_msg)
{
    Prng dummy_rng(0, 0);
    const auto params = module.Parameters();
    return SaveCheckpoint(path, 0, 0, dummy_rng, params, nullptr, error_msg);
}

bool LoadModule(const std::filesystem::path& path, nn::Module& module, std::string* error_msg)
{
    Prng dummy_rng;
    u64 step = 0;
    u64 epoch = 0;
    const auto params = module.Parameters();
    return LoadCheckpoint(path, step, epoch, dummy_rng, params, nullptr, error_msg);
}

bool SaveMat(const std::filesystem::path& path, const Mat& mat, std::string* error_msg)
{
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        SetError(error_msg, "Failed to open Mat file for writing: " + path.string());
        return false;
    }
    if (!WriteBytes(os, kMagic, 4) || !WriteVal(os, kCurrentVersion)) { return false; }
    const u32 rows = mat.rows;
    const u32 cols = mat.cols;
    if (!WriteVal(os, rows) || !WriteVal(os, cols)) { return false; }
    const u64 sz = static_cast<u64>(rows) * cols;
    if (!WriteBytes(os, mat.data, sz * sizeof(f32))) { return false; }
    return WriteBytes(os, kFooter, 4);
}

bool LoadMat(const std::filesystem::path& path, Mat& mat, std::string* error_msg)
{
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        SetError(error_msg, "Failed to open Mat file for reading: " + path.string());
        return false;
    }
    char magic[4] = {};
    if (!ReadBytes(is, magic, 4) || std::memcmp(magic, kMagic, 4) != 0) { return false; }
    u32 version = 0;
    if (!ReadVal(is, version) || version > kCurrentVersion) { return false; }
    u32 rows = 0;
    u32 cols = 0;
    if (!ReadVal(is, rows) || !ReadVal(is, cols)) { return false; }
    if (mat.rows != rows || mat.cols != cols) {
        SetError(error_msg, "Shape mismatch: expected [" + std::to_string(mat.rows) +
                 ", " + std::to_string(mat.cols) + "], found [" +
                 std::to_string(rows) + ", " + std::to_string(cols) + "]");
        return false;
    }
    const u64 sz = static_cast<u64>(rows) * cols;
    if (!ReadBytes(is, mat.data, sz * sizeof(f32))) { return false; }
    char footer[4] = {};
    return ReadBytes(is, footer, 4) && std::memcmp(footer, kFooter, 4) == 0;
}

} // namespace serialize
