# Training Strategy
# This need work! but pretty much `straightforward`!

using Lux
using NNlib
using Zygote
using Optimisers
using JLD2
using Accessors
using Distributions
using Random

# ── 1. Data Format ────────────────────────────────────────────────────────────
#
# mel:     (3000, 80, batch)   — log mel spectrogram, one 30s chunk per item
# tokens:  (seq_len, batch)    — input token ids, starts with SOT token
# targets: (seq_len, batch)    — tokens shifted left by 1, ends with EOT token
#
# Example for a single item:
#   audio:   "Hello world"
#   tokens:  [SOT, "Hello", " world"]
#   targets: ["Hello", " world", EOT]
#
# The model learns: given audio + previous tokens, predict the next token.

function make_targets(tokens::Matrix{Int64}, eot_id::Int64)
    # Shift tokens left by 1 and append EOT
    targets = similar(tokens)
    targets[1:(end - 1), :] = tokens[2:end, :]
    targets[end, :] .= eot_id
    return targets
end

# ── 2. Loss Function ──────────────────────────────────────────────────────────

function loss_fn(model, ps, st, mel, tokens, targets)
    logits, st_new = model(mel, tokens, ps, st)

    # logits:  (n_vocab, seq_len, batch)
    # targets: (seq_len, batch)
    n_vocab = size(logits, 1)

    # Flatten to 2D for cross entropy
    logits_2d = reshape(logits, n_vocab, :)    # (n_vocab, seq_len*batch)
    targets_1d = reshape(targets, :)             # (seq_len*batch,)

    # One-hot encode targets for NNlib crossentropy
    targets_oh = Lux.onehotbatch(targets_1d, 1:n_vocab)  # (n_vocab, seq_len*batch)

    l = mean(crossentropy(logits_2d, targets_oh))
    return l, st_new
end

# ── 3. Optimizer Setup ────────────────────────────────────────────────────────

function setup_optimizer(
        ps;
        lr::Float32 = 1.0f-4,
        weight_decay::Float32 = 1.0f-2,
        freeze_encoder::Bool = false
    )
    if freeze_encoder
        # Only optimize decoder — encoder weights stay fixed
        opt = Optimisers.AdamW(lr, (0.9f0, 0.999f0), weight_decay)
        opt_state = Optimisers.setup(opt, ps.decoder)
        @info "Optimizer: AdamW lr=$lr — encoder frozen, decoder only"
    else
        # Full fine-tune — optimize everything
        opt = Optimisers.AdamW(lr, (0.9f0, 0.999f0), weight_decay)
        opt_state = Optimisers.setup(opt, ps)
        @info "Optimizer: AdamW lr=$lr — full model"
    end
    return opt_state
end

# ── 4. Training Step ──────────────────────────────────────────────────────────

function train_step!(
        model, ps, st, opt_state, mel, tokens, targets;
        freeze_encoder::Bool = false
    )
    # Compute loss and gradients
    (l, st), grads = Zygote.withgradient(ps) do ps
        loss_fn(model, ps, st, mel, tokens, targets)
    end

    if freeze_encoder
        # Only update decoder parameters
        opt_state, ps_dec = Optimisers.update!(opt_state, ps.decoder, grads[1].decoder)
        ps = @set ps.decoder = ps_dec
    else
        # Update all parameters
        opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
    end

    return l, ps, st, opt_state
end

# ── 5. Validation Step ────────────────────────────────────────────────────────

function val_step(model, ps, st, mel, tokens, targets)
    # No gradient tracking needed
    l, _ = loss_fn(model, ps, st, mel, tokens, targets)
    return l
end

# ── 6. Learning Rate Scheduler ────────────────────────────────────────────────
#
# Linear warmup then cosine decay — standard for transformer fine-tuning

function lr_schedule(
        step::Int;
        warmup_steps::Int = 1000,
        total_steps::Int = 10000,
        base_lr::Float32 = 1.0f-4,
        min_lr::Float32 = 1.0f-6
    )
    if step < warmup_steps
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5f0 * (base_lr - min_lr) * (1 + cos(π * progress))
    end
end

function update_lr!(opt_state, lr::Float32)
    # Walk the optimizer state tree and update learning rate
    return Optimisers.adjust!(opt_state, lr)
end

# ── 7. Checkpointing ──────────────────────────────────────────────────────────

function save_checkpoint(path::String, ps, st, opt_state, epoch::Int, step::Int, loss::Float32)
    @save path ps st opt_state epoch step loss
    return @info "Checkpoint saved: $path (epoch=$epoch, step=$step, loss=$loss)"
end

function load_checkpoint(path::String)
    @load path ps st opt_state epoch step loss
    @info "Checkpoint loaded: $path (epoch=$epoch, step=$step, loss=$loss)"
    return ps, st, opt_state, epoch, step, loss
end

# ── 8. Training Loop ──────────────────────────────────────────────────────────

function train!(
        model, ps, st, dataloader, val_dataloader;
        n_epochs::Int = 10,
        lr::Float32 = 1.0f-4,
        weight_decay::Float32 = 1.0f-2,
        warmup_steps::Int = 1000,
        freeze_encoder::Bool = false,
        checkpoint_dir::String = "checkpoints",
        checkpoint_every::Int = 1000,
        log_every::Int = 10,
    )
    isdir(checkpoint_dir) || mkdir(checkpoint_dir)

    opt_state = setup_optimizer(ps; lr, weight_decay, freeze_encoder)
    total_steps = n_epochs * length(dataloader)
    global_step = 0
    best_val_loss = Inf32

    for epoch in 1:n_epochs
        epoch_loss = 0.0f0
        n_batches = 0

        for (mel, tokens, targets) in dataloader
            global_step += 1

            # Update learning rate
            current_lr = lr_schedule(
                global_step;
                warmup_steps, total_steps, base_lr = lr
            )
            update_lr!(opt_state, Float32(current_lr))

            # Training step
            l, ps, st, opt_state = train_step!(
                model, ps, st, opt_state, mel, tokens, targets;
                freeze_encoder
            )

            epoch_loss += l
            n_batches += 1

            # Logging
            if global_step % log_every == 0
                @info "step=$global_step epoch=$epoch loss=$(round(l, digits = 4)) lr=$(round(current_lr, sigdigits = 3))"
            end

            # Checkpointing
            if global_step % checkpoint_every == 0
                path = joinpath(checkpoint_dir, "step_$(global_step).jld2")
                save_checkpoint(path, ps, st, opt_state, epoch, global_step, l)
            end
        end

        # Validation
        val_loss = mean(
            [
                val_step(model, ps, st, mel, tokens, targets)
                    for (mel, tokens, targets) in val_dataloader
            ]
        )

        avg_train_loss = epoch_loss / n_batches
        @info "═══ Epoch $epoch complete — train_loss=$(round(avg_train_loss, digits = 4)) val_loss=$(round(val_loss, digits = 4))"

        # Save best model
        if val_loss < best_val_loss
            best_val_loss = val_loss
            save_checkpoint(
                joinpath(checkpoint_dir, "best.jld2"),
                ps, st, opt_state, epoch, global_step, val_loss
            )
            @info "New best model saved (val_loss=$val_loss)"
        end
    end

    return ps, st
end

# ── 9. Synthetic Data Test ────────────────────────────────────────────────────
#
# Use this to verify gradient flow before using real data.

function synthetic_dataloader(;
        n_batches::Int = 10,
        batch_size::Int = 4,
        seq_len::Int = 10,
        n_mels::Int = 80,
        n_frames::Int = 3000,
        n_vocab::Int = 51864,
        eot_id::Int = 1,
    )
    return [
        (
                randn(Float32, n_frames, n_mels, batch_size),                          # mel
                rand(Int64.(1:n_vocab), seq_len, batch_size),                          # tokens
                rand(Int64.(1:n_vocab), seq_len, batch_size),                          # targets
            )
            for _ in 1:n_batches
    ]
end

# ── 10. Fine-tuning Strategies ────────────────────────────────────────────────
#
# Strategy A — Full fine-tune
#   Best when: large domain-specific dataset, lots of compute
#   freeze_encoder = false
#
# Strategy B — Freeze encoder
#   Best when: small dataset, encoder features are strong
#   freeze_encoder = true
#   Only decoder weights are updated — faster and less prone to overfitting
#
# Strategy C — LoRA (future)
#   Best when: very small dataset, minimal compute
#   Inject low-rank adapter matrices into attention layers
#   Not implemented here — requires additional tooling
#
# ── 11. Recommended Datasets ─────────────────────────────────────────────────
#
# General English:     LibriSpeech        https://www.openslr.org/12
# Multiple languages:  Mozilla CommonVoice https://commonvoice.mozilla.org
# Your own domain:     Record + transcribe with existing Whisper model
#
# ── 12. Usage Example ─────────────────────────────────────────────────────────
#
# model, ps, st = Evie.Whisper.load_model("tiny.en")
#
# # Synthetic test — verify gradient flow
# train_dl = synthetic_dataloader(; n_batches=10, batch_size=4)
# val_dl   = synthetic_dataloader(; n_batches=2,  batch_size=4)
#
# ps, st = train!(model, ps, st, train_dl, val_dl;
#     n_epochs       = 2,
#     lr             = 1f-4,
#     freeze_encoder = true,     # start with encoder frozen
#     checkpoint_dir = "checkpoints/tiny_en",
# )
#
# # Save final weights
# @save "my_whisper_finetuned.jld2" ps st
