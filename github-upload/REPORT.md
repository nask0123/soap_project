# Soup DPO take-home: DON'T SHIP

## 1. Experiment and decision

Qwen2.5-1.5B-Instruct (revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306), Soup 0.75.1, TRL 0.29.1, Transformers 5.17.0, Torch 2.14.0+cu130, Python 3.12.3; Colab Tesla T4, 15,360 MiB. FP16 frozen base, no quantization, RAM layer streaming (28 layers, two pageable buffers), q_proj/v_proj LoRA rank 8, alpha 16, dropout 0; batch 1, accumulation 8, LR 1e-5, seed 42, one epoch, max length 512. Full run: 20260929T202553230689Z-full-dpo. The Soup DPO wrapper/TRL trainer was called from instrumented code, not the soup train CLI.

Data: trl-lib/ultrafeedback_binarized revision 47124cb5778f5d50de1c7676a412828f3ea7c555. Deterministic streaming shuffle (seed 42, buffer 2000); accepted 500 train and 100 source-test pairs. Required single-turn shared prompts, strict preference scores, distinct nonempty answers, prompt <=256 and complete conversations <=512 tokens. No truncation; normalized prompt hashes do not overlap. Selection scanned 1,115/184 rows. This is a short-answer, general preference proxy, not Russian support tickets. Semantic duplicates were not ruled out.

DON'T SHIP: held-out sum-log-probability ranking accuracy is 56% before and after; zero pairs flip either way. Length-normalized accuracy is also unchanged (71%). Mean preference-margin change is +0.0377 nats; 58/100 margins increase, but this is not demonstrated task accuracy improvement. The paired bootstrap accuracy interval [0,0] is degenerate because every observed paired difference is zero; it does not establish population equivalence. soup ship returns exit 2, DON'T SHIP, failed_rule=missing_baseline: no general-capability benchmark was supplied. The task-win condition also fails independently.

## 2. Memory budget, before versus measured

Pre-run assumptions: h=1536, intermediate=8960, 28 layers, vocab=151936, two sequences per preference pair. Full FP16 base: 1.5437B*2 = 2.88 GiB, primarily host-side. Tied embedding/head: 151936*1536*2 = 445 MiB resident. Decoder weights: ~46.8M/layer; two FP16 buffers = 179 MiB. LoRA count: 28*8*((1536+1536)+(1536+256)) = 1,089,536; FP32 weights, gradients and Adam moments = 16.6 MiB. Frozen reference adapter adds ~4.16 MiB, not a second base model. Checkpoint inputs: 28*2*512*1536*2 = 84 MiB, only part of activations. One FP32 logits tensor: 2*512*151936*4 = 593.5 MiB; casts/log-probabilities, reference temporaries overlapping policy state, recomputation and CUDA workspaces add more. Reserve allocator/context headroom. The saved planning range was 3-7 GiB, not a bound.

Measured: host layer store 2,620,678,144 bytes; GPU buffers 187,191,296 bytes, agreeing with arithmetic. Training peaks: torch allocated 3.33 GiB, reserved 6.86 GiB; sampled nvidia-smi 7,189 MiB (7.02 GiB). Reserved-minus-allocated includes caching and temporaries with different lifetimes; driver/context and different measurement boundaries also matter. nvidia-smi was sampled every 500 ms, so it may miss shorter peaks. Host peak RSS was not instrumented. Disk also includes HF cache, a regular-file materialization and layer shards (~9.26 GB total weights/copies).

Independent throughput: 246,845 nonpadding pair tokens / 1,125.576 s = 219.31 tok/s, 0.444 pairs/s; prompts count twice, reference-pass compute is not additional tokens. This includes skipped attempts. Full-run TRL throughput agrees. soup profile's 625 tok/s and streaming preflight's 1,852-2,723 tok/s were forecasts, not observations. The earlier early-stopped smoke reported planned rather than actually presented sample throughput.

<!-- PAGE -->
# Verification, failures and release conditions

## 3. Evidence of actual training

A falling loss cannot establish active LoRA, saved-adapter correctness, valid masks or correct gradients; batches and numerical regimes also change loss. Instrumentation checked every final trainer token sequence against chat-template IDs/EOS, all 500 pairs, and asserted 1,089,536 materialized trainable parameters plus exact optimizer membership. Of 63 attempted optimizer steps, 60 changed weights, one had zero LR for warmup and two overflowed. GradScaler skipped both without changing parameters and backed off its scale; subsequent updates were finite. A failed earlier callback stopped before scaling recovery could be observed; its output is retained.

Maximum total parameter change: 2.6479e-4. On three fixed training prompts, adapter-enabled last-token logits changed by up to 0.125595; disabled-base drift and repeat noise were both zero. The frozen reference adapter stayed unchanged. Independent resident reload matched all 112 saved LoRA tensors exactly and reproduced both enabled and disabled streamed logits with observed max error zero under matched FP16 autocast. The disabled-repeat negative control produces no change and fails the effect criterion. Scripts: smoke_dpo.py --full and verify_adapter.py FULL_RUN_DIRECTORY.

Limits: these checks do not establish gradient correctness, generalization, multi-seed robustness, semantic deduplication or broad behavior preservation. Probes cover only three prompts and last-token logits. No resident-versus-streamed backward parity was performed. Saved-weight equality alone is insufficient; it is paired with functional adapter tests.

## 4. Failure audit and Soup coverage

Environment: Colab Python 3.13 was rejected by pip; a separate 3.12 environment fixed setup. Soup doctor later passed. Loud installation failure, not a silent training defect.

Data/API: assistant-authored preparation assumed apply_chat_template returned a list; Transformers returned a dictionary. Zero accepted pairs exposed the error. Explicit return_dict=False and type checks fixed it. A subsequent raw-string conversion would bypass TRL's chat template; source review caught this before training. Conversational prompt/chosen/rejected lists and actual trainer-token equality fixed the mismatch. Doctor on a separate ChatML projection did not establish correctness of DPO's input path.

Checks with limits: doctor scanned 200/1000 dialogs at its default 2048, not configured 512, and warned about missing generation markers. The actual DPO path splits completion IDs at prompt boundaries rather than using doctor's assistant-mask heuristic. Our all-pair audit covered this gap. Lint marked a skipped near-duplicate check OK because datasketch was absent; prompt echo was flagged in 2/500 pairs and remains unadjudicated. Length bias d=0.221 passed its threshold, yet sum-logp accuracy is 27.5% when chosen is longer versus 85.7% otherwise; length remains a major metric confound.

Infrastructure/measurement: Soup rejected symlinked HF cached weights; ordinary copies with SHA-256 equality fixed the loud failure. profile predicted 6,422,528 trainable parameters because it assumes seven target modules, ignoring our two; it also does not model this streaming DPO configuration. Actual inspection confirmed 1,089,536. Initial reload parity failed because the verification omitted autocast; adding a matched-mode comparison achieved exact equality without relaxing tolerances. These were not evidence of a corrupt saved adapter.

## 5. What must change before shipping; reflection and AI use

Before release: establish a relevant, manually reviewed Russian-support evaluation and an independent general-capability/forgetting suite; resolve duplicate/prompt-echo risks; measure improvement with sufficient held-out data and uncertainty across seeds; verify streamed-versus-resident gradients on matched batches. Fix any discrepancies and rerun. More epochs alone would not satisfy these conditions. soup ship consumes supplied metrics; it does not certify adapter activity, domain relevance or backward correctness.

What surprised me: a real adapter update can coexist with completely unchanged held-out decisions, and a legitimate FP16 recovery looked like failure to an over-strict check. I remain concerned about length-sensitive metrics, the narrow evaluation and untested backward parity. AI assistance authored scripts and this report, suggested checks and reviewed source. The candidate executed Colab cells and provided raw outputs; AI suggestions were checked against those outputs and installed code. Rejected/redone assumptions include Python compatibility, tokenizer return type, raw-string DPO formatting, immediate failure on AMP overflow and mismatched autocast verification. No claim of unaided candidate authorship is made; all failed attempts are retained.
