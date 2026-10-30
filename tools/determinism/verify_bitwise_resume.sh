#!/bin/bash
# Verify that resuming from a checkpoint is bit-identical to never having stopped.
#
#   MODEL_ARGS="<your training args>" ./verify_bitwise_resume.sh
#
# Runs the same configuration three times and compares checkpoints byte for byte:
#
#   cont   train 0..STEPS in one process
#   save   train 0..RESUME_AT in a fresh process
#   load   resume from save's checkpoint in a third process, train to STEPS
#
#   A/A      cont vs save at iter_RESUME_AT -- both trained from scratch, so they MUST match.
#            If they do not, the config is not even run-to-run reproducible and the resume
#            comparison below would be confounded, so the script stops there.
#   RESUME   cont vs load at iter_STEPS     -- the actual measurement.
#
# WHAT IS COMPARED, AND WHY IT IS A PROXY.
#
# This script compares CHECKPOINTS, not training metrics: sha256 over every .distcp shard of one
# iteration. That is a proxy for "the two runs computed the same thing", and a deliberately strict
# one -- the shard carries the parameters, the fp32 masters, the optimizer state, main_grad and the
# per-rank RNG, so any difference in the forward or backward has to land in one of them by the next
# optimizer step. It needs no tolerance, which is the point: two runs can log an identical `lm loss`
# at print precision while their weights differ in the last bits, so a metric comparison can pass on
# a run this check fails.
#
# The proxy is one-directional. It answers "did the state diverge", not "when" and not "by how
# much". A real training run should ALSO compare the per-step scalars end to end -- `lm loss`,
# `grad norm`, `num-zeros-in-grad` from the training log or TensorBoard -- because those localize
# the first diverging step and show the magnitude, which a pass/fail hash cannot. The two are
# complementary: metrics tell you where to look, the checkpoint hash tells you whether anything
# actually differs. This script ships only the second because it is the one with no tolerance to
# argue about and no parsing to maintain.
#
# Three processes are the point. Resume determinism is a property of a whole process lifetime:
# state that lives in a Python object, is rebuilt at construction, and is never checkpointed is
# invisible to any in-process check, because both sides make the same mistake and agree.
#
# Exit: 0 identical, 1 differs (a RESULT), 8 could not be evaluated (a BROKEN CHECK), 9 A/A failed.
# 1 and 8 are kept apart on purpose -- a comparison that measured nothing must never read as a pass.
#
# See docs/user-guide/deterministic-training.md for what to vary and why.

set -uo pipefail

STEPS=${STEPS:-10}
RESUME_AT=${RESUME_AT:-5}
OUT=${OUT:-./bitwise_resume_$$}
REPO=${MEGATRON_LM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ENTRYPOINT=${ENTRYPOINT:-pretrain_hybrid.py}

# The divergence shows up one step AFTER the resume, not at it, so a one- or two-step window can
# come back clean on genuinely broken code.
[ $((STEPS - RESUME_AT)) -ge 3 ] || { echo "need STEPS-RESUME_AT >= 3" >&2; exit 8; }

# ---- edit these two ---------------------------------------------------------------------------
# The configuration under test. This example is the one the is_first_microbatch bug reproduces in;
# --mtp-use-repeated-layer is what makes that bug destructive rather than merely wrong.
MODEL_ARGS=${MODEL_ARGS:-"--deterministic-mode --bf16 --ckpt-format torch_dist --mtp-use-repeated-layer"}
# How to start a distributed job -- the part that does not generalize across sites.
LAUNCH=${LAUNCH:-"python -u -m torch.distributed.run --nproc-per-node 8"}
# -----------------------------------------------------------------------------------------------

mkdir -p "$OUT" || exit 8
echo "[det] repo=$REPO steps=$STEPS resume_at=$RESUME_AT out=$OUT"

run_leg() {   # $1=tag  $2=exit-interval  $3=load dir ('' for none)
    echo "[det] === leg $1 ($(date '+%H:%M:%S'))"
    # noglob: MODEL_ARGS must word-split, but --hybrid-layer-pattern's value contains '*' and an
    # unquoted expansion also runs PATHNAME expansion -- a matching file in the cwd would silently
    # replace the pattern. An unmatched glob survives intact, so this passes by luck almost
    # everywhere and fails somewhere else.
    set -f
    $LAUNCH "$REPO/$ENTRYPOINT" $MODEL_ARGS \
        --exit-interval "$2" --save-interval "$RESUME_AT" --save-retain-interval "$RESUME_AT" \
        --save "$OUT/ckpt_$1" ${3:+--load "$3"} 2>&1 | tee "$OUT/$1.log"
    local rc=${PIPESTATUS[0]}
    set +f
    echo "[det] leg $1 rc=$rc"
    return $rc
}

# Compare every .distcp shard of one iteration. Rank 0's shard differs BY CONSTRUCTION -- it
# carries the pickled args, and the legs necessarily differ in --exit-interval/--save/--load -- so
# it is reported separately rather than counted as a mismatch or silently ignored.
ckpt_cmp() {  # $1=label  $2=legA  $3=legB  $4=iteration -> 0 same, 1 differs, 8 unevaluable
    local it A B n=0 nd=0 f
    it=$(printf 'iter_%07d' "$4")
    A="$OUT/ckpt_$2/$it"; B="$OUT/ckpt_$3/$it"
    echo
    echo "[det] $1  ($it)"
    [ -d "$A" ] && [ -d "$B" ] || { echo "[det]   NOT EVALUABLE -- missing $A or $B"; return 8; }
    for f in $(cd "$A" && find . -name '*.distcp' | sort); do
        # n counts STATE shards only, so the summary line cannot claim "all N identical" right
        # after reporting that rank 0 differed.
        [ "$f" = ./__0_0.distcp ] || n=$((n + 1))
        if [ "$(sha256sum "$A/$f" | cut -d' ' -f1)" != "$(sha256sum "$B/$f" | cut -d' ' -f1)" ]; then
            case "$f" in
                ./__0_0.distcp) echo "[det]   $f differs (rank 0 = args blob; expected)" ;;
                *) nd=$((nd + 1)); echo "[det]   $f DIFFERS  <-- STATE MISMATCH" ;;
            esac
        fi
    done
    [ "$n" -gt 0 ] || { echo "[det]   NOT EVALUABLE -- no state shards"; return 8; }
    [ "$nd" -eq 0 ] && { echo "[det]   all $n state shards byte-identical"; return 0; }
    echo "[det]   $nd/$n state shards differ"; return 1
}

# Capture the status BEFORE echoing: `exit $?` after an `echo` reads the echo's status, not the
# leg's, so a failed leg would exit 0 and report success for a run that never trained.
run_leg cont "$STEPS" ''
rc=$?
if [ $rc -ne 0 ]; then
    echo "[det] leg cont failed; nothing downstream is comparable. Logs: $OUT/"
    echo "DONE_RC=$rc"; exit $rc
fi
run_leg save "$RESUME_AT" ''
run_leg load "$STEPS"     "$OUT/ckpt_save"

ckpt_cmp "A/A     (cont vs save)" cont save "$RESUME_AT"
case $? in
    0) ;;
    1) echo; echo "[det] ABORT: A/A differs -- this config is not run-to-run reproducible, so the"
       echo "[det]        resume comparison would be confounded. Fix that first."
       echo "DONE_RC=9"; exit 9 ;;
    *) echo; echo "[det] ABORT: A/A could not be compared. Nothing was measured; see $OUT/*.log."
       echo "DONE_RC=8"; exit 8 ;;
esac

ckpt_cmp "RESUME  (cont vs load)" cont load "$STEPS"
rc=$?
echo
echo "DONE_RC=$rc"
exit $rc
