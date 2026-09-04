#!/usr/bin/env bash
# The remaining experiment programme, writing into results/ inside the repository.
#
# Resumable: a run whose summary.json already exists is skipped, so this can be interrupted and
# restarted at any point without losing or repeating work. Results therefore survive a reboot,
# unlike a scratch directory under /tmp.
set -uo pipefail
cd "$(dirname "$0")"
RESULTS="results"
LOGS="$RESULTS/logs"
export TF_CPP_MIN_LOG_LEVEL=2 KAGGLEHUB_DISABLE_PROGRESS_BAR=1
mkdir -p "$LOGS"

run () {   # run <outdir> <label> <args...>
  local out="$1"; local label="$2"; shift 2
  if [ -f "$out/summary.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $label"
    return
  fi
  if timeout 7200 uv run --extra cuda python experiment.py "$@" \
       --num-val-images 200 --max-iter-uni 10 --out "$out" > "$LOGS/${label}.log" 2>&1; then
    grep -E "^[a-z0-9_]+ +[0-9]" "$LOGS/${label}.log" | sed "s/^/[$(date +%H:%M:%S)] DONE /"
  else
    echo "[$(date +%H:%M:%S)] FAILED $label :: $(tail -3 "$LOGS/${label}.log" | tr '\n' ' ' | cut -c1-180)"
  fi
}

refresh () {
  uv run python export_docs_data.py \
    --sweep images="$RESULTS/images" \
    --sweep multiplicity="$RESULTS/multiplicity" \
    --sweep modern="$RESULTS/modern" \
    --out docs/data/results.json >/dev/null 2>&1
}

# Modern architectures first: 24 cheap runs covering the transformer paradigms, ahead of the
# multiplicity tail, where a handful of large-M runs cost ten hours between them and are
# distorted by the known convergence defect anyway.
echo "[$(date +%H:%M:%S)] STAGE A  modern architectures"
for MODEL in vit_b16 swin_tiny deit_b16_distilled convnext_tiny mobilenet_v3_large resnet_vd_50_ssld; do
  for N in 16 64 128 256; do
    run "$RESULTS/modern/$MODEL/$N" "mod_${MODEL}_${N}" --models "$MODEL" --num-images "$N" --search-num 5
  done
  refresh
  echo "[$(date +%H:%M:%S)] STAGE A  $MODEL complete"
done
echo "[$(date +%H:%M:%S)] STAGE A  done"

echo "[$(date +%H:%M:%S)] STAGE B  multiplicity sweep, M=1..20 at n=128"
for MODEL in resnet50 inception5h; do
  for M in $(seq 1 20); do
    run "$RESULTS/multiplicity/$MODEL/$M" "m_${MODEL}_${M}" --models "$MODEL" --num-images 128 --search-num "$M"
  done
  refresh
  echo "[$(date +%H:%M:%S)] STAGE B  $MODEL complete"
done
echo "[$(date +%H:%M:%S)] STAGE B  done"

echo "[$(date +%H:%M:%S)] STAGE C  figures and site data"
uv run python plot_results.py "$RESULTS/images"       --out "$RESULTS/figures" --prefix cnn_    >/dev/null 2>&1
uv run python plot_results.py "$RESULTS/multiplicity" --out "$RESULTS/figures" --prefix m_      >/dev/null 2>&1
uv run python plot_results.py "$RESULTS/modern"       --out "$RESULTS/figures" --prefix modern_ >/dev/null 2>&1
refresh
echo "[$(date +%H:%M:%S)] STAGE C  done: $(ls "$RESULTS/figures" | wc -l) figures"
echo "PIPELINE COMPLETE"
