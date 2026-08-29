#!/usr/bin/env bash
# Stage 0 of the litert migration: the five ImageNet classification backbones, dense only.
# Produces results/classification/litert_stage0.json. Needs a real calibration .npy at $CAL
# -- see that file's `not_established.accuracy` for how the one used was made.
set -uo pipefail
L="$(cd "$(dirname "$0")" && pwd)"
R="${ETPU_REPO:-$(cd "$L/../../.." && pwd)}"
A="${AIEDGE_ENV:?set AIEDGE_ENV to the venv with litert-torch 0.9.3 + ai-edge-quantizer 0.8.0 + torchreid}/bin/python"
W="${WORK:-${TMPDIR:-/tmp}/etpu_stage0}"
CAL=$W/imagenet_from_clipcalib_n100.npy
mkdir -p $W/float $W/int8 $W/etpu $W/logs
S=$W/mapping_summary.tsv
printf 'model\tstatus\ttotal_ops\ttpu_ops\tcpu_ops\tsubgraphs\tonchip\toffchip\tfixups\n' > $S
for n in mobilenetv2_imagenet resnet50 squeezenet1_1 efficientnet_lite0 mnasnet1_0; do
  echo "=== $n"
  $A $L/cls_float.py $R/outputs/models/$n.pt 224 $W/float/$n.tflite > $W/logs/${n}_f.log 2>&1 \
    || { echo "  FLOAT FAILED"; tail -2 $W/logs/${n}_f.log; printf '%s\tfloat_failed\t-\t-\t-\t-\t-\t-\t-\n' "$n" >> $S; continue; }
  CALIB_NPY=$CAL CALIB_N=100 QSV_MINMAX=1 $A $L/aeq_quantize.py $W/float/$n.tflite $W/int8/${n}_raw.tflite \
    > $W/logs/${n}_q.log 2>&1 || { echo "  QUANT FAILED"; tail -2 $W/logs/${n}_q.log; printf '%s\tquant_failed\t-\t-\t-\t-\t-\t-\t-\n' "$n" >> $S; continue; }
  # inline FIRST: etpu_fixups re-serializes and would drop external buffers.
  PYTHONPATH=$R/src $A -m etpu.convert.flatbuffer inline $W/int8/${n}_raw.tflite $W/int8/${n}_inl.tflite > $W/logs/${n}_i.log 2>&1 \
    || { echo "  INLINE FAILED"; continue; }
  fx=$(PYTHONPATH=$R/src $A -m etpu.convert.flatbuffer fixups $W/int8/${n}_inl.tflite $W/int8/${n}_int8.tflite 2>&1 | tee $W/logs/${n}_fx.log | grep -oP '\[fixups\] \K.*(?= ->)')
  log=$(edgetpu_compiler -o $W/etpu -s $W/int8/${n}_int8.tflite 2>&1); echo "$log" > $W/logs/${n}_c.log
  tot=$(grep -oP 'Total number of operations: \K[0-9]+' <<<"$log"|tail -1)
  cpu=$(grep -oP 'run on CPU: \K[0-9]+' <<<"$log"|tail -1)
  tpu=$(grep -oP 'run on Edge TPU: \K[0-9]+' <<<"$log"|tail -1)
  sub=$(grep -oP 'Number of Edge TPU subgraphs: \K[0-9]+' <<<"$log"|tail -1)
  on=$(grep -oP 'On-chip memory used for caching model parameters: \K.*' <<<"$log"|tail -1)
  off=$(grep -oP 'Off-chip memory used for streaming uncached model parameters: \K.*' <<<"$log"|tail -1)
  if [[ -z "${cpu:-}" && "${sub:-0}" == 1 && -n "${tot:-}" ]]; then tpu=$tot; cpu=0; st=ok
  elif [[ -z "${tot:-}" ]]; then st=compile_failed
  else st=cpu_fallback; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$st" "${tot:-?}" "${tpu:-?}" "${cpu:-?}" "${sub:-?}" "${on:-?}" "${off:-?}" "$fx" >> $S
  echo "  -> $st  ${tpu:-?}/${tot:-?} TPU, CPU ${cpu:-?}  on-chip ${on:-?} off-chip ${off:-?}  fixups: $fx"
done
echo; column -t -s $'\t' $S
