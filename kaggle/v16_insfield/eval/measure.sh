#!/bin/zsh
# usage: measure.sh <arm: A|A0|C> <datasets: test,static,dynamic,vl>   (sequential, one process, light)
# Re-measures every image with the new field maps (apo maps, fragments and all geometry rules unchanged) into an
# empty cache dir, then the benchmark comparison through the unchanged Final A recipe vs KG0's maps (N0).
R="<repo>"
E="$R/artifacts/candidates_v16/nextgen-models/eval_$1"
mkdir -p "$E/empty" "$E/cache"
cd "$R"
sed "s/insfield_A/insfield_$1/; s#v16_insfield_A#v16_insfield_$1#; s#eval_A#eval_$1#" kaggle/v16_insfield/eval/config_INS.yaml > "$E/config_$1.yaml"
case $1 in A0) sed -i '' 's/model: fasc_ins_v16}/model: fasc_ins_v16ctl}/' "$E/config_$1.yaml";;
           C) sed -i '' 's/model: fasc_ins_v16}/model: fasc_ins_v16c}/' "$E/config_$1.yaml";; esac
PYTHONPATH=src .venv/bin/python scripts/umud_run.py --config "$E/config_$1.yaml" --stage measure --full --datasets "$2" \
  --set "paths.v3_cache=$E/empty" --set "paths.v4_cache=$E/empty" > "$E/measure_${2//,/_}.log" 2>&1
echo "exit $?" >> "$E/measure_${2//,/_}.log"
# then (after all datasets are measured):
#   .venv/bin/python artifacts/kaggle/native_v1/eval/report/scripts/bench_compare.py "$E/bench/INS" \
#       N0=configs/final_A.yaml V=$E/config_$1.yaml:<measure output cache dir>
