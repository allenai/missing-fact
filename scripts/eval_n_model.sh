#!/usr/bin/env bash


set -e

t=${NUM-5}
outdir=${OUTDIR-/output}
eval_file=${EVAL-/inputs/test_prefetched_SPAN_PRED.jsonl}

out_suffix=${eval_file//\//_}
file_list=""
for n in `seq 1 $t`; do
    run_dir=${outdir}/run${n}
    output_file=${run_dir}/output_${out_suffix}
    python -m allennlp.run \
        evaluate \
        --output-file ${output_file} \
        --cuda-device 0 \
        --include-package missingfact \
        ${run_dir}/model.tar.gz \
        ${eval_file}
    file_list="${file_list} ${output_file}"
done

grep "choice_accuracy" ${file_list} | cut -f1,3 -d":" > ${outdir}/accuracies_${out_suffix}
