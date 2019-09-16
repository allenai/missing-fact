#!/usr/bin/env bash

set -e

t=${NUM-5}
outdir=${OUTDIR-/output}
config=${CONF-/config.json}
cuda_device=-1
file_list=""
for n in `seq 1 $t`; do
    run_dir=${outdir}/run${n}
    if [[ -d "$run_dir" ]]; then
        read -p "Directory: ${run_dir} already exists! Delete and continue training ? [Y/n]:" response
        if [[ "$response" == "Y" ]]; then
          rm -rf ${run_dir}
        else
          exit
        fi
    fi
	mkdir -p ${run_dir}
	seed=`expr $n \* 1337`
	python -m allennlp.run \
	 train  \
	 -o "{\"pytorch_seed\": $seed, \"numpy_seed\": $seed, \"random_seed\": $seed, \"trainer.cuda_device\": ${cuda_device}}" \
	 -s ${run_dir} \
	 --file-friendly-logging \
	 --include-package missingfact \
	${config}

	 file_list="${file_list}${run_dir}/metrics.json;"
done

python scripts/merge_metrics.py ${file_list} ${outdir}/metrics.json
