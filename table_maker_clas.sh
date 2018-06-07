#!/bin/bash

main_dir=$1
deep_dir=$2
model=$3

arccdw_ccr=( $(cat ${main_dir}/*model${model}_comb0_seed*/${deep_dir}/info.json | grep ccr -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
ardw_ccr=( $(cat ${main_dir}/*model${model}_comb1_seed*/${deep_dir}/info.json | grep ccr -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
arcc_ccr=( $(cat ${main_dir}/*model${model}_comb2_seed*/${deep_dir}/info.json | grep ccr -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
ar_ccr=( $(cat ${main_dir}/*model${model}_comb3_seed*/${deep_dir}/info.json | grep ccr -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
ccdw_ccr=( $(cat ${main_dir}/*model${model}_comb4_seed*/${deep_dir}/info.json | grep ccr -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
dw_ccr=( $(cat ${main_dir}/*model${model}_comb5_seed*/${deep_dir}/info.json | grep ccr -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
cc_ccr=( $(cat ${main_dir}/*model${model}_comb6_seed*/${deep_dir}/info.json | grep ccr -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )

arccdw_gms=( $(cat ${main_dir}/*model${model}_comb0_seed*/${deep_dir}/info.json | grep gm | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
ardw_gms=( $(cat ${main_dir}/*model${model}_comb1_seed*/${deep_dir}/info.json | grep gm | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
arcc_gms=( $(cat ${main_dir}/*model${model}_comb2_seed*/${deep_dir}/info.json | grep gm | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
ar_gms=( $(cat ${main_dir}/*model${model}_comb3_seed*/${deep_dir}/info.json | grep gm | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
ccdw_gms=( $(cat ${main_dir}/*model${model}_comb4_seed*/${deep_dir}/info.json | grep gm | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
dw_gms=( $(cat ${main_dir}/*model${model}_comb5_seed*/${deep_dir}/info.json | grep gm | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
cc_gms=( $(cat ${main_dir}/*model${model}_comb6_seed*/${deep_dir}/info.json | grep gm | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )

echo "\begin{tabular}{c|clcclccllcllclcccc|l|}
\cline{2-20}
\textbf{}                       & \multicolumn{3}{c|}{AR}                                & \multicolumn{3}{c|}{DW}                                & \multicolumn{3}{c|}{CC}                                & \multicolumn{3}{c|}{AR DW}                               & \multicolumn{2}{c|}{AR CC}                    & \multicolumn{2}{c|}{CC DW}                    & \multicolumn{3}{c|}{AR DW CC}                                \\\\ \hline
\multicolumn{1}{|l|}{Ejecución} & \multicolumn{2}{c}{GMS}        & CCR                   & \multicolumn{2}{c}{GMS}        & CCR                   & GMS                   & \multicolumn{2}{c}{CCR}        & GMS                   & \multicolumn{2}{c}{CCR}          & GMS                   & CCR                   & GMS                   & CCR                   & GMS                    & \multicolumn{2}{c|}{CCR}            \\\\ \hline
\multicolumn{1}{|c|}{1}         & \multicolumn{2}{c}{${ar_gms[0]}} & ${ar_ccr[0]}            & \multicolumn{2}{c}{${dw_gms[0]}} & ${dw_ccr[0]}            & ${cc_gms[0]}            & \multicolumn{2}{l}{${cc_ccr[0]}} & ${ardw_gms[0]}          & \multicolumn{2}{l}{${ardw_ccr[0]}} & ${arcc_gms[0]}          & ${arcc_ccr[0]}          & ${ccdw_gms[0]}          & ${ccdw_ccr[0]}          & ${arccdw_gms[0]}         & \multicolumn{2}{c|}{${arccdw_ccr[0]}} \\\\
\multicolumn{1}{|c|}{2}         & \multicolumn{2}{c}{${ar_gms[1]}} & ${ar_ccr[1]}             & \multicolumn{2}{c}{${dw_gms[1]}} & ${dw_ccr[1]}             & ${cc_gms[1]}            & \multicolumn{2}{l}{${cc_ccr[1]}} & ${ardw_gms[1]}          & \multicolumn{2}{l}{${ardw_ccr[1]}} & ${arcc_gms[1]}          & ${arcc_ccr[1]}          & ${ccdw_gms[1]}          & ${ccdw_ccr[1]}          & ${arccdw_gms[1]}       & \multicolumn{2}{c|}{${arccdw_ccr[1]}} \\\\
\multicolumn{1}{|c|}{3}         & \multicolumn{2}{c}{${ar_gms[2]}} & ${ar_ccr[2]}             & \multicolumn{2}{c}{${dw_gms[2]}} & ${dw_ccr[2]}             & ${cc_gms[2]}            & \multicolumn{2}{l}{${cc_ccr[2]}} & ${ardw_gms[2]}          & \multicolumn{2}{l}{${ardw_ccr[2]}} & ${arcc_gms[2]}          & ${arcc_ccr[2]}          & ${ccdw_gms[2]}          & ${ccdw_ccr[2]}          & ${arccdw_gms[2]}         & \multicolumn{2}{c|}{${arccdw_ccr[2]}} \\\\
\multicolumn{1}{|c|}{4}         & \multicolumn{2}{c}{${ar_gms[3]}} & ${ar_ccr[3]}             & \multicolumn{2}{c}{${dw_gms[3]}} & ${dw_ccr[3]}             & ${cc_gms[3]}            & \multicolumn{2}{l}{${cc_ccr[3]}} & ${ardw_gms[3]}          & \multicolumn{2}{l}{${ardw_ccr[3]}} & ${arcc_gms[3]}         & ${arcc_ccr[3]}          & ${ccdw_gms[3]}          & ${ccdw_ccr[3]}          & ${arccdw_gms[3]}         & \multicolumn{2}{c|}{${arccdw_ccr[3]}} \\\\
\multicolumn{1}{|c|}{5}         & \multicolumn{2}{c}{${ar_gms[4]}} & ${ar_ccr[4]}            & \multicolumn{2}{c}{${dw_gms[4]}} & ${dw_ccr[4]}             & ${cc_gms[4]}            & \multicolumn{2}{l}{${cc_ccr[4]}} & ${ardw_gms[4]}          & \multicolumn{2}{l}{${ardw_ccr[4]}} & ${arcc_gms[4]}          & ${arcc_ccr[4]}         & ${ccdw_gms[4]}          & ${ccdw_ccr[4]}         & ${arccdw_gms[4]}         & \multicolumn{2}{c|}{${arccdw_ccr[4]}} \\\\ \hline
\multicolumn{1}{|l|}{}          & \multicolumn{2}{l|}{}          & \multicolumn{1}{l|}{} & \multicolumn{2}{l|}{}          & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{2}{l|}{}          & \multicolumn{1}{l|}{} & \multicolumn{2}{l|}{}            & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{}  & \multicolumn{2}{l|}{}               \\\\ \hline
\end{tabular}"