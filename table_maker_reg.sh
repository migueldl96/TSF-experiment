#!/bin/bash

main_dir=$1
deep_dir=$2
model=$3

ar_mse=( $(cat ${main_dir}/*model${model}_comb3*/${deep_dir}/info.json | grep mse -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
dw_mse=( $(cat ${main_dir}/*model${model}_comb5*/${deep_dir}/info.json | grep mse -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )
ardw_mse=( $(cat ${main_dir}/*model${model}_comb1*/${deep_dir}/info.json | grep mse -A 3 | grep test | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | sed 's/\([0-9]\.[0-9]\{4\}\)[0-9]\{1,\}/\1/') )

echo "\begin{tabular}{c|cllcllc|l|l|}
\cline{2-10}
\textbf{}                       & \multicolumn{3}{c|}{AR}        & \multicolumn{3}{c|}{DW}        & \multicolumn{3}{c|}{AR DW}      \\\\ \hline
\multicolumn{1}{|l|}{Ejecución} & \multicolumn{3}{c}{MSE}        & \multicolumn{3}{c}{MSE}        & \multicolumn{3}{c|}{MSE}        \\\\ \hline
\multicolumn{1}{|c|}{1}         & \multicolumn{3}{c}{${ar_mse[0]}} & \multicolumn{3}{c}{${dw_mse[0]}} & \multicolumn{3}{c|}{${ardw_mse[0]}} \\\\
\multicolumn{1}{|c|}{2}         & \multicolumn{3}{c}{${ar_mse[1]}} & \multicolumn{3}{c}{${dw_mse[1]}} & \multicolumn{3}{c|}{${ardw_mse[1]}} \\\\
\multicolumn{1}{|c|}{3}         & \multicolumn{3}{c}{${ar_mse[2]}} & \multicolumn{3}{c}{${dw_mse[2]}} & \multicolumn{3}{c|}{${ardw_mse[2]}} \\\\
\multicolumn{1}{|c|}{4}         & \multicolumn{3}{c}{${ar_mse[3]}} & \multicolumn{3}{c}{${dw_mse[3]}} & \multicolumn{3}{c|}{${ardw_mse[3]}} \\\\
\multicolumn{1}{|c|}{5}         & \multicolumn{3}{c}{${ar_mse[4]}} & \multicolumn{3}{c}{${dw_mse[4]}} & \multicolumn{3}{c|}{${ardw_mse[4]}} \\\\ \hline
\multicolumn{1}{|c|}{Media}     & \multicolumn{3}{l|}{ar\_mean}  & \multicolumn{3}{l|}{dw\_mean}  & \multicolumn{3}{l|}{cc\_mean}   \\\\ \hline
\end{tabular}"