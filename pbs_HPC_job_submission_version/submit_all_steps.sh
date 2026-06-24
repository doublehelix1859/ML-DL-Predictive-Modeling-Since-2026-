#!/bin/bash
set -euo pipefail
# Submit sequentially with PBS dependencies.
# Step0 is optional. If DO_PREPROCESS=0, it still runs safely but only writes encoded copies.

j0=$(qsub step0_preprocess.pbs); echo STEP0: $j0
j1=$(qsub -W depend=afterok:$j0 step1_diagnostics.pbs); echo STEP1: $j1
j2=$(qsub -W depend=afterok:$j1 step2_split.pbs); echo STEP2: $j2
j3=$(qsub -W depend=afterok:$j2 step3_regression.pbs); echo STEP3: $j3
j4=$(qsub -W depend=afterok:$j2 step4_classification.pbs); echo STEP4: $j4
j5=$(qsub -W depend=afterok:$j3:$j4 step5_select_top3.pbs); echo STEP5: $j5
j6=$(qsub -W depend=afterok:$j5 step6_shap_training_top3.pbs); echo STEP6: $j6
j7=$(qsub -W depend=afterok:$j5 step7_internal_validation_top3.pbs); echo STEP7: $j7
j8=$(qsub -W depend=afterok:$j7 step8_shap_internal_validation_top3.pbs); echo STEP8: $j8
j9=$(qsub -W depend=afterok:$j6 step9_shap_venn_comparison.pbs); echo STEP9: $j9
j10=$(qsub -W depend=afterok:$j9 step10_external_validation_common_features.pbs); echo STEP10: $j10
