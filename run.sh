#!/bin/bash

source wandb.sh
source vibnet/bin/activate
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-CWRU --classifier RF --dataset CWRU
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-EAS --classifier RF --dataset EAS
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-FEMFTO --classifier RF --dataset FEMFTO
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-IMS --classifier RF --dataset IMS
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-MAFAULDA --classifier RF --dataset MAFAULDA
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-MFPT --classifier RF --dataset MFPT
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-PU --classifier RF --dataset PU
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-RPDBCS --classifier RF --dataset RPDBCS
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-SEU --classifier RF --dataset SEU
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-UOC --classifier RF --dataset UOC
python3 main.py --cfg cfgs/vggish.yaml --run VGGISH-RUN-XJTU --classifier RF --dataset XJTU

#!/bin/bash