#/bin/bash

python -u main.py --drop_last y \
                --gathered n \
                --loss_balance y \
                --log_to_file n \
                --distributed \
