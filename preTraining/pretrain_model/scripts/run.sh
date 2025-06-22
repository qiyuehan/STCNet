# ETTm2
--seq_len 96
--label_len 0
--train_epochs 20
--data_class ETTm
--model_id ETTm2
--data_path ETTm2.csv
--patience 3
--enc_in 7
--dec_in 7
--c_out 7


#ILI
--seq_len 36 --label_len 0 --train_epochs 20 --data_class custom --model_id ILI --data_path ili.csv --patience 3 --enc_in 7 --dec_in 7 --c_out 7 --block_size 12