export HF_DATASETS_CACHE="/data1/"

pkill python

find /dev/shm/* -user elronbandel -exec rm -rf {} \;


python -m torch.distributed.launch --nproc_per_node=4 run_mlm_wwm.py --model_type charconvbert --tokenizer_name tokenizer.json --n_layers 6 --hidden_size 300 --embedding_size 300 --max_position_embeddings 2050 --train_file /home/nlp/amitse/alephbert/data/raw/oscar/he_dedup.txt --output_dir /home/nlp/elronbandel/projects/hebrew-bert/train_charformer/outputs_convbert --do_train --per_device_eval_batch_size 5 --per_device_train_batch_size 2 --dataloader_num_workers 10 --preprocessing_num_workers 10 --cache_dir .cache2 --max_seq_length 2048 --run_name charconvbert --gradient_accumulation_steps 32 --learning_rate 1e-4 --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 --warmup_steps 2000 --save_steps 10 --sharded_ddp "zero_dp_3" --fp16