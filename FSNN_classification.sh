python -u run.py --task_name classification --is_training 1 --root_path ./dataset/EthanolConcentration/  --model_id EthanolConcentration  --model FSNN  --data UEA  --e_layers 5  --batch_size 32  --d_model 32  --d_ff 16  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0

python -u run.py  --task_name classification  --is_training 1  --root_path ./dataset/FaceDetection/  --model_id FaceDetection  --model FSNN  --data UEA  --e_layers 5  --batch_size 32  --d_model 128  --d_ff 64  --top_k 3  --num_kernels 4  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0

python run.py  --task_name classification  --is_training 1  --root_path ./dataset/Handwriting/  --model_id Handwriting  --model  FSNN  --data UEA  --e_layers 2  --batch_size 32  --d_model 72  --d_ff 32  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0 --dropout 0.3

python -u run.py  --task_name classification  --is_training 1  --root_path ./dataset/Heartbeat/  --model_id Heartbeat  --model  FSNN  --data UEA  --e_layers 1  --batch_size 32  --d_model 48  --d_ff 64  --top_k 1  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0 --dropout 0.3

python -u run.py  --task_name classification  --is_training 1  --root_path ./dataset/JapaneseVowels/  --model_id JapaneseVowels  --model  FSNN  --data UEA  --e_layers 4  --batch_size 32  --d_model 20  --d_ff 16  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0 --dropout 0.3

python -u run.py  --task_name classification  --is_training 1  --root_path ./dataset/PEMS-SF/  --model_id PEMS-SF  --model FSNN  --data UEA  --e_layers 3  --batch_size 32  --d_model 8  --d_ff 64  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.01  --train_epochs 5  --patience 20 --num_workers 0 --dropout 0.1

python -u run.py  --task_name classification  --is_training 1  --root_path ./dataset/SelfRegulationSCP1/  --model_id SelfRegulationSCP1  --model  FSNN  --data UEA  --e_layers 1  --batch_size 32  --d_model 16  --d_ff 16  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0 --dropout 0.3

python -u run.py  --task_name classification  --is_training 1  --root_path ./dataset/SelfRegulationSCP2/  --model_id SelfRegulationSCP2  --model  FSNN  --data UEA  --e_layers 1  --batch_size 32  --d_model 64  --d_ff 32  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0 --dropout 0.3

python -u run.py  --task_name classification  --is_training 1  --root_path ./dataset/SpokenArabicDigits/  --model_id SpokenArabicDigits  --model  FSNN  --data UEA  --e_layers 6  --batch_size 64  --d_model 32  --d_ff 32  --top_k 2  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0 --dropout 0.3

python -u run.py  --task_name classification  --is_training 1  --root_path ./dataset/UWaveGestureLibrary/  --model_id UWaveGestureLibrary  --model  FSNN  --data UEA  --e_layers 5  --batch_size 32  --d_model 256  --d_ff 32  --top_k 3  --des 'Exp'  --itr 1  --learning_rate 0.001  --train_epochs 5  --patience 20 --num_workers 0 --dropout 0.3

