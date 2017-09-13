#Train Automation


#ours wout reg
python train.py -lr 0.0001  -load 0 -aux 0.0 -ds -2 -bn 0 -dr 0 -fs 8
python train.py -lr 0.00001 -load 1 -aux 0.0 -ds -2 -bn 0 -dr 0 -fs 8

#ours wout bn
python train.py -lr 0.0001  -load 0 -aux 0.5 -ds 2 -bn 0 -dr 1 -fs 8
python train.py -lr 0.00001 -load 1 -aux 0.5 -ds 2 -bn 0 -dr 1 -fs 8

#ours wout dropout
python train.py -lr 0.0001  -load 0 -aux 0.5 -ds 2 -bn 1 -dr 0 -fs 16
python train.py -lr 0.00001 -load 1 -aux 0.5 -ds 2 -bn 1 -dr 0 -fs 16

#ours wout ds 
python train.py -lr 0.0001  -load 0 -aux 0.5 -ds -2 -bn 1 -dr 1 -fs 16
python train.py -lr 0.00001 -load 1 -aux 0.5 -ds -2 -bn 1 -dr 1 -fs 16

#ours wout mtl
python train.py -lr 0.0001  -load 0 -aux 0.0 -ds 2 -bn 1 -dr 1 -fs 16
python train.py -lr 0.00001 -load 1 -aux 0.0 -ds 2 -bn 1 -dr 1 -fs 16


#ours
python train.py -lr 0.0001  -load 0 -aux 0.5 -ds 2 -bn 1 -dr 1 -fs 16
python train.py -lr 0.00001 -load 1 -aux 0.5 -ds 2 -bn 1 -dr 1 -fs 16


