cd src
# train
python main.py ctdet --exp_id oid_dla_1x --num_epochs 50 --batch_size 128 --lr 5e-4 --num_workers 16
# test
python test.py ctdet --exp_id oid_dla_1x --keep_res --resume
# flip test
python test.py ctdet --exp_id oid_dla_1x --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id oid_dla_1x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
