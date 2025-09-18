for test in {0..9}; do
	echo $(printf "Start test seed %d" $test)
	CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=3 python evaluation_cmr.py --log_dir log_nus_raw_center_8192_10_2_small_new --gpu 3 --cmr_seed $test --dataset kd_cmr_snr
done
