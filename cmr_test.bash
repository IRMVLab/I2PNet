for test in {0..9}; do
	echo $(printf "Start test seed %d" $test)
	CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python evaluation_cmr.py --log_dir log_TRO_kd_cmr2_clip10_continue --gpu 0 --cmr_seed $test --dataset kd_cmr_snr
done
