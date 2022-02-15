mkdir -p logs &&
/usr/bin/python bsl_v2.py > logs/bsl_v2.log 2>&1
# v3
/usr/bin/python icf_v3.py --output_dirnm icf_v3_001 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1-t2'  --t2_markets 's1-s2-s3-t1-t2' --add_tval 0 > logs/icf_v3_001.log 2>&1 
/usr/bin/python icf_v3.py --output_dirnm icf_v3_002 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1-t2'  --t2_markets 's1-s2-s3-t1-t2' --add_tval 1 > logs/icf_v3_002.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_003 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1'  --t2_markets 's1-s2-s3-t2' --add_tval 0 > logs/icf_v3_003.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_004 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1'  --t2_markets 's1-s2-s3-t2' --add_tval 1 > logs/icf_v3_004.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_005 --sub_dirnm 00_NEW --t1_markets 't1'  --t2_markets 't2' --add_tval 0 > logs/icf_v3_005.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_006 --sub_dirnm 00_NEW --t1_markets 't1'  --t2_markets 't2' --add_tval 1 > logs/icf_v3_006.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_007 --sub_dirnm 00_NEW --t1_markets 't1-t2'  --t2_markets 't1-t2' --add_tval 0 > logs/icf_v3_007.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_008 --sub_dirnm 00_NEW --t1_markets 't1-t2'  --t2_markets 't1-t2' --add_tval 1 > logs/icf_v3_008.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_009 --sub_dirnm 00_NEW --t1_markets 's1-s3-t1'  --t2_markets 's1-s3-t2' --add_tval 0 > logs/icf_v3_009.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_010 --sub_dirnm 00_NEW --t1_markets 's1-s3-t1'  --t2_markets 's1-s3-t2' --add_tval 1 > logs/icf_v3_010.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_011 --sub_dirnm 00_NEW --t1_markets 's1-s2-t1'  --t2_markets 's1-s2-t2' --add_tval 0 > logs/icf_v3_011.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_012 --sub_dirnm 00_NEW --t1_markets 's1-s2-t1'  --t2_markets 's1-s2-t2' --add_tval 1 > logs/icf_v3_012.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_013 --sub_dirnm 00_NEW --t1_markets 's2-s3-t1'  --t2_markets 's2-s3-t2' --add_tval 0 > logs/icf_v3_013.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_014 --sub_dirnm 00_NEW --t1_markets 's2-s3-t1'  --t2_markets 's2-s3-t2' --add_tval 1 > logs/icf_v3_014.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_015 --sub_dirnm 00_NEW --t1_markets 's1-t1'  --t2_markets 's1-t2' --add_tval 0 > logs/icf_v3_015.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_016 --sub_dirnm 00_NEW --t1_markets 's1-t1'  --t2_markets 's1-t2' --add_tval 1 > logs/icf_v3_016.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_017 --sub_dirnm 00_NEW --t1_markets 's2-t1'  --t2_markets 's2-t2' --add_tval 0 > logs/icf_v3_017.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_018 --sub_dirnm 00_NEW --t1_markets 's2-t1'  --t2_markets 's2-t2' --add_tval 1 > logs/icf_v3_018.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_019 --sub_dirnm 00_NEW --t1_markets 's3-t1'  --t2_markets 's3-t2' --add_tval 0 > logs/icf_v3_019.log 2>&1
/usr/bin/python icf_v3.py --output_dirnm icf_v3_020 --sub_dirnm 00_NEW --t1_markets 's3-t1'  --t2_markets 's3-t2' --add_tval 1 > logs/icf_v3_020.log 2>&1
# v2
/usr/bin/python icf_v2.py --output_dirnm icf_v2_001 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1-t2'  --t2_markets 's1-s2-s3-t1-t2' --add_tval 0 > logs/icf_v2_001.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_002 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1-t2'  --t2_markets 's1-s2-s3-t1-t2' --add_tval 1 > logs/icf_v2_002.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_003 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1'  --t2_markets 's1-s2-s3-t2' --add_tval 0 > logs/icf_v2_003.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_004 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1'  --t2_markets 's1-s2-s3-t2' --add_tval 1 > logs/icf_v2_004.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_005 --sub_dirnm 00_NEW --t1_markets 't1'  --t2_markets 't2' --add_tval 0 > logs/icf_v2_005.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_006 --sub_dirnm 00_NEW --t1_markets 't1'  --t2_markets 't2' --add_tval 1 > logs/icf_v2_006.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_007 --sub_dirnm 00_NEW --t1_markets 't1-t2'  --t2_markets 't1-t2' --add_tval 0 > logs/icf_v2_007.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_008 --sub_dirnm 00_NEW --t1_markets 't1-t2'  --t2_markets 't1-t2' --add_tval 1 > logs/icf_v2_008.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_009 --sub_dirnm 00_NEW --t1_markets 's1-s3-t1'  --t2_markets 's1-s3-t2' --add_tval 0 > logs/icf_v2_009.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_010 --sub_dirnm 00_NEW --t1_markets 's1-s3-t1'  --t2_markets 's1-s3-t2' --add_tval 1 > logs/icf_v2_010.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_011 --sub_dirnm 00_NEW --t1_markets 's1-s2-t1'  --t2_markets 's1-s2-t2' --add_tval 0 > logs/icf_v2_011.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_012 --sub_dirnm 00_NEW --t1_markets 's1-s2-t1'  --t2_markets 's1-s2-t2' --add_tval 1 > logs/icf_v2_012.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_013 --sub_dirnm 00_NEW --t1_markets 's2-s3-t1'  --t2_markets 's2-s3-t2' --add_tval 0 > logs/icf_v2_013.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_014 --sub_dirnm 00_NEW --t1_markets 's2-s3-t1'  --t2_markets 's2-s3-t2' --add_tval 1 > logs/icf_v2_014.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_015 --sub_dirnm 00_NEW --t1_markets 's1-t1'  --t2_markets 's1-t2' --add_tval 0 > logs/icf_v2_015.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_016 --sub_dirnm 00_NEW --t1_markets 's1-t1'  --t2_markets 's1-t2' --add_tval 1 > logs/icf_v2_016.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_017 --sub_dirnm 00_NEW --t1_markets 's2-t1'  --t2_markets 's2-t2' --add_tval 0 > logs/icf_v2_017.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_018 --sub_dirnm 00_NEW --t1_markets 's2-t1'  --t2_markets 's2-t2' --add_tval 1 > logs/icf_v2_018.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_019 --sub_dirnm 00_NEW --t1_markets 's3-t1'  --t2_markets 's3-t2' --add_tval 0 > logs/icf_v2_019.log 2>&1
/usr/bin/python icf_v2.py --output_dirnm icf_v2_020 --sub_dirnm 00_NEW --t1_markets 's3-t1'  --t2_markets 's3-t2' --add_tval 1 > logs/icf_v2_020.log 2>&1

# swing
/usr/bin/python swing.py --output_dirnm swing_v2_train_only_t1t2_cross --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1-t2'  --t2_markets 's1-s2-s3-t1-t2' --add_tval 0 > logs/swing_v2_train_only_t1t2_cross.log 2>&1
/usr/bin/python swing.py --output_dirnm swing_v2_train_only_no_t1t2_cross --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1-t2'  --t2_markets 's1-s2-s3-t1-t2' --add_tval 0 --add_crossval 0 > logs/swing_v2_train_only_no_t1t2_cross.log 2>&1

## swing
# /usr/bin/python swing.py --output_dirnm swing_001 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1-t2'  --t2_markets 's1-s2-s3-t1-t2' --add_tval 0 > logs/swing_001.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_002 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1-t2'  --t2_markets 's1-s2-s3-t1-t2' --add_tval 1 > logs/swing_002.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_003 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1'  --t2_markets 's1-s2-s3-t2' --add_tval 0 > logs/swing_003.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_004 --sub_dirnm 00_NEW --t1_markets 's1-s2-s3-t1'  --t2_markets 's1-s2-s3-t2' --add_tval 1 > logs/swing_004.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_005 --sub_dirnm 00_NEW --t1_markets 't1'  --t2_markets 't2' --add_tval 0 > logs/swing_005.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_006 --sub_dirnm 00_NEW --t1_markets 't1'  --t2_markets 't2' --add_tval 1 > logs/swing_006.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_007 --sub_dirnm 00_NEW --t1_markets 't1-t2'  --t2_markets 't1-t2' --add_tval 0 > logs/swing_007.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_008 --sub_dirnm 00_NEW --t1_markets 't1-t2'  --t2_markets 't1-t2' --add_tval 1 > logs/swing_008.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_009 --sub_dirnm 00_NEW --t1_markets 's1-s3-t1'  --t2_markets 's1-s3-t2' --add_tval 0 > logs/swing_009.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_010 --sub_dirnm 00_NEW --t1_markets 's1-s3-t1'  --t2_markets 's1-s3-t2' --add_tval 1 > logs/swing_010.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_011 --sub_dirnm 00_NEW --t1_markets 's1-s2-t1'  --t2_markets 's1-s2-t2' --add_tval 0 > logs/swing_011.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_012 --sub_dirnm 00_NEW --t1_markets 's1-s2-t1'  --t2_markets 's1-s2-t2' --add_tval 1 > logs/swing_012.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_013 --sub_dirnm 00_NEW --t1_markets 's2-s3-t1'  --t2_markets 's2-s3-t2' --add_tval 0 > logs/swing_013.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_014 --sub_dirnm 00_NEW --t1_markets 's2-s3-t1'  --t2_markets 's2-s3-t2' --add_tval 1 > logs/swing_014.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_015 --sub_dirnm 00_NEW --t1_markets 's1-t1'  --t2_markets 's1-t2' --add_tval 0 > logs/swing_015.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_016 --sub_dirnm 00_NEW --t1_markets 's1-t1'  --t2_markets 's1-t2' --add_tval 1 > logs/swing_016.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_017 --sub_dirnm 00_NEW --t1_markets 's2-t1'  --t2_markets 's2-t2' --add_tval 0 > logs/swing_017.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_018 --sub_dirnm 00_NEW --t1_markets 's2-t1'  --t2_markets 's2-t2' --add_tval 1 > logs/swing_018.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_019 --sub_dirnm 00_NEW --t1_markets 's3-t1'  --t2_markets 's3-t2' --add_tval 0 > logs/swing_019.log 2>&1
# /usr/bin/python swing.py --output_dirnm swing_020 --sub_dirnm 00_NEW --t1_markets 's3-t1'  --t2_markets 's3-t2' --add_tval 1 > logs/swing_020.log 2>&1