/usr/bin/python -c "from src.xmrec_utils.config import *" &&
cd /home/workspace/src/retrieval/lgcn_v1 && /usr/bin/python data_preprocess.py && sh run_lightgcn.sh &&
conda deactivate && conda deactivate &&
cd /home/workspace/src/retrieval/lgcn_v2 && /usr/bin/python wsdm_dataflow.py && sh run_lightgcn.sh &&
cd /home/workspace/src/retrieval/node2vec_v1 && /usr/bin/python i2idash.py &&
cd /home/workspace/src/retrieval/node2vec_v2 && /usr/bin/python i2idash.py &&
cd /home/workspace/src/retrieval/ && bash train_i2i.sh &&
cd /home/workspace/src && bash fix_ggcn_name.sh
