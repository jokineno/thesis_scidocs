#/bin/bash 

set -x 

conda create -y --name scidocs python==3.7
conda activate scidocs
conda install -y -q -c conda-forge numpy pandas scikit-learn=0.22.2 jsonlines tqdm sklearn-contrib-lightning pytorch
pip install pytrec_eval awscli allennlp==0.9 overrides==3.1.0
python setup.py install
