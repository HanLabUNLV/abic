python src/learning/learning.py --dir data/ --outdir run --port 49091 --studyname boruta7 --init
bash opt.sh
bash featureselection.sh
python src/learning/learning.py --dir data/ --outdir run --port 49091 --studyname boruta7 --dropfeatures
python src/learning/learning.py --dir data/ --outdir run --port 49091 --studyname boruta7 --init2pass
bash opt2pass.sh
python src/learning/learning.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --test
bash src/learning/apply_model.sh

python -i src/learning/runshap.py --modeldir run/ --studyname boruta7.2pass --outdir applymodel/
