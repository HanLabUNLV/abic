python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new --init --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.train.txt
bash src/learning/opt.new.sh
bash src/learning/featureselection.sh
python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new --dropfeatures
python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new --init2pass
bash src/learning/opt2pass.sh
python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new.2pass --test
bash src/learning/apply_model.sh

python src/learning/runshap.py --modeldir run.new/ --studyname new.2pass --outdir apply.new/shap

python src/learning/plotprcurve.py --traindir run.new --testdir apply.new/Gasperini/ --studyname new --testname chr5,10,15,20
python src/learning/plotprcurve.py --traindir run.new --testdir apply.new/Fulco/ --studyname new --testname Fulco
python src/learning/plotprcurve.py  --traindir run.new --testdir apply.new/Shraivogel/ --studyname new --testname Shraivogel

python src/learning/plotprcurve.py --traindir run.new  --testdir apply.new/Gasperini.atleast1sig/ --studyname new --testname chr5,10,15,20
python src/learning/plotprcurve.py --traindir run.new  --testdir apply.new/Fulco.atleast1sig/ --studyname new --testname Fulco
python src/learning/plotprcurve.py --traindir run.new  --testdir apply.new/Shraivogel.atleast1sig/ --studyname new --testname Shraivogel
