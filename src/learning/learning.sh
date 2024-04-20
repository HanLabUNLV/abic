
python src/learning/learning.py --dir data/Gasperini/  --outdir run.full --port 49091 --studyname full --init --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.train.dropna.txt
bash src/learning/opt1pass.sh
python src/learning/learning.py  --outdir run.full --port 49091 --studyname full --test &> run.full/test1pass.log
bash src/learning/featureselection.sh
python src/learning/learning.py  --outdir run.full --port 49091 --studyname full --dropfeatures
python src/learning/learning.py  --outdir run.full --port 49091 --studyname full --init2pass
bash src/learning/opt2pass.sh
python src/learning/learning.py  --outdir run.full --port 49091 --studyname full.2pass --test &> run.full/test2pass.log
bash src/learning/apply_model.sh


python src/learning/learning.py --dir data/Gasperini/  --outdir run.reduced --port 49091 --studyname reduced --init_reduced --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.train.dropna.txt
bash src/learning/opt1pass.reduced.sh
python src/learning/learning.py  --outdir run.reduced --port 49091 --studyname reduced --test &> run.reduced/test1pass.log
bash src/learning/apply_model.reduced.sh


python src/learning/runshap.py --modeldir run.full/ --studyname full.2pass --outdir apply.full/shap.full
python src/learning/runshap.py --modeldir run.reduced.atleast1sig/ --studyname reduced.atleast1sig --outdir apply.reduced.atleast1sig/shap.reduced.atleast1sig
python src/learning/runshap.py --modeldir run.reduced/ --studyname reduced --outdir apply.reduced/shap.reduced


python src/learning/plotprcurve.py --traindir run.full --studyname full.2pass --traindir run.reduced --studyname reduced  &> CV.prcurve.log
python src/learning/plotprcurve.py --testdir apply.full/Gasperini/ --studyname full.2pass --testname chr5,10,15,20 --testdir apply.reduced/Gasperini --studyname reduced  &> gasperini.prcurve.log
python src/learning/plotprcurve.py --testdir apply.full/Fulco/ --studyname full.2pass --testname Fulco --testdir apply.reduced/Fulco --studyname reduced  &> fulco.prcurve.log
python src/learning/plotprcurve.py --testdir apply.full/Schraivogel/ --studyname full.2pass --testname Schraivogel --testdir apply.reduced/Schraivogel --studyname reduced  &> schraivogel.prcurve.log

python src/learning/plotprcurve.py --testdir apply.full/Gasperini.atleast1sig/ --studyname full.2pass --testname chr5,10,15,20 --testdir apply.reduced/Gasperini.atleast1sig --studyname reduced  &> gasperini.atleast1sig.prcurve.log
python src/learning/plotprcurve.py --testdir apply.full/Fulco.atleast1sig/ --studyname full.2pass --testname Fulco --testdir apply.reduced/Fulco.atleast1sig --studyname reduced  &> fulco.atleast1sig.prcurve.log
python src/learning/plotprcurve.py --testdir apply.full/Schraivogel.atleast1sig/ --studyname full.2pass --testname Schraivogel --testdir apply.reduced/Schraivogel.atleast1sig --studyname reduced  &> schraivogel.atleast1sig.prcurve.log


