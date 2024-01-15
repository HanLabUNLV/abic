python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new --init --infile Gasperini2019.at_scale.ABC.TF.erole.grouped.train.txt
bash src/learning/opt1pass.sh
python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new --test &> run.new/test1pass.log
bash src/learning/featureselection.sh
python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new --dropfeatures
python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new --init2pass
bash src/learning/opt2pass.sh
python src/learning/learning.py --dir data/Gasperini/ --outdir run.new --port 17203 --studyname new.2pass --test &> run.new/test2pass.log
bash src/learning/apply_model.sh

python src/learning/runshap.py --modeldir run.new/ --studyname new.2pass --outdir apply.new/shap.new
python src/learning/runshap.py --modeldir run.noTF.atleast1sig/ --studyname noTF.atleast1sig --outdir apply.noTF.atleast1sig/shap.noTF.atleast1sig
python src/learning/runshap.py --modeldir run.noTF/ --studyname noTF --outdir apply.noTF/shap.noTF


python src/learning/plotprcurve.py --traindir run.new --studyname new.2pass --traindir run.noTF --studyname noTF 
python src/learning/plotprcurve.py --testdir apply.new/Gasperini/ --studyname new.2pass --testname chr5,10,15,20 --testdir apply.noTF/Gasperini --studyname noTF 
python src/learning/plotprcurve.py --testdir apply.new/Fulco/ --studyname new.2pass --testname Fulco --testdir apply.noTF/Fulco --studyname noTF 
python src/learning/plotprcurve.py --testdir apply.new/Shraivogel/ --studyname new.2pass --testname Shraivogel --testdir apply.noTF/Shraivogel --studyname noTF 

python src/learning/plotprcurve.py --testdir apply.new/Gasperini.atleast1sig/ --studyname new.2pass --testname chr5,10,15,20 --testdir apply.noTF.atleast1sig/Gasperini.atleast1sig --studyname noTF.atleast1sig 
python src/learning/plotprcurve.py --testdir apply.new/Fulco.atleast1sig/ --studyname new.2pass --testname Fulco --testdir apply.noTF.atleast1sig/Fulco.atleast1sig --studyname noTF.atleast1sig 
python src/learning/plotprcurve.py --testdir apply.new/Shraivogel.atleast1sig/ --studyname new.2pass --testname Shraivogel --testdir apply.noTF.atleast1sig/Shraivogel.atleast1sig --studyname noTF.atleast1sig 


