python src/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta6 --init
bash opt.sh
bash featureselection.sh
python src/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta6 --dropfeatures
python src/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta6 --init2pass
bash opt2pass.sh
python src/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta6.2pass --test
bash src/apply_model.sh

python -i src/runshap.py --modeldir run/ --studyname boruta6.2pass --outdir applymodel/
