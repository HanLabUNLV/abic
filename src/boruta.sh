python src/mira_cross_val_bayescv.eroles.xgb.boruta.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.boruta/run.boruta4 --port 17203 --studyname boruta4 --init
bash opt.sh
bash featureselection.sh
python src/mira_cross_val_bayescv.eroles.xgb.boruta.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.boruta/run.boruta4 --port 17203 --studyname boruta4 --dropfeatures
python src/mira_cross_val_bayescv.eroles.xgb.boruta.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.boruta/run.boruta4 --port 17203 --studyname boruta4 --init2pass
bash opt2pass.sh
python src/mira_cross_val_bayescv.eroles.xgb.boruta.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.boruta/run.boruta4 --port 17203 --studyname boruta4.2pass --test

python -i src/runshap.py --modeldir run.boruta/run.boruta4/ --studyname boruta4.2pass --outdir run.applymodel/boruta4/
