nohup python src/mira_cross_val_bayescv.eroles.xgb.boruta.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.boruta/run.boruta4 --port 17203 --studyname boruta4 --fs --model 'xgb' --outerfold 0 &> fs.0.log  &
nohup python src/mira_cross_val_bayescv.eroles.xgb.boruta.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.boruta/run.boruta4 --port 17203 --studyname boruta4 --fs --model 'xgb' --outerfold 1 &> fs.1.log &
nohup python src/mira_cross_val_bayescv.eroles.xgb.boruta.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.boruta/run.boruta4 --port 17203 --studyname boruta4 --fs --model 'xgb' --outerfold 2  &> fs.2.log &
nohup python src/mira_cross_val_bayescv.eroles.xgb.boruta.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.boruta/run.boruta4 --port 17203 --studyname boruta4 --fs --model 'xgb' --outerfold 3 &> fs.3.log &

