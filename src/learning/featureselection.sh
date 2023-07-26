nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --fs --model 'xgb' --outerfold 0 &> fs.0.log  &
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --fs --model 'xgb' --outerfold 1 &> fs.1.log &
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --fs --model 'xgb' --outerfold 2  &> fs.2.log &
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --fs --model 'xgb' --outerfold 3 &> fs.3.log &

