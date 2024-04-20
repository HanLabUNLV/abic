nohup python src/learning/learning.py  --outdir run.full --port 49091 --studyname full --fs --model 'xgb' --outerfold 0 &> fs.0.log  &
nohup python src/learning/learning.py  --outdir run.full --port 49091 --studyname full --fs --model 'xgb' --outerfold 1 &> fs.1.log &
nohup python src/learning/learning.py  --outdir run.full --port 49091 --studyname full --fs --model 'xgb' --outerfold 2  &> fs.2.log &
nohup python src/learning/learning.py  --outdir run.full --port 49091 --studyname full --fs --model 'xgb' --outerfold 3 &> fs.3.log &

