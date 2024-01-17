

python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --init  --infile Gasperini2019.bygene.ABC.TF.grouped.train.txt
 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 0 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 0 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 0 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 0 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 1 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 1 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 1 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 1 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 2 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 2 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 2 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --opt --model 'xgb' --outerfold 2 & 

  
python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --test

python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --fs --model 'xgb' --outerfold 0 &> fs.0.log &
python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --fs --model 'xgb' --outerfold 1 &> fs.1.log &
python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --fs --model 'xgb' --outerfold 2 &> fs.2.log &

python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --dropfeatures 


python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes --init2pass 

nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 0 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 0 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 0 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 0 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 1 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 1 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 1 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 1 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 2 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 2 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 2 & 
nohup python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --opt --model 'xgb' --outerfold 2 & 
   


python src/learning/learning_genes.py --dir data/Gasperini/ --outdir run_genes --port 17203 --studyname genes.2pass --test


python src/learning/apply_model.py --outdir apply.genes/ --modelfile run_genes/genes.2pass.save.0.json --scalerfile model/genes.scaler.0.gz  --features data/Gasperini/Gasperini2019.bygene.ABC.TF.grouped.test.txt --targets data/Gasperini/Gasperini2019.bygene.ABC.TF.grouped.test.target.txt  
python src/learning/apply_model.py --outdir apply.genes/ --modelfile run_genes/genes.2pass.save.1.json --scalerfile model/genes.scaler.1.gz  --features data/Gasperini/Gasperini2019.bygene.ABC.TF.grouped.test.txt --targets data/Gasperini/Gasperini2019.bygene.ABC.TF.grouped.test.target.txt  
python src/learning/apply_model.py --outdir apply.genes/ --modelfile run_genes/genes.2pass.save.2.json --scalerfile model/genes.scaler.2.gz  --features data/Gasperini/Gasperini2019.bygene.ABC.TF.grouped.test.txt --targets data/Gasperini/Gasperini2019.bygene.ABC.TF.grouped.test.target.txt  


