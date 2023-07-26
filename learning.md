# machine learning  
pipeline to predict positive enhancer-promoter pairs.  
  
  
## Data  
  
Data download: [link](https://drive.google.com/drive/folders/1afVv9AaLuRGDwD4U6sgCmkWbdnmuthom?usp=sharing)  
  
## optuna   
  
the pipeline uses optuna for hyperparameter optimization.  
It uses an RDB server to run parallel optimization across multiple processes.  
It needs a running RDB server on the localhost with the port number assigned.   
See the optuna documentation for more information.  
  
[optuna parallelization with an RDB server](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)  
  
  
## initialization  
Sets up the grouped nested cross validation folds.  
Creates the study on the optuna RDB server with the user provided study name  
Here we are using boruta7 as the study name and 49091 as the port number for the RDB server as an example.  
```  
python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --init  
```  
  
## preliminary learning of boosted trees for feature selection  
optimize the hyperparameters for xgboost based on CV.  
optimization is called on each outer folds independently.   
here we are utilizing parallelization and calling 3 processes per fold.  
  
```  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 0 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 0 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 0 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 1 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 1 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 1 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 2  &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 2  &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 2  &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 3 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 3  &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --opt --model 'xgb' --outerfold 3  &> /dev/null &  
```  
  
  
##  feature selection  
we use boruta for feature selection  
  
```  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --fs --model 'xgb' --outerfold 0 &> fs.0.log  &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --fs --model 'xgb' --outerfold 1 &> fs.1.log &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --fs --model 'xgb' --outerfold 2  &> fs.2.log &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --fs --model 'xgb' --outerfold 3 &> fs.3.log &  
```  
  
## drop features determined unimportant in all folds  
```  
python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --dropfeatures  
```  
  
## initialize the 2nd pass learning based on selected features  
creates a new optuna study with the name _[studyname].2pass_
```  
python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7 --init2pass  
```  
  
  
## new model learning based on the selected features  
  
```  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 0 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 0 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 0 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 1 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 1 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 1 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 2  &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 2  &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 2  &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 3 &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 3  &> /dev/null &  
nohup python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --opt --model 'xgb' --outerfold 3  &> /dev/null &  
```  
  
  
## evaluate the learned models on the outer fold test partitions  
```  
python src/learning/mira_cross_val_bayescv.eroles.xgb.py --dir data/ --outdir run --port 49091 --studyname boruta7.2pass --test  
```  
  
  
## apply the saved model to new data  
We first apply the four models trained from the four folds to the test data set aside from Gasperini2019  
  
```  
python src/learning/apply_model.py --outdir applymodel/ --modelfile model/boruta7.2pass.save.0.json --scalerfile model/boruta7.scaler.0.gz  --features data/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.test.txt --targets data/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.test.target.txt  
python src/learning/apply_model.py --outdir applymodel/ --modelfile model/boruta7.2pass.save.1.json --scalerfile model/boruta7.scaler.1.gz  --features data/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.test.txt --targets data/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.test.target.txt  
python src/learning/apply_model.py --outdir applymodel/ --modelfile model/boruta7.2pass.save.2.json --scalerfile model/boruta7.scaler.2.gz  --features data/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.test.txt --targets data/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.test.target.txt  
python src/learning/apply_model.py --outdir applymodel/ --modelfile model/boruta7.2pass.save.3.json --scalerfile model/boruta7.scaler.3.gz  --features data/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.test.txt --targets data/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.test.target.txt  
```  
  
Then we try applying the four models to the independent test data set from Fulco2019  
```  
python src/learning/apply_model.py --outdir applymodel/ --modelfile model/boruta7.2pass.save.0.json --scalerfile model/boruta7.scaler.0.gz  --features data/Fulco2019.CRISPR.ABC.TF.cobinding.txt --targets data/Fulco2019.CRISPR.ABC.TF.cobinding.target.txt  
python src/learning/apply_model.py --outdir applymodel/ --modelfile model/boruta7.2pass.save.1.json --scalerfile model/boruta7.scaler.1.gz  --features data/Fulco2019.CRISPR.ABC.TF.cobinding.txt --targets data/Fulco2019.CRISPR.ABC.TF.cobinding.target.txt  
python src/learning/apply_model.py --outdir applymodel/ --modelfile model/boruta7.2pass.save.2.json --scalerfile model/boruta7.scaler.2.gz  --features data/Fulco2019.CRISPR.ABC.TF.cobinding.txt --targets data/Fulco2019.CRISPR.ABC.TF.cobinding.target.txt  
python src/learning/apply_model.py --outdir applymodel/ --modelfile model/boruta7.2pass.save.3.json --scalerfile model/boruta7.scaler.3.gz  --features data/Fulco2019.CRISPR.ABC.TF.cobinding.txt --targets data/Fulco2019.CRISPR.ABC.TF.cobinding.target.txt  
```   
