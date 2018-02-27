
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir('G:/Sopranos/ml/Felicity')

train_9=pd.read_csv('train9.csv')
test_9=pd.read_csv('test9.csv')
train_1=pd.read_csv('train1.csv')
test_1=pd.read_csv('test1.csv')
hero=pd.read_csv('hero_data.csv')
train=pd.concat(objs=[train_9,train_1],axis=0)
z=train
prateek=test_1

columns= ['primary_attr','attack_type', 'roles', 'base_health',
       'base_health_regen', 'base_mana', 'base_mana_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence', 'strength_gain',
       'agility_gain', 'intelligence_gain', 'attack_range', 'projectile_speed',
       'attack_rate', 'move_speed', 'turn_rate']
for i in columns:
       mapping = dict(hero[['hero_id', i]].values)
       train[i] = train.hero_id.map(mapping) 
      
train_final=train.iloc[:,[3,6,7,10,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]       
train_final = pd.get_dummies(train_final)
train_dependent=train.iloc[:,[4]]

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=50,criterion='mse')
regressor=regressor.fit(train_final,train_dependent)

z1=test_9.values
for i in columns:
       mapping = dict(hero[['hero_id', i]].values)
       test_9[i] = test_9.hero_id.map(mapping) 
       
test_9_final=test_9.iloc[:,[3,6,7,10,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]       
test_9_final = pd.get_dummies(test_9_final)
test_9_dependent=test_9.iloc[:,[4]]

y=regressor.predict(test_9_final)



for i in columns:
       mapping = dict(hero[['hero_id', i]].values)
       test_1[i] = test_1.hero_id.map(mapping) 
       
test_1_final=test_1.iloc[:,[3,4,5,8,11,13,14,15,16,17,18,19,20,21,22,23,24,25]]       
test_1_final = pd.get_dummies(test_1_final)

column_to_be_added=regressor.predict(test_1_final)

prateek=test_1_final
prateek['num_wins']=column_to_be_added
cols=prateek.columns.tolist()
cols=['num_games',
      'num_wins',
 'base_health_regen',
 'base_armor',
 'base_attack_min',
 'base_attack_max',
 'base_strength',
 'base_agility',
 'base_intelligence',
 'strength_gain',
 'agility_gain',
 'intelligence_gain',
 'attack_range',
 'projectile_speed',
 'attack_rate',
 'move_speed',
 'turn_rate',
 'primary_attr_agi',
 'primary_attr_int',
 'primary_attr_str',
 'attack_type_Melee',
 'attack_type_Ranged']
prateek=prateek[cols]
 




#final work
train_new=z.iloc[:,[3,4,6,7,10,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]
train_new_1 = pd.get_dummies(train_new)
q=z.iloc[:,[5]]
from sklearn.ensemble import RandomForestRegressor
regressor1=RandomForestRegressor(n_estimators=50,criterion='mse')
regressor1=regressor1.fit(train_new_1,q)

final_1=regressor1.predict(prateek)


bhupesh=test_9.iloc[:,[3,4,6,7,10,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]
bhupesh=pd.get_dummies(bhupesh)
final_2=regressor1.predict(bhupesh)

datasetgf=pd.read_csv('test1.csv')
dgf=datasetgf.drop(['user_id','hero_id','num_games'],axis=1)
dgf['kda_ratio']=final_1





dgf.to_csv('G:/Sopranos/ml/ML/Felicity.csv')