from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, auc#, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_raw = pd.read_excel('SIP_June2020_Rik.xlsx')
data_raw = data_raw.drop(['BirthSequence', 'AdmittingDisposition','ROM', 'TimingofRupture', 'IUG', 'PlaceOfBirth', 
                          'CpHVenus', 'CPHArterial','SGA-Y-N', 'WeightPcnt', 'LengthPcnt', 'HeadCircPcnt', 
                          'AntenatalSteroids', 'bowel perforation', 'intestinal perforation', 'Perforation', 
                          'Final Status', 'QtrYear', ' NEC episode I', 'NEC Stage', 'quarter of NEC episode I', 
                          'Qt2vs Others', '#Antibiotics', '#Days Antibiotics', 'rop stage', 'GA_By_DatesWeeks.1',
                          'Database'], axis=1) # remove super-sparse columns
data_raw = data_raw[data_raw["Sex"]!="A"] # remove gender minority for better one-hot labeling
print(data_raw.shape)
sparse_features = ["Length", "FOC", "rop", "rop severe", "laser ROP"]#["GA nom", "GA", "Cord pH Arterial", "Cord pH Venous"]
data_raw = data_raw.drop(sparse_features, axis=1)
print(data_raw.shape)
data_raw.head()

print(data_raw.shape)
data_raw.fillna({'Race':'Others'}, inplace=True)
data_raw.fillna({'ivh severe':0}, inplace=True)
data_raw.fillna({'SGA':0}, inplace=True)

cleanup_nums = {'Sex': {'M': 0, "F": 1}, 'SGA': {'yes':1, "no":0}}

data_raw.replace(cleanup_nums, inplace=True)
data_raw['Race'] = data_raw['Race'].map({'O':'O', 'W':'W', 'B':'B', 'A':'A', 
                       'N':'Others', 1:'Others', 'U':'U', 'I':'Others', 
                       6:'Others', 5:'Others', 2:'Others', 0:'Others', 'Others':'Others'})

print(data_raw.shape)
cols = ["Race"]
for col in cols:
    data_raw[col] = data_raw[col].astype('category')
    data_raw = pd.get_dummies(data_raw, columns=[col])
print(data_raw.shape)
print("Before drop rows containing NAs: ", data_raw.shape, data_raw["spontaneous perforation"].sum())
data = data_raw.dropna()
print("After drop rows containing NAs: ", data.shape, data["spontaneous perforation"].sum())

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

data = data_raw.dropna()
data = data[data["Apgar_1"]!=("ND        ")]
SIP = data['spontaneous perforation']
data.drop(labels=['spontaneous perforation'], axis=1,inplace = True)
data.insert(0, 'SIP', SIP)
calculate_pvalues(data)

y = data['SIP'].values

features = np.array(['GA_By_DatesWeeks', 'BirthWeight', 'Multiple_gest', 'Sex', 'Apgar_5', 'PC-Preterm_Labor', 'ivh severe', 'Race_A', 'Race_B', 'Race_O', 
                     'Race_Others', 'Race_U', 'Race_W'])
X = data[features].values
#features = data.drop(['NEC episode I'], axis=1).columns
print("shape of X:", X.shape, "shape of y", y.shape) 
X = X.astype(float)
y = y.astype(float)

app = Flask(__name__)
objapi = Api(app)

scaler = MinMaxScaler()
clf = LogisticRegression(class_weight={1:25, 0:1}, solver='lbfgs')
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    scaler.fit(X[train])
    clf.fit(scaler.transform(X[train]), y[train])
    Y_test_pred = clf.predict(scaler.transform(X[test]))
    test_f1 = f1_score(y[test], Y_test_pred)
    test_acc = accuracy_score(y[test], Y_test_pred)
    #test_auc = auc(y[test], Y_test_pred)
    #test_accb = balanced_accuracy_score(y[test], Y_test_pred)
    print("F1 Score", test_f1, "Accuracy", test_acc)#, "Auc", test_auc)#, "Balanced Accuracy", test_accb)
    #print(np.argsort(np.abs(np.std(scaler.transform(X[train]), 0)*clf.coef_[0]))[::-1])
    print(clf.coef_[0])
    print(np.abs(np.std(scaler.transform(X[train]), 0)*clf.coef_[0]))
    print("Feature Importance:", features[np.argsort(np.abs(np.std(scaler.transform(X[train]), 0)*clf.coef_[0]))][::-1])



class getData(Resource):
    def get(self, intGA, intBW, strMG, strSex, intAPG5, strPC, strIVH, strRace):

        #[28, 1240, 1.0, 0, 8, 0, 0, 0.0, 0, 0, 0, 0, 0, 1]

        if(strMG == "yes"):
            intMG = 1.0
        else:
            intMG = 0.0
        
        if(strSex == "male"):
            intSex = 0
        else:
            intSex = 1

        if(strPC == "yes"):
            intPC = 1
        else:
            intPC = 0
            
        if(strIVH == "yes"):
            intIVH = 1.0
        else:
            intIVH = 0.0

        raceA = 0
        raceB = 0
        raceO = 0
        raceOthers = 0
        raceU = 0
        raceW = 0

        if(strRace == "A"):
            raceA = 1
        elif(strRace == "B"):
            raceB = 1
        elif(strRace == "O"):
            raceO = 1
        elif(strRace == "U"):
            raceU = 1
        elif(strRace == "W"):
            raceW = 1
        else:
            raceOthers = 1

        print(intGA)
        print(intBW)
        print(intMG)
        print(intSex)
        print(intAPG5)
        print(intPC)
        print(intIVH)
        print(raceA)
        print(raceB)
        print(raceO)
        print(raceOthers)
        print(raceU)
        print(raceW)
        
        arr = [[intGA, intBW, intMG, intSex, intAPG5, intPC, intIVH, raceA, raceB, raceO, 
                     raceOthers, raceU, raceW]]
        
        Y_pred = clf.predict_proba(scaler.transform(arr))
        
        userData = {}
        print(Y_pred)
            
        userData["Score"] = Y_pred[0][0]


        response = jsonify(userData)
        response.headers.add('Access-Control-Allow-Origin', '*')    
        
        return response


objapi.add_resource(getData, "/getData/<intGA>/<intBW>/<strMG>/<strSex>/<intAPG5>/<strPC>/<strIVH>/<strRace>")

#app.run(debug=True)
app.run(host='0.0.0.0', port=5000)
