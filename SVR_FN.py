
import pandas as pd 
import numpy as np 
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVR

class SVR_predictor(object):
    
    def __init__(self,df,trainPeriod,cvPeriod,frequency,Target,Features,kernel):
        self.df = df
        self.trainPeriod = trainPeriod
        self.cvPeriod = cvPeriod
        self.frequency = frequency
        self.Target = Target
        self.Features = Features
        self.kernel = kernel
        
    
    def _initialize(self):
        self.Train = pd.DataFrame()
        self.CV = pd.DataFrame()
        self.model = None
        return
        
    def train_cv_split(self):
        self._initialize()
        
        if len(self.cvPeriod)==1:
            self.Train = self.df.loc[self.df.index.strftime(self.frequency).isin(self.trainPeriod)]
            self.CV = self.df.loc[self.df.index.strftime(self.frequency).isin([self.cvPeriod])]
        else:
            self.Train = self.df.loc[self.df.index.strftime(self.frequency).isin(self.trainPeriod)]
            self.CV = self.df.loc[self.df.index.strftime(self.frequency).isin([self.cvPeriod])]
        return
    
    def fit(self):
        X_train = self.Train[self.Features].values
        y_train = self.Train[self.Target].values
        
        if len(y_train.shape) > 1:
            y_train = y_train.ravel()
            
        clf = SVR(kernel = self.kernel)
        clf.fit(X_train,y_train)
        
        self.model = clf
        return 
    
    def predict(self):
        Idx = self.CV.index
        X_cv = self.CV[self.Features].values
        y_cv = self.CV[self.Target].values
        
        y_pred = pd.Series(self.model.predict(X_cv),index = Idx,name = 'estimated')
        Y = pd.concat([self.CV[self.Target],y_pred],axis = 1)
        return Y
    
def forecast_ols_evaluation(Y_cv,Y_Ypred):
    X = sm.add_constant(Y_Ypred)
    y = Y_cv
    mod = sm.OLS(y,X).fit()
    ci = mod.conf_int(alpha=0.05)
    ci.columns = ['lb','ub']

    out = dict(parms = mod.params.to_dict(),
               pvalues = mod.pvalues.to_dict(),
               ci = ci.to_dict())
    return out

def regression_cm(Y):
    signY = np.sign(Y)

    correctNeg = signY.loc[(signY.Target == -1) & (signY.estimated == -1),'Target'].count()
    correctPos = signY.loc[(signY.Target == 1) & (signY.estimated == 1),'Target'].count()
    FalsePos = signY.loc[(signY.Target == -1) & (signY.estimated == 1),'Target'].count()
    FalseNeg = signY.loc[(signY.Target == 1) & (signY.estimated == -1),'Target'].count()
    
    prc = (correctNeg+correctPos)/(correctNeg+correctPos+FalsePos+FalseNeg)
    return prc