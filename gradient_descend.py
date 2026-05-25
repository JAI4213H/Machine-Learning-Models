import numpy as np
from sklearn.datasets import make_regression, load_diabetes
from sklearn.linear_model import LinearRegression

x,y = load_diabetes(return_X_y=True)


model = LinearRegression()

model.fit(x,y)


        
        
class SingleGradiendDescend:
    def __init__(self,mew,n):
         self.mew = mew
         self.n = n
         self.intercept = 0
         
    def fit(self,x,y):
        slope = 10
        intercept = 0
        
        for i in range(self.n):
            
            
            intercept_slope = -2 * np.sum(y - (slope*x.ravel()) - intercept)
            slope_slope = -2 * np.sum((y - (slope*x.ravel()) - intercept)*x.ravel())
            
            
            intercept = intercept - (self.mew * intercept_slope)
            
            
            slope = slope - (self.mew * slope_slope)
            
        return intercept,slope
    

class MultipleGD:
    def __init__(self,mew,n):
         self.mew = mew
         self.n = n
         
    def fit(self,x,y):
        self.slopes = np.ones(x.shape[1])
        self.intercept = 1
        
        for i in range(self.n):

            y_pred = self.intercept + np.dot(x,self.slopes)

            intercept_slope = -2 * np.mean(y - y_pred)
            slope_slope = (-2/x.shape[0]) * np.dot(x.T, y - y_pred)

            self.intercept = self.intercept - (self.mew * intercept_slope)

            self.slopes = self.slopes - (self.mew * slope_slope)

        print("my:",self.intercept,self.slopes)
        
ad = MultipleGD(0.01, 5000)
ad.fit(x,y)
    
print("\n\nHeres Sklearn Values\n",model.intercept_,model.coef_ )

