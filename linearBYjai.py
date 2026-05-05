

class LinearRegression():
    def fit(self,x,y): ##y is depedent variable $ x is independent variable
        add_x = 0
        add_y = 0
        for i in x:
            add_x = add_x + i
        for i in y:
            add_y = add_y + i
            
        mean_x = add_x / len(x)
        mean_y = add_y / len(y)
        deviation_x = [i - mean_x for i in x]
        deviation_y = [i - mean_y for i in y]
        
        product_deviation = [deviation_x[i]*deviation_y[i] for i in range(len(deviation_x))]
        sum_product_deviation = 0
        for i in product_deviation:
            sum_product_deviation = sum_product_deviation + i
            
        square_devation_x = list(map(lambda x: x*x,deviation_x))
        sum_square_devation_x = 0
        for i in square_devation_x:
            sum_square_devation_x = sum_square_devation_x + i
            
        self.slope = sum_product_deviation/sum_square_devation_x
        self.intercept = mean_y - (self.slope * mean_x)
        
        
    def predict(self, inpt):
        ##Using formula Y = mx + c    where m is slope and c is intercept
        
        return (self.slope*inpt) + self.intercept
        
        
        
            
            
        
        






