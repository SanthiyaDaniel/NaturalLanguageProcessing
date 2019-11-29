from sympy import symbols, diff
def weigts_init():
    a = 1
    b = 1.2
    c = 1.1    
    return a,b,c

#single gate with ouput o = x2 + 4y + z
def forward_propagation(input_,a,b,c):
    x, y, z = input_
    return a*(x**2) + b*(4*y) + c*z
    
def Derivates_wrt_weights():
    a,b,c,x,y,z = symbols('a b c x y z', real=True)
    f= a*(x**2) + b*(4*y) + c*z #Hypothesis function
    return diff(f,x), diff(f,b), diff(f,c)

def Derivates_wrt_function():
    x,y,z = symbols('x y z', real=True)
    f= (x**2) +(4*y) + z
    return diff(f,x), diff(f,y), diff(f,z)

def back_propagation(input_,a,b,c, loss):
    d_a, d_b, d_c = Derivates_wrt_weights()
    alpha = 0.01
    x, y, z = input_
    new_a = a - alpha*(x**2 * loss)#check for multiple variables (deltaJ/deltam == m - (error * X * learnign rate))
    new_b = b - alpha*(4*y * loss)#gradient computation/weight update
    new_c = c - alpha*(z * loss)
    return new_a, new_b, new_c

def main():
    a,b,c = weigts_init()
    dict_weights = {'a':[], 'b':[], 'c':[]} 
    input_ = [[1,2,3],[4,5,6],[3,2,1]]
    output_ = [12, 42, 18]
    pred_list = []
    losses = []
    for epoch in range(30):
        for i,j in enumerate(input_):
            dict_weights['a'].append(a);dict_weights['b'].append(b);dict_weights['c'].append(c)
            pred = (forward_propagation(j,a,b,c))
            pred_list.append(pred)
            loss = (output_[i]-pred)**2
            losses.append(loss)
            a_, b_, c_ = back_propagation(j, a, b,c, loss)
            #dict update        
            a,b,c = a_,b_,c_
            print('Running next sample')
        print('Next epoch')
        print(loss)

if __name__ == '__main__':
    main()
    
    
    