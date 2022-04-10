
import numpy as np
import util_global as ug


def test__two_vectors_same_shape() :
    
    x1 = np.array([1,2,3])
    y1 = np.array([4,5,6])
    n1 = ug.two_vectors_same_shape(x1, y1)
    ug.print_var( "n1" , n1 )
        
    x2 = np.array([1,2,3,4])
    y2 = np.array([4,5,6])
    n2 = ug.two_vectors_same_shape(x2, y2)
    ug.print_var( "n2" , n2 )
        

def test__distance_Thompson_base10() :
    
    base_metric = "Linf"
    
    x1 = np.array([ 1.0 ,  0.1 , 100.0 ])
    y1 = np.array([ 0.1 , 10.0 ,  10.0 ])
    d1 = ug.distance_Thompson_base10( x1 , y1 , base_metric )
    ug.print_var( "d1" , d1 )
    
    x2 = np.array([ 1.0 ,  0.0 , 100.0 ])
    y2 = np.array([ 0.1 , 10.0 ,  10.0 ])
    d2 = ug.distance_Thompson_base10( x2 , y2 , base_metric )
    ug.print_var( "d2" , d2 )
    
    x3 = np.array([ 1.0 , -0.1 , 100.0 ])
    y3 = np.array([ 0.1 , 10.0 ,  10.0 ])
    d3 = ug.distance_Thompson_base10( x3 , y3 , base_metric )
    ug.print_var( "d3" , d3 )
    
    x4 = np.array([ 1.0 ,  0.1 , 100.0 ])
    y4 = np.array([ 0.1 , 10.0 ,  10.0 , 0.01 ])
    d4 = ug.distance_Thompson_base10( x4 , y4 , base_metric )
    ug.print_var( "d4" , d4 )


####################


test__two_vectors_same_shape()

test__distance_Thompson_base10()
    