
import numpy as np


def print_var( var_descr , var_value ) :
    print( "{} = {}".format( var_descr , var_value ) )


def two_vectors_same_shape( x , y ) :
    
    shape_x = x.shape
    shape_y = y.shape
    
    if ( ( shape_x == shape_y ) and ( len(shape_x) == 1 ) ) :
        n = shape_x[0]
    else :
        n = None
    
    return n
     

def distance_Thompson_base10( x , y , base_metric ) :
    
    distce = np.NaN
        
    n = two_vectors_same_shape( x , y )
    
    if ( ( n is not None ) and ( n > 0 ) ) :
        
        pos_nan_x = np.isnan(x)
        pos_nan_y = np.isnan(y)
        
        pos_npos_x = x <= 0.0
        pos_npos_y = y <= 0.0
        
        pos_valid = ~ ( pos_nan_x | pos_nan_y | pos_npos_x | pos_npos_y )
        
        nn = sum(pos_valid)
        
        if ( nn > 0 ) :
            
            xx = x[pos_valid]
            yy = y[pos_valid]
            
            if ( np.all( xx > 0.0 ) and np.all( yy > 0.0 ) ) :
                
                z1 = np.true_divide( xx , yy )
                z2 = np.log10(z1)
                z3 = np.abs(z2)
                
                # match base_metric :
                #     case "L1_mean" :
                #         distce = np.sum(z3) / float(nn)
                #     case "Linf":
                #         distce = np.max(z3)
                        
                if ( base_metric == "L1_mean" ) :
                    distce = np.sum(z3) / float(nn)
                elif ( base_metric == "Linf" ) :
                    distce = np.max(z3)
    
    return distce
    