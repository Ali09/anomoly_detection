import numpy as np
import pdb
import time
from numpy import genfromtxt


def itree(X,ind,depth_vec,depth,d,F):
    #global depth
    #global depth_vec  

    #F = feature importance matrix 
    max_depth = 70
    if depth > max_depth:
        return depth_vec, F

    
    ind_l = []
    ind_r = []
       
    depth+=1    
    
    d_min = np.min(X[ind],axis=0)
    d_max = np.max(X[ind],axis=0)
        
    split_dim = np.random.choice(range(d))   
    #split_val = (np.random.rand()*(d_max[split_dim]-d_min[split_dim]))+d_min[split_dim]
    split_val = np.random.uniform(d_min[split_dim],d_max[split_dim])
    
    
    #print len(ind), split_dim
    
    if d_max[split_dim]==d_min[split_dim]:        
        ind_l = ind[:int(len(ind)/2)]
        ind_r = ind[int(len(ind)/2):]
    else:
        ind_l = [i for i in ind if X[i,split_dim]<=split_val]
        ind_r = [i for i in ind if X[i,split_dim]>split_val]   
        
    if len(ind_l)==0 or len(ind_r)==0:
        ind_l = ind[:int(len(ind)/2)]
        ind_r = ind[int(len(ind)/2):]

    if len(ind_l)==0 or len(ind_r)==0:
        pdb.set_trace()
      
    left_fac = np.max(2*len(ind)/len(ind_l)-1,0)
    right_fac = np.max(2*len(ind)/len(ind_r)-1,0)
    F[ind_l,split_dim] += left_fac
    F[ind_r,split_dim] += right_fac
    #print len(ind_l), len(ind_r)
    
    
    if len(ind_l)>1:
        depth_vec, F = itree(X,ind_l,depth_vec,depth,d,F)             
    else:
        depth_vec[ind_l[0]] = depth
        
    if len(ind_r)>1:
        depth_vec, F  = itree(X,ind_r,depth_vec,depth,d,F)                
    else:
        depth_vec[ind_r[0]] = depth
        
    return depth_vec, F
    

def iforest(X,num_forest,anomaly_list=[]):    
    d = X.shape[1]
    n = X.shape[0]
    m = len(anomaly_list)
    
    
    ind = range(n)

    avg_depth_vec = np.array([0.0]*n)
    avg_F_mat = np.zeros(X.shape)
    
    for i in range(num_forest):
        print "Tree number: ", i, "/", num_forest
        t0=time.clock()
        
        depth_vec = [0]*n
        F = np.zeros(X.shape)
        depth = 0
        
        depth_vec, F = itree(X,ind,depth_vec,depth,d,F)
        
        avg_depth_vec += np.array(depth_vec)        
        avg_F_mat += F
        print "Time taken for ", i, "th tree is:", time.clock()-t0
            
    avg_depth_vec = avg_depth_vec/num_forest        
    avg_F_mat = avg_F_mat/num_forest
    
    
    return avg_depth_vec, avg_F_mat
   
       
    
if __name__ == '__main__':   
    
    my_data = genfromtxt('ahidden_train.txt', delimiter=',')
    
    X = np.transpose(my_data)
    
    #number of forests
    num_forest = 2000
    
    #false alarm rate
    alpha=0.05
    
    #run iForest now and return list of anomalies, depth vector (for all points), and feature importance matrix(for all points) 
    avg_depth_vec, avg_F_mat = iforest(X,num_forest)
    
    N=20    
    
    np.savetxt('depth_val.txt', avg_depth_vec, delimiter=',')
    
    depth_ind = avg_depth_vec.argsort()[:N]

    np.savetxt('depth_val_ind.txt', depth_ind, delimiter=',',  fmt="%i")
    
    depth_ind = avg_depth_vec.argsort()[-N:][::-1]
    
    np.savetxt('depth_val_ind_max.txt', depth_ind, delimiter=',',  fmt="%i")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    