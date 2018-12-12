

"""
Created on Mon Feb  5 17:48:44 2018

@author: Sweta
"""

import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pylab
import pickle

DIR = r'C:\Users\sweta\Desktop\Deep Learning Assignments\BS13B028_ProgrammingAssignment1\Data'

'''Reading datafiles'''

print('Reading data from excel sheet!')

train_data = pd.read_csv(DIR + '/' + str('train.csv'))
test_data = pd.read_csv(DIR + '/'+str('test.csv'))
val_data = pd.read_csv(DIR + '/'+str('val.csv'))
sample_sub = pd.read_csv(DIR + '/'+str('sample_sub.csv'))

''' Training Data'''

#train_data.describe()
print('\nCreating training data...')

train_id = np.array(train_data.iloc[:,0])
train_ground_truth = np.array(train_data.iloc[:,-1])
train_features = np.array(train_data.iloc[:,1:785]).T / 255.0
no_of_classes = np.array(10)

val_id = np.array(val_data.iloc[:,0])
val_ground_truth = np.array(val_data.iloc[:,-1])
val_features = np.array(val_data.iloc[:,1:785]).T / 255.0

test_id = np.array(test_data.iloc[:,0])
test_ground_truth = np.array(sample_sub.iloc[:,-1])
test_features = np.array(test_data.iloc[:,1:785]).T / 255.0

layers = 4
hidden_neurons = 300 
output_neurons = 10
no_classes = np.array(output_neurons)
learning_rate = 0.001
mini_batch_size =55000
max_epochs = 500

def do_adam(train_features, train_ground_truth, val_features, val_ground_truth, test_features, learning_rate, mini_batch_size, layers, hidden_neurons, no_classes ):
    np.random.seed(1234)
    learning_rate = 0.001
    w, b = parameters(train_features, no_classes, layers, hidden_neurons)
    
    log_summary = open(r'C:\Users\sweta\Desktop\Deep Learning Assignments\BS13B028_ProgrammingAssignment1\Code\final code of diff GD\Log_file\{0}_{1}_{2}.csv'.format(layers, hidden_neurons,max_epochs),'w')
    log_summary.write(str('Epoch') + ',' +
                      str('train_ce_loss') + ',' +
                      str('val_ce_loss') + ',' + 
                      str('train_accuracy') + ',' +
                      str('val_accuracy') + ',' +
                      str('learning_rate') + '\n')
    
#    log_summary = open('Automied_Adam_MB_ANL_MB_w_to_test.csv','w')
#    log_summary.write(str('w_to_test') + '\n' +
#                      str('b_to_test') + '\n')
    
    train_indicator = np.zeros((output_neurons,train_features.shape[1]))
    for i in range(train_features.shape[1]):
        train_indicator[train_ground_truth[i], i] = 1
        
    val_indicator = np.zeros((output_neurons,val_features.shape[1]))
    for j in range(val_features.shape[1]):
        val_indicator[val_ground_truth[j], j] = 1
        
    test_indicator = np.zeros((output_neurons,test_features.shape[1]))
    for m in range(test_features.shape[1]):
        test_indicator[test_ground_truth[m], m] = 1

    val_ce_loss_old = 0;
    print('\nTraining Started')
    apnd_val_loss = []
    apnd_train_loss = []
    apnd_train_loss_plt = []
    apnd_val_loss_plt = []
    for epoch in range(max_epochs):
        train_ce_loss = 0
        train_count_cc = 0
        batch_len = int(train_features.shape[1]/mini_batch_size)
        train_data_init = 0
        d_a = []
        d_h = []
        d_w = []
        d_b = []
        m_w = []
        m_b = []
        v_w = []
        v_b = []
        g_dash_a = []
        #print('layers', layers)
        for layer in range(layers+1):
            d_a.append(0)
            d_h.append(0)
            d_w.append(0)
            d_b.append(0)
            m_w.append(0)
            m_b.append(0)
            v_w.append(0)
            v_b.append(0)
            g_dash_a.append(0)
#        print('d_a',np.array(d_a).shape)
#        print('d_h',d_h)
#        print('m_w',m_w)
#        print('m_b',m_b)   
    
        for l in range(batch_len):
            #print(batch_len)
            #print(l)
        #train_input = train_features#[:,train_data]
            train_data_final = train_data_init + mini_batch_size-1
            #print(train_data_init,train_data_final)
            train_input_minibatch = train_features[:,train_data_init:train_data_final+1]
#            print('train_input_minibatch',train_input_minibatch.shape)
            train_ground_truth_minibatch = train_ground_truth[train_data_init:train_data_final+1]
            #print('len',len(train_ground_truth_minibatch))
            train_indicator_minibatch = train_indicator[:,train_data_init:train_data_final+1]
#            print(train_input_minibatch.shape)
#            print(train_ground_truth[0])
#            print(train_ground_truth[999])
#            print(train_ground_truth_minibatch[0])
#            print(train_ground_truth_minibatch[999])
            #if epoch % 10 == 0:
            #    print('epoch', epoch, 'l', l, 'train_init', train_data_init, 'train_final', train_data_final)

            a, h, train_y_hat = forward_propagation(train_input_minibatch, layers, w, b)
            
#            print('w1', w[0].shape)#,'input_features', input_feature.shape)
#            print('w1', w[1].shape)#,'input_features', input_feature.shape)
#            print('w1', w[2].shape)#,'input_features', input_feature.shape)
#            print('w1', w[3].shape)#,'input_features', input_feature.shape)
#            print('w1', w[4].shape)#,'input_features', input_feature.shape)
#            print('w1', w[5].shape)#,'input_features', input_feature.shape)
#            
#            print('a_shape', a[0].shape)
#            print('a_shape', a[1].shape)
#            print('a_shape', a[2].shape)
#            print('a_shape', a[3].shape)
                        
#            print('a_shape', a[4].shape)
#            print('a_shape', a[5].shape)
#            
#            print('h_shape', h[0].shape)
#            print('h_shape', h[1].shape)
#            print('h_shape', h[2].shape)
#            print('h_shape', h[3].shape)
##            print('a_shape', h[4].shape)
#            print('a_shape', h[5].shape)
#            
#            print('Forward Prop done')
            
            for k in range(len(train_ground_truth_minibatch)):
                train_ce_loss += -np.log10(train_y_hat[train_ground_truth_minibatch[k],k])
                if np.argmax(train_indicator_minibatch[:,k]) == np.argmax(train_y_hat[:,k]):
                    train_count_cc+=1
    
            # start a loop

#            print('d_a',np.array(d_a).shape)
#            print('d_h',d_h)
#            print('m_w',m_w)
#            print('m_b',m_b)
            #train_input_mini = train_features[:,[train_data_init, train_data_final]]
       
            d_w, d_b = backward_propagation(train_input_minibatch, train_indicator_minibatch, layers, a, h, w, train_y_hat)
            
            eps, beta1, beta2 = 1e-8, 0.9,0.999
            for i in range(layers+1):
                m_w[i], m_b[i], v_w[i], v_b[i], w[i], b[i], d_w[i], d_b[i] = weight_update_using_adam(i, m_w[i], m_b[i], v_w[i], v_b[i], w[i], b[i], d_w[i], d_b[i], beta1, beta2, eps, learning_rate)
            
            train_data_init = train_data_final+1
            #Ending of the batch loop    
        
        val_ce_loss_new = 0
        val_count = 0
        #val_data = 100
        val_input = val_features#[:,val_data]
        
        val_a, val_h, val_y_hat = forward_propagation(val_input, layers, w, b)
        
        for j in range(len(val_ground_truth)):
            val_ce_loss_new += -np.log10(val_y_hat[val_ground_truth[j],j])
            #Finding the validation accuracy      
            if np.argmax(val_indicator[:,j]) == np.argmax(val_y_hat[:,j]):
                val_count+=1
        val_accuracy = 100 * val_count/val_features.shape[1]
        train_accuracy = 100 * train_count_cc/train_features.shape[1]
            
        
        #Annealing The Learning Rate
#        if epoch != 0:
#            if val_ce_loss_new >= val_ce_loss_old :
#                learning_rate = learning_rate / 2
#        val_ce_loss_old = val_ce_loss_new
            

        #if cross_entropy_loss < 0.000001:
        #train_data_init = train_data_final+1   
        if epoch%10 == 0:
            print('Epoch', epoch, 'train_ce_loss', train_ce_loss,'val_ce_loss', val_ce_loss_new,'train_accuracy', train_accuracy,'val_accuracy', val_accuracy,'learning_rate', learning_rate)#, 'y_hat', y_hat, 'y_hat_sum', sum(y_hat))
            apnd_val_loss_plt.append(val_ce_loss_new)
            apnd_train_loss_plt.append(train_ce_loss)
            #print('Epoch', epoch, 'train_ce_loss',train_ce_loss,'train_accuracy', train_accuracy)#, 'y_hat', y_hat, 'y_hat_sum', sum(y_hat))
        #time.sleep(10)
        log_summary.write(str(epoch) + ',' +
                          str(train_ce_loss) + ',' + 
                          str(val_ce_loss_new) + ',' +
                          str(train_accuracy) + ',' +
                          str(val_accuracy) + ',' +
                          str(learning_rate) + '\n')
        
        
        
        

        # usage
        # w0, w1, w2, b0, b1, b2 are the learned weights and biases
        # Note that the order of variables must be same while saving and loading
#        w_to_test = w
#        b_to_test = b
#        save_weights([w_to_test,b_to_test])
#        w_to_test,b_to_test = load_weights()
#        
        
        
        
        
        apnd_val_loss.append(val_ce_loss_new)
        apnd_train_loss.append(train_ce_loss)
        if epoch == 0:
#            w_to_test = 'w_to_test' + str(epoch)
#            b_to_test = 'b_to_test' + str(epoch)
            w_to_test = w
            b_to_test = b
            #print(w_to_test[0].shape,epoch,apnd_val_loss)
        if epoch > 0 and np.argmin(apnd_val_loss) == epoch:
#            w_to_test = 'w_to_test' + str(epoch)
#            b_to_test = 'b_to_test' + str(epoch)
            w_to_test = w 
            b_to_test = b
            #print(w_to_test[0].shape,epoch,apnd_val_loss)
       # Ending epoch here
#    log_summary.write(str(w_to_test) + '\n' +
#                          str(b_to_test) + '\n')
#    log_summary.close()
       
    log_summary.close()
    print('\nLog File Saved and Closed...')
    print('\nTesting Started...')
    a, h, test_y_hat = forward_propagation(test_features, layers, w_to_test, b_to_test)
    print('Testing done')
    
    test_ce_loss_new = 0
    test_count = 0
    testing_pred = np.zeros((10000,2))

    for tst in range(len(test_ground_truth)):
        test_ce_loss_new += -np.log10(test_y_hat[test_ground_truth[tst],tst])
        testing_pred[tst,0] = tst
        testing_pred[tst,1] = np.argmax(test_y_hat[:,tst])  
        if np.argmax(test_indicator[:,tst]) == np.argmax(test_y_hat[:,tst]):
            test_count+=1
            
    test_accuracy = 100 * test_count/test_features.shape[1]
    
    
    #print('\nTesting Accuracy',test_accuracy)

    return w, b, apnd_train_loss_plt, apnd_val_loss_plt, w_to_test, b_to_test, test_accuracy, testing_pred, test_y_hat


def forward_propagation(input_feature, layers, w, b):
    a = []
    h = []
    for i in range(layers+1):
        a.append(0)
        h.append(0)
        if i == 0:
           a[i] = np.add(np.matmul(w[i].T, input_feature) , b[i])
           h[i] = sigmoid(a[i])
        if i > 0 and i < layers:
           a[i] = np.add(np.matmul(w[i].T, a[i-1]) , b[i])
           h[i] = sigmoid(a[i])
        if i == layers:
           a[i] = np.add(np.matmul(w[i].T, a[i-1]) , b[i])
           h[i] = sigmoid(a[i])  
           y_hat = h[i]

    return a, h, y_hat
    

def backward_propagation(input_feature, indicator, layers, a, h, w, y_hat):
    #Compute gradient with respect to output unit
    d_a = []
    d_h = []
    d_w = []
    d_b = []
    g_dash_a = []
    for i in range((layers+1),0,-1):
        d_a.append(0)
        d_h.append(0)
        d_w.append(0)
        d_b.append(0)
        g_dash_a.append(0)
    
    d_a[layers] = -(indicator - y_hat)
    
    for i in range((layers+1),1,-1):  
        d_w[i-1] = np.matmul(d_a[i-1] , h[i-2].T) 
        d_b[i-1] = d_a[i-1]
        d_b[i-1] = np.reshape(np.sum(d_b[i-1],1),(d_b[i-1].shape[0],1))
        d_h[i-2] = np.matmul(w[i-1] , d_a[i-1])
        g_dash_a[i-2] = np.multiply(sigmoid(a[i-2]) , (1-sigmoid(a[i-2])))
        d_a[i-2] = np.multiply(d_h[i-2], g_dash_a[i-2])

        #if i == 1:
    d_w[0] = np.matmul(d_a[0] , input_feature.T)
    d_b[0] = d_a[0] 
    d_b[0] = np.reshape(np.sum(d_b[0],1),(d_b[0].shape[0],1))
    
#    print('dw', d_w[0].shape)
#    print('dw', d_w[1].shape)
#    print('dw', d_w[2].shape)
#    print('dw', d_w[3].shape)
    
    return d_w, d_b
    #return delta_theta

'''Activation functions: '''        
def softmax(a_L):
    a_M = np.divide(np.exp(a_L), sum(np.exp(a_L))) # Check once
    return a_M

def sigmoid(x):
    y = np.divide(1, (1 + (np.exp(np.multiply(-1,x)))))
    return y
 
def tanh(x):
    y = np.divide((np.exp(x) - np.exp(np.multiply(-1,x))) , (np.exp(x) + np.exp(np.multiply(-1,x))))
    return y
#Learning rate optimization algorithm AdaGrad, RMSProp and Adam 
def weight_update_using_adam(itr, m_w1, m_b1, v_w1, v_b1, w1, b1,dw1, db1, beta1, beta2, eps, eta):
    
    m_w1 = beta1 * m_w1 + (1-beta1) * dw1.T
    m_b1 = beta1 * m_b1 + (1-beta1) * db1
    
    v_w1 = beta2  * v_w1 + (1-beta2) * dw1.T**2
    v_b1 = beta2  * v_b1 + (1-beta2) * db1**2
    
    m_w1 = np.divide(m_w1,(1- math.pow(beta1, itr+1)))
    m_b1 = np.divide(m_b1,(1- math.pow(beta1, itr+1)))

    v_w1 = np.divide(v_w1,(1- math.pow(beta2, itr+1)))
    v_b1 = np.divide(v_b1,(1- math.pow(beta2, itr+1)))  
    
    w1 = w1 - (np.divide(eta,np.sqrt(v_w1 + eps)))* m_w1
    b1 = b1 - (np.divide(eta,np.sqrt(v_b1 + eps))) * m_b1
    return m_w1, m_b1, v_w1, v_b1, w1, b1, dw1, db1

def weight_update_using_AdaGrad(itr, v_w1, v_b1, w1, b1, dw1, db1, eps, eta):
    
    v_w1 = v_w1 +  dw1.T**2
    v_b1 = v_b1 +  db1**2

    w1 = w1 - (eta/np.sqrt(v_w1 + eps)) * dw1
    b1 = b1 - (eta/np.sqrt(v_b1 + eps)) * db1
    return v_w1, v_b1, w1, b1,dw1, db1

def weight_update_using_RMSProp(itr, v_w1, v_b1, w1, b1, dw1, db1, beta1, eps, eta):
    
    v_w1 = beta1  * v_w1 + (1-beta1) * dw1.T**2
    v_b1 = beta1  * v_b1 + (1-beta1) * db1**2

    w1 = w1 - (eta/np.sqrt(v_w1 + eps)) * dw1
    b1 = b1 - (eta/np.sqrt(v_b1 + eps)) * db1
    return v_w1, v_b1, w1, b1,dw1, db1

def save_weights(list_of_weights):
              with open('weights.pkl', 'w') as f:
                     pickle.dump(list_of_weights, f)

def load_weights():
      with open('weights.pkl') as f:
              list_of_weights = pickle.load(f)
      return list_of_weights




def parameters(train_features, no_classes, layers, hidden_neurons):
    np.random.seed(1234)
    w = []
    b = []
    for i in range(layers+1):
        w.append(0)
        b.append(0)
        if i == 0:
           w[i] = np.random.randn(train_features.shape[0], hidden_neurons)/10
           b[i] = np.random.randn(hidden_neurons,1)/10
        if i > 0 and i < layers:
            w[i] = np.random.randn(hidden_neurons, hidden_neurons)/10
            b[i] = np.random.randn(hidden_neurons,1)/10
        if i == layers:
           w[i] = np.random.randn(hidden_neurons, no_classes)/10
           b[i] = np.random.randn(no_classes,1)/10            
    return w, b


w, b, apnd_train_loss_plt, apnd_val_loss_plt, w_to_test, b_to_test, test_accuracy, testing_pred, test_y_hat = do_adam(train_features,train_ground_truth, val_features, val_ground_truth, test_features, learning_rate, mini_batch_size, layers, hidden_neurons, no_classes)

fig = plt.figure()

plt.plot(np.divide(np.array(apnd_train_loss_plt), 55000),'b', np.divide(np.array(apnd_val_loss_plt), 5000),'r')
plt.ylabel('Loss')
plt.xlabel('Epochs')
pylab.savefig(r'C:\Users\sweta\Desktop\Deep Learning Assignments\BS13B028_ProgrammingAssignment1\Code\final code of diff GD\images\{0}_{1}_{2}.png'.format(layers, hidden_neurons, max_epochs))   # save the figure to file
fig.savefig('temp.png', dpi=fig.dpi)
plt.close()  

plt.show()
pd.DataFrame(testing_pred).to_csv(r'C:\Users\sweta\Desktop\Deep Learning Assignments\BS13B028_ProgrammingAssignment1\Code\final code of diff GD\testing_pred\test_predictions_{0}_{1}_{2}.csv'.format(layers, hidden_neurons,max_epochs))
print('layer',layers,'hidden_neurons',hidden_neurons)