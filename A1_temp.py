def online_BP_algorithm(epochs,trainset_size,n,alpha,s_min,s_max):
    for epoch in range(0,epochs):
        err = aux1 = aux2 =  0
        for i in range(0,trainset_size):
            m = random.randint(0,sys.maxsize)%trainset_size
            nn.xi[0] = dataset.xtable[m]
            feed_forward_propagation()
            error_back_propagation(dataset.ytable[m])
            update_nn( n, alpha )
    
        for i in range(0,trainset_size):
            nn.xi[0] = dataset.xtable[i]
            y_pred = feed_forward_propagation();
            for j in range(0,dataset.no):
                aux1 += abs( descale_y_value( y_pred[j], j, s_min, s_max ) - descale_y_value( dataset.ytable[i][j], j, s_min, s_max ))
                aux2 += descale_y_value( dataset.ytable[i][j], j, s_min, s_max )
                err += pow( (y_pred[j] - dataset.ytable[i][j]), 2)
        err = err/2
        # print("Epoch: {} \tQuadratic error: {}\tPercentage of error: {}".format(epoch, err, (aux1/aux2*100)))


def reset_accum():    
    l=0
    for i in range(0,nn.n[l]):
        for j in range(0,dataset.nf):
            nn.d_w_prev[l][i][j] = 0
        nn.d_theta_prev[l][i] = 0
    
    for l in range(1,nn.L):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.d_w_prev[l][i][j] = 0
            nn.d_theta_prev[l][i] = 0

def batched_error_back_propagation(y):
    l = nn.L-1
    aux = 0
    for i in range(0,nn.n[l]):
        nn.delta[l][i] = compute_derivate_a(nn.h[l+1][i])*(nn.xi[l+1][i] - y[i])

    for l in range(nn.L-2,-1,-1):
        for j in range(0,nn.n[l]):
            aux = 0
            for i in range(0,nn.n[l+1]):
                aux += nn.delta[l+1][i] * nn.w[l+1][i][j]
            nn.delta[l][j] = compute_derivate_a( nn.h[l+1][j] )*aux

    l=0
    for i in range(0,nn.n[l]):
        for j in range(0,dataset.nf):
            nn.d_w_prev[l][i][j] += ( nn.delta[l][i] * nn.xi[l][j] )
        nn.d_theta_prev[l][i] += nn.delta[l][i]

    for l in range(1,nn.L):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.d_w_prev[l][i][j] += ( nn.delta[l][i] * nn.xi[l][j] )
            nn.d_theta_prev[l][i] += nn.delta[l][i]

def batched_update_nn(n,alpha):
    l=0
    for i in range(0,nn.n[l]):
        for j in range(0,dataset.nf):
            nn.d_w[l][i][j] = (-n * nn.d_w_prev[l][i][j] ) + ( alpha * nn.d_w[l][i][j] )
            nn.w[l][i][j] = nn.w[l][i][j] + nn.d_w[l][i][j]
        nn.d_theta[l][i] = (n * nn.d_theta_prev[l][i] ) + ( alpha * nn.d_theta[l][i] )
        nn.theta[l][i] = nn.theta[l][i] + nn.d_theta[l][i]

    for l in range(1,nn.L):
        for i in range(0,nn.n[l]):
            for j in range(0,nn.n[l-1]):
                nn.d_w[l][i][j] = (-n * nn.d_w_prev[l][i][j] ) + ( alpha * nn.d_w[l][i][j] )
                nn.w[l][i][j] = nn.w[l][i][j] + nn.d_w[l][i][j]
            nn.d_theta[l][i] = (n * nn.d_theta_prev[l][i] ) + ( alpha * nn.d_theta[l][i] )
            nn.theta[l][i] = nn.theta[l][i] + nn.d_theta[l][i]

def batched_BP_algorithm(epochs,trainset_size,n,alpha,s_min,s_max):
    for epoch in range(0,epochs):
        err = aux1= aux2=0
        reset_accum()
        for i in range(0,trainset_size):
            m = random.randint(0,sys.maxsize)%trainset_size
            nn.xi[0] = dataset.xtable[m]
            feed_forward_propagation()
            batched_error_back_propagation( dataset.ytable[m] )
        batched_update_nn( n, alpha )
        for i in range(0,trainset_size):
            nn.xi[0] = dataset.xtable[i]
            y_pred = feed_forward_propagation()
            for j in range(0,dataset.no):
                aux1 += abs( descale_y_value( y_pred[j], j, s_min, s_max ) - descale_y_value( dataset.ytable[i][j], j, s_min, s_max ) )
                aux2 += descale_y_value( dataset.ytable[i][j], j, s_min, s_max )
                err += pow( (y_pred[j] - dataset.ytable[i][j]), 2)

        err = err/2
        # print("Epoch: {} \tQuadratic error: {}\tPercentage of error: {}".format(epoch, err, (aux1/aux2*100) ))
        

def batched_CV_BP_algorithm(epochs,trainset_size,n,alpha,s_min,s_max, output_file):
    validationset_size = trainset_size//4
    outFile = open( output_file, "w" )
    for epoch in range(0,epochs):
        err = err_val=0
        reset_accum()
        for i in range(0,trainset_size-validationset_size):
            m = random.randint(0,sys.maxsize)%(trainset_size-validationset_size)
            nn.xi[0] = dataset.xtable[m]
            feed_forward_propagation()
            batched_error_back_propagation( dataset.ytable[m] )
        batched_update_nn( n, alpha )
        for i in range(0,trainset_size-validationset_size):
            nn.xi[0] = dataset.xtable[i]
            y_pred = feed_forward_propagation()
            for j in range(0,dataset.no):
                err += pow( (y_pred[j] - dataset.ytable[i][j]), 2)
        err = err/2
        for i in range(trainset_size-validationset_size,trainset_size):
            nn.xi[0] = dataset.xtable[i]
            y_pred = feed_forward_propagation()
            for j in range(0,dataset.no):
                err_val += pow( (y_pred[j] - dataset.ytable[i][j]), 2)

        err_val = err_val/2
        # print("Epoch: {} \tQuadratic error train set: {}\tQuadratic error validation set: {}".format(epoch, err, err_val ))
        outFile.write("{}, {}, {}\n".format(epoch, err, err_val ))

# online_CV_BP_algorithm( epochs, trainset_size, n, alpha, s_min, s_max, on_cv)
# reset_nn()
# online_BP_algorithm( epochs, trainset_size, n, alpha, s_min, s_max)
# result_online = test_BP_algorithm( trainset_size, s_min, s_max, on_test)
# batched_BP_algorithm( epochs, trainset_size, n, alpha, s_min, s_max)
# reset_nn()
# batched_CV_BP_algorithm( epochs, trainset_size, n, alpha, s_min, s_max, batch_cv)
# reset_nn()