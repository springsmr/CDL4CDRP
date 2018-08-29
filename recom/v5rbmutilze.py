#-*- coding:utf-8 -*-
# ===============================================================================================
from __future__ import print_function
from basiclib import *
from sklearn.metrics import classification_report,confusion_matrix
# rbm = RBM(input=x, n_visible=WS*self.n_class, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)	
class RBM(object):
	def __init__(
	
                self,
		input=None,
		n_visible=784,
		n_hidden=500,
		W=None,
		hbias=None,
		vbias=None,
		numpy_rng=None,
		theano_rng=None
	):
            self.input = input
            self.n_visible = n_visible
            self.n_hidden = n_hidden
            self.n_class=6
            if numpy_rng is None:
                # create a number generator
                numpy_rng = numpy.random.RandomState(1234)
            if theano_rng is None:
                # numpy.random.RandomState 的符号备份，可以实例化各种分布的随机变量
                theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

	    if W is None:
                initial_W = numpy.asarray(
                        numpy_rng.uniform(
                            low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                            high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                            size=(n_visible, n_hidden)),
                            dtype=theano.config.floatX
                            )
                # theano shared variables for weights and biases
                W = theano.shared(value=initial_W, name='W', borrow=True)

            if hbias is None:
            
                hbias = theano.shared(value=numpy.zeros(n_hidden,dtype=theano.config.floatX),
                            name='hbias',
                
                            borrow=True
                    )


            if vbias is None:
                    vbias = theano.shared(

                            value=numpy.zeros(
                                n_visible,
                            
                                dtype=theano.config.floatX
                            ),
                            name='vbias',
                            borrow=True
                    )
                    
            self.W = W
            self.hbias = hbias
            self.vbias = vbias
            self.theano_rng = theano_rng
            self.params = [self.W, self.hbias, self.vbias]
	#自由能量：F(v)
	def free_energy(self, v_sample):
		# w*v+b
		wx_b = T.dot(v_sample, self.W) + self.hbias
		# v*a
		vbias_term = T.dot(v_sample, self.vbias)
		# sum all hidden node :ln(1+exp(w*v+b))
		hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
		return -hidden_term - vbias_term
	
	#计算隐藏层神经元为1的概率：P(h=1|v)
	def propup(self, vis):
		#计算s型激活函数的x值：w*v+hbias,其中v为每个用户的评分数据
                pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
	
	#由可视层得到：1.s型激活函数的x值：w*v+hbia;2.隐藏层神经元为1的概率：P(h=1|v),即条件分布;3.隐藏成h的值(以P(h=1|v)的分布采样得到)
	def sample_h_given_v(self, v0_sample):
		#计算得到：1.s型激活函数的x值：w*v+hbia;2.隐藏层神经元为1的概率：P(h=1|v)
		pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
		#采样隐藏层h的值：以试验成功p=h1_mean的概率抽样n=1次，返回成功次数；产生数据大小为size=h1_mean.shape;
		h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)
		return pre_sigmoid_h1, h1_mean, h1_sample

	def propdown(self, hid):
		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	#由hidden层得到visible层
	def sample_v_given_h(self, mask, h0_sample):
		'''For recommendation, use softmax instead of sigmoid'''
		#计算得到：激活函数的x值
		pre_activation = T.dot(h0_sample, self.W.T) + self.vbias  # (n_visible, )
		#sz = pre_activation.shape[0]
		#变成：行数：所有电影的项目数，列数：5
		pre_activation = pre_activation.reshape((self.n_visible/self.n_class, self.n_class))
		#得到一个0-4的值，代表每行的哪列位置最大
		state = T.argmax(pre_activation, axis=1)
		#初始化输出结果为：0
		output = T.zeros_like(pre_activation).astype(theano.config.floatX)
		#设置输出state对应部分取1：set_subtensor(x,y),用y填充x，并返回x
		ret = T.set_subtensor(output[T.arange(state.shape[0]), state], 1.0).reshape(mask.shape)
		#乘以掩码，去除对缺失值的预测
		return ret * mask
        def sample_v_all_given_h(self, mask, h0_sample):
            pre_activation = T.dot(h0_sample, self.W.T) + self.vbias  # (n_visible, )
            pre_activation = pre_activation.reshape((self.n_visible/self.n_class, self.n_class))
            state = T.argmax(pre_activation, axis=1)
            output = T.zeros_like(pre_activation).astype(theano.config.floatX)
            ret = T.set_subtensor(output[T.arange(state.shape[0]), state], 1.0).reshape(mask.shape)
            return ret 
        #gibbs采样：h-v-h；返回：1.可视层的采样值；2.隐藏层s型激活函数的x；3.隐藏层神经元为1的概率：P(h=1|v);4.隐藏层的采样值
	def gibbs_hvh(self, h0_sample, mask):
		v1_sample = self.sample_v_given_h(mask, h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]
	#gibbs采样：v-h-v；返回：1.隐藏层的采样值；2.可视层的采样值
	def gibbs_vhv(self, v0_sample, mask):
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		v1_sample = self.sample_v_given_h(mask, h1_sample)
		return [h1_sample, v1_sample]
        def gibbs_vhv_all(self,v0_sample,mask):
            pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
            v1_sample = self.sample_v_all_given_h(mask, h1_sample)
            return v1_sample
	# start-snippet-2
	#cost, updates = rbm.get_cost_updates(mask, lr=lr, persistent=persistent_chain, k=cd_k)
	def get_cost_updates(self, mask, lr=0.1, persistent=None, k=1):
		#计算得到：1.s型激活函数的x值：w*v+hbia;2.隐藏层神经元为1的概率：P(h=1|v),即条件分布;3.隐藏成h的值(以P(h=1|v)的分布采样得到)
		pre_h_mean, h_mean, ph_sample = self.sample_h_given_v(self.input)

		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent
		#循环k步，已知h0，采样v1，由v1采样h1,...
		(
			[
				nv_samples,
				pre_nh_mean, 
				nh_mean,
				nh_samples
			],
			updates
		) = theano.scan(
			self.gibbs_hvh,
			outputs_info=[None, None, None, chain_start],
			n_steps=k,
			non_sequences = [mask],
			name="gibbs_hvh"
		)
		
		#最后一次循环可见层的采样值
		chain_end = nv_samples[-1]
		#损失函数cost：所有用户的自由能F(v)的均值的差；
		cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
		
		#计算梯度：考虑知道chain_end不变。
		gparams = T.grad(cost, self.params, consider_constant=[chain_end])
		#gw = T.dot(self.input.reshape((self.n_visible, 1)), h_mean.reshape((1, self.n_hidden))) - T.dot(chain_end.reshape((self.n_visible, 1)), nh_mean[-1].reshape((1, self.n_hidden)))
		#ghbias = h_mean.reshape(self.hbias.shape) - nh_mean[-1].reshape(self.hbias.shape)
		#gvbias = self.input.reshape(self.vbias.shape) - chain_end.reshape(self.vbias.shape)
		
		#gparams = [gw, ghbias, gvbias]
		#模型参数更新
		for gparam, param in zip(gparams, self.params):
			# make sure that the learning rate is of the right dtype
			updates[param] = param - gparam * T.cast(
				lr,
				dtype=theano.config.floatX
			)
		if persistent:
			updates[persistent] = nh_samples[-1]

		return cost, updates

	def get_pseudo_likelihood_cost(self, updates):
		bit_i_idx = theano.shared(value=0, name='bit_i_idx')
		xi = T.round(self.input)
		fe_xi = self.free_energy(xi)
		xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
		fe_xi_flip = self.free_energy(xi_flip)

		cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
															fe_xi)))

		updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

		return cost

	def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
		cross_entropy = T.mean(
			T.sum(
				self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
				(1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
				axis=1
			)
		)

		return cross_entropy
	#各个用户可视层一次采样前后的错误率	
	def get_reconstruction(self, x, mask):
		[h1_sample, v1_sample] = self.gibbs_vhv(x, mask)
		err = []
		x=x.astype('int8')
		v1_sample=v1_sample.astype('int8')
		return 1.0 - T.mean(T.all(T.eq(x, v1_sample).reshape((self.n_visible/self.n_class, self.n_class)), axis=1))
        def get_reconstruction_visible(self,x,mask):
            [h1_sample, v1_sample] = self.gibbs_vhv(x, mask)
            v1_sample=v1_sample.astype('int8')
            return v1_sample

def train_rbm(fname):
    lr = options["lr"]
    batch_size = options["batch_size"] #块大小为1，一个用户训练数据
    n_hidden = options["n_hidden"]
    n_epoch=options["n_epoch"]
    n_class=options["n_class"]
    #1.nan 变成0;2.min_user_id, max_user_id, min_movie_id, max_movie_id 得到
    path='/home/min/bigdata/data/ex1/data/discrete_data/rbm_datasets'
    train_set=numpy.load(path+'/'+fname)
    prex=prex='/testn_92_'
    testn=92
    if 'chronic' in fname:
        prex='/testn_101_'
        testn=101
    y_real=numpy.load(path+prex+fname)
    
    #print('source data')
    #print(train_set[:,train_set.shape[1]-1].flatten())
    min_user_id=0
    max_user_id,max_movie_id=train_set.shape
    min_movie_id=0
    print(min_user_id, max_user_id, min_movie_id, max_movie_id)
    HS, WS = train_set.shape
    #训练数据的输入：可视层神经元的值
    new_train_set = numpy.zeros((HS, WS*n_class))
    #训练数据的掩码：表示用户对哪些电影有评分，训练时，只用这部分数据；没有的部分表示缺失值，也是我们需要预测的值
    new_train_mask = numpy.zeros((HS, WS*n_class))
    #对原来的评分数据进行编码:3-->0,0,1,0,0
    for row in range(HS):
        for col in range(WS):
            r = int(train_set[row][col]) # (user, movie) = r 
            if r<-0.5:
                continue
            new_train_set[row][col*n_class+r] = 1
            new_train_mask[row][col*n_class:col*n_class+n_class] = 1
			
    print(numpy.mean(new_train_mask))
    #print(new_train_set[:,-n_class:])
    train_set[train_set<0]=0
    new_train_set = new_train_set.astype(theano.config.floatX)
    new_train_mask = new_train_mask.astype(theano.config.floatX)
    #块大小
    n_train_batches = new_train_set.shape[0] // batch_size

    x = T.matrix('x')  # the data is presented as rasterized images
    mask = T.matrix('mask')
    cd_k = T.iscalar('cd_k')
    lr = T.scalar('lr', dtype=theano.config.floatX)
    #设置随机数的种子，使得实验结果可复现
    rng = numpy.random.RandomState(123)
    #theano的随机数
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    #borrow=True 意味着numpy.zeros((batch_size, n_hidden)参数值改变，persistent_chain.get_value() 也会跟着改变
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible=WS*n_class, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)
    rbm.n_class=n_class
    #得到代价函数和参数更新
    cost, updates = rbm.get_cost_updates(mask, lr=lr, persistent=persistent_chain, k=cd_k)

    #训练模型函数：train_model
    train_model = theano.function([x, mask, cd_k, lr], outputs=cost, updates=updates, name='train_rbm')
    #可视层一次采样前后的误差
    check_model = theano.function([x, mask], outputs=rbm.get_reconstruction(x, mask), name='check_model')
    numpy.set_printoptions(threshold='nan') 
    
    output = open("output_persistent_k3_lr0.1.txt", "wb")
    mean_costs=[]
    errors=[]
    precisions=[]
    f1_scores=[]
    rbm_pred={}
	#20次迭代
    print('num of epoch:'+str(n_epoch))
    for epoch in range(n_epoch):
        mean_cost = []
        error = []
        p = []
        #训练数据行数，用户数目
        for batch_index in range(n_train_batches):
            if epoch<3:
                #迭代不到3次采样1次
                cd_k = 1
            else:
                #迭代到3次以后采样次数线性增加
                cd_k = 2 + (epoch - 3)/2
            if epoch<3:
                lr = 0.005
            else:
                #迭代到3次以后学习率线性增加
                lr = 0.005 + ((epoch - 3)/2)*0.01
            #训练数据块大小	
            batch_data = new_train_set[batch_index*batch_size:(batch_index+1)*batch_size]
            #训练数据掩码
            batch_data_mask = new_train_mask[batch_index*batch_size:(batch_index+1)*batch_size]
            #
            mean_cost += [train_model(batch_data, batch_data_mask, cd_k, lr)]
            
            error += [check_model(batch_data, batch_data_mask)]

            p = []
        #代价函数
        print("epoch %d end, cost: %lf"%(epoch, numpy.mean(mean_cost)))
        #实际误差率
        print("epoch %d end, error: %lf"%(epoch, numpy.mean(error)))
        mean_costs.append(numpy.mean(mean_cost))
        errors.append(numpy.mean(error))
        # output the epoch result
        v1sample=rbm.get_reconstruction_visible(x,mask)
        reconf=theano.function([x,mask],v1sample)
        result=[]
        for i in range(max_user_id):
            v0=numpy.copy(new_train_set[i,:]).reshape(1,WS*n_class)
            m0=numpy.copy(new_train_mask[i,:]).reshape(1,WS*n_class)
            v1=reconf(v0,m0)
           # print('v0')
           # print(v0.flatten()[-n_class:])
            label=v1.flatten()[-n_class:]
           # print(label)
            pred=numpy.argmax(label)
           # print(pred)
            result.append(pred)
       # print(result)
        rbm_pred[epoch]=result
        y_pred=numpy.array(result)
       # print(y_pred[-testn:])
       # print(y_real[-testn:])
        report=classification_report(y_pred[-testn:],y_real[-testn:],digits=5)
        lines=report.split('\n')
        line=lines[len(lines)-2]
        row_data = filter(None,line.split(' '))
        precis=float(row_data[3])
        f1=float(row_data[5])
        precisions.append(precis)
        f1_scores.append(f1)
        print('precision%.3f   f1-scores:%.3f'%(precis,f1))
        if(numpy.mean(error)<0.001):
            break

    outname=fname.replace('.npy','')
    outpath='/home/min/bigdata/data/ex1/data/discrete_data/rbm_result/'+outname+'_nhidden_'+str(rbm.n_hidden)
    report_info={}
    report_info['mean_costs']=mean_costs
    report_info['errors']=errors
    report_info['precisions']=precisions
    report_info['f1_scores']=f1_scores
    numpy.save(outpath+'_report',report_info)
    numpy.save(outpath+'_predictions',rbm_pred)


if __name__ == '__main__':
    fnames=['chronic.npy','chronic_missing10.npy','chronic_missing20.npy','chronic_missing30.npy','dermatology.npy','dermatology_missing10.npy','dermatology_missing20.npy','dermatology_missing30.npy']
    for fname in fnames:
        print(fname)
        train_rbm(fname)
        print()
        
