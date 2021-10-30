import numpy as np
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    """
    Motion dataset. 
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    def __init__(self, control_data, joint_data, seqlen, n_lookahead, dropout):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        seqlen: number of autoregressive body poses and previous control values
        n_lookahead: number of future control-values
        dropout: (0-1) dropout probability for previous poses
        """
        self.seqlen = seqlen
        self.dropout=dropout
        seqlen_control = seqlen + n_lookahead + 1
        
        #For LSTM network
        n_frames = joint_data.shape[1]
                    
        # Joint positions for n previous frames
        autoreg = self.concat_sequence(self.seqlen, joint_data[:,:n_frames-n_lookahead-1,:])
                    
        # Control for n previous frames + current frame
        control = self.concat_sequence(seqlen_control, control_data)

        # conditioning
        
        print("autoreg:" + str(autoreg.shape))        
        print("control:" + str(control.shape))        
        new_cond = np.concatenate((autoreg,control),axis=2)

        # joint positions for the current frame
        x_start = seqlen
        new_x = self.concat_sequence(1, joint_data[:,x_start:n_frames-n_lookahead,:])
        self.x = new_x
        self.cond = new_cond
        
        #TODO TEMP swap C and T axis to match existing implementation
        self.x = np.swapaxes(self.x, 1, 2)
        self.cond = np.swapaxes(self.cond, 1, 2)
        
        #TODO generate Mask for end-effector
        self.ee_LH_idx=[(16)*3 +0,(16)*3 +1,(16)*3 +2]
        self.ee_RH_idx=[(20)*3 +0,(20)*3 +1,(20)*3 +2]
        self.ee_RF_idx=[(8)*3 +0,(8)*3 +1,(8)*3 +2]
        self.ee_LF_idx=[(4)*3 +0,(4)*3 +1,(4)*3 +2]
        self.ee_HEAD_idx = [(12)*3 +0,(12)*3 +1,(12)*3 +2]

        self.ee_dim = 5*3 # Head, LH, RH, RF, LF 순서로 가자
        self.ee_feature = 3
        
        # input data inside ee_cond 
        self.ee_cond = np.zeros((self.x.shape[0],self.ee_dim,self.x.shape[2]),dtype=np.float32) 
        self.ee_cond[:,:3,:] = self.x[:,self.ee_HEAD_idx,:]
        self.ee_cond[:,(3):(3)+3,:] = self.x[:,self.ee_LH_idx,:]
        self.ee_cond[:,(6):(6)+3,:] = self.x[:,self.ee_RH_idx,:]
        self.ee_cond[:,(9):(9)+3,:] = self.x[:,self.ee_RF_idx,:]
        self.ee_cond[:,(12):(12)+3,:] = self.x[:,self.ee_LF_idx,:]

        print("self.x:" + str(self.x.shape))        
        print("self.cond:" + str(self.cond.shape))
        print("self.ee_cond: " + str(self.ee_cond.shape))
        
    def n_channels(self):
        return self.x.shape[1], self.cond.shape[1]
		
    def concat_sequence(self, seqlen, data):
        """ 
        Concatenates a sequence of features to one.
        """
        nn,n_timesteps,n_feats = data.shape
        L = n_timesteps-(seqlen-1)
        inds = np.zeros((L, seqlen)).astype(int)

        #create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        for ii in range(0,seqlen):  
            inds[:, ii] = np.transpose(rng[ii:(n_timesteps-(seqlen-ii-1))])  

        #slice each sample into L sequences and store as new samples 
        cc=data[:,inds,:].copy()
        
        #print ("cc: " + str(cc.shape))

        #reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen*n_feats))
        #print ("dd: " + str(dd.shape))
        return dd
                                                                                                                               
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """
        
        if self.dropout>0.:
            n_feats, tt = self.x[idx,:,:].shape
            cond_masked = self.cond[idx,:,:].copy()
            
            keep_pose = np.random.rand(self.seqlen, tt)<(1-self.dropout)

            #print(keep_pose)
            n_cond = cond_masked.shape[0]-(n_feats*self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis = 0)
            mask = np.concatenate((mask, mask_cond), axis=0)
            #print(mask)
            cond_masked = cond_masked[(n_feats*self.seqlen):,:]
            # # generate end-effector 
            # ee_cond_masked = self.ee_cond[idx,:,:].copy()
            # keep_ee_cond = np.random.rand(5, tt)<(1-0.5)
            # mask_head = np.repeat(keep_ee_cond[0:1,:],3, axis = 0 )
            # mask_LH = np.repeat(keep_ee_cond[1:2,:],3,axis=0)
            # mask_RH = np.repeat(keep_ee_cond[2:3,:],3,axis=0)
            # mask_RF = np.repeat(keep_ee_cond[3:4,:],3,axis=0)
            # mask_LF = np.repeat(keep_ee_cond[4:5,:],3,axis=0)
            # mask_ee_cond = np.concatenate((mask_head,mask_LH,mask_RH,mask_RF,mask_LF), axis=0)
            # #
           
            #ee_cond_masked = ee_cond_masked * mask_ee_cond
              # generate end-effector 
            ee_cond_masked = self.ee_cond[idx,:,:].copy()
            
            ### masked upper lower 
            p = np.random.rand(1)
            if  p > 0.25 and p < 0.5 :
                ee_cond_masked[-6:,:] =0.0 # lower
            if p >0.5 and p < 0.75 :
                ee_cond_masked[:-6,:] = 0.0 # upper
            if p >0.75 :
                ee_cond_masked[:,:] =0.0 # both

            """ mask joint level
            # keep_ee_cond = np.random.rand(5, 1)<(1-0.5)
            # keep_ee_cond = np.repeat(keep_ee_cond,tt, axis = 1 )
            # mask_head = np.repeat(keep_ee_cond[0:1,:],3, axis = 0 )
            # mask_LH = np.repeat(keep_ee_cond[1:2,:],3,axis=0)
            # mask_RH = np.repeat(keep_ee_cond[2:3,:],3,axis=0)
            # mask_RF = np.repeat(keep_ee_cond[3:4,:],3,axis=0)
            # mask_LF = np.repeat(keep_ee_cond[4:5,:],3,axis=0)
            # mask_ee_cond = np.concatenate((mask_head,mask_LH,mask_RH,mask_RF,mask_LF), axis=0)
            # 
            # ee_cond_masked = ee_cond_masked * mask_ee_cond
            """
           
            
            
            sample = {'x': self.x[idx,:,:], 'cond': cond_masked, 'ee_cond' : ee_cond_masked}
        else:
            sample = {'x': self.x[idx,:,:], 'cond': self.cond[idx,:,:], 'ee_cond' : self.ee_cond[idx,:,:]}
            
        return sample

class TestDataset(Dataset):
    """Test dataset."""

    def __init__(self, control_data, joint_data):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        """        
        # Joint positions
        self.autoreg = joint_data

        # Control
        self.control = control_data
        
    def __len__(self):
        return self.autoreg.shape[0]

    def __getitem__(self, idx):
        sample = {'autoreg': self.autoreg[idx,:], 'control': self.control[idx,:]}
        return sample
