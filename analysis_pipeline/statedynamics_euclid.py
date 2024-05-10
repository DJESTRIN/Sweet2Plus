import numpy as np
from math import dist as dis


aucsoh=np.asarray(primary_obj.recordings[subjectnumber].auc_vals)
water_auc_vals=aucsoh[:,0]
vanilla_auc_vals=aucsoh[:,1]
peanut_auc_vals=aucsoh[:,2]
tmt_auc_vals=aucsoh[:,3]

water_tmt_d=dis(water_auc_vals,tmt_auc_vals)
vanilla_tmt_d=dis(vanilla_auc_vals,tmt_auc_vals)
peanut_tmt_d=dis(peanut_auc_vals,tmt_auc_vals)
