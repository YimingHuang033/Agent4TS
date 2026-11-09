"""Adapter for the APN model shipped in the locked BITS source snapshot.

Imports only the torch model, not the incompatible BITS runtime. Prediction
passes query timestamps and zero placeholders, never future labels/masks.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .base import Forecaster

class APN(Forecaster):
    model_id = 'APN'
    def __init__(self, source, seed=0, epochs=10, model_id="APN", model_params=None):
        self.model_params = model_params or {}
        self.model_id = model_id
        self.source, self.seed, self.epochs = str(source), seed, epochs
    def _construct(self):
        spec = importlib.util.spec_from_file_location('agent4ts_upstream_apn', self.source)
        module = importlib.util.module_from_spec(spec)
        if self.model_id == "HyperIMTS":
            code = Path(self.source).read_text().replace("from utils.ExpConfigs import ExpConfigs", "from types import SimpleNamespace as ExpConfigs").replace("from utils.globals import logger", "import logging\nlogger = logging.getLogger('HyperIMTS')")
            exec(compile(code, self.source, "exec"), module.__dict__)
        elif self.model_id == "tPatchGNN":
            code = Path(self.source).read_text().replace(".cuda()", ".to(self.device)")
            exec(compile(code, self.source, "exec"), module.__dict__)
        else:
            spec.loader.exec_module(module)
        if self.model_id == 'APN':
            return module.Model(SimpleNamespace(**self.params), device=torch.device(self.device)).to(self.device)
        params = {**self.params, "device": torch.device(self.device)}
        constructor = module.KAFNet if self.model_id == "KAFNet" else module.Model
        return constructor(SimpleNamespace(**params)).to(self.device)
    def _inputs(self, lw):
        w=lw.window
        n=self.params['enc_in']; length=len(w.times)
        x=np.zeros((length,n),dtype='float32'); mask=np.zeros_like(x)
        for j,vid in enumerate(w.variable_ids):
            x[:,vid]=np.nan_to_num(w.x[j]);mask[:,vid]=w.mask[j]
        if getattr(self._view, "age_feature", False):
            for j,vid in enumerate(w.variable_ids):
                x[:,self.n_original+vid] = np.clip(np.nan_to_num(w.age[j],nan=self.params["history"],posinf=self.params["history"])/self.params["history"],0,1)
                mask[:,self.n_original+vid] = 1
        x*=mask
        q=np.unique(lw.query_times)
        raw=torch.as_tensor(w.times[None,:],dtype=torch.float32)
        return dict(x=torch.tensor(x[None]),x_mask=torch.tensor(mask[None]),
                    x_mark=(raw/self.params['history']).unsqueeze(-1),observed_tp_raw=raw,
                    y_mark=torch.tensor((q+w.cutoff)[None,:,None]/self.params['history'],dtype=torch.float32)),q
    def _forward(self, inputs):
        inputs = {key: value.to(self.device) for key,value in inputs.items()}
        if self.model_id in ('ASTGI', 'TFMixer', 'HyperIMTS'):
            b, q = inputs['y_mark'].shape[:2]
            # Placeholder tensors satisfy the upstream return contract; the
            # model computes predictions before accessing these placeholders.
            inputs = {**inputs, 'y': torch.zeros(b,q,self.params['enc_in'],device=self.device),
                      'y_mask': torch.ones(b,q,self.params['enc_in'],device=self.device)}
        if self.model_id == 'TiWeaver':
            b,q=inputs['y_mark'].shape[:2]
            return self.net(inputs['x'],inputs['x_mark'].squeeze(-1),inputs['y_mark'].squeeze(-1),
                            inputs['x_mask'],torch.ones(b,q,self.params['enc_in'],device=self.device))
        if self.model_id == 'KAFNet':
            return self.net.forecasting(inputs['y_mark'].squeeze(-1),inputs['x'],inputs['x_mark'].squeeze(-1),inputs['x_mask'])[0]
        if self.model_id == 'tPatchGNN':
            x=inputs['x'];mask=inputs['x_mask'];times=inputs['x_mark'].squeeze(-1)
            patches=self.params['npatch'];b,length,n=x.shape
            buckets=torch.clamp((times[0]*patches).long(),0,patches-1)
            width=max(1,max(int((buckets==i).sum()) for i in range(patches)))
            px=torch.zeros(b,patches,width,n,device=self.device);pm=torch.zeros_like(px);pt=torch.zeros_like(px)
            for i in range(patches):
                selected=buckets==i; count=int(selected.sum())
                px[:,i,:count]=x[:,selected];pm[:,i,:count]=mask[:,selected]
                pt[:,i,:count]=times[:,selected,None].expand(-1,-1,n)
            return self.net.forecasting(inputs['y_mark'].squeeze(-1),px,pt,pm)[0]
        return self.net(**inputs)['pred']
    def fit(self,train,bundle,grid,var_ids):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        import random
        random.seed(self.seed);np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.n_original = len(bundle.variable_names)
        n_features = self.n_original * (2 if getattr(self._view, "age_feature", False) else 1)
        self.params=dict(enc_in=n_features,hid_dim=16,te_dim=10,npatch=16,
                         patch_size=bundle.spec.history_duration/16,history=bundle.spec.history_duration,
                         dropout=.1,nlayer=1,attn_heads=1)
        if self.model_id == 'ASTGI':
            self.params.update(d_model=64,features='M',astgi_channel_dim=96,astgi_time_dim=96,
                               astgi_prop_layers=2,astgi_k_neighbors=40,astgi_mlp_ratio=3.,astgi_channel_dist_weight=1.)
        self.params.update(ndim=n_features,features='M',task_name='long_term_forecast',
                           seq_len=4096,pred_len=len(grid),seq_len_max_irr=4096,pred_len_max_irr=len(grid))
        self.params.update(self.model_params)
        self.net=self._construct();optimizer=torch.optim.Adam(self.net.parameters(),lr=.01)
        self.net.train()
        initial_weights=[p.detach().clone() for p in self.net.parameters()]
        self.training_diagnostics={"epoch_loss":[],"last_grad_norm":None}
        for epoch in range(self.epochs):
            losses=[]
            for lw in train:
                inputs,q=self._inputs(lw)
                pred=self._forward(inputs)[0]
                idx=np.searchsorted(q,lw.query_times)
                valid=lw.y_scored & np.isfinite(lw.y_true)
                if not valid.any():continue
                values=np.asarray([self._norm.transform(int(v),np.asarray(y)) for v,y in zip(lw.query_var_ids,lw.y_true)],dtype='float32')
                target=torch.tensor(values[valid],device=self.device)
                loss=((pred[idx[valid],lw.query_var_ids[valid]]-target)**2).mean()
                if not torch.isfinite(loss): raise RuntimeError('APN non-finite training loss')
                optimizer.zero_grad();loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                self.training_diagnostics["last_grad_norm"]=float(grad)
                optimizer.step();losses.append(float(loss.detach()))
            self.training_diagnostics["epoch_loss"].append(float(np.mean(losses)) if losses else None)
        self.training_diagnostics["weight_delta_l2"]=float(torch.sqrt(sum(((p.detach()-q)**2).sum() for p,q in zip(self.net.parameters(),initial_weights))))
        self.net.eval()
    def predict(self,lw):
        inputs,q=self._inputs(lw)
        with torch.no_grad(): pred=self._forward(inputs)[0].cpu().numpy()
        return np.asarray([self._norm.inverse(int(v),pred[np.searchsorted(q,t),int(v)]) for t,v in zip(lw.query_times,lw.query_var_ids)])
    def __getstate__(self):
        state=self.__dict__.copy();state['weights']={k:v.cpu() for k,v in self.net.state_dict().items()};state['device']='cpu';state.pop('net');return state
    def __setstate__(self,state):
        weights=state.pop('weights');self.__dict__.update(state);self.net=self._construct();self.net.load_state_dict(weights);self.net.eval()
