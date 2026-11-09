"""One finite domain, canonical defaults and legal neighbors for v3 policies."""
import copy,hashlib,itertools,json
from .schemas import Action,DataView,PipelineSpec
from .models.registry import CATALOG,runnable_ids
from .transforms import TRANSFORM_SPACE
from .search import apply_action

DEFAULTS={'DLinear':{'epochs':200},'PatchTST':{'patch':4,'d_model':32,'n_layers':2},'iTransformer':{'d_model':32,'n_layers':2}}

class ActionSpace:
    def __init__(self,mode='full',f0=None):
        assert mode in ('full','model_only','data_only')
        self.mode=mode;self.f0=f0
        self.models=[f0.model_id] if mode=='data_only' else runnable_ids()
        self.options={m:(dict(CATALOG[m]['options']) if mode!='data_only' else {}) for m in self.models}
        self.transforms={} if mode=='model_only' else TRANSFORM_SPACE
    def descriptor(self):
        return {'version':'finite_v3_1109','mode':self.mode,'models':self.models,'options':self.options,
                'defaults':DEFAULTS,'transforms':self.transforms,
                'fixed_view':self.f0.data_view.to_dict() if self.mode=='model_only' else None,
                'fixed_model_config':self.f0.model_config if self.mode=='data_only' else None,
                'compatibility':'Harness.validate + transforms.compatible; age_supported; canonical defaults'}
    @property
    def hash(self):return hashlib.sha256(json.dumps(self.descriptor(),sort_keys=True).encode()).hexdigest()
    def canonical(self,spec):
        result=copy.deepcopy(spec)
        result.model_config={**DEFAULTS.get(spec.model_id,{}),**spec.model_config}
        return result
    def key(self,spec):
        s=self.canonical(spec)
        return hashlib.sha256(json.dumps([s.model_id,s.model_config,s.data_view.to_dict(),s.seed],sort_keys=True).encode()).hexdigest()
    def permits(self,action):
        if action.op in ('STOP','INSPECT'):return True
        if action.op=='SELECT_MODEL':return self.mode!='data_only' and action.value in self.models
        if action.op=='SET_MODEL_OPTION':return self.mode!='data_only'
        return action.op in ('SELECT_DATA','SET_TRANSFORM') and action.target_field in self.transforms
    def neighbors(self,parent,h,seen):
        actions=[]
        if self.mode!='data_only':
            actions += [Action(op='SELECT_MODEL',value=m) for m in self.models]
            actions += [Action(op='SET_MODEL_OPTION',target_field=k,value=v) for k,vs in self.options[parent.model_id].items() for v in vs]
        actions += [Action(op='SET_TRANSFORM',target_field=k,value=v) for k,vs in self.transforms.items() for v in vs]
        out={}
        for a in actions:
            if not h.checks_ok(h.validate(a,parent)):continue
            p=self.canonical(apply_action(a,parent,len(h.archive)))
            k=self.key(p)
            if k not in seen and k!=self.key(parent):out[k]=(a,p)
        return [(k,*out[k]) for k in sorted(out)]
