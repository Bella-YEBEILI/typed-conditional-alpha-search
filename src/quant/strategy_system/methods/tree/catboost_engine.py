import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from catboost import CatBoostRegressor,CatBoostClassifier,Pool

DEFAULT_ROLLING_PARAMS = {
    "start":"20210101",
    "end":"20260131",
    "lookback_months":60,
    "step_months":1,
    "mode":"tri",
    "model":dict(
        iterations=2000,
        depth=5,
        learning_rate=0.05,
        subsample=0.8,
        rsm=0.5,
        l2_leaf_reg=10.0,
        thread_count=8,
        random_seed=0,
        allow_writing_files=False,
        verbose=False
    ),
    "early_stopping":dict(val_ratio=0.1,rounds=200,verbose=True)
}

class CATBOOST_ROLLINGEngine:
    def __init__(self,
                 factor_dict:dict[str,pd.DataFrame],        # 因子,日期x股票 
                 y_label:pd.DataFrame,                      # 标签,日期x股票
                 label_shift:int,
                 univ:pd.DataFrame,                         # 股票池,日期x股票
                 oth_dict:dict[str,pd.DataFrame]|None=None, # 其他特征,日期x股票
                 mkt_dict:dict[str,pd.Series]|None=None,    # 市场特征,日期    
                 lossweight:pd.DataFrame|None=None,         # 样本损失函数权重                      
                 params:dict[str,dict]=copy.deepcopy(DEFAULT_ROLLING_PARAMS),
                 ):
        self.factor_dict = factor_dict
        self.y_label = y_label
        self.univ = univ.astype(bool)
        self.oth_dict = oth_dict
        self.mkt_dict = mkt_dict
        self.label_shift = int(label_shift)
        self.idx = pd.to_datetime(y_label.index)
        self.cols = y_label.columns
        self.y_aligned = self.y_label.shift(-self.label_shift)
        self.lossweight = lossweight
        self.w_aligned = self.lossweight.shift(-self.label_shift) if self.lossweight is not None else pd.DataFrame(1.0,index=self.y_aligned.index,columns=self.y_aligned.columns,dtype=np.float64)
        self.factor_ids = sorted(list(self.factor_dict.keys()))
        self.oth_features = sorted(list(self.oth_dict.keys())) if self.oth_dict else []
        self.mkt_features = sorted(list(self.mkt_dict.keys())) if self.mkt_dict else []
        p = copy.deepcopy(params) 
        self.start = self.idx[self.idx>=pd.Timestamp(p["start"])][0]
        self.end = self.idx[self.idx<=pd.Timestamp(p["end"])][-1]
        self.lookback_months = p["lookback_months"]
        self.step_months = p["step_months"]
        self.mode = p["mode"]
        self.model_params = p["model"]
        self.early_stopping = p["early_stopping"]

    def _get_train_dates(self,t:pd.Timestamp)->pd.DatetimeIndex:
        start = (t-pd.DateOffset(months=self.lookback_months)).replace(day=1)
        end = t-pd.DateOffset(months=1)+pd.offsets.MonthEnd(0)
        return self.idx[(self.idx>=start)&(self.idx<=end)][:-(self.label_shift-1)] if self.label_shift>1 else self.idx[(self.idx>=start)&(self.idx<=end)]

    def _init_model(self):
        if self.mode=="reg":
            self.model_params.setdefault("loss_function","RMSE")
            self.model_params.setdefault("eval_metric","RMSE")
            model = CatBoostRegressor(**self.model_params)
        else:
            if self.mode=="bi":
                self.model_params.setdefault("loss_function","logLoss")
                self.model_params.setdefault("eval_metric","logLoss")
            elif self.mode=="tri":
                self.model_params.setdefault("loss_function","MultiClass")
                self.model_params.setdefault("eval_metric","MultiClass")
                self.model_params.setdefault("classes_count",3)
            else:
                raise ValueError(f"unknown mode:{self.mode}")
            model = CatBoostClassifier(**self.model_params)
        return model

    def _build_set(self,dates:pd.DatetimeIndex,apply_univ:bool,typ:str)->tuple:
        nf = len(self.factor_ids)+len(self.oth_features)+len(self.mkt_features)
        X_list = []
        y_list = []
        w_list = []
        date_list = []
        code_list = []
        for d in dates:
            y_row = self.y_aligned.loc[d].to_numpy(dtype=np.float32,copy=False)
            w_row = self.w_aligned.loc[d].to_numpy(dtype=np.float32,copy=False)
            if apply_univ:
                u_row = self.univ.loc[d].to_numpy(dtype=np.bool_,copy=False)
                if typ=="train":    
                    ok = u_row&np.isfinite(y_row)&np.isfinite(w_row)
                elif typ=="predict":
                    ok = u_row
            else:
                if typ=="train":
                    ok = np.isfinite(y_row)&np.isfinite(w_row)
                elif typ=="predict":
                    ok = np.ones(self.univ.shape[1],dtype=bool)
            cnt = int(ok.sum())
            idxs = np.where(ok)[0]
            codes = self.cols[idxs]
            date_list.append(np.repeat(d,cnt))
            code_list.append(codes.to_numpy(copy=False))
            y_ok = y_row[ok]
            w_ok = w_row[ok]
            Xd = np.empty((cnt,nf),dtype=np.float32)
            # 因子特征
            for j,fid in enumerate(self.factor_ids):
                Xd[:,j] = self.factor_dict[fid].loc[d].to_numpy(dtype=np.float32,copy=False)[ok]
            # 其他特征
            if self.oth_dict is not None and len(self.oth_features)>0:
                for j2,ft_name in enumerate(self.oth_features):
                    Xd[:,len(self.factor_ids)+j2] = self.oth_dict[ft_name].loc[d].to_numpy(dtype=np.float32,copy=False)[ok]
            # 市场特征广播
            if self.mkt_dict is not None and len(self.mkt_features)>0:
                for j3,mkt_ft_name in enumerate(self.mkt_features):
                    Xd[:,len(self.factor_ids)+len(self.oth_features)+j3] = np.float32(self.mkt_dict[mkt_ft_name].loc[d])
            X_list.append(Xd)
            y_list.append(y_ok.astype(np.float32,copy=False))
            w_list.append(w_ok.astype(np.float32,copy=False))
        X = np.concatenate(X_list,axis=0)
        y = np.concatenate(y_list)
        dates = np.concatenate(date_list)
        codes = np.concatenate(code_list)
        weights = np.concatenate(w_list)
        return X,y,dates,codes,weights

    def _fit(self,X_train:np.ndarray,y_train:np.ndarray,w_train:np.ndarray):
        model = self._init_model()
        if self.early_stopping is None:
            model.fit(X_train,y_train,sample_weight=w_train)
            return model
        # 早停
        vr = float(self.early_stopping.get("val_ratio",0.1))
        rounds = int(self.early_stopping.get("rounds",200))
        verbose = bool(self.early_stopping.get("verbose",True))
        n = X_train.shape[0]
        n_val = int(n*vr)
        X_tr = X_train[:-n_val,:]
        y_tr = y_train[:-n_val]
        w_tr = w_train[:-n_val]
        X_va = X_train[-n_val:,:]
        y_va = y_train[-n_val:]
        w_va = w_train[-n_val:]
        train_pool = Pool(X_tr,y_tr,weight=w_tr)
        val_pool = Pool(X_va,y_va,weight=w_va)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=rounds,
            use_best_model=True,
            verbose=verbose
        )
        return model


    def _predict(self,model,X_predict:np.ndarray)->np.ndarray:
        if self.mode=="reg":
            score = model.predict(X_predict).astype(np.float32,copy=False)
        else:
            proba = model.predict_proba(X_predict).astype(np.float32,copy=False)
            score = (proba[:,-1]-proba[:,0])
        return score

    def _calc_loss(self,model,X:np.ndarray,y:np.ndarray,w:np.ndarray)->float:
        mask = np.isfinite(y)&np.isfinite(w)
        if not mask.any():
            return float("nan")
        Xv = X[mask]
        yv = y[mask]
        wv = w[mask]
        wv = wv/wv.mean()
        if self.mode=="reg":
            y_pred = model.predict(Xv).astype(np.float32,copy=False)
            mse = np.sum(wv*(y_pred-yv)**2)/np.sum(wv)
            loss = np.sqrt(mse)
        else:
            proba = model.predict_proba(Xv).astype(np.float32,copy=False)
            eps = 1e-15
            y_int = yv.astype(np.int8,copy=False)
            if self.mode=="bi":
                p1 = np.clip(proba[:,1],eps,1.0-eps)
                loss = -np.sum(wv*(y_int*np.log(p1)+(1-y_int)*np.log(1.0-p1)))/np.sum(wv)
            else:
                n = y_int.shape[0]
                p = np.clip(proba[np.arange(n),y_int],eps,1.0)
                loss = -np.sum(wv*np.log(p))/np.sum(wv)
        return loss

    def run(self)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        alpha_idx = self.idx[(self.idx>=self.start)&(self.idx<=self.end)]
        feat_cols = self.factor_ids+self.oth_features+self.mkt_features
        score_list = []
        loss_dict = {}
        imp_dict = {}
        idx_months = alpha_idx.to_period('M')
        months = idx_months.unique().sort_values()
        for i in tqdm(range(0,len(months),self.step_months)):
            # 日期处理
            predict_months = months[i:i+self.step_months]
            predict_dates = alpha_idx[idx_months.isin(predict_months)]
            train_dates = self._get_train_dates(predict_dates[0])
            if len(predict_months)==1:
                mkey = predict_months[0].strftime("%Y%m")
            else:
                mkey = f"{predict_months[0].strftime('%Y%m')}-{predict_months[-1].strftime('%Y%m')}"
            # 构造集合
            X_train,y_train,_,_,w_train = self._build_set(train_dates,apply_univ=True,typ="train")
            X_predict,y_predict,dates,codes,w_predict = self._build_set(predict_dates,apply_univ=True,typ="predict")
            # 训练模型
            model = self._fit(X_train,y_train,w_train)
            # 预测
            score = self._predict(model,X_predict)
            mi = pd.MultiIndex.from_arrays([dates,codes],names=["date","code"])
            score = pd.Series(score,index=mi).unstack("code")
            score_list.append(score)
            # 计算损失
            train_loss = self._calc_loss(model,X_train,y_train,w_train)
            predict_loss = self._calc_loss(model,X_predict,y_predict,w_predict)
            loss_dict[mkey] = {"train":train_loss,"predict":predict_loss}
            # 获取特征重要度
            imp = model.get_feature_importance()
            imp_dict[mkey] = {c:imp[j] for j,c in enumerate(feat_cols)}
        # 汇总
        score_df = pd.concat(score_list,axis=0).reindex(index=alpha_idx,columns=self.cols)
        loss_df = pd.DataFrame(loss_dict).T
        fi_df = pd.DataFrame(imp_dict).T
        return score_df,loss_df,fi_df