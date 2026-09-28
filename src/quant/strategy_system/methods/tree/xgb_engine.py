import numpy as np
import pandas as pd
import xgboost as xgb
import copy
from itertools import product
from tqdm import tqdm

# =============================================================================
# 调参配置区 —— 用户可直接修改此区块
# =============================================================================
# 网格搜索超参空间:改这里即可。key需与xgboost参数名对齐。
# 仅 XGB_ROLLINGEngine 默认引用本字典(经由 DEFAULT_ROLLING_PARAMS["tune"]["space"]);
# XGB_SINGLEEngine 由调用方自带 params["tune"] 控制,默认不消费此常量。
# 8 维, 组合数 3·3·3·3·2·3·3·2 = 2916; 配 n_iter=80 (随机搜索, ~60 样本足以以 95% 概率命中前 5% 配置)。
# 显式从网格中固定(留给 model_params 处理)的项: gamma=0, tree_method=hist, max_bin=256。
HP_SEARCH_SPACE = {
    "max_depth":         [4, 5, 6],               # +5 接住 fixed 默认; 去 8 (深度 8 在因子噪声下基本过拟)
    "learning_rate":     [0.03, 0.05, 0.1],
    "n_estimators":      [800, 1500, 3000],       # CV 不开早停, 与 lr 上下限配平: lr=0.03 需更长, lr=0.1 需更稳
    "subsample":         [0.7, 0.8, 0.9],         # +0.8 (默认值), 三档行抽样
    "colsample_bytree":  [0.5, 0.7],              # 0.8 接近不抽列, alpha 库冗余高时无区分度
    "min_child_weight":  [50, 200, 500],          # 带权 × 百万样本: 20/50 形同无约束, 整体上移一量级
    "reg_lambda":        [0.1, 1.0, 10.0],        # +0.1 拉宽下界, 弱正则区也覆盖
    "reg_alpha":         [0.0, 1.0],              # 换掉无效的 gamma; L1 在高冗余因子库上促稀疏特征选择
}

# 调参模块配置(只在 tune.enabled=True 时生效)
# gap 说明:
#   训练窗内 CV 的 embargo (val 折前后需排除的训练天数) 默认等于 label_shift-1,
#   与固定参数路径 _get_train_dates 的 [:-(label_shift-1)] 完全一致,保证 train↔val
#   的 gap 与 train↔predict 的 gap 是同一个口径,不引入额外泄露也不做过度隔离。
DEFAULT_TUNE = {
    "enabled":               False,           # 总开关: False=原固定参数; True=滚动+网格搜索
    "k_folds":               3,               # K折(直接填数字即可)
    "space":                 HP_SEARCH_SPACE, # 超参空间(引用上面的字典)
    "n_iter":                80,              # 从笛卡尔积中随机采样的组合数; None/0=全量网格
    "embargo":               None,            # CV内 val折前后 purge 的交易日数; None=label_shift-1(与固定路径一致)
    "refit_early_stopping":  True,            # 用最优参数在全量train上refit时是否启用尾部早停
    "verbose":               True,
    "random_state":          0,
}
# =============================================================================

DEFAULT_ROLLING_PARAMS = {"start":"20210101",
                          "end":"20251231",
                          "lookback_months":60,
                          "step_months":1,
                          "mode":"tri", # reg,bi,tri
                          "model":dict(n_estimators=2000,max_depth=5,learning_rate=0.05,
                               subsample=0.8,colsample_bytree=0.5,reg_lambda=10.0,
                               n_jobs=8,tree_method="hist",min_child_weight=50,gamma=0.0),
                          "early_stopping":dict(val_ratio=0.1,rounds=200,verbose=True),
                          "tune":DEFAULT_TUNE}

class XGB_ROLLINGEngine:
    def __init__(self,
                 factor_dict:dict[str,pd.DataFrame],        # 因子,日期x股票 
                 y_label:pd.DataFrame,                      # 标签,日期x股票
                 label_shift:int,
                 univ:pd.DataFrame,                         # 股票池,日期x股票
                 oth_dict:dict[str,pd.DataFrame]|None=None, # 其他特征,日期x股票
                 mkt_dict:dict[str,pd.Series]|None=None,    # 市场特征,日期    
                 lossweight:pd.DataFrame|None=None,         # 样本损失函数权重                      
                 params:dict[str,dict]=DEFAULT_ROLLING_PARAMS):
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
        self.tune = p.get("tune")
        self.tune_history = {}  # 存放每个滚动窗口选出的最优参数,便于事后检查
        self._build_caches()

    def _build_caches(self):
        """预构建 ndarray 缓存,避免 _build_set 里逐日 .loc[d] 的热点。"""
        self._full_idx = pd.DatetimeIndex(self.idx)
        self._cols_arr = self.cols.to_numpy()
        self._factor_arrs = {fid: self.factor_dict[fid].reindex(index=self.idx,columns=self.cols).to_numpy(dtype=np.float32,copy=False) for fid in self.factor_ids}
        self._oth_arrs = {ft: self.oth_dict[ft].reindex(index=self.idx,columns=self.cols).to_numpy(dtype=np.float32,copy=False) for ft in self.oth_features} if self.oth_dict else {}
        self._mkt_arrs = {mk: self.mkt_dict[mk].reindex(self.idx).to_numpy(dtype=np.float32,copy=False) for mk in self.mkt_features} if self.mkt_dict else {}
        self._y_arr = self.y_aligned.reindex(index=self.idx,columns=self.cols).to_numpy(dtype=np.float32,copy=False)
        self._w_arr = self.w_aligned.reindex(index=self.idx,columns=self.cols).to_numpy(dtype=np.float32,copy=False)
        self._univ_arr = self.univ.reindex(index=self.idx,columns=self.cols).fillna(False).astype(bool).to_numpy(copy=False)

    def _get_train_dates(self,t:pd.Timestamp)->pd.DatetimeIndex:
        start = (t-pd.DateOffset(months=self.lookback_months)).replace(day=1)
        end = t-pd.DateOffset(months=1)+pd.offsets.MonthEnd(0)
        return self.idx[(self.idx>=start)&(self.idx<=end)][:-(self.label_shift-1)] if self.label_shift>1 else self.idx[(self.idx>=start)&(self.idx<=end)]

    def _init_model(self,override:dict|None=None):
        p = copy.deepcopy(self.model_params)
        if override:
            p.update(override)
        if self.mode=="reg":
            p.setdefault("objective","reg:squarederror")
            p.setdefault("eval_metric","rmse")
            p.setdefault("random_state",0)
            model = xgb.XGBRegressor(**p)
        else:
            if self.mode=="bi":
                p.setdefault("objective","binary:logistic")
                p.setdefault("eval_metric","logloss")
            elif self.mode=="tri":
                p.setdefault("objective","multi:softprob")
                p.setdefault("eval_metric","mlogloss")
                p.setdefault("num_class",3)
            else:
                raise ValueError(f"unknown mode:{self.mode}")
            p.setdefault("random_state",0)
            model = xgb.XGBClassifier(**p)
        return model

    def _build_set(self,dates:pd.DatetimeIndex,apply_univ:bool,typ:str)->tuple:
        date_pos = self._full_idx.get_indexer(pd.DatetimeIndex(dates))
        if (date_pos<0).any():
            raise ValueError("dates 含有不在 self.idx 中的日期")
        y_block = self._y_arr[date_pos]
        w_block = self._w_arr[date_pos]
        if apply_univ:
            u_block = self._univ_arr[date_pos]
        else:
            u_block = np.ones(y_block.shape,dtype=bool)
        if typ=="train":
            ok_block = u_block&np.isfinite(y_block)&np.isfinite(w_block)
        elif typ=="predict":
            ok_block = u_block
        else:
            raise ValueError(f"unknown typ:{typ}")
        row_off,col_off = np.where(ok_block)
        N = row_off.size
        nf = len(self.factor_ids)+len(self.oth_features)+len(self.mkt_features)
        X = np.empty((N,nf),dtype=np.float32)
        date_pos_rows = date_pos[row_off]
        j = 0
        for fid in self.factor_ids:
            X[:,j] = self._factor_arrs[fid][date_pos_rows,col_off]
            j += 1
        for ft_name in self.oth_features:
            X[:,j] = self._oth_arrs[ft_name][date_pos_rows,col_off]
            j += 1
        for mkt_ft_name in self.mkt_features:
            X[:,j] = self._mkt_arrs[mkt_ft_name][date_pos_rows]
            j += 1
        y = y_block[row_off,col_off].astype(np.float32,copy=False)
        weights = w_block[row_off,col_off].astype(np.float32,copy=False)
        dates_out = self._full_idx[date_pos_rows].to_numpy()
        codes_out = self._cols_arr[col_off]
        return X,y,dates_out,codes_out,weights

    def _purged_kfold_indices(self,dates_train:np.ndarray,k:int,embargo:int)->list:
        """基于日期的 purged K-fold 切分。
        - 按train set中的唯一交易日顺序切成k段连续折,依次作为验证折
        - 对训练侧样本 purge: 排除其在 self.idx 中位置 <= embargo 接近验证折边界的样本
        - 使用 self.idx 位置而非日历天,避免周末/停牌等偏差
        """
        full_idx = pd.DatetimeIndex(self.idx)
        train_pos = full_idx.get_indexer(pd.DatetimeIndex(dates_train))
        if (train_pos<0).any():
            raise ValueError("dates_train 存在不在 self.idx 中的日期")
        unique_pos = np.unique(train_pos)
        n = len(unique_pos)
        if n < k:
            raise ValueError(f"训练唯一日期数({n})少于折数({k})")
        fold_sizes = np.full(k,n//k,dtype=int)
        fold_sizes[:n%k] += 1
        starts = np.cumsum(np.insert(fold_sizes,0,0))
        splits = []
        for i in range(k):
            val_pos_set = unique_pos[starts[i]:starts[i+1]]
            val_min = int(val_pos_set.min())
            val_max = int(val_pos_set.max())
            val_mask = np.isin(train_pos,val_pos_set)
            # 训练侧: 同时剔除验证折及其前后 embargo 个交易日
            purge_mask = (train_pos>=val_min-embargo)&(train_pos<=val_max+embargo)
            tr_mask = (~val_mask)&(~purge_mask)
            if tr_mask.sum()==0 or val_mask.sum()==0:
                continue
            splits.append((np.where(tr_mask)[0],np.where(val_mask)[0]))
        return splits

    def _grid_search(self,X:np.ndarray,y:np.ndarray,w:np.ndarray,dates_train:np.ndarray)->tuple[dict,float]:
        cfg = self.tune
        space = cfg["space"]
        k = int(cfg.get("k_folds",5))
        embargo = cfg.get("embargo")
        if embargo is None:
            # 与 _get_train_dates 的 [:-(label_shift-1)] 保持一致的 gap 口径
            embargo = max(0,self.label_shift-1)
        embargo = int(embargo)
        n_iter = cfg.get("n_iter")
        verbose = bool(cfg.get("verbose",True))
        rs = int(cfg.get("random_state",0))
        keys = list(space.keys())
        grids = [list(space[kk]) for kk in keys]
        all_combos = list(product(*grids))
        if n_iter and n_iter>0 and n_iter<len(all_combos):
            rng = np.random.default_rng(rs)
            pick = rng.choice(len(all_combos),size=n_iter,replace=False)
            all_combos = [all_combos[int(i)] for i in pick]
        splits = self._purged_kfold_indices(dates_train,k,embargo)
        if len(splits)==0:
            raise RuntimeError("purged K-fold 未产生任何有效切分,请检查 k_folds/embargo 设置")
        best_loss = np.inf
        best_params = None
        for combo in all_combos:
            override = dict(zip(keys,combo))
            fold_losses = []
            for tr_idx,va_idx in splits:
                model = self._init_model(override=override)
                model.fit(X[tr_idx],y[tr_idx],sample_weight=w[tr_idx])
                loss = self._calc_loss(model,X[va_idx],y[va_idx],w[va_idx])
                fold_losses.append(loss)
            mean_loss = float(np.nanmean(fold_losses))
            if verbose:
                print(f"[tune] {override} cv_loss={mean_loss:.5f}")
            if mean_loss<best_loss:
                best_loss = mean_loss
                best_params = override
        return best_params,best_loss

    def _fit(self,X_train:np.ndarray,y_train:np.ndarray,w_train:np.ndarray,dates_train:np.ndarray|None=None):
        override = None
        tune_on = bool(getattr(self,"tune",None) and self.tune.get("enabled"))
        if tune_on:
            if dates_train is None:
                raise ValueError("启用 tune 时必须传入 dates_train")
            best_params,best_loss = self._grid_search(X_train,y_train,w_train,dates_train)
            override = best_params
            self._last_best_params = best_params
            self._last_best_loss = best_loss
            if self.tune.get("verbose",True):
                print(f"[tune] BEST {best_params} cv_loss={best_loss:.5f}")
        model = self._init_model(override=override)
        # 是否启用尾部早停 refit
        es_on = self.early_stopping is not None
        if tune_on and not self.tune.get("refit_early_stopping",True):
            es_on = False
        if not es_on:
            model.fit(X_train,y_train,sample_weight=w_train)
            return model
        # 早停:按唯一交易日切分(同一天的样本不会被拆分),并在 train 与 val 之间
        # 保留 embargo 个交易日,与 _get_train_dates / _purged_kfold_indices 口径一致
        # 优先用 val_days(末尾 N 个交易日作 val),否则退回 val_ratio
        val_days = self.early_stopping.get("val_days")
        vr = float(self.early_stopping.get("val_ratio",0.1))
        rounds = int(self.early_stopping.get("rounds",200))
        verbose = bool(self.early_stopping.get("verbose",True))
        embargo = max(0,self.label_shift-1)
        if dates_train is None or len(dates_train)==0:
            raise ValueError("启用早停时必须传入 dates_train,用于按日期切分 val")
        dt_train = pd.DatetimeIndex(dates_train)
        unique_dates = dt_train.unique().sort_values()
        if val_days is not None and int(val_days)>0:
            n_val_dates = min(int(val_days),len(unique_dates)-1)
        else:
            n_val_dates = max(1,int(len(unique_dates)*vr))
        n_val_dates = max(1,n_val_dates)
        val_start_date = unique_dates[-n_val_dates]
        val_mask = np.asarray(dt_train>=val_start_date)
        val_start_pos = self._full_idx.get_loc(val_start_date)
        purge_pos = max(0,val_start_pos-embargo)
        purge_cutoff_date = self._full_idx[purge_pos]
        tr_mask = np.asarray(dt_train<purge_cutoff_date)
        if tr_mask.sum()==0 or val_mask.sum()==0:
            raise RuntimeError(f"早停切分失败: train={int(tr_mask.sum())} val={int(val_mask.sum())},检查 val_days/val_ratio/embargo 设置")
        X_tr,y_tr,w_tr = X_train[tr_mask],y_train[tr_mask],w_train[tr_mask]
        X_va,y_va,w_va = X_train[val_mask],y_train[val_mask],w_train[val_mask]
        model.set_params(early_stopping_rounds=rounds)
        model.fit(X_tr,y_tr,sample_weight=w_tr,eval_set=[(X_va,y_va)],sample_weight_eval_set=[w_va],verbose=verbose)
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
            X_train,y_train,dates_train,_,w_train = self._build_set(train_dates,apply_univ=True,typ="train")
            X_predict,y_predict,dates,codes,w_predict = self._build_set(predict_dates,apply_univ=True,typ="predict")
            # 训练模型(可选网格搜索调参)
            model = self._fit(X_train,y_train,w_train,dates_train=dates_train)
            if getattr(self,"_last_best_params",None) is not None and self.tune and self.tune.get("enabled"):
                self.tune_history[mkey] = {"best_params":dict(self._last_best_params),"cv_loss":float(self._last_best_loss)}
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
            imp = model.feature_importances_
            imp_dict[mkey] = {c:imp[j] for j,c in enumerate(feat_cols)}
        # 汇总
        score_df = pd.concat(score_list,axis=0).reindex(index=alpha_idx,columns=self.cols)
        loss_df = pd.DataFrame(loss_dict).T
        fi_df = pd.DataFrame(imp_dict).T
        return score_df,loss_df,fi_df

class XGB_SINGLEEngine(XGB_ROLLINGEngine):
    def __init__(self,
                 factor_dict:dict[str,pd.DataFrame],
                 y_label:pd.DataFrame,
                 label_shift:int,
                 univ:pd.DataFrame,
                 oth_dict:dict[str,pd.DataFrame]|None=None,
                 mkt_dict:dict[str,pd.Series]|None=None,
                 lossweight:pd.DataFrame|None=None,
                 params:dict|None=None):
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
        if params is None:
            raise ValueError("XGB_SINGLEEngine requires params, including cur_dt/lookback_months/mode/model/early_stopping")
        p = copy.deepcopy(params)
        self.cur_dt = pd.to_datetime(p["cur_dt"])
        self.lookback_months = p["lookback_months"]
        self.mode = p["mode"]
        self.model_params = p["model"]
        self.early_stopping = p["early_stopping"]
        self.tune = p.get("tune")
        self.tune_history = {}
        self._build_caches()

    def run(self):
        cur_m = self.cur_dt.to_period("M")
        mask = (self.idx.to_period("M")==cur_m)&(self.idx<=self.cur_dt)
        predict_dates = self.idx[mask]
        train_dates = self._get_train_dates(self.cur_dt)
        feat_cols = self.factor_ids+self.oth_features+self.mkt_features
        mkey = cur_m.strftime("%Y%m")
        X_train,y_train,dates_train,_,w_train = self._build_set(train_dates,apply_univ=True,typ="train")
        X_predict,y_predict,dates,codes,w_predict = self._build_set(predict_dates,apply_univ=True,typ="predict")
        # train
        model = self._fit(X_train,y_train,w_train,dates_train=dates_train)
        if getattr(self,"_last_best_params",None) is not None and self.tune and self.tune.get("enabled"):
            self.tune_history[mkey] = {"best_params":dict(self._last_best_params),"cv_loss":float(self._last_best_loss)}
        # predict
        score = self._predict(model,X_predict)
        mi = pd.MultiIndex.from_arrays([dates,codes],names=["date","code"])
        score = pd.Series(score,index=mi).unstack("code").reindex(index=predict_dates,columns=self.cols)
        # loss
        loss_dict = {}
        train_loss = self._calc_loss(model,X_train,y_train,w_train)
        predict_loss = self._calc_loss(model,X_predict,y_predict,w_predict)
        loss_dict[mkey] = {"train":train_loss,"predict":predict_loss}
        loss = pd.DataFrame(loss_dict).T
        # f_imp
        imp_dict = {}
        imp = model.feature_importances_
        imp_dict[mkey] = {c:imp[j] for j,c in enumerate(feat_cols)}
        fi = pd.DataFrame(imp_dict).T
        return score,loss,fi
