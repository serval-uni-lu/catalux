import torch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, List

from .account import Transaction, Account
from .dataloader import (
    get_txs_for_ids,
    get_balance_for_id,
    to_account_features,
)

CONSTRAINTS = {
    'value': [1.0, 1e21],
    'gas': 21000,
    'gas_price': [1e9, 1e12]
}


class TabularAttack:
    """Attack implementation for tabular models."""
    
    def __init__(
        self,
        model,
        model_type: str,  # 'pytorch' or 'lightgbm'
        feature_names: List[str],
        scaler,
        evading_id: int,
        txs: pd.DataFrame,
        main_initial_balance: float,
        sybil_initial_balance: float,
        max_balance_prop: float = 0.8,
        remove_exit: bool = False,
    ):
        """Initialize the attack for tabular models."""
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.scaler = scaler
        self.evading_id = evading_id
        self.controlled_ids = [evading_id]
        self.txs_original = txs.copy()
        self.main_initial_balance = main_initial_balance
        self.sybil_initial_balance = sybil_initial_balance
        self.max_balance_prop = max_balance_prop
        self.remove_exit = remove_exit
        
        if model_type == 'pytorch':
            if hasattr(model, 'parameters'):
                try:
                    self.device = next(model.parameters()).device
                except StopIteration:
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        elif model_type == 'lightgbm':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
    
    def _create_sybil(self):
        """Create a new Sybil account."""
        max_id = max(self.txs_original['from_id'].max(), self.txs_original['to_id'].max())
        sybil_id = max_id + len(self.controlled_ids)
        self.controlled_ids.append(sybil_id)
        self.controlled_accounts[sybil_id] = Account(sybil_id, self.sybil_initial_balance)
        self.node_features = self._get_node_features(self.txs, self.controlled_ids)
    
    def run(
        self,
        num_steps: int,
        num_optim_steps: int,
        p_evasion_threshold: float,
        gas_penalty: float,
    ) -> Dict:
        """Run the attack."""
        self.num_steps = num_steps
        self.num_optim_steps = num_optim_steps
        self.p_evasion_threshold = p_evasion_threshold
        self.gas_penalty = gas_penalty
        results = {
            'success': False,
            'steps_taken': 0,
            'initial_prob': None,
            'final_prob': None,
            'transactions': [],
            'probabilities': [],
            'total_gas_cost': 0.0,
            'total_value_transferred': 0.0,
            'negative_balance': False,
            'sybils_created': len(self.controlled_ids) - 1,
            'early_stopped': False,
        }
        
        # setup
        id = self.controlled_ids[0]
        self.txs = get_txs_for_ids(self.txs_original, id)
        
        if self.remove_exit:
            outflow_txs = self.txs[self.txs['from_id'] == id]
            if len(outflow_txs) > 0:
                highest_outflow_tx = outflow_txs.sort_values('value', ascending=False).iloc[0]
                tx_idx = highest_outflow_tx.name
                self.txs = self.txs.drop(index=tx_idx).reset_index(drop=True)
        
        initial_balance = get_balance_for_id(self.txs, id)
        if initial_balance < 0:
            initial_balance = self.main_initial_balance
            results['negative_balance'] = True

        self.controlled_accounts = {id: Account(id, initial_balance)}
        self._create_sybil()
        results['sybils_created'] += 1

        # main loop
        for step in range(num_steps):
            if self.model_type == 'pytorch':
                gradients, probs = self._compute_gradients()
            else:
                importances, probs = self._compute_feature_importance()
                gradients = importances
            
            if step == 0:
                results['initial_prob'] = float(probs[0])
            
            results['probabilities'].append({
                'step': step + 1,
                'probs': {int(id): float(p) for id, p in zip(self.controlled_ids, probs)}
            })
            
            # find highest risk account
            high_risk_idx = probs.argmax()
            high_risk_id = self.controlled_ids[high_risk_idx]
            
            if probs[high_risk_idx] < p_evasion_threshold:
                results['success'] = True
                results['early_stopped'] = True
                results['steps_taken'] = step
                results['final_prob'] = float(probs[0])
                self.results = results
                return results
            
            # find optimal transaction
            adv_tx = self._find_optimal_tx(
                high_risk_id,
                gradients,
                probs,
            )
            
            if adv_tx is None:
                if step == 0: # early stopping: creating new sybils won't help
                    return results
                self._create_sybil()
                results['sybils_created'] += 1
            else:
                self._apply_tx(adv_tx)
                results['transactions'].append(adv_tx.to_dict())
                results['total_gas_cost'] += adv_tx.gas_cost
                results['total_value_transferred'] += adv_tx.value
        
        # final evaluation
        if self.model_type == 'pytorch':
            _, final_probs = self._compute_gradients()
        else:
            _, final_probs = self._compute_feature_importance()
        
        results['final_prob'] = float(final_probs[0])
        results['steps_taken'] = num_steps
        
        max_final_prob = float(final_probs.max())
        results['success'] = max_final_prob < p_evasion_threshold
        
        self.results = results
        return results
    
    def _compute_gradients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients for PyTorch models (aligned with GNN approach)."""
        features = self.node_features.copy()
        features_scaled = self.scaler.transform(features)
        X = torch.tensor(features_scaled, dtype=torch.float32, device=self.device, requires_grad=True)
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        # compute gradients with respect to input features for each controlled account
        gradients_list = []
        for i in range(len(self.controlled_ids)):
            if X.grad is not None:
                X.grad.zero_()
            
            if callable(self.model):
                logits = self.model(X)
            elif hasattr(self.model, 'forward'):
                logits = self.model.forward(X)
            else:
                raise ValueError(f"Model {type(self.model)} is not callable and has no forward method")
            
            logit_fraud = logits[i, 1]
            logit_fraud.backward(retain_graph=True)
            
            grad = X.grad[i].detach().clone()
            gradients_list.append(grad)
        
        # get final probabilities
        with torch.no_grad():
            if callable(self.model):
                logits = self.model(X)
            elif hasattr(self.model, 'forward'):
                logits = self.model.forward(X)
            else:
                raise ValueError(f"Model {type(self.model)} is not callable and has no forward method")
            probs = torch.softmax(logits, dim=1)[:, 1]
        
        gradients = torch.stack(gradients_list)
        return gradients, probs
    
    def _compute_feature_importance(self) -> Tuple[torch.Tensor, torch.Tensor]: # TODO: optimize
        """Compute feature importance for LightGBM as pseudo-gradients (aligned with GNN approach)."""
        features = self.node_features.copy()
        features_scaled = self.scaler.transform(features)
        
        # get predictions
        probs = self.model.predict_proba(features_scaled)[:, 1]
        
        # get feature importances
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_importance'):
            importances = self.model.model.feature_importance(importance_type='gain')
        elif hasattr(self.model, 'feature_importance'):
            importances = self.model.feature_importance(importance_type='gain')
        else:
            importances = np.ones(len(self.feature_names))
        
        
        # use probability weighted importance as pseudo-gradients
        pseudo_gradients = []
        importances = importances / (importances.max() + 1e-8)
        for i in range(len(self.controlled_ids)):
            prob_weight = max(probs[i], 0.01)
            grad = importances * prob_weight
            pseudo_gradients.append(grad)
        
        gradients = torch.tensor(np.array(pseudo_gradients), dtype=torch.float32, device=self.device)
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=self.device)
        
        return gradients, probs_tensor

    def _find_optimal_tx(
        self,
        high_risk_id,
        gradients,
        probs,
    ):
        """Find optimal transaction using L-BFGS-B optimization."""
        
        best_tx = None
        best_score = float('inf')
        
        pairs = [(high_risk_id, id) for id in self.controlled_ids if id != high_risk_id] + \
                [(id, high_risk_id) for id in self.controlled_ids if id != high_risk_id]
        
        for v_s, v_r in pairs:
            balance = self._balance(v_s)
            
            # gas cost calculation
            min_total_gas_cost = CONSTRAINTS['gas'] * CONSTRAINTS['gas_price'][0]
            min_cost = CONSTRAINTS['value'][0] + min_total_gas_cost
            
            if balance < min_cost:
                continue
            
            # gas price limits
            max_affordable_gas_price = min(
                CONSTRAINTS['gas_price'][1],
                (balance - CONSTRAINTS['value'][0]) / CONSTRAINTS['gas']
            )
            
            if max_affordable_gas_price < CONSTRAINTS['gas_price'][0]:
                continue
            
            def objective(params):
                value, gas_price = params
                
                value = max(CONSTRAINTS['value'][0], min(CONSTRAINTS['value'][1], value))
                gas_price = max(CONSTRAINTS['gas_price'][0], min(max_affordable_gas_price, gas_price))
                
                gas_cost = CONSTRAINTS['gas'] * gas_price
                total_cost = value + gas_cost
                
                # check balance constraint
                if total_cost > balance:
                    return 1e6  # large penalty for infeasible solutions
                
                tx = Transaction(
                    from_id=v_s,
                    to_id=v_r,
                    value=value,
                    gas=CONSTRAINTS['gas'],
                    gas_price=gas_price
                )
                
                # penalizes solutions with high gas prices to encourage efficient transactions
                impact = self._compute_tx_impact(tx, gradients, probs)
                gas_penalty_term = self.gas_penalty * (gas_price / 1e9)  # gas_price in gwei
                
                return impact + gas_penalty_term

            # reserve (1 - max_balance_prop)% for future transactions
            usable_balance = balance * self.max_balance_prop
            
            max_affordable_value = usable_balance - (CONSTRAINTS['gas'] * CONSTRAINTS['gas_price'][0])
            if max_affordable_value < CONSTRAINTS['value'][0]:
                continue
                
            bounds = [
                (CONSTRAINTS['value'][0], min(CONSTRAINTS['value'][1], max_affordable_value)),
                (CONSTRAINTS['gas_price'][0], max_affordable_gas_price)
            ]
            
            # diverse starting points for better optimization
            starting_points = [
                [CONSTRAINTS['value'][0], CONSTRAINTS['gas_price'][0]],         # minimal cost
                [max_affordable_value * 0.1, max_affordable_gas_price * 0.1],   # low cost
                [max_affordable_value * 0.5, max_affordable_gas_price * 0.5],   # medium cost
                [max_affordable_value * 0.8, CONSTRAINTS['gas_price'][0]],      # high value, low gas
            ]
            
            for start_point in starting_points:
                try:
                    # ensure starting point is within bounds
                    start_point[0] = max(bounds[0][0], min(bounds[0][1], start_point[0]))
                    start_point[1] = max(bounds[1][0], min(bounds[1][1], start_point[1]))
                    
                    result = minimize(
                        objective,
                        start_point,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': self.num_optim_steps}
                    )
                    
                    if result.success and result.fun < best_score:
                        best_score = result.fun
                        optimal_value, optimal_gas_price = result.x
                        
                        best_tx = Transaction(
                            from_id=v_s,
                            to_id=v_r,
                            value=optimal_value,
                            gas=CONSTRAINTS['gas'],
                            gas_price=optimal_gas_price
                        )
                        
                except Exception as e:
                    continue
        
        # a beneficial transaction should have negative impact (reducing probability)
        if best_score > 0:
            return None
        
        # if best_tx:
        #     print(f"optimal: {best_tx.from_id} → {best_tx.to_id} (impact: {best_score:.4f})")
        
        return best_tx
    
    def _compute_tx_impact(self, tx: Transaction, gradients: torch.Tensor, probs) -> float:
        """Compute the impact of a transaction on predictions."""
        delta_tensor = self._get_delta_features(tx)
        
        # convert probs to tensor if needed
        if isinstance(probs, np.ndarray):
            probs = torch.tensor(probs, device=self.device)
        
        logit_changes = (delta_tensor * gradients).sum(dim=1)
        prob_weights = probs / (probs.sum() + 1e-8)
        impact = float(prob_weights @ logit_changes)
        return impact
    
    def _get_delta_features(self, tx: Transaction) -> torch.Tensor:
        """Variation in node features caused by the transaction."""
        augmented_txs = pd.concat([self.txs, tx.to_df()])
        features_after = self._get_node_features(augmented_txs, self.controlled_ids)
        features_before_scaled = self.scaler.transform(self.node_features)
        features_after_scaled = self.scaler.transform(features_after)
        delta_scaled = features_after_scaled - features_before_scaled
        
        return torch.tensor(delta_scaled, dtype=torch.float32, device=self.device)
    
    def _apply_tx(self, tx: Transaction):
        """Apply the transaction."""
        self.txs = pd.concat([self.txs, tx.to_df()])
        self.node_features = self._get_node_features(self.txs, self.controlled_ids)
        self.controlled_accounts[tx.from_id].send(tx)
        self.controlled_accounts[tx.to_id].receive(tx)
    
    def _get_node_features(self, txs: pd.DataFrame, ids: List[int]) -> np.ndarray:
        """Extract node features from transactions (aligned with GNN approach)."""
        features_df = to_account_features(txs)
        features_df = features_df[features_df['node_id'].isin(ids)]
        
        feature_values = []
        for id in ids:
            id_features = features_df[features_df['node_id'] == id]
            if len(id_features) > 0:
                feature_values.append(id_features[self.feature_names].values[0])
            else:  # no transactions for this account yet
                feature_values.append(np.zeros(len(self.feature_names)))
        
        return np.array(feature_values)
    
    def _balance(self, id: int) -> float:
        """Get the balance of a controlled account."""
        return self.controlled_accounts[id].balance