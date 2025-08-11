import torch
import numpy as np
import pandas as pd
from typing import Tuple
from copy import deepcopy
from scipy.optimize import minimize

from .utils import md_print
from .account import Transaction, Account
from .dataloader import (
    DataPreprocessor, 
    get_txs_for_ids, 
    get_balance_for_id,
    to_account_features
)

CONSTRAINTS = {
    'value': [1.0, 1e+18],  # 1 wei to 1 ETH
    'gas': 21000,
    'gas_price': [1e+8, 1e+11]  # 1-100 gwei
}

class Attack:
    
    def __init__(
        self,
        model: torch.nn.Module,
        datapreprocessor: DataPreprocessor,
        evading_id: int
    ):
        self.model = model
        self.data = datapreprocessor
        self.controlled_ids = [evading_id]
        self.device = next(model.parameters()).device
        self._setup()
        # print(f'Initial balance: {self._balance(evading_id):.2e}')

    def _setup(self):
        id = self.controlled_ids[0]
        self.graph = deepcopy(self.data.graph).to(self.device)
        
        self.txs = get_txs_for_ids(self.data.txs, id)
        self.node_features = self._get_node_features(
            self.txs, self.controlled_ids)

        initial_balance = get_balance_for_id(self.data.txs, id)
        initial_balance = initial_balance if initial_balance > 0 else 1e+17

        self.controlled_accounts = {id: Account(
            id, initial_balance)}
        self._create_sybil()

    def _create_sybil(self):
        sybil_id = self.graph.add_node()
        self.controlled_ids.append(sybil_id)
        self.controlled_accounts[sybil_id] = Account(sybil_id)
        self.node_features = self._get_node_features(self.txs, self.controlled_ids)
    
    def run(
        self,
        num_steps,
        num_optim_steps,
        p_evasion_threshold,
        gas_penalty,
    ):
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
            'sybils_created': len(self.controlled_ids) - 1,
            'early_stopped': False,
        }
        
        for step in range(num_steps):
            # print(f"\nStep {step + 1}/{num_steps}")
            gradients, probs = self._compute_gradients()
            # self.display_probs(probs)
            
            if step == 0:
                results['initial_prob'] = float(probs[0])
            
            results['probabilities'].append({
                'step': step + 1,
                'probs': {id: float(p) for id, p in zip(self.controlled_ids, probs)}
            })
            
            # find highest risk account
            high_risk_idx = probs.argmax()
            high_risk_id = self.controlled_ids[high_risk_idx]
            
            if probs[high_risk_idx] < p_evasion_threshold:
                # print(f'Success! Early stopping after {step + 1} steps.')
                results['success'] = True
                results['early_stopped'] = True
                results['steps_taken'] = step + 1
                results['final_prob'] = float(probs[0])
                self.results = results
                return results
            
            adv_tx = self._find_optimal_tx(
                high_risk_id,
                gradients,
                probs,
            )

            if adv_tx is None:
                # print("Warning: No valid transaction found. Creating a new sybil.")
                self._create_sybil()
                results['sybils_created'] += 1
            else:
                self._apply_tx(adv_tx)
                results['transactions'].append(adv_tx)
                results['total_gas_cost'] += adv_tx.gas_cost
                results['total_value_transferred'] += adv_tx.value
                
        # final evaluation
        _, final_probs = self._compute_gradients()
        results['final_prob'] = float(final_probs[0])
        results['steps_taken'] = num_steps
        
        # check if any account is still above threshold
        max_final_prob = float(final_probs.max())
        if max_final_prob > p_evasion_threshold:
            results['success'] = False
        
        self.results = results
        return results

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
            
            # quick feasibility check
            min_cost = CONSTRAINTS['value'][0] + CONSTRAINTS['gas'] * CONSTRAINTS['gas_price'][0]
            if balance < min_cost:
                continue
            
            max_gas_price = min(
                CONSTRAINTS['gas_price'][1],
                balance / CONSTRAINTS['gas']
            )
            
            def objective(params):
                value, gas_price = params
                
                value = max(CONSTRAINTS['value'][0], min(CONSTRAINTS['value'][1], value))
                gas_price = max(CONSTRAINTS['gas_price'][0], min(max_gas_price, gas_price))
                
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
                impact = self._compute_tx_impact(tx, gradients, probs)

                return impact

            # reserve 40% for future transactions
            usable_balance = balance * 0.6
            max_affordable_value = usable_balance - CONSTRAINTS['gas'] * CONSTRAINTS['gas_price'][0]
            
            # ensure we have enough balance for at least minimum transaction
            if max_affordable_value < CONSTRAINTS['value'][0]:
                continue
                
            bounds = [
                (CONSTRAINTS['value'][0], min(CONSTRAINTS['value'][1], max_affordable_value)),
                (CONSTRAINTS['gas_price'][0], max_gas_price)
            ]
            
            # multiple starting points to avoid local minima
            starting_points = [
                [CONSTRAINTS['value'][0], CONSTRAINTS['gas_price'][0]], # Minimal cost
                [max_affordable_value * 0.1, max_gas_price * 0.1],      # Low cost
                [max_affordable_value * 0.5, max_gas_price * 0.5],      # Medium cost
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
        
        # could not find a beneficial solution
        if best_score > 0:
            return None
        
        # if best_tx:
        #     print(f"Optimal: {best_tx.from_id} → {best_tx.to_id} (Impact: {best_score:.4f})")
        
        return best_tx
    
    def _compute_tx_impact(self, tx, gradients, probs) -> float:
        # gradient_norms = torch.norm(gradients, dim=1)
        # gradient_weights = gradient_norms / (gradient_norms.sum() + 1e-8)
        delta_tensor = self._get_delta_features(tx)
        logit_changes = (delta_tensor * gradients).sum(dim=1)
        prob_weights = probs / (probs.sum() + 1e-8)        
        impact = float(prob_weights @ logit_changes)
        return impact

    def _get_delta_features(self, tx: Transaction) -> torch.Tensor:
        """Variation in node features caused by the transaction."""
        augmented_txs = pd.concat([self.txs, tx.to_df()])
        tensor_after = self._get_node_features(augmented_txs, self.controlled_ids)
        return tensor_after - self.node_features

    def _apply_tx(self, tx: Transaction):
        """Apply the transaction to the graph and update node features."""
        self.txs = pd.concat([self.txs, tx.to_df()])
        self.node_features = self._get_node_features(self.txs, self.controlled_ids)
        self.graph.update_node_features(self.controlled_ids, self.node_features)
        self.graph.add_edge(tx.from_id, tx.to_id, tx.to_edge_features())
        self.controlled_accounts[tx.from_id].send(tx)
        self.controlled_accounts[tx.to_id].receive(tx)
        
    def _get_node_features(self, txs: pd.DataFrame, ids) -> torch.Tensor:
        """Extract and normalize node features from transactions."""
        features_df = to_account_features(txs)
        features_df = features_df[features_df['node_id'].isin(ids)]
        feature_values = features_df[self.data.feature_names].values
        num_missing = len(self.controlled_ids) - feature_values.shape[0]
        if num_missing > 0: # no transactions for sybil yet
            feature_values = np.pad(feature_values, ((0, num_missing), (0, 0)), mode='constant')
        normalized = self.data.normalize_node_features(feature_values)
        return torch.tensor(normalized, dtype=torch.float32, device=self.device)

    def _compute_gradients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients and probabilities for specified nodes."""
        result = self.graph.compute_gradients(
            model=self.model,
            node_ids=self.controlled_ids)
        probs = result['probabilities'][:, 1]
        return result['gradients'], probs
    
    def _balance(self, id: int) -> float:
        """Get the balance of a controlled account."""
        return self.controlled_accounts[id].balance

    def display_tensor(self, tensor: torch.Tensor):
        tensor_df = pd.DataFrame(
            tensor.cpu().numpy(),
            columns=self.data.feature_names,
            index=self.controlled_ids)
        md_print(tensor_df, index=True)
        
    def display_probs(self, probs: list):
        probs_df = pd.DataFrame(
            probs.tolist(), columns=['Probability'], index=self.controlled_ids
        )
        md_print(probs_df, index=True)
