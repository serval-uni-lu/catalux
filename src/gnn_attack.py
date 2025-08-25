import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.optimize import minimize

from .account import Transaction, Account
from .dataloader import (
    DataPreprocessor, 
    get_txs_for_ids,
    get_balance_for_id,
    to_account_features,
)

CONSTRAINTS = {
    'value': [1.0, 1e21],
    'gas': 21000,
    'gas_price': [1e9, 1e12]
}

class GNNAttack:
    
    def __init__(
        self,
        model: torch.nn.Module,
        datapreprocessor: DataPreprocessor,
        evading_id: int,
        main_initial_balance: float,
        sybil_initial_balance: float,
        max_balance_prop: float = 0.8,
        remove_exit: bool = False,
    ):
        """Initialize the attack."""
        self.model = model
        self.data = datapreprocessor
        self.controlled_ids = [evading_id]
        self.main_initial_balance = main_initial_balance
        self.sybil_initial_balance = sybil_initial_balance
        self.max_balance_prop = max_balance_prop
        self.remove_exit = remove_exit
        self.device = next(model.parameters()).device
        
    def _create_sybil(self):
        """Create a new Sybil account."""
        sybil_id = self.graph.add_node()
        self.controlled_ids.append(sybil_id)
        self.controlled_accounts[sybil_id] = Account(sybil_id, self.sybil_initial_balance)
        self.node_features = self._get_node_features(self.txs, self.controlled_ids)
    
    def run(
        self,
        num_steps,
        num_optim_steps,
        p_evasion_threshold,
        gas_penalty,
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
        self.txs = get_txs_for_ids(self.data.txs, id)
        self.graph = self.data.graph.clone().to(self.device)

        if self.remove_exit:
            outflow_txs = self.txs.loc[self.txs['from_id'] == id]
            if len(outflow_txs) > 0:
                highest_outflow_tx = outflow_txs.sort_values('value', ascending=False).iloc[0]
                tx_idx = highest_outflow_tx.name
                affected_nodes = [highest_outflow_tx['from_id'], highest_outflow_tx['to_id']]
                self.txs = self.txs.drop(index=tx_idx).reset_index(drop=True)
                self.graph.delete_edge(tx_idx)
                updated_node_features = self._get_node_features(self.txs, affected_nodes)
                self.graph.update_node_features(affected_nodes, updated_node_features)

        initial_balance = get_balance_for_id(self.txs, id)
        if initial_balance < 0:
            initial_balance = self.main_initial_balance
            results['negative_balance'] = True

        self.controlled_accounts = {id: Account(id, initial_balance)}
        self._create_sybil()
        results['sybils_created'] += 1

        # main loop
        for step in range(num_steps):
            # print(f"\nStep {step + 1}/{num_steps}")
            gradients, probs = self._compute_gradients()
            
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
                # print(f'success! early stopping after {step + 1} steps.')
                results['success'] = True
                results['early_stopped'] = True
                results['steps_taken'] = step
                results['final_prob'] = float(probs[0])
                self.results = results
                return results
            
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
        _, final_probs = self._compute_gradients()
        results['final_prob'] = float(final_probs[0])
        results['steps_taken'] = num_steps
        
        max_final_prob = float(final_probs.max())
        results['success'] = max_final_prob < p_evasion_threshold
        
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
    
    def _compute_tx_impact(self, tx, gradients, probs) -> float:
        """Compute the impact of a transaction on predictions."""
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
        edge_features = tx.to_edge_features(device=self.device)
        self.graph.add_edge(tx.from_id, tx.to_id, edge_features)
        
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
    
