"""
Lightweight PyTorch Policy Network for Auto-SRE Agent.

Architecture : MLP(6 → 64 → 7)
Inputs       : normalised [cpu, ram] for each of 3 services (6 floats)
Outputs      : logits over 7 actions

Action index map:
  0 → restart  frontend
  1 → restart  auth_api
  2 → restart  database
  3 → scale_up frontend
  4 → scale_up auth_api
  5 → scale_up database
  6 → no_op
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ACTIONS = [
        ("frontend", "restart"),
        ("auth_api", "restart"),
        ("database", "restart"),
        ("frontend", "scale_up"),
        ("auth_api", "scale_up"),
        ("database", "scale_up"),
        ("frontend", "no_op"),   # service field is ignored for no_op
    ]

    SERVICES_ORDER = ["frontend", "auth_api", "database"]

    class PolicyNetwork(nn.Module):
        """
        2-layer MLP policy for SRE action selection.
        Initialized with domain-aware biases so it is immediately
        useful without any training:
          - Discourages no_op (index 6) when the system is stressed
          - Slightly favours restart actions for high-CPU services
        """

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, 64)
            self.fc2 = nn.Linear(64, 7)
            self._init_weights()

        def _init_weights(self):
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight)

            # Domain-aware output bias
            bias = torch.zeros(7)
            bias[6] = -1.5   # discourage no_op when system is stressed
            bias[0] += 0.3   # slight prior toward restarting stressed services
            bias[1] += 0.5   # auth_api is the most-stressed service in all 3 tasks
            bias[2] += 0.2   # database
            self.fc2.bias = nn.Parameter(bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            return self.fc2(x)


    def metrics_to_tensor(metrics: dict) -> torch.Tensor:
        """
        Convert env metrics dict → normalised [0, 1] tensor of shape (6,).
        Order: [cpu_front, ram_front, cpu_auth, ram_auth, cpu_db, ram_db]
        """
        vals = []
        for svc in SERVICES_ORDER:
            m = metrics.get(svc, {"cpu": 0.0, "ram": 0.0})
            vals.append(min(m["cpu"] / 100.0, 1.0))
            vals.append(min(m["ram"] / 100.0, 1.0))
        return torch.tensor(vals, dtype=torch.float32)


    def get_policy_suggestion(net: PolicyNetwork, metrics: dict):
        """
        Run a forward pass and return the highest-confidence action.

        Returns:
            service    (str)   — e.g. 'auth_api'
            command    (str)   — e.g. 'restart'
            confidence (float) — softmax probability of the chosen action
        """
        with torch.no_grad():
            x      = metrics_to_tensor(metrics).unsqueeze(0)   # shape (1, 6)
            logits = net(x)                                     # shape (1, 7)
            probs  = torch.softmax(logits, dim=-1)[0]           # shape (7,)
            idx    = int(torch.argmax(probs).item())
            conf   = float(probs[idx].item())
        service, command = ACTIONS[idx]
        return service, command, conf

    TORCH_AVAILABLE = True

except ImportError:
    # torch not installed — inference.py will fall back to LLM-only mode
    TORCH_AVAILABLE       = False
    PolicyNetwork         = None
    metrics_to_tensor     = None
    get_policy_suggestion = None
    ACTIONS               = []
