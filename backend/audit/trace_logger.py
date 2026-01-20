import logging
import json
import os
import time
from datetime import datetime

class TraceLogger:
    """Records execution traces for every agent turn to ensure auditability."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Configure the standard logger
        self.logger = logging.getLogger("trace_logger")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(
            os.path.join(log_dir, f"trace_{datetime.now().strftime('%Y%m%d')}.log")
        )
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_agent_step(self, trace_id: str, agent_name: str, input_data: any, output_data: any, duration: float):
        """Logs a single step in the agentic workflow."""
        entry = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent_name,
            "execution_time_sec": round(duration, 3),
            "input": str(input_data)[:500],  # Truncate for readability
            "output": output_data.model_dump() if hasattr(output_data, 'model_dump') else str(output_data)
        }
        
        self.logger.info(json.dumps(entry))

# Initialize a global trace logger instance
audit_logger = TraceLogger()